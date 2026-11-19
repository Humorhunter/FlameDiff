"""Conditional diffusion of zero-mean spatial residuals for 4x super-resolution.

The full-field base (typically a frozen U-Net reconstruction) is treated as
fixed conditioning. The diffusion model learns only its high-frequency
residual. The base is first aligned to the low-resolution observations, and
the final residual is projected to zero mean in every 4x4 block, so adding it
back cannot change those observations.
"""

from math import isfinite, log

import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.nn import functional as F


FIELD_HEIGHT = 192
FIELD_WIDTH = 256
OBS_HEIGHT = 48
OBS_WIDTH = 64
DEFAULT_SCALE = 4
DEFAULT_RESIDUAL_SCALE = 128.0


def residual_diffusion_scheduler(train_steps=1000):
    """Build the epsilon DDPM schedule used to train and sample residuals.

    Clipping is disabled because the diffusion target is the 128x-scaled
    residual, not a normalized temperature field. Linspace spacing ensures a
    pure Gaussian sample starts at the highest training timestep, including
    when using a short inference schedule.
    """
    if not isinstance(train_steps, int) or isinstance(train_steps, bool) or train_steps < 2:
        raise ValueError("train_steps must be an integer of at least two")
    return DDPMScheduler(
        num_train_timesteps=train_steps,
        prediction_type="epsilon",
        clip_sample=False,
        timestep_spacing="linspace",
    )


def _validate_field(field, name):
    if not isinstance(field, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if field.ndim != 4 or tuple(field.shape[1:]) != (1, FIELD_HEIGHT, FIELD_WIDTH):
        raise ValueError(f"{name} must have shape [B, 1, {FIELD_HEIGHT}, {FIELD_WIDTH}]")
    if not torch.is_floating_point(field):
        raise ValueError(f"{name} must be floating point")


def _validate_observation(observation, batch_size, name="observation"):
    if not isinstance(observation, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if observation.ndim != 4 or tuple(observation.shape[1:]) != (1, OBS_HEIGHT, OBS_WIDTH):
        raise ValueError(f"{name} must have shape [B, 1, {OBS_HEIGHT}, {OBS_WIDTH}]")
    if observation.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match the full-field inputs")
    if not torch.is_floating_point(observation):
        raise ValueError(f"{name} must be floating point")


def _validate_scale(scale):
    if not isinstance(scale, int) or isinstance(scale, bool) or scale != DEFAULT_SCALE:
        raise ValueError("Only scale=4 is supported by this 192x256 spatial model")


def _block_average(field, scale):
    return F.avg_pool2d(field, kernel_size=scale, stride=scale)


def _repeat_blocks(field, scale):
    return field.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)


def _zero_packed_channel_mean(packed):
    """Project pixel-unshuffled blocks onto the zero-mean residual subspace."""
    return packed - packed.mean(dim=1, keepdim=True)


def zero_block_mean(field, scale=DEFAULT_SCALE):
    """Remove each non-overlapping ``scale``-square block's spatial mean.

    ``field`` is BCHW with spatial dimensions divisible by ``scale``. This
    projection is used for training targets and sampled residuals; it leaves
    each block-average measurement unchanged when the residual is added to an
    aligned base.
    """
    _validate_scale(scale)
    if not isinstance(field, torch.Tensor) or field.ndim != 4:
        raise ValueError("field must be a BCHW tensor")
    if field.shape[-2] % scale or field.shape[-1] % scale:
        raise ValueError("field height and width must be divisible by scale")
    if not torch.is_floating_point(field):
        raise ValueError("field must be floating point")
    return field - _repeat_blocks(_block_average(field, scale), scale)


def align_base(base, obs, scale=DEFAULT_SCALE):
    """Shift each base block by a constant so its average equals ``obs``.

    This deterministic affine correction is applied to the frozen base before
    residual diffusion. The residual model does not update the base, and its
    zero-block-mean output preserves the corrected block averages.
    """
    _validate_scale(scale)
    _validate_field(base, "base")
    _validate_observation(obs, base.shape[0], "obs")
    if base.device != obs.device or base.dtype != obs.dtype:
        raise ValueError("base and obs must share device and dtype")
    if not torch.isfinite(base).all() or not torch.isfinite(obs).all():
        raise ValueError("base and obs must be finite")
    correction = obs - _block_average(base, scale)
    return base + _repeat_blocks(correction, scale)


def _assert_aligned(base, observation, scale, name="base"):
    discrepancy = (_block_average(base, scale) - observation).abs().max()
    if discrepancy.item() > 5e-4:
        raise ValueError(
            f"{name} must already match observation block averages; "
            "call align_base(base, observation) first"
        )


def _sinusoidal_embedding(timesteps, dimension, dtype):
    half = dimension // 2
    if half < 1:
        raise ValueError("timestep embedding dimension must be at least 2")
    frequencies = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (-log(10000.0) / max(half - 1, 1))
    )
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(dtype=dtype)


class _TimeResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, x, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (self.skip(x) + hidden) * (2**-0.5)


class ResidualDenoiser(nn.Module):
    """Compact conditional U-Net that predicts epsilon in packed residual space.

    Args:
        width: Base channel count. Must be a positive multiple of eight.
        residual_scale: Fixed gain applied to normalized residuals before the
            diffusion process. 128 makes typical temperature-field residuals
            order one while the base and observations remain normalized.

    ``forward(noisy_packed, base, observation, timesteps)`` expects the noisy
    residual already pixel-unshuffled to 16 channels. The fixed, aligned base
    is pixel-unshuffled internally and concatenated with the LR observation.
    The U-Net base is conditioning only; only this denoiser is optimized.
    """

    def __init__(self, width=64, residual_scale=DEFAULT_RESIDUAL_SCALE):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError("width must be an integer multiple of 8 and at least 8")
        try:
            residual_scale = float(residual_scale)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("residual_scale must be finite and positive") from None
        if not isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        self.residual_scale = residual_scale

        time_channels = width * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(width, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(33, width, 3, padding=1)
        self.enc1 = _TimeResidualBlock(width, width, time_channels)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.enc2 = _TimeResidualBlock(width * 2, width * 2, time_channels)
        self.down2 = nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1)
        self.middle1 = _TimeResidualBlock(width * 3, width * 3, time_channels)
        self.middle2 = _TimeResidualBlock(width * 3, width * 3, time_channels)
        self.dec2 = _TimeResidualBlock(width * 5, width * 2, time_channels)
        self.dec1 = _TimeResidualBlock(width * 3, width, time_channels)
        self.output_norm = nn.GroupNorm(8, width)
        self.output = nn.Conv2d(width, 16, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_packed, base, observation, timesteps):
        _validate_field(base, "base")
        _validate_observation(observation, base.shape[0])
        if noisy_packed.ndim != 4 or tuple(noisy_packed.shape[1:]) != (16, OBS_HEIGHT, OBS_WIDTH):
            raise ValueError(f"noisy_packed must have shape [B, 16, {OBS_HEIGHT}, {OBS_WIDTH}]")
        if noisy_packed.shape[0] != base.shape[0]:
            raise ValueError("noisy_packed batch size must match base")
        if timesteps.ndim not in (1, 2) or timesteps.shape[0] != base.shape[0]:
            raise ValueError("timesteps must have one entry per batch item")
        if timesteps.ndim == 2 and timesteps.numel() != base.shape[0]:
            raise ValueError("timesteps must have one entry per batch item")
        if not (noisy_packed.device == base.device == observation.device):
            raise ValueError("noisy_packed, base, and observation must share a device")
        if not (noisy_packed.dtype == base.dtype == observation.dtype):
            raise ValueError("noisy_packed, base, and observation must share a dtype")

        time = timesteps.reshape(-1).to(device=noisy_packed.device)
        time_embedding = self.time_mlp(
            _sinusoidal_embedding(time, self.time_mlp[0].in_features, noisy_packed.dtype)
        )
        base_packed = F.pixel_unshuffle(base, DEFAULT_SCALE)
        hidden = self.input(torch.cat((noisy_packed, base_packed, observation), dim=1))
        skip1 = self.enc1(hidden, time_embedding)
        skip2 = self.enc2(self.down1(skip1), time_embedding)
        middle = self.middle1(self.down2(skip2), time_embedding)
        middle = self.middle2(middle, time_embedding)
        middle = F.interpolate(middle, size=skip2.shape[-2:], mode="nearest")
        middle = self.dec2(torch.cat((middle, skip2), dim=1), time_embedding)
        middle = F.interpolate(middle, size=skip1.shape[-2:], mode="nearest")
        middle = self.dec1(torch.cat((middle, skip1), dim=1), time_embedding)
        return self.output(F.silu(self.output_norm(middle)))


def _validate_model_batch(model, base, observation):
    _validate_field(base, "base")
    _validate_observation(observation, base.shape[0])
    if base.device != observation.device or base.dtype != observation.dtype:
        raise ValueError("base and observation must share device and dtype")
    if not torch.isfinite(base).all() or not torch.isfinite(observation).all():
        raise ValueError("base and observation must be finite")
    residual_scale = getattr(model, "residual_scale", None)
    if residual_scale is None or not isfinite(float(residual_scale)) or float(residual_scale) <= 0:
        raise ValueError("model must define a finite positive residual_scale")
    return float(residual_scale)


def residual_loss(model, base, observation, target, scheduler, generator=None):
    """Train epsilon prediction on scaled, zero-block-mean ``target - base``.

    The base and observation are detached conditioning tensors, so gradients
    update the residual denoiser only. ``base`` must already be aligned to the
    measured LR field (use :func:`align_base` first).
    """
    residual_scale = _validate_model_batch(model, base, observation)
    _validate_field(target, "target")
    if target.shape != base.shape:
        raise ValueError("target and base must have the same shape")
    if target.device != base.device or target.dtype != base.dtype:
        raise ValueError("target and base must share device and dtype")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("residual diffusion requires epsilon prediction")
    _assert_aligned(base.detach(), observation.detach(), DEFAULT_SCALE)

    with torch.no_grad():
        clean = zero_block_mean(target.detach() - base.detach()) * residual_scale
        clean = F.pixel_unshuffle(clean, DEFAULT_SCALE)
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        noise = _zero_packed_channel_mean(noise)
        timesteps = torch.randint(
            scheduler.config.num_train_timesteps,
            (base.shape[0],),
            device=base.device,
            generator=generator,
        )
        noisy = _zero_packed_channel_mean(scheduler.add_noise(clean, noise, timesteps))
    predicted_noise = model(
        noisy,
        base.detach(),
        observation.detach(),
        timesteps,
    )
    if predicted_noise.shape != noise.shape:
        raise ValueError("model output must match the packed diffusion noise shape")
    predicted_noise = _zero_packed_channel_mean(predicted_noise)
    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def sample_residual(model, base, observation, scheduler, *, steps, generator=None):
    """Sample a normalized full-field prediction without accessing a target.

    The returned field is the fixed aligned base plus a denoised residual. A
    residual-space projection removes each 4x4 block mean at the end, enforcing
    exact LR block averages up to floating-point roundoff. ``scheduler`` is
    cloned with ``clip_sample=False`` because clipping the 128x-scaled residual
    would erase or distort valid denoising updates; the caller's scheduler is
    left untouched.
    """
    residual_scale = _validate_model_batch(model, base, observation)
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer")
    if not 2 <= steps <= scheduler.config.num_train_timesteps:
        raise ValueError("steps must be between two and the training diffusion length")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("residual diffusion requires epsilon prediction")
    _assert_aligned(base, observation, DEFAULT_SCALE)
    if model.training:
        raise ValueError("sampling requires the residual denoiser in eval mode")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("sampling uses scheduler updates; use torch.no_grad instead")

    # DDPMScheduler.from_config preserves the trained beta schedule while
    # allowing unclipped x_0 predictions for this deliberately scaled target.
    sampling_scheduler = DDPMScheduler.from_config(
        scheduler.config, clip_sample=False, timestep_spacing="linspace"
    )
    sampling_scheduler.set_timesteps(steps, device=base.device)
    latent = _zero_packed_channel_mean(torch.randn(
        (base.shape[0], 16, OBS_HEIGHT, OBS_WIDTH),
        device=base.device,
        dtype=base.dtype,
        generator=generator,
    ))
    for timestep in sampling_scheduler.timesteps:
        timestep_value = int(timestep.item())
        times = torch.full(
            (base.shape[0],), timestep_value, device=base.device, dtype=torch.long
        )
        predicted_noise = model(latent, base, observation, times)
        predicted_noise = _zero_packed_channel_mean(predicted_noise)
        latent = sampling_scheduler.step(
            predicted_noise, timestep, latent, generator=generator
        ).prev_sample
        # DDPMScheduler adds unconstrained posterior noise internally. Project
        # its result back to the valid residual subspace after every update.
        latent = _zero_packed_channel_mean(latent)

    residual = F.pixel_shuffle(latent / residual_scale, DEFAULT_SCALE)
    residual = zero_block_mean(residual, DEFAULT_SCALE)
    return base + residual
