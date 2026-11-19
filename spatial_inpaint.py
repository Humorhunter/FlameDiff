"""Conditional residual diffusion for fixed-mask spatial inpainting."""

from math import isfinite, log

import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.nn import functional as F


FIELD_HEIGHT = 192
FIELD_WIDTH = 256
DEFAULT_RESIDUAL_SCALE = 16.0


def inpaint_diffusion_scheduler(train_steps=1000):
    """Create the epsilon-prediction DDPM schedule for inpaint residuals."""
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


def _validate_inputs(base, observation, known_mask):
    _validate_field(base, "base")
    _validate_field(observation, "observation")
    if base.shape != observation.shape:
        raise ValueError("base and observation must have the same shape")
    if base.device != observation.device or base.dtype != observation.dtype:
        raise ValueError("base and observation must share device and dtype")
    if known_mask.shape != base.shape or known_mask.dtype != torch.bool:
        raise ValueError("known_mask must be a boolean tensor matching base shape")
    if known_mask.device != base.device:
        raise ValueError("known_mask must be on the same device as base")
    if not torch.isfinite(base).all() or not torch.isfinite(observation).all():
        raise ValueError("base and observation must be finite")
    missing_mask = ~known_mask
    if not missing_mask.any(dim=(1, 2, 3)).all():
        raise ValueError("known_mask must leave at least one missing pixel per sample")
    if not known_mask.any(dim=(1, 2, 3)).all():
        raise ValueError("known_mask must contain at least one observed pixel per sample")
    return missing_mask


def _sinusoidal_embedding(timesteps, dimension, dtype):
    half = dimension // 2
    frequencies = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (-log(10000.0) / max(half - 1, 1))
    )
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(dtype=dtype)


def _spatial_position_encoding(height, width, channels, *, device, dtype):
    """Build fixed 2D sine/cosine coordinates for bottleneck attention."""
    axis_channels = channels // 2
    pairs = axis_channels // 2
    frequencies = torch.exp(
        torch.arange(pairs, device=device, dtype=torch.float32)
        * (-log(10000.0) / max(pairs - 1, 1))
    )
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    y_angles = y[:, None] * frequencies[None, :]
    x_angles = x[:, None] * frequencies[None, :]
    y_encoding = torch.cat((y_angles.sin(), y_angles.cos()), dim=1)
    x_encoding = torch.cat((x_angles.sin(), x_angles.cos()), dim=1)
    y_encoding = y_encoding[:, None, :].expand(height, width, axis_channels)
    x_encoding = x_encoding[None, :, :].expand(height, width, axis_channels)
    return torch.cat((y_encoding, x_encoding), dim=-1).reshape(1, height * width, channels).to(dtype)


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


class InpaintResidualDenoiser(nn.Module):
    """Convolutional residual U-Net with optional bottleneck self-attention.

    ``attention_blocks=0`` gives the convolution-only baseline. Values one or
    two add Transformer blocks at 24x32 resolution, where global context can
    cross the center hole without applying attention to the full-resolution
    feature maps.

    ``known_mask`` is True for measured pixels. The model receives the noisy
    residual, frozen U-Net base, visible observation, known-pixel mask, and
    diffusion timestep as separate inputs.
    """

    def __init__(self, width=64, residual_scale=DEFAULT_RESIDUAL_SCALE, attention_blocks=0):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError("width must be an integer multiple of eight and at least 8")
        if not isinstance(attention_blocks, int) or isinstance(attention_blocks, bool):
            raise ValueError("attention_blocks must be an integer from zero to two")
        if attention_blocks not in (0, 1, 2):
            raise ValueError("attention_blocks must be zero, one, or two")
        try:
            residual_scale = float(residual_scale)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("residual_scale must be finite and positive") from None
        if not isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        self.residual_scale = residual_scale
        self.attention_blocks = attention_blocks

        time_channels = width * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(width, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(4, width, 3, padding=1)
        self.enc0 = _TimeResidualBlock(width, width, time_channels)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.enc1 = _TimeResidualBlock(width * 2, width * 2, time_channels)
        self.down2 = nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1)
        self.enc2 = _TimeResidualBlock(width * 3, width * 3, time_channels)
        self.down3 = nn.Conv2d(width * 3, width * 4, 3, stride=2, padding=1)
        self.middle1 = _TimeResidualBlock(width * 4, width * 4, time_channels)
        self.middle2 = _TimeResidualBlock(width * 4, width * 4, time_channels)
        self.dec2 = _TimeResidualBlock(width * 7, width * 3, time_channels)
        self.dec1 = _TimeResidualBlock(width * 5, width * 2, time_channels)
        self.dec0 = _TimeResidualBlock(width * 3, width, time_channels)
        self.output_norm = nn.GroupNorm(8, width)
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        # Create optional attention last so shared convolutional parameters
        # retain identical seeded initialization across ablation variants.
        self.attention = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width * 4,
                nhead=4,
                dim_feedforward=width * 8,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(attention_blocks)
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_residual, base, observation, known_mask, timesteps):
        missing_mask = _validate_inputs(base, observation, known_mask)
        _validate_field(noisy_residual, "noisy_residual")
        if noisy_residual.shape != base.shape:
            raise ValueError("noisy_residual and base must have the same shape")
        if noisy_residual.device != base.device or noisy_residual.dtype != base.dtype:
            raise ValueError("noisy_residual and base must share device and dtype")
        if timesteps.ndim not in (1, 2) or timesteps.shape[0] != base.shape[0]:
            raise ValueError("timesteps must have one entry per batch item")
        if timesteps.ndim == 2 and timesteps.numel() != base.shape[0]:
            raise ValueError("timesteps must have one entry per batch item")

        time = timesteps.reshape(-1).to(device=base.device)
        time_embedding = self.time_mlp(
            _sinusoidal_embedding(time, self.time_mlp[0].in_features, base.dtype)
        )
        model_input = torch.cat(
            (noisy_residual * missing_mask, base, observation, known_mask.to(base.dtype)),
            dim=1,
        )
        skip0 = self.enc0(self.input(model_input), time_embedding)
        skip1 = self.enc1(self.down1(skip0), time_embedding)
        skip2 = self.enc2(self.down2(skip1), time_embedding)
        middle = self.middle1(self.down3(skip2), time_embedding)
        if self.attention:
            batch, channels, height, width = middle.shape
            position = _spatial_position_encoding(
                height, width, channels, device=middle.device, dtype=middle.dtype
            )
            tokens = middle.flatten(2).transpose(1, 2) + position
            for block in self.attention:
                tokens = block(tokens)
            middle = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        middle = self.middle2(middle, time_embedding)
        middle = F.interpolate(middle, size=skip2.shape[-2:], mode="nearest")
        middle = self.dec2(torch.cat((middle, skip2), dim=1), time_embedding)
        middle = F.interpolate(middle, size=skip1.shape[-2:], mode="nearest")
        middle = self.dec1(torch.cat((middle, skip1), dim=1), time_embedding)
        middle = F.interpolate(middle, size=skip0.shape[-2:], mode="nearest")
        middle = self.dec0(torch.cat((middle, skip0), dim=1), time_embedding)
        return self.output(F.silu(self.output_norm(middle))) * missing_mask


def masked_residual_loss(
    model, base, observation, known_mask, target, scheduler, generator=None
):
    """Train noise prediction only on the unknown region's U-Net residual."""
    missing_mask = _validate_inputs(base, observation, known_mask)
    _validate_field(target, "target")
    if target.shape != base.shape:
        raise ValueError("target and base must have the same shape")
    if target.device != base.device or target.dtype != base.dtype:
        raise ValueError("target and base must share device and dtype")
    residual_scale = model.residual_scale
    if not isfinite(float(residual_scale)) or float(residual_scale) <= 0:
        raise ValueError("model must define a finite positive residual_scale")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("inpaint residual diffusion requires epsilon prediction")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")

    with torch.no_grad():
        clean = (target - base) * missing_mask * residual_scale
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        ) * missing_mask
        timesteps = torch.randint(
            scheduler.config.num_train_timesteps,
            (base.shape[0],),
            device=base.device,
            generator=generator,
        )
        noisy = scheduler.add_noise(clean, noise, timesteps) * missing_mask
    predicted_noise = model(noisy, base.detach(), observation.detach(), known_mask, timesteps)
    predicted_noise = predicted_noise * missing_mask
    squared_error = (predicted_noise - noise).square() * missing_mask
    return squared_error.sum() / missing_mask.expand_as(squared_error).sum()


@torch.no_grad()
def sample_inpaint_residual(
    model, base, observation, known_mask, scheduler, *, steps, generator=None
):
    """Sample an unknown-region residual and copy observed pixels exactly."""
    missing_mask = _validate_inputs(base, observation, known_mask)
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer")
    if not 2 <= steps <= scheduler.config.num_train_timesteps:
        raise ValueError("steps must be between two and the training diffusion length")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("inpaint residual diffusion requires epsilon prediction")
    if model.training:
        raise ValueError("sampling requires the inpaint residual denoiser in eval mode")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("sampling uses scheduler updates; use torch.no_grad instead")

    sampling_scheduler = DDPMScheduler.from_config(
        scheduler.config, clip_sample=False, timestep_spacing="linspace"
    )
    sampling_scheduler.set_timesteps(steps, device=base.device)
    latent = torch.randn(
        base.shape, device=base.device, dtype=base.dtype, generator=generator
    ) * missing_mask
    for timestep in sampling_scheduler.timesteps:
        times = torch.full(
            (base.shape[0],), int(timestep.item()), device=base.device, dtype=torch.long
        )
        predicted_noise = model(latent, base, observation, known_mask, times) * missing_mask
        latent = sampling_scheduler.step(
            predicted_noise, timestep, latent, generator=generator
        ).prev_sample * missing_mask

    residual = latent / float(model.residual_scale)
    return torch.where(known_mask, observation, base + residual)
