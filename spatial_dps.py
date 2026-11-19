"""Single-frame latent diffusion with Du et al. style posterior sampling."""

from dataclasses import dataclass
from math import ceil, isfinite, sqrt

import torch
from diffusers import DDPMScheduler
from torch.nn import functional


FIELD_SHAPE = (1, 192, 256)
LATENT_SHAPE = (1, 48, 64)


@dataclass(frozen=True)
class SpatialObservation:
    task: str
    scale: int = 4

    def __post_init__(self):
        if self.task not in {"inpaint", "sr"} or self.scale != 4:
            raise ValueError("Supported observations are centered inpaint and 4x block-average SR")

    @property
    def shape(self):
        return FIELD_SHAPE if self.task == "inpaint" else (1, 48, 64)

    @property
    def measured_count(self):
        if self.task == "inpaint":
            return 192 * 256 - 96 * 128
        return 48 * 64

    def mask(self, reference):
        if self.task != "inpaint":
            raise ValueError("Only inpainting uses a spatial mask")
        known = torch.ones_like(reference, dtype=torch.bool)
        known[..., 48:144, 64:192] = False
        return known

    def observe(self, field):
        if field.ndim != 4 or tuple(field.shape[1:]) != FIELD_SHAPE:
            raise ValueError(f"Expected a batch of fields with shape {FIELD_SHAPE}")
        if self.task == "inpaint":
            return field.masked_fill(~self.mask(field), 0)
        return functional.avg_pool2d(field, self.scale)

    def restore_known(self, field, observation):
        if self.task == "inpaint":
            return torch.where(self.mask(field), observation, field)
        return field


def diffusion_scheduler(train_steps=1000):
    return DDPMScheduler(
        num_train_timesteps=train_steps,
        prediction_type="epsilon",
        clip_sample=True,
        timestep_spacing="linspace",
    )


def diffusion_loss(model, autoencoder, scheduler, target, generator=None):
    with torch.no_grad():
        latent = autoencoder.encoder(target)
        noise = torch.randn(
            latent.shape, dtype=latent.dtype, device=latent.device, generator=generator
        )
        timesteps = torch.randint(
            scheduler.config.num_train_timesteps,
            (target.shape[0],),
            device=target.device,
            generator=generator,
        )
        noisy = scheduler.add_noise(latent, noise, timesteps)
    return functional.mse_loss(model(noisy, None, timesteps[:, None]), noise)


def warm_timesteps(scheduler, steps, strength):
    """Set and return the active suffix of the full linspace inference schedule.

    This updates ``scheduler.timesteps`` to the explicit suffix so DDPM's
    ``previous_timestep`` follows that same schedule. A one-step warm start
    uses the full one-step linspace schedule, whose sole timestep is zero.
    """
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("Inference steps must be an integer")
    if not 1 <= steps <= scheduler.config.num_train_timesteps:
        raise ValueError("Inference steps must be between one and the training diffusion length")
    try:
        strength = float(strength)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Warm-start strength must be finite and in (0, 1]") from None
    if not isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Warm-start strength must be finite and in (0, 1]")

    scheduler.set_timesteps(steps)
    full_schedule = scheduler.timesteps.tolist()
    active_count = max(1, ceil(steps * strength))
    active_schedule = full_schedule[-active_count:]
    scheduler.set_timesteps(timesteps=active_schedule)
    return scheduler.timesteps.tolist()


def posterior_noise(predicted_noise, likelihood_score, alpha_bar):
    """Convert an additive VP likelihood score to an epsilon correction."""
    return predicted_noise - (1 - alpha_bar).sqrt() * likelihood_score


def sample_dps(
    model,
    decoder,
    operator,
    observation,
    *,
    scheduler,
    steps=1000,
    guidance=1.0,
    generator=None,
    initial_latent=None,
    strength=None,
    mmps_sigma=None,
    mmps_maxiter=3,
    mmps_diagnostics=None,
):
    """Generate a normalized field using DPS, or MMPS when a noise scale is provided."""
    if model.training or decoder.training:
        raise ValueError("DPS requires the denoiser and decoder in eval mode")
    if observation.ndim != 4 or tuple(observation.shape[1:]) != operator.shape:
        raise ValueError(f"Expected observations of shape [B, {operator.shape}]")
    if not isfinite(guidance) or guidance < 0:
        raise ValueError("Guidance must be finite and nonnegative")
    if not 1 <= steps <= scheduler.config.num_train_timesteps:
        raise ValueError("Inference steps must be between one and the training diffusion length")
    if (initial_latent is None) != (strength is None):
        raise ValueError("Warm starts require both initial_latent and strength")
    use_mmps = mmps_sigma is not None
    if use_mmps:
        from spatial_mmps import mmps_guidance

        if operator.task != "sr" or observation.shape[0] != 1:
            raise ValueError("Experimental MMPS supports SR with one observation per call")
        if scheduler.config.prediction_type != "epsilon" or guidance != 1.0:
            raise ValueError("MMPS requires epsilon prediction and unit score scaling")
        if not isfinite(mmps_sigma) or mmps_sigma <= 0:
            raise ValueError("MMPS observation sigma must be finite and positive")
        alpha_schedule = scheduler.alphas_cumprod.to(observation)

        def measure(clean_latent):
            return operator.observe(decoder(clean_latent))

    if guidance and torch.is_inference_mode_enabled():
        raise RuntimeError("DPS needs autograd; use no_grad rather than inference_mode")

    observation = observation.detach()
    if operator.task == "inpaint":
        observation = operator.observe(observation)
    if not torch.isfinite(observation).all():
        raise ValueError("Measured temperatures must be finite")

    warm_start = initial_latent is not None
    if warm_start:
        if not isinstance(initial_latent, torch.Tensor):
            raise ValueError("initial_latent must be a tensor")
        expected_shape = (observation.shape[0], *LATENT_SHAPE)
        if tuple(initial_latent.shape) != expected_shape:
            raise ValueError(f"Expected initial_latent of shape {expected_shape}")
        if initial_latent.device != observation.device:
            raise ValueError("initial_latent and observation must be on the same device")
        if initial_latent.dtype != observation.dtype:
            raise ValueError("initial_latent and observation must have the same dtype")
        if not torch.isfinite(initial_latent).all():
            raise ValueError("initial_latent must be finite")
        active_timesteps = warm_timesteps(scheduler, steps, strength)
        start_timestep = torch.full(
            (observation.shape[0],),
            active_timesteps[0],
            device=observation.device,
            dtype=torch.long,
        )
        warm_noise = torch.randn(
            initial_latent.shape,
            device=initial_latent.device,
            dtype=initial_latent.dtype,
            generator=generator,
        )
        latent = scheduler.add_noise(initial_latent.detach(), warm_noise, start_timestep)
    else:
        if steps == 1:
            scheduler.set_timesteps(timesteps=[scheduler.config.num_train_timesteps - 1])
        else:
            scheduler.set_timesteps(steps)
        latent = torch.randn(
            (observation.shape[0], *LATENT_SHAPE),
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        )
    norm_scale = sqrt(operator.measured_count)
    for timestep in scheduler.timesteps.tolist():
        time_batch = torch.full(
            (latent.shape[0], 1), timestep, device=latent.device, dtype=torch.long
        )
        with torch.set_grad_enabled(bool(guidance)):
            current = latent.detach().requires_grad_(bool(guidance))
            predicted_noise = model(current, None, time_batch)
            if use_mmps:
                alpha = alpha_schedule[timestep]
                clean = (current - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
                score, diagnostics = mmps_guidance(
                    current, clean, observation, measure, alpha,
                    sigma=mmps_sigma, maxiter=mmps_maxiter,
                )
                with torch.no_grad():
                    conditioned_noise = posterior_noise(predicted_noise, score, alpha)
                    latent = scheduler.step(
                        conditioned_noise, timestep, current, generator=generator
                    ).prev_sample
                if mmps_diagnostics is not None:
                    mmps_diagnostics({**diagnostics, "timestep": timestep})
                del clean, score, conditioned_noise
            else:
                step = scheduler.step(predicted_noise, timestep, current, generator=generator)
                if guidance:
                    measured = operator.observe(decoder(step.pred_original_sample))
                    residual = (measured - observation).flatten(1)
                    distance = torch.linalg.vector_norm(residual, dim=1) / norm_scale
                    gradient, = torch.autograd.grad(distance.sum(), current)
                    latent = step.prev_sample.detach() - guidance * gradient
                    del measured, residual, distance, gradient
                else:
                    latent = step.prev_sample.detach()
                del step
        del current, predicted_noise

    with torch.no_grad():
        field = decoder(latent)
        return operator.restore_known(field, observation) if guidance or warm_start else field
