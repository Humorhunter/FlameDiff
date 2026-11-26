"""MMPS-only posterior sampling for spatial latent diffusion.

The sampler uses one unconditional epsilon predictor and a frozen field
decoder. Task-specific conditioning is supplied as ``measure(field)``, which
maps a decoded field to its measurements (eight sensors, known inpainting
pixels, or block-averaged low-resolution pixels).
"""

import inspect
from math import ceil, isfinite

import torch

from spatial_mmps import mmps_guidance


FIELD_SHAPE = (1, 192, 256)
LATENT_SHAPE = (1, 48, 64)


def _validate_positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _set_custom_timesteps(scheduler, active_schedule):
    if "timesteps" in inspect.signature(scheduler.set_timesteps).parameters:
        scheduler.set_timesteps(timesteps=active_schedule)
    else:
        scheduler.timesteps = torch.tensor(
            active_schedule,
            dtype=scheduler.timesteps.dtype,
            device=scheduler.timesteps.device,
        )


def _warm_timesteps(scheduler, steps, strength):
    try:
        strength = float(strength)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("strength must be finite and in (0, 1]") from None
    if not isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("strength must be finite and in (0, 1]")
    scheduler.set_timesteps(steps)
    full_schedule = scheduler.timesteps.tolist()
    active_count = max(1, ceil(steps * strength))
    active_schedule = full_schedule[-active_count:]
    _set_custom_timesteps(scheduler, active_schedule)
    return scheduler.timesteps.tolist()


def _posterior_noise(predicted_noise, likelihood_score, alpha_bar):
    """Convert the MMPS likelihood score to an epsilon correction."""
    return predicted_noise - (1 - alpha_bar).sqrt() * likelihood_score


def _module_device_dtype(module):
    for tensor in module.parameters():
        return tensor.device, tensor.dtype
    for tensor in module.buffers():
        return tensor.device, tensor.dtype
    return None, None


def sample_mmps(
    model,
    decoder,
    measure,
    observation,
    *,
    scheduler,
    steps=1000,
    sigma=0.05,
    maxiter=8,
    generator=None,
    seed=None,
    initial_latent=None,
    strength=None,
    latent_channels=None,
    field_channels=None,
    latent_scale=None,
    solver_space="auto",
    guidance_mode="mmps",
    dps_weight=0.30,
    mmps_adaptive_weight=None,
    channel_weights=None,
    guidance_max_timestep=None,
    score_clip=None,
    eta=None,
    diagnostics=None,
):
    """Sample one field using MMPS guidance in the learned latent space.

    Args:
        model: Frozen unconditional epsilon-prediction denoiser. It is called
            as ``model(latent, None, timestep_batch)``.
        decoder: Frozen latent-to-field decoder. Gradients through its input
            must remain enabled while model and decoder parameters stay frozen.
        measure: Callable mapping a decoded field to task observations, e.g.
            ``operator.measure`` from ``spatial_mmps_observation``. Its output
            dimension defines the MMPS linear system; eight sensors therefore
            produce an 8-dimensional GMRES system.
        observation: Batched measured values whose shape exactly matches the
            callable's output. Inpainting callers should pass only known pixels
            in the same compact order returned by their measurement callable.
        scheduler: Epsilon-prediction VP/DDPM scheduler.
        steps: Number of inference steps from the scheduler's full schedule.
        sigma: Positive observation-noise scale, in the normalized field units
            used by the training data.
        maxiter: Maximum GMRES Krylov iterations per diffusion step. The
            solver caps this at the number of scalar observations.
        generator: Optional device-compatible ``torch.Generator``.
        seed: Optional seed used to create a generator on the observation
            device. It cannot be supplied together with ``generator``.
        initial_latent: Optional encoded initializer (e.g. AE-encoded harmonic
            or bicubic reconstruction) for suffix/warm-start sampling.
        strength: Active fraction of the full diffusion schedule when
            ``initial_latent`` is supplied. Must be in ``(0, 1]``.
        diagnostics: Optional callback receiving per-step solver diagnostics
            and a final summary including raw observation RMSE. The returned
            field is not projected; callers can apply the operator's exact
            affine projection separately.

    Returns:
        A normalized ``[1, 1, 192, 256]`` field before optional exact
        observation projection.
    """
    if model.training or decoder.training:
        raise ValueError("MMPS requires the denoiser and decoder in eval mode")
    if not callable(measure):
        raise ValueError("measure must be callable")
    if not isinstance(observation, torch.Tensor):
        raise ValueError("observation must be a tensor")
    _validate_positive_int("steps", steps)
    _validate_positive_int("maxiter", maxiter)
    if steps > scheduler.config.num_train_timesteps:
        raise ValueError("steps cannot exceed the scheduler's training diffusion length")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("MMPS requires an epsilon-prediction scheduler")
    try:
        sigma = float(sigma)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("sigma must be finite and strictly positive") from None
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive")
    if (initial_latent is None) != (strength is None):
        raise ValueError("Warm starts require both initial_latent and strength")
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if generator is not None:
            raise ValueError("Provide either seed or generator, not both")
    if solver_space not in ("auto", "observation", "latent"):
        raise ValueError("solver_space must be one of: auto, observation, latent")
    if guidance_mode not in ("mmps", "dps"):
        raise ValueError("guidance_mode must be one of: mmps, dps")
    try:
        dps_weight = float(dps_weight)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("dps_weight must be finite and strictly positive") from None
    if not isfinite(dps_weight) or dps_weight <= 0:
        raise ValueError("dps_weight must be finite and strictly positive")
    if mmps_adaptive_weight is not None:
        try:
            mmps_adaptive_weight = float(mmps_adaptive_weight)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("mmps_adaptive_weight must be finite and strictly positive") from None
        if not isfinite(mmps_adaptive_weight) or mmps_adaptive_weight <= 0:
            raise ValueError("mmps_adaptive_weight must be finite and strictly positive")
    if guidance_max_timestep is not None:
        if (
            not isinstance(guidance_max_timestep, int)
            or isinstance(guidance_max_timestep, bool)
            or guidance_max_timestep < 0
        ):
            raise ValueError("guidance_max_timestep must be a non-negative integer")
    if score_clip is not None:
        try:
            score_clip = float(score_clip)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("score_clip must be finite and strictly positive") from None
        if not isfinite(score_clip) or score_clip <= 0:
            raise ValueError("score_clip must be finite and strictly positive")
    if eta is not None:
        try:
            eta = float(eta)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("eta must be finite and in [0, 1]") from None
        if not isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be finite and in [0, 1]")

    if torch.is_inference_mode_enabled():
        raise RuntimeError("MMPS needs autograd; use torch.no_grad rather than inference_mode")

    device, dtype = _module_device_dtype(model)
    decoder_device, decoder_dtype = _module_device_dtype(decoder)
    if device is None:
        device, dtype = decoder_device, decoder_dtype
    if device is None:
        device, dtype = observation.device, observation.dtype
    if decoder_device is not None and decoder_device != device:
        raise ValueError("denoiser and decoder must be on the same device")
    if decoder_dtype is not None and decoder_dtype != dtype:
        raise ValueError("denoiser and decoder must use the same dtype")
    if observation.device != device or observation.dtype != dtype:
        raise ValueError("observation, denoiser, and decoder must share device and dtype")

    # MMPS's matrix-free solve currently operates on one frame at a time. The
    # measurement dimension may vary by task, but its leading batch dimension
    # must remain one.
    if observation.ndim < 2:
        raise ValueError("observation must have a batch dimension and measurement dimensions")
    batch_size = observation.shape[0]
    if batch_size != 1:
        raise ValueError("MMPS currently requires exactly one observation per call")
    if observation[0].numel() == 0:
        raise ValueError("observation must contain at least one measured value")
    observation = observation.detach()
    if not torch.isfinite(observation).all().item():
        raise ValueError("observation contains non-finite values")

    model_channels = getattr(model, "in_channels", None)
    decoder_channels = getattr(decoder, "latent_channels", None)
    if (
        model_channels is not None
        and decoder_channels is not None
        and model_channels != decoder_channels
    ):
        raise ValueError(
            f"denoiser in_channels ({model_channels}) must match decoder latent_channels ({decoder_channels})"
        )
    if latent_channels is None:
        if model_channels is not None:
            latent_channels = int(model_channels)
        elif decoder_channels is not None:
            latent_channels = int(decoder_channels)
        elif initial_latent is not None and isinstance(initial_latent, torch.Tensor) and initial_latent.ndim == 4:
            latent_channels = int(initial_latent.shape[1])
        else:
            latent_channels = LATENT_SHAPE[0]
    _validate_positive_int("latent_channels", latent_channels)

    expected_latent_shape = (batch_size, latent_channels, LATENT_SHAPE[1], LATENT_SHAPE[2])
    if initial_latent is not None:
        if not isinstance(initial_latent, torch.Tensor):
            raise ValueError("initial_latent must be a tensor")
        if tuple(initial_latent.shape) != expected_latent_shape:
            raise ValueError(f"Expected initial_latent with shape {expected_latent_shape}")
        if initial_latent.device != device or initial_latent.dtype != dtype:
            raise ValueError("initial_latent must share device and dtype with the models")
        if not torch.isfinite(initial_latent).all().item():
            raise ValueError("initial_latent contains non-finite values")

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    elif generator is not None:
        if not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator")
        generator_device = generator.device
        if generator_device.type != device.type:
            raise ValueError("generator and models must use the same device type")
        if (generator_device.index is not None and device.index is not None
                and generator_device.index != device.index):
            raise ValueError("generator and models must use the same device")

    if initial_latent is not None:
        active_timesteps = _warm_timesteps(scheduler, steps, strength)
        start_timestep = torch.full(
            (batch_size,), active_timesteps[0], device=device, dtype=torch.long
        )
        noise = torch.randn(
            initial_latent.shape, device=device, dtype=dtype, generator=generator
        )
        latent = scheduler.add_noise(initial_latent.detach(), noise, start_timestep)
    else:
        if steps == 1:
            scheduler.set_timesteps(1)
            _set_custom_timesteps(scheduler, [scheduler.config.num_train_timesteps - 1])
        else:
            scheduler.set_timesteps(steps)
        latent = torch.randn(
            expected_latent_shape, device=device, dtype=dtype, generator=generator
        )

    alpha_schedule = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    observation_dimension = observation[0].numel()
    step_count = len(scheduler.timesteps)
    total_gmres_iterations = 0

    if field_channels is None:
        inferred_out = getattr(decoder, "out_channels", None)
        field_channels = int(inferred_out) if inferred_out is not None else FIELD_SHAPE[0]
    _validate_positive_int("field_channels", field_channels)
    expected_field_shape = (field_channels, FIELD_SHAPE[1], FIELD_SHAPE[2])

    scale_mean = None
    scale_std = None
    if latent_scale is not None:
        scale_mean = torch.as_tensor(latent_scale["mean"], device=device, dtype=dtype).view(1, -1, 1, 1)
        scale_std = torch.as_tensor(latent_scale["std"], device=device, dtype=dtype).view(1, -1, 1, 1)
        if scale_mean.shape[1] != latent_channels or scale_std.shape[1] != latent_channels:
            raise ValueError("latent_scale channel count must match latent_channels")
        if not torch.isfinite(scale_mean).all().item() or not torch.isfinite(scale_std).all().item():
            raise ValueError("latent_scale must contain finite values")
        if bool((scale_std <= 0).any().item()):
            raise ValueError("latent_scale std must be strictly positive")

    def decode_standardized(std_latent):
        raw_latent = std_latent if scale_mean is None else (std_latent * scale_std + scale_mean)
        return decoder(raw_latent)

    def measure_clean(clean_latent):
        return measure(decode_standardized(clean_latent))

    step_supports_eta = "eta" in inspect.signature(scheduler.step).parameters

    clip_enabled = bool(getattr(scheduler.config, "clip_sample", False))
    clip_bound = float(getattr(scheduler.config, "clip_sample_range", 1.0))

    # Optional per-observed-channel weighting tensor for balanced multi-field DPS
    dps_ch_w = None
    if channel_weights is not None:
        dps_ch_w = torch.as_tensor(channel_weights, device=device, dtype=dtype)

    # The custom transformer uses MultiheadAttention. Disabling the fused
    # fastpath keeps the repeated forward/JVP path available on CPU and CUDA.
    previous_mha_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for step_index, timestep in enumerate(scheduler.timesteps.tolist(), start=1):
            time_batch = torch.full(
                (batch_size, 1), timestep, device=device, dtype=torch.long
            )
            alpha_bar = alpha_schedule[timestep]
            guided = guidance_max_timestep is None or int(timestep) <= guidance_max_timestep

            if guided and guidance_mode == "mmps":
                with torch.enable_grad():
                    current = latent.detach().requires_grad_(True)
                    predicted_noise = model(current, None, time_batch)
                    if not isinstance(predicted_noise, torch.Tensor):
                        raise TypeError("denoiser must return an epsilon tensor")
                    if predicted_noise.shape != current.shape:
                        raise ValueError("denoiser output shape must match the latent shape")
                    clean = (current - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                    clean_for_meas = (
                        clean.clamp(-clip_bound, clip_bound)
                        if (clip_enabled and clip_bound > 1.0)
                        else clean
                    )
                    score, solver = mmps_guidance(
                        current,
                        clean_for_meas,
                        observation,
                        measure_clean,
                        alpha_bar,
                        sigma=sigma,
                        maxiter=maxiter,
                        solver_space=solver_space,
                    )
            elif guided and guidance_mode == "dps":
                # Adaptive Normalized DPS (Wu, Liang & Luo, Combustion & Flame 282, 2025, Eq. 18-19):
                # scales the likelihood gradient norm to dps_weight * ||eps||_2 per step,
                # avoiding GMRES ill-conditioning and working identically on CPU, CUDA, and MPS.
                with torch.enable_grad():
                    current = latent.detach().requires_grad_(True)
                    predicted_noise = model(current, None, time_batch)
                    if not isinstance(predicted_noise, torch.Tensor):
                        raise TypeError("denoiser must return an epsilon tensor")
                    if predicted_noise.shape != current.shape:
                        raise ValueError("denoiser output shape must match the latent shape")
                    clean = (current - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                    clean_for_meas = (
                        clean.clamp(-clip_bound, clip_bound) if clip_enabled else clean
                    )
                    measured = measure_clean(clean_for_meas)
                    residual = observation - measured
                    if dps_ch_w is not None:
                        if dps_ch_w.shape == residual.shape:
                            loss_val = 0.5 * (residual.square() * dps_ch_w).sum()
                        elif residual.ndim >= 3 and residual.shape[1] == dps_ch_w.numel():
                            view_shape = (1, -1) + (1,) * (residual.ndim - 2)
                            loss_val = 0.5 * (residual.square() * dps_ch_w.view(view_shape)).sum()
                        else:
                            loss_val = 0.5 * residual.square().sum()
                    else:
                        loss_val = 0.5 * residual.square().sum()
                    grad_cur, = torch.autograd.grad(loss_val, current)
                with torch.no_grad():
                    eps_norm_t = torch.linalg.vector_norm(predicted_noise.detach())
                    grad_norm_t = torch.linalg.vector_norm(grad_cur.detach())
                    denom_alpha = (1.0 - alpha_bar).sqrt().clamp_min(1e-8)
                    if float(grad_norm_t.item()) > 1e-12:
                        # _posterior_noise uses eps_cond = eps - sqrt(1-ab)*score, so
                        # moving z toward -grad_cur means score = -(dps_weight * ||eps|| / ||grad||) * grad / sqrt(1-ab)
                        score = -(dps_weight * eps_norm_t / grad_norm_t / denom_alpha) * grad_cur.detach()
                    else:
                        score = torch.zeros_like(predicted_noise)
                    solver = {
                        "relative_residual": 0.0,
                        "iterations": 1.0,
                        "converged": 1.0,
                        "breakdown": 0.0,
                        "measurement_rmse": float(residual.detach().square().mean().sqrt().item()),
                        "sigma": float(sigma),
                        "alpha_bar": float(alpha_bar.item()),
                        "solver_space": "adaptive_dps",
                    }
            else:
                with torch.no_grad():
                    current = latent.detach()
                    predicted_noise = model(current, None, time_batch)
                    if not isinstance(predicted_noise, torch.Tensor):
                        raise TypeError("denoiser must return an epsilon tensor")
                    if predicted_noise.shape != current.shape:
                        raise ValueError("denoiser output shape must match the latent shape")
                    clean = (current - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                    score = torch.zeros_like(predicted_noise)
                    solver = {
                        "relative_residual": 0.0,
                        "iterations": 0.0,
                        "converged": 1.0,
                        "breakdown": 0.0,
                        "measurement_rmse": None,
                        "sigma": float(sigma),
                        "alpha_bar": float(alpha_bar.item()),
                        "solver_space": "unconditional",
                    }

            with torch.no_grad():
                eps_norm = float(torch.linalg.vector_norm(predicted_noise.detach()).item())
                correction = (1 - alpha_bar).sqrt() * score
                corr_norm = float(torch.linalg.vector_norm(correction).item())
                if guidance_mode == "mmps":
                    if mmps_adaptive_weight is not None and corr_norm > 1e-12:
                        # Wu et al. (2025, Eq. 18-19) adaptive norm matching applied directly
                        # to MMPS's exact covariance-inverted likelihood score direction.
                        target_corr = mmps_adaptive_weight * max(eps_norm, 1e-12)
                        score = score * (target_corr / corr_norm)
                        corr_norm = target_corr
                    elif score_clip is not None and corr_norm > 0.0:
                        max_corr = score_clip * max(eps_norm, 1e-12)
                        if corr_norm > max_corr:
                            score = score * (max_corr / corr_norm)
                            corr_norm = max_corr

                conditioned_noise = _posterior_noise(predicted_noise, score, alpha_bar)
                step_kwargs = {"generator": generator}
                if step_supports_eta and eta is not None:
                    step_kwargs["eta"] = eta
                latent = scheduler.step(
                    conditioned_noise,
                    timestep,
                    current,
                    **step_kwargs,
                ).prev_sample

                if diagnostics is not None:
                    clip_bound = float(getattr(scheduler.config, "clip_sample_range", 1.0))
                    clean_oob = float((clean.detach().abs() > clip_bound).float().mean().item())
                    posterior_clean = (
                        current.detach() - (1 - alpha_bar).sqrt() * conditioned_noise
                    ) / alpha_bar.sqrt()
                    posterior_clipped = (
                        float((posterior_clean.abs() > clip_bound).float().mean().item())
                        if getattr(scheduler.config, "clip_sample", False)
                        else 0.0
                    )
                    score_ratio = corr_norm / max(eps_norm, 1e-12)

            total_gmres_iterations += int(solver["iterations"])
            if diagnostics is not None:
                diagnostics({
                    **solver,
                    "phase": "step",
                    "step": step_index,
                    "steps": step_count,
                    "timestep": int(timestep),
                    "guided": guided,
                    "measurement_dimension": int(observation_dimension),
                    "measurement_rmse": solver["measurement_rmse"],
                    "clean_out_of_unit_box_fraction": clean_oob,
                    "posterior_clipped_fraction": posterior_clipped,
                    "score_to_noise_norm_ratio": score_ratio,
                })
            del current, predicted_noise, clean, score, conditioned_noise
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_mha_fastpath)

    with torch.no_grad():
        field = decode_standardized(latent)
        if not isinstance(field, torch.Tensor) or tuple(field.shape[1:]) != expected_field_shape:
            raise ValueError(
                f"decoder must return [B, {expected_field_shape[0]}, "
                f"{expected_field_shape[1]}, {expected_field_shape[2]}]"
            )
        if not torch.isfinite(field).all().item():
            raise FloatingPointError("decoder produced a non-finite temperature field")
        raw_measurement = measure(field)
        if not isinstance(raw_measurement, torch.Tensor) or raw_measurement.shape != observation.shape:
            raise ValueError("measure(decoded field) must match observation.shape")
        if raw_measurement.device != observation.device or raw_measurement.dtype != observation.dtype:
            raise ValueError("measure(decoded field) and observation must share device and dtype")
        if not torch.isfinite(raw_measurement).all().item():
            raise FloatingPointError("measurement operator produced non-finite values")
        raw_observation_rmse = float((raw_measurement - observation).square().mean().sqrt().item())

    if diagnostics is not None:
        diagnostics({
            "phase": "summary",
            "steps": step_count,
            "measurement_dimension": int(observation_dimension),
            "total_gmres_iterations": total_gmres_iterations,
            "raw_observation_rmse": raw_observation_rmse,
        })
    return field
