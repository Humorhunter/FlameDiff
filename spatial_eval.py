"""Metrics for single-frame spatial reconstruction tasks.

Inputs and predictions use the training-set temperature scale. Reported
temperature errors are converted back to Kelvin using the training data range.
"""

from __future__ import annotations

import math
import time
from itertools import islice
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


_SSIM_WINDOW = 7


def _as_frame(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """Convert one [1,H,W] or [1,1,H,W] frame to BCHW and validate it."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError(f"{name} must describe one [1,H,W] frame; got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _ssim(prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor) -> Optional[float]:
    """Uniform 7x7 local SSIM, averaged over windows fully inside region."""
    height, width = prediction.shape[-2:]
    window = min(_SSIM_WINDOW, height, width)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return None

    kernel = (window, window)
    mu_pred = F.avg_pool2d(prediction, kernel, stride=1)
    mu_target = F.avg_pool2d(target, kernel, stride=1)
    pred_sq = F.avg_pool2d(prediction.square(), kernel, stride=1) - mu_pred.square()
    target_sq = F.avg_pool2d(target.square(), kernel, stride=1) - mu_target.square()
    covariance = F.avg_pool2d(prediction * target, kernel, stride=1) - mu_pred * mu_target
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_pred * mu_target + c1) * (2 * covariance + c2)) / (
        (mu_pred.square() + mu_target.square() + c1) * (pred_sq + target_sq + c2)
    ).clamp_min(1e-12)

    valid = F.avg_pool2d(region.float(), kernel, stride=1) >= 1.0 - 1e-6
    if not valid.any():
        return None
    return float(score[valid].mean().item())


def _gradient_metrics(
    prediction_k: torch.Tensor,
    target_k: torch.Tensor,
    region: torch.Tensor,
    known: Optional[torch.Tensor],
) -> tuple[float, Optional[float]]:
    """Return mean finite-difference gradient error and boundary-only error."""
    errors: list[torch.Tensor] = []
    boundary_errors: list[torch.Tensor] = []

    # Differences are measured per grid cell, since the source grid spacing is
    # not part of this evaluator's API.
    for axis in (-2, -1):
        pred_delta = torch.diff(prediction_k, dim=axis)
        target_delta = torch.diff(target_k, dim=axis)
        delta_error = (pred_delta - target_delta).abs()
        if axis == -2:
            region_pair = region[..., 1:, :] | region[..., :-1, :]
            boundary_pair = None if known is None else (known[..., 1:, :] != known[..., :-1, :])
        else:
            region_pair = region[..., :, 1:] | region[..., :, :-1]
            boundary_pair = None if known is None else (known[..., :, 1:] != known[..., :, :-1])
        errors.append(delta_error[region_pair])
        if boundary_pair is not None and boundary_pair.any():
            boundary_errors.append(delta_error[boundary_pair])

    populated_errors = [item.reshape(-1) for item in errors if item.numel()]
    error_values = torch.cat(populated_errors) if populated_errors else torch.empty(0, device=prediction_k.device)
    gradient_error = float(error_values.mean().item()) if error_values.numel() else 0.0
    if boundary_errors:
        boundary_values = torch.cat([item.reshape(-1) for item in boundary_errors])
        boundary_error = float(boundary_values.mean().item())
    else:
        boundary_error = None
    return gradient_error, boundary_error


def _frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    predicted_observation: torch.Tensor,
    known: torch.Tensor,
    *,
    task: str,
    temperature_range: tuple[float, float],
) -> dict[str, Optional[float]]:
    low, high = temperature_range
    delta_t = high - low
    prediction_k = prediction * delta_t + low
    target_k = target * delta_t + low

    if task == "inpaint":
        region = ~known
        observed_frame = _as_frame(observed, name="inpainting observation")
        predicted_observation_frame = _as_frame(predicted_observation, name="predicted inpainting observation")
        if observed_frame.shape != target.shape or predicted_observation_frame.shape != target.shape:
            raise ValueError("Inpainting observations and reference frames must have identical shapes")
        observation_region = known
        if observation_region.any():
            obs_rmse = float((((predicted_observation_frame - observed_frame)[observation_region]
                               * delta_t).square().mean().sqrt()).item())
        else:
            obs_rmse = None
    else:
        region = torch.ones_like(known, dtype=torch.bool)
        if predicted_observation.shape != observed.shape:
            raise ValueError("operator.observe(prediction) must match the observation shape")
        observation_delta = (predicted_observation - observed) * delta_t
        obs_rmse = float(observation_delta.square().mean().sqrt().item())

    if not region.any():
        raise ValueError("Evaluation region is empty")
    residual = (prediction_k - target_k)[region]
    gradient_error, boundary_gradient_error = _gradient_metrics(
        prediction_k, target_k, region, known if task == "inpaint" else None
    )

    return {
        "mae_k": float(residual.abs().mean().item()),
        "rmse_k": float(residual.square().mean().sqrt().item()),
        "gradient_mae_k_per_grid_cell": gradient_error,
        "boundary_gradient_mae_k_per_grid_cell": boundary_gradient_error,
        "ssim": _ssim(prediction, target, region),
        "observation_rmse_k": obs_rmse,
    }


def _summarize_metrics(records: list[dict[str, Optional[float]]]) -> dict[str, Optional[float]]:
    keys = records[0].keys() if records else (
        "mae_k", "rmse_k", "gradient_mae_k_per_grid_cell",
        "boundary_gradient_mae_k_per_grid_cell", "ssim", "observation_rmse_k",
    )
    result: dict[str, Optional[float]] = {}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        result[key] = _mean(values)
    return result


def evaluate(
    dataset: Iterable[torch.Tensor],
    operator,
    predict: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: str | torch.device,
    temperature_range: Sequence[float],
    draws: int = 1,
    limit: Optional[int] = None,
) -> dict[str, object]:
    """Evaluate a single-frame spatial reconstruction method.

    ``dataset`` yields normalized ``[1,H,W]`` frames. ``operator.observe``
    creates the available measurement. For inpainting, ``operator.mask``
    marks known pixels with True. ``predict`` receives only that measurement and must return a
    normalized full-resolution ``[1,1,H,W]`` prediction. It is called outside
    any ``no_grad`` context so posterior-sampling predictors can use autograd.

    Inpainting errors are measured over missing pixels (and finite differences
    touching them); super-resolution errors cover the full field. The SSIM
    score uses a uniform 7x7 window, data range 1, and only windows fully
    inside the evaluated region.
    """
    if getattr(operator, "task", None) not in {"inpaint", "sr"}:
        raise ValueError("operator.task must be 'inpaint' or 'sr'")
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if len(temperature_range) != 2:
        raise ValueError("temperature_range must be (minimum_kelvin, maximum_kelvin)")
    low, high = (float(temperature_range[0]), float(temperature_range[1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("temperature_range must contain finite values with maximum > minimum")

    target_device = torch.device(device)
    if target_device.type == "cuda":
        _sync(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    individual_records: list[dict[str, Optional[float]]] = []
    predictive_mean_records: list[dict[str, Optional[float]]] = []
    standard_deviations_k: list[float] = []
    latencies: list[float] = []
    frame_count = 0

    for raw_frame in islice(dataset, limit):
        reference = _as_frame(raw_frame, name="dataset frame").to(target_device)
        observation = operator.observe(reference)
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, device=target_device)
        observation = observation.to(target_device)
        if operator.task == "inpaint":
            known = torch.as_tensor(operator.mask(reference), device=target_device, dtype=torch.bool)
            if known.shape != reference.shape:
                raise ValueError(f"operator.mask must match reference shape; got {tuple(known.shape)}")
        else:
            known = torch.zeros_like(reference, dtype=torch.bool)

        samples: list[torch.Tensor] = []
        sample_metrics: list[dict[str, Optional[float]]] = []
        for _ in range(draws):
            _sync(target_device)
            start = time.perf_counter()
            prediction = predict(observation)
            _sync(target_device)
            latencies.append(time.perf_counter() - start)
            prediction = _as_frame(prediction, name="prediction").to(target_device)
            if prediction.shape != reference.shape:
                raise ValueError(
                    f"prediction must match reference shape {tuple(reference.shape)}; "
                    f"got {tuple(prediction.shape)}"
                )
            prediction = prediction.detach()
            samples.append(prediction)

            # For observation consistency, compare the predicted field after
            # applying the same observation operator used for the input.
            predicted_observation = operator.observe(prediction)
            if not torch.is_tensor(predicted_observation):
                predicted_observation = torch.as_tensor(predicted_observation, device=target_device)
            predicted_observation = predicted_observation.to(target_device).detach()
            obs_metric = _frame_metrics(
                prediction, reference, observation, predicted_observation, known,
                task=operator.task, temperature_range=(low, high)
            )
            sample_metrics.append(obs_metric)

        individual_records.extend(sample_metrics)
        if draws > 1:
            sample_stack = torch.stack(samples, dim=0)
            predictive_mean = sample_stack.mean(dim=0)
            mean_observation = operator.observe(predictive_mean)
            if not torch.is_tensor(mean_observation):
                mean_observation = torch.as_tensor(mean_observation, device=target_device)
            mean_observation = mean_observation.to(target_device)
            predictive_mean_records.append(
                _frame_metrics(
                    predictive_mean, reference, observation, mean_observation, known,
                    task=operator.task, temperature_range=(low, high)
                )
            )
            standard_deviations_k.append(
                float(sample_stack.std(dim=0, unbiased=False)[~known].mean().item() * (high - low))
            )
        frame_count += 1

    if frame_count == 0:
        raise ValueError("dataset yielded no frames")

    return {
        "task": operator.task,
        "frames": frame_count,
        "draws_per_frame": draws,
        "aggregation": "mean_over_frames_and_draws",
        "temperature_range_k": [low, high],
        "single_draw_metrics": _summarize_metrics(individual_records),
        "predictive_mean_metrics": _summarize_metrics(predictive_mean_records) if draws > 1 else None,
        "mean_predictive_standard_deviation_k": _mean(standard_deviations_k) if draws > 1 else None,
        "evaluation_region": "missing_rectangle" if operator.task == "inpaint" else "full_field",
        "mean_inference_seconds_per_draw": _mean(latencies),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(target_device) / (1024 ** 2)
            if target_device.type == "cuda" else None
        ),
        "ssim_protocol": "uniform 7x7 local window; normalized data_range=1; windows fully inside evaluation region",
    }
