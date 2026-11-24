"""Reconstruct a single-frame temperature field with a shared latent MMPS prior."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from model.ViTDecoder import ViTDecoder
from spatial import check_provenance, load_autoencoder, load_checkpoint, positive_int, save_json
from spatial_baselines import interpolate
from spatial_data import TemperatureData
from spatial_dps import FIELD_SHAPE
from spatial_eval import _gradient_metrics, _ssim, _sync
from spatial_mmps_observation import SpatialObservation
from spatial_mmps_sampler import sample_mmps
from spatial_sr_diagnostic import validation_panel


TASKS = ("sensors", "inpaint", "sr")
TASK_DEFAULTS = {
    "sensors": {
        "steps": 20,
        "maxiter": 8,
        "sigma_k": 100.0,
        "warm_strength": 0.0,
        "guidance_max_timestep": 500,
    },
    "inpaint": {
        "steps": 20,
        "maxiter": 16,
        "sigma_k": 10.0,
        "warm_strength": 0.0,
        "guidance_max_timestep": 700,
    },
    "sr": {
        "steps": 50,
        "maxiter": 16,
        "sigma_k": 10.0,
        "warm_strength": 0.1,
        "guidance_max_timestep": None,
    },
}
WARM_DEFAULT = {task: cfg["warm_strength"] for task, cfg in TASK_DEFAULTS.items()}
HOT_THRESHOLDS_K = (1000.0, 1200.0)
REQUIRED_METRIC_KEYS = frozenset({
    "mae_k",
    "rmse_k",
    "gradient_mae_k_per_grid_cell",
    "boundary_gradient_mae_k_per_grid_cell",
    "ssim",
    "observation_rmse_k",
    *(
        f"hot_{kind}_{int(threshold)}k{unit}"
        for threshold in HOT_THRESHOLDS_K
        for kind, unit in (("mae", "_k"), ("rmse", "_k"), ("iou", ""), ("dice", ""))
    ),
})


def build_scheduler(train_steps: int, *, sampler: str = "ddpm", clip_sample: bool = True):
    """Build an epsilon-prediction DDPM or DDIM scheduler with explicit clipping."""
    kwargs = dict(
        num_train_timesteps=train_steps,
        prediction_type="epsilon",
        clip_sample=clip_sample,
        timestep_spacing="linspace",
    )
    if sampler == "ddpm":
        return DDPMScheduler(**kwargs)
    if sampler == "ddim":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"sampler must be 'ddpm' or 'ddim', got {sampler!r}")


def parse_device(value: str) -> str:
    """Validate CLI --device values: cpu, cuda, or cuda:<non-negative int>."""
    if value in ("cpu", "cuda") or re.fullmatch(r"cuda:\d+", value):
        return value
    raise argparse.ArgumentTypeError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    save_json(pending, payload)
    pending.replace(path)


def hot_zone_metrics(
    prediction_k: torch.Tensor,
    target_k: torch.Tensor,
    region: torch.Tensor,
    thresholds_k: tuple[float, ...] = HOT_THRESHOLDS_K,
) -> dict[str, float | None]:
    """Compute hot-zone MAE, RMSE, IoU, and Dice over ``region`` in Kelvin."""
    result: dict[str, float | None] = {}
    for threshold in thresholds_k:
        suffix = f"{int(threshold)}k"
        target_hot = (target_k >= threshold) & region
        pred_hot = (prediction_k >= threshold) & region
        if bool(target_hot.any().item()):
            diff = (prediction_k - target_k)[target_hot]
            result[f"hot_mae_{suffix}_k"] = float(diff.abs().mean().item())
            result[f"hot_rmse_{suffix}_k"] = float(diff.square().mean().sqrt().item())
        else:
            result[f"hot_mae_{suffix}_k"] = None
            result[f"hot_rmse_{suffix}_k"] = None

        intersection = int((pred_hot & target_hot).sum().item())
        union = int((pred_hot | target_hot).sum().item())
        denom = int(pred_hot.sum().item()) + int(target_hot.sum().item())
        result[f"hot_iou_{suffix}"] = float(intersection / union) if union > 0 else None
        result[f"hot_dice_{suffix}"] = float(2.0 * intersection / denom) if denom > 0 else None
    return result


def aggregate_metric_rows(metric_rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Average each metric key across frames, skipping ``None`` entries."""
    if not metric_rows:
        return {}
    keys: list[str] = []
    seen: set[str] = set()
    for row in metric_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    summary: dict[str, float | None] = {}
    for key in keys:
        values = [row[key] for row in metric_rows if row.get(key) is not None]
        summary[key] = float(mean(values)) if values else None
    return summary


def _row_key(sample: dict, seed: int) -> tuple[str, str, int, int]:
    return (str(sample["case"]), str(sample["slice"]), int(sample["frame"]), int(seed))


def _load_resume_rows(
    path: Path,
    *,
    expected_settings: dict,
    checkpoint_sha256: str,
    epoch: int,
    target_keys: set[tuple[str, str, int, int]] | None = None,
) -> dict[tuple[str, str, int, int], dict]:
    """Load compatible completed per-frame rows from an existing evaluation JSON."""
    if not path.exists():
        return {}
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"Resume file {path} does not contain a JSON object")

    check_keys = (
        "task",
        "temperature_range_k",
        "measured_count",
        "steps",
        "maxiter",
        "sigma_k",
        "warm_strength",
        "seed",
        "sampler",
        "ddim_eta",
        "clip_sample",
        "solver_space",
        "guidance_max_timestep",
        "score_clip",
    )
    for key in check_keys:
        if existing.get(key) != expected_settings.get(key):
            raise ValueError(
                f"Cannot resume {path}: setting mismatch for {key!r} "
                f"(existing={existing.get(key)!r}, current={expected_settings.get(key)!r})"
            )
    if existing.get("epoch") != epoch:
        raise ValueError(
            f"Cannot resume {path}: checkpoint epoch mismatch "
            f"(existing={existing.get('epoch')!r}, current={epoch!r})"
        )
    existing_sha = existing.get("checkpoint_sha256")
    if existing_sha != checkpoint_sha256:
        raise ValueError(
            f"Cannot resume {path}: missing or mismatched checkpoint SHA-256"
        )

    resumed: dict[tuple[str, str, int, int], dict] = {}
    for row in existing.get("per_frame", []):
        for section in ("metrics", "raw_metrics"):
            section_data = row.get(section)
            if not isinstance(section_data, dict) or not REQUIRED_METRIC_KEYS.issubset(section_data):
                raise ValueError(
                    f"Cannot resume {path}: row for {row.get('case')!r} has an incomplete "
                    f"{section!r} schema; start a fresh output file instead"
                )
        key = _row_key(row, row["seed"])
        resumed[key] = row

    if target_keys is not None:
        extra_keys = set(resumed) - target_keys
        if extra_keys:
            raise ValueError(
                f"Cannot resume {path}: existing file contains {len(extra_keys)} completed "
                f"frame(s) outside the requested selection (refusing to truncate)"
            )
    return resumed


class TemperatureReconstructor:
    """Keep the autoencoder and unconditional prior frozen across observations."""

    def __init__(self, args, checkpoint):
        model_config = checkpoint["model_config"]
        if model_config.get("num_feats") != 0:
            raise ValueError("Three-task MMPS requires a prior trained without sensor conditioning")
        self.low, self.high = map(float, checkpoint["temperature_range"])
        self.span = self.high - self.low
        if not all(map(isfinite, (self.low, self.high, self.span))) or self.span <= 0:
            raise ValueError("Checkpoint temperature bounds must be finite and increasing")

        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Select --device cpu or use a CUDA instance.")
            if self.device.index is not None and self.device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {self.device.index} is out of range "
                    f"(found {torch.cuda.device_count()} device(s))"
                )
        self.operator = SpatialObservation(args.task)
        self.prior = ViTDecoder(**model_config).to(self.device).eval().requires_grad_(False)
        self.prior.load_state_dict(checkpoint["weights"])
        self.autoencoder = load_autoencoder(
            checkpoint["ae_weights"], self.device, ae_config=checkpoint.get("ae_config")
        )
        self.diffusion_steps = int(checkpoint["diffusion_steps"])
        defaults = TASK_DEFAULTS[args.task]
        self.steps = defaults["steps"] if args.steps is None else args.steps
        if not 2 <= self.steps <= self.diffusion_steps:
            raise ValueError("--steps must be between 2 and the checkpoint's diffusion step count")
        self.maxiter = defaults["maxiter"] if args.maxiter is None else args.maxiter
        self.seed = args.seed
        self.sigma_k = defaults["sigma_k"] if args.sigma_k is None else args.sigma_k
        if not isfinite(self.sigma_k) or self.sigma_k <= 0:
            raise ValueError("--sigma-k must be finite and strictly positive")
        self.strength = defaults["warm_strength"] if args.warm_strength is None else args.warm_strength
        if not isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("--warm-strength must be between 0 and 1")
        if args.task == "sensors" and self.strength:
            raise ValueError("Eight-sensor reconstruction starts from noise. Use --warm-strength 0.")

        self.sampler = getattr(args, "sampler", "ddpm")
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError("--sampler must be 'ddpm' or 'ddim'")
        self.ddim_eta = float(getattr(args, "ddim_eta", 0.0)) if self.sampler == "ddim" else None
        if self.ddim_eta is not None and (not isfinite(self.ddim_eta) or not 0.0 <= self.ddim_eta <= 1.0):
            raise ValueError("--ddim-eta must be finite and in [0, 1]")
        self.clip_sample = not bool(getattr(args, "no_clip_sample", False))
        self.solver_space = getattr(args, "solver_space", "auto")
        if self.solver_space not in ("auto", "observation", "latent"):
            raise ValueError("--solver-space must be one of: auto, observation, latent")
        raw_gmax = getattr(args, "guidance_max_timestep", None)
        if raw_gmax is None:
            self.guidance_max_timestep = defaults["guidance_max_timestep"]
        elif raw_gmax == -1:
            self.guidance_max_timestep = None
        else:
            self.guidance_max_timestep = raw_gmax
        if self.guidance_max_timestep is not None:
            if (
                not isinstance(self.guidance_max_timestep, int)
                or isinstance(self.guidance_max_timestep, bool)
                or not 0 <= self.guidance_max_timestep < self.diffusion_steps
            ):
                raise ValueError("--guidance-max-timestep must be -1 (disabled) or in [0, diffusion_steps - 1]")
        self.score_clip = getattr(args, "score_clip", None)
        if self.score_clip is not None:
            self.score_clip = float(self.score_clip)
            if not isfinite(self.score_clip) or self.score_clip <= 0:
                raise ValueError("--score-clip must be finite and strictly positive")

    def settings(self):
        result = {
            "task": self.operator.task,
            "device": str(self.device),
            "temperature_range_k": [self.low, self.high],
            "measured_count": self.operator.measured_count,
            "steps": self.steps,
            "maxiter": self.maxiter,
            "sigma_k": self.sigma_k,
            "warm_strength": self.strength,
            "seed": self.seed,
            "sampler": self.sampler,
            "ddim_eta": self.ddim_eta,
            "clip_sample": self.clip_sample,
            "solver_space": self.solver_space,
            "guidance_max_timestep": self.guidance_max_timestep,
            "score_clip": self.score_clip,
            "latent_channels": getattr(self.autoencoder, "latent_channels", 1),
            "ae_output_activation": getattr(self.autoencoder, "output_activation", "relu"),
        }
        if self.operator.task == "sensors":
            result["sensor_coordinates_xy"] = self.operator.sensor_coords_xy
        return result

    def reconstruct(self, observation, *, seed):
        """Accept one observation in Kelvin and return field, raw field and diagnostics."""
        self.operator.validate_observation(observation, finite=True)
        if observation.shape[0] != 1:
            raise ValueError("MMPS reconstructs one temperature field at a time")
        normalized = (observation - self.low) / self.span
        if self.operator.task == "inpaint":
            normalized = self.operator.measure(normalized)

        measure = self.operator.measure
        measured = normalized
        if self.operator.task == "inpaint":
            # Restrict the Krylov system to measured pixels. The file format
            # remains a full-sized masked image for harmonic initialization.
            indices = self.operator.mask(normalized).flatten().nonzero().flatten()

            def measure(field):
                return field.flatten(1).index_select(1, indices)

            measured = measure(normalized)

        initial = {}
        if self.strength:
            method = "harmonic" if self.operator.task == "inpaint" else "bicubic"
            with torch.no_grad():
                base = interpolate(normalized, self.operator, method)
                base = self.operator.project(base, normalized)
                initial = {
                    "initial_latent": self.autoencoder.encoder(base),
                    "strength": self.strength,
                }

        diagnostics = []
        scheduler = build_scheduler(
            self.diffusion_steps,
            sampler=self.sampler,
            clip_sample=self.clip_sample,
        )
        raw_normalized = sample_mmps(
            self.prior, self.autoencoder.decoder, measure, measured,
            scheduler=scheduler, steps=self.steps, sigma=self.sigma_k / self.span,
            maxiter=self.maxiter, generator=torch.Generator(device=self.device).manual_seed(seed),
            latent_channels=getattr(self.autoencoder, "latent_channels", None),
            solver_space=self.solver_space,
            guidance_max_timestep=self.guidance_max_timestep,
            score_clip=self.score_clip,
            eta=self.ddim_eta,
            diagnostics=diagnostics.append, **initial,
        )
        raw = raw_normalized * self.span + self.low
        projected = self.operator.project(raw, observation)
        self.operator.validate_field(projected, finite=True)
        steps = [record for record in diagnostics if record["phase"] == "step"]
        guided_steps = [step for step in steps if step.get("guided", True)]
        residuals = [step["relative_residual"] for step in guided_steps]
        clean_oob = [
            step["clean_out_of_unit_box_fraction"]
            for step in steps
            if "clean_out_of_unit_box_fraction" in step
        ]
        posterior_clipped = [
            step["posterior_clipped_fraction"]
            for step in steps
            if "posterior_clipped_fraction" in step
        ]
        score_ratios = [
            step["score_to_noise_norm_ratio"]
            for step in guided_steps
            if "score_to_noise_norm_ratio" in step
        ]
        resolved_spaces = sorted({
            step["solver_space"] for step in guided_steps if "solver_space" in step
        })
        return projected, raw, {
            "observation_rmse_k_before_projection": self.observation_rmse(raw, observation),
            "observation_rmse_k_after_projection": self.observation_rmse(projected, observation),
            "total_diffusion_steps": len(steps),
            "solver_steps": len(guided_steps),
            "solver_converged_steps": sum(bool(step["converged"]) for step in guided_steps),
            "solver_space_used": resolved_spaces[0] if len(resolved_spaces) == 1 else resolved_spaces,
            "solver_median_relative_residual": median(residuals) if residuals else None,
            "solver_max_relative_residual": max(residuals) if residuals else None,
            "max_clean_out_of_unit_box_fraction": max(clean_oob) if clean_oob else None,
            "max_posterior_clipped_fraction": max(posterior_clipped) if posterior_clipped else None,
            "max_score_to_noise_norm_ratio": max(score_ratios) if score_ratios else None,
        }

    def observation_rmse(self, field, observation):
        residual = self.operator.measure(field) - observation
        if self.operator.task == "inpaint":
            residual = residual[self.operator.mask(field)]
        return float(residual.square().mean().sqrt().item())

    def metrics(self, prediction, target, observation):
        if self.operator.task == "sr":
            region = torch.ones_like(target, dtype=torch.bool)
            known = None
        else:
            known_mask = self.operator.mask(target)
            region = ~known_mask
            known = known_mask if self.operator.task == "inpaint" else None

        residual = (prediction - target)[region]
        gradient_error, boundary_gradient_error = _gradient_metrics(
            prediction, target, region, known
        )
        normalized = (prediction - self.low) / self.span
        reference = (target - self.low) / self.span
        result = {
            "mae_k": float(residual.abs().mean().item()),
            "rmse_k": float(residual.square().mean().sqrt().item()),
            "gradient_mae_k_per_grid_cell": gradient_error,
            "boundary_gradient_mae_k_per_grid_cell": boundary_gradient_error,
            "ssim": _ssim(normalized, reference, region),
            "observation_rmse_k": self.observation_rmse(prediction, observation),
        }
        result.update(hot_zone_metrics(prediction, target, region))
        return result


def build_evaluation_report(
    *,
    settings: dict,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    epoch: int,
    task: str,
    target_frames: int,
    rows: list[dict],
    status: str,
    peak_cuda_memory_mb: float | None = None,
    panel_description: str = "middle slice and validation frame of each selected case",
) -> dict:
    """Assemble the evaluation JSON payload for incremental or final saves."""
    summary_metrics = aggregate_metric_rows([row["metrics"] for row in rows])
    summary_raw_metrics = aggregate_metric_rows([row["raw_metrics"] for row in rows])
    residuals = [
        row["solver_median_relative_residual"]
        for row in rows
        if row.get("solver_median_relative_residual") is not None
    ]
    clean_oob = [
        row["max_clean_out_of_unit_box_fraction"]
        for row in rows
        if row.get("max_clean_out_of_unit_box_fraction") is not None
    ]
    posterior_clipped = [
        row["max_posterior_clipped_fraction"]
        for row in rows
        if row.get("max_posterior_clipped_fraction") is not None
    ]
    score_ratios = [
        row["max_score_to_noise_norm_ratio"]
        for row in rows
        if row.get("max_score_to_noise_norm_ratio") is not None
    ]
    return {
        **settings,
        "status": status,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": epoch,
        "split_name": "val",
        "panel": panel_description,
        "evaluation_region": {
            "sensors": "unmeasured_pixels",
            "inpaint": "missing_rectangle",
            "sr": "full_field",
        }[task],
        "target_frames": target_frames,
        "frames": len(rows),
        "mean_rmse_k": summary_metrics.get("rmse_k"),
        "mean_raw_rmse_k": summary_raw_metrics.get("rmse_k"),
        "mean_seconds_per_frame": float(mean(row["seconds"] for row in rows)) if rows else None,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "summary_metrics": summary_metrics,
        "summary_raw_metrics": summary_raw_metrics,
        "summary_observation": {
            "mean_observation_rmse_k_before_projection": (
                float(mean(row["observation_rmse_k_before_projection"] for row in rows))
                if rows else None
            ),
            "mean_observation_rmse_k_after_projection": (
                float(mean(row["observation_rmse_k_after_projection"] for row in rows))
                if rows else None
            ),
            "total_solver_steps": sum(int(row["solver_steps"]) for row in rows),
            "total_solver_converged_steps": sum(int(row["solver_converged_steps"]) for row in rows),
            "median_solver_relative_residual": float(median(residuals)) if residuals else None,
            "max_clean_out_of_unit_box_fraction": float(max(clean_oob)) if clean_oob else None,
            "max_posterior_clipped_fraction": (
                float(max(posterior_clipped)) if posterior_clipped else None
            ),
            "max_score_to_noise_norm_ratio": float(max(score_ratios)) if score_ratios else None,
        },
        "per_frame": rows,
    }


def reconstruct(args):
    checkpoint = load_checkpoint(args.checkpoint, "prior")
    model = TemperatureReconstructor(args, checkpoint)
    if args.output.resolve() == args.observation.resolve():
        raise ValueError("--output must differ from the observation path")
    value = np.load(args.observation, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise ValueError("Expected a real numeric .npy observation in Kelvin")
    expected_shape = model.operator.shape
    file_shape = expected_shape[1:] if args.task != "sensors" else expected_shape
    if value.shape != file_shape:
        raise ValueError(f"{args.task} expects a Kelvin observation of shape {file_shape}")
    observation = torch.as_tensor(value.astype(np.float32), device=model.device).reshape(1, *expected_shape)
    field, _, diagnostics = model.reconstruct(observation, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        np.save(destination, field[0, 0].cpu().numpy())
    result = {
        **model.settings(), **diagnostics, "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"], "field_shape": list(FIELD_SHAPE[1:]), "saved": str(args.output),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


def evaluate(args):
    data = TemperatureData(args.data)
    checkpoint = load_checkpoint(args.checkpoint, "prior")
    temperature_range = data.training_range()
    check_provenance(checkpoint, data, temperature_range)
    model = TemperatureReconstructor(args, checkpoint)
    settings = model.settings()
    checkpoint_sha256 = _file_sha256(args.checkpoint)
    dataset, samples = validation_panel(data, temperature_range)
    panel_mode = str(getattr(args, "panel", "9"))
    if panel_mode == "45":
        selected_indices = list(range(len(samples)))[:args.limit]
        panel_description = "all five slices middle validation frame of each case"
    else:
        by_case = {}
        for index, sample in enumerate(samples):
            by_case.setdefault(sample["case"], []).append(index)
        selected_indices = [group[len(group) // 2] for group in by_case.values()][:args.limit]
        panel_description = "middle slice and validation frame of each selected case"
    target_keys = {_row_key(samples[index], args.seed + index) for index in selected_indices}
    resumed_rows = (
        _load_resume_rows(
            args.output,
            expected_settings=settings,
            checkpoint_sha256=checkpoint_sha256,
            epoch=checkpoint["epoch"],
            target_keys=target_keys,
        )
        if getattr(args, "resume", False)
        else {}
    )
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)

    def _peak_cuda_mb() -> float | None:
        if model.device.type != "cuda":
            return None
        return float(torch.cuda.max_memory_allocated(model.device) / (1024.0 * 1024.0))

    rows = []
    for index in selected_indices:
        sample = samples[index]
        seed = args.seed + index
        key = _row_key(sample, seed)
        if key in resumed_rows:
            row = resumed_rows[key]
            rows.append(row)
            print(
                f"{row['case']}: resumed (RMSE {row['metrics']['rmse_k']:.2f} K, "
                f"raw observation RMSE {row['observation_rmse_k_before_projection']:.2f} K)",
                flush=True,
            )
            continue

        # The reference is used only to create synthetic observations and score the output.
        target = dataset[index].unsqueeze(0).to(model.device) * model.span + model.low
        observation = model.operator.measure(target)
        _sync(model.device)
        started = perf_counter()
        prediction, raw, diagnostics = model.reconstruct(observation, seed=seed)
        _sync(model.device)
        elapsed = perf_counter() - started
        row = {
            **sample, "seed": seed, "seconds": elapsed, **diagnostics,
            "metrics": model.metrics(prediction, target, observation),
            "raw_metrics": model.metrics(raw, target, observation),
        }
        rows.append(row)
        print(f"{row['case']}: RMSE {row['metrics']['rmse_k']:.2f} K, "
              f"raw observation RMSE {row['observation_rmse_k_before_projection']:.2f} K", flush=True)
        partial = build_evaluation_report(
            settings=settings,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            epoch=checkpoint["epoch"],
            task=args.task,
            target_frames=len(selected_indices),
            rows=rows,
            status="running",
            peak_cuda_memory_mb=_peak_cuda_mb(),
            panel_description=panel_description,
        )
        _write_json_atomic(args.output, partial)

    result = build_evaluation_report(
        settings=settings,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        epoch=checkpoint["epoch"],
        task=args.task,
        target_frames=len(selected_indices),
        rows=rows,
        status="complete",
        peak_cuda_memory_mb=_peak_cuda_mb(),
        panel_description=panel_description,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"},
                     indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("reconstruct", help="Reconstruct one field from a Kelvin .npy observation")
    inference.add_argument("--observation", type=Path, required=True)
    inference.set_defaults(run=reconstruct)
    evaluation = commands.add_parser("evaluate", help="Measure accuracy on validation frames")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--panel", choices=("9", "45"), default="9", help="Validation panel size (9=1/case, 45=all slices)")
    evaluation.add_argument("--limit", type=positive_int, help="Maximum number of frames to evaluate")
    evaluation.add_argument("--resume", action="store_true", help="Resume from an existing compatible output JSON")
    evaluation.set_defaults(run=evaluate)

    for command in (inference, evaluation):
        command.add_argument("--task", choices=TASKS, required=True)
        command.add_argument("--checkpoint", type=Path, required=True, help="Unconditional spatial prior")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", type=parse_device, default="cpu", help="cpu, cuda, or cuda:<index>")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--steps", type=positive_int, help="Diffusion steps (defaults: 20 for sensors/inpaint, 50 for sr)")
        command.add_argument("--sigma-k", type=float, help="MMPS likelihood regularization scale in Kelvin (defaults: 100 for sensors, 10 for inpaint/sr)")
        command.add_argument("--maxiter", type=positive_int, help="GMRES iterations per step (defaults: 8 for sensors, 16 for inpaint/sr)")
        command.add_argument("--warm-strength", type=float, help="Fraction of the schedule to use (defaults: 0 for sensors/inpaint, 0.1 for sr)")
        command.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm", help="Reverse diffusion scheduler")
        command.add_argument("--ddim-eta", type=float, default=0.0, help="Stochasticity eta for --sampler ddim (0=deterministic)")
        command.add_argument("--no-clip-sample", action="store_true", help="Disable scheduler clip_sample to [-1, 1]")
        command.add_argument("--solver-space", choices=("auto", "observation", "latent"), default="auto", help="GMRES linear system space")
        command.add_argument("--guidance-max-timestep", type=int, help="Skip MMPS guidance above this timestep (-1 disables default cap; defaults: 500 for sensors, 700 for inpaint, none for sr)")
        command.add_argument("--score-clip", type=float, help="Cap ||sqrt(1-alpha_bar)*score|| / ||predicted_noise||")

    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must differ from the checkpoint path")
    args.run(args)


if __name__ == "__main__":
    main()
