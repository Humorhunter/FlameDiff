"""Reconstruct a single-frame temperature field with a shared latent MMPS prior."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch

from model.ViTDecoder import ViTDecoder
from spatial import check_provenance, load_autoencoder, load_checkpoint, positive_int, save_json
from spatial_baselines import interpolate
from spatial_data import TemperatureData
from spatial_dps import diffusion_scheduler, FIELD_SHAPE
from spatial_eval import _gradient_metrics, _ssim, _sync
from spatial_mmps_observation import SpatialObservation
from spatial_mmps_sampler import sample_mmps
from spatial_sr_diagnostic import validation_panel


TASKS = ("sensors", "inpaint", "sr")
WARM_DEFAULT = {"sensors": 0.0, "inpaint": 0.25, "sr": 0.1}
HOT_THRESHOLDS_K = (1000.0, 1200.0)


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
    summary: dict[str, float | None] = {}
    for key in metric_rows[0]:
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
    if existing_sha is not None and existing_sha != checkpoint_sha256:
        raise ValueError(f"Cannot resume {path}: checkpoint SHA-256 mismatch")

    resumed: dict[tuple[str, str, int, int], dict] = {}
    for row in existing.get("per_frame", []):
        key = _row_key(row, row["seed"])
        resumed[key] = row
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
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Select --device cpu or use a CUDA instance.")
        self.operator = SpatialObservation(args.task)
        self.prior = ViTDecoder(**model_config).to(self.device).eval().requires_grad_(False)
        self.prior.load_state_dict(checkpoint["weights"])
        self.autoencoder = load_autoencoder(checkpoint["ae_weights"], self.device)
        self.scheduler = diffusion_scheduler(checkpoint["diffusion_steps"])
        self.steps = args.steps
        if not 2 <= self.steps <= self.scheduler.config.num_train_timesteps:
            raise ValueError("--steps must be between 2 and the checkpoint's diffusion step count")
        self.maxiter = args.maxiter
        self.seed = args.seed
        self.sigma_k = args.sigma_k
        if self.sigma_k is None:
            self.sigma_k = 0.05 * self.span if args.task == "sensors" else 10.0
        if not isfinite(self.sigma_k) or self.sigma_k <= 0:
            raise ValueError("--sigma-k must be finite and strictly positive")
        self.strength = WARM_DEFAULT[args.task] if args.warm_strength is None else args.warm_strength
        if not isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("--warm-strength must be between 0 and 1")
        if args.task == "sensors" and self.strength:
            raise ValueError("Eight-sensor reconstruction starts from noise. Use --warm-strength 0.")

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
        raw_normalized = sample_mmps(
            self.prior, self.autoencoder.decoder, measure, measured,
            scheduler=self.scheduler, steps=self.steps, sigma=self.sigma_k / self.span,
            maxiter=self.maxiter, generator=torch.Generator(device=self.device).manual_seed(seed),
            diagnostics=diagnostics.append, **initial,
        )
        raw = raw_normalized * self.span + self.low
        projected = self.operator.project(raw, observation)
        self.operator.validate_field(projected, finite=True)
        steps = [record for record in diagnostics if record["phase"] == "step"]
        residuals = [step["relative_residual"] for step in steps]
        return projected, raw, {
            "observation_rmse_k_before_projection": self.observation_rmse(raw, observation),
            "observation_rmse_k_after_projection": self.observation_rmse(projected, observation),
            "solver_steps": len(steps),
            "solver_converged_steps": sum(bool(step["converged"]) for step in steps),
            "solver_median_relative_residual": median(residuals) if residuals else None,
            "solver_max_relative_residual": max(residuals) if residuals else None,
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
) -> dict:
    """Assemble the evaluation JSON payload for incremental or final saves."""
    summary_metrics = aggregate_metric_rows([row["metrics"] for row in rows])
    summary_raw_metrics = aggregate_metric_rows([row["raw_metrics"] for row in rows])
    residuals = [
        row["solver_median_relative_residual"]
        for row in rows
        if row.get("solver_median_relative_residual") is not None
    ]
    return {
        **settings,
        "status": status,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": epoch,
        "split_name": "val",
        "panel": "middle slice and validation frame of each selected case",
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
    by_case = {}
    for index, sample in enumerate(samples):
        by_case.setdefault(sample["case"], []).append(index)
    selected_indices = [group[len(group) // 2] for group in by_case.values()][:args.limit]
    resumed_rows = (
        _load_resume_rows(
            args.output,
            expected_settings=settings,
            checkpoint_sha256=checkpoint_sha256,
            epoch=checkpoint["epoch"],
        )
        if getattr(args, "resume", False)
        else {}
    )

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
    evaluation = commands.add_parser("evaluate", help="Measure accuracy on one validation frame per case")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--limit", type=positive_int, help="Maximum number of cases to evaluate")
    evaluation.add_argument("--resume", action="store_true", help="Resume from an existing compatible output JSON")
    evaluation.set_defaults(run=evaluate)

    for command in (inference, evaluation):
        command.add_argument("--task", choices=TASKS, required=True)
        command.add_argument("--checkpoint", type=Path, required=True, help="Unconditional spatial prior")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--steps", type=positive_int, default=20)
        command.add_argument("--sigma-k", type=float, help="MMPS likelihood regularization scale in Kelvin")
        command.add_argument("--maxiter", type=positive_int, default=8, help="GMRES iterations per step")
        command.add_argument("--warm-strength", type=float, help="Fraction of the schedule to use. Set 0 for noise start.")

    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must differ from the checkpoint path")
    args.run(args)


if __name__ == "__main__":
    main()
