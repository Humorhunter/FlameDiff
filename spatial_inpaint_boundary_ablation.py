"""Compare residual-amplitude and boundary-taper ablations for inpainting."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Subset

from spatial import check_provenance, load_checkpoint, prepare, save_json
from spatial_dps import SpatialObservation
from spatial_eval import _frame_metrics
from spatial_inpaint import InpaintResidualDenoiser, inpaint_diffusion_scheduler, sample_inpaint_residual
from spatial_inpaint_experiment import (
    _distance_layers,
    _frame_metadata,
    _load_unet,
    _region_metrics,
    _sha256,
    _validation_data,
)
from spatial_data import TemperatureData
from spatial_sr_diagnostic import validation_panel


def _distance_from_boundary(known_mask: torch.Tensor) -> torch.Tensor:
    """Return each masked pixel's distance in grid cells from the hole edge."""
    missing = ~known_mask[0, 0]
    height, width = missing.shape
    rows = torch.arange(height, device=known_mask.device)[:, None]
    columns = torch.arange(width, device=known_mask.device)[None, :]
    distance = torch.minimum(
        torch.minimum(rows - 48 + 1, 144 - rows),
        torch.minimum(columns - 64 + 1, 192 - columns),
    )
    return distance.clamp_min(0).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def _residual_weights(distance: torch.Tensor, missing: torch.Tensor, width: int | None):
    if width is None:
        return missing.to(dtype=distance.dtype)
    progress = ((distance - 1.0) / (width - 1)).clamp(0.0, 1.0)
    weight = 0.5 * (1.0 + torch.cos(torch.pi * progress))
    return weight * (distance < width) * missing


def _summary(rows):
    if not rows:
        return None
    keys = [
        key for key, value in rows[0].items()
        if key not in {"case", "slice", "path", "frame", "distance_layers"}
        and isinstance(value, (int, float))
    ]
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def _group_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {name: _summary(values) for name, values in sorted(groups.items())}


def _distance_summary(rows):
    if not rows:
        return {}
    names = rows[0]["distance_layers"].keys()
    return {
        name: _summary([
            {"case": row["case"], **row["distance_layers"][name]}
            for row in rows
        ])
        for name in names
    }


def _evaluation_data(data, temperature_range, panel, limit):
    if panel in {"45", "full"}:
        return _validation_data(data, temperature_range, panel, limit)
    split = "test" if panel == "test" else "val"
    dataset = data.dataset(split, temperature_range)
    samples = _frame_metadata(data, split)
    excluded = []
    if panel == "remaining":
        _, excluded = validation_panel(data, temperature_range)
    excluded_keys = {(row["path"], row["frame"]) for row in excluded}
    indices = [
        index for index, row in enumerate(samples)
        if (row["path"], row["frame"]) not in excluded_keys
    ][:limit]
    return Subset(dataset, indices), [samples[index] for index in indices]


@torch.no_grad()
def run(args):
    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    checkpoint = load_checkpoint(args.checkpoint, "inpaint_residual")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "inpaint":
        raise ValueError("Residual checkpoint must be trained for inpainting")

    unet_path = args.unet_checkpoint or Path(checkpoint["unet_checkpoint"])
    if _sha256(unet_path) != checkpoint["unet_checkpoint_sha256"]:
        raise ValueError("Frozen U-Net checkpoint does not match residual training provenance")
    unet_checkpoint, unet_predictor = _load_unet(
        unet_path, data, temperature_range, device
    )
    if unet_checkpoint["epoch"] != checkpoint["unet_epoch"]:
        raise ValueError("Frozen U-Net epoch does not match residual training provenance")

    dataset, samples = _evaluation_data(
        data, temperature_range, args.panel, args.limit
    )
    if not samples:
        raise ValueError("Validation dataset is empty")
    model = InpaintResidualDenoiser(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval().requires_grad_(False)
    scheduler = inpaint_diffusion_scheduler(checkpoint["diffusion_steps"])
    if args.steps > scheduler.config.num_train_timesteps:
        raise ValueError("--steps cannot exceed the checkpoint diffusion training length")

    operator = SpatialObservation("inpaint")
    method_names = (
        "unet",
        "raw",
        "uniform_alpha_0_25",
        "uniform_alpha_0_5",
        "taper_8",
        "taper_16",
        "taper_32",
    )
    if args.panel in {"remaining", "test"}:
        method_names = ("unet", "uniform_alpha_0_25", "taper_8")
    single_draw_rows = {name: [] for name in method_names}
    mean_rows = {name: [] for name in method_names}
    generator = torch.Generator(device=device).manual_seed(args.seed)
    known_samples_exact = {name: True for name in method_names}
    mean_known_error = {name: 0.0 for name in method_names}

    for index, (raw_frame, sample_info) in enumerate(zip(dataset, samples)):
        target = raw_frame.unsqueeze(0).to(device)
        observation = operator.observe(target)
        known_mask = operator.mask(target).to(device=device, dtype=torch.bool)
        missing_mask = ~known_mask
        distance = _distance_from_boundary(known_mask)
        bands = _distance_layers(*target.shape[-2:], known_mask, device)

        base = unet_predictor(observation)
        predictions = {name: [] for name in method_names}
        predictions["unet"].append(base)
        single_draw_rows["unet"].append(
            {**sample_info, **_measure(base, target, observation, known_mask, bands, temperature_range)}
        )

        for _ in range(args.draws):
            sampled = sample_inpaint_residual(
                model,
                base,
                observation,
                known_mask,
                scheduler,
                steps=args.steps,
                generator=generator,
            )
            residual = (sampled - base) * missing_mask
            candidates = {
                "raw": residual,
                "uniform_alpha_0_25": residual * 0.25,
                "uniform_alpha_0_5": residual * 0.5,
            }
            for width in (8, 16, 32):
                candidates[f"taper_{width}"] = residual * _residual_weights(
                    distance, missing_mask, width
                )

            for name in method_names[1:]:
                correction = candidates[name]
                prediction = torch.where(known_mask, observation, base + correction)
                predictions[name].append(prediction)
                known_samples_exact[name] = (
                    known_samples_exact[name]
                    and torch.equal(prediction[known_mask], observation[known_mask])
                )
                single_draw_rows[name].append({
                    **sample_info,
                    **_measure(prediction, target, observation, known_mask, bands, temperature_range),
                })

        for name, draws in predictions.items():
            if name == "unet":
                mean_prediction = draws[0]
            else:
                mean_prediction = torch.stack(draws).mean(dim=0)
            max_known_error = float(
                (mean_prediction[known_mask] - observation[known_mask]).abs().max().item()
            )
            mean_known_error[name] = max(mean_known_error[name], max_known_error)
            mean_rows[name].append({
                **sample_info,
                **_measure(mean_prediction, target, observation, known_mask, bands, temperature_range),
            })

        if (index + 1) % 10 == 0:
            print(f"Evaluated {index + 1}/{len(samples)} frames", flush=True)

    methods = {}
    for name in method_names:
        methods[name] = {
            "single_draw": {
                "metrics": _summary(single_draw_rows[name]),
                "per_case": _group_summary(single_draw_rows[name], "case"),
                "distance_layers": _distance_summary(single_draw_rows[name]),
                "per_frame_draws": single_draw_rows[name],
            },
            "predictive_mean": {
                "metrics": _summary(mean_rows[name]),
                "per_case": _group_summary(mean_rows[name], "case"),
                "distance_layers": _distance_summary(mean_rows[name]),
                "per_frame": mean_rows[name],
            },
            "individual_samples_preserve_known_pixels_exactly": known_samples_exact[name],
            "predictive_mean_max_known_pixel_abs_error_normalized": mean_known_error[name],
        }

    report = {
        "method": "inpaint_residual_boundary_ablation",
        "task": "inpaint",
        "split_name": "test" if args.panel == "test" else "val",
        "validation_panel": args.panel,
        "split": data.split_manifest(),
        "samples": samples,
        "temperature_range_k": list(temperature_range),
        "device": str(device),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "variant": checkpoint["variant"],
        "sampling": {"steps": args.steps, "draws_per_frame": args.draws},
        "taper": "cosine weight equals 1 at the hole edge and decays to 0 at the stated inward width",
        "methods": methods,
    }
    save_json(args.output, report)
    print(json.dumps({
        name: {
            "single_draw_rmse_k": methods[name]["single_draw"]["metrics"]["rmse_k"],
            "predictive_mean_rmse_k": methods[name]["predictive_mean"]["metrics"]["rmse_k"],
            "single_draw_boundary_gradient_mae_k_per_grid_cell": methods[name]["single_draw"]["metrics"]["boundary_gradient_mae_k_per_grid_cell"],
            "predictive_mean_boundary_gradient_mae_k_per_grid_cell": methods[name]["predictive_mean"]["metrics"]["boundary_gradient_mae_k_per_grid_cell"],
        }
        for name in method_names
    } | {"saved": str(args.output)}, indent=2, allow_nan=False), flush=True)


def _measure(prediction, target, observation, known_mask, bands, temperature_range):
    metrics = _frame_metrics(
        prediction,
        target,
        observation,
        SpatialObservation("inpaint").observe(prediction),
        known_mask,
        task="inpaint",
        temperature_range=temperature_range,
    )
    metrics["distance_layers"] = {
        name: _region_metrics(prediction[0], target[0], mask, temperature_range)
        for name, mask in bands.items()
    }
    return metrics


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def diffusion_steps(value):
    steps = positive_int(value)
    if steps < 2:
        raise argparse.ArgumentTypeError("Diffusion requires at least two steps")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unet-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--panel", choices=("45", "full", "remaining", "test"), default="45")
    parser.add_argument("--steps", type=diffusion_steps, default=50)
    parser.add_argument("--draws", type=positive_int, default=4)
    parser.add_argument("--limit", type=positive_int, help="Smoke runs only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
