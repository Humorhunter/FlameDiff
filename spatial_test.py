"""Run the frozen spatial comparison on all held-out test frames."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from torch.utils.data import Subset

from spatial import check_provenance, load_checkpoint, make_predictor, prepare, save_json
from spatial_baselines import interpolate
from spatial_data import TemperatureData
from spatial_dps import SpatialObservation, diffusion_scheduler, warm_timesteps
from spatial_sr_diagnostic import align_block_means, compare_prediction


STEPS = 20
SEED = 0
PROTOCOL = {
    "sr": {"interpolation": "bicubic", "dps_guidance": 20.0, "warm_strength": 0.1},
    "inpaint": {"interpolation": "harmonic", "dps_guidance": 1.0, "warm_strength": 0.25},
}


def test_case_subsets(data, temperature_range):
    manifest = data.split_manifest()
    start, stop = manifest["split_ranges_half_open"]["test"]
    frames_per_file = stop - start
    dataset = data.dataset("test", temperature_range)
    sources = manifest["files"]
    if frames_per_file <= 0 or not sources or len(sources) * frames_per_file != len(dataset):
        raise ValueError("Test dataset does not match the split manifest")
    groups = {}
    for file_index, source in enumerate(sources):
        file_start = file_index * frames_per_file
        groups.setdefault(source["case"], []).extend(range(file_start, file_start + frames_per_file))
    return {case: Subset(dataset, indices) for case, indices in groups.items()}


def aggregate(per_case):
    """Weight case means by the number of frames to preserve the evaluator's definition."""
    first = next(iter(per_case.values()))
    result = {}
    for variant, metrics in first["metrics"].items():
        result[variant] = {}
        for name in metrics:
            values = [
                (entry["frames"], entry["metrics"][variant][name])
                for entry in per_case.values()
                if entry["metrics"][variant][name] is not None
            ]
            result[variant][name] = (
                sum(count * value for count, value in values) / sum(count for count, _ in values)
                if values else None
            )
    return result


def identify_checkpoint(path, checkpoint):
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "epoch": checkpoint["epoch"]}


def write_report(path, report):
    pending = path.with_name(path.name + ".tmp")
    save_json(pending, report)
    pending.replace(path)


def run(args):
    device = prepare(args.device, SEED)
    operator = SpatialObservation(args.task)
    protocol = PROTOCOL[args.task]
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    prior = load_checkpoint(args.prior_checkpoint, "prior")
    unet = load_checkpoint(args.unet_checkpoint, "unet")
    for checkpoint in (prior, unet):
        check_provenance(checkpoint, data, temperature_range)
    train_count = len(data.dataset("train"))
    if any(count != train_count for count in (
        prior["train_count"], prior["ae_train_count"], unet["train_count"],
    )):
        raise ValueError("The formal test requires models trained on the full training split")

    groups = test_case_subsets(data, temperature_range)
    frame_count = sum(map(len, groups.values()))
    interpolation = protocol["interpolation"]

    def interpolated(observation):
        return interpolate(observation, operator, interpolation)

    def initial_field(observation):
        prediction = interpolated(observation)
        if args.task == "sr":
            return align_block_means(prediction, observation, operator)
        return prediction

    pure_settings = SimpleNamespace(method="dps", steps=STEPS, guidance=protocol["dps_guidance"], seed=SEED)
    warm_settings = SimpleNamespace(method="dps", steps=STEPS, guidance=1.0, seed=SEED)
    unet_settings = SimpleNamespace(method="unet", task=args.task)
    strength = protocol["warm_strength"]
    predictors = {
        interpolation: interpolated,
        "unet": make_predictor(unet_settings, operator, device, unet),
        "dps": make_predictor(pure_settings, operator, device, prior),
        "warm-dps": make_predictor(warm_settings, operator, device, prior,
                                   initializer=initial_field, strength=strength),
    }
    timesteps = warm_timesteps(diffusion_scheduler(prior["diffusion_steps"]), STEPS, strength)
    report = {
        "protocol_version": 1,
        "status": "running",
        "task": args.task,
        "split": "test",
        "frames": frame_count,
        "frames_per_case": {case: len(subset) for case, subset in groups.items()},
        "split_manifest": data.split_manifest(),
        "temperature_range_k": list(temperature_range),
        "aggregation": "mean of per-frame metrics, weighted by case frame count",
        "device": str(device),
        "checkpoints": {
            "prior": identify_checkpoint(args.prior_checkpoint, prior),
            "unet": identify_checkpoint(args.unet_checkpoint, unet),
        },
        "sampling": {
            "seed": SEED,
            "draws_per_frame": 1,
            "dps": {"steps": STEPS, "guidance": pure_settings.guidance},
            "warm-dps": {"timesteps": timesteps, "guidance": warm_settings.guidance,
                         "strength": strength, "initializer": interpolation},
        },
        "metrics": {},
        "cases": {},
    }
    if args.task == "sr":
        report["sampling"]["warm-dps"]["initializer"] = "bicubic-corrected"

    print(f"Independent test: {frame_count} frames across {len(groups)} cases", flush=True)
    for method, predict in predictors.items():
        per_case = {}
        report["cases"][method] = per_case
        for case, subset in groups.items():
            print(f"Evaluating {method} on {case} ({len(subset)} frames)...", flush=True)
            begin = perf_counter()
            metrics = compare_prediction(
                subset, operator, predict, device=device, temperature_range=temperature_range
            )
            per_case[case] = {
                "frames": len(subset), "metrics": metrics,
                "benchmark_wall_seconds": perf_counter() - begin,
            }
            write_report(args.output, report)
        report["metrics"][method] = aggregate(per_case)
        write_report(args.output, report)
    report["status"] = "complete"
    write_report(args.output, report)
    return report


def print_summary(report):
    sr = report["task"] == "sr"
    extra = "obs RMSE (K)" if sr else "boundary MAE"
    extra_key = "observation_rmse_k" if sr else "boundary_gradient_mae_k_per_grid_cell"
    print(f"{'method':<22} {'MAE (K)':>9} {'RMSE (K)':>10} {'SSIM':>8} {extra:>14} {'grad MAE':>10}")
    for method, variants in report["metrics"].items():
        for variant, metrics in variants.items():
            label = f"{method}-{variant}"
            print(f"{label:<22} {metrics['mae_k']:>9.3f} {metrics['rmse_k']:>10.3f} "
                  f"{metrics['ssim']:>8.4f} {metrics[extra_key]:>14.6f} "
                  f"{metrics['gradient_mae_k_per_grid_cell']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(PROTOCOL), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--unet-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; use a new filename to preserve the earlier report")
    print_summary(run(args))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
