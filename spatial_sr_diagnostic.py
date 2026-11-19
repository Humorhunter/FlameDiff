"""Compare spatial reconstruction methods on the same validation panel."""

import argparse
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.utils.data import Subset

from spatial import (
    check_provenance, load_autoencoder, load_checkpoint, make_predictor, positive_int, prepare,
    save_json,
)
from spatial_baselines import interpolate
from spatial_data import TemperatureData
from spatial_dps import SpatialObservation, diffusion_scheduler, warm_timesteps
from spatial_eval import evaluate


def validation_panel(data, temperature_range):
    manifest = data.split_manifest()
    start, stop = manifest["split_ranges_half_open"]["val"]
    frame = (start + stop) // 2
    sources = manifest["files"]
    dataset = data.dataset("val", temperature_range)
    if len(dataset) != len(sources) * (stop - start):
        raise ValueError("Validation dataset does not match the split manifest")
    indices = [
        file_index * (stop - start) + frame - start
        for file_index in range(len(sources))
    ]
    return Subset(dataset, indices), [
        {"case": source["case"], "slice": source["slice"], "path": source["path"], "frame": frame}
        for source in sources
    ]


def align_block_means(prediction, observation, operator):
    discrepancy = observation - operator.observe(prediction)
    return prediction + discrepancy.repeat_interleave(operator.scale, dim=-2).repeat_interleave(
        operator.scale, dim=-1
    )


def compare_prediction(dataset, operator, predictor, *, device, temperature_range):
    kwargs = {"device": device, "temperature_range": temperature_range}
    if operator.task == "inpaint":
        result = evaluate(dataset, operator, predictor, **kwargs)
        return {"original": result["single_draw_metrics"]}

    cached = []

    def record_prediction(observation):
        prediction = predictor(observation)
        cached.append(prediction.detach().cpu())
        return prediction

    def correct_prediction(observation):
        prediction = next(recorded).to(device)
        return align_block_means(prediction, observation, operator)

    original = evaluate(dataset, operator, record_prediction, **kwargs)
    recorded = iter(cached)
    corrected = evaluate(dataset, operator, correct_prediction, **kwargs)
    return {
        "original": original["single_draw_metrics"],
        "corrected": corrected["single_draw_metrics"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sr", "inpaint"), default="sr")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unet-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=positive_int, default=20)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--warm-strengths", type=float, nargs="+", metavar="RATIO",
                        help="Try warm-start DPS with these fractions of the full step grid")
    parser.add_argument("--warm-guidance", type=float, default=1.0)
    parser.add_argument("--mmps-sigma-k", type=float, metavar="K",
                        help="Add local-linear MMPS for SR with this likelihood noise scale")
    parser.add_argument("--mmps-maxiter", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warm_strengths and args.guidance is not None:
        parser.error("--guidance controls pure-noise DPS; use --warm-guidance with --warm-strengths")
    if args.mmps_sigma_k is not None:
        if args.task != "sr":
            parser.error("The first MMPS diagnostic supports --task sr only")
        if not isfinite(args.mmps_sigma_k) or args.mmps_sigma_k <= 0:
            parser.error("--mmps-sigma-k must be finite and positive")

    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    checkpoint = load_checkpoint(args.checkpoint, "prior")
    check_provenance(checkpoint, data, temperature_range)
    dataset, samples = validation_panel(data, temperature_range)
    operator = SpatialObservation(args.task)
    guidance = args.guidance if args.guidance is not None else (20.0 if args.task == "sr" else 1.0)
    settings = SimpleNamespace(method="dps", seed=args.seed, steps=args.steps, guidance=guidance)
    interpolation = "bicubic" if args.task == "sr" else "harmonic"

    def interpolated(observation):
        return interpolate(observation, operator, interpolation)

    def initial_field(observation):
        prediction = interpolated(observation)
        if args.task == "sr":
            return align_block_means(prediction, observation, operator)
        return prediction

    predictors = {interpolation: interpolated}
    schedule = {}
    if args.warm_strengths:
        scheduler = diffusion_scheduler(checkpoint["diffusion_steps"])
        autoencoder = load_autoencoder(checkpoint["ae_weights"], device)
        settings.guidance = args.warm_guidance

        @torch.no_grad()
        def ae_roundtrip(observation):
            prediction = autoencoder(initial_field(observation))
            return operator.restore_known(prediction, observation)

        predictors["ae-initialized"] = ae_roundtrip
        for strength in dict.fromkeys(args.warm_strengths):
            timesteps = warm_timesteps(scheduler, args.steps, strength)
            name = f"warm-dps-{strength}"
            schedule[name] = {"strength": strength, "timesteps": timesteps}
            predictors[name] = make_predictor(
                settings, operator, device, checkpoint, initializer=initial_field, strength=strength
            )
    else:
        predictors["dps"] = make_predictor(settings, operator, device, checkpoint)

    mmps_records = {}
    if args.mmps_sigma_k is not None:
        sigma = args.mmps_sigma_k / (temperature_range[1] - temperature_range[0])
        strengths = list(dict.fromkeys(args.warm_strengths)) if args.warm_strengths else [None]
        for strength in strengths:
            name = "mmps" if strength is None else f"warm-mmps-{strength}"
            records = mmps_records[name] = []
            mmps_settings = SimpleNamespace(
                method="mmps", seed=args.seed, steps=args.steps, mmps_sigma=sigma,
                mmps_maxiter=args.mmps_maxiter, mmps_diagnostics=records.append,
            )
            predictors[name] = make_predictor(
                mmps_settings, operator, device, checkpoint,
                initializer=initial_field if strength is not None else None, strength=strength,
            )
            if strength is not None:
                schedule[name] = schedule[f"warm-dps-{strength}"]

    if args.unet_checkpoint:
        unet = load_checkpoint(args.unet_checkpoint, "unet")
        check_provenance(unet, data, temperature_range)
        unet_settings = SimpleNamespace(method="unet", task=args.task)
        predictors["unet"] = make_predictor(unet_settings, operator, device, unet)
    cases = len({sample["case"] for sample in samples})
    print(f"Validation panel: {len(samples)} frames from {cases} cases", flush=True)
    results = {}
    wall_seconds = {}
    for method, predictor in predictors.items():
        print(f"Evaluating {method} (one prediction per frame)...", flush=True)
        start = perf_counter()
        results[method] = compare_prediction(
            dataset, operator, predictor, device=device, temperature_range=temperature_range
        )
        wall_seconds[method] = perf_counter() - start
    report = {
        "split": "val",
        "task": args.task,
        "samples": samples,
        "temperature_range_k": list(temperature_range),
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "sampling": {"steps": args.steps, "guidance": settings.guidance, "seed": args.seed},
        "metrics": results,
        "benchmark_wall_seconds": wall_seconds,
    }
    if mmps_records:
        report["mmps"] = {
            "observation_sigma_k": args.mmps_sigma_k,
            "observation_sigma_normalized": sigma,
            "maxiter": args.mmps_maxiter,
            "solver": mmps_records,
        }
    if schedule:
        report["warm_start"] = {
            "initializer": "bicubic-corrected" if args.task == "sr" else "harmonic-original",
            "schedules": schedule,
        }
    if args.unet_checkpoint:
        report["unet_checkpoint"] = str(args.unet_checkpoint)
        report["unet_epoch"] = unet["epoch"]
    save_json(args.output, report)

    if args.task == "sr":
        extra = "obs RMSE (K)"
        extra_key = "observation_rmse_k"
    else:
        extra = "boundary MAE"
        extra_key = "boundary_gradient_mae_k_per_grid_cell"
    print(f"{'method':<26} {'RMSE (K)':>10} {'SSIM':>9} {extra:>14} {'grad MAE':>11}")
    for method, variants in results.items():
        for variant, metrics in variants.items():
            name = f"{method}-{variant}"
            print(f"{name:<26} {metrics['rmse_k']:>10.3f} {metrics['ssim']:>9.4f} "
                  f"{metrics[extra_key]:>14.6f} {metrics['gradient_mae_k_per_grid_cell']:>11.3f}")
    for method, records in mmps_records.items():
        residuals = [record["relative_residual"] for record in records]
        print(f"{method}: GMRES relative residual median={median(residuals):.4g}, "
              f"max={max(residuals):.4g}; benchmark time={wall_seconds[method]:.1f}s")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
