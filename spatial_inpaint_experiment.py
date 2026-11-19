"""Train and benchmark a frozen U-Net plus masked residual diffusion for repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset

from spatial import (
    CHECKPOINT_VERSION,
    check_provenance,
    load_checkpoint,
    make_predictor,
    positive_int,
    prepare,
    save_checkpoint,
    save_json,
    subset,
)
from spatial_data import TemperatureData
from spatial_dps import SpatialObservation
from spatial_eval import _frame_metrics
from spatial_inpaint import (
    InpaintResidualDenoiser,
    inpaint_diffusion_scheduler,
    masked_residual_loss,
    sample_inpaint_residual,
)
from spatial_sr_diagnostic import validation_panel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diffusion_steps(value):
    steps = positive_int(value)
    if steps < 2:
        raise argparse.ArgumentTypeError("Diffusion requires at least two steps")
    return steps


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer")
    return number


def _load_unet(path: Path, data: TemperatureData, temperature_range, device):
    checkpoint = load_checkpoint(path, "unet")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "inpaint":
        raise ValueError("The frozen U-Net checkpoint must be trained for inpainting")
    settings = SimpleNamespace(method="unet", task="inpaint")
    operator = SpatialObservation("inpaint")
    predictor = make_predictor(settings, operator, device, checkpoint)
    return checkpoint, predictor


def _residual_checkpoint(args, metadata, model, epoch, validation_loss):
    return {
        **metadata,
        "epoch": epoch,
        "val_loss": validation_loss,
        "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def train(args):
    if args.save_epochs and max(args.save_epochs) > args.epochs:
        raise ValueError("--save-epochs cannot contain an epoch beyond --epochs")
    if not isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if args.attention_blocks > 2:
        raise ValueError("--attention-blocks must be 0, 1, or 2")
    if args.variant == "conv" and args.attention_blocks != 0:
        raise ValueError("The conv variant requires --attention-blocks 0")
    if args.variant == "transformer" and args.attention_blocks not in (1, 2):
        raise ValueError("The transformer variant requires --attention-blocks 1 or 2")

    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    unet_checkpoint, unet_predictor = _load_unet(
        args.unet_checkpoint, data, temperature_range, device
    )
    operator = SpatialObservation("inpaint")
    train_data = subset(data.dataset("train", temperature_range), args.train_limit)
    val_data = subset(data.dataset("val", temperature_range), args.val_limit)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model_config = {
        "width": args.width,
        "residual_scale": args.residual_scale,
        "attention_blocks": args.attention_blocks,
    }
    model = InpaintResidualDenoiser(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = inpaint_diffusion_scheduler(args.diffusion_steps)
    metadata = {
        "version": CHECKPOINT_VERSION,
        "stage": "inpaint_residual",
        "task": "inpaint",
        "variant": args.variant,
        "split": data.split_manifest(),
        "temperature_range": list(temperature_range),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "seed": args.seed,
        "model_config": model_config,
        "diffusion_steps": args.diffusion_steps,
        "unet_checkpoint": str(args.unet_checkpoint),
        "unet_checkpoint_sha256": _sha256(args.unet_checkpoint),
        "unet_epoch": unet_checkpoint["epoch"],
        "unet_train_count": unet_checkpoint["train_count"],
    }

    best = float("inf")
    save_epochs = set(args.save_epochs)
    train_noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for target in train_loader:
            target = target.to(device)
            observation = operator.observe(target)
            known_mask = operator.mask(target)
            with torch.no_grad():
                base = unet_predictor(observation)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_residual_loss(
                model, base, observation, known_mask, target, scheduler,
                generator=train_noise_generator,
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite masked residual diffusion training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_total += loss.detach().item() * len(target)

        model.eval()
        val_total = 0.0
        validation_generator = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            for target in val_loader:
                target = target.to(device)
                observation = operator.observe(target)
                known_mask = operator.mask(target)
                base = unet_predictor(observation)
                loss = masked_residual_loss(
                    model,
                    base,
                    observation,
                    known_mask,
                    target,
                    scheduler,
                    generator=validation_generator,
                )
                val_total += loss.item() * len(target)

        validation_loss = val_total / len(val_data)
        if not isfinite(validation_loss):
            raise ValueError("Non-finite validation loss")
        print(
            f"epoch {epoch}: train={train_total / len(train_data):.6g} "
            f"val={validation_loss:.6g}",
            flush=True,
        )
        is_best = validation_loss < best
        if is_best or epoch in save_epochs:
            checkpoint = _residual_checkpoint(
                args, metadata, model, epoch, validation_loss
            )
            if is_best:
                best = validation_loss
                save_checkpoint(args.output, checkpoint)
            if epoch in save_epochs:
                path = args.output.with_name(
                    f"{args.output.stem}-epoch-{epoch:04d}{args.output.suffix}"
                )
                save_checkpoint(path, checkpoint)
                print(f"Saved scheduled checkpoint: {path}", flush=True)


def _frame_metadata(data, split):
    manifest = data.split_manifest()
    frame_start, frame_stop = manifest["split_ranges_half_open"][split]
    records = []
    for source in manifest["files"]:
        for frame in range(frame_start, frame_stop):
            records.append({
                "case": source["case"],
                "slice": source["slice"],
                "path": source["path"],
                "frame": frame,
            })
    return records


def _validation_data(data, temperature_range, panel, limit):
    if panel == "45":
        dataset, samples = validation_panel(data, temperature_range)
        if limit is not None:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
            samples = samples[:len(dataset)]
        return dataset, samples
    dataset = data.dataset("val", temperature_range)
    samples = _frame_metadata(data, "val")
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
        samples = samples[:len(dataset)]
    return dataset, samples


def _known_mask(operator, reference):
    return operator.mask(reference).to(device=reference.device, dtype=torch.bool)


def _distance_layers(height, width, known_mask, device):
    missing = ~known_mask[0, 0]
    rows = torch.arange(height, device=device)[:, None]
    columns = torch.arange(width, device=device)[None, :]
    distance = torch.minimum(
        torch.minimum(rows - 48 + 1, 144 - rows),
        torch.minimum(columns - 64 + 1, 192 - columns),
    )
    distance = distance.masked_fill(~missing, 0)
    definitions = (("edge_1_8", 1, 8), ("edge_9_16", 9, 16),
                   ("interior_17_32", 17, 32), ("interior_gt_32", 33, height + width))
    return {
        name: missing & (distance >= low) & (distance <= high)
        for name, low, high in definitions
    }


def _region_metrics(prediction, target, mask, temperature_range):
    delta_kelvin = temperature_range[1] - temperature_range[0]
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(0)
    difference = (prediction - target)[mask] * delta_kelvin
    return {
        "mae_k": difference.abs().mean().item(),
        "rmse_k": difference.square().mean().sqrt().item(),
        "pixels": int(mask.sum().item()),
    }


def _summary(records):
    if not records:
        return None
    keys = [key for key in records[0] if key != "case" and isinstance(records[0][key], (int, float))]
    return {key: sum(float(record[key]) for record in records) / len(records) for key in keys}


def _group_summary(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return {name: _summary(items) for name, items in sorted(groups.items())}


@torch.no_grad()
def benchmark(args):
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

    if args.split == "val":
        dataset, samples = _validation_data(data, temperature_range, args.panel, args.limit)
    else:
        dataset = data.dataset("test", temperature_range)
        samples = _frame_metadata(data, "test")
        if args.limit is not None:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
            samples = samples[:len(dataset)]
    if not samples:
        raise ValueError("Evaluation dataset is empty")

    model = InpaintResidualDenoiser(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval().requires_grad_(False)
    scheduler = inpaint_diffusion_scheduler(checkpoint["diffusion_steps"])
    if args.steps > scheduler.config.num_train_timesteps:
        raise ValueError("--steps cannot exceed the checkpoint diffusion training length")

    operator = SpatialObservation("inpaint")
    layers = None
    methods = {"unet": [], "residual_diffusion": []}
    per_frame = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    elapsed = defaultdict(list)
    individual_exact_preservation = True
    predictive_mean_max_observation_error = 0.0
    for index, (raw_frame, sample_info) in enumerate(zip(dataset, samples)):
        target = raw_frame.unsqueeze(0).to(device)
        observation = operator.observe(target)
        known_mask = _known_mask(operator, target)
        if device.type == "cuda":
            unet_start = torch.cuda.Event(enable_timing=True)
            unet_end = torch.cuda.Event(enable_timing=True)
            unet_start.record()
        else:
            import time
            unet_start_time = time.perf_counter()
        base = unet_predictor(observation)
        if device.type == "cuda":
            unet_end.record()
            torch.cuda.synchronize(device)
            elapsed["unet"].append(unet_start.elapsed_time(unet_end) / 1000.0)
        else:
            elapsed["unet"].append(time.perf_counter() - unet_start_time)
        if layers is None:
            layers = _distance_layers(*target.shape[-2:], known_mask, device)

        def record(method, prediction):
            nonlocal individual_exact_preservation
            if method == "residual_diffusion":
                individual_exact_preservation = (
                    individual_exact_preservation
                    and torch.equal(prediction[known_mask], observation[known_mask])
                )
            predicted_observation = operator.observe(prediction)
            metrics = _frame_metrics(
                prediction,
                target,
                observation,
                predicted_observation,
                known_mask,
                task="inpaint",
                temperature_range=temperature_range,
            )
            row = {**sample_info, **metrics}
            row["distance_layers"] = {
                name: _region_metrics(prediction[0], target[0], layer, temperature_range)
                for name, layer in layers.items()
            }
            methods[method].append(row)
            return row

        record("unet", base)

        individual_predictions = []
        for _ in range(args.draws):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                import time
                start_time = time.perf_counter()
            prediction = sample_inpaint_residual(
                model,
                base,
                observation,
                known_mask,
                scheduler,
                steps=args.steps,
                generator=generator,
            )
            if device.type == "cuda":
                end.record()
                torch.cuda.synchronize(device)
                elapsed["residual_diffusion"].append(start.elapsed_time(end) / 1000.0)
            else:
                elapsed["residual_diffusion"].append(time.perf_counter() - start_time)
            individual_predictions.append(prediction)
            record("residual_diffusion", prediction)

        if args.draws > 1:
            predictive_mean = torch.stack(individual_predictions).mean(dim=0)
            known_error = (
                predictive_mean[known_mask] - observation[known_mask]
            ).abs().max().item()
            predictive_mean_max_observation_error = max(
                predictive_mean_max_observation_error, known_error
            )
            predicted_observation = operator.observe(predictive_mean)
            metrics = _frame_metrics(
                predictive_mean,
                target,
                observation,
                predicted_observation,
                known_mask,
                task="inpaint",
                temperature_range=temperature_range,
            )
            row = {**sample_info, **metrics}
            row["distance_layers"] = {
                name: _region_metrics(predictive_mean[0], target[0], layer, temperature_range)
                for name, layer in layers.items()
            }
            per_frame.append(row)

        if (index + 1) % 10 == 0:
            print(f"Evaluated {index + 1}/{len(samples)} frames", flush=True)

    report = {
        "method": "frozen_unet_plus_masked_residual_diffusion",
        "task": "inpaint",
        "split_name": args.split,
        "validation_panel": args.panel if args.split == "val" else None,
        "split": data.split_manifest(),
        "samples": samples,
        "temperature_range_k": list(temperature_range),
        "device": str(device),
        "seed": args.seed,
        "residual_checkpoint": str(args.checkpoint),
        "residual_epoch": checkpoint["epoch"],
        "variant": checkpoint["variant"],
        "model_config": checkpoint["model_config"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "unet_checkpoint": str(unet_path),
        "unet_epoch": unet_checkpoint["epoch"],
        "unet_train_count": unet_checkpoint["train_count"],
        "sampling": {"steps": args.steps, "draws_per_frame": args.draws},
        "unet": {
            "metrics": _summary(methods["unet"]),
            "per_case": _group_summary(methods["unet"], "case"),
            "distance_layers": _distance_summary(methods["unet"]),
            "per_frame": methods["unet"],
            "mean_seconds_per_frame": sum(elapsed["unet"]) / len(elapsed["unet"]),
        },
        "residual_single_draw": {
            "metrics": _summary(methods["residual_diffusion"]),
            "per_case": _group_summary(methods["residual_diffusion"], "case"),
            "distance_layers": _distance_summary(methods["residual_diffusion"]),
            "per_frame_draws": methods["residual_diffusion"],
            "mean_seconds_per_draw": sum(elapsed["residual_diffusion"]) / len(elapsed["residual_diffusion"]),
        },
        "residual_predictive_mean": {
            "metrics": _summary(per_frame),
            "per_case": _group_summary(per_frame, "case"),
            "distance_layers": _distance_summary(per_frame),
            "per_frame": per_frame,
        } if args.draws > 1 else None,
        "individual_samples_preserve_known_pixels_exactly": individual_exact_preservation,
        "predictive_mean_max_known_pixel_abs_error_normalized": predictive_mean_max_observation_error,
    }
    save_json(args.output, report)
    print(json.dumps({
        "frames": len(samples),
        "variant": checkpoint["variant"],
        "parameter_count": report["parameter_count"],
        "unet_rmse_k": report["unet"]["metrics"]["rmse_k"],
        "residual_single_draw_rmse_k": report["residual_single_draw"]["metrics"]["rmse_k"],
        "residual_predictive_mean_rmse_k": (
            None if report["residual_predictive_mean"] is None
            else report["residual_predictive_mean"]["metrics"]["rmse_k"]
        ),
        "saved": str(args.output),
    }, indent=2, allow_nan=False), flush=True)


def _distance_summary(records):
    if not records:
        return {}
    names = records[0]["distance_layers"].keys()
    return {
        name: _summary([
            {"case": record["case"], **record["distance_layers"][name]}
            for record in records
        ])
        for name in names
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train masked inpaint residual diffusion")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--unet-checkpoint", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--variant", choices=("conv", "transformer"), required=True)
    training.add_argument("--attention-blocks", type=nonnegative_int, default=0)
    training.add_argument("--epochs", type=positive_int, default=20)
    training.add_argument("--batch-size", type=positive_int, default=4)
    training.add_argument("--width", type=positive_int, default=64)
    training.add_argument("--residual-scale", type=float, default=16.0)
    training.add_argument("--diffusion-steps", type=diffusion_steps, default=1000)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--train-limit", type=positive_int, help="Smoke runs only")
    training.add_argument("--val-limit", type=positive_int, help="Smoke runs only")
    training.add_argument("--save-epochs", nargs="+", type=positive_int, default=())
    training.add_argument("--seed", type=int, default=0)
    training.set_defaults(run=train)

    testing = commands.add_parser("benchmark", help="Compare the U-Net and residual diffusion")
    testing.add_argument("--data", type=Path, required=True)
    testing.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--unet-checkpoint", type=Path)
    testing.add_argument("--output", type=Path, required=True)
    testing.add_argument("--split", choices=("val", "test"), default="val")
    testing.add_argument("--panel", choices=("45", "full"), default="45")
    testing.add_argument("--steps", type=diffusion_steps, default=50)
    testing.add_argument("--draws", type=positive_int, default=4)
    testing.add_argument("--limit", type=positive_int, help="Smoke runs only")
    testing.add_argument("--seed", type=int, default=0)
    testing.set_defaults(run=benchmark)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
