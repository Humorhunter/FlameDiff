"""Train and benchmark residual diffusion for single-frame 4x SR."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import torch
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
from spatial_eval import evaluate
from spatial_residual import (
    DEFAULT_RESIDUAL_SCALE,
    ResidualDenoiser,
    align_base,
    residual_diffusion_scheduler,
    residual_loss,
    sample_residual,
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


def _load_unet(path: Path, data: TemperatureData, temperature_range, device):
    checkpoint = load_checkpoint(path, "unet")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "sr":
        raise ValueError("The frozen U-Net checkpoint must be trained for SR")
    settings = SimpleNamespace(method="unet", task="sr")
    operator = SpatialObservation("sr")
    predictor = make_predictor(settings, operator, device, checkpoint)
    return checkpoint, predictor


def _base_from_observation(predictor, observation):
    with torch.no_grad():
        return align_base(predictor(observation), observation, scale=4)


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

    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    unet_checkpoint, unet_predictor = _load_unet(
        args.unet_checkpoint, data, temperature_range, device
    )
    operator = SpatialObservation("sr")
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
        "residual_scale": DEFAULT_RESIDUAL_SCALE,
    }
    model = ResidualDenoiser(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = residual_diffusion_scheduler(args.diffusion_steps)
    metadata = {
        "version": CHECKPOINT_VERSION,
        "stage": "residual",
        "task": "sr",
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for target in train_loader:
            target = target.to(device)
            observation = operator.observe(target)
            base = _base_from_observation(unet_predictor, observation)
            optimizer.zero_grad(set_to_none=True)
            loss = residual_loss(model, base, observation, target, scheduler)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite residual diffusion training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_total += loss.detach().item() * len(target)

        model.eval()
        val_total = 0.0
        # Reuse the same validation noise sequence each epoch so validation
        # loss changes reflect model updates rather than different noise.
        validation_generator = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            for target in val_loader:
                target = target.to(device)
                observation = operator.observe(target)
                base = _base_from_observation(unet_predictor, observation)
                loss = residual_loss(
                    model, base, observation, target, scheduler,
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


def benchmark(args):
    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    residual_checkpoint = load_checkpoint(args.checkpoint, "residual")
    check_provenance(residual_checkpoint, data, temperature_range)
    if residual_checkpoint.get("task") != "sr":
        raise ValueError("Residual checkpoint must be trained for SR")

    unet_path = args.unet_checkpoint
    if unet_path is None:
        unet_path = Path(residual_checkpoint["unet_checkpoint"])
    if _sha256(unet_path) != residual_checkpoint["unet_checkpoint_sha256"]:
        raise ValueError("Frozen U-Net checkpoint does not match residual training provenance")
    unet_checkpoint, unet_predictor = _load_unet(
        unet_path, data, temperature_range, device
    )
    if unet_checkpoint["epoch"] != residual_checkpoint["unet_epoch"]:
        raise ValueError("Frozen U-Net epoch does not match residual training provenance")

    operator = SpatialObservation("sr")
    dataset, samples = validation_panel(data, temperature_range)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        samples = samples[: len(dataset)]
    if not samples:
        raise ValueError("Validation panel is empty")

    model_config = residual_checkpoint["model_config"]
    model = ResidualDenoiser(**model_config).to(device)
    model.load_state_dict(residual_checkpoint["weights"])
    model.eval().requires_grad_(False)
    scheduler = residual_diffusion_scheduler(residual_checkpoint["diffusion_steps"])
    if args.steps > scheduler.config.num_train_timesteps:
        raise ValueError("--steps cannot exceed the checkpoint diffusion training length")

    def projected_unet(observation):
        return _base_from_observation(unet_predictor, observation)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    def residual_prediction(observation):
        base = _base_from_observation(unet_predictor, observation)
        return sample_residual(
            model,
            base,
            observation,
            scheduler,
            steps=args.steps,
            generator=generator,
        )

    base_result = evaluate(
        dataset,
        operator,
        projected_unet,
        device=device,
        temperature_range=temperature_range,
    )
    residual_result = evaluate(
        dataset,
        operator,
        residual_prediction,
        device=device,
        temperature_range=temperature_range,
        draws=args.draws,
    )
    base_rmse = base_result["single_draw_metrics"]["rmse_k"]
    residual_single_rmse = residual_result["single_draw_metrics"]["rmse_k"]
    predictive_mean = residual_result["predictive_mean_metrics"]
    report = {
        "method": "unet_base_plus_residual_diffusion",
        "stage": residual_checkpoint["stage"],
        "task": "sr",
        "split_name": "val",
        "split": data.split_manifest(),
        "validation_panel": samples,
        "temperature_range_k": list(temperature_range),
        "device": str(device),
        "seed": args.seed,
        "residual_checkpoint": str(args.checkpoint),
        "residual_checkpoint_sha256": _sha256(args.checkpoint),
        "residual_epoch": residual_checkpoint["epoch"],
        "residual_training_samples": residual_checkpoint["train_count"],
        "residual_validation_samples": residual_checkpoint["val_count"],
        "residual_model_config": residual_checkpoint["model_config"],
        "diffusion_training_steps": residual_checkpoint["diffusion_steps"],
        "unet_checkpoint": str(unet_path),
        "unet_checkpoint_sha256": residual_checkpoint["unet_checkpoint_sha256"],
        "unet_epoch": unet_checkpoint["epoch"],
        "unet_training_samples": unet_checkpoint["train_count"],
        "sampling": {"steps": args.steps, "draws_per_frame": args.draws},
        "projected_unet": base_result,
        "residual_diffusion": residual_result,
        "delta_rmse_k_vs_projected_unet": {
            "single_draw_mean": residual_single_rmse - base_rmse,
            "predictive_mean": (
                None if predictive_mean is None else predictive_mean["rmse_k"] - base_rmse
            ),
        },
        "observation_consistency_rmse_k": {
            "projected_unet": base_result["single_draw_metrics"]["observation_rmse_k"],
            "residual_single_draw_mean": residual_result["single_draw_metrics"]["observation_rmse_k"],
            "residual_predictive_mean": (
                None if predictive_mean is None else predictive_mean["observation_rmse_k"]
            ),
        },
    }
    save_json(args.output, report)
    summary = {
        "frames": len(samples),
        "draws_per_frame": args.draws,
        "projected_unet_rmse_k": base_rmse,
        "residual_single_draw_rmse_k": residual_single_rmse,
        "residual_predictive_mean_rmse_k": (
            None if predictive_mean is None else predictive_mean["rmse_k"]
        ),
        "delta_rmse_k_vs_projected_unet": report["delta_rmse_k_vs_projected_unet"],
        "observation_consistency_rmse_k": report["observation_consistency_rmse_k"],
        "saved": str(args.output),
    }
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train an SR residual diffusion model")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--unet-checkpoint", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=positive_int, default=20)
    training.add_argument("--batch-size", type=positive_int, default=4)
    training.add_argument("--width", type=positive_int, default=64)
    training.add_argument("--diffusion-steps", type=diffusion_steps, default=1000)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--train-limit", type=positive_int, help="Smoke runs only")
    training.add_argument("--val-limit", type=positive_int, help="Smoke runs only")
    training.add_argument("--save-epochs", nargs="+", type=positive_int, default=())
    training.add_argument("--seed", type=int, default=0)
    training.set_defaults(run=train)

    testing = commands.add_parser("benchmark", help="Evaluate on the fixed validation panel")
    testing.add_argument("--data", type=Path, required=True)
    testing.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--unet-checkpoint", type=Path)
    testing.add_argument("--output", type=Path, required=True)
    testing.add_argument("--steps", type=diffusion_steps, default=20)
    testing.add_argument("--draws", type=positive_int, default=4)
    testing.add_argument("--limit", type=positive_int, help="Smoke runs only")
    testing.add_argument("--seed", type=int, default=0)
    testing.set_defaults(run=benchmark)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
