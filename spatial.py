"""Training and evaluation entry points for the single-frame spatial benchmark."""

import argparse
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Subset

from model.EnDeCoder import AutoEnDecoder
from model.ViTDecoder import ViTDecoder
from spatial_baselines import SpatialUNet, interpolate
from spatial_data import TemperatureData
from spatial_dps import SpatialObservation, diffusion_loss, diffusion_scheduler, sample_dps
from spatial_eval import evaluate


CHECKPOINT_VERSION = 1
METHODS = {
    "inpaint": ("harmonic", "unet", "dps", "unguided", "ae"),
    "sr": ("bilinear", "bicubic", "unet", "dps", "unguided", "ae"),
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def load_checkpoint(path, stage):
    if path is None:
        raise ValueError(f"A spatial {stage} checkpoint is required")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("version") != CHECKPOINT_VERSION
            or checkpoint.get("stage") != stage):
        raise ValueError(f"{path} is not a version {CHECKPOINT_VERSION} spatial {stage} checkpoint")
    return checkpoint


def check_provenance(checkpoint, data, temperature_range):
    if checkpoint["split"] != data.split_manifest():
        raise ValueError("Checkpoint split does not match the current data manifest")
    if tuple(checkpoint["temperature_range"]) != temperature_range:
        raise ValueError("Checkpoint temperature range does not match the training split")


def save_checkpoint(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, pending)
    pending.replace(path)


def save_json(path, result):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def prepare(device_name, seed):
    torch.manual_seed(seed)
    return torch.device(device_name)


def load_autoencoder(weights, device, ae_config=None):
    if ae_config is None:
        inferred_channels = 1
        if isinstance(weights, dict) and "encoder.proj2.bias" in weights:
            inferred_channels = int(weights["encoder.proj2.bias"].shape[0])
        ae_config = {"latent_channels": inferred_channels, "output_activation": "relu"}
    model = AutoEnDecoder(
        latent_channels=int(ae_config.get("latent_channels", 1)),
        output_activation=str(ae_config.get("output_activation", "relu")),
    )
    model.load_state_dict(weights)
    return model.to(device).eval().requires_grad_(False)


def _gradient_l1_loss(prediction, target):
    dy = functional.l1_loss(
        prediction[..., 1:, :] - prediction[..., :-1, :],
        target[..., 1:, :] - target[..., :-1, :],
    )
    dx = functional.l1_loss(
        prediction[..., :, 1:] - prediction[..., :, :-1],
        target[..., :, 1:] - target[..., :, :-1],
    )
    return 0.5 * (dy + dx)


def subset(data, limit):
    return data if limit is None else Subset(data, range(min(limit, len(data))))


def train(args):
    checkpoint_epochs = set(args.save_epochs)
    if checkpoint_epochs and max(checkpoint_epochs) > args.epochs:
        raise ValueError("--save-epochs cannot contain an epoch beyond --epochs")
    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    train_data = subset(data.dataset("train", temperature_range), args.train_limit)
    val_data = subset(data.dataset("val", temperature_range), args.val_limit)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.workers
    )
    metadata = {
        "version": CHECKPOINT_VERSION,
        "stage": args.stage,
        "split": data.split_manifest(),
        "temperature_range": list(temperature_range),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "seed": args.seed,
    }

    if args.stage == "ae":
        ae_config = {
            "latent_channels": args.latent_channels,
            "output_activation": args.ae_output_activation,
        }
        grad_weight = float(args.ae_grad_weight)
        if not isfinite(grad_weight) or grad_weight < 0:
            raise ValueError("--ae-grad-weight must be finite and non-negative")
        model = AutoEnDecoder(**ae_config).to(device)
        torch.nn.init.zeros_(model.decoder.conv8.weight)
        torch.nn.init.constant_(model.decoder.conv8.bias, 0.5)
        metadata.update(ae_config=ae_config, ae_grad_weight=grad_weight)

        def loss(target, generator=None):
            prediction = model(target)
            value = functional.mse_loss(prediction, target)
            if grad_weight > 0.0:
                value = value + grad_weight * _gradient_l1_loss(prediction, target)
            return value

    elif args.stage == "prior":
        if args.width % 8:
            raise ValueError("Denoiser width must be a multiple of eight")
        pretrained = load_checkpoint(args.ae_checkpoint, "ae")
        check_provenance(pretrained, data, temperature_range)
        ae_config = pretrained.get("ae_config")
        if ae_config is None:
            inferred_channels = int(pretrained["weights"]["encoder.proj2.bias"].shape[0])
            ae_config = {"latent_channels": inferred_channels, "output_activation": "relu"}
        autoencoder = load_autoencoder(pretrained["weights"], device, ae_config=ae_config)
        model_config = {"d_model": args.width, "num_layers": args.layers, "num_feats": 0}
        if autoencoder.latent_channels != 1:
            model_config["in_channels"] = autoencoder.latent_channels
        model = ViTDecoder(**model_config).to(device)
        scheduler = diffusion_scheduler(args.diffusion_steps)
        metadata.update(
            model_config=model_config,
            diffusion_steps=args.diffusion_steps,
            ae_config=ae_config,
            ae_weights=pretrained["weights"],
            ae_train_count=pretrained["train_count"],
        )

        def loss(target, generator=None):
            return diffusion_loss(model, autoencoder, scheduler, target, generator)

    else:
        if args.task is None:
            raise ValueError("U-Net training requires --task inpaint or --task sr")
        operator = SpatialObservation(args.task)
        model = SpatialUNet(args.task, args.unet_channels).to(device)
        metadata.update(task=args.task, unet_channels=args.unet_channels)

        def loss(target, generator=None):
            output = model(operator.observe(target), operator)
            if args.task == "inpaint":
                missing = ~operator.mask(target)
                return functional.l1_loss(output[missing], target[missing])
            return functional.l1_loss(output, target)

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-5 if args.stage == "prior" else 1e-4
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for target in train_loader:
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = loss(target)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss += value.detach().item() * len(target)
        model.eval()
        validation_loss = 0.0
        validation_noise = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            ae_nonzero = torch.zeros((), dtype=torch.bool, device=device)
            for target in val_loader:
                target = target.to(device)
                if args.stage == "ae":
                    reconstruction = model(target)
                    ae_nonzero |= reconstruction.ne(0).any()
                    value = functional.mse_loss(reconstruction, target)
                else:
                    value = loss(target, validation_noise)
                validation_loss += value.item() * len(target)
            if args.stage == "ae" and not ae_nonzero.item():
                raise RuntimeError("Autoencoder outputs are all zero on validation; refusing to save checkpoint")
        validation_loss /= len(val_data)
        if not isfinite(validation_loss):
            raise ValueError("Non-finite validation loss")
        if args.stage == "ae":
            span = temperature_range[1] - temperature_range[0]
            val_rmse_k = (validation_loss ** 0.5) * span
            print(
                f"epoch {epoch}: train={train_loss / len(train_data):.6g} "
                f"val_mse={validation_loss:.6g} val_rmse_k={val_rmse_k:.2f} K",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch}: train={train_loss / len(train_data):.6g} val={validation_loss:.6g}",
                flush=True,
            )
        is_best = validation_loss < best
        if is_best or epoch in checkpoint_epochs:
            checkpoint = {
                **metadata, "epoch": epoch, "val_loss": validation_loss, "weights": model.state_dict()
            }
            if is_best:
                best = validation_loss
                save_checkpoint(args.output, checkpoint)
            if epoch in checkpoint_epochs:
                path = args.output.with_name(f"{args.output.stem}-epoch-{epoch:04d}{args.output.suffix}")
                save_checkpoint(path, checkpoint)
                print(f"Saved scheduled checkpoint: {path}")


def make_predictor(args, operator, device, checkpoint, *, initializer=None, strength=None):
    if (initializer is None) != (strength is None):
        raise ValueError("Warm-start sampling requires an initializer and strength together")
    if initializer is not None and args.method not in {"dps", "mmps"}:
        raise ValueError("The initializer is only supported for DPS or MMPS")

    if args.method in {"dps", "unguided", "mmps"}:
        model = ViTDecoder(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["weights"])
        model.eval().requires_grad_(False)
        autoencoder = load_autoencoder(
            checkpoint["ae_weights"], device, ae_config=checkpoint.get("ae_config")
        )
        scheduler = diffusion_scheduler(checkpoint["diffusion_steps"])
        generator = torch.Generator(device=device).manual_seed(args.seed)
        guidance = args.guidance if args.method == "dps" else float(args.method == "mmps")
        steps = args.steps if args.steps is not None else scheduler.config.num_train_timesteps
        conditioning = {}
        if args.method == "mmps":
            conditioning = {
                "mmps_sigma": args.mmps_sigma,
                "mmps_maxiter": args.mmps_maxiter,
                "mmps_diagnostics": args.mmps_diagnostics,
            }

        def predict(observation):
            warm = {}
            if initializer is not None:
                with torch.no_grad():
                    warm = {
                        "initial_latent": autoencoder.encoder(initializer(observation)),
                        "strength": strength,
                    }
            return sample_dps(
                model, autoencoder.decoder, operator, observation,
                scheduler=scheduler, steps=steps, guidance=guidance, generator=generator,
                **warm, **conditioning,
            )

        return predict

    if args.method == "unet":
        if checkpoint["task"] != args.task:
            raise ValueError("U-Net checkpoint was trained for a different task")
        model = SpatialUNet(args.task, checkpoint["unet_channels"]).to(device)
        model.load_state_dict(checkpoint["weights"])
        model.eval()

        @torch.no_grad()
        def predict(observation):
            return model(observation, operator)

        return predict

    return lambda observation: interpolate(observation, operator, args.method)


def benchmark(args):
    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    dataset = data.dataset(args.split, temperature_range)
    operator = SpatialObservation(args.task)
    if args.method not in METHODS[args.task]:
        raise ValueError(f"{args.method} is incompatible with {args.task}")
    stage = {"dps": "prior", "unguided": "prior", "unet": "unet", "ae": "ae"}.get(args.method)
    if args.draws > 1 and stage != "prior":
        raise ValueError("Multiple draws are only meaningful for diffusion")
    checkpoint = load_checkpoint(args.checkpoint, stage) if stage else None
    if checkpoint:
        check_provenance(checkpoint, data, temperature_range)
    if args.method == "ae":
        model = load_autoencoder(
            checkpoint["weights"], device, ae_config=checkpoint.get("ae_config")
        )
        with torch.no_grad():
            errors = [(model(frame[None].to(device)).cpu() - frame[None]).square().mean().item()
                      for frame in subset(dataset, args.limit)]
        result = {
            "diagnostic": "Full-field autoencoder reconstruction, not a benchmark competitor",
            "frames": len(errors),
            "rmse_K": (sum(errors) / len(errors)) ** 0.5 * (temperature_range[1] - temperature_range[0]),
        }
    else:
        predict = make_predictor(args, operator, device, checkpoint)
        result = evaluate(
            dataset, operator, predict, device=device, temperature_range=temperature_range,
            draws=args.draws, limit=args.limit,
        )
    details = {
        "method": args.method,
        "task": args.task,
        "split_name": args.split,
        "split": data.split_manifest(),
        "temperature_range": list(temperature_range),
        "device": str(device),
        "seed": args.seed,
    }
    if checkpoint:
        details.update(checkpoint=str(args.checkpoint), epoch=checkpoint["epoch"],
                       training_samples=checkpoint["train_count"])
    if stage == "prior":
        details.update(steps=args.steps or checkpoint["diffusion_steps"],
                       guidance=args.guidance if args.method == "dps" else 0.0,
                       ae_training_samples=checkpoint["ae_train_count"])
    valid = {**details, **result}
    save_json(args.output, valid)
    print(json.dumps({key: value for key, value in valid.items() if key != "split"},
                     indent=2, allow_nan=False))


def reconstruct(args):
    device = prepare(args.device, args.seed)
    operator = SpatialObservation(args.task)
    checkpoint = load_checkpoint(args.checkpoint, "prior")
    observation = np.load(args.observation, allow_pickle=False)
    if observation.shape != operator.shape[1:]:
        raise ValueError(f"Expected a 2-D Kelvin observation of shape {operator.shape[1:]}")
    source = torch.as_tensor(observation, dtype=torch.float32, device=device)[None, None]
    low, high = checkpoint["temperature_range"]
    normalized = (source - low) / (high - low)
    if args.task == "inpaint":
        normalized = operator.observe(normalized)
    args.method = "dps"
    predict = make_predictor(args, operator, device, checkpoint)
    output = predict(normalized) * (high - low) + low
    if args.guidance:
        output = operator.restore_known(output, source)
    if not torch.isfinite(output).all():
        raise ValueError("Reconstruction contains non-finite temperatures")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        np.save(file, output[0, 0].cpu().numpy())


def add_sampling_args(parser):
    parser.add_argument("--task", choices=("inpaint", "sr"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--guidance", type=float, default=1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("stage", choices=("ae", "prior", "unet"))
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--epochs", default=100, type=positive_int)
    training.add_argument("--save-epochs", nargs="+", type=positive_int, default=(),
                          help="Save these exact epochs alongside the best checkpoint")
    training.add_argument("--batch-size", default=4, type=positive_int)
    training.add_argument("--workers", default=0, type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--latent-channels", default=1, type=positive_int,
                          help="Autoencoder latent channel count (default 1)")
    training.add_argument("--ae-output-activation", choices=("relu", "linear"), default="relu",
                          help="Autoencoder decoder output activation (linear removes dead-ReLU zones)")
    training.add_argument("--ae-grad-weight", default=0.0, type=float,
                          help="Optional finite-difference gradient L1 weight for AE training")
    training.add_argument("--ae-checkpoint", type=Path)
    training.add_argument("--width", default=256, type=positive_int)
    training.add_argument("--layers", default=8, type=positive_int)
    training.add_argument("--diffusion-steps", default=1000, type=positive_int)
    training.add_argument("--unet-channels", default=32, type=positive_int)
    training.add_argument("--task", choices=("inpaint", "sr"))
    training.add_argument("--train-limit", type=positive_int, help="For smoke runs only")
    training.add_argument("--val-limit", type=positive_int, help="For smoke runs only")
    training.set_defaults(run=train)

    testing = commands.add_parser("benchmark")
    add_sampling_args(testing)
    testing.add_argument("--method", choices=sorted(set().union(*METHODS.values())), required=True)
    testing.add_argument("--split", choices=("val", "test"), default="test")
    testing.add_argument("--draws", type=positive_int, default=1)
    testing.add_argument("--limit", type=positive_int, help="Use only the first N frames")
    testing.add_argument("--output", type=Path)
    testing.set_defaults(run=benchmark)

    inference = commands.add_parser("reconstruct")
    add_sampling_args(inference)
    inference.add_argument("--observation", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.set_defaults(run=reconstruct)

    for command in (training, testing, inference):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
    for command in (training, testing):
        command.add_argument("--data", type=Path, required=True)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
