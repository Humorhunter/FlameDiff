"""Training and evaluation entry points for single- and multi-field spatial FlameDiff."""

import argparse
import copy
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Subset, TensorDataset

from model.EnDeCoder import AutoEnDecoder, build_autoencoder
from model.LatentUNet import LatentUNet
from model.ViTDecoder import ViTDecoder
from spatial_baselines import SpatialUNet, interpolate
from spatial_data import SUPPORTED_FIELDS, TemperatureData, normalize_field_names
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


def checkpoint_fields(checkpoint) -> tuple[str, ...]:
    fields = checkpoint.get("fields")
    if fields is None:
        return ("T",)
    return normalize_field_names(fields)


def checkpoint_field_ranges(checkpoint) -> dict[str, tuple[float, float]]:
    fields = checkpoint_fields(checkpoint)
    if "field_ranges" in checkpoint:
        return {name: (float(bounds[0]), float(bounds[1])) for name, bounds in checkpoint["field_ranges"].items()}
    if fields == ("T",) and "temperature_range" in checkpoint:
        low, high = map(float, checkpoint["temperature_range"])
        return {"T": (low, high)}
    raise ValueError("Checkpoint is missing normalization ranges")


def check_provenance(checkpoint, data, temperature_range=None):
    if checkpoint["split"] != data.split_manifest():
        raise ValueError("Checkpoint split does not match the current data manifest")
    fields = checkpoint_fields(checkpoint)
    if fields != data.fields:
        raise ValueError(f"Checkpoint fields {fields} do not match dataset fields {data.fields}")
    if fields == ("T",):
        expected_range = data.training_range() if temperature_range is None else temperature_range
        if tuple(checkpoint["temperature_range"]) != tuple(expected_range):
            raise ValueError("Checkpoint temperature range does not match the training split")
    else:
        expected_ranges = data.training_ranges()
        stored_ranges = checkpoint_field_ranges(checkpoint)
        if stored_ranges != expected_ranges:
            raise ValueError("Checkpoint field ranges do not match the training split")


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
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def load_autoencoder(weights, device, ae_config=None):
    if ae_config is None:
        inferred_latent = 1
        inferred_in = 1
        if isinstance(weights, dict):
            if "encoder.proj2.bias" in weights:
                inferred_latent = int(weights["encoder.proj2.bias"].shape[0])
            if "encoder.conv1.weight" in weights:
                inferred_in = int(weights["encoder.conv1.weight"].shape[1])
        ae_config = {
            "arch": "legacy",
            "in_channels": inferred_in,
            "latent_channels": inferred_latent,
            "output_activation": "relu",
            "latent_activation": "tanh",
        }
    model = build_autoencoder(ae_config)
    model.load_state_dict(weights)
    return model.to(device).eval().requires_grad_(False)


def build_prior_model(prior_arch: str, model_config: dict):
    if prior_arch == "vit":
        return ViTDecoder(**model_config)
    if prior_arch == "unet":
        return LatentUNet(**model_config)
    raise ValueError(f"Unsupported prior_arch {prior_arch!r}; expected 'vit' or 'unet'")


def load_prior_model(checkpoint: dict, device):
    prior_arch = str(checkpoint.get("prior_arch", "vit"))
    model = build_prior_model(prior_arch, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    return model.eval().requires_grad_(False)


def _latent_tv_loss(latent: torch.Tensor) -> torch.Tensor:
    """Spatial total-variation L1 smoothness penalty on latent maps (Wu et al. Fig. 5)."""
    dy = (latent[..., 1:, :] - latent[..., :-1, :]).abs().mean()
    dx = (latent[..., :, 1:] - latent[..., :, :-1]).abs().mean()
    return 0.5 * (dy + dx)


def _weighted_pixel_loss(prediction, target, *, loss_kind="mse", channel_weights=None):
    if loss_kind == "l1":
        err = (prediction - target).abs()
    else:
        err = (prediction - target).square()
    if channel_weights is not None:
        err = err * channel_weights.view(1, -1, 1, 1)
    return err.mean()


def _gradient_l1_loss(prediction, target, *, channel_weights=None):
    dy = (
        (prediction[..., 1:, :] - prediction[..., :-1, :])
        - (target[..., 1:, :] - target[..., :-1, :])
    ).abs()
    dx = (
        (prediction[..., :, 1:] - prediction[..., :, :-1])
        - (target[..., :, 1:] - target[..., :, :-1])
    ).abs()
    if channel_weights is not None:
        w = channel_weights.view(1, -1, 1, 1)
        dy = dy * w
        dx = dx * w
    return 0.5 * (dy.mean() + dx.mean())


def compute_latent_scale(autoencoder, loader, device) -> dict[str, list[float]]:
    """Compute exact per-latent-channel mean and std over training frames."""
    sum_c = None
    sq_c = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            z = autoencoder.encoder(batch.to(device)).to(dtype=torch.float64)
            # z: [B, C_lat, H, W]
            if sum_c is None:
                sum_c = torch.zeros(z.shape[1], dtype=torch.float64, device=device)
                sq_c = torch.zeros(z.shape[1], dtype=torch.float64, device=device)
            sum_c += z.sum(dim=(0, 2, 3))
            sq_c += z.square().sum(dim=(0, 2, 3))
            count += int(z.shape[0] * z.shape[2] * z.shape[3])
    mean_c = sum_c / count
    var_c = (sq_c / count - mean_c.square()).clamp_min(1e-12)
    std_c = var_c.sqrt()
    return {
        "mean": [float(v) for v in mean_c.cpu().tolist()],
        "std": [float(v) for v in std_c.cpu().tolist()],
    }


def subset(data, limit):
    return data if limit is None else Subset(data, range(min(limit, len(data))))


def train(args):
    checkpoint_epochs = set(args.save_epochs)
    if checkpoint_epochs and max(checkpoint_epochs) > args.epochs:
        raise ValueError("--save-epochs cannot contain an epoch beyond --epochs")
    device = prepare(args.device, args.seed)

    # Determine fields: for stage=='prior', inherit fields from ae_checkpoint automatically.
    pretrained = None
    if args.stage == "prior":
        pretrained = load_checkpoint(args.ae_checkpoint, "ae")
        fields = checkpoint_fields(pretrained)
    else:
        fields = normalize_field_names(getattr(args, "fields", ("T",)))

    holdout_cases = getattr(args, "holdout_cases", None)
    data = TemperatureData(args.data, fields=fields, holdout_cases=holdout_cases)
    field_ranges = data.training_ranges()
    channel_stats = data.training_channel_stats()
    norm_arg = field_ranges["T"] if fields == ("T",) else field_ranges

    train_data = subset(data.dataset("train", norm_arg), args.train_limit)
    val_data = subset(data.dataset("val", norm_arg), args.val_limit)
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
        "train_count": len(train_data),
        "val_count": len(val_data),
        "seed": args.seed,
    }
    if fields == ("T",):
        metadata["temperature_range"] = list(field_ranges["T"])
    else:
        metadata["fields"] = list(fields)
        metadata["field_ranges"] = {k: list(v) for k, v in field_ranges.items()}
        metadata["channel_stats"] = channel_stats
        if "T" in field_ranges:
            metadata["temperature_range"] = list(field_ranges["T"])

    ema_model = None
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    if not isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")

    if args.stage == "ae":
        ae_arch = str(getattr(args, "ae_arch", "legacy"))
        ae_config = {
            "arch": ae_arch,
            "in_channels": len(fields),
            "latent_channels": args.latent_channels,
            "output_activation": args.ae_output_activation,
            "latent_activation": getattr(args, "ae_latent_activation", "tanh"),
        }
        if ae_arch == "mscae":
            ae_config["base_channels"] = int(getattr(args, "ae_base_channels", 48))
        grad_weight = float(args.ae_grad_weight)
        latent_reg = float(getattr(args, "ae_latent_reg", 0.0))
        latent_tv_weight = float(getattr(args, "ae_latent_tv_weight", 0.0))
        loss_kind = str(getattr(args, "ae_loss", "mse"))
        channel_balance = bool(getattr(args, "ae_channel_balance", False))
        if not isfinite(grad_weight) or grad_weight < 0:
            raise ValueError("--ae-grad-weight must be finite and non-negative")
        if not isfinite(latent_reg) or latent_reg < 0:
            raise ValueError("--ae-latent-reg must be finite and non-negative")
        if not isfinite(latent_tv_weight) or latent_tv_weight < 0:
            raise ValueError("--ae-latent-tv-weight must be finite and non-negative")

        model = build_autoencoder(ae_config).to(device)
        out_head = model.decoder.conv8 if ae_arch == "legacy" else model.decoder.out_conv
        if ae_config["output_activation"] == "relu":
            torch.nn.init.zeros_(out_head.weight)
            torch.nn.init.constant_(out_head.bias, 0.5)
        else:
            # Initialize linear decoder bias to training normalized channel means
            # so sparse species channels (OH/HO2) start near their true mean instead of 0.5.
            with torch.no_grad():
                for c_idx, fname in enumerate(fields):
                    out_head.bias[c_idx] = float(channel_stats[fname]["norm_mean"])

        channel_weights_list = [1.0] * len(fields)
        if channel_balance and len(fields) > 1:
            power = 2.0 if loss_kind == "mse" else 1.0
            inv_scales = [1.0 / (max(channel_stats[fname]["norm_std"], 1e-4) ** power) for fname in fields]
            mean_inv = sum(inv_scales) / len(inv_scales)
            channel_weights_list = [v / mean_inv for v in inv_scales]
        channel_weights = torch.tensor(channel_weights_list, dtype=torch.float32, device=device)

        metadata.update(
            ae_config=ae_config,
            ae_loss=loss_kind,
            ae_grad_weight=grad_weight,
            ae_latent_reg=latent_reg,
            ae_latent_tv_weight=latent_tv_weight,
            ae_channel_weights=channel_weights_list,
        )

        def loss(target, generator=None):
            latent = model.encoder(target)
            prediction = model.decoder(latent)
            value = _weighted_pixel_loss(
                prediction, target, loss_kind=loss_kind, channel_weights=channel_weights
            )
            if grad_weight > 0.0:
                value = value + grad_weight * _gradient_l1_loss(
                    prediction, target, channel_weights=channel_weights
                )
            if latent_reg > 0.0:
                value = value + latent_reg * latent.square().mean()
            if latent_tv_weight > 0.0:
                value = value + latent_tv_weight * _latent_tv_loss(latent)
            return value

    elif args.stage == "prior":
        prior_arch = str(getattr(args, "prior_arch", "vit"))
        if prior_arch == "vit" and args.width % 8:
            raise ValueError("Denoiser width must be a multiple of eight")
        check_provenance(pretrained, data)
        ae_config = pretrained.get("ae_config")
        if ae_config is None:
            inferred_latent = int(pretrained["weights"]["encoder.proj2.bias"].shape[0])
            inferred_in = int(pretrained["weights"]["encoder.conv1.weight"].shape[1])
            ae_config = {
                "arch": "legacy",
                "in_channels": inferred_in,
                "latent_channels": inferred_latent,
                "output_activation": "relu",
                "latent_activation": "tanh",
            }
        autoencoder = load_autoencoder(pretrained["weights"], device, ae_config=ae_config)

        # Pre-encode frozen AE latents once (only ~150 MB for 5,850 frames) so diffusion prior
        # training avoids redundant full-resolution encoding and disk I/O every epoch.
        def _cache_latents(dataset_obj):
            enc_loader = DataLoader(dataset_obj, batch_size=max(args.batch_size, 32), num_workers=args.workers)
            chunks = []
            with torch.no_grad():
                for batch in enc_loader:
                    chunks.append(autoencoder.encoder(batch.to(device)).cpu())
            return torch.cat(chunks, dim=0)

        train_latents = _cache_latents(train_data)
        val_latents = _cache_latents(val_data)
        latent_scale = None
        if getattr(args, "standardize_latent", False):
            z64 = train_latents.to(dtype=torch.float64)
            mean_c = z64.mean(dim=(0, 2, 3))
            std_c = z64.var(dim=(0, 2, 3), unbiased=False).clamp_min(1e-12).sqrt()
            latent_scale = {
                "mean": [float(v) for v in mean_c.tolist()],
                "std": [float(v) for v in std_c.tolist()],
            }
            print(f"Standardizing latent space: mean={latent_scale['mean']}, std={latent_scale['std']}", flush=True)
            if args.train_limit is None and min(latent_scale["std"]) < 1e-2:
                raise RuntimeError(
                    f"Refusing to train prior: AE latent channel collapsed (std={latent_scale['std']})"
                )
            m_t = mean_c.to(dtype=train_latents.dtype).view(1, -1, 1, 1)
            s_t = std_c.to(dtype=train_latents.dtype).view(1, -1, 1, 1)
            train_latents = (train_latents - m_t) / s_t
            val_latents = (val_latents - m_t) / s_t

        pin = device.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(train_latents), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin
        )
        val_loader = DataLoader(
            TensorDataset(val_latents), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )
        print(
            f"Cached frozen AE latents in RAM: train={tuple(train_latents.shape)}, val={tuple(val_latents.shape)}",
            flush=True,
        )

        if prior_arch == "vit":
            model_config = {"d_model": args.width, "num_layers": args.layers, "num_feats": 0}
            if autoencoder.latent_channels != 1:
                model_config["in_channels"] = autoencoder.latent_channels
        elif prior_arch == "unet":
            model_config = {
                "in_channels": autoencoder.latent_channels,
                "base_channels": int(getattr(args, "prior_base_channels", 64)),
                "channel_mults": [1, 1, 2, 3],
                "num_res_blocks": 2,
                "attention_levels": [2, 3],
                "num_heads": 4,
            }
        else:
            raise ValueError(f"Unsupported --prior-arch {prior_arch!r}")
        model = build_prior_model(prior_arch, model_config).to(device)
        if ema_decay > 0.0:
            ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        beta_schedule = str(getattr(args, "beta_schedule", "linear"))
        clip_sample = latent_scale is None
        scheduler = diffusion_scheduler(
            args.diffusion_steps, beta_schedule=beta_schedule, clip_sample=clip_sample
        )
        metadata.update(
            prior_arch=prior_arch,
            model_config=model_config,
            diffusion_steps=args.diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            ae_config=ae_config,
            ae_weights=pretrained["weights"],
            ae_train_count=pretrained["train_count"],
        )
        if latent_scale is not None:
            metadata["latent_scale"] = latent_scale
        if ema_decay > 0.0:
            metadata["ema_decay"] = ema_decay

        def loss(latent_batch, generator=None, eval_model=None):
            active_model = model if eval_model is None else eval_model
            latent = latent_batch[0] if isinstance(latent_batch, (tuple, list)) else latent_batch
            latent = latent.to(device, non_blocking=pin)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latent.shape[0],),
                device=device,
                generator=generator,
            )
            noise = torch.randn(latent.shape, dtype=latent.dtype, device=device, generator=generator)
            noisy = scheduler.add_noise(latent, noise, timesteps)
            predicted = active_model(noisy, None, timesteps[:, None])
            return functional.mse_loss(predicted, noise)

    else:
        if args.task is None:
            raise ValueError("U-Net training requires --task inpaint or --task sr")
        if fields != ("T",):
            raise ValueError("SpatialUNet legacy training currently supports single-field ('T',) only")
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
    lr_schedule = str(getattr(args, "lr_schedule", "constant"))
    max_steps = getattr(args, "max_steps", None)
    steps_per_epoch = max(len(train_loader), 1)
    total_epochs = (
        max(args.epochs, (max_steps + steps_per_epoch - 1) // steps_per_epoch)
        if max_steps is not None
        else args.epochs
    )
    total_grad_steps = max_steps if max_steps is not None else total_epochs * steps_per_epoch
    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_grad_steps if max_steps is not None else total_epochs, 1),
            eta_min=learning_rate * 0.01,
        )
        if lr_schedule == "cosine"
        else None
    )

    best = float("inf")
    global_step = 0
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss = 0.0
        seen_samples = 0
        for target in train_loader:
            if isinstance(target, (tuple, list)):
                batch_len = len(target[0])
                batch_in = target
            else:
                batch_len = len(target)
                batch_in = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = loss(batch_in)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            if lr_scheduler is not None and max_steps is not None:
                lr_scheduler.step()
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
            train_loss += value.detach().item() * batch_len
            seen_samples += batch_len
            if max_steps is not None and global_step >= max_steps:
                break
        if lr_scheduler is not None and max_steps is None:
            lr_scheduler.step()

        model.eval()
        eval_net = ema_model if ema_model is not None else model
        validation_loss = 0.0
        per_ch_sq = torch.zeros(len(fields), dtype=torch.float64, device=device) if args.stage == "ae" else None
        per_ch_count = 0
        z_sum = None
        z_sq = None
        z_count = 0
        validation_noise = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            ae_nonzero = torch.zeros((), dtype=torch.bool, device=device)
            for target in val_loader:
                if args.stage == "prior":
                    batch_len = len(target[0]) if isinstance(target, (tuple, list)) else len(target)
                    value = loss(target, validation_noise, eval_model=eval_net)
                else:
                    target = target.to(device)
                    batch_len = len(target)
                    if args.stage == "ae":
                        z_val = model.encoder(target)
                        reconstruction = model.decoder(z_val)
                        ae_nonzero |= reconstruction.ne(0).any()
                        value = _weighted_pixel_loss(
                            reconstruction, target, loss_kind=loss_kind, channel_weights=channel_weights
                        )
                        per_ch_sq += (reconstruction - target).to(torch.float64).square().sum(dim=(0, 2, 3))
                        per_ch_count += int(target.shape[0] * target.shape[2] * target.shape[3])
                        z64 = z_val.to(torch.float64)
                        if z_sum is None:
                            z_sum = torch.zeros(z64.shape[1], dtype=torch.float64, device=device)
                            z_sq = torch.zeros(z64.shape[1], dtype=torch.float64, device=device)
                        z_sum += z64.sum(dim=(0, 2, 3))
                        z_sq += z64.square().sum(dim=(0, 2, 3))
                        z_count += int(z64.shape[0] * z64.shape[2] * z64.shape[3])
                    else:
                        value = loss(target, validation_noise)
                validation_loss += value.item() * batch_len
            if args.stage == "ae" and not ae_nonzero.item():
                raise RuntimeError("Autoencoder outputs are all zero on validation; refusing to save checkpoint")
        validation_loss /= len(val_data)
        if not isfinite(validation_loss):
            raise ValueError("Non-finite validation loss")
        if args.stage == "ae":
            z_mean_c = z_sum / max(z_count, 1)
            z_std_c = (z_sq / max(z_count, 1) - z_mean_c.square()).clamp_min(0.0).sqrt().cpu().tolist()
            ch_parts = []
            ch_rmses = {}
            for c_idx, fname in enumerate(fields):
                low_c, high_c = field_ranges[fname]
                rmse_phys = float(((per_ch_sq[c_idx] / per_ch_count).sqrt().item()) * (high_c - low_c))
                ch_rmses[fname] = rmse_phys
                ch_parts.append(f"{fname}_rmse={rmse_phys:.4g}")
            print(
                f"epoch {epoch} (step {global_step}): train={train_loss / max(seen_samples, 1):.6g} "
                f"val={validation_loss:.6g} " + " ".join(ch_parts)
                + f" z_std={[round(v, 4) for v in z_std_c]}",
                flush=True,
            )
            # Fail-fast circuit breaker: catch constant/mode collapse early instead of wasting GPU hours.
            if args.train_limit is None and epoch >= 5:
                if min(z_std_c) < 1e-3:
                    raise RuntimeError(
                        f"AE latent collapsed at epoch {epoch} (z_std={z_std_c}); aborting."
                    )
                collapsed_channels = [
                    f"{fn} ({ch_rmses[fn]:.4g} >= 0.90*{channel_stats[fn]['std']:.4g})"
                    for fn in fields
                    if ch_rmses[fn] >= 0.90 * channel_stats[fn]["std"]
                ]
                if collapsed_channels:
                    raise RuntimeError(
                        f"AE failed to beat constant baseline at epoch {epoch} on {', '.join(collapsed_channels)}; aborting."
                    )
        else:
            print(
                f"epoch {epoch} (step {global_step}): train={train_loss / max(seen_samples, 1):.6g} val={validation_loss:.6g}",
                flush=True,
            )
        is_best = validation_loss < best
        if is_best or epoch in checkpoint_epochs:
            checkpoint = {
                **metadata,
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": validation_loss,
                "weights": eval_net.state_dict(),
            }
            if is_best:
                best = validation_loss
                save_checkpoint(args.output, checkpoint)
            if epoch in checkpoint_epochs:
                path = args.output.with_name(f"{args.output.stem}-epoch-{epoch:04d}{args.output.suffix}")
                save_checkpoint(path, checkpoint)
                print(f"Saved scheduled checkpoint: {path}")
        if max_steps is not None and global_step >= max_steps:
            break


def make_predictor(args, operator, device, checkpoint, *, initializer=None, strength=None):
    if (initializer is None) != (strength is None):
        raise ValueError("Warm-start sampling requires an initializer and strength together")
    if initializer is not None and args.method not in {"dps", "mmps"}:
        raise ValueError("The initializer is only supported for DPS or MMPS")

    if args.method in {"dps", "unguided", "mmps"}:
        model = load_prior_model(checkpoint, device)
        autoencoder = load_autoencoder(
            checkpoint["ae_weights"], device, ae_config=checkpoint.get("ae_config")
        )
        scheduler = diffusion_scheduler(
            checkpoint["diffusion_steps"],
            beta_schedule=checkpoint.get("beta_schedule", "linear"),
            clip_sample=checkpoint.get("clip_sample", True),
        )
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
    training.add_argument("--fields", nargs="+", choices=SUPPORTED_FIELDS, default=["T"],
                          help="Physical scalar fields for AE training (e.g. T OH HO2; prior inherits from AE)")
    training.add_argument("--epochs", default=100, type=positive_int)
    training.add_argument("--max-steps", type=positive_int,
                          help="Optional total optimizer step budget (overrides epochs if larger; steps cosine LR per update)")
    training.add_argument("--save-epochs", nargs="+", type=positive_int, default=(),
                          help="Save these exact epochs alongside the best checkpoint")
    training.add_argument("--holdout-cases", type=str,
                          help="Comma-separated case names held out for Leave-One-Case-Out testing (e.g. T1050)")
    training.add_argument("--batch-size", default=4, type=positive_int)
    training.add_argument("--workers", default=0, type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                          help="Learning-rate schedule")
    training.add_argument("--ae-arch", choices=("legacy", "mscae"), default="legacy",
                          help="Autoencoder architecture (mscae = parallel 3/5/7 multiscale + PixelShuffle)")
    training.add_argument("--ae-base-channels", default=48, type=positive_int,
                          help="Base channel width for --ae-arch mscae")
    training.add_argument("--latent-channels", default=1, type=positive_int,
                          help="Autoencoder latent channel count (default 1)")
    training.add_argument("--ae-output-activation", choices=("relu", "linear"), default="relu",
                          help="Autoencoder decoder output activation (linear removes dead-ReLU zones)")
    training.add_argument("--ae-latent-activation", choices=("tanh", "linear"), default="tanh",
                          help="Autoencoder bottleneck activation (linear avoids tanh variance collapse)")
    training.add_argument("--ae-loss", choices=("mse", "l1"), default="mse",
                          help="Pixel reconstruction loss for AE training")
    training.add_argument("--ae-grad-weight", default=0.0, type=float,
                          help="Optional finite-difference gradient L1 weight for AE training")
    training.add_argument("--ae-latent-reg", default=0.0, type=float,
                          help="Optional L2 regularization weight on AE latents")
    training.add_argument("--ae-latent-tv-weight", default=0.0, type=float,
                          help="Optional spatial TV smoothness penalty on AE latents (Wu et al. Fig. 5)")
    training.add_argument("--ae-channel-balance", action="store_true",
                          help="Weight AE channel losses inversely by normalized channel std")
    training.add_argument("--ae-checkpoint", type=Path)
    training.add_argument("--prior-arch", choices=("vit", "unet"), default="vit",
                          help="Diffusion prior backbone (unet = multi-scale Attention U-Net)")
    training.add_argument("--prior-base-channels", default=64, type=positive_int,
                          help="Base channels for --prior-arch unet")
    training.add_argument("--standardize-latent", action="store_true",
                          help="Standardize AE latents to zero mean & unit variance for diffusion prior")
    training.add_argument("--beta-schedule", choices=("linear", "squaredcos_cap_v2"), default="linear",
                          help="DDPM noise schedule for prior training")
    training.add_argument("--ema-decay", default=0.0, type=float,
                          help="Optional EMA weight decay for prior training (e.g. 0.999)")
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
        command.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
    for command in (training, testing):
        command.add_argument("--data", type=Path, required=True)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
