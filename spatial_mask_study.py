"""Train and evaluate a shared U-Net across centered inpainting mask sizes.

The training mask assignment is balanced per epoch: each training frame is
paired with one of the small, existing, or large centered holes, with counts
that differ by at most one. Benchmarking reports each mask size separately.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from math import isfinite
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset

from spatial import (
    CHECKPOINT_VERSION,
    check_provenance,
    load_checkpoint,
    make_predictor,
    prepare,
    save_checkpoint,
    save_json,
)
from spatial_baselines import SpatialUNet, interpolate
from spatial_data import TemperatureData
from spatial_eval import _as_frame, _frame_metrics
from spatial_inpaint import (
    InpaintResidualDenoiser,
    inpaint_diffusion_scheduler,
    masked_residual_loss,
    sample_inpaint_residual,
)


FIELD_SHAPE = (1, 192, 256)
MASK_SIZES = {
    "small": (48, 64),
    "medium": (96, 128),  # Existing FlameDiff inpainting mask.
    "large": (144, 192),
}
DEPTH_BINS = ((1, 1), (2, 4), (5, 8), (9, 16), (17, None))
HOT_THRESHOLD_K = 1200.0


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def diffusion_steps(value):
    number = positive_int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("Diffusion requires at least two steps")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer")
    return number


class CenteredMask:
    """Inpainting observation operator for one or more centered hole sizes."""

    task = "inpaint"
    shape = FIELD_SHAPE

    def __init__(self, mask_sizes: str | Sequence[str]):
        if isinstance(mask_sizes, str):
            names = (mask_sizes,)
            self._single_size = True
        else:
            names = tuple(mask_sizes)
            self._single_size = False
        if not names or any(name not in MASK_SIZES for name in names):
            raise ValueError(f"mask_sizes must use names from {tuple(MASK_SIZES)}")
        self.mask_sizes = names
        self.mask_size = names[0] if self._single_size else None

    @property
    def measured_count(self):
        if not self._single_size:
            raise ValueError("measured_count is undefined for mixed-size batches")
        hole_height, hole_width = MASK_SIZES[self.mask_sizes[0]]
        return 192 * 256 - hole_height * hole_width

    def mask(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 4 or tuple(reference.shape[1:]) != FIELD_SHAPE:
            raise ValueError(f"Expected [B, {FIELD_SHAPE[0]}, 192, 256] field tensors")
        if self._single_size:
            names = self.mask_sizes * reference.shape[0]
        else:
            names = self.mask_sizes
            if len(names) != reference.shape[0]:
                raise ValueError("The number of per-sample masks must match batch size")

        known = torch.ones_like(reference, dtype=torch.bool)
        height, width = reference.shape[-2:]
        for index, name in enumerate(names):
            hole_height, hole_width = MASK_SIZES[name]
            top = (height - hole_height) // 2
            left = (width - hole_width) // 2
            known[index, :, top:top + hole_height, left:left + hole_width] = False
        return known

    def observe(self, field: torch.Tensor) -> torch.Tensor:
        return field.masked_fill(~self.mask(field), 0)

    def restore_known(self, field: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return torch.where(self.mask(field), observation, field)


def _data_for(args):
    if args.test_case is None:
        return TemperatureData(args.data)
    try:
        from spatial_case_holdout import CaseHoldoutData
    except ImportError as error:
        raise RuntimeError(
            "--test-case requires spatial_case_holdout.py to be present"
        ) from error
    return CaseHoldoutData(args.data, args.test_case)


def _subset(dataset, limit):
    return dataset if limit is None else Subset(dataset, range(min(limit, len(dataset))))


def _per_sample_masked_l1(prediction, target, operator):
    missing = ~operator.mask(target)
    errors = (prediction - target).abs().masked_fill(~missing, 0)
    return (errors.sum(dim=(1, 2, 3)) / missing.sum(dim=(1, 2, 3))).mean()


def _balanced_mask_schedule(sample_count: int, generator: torch.Generator) -> list[str]:
    names = list(MASK_SIZES)
    assignments = [names[index % len(names)] for index in range(sample_count)]
    order = torch.randperm(sample_count, generator=generator).tolist()
    return [assignments[index] for index in order]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_shared_unet(path, data, temperature_range, device):
    checkpoint = load_checkpoint(path, "mask_curriculum_unet")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "inpaint":
        raise ValueError("The shared U-Net checkpoint must be an inpainting model")
    if checkpoint.get("mask_sizes") != {
        name: list(size) for name, size in MASK_SIZES.items()
    }:
        raise ValueError("The shared U-Net checkpoint uses different mask geometries")
    model = SpatialUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval().requires_grad_(False)
    return checkpoint, model


def _validate_epoch(model, loader, device):
    model.eval()
    losses = {}
    with torch.no_grad():
        for name in MASK_SIZES:
            operator = CenteredMask(name)
            total = 0.0
            count = 0
            for raw_target in loader:
                target = raw_target.to(device)
                prediction = model(operator.observe(target), operator)
                value = _per_sample_masked_l1(prediction, target, operator)
                total += value.item() * len(target)
                count += len(target)
            if count == 0:
                raise ValueError("Validation split is empty")
            losses[name] = total / count
    return losses


def train(args):
    if args.save_epochs and max(args.save_epochs) > args.epochs:
        raise ValueError("--save-epochs cannot contain an epoch beyond --epochs")
    if not isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")

    device = prepare(args.device, args.seed)
    data = _data_for(args)
    temperature_range = data.training_range()
    train_data = _subset(data.dataset("train", temperature_range), args.train_limit)
    val_data = _subset(data.dataset("val", temperature_range), args.val_limit)
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Training and validation splits must be non-empty")

    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    mask_generator = torch.Generator().manual_seed(args.seed + 2)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    model_config = {"task": "inpaint", "base_channels": args.unet_channels}
    model = SpatialUNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metadata = {
        "version": CHECKPOINT_VERSION,
        "stage": "mask_curriculum_unet",
        "task": "inpaint",
        "model_config": model_config,
        "mask_sizes": {name: list(size) for name, size in MASK_SIZES.items()},
        "mask_assignment": "balanced_per_training_epoch; counts differ by at most one",
        "split": data.split_manifest(),
        "temperature_range": list(temperature_range),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "seed": args.seed,
        "test_case": args.test_case,
    }

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        schedule = _balanced_mask_schedule(len(train_data), mask_generator)
        model.train()
        total_loss = 0.0
        sample_offset = 0
        for raw_target in train_loader:
            target = raw_target.to(device)
            names = schedule[sample_offset:sample_offset + len(target)]
            sample_offset += len(target)
            operator = CenteredMask(names)
            observation = operator.observe(target)
            prediction = model(observation, operator)
            loss = _per_sample_masked_l1(prediction, target, operator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.detach().item() * len(target)
        if sample_offset != len(train_data):
            raise RuntimeError("Mask schedule and training loader yielded different sample counts")

        validation = _validate_epoch(model, val_loader, device)
        validation_loss = sum(validation.values()) / len(validation)
        if not isfinite(validation_loss):
            raise ValueError("Non-finite validation loss")
        print(
            f"epoch {epoch}: train={total_loss / len(train_data):.6g} "
            f"val={validation_loss:.6g} "
            + " ".join(f"val_{name}={value:.6g}" for name, value in validation.items()),
            flush=True,
        )
        is_best = validation_loss < best
        if is_best or epoch in args.save_epochs:
            checkpoint = {
                **metadata,
                "epoch": epoch,
                "val_loss": validation_loss,
                "val_loss_by_mask": validation,
                "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            if is_best:
                best = validation_loss
                save_checkpoint(args.output, checkpoint)
            if epoch in args.save_epochs:
                path = args.output.with_name(
                    f"{args.output.stem}-epoch-{epoch:04d}{args.output.suffix}"
                )
                save_checkpoint(path, checkpoint)
                print(f"Saved scheduled checkpoint: {path}", flush=True)


def _residual_validation(model, unet, loader, device, scheduler, seed):
    model.eval()
    unet.eval()
    losses = {}
    with torch.no_grad():
        for name in MASK_SIZES:
            operator = CenteredMask(name)
            generator = torch.Generator(device=device).manual_seed(seed)
            total = 0.0
            count = 0
            for raw_target in loader:
                target = raw_target.to(device)
                observation = operator.observe(target)
                known = operator.mask(target)
                base = unet(observation, operator)
                loss = masked_residual_loss(
                    model, base, observation, known, target, scheduler,
                    generator=generator,
                )
                total += loss.item() * len(target)
                count += len(target)
            if count == 0:
                raise ValueError("Validation split is empty")
            losses[name] = total / count
    return losses


def train_residual(args):
    if args.save_epochs and max(args.save_epochs) > args.epochs:
        raise ValueError("--save-epochs cannot contain an epoch beyond --epochs")
    if not isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if args.variant == "conv" and args.attention_blocks != 0:
        raise ValueError("The conv variant requires --attention-blocks 0")
    if args.variant == "transformer" and args.attention_blocks not in (1, 2):
        raise ValueError("The transformer variant requires --attention-blocks 1 or 2")

    device = prepare(args.device, args.seed)
    data = _data_for(args)
    temperature_range = data.training_range()
    unet_checkpoint, unet = _load_shared_unet(
        args.unet_checkpoint, data, temperature_range, device
    )
    train_data = _subset(data.dataset("train", temperature_range), args.train_limit)
    val_data = _subset(data.dataset("val", temperature_range), args.val_limit)
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Training and validation splits must be non-empty")

    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    mask_generator = torch.Generator().manual_seed(args.seed + 2)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model_config = {
        "width": args.width,
        "residual_scale": args.residual_scale,
        "attention_blocks": args.attention_blocks,
    }
    model = InpaintResidualDenoiser(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = inpaint_diffusion_scheduler(args.diffusion_steps)
    train_noise = torch.Generator(device=device).manual_seed(args.seed + 3)
    metadata = {
        "version": CHECKPOINT_VERSION,
        "stage": "mask_residual_diffusion",
        "task": "inpaint",
        "variant": args.variant,
        "model_config": model_config,
        "diffusion_steps": args.diffusion_steps,
        "mask_sizes": {name: list(size) for name, size in MASK_SIZES.items()},
        "mask_assignment": "balanced_per_training_epoch; counts differ by at most one",
        "split": data.split_manifest(),
        "temperature_range": list(temperature_range),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "seed": args.seed,
        "test_case": args.test_case,
        "unet_checkpoint": str(args.unet_checkpoint),
        "unet_checkpoint_sha256": _sha256(args.unet_checkpoint),
        "unet_epoch": unet_checkpoint["epoch"],
        "unet_train_count": unet_checkpoint["train_count"],
    }

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        schedule = _balanced_mask_schedule(len(train_data), mask_generator)
        model.train()
        total_loss = 0.0
        sample_offset = 0
        for raw_target in train_loader:
            target = raw_target.to(device)
            names = schedule[sample_offset:sample_offset + len(target)]
            sample_offset += len(target)
            losses = []
            for name in MASK_SIZES:
                indices = [index for index, assigned in enumerate(names) if assigned == name]
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                group_target = target.index_select(0, index_tensor)
                operator = CenteredMask(name)
                observation = operator.observe(group_target)
                known = operator.mask(group_target)
                with torch.no_grad():
                    base = unet(observation, operator)
                group_loss = masked_residual_loss(
                    model,
                    base,
                    observation,
                    known,
                    group_target,
                    scheduler,
                    generator=train_noise,
                )
                losses.append((group_loss, len(group_target)))
            if not losses:
                raise RuntimeError("Balanced mask schedule assigned no samples to a batch")
            loss = sum(value * count for value, count in losses) / len(target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite masked residual diffusion training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.detach().item() * len(target)
        if sample_offset != len(train_data):
            raise RuntimeError("Mask schedule and training loader yielded different sample counts")

        validation = _residual_validation(
            model, unet, val_loader, device, scheduler, args.seed + 4
        )
        validation_loss = sum(validation.values()) / len(validation)
        if not isfinite(validation_loss):
            raise ValueError("Non-finite residual validation loss")
        print(
            f"epoch {epoch}: train={total_loss / len(train_data):.6g} "
            f"val={validation_loss:.6g} "
            + " ".join(f"val_{name}={value:.6g}" for name, value in validation.items()),
            flush=True,
        )
        is_best = validation_loss < best
        if is_best or epoch in args.save_epochs:
            checkpoint = {
                **metadata,
                "epoch": epoch,
                "val_loss": validation_loss,
                "val_loss_by_mask": validation,
                "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            if is_best:
                best = validation_loss
                save_checkpoint(args.output, checkpoint)
            if epoch in args.save_epochs:
                path = args.output.with_name(
                    f"{args.output.stem}-epoch-{epoch:04d}{args.output.suffix}"
                )
                save_checkpoint(path, checkpoint)
                print(f"Saved scheduled checkpoint: {path}", flush=True)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _depth_mae(prediction, target, name, temperature_range):
    height, width = target.shape[-2:]
    hole_height, hole_width = MASK_SIZES[name]
    top, left = (height - hole_height) // 2, (width - hole_width) // 2
    rows = torch.arange(hole_height, device=target.device)[:, None]
    cols = torch.arange(hole_width, device=target.device)[None, :]
    depth = torch.minimum(
        torch.minimum(rows + 1, hole_height - rows),
        torch.minimum(cols + 1, hole_width - cols),
    )
    error_k = (prediction - target).abs()[0, 0, top:top + hole_height, left:left + hole_width]
    error_k = error_k * (temperature_range[1] - temperature_range[0])
    result = {}
    for lower, upper in DEPTH_BINS:
        key = str(lower) if upper == lower else f"{lower}-{upper}" if upper else f">={lower}"
        region = depth >= lower
        if upper is not None:
            region &= depth <= upper
        result[key] = float(error_k[region].mean().item())
    return result


def _hot_region_error(prediction, target, known, temperature_range):
    low, high = temperature_range
    target_k = target * (high - low) + low
    prediction_k = prediction * (high - low) + low
    hole = ~known
    truth_hot = hole & (target_k >= HOT_THRESHOLD_K)
    predicted_hot = hole & (prediction_k >= HOT_THRESHOLD_K)
    hot_count = int(truth_hot.sum().item())
    predicted_count = int(predicted_hot.sum().item())
    intersection = int((truth_hot & predicted_hot).sum().item())
    denominator = hot_count + predicted_count
    dice = 2.0 * intersection / denominator if denominator else None
    centroid_error = None
    if hot_count and predicted_count:
        rows, columns = torch.where(truth_hot[0, 0])
        truth_center = torch.stack((rows.float().mean(), columns.float().mean()))
        rows, columns = torch.where(predicted_hot[0, 0])
        prediction_center = torch.stack((rows.float().mean(), columns.float().mean()))
        centroid_error = float(torch.linalg.vector_norm(truth_center - prediction_center).item())

    error_k = (prediction_k - target_k)[truth_hot]
    return {
        "reference_pixels": hot_count,
        "predicted_hot_pixel_count": predicted_count,
        "intersection_count": intersection,
        "dice": dice,
        "centroid_displacement_px": centroid_error,
        "pixel_count": int(error_k.numel()),
        "absolute_error_sum_k": float(error_k.abs().sum().item()),
        "squared_error_sum_k2": float(error_k.square().sum().item()),
    }


def _empty_hot_stats():
    return {
        "pixel_count": 0,
        "absolute_error_sum_k": 0.0,
        "squared_error_sum_k2": 0.0,
        "reference_hot_pixel_count": 0,
        "predicted_hot_pixel_count": 0,
        "intersection_count": 0,
        "dice_sum": 0.0,
        "dice_valid_frames": 0,
        "frames": 0,
        "reference_nonempty_frames": 0,
        "detected_reference_frames": 0,
        "false_positive_frames": 0,
        "centroid_displacement_sum_px": 0.0,
        "centroid_valid_frames": 0,
    }


def _merge_hot_stats(total, value):
    total["pixel_count"] += value["pixel_count"]
    total["absolute_error_sum_k"] += value["absolute_error_sum_k"]
    total["squared_error_sum_k2"] += value["squared_error_sum_k2"]
    total["reference_hot_pixel_count"] += value["reference_pixels"]
    total["predicted_hot_pixel_count"] += value["predicted_hot_pixel_count"]
    total["intersection_count"] += value["intersection_count"]
    total["frames"] += 1
    reference_count = value["reference_pixels"]
    predicted_count = value["predicted_hot_pixel_count"]
    total["reference_nonempty_frames"] += int(reference_count > 0)
    total["detected_reference_frames"] += int(reference_count > 0 and predicted_count > 0)
    total["false_positive_frames"] += int(reference_count == 0 and predicted_count > 0)
    if value["dice"] is not None:
        total["dice_sum"] += value["dice"]
        total["dice_valid_frames"] += 1
    if value["centroid_displacement_px"] is not None:
        total["centroid_displacement_sum_px"] += value["centroid_displacement_px"]
        total["centroid_valid_frames"] += 1


def _finish_hot_stats(total):
    count = total["pixel_count"]
    if total["frames"] == 0:
        return {
            "intensity_error_pixel_count": 0, "mae_k": None, "rmse_k": None,
            "frames": 0,
            "reference_nonempty_frames": 0,
            "reference_coverage": None,
            "detected_reference_frames": 0,
            "detection_coverage_given_reference": None,
            "false_positive_frames": 0,
            "reference_empty_prediction_nonempty_frames": 0,
            "dice_valid_frames": 0,
            "mean_dice_nonempty_union": None,
            "centroid_valid_frames": 0,
            "centroid_valid_coverage": None,
            "mean_centroid_displacement_px": None,
            "reference_hot_pixel_count": 0,
            "predicted_hot_pixel_count": 0,
            "intersection_hot_pixel_count": 0,
            "false_positive_hot_pixel_count": 0,
        }
    return {
        "intensity_error_pixel_count": count,
        "mae_k": total["absolute_error_sum_k"] / count if count else None,
        "rmse_k": (total["squared_error_sum_k2"] / count) ** 0.5 if count else None,
        "frames": total["frames"],
        "reference_nonempty_frames": total["reference_nonempty_frames"],
        "reference_coverage": total["reference_nonempty_frames"] / total["frames"],
        "detected_reference_frames": total["detected_reference_frames"],
        "detection_coverage_given_reference": (
            total["detected_reference_frames"] / total["reference_nonempty_frames"]
            if total["reference_nonempty_frames"] else None
        ),
        "false_positive_frames": total["false_positive_frames"],
        "reference_empty_prediction_nonempty_frames": total["false_positive_frames"],
        "dice_valid_frames": total["dice_valid_frames"],
        "mean_dice_nonempty_union": (
            total["dice_sum"] / total["dice_valid_frames"]
            if total["dice_valid_frames"] else None
        ),
        "centroid_valid_frames": total["centroid_valid_frames"],
        "centroid_valid_coverage": total["centroid_valid_frames"] / total["frames"],
        "mean_centroid_displacement_px": (
            total["centroid_displacement_sum_px"] / total["centroid_valid_frames"]
            if total["centroid_valid_frames"] else None
        ),
        "reference_hot_pixel_count": total["reference_hot_pixel_count"],
        "predicted_hot_pixel_count": total["predicted_hot_pixel_count"],
        "intersection_hot_pixel_count": total["intersection_count"],
        "false_positive_hot_pixel_count": (
            total["predicted_hot_pixel_count"] - total["intersection_count"]
        ),
    }


def _balanced_panel(data, split, temperature_range, panel_size):
    manifest = data.split_manifest()
    frame_start, frame_stop = manifest["split_ranges_half_open"][split]
    frames_per_file = frame_stop - frame_start
    sources = manifest["files"]
    if hasattr(data, "case_sets"):
        allowed_cases = set(data.case_sets()[split])
        sources = [source for source in sources if source["case"] in allowed_cases]
    dataset = data.dataset(split, temperature_range)
    if not sources or frames_per_file <= 0 or len(dataset) != len(sources) * frames_per_file:
        raise ValueError("Evaluation dataset does not match its split manifest")
    panel_size = min(panel_size, len(dataset))
    base_quota, remainder = divmod(panel_size, len(sources))
    indices, samples = [], []
    for file_index, source in enumerate(sources):
        quota = base_quota + int(file_index < remainder)
        if not quota:
            continue
        if quota == 1:
            offsets = [(frames_per_file - 1) // 2]
        else:
            offsets = torch.linspace(0, frames_per_file - 1, quota).round().long().tolist()
        for offset in offsets:
            indices.append(file_index * frames_per_file + offset)
            samples.append({
                "case": source["case"],
                "slice": source["slice"],
                "path": source["path"],
                "frame": frame_start + offset,
            })
    if not indices:
        raise ValueError("Evaluation panel is empty")
    return Subset(dataset, indices), samples


def _evaluate_method(dataset, operator, predictor, method, mask_name, device,
                     temperature_range, limit):
    metric_records = []
    depth_records = defaultdict(list)
    latencies = []
    exact_known_errors = []
    hot_stats = _empty_hot_stats()
    frame_count = 0
    needs_grad = method == "dps"
    for raw_target in _subset(dataset, limit):
        target = _as_frame(raw_target, name="dataset frame").to(device)
        observation = operator.observe(target)
        known = operator.mask(target)
        _sync(device)
        start = perf_counter()
        with torch.set_grad_enabled(needs_grad):
            prediction = predictor(observation)
        _sync(device)
        latencies.append(perf_counter() - start)
        prediction = _as_frame(prediction, name="prediction").to(device).detach()
        if not torch.isfinite(prediction).all():
            raise ValueError(f"{method} produced non-finite values for mask {mask_name}")
        predicted_observation = operator.observe(prediction)
        metric_records.append(
            _frame_metrics(
                prediction,
                target,
                observation,
                predicted_observation,
                known,
                task="inpaint",
                temperature_range=temperature_range,
            )
        )
        _merge_hot_stats(
            hot_stats,
            _hot_region_error(prediction, target, known, temperature_range),
        )
        for depth, value in _depth_mae(
            prediction, target, mask_name, temperature_range
        ).items():
            depth_records[depth].append(value)
        if known.any():
            exact_known_errors.append(
                float((prediction[known] - observation[known]).abs().max().item()
                      * (temperature_range[1] - temperature_range[0]))
            )
        frame_count += 1
    if frame_count == 0:
        raise ValueError("Evaluation split is empty")
    metrics = {}
    for key in metric_records[0]:
        values = [record[key] for record in metric_records if record[key] is not None]
        metrics[key] = sum(values) / len(values) if values else None
    mean_latency = sum(latencies) / len(latencies)
    return {
        "frames": frame_count,
        "metrics": metrics,
        "depth_mae_k": {
            key: sum(values) / len(values) for key, values in sorted(depth_records.items())
        },
        "known_pixel_max_abs_error_k": max(exact_known_errors, default=None),
        "high_temperature_hole_metrics": _finish_hot_stats(hot_stats),
        "mean_inference_seconds_per_frame": mean_latency,
        "inference_frames_per_second": 1.0 / mean_latency if mean_latency > 0 else None,
    }


def _summarize_records(records):
    if not records:
        return {}
    summary = {}
    for key in records[0]:
        values = [record[key] for record in records if record[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    return summary


def _spread_error_summary(spreads, errors):
    if not spreads:
        return {}
    spread = torch.cat(spreads)
    error = torch.cat(errors)
    spread_centered = spread - spread.mean()
    error_centered = error - error.mean()
    denom = spread_centered.square().sum().sqrt() * error_centered.square().sum().sqrt()
    correlation = (
        float((spread_centered * error_centered).sum().item() / denom.item())
        if denom.item() > 0 else None
    )
    quantile_edges = torch.quantile(spread, torch.tensor(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=spread.dtype
    ))
    bin_ids = torch.bucketize(spread, quantile_edges[1:-1], right=True)
    bins = {}
    for index in range(5):
        selected = bin_ids == index
        if selected.any():
            bins[f"q{index + 1}"] = {
                "pixel_count": int(selected.sum().item()),
                "mean_spread_k": float(spread[selected].mean().item()),
                "mean_abs_error_k": float(error[selected].mean().item()),
            }
    return {
        "pixelwise_spread_abs_error_pearson_r": correlation,
        "spread_quintiles": bins,
    }


def _evaluate_residual(dataset, operator, unet, model, scheduler, mask_name, device,
                       temperature_range, steps, draws, seed):
    model.eval().requires_grad_(False)
    unet.eval().requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(seed)
    single_draw_metrics, predictive_mean_metrics = [], []
    single_depth, mean_depth = defaultdict(list), defaultdict(list)
    spreads, errors = [], []
    hot_single = _empty_hot_stats()
    hot_mean = _empty_hot_stats()
    unet_latencies, residual_latencies = [], []
    exact_known = True
    max_mean_known_error = 0.0
    frame_count = 0
    delta_t = temperature_range[1] - temperature_range[0]

    for raw_target in dataset:
        target = _as_frame(raw_target, name="dataset frame").to(device)
        observation = operator.observe(target)
        known = operator.mask(target)
        _sync(device)
        start = perf_counter()
        with torch.no_grad():
            base = unet(observation, operator)
        _sync(device)
        unet_latencies.append(perf_counter() - start)

        samples = []
        for _ in range(draws):
            _sync(device)
            start = perf_counter()
            prediction = sample_inpaint_residual(
                model,
                base,
                observation,
                known,
                scheduler,
                steps=steps,
                generator=generator,
            )
            _sync(device)
            residual_latencies.append(perf_counter() - start)
            prediction = prediction.detach()
            samples.append(prediction)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Residual diffusion produced non-finite values for {mask_name}")
            exact_known = exact_known and torch.equal(prediction[known], observation[known])
            predicted_observation = operator.observe(prediction)
            single_draw_metrics.append(_frame_metrics(
                prediction,
                target,
                observation,
                predicted_observation,
                known,
                task="inpaint",
                temperature_range=temperature_range,
            ))
            _merge_hot_stats(
                hot_single,
                _hot_region_error(prediction, target, known, temperature_range),
            )
            for depth, value in _depth_mae(
                prediction, target, mask_name, temperature_range
            ).items():
                single_depth[depth].append(value)

        stack = torch.stack(samples)
        predictive_mean = stack.mean(dim=0)
        predictive_std = stack.std(dim=0, unbiased=False)
        mean_known_error = float(
            (predictive_mean[known] - observation[known]).abs().max().item()
        )
        max_mean_known_error = max(max_mean_known_error, mean_known_error)
        predicted_observation = operator.observe(predictive_mean)
        predictive_mean_metrics.append(_frame_metrics(
            predictive_mean,
            target,
            observation,
            predicted_observation,
            known,
            task="inpaint",
            temperature_range=temperature_range,
        ))
        _merge_hot_stats(
            hot_mean,
            _hot_region_error(predictive_mean, target, known, temperature_range),
        )
        for depth, value in _depth_mae(
            predictive_mean, target, mask_name, temperature_range
        ).items():
            mean_depth[depth].append(value)
        missing = ~known
        spreads.append((predictive_std[missing] * delta_t).detach().cpu())
        errors.append(((predictive_mean - target).abs()[missing] * delta_t).detach().cpu())
        frame_count += 1

    if frame_count == 0:
        raise ValueError("Evaluation panel is empty")
    spread = torch.cat(spreads)
    return {
        "frames": frame_count,
        "draws_per_frame": draws,
        "single_draw_metrics": _summarize_records(single_draw_metrics),
        "predictive_mean_metrics": _summarize_records(predictive_mean_metrics),
        "single_draw_high_temperature_hole_metrics": _finish_hot_stats(hot_single),
        "predictive_mean_high_temperature_hole_metrics": _finish_hot_stats(hot_mean),
        "single_draw_depth_mae_k": {
            key: sum(values) / len(values) for key, values in sorted(single_depth.items())
        },
        "predictive_mean_depth_mae_k": {
            key: sum(values) / len(values) for key, values in sorted(mean_depth.items())
        },
        "mean_predictive_standard_deviation_k": float(spread.mean().item()),
        "spread_error_relation": _spread_error_summary(spreads, errors),
        "individual_draws_preserve_known_pixels_exactly": exact_known,
        "predictive_mean_max_known_pixel_abs_error_k": max_mean_known_error * delta_t,
        "mean_unet_seconds_per_frame": sum(unet_latencies) / len(unet_latencies),
        "mean_residual_seconds_per_draw": sum(residual_latencies) / len(residual_latencies),
        "residual_draws_per_second": (
            1.0 / (sum(residual_latencies) / len(residual_latencies))
            if residual_latencies and sum(residual_latencies) > 0 else None
        ),
    }


def benchmark(args):
    device = prepare(args.device, args.seed)
    data = _data_for(args)
    temperature_range = data.training_range()
    checkpoint = load_checkpoint(args.checkpoint, "mask_curriculum_unet")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "inpaint":
        raise ValueError("The mask-study checkpoint must be an inpainting U-Net")
    model_config = checkpoint["model_config"]
    model = SpatialUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval().requires_grad_(False)
    dataset, panel_samples = _balanced_panel(
        data, args.split, temperature_range, args.panel_size
    )

    dps_checkpoint = None
    if args.prior_checkpoint is not None:
        dps_checkpoint = load_checkpoint(args.prior_checkpoint, "prior")
        check_provenance(dps_checkpoint, data, temperature_range)
    if args.dps_steps is not None and dps_checkpoint is None:
        raise ValueError("--dps-steps requires --prior-checkpoint")

    residual_checkpoint = None
    residual_model = None
    residual_scheduler = None
    if args.residual_checkpoint is not None:
        residual_checkpoint = load_checkpoint(
            args.residual_checkpoint, "mask_residual_diffusion"
        )
        check_provenance(residual_checkpoint, data, temperature_range)
        if residual_checkpoint.get("task") != "inpaint":
            raise ValueError("Residual checkpoint must be for inpainting")
        if residual_checkpoint.get("mask_sizes") != checkpoint.get("mask_sizes"):
            raise ValueError("Residual and U-Net checkpoints use different mask geometries")
        if residual_checkpoint.get("unet_checkpoint_sha256") != _sha256(args.checkpoint):
            raise ValueError("Residual checkpoint was trained with a different shared U-Net")
        if residual_checkpoint.get("unet_epoch") != checkpoint["epoch"]:
            raise ValueError("Residual checkpoint U-Net epoch does not match --checkpoint")
        residual_model = InpaintResidualDenoiser(
            **residual_checkpoint["model_config"]
        ).to(device)
        residual_model.load_state_dict(residual_checkpoint["weights"])
        residual_model.eval().requires_grad_(False)
        residual_scheduler = inpaint_diffusion_scheduler(
            residual_checkpoint["diffusion_steps"]
        )
        if args.steps > residual_checkpoint["diffusion_steps"]:
            raise ValueError("--steps cannot exceed residual checkpoint diffusion steps")

    dps_steps = None
    if dps_checkpoint:
        dps_steps = args.dps_steps or min(20, dps_checkpoint["diffusion_steps"])
        if dps_steps > dps_checkpoint["diffusion_steps"]:
            raise ValueError("--dps-steps cannot exceed prior checkpoint diffusion steps")

    result = {
        "protocol_version": 1,
        "status": "running",
        "stage": "centered_mask_size_study",
        "split_name": args.split,
        "split": data.split_manifest(),
        "test_case": args.test_case,
        "temperature_range_k": list(temperature_range),
        "mask_sizes": {name: list(size) for name, size in MASK_SIZES.items()},
        "training": {
            "checkpoint": str(args.checkpoint),
            "epoch": checkpoint["epoch"],
            "seed": checkpoint["seed"],
            "train_count": checkpoint["train_count"],
            "mask_assignment": checkpoint["mask_assignment"],
        },
        "evaluation": {
            "seed": args.seed,
            "device": str(device),
            "panel_size": len(panel_samples),
            "panel_selection": "evenly spaced frames within each case/slice file",
            "samples": panel_samples,
            "depth_bins": "inward pixel layers from the rectangular hole boundary",
            "high_temperature_threshold_k": HOT_THRESHOLD_K,
            "high_temperature_metrics_region": "missing pixels with ground truth temperature at or above threshold",
        },
        "models": {},
    }
    if dps_checkpoint:
        result["dps"] = {
            "checkpoint": str(args.prior_checkpoint),
            "epoch": dps_checkpoint["epoch"],
            "steps": dps_steps,
            "guidance": args.dps_guidance,
        }
    if residual_checkpoint:
        result["residual_diffusion"] = {
            "checkpoint": str(args.residual_checkpoint),
            "epoch": residual_checkpoint["epoch"],
            "unet_epoch": residual_checkpoint["unet_epoch"],
            "variant": residual_checkpoint["variant"],
            "model_config": residual_checkpoint["model_config"],
            "steps": args.steps,
            "draws_per_frame": args.draws,
        }

    for name in MASK_SIZES:
        operator = CenteredMask(name)

        def predict_unet(observation, op=operator):
            return model(observation, op)

        methods = {"shared_unet": predict_unet}
        if args.include_harmonic:
            methods["harmonic"] = lambda observation, op=operator: interpolate(
                observation, op, "harmonic"
            )
        if dps_checkpoint:
            settings = SimpleNamespace(
                method="dps",
                task="inpaint",
                seed=args.seed,
                steps=dps_steps,
                guidance=args.dps_guidance,
            )
            methods["dps"] = make_predictor(
                settings, operator, device, dps_checkpoint
            )
        for method, predictor in methods.items():
            print(f"Evaluating {method} with {name} mask {MASK_SIZES[name]}...", flush=True)
            record = _evaluate_method(
                dataset,
                operator,
                predictor,
                "dps" if method == "dps" else method,
                name,
                device,
                temperature_range,
                None,
            )
            result["models"].setdefault(method, {})[name] = record
            if method == "shared_unet":
                model.eval()
        if residual_checkpoint:
            print(
                f"Evaluating residual diffusion with {name} mask {MASK_SIZES[name]} "
                f"({args.draws} draws)...",
                flush=True,
            )
            result["models"].setdefault("residual_diffusion", {})[name] = _evaluate_residual(
                dataset,
                operator,
                model,
                residual_model,
                residual_scheduler,
                name,
                device,
                temperature_range,
                args.steps,
                args.draws,
                args.seed + 10,
            )
        output_path = args.output
        if output_path:
            save_json(output_path, result)

    result["status"] = "complete"
    if args.output:
        save_json(args.output, result)

    print(f"{'method':<14} {'mask':<10} {'RMSE (K)':>10} {'MAE (K)':>10} "
          f"{'SSIM':>8} {'FPS':>9}")
    for method, masks in result["models"].items():
        for name, record in masks.items():
            metrics = (
                record["predictive_mean_metrics"]
                if method == "residual_diffusion" else record["metrics"]
            )
            if method == "residual_diffusion":
                latency = (
                    record["mean_unet_seconds_per_frame"]
                    + args.draws * record["mean_residual_seconds_per_draw"]
                )
                fps = 1.0 / latency if latency > 0 else None
            else:
                fps = record["inference_frames_per_second"]
            print(f"{method:<14} {name:<10} {metrics['rmse_k']:>10.3f} "
                  f"{metrics['mae_k']:>10.3f} {metrics['ssim']:>8.4f} "
                  f"{fps:>9.2f}")
    if args.output:
        print(f"Saved {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train a balanced shared-mask U-Net")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--test-case", help="Use the case-holdout split for this case")
    training.add_argument("--epochs", type=positive_int, default=30)
    training.add_argument("--save-epochs", nargs="+", type=positive_int, default=())
    training.add_argument("--batch-size", type=positive_int, default=4)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--unet-channels", type=positive_int, default=32)
    training.add_argument("--train-limit", type=positive_int)
    training.add_argument("--val-limit", type=positive_int)
    training.set_defaults(run=train)

    residual_training = commands.add_parser(
        "train-residual", help="Train conditional residual diffusion on the same balanced masks"
    )
    residual_training.add_argument("--data", type=Path, required=True)
    residual_training.add_argument("--unet-checkpoint", type=Path, required=True)
    residual_training.add_argument("--output", type=Path, required=True)
    residual_training.add_argument("--test-case", help="Use the case-holdout split for this case")
    residual_training.add_argument("--variant", choices=("conv", "transformer"), default="conv")
    residual_training.add_argument("--attention-blocks", type=nonnegative_int, default=0)
    residual_training.add_argument("--epochs", type=positive_int, default=30)
    residual_training.add_argument("--save-epochs", nargs="+", type=positive_int, default=())
    residual_training.add_argument("--batch-size", type=positive_int, default=2)
    residual_training.add_argument("--workers", type=int, default=0)
    residual_training.add_argument("--learning-rate", type=float, default=1e-4)
    residual_training.add_argument("--width", type=positive_int, default=32)
    residual_training.add_argument("--residual-scale", type=float, default=16.0)
    residual_training.add_argument("--diffusion-steps", type=diffusion_steps, default=1000)
    residual_training.add_argument("--train-limit", type=positive_int)
    residual_training.add_argument("--val-limit", type=positive_int)
    residual_training.set_defaults(run=train_residual)

    testing = commands.add_parser("benchmark", help="Evaluate every mask size separately")
    testing.add_argument("--data", type=Path, required=True)
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--residual-checkpoint", type=Path,
                         help="Optionally evaluate conditional residual diffusion")
    testing.add_argument("--output", type=Path)
    testing.add_argument("--test-case", help="Use the case-holdout split for this case")
    testing.add_argument("--split", choices=("val", "test"), default="val")
    testing.add_argument("--panel-size", type=positive_int, default=45)
    testing.add_argument("--steps", type=diffusion_steps, default=20)
    testing.add_argument("--draws", type=positive_int, default=4)
    testing.add_argument("--include-harmonic", action="store_true")
    testing.add_argument("--prior-checkpoint", type=Path,
                         help="Optionally evaluate conditional DPS using this prior")
    testing.add_argument("--dps-steps", type=positive_int)
    testing.add_argument("--dps-guidance", type=float, default=1.0)
    testing.set_defaults(run=benchmark)

    for command in (training, residual_training, testing):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
