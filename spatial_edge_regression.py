"""Train and benchmark a deterministic tapered residual regressor for inpainting.

The model learns a correction to a frozen inpainting U-Net. A cosine taper
restricts that correction to the eight grid cells next to the missing-region
boundary, matching the taper used by the residual-diffusion comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
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
    sample_inpaint_residual,
)
from spatial_inpaint_boundary_ablation import (
    _distance_from_boundary,
    _residual_weights,
)
from spatial_inpaint_experiment import (
    _distance_layers,
    _load_unet,
    _region_metrics,
    _sha256,
)
from spatial_sr_diagnostic import validation_panel


class _ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups_in = min(8, input_channels)
        while input_channels % groups_in:
            groups_in -= 1
        groups_out = min(8, output_channels)
        while output_channels % groups_out:
            groups_out -= 1
        self.norm1 = nn.GroupNorm(groups_in, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (self.skip(x) + hidden) * (2**-0.5)


class EdgeResidualRegressor(nn.Module):
    """Time-free convolutional U-Net predicting a frozen-U-Net correction."""

    def __init__(self, width: int = 32):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of eight, at least 8")
        self.width = width
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.enc0 = _ResidualBlock(width, width)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.enc1 = _ResidualBlock(width * 2, width * 2)
        self.down2 = nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1)
        self.enc2 = _ResidualBlock(width * 3, width * 3)
        self.down3 = nn.Conv2d(width * 3, width * 4, 3, stride=2, padding=1)
        self.middle1 = _ResidualBlock(width * 4, width * 4)
        self.middle2 = _ResidualBlock(width * 4, width * 4)
        self.dec2 = _ResidualBlock(width * 7, width * 3)
        self.dec1 = _ResidualBlock(width * 5, width * 2)
        self.dec0 = _ResidualBlock(width * 3, width)
        self.output_norm = nn.GroupNorm(8, width)
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        base: torch.Tensor,
        observation: torch.Tensor,
        known_mask: torch.Tensor,
    ) -> torch.Tensor:
        if base.ndim != 4 or base.shape[1] != 1:
            raise ValueError("base must have shape [B,1,H,W]")
        if observation.shape != base.shape or known_mask.shape != base.shape:
            raise ValueError("base, observation, and known_mask must have matching shapes")
        if known_mask.dtype != torch.bool:
            raise ValueError("known_mask must be boolean")
        x = torch.cat((base, observation, known_mask.to(dtype=base.dtype)), dim=1)
        skip0 = self.enc0(self.input(x))
        skip1 = self.enc1(self.down1(skip0))
        skip2 = self.enc2(self.down2(skip1))
        middle = self.middle2(self.middle1(self.down3(skip2)))
        middle = F.interpolate(middle, size=skip2.shape[-2:], mode="nearest")
        middle = self.dec2(torch.cat((middle, skip2), dim=1))
        middle = F.interpolate(middle, size=skip1.shape[-2:], mode="nearest")
        middle = self.dec1(torch.cat((middle, skip1), dim=1))
        middle = F.interpolate(middle, size=skip0.shape[-2:], mode="nearest")
        middle = self.dec0(torch.cat((middle, skip0), dim=1))
        return self.output(F.silu(self.output_norm(middle)))


def _load_unet_for_run(path: Path, data: TemperatureData, temperature_range, device):
    checkpoint, predictor = _load_unet(path, data, temperature_range, device)
    return checkpoint, predictor


def _edge_taper(known_mask: torch.Tensor, width: int = 8) -> torch.Tensor:
    distance = _distance_from_boundary(known_mask)
    return _residual_weights(distance, ~known_mask, width)


def _compose_prediction(base, correction, observation, known_mask):
    return torch.where(known_mask, observation, base + correction)


def _edge_prediction(model, base, observation, known_mask, taper):
    raw_correction = model(base, observation, known_mask)
    return _compose_prediction(base, raw_correction * taper, observation, known_mask)


def _sample_metadata(data: TemperatureData, split: str):
    manifest = data.split_manifest()
    frame_start, frame_stop = manifest["split_ranges_half_open"][split]
    return [
        {
            "case": source["case"],
            "slice": source["slice"],
            "path": source["path"],
            "frame": frame,
        }
        for source in manifest["files"]
        for frame in range(frame_start, frame_stop)
    ]


def _hot_band_metrics(prediction, target, missing_mask, temperature_range, threshold_k):
    low, high = temperature_range
    if not low <= threshold_k <= high:
        raise ValueError(
            f"--hot-threshold-k must lie within the training range [{low}, {high}] K"
        )
    threshold = (threshold_k - low) / (high - low)
    region = missing_mask[0, 0]
    reference = (target[0, 0] >= threshold) & region
    predicted = (prediction[0, 0] >= threshold) & region
    reference_count = int(reference.sum().item())
    predicted_count = int(predicted.sum().item())
    if reference_count or predicted_count:
        dice = (2 * int((reference & predicted).sum().item()) / (reference_count + predicted_count))
    else:
        dice = None
    displacement = None
    if reference_count and predicted_count:
        rows = torch.arange(region.shape[0], device=region.device)[:, None]
        cols = torch.arange(region.shape[1], device=region.device)[None, :]
        reference_y = rows.expand_as(region)[reference].float().mean()
        reference_x = cols.expand_as(region)[reference].float().mean()
        predicted_y = rows.expand_as(region)[predicted].float().mean()
        predicted_x = cols.expand_as(region)[predicted].float().mean()
        displacement = float(
            torch.sqrt((reference_y - predicted_y).square() + (reference_x - predicted_x).square()).item()
        )
    return {
        "threshold_k": float(threshold_k),
        "reference_pixels": reference_count,
        "prediction_pixels": predicted_count,
        "dice": dice,
        "centroid_displacement_px": displacement,
    }


def _measure(prediction, target, observation, known_mask, bands, temperature_range, threshold_k):
    operator = SpatialObservation("inpaint")
    metrics = _frame_metrics(
        prediction,
        target,
        observation,
        operator.observe(prediction),
        known_mask,
        task="inpaint",
        temperature_range=temperature_range,
    )
    metrics["distance_layers"] = {
        name: _region_metrics(prediction[0], target[0], mask, temperature_range)
        for name, mask in bands.items()
    }
    metrics["hot_band"] = _hot_band_metrics(
        prediction, target, ~known_mask, temperature_range, threshold_k
    )
    return metrics


def _numeric_summary(rows):
    excluded = {"case", "slice", "path", "frame", "distance_layers", "hot_band"}
    if not rows:
        return {}
    keys = [
        key for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float))
    ]
    return {
        key: mean(float(row[key]) for row in rows if row.get(key) is not None)
        if any(row.get(key) is not None for row in rows) else None
        for key in keys
    }


def _group_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: _numeric_summary(values) for name, values in sorted(groups.items())}


def _distance_summary(rows):
    if not rows:
        return {}
    names = rows[0]["distance_layers"].keys()
    return {
        name: _numeric_summary([
            {"case": row["case"], **row["distance_layers"][name]} for row in rows
        ])
        for name in names
    }


def _hot_band_summary(rows):
    total = len(rows)
    reference_frames = [row["hot_band"] for row in rows if row["hot_band"]["reference_pixels"] > 0]
    predicted_any = [item for item in reference_frames if item["prediction_pixels"] > 0]
    valid_pairs = [
        row["hot_band"] for row in rows
        if row["hot_band"]["reference_pixels"] > 0 and row["hot_band"]["prediction_pixels"] > 0
    ]
    dice_rows = [row["hot_band"]["dice"] for row in rows if row["hot_band"]["dice"] is not None]
    return {
        "threshold_k": rows[0]["hot_band"]["threshold_k"] if rows else None,
        "frames": total,
        "reference_nonempty_frames": len(reference_frames),
        "reference_coverage": len(reference_frames) / total if total else None,
        "detected_reference_frames": len(predicted_any),
        "detection_coverage_given_reference": (
            len(predicted_any) / len(reference_frames) if reference_frames else None
        ),
        "centroid_valid_frames": len(valid_pairs),
        "centroid_valid_coverage": len(valid_pairs) / total if total else None,
        "mean_centroid_displacement_px": (
            mean(row["centroid_displacement_px"] for row in valid_pairs) if valid_pairs else None
        ),
        "dice_valid_frames": len(dice_rows),
        "mean_dice_nonempty_union": mean(dice_rows) if dice_rows else None,
        "reference_empty_prediction_nonempty_frames": sum(
            row["hot_band"]["reference_pixels"] == 0 and row["hot_band"]["prediction_pixels"] > 0
            for row in rows
        ),
    }


def _timed(fn, device):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        torch.cuda.synchronize(device)
        elapsed = start.elapsed_time(end) / 1000.0
    else:
        started = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - started
    return value, elapsed


def _build_method_report(
    rows, elapsed_seconds, *, draws=None, exact_observation=True,
    component_seconds=None, timing_stage="full forward",
):
    return {
        "metrics": _numeric_summary(rows),
        "per_case": _group_summary(rows, "case"),
        "distance_layers": _distance_summary(rows),
        "hot_band_metrics": _hot_band_summary(rows),
        "per_frame": rows,
        "timing": {
            "total_seconds": float(sum(elapsed_seconds)),
            "mean_seconds_per_frame_end_to_end": (
                float(mean(elapsed_seconds)) if elapsed_seconds else None
            ),
            "component_seconds": (
                float(sum(component_seconds)) if component_seconds is not None else None
            ),
            "mean_component_seconds_per_frame_or_draw": (
                float(mean(component_seconds))
                if component_seconds else None
            ),
            "draws_per_frame": draws,
            "timed_stage": timing_stage,
            "warmup_runs_per_model": 1,
            "cuda_synchronized": True,
            "base_unet_forward_included": True,
        },
        "preserves_observed_pixels_exactly": exact_observation,
    }


def _validate_training_args(args):
    if not isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if args.taper_width != 8:
        raise ValueError("This stage is defined with the matching 8-cell taper")


def diffusion_steps(value):
    steps = positive_int(value)
    if steps < 2:
        raise argparse.ArgumentTypeError("Diffusion sampling requires at least two steps")
    return steps


def train(args):
    _validate_training_args(args)
    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    unet_checkpoint, unet_predictor = _load_unet_for_run(
        args.unet_checkpoint, data, temperature_range, device
    )
    operator = SpatialObservation("inpaint")
    train_data = subset(data.dataset("train", temperature_range), args.train_limit)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=loader_generator,
    )
    panel_data, panel_samples = validation_panel(data, temperature_range)
    if args.val_limit is not None:
        panel_data = Subset(panel_data, range(min(args.val_limit, len(panel_data))))
        panel_samples = panel_samples[:len(panel_data)]
    if not panel_samples:
        raise ValueError("The fixed validation panel is empty")

    model_config = {"width": args.width}
    model = EdgeResidualRegressor(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    taper = _edge_taper(
        operator.mask(torch.zeros((1, 1, 192, 256), device=device, dtype=torch.float32)),
        args.taper_width,
    )
    metadata = {
        "version": CHECKPOINT_VERSION,
        "stage": "edge_regression",
        "task": "inpaint",
        "split": data.split_manifest(),
        "temperature_range": list(temperature_range),
        "train_count": len(train_data),
        "val_panel": "45" if args.val_limit is None else "45_subset",
        "val_panel_count": len(panel_data),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "seed": args.seed,
        "model_config": model_config,
        "taper_width": args.taper_width,
        "taper_definition": "cosine: 1 at distance 1; 0 at distance 8; 0 beyond",
        "learning_rate": args.learning_rate,
        "unet_checkpoint": str(args.unet_checkpoint),
        "unet_checkpoint_sha256": _sha256(args.unet_checkpoint),
        "unet_epoch": unet_checkpoint["epoch"],
        "unet_train_count": unet_checkpoint["train_count"],
        "selection_metric": "mean full-missing-region reconstruction RMSE in Kelvin on fixed validation panel",
    }

    best_rmse = float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for target in train_loader:
            target = target.to(device)
            observation = operator.observe(target)
            known_mask = operator.mask(target).to(dtype=torch.bool)
            missing = ~known_mask
            with torch.no_grad():
                base = unet_predictor(observation)
            optimizer.zero_grad(set_to_none=True)
            correction = model(base, observation, known_mask) * taper
            prediction = _compose_prediction(base, correction, observation, known_mask)
            support = missing & (taper > 0)
            if not support.any():
                raise ValueError("The taper has no trainable pixels")
            loss = (prediction[support] - target[support]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite deterministic edge residual loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_total += loss.detach().item() * len(target)

        model.eval().requires_grad_(False)
        val_rows = []
        with torch.no_grad():
            for raw_frame, sample_info in zip(panel_data, panel_samples):
                target = raw_frame.unsqueeze(0).to(device)
                observation = operator.observe(target)
                known_mask = operator.mask(target).to(dtype=torch.bool)
                base = unet_predictor(observation)
                prediction = _edge_prediction(model, base, observation, known_mask, taper)
                bands = _distance_layers(*target.shape[-2:], known_mask, device)
                val_rows.append({
                    **sample_info,
                    **_measure(
                        prediction, target, observation, known_mask, bands,
                        temperature_range, args.hot_threshold_k,
                    ),
                })
        val_rmse = mean(float(row["rmse_k"]) for row in val_rows)
        epoch_record = {
            "epoch": epoch,
            "train_edge_mse": train_total / len(train_data),
            "val45_rmse_k": val_rmse,
            "val45_mae_k": _numeric_summary(val_rows)["mae_k"],
            "val45_ssim": _numeric_summary(val_rows)["ssim"],
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}: train_edge_mse={epoch_record['train_edge_mse']:.6g} "
            f"val45_rmse_k={val_rmse:.6g}",
            flush=True,
        )

        model.requires_grad_(True)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            checkpoint = {
                **metadata,
                "epoch": epoch,
                "val45_rmse_k": val_rmse,
                "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            save_checkpoint(args.output, checkpoint)
            print(f"Saved best validation checkpoint: {args.output}", flush=True)

    training_report = {
        "method": "deterministic_edge_residual_regression",
        **metadata,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_val45_rmse_k": best_rmse,
        "history": history,
        "checkpoint": str(args.output),
    }
    report_path = args.log_output or args.output.with_suffix(".train.json")
    save_json(report_path, training_report)
    print(json.dumps({
        "best_epoch": best_epoch,
        "best_val45_rmse_k": best_rmse,
        "checkpoint": str(args.output),
        "training_report": str(report_path),
    }, indent=2, allow_nan=False), flush=True)


def _load_edge_checkpoint(path, data, temperature_range, device):
    checkpoint = load_checkpoint(path, "edge_regression")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "inpaint":
        raise ValueError("Edge-regression checkpoint must be trained for inpainting")
    model = EdgeResidualRegressor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval().requires_grad_(False)
    return checkpoint, model


def _load_diffusion_checkpoint(path, data, temperature_range, unet_path, unet_checkpoint, device):
    if path is None:
        return None
    checkpoint = load_checkpoint(path, "inpaint_residual")
    check_provenance(checkpoint, data, temperature_range)
    if checkpoint.get("task") != "inpaint":
        raise ValueError("Diffusion checkpoint must be trained for inpainting")
    if _sha256(unet_path) != checkpoint["unet_checkpoint_sha256"]:
        raise ValueError("Diffusion and edge-regression checkpoints use different U-Net weights")
    if checkpoint.get("unet_epoch") != unet_checkpoint["epoch"]:
        raise ValueError("Diffusion checkpoint uses a different frozen U-Net epoch")
    model = InpaintResidualDenoiser(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval().requires_grad_(False)
    scheduler = inpaint_diffusion_scheduler(checkpoint["diffusion_steps"])
    return checkpoint, model, scheduler


@torch.no_grad()
def benchmark(args):
    device = prepare(args.device, args.seed)
    data = TemperatureData(args.data)
    temperature_range = data.training_range()
    edge_checkpoint = load_checkpoint(args.checkpoint, "edge_regression")
    check_provenance(edge_checkpoint, data, temperature_range)
    unet_path = args.unet_checkpoint or Path(edge_checkpoint["unet_checkpoint"])
    if _sha256(unet_path) != edge_checkpoint["unet_checkpoint_sha256"]:
        raise ValueError("Frozen U-Net checkpoint does not match edge-regression provenance")
    unet_checkpoint, unet_predictor = _load_unet_for_run(
        unet_path, data, temperature_range, device
    )
    if unet_checkpoint["epoch"] != edge_checkpoint["unet_epoch"]:
        raise ValueError("Frozen U-Net epoch does not match edge-regression provenance")
    edge_checkpoint, edge_model = _load_edge_checkpoint(
        args.checkpoint, data, temperature_range, device
    )
    if args.split == "val":
        if args.panel == "45":
            dataset, samples = validation_panel(data, temperature_range)
        else:
            dataset = data.dataset("val", temperature_range)
            samples = _sample_metadata(data, "val")
    else:
        dataset = data.dataset("test", temperature_range)
        samples = _sample_metadata(data, "test")
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        samples = samples[:len(dataset)]
    if not samples:
        raise ValueError("Evaluation dataset is empty")

    diffusion_loaded = _load_diffusion_checkpoint(
        args.diffusion_checkpoint, data, temperature_range, unet_path, unet_checkpoint, device
    )
    if diffusion_loaded is not None:
        diffusion_checkpoint, diffusion_model, diffusion_scheduler = diffusion_loaded
        if args.diffusion_steps > diffusion_scheduler.config.num_train_timesteps:
            raise ValueError("--diffusion-steps exceeds the checkpoint diffusion training length")

    operator = SpatialObservation("inpaint")
    taper = _edge_taper(
        operator.mask(torch.zeros((1, 1, 192, 256), device=device, dtype=torch.float32)),
        edge_checkpoint["taper_width"],
    )
    layers = None
    rows = {"unet": [], "edge_regression": []}
    timing = defaultdict(list)
    exact = {"unet": True, "edge_regression": True}
    diffusion_rows_one = []
    diffusion_rows_mean = []
    diffusion_elapsed = []
    diffusion_one_e2e_elapsed = []
    diffusion_four_e2e_elapsed = []
    diffusion_aggregation_elapsed = []
    edge_component_elapsed = []
    diffusion_single_exact = True
    diffusion_mean_exact = True
    generator = torch.Generator(device=device).manual_seed(args.seed + 10)

    # One untimed warm-up per model keeps first-use CUDA initialization out of
    # the reported per-frame latency. Use a separate RNG so warm-up sampling
    # does not change the benchmark draw sequence.
    warmup_target = dataset[0].unsqueeze(0).to(device)
    warmup_observation = operator.observe(warmup_target)
    warmup_known = operator.mask(warmup_target).to(dtype=torch.bool)
    warmup_base = unet_predictor(warmup_observation)
    _edge_prediction(edge_model, warmup_base, warmup_observation, warmup_known, taper)
    if diffusion_loaded is not None:
        _ = sample_inpaint_residual(
            diffusion_model,
            warmup_base,
            warmup_observation,
            warmup_known,
            diffusion_scheduler,
            steps=args.diffusion_steps,
            generator=torch.Generator(device=device).manual_seed(args.seed + 100),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for index, (raw_frame, sample_info) in enumerate(zip(dataset, samples)):
        target = raw_frame.unsqueeze(0).to(device)
        observation = operator.observe(target)
        known_mask = operator.mask(target).to(dtype=torch.bool)
        if layers is None:
            layers = _distance_layers(*target.shape[-2:], known_mask, device)
        distance_taper = taper

        base, elapsed = _timed(lambda: unet_predictor(observation), device)
        timing["unet"].append(elapsed)
        prediction_rows = {"unet": base}
        edge_prediction, elapsed = _timed(
            lambda: _edge_prediction(edge_model, base, observation, known_mask, distance_taper),
            device,
        )
        edge_component_elapsed.append(elapsed)
        timing["edge_regression"].append(timing["unet"][-1] + elapsed)
        prediction_rows["edge_regression"] = edge_prediction
        for name, prediction in prediction_rows.items():
            exact[name] = exact[name] and torch.equal(prediction[known_mask], observation[known_mask])
            rows[name].append({
                **sample_info,
                **_measure(
                    prediction, target, observation, known_mask, layers,
                    temperature_range, args.hot_threshold_k,
                ),
            })

        if diffusion_loaded is not None:
            diffusion_draw_predictions = []
            frame_sampler_seconds = []
            for draw_index in range(args.diffusion_draws):
                def predict_diffusion():
                    sampled = sample_inpaint_residual(
                        diffusion_model,
                        base,
                        observation,
                        known_mask,
                        diffusion_scheduler,
                        steps=args.diffusion_steps,
                        generator=generator,
                    )
                    correction = (sampled - base) * distance_taper
                    return _compose_prediction(base, correction, observation, known_mask)

                diffusion_prediction, elapsed = _timed(predict_diffusion, device)
                diffusion_elapsed.append(elapsed)
                frame_sampler_seconds.append(elapsed)
                diffusion_draw_predictions.append(diffusion_prediction)
                diffusion_single_exact = (
                    diffusion_single_exact
                    and torch.equal(diffusion_prediction[known_mask], observation[known_mask])
                )
                if draw_index == 0:
                    diffusion_rows_one.append({
                        **sample_info,
                        **_measure(
                            diffusion_prediction, target, observation, known_mask, layers,
                            temperature_range, args.hot_threshold_k,
                        ),
                    })

            diffusion_one_e2e_elapsed.append(timing["unet"][-1] + frame_sampler_seconds[0])
            mean_prediction, aggregation_elapsed = _timed(
                lambda: torch.stack(diffusion_draw_predictions).mean(dim=0), device
            )
            diffusion_aggregation_elapsed.append(aggregation_elapsed)
            diffusion_four_e2e_elapsed.append(
                timing["unet"][-1] + sum(frame_sampler_seconds) + aggregation_elapsed
            )
            diffusion_mean_exact = (
                diffusion_mean_exact
                and torch.equal(mean_prediction[known_mask], observation[known_mask])
            )
            diffusion_rows_mean.append({
                **sample_info,
                **_measure(
                    mean_prediction, target, observation, known_mask, layers,
                    temperature_range, args.hot_threshold_k,
                ),
            })
        if (index + 1) % 10 == 0:
            print(f"Evaluated {index + 1}/{len(samples)} frames", flush=True)

    methods = {
        name: _build_method_report(
            rows[name],
            timing[name],
            exact_observation=exact[name],
            component_seconds=edge_component_elapsed if name == "edge_regression" else None,
            timing_stage=("U-Net forward plus edge correction" if name == "edge_regression" else "U-Net forward only"),
        )
        for name in ("unet", "edge_regression")
    }
    if diffusion_loaded is not None:
        methods["diffusion_taper_8_single_draw"] = _build_method_report(
            diffusion_rows_one,
            diffusion_one_e2e_elapsed,
            draws=1,
            exact_observation=diffusion_single_exact,
            component_seconds=diffusion_elapsed[::args.diffusion_draws],
            timing_stage="U-Net forward plus one residual-diffusion draw and taper",
        )
        methods["diffusion_taper_8_predictive_mean"] = _build_method_report(
            diffusion_rows_mean,
            diffusion_four_e2e_elapsed,
            draws=args.diffusion_draws,
            exact_observation=diffusion_mean_exact,
            component_seconds=[
                sum(diffusion_elapsed[i:i + args.diffusion_draws]) + aggregation_elapsed
                for i, aggregation_elapsed in zip(
                    range(0, len(diffusion_elapsed), args.diffusion_draws),
                    diffusion_aggregation_elapsed,
                )
            ],
            timing_stage=(
                f"U-Net forward plus {args.diffusion_draws} residual-diffusion draws, "
                "taper, and predictive mean"
            ),
        )
        methods["diffusion_taper_8_single_draw"]["timing"]["mean_seconds_per_draw_sampling_only"] = mean(
            diffusion_elapsed[::args.diffusion_draws]
        )
        methods["diffusion_taper_8_predictive_mean"]["timing"]["mean_seconds_per_draw_sampling_only"] = mean(
            diffusion_elapsed
        )
        methods["diffusion_taper_8_predictive_mean"]["timing"]["mean_aggregation_seconds_per_frame"] = mean(
            diffusion_aggregation_elapsed
        )

    report = {
        "method": "frozen_unet_deterministic_edge_regression_and_optional_diffusion",
        "task": "inpaint",
        "split_name": args.split,
        "validation_panel": args.panel if args.split == "val" else None,
        "split": data.split_manifest(),
        "samples": samples,
        "temperature_range_k": list(temperature_range),
        "hot_band_protocol": {
            "threshold_k": args.hot_threshold_k,
            "threshold_type": "predeclared absolute Kelvin threshold",
            "region": "missing rectangle only",
            "centroid_displacement": "Euclidean distance between centroids; undefined when either mask is empty",
            "dice": "both-empty rows excluded; one-empty rows receive Dice 0",
            "detection_coverage": "fraction of reference-nonempty frames with a nonempty predicted band",
        },
        "device": str(device),
        "seed": args.seed,
        "edge_checkpoint": str(args.checkpoint),
        "edge_epoch": edge_checkpoint["epoch"],
        "edge_model_config": edge_checkpoint["model_config"],
        "edge_parameter_count": sum(p.numel() for p in edge_model.parameters()),
        "unet_checkpoint": str(unet_path),
        "unet_checkpoint_sha256": _sha256(unet_path),
        "unet_epoch": unet_checkpoint["epoch"],
        "unet_train_count": unet_checkpoint["train_count"],
        "taper": edge_checkpoint["taper_definition"],
        "diffusion_checkpoint": str(args.diffusion_checkpoint) if args.diffusion_checkpoint else None,
        "diffusion_sampling": (
            {"steps": args.diffusion_steps, "draws_per_frame": args.diffusion_draws}
            if diffusion_loaded is not None else None
        ),
        "methods": methods,
    }
    save_json(args.output, report)
    print(json.dumps({
        name: {
            "rmse_k": method["metrics"].get("rmse_k"),
            "mae_k": method["metrics"].get("mae_k"),
            "ssim": method["metrics"].get("ssim"),
            "boundary_gradient_mae_k_per_grid_cell": method["metrics"].get("boundary_gradient_mae_k_per_grid_cell"),
            "hot_band": method["hot_band_metrics"],
        }
        for name, method in methods.items()
    } | {"saved": str(args.output)}, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train the deterministic edge residual regressor")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--unet-checkpoint", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--log-output", type=Path)
    training.add_argument("--epochs", type=positive_int, default=30)
    training.add_argument("--batch-size", type=positive_int, default=4)
    training.add_argument("--width", type=positive_int, default=32)
    training.add_argument("--taper-width", type=positive_int, default=8)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--train-limit", type=positive_int, help="Smoke runs only")
    training.add_argument("--val-limit", type=positive_int, help="Smoke runs only")
    training.add_argument("--hot-threshold-k", type=float, default=1200.0)
    training.add_argument("--seed", type=int, default=0)
    training.set_defaults(run=train)

    testing = commands.add_parser("benchmark", help="Compare U-Net, edge regressor, and optional tapered diffusion")
    testing.add_argument("--data", type=Path, required=True)
    testing.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--unet-checkpoint", type=Path)
    testing.add_argument("--diffusion-checkpoint", type=Path)
    testing.add_argument("--output", type=Path, required=True)
    testing.add_argument("--split", choices=("val", "test"), default="val")
    testing.add_argument("--panel", choices=("45", "full"), default="45")
    testing.add_argument("--diffusion-steps", type=diffusion_steps, default=50)
    testing.add_argument("--diffusion-draws", type=positive_int, default=4)
    testing.add_argument("--limit", type=positive_int, help="Smoke runs only")
    testing.add_argument("--hot-threshold-k", type=float, default=1200.0)
    testing.add_argument("--seed", type=int, default=0)
    testing.set_defaults(run=benchmark)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
