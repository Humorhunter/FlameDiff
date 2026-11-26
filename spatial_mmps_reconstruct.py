"""Reconstruct single- or multi-field flame snapshots with a shared latent MMPS prior."""

import argparse
from hashlib import sha256
import json
from math import isfinite, pi, sqrt
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import Subset

from model.ViTDecoder import ViTDecoder
from spatial import (
    check_provenance,
    checkpoint_field_ranges,
    checkpoint_fields,
    load_autoencoder,
    load_checkpoint,
    load_prior_model,
    positive_int,
    save_json,
)
from spatial_baselines import interpolate
from spatial_data import TemperatureData, normalize_field_names
from spatial_dps import FIELD_SHAPE, standardize_latent
from spatial_eval import _gradient_metrics, _ssim, _sync
from spatial_mmps_observation import SpatialObservation
from spatial_mmps_sampler import sample_mmps
from spatial_sr_diagnostic import validation_panel


TASKS = ("sensors", "inpaint", "sr")
TASK_DEFAULTS = {
    "sensors": {
        "steps": 20,
        "maxiter": 8,
        "sigma_k": 100.0,
        "sigma_norm": 0.03,
        "warm_strength": 0.0,
        "guidance_max_timestep": 500,
    },
    "inpaint": {
        "steps": 20,
        "maxiter": 16,
        "sigma_k": 10.0,
        "sigma_norm": 0.01,
        "warm_strength": 0.0,
        "guidance_max_timestep": 700,
    },
    "sr": {
        "steps": 50,
        "maxiter": 16,
        "sigma_k": 10.0,
        "sigma_norm": 0.01,
        "warm_strength": 0.1,
        "guidance_max_timestep": None,
    },
}
WARM_DEFAULT = {task: cfg["warm_strength"] for task, cfg in TASK_DEFAULTS.items()}
HOT_THRESHOLDS_K = (1000.0, 1200.0)
CHANNEL_ZONE_THRESHOLDS = {
    "T": (("1000k", 1000.0), ("1200k", 1200.0)),
    "OH": (("2e4", 2e-4), ("5e4", 5e-4)),
    "HO2": (("1e5", 1e-5), ("3e5", 3e-5)),
}
REQUIRED_METRIC_KEYS = frozenset({
    "mae_k",
    "rmse_k",
    "gradient_mae_k_per_grid_cell",
    "boundary_gradient_mae_k_per_grid_cell",
    "ssim",
    "observation_rmse_k",
    *(
        f"hot_{kind}_{int(threshold)}k{unit}"
        for threshold in HOT_THRESHOLDS_K
        for kind, unit in (("mae", "_k"), ("rmse", "_k"), ("iou", ""), ("dice", ""))
    ),
})


def build_scheduler(
    train_steps: int,
    *,
    sampler: str = "ddpm",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    beta_schedule: str = "linear",
    timestep_spacing: str = "linspace",
):
    """Build an epsilon-prediction DDPM or DDIM scheduler."""
    kwargs = dict(
        num_train_timesteps=train_steps,
        prediction_type="epsilon",
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        clip_sample_range=float(clip_sample_range),
        timestep_spacing=timestep_spacing,
    )
    if sampler == "ddpm":
        return DDPMScheduler(**kwargs)
    if sampler == "ddim":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"sampler must be 'ddpm' or 'ddim', got {sampler!r}")


def parse_device(value: str) -> str:
    """Validate CLI --device values: cpu, mps, cuda, or cuda:<non-negative int>."""
    if value in ("cpu", "mps", "cuda") or re.fullmatch(r"cuda:\d+", value):
        return value
    raise argparse.ArgumentTypeError("device must be 'cpu', 'mps', 'cuda', or 'cuda:<index>'")


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


def radial_power_spectrum(field_2d: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute azimuthally averaged 2D scalar power spectrum E(k) and HF energy ratio (r>15)."""
    arr = np.asarray(field_2d, dtype=np.float64)
    arr = arr - arr.mean()
    h, w = arr.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    total_energy = float(power.sum())
    hf_ratio = float(power[radius > 15.0].sum() / total_energy) if total_energy > 0.0 else 0.0

    max_k = min(h, w) // 2
    bins = np.floor(radius + 0.5).astype(np.int64)
    valid = (bins >= 1) & (bins <= max_k)
    sums = np.bincount(bins[valid], weights=power[valid], minlength=max_k + 1)[1:]
    counts = np.bincount(bins[valid], minlength=max_k + 1)[1:].clip(min=1)
    spectrum = sums / counts
    return spectrum, hf_ratio


def spectral_metrics(prediction_2d: torch.Tensor, target_2d: torch.Tensor) -> dict[str, float]:
    """Compute 2D radial power spectrum relative error (Wu et al. 2026 Eq. 10) and HF ratios."""
    pred_np = prediction_2d.detach().cpu().numpy()
    gt_np = target_2d.detach().cpu().numpy()
    spec_pred, hf_pred = radial_power_spectrum(pred_np)
    spec_gt, hf_gt = radial_power_spectrum(gt_np)

    eps = max(float(spec_gt.max()) * 1e-12, 1e-30)
    log_gt = np.log10(np.maximum(spec_gt, eps))
    log_pred = np.log10(np.maximum(spec_pred, eps))
    denom = np.abs(log_gt - float(log_gt.min()))
    active = denom > 1e-6
    if np.any(active):
        rel_err_pct = float(100.0 * np.mean(np.abs(log_gt[active] - log_pred[active]) / denom[active]))
    else:
        rel_err_pct = 0.0
    return {
        "hf_energy_ratio_pred": hf_pred,
        "hf_energy_ratio_gt": hf_gt,
        "log_spectrum_rel_error_pct": rel_err_pct,
    }


def gaussian_crps(mean_pred: torch.Tensor, std_pred: torch.Tensor, target: torch.Tensor, region: torch.Tensor) -> float:
    """Closed-form Gaussian CRPS averaged over ``region`` (lower is better)."""
    mu = mean_pred[region].to(dtype=torch.float64)
    sigma = std_pred[region].to(dtype=torch.float64).clamp_min(1e-12)
    y = target[region].to(dtype=torch.float64)
    z = (y - mu) / sigma
    inv_sqrt_2pi = 1.0 / sqrt(2.0 * pi)
    pdf = torch.exp(-0.5 * z * z) * inv_sqrt_2pi
    cdf = 0.5 * (1.0 + torch.erf(z / sqrt(2.0)))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / sqrt(pi))
    return float(crps.mean().item())


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


def channel_zone_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
    field_name: str,
) -> dict[str, float | None]:
    """Compute thresholded reaction/hot-zone MAE, RMSE, IoU, and Dice for one field."""
    thresholds = CHANNEL_ZONE_THRESHOLDS.get(field_name, ())
    result: dict[str, float | None] = {}
    for label, threshold in thresholds:
        target_hot = (target >= threshold) & region
        pred_hot = (prediction >= threshold) & region
        if bool(target_hot.any().item()):
            diff = (prediction - target)[target_hot]
            result[f"zone_mae_{label}"] = float(diff.abs().mean().item())
            result[f"zone_rmse_{label}"] = float(diff.square().mean().sqrt().item())
        else:
            result[f"zone_mae_{label}"] = None
            result[f"zone_rmse_{label}"] = None
        intersection = int((pred_hot & target_hot).sum().item())
        union = int((pred_hot | target_hot).sum().item())
        denom = int(pred_hot.sum().item()) + int(target_hot.sum().item())
        result[f"zone_iou_{label}"] = float(intersection / union) if union > 0 else None
        result[f"zone_dice_{label}"] = float(2.0 * intersection / denom) if denom > 0 else None
    return result


def _compute_single_channel_metrics(
    pred_c: torch.Tensor,
    target_c: torch.Tensor,
    region_c: torch.Tensor,
    known_c: torch.Tensor | None,
    *,
    field_name: str,
    low: float,
    span: float,
    obs_rmse: float | None,
    legacy_t_keys: bool = False,
) -> dict:
    residual = (pred_c - target_c)[region_c]
    gt_abs_sum = float(target_c[region_c].abs().sum().item())
    nmae_pct = (
        float(100.0 * residual.abs().sum().item() / gt_abs_sum)
        if gt_abs_sum > 1e-30
        else 0.0
    )
    grad_err, bgrad_err = _gradient_metrics(pred_c, target_c, region_c, known_c)
    pred_norm = (pred_c - low) / span
    gt_norm = (target_c - low) / span
    ssim_val = _ssim(pred_norm, gt_norm, region_c)
    spec = spectral_metrics(pred_c[0, 0], target_c[0, 0])

    if legacy_t_keys:
        out = {
            "mae_k": float(residual.abs().mean().item()),
            "nmae_pct": nmae_pct,
            "rmse_k": float(residual.square().mean().sqrt().item()),
            "gradient_mae_k_per_grid_cell": grad_err,
            "boundary_gradient_mae_k_per_grid_cell": bgrad_err,
            "ssim": ssim_val,
            "observation_rmse_k": obs_rmse,
            **spec,
        }
        out.update(hot_zone_metrics(pred_c, target_c, region_c))
        return out

    out = {
        "mae": float(residual.abs().mean().item()),
        "nmae_pct": nmae_pct,
        "rmse": float(residual.square().mean().sqrt().item()),
        "gradient_mae_per_grid_cell": grad_err,
        "boundary_gradient_mae_per_grid_cell": bgrad_err,
        "ssim": ssim_val,
        "observation_rmse": obs_rmse,
        **spec,
    }
    out.update(channel_zone_metrics(pred_c, target_c, region_c, field_name))
    return out


def aggregate_metric_rows(metric_rows: list[dict]) -> dict:
    """Average each metric key across frames, recursing into nested dicts like per_channel."""
    if not metric_rows:
        return {}
    keys: list[str] = []
    seen: set[str] = set()
    for row in metric_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    summary: dict = {}
    for key in keys:
        values = [row[key] for row in metric_rows if row.get(key) is not None]
        if not values:
            summary[key] = None
        elif isinstance(values[0], dict):
            summary[key] = aggregate_metric_rows(values)
        else:
            summary[key] = float(mean(values))
    return summary


def evaluation_panel(data: TemperatureData, split: str, field_ranges):
    """Build the balanced panel (middle frame of `split` for every case/slice)."""
    manifest = data.split_manifest()
    if split == "holdout":
        start, stop = manifest["split_ranges_half_open"]["test"]
        frame = (start + stop) // 2
        sources = [
            {"case": item["case"], "slice": item["slice"], "path": item["relative_path"]}
            for item in data._holdout_files
        ]
    else:
        start, stop = manifest["split_ranges_half_open"][split]
        frame = (start + stop) // 2
        sources = manifest["files"]
    dataset = data.dataset(split, field_ranges)
    if len(dataset) != len(sources) * (stop - start):
        raise ValueError(f"{split} dataset does not match the split manifest")
    indices = [
        file_index * (stop - start) + frame - start
        for file_index in range(len(sources))
    ]
    return Subset(dataset, indices), [
        {"case": source["case"], "slice": source["slice"], "path": source["path"], "frame": frame}
        for source in sources
    ]


def _row_key(sample: dict, seed: int) -> tuple[str, str, int, int]:
    return (str(sample["case"]), str(sample["slice"]), int(sample["frame"]), int(seed))


def _load_resume_rows(
    path: Path,
    *,
    expected_settings: dict,
    checkpoint_sha256: str,
    epoch: int,
    split_name: str = "val",
    target_keys: set[tuple[str, str, int, int]] | None = None,
) -> dict[tuple[str, str, int, int], dict]:
    """Load compatible completed per-frame rows from an existing evaluation JSON."""
    if not path.exists():
        return {}
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"Resume file {path} does not contain a JSON object")

    check_keys = (
        "task",
        "fields",
        "observed_fields",
        "temperature_range_k",
        "field_ranges",
        "measured_count",
        "steps",
        "maxiter",
        "sigma_k",
        "sigma_norm",
        "warm_strength",
        "seed",
        "draws",
        "obs_noise_level",
        "sampler",
        "ddim_eta",
        "clip_sample",
        "beta_schedule",
        "solver_space",
        "guidance_mode",
        "dps_weight",
        "dps_channel_balance",
        "mmps_channel_whiten",
        "mmps_adaptive_weight",
        "guidance_max_timestep",
        "score_clip",
    )
    for key in check_keys:
        if key in expected_settings or key in existing:
            # Allow old pre-upgrade JSON files (missing new keys) to resume when current mode uses legacy defaults.
            if key == "guidance_mode" and key not in existing and expected_settings.get(key) == "mmps":
                continue
            if key in ("dps_weight", "dps_channel_balance", "mmps_adaptive_weight") and key not in existing and expected_settings.get(key) is None:
                continue
            if key == "mmps_channel_whiten" and key not in existing and expected_settings.get(key) is False:
                continue
            if existing.get(key) != expected_settings.get(key):
                raise ValueError(
                    f"Cannot resume {path}: setting mismatch for {key!r} "
                    f"(existing={existing.get(key)!r}, current={expected_settings.get(key)!r})"
                )
    if existing.get("split_name", "val") != split_name:
        raise ValueError(
            f"Cannot resume {path}: split mismatch "
            f"(existing={existing.get('split_name')!r}, current={split_name!r})"
        )
    if existing.get("epoch") != epoch:
        raise ValueError(
            f"Cannot resume {path}: checkpoint epoch mismatch "
            f"(existing={existing.get('epoch')!r}, current={epoch!r})"
        )
    existing_sha = existing.get("checkpoint_sha256")
    if existing_sha != checkpoint_sha256:
        raise ValueError(
            f"Cannot resume {path}: missing or mismatched checkpoint SHA-256"
        )

    is_single_t = expected_settings.get("fields", ["T"]) == ["T"]
    resumed: dict[tuple[str, str, int, int], dict] = {}
    for row in existing.get("per_frame", []):
        for section in ("metrics", "raw_metrics"):
            section_data = row.get(section)
            if not isinstance(section_data, dict):
                raise ValueError(f"Cannot resume {path}: missing {section!r}")
            if is_single_t and not REQUIRED_METRIC_KEYS.issubset(section_data):
                raise ValueError(
                    f"Cannot resume {path}: row for {row.get('case')!r} has an incomplete "
                    f"{section!r} schema; start a fresh output file instead"
                )
            if not is_single_t and "per_channel" not in section_data:
                raise ValueError(
                    f"Cannot resume {path}: multi-field row missing 'per_channel' in {section!r}"
                )
        key = _row_key(row, row["seed"])
        resumed[key] = row

    if target_keys is not None:
        extra_keys = set(resumed) - target_keys
        if extra_keys:
            raise ValueError(
                f"Cannot resume {path}: existing file contains {len(extra_keys)} completed "
                f"frame(s) outside the requested selection (refusing to truncate)"
            )
    return resumed


class TemperatureReconstructor:
    """Keep the autoencoder and unconditional prior frozen across single- or multi-field observations."""

    def __init__(self, args, checkpoint):
        model_config = checkpoint["model_config"]
        if str(checkpoint.get("prior_arch", "vit")) == "vit" and model_config.get("num_feats") != 0:
            raise ValueError("Three-task MMPS requires a prior trained without sensor conditioning")

        self.fields: tuple[str, ...] = checkpoint_fields(checkpoint)
        self.num_fields = len(self.fields)
        self.is_single_temperature = self.fields == ("T",)
        self.field_ranges = checkpoint_field_ranges(checkpoint)
        self.channel_stats = checkpoint.get("channel_stats")

        lows, highs, spans = [], [], []
        for fname in self.fields:
            lo, hi = self.field_ranges[fname]
            sp = hi - lo
            if not all(map(isfinite, (lo, hi, sp))) or sp <= 0:
                raise ValueError(f"Checkpoint bounds for {fname!r} must be finite and increasing")
            lows.append(lo)
            highs.append(hi)
            spans.append(sp)
        self._lows = lows
        self._highs = highs
        self._spans = spans
        # Legacy single-T scalar attributes
        self.low = lows[0]
        self.high = highs[0]
        self.span = spans[0]

        raw_obs_fields = getattr(args, "observed_fields", None)
        if raw_obs_fields is None:
            self.observed_fields = self.fields
        else:
            self.observed_fields = normalize_field_names(raw_obs_fields)
            missing = [f for f in self.observed_fields if f not in self.fields]
            if missing:
                raise ValueError(
                    f"Requested --observed-fields {missing} not present in checkpoint fields {self.fields}"
                )
        self.observed_channels = tuple(self.fields.index(f) for f in self.observed_fields)

        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Select --device cpu or use a CUDA instance.")
            if self.device.index is not None and self.device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {self.device.index} is out of range "
                    f"(found {torch.cuda.device_count()} device(s))"
                )

        self._low_t = torch.tensor(lows, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self._span_t = torch.tensor(spans, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self._obs_low_t = torch.tensor(
            [lows[i] for i in self.observed_channels], dtype=torch.float32, device=self.device
        )
        self._obs_span_t = torch.tensor(
            [spans[i] for i in self.observed_channels], dtype=torch.float32, device=self.device
        )

        self.operator = SpatialObservation(
            args.task,
            num_channels=self.num_fields,
            observed_channels=self.observed_channels,
        )
        self.prior_arch = str(checkpoint.get("prior_arch", "vit"))
        self.prior = load_prior_model(checkpoint, self.device)
        self.autoencoder = load_autoencoder(
            checkpoint["ae_weights"], self.device, ae_config=checkpoint.get("ae_config")
        )
        self.latent_scale = checkpoint.get("latent_scale")
        self.beta_schedule = str(checkpoint.get("beta_schedule", "linear"))
        self.diffusion_steps = int(checkpoint["diffusion_steps"])
        self.channel_stats = checkpoint.get("channel_stats")

        defaults = TASK_DEFAULTS[args.task]
        self.steps = defaults["steps"] if args.steps is None else args.steps
        if not 2 <= self.steps <= self.diffusion_steps:
            raise ValueError("--steps must be between 2 and the checkpoint's diffusion step count")
        if args.maxiter is None:
            # For multi-channel 8-sensor reconstruction (8 * C_obs measurements), scale default
            # Krylov budget to full observation rank so GMRES solves the linear system exactly.
            self.maxiter = (
                defaults["maxiter"] * len(self.observed_channels)
                if args.task == "sensors"
                else defaults["maxiter"]
            )
        else:
            self.maxiter = args.maxiter
        self.seed = args.seed

        sigma_norm_arg = getattr(args, "sigma_norm", None)
        if sigma_norm_arg is not None:
            self.sigma_norm = float(sigma_norm_arg)
            if not isfinite(self.sigma_norm) or self.sigma_norm <= 0:
                raise ValueError("--sigma-norm must be finite and strictly positive")
            self.sigma_k = self.sigma_norm * self.span if self.is_single_temperature else None
        elif self.is_single_temperature:
            self.sigma_k = defaults["sigma_k"] if args.sigma_k is None else args.sigma_k
            if not isfinite(self.sigma_k) or self.sigma_k <= 0:
                raise ValueError("--sigma-k must be finite and strictly positive")
            self.sigma_norm = self.sigma_k / self.span
        else:
            if args.sigma_k is not None:
                # Allow specifying sigma_k in Kelvin units when T is channel 0; convert to normalized scale
                t_span = self.field_ranges["T"][1] - self.field_ranges["T"][0] if "T" in self.field_ranges else self.span
                self.sigma_norm = float(args.sigma_k) / t_span
                self.sigma_k = float(args.sigma_k)
            else:
                self.sigma_norm = float(defaults["sigma_norm"])
                self.sigma_k = None
            if not isfinite(self.sigma_norm) or self.sigma_norm <= 0:
                raise ValueError("Observation sigma must be finite and strictly positive")

        if args.warm_strength is None:
            # Incomplete-physics (unobserved channels) cannot warm-start from spatial interpolation alone
            if len(self.observed_channels) < self.num_fields:
                self.strength = 0.0
            else:
                self.strength = defaults["warm_strength"]
        else:
            self.strength = args.warm_strength
        if not isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("--warm-strength must be between 0 and 1")
        if args.task == "sensors" and self.strength:
            raise ValueError("Eight-sensor reconstruction starts from noise. Use --warm-strength 0.")
        if len(self.observed_channels) < self.num_fields and self.strength:
            raise ValueError(
                "Incomplete-physics reconstruction (unobserved channels) must start from noise (--warm-strength 0)."
            )

        self.sampler = getattr(args, "sampler", "ddpm")
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError("--sampler must be 'ddpm' or 'ddim'")
        self.ddim_eta = float(getattr(args, "ddim_eta", 0.0)) if self.sampler == "ddim" else None
        if self.ddim_eta is not None and (not isfinite(self.ddim_eta) or not 0.0 <= self.ddim_eta <= 1.0):
            raise ValueError("--ddim-eta must be finite and in [0, 1]")

        # Standardized Gaussian latents span ~[-4, +4]; clip_sample_range=4.0 prevents
        # 1/sqrt(alpha_bar) blowup at t=999 (especially under squaredcos_cap_v2) without
        # clipping true standardized latent features.
        raw_clip_range = getattr(args, "clip_sample_range", None)
        if raw_clip_range is not None:
            self.clip_sample_range = float(raw_clip_range)
            if not isfinite(self.clip_sample_range) or self.clip_sample_range <= 0:
                raise ValueError("--clip-sample-range must be finite and strictly positive")
        else:
            self.clip_sample_range = 4.0 if self.latent_scale is not None else 1.0

        if bool(getattr(args, "no_clip_sample", False)):
            self.clip_sample = False
        else:
            self.clip_sample = True if self.latent_scale is not None else bool(checkpoint.get("clip_sample", True))

        self.timestep_spacing = "trailing" if self.beta_schedule == "squaredcos_cap_v2" else "linspace"

        self.solver_space = getattr(args, "solver_space", "auto")
        if self.solver_space not in ("auto", "observation", "latent"):
            raise ValueError("--solver-space must be one of: auto, observation, latent")
        self.guidance_mode = str(getattr(args, "guidance_mode", "mmps"))
        if self.guidance_mode not in ("mmps", "dps"):
            raise ValueError("--guidance-mode must be 'mmps' or 'dps'")
        self.dps_weight = float(getattr(args, "dps_weight", 0.30))
        if not isfinite(self.dps_weight) or self.dps_weight <= 0:
            raise ValueError("--dps-weight must be finite and strictly positive")
        self.dps_channel_balance = bool(getattr(args, "dps_channel_balance", False))
        self.mmps_channel_whiten = bool(getattr(args, "mmps_channel_whiten", False))
        raw_mmps_adap = getattr(args, "mmps_adaptive_weight", None)
        if raw_mmps_adap is None:
            self.mmps_adaptive_weight = None
        else:
            self.mmps_adaptive_weight = float(raw_mmps_adap)
            if not isfinite(self.mmps_adaptive_weight) or self.mmps_adaptive_weight <= 0:
                raise ValueError("--mmps-adaptive-weight must be finite and strictly positive")

        self._dps_obs_weights: torch.Tensor | None = None
        self._mmps_whiten_vec: torch.Tensor | None = None
        if len(self.observed_fields) > 1 and isinstance(self.channel_stats, dict):
            inv_stds = [
                1.0 / max(float(self.channel_stats[fn]["norm_std"]), 1e-4)
                for fn in self.observed_fields
            ]
            if self.dps_channel_balance:
                inv_vars = [v * v for v in inv_stds]
                mean_iv = sum(inv_vars) / len(inv_vars)
                self._dps_obs_weights = torch.tensor(
                    [v / mean_iv for v in inv_vars], dtype=torch.float32, device=self.device
                )
            if self.mmps_channel_whiten:
                # Unit-RMS diagonal whitening W_c proportional to 1 / norm_std_c:
                # transforms isotropic sigma^2 I into exact anisotropic covariance
                # Sigma_y = sigma^2 * W^{-2} proportional to diag(norm_std_c^2)
                # while preserving the intuitive [0, 1] scale of --sigma-norm.
                rms = sqrt(sum(v * v for v in inv_stds) / len(inv_stds))
                self._mmps_whiten_vec = torch.tensor(
                    [v / rms for v in inv_stds], dtype=torch.float32, device=self.device
                )

        raw_gmax = getattr(args, "guidance_max_timestep", None)
        if raw_gmax is None:
            if self.latent_scale is not None:
                # Safe clean clamping inside sample_mmps lets guidance start at 95% of schedule
                # (t <= 950) so macro flame lift-off and boundaries are conditioned early.
                self.guidance_max_timestep = (
                    None if self.strength > 0 else min(950, max(0, int(0.95 * self.diffusion_steps)))
                )
            else:
                self.guidance_max_timestep = defaults["guidance_max_timestep"]
        elif raw_gmax == -1:
            self.guidance_max_timestep = None
        else:
            self.guidance_max_timestep = raw_gmax
        if self.guidance_max_timestep is not None:
            if (
                not isinstance(self.guidance_max_timestep, int)
                or isinstance(self.guidance_max_timestep, bool)
                or not 0 <= self.guidance_max_timestep < self.diffusion_steps
            ):
                raise ValueError("--guidance-max-timestep must be -1 (disabled) or in [0, diffusion_steps - 1]")
        raw_score_clip = getattr(args, "score_clip", None)
        if raw_score_clip is None:
            self.score_clip = 1.0 if self.latent_scale is not None else None
        elif float(raw_score_clip) < 0:
            self.score_clip = None
        else:
            self.score_clip = float(raw_score_clip)
            if not isfinite(self.score_clip) or self.score_clip <= 0:
                raise ValueError("--score-clip must be finite and strictly positive (or -1 to disable)")
        self.draws = int(getattr(args, "draws", 1))
        if self.draws < 1:
            raise ValueError("--draws must be a positive integer")
        self.obs_noise_level = float(getattr(args, "obs_noise_level", 0.0))
        if not isfinite(self.obs_noise_level) or self.obs_noise_level < 0.0:
            raise ValueError("--obs-noise-level must be finite and non-negative")

    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if self.operator.task == "sensors":
            if self.num_fields == 1:
                return (observation - self.low) / self.span
            lo = self._obs_low_t.view(1, -1, 1)
            sp = self._obs_span_t.view(1, -1, 1)
            return (observation - lo) / sp
        lo = self._obs_low_t.view(1, -1, 1, 1)
        sp = self._obs_span_t.view(1, -1, 1, 1)
        return (observation - lo) / sp

    def _denormalize_field(self, field_norm: torch.Tensor) -> torch.Tensor:
        return field_norm * self._span_t + self._low_t

    def _normalize_field(self, field_phys: torch.Tensor) -> torch.Tensor:
        return (field_phys - self._low_t) / self._span_t

    def settings(self):
        result = {
            "task": self.operator.task,
            "fields": list(self.fields),
            "observed_fields": list(self.observed_fields),
            "device": str(self.device),
            "measured_count": self.operator.measured_count,
            "steps": self.steps,
            "maxiter": self.maxiter,
            "sigma_k": self.sigma_k,
            "sigma_norm": self.sigma_norm,
            "warm_strength": self.strength,
            "seed": self.seed,
            "draws": self.draws,
            "obs_noise_level": self.obs_noise_level,
            "sampler": self.sampler,
            "ddim_eta": self.ddim_eta,
            "clip_sample": self.clip_sample,
            "clip_sample_range": self.clip_sample_range,
            "beta_schedule": self.beta_schedule,
            "timestep_spacing": self.timestep_spacing,
            "latent_standardized": self.latent_scale is not None,
            "solver_space": self.solver_space,
            "guidance_mode": self.guidance_mode,
            "dps_weight": self.dps_weight if self.guidance_mode == "dps" else None,
            "dps_channel_balance": self.dps_channel_balance if self.guidance_mode == "dps" else None,
            "mmps_channel_whiten": self.mmps_channel_whiten,
            "mmps_adaptive_weight": self.mmps_adaptive_weight if self.guidance_mode == "mmps" else None,
            "guidance_max_timestep": self.guidance_max_timestep,
            "score_clip": self.score_clip,
            "prior_arch": self.prior_arch,
            "ae_arch": getattr(self.autoencoder, "arch", "legacy"),
            "latent_channels": getattr(self.autoencoder, "latent_channels", 1),
            "ae_output_activation": getattr(self.autoencoder, "output_activation", "relu"),
            "ae_latent_activation": getattr(self.autoencoder, "latent_activation", "tanh"),
        }
        if self.is_single_temperature:
            result["temperature_range_k"] = [self.low, self.high]
        else:
            result["field_ranges"] = {k: list(v) for k, v in self.field_ranges.items()}
        if self.operator.task == "sensors":
            result["sensor_coordinates_xy"] = self.operator.sensor_coords_xy
        return result

    def reconstruct(self, observation, *, seed):
        """Accept one observation in physical units and return projected field, raw field and diagnostics."""
        self.operator.validate_observation(observation, finite=True)
        if observation.shape[0] != 1:
            raise ValueError("MMPS reconstructs one snapshot at a time")
        normalized = self._normalize_observation(observation)
        if self.operator.task == "inpaint":
            zero = torch.zeros((), dtype=normalized.dtype, device=normalized.device)
            normalized = torch.where(self.operator.observation_mask(normalized), normalized, zero)

        measure = self.operator.measure
        measured = normalized
        dps_weights = self._dps_obs_weights
        whiten_vec = self._mmps_whiten_vec if self.guidance_mode == "mmps" else None
        if self.operator.task == "inpaint":
            # Restrict the Krylov/DPS system to known pixels on observed channels only.
            obs_mask = self.operator.observation_mask(normalized)
            obs_mask_flat = obs_mask[0].flatten()
            indices = obs_mask_flat.nonzero().flatten()

            if whiten_vec is not None:
                whiten_flat = whiten_vec.view(1, -1, 1, 1).expand_as(normalized).flatten(1).index_select(1, indices)

                def measure(field):
                    obs_part = self.operator.measure(field)
                    return obs_part.flatten(1).index_select(1, indices) * whiten_flat

                measured = normalized.flatten(1).index_select(1, indices) * whiten_flat
            else:
                def measure(field):
                    obs_part = self.operator.measure(field)
                    return obs_part.flatten(1).index_select(1, indices)

                measured = normalized.flatten(1).index_select(1, indices)
            if dps_weights is not None:
                weight_grid = dps_weights.view(1, -1, 1, 1).expand_as(normalized)
                dps_weights = weight_grid.flatten(1).index_select(1, indices)
        elif whiten_vec is not None:
            # Sensors ([1, C_obs, 8]) or SR ([1, C_obs, 48, 64]): exact diagonal channel whitening
            whiten_shape = (1, -1) + (1,) * (normalized.ndim - 2)
            whiten_b = whiten_vec.view(whiten_shape)

            def measure(field):
                return self.operator.measure(field) * whiten_b

            measured = normalized * whiten_b

        initial = {}
        if self.strength:
            method = "harmonic" if self.operator.task == "inpaint" else "bicubic"
            with torch.no_grad():
                base = interpolate(normalized, self.operator, method)
                base = self.operator.project(base, normalized)
                init_z = standardize_latent(self.autoencoder.encoder(base), self.latent_scale)
                initial = {
                    "initial_latent": init_z,
                    "strength": self.strength,
                }

        diagnostics = []
        scheduler = build_scheduler(
            self.diffusion_steps,
            sampler=self.sampler,
            clip_sample=self.clip_sample,
            clip_sample_range=self.clip_sample_range,
            beta_schedule=self.beta_schedule,
            timestep_spacing=self.timestep_spacing,
        )
        raw_normalized = sample_mmps(
            self.prior,
            self.autoencoder.decoder,
            measure,
            measured,
            scheduler=scheduler,
            steps=self.steps,
            sigma=self.sigma_norm,
            maxiter=self.maxiter,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            latent_channels=getattr(self.autoencoder, "latent_channels", None),
            field_channels=self.num_fields,
            latent_scale=self.latent_scale,
            solver_space=self.solver_space,
            guidance_mode=self.guidance_mode,
            dps_weight=self.dps_weight,
            mmps_adaptive_weight=self.mmps_adaptive_weight,
            channel_weights=dps_weights,
            guidance_max_timestep=self.guidance_max_timestep,
            score_clip=self.score_clip,
            eta=self.ddim_eta,
            diagnostics=diagnostics.append,
            **initial,
        )
        raw = self._denormalize_field(raw_normalized)
        projected = self.operator.project(raw, observation)
        self.operator.validate_field(projected, finite=True)

        steps = [record for record in diagnostics if record["phase"] == "step"]
        guided_steps = [step for step in steps if step.get("guided", True)]
        residuals = [step["relative_residual"] for step in guided_steps]
        clean_oob = [
            step["clean_out_of_unit_box_fraction"]
            for step in steps
            if "clean_out_of_unit_box_fraction" in step
        ]
        posterior_clipped = [
            step["posterior_clipped_fraction"]
            for step in steps
            if "posterior_clipped_fraction" in step
        ]
        score_ratios = [
            step["score_to_noise_norm_ratio"]
            for step in guided_steps
            if "score_to_noise_norm_ratio" in step
        ]
        resolved_spaces = sorted({
            step["solver_space"] for step in guided_steps if "solver_space" in step
        })
        obs_before_norm = self.observation_rmse_norm(raw, observation)
        obs_after_norm = self.observation_rmse_norm(projected, observation)
        diag_out = {
            "observation_rmse_norm_before_projection": obs_before_norm,
            "observation_rmse_norm_after_projection": obs_after_norm,
            "total_diffusion_steps": len(steps),
            "solver_steps": len(guided_steps),
            "solver_converged_steps": sum(bool(step["converged"]) for step in guided_steps),
            "solver_space_used": resolved_spaces[0] if len(resolved_spaces) == 1 else resolved_spaces,
            "solver_median_relative_residual": median(residuals) if residuals else None,
            "solver_max_relative_residual": max(residuals) if residuals else None,
            "max_clean_out_of_unit_box_fraction": max(clean_oob) if clean_oob else None,
            "max_posterior_clipped_fraction": max(posterior_clipped) if posterior_clipped else None,
            "max_score_to_noise_norm_ratio": max(score_ratios) if score_ratios else None,
        }
        if self.is_single_temperature:
            diag_out["observation_rmse_k_before_projection"] = self.observation_rmse(raw, observation)
            diag_out["observation_rmse_k_after_projection"] = self.observation_rmse(projected, observation)
        return projected, raw, diag_out

    def observation_rmse(self, field, observation):
        residual = self.operator.measure(field) - observation
        if self.operator.task == "inpaint":
            residual = residual[self.operator.observation_mask(observation)]
        return float(residual.square().mean().sqrt().item())

    def observation_rmse_norm(self, field, observation):
        residual = self._normalize_observation(self.operator.measure(field)) - self._normalize_observation(observation)
        if self.operator.task == "inpaint":
            residual = residual[self.operator.observation_mask(observation)]
        return float(residual.square().mean().sqrt().item())

    def metrics(self, prediction, target, observation):
        if getattr(self, "is_single_temperature", True):
            if self.operator.task == "sr":
                region = torch.ones_like(target, dtype=torch.bool)
                known = None
            else:
                known_mask = self.operator.mask(target)
                region = ~known_mask
                known = known_mask if self.operator.task == "inpaint" else None
            return _compute_single_channel_metrics(
                prediction,
                target,
                region,
                known,
                field_name="T",
                low=self.low,
                span=self.span,
                obs_rmse=self.observation_rmse(prediction, observation),
                legacy_t_keys=True,
            )

        # Multi-field evaluation: compute complete per-channel metrics + clean summary
        full_known = self.operator.mask(target)  # [1, C, H, W]
        pred_obs = self.operator.measure(prediction)
        obs_mask = self.operator.observation_mask(observation) if self.operator.task == "inpaint" else None

        per_channel: dict[str, dict] = {}
        for c_idx, fname in enumerate(self.fields):
            pred_c = prediction[:, c_idx : c_idx + 1]
            target_c = target[:, c_idx : c_idx + 1]
            is_obs_channel = c_idx in self.observed_channels
            if self.operator.task == "inpaint" and is_obs_channel:
                known_c = full_known[:, c_idx : c_idx + 1]
                region_c = ~known_c
            elif self.operator.task == "sensors" and is_obs_channel:
                known_c = None
                region_c = ~full_known[:, c_idx : c_idx + 1]
            else:
                known_c = None
                region_c = torch.ones_like(target_c, dtype=torch.bool)

            obs_rmse_c = None
            if is_obs_channel:
                obs_pos = self.observed_channels.index(c_idx)
                if self.operator.task == "sensors":
                    diff_c = pred_obs[:, obs_pos] - observation[:, obs_pos]
                elif self.operator.task == "inpaint":
                    diff_c = (pred_obs[:, obs_pos : obs_pos + 1] - observation[:, obs_pos : obs_pos + 1])[
                        obs_mask[:, obs_pos : obs_pos + 1]
                    ]
                else:
                    diff_c = pred_obs[:, obs_pos : obs_pos + 1] - observation[:, obs_pos : obs_pos + 1]
                obs_rmse_c = float(diff_c.square().mean().sqrt().item())

            per_channel[fname] = _compute_single_channel_metrics(
                pred_c,
                target_c,
                region_c,
                known_c,
                field_name=fname,
                low=self._lows[c_idx],
                span=self._spans[c_idx],
                obs_rmse=obs_rmse_c,
                legacy_t_keys=False,
            )

        return {
            "mean_nmae_pct": float(mean(m["nmae_pct"] for m in per_channel.values())),
            "mean_ssim": float(mean(m["ssim"] for m in per_channel.values() if m["ssim"] is not None)),
            "mean_log_spectrum_rel_error_pct": float(
                mean(m["log_spectrum_rel_error_pct"] for m in per_channel.values())
            ),
            "observation_rmse_norm": self.observation_rmse_norm(prediction, observation),
            "per_channel": per_channel,
        }


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
    split_name: str = "val",
    peak_cuda_memory_mb: float | None = None,
    panel_description: str = "middle slice and validation frame of each selected case",
) -> dict:
    """Assemble the evaluation JSON payload for incremental or final saves."""
    summary_metrics = aggregate_metric_rows([row["metrics"] for row in rows])
    summary_raw_metrics = aggregate_metric_rows([row["raw_metrics"] for row in rows])
    ensemble_rows = [row["ensemble_metrics"] for row in rows if row.get("ensemble_metrics") is not None]
    summary_ensemble_metrics = aggregate_metric_rows(ensemble_rows) if ensemble_rows else None
    uncertainty_rows = [row["uncertainty"] for row in rows if row.get("uncertainty") is not None]
    summary_uncertainty = aggregate_metric_rows(uncertainty_rows) if uncertainty_rows else None

    residuals = [
        row["solver_median_relative_residual"]
        for row in rows
        if row.get("solver_median_relative_residual") is not None
    ]
    clean_oob = [
        row["max_clean_out_of_unit_box_fraction"]
        for row in rows
        if row.get("max_clean_out_of_unit_box_fraction") is not None
    ]
    posterior_clipped = [
        row["max_posterior_clipped_fraction"]
        for row in rows
        if row.get("max_posterior_clipped_fraction") is not None
    ]
    score_ratios = [
        row["max_score_to_noise_norm_ratio"]
        for row in rows
        if row.get("max_score_to_noise_norm_ratio") is not None
    ]

    summary_obs = {
        "mean_observation_rmse_norm_before_projection": (
            float(mean(row["observation_rmse_norm_before_projection"] for row in rows))
            if rows and "observation_rmse_norm_before_projection" in rows[0]
            else None
        ),
        "mean_observation_rmse_norm_after_projection": (
            float(mean(row["observation_rmse_norm_after_projection"] for row in rows))
            if rows and "observation_rmse_norm_after_projection" in rows[0]
            else None
        ),
        "total_solver_steps": sum(int(row["solver_steps"]) for row in rows),
        "total_solver_converged_steps": sum(int(row["solver_converged_steps"]) for row in rows),
        "median_solver_relative_residual": float(median(residuals)) if residuals else None,
        "max_clean_out_of_unit_box_fraction": float(max(clean_oob)) if clean_oob else None,
        "max_posterior_clipped_fraction": float(max(posterior_clipped)) if posterior_clipped else None,
        "max_score_to_noise_norm_ratio": float(max(score_ratios)) if score_ratios else None,
    }
    if rows and "observation_rmse_k_before_projection" in rows[0]:
        summary_obs["mean_observation_rmse_k_before_projection"] = float(
            mean(row["observation_rmse_k_before_projection"] for row in rows)
        )
        summary_obs["mean_observation_rmse_k_after_projection"] = float(
            mean(row["observation_rmse_k_after_projection"] for row in rows)
        )

    payload = {
        **settings,
        "status": status,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": epoch,
        "split_name": split_name,
        "panel": panel_description,
        "evaluation_region": {
            "sensors": "unmeasured_pixels",
            "inpaint": "missing_rectangle",
            "sr": "full_field",
        }[task],
        "target_frames": target_frames,
        "frames": len(rows),
        "mean_rmse_k": summary_metrics.get("rmse_k") if "rmse_k" in summary_metrics else (
            summary_metrics.get("per_channel", {}).get("T", {}).get("rmse")
        ),
        "mean_raw_rmse_k": summary_raw_metrics.get("rmse_k") if "rmse_k" in summary_raw_metrics else (
            summary_raw_metrics.get("per_channel", {}).get("T", {}).get("rmse")
        ),
        "mean_seconds_per_frame": float(mean(row["seconds"] for row in rows)) if rows else None,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "summary_metrics": summary_metrics,
        "summary_raw_metrics": summary_raw_metrics,
        "summary_observation": summary_obs,
        "per_frame": rows,
    }
    if summary_ensemble_metrics is not None:
        payload["summary_ensemble_metrics"] = summary_ensemble_metrics
    if summary_uncertainty is not None:
        payload["summary_uncertainty"] = summary_uncertainty
    return payload


def reconstruct(args):
    checkpoint = load_checkpoint(args.checkpoint, "prior")
    model = TemperatureReconstructor(args, checkpoint)
    if args.output.resolve() == args.observation.resolve():
        raise ValueError("--output must differ from the observation path")
    value = np.load(args.observation, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise ValueError("Expected a real numeric .npy observation in physical units")
    expected_shape = model.operator.shape
    if model.is_single_temperature:
        file_shape = expected_shape[1:] if args.task != "sensors" else expected_shape
    else:
        file_shape = expected_shape
    if value.shape != file_shape:
        raise ValueError(f"{args.task} expects an observation of shape {file_shape}, got {value.shape}")
    observation = torch.as_tensor(value.astype(np.float32), device=model.device).reshape(1, *expected_shape)
    field, _, diagnostics = model.reconstruct(observation, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        out_arr = field[0, 0].cpu().numpy() if model.is_single_temperature else field[0].cpu().numpy()
        np.save(destination, out_arr)
    result = {
        **model.settings(),
        **diagnostics,
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "field_shape": list(FIELD_SHAPE[1:]) if model.is_single_temperature else [model.num_fields, *FIELD_SHAPE[1:]],
        "saved": str(args.output),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


def _add_observation_noise(
    observation: torch.Tensor,
    model: TemperatureReconstructor,
    channel_stats: dict[str, dict[str, float]],
    *,
    noise_level: float,
    seed: int,
) -> torch.Tensor:
    if noise_level <= 0.0:
        return observation
    gen = torch.Generator(device=observation.device).manual_seed(seed + 7919)
    stds = [float(channel_stats[fname]["std"]) * noise_level for fname in model.observed_fields]
    if model.operator.task == "sensors" and model.num_fields == 1:
        scale = torch.tensor(stds[0], dtype=observation.dtype, device=observation.device)
    elif model.operator.task == "sensors":
        scale = torch.tensor(stds, dtype=observation.dtype, device=observation.device).view(1, -1, 1)
    else:
        scale = torch.tensor(stds, dtype=observation.dtype, device=observation.device).view(1, -1, 1, 1)
    noise = torch.randn(observation.shape, dtype=observation.dtype, device=observation.device, generator=gen) * scale
    noisy = observation + noise
    if model.operator.task == "inpaint":
        zero = torch.zeros((), dtype=observation.dtype, device=observation.device)
        noisy = torch.where(model.operator.observation_mask(observation), noisy, zero)
    return noisy


def evaluate(args):
    checkpoint = load_checkpoint(args.checkpoint, "prior")
    fields = checkpoint_fields(checkpoint)
    holdout_cases = getattr(args, "holdout_cases", None) or checkpoint.get("split", {}).get("holdout_cases")
    data_kwargs = {}
    if fields != ("T",):
        data_kwargs["fields"] = fields
    if holdout_cases:
        data_kwargs["holdout_cases"] = holdout_cases
    data = TemperatureData(args.data, **data_kwargs)
    if fields == ("T",):
        norm_arg = data.training_range()
        check_provenance(checkpoint, data, norm_arg)
    else:
        norm_arg = data.training_ranges()
        check_provenance(checkpoint, data)
    channel_stats = data.training_channel_stats() if hasattr(data, "training_channel_stats") else None

    model = TemperatureReconstructor(args, checkpoint)
    settings = model.settings()
    checkpoint_sha256 = _file_sha256(args.checkpoint)
    split_name = str(getattr(args, "split", "val"))
    if split_name == "val":
        dataset, samples = validation_panel(data, norm_arg)
    else:
        dataset, samples = evaluation_panel(data, split_name, norm_arg)
    panel_mode = str(getattr(args, "panel", "9"))
    if panel_mode == "45":
        selected_indices = list(range(len(samples)))[: args.limit]
        panel_description = f"all five slices middle {split_name} frame of each case"
    else:
        by_case = {}
        for index, sample in enumerate(samples):
            by_case.setdefault(sample["case"], []).append(index)
        selected_indices = [group[len(group) // 2] for group in by_case.values()][: args.limit]
        panel_description = f"middle slice and {split_name} frame of each selected case"

    target_keys = {_row_key(samples[index], args.seed + index) for index in selected_indices}
    resumed_rows = (
        _load_resume_rows(
            args.output,
            expected_settings=settings,
            checkpoint_sha256=checkpoint_sha256,
            epoch=checkpoint["epoch"],
            split_name=split_name,
            target_keys=target_keys,
        )
        if getattr(args, "resume", False)
        else {}
    )
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)

    def _peak_cuda_mb() -> float | None:
        if model.device.type != "cuda":
            return None
        return float(torch.cuda.max_memory_allocated(model.device) / (1024.0 * 1024.0))

    draws = int(getattr(model, "draws", 1))
    obs_noise_level = float(getattr(model, "obs_noise_level", 0.0))
    is_single_t = bool(getattr(model, "is_single_temperature", True))
    model_fields = tuple(getattr(model, "fields", ("T",)))

    rows = []
    for index in selected_indices:
        sample = samples[index]
        seed = args.seed + index
        key = _row_key(sample, seed)
        if key in resumed_rows:
            row = resumed_rows[key]
            rows.append(row)
            print(f"{row['case']}: resumed", flush=True)
            continue

        target_norm = dataset[index].unsqueeze(0).to(model.device)
        if hasattr(model, "_denormalize_field"):
            target = model._denormalize_field(target_norm)
        else:
            target = target_norm * model.span + model.low
        clean_obs = model.operator.measure(target)
        observation = (
            _add_observation_noise(
                clean_obs, model, channel_stats, noise_level=obs_noise_level, seed=seed
            )
            if obs_noise_level > 0.0
            else clean_obs
        )

        draw_projs, draw_raws, draw_diags = [], [], []
        draw_metrics, draw_raw_metrics = [], []
        _sync(model.device)
        started = perf_counter()
        for draw_idx in range(draws):
            draw_seed = seed if draw_idx == 0 else (seed + draw_idx * 10007)
            proj_d, raw_d, diag_d = model.reconstruct(observation, seed=draw_seed)
            draw_projs.append(proj_d)
            draw_raws.append(raw_d)
            draw_diags.append(diag_d)
            draw_metrics.append(model.metrics(proj_d, target, clean_obs))
            draw_raw_metrics.append(model.metrics(raw_d, target, clean_obs))
        _sync(model.device)
        elapsed = (perf_counter() - started) / draws

        agg_diag = dict(draw_diags[0])
        if draws > 1:
            agg_diag["solver_steps"] = sum(int(d["solver_steps"]) for d in draw_diags)
            agg_diag["solver_converged_steps"] = sum(int(d["solver_converged_steps"]) for d in draw_diags)
            if "observation_rmse_norm_before_projection" in agg_diag:
                agg_diag["observation_rmse_norm_before_projection"] = float(
                    mean(d["observation_rmse_norm_before_projection"] for d in draw_diags)
                )
            if "observation_rmse_k_before_projection" in agg_diag:
                agg_diag["observation_rmse_k_before_projection"] = float(
                    mean(d["observation_rmse_k_before_projection"] for d in draw_diags)
                )

        row = {
            **sample,
            "seed": seed,
            "seconds": elapsed,
            **agg_diag,
            "metrics": draw_metrics[0] if draws == 1 else aggregate_metric_rows(draw_metrics),
            "raw_metrics": draw_raw_metrics[0] if draws == 1 else aggregate_metric_rows(draw_raw_metrics),
        }

        if draws > 1:
            stack_raw = torch.cat(draw_raws, dim=0)  # [D, C, H, W]
            ens_raw = stack_raw.mean(dim=0, keepdim=True)
            ens_proj = model.operator.project(ens_raw, observation)
            ens_std = stack_raw.std(dim=0, keepdim=True, unbiased=True)
            row["ensemble_metrics"] = model.metrics(ens_proj, target, clean_obs)
            unc = {}
            full_known = model.operator.mask(target)
            obs_channels = getattr(model, "observed_channels", tuple(range(len(model_fields))))
            for c_idx, fname in enumerate(model_fields):
                is_obs = c_idx in obs_channels
                if model.operator.task in ("inpaint", "sensors") and is_obs:
                    reg = ~full_known[:, c_idx : c_idx + 1]
                else:
                    reg = torch.ones_like(target[:, c_idx : c_idx + 1], dtype=torch.bool)
                std_c = ens_std[:, c_idx : c_idx + 1]
                mean_c = ens_raw[:, c_idx : c_idx + 1]
                gt_c = target[:, c_idx : c_idx + 1]
                unc[fname] = {
                    "mean_predictive_std": float(std_c[reg].mean().item()),
                    "crps": gaussian_crps(mean_c, std_c, gt_c, reg),
                }
            row["uncertainty"] = unc

        rows.append(row)
        if is_single_t:
            print(
                f"{row['case']}: RMSE {row['metrics']['rmse_k']:.2f} K, "
                f"raw observation RMSE {row['observation_rmse_k_before_projection']:.2f} K",
                flush=True,
            )
        else:
            pc = row["metrics"]["per_channel"]
            parts = [f"{fn} NMAE={pc[fn]['nmae_pct']:.1f}%" for fn in model_fields]
            print(f"{row['case']}: " + " | ".join(parts), flush=True)

        partial = build_evaluation_report(
            settings=settings,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            epoch=checkpoint["epoch"],
            task=args.task,
            target_frames=len(selected_indices),
            rows=rows,
            status="running",
            split_name=split_name,
            peak_cuda_memory_mb=_peak_cuda_mb(),
            panel_description=panel_description,
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
        split_name=split_name,
        peak_cuda_memory_mb=_peak_cuda_mb(),
        panel_description=panel_description,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"},
                     indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("reconstruct", help="Reconstruct one field from a physical-unit .npy observation")
    inference.add_argument("--observation", type=Path, required=True)
    inference.set_defaults(run=reconstruct)
    evaluation = commands.add_parser("evaluate", help="Measure accuracy on validation or test frames")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test", "holdout"), default="val", help="Dataset split to evaluate")
    evaluation.add_argument("--holdout-cases", type=str, help="Optional holdout case override for --split holdout")
    evaluation.add_argument("--panel", choices=("9", "45"), default="9", help="Panel size (9=1/case, 45=all slices)")
    evaluation.add_argument("--limit", type=positive_int, help="Maximum number of frames to evaluate")
    evaluation.add_argument("--draws", type=positive_int, default=1, help="Independent draws per frame for ensemble/uncertainty stats")
    evaluation.add_argument("--obs-noise-level", type=float, default=0.0, help="Relative Gaussian observation noise level k (noise std = k * field_std)")
    evaluation.add_argument("--resume", action="store_true", help="Resume from an existing compatible output JSON")
    evaluation.set_defaults(run=evaluate)

    for command in (inference, evaluation):
        command.add_argument("--task", choices=TASKS, required=True)
        command.add_argument("--checkpoint", type=Path, required=True, help="Unconditional spatial prior")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--observed-fields", type=str, help="Comma-separated observed channels subset (e.g. 'T', 'OH', 'T,OH,HO2')")
        command.add_argument("--device", type=parse_device, default="cpu", help="cpu, cuda, or cuda:<index>")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--steps", type=positive_int, help="Diffusion steps (defaults: 20 for sensors/inpaint, 50 for sr)")
        command.add_argument("--sigma-k", type=float, help="MMPS likelihood regularization scale in Kelvin (single-T or converted via T span)")
        command.add_argument("--sigma-norm", type=float, help="MMPS likelihood regularization scale in normalized [0,1] units")
        command.add_argument("--maxiter", type=positive_int, help="GMRES iterations per step (defaults: 8 for sensors, 16 for inpaint/sr)")
        command.add_argument("--warm-strength", type=float, help="Fraction of the schedule to use (defaults: 0 for sensors/inpaint/incomplete-physics, 0.1 for full-channel sr)")
        command.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm", help="Reverse diffusion scheduler")
        command.add_argument("--ddim-eta", type=float, default=0.0, help="Stochasticity eta for --sampler ddim (0=deterministic)")
        command.add_argument("--no-clip-sample", action="store_true", help="Disable scheduler clip_sample")
        command.add_argument("--clip-sample-range", type=float, help="Scheduler clip range (default 4.0 for standardized latents, 1.0 legacy)")
        command.add_argument("--solver-space", choices=("auto", "observation", "latent"), default="auto", help="GMRES linear system space")
        command.add_argument("--guidance-mode", choices=("mmps", "dps"), default="mmps", help="Posterior guidance algorithm (mmps or Wu et al. adaptive normalized dps)")
        command.add_argument("--dps-weight", type=float, default=0.30, help="Adaptive DPS relative step norm weight omega_1")
        command.add_argument("--dps-channel-balance", action="store_true", help="Weight multi-field DPS residuals inversely by channel variance")
        command.add_argument("--mmps-channel-whiten", action="store_true", help="Use exact per-channel whitened covariance Sigma_y ~ diag(norm_std_c^2) in MMPS")
        command.add_argument("--mmps-adaptive-weight", type=float, help="Wu et al.-style adaptive norm matching omega_1 applied to MMPS covariance-inverted score")
        command.add_argument("--guidance-max-timestep", type=int, help="Skip guidance above this timestep (-1 disables default cap)")
        command.add_argument("--score-clip", type=float, help="Cap ||sqrt(1-alpha_bar)*score|| / ||predicted_noise||")

    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must differ from the checkpoint path")
    args.run(args)


if __name__ == "__main__":
    main()
