#!/usr/bin/env python3
"""Deterministic 3-field baselines + representative visual snapshot exporter.

Computes exact apples-to-apples deterministic baselines (8-sensor Ridge clean/noisy,
8-sensor 1-NN, harmonic inpaint, bicubic SR, and AE3 oracle) on the identical
evaluation panel used by spatial_mmps_reconstruct.py, and exports representative
(T, OH, HO2) field snapshots + 4-draw uncertainty std maps for T1030, T1050, T1070.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from spatial_baselines import interpolate
from spatial_data import SPLIT_RANGES, TemperatureData
from spatial_mmps_observation import SpatialObservation
from spatial_mmps_reconstruct import (
    TemperatureReconstructor,
    _add_observation_noise,
    _compute_single_channel_metrics,
    evaluation_panel,
    load_checkpoint,
)


def _panel_selection(data: TemperatureData, split_name: str, ranges: dict, panel_mode: str = "9"):
    dataset, samples = evaluation_panel(data, split_name, ranges)
    if panel_mode == "45":
        selected = list(range(len(samples)))
    else:
        by_case: dict[str, list[int]] = {}
        for idx, sample in enumerate(samples):
            by_case.setdefault(sample["case"], []).append(idx)
        selected = [group[len(group) // 2] for group in by_case.values()]
    return dataset, samples, selected


def _per_channel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    operator: SpatialObservation,
    fields: tuple[str, ...],
    ranges: dict[str, tuple[float, float]],
    observed_channels: tuple[int, ...],
) -> dict[str, dict]:
    full_known = operator.mask(target)
    out: dict[str, dict] = {}
    for c_idx, fname in enumerate(fields):
        pred_c = pred[:, c_idx : c_idx + 1]
        target_c = target[:, c_idx : c_idx + 1]
        is_obs = c_idx in observed_channels
        if operator.task == "inpaint" and is_obs:
            known_c = full_known[:, c_idx : c_idx + 1]
            region_c = ~known_c
        elif operator.task == "sensors" and is_obs:
            known_c = None
            region_c = ~full_known[:, c_idx : c_idx + 1]
        else:
            known_c = None
            region_c = torch.ones_like(target_c, dtype=torch.bool)
        lo, hi = ranges[fname]
        out[fname] = _compute_single_channel_metrics(
            pred_c,
            target_c,
            region_c,
            known_c,
            field_name=fname,
            low=lo,
            span=hi - lo,
            obs_rmse=None,
            legacy_t_keys=False,
        )
    return out


def _aggregate_per_channel(rows: list[dict[str, dict]], fields: tuple[str, ...]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for fname in fields:
        keys = rows[0][fname].keys()
        summary[fname] = {}
        for k in keys:
            vals = [r[fname][k] for r in rows if r[fname][k] is not None]
            summary[fname][k] = float(sum(vals) / len(vals)) if vals else None
    return summary


def _fit_train_sensor_models(
    data: TemperatureData,
    fields: tuple[str, ...],
    ranges: dict[str, tuple[float, float]],
    alpha: float = 1e-2,
):
    """Streaming Ridge fit + compact sensor bank for 1-NN over all train frames."""
    op3 = SpatialObservation("sensors", num_channels=len(fields), observed_channels=tuple(range(len(fields))))
    lows = np.array([ranges[f][0] for f in fields], dtype=np.float32)[:, None, None]
    spans = np.array([ranges[f][1] - ranges[f][0] for f in fields], dtype=np.float32)[:, None, None]
    sx = np.array([x for x, _ in op3.sensor_coords_xy], dtype=np.int64)
    sy = np.array([y for _, y in op3.sensor_coords_xy], dtype=np.int64)
    train_start, train_end = SPLIT_RANGES["train"]
    n_out = len(fields) * 192 * 256

    XtX_3 = np.zeros((25, 25), dtype=np.float64)
    XtY_3 = np.zeros((25, n_out), dtype=np.float64)
    XtX_t = np.zeros((9, 9), dtype=np.float64)
    XtY_t = np.zeros((9, n_out), dtype=np.float64)

    X_list = []
    refs: list[tuple[dict, int]] = []
    for entry in data._files:
        arrays = [np.load(entry["field_paths"][f], mmap_mode="r") for f in fields]
        frames = np.stack([a[train_start:train_end] for a in arrays], axis=1).astype(np.float32)  # [T, C, H, W]
        norm = (frames - lows[None]) / spans[None]
        sens = norm[:, :, sy, sx]  # [T, C, 8]
        X_list.append(sens)
        for fi in range(train_start, train_end):
            refs.append((entry, fi))

        Y_chunk = norm.reshape(frames.shape[0], -1).astype(np.float64)
        ones = np.ones((frames.shape[0], 1), dtype=np.float64)
        X3_aug = np.concatenate([sens.reshape(frames.shape[0], -1).astype(np.float64), ones], axis=1)
        Xt_aug = np.concatenate([sens[:, 0, :].astype(np.float64), ones], axis=1)
        XtX_3 += X3_aug.T @ X3_aug
        XtY_3 += X3_aug.T @ Y_chunk
        XtX_t += Xt_aug.T @ Xt_aug
        XtY_t += Xt_aug.T @ Y_chunk

    reg3 = np.eye(25, dtype=np.float64) * alpha
    reg3[-1, -1] = 0.0
    W_ridge_3ch = np.linalg.solve(XtX_3 + reg3, XtY_3).astype(np.float32)

    regt = np.eye(9, dtype=np.float64) * alpha
    regt[-1, -1] = 0.0
    W_ridge_tonly = np.linalg.solve(XtX_t + regt, XtY_t).astype(np.float32)

    X_all = np.concatenate(X_list, axis=0)  # [N, C, 8]
    return X_all, refs, W_ridge_3ch, W_ridge_tonly, lows, spans


def _load_ref_frame(ref: tuple[dict, int], fields: tuple[str, ...]) -> torch.Tensor:
    entry, fi = ref
    arr = np.stack([np.load(entry["field_paths"][f], mmap_mode="r")[fi] for f in fields], axis=0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _predict_ridge(W: np.ndarray, x_flat: np.ndarray, shape: tuple[int, ...], lows: np.ndarray, spans: np.ndarray) -> torch.Tensor:
    x_aug = np.concatenate([x_flat.astype(np.float32), np.ones((1,), dtype=np.float32)], axis=0)
    y_norm = (x_aug @ W).reshape(shape)
    y_phys = y_norm * spans + lows
    return torch.from_numpy(y_phys).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vis-cases", default="T1030,T1050,T1070")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.checkpoint, "prior")
    fields = tuple(ckpt["fields"])
    data = TemperatureData(args.data, fields=fields)
    ranges = data.training_ranges()
    channel_stats = data.training_channel_stats()
    dataset, samples, selected = _panel_selection(data, args.split, ranges, panel_mode="9")

    # 1. Fit deterministic 8-sensor baselines on ALL 5850 train frames [0, 130)
    print(f"Streaming Ridge + sensor bank over {SPLIT_RANGES['train']} x {len(data._files)} slices...", flush=True)
    X_train, train_refs, W_ridge_3ch, W_ridge_tonly, lows, spans = _fit_train_sensor_models(data, fields, ranges, alpha=1e-2)
    N_train = X_train.shape[0]
    X_3ch = X_train.reshape(N_train, -1)  # [N, 24]
    X_tonly = X_train[:, 0, :]  # [N, 8]
    print(f"Fitted Ridge & sensor bank on {N_train} training frames.", flush=True)

    # Standardize 1-NN coordinates by train std so T/OH/HO2 contribute equally
    x3_mean, x3_std = X_3ch.mean(axis=0, keepdims=True), np.maximum(X_3ch.std(axis=0, keepdims=True), 1e-6)
    X_3ch_z = (X_3ch - x3_mean) / x3_std
    xt_mean, xt_std = X_tonly.mean(axis=0, keepdims=True), np.maximum(X_tonly.std(axis=0, keepdims=True), 1e-6)
    X_tonly_z = (X_tonly - xt_mean) / xt_std

    op_sens_3ch = SpatialObservation("sensors", num_channels=3, observed_channels=(0, 1, 2))
    op_sens_tonly = SpatialObservation("sensors", num_channels=3, observed_channels=(0,))
    op_inp_3ch = SpatialObservation("inpaint", num_channels=3, observed_channels=(0, 1, 2))
    op_sr_3ch = SpatialObservation("sr", num_channels=3, observed_channels=(0, 1, 2))

    # Build a dummy reconstructor to reuse exact _add_observation_noise and AE3 oracle
    base_args = SimpleNamespace(
        task="sensors",
        observed_fields="T,OH,HO2",
        device=args.device,
        seed=0,
        steps=50,
        maxiter=8,
        sigma_k=None,
        sigma_norm=0.03,
        warm_strength=0.0,
        draws=1,
        obs_noise_level=0.0,
    )
    rec_sens3 = TemperatureReconstructor(base_args, ckpt)

    lows_t = torch.tensor([ranges[f][0] for f in fields], dtype=torch.float32).view(1, 3, 1, 1)
    spans_t = torch.tensor([ranges[f][1] - ranges[f][0] for f in fields], dtype=torch.float32).view(1, 3, 1, 1)

    buckets: dict[str, list[dict]] = {
        "ae3_oracle": [],
        "ridge_sensors_3ch_clean": [],
        "ridge_sensors_3ch_noise10": [],
        "ridge_sensors_3ch_noise30": [],
        "ridge_sensors_tonly_clean": [],
        "nn1_sensors_3ch_clean": [],
        "nn1_sensors_3ch_noise10": [],
        "nn1_sensors_3ch_noise30": [],
        "nn1_sensors_tonly_clean": [],
        "harmonic_inpaint_3ch": [],
        "bicubic_sr_3ch": [],
    }
    vis_case_set = {c.strip() for c in args.vis_cases.split(",") if c.strip()}
    vis_arrays: dict[str, np.ndarray] = {}

    for idx in selected:
        sample = samples[idx]
        case_name = sample["case"]
        norm_target = dataset[idx].unsqueeze(0)  # [1, 3, H, W]
        target = norm_target * spans_t + lows_t
        seed = idx

        # AE3 oracle
        with torch.no_grad():
            ae_norm = rec_sens3.autoencoder(norm_target.to(rec_sens3.device)).cpu()
        ae_pred = ae_norm * spans_t + lows_t
        buckets["ae3_oracle"].append(
            _per_channel_metrics(ae_pred, target, op_sr_3ch, fields, ranges, (0, 1, 2))
        )

        # Clean & noisy 3ch sensor observations (identical to evaluate())
        obs3_clean = op_sens_3ch.measure(target)  # [1, 3, 8]
        obs3_n10 = _add_observation_noise(obs3_clean, rec_sens3, channel_stats, noise_level=0.10, seed=seed)
        obs3_n30 = _add_observation_noise(obs3_clean, rec_sens3, channel_stats, noise_level=0.30, seed=seed)

        for tag, obs_phys in (
            ("clean", obs3_clean),
            ("noise10", obs3_n10),
            ("noise30", obs3_n30),
        ):
            obs_norm = ((obs_phys.numpy()[0] - lows[:, :, 0]) / spans[:, :, 0]).reshape(-1)
            pred_r = _predict_ridge(W_ridge_3ch, obs_norm, (3, 192, 256), lows, spans)
            buckets[f"ridge_sensors_3ch_{tag}"].append(
                _per_channel_metrics(pred_r, target, op_sens_3ch, fields, ranges, (0, 1, 2))
            )
            q_z = (obs_norm[None, :] - x3_mean) / x3_std
            nn_i = int(np.argmin(np.sum((X_3ch_z - q_z) ** 2, axis=1)))
            pred_nn = _load_ref_frame(train_refs[nn_i], fields)
            buckets[f"nn1_sensors_3ch_{tag}"].append(
                _per_channel_metrics(pred_nn, target, op_sens_3ch, fields, ranges, (0, 1, 2))
            )
            if case_name in vis_case_set and tag in ("clean", "noise30"):
                vis_arrays[f"{case_name}__ridge_sensors_3ch_{tag}"] = pred_r[0].numpy().astype(np.float32)

        # T-only 8-sensor Ridge & 1-NN
        obs_t_clean = op_sens_tonly.measure(target)  # [1, 1, 8]
        obs_t_norm = ((obs_t_clean.numpy()[0, 0] - lows[0, 0, 0]) / spans[0, 0, 0]).reshape(-1)
        pred_rt = _predict_ridge(W_ridge_tonly, obs_t_norm, (3, 192, 256), lows, spans)
        buckets["ridge_sensors_tonly_clean"].append(
            _per_channel_metrics(pred_rt, target, op_sens_tonly, fields, ranges, (0,))
        )
        qt_z = (obs_t_norm[None, :] - xt_mean) / xt_std
        nnt_i = int(np.argmin(np.sum((X_tonly_z - qt_z) ** 2, axis=1)))
        pred_nnt = _load_ref_frame(train_refs[nnt_i], fields)
        buckets["nn1_sensors_tonly_clean"].append(
            _per_channel_metrics(pred_nnt, target, op_sens_tonly, fields, ranges, (0,))
        )

        # Harmonic inpaint & Bicubic SR (3ch)
        obs_inp = op_inp_3ch.measure(target)
        pred_harm = interpolate(obs_inp, op_inp_3ch, "harmonic")
        buckets["harmonic_inpaint_3ch"].append(
            _per_channel_metrics(pred_harm, target, op_inp_3ch, fields, ranges, (0, 1, 2))
        )
        obs_sr = op_sr_3ch.measure(target)
        pred_bic = interpolate(obs_sr, op_sr_3ch, "bicubic")
        buckets["bicubic_sr_3ch"].append(
            _per_channel_metrics(pred_bic, target, op_sr_3ch, fields, ranges, (0, 1, 2))
        )

        if case_name in vis_case_set:
            vis_arrays[f"{case_name}__gt"] = target[0].numpy().astype(np.float32)
            vis_arrays[f"{case_name}__ae3_oracle"] = ae_pred[0].numpy().astype(np.float32)
            vis_arrays[f"{case_name}__harmonic_inpaint_3ch"] = pred_harm[0].numpy().astype(np.float32)
            vis_arrays[f"{case_name}__bicubic_sr_3ch"] = pred_bic[0].numpy().astype(np.float32)
            vis_arrays[f"{case_name}__ridge_sensors_tonly_clean"] = pred_rt[0].numpy().astype(np.float32)

    baselines_summary = {
        "split": args.split,
        "panel_frames": len(selected),
        "cases": [samples[i]["case"] for i in selected],
        "train_frames_total": int(N_train),
        "methods": {k: _aggregate_per_channel(v, fields) for k, v in buckets.items()},
    }
    baselines_path = out_dir / f"baselines_3field_{args.split}9.json"
    baselines_path.write_text(json.dumps(baselines_summary, indent=2), encoding="utf-8")
    print(f"Saved {baselines_path}", flush=True)

    # 2. Export representative MMPS 3-field predictions + 4-draw std maps for vis_cases
    modes = [
        ("sensors_3ch", "sensors", "T,OH,HO2", 50, 8, 0.03, 0.0, 4, 0.0),
        ("sensors_3ch_noise30", "sensors", "T,OH,HO2", 50, 8, 0.08, 0.0, 4, 0.30),
        ("sensors_tonly", "sensors", "T", 50, 8, 0.03, 0.0, 4, 0.0),
        ("inpaint_3ch", "inpaint", "T,OH,HO2", 30, 16, 0.015, 0.0, 4, 0.0),
        ("inpaint_tonly", "inpaint", "T", 30, 16, 0.015, 0.0, 4, 0.0),
        ("sr_3ch", "sr", "T,OH,HO2", 50, 16, 0.01, 0.10, 1, 0.0),
        ("sr_tonly", "sr", "T", 50, 16, 0.01, 0.0, 4, 0.0),
        ("sr_ohonly", "sr", "OH", 50, 16, 0.01, 0.0, 4, 0.0),
    ]
    for mode_name, task, obs_f, steps, maxiter, sigma_norm, warm, draws, noise_lvl in modes:
        m_args = SimpleNamespace(
            task=task,
            observed_fields=obs_f,
            device=args.device,
            seed=0,
            steps=steps,
            maxiter=maxiter,
            sigma_k=None,
            sigma_norm=sigma_norm,
            warm_strength=warm,
            draws=draws,
            obs_noise_level=noise_lvl,
        )
        rec = TemperatureReconstructor(m_args, ckpt)
        for idx in selected:
            case_name = samples[idx]["case"]
            if case_name not in vis_case_set:
                continue
            target = dataset[idx].unsqueeze(0) * spans_t + lows_t
            obs = rec.operator.measure(target).to(rec.device)
            if noise_lvl > 0:
                obs = _add_observation_noise(obs, rec, channel_stats, noise_level=noise_lvl, seed=idx)
            preds = []
            for d in range(draws):
                proj, raw, _ = rec.reconstruct(obs, seed=idx + d * 1009)
                # For noisy sensors, use raw (or proj — identical on 49144 unobserved pixels; raw avoids 8 noisy spikes)
                out_tensor = raw if noise_lvl > 0 else proj
                preds.append(out_tensor[0].detach().cpu().numpy().astype(np.float32))
            stack = np.stack(preds, axis=0)  # [D, 3, H, W]
            vis_arrays[f"{case_name}__{mode_name}__draw0"] = stack[0]
            if draws > 1:
                vis_arrays[f"{case_name}__{mode_name}__mean"] = stack.mean(axis=0)
                vis_arrays[f"{case_name}__{mode_name}__std"] = stack.std(axis=0, ddof=1)
            print(f"[vis] {case_name} {mode_name} (draws={draws}) done", flush=True)

    npz_path = out_dir / f"vis_fields_{'_'.join(sorted(vis_case_set))}_{args.split}.npz"
    np.savez_compressed(npz_path, **vis_arrays)
    print(f"Saved {npz_path} ({len(vis_arrays)} arrays)", flush=True)


if __name__ == "__main__":
    main()
