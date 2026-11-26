#!/usr/bin/env python3
"""Supervised 3-field (T, OH, HO2) U-Net baselines across all 3 spatial tasks.

Directly parallels Wu, Liang & Luo (Combustion and Flame 282, 2025, Section 4.3.2):
  - sensors: Voronoi tessellation (nearest-neighbor grid) + sensor mask -> 3-ch U-Net
             evaluated on clean, 10% noise, and 30% noise (exact same seeds as MMPS!)
  - inpaint: [masked 3-ch, known mask] -> 3-ch U-Net
  - sr:      bicubic 3-ch + residual 3-ch U-Net

Evaluated on the exact same held-out test9 panel with identical per-channel metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from spatial import _gradient_l1_loss, _weighted_pixel_loss, prepare
from spatial_baselines import _DoubleConv, interpolate
from spatial_data import TemperatureData
from spatial_mmps_observation import SpatialObservation
from spatial_mmps_reconstruct import _compute_single_channel_metrics, evaluation_panel


class MultiFieldUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3, base_channels: int = 48):
        super().__init__()
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = _DoubleConv(in_channels, c[0])
        self.enc2 = _DoubleConv(c[0], c[1])
        self.enc3 = _DoubleConv(c[1], c[2])
        self.mid = _DoubleConv(c[2], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c[2] * 2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c[1] * 2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        h = self.mid(F.max_pool2d(e3, 2))
        h = self.dec3(torch.cat([self.up3(h), e3], dim=1))
        h = self.dec2(torch.cat([self.up2(h), e2], dim=1))
        h = self.dec1(torch.cat([self.up1(h), e1], dim=1))
        return self.out(h)


def _build_voronoi_index(op: SpatialObservation, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute nearest sensor index [H, W] and binary sensor mask [1, 1, H, W]."""
    H, W = op.height, op.width
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.tensor(op.sensor_coords_xy, device=device, dtype=torch.float32)  # [8, 2] (x, y)
    d2 = (xx[None] - coords[:, 0, None, None]) ** 2 + (yy[None] - coords[:, 1, None, None]) ** 2
    nearest = d2.argmin(dim=0)  # [H, W] in [0..7]
    mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    for x, y in op.sensor_coords_xy:
        mask[0, 0, y, x] = 1.0
    return nearest, mask


def _voronoi_fill(obs_sensors: torch.Tensor, nearest: torch.Tensor, smask: torch.Tensor) -> torch.Tensor:
    """Map [B, C, 8] sensor readings to [B, C+1, H, W] Voronoi tessellation + sensor mask."""
    B, C, _ = obs_sensors.shape
    H, W = nearest.shape
    idx = nearest.reshape(1, 1, H * W).expand(B, C, H * W)
    filled = torch.gather(obs_sensors, 2, idx).reshape(B, C, H, W)
    return torch.cat([filled, smask.expand(B, -1, -1, -1)], dim=1)


def _per_channel_metrics(pred, target, op, fields, ranges):
    full_known = op.mask(target)
    out = {}
    for c_idx, fname in enumerate(fields):
        pred_c = pred[:, c_idx : c_idx + 1]
        target_c = target[:, c_idx : c_idx + 1]
        if op.task == "inpaint":
            known_c = full_known[:, c_idx : c_idx + 1]
            region_c = ~known_c
        elif op.task == "sensors":
            known_c = None
            region_c = ~full_known[:, c_idx : c_idx + 1]
        else:
            known_c = None
            region_c = torch.ones_like(target_c, dtype=torch.bool)
        lo, hi = ranges[fname]
        out[fname] = _compute_single_channel_metrics(
            pred_c, target_c, region_c, known_c,
            field_name=fname, low=lo, span=hi - lo, obs_rmse=None, legacy_t_keys=False,
        )
    return out


def _aggregate_pc(rows, fields):
    summary = {}
    for fn in fields:
        summary[fn] = {}
        for k in rows[0][fn].keys():
            vals = [r[fn][k] for r in rows if r[fn][k] is not None]
            summary[fn][k] = float(sum(vals) / len(vals)) if vals else None
    return summary


def train_one_task(
    task: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ch_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    base_channels: int,
    ckpt_path: Path,
):
    op = SpatialObservation(task, num_channels=3, observed_channels=(0, 1, 2))
    in_ch = 3 if task == "sr" else 4
    model = MultiFieldUNet(in_ch, out_channels=3, base_channels=base_channels).to(device)
    nearest, smask = _build_voronoi_index(op, device) if task == "sensors" else (None, None)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    def forward_task(batch_norm: torch.Tensor) -> torch.Tensor:
        obs = op.measure(batch_norm)
        if task == "sr":
            base = interpolate(obs, op, "bicubic")
            return base + model(base)
        if task == "inpaint":
            known = op.mask(batch_norm)[:, :1].to(dtype=batch_norm.dtype)
            return model(torch.cat([obs, known], dim=1))
        return model(_voronoi_fill(obs, nearest, smask))

    best = float("inf")
    best_state = None
    t0 = perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            pred = forward_task(batch)
            if task == "inpaint":
                miss = ~op.mask(batch)
                # Compute weighted MSE + grad over full field with 4x emphasis on missing hole
                w_spatial = torch.where(miss, 4.0, 1.0)
                loss = ((pred - batch).square() * ch_weights.view(1, 3, 1, 1) * w_spatial).mean()
            else:
                loss = _weighted_pixel_loss(pred, batch, loss_kind="mse", channel_weights=ch_weights)
            loss = loss + 0.02 * _gradient_l1_loss(pred, batch, channel_weights=ch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.detach().item()) * len(batch)
        sched.step()

        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = forward_task(batch)
                va += float(_weighted_pixel_loss(pred, batch, loss_kind="mse", channel_weights=ch_weights).item()) * len(batch)
        va /= len(val_loader.dataset)
        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep in (1, 10, 20, 30, epochs) or ep % 10 == 0:
            print(f"[unet3:{task}] ep {ep:02d}/{epochs} train={tr/len(train_loader.dataset):.5f} val={va:.5f}", flush=True)

    model.load_state_dict(best_state)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": task, "base_channels": base_channels, "best_val": best, "weights": best_state}, ckpt_path)
    print(f"[unet3:{task}] done in {perf_counter()-t0:.1f}s, best_val={best:.5f}", flush=True)
    return model.eval(), op, (nearest, smask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-cases", type=str, default=None)
    args = parser.parse_args()

    device = prepare(args.device, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ("T", "OH", "HO2")
    data = TemperatureData(args.data, fields=fields, holdout_cases=args.holdout_cases)
    ranges = data.training_ranges()
    stats = data.training_channel_stats()
    inv_v = [1.0 / (max(stats[f]["norm_std"], 1e-4) ** 2) for f in fields]
    m_iv = sum(inv_v) / len(inv_v)
    ch_weights = torch.tensor([v / m_iv for v in inv_v], dtype=torch.float32, device=device)

    train_loader = DataLoader(data.dataset("train", ranges), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(data.dataset("val", ranges), batch_size=args.batch_size, num_workers=args.workers)

    lows = torch.tensor([ranges[f][0] for f in fields], device=device).view(1, 3, 1, 1)
    spans = torch.tensor([ranges[f][1] - ranges[f][0] for f in fields], device=device).view(1, 3, 1, 1)

    split_name = "holdout" if args.holdout_cases else "test"
    dataset_eval, samples_eval = evaluation_panel(data, split_name, ranges)
    by_case: dict[str, list[int]] = {}
    for i, s in enumerate(samples_eval):
        by_case.setdefault(s["case"], []).append(i)
    selected = [g[len(g) // 2] for g in by_case.values()]

    results = {"split": split_name, "cases": [samples_eval[i]["case"] for i in selected], "methods": {}}

    for task in ("sr", "inpaint", "sensors"):
        model, op, aux = train_one_task(
            task, train_loader, val_loader, ch_weights, device,
            epochs=args.epochs, lr=args.lr, base_channels=args.base_channels,
            ckpt_path=out_dir / f"unet3_{task}.pt",
        )
        nearest, smask = aux
        noise_levels = (0.0, 0.10, 0.30) if task == "sensors" else (0.0,)
        for nl in noise_levels:
            rows = []
            t_inf = 0.0
            for idx in selected:
                norm_gt = dataset_eval[idx].unsqueeze(0).to(device)
                gt = norm_gt * spans + lows
                obs_phys = op.measure(gt)
                if nl > 0.0:
                    gen = torch.Generator(device=device).manual_seed(idx + 7919)
                    stds = torch.tensor([stats[f]["std"] * nl for f in fields], device=device).view(1, 3, 1)
                    obs_phys = obs_phys + torch.randn(obs_phys.shape, device=device, generator=gen) * stds
                obs_norm = (obs_phys - lows.squeeze(-1)) / spans.squeeze(-1) if task == "sensors" else (obs_phys - lows) / spans
                if task == "inpaint":
                    obs_norm = torch.where(op.observation_mask(obs_norm), obs_norm, torch.zeros((), device=device))

                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = perf_counter()
                with torch.no_grad():
                    if task == "sr":
                        base = interpolate(obs_norm, op, "bicubic")
                        pred_norm = base + model(base)
                    elif task == "inpaint":
                        known = op.mask(norm_gt)[:, :1].to(dtype=norm_gt.dtype)
                        pred_norm = model(torch.cat([obs_norm, known], dim=1))
                    else:
                        pred_norm = model(_voronoi_fill(obs_norm, nearest, smask))
                torch.cuda.synchronize() if device.type == "cuda" else None
                t_inf += perf_counter() - t0

                pred_phys = pred_norm * spans + lows
                if nl == 0.0:
                    pred_phys = op.project(pred_phys, obs_phys)
                rows.append(_per_channel_metrics(pred_phys.cpu(), gt.cpu(), op, fields, ranges))
            tag = f"unet3_{task}" if nl == 0.0 else f"unet3_{task}_noise{int(nl*100)}"
            results["methods"][tag] = {
                "mean_seconds_per_frame": t_inf / len(selected),
                "per_channel": _aggregate_pc(rows, fields),
            }
            pc = results["methods"][tag]["per_channel"]
            print(
                f"[eval] {tag:22s} | T={pc['T']['rmse']:.2f}K (HF={pc['T']['hf_energy_ratio_pred']*100:.2f}%) | "
                f"OH={pc['OH']['nmae_pct']:.1f}% (HF={pc['OH']['hf_energy_ratio_pred']*100:.2f}%) | "
                f"HO2={pc['HO2']['nmae_pct']:.1f}% (HF={pc['HO2']['hf_energy_ratio_pred']*100:.2f}%)",
                flush=True,
            )

    out_json = out_dir / f"unet3_baselines_{split_name}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
