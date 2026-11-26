#!/usr/bin/env python3
"""Export representative 3-field (T, OH, HO2) prediction & 4-draw std snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from spatial_data import TemperatureData
from spatial_mmps_reconstruct import (
    TemperatureReconstructor,
    _add_observation_noise,
    evaluation_panel,
    load_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=("val", "test", "holdout"))
    p.add_argument("--holdout-cases", default=None)
    p.add_argument("--vis-cases", default="T1030,T1050,T1070")
    p.add_argument("--guidance-mode", default="mmps", choices=("mmps", "dps"))
    p.add_argument("--mmps-channel-whiten", action="store_true")
    p.add_argument("--mmps-adaptive-weight", type=float, default=None)
    p.add_argument("--dps-weight", type=float, default=0.30)
    p.add_argument("--dps-channel-balance", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint, "prior")
    fields = tuple(ckpt["fields"])
    hc = args.holdout_cases or ckpt.get("split", {}).get("holdout_cases")
    data = TemperatureData(args.data, fields=fields, holdout_cases=hc)
    ranges = data.training_ranges()
    stats = data.training_channel_stats()
    dataset, samples = evaluation_panel(data, args.split, ranges)
    by_case: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_case.setdefault(s["case"], []).append(i)
    selected = [g[len(g) // 2] for g in by_case.values()]

    vis_set = {c.strip() for c in args.vis_cases.split(",") if c.strip()}
    lows = torch.tensor([ranges[f][0] for f in fields], dtype=torch.float32).view(1, 3, 1, 1)
    spans = torch.tensor([ranges[f][1] - ranges[f][0] for f in fields], dtype=torch.float32).view(1, 3, 1, 1)

    modes = [
        ("sensors_3ch", "sensors", "T,OH,HO2", args.steps, 24, 0.025, 0.0, 4, 0.0),
        ("sensors_3ch_noise30", "sensors", "T,OH,HO2", args.steps, 24, 0.04, 0.0, 4, 0.30),
        ("inpaint_3ch", "inpaint", "T,OH,HO2", args.steps, 12, 0.015, 0.0, 4, 0.0),
        ("sr_3ch", "sr", "T,OH,HO2", 50, 16, 0.01, 0.10, 1, 0.0),
        ("sr_tonly", "sr", "T", args.steps, 16, 0.01, 0.0, 4, 0.0),
        ("sr_ohonly", "sr", "OH", args.steps, 16, 0.01, 0.0, 4, 0.0),
    ]

    arrays: dict[str, np.ndarray] = {}
    for idx in selected:
        cname = samples[idx]["case"]
        if cname not in vis_set:
            continue
        target = dataset[idx].unsqueeze(0) * spans + lows
        arrays[f"{cname}__gt"] = target[0].numpy().astype(np.float32)

    for mname, task, obs_f, st, mi, s_norm, warm, draws, nl in modes:
        use_mode = "mmps" if mname == "sr_3ch" else args.guidance_mode
        m_args = SimpleNamespace(
            task=task, observed_fields=obs_f, device=args.device, seed=0,
            steps=st, maxiter=mi, sigma_k=None, sigma_norm=s_norm, warm_strength=warm,
            draws=draws, obs_noise_level=nl,
            guidance_mode=use_mode,
            mmps_channel_whiten=args.mmps_channel_whiten,
            mmps_adaptive_weight=None if mname == "sr_3ch" else args.mmps_adaptive_weight,
            dps_weight=args.dps_weight,
            dps_channel_balance=args.dps_channel_balance,
        )
        rec = TemperatureReconstructor(m_args, ckpt)
        for idx in selected:
            cname = samples[idx]["case"]
            if cname not in vis_set:
                continue
            target = dataset[idx].unsqueeze(0) * spans + lows
            if f"{cname}__ae_oracle" not in arrays:
                with torch.no_grad():
                    ae_n = rec.autoencoder(dataset[idx].unsqueeze(0).to(rec.device)).cpu()
                arrays[f"{cname}__ae_oracle"] = (ae_n * spans + lows)[0].numpy().astype(np.float32)
            obs = rec.operator.measure(target).to(rec.device)
            if nl > 0:
                obs = _add_observation_noise(obs, rec, stats, noise_level=nl, seed=idx)
            preds = []
            for d in range(draws):
                proj, raw, _ = rec.reconstruct(obs, seed=idx + d * 1009)
                out_t = raw if nl > 0 else proj
                preds.append(out_t[0].detach().cpu().numpy().astype(np.float32))
            stack = np.stack(preds, axis=0)
            arrays[f"{cname}__{mname}__draw0"] = stack[0]
            if draws > 1:
                arrays[f"{cname}__{mname}__mean"] = stack.mean(axis=0)
                arrays[f"{cname}__{mname}__std"] = stack.std(axis=0, ddof=1)
            print(f"[vis] {cname} {mname} (mode={use_mode}, d={draws}) ok", flush=True)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, **arrays)
    print(f"Saved {out_p} ({len(arrays)} arrays)", flush=True)


if __name__ == "__main__":
    main()
