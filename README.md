# FlameDiff — Multi-Field Turbulent Flame Reconstruction & Super-Resolution

Unified latent diffusion framework for single- (`T`) and multi-field (`T, OH, HO2`) turbulent flame reconstruction across **8-sensor sparse reconstruction**, **central inpainting**, **4× spatial super-resolution**, and **zero-retraining cross-field inversion** (e.g., `T-only → T, OH, HO2`).

> **Latest Visual Progress Report**: [`reports/progress_report_20261126.md`](./reports/progress_report_20261126.md)  
> **Archived ablation/diagnostic scripts**: branch `archive/experiments-20261126`

---

## Core Architecture & Code Map

| File | Role |
|---|---|
| `model/EnDeCoder.py` | Legacy CNN AE & Multi-Scale Convolutional AE (`--ae-arch mscae`, parallel `3/5/7` conv + `PixelShuffle`) |
| `model/LatentUNet.py` | Multi-scale Attention U-Net latent diffusion prior (`--prior-arch unet`) |
| `model/ViTDecoder.py` | Transformer latent diffusion prior (`--prior-arch vit`) |
| `spatial_data.py` | Leak-free `train/val/test` + Leave-One-Case-Out (`--holdout-cases`) loader for `T` / `T, OH, HO2` |
| `spatial.py` | Stage 1 AE/MSCAE training & Stage 2 unconditional prior training (with frozen-latent RAM caching & cosine EMA) |
| `spatial_mmps.py` / `spatial_mmps_sampler.py` | Matrix-free Woodbury GMRES **Channel-Whitened Adaptive MMPS** & **Adaptive Normalized DPS** |
| `spatial_mmps_observation.py` | Linear operators (`sensors`, `inpaint`, `sr`) with exact adjoints and channel-selective `--observed-fields` |
| `spatial_mmps_reconstruct.py` | Unified reconstruction & held-out evaluation CLI (4-draw CRPS, 2D power spectra, noise tests) |
| `train_eval_3ch_unet.py` | Dedicated 3-field supervised U-Net baselines (`sr`, `inpaint`, Voronoi `sensors`) |
| `test_spatial_mmps.py` | 26-test unit & mathematical verification suite |

---

## Quick Start

### 1. Environment
```bash
pip install torch numpy scipy "diffusers==0.31.0"
python -m unittest test_spatial_mmps.py   # 26/26 tests
```

### 2. Train 3-Field `(T, OH, HO2)` Unconditional Prior
```bash
DATA=../H2_Tco_jet_flames_3fields
OUT=outputs/mscae-unet-3field

# Stage 1: Multi-scale AE (3 fields -> 2x48x64 linear latent)
python spatial.py train ae --data "$DATA" --fields T OH HO2 \
  --ae-arch mscae --latent-channels 2 --ae-output-activation linear --ae-latent-activation linear \
  --ae-loss mse --ae-grad-weight 0.02 --ae-latent-tv-weight 1e-3 --ae-channel-balance \
  --lr-schedule cosine --learning-rate 3e-4 --batch-size 16 --epochs 80 --device cuda \
  --output "$OUT/mscae3.pt"

# Stage 2: Standardized unit-variance cosine EMA LatentUNet prior
python spatial.py train prior --data "$DATA" --ae-checkpoint "$OUT/mscae3.pt" \
  --prior-arch unet --standardize-latent --beta-schedule squaredcos_cap_v2 --ema-decay 0.999 \
  --batch-size 16 --learning-rate 1e-4 --lr-schedule cosine --max-steps 120000 --device cuda \
  --output "$OUT/unet_prior3.pt"
```

### 3. Zero-Retraining Multi-Task & Cross-Field Reconstruction
```bash
# 8-Sensor 3-ch (Channel-Whitened Adaptive MMPS)
python spatial_mmps_reconstruct.py evaluate --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" --split test --panel 9 --steps 50 --draws 4 \
  --device cuda --output "$OUT/test9_sensors_mmps.json"

# Central Inpaint 3-ch (Fast Adaptive DPS, ~1.8s/draw on A100)
python spatial_mmps_reconstruct.py evaluate --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --output "$OUT/test9_inpaint_dps.json"

# Zero-Retraining Cross-Field Inversion (Low-Res T-only -> T, OH, HO2)
python spatial_mmps_reconstruct.py evaluate --task sr --observed-fields T \
  --guidance-mode mmps --mmps-adaptive-weight 0.30 --warm-strength 0.0 \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" --split test --panel 9 --steps 50 --draws 4 \
  --device cuda --output "$OUT/test9_sr_tonly.json"
```

---

## Key Results Summary (`test9` Held-Out Panel, 3 Fields)

| Task | `Adap-W-MMPS` (Ours, Zero-Retrain) | `Adap-DPS` (Zero-Retrain) | 3-ch Supervised U-Net (Per-Task) |
|---|---|---|---|
| **8-Sensor Clean** (`T` / `OH` / `HO2`) | `71.3 K` / **`54.5%`** / `72.0%` | **`70.3 K`** / `55.8%` / **`70.0%`** | `64.3 K` / `52.8%` / `71.6%` *(OH HF 23× loss)* |
| **8-Sensor 30% Noise** | **`74.1 K`** / **`57.0%`** / **`76.8%`** | **`72.0 K`** / **`56.9%`** / **`70.5%`** | `92.1 K` / `71.5%` / `104.6%` *(degrades 19–43%)* |
| **Central Inpaint** | `122.6 K` / **`54.8%`** / `77.2%` | `119.8 K` / `55.6%` / `71.3%` | `104.2 K` / `47.5%` / `66.6%` *(OH HF 10× loss)* |
| **4× SR (`T-only → 3ch`)** | **`10.2 K` / `18.0%` / `42.0%`** *(IoU 0.703)* | `10.6 K` / `18.7%` / `43.4%` | *Cannot run without retraining* |

*(Original upstream Lightning scripts `encoder_training.py` / `vit_training.py` / `inference.py` remain intact for baseline reproduction.)*
