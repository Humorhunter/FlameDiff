#!/usr/bin/env bash
# Upgraded 3-field (T, OH, HO2) MSCAE + Multi-Scale LatentUNet training & evaluation runner.
# Incorporates Wu, Liang & Luo (Combustion and Flame 282, 2025) architectural & guidance lessons:
#   1. MSCAE (parallel 3/5/7 multiscale kernels + GroupNorm + PixelShuffle + latent TV smoothness)
#   2. Single-channel (or 2-channel) standardized unit-variance latent space
#   3. Multi-scale Attention LatentUNet trained with step budget (150k steps) + cosine EMA 0.999
#   4. Head-to-head Adaptive Normalized DPS vs MMPS evaluation on held-out test9 (+ optional LOCO holdout)
set -euo pipefail

PY="${FLAMEDIFF_PYTHON:-python}"
DATA="${FLAMEDIFF_DATA:-../H2_Tco_jet_flames_3fields}"
OUT="${FLAMEDIFF_OUT:-outputs/mscae-unet-3field}"
EVAL_DIR="$OUT/eval"
LATENT_CH="${LATENT_CH:-2}"
MAX_STEPS="${MAX_STEPS:-150000}"
HOLDOUT_ARG=()
if [[ -n "${HOLDOUT_CASES:-}" ]]; then
  HOLDOUT_ARG=(--holdout-cases "$HOLDOUT_CASES")
fi
mkdir -p "$OUT" "$EVAL_DIR"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== Stage 1: Train 3-field MSCAE (parallel 3/5/7 + PixelShuffle + latent TV smoothness) ==="
  "$PY" spatial.py train ae \
    --data "$DATA" \
    --fields T OH HO2 \
    --ae-arch mscae \
    --ae-base-channels 48 \
    --latent-channels "$LATENT_CH" \
    --ae-output-activation linear \
    --ae-latent-activation linear \
    --ae-loss mse \
    --ae-grad-weight 0.02 \
    --ae-latent-reg 1e-6 \
    --ae-latent-tv-weight 1e-3 \
    --ae-channel-balance \
    --lr-schedule cosine \
    --learning-rate 3e-4 \
    --batch-size 16 \
    --workers 4 \
    --epochs 100 \
    "${HOLDOUT_ARG[@]}" \
    --device cuda \
    --output "$OUT/mscae3.pt" | tee "$OUT/mscae3.log"

  echo "=== Stage 2: Train 3-field Multi-Scale Attention LatentUNet (${MAX_STEPS} steps, cosine EMA) ==="
  "$PY" spatial.py train prior \
    --data "$DATA" \
    --ae-checkpoint "$OUT/mscae3.pt" \
    --prior-arch unet \
    --prior-base-channels 64 \
    --standardize-latent \
    --beta-schedule squaredcos_cap_v2 \
    --ema-decay 0.999 \
    --batch-size 16 \
    --workers 4 \
    --learning-rate 1e-4 \
    --lr-schedule cosine \
    --max-steps "$MAX_STEPS" \
    "${HOLDOUT_ARG[@]}" \
    --device cuda \
    --output "$OUT/unet_prior3.pt" | tee "$OUT/unet_prior3.log"
fi

echo "=== Stage 3: Adaptive Normalized DPS (100 steps, channel-balanced) on held-out test9 ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_dps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_inpaint_3ch_dps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T,OH,HO2 \
  --guidance-mode mmps --warm-strength 0.10 --sigma-norm 0.01 \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_3ch_mmps.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T \
  --guidance-mode dps --dps-weight 0.30 \
  --checkpoint "$OUT/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_tonly_dps_d4.json"

echo "MSCAE_UNET_3FIELD_COMPLETE"
