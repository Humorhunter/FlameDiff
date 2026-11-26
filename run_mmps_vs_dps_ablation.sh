#!/usr/bin/env bash
# Head-to-head GPU comparison on the SAME shared 3-field prior (prior3.pt, zero retraining):
#   1. Legacy Isotropic MMPS (baseline)
#   2. Wu et al. (2025) Adaptive Normalized DPS (channel-balanced)
#   3. NEW Channel-Whitened Full-Rank MMPS (exact anisotropic Sigma_y ~ diag(norm_std_c^2))
#   4. NEW Adaptive-Normalized Channel-Whitened MMPS (Wu et al. Eq.18-19 rewritten in MMPS form)
set -euo pipefail

PY="${FLAMEDIFF_PYTHON:-/root/miniconda3/bin/python}"
DATA="${FLAMEDIFF_DATA:-/root/autodl-tmp/flamediff/H2_Tco_jet_flames_3fields}"
OUT="${FLAMEDIFF_OUT:-outputs/three-field-20261125}"
EVAL_DIR="$OUT/eval_mmps_upgrade"
CKPT="$OUT/prior3.pt"
mkdir -p "$EVAL_DIR"

echo "=== 1. 8-Sensor (3ch, clean, test9, 4-draw) ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --sigma-norm 0.025 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.02 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_dps100_d4.json"

echo "=== 2. 8-Sensor (3ch, 30% Gaussian noise, test9, 4-draw) ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.04 \
  --obs-noise-level 0.30 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_noise30_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --obs-noise-level 0.30 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_noise30_dps100_d4.json"

echo "=== 3. Central Inpaint (3ch, test9, 4-draw) ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --sigma-norm 0.015 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 30 --maxiter 16 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_inpaint_3ch_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.015 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 12 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_inpaint_3ch_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_inpaint_3ch_dps100_d4.json"

echo "=== 4. 4x SR (3ch & T-only cross-field, test9) ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --sigma-norm 0.01 --warm-strength 0.10 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_3ch_wmmps.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T \
  --guidance-mode mmps --mmps-adaptive-weight 0.30 --sigma-norm 0.01 --warm-strength 0.0 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_tonly_adap_mmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T \
  --guidance-mode dps --dps-weight 0.30 --warm-strength 0.0 \
  --checkpoint "$CKPT" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_tonly_dps100_d4.json"

echo "MMPS_VS_DPS_ABLATION_COMPLETE"
