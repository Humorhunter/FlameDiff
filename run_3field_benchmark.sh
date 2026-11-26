#!/usr/bin/env bash
# End-to-end 3-field (T, OH, HO2) training & multi-mode evaluation runner.
# Designed for a single 24-32GB CUDA GPU (e.g., AutoDL V100S / RTX 4090).
set -euo pipefail

PY="${FLAMEDIFF_PYTHON:-python}"
DATA="${FLAMEDIFF_DATA:-../H2_Tco_jet_flames_3fields}"
OUT="${FLAMEDIFF_OUT:-outputs/three-field-20261125}"
EVAL_DIR="$OUT/eval"
mkdir -p "$OUT" "$EVAL_DIR"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== Stage 1: Train 3-field Linear-Latent Linear-Output AE (80 epochs, MSE + grad + variance-balanced) ==="
  "$PY" spatial.py train ae \
    --data "$DATA" \
    --fields T OH HO2 \
    --latent-channels 2 \
    --ae-output-activation linear \
    --ae-latent-activation linear \
    --ae-loss mse \
    --ae-grad-weight 0.02 \
    --ae-latent-reg 1e-6 \
    --ae-channel-balance \
    --lr-schedule cosine \
    --learning-rate 1.5e-4 \
    --batch-size 16 \
    --workers 4 \
    --epochs 80 \
    --device cuda \
    --output "$OUT/ae3.pt" | tee "$OUT/ae3.log"

  echo "=== Stage 2: Train 3-field Standardized Unit-Variance Cosine EMA Prior (80 epochs) ==="
  "$PY" spatial.py train prior \
    --data "$DATA" \
    --ae-checkpoint "$OUT/ae3.pt" \
    --standardize-latent \
    --beta-schedule squaredcos_cap_v2 \
    --ema-decay 0.999 \
    --width 256 \
    --layers 8 \
    --batch-size 8 \
    --workers 4 \
    --learning-rate 2e-5 \
    --lr-schedule cosine \
    --epochs 80 \
    --save-epochs 40 60 80 \
    --device cuda \
    --output "$OUT/prior3.pt" | tee "$OUT/prior3.log"
fi

echo "=== Stage 3A: Matched 3-channel (T,OH,HO2) evaluation on held-out test panel ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 8 --sigma-norm 0.03 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 30 --maxiter 16 --sigma-norm 0.015 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_inpaint_3ch_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 --sigma-norm 0.01 --warm-strength 0.10 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_3ch.json"

echo "=== Stage 3B: Incomplete-physics cross-field inversion (T-only / OH-only -> T,OH,HO2, zero retraining) ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 --sigma-norm 0.01 --warm-strength 0.0 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_tonly_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields OH \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 --sigma-norm 0.01 --warm-strength 0.0 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sr_ohonly_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 8 --sigma-norm 0.03 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_tonly_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 30 --maxiter 16 --sigma-norm 0.015 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_inpaint_tonly_d4.json"

echo "=== Stage 3C: Observation noise robustness (10% and 30% Gaussian noise on sensors, 4 draws) ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 8 --sigma-norm 0.04 --obs-noise-level 0.10 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_noise10_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 8 --sigma-norm 0.08 --obs-noise-level 0.30 --draws 4 \
  --device cuda --resume --output "$EVAL_DIR/test9_sensors_3ch_noise30_d4.json"

echo "=== Stage 3D: Apples-to-apples val9 check vs legacy 1-ch e59 ==="
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split val --panel 9 --steps 50 --maxiter 8 --sigma-norm 0.03 \
  --device cuda --resume --output "$EVAL_DIR/val9_sensors_3ch.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split val --panel 9 --steps 30 --maxiter 16 --sigma-norm 0.015 \
  --device cuda --resume --output "$EVAL_DIR/val9_inpaint_3ch.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T,OH,HO2 \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split val --panel 9 --steps 50 --maxiter 16 --sigma-norm 0.01 --warm-strength 0.10 \
  --device cuda --resume --output "$EVAL_DIR/val9_sr_3ch.json"

echo "=== Stage 3E: Deterministic 3-field baselines + representative visual snapshots ==="
"$PY" eval_3field_extras.py \
  --checkpoint "$OUT/prior3.pt" --data "$DATA" \
  --split test --device cuda --out-dir "$EVAL_DIR" \
  --vis-cases T1030,T1050,T1070

echo "THREE_FIELD_BENCHMARK_COMPLETE"
