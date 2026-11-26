#!/usr/bin/env bash
# Master A100-40GB pipeline executing ALL 4 requested experiments cleanly & atomically:
#   Task 1: Zero-retraining MMPS vs Adap-W-MMPS vs Adap-DPS ablation on existing prior3.pt
#   Task 2: Train & fully evaluate upgraded MSCAE + Multi-Scale Attention LatentUNet (120k steps)
#   Task 3: Leave-One-Case-Out (LOCO, holdout T1050) MSCAE + LatentUNet unseen-case generalization
#   Task 4: Supervised 3-field U-Net baselines (SR, Inpaint, Voronoi 8-Sensor clean/10%/30% noise)
set -euo pipefail

PY="${FLAMEDIFF_PYTHON:-/root/miniconda3/bin/python}"
DATA="${FLAMEDIFF_DATA:-/root/autodl-tmp/flamediff/H2_Tco_jet_flames_3fields}"
ROOT_OUT="${FLAMEDIFF_ROOT_OUT:-outputs/a100-suite-20261126}"
mkdir -p "$ROOT_OUT"

echo "==================== [0] Unit Tests (26 tests) ===================="
"$PY" -m unittest test_spatial_mmps.py

echo "==================== [TASK 1] MMPS vs Adap-W-MMPS vs Adap-DPS on prior3.pt ===================="
FLAMEDIFF_PYTHON="$PY" FLAMEDIFF_DATA="$DATA" FLAMEDIFF_OUT="outputs/three-field-20261125" \
  bash run_mmps_vs_dps_ablation.sh | tee "$ROOT_OUT/task1_ablation.log"

echo "==================== [TASK 4] Supervised 3-Field U-Net Baselines (test9) ===================="
"$PY" train_eval_3ch_unet.py \
  --data "$DATA" \
  --out-dir "$ROOT_OUT/task4_unet3_baselines" \
  --epochs 40 --batch-size 16 --lr 3e-4 --base-channels 48 --workers 4 --device cuda \
  | tee "$ROOT_OUT/task4_unet3.log"

echo "==================== [TASK 2] Upgraded 3-Field MSCAE + LatentUNet (120k steps) ===================="
T2="$ROOT_OUT/task2_mscae_unet"
mkdir -p "$T2/eval"

if [[ ! -f "$T2/mscae3.pt" ]]; then
  "$PY" spatial.py train ae \
    --data "$DATA" --fields T OH HO2 \
    --ae-arch mscae --ae-base-channels 48 --latent-channels 2 \
    --ae-output-activation linear --ae-latent-activation linear \
    --ae-loss mse --ae-grad-weight 0.02 --ae-latent-reg 1e-6 --ae-latent-tv-weight 1e-3 \
    --ae-channel-balance --lr-schedule cosine --learning-rate 3e-4 \
    --batch-size 16 --workers 4 --epochs 80 --device cuda \
    --output "$T2/mscae3.pt" | tee "$T2/mscae3.log"
fi

if [[ ! -f "$T2/unet_prior3.pt" ]]; then
  "$PY" spatial.py train prior \
    --data "$DATA" --ae-checkpoint "$T2/mscae3.pt" \
    --prior-arch unet --prior-base-channels 64 \
    --standardize-latent --beta-schedule squaredcos_cap_v2 --ema-decay 0.999 \
    --batch-size 16 --workers 4 --learning-rate 1e-4 --lr-schedule cosine \
    --max-steps 120000 --device cuda \
    --output "$T2/unet_prior3.pt" | tee "$T2/unet_prior3.log"
fi

# Evaluate Task 2 with BOTH Upgraded MMPS (Channel-Whitened + Adaptive) AND Exact Whiten-MMPS AND Adap-DPS
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.02 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_sensors_3ch_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --sigma-norm 0.025 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_sensors_3ch_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.04 \
  --obs-noise-level 0.30 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_sensors_3ch_noise30_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.015 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 12 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_inpaint_3ch_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode dps --dps-weight 0.30 --dps-channel-balance \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 100 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_inpaint_3ch_dps100_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --sigma-norm 0.01 --warm-strength 0.10 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 \
  --device cuda --resume --output "$T2/eval/test9_sr_3ch_wmmps.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T \
  --guidance-mode mmps --mmps-adaptive-weight 0.30 --sigma-norm 0.01 --warm-strength 0.0 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_sr_tonly_adap_mmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields OH \
  --guidance-mode mmps --mmps-adaptive-weight 0.30 --sigma-norm 0.01 --warm-strength 0.0 \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" \
  --split test --panel 9 --steps 50 --maxiter 16 --draws 4 \
  --device cuda --resume --output "$T2/eval/test9_sr_ohonly_adap_mmps_d4.json"

"$PY" export_vis_snapshots.py \
  --checkpoint "$T2/unet_prior3.pt" --data "$DATA" --split test --vis-cases T1030,T1050,T1070 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --steps 50 --device cuda \
  --output "$T2/eval/vis_mscae_unet_T1030_T1050_T1070_test.npz"

echo "==================== [TASK 3] LOCO Unseen-Case Generalization (Holdout T1050) ===================="
T3="$ROOT_OUT/task3_loco_T1050"
mkdir -p "$T3/eval"

if [[ ! -f "$T3/mscae3_loco.pt" ]]; then
  "$PY" spatial.py train ae \
    --data "$DATA" --fields T OH HO2 --holdout-cases T1050 \
    --ae-arch mscae --ae-base-channels 48 --latent-channels 2 \
    --ae-output-activation linear --ae-latent-activation linear \
    --ae-loss mse --ae-grad-weight 0.02 --ae-latent-reg 1e-6 --ae-latent-tv-weight 1e-3 \
    --ae-channel-balance --lr-schedule cosine --learning-rate 3e-4 \
    --batch-size 16 --workers 4 --epochs 60 --device cuda \
    --output "$T3/mscae3_loco.pt" | tee "$T3/mscae3_loco.log"
fi

if [[ ! -f "$T3/unet_prior3_loco.pt" ]]; then
  "$PY" spatial.py train prior \
    --data "$DATA" --ae-checkpoint "$T3/mscae3_loco.pt" --holdout-cases T1050 \
    --prior-arch unet --prior-base-channels 64 \
    --standardize-latent --beta-schedule squaredcos_cap_v2 --ema-decay 0.999 \
    --batch-size 16 --workers 4 --learning-rate 1e-4 --lr-schedule cosine \
    --max-steps 80000 --device cuda \
    --output "$T3/unet_prior3_loco.pt" | tee "$T3/unet_prior3_loco.log"
fi

# Evaluate on ALL 5 slices of completely unseen case T1050 (--split holdout --panel 45)!
"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sensors --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.02 \
  --checkpoint "$T3/unet_prior3_loco.pt" --data "$DATA" \
  --split holdout --panel 45 --steps 50 --maxiter 24 --draws 4 \
  --device cuda --resume --output "$T3/eval/holdout_T1050_sensors_3ch_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task inpaint --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --sigma-norm 0.015 \
  --checkpoint "$T3/unet_prior3_loco.pt" --data "$DATA" \
  --split holdout --panel 45 --steps 50 --maxiter 12 --draws 4 \
  --device cuda --resume --output "$T3/eval/holdout_T1050_inpaint_3ch_adap_wmmps_d4.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T,OH,HO2 \
  --guidance-mode mmps --mmps-channel-whiten --sigma-norm 0.01 --warm-strength 0.10 \
  --checkpoint "$T3/unet_prior3_loco.pt" --data "$DATA" \
  --split holdout --panel 45 --steps 50 --maxiter 16 \
  --device cuda --resume --output "$T3/eval/holdout_T1050_sr_3ch_wmmps.json"

"$PY" spatial_mmps_reconstruct.py evaluate \
  --task sr --observed-fields T \
  --guidance-mode mmps --mmps-adaptive-weight 0.30 --sigma-norm 0.01 --warm-strength 0.0 \
  --checkpoint "$T3/unet_prior3_loco.pt" --data "$DATA" \
  --split holdout --panel 45 --steps 50 --maxiter 16 --draws 4 \
  --device cuda --resume --output "$T3/eval/holdout_T1050_sr_tonly_adap_mmps_d4.json"

"$PY" export_vis_snapshots.py \
  --checkpoint "$T3/unet_prior3_loco.pt" --data "$DATA" --split holdout --vis-cases T1050 \
  --guidance-mode mmps --mmps-channel-whiten --mmps-adaptive-weight 0.30 --steps 50 --device cuda \
  --output "$T3/eval/vis_loco_T1050_holdout.npz"

echo "A100_ALL_TASKS_COMPLETE"
if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
  echo "AUTO_SHUTDOWN=1 -> syncing and shutting down..."
  sync && /usr/bin/shutdown
fi
