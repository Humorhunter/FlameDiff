#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

data=${FLAMEDIFF_DATA:-../H2_Tco_jet_flames_3fields}
unet=${FLAMEDIFF_INPAINT_UNET:-outputs/unet-full-20261118-123140/unet-inpaint.pt}
out=${1:-outputs/inpaint-ablation-$(date +%Y%m%d-%H%M%S)}
python=${PYTHON:-python}
mkdir -p "$out"
test -d "$data"
test -s "$unet"
"$python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_name(0))'
echo "Output directory $out"

train_args=(--data "$data" --unet-checkpoint "$unet" --device cuda --seed 0)
eval_args=(--data "$data" --unet-checkpoint "$unet" --device cuda --seed 0 --split val)
for variant in conv transformer; do
    architecture=(--variant "$variant")
    if [[ $variant == transformer ]]; then
        architecture+=(--attention-blocks 2)
    fi
    "$python" -u spatial_inpaint_experiment.py train "${train_args[@]}" "${architecture[@]}" \
        --width 32 --epochs 1 --batch-size 2 --train-limit 4 --val-limit 2 --output "$out/$variant-smoke.pt" \
        2>&1 | tee "$out/$variant-smoke-train.log"
    "$python" -u spatial_inpaint_experiment.py benchmark "${eval_args[@]}" \
        --checkpoint "$out/$variant-smoke.pt" --steps 3 --draws 2 --limit 2 \
        --output "$out/$variant-smoke.json" 2>&1 | tee "$out/$variant-smoke-eval.log"
done

for variant in conv transformer; do
    architecture=(--variant "$variant")
    if [[ $variant == transformer ]]; then
        architecture+=(--attention-blocks 2)
    fi
    "$python" -u spatial_inpaint_experiment.py train "${train_args[@]}" "${architecture[@]}" \
        --width 32 --epochs 20 --batch-size 4 --save-epochs 10 20 --output "$out/$variant.pt" \
        2>&1 | tee "$out/$variant-train.log"
    for epoch in 0010 0020; do
        "$python" -u spatial_inpaint_experiment.py benchmark "${eval_args[@]}" \
            --checkpoint "$out/$variant-epoch-$epoch.pt" --steps 50 --draws 4 \
            --output "$out/$variant-epoch-$epoch-val45.json" \
            2>&1 | tee "$out/$variant-epoch-$epoch-val45.log"
    done
done

"$python" summarize_inpaint_ablation.py "$out"
