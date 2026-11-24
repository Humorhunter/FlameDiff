# FlameDiff: Flame Field based Diffusion Model

As you know, AI can paint:

![AI Paint](./images/gpt_flame.png)

What if we want AI to paint a real turbulant flame field? This make sense!

![FlameDiff](./images/flame.png)

## Running the code

### Prerequisites

Install the required enviroment:
```
conda create -f environment.yml
```

### Running the code

The training process is based on `pytorch` and `pytorch-lightning`. All the configurations are stored in the `config` folder and will be arranged by `hydra`.

Before running the training process, modify the `config/default_training.yaml` file to fit your own data path. `cache_path` should be the directory where `T1030` and similar folders are stored. For more information about the parameters in the config, look at the `xxx_training.py` for what parameters are used. Also, you may need to modify the variable `devices` in the training python script to fit your hardware. The experiments are conducted on 8 * 3090 GPUs.

To train the `autoencoder` model, run the following command:

```
python encoder_training.py exp_name=autoencoder trainer.max_epochs=100
```

To train the `flame` model, run the following command:

```
python vit_training.py exp_name=vit trainer.max_epochs=10000
```

## Single-frame spatial benchmark

### Shared latent MMPS reconstruction

Use `spatial_mmps_reconstruct.py` to reconstruct one temperature field
from eight sensors, a center mask, or a 4× coarse grid. The three tasks
share the same unconditional spatial prior checkpoint (from
`spatial.py train prior`) and its embedded autoencoder.

| `--task` | Kelvin `.npy` input |
| --- | --- |
| `sensors` | Shape `(8,)` in the `(x, y)` order in `config/default_training.yaml` |
| `inpaint` | Shape `(192, 256)`. Values at rows `[48, 144)` and columns `[64, 192)` are ignored and may be NaN. |
| `sr` | Shape `(48, 64)`, the means of non-overlapping `4×4` blocks |

```sh
python spatial_mmps_reconstruct.py reconstruct --task sr \
  --checkpoint outputs/spatial/prior.pt --observation coarse.npy \
  --output temperature.npy --device cuda
```

The output is a `(192, 256)` array in Kelvin. The final field copies known
pixels for the sensor and inpainting tasks, and restores block averages for
SR. The console reports observation RMSE before and after that projection
and the GMRES diagnostics. This assumes exact input measurements.

The default initializations are noise for sensors, harmonic interpolation for
inpainting, and block-mean-corrected bicubic interpolation for SR. The warm
initializations use the last 25% and 10% of the diffusion schedule,
respectively. Use `--warm-strength 0` to start from noise for either task.
`--sigma-k` controls the MMPS likelihood regularization scale in Kelvin.
Defaults are 5% of the checkpoint temperature span for sensors and 10 K for
the other two tasks. These are starting settings for validation. The MMPS
linear solve uses a finite iteration budget, and convergence is reported.

```sh
python spatial_mmps_reconstruct.py evaluate --task inpaint \
  --checkpoint outputs/spatial/prior.pt --data ../H2_Tco_jet_flames_3fields \
  --output outputs/spatial/mmps-inpaint-val.json --device cuda --resume
```

Evaluation uses the middle slice and middle validation frame of each
case. It measures errors over unmeasured pixels for sensors, the missing
rectangle for inpainting, and the full field for SR, reporting both raw
and projected results for MAE, RMSE, gradient MAE (plus boundary gradient
MAE for inpainting), SSIM, hot-zone MAE/RMSE/IoU/Dice at 1000 K and
1200 K, observation RMSE, and GMRES diagnostics. Progress is written
atomically after each frame (`status: "running"` until `"complete"`), and
`--resume` skips already-completed compatible frames.

### DPS and spatial benchmark

`spatial.py` adds a separate workflow for centered inpainting and 4× spatial
super-resolution. The original eight-sensor training and inference path still
uses sensor tokens. The spatial workflow trains an unconditional prior with the
same convolutional autoencoder and convolution/Transformer denoiser backbone.
Its sampler adapts [Du et al. (2024)](https://doi.org/10.1038/s41467-024-54712-1):
at every DDPM step it decodes the estimated clean latent, applies the observation
operator, and uses the gradient of the observation residual to correct the next
latent. The correction uses per-frame observation RMSE on the normalized scale.
The same prior checkpoint serves both tasks. `--guidance 0` (or benchmark method
`unguided`) omits the correction and the known-pixel copy. This is an adaptation
to FlameDiff's existing architecture, rather than a reproduction of the
paper's neural field or sequence model.

| Task | Observation | Baselines |
| --- | --- | --- |
| `inpaint` | Mask rows `[48, 144)` and columns `[64, 192)` on a `192×256` field | `harmonic` (discrete Dirichlet interpolation), mask-conditioned `unet` |
| `sr` | Non-overlapping `4×4` block averages give a `48×64` observation | `bilinear`, `bicubic`, residual `unet` |

Operators act in array coordinates. Block averages weight array cells equally;
they are not area-weighted physical averages on the original mesh. The final
inpainting output copies observed values exactly. SR does not apply an output
projection, so its observation error measures how closely the result matches
the input when downsampled.

Set `DATA=../H2_Tco_jet_flames_3fields` from the FlameDiff directory. The loader
uses only the primary `T.npy` files. All angles of a case share the same
zero-based frame index assignments: train `[0,130)`, validation `[140,160)`,
test `[170,200)`. Ten frames between neighboring splits are excluded. For the
provided dataset this gives 5,850 / 900 / 1,350 frames and 900 excluded frames.
This is a same-case evaluation. Frames remain separate samples; grouping
neighboring indices only reduces leakage across splits. Temperature limits
come from the training split. Checkpoints store the split manifest and limits;
retain the original data files when comparing runs. Legacy weights are not
accepted as spatial checkpoints.

The spatial path requires PyTorch, NumPy and `diffusers==0.32.2`. It defaults to
CPU; use `--device cuda` on a suitable training machine. From the FlameDiff
directory, train and evaluate as follows:

```sh
DATA=../H2_Tco_jet_flames_3fields
python spatial.py train ae --data "$DATA" --output outputs/spatial/ae.pt
python spatial.py train prior --data "$DATA" --ae-checkpoint outputs/spatial/ae.pt --output outputs/spatial/prior.pt
python spatial.py train unet --task inpaint --data "$DATA" --output outputs/spatial/unet-inpaint.pt
python spatial.py train unet --task sr --data "$DATA" --output outputs/spatial/unet-sr.pt

python spatial.py benchmark --data "$DATA" --task inpaint --method harmonic
python spatial.py benchmark --data "$DATA" --task sr --method bicubic
python spatial.py benchmark --data "$DATA" --task inpaint --method unet --checkpoint outputs/spatial/unet-inpaint.pt
python spatial.py benchmark --data "$DATA" --task inpaint --method dps --checkpoint outputs/spatial/prior.pt --split val --steps 50 --guidance 1
python spatial.py benchmark --data "$DATA" --task sr --method dps --checkpoint outputs/spatial/prior.pt --steps 50 --guidance 1 --draws 8 --output outputs/spatial/dps-sr.json
```

For a training-length comparison, add `--epochs 60 --save-epochs 20 40 60`
to the prior training command. This retains the best validation checkpoint
at the requested output path alongside exact-epoch snapshots such as
`prior-epoch-0020.pt`. These files are for inference; they do not store the
optimizer state for resuming training.

The sampling settings above are examples. Select step count and guidance on
validation data for each task and freeze them before the final test. The test
command defaults to all test frames. `--limit`, `--train-limit`, and
`--val-limit` support small runnable checks, whose scores are not benchmark
results. `--method ae` diagnoses autoencoder reconstruction error with
complete input and should be reported separately from the inverse problems.

The benchmark reports Kelvin MAE and per-frame RMSE, fixed-range SSIM,
finite-difference error per array cell, observation RMSE, and mean inference
time per draw. On CUDA it also reports peak allocated GPU memory. The scored
region is the hole for inpainting and the full
field for SR; inpainting also reports error across the hole boundary. For
multiple draws the report separates expected single-draw error and the error
of the mean prediction, and summarizes output spread in the evaluated region.
Inpainting observation RMSE is zero when known values are copied back.

For a measurement-only run, supply a 2-D Kelvin `.npy` with either the
`48×64` coarse observation or the `192×256` inpainting input (values inside the
fixed hole are ignored). No reference field or dataset is required:

```sh
python spatial.py reconstruct --task sr --checkpoint outputs/spatial/prior.pt --observation coarse.npy --steps 50 --output recovered.npy
```

For a validation-only warm-start ablation, run `spatial_sr_diagnostic.py`
with `--task sr` or `--task inpaint`, a trained prior checkpoint, the matching
`--unet-checkpoint`, and `--warm-strengths 0.1 0.25 --steps 20`. The initial
field comes from block-mean-corrected bicubic interpolation for SR, or harmonic
interpolation for inpainting. Each ratio retains a suffix of the DDPM timestep
grid and noises the encoded initial field at the first retained timestep.
The report compares interpolation, its autoencoder round trip, warm-start DPS,
and U-Net on the same validation frame from every case/angle pair. For SR,
compare the corrected variants, which all meet the same block-average input.

Add `--mmps-sigma-k 10 --mmps-maxiter 3` to that command for the exploratory
SR-only MMPS comparison. This uses the existing prior and a local
linearization of the decoder followed by block averaging. It propagates the
second Tweedie moment through this linearization and uses matrix-free GMRES
to weight the likelihood score. This is an adaptation of
[Rozet et al.](https://papers.neurips.cc/paper_files/paper/2024/file/9f94298bac4668db4dc77ddb0a244301-Paper-Conference.pdf),
whose observation model is linear in the state being diffused. The sigma is
a likelihood regularizer for synthetic observations, in Kelvin, not measured
sensor noise. The report includes per-step relative solver residuals and wall
times. The MMPS score uses the unclipped clean latent estimate; the DDPM
transition retains the scheduler's existing clipping. Compare the raw
observation error as well as the corrected full-field
and gradient errors; do not use the held-out test set to choose the sigma.

Once the settings are frozen, run `spatial_test.py` with the same prior and
the validated U-Net checkpoint for each task:

```sh
python spatial_test.py --task sr --data "$DATA" --device cuda \
  --prior-checkpoint outputs/spatial/prior.pt --unet-checkpoint outputs/spatial/unet-sr.pt \
  --output outputs/spatial/sr-test.json
```

The entry point uses the entire test split, with one seeded draw per input
for each diffusion method. It reports per-case and frame-weighted metrics for
interpolation, U-Net, pure-noise DPS and the selected warm-start DPS variant.
SR variants are compared before and after the same block-average correction;
the reported inpainting scores cover the hole and its boundary. Its JSON
contains checkpoint hashes, the split manifest, and the sampling settings.
Progress is saved after every completed case; only `status: "complete"` is a
final result. This test covers held-out frames from the nine available cases.

The conditional residual diffusion experiment is a separate SR-only path:
`spatial_residual_experiment.py` trains on the difference between the
reference and the frozen residual U-Net's block-average-corrected prediction.
The residual predictor conditions on that base and the coarse observation.
The generated addition has zero mean in each 4×4 block, so it preserves the
coarse input. This path does not use the AE or the unconditional MMPS prior.
Its validation command compares the frozen base with individual draws and the
mean of the draws on the same balanced 45-frame panel. Use these validation
results to decide whether to keep the diffusion stage before evaluating the
held-out test split. A smaller error of the sample mean should be reported
separately from the error and cost of an individual draw.

For center inpainting, `spatial_inpaint_experiment.py` trains masked residual
diffusion on top of a frozen inpainting U-Net. The convolutional variant and
the optional Transformer bottleneck use the same observed temperature field,
known-pixel mask, and base prediction as conditions. Samples only modify the
hole. Validation reports hole error, boundary-gradient error, distance-to-edge
bands, and per-case results for both individual draws and their mean.

On a CUDA instance with the data and the inpainting U-Net checkpoint in the
default locations, run `bash run_inpaint_ablation.sh`. The runner performs a
small CUDA smoke check for both variants before starting the matched training
runs and evaluating epochs 10, 20, and 30 on the same 45 validation frames.
Override the data and base checkpoint with `FLAMEDIFF_DATA` and
`FLAMEDIFF_INPAINT_UNET` if needed.
