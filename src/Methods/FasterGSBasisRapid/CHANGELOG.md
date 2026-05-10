# FasterGSBasisRapid Changelog

## fastergsbasisrapid-v0.6.0 - 2026-05-10

Implementation/config changes:

- Added `configs/fastergsbasisrapid_v0_6_vram_data/bicycle.yaml`, copied from v0.5 and bumped to `fastergsbasisrapid-v0.6.0`.
- Changed `TRAINING.DATA.PRELOADING_LEVEL` from `1` (RAM) to `2` (VRAM) for the bicycle benchmark config.
- This matches RapidGS `Camera(..., data_device="cuda")`, where `original_image` is stored on CUDA during scene construction.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSBasisRapid \
  --config-dir configs/fastergsbasisrapid_v0_6_vram_data \
  --repeats 1 \
  --suite-name fastergsbasisrapid_v0_6_vram_data_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsbasisrapid-v0.6.0 | bicycle | 0.3234937323 | 521.89s | 1,500,079 | 25.7339 | 0.7665 | 0.2772 | 5.46GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 130,934 -> 162,894 | 0.6904 | 0.4315 | 12.0700 | 0.2568 | 0.9082 | 14.3570 |
| 14000-14100 | 1,705,566 -> 1,705,032 | 1.1884 | 0.4298 | 14.6512 | 0.4674 | 1.8301 | 18.5670 |
| 25000-25100 | 1,505,583 -> 1,505,583 | 1.2002 | 0.4266 | 13.0011 | 0.0000 | 0.1470 | 14.7750 |

Interpretation:

- VRAM preloading reduces train time from `604.62s` to `521.89s` and keeps quality/Gaussian count stable.
- `loss_ms` drops from about `2.5-3.0ms` to `0.43ms`, matching the RapidGS data-device behavior.
- Peak allocated VRAM rises from `2.28GiB` to `5.46GiB`, which is expected because all training images now stay on CUDA.
- The remaining speed gap is dominated by rasterizer backward: `13-15ms` here versus about `4.6-4.8ms` in the RapidGS reference windows.

## fastergsbasisrapid-v0.5.0 - 2026-05-10

Implementation/config changes:

- Changed the `FasterGSBasisRapid` rasterizer API to accept `sh_coefficients_0` and `sh_coefficients_rest` separately.
- Removed the Python-side full-SH `torch.cat` path from training, metric-count rendering, and inference rendering.
- Changed CUDA backward to return separate gradients for DC SH and rest SH coefficients, matching the model's optimizer parameter split.
- Rebuilt and installed `FasterGSBasisRapidCudaBackend` with CUDA 12.8 after the API change.
- Added `configs/fastergsbasisrapid_v0_5_split_sh/bicycle.yaml`, copied from v0.4 and bumped to `fastergsbasisrapid-v0.5.0`.

Build note:

- `pip install --force-reinstall` must use CUDA 12.8 for this environment. CUDA 13.0 `nvcc` fails because PyTorch was compiled for CUDA 12.8.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSBasisRapid \
  --config-dir configs/fastergsbasisrapid_v0_5_split_sh \
  --repeats 1 \
  --suite-name fastergsbasisrapid_v0_5_split_sh_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsbasisrapid-v0.5.0 | bicycle | 0.3234937323 | 604.62s | 1,516,081 | 25.7350 | 0.7662 | 0.2778 |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 130,736 -> 162,595 | 0.9533 | 2.5107 | 12.0814 | 0.4724 | 0.9718 | 16.9895 |
| 14000-14100 | 1,723,448 -> 1,723,538 | 1.4982 | 3.0215 | 14.7271 | 0.7331 | 1.9533 | 21.9332 |
| 25000-25100 | 1,522,089 -> 1,522,089 | 1.4137 | 2.5331 | 13.1394 | 0.0000 | 0.1433 | 17.2296 |

Interpretation:

- Split SH reduces training time from `630.77s` to `604.62s` and lowers peak allocated VRAM from `2.52GiB` to `2.28GiB`.
- Render and metric-count score passes improve, but backward remains about `12-15ms` per profiled iteration, still roughly 3x slower than the RapidGS reference windows.
- The remaining gap is therefore in the BasisRapid rasterizer/backward kernel structure and/or autograd gradient materialization, not in SH concatenation alone.

## fastergsbasisrapid-v0.4.0 - 2026-05-10

Implementation/config changes:

- Added a RapidGS-style inline training loss path for `FasterGSBasisRapidLoss` when wandb logging is disabled.
- The inline path computes `0.8 * L1 + 0.2 * DSSIM` directly and skips `BaseLoss` per-iteration logging accumulation.
- Kept the existing `BaseLoss` path when wandb logging is enabled, so PSNR/loss logging behavior remains available for logging runs.
- Added `configs/fastergsbasisrapid_v0_4_inline_loss/bicycle.yaml`, copied from the v0.3 profile config and bumped to `fastergsbasisrapid-v0.4.0`.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSBasisRapid \
  --config-dir configs/fastergsbasisrapid_v0_4_inline_loss \
  --repeats 1 \
  --suite-name fastergsbasisrapid_v0_4_inline_loss_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsbasisrapid-v0.4.0 | bicycle | 0.3234937323 | 630.77s | 1,503,911 | 25.7365 | 0.7664 | 0.2779 |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 130,970 -> 162,617 | 1.0817 | 2.9899 | 12.2244 | 0.5400 | 1.0074 | 17.8435 |
| 14000-14100 | 1,711,746 -> 1,712,020 | 2.3772 | 2.7659 | 14.8145 | 0.8628 | 1.9016 | 22.7219 |
| 25000-25100 | 1,509,697 -> 1,509,697 | 2.2043 | 2.5823 | 13.1642 | 0.0000 | 0.1445 | 18.0954 |

Interpretation:

- Inline loss is functionally correct and keeps quality/Gaussian count stable.
- Training time only improves from `639.67s` to `630.77s`; the remaining speed gap is still dominated by rasterizer backward and training graph overhead.
- The next optimization target is the backend SH/rasterization interface, especially removing the Python-side full-SH `torch.cat` path and matching RapidGS split DC/rest inputs.

## fastergsbasisrapid-v0.3.0 - 2026-05-10

Implementation/config changes:

- Added an optional CUDA event profiler for `FasterGSBasisRapid` training.
- The profiler records three 1-based iteration windows by default: `[1000, 1100)`, `[14000, 14100)`, and `[25000, 25100)`.
- Per window, it writes averaged `render`, `loss`, `backward`, `densify_prune`, `optimizer`, and `total` milliseconds to `profile_windows.csv` in the run output directory.
- The profiler also records first/last/min/max Gaussian counts observed in each window.
- Added `configs/fastergsbasisrapid_v0_3_profile/bicycle.yaml` as a new profiling config, leaving the v0.2.1 benchmark config unchanged.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSBasisRapid \
  --config-dir configs/fastergsbasisrapid_v0_3_profile \
  --repeats 1 \
  --suite-name fastergsbasisrapid_v0_3_profile_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsbasisrapid-v0.3.0 | bicycle | 0.3234937323 | 639.67s | 1,508,992 | 25.7132 | 0.7662 | 0.2777 |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 131,084 -> 162,799 | 1.0474 | 3.2728 | 12.2262 | 0.5415 | 0.9921 | 18.0800 |
| 14000-14100 | 1,720,481 -> 1,720,321 | 2.4021 | 3.0632 | 14.9785 | 0.8579 | 1.9584 | 23.2601 |
| 25000-25100 | 1,514,775 -> 1,514,775 | 2.2022 | 2.6490 | 13.3041 | 0.0000 | 0.1441 | 18.2994 |

Interpretation:

- Quality and final Gaussian count remain close to the RapidGS `train_big.sh` reference.
- Compared with RapidGS bicycle profiling, `loss` is about 5-6x slower and `backward` is about 3x slower in the same iteration windows.
- The next optimization target is therefore the training loss/backward path, not densification parameters.

## fastergsbasisrapid-v0.2.1 - 2026-05-10

Implementation/config changes:

- Updated `scripts/benchmark_360v2.py` so method-specific `--config-dir` scene YAML files keep their explicit `DATASET.IMAGE_SCALE_FACTOR`.
- Documented the benchmark scale-preservation rule in `benchmarks_360v2_pipeline.md`.
- Bumped `configs/fastergsbasisrapid_v0_2_trainbig/bicycle.yaml` experiment version to `fastergsbasisrapid-v0.2.1`.

Experiments:

| version | implementation/config | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fastergsbasisrapid-v0.2-intermediate | v0.2 code before benchmark scale fix | bicycle | 0.25 | 425.02s | 1,289,798 | 25.2203 | 0.7573 | 0.2733 | Benchmark still overwrote explicit scale, so this is not comparable to RapidGS `train_big.sh` resolution. |
| fastergsbasisrapid-v0.2.1 | `configs/fastergsbasisrapid_v0_2_trainbig` + fixed benchmark script | bicycle | 0.3234937323 | 640.23s | 1,505,236 | 25.7492 | 0.7668 | 0.2772 | Uses RapidGS-like 1600-wide input scale and bicycle `grad_abs_thresh=0.0008`. |

Interpretation:

- Gaussian count now nearly matches the RapidGS `train_big.sh` bicycle reference (`1.505M` vs `1.564M`), so the remaining gap is unlikely to be primarily densification parameters.
- Training time is still much slower than RapidGS (`640.23s` vs `233.03s`) even with similar Gaussian count, pointing to implementation/backend/training-loop differences.
- Timing shows `training_iteration` dominates at `604s` total, `20.13ms/iter`; densification and VCP score passes are only about `12s` total.

## fastergsbasisrapid-v0.2 - 2026-05-10

Implementation changes:

- Matched RapidGS `optimizer_type=default` update semantics more closely.
- Split `sh_coefficients_rest` into a separate Adam optimizer, matching RapidGS `shoptimizer`.
- Added the RapidGS optimizer schedule:
  - iterations `1..15000`: update main parameters every iteration, update SH-rest every 16 iterations.
  - iterations `15001..20000`: update all optimizer groups every 32 iterations.
  - iterations `20001..29999`: update all optimizer groups every 64 iterations.
  - iteration `30000`: skip the optimizer step, matching RapidGS training loop behavior.
- Changed the default SH-rest learning rate to `0.00025`, matching RapidGS `highfeature_lr / 20 = 0.005 / 20`.

Config changes:

- Added `configs/fastergsbasisrapid_v0_2_trainbig/bicycle.yaml` as a new experiment config without modifying prior configs.
- `fastergsbasisrapid-v0.2` bicycle config records an `EXPERIMENT` block with version, baseline, changelog path, and notes.
- Set bicycle `DATASET.IMAGE_SCALE_FACTOR=0.3234937323`, which resizes the original `4946x3286` image to `1600x1063`, matching RapidGS default `--resolution -1` behavior for large input images.
- Set bicycle `TRAINING.DENSIFICATION_ABS_GRAD_THRESHOLD=0.0008`, matching `RapidGS/train_big.sh` for bicycle.

Reference experiments:

| version | implementation/config | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidGS train_big.sh | `/root/codes/RapidGS/train_big.sh` bicycle | bicycle | 233.03s | 1,563,535 | 25.2623 | 0.7555 | 0.2450 | User-run reference; input GT renders are `1600x1063`. |
| fastergsbasisrapid-v0.1 | `configs/fastergsbasisrapid_rapidgs_1` | bicycle | 475.38s | 884,029 | 25.0371 | 0.7390 | 0.3094 | Compact-box rasterizer, but still updated all PyTorch Adam groups every iteration and used `IMAGE_SCALE_FACTOR=0.25`. |
| fastergsfused-baseline | `configs/fastergsfused_baseline_2` average | bicycle | 442.56s | 4,810,568 | 25.2828 | 0.7671 | 0.2312 | Existing fused baseline average over 8 runs. |

## 2026-05-10

- Added FastGS/RapidGS-style multi-view scoring to `FasterGSBasisRapid`.
- Extended the CUDA rasterizer with an optional per-pixel metric map and per-Gaussian metric counts.
- Added metric-count rendering in `Renderer.py` without changing the default `diff_rasterize` image-only API.
- Gated densification with FastGS view-consistent importance scores.
- Aligned densification-time pruning with RapidGS: split-source Gaussians are pruned immediately, while low-opacity and large candidates use normalized pruning scores for budgeted sampling.
- Matched RapidGS opacity handling after densification by clamping opacity to at most 0.8.
- Extended densification info to `(3, N)` with count, signed 2D gradient norm, and absolute 2D gradient norm so clone and split can use RapidGS-style separate thresholds.
- Added post-densification multi-view pruning using normalized photometric score counts.
- Added configurable FastGS parameters for score views, loss threshold, importance threshold, absolute split-gradient threshold, pruning interval, pruning opacity, and pruning score threshold.
- Matched RapidGS defaults more closely by setting the method defaults to `DENSIFICATION_PERCENT_DENSE=0.001` and post-densification pruning at iterations 18k, 21k, 24k, and 27k.
- Added RapidGS compact-box tile coverage to the fused rasterizer and exposed `RENDERER.COMPACT_BOX_MULT` with the RapidGS default `0.5`.
- Added a separate `configs/fastergsbasisrapid_rapidgs_1` benchmark config copy for the RapidGS-aligned settings, leaving `configs/fastergsbasisrapid_1` as the prior experiment config.
- Verified forced backend rebuild, Python compilation, backend import, metric-count forward path, image-only backward path, `(3, N)` densification info updates, and metric-count autograd compatibility.
