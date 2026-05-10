# FasterGSBasisRapid Changelog

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
