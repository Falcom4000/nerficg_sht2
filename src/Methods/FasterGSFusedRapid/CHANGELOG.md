# FasterGSFusedRapid Changelog

## fastergsfusedrapid-v0.3.6 - 2026-05-10

Implementation/config changes:

- Added a fused CUDA `forward_image` path for no-grad FastGS score, metric-count, and inference renders.
- Templated the blend kernel so training renders still store backward-only bucket/transmittance/processed-count buffers, while no-grad image renders skip those allocations and writes.
- Exposed `_C.forward_image` through `rasterize_forward()`; `diff_rasterize()` and the fused backward/Adam training path are unchanged.
- Added `configs/fastergsfusedrapid_v0_3_6_forward_image/bicycle.yaml`, copied from v0.3.5 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_6_forward_image \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_6_forward_image_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.6 | bicycle | 185.31s | 1,250,930 | 25.5908 | 0.7576 | 0.2947 | 4.64GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,823 -> 138,439 | 0.7965 | 0.4590 | 1.2405 | 0.2979 | 0.0000 | 2.7939 |
| 14000-14100 | 1,360,824 -> 1,360,813 | 1.1020 | 0.4542 | 4.1620 | 0.3694 | 0.0000 | 6.0876 |
| 25000-25100 | 1,254,523 -> 1,254,523 | 1.0909 | 0.4563 | 3.8807 | 0.0000 | 0.0000 | 5.4278 |

Interpretation:

- Quality and Gaussian count stay in the same normal range as v0.3.5, so the forward-image API did not perturb training semantics.
- The score/pruning windows improve slightly (`14000-14100` densify/prune `0.3919ms -> 0.3694ms`), but total train time remains effectively flat (`185.83s -> 185.31s`). The remaining runtime is still dominated by normal training backward, not no-grad FastGS score rendering.
- This version is still about `20.0%` faster than the RapidGS bicycle reference (`231.77s`) with better PSNR/SSIM and worse LPIPS.

## fastergsfusedrapid-v0.3.5 - 2026-05-10

Implementation/config changes:

- Restored the densification-stage pruning budget selection to RapidGS-compatible `torch.multinomial` sampling after the deterministic v0.3.4 experiment did not improve the observed quality/speed tradeoff.
- Added a no-grad `rasterize_forward()` Python wrapper that calls the fused CUDA forward directly without constructing an autograd context.
- Switched FastGS score renders, metric-count renders, and inference renders to `rasterize_forward()`; training renders still use `diff_rasterize()` so fused backward/Adam semantics are unchanged.
- Added `configs/fastergsfusedrapid_v0_3_5_no_grad_forward/bicycle.yaml`.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_5_no_grad_forward \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_5_no_grad_forward_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.5 | bicycle | 185.83s | 1,248,302 | 25.6203 | 0.7575 | 0.2947 | 4.64GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,257 -> 138,366 | 0.7932 | 0.4625 | 1.2095 | 0.3236 | 0.0000 | 2.7887 |
| 14000-14100 | 1,358,116 -> 1,357,884 | 1.0775 | 0.4512 | 4.1357 | 0.3919 | 0.0000 | 6.0562 |
| 25000-25100 | 1,251,976 -> 1,251,976 | 1.0640 | 0.4533 | 3.8500 | 0.0000 | 0.0000 | 5.3674 |

Interpretation:

- The run used suite `fastergsfusedrapid_v0_3_5_no_grad_forward_bicycle_run2` because the original suite name already existed.
- Compared with v0.3.4, the no-grad direct forward path keeps quality and final Gaussian count effectively stable while train time moves from `182.43s` to `185.83s`; the difference is within the noise expected from restoring RapidGS multinomial pruning.
- Compared with the RapidGS bicycle reference (`231.77s`, `1,554,313` Gaussians, `PSNR 25.2623`, `SSIM 0.7555`, `LPIPS 0.2450`), v0.3.5 is about `19.8%` faster with better PSNR/SSIM, fewer Gaussians, and worse LPIPS.

## fastergsfusedrapid-v0.3.4 - 2026-05-10

Implementation/config changes:

- Ported FastGS multi-view score collection into `FasterGSFusedRapid` with a dedicated metric-count rasterization path.
- Added `render_image_fastgs_score()` and `render_image_metric_counts()` so densification/pruning score computation no longer reuses the inference render path with mismatched semantics.
- Reworked `adaptive_density_control()` to apply FastGS importance filtering before clone/split and to prune the split parents first, matching the BasisRapid FastGS flow.
- Changed multi-view pruning selection from random subsampling to deterministic top-score selection, which removes run-to-run pruning noise without changing the score budget.

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.4 | bicycle | 182.43s | 1,260,164 | 25.6559 | 0.7579 | 0.2946 | 4.58GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 112,282 -> 136,033 | 0.7656 | 0.4591 | 1.1943 | 0.3044 | 0.0000 | 2.7233 |
| 14000-14100 | 1,295,286 -> 1,296,689 | 1.0606 | 0.4526 | 3.9875 | 0.3917 | 0.0000 | 5.8924 |
| 25000-25100 | 1,264,302 -> 1,264,302 | 1.0677 | 0.4568 | 3.8835 | 0.0000 | 0.0000 | 5.4081 |

Interpretation:

- This version stays above the target speed improvement over the RapidGS bicycle reference while keeping quality notably closer than the earlier over-pruned v0.3.0/v0.3.1 direction.
- The biggest remaining quality gap is LPIPS, not raw training speed, so later work should focus on the pruning score/path rather than more density reduction.

## fastergsfusedrapid-v0.3.0 - 2026-05-10

Implementation/config changes:

- Ported the FastGS multi-view score/pruning loop from `FasterGSBasisRapid` into `FasterGSFusedRapid`.
- Added optional metric-count collection to the fused CUDA forward path:
  - `metric_map` is accepted by the Python/C++ rasterization API,
  - `metric_counts` are accumulated per Gaussian during forward blending,
  - training renders still skip metric-count allocation by passing an empty map.
- Added `FasterGSFusedRapidRenderer.render_image_metric_counts`.
- Added `compute_fastgs_scores`, importance-gated densification, and scheduled multi-view pruning callbacks to `FasterGSFusedRapidTrainer`.
- Extended fused `adaptive_density_control` with `importance_score` and `pruning_score`, and added `prune_by_multiview_score`.
- Added `configs/fastergsfusedrapid_v0_3_fastgs_pruning/bicycle.yaml`, copied from v0.2 and making the FastGS score/pruning fields explicit.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_fastgs_pruning \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_fastgs_pruning_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.0 | bicycle | 0.3234937323 | pending | pending | pending | pending | pending | pending |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pending | pending | pending | pending | pending | pending | pending | pending |

Interpretation:

- Pending benchmark. This is the first version with the FastGS multi-view pruning semantics ported into the fused optimizer backend.

## fastergsfusedrapid-v0.2.0 - 2026-05-10

Implementation/config changes:

- Changed `FasterGSFusedRapid` densification info from two channels to three channels, matching `FasterGSBasisRapid`:
  - visibility denominator,
  - signed 2D mean-gradient magnitude,
  - absolute 2D mean-gradient magnitude.
- Added absolute 2D mean-gradient accumulation in the fused bucket backward and propagated it through fused preprocess into `densification_info[2]`.
- Changed fused adaptive density control to use the BasisRapid split/clone decision:
  - clone candidates use signed mean-gradient threshold,
  - split candidates use absolute mean-gradient threshold,
  - `DENSIFICATION_PERCENT_DENSE` defaults to `0.001`.
- Added `DENSIFICATION_ABS_GRAD_THRESHOLD` to `FasterGSFusedRapidTrainer`.
- Added `configs/fastergsfusedrapid_v0_2_absgrad_density/bicycle.yaml`, copied from v0.1.1, with `PRELOADING_LEVEL: 2`, `DENSIFICATION_PERCENT_DENSE: 0.001`, and profiler enabled.
- This does not add FastGS multi-view pruning yet; the v0.2 goal is to isolate the core density-control change before adding metric-count rendering.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_2_absgrad_density \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_2_absgrad_density_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.2.0 | bicycle | 0.3234937323 | 227.30s | 1,952,145 | 25.6683 | 0.7643 | 0.2665 | 5.22GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 119,386 -> 146,803 | 0.7587 | 0.4581 | 1.1993 | 0.0378 | 0.0000 | 2.4539 |
| 14000-14100 | 1,934,983 -> 1,938,308 | 1.2899 | 0.4540 | 5.6692 | 0.1063 | 0.0000 | 7.5194 |
| 25000-25100 | 1,964,426 -> 1,964,426 | 1.3412 | 0.4544 | 5.7542 | 0.0000 | 0.0000 | 7.5497 |

Interpretation:

- The density-control change is the main speed win so far: train time drops from v0.1.1 `482.67s` to `227.30s`, with PSNR/SSIM essentially unchanged and LPIPS only slightly worse.
- Gaussian count drops from `4.39M` to `1.95M`; this brings fused rapid close to RapidGS (`231.77s`) but not yet to the desired 20% speed margin, and it is still slightly slower than FasterGSBasisRapid v0.9.0 (`221.66s`).
- Remaining bottleneck is still late backward at `5.75ms` with about `1.96M` Gaussians. The next target is to reduce final Gaussian count further without quality loss, either by tuning abs-gradient density thresholds or porting multi-view FastGS pruning.

## fastergsfusedrapid-v0.1.1 - 2026-05-10

Implementation/config changes:

- Added CUDA event profiler support to `FasterGSFusedRapidTrainer`, matching the window output format used by `FasterGSBasisRapid`.
- The profiler records render, loss, backward, densify/prune, optimizer, total, and Gaussian-count columns in `profile_windows.csv`.
- For this fused method, the CUDA Adam/preprocess work is launched from the custom backward path, so the `optimizer_ms` column is intentionally `0.0`; optimizer work is included in `backward_ms`.
- Added `configs/fastergsfusedrapid_v0_1_1_profile/bicycle.yaml`, copied from v0.1 and enabling the standard `1000-1100`, `14000-14100`, and `25000-25100` profiler windows.
- No CUDA kernel, densification, loss, or optimizer semantics changed from v0.1.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_1_1_profile \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_1_1_profile_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.1.1 | bicycle | 0.3234937323 | 482.67s | 4,390,027 | 25.6571 | 0.7636 | 0.2617 | 4.52GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 95,577 -> 117,161 | 1.0315 | 2.5433 | 1.4161 | 0.0416 | 0.0000 | 5.0325 |
| 14000-14100 | 4,336,489 -> 4,352,024 | 2.4856 | 2.8379 | 12.1555 | 0.2169 | 0.0000 | 17.6958 |
| 25000-25100 | 4,472,170 -> 4,472,170 | 2.4922 | 3.0484 | 12.3850 | 0.0000 | 0.0000 | 17.9257 |

Interpretation:

- The profiler confirms that v0.1's main speed issue is Gaussian growth: by the 14k window the model already has about `4.34M` Gaussians, and backward dominates at `12.16ms`.
- The fused CUDA Adam work is included in `backward_ms`; `optimizer_ms` remains zero as expected.
- Next migration target is not kernel micro-optimization but BasisRapid/FastGS density control: lower `DENSIFICATION_PERCENT_DENSE`, abs-gradient split support, and multi-view pruning to keep the Gaussian count near the RapidGS/BasisRapid range.

## fastergsfusedrapid-v0.1.0 - 2026-05-10

Implementation/config changes:

- Added `FasterGSFusedRapid` as an independent method copy of `FasterGSFused`, including a renamed Python method namespace and renamed CUDA extension package (`FasterGSFusedRapidCudaBackend`).
- Kept the existing FasterGSFused fused optimizer design: rasterization backward still writes gradients that are consumed by the fused CUDA Adam/preprocess path rather than switching to the FasterGSBasisRapid optimizer layout.
- Aligned the bucket backward color-remainder recurrence with RapidGS:
  - forward bucket state is interpreted as color/transmittance before each 32-Gaussian bucket,
  - backward initializes the per-pixel remainder as `bucket_color_before - final_color_without_background`,
  - each lane adds `alpha * transmittance * color` before computing the alpha gradient,
  - background alpha contribution remains `-final_transmittance * dot(grad_pixel, background) / (1 - alpha)`.
- Added edge-safe pixel guards in the bucket backward shared-memory load and per-lane work path, so partial image-edge tiles do not read image, gradient, transmittance, or contributor-count memory out of bounds.
- Added `configs/fastergsfusedrapid_v0_1_bucket_semantics/bicycle.yaml`, copied from `configs/fastergsfused_baseline/bicycle.yaml`, with `METHOD_TYPE: FasterGSFusedRapid` and RapidGS/BasisRapid-aligned `IMAGE_SCALE_FACTOR: 0.3234937323`.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_1_bucket_semantics \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_1_bucket_semantics_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | image scale | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.1.0 | bicycle | 0.3234937323 | 480.81s | 4,355,388 | 25.6758 | 0.7631 | 0.2626 | 4.49GiB |

Interpretation:

- v0.1 preserves FasterGSFused's fused-optimizer and high-Gaussian-count behavior while making the bucket backward recurrence match RapidGS/FasterGSBasisRapid v0.9.0 more closely.
- Compared with the earlier FasterGSFused bicycle reference at `IMAGE_SCALE_FACTOR: 0.25` (`442.56s`, `4.81M` Gaussians, PSNR `25.2828`, SSIM `0.7671`, LPIPS `0.2312`), this run uses a larger image scale (`0.3234937323`) and therefore is not a direct speed regression comparison.
- Quality remains in the expected range for the higher scale, but Gaussian count is still much larger than RapidGS/FasterGSBasisRapid because this fused variant does not add the FastGS pruning schedule used by FasterGSBasisRapid.
