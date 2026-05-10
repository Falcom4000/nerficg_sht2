# FasterGSFusedRapid Changelog

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
