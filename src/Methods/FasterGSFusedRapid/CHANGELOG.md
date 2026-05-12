# FasterGSFusedRapid Changelog

## fastergsfusedrapid-v0.4.15 - 2026-05-12

Config changes:

- Added `configs/fastergsfusedrapid_v0_4_15_early_vcp_18k/*.yaml`, copied from v0.4.14.
- Kept `TRAINING.NUM_ITERATIONS=18000`.
- Moved `DENSIFICATION_END_ITERATION` from `14900` to `14500`.
- Moved `MORTON_ORDERING_END_ITERATION` from `15000` to `14500`.
- Moved FastGS VCP pruning into the active training window: `FASTGS_PRUNING_START_ITERATION=15000`, `FASTGS_PRUNING_END_ITERATION=18000`, `FASTGS_PRUNING_INTERVAL=1000`.
- Moved the late profiler window from `25000-25100` to `16000-16100`.

Reason:

- In v0.4.14, `FASTGS_PRUNING_START_ITERATION=18000` does not materially trigger under the 18k training schedule.
- This config tests whether three post-densification VCP passes at 15k/16k/17k can reduce late Gaussian count without hurting quality.

Verification:

- Benchmark: `python ./scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_15_early_vcp_18k --repeats 3 --suite-name fastergsfusedrapid_v0_4_15_early_vcp_18k_r3`
- Suite output: `output/benchmarks/fastergsfusedrapid_v0_4_15_early_vcp_18k_r3`.

| scene | train time | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 135.6009s | 25.3498 | 0.7455 | 0.2971 | 1,476,364 | 4.8867GiB |
| bonsai | 84.2426s | 31.3253 | 0.9358 | 0.2582 | 403,222 | 5.7147GiB |
| counter | 79.6811s | 28.5032 | 0.8955 | 0.2829 | 269,494 | 4.9929GiB |
| garden | 91.0747s | 26.6876 | 0.8343 | 0.1900 | 860,350 | 2.9938GiB |
| kitchen | 93.0223s | 30.9470 | 0.9190 | 0.1745 | 391,933 | 5.5863GiB |
| room | 82.4126s | 31.3762 | 0.9129 | 0.3050 | 351,319 | 6.0448GiB |
| stump | 93.3365s | 25.7879 | 0.7270 | 0.3021 | 1,165,065 | 2.5723GiB |
| mean | 94.1958s | 28.5681 | 0.8529 | 0.2586 | 702,535 | 4.6845GiB |

Compared with v0.4.14:

| version | mean train | mean PSNR | mean SSIM | mean LPIPS | mean n_gaussians | mean VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.4.14 | 94.3403s | 28.5537 | 0.8531 | 0.2584 | 732,530 | 4.6878GiB |
| v0.4.15 | 94.1958s | 28.5681 | 0.8529 | 0.2586 | 702,535 | 4.6845GiB |

Interpretation:

- Early VCP reduced final Gaussian count by about `30k` on the all-scene mean and slightly reduced mean train time by `0.1444s`.
- Quality stayed in the normal repeat-3 range: PSNR improved by `+0.0145`, SSIM changed by `-0.0002`, and LPIPS changed by `+0.0002`.
- The speed gain is small because the added VCP score renders offset part of the reduced late backward cost.
- Next config target: keep the same number of VCP passes but lower the pruning score threshold to test whether more aggressive pruning improves late training speed without quality loss.

## fastergsfusedrapid-v0.4.14 - 2026-05-12

Implementation/config changes:

- Removed the maintained Metric3D/depth supervision path from `FasterGSFusedRapidTrainer`: no depth prior loading callback, no inverse-depth loss, and no `TRAINING.DEPTH_SUPERVISION` config surface.
- Removed inverse-depth rendering and inverse-depth gradient support from the maintained fused CUDA backend ABI, buffers, forward kernels, backward kernels, and Python autograd binding.
- Simplified training rasterizer calls to return RGB plus the autograd dummy only.
- Removed Metric3D options from the fast-converging prior scripts; `prepare_fast_converging_priors.py` now prepares AnySplat priors only.
- Updated AnySplat default checkpoint paths to `/root/codes/siggraph_asia/anySplat/config.json` and `/root/codes/siggraph_asia/anySplat/model.safetensors`.
- Allow the AnySplat prior workspace to replace an old generated `images/` directory with a symlink to the scene images.
- Added `configs/fastergsfusedrapid_v0_4_14_anysplat_only_no_depth/*.yaml`, copied from v0.4.12 with the depth-supervision block removed and experiment metadata updated.
- Removed the profiler `depth_loss_ms` column; `loss_ms` now equals the RGB loss timing.
- Updated `benchmarks_360v2_pipeline.md` so the maintained offline-to-training path is AnySplat-only.

Verification:

- Build: `pip install --force-reinstall --no-build-isolation src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend`
- ABI check: installed `_C.forward` no longer takes the old `render_inv_depth` boolean argument and now returns `image, metric_counts, primitive_buffers, tile_buffers, instance_buffers, bucket_buffers, n_instances, n_buckets, selector`.
- Dry-run prior check: `python scripts/prepare_fast_converging_priors.py dataset/mipnerf360/bicycle --tasks anysplat --dry-run`
- Smoke benchmark: `python ./scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_14_anysplat_only_no_depth --repeats 1 --suite-name fastergsfusedrapid_v0_4_14_anysplat_only_no_depth_smoke4 --scenes bicycle`
- Suite output: `output/benchmarks/fastergsfusedrapid_v0_4_14_anysplat_only_no_depth_smoke4`.
- Full benchmark: `python ./scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_14_anysplat_only_no_depth --repeats 3 --suite-name fastergsfusedrapid_v0_4_14_anysplat_only_no_depth_r3`
- Full suite output: `output/benchmarks/fastergsfusedrapid_v0_4_14_anysplat_only_no_depth_r3`.

| scene | train time | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 135.5439s | 25.3055 | 0.7450 | 0.2974 | 1,530,661 | 4.8935GiB |

All-scene repeat-3 result:

| scene | train time | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 135.7220s | 25.3103 | 0.7453 | 0.2971 | 1,534,332 | 4.8970GiB |
| bonsai | 84.8711s | 31.3792 | 0.9357 | 0.2575 | 422,402 | 5.7147GiB |
| counter | 79.3743s | 28.3819 | 0.8939 | 0.2839 | 281,798 | 4.9930GiB |
| garden | 91.1657s | 26.8063 | 0.8356 | 0.1894 | 885,973 | 2.9974GiB |
| kitchen | 92.1517s | 30.8779 | 0.9192 | 0.1743 | 411,060 | 5.5863GiB |
| room | 82.7529s | 31.2651 | 0.9128 | 0.3046 | 376,021 | 6.0448GiB |
| stump | 94.3442s | 25.8549 | 0.7292 | 0.3016 | 1,216,121 | 2.5811GiB |
| mean | 94.3403s | 28.5537 | 0.8531 | 0.2584 | 732,530 | 4.6878GiB |

Compared with v0.4.12 all-scene repeat-3:

| version | mean train | mean PSNR | mean SSIM | mean LPIPS | mean n_gaussians | mean VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.4.12 | 96.9419s | 28.5738 | 0.8532 | 0.2583 | 730,799 | 4.7432GiB |
| v0.4.14 | 94.3403s | 28.5537 | 0.8531 | 0.2584 | 732,530 | 4.6878GiB |

Profile windows:

- `1000-1100`: `render=0.8525ms`, `rgb_loss=0.4628ms`, `backward=2.5726ms`, `densify/prune=0.3369ms`, total `4.2248ms`.
- `14000-14100`: `render=1.0324ms`, `rgb_loss=0.4517ms`, `backward=4.5957ms`, `densify/prune=0.3850ms`, total `6.4648ms`.

Interpretation:

- Removing the depth backend restores a smaller RGB-only ABI and removes dead inverse-depth buffer/kernel branches.
- The single bicycle smoke run is faster than the v0.4.12/v0.4.13 bicycle repeat means while quality stays in the normal AnySplat-only range.
- The full all-scene repeat-3 run is `2.6016s` faster than v0.4.12 on mean train time, while PSNR changes by `-0.0201`, SSIM by `-0.0001`, and LPIPS by `+0.0001`.
- Gaussian count is effectively unchanged from v0.4.12 on the all-scene mean, so the speed difference is not caused by a substantial reduction in final model size.

## fastergsfusedrapid-v0.4.13 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_13_optional_depth_buffers/*.yaml`, copied from v0.4.12 with experiment metadata updated.
- Made `PrimitiveBuffers::from_blob` and `BucketBuffers::from_blob` optionally omit inverse-depth storage.
- In training forward, primitive and bucket inverse-depth buffers are allocated only when `render_inv_depth=true`.
- In image-only forward, primitive inverse-depth storage is omitted.
- In backward, buffer layout is parsed from whether the forward pass produced an `inv_depth` tensor, while depth-gradient kernel code is still selected from whether `grad_inv_depth` is non-empty.
- Added a guard for the invalid case where `grad_inv_depth` is non-empty but the forward pass did not allocate inverse-depth buffers.

Verification:

- Build: `python ./scripts/install.py -m FasterGSFusedRapid`
- Benchmark: `python scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_13_optional_depth_buffers --repeats 3 --suite-name fastergsfusedrapid_v0_4_13_optional_depth_buffers_r3`
- Suite output: `output/benchmarks/fastergsfusedrapid_v0_4_13_optional_depth_buffers_r3`.

| scene | v0.4.12 train | v0.4.13 train | delta | PSNR delta | SSIM delta | LPIPS delta | VRAM delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 139.2352s | 139.2777s | +0.0425s | -0.0252 | -0.0001 | +0.0004 | -0.0064GiB |
| bonsai | 87.6279s | 87.4210s | -0.2069s | +0.0534 | +0.0005 | -0.0002 | +0.0000GiB |
| counter | 82.6409s | 82.4069s | -0.2340s | +0.0125 | -0.0001 | +0.0002 | -0.0002GiB |
| garden | 92.9373s | 93.4396s | +0.5023s | +0.0667 | +0.0004 | -0.0002 | +0.0019GiB |
| kitchen | 96.0067s | 96.4273s | +0.4206s | +0.0777 | +0.0003 | -0.0005 | +0.0000GiB |
| room | 85.2873s | 85.1672s | -0.1200s | -0.0096 | -0.0001 | +0.0000 | +0.0000GiB |
| stump | 94.8581s | 95.0969s | +0.2389s | +0.0041 | +0.0002 | +0.0002 | +0.0013GiB |
| mean | 96.9419s | 97.0338s | +0.0919s | +0.0257 | +0.0001 | -0.0000 | -0.0004GiB |

Profile interpretation:

- v0.4.13 is not a speed improvement over v0.4.12: all-scene mean train time regressed by `+0.0919s`.
- Peak allocated VRAM changed by only `-0.0004GiB` on the all-scene mean, so the omitted inverse-depth temporary buffers are not a meaningful share of total memory under the current preloaded-image training path.
- The `14000-14100` total profiler window improved on `bicycle`, `bonsai`, `kitchen`, and `room`, but regressed on `counter`, `garden`, and `stump`.
- Keep the implementation because it preserves the correct depth/no-depth buffer layout and is useful if depth rendering is toggled, but keep v0.4.12 as the current speed-focused default.

## fastergsfusedrapid-v0.4.12 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_12_forward_depth_template/*.yaml`, copied from v0.4.11 with experiment metadata updated.
- Templated fused CUDA `preprocess_cu` and `blend_cu` on whether inverse-depth output is needed.
- When Metric3D/depth supervision is disabled, the forward path now compiles out per-primitive inverse-depth stores, per-fragment inverse-depth loads, bucket inverse-depth writes, and image inverse-depth writes.
- When depth rendering is enabled, the original inverse-depth output path is still instantiated.
- Updated `scripts/run_fastergsfusedrapid_fast_converging.py` defaults to the v0.4.12 config directory.

Verification:

- Build: `python ./scripts/install.py -m FasterGSFusedRapid`
- Benchmark: `python scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_12_forward_depth_template --repeats 3 --suite-name fastergsfusedrapid_v0_4_12_forward_depth_template_r3`
- Suite output: `output/benchmarks/fastergsfusedrapid_v0_4_12_forward_depth_template_r3`.

| scene | v0.4.11 train | v0.4.12 train | delta | PSNR delta | SSIM delta | LPIPS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 138.8349s | 139.2352s | +0.4003s | +0.0141 | -0.0004 | -0.0002 |
| bonsai | 87.8292s | 87.6279s | -0.2013s | +0.0265 | +0.0004 | -0.0004 |
| counter | 82.6424s | 82.6409s | -0.0015s | -0.0380 | +0.0004 | -0.0000 |
| garden | 93.4276s | 92.9373s | -0.4903s | -0.0012 | -0.0004 | +0.0007 |
| kitchen | 96.4586s | 96.0067s | -0.4519s | -0.0446 | -0.0001 | -0.0004 |
| room | 84.8136s | 85.2873s | +0.4736s | +0.1123 | +0.0002 | -0.0005 |
| stump | 94.9300s | 94.8581s | -0.0719s | +0.0069 | +0.0000 | -0.0001 |
| mean | 96.9909s | 96.9419s | -0.0490s | +0.0108 | +0.0001 | -0.0001 |

Profile interpretation:

- The all-scene repeat-3 result is a small but positive code-level cleanup: mean train time changed by `-0.0490s` from v0.4.11 and `-0.0565s` from v0.4.10.
- Quality stayed in the normal repeat variance range: mean PSNR `28.5738`, SSIM `0.8532`, LPIPS `0.2583`.
- The `14000-14100` total profiler window improved on `counter`, `garden`, `kitchen`, and `stump`, was nearly neutral on `bonsai`, and regressed slightly on `bicycle` and `room`.
- Keep v0.4.12 as the current code-level baseline because it preserves RGB semantics and removes depth-disabled forward work without changing training parameters.
- The next target is to avoid allocating primitive and bucket inverse-depth buffers entirely when forward did not render inverse depth.

## fastergsfusedrapid-v0.4.11 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_11_inv_depth_template/*.yaml`, copied from v0.4.10 with only experiment metadata changed.
- Templated fused CUDA `blend_backward_cu` and `preprocess_backward_cu` on `use_inv_depth_grad`.
- When Metric3D/depth supervision is disabled, the inverse-depth gradient path is compiled out of the backward kernels.
- When depth supervision is enabled, the original inverse-depth gradient path is still instantiated.

Verification:

- Build: `python ./scripts/install.py -m FasterGSFusedRapid`
- Benchmark: `python scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_11_inv_depth_template --repeats 3 --suite-name fastergsfusedrapid_v0_4_11_inv_depth_template_r3`
- Suite output: `output/benchmarks/fastergsfusedrapid_v0_4_11_inv_depth_template_r3`.

| scene | v0.4.10 train | v0.4.11 train | delta | PSNR delta | SSIM delta | LPIPS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 138.5501s | 138.8349s | +0.2848s | -0.0004 | +0.0002 | +0.0002 |
| bonsai | 87.6096s | 87.8292s | +0.2196s | -0.0534 | -0.0006 | +0.0003 |
| counter | 82.4686s | 82.6424s | +0.1738s | +0.0232 | -0.0001 | -0.0002 |
| garden | 92.8292s | 93.4276s | +0.5984s | -0.0602 | +0.0003 | -0.0007 |
| kitchen | 96.9229s | 96.4586s | -0.4642s | -0.0309 | -0.0001 | +0.0007 |
| room | 85.2407s | 84.8136s | -0.4271s | -0.0307 | -0.0001 | +0.0003 |
| stump | 95.3679s | 94.9300s | -0.4379s | +0.0058 | -0.0002 | +0.0001 |
| mean | 96.9984s | 96.9909s | -0.0075s | -0.0209 | -0.0001 | +0.0001 |

Profile interpretation:

- The result is effectively neutral end-to-end: all-scene mean train time changed by only `-0.0075s`.
- The `14000-14100` backward window improved in `bicycle`, `bonsai`, `counter`, `room`, and `stump`, but regressed in `garden` and `kitchen`.
- Keep the change as a semantic cleanup of the depth-disabled backward path, not as a proven speed improvement.
- The next higher-value target is the forward/preprocess path, which still computes and stores per-primitive inverse depth even when `render_inv_depth=false`.

## fastergsfusedrapid-v0.4.10 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_10_all_scenes_baseline/*.yaml` for all 7 Mip-NeRF 360 scenes.
- Kept the v0.4.9 AnySplat-only 18k training semantics: transformed AnySplat initialization, Metric3D depth supervision disabled, and normal SH-degree schedule.
- Updated `scripts/run_fastergsfusedrapid_fast_converging.py` defaults to the v0.4.10 config directory.
- Generated missing AnySplat PLY priors for `bonsai`, `counter`, `garden`, `kitchen`, `room`, and `stump` using `/root/codes/siggraph_asia/anySplat/model.safetensors` plus `/root/codes/siggraph_asia/VGGT-1B/model.safetensors`.

Verification:

- `python scripts/benchmark_360v2.py -m FasterGSFusedRapid --config-dir configs/fastergsfusedrapid_v0_4_10_all_scenes_baseline --repeats 3 --suite-name fastergsfusedrapid_v0_4_10_all_scenes_baseline_r3`
- Suite output: `output/benchmarks/fastergsfusedrapid_v0_4_10_all_scenes_baseline_r3`.

| scene | runs | train time mean | train time std | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 3 | 138.5501s | 0.2221s | 25.3187 | 0.7457 | 0.2969 | 1,532,846 | 4.8952GiB |
| bonsai | 3 | 87.6096s | 0.5755s | 31.4221 | 0.9363 | 0.2575 | 422,112 | 5.8015GiB |
| counter | 3 | 82.4686s | 0.3285s | 28.4717 | 0.8947 | 0.2837 | 280,897 | 5.1175GiB |
| garden | 3 | 92.8292s | 0.3152s | 26.8531 | 0.8353 | 0.1894 | 882,466 | 2.9965GiB |
| kitchen | 3 | 96.9229s | 0.7635s | 30.9089 | 0.9192 | 0.1742 | 411,888 | 5.6806GiB |
| room | 3 | 85.2407s | 1.3148s | 31.2798 | 0.9126 | 0.3049 | 373,885 | 6.1339GiB |
| stump | 3 | 95.3679s | 0.5070s | 25.8333 | 0.7286 | 0.3017 | 1,212,369 | 2.5786GiB |
| mean | 21 | 96.9984s | 18.1088s | 28.5839 | 0.8532 | 0.2583 | 730,923 | 4.7434GiB |

Profile interpretation:

- Repeat variance is low enough for full-scene comparisons: most scene train-time std is below `1s`, with `room` at `1.31s`.
- Mid-training windows are still dominated by fused backward/Adam work, not FastGS scoring. At `14000-14100`, representative total/backward means are `bicycle 6.7786/4.8140ms`, `garden 4.8729/3.3702ms`, and `stump 5.2191/3.7213ms`.
- Densify/prune cost is small in the measured windows, generally `0.26-0.39ms`, so the next code-level target should be backward memory traffic or per-Gaussian fused optimizer work rather than changing densification parameters.

## fastergsfusedrapid-v0.4.9 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_9_anysplat_only_18k_sh_schedule/bicycle.yaml`.
- Kept v0.4.8's AnySplat-only 18k schedule, but set `TRAINING.ANYSPLAT_INITIALIZATION.SET_ACTIVE_SH_DEGREE=false`.
- Updated `scripts/run_fastergsfusedrapid_fast_converging.py` defaults to the v0.4.9 config.

Reason:

- AnySplat's own `GaussianAdapter` leaves SH rotation commented out, and the PLY exporter does not rotate harmonics into the transformed world frame.
- Starting from SH degree 0 and using the normal training SH schedule is closer to standard 3DGS training semantics than activating all imported SH coefficients at iteration 0.

Verification:

- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --skip-prior-generation --prior-mode none --config-dir configs/fastergsfusedrapid_v0_4_9_anysplat_only_18k_sh_schedule --suite-name fastergsfusedrapid_v0_4_9_anysplat_only_18k_sh_schedule_bicycle --repeats 1`
- Result: train `139.3274s`, wall `197.6210s`, PSNR `25.3361`, SSIM `0.7460`, LPIPS `0.2966`, final Gaussians `1,528,062`, VRAM allocated/reserved `4.8892/5.4062 GiB`.
- Profile windows:
  - `1000-1100`: `render=0.9298ms`, `rgb_loss=0.4638ms`, `depth_loss=0.0021ms`, `backward=2.7022ms`, `densify/prune=0.3528ms`, total `4.4507ms`, Gaussians `827,004 -> 861,776`.
  - `14000-14100`: `render=1.1200ms`, `rgb_loss=0.4518ms`, `depth_loss=0.0019ms`, `backward=4.8227ms`, `densify/prune=0.3880ms`, total `6.7844ms`, Gaussians `1,610,039 -> 1,609,128`.

Interpretation:

- v0.4.9 is now the recommended speed-focused config: it keeps v0.4.8 speed while improving PSNR, SSIM, and LPIPS.
- Compared with v0.3.14, v0.4.9 is `23.7%` faster in measured train time (`182.59s -> 139.33s`) with quality still in the normal single-run range.

## fastergsfusedrapid-v0.4.8 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_8_anysplat_only_18k/bicycle.yaml`.
- Kept the v0.4.5 AnySplat-only semantics and reduced only `TRAINING.NUM_ITERATIONS` from `30000` to `18000`.
- Updated `scripts/run_fastergsfusedrapid_fast_converging.py` defaults to the AnySplat-only 18k config and `--prior-mode anysplat`.

Baseline context:

| version | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.10 | 185.49s | 1,247,748 | 25.6526 | 0.7579 | 0.2945 | 4.64GiB |
| fastergsfusedrapid-v0.3.11 | 184.47s | 1,246,971 | 25.6292 | 0.7580 | 0.2941 | 4.64GiB |
| fastergsfusedrapid-v0.3.12 | 181.56s | 1,245,422 | 25.6352 | 0.7579 | 0.2951 | 4.63GiB |
| fastergsfusedrapid-v0.3.13 | 181.95s | 1,254,289 | 25.6297 | 0.7581 | 0.2946 | 4.64GiB |
| fastergsfusedrapid-v0.3.14 | 182.59s | 1,246,630 | 25.6324 | 0.7582 | 0.2941 | 4.63GiB |

Verification:

- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --skip-prior-generation --prior-mode none --config-dir configs/fastergsfusedrapid_v0_4_8_anysplat_only_18k --suite-name fastergsfusedrapid_v0_4_8_anysplat_only_18k_bicycle --repeats 1`
- Result: train `139.1008s`, wall `197.7430s`, PSNR `25.3164`, SSIM `0.7441`, LPIPS `0.2988`, final Gaussians `1,502,571`, VRAM allocated/reserved `4.8641/5.3477 GiB`.
- Profile windows:
  - `1000-1100`: `render=0.9916ms`, `rgb_loss=0.4744ms`, `depth_loss=0.0032ms`, `backward=3.1665ms`, `densify/prune=0.3654ms`, total `5.0010ms`, Gaussians `826,493 -> 860,771`.
  - `14000-14100`: `render=1.1312ms`, `rgb_loss=0.4513ms`, `depth_loss=0.0020ms`, `backward=4.8138ms`, `densify/prune=0.3885ms`, total `6.7868ms`, Gaussians `1,587,773 -> 1,587,273`.

Interpretation:

- v0.4.8 is the current speed-focused recommendation: it is `23.4%` faster than v0.3.14 train time while keeping bicycle quality in the normal range for a single run.
- Quality is lower than the v0.3.10-v0.3.14 baseline in PSNR/SSIM, so v0.4.6 20k remains the more conservative speed/quality point.
- Metric3D is not part of this recommended path because the split experiments below showed no clear quality gain and a bad interaction with AnySplat.

## fastergsfusedrapid-v0.4.7 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_7_anysplat_only_15k/bicycle.yaml`.
- Kept the v0.4.5 AnySplat-only semantics and reduced only `TRAINING.NUM_ITERATIONS` from `30000` to `15000`.

Verification:

- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --skip-prior-generation --prior-mode none --config-dir configs/fastergsfusedrapid_v0_4_7_anysplat_only_15k --suite-name fastergsfusedrapid_v0_4_7_anysplat_only_15k_bicycle --repeats 1`
- Result: train `120.4300s`, wall `178.5623s`, PSNR `25.1440`, SSIM `0.7390`, LPIPS `0.3092`, final Gaussians `1,507,526`, VRAM allocated/reserved `4.8669/5.3477 GiB`.

Interpretation:

- 15k is very fast, but quality drops more than the 18k/20k configs and should not be the default recommendation.

## fastergsfusedrapid-v0.4.6 - 2026-05-11

Implementation/config changes:

- Added `configs/fastergsfusedrapid_v0_4_6_anysplat_only_20k/bicycle.yaml`.
- Kept the v0.4.5 AnySplat-only semantics and reduced only `TRAINING.NUM_ITERATIONS` from `30000` to `20000`.

Verification:

- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --skip-prior-generation --prior-mode none --config-dir configs/fastergsfusedrapid_v0_4_6_anysplat_only_20k --suite-name fastergsfusedrapid_v0_4_6_anysplat_only_20k_bicycle --repeats 1`
- Result: train `150.4008s`, wall `209.7916s`, PSNR `25.3172`, SSIM `0.7464`, LPIPS `0.2966`, final Gaussians `1,463,917`, VRAM allocated/reserved `4.8590/5.3672 GiB`.
- Profile windows:
  - `1000-1100`: `render=0.9360ms`, `rgb_loss=0.4604ms`, `depth_loss=0.0018ms`, `backward=3.1376ms`, `densify/prune=0.3552ms`, total `4.8910ms`.
  - `14000-14100`: `render=1.1151ms`, `rgb_loss=0.4508ms`, `depth_loss=0.0016ms`, `backward=4.7993ms`, `densify/prune=0.3782ms`, total `6.7450ms`.

Interpretation:

- 20k is the conservative AnySplat-only speed/quality point: `17.6%` faster than v0.3.14 with LPIPS close to the baseline and SSIM lower but still in a normal range.

## fastergsfusedrapid-v0.4.5 - 2026-05-11

Implementation/config changes:

- Applied the Mip-NeRF 360 dataset `world_transform` to AnySplat Gaussian means, log-scales, and rotations before installing the PLY state into `FasterGSFusedRapid`.
- Stored the Mip-NeRF 360 PCA/rescale transform on the dataset so external Gaussian initializers use the same training coordinate system as cameras and the COLMAP point cloud.
- Moved Metric3D depth-prior attachment before the generic training-data preload callback, so `PRELOADING_LEVEL=2` also preloads depth priors into VRAM.
- Cached per-view Metric3D valid masks and changed depth L1 from boolean indexing to dense masked reduction.
- Split profiler loss timing into `rgb_loss_ms` and `depth_loss_ms`, while keeping aggregate `loss_ms`.
- Added isolated configs:
  - `configs/fastergsfusedrapid_v0_4_5_anysplat_only`
  - `configs/fastergsfusedrapid_v0_4_5_depth_only`
  - `configs/fastergsfusedrapid_v0_4_5_both_world_transform`
- Added `--prior-mode {both,metric3d,anysplat,none}` to `scripts/run_fastergsfusedrapid_fast_converging.py`.

Reason:

- v0.4.4 loaded AnySplat PLY coordinates in original COLMAP space while Mip-NeRF 360 cameras and the dataset point cloud were transformed by PCA/rescale. That mismatch explains the bad bicycle quality, excessive densification, and poor GPU utilization more directly than parameter choices.
- v0.4.4 also attached depth priors after the standard preload stage, so depth `.npy` files could be loaded during training instead of once before training.
- AnySplat and Metric3D are now isolated in separate configs because enabling both at once made quality/performance attribution ambiguous.

Verification:

- `python -m py_compile src/Datasets/MipNeRF360.py src/Methods/FasterGSFusedRapid/Model.py src/Methods/FasterGSFusedRapid/Trainer.py scripts/run_fastergsfusedrapid_fast_converging.py`
- Coordinate sanity check confirmed the transformed AnySplat PLY now matches the transformed COLMAP point-cloud coordinate frame.
- AnySplat-only 30k: train `211.3436s`, wall `270.1355s`, PSNR `25.3423`, SSIM `0.7484`, LPIPS `0.2916`, final Gaussians `1,450,320`, VRAM allocated/reserved `4.8661/5.3672 GiB`.
- Depth-only 30k: train `202.6496s`, wall `263.9578s`, PSNR `25.2751`, SSIM `0.7426`, LPIPS `0.3102`, final Gaussians `1,303,899`, VRAM allocated/reserved `6.8822/7.5938 GiB`.
- AnySplat+Depth 30k: train `243.8765s`, wall `303.2136s`, PSNR `24.3770`, SSIM `0.7036`, LPIPS `0.2967`, final Gaussians `1,914,620`, VRAM allocated/reserved `7.6076/8.5898 GiB`.

Interpretation:

- The coordinate transform fix restored AnySplat-only quality to the normal range.
- Metric3D depth supervision did not improve bicycle quality in this setup and increased VRAM; combined AnySplat+Depth caused a clear PSNR/SSIM regression and Gaussian-count growth.
- The recommended path after this split is AnySplat-only plus a shorter training schedule, captured in v0.4.6-v0.4.8.

## fastergsfusedrapid-v0.4.4 - 2026-05-11

Implementation/config changes:

- Updated `Gaussians.initialize_from_ply` to accept AnySplat PLY files with more SH coefficients than the configured `MODEL.SH_DEGREE`.
- When the PLY contains higher-order SH terms, the loader now keeps the lowest coefficients that fit the configured FasterGSFusedRapid SH degree and logs a warning instead of failing before optimization.
- Added `configs/fastergsfusedrapid_v0_4_4_anysplat_sh_truncation/bicycle.yaml`, copied from v0.4.3 with updated experiment metadata.
- Updated `scripts/run_fastergsfusedrapid_fast_converging.py` defaults to v0.4.4.

Reason:

- The local AnySplat PLY generated from VGGT-1B contains degree-4 SH (`72` `f_rest_*` values), while the current FasterGSFusedRapid benchmark config uses `MODEL.SH_DEGREE=3` (`45` rest coefficients).
- Truncating high-order coefficients preserves the configured renderer/training SH contract and avoids silently changing the benchmark model capacity.

Expected use:

```bash
python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_4_4_anysplat_sh_truncation_bicycle
```

Verification:

- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --skip-prior-generation --repeats 1 --suite-name fastergsfusedrapid_v0_4_4_anysplat_sh_truncation_bicycle_rebuilt`
- Result: train `422.7519s`, wall `485.3620s`, PSNR `21.6589`, SSIM `0.5126`, LPIPS `0.4228`, final Gaussians `2,206,821`, VRAM allocated/reserved `5.8429/6.5391 GiB`.
- Profile windows:
  - `1000-1100`: `render=0.8212ms`, `loss=4.9832ms`, `backward=3.4921ms`, `densify/prune=0.3577ms`, total `9.6542ms`, Gaussians `834,352 -> 857,490`.
  - `14000-14100`: `render=1.3068ms`, `loss=5.8255ms`, `backward=7.2144ms`, `densify/prune=0.4539ms`, total `14.8007ms`, Gaussians `2,496,286 -> 2,499,152`.
  - `25000-25100`: `render=1.2497ms`, `loss=4.9297ms`, `backward=6.6263ms`, total `12.8057ms`, Gaussians `2,216,418`.

Interpretation:

- The run completes, but quality is outside the normal bicycle range and training is much slower than v0.3.x.
- Root causes identified for v0.4.5: AnySplat PLY coordinate-frame mismatch with Mip-NeRF 360 PCA/rescale, and Metric3D priors not being attached early enough for normal data preloading.

## fastergsfusedrapid-v0.4.3 - 2026-05-11

Implementation/config/script changes:

- Added `configs/fastergsfusedrapid_v0_4_3_scaled_metric3d_priors/bicycle.yaml`, copied from v0.4.2 with `TRAINING.DEPTH_SUPERVISION.PRESCALED_TO_TRAINING_RESOLUTION=true`.
- `scripts/prepare_fast_converging_priors.py` can now generate the Metric3D workspace at a requested image scale through `--metric3d-image-scale`, including matching COLMAP PINHOLE intrinsics in the text model.
- Metric3D `.npy` priors are converted to float32 after inference by default through `--metric3d-output-dtype float32`.
- Removed the misleading `--vis False` Metric3D argument; the upstream parser treats non-empty strings as true.
- Added explicit `--vggt-weights` plumbing for the local AnySplat encoder. The default path is `/root/codes/siggraph_asia/VGGT-1B/model.safetensors`.
- AnySplat runs with `ANY_SPLAT_VGGT_WEIGHTS` set to that local VGGT file and `HF_HUB_OFFLINE=1`, so it does not try to download `facebook/VGGT-1B` during prior generation.
- Added a narrow Open3D import stub for the AnySplat trajectory-alignment path; this path only uses `estimate_similarity_transform` and does not call Open3D APIs.
- Isolated the `utils` namespace during AnySplat import so `/root/codes/siggraph_asia/utils/anysplat_utils.py` is used instead of the repo-local `scripts/utils.py`.
- Updated `scripts/run_fastergsfusedrapid_fast_converging.py` defaults to v0.4.3 and added `--vggt-weights`.

Expected use:

```bash
python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --metric3d-weights /root/codes/siggraph_asia/metric3D/weight/metric_depth_vit_giant2_800k.pth \
  --anysplat-weights /root/codes/siggraph_asia/anySplat/model.safetensors \
  --vggt-weights /root/codes/siggraph_asia/VGGT-1B/model.safetensors
```

Verification so far:

- `python -m py_compile scripts/prepare_fast_converging_priors.py scripts/run_fastergsfusedrapid_fast_converging.py src/Methods/FasterGSFusedRapid/Trainer.py`
- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --dry-run-priors --prepare-only --anysplat-weights /root/codes/siggraph_asia/anySplat/model.safetensors`
- Metric3D bicycle prior generation completed at training resolution: 169 files, shape `1063x1600`, dtype `float32`, total `1.1G`.

Pending:

- AnySplat PLY generation is waiting for the local VGGT-1B file at `/root/codes/siggraph_asia/VGGT-1B/model.safetensors`.
- Full bicycle benchmark should run after AnySplat writes `<scene>/anysplat_init/point_cloud.ply`.

## fastergsfusedrapid-v0.4.2 - 2026-05-11

Implementation/config/script changes:

- Integrated AnySplat Gaussian initialization into the `FasterGSFusedRapid` training setup.
- Added 3DGS-compatible PLY loading to `Gaussians.initialize_from_ply`, preserving means, DC/rest SH coefficients, raw opacity logits, raw log scales, and rotations from AnySplat output.
- Added `TRAINING.ANYSPLAT_INITIALIZATION` config fields: `ACTIVE`, `PATH`, `REQUIRE`, and `SET_ACTIVE_SH_DEGREE`.
- Training now resolves relative AnySplat PLY paths against the dataset scene root.
- When `ANYSPLAT_INITIALIZATION.REQUIRE` is true, missing AnySplat PLY files fail before optimization instead of silently falling back to COLMAP/random initialization.
- Added `configs/fastergsfusedrapid_v0_4_2_fast_converging/bicycle.yaml` with Metric3D depth supervision and AnySplat initialization enabled.
- Added `scripts/run_fastergsfusedrapid_fast_converging.py` to run the offline prior stage and the matching benchmark stage serially.
- Updated `benchmarks_360v2_pipeline.md` with the full offline-to-training workflow.

Expected use:

```bash
python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --metric3d-weights /path/to/metric_depth_vit_giant2_800k.pth \
  --anysplat-weights /path/to/model.safetensors
```

Verification:

- `python -m py_compile src/Methods/FasterGSFusedRapid/Model.py src/Methods/FasterGSFusedRapid/Trainer.py scripts/prepare_fast_converging_priors.py scripts/run_fastergsfusedrapid_fast_converging.py`
- `python scripts/run_fastergsfusedrapid_fast_converging.py --scene bicycle --dry-run-priors --prepare-only`

Interpretation:

- Metric3D is now integrated through offline `.npy` prior generation plus training-time inverse-depth supervision.
- AnySplat is now integrated through offline PLY generation plus training-time Gaussian state initialization.
- Full inference still requires valid local Metric3D and AnySplat checkpoint files at the configured paths.

## fastergsfusedrapid-v0.4.1 - 2026-05-11

Implementation/script changes:

- Added `scripts/prepare_fast_converging_priors.py` for offline Fast-Converging 3DGS prior preparation.
- Scoped the script to Mip-NeRF 360 scenes for now.
- The script prepares a shared COLMAP text-model workspace and writes the same Mip-NeRF 360 train/test split used by the Metric3D and AnySplat helpers.
- Exposed Metric3D model inputs through `--metric3d-config` and `--metric3d-weights`; defaults point at `/root/codes/siggraph_asia/metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py` and `/root/codes/siggraph_asia/metric3D/weight/metric_depth_vit_giant2_800k.pth`.
- Exposed AnySplat model inputs through `--anysplat-config` and `--anysplat-weights`; defaults point at `/root/codes/siggraph_asia/anySplat/ckpt/config.json` and `/root/codes/siggraph_asia/anySplat/ckpt/model.safetensors`.
- Removed VGGT-specific configuration from this pipeline; the local AnySplat wrapper is treated as the feed-forward Gaussian source.
- Metric3D outputs are expected at `<scene>/mono_depths/<image_stem>_depth.npy`.
- AnySplat outputs are expected at `<scene>/anysplat_init/point_cloud.ply`.

Expected use:

```bash
python scripts/prepare_fast_converging_priors.py \
  dataset/mipnerf360/bicycle \
  --tasks metric3d anysplat \
  --metric3d-weights /path/to/metric_depth_vit_giant2_800k.pth \
  --anysplat-weights /path/to/model.safetensors
```

Verification:

- `python -m py_compile scripts/prepare_fast_converging_priors.py`
- `python scripts/prepare_fast_converging_priors.py dataset/mipnerf360/bicycle --dry-run --tasks metric3d anysplat`

Interpretation:

- This is an offline preparation step, not part of timed training.
- The dry-run validates workspace generation and command wiring, but full inference still requires the Metric3D and AnySplat checkpoint files to exist at the configured paths.
- FasterGSFusedRapid v0.4.0 already consumes Metric3D inverse-depth priors. Loading the AnySplat Gaussian initialization into the trainer remains a follow-up integration step.

## fastergsfusedrapid-v0.4.0 - 2026-05-11

Implementation/config changes:

- Added an optional Metric3D inverse-depth supervision path for training.
- The fused CUDA training rasterizer can now output blended inverse depth using the same front-to-back alpha weights as RGB.
- Added inverse-depth bucket state so backward can reconstruct per-pixel depth remainders and propagate depth gradients through alpha, opacity, conic/mean2D, and centroid depth.
- Added a per-Gaussian inverse-depth helper gradient in preprocess backward, applying `d(1 / z) / dz = -1 / z^2` to the camera-space mean gradient.
- Kept inverse-depth rendering disabled unless `TRAINING.DEPTH_SUPERVISION.ACTIVE` is true, preserving normal RGB-only training behavior.
- Added optional Metric3D prior loading from `mono_depths/<image_stem>_depth.npy` for Mip-NeRF 360 style image paths.
- Added `TRAINING.DEPTH_SUPERVISION` config fields: `ACTIVE`, `DIRECTORY`, `WEIGHT_INIT`, `WEIGHT_FINAL`, `LOSS_MULTIPLIER`, and `MIN_VALID_INV_DEPTH`.
- Added `configs/fastergsfusedrapid_v0_4_0_depth_prior/bicycle.yaml`, copied from v0.3.14 with depth supervision enabled.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_4_0_depth_prior \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_4_0_depth_prior_bicycle \
  --scenes bicycle
```

Verification:

- `python -m py_compile src/Methods/FasterGSFusedRapid/Trainer.py src/Methods/FasterGSFusedRapid/Renderer.py src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/torch_bindings/rasterization.py`
- `python ./scripts/install.py -m FasterGSFusedRapid`

Interpretation:

- This is the first Metric3D prior plumbing commit. It does not generate priors itself; if `mono_depths` files are absent, the trainer logs missing priors and skips the depth term for those views.
- Full final-round parity also needs AnySplat feed-forward Gaussian initialization, which belongs in an offline preparation step rather than inside the timed training loop.

## fastergsfusedrapid-v0.3.14 - 2026-05-11

Implementation/config changes:

- Cached renderer-owned empty sentinel tensors for `metric_map=None` and inactive `densification_info` paths.
- Training renders now pass a cached empty bool metric map instead of letting the Python binding allocate one per call.
- Post-densification training renders now pass a cached empty float tensor instead of constructing `torch.empty(0)` per call.
- No-grad score and inference renders also reuse the cached empty bool metric map when no metric counts are requested.
- The C++ wrappers now use the metric-map pointer directly after validating non-empty maps as contiguous CUDA bool tensors, avoiding a redundant `.contiguous()` wrapper on score/prune metric-count renders.
- The Python binding keeps cached empty fallback tensors for direct calls that bypass the renderer-level sentinel cache.
- Added `configs/fastergsfusedrapid_v0_3_14_cached_empty_tensors/bicycle.yaml`, copied from v0.3.13 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_14_cached_empty_tensors \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_14_cached_empty_tensors_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.14 | bicycle | 182.59s | 1,246,630 | 25.6324 | 0.7582 | 0.2941 | 4.63GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,630 -> 138,586 | 0.6763 | 0.4625 | 1.2206 | 0.2852 | 0.0000 | 2.6447 |
| 14000-14100 | 1,356,005 -> 1,356,015 | 0.9711 | 0.4545 | 4.1219 | 0.3466 | 0.0000 | 5.8941 |
| 25000-25100 | 1,250,318 -> 1,250,318 | 1.0257 | 0.4599 | 3.9057 | 0.0000 | 0.0000 | 5.3913 |

Interpretation:

- This is a Python/CUDA boundary cleanup: it removes repeated zero-size tensor allocations and redundant metric-map contiguous wrappers without changing the CUDA kernels or FastGS/RapidGS density, pruning, and rendering semantics.
- Expected impact is mostly outside kernel math and should show up as small end-to-end train-time or wrapper-overhead changes.
- Single-run train time was slower than v0.3.13 (`181.95s -> 182.59s`), while quality and Gaussian count stayed normal.
- The measured profiler windows improved in early/mid render and densify/prune setup (`14000-14100` total `5.9593ms -> 5.8941ms`) but regressed in the late backward-heavy window. Treat this as a small wrapper cleanup with noisy end-to-end timing, not as a proven speedup.

## fastergsfusedrapid-v0.3.13 - 2026-05-11

Implementation/config changes:

- Removed repeated C++ `.contiguous()` calls for `w2c`, camera position, and background color in fused forward, no-grad forward image, and backward wrappers.
- Added debug-mode input checks for those camera/background tensors before taking direct data pointers.
- Relies on v0.3.12 renderer-side cached pose tensors and existing contiguous background tensors; rasterization math and training parameters are unchanged.
- Added `configs/fastergsfusedrapid_v0_3_13_direct_camera_pointers/bicycle.yaml`, copied from v0.3.12 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_13_direct_camera_pointers \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_13_direct_camera_pointers_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.13 | bicycle | 181.95s | 1,254,289 | 25.6297 | 0.7581 | 0.2946 | 4.64GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,712 -> 138,425 | 0.6893 | 0.4618 | 1.2357 | 0.2825 | 0.0000 | 2.6693 |
| 14000-14100 | 1,365,590 -> 1,365,545 | 0.9973 | 0.4525 | 4.1485 | 0.3610 | 0.0000 | 5.9593 |
| 25000-25100 | 1,258,071 -> 1,258,071 | 1.0185 | 0.4512 | 3.8596 | 0.0000 | 0.0000 | 5.3293 |

Interpretation:

- The direct pointer change is semantically neutral under the v0.3.12 renderer contract: camera pose tensors are cached contiguous CUDA tensors, and background tensors are already contiguous CUDA float tensors.
- Single-run train time was slightly slower than v0.3.12 (`181.56s -> 181.95s`), but the delta is small and quality/Gaussian count remain normal.
- Keep this as a wrapper cleanup; future comparisons should treat its observed timing difference as noise unless repeated runs show a consistent trend.

## fastergsfusedrapid-v0.3.12 - 2026-05-11

Implementation/config changes:

- Cached static per-view pose tensors (`w2c`, camera position) inside the FasterGSFusedRapid renderer.
- The cache key uses the backing `View._c2w` object identity plus the configured default device, so normal dataset-view setters invalidate by replacing `_c2w`.
- Left background color, image tensors, rasterization parameters, and FastGS density/pruning thresholds unchanged.
- Added `configs/fastergsfusedrapid_v0_3_12_cached_view_pose/bicycle.yaml`, copied from v0.3.11 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_12_cached_view_pose \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_12_cached_view_pose_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.12 | bicycle | 181.56s | 1,245,422 | 25.6352 | 0.7579 | 0.2951 | 4.63GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 112,930 -> 137,367 | 0.6990 | 0.4752 | 1.2168 | 0.2791 | 0.0000 | 2.6700 |
| 14000-14100 | 1,354,649 -> 1,354,461 | 0.9985 | 0.4513 | 4.1157 | 0.3618 | 0.0000 | 5.9272 |
| 25000-25100 | 1,249,070 -> 1,249,070 | 0.9801 | 0.4510 | 3.8241 | 0.0000 | 0.0000 | 5.2553 |

Interpretation:

- This targets repeated small CPU/GPU tensor construction at the renderer/CUDA boundary; it should preserve camera semantics for static training views.
- Full train time improved versus v0.3.11 (`184.47s -> 181.56s`) with quality and Gaussian count in the same normal range.
- CUDA event windows also improved in render/total time (`14000-14100` total `6.0559ms -> 5.9272ms`, `25000-25100` total `5.3266ms -> 5.2553ms`), consistent with less per-render setup overhead and normal single-run variance.

## fastergsfusedrapid-v0.3.11 - 2026-05-11

Implementation/config changes:

- Changed FastGS metric-count accumulation buffers from `int32` tensors to `float32` tensors in the fused CUDA forward and no-grad image paths.
- Kept the same per-pixel `metric_map` condition and one-count-per-contributing-Gaussian atomic update, but now uses `atomicAdd(..., 1.0f)` so Python score computation can consume the counts directly.
- Removed the per-score-view `metric_counts.to(dtype=torch.float32)` conversion in `compute_fastgs_scores`.
- Added `configs/fastergsfusedrapid_v0_3_11_float_metric_counts/bicycle.yaml`, copied from v0.3.10 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_11_float_metric_counts \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_11_float_metric_counts_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.11 | bicycle | 184.47s | 1,246,971 | 25.6292 | 0.7580 | 0.2941 | 4.64GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,235 -> 137,908 | 0.7917 | 0.4515 | 1.2244 | 0.2976 | 0.0000 | 2.7651 |
| 14000-14100 | 1,357,569 -> 1,357,384 | 1.0771 | 0.4525 | 4.1479 | 0.3784 | 0.0000 | 6.0559 |
| 25000-25100 | 1,250,715 -> 1,250,715 | 1.0600 | 0.4504 | 3.8162 | 0.0000 | 0.0000 | 5.3266 |

Interpretation:

- The score/prune math still receives exact small integer counts because float32 represents all expected per-view pixel counts exactly; this change only removes an intermediate dtype conversion allocation.
- Full train time improved versus v0.3.10 (`185.49s -> 184.47s`) and quality stayed in the same normal range.
- Late profile window improved (`25000-25100` total `5.4325ms -> 5.3266ms`), while early densify/prune was noisier. Keep this as a score/prune path cleanup rather than a core backward-kernel speedup.
- This follows the CUDA guide's profile-first iteration principle and targets unnecessary memory traffic at the Python/CUDA boundary without changing training parameters.

## fastergsfusedrapid-v0.3.10 - 2026-05-10

Implementation/config changes:

- Templated `blend_backward_cu` on whether densification info is active.
- Templated `preprocess_backward_cu` on whether densification info is active.
- The post-densification path now compiles out absolute 2D mean-gradient accumulation/atomics and densification-info updates instead of guarding them with runtime uniform branches.
- Added `configs/fastergsfusedrapid_v0_3_10_template_densification_backward/bicycle.yaml`, copied from v0.3.9 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_10_template_densification_backward \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_10_template_densification_backward_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.10 | bicycle | 185.49s | 1,247,748 | 25.6526 | 0.7579 | 0.2945 | 4.64GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,980 -> 138,531 | 0.7883 | 0.4661 | 1.2150 | 0.2844 | 0.0000 | 2.7539 |
| 14000-14100 | 1,356,913 -> 1,356,865 | 1.1063 | 0.4523 | 4.1254 | 0.3765 | 0.0000 | 6.0605 |
| 25000-25100 | 1,251,330 -> 1,251,330 | 1.1154 | 0.4557 | 3.8614 | 0.0000 | 0.0000 | 5.4325 |

Interpretation:

- The pre-14900 densification path is semantically identical to v0.3.9; the post-densification path compiles out disabled densification work.
- Target backward windows improved slightly versus v0.3.9 (`14000-14100`: `4.1638ms -> 4.1254ms`, `25000-25100`: `3.8721ms -> 3.8614ms`), but full train time regressed in this single run (`184.25s -> 185.49s`).
- Treat this as a kernel-path cleanup with local profile benefit, not as a proven end-to-end speedup.

## fastergsfusedrapid-v0.3.9 - 2026-05-10

Implementation/config changes:

- Kept FastGS metric maps as `torch.bool` tensors in Python instead of converting them to `int32` before metric-count rendering.
- Changed the fused CUDA forward API and `blend_cu` metric-map pointer from `const int*` to `const bool*`.
- Added a runtime type check for non-empty metric maps in the C++ wrappers.
- Left `metric_counts` as `int32`; only the per-pixel binary map storage and load type changed.
- Added `configs/fastergsfusedrapid_v0_3_9_bool_metric_map/bicycle.yaml`, copied from v0.3.8 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_9_bool_metric_map \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_9_bool_metric_map_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.9 | bicycle | 184.25s | 1,252,318 | 25.6789 | 0.7585 | 0.2940 | 4.64GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,975 -> 138,046 | 0.7696 | 0.4548 | 1.1949 | 0.2837 | 0.0000 | 2.7030 |
| 14000-14100 | 1,361,559 -> 1,361,434 | 1.1134 | 0.4538 | 4.1638 | 0.3716 | 0.0000 | 6.1027 |
| 25000-25100 | 1,255,957 -> 1,255,957 | 1.0819 | 0.4551 | 3.8721 | 0.0000 | 0.0000 | 5.4091 |

Interpretation:

- This change only affects no-grad FastGS metric-count renders during score/pruning stages; training render/backward math and density thresholds are unchanged.
- Full train time improved slightly versus v0.3.8 (`184.61s -> 184.25s`) and quality stayed normal.
- The `14000-14100` densify/prune average improved from `0.3770ms` to `0.3716ms`, consistent with removing Python-side bool-to-int32 conversion and reducing metric-map load width in CUDA.

## fastergsfusedrapid-v0.3.8 - 2026-05-10

Implementation/config changes:

- Reused the fused backward `grad_conic` helper allocation for opacity gradients by allocating a 4-plane `{4, n_primitives}` helper.
- Passed the fourth plane as `grad_opacities`, while the first three planes remain the conic gradient layout consumed by `preprocess_backward_cu`.
- Removed the separate `{n_primitives, 1}` opacity-gradient tensor allocation and zero-fill from the autograd backward wrapper.
- Added `configs/fastergsfusedrapid_v0_3_8_fuse_grad_opacity_buffer/bicycle.yaml`, copied from v0.3.7 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_8_fuse_grad_opacity_buffer \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_8_fuse_grad_opacity_buffer_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.8 | bicycle | 184.61s | 1,246,385 | 25.6242 | 0.7587 | 0.2936 | 4.63GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 114,247 -> 139,226 | 0.7750 | 0.4572 | 1.2103 | 0.2881 | 0.0000 | 2.7307 |
| 14000-14100 | 1,354,702 -> 1,354,566 | 1.1049 | 0.4517 | 4.1344 | 0.3770 | 0.0000 | 6.0679 |
| 25000-25100 | 1,250,066 -> 1,250,066 | 1.1120 | 0.4554 | 3.8928 | 0.0000 | 0.0000 | 5.4601 |

Interpretation:

- The buffer reuse is semantically exact: `blend_backward_cu` writes the same opacity gradient values, and `preprocess_backward_cu` reads them through the same `float*` interface.
- Full train time improved modestly versus v0.3.7 (`185.54s -> 184.61s`) with quality and Gaussian count in the same normal range.
- The profiler window deltas are mixed (`14000-14100` backward improved, `25000-25100` regressed), so this should be treated as a small allocation/zero-fill cleanup rather than a major kernel-level speedup.

## fastergsfusedrapid-v0.3.7 - 2026-05-10

Implementation/config changes:

- Skipped `grad_mean2d_abs_helper` allocation in the fused CUDA backward when `densification_info` is disabled.
- Passed `nullptr` for `grad_mean2d_abs` after the densification window, so `blend_backward_cu` no longer computes or atomically accumulates absolute 2D mean-gradient values that will not be consumed.
- Kept the densification-window path unchanged; FastGS/RapidGS absolute-gradient split semantics still use the same accumulation before `DENSIFICATION_END_ITERATION`.
- Added `configs/fastergsfusedrapid_v0_3_7_skip_absgrad_after_density/bicycle.yaml`, copied from v0.3.6 with only experiment metadata changed.

Expected use:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_3_7_skip_absgrad_after_density \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_3_7_skip_absgrad_after_density_bicycle \
  --scenes bicycle
```

Experiment:

| version | scene | train time | n_gaussians | PSNR | SSIM | LPIPS | peak allocated VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fastergsfusedrapid-v0.3.7 | bicycle | 185.54s | 1,245,324 | 25.6219 | 0.7584 | 0.2947 | 4.63GiB |

Profiler windows:

| window | n_gaussians | render ms | loss ms | backward ms | densify/prune ms | optimizer ms | total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000-1100 | 113,979 -> 138,888 | 0.7475 | 0.4522 | 1.1934 | 0.2849 | 0.0000 | 2.6781 |
| 14000-14100 | 1,354,027 -> 1,354,001 | 1.1089 | 0.4536 | 4.1487 | 0.3756 | 0.0000 | 6.0868 |
| 25000-25100 | 1,248,913 -> 1,248,913 | 1.0806 | 0.4522 | 3.8425 | 0.0000 | 0.0000 | 5.3753 |

Interpretation:

- The change is semantically neutral for FastGS/RapidGS density control because the absolute mean-gradient buffer is still allocated and accumulated for all iterations that can consume `densification_info`.
- Late backward improved slightly (`3.8807ms -> 3.8425ms` at `25000-25100`), while train time stayed effectively flat versus v0.3.6 (`185.31s -> 185.54s`). This is worth keeping as a small hot-path cleanup, but the dominant cost is still normal backward math and gradient atomics.
- The optimization follows the CUDA guide's profile-and-iterate flow and removes unused global-memory/atomic work in the measured hot path rather than changing training parameters.

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
