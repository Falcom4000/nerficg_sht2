# FasterGSFusedRapid Paper Method Notes

This document is a paper-writing oriented summary for the current
`FasterGSFusedRapid` line. It compares the implementation against the
original 3DGS paper, FastGS/RapidGS, Faster-GS/FasterGSFused, and the
SIGGRAPH Asia "Fast Converging 3D Gaussian Splatting for 1-Minute"
pipeline.

The intended use is to separate paper claims from implementation history:
what is algorithmically inherited, what is ported, what is only an
engineering optimization, and what should be described conservatively.

## 1. Source Material

Local papers:

- `/root/codes/papers/3dgs.pdf`
- `/root/codes/papers/fastgs.pdf`
- `/root/codes/papers/fastergs.pdf`
- `/root/codes/papers/Fast Converging 3D Gaussian Splatting for 1-Minute.pdf`

Local code and experiment records:

- `src/Methods/FasterGSFusedRapid`
- `src/Methods/FasterGSFusedRapid/CHANGELOG.md`
- `src/Methods/FasterGSFusedRapid/PORTING_WORKFLOW.md`
- `benchmarks_360v2_pipeline.md`
- reference RapidGS code at `/root/codes/RapidGS`
- baseline FasterGSFused code at `src/Methods/FasterGSFused`

## 2. Method Positioning

`FasterGSFusedRapid` should be described as a fused-backend 3DGS training
system that combines three ideas:

1. Standard 3D Gaussian representation and differentiable alpha-blended
   rasterization from 3DGS.
2. Multi-view consistent density control from FastGS/RapidGS, including
   score-guided densification and pruning.
3. Faster-GS style fused CUDA training backend, including fused rendering
   activations, separate SH buffers, fused backward/Adam update, and
   Morton/z-order reordering.

The v0.4 line also adds a fast-converging initialization path:

4. Offline AnySplat initialization, transformed into the Mip-NeRF 360 world
   frame and consumed as the initial Gaussian set before normal 3DGS-style
   optimization.

The current maintained path is RGB-only. The Metric3D/depth supervision path
was experimentally integrated and then removed in v0.4.14 because it did not
give a useful speed/quality gain in this codebase. For paper writing, depth
should be discussed as an explored but rejected component unless it is added
back and validated later.

## 3. 3DGS Baseline

The original 3DGS method initializes Gaussians from SfM points and optimizes
3D position, opacity, anisotropic covariance, and SH color coefficients. Its
adaptive density control creates new Gaussians from view-space positional
gradient statistics: small high-gradient Gaussians are cloned, large
high-gradient Gaussians are split, and low-opacity or overly large Gaussians
are pruned. Training usually runs to 30k iterations.

The renderer is a tile-based differentiable rasterizer:

- the image is split into 16x16 tiles;
- each Gaussian is projected and assigned to overlapping tiles;
- Gaussian/tile pairs are sorted by tile id and approximate depth;
- each tile blends its sorted list front-to-back;
- backward traverses the same tile list to recover gradients.

Relative to this baseline, `FasterGSFusedRapid` changes the training system
in four important ways:

- It can start from an AnySplat-generated Gaussian cloud rather than only the
  COLMAP/SfM sparse cloud.
- It replaces gradient-only density growth with FastGS/RapidGS multi-view
  score gating, so Gaussians are densified only when they are also associated
  with high-error pixels across sampled views.
- It keeps the standard RGB photometric loss and SH schedule in the
  maintained path, so the optimized representation remains standard 3DGS.
- It uses a fused CUDA backend for rendering, backward, and Adam updates,
  reducing Python/PyTorch kernel overhead and optimizer memory traffic.

A conservative paper claim is:

> We retain the standard 3DGS representation and RGB reconstruction objective,
> but replace its local gradient-only density control with a multi-view
> consistency signal and execute training through a fused CUDA backend.

## 4. FastGS and RapidGS Baseline

FastGS observes that vanilla 3DGS often creates redundant Gaussians because
gradient magnitude alone does not test whether a Gaussian improves
multi-view reconstruction. Its main algorithmic components are:

- **VCD, multi-view consistent densification.** Sample K training views,
  render them, build per-pixel L1 error maps, normalize them, threshold
  high-error pixels, and count how often each Gaussian's 2D footprint covers
  those pixels. A Gaussian is densified only if this multi-view importance
  score is high enough.
- **VCP, multi-view consistent pruning.** Compute a pruning score using the
  same high-error footprint counts weighted by photometric loss. Gaussians
  with high pruning score are treated as redundant or harmful and can be
  removed.
- **Compact box.** Reduce unnecessary Gaussian/tile pairs by shrinking the
  rasterization region using an additional contribution test.

RapidGS is the reference implementation used here for FastGS-like semantics.
The key RapidGS behavior ported into `FasterGSFusedRapid` is:

- score views are sampled from training views;
- score images are training-equivalent unclamped RGB renders;
- metric maps come from normalized per-pixel RGB error;
- metric-count rendering counts Gaussians that actually contribute to
  high-error pixels;
- clone candidates use signed 2D mean-gradient accumulation;
- split candidates use absolute 2D mean-gradient accumulation;
- candidate densification is gated by FastGS importance score;
- split parents are pruned immediately after children are added;
- densification-stage pruning uses the RapidGS weighted sampling semantics;
- final multi-view pruning ORs low-opacity pruning with score-threshold
  pruning.

Relative to RapidGS, `FasterGSFusedRapid` differs mainly in the backend and
initialization:

- RapidGS uses its own 3DGS-style Python/CUDA stack; `FasterGSFusedRapid`
  runs inside NeRFICG and uses the Faster-GS fused backend.
- RapidGS bicycle reference from `train_big.sh` reported about `233.03s`,
  `1,563,535` Gaussians, `PSNR 25.2623`, `SSIM 0.7555`, `LPIPS 0.2450`.
- `FasterGSFusedRapid` v0.4.10-v0.4.13 on bicycle trains in about
  `138-139s` with about `1.53M` Gaussians and PSNR around `25.31-25.33`.
- The Gaussian count is close to RapidGS, which suggests the density-control
  semantics are broadly aligned; the speed difference mainly comes from the
  fused backend and shorter AnySplat-assisted schedule, not from simply
  under-producing Gaussians.

Do not claim that the full FastGS compact-box rasterization contribution has
been ported unless the exact CUDA compact-box behavior is rechecked and
benchmarked. The current reliable claim is the VCD/VCP-style multi-view
score-guided density control.

## 5. Faster-GS / FasterGSFused Baseline

The Faster-GS paper is primarily a faithful 3DGS training acceleration work.
It surveys and consolidates backend optimizations, then adds further fused
training improvements. Relevant components are:

- fused scale/rotation/opacity activations inside the rasterizer;
- separate SH buffers, avoiding repeated PyTorch-side concatenation;
- optimized Adam updates, including a custom CUDA Adam implementation;
- optional fusion of backward and optimizer update;
- locality-preserving Morton/z-ordering during densification;
- per-Gaussian backward and other backend changes in the full version;
- careful discussion of invisible-Gaussian update skipping as optional,
  faster, but not fully equivalent to normal Adam semantics.

`FasterGSFusedRapid` builds on the fused backend rather than replacing it.
The main changes relative to `FasterGSFused` are algorithmic/training-control
changes:

- density control follows FastGS/RapidGS multi-view score semantics instead
  of the vanilla Faster-GS/3DGS density-control policy;
- AnySplat PLY initialization can replace the standard sparse-point
  initialization;
- training configs are versioned for the fast-converging path, especially the
  18k iteration schedule in the v0.4 line;
- profiler windows record render/loss/backward/densify-prune timing so code
  changes can be tied to measured bottlenecks;
- benchmark cleanup removes trained model folders after metrics, avoiding
  disk growth during repeated experiments.

The FasterGSFused bicycle reference supplied earlier was:

| method | train time | PSNR | SSIM | LPIPS | n_gaussians | VRAM allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FasterGSFused | `442.56s` | `25.2828` | `0.7671` | `0.2312` | `4.81M` | `4.94GiB` |

Compared with that reference, v0.4.13 uses far fewer Gaussians and much less
training time on bicycle, while PSNR stays in the same broad range. SSIM and
LPIPS are lower, so the paper should avoid saying it dominates quality.
Instead, phrase the result as a speed/compactness improvement with quality in
the normal range.

## 6. Fast-Converging 1-Minute Paper

The 1-minute paper distinguishes two settings:

- with noisy SLAM poses, it uses pose refinement, Neural-Gaussians,
  monocular-depth/feed-forward initialization, and optimized rasterization;
- with accurate COLMAP poses, it disables pose refinement, returns to the
  standard 3DGS ellipsoid representation to avoid MLP overhead, adds
  FastGS-inspired multi-view splitting/pruning, and uses Metric3D-v2 depth
  supervision.

The parts relevant to our maintained v0.4 line are:

- **AnySplat initialization.** The paper reports that high-quality
  feed-forward Gaussian initialization reduces the need for expensive
  convergence from sparse points. Our pipeline prepares an offline AnySplat
  PLY per Mip-NeRF 360 scene, then loads it into `FasterGSFusedRapid`.
- **Standard 3DGS representation.** Like the final-round solution, our
  maintained path trains ordinary Gaussians instead of keeping a
  Neural-Gaussian MLP in the loop.
- **FastGS score-guided density control.** We keep the multi-view score
  guided densification/pruning idea.

The part not retained is Metric3D/depth supervision:

- a depth path was integrated into trainer, scripts, and CUDA ABI;
- experiments did not show a useful speed/quality improvement for the
  current Mip-NeRF 360 path;
- v0.4.14 removed the depth supervision and inverse-depth CUDA backend path;
- the current maintained method should be described as AnySplat-only
  initialization plus RGB training, not as a depth-supervised method.

## 7. Current Version Interpretation

The v0.4.13 result is best treated as a historical bridge version:

- it made inverse-depth primitive/bucket buffers optional;
- it preserved RGB behavior and depth/no-depth buffer layout correctness;
- all-scene repeat-3 was effectively neutral: mean train time changed by
  `+0.0919s` relative to v0.4.12;
- it is not a speed contribution by itself.

The current maintained code direction is v0.4.14:

- Metric3D/depth supervision was removed from trainer/config/scripts;
- inverse-depth output/gradient support was removed from the maintained CUDA
  backend ABI;
- the RGB-only training path is now smaller and easier to reason about;
- the repeat-3 all-scene benchmark for
  `configs/fastergsfusedrapid_v0_4_14_anysplat_only_no_depth` is complete.

Use v0.4.13 when describing the history of optional depth buffers. Use
v0.4.14 when describing the maintained RGB-only implementation after depth
removal.

## 8. Experiment Snapshot

Recent bicycle references:

| method/config | train time | PSNR | SSIM | LPIPS | n_gaussians | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidGS `train_big.sh` | `233.03s` | `25.2623` | `0.7555` | `0.2450` | `1,563,535` | external reference run |
| FasterGSFused | `442.56s` | `25.2828` | `0.7671` | `0.2312` | `4,810,568` | fused baseline reference |
| FasterGSFusedRapid v0.3.14 | `182.59s` | `25.6324` | `0.7582` | `0.2941` | `1,246,630` | pre-AnySplat fused rapid baseline |
| FasterGSFusedRapid v0.4.10 | `138.55s` | `25.3187` | `0.7457` | `0.2969` | `1,532,846` | all-scene predecessor, bicycle mean |
| FasterGSFusedRapid v0.4.12 | `139.24s` | `25.3324` | `0.7456` | `0.2969` | `1,534,271` | last completed all-scene code baseline |
| FasterGSFusedRapid v0.4.13 | `139.28s` | `25.3072` | `0.7455` | `0.2973` | `1,529,843` | optional depth-buffer bridge |
| FasterGSFusedRapid v0.4.14 | `135.72s` | `25.3103` | `0.7453` | `0.2971` | `1,534,332` | maintained RGB-only no-depth path, bicycle mean |

Completed v0.4.14 all-scene repeat-3 result:

| scene | train time mean | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | `135.72s` | `25.3103` | `0.7453` | `0.2971` | `1,534,332` | `4.8970GiB` |
| bonsai | `84.87s` | `31.3792` | `0.9357` | `0.2575` | `422,402` | `5.7147GiB` |
| counter | `79.37s` | `28.3819` | `0.8939` | `0.2839` | `281,798` | `4.9930GiB` |
| garden | `91.17s` | `26.8063` | `0.8356` | `0.1894` | `885,973` | `2.9974GiB` |
| kitchen | `92.15s` | `30.8779` | `0.9192` | `0.1743` | `411,060` | `5.5863GiB` |
| room | `82.75s` | `31.2651` | `0.9128` | `0.3046` | `376,021` | `6.0448GiB` |
| stump | `94.34s` | `25.8549` | `0.7292` | `0.3016` | `1,216,121` | `2.5811GiB` |
| mean | `94.34s` | `28.5537` | `0.8531` | `0.2584` | `732,530` | `4.6878GiB` |

Compared with v0.4.12, v0.4.14 is `2.60s` faster on mean train time while
mean PSNR changes by `-0.020`, SSIM by `-0.0001`, and LPIPS by `+0.0001`.

## 9. What To Claim In A Paper

Safe claims:

- The method keeps the standard 3DGS Gaussian parameterization and RGB
  reconstruction objective.
- It combines FastGS/RapidGS multi-view score-guided density control with a
  Faster-GS style fused CUDA training backend.
- AnySplat initialization reduces the amount of optimization needed from
  scratch for the Mip-NeRF 360 pipeline.
- On bicycle, the v0.4 line reaches RapidGS-like Gaussian counts with much
  shorter measured training time in this implementation.
- Removing the Metric3D/depth path simplifies the maintained backend because
  depth supervision did not provide a useful improvement in the current
  experiments.

Claims that need caution:

- Do not say v0.4.13 is faster than v0.4.12; the completed all-scene mean
  regressed by `0.0919s`.
- Do not claim quality is unchanged in every metric. PSNR is normal on
  bicycle, but SSIM/LPIPS are lower than the supplied FasterGSFused reference.
- Do not claim full FastGS compact-box rasterization unless the CUDA backend
  is explicitly verified against the FastGS compact-box implementation.
- Do not claim depth supervision is part of the maintained method after
  v0.4.14.

Recommended paper phrasing:

> We integrate a feed-forward Gaussian initialization with multi-view
> consistency-guided density control and a fused CUDA 3DGS training backend.
> The resulting system preserves the standard 3DGS representation while
> reducing redundant Gaussian growth and reducing optimizer/rasterizer
> overhead.

When reporting v0.4.14, add:

> We found monocular depth supervision unnecessary for our current
> Mip-NeRF 360 setting and therefore use an RGB-only training backend.

## 10. Ablation Story

A clear ablation structure for the paper:

1. **Original 3DGS / FasterGSFused baseline.**
   Shows cost of vanilla density growth and large Gaussian count.
2. **Add FastGS/RapidGS density control.**
   Tests whether multi-view scoring reduces redundant Gaussians.
3. **Use fused backend.**
   Tests whether speed comes from backend fusion rather than only lower
   Gaussian count.
4. **Add AnySplat initialization.**
   Tests whether a better initial Gaussian cloud allows fewer iterations.
5. **Remove Metric3D/depth.**
   Shows that the maintained RGB-only path is simpler and not worse for the
   current benchmark.

For a stronger paper, run the same all-scene repeat-3 protocol for:

- `FasterGSFused`;
- `FasterGSFusedRapid` v0.3.14 or equivalent pre-AnySplat baseline;
- v0.4.12 or v0.4.13;
- v0.4.14 after depth removal.

## 11. Method Diagram Text

A concise pipeline description for a figure:

1. Input Mip-NeRF 360 scene with COLMAP cameras and RGB images.
2. Offline AnySplat predicts an initial Gaussian PLY.
3. The Gaussian PLY is transformed into the NeRFICG/Mip-NeRF 360 world frame.
4. Fused CUDA training renders RGB and performs backward plus Adam update.
5. At density-control intervals, K training views are sampled.
6. Per-pixel RGB errors produce high-error masks.
7. Metric-count rendering assigns high-error pixels back to contributing
   Gaussians.
8. FastGS/RapidGS scores gate clone/split and pruning.
9. Morton ordering periodically improves Gaussian memory locality.
10. Final cleanup prunes low-opacity/degenerate Gaussians and writes PLY.

## 12. Open Items

Potential future validation:

- exact CUDA comparison for whether compact-box semantics are present;
- all-scene direct comparison against a freshly run FasterGSFused baseline;
- an early-VCP config experiment, because the v0.4.14 `FASTGS_PRUNING_START_ITERATION=18000`
  does not materially trigger under an 18k training schedule;
- profiling table comparing RapidGS, v0.3.14, v0.4.12, and v0.4.14 at the
  same windows.
