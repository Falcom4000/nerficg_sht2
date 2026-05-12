# Feed-Forward Initialized and Fused Multi-View Consistent 3D Gaussian Training

> Draft main body only. Abstract, author block, acknowledgements, references,
> and final citation formatting are intentionally omitted.

## 1. Introduction

3D Gaussian Splatting (3DGS) has become a practical representation for novel
view synthesis because it combines explicit scene primitives, differentiable
rasterization, and real-time rendering after optimization. Unlike implicit
radiance fields, 3DGS directly optimizes a set of anisotropic Gaussian
primitives with opacity and spherical harmonics (SH) color coefficients. This
explicit representation makes rendering efficient, but training can still be
costly. A typical 3DGS pipeline starts from a sparse SfM point cloud, runs tens
of thousands of optimization steps, repeatedly densifies Gaussians from local
gradient statistics, and may grow to millions of primitives.

Recent work has attacked this cost from two complementary directions. FastGS
shows that many redundant Gaussians are created because vanilla 3DGS relies on
local image-space gradient magnitude for densification. It introduces
multi-view consistent densification and pruning, where sampled training views
identify high-error pixels and assign those errors back to the Gaussians that
contribute to them. Faster-GS instead focuses on the training backend. It
keeps the 3DGS representation faithful, but fuses expensive CUDA operations,
separates SH buffers, improves optimizer updates, and uses locality-preserving
ordering to reduce memory overhead.

A third direction is better initialization. The SIGGRAPH Asia fast
reconstruction pipeline observes that feed-forward Gaussian predictors such as
AnySplat can provide an initial Gaussian set that is much closer to a good
solution than a sparse COLMAP cloud. With an informative initialization, the
optimizer does not need to spend as many iterations discovering coarse scene
structure from scratch.

This work combines these directions in a single training pipeline,
`FasterGSFusedRapid`. The method retains the standard 3DGS representation and
RGB reconstruction objective, but replaces vanilla gradient-only density
control with FastGS/RapidGS-style multi-view score-guided density control,
uses a Faster-GS-style fused CUDA training backend, and optionally initializes
the scene from an offline AnySplat Gaussian PLY. The maintained implementation
is RGB-only: an intermediate Metric3D/depth-supervision branch was implemented
and evaluated, but removed after it failed to provide a useful speed/quality
improvement in the current Mip-NeRF 360 setting.

The result is not a new Gaussian representation. The contribution is a
training system: better initial primitives, fewer redundant density-control
decisions, and lower backend overhead. On the bicycle scene, the current
v0.4-line implementation reaches RapidGS-like Gaussian counts while reducing
training time from the RapidGS reference time of about 233 s to roughly
138-139 s in completed repeat measurements, with PSNR remaining in the same
range. Across seven Mip-NeRF 360 scenes, the last completed all-scene repeat-3
benchmark trains in 94.34 s on average with PSNR 28.55, SSIM 0.853, and LPIPS
0.258.

## 2. Related Work

### 2.1 3D Gaussian Splatting

3DGS represents a scene as a set of anisotropic 3D Gaussians. Each Gaussian
stores a 3D mean, opacity, covariance represented through scale and rotation,
and SH coefficients for view-dependent color. Rendering projects Gaussians
into screen space, sorts Gaussian/tile instances approximately by depth, and
alpha-blends them front-to-back inside 16x16 image tiles.

The original 3DGS training loop interleaves gradient descent with adaptive
density control. After a warm-up period, Gaussians with large accumulated
view-space position gradients are candidates for densification. Small
high-gradient Gaussians are cloned to fill under-reconstructed regions, while
large high-gradient Gaussians are split into smaller children. Gaussians with
low opacity or invalid scale are pruned. This strategy is simple and robust,
but it does not explicitly test whether a Gaussian is consistently useful
across multiple views. As a result, density growth can produce redundant
primitives and increase backward and optimizer cost.

### 2.2 Multi-View Consistent Density Control

FastGS addresses redundancy by making density control depend on multi-view
reconstruction quality. It samples a small set of training views, renders RGB
images, computes per-pixel reconstruction error, and thresholds high-error
regions. For each Gaussian, it counts how often the Gaussian contributes to
high-error pixels across the sampled views. This score is used for
multi-view consistent densification (VCD). A related score weighted by
photometric loss is used for multi-view consistent pruning (VCP).

The key distinction from vanilla 3DGS is that a Gaussian is not densified only
because it has a large local gradient. It must also be associated with
multi-view high-error regions. This creates a stricter condition for growth
and reduces redundant primitives. RapidGS provides the reference
implementation used in this work for the detailed semantics of FastGS-style
score computation, densification gating, and pruning.

### 2.3 Fused 3DGS Training Backends

Faster-GS studies how to accelerate 3DGS training while preserving the
standard Gaussian representation. It moves repeated PyTorch-side operations
into CUDA kernels, passes separate SH buffers to avoid unnecessary
concatenation, fuses activation functions into rasterization, improves Adam
updates, and uses Morton/z-order sorting to preserve spatial locality in
parameter buffers after densification.

These backend changes are orthogonal to density-control policy. A fused
backend reduces the cost of rendering, backward propagation, and parameter
updates; a better density policy reduces the number of Gaussians these kernels
must process. `FasterGSFusedRapid` uses both.

### 2.4 Feed-Forward Gaussian Initialization

The fast reconstruction challenge pipeline shows that a feed-forward Gaussian
model can provide a dense and informative initialization. In the final-round
setting with accurate COLMAP poses, that work returns to the standard 3DGS
ellipsoid representation, uses AnySplat-style initialization, applies
FastGS-inspired multi-view splitting/pruning, and adds Metric3D depth
supervision.

Our current pipeline adopts the feed-forward initialization and the standard
3DGS representation, but does not keep Metric3D depth supervision. In our
implementation, depth supervision required extending the trainer, autograd
binding, forward buffers, and backward kernels to produce and differentiate
inverse depth. The measured benefit was not enough to justify the added
complexity, so the maintained v0.4.14 path removes the depth backend and
keeps RGB-only training.

## 3. Method

### 3.1 Overview

Given a calibrated multi-view scene, the pipeline consists of four stages:

1. Prepare an optional offline AnySplat Gaussian PLY for the scene.
2. Transform the predicted Gaussian attributes into the Mip-NeRF 360 training
   coordinate frame.
3. Train standard 3DGS Gaussians using a fused CUDA renderer/backward/Adam
   backend.
4. At density-control intervals, compute FastGS/RapidGS multi-view scores and
   use them to guide clone, split, and prune decisions.

The optimized parameters are the standard 3DGS parameters:

```text
G_i = {mu_i, s_i, q_i, alpha_i, c_i}
```

where `mu_i` is the 3D mean, `s_i` is the anisotropic scale, `q_i` is the
rotation quaternion, `alpha_i` is opacity, and `c_i` contains SH color
coefficients. The training objective remains the RGB photometric objective:

```text
L = (1 - lambda) L1(I_render, I_gt) + lambda L_DSSIM(I_render, I_gt).
```

No depth loss is used in the maintained path.

### 3.2 AnySplat Initialization

Standard 3DGS initializes one Gaussian per sparse SfM point. This is reliable
but forces the optimizer to create much of the final density through repeated
clone/split operations. We instead allow initialization from an AnySplat PLY.
The offline prior generator creates a scene-local `anysplat_init/point_cloud.ply`.
During trainer setup, if this PLY exists and the config enables AnySplat
initialization, the model loads the Gaussian means, opacities, scales,
rotations, and SH coefficients directly from the PLY.

Mip-NeRF 360 preprocessing may apply a world transform to center, scale, or
orient the scene. Therefore, the imported Gaussian means, scales, and
rotations must be transformed consistently with the dataset world frame. This
is important: early experiments without correct world-frame handling produced
poor quality despite using a strong initialization. After applying the correct
transform, the AnySplat initialization became useful for reducing the training
iteration budget.

The current speed-focused v0.4 configuration uses 18k iterations. Imported
SH coefficients are not all activated at iteration 0; the training path keeps
the normal gradual SH-degree schedule, which is closer to standard 3DGS
training semantics and empirically produced more stable quality.

### 3.3 Multi-View Score Computation

At densification or pruning intervals, the trainer samples `K` training views.
For each sampled view, it renders an unclamped training-equivalent RGB image
without optimizer updates. It then compares this image with the ground-truth
RGB image and computes a per-pixel error map:

```text
e_j(u, v) = mean_c |I_j^render(c, u, v) - I_j^gt(c, u, v)|.
```

The error map is min-max normalized and thresholded:

```text
M_j(u, v) = 1[normalize(e_j(u, v)) > tau].
```

The renderer is called again in metric-count mode. In this mode, the CUDA
backend counts which Gaussians actually contribute to high-error pixels in
`M_j`. The count vector is accumulated across sampled views. For
densification, the importance score is the average high-error contribution
count:

```text
s_i^+ = floor((1 / K) sum_j count_ij).
```

For pruning, the score is weighted by the photometric loss of the sampled
view and normalized:

```text
s_i^- = normalize(sum_j count_ij * E_j^photo).
```

This implementation follows the RapidGS semantics that matter for
consistency:

- score views are sampled only from training views;
- score renders are unclamped and use training-equivalent Gaussian scales;
- metric-count renders count contributing Gaussians, not merely projected
  Gaussians;
- densification candidates are gated by the multi-view importance threshold;
- final pruning combines low-opacity pruning with multi-view score pruning.

### 3.4 Density Control

The model maintains a three-channel densification buffer:

```text
D_i = {denom_i, grad_i, abs_grad_i}.
```

The denominator counts visibility/update opportunities. The signed gradient
channel is used for clone candidates, and the absolute gradient channel is
used for split candidates. This matches the RapidGS behavior observed during
porting.

Clone and split candidates are first selected from accumulated gradient
statistics:

```text
clone_i = grad_i >= tau_grad * denom_i
split_i = abs_grad_i >= tau_abs * denom_i.
```

They are then gated by the FastGS importance score:

```text
clone_i = clone_i and (s_i^+ > tau_imp)
split_i = split_i and (s_i^+ > tau_imp).
```

Small selected Gaussians are duplicated. Large selected Gaussians are split
into two children sampled from the parent Gaussian distribution, with the
child scale reduced by the standard 3DGS factor. Split parents are pruned
immediately after child insertion. This immediate parent pruning is important
for keeping Gaussian count aligned with RapidGS.

Densification-stage pruning starts from low-opacity and large-scale masks.
When a pruning score is available, the implementation follows RapidGS-style
weighted sampling rather than deleting all possible candidates. This prevents
over-aggressive pruning inside densification windows. A separate multi-view
pruning callback later removes Gaussians whose score exceeds the final pruning
threshold or whose opacity is too low.

### 3.5 Fused Backend

The training render path calls the fused rasterizer with Gaussian parameter
buffers and Adam moment buffers. The backward pass computes image gradients
and applies the optimizer update in the CUDA backend through an autograd dummy
parameter. This keeps the PyTorch graph small while allowing the trainer to
use a normal loss call and `backward()`.

The fused backend inherits several Faster-GS-style ideas:

- scale, rotation, and opacity activations are performed inside CUDA;
- SH degree-0 and remaining SH coefficients are stored separately;
- Adam moment buffers are stored beside the Gaussian parameters;
- Morton ordering periodically reorders Gaussian parameters and moment
  buffers to improve memory locality;
- profiler windows record render, loss, backward, and density-control timing.

An intermediate implementation added inverse-depth output and inverse-depth
gradients to support Metric3D supervision. The maintained backend removes
those buffers and ABI arguments. This reduces implementation surface and
keeps the current method RGB-only.

## 4. Implementation

The implementation is in `src/Methods/FasterGSFusedRapid`. The key trainer
components are:

- `setup_gaussians`, which selects AnySplat PLY initialization or falls back
  to the standard point-cloud initialization;
- `compute_fastgs_scores`, which samples views, renders score images, builds
  high-error masks, and obtains metric counts;
- `densify`, which calls multi-view score-guided adaptive density control;
- `prune_multiview_inconsistent`, which applies FastGS-style pruning;
- `training_iteration`, which performs RGB rendering, loss computation, and
  fused backward/Adam update;
- `morton_ordering`, which periodically reorders Gaussian buffers.

The model implementation keeps Gaussian tensors and fused Adam moments in
parallel buffers. All prune, split, duplicate, and sort operations update the
parameter tensors and moment tensors together. This is necessary because the
optimizer state is not owned by a standard PyTorch optimizer object.

The benchmark script records per-run metrics, profiler windows, final Gaussian
counts, peak VRAM, and whether trained model artifacts were deleted after
evaluation. Deleting model folders after metric extraction is necessary for
repeat-3 all-scene experiments because trained PLY checkpoints otherwise
consume large amounts of disk space.

## 5. Experiments

### 5.1 Setup

Experiments use the seven Mip-NeRF 360 scenes currently configured in the
repository: `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, and
`stump`. Metrics are PSNR, SSIM, LPIPS, training time, final Gaussian count,
and peak allocated VRAM. Unless otherwise noted, results are from repeat-3
benchmark summaries recorded by `scripts/benchmark_360v2.py`.

The main baselines available in the current records are:

- RapidGS bicycle reference from `train_big.sh`;
- FasterGSFused bicycle reference;
- `FasterGSFusedRapid` v0.3.14 before AnySplat initialization;
- v0.4.10-v0.4.13 fast-converging AnySplat-based variants;
- v0.4.14 RGB-only no-depth path.

### 5.2 Bicycle Comparison

The bicycle scene is the most complete cross-method comparison available in
the current notes.

| method/config | train time | PSNR | SSIM | LPIPS | n_gaussians | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidGS `train_big.sh` | 233.03s | 25.2623 | 0.7555 | 0.2450 | 1,563,535 | external reference run |
| FasterGSFused | 442.56s | 25.2828 | 0.7671 | 0.2312 | 4,810,568 | fused baseline reference |
| FusedRapid v0.3.14 | 182.59s | 25.6324 | 0.7582 | 0.2941 | 1,246,630 | pre-AnySplat baseline |
| FusedRapid v0.4.10 | 138.55s | 25.3187 | 0.7457 | 0.2969 | 1,532,846 | all-scene predecessor |
| FusedRapid v0.4.12 | 139.24s | 25.3324 | 0.7456 | 0.2969 | 1,534,271 | last completed all-scene code baseline |
| FusedRapid v0.4.13 | 139.28s | 25.3072 | 0.7455 | 0.2973 | 1,529,843 | optional depth-buffer bridge |
| FusedRapid v0.4.14 | 135.72s | 25.3103 | 0.7453 | 0.2971 | 1,534,332 | maintained RGB-only path, bicycle mean |

The v0.4-line runs reach a final Gaussian count close to the RapidGS
reference while reducing training time substantially. This suggests that the
speedup is not simply caused by producing an unusually small Gaussian set.
The quality tradeoff is mixed: PSNR remains normal for bicycle, but SSIM and
LPIPS are lower than the FasterGSFused reference. Therefore, the result should
be described as faster and more compact with quality in a normal range, not
as uniformly higher quality.

### 5.3 All-Scene Results

The current maintained all-scene repeat-3 result is v0.4.14:

| scene | train time mean | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 135.72s | 25.3103 | 0.7453 | 0.2971 | 1,534,332 | 4.8970GiB |
| bonsai | 84.87s | 31.3792 | 0.9357 | 0.2575 | 422,402 | 5.7147GiB |
| counter | 79.37s | 28.3819 | 0.8939 | 0.2839 | 281,798 | 4.9930GiB |
| garden | 91.17s | 26.8063 | 0.8356 | 0.1894 | 885,973 | 2.9974GiB |
| kitchen | 92.15s | 30.8779 | 0.9192 | 0.1743 | 411,060 | 5.5863GiB |
| room | 82.75s | 31.2651 | 0.9128 | 0.3046 | 376,021 | 6.0448GiB |
| stump | 94.34s | 25.8549 | 0.7292 | 0.3016 | 1,216,121 | 2.5811GiB |
| mean | 94.34s | 28.5537 | 0.8531 | 0.2584 | 732,530 | 4.6878GiB |

Compared with v0.4.12, v0.4.14 reduces mean train time from 96.94 s to
94.34 s while keeping mean SSIM and LPIPS effectively unchanged. Mean PSNR is
lower by 0.020, which is within the range expected for repeat-level variation
in this experiment series.

### 5.4 Ablation Interpretation

The v0.4 history provides several useful ablation points:

- v0.3.14 is the pre-AnySplat fused rapid baseline.
- v0.4.5 validates that AnySplat can restore quality after the correct world
  transform is applied, but 30k iterations are slower.
- v0.4.6 and v0.4.8 test lower iteration counts with AnySplat initialization.
- v0.4.9 keeps the 18k schedule but returns to gradual SH activation.
- v0.4.10 extends the AnySplat-only 18k schedule to all seven scenes.
- v0.4.11-v0.4.13 remove unused depth-disabled work in stages, but are
  effectively neutral in end-to-end all-scene timing.
- v0.4.14 removes the depth-supervision path completely from the maintained
  trainer, scripts, CUDA ABI, and profiler.

This history suggests that the large speed improvement in the v0.4 line comes
primarily from the combination of AnySplat initialization and the reduced
18k training schedule, while the fused backend and FastGS/RapidGS density
control keep that shorter schedule viable. The depth-buffer cleanups are
engineering simplifications rather than major speed contributions.

## 6. Discussion

### 6.1 Why Multi-View Scores Matter

Vanilla 3DGS densification answers a local question: does this Gaussian have
a large image-space positional gradient? FastGS-style scoring answers a more
global question: is this Gaussian repeatedly associated with high-error pixels
from multiple views? The second question is better aligned with reducing
redundant growth. In our implementation, this distinction is important
because backward and optimizer costs scale with Gaussian count. Avoiding
unnecessary Gaussians directly reduces the work handled by the fused backend.

### 6.2 Why Initialization Matters

AnySplat initialization changes the optimization problem. Instead of starting
from sparse points and relying on many rounds of densification to discover
scene coverage, training starts from a denser feed-forward prediction. This
does not eliminate optimization: imported Gaussians still need to be adjusted,
pruned, split, and recolored. However, it makes a shorter training budget
plausible. The bicycle results show that an 18k schedule can remain in a
normal quality range after the initialization is transformed correctly and SH
activation follows the standard schedule.

### 6.3 Why Depth Was Removed

Metric3D depth supervision is attractive because it can provide geometric
guidance early in training. In practice, adding it to this fused training
stack is expensive: the renderer must output blended inverse depth, the
autograd binding must carry depth gradients, and CUDA forward/backward buffers
must store extra per-fragment or per-primitive state. In the tested Mip-NeRF
360 path, this did not produce a useful improvement. Removing depth restored a
smaller RGB-only backend and reduced the number of moving parts. This is a
negative result, but it is useful for the final method definition.

### 6.4 Limitations

The current evidence has several limitations:

- The strongest all-scene result currently available is v0.4.14, but the
  FasterGSFused comparison is still represented by a bicycle-only reference.
- The FasterGSFused comparison is currently represented by a bicycle
  reference, not a fresh seven-scene repeat-3 run under the same script.
- The FastGS compact-box rasterization contribution has not been claimed as
  fully ported in the maintained implementation.
- Quality is not uniformly better than all baselines; the main demonstrated
  advantage is speed and compactness with quality remaining in a normal range.

## 7. Conclusion

This work presents a practical fast 3DGS training pipeline that combines
feed-forward Gaussian initialization, multi-view consistent density control,
and a fused CUDA training backend. The method keeps the standard 3DGS
representation and RGB loss, but improves the training path around it:
AnySplat provides a stronger starting point, FastGS/RapidGS scores guide
where Gaussians should grow or be removed, and the fused backend reduces the
cost of rendering, backward propagation, and parameter updates.

The current results show that this combination can substantially reduce
training time on the bicycle scene while keeping Gaussian count close to
RapidGS and maintaining quality in a normal range. The maintained v0.4.14
RGB-only implementation gives an average training time of 94.34 s over seven
Mip-NeRF 360 scenes, with PSNR 28.55, SSIM 0.853, and LPIPS 0.258.
