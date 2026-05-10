# FasterGSBasisRapid Changelog

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
- Verified forced backend rebuild, Python compilation, backend import, metric-count forward path, image-only backward path, `(3, N)` densification info updates, and metric-count autograd compatibility.
