# FasterGSBasisRapid Changelog

## 2026-05-10

- Added FastGS/RapidGS-style multi-view scoring to `FasterGSBasisRapid`.
- Extended the CUDA rasterizer with an optional per-pixel metric map and per-Gaussian metric counts.
- Added metric-count rendering in `Renderer.py` without changing the default `diff_rasterize` image-only API.
- Gated densification with FastGS view-consistent importance scores.
- Added post-densification multi-view pruning using normalized photometric score counts.
- Added configurable FastGS parameters for score views, loss threshold, importance threshold, pruning interval, pruning opacity, and pruning score threshold.
- Verified backend rebuild, Python compilation, backend import, metric-count forward path, image-only backward path, and metric-count autograd compatibility.
