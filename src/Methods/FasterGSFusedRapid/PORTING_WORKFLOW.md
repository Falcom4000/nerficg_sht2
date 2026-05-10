# FasterGSFusedRapid Porting Workflow

This document records the working loop used while porting RapidGS/FastGS behavior into `FasterGSFusedRapid`.
The goal is to keep code changes, configs, benchmark results, changelog entries, and commits aligned.

## Scope

Primary target:

- `src/Methods/FasterGSFusedRapid`

Reference implementations:

- `src/Methods/FasterGSBasisRapid`
- `/root/codes/RapidGS`
- `src/Methods/FasterGSFused`

The migration priority is semantic consistency first, then performance. Pure parameter tuning should be separated into versioned configs and should not be mixed into kernel or trainer changes.

## Code Reading Checklist

Read these paths before changing behavior:

- `src/Methods/FasterGSFusedRapid/Trainer.py`
- `src/Methods/FasterGSFusedRapid/Model.py`
- `src/Methods/FasterGSFusedRapid/Renderer.py`
- `src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/torch_bindings/rasterization.py`
- `src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/rasterization/src/forward.cu`
- `src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/rasterization/src/backward.cu`
- `src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/rasterization/src/rasterization_api.cu`
- `src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/rasterization/include/kernels_forward.cuh`
- `src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/rasterization/include/kernels_backward.cuh`

Compare against these RapidGS/FastGS paths:

- `/root/codes/RapidGS/train.py`
- `/root/codes/RapidGS/utils/fast_utils.py`
- `/root/codes/RapidGS/scene/gaussian_model.py`
- `/root/codes/RapidGS/gaussian_renderer/__init__.py`
- `/root/codes/RapidGS/submodules/diff-gaussian-rasterization_fastgs/rasterize_points.cu`
- `/root/codes/RapidGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu`
- `src/Methods/FasterGSBasisRapid/Trainer.py`
- `src/Methods/FasterGSBasisRapid/Model.py`
- `src/Methods/FasterGSBasisRapid/Renderer.py`

Key semantics to verify:

- Densification info channels: visibility denominator, signed 2D mean-gradient magnitude, absolute 2D mean-gradient magnitude.
- Clone/split candidate masks: RapidGS clone uses signed gradient, split uses abs gradient.
- Importance gating: candidate densification should be gated by FastGS multi-view metric counts.
- Split-parent pruning: split parents must be pruned immediately after adding their children.
- Densification-stage pruning budget: RapidGS uses weighted `torch.multinomial` sampling.
- Final pruning: low-opacity pruning is OR'ed with multi-view score thresholding.
- Score render semantics: score images should be unclamped training-equivalent renders, not inference-only renders with scale modifiers.
- Metric-count render semantics: count the Gaussian id that actually contributes to high-error pixels.
- Fused optimizer semantics: training render/backward must still use the fused CUDA backward/Adam path.

## Build Check

After CUDA or binding edits, reinstall the backend:

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate nerficg && \
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-} \
pip install --force-reinstall --no-build-isolation -v \
  src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend
```

Run a minimal import check:

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate nerficg && \
LD_LIBRARY_PATH=/usr/local/miniconda3/envs/nerficg/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-} \
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from Methods.FasterGSFusedRapid.Trainer import FasterGSFusedRapidTrainer
from Methods.FasterGSFusedRapid.FasterGSFusedRapidCudaBackend import diff_rasterize, RasterizerSettings
import FasterGSFusedRapidCudaBackend._C
print(FasterGSFusedRapidTrainer.__name__, diff_rasterize.__name__, RasterizerSettings.__name__, 'backend ok')
PY
```

## Experiment Loop

Use one versioned config directory per change:

- Code changes: bump `EXPERIMENT.VERSION`.
- Config-only experiments: copy the previous config directory and change only the intended fields.
- Do not overwrite older benchmark configs.
- Keep `EXPERIMENT.BASELINE`, `EXPERIMENT.CHANGELOG`, and `EXPERIMENT.NOTES` updated.

Run bicycle first:

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh && conda activate nerficg && \
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/miniconda3/envs/nerficg/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-} \
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/<versioned_config_dir> \
  --repeats 1 \
  --suite-name <versioned_suite_name> \
  --scenes bicycle
```

## Metrics To Inspect

Read these files after each benchmark:

- `output/benchmarks/<suite_name>/results_runs.csv`
- `output/benchmarks/<suite_name>/runs/FasterGSFusedRapid/bicycle/run_01/profile_windows.csv`
- `output/benchmarks/<suite_name>/runs/FasterGSFusedRapid/bicycle/run_01/n_gaussians.txt`
- `output/benchmarks/<suite_name>/logs/FasterGSFusedRapid/bicycle/run_01.log`

Track these fields:

- `train_time_sec`
- `wall_time_sec`
- `PSNR`
- `SSIM`
- `LPIPS`
- `n_gaussians`
- `vram_allocated_gb`
- `vram_reserved_gb`
- `model_artifacts_deleted`
- profiler `render_ms`
- profiler `loss_ms`
- profiler `backward_ms`
- profiler `densify_prune_ms`
- profiler `total_ms`
- profiler Gaussian ranges per window

Compare against known references:

- RapidGS bicycle reference: `231.77s`, `1,554,313` Gaussians, `PSNR 25.2623`, `SSIM 0.7555`, `LPIPS 0.2450`.
- FasterGSFusedRapid v0.2: `227.30s`, `1,952,145` Gaussians, `PSNR 25.6683`, `SSIM 0.7643`, `LPIPS 0.2665`.
- FasterGSFusedRapid v0.3.4: `182.43s`, `1,260,164` Gaussians, `PSNR 25.6559`, `SSIM 0.7579`, `LPIPS 0.2946`.

Interpretation rules:

- If `n_gaussians` is far above RapidGS, focus on density/pruning semantics before kernel micro-optimization.
- If `train_time_sec` is good but `LPIPS` regresses, inspect pruning score computation and final pruning before reducing Gaussian count further.
- If profiler `backward_ms` dominates, Gaussian count and fused backward memory traffic are the main suspects.
- If profiler `densify_prune_ms` grows, inspect FastGS score view sampling, metric-count render overhead, and redundant autograd context creation.
- If `render_ms` changes without Gaussian count changes, inspect forward/bucket kernel changes.

## Changelog Rules

Every versioned change needs a `CHANGELOG.md` entry with:

- version and date,
- implementation/config changes,
- exact benchmark command,
- result table,
- profiler table,
- short interpretation.

Keep the changelog in descending version order.

Do not leave `pending` entries after a benchmark has completed. If a run was aborted, say so explicitly.

## Commit Rules

Before committing:

```bash
git status --short
git diff --stat
```

Clean build artifacts:

```bash
rm -rf \
  src/Methods/FasterGSFusedRapid/__pycache__ \
  src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/build \
  src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/__pycache__ \
  src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend.egg-info \
  src/Methods/FasterGSFusedRapid/FasterGSFusedRapidCudaBackend/FasterGSFusedRapidCudaBackend/torch_bindings/__pycache__
```

Stage only relevant files:

```bash
git add \
  src/Methods/FasterGSFusedRapid \
  configs/<versioned_config_dir>
```

Do not stage unrelated untracked directories such as:

- `src/Methods/FasterGSBasis/`
- `src/Methods/FasterGSFused/`

Commit after code compiles and the changelog records the experiment:

```bash
git commit -m "<concise versioned message>"
```

Use separate commits for separate steps:

- backend/API migration,
- trainer/model semantic migration,
- config-only experiments,
- benchmark result documentation,
- workflow/documentation updates.
