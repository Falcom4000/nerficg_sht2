# 360_v2 Benchmark Pipeline

This pipeline is for benchmarking one or more `src/Methods/<METHOD_NAME>` implementations on the Mip-NeRF 360 `360_v2` scenes.

All benchmark runs are strictly serial. The runner starts one `scripts/train.py` process, waits for it to finish, records its outputs, then starts the next method/scene/repeat. It does not run multiple scenes, repeats, or methods concurrently.

## 1. Prepare Method Code

Create each method under:

```bash
src/Methods/<METHOD_NAME>
```

Each method package must export these names in `src/Methods/<METHOD_NAME>/__init__.py`:

```python
MODEL = ...
RENDERER = ...
TRAINING_INSTANCE = ...
```

If the method has a CUDA extension, keep its install command exposed through the project's usual `Framework.ExtensionError` mechanism so `scripts/install.py` can discover it.

## 2. Recompile Extensions

Only needed after CUDA/C++ code changes or when setting up a method for the first time:

```bash
python ./scripts/install.py -m <METHOD_NAME>
```

When using the `environments/py311_cu128.yaml` Conda environment, PyTorch is built for CUDA 12.8. Make sure the extension build also uses a CUDA 12.8 `nvcc`; otherwise PyTorch's extension builder can fail with a version mismatch such as detected CUDA 13.0 vs PyTorch CUDA 12.8.

On machines where the default `nvcc` points to CUDA 13.0, run installs with:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
python ./scripts/install.py -m <METHOD_NAME>
```

Install `ninja` in the Conda environment before compiling CUDA extensions. Without it, PyTorch falls back to the slower distutils backend:

```bash
python -m pip install ninja
```

The benchmark runner can do this before benchmarking:

```bash
python ./scripts/benchmark_360v2.py \
  -m <METHOD_NAME> \
  --install
```

## 3. Create Or Choose Configs

Optional default config generation:

```bash
python ./scripts/create_config.py -m <METHOD_NAME> -d MipNeRF360 -o <CONFIG_NAME>
```

For benchmarking, prefer a config directory under `configs/` with one YAML per scene:

```text
configs/<CONFIG_DIR>/
  bicycle.yaml
  bonsai.yaml
  counter.yaml
  garden.yaml
  kitchen.yaml
  room.yaml
  stump.yaml
```

Pass that directory to the runner with `--config-dir configs/<CONFIG_DIR>`.

When a method-specific `--config-dir` is provided, scene YAML files are treated as the source of truth for dataset scale. The runner still fills `DATASET.PATH`, but it no longer overwrites an explicit `DATASET.IMAGE_SCALE_FACTOR` from the scene config with the built-in Mip-NeRF 360 scale defaults. The built-in scale defaults are only applied when running from generated/default configs or templates.

Useful benchmark defaults are forced by the runner:

- `TRAINING.WANDB.ACTIVATE=false`
- `TRAINING.TIMING.ACTIVATE=true`
- `TRAINING.WRITE_VRAM_STATS=true`
- `TRAINING.BACKUP.FINAL_CHECKPOINT=true`
- `TRAINING.BACKUP.RENDER_TESTSET=true`
- `TRAINING.BACKUP.INTERMEDIATE_RENDERINGS=false`

## 4. Prepare FasterGSFusedRapid Fast-Converging Priors

`FasterGSFusedRapid` can use the Fast-Converging 3DGS style offline priors for Mip-NeRF 360 scenes:

- Metric3D inverse-depth priors are written to `<SCENE>/mono_depths/<image_stem>_depth.npy`.
- AnySplat Gaussian initialization is written to `<SCENE>/anysplat_init/point_cloud.ply`.
- The local AnySplat encoder also needs VGGT-1B weights. The default path is `/root/codes/siggraph_asia/VGGT-1B/model.safetensors`.
- `configs/fastergsfusedrapid_v0_4_12_forward_depth_template` is the current all-scene code-level baseline and speed-focused recommendation: AnySplat initialization only, Mip-NeRF 360 PCA/rescale applied to Gaussian means/scales/rotations, Metric3D depth supervision disabled, `18000` training iterations, normal gradual SH-degree activation, and depth-disabled CUDA forward/backward paths compiled without inverse-depth work.
- `configs/fastergsfusedrapid_v0_4_10_all_scenes_baseline` is the predecessor all-scene baseline before the depth-disabled CUDA template cleanups.
- `configs/fastergsfusedrapid_v0_4_9_anysplat_only_18k_sh_schedule` is the single-scene predecessor of v0.4.10 for bicycle.
- `configs/fastergsfusedrapid_v0_4_8_anysplat_only_18k` is the same 18k schedule with all imported SH coefficients active from iteration 0.
- `configs/fastergsfusedrapid_v0_4_6_anysplat_only_20k` is the more conservative speed/quality point.
- `configs/fastergsfusedrapid_v0_4_5_both_world_transform` keeps both Metric3D and AnySplat enabled for diagnostics, but it is not the current recommendation because the bicycle split run regressed PSNR/SSIM and increased Gaussian count.
- Training consumes priors through `TRAINING.DEPTH_SUPERVISION` and `TRAINING.ANYSPLAT_INITIALIZATION`.
- Use the v0.4.5 split configs to isolate each prior source: `configs/fastergsfusedrapid_v0_4_5_anysplat_only`, `configs/fastergsfusedrapid_v0_4_5_depth_only`, and `configs/fastergsfusedrapid_v0_4_5_both_world_transform`.

Generate priors for one scene:

```bash
python scripts/prepare_fast_converging_priors.py \
  dataset/mipnerf360/bicycle \
  --tasks metric3d anysplat \
  --metric3d-weights /path/to/metric_depth_vit_giant2_800k.pth \
  --anysplat-weights /path/to/anysplat/model.safetensors \
  --vggt-weights /path/to/VGGT-1B/model.safetensors \
  --metric3d-image-scale 0.3234937323
```

Then run the matching config:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_4_12_forward_depth_template \
  --repeats 1 \
  --suite-name fastergsfusedrapid_v0_4_12_forward_depth_template_bicycle \
  --scenes bicycle
```

The one-command wrapper runs both stages serially:

```bash
python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --prior-mode anysplat \
  --metric3d-weights /path/to/metric_depth_vit_giant2_800k.pth \
  --anysplat-weights /path/to/anysplat/model.safetensors \
  --vggt-weights /path/to/VGGT-1B/model.safetensors
```

Use `--dry-run-priors --prepare-only` to validate paths, split generation, scaled Metric3D workspace generation, and commands without running model inference or training. Use `--prior-mode metric3d`, `--prior-mode anysplat`, or `--prior-mode none` when running the split configs.

Current split benchmark commands:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFusedRapid \
  --config-dir configs/fastergsfusedrapid_v0_4_12_forward_depth_template \
  --repeats 3 \
  --suite-name fastergsfusedrapid_v0_4_12_forward_depth_template_r3

python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --prior-mode none \
  --config-dir configs/fastergsfusedrapid_v0_4_9_anysplat_only_18k_sh_schedule \
  --suite-name fastergsfusedrapid_v0_4_9_anysplat_only_18k_sh_schedule_bicycle \
  --repeats 1

python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --prior-mode none \
  --config-dir configs/fastergsfusedrapid_v0_4_8_anysplat_only_18k \
  --suite-name fastergsfusedrapid_v0_4_8_anysplat_only_18k_bicycle \
  --repeats 1

python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --prior-mode none \
  --config-dir configs/fastergsfusedrapid_v0_4_6_anysplat_only_20k \
  --suite-name fastergsfusedrapid_v0_4_6_anysplat_only_20k_bicycle \
  --repeats 1

python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --prior-mode none \
  --config-dir configs/fastergsfusedrapid_v0_4_5_anysplat_only \
  --suite-name fastergsfusedrapid_v0_4_5_anysplat_only_bicycle \
  --repeats 1

python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --prior-mode none \
  --config-dir configs/fastergsfusedrapid_v0_4_5_depth_only \
  --suite-name fastergsfusedrapid_v0_4_5_depth_only_bicycle \
  --repeats 1

python scripts/run_fastergsfusedrapid_fast_converging.py \
  --scene bicycle \
  --skip-prior-generation \
  --prior-mode none \
  --config-dir configs/fastergsfusedrapid_v0_4_5_both_world_transform \
  --suite-name fastergsfusedrapid_v0_4_5_both_world_transform_bicycle \
  --repeats 1
```

Recent bicycle single-run comparison:

| config | train time | PSNR | SSIM | LPIPS | n_gaussians | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v0.3.14 baseline | 182.59s | 25.6324 | 0.7582 | 0.2941 | 1,246,630 | pre-prior fused rapid baseline |
| v0.4.5 AnySplat-only 30k | 211.34s | 25.3423 | 0.7484 | 0.2916 | 1,450,320 | quality restored after world transform |
| v0.4.5 depth-only 30k | 202.65s | 25.2751 | 0.7426 | 0.3102 | 1,303,899 | higher VRAM, no clear quality win |
| v0.4.5 AnySplat+Depth 30k | 243.88s | 24.3770 | 0.7036 | 0.2967 | 1,914,620 | not recommended |
| v0.4.6 AnySplat-only 20k | 150.40s | 25.3172 | 0.7464 | 0.2966 | 1,463,917 | conservative speed/quality point |
| v0.4.7 AnySplat-only 15k | 120.43s | 25.1440 | 0.7390 | 0.3092 | 1,507,526 | faster but quality drops |
| v0.4.8 AnySplat-only 18k | 139.10s | 25.3164 | 0.7441 | 0.2988 | 1,502,571 | all imported SH active from iteration 0 |
| v0.4.9 AnySplat-only 18k SH schedule | 139.33s | 25.3361 | 0.7460 | 0.2966 | 1,528,062 | single-scene predecessor |
| v0.4.10 all-scene baseline, bicycle mean | 138.55s | 25.3187 | 0.7457 | 0.2969 | 1,532,846 | all-scene predecessor |
| v0.4.11 depth-off backward template, bicycle mean | 138.83s | 25.3183 | 0.7460 | 0.2971 | 1,525,063 | neutral end-to-end, kept as depth path cleanup |
| v0.4.12 depth-off forward template, bicycle mean | 139.24s | 25.3324 | 0.7456 | 0.2969 | 1,534,271 | current code-level baseline; all-scene mean slightly faster |
| v0.4.13 optional depth buffers, bicycle mean | 139.28s | 25.3072 | 0.7455 | 0.2973 | 1,529,843 | not the default; tiny VRAM cleanup but all-scene mean slower |

Current all-scene repeat-3 baseline, `configs/fastergsfusedrapid_v0_4_12_forward_depth_template`:

| scene | train time mean | PSNR | SSIM | LPIPS | n_gaussians | peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 139.24s | 25.3324 | 0.7456 | 0.2969 | 1,534,271 | 4.8974GiB |
| bonsai | 87.63s | 31.3952 | 0.9361 | 0.2574 | 424,153 | 5.8015GiB |
| counter | 82.64s | 28.4569 | 0.8950 | 0.2834 | 280,981 | 5.1178GiB |
| garden | 92.94s | 26.7918 | 0.8352 | 0.1893 | 883,862 | 2.9975GiB |
| kitchen | 96.01s | 30.8333 | 0.9190 | 0.1745 | 411,713 | 5.6806GiB |
| room | 85.29s | 31.3614 | 0.9128 | 0.3047 | 374,206 | 6.1339GiB |
| stump | 94.86s | 25.8460 | 0.7284 | 0.3017 | 1,206,409 | 2.5740GiB |
| mean | 96.94s | 28.5738 | 0.8532 | 0.2583 | 730,799 | 4.7432GiB |

The current integration assumes Mip-NeRF 360 layout and uses scene-relative defaults:

```yaml
TRAINING:
  DEPTH_SUPERVISION:
    ACTIVE: true
    DIRECTORY: mono_depths
    PRESCALED_TO_TRAINING_RESOLUTION: true
  ANYSPLAT_INITIALIZATION:
    ACTIVE: true
    PATH: anysplat_init/point_cloud.ply
    REQUIRE: true
```

If `ANYSPLAT_INITIALIZATION.REQUIRE` is true and the PLY is missing, training fails before optimization instead of silently falling back to COLMAP initialization.

## 5. Run Full 360_v2 Benchmark

The dataset root is fixed to `dataset/mipnerf360` by default. The runner discovers all scene directories there and runs them serially.

Run all 7 scenes, 3 repeats per scene for one method:

```bash
python ./scripts/benchmark_360v2.py \
  -m <METHOD_NAME> \
  --config-dir configs/<CONFIG_DIR> \
  --repeats 3
```

For the current FasterGSFused baseline configs:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFused \
  --config-dir configs/fastergsfused_baseline \
  --repeats 3 \
  --suite-name fastergsfused_baseline
```

Multiple methods are supported, but then each config directory must be mapped explicitly:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFused FasterGSBasis \
  --config-dir FasterGSFused=configs/fastergsfused_baseline \
  --config-dir FasterGSBasis=configs/fastergsbasis_baseline \
  --repeats 3
```

Run a quick subset while developing:

```bash
python ./scripts/benchmark_360v2.py \
  -m <METHOD_NAME> \
  --config-dir configs/<CONFIG_DIR> \
  --repeats 1 \
  --set TRAINING.NUM_ITERATIONS=1000
```

This still runs all scenes, but for fewer iterations.

## 6. Output Layout

Each benchmark suite is written under:

```bash
output/benchmarks/benchmark360_<timestamp>/
```

Important files:

```text
results_runs.csv              # one row per method/scene/repeat
results_runs.json             # same data in JSON
summary_by_scene.csv          # repeat averages per method+scene
summary_by_scene.json
summary_by_method.csv         # machine-readable method-level aggregate
summary_by_method.json
summary.md                    # readable Markdown summary with commit, config dir, and per-scene results
configs/<method>/<scene>/     # generated config used for each run
logs/<method>/<scene>/        # train logs for each run
runs/<method>/<scene>/run_N/  # moved NeRFICG output directory
```

Per-run records include:

- method name
- scene name
- repeat index
- config name and path
- git commit hash
- dirty worktree flag
- wall-clock training time
- framework timing file value when available
- PSNR, SSIM, LPIPS
- final Gaussian count when available
- peak allocated and reserved VRAM
- output directory and log path

## 7. Single-Scene Manual Run

For debugging a generated config:

```bash
python ./scripts/train.py -c output/benchmarks/<SUITE>/configs/<METHOD>/<SCENE>/run_01_<CONFIG_NAME>.yaml
```

The benchmark script itself runs this command for every scene/repeat serially and moves the generated method output directory into the suite's `runs/` directory.
