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

## 4. Run Full 360_v2 Benchmark

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

## 5. Output Layout

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

## 6. Single-Scene Manual Run

For debugging a generated config:

```bash
python ./scripts/train.py -c output/benchmarks/<SUITE>/configs/<METHOD>/<SCENE>/run_01_<CONFIG_NAME>.yaml
```

The benchmark script itself runs this command for every scene/repeat serially and moves the generated method output directory into the suite's `runs/` directory.
