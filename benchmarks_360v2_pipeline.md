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

The benchmark script can also generate configs internally from the method and dataset defaults, so this step is optional.

For a custom method-specific template, pass it explicitly:

```bash
python ./scripts/benchmark_360v2.py \
  -m <METHOD_NAME> \
  --template <METHOD_NAME>=configs/<CONFIG_NAME>.yaml
```

Useful benchmark defaults are forced by the runner:

- `TRAINING.WANDB.ACTIVATE=false`
- `TRAINING.TIMING.ACTIVATE=true`
- `TRAINING.WRITE_VRAM_STATS=true`
- `TRAINING.BACKUP.FINAL_CHECKPOINT=true`
- `TRAINING.BACKUP.RENDER_TESTSET=true`
- `TRAINING.BACKUP.INTERMEDIATE_RENDERINGS=false`

## 4. Run Full 360_v2 Benchmark

Dataset root used in this workspace:

```bash
/root/codes/360_v2
```

Run all 7 scenes, 3 repeats per scene:

```bash
python ./scripts/benchmark_360v2.py \
  -m <METHOD_NAME> \
  --dataset-root /root/codes/360_v2 \
  --repeats 3
```

Run multiple methods:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFused FasterGSBasis <METHOD_NAME> \
  --dataset-root /root/codes/360_v2 \
  --repeats 3
```

Run with method templates and extension installation:

```bash
python ./scripts/benchmark_360v2.py \
  -m FasterGSFused FasterGSBasis \
  --template FasterGSFused=src/Methods/FasterGSFused/fastergsfused_garden.yaml \
  --template FasterGSBasis=src/Methods/FasterGSBasis/fastergsbasis_garden.yaml \
  --dataset-root /root/codes/360_v2 \
  --repeats 3 \
  --install
```

Run a quick subset while developing:

```bash
python ./scripts/benchmark_360v2.py \
  -m <METHOD_NAME> \
  --dataset-root /root/codes/360_v2 \
  --scenes garden bonsai \
  --repeats 1 \
  --set TRAINING.NUM_ITERATIONS=1000
```

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
summary_by_method.csv         # averages across all successful scenes/runs per method
summary_by_method.json
summary.md                    # readable Markdown summary
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
