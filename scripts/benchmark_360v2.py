#! /usr/bin/env python3

"""benchmark_360v2.py: Serial repeated Mip-NeRF 360 benchmark runner for one or more methods."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Datasets.utils import list_sorted_directories


MIPNERF360_SCALE_OVERRIDES = {
    'bonsai': 0.5,
    'counter': 0.5,
    'kitchen': 0.5,
    'room': 0.5,
}

RUN_COLUMNS = [
    'suite_name',
    'timestamp',
    'method',
    'scene',
    'repeat',
    'status',
    'returncode',
    'config_name',
    'config_path',
    'output_dir',
    'log_path',
    'commit',
    'dirty',
    'dataset_type',
    'dataset_root',
    'dataset_path',
    'num_iterations',
    'image_scale_factor',
    'wall_time_sec',
    'train_time_sec',
    'PSNR',
    'SSIM',
    'LPIPS',
    'vram_allocated_bytes',
    'vram_reserved_bytes',
    'vram_allocated_gb',
    'vram_reserved_gb',
    'n_gaussians',
]


def run_command(command: list[str], *, log_path: Path | None = None, cwd: Path | None = None) -> int:
    """Runs a command, teeing stdout/stderr to the terminal and an optional log file."""
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, 'w')
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        return process.wait()
    finally:
        if log_file is not None:
            log_file.close()


def command_output(command: list[str], default: str = '') -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def get_git_state() -> tuple[str, bool]:
    commit = command_output(['git', 'rev-parse', 'HEAD'], default='unknown')
    dirty = bool(command_output(['git', 'status', '--porcelain'], default=''))
    return commit, dirty


def normalize_dataset_type(dataset_type: str) -> str:
    if dataset_type.lower() == 'mipnerf360':
        return 'MipNeRF360'
    return dataset_type


def parse_template_args(items: list[str]) -> dict[str, Path]:
    templates = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'invalid --template "{item}", expected METHOD=path/to/config.yaml')
        method, path = item.split('=', 1)
        templates[method] = Path(path)
    return templates


def parse_config_dir_args(items: list[str], methods: list[str]) -> dict[str, Path]:
    config_dirs = {}
    for item in items:
        if '=' in item:
            method, path = item.split('=', 1)
        elif len(methods) == 1:
            method, path = methods[0], item
        else:
            raise ValueError(f'invalid --config-dir "{item}", expected METHOD=path when benchmarking multiple methods')
        config_dirs[method] = Path(path)
    return config_dirs


def parse_set_args(items: list[str]) -> dict[str, Any]:
    overrides = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'invalid --set "{item}", expected A.B=value')
        key, raw_value = item.split('=', 1)
        try:
            value = ast.literal_eval(raw_value)
        except Exception:
            value = raw_value
        overrides[key] = value
    return overrides


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    target: Any = config
    for part in dotted_key.split('.'):
        if not isinstance(target, dict) or part not in target:
            return default
        target = target[part]
    return target


def generate_default_config(method: str, dataset_type: str) -> dict[str, Any]:
    Framework.config = Framework.ConfigWrapper(GLOBAL=Framework.get_default_global_config())
    Framework.config.GLOBAL.METHOD_TYPE = method
    Framework.config.GLOBAL.DATASET_TYPE = dataset_type
    method_module = MI.import_method(method)
    Framework.config.MODEL = method_module.MODEL.get_default_parameters()
    Framework.config.RENDERER = method_module.RENDERER.get_default_parameters()
    Framework.config.TRAINING = method_module.TRAINING_INSTANCE.get_default_parameters()
    Framework.config.DATASET = DI.get_dataset_class(dataset_type).get_default_parameters()
    config = Framework.ConfigParameterList.toDict(Framework.config)
    if 'config' in Framework.__dict__:
        del Framework.config
    return config


def load_base_config(method: str, dataset_type: str, templates: dict[str, Path]) -> dict[str, Any]:
    if method in templates:
        with open(templates[method], 'r') as f:
            config = yaml.safe_load(f)
        config['GLOBAL']['METHOD_TYPE'] = method
        config['GLOBAL']['DATASET_TYPE'] = dataset_type
        return config
    return generate_default_config(method, dataset_type)


def load_scene_config(
    method: str,
    scene: str,
    dataset_type: str,
    templates: dict[str, Path],
    config_dirs: dict[str, Path],
) -> dict[str, Any]:
    if method in config_dirs:
        config_path = config_dirs[method] / f'{scene}.yaml'
        if not config_path.is_file():
            raise FileNotFoundError(f'missing scene config for {method}/{scene}: {config_path}')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['GLOBAL']['METHOD_TYPE'] = method
        config['GLOBAL']['DATASET_TYPE'] = dataset_type
        return config
    return load_base_config(method, dataset_type, templates)


def prepare_run_config(
    base_config: dict[str, Any],
    *,
    method: str,
    scene: str,
    repeat: int,
    dataset_root: Path,
    dataset_type: str,
    config_name_prefix: str,
    user_overrides: dict[str, Any],
    apply_mipnerf360_scale_defaults: bool,
) -> dict[str, Any]:
    config = yaml.safe_load(yaml.safe_dump(base_config))
    model_name = f'{config_name_prefix}_{method}_{scene}_r{repeat:02d}'
    config['GLOBAL']['METHOD_TYPE'] = method
    config['GLOBAL']['DATASET_TYPE'] = dataset_type
    config['TRAINING']['MODEL_NAME'] = model_name
    config['DATASET']['PATH'] = str(dataset_root / scene)
    if apply_mipnerf360_scale_defaults and dataset_type == 'MipNeRF360':
        config['DATASET']['IMAGE_SCALE_FACTOR'] = MIPNERF360_SCALE_OVERRIDES.get(scene, 0.25)

    for key, value in user_overrides.items():
        set_nested(config, key, value)

    config['TRAINING']['WANDB']['ACTIVATE'] = False
    config['TRAINING']['TIMING']['ACTIVATE'] = True
    config['TRAINING']['WRITE_VRAM_STATS'] = True
    config['TRAINING']['BACKUP']['FINAL_CHECKPOINT'] = True
    config['TRAINING']['BACKUP']['RENDER_TESTSET'] = True
    config['TRAINING']['BACKUP']['INTERMEDIATE_RENDERINGS'] = False
    config['TRAINING']['BACKUP']['INTERVAL'] = -1
    return config


def parse_colon_line(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    last_line = ''
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    values = {}
    for token in last_line.split():
        if ':' not in token:
            continue
        key, raw_value = token.split(':', 1)
        try:
            values[key] = float(raw_value)
        except ValueError:
            pass
    return values


def parse_n_gaussians(output_dir: Path) -> int | None:
    values = parse_colon_line(output_dir / 'n_gaussians.txt')
    if 'N_Gaussians' in values:
        return int(values['N_Gaussians'])

    checkpoint_path = output_dir / 'checkpoints' / 'final.pt'
    if not checkpoint_path.is_file():
        return None

    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', {})
        for key in [
            'gaussians._means',
            'gaussians._positions',
            'gaussians._features_dc',
        ]:
            tensor = state_dict.get(key)
            if tensor is not None:
                return int(tensor.shape[0])
    except Exception:
        return None
    return None


def detect_new_output_dir(method: str, before: set[Path], repo_root: Path) -> Path | None:
    method_output = repo_root / 'output' / method
    if not method_output.is_dir():
        return None
    after = {path for path in method_output.iterdir() if path.is_dir()}
    new_dirs = list(after - before)
    if not new_dirs:
        return None
    return max(new_dirs, key=lambda path: path.stat().st_mtime)


def parse_run_outputs(output_dir: Path | None, num_iterations: int) -> dict[str, Any]:
    if output_dir is None:
        return {}
    metrics = parse_colon_line(output_dir / f'test_{num_iterations}' / 'metrics_8bit.txt')
    timings = parse_colon_line(output_dir / 'timings.txt')
    vram = parse_colon_line(output_dir / 'vram_stats.txt')
    result = {
        'train_time_sec': timings.get('Time'),
        'PSNR': metrics.get('PSNR'),
        'SSIM': metrics.get('SSIM'),
        'LPIPS': metrics.get('LPIPS'),
        'vram_allocated_bytes': vram.get('VRAM_allocated'),
        'vram_reserved_bytes': vram.get('VRAM_reserved'),
        'n_gaussians': parse_n_gaussians(output_dir),
    }
    if result['vram_allocated_bytes'] is not None:
        result['vram_allocated_gb'] = result['vram_allocated_bytes'] / 1024 ** 3
    if result['vram_reserved_bytes'] is not None:
        result['vram_reserved_gb'] = result['vram_reserved_bytes'] / 1024 ** 3
    return result


def numeric_mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [record.get(field) for record in records if isinstance(record.get(field), (int, float))]
    return mean(values) if values else None


def numeric_stdev(records: list[dict[str, Any]], field: str) -> float | None:
    values = [record.get(field) for record in records if isinstance(record.get(field), (int, float))]
    return stdev(values) if len(values) > 1 else None


def summarize(records: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for record in records:
        if record.get('status') == 'ok':
            groups[tuple(record[field] for field in group_fields)].append(record)

    numeric_fields = [
        'wall_time_sec',
        'train_time_sec',
        'PSNR',
        'SSIM',
        'LPIPS',
        'vram_allocated_gb',
        'vram_reserved_gb',
        'n_gaussians',
    ]
    summary = []
    for key, group_records in sorted(groups.items()):
        item = dict(zip(group_fields, key))
        item['n_runs'] = len(group_records)
        for field in numeric_fields:
            item[f'{field}_mean'] = numeric_mean(group_records, field)
            item[f'{field}_std'] = numeric_stdev(group_records, field)
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row.keys()})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    path: Path,
    records: list[dict[str, Any]],
    scene_summary: list[dict[str, Any]],
    config_dirs: dict[str, Path],
    suite_dir: Path,
) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value)

    def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
        headers = [header for header, _ in columns]
        lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        for row in rows:
            lines.append('| ' + ' | '.join(fmt(row.get(key)) for _, key in columns) + ' |')
        return '\n'.join(lines)

    scene_columns = [
        ('method', 'method'),
        ('scene', 'scene'),
        ('runs', 'n_runs'),
        ('wall_time_sec', 'wall_time_sec_mean'),
        ('train_time_sec', 'train_time_sec_mean'),
        ('PSNR', 'PSNR_mean'),
        ('SSIM', 'SSIM_mean'),
        ('LPIPS', 'LPIPS_mean'),
        ('n_gaussians', 'n_gaussians_mean'),
        ('vram_allocated_gb', 'vram_allocated_gb_mean'),
        ('vram_reserved_gb', 'vram_reserved_gb_mean'),
    ]

    first_record = records[0] if records else {}
    methods = sorted({str(record.get('method')) for record in records if record.get('method')})
    config_rows = [
        {
            'method': method,
            'source_config_dir': str(config_dirs[method]) if method in config_dirs else 'generated from defaults',
            'generated_config_dir': str(suite_dir / 'configs' / method),
        }
        for method in methods
    ]

    with open(path, 'w') as f:
        f.write('# 360_v2 Benchmark Summary\n\n')
        f.write(f'- suite: {first_record.get("suite_name", "")}\n')
        f.write(f'- generated_at: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'- commit: {first_record.get("commit", "")}\n')
        f.write(f'- git_dirty: {first_record.get("dirty", "")}\n')
        f.write(f'- dataset_root: {first_record.get("dataset_root", "")}\n')
        f.write(f'- repeats: {max((int(record.get("repeat", 0)) for record in records), default=0)}\n\n')
        f.write('## Configs\n\n')
        f.write(table(config_rows, [
            ('method', 'method'),
            ('source_config_dir', 'source_config_dir'),
            ('generated_config_dir', 'generated_config_dir'),
        ]))
        f.write('\n\n## Results\n\n')
        f.write(table(scene_summary, scene_columns))
        f.write('\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run repeated 360_v2/MipNeRF360 benchmarks serially.')
    parser.add_argument('-m', '--methods', nargs='+', required=True, help='Method names in src/Methods.')
    parser.add_argument('--dataset-root', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('-d', '--dataset-type', default='MipNeRF360')
    parser.add_argument('--scenes', nargs='*', default=None, help=argparse.SUPPRESS)
    parser.add_argument('-r', '--repeats', type=int, default=3)
    parser.add_argument('--config-dir', action='append', default=[], help='Scene config directory. Use path for one method or METHOD=path for multiple methods.')
    parser.add_argument('--template', action='append', default=[], help=argparse.SUPPRESS)
    parser.add_argument('--set', dest='sets', action='append', default=[], help='Config override A.B=value. Can be repeated.')
    parser.add_argument('--suite-name', default=None)
    parser.add_argument('--config-name-prefix', default='benchmark360')
    parser.add_argument('--install', action='store_true', help='Run scripts/install.py -m METHOD before benchmarking.')
    parser.add_argument('--no-mipnerf360-scale-defaults', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root or (repo_root / 'dataset' / 'mipnerf360')
    dataset_type = normalize_dataset_type(args.dataset_type)
    templates = parse_template_args(args.template)
    config_dirs = parse_config_dir_args(args.config_dir, args.methods)
    user_overrides = parse_set_args(args.sets)
    scenes = args.scenes or list_sorted_directories(dataset_root)
    suite_name = args.suite_name or f'benchmark360_{datetime.now():%Y-%m-%d-%H-%M-%S}'
    suite_dir = repo_root / 'output' / 'benchmarks' / suite_name
    suite_dir.mkdir(parents=True, exist_ok=False)

    commit, dirty = get_git_state()
    records: list[dict[str, Any]] = []

    print('benchmark execution mode: serial; only one training process is run at a time')
    print(f'dataset root: {dataset_root}')
    print(f'scenes: {", ".join(scenes)}')

    for method in args.methods:
        if args.install:
            returncode = run_command(
                [sys.executable, './scripts/install.py', '-m', method],
                log_path=suite_dir / 'install_logs' / f'{method}.log',
                cwd=repo_root,
            )
            if returncode != 0:
                raise RuntimeError(f'extension installation failed for method {method}')

    for method in args.methods:
        for scene in scenes:
            for repeat in range(1, args.repeats + 1):
                base_config = load_scene_config(method, scene, dataset_type, templates, config_dirs)
                config = prepare_run_config(
                    base_config,
                    method=method,
                    scene=scene,
                    repeat=repeat,
                    dataset_root=dataset_root,
                    dataset_type=dataset_type,
                    config_name_prefix=args.config_name_prefix,
                    user_overrides=user_overrides,
                    apply_mipnerf360_scale_defaults=not args.no_mipnerf360_scale_defaults,
                )
                config_name = config['TRAINING']['MODEL_NAME']
                config_path = suite_dir / 'configs' / method / scene / f'run_{repeat:02d}_{config_name}.yaml'
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f, sort_keys=False)

                log_path = suite_dir / 'logs' / method / scene / f'run_{repeat:02d}.log'
                method_output = repo_root / 'output' / method
                before = {path for path in method_output.iterdir() if path.is_dir()} if method_output.is_dir() else set()

                print(f'\n=== {method} / {scene} / run {repeat}/{args.repeats} ===')
                start_time = time.perf_counter()
                if args.dry_run:
                    returncode = 0
                else:
                    returncode = run_command(
                        [sys.executable, './scripts/train.py', '-c', str(config_path)],
                        log_path=log_path,
                        cwd=repo_root,
                    )
                wall_time_sec = time.perf_counter() - start_time
                output_dir = None if args.dry_run else detect_new_output_dir(method, before, repo_root)

                final_output_dir = None
                if output_dir is not None:
                    final_output_dir = suite_dir / 'runs' / method / scene / f'run_{repeat:02d}'
                    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
                    if final_output_dir.exists():
                        shutil.rmtree(final_output_dir)
                    shutil.move(str(output_dir), str(final_output_dir))

                record = {
                    'suite_name': suite_name,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'method': method,
                    'scene': scene,
                    'repeat': repeat,
                    'status': 'ok' if returncode == 0 else 'failed',
                    'returncode': returncode,
                    'config_name': config_name,
                    'config_path': str(config_path),
                    'output_dir': None if final_output_dir is None else str(final_output_dir),
                    'log_path': str(log_path),
                    'commit': commit,
                    'dirty': dirty,
                    'dataset_type': dataset_type,
                    'dataset_root': str(dataset_root),
                    'dataset_path': config['DATASET']['PATH'],
                    'num_iterations': get_nested(config, 'TRAINING.NUM_ITERATIONS'),
                    'image_scale_factor': get_nested(config, 'DATASET.IMAGE_SCALE_FACTOR'),
                    'wall_time_sec': wall_time_sec,
                }
                record.update(parse_run_outputs(final_output_dir, int(record['num_iterations'] or 0)))
                records.append(record)

                if final_output_dir is not None:
                    with open(final_output_dir / 'benchmark_record.json', 'w') as f:
                        json.dump(record, f, indent=2)

                write_csv(suite_dir / 'results_runs.csv', records, RUN_COLUMNS)
                with open(suite_dir / 'results_runs.json', 'w') as f:
                    json.dump(records, f, indent=2)

                if returncode != 0:
                    raise RuntimeError(f'training failed for {method}/{scene}/run_{repeat:02d}; see {log_path}')

    scene_summary = summarize(records, ['method', 'scene'])
    method_summary = summarize(records, ['method'])
    write_csv(suite_dir / 'summary_by_scene.csv', scene_summary)
    write_csv(suite_dir / 'summary_by_method.csv', method_summary)
    with open(suite_dir / 'summary_by_scene.json', 'w') as f:
        json.dump(scene_summary, f, indent=2)
    with open(suite_dir / 'summary_by_method.json', 'w') as f:
        json.dump(method_summary, f, indent=2)
    write_markdown_summary(suite_dir / 'summary.md', records, scene_summary, config_dirs, suite_dir)
    print(f'\nbenchmark suite written to: {suite_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
