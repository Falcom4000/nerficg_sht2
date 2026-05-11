#! /usr/bin/env python3

"""Run the FasterGSFusedRapid Fast-Converging offline-to-training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='bicycle', help='Mip-NeRF 360 scene name.')
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset/mipnerf360'))
    parser.add_argument('--config-dir', type=Path, default=Path('configs/fastergsfusedrapid_v0_4_12_forward_depth_template'))
    parser.add_argument('--suite-name', default='fastergsfusedrapid_v0_4_12_forward_depth_template')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--prior-mode', choices=['both', 'metric3d', 'anysplat', 'none'], default='anysplat')
    parser.add_argument('--skip-prior-generation', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run-priors', action='store_true')
    parser.add_argument('--skip-existing-depths', action='store_true')
    parser.add_argument('--metric3d-config', type=Path, default=None)
    parser.add_argument('--metric3d-weights', type=Path, default=None)
    parser.add_argument('--metric3d-image-scale', type=float, default=None)
    parser.add_argument('--anysplat-config', type=Path, default=None)
    parser.add_argument('--anysplat-weights', type=Path, default=None)
    parser.add_argument('--vggt-weights', type=Path, default=None)
    parser.add_argument('--siggraph-asia-root', type=Path, default=Path('/root/codes/siggraph_asia'))
    return parser.parse_args()


def add_optional_path(command: list[str], flag: str, path: Path | None) -> None:
    if path is not None:
        command.extend([flag, str(path)])


def run(command: list[str], cwd: Path) -> None:
    print('[run]', ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def infer_metric3d_image_scale(config_dir: Path, scene: str) -> float | None:
    config_path = config_dir / f'{scene}.yaml'
    if not config_path.is_file():
        return None
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    depth_config = config.get('TRAINING', {}).get('DEPTH_SUPERVISION', {})
    if not depth_config.get('PRESCALED_TO_TRAINING_RESOLUTION', False):
        return None
    return config.get('DATASET', {}).get('IMAGE_SCALE_FACTOR')


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scene_path = args.dataset_root / args.scene

    if not args.skip_prior_generation:
        tasks_by_mode = {
            'both': ['metric3d', 'anysplat'],
            'metric3d': ['metric3d'],
            'anysplat': ['anysplat'],
            'none': [],
        }
        prior_tasks = tasks_by_mode[args.prior_mode]
        if prior_tasks:
            prepare_command = [
                sys.executable,
                str(repo_root / 'scripts/prepare_fast_converging_priors.py'),
                str(scene_path),
                '--tasks', *prior_tasks,
                '--siggraph-asia-root', str(args.siggraph_asia_root),
            ]
            if args.dry_run_priors:
                prepare_command.append('--dry-run')
            if args.skip_existing_depths:
                prepare_command.append('--skip-existing-depths')
            add_optional_path(prepare_command, '--metric3d-config', args.metric3d_config)
            add_optional_path(prepare_command, '--metric3d-weights', args.metric3d_weights)
            add_optional_path(prepare_command, '--anysplat-config', args.anysplat_config)
            add_optional_path(prepare_command, '--anysplat-weights', args.anysplat_weights)
            add_optional_path(prepare_command, '--vggt-weights', args.vggt_weights)
            metric3d_image_scale = args.metric3d_image_scale
            if metric3d_image_scale is None:
                metric3d_image_scale = infer_metric3d_image_scale(args.config_dir, args.scene)
            if metric3d_image_scale is not None:
                prepare_command.extend(['--metric3d-image-scale', str(metric3d_image_scale)])
            run(prepare_command, repo_root)

    if args.prepare_only:
        return

    benchmark_command = [
        sys.executable,
        str(repo_root / 'scripts/benchmark_360v2.py'),
        '-m', 'FasterGSFusedRapid',
        '--config-dir', str(args.config_dir),
        '--repeats', str(args.repeats),
        '--suite-name', args.suite_name,
        '--scenes', args.scene,
    ]
    run(benchmark_command, repo_root)


if __name__ == '__main__':
    main()
