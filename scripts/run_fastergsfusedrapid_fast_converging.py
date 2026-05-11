#! /usr/bin/env python3

"""Run the FasterGSFusedRapid Fast-Converging offline-to-training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='bicycle', help='Mip-NeRF 360 scene name.')
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset/mipnerf360'))
    parser.add_argument('--config-dir', type=Path, default=Path('configs/fastergsfusedrapid_v0_4_2_fast_converging'))
    parser.add_argument('--suite-name', default='fastergsfusedrapid_v0_4_2_fast_converging')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--skip-prior-generation', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run-priors', action='store_true')
    parser.add_argument('--skip-existing-depths', action='store_true')
    parser.add_argument('--metric3d-config', type=Path, default=None)
    parser.add_argument('--metric3d-weights', type=Path, default=None)
    parser.add_argument('--anysplat-config', type=Path, default=None)
    parser.add_argument('--anysplat-weights', type=Path, default=None)
    parser.add_argument('--siggraph-asia-root', type=Path, default=Path('/root/codes/siggraph_asia'))
    return parser.parse_args()


def add_optional_path(command: list[str], flag: str, path: Path | None) -> None:
    if path is not None:
        command.extend([flag, str(path)])


def run(command: list[str], cwd: Path) -> None:
    print('[run]', ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scene_path = args.dataset_root / args.scene

    if not args.skip_prior_generation:
        prepare_command = [
            sys.executable,
            str(repo_root / 'scripts/prepare_fast_converging_priors.py'),
            str(scene_path),
            '--tasks', 'metric3d', 'anysplat',
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
