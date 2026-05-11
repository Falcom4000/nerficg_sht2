#! /usr/bin/env python3

"""Prepare Fast-Converging 3DGS priors for Mip-NeRF 360 scenes.

This script intentionally runs AnySplat and Metric3D outside the timed training
loop. The prepared files are then consumed by FasterGSFusedRapid training:

- Metric3D: <scene>/mono_depths/<image_stem>_depth.npy, stored as inverse depth.
- AnySplat: <scene>/anysplat_init/point_cloud.ply, stored as 3DGS attributes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


DEFAULT_SIGGRAPH_ASIA_ROOT = Path('/root/codes/siggraph_asia')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene_path', type=Path, help='Mip-NeRF 360 scene directory, e.g. dataset/mipnerf360/bicycle')
    parser.add_argument('--dataset-type', default='mipnerf360', choices=['mipnerf360'], help='Only Mip-NeRF 360 is supported for now.')
    parser.add_argument('--tasks', nargs='+', choices=['metric3d', 'anysplat'], default=['metric3d', 'anysplat'])
    parser.add_argument('--siggraph-asia-root', type=Path, default=DEFAULT_SIGGRAPH_ASIA_ROOT)
    parser.add_argument('--images-dir', default='images', help='Image directory relative to the scene root. Use original-scale images for Metric3D/AnySplat.')
    parser.add_argument('--test-step', type=int, default=8, help='Mip-NeRF 360 train/test split stride; every Nth sorted image is test.')
    parser.add_argument('--work-dir', type=Path, default=None, help='Prepared COLMAP text-model workspace. Defaults to <scene>/.fast_converging_priors_work.')
    parser.add_argument('--dry-run', action='store_true', help='Print the planned commands without running model inference.')

    parser.add_argument('--metric3d-config', type=Path, default=None, help='Metric3D config path.')
    parser.add_argument('--metric3d-weights', type=Path, default=None, help='Metric3D checkpoint path.')
    parser.add_argument('--metric3d-output-root', type=Path, default=None, help='Metric3D output root; priors are written to <root>/mono_depths. Defaults to scene root.')
    parser.add_argument('--metric3d-batch-size', type=int, default=1)
    parser.add_argument('--skip-existing-depths', action='store_true', help='Skip Metric3D if all expected train inverse-depth files already exist.')

    parser.add_argument('--anysplat-config', type=Path, default=None, help='AnySplat config.json path.')
    parser.add_argument('--anysplat-weights', type=Path, default=None, help='AnySplat model.safetensors path.')
    parser.add_argument('--anysplat-output', type=Path, default=None, help='Output PLY path. Defaults to <scene>/anysplat_init/point_cloud.ply.')
    parser.add_argument('--anysplat-downsample', type=int, default=8, help='Use every Nth AnySplat Gaussian, matching siggraph_asia FF_downsample.')
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path


def set_default_paths(args: argparse.Namespace) -> None:
    root = args.siggraph_asia_root.expanduser().resolve()
    args.siggraph_asia_root = root
    args.metric3d_config = args.metric3d_config or root / 'metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py'
    args.metric3d_weights = args.metric3d_weights or root / 'metric3D/weight/metric_depth_vit_giant2_800k.pth'
    args.metric3d_output_root = args.metric3d_output_root or args.scene_path
    args.anysplat_config = args.anysplat_config or root / 'anySplat/ckpt/config.json'
    args.anysplat_weights = args.anysplat_weights or root / 'anySplat/ckpt/model.safetensors'
    args.anysplat_output = args.anysplat_output or args.scene_path / 'anysplat_init/point_cloud.ply'
    args.work_dir = args.work_dir or args.scene_path / '.fast_converging_priors_work'


def symlink_or_replace(source: Path, target: Path) -> None:
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        raise FileExistsError(f'cannot replace non-symlink path: {target}')
    target.symlink_to(source)


def build_split(image_names: list[str], test_step: int) -> dict[str, list[str]]:
    if test_step <= 0:
        return {'train': image_names, 'test': []}
    train, test = [], []
    for idx, name in enumerate(image_names):
        (test if idx % test_step == 0 else train).append(name)
    return {'train': train, 'test': test}


def prepare_mipnerf360_workspace(scene_path: Path, images_dir: str, work_dir: Path, test_step: int) -> tuple[Path, dict[str, list[str]]]:
    import pycolmap

    scene_path = require_dir(scene_path, 'scene directory')
    image_root = require_dir(scene_path / images_dir, 'scene image directory')
    sparse_root = require_dir(scene_path / 'sparse/0', 'scene COLMAP sparse model')
    work_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(str(sparse_root))
    image_names = sorted(image.name for image in reconstruction.images.values())
    split = build_split(image_names, test_step)

    sparse_text_root = work_dir / 'sparse/0'
    sparse_text_root.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(sparse_text_root))
    symlink_or_replace(image_root, work_dir / 'images')
    with (work_dir / 'train_test_split.json').open('w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)
    return work_dir, split


def run_metric3d(args: argparse.Namespace, work_dir: Path, split: dict[str, list[str]]) -> None:
    metric3d_config = args.metric3d_config.expanduser().resolve()
    metric3d_weights = args.metric3d_weights.expanduser().resolve()
    if not args.dry_run:
        metric3d_config = require_file(metric3d_config, 'Metric3D config')
        metric3d_weights = require_file(metric3d_weights, 'Metric3D weights')
    output_root = args.metric3d_output_root.expanduser().resolve()
    depth_dir = output_root / 'mono_depths'

    expected = [depth_dir / f'{Path(name).stem}_depth.npy' for name in split['train']]
    if args.skip_existing_depths and expected and all(path.is_file() for path in expected):
        print(f'[Metric3D] all {len(expected)} train priors already exist in {depth_dir}; skipping')
        return

    command = [
        sys.executable,
        str(args.siggraph_asia_root / 'metric3D/mono/tools/test_scale_cano.py'),
        str(metric3d_config),
        '--load-from', str(metric3d_weights),
        '--test_data_path', str(work_dir / 'images'),
        '--show-dir', str(work_dir),
        '--output-dir', str(output_root),
        '--launcher', 'None',
        '--batch_size', str(args.metric3d_batch_size),
        '--vis', 'False',
    ]
    print('[Metric3D]', ' '.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=str(args.siggraph_asia_root / 'metric3D'), check=True)


@contextmanager
def anysplat_runtime_root(args: argparse.Namespace, work_dir: Path):
    config = require_file(args.anysplat_config, 'AnySplat config')
    weights = require_file(args.anysplat_weights, 'AnySplat weights')
    runtime_root = work_dir / 'anysplat_runtime'
    ckpt_root = runtime_root / 'anySplat/ckpt'
    ckpt_root.mkdir(parents=True, exist_ok=True)
    symlink_or_replace(config, ckpt_root / 'config.json')
    symlink_or_replace(weights, ckpt_root / 'model.safetensors')

    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(args.siggraph_asia_root))
        os.chdir(runtime_root)
        yield runtime_root
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def run_anysplat(args: argparse.Namespace, work_dir: Path) -> None:
    output = args.anysplat_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f'[AnySplat] would write aligned Gaussian PLY to {output}')
        print(f'[AnySplat] config={args.anysplat_config} weights={args.anysplat_weights}')
        return

    with anysplat_runtime_root(args, work_dir):
        from utils.anysplat_utils import anySplat

        dataset = SimpleNamespace(
            source_path=str(work_dir),
            images='images',
            model_path=str(output.parent),
            train_test_exp=True,
        )
        opt = SimpleNamespace(optimizer_type='default')
        pipe = SimpleNamespace(FF_downsample=args.anysplat_downsample)
        gaussians, _, _ = anySplat(dataset, opt, pipe)
        gaussians.save_ply(str(output))
        metadata = {
            'scene_path': str(args.scene_path.resolve()),
            'work_dir': str(work_dir.resolve()),
            'anysplat_config': str(args.anysplat_config),
            'anysplat_weights': str(args.anysplat_weights),
            'anysplat_downsample': args.anysplat_downsample,
        }
        with output.with_suffix('.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f'[AnySplat] wrote {output}')


def main() -> None:
    args = parse_args()
    args.scene_path = args.scene_path.expanduser().resolve()
    set_default_paths(args)
    require_dir(args.siggraph_asia_root, 'siggraph_asia root')

    work_dir, split = prepare_mipnerf360_workspace(args.scene_path, args.images_dir, args.work_dir, args.test_step)
    print(f'[prepare] workspace: {work_dir}')
    print(f'[prepare] split: {len(split["train"])} train, {len(split["test"])} test')

    if 'metric3d' in args.tasks:
        run_metric3d(args, work_dir, split)
    if 'anysplat' in args.tasks:
        run_anysplat(args, work_dir)


if __name__ == '__main__':
    main()
