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
from types import ModuleType

import numpy as np
from PIL import Image


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
    parser.add_argument('--metric3d-image-scale', type=float, default=None, help='Resize Metric3D workspace images and COLMAP intrinsics by this factor before inference.')
    parser.add_argument('--metric3d-output-dtype', choices=['float16', 'float32', 'float64'], default='float32', help='Convert Metric3D .npy priors to this dtype after inference.')
    parser.add_argument('--skip-existing-depths', action='store_true', help='Skip Metric3D if all expected train inverse-depth files already exist.')

    parser.add_argument('--anysplat-config', type=Path, default=None, help='AnySplat config.json path.')
    parser.add_argument('--anysplat-weights', type=Path, default=None, help='AnySplat model.safetensors path.')
    parser.add_argument('--vggt-weights', type=Path, default=None, help='VGGT-1B model.safetensors path used by the local AnySplat encoder.')
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
    args.vggt_weights = args.vggt_weights or root / 'VGGT-1B/model.safetensors'
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


def compute_scaled_image_size(size: tuple[int, int], scale_factor: float | None) -> tuple[int, int]:
    if scale_factor is None or scale_factor == 1.0:
        return size
    return round(size[0] * scale_factor), round(size[1] * scale_factor)


def prepare_scaled_images(source_root: Path, target_root: Path, image_names: list[str], scale_factor: float) -> tuple[int, int]:
    target_root.mkdir(parents=True, exist_ok=True)
    first_size = None
    resample = Image.Resampling.LANCZOS
    for name in image_names:
        source = source_root / name
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            scaled_size = compute_scaled_image_size(image.size, scale_factor)
            if first_size is None:
                first_size = scaled_size
            if target.is_file():
                with Image.open(target) as existing:
                    if existing.size == scaled_size:
                        continue
            image.resize(scaled_size, resample=resample).save(target)
    if first_size is None:
        raise RuntimeError('cannot prepare scaled images for an empty image list')
    return first_size


def scale_cameras_text(cameras_path: Path, scale_factor: float, scaled_size: tuple[int, int]) -> None:
    lines = cameras_path.read_text(encoding='utf-8').splitlines()
    scaled_lines = []
    for line in lines:
        if not line or line.startswith('#'):
            scaled_lines.append(line)
            continue
        parts = line.split()
        if len(parts) < 8 or parts[1] != 'PINHOLE':
            raise RuntimeError(f'unsupported camera line for Metric3D scaling: {line}')
        parts[2] = str(scaled_size[0])
        parts[3] = str(scaled_size[1])
        for idx in range(4, 8):
            parts[idx] = f'{float(parts[idx]) * scale_factor:.12g}'
        scaled_lines.append(' '.join(parts))
    cameras_path.write_text('\n'.join(scaled_lines) + '\n', encoding='utf-8')


def prepare_mipnerf360_workspace(scene_path: Path, images_dir: str, work_dir: Path, test_step: int, metric3d_image_scale: float | None) -> tuple[Path, dict[str, list[str]]]:
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
    if metric3d_image_scale is None or metric3d_image_scale == 1.0:
        symlink_or_replace(image_root, work_dir / 'images')
    else:
        scaled_size = prepare_scaled_images(image_root, work_dir / 'images', image_names, metric3d_image_scale)
        scale_cameras_text(sparse_text_root / 'cameras.txt', metric3d_image_scale, scaled_size)
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
    ]
    print('[Metric3D]', ' '.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=str(args.siggraph_asia_root / 'metric3D'), check=True)
        convert_metric3d_outputs(expected, args.metric3d_output_dtype)


def convert_metric3d_outputs(paths: list[Path], dtype_name: str) -> None:
    dtype = np.dtype(dtype_name)
    converted = 0
    for path in paths:
        if not path.is_file():
            continue
        inv_depth = np.load(path)
        if inv_depth.dtype == dtype:
            continue
        np.save(path, inv_depth.astype(dtype, copy=False))
        converted += 1
    if converted:
        print(f'[Metric3D] converted {converted} priors to {dtype_name}')


@contextmanager
def anysplat_runtime_root(args: argparse.Namespace, work_dir: Path):
    config = require_file(args.anysplat_config, 'AnySplat config')
    weights = require_file(args.anysplat_weights, 'AnySplat weights')
    vggt_weights = require_file(args.vggt_weights, 'VGGT-1B weights')
    runtime_root = work_dir / 'anysplat_runtime'
    ckpt_root = runtime_root / 'anySplat/ckpt'
    ckpt_root.mkdir(parents=True, exist_ok=True)
    symlink_or_replace(config, ckpt_root / 'config.json')
    symlink_or_replace(weights, ckpt_root / 'model.safetensors')

    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_hf_offline = os.environ.get('HF_HUB_OFFLINE')
    previous_vggt_weights = os.environ.get('ANY_SPLAT_VGGT_WEIGHTS')
    try:
        sys.path.insert(0, str(args.siggraph_asia_root))
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['ANY_SPLAT_VGGT_WEIGHTS'] = str(vggt_weights)
        os.chdir(runtime_root)
        yield runtime_root
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        if previous_hf_offline is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = previous_hf_offline
        if previous_vggt_weights is None:
            os.environ.pop('ANY_SPLAT_VGGT_WEIGHTS', None)
        else:
            os.environ['ANY_SPLAT_VGGT_WEIGHTS'] = previous_vggt_weights


def install_open3d_import_stub() -> None:
    """Allow AnySplat trajectory alignment to import without the heavy Open3D wheel.

    The local AnySplat helper only calls `estimate_similarity_transform` from
    `tnt_eval.registration`; that function does not use Open3D. The imported
    module still imports Open3D at top level for other TnT utilities, so provide
    the minimum module shape needed for that import path when Open3D is absent.
    """
    if 'open3d' in sys.modules:
        return
    try:
        __import__('open3d')
        return
    except ModuleNotFoundError:
        pass

    open3d = ModuleType('open3d')
    registration = ModuleType('open3d.registration')
    pipelines = ModuleType('open3d.pipelines')
    pipelines.registration = registration
    open3d.registration = registration
    open3d.pipelines = pipelines
    sys.modules['open3d'] = open3d
    sys.modules['open3d.registration'] = registration
    sys.modules['open3d.pipelines'] = pipelines
    sys.modules['open3d.pipelines.registration'] = registration


def install_siggraph_utils_namespace(siggraph_root: Path) -> None:
    utils_path = siggraph_root / 'utils'
    utils_module = ModuleType('utils')
    utils_module.__path__ = [str(utils_path)]  # type: ignore[attr-defined]
    utils_module.__package__ = 'utils'
    sys.modules['utils'] = utils_module


def run_anysplat(args: argparse.Namespace, work_dir: Path) -> None:
    output = args.anysplat_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f'[AnySplat] would write aligned Gaussian PLY to {output}')
        print(f'[AnySplat] config={args.anysplat_config} weights={args.anysplat_weights}')
        print(f'[AnySplat] vggt_weights={args.vggt_weights}')
        return

    with anysplat_runtime_root(args, work_dir):
        install_open3d_import_stub()
        install_siggraph_utils_namespace(args.siggraph_asia_root)
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

    work_dir, split = prepare_mipnerf360_workspace(args.scene_path, args.images_dir, args.work_dir, args.test_step, args.metric3d_image_scale)
    print(f'[prepare] workspace: {work_dir}')
    print(f'[prepare] split: {len(split["train"])} train, {len(split["test"])} test')
    if args.metric3d_image_scale is not None:
        print(f'[prepare] Metric3D image scale: {args.metric3d_image_scale}')

    if 'metric3d' in args.tasks:
        run_metric3d(args, work_dir, split)
    if 'anysplat' in args.tasks:
        run_anysplat(args, work_dir)


if __name__ == '__main__':
    main()
