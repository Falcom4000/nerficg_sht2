"""FasterGSFusedRapid/Renderer.py"""

import math

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGSFusedRapid.Model import FasterGSFusedRapidModel
from Methods.FasterGSFusedRapid.FasterGSFusedRapidCudaBackend import diff_rasterize, rasterize_forward, RasterizerSettings


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    current_mean_lr: float,
    adam_step_count: int,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGSFusedRapid renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    w2c, position = _cached_view_pose_tensors(view)
    return RasterizerSettings(
        w2c,
        position,
        bg_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        current_mean_lr,
        adam_step_count,
    )


def _cached_view_pose_tensors(view: View) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns cached static camera pose tensors for training dataset views."""
    c2w = getattr(view, '_c2w', None)
    cache_key = (id(c2w), str(Framework.config.GLOBAL.DEFAULT_DEVICE))
    cache = getattr(view, '_fastergsfusedrapid_pose_cache', None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2]

    w2c = view.w2c.contiguous()
    position = view.position.contiguous()
    setattr(view, '_fastergsfusedrapid_pose_cache', (cache_key, w2c, position))
    return w2c, position


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
)
class FasterGSFusedRapidRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSFusedRapidModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGSFusedRapid renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGSFusedRapid renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        self._empty_float: torch.Tensor | None = None
        self._empty_bool: torch.Tensor | None = None

    def _empty_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        cache_name = '_empty_bool' if dtype == torch.bool else '_empty_float'
        empty = getattr(self, cache_name)
        device = self.model.gaussians.means.device
        if empty is None or empty.device != device:
            empty = torch.empty(0, dtype=dtype, device=device)
            setattr(self, cache_name, empty)
        return empty

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor, adam_step_count: int, autograd_dummy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Renders an image for a given view."""
        image, autograd_dummy = diff_rasterize(
            means=self.model.gaussians.means,
            moments_means=self.model.gaussians.moments_means,
            scales=self.model.gaussians.raw_scales,
            moments_scales=self.model.gaussians.moments_scales,
            rotations=self.model.gaussians.raw_rotations,
            moments_rotations=self.model.gaussians.moments_rotations,
            opacities=self.model.gaussians.raw_opacities,
            moments_opacities=self.model.gaussians.moments_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            moments_sh_coefficients_0=self.model.gaussians.moments_sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            moments_sh_coefficients_rest=self.model.gaussians.moments_sh_coefficients_rest,
            autograd_dummy=autograd_dummy,
            densification_info=self.model.gaussians.densification_info if update_densification_info else self._empty_tensor(torch.float32),
            metric_map=self._empty_tensor(torch.bool),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.model.gaussians.lr_means, adam_step_count),
        )
        return image, autograd_dummy

    @torch.no_grad()
    def render_image_fastgs_score(self, view: View, bg_color: torch.Tensor = None) -> torch.Tensor:
        """Renders an unclamped image for FastGS score computation without optimizer updates."""
        image = rasterize_forward(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            metric_map=self._empty_tensor(torch.bool),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color if bg_color is None else bg_color, 0.0, 0),
        )
        return image

    @torch.no_grad()
    def render_image_metric_counts(self, view: View, metric_map: torch.Tensor, bg_color: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Renders an image and counts high-error pixel hits per Gaussian."""
        image, metric_counts = rasterize_forward(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            metric_map=metric_map.reshape(-1).contiguous(),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color if bg_color is None else bg_color, 0.0, 0),
            return_metric_counts=True,
        )
        return image, metric_counts

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image = rasterize_forward(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            metric_map=self._empty_tensor(torch.bool),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, 0.0, 0),
        )
        image = image.clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
