from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from FasterGSFusedRapidCudaBackend import _C


_EMPTY_CPU = torch.empty(0)
_EMPTY_BOOL_BY_DEVICE: dict[torch.device, torch.Tensor] = {}
_EMPTY_FLOAT_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _empty_bool_like(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    empty = _EMPTY_BOOL_BY_DEVICE.get(device)
    if empty is None:
        empty = torch.empty(0, dtype=torch.bool, device=device)
        _EMPTY_BOOL_BY_DEVICE[device] = empty
    return empty


def _empty_float_like(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    empty = _EMPTY_FLOAT_BY_DEVICE.get(device)
    if empty is None:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        _EMPTY_FLOAT_BY_DEVICE[device] = empty
    return empty


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    bg_color: torch.Tensor  # background color in RGB format
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    current_mean_lr: float
    adam_step_count: int

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.current_mean_lr,
            self.adam_step_count,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        autograd_dummy: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        moments_means: torch.Tensor,
        moments_scales: torch.Tensor,
        moments_rotations: torch.Tensor,
        moments_opacities: torch.Tensor,
        moments_sh_coefficients_0: torch.Tensor,
        moments_sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        metric_map: torch.Tensor,
        render_inv_depth: bool,
        rasterizer_settings: RasterizerSettings,
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        (
            image,
            inv_depth,
            metric_counts,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            metric_map,
            render_inv_depth,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            inv_depth,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.means = means
        ctx.scales = scales
        ctx.rotations = rotations
        ctx.opacities = opacities
        ctx.sh_coefficients_0 = sh_coefficients_0
        ctx.sh_coefficients_rest = sh_coefficients_rest
        ctx.moments_means = moments_means
        ctx.moments_scales = moments_scales
        ctx.moments_rotations = moments_rotations
        ctx.moments_opacities = moments_opacities
        ctx.moments_sh_coefficients_0 = moments_sh_coefficients_0
        ctx.moments_sh_coefficients_rest = moments_sh_coefficients_rest
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(means)
        ctx.mark_non_differentiable(scales)
        ctx.mark_non_differentiable(rotations)
        ctx.mark_non_differentiable(opacities)
        ctx.mark_non_differentiable(sh_coefficients_0)
        ctx.mark_non_differentiable(sh_coefficients_rest)
        ctx.mark_non_differentiable(moments_means)
        ctx.mark_non_differentiable(moments_scales)
        ctx.mark_non_differentiable(moments_rotations)
        ctx.mark_non_differentiable(moments_opacities)
        ctx.mark_non_differentiable(moments_sh_coefficients_0)
        ctx.mark_non_differentiable(moments_sh_coefficients_rest)
        ctx.mark_non_differentiable(densification_info)
        ctx.mark_non_differentiable(metric_counts)
        return image, inv_depth, metric_counts, autograd_dummy

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
        grad_inv_depth: torch.Tensor | None,
        _,
        __,
    ) -> 'tuple[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]':
        _C.backward(
            ctx.densification_info,
            ctx.means,
            ctx.scales,
            ctx.rotations,
            ctx.opacities,
            ctx.sh_coefficients_0,
            ctx.sh_coefficients_rest,
            ctx.moments_means,
            ctx.moments_scales,
            ctx.moments_rotations,
            ctx.moments_opacities,
            ctx.moments_sh_coefficients_0,
            ctx.moments_sh_coefficients_rest,
            grad_image,
            _empty_float_like(grad_image) if grad_inv_depth is None else grad_inv_depth,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            None,  # autograd_dummy
            None,  # means
            None,  # scales
            None,  # rotations
            None,  # opacities
            None,  # sh_coefficients_0
            None,  # sh_coefficients_rest
            None,  # moments_means
            None,  # moments_scales
            None,  # moments_rotations
            None,  # moments_opacities
            None,  # moments_sh_coefficients_0
            None,  # moments_sh_coefficients_rest
            None,  # densification_info
            None,  # metric_map
            None,  # render_inv_depth
            None,  # rasterizer_settings
        )


def diff_rasterize(
    autograd_dummy: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    moments_means: torch.Tensor = None,
    moments_scales: torch.Tensor = None,
    moments_rotations: torch.Tensor = None,
    moments_opacities: torch.Tensor = None,
    moments_sh_coefficients_0: torch.Tensor = None,
    moments_sh_coefficients_rest: torch.Tensor = None,
    metric_map: torch.Tensor = None,
    return_metric_counts: bool = False,
    return_inv_depth: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    image, inv_depth, metric_counts, autograd_dummy = _Rasterize.apply(
        autograd_dummy,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        _EMPTY_CPU if moments_means is None else moments_means,
        _EMPTY_CPU if moments_scales is None else moments_scales,
        _EMPTY_CPU if moments_rotations is None else moments_rotations,
        _EMPTY_CPU if moments_opacities is None else moments_opacities,
        _EMPTY_CPU if moments_sh_coefficients_0 is None else moments_sh_coefficients_0,
        _EMPTY_CPU if moments_sh_coefficients_rest is None else moments_sh_coefficients_rest,
        densification_info,
        _empty_bool_like(means) if metric_map is None else metric_map,
        return_inv_depth,
        rasterizer_settings,
    )
    if return_metric_counts:
        return image, inv_depth, metric_counts
    return image, inv_depth, autograd_dummy


@torch.no_grad()
def rasterize_forward(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    metric_map: torch.Tensor = None,
    return_metric_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    (
        image,
        metric_counts,
    ) = _C.forward_image(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        _empty_bool_like(means) if metric_map is None else metric_map,
        *rasterizer_settings.as_tuple(),
    )
    if return_metric_counts:
        return image, metric_counts
    return image
