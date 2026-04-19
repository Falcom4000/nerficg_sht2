"""FastGS/Model.py"""

import math

import torch
import numpy as np

import Framework
from CudaUtils.MortonEncoding import morton_encode
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.Model import BaseModel
from Cameras.utils import quaternion_to_rotation_matrix
from Optim.adam_utils import replace_param_group_data, prune_param_groups, extend_param_groups, sort_param_groups
from Optim.lr_utils import LRDecayPolicy
from Optim.knn_utils import compute_root_mean_squared_knn_distances


class Gaussians(torch.nn.Module):
    """Stores a set of 3D Gaussians for FastGS."""

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.register_parameter('_means', None)
        self.register_parameter('_sh_coefficients_0', None)
        self.register_parameter('_sh_coefficients_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self._densification_info = None
        self.optimizer = None
        self.sh_optimizer = None
        self.percent_dense = 0.0
        self.training_cameras_extent = 1.0
        self.lr_means = 0.0
        self.lr_means_scheduler = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def means(self) -> torch.Tensor:
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        return self._scales.exp()

    @property
    def raw_scales(self) -> torch.Tensor:
        return self._scales

    @property
    def rotations(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotations)

    @property
    def raw_rotations(self) -> torch.Tensor:
        return self._rotations

    @property
    def opacities(self) -> torch.Tensor:
        return self._opacities.sigmoid()

    @property
    def raw_opacities(self) -> torch.Tensor:
        return self._opacities

    @property
    def sh_coefficients(self) -> torch.Tensor:
        return torch.cat([self._sh_coefficients_0, self._sh_coefficients_rest], dim=1)

    @property
    def sh_coefficients_0(self) -> torch.Tensor:
        return self._sh_coefficients_0

    @property
    def sh_coefficients_rest(self) -> torch.Tensor:
        return self._sh_coefficients_rest

    @property
    def densification_info(self) -> torch.Tensor:
        return self._densification_info

    @property
    def covariances(self) -> torch.Tensor:
        R = quaternion_to_rotation_matrix(self.rotations, normalize=False)
        S = torch.diag_embed(self.scales)
        RS = R @ S
        return RS @ RS.transpose(-2, -1)

    # ── SH degree ─────────────────────────────────────────────────────────────

    def increase_used_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, use_mcmc: bool = False) -> None:
        means = point_cloud.positions.cuda()
        n = means.shape[0]
        Logger.log_info(f'number of Gaussians at initialization: {n:,}')

        rgbs = torch.full_like(means, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        sh_coefficients_0 = ((rgbs - 0.5) / 0.28209479177387814)[:, None, :]
        sh_coefficients_rest = torch.zeros((n, (self.max_sh_degree + 1) ** 2 - 1, 3),
                                           dtype=torch.float32, device='cuda')

        distances = compute_root_mean_squared_knn_distances(means)
        scales = distances.log()[..., None].repeat(1, 3)

        rotations = torch.zeros((n, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0

        initial_opacity_logit = math.log(0.1 / 0.9)
        opacities = torch.full((n, 1), fill_value=initial_opacity_logit,
                               dtype=torch.float32, device='cuda')

        self._means = torch.nn.Parameter(means.contiguous())
        self._sh_coefficients_0 = torch.nn.Parameter(sh_coefficients_0.contiguous())
        self._sh_coefficients_rest = torch.nn.Parameter(sh_coefficients_rest.contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())

    # ── Training setup ────────────────────────────────────────────────────────

    def training_setup(self, training_wrapper, training_cameras_extent: float) -> None:
        self.percent_dense = training_wrapper.DENSIFICATION_PERCENT_DENSE
        self.training_cameras_extent = training_cameras_extent

        main_param_groups = [
            {'params': [self._means],
             'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * training_cameras_extent,
             'name': 'means'},
            {'params': [self._sh_coefficients_0],
             'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SH_COEFFICIENTS_0,
             'name': 'sh_coefficients_0'},
            {'params': [self._opacities],
             'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_OPACITIES,
             'name': 'opacities'},
            {'params': [self._scales],
             'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SCALES,
             'name': 'scales'},
            {'params': [self._rotations],
             'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_ROTATIONS,
             'name': 'rotations'},
        ]
        sh_param_groups = [
            {'params': [self._sh_coefficients_rest],
             'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SH_COEFFICIENTS_REST,
             'name': 'sh_coefficients_rest'},
        ]

        self.optimizer = torch.optim.Adam(main_param_groups, lr=0.0, eps=1e-15)
        self.sh_optimizer = torch.optim.Adam(sh_param_groups, lr=0.0, eps=1e-15)

        self.lr_means_scheduler = LRDecayPolicy(
            lr_init=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * training_cameras_extent,
            lr_final=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_FINAL * training_cameras_extent,
            max_steps=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_MAX_STEPS,
        )

    def update_learning_rate(self, iteration: int) -> None:
        self.lr_means = self.lr_means_scheduler(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'means':
                param_group['lr'] = self.lr_means

    # ── Opacity / densification state ─────────────────────────────────────────

    def reset_opacities(self) -> None:
        opacities_new = self._opacities.clamp_max(-4.595119953155518)  # sigmoid = 0.01
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def reset_densification_info(self) -> None:
        self._densification_info = torch.zeros(
            (2, self._means.shape[0]), dtype=torch.float32, device='cuda')

    # ── Parameter management (prune / sort) ───────────────────────────────────

    def prune(self, prune_mask: torch.Tensor) -> None:
        valid_mask = ~prune_mask
        main = prune_param_groups(self.optimizer, valid_mask)
        sh = prune_param_groups(self.sh_optimizer, valid_mask)
        self._means = main['means']
        self._sh_coefficients_0 = main['sh_coefficients_0']
        self._opacities = main['opacities']
        self._scales = main['scales']
        self._rotations = main['rotations']
        self._sh_coefficients_rest = sh['sh_coefficients_rest']
        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, valid_mask].contiguous()

    def sort(self, ordering: torch.Tensor) -> None:
        main = sort_param_groups(self.optimizer, ordering)
        sh = sort_param_groups(self.sh_optimizer, ordering)
        self._means = main['means']
        self._sh_coefficients_0 = main['sh_coefficients_0']
        self._opacities = main['opacities']
        self._scales = main['scales']
        self._rotations = main['rotations']
        self._sh_coefficients_rest = sh['sh_coefficients_rest']
        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, ordering].contiguous()

    def apply_morton_ordering(self) -> None:
        morton_encoding = morton_encode(self._means.data)
        order = torch.argsort(morton_encoding)
        self.sort(order)

    # ── FastGS densification / pruning ────────────────────────────────────────

    def densify_and_prune_fastgs(
        self,
        importance_score: torch.Tensor,
        pruning_score: torch.Tensor,
        min_opacity: float,
        extent: float,
        grad_threshold: float,
        percent_dense: float,
    ) -> None:
        """VCD + VCP: multi-view consistent densification and pruning (FastGS core)."""
        # Gradient-based qualification (same formula as vanilla ADC)
        densification_mask = (
            self._densification_info[1]
            >= grad_threshold * self._densification_info[0].clamp_min(1.0)
        )
        is_small = self._scales.max(dim=1).values <= math.log(percent_dense * extent)
        clone_mask = densification_mask & is_small
        split_mask = densification_mask & ~is_small

        # FastGS VCD gate: only densify Gaussians in multi-view high-error regions
        metric_mask = importance_score > 5
        clone_mask = clone_mask & metric_mask
        split_mask = split_mask & metric_mask

        # ── Clone ──────────────────────────────────────────────────────────────
        n_clones = clone_mask.sum().item()
        cloned_means = self._means[clone_mask]
        cloned_sh0 = self._sh_coefficients_0[clone_mask]
        cloned_shr = self._sh_coefficients_rest[clone_mask]
        cloned_opacities = self._opacities[clone_mask]
        cloned_scales = self._scales[clone_mask]
        cloned_rotations = self._rotations[clone_mask]

        # ── Split ──────────────────────────────────────────────────────────────
        n_splits_src = split_mask.sum().item()
        split_scales = self._scales[split_mask].exp().expand(2, -1, -1).flatten(end_dim=1)
        split_rotations = self._rotations[split_mask].expand(2, -1, -1).flatten(end_dim=1)
        offsets = (
            quaternion_to_rotation_matrix(split_rotations)
            @ (split_scales * torch.randn_like(split_scales))[..., None]
        )[..., 0]
        split_means = self._means[split_mask].expand(2, -1, -1).flatten(end_dim=1) + offsets
        split_scales = split_scales.mul(0.625).log()  # scale /= 1.6  (= 0.8 * 2)
        split_sh0 = self._sh_coefficients_0[split_mask].expand(2, -1, -1, -1).flatten(end_dim=1)
        split_shr = self._sh_coefficients_rest[split_mask].expand(2, -1, -1, -1).flatten(end_dim=1)
        split_opacities = self._opacities[split_mask].expand(2, -1, -1).flatten(end_dim=1)
        split_rotations_new = self._rotations[split_mask].expand(2, -1, -1).flatten(end_dim=1)

        n_new = n_clones + 2 * n_splits_src

        # ── Incorporate via extend on BOTH optimizers ─────────────────────────
        main_new = {
            'means':             torch.cat([cloned_means, split_means]),
            'sh_coefficients_0': torch.cat([cloned_sh0, split_sh0]),
            'opacities':         torch.cat([cloned_opacities, split_opacities]),
            'scales':            torch.cat([cloned_scales, split_scales]),
            'rotations':         torch.cat([cloned_rotations, split_rotations_new]),
        }
        sh_new = {
            'sh_coefficients_rest': torch.cat([cloned_shr, split_shr]),
        }
        main_params = extend_param_groups(self.optimizer, main_new)
        sh_params = extend_param_groups(self.sh_optimizer, sh_new)
        self._means = main_params['means']
        self._sh_coefficients_0 = main_params['sh_coefficients_0']
        self._opacities = main_params['opacities']
        self._scales = main_params['scales']
        self._rotations = main_params['rotations']
        self._sh_coefficients_rest = sh_params['sh_coefficients_rest']
        self._densification_info = None  # invalidated after size change

        # ── Prune: originals that were split + low opacity ────────────────────
        prune_mask = torch.cat([
            split_mask,
            torch.zeros(n_new, dtype=torch.bool, device='cuda'),
        ])
        prune_mask |= self._opacities.flatten() < math.log(min_opacity / (1.0 - min_opacity))
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        self.prune(prune_mask)

        # ── Clamp opacities to logit(0.8) max (FastGS post-densification) ─────
        opacities_clamped = self._opacities.clamp_max(math.log(0.8 / 0.2))
        replace_param_group_data(self.optimizer, opacities_clamped, 'opacities')

    def final_prune_fastgs(self, min_opacity: float, pruning_score: torch.Tensor) -> None:
        """VCP final-stage pruning."""
        prune_mask = self._opacities.flatten().sigmoid() < min_opacity
        prune_mask |= pruning_score > 0.9
        self.prune(prune_mask)

    # ── Post-training cleanup ─────────────────────────────────────────────────

    @torch.no_grad()
    def training_cleanup(self, min_opacity: float) -> int:
        self._densification_info = None
        prune_mask = self.opacities.flatten() < min_opacity
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        self.prune(prune_mask)
        self.apply_morton_ordering()
        self.optimizer.zero_grad()
        self.optimizer = None
        self.sh_optimizer.zero_grad()
        self.sh_optimizer = None
        return self.means.shape[0]

    # ── PLY export ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def as_ply_dict(self) -> dict[str, np.ndarray]:
        if self.means.shape[0] == 0:
            return {}
        means = self.means.detach().contiguous().cpu().numpy()
        sh_0 = self.sh_coefficients_0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self.sh_coefficients_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.raw_opacities.detach().contiguous().cpu().numpy()
        scales = self.raw_scales.detach().contiguous().cpu().numpy()
        rotations = self.rotations.detach().contiguous().cpu().numpy()
        attributes = np.concatenate((means, sh_0, sh_rest, opacities, scales, rotations), axis=1)
        attribute_names = (
              ['x', 'y', 'z']
            + ['f_dc_0', 'f_dc_1', 'f_dc_2']
            + [f'f_rest_{i}' for i in range(sh_rest.shape[-1])]
            + ['opacity']
            + ['scale_0', 'scale_1', 'scale_2']
            + ['rot_0', 'rot_1', 'rot_2', 'rot_3']
        )
        full_dtype = [(name, 'f4') for name in attribute_names]
        vertices = np.empty(means.shape[0], dtype=full_dtype)
        vertices[:] = list(map(tuple, attributes))
        return {'vertex': vertices}


@Framework.Configurable.configure(SH_DEGREE=3)
class FastGSModel(BaseModel):
    """Defines the FastGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None

    def build(self) -> 'FastGSModel':
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(self.SH_DEGREE, pretrained)
        return self

    def get_ply_dict(self) -> dict[str, np.ndarray | list[str]]:
        data: dict[str, np.ndarray | list[str]] = {}
        if self.gaussians is None or not (data := self.gaussians.as_ply_dict()):
            return data
        data['comments'] = ['SplatRenderMode: default', 'Generated with NeRFICG/FastGS']
        return data
