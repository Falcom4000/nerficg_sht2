"""FastGS/Trainer.py"""

import math
import random

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FastGS.Loss import FastGSLoss
from Methods.FasterGS.utils import enable_expandable_segments, carve
from Optim.Losses.DSSIM import fused_dssim
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=500,
    DENSIFICATION_END_ITERATION=14_900,
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,
    DENSIFICATION_PERCENT_DENSE=0.001,
    OPACITY_RESET_INTERVAL=3_000,
    EXTRA_OPACITY_RESET_ITERATION=500,
    FINAL_PRUNING_INTERVAL=3_000,
    FINAL_PRUNING_START_ITERATION=15_000,
    FINAL_PRUNING_END_ITERATION=30_000,
    MORTON_ORDERING_INTERVAL=5000,
    MORTON_ORDERING_END_ITERATION=15000,
    MULTIVIEW_CAMERAS=10,
    LOSS_THRESH=0.1,
    USE_RANDOM_BACKGROUND_COLOR=False,
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    RANDOM_INITIALIZATION=Framework.ConfigParameterList(
        FORCE=False,
        N_POINTS=100_000,
        ENABLE_CARVING=True,
        CARVING_IN_ALL_FRUSTUMS=False,
        CARVING_ENFORCE_ALPHA=False,
    ),
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.00025,  # 0.005 / 20
        LEARNING_RATE_OPACITIES=0.025,
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
    ),
)
class FastGSTrainer(GuiTrainer):
    """Trainer for FastGS: multi-view consistent densification/pruning + sparse SH optimizer."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info('using "expandable_segments:True" with the torch cuda memory allocator')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = FastGSLoss(loss_config=self.LOSS)

    # ── Pre-training ──────────────────────────────────────────────────────────

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (1.1 * torch.max(
            torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1)
        )).item()
        Logger.log_info(f'training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None and not self.RANDOM_INITIALIZATION.FORCE:
            point_cloud = dataset.point_cloud
        else:
            samples = torch.rand(
                (self.RANDOM_INITIALIZATION.N_POINTS, 3),
                dtype=torch.float32,
                device=Framework.config.GLOBAL.DEFAULT_DEVICE,
            )
            positions = samples * dataset.bounding_box.size + dataset.bounding_box.min
            if self.RANDOM_INITIALIZATION.ENABLE_CARVING:
                positions = carve(
                    positions, dataset,
                    self.RANDOM_INITIALIZATION.CARVING_IN_ALL_FRUSTUMS,
                    self.RANDOM_INITIALIZATION.CARVING_ENFORCE_ALPHA,
                )
            point_cloud = BasicPointCloud(positions)

        self.model.gaussians.initialize_from_point_cloud(point_cloud, use_mcmc=False)
        self.model.gaussians.training_setup(self, radius)
        self.model.gaussians.reset_densification_info()

    # ── Training callbacks ────────────────────────────────────────────────────

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(
        priority=100,
        start_iteration='DENSIFICATION_START_ITERATION',
        end_iteration='DENSIFICATION_END_ITERATION',
        iteration_stride='DENSIFICATION_INTERVAL',
    )
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        importance_score, pruning_score = self._compute_multiview_scores(
            dataset, compute_importance=True)
        self.model.gaussians.densify_and_prune_fastgs(
            importance_score=importance_score,
            pruning_score=pruning_score,
            min_opacity=0.005,
            extent=self.model.gaussians.training_cameras_extent,
            grad_threshold=self.DENSIFICATION_GRAD_THRESHOLD,
            percent_dense=self.DENSIFICATION_PERCENT_DENSE,
        )
        self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(
        priority=99,
        end_iteration='MORTON_ORDERING_END_ITERATION',
        iteration_stride='MORTON_ORDERING_INTERVAL',
    )
    @torch.no_grad()
    def morton_ordering(self, *_) -> None:
        self.model.gaussians.apply_morton_ordering()

    @training_callback(
        priority=90,
        start_iteration='OPACITY_RESET_INTERVAL',
        end_iteration='DENSIFICATION_END_ITERATION',
        iteration_stride='OPACITY_RESET_INTERVAL',
    )
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        self.model.gaussians.reset_opacities()

    @training_callback(
        priority=90,
        start_iteration='EXTRA_OPACITY_RESET_ITERATION',
        end_iteration='EXTRA_OPACITY_RESET_ITERATION',
    )
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        if dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()

    @training_callback(
        priority=85,
        start_iteration='FINAL_PRUNING_START_ITERATION',
        end_iteration='FINAL_PRUNING_END_ITERATION',
        iteration_stride='FINAL_PRUNING_INTERVAL',
    )
    @torch.no_grad()
    def final_pruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        _, pruning_score = self._compute_multiview_scores(dataset, compute_importance=False)
        self.model.gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        self.model.train()
        dataset.train()
        self.loss.train()

        self.model.gaussians.update_learning_rate(iteration + 1)

        view = self.train_sampler.get(dataset=dataset)['view']
        bg_color = (
            torch.rand_like(view.camera.background_color)
            if self.USE_RANDOM_BACKGROUND_COLOR
            else view.camera.background_color
        )

        image = self.renderer.render_image_training(
            view=view,
            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
        )

        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        loss = self.loss(image, rgb_gt)
        loss.backward()

        # Sparse optimizer schedule (FastGS key contribution)
        optimization_step = iteration + 1
        if optimization_step <= 15_000:
            self.model.gaussians.optimizer.step()
            self.model.gaussians.optimizer.zero_grad(set_to_none=True)
            if optimization_step % 16 == 0:
                self.model.gaussians.sh_optimizer.step()
                self.model.gaussians.sh_optimizer.zero_grad(set_to_none=True)
        elif optimization_step <= 20_000:
            if optimization_step % 32 == 0:
                self.model.gaussians.optimizer.step()
                self.model.gaussians.optimizer.zero_grad(set_to_none=True)
                self.model.gaussians.sh_optimizer.step()
                self.model.gaussians.sh_optimizer.zero_grad(set_to_none=True)
        else:
            if optimization_step % 64 == 0:
                self.model.gaussians.optimizer.step()
                self.model.gaussians.optimizer.zero_grad(set_to_none=True)
                self.model.gaussians.sh_optimizer.step()
                self.model.gaussians.sh_optimizer.zero_grad(set_to_none=True)

    # ── Post-training ─────────────────────────────────────────────────────────

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *_) -> None:
        n_gaussians = self.model.gaussians.training_cleanup(
            min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as f:
            f.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )

    # ── Multi-view scoring (VCD / VCP) ────────────────────────────────────────

    @torch.no_grad()
    def _compute_multiview_scores(
        self,
        dataset: 'BaseDataset',
        compute_importance: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute per-Gaussian importance and pruning scores over K random views.

        Returns (importance_score, pruning_score) tensors of shape (N,).
        importance_score is None when compute_importance=False.
        """
        dataset.train()
        n = self.model.gaussians.means.shape[0]
        bg_color = dataset.default_camera.background_color

        all_views = list(dataset)
        k = min(self.MULTIVIEW_CAMERAS, len(all_views))
        sampled_views = random.sample(all_views, k)

        full_metric_counts = torch.zeros(n, dtype=torch.float32, device='cuda')
        full_metric_score = torch.zeros(n, dtype=torch.float32, device='cuda')

        for view in sampled_views:
            # ── First pass: render to get normalized L1 loss map ──────────────
            image = self.renderer.render_image_training(
                view=view,
                update_densification_info=False,
                bg_color=bg_color,
            )
            rgb_gt = view.rgb
            if (alpha_gt := view.alpha) is not None:
                rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)

            # Photometric loss (scalar) — matches FastGS compute_photometric_loss
            photometric_loss = (
                (1.0 - self.LOSS.LAMBDA_DSSIM) * torch.nn.functional.l1_loss(image, rgb_gt)
                + self.LOSS.LAMBDA_DSSIM * (1.0 - fused_dssim(
                    image.unsqueeze(0), rgb_gt.unsqueeze(0)))
            ).item()

            # Normalized per-pixel L1 map → binary metric_map (FastGS get_loss + threshold)
            l1_map = image.sub(rgb_gt).abs().mean(dim=0)  # (H, W)
            l1_min, l1_max = l1_map.min(), l1_map.max()
            l1_norm = (l1_map - l1_min) / (l1_max - l1_min + 1e-8)
            metric_map = (l1_norm > self.LOSS_THRESH).int().flatten().contiguous()  # (H*W,) int32

            # ── Second pass: precise per-Gaussian pixel counts via CUDA kernel ─
            view_metric_counts = torch.zeros(n, dtype=torch.float32, device='cuda')
            self.renderer.render_for_scoring(view, bg_color, metric_map, view_metric_counts)

            if compute_importance:
                full_metric_counts += view_metric_counts
            full_metric_score += photometric_loss * view_metric_counts

        score_min = full_metric_score.min()
        score_max = full_metric_score.max()
        pruning_score = (full_metric_score - score_min) / (score_max - score_min + 1e-8)

        importance_score = None
        if compute_importance:
            importance_score = torch.div(full_metric_counts, k, rounding_mode='floor')

        return importance_score, pruning_score
