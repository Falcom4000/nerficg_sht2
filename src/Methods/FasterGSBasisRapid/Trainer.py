"""FasterGSBasisRapid/Trainer.py"""

import random

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGSBasisRapid.Loss import FasterGSBasisRapidLoss
from Methods.FasterGSBasisRapid.utils import enable_expandable_segments
from Optim.Losses.DSSIM import fused_dssim
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,  # while official code states 500, densification actually starts at 600 there
    DENSIFICATION_END_ITERATION=14_900,  # while official code states 15000, densification actually stops at 14900 there
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,
    DENSIFICATION_ABS_GRAD_THRESHOLD=0.0012,
    DENSIFICATION_PERCENT_DENSE=0.001,
    OPACITY_RESET_INTERVAL=3_000,
    EXTRA_OPACITY_RESET_ITERATION=500,
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    FASTGS_SCORE_VIEWS=10,
    FASTGS_LOSS_THRESHOLD=0.1,
    FASTGS_IMPORTANCE_THRESHOLD=5.0,
    FASTGS_PRUNING_START_ITERATION=18_000,
    FASTGS_PRUNING_END_ITERATION=27_000,
    FASTGS_PRUNING_INTERVAL=3_000,
    FASTGS_PRUNING_MIN_OPACITY=0.1,
    FASTGS_PRUNING_SCORE_THRESHOLD=0.9,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,  # weight for the per-pixel L1 loss on the rgb image
        LAMBDA_DSSIM=0.2,  # weight for the DSSIM loss on the rgb image
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.00025,  # RapidGS highfeature_lr / 20 = 0.005 / 20
        LEARNING_RATE_OPACITIES=0.025,  # recently updated in official code; used to be 0.05
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
        SH_REST_UPDATE_INTERVAL=16,
        LATE_UPDATE_START_ITERATION=15_000,
        LATE_UPDATE_INTERVAL=32,
        FINAL_UPDATE_START_ITERATION=20_000,
        FINAL_UPDATE_INTERVAL=64,
    ),
)
class FasterGSBasisRapidTrainer(GuiTrainer):
    """Defines the trainer for the FasterGSBasisRapid variant."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info('using "expandable_segments:True" with the torch cuda memory allocator')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = FasterGSBasisRapidLoss(loss_config=self.LOSS)

    @staticmethod
    def _compose_gt(view, bg_color: torch.Tensor) -> torch.Tensor:
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        return rgb_gt

    def _sample_score_views(self, dataset: 'BaseDataset') -> list:
        dataset.train()
        views = list(dataset)
        n_views = min(self.FASTGS_SCORE_VIEWS, len(views))
        return random.sample(views, n_views) if n_views < len(views) else views

    @torch.no_grad()
    def compute_fastgs_scores(self, dataset: 'BaseDataset', compute_importance: bool) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Computes FastGS multi-view densification and pruning scores."""
        full_metric_counts = None
        full_metric_score = None
        score_views = self._sample_score_views(dataset)

        for view in score_views:
            bg_color = view.camera.background_color
            image = self.renderer.render_image_training(view=view, update_densification_info=False)
            rgb_gt = self._compose_gt(view, bg_color)

            pixel_error = torch.mean(torch.abs(image - rgb_gt), dim=0)
            error_min, error_max = pixel_error.min(), pixel_error.max()
            error_range = error_max - error_min
            if error_range > 1e-12:
                loss_map = (pixel_error - error_min) / error_range
            else:
                loss_map = torch.zeros_like(pixel_error)
            metric_map = loss_map > self.FASTGS_LOSS_THRESHOLD

            _, metric_counts = self.renderer.render_image_metric_counts(view=view, metric_map=metric_map, bg_color=bg_color)
            metric_counts = metric_counts.to(dtype=torch.float32)

            if compute_importance:
                full_metric_counts = metric_counts.clone() if full_metric_counts is None else full_metric_counts + metric_counts

            photometric_loss = (
                self.LOSS.LAMBDA_L1 * torch.nn.functional.l1_loss(image, rgb_gt)
                + self.LOSS.LAMBDA_DSSIM * fused_dssim(image, rgb_gt)
            )
            full_metric_score = (
                photometric_loss * metric_counts
                if full_metric_score is None else full_metric_score + photometric_loss * metric_counts
            )

        pruning_min, pruning_max = full_metric_score.min(), full_metric_score.max()
        pruning_range = pruning_max - pruning_min
        if pruning_range > 1e-12:
            pruning_score = (full_metric_score - pruning_min) / pruning_range
        else:
            pruning_score = torch.zeros_like(full_metric_score)

        importance_score = None
        if compute_importance:
            importance_score = torch.div(full_metric_counts, len(score_views), rounding_mode='floor')
        return importance_score, pruning_score

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.log_info(f'training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, size = dataset.bounding_box.min, dataset.bounding_box.size
            positions = torch.rand((n_random_points, 3), dtype=torch.float32, device=size.device) * size + min_bounds
            point_cloud = BasicPointCloud(positions)
        self.model.gaussians.initialize_from_point_cloud(point_cloud)
        self.model.gaussians.training_setup(self, radius)
        self.model.gaussians.reset_densification_info()

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        importance_score, pruning_score = self.compute_fastgs_scores(dataset, compute_importance=True)
        self.model.gaussians.adaptive_density_control(
            self.DENSIFICATION_GRAD_THRESHOLD,
            self.DENSIFICATION_ABS_GRAD_THRESHOLD,
            0.005,
            iteration > self.OPACITY_RESET_INTERVAL,
            importance_score=importance_score,
            importance_threshold=self.FASTGS_IMPORTANCE_THRESHOLD,
            pruning_score=pruning_score,
        )
        if iteration < self.DENSIFICATION_END_ITERATION:
            self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=100, start_iteration='FASTGS_PRUNING_START_ITERATION', end_iteration='FASTGS_PRUNING_END_ITERATION', iteration_stride='FASTGS_PRUNING_INTERVAL')
    @torch.no_grad()
    def prune_multiview_inconsistent(self, _: int, dataset: 'BaseDataset') -> None:
        """Apply FastGS multi-view consistent pruning after densification."""
        _, pruning_score = self.compute_fastgs_scores(dataset, compute_importance=False)
        self.model.gaussians.prune_by_multiview_score(
            min_opacity=self.FASTGS_PRUNING_MIN_OPACITY,
            pruning_score=pruning_score,
            score_threshold=self.FASTGS_PRUNING_SCORE_THRESHOLD,
        )
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=90, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        self.model.gaussians.reset_opacities()

    @training_callback(priority=90, start_iteration='EXTRA_OPACITY_RESET_ITERATION', end_iteration='EXTRA_OPACITY_RESET_ITERATION')
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        # render
        image = self.renderer.render_image_training(
            view=view,
            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,
        )
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)
        loss = self.loss(image, rgb_gt)
        # backward
        loss.backward()
        # optimizer step
        self.model.gaussians.optimizer_step(
            optimization_step=iteration + 1,
            total_iterations=self.NUM_ITERATIONS,
            sh_rest_update_interval=self.OPTIMIZER.SH_REST_UPDATE_INTERVAL,
            late_update_start_iteration=self.OPTIMIZER.LATE_UPDATE_START_ITERATION,
            late_update_interval=self.OPTIMIZER.LATE_UPDATE_INTERVAL,
            final_update_start_iteration=self.OPTIMIZER.FINAL_UPDATE_START_ITERATION,
            final_update_interval=self.OPTIMIZER.FINAL_UPDATE_INTERVAL,
        )

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds Gaussian count to default Weights & Biases logging."""
        Framework.wandb.log({
            '#Gaussians': self.model.gaussians.means.shape[0]
        }, step=iteration)
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *_) -> None:
        """Clean up after training."""
        n_gaussians = self.model.gaussians.training_cleanup(min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as n_gaussians_file:
            n_gaussians_file.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )
