"""FasterGSBasisRapid/Loss.py"""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim


class FasterGSBasisRapidLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList) -> None:
        super().__init__()
        self.lambda_l1 = loss_config.LAMBDA_L1
        self.lambda_dssim = loss_config.LAMBDA_DSSIM
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.activate_logging:
            return self.lambda_l1 * torch.nn.functional.l1_loss(input, target) + self.lambda_dssim * fused_dssim(input, target)
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })
