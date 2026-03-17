from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        inputs = inputs.float()
        targets = targets.float()
        dims = (1, 2, 3)
        intersection = (inputs * targets).sum(dim=dims)
        denom = inputs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).float()
        targets = targets.float()
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma).mean()


class NBMPolypLossV2(nn.Module):
    def __init__(
        self,
        main_bce_weight: float = 0.5,
        main_ft_weight: float = 0.5,
        aux_weight: float = 0.3,
        edge_weight: float = 0.2,
        consistency_weight: float = 0.1,
        coarse_weight: float = 0.2,
        smooth: float = 1.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
    ):
        super().__init__()
        self.main_bce_weight = main_bce_weight
        self.main_ft_weight = main_ft_weight
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.consistency_weight = consistency_weight
        self.coarse_weight = coarse_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)

    def _mask_to_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _main_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        bce = self.bce(logits, targets)
        ft = self.ft(logits, targets)
        loss = self.main_bce_weight * bce + self.main_ft_weight * ft
        return loss, {"bce": float(bce.detach().item()), "focal_tversky": float(ft.detach().item())}

    def _aux_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

    def _edge_loss(self, edge_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        edge_targets = self._mask_to_boundary(targets)
        return 0.5 * self.bce(edge_logits, edge_targets) + 0.5 * self.dice(edge_logits, edge_targets)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, return_components: bool = False):
        main_loss, main_stats = self._main_loss(outputs["logits"], targets)
        coarse_loss, _ = self._main_loss(outputs["coarse_logits"], targets)
        aux_loss = self._aux_loss(outputs["aux_logits"], targets)
        edge_loss = self._edge_loss(outputs["edge_logits"], targets)

        prob_main = torch.sigmoid(outputs["logits"])
        prob_coarse = torch.sigmoid(outputs["coarse_logits"])
        consistency = F.l1_loss(prob_main, prob_coarse)

        total = (
            main_loss
            + self.coarse_weight * coarse_loss
            + self.aux_weight * aux_loss
            + self.edge_weight * edge_loss
            + self.consistency_weight * consistency
        )

        if not return_components:
            return total

        return total, {
            "loss_total": float(total.detach().item()),
            "loss_main": float(main_loss.detach().item()),
            "loss_coarse": float(coarse_loss.detach().item()),
            "loss_aux": float(aux_loss.detach().item()),
            "loss_edge": float(edge_loss.detach().item()),
            "loss_consistency": float(consistency.detach().item()),
            **main_stats,
        }
