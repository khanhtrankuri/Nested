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


class NBMPolypLoss(nn.Module):
    def __init__(
        self,
        main_weight: float = 1.0,
        aux_weight: float = 0.4,
        edge_weight: float = 0.2,
        consistency_weight: float = 0.1,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.consistency_weight = consistency_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def _mask_to_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _branch_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        total = bce + dice
        return total, bce, dice

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, return_components: bool = False):
        targets = targets.float()
        boundary_targets = self._mask_to_boundary(targets)

        main_total, main_bce, main_dice = self._branch_loss(outputs["logits"], targets)
        aux_total, aux_bce, aux_dice = self._branch_loss(outputs["aux_logits"], targets)
        edge_total, edge_bce, edge_dice = self._branch_loss(outputs["edge_logits"], boundary_targets)

        consistency = F.l1_loss(
            torch.sigmoid(outputs["logits"]),
            torch.sigmoid(outputs["aux_logits"]),
        )

        total = (
            self.main_weight * main_total
            + self.aux_weight * aux_total
            + self.edge_weight * edge_total
            + self.consistency_weight * consistency
        )

        if return_components:
            return total, {
                "loss_total": float(total.detach().item()),
                "loss_main": float(main_total.detach().item()),
                "loss_aux": float(aux_total.detach().item()),
                "loss_edge": float(edge_total.detach().item()),
                "loss_consistency": float(consistency.detach().item()),
                "loss_main_bce": float(main_bce.detach().item()),
                "loss_main_dice": float(main_dice.detach().item()),
                "loss_aux_bce": float(aux_bce.detach().item()),
                "loss_aux_dice": float(aux_dice.detach().item()),
                "loss_edge_bce": float(edge_bce.detach().item()),
                "loss_edge_dice": float(edge_dice.detach().item()),
            }

        return total
