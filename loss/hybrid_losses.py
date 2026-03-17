from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def derive_edge_targets(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).float().flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).float().flatten(1)
        targets = targets.float().flatten(1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma).mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, edge_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        edge_targets = derive_edge_targets(targets.float())
        return self.bce(edge_logits, edge_targets)


class HybridPolypLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 0.2,
        edge_weight: float = 0.1,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.consistency_weight = consistency_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.focal_tversky = FocalTverskyLoss()
        self.edge_loss = EdgeLoss()

    def _main_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        bce = self.bce(logits, targets)
        ft = self.focal_tversky(logits, targets)
        dice = self.dice(logits, targets)
        total = 0.4 * bce + 0.4 * ft + 0.2 * dice
        return total, {
            "bce": float(bce.detach().item()),
            "focal_tversky": float(ft.detach().item()),
            "soft_dice": float(dice.detach().item()),
        }

    def _aux_loss(self, aux_logits: Sequence[torch.Tensor], coarse_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        aux_terms: List[torch.Tensor] = []
        aux_terms.append(0.5 * self.bce(coarse_logits, targets) + 0.5 * self.dice(coarse_logits, targets))
        for aux in aux_logits:
            aux_terms.append(0.5 * self.bce(aux, targets) + 0.5 * self.dice(aux, targets))
        return torch.stack(aux_terms).mean() if aux_terms else coarse_logits.new_zeros(())

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, return_components: bool = False):
        aux_logits = outputs.get("aux_logits", [])
        if isinstance(aux_logits, torch.Tensor):
            aux_logits = [aux_logits]
        if not isinstance(aux_logits, (list, tuple)):
            raise TypeError("aux_logits must be a tensor, list, or tuple.")

        main_loss, main_stats = self._main_loss(outputs["logits"], targets)
        aux_loss = self._aux_loss(aux_logits, outputs["coarse_logits"], targets)
        edge_loss = self.edge_loss(outputs["edge_logits"], targets)
        consistency = F.l1_loss(torch.sigmoid(outputs["logits"]), torch.sigmoid(outputs["coarse_logits"]))

        total = main_loss + self.aux_weight * aux_loss + self.edge_weight * edge_loss + self.consistency_weight * consistency
        if not return_components:
            return total

        return total, {
            "loss_total": float(total.detach().item()),
            "loss_main": float(main_loss.detach().item()),
            "loss_aux": float(aux_loss.detach().item()),
            "loss_edge": float(edge_loss.detach().item()),
            "loss_consistency": float(consistency.detach().item()),
            **main_stats,
        }
