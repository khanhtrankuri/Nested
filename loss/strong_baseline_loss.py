from typing import Dict, Tuple

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow((1.0 - tversky), self.gamma).mean()


class LovaszHingeLoss(nn.Module):
    """Binary Lovasz hinge.
    Adapted to keep the baseline self-contained.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        return scores, labels

    def lovasz_hinge_flat(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.numel() == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        return torch.dot(torch.relu(errors_sorted), grad)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for logit, target in zip(logits, targets):
            logit_flat, target_flat = self.flatten_binary_scores(logit, target)
            losses.append(self.lovasz_hinge_flat(logit_flat, target_flat))
        return torch.stack(losses).mean()


class StrongBaselineLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.20,
        lovasz_weight: float = 0.30,
        focal_tversky_weight: float = 0.25,
        dice_weight: float = 0.15,
        aux_weight: float = 0.10,
        coarse_weight: float = 0.08,
        trust_weight: float = 0.04,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5)
        self.dice = SoftDiceLoss(smooth=1.0)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.coarse_weight = coarse_weight
        self.trust_weight = trust_weight

    def forward(self, outputs, targets: torch.Tensor, return_components: bool = False):
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            aux_logits = outputs.get("aux_logits")
            coarse_logits = outputs.get("coarse_logits")
        else:
            logits = outputs
            aux_logits = None
            coarse_logits = None

        loss_bce = self.bce(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)
        loss_ft = self.focal_tversky(logits, targets)
        loss_dice = self.dice(logits, targets)
        _zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if aux_logits is not None:
            loss_aux = 0.5 * self.bce(aux_logits, targets) + 0.5 * self.dice(aux_logits, targets)
        else:
            loss_aux = _zero
        if coarse_logits is not None:
            loss_coarse = 0.5 * self.bce(coarse_logits, targets) + 0.5 * self.dice(coarse_logits, targets)
            with torch.no_grad():
                coarse_probs = torch.sigmoid(coarse_logits)
                coarse_uncertainty = 1.0 - torch.abs(2.0 * coarse_probs - 1.0)
                confidence = 1.0 - coarse_uncertainty
            refined_probs = torch.sigmoid(logits)
            trust_denom = confidence.sum().clamp(min=1e-6)
            loss_trust = (torch.abs(refined_probs - coarse_probs.detach()) * confidence).sum() / trust_denom
        else:
            loss_coarse = _zero
            loss_trust = _zero

        total = (
            self.bce_weight * loss_bce
            + self.lovasz_weight * loss_lovasz
            + self.focal_tversky_weight * loss_ft
            + self.dice_weight * loss_dice
            + self.aux_weight * loss_aux
            + self.coarse_weight * loss_coarse
            + self.trust_weight * loss_trust
        )
        if not return_components:
            return total
        return total, {
            "loss_total": float(total.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_lovasz": float(loss_lovasz.detach().item()),
            "loss_focal_tversky": float(loss_ft.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_aux": float(loss_aux.detach().item()),
            "loss_coarse": float(loss_coarse.detach().item()),
            "loss_trust": float(loss_trust.detach().item()),
        }
