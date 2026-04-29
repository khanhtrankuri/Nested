"""
Boundary-Sensitive Losses for Medical Segmentation.

Provides:
1. SoftDiceBoundaryLoss — Dice loss with edge weighting
2. FocalBoundaryLoss — Focal loss focusing on boundary pixels
3. HausdorffDistanceLoss — Soft approximation of Hausdorff distance (optional, expensive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_edges(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Detect edges using Laplacian filter.

    Args:
        mask: (B, 1, H, W) binary or float mask

    Returns:
        edge: (B, 1, H, W) edge map (values > 0 at edges)
    """
    # Laplacian kernel
    if kernel_size == 3:
        kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                [1.0, -4.0, 1.0],
                                [0.0, 1.0, 0.0]]]], device=mask.device)
    else:
        raise ValueError("Only kernel_size=3 supported")
    return F.conv2d(mask, kernel, padding=1)


class SoftDiceBoundaryLoss(nn.Module):
    """
    Dice loss with boundary emphasis.

    L = 1 - (2 * (pred * (target + edge_weight)) + smooth) / (|pred| + |target + edge_weight|)

    Or separate: L_dice + lambda_edge * L_dice_on_edge_only
    """

    def __init__(self, smooth: float = 1e-5, boundary_weight: float = 2.0):
        super().__init__()
        self.smooth = smooth
        self.boundary_weight = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) raw logits (will apply sigmoid)
            target: (B, 1, H, W) binary {0, 1}

        Returns:
            loss: scalar
        """
        pred_prob = torch.sigmoid(pred)

        # Standard Dice on all pixels
        intersection_all = (pred_prob * target).sum(dim=[1, 2, 3])
        pred_sum = pred_prob.sum(dim=[1, 2, 3])
        target_sum = target.sum(dim=[1, 2, 3])
        dice_all = (2 * intersection_all + self.smooth) / (pred_sum + target_sum + self.smooth)
        loss_all = 1 - dice_all.mean()

        # Edge detection on target
        target_float = target.float()
        edge = detect_edges(target_float)  # non-zero at edges
        edge_mask = (edge.abs() > 0).float()

        if edge_mask.sum() > 0:
            # Dice on edge pixels only
            intersection_edge = (pred_prob * edge_mask).sum(dim=[1, 2, 3])
            pred_edge_sum = (pred_prob * edge_mask).sum(dim=[1, 2, 3])
            target_edge_sum = edge_mask.sum(dim=[1, 2, 3])
            dice_edge = (2 * intersection_edge + self.smooth) / (pred_edge_sum + target_edge_sum + self.smooth)
            loss_edge = 1 - dice_edge.mean()
        else:
            loss_edge = torch.tensor(0.0, device=pred.device)

        total = loss_all + self.boundary_weight * loss_edge
        return total


class FocalBoundaryLoss(nn.Module):
    """
    Focal loss with boundary emphasis.

    FL = -α_t * (1 - p_t)^γ * log(p_t)
    où p_t = pred if target==1 else 1-pred

    Boundary weighting: multiply loss by (1 + λ * edge_mask)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 boundary_weight: float = 3.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) raw logits
            target: (B, 1, H, W) binary {0,1}

        Returns:
            loss: scalar
        """
        pred_prob = torch.sigmoid(pred)

        # Focal loss standard
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_factor = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_factor * ce_loss  # (B,1,H,W)

        # Boundary mask from target
        edge = detect_edges(target.float())
        edge_mask = (edge.abs() > 0).float()

        # Weight boundary pixels more
        weight_map = 1.0 + self.boundary_weight * edge_mask

        weighted_loss = weight_map * focal_loss

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class SoftHausdorffLoss(nn.Module):
    """
    Differentiable approximation of Hausdorff distance using soft minimum.

    HD(A,B) ≈ mean_{a∈A} softmin_{b∈B} ||a-b||² + mean_{b∈B} softmin_{a∈A} ||a-b||²

    Uses coordinate distances, not feature distances.
    """

    def __init__(self, alpha: float = 2.0, temperature: float = 0.01):
        super().__init__()
        self.alpha = alpha  # typically 2 for squared Euclidean
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) probability after sigmoid [0,1]
            target: (B, 1, H, W) binary {0,1}

        Returns:
            loss: scalar (average HD distance)
        """
        B, _, H, W = pred.shape
        device = pred.device
        total_loss = 0.0

        # Create coordinate grid (H*W, 2)
        with torch.no_grad():
            y = torch.arange(H, device=device).float()
            x = torch.arange(W, device=device).float()
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N, 2)
            N = coords.shape[0]

        for b in range(B):
            p_prob = pred[b, 0].flatten()  # (N,)
            t_binary = target[b, 0].flatten() > 0.5

            if not t_binary.any():
                continue

            t_coords = coords[t_binary]  # (Q, 2)
            Q = t_coords.shape[0]

            # Compute pairwise squared distances: (N, Q)
            # Using broadcasting: (N,1,2) - (1,Q,2) -> (N,Q,2) -> sum(-1) -> (N,Q)
            dists_sq = ((coords.unsqueeze(1) - t_coords.unsqueeze(0)) ** 2).sum(-1)  # (N,Q)

            # Direction 1: For each predicted pixel (weighted by prob), distance to nearest target
            # Soft min: weighted average of distances with softmax(-dists / temp)
            with torch.amp.autocast(device.type, enabled=False):
                weights = torch.softmax(-dists_sq / self.temperature, dim=1)  # (N, Q)
            min_dists_p = (weights * dists_sq).sum(dim=1)  # (N,)
            loss_p = (p_prob * min_dists_p).sum()

            # Direction 2: For each target pixel, distance to nearest predicted pixel
            # Use softmin over predicted positions (unweighted or weighted by pred prob?)
            # Simpler: use hard min since target points are ground truth (few)
            t_to_pred_dists = []
            for i in range(Q):
                d = dists_sq[:, i]  # (N,)
                min_d = d.min()
                t_to_pred_dists.append(min_d)
            if t_to_pred_dists:
                loss_t = torch.stack(t_to_pred_dists).mean()
            else:
                loss_t = torch.tensor(0.0, device=device)

            total_loss += loss_p + loss_t

        return total_loss / B


class CombinedBoundaryLoss(nn.Module):
    """
    Combine multiple boundary-focused losses.
    """

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 0.5,
                 hausdorff_weight: float = 0.1):
        super().__init__()
        self.dice_b = SoftDiceBoundaryLoss()
        self.focal_b = FocalBoundaryLoss()
        self.hausdorff = SoftHausdorffLoss()
        self.w_dice = dice_weight
        self.w_focal = focal_weight
        self.w_hd = hausdorff_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_dice * self.dice_b(pred, target)
        loss = loss + self.w_focal * self.focal_b(pred, target)
        loss = loss + self.w_hd * self.hausdorff(pred, target)
        return loss


def _test():
    """Quick test."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W = 2, 64, 64

    # Create synthetic masks
    target = torch.zeros(B, 1, H, W, device=device)
    target[:, :, 20:40, 20:40] = 1.0  # square
    pred = torch.rand(B, 1, H, W, device=device)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    print("Testing boundary losses...")

    # SoftDiceBoundaryLoss
    dice_b = SoftDiceBoundaryLoss()
    loss1 = dice_b(pred, target)
    print(f"SoftDiceBoundaryLoss: {loss1.item():.4f}")

    # FocalBoundaryLoss
    focal_b = FocalBoundaryLoss()
    loss2 = focal_b(pred, target)
    print(f"FocalBoundaryLoss: {loss2.item():.4f}")

    # SoftHausdorffLoss
    hd = SoftHausdorffLoss()
    loss3 = hd(pred, target)
    print(f"SoftHausdorffLoss: {loss3.item():.4f}")

    # Combined
    combined = CombinedBoundaryLoss()
    loss4 = combined(pred, target)
    print(f"CombinedBoundaryLoss: {loss4.item():.4f}")

    # Gradient check
    loss4.backward()
    assert pred.grad is not None
    print("Gradient OK")

    print("All tests passed!")


if __name__ == "__main__":
    _test()
