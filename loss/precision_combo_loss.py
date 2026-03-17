import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = torch.pow((1.0 - tversky), self.gamma)
        return loss.mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kernel_x)
        self.register_buffer("ky", kernel_y)
        self.bce = nn.BCEWithLogitsLoss()

    def mask_to_edge(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        g = torch.sqrt(gx * gx + gy * gy + 1e-6)
        g = (g > 0.05).float()
        return g

    def forward(self, edge_logits, targets):
        edge_targets = self.mask_to_edge(targets)
        return self.bce(edge_logits, edge_targets)


class PrecisionComboLoss(nn.Module):
    """
    Kéo precision lên:
    - BCE ổn định training
    - Focal Tversky phạt FP mạnh hơn
    - Dice giữ overlap
    - Edge loss giữ biên
    """
    def __init__(
        self,
        bce_w: float = 0.35,
        ft_w: float = 0.40,
        dice_w: float = 0.20,
        edge_w: float = 0.05,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        self.dice = SoftDiceLoss(smooth=1.0)
        self.edge = EdgeLoss()

        self.bce_w = bce_w
        self.ft_w = ft_w
        self.dice_w = dice_w
        self.edge_w = edge_w

    def forward(self, logits, targets, edge_logits=None):
        loss = (
            self.bce_w * self.bce(logits, targets) +
            self.ft_w * self.ft(logits, targets) +
            self.dice_w * self.dice(logits, targets)
        )
        if edge_logits is not None:
            loss = loss + self.edge_w * self.edge(edge_logits, targets)
        return loss