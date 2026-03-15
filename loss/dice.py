import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss cho binary segmentation.
    Input:
        logits:  [B, 1, H, W]
        targets: [B, 1, H, W]
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        probs = probs.float()
        targets = targets.float()

        dims = (1, 2, 3)

        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        return loss.mean()