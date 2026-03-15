import torch
import torch.nn as nn

from loss.dice import DiceLoss


class BCEDiceLoss(nn.Module):
    """
    Tổng loss:
        total = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
 
        if return_components:
            return total_loss, {
                "loss_total": float(total_loss.detach().item()),
                "loss_bce": float(bce_loss.detach().item()),
                "loss_dice": float(dice_loss.detach().item()),
            }

        return total_loss