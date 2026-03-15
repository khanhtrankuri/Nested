from typing import Dict

import torch
from torch.cuda.amp import autocast

from metrics.segmentation_metrics import compute_segmentation_metrics


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device: str,
    epoch: int = 0,
    use_amp: bool = True,
    print_freq: int = 20,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    bce_meter = AverageMeter()
    dice_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    amp_enabled = use_amp and torch.cuda.is_available()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            logits = outputs["logits"]
            loss, loss_dict = criterion(
                logits, masks, return_components=True
            )

        metric_dict = compute_segmentation_metrics(logits, masks)

        batch_size = images.size(0)
        loss_meter.update(loss_dict["loss_total"], batch_size)
        bce_meter.update(loss_dict["loss_bce"], batch_size)
        dice_loss_meter.update(loss_dict["loss_dice"], batch_size)
        dice_meter.update(metric_dict["dice"], batch_size)
        iou_meter.update(metric_dict["iou"], batch_size)
        precision_meter.update(metric_dict["precision"], batch_size)
        recall_meter.update(metric_dict["recall"], batch_size)

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[Val][Epoch {epoch}] "
                f"Step {step+1}/{len(loader)} | "
                f"loss={loss_meter.avg:.4f} | "
                f"dice={dice_meter.avg:.4f} | "
                f"iou={iou_meter.avg:.4f}"
            )

    return {
        "loss": loss_meter.avg,
        "loss_bce": bce_meter.avg,
        "loss_dice": dice_loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }