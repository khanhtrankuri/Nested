import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
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


def _save_binary_mask(mask_tensor: torch.Tensor, save_path: str):
    """
    mask_tensor: [H, W], value in {0,1}
    """
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(save_path)


@torch.no_grad()
def test(
    model,
    loader,
    criterion,
    device: str,
    save_dir: Optional[str] = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    loss_meter = AverageMeter()
    bce_meter = AverageMeter()
    dice_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    amp_enabled = use_amp and torch.cuda.is_available()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        file_names = batch["file_name"]

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            logits = outputs["logits"]
            loss, loss_dict = criterion(
                logits, masks, return_components=True
            )

        metric_dict = compute_segmentation_metrics(
            logits, masks, threshold=threshold
        )

        batch_size = images.size(0)
        loss_meter.update(loss_dict["loss_total"], batch_size)
        bce_meter.update(loss_dict["loss_bce"], batch_size)
        dice_loss_meter.update(loss_dict["loss_dice"], batch_size)
        dice_meter.update(metric_dict["dice"], batch_size)
        iou_meter.update(metric_dict["iou"], batch_size)
        precision_meter.update(metric_dict["precision"], batch_size)
        recall_meter.update(metric_dict["recall"], batch_size)

        if save_dir is not None:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            for i in range(batch_size):
                pred_mask = preds[i, 0]
                save_path = os.path.join(save_dir, file_names[i])
                _save_binary_mask(pred_mask, save_path)

    results = {
        "loss": loss_meter.avg,
        "loss_bce": bce_meter.avg,
        "loss_dice": dice_loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }

    print(
        "[Test] "
        f"loss={results['loss']:.4f} | "
        f"dice={results['dice']:.4f} | "
        f"iou={results['iou']:.4f} | "
        f"precision={results['precision']:.4f} | "
        f"recall={results['recall']:.4f}"
    )

    return results