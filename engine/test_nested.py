import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast

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
        return 0.0 if self.count == 0 else self.sum / self.count


def _save_binary_mask(mask_tensor: torch.Tensor, save_path: str):
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(save_path)


@torch.no_grad()
def test_nested(
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

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and (device_type == "cuda")

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    memory = None

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        file_names = batch["file_name"]
        batch_size = images.size(0)

        if memory is None or memory.size(0) != batch_size:
            memory = model.init_memory(
                batch_size=batch_size,
                device=device,
                dtype=images.dtype,
            )

        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(images, memory)
            logits = outputs["logits"]
            feat = outputs["feat"]

            loss, loss_dict = criterion(logits, masks, return_components=True)

        metric_dict = compute_segmentation_metrics(
            logits, masks, threshold=threshold
        )

        loss_scalar = loss.detach().view(1, 1).repeat(batch_size, 1)
        memory = model.update_memory(
            feat=feat.detach(),
            memory=memory.detach(),
            loss_scalar=loss_scalar,
        )

        loss_meter.update(loss_dict["loss_total"], batch_size)
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
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }

    print(
        "[Nested-Test] "
        f"loss={results['loss']:.4f} | "
        f"dice={results['dice']:.4f} | "
        f"iou={results['iou']:.4f} | "
        f"precision={results['precision']:.4f} | "
        f"recall={results['recall']:.4f}"
    )

    return results
