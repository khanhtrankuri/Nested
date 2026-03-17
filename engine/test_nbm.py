from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch
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
def test_nbm(
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

        if memory is None or next(iter(memory.values())).size(0) != batch_size:
            memory = model.init_memory(batch_size=batch_size, device=device, dtype=images.dtype)

        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(images, memory)
            loss, loss_dict = criterion(outputs, masks, return_components=True)

        metric_dict = compute_segmentation_metrics(outputs["logits"], masks, threshold=threshold)
        memory = model.update_memory(
            memory_features={k: v.detach() for k, v in outputs["memory_features"].items()},
            memory={k: v.detach() for k, v in memory.items()},
            update_signals={k: v.detach() for k, v in outputs["update_signals"].items()},
        )

        loss_meter.update(loss_dict["loss_total"], batch_size)
        dice_meter.update(metric_dict["dice"], batch_size)
        iou_meter.update(metric_dict["iou"], batch_size)
        precision_meter.update(metric_dict["precision"], batch_size)
        recall_meter.update(metric_dict["recall"], batch_size)

        if save_dir is not None:
            probs = torch.sigmoid(outputs["logits"])
            preds = (probs > threshold).float()
            for i in range(batch_size):
                _save_binary_mask(preds[i, 0], os.path.join(save_dir, file_names[i]))

    results = {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }

    print(
        "[NBM-Test] "
        f"loss={results['loss']:.4f} | "
        f"dice={results['dice']:.4f} | "
        f"iou={results['iou']:.4f} | "
        f"precision={results['precision']:.4f} | "
        f"recall={results['recall']:.4f}"
    )
    return results
