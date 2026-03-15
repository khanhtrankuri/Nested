from typing import Dict

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


@torch.no_grad()
def validate_nested(
    model,
    loader,
    criterion,
    device: str,
    epoch: int = 0,
    use_amp: bool = True,
    print_freq: int = 20,
) -> Dict[str, float]:
    model.eval()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and (device_type == "cuda")

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    memory = None

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
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

        metric_dict = compute_segmentation_metrics(logits, masks)

        # update memory cho batch sau
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

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[Nested-Val][Epoch {epoch}] "
                f"Step {step+1}/{len(loader)} | "
                f"loss={loss_meter.avg:.4f} | "
                f"dice={dice_meter.avg:.4f} | "
                f"iou={iou_meter.avg:.4f}"
            )

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }
