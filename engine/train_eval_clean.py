import copy
import json
import os
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional

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
    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.sum / self.count


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)


def _forward_with_tta(model, images: torch.Tensor):
    outputs = []
    logits = model(images)["logits"]
    outputs.append(logits)

    h = torch.flip(images, dims=[3])
    outputs.append(torch.flip(model(h)["logits"], dims=[3]))

    v = torch.flip(images, dims=[2])
    outputs.append(torch.flip(model(v)["logits"], dims=[2]))

    hv = torch.flip(images, dims=[2, 3])
    outputs.append(torch.flip(model(hv)["logits"], dims=[2, 3]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def train_one_epoch_clean(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    scaler=None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    ema: Optional[ModelEMA] = None,
    print_freq: int = 20,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    amp_enabled = use_amp and torch.cuda.is_available()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks, return_components=True)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        metric_dict = compute_segmentation_metrics(outputs["logits"].detach(), masks)
        bs = images.size(0)
        loss_meter.update(loss_dict["loss_total"], bs)
        dice_meter.update(metric_dict["dice"], bs)
        iou_meter.update(metric_dict["iou"], bs)
        precision_meter.update(metric_dict["precision"], bs)
        recall_meter.update(metric_dict["recall"], bs)

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[Train][Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f}"
            )

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }


@torch.no_grad()
def evaluate_clean(
    model,
    loader,
    criterion,
    device: str,
    epoch: int = 0,
    use_amp: bool = True,
    threshold: float = 0.5,
    use_tta: bool = False,
    print_freq: int = 20,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    amp_enabled = use_amp and torch.cuda.is_available()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)
            logits = _forward_with_tta(model, images) if use_tta else outputs["logits"]
            loss, _ = criterion({"logits": logits}, masks, return_components=True)

        metric_dict = compute_segmentation_metrics(logits, masks, threshold=threshold)
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        dice_meter.update(metric_dict["dice"], bs)
        iou_meter.update(metric_dict["iou"], bs)
        precision_meter.update(metric_dict["precision"], bs)
        recall_meter.update(metric_dict["recall"], bs)

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[Eval][Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f}"
            )

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
        "threshold": threshold,
    }


@torch.no_grad()
def threshold_sweep_clean(
    model,
    loader,
    criterion,
    device: str,
    thresholds: Iterable[float],
    use_amp: bool = True,
    use_tta: bool = False,
) -> Dict[str, float]:
    best_result = None
    for threshold in thresholds:
        result = evaluate_clean(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            threshold=float(threshold),
            use_amp=use_amp,
            use_tta=use_tta,
            print_freq=max(len(loader), 1),
        )
        if best_result is None:
            best_result = result
        else:
            if result["iou"] > best_result["iou"] or (
                result["iou"] == best_result["iou"] and result["dice"] > best_result["dice"]
            ):
                best_result = result
    return best_result


@torch.no_grad()
def test_clean(
    model,
    loader,
    criterion,
    device: str,
    save_dir: Optional[str] = None,
    threshold: float = 0.5,
    use_amp: bool = True,
    use_tta: bool = True,
) -> Dict[str, float]:
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    result = evaluate_clean(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
        use_amp=use_amp,
        use_tta=use_tta,
        print_freq=max(len(loader), 1),
    )
    return result
