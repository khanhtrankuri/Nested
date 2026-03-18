import copy
import json
import os
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
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


def _scaled_hw(height: int, width: int, scale: float, divisor: int = 32) -> Tuple[int, int]:
    if abs(float(scale) - 1.0) < 1e-6:
        return int(height), int(width)
    scaled_h = max(divisor, int(round((height * float(scale)) / divisor) * divisor))
    scaled_w = max(divisor, int(round((width * float(scale)) / divisor) * divisor))
    return scaled_h, scaled_w


def _forward_logits(model, images: torch.Tensor, use_nested: bool = False) -> torch.Tensor:
    return model(images, use_nested=use_nested)["logits"]


def _forward_with_tta(
    model,
    images: torch.Tensor,
    use_nested: bool = False,
    tta_scales: Optional[Sequence[float]] = None,
):
    base_h, base_w = images.shape[-2:]
    scales = [float(scale) for scale in (tta_scales or [1.0])]
    outputs = []

    for scale in scales:
        target_h, target_w = _scaled_hw(base_h, base_w, scale=scale)
        scaled_images = images
        if (target_h, target_w) != (base_h, base_w):
            scaled_images = F.interpolate(images, size=(target_h, target_w), mode="bilinear", align_corners=False)

        for flip_dims in (None, [3], [2], [2, 3]):
            flipped_images = scaled_images if flip_dims is None else torch.flip(scaled_images, dims=flip_dims)
            logits = _forward_logits(model, flipped_images, use_nested=use_nested)
            if flip_dims is not None:
                logits = torch.flip(logits, dims=flip_dims)
            if logits.shape[-2:] != (base_h, base_w):
                logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)
            outputs.append(logits)

    return torch.stack(outputs, dim=0).mean(dim=0)


def _select_primary_outputs(outputs: Dict[str, torch.Tensor], use_coarse: bool = False) -> Dict[str, torch.Tensor]:
    selected = dict(outputs)
    if use_coarse and "coarse_logits" in outputs:
        selected["logits"] = outputs["coarse_logits"]
    return selected


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
    use_nested: bool = False,
    skip_nested_if_hurts: bool = True,
    nested_skip_margin: float = 0.002,
    nested_momentum: float = 0.03,
    nested_max_norm: float = 1.0,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    nested_use_meter = AverageMeter()
    nested_delta_meter = AverageMeter()
    prototype_norm_meter = AverageMeter()

    amp_enabled = use_amp and torch.cuda.is_available()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images, use_nested=use_nested)
            nested_available = bool(outputs.get("nested_cache") is not None)
            if nested_available:
                base_outputs = _select_primary_outputs(outputs, use_coarse=True)
                base_loss, base_loss_dict = criterion(base_outputs, masks, return_components=True)
                nested_loss, nested_loss_dict = criterion(outputs, masks, return_components=True)
                nested_hurts = bool((nested_loss.detach() > base_loss.detach() + nested_skip_margin).item())
                if skip_nested_if_hurts and nested_hurts:
                    chosen_outputs = base_outputs
                    loss = base_loss
                    loss_dict = base_loss_dict
                    nested_used = False
                else:
                    chosen_outputs = outputs
                    loss = nested_loss
                    loss_dict = nested_loss_dict
                    nested_used = True
            else:
                chosen_outputs = outputs
                loss, loss_dict = criterion(outputs, masks, return_components=True)
                nested_used = False

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

        if use_nested and nested_used:
            model.update_nested_prototypes(outputs["nested_cache"], momentum=nested_momentum, max_norm=nested_max_norm)
        if ema is not None:
            ema.update(model)

        metric_dict = compute_segmentation_metrics(chosen_outputs["logits"].detach(), masks)
        bs = images.size(0)
        loss_meter.update(loss_dict["loss_total"], bs)
        dice_meter.update(metric_dict["dice"], bs)
        iou_meter.update(metric_dict["iou"], bs)
        precision_meter.update(metric_dict["precision"], bs)
        recall_meter.update(metric_dict["recall"], bs)
        nested_use_meter.update(float(nested_used), bs)
        nested_delta_meter.update(float(outputs["nested_info"]["delta_mean"].detach().item()), bs)
        prototype_norm_meter.update(float(outputs["nested_info"]["prototype_norm"].detach().item()), bs)

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[Train][Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f} | "
                f"nested_used={nested_use_meter.avg:.3f} | nested_delta={nested_delta_meter.avg:.5f}"
            )

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
        "nested_used": nested_use_meter.avg,
        "nested_delta": nested_delta_meter.avg,
        "prototype_norm": prototype_norm_meter.avg,
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
    tta_scales: Optional[Sequence[float]] = None,
    print_freq: int = 20,
    use_nested: bool = False,
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
            logits = (
                _forward_with_tta(model, images, use_nested=use_nested, tta_scales=tta_scales)
                if use_tta
                else _forward_logits(model, images, use_nested=use_nested)
            )
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
        "use_nested": bool(use_nested),
        "use_tta": bool(use_tta),
        "tta_scales": [float(scale) for scale in (tta_scales or [1.0])],
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
    tta_scales: Optional[Sequence[float]] = None,
    use_nested: bool = False,
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
            tta_scales=tta_scales,
            print_freq=max(len(loader), 1),
            use_nested=use_nested,
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
    tta_scales: Optional[Sequence[float]] = None,
    use_nested: bool = False,
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
        tta_scales=tta_scales,
        print_freq=max(len(loader), 1),
        use_nested=use_nested,
    )
    return result
