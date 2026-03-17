from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.amp import autocast

from engine.train_one_epoch_hybrid import AverageMeter, _blend_outputs, _expand_memory_for_batch
from metrics.segmentation_metrics import compute_segmentation_metrics


class ThresholdMeter:
    def __init__(self, thresholds: Iterable[float]):
        self.thresholds = [float(t) for t in thresholds]
        self.dice = {t: AverageMeter() for t in self.thresholds}
        self.iou = {t: AverageMeter() for t in self.thresholds}
        self.precision = {t: AverageMeter() for t in self.thresholds}
        self.recall = {t: AverageMeter() for t in self.thresholds}

    def update(self, logits: torch.Tensor, targets: torch.Tensor, batch_size: int):
        for threshold in self.thresholds:
            metric = compute_segmentation_metrics(logits, targets, threshold=threshold)
            self.dice[threshold].update(metric["dice"], batch_size)
            self.iou[threshold].update(metric["iou"], batch_size)
            self.precision[threshold].update(metric["precision"], batch_size)
            self.recall[threshold].update(metric["recall"], batch_size)

    def best(self) -> Dict[str, float]:
        best_threshold = max(self.thresholds, key=lambda t: self.dice[t].avg)
        return {
            "best_threshold": best_threshold,
            "dice": self.dice[best_threshold].avg,
            "iou": self.iou[best_threshold].avg,
            "precision": self.precision[best_threshold].avg,
            "recall": self.recall[best_threshold].avg,
            "all_thresholds": {
                f"{threshold:.2f}": {
                    "dice": self.dice[threshold].avg,
                    "iou": self.iou[threshold].avg,
                    "precision": self.precision[threshold].avg,
                    "recall": self.recall[threshold].avg,
                }
                for threshold in self.thresholds
            },
        }


@torch.no_grad()
def validate_hybrid(
    model,
    loader,
    criterion,
    device: str,
    epoch: int = 0,
    use_amp: bool = True,
    threshold: float = 0.5,
    threshold_sweep: Optional[Iterable[float]] = None,
    freeze_eval_memory: bool = True,
    use_memory: bool = False,
    disable_memory: bool = False,
    memory_blend: float = 0.10,
    print_freq: int = 20,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    model.eval()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and (device_type == "cuda")
    memory_active = bool(use_memory and not disable_memory)

    meters = {
        "loss": AverageMeter(),
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
    }
    if memory_active:
        meters.update(
            {
                "seg_before": AverageMeter(),
                "seg_after": AverageMeter(),
                "memory_delta": AverageMeter(),
                "fast_norm_stage3": AverageMeter(),
                "slow_norm_stage3": AverageMeter(),
                "fast_norm_stage2": AverageMeter(),
                "slow_norm_stage2": AverageMeter(),
            }
        )
    sweep_meter = ThresholdMeter(threshold_sweep) if threshold_sweep is not None else None

    eval_memory_seed = None
    if memory_active:
        eval_memory_seed = model.summarize_memory(model.init_memory(batch_size=1, device=device, noise_std=0.0))

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        batch_memory = None
        if memory_active:
            batch_memory = _expand_memory_for_batch(eval_memory_seed, model, batch_size, device, images.dtype)

        with autocast(device_type=device_type, enabled=amp_enabled):
            base_outputs = model(images, memory=None, use_memory=False, disable_memory=True)
            if memory_active:
                memory_outputs = model(images, memory=batch_memory, use_memory=True, disable_memory=False)
                outputs = _blend_outputs(base_outputs, memory_outputs, memory_blend)
                loss_before, before_dict = criterion(base_outputs, masks, return_components=True)
                loss, loss_dict = criterion(outputs, masks, return_components=True)
            else:
                outputs = base_outputs
                loss_before, before_dict = None, None
                loss, loss_dict = criterion(outputs, masks, return_components=True)

        metrics = compute_segmentation_metrics(outputs["logits"], masks, threshold=threshold)
        for key in ("loss", "dice", "iou", "precision", "recall"):
            value = loss_dict["loss_total"] if key == "loss" else metrics[key]
            meters[key].update(value, batch_size)

        if sweep_meter is not None:
            sweep_meter.update(outputs["logits"], masks, batch_size)

        if memory_active:
            memory_info = outputs["memory_info"]
            meters["seg_before"].update(before_dict["loss_total"], batch_size)
            meters["seg_after"].update(loss_dict["loss_total"], batch_size)
            meters["memory_delta"].update(float(memory_info["memory_delta"].detach().item()), batch_size)
            meters["fast_norm_stage3"].update(float(memory_info["fast_norm_stage3"].detach().item()), batch_size)
            meters["slow_norm_stage3"].update(float(memory_info["slow_norm_stage3"].detach().item()), batch_size)
            meters["fast_norm_stage2"].update(float(memory_info["fast_norm_stage2"].detach().item()), batch_size)
            meters["slow_norm_stage2"].update(float(memory_info["slow_norm_stage2"].detach().item()), batch_size)

            if not freeze_eval_memory:
                eval_memory_seed = model.summarize_memory(
                    model.compute_updated_memory(
                        memory_features={k: v.detach() for k, v in memory_outputs["memory_features"].items()},
                        memory={k: v.detach() for k, v in batch_memory.items()},
                        update_signals={k: v.detach() for k, v in memory_outputs["update_signals"].items()},
                        attention_cache={k: v for k, v in memory_outputs["attention_cache"].items()},
                    )
                )

        if step % print_freq == 0 or step == len(loader):
            print(
                f"[NBM-Hybrid-Val][Epoch {epoch}] Step {step}/{len(loader)} | "
                f"loss={meters['loss'].avg:.4f} | dice={meters['dice'].avg:.4f} | "
                f"iou={meters['iou'].avg:.4f} | precision={meters['precision'].avg:.4f} | "
                f"recall={meters['recall'].avg:.4f}"
            )

    result = {key: meter.avg for key, meter in meters.items()}
    result["memory_active"] = float(memory_active)
    return result, (sweep_meter.best() if sweep_meter is not None else None)
