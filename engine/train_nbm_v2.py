from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

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
def _expand_memory_for_batch(memory, batch_size, device, dtype, model):
    if memory is None:
        return model.init_memory(batch_size=batch_size, device=device, dtype=dtype)
    if next(iter(memory.values())).size(0) == batch_size:
        return {k: v.to(device=device, dtype=dtype) for k, v in memory.items()}
    expanded = {}
    for stage, value in memory.items():
        stage_summary = value.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
        expanded[stage] = stage_summary.repeat(batch_size, 1, 1)
    return expanded


def _memory_reg(memory_before: Dict[str, torch.Tensor], memory_after: Dict[str, torch.Tensor]) -> torch.Tensor:
    reg = 0.0
    count = 0
    for stage in memory_before.keys():
        reg = reg + torch.mean((memory_after[stage] - memory_before[stage]) ** 2)
        count += 1
    return reg / max(count, 1)


@torch.no_grad()
def _memory_delta(memory_before: Dict[str, torch.Tensor], memory_after: Dict[str, torch.Tensor]) -> float:
    delta = 0.0
    for stage in memory_before.keys():
        delta += (memory_after[stage] - memory_before[stage]).abs().mean().detach().item()
    return delta / max(len(memory_before), 1)


@torch.no_grad()
def _memory_norm(memory: Dict[str, torch.Tensor]) -> float:
    value = 0.0
    for stage in memory.keys():
        value += memory[stage].abs().mean().detach().item()
    return value / max(len(memory), 1)


@torch.no_grad()
def _stack_memory_summaries(memory_summaries):
    result = {}
    for stage in memory_summaries[0].keys():
        result[stage] = torch.stack([m[stage] for m in memory_summaries], dim=0).mean(dim=0).detach()
    return result


def train_epoch_warmup_v2(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    threshold: float = 0.5,
    print_freq: int = 20,
):
    model.train()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and (device_type == "cuda")

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    start_time = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(images, memory=None, use_memory=False)
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

        metrics = compute_segmentation_metrics(outputs["logits"].detach(), masks, threshold=threshold)
        loss_meter.update(loss_dict["loss_total"], batch_size)
        dice_meter.update(metrics["dice"], batch_size)
        iou_meter.update(metrics["iou"], batch_size)
        precision_meter.update(metrics["precision"], batch_size)
        recall_meter.update(metrics["recall"], batch_size)

        batch_time_meter.update(time.time() - start_time)
        start_time = time.time()

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[NBM-Warmup][Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f} | "
                f"precision={precision_meter.avg:.4f} | recall={recall_meter.avg:.4f} | "
                f"time={batch_time_meter.avg:.3f}s"
            )

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
    }


def train_one_task_nbm_v2(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    task_id: int,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    memory_after_weight: float = 1.25,
    memory_stability_weight: float = 1e-4,
    improve_weight: float = 0.2,
    improve_margin: float = 0.002,
    before_weight: float = 0.5,
    threshold: float = 0.5,
    print_freq: int = 20,
    initial_memory: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.train()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and (device_type == "cuda")

    total_loss_meter = AverageMeter()
    seg_before_meter = AverageMeter()
    seg_after_meter = AverageMeter()
    improve_loss_meter = AverageMeter()
    mem_reg_meter = AverageMeter()
    mem_delta_meter = AverageMeter()
    mem_norm_meter = AverageMeter()
    dice_before_meter = AverageMeter()
    dice_after_meter = AverageMeter()
    dice_gain_meter = AverageMeter()
    iou_after_meter = AverageMeter()
    precision_after_meter = AverageMeter()
    recall_after_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    start_time = time.time()
    memory = None
    memory_summaries = []

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        if memory is None:
            memory = _expand_memory_for_batch(initial_memory, batch_size, device, images.dtype, model)
        elif next(iter(memory.values())).size(0) != batch_size:
            memory = _expand_memory_for_batch(memory, batch_size, device, images.dtype, model)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs_before = model(images, memory=memory, use_memory=True)
            seg_loss_before, seg_before_dict = criterion(outputs_before, masks, return_components=True)

            proposed_memory = model.compute_updated_memory(
                memory_features={k: v.detach() for k, v in outputs_before["memory_features"].items()},
                memory={k: v.detach() for k, v in memory.items()},
                update_signals={k: v.detach() for k, v in outputs_before["update_signals"].items()},
                attention_cache={k: v for k, v in outputs_before["attention_cache"].items()},
            )

            outputs_after = model(images, memory=proposed_memory, use_memory=True)
            seg_loss_after, seg_after_dict = criterion(outputs_after, masks, return_components=True)

            memory_reg = _memory_reg(memory, proposed_memory)
            improve_loss = torch.relu(seg_loss_after - seg_loss_before + improve_margin)
            total_loss = (
                before_weight * seg_loss_before
                + memory_after_weight * seg_loss_after
                + improve_weight * improve_loss
                + memory_stability_weight * memory_reg
            )

        if scaler is not None and amp_enabled:
            scaler.scale(total_loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        mem_delta_value = _memory_delta(memory, proposed_memory)
        mem_norm_value = _memory_norm(proposed_memory)

        with torch.no_grad():
            memory = {k: v.detach() for k, v in proposed_memory.items()}
            memory_summaries.append(model.summarize_memory(memory))

        metric_before = compute_segmentation_metrics(outputs_before["logits"].detach(), masks, threshold=threshold)
        metric_after = compute_segmentation_metrics(outputs_after["logits"].detach(), masks, threshold=threshold)

        total_loss_meter.update(float(total_loss.detach().item()), batch_size)
        seg_before_meter.update(seg_before_dict["loss_total"], batch_size)
        seg_after_meter.update(seg_after_dict["loss_total"], batch_size)
        improve_loss_meter.update(float(improve_loss.detach().item()), batch_size)
        mem_reg_meter.update(float(memory_reg.detach().item()), batch_size)
        mem_delta_meter.update(mem_delta_value, batch_size)
        mem_norm_meter.update(mem_norm_value, batch_size)
        dice_before_meter.update(metric_before["dice"], batch_size)
        dice_after_meter.update(metric_after["dice"], batch_size)
        dice_gain_meter.update(metric_after["dice"] - metric_before["dice"], batch_size)
        iou_after_meter.update(metric_after["iou"], batch_size)
        precision_after_meter.update(metric_after["precision"], batch_size)
        recall_after_meter.update(metric_after["recall"], batch_size)

        batch_time_meter.update(time.time() - start_time)
        start_time = time.time()

        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(
                f"[NBM-v2-Train][Task {task_id}][Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"loss={total_loss_meter.avg:.4f} | seg_before={seg_before_meter.avg:.4f} | seg_after={seg_after_meter.avg:.4f} | "
                f"improve={improve_loss_meter.avg:.6f} | mem_reg={mem_reg_meter.avg:.8e} | mem_delta={mem_delta_meter.avg:.8e} | "
                f"mem_norm={mem_norm_meter.avg:.8e} | dice_before={dice_before_meter.avg:.4f} | dice_after={dice_after_meter.avg:.4f} | "
                f"dice_gain={dice_gain_meter.avg:.4f} | iou_after={iou_after_meter.avg:.4f} | "
                f"precision_after={precision_after_meter.avg:.4f} | recall_after={recall_after_meter.avg:.4f} | time={batch_time_meter.avg:.3f}s"
            )

    if len(memory_summaries) == 0:
        raise RuntimeError("No memory summaries collected in this epoch.")

    metrics = {
        "loss": total_loss_meter.avg,
        "seg_before": seg_before_meter.avg,
        "seg_after": seg_after_meter.avg,
        "improve_loss": improve_loss_meter.avg,
        "mem_reg": mem_reg_meter.avg,
        "mem_delta": mem_delta_meter.avg,
        "mem_norm": mem_norm_meter.avg,
        "dice_before": dice_before_meter.avg,
        "dice": dice_after_meter.avg,
        "dice_gain": dice_gain_meter.avg,
        "iou": iou_after_meter.avg,
        "precision": precision_after_meter.avg,
        "recall": recall_after_meter.avg,
    }
    return metrics, _stack_memory_summaries(memory_summaries)
