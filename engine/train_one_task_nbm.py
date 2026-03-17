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


def _expand_memory_for_batch(
    memory_state: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    device,
    dtype,
    model,
):
    if memory_state is None:
        return model.init_memory(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            from_slow=True,
            noise_std=1e-3,
            slow_scale=0.2,
        )

    expanded = {}
    for stage, value in memory_state.items():
        if value.ndim == 1:
            expanded[stage] = value.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        elif value.ndim == 2 and value.size(0) == batch_size:
            expanded[stage] = value.to(device=device, dtype=dtype)
        elif value.ndim == 2:
            expanded[stage] = value.mean(dim=0).to(device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        else:
            raise ValueError(f"Unsupported memory shape for {stage}: {value.shape}")
    return expanded


def _memory_reg(memory_before: Dict[str, torch.Tensor], memory_after: Dict[str, torch.Tensor]) -> torch.Tensor:
    reg = 0.0
    for stage in memory_before.keys():
        reg = reg + (memory_after[stage] - memory_before[stage]).pow(2).mean()
    return reg / max(len(memory_before), 1)


def _memory_delta(memory_before: Dict[str, torch.Tensor], memory_after: Dict[str, torch.Tensor]) -> float:
    delta = 0.0
    for stage in memory_before.keys():
        delta += (memory_after[stage] - memory_before[stage]).abs().mean().detach().item()
    return delta / max(len(memory_before), 1)


def _memory_norm(memory: Dict[str, torch.Tensor]) -> float:
    value = 0.0
    for stage in memory.keys():
        value += memory[stage].abs().mean().detach().item()
    return value / max(len(memory), 1)


def _stack_memory_summaries(memory_summaries):
    result = {}
    for stage in memory_summaries[0].keys():
        result[stage] = torch.stack([m[stage] for m in memory_summaries], dim=0).mean(dim=0).detach()
    return result


def train_one_task_nbm(
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
    memory_after_weight: float = 1.5,
    memory_stability_weight: float = 1e-4,
    improve_weight: float = 0.2,
    improve_margin: float = 0.005,
    before_weight: float = 0.3,
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
            outputs_before = model(images, memory)
            seg_loss_before, seg_before_dict = criterion(outputs_before, masks, return_components=True)

            proposed_memory = model.compute_updated_memory(
                memory_features={k: v.detach() for k, v in outputs_before["memory_features"].items()},
                memory={k: v.detach() for k, v in memory.items()},
                update_signals={k: v.detach() for k, v in outputs_before["update_signals"].items()},
            )

            outputs_after = model(images, proposed_memory)
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

        with torch.no_grad():
            memory = {k: v.detach() for k, v in proposed_memory.items()}
            memory_summaries.append(model.summarize_memory(memory))

        metric_before = compute_segmentation_metrics(outputs_before["logits"].detach(), masks)
        metric_after = compute_segmentation_metrics(outputs_after["logits"].detach(), masks)
        mem_delta_value = _memory_delta(memory, proposed_memory)
        mem_norm_value = _memory_norm(proposed_memory)

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
                f"[NBM-Train][Task {task_id}][Epoch {epoch}] "
                f"Step {step+1}/{len(loader)} | "
                f"loss={total_loss_meter.avg:.4f} | "
                f"seg_before={seg_before_meter.avg:.4f} | "
                f"seg_after={seg_after_meter.avg:.4f} | "
                f"improve={improve_loss_meter.avg:.6f} | "
                f"mem_reg={mem_reg_meter.avg:.8e} | "
                f"mem_delta={mem_delta_meter.avg:.8e} | "
                f"mem_norm={mem_norm_meter.avg:.8e} | "
                f"dice_before={dice_before_meter.avg:.4f} | "
                f"dice_after={dice_after_meter.avg:.4f} | "
                f"dice_gain={dice_gain_meter.avg:.4f} | "
                f"iou_after={iou_after_meter.avg:.4f} | "
                f"time={batch_time_meter.avg:.3f}s"
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
