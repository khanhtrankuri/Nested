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
    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.sum / self.count


@torch.no_grad()
def _expand_memory_for_batch(memory, model, batch_size: int, device, dtype):
    if memory is None:
        return model.init_memory(batch_size=batch_size, device=device, dtype=dtype)
    expanded = {}
    for stage, value in memory.items():
        if value.ndim == 3:
            if value.size(0) == batch_size:
                expanded[stage] = value.to(device=device, dtype=dtype)
            else:
                summary = value.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
                expanded[stage] = summary.repeat(batch_size, 1, 1)
        elif value.ndim == 2:
            expanded[stage] = value.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            raise ValueError(f"Unexpected memory shape for {stage}: {tuple(value.shape)}")
    return expanded


@torch.no_grad()
def _memory_reg(memory_before: Dict[str, torch.Tensor], memory_after: Dict[str, torch.Tensor]) -> torch.Tensor:
    reg = 0.0
    for stage in memory_before:
        reg = reg + torch.mean((memory_after[stage] - memory_before[stage]) ** 2)
    return reg / max(len(memory_before), 1)


@torch.no_grad()
def _memory_delta(memory_before: Dict[str, torch.Tensor], memory_after: Dict[str, torch.Tensor]) -> float:
    delta = 0.0
    for stage in memory_before:
        delta += float((memory_after[stage] - memory_before[stage]).abs().mean().item())
    return delta / max(len(memory_before), 1)


def _blend_value(base_value, memory_value, blend: float):
    if isinstance(base_value, torch.Tensor) and isinstance(memory_value, torch.Tensor):
        return (1.0 - blend) * base_value + blend * memory_value
    if isinstance(base_value, (list, tuple)) and isinstance(memory_value, (list, tuple)):
        blended = [_blend_value(b, m, blend) for b, m in zip(base_value, memory_value)]
        return type(memory_value)(blended) if isinstance(memory_value, tuple) else blended
    return memory_value


def _blend_outputs(base_outputs: Dict, memory_outputs: Dict, blend: float) -> Dict:
    if blend <= 0.0:
        return base_outputs
    merged = dict(memory_outputs)
    for key in ("logits", "coarse_logits", "edge_logits", "aux_logits"):
        if key in base_outputs and key in memory_outputs:
            merged[key] = _blend_value(base_outputs[key], memory_outputs[key], blend)
    return merged


def _update_metric_meters(meters: Dict[str, AverageMeter], metrics: Dict[str, float], batch_size: int):
    for key, value in metrics.items():
        if key in meters:
            meters[key].update(float(value), batch_size)


def _build_metric_meters(include_memory: bool) -> Dict[str, AverageMeter]:
    meters = {
        "loss": AverageMeter(),
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
    }
    if include_memory:
        meters.update(
            {
                "seg_before": AverageMeter(),
                "seg_after": AverageMeter(),
                "improve_loss": AverageMeter(),
                "mem_reg": AverageMeter(),
                "mem_delta": AverageMeter(),
                "memory_delta": AverageMeter(),
                "fast_norm_stage3": AverageMeter(),
                "slow_norm_stage3": AverageMeter(),
                "fast_norm_stage2": AverageMeter(),
                "slow_norm_stage2": AverageMeter(),
                "dice_before": AverageMeter(),
                "dice_gain": AverageMeter(),
            }
        )
    return meters


def _print_train_status(
    prefix: str,
    epoch: int,
    step: int,
    total_steps: int,
    meters: Dict[str, AverageMeter],
    batch_time: AverageMeter,
    memory_active: bool,
):
    if memory_active:
        print(
            f"[{prefix}][Epoch {epoch}] Step {step}/{total_steps} | "
            f"loss={meters['loss'].avg:.4f} | seg_before={meters['seg_before'].avg:.4f} | "
            f"seg_after={meters['seg_after'].avg:.4f} | dice_before={meters['dice_before'].avg:.4f} | "
            f"dice={meters['dice'].avg:.4f} | dice_gain={meters['dice_gain'].avg:.4f} | "
            f"mem_delta={meters['mem_delta'].avg:.6f} | mem_reg={meters['mem_reg'].avg:.6f} | "
            f"memory_delta={meters['memory_delta'].avg:.6f} | "
            f"fast_s3={meters['fast_norm_stage3'].avg:.4f} | slow_s3={meters['slow_norm_stage3'].avg:.4f} | "
            f"fast_s2={meters['fast_norm_stage2'].avg:.4f} | slow_s2={meters['slow_norm_stage2'].avg:.4f} | "
            f"time={batch_time.avg:.3f}s"
        )
    else:
        print(
            f"[{prefix}][Epoch {epoch}] Step {step}/{total_steps} | "
            f"loss={meters['loss'].avg:.4f} | dice={meters['dice'].avg:.4f} | "
            f"iou={meters['iou'].avg:.4f} | precision={meters['precision'].avg:.4f} | "
            f"recall={meters['recall'].avg:.4f} | time={batch_time.avg:.3f}s"
        )


def _train_epoch_impl(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    stage_name: str,
    scaler=None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    threshold: float = 0.5,
    print_freq: int = 20,
    use_memory: bool = False,
    disable_memory: bool = False,
    memory_blend: float = 0.10,
    before_weight: float = 0.7,
    memory_after_weight: float = 1.0,
    memory_stability_weight: float = 1e-4,
    improve_weight: float = 0.1,
    improve_margin: float = 0.001,
    skip_memory_if_hurts: bool = True,
    skip_margin: float = 0.002,
    fast_memory_mode: str = "batch",
    initial_memory=None,
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    model.train()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and (device_type == "cuda")
    memory_active = bool(use_memory and not disable_memory)

    meters = _build_metric_meters(include_memory=memory_active)
    batch_time = AverageMeter()
    start_time = time.time()
    memory_summaries = []

    running_memory = initial_memory if fast_memory_mode == "task" else None
    if fast_memory_mode not in {"batch", "epoch", "task"}:
        raise ValueError("fast_memory_mode must be one of: batch, epoch, task")

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        if memory_active:
            if fast_memory_mode == "batch":
                current_memory = model.init_memory(batch_size=batch_size, device=device, dtype=images.dtype)
            else:
                current_memory = _expand_memory_for_batch(running_memory, model, batch_size, device, images.dtype)
        else:
            current_memory = None

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=amp_enabled):
            base_outputs = model(images, memory=None, use_memory=False, disable_memory=True)
            loss_before, before_dict = criterion(base_outputs, masks, return_components=True)

            if memory_active:
                seed_outputs = model(images, memory=current_memory, use_memory=True, disable_memory=False)
                proposed_memory = model.compute_updated_memory(
                    memory_features={k: v.detach() for k, v in seed_outputs["memory_features"].items()},
                    memory={k: v.detach() for k, v in current_memory.items()},
                    update_signals={k: v.detach() for k, v in seed_outputs["update_signals"].items()},
                    attention_cache={k: v for k, v in seed_outputs["attention_cache"].items()},
                )
                memory_outputs = model(images, memory=proposed_memory, use_memory=True, disable_memory=False)
                outputs = _blend_outputs(base_outputs, memory_outputs, memory_blend)
                loss_after, after_dict = criterion(outputs, masks, return_components=True)
                memory_reg = _memory_reg(current_memory, proposed_memory)
                improve_loss = torch.relu(loss_after - loss_before + improve_margin)
                total_loss = (
                    before_weight * loss_before
                    + memory_after_weight * loss_after
                    + memory_stability_weight * memory_reg
                    + improve_weight * improve_loss
                )

                memory_hurts = bool((loss_after.detach() > loss_before.detach() + skip_margin).item())
                if skip_memory_if_hurts and memory_hurts:
                    total_loss = loss_before
                    outputs = base_outputs
                    after_dict = before_dict
                    loss_after = loss_before
                    memory_reg = torch.zeros_like(memory_reg)
                    improve_loss = torch.zeros_like(improve_loss)
                    proposed_memory = current_memory
            else:
                outputs = base_outputs
                loss_after = loss_before
                after_dict = before_dict
                proposed_memory = None
                memory_reg = None
                improve_loss = None
                total_loss = loss_before

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

        output_metrics = compute_segmentation_metrics(outputs["logits"].detach(), masks, threshold=threshold)
        _update_metric_meters(meters, {"loss": float(total_loss.detach().item()), **output_metrics}, batch_size)

        if memory_active:
            before_metrics = compute_segmentation_metrics(base_outputs["logits"].detach(), masks, threshold=threshold)
            memory_info = outputs["memory_info"]
            mem_delta = _memory_delta(current_memory, proposed_memory)
            meters["seg_before"].update(before_dict["loss_total"], batch_size)
            meters["seg_after"].update(after_dict["loss_total"], batch_size)
            meters["improve_loss"].update(float(improve_loss.detach().item()), batch_size)
            meters["mem_reg"].update(float(memory_reg.detach().item()), batch_size)
            meters["mem_delta"].update(mem_delta, batch_size)
            meters["memory_delta"].update(float(memory_info["memory_delta"].detach().item()), batch_size)
            meters["fast_norm_stage3"].update(float(memory_info["fast_norm_stage3"].detach().item()), batch_size)
            meters["slow_norm_stage3"].update(float(memory_info["slow_norm_stage3"].detach().item()), batch_size)
            meters["fast_norm_stage2"].update(float(memory_info["fast_norm_stage2"].detach().item()), batch_size)
            meters["slow_norm_stage2"].update(float(memory_info["slow_norm_stage2"].detach().item()), batch_size)
            meters["dice_before"].update(before_metrics["dice"], batch_size)
            meters["dice_gain"].update(output_metrics["dice"] - before_metrics["dice"], batch_size)

            summary = model.summarize_memory(proposed_memory)
            memory_summaries.append(summary)
            if fast_memory_mode in {"epoch", "task"}:
                running_memory = summary

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        if step % print_freq == 0 or step == len(loader):
            _print_train_status(stage_name, epoch, step, len(loader), meters, batch_time, memory_active)

    metrics = {key: meter.avg for key, meter in meters.items()}
    epoch_summary = None
    if memory_summaries:
        epoch_summary = {}
        for stage in memory_summaries[0]:
            epoch_summary[stage] = torch.stack([summary[stage] for summary in memory_summaries], dim=0).mean(dim=0)

    next_seed = running_memory if memory_active and fast_memory_mode == "task" else None
    metrics["memory_active"] = float(memory_active)
    return metrics, epoch_summary, next_seed


def train_one_epoch_hybrid(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    scaler=None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    threshold: float = 0.5,
    print_freq: int = 20,
    use_memory: bool = False,
    disable_memory: bool = False,
    memory_blend: float = 0.10,
    before_weight: float = 0.7,
    memory_after_weight: float = 1.0,
    memory_stability_weight: float = 1e-4,
    improve_weight: float = 0.1,
    improve_margin: float = 0.001,
    skip_memory_if_hurts: bool = True,
    skip_margin: float = 0.002,
    fast_memory_mode: str = "batch",
    initial_memory=None,
):
    return _train_epoch_impl(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=epoch,
        stage_name="NBM-Hybrid-Train",
        scaler=scaler,
        use_amp=use_amp,
        grad_clip=grad_clip,
        threshold=threshold,
        print_freq=print_freq,
        use_memory=use_memory,
        disable_memory=disable_memory,
        memory_blend=memory_blend,
        before_weight=before_weight,
        memory_after_weight=memory_after_weight,
        memory_stability_weight=memory_stability_weight,
        improve_weight=improve_weight,
        improve_margin=improve_margin,
        skip_memory_if_hurts=skip_memory_if_hurts,
        skip_margin=skip_margin,
        fast_memory_mode=fast_memory_mode,
        initial_memory=initial_memory,
    )


def train_one_task_hybrid(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    task_id: int,
    epoch: int,
    scaler=None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    threshold: float = 0.5,
    print_freq: int = 20,
    use_memory: bool = True,
    disable_memory: bool = False,
    memory_blend: float = 0.10,
    before_weight: float = 0.7,
    memory_after_weight: float = 1.0,
    memory_stability_weight: float = 1e-4,
    improve_weight: float = 0.1,
    improve_margin: float = 0.001,
    skip_memory_if_hurts: bool = True,
    skip_margin: float = 0.002,
    fast_memory_mode: str = "batch",
    initial_memory=None,
):
    return _train_epoch_impl(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=epoch,
        stage_name=f"NBM-Hybrid-Task{task_id}",
        scaler=scaler,
        use_amp=use_amp,
        grad_clip=grad_clip,
        threshold=threshold,
        print_freq=print_freq,
        use_memory=use_memory,
        disable_memory=disable_memory,
        memory_blend=memory_blend,
        before_weight=before_weight,
        memory_after_weight=memory_after_weight,
        memory_stability_weight=memory_stability_weight,
        improve_weight=improve_weight,
        improve_margin=improve_margin,
        skip_memory_if_hurts=skip_memory_if_hurts,
        skip_margin=skip_margin,
        fast_memory_mode=fast_memory_mode,
        initial_memory=initial_memory,
    )
