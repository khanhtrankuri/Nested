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


def _blend_outputs(base_outputs: Dict[str, torch.Tensor], memory_outputs: Dict[str, torch.Tensor], blend: float) -> Dict[str, torch.Tensor]:
    blend = float(max(0.0, min(1.0, blend)))
    if blend <= 0.0:
        return base_outputs
    merged = dict(memory_outputs)
    merged['logits'] = (1.0 - blend) * base_outputs['logits'] + blend * memory_outputs['logits']
    merged['coarse_logits'] = (1.0 - blend) * base_outputs['coarse_logits'] + blend * memory_outputs['coarse_logits']
    merged['aux_logits'] = (1.0 - blend) * base_outputs['aux_logits'] + blend * memory_outputs['aux_logits']
    merged['edge_logits'] = (1.0 - blend) * base_outputs['edge_logits'] + blend * memory_outputs['edge_logits']
    return merged


def train_epoch_warmup_v3(model, loader, optimizer, criterion, device: str, epoch: int, scaler=None, use_amp: bool = True, grad_clip: Optional[float] = 1.0, threshold: float = 0.5, print_freq: int = 20):
    model.train()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = use_amp and (device_type == 'cuda')

    loss_meter = AverageMeter(); dice_meter = AverageMeter(); iou_meter = AverageMeter(); precision_meter = AverageMeter(); recall_meter = AverageMeter(); batch_time_meter = AverageMeter()
    start_time = time.time()
    for step, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        bs = images.size(0)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(images, memory=None, use_memory=False)
            loss, loss_dict = criterion(outputs, masks, return_components=True)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        metrics = compute_segmentation_metrics(outputs['logits'].detach(), masks, threshold=threshold)
        loss_meter.update(loss_dict['loss_total'], bs); dice_meter.update(metrics['dice'], bs); iou_meter.update(metrics['iou'], bs); precision_meter.update(metrics['precision'], bs); recall_meter.update(metrics['recall'], bs)
        batch_time_meter.update(time.time() - start_time); start_time = time.time()
        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(f"[NBM-v3-Warmup][Epoch {epoch}] Step {step+1}/{len(loader)} | loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f} | precision={precision_meter.avg:.4f} | recall={recall_meter.avg:.4f} | time={batch_time_meter.avg:.3f}s")
    return {'loss': loss_meter.avg, 'dice': dice_meter.avg, 'iou': iou_meter.avg, 'precision': precision_meter.avg, 'recall': recall_meter.avg}


def train_one_task_nbm_v3(model, loader, optimizer, criterion, device: str, task_id: int, epoch: int, scaler=None, use_amp: bool = True, grad_clip: Optional[float] = 1.0, memory_after_weight: float = 1.0, memory_stability_weight: float = 1e-4, improve_weight: float = 0.2, improve_margin: float = 0.002, before_weight: float = 0.6, threshold: float = 0.5, print_freq: int = 20, use_memory: bool = True, memory_blend: float = 0.2, keep_batch_memory: bool = False, skip_memory_if_hurts: bool = True, skip_margin: float = 0.003, distill_weight: float = 0.15):
    model.train()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = use_amp and (device_type == 'cuda')

    total_loss_meter = AverageMeter(); seg_before_meter = AverageMeter(); seg_after_meter = AverageMeter(); improve_loss_meter = AverageMeter(); mem_reg_meter = AverageMeter(); mem_delta_meter = AverageMeter(); mem_norm_meter = AverageMeter(); dice_before_meter = AverageMeter(); dice_after_meter = AverageMeter(); dice_gain_meter = AverageMeter(); iou_after_meter = AverageMeter(); precision_after_meter = AverageMeter(); recall_after_meter = AverageMeter(); batch_time_meter = AverageMeter()
    start_time = time.time()
    running_memory = None
    memory_summaries = []

    for step, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        bs = images.size(0)
        memory = model.init_memory(bs, device, images.dtype) if (running_memory is None or not keep_batch_memory or next(iter(running_memory.values())).size(0) != bs) else {k: v.to(device=device, dtype=images.dtype) for k, v in running_memory.items()}
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs_base = model(images, memory=None, use_memory=False)
            seg_loss_before, seg_before_dict = criterion(outputs_base, masks, return_components=True)

            if use_memory:
                outputs_seed = model(images, memory=memory, use_memory=True)
                proposed_memory = model.compute_updated_memory(
                    memory_features={k: v.detach() for k, v in outputs_seed['memory_features'].items()},
                    memory={k: v.detach() for k, v in memory.items()},
                    update_signals={k: v.detach() for k, v in outputs_seed['update_signals'].items()},
                    attention_cache={k: v for k, v in outputs_seed['attention_cache'].items()},
                )
                outputs_mem = model(images, memory=proposed_memory, use_memory=True)
                outputs_after = _blend_outputs(outputs_base, outputs_mem, memory_blend)
                seg_loss_after, seg_after_dict = criterion(outputs_after, masks, return_components=True)
                memory_reg = _memory_reg(memory, proposed_memory)
                improve_loss = torch.relu(seg_loss_after - seg_loss_before + improve_margin)
                distill = torch.nn.functional.mse_loss(torch.sigmoid(outputs_after['logits']), torch.sigmoid(outputs_base['logits']).detach())
                total_loss = before_weight * seg_loss_before + memory_after_weight * seg_loss_after + improve_weight * improve_loss + memory_stability_weight * memory_reg + distill_weight * distill
                hurt_mask = bool((seg_loss_after.detach() > seg_loss_before.detach() + skip_margin).item())
                if skip_memory_if_hurts and hurt_mask:
                    total_loss = seg_loss_before + 0.05 * distill
                    outputs_after = outputs_base
                    proposed_memory = memory
                    seg_after_dict = seg_before_dict
                    memory_reg = torch.zeros_like(memory_reg)
                    improve_loss = torch.zeros_like(improve_loss)
            else:
                proposed_memory = memory
                outputs_after = outputs_base
                seg_loss_after, seg_after_dict = seg_loss_before, seg_before_dict
                memory_reg = torch.zeros((), device=images.device, dtype=images.dtype)
                improve_loss = torch.zeros((), device=images.device, dtype=images.dtype)
                distill = torch.zeros((), device=images.device, dtype=images.dtype)
                total_loss = seg_loss_before

        if scaler is not None and amp_enabled:
            scaler.scale(total_loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        mem_delta_value = _memory_delta(memory, proposed_memory)
        mem_norm_value = _memory_norm(proposed_memory)
        with torch.no_grad():
            if use_memory:
                running_memory = {k: v.detach() for k, v in proposed_memory.items()} if keep_batch_memory else None
                memory_summaries.append(model.summarize_memory(proposed_memory))
        metric_before = compute_segmentation_metrics(outputs_base['logits'].detach(), masks, threshold=threshold)
        metric_after = compute_segmentation_metrics(outputs_after['logits'].detach(), masks, threshold=threshold)

        total_loss_meter.update(float(total_loss.detach().item()), bs); seg_before_meter.update(seg_before_dict['loss_total'], bs); seg_after_meter.update(seg_after_dict['loss_total'], bs); improve_loss_meter.update(float(improve_loss.detach().item()), bs); mem_reg_meter.update(float(memory_reg.detach().item()), bs); mem_delta_meter.update(mem_delta_value, bs); mem_norm_meter.update(mem_norm_value, bs); dice_before_meter.update(metric_before['dice'], bs); dice_after_meter.update(metric_after['dice'], bs); dice_gain_meter.update(metric_after['dice'] - metric_before['dice'], bs); iou_after_meter.update(metric_after['iou'], bs); precision_after_meter.update(metric_after['precision'], bs); recall_after_meter.update(metric_after['recall'], bs)
        batch_time_meter.update(time.time() - start_time); start_time = time.time()
        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(f"[NBM-v3-Train][Task {task_id}][Epoch {epoch}] Step {step+1}/{len(loader)} | loss={total_loss_meter.avg:.4f} | seg_before={seg_before_meter.avg:.4f} | seg_after={seg_after_meter.avg:.4f} | improve={improve_loss_meter.avg:.6f} | mem_reg={mem_reg_meter.avg:.8e} | mem_delta={mem_delta_meter.avg:.8e} | mem_norm={mem_norm_meter.avg:.8e} | dice_before={dice_before_meter.avg:.4f} | dice_after={dice_after_meter.avg:.4f} | dice_gain={dice_gain_meter.avg:.4f} | iou_after={iou_after_meter.avg:.4f} | precision_after={precision_after_meter.avg:.4f} | recall_after={recall_after_meter.avg:.4f} | time={batch_time_meter.avg:.3f}s")

    if len(memory_summaries) == 0:
        memory_summaries = [model.summarize_memory(model.init_memory(1, device))]
    metrics = {'loss': total_loss_meter.avg, 'seg_before': seg_before_meter.avg, 'seg_after': seg_after_meter.avg, 'improve_loss': improve_loss_meter.avg, 'mem_reg': mem_reg_meter.avg, 'mem_delta': mem_delta_meter.avg, 'mem_norm': mem_norm_meter.avg, 'dice_before': dice_before_meter.avg, 'dice': dice_after_meter.avg, 'dice_gain': dice_gain_meter.avg, 'iou': iou_after_meter.avg, 'precision': precision_after_meter.avg, 'recall': recall_after_meter.avg}
    return metrics, _stack_memory_summaries(memory_summaries)
