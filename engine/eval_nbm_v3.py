from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.amp import autocast

from metrics.segmentation_metrics import compute_segmentation_metrics
from engine.train_nbm_v3 import _blend_outputs


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


class ThresholdMeter:
    def __init__(self, thresholds: Iterable[float]):
        self.thresholds = [float(t) for t in thresholds]
        self.dice = {t: AverageMeter() for t in self.thresholds}
        self.iou = {t: AverageMeter() for t in self.thresholds}
        self.precision = {t: AverageMeter() for t in self.thresholds}
        self.recall = {t: AverageMeter() for t in self.thresholds}

    def update(self, logits: torch.Tensor, targets: torch.Tensor, batch_size: int):
        for t in self.thresholds:
            metric = compute_segmentation_metrics(logits, targets, threshold=t)
            self.dice[t].update(metric['dice'], batch_size)
            self.iou[t].update(metric['iou'], batch_size)
            self.precision[t].update(metric['precision'], batch_size)
            self.recall[t].update(metric['recall'], batch_size)

    def best(self):
        best_t = max(self.thresholds, key=lambda t: self.dice[t].avg)
        return {'best_threshold': best_t, 'dice': self.dice[best_t].avg, 'iou': self.iou[best_t].avg, 'precision': self.precision[best_t].avg, 'recall': self.recall[best_t].avg, 'all_thresholds': {f"{t:.2f}": {'dice': self.dice[t].avg, 'iou': self.iou[t].avg, 'precision': self.precision[t].avg, 'recall': self.recall[t].avg} for t in self.thresholds}}


@torch.no_grad()
def validate_nbm_v3(model, loader, criterion, device: str, epoch: int = 0, use_amp: bool = True, threshold: float = 0.5, threshold_sweep: Optional[Iterable[float]] = None, freeze_eval_memory: bool = True, use_memory: bool = True, memory_blend: float = 0.2, print_freq: int = 20) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    model.eval()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = use_amp and (device_type == 'cuda')
    loss_meter = AverageMeter(); dice_meter = AverageMeter(); iou_meter = AverageMeter(); precision_meter = AverageMeter(); recall_meter = AverageMeter(); sweep_meter = ThresholdMeter(threshold_sweep) if threshold_sweep is not None else None
    memory = None
    for step, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        bs = images.size(0)
        if memory is None or next(iter(memory.values())).size(0) != bs:
            memory = model.init_memory(batch_size=bs, device=device, dtype=images.dtype)
        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs_base = model(images, memory=None, use_memory=False)
            if use_memory:
                outputs_mem = model(images, memory=memory, use_memory=True)
                outputs = _blend_outputs(outputs_base, outputs_mem, memory_blend)
            else:
                outputs = outputs_base
            loss, loss_dict = criterion(outputs, masks, return_components=True)
        metric_dict = compute_segmentation_metrics(outputs['logits'], masks, threshold=threshold)
        if sweep_meter is not None:
            sweep_meter.update(outputs['logits'], masks, bs)
        if use_memory and not freeze_eval_memory:
            memory = model.update_memory(memory_features={k: v.detach() for k, v in outputs_mem['memory_features'].items()}, memory={k: v.detach() for k, v in memory.items()}, update_signals={k: v.detach() for k, v in outputs_mem['update_signals'].items()}, attention_cache={k: v for k, v in outputs_mem['attention_cache'].items()})
        loss_meter.update(loss_dict['loss_total'], bs); dice_meter.update(metric_dict['dice'], bs); iou_meter.update(metric_dict['iou'], bs); precision_meter.update(metric_dict['precision'], bs); recall_meter.update(metric_dict['recall'], bs)
        if (step + 1) % print_freq == 0 or (step + 1) == len(loader):
            print(f"[NBM-v3-Val][Epoch {epoch}] Step {step+1}/{len(loader)} | loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f} | precision={precision_meter.avg:.4f} | recall={recall_meter.avg:.4f}")
    metrics = {'loss': loss_meter.avg, 'dice': dice_meter.avg, 'iou': iou_meter.avg, 'precision': precision_meter.avg, 'recall': recall_meter.avg}
    return metrics, (sweep_meter.best() if sweep_meter is not None else None)


def _save_binary_mask(mask_tensor: torch.Tensor, save_path: str):
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(save_path)


@torch.no_grad()
def test_nbm_v3(model, loader, criterion, device: str, save_dir: Optional[str] = None, threshold: float = 0.5, use_amp: bool = True, freeze_eval_memory: bool = True, use_memory: bool = True, memory_blend: float = 0.2) -> Dict[str, float]:
    model.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = use_amp and (device_type == 'cuda')
    loss_meter = AverageMeter(); dice_meter = AverageMeter(); iou_meter = AverageMeter(); precision_meter = AverageMeter(); recall_meter = AverageMeter()
    memory = None
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        file_names = batch['file_name']
        bs = images.size(0)
        if memory is None or next(iter(memory.values())).size(0) != bs:
            memory = model.init_memory(batch_size=bs, device=device, dtype=images.dtype)
        with autocast(device_type=device_type, enabled=amp_enabled):
            outputs_base = model(images, memory=None, use_memory=False)
            if use_memory:
                outputs_mem = model(images, memory=memory, use_memory=True)
                outputs = _blend_outputs(outputs_base, outputs_mem, memory_blend)
            else:
                outputs = outputs_base
            loss, loss_dict = criterion(outputs, masks, return_components=True)
        metric_dict = compute_segmentation_metrics(outputs['logits'], masks, threshold=threshold)
        if use_memory and not freeze_eval_memory:
            memory = model.update_memory(memory_features={k: v.detach() for k, v in outputs_mem['memory_features'].items()}, memory={k: v.detach() for k, v in memory.items()}, update_signals={k: v.detach() for k, v in outputs_mem['update_signals'].items()}, attention_cache={k: v for k, v in outputs_mem['attention_cache'].items()})
        loss_meter.update(loss_dict['loss_total'], bs); dice_meter.update(metric_dict['dice'], bs); iou_meter.update(metric_dict['iou'], bs); precision_meter.update(metric_dict['precision'], bs); recall_meter.update(metric_dict['recall'], bs)
        if save_dir is not None:
            probs = torch.sigmoid(outputs['logits']); preds = (probs > threshold).float()
            for i in range(bs):
                _save_binary_mask(preds[i, 0], os.path.join(save_dir, file_names[i]))
    results = {'loss': loss_meter.avg, 'dice': dice_meter.avg, 'iou': iou_meter.avg, 'precision': precision_meter.avg, 'recall': recall_meter.avg, 'threshold': threshold, 'freeze_eval_memory': freeze_eval_memory, 'use_memory': use_memory, 'memory_blend': memory_blend}
    print(f"[NBM-v3-Test] loss={results['loss']:.4f} | dice={results['dice']:.4f} | iou={results['iou']:.4f} | precision={results['precision']:.4f} | recall={results['recall']:.4f} | threshold={threshold:.2f} | memory_blend={memory_blend:.2f}")
    return results
