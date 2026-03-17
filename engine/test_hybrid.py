from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.amp import autocast

from engine.train_one_epoch_hybrid import AverageMeter, _blend_outputs, _expand_memory_for_batch
from metrics.segmentation_metrics import compute_segmentation_metrics


def _save_binary_mask(mask_tensor: torch.Tensor, save_path: str):
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(save_path)


@torch.no_grad()
def _predict_logits_tta(model, images: torch.Tensor, memory, use_memory: bool, disable_memory: bool):
    variants = [(), (3,), (2,), (2, 3)]
    logits_list = []
    for dims in variants:
        if dims:
            current = torch.flip(images, dims=list(dims))
        else:
            current = images
        outputs = model(current, memory=memory, use_memory=use_memory, disable_memory=disable_memory)
        logits = outputs["logits"]
        if dims:
            logits = torch.flip(logits, dims=list(dims))
        logits_list.append(logits)
    return torch.stack(logits_list, dim=0).mean(dim=0)


@torch.no_grad()
def test_hybrid(
    model,
    loader,
    criterion,
    device: str,
    save_dir: Optional[str] = None,
    threshold: float = 0.5,
    use_amp: bool = True,
    freeze_eval_memory: bool = True,
    use_memory: bool = False,
    disable_memory: bool = False,
    memory_blend: float = 0.10,
    use_tta: bool = True,
) -> Dict[str, float]:
    model.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

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

    eval_memory_seed = None
    if memory_active:
        eval_memory_seed = model.summarize_memory(model.init_memory(batch_size=1, device=device, noise_std=0.0))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        file_names = batch["file_name"]
        batch_size = images.size(0)

        batch_memory = None
        if memory_active:
            batch_memory = _expand_memory_for_batch(eval_memory_seed, model, batch_size, device, images.dtype)

        with autocast(device_type=device_type, enabled=amp_enabled):
            base_outputs = model(images, memory=None, use_memory=False, disable_memory=True)
            if memory_active:
                memory_outputs = model(images, memory=batch_memory, use_memory=True, disable_memory=False)
                outputs = _blend_outputs(base_outputs, memory_outputs, memory_blend)
            else:
                memory_outputs = None
                outputs = base_outputs
            loss, loss_dict = criterion(outputs, masks, return_components=True)

            if use_tta:
                if memory_active:
                    tta_logits = _predict_logits_tta(model, images, batch_memory, use_memory=True, disable_memory=False)
                    base_tta = _predict_logits_tta(model, images, None, use_memory=False, disable_memory=True)
                    logits = (1.0 - memory_blend) * base_tta + memory_blend * tta_logits
                else:
                    logits = _predict_logits_tta(model, images, None, use_memory=False, disable_memory=True)
            else:
                logits = outputs["logits"]

        metric = compute_segmentation_metrics(logits, masks, threshold=threshold)
        meters["loss"].update(loss_dict["loss_total"], batch_size)
        meters["dice"].update(metric["dice"], batch_size)
        meters["iou"].update(metric["iou"], batch_size)
        meters["precision"].update(metric["precision"], batch_size)
        meters["recall"].update(metric["recall"], batch_size)

        if memory_active and not freeze_eval_memory:
            eval_memory_seed = model.summarize_memory(
                model.compute_updated_memory(
                    memory_features={k: v.detach() for k, v in memory_outputs["memory_features"].items()},
                    memory={k: v.detach() for k, v in batch_memory.items()},
                    update_signals={k: v.detach() for k, v in memory_outputs["update_signals"].items()},
                    attention_cache={k: v for k, v in memory_outputs["attention_cache"].items()},
                )
            )

        if save_dir is not None:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            for index in range(batch_size):
                _save_binary_mask(preds[index, 0], os.path.join(save_dir, file_names[index]))

    results = {key: meter.avg for key, meter in meters.items()}
    results.update(
        {
            "threshold": float(threshold),
            "use_tta": bool(use_tta),
            "memory_active": float(memory_active),
            "freeze_eval_memory": bool(freeze_eval_memory),
            "memory_blend": float(memory_blend),
        }
    )
    print(
        f"[NBM-Hybrid-Test] loss={results['loss']:.4f} | dice={results['dice']:.4f} | "
        f"iou={results['iou']:.4f} | precision={results['precision']:.4f} | "
        f"recall={results['recall']:.4f} | threshold={threshold:.2f}"
    )
    return results
