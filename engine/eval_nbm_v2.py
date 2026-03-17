import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def predict_with_tta(model, images):
    logits_list = []

    # gốc
    out = model(images, disable_memory=True)
    logits_list.append(out["logits"])

    # flip ngang
    x = torch.flip(images, dims=[3])
    out = model(x, disable_memory=True)
    logits_list.append(torch.flip(out["logits"], dims=[3]))

    # flip dọc
    x = torch.flip(images, dims=[2])
    out = model(x, disable_memory=True)
    logits_list.append(torch.flip(out["logits"], dims=[2]))

    # flip cả hai
    x = torch.flip(images, dims=[2, 3])
    out = model(x, disable_memory=True)
    logits_list.append(torch.flip(out["logits"], dims=[2, 3]))

    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return logits


@torch.no_grad()
def test_nbm_v2(
    model,
    loader,
    criterion,
    device,
    save_dir,
    threshold=0.45,
    use_amp=True,
    freeze_eval_memory=True,
    use_tta=True,
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n = 0

    for batch in tqdm(loader, desc="Test"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if use_tta:
            logits = predict_with_tta(model, images)
        else:
            out = model(images, disable_memory=True)
            logits = out["logits"]

        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        inter = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = ((2 * inter + 1.0) / (union + 1.0)).mean().item()

        iou = ((inter + 1.0) / ((preds + masks - preds * masks).sum(dim=(1, 2, 3)) + 1.0)).mean().item()

        precision = ((inter + 1.0) / (preds.sum(dim=(1, 2, 3)) + 1.0)).mean().item()
        recall = ((inter + 1.0) / (masks.sum(dim=(1, 2, 3)) + 1.0)).mean().item()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += dice * bs
        total_iou += iou * bs
        total_precision += precision * bs
        total_recall += recall * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "dice": total_dice / max(n, 1),
        "iou": total_iou / max(n, 1),
        "precision": total_precision / max(n, 1),
        "recall": total_recall / max(n, 1),
        "threshold": threshold,
        "tta": use_tta,
    }