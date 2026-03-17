
import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from data import build_dataloaders
from engine import train_one_epoch, validate
from loss import HybridIoULoss
from model import BaselinePolypModel


def _build_model_kwargs():
    return {
        "backbone_name": "unet",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 32,
        "bilinear": True,
        "norm_type": "group",
        "bottleneck_dropout": 0.1,
        "deep_supervision": True,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Train strong baseline for Kvasir with threshold sweep + TTA.")
    parser.add_argument("--file-path", default="datasets/Kvasir")
    parser.add_argument("--save-root", default="outputs/kvasir_baseline_iou85")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-augmentation", dest="train_augmentation", action="store_true")
    parser.add_argument("--no-train-augmentation", dest="train_augmentation", action="store_false")
    parser.set_defaults(train_augmentation=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-tta", action="store_true", default=True)
    parser.add_argument("--no-use-tta", dest="use_tta", action="store_false")
    return parser


def _safe_logit(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.clamp(min=eps, max=1.0 - eps)
    return torch.log(probs / (1.0 - probs))


@torch.no_grad()
def _tta_predict_probs(model, images: torch.Tensor, use_amp: bool = True, use_tta: bool = True) -> torch.Tensor:
    amp_enabled = use_amp and torch.cuda.is_available()

    def _forward(x):
        with autocast(device_type="cuda", enabled=amp_enabled):
            out = model(x)
            probs = torch.sigmoid(out["logits"])
        return probs

    probs_list = [_forward(images)]
    if use_tta:
        flips = [
            [3],      # hflip
            [2],      # vflip
            [2, 3],   # hvflip
        ]
        for dims in flips:
            imgs = torch.flip(images, dims=dims)
            probs = _forward(imgs)
            probs = torch.flip(probs, dims=dims)
            probs_list.append(probs)

    probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return probs


def _metrics_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    preds = (probs > threshold).float()
    targets = targets.float()
    dims = (1, 2, 3)

    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)

    pred_area = preds.sum(dim=dims)
    target_area = targets.sum(dim=dims)
    intersection = tp
    union = pred_area + target_area - intersection

    dice = (2.0 * intersection) / (pred_area + target_area + eps)
    iou = intersection / (union + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


@torch.no_grad()
def _evaluate_with_threshold(model, loader, criterion, device: str, threshold: float, use_tta: bool):
    model.eval()

    losses = []
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    amp_enabled = torch.cuda.is_available()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)
            logits = outputs["logits"]
            aux_logits = outputs.get("aux_logits", [])
            loss, _ = criterion(logits, masks, aux_logits=aux_logits, return_components=True)

        probs = _tta_predict_probs(model, images, use_amp=True, use_tta=use_tta)
        metric_dict = _metrics_from_probs(probs, masks, threshold=threshold)

        losses.append(float(loss.detach().item()))
        dice_scores.append(metric_dict["dice"])
        iou_scores.append(metric_dict["iou"])
        precision_scores.append(metric_dict["precision"])
        recall_scores.append(metric_dict["recall"])

    mean = lambda xs: float(sum(xs) / max(len(xs), 1))
    return {
        "loss": mean(losses),
        "dice": mean(dice_scores),
        "iou": mean(iou_scores),
        "precision": mean(precision_scores),
        "recall": mean(recall_scores),
        "threshold": float(threshold),
        "use_tta": bool(use_tta),
    }


@torch.no_grad()
def _threshold_sweep(model, loader, criterion, device: str, use_tta: bool):
    thresholds = [round(x, 2) for x in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]]
    all_results = []

    for thr in thresholds:
        result = _evaluate_with_threshold(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            threshold=thr,
            use_tta=use_tta,
        )
        all_results.append(result)
        print(
            f"[Threshold Sweep] thr={thr:.2f} | "
            f"dice={result['dice']:.4f} | "
            f"iou={result['iou']:.4f}"
        )

    best = max(all_results, key=lambda x: (x["iou"], x["dice"]))
    return best, all_results


def main():
    args = _build_parser().parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)

    _, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path=args.file_path,
        image_size=image_size,
        num_tasks=args.num_tasks,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        descending=True,
        train_augmentation=args.train_augmentation,
    )

    with open(os.path.join(args.save_root, "meta_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    model_kwargs = _build_model_kwargs()
    model = BaselinePolypModel(**model_kwargs).to(device)

    criterion = HybridIoULoss(
        bce_weight=0.4,
        dice_weight=0.3,
        lovasz_weight=0.3,
        smooth=1.0,
        aux_weights=(0.30, 0.15),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_iou = -1.0
    best_state = None
    history = []

    print(
        f"Training baseline with image_size={image_size}, "
        f"train_augmentation={args.train_augmentation}, "
        f"batch_size={args.batch_size}, "
        f"use_tta={args.use_tta}"
    )

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader_full,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=True,
            grad_clip=1.0,
            print_freq=20,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=True,
            print_freq=20,
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(
            f"\n[Epoch {epoch}] "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}\n"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "model_kwargs": model_kwargs,
                    "best_val_iou": best_iou,
                    "train_config": vars(args),
                },
                os.path.join(args.save_root, "best_model.pth"),
            )

        with open(os.path.join(args.save_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best val IoU: {best_iou:.4f}")

    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(best_state)

    best_thr_result, threshold_results = _threshold_sweep(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_tta=args.use_tta,
    )

    with open(os.path.join(args.save_root, "threshold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": best_thr_result,
                "all": threshold_results,
            },
            f,
            indent=2,
        )

    test_metrics = _evaluate_with_threshold(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=best_thr_result["threshold"],
        use_tta=args.use_tta,
    )

    with open(os.path.join(args.save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(
        f"[Test] thr={test_metrics['threshold']:.2f} | "
        f"dice={test_metrics['dice']:.4f} | "
        f"iou={test_metrics['iou']:.4f} | "
        f"precision={test_metrics['precision']:.4f} | "
        f"recall={test_metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
