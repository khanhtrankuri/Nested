import argparse
import copy
import json
import os

import torch
from torch.cuda.amp import GradScaler

from data import build_dataloaders
from engine import train_one_epoch, validate, test
from loss import BCEDiceLoss
from model import BaselinePolypModel


def _build_model_kwargs():
    return {
        "backbone_name": "unet",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 32,
        "bilinear": True,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Train a stronger baseline U-Net.")
    parser.add_argument("--file-path", default="datasets/Kvasir")
    parser.add_argument("--save-root", default="outputs/kvasir_baseline_aug448")
    parser.add_argument("--image-size", type=int, nargs=2, default=(448, 448), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-augmentation", dest="train_augmentation", action="store_true")
    parser.add_argument("--no-train-augmentation", dest="train_augmentation", action="store_false")
    parser.set_defaults(train_augmentation=True)
    return parser


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

    criterion = BCEDiceLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_dice = -1.0
    best_state = None
    history = []

    print(
        f"Training baseline with image_size={image_size}, "
        f"train_augmentation={args.train_augmentation}, "
        f"batch_size={args.batch_size}"
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

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"\n[Epoch {epoch}] "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}\n"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "model_kwargs": model_kwargs,
                    "best_val_dice": best_dice,
                    "train_config": vars(args),
                },
                os.path.join(args.save_root, "best_model.pth"),
            )

    with open(os.path.join(args.save_root, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best val dice: {best_dice:.4f}")

    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(best_state)

    test_metrics = test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=os.path.join(args.save_root, "predictions"),
        threshold=0.5,
        use_amp=True,
    )

    with open(os.path.join(args.save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
