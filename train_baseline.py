import copy
import json
import os

import torch
from torch.cuda.amp import GradScaler

from data import build_dataloaders
from model import BaselinePolypModel
from loss import BCEDiceLoss
from engine import train_one_epoch, validate, test

def main():
    file_path = "datasets/Kvasir"
    save_root = "outputs/kvasir_baseline"
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path=file_path,
        image_size=(358, 358),
        num_tasks=4,
        val_size=0.2,
        batch_size=8,
        num_workers=4,
        seed=42,
        descending=True,
    )

    model = BaselinePolypModel(
        backbone_name="unet",
        in_channels=3,
        out_channels=1,
        base_channels=32,
        bilinear=True,
    ).to(device)

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
        T_max=50,
        eta_min=1e-6,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_dice = -1.0
    best_state = None
    history = []

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
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
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}\n"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(save_root, "best_model.pth"))

    with open(os.path.join(save_root, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best val dice: {best_dice:.4f}")

    model.load_state_dict(best_state)

    test_metrics = test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=os.path.join(save_root, "predictions"),
        threshold=0.5,
        use_amp=True,
    )

    with open(os.path.join(save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()