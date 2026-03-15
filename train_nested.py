import copy
import json
import os

import torch
from torch.amp import GradScaler

from data import build_dataloaders
from model import NestedLitePolypModel
from loss import BCEDiceLoss
from engine import train_one_task_nested, validate_nested, test_nested


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def main():
    file_path = "datasets/Kvasir"
    save_root = "outputs/kvasir_nested_dual_memory_bottleneck"
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path=file_path,
        image_size=(358, 358),
        num_tasks=4,
        val_size=0.2,
        batch_size=32,
        num_workers=4,
        seed=42,
        descending=True,
    )

    model_kwargs = {
        "backbone_name": "unet_nl",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 32,
        "bilinear": True,
        "memory_dim": 64,
        "updater_hidden_dim": 128,
        "use_gate": True,
        "fast_init_std": 5e-2,
        "slow_init_std": 1e-2,
    }
    model = NestedLitePolypModel(**model_kwargs).to(device)

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

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_dice = -1.0
    best_state = None
    history = []

    task_epochs_map = {1: 5, 2: 5, 3: 5, 4: 1}
    task_lr_map = {1: 1.0e-4, 2: 8.0e-5, 3: 5.0e-5, 4: 1.0e-5}
    task_after_weight_map = {1: 1.2, 2: 1.5, 3: 1.5, 4: 1.5}
    task_slow_momentum_map = {1: 0.05, 2: 0.05, 3: 0.03, 4: 0.00}

    slow_max_norm = 1.0
    global_epoch = 0

    for task_idx, task_loader in enumerate(train_task_loaders, start=1):
        epochs_per_task = task_epochs_map.get(task_idx, 5)
        task_lr = task_lr_map.get(task_idx, 1.0e-5)

        set_optimizer_lr(optimizer, task_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs_per_task,
            eta_min=max(task_lr * 0.1, 1e-6),
        )

        print(f"\n========== START TASK {task_idx} ==========\n")
        print(f"[Task {task_idx}] init_lr = {get_optimizer_lr(optimizer):.8f}")
        print(f"[Task {task_idx}] slow_memory_norm(before) = {model.slow_memory.norm().item():.6f}\n")

        epoch_memory_summaries = []

        for local_epoch in range(1, epochs_per_task + 1):
            global_epoch += 1

            train_metrics, memory_summary = train_one_task_nested(
                model=model,
                loader=task_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task_id=task_idx,
                epoch=local_epoch,
                scaler=scaler,
                use_amp=True,
                grad_clip=1.0,
                memory_after_weight=task_after_weight_map[task_idx],
                memory_stability_weight=1e-4,
                improve_weight=0.2,
                improve_margin=0.005,
                before_weight=0.3,
                print_freq=10,
                initial_memory=None,
            )

            epoch_memory_summaries.append(memory_summary.detach().cpu())

            val_metrics = validate_nested(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                use_amp=True,
                print_freq=20,
            )

            scheduler.step()
            current_lr = get_optimizer_lr(optimizer)

            history.append(
                {
                    "task_id": task_idx,
                    "local_epoch": local_epoch,
                    "global_epoch": global_epoch,
                    "lr": current_lr,
                    "slow_memory_norm": float(model.slow_memory.norm().item()),
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            print(
                f"\n[Task {task_idx} | Epoch {local_epoch}] "
                f"lr={current_lr:.8f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"train_dice_gain={train_metrics['dice_gain']:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f}\n"
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "state_dict": best_state,
                        "model_kwargs": model_kwargs,
                        "best_val_dice": best_val_dice,
                    },
                    os.path.join(save_root, "best_model.pth"),
                )

        task_summary = torch.stack(epoch_memory_summaries, dim=0).mean(dim=0).to(device)
        task_summary_norm = task_summary.norm().item()

        print(f"[Task {task_idx}] task_summary_norm = {task_summary_norm:.6f}")

        model.update_slow_memory(
            task_summary,
            momentum=task_slow_momentum_map[task_idx],
            max_norm=slow_max_norm,
        )

        print(f"[Task {task_idx}] slow_memory_norm(after) = {model.slow_memory.norm().item():.6f}\n")

    with open(os.path.join(save_root, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best val dice: {best_val_dice:.4f}")

    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(best_state)

    test_metrics = test_nested(
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
