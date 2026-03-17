import copy
import json
import os

import torch
from torch.amp import GradScaler

from data import build_dataloaders
from model import NBMPolypNetV3   # sửa đúng tên class của bạn
from engine import validate_nbm_v2, train_one_epoch_base, test_nbm_v2  # sửa đúng tên hàm của bạn
from loss.precision_combo_loss import PrecisionComboLoss


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def main():
    file_path = "datasets/Kvasir"
    save_root = "outputs/kvasir_accuracy_first"
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path=file_path,
        image_size=(384, 384),   # tăng nhẹ độ phân giải
        num_tasks=4,
        val_size=0.2,
        batch_size=8,
        num_workers=4,
        seed=42,
        descending=True,
        train_augmentation=True,
    )

    model = NBMPolypNetV3(
        in_channels=3,
        out_channels=1,
        base_channels=32,
        memory_dim=64,
        num_prototypes=4,
        bilinear=True,
    ).to(device)

    criterion = PrecisionComboLoss(
        bce_w=0.35,
        ft_w=0.40,
        dice_w=0.20,
        edge_w=0.05,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=5e-4,
    )

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_dice = -1.0
    best_threshold = 0.5
    best_state = None
    history = []

    # ---- Accuracy-first: chỉ train phần có ích ----
    warmup_epochs = 12
    task_epochs_map = {
        1: 2,
        2: 3,
        3: 0,   # bỏ
        4: 0,   # bỏ
    }
    task_lr_map = {
        1: 6e-5,
        2: 4e-5,
    }

    # ------------------ WARMUP FULL TRAIN ------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=warmup_epochs,
        eta_min=1e-5,
    )

    global_epoch = 0

    for epoch in range(1, warmup_epochs + 1):
        global_epoch += 1

        # tắt memory hoàn toàn
        train_metrics = train_one_epoch_base(
            model=model,
            loader=train_loader_full,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=True,
            grad_clip=1.0,
            disable_memory=True,
        )

        val_metrics = validate_nbm_v2(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=global_epoch,
            use_amp=True,
            print_freq=20,
            freeze_eval_memory=True,
            threshold_sweep=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        )

        scheduler.step()
        current_lr = get_optimizer_lr(optimizer)

        history.append({
            "stage": "warmup",
            "epoch": epoch,
            "global_epoch": global_epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics["default"],
            "val_threshold_sweep": val_metrics["threshold_sweep"],
        })

        sweep_dice = val_metrics["threshold_sweep"]["dice"]
        sweep_thr = val_metrics["threshold_sweep"]["best_threshold"]

        if sweep_dice > best_val_dice:
            best_val_dice = sweep_dice
            best_threshold = sweep_thr
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(save_root, "best_model.pth"))

        print(
            f"[Warmup {epoch}] lr={current_lr:.8f} | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"best_val_dice={sweep_dice:.4f} @thr={sweep_thr}"
        )

    # ------------------ TASK 1 / TASK 2 ONLY ------------------
    for task_idx, task_loader in enumerate(train_task_loaders, start=1):
        epochs_per_task = task_epochs_map.get(task_idx, 0)
        if epochs_per_task <= 0:
            continue

        task_lr = task_lr_map[task_idx]
        set_optimizer_lr(optimizer, task_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs_per_task,
            eta_min=max(task_lr * 0.2, 1e-6),
        )

        for local_epoch in range(1, epochs_per_task + 1):
            global_epoch += 1

            train_metrics = train_one_epoch_base(
                model=model,
                loader=task_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=True,
                grad_clip=1.0,
                disable_memory=True,   # vẫn tắt memory
            )

            val_metrics = validate_nbm_v2(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                use_amp=True,
                print_freq=20,
                freeze_eval_memory=True,
                threshold_sweep=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
            )

            scheduler.step()
            current_lr = get_optimizer_lr(optimizer)

            history.append({
                "stage": f"task_{task_idx}",
                "task_id": task_idx,
                "local_epoch": local_epoch,
                "global_epoch": global_epoch,
                "lr": current_lr,
                "use_memory": False,
                "train": train_metrics,
                "val": val_metrics["default"],
                "val_threshold_sweep": val_metrics["threshold_sweep"],
            })

            sweep_dice = val_metrics["threshold_sweep"]["dice"]
            sweep_thr = val_metrics["threshold_sweep"]["best_threshold"]

            if sweep_dice > best_val_dice:
                best_val_dice = sweep_dice
                best_threshold = sweep_thr
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(save_root, "best_model.pth"))

            print(
                f"[Task {task_idx} | Epoch {local_epoch}] "
                f"lr={current_lr:.8f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"best_val_dice={sweep_dice:.4f} @thr={sweep_thr}"
            )

    # ------------------ JOINT FINETUNE NGẮN ------------------
    model.load_state_dict(best_state)
    set_optimizer_lr(optimizer, 2e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4,
        eta_min=1e-6,
    )

    for ft_epoch in range(1, 5):
        global_epoch += 1

        train_metrics = train_one_epoch_base(
            model=model,
            loader=train_loader_full,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=True,
            grad_clip=1.0,
            disable_memory=True,
        )

        val_metrics = validate_nbm_v2(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=global_epoch,
            use_amp=True,
            print_freq=20,
            freeze_eval_memory=True,
            threshold_sweep=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        )

        scheduler.step()
        current_lr = get_optimizer_lr(optimizer)

        history.append({
            "stage": "joint_finetune",
            "epoch": ft_epoch,
            "global_epoch": global_epoch,
            "lr": current_lr,
            "use_memory": False,
            "train": train_metrics,
            "val": val_metrics["default"],
            "val_threshold_sweep": val_metrics["threshold_sweep"],
        })

        sweep_dice = val_metrics["threshold_sweep"]["dice"]
        sweep_thr = val_metrics["threshold_sweep"]["best_threshold"]

        if sweep_dice > best_val_dice:
            best_val_dice = sweep_dice
            best_threshold = sweep_thr
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(save_root, "best_model.pth"))

        print(
            f"[Joint FT {ft_epoch}] lr={current_lr:.8f} | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"best_val_dice={sweep_dice:.4f} @thr={sweep_thr}"
        )

    with open(os.path.join(save_root, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best val sweep dice: {best_val_dice:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")

    model.load_state_dict(best_state)

    test_metrics = test_nbm_v2(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=os.path.join(save_root, "predictions"),
        threshold=best_threshold,
        use_amp=True,
        freeze_eval_memory=True,
        use_tta=True,
    )

    with open(os.path.join(save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()