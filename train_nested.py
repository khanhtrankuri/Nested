import copy
import json
import os

import torch
from torch.amp import GradScaler

from data import build_dataloaders, build_replay_task_loaders
from model import NestedLitePolypModel
from loss import HybridSegLoss
from engine import train_one_task_nested, validate_nested, test_nested
from engine.eval_nested_modes import sweep_thresholds, evaluate_nested_mode


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def main():
    file_path = "datasets/Kvasir"
    save_root = "outputs/kvasir_nested_ultra"
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
        train_augmentation=True,
    )

    replay_plan = {
        2: {1: 0.20},
        3: {1: 0.15, 2: 0.15},
        4: {1: 0.10, 2: 0.10, 3: 0.10},
    }
    replay_loaders = build_replay_task_loaders(
        image_dir=meta_info["train_image_dir"],
        mask_dir=meta_info["train_mask_dir"],
        task_infos=meta_info["task_infos"],
        image_size=(358, 358),
        batch_size=8,
        num_workers=4,
        replay_plan=replay_plan,
    )

    model = NestedLitePolypModel(
        backbone_name="unet_nl",
        in_channels=3,
        out_channels=1,
        base_channels=32,
        bilinear=True,
        memory_dim=96,
        updater_hidden_dim=256,
        use_gate=True,
        fast_init_std=5e-2,
        slow_init_std=1e-2,
    ).to(device)

    criterion = HybridSegLoss(bce_weight=0.25, dice_weight=0.35, ft_weight=0.40, smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_dice = -1.0
    best_state = None
    history = []
    task_epochs_map = {1: 5, 2: 5, 3: 5, 4: 2}
    task_lr_map = {1: 1.0e-4, 2: 8.0e-5, 3: 5.0e-5, 4: 2.5e-5}
    task_after_weight_map = {1: 1.2, 2: 1.5, 3: 1.5, 4: 1.6}
    task_slow_momentum_map = {1: 0.05, 2: 0.05, 3: 0.03, 4: 0.01}
    slow_max_norm = 1.0
    global_epoch = 0

    for task_idx, task_loader in enumerate(replay_loaders, start=1):
        epochs_per_task = task_epochs_map.get(task_idx, 5)
        task_lr = task_lr_map.get(task_idx, 1e-5)
        set_optimizer_lr(optimizer, task_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_task, eta_min=max(task_lr * 0.1, 1e-6))

        print(f"\n========== START TASK {task_idx} ==========" )
        print(f"[Task {task_idx}] init_lr = {get_optimizer_lr(optimizer):.8f}")
        print(f"[Task {task_idx}] slow_memory_norm(before) = {model.slow_memory.norm().item():.6f}\n")

        best_task_val = -1.0
        best_task_memory_summary = None

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
                improve_weight=0.25,
                improve_margin=0.003,
                print_freq=10,
                initial_memory=None,
            )
            val_metrics = validate_nested(model=model, loader=val_loader, criterion=criterion, device=device, epoch=global_epoch, use_amp=True, print_freq=20)
            scheduler.step()
            current_lr = get_optimizer_lr(optimizer)
            history.append({
                "task_id": task_idx,
                "local_epoch": local_epoch,
                "global_epoch": global_epoch,
                "lr": current_lr,
                "slow_memory_norm": float(model.slow_memory.norm().item()),
                "train": train_metrics,
                "val": val_metrics,
            })
            print(
                f"\n[Task {task_idx} | Epoch {local_epoch}] lr={current_lr:.8f} | train_dice={train_metrics['dice']:.4f} | train_gain={train_metrics['dice_gain']:.4f} | val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f}\n"
            )
            if val_metrics["dice"] > best_task_val:
                best_task_val = val_metrics["dice"]
                best_task_memory_summary = memory_summary.detach().cpu()
            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(save_root, "best_model.pth"))

        if best_task_memory_summary is not None:
            task_summary = best_task_memory_summary.to(device)
            model.update_slow_memory(task_summary, momentum=task_slow_momentum_map[task_idx], max_norm=slow_max_norm)
            print(f"[Task {task_idx}] task_summary_norm = {task_summary.norm().item():.6f}")
            print(f"[Task {task_idx}] slow_memory_norm(after) = {model.slow_memory.norm().item():.6f}\n")

    with open(os.path.join(save_root, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best val dice: {best_val_dice:.4f}")
    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")
    model.load_state_dict(best_state)

    modes = ["adaptive_slow", "adaptive_fresh", "static_slow"]
    eval_summary = {}
    for mode in modes:
        best_val_result, _ = sweep_thresholds(model=model, loader=val_loader, criterion=criterion, device=device, thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65], use_amp=True, mode=mode)
        best_thr = best_val_result["threshold"]
        test_result = evaluate_nested_mode(model=model, loader=test_loader, criterion=criterion, device=device, threshold=best_thr, use_amp=True, mode=mode, save_dir=os.path.join(save_root, f"predictions_{mode}"))
        eval_summary[mode] = {"best_val": best_val_result, "test": test_result}
    with open(os.path.join(save_root, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

if __name__ == "__main__":
    main()