from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler

from data import build_dataloaders
from engine.train_one_task_nbm import train_one_task_nbm
from engine.validate_nbm import validate_nbm
from engine.test_nbm import test_nbm
from loss.nbm_losses import NBMPolypLoss
from model.nbm_polyp_model import NBMPolypNet


def _parse_csv_numbers(value: str, cast=int) -> List:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _build_parser():
    parser = argparse.ArgumentParser(description="Train NBM-PolypNet on Kvasir-style data.")
    parser.add_argument("--file-path", default="datasets/Kvasir")
    parser.add_argument("--save-root", default="outputs/kvasir_nbm_polypnet")
    parser.add_argument("--image-size", type=int, nargs=2, default=(352, 352), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--updater-hidden-dim", type=int, default=128)
    parser.add_argument("--task-epochs", default="6,5,4,3")
    parser.add_argument("--task-lrs", default="1e-4,8e-5,5e-5,2e-5")
    parser.add_argument("--task-slow-momentum", default="0.05,0.05,0.03,0.02")
    parser.add_argument("--memory-after-weight", type=float, default=1.5)
    parser.add_argument("--before-weight", type=float, default=0.3)
    parser.add_argument("--improve-weight", type=float, default=0.2)
    parser.add_argument("--improve-margin", type=float, default=0.005)
    parser.add_argument("--memory-stability-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-augmentation", dest="train_augmentation", action="store_true")
    parser.add_argument("--no-train-augmentation", dest="train_augmentation", action="store_false")
    parser.set_defaults(train_augmentation=True)
    return parser


def main():
    args = _build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)

    task_epochs = _parse_csv_numbers(args.task_epochs, int)
    task_lrs = _parse_csv_numbers(args.task_lrs, float)
    task_slow_momentum = _parse_csv_numbers(args.task_slow_momentum, float)

    if not (len(task_epochs) == len(task_lrs) == len(task_slow_momentum) == args.num_tasks):
        raise ValueError("task-epochs, task-lrs, and task-slow-momentum must each have num_tasks values.")

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
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

    model_kwargs: Dict = {
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": args.base_channels,
        "memory_dim": args.memory_dim,
        "updater_hidden_dim": args.updater_hidden_dim,
    }
    model = NBMPolypNet(**model_kwargs).to(device)
    criterion = NBMPolypLoss(
        main_weight=1.0,
        aux_weight=0.4,
        edge_weight=0.2,
        consistency_weight=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=task_lrs[0], weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_dice = -1.0
    best_state = None
    history = []
    global_epoch = 0

    for task_idx, task_loader in enumerate(train_task_loaders, start=1):
        epochs_per_task = task_epochs[task_idx - 1]
        task_lr = task_lrs[task_idx - 1]
        task_momentum = task_slow_momentum[task_idx - 1]
        _set_optimizer_lr(optimizer, task_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs_per_task,
            eta_min=max(task_lr * 0.1, 1e-6),
        )

        print(f"\n========== START TASK {task_idx} ==========\n")
        print(f"[Task {task_idx}] init_lr = {_get_optimizer_lr(optimizer):.8f}")
        print(
            f"[Task {task_idx}] slow_memory_norms(before) = "
            f"s2={model.get_slow_memory('s2').norm().item():.6f}, "
            f"s3={model.get_slow_memory('s3').norm().item():.6f}, "
            f"s4={model.get_slow_memory('s4').norm().item():.6f}\n"
        )

        epoch_memory_summaries = []
        for local_epoch in range(1, epochs_per_task + 1):
            global_epoch += 1
            train_metrics, memory_summary = train_one_task_nbm(
                model=model,
                loader=task_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task_id=task_idx,
                epoch=local_epoch,
                scaler=scaler,
                use_amp=True,
                grad_clip=args.grad_clip,
                memory_after_weight=args.memory_after_weight,
                memory_stability_weight=args.memory_stability_weight,
                improve_weight=args.improve_weight,
                improve_margin=args.improve_margin,
                before_weight=args.before_weight,
                print_freq=10,
                initial_memory=None,
            )
            epoch_memory_summaries.append(memory_summary)

            val_metrics = validate_nbm(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                use_amp=True,
                print_freq=20,
            )
            scheduler.step()
            current_lr = _get_optimizer_lr(optimizer)

            history.append(
                {
                    "task_id": task_idx,
                    "local_epoch": local_epoch,
                    "global_epoch": global_epoch,
                    "lr": current_lr,
                    "slow_memory_norm_s2": float(model.get_slow_memory("s2").norm().item()),
                    "slow_memory_norm_s3": float(model.get_slow_memory("s3").norm().item()),
                    "slow_memory_norm_s4": float(model.get_slow_memory("s4").norm().item()),
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
                        "train_config": vars(args),
                    },
                    os.path.join(args.save_root, "best_model.pth"),
                )

        task_summary = {
            stage: torch.stack([summary[stage] for summary in epoch_memory_summaries], dim=0).mean(dim=0).to(device)
            for stage in epoch_memory_summaries[0].keys()
        }
        print(
            f"[Task {task_idx}] task_summary_norms = "
            f"s2={task_summary['s2'].norm().item():.6f}, "
            f"s3={task_summary['s3'].norm().item():.6f}, "
            f"s4={task_summary['s4'].norm().item():.6f}"
        )
        model.update_slow_memory(task_summary, momentum=task_momentum, max_norm=1.0)
        print(
            f"[Task {task_idx}] slow_memory_norms(after) = "
            f"s2={model.get_slow_memory('s2').norm().item():.6f}, "
            f"s3={model.get_slow_memory('s3').norm().item():.6f}, "
            f"s4={model.get_slow_memory('s4').norm().item():.6f}\n"
        )

        with open(os.path.join(args.save_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best val dice: {best_val_dice:.4f}")
    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(best_state)
    test_metrics = test_nbm(
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
