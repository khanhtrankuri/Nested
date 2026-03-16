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


def _set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _parse_schedule(raw_value: str, num_tasks: int, cast, name: str):
    values = [cast(item.strip()) for item in raw_value.split(",") if item.strip()]
    if len(values) == 1:
        values = values * num_tasks

    if len(values) != num_tasks:
        raise ValueError(
            f"{name} must contain either 1 value or {num_tasks} values. Got: {raw_value}"
        )

    return {task_idx: values[task_idx - 1] for task_idx in range(1, num_tasks + 1)}


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Train a sequential baseline without memory or replay."
    )
    parser.add_argument("--file-path", default="datasets/Kvasir")
    parser.add_argument(
        "--save-root",
        default="outputs/kvasir_baseline_sequential_aug448",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(448, 448), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-per-task", default="5,5,5,1")
    parser.add_argument("--task-lrs", default="1e-4,8e-5,5e-5,1e-5")
    parser.add_argument("--train-augmentation", dest="train_augmentation", action="store_true")
    parser.add_argument("--no-train-augmentation", dest="train_augmentation", action="store_false")
    parser.set_defaults(train_augmentation=True)
    return parser


def main():
    args = _build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)
    task_epochs_map = _parse_schedule(
        args.epochs_per_task,
        num_tasks=args.num_tasks,
        cast=int,
        name="epochs-per-task",
    )
    task_lr_map = _parse_schedule(
        args.task_lrs,
        num_tasks=args.num_tasks,
        cast=float,
        name="task-lrs",
    )

    train_task_loaders, _, val_loader, test_loader, meta_info = build_dataloaders(
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

    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_dice = -1.0
    best_state = None
    history = []
    global_epoch = 0

    print(
        f"Training sequential baseline with image_size={image_size}, "
        f"train_augmentation={args.train_augmentation}, "
        f"batch_size={args.batch_size}"
    )

    for task_idx, task_loader in enumerate(train_task_loaders, start=1):
        epochs_per_task = task_epochs_map[task_idx]
        task_lr = task_lr_map[task_idx]
        _set_optimizer_lr(optimizer, task_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs_per_task, 1),
            eta_min=max(task_lr * 0.1, 1e-6),
        )

        print(f"\n========== START TASK {task_idx} ==========\n")
        print(f"[Task {task_idx}] init_lr = {_get_optimizer_lr(optimizer):.8f}\n")

        for local_epoch in range(1, epochs_per_task + 1):
            global_epoch += 1

            train_metrics = train_one_epoch(
                model=model,
                loader=task_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                scaler=scaler,
                use_amp=True,
                grad_clip=1.0,
                print_freq=10,
            )

            val_metrics = validate(
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
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            print(
                f"\n[Task {task_idx} | Epoch {local_epoch}] "
                f"lr={current_lr:.8f} | "
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
