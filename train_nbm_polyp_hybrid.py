from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List, Optional

import torch
from torch.cuda.amp import GradScaler

from data import build_dataloaders
from engine.test_hybrid import test_hybrid
from engine.train_one_epoch_hybrid import train_one_epoch_hybrid, train_one_task_hybrid
from engine.validate_hybrid import validate_hybrid
from loss.hybrid_losses import HybridPolypLoss
from model.nbm_polyp_hybrid import NBMPolypNetHybrid


def _parse_csv_numbers(value: str, cast=float) -> List:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _expand_schedule(values: List, num_tasks: int, name: str) -> List:
    if len(values) == 1:
        return values * num_tasks
    if len(values) != num_tasks:
        raise ValueError(f"{name} must have either 1 value or num_tasks={num_tasks} values.")
    return values


def _set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def _get_optimizer_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _dump_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def _build_parser():
    parser = argparse.ArgumentParser(description="Train NBM-PolypNet-Hybrid on Kvasir-style data.")
    parser.add_argument("--file-path", default="datasets/Kvasir")
    parser.add_argument("--save-root", default="outputs/kvasir_nbm_polyp_hybrid")
    parser.add_argument("--image-size", type=int, nargs=2, default=(352, 352), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--warmup-lr", type=float, default=1e-4)
    parser.add_argument("--task-epochs", type=str, default="3,4,5,6")
    parser.add_argument("--task-lrs", type=str, default="8e-5,6e-5,5e-5,4e-5")
    parser.add_argument("--task-slow-momentum", type=str, default="0.03,0.04,0.05,0.06")
    parser.add_argument("--joint-finetune-epochs", type=int, default=6)
    parser.add_argument("--joint-finetune-lr", type=float, default=2e-5)

    parser.add_argument("--train-augmentation", dest="train_augmentation", action="store_true")
    parser.add_argument("--no-train-augmentation", dest="train_augmentation", action="store_false")
    parser.set_defaults(train_augmentation=True)

    parser.add_argument("--threshold-sweep", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--freeze-eval-memory", dest="freeze_eval_memory", action="store_true")
    parser.add_argument("--no-freeze-eval-memory", dest="freeze_eval_memory", action="store_false")
    parser.set_defaults(freeze_eval_memory=True)
    parser.add_argument("--memory-start-task", type=int, default=2)
    parser.add_argument("--memory-blend", type=float, default=0.10)
    parser.add_argument("--slow-memory-max-norm", type=float, default=1.0)
    parser.add_argument("--num-prototypes", type=int, default=6)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--use-tta", dest="use_tta", action="store_true")
    parser.add_argument("--no-use-tta", dest="use_tta", action="store_false")
    parser.set_defaults(use_tta=True)
    parser.add_argument("--skip-memory-if-hurts", dest="skip_memory_if_hurts", action="store_true")
    parser.add_argument("--no-skip-memory-if-hurts", dest="skip_memory_if_hurts", action="store_false")
    parser.set_defaults(skip_memory_if_hurts=True)

    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--before-weight", type=float, default=0.7)
    parser.add_argument("--memory-after-weight", type=float, default=1.0)
    parser.add_argument("--memory-stability-weight", type=float, default=1e-4)
    parser.add_argument("--improve-weight", type=float, default=0.1)
    parser.add_argument("--improve-margin", type=float, default=0.001)
    parser.add_argument("--skip-margin", type=float, default=0.002)
    parser.add_argument("--fast-memory-mode", choices=("batch", "epoch", "task"), default="batch")
    return parser


def _build_model_kwargs(args) -> Dict:
    return {
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": args.base_channels,
        "memory_dim": args.memory_dim,
        "num_prototypes": args.num_prototypes,
        "memory_blend": args.memory_blend,
        "slow_memory_max_norm": args.slow_memory_max_norm,
    }


def _save_checkpoint(path: str, model, model_kwargs: Dict, args, best_val_dice: float, best_threshold: float):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "best_val_dice": best_val_dice,
            "best_threshold": best_threshold,
            "train_config": vars(args),
        },
        path,
    )


def _record_history(history: List[Dict], save_root: str):
    _dump_json(os.path.join(save_root, "history.json"), history)


def main():
    args = _build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)

    task_epochs = _expand_schedule(_parse_csv_numbers(args.task_epochs, int), args.num_tasks, "task-epochs")
    task_lrs = _expand_schedule(_parse_csv_numbers(args.task_lrs, float), args.num_tasks, "task-lrs")
    task_slow_momentum = _expand_schedule(
        _parse_csv_numbers(args.task_slow_momentum, float), args.num_tasks, "task-slow-momentum"
    )
    threshold_sweep = _parse_csv_numbers(args.threshold_sweep, float)

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
    _dump_json(os.path.join(args.save_root, "meta_info.json"), meta_info)

    model_kwargs = _build_model_kwargs(args)
    model = NBMPolypNetHybrid(**model_kwargs).to(device)
    criterion = HybridPolypLoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_dice = -1.0
    best_threshold = 0.5
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict] = []
    global_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.warmup_lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.warmup_epochs,
            eta_min=max(args.warmup_lr * 0.1, 1e-6),
        )
        for epoch in range(1, args.warmup_epochs + 1):
            global_epoch += 1
            train_metrics, _, _ = train_one_epoch_hybrid(
                model=model,
                loader=train_loader_full,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                scaler=scaler,
                use_amp=True,
                grad_clip=args.grad_clip,
                threshold=best_threshold,
                print_freq=10,
                use_memory=False,
                disable_memory=True,
            )
            val_metrics, sweep = validate_hybrid(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                use_amp=True,
                threshold=best_threshold,
                threshold_sweep=threshold_sweep,
                freeze_eval_memory=True,
                use_memory=False,
                disable_memory=True,
                memory_blend=args.memory_blend,
                print_freq=20,
            )
            scheduler.step()

            chosen_dice = sweep["dice"] if sweep is not None else val_metrics["dice"]
            chosen_threshold = sweep["best_threshold"] if sweep is not None else best_threshold
            history.append(
                {
                    "stage": "warmup",
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "lr": _get_optimizer_lr(optimizer),
                    "train": train_metrics,
                    "val": val_metrics,
                    "val_threshold_sweep": sweep,
                }
            )
            _record_history(history, args.save_root)
            print(
                f"[Warmup {epoch}] lr={_get_optimizer_lr(optimizer):.8f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"best_val_dice={chosen_dice:.4f} @thr={chosen_threshold:.2f}"
            )

            if chosen_dice > best_val_dice:
                best_val_dice = chosen_dice
                best_threshold = chosen_threshold
                best_state = copy.deepcopy(model.state_dict())
                _save_checkpoint(
                    os.path.join(args.save_root, "best_model.pth"),
                    model,
                    model_kwargs,
                    args,
                    best_val_dice,
                    best_threshold,
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=task_lrs[0], weight_decay=args.weight_decay)
    for task_idx, task_loader in enumerate(train_task_loaders, start=1):
        task_lr = task_lrs[task_idx - 1]
        task_epochs_current = task_epochs[task_idx - 1]
        slow_momentum = task_slow_momentum[task_idx - 1]
        _set_optimizer_lr(optimizer, task_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(task_epochs_current, 1),
            eta_min=max(task_lr * 0.2, 1e-6),
        )

        memory_active = (not args.disable_memory) and (task_idx >= args.memory_start_task)
        print(
            f"\n========== START TASK {task_idx} ==========\n"
            f"memory_active={memory_active} | lr={task_lr:.8f} | "
            f"slow_stage3={model.get_slow_memory('stage3').norm().item():.4f} | "
            f"slow_stage2={model.get_slow_memory('stage2').norm().item():.4f}\n"
        )

        epoch_summaries = []
        next_memory_seed = None
        for local_epoch in range(1, task_epochs_current + 1):
            global_epoch += 1
            train_metrics, epoch_summary, next_memory_seed = train_one_task_hybrid(
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
                threshold=best_threshold,
                print_freq=10,
                use_memory=memory_active,
                disable_memory=args.disable_memory,
                memory_blend=args.memory_blend,
                before_weight=args.before_weight,
                memory_after_weight=args.memory_after_weight,
                memory_stability_weight=args.memory_stability_weight,
                improve_weight=args.improve_weight,
                improve_margin=args.improve_margin,
                skip_memory_if_hurts=args.skip_memory_if_hurts,
                skip_margin=args.skip_margin,
                fast_memory_mode=args.fast_memory_mode,
                initial_memory=next_memory_seed,
            )
            if epoch_summary is not None:
                epoch_summaries.append(epoch_summary)

            val_metrics, sweep = validate_hybrid(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                use_amp=True,
                threshold=best_threshold,
                threshold_sweep=threshold_sweep,
                freeze_eval_memory=args.freeze_eval_memory,
                use_memory=memory_active,
                disable_memory=args.disable_memory,
                memory_blend=args.memory_blend,
                print_freq=20,
            )
            scheduler.step()

            chosen_dice = sweep["dice"] if sweep is not None else val_metrics["dice"]
            chosen_threshold = sweep["best_threshold"] if sweep is not None else best_threshold
            history.append(
                {
                    "stage": f"task_{task_idx}",
                    "task_id": task_idx,
                    "local_epoch": local_epoch,
                    "global_epoch": global_epoch,
                    "lr": _get_optimizer_lr(optimizer),
                    "memory_active": bool(memory_active),
                    "train": train_metrics,
                    "val": val_metrics,
                    "val_threshold_sweep": sweep,
                }
            )
            _record_history(history, args.save_root)
            print(
                f"[Task {task_idx} | Epoch {local_epoch}] lr={_get_optimizer_lr(optimizer):.8f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"val_dice={chosen_dice:.4f} @thr={chosen_threshold:.2f}"
            )

            if chosen_dice > best_val_dice:
                best_val_dice = chosen_dice
                best_threshold = chosen_threshold
                best_state = copy.deepcopy(model.state_dict())
                _save_checkpoint(
                    os.path.join(args.save_root, "best_model.pth"),
                    model,
                    model_kwargs,
                    args,
                    best_val_dice,
                    best_threshold,
                )

        if memory_active and epoch_summaries:
            task_summary = {}
            for stage in epoch_summaries[0]:
                task_summary[stage] = torch.stack([summary[stage] for summary in epoch_summaries], dim=0).mean(dim=0)
            model.update_slow_memory(task_summary, momentum=slow_momentum, max_norm=args.slow_memory_max_norm)
            print(
                f"[Task {task_idx}] updated slow memory | "
                f"stage3={model.get_slow_memory('stage3').norm().item():.4f} | "
                f"stage2={model.get_slow_memory('stage2').norm().item():.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.joint_finetune_lr, weight_decay=args.weight_decay)
    joint_memory_active = (not args.disable_memory) and (args.num_tasks >= args.memory_start_task)
    if args.joint_finetune_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.joint_finetune_epochs,
            eta_min=max(args.joint_finetune_lr * 0.2, 1e-6),
        )
        next_memory_seed = None
        for epoch in range(1, args.joint_finetune_epochs + 1):
            global_epoch += 1
            train_metrics, _, next_memory_seed = train_one_epoch_hybrid(
                model=model,
                loader=train_loader_full,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                scaler=scaler,
                use_amp=True,
                grad_clip=args.grad_clip,
                threshold=best_threshold,
                print_freq=10,
                use_memory=joint_memory_active,
                disable_memory=args.disable_memory,
                memory_blend=min(args.memory_blend, 0.10),
                before_weight=args.before_weight,
                memory_after_weight=args.memory_after_weight,
                memory_stability_weight=args.memory_stability_weight,
                improve_weight=args.improve_weight,
                improve_margin=args.improve_margin,
                skip_memory_if_hurts=args.skip_memory_if_hurts,
                skip_margin=args.skip_margin,
                fast_memory_mode=args.fast_memory_mode,
                initial_memory=next_memory_seed,
            )
            val_metrics, sweep = validate_hybrid(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=global_epoch,
                use_amp=True,
                threshold=best_threshold,
                threshold_sweep=threshold_sweep,
                freeze_eval_memory=args.freeze_eval_memory,
                use_memory=joint_memory_active,
                disable_memory=args.disable_memory,
                memory_blend=min(args.memory_blend, 0.10),
                print_freq=20,
            )
            scheduler.step()

            chosen_dice = sweep["dice"] if sweep is not None else val_metrics["dice"]
            chosen_threshold = sweep["best_threshold"] if sweep is not None else best_threshold
            history.append(
                {
                    "stage": "joint_finetune",
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "lr": _get_optimizer_lr(optimizer),
                    "memory_active": bool(joint_memory_active),
                    "train": train_metrics,
                    "val": val_metrics,
                    "val_threshold_sweep": sweep,
                }
            )
            _record_history(history, args.save_root)
            print(
                f"[Joint {epoch}] lr={_get_optimizer_lr(optimizer):.8f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"val_dice={chosen_dice:.4f} @thr={chosen_threshold:.2f}"
            )

            if chosen_dice > best_val_dice:
                best_val_dice = chosen_dice
                best_threshold = chosen_threshold
                best_state = copy.deepcopy(model.state_dict())
                _save_checkpoint(
                    os.path.join(args.save_root, "best_model.pth"),
                    model,
                    model_kwargs,
                    args,
                    best_val_dice,
                    best_threshold,
                )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        _save_checkpoint(
            os.path.join(args.save_root, "best_model.pth"),
            model,
            model_kwargs,
            args,
            best_val_dice=max(best_val_dice, 0.0),
            best_threshold=best_threshold,
        )

    model.load_state_dict(best_state)
    test_metrics = test_hybrid(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=os.path.join(args.save_root, "predictions"),
        threshold=best_threshold,
        use_amp=True,
        freeze_eval_memory=args.freeze_eval_memory,
        use_memory=joint_memory_active,
        disable_memory=args.disable_memory,
        memory_blend=min(args.memory_blend, 0.10),
        use_tta=args.use_tta,
    )
    _dump_json(
        os.path.join(args.save_root, "test_metrics.json"),
        {
            "best_val_dice": best_val_dice,
            "best_threshold": best_threshold,
            "test": test_metrics,
        },
    )

    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
