from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler

from data import build_dataloaders
from engine.train_nbm_v3 import train_epoch_warmup_v3, train_one_task_nbm_v3
from engine.eval_nbm_v3 import validate_nbm_v3, test_nbm_v3
from loss.nbm_losses_v2 import NBMPolypLossV2
from model.nbm_polyp_model_v3 import NBMPolypNetV3


def _parse_csv_numbers(value: str, cast=int) -> List:
    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def _set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _get_optimizer_lr(optimizer) -> float:
    return optimizer.param_groups[0]['lr']


def _dump_json(path: str, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def _build_parser():
    parser = argparse.ArgumentParser(description='Train NBM-PolypNet-v3 (safe residual memory) on Kvasir-style data.')
    parser.add_argument('--file-path', default='datasets/Kvasir')
    parser.add_argument('--save-root', default='outputs/kvasir_nbm_polypnet_v3')
    parser.add_argument('--image-size', type=int, nargs=2, default=(352, 352), metavar=('H', 'W'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-tasks', type=int, default=4)
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-augmentation', action='store_true')

    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--memory-dim', type=int, default=64)
    parser.add_argument('--num-prototypes', type=int, default=4)
    parser.add_argument('--updater-hidden-dim', type=int, default=128)

    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--warmup-lr', type=float, default=1e-4)
    parser.add_argument('--task-epochs', type=str, default='3,4,6,8')
    parser.add_argument('--task-lrs', type=str, default='8e-5,6e-5,5e-5,4e-5')
    parser.add_argument('--task-slow-momentum', type=str, default='0.00,0.00,0.015,0.02')
    parser.add_argument('--joint-finetune-epochs', type=int, default=8)
    parser.add_argument('--joint-finetune-lr', type=float, default=1.5e-5)

    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--memory-after-weight', type=float, default=0.8)
    parser.add_argument('--memory-stability-weight', type=float, default=5e-5)
    parser.add_argument('--improve-weight', type=float, default=0.1)
    parser.add_argument('--improve-margin', type=float, default=0.002)
    parser.add_argument('--before-weight', type=float, default=0.8)
    parser.add_argument('--distill-weight', type=float, default=0.15)

    parser.add_argument('--memory-start-task', type=int, default=3)
    parser.add_argument('--memory-blend', type=float, default=0.15)
    parser.add_argument('--joint-memory-blend', type=float, default=0.10)
    parser.add_argument('--keep-batch-memory', action='store_true')
    parser.add_argument('--skip-memory-if-hurts', action='store_true', default=True)
    parser.add_argument('--skip-margin', type=float, default=0.003)
    parser.add_argument('--slow-memory-max-norm', type=float, default=0.60)

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--threshold-sweep', type=str, default='0.35,0.40,0.45,0.50,0.55,0.60,0.65')
    parser.add_argument('--freeze-eval-memory', action='store_true')
    parser.add_argument('--eval-use-memory', action='store_true', default=True)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = tuple(args.image_size)

    task_epochs = _parse_csv_numbers(args.task_epochs, int)
    task_lrs = _parse_csv_numbers(args.task_lrs, float)
    task_slow_momentum = _parse_csv_numbers(args.task_slow_momentum, float)
    threshold_sweep = _parse_csv_numbers(args.threshold_sweep, float)
    if len(task_epochs) != args.num_tasks or len(task_lrs) != args.num_tasks or len(task_slow_momentum) != args.num_tasks:
        raise ValueError('task-epochs, task-lrs, and task-slow-momentum must each have num_tasks values.')

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
    _dump_json(os.path.join(args.save_root, 'meta_info.json'), meta_info)

    model_kwargs: Dict = {
        'in_channels': 3,
        'out_channels': 1,
        'base_channels': args.base_channels,
        'memory_dim': args.memory_dim,
        'num_prototypes': args.num_prototypes,
        'updater_hidden_dim': args.updater_hidden_dim,
    }
    model = NBMPolypNetV3(**model_kwargs).to(device)
    criterion = NBMPolypLossV2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.warmup_lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_dice = -1.0
    best_threshold = args.threshold
    best_state = None
    history = []
    global_epoch = 0

    if args.warmup_epochs > 0:
        print("\n========== START WARMUP ==========")
        warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warmup_epochs, eta_min=max(args.warmup_lr * 0.1, 1e-6))
        for epoch in range(1, args.warmup_epochs + 1):
            global_epoch += 1
            train_metrics = train_epoch_warmup_v3(model, train_loader_full, optimizer, criterion, device, epoch, scaler=scaler, use_amp=True, grad_clip=args.grad_clip, threshold=args.threshold, print_freq=10)
            val_metrics, sweep = validate_nbm_v3(model, val_loader, criterion, device, epoch=global_epoch, use_amp=True, threshold=args.threshold, threshold_sweep=threshold_sweep, freeze_eval_memory=True, use_memory=False, print_freq=20)
            warmup_scheduler.step()
            current_lr = _get_optimizer_lr(optimizer)
            chosen_val = sweep['dice'] if sweep is not None else val_metrics['dice']
            chosen_threshold = sweep['best_threshold'] if sweep is not None else args.threshold
            history.append({'stage': 'warmup', 'epoch': epoch, 'global_epoch': global_epoch, 'lr': current_lr, 'train': train_metrics, 'val': val_metrics, 'val_threshold_sweep': sweep})
            print(f"\n[Warmup | Epoch {epoch}] lr={current_lr:.8f} | train_dice={train_metrics['dice']:.4f} | val_dice={chosen_val:.4f} @ thr={chosen_threshold:.2f}\n")
            if chosen_val > best_val_dice:
                best_val_dice = chosen_val; best_threshold = chosen_threshold; best_state = copy.deepcopy(model.state_dict())
                torch.save({'state_dict': best_state, 'model_kwargs': model_kwargs, 'best_val_dice': best_val_dice, 'best_threshold': best_threshold, 'train_config': vars(args)}, os.path.join(args.save_root, 'best_model.pth'))
            _dump_json(os.path.join(args.save_root, 'history.json'), history)

    optimizer = torch.optim.AdamW(model.parameters(), lr=task_lrs[0], weight_decay=args.weight_decay)
    for task_idx, task_loader in enumerate(train_task_loaders, start=1):
        epochs_per_task = task_epochs[task_idx - 1]
        task_lr = task_lrs[task_idx - 1]
        task_momentum = task_slow_momentum[task_idx - 1]
        _set_optimizer_lr(optimizer, task_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_task, eta_min=max(task_lr * 0.2, 1e-6))
        use_memory_this_task = task_idx >= args.memory_start_task

        print(f"\n========== START TASK {task_idx} ==========")
        print(f"[Task {task_idx}] use_memory={use_memory_this_task} | init_lr={_get_optimizer_lr(optimizer):.8f}")
        print(f"[Task {task_idx}] slow_memory_norms(before) = s2={model.get_slow_memory('s2').norm().item():.6f}, s3={model.get_slow_memory('s3').norm().item():.6f}, s4={model.get_slow_memory('s4').norm().item():.6f}\n")

        epoch_memory_summaries = []
        for local_epoch in range(1, epochs_per_task + 1):
            global_epoch += 1
            train_metrics, memory_summary = train_one_task_nbm_v3(
                model=model, loader=task_loader, optimizer=optimizer, criterion=criterion, device=device, task_id=task_idx, epoch=local_epoch, scaler=scaler, use_amp=True, grad_clip=args.grad_clip, memory_after_weight=args.memory_after_weight, memory_stability_weight=args.memory_stability_weight, improve_weight=args.improve_weight, improve_margin=args.improve_margin, before_weight=args.before_weight, threshold=best_threshold, print_freq=10, use_memory=use_memory_this_task, memory_blend=args.memory_blend, keep_batch_memory=args.keep_batch_memory, skip_memory_if_hurts=args.skip_memory_if_hurts, skip_margin=args.skip_margin, distill_weight=args.distill_weight)
            epoch_memory_summaries.append(memory_summary)
            val_metrics, sweep = validate_nbm_v3(model, val_loader, criterion, device, epoch=global_epoch, use_amp=True, threshold=best_threshold, threshold_sweep=threshold_sweep, freeze_eval_memory=args.freeze_eval_memory, use_memory=(args.eval_use_memory and use_memory_this_task), memory_blend=args.memory_blend, print_freq=20)
            scheduler.step(); current_lr = _get_optimizer_lr(optimizer)
            chosen_val = sweep['dice'] if sweep is not None else val_metrics['dice']
            chosen_threshold = sweep['best_threshold'] if sweep is not None else best_threshold
            history.append({'stage': f'task_{task_idx}', 'task_id': task_idx, 'local_epoch': local_epoch, 'global_epoch': global_epoch, 'lr': current_lr, 'slow_memory_norm_s2': float(model.get_slow_memory('s2').norm().item()), 'slow_memory_norm_s3': float(model.get_slow_memory('s3').norm().item()), 'slow_memory_norm_s4': float(model.get_slow_memory('s4').norm().item()), 'use_memory': use_memory_this_task, 'train': train_metrics, 'val': val_metrics, 'val_threshold_sweep': sweep})
            print(f"\n[Task {task_idx} | Epoch {local_epoch}] lr={current_lr:.8f} | train_dice={train_metrics['dice']:.4f} | train_dice_gain={train_metrics['dice_gain']:.4f} | val_dice={chosen_val:.4f} @ thr={chosen_threshold:.2f}\n")
            if chosen_val > best_val_dice:
                best_val_dice = chosen_val; best_threshold = chosen_threshold; best_state = copy.deepcopy(model.state_dict())
                torch.save({'state_dict': best_state, 'model_kwargs': model_kwargs, 'best_val_dice': best_val_dice, 'best_threshold': best_threshold, 'train_config': vars(args)}, os.path.join(args.save_root, 'best_model.pth'))
            _dump_json(os.path.join(args.save_root, 'history.json'), history)

        if use_memory_this_task and task_momentum > 0.0:
            task_summary = {stage: torch.stack([summary[stage] for summary in epoch_memory_summaries], dim=0).mean(dim=0).to(device) for stage in epoch_memory_summaries[0].keys()}
            print(f"[Task {task_idx}] task_summary_norms = s2={task_summary['s2'].norm().item():.6f}, s3={task_summary['s3'].norm().item():.6f}, s4={task_summary['s4'].norm().item():.6f}")
            model.update_slow_memory(task_summary, momentum=task_momentum, max_norm=args.slow_memory_max_norm)
            print(f"[Task {task_idx}] slow_memory_norms(after) = s2={model.get_slow_memory('s2').norm().item():.6f}, s3={model.get_slow_memory('s3').norm().item():.6f}, s4={model.get_slow_memory('s4').norm().item():.6f}\n")

    if args.joint_finetune_epochs > 0:
        print("\n========== START JOINT FINETUNE ==========")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.joint_finetune_lr, weight_decay=args.weight_decay)
        joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.joint_finetune_epochs, eta_min=max(args.joint_finetune_lr * 0.2, 1e-6))
        for epoch in range(1, args.joint_finetune_epochs + 1):
            global_epoch += 1
            train_metrics, _ = train_one_task_nbm_v3(model, train_loader_full, optimizer, criterion, device, task_id=999, epoch=epoch, scaler=scaler, use_amp=True, grad_clip=args.grad_clip, memory_after_weight=args.memory_after_weight, memory_stability_weight=args.memory_stability_weight, improve_weight=args.improve_weight, improve_margin=args.improve_margin, before_weight=args.before_weight, threshold=best_threshold, print_freq=10, use_memory=args.eval_use_memory, memory_blend=args.joint_memory_blend, keep_batch_memory=args.keep_batch_memory, skip_memory_if_hurts=args.skip_memory_if_hurts, skip_margin=args.skip_margin, distill_weight=args.distill_weight)
            val_metrics, sweep = validate_nbm_v3(model, val_loader, criterion, device, epoch=global_epoch, use_amp=True, threshold=best_threshold, threshold_sweep=threshold_sweep, freeze_eval_memory=args.freeze_eval_memory, use_memory=args.eval_use_memory, memory_blend=args.joint_memory_blend, print_freq=20)
            joint_scheduler.step(); current_lr = _get_optimizer_lr(optimizer)
            chosen_val = sweep['dice'] if sweep is not None else val_metrics['dice']
            chosen_threshold = sweep['best_threshold'] if sweep is not None else best_threshold
            history.append({'stage': 'joint_finetune', 'epoch': epoch, 'global_epoch': global_epoch, 'lr': current_lr, 'train': train_metrics, 'val': val_metrics, 'val_threshold_sweep': sweep})
            print(f"\n[Joint Finetune | Epoch {epoch}] lr={current_lr:.8f} | train_dice={train_metrics['dice']:.4f} | val_dice={chosen_val:.4f} @ thr={chosen_threshold:.2f}\n")
            if chosen_val > best_val_dice:
                best_val_dice = chosen_val; best_threshold = chosen_threshold; best_state = copy.deepcopy(model.state_dict())
                torch.save({'state_dict': best_state, 'model_kwargs': model_kwargs, 'best_val_dice': best_val_dice, 'best_threshold': best_threshold, 'train_config': vars(args)}, os.path.join(args.save_root, 'best_model.pth'))
            _dump_json(os.path.join(args.save_root, 'history.json'), history)

    print(f'Best val dice: {best_val_dice:.4f} @ threshold {best_threshold:.2f}')
    if best_state is None:
        raise RuntimeError('No best checkpoint was saved.')

    model.load_state_dict(best_state)
    test_metrics = test_nbm_v3(model, test_loader, criterion, device, save_dir=os.path.join(args.save_root, 'predictions'), threshold=best_threshold, use_amp=True, freeze_eval_memory=args.freeze_eval_memory, use_memory=args.eval_use_memory, memory_blend=args.joint_memory_blend)
    _dump_json(os.path.join(args.save_root, 'test_metrics.json'), test_metrics)


if __name__ == '__main__':
    main()
