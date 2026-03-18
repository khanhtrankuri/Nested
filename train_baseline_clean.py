import argparse
import copy
import json
import os

import torch
from torch.amp import GradScaler

from data.load_data_clean import build_clean_dataloaders
from engine.train_eval_clean import ModelEMA, test_clean, threshold_sweep_clean, train_one_epoch_clean
from loss.strong_baseline_loss import StrongBaselineLoss
from model.backbones.strong_baseline import StrongBaselinePolypModel


def build_parser():
    parser = argparse.ArgumentParser(description="Clean strong baseline for Kvasir polyp segmentation")
    parser.add_argument("--file-path", default="datasets/Kvasir/train")
    parser.add_argument("--save-root", default="outputs/kvasir_clean_baseline")
    parser.add_argument("--image-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", choices=["tiny_convnext", "convnext_tiny", "convnext_small"], default="convnext_tiny")
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--strict-pretrained", action="store_true")
    parser.add_argument("--pretrained-cache-dir", default="")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--enable-nested", action="store_true")
    parser.add_argument("--nested-start-epoch", type=int, default=20)
    parser.add_argument("--nested-dim", type=int, default=128)
    parser.add_argument("--nested-prototypes", type=int, default=8)
    parser.add_argument("--nested-residual-scale", type=float, default=0.05)
    parser.add_argument("--nested-max-norm", type=float, default=1.0)
    parser.add_argument("--nested-memory-hidden", type=int, default=128)
    parser.add_argument("--nested-slow-momentum-scale", type=float, default=0.25)
    parser.add_argument("--nested-momentum", type=float, default=0.03)
    parser.add_argument("--skip-nested-if-hurts", dest="skip_nested_if_hurts", action="store_true")
    parser.add_argument("--no-skip-nested-if-hurts", dest="skip_nested_if_hurts", action="store_false")
    parser.set_defaults(skip_nested_if_hurts=True)
    parser.add_argument("--nested-skip-margin", type=float, default=0.002)
    parser.add_argument("--protocol", choices=["strict", "kfold"], default="strict")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--small-polyp-sampling-power", type=float, default=0.35)
    parser.add_argument("--stratified-split", dest="stratified_split", action="store_true")
    parser.add_argument("--no-stratified-split", dest="stratified_split", action="store_false")
    parser.set_defaults(stratified_split=True)
    return parser


def _split_param_groups(model):
    if hasattr(model, "get_parameter_groups"):
        groups = model.get_parameter_groups()
        encoder_params = list(groups["encoder"])
        decoder_params = list(groups["decoder"])
    elif hasattr(model, "encoder"):
        encoder_params = list(model.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    elif all(hasattr(model, name) for name in ("stem", "layer1", "layer2", "layer3", "layer4")):
        encoder_params = (
            list(model.stem.parameters())
            + list(model.layer1.parameters())
            + list(model.layer2.parameters())
            + list(model.layer3.parameters())
            + list(model.layer4.parameters())
        )
        encoder_param_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    else:
        raise AttributeError(
            f"{model.__class__.__name__} does not expose encoder parameter groups. "
            "Expected `get_parameter_groups()`, `encoder`, or legacy `stem/layer1..4` attributes."
        )

    if not encoder_params:
        raise ValueError("Encoder parameter group is empty.")
    if not decoder_params:
        raise ValueError("Decoder parameter group is empty.")
    return encoder_params, decoder_params


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader, meta_info = build_clean_dataloaders(
        file_path=args.file_path,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        protocol=args.protocol,
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        train_augmentation=True,
        stratified_split=args.stratified_split,
        small_polyp_sampling_power=args.small_polyp_sampling_power,
    )

    pretrained_cache_dir = args.pretrained_cache_dir.strip() or os.path.join(args.save_root, ".torch_cache")
    model = StrongBaselinePolypModel(
        encoder_name=args.encoder_name,
        use_pretrained=args.use_pretrained,
        strict_pretrained=args.strict_pretrained,
        pretrained_cache_dir=pretrained_cache_dir,
        decoder_channels=args.decoder_channels,
        dropout=args.dropout,
        enable_nested=args.enable_nested,
        nested_dim=args.nested_dim,
        nested_prototypes=args.nested_prototypes,
        nested_residual_scale=args.nested_residual_scale,
        nested_max_norm=args.nested_max_norm,
        nested_memory_hidden=args.nested_memory_hidden,
        nested_slow_momentum_scale=args.nested_slow_momentum_scale,
    ).to(device)
    criterion = StrongBaselineLoss()
    model_kwargs = {
        "encoder_name": args.encoder_name,
        "use_pretrained": args.use_pretrained,
        "strict_pretrained": args.strict_pretrained,
        "pretrained_cache_dir": pretrained_cache_dir,
        "decoder_channels": args.decoder_channels,
        "dropout": args.dropout,
        "enable_nested": args.enable_nested,
        "nested_dim": args.nested_dim,
        "nested_prototypes": args.nested_prototypes,
        "nested_residual_scale": args.nested_residual_scale,
        "nested_max_norm": args.nested_max_norm,
        "nested_memory_hidden": args.nested_memory_hidden,
        "nested_slow_momentum_scale": args.nested_slow_momentum_scale,
    }
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(
            f"[StrongBaseline] Loaded init checkpoint: {args.init_checkpoint} | "
            f"missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)}"
        )
    meta_info["model"] = {
        **model_kwargs,
        "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
    }
    with open(os.path.join(args.save_root, "meta_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(
        f"[StrongBaseline] encoder={args.encoder_name} | "
        f"use_pretrained={args.use_pretrained} | "
        f"pretrained_loaded={getattr(model, 'pretrained_loaded', False)}"
    )

    encoder_params, decoder_params = _split_param_groups(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": decoder_params, "lr": args.decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.25,
        total_iters=max(args.warmup_epochs, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=0.999) if args.use_ema else None

    best_state = None
    best_val = {"iou": -1.0, "dice": -1.0, "threshold": 0.5}
    best_nested_active = False
    epochs_without_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        nested_active = bool(args.enable_nested and epoch >= args.nested_start_epoch)
        train_metrics = train_one_epoch_clean(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=True,
            grad_clip=1.0,
            ema=ema,
            print_freq=20,
            use_nested=nested_active,
            skip_nested_if_hurts=args.skip_nested_if_hurts,
            nested_skip_margin=args.nested_skip_margin,
            nested_momentum=args.nested_momentum,
            nested_max_norm=args.nested_max_norm,
        )
        scheduler.step()

        eval_model = ema.ema if ema is not None else model
        val_best = threshold_sweep_clean(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            thresholds=args.thresholds,
            use_amp=True,
            use_tta=args.use_tta,
            tta_scales=args.tta_scales,
            use_nested=nested_active,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "nested_active": nested_active,
            "train": train_metrics,
            "val": val_best,
        })

        print(
            f"\n[Epoch {epoch}] lr={current_lr:.7f} | "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"val_iou={val_best['iou']:.4f} | val_dice={val_best['dice']:.4f} | "
            f"best_thr={val_best['threshold']:.2f} | nested_active={nested_active}\n"
        )

        improved = False
        if val_best["iou"] > best_val["iou"]:
            improved = True
        elif val_best["iou"] == best_val["iou"] and val_best["dice"] > best_val["dice"]:
            improved = True

        if improved:
            best_val = val_best
            best_nested_active = nested_active
            best_state = copy.deepcopy((ema.ema if ema is not None else model).state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "best_val": best_val,
                    "best_nested_active": best_nested_active,
                    "model_kwargs": model_kwargs,
                    "train_config": vars(args),
                },
                os.path.join(args.save_root, "best_model.pth"),
            )
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        with open(os.path.join(args.save_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint saved.")

    model.load_state_dict(best_state)
    test_metrics = test_clean(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=os.path.join(args.save_root, "predictions"),
        threshold=best_val["threshold"],
        use_amp=True,
        use_tta=args.use_tta,
        tta_scales=args.tta_scales,
        use_nested=best_nested_active,
    )

    with open(os.path.join(args.save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Best validation IoU: {best_val['iou']:.4f}")
    print(f"Best validation Dice: {best_val['dice']:.4f}")
    print(f"Best threshold: {best_val['threshold']:.2f}")
    print(f"Best nested active: {best_nested_active}")
    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()
