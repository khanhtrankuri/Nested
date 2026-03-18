import argparse
import copy
import json
import os
from typing import Dict, Tuple

import torch
from torch.amp import GradScaler

from data.load_data_clean import build_standalone_loader
from engine.train_eval_clean import ModelEMA, test_clean, train_one_epoch_clean
from loss.strong_baseline_loss import StrongBaselineLoss
from model.backbones.strong_baseline import StrongBaselinePolypModel
from train_baseline_clean import _split_param_groups


def build_parser():
    parser = argparse.ArgumentParser(description="Train the clean baseline on the full official train split")
    parser.add_argument("--file-path", default="datasets/Kvasir/train")
    parser.add_argument("--test-file-path", default="datasets/Kvasir/test")
    parser.add_argument("--save-root", default="outputs/kvasir_clean_baseline_full")
    parser.add_argument("--image-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", choices=["tiny_convnext", "convnext_tiny", "convnext_small"], default="convnext_tiny")
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--strict-pretrained", action="store_true")
    parser.add_argument("--pretrained-cache-dir", default="")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--enable-nested", action="store_true")
    parser.add_argument("--nested-start-epoch", type=int, default=8)
    parser.add_argument("--nested-dim", type=int, default=128)
    parser.add_argument("--nested-prototypes", type=int, default=8)
    parser.add_argument("--nested-residual-scale", type=float, default=0.05)
    parser.add_argument("--nested-max-norm", type=float, default=1.0)
    parser.add_argument("--nested-memory-mode", choices=["fast_slow", "slow_only"], default="fast_slow")
    parser.add_argument("--nested-memory-hidden", type=int, default=128)
    parser.add_argument("--nested-slow-momentum-scale", type=float, default=0.25)
    parser.add_argument("--nested-momentum", type=float, default=0.03)
    parser.add_argument("--skip-nested-if-hurts", dest="skip_nested_if_hurts", action="store_true")
    parser.add_argument("--no-skip-nested-if-hurts", dest="skip_nested_if_hurts", action="store_false")
    parser.set_defaults(skip_nested_if_hurts=True)
    parser.add_argument("--nested-skip-margin", type=float, default=0.002)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[1.0, 0.875, 1.125])
    parser.add_argument("--test-nested-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--small-polyp-sampling-power", type=float, default=0.35)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    return parser


def _resolve_nested_usage(mode: str, nested_active: bool) -> bool:
    if mode == "auto":
        return bool(nested_active)
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(f"Unsupported nested usage mode: {mode}")


def _default_model_kwargs(args) -> Dict[str, object]:
    return {
        "encoder_name": args.encoder_name,
        "use_pretrained": args.use_pretrained,
        "strict_pretrained": args.strict_pretrained,
        "pretrained_cache_dir": args.pretrained_cache_dir.strip() or "",
        "decoder_channels": args.decoder_channels,
        "dropout": args.dropout,
        "enable_nested": args.enable_nested,
        "nested_dim": args.nested_dim,
        "nested_prototypes": args.nested_prototypes,
        "nested_residual_scale": args.nested_residual_scale,
        "nested_max_norm": args.nested_max_norm,
        "nested_memory_mode": args.nested_memory_mode,
        "nested_memory_hidden": args.nested_memory_hidden,
        "nested_slow_momentum_scale": args.nested_slow_momentum_scale,
    }


def _infer_model_kwargs_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    inferred = {}
    lateral5_weight = state_dict.get("decoder.lateral5.weight")
    if isinstance(lateral5_weight, torch.Tensor) and lateral5_weight.ndim == 4:
        encoder_channels = int(lateral5_weight.shape[1])
        if encoder_channels == 512:
            inferred["encoder_name"] = "tiny_convnext"
        elif encoder_channels == 768:
            inferred["encoder_name"] = "convnext_tiny"
    seg_head_weight = state_dict.get("seg_head.1.weight")
    if isinstance(seg_head_weight, torch.Tensor) and seg_head_weight.ndim == 4:
        inferred["decoder_channels"] = int(seg_head_weight.shape[1])
    inferred["enable_nested"] = any(key.startswith("nested_refiner.") for key in state_dict.keys())
    return inferred


def _resolve_model_kwargs(args, checkpoint_payload: Dict[str, object]) -> Dict[str, object]:
    resolved = _default_model_kwargs(args)
    checkpoint_model_kwargs = checkpoint_payload.get("model_kwargs", {}) if isinstance(checkpoint_payload, dict) else {}
    for key, value in checkpoint_model_kwargs.items():
        if key in {"use_pretrained", "strict_pretrained", "pretrained_cache_dir", "pretrained_loaded"}:
            continue
        resolved[key] = value
    if not checkpoint_model_kwargs and isinstance(checkpoint_payload, dict):
        state_dict = checkpoint_payload.get("state_dict")
        if isinstance(state_dict, dict):
            resolved.update(_infer_model_kwargs_from_state_dict(state_dict))
    return resolved


def _resolve_threshold(args, checkpoint_payload: Dict[str, object]) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if isinstance(checkpoint_payload, dict):
        best_val = checkpoint_payload.get("best_val", {})
        if isinstance(best_val, dict) and best_val.get("threshold") is not None:
            return float(best_val["threshold"])
    return 0.5


def _resolve_image_size(args, checkpoint_payload: Dict[str, object]) -> Tuple[int, int]:
    if args.image_size is not None:
        return int(args.image_size[0]), int(args.image_size[1])
    if isinstance(checkpoint_payload, dict):
        train_config = checkpoint_payload.get("train_config", {})
        if isinstance(train_config, dict) and train_config.get("image_size") is not None:
            image_size = train_config["image_size"]
            return int(image_size[0]), int(image_size[1])
    return (384, 384)


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_payload = {}
    if args.init_checkpoint:
        checkpoint_payload = torch.load(args.init_checkpoint, map_location=device)

    image_size = _resolve_image_size(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    threshold = _resolve_threshold(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    model_kwargs = _resolve_model_kwargs(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})

    train_loader, train_meta = build_standalone_loader(
        file_path=args.file_path,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        small_polyp_sampling_power=args.small_polyp_sampling_power,
    )

    test_loader = None
    test_meta = None
    if not args.skip_final_test:
        test_loader, test_meta = build_standalone_loader(
            file_path=args.test_file_path,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
        )

    pretrained_cache_dir = model_kwargs.get("pretrained_cache_dir") or os.path.join(args.save_root, ".torch_cache")
    model_kwargs["pretrained_cache_dir"] = pretrained_cache_dir
    model = StrongBaselinePolypModel(
        encoder_name=model_kwargs["encoder_name"],
        use_pretrained=bool(model_kwargs.get("use_pretrained", False)),
        strict_pretrained=bool(model_kwargs.get("strict_pretrained", False)),
        pretrained_cache_dir=pretrained_cache_dir,
        decoder_channels=int(model_kwargs["decoder_channels"]),
        dropout=float(model_kwargs["dropout"]),
        enable_nested=bool(model_kwargs["enable_nested"]),
        nested_dim=int(model_kwargs["nested_dim"]),
        nested_prototypes=int(model_kwargs["nested_prototypes"]),
        nested_residual_scale=float(model_kwargs["nested_residual_scale"]),
        nested_max_norm=float(model_kwargs["nested_max_norm"]),
        nested_memory_mode=str(model_kwargs.get("nested_memory_mode", "fast_slow")),
        nested_memory_hidden=int(model_kwargs.get("nested_memory_hidden", 128)),
        nested_slow_momentum_scale=float(model_kwargs.get("nested_slow_momentum_scale", 0.25)),
    ).to(device)

    if args.init_checkpoint:
        state_dict = checkpoint_payload["state_dict"] if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload else checkpoint_payload
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(
            f"[StrongBaselineFull] Loaded init checkpoint: {args.init_checkpoint} | "
            f"missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)}"
        )

    meta_info = {
        "mode": "full_train",
        "train": train_meta,
        "test": test_meta,
        "model": {
            **model_kwargs,
            "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
        },
        "threshold": float(threshold),
    }
    with open(os.path.join(args.save_root, "meta_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(
        f"[StrongBaselineFull] encoder={model_kwargs['encoder_name']} | "
        f"use_pretrained={bool(model_kwargs.get('use_pretrained', False))} | "
        f"pretrained_loaded={getattr(model, 'pretrained_loaded', False)} | "
        f"threshold={threshold:.2f}"
    )

    criterion = StrongBaselineLoss()
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

    history = []
    final_state = None
    final_nested_active = False
    for epoch in range(1, args.epochs + 1):
        nested_active = bool(model_kwargs.get("enable_nested", False) and epoch >= args.nested_start_epoch)
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

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "nested_active": nested_active,
                "train": train_metrics,
            }
        )
        with open(os.path.join(args.save_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"\n[Epoch {epoch}] lr={current_lr:.7f} | "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"nested_active={nested_active}\n"
        )

        final_state = copy.deepcopy((ema.ema if ema is not None else model).state_dict())
        final_nested_active = nested_active

        torch.save(
            {
                "state_dict": final_state,
                "model_kwargs": model_kwargs,
                "train_config": vars(args),
                "final_threshold": float(threshold),
                "final_nested_active": bool(final_nested_active),
                "full_train": True,
            },
            os.path.join(args.save_root, "last_model.pth"),
        )

    if final_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(final_state)
    final_metrics = None
    if test_loader is not None:
        test_nested_active = _resolve_nested_usage(args.test_nested_mode, final_nested_active)
        final_metrics = test_clean(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            save_dir=os.path.join(args.save_root, "predictions") if args.save_predictions else None,
            threshold=threshold,
            use_amp=True,
            use_tta=args.use_tta,
            tta_scales=args.tta_scales,
            use_nested=test_nested_active,
        )
        with open(os.path.join(args.save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        print(f"Final threshold: {threshold:.2f}")
        print(f"Final nested active: {final_nested_active}")
        print(f"Final test nested active: {test_nested_active}")
        print(f"Official Test IoU: {final_metrics['iou']:.4f}")
        print(f"Official Test Dice: {final_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()


