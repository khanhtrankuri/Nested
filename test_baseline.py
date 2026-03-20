import argparse
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.amp import autocast

from data.load_data_clean import build_standalone_loader
from engine.train_eval_clean import AverageMeter, _forward_with_tta
from loss.strong_baseline_loss import StrongBaselineLoss
from metrics.segmentation_metrics import compute_segmentation_metrics
from model.backbones.strong_baseline import StrongBaselinePolypModel


def build_parser():
    parser = argparse.ArgumentParser(description="Test clean strong baseline on a standalone test split")
    parser.add_argument("--file-path", "--test-file-path", dest="file_path", default="datasets/Kvasir/test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-root", default="outputs/kvasir_clean_test")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--use-tta", dest="use_tta", action="store_true")
    parser.add_argument("--no-use-tta", dest="use_tta", action="store_false")
    parser.set_defaults(use_tta=None)
    parser.add_argument("--tta-scales", type=float, nargs="+", default=None)
    parser.add_argument("--use-nested", dest="use_nested", action="store_true")
    parser.add_argument("--no-use-nested", dest="use_nested", action="store_false")
    parser.set_defaults(use_nested=None)
    parser.add_argument("--save-predictions", dest="save_predictions", action="store_true")
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.set_defaults(save_predictions=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--strict-load", action="store_true")

    # Fallback model config if checkpoint does not carry model_kwargs.
    parser.add_argument("--encoder-name", choices=["tiny_convnext", "convnext_tiny", "convnext_small", "convnext_base", "pvtv2_b2"], default="convnext_tiny")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--enable-nested", action="store_true")
    parser.add_argument("--nested-dim", type=int, default=128)
    parser.add_argument("--nested-prototypes", type=int, default=8)
    parser.add_argument("--nested-residual-scale", type=float, default=0.05)
    parser.add_argument("--nested-max-norm", type=float, default=1.0)
    parser.add_argument("--nested-memory-mode", choices=["fast_slow", "slow_only"], default="fast_slow")
    parser.add_argument("--nested-memory-hidden", type=int, default=128)
    parser.add_argument("--nested-slow-momentum-scale", type=float, default=0.25)
    return parser


def _normalize_image_size(image_size: Optional[Sequence[int]]) -> Tuple[int, int]:
    if image_size is None:
        return (384, 384)
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must be None, int, or (H, W), got: {image_size}")


def build_test_loader(file_path: str, image_size=(384, 384), batch_size: int = 8, num_workers: int = 4):
    loader, meta_info = build_standalone_loader(
        file_path=file_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
    )
    meta_info["num_test"] = meta_info.pop("num_samples")
    return loader, meta_info


def _default_model_kwargs(args) -> Dict[str, object]:
    return {
        "encoder_name": args.encoder_name,
        "use_pretrained": False,
        "strict_pretrained": False,
        "pretrained_cache_dir": None,
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
    lateral2_weight = state_dict.get("decoder.lateral2.weight")
    lateral3_weight = state_dict.get("decoder.lateral3.weight")
    lateral4_weight = state_dict.get("decoder.lateral4.weight")
    lateral5_weight = state_dict.get("decoder.lateral5.weight")
    if all(isinstance(weight, torch.Tensor) and weight.ndim == 4 for weight in (lateral2_weight, lateral3_weight, lateral4_weight, lateral5_weight)):
        encoder_channels = (
            int(lateral2_weight.shape[1]),
            int(lateral3_weight.shape[1]),
            int(lateral4_weight.shape[1]),
            int(lateral5_weight.shape[1]),
        )
        if encoder_channels == (64, 128, 256, 512):
            inferred["encoder_name"] = "tiny_convnext"
        elif encoder_channels == (64, 128, 320, 512):
            inferred["encoder_name"] = "pvtv2_b2"
        elif encoder_channels == (128, 256, 512, 1024):
            inferred["encoder_name"] = "convnext_base"
        elif encoder_channels == (96, 192, 384, 768):
            stage3_block_indices = set()
            for key in state_dict.keys():
                for prefix in ("encoder.features.5.", "features.5."):
                    if not key.startswith(prefix):
                        continue
                    block_index = key[len(prefix):].split(".", 1)[0]
                    if block_index.isdigit():
                        stage3_block_indices.add(int(block_index))
                    break
            inferred["encoder_name"] = "convnext_small" if len(stage3_block_indices) >= 20 else "convnext_tiny"
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


def _resolve_image_size(args, checkpoint_payload: Dict[str, object]) -> Tuple[int, int]:
    if args.image_size is not None:
        return _normalize_image_size(args.image_size)
    train_config = checkpoint_payload.get("train_config", {}) if isinstance(checkpoint_payload, dict) else {}
    if isinstance(train_config, dict) and train_config.get("image_size") is not None:
        return _normalize_image_size(train_config["image_size"])
    return (384, 384)


def _resolve_threshold(args, checkpoint_payload: Dict[str, object]) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    best_val = checkpoint_payload.get("best_val", {}) if isinstance(checkpoint_payload, dict) else {}
    if isinstance(best_val, dict) and best_val.get("threshold") is not None:
        return float(best_val["threshold"])
    return 0.5


def _resolve_use_tta(args, checkpoint_payload: Dict[str, object]) -> bool:
    if args.use_tta is not None:
        return bool(args.use_tta)
    train_config = checkpoint_payload.get("train_config", {}) if isinstance(checkpoint_payload, dict) else {}
    if isinstance(train_config, dict):
        return bool(train_config.get("use_tta", False))
    return False


def _resolve_tta_scales(args, checkpoint_payload: Dict[str, object]) -> Tuple[float, ...]:
    if args.tta_scales:
        return tuple(float(scale) for scale in args.tta_scales)
    train_config = checkpoint_payload.get("train_config", {}) if isinstance(checkpoint_payload, dict) else {}
    if isinstance(train_config, dict) and train_config.get("tta_scales"):
        return tuple(float(scale) for scale in train_config["tta_scales"])
    return (1.0,)


def _resolve_use_nested(args, checkpoint_payload: Dict[str, object], model_kwargs: Dict[str, object]) -> bool:
    if not bool(model_kwargs.get("enable_nested", False)):
        return False
    if args.use_nested is not None:
        return bool(args.use_nested)
    if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("best_nested_active") is not None:
        return bool(checkpoint_payload["best_nested_active"])
    return False


def _save_prediction_masks(logits: torch.Tensor, file_names, save_dir: str, threshold: float):
    probs = torch.sigmoid(logits).detach().cpu()
    preds = (probs > threshold).to(torch.uint8).squeeze(1).numpy() * 255
    for pred_mask, file_name in zip(preds, file_names):
        save_path = os.path.join(save_dir, file_name)
        Image.fromarray(pred_mask, mode="L").save(save_path)


@torch.no_grad()
def run_test(
    model,
    loader,
    criterion,
    device: str,
    threshold: float,
    use_tta: bool,
    tta_scales: Sequence[float],
    use_nested: bool,
    save_dir: Optional[str] = None,
    max_batches: int = 0,
):
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    amp_enabled = torch.cuda.is_available() and str(device).startswith("cuda")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    total_steps = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    for step, batch in enumerate(loader, start=1):
        if max_batches > 0 and step > max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = (
                _forward_with_tta(model, images, use_nested=use_nested, tta_scales=tta_scales)
                if use_tta
                else model(images, use_nested=use_nested)["logits"]
            )
            loss, _ = criterion({"logits": logits}, masks, return_components=True)

        metric_dict = compute_segmentation_metrics(logits, masks, threshold=threshold)
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        dice_meter.update(metric_dict["dice"], bs)
        iou_meter.update(metric_dict["iou"], bs)
        precision_meter.update(metric_dict["precision"], bs)
        recall_meter.update(metric_dict["recall"], bs)

        if save_dir is not None:
            _save_prediction_masks(logits, batch["file_name"], save_dir, threshold=threshold)

        print(
            f"[Test] Step {step}/{total_steps} | "
            f"loss={loss_meter.avg:.4f} | dice={dice_meter.avg:.4f} | iou={iou_meter.avg:.4f}"
        )

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "precision": precision_meter.avg,
        "recall": recall_meter.avg,
        "threshold": float(threshold),
        "use_tta": bool(use_tta),
        "tta_scales": [float(scale) for scale in tta_scales],
        "use_nested": bool(use_nested),
        "num_batches": int(total_steps),
    }


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint_payload["state_dict"] if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload else checkpoint_payload

    model_kwargs = _resolve_model_kwargs(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    image_size = _resolve_image_size(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    threshold = _resolve_threshold(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    use_tta = _resolve_use_tta(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    tta_scales = _resolve_tta_scales(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {})
    use_nested = _resolve_use_nested(args, checkpoint_payload if isinstance(checkpoint_payload, dict) else {}, model_kwargs)

    test_loader, test_meta = build_test_loader(
        file_path=args.file_path,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = StrongBaselinePolypModel(**model_kwargs).to(device)
    incompatible = model.load_state_dict(state_dict, strict=args.strict_load)
    if not args.strict_load:
        print(
            f"[StrongBaseline] Loaded checkpoint: {args.checkpoint} | "
            f"missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)}"
        )

    criterion = StrongBaselineLoss()
    prediction_dir = os.path.join(args.save_root, "predictions") if args.save_predictions else None
    test_metrics = run_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
        use_tta=use_tta,
        tta_scales=tta_scales,
        use_nested=use_nested,
        save_dir=prediction_dir,
        max_batches=args.max_batches,
    )

    test_meta.update(
        {
            "checkpoint": args.checkpoint,
            "device": device,
            "resolved_threshold": threshold,
            "resolved_use_tta": use_tta,
            "resolved_tta_scales": list(tta_scales),
            "resolved_use_nested": use_nested,
            "model_kwargs": model_kwargs,
            "save_predictions": bool(args.save_predictions),
            "max_batches": int(args.max_batches),
        }
    )
    if isinstance(checkpoint_payload, dict):
        if "best_val" in checkpoint_payload:
            test_meta["checkpoint_best_val"] = checkpoint_payload["best_val"]
        if "best_nested_active" in checkpoint_payload:
            test_meta["checkpoint_best_nested_active"] = checkpoint_payload["best_nested_active"]
        if "train_config" in checkpoint_payload:
            test_meta["train_config"] = checkpoint_payload["train_config"]

    with open(os.path.join(args.save_root, "test_meta.json"), "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)
    with open(os.path.join(args.save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Threshold: {test_metrics['threshold']:.2f}")
    print(f"Use TTA: {test_metrics['use_tta']}")
    print(f"TTA scales: {test_metrics['tta_scales']}")
    print(f"Use nested: {test_metrics['use_nested']}")


if __name__ == "__main__":
    main()
