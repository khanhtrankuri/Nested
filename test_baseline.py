import argparse
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader

from data.load_data_clean import CleanPolypDataset
from engine.train_eval_clean import AverageMeter
from loss.strong_baseline_loss import StrongBaselineLoss
from metrics.segmentation_metrics import compute_segmentation_metrics
from model.backbones.strong_baseline import StrongBaselinePolypModel


def build_parser():
    parser = argparse.ArgumentParser(description="Test clean strong baseline on a standalone test split")
    parser.add_argument("--file-path", default="datasets/Kvasir/test")
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
    parser.add_argument("--use-nested", dest="use_nested", action="store_true")
    parser.add_argument("--no-use-nested", dest="use_nested", action="store_false")
    parser.set_defaults(use_nested=None)
    parser.add_argument("--save-predictions", dest="save_predictions", action="store_true")
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.set_defaults(save_predictions=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--strict-load", action="store_true")

    # Fallback model config if checkpoint does not carry model_kwargs.
    parser.add_argument("--encoder-name", choices=["tiny_convnext", "convnext_tiny", "convnext_small"], default="convnext_tiny")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--enable-nested", action="store_true")
    parser.add_argument("--nested-dim", type=int, default=128)
    parser.add_argument("--nested-prototypes", type=int, default=8)
    parser.add_argument("--nested-residual-scale", type=float, default=0.05)
    parser.add_argument("--nested-max-norm", type=float, default=1.0)
    return parser


def _normalize_image_size(image_size: Optional[Sequence[int]]) -> Tuple[int, int]:
    if image_size is None:
        return (384, 384)
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must be None, int, or (H, W), got: {image_size}")


def _validate_pairs(image_dir: str, mask_dir: str):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

    image_names = sorted(
        file_name
        for file_name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, file_name))
    )
    mask_names = sorted(
        file_name
        for file_name in os.listdir(mask_dir)
        if os.path.isfile(os.path.join(mask_dir, file_name))
    )
    if not image_names:
        raise ValueError(f"No test images found in: {image_dir}")
    if image_names != mask_names:
        missing_masks = sorted(set(image_names) - set(mask_names))
        missing_images = sorted(set(mask_names) - set(image_names))
        raise ValueError(
            "Image/mask files do not match. "
            f"Missing masks: {missing_masks[:5]}. "
            f"Missing images: {missing_images[:5]}."
        )
    return image_names


def build_test_loader(file_path: str, image_size=(384, 384), batch_size: int = 8, num_workers: int = 4):
    image_dir = os.path.join(file_path, "images")
    mask_dir = os.path.join(file_path, "masks")
    file_names = _validate_pairs(image_dir, mask_dir)
    dataset = CleanPolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_names=file_names,
        image_size=image_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    meta_info = {
        "file_path": file_path,
        "num_test": len(dataset),
        "image_size": list(_normalize_image_size(image_size)),
    }
    return loader, meta_info


def _forward_with_tta(model, images: torch.Tensor, use_nested: bool = False):
    outputs = []
    logits = model(images, use_nested=use_nested)["logits"]
    outputs.append(logits)

    h = torch.flip(images, dims=[3])
    outputs.append(torch.flip(model(h, use_nested=use_nested)["logits"], dims=[3]))

    v = torch.flip(images, dims=[2])
    outputs.append(torch.flip(model(v, use_nested=use_nested)["logits"], dims=[2]))

    hv = torch.flip(images, dims=[2, 3])
    outputs.append(torch.flip(model(hv, use_nested=use_nested)["logits"], dims=[2, 3]))
    return torch.stack(outputs, dim=0).mean(dim=0)


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
    }


def _resolve_model_kwargs(args, checkpoint_payload: Dict[str, object]) -> Dict[str, object]:
    resolved = _default_model_kwargs(args)
    checkpoint_model_kwargs = checkpoint_payload.get("model_kwargs", {}) if isinstance(checkpoint_payload, dict) else {}
    for key, value in checkpoint_model_kwargs.items():
        if key in {"use_pretrained", "strict_pretrained", "pretrained_cache_dir", "pretrained_loaded"}:
            continue
        resolved[key] = value
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
            logits = _forward_with_tta(model, images, use_nested=use_nested) if use_tta else model(
                images,
                use_nested=use_nested,
            )["logits"]
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
    print(f"Use nested: {test_metrics['use_nested']}")


if __name__ == "__main__":
    main()
