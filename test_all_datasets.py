"""
Evaluate a checkpoint on every dataset subfolder inside --test-root.

Expected layout:
    <test-root>/
        CVC-300/
            images/
            masks/
        CVC-ClinicDB/
            images/
            masks/
        ...

Results are written to:
    <save-root>/
        CVC-300/test_metrics.json
        CVC-ClinicDB/test_metrics.json
        ...
        summary.json          # all datasets in one file
"""

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on all dataset subfolders inside --test-root"
    )
    parser.add_argument(
        "--test-root",
        default="datasets/TestDataset",
        help="Parent folder that contains one subfolder per dataset (each with images/ and masks/).",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-root", default="outputs/test_all")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.set_defaults(save_predictions=False)

    parser.add_argument("--strict-load", action="store_true")

    # Fallback model config (used only when checkpoint has no model_kwargs)
    parser.add_argument(
        "--encoder-name",
        choices=["tiny_convnext", "convnext_tiny", "convnext_small", "convnext_base", "pvtv2_b2"],
        default="convnext_tiny",
    )
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


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _discover_datasets(test_root: str) -> list:
    """Return sorted list of (dataset_name, dataset_path) for every valid subfolder."""
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"--test-root not found: {test_root}")
    found = []
    for name in sorted(os.listdir(test_root)):
        dataset_path = os.path.join(test_root, name)
        if not os.path.isdir(dataset_path):
            continue
        images_dir = os.path.join(dataset_path, "images")
        masks_dir = os.path.join(dataset_path, "masks")
        if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
            found.append((name, dataset_path))
    if not found:
        raise ValueError(f"No valid dataset subfolders (images/ + masks/) found under: {test_root}")
    return found


# ---------------------------------------------------------------------------
# Checkpoint resolution helpers  (same logic as test_baseline.py)
# ---------------------------------------------------------------------------

def _normalize_image_size(image_size) -> Tuple[int, int]:
    if image_size is None:
        return (384, 384)
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    return (int(image_size[0]), int(image_size[1]))


def _default_model_kwargs(args) -> Dict:
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


def _infer_model_kwargs_from_state_dict(state_dict: Dict) -> Dict:
    inferred = {}
    lateral2_w = state_dict.get("decoder.lateral2.weight")
    lateral3_w = state_dict.get("decoder.lateral3.weight")
    lateral4_w = state_dict.get("decoder.lateral4.weight")
    lateral5_w = state_dict.get("decoder.lateral5.weight")
    if all(isinstance(w, torch.Tensor) and w.ndim == 4 for w in (lateral2_w, lateral3_w, lateral4_w, lateral5_w)):
        enc_ch = (int(lateral2_w.shape[1]), int(lateral3_w.shape[1]), int(lateral4_w.shape[1]), int(lateral5_w.shape[1]))
        if enc_ch == (64, 128, 256, 512):
            inferred["encoder_name"] = "tiny_convnext"
        elif enc_ch == (64, 128, 320, 512):
            inferred["encoder_name"] = "pvtv2_b2"
        elif enc_ch == (128, 256, 512, 1024):
            inferred["encoder_name"] = "convnext_base"
        elif enc_ch == (96, 192, 384, 768):
            stage3 = {
                int(k[len(p):].split(".", 1)[0])
                for k in state_dict
                for p in ("encoder.features.5.", "features.5.")
                if k.startswith(p) and k[len(p):].split(".", 1)[0].isdigit()
            }
            inferred["encoder_name"] = "convnext_small" if len(stage3) >= 20 else "convnext_tiny"
    seg_w = state_dict.get("seg_head.1.weight")
    if isinstance(seg_w, torch.Tensor) and seg_w.ndim == 4:
        inferred["decoder_channels"] = int(seg_w.shape[1])
    inferred["enable_nested"] = any(k.startswith("nested_refiner.") for k in state_dict)
    return inferred


def _resolve_model_kwargs(args, payload: Dict) -> Dict:
    resolved = _default_model_kwargs(args)
    ckpt_kwargs = payload.get("model_kwargs", {}) if isinstance(payload, dict) else {}
    for k, v in ckpt_kwargs.items():
        if k in {"use_pretrained", "strict_pretrained", "pretrained_cache_dir", "pretrained_loaded"}:
            continue
        resolved[k] = v
    if not ckpt_kwargs and isinstance(payload, dict):
        sd = payload.get("state_dict")
        if isinstance(sd, dict):
            resolved.update(_infer_model_kwargs_from_state_dict(sd))
    return resolved


def _resolve_image_size(args, payload: Dict) -> Tuple[int, int]:
    if args.image_size is not None:
        return _normalize_image_size(args.image_size)
    tc = payload.get("train_config", {}) if isinstance(payload, dict) else {}
    if isinstance(tc, dict) and tc.get("image_size") is not None:
        return _normalize_image_size(tc["image_size"])
    return (384, 384)


def _resolve_threshold(args, payload: Dict) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    bv = payload.get("best_val", {}) if isinstance(payload, dict) else {}
    if isinstance(bv, dict) and bv.get("threshold") is not None:
        return float(bv["threshold"])
    return 0.5


def _resolve_use_tta(args, payload: Dict) -> bool:
    if args.use_tta is not None:
        return bool(args.use_tta)
    tc = payload.get("train_config", {}) if isinstance(payload, dict) else {}
    return bool(tc.get("use_tta", False)) if isinstance(tc, dict) else False


def _resolve_tta_scales(args, payload: Dict) -> Tuple[float, ...]:
    if args.tta_scales:
        return tuple(float(s) for s in args.tta_scales)
    tc = payload.get("train_config", {}) if isinstance(payload, dict) else {}
    if isinstance(tc, dict) and tc.get("tta_scales"):
        return tuple(float(s) for s in tc["tta_scales"])
    return (1.0,)


def _resolve_use_nested(args, payload: Dict, model_kwargs: Dict) -> bool:
    if not bool(model_kwargs.get("enable_nested", False)):
        return False
    if args.use_nested is not None:
        return bool(args.use_nested)
    if isinstance(payload, dict) and payload.get("best_nested_active") is not None:
        return bool(payload["best_nested_active"])
    return False


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _save_predictions(logits: torch.Tensor, file_names, save_dir: str, threshold: float):
    probs = torch.sigmoid(logits).detach().cpu()
    preds = (probs > threshold).to(torch.uint8).squeeze(1).numpy() * 255
    for pred, name in zip(preds, file_names):
        Image.fromarray(pred, mode="L").save(os.path.join(save_dir, name))


@torch.no_grad()
def evaluate_dataset(
    model,
    dataset_path: str,
    dataset_name: str,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    criterion,
    device: str,
    threshold: float,
    use_tta: bool,
    tta_scales: Sequence[float],
    use_nested: bool,
    save_dir: Optional[str],
) -> Dict:
    loader, meta = build_standalone_loader(
        file_path=dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
    )
    model.eval()
    loss_m = AverageMeter()
    dice_m = AverageMeter()
    iou_m = AverageMeter()
    prec_m = AverageMeter()
    rec_m = AverageMeter()
    amp_enabled = torch.cuda.is_available() and str(device).startswith("cuda")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    total = len(loader)
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = (
                _forward_with_tta(model, images, use_nested=use_nested, tta_scales=tta_scales)
                if use_tta
                else model(images, use_nested=use_nested)["logits"]
            )
            loss, _ = criterion({"logits": logits}, masks, return_components=True)

        m = compute_segmentation_metrics(logits, masks, threshold=threshold)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        dice_m.update(m["dice"], bs)
        iou_m.update(m["iou"], bs)
        prec_m.update(m["precision"], bs)
        rec_m.update(m["recall"], bs)

        if save_dir is not None:
            _save_predictions(logits, batch["file_name"], save_dir, threshold)

        print(
            f"  [{dataset_name}] {step}/{total} | "
            f"dice={dice_m.avg:.4f} | iou={iou_m.avg:.4f}"
        )

    return {
        "loss": loss_m.avg,
        "dice": dice_m.avg,
        "iou": iou_m.avg,
        "precision": prec_m.avg,
        "recall": rec_m.avg,
        "threshold": float(threshold),
        "use_tta": bool(use_tta),
        "tta_scales": [float(s) for s in tta_scales],
        "use_nested": bool(use_nested),
        "num_samples": meta["num_samples"],
        "num_batches": int(total),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(summary: Dict[str, Dict]):
    col_w = 16
    header = f"{'Dataset':<{col_w}}{'Dice':>8}{'IoU':>8}{'Precision':>12}{'Recall':>10}{'Samples':>10}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("SUMMARY — all datasets")
    print(sep)
    print(header)
    print(sep)
    for name, m in summary.items():
        print(
            f"{name:<{col_w}}"
            f"{m['dice']:>8.4f}"
            f"{m['iou']:>8.4f}"
            f"{m['precision']:>12.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['num_samples']:>10}"
        )
    print(sep)

    # Macro average across datasets
    keys = ["dice", "iou", "precision", "recall"]
    avgs = {k: sum(m[k] for m in summary.values()) / len(summary) for k in keys}
    total_samples = sum(m["num_samples"] for m in summary.values())
    print(
        f"{'MEAN':<{col_w}}"
        f"{avgs['dice']:>8.4f}"
        f"{avgs['iou']:>8.4f}"
        f"{avgs['precision']:>12.4f}"
        f"{avgs['recall']:>10.4f}"
        f"{total_samples:>10}"
    )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint once ---
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

    model_kwargs = _resolve_model_kwargs(args, payload if isinstance(payload, dict) else {})
    image_size = _resolve_image_size(args, payload if isinstance(payload, dict) else {})
    threshold = _resolve_threshold(args, payload if isinstance(payload, dict) else {})
    use_tta = _resolve_use_tta(args, payload if isinstance(payload, dict) else {})
    tta_scales = _resolve_tta_scales(args, payload if isinstance(payload, dict) else {})
    use_nested = _resolve_use_nested(args, payload if isinstance(payload, dict) else {}, model_kwargs)

    print(f"[Config] encoder={model_kwargs['encoder_name']} | image_size={image_size}")
    print(f"[Config] threshold={threshold:.2f} | use_tta={use_tta} | tta_scales={list(tta_scales)} | use_nested={use_nested}")

    model = StrongBaselinePolypModel(**model_kwargs).to(device)
    incompatible = model.load_state_dict(state_dict, strict=args.strict_load)
    if not args.strict_load and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"[Checkpoint] missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)}")
    model.eval()

    criterion = StrongBaselineLoss()

    # --- Discover datasets ---
    datasets = _discover_datasets(args.test_root)
    print(f"\nFound {len(datasets)} dataset(s) under: {args.test_root}")
    for name, _ in datasets:
        print(f"  • {name}")
    print()

    # --- Evaluate each dataset ---
    summary = {}
    for dataset_name, dataset_path in datasets:
        print(f"\n>>> Evaluating: {dataset_name}")
        dataset_save_root = os.path.join(args.save_root, dataset_name)
        os.makedirs(dataset_save_root, exist_ok=True)

        pred_dir = os.path.join(dataset_save_root, "predictions") if args.save_predictions else None
        metrics = evaluate_dataset(
            model=model,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            criterion=criterion,
            device=device,
            threshold=threshold,
            use_tta=use_tta,
            tta_scales=tta_scales,
            use_nested=use_nested,
            save_dir=pred_dir,
        )
        summary[dataset_name] = metrics

        with open(os.path.join(dataset_save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  → dice={metrics['dice']:.4f} | iou={metrics['iou']:.4f} | saved to {dataset_save_root}")

    # --- Save summary ---
    summary_path = os.path.join(args.save_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
