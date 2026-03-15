import json
import os

import torch

from data import build_dataloaders
from model import NestedLitePolypModel
from loss import BCEDiceLoss
from engine.eval_nested_modes import sweep_thresholds, evaluate_nested_mode


def _extract_checkpoint_payload(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_kwargs = checkpoint.get("model_kwargs", {})
        return state_dict, model_kwargs

    return checkpoint, {}


def _infer_model_kwargs_from_state_dict(state_dict):
    model_kwargs = {"backbone_name": "unet_nl"}

    inc_weight = state_dict.get("backbone.inc.block.0.weight")
    if inc_weight is not None and inc_weight.ndim == 4:
        model_kwargs["in_channels"] = inc_weight.shape[1]
        model_kwargs["base_channels"] = inc_weight.shape[0]

    out_weight = state_dict.get("backbone.outc.conv.weight")
    if out_weight is not None and out_weight.ndim == 4:
        model_kwargs["out_channels"] = out_weight.shape[0]

    slow_memory = state_dict.get("slow_memory")
    if slow_memory is not None and slow_memory.ndim == 1:
        model_kwargs["memory_dim"] = slow_memory.numel()

    updater_first = state_dict.get("memory_updater.delta_mlp.0.weight")
    if updater_first is not None and updater_first.ndim == 2:
        model_kwargs["updater_hidden_dim"] = updater_first.shape[0]

    model_kwargs["use_gate"] = any(
        key.startswith("memory_updater.gate_mlp.") for key in state_dict
    )
    model_kwargs["bilinear"] = not any(
        key.startswith("backbone.up1.up.weight") for key in state_dict
    )

    return model_kwargs


def main():
    file_path = "datasets/Kvasir"
    ckpt_path = "outputs/kvasir_nested_dual_memory_bottleneck/best_model.pth"
    save_root = "outputs/eval_best_checkpoint_modes"
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path=file_path,
        image_size=(358, 358),
        num_tasks=4,
        val_size=0.2,
        batch_size=32,
        num_workers=4,
        seed=42,
        descending=True,
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict, saved_model_kwargs = _extract_checkpoint_payload(checkpoint)

    model_kwargs = {
        "backbone_name": "unet_nl",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 32,
        "bilinear": True,
        "memory_dim": 64,
        "updater_hidden_dim": 128,
        "use_gate": True,
        "fast_init_std": 5e-2,
        "slow_init_std": 1e-2,
    }
    model_kwargs.update(_infer_model_kwargs_from_state_dict(state_dict))
    model_kwargs.update(saved_model_kwargs)

    print(f"Loading checkpoint with model kwargs: {model_kwargs}")

    model = NestedLitePolypModel(**model_kwargs).to(device)

    model.load_state_dict(state_dict)

    criterion = BCEDiceLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1.0,
    )

    modes = ["adaptive_slow", "adaptive_fresh", "static_slow"]
    summary = {}

    for mode in modes:
        print(f"\n========== MODE: {mode} ==========\n")

        best_val_result, all_val_results = sweep_thresholds(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            thresholds=[round(x, 2) for x in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]],
            use_amp=True,
            mode=mode,
        )

        best_thr = best_val_result["threshold"]
        print(
            f"[Best Val][{mode}] "
            f"thr={best_thr:.2f} | "
            f"dice={best_val_result['dice']:.4f} | "
            f"iou={best_val_result['iou']:.4f}"
        )

        test_result = evaluate_nested_mode(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=best_thr,
            use_amp=True,
            mode=mode,
            save_dir=os.path.join(save_root, f"{mode}_predictions"),
        )

        print(
            f"[Test][{mode}] "
            f"thr={best_thr:.2f} | "
            f"dice={test_result['dice']:.4f} | "
            f"iou={test_result['iou']:.4f} | "
            f"precision={test_result['precision']:.4f} | "
            f"recall={test_result['recall']:.4f}"
        )

        summary[mode] = {
            "best_val": best_val_result,
            "all_val_results": all_val_results,
            "test": test_result,
        }

    with open(os.path.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {os.path.join(save_root, 'summary.json')}")


if __name__ == "__main__":
    main()
