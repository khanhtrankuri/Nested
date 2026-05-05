"""
Inference script to visualize outputs from each block in the Nested Learning model.

Usage:
    python infer.py --model-path <path_to_checkpoint> --image-path <path_to_image> [--encoder convnext_base] [--output-dir results]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize block outputs of Nested model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--encoder", type=str, default="convnext_base", help="Encoder name")
    parser.add_argument("--img-size", type=int, default=384, help="Input image size")
    parser.add_argument("--output-dir", type=str, default="infer_results", help="Output directory")
    parser.add_argument("--no-nested", action="store_true", help="Disable nested refinement")
    return parser.parse_args()


def load_image(image_path, img_size):
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    # Resize
    image = image.resize((img_size, img_size), Image.BILINEAR)
    # To tensor [0,1]
    image = torch.from_numpy(np.array(image)).float() / 255.0
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image.permute(2, 0, 1) - mean) / std
    return image.unsqueeze(0)  # [1, 3, H, W]


def visualize_feature_map(feature, title, save_path):
    """Visualize feature map as heatmap."""
    if feature.dim() == 4:
        # Average across channels
        fmap = feature[0].mean(dim=0).cpu().numpy()
    else:
        fmap = feature.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(fmap, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_all_channels(feature, title, save_path, max_channels=16):
    """Visualize up to max_channels feature maps in a grid."""
    if feature.dim() == 4:
        B, C, H, W = feature.shape
        n_channels = min(C, max_channels)
        rows = int(np.ceil(n_channels / 4))
        cols = min(4, n_channels)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_channels):
            ax = axes[i // cols, i % cols]
            fmap = feature[0, i].cpu().numpy()
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Ch {i}", fontsize=8)
            ax.axis("off")

        # Hide unused subplots
        for i in range(n_channels, rows * cols):
            axes[i // cols, i % cols].axis("off")

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def visualize_prediction(logits, title, save_path, original_image=None):
    """Visualize prediction logits/map."""
    prob = torch.sigmoid(logits[0, 0]).cpu().numpy()

    if original_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Original image
        img = original_image[0].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        # Prediction heatmap
        im = axes[1].imshow(prob, cmap="jet")
        axes[1].set_title("Prediction Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        # Overlay
        axes[2].imshow(img)
        axes[2].imshow(prob, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(prob, cmap="jet")
        plt.colorbar()
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    from model.backbones.strong_baseline import StrongBaselinePolypModel

    model = StrongBaselinePolypModel(
        encoder_name=args.encoder,
        use_pretrained=False,
        enable_nested=True,
    )

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load image
    print(f"Loading image from {args.image_path}")
    image = load_image(args.image_path, args.img_size)
    image = image.to(device)

    # Store original for visualization
    orig_image = image.clone()

    # Register hooks to capture intermediate outputs
    features = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # Hook encoder outputs
    encoder = model.encoder
    if hasattr(encoder, "features"):  # Torchvision ConvNeXt
        hooks.append(encoder.features[1].register_forward_hook(make_hook("c2")))
        hooks.append(encoder.features[3].register_forward_hook(make_hook("c3")))
        hooks.append(encoder.features[5].register_forward_hook(make_hook("c4")))
        hooks.append(encoder.features[7].register_forward_hook(make_hook("c5")))
    else:
        print("Warning: Could not hook encoder stages automatically")

    # Hook decoder outputs
    if hasattr(model.decoder, "lateral2"):
        hooks.append(model.decoder.lateral2.register_forward_hook(make_hook("p2")))
        hooks.append(model.decoder.lateral3.register_forward_hook(make_hook("p3")))
        hooks.append(model.decoder.lateral4.register_forward_hook(make_hook("p4")))
        hooks.append(model.decoder.lateral5.register_forward_hook(make_hook("p5")))

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(image, use_nested=not args.no_nested)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Visualize results
    print("Visualizing results...")

    # 1. Visualize encoder features (c2, c3, c4, c5)
    for stage_name in ["c2", "c3", "c4", "c5"]:
        if stage_name in features:
            fmap = features[stage_name]
            print(f"  {stage_name}: shape={tuple(fmap.shape)}")
            save_path = os.path.join(args.output_dir, f"{stage_name}_feature.png")
            visualize_all_channels(fmap, f"Encoder {stage_name.upper()} Features", save_path)
            # Also save averaged feature map
            avg_save_path = os.path.join(args.output_dir, f"{stage_name}_avg.png")
            visualize_feature_map(fmap, f"Encoder {stage_name.upper()} (Avg)", avg_save_path)

    # 2. Visualize decoder features (p2, p3, p4, p5)
    for stage_name in ["p2", "p3", "p4", "p5"]:
        if stage_name in features:
            fmap = features[stage_name]
            print(f"  {stage_name}: shape={tuple(fmap.shape)}")
            save_path = os.path.join(args.output_dir, f"{stage_name}_feature.png")
            visualize_all_channels(fmap, f"Decoder {stage_name.upper()} Features", save_path)

    # 3. Visualize predictions
    # Coarse prediction
    if "coarse_logits" in outputs:
        coarse = outputs["coarse_logits"]
        print(f"  coarse_logits: shape={tuple(coarse.shape)}")
        save_path = os.path.join(args.output_dir, "coarse_prediction.png")
        visualize_prediction(coarse, "Coarse Prediction", save_path, orig_image)

    # Refined prediction
    if "logits" in outputs:
        logits = outputs["logits"]
        print(f"  logits: shape={tuple(logits.shape)}")
        save_path = os.path.join(args.output_dir, "refined_prediction.png")
        visualize_prediction(logits, "Refined Prediction", save_path, orig_image)

    # Aux prediction
    if "aux_logits" in outputs:
        aux = outputs["aux_logits"]
        print(f"  aux_logits: shape={tuple(aux.shape)}")
        save_path = os.path.join(args.output_dir, "aux_prediction.png")
        visualize_prediction(aux, "Auxiliary Prediction", save_path, orig_image)

    # 4. Create summary figure
    print("Creating summary figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    img = orig_image[0].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # Coarse prediction
    if "coarse_logits" in outputs:
        coarse_prob = torch.sigmoid(outputs["coarse_logits"][0, 0]).cpu().numpy()
        im = axes[0, 1].imshow(coarse_prob, cmap="jet")
        axes[0, 1].set_title("Coarse Prediction")
        axes[0, 1].axis("off")
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Refined prediction
    if "logits" in outputs:
        refined_prob = torch.sigmoid(outputs["logits"][0, 0]).cpu().numpy()
        im = axes[0, 2].imshow(refined_prob, cmap="jet")
        axes[0, 2].set_title("Refined Prediction")
        axes[0, 2].axis("off")
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Feature maps
    for idx, stage in enumerate(["c2", "c3", "c4"]):
        if stage in features:
            fmap = features[stage][0].mean(dim=0).cpu().numpy()
            row = 1
            col = idx
            axes[row, col].imshow(fmap, cmap="viridis")
            axes[row, col].set_title(f"Feature {stage.upper()} (Avg)")
            axes[row, col].axis("off")

    # Difference map
    if "coarse_logits" in outputs and "logits" in outputs:
        diff = (torch.sigmoid(outputs["logits"]) - torch.sigmoid(outputs["coarse_logits"]))[0, 0].cpu().numpy()
        axes[1, 3].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[1, 3].set_title("Refinement Delta")
        axes[1, 3].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, "summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Print nested info if available
    if "nested_info" in outputs and outputs["nested_info"] is not None:
        print("\nNested Info:")
        for key, value in outputs["nested_info"].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")

    print(f"\nAll results saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
