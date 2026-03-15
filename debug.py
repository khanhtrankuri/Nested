import torch

from data import build_dataloaders
from model import BaselinePolypModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path="datasets/Kvasir",
        image_size=(358, 358),
        num_tasks=4,
        val_size=0.2,
        batch_size=4,
        num_workers=2,
        seed=42,
        descending=True,
    )

    model = BaselinePolypModel(
        backbone_name="unet",
        in_channels=3,
        out_channels=1,
        base_channels=64,
        bilinear=True,
    ).to(device)

    batch = next(iter(train_loader_full))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    outputs = model(images)
    logits = outputs["logits"]

    print("images :", images.shape)   # [B, 3, H, W]
    print("masks  :", masks.shape)    # [B, 1, H, W]
    print("logits :", logits.shape)   # [B, 1, H, W]


if __name__ == "__main__":
    main()