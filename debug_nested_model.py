import torch

from data import build_dataloaders
from model import NestedLitePolypModel
from loss import BCEDiceLoss


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

    model = NestedLitePolypModel(
        backbone_name="unet_nl",
        in_channels=3,
        out_channels=1,
        base_channels=32,
        bilinear=True,
        memory_dim=64,
        updater_hidden_dim=128,
        use_gate=True,
    ).to(device)

    criterion = BCEDiceLoss()

    batch = next(iter(train_task_loaders[0]))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    memory = model.init_memory(batch_size=images.size(0), device=device, dtype=images.dtype)

    outputs = model(images, memory)
    logits = outputs["logits"]
    feat = outputs["feat"]

    loss = criterion(logits, masks)

    print("images :", images.shape)
    print("masks  :", masks.shape)
    print("memory :", memory.shape)
    print("feat   :", feat.shape)
    print("logits :", logits.shape)
    print("loss   :", loss.item())

    with torch.no_grad():
        loss_scalar = loss.detach().view(1, 1).repeat(images.size(0), 1)
        new_memory = model.update_memory(
            feat=feat.detach(),
            memory=memory.detach(),
            loss_scalar=loss_scalar,
        )

    print("new_memory:", new_memory.shape)


if __name__ == "__main__":
    main()