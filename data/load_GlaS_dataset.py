import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _normalize_image_size(image_size) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must be int or (H, W), got: {image_size}")


def _parse_glas_folder(root_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Parse GlaS flat folder into official splits.

    Returns dict with keys 'train', 'testA', 'testB'.
    Each value is a sorted list of (image_path, mask_path) tuples.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"GlaS dataset folder not found: {root_dir}")

    all_files = set(os.listdir(root_dir))
    splits: Dict[str, List[Tuple[str, str]]] = {"train": [], "testA": [], "testB": []}

    for fname in sorted(all_files):
        if not fname.endswith(".bmp") or "_anno" in fname:
            continue

        base = fname[:-4]  # remove .bmp
        mask_name = f"{base}_anno.bmp"
        if mask_name not in all_files:
            raise FileNotFoundError(f"Mask not found for image {fname}: expected {mask_name}")

        image_path = os.path.join(root_dir, fname)
        mask_path = os.path.join(root_dir, mask_name)

        if base.startswith("train_"):
            splits["train"].append((image_path, mask_path))
        elif base.startswith("testA_"):
            splits["testA"].append((image_path, mask_path))
        elif base.startswith("testB_"):
            splits["testB"].append((image_path, mask_path))
        else:
            raise ValueError(f"Unknown split prefix for file: {fname}")

    for split_name, pairs in splits.items():
        if not pairs:
            raise ValueError(f"No images found for split '{split_name}' in {root_dir}")

    return splits


class GlaSDataset(Dataset):
    """GlaS (Gland Segmentation) dataset.

    Handles instance-level annotation masks by converting them to binary
    (any non-zero pixel = gland foreground).
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        mean: Tuple[float, ...] = IMAGE_MEAN,
        std: Tuple[float, ...] = IMAGE_STD,
    ):
        self.pairs = list(pairs)
        self.image_size = _normalize_image_size(image_size)
        self.augment = augment
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.pairs)

    def _resize_pair(self, image: Image.Image, mask: Image.Image):
        image = image.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size[1], self.image_size[0]), resample=Image.NEAREST)
        return image, mask

    def _apply_flip(self, image, mask):
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < 0.15:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        return image, mask

    def _apply_affine(self, image, mask):
        angle = random.uniform(-20.0, 20.0)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return image, mask

    def _apply_color_aug(self, image: Image.Image):
        if random.random() < 0.8:
            image = ImageOps.autocontrast(image)
        if random.random() < 0.8:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.25))
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.20))
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.20))
        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.4)))
        return image

    def _to_tensor(self, image: Image.Image, mask: Image.Image):
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=2)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mean = torch.tensor(self.mean, dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Instance mask → binary mask: any non-zero pixel is foreground
        mask_np = np.asarray(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        file_name = os.path.basename(image_path)
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image = image.convert("RGB")
            mask = mask.convert("L")
            if self.augment:
                image, mask = self._apply_flip(image, mask)
                image, mask = self._apply_affine(image, mask)
                image = self._apply_color_aug(image)
            image, mask = self._resize_pair(image, mask)
            image_tensor, mask_tensor = self._to_tensor(image, mask)
        return {"image": image_tensor, "mask": mask_tensor, "file_name": file_name}


def build_glas_dataloaders(
    root_dir: str,
    image_size=(512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    train_augmentation: bool = True,
    val_ratio: float = 0.0,
):
    """Build DataLoaders for GlaS dataset using official splits.

    Args:
        root_dir: Path to Warwick_QU_Dataset folder.
        image_size: (H, W) to resize images.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducibility.
        train_augmentation: Whether to apply augmentation on train set.
        val_ratio: Fraction of train set to hold out as validation (0.0 = no val split,
                   use testA as val). When > 0, splits train into train/val.

    Returns:
        train_loader, val_loader, testA_loader, testB_loader, meta_info
    """
    image_size = _normalize_image_size(image_size)
    splits = _parse_glas_folder(root_dir)

    train_pairs = splits["train"]
    testA_pairs = splits["testA"]
    testB_pairs = splits["testB"]

    if val_ratio > 0:
        random.seed(seed)
        shuffled = list(train_pairs)
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_pairs = shuffled[:n_val]
        train_pairs = shuffled[n_val:]
    else:
        val_pairs = testA_pairs

    train_ds = GlaSDataset(train_pairs, image_size=image_size, augment=train_augmentation)
    val_ds = GlaSDataset(val_pairs, image_size=image_size, augment=False)
    testA_ds = GlaSDataset(testA_pairs, image_size=image_size, augment=False)
    testB_ds = GlaSDataset(testB_pairs, image_size=image_size, augment=False)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    testA_loader = DataLoader(
        testA_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    testB_loader = DataLoader(
        testB_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    meta_info = {
        "dataset": "GlaS",
        "root_dir": root_dir,
        "image_size": list(image_size),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_testA": len(testA_ds),
        "num_testB": len(testB_ds),
        "val_source": "train_split" if val_ratio > 0 else "testA",
        "val_ratio": val_ratio,
        "seed": seed,
    }
    return train_loader, val_loader, testA_loader, testB_loader, meta_info
