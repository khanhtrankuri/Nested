import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _normalize_image_size(image_size) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must be int or (H, W), got: {image_size}")


def _list_files(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    file_names = sorted(
        file_name
        for file_name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file_name))
    )
    if not file_names:
        raise ValueError(f"No files found in: {folder_path}")
    return file_names


def _strip_mask_suffix(name: str) -> str:
    """Bỏ hậu tố _mask trước phần mở rộng. VD: 'foo_mask.png' -> 'foo.png'."""
    base, ext = os.path.splitext(name)
    if base.endswith("_mask"):
        return base[:-5] + ext
    return name


def _validate_pairs(image_dir: str, mask_dir: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Kiểm tra image/mask khớp nhau, hỗ trợ mask có hậu tố _mask.

    Returns:
        image_names: danh sách tên file ảnh (dùng làm file_names chính).
        mask_map: dict {image_name -> mask_file_name}.
                  Nếu tên giống nhau thì mask_map[x] == x.
    """
    image_names = _list_files(image_dir)
    mask_names  = _list_files(mask_dir)

    normalised_masks = [_strip_mask_suffix(n) for n in mask_names]
    mask_map = dict(zip(normalised_masks, mask_names))   # norm_name -> raw_mask_name

    image_set  = set(image_names)
    normed_set = set(normalised_masks)

    if image_set != normed_set:
        missing_masks  = sorted(image_set  - normed_set)
        missing_images = sorted(normed_set - image_set)
        raise ValueError(
            "Image/mask files do not match. "
            f"Missing masks: {missing_masks[:5]}. "
            f"Missing images: {missing_images[:5]}."
        )

    return image_names, mask_map


@dataclass
class SplitInfo:
    train_files: List[str]
    val_files: List[str]
    test_files: List[str]


def _compute_mask_ratio(mask_dir: str, mask_file_name: str) -> float:
    mask_path = os.path.join(mask_dir, mask_file_name)
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        mask_np = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return float(mask_np.mean())


def _build_ratio_dict(
    mask_dir: str,
    file_names: Sequence[str],
    mask_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Tính mask ratio cho mỗi file.
    mask_map: {image_name -> mask_file_name}. None = tên giống nhau.
    Key trong dict trả về luôn là image_name (không phải mask_file_name).
    """
    mask_map = mask_map or {}
    return {
        file_name: _compute_mask_ratio(
            mask_dir,
            mask_map.get(file_name, file_name),
        )
        for file_name in file_names
    }


def _build_stratify_labels(
    file_names: Sequence[str],
    ratio_dict: Dict[str, float],
    max_bins: int,
    min_count_per_bin: int,
) -> Optional[List[int]]:
    ratios = np.asarray([ratio_dict[file_name] for file_name in file_names], dtype=np.float32)
    if ratios.size == 0:
        return None
    for num_bins in range(max_bins, 1, -1):
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.unique(np.quantile(ratios, quantiles))
        if edges.size < 3:
            continue
        labels = np.digitize(ratios, bins=edges[1:-1], right=True)
        counts = np.bincount(labels, minlength=edges.size - 1)
        if counts.min() >= min_count_per_bin:
            return labels.tolist()
    return None


def _split_with_optional_stratify(
    file_names: Sequence[str],
    test_size: float,
    seed: int,
    ratio_dict: Optional[Dict[str, float]] = None,
    min_count_per_bin: int = 2,
) -> Tuple[List[str], List[str]]:
    file_names = list(file_names)
    stratify = None
    if ratio_dict is not None:
        if isinstance(test_size, float):
            holdout_size = max(1, int(round(len(file_names) * test_size)))
        else:
            holdout_size = int(test_size)
        max_bins = max(2, min(5, holdout_size))
        stratify = _build_stratify_labels(
            file_names=file_names,
            ratio_dict=ratio_dict,
            max_bins=max_bins,
            min_count_per_bin=min_count_per_bin,
        )
    train_files, holdout_files = train_test_split(
        file_names,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_files, holdout_files


def build_strict_split(
    file_names: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    ratio_dict: Optional[Dict[str, float]] = None,
) -> SplitInfo:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")
    file_names = list(file_names)
    train_files, test_files = _split_with_optional_stratify(
        file_names=file_names,
        test_size=test_ratio,
        seed=seed,
        ratio_dict=ratio_dict,
    )
    rel_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_ratio_dict = None if ratio_dict is None else {name: ratio_dict[name] for name in train_files}
    train_files, val_files = _split_with_optional_stratify(
        file_names=train_files,
        test_size=rel_val_ratio,
        seed=seed,
        ratio_dict=train_ratio_dict,
    )
    return SplitInfo(train_files=train_files, val_files=val_files, test_files=test_files)


def build_kfold_split(
    file_names: Sequence[str],
    fold_index: int,
    num_folds: int = 5,
    seed: int = 42,
    ratio_dict: Optional[Dict[str, float]] = None,
) -> SplitInfo:
    file_names = list(file_names)
    splitter = None
    if ratio_dict is not None:
        labels = _build_stratify_labels(
            file_names=file_names,
            ratio_dict=ratio_dict,
            max_bins=5,
            min_count_per_bin=num_folds,
        )
        if labels is not None:
            splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed).split(file_names, labels)
    if splitter is None:
        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed).split(file_names)
    folds = list(splitter)
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds)-1}]")
    train_val_idx, test_idx = folds[fold_index]
    train_val_files = [file_names[i] for i in train_val_idx]
    test_files = [file_names[i] for i in test_idx]
    train_val_ratio_dict = None if ratio_dict is None else {name: ratio_dict[name] for name in train_val_files}
    train_files, val_files = _split_with_optional_stratify(
        file_names=train_val_files,
        test_size=0.1111111111,
        seed=seed,
        ratio_dict=train_val_ratio_dict,
    )
    return SplitInfo(train_files=train_files, val_files=val_files, test_files=test_files)


def _build_small_polyp_sampler(
    file_names: Sequence[str],
    ratio_dict: Dict[str, float],
    sampling_power: float,
) -> Optional[WeightedRandomSampler]:
    if sampling_power <= 0:
        return None
    weights = []
    for file_name in file_names:
        ratio = max(float(ratio_dict[file_name]), 1e-4)
        weights.append((1.0 / ratio) ** float(sampling_power))
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    weights_tensor = weights_tensor / weights_tensor.mean().clamp(min=1e-8)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)


class CleanPolypDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        file_names: Sequence[str],
        mask_map: Optional[Dict[str, str]] = None,   # {image_name -> mask_file_name}
        image_size=(384, 384),
        augment: bool = False,
        mean=IMAGE_MEAN,
        std=IMAGE_STD,
    ):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.file_names = list(file_names)
        self.mask_map   = mask_map or {}              # empty = tên giống nhau
        self.image_size = _normalize_image_size(image_size)
        self.augment = augment
        self.mean = mean
        self.std  = std

    def __len__(self):
        return len(self.file_names)

    def _resize_pair(self, image: Image.Image, mask: Image.Image):
        image = image.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        mask  = mask.resize((self.image_size[1], self.image_size[0]), resample=Image.NEAREST)
        return image, mask

    def _apply_flip(self, image, mask):
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask  = ImageOps.mirror(mask)
        if random.random() < 0.15:
            image = ImageOps.flip(image)
            mask  = ImageOps.flip(mask)
        return image, mask

    def _apply_affine(self, image, mask):
        angle = random.uniform(-20.0, 20.0)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        mask  = mask.rotate(angle,  resample=Image.NEAREST,  fillcolor=0)
        return image, mask

    def _apply_random_crop(self, image, mask):
        if random.random() > 0.5:
            return image, mask
        w, h = image.size
        scale  = random.uniform(0.65, 1.0)
        ratio  = random.uniform(0.8, 1.25)
        crop_w = int(min(w, max(w * 0.3, w * scale * ratio ** 0.5)))
        crop_h = int(min(h, max(h * 0.3, h * scale / ratio ** 0.5)))
        x0 = random.randint(0, max(0, w - crop_w))
        y0 = random.randint(0, max(0, h - crop_h))
        image = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        mask  = mask.crop((x0, y0, x0 + crop_w, y0 + crop_h))
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

    def _to_tensor(self, image: Image.Image, mask: Image.Image, apply_specular: bool = False):
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=2)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        if apply_specular:
            image_tensor = self._apply_specular(image_tensor)
        mean = torch.tensor(self.mean, dtype=image_tensor.dtype).view(3, 1, 1)
        std  = torch.tensor(self.std,  dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy((mask_np > 0.5).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor

    def _apply_specular(self, image_tensor: torch.Tensor):
        if random.random() > 0.35:
            return image_tensor
        _, h, w = image_tensor.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h),
            torch.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        overlay = torch.zeros((1, h, w), dtype=image_tensor.dtype)
        for _ in range(random.randint(1, 3)):
            cx      = random.uniform(-0.7, 0.7)
            cy      = random.uniform(-0.7, 0.7)
            sigma_x = random.uniform(0.05, 0.22)
            sigma_y = random.uniform(0.05, 0.22)
            amp     = random.uniform(0.15, 0.40)
            gauss   = torch.exp(
                -(((xx - cx) ** 2) / (2 * sigma_x ** 2) + ((yy - cy) ** 2) / (2 * sigma_y ** 2))
            )
            overlay = torch.maximum(overlay, amp * gauss.unsqueeze(0))
        return torch.clamp(image_tensor + overlay, 0.0, 1.0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_name  = self.file_names[index]
        image_path = os.path.join(self.image_dir, file_name)
        # Dùng mask_map để tìm đúng tên file mask (có thể khác tên ảnh)
        mask_file  = self.mask_map.get(file_name, file_name)
        mask_path  = os.path.join(self.mask_dir, mask_file)
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image = image.convert("RGB")
            mask  = mask.convert("L")
            if self.augment:
                image, mask = self._apply_flip(image, mask)
                image, mask = self._apply_random_crop(image, mask)
                image, mask = self._apply_affine(image, mask)
                image = self._apply_color_aug(image)
            image, mask = self._resize_pair(image, mask)
            image_tensor, mask_tensor = self._to_tensor(image, mask, apply_specular=self.augment)
        return {"image": image_tensor, "mask": mask_tensor, "file_name": file_name}


# ---------------------------------------------------------------------------
# build_clean_dataloaders  (tự split)
# ---------------------------------------------------------------------------

def build_clean_dataloaders(
    file_path: str,
    image_size=(384, 384),
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    protocol: str = "strict",
    fold_index: int = 0,
    num_folds: int = 5,
    train_augmentation: bool = True,
    stratified_split: bool = True,
    small_polyp_sampling_power: float = 0.0,
):
    image_dir = os.path.join(file_path, "images")
    mask_dir  = os.path.join(file_path, "masks")

    file_names, mask_map = _validate_pairs(image_dir, mask_dir)

    ratio_dict = (
        _build_ratio_dict(mask_dir, file_names, mask_map)
        if (stratified_split or small_polyp_sampling_power > 0)
        else None
    )

    if protocol == "strict":
        split_info = build_strict_split(
            file_names=file_names,
            seed=seed,
            ratio_dict=ratio_dict if stratified_split else None,
        )
    elif protocol == "kfold":
        split_info = build_kfold_split(
            file_names=file_names,
            fold_index=fold_index,
            num_folds=num_folds,
            seed=seed,
            ratio_dict=ratio_dict if stratified_split else None,
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    train_ds = CleanPolypDataset(image_dir, mask_dir, split_info.train_files, mask_map=mask_map, image_size=image_size, augment=train_augmentation)
    val_ds   = CleanPolypDataset(image_dir, mask_dir, split_info.val_files,   mask_map=mask_map, image_size=image_size, augment=False)
    test_ds  = CleanPolypDataset(image_dir, mask_dir, split_info.test_files,  mask_map=mask_map, image_size=image_size, augment=False)

    pin_memory         = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_ratio_dict = (
        None if ratio_dict is None
        else {name: ratio_dict[name] for name in split_info.train_files}
    )
    train_sampler = (
        None if train_ratio_dict is None
        else _build_small_polyp_sampler(
            split_info.train_files,
            ratio_dict=train_ratio_dict,
            sampling_power=small_polyp_sampling_power,
        )
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers)

    meta_info = {
        "protocol": protocol,
        "fold_index": fold_index,
        "num_folds": num_folds,
        "num_train": len(train_ds),
        "num_val":   len(val_ds),
        "num_test":  len(test_ds),
        "image_size": list(_normalize_image_size(image_size)),
        "seed": seed,
        "stratified_split": bool(stratified_split),
        "small_polyp_sampling_power": float(small_polyp_sampling_power),
    }
    return train_loader, val_loader, test_loader, meta_info


# ---------------------------------------------------------------------------
# build_standalone_loader
# ---------------------------------------------------------------------------

def build_standalone_loader(
    file_path: str,
    image_size=(384, 384),
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = False,
    small_polyp_sampling_power: float = 0.0,
):
    image_dir = os.path.join(file_path, "images")
    mask_dir  = os.path.join(file_path, "masks")

    file_names, mask_map = _validate_pairs(image_dir, mask_dir)

    ratio_dict = (
        _build_ratio_dict(mask_dir, file_names, mask_map)
        if (augment and small_polyp_sampling_power > 0)
        else None
    )

    dataset = CleanPolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_names=file_names,
        mask_map=mask_map,
        image_size=image_size,
        augment=augment,
    )
    sampler = (
        None if ratio_dict is None
        else _build_small_polyp_sampler(
            file_names=file_names,
            ratio_dict=ratio_dict,
            sampling_power=small_polyp_sampling_power,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(augment and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    meta_info = {
        "file_path": file_path,
        "num_samples": len(dataset),
        "image_size": list(_normalize_image_size(image_size)),
        "augment": bool(augment),
        "small_polyp_sampling_power": float(small_polyp_sampling_power),
    }
    return loader, meta_info


# ---------------------------------------------------------------------------
# build_presplit_dataloaders  (folder đã chia sẵn train/val/test)
# ---------------------------------------------------------------------------
# Cấu trúc folder mong đợi:
#   root/
#     train/  images/  masks/
#     val/    images/  masks/
#     test/   images/  masks/
#
# Mask có thể có hậu tố _mask (VD: foo_mask.png) hoặc tên giống ảnh (foo.png).
# ---------------------------------------------------------------------------

def build_presplit_dataloaders(
    root: str,
    image_size=(384, 384),
    batch_size: int = 8,
    num_workers: int = 4,
    train_augmentation: bool = True,
    small_polyp_sampling_power: float = 0.0,
    train_subdir: str = "train",
    val_subdir: str = "val",
    test_subdir: str = "test",
):
    """
    Tạo DataLoader từ folder đã chia sẵn train/val/test.

    Args:
        root: Đường dẫn đến folder gốc (vd: "processed_ebhi").
        image_size: Kích thước ảnh (H, W) hoặc int.
        batch_size: Số ảnh mỗi batch.
        num_workers: Số worker cho DataLoader.
        train_augmentation: Bật augmentation cho tập train.
        small_polyp_sampling_power: > 0 để oversample polyp nhỏ ở train.
        train_subdir / val_subdir / test_subdir: Tên subfolder.

    Returns:
        train_loader, val_loader, test_loader, meta_info
    """
    splits = {
        "train": (train_subdir, train_augmentation),
        "val":   (val_subdir,   False),
        "test":  (test_subdir,  False),
    }

    loaders = {}
    counts  = {}

    pin_memory         = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    for split_name, (subdir, augment) in splits.items():
        image_dir = os.path.join(root, subdir, "images")
        mask_dir  = os.path.join(root, subdir, "masks")

        file_names, mask_map = _validate_pairs(image_dir, mask_dir)

        # Weighted sampler chỉ dùng cho train
        sampler = None
        if split_name == "train" and small_polyp_sampling_power > 0:
            ratio_dict = _build_ratio_dict(mask_dir, file_names, mask_map)
            sampler = _build_small_polyp_sampler(
                file_names=file_names,
                ratio_dict=ratio_dict,
                sampling_power=small_polyp_sampling_power,
            )

        dataset = CleanPolypDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            file_names=file_names,
            mask_map=mask_map,
            image_size=image_size,
            augment=augment,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train" and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        loaders[split_name] = loader
        counts[split_name]  = len(dataset)

    meta_info = {
        "root": root,
        "num_train": counts["train"],
        "num_val":   counts["val"],
        "num_test":  counts["test"],
        "image_size": list(_normalize_image_size(image_size)),
        "train_augmentation": train_augmentation,
        "small_polyp_sampling_power": float(small_polyp_sampling_power),
    }

    return loaders["train"], loaders["val"], loaders["test"], meta_info