import json
import os
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


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


def _validate_pairs(image_dir: str, mask_dir: str) -> List[str]:
    image_names = _list_files(image_dir)
    mask_names = _list_files(mask_dir)

    if image_names != mask_names:
        missing_masks = sorted(set(image_names) - set(mask_names))
        missing_images = sorted(set(mask_names) - set(image_names))
        raise ValueError(
            "Image/mask files do not match. "
            f"Missing masks: {missing_masks[:5]}. "
            f"Missing images: {missing_images[:5]}."
        )

    return image_names


def _build_image_transform(image_size=None, mean=IMAGE_MEAN, std=IMAGE_STD):
    transform_steps = []

    if image_size is not None:
        transform_steps.append(
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR)
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(transform_steps)


def _build_mask_transform(image_size=None):
    transform_steps = []

    if image_size is not None:
        transform_steps.append(
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST)
        )

    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


class PolypDataset(Dataset):
    """
    Lazy-loading dataset:
    - chỉ đọc ảnh khi __getitem__
    - tiết kiệm RAM hơn rất nhiều so với preload toàn bộ
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        file_names: List[str],
        image_size=None,
        mean=IMAGE_MEAN,
        std=IMAGE_STD,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_names = file_names
        self.image_transform = _build_image_transform(
            image_size=image_size, mean=mean, std=std
        )
        self.mask_transform = _build_mask_transform(image_size=image_size)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int):
        file_name = self.file_names[index]
        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        with Image.open(image_path) as image:
            image = self.image_transform(image.convert("RGB"))

        with Image.open(mask_path) as mask:
            mask = self.mask_transform(mask.convert("L"))
            mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "file_name": file_name,
        }


def _compute_mask_ratio(mask_path: str, image_size=None) -> float:
    mask_transform = _build_mask_transform(image_size=image_size)

    with Image.open(mask_path) as mask:
        mask_tensor = mask_transform(mask.convert("L"))
        mask_tensor = (mask_tensor > 0.5).float()

    return float(mask_tensor.mean().item())


def _load_or_build_mask_ratio_cache(
    mask_dir: str,
    file_names: List[str],
    image_size=None,
    cache_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Cache mask_ratio để tránh tính lại mỗi lần chạy.
    """
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            ratio_dict = json.load(f)

        # kiểm tra đủ key
        if all(file_name in ratio_dict for file_name in file_names):
            return {file_name: float(ratio_dict[file_name]) for file_name in file_names}

    ratio_dict = {}
    for file_name in file_names:
        mask_path = os.path.join(mask_dir, file_name)
        ratio_dict[file_name] = _compute_mask_ratio(mask_path, image_size=image_size)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ratio_dict, f, indent=2)

    return ratio_dict


def split_file_names_by_mask_size(
    image_dir: str,
    mask_dir: str,
    file_names: List[str],
    num_tasks: int = 4,
    image_size=None,
    descending: bool = True,
    cache_path: Optional[str] = None,
) -> List[Dict]:
    """
    Chia một danh sách file thành nhiều task theo kích thước polyp.
    Thường dùng cho TRAIN split.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be greater than 0.")

    if len(file_names) < num_tasks:
        raise ValueError(
            f"Number of samples ({len(file_names)}) is smaller than num_tasks ({num_tasks})."
        )

    ratio_dict = _load_or_build_mask_ratio_cache(
        mask_dir=mask_dir,
        file_names=file_names,
        image_size=image_size,
        cache_path=cache_path,
    )

    ratio_info = [(file_name, ratio_dict[file_name]) for file_name in file_names]
    ratio_info.sort(key=lambda item: item[1], reverse=descending)

    total_samples = len(ratio_info)
    base_size, remainder = divmod(total_samples, num_tasks)

    tasks = []
    start_index = 0

    for task_index in range(num_tasks):
        current_size = base_size + (1 if task_index < remainder else 0)
        end_index = start_index + current_size
        group = ratio_info[start_index:end_index]

        tasks.append(
            {
                "task_id": task_index + 1,
                "file_names": [file_name for file_name, _ in group],
                "mask_ratios": [ratio for _, ratio in group],
                "num_samples": len(group),
            }
        )

        start_index = end_index

    return tasks


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )


def build_dataloaders(
    file_path: str,
    image_size=(352, 352),
    num_tasks: int = 4,
    val_size: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    mean=IMAGE_MEAN,
    std=IMAGE_STD,
    descending: bool = True,
):
    """
    Trả về:
    - train_task_loaders: list DataLoader theo task (cho nested learning)
    - train_loader_full: 1 DataLoader train đầy đủ (cho baseline nếu cần)
    - val_loader
    - test_loader
    - meta_info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    train_image_dir = os.path.join(file_path, "train/images")
    train_mask_dir = os.path.join(file_path, "train/masks")
    test_image_dir = os.path.join(file_path, "test/images")
    test_mask_dir = os.path.join(file_path, "test/masks")

    train_file_names = _validate_pairs(train_image_dir, train_mask_dir)
    test_file_names = _validate_pairs(test_image_dir, test_mask_dir)

    # 1) Chia TRAIN gốc thành train/val ở mức sample
    inner_train_names, val_names = train_test_split(
        train_file_names,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )

    # 2) Chia phần inner_train thành các task theo mask size
    ratio_cache_path = os.path.join(file_path, "mask_ratio_cache.json")
    task_infos = split_file_names_by_mask_size(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        file_names=inner_train_names,
        num_tasks=num_tasks,
        image_size=image_size,
        descending=descending,
        cache_path=ratio_cache_path,
    )

    # 3) Tạo dataset + loader cho từng task
    train_task_datasets = []
    train_task_loaders = []

    for task_info in task_infos:
        task_dataset = PolypDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            file_names=task_info["file_names"],
            image_size=image_size,
            mean=mean,
            std=std,
        )
        train_task_datasets.append(task_dataset)

        task_loader = _build_loader(
            dataset=task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
        train_task_loaders.append(task_loader)

    # 4) Nếu bạn muốn baseline train trên toàn bộ train split
    train_dataset_full = PolypDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        file_names=inner_train_names,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    train_loader_full = _build_loader(
        dataset=train_dataset_full,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    # 5) Validation loader
    val_dataset = PolypDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        file_names=val_names,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    val_loader = _build_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # 6) Test loader
    test_dataset = PolypDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        file_names=test_file_names,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    test_loader = _build_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    meta_info = {
        "num_total_train_samples": len(train_file_names),
        "num_inner_train_samples": len(inner_train_names),
        "num_val_samples": len(val_names),
        "num_test_samples": len(test_file_names),
        "task_infos": task_infos,
    }

    return train_task_loaders, train_loader_full, val_loader, test_loader, meta_info


if __name__ == "__main__":
    file_path = "datasets/Kvasir"

    train_task_loaders, train_loader_full, val_loader, test_loader, meta_info = build_dataloaders(
        file_path=file_path,
        image_size=(358, 358),
        num_tasks=4,
        val_size=0.2,
        batch_size=8,
        num_workers=4,
        seed=42,
        descending=True,   # Task 1 = polyp lớn nhất
    )

    print("===== META INFO =====")
    print(f"Total original train samples: {meta_info['num_total_train_samples']}")
    print(f"Inner train samples: {meta_info['num_inner_train_samples']}")
    print(f"Val samples: {meta_info['num_val_samples']}")
    print(f"Test samples: {meta_info['num_test_samples']}")

    print("\n===== TASK INFO =====")
    for task in meta_info["task_infos"]:
        ratios = task["mask_ratios"]
        print(
            f"Task {task['task_id']}: "
            f"{task['num_samples']} samples | "
            f"min={min(ratios):.6f}, "
            f"max={max(ratios):.6f}, "
            f"mean={sum(ratios)/len(ratios):.6f}"
        )

    # thử đọc 1 batch
    first_batch = next(iter(train_loader_full))
    print("\n===== ONE BATCH FROM TRAIN =====")
    print(first_batch["image"].shape)   # [B, 3, H, W]
    print(first_batch["mask"].shape)    # [B, 1, H, W]
    print(first_batch["file_name"][:3])