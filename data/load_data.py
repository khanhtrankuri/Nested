import json
import os
import random
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
        file_name for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))
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
            f"Missing masks: {missing_masks[:5]}. Missing images: {missing_images[:5]}."
        )
    return image_names


def _build_image_transform(image_size=None, mean=IMAGE_MEAN, std=IMAGE_STD, augment: bool = False):
    steps = []
    if image_size is not None:
        steps.append(transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR))
    if augment:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.3),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, interpolation=InterpolationMode.BILINEAR),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


def _build_mask_transform(image_size=None, augment: bool = False):
    steps = []
    if image_size is not None:
        steps.append(transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST))
    if augment:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, interpolation=InterpolationMode.NEAREST),
        ])
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


class PolypDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, file_names: List[str], image_size=None, mean=IMAGE_MEAN, std=IMAGE_STD, augment: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_names = file_names
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.image_transform = _build_image_transform(image_size=image_size, mean=mean, std=std, augment=False)
        self.mask_transform = _build_mask_transform(image_size=image_size, augment=False)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int):
        file_name = self.file_names[index]
        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.augment:
                seed = torch.randint(0, 2**31 - 1, (1,)).item()
                random.seed(seed)
                torch.manual_seed(seed)
                image = _build_image_transform(self.image_size, self.mean, self.std, augment=True)(image)
            else:
                image = self.image_transform(image)
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            if self.augment:
                random.seed(seed)
                torch.manual_seed(seed)
                mask = _build_mask_transform(self.image_size, augment=True)(mask)
            else:
                mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        return {"image": image, "mask": mask, "file_name": file_name}


def _compute_mask_ratio(mask_path: str, image_size=None) -> float:
    mask_transform = _build_mask_transform(image_size=image_size, augment=False)
    with Image.open(mask_path) as mask:
        mask_tensor = mask_transform(mask.convert("L"))
        mask_tensor = (mask_tensor > 0.5).float()
    return float(mask_tensor.mean().item())


def _load_or_build_mask_ratio_cache(mask_dir: str, file_names: List[str], image_size=None, cache_path: Optional[str] = None) -> Dict[str, float]:
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            ratio_dict = json.load(f)
        if all(file_name in ratio_dict for file_name in file_names):
            return {file_name: float(ratio_dict[file_name]) for file_name in file_names}
    ratio_dict = {}
    for file_name in file_names:
        ratio_dict[file_name] = _compute_mask_ratio(os.path.join(mask_dir, file_name), image_size=image_size)
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ratio_dict, f, indent=2)
    return ratio_dict


def split_file_names_by_mask_size(image_dir: str, mask_dir: str, file_names: List[str], num_tasks: int = 4, image_size=None, descending: bool = True, cache_path: Optional[str] = None) -> List[Dict]:
    ratio_dict = _load_or_build_mask_ratio_cache(mask_dir=mask_dir, file_names=file_names, image_size=image_size, cache_path=cache_path)
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
        tasks.append({
            "task_id": task_index + 1,
            "file_names": [file_name for file_name, _ in group],
            "mask_ratios": [ratio for _, ratio in group],
            "num_samples": len(group),
        })
        start_index = end_index
    return tasks


def _build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0), drop_last=drop_last)


def build_replay_task_loaders(
    image_dir: str,
    mask_dir: str,
    task_infos: List[Dict],
    image_size=(448, 448),
    batch_size: int = 16,
    num_workers: int = 4,
    mean=IMAGE_MEAN,
    std=IMAGE_STD,
    replay_plan: Optional[Dict[int, Dict[int, float]]] = None,
):
    replay_plan = replay_plan or {}
    loaders = []
    for task in task_infos:
        task_id = task["task_id"]
        file_names = list(task["file_names"])
        own_count = len(file_names)
        extra_names = []
        plan = replay_plan.get(task_id, {})
        for prev_task_id, ratio in plan.items():
            prev_files = next(t["file_names"] for t in task_infos if t["task_id"] == prev_task_id)
            sample_size = max(1, int(own_count * ratio)) if ratio > 0 else 0
            if sample_size > 0:
                if sample_size <= len(prev_files):
                    extra_names.extend(random.sample(prev_files, sample_size))
                else:
                    extra_names.extend(random.choices(prev_files, k=sample_size))
        dataset = PolypDataset(image_dir=image_dir, mask_dir=mask_dir, file_names=file_names + extra_names, image_size=image_size, mean=mean, std=std, augment=True)
        loaders.append(_build_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers))
    return loaders


def build_dataloaders(file_path: str, image_size=(352, 352), num_tasks: int = 4, val_size: float = 0.2, batch_size: int = 8, num_workers: int = 4, seed: int = 42, mean=IMAGE_MEAN, std=IMAGE_STD, descending: bool = True, train_augmentation: bool = False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    random.seed(seed)
    train_image_dir = os.path.join(file_path, "train/images")
    train_mask_dir = os.path.join(file_path, "train/masks")
    test_image_dir = os.path.join(file_path, "test/images")
    test_mask_dir = os.path.join(file_path, "test/masks")
    train_file_names = _validate_pairs(train_image_dir, train_mask_dir)
    test_file_names = _validate_pairs(test_image_dir, test_mask_dir)
    inner_train_names, val_names = train_test_split(train_file_names, test_size=val_size, random_state=seed, shuffle=True)
    ratio_cache_path = os.path.join(file_path, f"mask_ratio_cache_{image_size[0]}x{image_size[1]}.json")
    task_infos = split_file_names_by_mask_size(train_image_dir, train_mask_dir, inner_train_names, num_tasks=num_tasks, image_size=image_size, descending=descending, cache_path=ratio_cache_path)
    train_task_loaders = []
    for task_info in task_infos:
        task_dataset = PolypDataset(train_image_dir, train_mask_dir, task_info["file_names"], image_size=image_size, mean=mean, std=std, augment=train_augmentation)
        train_task_loaders.append(_build_loader(task_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False))
    train_dataset_full = PolypDataset(train_image_dir, train_mask_dir, inner_train_names, image_size=image_size, mean=mean, std=std, augment=train_augmentation)
    train_loader_full = _build_loader(train_dataset_full, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = PolypDataset(train_image_dir, train_mask_dir, val_names, image_size=image_size, mean=mean, std=std, augment=False)
    val_loader = _build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = PolypDataset(test_image_dir, test_mask_dir, test_file_names, image_size=image_size, mean=mean, std=std, augment=False)
    test_loader = _build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    meta_info = {
        "num_total_train_samples": len(train_file_names),
        "num_inner_train_samples": len(inner_train_names),
        "num_val_samples": len(val_names),
        "num_test_samples": len(test_file_names),
        "train_image_dir": train_image_dir,
        "train_mask_dir": train_mask_dir,
        "task_infos": task_infos,
    }
    return train_task_loaders, train_loader_full, val_loader, test_loader, meta_info
