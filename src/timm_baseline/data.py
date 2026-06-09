"""Dataset and data-loading utilities for the timm baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dataloader.preprocessing import crop_to_mask


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD  = [0.229, 0.224, 0.225]


def build_class_mapping(metadata_df: pd.DataFrame):
    """Return (class_names, label_to_idx) with deterministic ordering."""
    class_names = sorted(metadata_df["label"].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    return class_names, label_to_idx


def build_transforms(img_size: int, train: bool, normalize: bool = True):
    """Build a torchvision transform pipeline.

    Training adds random flips, colour jitter and rotation.
    Validation/test uses resize-only.
    """
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomRotation(15),
        ])
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD))
    return transforms.Compose(steps)


def _prepare_pil_image(image_path, mask_path, use_mask_crop,
                       mask_threshold, mask_padding):
    image = Image.open(image_path).convert("RGB")
    if not use_mask_crop or not mask_path:
        return image
    mask = np.array(Image.open(mask_path).convert("L"))
    binary_mask = mask > mask_threshold
    if not binary_mask.any():
        raise ValueError(f"Empty lesion mask: {mask_path}")
    image_array = np.array(image)
    crop_image, _ = crop_to_mask(image_array, binary_mask, padding=mask_padding)
    if crop_image.size == 0:
        return image
    return Image.fromarray(crop_image.astype(np.uint8))


class LesionClassificationDataset(Dataset):
    """PyTorch Dataset that serves lesion images with their split metadata."""

    def __init__(self, metadata_df, label_to_idx, img_size, train,
                 use_mask_crop, mask_threshold, mask_padding, normalize=True):
        self.records = metadata_df.reset_index(drop=True).to_dict("records")
        self.label_to_idx = label_to_idx
        self.use_mask_crop = use_mask_crop
        self.mask_threshold = mask_threshold
        self.mask_padding = mask_padding
        self.transform = build_transforms(img_size, train=train, normalize=normalize)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = _prepare_pil_image(
            record["image_path"], record["mask_path"],
            self.use_mask_crop, self.mask_threshold, self.mask_padding,
        )
        image_tensor = self.transform(image)
        label_name = str(record["label"])
        label_idx  = int(self.label_to_idx[label_name])
        meta = {
            "image_id":       record["image_id"],
            "label":          label_name,
            "label_idx":      label_idx,
            "image_path":     record["image_path"],
            "mask_path":      record["mask_path"],
            "base_id":        record["base_id"],
            "augmentation_id": record["augmentation_id"],
        }
        return image_tensor, label_idx, meta


def build_split_loaders(metadata_df, split_df, config):
    """Build DataLoaders for train / val / test splits.

    Returns (loaders_dict, class_names, label_to_idx, merged_df).
    """
    timm_cfg = config["timm"]
    merged = metadata_df.merge(
        split_df[["image_id", "split"]],
        on="image_id", how="inner", validate="one_to_one",
    )
    class_names, label_to_idx = build_class_mapping(merged)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        split_part = merged[merged["split"] == split_name].copy()
        dataset = LesionClassificationDataset(
            split_part,
            label_to_idx=label_to_idx,
            img_size=int(timm_cfg.get("img_size", 224)),
            train=split_name == "train",
            use_mask_crop=bool(timm_cfg.get("use_mask_crop", True)),
            mask_threshold=int(timm_cfg.get("mask_threshold", 127)),
            mask_padding=int(timm_cfg.get("mask_padding", 4)),
            normalize=bool(timm_cfg.get("normalize", True)),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=int(timm_cfg.get("batch_size", 32)),
            shuffle=split_name == "train",
            num_workers=int(timm_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders, class_names, label_to_idx, merged


def prepare_single_tensor(image_path, mask_path, config, device=None):
    """Prepare a single-image tensor for inference (mirrors test-time preprocessing)."""
    timm_cfg = config["timm"]
    image = _prepare_pil_image(
        image_path, mask_path,
        bool(timm_cfg.get("use_mask_crop", True)),
        int(timm_cfg.get("mask_threshold", 127)),
        int(timm_cfg.get("mask_padding", 4)),
    )
    transform = build_transforms(
        int(timm_cfg.get("img_size", 224)),
        train=False,
        normalize=bool(timm_cfg.get("normalize", True)),
    )
    tensor = transform(image).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
