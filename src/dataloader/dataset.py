from pathlib import Path

import pandas as pd
from PIL import Image


def parse_base_id(image_id):
    """Strip augmentation suffix to get the original image id.

    Augmented files follow the naming convention:
        <base_id>_aug1.jpg, <base_id>_aug2.jpg, ...
    This function strips everything after the first recognised suffix token.
    In practice the dataset uses _aug1 / _aug2 for data/ and explicit geometry
    names for data; base_id is stored in label.csv so we only need
    to strip the final augmentation tag.
    """
    image_id = str(image_id)
    for suffix in ("_aug1", "_aug2", "_aug3", "_aug4", "_aug5",
                   "_flip_h", "_flip_v", "_rot90", "_rot180", "_rot270",
                   "_bright", "_gamma", "_clahe", "_blur"):
        if image_id.endswith(suffix):
            return image_id[: -len(suffix)]
    return image_id


def parse_augmentation_id(image_id):
    image_id = str(image_id)
    for suffix in ("_aug1", "_aug2", "_aug3", "_aug4", "_aug5",
                   "_flip_h", "_flip_v", "_rot90", "_rot180", "_rot270",
                   "_bright", "_gamma", "_clahe", "_blur"):
        if image_id.endswith(suffix):
            return suffix.lstrip("_")
    return "original"


def build_metadata(config):
    """Build a unified metadata DataFrame from label.csv + image/mask dirs.

    Returns a DataFrame with columns:
        image_id, label, image_path, mask_path,
        base_id, is_augmented, augmentation_id
    """
    data_config = config["data"]
    image_dir = Path(data_config["image_dir"])
    mask_dir = Path(data_config["mask_dir"])
    label_csv = Path(data_config["label_csv"])

    df = pd.read_csv(label_csv)
    if not {"image_id", "dx"}.issubset(df.columns):
        raise ValueError("label.csv must contain 'image_id' and 'dx' columns.")

    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["dx"].astype(str)
    df["base_id"] = df["image_id"].apply(parse_base_id)
    df["augmentation_id"] = df["image_id"].apply(parse_augmentation_id)
    df["is_augmented"] = df["augmentation_id"] != "original"
    df["image_path"] = df["image_id"].apply(lambda x: str(image_dir / f"{x}.jpg"))
    df["mask_path"] = df["image_id"].apply(lambda x: str(mask_dir / f"mask_{x}.jpg"))
    return df[
        [
            "image_id",
            "label",
            "image_path",
            "mask_path",
            "base_id",
            "is_augmented",
            "augmentation_id",
        ]
    ]


def validate_metadata(df, strict_groups=True):
    """Validate that every image and mask file exists and sizes match.

    When strict_groups=True (default when augmented data is present) also
    checks that every base_id has the same set of augmentation variants.
    """
    errors = []

    for row in df.itertuples(index=False):
        image_path = Path(row.image_path)
        mask_path = Path(row.mask_path)
        if not image_path.exists():
            errors.append(f"Missing image: {image_path}")
            continue
        if not mask_path.exists():
            errors.append(f"Missing mask: {mask_path}")
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                errors.append(
                    f"Image/mask size mismatch for {row.image_id}: "
                    f"{image.size} vs {mask.size}"
                )

    if strict_groups:
        expected_aug_ids = None
        for base_id, group in df.groupby("base_id"):
            aug_ids = set(group["augmentation_id"])
            if "original" not in aug_ids:
                errors.append(f"base_id={base_id} missing original image")
                continue
            if expected_aug_ids is None:
                expected_aug_ids = aug_ids
            elif aug_ids != expected_aug_ids:
                errors.append(
                    f"base_id={base_id} has augmentations {sorted(aug_ids)}, "
                    f"expected {sorted(expected_aug_ids)}"
                )

    if errors:
        sample = "\n".join(errors[:20])
        raise ValueError(
            f"Metadata validation failed with {len(errors)} errors:\n{sample}"
        )

    return True
