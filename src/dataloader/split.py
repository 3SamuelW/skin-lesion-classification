import pandas as pd
from sklearn.model_selection import train_test_split


def create_grouped_split(df, config):
    """Split the dataset by base_id to prevent augmentation leakage.

    The split is stratified by label so that class proportions are preserved
    in every partition.  All augmented variants of an original image are
    guaranteed to end up in the same split as the original.

    Config key path:  config[config["model"]]["split"]
    Required keys:    train_size, val_size, test_size  (must sum to 1.0)
    Optional keys:    random_state (default 42)
    """
    split_cfg = config[config["model"]]["split"]
    train_size = split_cfg.get("train_size", 0.8)
    val_size = split_cfg.get("val_size", 0.1)
    test_size = split_cfg.get("test_size", 0.1)
    random_state = split_cfg.get("random_state", 42)

    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    group_labels = (
        df.groupby("base_id")["label"]
        .agg(lambda labels: labels.value_counts().index[0])
        .reset_index()
    )

    train_groups, temp_groups = train_test_split(
        group_labels,
        train_size=train_size,
        random_state=random_state,
        stratify=group_labels["label"],
    )

    relative_val_size = val_size / (val_size + test_size)
    if relative_val_size >= 1.0:
        val_groups = temp_groups
        test_groups = temp_groups.iloc[0:0]
    else:
        val_groups, test_groups = train_test_split(
            temp_groups,
            train_size=relative_val_size,
            random_state=random_state,
            stratify=temp_groups["label"],
        )

    split_map = {}
    split_map.update({base_id: "train" for base_id in train_groups["base_id"]})
    split_map.update({base_id: "val" for base_id in val_groups["base_id"]})
    split_map.update({base_id: "test" for base_id in test_groups["base_id"]})

    split_df = df.copy()
    split_df["split"] = split_df["base_id"].map(split_map)
    if split_df["split"].isna().any():
        raise ValueError("Some samples were not assigned to a split.")

    _validate_no_group_leakage(split_df)
    return split_df


def _validate_no_group_leakage(split_df):
    leakage = (
        split_df.groupby("base_id")["split"]
        .nunique()
        .loc[lambda values: values > 1]
    )
    if len(leakage) > 0:
        raise ValueError(
            f"Group leakage detected for base_id: {list(leakage.index)}"
        )


def split_features(features_df, split_df):
    """Attach split labels to a features DataFrame and return per-split tuples.

    Returns a dict:
        {
            "train": (X_train, y_train, meta_train),
            "val":   (X_val,   y_val,   meta_val),
            "test":  (X_test,  y_test,  meta_test),
        }
    where X_* are pure feature DataFrames and meta_* contain all metadata cols.
    """
    merged = features_df.merge(
        split_df[["image_id", "split"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    metadata_cols = {
        "image_id", "label", "image_path", "mask_path",
        "base_id", "is_augmented", "augmentation_id", "split",
    }
    feature_cols = [col for col in merged.columns if col not in metadata_cols]

    result = {}
    for split_name in ["train", "val", "test"]:
        part = merged[merged["split"] == split_name].copy()
        result[split_name] = (
            part[feature_cols],
            part["label"],
            part,
        )
    return result
