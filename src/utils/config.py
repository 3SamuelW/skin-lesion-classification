from pathlib import Path

import yaml


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    """Save the resolved config used by an experiment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def apply_data_root(config, data_root):
    """Override image_dir / mask_dir / label_csv in config with a new data root.

    Usage:
        config = apply_data_root(config, "data")
    This sets:
        config["data"]["image_dir"] = "data/image"
        config["data"]["mask_dir"]  = "data/mask"
        config["data"]["label_csv"] = "data/label.csv"
    """
    config = {**config}
    config["data"] = {**config["data"]}
    config["data"]["image_dir"] = str(Path(data_root) / "image")
    config["data"]["mask_dir"] = str(Path(data_root) / "mask")
    config["data"]["label_csv"] = str(Path(data_root) / "label.csv")
    return config
