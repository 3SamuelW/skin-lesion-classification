"""timm single-model training entry point.

Usage:
    python src/train_timm.py --config config/timm_baseline.yaml --experiment_id exp001
    python src/train_timm.py --config config/timm_baseline.yaml --experiment_id exp001 --data_root /path/to/dataset

Arguments:
    --config          Path to YAML config file
    --experiment_id   Experiment id; outputs saved to outputs/timm_{experiment_id}/
    --data_root       Optional dataset root directory

Outputs:
    outputs/timm_{experiment_id}/
        config.yaml
        metadata.csv
        split.csv
        history.csv               Per-epoch train/val metrics
        model_best.pth            Best checkpoint
        metrics.json              train/val/test summary metrics
        predictions.csv
        robustness_detail.csv
        classification_report.txt
        confusion_matrix.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.dataset import build_metadata, validate_metadata
from src.dataloader.split import create_grouped_split
from src.timm_baseline.data import build_split_loaders
from src.timm_baseline.engine import (
    evaluate_model,
    save_classification_report_text,
    save_confusion_matrix,
    set_seed,
    train_model,
)
from src.timm_baseline.model import create_model, load_model_from_checkpoint
from src.utils.config import apply_data_root, load_config, save_config
from src.utils.evaluation import augmentation_robustness
from src.utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a timm baseline for lesion classification."
    )
    parser.add_argument("--config",        required=True, help="Path to YAML config.")
    parser.add_argument("--experiment_id", required=True,
                        help="Experiment id. Outputs saved to outputs/timm_{experiment_id}/.")
    parser.add_argument("--data_root",     default=None,
                        help="Dataset root dir (e.g. data). "
                             "Overrides config image_dir/mask_dir/label_csv.")
    return parser.parse_args()


def _print_summary(metrics, class_names):
    print("Timm split summary (accuracy / balanced_acc / macro_f1):")
    for split_name in ["train", "val", "test"]:
        m = metrics[split_name]
        print(f"  {split_name:5s}  acc={m['accuracy']:.4f}  "
              f"bal_acc={m['balanced_accuracy']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}")

    test_report = metrics["test"]["classification_report"]
    recall_parts = []
    for label in class_names:
        r = test_report.get(label, {})
        if "recall" in r:
            recall_parts.append(f"{label}={r['recall']:.4f}")
    if recall_parts:
        print("Test class recall: " + ", ".join(recall_parts))

    robustness    = metrics["test"].get("augmentation_robustness", {})
    consistency   = robustness.get("prediction_consistency")
    if consistency is not None:
        print(f"Test augmentation consistency: {consistency:.4f}")


def run_experiment(config, experiment_id, model_name_override=None):
    """Core training logic shared by single-model and suite entry points.

    When model_name_override is None the output directory is flat
    (outputs/timm_{experiment_id}/).  When called from the suite the output
    is nested under a per-model subdirectory.

    Returns (output_dir, metrics_dict).
    """
    if config.get("model") != "timm_baseline":
        raise ValueError(
            "Config must set  model: timm_baseline  for this training script."
        )

    timm_cfg   = dict(config["timm"])
    if model_name_override is not None:
        timm_cfg["model_name"] = model_name_override
    config     = {**config, "timm": timm_cfg}

    set_seed(int(timm_cfg.get("seed", 42)))

    model_name      = str(timm_cfg.get("model_name", "efficientnet_b0"))
    safe_model_name = model_name.replace("/", "_")

    if model_name_override is None:
        output_dir = ensure_dir(
            Path(config["data"]["output_dir"]) / f"timm_{experiment_id}"
        )
    else:
        output_dir = ensure_dir(
            Path(config["data"]["output_dir"])
            / f"timm_{experiment_id}"
            / safe_model_name
        )

    save_config(config, output_dir / "config.yaml")

    metadata = build_metadata(config)
    validate_metadata(metadata)

    # create_grouped_split expects config[config["model"]]["split"]
    # timm uses config["timm"]["split"], so we bridge here
    split_bridge = {"model": "svm", "svm": {"split": timm_cfg["split"]}}
    split_df     = create_grouped_split(metadata, split_bridge)
    split_df.to_csv(output_dir / "split.csv", index=False)

    loaders, class_names, label_to_idx, merged = build_split_loaders(
        metadata, split_df, config
    )
    merged.to_csv(output_dir / "metadata.csv", index=False)

    device = timm_cfg.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = create_model(
        model_name,
        num_classes=len(class_names),
        pretrained=bool(timm_cfg.get("pretrained", True)),
        dropout=float(timm_cfg.get("dropout", 0.0)),
    ).to(device)

    checkpoint_path = output_dir / "model_best.pth"
    history_df, best_info = train_model(
        model, loaders, device, config,
        class_names, label_to_idx, model_name, checkpoint_path,
    )
    history_df.to_csv(output_dir / "history.csv", index=False)
    print(
        f"Training finished. Best val score: {best_info['best_score']:.4f} "
        f"at epoch {best_info['best_epoch']}"
    )

    best_model, _ = load_model_from_checkpoint(checkpoint_path, device=device)

    metrics           = {}
    prediction_frames = []
    for split_name in ["train", "val", "test"]:
        split_metrics, predictions, labels, preds, probs = evaluate_model(
            best_model, loaders[split_name], device, class_names
        )
        metrics[split_name] = split_metrics
        predictions["split"] = split_name
        prediction_frames.append(predictions)

    all_predictions  = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(output_dir / "predictions.csv", index=False)

    test_predictions = all_predictions[all_predictions["split"] == "test"].copy()
    robustness       = augmentation_robustness(test_predictions)
    robustness_detail = robustness.pop("detail")
    if not robustness_detail.empty:
        robustness_detail.to_csv(output_dir / "robustness_detail.csv", index=False)
    metrics["test"]["augmentation_robustness"] = robustness

    save_json(metrics, output_dir / "metrics.json")
    save_classification_report_text(
        test_predictions["label"],
        test_predictions["pred_label"],
        class_names,
        output_dir / "classification_report.txt",
        title=f"Classification Report — {model_name}",
    )
    save_confusion_matrix(
        test_predictions["label"],
        test_predictions["pred_label"],
        class_names,
        output_dir / "confusion_matrix.png",
    )

    print(f"Experiment finished: {output_dir}")
    _print_summary(metrics, class_names)
    return output_dir, metrics


def main():
    args   = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config = apply_data_root(config, args.data_root)
        print(f"[data_root] Using dataset: {args.data_root}")
    run_experiment(config, args.experiment_id)


if __name__ == "__main__":
    main()
