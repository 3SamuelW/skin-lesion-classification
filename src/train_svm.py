"""SVM lesion classifier training entry point.

Usage:
    python src/train_svm.py --config config/svm.yaml --experiment_id svm_exp001
    python src/train_svm.py --config config/svm.yaml --experiment_id svm_exp001 --data_root /path/to/dataset

Arguments:
    --config          Path to YAML config file
    --experiment_id   Experiment id; outputs saved to outputs/{experiment_id}/
    --data_root       Optional dataset root directory (overrides image_dir/mask_dir/label_csv in config)
    --reuse_features  Reuse features.csv if it already exists in the output directory

Outputs:
    outputs/{experiment_id}/
        config.yaml            Full config snapshot for this run
        features.csv           Extracted feature matrix
        split.csv              train/val/test split assignments
        model.joblib           Trained SVM pipeline
        metrics.json           train/val/test metrics
        predictions.csv        Per-sample predictions
        robustness_detail.csv  Augmentation robustness detail
        confusion_matrix.png   Test-set confusion matrix
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.dataset import build_metadata, validate_metadata
from src.dataloader.features import extract_feature_table
from src.dataloader.split import create_grouped_split, split_features
from src.model.svm import save_model_bundle, train_svm
from src.utils.config import apply_data_root, load_config, save_config
from src.utils.evaluation import (
    augmentation_robustness,
    build_predictions_frame,
    evaluate_predictions,
    save_confusion_matrix,
)
from src.utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SVM lesion classifier.")
    parser.add_argument("--config",        required=True, help="Path to YAML config.")
    parser.add_argument("--experiment_id", required=True,
                        help="Experiment id. Outputs saved to outputs/{experiment_id}/.")
    parser.add_argument("--data_root",     default=None,
                        help="Dataset root dir (e.g. data_2). "
                             "Overrides config image_dir/mask_dir/label_csv.")
    parser.add_argument("--reuse_features", action="store_true",
                        help="Reuse features.csv if it already exists.")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config = apply_data_root(config, args.data_root)
        print(f"[data_root] Using dataset: {args.data_root}")

    output_dir = ensure_dir(
        Path(config["data"]["output_dir"]) / args.experiment_id
    )

    # ---- Metadata ----
    metadata = build_metadata(config)
    has_aug  = (metadata["augmentation_id"] != "original").any()
    validate_metadata(metadata, strict_groups=has_aug)

    # ---- Features ----
    features_path = output_dir / "features.csv"
    if args.reuse_features and features_path.exists():
        print(f"Reusing features from {features_path}")
        features_df = pd.read_csv(features_path)
    else:
        features_df = extract_feature_table(metadata, config)
        features_df.to_csv(features_path, index=False)

    labels = sorted(metadata["label"].unique().tolist())

    # ---- Split ----
    split_df   = create_grouped_split(metadata, config)
    split_data = split_features(features_df, split_df)
    X_train, y_train, train_meta = split_data["train"]
    X_val,   y_val,   val_meta   = split_data["val"]
    X_test,  y_test,  test_meta  = split_data["test"]

    print(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, "
          f"test: {len(X_test)}")

    # ---- Train ----
    model = train_svm(
        X_train, y_train,
        groups=train_meta["base_id"].to_numpy(),
        config=config,
    )

    # ---- Evaluate ----
    metrics           = {}
    prediction_frames = []
    for split_name, X, y, meta in [
        ("train", X_train, y_train, train_meta),
        ("val",   X_val,   y_val,   val_meta),
        ("test",  X_test,  y_test,  test_meta),
    ]:
        if len(X) == 0:
            metrics[split_name] = {
                "accuracy": None, "balanced_accuracy": None,
                "macro_precision": None, "macro_recall": None,
                "macro_f1": None, "classification_report": {},
            }
            continue
        y_pred  = model.predict(X)
        y_prob  = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        metrics[split_name] = evaluate_predictions(y, y_pred, labels)
        frame = build_predictions_frame(meta, y_pred, y_prob, model.classes_)
        frame["split"] = split_name
        prediction_frames.append(frame)
        print(
            f"  {split_name:5s} | acc={metrics[split_name]['accuracy']:.4f} | "
            f"bal_acc={metrics[split_name]['balanced_accuracy']:.4f} | "
            f"macro_f1={metrics[split_name]['macro_f1']:.4f}"
        )

    predictions      = pd.concat(prediction_frames, ignore_index=True)
    test_predictions = predictions[predictions["split"] == "test"].copy()

    if not test_predictions.empty:
        robustness        = augmentation_robustness(test_predictions)
        robustness_detail = robustness.pop("detail")
        metrics["test"]["augmentation_robustness"] = robustness
    else:
        robustness_detail = pd.DataFrame()

    # ---- Save outputs ----
    save_config(config, output_dir / "config.yaml")
    split_df.to_csv(output_dir / "split.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    robustness_detail.to_csv(output_dir / "robustness_detail.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")

    if not test_predictions.empty:
        save_confusion_matrix(
            test_predictions["label"],
            test_predictions["pred_label"],
            labels,
            output_dir / "confusion_matrix.png",
        )

    save_model_bundle(
        model,
        X_train.columns.tolist(),
        output_dir / "model.joblib",
    )

    print(f"\nExperiment finished: {output_dir}")
    if not test_predictions.empty:
        print(f"Test macro F1:         {metrics['test']['macro_f1']:.4f}")
        print(f"Test balanced accuracy: {metrics['test']['balanced_accuracy']:.4f}")
    else:
        print(f"Val macro F1:          {metrics['val']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
