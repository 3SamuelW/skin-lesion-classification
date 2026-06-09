"""SVM single-image prediction entry point.

Usage:
    python src/predict_svm.py \
        --config config/svm.yaml \
        --experiment_id svm_exp001 \
        --image_path /path/to/image.jpg \
        --mask_path  /path/to/mask.jpg

Outputs:
    outputs/{experiment_id}/single_prediction.csv
    Predicted class and per-class probabilities printed to stdout
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.features import extract_features
from src.model.svm import load_model_bundle
from src.utils.config import load_config
from src.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one lesion image with SVM.")
    parser.add_argument("--config",        required=True, help="Path to YAML config.")
    parser.add_argument("--experiment_id", required=True, help="Experiment id.")
    parser.add_argument("--image_path",    required=True, help="Input image path.")
    parser.add_argument("--mask_path",     required=True, help="Input mask path.")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    output_dir = ensure_dir(
        Path(config["data"]["output_dir"]) / args.experiment_id
    )
    bundle          = load_model_bundle(output_dir / "model.joblib")
    model           = bundle["model"]
    feature_columns = bundle["feature_columns"]

    feature_dict = extract_features(args.image_path, args.mask_path, config)
    X = pd.DataFrame([feature_dict]).reindex(columns=feature_columns, fill_value=0.0)

    pred_label = model.predict(X)[0]
    result = {
        "image_path": args.image_path,
        "mask_path":  args.mask_path,
        "pred_label": pred_label,
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        for label, prob in zip(model.classes_, probs):
            result[f"prob_{label}"] = float(prob)

    prediction_path = output_dir / "single_prediction.csv"
    pd.DataFrame([result]).to_csv(prediction_path, index=False)
    print(result)
    print(f"Saved to: {prediction_path}")


if __name__ == "__main__":
    main()
