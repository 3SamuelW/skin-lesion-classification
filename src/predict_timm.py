"""timm single-image inference entry point.

Usage:
    python src/predict_timm.py \
        --config config/timm_baseline.yaml \
        --experiment_id exp001 \
        --image_path /path/to/image.jpg \
        --mask_path  /path/to/mask.jpg

    # For a suite-trained checkpoint (nested subdirectory), specify --model_name:
    python src/predict_timm.py \
        --config config/timm_baseline.yaml \
        --experiment_id suite001 \
        --model_name vit_base_patch16_224 \
        --image_path /path/to/image.jpg \
        --mask_path  /path/to/mask.jpg

Outputs:
    outputs/timm_{experiment_id}[/{model_name}]/single_prediction.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.timm_baseline.data import prepare_single_tensor
from src.timm_baseline.model import load_model_from_checkpoint
from src.utils.config import apply_data_root, load_config
from src.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a single lesion image with a trained timm model."
    )
    parser.add_argument("--config",        required=True, help="Path to YAML config.")
    parser.add_argument("--experiment_id", required=True, help="Experiment id.")
    parser.add_argument("--image_path",    required=True, help="Input image path.")
    parser.add_argument("--mask_path",     required=True, help="Input mask path.")
    parser.add_argument("--model_name",    default=None,
                        help="Model name (needed when using a suite output).")
    parser.add_argument("--data_root",     default=None,
                        help="Dataset root dir (only affects config resolution).")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config = apply_data_root(config, args.data_root)

    base = Path(config["data"]["output_dir"]) / f"timm_{args.experiment_id}"
    if args.model_name is not None:
        base = base / args.model_name.replace("/", "_")
    output_dir = ensure_dir(base)

    checkpoint_path = output_dir / "model_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = config["timm"].get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    class_names       = checkpoint["class_names"]

    image_tensor = prepare_single_tensor(
        args.image_path, args.mask_path, config, device=device
    )
    with torch.no_grad():
        outputs       = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
        pred_idx      = int(probabilities.argmax())
        pred_label    = class_names[pred_idx]

    result = {
        "image_path": args.image_path,
        "mask_path":  args.mask_path,
        "pred_label": pred_label,
    }
    for idx, label in enumerate(class_names):
        result[f"prob_{label}"] = float(probabilities[idx])

    prediction_path = output_dir / "single_prediction.csv"
    pd.DataFrame([result]).to_csv(prediction_path, index=False)
    print(result)
    print(f"Saved to: {prediction_path}")


if __name__ == "__main__":
    main()
