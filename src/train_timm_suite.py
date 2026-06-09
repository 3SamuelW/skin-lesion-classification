"""timm batch training suite — runs the same training pipeline across multiple architectures.

Usage:
    # Run all 4 default models
    python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id suite001

    # Run a subset of models
    python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id suite001 \
        --models efficientnet_b0 resnet18

    # Specify a dataset root
    python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id suite001 \
        --data_root /path/to/dataset

Default model list:
    efficientnet_b0
    resnet18
    vit_base_patch16_224
    samvit_base_patch16

Outputs:
    outputs/timm_{experiment_id}/
        suite_summary.csv          Comparison table across all models
        {model_name}/              Per-model output directory (same layout as train_timm.py)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_timm import run_experiment
from src.utils.config import apply_data_root, load_config


DEFAULT_MODELS = [
    "efficientnet_b0",
    "resnet18",
    "vit_base_patch16_224",
    "samvit_base_patch16",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run timm baseline suite across multiple models."
    )
    parser.add_argument("--config",        required=True, help="Path to YAML config.")
    parser.add_argument("--experiment_id", required=True,
                        help="Suite id. Results saved under outputs/timm_{experiment_id}/.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="Optional subset of model names to run.")
    parser.add_argument("--data_root", default=None,
                        help="Dataset root dir (e.g. data_2).")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config = apply_data_root(config, args.data_root)
        print(f"[data_root] Using dataset: {args.data_root}")

    suite_root = (
        Path(config["data"]["output_dir"]) / f"timm_{args.experiment_id}"
    )
    suite_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name in args.models:
        print("=" * 80)
        print(f"Running model: {model_name}")
        model_config = {**config, "timm": {**config["timm"], "model_name": model_name}}
        output_dir, metrics = run_experiment(
            model_config, args.experiment_id, model_name_override=model_name
        )
        summary_rows.append({
            "model_name":            model_name,
            "output_dir":            str(output_dir),
            "test_accuracy":         metrics["test"]["accuracy"],
            "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
            "test_macro_f1":         metrics["test"]["macro_f1"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = suite_root / "suite_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("=" * 80)
    print(f"Suite finished. Summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
