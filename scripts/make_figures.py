# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate figures from a trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.data.make_dataset import make_dataset
from src.models.plots import plot_calibration_curve, plot_pr_curves, plot_roc_curves
from src.utils.config import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript figures")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging()

    paths = config["paths"]
    _, X_test, _, y_test = make_dataset(
        raw_path=paths["data"],
        target_col=config["dataset"]["target_col"],
        test_size=config["split"]["test_size"],
        random_state=config["random_state"],
    )

    model_path = Path(paths["results"]) / "best_model.joblib"
    model = joblib.load(model_path)

    plot_roc_curves(model, X_test, y_test, config["class_labels"], paths["figures"])
    plot_pr_curves(model, X_test, y_test, config["class_labels"], paths["figures"])
    plot_calibration_curve(
        model, X_test, y_test, config["class_labels"], paths["figures"]
    )


if __name__ == "__main__":
    main()
