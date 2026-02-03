"""Generate tables from a trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.make_dataset import make_dataset
from src.models.evaluate import evaluate_model
from src.utils.config import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript tables")
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

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_labels=config["class_labels"],
    )

    tables_dir = Path(paths["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame(metrics["classification_report"]).to_csv(
        tables_dir / "classification_report.csv"
    )


if __name__ == "__main__":
    main()