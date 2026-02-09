# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Run the full preprocessing, training, evaluation, and explainability pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.make_dataset import make_dataset
from src.features.build_features import build_preprocessor
from src.interpret.shap_explain import explain_with_shap
from src.models.evaluate import evaluate_model, summarize_cv_metrics
from src.models.plots import plot_calibration_curve, plot_pr_curves, plot_roc_curves
from src.models.train import train_with_cv
from src.utils.config import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end model pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


def run_pipeline(config_path: str) -> None:
    """Run the pipeline for a single configuration file."""
    config = load_config(config_path)
    setup_logging()

    paths = config["paths"]
    X_train, X_test, y_train, y_test = make_dataset(
        raw_path=paths["data"],
        target_col=config["dataset"]["target_col"],
        test_size=config["split"]["test_size"],
        random_state=config["random_state"],
    )

    preprocessor = build_preprocessor(
        numeric_features=config["dataset"]["numeric_features"],
        categorical_features=config["dataset"]["categorical_features"],
    )

    model, cv_results = train_with_cv(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        model_grid=config["model_grid"],
        cv_folds=config["cv_folds"],
        random_state=config["random_state"],
        results_path=paths["results"],
    )

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_labels=config["class_labels"],
    )
    cv_summary = summarize_cv_metrics(cv_results)

    tables_dir = Path(paths["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cv_summary_path = tables_dir / "cv_summary.json"
    cv_summary_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")

    pd.DataFrame(metrics["classification_report"]).to_csv(
        tables_dir / "classification_report.csv"
    )

    plot_roc_curves(model, X_test, y_test, config["class_labels"], paths["figures"])
    plot_pr_curves(model, X_test, y_test, config["class_labels"], paths["figures"])
    plot_calibration_curve(
        model, X_test, y_test, config["class_labels"], paths["figures"]
    )

    X_sample = X_train.sample(
        n=min(len(X_train), 200), random_state=config["random_state"]
    )
    explain_with_shap(
        model=model,
        X_sample=model.named_steps["preprocess"].transform(X_sample),
        feature_names=model.named_steps["preprocess"].get_feature_names_out(),
        output_dir=paths["figures"],
    )


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
