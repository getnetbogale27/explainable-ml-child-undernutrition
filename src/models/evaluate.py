"""Model evaluation utilities."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: List[str],
) -> Dict[str, Any]:
    """Evaluate model on test data with per-class metrics."""
    LOGGER.info("Evaluating model")
    y_pred = model.predict(X_test)

    metrics = {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=class_labels, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def summarize_cv_metrics(cv_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Summarize CV mean and SD for macro-F1 and balanced accuracy."""
    summary = {}
    for metric in ["macro_f1", "balanced_accuracy"]:
        means = cv_results["cv_results"][f"mean_test_{metric}"]
        stds = cv_results["cv_results"][f"std_test_{metric}"]
        best_idx = int(np.argmax(means))
        summary[metric] = {
            "mean": float(means[best_idx]),
            "std": float(stds[best_idx]),
        }
    return summary