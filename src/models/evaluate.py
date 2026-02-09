# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Evaluation utilities: compute test metrics and summarize CV results."""
from __future__ import annotations
from typing import Any, Mapping, Protocol, TypedDict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)


class _Classifier(Protocol):
    """Classifier interface with predicted classes and probabilities."""

    def predict(self, X_test: Any) -> Any:
        """Return predicted class labels."""

    def predict_proba(self, X_test: Any) -> Any:
        """Return predicted class probabilities."""


class EvaluationResult(TypedDict):
    """Evaluation outputs for a fitted classifier."""

    accuracy: float
    balanced_accuracy: float
    roc_auc_ovr_macro: float | None
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]


def evaluate_model(
    model: _Classifier,
    X_test: Any,
    y_test: Any,
    class_labels: list[str],
) -> EvaluationResult:
    """Evaluate a fitted model on a hold-out test set."""
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_labels,
        output_dict=True,
    )
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            roc_auc = None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc_ovr_macro": roc_auc,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def summarize_cv_metrics(cv_results: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact summary of GridSearchCV results."""
    rank_scores = cv_results.get("rank_test_score")
    best_index = int(np.argmin(rank_scores)) if rank_scores is not None else None
    out = {
        "mean_test_score": float(
            np.max(cv_results.get("mean_test_score", np.array([np.nan])))
        ),
        "best_index": best_index,
        "params": cv_results.get("params", [])[:5],
    }
    return out
