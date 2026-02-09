# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Evaluation utilities: compute test metrics and summarize CV metrics."""
from __future__ import annotations
import json
import numpy as np
from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model: Any, X_test, y_test, class_labels: list[str]):
    """Evaluate final model on hold-out test set and return metrics dict."""
    y_pred = model.predict(X_test)
    # For OvR classifiers sklearn returns a list of estimators; if pipeline wrapped model then predict_proba is available
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        # some estimators do not support predict_proba
        y_proba = None

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc_ovr_macro": roc_auc,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


def summarize_cv_metrics(cv_results) -> Dict:
    """Compact summary of cv_results from GridSearchCV (mean test score, best index, etc.)."""
    rank_scores = cv_results.get("rank_test_score")
    best_index = int(np.argmin(rank_scores)) if rank_scores is not None else None
    out = {
        "mean_test_score": float(np.max(cv_results.get("mean_test_score", np.array([np.nan])))),
        "best_index": best_index,
        "params": cv_results.get("params", [])[:5]
    }
    return out
