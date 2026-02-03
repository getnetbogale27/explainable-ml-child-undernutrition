"""Plot ROC, PR, and calibration curves."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize

LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_roc_curves(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: List[str],
    output_dir: str | Path,
) -> Path:
    """Generate and save multi-class ROC curves."""
    output_dir = _ensure_dir(output_dir)
    y_score = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(class_labels):
        RocCurveDisplay.from_predictions(
            y_bin[:, idx],
            y_score[:, idx],
            name=label,
            ax=ax,
        )
    ax.set_title("ROC Curves (One-vs-Rest)")
    fig.tight_layout()

    out_file = output_dir / "roc_curves.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved ROC curves to %s", out_file)
    return out_file


def plot_pr_curves(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: List[str],
    output_dir: str | Path,
) -> Path:
    """Generate and save multi-class precision-recall curves."""
    output_dir = _ensure_dir(output_dir)
    y_score = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(class_labels):
        PrecisionRecallDisplay.from_predictions(
            y_bin[:, idx],
            y_score[:, idx],
            name=label,
            ax=ax,
        )
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    fig.tight_layout()

    out_file = output_dir / "pr_curves.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved PR curves to %s", out_file)
    return out_file


def plot_calibration_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: List[str],
    output_dir: str | Path,
) -> Path:
    """Generate and save calibration curves per class."""
    output_dir = _ensure_dir(output_dir)
    y_score = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(class_labels):
        prob_true, prob_pred = calibration_curve(
            y_bin[:, idx],
            y_score[:, idx],
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curves")
    ax.legend()
    fig.tight_layout()

    out_file = output_dir / "calibration_curves.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved calibration curves to %s", out_file)
    return out_file