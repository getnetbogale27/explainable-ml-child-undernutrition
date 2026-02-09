"""Metric utilities for model evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import math
import numpy as np


def clamp_ci(lower: float, upper: float) -> Tuple[float, float]:
    """Clamp confidence interval bounds to the [0, 1] range."""
    return max(0.0, lower), min(1.0, upper)


def normal_approx_ci(value: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute a normal-approximation confidence interval for a proportion."""
    if n <= 0 or np.isnan(value):
        return np.nan, np.nan
    se = math.sqrt(value * (1 - value) / n)
    lower = value - z * se
    upper = value + z * se
    return clamp_ci(lower, upper)


def macro_specificity(
    y_true: Sequence, y_pred: Sequence, labels: Iterable
) -> float:
    """Compute macro-averaged specificity across classes."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels_list = list(labels) if labels is not None else list(np.unique(y_true_arr))
    if not labels_list:
        return float("nan")
    specifics = []
    for label in labels_list:
        tn = np.sum((y_true_arr != label) & (y_pred_arr != label))
        fp = np.sum((y_true_arr != label) & (y_pred_arr == label))
        denom = tn + fp
        specifics.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specifics))
