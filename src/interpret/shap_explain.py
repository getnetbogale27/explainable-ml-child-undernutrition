# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""SHAP explainability utilities (associational explanations)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import shap

LOGGER = logging.getLogger(__name__)


class _PipelineWithClassifier(Protocol):
    """Pipeline interface exposing a fitted classifier step."""

    named_steps: Mapping[str, object]


def explain_with_shap(
    model: _PipelineWithClassifier,
    X_sample: ArrayLike,
    feature_names: Sequence[str],
    output_dir: str | Path,
) -> Path:
    """Generate a SHAP summary plot for model associations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Computing SHAP values")
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
    )
    out_file = output_dir / "shap_summary.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved SHAP summary to %s", out_file)
    return out_file
