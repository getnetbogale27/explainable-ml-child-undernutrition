# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""SHAP explainability: Beeswarm and Summary plots for multiclass models."""
from __future__ import annotations
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

def explain_with_shap(model, X_sample, feature_names, output_dir: str | Path, class_index: int = 0):
    """
    Generates SHAP Beeswarm plots matching the manuscript style.
    
    Parameters:
    - model: The trained model (or pipeline).
    - X_sample: Preprocessed numeric matrix (numpy array or DataFrame).
    - feature_names: List of feature names corresponding to X_sample columns.
    - output_dir: Directory to save the plots.
    - class_index: The index of the class to explain (e.g., 0 for 'Normal').
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If model is a pipeline, extract the final estimator
    if hasattr(model, "named_steps"):
        final_model = model.named_steps["model"]
    else:
        final_model = model

    LOGGER.info("Calculating SHAP values for class index %d", class_index)
    
    # Use TreeExplainer for Random Forest / Gradient Boosting
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_sample)

    # In multiclass, shap_values is a list of arrays (one per class)
    # We select the specific class requested by the user
    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_index]
    else:
        class_shap_values = shap_values

    # 1. Generate Beeswarm Plot
    plt.figure(figsize=(12, 10))
    # We create a SHAP Explanation object to use the modern beeswarm API
    explanation = shap.Explanation(
        values=class_shap_values,
        data=X_sample,
        feature_names=feature_names
    )
    
    shap.plots.beeswarm(explanation, max_display=30, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - Class {class_index}")
    plt.tight_layout()
    
    beeswarm_path = output_path / f"shap_beeswarm_class_{class_index}.png"
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    
    # 2. Generate Global Importance Bar Plot (Optional, matching your other image)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(class_shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"Feature Importance (SHAP) - Class {class_index}")
    plt.tight_layout()
    
    bar_path = output_path / f"shap_importance_class_{class_index}.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()

    LOGGER.info("SHAP plots saved to %s", output_path)
