# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Plotting utilities for model evaluation, including gender-stratified confusion matrices."""
from __future__ import annotations
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

LOGGER = logging.getLogger(__name__)

def plot_gender_stratified_confusion_matrices(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    gender_col: str,
    class_labels: list[str],
    output_path: str | Path
):
    """
    Generates side-by-side confusion matrices for Male and Female subgroups.
    Matches the style of Fig. 7 in the manuscript.
    """
    # The gender column must exist in X_test, using the same representation as gender_col.
    # Use the encoded column name if gender was one-hot encoded.
    
    # Create a combined dataframe for easy subsetting
    test_df = X_test.copy()
    test_df['true_label'] = y_test.values
    test_df['pred_label'] = model.predict(X_test)

    # Define subgroups based on gender_col encoding (e.g., 0 = Male, 1 = Female).
    genders = test_df[gender_col].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    for i, gender_val in enumerate(sorted(genders)):
        subset = test_df[test_df[gender_col] == gender_val]
        cm = confusion_matrix(subset['true_label'], subset['pred_label'], labels=class_labels)
        
        # Plotting
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=axes[i],
            xticklabels=class_labels, 
            yticklabels=class_labels,
            cbar=(i == 1), # Only show colorbar on the second plot
            linewidths=.5,
            linecolor='gray'
        )
        
        title = "A) Male" if i == 0 else "B) Female"
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('Predicted Nutritional Status', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('True Nutritional Status', fontsize=12)
        else:
            axes[i].set_ylabel('')

    plt.tight_layout()
    
    # Save the figure
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    LOGGER.info("Gender-stratified confusion matrix saved to %s", save_path)
