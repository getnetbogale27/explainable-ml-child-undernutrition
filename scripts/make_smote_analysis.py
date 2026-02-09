# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Fig. S1 (Pie Charts) and Table S3 (Gender-wise distribution across SMOTE)."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_smote_analysis(
    X_train_orig: pd.DataFrame, y_train_orig: pd.Series,
    X_train_res: pd.DataFrame, y_train_res: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    gender_col: str,
    output_dir: str | Path
):
    """
    Produces the pie charts for Fig. S1 and the complex cross-tabulation for Table S3.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    class_order = ['N', 'U', 'S', 'W', 'US', 'UW', 'USW']

    # --- 1. Generate Fig. S1: Pie Charts ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Before SMOTE
    y_train_orig.value_counts(normalize=True).reindex(class_order).plot.pie(
        ax=axes[0], autopct='%1.1f%%', startangle=90, title="Target Distribution Before SMOTE"
    )
    # After SMOTE
    y_train_res.value_counts(normalize=True).reindex(class_order).plot.pie(
        ax=axes[1], autopct='%1.1f%%', startangle=90, title="Target Distribution After SMOTE"
    )
    
    plt.tight_layout()
    plt.savefig(out_path / "fig_s1_smote_pie.png", dpi=300)
    plt.close()

    # --- 2. Generate Table S3: Gender-wise Distribution ---
    def get_counts(X, y, name):
        df = X.copy()
        df['target'] = y.values
        ct = pd.crosstab(df['target'], df[gender_col])
        ct = ct.reindex(class_order, fill_value=0)
        ct['Total'] = ct.sum(axis=1)
        # Rename columns to include the stage name
        ct.columns = [f"{name}_{c}" for c in ct.columns]
        return ct

    # Get counts for all three stages
    before_smote = get_counts(X_train_orig, y_train_orig, "Before_SMOTE")
    after_smote = get_counts(X_train_res, y_train_res, "After_SMOTE")
    test_set = get_counts(X_test, y_test, "Test_Set")

    # Combine into Table S3
    table_s3 = pd.concat([before_smote, after_smote, test_set], axis=1)
    table_s3.to_csv(out_path / "table_s3_distribution.csv")
    
    return table_s3
