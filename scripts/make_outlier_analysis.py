# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Fig. S3: Boxplots showing continuous variables before and after outlier treatment."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def treat_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Applies outlier treatment using the IQR method (clipping values outside 1.5 * IQR).
    """
    df_treated = df.copy()
    for col in columns:
        Q1 = df_treated[col].quantile(0.25)
        Q3 = df_treated[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Clipping values to the bounds
        df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
    return df_treated

def plot_outlier_comparison(df_orig: pd.DataFrame, df_treated: pd.DataFrame, columns: list[str], output_path: str | Path):
    """
    Creates a grid of boxplots matching the style of Fig. S3.
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 8), sharey=False)
    
    for i, col in enumerate(columns):
        # Top Row: Before Treatment
        sns.boxplot(data=df_orig, y=col, ax=axes[0, i], color="#2b7bba")
        axes[0, i].set_title(f"Before - {col}", fontsize=10)
        axes[0, i].set_ylabel("")
        axes[0, i].set_xlabel(col)
        axes[0, i].tick_params(axis='x', rotation=45)

        # Bottom Row: After Treatment
        sns.boxplot(data=df_treated, y=col, ax=axes[1, i], color="#2b7bba")
        axes[1, i].set_title(f"After - {col}", fontsize=10)
        axes[1, i].set_ylabel("")
        axes[1, i].set_xlabel(col)
        axes[1, i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# Example usage:
# cont_vars = ['ch_age_mon', 'BMI', 'care_age', 'mom_age', 'head_age', 'hh_size', 'Num_antenatal_visits', 'dadage']
# df_treated = treat_outliers_iqr(df_raw, cont_vars)
# plot_outlier_comparison(df_raw, df_treated, cont_vars, "results/figures/fig_s3_outliers.png")
