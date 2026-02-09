# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate bar charts for categorical variables and box plots for continuous variables."""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_categorical_vars(df: pd.DataFrame, cat_vars: list[str], output_dir: str | Path):
    """Plot bar charts for all categorical variables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for var in cat_vars:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=var, order=df[var].value_counts().index)
        plt.title(f"Bar Chart for {var}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"bar_chart_{var}.png", dpi=300)
        plt.close()

def plot_continuous_vars(df: pd.DataFrame, cont_vars: list[str], output_dir: str | Path):
    """Plot box plots for all continuous variables grouped by nutritional status."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for var in cont_vars:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='concurrent_conditions', y=var, order=['N', 'U', 'S', 'W', 'US', 'UW', 'USW'])
        plt.title(f"Box Plot for {var} by Nutritional Status")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"box_plot_{var}.png", dpi=300)
        plt.close()
