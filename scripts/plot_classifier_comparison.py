"""Script to run classifier comparison and produce bar plot with 95% CI error bars.

Usage:
    python scripts/plot_classifier_comparison.py --data /path/to/processed_X.csv --target /path/to/target.csv --out figures/classifier_comparison.png

Or import compare_classifiers() and call from your pipeline after preprocessing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.models.compare_classifiers import compare_classifiers, default_classifiers

def plot_results(df_summary: pd.DataFrame, out_path: str | Path, figsize=(10,6)):
    """Barplot of mean accuracy with 95% CI error bars."""
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)
    order = df_summary["name"].tolist()
    means = df_summary["mean"].values
    err_low = df_summary["mean"].values - df_summary["ci_lower"].values
    err_high = df_summary["ci_upper"].values - df_summary["mean"].values
    yerr = np.vstack([err_low, err_high])

    ax = plt.gca()
    bars = ax.barh(order, means, xerr=yerr, align="center", color="skyblue", edgecolor="k")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean Accuracy (repeated CV)")
    ax.set_title("Classifier comparison (mean ± 95% CI)")
    # Add numeric labels
    for bar, m in zip(bars, means):
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{m:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main(args):
    X = pd.read_csv(args.data, index_col=0)
    y = pd.read_csv(args.target, squeeze=True, index_col=0)
    # if target loaded as DataFrame, convert to Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:,0]

    classifiers = default_classifiers(random_state=args.random_state)
    df_summary = compare_classifiers(
        X=X,
        y=y,
        classifiers=classifiers,
        preprocessor=None,  # set if you want to include your ColumnTransformer
        cv_folds=args.cv,
        repeats=args.repeats,
        scoring="accuracy",
        random_seed_start=args.seed
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_results(df_summary, args.out)
    print("Saved plot to", args.out)
    print(df_summary[["name", "mean", "ci_lower", "ci_upper"]])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV file with feature matrix X (preprocessed numeric columns preferred)")
    p.add_argument("--target", required=True, help="CSV/TSV file with target series y (index must align)")
    p.add_argument("--out", required=True, help="Output image path (png)")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--repeats", type=int, default=5, help="Number of repeated runs (different RNG seeds)")
    p.add_argument("--seed", type=int, default=0, help="Seed start for repeats")
    p.add_argument("--random_state", type=int, default=42, help="Classifier random state")
    args = p.parse_args()
    main(args)