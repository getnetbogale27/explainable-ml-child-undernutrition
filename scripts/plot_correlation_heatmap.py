#!/usr/bin/env python3
# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""
scripts/plot_correlation_heatmap.py

Generate an annotated correlation heatmap for continuous variables.

Usage:
    python scripts/plot_correlation_heatmap.py --data data/processed/processed_data.csv \
        --vars ch_age_mon,BMI,care_age,mom_age,head_age,hh_size,Num_antenatal_visits,dadage \
        --method pearson --output results/figures/fig_s9_correlation.png --figsize 10 8

Options:
    --data      : path to CSV file
    --vars      : comma-separated list of variables to include (default: infer numeric columns)
    --method    : 'pearson' (default) or 'spearman'
    --cluster   : include this flag to reorder variables by hierarchical clustering
    --output    : output image path (png, pdf)
    --figsize   : two numbers: width height (default 10 8)
    --cmap      : matplotlib colormap (default 'RdBu_r')
    --annot     : annotate values (default True)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_corr(df, vars=None, method="pearson"):
    if vars:
        cols = vars
    else:
        # automatically pick numeric columns
        cols = df.select_dtypes(include=["number"]).columns.tolist()
    df_num = df[cols].copy()
    # Option: drop rows with all-NaN in selected columns
    df_num = df_num.dropna(how="all", subset=cols)
    corr = df_num.corr(method=method)
    return corr, cols


def plot_heatmap(corr, output, figsize=(10, 8), cmap="RdBu_r", annot=True, fmt=".2f", vmax=None, vmin=None):
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    plt.figure(figsize=figsize)
    sns.set(style="white")
    # center at 0 for diverging map
    if vmax is None:
        vmax = 1.0
    if vmin is None:
        vmin = -1.0

    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=annot,
        fmt=fmt,
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Correlation"}
    )

    # Improve tick label appearance
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("S9: Correlation Heatmap Plot", fontsize=14, pad=12)
    plt.tight_layout()
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=300)
    plt.close()


def plot_clustermap(corr, output, figsize=(10, 8), cmap="RdBu_r", annot=True, fmt=".2f", vmax=None, vmin=None):
    # Seaborn clustermap reorders rows/cols by hierarchical clustering
    if vmax is None:
        vmax = 1.0
    if vmin is None:
        vmin = -1.0

    cg = sns.clustermap(
        corr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        figsize=figsize,
        cbar_kws={"label": "Correlation"},
        xticklabels=True,
        yticklabels=True
    )
    plt.suptitle("S9: Correlation Heatmap (Clustered)", y=1.02)
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cg.savefig(outp, dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV file with processed data (must contain requested variables)")
    p.add_argument("--vars", default=None, help="Comma-separated variable names to include (default: infer numeric columns)")
    p.add_argument("--method", default="pearson", choices=["pearson", "spearman"], help="Correlation method")
    p.add_argument("--cluster", action="store_true", help="Use hierarchical clustering to reorder variables (clustermap)")
    p.add_argument("--output", required=True, help="Output image path (png/pdf)")
    p.add_argument("--figsize", nargs=2, type=float, default=[10, 8], help="Figure size: width height")
    p.add_argument("--cmap", default="RdBu_r", help="Colormap (default 'RdBu_r')")
    p.add_argument("--annot", type=lambda s: s.lower() in ("true", "1", "yes"), default=True, help="Annotate cell values")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    vars_list = args.vars.split(",") if args.vars else None

    corr, used_cols = compute_corr(df, vars=vars_list, method=args.method)
    # Reorder rows/cols if user passed vars list in different order: keep that order
    if vars_list:
        corr = corr.reindex(index=used_cols, columns=used_cols)

    if args.cluster:
        plot_clustermap(corr, args.output, figsize=tuple(args.figsize), cmap=args.cmap, annot=args.annot)
    else:
        plot_heatmap(corr, args.output, figsize=tuple(args.figsize), cmap=args.cmap, annot=args.annot)


if __name__ == "__main__":
    main()
