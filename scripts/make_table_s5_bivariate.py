#!/usr/bin/env python3
# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""
Generate Table S5: Bivariate chi-square analysis of 'Healthy (N)' vs 'Not Healthy' across categorical variables.

Usage:
  python scripts/make_table_s5_bivariate.py --data data/processed/processed_data.csv \
      --vars region,residence,ch_sex,... \
      --target concurrent_conditions \
      --healthy N \
      --out results/tables/table_s5_bivariate.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def significance_stars(p):
    if p < 0.01:
        return "<0.001***" if p < 0.001 else f"{p:.3f}***"
    if p < 0.05:
        return f"{p:.3f}**"
    if p < 0.1:
        return f"{p:.3f}*"
    return f"{p:.3f}"

def make_bivariate_table(df, vars_list, target_col="concurrent_conditions", healthy_label="N"):
    rows = []
    for var in vars_list:
        if var not in df.columns:
            print(f"Warning: {var} not in dataframe; skipping.")
            continue
        ct = pd.crosstab(df[var], df[target_col].apply(lambda x: healthy_label if x == healthy_label else f"not_{healthy_label}"))
        # Ensure both columns exist
        for col in [healthy_label, f"not_{healthy_label}"]:
            if col not in ct.columns:
                ct[col] = 0
        # Chi-square on contingency (rows = categories, columns = binary)
        try:
            chi2, p, dof, ex = chi2_contingency(ct.values)
        except Exception as e:
            chi2, p = np.nan, np.nan

        # For each category, compute count and percent within that category
        for idx in ct.index:
            healthy_count = int(ct.loc[idx, healthy_label])
            not_count = int(ct.loc[idx, f"not_{healthy_label}"])
            total = healthy_count + not_count
            pct_h = 100.0 * healthy_count / total if total > 0 else 0.0
            pct_not = 100.0 * not_count / total if total > 0 else 0.0

            rows.append({
                "Variable": var,
                "Category": idx,
                "Healthy_n_pct": f"{healthy_count} ({pct_h:.2f}%)",
                "Not_Healthy_n_pct": f"{not_count} ({pct_not:.2f}%)",
                "Chi2_statistic": "" if idx != ct.index[0] else (f"{chi2:.3f}" if not np.isnan(chi2) else ""),
                "p_value": "" if idx != ct.index[0] else significance_stars(p) if not np.isnan(p) else ""
            })
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV file with data (must include target and variables)")
    p.add_argument("--vars", required=True, help="Comma-separated list of categorical variables to test")
    p.add_argument("--target", default="concurrent_conditions", help="Target column name")
    p.add_argument("--healthy", default="N", help="Label in target considered 'Healthy' (default 'N')")
    p.add_argument("--out", required=True, help="Output CSV path for Table S5")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    vars_list = [v.strip() for v in args.vars.split(",") if v.strip()]
    table = make_bivariate_table(df, vars_list, target_col=args.target, healthy_label=args.healthy)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(outp, index=False)
    print("Saved Table S5 to", outp)

if __name__ == "__main__":
    main()
