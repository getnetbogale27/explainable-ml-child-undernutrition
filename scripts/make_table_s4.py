# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Table S4: Descriptive statistics of all categorical variables by nutritional status."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.data.make_dataset import load_raw_data, encode_outcome, basic_cleaning

def generate_table_s4(df: pd.DataFrame, categorical_features: list[str], output_path: str | Path):
    """
    Calculates n (%) for each category across nutritional statuses.
    """
    class_order = ['N', 'U', 'S', 'W', 'US', 'UW', 'USW']
    all_rows = []

    for var in categorical_features:
        # Create crosstab
        ct = pd.crosstab(df[var], df['concurrent_conditions'])
        ct = ct.reindex(columns=class_order, fill_value=0)
        
        # Calculate percentages row-wise (within each category)
        # Note: Your table shows percentages relative to the category total
        row_totals = ct.sum(axis=1)
        
        formatted_df = pd.DataFrame(index=ct.index)
        for col in class_order:
            # Format as "n (percentage%)"
            formatted_df[col] = ct[col].astype(str) + " (" + \
                                (ct[col] / row_totals * 100).round(2).astype(str) + ")"
        
        # Reset index to make 'Variable' and 'Category' columns
        formatted_df = formatted_df.reset_index()
        formatted_df.insert(0, 'Variable', var)
        formatted_df.rename(columns={var: 'Category'}, inplace=True)
        
        all_rows.append(formatted_df)

    # Combine all variables into one large table
    final_table_s4 = pd.concat(all_rows, ignore_index=True)
    
    # Save to CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_table_s4.to_csv(out, index=False)
    
    return final_table_s4

# Example usage:
# cat_features = ['region', 'residence', 'ch_sex', 'ch_longterm_health_problem', ...]
# table_s4 = generate_table_s4(df, cat_features, "results/tables/table_s4_descriptive.csv")
