"""Generate Supplementary Tables S1 and S2: Nutrition Status by Gender and Region."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.data.make_dataset import load_raw_data, encode_outcome, basic_cleaning

def generate_supplementary_tables(df: pd.DataFrame, output_dir: str | Path):
    """
    Creates cross-tabulations for Gender and Region against the encoded outcome.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the order of classes as shown in your tables
    class_order = ['N', 'U', 'S', 'W', 'US', 'UW', 'USW']
    
    # 1. Table S1: Nutrition Status by Gender
    # Assuming 'sex' or 'child_sex' is the column name
    gender_col = 'sex' # Update this to match your actual column name
    table_s1 = pd.crosstab(df[gender_col], df['concurrent_conditions'])
    
    # Reorder columns and add Total
    table_s1 = table_s1.reindex(columns=class_order, fill_value=0)
    table_s1['Total'] = table_s1.sum(axis=1)
    
    # Add Total row
    total_row_s1 = table_s1.sum().to_frame().T
    total_row_s1.index = ['Total']
    table_s1 = pd.concat([table_s1, total_row_s1])

    # 2. Table S2: Nutrition Status by Region
    region_col = 'region' # Update this to match your actual column name
    table_s2 = pd.crosstab(df[region_col], df['concurrent_conditions'])
    
    # Reorder columns and add Total
    table_s2 = table_s2.reindex(columns=class_order, fill_value=0)
    table_s2['Total'] = table_s2.sum(axis=1)
    
    # Add Total row
    total_row_s2 = table_s2.sum().to_frame().T
    total_row_s2.index = ['Total']
    table_s2 = pd.concat([table_s2, total_row_s2])

    # Save to CSV
    table_s1.to_csv(output_path / "table_s1_gender.csv")
    table_s2.to_csv(output_path / "table_s2_region.csv")
    
    return table_s1, table_s2

# Example usage:
# df = load_raw_data("data/raw/young_lives_ethiopia.csv")
# df = basic_cleaning(df)
# df = encode_outcome(df)
# s1, s2 = generate_supplementary_tables(df, "results/tables/")