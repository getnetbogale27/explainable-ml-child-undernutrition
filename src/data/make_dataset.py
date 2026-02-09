"""Data loading, outcome encoding, and train/test splitting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def encode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the concurrent undernutrition outcome based on WHO Z-score thresholds.
    Threshold: Z-score < -2 indicates undernutrition.
    
    Classes:
    - N: Normal (All > -2)
    - U: Underweight only (WAZ < -2)
    - S: Stunting only (HAZ < -2)
    - W: Wasting only (WHZ < -2)
    - US: Underweight & Stunting (WAZ & HAZ < -2)
    - UW: Underweight & Wasting (WAZ & WHZ < -2)
    - USW: Underweight, Stunting, & Wasting (All < -2)
    """
    LOGGER.info("Encoding outcome variable 'concurrent_conditions'")
    
    # Define indicators
    underweight = df['waz'] < -2
    stunting = df['haz'] < -2
    wasting = df['whz'] < -2

    # Initialize outcome column
    df['concurrent_conditions'] = 'N'
    
    # Apply logic for concurrent conditions
    df.loc[underweight & ~stunting & ~wasting, 'concurrent_conditions'] = 'U'
    df.loc[~underweight & stunting & ~wasting, 'concurrent_conditions'] = 'S'
    df.loc[~underweight & ~stunting & wasting, 'concurrent_conditions'] = 'W'
    df.loc[underweight & stunting & ~wasting, 'concurrent_conditions'] = 'US'
    df.loc[underweight & ~stunting & wasting, 'concurrent_conditions'] = 'UW'
    # Note: SW (Stunting & Wasting) is skipped as per manuscript (not observed)
    df.loc[underweight & stunting & wasting, 'concurrent_conditions'] = 'USW'
    
    return df


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw data from CSV."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    LOGGER.info("Loading data from %s", data_path)
    return pd.read_csv(data_path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning steps and strip column names."""
    LOGGER.info("Applying basic cleaning")
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    # Drop rows with missing Z-scores needed for outcome encoding
    cleaned = cleaned.dropna(subset=['waz', 'haz', 'whz'])
    return cleaned


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test partitions."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    LOGGER.info("Split data: train=%s, test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def make_dataset(
    raw_path: str | Path,
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Full dataset preparation: load, clean, encode outcome, split."""
    df = load_raw_data(raw_path)
    df = basic_cleaning(df)
    df = encode_outcome(df)
    return split_data(df, target_col, test_size, random_state)