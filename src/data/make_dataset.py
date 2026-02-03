"""Data loading and train/test splitting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw data from CSV."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    LOGGER.info("Loading data from %s", data_path)
    return pd.read_csv(data_path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning steps (placeholder)."""
    LOGGER.info("Applying basic cleaning")
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    return cleaned


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test partitions before preprocessing."""
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
    """Full dataset preparation: load, clean, split."""
    df = load_raw_data(raw_path)
    df = basic_cleaning(df)
    return split_data(df, target_col, test_size, random_state)