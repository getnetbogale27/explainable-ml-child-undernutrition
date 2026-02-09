"""Preprocessing pipeline: cleaning, encoding, scaling, split, and SMOTE.

Usage:
    from src.data.preprocess import preprocess_and_oversample

    results = preprocess_and_oversample(
        df=df_raw,
        numeric_features=config["dataset"]["numeric_features"],
        categorical_features=config["dataset"]["categorical_features"],
        ordinal_mappings=config.get("ordinal_mappings", {}),
        drop_features=config.get("drop_features", []),
        target_col="concurrent_conditions",
        test_size=0.10,
        random_state=42,
        smote_kwargs={"random_state": 42, "k_neighbors": 5}
    )

Returns a dict with:
    X_train_res (pd.DataFrame), y_train_res (pd.Series),
    X_test (pd.DataFrame), y_test (pd.Series),
    preprocessor (fitted ColumnTransformer),
    feature_names (list[str])  # column names after transformation
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

LOGGER = logging.getLogger(__name__)


def _validate_features(df: pd.DataFrame, features: List[str], name: str) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {name} columns in dataframe: {missing}")


def _strip_feature_names(features: List[str]) -> List[str]:
    return [c.strip() for c in features]


def _validate_ordinal_mappings(
    df: pd.DataFrame, ordinal_mappings: Dict[str, List[str]]
) -> List[str]:
    ordinal_features = list(ordinal_mappings.keys())
    _validate_features(df, ordinal_features, "ordinal")
    empty_categories = [col for col, categories in ordinal_mappings.items() if not categories]
    if empty_categories:
        raise ValueError(
            "Ordinal mappings must define at least one category for: "
            f"{empty_categories}"
        )
    return ordinal_features


def build_minmax_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    ordinal_features: Optional[List[str]] = None,
    ordinal_mappings: Optional[Dict[str, List[str]]] = None,
) -> ColumnTransformer:
    """Build a preprocessor using mean/mode imputation and MinMax scaling."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),  # scales to [0,1] as described
        ]
    )

    # Ordinal encoder (if any)
    transformers = [("num", numeric_pipeline, numeric_features)]

    if ordinal_features:
        # For OrdinalEncoder we need to provide categories for each column
        cats = [ordinal_mappings.get(col, []) for col in ordinal_features]
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(categories=cats, dtype=float)),
            ]
        )
        transformers.append(("ord", ordinal_pipeline, ordinal_features))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def reconstruct_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return feature names after ColumnTransformer (sklearn >=1.0)."""
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

    names: List[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                names.extend(list(trans.get_feature_names_out(cols)))
            except TypeError:
                names.extend(list(trans.get_feature_names_out()))
        else:
            names.extend(list(cols))
    return names


def preprocess_and_oversample(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    ordinal_mappings: Optional[Dict[str, List[str]]] = None,
    drop_features: Optional[List[str]] = None,
    target_col: str = "concurrent_conditions",
    test_size: float = 0.10,
    random_state: int = 42,
    smote_kwargs: Optional[Dict] = None,
) -> Dict[str, object]:
    """Complete preprocessing pipeline and SMOTE oversampling applied on training set only."""
    if drop_features is None:
        drop_features = []
    if ordinal_mappings is None:
        ordinal_mappings = {}
    if smote_kwargs is None:
        smote_kwargs = {"random_state": random_state}

    df = df.copy()

    # Basic cleaning: strip column names
    df.columns = [c.strip() for c in df.columns]
    numeric_features = _strip_feature_names(numeric_features)
    categorical_features = _strip_feature_names(categorical_features)
    drop_features = _strip_feature_names(drop_features)

    # Drop redundant cols
    if drop_features:
        LOGGER.info("Dropping %d redundant features", len(drop_features))
        df = df.drop(columns=[c for c in drop_features if c in df.columns], errors="ignore")

    # Ensure target present
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Validate feature lists
    _validate_features(df, numeric_features, "numeric")
    _validate_features(df, categorical_features, "categorical")
    ordinal_features = (
        _validate_ordinal_mappings(df, ordinal_mappings) if ordinal_mappings else []
    )
    overlap = (
        set(numeric_features) & set(categorical_features)
        | set(numeric_features) & set(ordinal_features)
        | set(categorical_features) & set(ordinal_features)
    )
    if overlap:
        raise ValueError(
            "Features must be uniquely assigned to a single type. "
            f"Overlaps found: {sorted(overlap)}"
        )

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Train/test split (90:10)
    from sklearn.model_selection import train_test_split

    X = df[numeric_features + categorical_features + ordinal_features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    LOGGER.info("Data split: train=%d, test=%d", len(X_train), len(X_test))

    # Build preprocessor (MinMax, mean/mode imputation)
    preprocessor = build_minmax_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        ordinal_mappings=ordinal_mappings,
    )

    # Fit preprocessor on training data only
    LOGGER.info("Fitting preprocessor on training data")
    X_train_trans = preprocessor.fit_transform(X_train)
    LOGGER.info("Transforming test data")
    X_test_trans = preprocessor.transform(X_test)

    # Retrieve feature names after transform
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # fallback attempt if sklearn older/newer. Still try to build names
        feature_names = reconstruct_feature_names(preprocessor)

    # Convert to DataFrame for easier downstream processing
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Apply SMOTE on training partition only
    if smote_kwargs is not None:
        LOGGER.info("Applying SMOTE to training set with args: %s", smote_kwargs)
        smote = SMOTE(**smote_kwargs)
        X_train_res, y_train_res = smote.fit_resample(X_train_df, y_train)
        # Ensure DataFrame columns preserved
        X_train_res = pd.DataFrame(X_train_res, columns=feature_names)
        y_train_res = pd.Series(y_train_res, name=y_train.name)
        LOGGER.info(
            "SMOTE applied: train size before=%d after=%d", len(X_train_df), len(X_train_res)
        )
    else:
        X_train_res, y_train_res = X_train_df, y_train

    return {
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "X_test": X_test_df,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
