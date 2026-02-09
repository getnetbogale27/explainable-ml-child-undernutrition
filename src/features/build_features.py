# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Feature engineering: imputation, encoding, scaling."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline with imputation, encoding, scaling."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the preprocessor on training data and transform train/test sets."""
    LOGGER.info("Fitting preprocessor on training data")
    X_train_proc = preprocessor.fit_transform(X_train)
    LOGGER.info("Transforming test data")
    X_test_proc = preprocessor.transform(X_test)
    return X_train_proc, X_test_proc
