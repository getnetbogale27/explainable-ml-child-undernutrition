# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""RandomForest Gini-importance feature selection with a median cutoff."""
from __future__ import annotations
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LOGGER = logging.getLogger(__name__)


class FeatureSelectionResult(TypedDict):
    """Outputs from the RandomForest feature-selection step."""

    selected_features: list[str]
    importances_df: pd.DataFrame
    median_importance: float


def select_features_rf_median(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_estimators: int = 500,
    min_samples_leaf: int = 1,
) -> FeatureSelectionResult:
    """Fit a RandomForest model and keep features above the median importance."""
    if X.shape[0] == 0:
        raise ValueError("Empty training data provided to feature selector.")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=min_samples_leaf,
    )
    rf.fit(X, y)
    importances = rf.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    median_imp = float(np.median(importances))
    selected = imp_df.loc[imp_df["importance"] > median_imp, "feature"].tolist()
    LOGGER.info(
        "Selected %d features (median importance %.5f) from %d",
        len(selected),
        median_imp,
        X.shape[1],
    )
    return {"selected_features": selected, "importances_df": imp_df, "median_importance": median_imp}
