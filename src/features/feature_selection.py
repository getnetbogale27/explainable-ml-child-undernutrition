"""RandomForest Gini importance feature selection with median cutoff."""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LOGGER = logging.getLogger(__name__)


def select_features_rf_median(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_estimators: int = 500,
    min_samples_leaf: int = 1,
) -> Dict[str, object]:
    """
    Fit RandomForest on (X,y) and return features with importance > median importance.
    Returns selected_features, importances_df, median_importance
    """
    if X.shape[0] == 0:
        raise ValueError("Empty training data provided to feature selector.")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, min_samples_leaf=min_samples_leaf)
    rf.fit(X, y)
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False).reset_index(drop=True)
    median_imp = float(np.median(importances))
    selected = imp_df.loc[imp_df["importance"] > median_imp, "feature"].tolist()
    LOGGER.info("Selected %d features (median importance %.5f) from %d", len(selected), median_imp, X.shape[1])
    return {"selected_features": selected, "importances_df": imp_df, "median_importance": median_imp}