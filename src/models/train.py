# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Model training with cross-validation and hyperparameter tuning."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

LOGGER = logging.getLogger(__name__)


def build_model_pipeline(preprocessor: Any, random_state: int) -> ImbPipeline:
    """Build a leakage-safe pipeline with preprocessing and SMOTE."""
    classifier = RandomForestClassifier(random_state=random_state)
    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", classifier),
        ]
    )
    return pipeline


def train_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    model_grid: Dict[str, Any],
    cv_folds: int,
    random_state: int,
    results_path: str | Path,
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    """Run cross-validation with hyperparameter tuning."""
    pipeline = build_model_pipeline(preprocessor, random_state)
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro"),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=model_grid,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    LOGGER.info("Starting grid search with %s folds", cv_folds)
    grid.fit(X_train, y_train)

    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    model_path = results_path / "best_model.joblib"
    joblib.dump(grid.best_estimator_, model_path)
    LOGGER.info("Saved best model to %s", model_path)

    cv_results = {
        "best_params": grid.best_params_,
        "best_score_macro_f1": grid.best_score_,
        "cv_results": grid.cv_results_,
    }
    return grid.best_estimator_, cv_results


def extract_cv_summary(cv_results: Dict[str, Any], metric: str) -> Dict[str, float]:
    """Compute mean and standard deviation for a CV metric."""
    scores = cv_results["cv_results"][f"mean_test_{metric}"]
    stds = cv_results["cv_results"][f"std_test_{metric}"]
    best_idx = int(np.argmax(scores))
    return {
        "mean": float(scores[best_idx]),
        "std": float(stds[best_idx]),
    }
