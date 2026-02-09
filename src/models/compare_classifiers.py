# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Compare classifiers with repeated stratified CV and summarize performance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Mapping

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def default_classifiers(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Return a dict of classifier instances to evaluate (names -> estimator)."""
    return {
        "Support Vector Machine": SVC(
            probability=True, kernel="rbf", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
        "Logistic Regression (L2)": LogisticRegression(
            penalty="l2", solver="saga", max_iter=5000, random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression (L1)": LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, random_state=random_state
        ),
        "Elastic-Net Logistic": LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            max_iter=5000,
            random_state=random_state,
        ),
    }


def _compute_confidence_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a t-distribution confidence interval for the provided values."""
    n = len(values)
    mean = float(np.mean(values))
    sem = float(stats.sem(values, nan_policy="omit"))
    if n > 1:
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    else:
        h = 0.0
    return mean - h, mean + h


def compare_classifiers(
    X: ArrayLike,
    y: ArrayLike,
    classifiers: Mapping[str, BaseEstimator] | None = None,
    preprocessor: TransformerMixin | None = None,
    cv_folds: int = 5,
    repeats: int = 5,
    scoring: str = "accuracy",
    random_seed_start: int = 0,
) -> pd.DataFrame:
    """Run repeated stratified CV and return a summary DataFrame."""
    if classifiers is None:
        classifiers = default_classifiers(random_state=42)

    results = []

    for clf_name, clf in classifiers.items():
        all_scores: list[float] = []
        for r in range(repeats):
            seed = random_seed_start + r
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            steps = []
            if preprocessor is not None:
                steps.append(("preprocess", preprocessor))
            else:
                steps.append(("scaler", StandardScaler()))
            steps.append(("clf", clf))
            pipe = Pipeline(steps=steps)

            scores = cross_val_score(
                pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
            )
            all_scores.extend(scores.tolist())

        arr = np.array(all_scores)
        mean = float(np.mean(arr))
        if len(arr) > 1:
            std = float(np.std(arr, ddof=1))
            sem = float(stats.sem(arr, nan_policy="omit"))
        else:
            std = 0.0
            sem = 0.0
        ci_low, ci_high = _compute_confidence_interval(arr)
        results.append(
            {
                "name": clf_name,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_scores": len(arr),
                "scores": arr.tolist(),
            }
        )

    df = pd.DataFrame(results).sort_values("mean", ascending=False).reset_index(drop=True)
    return df
