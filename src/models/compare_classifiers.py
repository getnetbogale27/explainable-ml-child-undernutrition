"""Compare many classifiers using repeated stratified CV and return summary stats.

Functions:
- compare_classifiers: run repeated CV for list of classifiers and return DataFrame of mean, std, sem, and 95% CI
- default_classifiers: returns a dict of commonly used classifiers matching your figure
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# classifiers
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

def default_classifiers(random_state: int = 42) -> Dict[str, Any]:
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
) -> Tuple[float, float]:
    """Return (lower, upper) 95% CI for the array of values using t-distribution."""
    n = len(values)
    mean = float(np.mean(values))
    sem = float(stats.sem(values, nan_policy="omit"))
    if n > 1:
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    else:
        h = 0.0
    return mean - h, mean + h


def compare_classifiers(
    X,
    y,
    classifiers: Optional[Dict[str, Any]] = None,
    preprocessor: Optional[Any] = None,
    cv_folds: int = 5,
    repeats: int = 5,
    scoring: str = "accuracy",
    random_seed_start: int = 0,
) -> pd.DataFrame:
    """
    Run repeated stratified CV for each classifier and return summary DataFrame with:
    columns = [name, mean, std, sem, ci_lower, ci_upper, scores (list)]

    Parameters:
    - X, y : array-like or DataFrame/Series (must be aligned)
    - classifiers: dict(name -> estimator). If None, uses default_classifiers().
    - preprocessor: optional transformer (e.g., ColumnTransformer). If provided, it's included in a Pipeline before the estimator.
    - cv_folds: number of folds for StratifiedKFold
    - repeats: number of repeated CV runs with different RNG seeds (shuffle=True)
    - scoring: metric for cross_val_score (default 'accuracy')
    - random_seed_start: integer to seed first repeat; subsequent repeats use incremented seeds.
    """
    if classifiers is None:
        classifiers = default_classifiers(random_state=42)

    results = []

    for clf_name, clf in classifiers.items():
        all_scores: List[float] = []
        for r in range(repeats):
            seed = random_seed_start + r
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            # build pipeline: optional preprocessor -> scaler -> classifier
            steps = []
            if preprocessor is not None:
                steps.append(("preprocess", preprocessor))
            else:
                # if no preprocessor provided, at least scale numeric input to help some classifiers
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
        results.append({
            "name": clf_name,
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "n_scores": len(arr),
            "scores": arr.tolist()
        })

    df = pd.DataFrame(results).sort_values("mean", ascending=False).reset_index(drop=True)
    return df
