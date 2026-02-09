#!/usr/bin/env python3
# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""
Generate Table S7: Performance metrics of ML models across train/test splits.

Usage:
  python scripts/make_table_s7_performance_splits.py --data data/raw/dataset.csv \
      --target concurrent_conditions --splits 0.6,0.7,0.75,0.8,0.85,0.9 \
      --out results/tables/table_s7_performance_splits.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.metrics import macro_specificity, normal_approx_ci

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with raw data")
    parser.add_argument("--target", default="concurrent_conditions", help="Target column name")
    parser.add_argument(
        "--splits",
        default="0.6,0.7,0.75,0.8,0.85,0.9",
        help="Comma-separated train ratios",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--random_state", type=int, default=42, help="Random state seed")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Use bootstrap confidence intervals for accuracy/AUC",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000, help="Number of bootstrap resamples"
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for estimators")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
    )


def _validate_and_prepare_data(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate dataset, drop non-numeric columns, and return features/target."""
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset. "
            "Please confirm the target name."
        )
    X = df.drop(columns=[target])
    y = df[target]
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        LOGGER.warning(
            "Dropping non-numeric columns: %s", ", ".join(non_numeric_cols)
        )
        X = X.drop(columns=non_numeric_cols)
    if X.empty:
        raise ValueError(
            "No numeric features remain after dropping non-numeric columns. "
            "Please encode categorical features before running this script."
        )
    return X, y


def _build_classifiers(random_state: int, n_jobs: int) -> Dict[str, object]:
    """Construct classifier dictionary with consistent random_state values."""
    return {
        "Support Vector Machine": SVC(
            probability=True, kernel="rbf", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="saga", random_state=random_state, n_jobs=n_jobs
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "LASSO": LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Ridge": LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=2000,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Elastic Net": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=2000,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
    }


def _scores_to_proba(scores: np.ndarray) -> np.ndarray:
    """Convert decision_function scores to probability-like outputs."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        probs_pos = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs_pos, probs_pos])
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _get_prediction_scores(clf: object, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return class probabilities or probability-like scores for AUC."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        return _scores_to_proba(scores)
    LOGGER.warning(
        "Classifier %s has no predict_proba/decision_function; AUC set to NaN.",
        clf.__class__.__name__,
    )
    return None


def _compute_auc(
    y_true: pd.Series,
    y_score: Optional[np.ndarray],
    labels: Iterable,
    log_failure: bool = True,
) -> float:
    """Compute AUC for binary or multiclass probabilities."""
    if y_score is None:
        return float("nan")
    try:
        y_score_arr = np.asarray(y_score)
        if y_score_arr.ndim == 2 and y_score_arr.shape[1] == 2:
            return roc_auc_score(y_true, y_score_arr[:, 1])
        return roc_auc_score(
            y_true, y_score_arr, multi_class="ovr", average="macro", labels=list(labels)
        )
    except Exception as exc:
        if log_failure:
            LOGGER.warning("AUC computation failed: %s", exc)
        return float("nan")


def _bootstrap_ci(
    metric_fn, n_samples: int, n_bootstrap: int, rng: np.random.Generator
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    stats: List[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        value = metric_fn(indices)
        if not np.isnan(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    lower = float(np.percentile(stats, 2.5))
    upper = float(np.percentile(stats, 97.5))
    return max(0.0, lower), min(1.0, upper)


def _format_ci(value: float, lower: float, upper: float) -> str:
    """Format point estimate and confidence interval."""
    if np.isnan(value) or np.isnan(lower) or np.isnan(upper):
        return "NA"
    return f"{value:.3f} ({lower:.3f}, {upper:.3f})"


def generate_table_s7(
    df: pd.DataFrame,
    target: str,
    train_ratios: Sequence[float],
    out_path: Path,
    random_state: int = 42,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Generate performance metrics across train/test splits and save to CSV."""
    classifiers = _build_classifiers(random_state=random_state, n_jobs=n_jobs)
    rng = np.random.default_rng(random_state)

    results: List[Dict[str, object]] = []

    for train_ratio in train_ratios:
        LOGGER.info("Evaluating train ratio %.2f", train_ratio)
        train_df, test_df = train_test_split(
            df,
            train_size=train_ratio,
            stratify=df[target],
            random_state=random_state,
        )
        X_train, y_train = _validate_and_prepare_data(train_df, target)
        X_test, y_test = _validate_and_prepare_data(test_df, target)
        labels = list(pd.unique(y_train))

        for clf_name, clf in classifiers.items():
            LOGGER.info("Fitting %s", clf_name)
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            except Exception as exc:
                LOGGER.exception("Failed to fit/predict with %s: %s", clf_name, exc)
                continue

            y_score = None
            try:
                y_score = _get_prediction_scores(clf, X_test)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to compute prediction scores for %s: %s", clf_name, exc
                )

            sens = recall_score(y_test, y_pred, average="macro", zero_division=0)
            spec = macro_specificity(y_test, y_pred, labels)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            auc = _compute_auc(y_test, y_score, labels)

            n_samples = len(y_test)
            if bootstrap:
                acc_ci_low, acc_ci_high = _bootstrap_ci(
                    lambda idx: accuracy_score(y_test.iloc[idx], y_pred[idx]),
                    n_samples,
                    n_bootstrap,
                    rng,
                )
                auc_ci_low, auc_ci_high = (
                    _bootstrap_ci(
                        lambda idx: _compute_auc(
                            y_test.iloc[idx],
                            None if y_score is None else y_score[idx],
                            labels,
                            log_failure=False,
                        ),
                        n_samples,
                        n_bootstrap,
                        rng,
                    )
                    if not np.isnan(auc)
                    else (float("nan"), float("nan"))
                )
            else:
                acc_ci_low, acc_ci_high = normal_approx_ci(acc, n_samples)
                auc_ci_low, auc_ci_high = (
                    normal_approx_ci(auc, n_samples)
                    if not np.isnan(auc)
                    else (float("nan"), float("nan"))
                )

            results.append(
                {
                    "Train/Test Ratio": train_ratio,
                    "Classifier": clf_name,
                    "Sensitivity": round(sens, 3),
                    "Specificity": round(spec, 3),
                    "Precision": round(prec, 3),
                    "F1": round(f1, 3),
                    "AUC (95% CI)": _format_ci(auc, auc_ci_low, auc_ci_high),
                    "Accuracy (95% CI)": _format_ci(acc, acc_ci_low, acc_ci_high),
                }
            )

    df_results = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    LOGGER.info("Saved Table S7 to %s", out_path)
    return df_results


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    df = pd.read_csv(args.data)
    train_ratios = [float(x) for x in args.splits.split(",")]
    generate_table_s7(
        df,
        target=args.target,
        train_ratios=train_ratios,
        out_path=Path(args.out),
        random_state=args.random_state,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
