#!/usr/bin/env python3
"""
Generate Table S6: Fold accuracies, mean and std for a set of classifiers.

Usage:
  python scripts/make_table_s6_fold_accuracies.py --X data/processed/X_train.csv --y data/processed/y_train.csv \
      --out results/tables/table_s6_fold_accuracies.csv --n_splits 5 --random_state 42
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def default_classifiers(random_state: int = 42):
    return {
        "Support Vector Machine": SVC(probability=False, kernel="rbf", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="saga", random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "L-1 Regularization (LASSO)": LogisticRegression(penalty="l1", solver="saga", max_iter=2000, random_state=random_state),
        "L-2 Regularization (Ridge)": LogisticRegression(penalty="l2", solver="saga", max_iter=2000, random_state=random_state),
        "Elastic Net Regularization": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000, random_state=random_state),
    }

def compute_fold_accuracies(X, y, classifiers, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    for name, clf in classifiers.items():
        fold_accs = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_val, y_val)
            fold_accs.append(acc)
        results.append((name, fold_accs))
    # Build DataFrame
    rows = []
    for name, fold_accs in results:
        arr = np.array(fold_accs)
        row = {"Model": name}
        for i, a in enumerate(arr, start=1):
            row[f"Fold{i}"] = round(float(a), 3)
        row["Mean Accuracy"] = round(float(arr.mean()), 3)
        row["Std Accuracy"] = round(float(arr.std(ddof=1)), 3) if arr.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--X", required=True, help="CSV of feature matrix (rows aligned with y).")
    p.add_argument("--y", required=True, help="CSV (single column) or path to series for labels.")
    p.add_argument("--out", required=True, help="Output CSV path for Table S6.")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    X = pd.read_csv(args.X, index_col=None)
    y_df = pd.read_csv(args.y, index_col=None)
    # If y is a dataframe with one column, extract it
    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        # assume a single-column CSV, else try a column named 'target' or 'y'
        if "target" in y_df.columns:
            y = y_df["target"]
        elif "y" in y_df.columns:
            y = y_df["y"]
        else:
            # fallback to first column
            y = y_df.iloc[:, 0]

    classifiers = default_classifiers(random_state=args.random_state)
    df_results = compute_fold_accuracies(X, y, classifiers, n_splits=args.n_splits, random_state=args.random_state)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(outp, index=False)
    print("Saved Table S6 to", outp)

if __name__ == "__main__":
    main()