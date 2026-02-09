#!/usr/bin/env python3
"""
Generate Table S7: Performance metrics of ML models across train/test splits.

Usage:
  python scripts/make_table_s7_performance_splits.py --data data/raw/dataset.csv \
      --target concurrent_conditions --splits 0.6,0.7,0.75,0.8,0.85,0.9 \
      --out results/tables/table_s7_performance_splits.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def compute_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0
    return mean, mean - h, mean + h

def specificity_score(y_true, y_pred, pos_label):
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, 'not_'+pos_label])
    if cm.shape != (2,2):
        # fallback for multiclass: compute specificity as TN/(TN+FP) for pos_label vs rest
        tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def evaluate_metrics(y_true, y_pred, y_proba, pos_label):
    sens = recall_score(y_true, y_pred, pos_label=pos_label)
    spec = specificity_score(y_true, y_pred, pos_label)
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score((y_true == pos_label).astype(int), y_proba[:, pos_label]) if y_proba is not None else np.nan
    return sens, spec, prec, f1, auc, acc, bal_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with raw data")
    parser.add_argument("--target", default="concurrent_conditions", help="Target column name")
    parser.add_argument("--splits", default="0.6,0.7,0.75,0.8,0.85,0.9", help="Comma-separated train ratios")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_ratios = [float(x) for x in args.splits.split(",")]

    # Define classifiers to evaluate
    classifiers = {
        "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="saga", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "LASSO": LogisticRegression(penalty="l1", solver="saga", max_iter=2000, random_state=42),
        "Ridge": LogisticRegression(penalty="l2", solver="saga", max_iter=2000, random_state=42),
        "Elastic Net": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000, random_state=42),
    }

    # Encode target as numeric labels for AUC indexing
    classes = sorted(df[args.target].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    results = []

    for train_ratio in train_ratios:
        test_ratio = 1 - train_ratio
        # Stratified split
        train_df, test_df = train_test_split(df, train_size=train_ratio, stratify=df[args.target], random_state=42)
        X_train = train_df.drop(columns=[args.target])
        y_train = train_df[args.target]
        X_test = test_df.drop(columns=[args.target])
        y_test = test_df[args.target]

        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

            # For multiclass, compute metrics per class and average or pick main class? 
            # Your table seems to show overall metrics, so compute macro averages:
            sens = recall_score(y_test, y_pred, average="macro")
            prec = precision_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            # AUC macro average
            if y_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan

            # Compute 95% CI for AUC and Accuracy using bootstrapping or normal approx
            # Here, approximate normal CI for accuracy:
            n = len(y_test)
            se_acc = np.sqrt(acc * (1 - acc) / n)
            ci_acc_low = acc - 1.96 * se_acc
            ci_acc_high = acc + 1.96 * se_acc

            # For AUC CI, approximate with normal approx (less accurate)
            if not np.isnan(auc):
                se_auc = np.sqrt(auc * (1 - auc) / n)
                ci_auc_low = auc - 1.96 * se_auc
                ci_auc_high = auc + 1.96 * se_auc
            else:
                ci_auc_low = ci_auc_high = np.nan

            results.append({
                "Train/Test Ratio": train_ratio,
                "Classifier": clf_name,
                "Sensitivity": round(sens, 3),
                "Specificity": np.nan,  # Specificity macro not directly available; can be computed if needed
                "Precision": round(prec, 3),
                "F1": round(f1, 3),
                "AUC (95% CI)": f"{auc:.3f} ({ci_auc_low:.3f}, {ci_auc_high:.3f})" if not np.isnan(auc) else "NA",
                "Accuracy (95% CI)": f"{acc:.3f} ({ci_acc_low:.3f}, {ci_acc_high:.3f})"
            })

    df_results = pd.DataFrame(results)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(outp, index=False)
    print(f"Saved Table S7 to {outp}")

if __name__ == "__main__":
    main()