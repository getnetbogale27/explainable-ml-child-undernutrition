# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Table 3 comparing 11 classifiers: CV mean±SD and test-set metrics + paired t-tests vs RF."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

from src.data.make_dataset import load_raw_data, basic_cleaning, encode_outcome
from src.data.preprocess import preprocess_and_oversample
from src.models.compare_classifiers import default_classifiers, cross_validate_for_table3

LOGGER = logging.getLogger(__name__)

def evaluate_test_set(classifiers, X_train, y_train, X_test, y_test):
    rows = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
        rows.append({
            "Model": name,
            "Test Acc.": float(accuracy_score(y_test, y_pred)),
            "Test Macro-F1": float(f1_score(y_test, y_pred, average="macro")),
            "Test Balanced Acc.": float(balanced_accuracy_score(y_test, y_pred)),
            "Test ROC-AUC": float(roc_auc_score(y_test, y_proba, multi_class="ovr")) if y_proba is not None else None
        })
    return pd.DataFrame(rows)

def paired_t_tests_vs_rf(per_fold_preds, fold_trues):
    # compute accuracy per fold arrays and run paired t-test vs RF
    rf_preds = per_fold_preds["Random Forest"]
    results = {}
    for name, preds in per_fold_preds.items():
        if name == "Random Forest":
            results[name] = {"p_value": None, "significant": False}
            continue
        rf_scores = []
        other_scores = []
        for i in range(len(rf_preds)):
            rf_scores.append(accuracy_score(fold_trues[i], rf_preds[i]))
            other_scores.append(accuracy_score(fold_trues[i], preds[i]))
        stat, p = ttest_rel(rf_scores, other_scores)
        results[name] = {"p_value": float(p), "significant": bool(p < 0.05)}
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import yaml
    config = yaml.safe_load(open(args.config))
    df_raw = load_raw_data(config["paths"]["data"])
    df_raw = basic_cleaning(df_raw, dropna_on=[config["dataset"].get("waz","waz"), config["dataset"].get("haz","haz"), config["dataset"].get("whz","whz")])
    df_raw = encode_outcome(df_raw, waz_col=config["dataset"].get("waz","waz"), haz_col=config["dataset"].get("haz","haz"), whz_col=config["dataset"].get("whz","whz"), out_col=config["dataset"]["target_col"])

    # Preprocess once with a single fixed test_size (10% as in Table 3)
    pre = preprocess_and_oversample(
        df_raw,
        numeric_features=config["dataset"]["numeric_features"],
        categorical_features=config["dataset"]["categorical_features"],
        ordinal_features=config["dataset"].get("ordinal_features", []),
        ordinal_mappings=config["dataset"].get("ordinal_mappings", {}),
        drop_features=config["dataset"].get("drop_features", []),
        target_col=config["dataset"]["target_col"],
        test_size=config["split"]["test_size"],
        random_state=config["random_state"],
        smote_kwargs={"random_state": config["random_state"]}
    )

    X_train_res = pre["X_train_res"]
    y_train_res = pre["y_train_res"]
    X_test = pre["X_test"]
    y_test = pre["y_test"]

    classifiers = default_classifiers(random_state=config.get("random_state", 42))

    # CV (k-fold) on training set to compute per-fold metrics
    cv_summary_df, per_fold_preds, fold_trues = cross_validate_for_table3(X_train_res, y_train_res, classifiers, cv_folds=config.get("cv_folds", 5))

    # Test set evaluation
    test_df = evaluate_test_set(classifiers, X_train_res, y_train_res, X_test, y_test)

    # Paired t-tests (RF vs others) using per-fold accuracies
    ttest_res = paired_t_tests_vs_rf(per_fold_preds, fold_trues)

    # Merge CV summary and test metrics
    merged = cv_summary_df.merge(test_df, on="Model", how="left")
    merged["p_value_vs_RF"] = merged["Model"].map(lambda m: ttest_res.get(m, {}).get("p_value"))
    merged["significant_vs_RF"] = merged["Model"].map(lambda m: ttest_res.get(m, {}).get("significant", False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print("Saved Table 3 to", args.out)

if __name__ == "__main__":
    main()
