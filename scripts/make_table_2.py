# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Table 2 style comparisons across multiple train/test splits and classifiers."""
import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score

from src.data.make_dataset import load_raw_data, encode_outcome, basic_cleaning
from src.data.preprocess import preprocess_and_oversample

LOGGER = logging.getLogger(__name__)

CLASSIFIERS = {
    "RF": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "GB": GradientBoostingClassifier(n_estimators=300, random_state=42)
}

def run_split_experiments(df_raw, config, test_sizes=[0.40, 0.30, 0.25, 0.20, 0.15, 0.10], repeats=5):
    rows = []
    for test_size in test_sizes:
        accs = []
        results_for_split = []
        for r in range(repeats):
            pre = preprocess_and_oversample(
                df_raw,
                numeric_features=config["dataset"]["numeric_features"],
                categorical_features=config["dataset"]["categorical_features"],
                ordinal_features=config["dataset"].get("ordinal_features", []),
                ordinal_mappings=config["dataset"].get("ordinal_mappings", {}),
                drop_features=config["dataset"].get("drop_features", []),
                target_col=config["dataset"]["target_col"],
                test_size=test_size,
                random_state=42 + r,
                smote_kwargs={"random_state": 42 + r}
            )
            X_train, y_train = pre["X_train_res"], pre["y_train_res"]
            X_test, y_test = pre["X_test"], pre["y_test"]

            best_model_name = None
            best_macro_f1 = -1
            best_metrics = None
            # evaluate classifiers on this split
            for name, clf in CLASSIFIERS.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
                macro_f1 = f1_score(y_test, y_pred, average="macro")
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_model_name = name
                    best_metrics = {
                        "sensitivity": float(recall_score(y_test, y_pred, average="macro")),
                        "precision": float(precision_score(y_test, y_pred, average="macro")),
                        "f1": float(macro_f1),
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                        "roc_auc": float(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class="ovr")) if y_proba is not None else None
                    }
            results_for_split.append((best_model_name, best_metrics, X_train.shape[0], X_test.shape[0]))
        # aggregate metrics across repeats -> mean +/- CI for Accuracy and AUC
        accuracies = np.array([m[1]["accuracy"] for m in results_for_split])
        aucs = np.array([m[1]["roc_auc"] if m[1]["roc_auc"] is not None else np.nan for m in results_for_split])
        mean_acc = accuracies.mean()
        se_acc = accuracies.std(ddof=1) / np.sqrt(len(accuracies))
        ci = 1.96 * se_acc
        mean_auc = np.nanmean(aucs)
        se_auc = np.nanstd(aucs, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(aucs))) if np.count_nonzero(~np.isnan(aucs))>0 else np.nan
        auc_ci = 1.96 * se_auc if not np.isnan(se_auc) else np.nan

        # choose the modal "best" classifier across repeats
        best_models = [m[0] for m in results_for_split]
        best_mode = max(set(best_models), key=best_models.count)
        example_metrics = next(m[1] for m in results_for_split if m[0] == best_mode)

        rows.append({
            "Train/Test Split": f"{int((1-test_size)*100)}:{int(test_size*100)}",
            "Shape of Train": f"{results_for_split[0][2]}, {len(config['dataset']['numeric_features']) + len(config['dataset']['categorical_features'])}",
            "Shape of Test": f"{results_for_split[0][3]}, {len(config['dataset']['numeric_features']) + len(config['dataset']['categorical_features'])}",
            "Best Classifier": best_mode,
            "Sensitivity": example_metrics["sensitivity"],
            "Precision": example_metrics["precision"],
            "F1-score": example_metrics["f1"],
            "AUC (95% CI)": f"{mean_auc:.3f} ({mean_auc-auc_ci:.3f}, {mean_auc+auc_ci:.3f})" if not np.isnan(mean_auc) else "NA",
            "Accuracy (95% CI)": f"{mean_acc:.3f} ({mean_acc-ci:.3f}, {mean_acc+ci:.3f})",
            "Balanced Accuracy (mean)": float(np.mean([m[1]["balanced_accuracy"] for m in results_for_split])),
            "Macro-F1 (mean)": float(np.mean([m[1]["f1"] for m in results_for_split])),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config JSON/YAML with dataset keys and feature lists")
    p.add_argument("--out", required=True, help="Output CSV for table2")
    args = p.parse_args()
    import yaml
    config = yaml.safe_load(open(args.config))
    df_raw = load_raw_data(config["paths"]["data"])
    df_raw = basic_cleaning(df_raw, dropna_on=[config["dataset"].get("waz", "waz"), config["dataset"].get("haz","haz"), config["dataset"].get("whz","whz")])
    df_raw = encode_outcome(df_raw, waz_col=config["dataset"].get("waz","waz"), haz_col=config["dataset"].get("haz","haz"), whz_col=config["dataset"].get("whz","whz"), out_col=config["dataset"]["target_col"])
    df_table = run_split_experiments(df_raw, config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(args.out, index=False)
    print("Saved Table 2 to", args.out)

if __name__ == "__main__":
    main()
