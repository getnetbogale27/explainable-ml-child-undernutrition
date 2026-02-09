# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Table 5: Generalization performance (CV Folds vs. Test Set)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score, confusion_matrix

def calculate_specificity(y_true, y_pred):
    """Helper to calculate macro-averaged specificity."""
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    return np.mean(specificities)

def generate_table_5(model_class, X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Computes metrics for each CV fold and the final test set.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_results = []

    # 1. Evaluate each fold
    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Fit model on this fold's training data
        model_class.fit(X_tr, y_tr)
        y_pred = model_class.predict(X_val)
        
        fold_results.append({
            f"Fold {i+1}": {
                "Specificity": round(calculate_specificity(y_val, y_pred), 4),
                "Sensitivity": round(recall_score(y_val, y_pred, average='macro'), 4),
                "Balanced Accuracy": round(balanced_accuracy_score(y_val, y_pred), 4),
                "Macro-F1-avg": round(f1_score(y_val, y_pred, average='macro'), 4)
            }
        })

    # 2. Evaluate on the final hold-out test set
    model_class.fit(X_train, y_train) # Fit on full training set
    y_test_pred = model_class.predict(X_test)
    
    test_metrics = {
        "Testing": {
            "Specificity": round(calculate_specificity(y_test, y_test_pred), 4),
            "Sensitivity": round(recall_score(y_test, y_test_pred, average='macro'), 4),
            "Balanced Accuracy": round(balanced_accuracy_score(y_test, y_test_pred), 4),
            "Macro-F1-avg": round(f1_score(y_test, y_test_pred, average='macro'), 4)
        }
    }

    # Combine into one DataFrame
    df_folds = pd.DataFrame({k: v for d in fold_results for k, v in d.items()})
    df_test = pd.DataFrame(test_metrics)
    
    table_5 = pd.concat([df_test, df_folds], axis=1)
    
    # Identify the "Best Fold" (e.g., Fold 4* in your image)
    # Logic: Fold with highest Macro-F1
    best_fold_idx = df_folds.loc["Macro-F1-avg"].idxmax()
    table_5.rename(columns={best_fold_idx: f"{best_fold_idx}*"}, inplace=True)
    
    return table_5
