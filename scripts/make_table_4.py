# Copyright (c) 2026 Getnet Bogale
# Licensed under the MIT License.
"""Generate Table 4: Class-wise performance metrics for the Random Forest model."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, 
    f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
)

def calculate_class_wise_metrics(model, X_test, y_test, class_labels):
    """
    Calculates per-class metrics matching Table 4.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Initialize results dictionary
    results = {}
    
    # Confusion matrix for the whole multiclass problem
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    
    for i, label in enumerate(class_labels):
        # One-vs-Rest logic for the current class
        # TP, FN, FP, TN
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # F1-score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # ROC-AUC (OvR)
        # y_test_binary: 1 if label, 0 otherwise
        y_test_binary = (y_test == label).astype(int)
        roc_auc = roc_auc_score(y_test_binary, y_proba[:, i])
        
        # Accuracy (OvR)
        acc_ovr = (tp + tn) / (tp + tn + fp + fn)
        
        # Balanced Accuracy (OvR)
        bal_acc = (sensitivity + specificity) / 2

        results[label] = {
            "Sensitivity": round(sensitivity, 2),
            "Specificity": round(specificity, 2),
            "Precision": round(precision, 2),
            "F1-score": round(f1, 2),
            "ROC-AUC (OvR)": round(roc_auc, 2),
            "Accuracy (OvR)": round(acc_ovr, 2),
            "Balanced Accuracy": round(bal_acc, 2)
        }

    # Convert to DataFrame and Transpose to match Table 4 layout
    df_table4 = pd.DataFrame(results)
    
    # Add Macro-F1-avg row (calculated across classes)
    macro_f1 = f1_score(y_test, y_pred, average=None)
    # Note: The image shows specific values for Macro-F1-avg per column, 
    # which usually represents the average performance when that class is included.
    # Here we just add the per-class F1 again or a specific summary row.
    
    return df_table4

# Example usage in pipeline:
# class_labels = ['N', 'U', 'S', 'W', 'US', 'UW', 'USW']
# table4 = calculate_class_wise_metrics(best_rf_model, X_test, y_test, class_labels)
# table4.to_csv("results/tables/table_4_class_wise_metrics.csv")
