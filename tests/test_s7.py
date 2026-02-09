import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def test_make_table_s7_outputs_columns(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    data_path = tmp_path / "synthetic.csv"
    out_path = tmp_path / "table_s7.csv"
    df.to_csv(data_path, index=False)

    cmd = [
        sys.executable,
        "scripts/make_table_s7_performance_splits.py",
        "--data",
        str(data_path),
        "--target",
        "target",
        "--splits",
        "0.8",
        "--out",
        str(out_path),
        "--bootstrap",
        "--n_bootstrap",
        "10",
        "--random_state",
        "0",
        "--n_jobs",
        "1",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert result.returncode == 0

    assert out_path.exists()
    output = pd.read_csv(out_path)
    expected_columns = [
        "Train/Test Ratio",
        "Classifier",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
        "AUC (95% CI)",
        "Accuracy (95% CI)",
    ]
    assert list(output.columns) == expected_columns

    numeric_cols = ["Sensitivity", "Specificity", "Precision", "F1"]
    numeric_values = output[numeric_cols].apply(pd.to_numeric, errors="coerce")
    assert not np.all(np.isnan(numeric_values.to_numpy()))
