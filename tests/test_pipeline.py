"""Unit tests for leakage checks and data shape consistency."""
from __future__ import annotations

import pandas as pd

from src.features.build_features import build_preprocessor, fit_transform_preprocessor
from src.models.train import build_model_pipeline


def test_preprocessor_handles_unseen_categories() -> None:
    train_df = pd.DataFrame(
        {
            "age": [12, 24, 36],
            "sex": ["M", "F", "M"],
            "outcome": ["N", "U", "S"],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [48],
            "sex": ["Other"],
            "outcome": ["W"],
        }
    )

    preprocessor = build_preprocessor(numeric_features=["age"], categorical_features=["sex"])
    X_train, X_test = fit_transform_preprocessor(
        preprocessor,
        train_df.drop(columns=["outcome"]),
        test_df.drop(columns=["outcome"]),
    )

    assert X_train.shape[1] == X_test.shape[1]


def test_pipeline_contains_smote() -> None:
    preprocessor = build_preprocessor(numeric_features=["age"], categorical_features=["sex"])
    pipeline = build_model_pipeline(preprocessor=preprocessor, random_state=7)
    step_names = [name for name, _ in pipeline.steps]
    assert "smote" in step_names