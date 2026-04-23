"""WB-01 to WB-05: Robustness / White Box Tests -- edge cases."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tests.conftest import DROP_COLS, TARGET_COL, make_synthetic_df, build_preprocessor


def _build_pipeline_for(X):
    """Build a full pipeline matching the given X columns."""
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


class TestRobustness:

    # WB-01 -- Empty dataset
    def test_empty_dataset_raises(self, X_y):
        X, _ = X_y
        empty_X = X.iloc[:0]
        empty_y = pd.Series(dtype=int)
        pipeline = _build_pipeline_for(X)
        with pytest.raises((ValueError, IndexError)):
            pipeline.fit(empty_X, empty_y)

    # WB-02 -- All categorical columns are NaN
    def test_all_categorical_nan(self):
        synth = make_synthetic_df(n_rows=50)
        synth = synth.drop(columns=DROP_COLS, errors="ignore")
        X = synth.drop(columns=[TARGET_COL])
        y = synth[TARGET_COL]

        cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
        X[cat_cols] = np.nan
        X[cat_cols] = X[cat_cols].astype("string").fillna("missing")

        pipeline = _build_pipeline_for(X)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(X)

    # WB-03 -- Single-row dataset
    def test_single_row_dataset(self):
        synth = make_synthetic_df(n_rows=10)
        synth = synth.drop(columns=DROP_COLS, errors="ignore")
        X = synth.drop(columns=[TARGET_COL]).head(1)
        y = synth[TARGET_COL].head(1)

        cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
        X[cat_cols] = X[cat_cols].astype("string").fillna("missing")

        pipeline = _build_pipeline_for(X)
        try:
            pipeline.fit(X, y)
            preds = pipeline.predict(X)
            assert len(preds) == 1
        except ValueError:
            pass  # acceptable to raise on single-row fit

    # WB-04 -- Constant numeric column
    def test_constant_numeric_column(self):
        synth = make_synthetic_df(n_rows=50)
        synth = synth.drop(columns=DROP_COLS, errors="ignore")

        X = synth.drop(columns=[TARGET_COL])
        y = synth[TARGET_COL]

        for col in X.select_dtypes(include=np.number).columns:
            X[col] = 42.0

        cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
        X[cat_cols] = X[cat_cols].astype("string").fillna("missing")

        pipeline = _build_pipeline_for(X)
        pipeline.fit(X, y)
        transformed = pipeline.named_steps["preprocess"].transform(X)

        num_count = len(X.select_dtypes(include=np.number).columns)
        num_part = transformed[:, :num_count]
        if hasattr(num_part, "toarray"):
            num_part = num_part.toarray()
        assert np.isfinite(num_part).all(), "Inf or NaN in scaled constant columns"

    # WB-05 -- StratifiedKFold vs cv=5 (int) behavior difference
    def test_stratified_vs_int_cv(self, log_pipeline, X_y):
        X, y = X_y

        cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        res_strat = cross_validate(
            log_pipeline, X, y, cv=cv_strat, scoring={"recall": "recall"},
        )

        res_int = cross_validate(
            log_pipeline, X, y, cv=5, scoring={"recall": "recall"},
        )

        mean_strat = np.mean(res_strat["test_recall"])
        mean_int = np.mean(res_int["test_recall"])
        diff = abs(mean_strat - mean_int)
        assert diff < 0.05, (
            f"StratifiedKFold vs cv=5 differ by {diff:.4f} — "
            "verify MLflow logging uses stratified CV"
        )
