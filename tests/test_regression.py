"""RG-01 to RG-04: Regression Tests -- prevent metric degradation."""

import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_CSV = PROJECT_ROOT / "notebooks" / "results.csv"

SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

THRESHOLDS = {
    "recall": 0.73,
    "roc_auc": 0.84,
    "f1": 0.62,
    "precision": 0.50,
}


class TestRegression:

    @pytest.fixture(scope="class")
    def cv_results(self, log_pipeline, X_y):
        X, y = X_y
        return cross_validate(
            log_pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False,
        )

    # RG-01 -- Metrics do not degrade below thresholds
    @pytest.mark.parametrize("metric,threshold", list(THRESHOLDS.items()))
    def test_metric_above_threshold(self, cv_results, metric, threshold):
        mean_val = np.mean(cv_results[f"test_{metric}"])
        assert mean_val >= threshold, (
            f"{metric} degraded: {mean_val:.4f} < {threshold}"
        )

    # RG-02 -- Pipeline runs without errors on current dependency versions
    def test_pipeline_runs_on_current_deps(self, log_pipeline, X_y):
        X, y = X_y
        log_pipeline.fit(X, y)
        preds = log_pipeline.predict(X.head(10))
        assert len(preds) == 10

    # RG-03 -- No metric falls below baseline
    def test_no_metric_below_baseline(self, cv_results, dummy_pipeline, X_y):
        X, y = X_y
        dummy_res = cross_validate(
            dummy_pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False,
        )
        for metric in SCORING:
            key = f"test_{metric}"
            log_mean = np.mean(cv_results[key])
            dummy_mean = np.mean(dummy_res[key])
            assert log_mean >= dummy_mean, (
                f"{metric}: log({log_mean:.4f}) < dummy({dummy_mean:.4f})"
            )

    # RG-04 -- Reproducibility against saved results
    def test_results_match_saved(self, log_pipeline, X_y):
        if not RESULTS_CSV.exists():
            pytest.skip("results.csv not found")

        saved = pd.read_csv(RESULTS_CSV)
        X, y = X_y

        cv_no_strat = cross_validate(
            log_pipeline, X, y, cv=5,
            scoring={
                "f1": "f1",
                "roc_auc": "roc_auc",
                "precision": "precision",
                "recall": "recall",
            },
        )

        current = pd.DataFrame({
            "f1": cv_no_strat["test_f1"],
            "roc_auc": cv_no_strat["test_roc_auc"],
            "precision": cv_no_strat["test_precision"],
            "recall": cv_no_strat["test_recall"],
        })

        for col in saved.columns:
            saved_mean = saved[col].mean()
            current_mean = current[col].mean()
            diff = abs(saved_mean - current_mean)
            assert diff < 0.05, (
                f"{col}: saved mean={saved_mean:.4f}, current={current_mean:.4f}, "
                f"diff={diff:.4f} > 0.05"
            )
