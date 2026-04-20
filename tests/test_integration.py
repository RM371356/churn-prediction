"""IT-01 to IT-05: Integration Tests -- end-to-end pipeline validation."""

import pathlib
import tempfile

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


class TestIntegration:

    # IT-01
    def test_end_to_end_pipeline(self, log_pipeline, X_y):
        X, y = X_y
        log_pipeline.fit(X, y)
        preds = log_pipeline.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})
        churn_rate = preds.mean()
        assert 0.05 < churn_rate < 0.95, "Predictions are all one class"

    # IT-02
    def test_cross_validate_completes(self, log_pipeline, X_y):
        X, y = X_y
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = cross_validate(
            log_pipeline, X, y, cv=cv, scoring=SCORING, return_train_score=False,
        )
        for metric in SCORING:
            key = f"test_{metric}"
            assert key in results
            assert len(results[key]) == 5

    # IT-03
    def test_mlflow_logging(self, log_pipeline, X_y):
        import mlflow
        import mlflow.sklearn

        X, y = X_y

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file:{tmpdir}")
            mlflow.set_experiment("test-integration")

            with mlflow.start_run() as run:
                mlflow.log_metric("test_recall", 0.78)
                mlflow.log_param("model", "LogisticRegression")
                log_pipeline.fit(X, y)
                mlflow.sklearn.log_model(log_pipeline, "model")

            run_data = mlflow.get_run(run.info.run_id)
            assert run_data.data.metrics["test_recall"] == 0.78
            assert run_data.data.params["model"] == "LogisticRegression"

    # IT-04
    def test_mlflow_model_save_load(self, log_pipeline, X_y):
        import mlflow
        import mlflow.sklearn

        X, y = X_y

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file:{tmpdir}")
            mlflow.set_experiment("test-model-persist")

            with mlflow.start_run() as run:
                log_pipeline.fit(X, y)
                mlflow.sklearn.log_model(log_pipeline, "model")

            model_uri = f"runs:/{run.info.run_id}/model"
            loaded = mlflow.sklearn.load_model(model_uri)
            sample = X.head(20)
            np.testing.assert_array_equal(
                log_pipeline.predict(sample),
                loaded.predict(sample),
            )

    # IT-05
    def test_results_csv_format(self):
        if not RESULTS_CSV.exists():
            pytest.skip("results.csv not found")

        df = pd.read_csv(RESULTS_CSV)
        assert list(df.columns) == ["f1", "roc_auc", "precision", "recall"]
        assert len(df) == 5  # 5 folds
        assert (df >= 0).all().all()
        assert (df <= 1).all().all()
