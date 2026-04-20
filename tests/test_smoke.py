"""SM-01 to SM-05: Smoke Tests -- quick sanity checks for the environment."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSmoke:
    """Validate that the basic environment is functional."""

    # SM-01
    def test_main_py_runs_without_error(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Hello from churn-prediction-01!" in result.stdout

    # SM-02
    @pytest.mark.parametrize("module", ["pandas", "sklearn", "mlflow", "numpy"])
    def test_dependency_imports(self, module):
        __import__(module)

    # SM-03
    def test_dataset_loads(self, raw_df):
        assert raw_df.shape == (7043, 33)

    # SM-04
    def test_pipeline_fit_predict(self, log_pipeline, X_y):
        X, y = X_y
        pipe = log_pipeline
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1})

    # SM-05
    def test_mlflow_tracking_uri(self):
        import mlflow

        uri = mlflow.get_tracking_uri()
        assert uri is not None and len(uri) > 0
