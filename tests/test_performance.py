"""PF-01 to PF-05: Performance Tests -- timing and resource usage."""

import time
import tracemalloc

import numpy as np
import pytest


class TestPerformance:

    # PF-01 -- Training time
    def test_training_time(self, log_pipeline, X_y):
        X, y = X_y
        start = time.perf_counter()
        log_pipeline.fit(X, y)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Training took {elapsed:.2f}s (limit: 10s)"

    # PF-02 -- Single prediction time (median of multiple runs to reduce noise)
    def test_single_prediction_time(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        single = X.head(1)

        for _ in range(3):
            fitted_log_pipeline.predict(single)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            fitted_log_pipeline.predict(single)
            times.append(time.perf_counter() - start)

        median_time = sorted(times)[len(times) // 2]
        assert median_time < 0.10, (
            f"Single prediction took {median_time*1000:.1f}ms median (limit: 100ms)"
        )

    # PF-03 -- Batch prediction time
    def test_batch_prediction_time(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        start = time.perf_counter()
        fitted_log_pipeline.predict(X)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Batch prediction took {elapsed:.2f}s (limit: 2s)"

    # PF-04 -- Memory usage during training
    def test_memory_usage(self, log_pipeline, X_y):
        X, y = X_y
        tracemalloc.start()
        log_pipeline.fit(X, y)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 500, f"Peak memory: {peak_mb:.1f}MB (limit: 500MB)"

    # PF-05 -- API startup time
    def test_api_startup_time(self):
        try:
            import importlib
            start = time.perf_counter()
            importlib.import_module("src.app.main")
            elapsed = time.perf_counter() - start
            assert elapsed < 5.0, f"API module import took {elapsed:.2f}s"
        except ImportError:
            pytest.skip("FastAPI app module not importable")
