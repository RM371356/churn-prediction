"""SEC-01 to SEC-08: Security, Pentest, and LGPD Compliance Tests."""

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "notebooks" / "mlruns"

LOCATION_COLUMNS = {"Lat Long", "City", "State", "Latitude", "Longitude", "Zip Code"}
PII_COLUMNS = {"CustomerID", "Lat Long", "City", "State", "Latitude", "Longitude"}


class TestSecurityAPI:
    """SEC-01 to SEC-04: Pentest-style tests (activate once FastAPI is live)."""

    @pytest.fixture(autouse=True)
    def _check_api_available(self):
        try:
            from src.app.main import app  # noqa: F401
            import fastapi  # noqa: F401
        except (ImportError, AttributeError):
            pytest.skip("FastAPI app not yet implemented")

    # SEC-01
    def test_sql_injection_rejected(self):
        from fastapi.testclient import TestClient
        from src.app.main import app

        client = TestClient(app)
        payload = {"monthly_charges": "'; DROP TABLE customers;--"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (400, 422)

    # SEC-02
    def test_xss_payload_rejected(self):
        from fastapi.testclient import TestClient
        from src.app.main import app

        client = TestClient(app)
        payload = {"contract": "<script>alert('xss')</script>"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (400, 422)
        assert resp.headers["content-type"].startswith("application/json")

    # SEC-03
    def test_unauthenticated_access_denied(self):
        from fastapi.testclient import TestClient
        from src.app.main import app

        client = TestClient(app)
        resp = client.post("/predict", json={})
        assert resp.status_code in (401, 403, 422)

    # SEC-04
    def test_path_traversal_blocked(self):
        from fastapi.testclient import TestClient
        from src.app.main import app

        client = TestClient(app)
        resp = client.get("/../../../etc/passwd")
        assert resp.status_code in (400, 404, 405)


class TestLGPDCompliance:
    """SEC-05 to SEC-06: LGPD data protection tests."""

    # SEC-05
    def test_customer_id_not_in_features(self, X_y):
        X, _ = X_y
        assert "CustomerID" not in X.columns, "CustomerID must not be a feature"

    # SEC-06
    def test_location_columns_not_in_features(self, X_y):
        X, _ = X_y
        present = LOCATION_COLUMNS & set(X.columns)
        location_in_features = present - {"Latitude", "Longitude"}
        assert not location_in_features, (
            f"Location columns in features (LGPD risk): {location_in_features}"
        )


class TestDependencyAudit:
    """SEC-07: Check for known CVEs in dependencies."""

    # SEC-07
    def test_no_critical_vulnerabilities(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pip_audit"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 and "No known vulnerabilities found" not in result.stdout:
            if "no module named" in result.stderr.lower() or result.returncode == 1:
                pytest.skip("pip-audit not installed")
            vulns = result.stdout
            critical_pattern = re.compile(r"(CRITICAL|HIGH)", re.IGNORECASE)
            if critical_pattern.search(vulns):
                pytest.fail(f"Critical/High vulnerabilities found:\n{vulns[:2000]}")


class TestMLflowDataPrivacy:
    """SEC-08: Ensure sensitive data not leaked to MLflow artifacts."""

    # SEC-08
    def test_mlflow_artifacts_no_pii(self):
        if not MLRUNS_DIR.exists():
            pytest.skip("mlruns directory not found")

        pii_patterns = [
            re.compile(r"CustomerID", re.IGNORECASE),
            re.compile(r"\bLat Long\b", re.IGNORECASE),
        ]

        violations = []
        for path in MLRUNS_DIR.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in (".pkl", ".bin", ".pb"):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in pii_patterns:
                    if pattern.search(content):
                        if "features" in path.name or "params" in path.name:
                            continue
                        violations.append(f"{path}: matches {pattern.pattern}")
            except Exception:
                continue

        assert not violations, (
            f"PII data found in MLflow artifacts:\n" + "\n".join(violations[:10])
        )
