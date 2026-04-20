"""API-01 to API-07: API Tests (activated once FastAPI app is implemented)."""

import pytest


def _get_client():
    """Attempt to import and build a TestClient for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        if not hasattr(app, "routes") or not app.routes:
            return None
        return TestClient(app)
    except Exception:
        return None


@pytest.fixture
def client():
    c = _get_client()
    if c is None:
        pytest.skip("FastAPI app not yet implemented")
    return c


VALID_PAYLOAD = {
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 24,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 79.85,
    "Total Charges": 1889.50,
    "CLTV": 4500,
    "Country": "United States",
    "Latitude": 34.05,
    "Longitude": -118.25,
}


class TestAPI:

    # API-01
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    # API-02
    def test_predict_valid_payload(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] in (0, 1)
        if "churn_probability" in data:
            assert 0.0 <= data["churn_probability"] <= 1.0

    # API-03
    def test_predict_missing_field(self, client):
        incomplete = {"Monthly Charges": 50.0}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422

    # API-04
    def test_predict_invalid_type(self, client):
        bad = VALID_PAYLOAD.copy()
        bad["Monthly Charges"] = "not_a_number"
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    # API-05
    def test_predict_extreme_values(self, client):
        extreme = VALID_PAYLOAD.copy()
        extreme["Monthly Charges"] = 999999.99
        extreme["Tenure Months"] = 0
        resp = client.post("/predict", json=extreme)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data

    # API-06
    def test_concurrent_requests(self, client):
        import concurrent.futures

        def send_request():
            return client.post("/predict", json=VALID_PAYLOAD)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(send_request) for _ in range(10)]
            results = [f.result() for f in futures]

        for r in results:
            assert r.status_code == 200

    # API-07
    def test_extra_fields_ignored(self, client):
        extended = VALID_PAYLOAD.copy()
        extended["unexpected_field"] = "should_be_ignored"
        extended["another_extra"] = 12345
        resp = client.post("/predict", json=extended)
        assert resp.status_code == 200
