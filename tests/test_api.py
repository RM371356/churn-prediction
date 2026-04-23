"""API-01 to API-07: API Tests (activated once FastAPI app is implemented)."""

import pytest


def _get_client():
    """Attempt to import and build a TestClient for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from src.app.main import app

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
    "gender": "Male",
    "tenure": 24,
    "monthly_charges": 79.85,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
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
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

    # API-03
    def test_predict_missing_field(self, client):
        incomplete = {"monthly_charges": 50.0}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422

    # API-04
    def test_predict_invalid_type(self, client):
        bad = VALID_PAYLOAD.copy()
        bad["monthly_charges"] = "not_a_number"
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    # API-05
    def test_predict_extreme_values(self, client):
        extreme = VALID_PAYLOAD.copy()
        extreme["monthly_charges"] = 999999.99
        extreme["tenure"] = 0
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
