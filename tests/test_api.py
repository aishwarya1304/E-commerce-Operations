"""
tests/test_api.py
──────────────────
Pytest test suite for all API endpoints.

Run:
    pytest tests/ -v --cov=api --cov-report=term-missing
"""

import pytest
import json
from app import create_app
from api.extensions import db as _db


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Create app with in-memory SQLite for tests."""
    application = create_app("testing")
    with application.app_context():
        _db.create_all()
        yield application
        _db.drop_all()


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def auth_headers(client):
    """Login once and return JWT headers for reuse."""
    resp = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    token = resp.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_order():
    return {
        "order_id":               "ORD99001",
        "customer_id":            "CUST0001",
        "warehouse_id":           "WH_North",
        "city":                   "Delhi",
        "distance_km":            200,
        "order_value":            1500.0,
        "order_items":            3,
        "promised_delivery_days": 3,
        "warehouse_load":         180,
        "past_delays":            2,
    }


# ── Auth tests ───────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_success(self, client):
        resp = client.post("/api/auth/login",
                           json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "access_token" in data
        assert data["username"] == "admin"

    def test_login_wrong_password(self, client):
        resp = client.post("/api/auth/login",
                           json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post("/api/auth/login",
                           json={"username": "ghost", "password": "x"})
        assert resp.status_code == 401

    def test_protected_route_without_token(self, client):
        resp = client.get("/api/orders/")
        assert resp.status_code == 401


# ── Health check ─────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


# ── Orders tests ─────────────────────────────────────────────────────────────

class TestOrders:
    def test_list_orders_empty(self, client, auth_headers):
        resp = client.get("/api/orders/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "orders" in data
        assert "total" in data

    def test_create_order_success(self, client, auth_headers, sample_order):
        resp = client.post("/api/orders/",
                           json=sample_order,
                           headers=auth_headers)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["order"]["order_id"] == "ORD99001"
        assert "prediction" in data
        assert "delay_probability" in data["prediction"]
        assert data["prediction"]["risk_category"] in ("Low", "Medium", "High")

    def test_create_order_duplicate(self, client, auth_headers, sample_order):
        # Same order_id a second time
        resp = client.post("/api/orders/",
                           json=sample_order,
                           headers=auth_headers)
        assert resp.status_code == 409

    def test_create_order_missing_field(self, client, auth_headers):
        bad_order = {"order_id": "ORD99002", "city": "Mumbai"}
        resp = client.post("/api/orders/", json=bad_order, headers=auth_headers)
        assert resp.status_code == 400
        assert "Missing fields" in resp.get_json()["error"]

    def test_get_order(self, client, auth_headers):
        resp = client.get("/api/orders/ORD99001", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["order_id"] == "ORD99001"

    def test_get_order_not_found(self, client, auth_headers):
        resp = client.get("/api/orders/NOTEXIST", headers=auth_headers)
        assert resp.status_code == 404

    def test_update_order_status(self, client, auth_headers):
        resp = client.put(
            "/api/orders/ORD99001/status",
            json={"status": "delivered", "actual_delivery_days": 4},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "delivered"
        assert data["actual_delivery_days"] == 4

    def test_list_orders_with_filter(self, client, auth_headers):
        resp = client.get("/api/orders/?city=Delhi", headers=auth_headers)
        assert resp.status_code == 200
        orders = resp.get_json()["orders"]
        for o in orders:
            assert o["city"] == "Delhi"


# ── Predictions tests ─────────────────────────────────────────────────────────

class TestPredictions:
    def test_model_info(self, client):
        resp = client.get("/api/predictions/model-info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_loaded" in data
        assert "model_version" in data

    def test_single_prediction(self, client, auth_headers):
        payload = {
            "distance_km":            300,
            "warehouse_load":         220,
            "order_items":            5,
            "past_delays":            3,
            "order_value":            2500,
            "promised_delivery_days": 3,
            "warehouse_id":           "WH_West",
        }
        resp = client.post("/api/predictions/single",
                           json=payload,
                           headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert 0.0 <= data["delay_probability"] <= 1.0
        assert data["risk_category"] in ("Low", "Medium", "High")

    def test_single_prediction_missing_fields(self, client, auth_headers):
        resp = client.post("/api/predictions/single",
                           json={"distance_km": 100},
                           headers=auth_headers)
        assert resp.status_code == 400

    def test_batch_prediction(self, client, auth_headers):
        orders = [
            {"distance_km": 50,  "warehouse_load": 80,  "order_items": 2,
             "past_delays": 0, "order_value": 800,  "promised_delivery_days": 2,
             "warehouse_id": "WH_South"},
            {"distance_km": 400, "warehouse_load": 230, "order_items": 10,
             "past_delays": 4, "order_value": 15000, "promised_delivery_days": 5,
             "warehouse_id": "WH_West"},
        ]
        resp = client.post("/api/predictions/batch",
                           json={"orders": orders},
                           headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_prediction_limit(self, client, auth_headers):
        # 501 orders → should reject
        orders = [{"distance_km": 10}] * 501
        resp = client.post("/api/predictions/batch",
                           json={"orders": orders},
                           headers=auth_headers)
        assert resp.status_code == 400

    def test_high_risk_order_creates_alert(self, client, auth_headers):
        """A high-distance, high-load, many-past-delays order should be flagged."""
        high_risk = {
            "order_id":               "ORD99HR1",
            "customer_id":            "CUST0001",
            "warehouse_id":           "WH_West",
            "city":                   "Kolkata",
            "distance_km":            490,
            "order_value":            5000,
            "order_items":            12,
            "promised_delivery_days": 2,
            "warehouse_load":         249,
            "past_delays":            5,
        }
        resp = client.post("/api/orders/", json=high_risk, headers=auth_headers)
        assert resp.status_code == 201
        # rule-based or model should flag this as medium/high
        assert resp.get_json()["prediction"]["risk_category"] in ("Medium", "High")


# ── Analytics tests ───────────────────────────────────────────────────────────

class TestAnalytics:
    def test_kpis(self, client, auth_headers):
        resp = client.get("/api/analytics/kpis", headers=auth_headers)
        assert resp.status_code == 200

    def test_city_analysis(self, client, auth_headers):
        resp = client.get("/api/analytics/city-analysis", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)

    def test_warehouse_performance(self, client, auth_headers):
        resp = client.get("/api/analytics/warehouse-perf", headers=auth_headers)
        assert resp.status_code == 200

    def test_financial_impact(self, client, auth_headers):
        resp = client.get("/api/analytics/financial-impact", headers=auth_headers)
        assert resp.status_code == 200
