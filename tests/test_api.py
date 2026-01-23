"""
tests/test_api.py — Async endpoint tests using httpx + pytest-asyncio.

Tests all API routes with both valid and invalid inputs.
Mocks the predictor so tests run without a trained model file.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_predictor():
    pred = MagicMock()
    pred.is_loaded = True
    pred.predict.return_value = {
        "observation_id": "test_obs",
        "is_anomaly": False,
        "anomaly_score": -0.127,
        "confidence": 0.873,
        "votes": {"isolation_forest": False, "lof": False, "zscore": False},
        "model_version": "1.0.0",
        "inference_ms": 1.5,
        "cache_hit": False,
    }
    pred.model_info.return_value = {
        "model_version": "1.0.0",
        "algorithm": "Ensemble (IsolationForest + LOF + Z-score)",
        "n_features": 8,
        "contamination_rate": 0.05,
        "anomaly_threshold": 3.0,
        "training_samples": 8000,
        "f1_score": 0.91,
        "precision": 0.89,
        "recall": 0.93,
        "detectors": ["isolation_forest", "lof", "zscore"],
    }
    return pred


@pytest.fixture
def client(mock_predictor):
    from app import main as app_module
    app_module._predictor = mock_predictor
    app_module._redis_client = None
    with TestClient(app_module.app) as c:
        yield c


# ── Health endpoint ────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_model_loaded(client):
    data = client.get("/health").json()
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"


def test_health_has_uptime(client):
    data = client.get("/health").json()
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


# ── Model info endpoint ────────────────────────────────────────────────────────

def test_model_info_returns_200(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200


def test_model_info_fields(client):
    data = client.get("/model/info").json()
    assert data["n_features"] == 8
    assert data["f1_score"] == 0.91
    assert "detectors" in data
    assert "isolation_forest" in data["detectors"]


# ── Single predict endpoint ────────────────────────────────────────────────────

VALID_FEATURES = [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.1, 0.9]

def test_predict_valid_input(client):
    resp = client.post("/predict", json={"features": VALID_FEATURES})
    assert resp.status_code == 200


def test_predict_response_schema(client):
    data = client.post("/predict", json={"features": VALID_FEATURES}).json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "confidence" in data
    assert "votes" in data
    assert "model_version" in data
    assert "inference_ms" in data


def test_predict_with_observation_id(client):
    resp = client.post("/predict", json={
        "features": VALID_FEATURES,
        "observation_id": "sensor_test_001"
    })
    assert resp.status_code == 200
    assert resp.json()["observation_id"] == "test_obs"  # from mock


def test_predict_empty_features_rejected(client):
    resp = client.post("/predict", json={"features": []})
    assert resp.status_code == 422


def test_predict_missing_features_rejected(client):
    resp = client.post("/predict", json={"observation_id": "x"})
    assert resp.status_code == 422


def test_predict_infinite_value_rejected(client):
    resp = client.post("/predict", json={"features": [float("inf")] + VALID_FEATURES[1:]})
    assert resp.status_code == 422


# ── Batch predict endpoint ─────────────────────────────────────────────────────

def test_batch_predict_valid(client):
    resp = client.post("/predict/batch", json={
        "observations": [
            {"features": VALID_FEATURES},
            {"features": [8.1, -5.3, 7.2, 9.0, -8.4, 6.1, 8.8, -7.2]},
        ]
    })
    assert resp.status_code == 200


def test_batch_predict_response_schema(client):
    data = client.post("/predict/batch", json={
        "observations": [{"features": VALID_FEATURES}]
    }).json()
    assert "predictions" in data
    assert "total" in data
    assert "anomaly_count" in data
    assert "anomaly_rate" in data
    assert data["total"] == 1


def test_batch_predict_empty_rejected(client):
    resp = client.post("/predict/batch", json={"observations": []})
    assert resp.status_code == 422


def test_batch_predict_returns_correct_count(client):
    n = 5
    resp = client.post("/predict/batch", json={
        "observations": [{"features": VALID_FEATURES}] * n
    })
    assert resp.json()["total"] == n


# ── Model unloaded scenarios ───────────────────────────────────────────────────

def test_predict_503_when_model_not_loaded(client, mock_predictor):
    mock_predictor.is_loaded = False
    resp = client.post("/predict", json={"features": VALID_FEATURES})
    assert resp.status_code == 503


def test_health_degraded_when_model_not_loaded(client, mock_predictor):
    mock_predictor.is_loaded = False
    data = client.get("/health").json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False

# FastAPI TestClient endpoint tests: health, predict, batch

# Edge case tests: empty features, infinite values, 503 when unloaded
