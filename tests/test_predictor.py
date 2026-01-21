"""
tests/test_predictor.py — Unit tests for AnomalyPredictor inference logic.
Tests run against a freshly-trained in-memory model (no saved file needed).
"""

import numpy as np
import pytest
import joblib
import tempfile
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from app.predictor import AnomalyPredictor, N_FEATURES


@pytest.fixture(scope="session")
def trained_model_path():
    """Train a minimal model and save it to a temp file."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((500, N_FEATURES))

    iforest = IsolationForest(n_estimators=50, contamination=0.05, random_state=0)
    iforest.fit(X_train)

    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05, novelty=True)
    lof.fit(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    bundle = {
        "model": {"isolation_forest": iforest, "lof": lof, "scaler": scaler},
        "metadata": {"contamination": 0.05, "z_threshold": 3.0, "n_train": 500,
                     "f1": 0.85, "precision": 0.82, "recall": 0.88},
    }
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump(bundle, tmp.name)
    return tmp.name


@pytest.fixture
def predictor(trained_model_path):
    return AnomalyPredictor(model_path=trained_model_path, redis_client=None)


def test_predictor_loads(predictor):
    assert predictor.is_loaded


def test_predict_normal_returns_dict(predictor):
    features = [0.1] * N_FEATURES
    result = predictor.predict(features)
    assert isinstance(result, dict)


def test_predict_has_required_keys(predictor):
    result = predictor.predict([0.1] * N_FEATURES)
    assert "is_anomaly" in result
    assert "anomaly_score" in result
    assert "confidence" in result
    assert "votes" in result
    assert "model_version" in result
    assert "inference_ms" in result


def test_predict_votes_are_booleans(predictor):
    result = predictor.predict([0.1] * N_FEATURES)
    for k, v in result["votes"].items():
        assert isinstance(v, bool), f"Vote {k} should be bool, got {type(v)}"


def test_predict_confidence_in_range(predictor):
    result = predictor.predict([0.1] * N_FEATURES)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_anomaly_detected_for_extreme_values(predictor):
    # Extreme values should trigger at least the Z-score detector
    extreme = [10.0] * N_FEATURES
    result = predictor.predict(extreme)
    # Z-score vote should be True
    assert result["votes"]["zscore"] is True


def test_predict_wrong_feature_count_raises(predictor):
    with pytest.raises(ValueError, match="Expected 8 features"):
        predictor.predict([0.1, 0.2, 0.3])


def test_predict_batch_returns_list(predictor):
    features = [[0.1] * N_FEATURES, [0.5] * N_FEATURES]
    results = predictor.predict_batch(features)
    assert isinstance(results, list)
    assert len(results) == 2


def test_model_info_returns_metadata(predictor):
    info = predictor.model_info()
    assert "model_version" in info
    assert info["n_features"] == N_FEATURES
    assert "detectors" in info


def test_predictor_not_loaded_for_missing_path():
    pred = AnomalyPredictor(model_path="/nonexistent/path.joblib")
    assert not pred.is_loaded


def test_predict_raises_when_not_loaded():
    pred = AnomalyPredictor(model_path="/nonexistent/path.joblib")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        pred.predict([0.1] * N_FEATURES)

# AnomalyPredictor unit tests with in-memory model fixture
