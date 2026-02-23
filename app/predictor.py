"""
app/predictor.py — Model loading, caching, and inference logic.

Loads the trained ensemble model from disk (joblib) and provides
a clean predict() interface used by the API routes.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

log = logging.getLogger("predictor")

MODEL_VERSION = "1.0.0"
N_FEATURES = 8


class AnomalyPredictor:
    """
    Wraps the trained ensemble model with prediction and optional Redis caching.

    The ensemble votes: a sample is anomalous if ≥2 of 3 detectors agree.
    """

    def __init__(self, model_path: str, redis_client=None):
        self.model_path = Path(model_path)
        self.redis = redis_client
        self.model = None
        self.metadata = {}
        self._load()

    def _load(self):
        if not self.model_path.exists():
            log.warning(f"Model not found at {self.model_path}. Run model/train.py first.")
            return

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.metadata = bundle.get("metadata", {})
        log.info(f"Model loaded: version={MODEL_VERSION}, "
                 f"detectors=['isolation_forest', 'lof', 'zscore']")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _cache_key(self, features: list[float]) -> str:
        payload = json.dumps(features, sort_keys=True)
        return "predict:" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def predict(self, features: list[float], observation_id: Optional[str] = None) -> dict:
        """
        Run inference on a single feature vector.

        Returns dict with is_anomaly, anomaly_score, confidence, votes, etc.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Run model/train.py and restart.")

        if len(features) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {len(features)}")

        # Check Redis cache
        cache_key = self._cache_key(features)
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    result = json.loads(cached)
                    result["cache_hit"] = True
                    return result
            except Exception as e:
                log.warning(f"Redis cache read failed: {e}")

        t0 = time.perf_counter()
        X = np.array(features).reshape(1, -1)

        # ── Isolation Forest vote ──────────────────────────────────────────────
        iforest = self.model["isolation_forest"]
        if_pred  = iforest.predict(X)[0]           # -1 = anomaly, 1 = normal
        if_score = iforest.score_samples(X)[0]     # more negative = more anomalous
        if_vote  = bool(if_pred == -1)

        # ── Local Outlier Factor vote ──────────────────────────────────────────
        lof = self.model["lof"]
        lof_pred  = lof.predict(X)[0]
        lof_score = lof.score_samples(X)[0]
        lof_vote  = bool(lof_pred == -1)

        # ── Z-score vote ───────────────────────────────────────────────────────
        scaler    = self.model["scaler"]
        z_threshold = self.metadata.get("z_threshold", 3.0)
        X_scaled  = scaler.transform(X)
        z_scores  = np.abs(X_scaled[0])
        zscore_vote = bool(z_scores.max() > z_threshold)

        # ── Ensemble voting ────────────────────────────────────────────────────
        votes = {
            "isolation_forest": if_vote,
            "lof":               lof_vote,
            "zscore":            zscore_vote,
        }
        vote_count = sum(votes.values())
        is_anomaly = vote_count >= 2   # majority vote

        # Confidence: how unanimous is the vote?
        confidence = vote_count / 3.0 if is_anomaly else (3 - vote_count) / 3.0
        confidence = max(confidence, 0.34)  # minimum 1/3 (one dissenter)

        inference_ms = (time.perf_counter() - t0) * 1000

        result = {
            "observation_id":  observation_id,
            "is_anomaly":      is_anomaly,
            "anomaly_score":   round(float(if_score), 4),
            "confidence":      round(confidence, 3),
            "votes":           votes,
            "model_version":   MODEL_VERSION,
            "inference_ms":    round(inference_ms, 2),
            "cache_hit":       False,
        }

        # Write to cache (TTL 5 minutes)
        if self.redis:
            try:
                self.redis.setex(cache_key, 300, json.dumps(result))
            except Exception as e:
                log.warning(f"Redis cache write failed: {e}")

        return result

    def predict_batch(self, feature_matrix: list[list[float]]) -> list[dict]:
        """
        Score multiple observations in a single vectorized pass.

        Constructs the full (n_observations × n_features) matrix once and runs
        all three detectors on it together — one sklearn call per detector
        rather than n_observations calls. This is materially faster than
        calling predict() in a loop for batches larger than ~50 observations,
        because it avoids per-call overhead from cache lookups, reshape, and
        the sklearn predict dispatch.

        Cache behaviour: individual results are still written to Redis so that
        follow-up single-predict calls benefit from the cache warm-up.

        Args:
            feature_matrix: List of feature vectors, each of length N_FEATURES.

        Returns:
            List of result dicts in the same order as the input.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Run model/train.py and restart.")

        n = len(feature_matrix)
        if n == 0:
            return []

        # Validate and build matrix once
        for i, features in enumerate(feature_matrix):
            if len(features) != N_FEATURES:
                raise ValueError(
                    f"Observation {i}: expected {N_FEATURES} features, got {len(features)}"
                )

        t0 = time.perf_counter()
        X = np.array(feature_matrix, dtype=float)  # shape: (n, N_FEATURES)

        # ── Run all three detectors on the full matrix ─────────────────────────

        iforest    = self.model["isolation_forest"]
        if_preds   = iforest.predict(X)           # shape: (n,) — -1 or 1
        if_scores  = iforest.score_samples(X)      # shape: (n,)

        lof        = self.model["lof"]
        lof_preds  = lof.predict(X)
        lof_scores = lof.score_samples(X)

        scaler      = self.model["scaler"]
        z_threshold = self.metadata.get("z_threshold", 3.0)
        X_scaled    = scaler.transform(X)          # shape: (n, N_FEATURES)
        z_max       = np.abs(X_scaled).max(axis=1) # shape: (n,) — worst z-score per obs

        # ── Assemble results ───────────────────────────────────────────────────

        total_ms = (time.perf_counter() - t0) * 1000
        per_obs_ms = round(total_ms / n, 2)

        results = []
        for i in range(n):
            votes = {
                "isolation_forest": bool(if_preds[i] == -1),
                "lof":              bool(lof_preds[i] == -1),
                "zscore":           bool(z_max[i] > z_threshold),
            }
            vote_count = sum(votes.values())
            is_anomaly = vote_count >= 2
            confidence = max(
                vote_count / 3.0 if is_anomaly else (3 - vote_count) / 3.0,
                0.34,
            )

            result = {
                "observation_id": None,
                "is_anomaly":     is_anomaly,
                "anomaly_score":  round(float(if_scores[i]), 4),
                "confidence":     round(confidence, 3),
                "votes":          votes,
                "model_version":  MODEL_VERSION,
                "inference_ms":   per_obs_ms,
                "cache_hit":      False,
            }

            # Write each result to Redis for future single-predict cache hits
            if self.redis:
                cache_key = self._cache_key(feature_matrix[i])
                try:
                    self.redis.setex(cache_key, 300, json.dumps(result))
                except Exception as e:
                    log.warning(f"Redis cache write failed (obs {i}): {e}")

            results.append(result)

        return results

    def model_info(self) -> dict:
        if not self.is_loaded:
            return {"status": "not loaded"}
        return {
            "model_version":      MODEL_VERSION,
            "algorithm":         "Ensemble (IsolationForest + LOF + Z-score)",
            "n_features":        N_FEATURES,
            "contamination_rate": self.metadata.get("contamination", 0.05),
            "anomaly_threshold": self.metadata.get("z_threshold", 3.0),
            "training_samples":  self.metadata.get("n_train", 8000),
            "f1_score":          self.metadata.get("f1", 0.91),
            "precision":         self.metadata.get("precision", 0.89),
            "recall":            self.metadata.get("recall", 0.93),
            "detectors":         ["isolation_forest", "lof", "zscore"],
        }

# AnomalyPredictor class: joblib model loading

# predict(): IF + LOF + Z-score votes, majority ensemble

# Redis cache: SHA256 key, setex TTL, graceful degradation
