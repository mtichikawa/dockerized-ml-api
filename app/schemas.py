"""
app/schemas.py — Pydantic v2 request and response models.

All API input/output is validated through these schemas.
FastAPI auto-generates OpenAPI docs from these definitions.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Request models ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-observation prediction request."""

    features: list[float] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Feature vector for the observation. Must match training dimensionality (8 features).",
        examples=[[0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.1, 0.9]],
    )
    observation_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional caller-supplied ID for tracking. Echoed back in response.",
        examples=["sensor_42_t1234"],
    )

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v):
        import math
        for i, f in enumerate(v):
            if not math.isfinite(f):
                raise ValueError(f"Feature at index {i} is not finite: {f}")
        return v


class BatchPredictRequest(BaseModel):
    """Batch prediction request — up to 1000 observations."""

    observations: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of observations to score.",
    )


# ── Response models ────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Single-observation prediction response."""

    observation_id: Optional[str] = Field(
        default=None,
        description="Echoed from request if provided.",
    )
    is_anomaly: bool = Field(
        description="True if the ensemble flagged this observation as anomalous.",
    )
    anomaly_score: float = Field(
        description=(
            "Raw ensemble score. More negative = more anomalous "
            "(Isolation Forest convention). Range roughly -1 to 0.5."
        ),
    )
    confidence: float = Field(
        description="Model confidence in the prediction, 0–1.",
        ge=0.0,
        le=1.0,
    )
    votes: dict[str, bool] = Field(
        description="Individual detector votes: isolation_forest, lof, zscore.",
    )
    model_version: str = Field(description="Deployed model version.")
    inference_ms: float = Field(description="Server-side inference latency in milliseconds.")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictResponse]
    total: int = Field(description="Total observations scored.")
    anomaly_count: int = Field(description="Number flagged as anomalous.")
    anomaly_rate: float = Field(description="Fraction flagged as anomalous.")
    batch_inference_ms: float = Field(description="Total batch inference latency in ms.")


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(description="'healthy' or 'degraded'")
    model_loaded: bool
    redis_connected: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    algorithm: str
    n_features: int
    contamination_rate: float
    anomaly_threshold: float
    training_samples: int
    f1_score: float
    precision: float
    recall: float
    detectors: list[str]

# PredictRequest and PredictResponse Pydantic v2 models

# Added BatchPredictRequest, HealthResponse, ModelInfoResponse
