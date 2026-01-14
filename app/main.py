"""
app/main.py — FastAPI application entry point.

Routes:
    GET  /health           Service health + model status
    GET  /model/info       Model metadata and evaluation metrics
    POST /predict          Single observation prediction
    POST /predict/batch    Batch prediction (up to 1000 rows)
    GET  /docs             Auto-generated Swagger UI (FastAPI built-in)
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.predictor import AnomalyPredictor
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("main")

# ── Global state ───────────────────────────────────────────────────────────────
_predictor: AnomalyPredictor | None = None
_redis_client = None
_start_time = time.time()


# ── Lifespan: startup / shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to Redis on startup; clean up on shutdown."""
    global _predictor, _redis_client

    # Redis (optional — gracefully degrade if not available)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        _redis_client = redis_lib.from_url(redis_url, decode_responses=True, socket_timeout=2)
        _redis_client.ping()
        log.info(f"Redis connected: {redis_url}")
    except Exception as e:
        log.warning(f"Redis unavailable ({e}) — running without cache")
        _redis_client = None

    # Model
    model_path = os.getenv("MODEL_PATH", "model/artifacts/ensemble_model.joblib")
    _predictor = AnomalyPredictor(model_path=model_path, redis_client=_redis_client)

    if _predictor.is_loaded:
        log.info("Model loaded successfully. API ready.")
    else:
        log.warning("Model not loaded — /predict will return 503 until model is trained.")

    yield

    # Shutdown
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Anomaly Detection ML API",
    description=(
        "REST API for real-time anomaly detection using an ensemble of "
        "Isolation Forest, Local Outlier Factor, and Z-score detectors. "
        "Built with FastAPI + Docker."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: request logging ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000
    log.info(f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.1f}ms)")
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Service health check. Returns model status and Redis connectivity."""
    redis_ok = False
    if _redis_client:
        try:
            _redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    model_loaded = _predictor is not None and _predictor.is_loaded
    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        redis_connected=redis_ok,
        model_version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Model metadata: algorithm, evaluation metrics, feature count."""
    if _predictor is None or not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(**_predictor.model_info())


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Score a single observation.

    Returns anomaly flag, score, confidence, and per-detector votes.
    """
    if _predictor is None or not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")

    try:
        result = _predictor.predict(
            features=request.features,
            observation_id=request.observation_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    Score multiple observations in a single call.

    More efficient than repeated single-predict calls.
    Maximum 1000 observations per request.
    """
    if _predictor is None or not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")

    t0 = time.perf_counter()
    predictions = []

    for obs in request.observations:
        try:
            result = _predictor.predict(
                features=obs.features,
                observation_id=obs.observation_id,
            )
            predictions.append(PredictResponse(**result))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            log.error(f"Batch inference error: {e}")
            raise HTTPException(status_code=500, detail="Batch inference failed")

    batch_ms = (time.perf_counter() - t0) * 1000
    anomaly_count = sum(1 for p in predictions if p.is_anomaly)

    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(predictions), 3),
        batch_inference_ms=round(batch_ms, 2),
    )


# ── Exception handlers ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# FastAPI app with lifespan: model load + Redis connect on startup
