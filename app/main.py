"""
app/main.py — FastAPI application entry point.

Routes:
    GET  /health           Service health + model status
    GET  /model/info       Model metadata and evaluation metrics
    POST /predict          Single observation prediction (sync)
    POST /predict/batch    Batch prediction (up to 1000 rows, sync)
    POST /predict/async    Submit prediction job (returns immediately)
    GET  /jobs/{job_id}    Poll for async job result
    GET  /docs             Auto-generated Swagger UI (FastAPI built-in)
"""

import asyncio
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.predictor import AnomalyPredictor
from app.schemas import (
    AsyncPredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    JobStatusResponse,
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

# In-memory job store for async predictions.
# Production would use Redis or a task queue like Celery.
_jobs: dict[str, dict] = {}
_JOB_TTL_SECONDS = 300


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
    Score multiple observations in a single vectorized pass.

    Builds the full feature matrix once and runs all three detectors on it
    together — materially faster than repeated single-predict calls for
    batches larger than ~50 observations. Individual results are still
    written to the Redis cache so follow-up single-predict calls benefit
    from the warm-up.

    Maximum 1000 observations per request.
    """
    if _predictor is None or not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")

    t0 = time.perf_counter()

    # Build feature matrix and delegate to vectorized predict_batch()
    feature_matrix = [obs.features for obs in request.observations]
    obs_ids        = [obs.observation_id for obs in request.observations]

    try:
        raw_results = _predictor.predict_batch(feature_matrix)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail="Batch inference failed")

    # Reattach caller-supplied observation IDs (predict_batch doesn't know them)
    predictions = []
    for result, obs_id in zip(raw_results, obs_ids):
        result["observation_id"] = obs_id
        predictions.append(PredictResponse(**result))

    batch_ms      = (time.perf_counter() - t0) * 1000
    anomaly_count = sum(1 for p in predictions if p.is_anomaly)

    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(predictions), 3),
        batch_inference_ms=round(batch_ms, 2),
    )


# ── Async inference ────────────────────────────────────────────────────────────

async def _run_prediction(job_id: str, features: list[float], observation_id: str | None):
    """Run prediction in a thread pool and store the result."""
    try:
        result = await asyncio.to_thread(
            _predictor.predict,
            features=features,
            observation_id=observation_id,
        )
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["result"] = result
        _jobs[job_id]["completed_at"] = time.time()
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["completed_at"] = time.time()
        log.error(f"Async job {job_id} failed: {e}")


def _expire_old_jobs():
    """Remove jobs older than TTL to prevent memory leaks."""
    now = time.time()
    expired = [
        jid for jid, job in _jobs.items()
        if now - job["created_at"] > _JOB_TTL_SECONDS
    ]
    for jid in expired:
        del _jobs[jid]


@app.post("/predict/async", response_model=AsyncPredictResponse, tags=["Async Inference"])
async def predict_async(request: PredictRequest):
    """
    Submit a prediction job that runs in a background thread.

    Returns a job_id immediately. Poll GET /jobs/{job_id} for the result.
    Useful when the caller doesn't want to block on inference latency.
    """
    if _predictor is None or not _predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    _expire_old_jobs()

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "completed_at": None,
    }

    asyncio.create_task(_run_prediction(job_id, request.features, request.observation_id))
    log.info(f"Async job submitted: {job_id}")

    return AsyncPredictResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Async Inference"])
async def get_job_status(job_id: str):
    """
    Check the status of an async prediction job.

    Returns pending/complete/failed. When complete, includes the full
    prediction result.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    job = _jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
    )


# ── Exception handlers ─────────────────────────────────────────────────────────

def _sanitize_for_json(obj):
    """Replace inf/nan and non-serializable objects so JSON encoding doesn't crash."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # Catch non-serializable objects (e.g. Exception instances in Pydantic ctx)
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 422 with sanitized error details (handles inf/nan in input)."""
    errors = _sanitize_for_json(exc.errors())
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
