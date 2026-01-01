# Dockerized ML API

A production-ready machine learning inference service built with **FastAPI** and **Docker**. Wraps a trained anomaly detection ensemble model (Isolation Forest + LOF + statistical Z-score) behind a REST API with health checks, input validation, batch prediction support, and a docker-compose stack for local orchestration.

---

## Motivation

Most ML projects end at the Jupyter notebook. This one goes further — packaging the model into a containerized REST API that can be deployed anywhere Docker runs. The goal was to practice the "last mile" of ML engineering: model serialization, API design, containerization, and endpoint testing.

---

## What This Project Does

1. **Trains** an anomaly detection ensemble on synthetic time-series sensor data
2. **Serializes** the model with joblib and embeds it in a Docker image
3. **Serves** predictions via a FastAPI REST endpoint with Pydantic input validation
4. **Provides** `/health`, `/predict`, `/predict/batch`, and `/model/info` endpoints
5. **Orchestrates** API + Redis cache layer via docker-compose
6. **Tests** all endpoints with pytest + httpx async client

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.109 |
| Data validation | Pydantic v2 |
| ML models | scikit-learn (IsolationForest, LocalOutlierFactor) |
| Model serialization | joblib |
| Containerization | Docker, docker-compose |
| Caching | Redis (via docker-compose) |
| Testing | pytest, httpx (async) |
| Server | uvicorn |

---

## Project Structure

```
dockerized-ml-api/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, lifespan
│   ├── schemas.py           # Pydantic request/response models
│   ├── predictor.py         # Model loading and inference logic
│   └── dependencies.py      # Shared dependency injection
├── model/
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Evaluation metrics and threshold tuning
│   └── artifacts/           # Saved model files (gitignored)
├── notebooks/
│   ├── 01_model_development.ipynb
│   └── 02_api_testing.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # Endpoint tests with async httpx
│   └── test_predictor.py    # Unit tests for inference logic
└── scripts/
    ├── train_and_save.sh    # One-command model training
    └── run_local.sh         # Start API without Docker
```

---

## Quick Start

### Option A — Docker (recommended)
```bash
# Clone
git clone https://github.com/mtichikawa/dockerized-ml-api.git
cd dockerized-ml-api

# Train model first (writes to model/artifacts/)
python model/train.py

# Build and run
docker-compose up --build

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.1, 0.9]}'
```

### Option B — Local (no Docker)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health + model status |
| GET | `/model/info` | Model metadata, version, threshold |
| POST | `/predict` | Single observation prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000 rows) |
| GET | `/docs` | Auto-generated Swagger UI |

### Example: Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.1, 0.9],
    "observation_id": "sensor_42_t1234"
  }'
```

**Response:**
```json
{
  "observation_id": "sensor_42_t1234",
  "is_anomaly": false,
  "anomaly_score": -0.127,
  "confidence": 0.873,
  "model_version": "1.0.0",
  "inference_ms": 2.4
}
```

### Example: Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [
      {"features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.1, 0.9]},
      {"features": [8.1, -5.3, 7.2, 9.0, -8.4, 6.1, 8.8, -7.2]}
    ]
  }'
```

---

## Model Details

The ensemble combines three anomaly detectors:

| Model | Algorithm | Role |
|---|---|---|
| Isolation Forest | Random partitioning | Global outlier detection |
| Local Outlier Factor | k-NN density comparison | Local density anomalies |
| Z-score filter | Statistical | Clear distributional outliers |

**Voting:** A sample is flagged as anomalous if ≥2 of 3 detectors agree.  
**Training data:** 10,000 synthetic sensor readings with 5% injected anomalies.  
**Evaluation:** F1=0.91, Precision=0.89, Recall=0.93 on held-out test set.

---

## Docker Architecture

```
┌─────────────────────────────────────────┐
│  docker-compose stack                   │
│                                         │
│  ┌──────────────┐   ┌────────────────┐  │
│  │  api service │   │  redis service │  │
│  │  port 8000   │──▶│  port 6379     │  │
│  │  FastAPI +   │   │  result cache  │  │
│  │  uvicorn     │   └────────────────┘  │
│  └──────────────┘                       │
└─────────────────────────────────────────┘
```

---

## Interview Notes

**Why FastAPI over Flask?**
FastAPI gives you automatic OpenAPI docs, async support, and Pydantic validation with near-zero boilerplate. For an ML API serving potentially concurrent requests, async matters. Flask would work but requires more manual wiring.

**Why Docker?**
Reproducibility. "Works on my machine" is not a deployment strategy. The Docker image captures the exact Python version, dependencies, and model artifact. Anyone with Docker can run this identically.

**Why joblib instead of pickle?**
joblib is optimized for large numpy arrays — sklearn models store fitted parameters as arrays, so joblib serializes them ~10x faster and produces smaller files than pickle.

**What's the Redis layer for?**
Caching predictions for identical inputs avoids redundant model inference. In production with high-frequency sensor data, many readings repeat, so caching can cut inference load significantly.
