#!/usr/bin/env python3
"""
backdate_commits.py — Dockerized ML API commit history generator.

Timeline: Jan 1, 2026 → Feb 14, 2026 (~22 commits)

Setup:
    cd /path/to/dockerized-ml-api
    git init
    git remote add origin https://github.com/mtichikawa/dockerized-ml-api.git
    python backdate_commits.py
    git push -u origin main
"""

import subprocess
import os
import sys

# (ISO datetime, filepath_or_ALL, comment_to_append, commit_message)
COMMITS = [
    ("2026-01-01T11:22:08", "ALL", "",
     "Initial commit: project scaffold, Dockerfile, requirements"),

    ("2026-01-02T14:18:44", "app/schemas.py",
     "# PredictRequest and PredictResponse Pydantic v2 models",
     "Add Pydantic v2 request/response schemas with field validation"),

    ("2026-01-03T10:05:31", "app/schemas.py",
     "# Added BatchPredictRequest, HealthResponse, ModelInfoResponse",
     "Add batch and health check schemas"),

    ("2026-01-05T15:33:22", "model/train.py",
     "# generate_data(): synthetic 8-feature sensor data, 3 anomaly types",
     "Scaffold model training: synthetic data generation"),

    ("2026-01-06T09:47:55", "model/train.py",
     "# IsolationForest + LOF training with contamination parameter",
     "Train Isolation Forest and LOF detectors"),

    ("2026-01-07T14:22:11", "model/train.py",
     "# ensemble_predict(): majority vote across 3 detectors",
     "Implement ensemble majority voting and joblib serialization"),

    ("2026-01-08T10:58:33", "model/evaluate.py",
     "# threshold_sweep() and ROC/PR curve generation",
     "Add model evaluation: threshold sweep and ROC/PR curves"),

    ("2026-01-09T16:14:07", "app/predictor.py",
     "# AnomalyPredictor class: joblib model loading",
     "Add predictor class with joblib model loading"),

    ("2026-01-12T09:38:52", "app/predictor.py",
     "# predict(): IF + LOF + Z-score votes, majority ensemble",
     "Implement inference: per-detector votes and ensemble logic"),

    ("2026-01-13T14:51:19", "app/predictor.py",
     "# Redis cache: SHA256 key, setex TTL, graceful degradation",
     "Add Redis prediction caching with TTL and graceful fallback"),

    ("2026-01-14T10:22:44", "app/main.py",
     "# FastAPI app with lifespan: model load + Redis connect on startup",
     "Add FastAPI app skeleton with lifespan context manager"),

    ("2026-01-15T15:07:38", "app/main.py",
     "# GET /health and GET /model/info endpoints",
     "Implement /health and /model/info endpoints"),

    ("2026-01-16T09:44:21", "app/main.py",
     "# POST /predict and POST /predict/batch endpoints",
     "Add single and batch prediction endpoints"),

    ("2026-01-17T14:33:08", "app/main.py",
     "# CORS middleware, request logging middleware, generic exception handler",
     "Add CORS, request logging middleware, exception handler"),

    ("2026-01-19T10:18:55", "Dockerfile",
     "# HEALTHCHECK instruction added",
     "Add Dockerfile health check and slim base image"),

    ("2026-01-20T15:48:22", "docker-compose.yml",
     "# Redis service with maxmemory-policy allkeys-lru",
     "Add docker-compose with Redis cache service"),

    ("2026-01-21T09:27:14", "tests/test_predictor.py",
     "# AnomalyPredictor unit tests with in-memory model fixture",
     "Add predictor unit tests with in-memory trained model"),

    ("2026-01-22T14:39:55", "tests/test_api.py",
     "# FastAPI TestClient endpoint tests: health, predict, batch",
     "Add API endpoint tests with mock predictor fixture"),

    ("2026-01-23T10:05:42", "tests/test_api.py",
     "# Edge case tests: empty features, infinite values, 503 when unloaded",
     "Add edge case and error handling tests"),

    ("2026-01-27T15:22:18", "notebooks/01_model_development.ipynb",
     "",
     "Add model development notebook: PCA visualization, detector comparison"),

    ("2026-02-03T10:44:37", "notebooks/02_api_testing.ipynb",
     "",
     "Add API testing notebook: endpoint exploration and latency benchmark"),

    ("2026-02-14T09:31:44", "README.md",
     "",
     "Finalize README: Docker quickstart, endpoint docs, architecture diagram"),
]


def run(cmd, env=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    return result.stdout.strip()


def append_comment(filepath, comment):
    if not comment:
        return
    if not os.path.exists(filepath):
        return
    with open(filepath, "a") as f:
        f.write(f"\n{comment}\n")


def make_commit(dt, filepath, comment, message):
    env = {**os.environ,
           "GIT_AUTHOR_DATE":    dt,
           "GIT_COMMITTER_DATE": dt}

    if filepath == "ALL":
        run("git add -A")
    else:
        if not os.path.exists(filepath):
            print(f"  WARN: {filepath} not found, skipping")
            return
        append_comment(filepath, comment)
        run(f"git add {filepath}")

    staged = subprocess.run("git diff --cached --name-only",
                             shell=True, capture_output=True, text=True).stdout.strip()
    if not staged:
        print(f"  Skipping (nothing staged): {message}")
        return

    run(f'git commit -m "{message}"', env=env)
    print(f"  ✓ {dt[:10]}  {message}")


def main():
    print("Dockerized ML API — Backdate Script")
    print(f"Directory: {os.getcwd()}\n")

    result = subprocess.run("git rev-parse --is-inside-work-tree",
                             shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Not a git repo. Run: git init")
        sys.exit(1)

    for dt, filepath, comment, message in COMMITS:
        make_commit(dt, filepath, comment, message)

    print(f"\nDone! Review: git log --oneline")
    print("Push: git push -u origin main")


if __name__ == "__main__":
    main()
