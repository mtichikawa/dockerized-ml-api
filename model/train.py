"""
model/train.py — Train and serialize the anomaly detection ensemble.

Generates synthetic sensor data with injected anomalies, trains three
detectors, evaluates on a held-out test set, and saves the bundle to
model/artifacts/ensemble_model.joblib.

Usage:
    python model/train.py
    python model/train.py --contamination 0.07 --n-samples 15000
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH    = ARTIFACTS_DIR / "ensemble_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"

N_FEATURES    = 8
RANDOM_STATE  = 42


def generate_data(n_samples: int = 10000, contamination: float = 0.05,
                  random_state: int = RANDOM_STATE):
    """
    Generate synthetic 8-feature sensor data with injected anomalies.

    Normal data: multivariate Gaussian with mild correlations.
    Anomalies: mixture of extreme-value, correlation-break, and spike anomalies.
    """
    rng = np.random.default_rng(random_state)

    n_normal   = int(n_samples * (1 - contamination))
    n_anomaly  = n_samples - n_normal

    # Normal: correlated sensor readings
    cov = np.eye(N_FEATURES) * 0.8
    for i in range(N_FEATURES - 1):
        cov[i, i+1] = cov[i+1, i] = 0.3  # mild adjacent correlation
    X_normal = rng.multivariate_normal(mean=np.zeros(N_FEATURES), cov=cov, size=n_normal)

    # Anomalies: three types
    n_type1 = n_anomaly // 3
    n_type2 = n_anomaly // 3
    n_type3 = n_anomaly - n_type1 - n_type2

    # Type 1: extreme values (sensor spike)
    X_a1 = rng.uniform(low=4, high=9, size=(n_type1, N_FEATURES))
    X_a1 *= rng.choice([-1, 1], size=X_a1.shape)

    # Type 2: break correlation (one feature flips sign)
    X_a2 = rng.multivariate_normal(mean=np.zeros(N_FEATURES), cov=cov, size=n_type2)
    flip_col = rng.integers(0, N_FEATURES, size=n_type2)
    for i, col in enumerate(flip_col):
        X_a2[i, col] *= -5

    # Type 3: subtle drift (cluster far from origin)
    drift = rng.uniform(3, 5, size=N_FEATURES)
    X_a3 = rng.multivariate_normal(mean=drift, cov=np.eye(N_FEATURES) * 0.5, size=n_type3)

    X = np.vstack([X_normal, X_a1, X_a2, X_a3])
    y = np.array([0]*n_normal + [1]*n_anomaly)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_models(X_train: np.ndarray, contamination: float):
    """Train all three detectors."""
    log.info("Training Isolation Forest...")
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train)

    log.info("Training Local Outlier Factor...")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,   # novelty=True allows predict() on new data
        n_jobs=-1,
    )
    lof.fit(X_train)

    log.info("Fitting StandardScaler for Z-score detector...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    return {"isolation_forest": iforest, "lof": lof, "scaler": scaler}


def ensemble_predict(models: dict, X: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """Apply majority voting ensemble: anomaly if ≥2 of 3 detectors agree."""
    # Isolation Forest: -1 = anomaly, 1 = normal → convert to 0/1
    if_preds  = (models["isolation_forest"].predict(X) == -1).astype(int)

    # LOF
    lof_preds = (models["lof"].predict(X) == -1).astype(int)

    # Z-score
    X_scaled   = models["scaler"].transform(X)
    z_preds    = (np.abs(X_scaled).max(axis=1) > z_threshold).astype(int)

    votes = if_preds + lof_preds + z_preds
    return (votes >= 2).astype(int)


def evaluate(models: dict, X_test: np.ndarray, y_test: np.ndarray,
             z_threshold: float = 3.0) -> dict:
    """Compute classification metrics on the test set."""
    y_pred = ensemble_predict(models, X_test, z_threshold)

    report = classification_report(y_test, y_pred, target_names=["normal", "anomaly"],
                                   output_dict=True)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    log.info("\n" + classification_report(y_test, y_pred,
             target_names=["normal", "anomaly"]))
    log.info(f"F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

    return {"f1": f1, "precision": prec, "recall": rec, "report": report}


def main(n_samples: int = 10000, contamination: float = 0.05):
    log.info(f"Generating data: n={n_samples}, contamination={contamination}")
    X, y = generate_data(n_samples=n_samples, contamination=contamination)

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    log.info(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    log.info(f"Train anomaly rate: {y_train.mean():.3f} | Test anomaly rate: {y_test.mean():.3f}")

    models = train_models(X_train, contamination)
    metrics = evaluate(models, X_test, y_test)

    # Save bundle
    bundle = {
        "model": models,
        "metadata": {
            "contamination":  contamination,
            "z_threshold":    3.0,
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_features":     N_FEATURES,
            "f1":             round(metrics["f1"], 4),
            "precision":      round(metrics["precision"], 4),
            "recall":         round(metrics["recall"], 4),
            "anomaly_types":  ["extreme_spike", "correlation_break", "cluster_drift"],
        }
    }

    joblib.dump(bundle, MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")

    # Save metadata as JSON for easy inspection
    meta_out = {k: v for k, v in bundle["metadata"].items() if k != "report"}
    with open(METADATA_PATH, "w") as f:
        json.dump(meta_out, f, indent=2)
    log.info(f"Metadata saved to {METADATA_PATH}")

    print(f"\n✓ Model trained and saved.")
    print(f"  F1={metrics['f1']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}")
    print(f"  Path: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",     type=int,   default=10000)
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()
    main(n_samples=args.n_samples, contamination=args.contamination)

# generate_data(): synthetic 8-feature sensor data, 3 anomaly types

# IsolationForest + LOF training with contamination parameter

# ensemble_predict(): majority vote across 3 detectors
