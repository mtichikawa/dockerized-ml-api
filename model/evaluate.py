"""
model/evaluate.py — Post-hoc evaluation and threshold tuning.

Load the saved model bundle and evaluate it across a range of
Z-score thresholds to find the optimal operating point.

Usage:
    python model/evaluate.py
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from model.train import generate_data, ensemble_predict, N_FEATURES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "ensemble_model.joblib"
EVAL_OUT      = ARTIFACTS_DIR / "evaluation_report.json"


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["metadata"]


def threshold_sweep(models, X_test, y_test):
    """Sweep Z-score threshold from 1.5 to 5.0 and report metrics."""
    thresholds = np.arange(1.5, 5.1, 0.25)
    results = []
    for t in thresholds:
        y_pred = ensemble_predict(models, X_test, z_threshold=t)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({"threshold": round(float(t), 2), "f1": round(f1, 4)})
    return results


def plot_roc_and_pr(models, X_test, y_test):
    """Generate ROC and Precision-Recall curves using IF anomaly scores."""
    if_scores = -models["isolation_forest"].score_samples(X_test)  # higher = more anomalous

    fpr, tpr, _ = roc_curve(y_test, if_scores)
    auc = roc_auc_score(y_test, if_scores)

    prec, rec, _ = precision_recall_curve(y_test, if_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Anomaly Detection Model — Evaluation Curves\n(Isolation Forest scores)")

    axes[0].plot(fpr, tpr, color="#2C7BB6", linewidth=2.5, label=f"ROC (AUC={auc:.3f})")
    axes[0].plot([0,1],[0,1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(rec, prec, color="#1A9641", linewidth=2.5, label="Precision-Recall")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.4,
                    label=f"Baseline (prevalence={y_test.mean():.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    out = ARTIFACTS_DIR / "evaluation_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved evaluation curves to {out}")
    return auc


def main():
    if not MODEL_PATH.exists():
        log.error(f"Model not found: {MODEL_PATH}. Run model/train.py first.")
        return

    models, metadata = load_model()
    log.info(f"Loaded model metadata: {metadata}")

    # Fresh test data
    _, y_all = generate_data(n_samples=10000, contamination=metadata.get("contamination", 0.05))
    X_all, _ = generate_data(n_samples=10000, contamination=metadata.get("contamination", 0.05))
    X_test, y_test = X_all[8000:], y_all[8000:]

    log.info(f"Test set: {len(X_test)} samples, {y_test.sum()} anomalies ({y_test.mean():.2%})")

    # Threshold sweep
    sweep = threshold_sweep(models, X_test, y_test)
    best = max(sweep, key=lambda x: x["f1"])
    log.info(f"Best threshold: {best['threshold']} (F1={best['f1']})")

    # ROC/PR
    auc = plot_roc_and_pr(models, X_test, y_test)

    # Final report
    y_pred = ensemble_predict(models, X_test, z_threshold=best["threshold"])
    report = classification_report(y_test, y_pred, target_names=["normal", "anomaly"],
                                   output_dict=True)
    print("\n" + classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))

    full_report = {
        "roc_auc":         round(auc, 4),
        "best_threshold":  best["threshold"],
        "best_f1":         best["f1"],
        "threshold_sweep": sweep,
        "classification":  report,
    }
    with open(EVAL_OUT, "w") as f:
        json.dump(full_report, f, indent=2)
    log.info(f"Evaluation report saved to {EVAL_OUT}")


if __name__ == "__main__":
    main()

# threshold_sweep() and ROC/PR curve generation
