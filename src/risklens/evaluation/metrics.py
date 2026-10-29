"""Shared classification and probability metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)

FIXED_THRESHOLD = 0.5


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Fixed-threshold diagnostics plus threshold-free ranking and probability scores."""
    labels = (probabilities >= FIXED_THRESHOLD).astype(int)
    tn, fp, fn, tp = (
        int(x) for x in confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    )
    two_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 8)
        if two_classes
        else None,
        "average_precision": round(
            float(average_precision_score(y_true, probabilities)), 8
        )
        if two_classes
        else None,
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 8),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1])), 8),
        "threshold": FIXED_THRESHOLD,
        "precision_at_threshold": round(
            float(precision_score(y_true, labels, zero_division=0)), 8
        ),
        "recall_at_threshold": round(
            float(recall_score(y_true, labels, zero_division=0)), 8
        ),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "warning": (
            "ROC-AUC and AP are undefined for a one-class validation sample"
            if not two_classes
            else None
        ),
    }


def _pooled_probability_scores(frame: pd.DataFrame, col: str) -> dict:
    """Compute proper probability scores on the same pooled fold observations."""
    y = frame["actual_downside_event"].to_numpy(dtype=int)
    p = frame[col].to_numpy(dtype=float)
    clipped = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return {
        "brier_score": round(float(np.mean((p - y) ** 2)), 8),
        "log_loss": round(
            float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))), 8
        ),
    }
