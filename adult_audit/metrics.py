"""Aggregate and descriptive subgroup measures."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)

from .data import AUDIT_GROUPS


def binary_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "positive_prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def group_metrics(frame, probabilities, threshold):
    """Rates are descriptive; no single metric establishes fairness."""
    probabilities = np.asarray(probabilities)
    output = {}
    for attribute in AUDIT_GROUPS:
        rows = []
        for group, subset in frame.groupby(attribute, dropna=False, sort=True):
            indices = frame.index.get_indexer(subset.index)
            y = subset["target"].to_numpy(dtype=int)
            predicted = probabilities[indices] >= threshold
            tp = int(((y == 1) & predicted).sum())
            fp = int(((y == 0) & predicted).sum())
            positives = int((y == 1).sum())
            negatives = int((y == 0).sum())
            rows.append({
                "group": str(group), "n": int(len(y)), "positive_prevalence": float(y.mean()),
                "selection_rate": float(predicted.mean()),
                "true_positive_rate": tp / positives if positives else None,
                "false_positive_rate": fp / negatives if negatives else None,
                "precision": tp / (tp + fp) if tp + fp else None,
            })
        output[attribute] = rows
    return output

