"""
Metric computation functions for knee OA classification evaluation.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(labels, preds, probs):
    """
    Compute classification metrics.

    Args:
        labels: Ground truth binary labels (0=KL01, 1=KL234)
        preds: Predicted binary labels
        probs: Predicted probabilities for the positive class

    Returns:
        Dictionary of metric name -> value
    """
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    probs = np.nan_to_num(probs, nan=0.5)
    probs = np.clip(probs, 0, 1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "auroc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        "f1": f1_score(labels, preds, zero_division=0),
        "ppv": precision_score(labels, preds, zero_division=0),
        "sensitivity": recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    # Per-class accuracy
    kl01_mask = labels == 0
    kl234_mask = labels == 1
    metrics["kl01_accuracy"] = (
        float((preds[kl01_mask] == 0).mean()) if kl01_mask.sum() > 0 else 0.0
    )
    metrics["kl234_accuracy"] = (
        float((preds[kl234_mask] == 1).mean()) if kl234_mask.sum() > 0 else 0.0
    )

    return {k: float(v) if not isinstance(v, int) else v for k, v in metrics.items()}


def compute_per_grade_metrics(labels, preds, grades):
    """
    Compute accuracy for each individual KL grade (0-4).

    Args:
        labels: Ground truth binary labels
        preds: Predicted binary labels
        grades: KL grade (0-4) for each sample

    Returns:
        Dictionary of grade -> {accuracy, n}
    """
    labels = np.array(labels)
    preds = np.array(preds)
    grades = np.array(grades)

    grade_metrics = {}
    for g in range(5):
        mask = grades == g
        if mask.sum() == 0:
            grade_metrics[g] = {"accuracy": 0.0, "n": 0}
            continue
        expected = 0 if g <= 1 else 1
        acc = float((preds[mask] == expected).mean())
        grade_metrics[g] = {"accuracy": acc, "n": int(mask.sum())}

    return grade_metrics
