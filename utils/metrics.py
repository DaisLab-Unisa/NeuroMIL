"""
utils/metrics.py

Task-adaptive evaluation metrics: automatically switches between binary
and multiclass metric sets based on `num_classes`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: List[str]
) -> Dict:
    """
    y_true : [N] int labels
    y_pred : [N] int predicted labels
    y_proba: [N, C] predicted class probabilities
    """
    num_classes = len(class_names)
    metrics: Dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class precision/recall/f1, flattened as precision_class_<name> etc.
    # so summarize_across_folds's generic scalar-averaging picks them up
    # exactly like the macro/micro/weighted keys above.
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    for i, cname in enumerate(class_names):
        metrics[f"precision_class_{cname}"] = float(per_class_p[i])
        metrics[f"recall_class_{cname}"] = float(per_class_r[i])
        metrics[f"f1_class_{cname}"] = float(per_class_f1[i])

    try:
        if num_classes == 2:
            metrics["auroc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["auroc_ovr_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        # e.g. a class missing entirely from a small val fold
        metrics["auroc_error"] = str(e)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_names"] = class_names
    metrics["n_samples"] = int(len(y_true))

    return metrics


def summarize_across_folds(fold_metrics: List[Dict]) -> Dict:
    """Aggregate mean +/- std for each scalar metric across CV folds."""
    keys = [
        k for k in fold_metrics[0].keys()
        if isinstance(fold_metrics[0][k], (int, float)) and k != "n_samples"
    ]
    summary = {}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics if k in fm]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary
