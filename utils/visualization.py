"""
utils/visualization.py

All plotting utilities (requirement 17):
  1. training curves (loss / val loss / val F1)
  2. confusion matrix
  3. ROC curves
  4. class distribution
  5. attention-over-time visualization per test patient
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_training_curves(history: Dict[str, List[float]], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(history["val_macro_f1"], label="val_macro_f1", color="darkgreen")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("macro F1"); axes[1].legend()
    axes[1].set_title("Validation Macro-F1")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str, normalize: bool = True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], out_path: str):
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
            ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve(s) - One-vs-Rest"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(class_counts: Dict[str, int], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(class_counts.keys())
    counts = list(class_counts.values())
    ax.bar(names, counts, color="steelblue")
    ax.set_ylabel("Number of patients"); ax.set_title("Class Distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha="center")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention_timeline(
    record_id: int,
    clip_windows: List[List[float]],
    attn_weights: np.ndarray,
    out_path: str,
    true_label: Optional[str] = None,
    pred_label: Optional[str] = None,
):
    """For a single test patient: timeline of clips with attention weight
    per clip, to visualize which parts of the interview drove the
    prediction (requirement 17.5)."""
    starts = [w[0] for w in clip_windows]
    widths = [w[1] - w[0] for w in clip_windows]
    n = len(starts)
    weights = attn_weights[:n]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), 2.5))
    colors = plt.cm.Reds(weights / (weights.max() + 1e-8))
    ax.bar(starts, height=1.0, width=widths, align="edge", color=colors, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Interview time (s)")
    ax.set_yticks([])
    title = f"Patient {record_id} - attention over interview timeline"
    if true_label is not None:
        title += f" | true={true_label}"
    if pred_label is not None:
        title += f" pred={pred_label}"
    ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=0, vmax=weights.max()))
    fig.colorbar(sm, ax=ax, label="attention weight", fraction=0.03, pad=0.02)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
