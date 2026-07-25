"""
training/evaluate.py

Loads a trained checkpoint, runs inference on a held-out split (or the
full dataset for a final report), and produces:
    results/metrics.json
    results/predictions.csv   (record_id, true_label, predicted_label, class probabilities)
    results/plots/*.png       (confusion matrix, ROC curves, attention timelines)

Usage
-----
    python -m training.evaluate --config configs/config.yaml \
        --checkpoint checkpoint/fold_1/best_model.pt \
        --patient_ids 12,45,88   # optional subset, default = all in manifest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.ruggi_dataset import RuggiMILDataset, build_dataset_manifest, mil_collate_fn
from models.classifier import PainMILModel
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics
from utils.visualization import (
    plot_attention_timeline,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_checkpoint(
    config: Dict, checkpoint_path: str, patient_ids: Optional[List[int]] = None
) -> Dict:
    device = config["embedding"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    features_dir = config["paths"]["features_dir"]
    results_dir = Path(config["paths"]["results_dir"])
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest, processor = build_dataset_manifest(config)
    if patient_ids:
        manifest = manifest[manifest["record_id"].isin(patient_ids)]

    labels = dict(zip(manifest["record_id"], manifest["label"]))
    ids = manifest["record_id"].tolist()
    class_names = processor.class_names

    ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_class_names = ckpt_raw.get("extra", {}).get("class_names", class_names)

    model = PainMILModel(config, num_classes=len(ckpt_class_names)).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device, restore_rng=False)
    model.eval()

    ds = RuggiMILDataset(ids, labels, features_dir)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False,
                         collate_fn=mil_collate_fn)

    all_true, all_pred, all_proba, all_ids, all_attn, all_masks = [], [], [], [], [], []

    for batch in loader:
        record_ids_batch = batch["record_id"]
        batch_t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        logits, attn = model(batch_t, return_attention=True)
        proba = F.softmax(logits, dim=-1).cpu().numpy()
        pred = proba.argmax(axis=1)

        all_true.extend(batch["label"].numpy().tolist())
        all_pred.extend(pred.tolist())
        all_proba.append(proba)
        all_ids.extend(record_ids_batch)
        all_attn.append(attn.cpu().numpy())
        all_masks.append(batch["mask"].numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_proba = np.concatenate(all_proba, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_proba, ckpt_class_names)

    # ---- predictions.csv ----
    pred_rows = []
    for i, rid in enumerate(all_ids):
        row = {
            "record_id": rid,
            "true_label": ckpt_class_names[y_true[i]],
            "predicted_label": ckpt_class_names[y_pred[i]],
        }
        for c, cname in enumerate(ckpt_class_names):
            row[f"proba_{cname}"] = float(y_proba[i, c])
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    # ---- metrics.json ----
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # ---- plots ----
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), ckpt_class_names,
                           str(plots_dir / "confusion_matrix.png"))
    plot_roc_curves(y_true, y_proba, ckpt_class_names, str(plots_dir / "roc_curves.png"))
    plot_class_distribution(processor.class_distribution(manifest["label"]),
                             str(plots_dir / "class_distribution.png"))

    history = ckpt_raw.get("extra", {}).get("history")
    if history:
        plot_training_curves(history, str(plots_dir / "training_curves.png"))

    # ---- per-patient attention timelines ----
    attn_concat = np.concatenate(all_attn, axis=0)
    mask_concat = np.concatenate(all_masks, axis=0)
    for i, rid in enumerate(all_ids):
        meta_path = Path(features_dir) / str(rid) / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n = mask_concat[i].sum()
        plot_attention_timeline(
            record_id=rid,
            clip_windows=meta["clip_windows"][:n],
            attn_weights=attn_concat[i][:n],
            out_path=str(plots_dir / "attention" / f"patient_{rid}.png"),
            true_label=ckpt_class_names[y_true[i]],
            pred_label=ckpt_class_names[y_pred[i]],
        )

    logger.info("Evaluation complete. Macro-F1=%.4f Balanced-Acc=%.4f",
                metrics["macro_f1"], metrics["balanced_accuracy"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patient_ids", type=str, default=None,
                         help="Comma-separated record_ids to restrict evaluation to")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    patient_ids = None
    if args.patient_ids:
        patient_ids = [int(x) for x in args.patient_ids.split(",")]

    evaluate_checkpoint(config, args.checkpoint, patient_ids)


if __name__ == "__main__":
    main()
