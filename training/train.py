"""
training/train.py

Trains the PainMILModel with patient-level stratified K-fold cross
validation (no patient leakage between folds), weighted CE or focal loss,
AdamW + LR scheduler, early stopping, and full checkpoint/resume support.

Usage
-----
    python -m training.train --config configs/config.yaml
    python -m training.train --config configs/config.yaml --resume checkpoint/fold_1/last_checkpoint.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from datasets.ruggi_dataset import RuggiMILDataset, build_dataset_manifest, mil_collate_fn
from models.classifier import PainMILModel
from utils.checkpoint import load_checkpoint, save_checkpoint, set_seed
from utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =========================================================================
# Losses
# =========================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, target, weight=self.weight, reduction="none")
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def build_loss_fn(loss_name: str, class_weights: torch.Tensor, focal_gamma: float, device: str):
    class_weights = class_weights.to(device)
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0  # avoid div-by-zero for absent classes in a fold
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# =========================================================================
# Train / eval loops
# =========================================================================

def apply_modality_dropout(batch: Dict, modalities: List[str], p: float) -> None:
    """Training-time-only regularizer: independently zero out each sample's
    embedding for a given modality with probability `p`. Forces the fusion
    to not over-rely on a single modality instead of dropping any modality
    from the architecture -- inference/eval always sees all modalities."""
    if p <= 0:
        return
    batch_size = batch[f"{modalities[0]}_emb"].shape[0]
    for m in modalities:
        drop = torch.rand(batch_size, device=batch[f"{m}_emb"].device) < p
        if drop.any():
            batch[f"{m}_emb"][drop] = 0.0


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip: Optional[float] = None,
              modalities: Optional[List[str]] = None, modality_dropout_p: float = 0.0):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    all_true, all_pred, all_proba = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if is_train and modalities and modality_dropout_p > 0:
                apply_modality_dropout(batch, modalities, modality_dropout_p)

            logits, _ = model(batch)
            loss = loss_fn(logits, batch["label"])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            proba = F.softmax(logits, dim=-1).detach().cpu().numpy()
            pred = proba.argmax(axis=1)
            all_true.extend(batch["label"].cpu().numpy().tolist())
            all_pred.extend(pred.tolist())
            all_proba.append(proba)

    avg_loss = total_loss / max(n_batches, 1)
    all_proba = np.concatenate(all_proba, axis=0) if all_proba else np.zeros((0, 1))
    return avg_loss, np.array(all_true), np.array(all_pred), all_proba


def train_fold(
    fold_idx: int,
    train_ids: List[int],
    val_ids: List[int],
    labels: Dict[int, int],
    class_names: List[str],
    config: Dict,
    resume_path: Optional[str] = None,
) -> Dict:
    device = config["embedding"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["training"]
    features_dir = config["paths"]["features_dir"]

    set_seed(train_cfg["seed"] + fold_idx)

    # Which modalities to load is driven by the fusion config, so ablations
    # (e.g. text-only) can drop modalities without the dataset/collate still
    # trying to read cached files for modalities the model won't consume.
    modalities = tuple(config["model"]["fusion"]["input_dims"].keys())
    collate = lambda batch: mil_collate_fn(batch, modalities=modalities)

    train_ds = RuggiMILDataset(train_ids, labels, features_dir, modalities=modalities)
    val_ds = RuggiMILDataset(val_ids, labels, features_dir, modalities=modalities)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                               collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                             collate_fn=collate)

    num_classes = len(class_names)
    model = PainMILModel(config, num_classes=num_classes).to(device)

    train_labels_arr = np.array([labels[i] for i in train_ids])
    class_weights = compute_class_weights(train_labels_arr, num_classes)
    loss_fn = build_loss_fn(train_cfg["loss"], class_weights, train_cfg.get("focal_gamma", 2.0), device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = None
    if train_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    elif train_cfg["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)

    start_epoch = 0
    best_metric = -np.inf
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    ckpt_dir = Path(config["paths"]["checkpoint_dir"]) / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]
        history = ckpt["extra"].get("history", history)
        logger.info("Fold %d: resumed from epoch %d (best_metric=%.4f)", fold_idx, start_epoch, best_metric)

    modality_dropout_p = train_cfg.get("modality_dropout", 0.0)
    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_loss, _, _, _ = run_epoch(
            model, train_loader, loss_fn, device, optimizer, train_cfg.get("grad_clip_norm"),
            modalities=list(modalities), modality_dropout_p=modality_dropout_p,
        )
        val_loss, y_true, y_pred, y_proba = run_epoch(model, val_loader, loss_fn, device)

        val_metrics = compute_metrics(y_true, y_pred, y_proba, class_names) if len(y_true) else {"macro_f1": 0.0}
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)

        if scheduler is not None:
            if train_cfg["scheduler"] == "plateau":
                scheduler.step(val_f1)
            else:
                scheduler.step()

        logger.info(
            "Fold %d | Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f",
            fold_idx, epoch + 1, train_cfg["epochs"], train_loss, val_loss, val_f1,
        )

        save_checkpoint(
            str(ckpt_dir / "last_checkpoint.pt"), model, optimizer, scheduler,
            epoch, best_metric, extra={"history": history},
        )

        improved = val_f1 > best_metric
        if improved:
            best_metric = val_f1
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                str(ckpt_dir / "best_model.pt"), model, optimizer, scheduler,
                epoch, best_metric, extra={"history": history, "class_names": class_names},
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                logger.info("Fold %d: early stopping at epoch %d", fold_idx, epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, y_true, y_pred, y_proba = run_epoch(model, val_loader, loss_fn, device)
    final_metrics = compute_metrics(y_true, y_pred, y_proba, class_names)
    final_metrics["fold"] = fold_idx
    final_metrics["history"] = history

    return final_metrics


def apply_cached_embedding_dims(config: Dict) -> Dict:
    """Overrides config['model']['fusion']['input_dims'] with the backend's
    real output dims (recorded by extraction_pipeline.run_extraction), since
    the static config value is just a placeholder default and backends like
    qwen3vl produce a different hidden size than e.g. jina. Only corrects
    dims for modalities already listed in input_dims, so a modality-ablation
    config (e.g. text-only) keeps its reduced modality set."""
    dims_path = Path(config["paths"]["features_dir"]) / "embedding_dims.json"
    if not dims_path.exists():
        return config
    with open(dims_path, "r", encoding="utf-8") as f:
        dims = json.load(f)
    config = copy.deepcopy(config)
    config["model"]["fusion"]["input_dims"] = {
        m: dims[m] for m in config["model"]["fusion"]["input_dims"]
    }
    logger.info("Applied cached embedding dims from %s: %s", dims_path, config["model"]["fusion"]["input_dims"])
    return config


def run_cross_validation(config: Dict, resume_path: Optional[str] = None) -> List[Dict]:
    config = apply_cached_embedding_dims(config)
    manifest, processor = build_dataset_manifest(config)
    labels = dict(zip(manifest["record_id"], manifest["label"]))
    record_ids = manifest["record_id"].tolist()
    y = manifest["label"].to_numpy()

    cv_cfg = config["training"]["cross_validation"]
    n_folds = cv_cfg["n_folds"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["training"]["seed"])

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(record_ids, y)):
        train_ids = [record_ids[i] for i in train_idx]
        val_ids = [record_ids[i] for i in val_idx]

        # Sanity check: no patient leakage between train/val
        assert set(train_ids).isdisjoint(set(val_ids)), "Patient leakage detected between folds!"

        logger.info("=== Fold %d/%d: %d train / %d val patients ===",
                    fold_idx + 1, n_folds, len(train_ids), len(val_ids))

        resume_this_fold = resume_path if resume_path and f"fold_{fold_idx}" in resume_path else None
        metrics = train_fold(
            fold_idx, train_ids, val_ids, labels, processor.class_names, config,
            resume_path=resume_this_fold,
        )
        fold_metrics.append(metrics)

    return fold_metrics


def run_fixed_split(config: Dict, test_size: float = 0.2, resume_path: Optional[str] = None) -> Dict:
    """Single patient-level stratified train/test split (e.g. 80/20), trained
    once and early-stopped/evaluated on the held-out test patients. Videos
    are read from wherever config['paths']['video_dir'] points (via the
    standard dataset manifest) -- no separate physical train/test folders
    are created or required."""
    config = apply_cached_embedding_dims(config)
    manifest, processor = build_dataset_manifest(config)
    labels = dict(zip(manifest["record_id"], manifest["label"]))
    record_ids = manifest["record_id"].tolist()
    y = manifest["label"].to_numpy()

    train_ids, test_ids = train_test_split(
        record_ids, test_size=test_size, random_state=config["training"]["seed"], stratify=y
    )

    # Sanity check: no patient leakage between train/test
    assert set(train_ids).isdisjoint(set(test_ids)), "Patient leakage detected in fixed split!"

    logger.info("=== Fixed %d/%d split: %d train / %d test patients ===",
                round((1 - test_size) * 100), round(test_size * 100), len(train_ids), len(test_ids))

    metrics = train_fold(0, train_ids, test_ids, labels, processor.class_names, config, resume_path=resume_path)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--protocol", type=str, choices=["5fold", "80_20"], default="5fold",
                         help="Evaluation protocol: patient-level stratified 5-fold CV, "
                              "or a single fixed 80/20 patient-level train/test split")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a fold's checkpoint (e.g. checkpoint/fold_1/last_checkpoint.pt)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    if args.protocol == "5fold":
        fold_metrics = run_cross_validation(config, resume_path=args.resume)
        out_name = "cv_fold_metrics.json"
    else:
        fold_metrics = [run_fixed_split(config, test_size=0.2, resume_path=args.resume)]
        out_name = "fixed_split_metrics.json"

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / out_name, "w") as f:
        json.dump(fold_metrics, f, indent=2, default=str)

    logger.info("%s complete. Results saved to %s",
                "Cross-validation" if args.protocol == "5fold" else "Fixed split training",
                results_dir / out_name)


if __name__ == "__main__":
    main()
