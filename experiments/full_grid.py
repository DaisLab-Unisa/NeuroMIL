"""
experiments/full_grid.py

Runs the full (embedding_backend x label_config) grid: for every locally
runnable embedding backend and every clinical label configuration, extracts
embeddings (if not already cached) and trains a stratified 5-fold CV MIL
classifier, then reports precision/recall/F1 (micro/macro/weighted) plus
accuracy, aggregated across folds.

Embeddings are cached per-backend (label-independent), so extraction only
runs once per backend even though multiple label configs reuse it.

Usage
-----
    python -m experiments.full_grid --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from embeddings.extraction_pipeline import run_extraction
from training.train import run_cross_validation
from utils.metrics import summarize_across_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Backends runnable with local resources in this environment. 'google' is
# excluded: configs/config.yaml has a placeholder project_id and no GCP
# credentials/data-processing agreement is configured here.
BACKENDS = ["jina", "qwen3vl"]

# DN4 (Douleur Neuropathique 4) binary neuropathic-pain classification: the
# only clinical target reported in the NeuroMIL paper. Cutoff >=4/10 on the
# standard DN4 score is the clinical threshold for neuropathic pain.
LABEL_CONFIGS: Dict[str, Dict] = {
    "dn4_score": {
        "source_column": "dn4_score",
        "task_type": "binary",
        "missing_value_strategy": "drop",
        "classes": {
            "no_neuropathic": {"min": 0, "max": 3},
            "neuropathic": {"min": 4, "max": 10},
        },
    },
}

METRIC_KEYS = [
    "accuracy", "balanced_accuracy",
    "precision_micro", "recall_micro", "f1_micro",
    "precision_macro", "recall_macro", "macro_f1",
    "precision_weighted", "recall_weighted", "weighted_f1",
]


def _backend_config(base_config: Dict, backend: str) -> Dict:
    cfg = copy.deepcopy(base_config)
    cfg["embedding"]["backend"] = backend
    cfg["paths"]["features_dir"] = str(Path(base_config["paths"]["features_dir"]) / backend)
    return cfg


def _label_config(backend_config: Dict, label_name: str, label_spec: Dict) -> Dict:
    cfg = copy.deepcopy(backend_config)
    cfg["label"] = label_spec
    backend = cfg["embedding"]["backend"]
    run_name = f"{backend}_{label_name}"
    cfg["paths"]["checkpoint_dir"] = str(Path(cfg["paths"]["checkpoint_dir"]) / run_name)
    cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / run_name)
    return cfg


def run_full_grid(base_config: Dict) -> pd.DataFrame:
    rows = []
    for backend in BACKENDS:
        backend_cfg = _backend_config(base_config, backend)
        logger.info("=" * 70)
        logger.info("Extracting embeddings for backend=%s", backend)
        logger.info("=" * 70)
        try:
            run_extraction(backend_cfg)
        except Exception as e:
            logger.exception("Extraction failed for backend %s -- skipping all its label configs", backend)
            continue

        for label_name, label_spec in LABEL_CONFIGS.items():
            run_cfg = _label_config(backend_cfg, label_name, label_spec)
            logger.info("-" * 70)
            logger.info("Training backend=%s label=%s", backend, label_name)
            logger.info("-" * 70)
            try:
                fold_metrics = run_cross_validation(run_cfg)
            except Exception as e:
                logger.exception("Training failed for backend=%s label=%s -- skipping", backend, label_name)
                continue

            results_dir = Path(run_cfg["paths"]["results_dir"])
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / "cv_fold_metrics.json", "w") as f:
                json.dump(fold_metrics, f, indent=2, default=str)

            summary = summarize_across_folds(fold_metrics)
            row = {"embedding_backend": backend, "label": label_name, "n_folds": len(fold_metrics)}
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = summary.get(k, {}).get("mean")
                row[f"{k}_std"] = summary.get(k, {}).get("std")
            rows.append(row)

            comparison_df = pd.DataFrame(rows)
            results_dir_top = Path(base_config["paths"]["results_dir"])
            results_dir_top.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(results_dir_top / "full_grid_comparison.csv", index=False)
            logger.info("Updated running comparison -> %s", results_dir_top / "full_grid_comparison.csv")

    comparison_df = pd.DataFrame(rows)
    results_dir_top = Path(base_config["paths"]["results_dir"])
    results_dir_top.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir_top / "full_grid_comparison.csv", index=False)
    logger.info("Full grid comparison saved to %s", results_dir_top / "full_grid_comparison.csv")
    return comparison_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = run_full_grid(config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
