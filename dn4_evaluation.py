"""
dn4_evaluation.py

Reproduces the two evaluation protocols reported in the NeuroMIL paper for
dn4_score (binary neuropathic-pain classification: DN4 >= 4), across all 7
modality combinations (video, audio, text, and every union of the three):

  - a fixed patient-level 80/20 train/test split (paper Table I)
  - a patient-level stratified 5-fold cross-validation (paper Table II)

Both protocols are run through training.train (run_fixed_split /
run_cross_validation) -- the same code path used for a normal training run,
with early stopping, checkpointing, and the anti-leakage disjointness
assertion already built in. Videos are read from wherever
configs/config.yaml -> paths.video_dir points, via the standard dataset
manifest; no separate physical train/test folders are created.

Model configuration: Jina backend (Jina-CLIP-v2 video, Whisper large-v3
audio, Multilingual-E5-Large text -- see embeddings/jina_embeddings.py),
dropout 0.2 on fusion/mil_attention/classifier, matching Section IV-D of
the paper. Run embeddings/extraction_pipeline.py first to populate
features/saved_embeddings/jina/.

Usage: python dn4_evaluation.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.full_grid import LABEL_CONFIGS, METRIC_KEYS, _backend_config, _label_config
from training.train import run_cross_validation, run_fixed_split
from utils.metrics import summarize_across_folds

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)

jina_backend_cfg = _backend_config(base_config, "jina")
FEATURES_DIR = jina_backend_cfg["paths"]["features_dir"]
embedding_dims = json.load(open(f"{FEATURES_DIR}/embedding_dims.json"))

MODALITY_COMBOS = [
    ("video",), ("audio",), ("text",),
    ("video", "audio"), ("video", "text"), ("audio", "text"),
    ("video", "audio", "text"),
]


def build_cfg(combo: tuple, protocol: str) -> dict:
    name = f"dn4_score__{'+'.join(combo)}__{protocol}"
    cfg = _label_config(jina_backend_cfg, name, LABEL_CONFIGS["dn4_score"])
    cfg["model"] = copy.deepcopy(base_config["model"])
    cfg["model"]["fusion"]["input_dims"] = {m: embedding_dims[m] for m in combo}
    cfg["model"]["fusion"]["dropout"] = 0.2
    cfg["model"]["mil_attention"]["dropout"] = 0.2
    cfg["model"]["classifier"]["dropout"] = 0.2
    return cfg


def run_protocol(protocol: str) -> pd.DataFrame:
    rows = []
    for combo in MODALITY_COMBOS:
        name = "+".join(combo)
        cfg = build_cfg(combo, protocol)
        print(f"=== [{protocol}] modalities={name} ===", flush=True)

        if protocol == "80_20":
            fold_metrics = [run_fixed_split(cfg, test_size=0.2)]
        else:
            fold_metrics = run_cross_validation(cfg)

        results_dir = Path(cfg["paths"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2, default=str)

        summary = summarize_across_folds(fold_metrics)
        row = {"modalities": name}
        for k in METRIC_KEYS:
            row[k] = summary.get(k, {}).get("mean")
        rows.append(row)
        print(f"    accuracy={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}", flush=True)

    return pd.DataFrame(rows)


def main():
    Path("results").mkdir(exist_ok=True)

    df_80_20 = run_protocol("80_20")
    df_80_20.to_csv("results/dn4_80_20.csv", index=False)
    print("\nSaved to results/dn4_80_20.csv")
    print(df_80_20.to_string(index=False))

    df_5fold = run_protocol("5fold")
    df_5fold.to_csv("results/dn4_5fold.csv", index=False)
    print("\nSaved to results/dn4_5fold.csv")
    print(df_5fold.to_string(index=False))


if __name__ == "__main__":
    main()
