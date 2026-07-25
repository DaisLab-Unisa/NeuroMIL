"""
datasets/label_processor.py

ClinicalLabelProcessor
=======================
Turns a raw clinical CSV column (selected entirely via YAML config, never
hardcoded) into classification labels for the MIL model.

Supports:
    - binary classification
    - multiclass classification
    - ordinal classification
    - range-based bucketing (e.g. NRS 0-3 / 4-6 / 7-10)
    - value-set bucketing (e.g. bpi_complete in {0,"No"} -> incomplete)

The number of output classes is inferred automatically from the YAML
`label.classes` block, so adding a new clinical variable / task never
requires touching the model code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelSpec:
    """Parsed representation of the `label` section of config.yaml."""

    source_column: str
    task_type: str  # binary | multiclass | ordinal
    missing_value_strategy: str
    classes: Dict[str, Dict[str, Any]]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LabelSpec":
        label_cfg = cfg["label"]
        return cls(
            source_column=label_cfg["source_column"],
            task_type=label_cfg.get("task_type", "multiclass"),
            missing_value_strategy=label_cfg.get("missing_value_strategy", "drop"),
            classes=label_cfg.get("classes", {}),
        )


class ClinicalLabelProcessor:
    """
    Reads the clinical CSV, extracts the configured target column, and
    converts it into integer class labels + human-readable class names.

    Usage
    -----
    >>> processor = ClinicalLabelProcessor(config)
    >>> df_labels = processor.fit_transform(clinical_df)
    >>> processor.class_names
    ['mild', 'moderate', 'severe']
    >>> processor.num_classes
    3
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spec = LabelSpec.from_config(config)
        self._class_names: List[str] = list(self.spec.classes.keys())
        self._class_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self._class_names)
        }
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def source_column(self) -> str:
        return self.spec.source_column

    def fit_transform(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        clinical_df : DataFrame containing at least `record_id` and the
            configured `source_column`.

        Returns
        -------
        DataFrame with columns [record_id, raw_value, label, class_name],
        one row per patient with a *valid* label (rows dropped/imputed
        according to `missing_value_strategy`).
        """
        if self.spec.source_column not in clinical_df.columns:
            raise KeyError(
                f"Configured label.source_column='{self.spec.source_column}' "
                f"not found in clinical CSV. Available columns: "
                f"{list(clinical_df.columns)[:20]}..."
            )

        df = clinical_df[["record_id", self.spec.source_column]].copy()
        df = df.rename(columns={self.spec.source_column: "raw_value"})

        df = self._handle_missing(df)

        if not self._class_names:
            # No explicit class config -> infer classes directly from
            # unique raw values (categorical passthrough).
            self._infer_classes_from_values(df["raw_value"])

        df["class_name"] = df["raw_value"].apply(self._bucket_value)
        df = df[df["class_name"].notna()].reset_index(drop=True)
        df["label"] = df["class_name"].map(self._class_to_idx).astype(int)

        self._fitted = True
        self._log_distribution(df)
        return df[["record_id", "raw_value", "label", "class_name"]]

    def class_distribution(self, labels: pd.Series) -> Dict[str, int]:
        counts = labels.value_counts().to_dict()
        return {self._class_names[k]: v for k, v in counts.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.spec.missing_value_strategy
        n_missing = df["raw_value"].isna().sum()
        if n_missing:
            logger.info(
                "Column '%s': %d/%d missing values -> strategy='%s'",
                self.spec.source_column, n_missing, len(df), strategy,
            )
        if strategy == "drop":
            df = df[df["raw_value"].notna()]
        elif strategy == "impute_median":
            numeric = pd.to_numeric(df["raw_value"], errors="coerce")
            median = numeric.median()
            df["raw_value"] = numeric.fillna(median)
        elif strategy == "impute_mode":
            mode = df["raw_value"].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else df["raw_value"].iloc[0]
            df["raw_value"] = df["raw_value"].fillna(fill)
        else:
            raise ValueError(f"Unknown missing_value_strategy: {strategy}")
        return df

    def _infer_classes_from_values(self, values: pd.Series) -> None:
        uniques = sorted(v for v in values.unique() if pd.notna(v))
        self._class_names = [str(v) for v in uniques]
        self._class_to_idx = {name: i for i, name in enumerate(self._class_names)}
        # Rebuild an implicit value-set class spec so `_bucket_value` works.
        self.spec.classes = {
            str(v): {"values": [v]} for v in uniques
        }
        logger.info(
            "No explicit `label.classes` given -> inferred %d classes from data: %s",
            len(self._class_names), self._class_names,
        )

    def _bucket_value(self, value: Any) -> Optional[str]:
        """Map a raw value to a class name using either a numeric [min,max]
        range or an explicit `values` set, as declared in config."""
        for class_name, spec in self.spec.classes.items():
            if "min" in spec and "max" in spec:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if spec["min"] <= numeric <= spec["max"]:
                    return class_name
            elif "values" in spec:
                allowed = spec["values"]
                if value in allowed or str(value) in [str(a) for a in allowed]:
                    return class_name
        return None

    def _log_distribution(self, df: pd.DataFrame) -> None:
        dist = df["class_name"].value_counts().to_dict()
        logger.info(
            "Label '%s' (task_type=%s) class distribution: %s (n=%d patients)",
            self.spec.source_column, self.spec.task_type, dist, len(df),
        )


def load_clinical_csv(csv_path: str) -> pd.DataFrame:
    """Robust CSV loader tolerant of encoding / separator quirks common in
    REDCap-style exports (e.g. RUGGIStudy_DATA_*.csv)."""
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    if "record_id" not in df.columns:
        raise KeyError(
            f"Expected a 'record_id' column in {csv_path}, "
            f"got columns: {list(df.columns)[:20]}"
        )
    df["record_id"] = pd.to_numeric(df["record_id"], errors="coerce")
    df = df.dropna(subset=["record_id"])
    df["record_id"] = df["record_id"].astype(int)
    return df
