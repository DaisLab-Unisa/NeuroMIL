"""
utils/checkpoint.py

Robust checkpointing with full training-state resume support:
  - model / optimizer / scheduler state
  - epoch, best metric value
  - random seeds (python, numpy, torch, cuda) for exact reproducibility
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG state tensors must stay on CPU regardless of the checkpoint's
    # map_location: torch.load(..., map_location="cuda") also remaps these,
    # but torch.set_rng_state()/cuda.set_rng_state_all() require CPU tensors.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    best_metric: float,
    extra: Optional[Dict[str, Any]] = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "rng_state": get_rng_state(),
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if restore_rng and "rng_state" in ckpt:
        set_rng_state(ckpt["rng_state"])
    return ckpt
