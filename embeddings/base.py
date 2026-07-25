"""
embeddings/base.py

Common interface every multimodal embedding backend must implement, so the
rest of the pipeline (extraction_pipeline.py, training) is fully agnostic
to which foundation model produced the embeddings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ClipInput:
    """Everything a backend needs to embed a single temporal clip."""
    frames: List[np.ndarray]      # sampled RGB frames, HxWx3 uint8
    audio_waveform: Optional[np.ndarray]  # mono waveform for this clip window, or None
    audio_sr: Optional[int]
    text: str                     # Italian transcript segment overlapping this clip


@dataclass
class ClipEmbedding:
    video_emb: Optional[np.ndarray] = None   # [D_video]
    audio_emb: Optional[np.ndarray] = None   # [D_audio]
    text_emb: Optional[np.ndarray] = None    # [D_text]
    multimodal_emb: Optional[np.ndarray] = None  # [D_joint], if backend produces a fused vector


class EmbeddingExtractor(ABC):
    """
    Abstract base class for all embedding backends (Qwen3-VL, Jina).

    Backends are used strictly as *encoders* here -- never for text
    generation -- even when the underlying checkpoint is a generative VLM
    (see qwen3vl_embeddings.py for the pooling strategy that enforces this).
    """

    name: str = "base"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def embed_clip(self, clip: ClipInput) -> ClipEmbedding:
        """Embed a single clip. Implementations should batch internally
        where possible; extraction_pipeline.py calls this per-clip but
        backends are free to cache/batch under the hood."""
        raise NotImplementedError

    def embed_batch(self, clips: List[ClipInput]) -> List[ClipEmbedding]:
        """Default naive loop; override for true batched inference."""
        return [self.embed_clip(c) for c in clips]

    @property
    @abstractmethod
    def video_dim(self) -> int: ...

    @property
    @abstractmethod
    def audio_dim(self) -> int: ...

    @property
    @abstractmethod
    def text_dim(self) -> int: ...


def get_embedding_extractor(config: dict) -> EmbeddingExtractor:
    """Factory: instantiate the backend selected in config['embedding']['backend']."""
    backend = config["embedding"]["backend"]
    if backend == "qwen3vl":
        from .qwen3vl_embeddings import Qwen3VLEmbeddingExtractor
        return Qwen3VLEmbeddingExtractor(config)
    elif backend == "jina":
        from .jina_embeddings import JinaEmbeddingExtractor
        return JinaEmbeddingExtractor(config)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
