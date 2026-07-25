"""
models/fusion.py

Per-clip multimodal fusion:

    video_emb + audio_emb + text_emb  ->  Fusion MLP  ->  clip representation

Each modality is first projected to a common hidden dim (handles backends
with different native dims, e.g. Qwen3-VL video vs wav2vec2 audio), then
concatenated and passed through an MLP.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Projects a single modality's embedding to a common dimension."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusion(nn.Module):
    """
    Fuses per-clip video/audio/text embeddings into a single clip
    representation used as the MIL instance embedding.

    Input shapes: [B, N_clips, D_modality] per modality.
    Output shape: [B, N_clips, hidden_dim].
    """

    def __init__(self, config: Dict):
        super().__init__()
        fusion_cfg = config["model"]["fusion"]
        input_dims = fusion_cfg["input_dims"]
        hidden_dim = fusion_cfg["hidden_dim"]
        dropout = fusion_cfg.get("dropout", 0.3)
        proj_dim = hidden_dim // 2  # each modality projected, then concatenated

        self.modalities = list(input_dims.keys())
        self.projectors = nn.ModuleDict({
            m: ModalityProjector(input_dims[m], proj_dim, dropout) for m in self.modalities
        })

        concat_dim = proj_dim * len(self.modalities)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, modality_embs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        modality_embs: {"video": [B,N,Dv], "audio": [B,N,Da], "text": [B,N,Dt]}
        returns: [B, N, hidden_dim]
        """
        projected = [self.projectors[m](modality_embs[m]) for m in self.modalities]
        concat = torch.cat(projected, dim=-1)  # [B, N, proj_dim * n_modalities]
        return self.fusion_mlp(concat)
