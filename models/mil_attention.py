"""
models/mil_attention.py

Attention-based Multiple Instance Learning pooling (Ilse et al., 2018,
"Attention-based Deep Multiple Instance Learning"), including the gated
variant, which learns which temporal clips are relevant to the
patient-level clinical outcome and aggregates them into a single patient
representation.

    clip representations [B, N, H]
            |
            v
    attention weights a_i = softmax_i( w^T tanh(V h_i) [* sigmoid(U h_i)] )
            |
            v
    patient embedding = sum_i a_i * h_i    [B, H]
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        mil_cfg = config["model"]["mil_attention"]
        input_dim = config["model"]["fusion"]["hidden_dim"]
        hidden_dim = mil_cfg["hidden_dim"]
        attn_dim = mil_cfg["attention_dim"]
        dropout = mil_cfg.get("dropout", 0.3)
        self.gated = mil_cfg.get("gated", True)

        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        if self.gated:
            self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1)

    def forward(
        self, clip_reprs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        clip_reprs: [B, N, D]  (per-clip fused representations)
        mask:       [B, N] bool, True for valid (non-padded) clips

        Returns
        -------
        patient_emb : [B, H]
        attn_weights: [B, N]  (softmax attention over clips; padded clips = 0)
        """
        h = self.instance_encoder(clip_reprs)  # [B, N, H]

        att_v = self.attention_V(h)  # [B, N, attn_dim]
        if self.gated:
            att_u = self.attention_U(h)
            att = self.attention_w(att_v * att_u).squeeze(-1)  # [B, N]
        else:
            att = self.attention_w(att_v).squeeze(-1)  # [B, N]

        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(att, dim=1)  # [B, N]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)

        patient_emb = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # [B, H]
        return patient_emb, attn_weights
