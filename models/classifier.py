"""
models/classifier.py

Patient-level classifier head + the end-to-end PainMILModel that chains
Fusion -> AttentionMIL -> Classifier. The number of output classes is
taken directly from `ClinicalLabelProcessor.num_classes`, so the exact
same architecture serves NRS severity, DN4, BPI-complete, or any future
clinical variable, without code changes (requirement 5D).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .fusion import MultimodalFusion
from .mil_attention import AttentionMIL


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PainMILModel(nn.Module):
    """
    Full pipeline:

        {video,audio,text}_emb  [B,N,D_m]
                |
                v
        MultimodalFusion         -> clip repr [B,N,H]
                |
                v
        AttentionMIL              -> patient_emb [B,H], attn_weights [B,N]
                |
                v
        ClassifierHead             -> logits [B, num_classes]
    """

    def __init__(self, config: Dict, num_classes: int):
        super().__init__()
        self.fusion = MultimodalFusion(config)
        self.mil = AttentionMIL(config)

        mil_hidden = config["model"]["mil_attention"]["hidden_dim"]
        clf_cfg = config["model"]["classifier"]
        self.classifier = ClassifierHead(
            input_dim=mil_hidden,
            num_classes=num_classes,
            hidden_dim=clf_cfg["hidden_dim"],
            dropout=clf_cfg.get("dropout", 0.3),
        )

    def forward(
        self, batch: Dict[str, torch.Tensor], return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        modality_embs = {m: batch[f"{m}_emb"] for m in self.fusion.modalities}
        mask = batch.get("mask", None)

        clip_reprs = self.fusion(modality_embs)             # [B, N, H]
        patient_emb, attn_weights = self.mil(clip_reprs, mask=mask)  # [B, H], [B, N]
        logits = self.classifier(patient_emb)                # [B, num_classes]

        if return_attention:
            return logits, attn_weights
        return logits, None
