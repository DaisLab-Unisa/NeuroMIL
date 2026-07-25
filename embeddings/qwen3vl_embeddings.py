"""
embeddings/qwen3vl_embeddings.py

Uses Qwen3-VL purely as a frozen multimodal *encoder*: frames + Italian
transcript segment go in, a pooled hidden-state vector comes out. No
`generate()` call is ever made -- we only run a forward pass and pool the
last hidden states, which is the standard way to repurpose a decoder-only
VLM as an embedding extractor.

Audio is not natively consumed by Qwen3-VL, so this backend produces:
  - video_emb  : pooled hidden state conditioned on frames + text (vision-grounded)
  - text_emb   : pooled hidden state from a text-only forward pass
  - audio_emb  : delegated to a separate lightweight speech encoder
                 (wav2vec2 / whisper encoder), since Qwen3-VL has no audio tower.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from .base import ClipEmbedding, ClipInput, EmbeddingExtractor

logger = logging.getLogger(__name__)


class Qwen3VLEmbeddingExtractor(EmbeddingExtractor):
    name = "qwen3vl"

    def __init__(self, config: dict):
        super().__init__(config)
        q_cfg = config["embedding"]["qwen3vl"]
        self.model_path = q_cfg["model_name_or_path"]
        self.pooling = q_cfg.get("pooling", "mean_last_hidden")
        self.dtype_str = q_cfg.get("dtype", "bfloat16")
        self.device = config["embedding"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None
        self._audio_encoder = None  # lazy wav2vec2/whisper-encoder for audio_emb

    # ------------------------------------------------------------------
    def _lazy_init(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText

        dtype = getattr(torch, self.dtype_str, torch.bfloat16)
        logger.info("Loading Qwen3-VL from %s (dtype=%s)...", self.model_path, dtype)

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=self.device
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        self._hidden_size = self._model.config.get_text_config().hidden_size

    def _lazy_init_audio(self):
        if self._audio_encoder is not None:
            return
        from transformers import AutoModel, AutoFeatureExtractor

        audio_model_name = "facebook/wav2vec2-large-xlsr-53"  # multilingual, incl. Italian
        self._audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
        self._audio_encoder = AutoModel.from_pretrained(audio_model_name).to(self.device).eval()
        for p in self._audio_encoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed_clip(self, clip: ClipInput) -> ClipEmbedding:
        self._lazy_init()

        # ---- Vision+text conditioned embedding (video_emb) ----
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": frame} for frame in clip.frames],
                {"type": "text", "text": clip.text or "(nessun parlato in questo segmento)"},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        outputs = self._model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, H]
        video_emb = self._pool(last_hidden, inputs.get("attention_mask"))

        # ---- Text-only embedding (text_emb) ----
        text_inputs = self._processor.tokenizer(
            clip.text or "", return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        text_hidden = self._model.get_input_embeddings()(text_inputs["input_ids"])
        # simple mean pooling over token embeddings as a lightweight text tower
        text_emb = text_hidden.mean(dim=1)

        # ---- Audio embedding via auxiliary multilingual speech encoder ----
        audio_emb = self._embed_audio(clip)

        return ClipEmbedding(
            video_emb=video_emb.float().cpu().numpy().squeeze(0),
            text_emb=text_emb.float().cpu().numpy().squeeze(0),
            audio_emb=audio_emb,
        )

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        if self.pooling == "eos_token":
            return hidden_states[:, -1, :]
        # default: mean pooling over valid (non-padding) tokens
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    @torch.no_grad()
    def _embed_audio(self, clip: ClipInput) -> np.ndarray:
        if clip.audio_waveform is None:
            return np.zeros(self.audio_dim, dtype=np.float32)
        self._lazy_init_audio()
        inputs = self._audio_feature_extractor(
            clip.audio_waveform, sampling_rate=clip.audio_sr, return_tensors="pt"
        ).to(self.device)
        out = self._audio_encoder(**inputs)
        pooled = out.last_hidden_state.mean(dim=1)
        return pooled.float().cpu().numpy().squeeze(0)

    # ------------------------------------------------------------------
    @property
    def video_dim(self) -> int:
        self._lazy_init()
        return self._hidden_size

    @property
    def audio_dim(self) -> int:
        return 1024  # wav2vec2-large hidden size

    @property
    def text_dim(self) -> int:
        self._lazy_init()
        return self._hidden_size
