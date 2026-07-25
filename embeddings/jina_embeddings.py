"""
embeddings/jina_embeddings.py

Backend implementing the NeuroMIL paper's Multimodal Feature Extraction
stage (Section IV-A):
  - Jina-CLIP-v2        -> video embeddings (Dv = 1024)
  - Whisper large-v3     -> audio embeddings, mean-pooled encoder hidden
                            states (Da = whisper encoder hidden size)
  - Multilingual-E5-Large -> text embeddings of the Italian ASR transcript
                            overlapping each clip (Dt = 1024)

Whisper large-v3 is used twice in the overall pipeline: once via
faster-whisper for full-interview Italian ASR transcription
(embeddings/extraction_pipeline.py), and once here, via the Hugging Face
`WhisperModel` encoder, to produce the dense per-clip acoustic
representation described in the paper.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ClipEmbedding, ClipInput, EmbeddingExtractor

logger = logging.getLogger(__name__)


def _resolve_whisper_repo(model_size: str) -> str:
    """Maps a faster-whisper style size string (e.g. 'large-v3') to its
    Hugging Face Whisper checkpoint id (e.g. 'openai/whisper-large-v3')."""
    return model_size if "/" in model_size else f"openai/whisper-{model_size}"


class JinaEmbeddingExtractor(EmbeddingExtractor):
    name = "jina"

    def __init__(self, config: dict):
        super().__init__(config)
        j_cfg = config["embedding"]["jina"]
        self.clip_model_name = j_cfg.get("clip_model_name", "jinaai/jina-clip-v2")
        self.text_model_name = config["text"]["model_name"]
        self.audio_model_name = _resolve_whisper_repo(config["audio"]["whisper_model"])
        self.device = config["embedding"].get("device", "cuda")

        self._clip_model = None
        self._text_model = None
        self._text_tokenizer = None
        self._audio_model = None
        self._audio_feature_extractor = None

    # ------------------------------------------------------------------
    @staticmethod
    def _patch_transformers_clip_loss():
        """jina-clip-v2's remote modeling code imports `clip_loss` from
        transformers.models.clip.modeling_clip; newer transformers releases
        dropped that helper (keeping `contrastive_loss`). It is never
        actually called in inference-only usage here, so a compatible shim
        is sufficient to satisfy the import."""
        import transformers.models.clip.modeling_clip as clip_mod
        if not hasattr(clip_mod, "clip_loss"):
            def clip_loss(similarity):
                caption_loss = clip_mod.contrastive_loss(similarity)
                image_loss = clip_mod.contrastive_loss(similarity.t())
                return (caption_loss + image_loss) / 2.0
            clip_mod.clip_loss = clip_loss

    def _lazy_init(self):
        if self._clip_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

        self._patch_transformers_clip_loss()

        logger.info("Loading video encoder (Jina-CLIP-v2): %s", self.clip_model_name)
        self._clip_model = AutoModel.from_pretrained(
            self.clip_model_name, trust_remote_code=True
        ).to(self.device).eval()

        logger.info("Loading text encoder (Multilingual-E5-Large): %s", self.text_model_name)
        self._text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self._text_model = AutoModel.from_pretrained(self.text_model_name).to(self.device).eval()

        logger.info("Loading audio encoder (Whisper large-v3): %s", self.audio_model_name)
        self._audio_feature_extractor = WhisperFeatureExtractor.from_pretrained(self.audio_model_name)
        self._audio_model = WhisperModel.from_pretrained(self.audio_model_name).to(self.device).eval()

    # ------------------------------------------------------------------
    def embed_clip(self, clip: ClipInput) -> ClipEmbedding:
        import torch
        from PIL import Image

        self._lazy_init()
        mid_frame = clip.frames[len(clip.frames) // 2]
        pil_image = Image.fromarray(mid_frame)

        with torch.no_grad():
            video_emb = self._clip_model.encode_image([pil_image])[0]
            text_emb = (
                self._embed_text(clip.text) if clip.text
                else np.zeros(self.text_dim, dtype=np.float32)
            )
            audio_emb = (
                self._embed_audio(clip.audio_waveform, clip.audio_sr)
                if clip.audio_waveform is not None and clip.audio_waveform.size > 0
                else np.zeros(self.audio_dim, dtype=np.float32)
            )

        video_emb = np.asarray(video_emb, dtype=np.float32)
        text_emb = np.asarray(text_emb, dtype=np.float32)
        multimodal_emb = self._fuse(video_emb, text_emb)

        return ClipEmbedding(
            video_emb=video_emb, text_emb=text_emb,
            audio_emb=audio_emb, multimodal_emb=multimodal_emb,
        )

    def _embed_text(self, text: str) -> np.ndarray:
        """Multilingual-E5-Large embedding: mean-pooled, L2-normalized, with
        the "passage: " prefix required by E5's usage convention."""
        import torch

        inputs = self._text_tokenizer(
            f"passage: {text}", return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        out = self._text_model(**inputs)
        pooled = self._mean_pool(out.last_hidden_state, inputs["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.float().cpu().numpy().squeeze(0)

    def _embed_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Whisper large-v3 dense acoustic representation: mean-pooled
        encoder hidden states over the clip's audio window."""
        inputs = self._audio_feature_extractor(
            waveform, sampling_rate=sr, return_tensors="pt"
        ).to(self.device)
        encoder_out = self._audio_model.encoder(inputs.input_features)
        pooled = encoder_out.last_hidden_state.mean(dim=1)
        return pooled.float().cpu().numpy().squeeze(0)

    @staticmethod
    def _mean_pool(hidden_states: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    @staticmethod
    def _fuse(video_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        v = video_emb / (np.linalg.norm(video_emb) + 1e-8)
        t = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        return np.concatenate([v, t], axis=0)

    # ------------------------------------------------------------------
    @property
    def video_dim(self) -> int:
        return 1024  # jina-clip-v2 default output dim

    @property
    def audio_dim(self) -> int:
        self._lazy_init()
        return self._audio_model.config.d_model  # whisper large-v3 encoder hidden size

    @property
    def text_dim(self) -> int:
        return 1024  # multilingual-e5-large default output dim