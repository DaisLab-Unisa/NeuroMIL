"""
embeddings/extraction_pipeline.py

Mode A (offline embedding extraction) of the framework:

    video/audio/text (Italian)
            |
            v
    selected foundation model (embeddings/*.py backend)
            |
            v
    features/saved_embeddings/<record_id>/{video,audio,text}_embeddings.pt

Run:
    python -m embeddings.extraction_pipeline --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.ruggi_dataset import (
    ClipRef,
    build_dataset_manifest,
    probe_duration_ffprobe,
    scan_video_directory,
    segment_video_into_clips,
)
from embeddings.base import ClipInput, get_embedding_extractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =========================================================================
# Audio extraction + Italian ASR (Whisper large-v3)
# =========================================================================

def extract_audio_waveform(video_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Extract mono waveform from a video file via ffmpeg (through torchaudio/moviepy-free path)."""
    import os
    import subprocess
    import tempfile
    import soundfile as sf

    # delete=False + explicit close: on Windows, ffmpeg cannot open a file
    # for writing while the Python process still holds its own handle open
    # (unlike POSIX), so a plain `with NamedTemporaryFile(...)` makes every
    # ffmpeg invocation fail with an access-denied error.
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", str(target_sr), "-vn", tmp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        waveform, sr = sf.read(tmp_path, dtype="float32")
    finally:
        os.remove(tmp_path)
    return waveform, sr


# Whisper transcription is backend-independent (video/audio content, not the
# embedding model), but run_extraction is invoked once per embedding backend
# -- cache transcripts across backends so a 91-video ASR pass isn't repeated.
TRANSCRIPTS_CACHE_DIR = Path("features/transcripts_cache")


def get_or_transcribe(record_id: int, video_path: str, audio_cfg: Dict) -> List[dict]:
    TRANSCRIPTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TRANSCRIPTS_CACHE_DIR / f"{record_id}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    waveform, sr = extract_audio_waveform(video_path)
    segments = transcribe_italian(waveform, sr, audio_cfg["whisper_model"], audio_cfg["device"])
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    return segments


def transcribe_italian(waveform: np.ndarray, sr: int, model_size: str, device: str) -> List[dict]:
    """Run faster-whisper (large-v3) on the full-interview waveform, returning
    a list of {start, end, text} segments in Italian with word-level timing
    granularity preserved at the segment level."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        waveform, language="it", vad_filter=True, beam_size=5
    )
    result = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    logger.info("Transcribed %d segments", len(result))
    return result


# =========================================================================
# Frame sampling
# =========================================================================

def sample_frames(video_path: str, start_sec: float, end_sec: float, n_frames: int) -> List[np.ndarray]:
    """Uniformly sample `n_frames` RGB frames between start_sec and end_sec."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    timestamps = np.linspace(start_sec, max(end_sec - 1e-3, start_sec), n_frames)

    frames = []
    for t in timestamps:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            # fall back to last successfully read frame, or a blank frame
            frame_bgr = frames[-1][:, :, ::-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        frames.append(frame_bgr[:, :, ::-1].copy())  # BGR -> RGB
    cap.release()
    return frames


def slice_waveform(waveform: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    return waveform[start_idx:end_idx]


# =========================================================================
# Full pipeline
# =========================================================================

def run_extraction(config: Dict):
    paths = config["paths"]
    temporal_cfg = config["temporal"]
    audio_cfg = config["audio"]
    embed_cfg = config["embedding"]

    features_dir = Path(paths["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest, _processor = build_dataset_manifest(config)
    extractor = get_embedding_extractor(config)

    # Persist the backend's *actual* output dims so training can size the
    # fusion projectors correctly instead of relying on the (often stale)
    # config['model']['fusion']['input_dims'] guess.
    dims_path = features_dir / "embedding_dims.json"
    if not dims_path.exists():
        dims = {
            "backend": extractor.name,
            "video": extractor.video_dim,
            "audio": extractor.audio_dim,
            "text": extractor.text_dim,
        }
        with open(dims_path, "w", encoding="utf-8") as f:
            json.dump(dims, f, indent=2)
        logger.info("Recorded embedding dims for backend '%s': %s", extractor.name, dims)

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Patients"):
        record_id = int(row["record_id"])
        video_path = row["video_path"]
        patient_out_dir = features_dir / str(record_id)

        if (patient_out_dir / "meta.json").exists():
            logger.info("Skipping record %d (already extracted)", record_id)
            continue

        patient_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            _extract_patient(
                record_id, video_path, patient_out_dir,
                temporal_cfg, audio_cfg, embed_cfg, extractor,
            )
        except Exception as e:
            logger.error("Failed extraction for record %d (%s): %s", record_id, video_path, e)
            continue


def _extract_patient(record_id, video_path, out_dir, temporal_cfg, audio_cfg, embed_cfg, extractor):
    # 1. Duration + clip segmentation
    duration = probe_duration_ffprobe(Path(video_path))

    class _PV:  # lightweight shim matching PatientVideo interface
        pass
    pv = _PV()
    pv.record_id = record_id
    pv.video_path = Path(video_path)
    pv.duration_sec = duration

    clips: List[ClipRef] = segment_video_into_clips(
        pv,
        clip_duration_sec=temporal_cfg["clip_duration_sec"],
        clip_stride_sec=temporal_cfg["clip_stride_sec"],
        max_clips=temporal_cfg["max_clips_per_patient"],
        min_clips=temporal_cfg["min_clips_per_patient"],
    )

    # 2. Full-interview audio + Italian ASR (once per patient, cached across backends)
    waveform, sr = extract_audio_waveform(video_path)
    transcript_segments = get_or_transcribe(record_id, video_path, audio_cfg)

    # 3. Per-clip embedding extraction
    video_embs, audio_embs, text_embs = [], [], []
    for clip in clips:
        frames = sample_frames(
            video_path, clip.start_sec, clip.end_sec,
            n_frames=embed_cfg.get("frames_per_clip", 8),
        )
        clip_waveform = slice_waveform(waveform, sr, clip.start_sec, clip.end_sec)
        overlapping_text = " ".join(
            s["text"] for s in transcript_segments
            if s["start"] < clip.end_sec and s["end"] > clip.start_sec
        ).strip()

        clip_input = ClipInput(
            frames=frames, audio_waveform=clip_waveform, audio_sr=sr, text=overlapping_text
        )
        emb = extractor.embed_clip(clip_input)

        video_embs.append(emb.video_emb)
        audio_embs.append(emb.audio_emb)
        text_embs.append(emb.text_emb)

    # 4. Save cached tensors
    torch.save(torch.tensor(np.stack(video_embs), dtype=torch.float32), out_dir / "video_embeddings.pt")
    torch.save(torch.tensor(np.stack(audio_embs), dtype=torch.float32), out_dir / "audio_embeddings.pt")
    torch.save(torch.tensor(np.stack(text_embs), dtype=torch.float32), out_dir / "text_embeddings.pt")

    meta = {
        "record_id": record_id,
        "video_path": str(video_path),
        "n_clips": len(clips),
        "clip_windows": [[c.start_sec, c.end_sec] for c in clips],
        "duration_sec": duration,
        "embedding_backend": extractor.name,
        "transcript_segments": transcript_segments,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Record %d: cached %d clip embeddings -> %s", record_id, len(clips), out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_extraction(config)


if __name__ == "__main__":
    main()
