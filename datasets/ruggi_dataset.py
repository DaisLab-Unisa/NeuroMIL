"""
datasets/ruggi_dataset.py

Handles:
  1. Scanning the video directory and extracting patient IDs from filenames.
  2. Matching videos to clinical metadata (record_id).
  3. Segmenting each interview into temporal clips (MIL "instances").
  4. A PyTorch Dataset that serves either:
       - raw clip references (for offline embedding extraction), or
       - cached embeddings as an MIL "bag" (for training).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .label_processor import ClinicalLabelProcessor, load_clinical_csv

logger = logging.getLogger(__name__)


# =========================================================================
# Video <-> record_id matching
# =========================================================================

@dataclass
class PatientVideo:
    record_id: int
    video_path: Path
    duration_sec: Optional[float] = None


def scan_video_directory(video_dir: str, extensions: List[str], id_regex: str) -> List[PatientVideo]:
    """Scan `video_dir` for supported video files and extract record_id
    from each filename using `id_regex` (applied to the file stem)."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    pattern = re.compile(id_regex)
    exts = {e.lower() for e in extensions}
    found: Dict[int, PatientVideo] = {}

    for f in sorted(video_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        match = pattern.match(f.stem)
        if not match:
            logger.warning("Could not extract record_id from filename: %s", f.name)
            continue
        record_id = int(match.group(1))
        if record_id in found:
            logger.warning(
                "Duplicate record_id=%d for files %s and %s -- keeping first, skipping duplicate",
                record_id, found[record_id].video_path.name, f.name,
            )
            continue
        found[record_id] = PatientVideo(record_id=record_id, video_path=f)

    logger.info("Scanned %s: found %d videos", video_dir, len(found))
    return list(found.values())


def match_videos_with_clinical_data(
    videos: List[PatientVideo], clinical_df: pd.DataFrame
) -> Tuple[List[PatientVideo], List[int], List[int]]:
    """
    Returns
    -------
    matched   : videos that have a corresponding row in clinical_df
    unmatched_videos : record_ids present on disk but missing from CSV
    unmatched_csv     : record_ids present in CSV but missing on disk
    """
    csv_ids = set(clinical_df["record_id"].tolist())
    video_ids = {v.record_id for v in videos}

    matched = [v for v in videos if v.record_id in csv_ids]
    unmatched_videos = sorted(video_ids - csv_ids)
    unmatched_csv = sorted(csv_ids - video_ids)

    if unmatched_videos:
        logger.warning("%d videos have no matching clinical record: %s",
                        len(unmatched_videos), unmatched_videos)
    if unmatched_csv:
        logger.warning("%d clinical records have no matching video: %s",
                        len(unmatched_csv), unmatched_csv)

    logger.info("Matched %d / %d videos to clinical data", len(matched), len(videos))
    return matched, unmatched_videos, unmatched_csv


# =========================================================================
# Temporal clip segmentation
# =========================================================================

@dataclass
class ClipRef:
    record_id: int
    clip_index: int
    start_sec: float
    end_sec: float
    video_path: Path


def segment_video_into_clips(
    patient_video: PatientVideo,
    clip_duration_sec: float,
    clip_stride_sec: float,
    max_clips: int,
    min_clips: int,
) -> List[ClipRef]:
    """Produce non-overlapping (by default) temporal windows covering the
    whole interview. Requires the video duration to be known; if not yet
    probed, callers should populate `patient_video.duration_sec` first
    (see `utils.video_probe`, invoked lazily during extraction)."""
    duration = patient_video.duration_sec
    if duration is None:
        raise ValueError(
            f"Duration unknown for {patient_video.video_path}. "
            f"Call probe_duration() before segmenting."
        )

    clips: List[ClipRef] = []
    t = 0.0
    idx = 0
    while t < duration and idx < max_clips:
        end = min(t + clip_duration_sec, duration)
        clips.append(ClipRef(
            record_id=patient_video.record_id,
            clip_index=idx,
            start_sec=t,
            end_sec=end,
            video_path=patient_video.video_path,
        ))
        idx += 1
        t += clip_stride_sec

    if len(clips) < min_clips:
        logger.warning(
            "Patient %d: only %d clips generated (< min_clips=%d). "
            "Interview may be too short for reliable MIL bag.",
            patient_video.record_id, len(clips), min_clips,
        )
    return clips


def probe_duration_ffprobe(video_path: Path) -> float:
    """Probe video duration in seconds using ffprobe (must be on PATH)."""
    import subprocess

    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


# =========================================================================
# Datasets
# =========================================================================

class RuggiRawClipDataset(Dataset):
    """
    Serves raw (video_path, start, end, transcript_segment) references for
    every clip of every matched patient. Used ONLY by the offline
    extraction pipeline (embeddings/extraction_pipeline.py) -- never
    consumed directly by the MIL model.
    """

    def __init__(self, clip_refs: List[ClipRef], transcripts: Optional[Dict[int, List[dict]]] = None):
        self.clip_refs = clip_refs
        # transcripts[record_id] = list of {"start", "end", "text"} whisper segments
        self.transcripts = transcripts or {}

    def __len__(self) -> int:
        return len(self.clip_refs)

    def __getitem__(self, idx: int) -> Dict:
        clip = self.clip_refs[idx]
        text = self._transcript_for_clip(clip)
        return {
            "record_id": clip.record_id,
            "clip_index": clip.clip_index,
            "video_path": str(clip.video_path),
            "start_sec": clip.start_sec,
            "end_sec": clip.end_sec,
            "text": text,
        }

    def _transcript_for_clip(self, clip: ClipRef) -> str:
        segments = self.transcripts.get(clip.record_id, [])
        overlapping = [
            s["text"] for s in segments
            if s["start"] < clip.end_sec and s["end"] > clip.start_sec
        ]
        return " ".join(overlapping).strip()


class RuggiMILDataset(Dataset):
    """
    Loads cached per-clip embeddings for each patient and serves them as
    an MIL "bag": (clip_embeddings [N_clips, D], label, record_id).

    Bags have variable length -> use `mil_collate_fn` in the DataLoader.
    """

    def __init__(
        self,
        record_ids: List[int],
        labels: Dict[int, int],
        features_dir: str,
        modalities: Tuple[str, ...] = ("video", "audio", "text"),
    ):
        self.record_ids = record_ids
        self.labels = labels
        self.features_dir = Path(features_dir)
        self.modalities = modalities

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> Dict:
        record_id = self.record_ids[idx]
        patient_dir = self.features_dir / str(record_id)

        modality_tensors = {}
        for m in self.modalities:
            path = patient_dir / f"{m}_embeddings.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached embedding {path}. Run "
                    f"embeddings/extraction_pipeline.py first."
                )
            modality_tensors[m] = torch.load(path, map_location="cpu")  # [N_clips, D_m]

        n_clips = modality_tensors[self.modalities[0]].shape[0]
        for m, t in modality_tensors.items():
            assert t.shape[0] == n_clips, (
                f"Clip count mismatch for record {record_id}, modality {m}: "
                f"{t.shape[0]} vs {n_clips}"
            )

        return {
            "record_id": record_id,
            "label": self.labels[record_id],
            **{f"{m}_emb": modality_tensors[m] for m in self.modalities},
            "n_clips": n_clips,
        }


def mil_collate_fn(batch: List[Dict], modalities: Tuple[str, ...] = ("video", "audio", "text")) -> Dict:
    """Pads variable-length clip bags to the max bag size in the batch and
    returns an attention mask so padded clips are ignored by MIL attention."""
    max_clips = max(item["n_clips"] for item in batch)
    B = len(batch)

    out = {"record_id": [], "label": [], "mask": torch.zeros(B, max_clips, dtype=torch.bool)}
    for m in modalities:
        d = batch[0][f"{m}_emb"].shape[1]
        out[f"{m}_emb"] = torch.zeros(B, max_clips, d)

    for i, item in enumerate(batch):
        n = item["n_clips"]
        out["record_id"].append(item["record_id"])
        out["label"].append(item["label"])
        out["mask"][i, :n] = True
        for m in modalities:
            out[f"{m}_emb"][i, :n] = item[f"{m}_emb"]

    out["label"] = torch.tensor(out["label"], dtype=torch.long)
    return out


# =========================================================================
# High-level convenience builder
# =========================================================================

def build_dataset_manifest(config: Dict) -> pd.DataFrame:
    """
    End-to-end: scan videos, load clinical CSV, compute labels, match, and
    return a single manifest DataFrame:

        record_id | video_path | label | class_name
    """
    paths = config["paths"]
    matching = config["data_matching"]

    videos = scan_video_directory(
        paths["video_dir"], matching["video_extensions"], matching["id_regex"]
    )
    clinical_df = load_clinical_csv(paths["clinical_csv"])

    processor = ClinicalLabelProcessor(config)
    label_df = processor.fit_transform(clinical_df)

    matched, unmatched_videos, unmatched_csv = match_videos_with_clinical_data(
        videos, label_df.rename(columns={"record_id": "record_id"})
    )

    video_by_id = {v.record_id: v for v in matched}
    manifest = label_df[label_df["record_id"].isin(video_by_id.keys())].copy()
    manifest["video_path"] = manifest["record_id"].map(
        lambda rid: str(video_by_id[rid].video_path)
    )
    manifest = manifest.reset_index(drop=True)

    logger.info(
        "Final manifest: %d patients with both video and valid label", len(manifest)
    )
    return manifest, processor
