"""Deterministic footage analysis: stream facts, shot changes, silences, speech spans, transcript.

Everything except the optional transcript is pure FFmpeg signal processing — no model, same answer
every run. The transcript uses a local faster-whisper model (CUDA when available) only on request.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from shared.syscmd import find_tool

from .probe import probe

_PTS = re.compile(r"pts_time:([0-9.]+)")
_SIL_START = re.compile(r"silence_start: (-?[0-9.]+)")
_SIL_END = re.compile(r"silence_end: (-?[0-9.]+)")


def _ffmpeg_filter_log(path: str, args: list[str]) -> str:
    result = subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", path, *args, "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg analysis failed for {path}: {result.stderr.strip()[-400:]}")
    return result.stderr


def detect_scenes(path: str, threshold: float = 0.3) -> list[float]:
    """Shot-change timestamps (seconds) from FFmpeg's scene score on a downscaled stream."""
    log = _ffmpeg_filter_log(path, ["-an", "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo"])
    return [round(float(m), 3) for m in _PTS.findall(log)]


def detect_silences(path: str, noise_db: float = -35.0, min_silence: float = 0.5, duration: float | None = None):
    """[(start, end)] spans quieter than ``noise_db`` for at least ``min_silence`` seconds."""
    log = _ffmpeg_filter_log(path, ["-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}"])
    spans, start = [], None
    for line in log.splitlines():
        if m := _SIL_START.search(line):
            start = max(float(m.group(1)), 0.0)
        elif (m := _SIL_END.search(line)) and start is not None:
            spans.append((round(start, 3), round(float(m.group(1)), 3)))
            start = None
    if start is not None and duration is not None:
        spans.append((round(start, 3), round(duration, 3)))
    return spans


def speech_spans(silences, duration: float, pad: float = 0.15, min_len: float = 0.3):
    """Complement of ``silences`` over [0, duration], padded and merged — the 'keep' list for a jump cut."""
    keep, cursor = [], 0.0
    for s, e in silences:
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        keep.append((cursor, duration))
    padded = [(max(s - pad, 0.0), min(e + pad, duration)) for s, e in keep]
    merged: list[list[float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged if e - s >= min_len]


def transcribe(path: str, model: str = "small", language: str | None = None) -> list[dict]:
    """Local Whisper transcript via faster-whisper (CUDA float16 when a GPU is usable, else CPU int8)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("transcription needs faster-whisper: pip install faster-whisper") from exc
    try:
        whisper = WhisperModel(model, device="cuda", compute_type="float16")
    except Exception:  # noqa: BLE001 — no usable CUDA runtime; fall back to CPU
        whisper = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(path, language=language, vad_filter=True)
    return [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in segments]


def analyze(
    path: str | Path,
    *,
    scenes: bool = True,
    scene_threshold: float = 0.3,
    silences: bool = True,
    noise_db: float = -35.0,
    min_silence: float = 0.5,
    transcript: bool = False,
    whisper_model: str = "small",
    language: str | None = None,
) -> dict:
    info = probe(path)
    report: dict = {"media": info.as_dict()}
    if info.is_image:
        return report
    if scenes and info.has_video:
        report["scene_changes"] = detect_scenes(info.path, scene_threshold)
    if silences and info.has_audio and info.duration:
        sil = detect_silences(info.path, noise_db, min_silence, info.duration)
        report["silences"] = [list(s) for s in sil]
        report["speech"] = [list(s) for s in speech_spans(sil, info.duration)]
    if transcript and info.has_audio:
        report["transcript"] = transcribe(info.path, whisper_model, language)
    return report
