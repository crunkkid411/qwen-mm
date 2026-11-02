"""ffprobe wrapper and NVENC detection."""

from __future__ import annotations

import functools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from shared.env import get_int_env
from shared.syscmd import find_tool

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    has_video: bool
    has_audio: bool
    is_image: bool

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _rate(value: str | None) -> float | None:
    if not value or value in ("0/0", "0"):
        return None
    num, _, den = value.partition("/")
    try:
        return round(float(num) / float(den or 1), 3)
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: str | Path) -> MediaInfo:
    """Return stream facts for ``path`` (raises FileNotFoundError / RuntimeError)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    is_image = path.suffix.lower() in IMAGE_EXTS
    result = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        timeout=get_int_env("QWEN_MM_FFMPEG_TIMEOUT"),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[-400:]}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = data.get("format", {}).get("duration") or (video or {}).get("duration")
    return MediaInfo(
        path=str(path),
        duration=None if is_image or duration is None else round(float(duration), 3),
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        fps=None if is_image or not video else _rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_video=video is not None,
        has_audio=audio is not None and not is_image,
        is_image=is_image,
    )


@functools.lru_cache(maxsize=None)
def nvenc_available(codec: str = "h264_nvenc") -> bool:
    """True when ``codec`` actually encodes here (listed AND a working NVIDIA GPU/driver)."""
    try:
        result = subprocess.run(
            [
                find_tool("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.2",
                "-c:v",
                codec,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
