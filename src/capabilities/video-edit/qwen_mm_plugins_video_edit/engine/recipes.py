"""Model-free recipes: analysis in, EditSpec out."""

from __future__ import annotations

from pathlib import Path

from .analyze import detect_silences, speech_spans
from .probe import probe
from .spec import EditSpec


def autocut(
    src: str,
    output: str,
    *,
    noise_db: float = -35.0,
    min_silence: float = 0.5,
    pad: float = 0.15,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> EditSpec:
    """Jump-cut ``src`` to its speech: drop every silence longer than ``min_silence``."""
    info = probe(src)
    if not info.has_audio or not info.duration:
        raise ValueError(f"{src} has no audio to cut on")
    keep = speech_spans(detect_silences(info.path, noise_db, min_silence, info.duration), info.duration, pad)
    if not keep:
        raise ValueError("no speech found above the noise floor; lower noise_db")
    w, h = width or info.width or 1920, height or info.height or 1080
    return EditSpec.model_validate(
        {
            "output": {
                "path": str(Path(output).expanduser().resolve()),
                "width": w - w % 2,
                "height": h - h % 2,
                "fps": fps or info.fps or 30,
            },
            "clips": [{"src": info.path, "start": s, "end": e} for s, e in keep],
        }
    )
