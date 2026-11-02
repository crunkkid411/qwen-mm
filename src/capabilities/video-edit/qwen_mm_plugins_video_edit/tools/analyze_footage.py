"""MCP tool: deterministic footage analysis (probe, shot changes, silences, speech, local transcript)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from shared.content import json_text, text_error


class AnalyzeFootageArgs(BaseModel):
    paths: list[str] = Field(min_length=1)
    scene_threshold: float = Field(0.3, gt=0, lt=1)
    noise_db: float = -35.0
    min_silence: float = Field(0.5, gt=0)
    transcript: bool = False
    whisper_model: str = "small"
    language: Optional[str] = None


TOOL = {"name": "analyze_footage", "args": AnalyzeFootageArgs}


def handle(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Measure source footage locally so in/out points come from signal, not guesses. Returns JSON per
    file: stream facts (duration, size, fps, audio), FFmpeg scene-change timestamps, silence spans and
    the complementary padded speech spans (ready to use as render_edit clip ranges), and optionally a
    local faster-whisper transcript (CUDA when available). No model API is called.

    Args:
        paths: Local video, audio, or image files to analyze.
        scene_threshold: FFmpeg scene score (0-1) above which a frame counts as a shot change.
        noise_db: Audio level in dB below which audio counts as silence.
        min_silence: Minimum silence length in seconds.
        transcript: Also produce a timestamped local Whisper transcript (needs faster-whisper).
        whisper_model: faster-whisper model size or path, e.g. tiny, small, medium, large-v3.
        language: Spoken-language code for the transcript; omit to auto-detect.
    """
    from qwen_mm_plugins_video_edit.engine.analyze import analyze

    reports = []
    for path in arguments["paths"]:
        try:
            reports.append(
                analyze(
                    path,
                    scene_threshold=arguments.get("scene_threshold", 0.3),
                    noise_db=arguments.get("noise_db", -35.0),
                    min_silence=arguments.get("min_silence", 0.5),
                    transcript=arguments.get("transcript", False),
                    whisper_model=arguments.get("whisper_model", "small"),
                    language=arguments.get("language"),
                )
            )
        except (FileNotFoundError, RuntimeError) as exc:
            return text_error(str(exc))
    return [json_text(reports)]
