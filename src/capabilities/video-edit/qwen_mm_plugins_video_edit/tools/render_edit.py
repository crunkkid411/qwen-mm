"""MCP tool: render a declarative edit spec with FFmpeg — deterministic, local, no model or API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from shared.content import text, text_error


class RenderEditArgs(BaseModel):
    spec: dict[str, Any]
    dry_run: bool = False


TOOL = {"name": "render_edit", "args": RenderEditArgs}


def handle(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Render a complete edit from a JSON edit spec in one local FFmpeg pass. No model or API call:
    the same spec always yields the same command. Handles trimming, speed, scaling/letterboxing,
    fades, cut or xfade transitions, burned-in captions or an .srt/.ass file, a ducked music bed,
    EBU R128 loudness, and NVENC (CUDA) encoding when a working NVIDIA GPU is present. Writes
    `<output>.edit.json` (spec + exact FFmpeg argv) beside the output for reproducibility. Use
    analyze_footage first to pick in/out points.

    Args:
        spec: Edit spec object. Shape: {"output": {"path", "width"=1920, "height"=1080, "fps"=30,
            "encoder"="auto", "crf"=20, "loudness"=-14}, "clips": [{"src", "start"=0, "end",
            "speed"=1, "volume"=1, "fade_in"=0, "fade_out"=0, "transition": {"type": "cut" | xfade
            name such as "fade"/"dissolve"/"wipeleft", "duration"=0.5}}], "captions": [{"text",
            "start", "end"}] (output-timeline seconds), "caption_style": {"font", "size",
            "position"}, "subtitles_file", "music": {"src", "volume"=0.3, "duck"=true, "fade_out"=2}}.
            Unknown keys are rejected. Still images are valid clips when "end" gives their duration.
        dry_run: Validate and return the exact FFmpeg command without rendering.
    """
    from pydantic import ValidationError

    from qwen_mm_plugins_video_edit.engine.render import render
    from qwen_mm_plugins_video_edit.engine.spec import EditSpec

    try:
        spec = EditSpec.model_validate(arguments["spec"])
    except ValidationError as exc:
        return text_error(f"invalid edit spec:\n{exc}")
    try:
        plan = render(spec, dry_run=arguments.get("dry_run", False))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return text_error(str(exc))
    if arguments.get("dry_run"):
        return [text(f"Duration: {plan.duration}s, encoder: {plan.encoder}\ncwd: {plan.cwd}\n" + " ".join(plan.argv))]
    out = plan.argv[-1]
    return [text(f"Rendered {out}\nDuration: {plan.duration}s\nEncoder: {plan.encoder}\nRecord: {out}.edit.json")]
