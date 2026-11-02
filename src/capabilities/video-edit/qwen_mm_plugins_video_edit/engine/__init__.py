"""Deterministic, model-agnostic edit engine for the video-edit capability.

The engine turns a declarative edit spec (JSON, validated by :class:`EditSpec`) into one FFmpeg
invocation. The same spec always produces the same command, so an edit is reproducible without any
model in the loop. Any model — local or remote, through any harness — only needs to write the spec;
the engine owns probing, filter-graph construction, encoder choice, and loudness.

Entry points:

- ``python -m qwen_mm_plugins_video_edit.engine`` (CLI: analyze / autocut / plan / render)
- the ``analyze_footage`` and ``render_edit`` MCP tools
"""

from .spec import EditSpec

__all__ = ["EditSpec"]
