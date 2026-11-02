"""Qwen-MM-Plugins video-edit: a deterministic local edit engine (analyze_footage / render_edit) plus
image/video/audio generation tools (DashScope and MiniMax), paired with the ffmpeg-based editing skill.
"""

from mcp_framework import build_registry

__version__ = "1.2.0"

SPECS, get_handler, list_tools = build_registry(__name__, ["tools"])

# ffmpeg backs the skill and the local edit tools; generation-only users never need it, so report it at
# install time only (startup:False) and let analyze_footage / render_edit raise a clear error.
SYSTEM_DEPS = [
    {
        "label": "video-edit: local editing engine + skill (ffmpeg)",
        "extra": "video-edit",
        "tools": ["ffmpeg", "ffprobe"],
        "hint": "apt install ffmpeg   |   brew install ffmpeg",
        "startup": False,
    },
]
SYSTEM_DEPS_NOTE = (
    "  analyze_footage / render_edit run locally on ffmpeg (no API key). The generation tools\n"
    "  (image/tts/video) call remote APIs — no system tools needed for those."
)

USAGE_NOTE = (
    "Local, model-free editing: analyze_footage + render_edit (ffmpeg; NVENC when available).\n"
    "Same engine as a CLI: python -m qwen_mm_plugins_video_edit.engine --help\n"
    "Generation tools (qwen_image / qwen_tts / wan_s2v / wan_t2v / happyhorse) via DashScope.\n"
    "DashScope tools need DASHSCOPE_API_KEY; minimax_tts needs MINIMAX_API_KEY.\n"
    "Set provider keys via environment or ~/.qwen-mm-plugins/config."
)
