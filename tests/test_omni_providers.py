"""Offline tests: Omni tools run against any OpenAI-compatible omni provider, not only DashScope."""

from types import SimpleNamespace

import pytest

import shared.api_omni as omni
import shared.omni_media as om
import shared.retry as sr
from shared.env import DEFAULT_DASHSCOPE_BASE_URL

_OMNI_VARS = (
    "QWEN_MM_OMNI_BASE_URL",
    "QWEN_MM_OMNI_API_KEY",
    "QWEN_MM_OMNI_PROTOCOL",
    "QWEN_MM_OMNI_VIDEO_INPUT",
    "DASHSCOPE_BASE_URL",
    "DASHSCOPE_API_KEY",
    "OPENROUTER_API_KEY",
    "QWEN_MM_AUDIO_RAW_B64",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _OMNI_VARS:
        monkeypatch.delenv(name, raising=False)


def _messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}, "fps": 2, "max_pixels": 9},
                {"type": "input_audio", "input_audio": {"data": "data:;base64,QUJD", "format": "mp3"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]


def test_omni_endpoint_uses_its_own_url_and_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    monkeypatch.setenv("QWEN_MM_OMNI_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("QWEN_MM_OMNI_API_KEY", "local-key")
    assert omni.resolve_omni_endpoint({}) == ("http://127.0.0.1:8080/v1", "local-key")
    # An explicit tool argument still wins, and a different host does not receive the Omni key.
    assert omni.resolve_omni_endpoint({"base_url": "http://other/v1"}) == ("http://other/v1", "EMPTY")
    assert omni.resolve_omni_endpoint({"api_key": "explicit"})[1] == "explicit"


def test_omni_endpoint_without_key_is_keyless_for_local_servers(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    monkeypatch.setenv("QWEN_MM_OMNI_BASE_URL", "http://localhost:1234/v1")
    assert omni.resolve_omni_endpoint({}) == ("http://localhost:1234/v1", "EMPTY")


def test_omni_endpoint_uses_the_hosts_key_variable(monkeypatch):
    monkeypatch.setenv("QWEN_MM_OMNI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert omni.resolve_omni_endpoint({}) == ("https://openrouter.ai/api/v1", "or-key")


def test_omni_endpoint_defaults_to_dashscope(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    assert omni.resolve_omni_endpoint({}) == (DEFAULT_DASHSCOPE_BASE_URL, "ds-key")


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (DEFAULT_DASHSCOPE_BASE_URL, "dashscope"),
        ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "dashscope"),
        ("https://openrouter.ai/api/v1", "openai"),
        ("https://generativelanguage.googleapis.com/v1beta/openai", "openai"),
        ("http://127.0.0.1:8080/v1", "openai"),
    ],
)
def test_protocol_auto_detects_dashscope_hosts_only(base_url, expected):
    assert omni.omni_protocol(base_url) == expected


def test_protocol_can_be_forced(monkeypatch):
    monkeypatch.setenv("QWEN_MM_OMNI_PROTOCOL", "dashscope")
    assert omni.omni_protocol("http://127.0.0.1:8000/v1") == "dashscope"
    monkeypatch.setenv("QWEN_MM_OMNI_PROTOCOL", "openai")
    assert omni.omni_protocol(DEFAULT_DASHSCOPE_BASE_URL) == "openai"


def test_openai_protocol_sends_standard_parts_without_mutating_input():
    messages = _messages()
    adapted = omni.adapt_omni_messages(messages, "openai")
    video, audio, text = adapted[0]["content"]
    assert video == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
    assert audio["input_audio"] == {"data": "QUJD", "format": "mp3"}
    assert text["text"] == "describe"
    assert messages == _messages()


def test_dashscope_protocol_is_unchanged():
    assert omni.adapt_omni_messages(_messages(), "dashscope") == _messages()


def test_call_omni_sends_openai_format_to_other_hosts(monkeypatch):
    import openai

    seen = []

    class Completions:
        def create(self, **kwargs):
            seen.append(kwargs)
            delta = SimpleNamespace(content="ok")
            return [SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta)])]

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(sr.time, "sleep", lambda *_: None)

    omni.call_omni(base_url="https://openrouter.ai/api/v1", api_key="k", model="m", messages=_messages())
    video, audio, _ = seen[0]["messages"][0]["content"]
    assert "fps" not in video and audio["input_audio"]["data"] == "QUJD"

    omni.call_omni(base_url=DEFAULT_DASHSCOPE_BASE_URL, api_key="k", model="m", messages=_messages())
    assert seen[1]["messages"] == _messages()


def test_frames_mode_sends_frames_and_audio_instead_of_the_video(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_MM_OMNI_VIDEO_INPUT", "frames")
    calls = []

    def fake_frames(file_path, fps, max_pixels, cleanup, **kwargs):
        calls.append((file_path, fps, kwargs))
        return [{"type": "video", "video": ["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"]}]

    def never(*args, **kwargs):
        raise AssertionError("a whole video file must not be prepared in frames mode")

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    parts = om.local_video_parts(
        str(clip),
        1.0,
        200704,
        [],
        None,
        "any-omni",
        start_time=2.0,
        duration=5.0,
        preprocess_video_fn=never,
        frames_and_audio_parts_fn=fake_frames,
        transcode_and_upload_fn=never,
    )
    assert parts[0]["type"] == "video"
    assert calls == [(str(clip), 1.0, {"start_time": 2.0, "duration": 5.0})]


def test_omni_memory_follows_the_omni_endpoint(monkeypatch):
    from qwen_mm_plugins_omni_memory import config

    monkeypatch.setenv("QWEN_MM_OMNI_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("QWEN_MM_API_OMNI_MODEL", "local-omni")
    assert config.chat_config() == ("http://127.0.0.1:8080/v1", "local-omni", "EMPTY")
    monkeypatch.setenv("QWEN_MM_OMNI_API_KEY", "k")
    assert config.chat_config()[2] == "k"
