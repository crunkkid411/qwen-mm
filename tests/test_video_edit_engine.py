"""Offline tests for the deterministic video-edit engine (spec -> FFmpeg command, analysis helpers)."""

import json
import shutil
import subprocess
import types

import pytest
from pydantic import ValidationError

from qwen_mm_plugins_video_edit.engine import analyze, plan, render
from qwen_mm_plugins_video_edit.engine.probe import MediaInfo
from qwen_mm_plugins_video_edit.engine.spec import EditSpec


def _info(path, duration=10.0, audio=True, image=False):
    return MediaInfo(path, None if image else duration, 640, 360, None if image else 25.0, True, audio, image)


def _spec(tmp_path, **extra):
    data = {
        "output": {"path": str(tmp_path / "out.mp4"), "width": 1280, "height": 720, "fps": 30},
        "clips": [{"src": "a.mp4", "start": 1, "end": 5}, {"src": "b.mp4", "end": 3}],
    }
    data.update(extra)
    return EditSpec.model_validate(data)


@pytest.fixture(autouse=True)
def _fake_ffmpeg(monkeypatch):
    monkeypatch.setattr(render, "find_tool", lambda name: f"/usr/bin/{name}")


def _graph(plan_):
    return plan_.argv[plan_.argv.index("-filter_complex") + 1]


def test_spec_rejects_unknown_keys_and_bad_ranges():
    with pytest.raises(ValidationError):
        EditSpec.model_validate({"output": {"path": "o.mp4"}, "clips": [{"src": "a", "bogus": 1}]})
    with pytest.raises(ValidationError):
        EditSpec.model_validate({"output": {"path": "o.mp4"}, "clips": [{"src": "a", "start": 5, "end": 2}]})
    with pytest.raises(ValidationError):
        EditSpec.model_validate({"output": {"path": "o.mp4", "width": 1281}, "clips": [{"src": "a"}]})


def test_all_cuts_use_one_concat_and_command_is_deterministic(tmp_path):
    spec = _spec(tmp_path)
    infos = [_info("/m/a.mp4"), _info("/m/b.mp4", audio=False)]
    first = render.build_command(spec, infos, "libx264")
    second = render.build_command(spec, infos, "libx264")
    assert first.argv == second.argv
    assert first.duration == 7.0
    graph = _graph(first)
    assert "concat=n=2:v=1:a=1[vj][aj]" in graph
    assert "trim=start=1:end=5" in graph
    assert "anullsrc=r=48000" in graph  # silent clip gets a generated audio track
    assert "loudnorm=I=-14" in graph
    assert first.argv[-1] == str(tmp_path / "out.mp4")


def test_xfade_offsets_account_for_overlap(tmp_path):
    spec = _spec(tmp_path)
    spec.clips[0].transition.type = "dissolve"
    spec.clips[0].transition.duration = 0.5
    p = render.build_command(spec, [_info("/m/a.mp4"), _info("/m/b.mp4")], "libx264")
    assert "xfade=transition=dissolve:duration=0.5:offset=3.5" in _graph(p)
    assert "acrossfade=d=0.5" in _graph(p)
    assert p.duration == 6.5


def test_transition_longer_than_clip_is_rejected(tmp_path):
    spec = _spec(tmp_path)
    spec.clips[0].transition.type = "fade"
    spec.clips[0].transition.duration = 3.5
    with pytest.raises(ValueError, match="shorter than both clips"):
        render.build_command(spec, [_info("/m/a.mp4"), _info("/m/b.mp4")], "libx264")


def test_speed_and_still_images(tmp_path):
    spec = _spec(tmp_path, clips=[{"src": "a.mp4", "end": 4, "speed": 4}, {"src": "c.png", "end": 2}])
    p = render.build_command(spec, [_info("/m/a.mp4"), _info("/m/c.png", image=True)], "libx264")
    assert p.duration == 3.0
    assert "atempo=2,atempo=2" in _graph(p)
    assert ["-loop", "1"] == p.argv[p.argv.index("/m/c.png") - 7 : p.argv.index("/m/c.png") - 5]


def test_still_image_needs_duration(tmp_path):
    spec = _spec(tmp_path, clips=[{"src": "c.png"}])
    with pytest.raises(ValueError, match="explicit 'end'"):
        render.build_command(spec, [_info("/m/c.png", image=True)], "libx264")


def test_end_past_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="past the source end"):
        render.build_command(_spec(tmp_path), [_info("/m/a.mp4", duration=4), _info("/m/b.mp4")], "libx264")


def test_captions_become_srt_referenced_by_basename(tmp_path):
    spec = _spec(tmp_path, captions=[{"text": "Hi: 'there'", "start": 0, "end": 1.25}])
    p = render.build_command(spec, [_info("/m/a.mp4"), _info("/m/b.mp4")], "libx264")
    srt_path = str(tmp_path / "out.captions.srt")
    assert p.files[srt_path] == "1\n00:00:00,000 --> 00:00:01,250\nHi: 'there'\n"
    assert p.cwd == str(tmp_path)
    assert "subtitles=out.captions.srt:force_style='FontName=Arial" in _graph(p)


def test_music_bed_is_ducked_and_looped(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    spec = _spec(tmp_path, music={"src": str(song), "volume": 0.25})
    p = render.build_command(spec, [_info("/m/a.mp4"), _info("/m/b.mp4")], "libx264")
    assert "-stream_loop" in p.argv
    assert "sidechaincompress" in _graph(p)
    assert "[2:a]atrim=duration=7" in _graph(p)


@pytest.mark.parametrize(
    ("encoder", "expected"),
    [("libx264", ["-crf", "20"]), ("h264_nvenc", ["-cq", "20"]), ("hevc_nvenc", ["-tag:v", "hvc1"])],
)
def test_encoder_arguments(tmp_path, encoder, expected):
    p = render.build_command(_spec(tmp_path), [_info("/m/a.mp4"), _info("/m/b.mp4")], encoder)
    joined = " ".join(p.argv)
    assert f"-c:v {encoder}" in joined and " ".join(expected) in joined


def test_auto_encoder_prefers_working_nvenc(monkeypatch):
    monkeypatch.setattr(render, "nvenc_available", lambda codec: True)
    assert render.resolve_encoder("auto") == "h264_nvenc"
    monkeypatch.setattr(render, "nvenc_available", lambda codec: False)
    assert render.resolve_encoder("auto") == "libx264"
    assert render.resolve_encoder("libx265") == "libx265"


def test_speech_spans_complement_pad_and_merge():
    silences = [(1.0, 2.0), (2.1, 3.0), (8.0, 10.0)]
    assert analyze.speech_spans(silences, 10.0, pad=0.1) == [(0.0, 1.1), (1.9, 2.2), (2.9, 8.1)]
    assert analyze.speech_spans([], 4.0, pad=0.1) == [(0.0, 4.0)]


def test_detect_silences_parses_ffmpeg_log(monkeypatch):
    log = "[silencedetect] silence_start: 1.5\n[silencedetect] silence_end: 2.75 | silence_duration: 1.25\n"
    log += "[silencedetect] silence_start: 9.0\n"
    monkeypatch.setattr(analyze, "_ffmpeg_filter_log", lambda path, args: log)
    assert analyze.detect_silences("x", duration=10.0) == [(1.5, 2.75), (9.0, 10.0)]


def test_plan_repairs_invalid_json_once(monkeypatch, tmp_path):
    import shared.api_openai as api

    replies = iter(
        [
            '{"clips": [{"src": "a.mp4", "start": 3, "end": 1}]}',
            '```json\n{"output": {"path": "x"}, "clips": [{"src": "a.mp4", "end": 2}]}\n```',
        ]
    )
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        msg = types.SimpleNamespace(content=next(replies))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(api, "call_openai_chat", fake_chat)
    out = str(tmp_path / "o.mp4")
    spec = plan.plan_edit("cut it", [{"media": {"path": "a.mp4"}}], out, model="m", base_url="http://127.0.0.1:8080/v1")
    assert spec.output.path == out and spec.clips[0].end == 2
    assert len(calls) == 2 and calls[0]["base_url"] == "http://127.0.0.1:8080/v1" and calls[0]["api_key"] == "EMPTY"
    assert "That JSON is invalid" in calls[1]["messages"][-1]["content"]


def test_render_edit_tool_reports_invalid_spec():
    from qwen_mm_plugins_video_edit.tools import render_edit

    result = render_edit.handle({"spec": {"clips": []}})
    assert result[0]["text"].startswith("Error: invalid edit spec")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_real_render_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.undo()  # use the real ffmpeg
    src = tmp_path / "a.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=2", "-f", "lavfi",
         "-i", "sine=d=2", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(src)],
        check=True,
    )  # fmt: skip
    spec = EditSpec.model_validate(
        {
            "output": {"path": str(tmp_path / "o.mp4"), "width": 320, "height": 240, "fps": 25,
                       "encoder": "libx264", "preset": "ultrafast"},
            "clips": [{"src": str(src), "end": 1.2, "transition": {"type": "fade", "duration": 0.4}},
                      {"src": str(src), "start": 0.5}],
            "captions": [{"text": "hi", "start": 0, "end": 1}],
        }
    )  # fmt: skip
    render.render(spec)
    first = (tmp_path / "o.mp4").read_bytes()
    render.render(spec)
    assert (tmp_path / "o.mp4").read_bytes() == first
    record = json.loads((tmp_path / "o.mp4.edit.json").read_text())
    assert record["duration"] == 2.3 and record["encoder"] == "libx264"
