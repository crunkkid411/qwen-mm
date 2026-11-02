"""EditSpec -> one FFmpeg command. ``build_command`` is pure; ``render`` probes, runs, and records."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from shared.syscmd import find_tool

from .probe import MediaInfo, nvenc_available, probe
from .spec import EditSpec

_ALIGN = {"bottom": 2, "center": 5, "top": 8}
_A_FMT = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"


@dataclass
class Plan:
    """A resolved render: argv, the working directory to run it in, and files to write first."""

    argv: list[str]
    cwd: str
    duration: float
    encoder: str
    files: dict[str, str] = field(default_factory=dict)  # absolute path -> text content


def _f(x: float) -> str:
    """Stable decimal formatting so identical specs yield byte-identical commands."""
    return f"{x:.6f}".rstrip("0").rstrip(".") or "0"


def _atempo(speed: float) -> list[str]:
    parts, s = [], speed
    while s > 2.0:
        parts.append("atempo=2")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-9:
        parts.append(f"atempo={_f(s)}")
    return parts


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def captions_to_srt(captions) -> str:
    blocks = []
    for i, c in enumerate(sorted(captions, key=lambda c: (c.start, c.end)), 1):
        blocks.append(f"{i}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text.strip()}\n")
    return "\n".join(blocks)


def clip_durations(spec: EditSpec, infos: list[MediaInfo]) -> list[float]:
    out = []
    for i, (clip, info) in enumerate(zip(spec.clips, infos)):
        if info.is_image:
            if clip.end is None:
                raise ValueError(f"clip {i}: still image {clip.src} needs an explicit 'end' (its duration)")
            src_len = clip.end - clip.start
        else:
            if not info.has_video:
                raise ValueError(f"clip {i}: {clip.src} has no video stream")
            end = clip.end if clip.end is not None else info.duration
            if end is None:
                raise ValueError(f"clip {i}: cannot determine the duration of {clip.src}; set 'end'")
            if info.duration is not None and end > info.duration + 0.05:
                raise ValueError(f"clip {i}: end {end}s is past the source end ({info.duration}s)")
            src_len = end - clip.start
        if src_len <= 0:
            raise ValueError(f"clip {i}: empty range")
        out.append(src_len / clip.speed)
    return out


def _encoder_args(encoder: str, crf: int, preset: str | None) -> list[str]:
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", preset or "medium", "-crf", str(crf)]
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", preset or "medium", "-crf", str(crf), "-tag:v", "hvc1"]
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        extra = ["-tag:v", "hvc1"] if encoder == "hevc_nvenc" else []
        return ["-c:v", encoder, "-preset", preset or "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0", *extra]
    raise ValueError(f"unknown encoder {encoder!r}")


def resolve_encoder(requested: str) -> str:
    if requested != "auto":
        return requested
    return "h264_nvenc" if nvenc_available("h264_nvenc") else "libx264"


def build_command(spec: EditSpec, infos: list[MediaInfo], encoder: str) -> Plan:
    """Translate a validated spec + probed sources into an FFmpeg argv. Pure and deterministic."""
    if len(infos) != len(spec.clips):
        raise ValueError("one MediaInfo per clip is required")
    out = spec.output
    out_path = Path(out.path).expanduser().resolve()
    W, H, fps = out.width, out.height, _f(out.fps)
    durs = clip_durations(spec, infos)

    argv = [find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-y"]
    graph: list[str] = []

    for i, (clip, info, dur) in enumerate(zip(spec.clips, infos, durs)):
        if info.is_image:
            argv += ["-loop", "1", "-framerate", fps, "-t", _f(clip.end - clip.start), "-i", info.path]
            vtrim = ""
        else:
            argv += ["-i", info.path]
            end = clip.end if clip.end is not None else info.duration
            vtrim = f"trim=start={_f(clip.start)}:end={_f(end)},"
        v = (
            f"[{i}:v]{vtrim}setpts=(PTS-STARTPTS)/{_f(clip.speed)},"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p"
        )
        if clip.fade_in:
            v += f",fade=t=in:st=0:d={_f(clip.fade_in)}"
        if clip.fade_out:
            v += f",fade=t=out:st={_f(max(dur - clip.fade_out, 0))}:d={_f(clip.fade_out)}"
        graph.append(f"{v},settb=AVTB[v{i}]")

        if info.has_audio and clip.volume > 0:
            a = [f"[{i}:a]atrim=start={_f(clip.start)}:end={_f(clip.start + dur * clip.speed)}", "asetpts=PTS-STARTPTS"]
            a += _atempo(clip.speed)
            if clip.volume != 1.0:
                a.append(f"volume={_f(clip.volume)}")
        else:
            a = [f"anullsrc=r=48000:cl=stereo,atrim=duration={_f(dur)}", "asetpts=PTS-STARTPTS"]
        if clip.fade_in:
            a.append(f"afade=t=in:st=0:d={_f(clip.fade_in)}")
        if clip.fade_out:
            a.append(f"afade=t=out:st={_f(max(dur - clip.fade_out, 0))}:d={_f(clip.fade_out)}")
        a += [f"apad=whole_dur={_f(dur)}", f"atrim=duration={_f(dur)}", _A_FMT]
        graph.append(",".join(a) + f"[a{i}]")

    # Join clips.
    transitions = [c.transition for c in spec.clips[:-1]]
    if all(t.type == "cut" for t in transitions):
        pads = "".join(f"[v{i}][a{i}]" for i in range(len(durs)))
        graph.append(f"{pads}concat=n={len(durs)}:v=1:a=1[vj][aj]")
        total = sum(durs)
    else:
        cur_v, cur_a, total = "v0", "a0", durs[0]
        for i in range(1, len(durs)):
            t = transitions[i - 1]
            nv, na = f"vj{i}", f"aj{i}"
            if t.type == "cut":
                graph.append(f"[{cur_v}][{cur_a}][v{i}][a{i}]concat=n=2:v=1:a=1[{nv}][{na}]")
                total += durs[i]
            else:
                d = t.duration
                if d >= min(durs[i - 1], durs[i]):
                    raise ValueError(f"transition {i - 1}->{i} ({d}s) must be shorter than both clips")
                graph.append(f"[{cur_v}][v{i}]xfade=transition={t.type}:duration={_f(d)}:offset={_f(total - d)}[{nv}]")
                graph.append(f"[{cur_a}][a{i}]acrossfade=d={_f(d)}:c1=tri:c2=tri[{na}]")
                total += durs[i] - d
            cur_v, cur_a = nv, na
        graph.append(f"[{cur_v}]null[vj]")
        graph.append(f"[{cur_a}]anull[aj]")

    # Captions / subtitles. FFmpeg runs inside the output directory and references the subtitle
    # file by a plain basename, which sidesteps filter-graph escaping of drive letters and quotes.
    files: dict[str, str] = {}
    sub_name = None
    if spec.captions:
        if any(c.start >= total for c in spec.captions):
            raise ValueError(f"a caption starts after the end of the edit ({_f(total)}s)")
        sub_name = f"{_safe_stem(out_path)}.captions.srt"
        files[str(out_path.parent / sub_name)] = captions_to_srt(spec.captions)
    elif spec.subtitles_file:
        src = Path(spec.subtitles_file).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"subtitles_file not found: {src}")
        sub_name = f"{_safe_stem(out_path)}.subs{src.suffix.lower()}"
        files[str(out_path.parent / sub_name)] = src.read_text(encoding="utf-8-sig")
    if sub_name:
        st = spec.caption_style
        font = re.sub(r"[^A-Za-z0-9 _-]", "", st.font) or "Arial"
        style = f"FontName={font},FontSize={st.size},Alignment={_ALIGN[st.position]},Outline={_f(st.outline)}"
        graph.append(f"[vj]subtitles={sub_name}:force_style='{style}'[vout]")
    else:
        graph.append("[vj]null[vout]")

    # Music bed.
    audio = "aj"
    if spec.music:
        m = spec.music
        mi = len(spec.clips)
        mpath = Path(m.src).expanduser().resolve()
        if not mpath.is_file():
            raise FileNotFoundError(f"music not found: {mpath}")
        argv += ["-stream_loop", "-1", "-ss", _f(m.start), "-i", str(mpath)]
        chain = [f"[{mi}:a]atrim=duration={_f(total)}", "asetpts=PTS-STARTPTS", f"volume={_f(m.volume)}"]
        if m.fade_in:
            chain.append(f"afade=t=in:st=0:d={_f(m.fade_in)}")
        if m.fade_out:
            chain.append(f"afade=t=out:st={_f(max(total - m.fade_out, 0))}:d={_f(m.fade_out)}")
        graph.append(",".join([*chain, _A_FMT]) + "[mus]")
        if m.duck:
            graph.append("[aj]asplit=2[dlg][sc]")
            graph.append("[mus][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[musd]")
            graph.append("[dlg][musd]amix=inputs=2:duration=first:normalize=0[amx]")
        else:
            graph.append("[aj][mus]amix=inputs=2:duration=first:normalize=0[amx]")
        audio = "amx"

    if out.loudness is not None:
        graph.append(f"[{audio}]loudnorm=I={_f(out.loudness)}:TP=-1.5:LRA=11,{_A_FMT}[aout]")
    else:
        graph.append(f"[{audio}]anull[aout]")

    argv += ["-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "[aout]"]
    argv += _encoder_args(encoder, out.crf, out.preset)
    argv += ["-pix_fmt", "yuv420p", "-r", fps, "-c:a", "aac", "-b:a", out.audio_bitrate, "-ar", "48000"]
    argv += ["-t", _f(total), "-map_metadata", "-1", "-movflags", "+faststart", str(out_path)]
    return Plan(argv=argv, cwd=str(out_path.parent), duration=round(total, 3), encoder=encoder, files=files)


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or "edit"


def plan_render(spec: EditSpec) -> Plan:
    infos = [probe(c.src) for c in spec.clips]
    return build_command(spec, infos, resolve_encoder(spec.output.encoder))


def render(spec: EditSpec, *, dry_run: bool = False) -> Plan:
    """Render ``spec``. Writes ``<output>.edit.json`` (spec + exact command) beside the output."""
    plan = plan_render(spec)
    out_path = Path(plan.argv[-1])
    if dry_run:
        return plan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in plan.files.items():
        Path(path).write_text(text, encoding="utf-8")
    result = subprocess.run(plan.argv, cwd=plan.cwd, capture_output=True, text=True)
    if result.returncode != 0 and plan.encoder.endswith("_nvenc") and spec.output.encoder == "auto":
        # NVENC probed fine but failed on the real job (session limit, odd size): fall back once.
        plan = build_command(spec, [probe(c.src) for c in spec.clips], "libx264")
        result = subprocess.run(plan.argv, cwd=plan.cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr.strip()[-2000:]}")
    record = {"spec": spec.model_dump(mode="json"), "encoder": plan.encoder, "duration": plan.duration}
    record["ffmpeg"] = [Path(plan.argv[0]).name, *plan.argv[1:]]
    out_path.with_name(out_path.name + ".edit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return plan
