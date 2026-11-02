"""CLI: ``python -m qwen_mm_plugins_video_edit.engine <analyze|autocut|plan|render|schema> ...``.

Typical model-free run::

    python -m qwen_mm_plugins_video_edit.engine autocut talk.mp4 -o talk_cut.mp4 --render

Typical planned run with a local model (llama.cpp / LM Studio / Ollama, any OpenAI-compatible URL)::

    python -m qwen_mm_plugins_video_edit.engine analyze a.mp4 b.mp4 > analysis.json
    python -m qwen_mm_plugins_video_edit.engine plan analysis.json -b "60s upbeat recap" -o out.mp4 \
        --base-url http://127.0.0.1:8080/v1 --model local > edit.json
    python -m qwen_mm_plugins_video_edit.engine render edit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # run from source without installing
    _pkg = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_pkg.parent), str(_pkg.parents[2])]
    __package__ = "qwen_mm_plugins_video_edit.engine"


def _emit(obj, out: str | None) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(out, file=sys.stderr)
    else:
        print(text)


def _render(spec, dry_run: bool) -> None:
    from qwen_mm_plugins_video_edit.engine.render import render

    plan = render(spec, dry_run=dry_run)
    if dry_run:
        print(json.dumps({"cwd": plan.cwd, "argv": plan.argv, "files": list(plan.files)}, indent=2))
    else:
        print(f"rendered {plan.argv[-1]} ({plan.duration}s, {plan.encoder})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    from qwen_mm_plugins_video_edit.engine.spec import EditSpec

    p = argparse.ArgumentParser(prog="qwen-mm-edit", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="probe + shot changes + silences/speech (+ local Whisper transcript)")
    a.add_argument("media", nargs="+")
    a.add_argument("--transcript", action="store_true", help="local faster-whisper transcript (CUDA if available)")
    a.add_argument("--whisper-model", default="small")
    a.add_argument("--language")
    a.add_argument("--scene-threshold", type=float, default=0.3)
    a.add_argument("--noise-db", type=float, default=-35.0)
    a.add_argument("--min-silence", type=float, default=0.5)
    a.add_argument("-o", "--out")

    c = sub.add_parser("autocut", help="model-free jump cut: keep speech, drop silences")
    c.add_argument("media")
    c.add_argument("-o", "--output", required=True, help="output video path")
    c.add_argument("--noise-db", type=float, default=-35.0)
    c.add_argument("--min-silence", type=float, default=0.5)
    c.add_argument("--pad", type=float, default=0.15)
    c.add_argument("--spec-out", help="write the spec here (default: stdout)")
    c.add_argument("--render", action="store_true", help="render immediately")

    pl = sub.add_parser("plan", help="one call to ANY OpenAI-compatible model -> validated edit spec")
    pl.add_argument("analysis", help="JSON from `analyze` (a list of reports)")
    pl.add_argument("-b", "--brief", required=True)
    pl.add_argument("-o", "--output", required=True, help="output video path to put in the spec")
    pl.add_argument("--model", help="model id (default: QWEN_MM_API_VL_MODEL)")
    pl.add_argument("--base-url", help="OpenAI-compatible base URL (default: DASHSCOPE_BASE_URL)")
    pl.add_argument("--api-key", help="default: the host's key env var, else 'EMPTY' (local servers)")
    pl.add_argument("--spec-out")

    r = sub.add_parser("render", help="render an edit spec with FFmpeg (deterministic)")
    r.add_argument("spec")
    r.add_argument("--dry-run", action="store_true", help="print the exact FFmpeg command and exit")

    sub.add_parser("schema", help="print the edit-spec JSON Schema")

    args = p.parse_args(argv)
    try:
        if args.cmd == "analyze":
            from qwen_mm_plugins_video_edit.engine.analyze import analyze

            reports = [
                analyze(
                    m,
                    scene_threshold=args.scene_threshold,
                    noise_db=args.noise_db,
                    min_silence=args.min_silence,
                    transcript=args.transcript,
                    whisper_model=args.whisper_model,
                    language=args.language,
                )
                for m in args.media
            ]
            _emit(reports, args.out)
        elif args.cmd == "autocut":
            from qwen_mm_plugins_video_edit.engine.recipes import autocut

            spec = autocut(args.media, args.output, noise_db=args.noise_db, min_silence=args.min_silence, pad=args.pad)
            if args.spec_out or not args.render:
                _emit(spec.model_dump(mode="json"), args.spec_out)
            if args.render:
                _render(spec, dry_run=False)
        elif args.cmd == "plan":
            from qwen_mm_plugins_video_edit.engine.plan import plan_edit

            analyses = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
            if isinstance(analyses, dict):
                analyses = [analyses]
            spec = plan_edit(
                args.brief, analyses, args.output, model=args.model, base_url=args.base_url, api_key=args.api_key
            )
            _emit(spec.model_dump(mode="json"), args.spec_out)
        elif args.cmd == "render":
            import os

            spec_path = Path(args.spec).resolve()
            spec = EditSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
            os.chdir(spec_path.parent)  # relative paths in a spec are relative to the spec file
            _render(spec, args.dry_run)
        elif args.cmd == "schema":
            _emit(EditSpec.model_json_schema(), None)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: one clear line, non-zero exit
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
