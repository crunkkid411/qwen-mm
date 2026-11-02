# Edit Engine — Deterministic FFmpeg Assembly From a Spec

The FFmpeg route has an executor: a JSON **edit spec** rendered by
`render_edit` (MCP) or the CLI below. Write the spec; do not hand-build
`-filter_complex` strings. The same spec always produces the same FFmpeg
command and byte-identical output, and the command is recorded beside the
render as `<output>.edit.json`.

No model or API key is involved. Any harness and any main model (local or
remote) can drive it, and it also runs with no agent at all.

## Tools

| Step | MCP tool | CLI (`python -m qwen_mm_plugins_video_edit.engine ...`) |
|------|----------|----------------------------------------------------------|
| Measure footage: probe, shot changes, silences, speech spans, optional local Whisper transcript | `analyze_footage` | `analyze a.mp4 b.mp4 [--transcript] -o analysis.json` |
| Model-free jump cut (keep speech, drop silences) | — | `autocut talk.mp4 -o talk_cut.mp4 --render` |
| One-shot planning by ANY OpenAI-compatible model | — (the host agent is the planner) | `plan analysis.json -b "<brief>" -o out.mp4 --base-url http://127.0.0.1:8080/v1 --model <id>` |
| Render | `render_edit` (`dry_run` prints the command) | `render edit.json [--dry-run]` |
| Spec schema | — | `schema` |

## Spec at a glance

```json
{
  "output": {"path": "edit/final.mp4", "width": 1920, "height": 1080, "fps": 30,
             "encoder": "auto", "crf": 20, "loudness": -14},
  "clips": [
    {"src": "a.mp4", "start": 3.2, "end": 9.8, "transition": {"type": "fade", "duration": 0.4}},
    {"src": "b.mp4", "start": 0, "end": 4, "speed": 1.5},
    {"src": "title.png", "end": 2, "fade_in": 0.3}
  ],
  "captions": [{"text": "Day one", "start": 0.2, "end": 2.0}],
  "music": {"src": "bgm.mp3", "volume": 0.3, "duck": true, "fade_out": 2}
}
```

- Clip times are in SOURCE seconds; caption times are on the OUTPUT timeline.
- `transition` belongs to the clip it leaves: `cut` or an FFmpeg `xfade` name.
- `encoder: auto` picks NVENC when a working NVIDIA GPU is present, else libx264.
- Relative paths in a spec file resolve against the spec file's directory.

## Where it fits

It is the FFmpeg route in `README.md`: mechanical deliverables, trims and
conforms, subtitle burns, music beds, loudness finishing, and prep renders.
Designed deliverables still follow the HyperFrames handoff rule; the engine
does not lift `[hyperframes-handoff]`.
