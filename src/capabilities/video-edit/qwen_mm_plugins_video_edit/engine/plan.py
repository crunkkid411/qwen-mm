"""Optional single-shot planner: any OpenAI-compatible chat model writes an EditSpec.

This is the ONLY model touchpoint in the engine, and it is optional — a person, a script, or the
agent already driving the harness can write the spec instead. It speaks plain chat completions, so it
works with llama.cpp ``llama-server``, LM Studio, Ollama, vLLM, OpenRouter, DashScope, OpenAI, etc.
The model sees text only (probe facts, shot changes, speech spans, transcript), so a small local text
model is enough. Its output is schema-validated; invalid JSON is fed back once for repair.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .spec import EditSpec

_SYSTEM = """You are a video editor. Produce an edit decision list as ONE JSON object that validates
against the JSON Schema below. Output JSON only — no prose, no code fences.

Rules:
- Use only source paths that appear in the footage analysis, with in/out points inside each source.
- Prefer cutting inside "speech" spans and on or near "scene_changes"; avoid cutting mid-word.
- Caption times are on the OUTPUT timeline (after cuts, speed changes, and transitions).
- Keep transitions short (0.3-0.8 s) and use "cut" unless the brief asks for something else.

JSON Schema:
"""


def _extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model reply contained no JSON object")
    return json.loads(text[start : end + 1])


def plan_edit(
    brief: str,
    analyses: list[dict],
    output: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_repairs: int = 1,
) -> EditSpec:
    from shared.api_openai import call_openai_chat, resolve_openai_endpoint, resolve_vl_model

    url, key = resolve_openai_endpoint({"base_url": base_url, "api_key": api_key})
    model = model or resolve_vl_model()
    schema = json.dumps(EditSpec.model_json_schema(), separators=(",", ":"))
    user = (
        f"Brief:\n{brief}\n\nOutput path (use exactly): {output}\n\n"
        f"Footage analysis:\n{json.dumps(analyses, ensure_ascii=False)}"
    )
    messages: list[dict] = [{"role": "system", "content": _SYSTEM + schema}, {"role": "user", "content": user}]
    error = ""
    for _ in range(max_repairs + 1):
        response = call_openai_chat(
            base_url=url,
            api_key=key,
            model=model,
            messages=messages,
            temperature=0,
            optional_extra_body={"response_format": {"type": "json_object"}},
        )
        reply = response.choices[0].message.content or ""
        try:
            data = _extract_json(reply)
            data.setdefault("output", {})["path"] = output
            return EditSpec.model_validate(data)
        except (ValueError, ValidationError) as exc:
            error = str(exc)
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That JSON is invalid:\n{error}\nReturn the corrected JSON only."},
            ]
    raise ValueError(f"model {model!r} did not produce a valid EditSpec: {error}")
