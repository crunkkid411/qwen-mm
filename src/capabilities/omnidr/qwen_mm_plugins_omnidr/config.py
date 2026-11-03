"""Qwen Omni configuration and streaming client."""

from shared import api_omni
from shared.api_openai import expand_video_frames
from shared.env import DEFAULT_DASHSCOPE_BASE_URL, DEFAULT_OMNI_MODEL, get_env, get_int_env
from shared.retry import retry_call

from .common import InputError, is_provider_policy_error, public_url


def omni_settings(model=None):
    omni_base = (get_env("QWEN_MM_OMNI_BASE_URL") or "").strip()
    if omni_base:
        # Another provider or a local server: its own key, or none for a keyless local server.
        base = omni_base.rstrip("/")
        key = (get_env("QWEN_MM_OMNI_API_KEY") or "").strip() or "EMPTY"
    else:
        base = (get_env("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL).strip().rstrip("/")
        key = (get_env("DASHSCOPE_API_KEY") or "").strip()
        if not key:
            raise InputError("Set DASHSCOPE_API_KEY (or QWEN_MM_OMNI_BASE_URL for another provider) first.")
    public_url(base)
    selected = (model or get_env("QWEN_MM_API_OMNI_MODEL") or DEFAULT_OMNI_MODEL).strip()
    if not selected:
        raise InputError("QWEN_MM_API_OMNI_MODEL must name an Omni-compatible model.")
    return base, key, selected


def chat_options(parts, asr=False):
    messages = [{"role": "user", "content": parts}]
    if not asr:
        return {"messages": messages}
    from .transcription import PROMPT

    return {"messages": [{"role": "system", "content": PROMPT}, *messages], "temperature": 0.0}


def call_omni(*, base_url, api_key, model, messages, timeout=None, temperature=None):
    """Stream an Omni response with explicit settings and bounded retries."""
    import openai

    timeout = timeout or get_int_env("QWEN_MM_CHAT_TIMEOUT")

    if any(size > api_omni.OMNI_MAX_B64_BYTES for size in api_omni.inline_b64_item_bytes(messages)):
        raise InputError("An inline media item exceeds 10 MB; reduce its size or inspect a shorter interval.")
    messages = api_omni.adapt_omni_messages(expand_video_frames(messages), api_omni.omni_protocol(base_url))
    request = {"temperature": temperature} if temperature is not None else {}
    client = openai.OpenAI(
        api_key=api_key, base_url=base_url, timeout=timeout, organization="", project="", max_retries=0
    )

    def complete():
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"modalities": ["text"]},
            **request,
        )
        parts, usage = [], None
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    parts.append(content)
        text = "".join(parts)
        if not text.strip():
            raise api_omni.EmptyCompletionError("The model returned no observations.")
        return text, usage

    def retryable(error):
        if is_provider_policy_error(error):
            return False
        return isinstance(
            error, (api_omni.EmptyCompletionError, openai.APIConnectionError, openai.APITimeoutError)
        ) or (isinstance(error, openai.APIStatusError) and error.status_code in {429, 500, 502, 503, 504})

    try:
        return retry_call(
            complete,
            attempts=4,
            base_backoff=1.0,
            mode="exp",
            cap=60.0,
            should_retry=retryable,
        )
    finally:
        client.close()


def chat(parts, mode="text", model=None, *, asr=False):
    """Return text and model using the same streaming protocol for every modality."""
    if mode not in {"text", "image", "frames", "video", "audio"}:
        raise InputError("Unsupported media mode.")
    if asr and mode != "audio":
        raise InputError("ASR requires audio mode.")
    base, key, selected = omni_settings(model)
    if isinstance(parts, str):
        parts = [{"type": "text", "text": parts}]
    text, _ = call_omni(
        base_url=base,
        api_key=key,
        model=selected,
        **chat_options(parts, asr),
    )
    if not text or not text.strip():
        raise InputError("The model returned no observations.")
    return text, selected
