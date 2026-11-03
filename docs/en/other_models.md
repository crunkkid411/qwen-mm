# Using other models instead of DashScope

The audio and video tools (transcription, "what happens in this video", finding moments,
counting, music analysis, omni-memory, video-to-notes, ChatCut, deep research) all ask one
**omni model**: a model that can hear and see. Out of the box that is Qwen on DashScope. You can
point it at any other omni model, either on a website that sells API access or running on your own
PC.

## 1. Open the settings file

It is a plain text file. On Windows:

```text
C:\Users\<your name>\.qwen-mm-plugins\config
```

If the `.qwen-mm-plugins` folder or the `config` file does not exist, create it (Notepad is fine;
save as `config` with no `.txt` ending). One setting per line, `NAME=value`, no quotes.

## 2. Paste one block

**A model running on your own PC** (llama.cpp, LM Studio, vLLM, Ollama, …). Use the address and
model name your server shows when it starts:

```text
QWEN_MM_OMNI_BASE_URL=http://127.0.0.1:8080/v1
QWEN_MM_API_OMNI_MODEL=name-of-your-model
```

**A model from an online provider** (OpenRouter shown; any provider with an "OpenAI-compatible"
address works the same way). Use the model name exactly as the provider lists it:

```text
QWEN_MM_OMNI_BASE_URL=https://openrouter.ai/api/v1
QWEN_MM_OMNI_API_KEY=your-key-here
QWEN_MM_API_OMNI_MODEL=provider/model-name
```

**If the model can see images and hear audio but does not accept video files**, add:

```text
QWEN_MM_OMNI_VIDEO_INPUT=frames
```

The tools then send still frames with their timestamps plus the soundtrack.

**For the image tools as well** (`vision_chat`, OCR, finding objects in a picture), add these two
lines pointing at the same place. The first name says DASHSCOPE but it accepts any address:

```text
DASHSCOPE_BASE_URL=http://127.0.0.1:8080/v1
QWEN_MM_API_VL_MODEL=name-of-your-model
```

With an online provider, also add `DASHSCOPE_API_KEY=your-key-here` for those image tools.

## 3. Compare models

Change the `QWEN_MM_API_OMNI_MODEL=` line (and the address, if the model lives somewhere else),
save, and run the same request again. The file is re-read automatically; there is no need to restart
anything.

To check what an audio/video tool will call before it runs, ask your agent to run it with `dry_run` on. The
reply shows the `endpoint` and `model` it would use.

## What still needs DashScope

These call Alibaba-only services and have no setting to change them:

- generating images, video and speech: `qwen_image`, `wan_t2v`, `wan_s2v`, `happyhorse`, `qwen_tts`
  (`minimax_tts` uses MiniMax instead);
- `transcribe_audio` (Qwen3-ASR). Use `omni_asr_timestamped` instead: it transcribes with your
  omni model. The video-edit workflow already does;
- smart search inside omni-memory and video-memory. Without DashScope, omni-memory falls back to
  keyword search;
- the image and video providers that ChatCut's music-video workflow generates with.

Very large files: on DashScope they are uploaded temporarily. Everywhere else the tools shrink the
file or send frames instead, which happens automatically.
