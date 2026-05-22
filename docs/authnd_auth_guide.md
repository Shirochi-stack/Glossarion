# AuthND Standalone User Guide

`src/authnd_auth.py` sends chat requests through NVIDIA Build's browser-backed route.
It does not use a NVIDIA API key. It opens the public model page with Qt WebEngine,
gets the same browser captcha token the page uses, then sends the NVIDIA Build
prediction request.

## Requirements

- Python environment with the Glossarion dependencies installed.
- `PySide6` / Qt WebEngine available for the browser token flow.
- `httpx` installed for best real-time SSE streaming. If missing, AuthND falls
  back to `requests`, but streaming may be less responsive.

## Basic Usage

Run from the repository root:

```powershell
python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --prompt "Reply with exactly: ok"
```

The final model output is printed to stdout. Progress logs are printed to stderr,
so you can redirect the answer cleanly:

```powershell
python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --prompt "Summarize NVIDIA Build." > answer.txt
```

## Prompt Files

```powershell
python src/authnd_auth.py --model z-ai/glm-5.1 --prompt-file prompt.txt
```

Read the prompt from stdin:

```powershell
Get-Content prompt.txt -Raw | python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --prompt -
```

Add a system prompt:

```powershell
python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --system "Be concise." --prompt "Explain SSE streaming."
```

## Messages JSON

Use `--messages` when you want full chat history:

```json
[
  {"role": "system", "content": "Be concise."},
  {"role": "user", "content": "What is AuthND?"}
]
```

Run:

```powershell
python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --messages messages.json
```

The file may also be an object with a `messages` list:

```json
{"messages": [{"role": "user", "content": "Hello"}]}
```

## Sampling Options

```powershell
python src/authnd_auth.py `
  --model deepseek-ai/deepseek-v4-flash `
  --prompt "Write one sentence." `
  --temperature 0.2 `
  --top-p 0.7 `
  --frequency-penalty 0 `
  --presence-penalty 0 `
  --max-tokens 128
```

## Streaming

Streaming is enabled by default unless `AUTHND_STREAM=0` is set.

Force streaming:

```powershell
python src/authnd_auth.py --stream --model deepseek-ai/deepseek-v4-flash --prompt "Write a short paragraph."
```

Disable streaming:

```powershell
python src/authnd_auth.py --no-stream --model deepseek-ai/deepseek-v4-flash --prompt "Write a short paragraph."
```

Hide streamed text chunks but keep progress logs:

```powershell
$env:AUTHND_LOG_STREAM_CHUNKS = "0"
python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --prompt "Write a short paragraph."
```

Suppress all progress logs:

```powershell
python src/authnd_auth.py --quiet --model deepseek-ai/deepseek-v4-flash --prompt "Reply with exactly: ok"
```

## JSON Output

Print the full result object:

```powershell
python src/authnd_auth.py --json --model deepseek-ai/deepseek-v4-flash --prompt "Reply with exactly: ok"
```

Write only the final content to a file:

```powershell
python src/authnd_auth.py --model deepseek-ai/deepseek-v4-flash --prompt-file prompt.txt --output answer.txt
```

## Useful Environment Variables

- `AUTHND_DEBUG=1`: print sanitized request metadata, parameter summaries, and response status.
- `AUTHND_STREAM=0`: default standalone and app calls to non-streaming mode.
- `AUTHND_LOG_STREAM_CHUNKS=0`: keep streaming transport but hide streamed text logs.
- `AUTHND_TOKEN_TIMEOUT=120`: browser captcha token timeout in seconds.
- `AUTHND_TIMEOUT=180`: request read timeout in seconds.
- `AUTHND_TOKEN_MODE=inline`: run the Qt token helper inline instead of subprocess.
- `AUTHND_DISABLE_SOFTWARE_RASTERIZER=1`: advanced workaround to disable Chromium software rendering. Leave unset on Linux unless debugging GPU-specific issues.
- `AUTHND_ENABLE_THINKING=1`: enable model-specific thinking where supported.
- `AUTHND_REASONING_EFFORT`, `GPT_EFFORT`, `REASONING_EFFORT`: map to AuthND thinking enablement.
- `AUTHND_ENABLE_STG_PREFIX=1`: legacy mode that sends `stg/<publisher>/<model>`.
- `AUTHND_LEGACY_EXTRA_HEADERS=1`: legacy mode that sends the older extra NVIDIA headers.

## Debugging

Use `--debug` for one run:

```powershell
python src/authnd_auth.py --debug --model deepseek-ai/deepseek-v4-flash --prompt "Reply with exactly: ok"
```

The expected log order is:

```text
AuthND: opening browser token flow ...
AuthND: captcha token acquired; sending NVIDIA request
API call in progress
AuthND: Stream opened ...
AuthND: First token ...
AuthND: Stream finished ...
```

If it pauses before `captcha token acquired`, the browser token flow is waiting.
If it pauses after `Stream opened`, the model or NVIDIA queue is the wait point.
