# antigravity_proxy.py - Antigravity Cloud Code proxy integration
# Routes requests through the antigravity-claude-proxy (github.com/badrisnarayanan/antigravity-claude-proxy)
# which exposes an Anthropic Messages API backed by Google Cloud Code.
#
# Usage: prefix models with 'antigravity/' (e.g., antigravity/claude-sonnet-4-5, antigravity/gemini-3-flash)
"""
Antigravity Proxy adapter for Glossarion.

The antigravity-claude-proxy runs as a local Node.js server (default: http://localhost:8080)
and exposes an Anthropic-compatible Messages API backed by Google Cloud Code.

Supported models (via the proxy):
  - Claude:  claude-sonnet-4-5, claude-sonnet-4-5-thinking, claude-opus-4-6-thinking
  - Gemini:  gemini-3-flash, gemini-3.1-pro-high, gemini-3.1-pro-low

Prerequisites:
  1. Install the proxy:  npm install -g antigravity-claude-proxy
  2. Start the proxy:   antigravity-claude-proxy start   (or: npx antigravity-claude-proxy@latest start)
  3. Link account:      Open http://localhost:8080 and add your Google account
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROXY_URL = "http://localhost:8080"
MESSAGES_ENDPOINT = "/v1/messages"
HEALTH_ENDPOINT = "/health"

# The proxy accepts any value as auth token (it uses its own Google OAuth)
DUMMY_AUTH_TOKEN = "antigravity-proxy"

# Module-level cancellation flag
_cancel_event = threading.Event()


def cancel_stream():
    """Signal any active Antigravity proxy stream to abort."""
    _cancel_event.set()


def reset_cancel():
    """Clear the cancellation flag before a new request."""
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


# ---------------------------------------------------------------------------
# Proxy URL resolution
# ---------------------------------------------------------------------------

def get_proxy_url() -> str:
    """Get the Antigravity proxy URL from env or default."""
    return os.environ.get("ANTIGRAVITY_PROXY_URL", DEFAULT_PROXY_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_proxy_health() -> Dict[str, Any]:
    """Check if the Antigravity proxy is running and healthy.
    
    Returns a dict with 'healthy' (bool) and optional 'details'.
    """
    try:
        url = f"{get_proxy_url()}{HEALTH_ENDPOINT}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            return {"healthy": True, "details": data}
        return {"healthy": False, "error": f"HTTP {resp.status_code}"}
    except requests.ConnectionError:
        return {"healthy": False, "error": "Connection refused – is the antigravity-claude-proxy running?"}
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------

def _convert_messages_to_anthropic(messages: List[Dict]) -> tuple:
    """Convert OpenAI-style messages to Anthropic Messages API format.
    
    Returns (system_prompt, anthropic_messages).
    """
    system_prompt = ""
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Anthropic takes system as a top-level parameter
            if system_prompt:
                system_prompt += "\n\n" + content
            else:
                system_prompt = content
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": content})
        else:
            # user, function, tool → user
            anthropic_messages.append({"role": "user", "content": content})

    # Ensure messages alternate user/assistant (Anthropic requirement)
    # Merge consecutive same-role messages
    merged = []
    for msg in anthropic_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    # Must start with user message
    if not merged or merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "Please continue."})

    return system_prompt, merged


# ---------------------------------------------------------------------------
# Send request (non-streaming)
# ---------------------------------------------------------------------------

def send_message(
    messages: List[Dict],
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
) -> Dict[str, Any]:
    """Send a message to the Antigravity proxy (Anthropic Messages API).
    
    Args:
        messages: OpenAI-format messages list
        model: Model name (without 'antigravity/' prefix)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        timeout: Request timeout in seconds
        log_fn: Optional logging function (e.g. print)
        
    Returns:
        Dict with keys: content, finish_reason, usage, raw_response
        
    Raises:
        RuntimeError on proxy errors
    """
    proxy_url = get_proxy_url()
    url = f"{proxy_url}{MESSAGES_ENDPOINT}"

    system_prompt, anthropic_messages = _convert_messages_to_anthropic(messages)

    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": DUMMY_AUTH_TOKEN,
        "anthropic-version": "2023-06-01",
    }

    if log_fn:
        log_fn(f"🌀 Antigravity: Sending to proxy at {proxy_url} (model={model})")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.ConnectionError:
        raise RuntimeError(
            "Antigravity proxy connection refused. "
            "Make sure the proxy is running:\n"
            "  npx antigravity-claude-proxy@latest start\n"
            "  Then open http://localhost:8080 and add your Google account."
        )
    except requests.Timeout:
        raise RuntimeError(
            f"Antigravity proxy request timed out after {timeout}s. "
            "The model may need more time for long translations."
        )

    if resp.status_code != 200:
        error_body = resp.text
        try:
            error_json = resp.json()
            error_msg = error_json.get("error", {}).get("message", error_body)
        except Exception:
            error_msg = error_body
        raise RuntimeError(
            f"Antigravity: {resp.status_code} - {error_msg}"
        )

    data = resp.json()

    # Extract content from Anthropic Messages API response
    content = ""
    if "content" in data and isinstance(data["content"], list):
        text_blocks = [
            block.get("text", "")
            for block in data["content"]
            if block.get("type") == "text"
        ]
        content = "".join(text_blocks)
    elif "content" in data and isinstance(data["content"], str):
        content = data["content"]

    finish_reason = data.get("stop_reason", "end_turn")
    # Normalize to OpenAI-style finish reasons
    if finish_reason == "end_turn":
        finish_reason = "stop"
    elif finish_reason == "max_tokens":
        finish_reason = "length"

    usage = None
    if "usage" in data:
        u = data["usage"]
        usage = {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
        }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": data,
    }


# ---------------------------------------------------------------------------
# Send request (streaming)
# ---------------------------------------------------------------------------

def send_message_stream(
    messages: List[Dict],
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
) -> Dict[str, Any]:
    """Send a streaming message to the Antigravity proxy.
    
    Collects all streamed chunks and returns once complete.
    Checks _cancel_event between chunks for cancellation support.
    
    Returns same format as send_message().
    """
    proxy_url = get_proxy_url()
    url = f"{proxy_url}{MESSAGES_ENDPOINT}"

    system_prompt, anthropic_messages = _convert_messages_to_anthropic(messages)

    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": DUMMY_AUTH_TOKEN,
        "anthropic-version": "2023-06-01",
    }

    if log_fn:
        log_fn(f"🌀 Antigravity: Streaming from proxy at {proxy_url} (model={model})")

    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=timeout, stream=True
        )
    except requests.ConnectionError:
        raise RuntimeError(
            "Antigravity proxy connection refused. "
            "Make sure the proxy is running:\n"
            "  npx antigravity-claude-proxy@latest start"
        )
    except requests.Timeout:
        raise RuntimeError(
            f"Antigravity proxy streaming request timed out after {timeout}s."
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Antigravity: {resp.status_code} - {resp.text[:500]}")

    # Collect SSE events
    collected_content = []
    finish_reason = "stop"
    usage = None

    for line in resp.iter_lines(decode_unicode=True):
        if _cancel_event.is_set():
            resp.close()
            raise RuntimeError("Antigravity: stream cancelled by user")

        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:]  # Strip "data: " prefix
        if data_str.strip() == "[DONE]":
            break

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                collected_content.append(delta.get("text", ""))

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            stop = delta.get("stop_reason", "")
            if stop:
                finish_reason = "stop" if stop == "end_turn" else (
                    "length" if stop == "max_tokens" else stop
                )
            if "usage" in delta:
                u = delta["usage"]
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                }

        elif event_type == "message_start":
            msg = event.get("message", {})
            if "usage" in msg:
                u = msg["usage"]
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                }

    content = "".join(collected_content)
    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": None,
    }
