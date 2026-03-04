# antigravity_proxy.py - Antigravity Cloud Code proxy integration
# Routes requests through the antigravity-claude-proxy (github.com/badrisnarayanan/antigravity-claude-proxy)
# which exposes an Anthropic Messages API backed by Google Cloud Code.
#
import webbrowser
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
import sys
import json
import time
import logging
import shutil
import subprocess
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
CONFIG_ENDPOINT = "/api/config"

# Cached API key (read from proxy config file)
_cached_api_key: Optional[str] = None
_api_key_lock = threading.Lock()

# Module-level cancellation flag
_cancel_event = threading.Event()

# Module-level proxy subprocess tracking
_proxy_process: Optional[subprocess.Popen] = None
_proxy_launch_lock = threading.Lock()

# Auth browser tracking — only open the browser once per session
_auth_browser_opened = False
_auth_browser_lock = threading.Lock()


def _open_auth_browser_once(proxy_url: str, log_fn=None) -> bool:
    """Open the proxy auth URL in the browser, but only once per session.

    Returns True if the browser was opened (first call), False if already opened.
    """
    global _auth_browser_opened
    with _auth_browser_lock:
        if _auth_browser_opened:
            return False
        _auth_browser_opened = True
    _log = log_fn or (lambda msg: None)
    _log(f"🔐 Antigravity: Opening {proxy_url} in your browser for Google account linking...")
    try:
        webbrowser.open(proxy_url)
    except Exception:
        pass
    return True


def _get_proxy_api_key() -> str:
    """Read the proxy's API key from its config file.

    The antigravity-claude-proxy stores its config (including apiKey) in
    ~/.config/antigravity-proxy/config.json.  We read it and cache it so
    every request sends the matching key.

    Returns the key string, or empty string if no key is configured.
    """
    global _cached_api_key
    with _api_key_lock:
        if _cached_api_key is not None:
            return _cached_api_key

        config_path = os.path.join(
            os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
        )
        key = ""
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())
                key = data.get("apiKey", "") or ""
            except Exception:
                pass

        # If config file has no key, the proxy may still have one in memory.
        # Try to recover it by forcing a config save (the proxy writes the
        # full in-memory config, including apiKey, to disk on any valid save).
        if not key:
            try:
                proxy_url = get_proxy_url()
                resp = requests.get(
                    f"{proxy_url}{CONFIG_ENDPOINT}", timeout=3
                )
                if resp.status_code == 200:
                    cfg = resp.json().get("config", {})
                    live_key = cfg.get("apiKey", "")
                    if live_key and live_key != "":
                        # Proxy has a key in memory — force a config dump
                        # by toggling a harmless setting (debug).
                        debug_val = cfg.get("debug", False)
                        requests.post(
                            f"{proxy_url}{CONFIG_ENDPOINT}",
                            json={"debug": not debug_val},
                            timeout=3,
                        )
                        # Toggle it back immediately
                        requests.post(
                            f"{proxy_url}{CONFIG_ENDPOINT}",
                            json={"debug": debug_val},
                            timeout=3,
                        )
                        # Re-read the config file — it now has the key
                        if os.path.isfile(config_path):
                            with open(config_path, "r", encoding="utf-8") as f:
                                data = json.loads(f.read())
                            key = data.get("apiKey", "") or ""
                        if key:
                            logger.info(
                                "Antigravity: recovered API key from proxy config."
                            )
            except Exception:
                pass

        _cached_api_key = key
        return key


def invalidate_api_key_cache():
    """Clear the cached API key so it is re-read on next request."""
    global _cached_api_key
    with _api_key_lock:
        _cached_api_key = None


def _build_headers() -> Dict[str, str]:
    """Build the HTTP headers for a proxy request."""
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    key = _get_proxy_api_key()
    if key:
        headers["x-api-key"] = key
    return headers


def _wait_for_auth(
    url: str,
    payload: dict,
    headers: dict,
    proxy_url: str,
    log_fn=None,
    max_wait: int = 120,
    poll_interval: int = 5,
    stream: bool = False,
):
    """Open browser once and poll until authentication succeeds or timeout.

    Returns the successful requests.Response, or None if timed out.
    """
    _open_auth_browser_once(proxy_url, log_fn)
    _log = log_fn or (lambda msg: None)
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        if _cancel_event.is_set():
            return None
        _log(f"⏳ Waiting for authentication... ({elapsed}s / {max_wait}s)")
        try:
            retry_resp = requests.post(
                url, json=payload, headers=headers, timeout=30, stream=stream
            )
            if retry_resp.status_code not in (401, 403):
                return retry_resp
        except Exception:
            continue
    return None


def cancel_stream():
    """Signal any active Antigravity proxy stream to abort."""
    _cancel_event.set()


def reset_cancel():
    """Clear the cancellation flag before a new request."""
    _cancel_event.clear()
    reset_auth_browser()


def reset_auth_browser():
    """Reset the auth browser flag so the browser can be re-opened."""
    global _auth_browser_opened
    with _auth_browser_lock:
        _auth_browser_opened = False


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
# Auto-launch proxy
# ---------------------------------------------------------------------------

def _find_npx() -> Optional[str]:
    """Locate the npx executable on PATH or common install locations."""
    # Try PATH first
    npx = shutil.which("npx")
    if npx:
        return npx

    # On Windows, check common Node.js install locations
    if sys.platform == "win32":
        candidates = []
        for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA", "APPDATA"):
            base = os.environ.get(env_var, "")
            if base:
                candidates.append(os.path.join(base, "nodejs", "npx.cmd"))
                candidates.append(os.path.join(base, "fnm", "node-versions"))  # fnm
        # nvm-windows
        nvm_home = os.environ.get("NVM_HOME", "")
        if nvm_home:
            nvm_symlink = os.environ.get("NVM_SYMLINK", os.path.join(nvm_home, "..","nodejs"))
            candidates.append(os.path.join(nvm_symlink, "npx.cmd"))
        # Volta
        volta_home = os.environ.get("VOLTA_HOME", "")
        if volta_home:
            candidates.append(os.path.join(volta_home, "bin", "npx.cmd"))
        # Common default paths
        candidates.extend([
            os.path.expandvars(r"%PROGRAMFILES%\nodejs\npx.cmd"),
            os.path.expandvars(r"%APPDATA%\npm\npx.cmd"),
        ])
        for path in candidates:
            if os.path.isfile(path):
                return path

    return None


def _ensure_proxy_config():
    """Ensure the proxy config directory exists.

    Note: We intentionally do NOT modify the apiKey in the config file.
    The user sets their API key via the Antigravity desktop app or CLI,
    and we read it from the config file at runtime.
    """
    try:
        config_dir = os.path.join(os.path.expanduser("~"), ".config", "antigravity-proxy")
        os.makedirs(config_dir, exist_ok=True)
    except Exception:
        pass  # Non-critical


def ensure_proxy_running(log_fn=None) -> Dict[str, Any]:
    """Ensure the Antigravity proxy is running, auto-launching if needed.

    1. Checks health – if already running, returns immediately.
    2. Finds npx on PATH (or common Node.js install locations).
    3. Launches `npx -y antigravity-claude-proxy@latest start` in background.
    4. Waits up to 20s for the proxy to become healthy.

    Returns dict with 'running' (bool), 'auto_launched' (bool), and optional 'error'.
    """
    global _proxy_process

    _log = log_fn or (lambda msg: None)

    # Ensure proxy config disables API key auth (localhost doesn't need it)
    _ensure_proxy_config()

    # Already running?
    health = check_proxy_health()
    if health.get("healthy"):
        return {"running": True, "auto_launched": False}

    with _proxy_launch_lock:
        # Double-check after acquiring lock (another thread may have launched it)
        health = check_proxy_health()
        if health.get("healthy"):
            return {"running": True, "auto_launched": False}

        # If we already launched a process, check if it's still alive
        if _proxy_process is not None:
            if _proxy_process.poll() is None:
                # Process is still alive but not healthy yet – wait a bit
                _log("🌀 Antigravity proxy process is running, waiting for it to become healthy...")
                for _ in range(10):
                    time.sleep(2)
                    health = check_proxy_health()
                    if health.get("healthy"):
                        return {"running": True, "auto_launched": True}
                return {
                    "running": False,
                    "auto_launched": True,
                    "error": "Proxy was launched but did not become healthy within 20s."
                }
            else:
                # Process exited – clear it so we can try again
                _proxy_process = None

        # Find npx
        npx_path = _find_npx()
        if not npx_path:
            return {
                "running": False,
                "auto_launched": False,
                "error": (
                    "Node.js (npx) is not installed or not on PATH.\n"
                    "Install Node.js from https://nodejs.org/ then restart Glossarion,\n"
                    "or manually run: npx -y antigravity-claude-proxy@latest start"
                )
            }

        # Launch the proxy as a detached background process
        _log("🚀 Auto-launching Antigravity proxy...")
        try:
            # Build platform-appropriate launch args
            kwargs: Dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
            }

            # Ensure node.exe's directory is on PATH for the subprocess
            # (npx.cmd invokes "node" and needs it resolvable)
            npx_dir = os.path.dirname(npx_path)
            env = os.environ.copy()
            if npx_dir not in env.get("PATH", ""):
                env["PATH"] = npx_dir + os.pathsep + env.get("PATH", "")
            kwargs["env"] = env

            if sys.platform == "win32":
                # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS so it survives app close
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                DETACHED_PROCESS = 0x00000008
                kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
            else:
                kwargs["start_new_session"] = True

            cmd = [npx_path, "-y", "antigravity-claude-proxy@latest", "start"]
            _proxy_process = subprocess.Popen(cmd, **kwargs)
            _log(f"🌀 Proxy process started (PID {_proxy_process.pid}), waiting for it to become healthy...")

        except Exception as exc:
            return {
                "running": False,
                "auto_launched": False,
                "error": f"Failed to launch proxy: {exc}"
            }

        # Wait for it to become healthy (up to 20s)
        for attempt in range(20):
            time.sleep(1)
            # Check the process hasn't crashed
            if _proxy_process.poll() is not None:
                _proxy_process = None
                return {
                    "running": False,
                    "auto_launched": True,
                    "error": (
                        "Proxy process exited immediately. "
                        "Try running manually: npx -y antigravity-claude-proxy@latest start"
                    )
                }
            health = check_proxy_health()
            if health.get("healthy"):
                _log("✅ Antigravity proxy is now running!")
                return {"running": True, "auto_launched": True}

        return {
            "running": False,
            "auto_launched": True,
            "error": "Proxy launched but did not become healthy within 20s. Check the proxy logs."
        }


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
    model: str = "claude-sonnet-4-6",
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

    # Google Cloud Code caps output at ~64k tokens for this model
    if model == "claude-opus-4-6-thinking" and max_tokens > 64000:
        if log_fn:
            log_fn(f"⚠️ Antigravity: Capping max_tokens from {max_tokens} to 64000 (Cloud Code limit for {model})")
        max_tokens = 64000

    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = _build_headers()

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

    # Handle auth failure
    if resp.status_code in (401, 403):
        # Parse error to distinguish API key rejection from Google auth
        error_detail = ""
        try:
            err_json = resp.json()
            error_detail = err_json.get("error", {}).get("message", "")
        except Exception:
            error_detail = resp.text[:200]

        if "api key" in error_detail.lower() or "apikey" in error_detail.lower():
            # API key mismatch — clear cache & fail fast with a helpful message
            invalidate_api_key_cache()
            raise RuntimeError(
                f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                f"Open {proxy_url} → Settings and check the API Key, "
                f"or remove it to disable key validation."
            )

        # Otherwise it's likely a Google account auth issue → open browser
        auth_resp = _wait_for_auth(
            url, payload, headers, proxy_url, log_fn, stream=False
        )
        if auth_resp is not None and auth_resp.status_code == 200:
            resp = auth_resp
        else:
            raise RuntimeError(
                f"Antigravity: Authentication timed out.\n"
                f"Open {proxy_url} in your browser and link your Google account,\n"
                f"then try again."
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
    model: str = "claude-sonnet-4-6",
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

    if model == "claude-opus-4-6-thinking" and max_tokens > 64000:
        if log_fn:
            log_fn(f"⚠️ Antigravity: Capping max_tokens from {max_tokens} to 64000 (Cloud Code limit for {model})")
        max_tokens = 64000

    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = _build_headers()

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

    # Handle auth failure
    if resp.status_code in (401, 403):
        error_detail = ""
        try:
            err_json = resp.json()
            error_detail = err_json.get("error", {}).get("message", "")
        except Exception:
            error_detail = resp.text[:200]

        if "api key" in error_detail.lower() or "apikey" in error_detail.lower():
            invalidate_api_key_cache()
            raise RuntimeError(
                f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                f"Open {proxy_url} → Settings and check the API Key."
            )

        auth_resp = _wait_for_auth(
            url, payload, headers, proxy_url, log_fn, stream=True
        )
        if auth_resp is not None and auth_resp.status_code == 200:
            resp = auth_resp
        else:
            raise RuntimeError(
                f"Antigravity: Authentication timed out.\n"
                f"Open {proxy_url} in your browser and link your Google account,\n"
                f"then try again."
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
