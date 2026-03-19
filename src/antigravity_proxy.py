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
    """Auto-fetch the proxy's API key so requests are authenticated.

    Only reads from the live proxy's /api/config endpoint (most reliable).
    We do NOT read from the config file on disk because Glossarion settings
    are merged into the same file and can contain a stale apiKey.

    Returns the key string, or empty string if no key is configured.
    """
    global _cached_api_key
    with _api_key_lock:
        if _cached_api_key is not None:
            return _cached_api_key

        key = ""

        # Try live proxy endpoint (always up-to-date)
        try:
            proxy_url = get_proxy_url()
            resp = requests.get(
                f"{proxy_url}{CONFIG_ENDPOINT}", timeout=3
            )
            if resp.status_code == 200:
                cfg = resp.json().get("config", {})
                live_key = cfg.get("apiKey", "") or ""
                if live_key:
                    key = live_key
                    logger.info("Antigravity: API key fetched from live proxy.")
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


def _kill_proxy_by_port(port: int = 8080):
    """Kill any process listening on the proxy port (Windows & Unix)."""
    try:
        if sys.platform == "win32":
            # Find PID using netstat
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True, timeout=5)
        else:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=5)
    except Exception:
        pass


def restart_proxy(log_fn=None) -> Dict[str, Any]:
    """Kill the running proxy and relaunch it.
    
    Used when persistent auth failures indicate a stale API key.
    """
    global _proxy_process
    _log = log_fn or (lambda msg: None)
    _log("🔄 Antigravity: Restarting proxy to refresh API key...")

    # Kill tracked process
    if _proxy_process is not None:
        try:
            _proxy_process.terminate()
            _proxy_process.wait(timeout=5)
        except Exception:
            try:
                _proxy_process.kill()
            except Exception:
                pass
        _proxy_process = None

    # Also kill anything on port 8080 (the proxy might have been started
    # outside of Glossarion, e.g. via the Antigravity desktop app)
    proxy_url = get_proxy_url()
    try:
        port = int(proxy_url.rsplit(":", 1)[1].split("/")[0])
    except Exception:
        port = 8080
    _kill_proxy_by_port(port)

    # Clear apiKey from the proxy's own config file so it restarts
    # without requiring API key auth (fixes stale key after OAuth refresh)
    config_path = os.path.join(
        os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
    )
    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                proxy_cfg = json.load(f)
            if proxy_cfg.get("apiKey"):
                _log(f"🔑 Clearing stale API key from {config_path}")
                proxy_cfg["apiKey"] = ""
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(proxy_cfg, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"⚠️ Could not clear proxy API key: {exc}")

    # Also clear ANTHROPIC_AUTH_TOKEN from ~/.claude/settings.json
    # (Antigravity IDE stores its auth token here; stale values cause 401s)
    claude_settings_path = os.path.join(
        os.path.expanduser("~"), ".claude", "settings.json"
    )
    try:
        if os.path.isfile(claude_settings_path):
            with open(claude_settings_path, "r", encoding="utf-8") as f:
                claude_cfg = json.load(f)
            env_block = claude_cfg.get("env", {})
            if env_block.get("ANTHROPIC_AUTH_TOKEN"):
                _log(f"🔑 Clearing stale ANTHROPIC_AUTH_TOKEN from {claude_settings_path}")
                env_block["ANTHROPIC_AUTH_TOKEN"] = ""
                with open(claude_settings_path, "w", encoding="utf-8") as f:
                    json.dump(claude_cfg, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"⚠️ Could not clear ANTHROPIC_AUTH_TOKEN: {exc}")

    # Wait for port to free up
    time.sleep(2)

    # Clear API key cache so we fetch a fresh one
    invalidate_api_key_cache()

    # Relaunch
    return ensure_proxy_running(log_fn=log_fn)


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

    # Retry on transient 401/403 (OAuth token refresh race condition)
    resp = None
    for _auth_attempt in range(4):  # 1 initial + 3 retries
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

        if resp.status_code not in (401, 403) or _auth_attempt >= 3:
            break
        # Transient auth failure — wait for token refresh and retry
        wait = 2 * (_auth_attempt + 1)  # 2s, 4s, 6s
        if log_fn:
            log_fn(f"⏳ Antigravity: Auth error (HTTP {resp.status_code}), retrying in {wait}s... ({_auth_attempt + 1}/3)")
        time.sleep(wait)
        invalidate_api_key_cache()
        headers = _build_headers()  # refresh key
        # On last retry, also try without API key (localhost may accept)
        if _auth_attempt == 2:
            headers.pop("x-api-key", None)

    # All retries failed — restart the proxy as last resort
    if resp.status_code in (401, 403):
        restart_result = restart_proxy(log_fn=log_fn)
        if restart_result.get("running"):
            # Proxy restarted — rebuild headers with fresh key and retry once
            invalidate_api_key_cache()
            headers = _build_headers()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except Exception:
                pass  # fall through to existing error handling

    # Handle auth failure (after all retry/restart attempts)
    if resp.status_code in (401, 403):
        # Parse error to distinguish API key rejection from Google auth
        error_detail = ""
        try:
            err_json = resp.json()
            error_detail = err_json.get("error", {}).get("message", "")
        except Exception:
            error_detail = resp.text[:200]

        if "api key" in error_detail.lower() or "apikey" in error_detail.lower():
            # API key mismatch — clear cache & retry once with a fresh key
            invalidate_api_key_cache()
            fresh_key = _get_proxy_api_key()
            if fresh_key:
                headers["x-api-key"] = fresh_key
                try:
                    retry_resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                    if retry_resp.status_code not in (401, 403):
                        resp = retry_resp
                    else:
                        # Still failing — raise with helpful message
                        config_path = os.path.join(
                            os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
                        )
                        raise RuntimeError(
                            f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                            f"Fix: edit the apiKey in {config_path}\n"
                            f"or open {proxy_url} → Settings and remove/change the API Key."
                        )
                except requests.RequestException:
                    config_path = os.path.join(
                        os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
                    )
                    raise RuntimeError(
                        f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                        f"Fix: edit the apiKey in {config_path}\n"
                        f"or open {proxy_url} → Settings and remove/change the API Key."
                    )
            else:
                config_path = os.path.join(
                    os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
                )
                raise RuntimeError(
                    f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                    f"Fix: edit the apiKey in {config_path}\n"
                    f"or open {proxy_url} → Settings and remove/change the API Key."
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
    log_stream: bool = True,
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

    _log = log_fn or (lambda msg: None)
    _log(f"🌀 Antigravity: Streaming from proxy at {proxy_url} (model={model})")
    t_start = time.time()

    # Try httpx first (same as authgpt — streams SSE without buffering),
    # fall back to requests if httpx is not available.
    _httpx = None
    try:
        import httpx as _httpx
    except ImportError:
        pass

    use_httpx = _httpx is not None
    resp = None
    _httpx_ctx = None

    # Retry on transient 401/403 (OAuth token refresh race condition)
    for _auth_attempt in range(4):  # 1 initial + 3 retries
        if use_httpx:
            try:
                _timeout = _httpx.Timeout(timeout, connect=30.0)
                _httpx_ctx = _httpx.stream(
                    "POST", url, json=payload, headers=headers, timeout=_timeout
                )
                resp = _httpx_ctx.__enter__()
            except Exception as exc:
                if _httpx_ctx is not None:
                    try:
                        _httpx_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                exc_str = str(exc).lower()
                if "connect" in exc_str or "refused" in exc_str:
                    raise RuntimeError(
                        "Antigravity proxy connection refused. "
                        "Make sure the proxy is running:\n"
                        "  npx antigravity-claude-proxy@latest start"
                    )
                raise RuntimeError(
                    f"Antigravity proxy streaming error: {exc}"
                )
        else:
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

        if resp.status_code not in (401, 403) or _auth_attempt >= 3:
            break
        # Transient auth failure — close stream and retry after backoff
        if use_httpx and _httpx_ctx:
            try:
                _httpx_ctx.__exit__(None, None, None)
            except Exception:
                pass
            _httpx_ctx = None
        wait = 2 * (_auth_attempt + 1)  # 2s, 4s, 6s
        _log(f"⏳ Antigravity: Auth error (HTTP {resp.status_code}), retrying in {wait}s... ({_auth_attempt + 1}/3)")
        time.sleep(wait)
        invalidate_api_key_cache()
        headers = _build_headers()  # refresh key
        # On last retry, also try without API key (localhost may accept)
        if _auth_attempt == 2:
            headers.pop("x-api-key", None)
        resp = None

    # All retries failed — restart the proxy as last resort
    if resp.status_code in (401, 403):
        if use_httpx and _httpx_ctx:
            try:
                _httpx_ctx.__exit__(None, None, None)
            except Exception:
                pass
            _httpx_ctx = None
        restart_result = restart_proxy(log_fn=log_fn)
        if restart_result.get("running"):
            invalidate_api_key_cache()
            headers = _build_headers()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
                use_httpx = False  # switched to requests for restart retry
            except Exception:
                pass

    # Handle auth failure (after all retry/restart attempts)
    if resp.status_code in (401, 403):
        if use_httpx and _httpx_ctx:
            try:
                _httpx_ctx.__exit__(None, None, None)
            except Exception:
                pass

        error_detail = ""
        try:
            if use_httpx:
                error_detail = resp.read().decode("utf-8", errors="replace")
                try:
                    error_detail = json.loads(error_detail).get("error", {}).get("message", error_detail)
                except Exception:
                    pass
            else:
                err_json = resp.json()
                error_detail = err_json.get("error", {}).get("message", "")
        except Exception:
            try:
                error_detail = resp.text[:200] if not use_httpx else str(resp.read()[:200])
            except Exception:
                error_detail = ""

        if "api key" in error_detail.lower() or "apikey" in error_detail.lower():
            # API key mismatch — clear cache & retry once with a fresh key
            invalidate_api_key_cache()
            fresh_key = _get_proxy_api_key()
            if fresh_key:
                headers["x-api-key"] = fresh_key
                try:
                    retry_resp = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
                    if retry_resp.status_code not in (401, 403):
                        resp = retry_resp
                        use_httpx = False  # switched to requests for retry
                    else:
                        config_path = os.path.join(
                            os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
                        )
                        raise RuntimeError(
                            f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                            f"Fix: edit the apiKey in {config_path}\n"
                            f"or open {proxy_url} → Settings and remove/change the API Key."
                        )
                except requests.RequestException:
                    config_path = os.path.join(
                        os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
                    )
                    raise RuntimeError(
                        f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                        f"Fix: edit the apiKey in {config_path}\n"
                        f"or open {proxy_url} → Settings and remove/change the API Key."
                    )
            else:
                config_path = os.path.join(
                    os.path.expanduser("~"), ".config", "antigravity-proxy", "config.json"
                )
                raise RuntimeError(
                    f"Antigravity: API key rejected by proxy ({error_detail}).\n"
                    f"Fix: edit the apiKey in {config_path}\n"
                    f"or open {proxy_url} → Settings and remove/change the API Key."
                )

        auth_resp = _wait_for_auth(
            url, payload, headers, proxy_url, log_fn, stream=True
        )
        if auth_resp is not None and auth_resp.status_code == 200:
            resp = auth_resp
            use_httpx = False  # auth fallback always returns requests response
        else:
            raise RuntimeError(
                f"Antigravity: Authentication timed out.\n"
                f"Open {proxy_url} in your browser and link your Google account,\n"
                f"then try again."
            )

    if resp.status_code != 200:
        error_text = ""
        try:
            error_text = resp.text[:500] if not use_httpx else resp.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        if use_httpx and _httpx_ctx:
            try:
                _httpx_ctx.__exit__(None, None, None)
            except Exception:
                pass
        raise RuntimeError(f"Antigravity: {resp.status_code} - {error_text}")

    # Collect SSE events
    collected_content = []
    finish_reason = "stop"
    usage = None
    got_first_data = False
    log_buf: list = []
    # Thinking streaming state
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")
    ag_thinking_started = False
    ag_thinking_buf: list = []
    ag_thinking_chunks = 0
    ag_thinking_start_ts = None
    ag_current_block_type = None
    ag_thinking_flushed = False  # True once 0.5s passed and buffer was printed
    ag_thinking_deferred_lines: list = []  # holds lines until threshold
    _AG_THINKING_THRESHOLD = 0.5

    # httpx iter_lines() yields str directly; requests iter_lines needs chunk_size=1
    line_iter = resp.iter_lines() if use_httpx else resp.iter_lines(decode_unicode=True, chunk_size=1)

    for line in line_iter:
        if _cancel_event.is_set():
            resp.close()
            raise RuntimeError("Antigravity: stream cancelled by user")

        if not line or not line.startswith("data: "):
            continue

        if not got_first_data:
            got_first_data = True
            ttft = time.time() - t_start
            _log(f"📡 Antigravity: First token in {ttft:.1f}s, streaming…")

        data_str = line[6:]  # Strip "data: " prefix
        if data_str.strip() == "[DONE]":
            break

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        # Track block types for thinking detection
        if event_type == "content_block_start":
            cb = event.get("content_block", {})
            ag_current_block_type = cb.get("type", "")
            if ag_current_block_type == "thinking" and stream_thinking:
                if not ag_thinking_started:
                    ag_thinking_started = True
                    # Don't print header yet — defer until threshold

        elif event_type == "content_block_stop":
            if ag_current_block_type == "thinking" and stream_thinking and ag_thinking_started:
                if ag_thinking_flushed:
                    # Flush remaining thinking buffer
                    if ag_thinking_buf:
                        remaining = "".join(ag_thinking_buf)
                        if remaining.strip():
                            _log(f"    {remaining}")
                        ag_thinking_buf = []
                    thinking_dur = time.time() - ag_thinking_start_ts if ag_thinking_start_ts else 0
                    _log(f"🧠 [antigravity] Thinking complete ({ag_thinking_chunks} chunks, {thinking_dur:.1f}s)")
                # else: thinking was < 0.5s, discard silently
                ag_thinking_started = False
                ag_thinking_deferred_lines = []
            ag_current_block_type = None

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "thinking_delta":
                ag_thinking_chunks += 1
                if ag_thinking_start_ts is None:
                    ag_thinking_start_ts = time.time()
                thinking_text = delta.get("thinking", "")
                if thinking_text and log_stream and stream_thinking:
                    combined = "".join(ag_thinking_buf) + thinking_text
                    if "\n" in combined:
                        parts = combined.split("\n")
                        for part in parts[:-1]:
                            if ag_thinking_flushed:
                                _log(f"    {part}")
                            else:
                                ag_thinking_deferred_lines.append(part)
                        ag_thinking_buf = [parts[-1]]
                    else:
                        ag_thinking_buf.append(thinking_text)
                        if len("".join(ag_thinking_buf)) > 150:
                            if ag_thinking_flushed:
                                _log(f"    {''.join(ag_thinking_buf)}")
                            else:
                                ag_thinking_deferred_lines.append("".join(ag_thinking_buf))
                            ag_thinking_buf = []
                    # Check if threshold passed — flush deferred content
                    if not ag_thinking_flushed and ag_thinking_start_ts and (time.time() - ag_thinking_start_ts) >= _AG_THINKING_THRESHOLD:
                        ag_thinking_flushed = True
                        _log("🧠 [antigravity] Thinking...")
                        for dl in ag_thinking_deferred_lines:
                            _log(f"    {dl}")
                        ag_thinking_deferred_lines = []

            elif delta_type == "text_delta":
                text = delta.get("text", "")
                collected_content.append(text)
                # Real-time stream logging (HTML-tag-aware line buffering)
                if log_stream and text:
                    combined = "".join(log_buf) + text
                    for tag in ('</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</p>'):
                        combined = combined.replace(tag, tag + '\n')
                    if "\n" in combined:
                        parts = combined.split("\n")
                        for part in parts[:-1]:
                            _log(part)
                        log_buf = [parts[-1]]
                    else:
                        log_buf.append(text)
                        if len("".join(log_buf)) > 150:
                            _log("".join(log_buf))
                            log_buf = []

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

    # Flush remaining log buffer
    if log_stream and log_buf:
        remainder = "".join(log_buf).strip()
        if remainder:
            _log(remainder)

    # Clean up httpx context manager
    if use_httpx and _httpx_ctx:
        try:
            _httpx_ctx.__exit__(None, None, None)
        except Exception:
            pass

    content = "".join(collected_content)
    t_total = time.time() - t_start
    _log(f"📡 Antigravity: Stream finished in {t_total:.1f}s")
    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": None,
    }
