# antigravity_proxy.py - Antigravity Proxy integration
# Routes requests through frieser/antigravity-proxy:
# https://github.com/frieser/antigravity-proxy
#
# Glossarion keeps using model names prefixed with "antigravity/".
# This module translates them to the OpenAI-compatible model ids expected by
# frieser/antigravity-proxy and exposes the same Python API as the previous
# Antigravity adapter.

"""Antigravity Proxy adapter for Glossarion.

The current upstream proxy is an OpenAI-compatible local server. By default it
listens on http://localhost:3000 and exposes:

  - GET  /api/status
  - GET  /v1/models
  - POST /v1/chat/completions
  - GET  /oauth/start

Prerequisites:
  1. Install Bun from https://bun.sh/
  2. Start the proxy with: bunx antigravity-proxy@latest
  3. Open http://localhost:3000 and add a Google account

Glossarion can auto-launch the proxy when Bun/Bunx is on PATH. A Node/npx
fallback is attempted for environments where the package can bootstrap Bun.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROXY_URL = "http://localhost:3000"
PROXY_PACKAGE_NAME = "antigravity-proxy"
PROXY_PACKAGE = os.environ.get("ANTIGRAVITY_PROXY_PACKAGE", f"{PROXY_PACKAGE_NAME}@latest")
PROXY_GITHUB_REPO = "https://github.com/frieser/antigravity-proxy.git"

CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODELS_ENDPOINT = "/v1/models"
STATUS_ENDPOINT = "/api/status"
OAUTH_START_ENDPOINT = "/oauth/start"

PROXY_REPO_URL = "https://github.com/frieser/antigravity-proxy"

# Module-level cancellation flag.
_cancel_event = threading.Event()

# Module-level proxy subprocess tracking.
_proxy_process: Optional[subprocess.Popen] = None
_proxy_launch_lock = threading.Lock()

# Auth browser tracking - only open the browser once per session.
_auth_browser_opened = False
_auth_browser_lock = threading.Lock()


def _log_noop(_: str) -> None:
    return None


def _proxy_command_for_humans() -> str:
    return f"bunx {PROXY_PACKAGE}"


def _get_proxy_data_dir() -> str:
    """Return the stable data/config directory used for auto-launched proxy."""
    return os.environ.get(
        "ANTIGRAVITY_PROXY_DATA_DIR",
        os.path.join(os.path.expanduser("~"), ".config", "antigravity-proxy"),
    )


def _parse_semver(value: str) -> tuple:
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", value or "")
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def _version_to_str(version: tuple) -> str:
    return ".".join(str(part) for part in version)


def _run_version_probe(cmd: List[str], timeout: int = 12) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        if result.returncode == 0:
            value = (result.stdout or "").strip().splitlines()
            return value[-1].strip() if value else None
    except Exception:
        return None
    return None


def _latest_proxy_version() -> str:
    """Best-effort latest upstream version for dashboard/status metadata.

    npm currently publishes antigravity-proxy@0.7.0 while the GitHub repo has a
    newer v1.7.1 tag. Prefer the highest semantic version we can discover so
    the proxy does not render an "unsupported version" page because Glossarion
    launched it from a data directory.
    """
    override = os.environ.get("ANTIGRAVITY_PROXY_VERSION", "").strip().lstrip("v")
    if override:
        return override

    versions = []

    npm = _candidate_executable("npm")
    if npm:
        npm_version = _run_version_probe([npm, "view", PROXY_PACKAGE_NAME, "version", "--silent"])
        if npm_version:
            versions.append(_parse_semver(npm_version))

    git = _candidate_executable("git")
    if git:
        tags = _run_version_probe(
            [git, "ls-remote", "--tags", "--sort=v:refname", PROXY_GITHUB_REPO],
            timeout=20,
        )
        if tags:
            for line in tags.splitlines():
                if "^{}" in line:
                    continue
                tag = line.rsplit("/", 1)[-1].strip()
                versions.append(_parse_semver(tag))

    latest = max(versions) if versions else (0, 7, 0)
    return _version_to_str(latest)


def _write_proxy_runtime_package_json(data_dir: str) -> None:
    package_json = os.path.join(data_dir, "package.json")
    version = _latest_proxy_version()
    try:
        existing: Dict[str, Any] = {}
        if os.path.exists(package_json):
            with open(package_json, "r", encoding="utf-8") as f:
                existing = json.load(f)

        desired = {
            **existing,
            "name": PROXY_PACKAGE_NAME,
            "private": True,
            "version": version,
        }

        if existing != desired:
            with open(package_json, "w", encoding="utf-8") as f:
                json.dump(desired, f, indent=2)
    except Exception as exc:
        logger.debug("Could not update Antigravity proxy package.json: %s", exc)


def _ensure_proxy_config() -> str:
    """Ensure the auto-launched proxy has a stable working/data directory.

    frieser/antigravity-proxy stores config.json relative to process.cwd() and
    stores accounts in ACCOUNTS_FILE when that env var is set. We launch from a
    dedicated directory so Glossarion does not litter the repository root with
    proxy state.
    """
    data_dir = _get_proxy_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    _write_proxy_runtime_package_json(data_dir)

    return data_dir


def get_proxy_url() -> str:
    """Get the Antigravity proxy URL from env or default."""
    return os.environ.get("ANTIGRAVITY_PROXY_URL", DEFAULT_PROXY_URL).rstrip("/")


def _get_proxy_port() -> int:
    try:
        parsed = urlparse(get_proxy_url())
        if parsed.port:
            return parsed.port
    except Exception:
        pass
    return 3000


def _build_headers() -> Dict[str, str]:
    """Build HTTP headers for the OpenAI-compatible proxy."""
    return {
        "Content-Type": "application/json",
        # The upstream server currently does not validate this, but including an
        # OpenAI-shaped Authorization header keeps generic middleware happy.
        "Authorization": "Bearer sk-antigravity",
    }


def invalidate_api_key_cache() -> None:
    """Compatibility no-op for callers from the older adapter."""
    return None


# ---------------------------------------------------------------------------
# Cancellation and auth-browser helpers
# ---------------------------------------------------------------------------

def cancel_stream() -> None:
    """Signal any active Antigravity proxy stream to abort."""
    _cancel_event.set()


def reset_cancel() -> None:
    """Clear the cancellation flag before a new request."""
    _cancel_event.clear()
    reset_auth_browser()


def reset_auth_browser() -> None:
    """Reset the auth browser flag so the browser can be re-opened."""
    global _auth_browser_opened
    with _auth_browser_lock:
        _auth_browser_opened = False


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def _open_auth_browser_once(proxy_url: str, log_fn=None) -> bool:
    """Open the proxy OAuth URL in the browser, but only once per session."""
    global _auth_browser_opened
    with _auth_browser_lock:
        if _auth_browser_opened:
            return False
        _auth_browser_opened = True

    _log = log_fn or _log_noop
    auth_url = f"{proxy_url}{OAUTH_START_ENDPOINT}"
    _log(f"Antigravity: opening {auth_url} for Google account linking...")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Model and response conversion
# ---------------------------------------------------------------------------

def _normalize_model_name(model: str) -> str:
    """Convert Glossarion's model ids to frieser/antigravity-proxy ids.

    Glossarion strips the public "antigravity/" provider prefix before calling
    this module. This adapter always uses the proxy's explicit "antigravity-"
    namespace so requests stay on the upstream sandbox/agent route instead of
    accidentally falling into the CLI-pool Gemini route.
    """
    clean = (model or "").strip()
    lower = clean.lower()

    if lower.startswith("antigravity/"):
        clean = clean.split("/", 1)[1].lstrip("/")
        lower = clean.lower()

    if lower.startswith("antigravity-"):
        return clean

    # Claude, GPT-equivalent, Gemini, image, and thinking models should use the
    # explicit Antigravity model namespace from this adapter.
    if (
        lower.startswith(("claude-", "gpt-", "gemini-"))
        or "image" in lower
        or lower.startswith("anthropic/")
    ):
        return f"antigravity-{clean}"

    return clean


def _payload_for_openai_chat(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> Dict[str, Any]:
    return {
        "model": _normalize_model_name(model),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


def _parse_openai_chat_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an OpenAI Chat Completions response to Glossarion's shape."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content = message.get("content") or ""
    finish_reason = choice.get("finish_reason") or "stop"
    usage = data.get("usage")

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": data,
    }


def _extract_error_message(resp: requests.Response) -> str:
    try:
        data = resp.json()
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    except Exception:
        pass
    return (getattr(resp, "text", "") or "")[:1000]


def _health_details_have_accounts(details: Any) -> bool:
    if not isinstance(details, dict):
        return False
    accounts = details.get("accounts")
    return isinstance(accounts, list) and len(accounts) > 0


def _proxy_has_accounts() -> bool:
    health = check_proxy_health()
    return bool(health.get("healthy") and _health_details_have_accounts(health.get("details")))


def _should_wait_for_auth(resp: requests.Response) -> bool:
    """Return True when the request likely failed because no account is linked."""
    if resp.status_code in (401, 403):
        return not _proxy_has_accounts()

    if resp.status_code == 429:
        # Upstream returns a generic quota-exhausted 429 when there are no
        # accounts at all. Only treat that as auth setup when status confirms
        # an empty account list.
        return not _proxy_has_accounts()

    error_text = _extract_error_message(resp).lower()
    if "no account" in error_text or "add account" in error_text:
        return True

    return False


def _wait_for_auth(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    proxy_url: str,
    log_fn=None,
    max_wait: int = 180,
    poll_interval: int = 5,
    stream: bool = False,
    request_timeout: float = 300,
):
    """Open OAuth and poll until an account is linked or timeout is reached."""
    _open_auth_browser_once(proxy_url, log_fn)
    _log = log_fn or _log_noop

    _log("")
    _log("Antigravity: Google account link required.")
    _log(f"Open {proxy_url}, add a Google account, then keep this window open.")
    _log(f"Waiting for account linking to complete... (timeout: {max_wait}s)")

    elapsed = 0
    while elapsed < max_wait:
        if _cancel_event.is_set():
            return None

        time.sleep(poll_interval)
        elapsed += poll_interval

        if _proxy_has_accounts():
            _log("Antigravity: account detected, retrying request...")
            try:
                retry_resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                    stream=stream,
                )
                if retry_resp.status_code < 400:
                    return retry_resp
            except Exception:
                pass

        _log(f"Still waiting for Antigravity account linking... ({elapsed}s / {max_wait}s)")

    return None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _proxy_reports_unsupported(proxy_url: Optional[str] = None) -> bool:
    """Detect the upstream unsupported-version interstitial."""
    try:
        resp = requests.get(proxy_url or get_proxy_url(), timeout=5)
        text = (resp.text or "").lower()
        return "version of antigravity is no longer supported" in text
    except Exception:
        return False


def check_proxy_health() -> Dict[str, Any]:
    """Check if frieser/antigravity-proxy is running."""
    proxy_url = get_proxy_url()

    try:
        resp = requests.get(f"{proxy_url}{STATUS_ENDPOINT}", timeout=5)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if _proxy_reports_unsupported(proxy_url):
                return {"healthy": False, "error": "Antigravity proxy version is no longer supported.", "details": data}
            return {"healthy": True, "details": data}
        status_error = f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return {"healthy": False, "error": "Connection refused - is antigravity-proxy running?"}
    except Exception as exc:
        status_error = str(exc)

    # Fallback for future-compatible OpenAI-only deployments.
    try:
        resp = requests.get(f"{proxy_url}{MODELS_ENDPOINT}", headers=_build_headers(), timeout=5)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if _proxy_reports_unsupported(proxy_url):
                return {"healthy": False, "error": "Antigravity proxy version is no longer supported.", "details": data}
            return {"healthy": True, "details": data}
    except Exception:
        pass

    if _proxy_reports_unsupported(proxy_url):
        return {"healthy": False, "error": "Antigravity proxy version is no longer supported."}

    return {"healthy": False, "error": status_error}


# ---------------------------------------------------------------------------
# Auto-launch proxy
# ---------------------------------------------------------------------------

def _candidate_executable(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path

    if sys.platform == "win32":
        candidates: List[str] = []
        home = os.path.expanduser("~")
        candidates.extend(
            [
                os.path.join(home, ".bun", "bin", f"{name}.exe"),
                os.path.join(home, ".bun", "bin", f"{name}.cmd"),
                os.path.join(os.environ.get("APPDATA", ""), "npm", f"{name}.cmd"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "bun", f"{name}.exe"),
            ]
        )
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate

    return None


def _find_proxy_launch_command() -> Optional[List[str]]:
    """Return a launch command for the upstream proxy package."""
    override = os.environ.get("ANTIGRAVITY_PROXY_LAUNCH_CMD", "").strip()
    if override:
        try:
            return shlex.split(override, posix=(sys.platform != "win32"))
        except Exception:
            return override.split()

    bunx = _candidate_executable("bunx")
    if bunx:
        return [bunx, PROXY_PACKAGE]

    bun = _candidate_executable("bun")
    if bun:
        return [bun, "x", PROXY_PACKAGE]

    npx = _candidate_executable("npx")
    if npx:
        return [npx, "--yes", "--package", PROXY_PACKAGE, PROXY_PACKAGE_NAME]

    return None


def _proxy_launch_error() -> str:
    return (
        "Bun/Bunx is not installed or not on PATH.\n"
        "Install Bun from https://bun.sh/ then restart Glossarion,\n"
        f"or manually run: {_proxy_command_for_humans()}"
    )


def ensure_proxy_running(log_fn=None) -> Dict[str, Any]:
    """Ensure the Antigravity proxy is running, auto-launching if needed."""
    global _proxy_process

    _log = log_fn or _log_noop
    data_dir = _ensure_proxy_config()

    health = check_proxy_health()
    if health.get("healthy"):
        return {"running": True, "auto_launched": False}

    with _proxy_launch_lock:
        health = check_proxy_health()
        if health.get("healthy"):
            return {"running": True, "auto_launched": False}

        if "no longer supported" in str(health.get("error", "")).lower():
            _log("Antigravity proxy reports an unsupported version; replacing it with the latest available package...")
            _kill_proxy_by_port(_get_proxy_port())
            time.sleep(2)

        if _proxy_process is not None:
            if _proxy_process.poll() is None:
                _log("Antigravity proxy process is running, waiting for it to become healthy...")
                for _ in range(15):
                    time.sleep(2)
                    health = check_proxy_health()
                    if health.get("healthy"):
                        return {"running": True, "auto_launched": True}
                return {
                    "running": False,
                    "auto_launched": True,
                    "error": "Proxy was launched but did not become healthy within 30s.",
                }
            _proxy_process = None

        cmd = _find_proxy_launch_command()
        if not cmd:
            return {"running": False, "auto_launched": False, "error": _proxy_launch_error()}

        _log(f"Auto-launching Antigravity proxy with: {' '.join(cmd)}")

        try:
            env = os.environ.copy()
            env.setdefault("BASE_URL", get_proxy_url())
            env.setdefault(
                "ACCOUNTS_FILE",
                os.path.join(data_dir, "antigravity-accounts.json"),
            )

            kwargs: Dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
                "cwd": data_dir,
                "env": env,
            }

            if sys.platform == "win32":
                create_new_process_group = 0x00000200
                detached_process = 0x00000008
                try:
                    from shutdown_utils import subprocess_no_window_kwargs

                    kwargs.update(
                        subprocess_no_window_kwargs(
                            creationflags=create_new_process_group | detached_process
                        )
                    )
                except Exception:
                    kwargs["creationflags"] = create_new_process_group | detached_process
            else:
                kwargs["start_new_session"] = True

            _proxy_process = subprocess.Popen(cmd, **kwargs)
            _log(f"Antigravity proxy process started (PID {_proxy_process.pid}).")
        except Exception as exc:
            return {
                "running": False,
                "auto_launched": False,
                "error": f"Failed to launch proxy: {exc}",
            }

        for _ in range(30):
            time.sleep(1)
            if _proxy_process.poll() is not None:
                _proxy_process = None
                return {
                    "running": False,
                    "auto_launched": True,
                    "error": (
                        "Proxy process exited immediately. "
                        f"Try running manually: {_proxy_command_for_humans()}"
                    ),
                }
            health = check_proxy_health()
            if health.get("healthy"):
                _log("Antigravity proxy is now running.")
                return {"running": True, "auto_launched": True}

        return {
            "running": False,
            "auto_launched": True,
            "error": "Proxy launched but did not become healthy within 30s. Check the proxy logs.",
        }


def _kill_proxy_by_port(port: int = 3000) -> None:
    """Kill any process listening on the proxy port."""
    try:
        if sys.platform == "win32":
            try:
                from shutdown_utils import run_no_window
            except Exception:
                run_no_window = subprocess.run

            result = run_no_window(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    run_no_window(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
        else:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass


def restart_proxy(log_fn=None) -> Dict[str, Any]:
    """Kill the running proxy and relaunch it."""
    global _proxy_process
    _log = log_fn or _log_noop
    _log("Antigravity: restarting proxy...")

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

    _kill_proxy_by_port(_get_proxy_port())
    time.sleep(2)
    return ensure_proxy_running(log_fn=log_fn)


# ---------------------------------------------------------------------------
# Send request helpers
# ---------------------------------------------------------------------------

def _post_chat(payload: Dict[str, Any], timeout: float, stream: bool) -> requests.Response:
    return requests.post(
        f"{get_proxy_url()}{CHAT_COMPLETIONS_ENDPOINT}",
        json=payload,
        headers=_build_headers(),
        timeout=timeout,
        stream=stream,
    )


def _raise_connection_refused() -> None:
    raise RuntimeError(
        "Antigravity proxy connection refused. "
        "Make sure the proxy is running:\n"
        f"  {_proxy_command_for_humans()}\n"
        f"Then open {get_proxy_url()} and add your Google account."
    )


def _raise_for_proxy_status(resp: requests.Response) -> None:
    error_msg = _extract_error_message(resp)
    raise RuntimeError(f"Antigravity: HTTP {resp.status_code} - {error_msg}")


def send_message(
    messages: List[Dict],
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
) -> Dict[str, Any]:
    """Send a non-streaming message to frieser/antigravity-proxy."""
    proxy_url = get_proxy_url()
    url = f"{proxy_url}{CHAT_COMPLETIONS_ENDPOINT}"
    payload = _payload_for_openai_chat(messages, model, temperature, max_tokens, stream=False)

    _log = log_fn or _log_noop
    _log(f"Antigravity: sending to {proxy_url} (model={payload['model']})")

    try:
        resp = _post_chat(payload, timeout=timeout, stream=False)
    except requests.ConnectionError:
        _raise_connection_refused()
    except requests.Timeout:
        raise RuntimeError(
            f"Antigravity proxy request timed out after {timeout}s. "
            "The model may need more time for long translations."
        )

    if _should_wait_for_auth(resp):
        auth_resp = _wait_for_auth(
            url,
            payload,
            _build_headers(),
            proxy_url,
            log_fn,
            stream=False,
            request_timeout=timeout,
        )
        if auth_resp is not None:
            resp = auth_resp
        else:
            raise RuntimeError(
                f"Antigravity: authentication timed out.\n"
                f"Open {proxy_url} and add your Google account, then try again."
            )

    if resp.status_code != 200:
        _raise_for_proxy_status(resp)

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Antigravity: invalid JSON response from proxy: {exc}")

    return _parse_openai_chat_response(data)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _log_text_stream(text: str, log_buf: List[str], log_fn) -> None:
    if not text:
        return

    combined = "".join(log_buf) + text
    for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
        combined = combined.replace(tag, tag + "\n")

    if "\n" in combined:
        parts = combined.split("\n")
        for part in parts[:-1]:
            log_fn(part)
        log_buf[:] = [parts[-1]]
    else:
        log_buf[:] = [combined]
        if len(combined) > 150:
            log_fn(combined)
            log_buf.clear()


def _consume_openai_stream(resp: requests.Response, log_fn=None, log_stream: bool = True) -> Dict[str, Any]:
    _log = log_fn or _log_noop
    collected_content: List[str] = []
    finish_reason = "stop"
    usage = None
    got_first_data = False
    t_start = time.time()
    log_buf: List[str] = []
    thinking_buf: List[str] = []
    thinking_started = False
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    try:
        for line in resp.iter_lines(decode_unicode=True, chunk_size=1):
            if _cancel_event.is_set():
                resp.close()
                raise RuntimeError("Antigravity: stream cancelled by user")

            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not got_first_data:
                got_first_data = True
                _log(f"Antigravity: first token in {time.time() - t_start:.1f}s, streaming...")

            if event.get("usage"):
                usage = event.get("usage")

            choices = event.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content") or ""
            if reasoning and log_stream and stream_thinking:
                if not thinking_started:
                    thinking_started = True
                    _log("[antigravity] Thinking...")
                _log_text_stream(reasoning, thinking_buf, _log)

            text = delta.get("content") or ""
            if text:
                collected_content.append(text)
                if log_stream:
                    _log_text_stream(text, log_buf, _log)

            stop = choice.get("finish_reason")
            if stop:
                finish_reason = stop

        if log_stream and thinking_buf:
            remainder = "".join(thinking_buf).strip()
            if remainder:
                _log(f"    {remainder}")

        if log_stream and log_buf:
            remainder = "".join(log_buf).strip()
            if remainder:
                _log(remainder)

    finally:
        try:
            resp.close()
        except Exception:
            pass

    _log(f"Antigravity: stream finished in {time.time() - t_start:.1f}s")
    return {
        "content": "".join(collected_content),
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": None,
    }


def send_message_stream(
    messages: List[Dict],
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
    log_stream: bool = True,
) -> Dict[str, Any]:
    """Send a streaming message to frieser/antigravity-proxy."""
    proxy_url = get_proxy_url()
    url = f"{proxy_url}{CHAT_COMPLETIONS_ENDPOINT}"
    payload = _payload_for_openai_chat(messages, model, temperature, max_tokens, stream=True)

    _log = log_fn or _log_noop
    _log(f"Antigravity: streaming from {proxy_url} (model={payload['model']})")

    try:
        resp = _post_chat(payload, timeout=timeout, stream=True)
    except requests.ConnectionError:
        _raise_connection_refused()
    except requests.Timeout:
        raise RuntimeError(f"Antigravity proxy streaming request timed out after {timeout}s.")

    if _should_wait_for_auth(resp):
        try:
            resp.close()
        except Exception:
            pass
        auth_resp = _wait_for_auth(
            url,
            payload,
            _build_headers(),
            proxy_url,
            log_fn,
            stream=True,
            request_timeout=timeout,
        )
        if auth_resp is not None:
            resp = auth_resp
        else:
            raise RuntimeError(
                f"Antigravity: authentication timed out.\n"
                f"Open {proxy_url} and add your Google account, then try again."
            )

    if resp.status_code != 200:
        try:
            error_text = resp.text[:1000]
        except Exception:
            error_text = ""
        try:
            resp.close()
        except Exception:
            pass
        raise RuntimeError(f"Antigravity: HTTP {resp.status_code} - {error_text}")

    return _consume_openai_stream(resp, log_fn=log_fn, log_stream=log_stream)
