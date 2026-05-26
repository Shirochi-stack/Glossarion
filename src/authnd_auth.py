"""
authnd_auth.py - NVIDIA Build browser-backed chat route.

This module intentionally does not use the NVIDIA API key flow. It opens the
public build.nvidia.com model page in Qt WebEngine to obtain the same hCaptcha
token the page uses, then sends the hidden /v2/predict request with requests.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests


BUILD_BASE_URL = "https://build.nvidia.com"
API_BASE_URL = "https://api.ngc.nvidia.com"
DEFAULT_ORG_ID = "qc69jvmznzxy"
DEFAULT_HCAPTCHA_SITEKEY = "0c6a1e45-75d7-43cc-b836-a0c9d886b8ee"
DEFAULT_PUBLISHER = "z-ai"
DEFAULT_TIMEOUT = 180
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_cancel_event = threading.Event()
_thread_local = threading.local()
_metadata_cache: Dict[str, Dict[str, str]] = {}
_metadata_lock = threading.Lock()
_chat_template_unsupported_models: set = set()
_chat_template_unsupported_lock = threading.Lock()
_captcha_mint_lock = threading.Lock()
_active_helper_processes: set = set()
_active_helper_lock = threading.Lock()
_active_sessions: set = set()
_active_sessions_lock = threading.Lock()
_active_response_closers: set = set()
_active_response_lock = threading.Lock()


def _debug_enabled() -> bool:
    value = os.getenv("AUTHND_DEBUG", "").strip().lower()
    return value in ("1", "true", "yes", "on", "debug")


def _log(log_fn: Optional[Callable[[str], None]], message: str, *, debug_only: bool = False) -> None:
    if not log_fn:
        return
    if debug_only and not _debug_enabled():
        return
    log_fn(message)


def _short_error(error: Any, limit: int = 1200) -> str:
    text = str(error or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _terminate_process_tree(proc: Any, *, kill: bool = False) -> None:
    try:
        if proc is None or proc.poll() is not None:
            return
    except Exception:
        return
    if os.name == "nt":
        try:
            args = ["taskkill", "/T", "/PID", str(proc.pid)]
            if kill:
                args.insert(1, "/F")
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
            return
        except Exception:
            pass
    try:
        if kill:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


def _message_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    roles: Dict[str, int] = {}
    chars = 0
    images = 0
    for message in messages:
        role = str(message.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    chars += len(str(part or ""))
                    continue
                part_type = str(part.get("type") or "").lower()
                if part_type in ("image_url", "input_image", "image"):
                    images += 1
                elif part_type in ("text", "input_text"):
                    chars += len(str(part.get("text") or ""))
                else:
                    chars += len(str(part or ""))
        else:
            chars += len(str(content or ""))
    return {"count": len(messages), "roles": roles, "chars": chars, "images": images}


def _payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "messages": _message_summary(payload.get("messages") or []),
    }
    for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "reasoning_effort"):
        if key in payload:
            summary[key] = payload.get(key)
    if "chat_template_kwargs" in payload:
        summary["chat_template_kwargs"] = payload.get("chat_template_kwargs")
    return summary


def _stream_logging_enabled() -> bool:
    value = os.getenv("AUTHND_LOG_STREAM_CHUNKS")
    if value is None:
        value = os.getenv("LOG_STREAM_CHUNKS", "1")
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _stream_thinking_logging_enabled() -> bool:
    value = os.getenv("AUTHND_STREAM_THINKING_LOGS")
    if value is None:
        value = os.getenv("STREAM_THINKING_LOGS", "1")
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def cancel_stream() -> None:
    """Signal any active AuthND stream/request to stop."""
    _cancel_event.set()
    with _active_response_lock:
        closers = list(_active_response_closers)
    for closer in closers:
        try:
            closer()
        except Exception:
            pass
    with _active_sessions_lock:
        sessions = list(_active_sessions)
    for session in sessions:
        try:
            session.close()
        except Exception:
            pass
    with _active_helper_lock:
        helpers = list(_active_helper_processes)
    for proc in helpers:
        _terminate_process_tree(proc, kill=True)


def reset_cancel() -> None:
    """Clear the cancellation flag before a new request."""
    _cancel_event.clear()


def _is_cancelled() -> bool:
    return _cancel_event.is_set()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _build_page_model_slug(model_id: str) -> str:
    """Return the NVIDIA Build page slug for a model id.

    Build pages use underscore separators for numeric version components
    in several public model slugs (for example llama-3_1-70b-instruct),
    while the chat payload still expects the dotted model name.
    """
    model_id = str(model_id or "").strip("/")
    return re.sub(r"(?<=\d)\.(?=\d)", "_", model_id)


def _normalize_model(model: str) -> Tuple[str, str, str]:
    """
    Return (publisher, model_id, page_url) from authnd/z-ai/glm-5.1 or glm-5.1.
    """
    raw = (model or "").strip()
    raw = re.sub(r"^authnd\d{0,4}/", "", raw, flags=re.IGNORECASE)
    if raw.lower().startswith("authnd"):
        raw = raw[len("authnd"):].lstrip("/")
    raw = raw.strip("/")
    if not raw:
        raw = f"{DEFAULT_PUBLISHER}/glm-5.1"

    if "/" in raw:
        publisher, model_id = raw.split("/", 1)
        model_id = model_id.strip("/")
    else:
        publisher = os.getenv("AUTHND_DEFAULT_PUBLISHER", DEFAULT_PUBLISHER).strip("/") or DEFAULT_PUBLISHER
        model_id = raw

    page_url = f"{BUILD_BASE_URL}/{publisher}/{_build_page_model_slug(model_id)}"
    return publisher, model_id, page_url


def _payload_model_name(model_path: str) -> str:
    model = model_path.strip("/")
    if model.startswith("stg/"):
        return model
    if os.getenv("AUTHND_ENABLE_STG_PREFIX", "0").lower() in ("1", "true", "yes"):
        return f"stg/{model}"
    return model


def _resolve_model_metadata(page_url: str) -> Dict[str, str]:
    with _metadata_lock:
        cached = _metadata_cache.get(page_url)
        if cached:
            return dict(cached)

    response = requests.get(
        page_url,
        headers={"user-agent": USER_AGENT, "accept": "text/html,application/xhtml+xml"},
        timeout=30,
    )
    response.raise_for_status()
    html = response.text or ""

    def _match(patterns: Iterable[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return ""

    function_id = os.getenv("AUTHND_NVCF_FUNCTION_ID", "").strip() or _match((
        r'\\"nvcfFunctionId\\":\\"([^"\\]+)\\"',
        r'"nvcfFunctionId"\s*:\s*"([^"]+)"',
    ))
    artifact_name = _match((
        r'\\"artifactName\\":\\"([^"\\]+)\\"',
        r'"artifactName"\s*:\s*"([^"]+)"',
    ))
    payload_model = _match((
        r'\\"model\\"\s*:\s*\\"([^"\\]+)\\"',
        r'"model"\s*:\s*"([^"]+)"',
    ))
    namespace = os.getenv("AUTHND_NGC_ORG", "").strip("/") or _match((
        r'\\"namespace\\":\\"([^"\\]+)\\"',
        r'"namespace"\s*:\s*"([^"]+)"',
    )) or DEFAULT_ORG_ID

    endpoint_id = os.getenv("AUTHND_PREDICT_ID", "").strip() or artifact_name or function_id
    metadata = {
        "endpoint_id": endpoint_id,
        "function_id": function_id,
        "artifact_name": artifact_name,
        "namespace": namespace,
        "payload_model": payload_model,
    }
    with _metadata_lock:
        _metadata_cache[page_url] = dict(metadata)
    return metadata


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in (None, "text", "input_text"):
                    parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _image_part_url(item: Dict[str, Any]) -> str:
    image_url = item.get("image_url")
    image = item.get("image")
    url = ""
    mime = item.get("mime_type") or item.get("mimeType") or "image/png"

    if isinstance(image_url, dict):
        url = image_url.get("url") or image_url.get("data") or image_url.get("base64") or ""
        mime = image_url.get("mime_type") or image_url.get("mimeType") or mime
    elif isinstance(image_url, str):
        url = image_url

    if not url:
        if isinstance(image, dict):
            url = image.get("url") or image.get("data") or image.get("base64") or ""
            mime = image.get("mime_type") or image.get("mimeType") or mime
        elif isinstance(image, str):
            url = image

    if not url:
        url = item.get("url") or item.get("data") or item.get("base64") or ""
    url = str(url or "").strip()
    if not url:
        raise RuntimeError("AuthND image part is missing image_url.url/data/base64")
    if url.startswith(("http://", "https://", "data:image/")):
        return url
    return f"data:{mime};base64,{url}"


def _normalize_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            text = str(item or "")
            if text:
                parts.append({"type": "text", "text": text})
            continue

        item_type = str(item.get("type") or "").lower()
        if item_type in ("", "text", "input_text"):
            text = str(item.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text})
        elif item_type in ("image_url", "input_image", "image"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": _image_part_url(item)},
            })

    if not parts:
        return ""
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(str(part.get("text") or "") for part in parts if part.get("text"))
    return parts


def _prepend_system_content(content: Any, system_text: str) -> Any:
    prefix = f"System instructions:\n{system_text}"
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    if content:
        return f"{prefix}\n\n{content}"
    return prefix


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_parts: List[str] = []
    normalized: List[Dict[str, Any]] = []

    for message in messages or []:
        role = str(message.get("role", "user")).lower()
        raw_content = message.get("content")
        content = _content_to_text(raw_content) if role == "system" else _normalize_content(raw_content)
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        normalized.append({"role": role, "content": content})

    if system_parts:
        system_text = "\n\n".join(system_parts)
        for message in normalized:
            if message["role"] == "user":
                message["content"] = _prepend_system_content(message.get("content"), system_text)
                break
        else:
            normalized.insert(0, {"role": "user", "content": _prepend_system_content("", system_text)})

    if not normalized:
        normalized.append({"role": "user", "content": ""})
    return normalized


def _reasoning_toggle_enabled() -> bool:
    shared_toggle = os.getenv("ENABLE_GPT_THINKING")
    if shared_toggle is not None and shared_toggle.strip().lower() not in ("1", "true", "yes", "on", "enabled"):
        return False

    explicit = os.getenv("AUTHND_ENABLE_THINKING")
    if explicit is not None:
        if explicit.strip().lower() in ("0", "false", "no", "none", "off", "disabled"):
            return False
    return True


def _reasoning_effort() -> str:
    """Return the AuthND reasoning effort selected by shared GUI controls."""
    if not _reasoning_toggle_enabled():
        return "none"

    explicit = os.getenv("AUTHND_ENABLE_THINKING")
    shared_toggle = os.getenv("ENABLE_GPT_THINKING")
    effort = (
        os.getenv("AUTHND_REASONING_EFFORT")
        or os.getenv("GPT_EFFORT")
        or os.getenv("REASONING_EFFORT")
        or ""
    ).strip().lower()
    if effort in ("0", "false", "no", "none", "off", "disabled"):
        return "none"
    if effort in ("low", "medium", "high", "xhigh", "max", "heavy"):
        return effort

    return "medium" if explicit is not None or shared_toggle is not None else "none"


def _reasoning_control_configured() -> bool:
    return any(
        os.getenv(name) is not None
        for name in ("AUTHND_ENABLE_THINKING", "AUTHND_REASONING_EFFORT", "GPT_EFFORT", "REASONING_EFFORT", "ENABLE_GPT_THINKING")
    )


def _reasoning_enabled() -> bool:
    return _reasoning_control_configured() and _reasoning_effort() != "none"


def _is_mistral_authnd_model(model_path: str) -> bool:
    model_lower = (model_path or "").strip("/").lower()
    if not model_lower:
        return False
    publisher, _, model_id = model_lower.partition("/")
    return (
        publisher in ("mistral", "mistralai", "mistral-ai")
        or model_id.startswith(("mistral", "mixtral", "codestral", "devstral", "ministral", "magistral"))
    )


def _is_chat_template_unsupported_error(error: Any) -> bool:
    return "chat_template is not supported" in str(error or "").lower()


def _chat_template_cache_key(model_path: str) -> str:
    return (model_path or "").strip("/").lower()


def _model_requires_no_chat_template_kwargs(model_path: str) -> bool:
    if _is_mistral_authnd_model(model_path):
        return True
    key = _chat_template_cache_key(model_path)
    if not key:
        return False
    with _chat_template_unsupported_lock:
        return key in _chat_template_unsupported_models


def _remember_chat_template_unsupported_model(model_path: str) -> None:
    key = _chat_template_cache_key(model_path)
    if not key:
        return
    with _chat_template_unsupported_lock:
        _chat_template_unsupported_models.add(key)


def _apply_reasoning_payload(payload: Dict[str, Any], model_path: str) -> None:
    """
    NVIDIA NIM uses model-specific reasoning controls:
    - GPT-OSS supports top-level reasoning_effort: low/medium/high.
    - Nemotron 3 Nano supports chat_template_kwargs.parallel_reasoning_mode.
    - Other thinking models generally use chat_template_kwargs.enable_thinking.
    """
    if not _reasoning_control_configured():
        return

    effort = _reasoning_effort()
    reasoning_disabled = not _reasoning_toggle_enabled() or effort == "none"
    model_lower = (model_path or "").lower()
    if _model_requires_no_chat_template_kwargs(model_lower):
        return

    def _kwargs() -> Dict[str, Any]:
        existing = payload.setdefault("chat_template_kwargs", {})
        return existing if isinstance(existing, dict) else {}

    if reasoning_disabled:
        kwargs = _kwargs()
        kwargs["enable_thinking"] = False
        if "nemotron-3-nano" in model_lower:
            kwargs["parallel_reasoning_mode"] = "none"
        payload["chat_template_kwargs"] = kwargs
        return

    if "gpt-oss" in model_lower:
        payload["reasoning_effort"] = "high" if effort in ("xhigh", "max", "heavy") else effort
        return

    if "nemotron-3-nano" in model_lower:
        kwargs = _kwargs()
        kwargs["enable_thinking"] = True
        kwargs["parallel_reasoning_mode"] = "heavy" if effort in ("high", "xhigh", "max", "heavy") else effort
        payload["chat_template_kwargs"] = kwargs
        return

    if "deepseek-v4" in model_lower:
        payload["reasoning_effort"] = "max" if effort in ("xhigh", "max", "heavy") else "high"

    kwargs = _kwargs()
    kwargs.setdefault("enable_thinking", True)
    kwargs.setdefault("clear_thinking", False)
    payload["chat_template_kwargs"] = kwargs


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"user-agent": USER_AGENT})
        _thread_local.session = session
        with _active_sessions_lock:
            _active_sessions.add(session)
    return session


def _post_with_cancel(session: requests.Session, *args, **kwargs) -> requests.Response:
    """Run requests.post in a helper thread so AuthND hard-stop can return immediately.

    Closing a requests.Session from another thread does not reliably interrupt a
    blocking POST before a response object exists, especially on Windows. This
    wrapper lets the caller observe _cancel_event and raise while the socket is
    being torn down in the background.
    """
    result_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)

    def _run_post() -> None:
        try:
            response = session.post(*args, **kwargs)
        except BaseException as exc:
            try:
                result_queue.put_nowait(("error", exc))
            except Exception:
                pass
        else:
            try:
                result_queue.put_nowait(("response", response))
            except Exception:
                try:
                    response.close()
                except Exception:
                    pass

    worker = threading.Thread(target=_run_post, name="AuthNDRequest", daemon=True)
    worker.start()
    while True:
        try:
            kind, value = result_queue.get(timeout=0.1)
        except queue.Empty:
            if _is_cancelled():
                try:
                    session.close()
                except Exception:
                    pass
                raise RuntimeError("stream cancelled")
            continue
        if kind == "error":
            raise value
        if _is_cancelled():
            try:
                value.close()
            except Exception:
                pass
            raise RuntimeError("stream cancelled")
        return value


def _register_response_closer(closer: Callable[[], None]) -> Callable[[], None]:
    with _active_response_lock:
        _active_response_closers.add(closer)
    return closer


def _unregister_response_closer(closer: Callable[[], None]) -> None:
    with _active_response_lock:
        _active_response_closers.discard(closer)


def _extract_json_from_process(stdout: str) -> Dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("AuthND token helper did not return JSON")


def _is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def _qtwebengine_chromium_flags(existing: str = "") -> str:
    try:
        tokens = shlex.split(existing or "")
    except ValueError:
        tokens = (existing or "").split()

    # Linux users can run without usable GPU acceleration.  Disabling both GPU
    # and Chromium's software rasterizer can make Qt WebEngine fail the page
    # load before hCaptcha can be minted.
    if not _env_bool("AUTHND_DISABLE_SOFTWARE_RASTERIZER", False):
        tokens = [flag for flag in tokens if flag != "--disable-software-rasterizer"]

    required_flags = [
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
    ]
    seen = set()
    merged = []
    for flag in [*tokens, *required_flags]:
        if flag in seen:
            continue
        seen.add(flag)
        merged.append(flag)
    return " ".join(merged)


def _mint_captcha_token_subprocess(page_url: str, timeout: int) -> str:
    helper_timeout = max(30, int(timeout))
    if _is_frozen_app():
        # In PyInstaller builds sys.executable is the Glossarion exe, not a
        # Python interpreter.  Use a dedicated app-level helper argument so the
        # child process mints a token and exits instead of launching the GUI.
        cmd = [
            sys.executable,
            "--authnd-mint-token",
            page_url,
            "--timeout",
            str(helper_timeout),
        ]
    else:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mint-token",
            page_url,
            "--timeout",
            str(helper_timeout),
        ]
    env = os.environ.copy()
    env["AUTHND_TOKEN_HELPER"] = "1"
    env["QTWEBENGINE_CHROMIUM_FLAGS"] = _qtwebengine_chromium_flags(env.get("QTWEBENGINE_CHROMIUM_FLAGS", ""))
    env.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        creationflags=creationflags,
    )
    with _active_helper_lock:
        _active_helper_processes.add(proc)
    deadline = time.time() + helper_timeout + 20
    stdout = ""
    stderr = ""
    try:
        while proc.poll() is None:
            if _is_cancelled():
                _terminate_process_tree(proc, kill=True)
                raise RuntimeError("stream cancelled")
            if time.time() >= deadline:
                _terminate_process_tree(proc, kill=True)
                raise RuntimeError(f"AuthND token helper timed out after {helper_timeout}s")
            time.sleep(0.1)
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except Exception:
            stdout, stderr = "", ""
    finally:
        with _active_helper_lock:
            _active_helper_processes.discard(proc)
    if _is_cancelled() and proc.returncode:
        raise RuntimeError("stream cancelled")
    if proc.returncode != 0:
        try:
            result = _extract_json_from_process(stdout)
            error = str(result.get("error") or "").strip()
            if error:
                raise RuntimeError(f"AuthND token helper failed ({proc.returncode}): {error}")
        except RuntimeError as exc:
            if str(exc).startswith("AuthND token helper failed"):
                raise
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(f"AuthND token helper failed ({proc.returncode}): {detail[-1200:]}")
    result = _extract_json_from_process(stdout)
    token = str(result.get("token") or "").strip()
    if not token:
        raise RuntimeError(f"AuthND token helper returned no token: {result}")
    return token


def _mint_captcha_token_qt(page_url: str, timeout: int) -> str:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _qtwebengine_chromium_flags(
        os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    )
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(["authnd-token-helper"])
        created_app = True

    profile_root = os.path.join(
        os.path.expanduser("~"),
        ".glossarion",
        "authnd_browser",
        str(uuid.uuid4()),
    )
    os.makedirs(profile_root, exist_ok=True)

    profile = QWebEngineProfile(f"authnd-{uuid.uuid4().hex}", app)
    profile.setHttpUserAgent(USER_AGENT)
    try:
        profile.setPersistentStoragePath(profile_root)
        profile.setCachePath(os.path.join(profile_root, "cache"))
    except Exception:
        pass

    page = QWebEnginePage(profile, app)
    try:
        def run_js(script: str, js_timeout_ms: int = 15000) -> Any:
            holder: Dict[str, Any] = {}
            loop = QEventLoop()

            def _done(value: Any) -> None:
                holder["value"] = value
                loop.quit()

            page.runJavaScript(script, _done)
            QTimer.singleShot(js_timeout_ms, loop.quit)
            loop.exec()
            return holder.get("value")

        load_loop = QEventLoop()
        load_state: Dict[str, Any] = {"ok": False, "error": ""}

        def _loaded(ok: bool) -> None:
            load_state["ok"] = bool(ok)
            load_loop.quit()

        def _loading_changed(info: Any) -> None:
            try:
                status = str(info.status()).lower()
                if "loadfailed" in status or "failed" in status:
                    err = getattr(info, "errorString", lambda: "")()
                    code = getattr(info, "errorCode", lambda: "")()
                    load_state["error"] = f"{err} ({code})".strip()
            except Exception:
                pass

        page.loadFinished.connect(_loaded)
        try:
            page.loadingChanged.connect(_loading_changed)
        except Exception:
            pass
        page.load(QUrl(page_url))
        QTimer.singleShot(min(max(timeout * 1000, 15000), 60000), load_loop.quit)
        load_loop.exec()
        if not load_state.get("ok"):
            detail = str(load_state.get("error") or "").strip()
            if detail:
                raise RuntimeError(f"AuthND browser failed to load {page_url}: {detail}")
            raise RuntimeError(f"AuthND browser failed to load {page_url}")

        sitekey = os.getenv("AUTHND_HCAPTCHA_SITEKEY", DEFAULT_HCAPTCHA_SITEKEY)
        script = f"""
(() => {{
  window.__authndResult = {{pending: true, step: "starting"}};
  const sitekey = {json.dumps(sitekey)};
  const waitFor = (fn, timeoutMs = 20000) => new Promise((resolve, reject) => {{
    const start = Date.now();
    const tick = () => {{
      try {{
        if (fn()) return resolve(true);
      }} catch (e) {{}}
      if (Date.now() - start > timeoutMs) return reject(new Error("timeout waiting for hcaptcha"));
      setTimeout(tick, 250);
    }};
    tick();
  }});
  const loadScript = () => new Promise((resolve, reject) => {{
    if (window.hcaptcha) return resolve(true);
    const existing = document.querySelector("script[src*='hcaptcha.com/1/api.js']");
    if (existing) {{
      existing.addEventListener("load", () => resolve(true), {{once: true}});
      existing.addEventListener("error", () => reject(new Error("hcaptcha script failed")), {{once: true}});
      return;
    }}
    const script = document.createElement("script");
    script.src = "https://js.hcaptcha.com/1/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.onload = () => resolve(true);
    script.onerror = () => reject(new Error("hcaptcha script failed"));
    document.head.appendChild(script);
  }});
  (async () => {{
    try {{
      await loadScript();
      await waitFor(() => window.hcaptcha && window.hcaptcha.render && window.hcaptcha.execute);
      let container = document.getElementById("__authnd_hcaptcha");
      if (!container) {{
        container = document.createElement("div");
        container.id = "__authnd_hcaptcha";
        container.style.position = "fixed";
        container.style.left = "-10000px";
        container.style.top = "0";
        document.body.appendChild(container);
      }}
      let widgetId = window.__authndWidgetId;
      if (widgetId === undefined || widgetId === null) {{
        widgetId = window.hcaptcha.render(container, {{sitekey, size: "invisible"}});
        window.__authndWidgetId = widgetId;
      }}
      const execResult = await window.hcaptcha.execute(widgetId, {{async: true}});
      const token = (execResult && execResult.response) || window.hcaptcha.getResponse(widgetId) || "";
      window.__authndResult = {{pending: false, token, execResult, error: null}};
    }} catch (error) {{
      window.__authndResult = {{
        pending: false,
        token: "",
        error: String(error && (error.stack || error.message || error))
      }};
    }}
  }})();
  return true;
}})();
"""
        run_js(script, js_timeout_ms=10000)

        deadline = time.time() + max(timeout, 30)
        last_result: Dict[str, Any] = {}
        while time.time() < deadline:
            raw = run_js("JSON.stringify(window.__authndResult || {pending:true})", js_timeout_ms=5000)
            try:
                result = json.loads(raw or "{}")
            except Exception:
                result = {}
            last_result = result
            if result and not result.get("pending", True):
                token = str(result.get("token") or "").strip()
                if token:
                    return token
                raise RuntimeError(f"AuthND hCaptcha failed: {result.get('error') or result}")
            if _is_cancelled():
                raise RuntimeError("stream cancelled")
            wait_loop = QEventLoop()
            QTimer.singleShot(100, wait_loop.quit)
            wait_loop.exec()

        raise RuntimeError(f"AuthND hCaptcha timed out: {last_result}")
    finally:
        try:
            page.deleteLater()
            profile.deleteLater()
            cleanup_loop = QEventLoop()
            QTimer.singleShot(100, cleanup_loop.quit)
            cleanup_loop.exec()
        except Exception:
            pass


def get_captcha_token(page_url: str, timeout: int = 90) -> str:
    mode = os.getenv("AUTHND_TOKEN_MODE", "subprocess").strip().lower()
    if mode == "inline":
        return _mint_captcha_token_qt(page_url, timeout)
    return _mint_captcha_token_subprocess(page_url, timeout)


def _get_captcha_token_for_request(
    page_url: str,
    timeout: int = 90,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    if not _env_bool("AUTHND_SERIALIZE_CAPTCHA_MINT", True):
        return get_captcha_token(page_url, timeout)

    waited = False
    while not _captcha_mint_lock.acquire(timeout=0.1):
        if _is_cancelled():
            raise RuntimeError("stream cancelled")
        if not waited:
            _log(log_fn, "⏳ AuthND: waiting for browser token slot")
            waited = True
    try:
        return get_captcha_token(page_url, timeout)
    finally:
        _captcha_mint_lock.release()


def _extract_content_from_obj(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        for candidate in (
            delta.get("content"),
            message.get("content"),
            choice.get("text"),
            choice.get("content"),
        ):
            if isinstance(candidate, str) and candidate:
                return candidate
    for key in ("output_text", "text", "content", "response"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_reasoning_from_obj(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        for candidate in (
            delta.get("reasoning_content"),
            delta.get("reasoning"),
            delta.get("thinking"),
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("thinking"),
            choice.get("reasoning_content"),
            choice.get("reasoning"),
            choice.get("thinking"),
        ):
            if isinstance(candidate, str) and candidate:
                return candidate
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_finish_reason(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        reason = (choices[0] or {}).get("finish_reason")
        if reason:
            return str(reason)
    reason = obj.get("finish_reason") or obj.get("stop_reason")
    return str(reason) if reason else None


def _usage_completion_tokens(usage: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _usage_reasoning_tokens(usage: Optional[Dict[str, Any]]) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in ("reasoning_tokens", "thinking_tokens", "thoughts_tokens", "thoughtsTokenCount"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _infer_finish_reason(
    *,
    explicit_finish_reason: Optional[str],
    content: str,
    usage: Optional[Dict[str, Any]],
    requested_max_tokens: Optional[int],
    saw_done: bool,
    saw_event: bool,
    stream: bool,
) -> Tuple[str, bool, str]:
    if explicit_finish_reason:
        return explicit_finish_reason, True, "provider"

    completion_tokens = _usage_completion_tokens(usage)
    if requested_max_tokens and completion_tokens is not None and completion_tokens >= int(requested_max_tokens):
        return "length", False, "completion_tokens_reached_max_tokens"

    if not (content or "").strip():
        if saw_done or saw_event or not stream:
            return "error", False, "empty_content_without_finish_reason"
        return "incomplete", False, "no_content_no_done"

    if stream and not saw_done:
        return "incomplete", False, "stream_ended_without_done"

    return "stop", False, "done_without_finish_reason"


def _iter_utf8_lines(byte_iter: Iterable[Any]) -> Iterable[str]:
    """Yield text lines from raw SSE bytes, decoded explicitly as UTF-8.

    This avoids letting requests/httpx infer text encodings from platform or
    headers, which is how UTF-8 punctuation can be decoded incorrectly before
    JSON parsing sees it.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    for chunk in byte_iter:
        if not chunk:
            continue
        if isinstance(chunk, str):
            text = chunk
        else:
            text = decoder.decode(bytes(chunk), final=False)
        buffer += text
        while True:
            newline = buffer.find("\n")
            if newline < 0:
                break
            line = buffer[:newline]
            buffer = buffer[newline + 1:]
            yield line.rstrip("\r")

    tail = decoder.decode(b"", final=True)
    if tail:
        buffer += tail
    if buffer:
        yield buffer.rstrip("\r")


def _parse_sse_lines(
    line_iter: Iterable[Any],
    *,
    close_fn: Optional[Callable[[], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
    t_start: Optional[float] = None,
    requested_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    parts: List[str] = []
    reasoning_parts: List[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw_tail: List[Any] = []
    text_log_buf: List[str] = []
    thought_log_buf: List[str] = []
    first_token_ts: Optional[float] = None
    thinking_started = False
    thinking_ended = False
    thinking_chunks = 0
    thinking_start_ts: Optional[float] = None
    stream_started_ts = t_start or time.time()
    stream_thinking = _stream_thinking_logging_enabled()
    saw_done = False
    saw_event = False

    def _mark_first_token() -> None:
        nonlocal first_token_ts
        if first_token_ts is None:
            first_token_ts = time.time()
            if log_stream:
                _log(log_fn, f"📡 AuthND: First token in {first_token_ts - stream_started_ts:.1f}s, streaming...")

    def _emit_stream_text(fragment: str) -> None:
        if not log_fn or not log_stream or not fragment:
            return
        text_log_buf.append(fragment.replace("\x1f", "\\x1F"))
        combined = "".join(text_log_buf)
        for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
            combined = combined.replace(tag, tag + "\n")
        if "\n" in combined:
            lines = combined.split("\n")
            for line in lines[:-1]:
                if line:
                    log_fn(line)
            text_log_buf[:] = [lines[-1]]
        elif len(combined) >= 160:
            log_fn(combined)
            text_log_buf.clear()

    def _emit_thinking(fragment: str) -> None:
        nonlocal thinking_started, thinking_chunks, thinking_start_ts
        if not fragment:
            return
        if not thinking_started:
            thinking_started = True
            thinking_start_ts = time.time()
            thinking_chunks = 0
            if log_fn and log_stream and stream_thinking:
                log_fn("🧠 [authnd] Thinking...")
        thinking_chunks += 1
        if not log_fn or not log_stream or not stream_thinking:
            return
        thought_log_buf.append(fragment.replace("\\n", "\n").replace("\x1f", "\\x1F"))
        combined = "".join(thought_log_buf)
        if "\n" in combined:
            lines = combined.split("\n")
            for line in lines[:-1]:
                log_fn(f"    {line}")
            thought_log_buf[:] = [lines[-1]]
        elif len(combined) >= 160:
            log_fn(f"    {combined}")
            thought_log_buf.clear()

    def _finish_thinking_before_text() -> None:
        nonlocal thinking_ended
        if not thinking_started or thinking_ended:
            return
        thinking_ended = True
        if not log_fn or not log_stream or not stream_thinking:
            thought_log_buf.clear()
            return
        remainder = "".join(thought_log_buf).rstrip("\n")
        if remainder:
            for line in remainder.split("\n"):
                log_fn(f"    {line}")
        thought_log_buf.clear()
        duration = time.time() - (thinking_start_ts or stream_started_ts)
        log_fn(f"🧠 [authnd] Thinking complete ({thinking_chunks} chunks, {duration:.1f}s)")
        log_fn("─" * 50)
        log_fn("📡 Text streaming...")

    for raw_line in line_iter:
        if _is_cancelled():
            if close_fn:
                close_fn()
            raise RuntimeError("stream cancelled")
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            if line == "[DONE]":
                saw_done = True
                break
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_event = True
        raw_tail.append(obj)
        if len(raw_tail) > 5:
            raw_tail.pop(0)
        text = _extract_content_from_obj(obj)
        reasoning = _extract_reasoning_from_obj(obj)
        if reasoning:
            _mark_first_token()
            reasoning_parts.append(reasoning)
            _emit_thinking(reasoning)
        if text:
            _mark_first_token()
            _finish_thinking_before_text()
            parts.append(text)
            _emit_stream_text(text)
        reason = _extract_finish_reason(obj)
        if reason:
            finish_reason = reason
        if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
            usage = obj.get("usage")

    if log_fn and log_stream and text_log_buf:
        remainder = "".join(text_log_buf).strip()
        if remainder:
            log_fn(remainder)
    if log_fn and log_stream and stream_thinking and thinking_started and not thinking_ended:
        remainder = "".join(thought_log_buf).rstrip("\n")
        if remainder:
            for line in remainder.split("\n"):
                log_fn(f"    {line}")
        duration = time.time() - (thinking_start_ts or stream_started_ts)
        log_fn(f"🧠 [authnd] Thinking complete ({thinking_chunks} chunks, {duration:.1f}s)")
    thinking_tokens = _usage_reasoning_tokens(usage)
    if log_fn and log_stream:
        if thinking_tokens:
            log_fn(f"   💭 Thinking tokens used: {thinking_tokens:,}")
        elif reasoning_parts:
            estimated_tokens = max(1, len("".join(reasoning_parts)) // 4)
            log_fn(f"   💭 Thinking tokens used: ~{estimated_tokens:,}")
    if log_stream:
        _log(log_fn, f"📡 AuthND: Stream finished in {time.time() - stream_started_ts:.1f}s")

    content = "".join(parts)
    final_finish_reason, finish_reason_explicit, finish_reason_inference = _infer_finish_reason(
        explicit_finish_reason=finish_reason,
        content=content,
        usage=usage,
        requested_max_tokens=requested_max_tokens,
        saw_done=saw_done,
        saw_event=saw_event,
        stream=True,
    )

    return {
        "content": content,
        "finish_reason": final_finish_reason,
        "finish_reason_explicit": finish_reason_explicit,
        "finish_reason_inference": finish_reason_inference,
        "usage": usage,
        "reasoning_content": "".join(reasoning_parts) if reasoning_parts else None,
        "raw_response": raw_tail,
    }


def _parse_sse_response(
    response: requests.Response,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
    t_start: Optional[float] = None,
    requested_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return _parse_sse_lines(
        _iter_utf8_lines(response.iter_content(chunk_size=1)),
        close_fn=response.close,
        log_fn=log_fn,
        log_stream=log_stream,
        t_start=t_start,
        requested_max_tokens=requested_max_tokens,
    )


def _parse_json_response(response: requests.Response, *, requested_max_tokens: Optional[int] = None) -> Dict[str, Any]:
    try:
        obj = response.json()
    except ValueError:
        text = response.text or ""
        final_finish_reason, finish_reason_explicit, finish_reason_inference = _infer_finish_reason(
            explicit_finish_reason=None,
            content=text,
            usage=None,
            requested_max_tokens=requested_max_tokens,
            saw_done=True,
            saw_event=bool(text),
            stream=False,
        )
        return {
            "content": text,
            "finish_reason": final_finish_reason,
            "finish_reason_explicit": finish_reason_explicit,
            "finish_reason_inference": finish_reason_inference,
            "usage": None,
            "reasoning_content": None,
            "raw_response": text,
        }

    finish_reason = _extract_finish_reason(obj)
    content = _extract_content_from_obj(obj)
    reasoning = _extract_reasoning_from_obj(obj)
    usage = obj.get("usage") if isinstance(obj, dict) else None
    final_finish_reason, finish_reason_explicit, finish_reason_inference = _infer_finish_reason(
        explicit_finish_reason=finish_reason,
        content=content,
        usage=usage,
        requested_max_tokens=requested_max_tokens,
        saw_done=True,
        saw_event=True,
        stream=False,
    )

    return {
        "content": content,
        "finish_reason": final_finish_reason,
        "finish_reason_explicit": finish_reason_explicit,
        "finish_reason_inference": finish_reason_inference,
        "usage": usage,
        "reasoning_content": reasoning or None,
        "raw_response": obj,
    }


def _log_non_stream_summary(
    result: Dict[str, Any],
    *,
    log_fn: Optional[Callable[[str], None]],
    started_at: float,
) -> None:
    if not log_fn:
        return
    content = str(result.get("content") or "")
    reasoning = str(result.get("reasoning_content") or "")
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else None
    elapsed = time.time() - started_at
    _log(log_fn, f"📡 AuthND: Non-stream response finished in {elapsed:.1f}s ({len(content):,} chars)")
    thinking_tokens = _usage_reasoning_tokens(usage)
    if thinking_tokens:
        _log(log_fn, f"   💭 Thinking tokens used: {thinking_tokens:,}")
    elif reasoning:
        estimated_tokens = max(1, len(reasoning) // 4)
        _log(log_fn, f"   💭 Thinking tokens used: ~{estimated_tokens:,}")
    else:
        _log(log_fn, "   💭 Thinking tokens used: 0")


def _raise_for_status(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    nv_error = response.headers.get("x-nv-error-msg") or response.headers.get("x-nv-error-code") or ""
    body = (response.text or "").strip()
    detail = " ".join(part for part in (nv_error, body[:1000]) if part)
    raise RuntimeError(f"AuthND HTTP {response.status_code}: {detail or response.reason}")


def _httpx_status_error(resp: Any) -> RuntimeError:
    headers = getattr(resp, "headers", {}) or {}
    nv_error = headers.get("x-nv-error-msg") or headers.get("x-nv-error-code") or ""
    reason = getattr(resp, "reason_phrase", "") or ""
    try:
        body = resp.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    detail = " ".join(part for part in (nv_error, body[:1000]) if part)
    return RuntimeError(f"AuthND HTTP {resp.status_code}: {detail or reason or 'HTTP error'}")


def _post_prediction(
    *,
    messages: List[Dict[str, Any]],
    model_id: str,
    model_path: str,
    page_url: str,
    captcha_token: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    timeout: int,
    connect_timeout: Optional[float],
    stream: bool,
    log_stream: Optional[bool] = None,
    progress_label: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    suppress_chat_template_kwargs: bool = False,
) -> Dict[str, Any]:
    metadata = _resolve_model_metadata(page_url)
    org_id = metadata.get("namespace") or DEFAULT_ORG_ID
    endpoint_id = metadata.get("endpoint_id") or model_id
    payload_model = metadata.get("payload_model") or _payload_model_name(model_path)
    url = f"{API_BASE_URL}/v2/predict/models/{org_id}/{endpoint_id}"
    payload: Dict[str, Any] = {
        "messages": messages,
        "model": payload_model,
        "stream": bool(stream),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if frequency_penalty is not None:
        payload["frequency_penalty"] = float(frequency_penalty)
    if presence_penalty is not None:
        payload["presence_penalty"] = float(presence_penalty)
    _apply_reasoning_payload(payload, model_path)
    if suppress_chat_template_kwargs:
        payload.pop("chat_template_kwargs", None)

    _log(
        log_fn,
        "🔎 AuthND debug: "
        + json.dumps(
            {
                "url": url,
                "page_url": page_url,
                "metadata": {
                    "namespace": org_id,
                    "endpoint_id": endpoint_id,
                    "function_id": metadata.get("function_id") or "",
                    "artifact_name": metadata.get("artifact_name") or "",
                    "payload_model": payload_model,
                },
                "payload": _payload_summary(payload),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        debug_only=True,
    )

    request_id = str(uuid.uuid4())
    headers = {
        "accept": "text/event-stream" if stream else "application/json",
        "content-type": "application/json",
        "accept-encoding": "identity",
        "origin": BUILD_BASE_URL,
        "referer": page_url,
        "host": "api.ngc.nvidia.com",
        "nv-captcha-token": captcha_token,
        "user-agent": USER_AGENT,
    }
    if os.getenv("AUTHND_LEGACY_EXTRA_HEADERS", "0").lower() in ("1", "true", "yes"):
        headers.update({
            "nv-function-id": endpoint_id,
            "nv-model-name": model_path,
            "nv-session-id": str(uuid.uuid4()),
            "nvcf-request-id": request_id,
        })
    _log(
        log_fn,
        "🔎 AuthND debug headers: "
        + json.dumps(
            {
                "accept": headers["accept"],
                "origin": headers["origin"],
                "referer": headers["referer"],
                "legacy_extra_headers": "nv-function-id" in headers,
                "nv-captcha-token-length": len(captcha_token or ""),
                "local_request_id": request_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        debug_only=True,
    )

    request_started = time.time()
    if stream:
        try:
            import httpx as _httpx

            _timeout = _httpx.Timeout(timeout, connect=connect_timeout)
            with _httpx.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=_timeout,
            ) as response:
                closer = _register_response_closer(response.close)
                _log(
                    log_fn,
                    f"🔎 AuthND debug response: status={response.status_code}, content_type={response.headers.get('content-type', '')}, transport=httpx",
                    debug_only=True,
                )
                if response.status_code >= 400:
                    exc = _httpx_status_error(response)
                    _log(log_fn, f"⚠️ AuthND HTTP failure: {_short_error(exc)}")
                    _unregister_response_closer(closer)
                    raise exc
                if progress_label:
                    _log(log_fn, progress_label)
                if log_stream is None or log_stream:
                    _log(log_fn, f"📡 AuthND: Stream opened (status={response.status_code}, transport=httpx)")
                try:
                    return _parse_sse_lines(
                        _iter_utf8_lines(response.iter_raw()),
                        close_fn=response.close,
                        log_fn=log_fn,
                        log_stream=_stream_logging_enabled() if log_stream is None else bool(log_stream),
                        t_start=request_started,
                        requested_max_tokens=max_tokens,
                    )
                finally:
                    _unregister_response_closer(closer)
        except ImportError:
            _log(log_fn, "⚠️ AuthND: httpx not installed, falling back to requests (streaming may be buffered)")

    request_timeout: Any = timeout
    if connect_timeout is not None:
        request_timeout = (connect_timeout, timeout)

    session = _get_session()
    response = _post_with_cancel(
        session,
        url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
        stream=stream,
    )
    _log(
        log_fn,
        f"🔎 AuthND debug response: status={response.status_code}, content_type={response.headers.get('content-type', '')}",
        debug_only=True,
    )
    if response.status_code >= 400:
        nv_error = response.headers.get("x-nv-error-msg") or response.headers.get("x-nv-error-code") or ""
        body = (response.text or "").strip()
        _log(
            log_fn,
            f"⚠️ AuthND HTTP failure: status={response.status_code}, nv_error={_short_error(nv_error, 300)}, body={_short_error(body, 900)}",
        )
    _raise_for_status(response)
    content_type = (response.headers.get("content-type") or "").lower()
    if stream or "text/event-stream" in content_type:
        if progress_label:
            _log(log_fn, progress_label)
        if log_stream is None or log_stream:
            _log(log_fn, f"📡 AuthND: Stream opened (status={response.status_code})")
        closer = _register_response_closer(response.close)
        try:
            return _parse_sse_response(
                response,
                log_fn=log_fn,
                log_stream=_stream_logging_enabled() if log_stream is None else bool(log_stream),
                t_start=request_started,
                requested_max_tokens=max_tokens,
            )
        finally:
            _unregister_response_closer(closer)
    result = _parse_json_response(response, requested_max_tokens=max_tokens)
    _log_non_stream_summary(result, log_fn=log_fn, started_at=request_started)
    return result


def send_chat_completion(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    timeout: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    connect_timeout: Optional[float] = None,
    account_id: int = 0,
    stream: Optional[bool] = None,
    log_stream: Optional[bool] = None,
    progress_label: Optional[str] = None,
) -> Dict[str, Any]:
    del account_id  # AuthND has no account slots; kept for unified handler symmetry.
    if _is_cancelled():
        raise RuntimeError("stream cancelled")

    publisher, model_id, page_url = _normalize_model(model)
    model_path = f"{publisher}/{model_id}"
    suppress_chat_template_kwargs = _model_requires_no_chat_template_kwargs(model_path)
    timeout_value = int(timeout or _env_int("AUTHND_TIMEOUT", DEFAULT_TIMEOUT))
    token_timeout = _env_int("AUTHND_TOKEN_TIMEOUT", min(max(timeout_value, 60), 180))
    use_stream = stream
    if use_stream is None:
        use_stream = os.getenv("AUTHND_STREAM", "1").lower() not in ("0", "false", "no")

    if log_fn:
        log_fn(f"🌐 AuthND: opening browser token flow for {page_url}")

    normalized_messages = _normalize_messages(messages)
    _log(
        log_fn,
        "🔎 AuthND debug request: "
        + json.dumps(
            {
                "model_path": model_path,
                "page_url": page_url,
                "timeouts": {"request": timeout_value, "token": token_timeout, "connect": connect_timeout},
                "stream": bool(use_stream),
                "params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "reasoning_enabled": _reasoning_enabled(),
                    "reasoning_effort": _reasoning_effort(),
                },
                "messages": _message_summary(normalized_messages),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        debug_only=True,
    )
    last_error: Optional[Exception] = None

    for attempt in range(2):
        if _is_cancelled():
            raise RuntimeError("stream cancelled")
        try:
            captcha_token = _get_captcha_token_for_request(page_url, token_timeout, log_fn=log_fn)
        except RuntimeError as exc:
            _log(
                log_fn,
                f"⚠️ AuthND captcha token flow failed (attempt {attempt + 1}/2): {_short_error(exc)}",
            )
            raise
        if _is_cancelled():
            raise RuntimeError("stream cancelled")
        _log(
            log_fn,
            f"🔎 AuthND debug: captcha token acquired (length={len(captcha_token)})",
            debug_only=True,
        )
        _log(log_fn, "📨 AuthND: captcha token acquired; sending NVIDIA request")
        post_progress_label = progress_label or f"📤 [{threading.current_thread().name}] API call in progress"
        try:
            result = _post_prediction(
                messages=normalized_messages,
                model_id=model_id,
                model_path=model_path,
                page_url=page_url,
                captcha_token=captcha_token,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                timeout=timeout_value,
                connect_timeout=connect_timeout,
                stream=bool(use_stream),
                log_stream=log_stream,
                progress_label=post_progress_label,
                log_fn=log_fn,
                suppress_chat_template_kwargs=suppress_chat_template_kwargs,
            )
            result["model"] = model_id
            result["page_url"] = page_url
            return result
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            if (
                _is_chat_template_unsupported_error(exc)
                and _reasoning_control_configured()
                and not suppress_chat_template_kwargs
            ):
                _remember_chat_template_unsupported_model(model_path)
                suppress_chat_template_kwargs = True
                if log_fn:
                    log_fn(
                        "AuthND: NVIDIA rejected chat_template_kwargs; "
                        "caching this model without chat template controls and retrying once"
                    )
                result = _post_prediction(
                    messages=normalized_messages,
                    model_id=model_id,
                    model_path=model_path,
                    page_url=page_url,
                    captcha_token=captcha_token,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    timeout=timeout_value,
                    connect_timeout=connect_timeout,
                    stream=bool(use_stream),
                    log_stream=log_stream,
                    progress_label=post_progress_label,
                    log_fn=log_fn,
                    suppress_chat_template_kwargs=True,
                )
                result["model"] = model_id
                result["page_url"] = page_url
                return result
            is_context_length_error = (
                "maximum context length" in message
                or "max context length" in message
            )
            is_captcha_error = (
                "captcha" in message
                or "nv-captcha" in message
                or "hcaptcha" in message
            )
            if attempt == 0 and is_captcha_error and not is_context_length_error:
                if log_fn:
                    log_fn(f"⚠️ AuthND: captcha token was rejected; retrying with a fresh browser token ({_short_error(exc)})")
                continue
            raise

    raise RuntimeError(f"AuthND request failed: {last_error}")


def _read_cli_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    if value == "-":
        return sys.stdin.read()
    return value


def _read_cli_file(path: Optional[str]) -> str:
    if not path:
        return ""
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_cli_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.messages:
        with open(args.messages, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            data = data["messages"]
        if not isinstance(data, list):
            raise ValueError("--messages must contain a JSON list or an object with a messages list")
        return data

    prompt = _read_cli_text(args.prompt) or _read_cli_file(args.prompt_file)
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    if not prompt:
        raise ValueError("provide --prompt, --prompt-file, --messages, or pipe prompt text on stdin")

    messages: List[Dict[str, Any]] = []
    if args.system:
        messages.append({"role": "system", "content": _read_cli_text(args.system)})
    elif args.system_file:
        messages.append({"role": "system", "content": _read_cli_file(args.system_file)})
    messages.append({"role": "user", "content": prompt})
    return messages


def _main() -> int:
    for stream_name in ("stdout", "stderr"):
        stream_obj = getattr(sys, stream_name, None)
        if hasattr(stream_obj, "reconfigure"):
            try:
                stream_obj.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Send chat requests through NVIDIA Build's browser-backed AuthND route.",
    )
    parser.add_argument("--mint-token", dest="page_url", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="z-ai/glm-5.1", help="Model path, e.g. deepseek-ai/deepseek-v4-flash")
    parser.add_argument("--prompt", help="User prompt text. Use '-' to read stdin.")
    parser.add_argument("--prompt-file", help="UTF-8 file containing the user prompt. Use '-' to read stdin.")
    parser.add_argument("--system", help="Optional system prompt text. Use '-' to read stdin.")
    parser.add_argument("--system-file", help="UTF-8 file containing the system prompt.")
    parser.add_argument("--messages", help="JSON file containing messages list or {'messages': [...]}.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--frequency-penalty", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--token-timeout", type=int, help="Browser captcha token timeout in seconds.")
    parser.add_argument("--connect-timeout", type=float)
    parser.add_argument("--stream", dest="stream", action="store_true", default=None, help="Force SSE streaming.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable SSE streaming.")
    parser.add_argument("--json", action="store_true", help="Print the full result object as JSON.")
    parser.add_argument("--output", help="Write final content to a UTF-8 file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    parser.add_argument("--debug", action="store_true", help="Enable sanitized AuthND debug logs.")
    args = parser.parse_args()
    if args.page_url:
        try:
            token = _mint_captcha_token_qt(args.page_url, args.timeout)
            print(json.dumps({"token": token}, separators=(",", ":")), flush=True)
            return 0
        except Exception as exc:
            message = _short_error(exc)
            print(json.dumps({"error": message}, separators=(",", ":")), flush=True)
            print(f"AuthND token helper error: {message}", file=sys.stderr)
            return 1

    if args.debug:
        os.environ["AUTHND_DEBUG"] = "1"
    if args.token_timeout:
        os.environ["AUTHND_TOKEN_TIMEOUT"] = str(args.token_timeout)

    try:
        messages = _load_cli_messages(args)
        log_fn = None
        if not args.quiet:
            log_fn = lambda message: print(message, file=sys.stderr, flush=True)
        result = send_chat_completion(
            messages=messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            timeout=args.timeout,
            connect_timeout=args.connect_timeout,
            stream=args.stream,
            log_fn=log_fn,
        )
    except Exception as exc:
        print(f"AuthND error: {_short_error(exc)}", file=sys.stderr)
        return 1

    content = str(result.get("content") or "")
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        print(content, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
