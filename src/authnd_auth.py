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
import re
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


def _message_summary(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    roles: Dict[str, int] = {}
    chars = 0
    for message in messages:
        role = str(message.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
        chars += len(str(message.get("content") or ""))
    return {"count": len(messages), "roles": roles, "chars": chars}


def _payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "messages": _message_summary(payload.get("messages") or []),
    }
    for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
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


def _build_common_mojibake_replacements() -> Dict[str, str]:
    replacements: Dict[str, str] = {}
    for char in ("“", "”", "‘", "’", "–", "—", "…", "•", "«", "»"):
        data = char.encode("utf-8")
        for encoding in ("latin-1", "cp1252"):
            bad = data.decode(encoding, errors="replace")
            if bad != char:
                replacements[bad] = char
    replacements["Â\xa0"] = " "
    replacements["Â "] = " "
    return replacements


_COMMON_MOJIBAKE_REPLACEMENTS = _build_common_mojibake_replacements()


def _mojibake_score(text: str) -> int:
    if not text:
        return 0
    score = 0
    score += text.count("\ufffd") * 4
    score += sum(2 for ch in text if 0x80 <= ord(ch) <= 0x9F)
    for marker in ("â", "Â", "Ã"):
        score += text.count(marker)
    for sequence in ("â€", "â€™", "â€œ", "â€�", "â€“", "â€”", "Â ", "Ã©", "Ã¨", "Ãª"):
        score += text.count(sequence) * 3
    return score


def _repair_mojibake_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    if os.getenv("AUTHND_FIX_UNICODE", "1").strip().lower() in ("0", "false", "no", "off"):
        return text

    best = text
    best_score = _mojibake_score(text)
    if best_score <= 0:
        return text

    mapped = text
    for bad, good in _COMMON_MOJIBAKE_REPLACEMENTS.items():
        mapped = mapped.replace(bad, good)
    mapped_score = _mojibake_score(mapped)
    if mapped_score < best_score:
        best = mapped
        best_score = mapped_score

    # Common failure modes:
    # - UTF-8 bytes decoded as latin-1: "â\x80\x9c"
    # - UTF-8 bytes decoded as cp1252: "â€œ"
    # Run a couple of passes for occasional double-decoding.
    candidates = {text}
    current = text
    for _ in range(2):
        next_candidates = set()
        for candidate in candidates:
            for encoding in ("latin-1", "cp1252"):
                try:
                    repaired = candidate.encode(encoding).decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
                next_candidates.add(repaired)
                score = _mojibake_score(repaired)
                if score < best_score:
                    best = repaired
                    best_score = score
        if not next_candidates or current in next_candidates:
            break
        candidates = next_candidates
        current = best

    return best


def cancel_stream() -> None:
    """Signal any active AuthND stream/request to stop."""
    _cancel_event.set()


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

    page_url = f"{BUILD_BASE_URL}/{publisher}/{model_id}"
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


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_parts: List[str] = []
    normalized: List[Dict[str, str]] = []

    for message in messages or []:
        role = str(message.get("role", "user")).lower()
        content = _content_to_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        normalized.append({"role": role, "content": content})

    if system_parts:
        system_text = "System instructions:\n" + "\n\n".join(system_parts)
        for message in normalized:
            if message["role"] == "user":
                message["content"] = f"{system_text}\n\n{message['content']}"
                break
        else:
            normalized.insert(0, {"role": "user", "content": system_text})

    if not normalized:
        normalized.append({"role": "user", "content": ""})
    return normalized


def _reasoning_enabled() -> bool:
    """
    NVIDIA Build exposes reasoning as a boolean chat_template_kwargs toggle for
    GLM-5.1, not as a graded reasoning_effort parameter. Map the repo's effort
    variables onto that boolean so existing settings still have an effect.
    """
    explicit = os.getenv("AUTHND_ENABLE_THINKING")
    if explicit is not None:
        return explicit.strip().lower() not in ("0", "false", "no", "none", "off", "disabled")

    effort = (
        os.getenv("AUTHND_REASONING_EFFORT")
        or os.getenv("GPT_EFFORT")
        or os.getenv("REASONING_EFFORT")
        or ""
    ).strip().lower()
    if effort:
        return effort not in ("0", "false", "no", "none", "off", "disabled")

    shared_toggle = os.getenv("ENABLE_GPT_THINKING")
    if shared_toggle is not None:
        return shared_toggle.strip().lower() in ("1", "true", "yes", "on", "enabled")

    return False


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"user-agent": USER_AGENT})
        _thread_local.session = session
    return session


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


def _mint_captcha_token_subprocess(page_url: str, timeout: int) -> str:
    helper_timeout = max(30, int(timeout))
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mint-token",
        page_url,
        "--timeout",
        str(helper_timeout),
    ]
    env = os.environ.copy()
    flags = env.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    required_flags = "--disable-gpu --disable-software-rasterizer"
    env["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} {required_flags}".strip()

    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=helper_timeout + 20,
        creationflags=creationflags,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"AuthND token helper failed ({proc.returncode}): {detail[-1200:]}")
    result = _extract_json_from_process(proc.stdout)
    token = str(result.get("token") or "").strip()
    if not token:
        raise RuntimeError(f"AuthND token helper returned no token: {result}")
    return token


def _mint_captcha_token_qt(page_url: str, timeout: int) -> str:
    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer")

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
    load_state: Dict[str, Any] = {"ok": False}

    def _loaded(ok: bool) -> None:
        load_state["ok"] = bool(ok)
        load_loop.quit()

    page.loadFinished.connect(_loaded)
    page.load(QUrl(page_url))
    QTimer.singleShot(min(max(timeout * 1000, 15000), 60000), load_loop.quit)
    load_loop.exec()
    if not load_state.get("ok"):
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
        QTimer.singleShot(1000, wait_loop.quit)
        wait_loop.exec()

    raise RuntimeError(f"AuthND hCaptcha timed out: {last_result}")


def get_captcha_token(page_url: str, timeout: int = 90) -> str:
    mode = os.getenv("AUTHND_TOKEN_MODE", "subprocess").strip().lower()
    if mode == "inline":
        return _mint_captcha_token_qt(page_url, timeout)
    return _mint_captcha_token_subprocess(page_url, timeout)


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
            delta.get("reasoning_content"),
            delta.get("reasoning"),
            message.get("content"),
            message.get("reasoning_content"),
            message.get("reasoning"),
            choice.get("text"),
            choice.get("content"),
        ):
            if isinstance(candidate, str) and candidate:
                return _repair_mojibake_text(candidate)
    for key in ("output_text", "text", "content", "response"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return _repair_mojibake_text(value)
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


def _iter_utf8_lines(byte_iter: Iterable[Any]) -> Iterable[str]:
    """Yield text lines from raw SSE bytes, decoded explicitly as UTF-8.

    This avoids letting requests/httpx infer text encodings from platform or
    headers, which is how UTF-8 punctuation can turn into mojibake like
    ``â\x80\x9c`` before JSON parsing sees it.
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
) -> Dict[str, Any]:
    parts: List[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw_tail: List[Any] = []
    text_log_buf: List[str] = []
    first_text_ts: Optional[float] = None
    stream_started_ts = t_start or time.time()

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
                break
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_tail.append(obj)
        if len(raw_tail) > 5:
            raw_tail.pop(0)
        text = _extract_content_from_obj(obj)
        if text:
            if first_text_ts is None:
                first_text_ts = time.time()
                _log(log_fn, f"📡 AuthND: First token in {first_text_ts - stream_started_ts:.1f}s, streaming...")
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
    _log(log_fn, f"📡 AuthND: Stream finished in {time.time() - stream_started_ts:.1f}s")

    return {
        "content": _repair_mojibake_text("".join(parts)),
        "finish_reason": finish_reason or "stop",
        "usage": usage,
        "raw_response": raw_tail,
    }


def _parse_sse_response(
    response: requests.Response,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
    t_start: Optional[float] = None,
) -> Dict[str, Any]:
    return _parse_sse_lines(
        _iter_utf8_lines(response.iter_content(chunk_size=1)),
        close_fn=response.close,
        log_fn=log_fn,
        log_stream=log_stream,
        t_start=t_start,
    )


def _parse_json_response(response: requests.Response) -> Dict[str, Any]:
    try:
        obj = response.json()
    except ValueError:
        text = response.text or ""
        return {"content": _repair_mojibake_text(text), "finish_reason": "stop", "usage": None, "raw_response": text}

    return {
        "content": _extract_content_from_obj(obj),
        "finish_reason": _extract_finish_reason(obj) or "stop",
        "usage": obj.get("usage") if isinstance(obj, dict) else None,
        "raw_response": obj,
    }


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
    messages: List[Dict[str, str]],
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
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    metadata = _resolve_model_metadata(page_url)
    org_id = metadata.get("namespace") or DEFAULT_ORG_ID
    endpoint_id = metadata.get("endpoint_id") or model_id
    url = f"{API_BASE_URL}/v2/predict/models/{org_id}/{endpoint_id}"
    payload: Dict[str, Any] = {
        "messages": messages,
        "model": _payload_model_name(model_path),
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
    if _reasoning_enabled():
        payload.setdefault("chat_template_kwargs", {"enable_thinking": True, "clear_thinking": False})

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
                _log(
                    log_fn,
                    f"🔎 AuthND debug response: status={response.status_code}, content_type={response.headers.get('content-type', '')}, transport=httpx",
                    debug_only=True,
                )
                if response.status_code >= 400:
                    exc = _httpx_status_error(response)
                    _log(log_fn, f"⚠️ AuthND HTTP failure: {_short_error(exc)}")
                    raise exc
                _log(log_fn, f"📡 AuthND: Stream opened (status={response.status_code}, transport=httpx)")
                return _parse_sse_lines(
                    _iter_utf8_lines(response.iter_raw()),
                    close_fn=response.close,
                    log_fn=log_fn,
                    log_stream=_stream_logging_enabled(),
                    t_start=request_started,
                )
        except ImportError:
            _log(log_fn, "⚠️ AuthND: httpx not installed, falling back to requests (streaming may be buffered)")

    request_timeout: Any = timeout
    if connect_timeout is not None:
        request_timeout = (connect_timeout, timeout)

    session = _get_session()
    response = session.post(
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
        _log(log_fn, f"📡 AuthND: Stream opened (status={response.status_code})")
        return _parse_sse_response(
            response,
            log_fn=log_fn,
            log_stream=_stream_logging_enabled(),
            t_start=request_started,
        )
    return _parse_json_response(response)


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
    progress_label: Optional[str] = None,
) -> Dict[str, Any]:
    del account_id  # AuthND has no account slots; kept for unified handler symmetry.
    if _is_cancelled():
        raise RuntimeError("stream cancelled")

    publisher, model_id, page_url = _normalize_model(model)
    model_path = f"{publisher}/{model_id}"
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
            captcha_token = get_captcha_token(page_url, token_timeout)
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
        _log(log_fn, "✅ AuthND: captcha token acquired; sending NVIDIA request")
        if progress_label:
            _log(log_fn, progress_label)
        else:
            _log(log_fn, f"📤 [{threading.current_thread().name}] API call in progress")
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
                log_fn=log_fn,
            )
            result["model"] = model_id
            result["page_url"] = page_url
            return result
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            if attempt == 0 and ("captcha" in message or "400" in message):
                if log_fn:
                    log_fn(f"⚠️ AuthND: captcha token was rejected; retrying with a fresh browser token ({_short_error(exc)})")
                continue
            raise

    raise RuntimeError(f"AuthND request failed: {last_error}")


def _main() -> int:
    parser = argparse.ArgumentParser(description="AuthND token helper")
    parser.add_argument("--mint-token", dest="page_url")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.page_url:
        token = _mint_captcha_token_qt(args.page_url, args.timeout)
        print(json.dumps({"token": token}, separators=(",", ":")), flush=True)
        return 0
    parser.error("nothing to do")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
