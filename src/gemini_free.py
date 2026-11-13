"""
gemini_free.py - the ``search/gemini`` route: google.com with AI Mode on, in a browser page.

This module is intentionally not an official Gemini API client. One chat request is one AI
Mode question on its own page: the page opens ``google.com/search?udm=50`` (AI Mode), the
system prompt is typed into the prompt box, the text to work on and any images are attached
as files, and the finished answer is read from the page. The public entry point mirrors the
other browser-backed auth modules: send_chat_completion(...) returns a dict with
content/finish_reason/raw_response.

Three browsers can hold the page: the QtWebEngine helper (desktop, signed out); a headless tab
of Glossarion's own Chrome/Edge profile once the user signed in to Google with ``login()``
(signed in, AI Mode offers its larger models: ``search/gemini-<model>``); and, where a
browser_driver is in use (Glossarion Mobile's hidden WebView, no helper processes), a page of
the driver.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse

import browser_driver
import cdp_browser
import mobile_runtime


SEARCH_BASE_URL = "https://www.google.com/search"
# google.com with AI Mode on: ``udm=50`` is the AI Mode tab; ``hl=en`` keeps the page's own
# texts (the "response is ready" announcement, the menu labels) in English.
DEFAULT_SEARCH_PARAMS = {
    "udm": "50",
    "hl": "en",
}
DEFAULT_URL = f"{SEARCH_BASE_URL}?{urlencode(DEFAULT_SEARCH_PARAMS)}"
DEFAULT_MODEL = "gemini"
DEFAULT_TIMEOUT = 90
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# browser_driver page owner: cancel_stream() cancels only this route's in-app browser pages.
BROWSER_OWNER = "gemini_free"

_cancel_event = threading.Event()
_active_helper_processes: set = set()
_active_helper_lock = threading.Lock()


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parent


def _debug_enabled() -> bool:
    return os.getenv("GEMINI_FREE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _log(log_fn: Optional[Callable[[str], None]], message: str, *, debug_only: bool = False) -> None:
    if not log_fn:
        return
    if debug_only and not _debug_enabled():
        return
    log_fn(message)


def _short_error(error: Any, limit: int = 1200) -> str:
    text = str(error or "").replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_url_for_log(url: Any, limit: int = 500) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if parsed.query:
            params = []
            for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if key == "q":
                    params.append((key, f"<{len(value)} chars omitted>"))
                else:
                    params.append((key, value))
            text = parsed._replace(query=urlencode(params)).geturl()
    except Exception:
        pass
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def cancel_stream() -> None:
    """Cancel an active helper subprocess if the translation run is stopped.

    Must never block: Stop handlers reach this through set_stop_flag() and
    hard_cancel_all(), some of them on the GUI thread, and each helper tree is a
    taskkill of up to 3s. The flag is set inline (the waiting worker polls it
    every 0.1s and kills its own helper); the sweep runs on a daemon thread.
    The helpers are snapshotted here so a late sweep cannot reach the next run's.
    """
    _cancel_event.set()
    # Pages of a registered browser driver (Glossarion Mobile's hidden WebViews): their
    # pending calls fail at once. Never blocks; a no-op without a driver (desktop).
    browser_driver.cancel_pages(BROWSER_OWNER)
    with _active_helper_lock:
        helpers = list(_active_helper_processes)
    if not helpers:
        return
    try:
        threading.Thread(
            target=_terminate_helper_processes,
            args=(helpers,),
            kwargs={"kill": True},
            name="GeminiFreeCancel",
            daemon=True,
        ).start()
    except Exception:
        pass


def _terminate_helper_processes(helpers, *, kill: bool = False) -> None:
    for proc in helpers:
        _terminate_process_tree(proc, kill=kill)


def _terminate_active_helper_processes(*, kill: bool = False) -> None:
    with _active_helper_lock:
        helpers = list(_active_helper_processes)
    _terminate_helper_processes(helpers, kill=kill)


def reset_cancel() -> None:
    # Refuse to clear while a hard stop is active. This is called per-attempt /
    # per-chapter, so a just-starting worker must not wipe the cancel signal the
    # Stop button set; the helper subprocess only observes _cancel_event.
    # TRANSLATION_CANCELLED is only set on immediate stop (never graceful).
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return
    _cancel_event.clear()


def _is_cancelled() -> bool:
    # Also honor the hard-abort env var so a racing reset_cancel() can't let an
    # in-flight browser-backed request bypass Stop.
    return _cancel_event.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"


def _terminate_process_tree(proc: Any, *, kill: bool = False) -> None:
    try:
        if proc is None or proc.poll() is not None:
            return
    except Exception:
        return
    if os.name == "nt":
        try:
            from shutdown_utils import terminate_subprocess_tree
            terminate_subprocess_tree(proc, kill=kill, timeout=3)
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


def _qtwebengine_chromium_flags(existing: str = "") -> str:
    try:
        sys.path.insert(0, str(_repo_src_dir()))
        from authnd_auth import _qtwebengine_chromium_flags as authnd_flags

        return authnd_flags(existing)
    except Exception:
        pass

    try:
        tokens = shlex.split(existing or "")
    except ValueError:
        tokens = (existing or "").split()

    required_flags = [
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
    ]
    seen = set()
    merged: List[str] = []
    for flag in [*tokens, *required_flags]:
        if flag in seen:
            continue
        seen.add(flag)
        merged.append(flag)
    return " ".join(merged)


def _default_user_agent() -> str:
    try:
        sys.path.insert(0, str(_repo_src_dir()))
        from authnd_auth import USER_AGENT

        return USER_AGENT
    except Exception:
        return DEFAULT_USER_AGENT


def _origin_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    return f"{parsed.scheme}://{parsed.netloc}/"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _timeout_seconds(timeout: Optional[int], default: int = DEFAULT_TIMEOUT) -> Optional[int]:
    try:
        value = default if timeout is None else int(timeout)
    except (TypeError, ValueError):
        value = default
    if value <= 0:
        return None
    return max(1, value)


def _timeout_deadline(timeout: Optional[int], default: int = DEFAULT_TIMEOUT) -> Optional[float]:
    seconds = _timeout_seconds(timeout, default)
    if seconds is None:
        return None
    return time.time() + seconds


def _deadline_expired(deadline: Optional[float]) -> bool:
    return deadline is not None and time.time() >= deadline


def _timeout_label(timeout: Optional[int], default: int = DEFAULT_TIMEOUT) -> str:
    seconds = _timeout_seconds(timeout, default)
    return "disabled" if seconds is None else f"{seconds}s"


def _parse_headers(values: Optional[List[str]]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError(f"header must be in 'Name: value' form: {value}")
        name, header_value = value.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"header name is empty: {value}")
        headers[name] = header_value.strip()
    return headers


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_text(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(value)


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
                item_type = str(item.get("type") or "").lower()
                if item_type in ("", "text", "input_text"):
                    parts.append(str(item.get("text") or ""))
                elif item_type in ("image_url", "input_image", "image"):
                    parts.append("[image omitted]")
        return "\n".join(part for part in parts if part)
    return str(content)


def _configured_text_extraction_method() -> str:
    method = os.getenv("TEXT_EXTRACTION_METHOD", "").strip().lower()
    extraction_mode = os.getenv("EXTRACTION_MODE", "").strip().lower()
    use_html2text = os.getenv("USE_HTML2TEXT", "").strip().lower() in ("1", "true", "yes", "on")
    if method in ("enhanced", "html2text", "markdown") or extraction_mode == "enhanced" or use_html2text:
        return "enhanced"
    if method in ("standard", "beautifulsoup", "bs"):
        return "standard"
    return ""


def _content_looks_like_html_payload(text: str) -> bool:
    raw = str(text or "")
    if not raw:
        return False
    return bool(
        re.search(
            r"<(?:!doctype|html|head|body|article|section|main|div|p|h[1-6]|ul|ol|li|table|tr|td|blockquote)\b|</(?:div|p|h[1-6]|ul|ol|li|table|tr|td|blockquote)>",
            raw,
            re.IGNORECASE,
        )
    )


def _strip_search_prefix(model: str) -> str:
    raw = str(model or "").strip()
    match = re.match(r"^search\d{0,4}/", raw, flags=re.IGNORECASE)
    if match:
        raw = raw[match.end():]
    elif raw.lower().startswith("search"):
        raw = raw[len("search"):].lstrip("/")
    return raw.strip("/") or DEFAULT_MODEL


def _messages_expect_markdown_response(messages: Iterable[Dict[str, Any]]) -> bool:
    if _configured_text_extraction_method() == "enhanced":
        return True
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        if role in ("system", "assistant"):
            continue
        text = _content_to_text(message.get("content")).lower()
        if "html2text" in text or "markdown" in text:
            return True
    return False


def _setup_qt_environment() -> None:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _qtwebengine_chromium_flags(
        os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    )
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")


def _create_profile(app: Any, *, user_agent: Optional[str] = None) -> tuple[Any, Path]:
    from PySide6.QtWebEngineCore import QWebEngineProfile

    profile_root = Path.home() / ".glossarion" / "gemini_free_browser" / uuid.uuid4().hex
    profile_root.mkdir(parents=True, exist_ok=True)

    profile = QWebEngineProfile(f"gemini-free-{uuid.uuid4().hex}", app)
    profile.setHttpUserAgent(user_agent or _default_user_agent())
    try:
        profile.setPersistentStoragePath(str(profile_root))
        profile.setCachePath(str(profile_root / "cache"))
    except Exception:
        pass
    try:
        profile.setHttpAcceptLanguage(os.getenv("GEMINI_FREE_ACCEPT_LANGUAGE", "en-US,en;q=0.9"))
    except Exception:
        pass
    return profile, profile_root


def _cleanup_profile_root(profile_root: Any) -> None:
    try:
        from shutdown_utils import cleanup_generated_browser_profile_dir
        cleanup_generated_browser_profile_dir(str(profile_root), "gemini_free_browser")
    except Exception:
        pass


def _default_viewport_size() -> tuple[int, int]:
    width = max(320, _env_int("GEMINI_FREE_VIEWPORT_WIDTH", 1280))
    height = max(480, _env_int("GEMINI_FREE_VIEWPORT_HEIGHT", 900))
    return width, height


def _set_default_page_viewport(page: Any) -> None:
    try:
        from PySide6.QtCore import QSize

        width, height = _default_viewport_size()
        page.setViewportSize(QSize(width, height))
    except Exception:
        pass


def run_qtwebengine_request(
    *,
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    bootstrap_url: Optional[str] = None,
    credentials: str = "include",
    mode: str = "same-origin",
    timeout: int = 60,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a browser-side fetch() from a Qt WebEngine page."""
    _setup_qt_environment()

    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["gemini-free-request"])

    profile, profile_root = _create_profile(app, user_agent=user_agent)
    page = QWebEnginePage(profile, app)
    _set_default_page_viewport(page)
    timeout_seconds = _timeout_seconds(timeout, 60)

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

    try:
        load_loop = QEventLoop()
        load_state: Dict[str, Any] = {"ok": False, "error": "", "last_event": ""}

        def _loaded(ok: bool) -> None:
            load_state["ok"] = bool(ok)
            load_loop.quit()

        page.loadFinished.connect(_loaded)
        page.load(QUrl(bootstrap_url or _origin_url(url)))
        if timeout_seconds is not None:
            QTimer.singleShot(max(1000, int(timeout_seconds * 1000)), load_loop.quit)
        load_loop.exec()
        if not load_state.get("ok"):
            raise RuntimeError(f"Qt WebEngine failed to load bootstrap page {bootstrap_url}")

        payload = {
            "url": url,
            "method": method.upper(),
            "headers": headers or {},
            "body": body,
            "credentials": credentials,
            "mode": mode,
            "timeoutMs": max(0, int((timeout_seconds or 0) * 1000)),
        }
        script = f"""
(() => {{
  const request = {json.dumps(payload, ensure_ascii=False)};
  window.__geminiFreeRequestResult = {{pending: true}};
  const controller = new AbortController();
  const timer = request.timeoutMs > 0 ? setTimeout(() => controller.abort(), request.timeoutMs) : null;
  (async () => {{
    try {{
      const init = {{
        method: request.method,
        credentials: request.credentials,
        redirect: "follow",
        headers: request.headers || {{}},
        signal: controller.signal
      }};
      if (request.mode) init.mode = request.mode;
      if (!["GET", "HEAD"].includes(request.method) && request.body !== null && request.body !== undefined) {{
        init.body = String(request.body);
      }}
      const response = await fetch(request.url, init);
      const responseHeaders = {{}};
      try {{
        response.headers.forEach((value, key) => {{ responseHeaders[key] = value; }});
      }} catch (error) {{}}
      const text = await response.text();
      window.__geminiFreeRequestResult = {{
        pending: false,
        ok: response.ok,
        status: response.status,
        statusText: response.statusText,
        url: response.url,
        redirected: response.redirected,
        type: response.type,
        headers: responseHeaders,
        body: text,
        bodyLength: text.length,
        truncated: false,
        error: null
      }};
    }} catch (error) {{
      window.__geminiFreeRequestResult = {{
        pending: false,
        ok: false,
        status: 0,
        statusText: "Qt WebEngine fetch failed",
        url: request.url,
        redirected: false,
        type: "",
        headers: {{}},
        body: "",
        bodyLength: 0,
        truncated: false,
        error: String(error && (error.stack || error.message || error))
      }};
    }} finally {{
      if (timer) clearTimeout(timer);
    }}
  }})();
  return true;
}})();
"""
        run_js(script, js_timeout_ms=10000)

        deadline = _timeout_deadline(timeout, 60)
        last_result: Dict[str, Any] = {}
        while not _deadline_expired(deadline):
            if _is_cancelled():
                raise RuntimeError("stream cancelled")
            raw = run_js("JSON.stringify(window.__geminiFreeRequestResult || {pending:true})", js_timeout_ms=5000)
            try:
                result = json.loads(raw or "{}")
            except Exception:
                result = {}
            last_result = result
            if result and not result.get("pending", True):
                result["bootstrap_url"] = bootstrap_url
                return result

            wait_loop = QEventLoop()
            QTimer.singleShot(100, wait_loop.quit)
            wait_loop.exec()

        raise RuntimeError(f"Qt WebEngine fetch timed out: {last_result}")
    finally:
        try:
            page.deleteLater()
            profile.deleteLater()
            cleanup_loop = QEventLoop()
            QTimer.singleShot(100, cleanup_loop.quit)
            cleanup_loop.exec()
        except Exception:
            pass
        _cleanup_profile_root(profile_root)


class AIModeError(RuntimeError):
    """A google.com AI Mode request that failed for a named reason.

    ``kind`` (also ``error_type``, the name ``cdp_browser`` reads) tells the caller whether a
    retry can help: see the ``KIND_*`` constants. The helper process reports it next to the
    error text (``error_kind``), so it survives the process boundary.
    """

    def __init__(self, message: str, kind: str = "") -> None:
        super().__init__(message)
        self.kind = kind or ""
        self.error_type = self.kind or "api_error"


#: Google asks for a CAPTCHA ("unusual traffic"): the user has to clear it; never retried.
KIND_VERIFICATION = "verification"
#: Google's cookie consent page came up instead of AI Mode.
KIND_CONSENT = "consent"
#: A Google sign-in page came up, the saved session ended, or the model needs an account.
KIND_SIGN_IN = "sign_in"
#: AI Mode did not open (region, account or browser without AI Mode).
KIND_UNAVAILABLE = "unavailable"
#: AI Mode answered with web results or "no response available" instead of an AI answer.
KIND_NO_ANSWER = "no_answer"
#: AI Mode said the content was not generated.
KIND_GENERATION_FAILURE = "generation_failure"
#: The prompt box text is too long for AI Mode even with the texts moved into attachments.
KIND_TOO_LONG = "too_long"
#: An attachment was not accepted or did not finish uploading.
KIND_UPLOAD = "upload"
#: AI Mode closed the HTML code block before the end of the answer (the text is too long for
#: one answer of this model): the tags of the rest are lost, so the answer is refused.
KIND_INCOMPLETE = "incomplete"
#: The Chrome/Edge page of the signed-in profile failed (``cdp_browser`` error types).
KIND_BROWSER = "browser_error"
KIND_CANCELLED = "cancelled"
#: Kinds a new attempt cannot fix: the unified client reports them at once.
FATAL_KINDS = (KIND_VERIFICATION, KIND_CONSENT, KIND_SIGN_IN, KIND_UNAVAILABLE, KIND_TOO_LONG,
               KIND_INCOMPLETE)
#: The entry of AI Mode's model menu that plain ``search/gemini`` asks for (where it exists).
AUTO_MODEL = "auto"

#: A request this short, with plain text and no images, is typed into AI Mode's prompt box as
#: it is; anything longer, any HTML and any image goes in as an attachment.
INLINE_PROMPT_CHARS = 2000
#: AI Mode puts the prompt box text into the result URL (``q=``) and answers with web results
#: only once that URL passes roughly 48,000 characters (measured: 47,700 answered, 53,600 did
#: not; a Hangul character is nine URL characters). The instructions move into an attached file
#: before that.
MAX_PROMPT_URL_CHARS = 40000
#: Names of the attached files.
HTML_FILE_NAME = "chapter.html"
TEXT_FILE_NAME = "text.txt"
CONTEXT_FILE_NAME = "context.txt"
INSTRUCTIONS_FILE_NAME = "instructions.txt"
#: Attached texts are uploaded as plain text whatever their name: AI Mode then reads the raw
#: markup. Uploaded as ``text/html`` it reads the rendered page and the reply loses tags.
TEXT_FILE_MIME = "text/plain"
_IMAGE_EXTENSIONS = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp",
    "image/gif": "gif", "image/bmp": "bmp", "image/avif": "avif", "image/heic": "heic",
    "image/heif": "heif", "image/tiff": "tiff",
}
_NO_FOLLOW_UP = "with no follow-up questions or commentary"


def _content_images(content: Any) -> List[tuple[str, bytes]]:
    """The images of one message's content as ``(mime type, bytes)``.

    Reads the shapes the unified client sends: ``image_url`` parts (a ``data:`` URL, directly
    or under ``url``), ``input_image`` parts and Anthropic-style ``image`` parts with a base64
    ``source``. An image given only by a web address is skipped: AI Mode takes uploads.
    """
    import base64

    images: List[tuple[str, bytes]] = []
    if not isinstance(content, list):
        return images
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type not in ("image_url", "input_image", "image"):
            continue
        mime = ""
        encoded = ""
        source = item.get("source")
        if isinstance(source, dict) and source.get("data"):
            mime = str(source.get("media_type") or source.get("mime_type") or "image/png")
            encoded = str(source.get("data") or "")
        else:
            url = item.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            url = str(url or item.get("url") or "")
            match = re.match(r"^data:([\w.+/-]+)?(?:;[\w=.+-]+)*;base64,(.*)$", url, flags=re.DOTALL | re.IGNORECASE)
            if not match:
                continue
            mime = match.group(1) or "image/png"
            encoded = match.group(2)
        try:
            data = base64.b64decode(re.sub(r"\s+", "", encoded), validate=False)
        except Exception:
            continue
        if data:
            images.append((mime.lower(), data))
    return images


def _message_text(content: Any) -> str:
    """``_content_to_text`` without the image placeholders (images go in as attachments)."""
    text = _content_to_text(content)
    return "\n".join(line for line in text.split("\n") if line.strip() != "[image omitted]").strip()


def _ai_mode_model_choice(model: str) -> str:
    """The AI Mode model a ``search/`` model name asks for, lower case; "" for the page's own.

    ``search/gemini`` is AI Mode's own choice: its "Auto" entry where the menu has one, else the
    model the page has selected. ``search/gemini-<name>`` picks the entry of AI Mode's model
    menu whose name contains ``<name>`` (``search/gemini-4-carbon``: "4 Carbon",
    ``search/gemini-3.8-flash``: "3.8 Flash"); ``list_models()`` reads the names an account is
    offered. A signed-out page offers one model only.
    """
    actual = _strip_search_prefix(model).lower()
    if actual == DEFAULT_MODEL:
        return ""
    if actual.startswith(DEFAULT_MODEL + "-") and len(actual) > len(DEFAULT_MODEL) + 1:
        return actual[len(DEFAULT_MODEL) + 1:].replace("_", " ").replace("-", " ").strip()
    raise RuntimeError(
        f"Gemini Free supports search/{DEFAULT_MODEL} and search/{DEFAULT_MODEL}-<AI Mode model>; "
        f"got search/{actual}"
    )


def _text_file(name: str, text: str) -> Dict[str, Any]:
    return {"name": name, "mime": TEXT_FILE_MIME, "kind": "file", "data": str(text).encode("utf-8")}


#: An HTML line longer than this is broken up before it is attached.
MAX_HTML_LINE_CHARS = 1500
_BLOCK_END = re.compile(
    r"(</(?:p|div|h[1-6]|li|ul|ol|table|tr|blockquote|section|article|head|body|html)>)(?=<)", re.IGNORECASE)


def _one_block_per_line(markup: str) -> str:
    """``markup`` with a line break after each closing block tag that another tag follows.

    Chapters often come as one line of many thousand characters. The model returns HTML the
    way it got it, and AI Mode shows an HTML answer that is one very long line as an empty code
    block with the text, stripped of its tags, beside it. A break between two blocks changes
    nothing in the rendered chapter. Markup whose lines are short already is left alone.
    """
    text = str(markup or "")
    if max((len(line) for line in text.split("\n")), default=0) <= MAX_HTML_LINE_CHARS:
        return text
    return _BLOCK_END.sub(r"\1\n", text)


def _build_ai_mode_request(messages: Iterable[Dict[str, Any]], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """What one chat request puts on the AI Mode page: the prompt box text and the attachments.

    The system prompt is typed into the prompt box. The text to work on (the last user message)
    is attached as a file when it is HTML or long, so its size never meets AI Mode's prompt
    limit and its tags reach the model untouched; images are attached as images; earlier
    messages of the conversation go into ``context.txt``. A short plain request with nothing to
    attach is typed in as it is. Role labels ("System instructions:", "User:") are never
    written: with an attachment AI Mode answers such a prompt with "no response available".
    """
    system_parts: List[str] = []
    turns: List[tuple[str, str]] = []
    images: List[tuple[str, bytes]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = message.get("content")
        text = _message_text(content)
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        if role != "assistant":
            role = "user"
            images.extend(_content_images(content))
        if text:
            turns.append((role, text))

    payload = ""
    for index in range(len(turns) - 1, -1, -1):
        if turns[index][0] == "user":
            payload = turns.pop(index)[1]
            break
    history = turns
    system_text = "\n\n".join(system_parts).strip()
    expects_html = _content_looks_like_html_payload(payload)
    expects_markdown = bool(
        not expects_html and _messages_expect_markdown_response([{"role": "user", "content": payload}])
    )
    attach_payload = bool(payload) and (
        expects_html or len(system_text) + len(payload) > INLINE_PROMPT_CHARS
    )

    files: List[Dict[str, Any]] = []
    notes: List[str] = []
    payload_name = HTML_FILE_NAME if expects_html else TEXT_FILE_NAME
    if history:
        context = "\n\n".join(
            ("Earlier request:\n" if role == "user" else "Earlier reply:\n") + text for role, text in history
        )
        files.append(_text_file(CONTEXT_FILE_NAME, context))
        notes.append(
            f"The attached file {CONTEXT_FILE_NAME} holds the earlier messages of this conversation, "
            "for reference only."
        )
    if attach_payload:
        files.append(_text_file(payload_name, _one_block_per_line(payload) if expects_html else payload))
    for number, (mime, data) in enumerate(images, start=1):
        extension = _IMAGE_EXTENSIONS.get(mime, "png")
        files.append({
            "name": f"image{number:03d}.{extension}",
            "mime": mime if mime in _IMAGE_EXTENSIONS else "image/png",
            "kind": "image",
            "data": data,
        })
    if images:
        notes.append("The attached image belongs to the request." if len(images) == 1
                     else f"The {len(images)} attached images belong to the request.")

    task = ""
    if attach_payload:
        task = (f"Apply the instructions above to the attached file {payload_name}." if system_text
                else f"Read the attached file {payload_name} and do what it asks.")
        task += (f" Reply with only the complete result as HTML inside one ```html code block, {_NO_FOLLOW_UP}."
                 if expects_html else f" Reply with only the complete result, {_NO_FOLLOW_UP}.")

    def compose(instructions: str, task_line: str) -> str:
        inline = payload if payload and not attach_payload else ""
        tail = " ".join(part for part in (*notes, task_line) if part)
        return "\n\n".join(part for part in (instructions, inline, tail) if part).strip()

    prompt = compose(system_text, task)
    instructions_attached = False
    if system_text and len(_build_search_url(prompt)) > MAX_PROMPT_URL_CHARS:
        # A long system prompt (a glossary appended to it): it goes into a file as well.
        instructions_attached = True
        files.insert(0, _text_file(INSTRUCTIONS_FILE_NAME, system_text))
        lead = f"Follow the instructions in the attached file {INSTRUCTIONS_FILE_NAME}."
        prompt = compose(lead, task.replace("Apply the instructions above", "Apply them", 1))
    return {
        "prompt": prompt,
        "files": files,
        "model": _ai_mode_model_choice(model),
        "expects_html": expects_html,
        "expects_markdown": expects_markdown,
        "instructions_attached": instructions_attached,
        "prompt_chars": len(system_text) + len(payload) + sum(len(text) for _role, text in history),
    }


def _search_params() -> Dict[str, str]:
    params = dict(DEFAULT_SEARCH_PARAMS)
    extra = os.getenv("GEMINI_FREE_SEARCH_PARAMS", "").strip()
    if extra:
        for key, value in parse_qsl(extra.lstrip("?"), keep_blank_values=True):
            if key:
                params[key] = value
    return params


def _build_search_base_url() -> str:
    """google.com with AI Mode on and an empty prompt box: the page every request opens."""
    base = os.getenv("GEMINI_FREE_SEARCH_BASE_URL", SEARCH_BASE_URL).strip() or SEARCH_BASE_URL
    params = _search_params()
    params.pop("q", None)
    return f"{base}?{urlencode(params)}"


def _build_search_url(prompt: str) -> str:
    """The AI Mode address of ``prompt`` (what the page navigates to when the prompt is sent)."""
    base = os.getenv("GEMINI_FREE_SEARCH_BASE_URL", SEARCH_BASE_URL).strip() or SEARCH_BASE_URL
    params = _search_params()
    params["q"] = prompt
    return f"{base}?{urlencode(params)}"


# Page script: the state of the AI Mode page and its latest answer (a JSON string).
#
# The answer is the page's answer column (``data-container-id="main-col"``; the whole turn,
# ``data-subtree="aimc"``, when that name is gone). It is read from the DOM, not from
# ``innerText``, so the sources chips and the feedback form (buttons, hidden blocks) stay out and
# a code block comes back whole. Visibility is judged by ``display`` / ``visibility`` alone: a
# hidden page (the mobile WebView, a headless tab) lays out nothing, and AI Mode's wrappers are
# ``display: contents``, so box sizes say nothing there. ``complete`` is AI Mode's own "response
# is ready" announcement (``hl=en``).
AI_MODE_SNAPSHOT_SCRIPT = r"""
(() => {
  const doc = document;
  const styleOf = (element) => window.getComputedStyle(element);
  const hidden = (element) => {
    if (element.getAttribute("aria-hidden") === "true") return true;
    const style = styleOf(element);
    return style.display === "none" || style.visibility === "hidden";
  };
  const SKIP = new Set(["BUTTON", "SVG", "TEXTAREA", "INPUT", "SELECT", "SCRIPT", "STYLE", "NOSCRIPT",
    "IMG", "VIDEO", "AUDIO", "IFRAME", "CANVAS"]);
  const BLOCK = /^(block|flex|grid|list-item|table|flow-root)$/;
  const headingLevel = (element) => {
    const tag = String(element.tagName || "").toUpperCase();
    if (/^H[1-6]$/.test(tag)) return parseInt(tag.slice(1), 10);
    if ((element.getAttribute("role") || "") === "heading") {
      return Math.min(6, Math.max(1, parseInt(element.getAttribute("aria-level") || "2", 10) || 2));
    }
    return 0;
  };
  const answers = Array.from(doc.querySelectorAll('[data-container-id="main-col"]'));
  const turns = Array.from(doc.querySelectorAll('[data-subtree="aimc"]'));
  const root = answers.length ? answers[answers.length - 1] : (turns.length ? turns[turns.length - 1] : null);
  let topLevel = 7;
  if (root) {
    for (const element of root.querySelectorAll('h1,h2,h3,h4,h5,h6,[role="heading"]')) {
      const level = headingLevel(element);
      if (level && level < topLevel) topLevel = level;
    }
  }
  const tidy = (value) => String(value || "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  const render = (node, markdown) => {
    if (node.nodeType === 3) return String(node.nodeValue || "").replace(/ /g, " ").replace(/\s+/g, " ");
    if (node.nodeType !== 1) return "";
    const tag = String(node.tagName || "").toUpperCase();
    if (SKIP.has(tag) || hidden(node)) return "";
    if (tag === "BR") return "\n";
    if (tag === "HR") return "\n\n---\n\n";
    if (tag === "PRE") return "\n\n```\n" + String(node.textContent || "").replace(/\s+$/, "") + "\n```\n\n";
    const inner = (element) => Array.from(element.childNodes).map((child) => render(child, markdown)).join("");
    const level = headingLevel(node);
    if (level) {
      const title = tidy(inner(node));
      if (!title) return "";
      return "\n\n" + (markdown ? "#".repeat(Math.max(1, level - topLevel + 1)) + " " : "") + title + "\n\n";
    }
    if (tag === "UL" || tag === "OL") {
      const rows = [];
      for (const item of Array.from(node.children)) {
        if (String(item.tagName || "").toUpperCase() !== "LI" || hidden(item)) continue;
        const body = tidy(inner(item));
        if (!body) continue;
        rows.push((tag === "OL" ? (rows.length + 1) + ". " : "- ") + body.replace(/\n/g, "\n  "));
      }
      return rows.length ? "\n\n" + rows.join("\n") + "\n\n" : "";
    }
    if (tag === "TABLE") {
      const rows = Array.from(node.querySelectorAll("tr")).map((row) =>
        Array.from(row.querySelectorAll("th,td")).map((cell) => tidy(inner(cell)).replace(/\n+/g, " ")).join(" | "));
      return rows.length ? "\n\n" + rows.filter(Boolean).join("\n") + "\n\n" : "";
    }
    const text = inner(node);
    if (markdown && (tag === "STRONG" || tag === "B")) return text.trim() ? "**" + text.trim() + "**" : "";
    if (markdown && (tag === "EM" || tag === "I")) return text.trim() ? "*" + text.trim() + "*" : "";
    if (tag === "P" || tag === "LI" || tag === "BLOCKQUOTE" || BLOCK.test(styleOf(node).display)) {
      const body = tidy(text);
      return body ? "\n\n" + body + "\n\n" : "";
    }
    return text;
  };
  const bodyText = doc.body ? doc.body.innerText || "" : "";
  const announcements = Array.from(doc.querySelectorAll("[aria-live]")).map((element) => element.textContent || "");
  return JSON.stringify({
    url: location.href,
    title: doc.title || "",
    ready: doc.readyState || "",
    text: bodyText.slice(0, 6000),
    aiMode: turns.length > 0 || answers.length > 0 || !!doc.querySelector('[data-xid="aim-mars-turn-root"]'),
    promptBox: doc.querySelectorAll("textarea").length > 0,
    answerFound: !!root,
    answerText: root ? tidy(render(root, false)) : "",
    answerMarkdown: root ? tidy(render(root, true)) : "",
    codeBlocks: root ? Array.from(root.querySelectorAll("pre")).filter((block) => !hidden(block))
      .map((block) => String(block.textContent || "")) : [],
    complete: announcements.some((text) => /AI Mode response is ready/i.test(text)),
    pending: root ? root.querySelectorAll('[data-complete="false"]').length : 0,
    htmlLength: doc.documentElement ? doc.documentElement.outerHTML.length : 0
  });
})()
"""

# Page script (shared head): ``modelItems()`` is the model entries of AI Mode's attach menu, the
# menu that holds "Add images" / "Add files". Other menus of the page have radio entries too
# (the phone layout's search filters: "Past hour", "Verbatim").
_AI_MODE_MODEL_ITEMS_JS = r"""
  const textOf = (item) => (item.innerText || item.textContent || "").replace(/\s+/g, " ").trim();
  const modelItems = () => {
    const menu = Array.from(document.querySelectorAll('[role="menu"]')).find((entry) =>
      Array.from(entry.querySelectorAll('[role="menuitem"]')).some((item) =>
        /^add (files|images)/i.test(item.getAttribute("aria-label") || textOf(item))));
    return menu ? Array.from(menu.querySelectorAll('[role="menuitemradio"]')) : [];
  };
"""

# Page script: open AI Mode's "Add files, tools, and select a model" menu when its file inputs
# and model entries are not in the page yet, and list them (JSON: ok, inputs, models).
AI_MODE_MENU_SCRIPT = r"""
(() => {
  const labelOf = (element) => element.getAttribute("aria-label") || "";
__MODEL_ITEMS__
  const buttons = Array.from(document.querySelectorAll('button,[role="button"]'));
  const opener = buttons.find((button) => /^add files/i.test(labelOf(button))) ||
    buttons.find((button) => button.getAttribute("aria-haspopup") === "menu" &&
      /\b(add|attach|upload)\b/i.test(labelOf(button)));
  const inputs = Array.from(document.querySelectorAll('input[type="file"]'));
  const models = modelItems().map((item) => ({
    label: textOf(item),
    checked: item.getAttribute("aria-checked") === "true",
    disabled: item.getAttribute("aria-disabled") === "true"
  }));
  let clicked = false;
  if (opener && !inputs.length && !models.length && opener.getAttribute("aria-expanded") !== "true") {
    opener.click();
    clicked = true;
  }
  return JSON.stringify({
    ok: inputs.length > 0 || models.length > 0,
    opener: !!opener,
    clicked,
    inputs: inputs.map((input) => input.accept || ""),
    models
  });
})()
""".replace("__MODEL_ITEMS__", _AI_MODE_MODEL_ITEMS_JS.strip("\n"))

#: Characters of base64 one page script carries when a file is handed to the page.
FILE_CHUNK_CHARS = 240000


def _ai_mode_select_model_script(choice: str) -> str:
    """Page script: tick the model menu entry whose name contains ``choice`` (JSON: ok, label,
    error, models). The name is the part before the "•" ("4 Carbon • Most complex work")."""
    return r"""
(() => {
  const want = __WANT__;
__MODEL_ITEMS__
  const items = modelItems();
  const models = items.map(textOf);
  const item = items.find((entry) => textOf(entry).toLowerCase().split("•")[0].includes(want)) ||
    items.find((entry) => textOf(entry).toLowerCase().includes(want));
  if (!item) return JSON.stringify({ok: false, error: "not listed", models});
  if (item.getAttribute("aria-disabled") === "true" && item.getAttribute("aria-checked") !== "true") {
    return JSON.stringify({ok: false, error: "disabled", models});
  }
  const already = item.getAttribute("aria-checked") === "true";
  if (!already) item.click();
  return JSON.stringify({ok: true, label: textOf(item), already, models});
})()
""".replace("__MODEL_ITEMS__", _AI_MODE_MODEL_ITEMS_JS.strip("\n")).replace(
        "__WANT__", json.dumps(str(choice or "").lower(), ensure_ascii=False))


def _ai_mode_stage_file_script(index: int, name: str, mime: str, kind: str, chunk: str) -> str:
    """Page script: add one base64 ``chunk`` of attachment ``index`` to the page's staging list
    (the files reach the page in pieces; returns the staged length)."""
    return (
        "(() => { const staged = (window.__glossarionAiFiles = window.__glossarionAiFiles || []);"
        f" const index = {int(index)};"
        " staged[index] = staged[index] || {name: " + json.dumps(str(name), ensure_ascii=False)
        + ", mime: " + json.dumps(str(mime)) + ", kind: " + json.dumps(str(kind)) + ', b64: ""};'
        " staged[index].b64 += " + json.dumps(str(chunk)) + ";"
        " return staged[index].b64.length; })()"
    )


def _ai_mode_attach_script(kind: str) -> str:
    """Page script: give the staged attachments of ``kind`` ("file" / "image") to AI Mode's file
    input, as a file picker would (JSON: ok, names, error, inputs). Images go to the image input
    (its ``accept`` lists image types), everything else to the general one."""
    return r"""
(() => {
  const kind = __KIND__;
  const staged = (window.__glossarionAiFiles || []).filter((file) => file && file.kind === kind);
  const inputs = Array.from(document.querySelectorAll('input[type="file"]'));
  const takesImages = (input) => /image\//i.test(input.accept || "");
  // the phone layout has camera inputs (``capture``) before its image picker
  const picker = (entry) => takesImages(entry) && !entry.hasAttribute("capture") && entry.multiple;
  const input = kind === "image"
    ? (inputs.find(picker) || inputs.find(takesImages) || inputs.find((entry) => !entry.accept))
    : (inputs.find((entry) => !takesImages(entry)) || null);
  if (!input) {
    return JSON.stringify({ok: false, error: "AI Mode file input not found", inputs: inputs.map((entry) => entry.accept || "")});
  }
  const transfer = new DataTransfer();
  for (const file of staged) {
    const binary = atob(file.b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    transfer.items.add(new File([bytes], file.name, {type: file.mime}));
  }
  input.files = transfer.files;
  input.dispatchEvent(new Event("change", {bubbles: true}));
  return JSON.stringify({ok: true, names: staged.map((file) => file.name)});
})()
""".replace("__KIND__", json.dumps(str(kind)))


def _ai_mode_upload_state_script(names: Iterable[str]) -> str:
    """Page script: which of the attachments ``names`` show as a chip in the prompt box, and how
    many uploads the page finished (JSON: chips, uploads; ``uploads`` -1 when the page cannot
    tell)."""
    return r"""
(() => {
  const names = __NAMES__;
  const labels = new Set(Array.from(document.querySelectorAll("[aria-label]")).map((element) => element.getAttribute("aria-label")));
  let uploads = -1;
  try {
    uploads = performance.getEntriesByType("resource")
      .filter((entry) => /\/crupload(\?|$)/.test(entry.name) && entry.responseEnd > 0).length;
  } catch (error) {}
  return JSON.stringify({chips: names.filter((name) => labels.has(name)), uploads});
})()
""".replace("__NAMES__", json.dumps([str(name) for name in names], ensure_ascii=False))


GENERATION_FAILURE_MARKERS = (
    "something went wrong and the content wasn't generated",
    "something went wrong and the content was not generated",
    "content wasn't generated",
    "content was not generated",
)
#: AI Mode's answer when it has nothing to say for a request.
NO_RESPONSE_MARKER = "no response available for this search"
#: Lines of AI Mode's own that are not part of an answer.
_ANSWER_NOISE_LINES = ("use code with caution.",)
_ANSWER_NOISE_PREFIXES = ("ai can make mistakes", "ai responses may include mistakes", "ai mode reply for")


def _google_blocked(page_data: Dict[str, Any]) -> bool:
    url = str(page_data.get("url") or "").lower()
    text = str(page_data.get("text") or "")
    text_l = text.lower()
    return (
        "google.com/sorry/" in url
        or "our systems have detected unusual traffic" in text_l
        or ("about this page" in text_l and "unusual traffic" in text_l)
        or "to continue, please type the characters" in text_l
    )


def _page_problem(page_data: Dict[str, Any]) -> Optional[tuple[str, str]]:
    """``(kind, message)`` when the page is not AI Mode but a page of Google's that needs the
    user: a verification (CAPTCHA) page, the cookie consent page, a sign-in page, or the notice
    that AI Mode is not available. None for anything else (AI Mode itself, a page still loading).
    """
    url = str(page_data.get("url") or "")
    host = (urlparse(url).netloc or "").lower()
    text_l = str(page_data.get("text") or "").lower()
    if _google_blocked(page_data):
        return (
            KIND_VERIFICATION,
            "Google Search returned a verification page instead of AI Mode. "
            "Open Google in a regular browser on this network and clear the verification, or retry later.",
        )
    if host.startswith("consent.") or "before you continue to google" in text_l:
        return (
            KIND_CONSENT,
            "Google showed its cookie consent page instead of AI Mode, and Glossarion does not answer it. "
            "Sign in with the Google Login button next to the model (the consent is saved with the "
            "account's browser profile), or use another route.",
        )
    if host.startswith("accounts.google."):
        return (
            KIND_SIGN_IN,
            "Google asked to sign in before showing AI Mode. Use the Google Login button next to the model.",
        )
    if re.search(r"ai mode (is not|isn't|isn’t) (currently )?available", text_l) or (
        "ai mode" in text_l and "not available in your" in text_l
    ):
        return (
            KIND_UNAVAILABLE,
            "Google says AI Mode is not available here (region, account or language). "
            "search/gemini needs google.com AI Mode.",
        )
    return None


def _contains_generation_failure(lines: Iterable[str]) -> bool:
    for line in lines or []:
        current = str(line or "").strip().lower()
        if any(marker in current for marker in GENERATION_FAILURE_MARKERS):
            return True
    return False


def _raise_if_generation_failure_text(value: str) -> None:
    lines = [line.strip() for line in str(value or "").replace("\r", "\n").split("\n") if line.strip()]
    if _contains_generation_failure(lines):
        raise AIModeError(
            "Google Search AI Mode returned a generation failure: "
            "Something went wrong and the content wasn't generated.",
            KIND_GENERATION_FAILURE,
        )


def _web_results_only(answer_text: str) -> bool:
    """AI Mode's answer when it does not generate one: a list of web results."""
    head = str(answer_text or "").strip()[:300].lower()
    return bool(re.match(r"^\d+\s+sites?\b", head)) or "here are top web results" in head


def _code_block_payload(block: str) -> str:
    """The code of one answer code block. AI Mode sometimes closes the block late: the model's
    own closing fence and whatever it wrote after it are then inside the block, and are cut."""
    text = str(block or "").replace("\r\n", "\n")
    match = re.search(r"(?m)^[ \t]*```[ \t]*$", text)
    if match:
        text = text[: match.start()]
    return text.strip("\n").rstrip()


def _clean_answer_text(value: str) -> str:
    """An answer without AI Mode's own lines: the "AI can make mistakes" note, a code block's
    language label (it becomes the fence's) and its "Use code with caution." line, and the
    model's ``<FollowUp>`` block."""
    text = str(value or "").replace("\r\n", "\n")
    text = re.sub(r"<FollowUp>.*?(</FollowUp>|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?m)^([a-z][a-z0-9+#.-]{0,19})\n+```\n", r"```\1\n", text)
    lines = []
    for line in text.split("\n"):
        probe = line.strip().lower()
        if probe in _ANSWER_NOISE_LINES or any(probe.startswith(prefix) for prefix in _ANSWER_NOISE_PREFIXES):
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _extract_rendered_content(
    page_data: Dict[str, Any],
    *,
    prefer_html: bool = False,
    prefer_markdown: bool = False,
    prompt_url_chars: int = 0,
) -> str:
    """The model reply in an AI Mode page snapshot (``AI_MODE_SNAPSHOT_SCRIPT``).

    A request for HTML gets the code block(s) of the answer: AI Mode shows HTML only inside
    one, the request asks for it there. Otherwise the answer text, as Markdown when the request
    works in Markdown. Raises :class:`AIModeError` for a page that is not an answer.
    """
    problem = _page_problem(page_data)
    if problem:
        raise AIModeError(problem[1], problem[0])
    title = str(page_data.get("title") or "").strip()
    if page_data.get("notAiMode"):
        raise AIModeError(
            "AI Mode did not open: Google served another page "
            f"(title={title!r}, url={_safe_url_for_log(page_data.get('url'))}). "
            "AI Mode may not be available for this region, account or browser.",
            KIND_UNAVAILABLE,
        )
    answer_text = str(page_data.get("answerText") or "").strip()
    markdown = str(page_data.get("answerMarkdown") or "").strip()
    _raise_if_generation_failure_text(answer_text or str(page_data.get("text") or ""))
    if NO_RESPONSE_MARKER in answer_text.lower():
        raise AIModeError(
            "Google AI Mode had no response for this request (\"no response available for this search\").",
            KIND_NO_ANSWER,
        )
    if _web_results_only(answer_text):
        if prompt_url_chars > MAX_PROMPT_URL_CHARS:
            raise AIModeError(
                "Google AI Mode answered with web results only: the prompt box text is too long for it "
                f"(its address is {prompt_url_chars:,} characters, AI Mode answers up to about 48,000). "
                "Shorten the system prompt or the glossary appended to it.",
                KIND_TOO_LONG,
            )
        raise AIModeError("Google AI Mode answered with web results only instead of an AI answer.", KIND_NO_ANSWER)

    if prefer_html:
        blocks = [_code_block_payload(block) for block in (page_data.get("codeBlocks") or [])]
        blocks = [block for block in blocks if block.strip()]
        code = "\n".join(blocks).strip()
        outside = _clean_answer_text(re.sub(r"(?s)```.*?(```|\Z)", "", _clean_answer_text(answer_text)))
        if len(outside) > max(300, len(code) // 5):
            # HTML outside a code block is shown rendered: its tags are gone from the page.
            raise AIModeError(
                "Google AI Mode did not keep the HTML answer inside its code block "
                f"({len(code):,} characters inside it, {len(outside):,} after it without their tags). "
                "The text is too long for one answer of this AI Mode model: translate in smaller chunks "
                "(lower the input token limit)"
                + ("" if page_data.get("signed_in") else ", or sign in with Google Login for the larger models")
                + ".",
                KIND_INCOMPLETE,
            )
        if code:
            return code
    content = _clean_answer_text((markdown or answer_text) if prefer_markdown else (answer_text or markdown))
    if not content:
        raise RuntimeError(f"Google AI Mode returned an empty answer. title={title!r}")
    return content


def _ai_mode_set_prompt_script(prompt: str) -> str:
    """Page script: put ``prompt`` into the AI Mode textarea (JSON result: ok, error, counts)."""
    return f"""
(() => {{
  const promptText = {json.dumps(prompt, ensure_ascii=False)};
  const visibleScore = (element) => {{
    if (!element) return 0;
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    if (style.visibility === "hidden" || style.display === "none") return 0;
    if (rect.width <= 0 || rect.height <= 0) return 0;
    return rect.width * rect.height;
  }};
  const textareas = Array.from(document.querySelectorAll("textarea"));
  const ranked = textareas
    .map((element) => ({{element, score: visibleScore(element)}}))
    .sort((a, b) => b.score - a.score);
  const textarea =
    ranked.find((item) => item.score > 0 && /ask/i.test(item.element.placeholder || ""))?.element ||
    ranked.find((item) => item.score > 0)?.element ||
    textareas.find((element) => /ask/i.test(element.placeholder || "")) ||
    textareas[0];
  if (!textarea) {{
    return JSON.stringify({{ok: false, error: "AI Mode textarea not found", textareaCount: textareas.length}});
  }}
  textarea.scrollIntoView({{block: "center", inline: "center"}});
  textarea.focus();
  const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
  if (valueSetter) {{
    valueSetter.call(textarea, promptText);
  }} else {{
    textarea.value = promptText;
  }}
  textarea.dispatchEvent(new Event("focus", {{bubbles: true}}));
  try {{
    textarea.dispatchEvent(new InputEvent("input", {{
      bubbles: true,
      inputType: "insertText",
      data: promptText
    }}));
  }} catch (error) {{
    textarea.dispatchEvent(new Event("input", {{bubbles: true}}));
  }}
  textarea.dispatchEvent(new Event("change", {{bubbles: true}}));
  return JSON.stringify({{
    ok: true,
    textareaCount: textareas.length,
    valueLength: textarea.value.length,
    placeholder: textarea.placeholder || ""
  }});
}})()
"""


# Page script: press AI Mode's Send button (JSON result: ok, error, label, button counts).
AI_MODE_CLICK_SEND_SCRIPT = """
(() => {
  const visibleScore = (element) => {
    if (!element) return 0;
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    if (style.visibility === "hidden" || style.display === "none") return 0;
    if (rect.width <= 0 || rect.height <= 0) return 0;
    return rect.width * rect.height;
  };
  const labelOf = (element) => [
    element.getAttribute("aria-label") || "",
    element.getAttribute("title") || "",
    element.innerText || "",
    element.value || ""
  ].join(" ").replace(/\\s+/g, " ").trim();
  const buttons = Array.from(document.querySelectorAll('button,[role="button"],input[type="submit"]'));
  const ranked = buttons
    .map((element) => ({element, label: labelOf(element), score: visibleScore(element)}))
    .sort((a, b) => b.score - a.score);
  const exactSend = (item) => /^send$/i.test(item.element.getAttribute("aria-label") || "");
  const enabled = (item) => !item.element.disabled && item.element.getAttribute("aria-disabled") !== "true";
  const candidate =
    ranked.find((item) => exactSend(item) && enabled(item)) ||
    ranked.find((item) => /\\bsend\\b/i.test(item.label) && enabled(item)) ||
    ranked.find((item) => exactSend(item));
  if (!candidate) {
    return JSON.stringify({
      ok: false,
      error: "AI Mode send button not ready",
      buttonCount: buttons.length,
      labels: ranked.slice(0, 8).map((item) => item.label)
    });
  }
  candidate.element.click();
  return JSON.stringify({ok: true, label: candidate.label, buttonCount: buttons.length});
})()
"""


def _run_page_json(run_js: Callable[..., Any], script: str, js_timeout_ms: int = 15000) -> Dict[str, Any]:
    raw = run_js(script, js_timeout_ms=js_timeout_ms)
    try:
        value = json.loads(raw or "{}")
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _raise_if_cancelled() -> None:
    if _is_cancelled():
        raise RuntimeError("stream cancelled")


def _open_ai_mode_menu(run_js: Callable[..., Any], wait_ms: Callable[[int], None], *, seconds: float = 12.0) -> Dict[str, Any]:
    """Open AI Mode's attach / model menu and return its listing (``AI_MODE_MENU_SCRIPT``)."""
    deadline = time.time() + seconds
    state: Dict[str, Any] = {}
    while True:
        _raise_if_cancelled()
        state = _run_page_json(run_js, AI_MODE_MENU_SCRIPT, js_timeout_ms=10000)
        if state.get("ok"):
            return state
        if time.time() >= deadline:
            raise RuntimeError(
                "AI Mode's attach / model menu did not open"
                + ("" if state.get("opener") else " (its button was not found)")
            )
        wait_ms(300)


def _select_ai_mode_model(
    choice: str,
    run_js: Callable[..., Any],
    wait_ms: Callable[[int], None],
    *,
    required: bool = True,
) -> str:
    """Tick the AI Mode model named by ``choice``; returns its menu label.

    ``required`` False is plain ``search/gemini`` asking for "Auto": a page without that entry
    (or without the menu) keeps the model it has, and "" is returned.
    """
    try:
        _open_ai_mode_menu(run_js, wait_ms, seconds=12.0 if required else 4.0)
    except RuntimeError:
        if required or _is_cancelled():
            raise
        return ""
    state = _run_page_json(run_js, _ai_mode_select_model_script(choice), js_timeout_ms=10000)
    if not state.get("ok") and not required:
        return ""
    if not state.get("ok"):
        offered = ", ".join(str(label) for label in (state.get("models") or [])) or "none"
        raise AIModeError(
            f"AI Mode does not offer the model {choice!r} here (offered: {offered}). "
            "Signed out, AI Mode has one model: use the Google Login button next to the model, "
            f"or pick search/{DEFAULT_MODEL}.",
            KIND_SIGN_IN,
        )
    if not state.get("already"):
        wait_ms(800)
    return str(state.get("label") or choice)


def _read_ai_mode_models(run_js: Callable[..., Any], wait_ms: Callable[[int], None]) -> List[str]:
    """The labels of AI Mode's model menu on this page (the account's models)."""
    deadline = time.time() + 6
    while True:
        state = _open_ai_mode_menu(run_js, wait_ms)
        labels = [str(item.get("label") or "") for item in (state.get("models") or []) if item.get("label")]
        if labels or time.time() >= deadline:
            return labels
        wait_ms(300)  # the menu's file inputs can be in the page a moment before its model entries


def _model_ids(labels: Iterable[str]) -> List[str]:
    """``search/gemini`` model names for AI Mode's model menu labels.

    "4 Carbon • Most complex work" is ``search/gemini-4-carbon``; "Auto" is plain
    ``search/gemini``, which is always listed first.
    """
    names = [f"search/{DEFAULT_MODEL}"]
    for label in labels or []:
        slug = re.sub(r"[^a-z0-9.]+", "-", str(label).split("\u2022")[0].strip().lower()).strip("-.")
        if not slug or slug == AUTO_MODEL:
            continue
        name = f"search/{DEFAULT_MODEL}-{slug}"
        if name not in names:
            names.append(name)
    return names


def _attach_ai_mode_files(
    files: List[Dict[str, Any]],
    run_js: Callable[..., Any],
    wait_ms: Callable[[int], None],
    *,
    deadline: Optional[float],
) -> Dict[str, Any]:
    """Attach ``files`` to the AI Mode prompt box and wait until the page uploaded them.

    The files are handed to the page in base64 pieces, given to its file inputs as a file
    picker would, and shown by AI Mode as chips in the prompt box while it uploads them.
    """
    import base64

    for index, item in enumerate(files):
        encoded = base64.b64encode(item["data"]).decode("ascii")
        for start in range(0, max(1, len(encoded)), FILE_CHUNK_CHARS):
            _raise_if_cancelled()
            run_js(
                _ai_mode_stage_file_script(index, item["name"], item["mime"], item["kind"],
                                           encoded[start:start + FILE_CHUNK_CHARS]),
                js_timeout_ms=20000,
            )
    names = [str(item["name"]) for item in files]
    for kind in ("file", "image"):
        if not any(item["kind"] == kind for item in files):
            continue
        _open_ai_mode_menu(run_js, wait_ms)
        state = _run_page_json(run_js, _ai_mode_attach_script(kind), js_timeout_ms=20000)
        if not state.get("ok"):
            raise AIModeError(str(state.get("error") or "AI Mode file input not found"), KIND_UPLOAD)
        wait_ms(500)

    upload_seconds = max(10, _env_int("GEMINI_FREE_UPLOAD_TIMEOUT", 90))
    upload_deadline = time.time() + upload_seconds
    if deadline is not None:
        upload_deadline = min(upload_deadline, deadline)
    # Without the page's upload count (a renamed endpoint): how long the chips are given.
    settle_seconds = max(2, _env_int("GEMINI_FREE_UPLOAD_SETTLE_SECONDS", 15))
    chips_since: Optional[float] = None
    state = {}
    while True:
        _raise_if_cancelled()
        state = _run_page_json(run_js, _ai_mode_upload_state_script(names), js_timeout_ms=10000)
        chips = state.get("chips") or []
        uploads = state.get("uploads")
        if len(chips) >= len(names):
            if chips_since is None:
                chips_since = time.time()
            if isinstance(uploads, int) and uploads >= len(names):
                return state
            if time.time() - chips_since >= settle_seconds:
                return state
        if time.time() >= upload_deadline:
            missing = [name for name in names if name not in chips]
            raise AIModeError(
                "Google AI Mode did not take the attachment"
                + (f" {', '.join(missing)}" if missing else "s")
                + f" within {upload_seconds}s.",
                KIND_UPLOAD,
            )
        wait_ms(400)


def _wait_for_ai_mode_answer(
    run_js: Callable[..., Any],
    wait_ms: Callable[[int], None],
    *,
    base_url: str,
    timeout: int,
    wait_after_load_ms: Optional[int],
) -> Dict[str, Any]:
    """Poll the page until AI Mode's answer is finished and return its snapshot.

    Finished means AI Mode announced "response is ready"; without the announcement (another
    page language), an answer that stopped changing for ``GEMINI_FREE_STABLE_MS``. A page of
    Google's that needs the user (verification, consent, sign-in) is returned at once.
    """
    wait_after_ms = wait_after_load_ms
    if wait_after_ms is None:
        wait_after_ms = _env_int("GEMINI_FREE_WAIT_AFTER_LOAD_MS", 0)
    stable_ms = _env_int("GEMINI_FREE_STABLE_MS", 8000)
    settle_ms = _env_int("GEMINI_FREE_SETTLE_MS", 400)
    no_ai_mode_seconds = max(5, _env_int("GEMINI_FREE_NO_AI_MODE_SECONDS", 25))
    deadline = _timeout_deadline(timeout)
    started = time.time()
    min_ready_at = started + max(0, int(wait_after_ms)) / 1000.0
    last_result: Dict[str, Any] = {}
    last_answer = ""
    last_change_at = time.time()

    while not _deadline_expired(deadline):
        _raise_if_cancelled()
        result = _run_page_json(run_js, AI_MODE_SNAPSHOT_SCRIPT, js_timeout_ms=5000)
        if result:
            last_result = result
        if _page_problem(result):
            return result
        answer = str(result.get("answerText") or "")
        if _contains_generation_failure(answer.splitlines()):
            return result
        if answer != last_answer:
            last_answer = answer
            last_change_at = time.time()
        now = time.time()
        if result.get("answerFound") and answer and now >= min_ready_at:
            stable = (now - last_change_at) * 1000 >= max(0, int(stable_ms)) and not result.get("pending")
            if result.get("complete") or stable:
                if settle_ms > 0:
                    # the announcement can come a moment before the last piece of the answer
                    wait_ms(settle_ms)
                    final = _run_page_json(run_js, AI_MODE_SNAPSHOT_SCRIPT, js_timeout_ms=5000)
                    if final.get("answerFound") and str(final.get("answerText") or ""):
                        result = final
                return result
        if (
            result
            and not result.get("aiMode")
            and not result.get("promptBox")
            and str(result.get("ready") or "") == "complete"
            and now - started >= no_ai_mode_seconds
        ):
            result["notAiMode"] = True
            return result
        wait_ms(500)

    title = str(last_result.get("title") or "").strip()
    url = _safe_url_for_log(last_result.get("url") or base_url)
    raise RuntimeError(
        "Google AI Mode did not finish an answer in "
        f"{_timeout_label(timeout)}. title={title!r} url={url}"
    )


def _submit_ai_mode_request(
    request: Dict[str, Any],
    *,
    run_js: Callable[..., Any],
    wait_ms: Callable[[int], None],
    base_url: str,
    timeout: int,
    wait_after_load_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one request on a loaded AI Mode page: pick the model, attach the files, type the
    prompt, press Send and wait for the finished answer (its page snapshot).

    Shared by the QtWebEngine helper page, the signed-in Chrome/Edge tab and a browser-driver
    page (Glossarion Mobile's hidden WebView): same scripts and timing. ``run_js(script,
    js_timeout_ms)`` returns the expression's value or None; ``wait_ms(ms)`` lets the page run.
    """
    timeout_seconds = _timeout_seconds(timeout)
    deadline = _timeout_deadline(timeout)

    first = _run_page_json(run_js, AI_MODE_SNAPSHOT_SCRIPT, js_timeout_ms=5000)
    if _page_problem(first):
        return first

    if request.get("list_models"):
        first["models"] = _read_ai_mode_models(run_js, wait_ms)
        return first

    choice = str(request.get("model") or "")
    model_label = _select_ai_mode_model(choice or AUTO_MODEL, run_js, wait_ms, required=bool(choice))
    files = list(request.get("files") or [])
    upload_state: Dict[str, Any] = {}
    if files:
        upload_state = _attach_ai_mode_files(files, run_js, wait_ms, deadline=deadline)

    set_state = _run_page_json(run_js, _ai_mode_set_prompt_script(str(request.get("prompt") or "")), js_timeout_ms=15000)
    if not set_state.get("ok"):
        detail = str(set_state.get("error") or "AI Mode textarea not found")
        raise RuntimeError(detail)

    click_state: Dict[str, Any] = {}
    click_timeout = min(10, max(2, int(timeout_seconds or 10)))
    click_deadline = time.time() + click_timeout
    while time.time() < click_deadline:
        _raise_if_cancelled()
        wait_ms(250)
        click_state = _run_page_json(run_js, AI_MODE_CLICK_SEND_SCRIPT, js_timeout_ms=10000)
        if click_state.get("ok"):
            break
    if not click_state.get("ok"):
        detail = str(click_state.get("error") or "AI Mode send button not ready")
        raise RuntimeError(detail)

    result = _wait_for_ai_mode_answer(
        run_js,
        wait_ms,
        base_url=base_url,
        timeout=timeout,
        wait_after_load_ms=wait_after_load_ms,
    )
    result["search_base_url"] = base_url
    result["model_label"] = model_label
    result["submit_state"] = {"set": set_state, "click": click_state, "upload": upload_state}
    return result


def load_ai_mode_answer(
    request: Dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    wait_after_load_ms: Optional[int] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Open google.com in AI Mode, run ``request`` on it and return the answer's page snapshot."""
    if browser_driver.use_driver("GEMINI_FREE_MODE"):
        # Glossarion Mobile (no helper processes) or GEMINI_FREE_MODE=driver: the same flow
        # on a page of the registered browser driver (hidden WebView).
        return _load_ai_mode_answer_driver(
            request,
            timeout=timeout,
            wait_after_load_ms=wait_after_load_ms,
            user_agent=user_agent,
        )
    _setup_qt_environment()

    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["gemini-free-ai-mode"])

    profile, profile_root = _create_profile(app, user_agent=user_agent)
    page = QWebEnginePage(profile, app)
    _set_default_page_viewport(page)
    base_url = _build_search_base_url()
    timeout_seconds = _timeout_seconds(timeout)

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

    def wait_ms(ms: int) -> None:
        wait_loop = QEventLoop()
        QTimer.singleShot(max(1, int(ms)), wait_loop.quit)
        wait_loop.exec()

    try:
        load_loop = QEventLoop()
        load_state: Dict[str, Any] = {"ok": False}

        def _loaded(ok: bool) -> None:
            load_state["ok"] = bool(ok)
            load_loop.quit()

        page.loadFinished.connect(_loaded)
        page.load(QUrl(base_url))
        if timeout_seconds is not None:
            QTimer.singleShot(max(1000, int(timeout_seconds * 1000)), load_loop.quit)
        load_loop.exec()
        if not load_state.get("ok"):
            raise RuntimeError(f"Qt WebEngine failed to load {base_url}")

        return _submit_ai_mode_request(
            request,
            run_js=run_js,
            wait_ms=wait_ms,
            base_url=base_url,
            timeout=timeout,
            wait_after_load_ms=wait_after_load_ms,
        )
    finally:
        try:
            page.deleteLater()
            profile.deleteLater()
            cleanup_loop = QEventLoop()
            QTimer.singleShot(100, cleanup_loop.quit)
            cleanup_loop.exec()
        except Exception:
            pass
        _cleanup_profile_root(profile_root)


def _load_ai_mode_answer_driver(
    request: Dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    wait_after_load_ms: Optional[int] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """``load_ai_mode_answer`` on a page of the registered browser driver.

    Glossarion Mobile's hidden WebView loads AI Mode and runs the QtWebEngine helper's flow
    (``_submit_ai_mode_request``) in this worker thread. Like the helper it waits for the first
    finished load (or the request timeout); a Stop (``cancel_stream``) ends the page's calls.
    """
    driver = browser_driver.require_driver("Gemini Free")
    base_url = _build_search_base_url()
    timeout_seconds = _timeout_seconds(timeout)
    page = driver.open_page(
        owner=BROWSER_OWNER,
        user_agent=user_agent or DEFAULT_USER_AGENT,
        viewport=_default_viewport_size(),
        cancel_check=_is_cancelled,
    )
    try:
        page.load(base_url)
        if not browser_driver.wait_until_loaded(page, timeout_seconds):
            raise RuntimeError(f"The in-app browser failed to load {base_url}")
        return _submit_ai_mode_request(
            request,
            run_js=page.run_js,
            wait_ms=page.wait,
            base_url=base_url,
            timeout=timeout,
            wait_after_load_ms=wait_after_load_ms,
        )
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Signed in: a dedicated, persistent Chrome/Edge profile (optional).
#
# Signed out, AI Mode has one small model. Signed in to a Google account it offers the larger
# ones. Google refuses its sign-in page to embedded browsers (QtWebEngine), so the account lives
# in a Chrome/Edge profile of Glossarion's own, the way ``authds/`` keeps its DeepSeek session:
# the Google Login button opens that profile's window on Google's sign-in page, the user signs
# in there, and requests then run in headless tabs of the same profile. Glossarion never sees
# the password. Desktop only.
# ---------------------------------------------------------------------------

GOOGLE_SIGN_IN_URL = "https://accounts.google.com/ServiceLogin"
#: Cookies a signed-in google.com session has.
_GOOGLE_SESSION_COOKIES = ("SID", "__Secure-1PSID", "__Secure-3PSID")
_login_gate = threading.Lock()


def profile_dir() -> Path:
    """The Chrome/Edge profile that holds the Google sign-in (``GEMINI_FREE_PROFILE_DIR``)."""
    return Path(os.environ.get("GEMINI_FREE_PROFILE_DIR") or (Path.home() / ".glossarion" / "search_gemini_google"))


def has_session() -> bool:
    """Whether a Google sign-in was saved. A hint: every request checks the cookies itself."""
    return (profile_dir() / ".signed_in").exists()


def _mark_session(present: bool) -> None:
    marker = profile_dir() / ".signed_in"
    if present:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(mode=0o600)
    else:
        marker.unlink(missing_ok=True)


def _check(cancel_check: Optional[Callable[[], bool]] = None) -> None:
    if _is_cancelled() or (cancel_check and cancel_check()):
        raise AIModeError("stream cancelled", KIND_CANCELLED)


def _login_check(cancel_check: Optional[Callable[[], bool]] = None) -> None:
    """The cancel check of the sign-in window: only its own ``cancel_check`` ends it (a Stop of
    an earlier translation must not close a sign-in the user just opened)."""
    if cancel_check and cancel_check():
        raise AIModeError("Google sign-in cancelled.", KIND_CANCELLED)


class _GooglePage(cdp_browser.CdpPage):
    command_failed = "Gemini Free browser command failed."
    command_timed_out = "Gemini Free browser command timed out."
    script_failed = "Gemini Free browser script failed."

    def check(self, cancel_check):
        _check(cancel_check)

    def error(self, message, error_type):
        return AIModeError(message, error_type)


class _GoogleLoginPage(_GooglePage):
    def check(self, cancel_check):
        _login_check(cancel_check)


def _browser_binary() -> str:
    found = cdp_browser.find_browser_binary(
        "GEMINI_FREE_BROWSER_BINARY", lambda message: AIModeError(message, "config_error"))
    if found:
        return found
    raise AIModeError(
        "The Google sign-in for search/gemini needs Google Chrome or Microsoft Edge installed. "
        "You can also set GEMINI_FREE_BROWSER_BINARY.",
        "config_error",
    )


@contextmanager
def _launch_google_browser(*, visible: bool = False, cancel_check=None):
    # Keep the process-spawn gate in the same function for the mobile collector.
    if not mobile_runtime.subprocesses_available():
        raise AIModeError("The Google sign-in for search/gemini needs the desktop app and Chrome/Edge.", "config_error")
    width, height = _default_viewport_size()
    with cdp_browser.launch(
            binary=_browser_binary(), profile=profile_dir(), start_url="about:blank",
            ready=lambda page, check: None,
            page_factory=lambda url, check: (_GoogleLoginPage if visible else _GooglePage)(url, check),
            check=_login_check if visible else _check, error=AIModeError,
            closed_message="The Gemini Free browser closed. Close any other Google Login window of Glossarion and try again.",
            connect_message="Gemini Free could not connect to its browser.", visible=visible,
            # Remote debugging sets navigator.webdriver; Google treats such a browser as automated.
            extra_args=("--disable-blink-features=AutomationControlled", f"--window-size={width},{height}"),
            cancel_check=cancel_check) as page:
        yield page


def _prepare_google_tab(page) -> None:
    """Before a tab's first navigation: a headless browser names itself HeadlessChrome."""
    agent = str(page.call("Browser.getVersion").get("userAgent") or "")
    if "HeadlessChrome" in agent:
        page.call("Emulation.setUserAgentOverride", {"userAgent": agent.replace("HeadlessChrome", "Chrome")})


_TAB_READY = "document.readyState === 'complete' && /(^|\\.)google\\.[a-z.]+$/.test(location.hostname)"


def _google_tab_ready(page, cancel_check) -> None:
    cdp_browser.wait_until(
        page, _TAB_READY, timeout=40, check=_check, error=AIModeError,
        message="Google AI Mode did not finish loading. Check your connection and try again.",
        cancel_check=cancel_check, tolerate=("browser_error", "timeout", "api_error"))


_LEASES = cdp_browser.BrowserLeases(
    launch=lambda **kwargs: _launch_google_browser(**kwargs),
    page_factory=lambda url, check: _GooglePage(url, check),
    check=_check, url=SEARCH_BASE_URL, ready=_google_tab_ready, prepare=_prepare_google_tab)


def _google_signed_in(page) -> bool:
    """Whether the profile's google.com cookies hold a signed-in session."""
    cookies = page.call("Network.getCookies", {"urls": ["https://www.google.com/"]}).get("cookies") or []
    return any(str(cookie.get("name") or "") in _GOOGLE_SESSION_COOKIES and cookie.get("value") for cookie in cookies)


def login(log_fn: Callable[[str], None] = print, cancel_check=None, timeout: int = 300) -> None:
    """Open the profile's browser window on Google's sign-in page and wait until the user
    signed in there (``timeout`` seconds). The session stays in the profile."""
    target = f"{GOOGLE_SIGN_IN_URL}?{urlencode({'hl': 'en', 'continue': _build_search_base_url()})}"
    with _login_gate:
        with _LEASES.open_interactive(cancel_check) as page:
            page.call("Page.navigate", {"url": target})
            log_fn("Gemini Free: Sign in on the Google window that opened. "
                   "Glossarion does not receive your Google password.")
            deadline = time.monotonic() + timeout
            while True:
                _login_check(cancel_check)
                try:
                    if _google_signed_in(page):
                        break
                except AIModeError as exc:
                    if exc.kind == KIND_CANCELLED:
                        raise
                except Exception as exc:
                    raise AIModeError("The Google sign-in window was closed before the sign-in finished.",
                                      KIND_SIGN_IN) from exc
                if time.monotonic() >= deadline:
                    raise AIModeError("Google sign-in timed out. Select Google Login and try again.", KIND_SIGN_IN)
                time.sleep(.5)
            _mark_session(True)
            time.sleep(1.5)  # let the browser write the session cookies before it closes
    log_fn("Gemini Free: Google session saved.")


def logout(cancel_check=None) -> None:
    """Forget the saved Google session: the marker, and the profile's cookies."""
    _mark_session(False)
    with _login_gate:
        with _LEASES.open(cancel_check=cancel_check, prepare=lambda page: None) as page:
            page.call("Network.clearBrowserCookies")


def _load_ai_mode_answer_chrome(
    request: Dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    wait_after_load_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """``load_ai_mode_answer`` in a headless tab of the signed-in Chrome/Edge profile."""
    base_url = _build_search_base_url()

    def prepare(page) -> None:
        _prepare_google_tab(page)
        if not _google_signed_in(page):
            _mark_session(False)
            raise AIModeError(
                "The saved Google session ended. Use the Google Login button next to the model to sign in "
                "again (search/gemini then runs signed in), or send the request again to run signed out.",
                KIND_SIGN_IN,
            )

    _LEASES.url = base_url
    with _LEASES.open(cancel_check=_is_cancelled, prepare=prepare) as page:
        def run_js(script: str, js_timeout_ms: int = 15000) -> Any:
            try:
                return page.evaluate(script)
            except AIModeError as exc:
                if exc.kind == KIND_CANCELLED:
                    raise RuntimeError("stream cancelled") from exc
                return None  # a navigation replaced the page's script context: asked again

        def wait_ms(ms: int) -> None:
            end = time.monotonic() + max(1, int(ms)) / 1000.0
            while True:
                _raise_if_cancelled()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(.1, remaining))

        result = _submit_ai_mode_request(
            request,
            run_js=run_js,
            wait_ms=wait_ms,
            base_url=base_url,
            timeout=timeout,
            wait_after_load_ms=wait_after_load_ms,
        )
        result["signed_in"] = True
        return result


def list_models(timeout: int = 90, log_fn: Optional[Callable[[str], None]] = None, cancel_check=None) -> List[str]:
    """The ``search/gemini`` model names this browser's AI Mode offers (its model menu).

    Read on the page a request would use: the signed-in Chrome/Edge profile when the user used
    Google Login (an account's models), else the signed-out helper (one model), or the browser
    driver's page on Glossarion Mobile. ``search/gemini`` (Auto) is always first.
    """
    del cancel_check
    request = {"list_models": True}
    timeout_value = int(timeout or DEFAULT_TIMEOUT)
    _log(log_fn, "Gemini Free: reading google.com AI Mode's models")
    if browser_driver.use_driver("GEMINI_FREE_MODE"):
        data = load_ai_mode_answer(request, timeout=timeout_value)
    elif has_session():
        try:
            data = _load_ai_mode_answer_chrome(request, timeout=timeout_value)
        except AIModeError as exc:
            if exc.kind == KIND_CANCELLED:
                raise RuntimeError("stream cancelled") from exc
            raise
    else:
        data = _run_search_subprocess_once(
            messages=[], model=DEFAULT_MODEL, timeout=timeout_value, max_tokens=None, list_models=True)
    problem = _page_problem(data)
    if problem:
        raise AIModeError(problem[1], problem[0])
    return _model_ids(data.get("models") or [])


def _complete_ai_mode_request(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str,
    timeout: int,
    log_fn: Optional[Callable[[str], None]],
    loader: Callable[..., Dict[str, Any]],
    loader_kwargs: Optional[Dict[str, Any]] = None,
    signed_in: bool = False,
) -> Dict[str, Any]:
    """One chat request on an AI Mode page opened by ``loader``; the chat-completion result."""
    request = _build_ai_mode_request(messages, model)
    prompt = str(request.get("prompt") or "")
    files = list(request.get("files") or [])
    if not prompt and not files:
        raise RuntimeError("Gemini Free request has an empty prompt")
    prompt_url_chars = len(_build_search_url(prompt))
    attached = ", ".join(f"{item['name']} ({len(item['data']):,} bytes)" for item in files)
    _log(
        log_fn,
        "Gemini Free: opening google.com AI Mode"
        + (" (signed in)" if signed_in else "")
        + (f", model {request['model']}" if request.get("model") else "")
        + (f"; attaching {attached}" if attached else ""),
    )
    _log(log_fn, f"Gemini Free debug URL: {_build_search_base_url()[:1000]}", debug_only=True)
    page_data = loader(request, timeout=timeout, **(loader_kwargs or {}))
    content = _extract_rendered_content(
        page_data,
        prefer_html=bool(request.get("expects_html")),
        prefer_markdown=bool(request.get("expects_markdown")),
        prompt_url_chars=prompt_url_chars,
    )
    return {
        "content": content,
        "finish_reason": "stop",
        "usage": {
            "prompt_chars": int(request.get("prompt_chars") or 0),
            "completion_chars": len(content),
        },
        "raw_response": {
            "model": _strip_search_prefix(model),
            "ai_mode_model": page_data.get("model_label") or "",
            "signed_in": bool(signed_in),
            "search_url": _safe_url_for_log(_build_search_base_url()),
            "page_url": _safe_url_for_log(page_data.get("url")),
            "title": page_data.get("title"),
            "ready": page_data.get("ready"),
            "html_length": page_data.get("htmlLength"),
            "attachments": [
                {"name": item["name"], "kind": item["kind"], "bytes": len(item["data"])} for item in files
            ],
            "prompt_box_chars": len(prompt),
            "prompt_url_chars": prompt_url_chars,
            "instructions_attached": bool(request.get("instructions_attached")),
            "code_blocks": len(page_data.get("codeBlocks") or []),
            "prefer_html_response": bool(request.get("expects_html")),
            "prefer_markdown_response": bool(request.get("expects_markdown")),
        },
    }


def _send_chat_completion_qt(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """The request on a signed-out page: the helper's QtWebEngine page, or a browser-driver
    page (``load_ai_mode_answer`` picks)."""
    del max_tokens
    return _complete_ai_mode_request(
        messages=messages, model=model, timeout=timeout, log_fn=log_fn,
        loader=load_ai_mode_answer, loader_kwargs={"user_agent": user_agent},
    )


def _send_chat_completion_chrome(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """The request in the signed-in Chrome/Edge profile."""
    try:
        return _complete_ai_mode_request(
            messages=messages, model=model, timeout=timeout, log_fn=log_fn,
            loader=_load_ai_mode_answer_chrome, signed_in=True,
        )
    except AIModeError as exc:
        if exc.kind == KIND_CANCELLED:
            raise RuntimeError("stream cancelled") from exc
        raise


def _is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def _extract_json_from_process(stdout: str) -> Dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Gemini Free helper did not return JSON")


def _run_search_driver_once(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str,
    timeout: int,
    max_tokens: Optional[int],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """One helper request, in this worker thread, on a browser-driver page.

    The desktop helper process (``--search-helper``) runs ``_send_chat_completion_qt`` with
    the request's messages, model, max tokens and helper timeout on its QtWebEngine page;
    without helper processes (Glossarion Mobile) the same call runs here and
    ``load_ai_mode_answer`` opens a page of the registered browser driver. The helper's
    error text is kept; a Stop ends it with "stream cancelled".
    """
    browser_driver.require_driver("Gemini Free")
    message_list = list(messages or [])
    helper_timeout = _timeout_seconds(timeout)
    helper_timeout_arg = 0 if helper_timeout is None else max(30, int(helper_timeout))
    wait_timer: Optional[threading.Timer] = None
    if log_fn:
        def _log_still_waiting() -> None:
            if not _is_cancelled():
                _log(log_fn, "⏳ Gemini Free: still waiting for the in-app browser...")

        wait_timer = threading.Timer(10, _log_still_waiting)
        wait_timer.daemon = True
        wait_timer.start()
    try:
        return _send_chat_completion_qt(
            messages=message_list,
            model=model,
            timeout=helper_timeout_arg,
            max_tokens=max_tokens,
            log_fn=log_fn,
        )
    except browser_driver.BrowserUnavailable:
        raise
    except Exception as exc:
        if _is_cancelled():
            raise RuntimeError("stream cancelled") from exc
        raise AIModeError(
            f"Gemini Free browser helper failed: {_short_error(exc)}", getattr(exc, "kind", "")
        ) from exc
    finally:
        if wait_timer is not None:
            wait_timer.cancel()


def _run_search_subprocess_once(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str,
    timeout: int,
    max_tokens: Optional[int],
    log_fn: Optional[Callable[[str], None]] = None,
    list_models: bool = False,
) -> Dict[str, Any]:
    if not mobile_runtime.subprocesses_available() or browser_driver.use_driver("GEMINI_FREE_MODE"):
        # The helper boundary. Glossarion Mobile has no helper processes (GEMINI_FREE_MODE=driver
        # asks for the same): the helper's request runs in this worker thread on a page of the
        # registered browser driver instead.
        return _run_search_driver_once(
            messages=messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            log_fn=log_fn,
        )
    message_list = list(messages or [])
    helper_timeout = _timeout_seconds(timeout)
    helper_timeout_arg = 0 if helper_timeout is None else max(30, int(helper_timeout))
    temp_dir = tempfile.mkdtemp(prefix="glossarion_gemini_free_")
    prompt_path = os.path.join(temp_dir, "messages.json")
    try:
        with open(prompt_path, "w", encoding="utf-8") as handle:
            json.dump({"messages": message_list}, handle, ensure_ascii=False)

        if _is_frozen_app():
            cmd = [
                sys.executable,
                "--gemini-free-search",
                "--messages",
                prompt_path,
                "--model",
                model,
                "--timeout",
                str(helper_timeout_arg),
            ]
        else:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--search-helper",
                "--messages",
                prompt_path,
                "--model",
                model,
                "--timeout",
                str(helper_timeout_arg),
            ]
        if max_tokens:
            cmd.extend(["--max-tokens", str(int(max_tokens))])
        if list_models:
            cmd.append("--list-models")  # the helper answers with AI Mode's model menu instead

        env = os.environ.copy()
        env["GEMINI_FREE_HELPER"] = "1"
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = _qtwebengine_chromium_flags(env.get("QTWEBENGINE_CHROMIUM_FLAGS", ""))
        env.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

        try:
            from shutdown_utils import subprocess_no_window_kwargs
            no_window_kwargs = subprocess_no_window_kwargs()
        except Exception:
            no_window_kwargs = {}

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            **no_window_kwargs,
        )
        with _active_helper_lock:
            _active_helper_processes.add(proc)
        deadline = None if helper_timeout_arg <= 0 else time.time() + helper_timeout_arg + 20
        stdout = ""
        stderr = ""
        result_queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)

        def _drain_output() -> None:
            try:
                out, err = proc.communicate()
            except Exception as exc:
                out, err = "", str(exc)
            try:
                result_queue.put_nowait((out or "", err or ""))
            except Exception:
                pass

        output_thread = threading.Thread(
            target=_drain_output,
            name="GeminiFreeHelperOutput",
            daemon=True,
        )
        output_thread.start()
        try:
            next_wait_log = time.time() + 10
            wait_log_printed = False
            while True:
                try:
                    stdout, stderr = result_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    pass
                if _is_cancelled():
                    _terminate_process_tree(proc, kill=True)
                    raise RuntimeError("stream cancelled")
                if _deadline_expired(deadline):
                    _terminate_process_tree(proc, kill=True)
                    raise RuntimeError(f"Gemini Free helper timed out after {helper_timeout_arg}s")
                if log_fn and not wait_log_printed and time.time() >= next_wait_log:
                    _log(log_fn, "⏳ Gemini Free: still waiting for Qt WebEngine helper...")
                    wait_log_printed = True
        finally:
            with _active_helper_lock:
                _active_helper_processes.discard(proc)

        result = _extract_json_from_process(stdout)
        error_kind = str(result.get("error_kind") or "")
        if proc.returncode != 0:
            error = str(result.get("error") or "").strip()
            if error:
                raise AIModeError(f"Gemini Free helper failed ({proc.returncode}): {error}", error_kind)
            detail = (stderr or stdout or "").strip()
            raise RuntimeError(f"Gemini Free helper failed ({proc.returncode}): {detail[-1200:]}")
        if result.get("error"):
            raise AIModeError(str(result.get("error")), error_kind)
        return result
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def send_chat_completion(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Send a chat-style prompt to google.com AI Mode and return its answer.

    One request is one AI Mode question on its own page: the system prompt is typed into the
    prompt box, the text to work on and any images are attached as files. Where the answer
    comes from: a browser-driver page (Glossarion Mobile's hidden WebView), a headless tab of
    the signed-in Chrome/Edge profile when the user used Google Login, else the signed-out
    QtWebEngine helper (a subprocess; ``GEMINI_FREE_MODE=inline`` runs it in this process).
    """
    del temperature
    timeout_value = int(timeout or _env_int("GEMINI_FREE_TIMEOUT", DEFAULT_TIMEOUT))
    _ai_mode_model_choice(model)  # an unknown search/ model fails before any browser starts
    if browser_driver.use_driver("GEMINI_FREE_MODE"):
        # Glossarion Mobile (no helper processes) or GEMINI_FREE_MODE=driver: the helper's
        # request on a page of the registered browser driver (hidden WebView).
        _log(log_fn, "🌐 Gemini Free: opening the in-app browser (hidden WebView)")
        return _run_search_subprocess_once(
            messages=messages,
            model=model,
            timeout=timeout_value,
            max_tokens=max_tokens,
            log_fn=log_fn,
        )
    mode = os.getenv("GEMINI_FREE_MODE", "subprocess").strip().lower()
    if mode == "inline":
        return _send_chat_completion_qt(
            messages=messages,
            model=model,
            timeout=timeout_value,
            max_tokens=max_tokens,
            log_fn=log_fn,
        )
    if has_session():
        _log(log_fn, "🌐 Gemini Free: using the signed-in Google profile (Chrome/Edge)")
        return _send_chat_completion_chrome(
            messages=messages,
            model=model,
            timeout=timeout_value,
            log_fn=log_fn,
        )
    _log(log_fn, "🌐 Gemini Free: starting Qt WebEngine helper subprocess (signed out)")
    return _run_search_subprocess_once(
        messages=messages,
        model=model,
        timeout=timeout_value,
        max_tokens=max_tokens,
        log_fn=log_fn,
    )


def _load_cli_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.messages:
        with open(args.messages, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            data = data["messages"]
        if not isinstance(data, list):
            raise ValueError("--messages must contain a JSON list or an object with a messages list")
        return data
    prompt = ""
    if args.prompt_file:
        prompt = _read_text(args.prompt_file)
    elif args.prompt == "-":
        prompt = sys.stdin.read()
    elif args.prompt:
        prompt = args.prompt
    return [{"role": "user", "content": prompt}]


def _print_summary(result: Dict[str, Any]) -> None:
    if "content" in result:
        print(result.get("content") or "")
        return
    body = str(result.get("body") or "")
    print(f"{result.get('status')} {result.get('statusText')}")
    print(f"url: {result.get('url')}")
    print(f"ok: {result.get('ok')} redirected: {result.get('redirected')} type: {result.get('type')}")
    print(f"bodyLength: {result.get('bodyLength')} truncated: {result.get('truncated')}")
    if result.get("error"):
        print(f"error: {result.get('error')}")
    if body:
        print("\nbody:")
        print(body)


def _main() -> int:
    for stream_name in ("stdout", "stderr"):
        stream_obj = getattr(sys, stream_name, None)
        if hasattr(stream_obj, "reconfigure"):
            try:
                stream_obj.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Ask google.com AI Mode from a Qt WebEngine page (the search/gemini route).",
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Raw request URL for fetch mode.")
    parser.add_argument("--search-helper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list-models", action="store_true", help="Print AI Mode's model menu as JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", help="Prompt text. Use '-' to read stdin.")
    parser.add_argument("--prompt-file", help="UTF-8 file containing prompt text.")
    parser.add_argument("--messages", help="JSON file containing messages list or {'messages': [...]}.")
    parser.add_argument("--method", default="GET", help="Raw fetch HTTP method. Default: GET.")
    parser.add_argument("--header", action="append", help="Raw fetch request header in 'Name: value' form.")
    parser.add_argument("--data", help="Raw fetch request body text.")
    parser.add_argument("--data-file", help="Read raw fetch request body text from a UTF-8 file.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds.")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", help="Write response body/content to a UTF-8 file.")
    parser.add_argument("--json", action="store_true", help="Print the full result object as JSON.")
    args = parser.parse_args()

    try:
        if args.list_models:
            page_data = load_ai_mode_answer({"list_models": True}, timeout=args.timeout)
            problem = _page_problem(page_data)
            if problem:
                raise AIModeError(problem[1], problem[0])
            result = {"content": "", "models": page_data.get("models") or [], "url": page_data.get("url")}
            args.json = True
        elif args.search_helper or args.prompt or args.prompt_file or args.messages:
            result = _send_chat_completion_qt(
                messages=_load_cli_messages(args),
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
        else:
            body = _read_text(args.data_file) if args.data_file else args.data
            result = run_qtwebengine_request(
                url=args.url,
                method=args.method,
                headers=_parse_headers(args.header),
                body=body,
                bootstrap_url=_origin_url(args.url),
                timeout=args.timeout,
            )
        if args.output:
            _write_text(args.output, str(result.get("content") or result.get("body") or ""))
        if args.json or args.search_helper:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            _print_summary(result)
        return 0 if not result.get("error") else 1
    except Exception as exc:
        error = {
            "content": "",
            "finish_reason": "error",
            "error": _short_error(exc),
            "error_kind": str(getattr(exc, "kind", "") or ""),
        }
        if args.json or args.search_helper:
            print(json.dumps(error, ensure_ascii=False, separators=(",", ":")))
        else:
            print(f"Gemini Free request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
