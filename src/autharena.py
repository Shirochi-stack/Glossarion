"""Arena's browser-backed, text-only direct route.

Each completion creates a new evaluation. Only browser authentication persists;
conversation identifiers and prompts are never reused. The website controls
sampling and output limits. Public model polling uses HTTP and never launches UI.

Qt WebEngine runs in a child process because callers commonly run in GUI workers.
Cookies and reCAPTCHA tokens remain inside that browser's same-origin context.
"""

from __future__ import annotations

import argparse
import codecs
from collections import deque
from contextlib import contextmanager
import html
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote

import requests


ARENA_BASE_URL = "https://arena.ai"
CATALOG_URL = ARENA_BASE_URL + "/text/direct"
CREATE_EVALUATION_PATH = "/nextjs-api/stream/create-evaluation"
DEFAULT_MODEL = "gpt-6-astra-medium"
DEFAULT_TIMEOUT = 180
# Public widget configuration, not authentication credentials. The v3 key is
# discovered from the loaded page; this v2 key is the site's interactive widget.
RECAPTCHA_V2_SITEKEY = "6Le3_cYsAAAAAGwWOK2RLDgNI15Bh8C0yLBOL1yL"
_cancel_event = threading.Event()
_active_lock = threading.Lock()
_active_helpers = set()
_profile_locks = {}
_profile_locks_guard = threading.Lock()
_warned_options = set()
_warn_lock = threading.Lock()


class AuthArenaError(RuntimeError):
    def __init__(self, message, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _env_bool(name, default=True):
    return os.getenv(name, str(default)).strip().lower() not in ("0", "false", "no", "off", "")


def _cancelled(cancel_check=None):
    # UnifiedClient's callback distinguishes graceful stop before dispatch from
    # a hard/local stop during generation. Its environment flag alone cannot.
    return (_cancel_event.is_set() or (bool(cancel_check()) if cancel_check is not None
                                      else os.getenv("TRANSLATION_CANCELLED") == "1"))


def _check_wait(deadline, cancel_check=None):
    if _cancelled(cancel_check):
        raise RuntimeError("AuthArena stream cancelled")
    if time.monotonic() >= deadline:
        raise TimeoutError("AuthArena request timed out (including browser/login and profile queue time)")


def reset_cancel():
    if os.getenv("TRANSLATION_CANCELLED") != "1":
        _cancel_event.clear()


def _terminate_helper(proc):
    if proc.poll() is not None:
        return
    # QtWebEngineProcess children otherwise retain the profile and frozen DLLs.
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=5)
        except (OSError, subprocess.SubprocessError):
            proc.kill()
    else:
        try:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def cancel_stream():
    _cancel_event.set()
    with _active_lock:
        processes = list(_active_helpers)
    for proc in processes:
        try:
            _terminate_helper(proc)
        except (OSError, subprocess.SubprocessError):
            pass


def _positive_timeout(value):
    value = DEFAULT_TIMEOUT if value is None else float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("AuthArena timeout must be a positive finite number")
    return value


def _account_number(account_id):
    if isinstance(account_id, bool) or not re.fullmatch(r"\d+", str(account_id)):
        raise ValueError("AuthArena account_id must be a nonnegative integer")
    return int(account_id)


def _profile_path(account_id):
    return Path.home() / ".glossarion" / "autharena_browser" / str(_account_number(account_id))


@contextmanager
def _profile_gate(account_id, deadline, cancel_check=None, log_fn=None):
    """Serialize each persistent Chromium profile across threads and processes."""
    path = _profile_path(account_id)
    with _profile_locks_guard:
        lock = _profile_locks.setdefault(str(path), threading.Lock())
    _check_wait(deadline, cancel_check)
    notified = False
    while not lock.acquire(timeout=0.1):
        _check_wait(deadline, cancel_check)
        if log_fn and not notified:
            log_fn("AuthArena: waiting for this account's browser profile")
            notified = True
    handle = None
    acquired = False
    try:
        path.mkdir(parents=True, exist_ok=True)
        handle = (path / ".request.lock").open("a+b")
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        while not acquired:
            _check_wait(deadline, cancel_check)
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, PermissionError, OSError) as exc:
                if getattr(exc, "errno", None) not in (None, 11, 13, 35, 36):
                    raise
                if log_fn and not notified:
                    log_fn("AuthArena: waiting for this account's browser profile")
                    notified = True
                time.sleep(0.1)
        yield path
    finally:
        if handle is not None:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        lock.release()


def _flight_text(page):
    """Decode Next.js flight string chunks without evaluating JavaScript."""
    chunks = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?:self\.)?__next_f\.push\(\s*", page):
        try:
            packet, _ = decoder.raw_decode(page[match.end():])
        except (ValueError, TypeError):
            continue
        if isinstance(packet, list) and len(packet) > 1 and isinstance(packet[1], str):
            chunks.append(packet[1])
    return "".join(chunks) if chunks else page


def _parse_model_catalog(page):
    data = _flight_text(page)
    decoder = json.JSONDecoder()
    found = False
    models = {}
    for match in re.finditer(r'"initialModels"\s*:\s*', data):
        try:
            entries, _ = decoder.raw_decode(data[match.end():])
        except ValueError:
            continue
        if not isinstance(entries, list):
            continue
        found = True
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("userSelectable") is not True:
                continue
            # Historical entries can retain userSelectable without a provider.
            # The website's direct picker excludes these unavailable models.
            if entry.get("provider") is None or entry.get("organization") is None:
                continue
            capabilities = entry.get("capabilities") or {}
            if not isinstance(capabilities, dict):
                continue
            inputs = capabilities.get("inputCapabilities") or {}
            outputs = capabilities.get("outputCapabilities") or {}
            if not isinstance(inputs, dict) or not isinstance(outputs, dict):
                continue
            if inputs.get("text") is not True or outputs.get("text") is not True:
                continue
            name = entry.get("displayName") or entry.get("publicName") or entry.get("name")
            if str(entry.get("publicName") or "").lower() == "max":
                name = "max"
            identifier = entry.get("id")
            if not isinstance(name, str) or not name.strip() or not isinstance(identifier, str):
                continue
            try:
                uuid.UUID(identifier)
            except ValueError:
                continue
            models.setdefault(name, identifier)  # Same first eligible match as the website.
    if not found or not models:
        raise RuntimeError("AuthArena: selectable text model catalog was absent or empty; Arena may have blocked HTTP access or changed its page format")
    return models


def fetch_available_models(timeout=15, account_id=0):
    """Return canonical names from public HTML. No cookies or browser are used."""
    _account_number(account_id)
    response = requests.get(CATALOG_URL, timeout=_positive_timeout(timeout), headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    })
    try:
        if response.status_code != 200:
            raise RuntimeError(f"AuthArena model polling HTTP {response.status_code}; browser authentication is not opened by automatic polling")
        return sorted(_parse_model_catalog(response.text), key=str.casefold)
    finally:
        response.close()


def _normalize_model(model):
    value = str(model or "").strip()
    prefix = re.match(r"(?i)^autharena(\d*)/", value)
    if prefix:
        value = value[prefix.end():]
    if not value or any(ch in value for ch in "\r\n\0"):
        raise ValueError("AuthArena requires a canonical model name")
    return value


def _model_account(model, account_id):
    prefix = re.match(r"(?i)^autharena(\d+)/", str(model or "").strip())
    account_id = _account_number(account_id)
    if prefix:
        numbered = int(prefix.group(1))
        if account_id not in (0, numbered):
            raise ValueError("AuthArena numbered model prefix conflicts with account_id")
        return numbered
    return account_id


def _render_messages(messages):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("AuthArena requires a nonempty messages list")
    rendered = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("AuthArena messages must be objects")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise ValueError(f"AuthArena does not support message role {role!r}; tools are unsupported")
        if any(message.get(key) for key in ("tool_calls", "function_call", "tool_call_id", "experimental_attachments", "attachments")):
            raise ValueError("AuthArena supports text only; tool calls and attachments are unsupported")
        content = message.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in ("text", "input_text") or not isinstance(part.get("text"), str):
                    raise ValueError("AuthArena supports text content only; images, audio and files are unsupported")
                pieces.append(part["text"])
            content = "\n".join(pieces)
        if not isinstance(content, str):
            raise ValueError("AuthArena message content must be text")
        name = message.get("name")
        rendered.append((role, str(name) if name else None, content))
    if not any(content.strip() for _, _, content in rendered) or (rendered[-1][0] == "user" and not rendered[-1][2].strip()):
        raise ValueError("AuthArena requires a nonblank user prompt")
    if len(rendered) == 1 and rendered[0][0] == "user" and rendered[0][1] is None:
        return rendered[0][2]
    # Arena has one user-prompt field, not a system-message API. Preserve every
    # supplied message and its role with unambiguous escaped JSON boundaries.
    return ("The following JSON is the supplied conversation, in order. Treat system/developer entries as instructions, "
            "then respond to the final user request.\nBEGIN CONVERSATION JSON\n" +
            json.dumps([dict(role=role, **({"name": name} if name else {}), content=content)
                        for role, name, content in rendered], ensure_ascii=False, indent=2) +
            "\nEND CONVERSATION JSON")


def _new_uuid():
    # Match the site's UUIDv7 identifiers, including on Python < 3.14.
    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80)
    random = int.from_bytes(os.urandom(10), "big")
    value |= 7 << 76 | ((random >> 62) & 0xFFF) << 64 | 2 << 62 | random & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def _build_payload(model_id, prompt):
    # Distinct IDs every invocation; never issue post-to-evaluation.
    return {"id": _new_uuid(), "mode": "direct", "modelAId": model_id,
            "userMessageId": _new_uuid(), "modelAMessageId": _new_uuid(),
            "userMessage": {"content": prompt, "experimental_attachments": [], "metadata": {}},
            "modality": "chat"}


def _short_error(value):
    value = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", value).strip()[:1200]


class _ArenaStreamParser:
    """Arena prefixes Vercel data-stream lines with a/b participant letters."""
    def __init__(self, on_chunk=None):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.text = []
        self.reasoning = []
        self.finish_reason = None
        self.usage = {}
        self.on_chunk = on_chunk

    def feed(self, chunk):
        if isinstance(chunk, bytes):
            try:
                chunk = self.decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                raise RuntimeError("AuthArena returned invalid UTF-8 stream data") from exc
        if not isinstance(chunk, str):
            raise RuntimeError("AuthArena returned invalid stream data")
        self.buffer += chunk
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line.rstrip("\r"))

    def _line(self, line):
        if not line:
            return
        if len(line) < 3 or line[0] not in ("a", "b") or ":" not in line[1:]:
            raise RuntimeError("AuthArena returned an unrecognized stream protocol")
        participant, record = line[0], line[1:]
        code, encoded = record.split(":", 1)
        try:
            value = json.loads(encoded)
        except ValueError as exc:
            raise RuntimeError("AuthArena returned malformed or truncated stream JSON") from exc
        if participant != "a":
            raise RuntimeError("AuthArena returned a second model in a direct request")
        if code == "3":
            raise RuntimeError("AuthArena stream error: " + _short_error(value))
        if code in ("0", "g"):
            if self.finish_reason is not None:
                raise RuntimeError("AuthArena returned content after the terminal event")
            if not isinstance(value, str):
                raise RuntimeError("AuthArena returned nontext content in a text stream")
            (self.text if code == "0" else self.reasoning).append(value)
            if value and self.on_chunk:
                self.on_chunk("content" if code == "0" else "reasoning", value)
        elif code == "d":
            reason = value.get("finishReason") if isinstance(value, dict) else None
            if not isinstance(reason, str) or not reason:
                raise RuntimeError("AuthArena terminal event is missing finishReason")
            if reason == "error":
                raise RuntimeError(f"AuthArena generation failed: {reason}")
            self.finish_reason = reason
            usage = value.get("usage")
            if isinstance(usage, dict):
                for key, output in (("promptTokens", "prompt_tokens"), ("completionTokens", "completion_tokens")):
                    count = usage.get(key)
                    if isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(count) and count >= 0:
                        self.usage[output] = int(count)
                if len(self.usage) == 2:
                    self.usage["total_tokens"] = sum(self.usage.values())
        elif code == "2":
            if not isinstance(value, list):
                raise RuntimeError("AuthArena returned malformed data metadata")
            for part in value:
                if isinstance(part, dict) and part.get("type") in ("image", "video", "webdev"):
                    raise RuntimeError("AuthArena returned unsupported nontext output")
        elif code not in ("8", "9", "a", "b", "c", "e", "f", "h", "i", "j", "k"):
            raise RuntimeError(f"AuthArena returned unsupported stream record {code!r}")

    def finish(self):
        try:
            self.buffer += self.decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise RuntimeError("AuthArena stream ended with truncated UTF-8") from exc
        if self.buffer:
            self._line(self.buffer.rstrip("\r"))
            self.buffer = ""
        if self.finish_reason is None:
            raise RuntimeError("AuthArena stream ended without a terminal event; output may be truncated")
        return {"content": "".join(self.text), "reasoning_content": "".join(self.reasoning),
                "finish_reason": self.finish_reason, "finish_reason_explicit": True, "usage": self.usage}


def _helper_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--autharena-helper"]
    return [sys.executable, str(Path(__file__).resolve()), "--browser-helper"]


def _run_browser_helper(config, deadline, *, cancel_check=None, before_send_callback=None, after_rejection_callback=None,
                        log_fn=None, on_chunk=None):
    environment = os.environ.copy()
    environment["AUTHARENA_BROWSER_HELPER"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
    _check_wait(deadline, cancel_check)
    proc = subprocess.Popen(_helper_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=environment, **kwargs)
    events = queue.Queue()
    diagnostics = deque(maxlen=8)

    def read_stdout():
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # Libraries can print startup diagnostics.
                if isinstance(event, dict) and event.get("autharena") == 1:
                    events.put(event)
        finally:
            events.put({"event": "eof"})

    def read_stderr():
        for line in proc.stderr:
            diagnostics.append(_short_error(line))

    readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    with _active_lock:
        _active_helpers.add(proc)
    for reader in readers:
        reader.start()
    parser = _ArenaStreamParser(on_chunk)
    dispatched = False
    try:
        proc.stdin.write(json.dumps(config, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        while True:
            _check_wait(deadline, cancel_check)
            try:
                event = events.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            kind = event.get("event")
            if kind == "ready":
                _check_wait(deadline, cancel_check)
                if before_send_callback:
                    before_send_callback()
                _check_wait(deadline, cancel_check)
                proc.stdin.write('{"command":"dispatch"}\n')
                proc.stdin.flush()
                dispatched = True
            elif kind == "chunk":
                parser.feed(event.get("data"))
            elif kind == "rejected":
                dispatched = False
                if after_rejection_callback:
                    after_rejection_callback()
            elif kind == "done":
                result = {"profile_saved": True} if config.get("login") else parser.finish()
                return result
            elif kind == "status":
                if log_fn:
                    log_fn("AuthArena: " + _short_error(event.get("message")))
            elif kind == "error":
                if event.get("error_type") == "timeout":
                    raise TimeoutError("AuthArena: " + _short_error(event.get("message")))
                if event.get("error_type") == "configuration":
                    raise ImportError("AuthArena: " + _short_error(event.get("message")))
                raise AuthArenaError("AuthArena: " + _short_error(event.get("message")),
                                     event.get("status_code"), event.get("retry_after"))
            elif kind == "eof":
                _check_wait(deadline, cancel_check)
                suffix = " after dispatch; the request was not retried" if dispatched else " before dispatch"
                detail = "; ".join(diagnostics)[-1200:]
                raise RuntimeError("AuthArena browser helper exited" + suffix + (": " + detail if detail else ""))
    finally:
        # Give Qt a chance to flush its persistent cookie store on normal exit.
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _terminate_helper(proc)
        with _active_lock:
            _active_helpers.discard(proc)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass
        for reader in readers:
            reader.join(timeout=0.2)


def send_chat_completion(*, messages, model, temperature=None, max_tokens=None,
                         top_p=None, frequency_penalty=None, presence_penalty=None,
                         timeout=None, connect_timeout=None, account_id=0,
                         stream=None, log_stream=None, progress_label=None, log_fn=None,
                         cancel_check=None, before_send_callback=None, after_rejection_callback=None):
    prompt = _render_messages(messages)
    account_id = _model_account(model, account_id)
    model = _normalize_model(model)
    timeout = _positive_timeout(timeout)
    deadline = time.monotonic() + timeout
    _check_wait(deadline, cancel_check)
    ignored = [name for name, value in (("temperature", temperature), ("max_tokens", max_tokens),
               ("top_p", top_p), ("frequency_penalty", frequency_penalty),
               ("presence_penalty", presence_penalty), ("connect_timeout", connect_timeout)) if value is not None]
    if ignored and log_fn:
        with _warn_lock:
            new = [name for name in ignored if name not in _warned_options]
            _warned_options.update(ignored)
        if new:
            log_fn("AuthArena: Arena controls generation settings; unsupported options are ignored: " + ", ".join(new))
    should_log = (_env_bool("AUTHARENA_STREAM") if stream is None else bool(stream))
    should_log = should_log and (_env_bool("AUTHARENA_LOG_STREAM_CHUNKS", _env_bool("LOG_STREAM_CHUNKS")) if log_stream is None else bool(log_stream))
    thinking_log = _env_bool("AUTHARENA_STREAM_THINKING_LOGS", _env_bool("STREAM_THINKING_LOGS"))

    log_buffers = {"content": "", "reasoning": ""}
    started = time.monotonic()

    def on_chunk(kind, text):
        if should_log and log_fn and (kind != "reasoning" or thinking_log):
            log_buffers[kind] += text.replace("\x1f", "\\x1F")
            if "\n" in log_buffers[kind] or len(log_buffers[kind]) >= 160:
                text, log_buffers[kind] = log_buffers[kind], ""
                log_fn(("    " if kind == "reasoning" else "") + text)

    def before_send():
        if before_send_callback:
            before_send_callback()
        if progress_label and log_fn:
            log_fn(progress_label)

    with _profile_gate(account_id, deadline, cancel_check, log_fn) as profile:
        config = {"profile": str(profile), "account_id": account_id, "model": model,
                  "timeout": max(0.01, deadline - time.monotonic()),
                  "payload": _build_payload(None, prompt), "login": False}
        if log_fn:
            log_fn(f"AuthArena: preparing a new isolated request for {model}")
        result = _run_browser_helper(config, deadline, cancel_check=cancel_check,
                                     before_send_callback=before_send, after_rejection_callback=after_rejection_callback,
                                     log_fn=log_fn, on_chunk=on_chunk)
        if should_log and log_fn:
            for kind, text in log_buffers.items():
                if text:
                    log_fn(("    " if kind == "reasoning" else "") + text)
            log_fn(f"📡 AuthArena: Stream finished in {time.monotonic() - started:.1f}s")
        return result


def login(account_id=0, timeout=DEFAULT_TIMEOUT, log_fn=None):
    deadline = time.monotonic() + _positive_timeout(timeout)
    with _profile_gate(account_id, deadline, log_fn=log_fn) as profile:
        return _run_browser_helper({"profile": str(profile), "account_id": _account_number(account_id),
                                    "model": DEFAULT_MODEL, "login": True,
                                    "timeout": max(0.01, deadline - time.monotonic())},
                                   deadline, log_fn=log_fn)


# Only these scripts touch site credentials. Tokens stay in the page and are
# never included in helper output, Python HTTP requests, logs or files.
_PREPARE_JS = r"""
(() => {
  if (location.origin !== 'https://arena.ai') return;
  const state = window.__glossarionArena = {events: [], payload: __PAYLOAD__, phase: 'preparing', rejections: __REJECTIONS__};
  const emit = (event, details = {}) => state.events.push({event, ...details});
  const fail = (message, status_code = null, retry_after = null) => {
    state.phase = 'error'; emit('error', {message, status_code, retry_after});
  };
  const needUser = message => {state.phase = 'waiting'; emit('action', {message});};
  const deadline = Date.now() + __TIMEOUT_MS__;
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const readValue = (data, start) => {
    if (data[start] !== '[' && data[start] !== '{') return null;
    let depth = 0, quoted = false, escape = false;
    for (let i = start; i < data.length; i++) {
      const c = data[i];
      if (quoted) {
        if (escape) escape = false;
        else if (c === '\\') escape = true;
        else if (c === '"') quoted = false;
      } else if (c === '"') quoted = true;
      else if (c === '[' || c === '{') depth++;
      else if ((c === ']' || c === '}') && --depth === 0) return JSON.parse(data.slice(start, i + 1));
    }
    return null;
  };
  const readJSON = (data, name) => {
    const match = new RegExp('"' + name + '"\\s*:\\s*').exec(data);
    return match ? readValue(data, match.index + match[0].length) : null;
  };
  const readFlight = () => {
    // Next drains __next_f during hydration and replaces push with a consumer.
    // The original inline script text remains available after loadFinished.
    const chunks = [];
    for (const script of document.scripts) {
      const source = script.textContent || '';
      const pattern = /(?:self\.)?__next_f\.push\(\s*/g;
      let match;
      while ((match = pattern.exec(source))) {
        try {
          const packet = readValue(source, match.index + match[0].length);
          if (Array.isArray(packet) && typeof packet[1] === 'string') chunks.push(packet[1]);
        } catch (_) {}
      }
    }
    return chunks.length ? chunks.join('') : (window.__next_f || [])
      .filter(p => Array.isArray(p) && typeof p[1] === 'string').map(p => p[1]).join('');
  };
  const getToken = async action => {
    let key;
    while (Date.now() < deadline) {
      for (const script of document.scripts) {
        try {
          const url = new URL(script.src);
          if (/\/recaptcha\/enterprise\.js$/.test(url.pathname)) key = url.searchParams.get('render');
        } catch (_) {}
      }
      if (key && key !== 'explicit' && window.grecaptcha?.enterprise?.execute) break;
      await sleep(200);
    }
    if (!key || !window.grecaptcha?.enterprise?.execute) throw Error('Arena reCAPTCHA is not ready. Complete any browser security screen, then click Continue.');
    await new Promise(resolve => window.grecaptcha.enterprise.ready(resolve));
    const token = await window.grecaptcha.enterprise.execute(key, {action});
    if (!token) throw Error('Arena did not issue a reCAPTCHA token. Complete browser verification, then click Continue.');
    return token;
  };
  state.challenge = () => {
    state.phase = 'challenge';
    emit('action', {message: 'Complete the visible Arena reCAPTCHA challenge. The request will continue when it succeeds.', challenge: true});
    let container = document.getElementById('glossarion-arena-captcha');
    if (container) container.remove();
    container = document.createElement('div');
    container.id = 'glossarion-arena-captcha';
    container.style.cssText = 'position:fixed;z-index:2147483647;left:24px;top:100px;padding:24px;background:white;color:black;border:2px solid #444;border-radius:12px;box-shadow:0 4px 24px #0008';
    const heading = document.createElement('p'); heading.textContent = 'Complete Arena security verification to send this request.';
    const widget = document.createElement('div'); container.append(heading, widget); document.body.append(container);
    try {
      window.grecaptcha.enterprise.render(widget, {
        sitekey: __V2_SITEKEY__,
        callback: token => {
          delete state.payload.recaptchaV3Token;
          state.payload.recaptchaV2Token = token;
          container.remove(); state.phase = 'ready'; emit('ready');
        },
        'error-callback': () => needUser('Arena security verification failed. Click Continue to reload and try again.'),
        'expired-callback': () => needUser('Arena security verification expired. Click Continue to reload and try again.'),
        theme: 'light'
      });
    } catch (_) {needUser('Arena could not display security verification. Sign in on the page, then click Continue.');}
  };
  state.dispatch = async () => {
    if (state.phase !== 'ready') return;
    state.phase = 'dispatched';
    state.controller = new AbortController();
    let response;
    try {
      response = await fetch('/nextjs-api/stream/create-evaluation', {
        method: 'POST', credentials: 'same-origin', body: JSON.stringify(state.payload), signal: state.controller.signal
      });
      if (!response.ok) {
        let body = {};
        try {body = await response.json();} catch (_) {}
        const message = typeof body.error === 'string' ? body.error : (typeof body.message === 'string' ? body.message : 'Request rejected');
        const code = typeof body.code === 'string' ? body.code : '';
        if (response.status === 401 && code === 'LOGIN_GATE') {
          emit('rejected');
          needUser('Arena requires a signed-in account for this model. Sign in using the page, then click Continue.'); return;
        }
        if (state.rejections < 2 && /recaptcha|captcha|prompt failed/i.test(message) && [400, 401, 403, 429].includes(response.status)) {
          state.rejections++; emit('rejected'); state.challenge(); return;
        }
        fail('HTTP ' + response.status + ': ' + (code ? code + ': ' : '') + message, response.status, response.headers.get('Retry-After')); return;
      }
      const type = response.headers.get('Content-Type') || '';
      if (/text\/html|application\/json/.test(type)) {
        fail('Arena returned an unexpected response format after dispatch; the request was not retried.'); return;
      }
      if (!response.body) {fail('Arena returned no response body after dispatch; the request was not retried.'); return;}
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8', {fatal: true});
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const data = decoder.decode(value, {stream: true});
        if (data) emit('chunk', {data});
        // Backpressure prevents very fast streams from accumulating without a
        // bound while Qt or the parent is briefly busy.
        while (state.events.length > 128) await sleep(25);
      }
      const tail = decoder.decode();
      if (tail) emit('chunk', {data: tail});
      state.phase = 'done'; emit('done');
    } catch (_) {
      fail('Browser transport failed after dispatch; completion is uncertain and the request was not retried.');
    } finally {
      delete state.payload.recaptchaV3Token; delete state.payload.recaptchaV2Token;
    }
  };
  (async () => {
    try {
      let flight = '', models = null;
      for (let attempt = 0; attempt < 100 && Date.now() < deadline; attempt++) {
        flight = readFlight();
        models = readJSON(flight, 'initialModels');
        if (Array.isArray(models)) break;
        await sleep(100);
      }
      if (!Array.isArray(models)) {needUser('Arena model data is unavailable. Complete any browser security screen, then click Continue.'); return;}
      const eligible = models.filter(m => m.organization != null && m.provider != null && m.userSelectable === true && m.capabilities?.inputCapabilities?.text === true && m.capabilities?.outputCapabilities?.text === true);
      const wanted = __MODEL__;
      const model = (wanted.toLowerCase() === 'max' ? eligible.find(m => m.publicName?.toLowerCase() === 'max') : null)
        || eligible.find(m => m.displayName === wanted) || eligible.find(m => m.publicName === wanted)
        || eligible.find(m => m.name === wanted) || eligible.find(m => m.id === wanted);
      if (!model || !model.id) {fail('The requested model is not currently selectable for text requests: ' + __MODEL__); return;}
      state.payload.modelAId = model.id;
      const user = readJSON(flight, 'user');
      // The native Direct form is login-gated for every anonymous account.
      // First-use terms must also be accepted by the user in Arena's own UI.
      if (!user || typeof user.email !== 'string' || !user.email) {
        needUser('Arena Direct requires sign-in. Sign in using the Arena page, then click Continue.'); return;
      }
      if (!user.touConsentTimestamp) {
        needUser('Complete Arena first-use terms in its normal chat interface, then click Continue.'); return;
      }
      state.payload.recaptchaV3Token = await getToken('chat_submit');
      state.phase = 'ready'; emit('ready');
    } catch (error) {needUser(error?.message || 'Arena browser preparation failed. Sign in or complete verification, then click Continue.');}
  })();
})();
"""


def _prepare_script(payload, model, timeout, rejections=0):
    replacements = {"__PAYLOAD__": json.dumps(payload), "__MODEL__": json.dumps(model),
                    "__TIMEOUT_MS__": str(max(1, int(timeout * 1000))),
                    "__V2_SITEKEY__": json.dumps(RECAPTCHA_V2_SITEKEY), "__REJECTIONS__": str(rejections)}
    # One pass prevents markers contained in user prompts from being replaced.
    return re.sub("|".join(map(re.escape, replacements)), lambda m: replacements[m.group()], _PREPARE_JS)


def _browser_helper(config):
    """Run only in the isolated subprocess, never a GUI worker thread."""
    try:
        from PySide6.QtCore import QEventLoop, QTimer, QUrl
        from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
        from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        raise ImportError("AuthArena requires PySide6 QtWebEngine. Install the full PySide6 package or use a Glossarion build with QtWebEngine support.") from exc

    app = QApplication(["autharena-browser-helper"])
    profile_path = Path(config["profile"])
    profile_path.mkdir(parents=True, exist_ok=True)
    profile = QWebEngineProfile("glossarion-autharena-" + str(config["account_id"]), app)
    profile.setPersistentStoragePath(str(profile_path))
    profile.setCachePath(str(profile_path / "cache"))
    profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)
    # Retain the actual Chromium version while removing Qt's application label.
    profile.setHttpUserAgent(re.sub(r"\s+QtWebEngine/\S+", "", profile.httpUserAgent()))
    window = QWidget()
    window.setWindowTitle("Glossarion — Arena browser authentication")
    window.resize(1100, 800)
    layout = QVBoxLayout(window)
    status = QLabel("Sign in or complete any Arena verification, then click Done." if config.get("login") else "Preparing a new Arena request…")
    status.setWordWrap(True)
    layout.addWidget(status)
    view = QWebEngineView(window)
    page = QWebEnginePage(profile, view)
    view.setPage(page)
    layout.addWidget(view)
    proceed = QPushButton("Done" if config.get("login") else "Continue")
    layout.addWidget(proceed)
    cancel = QPushButton("Cancel")
    layout.addWidget(cancel)
    deadline = time.monotonic() + _positive_timeout(config.get("timeout"))
    state = {"finished": False, "injected": False, "polling": False, "loaded": False,
             "rejections": 0, "action": False, "started": time.monotonic()}
    incoming = queue.Queue()
    target_url = CATALOG_URL + "?model_a=" + quote(_normalize_model(config["model"]), safe="")

    def emit(event, **details):
        print(json.dumps({"autharena": 1, "event": event, **details}, ensure_ascii=False), flush=True)

    def finish(event, **details):
        if state["finished"]:
            return
        state["finished"] = True
        timer.stop()
        emit(event, **details)
        window.hide()
        QTimer.singleShot(100, app.quit)

    def show_action(message):
        state["action"] = True
        proceed.setEnabled(True)
        status.setText(message)
        window.show()
        window.raise_()
        emit("status", message=message)

    def inject():
        if state["injected"] or state["finished"] or config.get("login"):
            return
        if page.url().host() != "arena.ai" or page.url().scheme() != "https":
            return
        state["injected"] = True
        page.runJavaScript(_prepare_script(config["payload"], config["model"],
                                          deadline - time.monotonic(), state["rejections"]))

    def on_loaded(ok):
        state["loaded"] = bool(ok)
        if ok:
            inject()
        elif not ok and not config.get("login"):
            show_action("Arena could not load. Check the visible browser, then click Continue.")

    def on_started():
        if state.get("dispatched"):
            finish("error", message="Browser navigation interrupted a dispatched request; completion is uncertain and the request was not retried")
            return
        state["loaded"] = False
        state["injected"] = False

    def continue_clicked():
        if config.get("login"):
            finish("done")
            return
        if state.get("dispatched"):
            return
        state["action"] = False
        state["injected"] = False
        state["started"] = time.monotonic()
        status.setText("Preparing Arena request…")
        page.setUrl(QUrl(target_url))

    def receive_events(value):
        state["polling"] = False
        if state["finished"] or not isinstance(value, str):
            return
        try:
            events = json.loads(value)
        except ValueError:
            return
        for event in events:
            kind = event.pop("event", None)
            if kind == "action":
                state["dispatched"] = False  # Only preparation or explicit HTTP rejection can emit action.
                show_action(event.get("message", "Complete browser verification, then click Continue."))
            elif kind == "rejected":
                state["rejections"] += 1
                emit(kind)
            elif kind in ("error", "done"):
                finish(kind, **event)
                break
            elif kind == "ready":
                state["action"] = False
                proceed.setEnabled(False)
                status.setText("Arena is ready; waiting to send…")
                emit(kind, **event)
            elif kind in ("chunk", "status"):
                emit(kind, **event)

    def tick():
        if state["finished"]:
            return
        if time.monotonic() >= deadline:
            finish("error", message="Request timed out while waiting for Arena/browser verification", error_type="timeout")
            return
        if not config.get("login") and not window.isVisible() and time.monotonic() - state["started"] > 12 and not state.get("dispatched"):
            show_action("Arena is still preparing. Complete any visible browser verification; Continue reloads the page if needed.")
        while not incoming.empty():
            command = incoming.get_nowait()
            if command.get("command") == "dispatch":
                state["dispatched"] = True
                status.setText("Generating an isolated Arena response…")
                page.runJavaScript("if(location.origin==='https://arena.ai')window.__glossarionArena?.dispatch();")
            elif command.get("command") == "cancel":
                finish("error", message="Browser request cancelled")
                return
        if state["loaded"] and not state["polling"] and not config.get("login"):
            state["polling"] = True
            page.runJavaScript("location.origin==='https://arena.ai' ? JSON.stringify(window.__glossarionArena?.events.splice(0,64)||[]) : '[]'", receive_events)

    def read_commands():
        for line in sys.stdin:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    incoming.put(value)
            except ValueError:
                pass
        incoming.put({"command": "cancel"})

    threading.Thread(target=read_commands, daemon=True).start()
    timer = QTimer(window)
    timer.setInterval(50)
    timer.timeout.connect(tick)
    page.loadStarted.connect(on_started)
    page.loadFinished.connect(on_loaded)
    page.newWindowRequested.connect(lambda request: request.openIn(page))
    proceed.clicked.connect(continue_clicked)
    cancel.clicked.connect(lambda: finish("error", message="Browser request cancelled"))
    # Closing the browser must not leave the parent waiting until its timeout.
    original_close = window.closeEvent

    def close_event(event):
        if not state["finished"]:
            finish("error", message="Arena browser was closed")
        original_close(event)

    window.closeEvent = close_event
    page.setUrl(QUrl(target_url))
    if config.get("login"):
        window.show()
        emit("status", message="Sign in or initialize Arena in the browser, then click Done. The browser profile is saved for this account.")
    timer.start()
    app.exec()
    # Destroy the page before the profile and allow Chromium to flush storage.
    view.setPage(None)
    page.deleteLater()
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec()
    profile.deleteLater()
    app.processEvents()
    return 0 if state["finished"] else 1


def _load_cli_messages(args):
    def read_file(value):
        return sys.stdin.read() if value == "-" else Path(value).read_text(encoding="utf-8-sig")

    if args.messages and (args.prompt is not None or args.prompt_file):
        raise ValueError("Use --messages or --prompt/--prompt-file, not both")
    if args.prompt is not None and args.prompt_file:
        raise ValueError("Use --prompt or --prompt-file, not both")
    if args.system is not None and args.system_file:
        raise ValueError("Use --system or --system-file, not both")
    stdin_values = [args.messages, args.prompt_file, args.system_file, args.prompt, args.system]
    if stdin_values.count("-") > 1:
        raise ValueError("Only one input may read stdin")
    if args.messages:
        messages = json.loads(read_file(args.messages))
        if isinstance(messages, dict):
            messages = messages.get("messages")
        if not isinstance(messages, list):
            raise ValueError("--messages must contain a JSON list or an object with a messages list")
    else:
        prompt = read_file(args.prompt_file) if args.prompt_file else args.prompt
        if prompt == "-":
            prompt = sys.stdin.read()
        if prompt is None:
            raise ValueError("Provide --prompt, --prompt-file or --messages")
        messages = [{"role": "user", "content": prompt}]
    system = read_file(args.system_file) if args.system_file else args.system
    if system == "-":
        system = sys.stdin.read()
    if system is not None:
        messages = [{"role": "system", "content": system}] + messages
    return messages


def _main():
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Arena browser-backed text completions. Every request starts a new conversation.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", help="Prompt text, or '-' for stdin")
    parser.add_argument("--prompt-file", help="UTF-8 prompt file, or '-' for stdin")
    parser.add_argument("--messages", help="JSON message-list file, or '-' for stdin")
    parser.add_argument("--system")
    parser.add_argument("--system-file")
    parser.add_argument("--list-models", action="store_true", help="Poll public text models over HTTP without opening a browser")
    parser.add_argument("--login", action="store_true", help="Open the persistent Arena browser profile for sign-in")
    parser.add_argument("--account-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--browser-helper", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.browser_helper:
        os.environ["AUTHARENA_BROWSER_HELPER"] = "1"
        try:
            return _browser_helper(json.loads(sys.stdin.readline()))
        except Exception as exc:
            print(json.dumps({"autharena": 1, "event": "error", "message": _short_error(exc),
                              "error_type": "configuration" if isinstance(exc, ImportError) else
                              "timeout" if isinstance(exc, TimeoutError) else "api"}), flush=True)
            return 1
    log_fn = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
    try:
        if args.list_models and args.login:
            raise ValueError("Use --list-models or --login, not both")
        if args.list_models:
            result = fetch_available_models(timeout=args.timeout, account_id=args.account_id)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "\n".join(result))
        elif args.login:
            result = login(account_id=args.account_id, timeout=args.timeout, log_fn=log_fn)
            print(json.dumps(result) if args.json else "Arena browser profile saved.")
        else:
            result = send_chat_completion(messages=_load_cli_messages(args), model=args.model,
                                          timeout=args.timeout, account_id=args.account_id, log_fn=log_fn)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["content"])
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            cancel_stream()
        print("AuthArena error: " + _short_error(exc or "cancelled"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
