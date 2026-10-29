"""LocalLLMServer: the on-device engine's OpenAI-compatible endpoint on 127.0.0.1.

Modelled on ``reader_server.ReaderServer``: stdlib ``ThreadingHTTPServer`` on ``("127.0.0.1", 0)``,
daemon thread ``gl-llm-http``, a random 192-bit path token compared in constant time, a Host-header
guard (DNS rebinding) and bare 404s for everything else. ``base_url`` is
``http://127.0.0.1:<port>/<token>/v1`` (published as ``GLOSSARION_LOCAL_LLM_URL``): the OpenAI SDK
appends ``/chat/completions``, so the token travels in the path and the dummy Bearer key is ignored.

* ``GET /<token>/v1/models``: the downloaded models;
* ``POST /<token>/v1/chat/completions``: streaming (SSE, ``data: [DONE]``) and plain answers in the
  OpenAI schema with ``usage``. A vision model (downloaded with its projector) takes ``image_url``
  parts with ``data:`` URLs; a text-only model answers 400 "no vision support". Errors use the OpenAI
  error shape (404 ``model_not_found``, 400 ``context_length_exceeded`` / no vision / bad image, 409
  busy with a job's model, 503 queue full);
* ``GET /<token>/health``: ``{"state", "model", "queue", "busy", "load_seconds"}``; ``?model=<name>`` adds
  ``target`` (``{"stem", "size", "loaded"}``), which the shared client reads to log loading / waiting.

``start()`` after ``stop()`` binds the same port again (same token): ``base_url`` does not change.

A generation stops within a few tokens when the client goes away (Stop in the app closes the
connection): the engine's stop check peeks at the socket, and a failed streaming write stops it too.
A leading ``<think>…</think>`` block is never sent.

The model work lives in a backend (``service.LocalEngineService``): ``list_models()``, ``health()``
and ``chat(model, body, should_stop) -> ChatRun``. Pure standard library; no Flet import.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import select
import socket
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator, Optional
from urllib.parse import parse_qs, unquote, urlsplit

log = logging.getLogger("glossarion.local_llm")

__all__ = ["ChatRun", "LocalLLMError", "LocalLLMServer", "ThinkStripper", "client_gone"]

MAX_BODY_BYTES = 32 * 1024 * 1024  # base64 photos (the service scales them down)
MAX_DRAIN_BYTES = 1024 * 1024


class LocalLLMError(Exception):
    """An OpenAI-shaped error answer: HTTP ``status``, ``code`` and the message the app shows."""

    def __init__(self, status: int, code: str, message: str, err_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message
        self.err_type = err_type

    def body(self) -> bytes:
        return json.dumps({"error": {"message": self.message, "type": self.err_type, "code": self.code,
                                     "param": None}}).encode("utf-8")


def wants_thinking(body: Any) -> bool:
    """Whether a request asks the model to think: ``chat_template_kwargs.enable_thinking`` when it says,
    else an OpenAI ``reasoning_effort`` other than none (hotfix 8, owner 2026-10-30: "Pass thinking to
    unknown routes" sends ``reasoning_effort``, which the engine used to ignore - Qwen3.5 never thought)."""
    if not isinstance(body, dict):
        return False
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        return bool(kwargs.get("enable_thinking"))
    effort = str(body.get("reasoning_effort") or "").strip().lower()
    return bool(effort) and effort not in ("none", "off", "disabled", "false", "0")


class ChatRun:
    """One admitted generation: iterate ``deltas`` (text), then read ``usage`` / ``finish_reason``;
    ``close()`` (always) gives the engine back to the next request."""

    def __init__(self, stem: str, deltas: Iterator[str], *, usage: Callable[[], dict],
                 finish: Callable[[], str], release: Callable[[], None]) -> None:
        self.stem = stem
        self.deltas = deltas
        self._usage = usage
        self._finish = finish
        self._release = release
        self._closed = False

    @property
    def usage(self) -> dict:
        return dict(self._usage() or {})

    @property
    def finish_reason(self) -> str:
        return self._finish() or "stop"

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                close = getattr(self.deltas, "close", None)
                if close is not None:
                    close()
            finally:
                self._release()

    def __enter__(self) -> "ChatRun":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class ThinkStripper:
    """Drops a leading ``<think>…</think>`` block (and the blank lines after it) from a stream."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buffer = ""
        self.mode = "start"  # start | think | after | pass

    def feed(self, text: str) -> str:
        if self.mode == "pass":
            return text
        self.buffer += text
        if self.mode == "start":
            stripped = self.buffer.lstrip()
            if not stripped:
                return ""
            if stripped.startswith(self.OPEN):
                self.mode = "think"
                self.buffer = stripped[len(self.OPEN):]
            elif self.OPEN.startswith(stripped):
                return ""  # could still become "<think>"
            else:
                self.mode = "pass"
                out, self.buffer = self.buffer, ""
                return out
        if self.mode == "think":
            index = self.buffer.find(self.CLOSE)
            if index < 0:
                self.buffer = self.buffer[-len(self.CLOSE):]
                return ""
            self.buffer = self.buffer[index + len(self.CLOSE):]
            self.mode = "after"
        if self.mode == "after":
            stripped = self.buffer.lstrip()
            if not stripped:
                return ""
            self.mode = "pass"
            self.buffer = ""
            return stripped
        return ""

    def flush(self) -> str:
        if self.mode in ("start", "pass"):
            out, self.buffer = self.buffer, ""
            self.mode = "pass"
            return out
        self.buffer = ""
        return ""


def client_gone(sock: Any) -> bool:
    """True when the peer closed the connection (readable and a zero-byte peek)."""
    try:
        readable, _w, _x = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Linux / Android: rebinding the previous port right after a stop must not fail on the connections
    # still in TIME_WAIT (SO_REUSEADDR there never lets two listeners share a port). Windows' SO_REUSEADDR
    # does allow that, so it stays off there.
    allow_reuse_address = os.name != "nt"
    # several jobs' worker threads connect at once (manga: 4); the default 5 is the kernel's backlog
    request_queue_size = 64

    def __init__(self, owner: "LocalLLMServer", port: int = 0) -> None:
        self.owner = owner
        super().__init__((owner.host, int(port)), _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            log.debug("llm http: %s from %s", type(error).__name__, client_address)
            return
        log.exception("llm http: request from %s failed", client_address)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"
    server_version = "GlossarionLLM"
    sys_version = ""
    # socket timeout per read/write: an idle or stalled connection (another app that never sends a
    # request line) frees its thread; generation itself does no socket reads, so it is not limited
    timeout = 60

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("llm http: " + format, *args)

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/json",
              headers: Optional[dict] = None) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _not_found(self) -> None:
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_address[1]
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _route(self) -> Optional[list]:
        owner = self.server.owner
        if not self._host_ok():
            return None
        segments = [unquote(s) for s in urlsplit(self.path).path.split("/") if s]
        if not segments or not owner.check_token(segments[0]):
            return None
        return segments[1:]

    def do_GET(self) -> None:  # noqa: N802
        rest = self._route()
        owner = self.server.owner
        if rest == ["v1", "models"]:
            models = owner.backend.list_models()
            self._json(200, {"object": "list", "data": [
                {"id": m["id"], "object": "model", "created": int(m.get("created") or 0), "owned_by": "local",
                 "meta": {k: v for k, v in m.items() if k not in ("id", "created")}} for m in models]})
        elif rest == ["health"]:
            query = parse_qs(urlsplit(self.path).query)
            model = (query.get("model") or [None])[0]
            self._json(200, owner.backend.health(model) if model else owner.backend.health())
        else:
            self._not_found()

    def do_HEAD(self) -> None:  # noqa: N802
        self._not_found()

    def do_POST(self) -> None:  # noqa: N802
        rest = self._route()
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if rest != ["v1", "chat", "completions"]:
            self.close_connection = True
            if 0 < length <= MAX_DRAIN_BYTES:
                self._drain(length)
            self._not_found()
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, LocalLLMError(413, "request_too_large",
                                                                          "The request is too large").body())
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("not an object")
        except (ValueError, UnicodeDecodeError):
            self._send(400, LocalLLMError(400, "invalid_json", "The request body is not JSON").body())
            return
        self._chat(body)

    def _drain(self, length: int) -> None:
        try:
            self.connection.settimeout(5)
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            pass

    def _chat(self, body: dict) -> None:
        owner = self.server.owner
        sock = self.connection
        gone = [False]

        def should_stop() -> bool:
            if not gone[0] and client_gone(sock):
                gone[0] = True
                log.info("on-device generation: the client went away; stopping")
            return gone[0]

        stream = bool(body.get("stream"))
        model_name = str(body.get("model") or "")
        try:
            run = owner.backend.chat(model_name, body, should_stop)
        except LocalLLMError as exc:
            self._send(exc.status, exc.body())
            return
        except Exception as exc:
            log.exception("on-device chat failed")
            self._send(500, LocalLLMError(500, "engine_error", f"The on-device engine failed: {exc}",
                                          "server_error").body())
            return
        created = int(time.time())
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        model_id = f"local/{run.stem}"
        thinking = wants_thinking(body)
        strip = None if thinking else ThinkStripper()
        with run:
            if not stream:
                parts: list = []
                try:
                    for delta in run.deltas:
                        parts.append(delta)
                except Exception as exc:
                    log.exception("on-device generation failed")
                    self._send(500, LocalLLMError(500, "engine_error", f"The on-device engine failed: {exc}",
                                                  "server_error").body())
                    return
                text = "".join(parts)
                if strip is not None:
                    text = (strip.feed(text) + strip.flush())
                if gone[0]:
                    return
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": run.finish_reason, "logprobs": None}],
                    "usage": _usage(run.usage),
                })
                return
            self.close_connection = True
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self._event({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                          "finish_reason": None}]})
                for delta in run.deltas:
                    text = strip.feed(delta) if strip is not None else delta
                    if text:
                        self._event(_chunk(rid, created, model_id, {"content": text}))
                tail = strip.flush() if strip is not None else ""
                if tail:
                    self._event(_chunk(rid, created, model_id, {"content": tail}))
                final = _chunk(rid, created, model_id, {}, run.finish_reason)
                final["usage"] = _usage(run.usage)
                self._event(final)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                gone[0] = True
                log.info("on-device generation: the stream was closed by the client")
            except Exception as exc:
                log.exception("on-device generation failed")
                try:
                    self._event({"error": {"message": f"The on-device engine failed: {exc}", "type": "server_error",
                                           "code": "engine_error"}})
                except OSError:
                    pass

    def _event(self, data: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()


def _chunk(rid: str, created: int, model: str, delta: dict, finish: Optional[str] = None) -> dict:
    return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _usage(usage: dict) -> dict:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


class LocalLLMServer:
    """The loopback endpoint (``start()`` / ``stop()``); ``backend`` does the model work."""

    def __init__(self, backend: Any, *, host: str = "127.0.0.1") -> None:
        self.backend = backend
        self.host = host
        self.token = secrets.token_urlsafe(24)  # 192 bits
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_port = 0  # a restart binds it again: base_url (token and port) never changes

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def port(self) -> int:
        server = self._server
        return int(server.server_address[1]) if server is not None else 0

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<port>/<token>/v1`` (the OpenAI base URL)."""
        return f"{self.origin}/{self.token}/v1"

    @property
    def health_url(self) -> str:
        return f"{self.origin}/{self.token}/health"

    def check_token(self, value: Any) -> bool:
        text = str(value or "")
        return bool(text) and hmac.compare_digest(text.encode("utf-8", "replace"), self.token.encode("ascii"))

    def start(self) -> int:
        with self._lock:
            if self._server is not None:
                return self.port
            server = None
            if self._last_port:
                try:
                    server = _Server(self, self._last_port)
                except OSError:
                    log.info("on-device LLM endpoint: port %d is taken; using a new one", self._last_port)
            if server is None:
                server = _Server(self)
            self._server = server
            self._last_port = int(server.server_address[1])
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.25},
                                      name="gl-llm-http", daemon=True)
            self._thread = thread
        thread.start()
        log.info("on-device LLM endpoint on 127.0.0.1:%d", self.port)
        return self.port

    def stop(self) -> None:
        with self._lock:
            server, self._server = self._server, None
            thread, self._thread = self._thread, None
        if server is None:
            return
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            log.debug("stopping the LLM endpoint failed", exc_info=True)
        if thread is not None:
            thread.join(timeout=2.0)
