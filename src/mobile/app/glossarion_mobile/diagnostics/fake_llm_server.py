"""A fake OpenAI-compatible LLM server on 127.0.0.1 for the offline end-to-end test.

The backend talks to it through its normal custom-endpoint route
(``USE_CUSTOM_OPENAI_ENDPOINT=1``, ``OPENAI_CUSTOM_BASE_URL=<server.url>``, any
dummy key), so a run exercises the real ``unified_api_client`` -> ``openai`` ->
``httpx`` stack without a network connection or an account. Stdlib only, Python
3.10 compatible, safe to import on a device (no backend import).

Endpoints::

    GET  /v1/models                  model list (OpenAI shape)
    GET  /v1/models/<id>             one model
    POST /v1/chat/completions        JSON, or SSE when "stream": true
    POST /v1/images/generations      a PNG as b64_json (the OpenAI Images API shape)
    GET  /health                     {"ok": true}

What it answers:

* **Glossary extraction requests** (the prompt asks for ``raw_name`` /
  ``translated_name`` columns): a CSV glossary
  ``type,raw_name,translated_name,gender,description`` with the entries of
  ``glossary`` whose raw name occurs in the request (the self-test EPUB's
  recurring names by default, see ``assets/selftest/MANIFEST.toml``).
* **Everything else is "translated" by tagging**: the last user message comes
  back unchanged except that every run of Hangul becomes ``<marker> <romanized
  text>`` (``GLFAKE saebyeok angae...``). Glossary terms the prompt carries (a
  line naming both the raw and the translated name, i.e. the glossary was
  injected) are substituted first, so a run that used its glossary shows
  ``Seo-yeon Lee`` instead of the romanized name. Markup, chapter split markers
  and JSON stay intact, so the backend's own parsing works unchanged.
* **Vision requests** (a user message with an ``image_url`` part, i.e. ``send_image``):
  the fixed OCR text ``FAKE_OCR_TEXT`` (U7 Vision output mode).
* **Image generation** (``/v1/images/generations``, the client's Images API route for the
  Image output mode): ``FAKE_PNG`` as ``b64_json``; the prompt is recorded in ``preview``.

Test controls: ``delay`` / ``stream_chunk_delay`` slow responses down;
``hold()`` / ``release(abort=...)`` park new requests until released (a "stuck" model
for force-stop tests; parked requests answer after ``hold_timeout`` at the latest, or are
dropped without a response when released with ``abort=True``);
``on_request`` / ``on_response`` hooks run on the handler thread with the
``RequestRecord`` (the E2E uses them to press Stop at a precise point);
``leave_raw_once[chapter] = text`` makes the next answer for that chapter keep ``text``
(Korean) untranslated, so a QA scan flags the chapter (U7 Resolve QA).
Every request is recorded in ``requests``.
"""

from __future__ import annotations

import base64
import json
import re
import socket
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = [
    "DEFAULT_GLOSSARY",
    "FAKE_MARKER",
    "FAKE_MODEL",
    "FAKE_OCR_TEXT",
    "FAKE_PNG",
    "FakeLLMServer",
    "GlossaryEntry",
    "RequestRecord",
    "applied_entries",
    "chapter_numbers",
    "classify_request",
    "fake_translate",
    "glossary_csv",
    "has_image_part",
    "png_bytes",
    "romanize_hangul",
]

FAKE_MARKER = "GLFAKE"
FAKE_MODEL = "gpt-4o-mini"  # an OpenAI chat model name: routed through the custom endpoint as-is
#: What the model "reads" from any image (Vision output mode).
FAKE_OCR_TEXT = "GLFAKE-OCR The knight of the Silver Forest raised her sword at dawn."


def png_bytes(width: int = 8, height: int = 8, rgb: tuple = (200, 40, 40)) -> bytes:
    """A valid solid-colour RGB PNG (stdlib only)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * int(width)
    raw = row * int(height)
    header = struct.pack(">IIBBBBB", int(width), int(height), 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


#: The image every generation request gets (32x32, blue).
FAKE_PNG = png_bytes(32, 32, (30, 90, 200))


@dataclass(frozen=True)
class GlossaryEntry:
    type: str
    raw_name: str
    translated_name: str
    gender: str = ""
    description: str = ""


#: The self-test EPUB's recurring names (assets/selftest/MANIFEST.toml [glossary]).
DEFAULT_GLOSSARY: tuple = (
    GlossaryEntry("character", "이서연", "Seo-yeon Lee", "Female", "Young knight of the Silver Forest"),
    GlossaryEntry("character", "강민호", "Min-ho Kang", "Male", "Swordsman of the Azure Knights"),
    GlossaryEntry("character", "한지우", "Ji-woo Han", "Female", "Heir to the Arden throne"),
    GlossaryEntry("character", "백도윤", "Do-yoon Baek", "Male", "Master of the Black Tower"),
    GlossaryEntry("term", "아르덴 왕국", "Kingdom of Arden", "", "The kingdom"),
    GlossaryEntry("term", "은빛 숲", "Silver Forest", "", "Forest north of the capital"),
    GlossaryEntry("term", "검은 탑", "Black Tower", "", "Sorcerer's tower"),
    GlossaryEntry("term", "루미나 성", "Lumina Castle", "", "Royal castle"),
    GlossaryEntry("term", "마나석", "mana stone", "", "Stone that stores mana"),
    GlossaryEntry("term", "성검 엘리시온", "Holy Sword Elysion", "", "Legendary sword"),
    GlossaryEntry("term", "푸른 기사단", "Azure Knights", "", "Knightly order"),
)

# ---------------------------------------------------------------------------
# Deterministic "translation"
# ---------------------------------------------------------------------------

_INITIALS = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h")
_MEDIALS = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi",
            "yu", "eu", "ui", "i")
_FINALS = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t",
           "ng", "t", "t", "k", "t", "p", "t")
#: A run of Hangul words (digits and blanks inside a run belong to it: "제12화 새로운 아침").
_HANGUL_RUN = re.compile("[\uac00-\ud7a3](?:[\uac00-\ud7a3\\d \\t]*[\uac00-\ud7a3\\d])?")
_CHAPTER_HEADING = re.compile(r"제\s*(\d+)\s*화")


def romanize_hangul(text: str) -> str:
    """Romanize Hangul syllables (simplified Revised Romanization; other characters unchanged)."""
    out = []
    for ch in str(text):
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_INITIALS[code // 588] + _MEDIALS[(code % 588) // 28] + _FINALS[code % 28])
        else:
            out.append(ch)
    return "".join(out)


def applied_entries(request_text: str, glossary: Iterable[GlossaryEntry]) -> list:
    """Entries the request carries: one prompt line names both the raw and the translated name
    (``* 이서연 = Seo-yeon Lee [Female]: ...`` or ``character,이서연,Seo-yeon Lee,...``), which is
    what an injected glossary looks like; a translated name inside a description does not count."""
    lines = str(request_text).splitlines()
    return [
        e for e in glossary
        if e.raw_name and e.translated_name and any(e.raw_name in line and e.translated_name in line for line in lines)
    ]


def fake_translate(text: str, glossary: Iterable[GlossaryEntry] = (), *, marker: str = FAKE_MARKER) -> str:
    """Substitute ``glossary`` terms, then tag and romanize every remaining Hangul run."""
    result = str(text)
    for entry in sorted(glossary, key=lambda e: len(e.raw_name), reverse=True):
        if entry.raw_name:
            result = result.replace(entry.raw_name, f"{marker} {entry.translated_name} ")
    return _HANGUL_RUN.sub(lambda m: f"{marker} {romanize_hangul(m.group(0))}", result)


def glossary_csv(text: str, glossary: Iterable[GlossaryEntry]) -> str:
    """The CSV glossary for the entries whose raw name occurs in ``text`` (header always present)."""
    lines = ["type,raw_name,translated_name,gender,description"]
    for entry in glossary:
        if entry.raw_name and entry.raw_name in text:
            fields = (entry.type, entry.raw_name, entry.translated_name, entry.gender, entry.description)
            lines.append(",".join(_csv_field(v) for v in fields))
    return "\n".join(lines) + "\n"


def _csv_field(value: str) -> str:
    value = str(value or "")
    if any(c in value for c in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def chapter_numbers(text: str) -> list:
    """Chapter numbers named in ``text`` (the fixture's ``제N화`` headings), sorted, unique."""
    return sorted({int(n) for n in _CHAPTER_HEADING.findall(str(text))})


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text"):
                    parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


#: The extraction prompt asks for the ``type`` column first; an injected reference glossary in a
#: translation prompt lists ``raw_name, translated_name, ...`` without it.
_EXTRACTION_COLUMNS = re.compile(r"\btype\s*,\s*raw_name\s*,\s*translated_name", re.IGNORECASE)


def has_image_part(payload: dict) -> bool:
    """A message carries an ``image_url`` part (``UnifiedClient.send_image``)."""
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "input_image")
                                             for p in content):
            return True
    return False


def classify_request(payload: dict) -> str:
    """``"glossary"`` for glossary extraction prompts, ``"vision"`` for image requests, else ``"translation"``."""
    if has_image_part(payload):
        return "vision"
    messages = payload.get("messages") or []
    text = "\n".join(_content_text(m.get("content")) for m in messages if isinstance(m, dict))
    lowered = text.lower()
    if "glossary extraction" in lowered or ("extract" in lowered and _EXTRACTION_COLUMNS.search(text)):
        return "glossary"
    return "translation"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """One request the server answered (or is answering)."""

    id: int
    path: str
    kind: str = ""
    model: str = ""
    stream: bool = False
    started: float = 0.0
    finished: Optional[float] = None
    status: str = "pending"  # pending | ok | aborted | error
    parked: bool = False  # waited in hold()
    chapters: list = field(default_factory=list)  # 제N화 headings in the last user message
    glossary_applied: list = field(default_factory=list)  # raw names substituted from the prompt's glossary
    # prompt lines outside the last user message that name a raw glossary term (the injected glossary)
    glossary_lines: list = field(default_factory=list)
    prompt_chars: int = 0
    reply_chars: int = 0
    reply: str = ""
    preview: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "path": self.path, "kind": self.kind, "model": self.model, "stream": self.stream,
            "started": self.started, "finished": self.finished, "status": self.status, "parked": self.parked,
            "chapters": list(self.chapters),
            "glossary_applied": list(self.glossary_applied), "glossary_lines": list(self.glossary_lines),
            "prompt_chars": self.prompt_chars,
            "reply_chars": self.reply_chars, "preview": self.preview,
        }


Hook = Callable[[RequestRecord], Any]


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple, handler: type, owner: "FakeLLMServer") -> None:
        self.owner = owner
        super().__init__(address, handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that hung up (stop, keep-alive reset) is normal here: no traceback on stderr."""
        import sys

        exc = sys.exc_info()[1]
        self.owner._access(f"connection error from {client_address}: {type(exc).__name__}: {exc}")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler API
        self.server.owner._access(format % args)

    def setup(self) -> None:
        super().setup()
        self.server.owner._track(self.connection, True)

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.server.owner._track(self.connection, False)

    # ---- helpers ------------------------------------------------------------------------

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": kind, "code": None}})

    def _read_json(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    # ---- routes ---------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        owner = self.server.owner
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/health", ""):
            self._send_json(200, {"ok": True})
        elif path in ("/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": [owner._model_entry(m) for m in owner.models]})
        elif path.startswith(("/v1/models/", "/models/")):
            model = path.rsplit("/", 1)[-1]
            self._send_json(200, owner._model_entry(model))
        else:
            self._error(404, f"Unknown path {path}")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        owner = self.server.owner
        path = self.path.split("?", 1)[0].rstrip("/")
        payload = self._read_json()
        if path in ("/v1/images/generations", "/images/generations"):
            if payload is None:
                self._error(400, "Request body is not a JSON object")
                return
            owner._serve_image(self, payload, path)
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            owner._record_unsupported(path)
            self._error(404, f"The fake server only implements /v1/chat/completions (got {path})")
            return
        if payload is None:
            self._error(400, "Request body is not a JSON object")
            return
        owner._serve_chat(self, payload, path)


class FakeLLMServer:
    """Threaded fake OpenAI server; ``with FakeLLMServer() as server: server.url``."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        marker: str = FAKE_MARKER,
        glossary: Sequence[GlossaryEntry] = DEFAULT_GLOSSARY,
        models: Sequence[str] = (FAKE_MODEL,),
        delay: float = 0.0,
        stream_chunk_delay: float = 0.0,
        stream_chunks: int = 4,
        hold_timeout: float = 120.0,
    ) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the fake LLM server only listens on the loopback interface")
        self.host = host
        self.port = int(port)
        self.marker = marker
        self.glossary = tuple(glossary)
        self.models = tuple(models)
        self.delay = float(delay)
        self.kind_delays: dict = {}
        self.stream_chunk_delay = float(stream_chunk_delay)
        self.stream_chunks = max(1, int(stream_chunks))
        self.hold_timeout = float(hold_timeout)
        self.on_request: list = []
        self.on_response: list = []
        #: chapter number -> Korean text the next answer for that chapter keeps untranslated (once)
        self.leave_raw_once: dict = {}
        self.requests: list = []
        self.access_log: list = []
        self._lock = threading.Lock()
        self._gate = threading.Condition()
        self._holding = False
        self._hold_generation = 0
        self._aborted_generations: set = set()
        self._parked = 0
        self._stopping = threading.Event()
        self._connections: set = set()
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------------------------

    @property
    def url(self) -> str:
        """The OpenAI base URL (``http://127.0.0.1:<port>/v1``)."""
        if self._server is None:
            raise RuntimeError("the fake LLM server is not running")
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> "FakeLLMServer":
        if self._server is not None:
            return self
        self._stopping.clear()
        self._server = _Server((self.host, self.port), _Handler, self)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1},
                                        name="gl-fake-llm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        server, self._server = self._server, None
        self._stopping.set()
        self.release(abort=True)
        if server is None:
            return
        server.shutdown()
        with self._lock:
            connections = list(self._connections)
        for conn in connections:  # keep-alive connections would park their handler threads
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        server.server_close()
        if self._thread is not None:
            self._thread.join(5)
            self._thread = None

    def __enter__(self) -> "FakeLLMServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ---- controls ---------------------------------------------------------------------------

    def hold(self) -> None:
        """Park new requests (a model that stops answering) until ``release()``."""
        with self._gate:
            if not self._holding:
                self._holding = True
                self._hold_generation += 1

    def release(self, *, abort: bool = False) -> None:
        """Let parked requests continue; ``abort=True`` drops them instead (the connection closes
        without a response, as when a model host goes away), so a stopped run never receives them."""
        with self._gate:
            if self._holding:
                if abort:
                    self._aborted_generations.add(self._hold_generation)
                self._holding = False
            self._gate.notify_all()

    @property
    def holding(self) -> bool:
        with self._gate:
            return self._holding

    @property
    def parked(self) -> int:
        """Requests waiting in ``hold()`` right now."""
        with self._gate:
            return self._parked

    def set_delay(self, kind: str, seconds: float) -> None:
        """Extra delay before answering requests of ``kind`` ("translation" / "glossary")."""
        self.kind_delays[str(kind)] = float(seconds)

    def reset(self) -> None:
        """Forget recorded requests and hooks; release held requests; clear delays."""
        with self._lock:
            self.requests = []
            self.access_log = []
        self.on_request = []
        self.on_response = []
        self.kind_delays = {}
        self.leave_raw_once = {}
        self.release(abort=True)

    # ---- queries ---------------------------------------------------------------------------

    def records(self, kind: Optional[str] = None, *, status: Optional[str] = None, since: int = 0) -> list:
        with self._lock:
            items = list(self.requests[since:])
        return [r for r in items if (kind is None or r.kind == kind) and (status is None or r.status == status)]

    def count(self, kind: Optional[str] = None, *, status: Optional[str] = "ok", since: int = 0) -> int:
        return len(self.records(kind, status=status, since=since))

    def mark(self) -> int:
        """Index for ``records(since=...)`` (requests recorded so far)."""
        with self._lock:
            return len(self.requests)

    def in_flight(self) -> int:
        return len(self.records(status="pending"))

    # ---- internals ---------------------------------------------------------------------------

    def _track(self, conn: Any, add: bool) -> None:
        with self._lock:
            if add:
                self._connections.add(conn)
            else:
                self._connections.discard(conn)

    def _access(self, line: str) -> None:
        with self._lock:
            self.access_log.append(line)
            del self.access_log[:-500]

    def _model_entry(self, model: str) -> dict:
        return {"id": model, "object": "model", "created": 1767225600, "owned_by": "glossarion-fake"}

    def _record_unsupported(self, path: str) -> None:
        with self._lock:
            record = RequestRecord(id=len(self.requests) + 1, path=path, kind="unsupported", started=time.time(),
                                   finished=time.time(), status="error")
            self.requests.append(record)

    def _run_hooks(self, hooks: list, record: RequestRecord) -> None:
        for hook in list(hooks):
            try:
                hook(record)
            except Exception:  # a test hook must never break the response
                pass

    def _wait_released(self, record: RequestRecord) -> bool:
        """Wait while held; False when the hold this request waited in was released with ``abort``."""
        with self._gate:
            if not self._holding:
                return True
            generation = self._hold_generation
            record.parked = True
            self._parked += 1
            deadline = time.monotonic() + self.hold_timeout
            try:
                while self._holding and self._hold_generation == generation and not self._stopping.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._gate.wait(min(0.1, remaining))
            finally:
                self._parked -= 1
            return generation not in self._aborted_generations

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._stopping.wait(seconds)

    def _reply_for(self, payload: dict, record: RequestRecord) -> str:
        messages = [m for m in (payload.get("messages") or []) if isinstance(m, dict)]
        full_text = "\n".join(_content_text(m.get("content")) for m in messages)
        last_user = next((_content_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
        record.prompt_chars = len(full_text)
        record.preview = " ".join(last_user.split())[:160]
        record.chapters = chapter_numbers(last_user)
        if record.kind == "glossary":
            return glossary_csv(last_user or full_text, self.glossary)
        if record.kind == "vision":
            return FAKE_OCR_TEXT
        applied = applied_entries(full_text, self.glossary)
        record.glossary_applied = [e.raw_name for e in applied if e.raw_name in last_user]
        user_lines = set(last_user.splitlines())
        record.glossary_lines = [line for line in full_text.splitlines() if line not in user_lines
                                 and any(e.raw_name and e.raw_name in line for e in self.glossary)]
        reply = fake_translate(last_user, applied, marker=self.marker)
        with self._lock:
            # only a chapter's own request (one heading), never a TOC / headers batch naming many chapters
            raw = (self.leave_raw_once.pop(record.chapters[0], None) if len(record.chapters) == 1 else None)
        if raw:
            # a model that left a passage of the chapter untranslated (the QA scan flags it)
            reply = reply.replace("<p>", f"<p>{raw} ", 1) if "<p>" in reply else f"{reply}\n{raw}"
        return reply

    def _serve_image(self, handler: _Handler, payload: dict, path: str) -> None:
        """``/v1/images/generations``: one ``FAKE_PNG`` per request (``n`` ignored)."""
        with self._lock:
            record = RequestRecord(id=len(self.requests) + 1, path=path, kind="image_generation",
                                   model=str(payload.get("model") or ""), started=time.time())
            self.requests.append(record)
        try:
            prompt = str(payload.get("prompt") or "")
            record.prompt_chars = len(prompt)
            record.preview = " ".join(prompt.split())[:160]
            self._run_hooks(self.on_request, record)
            if not self._wait_released(record):
                record.status = "aborted"
                handler.close_connection = True
                return
            self._sleep(self.delay + self.kind_delays.get("image_generation", 0.0))
            body = {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(FAKE_PNG).decode("ascii")}]}
            record.reply = "<png>"
            record.reply_chars = len(FAKE_PNG)
            self._complete(record)
            handler._send_json(200, body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout, OSError):
            record.status = "aborted"
        finally:
            if record.finished is None:
                record.finished = time.time()
            if record.status == "ok":
                self._run_hooks(self.on_response, record)

    def _serve_chat(self, handler: _Handler, payload: dict, path: str) -> None:
        kind = classify_request(payload)
        with self._lock:
            record = RequestRecord(id=len(self.requests) + 1, path=path, kind=kind, model=str(payload.get("model") or ""),
                                   stream=bool(payload.get("stream")), started=time.time())
            self.requests.append(record)
        try:
            reply = self._reply_for(payload, record)
            record.reply = reply
            record.reply_chars = len(reply)
            self._run_hooks(self.on_request, record)
            if not self._wait_released(record):
                record.status = "aborted"
                handler.close_connection = True  # no response at all
                return
            self._sleep(self.delay + self.kind_delays.get(kind, 0.0))
            if self._stopping.is_set():
                record.status = "aborted"
                handler.close_connection = True
                return
            usage = {
                "prompt_tokens": max(1, record.prompt_chars // 4),
                "completion_tokens": max(1, len(reply) // 4),
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            model = record.model or (self.models[0] if self.models else FAKE_MODEL)
            if record.stream:
                self._stream(handler, payload, reply, model, usage, record)
            else:
                body = {
                    "id": f"chatcmpl-fake-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                                 "finish_reason": "stop", "logprobs": None}],
                    "usage": usage,
                }
                self._complete(record)  # before the client can see the answer (tests read it right after)
                handler._send_json(200, body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout, OSError):
            record.status = "aborted"  # the client hung up (a stop closed the stream)
        finally:
            if record.finished is None:
                record.finished = time.time()
            if record.status == "ok":
                self._run_hooks(self.on_response, record)

    @staticmethod
    def _complete(record: RequestRecord) -> None:
        record.status = "ok"
        record.finished = time.time()

    def _stream(self, handler: _Handler, payload: dict, reply: str, model: str, usage: dict,
                record: RequestRecord) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        completion_id = f"chatcmpl-fake-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def event(data: Any) -> None:
            text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            body = f"data: {text}\n\n".encode("utf-8")
            handler.wfile.write(f"{len(body):X}\r\n".encode("ascii") + body + b"\r\n")
            handler.wfile.flush()

        def chunk(delta: dict, finish: Optional[str] = None) -> dict:
            return {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}

        event(chunk({"role": "assistant", "content": ""}))
        size = max(1, -(-len(reply) // self.stream_chunks))
        for start in range(0, len(reply), size):
            if self._stopping.is_set():
                raise ConnectionAbortedError("server stopping")
            event(chunk({"content": reply[start:start + size]}))
            self._sleep(self.stream_chunk_delay)
        event(chunk({}, "stop"))
        options = payload.get("stream_options") or {}
        if isinstance(options, dict) and options.get("include_usage"):
            event({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [], "usage": usage})
        self._complete(record)  # before the terminating events reach the client
        event("[DONE]")
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()


def main(argv: Optional[list] = None) -> int:  # pragma: no cover - manual use
    import argparse

    parser = argparse.ArgumentParser(description="Fake OpenAI-compatible server (Glossarion offline E2E)")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args(argv)
    with FakeLLMServer(port=args.port, delay=args.delay) as server:
        print(f"fake LLM server on {server.url} (Ctrl+C to stop)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
