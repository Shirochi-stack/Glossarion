"""ReaderServer: the Reader's in-app HTTP server (UI_SPEC §3.11 "Document").

The Reader's WebView loads ``reader_doc.wrap_reader_html(mobile=True)`` from
``http://127.0.0.1:<port>/<token>/reader.html?v=N`` instead of ``file://`` or
``load_html`` (Android API 30+ file-access limits, iOS read-access scoping,
HTML size limits for books with many images; mobile blueprint §8 item 3):

* bound to ``127.0.0.1`` on a random port (``("127.0.0.1", 0)``), one daemon
  thread (``gl-reader-http``) plus one short thread per request;
* every resource lives under a random 192-bit path token
  (``/<token>/reader.html``, ``/<token>/img/<id>``); the token is compared in
  constant time and anything else is a bare 404, so another app on the device
  cannot read the book by scanning localhost ports;
* the page's event fallback ``fetch('/__ev')`` (when the WebView console
  bridge is unavailable) is accepted at ``/<token>/__ev`` and at ``/__ev`` with
  the token in the ``glrdr`` cookie (set ``HttpOnly; SameSite=Strict`` on every
  document response), an ``X-GLRDR-Token`` header or a ``t=`` query value;
* the ``Host`` header must name this server (DNS-rebinding guard);
* documents are served ``no-store`` with a Content-Security-Policy that keeps
  book content on this origin (``connect-src``/``img-src`` 'self' and
  ``data:``) and runs only the page's own scripts: ``script-src 'nonce-...'``
  with a fresh nonce per published page (``publish(html, script_nonce=)``;
  the Reader strips book scripts and handlers before it stamps the nonce on
  the shell and its extras). A page published without a nonce runs no script;
* events are accepted only as ``POST`` with a JSON body (what the page's
  ``fetch`` sends), never from a URL a book could embed (``<img src=...>``,
  CSS ``url()``, which would carry the cookie); a form cannot post either
  (``form-action 'none'``);
* ``/img/<id>`` serves only bytes that are an image by content (the shared
  ``library_covers`` signatures; the sniffed type is the one sent), with a
  sandboxing CSP of its own, so a file that is not an image never leaves the
  app through the reader.

Nothing is read from disk by path: documents are stored in memory (the last
few versions per name) and images are served only through ids registered by
the Reader (``register_image``), each backed by bytes, a file the Reader chose
or a loader callable (EPUB images are read lazily from the zip).

Pure standard library, no Flet import; Python 3.10 compatible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qs, quote, unquote, urlsplit

__all__ = [
    "COOKIE_NAME",
    "DOCUMENT_CSP",
    "EVENT_PATH",
    "IMAGE_CSP",
    "MAX_DRAIN_BYTES",
    "MAX_EVENT_BYTES",
    "ReaderServer",
    "TOKEN_HEADER",
    "document_csp",
    "image_id_for",
    "sniff_font_mime",
    "sniff_image_mime",
]

log = logging.getLogger("glossarion.reader.server")

COOKIE_NAME = "glrdr"
TOKEN_HEADER = "X-GLRDR-Token"
EVENT_PATH = "__ev"
MAX_EVENT_BYTES = 64 * 1024
#: An oversized event body up to this size is read and dropped before the 413: closing the socket
#: with request bytes still unread sends a TCP reset, which on Windows discards the response the
#: client has not read yet (it sees ConnectionResetError / ConnectionAbortedError, not the 413).
MAX_DRAIN_BYTES = 1024 * 1024
DOC_VERSIONS_KEPT = 4  # per document name (a slow WebView may still ask for the previous version)
MAX_IMAGES = 20000

#: Content-Security-Policy of reader documents: book HTML may only talk to this origin, and only
#: the page's own scripts (the ``script_nonce`` they carry) run.
_DOCUMENT_CSP_TEMPLATE = (
    "default-src 'self' data: blob:; "
    "{script}; "
    "style-src 'self' 'unsafe-inline' data:; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "media-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'none'"
)
#: A document published without a nonce runs no script at all.
DOCUMENT_CSP = _DOCUMENT_CSP_TEMPLATE.format(script="script-src 'none'")
#: Image responses: never a document that could run anything (an SVG opened directly).
IMAGE_CSP = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox"
_NONCE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")

#: Content type by the shared ``library_covers._IMAGE_MAGIC`` signature (first 4 bytes at most).
_MAGIC_MIME = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"BM": "image/bmp",
    b"\x00\x00\x01\x00": "image/x-icon",
    b"II*\x00": "image/tiff",
    b"MM\x00*": "image/tiff",
}


def document_csp(nonce: Optional[str] = None) -> str:
    """The CSP of a published page: ``script-src 'nonce-<nonce>'`` (else no script)."""
    if nonce:
        return _DOCUMENT_CSP_TEMPLATE.format(script=f"script-src 'nonce-{nonce}'")
    return DOCUMENT_CSP


def sniff_image_mime(data: Any) -> Optional[str]:
    """The image type of ``data`` by content, or None when it is not an image.

    The shared header signatures of ``library_covers`` (``_IMAGE_MAGIC``, ``_looks_like_svg``);
    WebP / AVIF are RIFF / ISO-BMFF containers."""
    head = bytes(data[:4096]) if data else b""
    if not head:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:12] in (b"ftypavif", b"ftypavis"):
        return "image/avif"
    try:
        from library_covers import _IMAGE_MAGIC, _looks_like_svg
    except Exception:  # pragma: no cover - the bundle always carries library_covers
        log.warning("library_covers unavailable: reader images are not served")
        return None
    for magic in _IMAGE_MAGIC:
        if head.startswith(magic):
            return _MAGIC_MIME.get(magic[:4])
    if _looks_like_svg(head):
        return "image/svg+xml"
    return None

#: Font files the Reader serves (the Converter's "Load Font…" folder), by signature.
_FONT_MAGIC = (
    (b"\x00\x01\x00\x00", "font/ttf"),
    (b"true", "font/ttf"),
    (b"OTTO", "font/otf"),
    (b"ttcf", "font/collection"),
    (b"wOFF", "font/woff"),
    (b"wOF2", "font/woff2"),
)
MAX_FONT_BYTES = 64 * 1024 * 1024


def sniff_font_mime(data: Any) -> Optional[str]:
    """The font type of ``data`` by its signature, or None when it is not a font file."""
    head = bytes(data[:4]) if data else b""
    for magic, mime in _FONT_MAGIC:
        if head == magic:
            return mime
    return None


ImageSource = Union[bytes, str, Callable[[], Optional[bytes]]]
EventCallback = Callable[[dict], Any]


def image_id_for(key: str) -> str:
    """Opaque, stable id for an image key (an EPUB member name or a file path) plus its extension."""
    text = str(key or "")
    ext = os.path.splitext(text.split("?", 1)[0].split("#", 1)[0])[1].lower()
    if not ext or len(ext) > 6 or not ext[1:].isalnum():
        ext = ""
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:20] + ext


@dataclass
class _Image:
    key: str
    source: ImageSource
    mime: str


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, owner: "ReaderServer") -> None:
        self.owner = owner
        super().__init__((owner.host, 0), _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """The WebView dropping a keep-alive connection (a chapter turn, the page closed) is routine:
        a debug line, not ``socketserver``'s traceback on stderr. Anything else is logged."""
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            log.debug("reader http: %s from %s", type(error).__name__, client_address)
            return
        log.exception("reader http: request from %s failed", client_address)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"
    server_version = "GlossarionReader"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler API
        log.debug("reader http: " + format, *args)

    # ---- helpers -------------------------------------------------------------------------

    def _send(self, status: int, body: bytes = b"", content_type: str = "text/plain; charset=utf-8",
              headers: Optional[dict] = None) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _not_found(self) -> None:
        self._send(HTTPStatus.NOT_FOUND, b"not found")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_address[1]
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _cookie_token(self) -> str:
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE_NAME:
                return value.strip()
        return ""

    # ---- verbs -----------------------------------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch(body=None)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_EVENT_BYTES:
            self.close_connection = True
            if 0 < length <= MAX_DRAIN_BYTES:
                self._drain(length)
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"too large")
            return
        body = self.rfile.read(length) if length else b""
        self._dispatch(body=body)

    def _drain(self, length: int) -> None:
        """Read and drop ``length`` request bytes (see ``MAX_DRAIN_BYTES``); a client that sends
        less is given a few seconds, then the connection closes anyway."""
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

    def do_OPTIONS(self) -> None:  # noqa: N802 - no CORS: the page and the server share one origin
        self._not_found()

    def _dispatch(self, body: Optional[bytes]) -> None:
        owner = self.server.owner
        if not self._host_ok():
            self._not_found()
            return
        parts = urlsplit(self.path)
        segments = [unquote(s) for s in parts.path.split("/") if s]
        query = parse_qs(parts.query, keep_blank_values=True)
        if not segments:
            self._not_found()
            return
        # /__ev with the token in a cookie, header or query value
        if segments == [EVENT_PATH]:
            token = self._cookie_token() or (self.headers.get(TOKEN_HEADER) or "").strip() or (query.get("t") or [""])[0]
            if not owner.check_token(token):
                self._not_found()
                return
            self._event(body)
            return
        if not owner.check_token(segments[0]):
            self._not_found()
            return
        rest = segments[1:]
        if rest == [EVENT_PATH]:
            self._event(body)
            return
        if body is not None:  # POST only for events
            self._not_found()
            return
        if len(rest) == 1 and rest[0].endswith(".html"):
            entry = owner.document_entry(rest[0], (query.get("v") or [""])[0])
            if entry is None:
                self._not_found()
                return
            doc, nonce = entry
            self._send(HTTPStatus.OK, doc, "text/html; charset=utf-8", {
                "Cache-Control": "no-store",
                "Content-Security-Policy": document_csp(nonce),
                "Set-Cookie": f"{COOKIE_NAME}={owner.token}; Path=/; HttpOnly; SameSite=Strict",
            })
            return
        if len(rest) == 2 and rest[0] == "font":
            found = owner.font(rest[1])
            if found is None:
                self._not_found()
                return
            data, mime = found
            self._send(HTTPStatus.OK, data, mime, {"Cache-Control": "private, max-age=3600",
                                                   "Content-Security-Policy": IMAGE_CSP})
            return
        if len(rest) == 2 and rest[0] == "img":
            found = owner.image(rest[1])
            if found is None:
                self._not_found()
                return
            data, mime = found
            self._send(HTTPStatus.OK, data, mime, {"Cache-Control": "private, max-age=3600",
                                                   "Content-Security-Policy": IMAGE_CSP})
            return
        self._not_found()

    def _event(self, body: Optional[bytes]) -> None:
        """An event from the page's ``fetch``: POST + JSON only. A GET (an ``<img>`` / CSS URL a
        book could embed carries the cookie) or a form post is refused, never delivered."""
        owner = self.server.owner
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if body is None or content_type != "application/json":
            self._not_found()
            return
        payload: Any = None
        try:
            if body:
                payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            owner.deliver_event(payload, channel="http")
        self._send(HTTPStatus.NO_CONTENT, b"", headers={"Cache-Control": "no-store"})


class ReaderServer:
    """Localhost server for the Reader document, its images and the event fallback.

    ``start()`` binds ``127.0.0.1:0``; ``publish(html)`` returns the document URL;
    ``register_image(key, source)`` returns an image URL; ``on_event`` receives
    every event dict posted to ``/__ev`` (on the server thread: post to the UI loop).
    """

    def __init__(self, *, on_event: Optional[EventCallback] = None, host: str = "127.0.0.1",
                 token: Optional[str] = None) -> None:
        if host not in ("127.0.0.1", "::1"):
            raise ValueError("the reader server only binds to the loopback interface")
        self.host = host
        self.token = token or secrets.token_urlsafe(24)
        self.on_event = on_event
        self._lock = threading.Lock()
        self._docs: dict[str, OrderedDict] = {}
        self._doc_serial = 0
        self._images: "OrderedDict[str, _Image]" = OrderedDict()
        self._fonts: dict = {}  # font id -> file path (Aa › Text imported fonts)
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        self.events_received = 0

    # ---- lifecycle -----------------------------------------------------------------------

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
    def base_path(self) -> str:
        return f"/{self.token}/"

    @property
    def base_url(self) -> str:
        return self.origin + self.base_path

    @property
    def event_path(self) -> str:
        """Same-origin event URL with the token in the path (the page may also use ``/__ev``)."""
        return f"{self.base_path}{EVENT_PATH}"

    def start(self) -> int:
        with self._lock:
            if self._server is not None:
                return self.port
            server = _Server(self)
            self._server = server
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.25},
                                      name="gl-reader-http", daemon=True)
            self._thread = thread
        thread.start()
        log.info("reader server on 127.0.0.1:%d", self.port)
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
            log.debug("stopping the reader server failed", exc_info=True)
        if thread is not None:
            thread.join(timeout=2.0)

    def check_token(self, value: Any) -> bool:
        text = str(value or "")
        return bool(text) and hmac.compare_digest(text.encode("utf-8", "replace"), self.token.encode("ascii"))

    # ---- documents ---------------------------------------------------------------------------

    def publish(self, html: str, *, name: str = "reader.html", script_nonce: Optional[str] = None) -> str:
        """Store ``html`` as a new version of document ``name``; returns its absolute URL.

        ``script_nonce``: the nonce the page's own ``<script>`` elements carry (its CSP allows
        those only); without one the page runs no script."""
        if not name.endswith(".html") or "/" in name or "\\" in name:
            raise ValueError(f"invalid document name {name!r}")
        if script_nonce is not None and (not script_nonce or not set(script_nonce) <= _NONCE_CHARS):
            raise ValueError("invalid script nonce")
        data = html.encode("utf-8") if isinstance(html, str) else bytes(html)
        with self._lock:
            self._doc_serial += 1
            version = str(self._doc_serial)
            versions = self._docs.setdefault(name, OrderedDict())
            versions[version] = (data, script_nonce)
            while len(versions) > DOC_VERSIONS_KEPT:
                versions.popitem(last=False)
        return f"{self.base_url}{quote(name)}?v={version}"

    def document_entry(self, name: str, version: str = "") -> Optional[tuple]:
        """``(bytes, script nonce)`` of a published version (an unknown version: the latest)."""
        with self._lock:
            versions = self._docs.get(name)
            if not versions:
                return None
            if version and version in versions:
                return versions[version]
            return next(reversed(versions.values()))

    def document(self, name: str, version: str = "") -> Optional[bytes]:
        entry = self.document_entry(name, version)
        return entry[0] if entry is not None else None

    @property
    def documents_published(self) -> int:
        return self._doc_serial

    # ---- images -------------------------------------------------------------------------------

    def register_image(self, key: str, source: ImageSource, mime: Optional[str] = None) -> str:
        """Register an image (bytes, a file path chosen by the Reader, or a loader); returns its URL path.

        The returned value is a root-relative URL (``/<token>/img/<id>``): reader
        documents are served by this server, so it resolves on the same origin.
        """
        image_id = image_id_for(key)
        if mime is None:
            guessed = mimetypes.guess_type("x" + os.path.splitext(image_id)[1])[0]
            mime = guessed or "application/octet-stream"
        with self._lock:
            self._images[image_id] = _Image(str(key), source, mime)
            self._images.move_to_end(image_id)
            while len(self._images) > MAX_IMAGES:
                self._images.popitem(last=False)
        return f"{self.base_path}img/{image_id}"

    def image(self, image_id: str) -> Optional[tuple[bytes, str]]:
        with self._lock:
            entry = self._images.get(image_id)
        if entry is None:
            return None
        source = entry.source
        try:
            if isinstance(source, (bytes, bytearray)):
                data: Optional[bytes] = bytes(source)
            elif isinstance(source, str):
                with open(source, "rb") as handle:
                    data = handle.read()
            else:
                data = source()
        except Exception as exc:
            log.info("reader image %s unavailable: %s", entry.key[-80:], exc)
            return None
        if data is None:
            return None
        data = bytes(data)
        mime = sniff_image_mime(data)
        if mime is None:
            # Not an image by content (e.g. a book <img src="../../..."> that resolved to an
            # app-private file): never served.
            log.warning("reader image %s refused: not an image", entry.key[-80:])
            return None
        return data, mime

    def register_font(self, path: str) -> str:
        """Serve a font file chosen by the Reader (the "Load Font…" folder); returns its URL path."""
        font_id = image_id_for("font:" + os.path.abspath(str(path)))
        with self._lock:
            self._fonts[font_id] = str(path)
        return f"{self.base_path}font/{font_id}"

    def font(self, font_id: str) -> Optional[tuple]:
        with self._lock:
            path = self._fonts.get(font_id)
        if not path:
            return None
        try:
            if os.path.getsize(path) > MAX_FONT_BYTES:
                return None
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError as exc:
            log.info("reader font unavailable: %s", exc)
            return None
        mime = sniff_font_mime(data)
        if mime is None:
            log.warning("reader font %s refused: not a font file", os.path.basename(path))
            return None
        return data, mime

    def clear_images(self) -> None:
        with self._lock:
            self._images.clear()

    @property
    def image_count(self) -> int:
        with self._lock:
            return len(self._images)

    # ---- events --------------------------------------------------------------------------------

    def deliver_event(self, payload: dict, *, channel: str = "http") -> None:
        self.events_received += 1
        callback = self.on_event
        if callback is None:
            return
        try:
            callback(dict(payload, _channel=channel))
        except Exception:
            log.exception("reader event handler failed")
