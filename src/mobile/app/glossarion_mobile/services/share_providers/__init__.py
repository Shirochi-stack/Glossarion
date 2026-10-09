"""Share-link providers (U10 "Share file via link"): shared types, consent text and a small HTTP client.

Every provider is opt-in, tap-only and off until the user enables it in Settings and accepts its
consent sheet (``services.share_links`` enforces that; nothing here uploads on its own):

* ``transferit_handoff`` - transfer.it is a **browser handoff only**: the app makes the file
  reachable (Android: Downloads/Glossarion; iOS: its Files path), opens https://transfer.it/start in
  the in-app browser and lets the user paste the link back. transfer.it's private API is never
  called (MEGA's Terms "Our IP": API use needs developer registration and approval);
* ``gofile`` - Gofile's documented API with one reused anonymous guest account;
* ``send_e2ee`` - Send (send.vis.ee, the timvisee/send protocol): encrypted on the phone, the key
  only in the link's ``#`` fragment;
* ``pixeldrain`` - pixeldrain's documented API with the user's own API key.

Retries follow the U10 critic: no automatic retry, except Gofile's HTTP 429 with a Retry-After of
at most a minute, retried once. A read timeout after the whole body was sent is never retried: the
upload may have completed, so the user decides (``ShareError.maybe_uploaded``).

The HTTP client is the standard library's ``http.client``, on purpose: the backend's optional
``http_logger`` patches ``requests`` and ``httpx`` and would write response bodies (Gofile's guest
token) into ``<logs>/http_requests``. Plain ``http://`` is refused except to a loopback address
(the host tests' fake servers). Pure Python 3.10, no Flet import; blocking (worker threads only).
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import os
import socket
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence
from urllib.parse import quote, urlsplit

__all__ = [
    "CancelToken",
    "ConsentText",
    "abort_socket",
    "HttpClient",
    "HttpResponse",
    "MAX_RETRY_AFTER",
    "MAYBE_CANCELLED",
    "PROVIDER_IDS",
    "ProviderInfo",
    "Progress",
    "ShareError",
    "UploadResult",
    "consent_text",
    "format_size",
    "is_loopback_host",
    "multipart_file_body",
    "parse_retry_after",
    "provider_info",
    "user_agent",
]

log = logging.getLogger("glossarion.share")

#: Menu order (UI_SPEC U10): the handoff first (no third-party API), then the direct providers.
PROVIDER_IDS = ("transferit", "gofile", "send", "pixeldrain")

#: Gofile 429: wait for Retry-After (and retry once) only when it is at most this long (seconds).
MAX_RETRY_AFTER = 60.0
CHUNK = 256 * 1024
CONNECT_TIMEOUT = 15.0
IO_TIMEOUT = 60.0
MAX_RESPONSE_BYTES = 1024 * 1024
#: Cancel tapped while the service was answering, after the whole file was sent (``maybe_uploaded``).
MAYBE_CANCELLED = ("Stopped waiting after the whole file was sent: it may already be on the service without a link "
                   "coming back. Uploading again may leave a second copy there.")

Progress = Callable[[int, int], Any]  # (bytes of the file sent, file size)


# ---------------------------------------------------------------------------
# errors and cancellation
# ---------------------------------------------------------------------------


class ShareError(Exception):
    """A share-link failure with a user-facing ``message`` (never a path, URL, key or token).

    ``code``: ``disabled`` · ``consent`` · ``no_key`` · ``secrets_unavailable`` · ``missing_file`` ·
    ``not_allowed`` · ``too_large`` · ``empty`` · ``busy`` · ``cancelled`` · ``network`` ·
    ``maybe_uploaded`` (timed out after the whole file was sent) · ``rate_limited`` · ``auth`` ·
    ``limits`` · ``server`` · ``protocol`` · ``bad_link`` · ``unsupported`` · ``not_found``.
    """

    def __init__(self, code: str, message: str, *, retry_after: Optional[float] = None,
                 status: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after = retry_after
        self.status = status

    @property
    def maybe_uploaded(self) -> bool:
        return self.code == "maybe_uploaded"

    @property
    def retryable(self) -> bool:
        """Whether a Retry button makes sense (the user taps it; nothing retries by itself)."""
        return self.code in ("network", "maybe_uploaded", "rate_limited", "server", "busy")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ShareError({self.code!r}, {self.message!r})"


def abort_socket(sock: Any) -> None:
    """Unblock a thread stuck in ``send``/``recv`` on ``sock`` (Cancel): ``shutdown`` does it on Linux,
    Android and iOS; Windows (desktop dev, host tests) only wakes the thread when the socket is closed."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    if os.name == "nt":
        try:
            sock.close()
        except OSError:
            pass


class CancelToken:
    """Thread-safe cancel flag. ``on_cancel`` callbacks abort a blocking socket (``abort_socket``)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks, self._callbacks = list(self._callbacks), []
        for callback in callbacks:
            try:
                callback()
            except Exception:  # pragma: no cover - best effort
                log.debug("cancel callback failed", exc_info=True)

    def on_cancel(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """Run ``callback`` when cancelled (at once when already cancelled); returns a remover."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)

                def remove() -> None:
                    with self._lock:
                        try:
                            self._callbacks.remove(callback)
                        except ValueError:
                            pass

                return remove
        callback()
        return lambda: None

    def check(self) -> None:
        if self._event.is_set():
            raise ShareError("cancelled", "Upload cancelled")

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; True when cancelled meanwhile."""
        return self._event.wait(max(0.0, float(seconds)))


# ---------------------------------------------------------------------------
# provider description and consent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str  # the name the UI shows
    operator: str
    country: str
    site: str  # https://… (shown, opened only when the user taps it)
    e2ee: bool  # end-to-end encrypted: the service cannot read the file
    direct: bool  # False: a browser handoff (Glossarion makes no request)
    needs_api_key: bool
    can_delete: bool  # the app can delete the upload later
    max_bytes: Optional[int]  # None: no stated limit
    retention: str
    at_rest_note: str = ""  # only where the operator states it
    terms_url: str = ""
    consent_version: int = 1

    @property
    def readable_by_service(self) -> bool:
        return not self.e2ee


GIB = 1024 ** 3

_INFO = {
    "transferit": ProviderInfo(
        id="transferit", label="transfer.it", operator="Mega Privacy (NZ) Limited, a MEGA group company",
        country="New Zealand", site="https://transfer.it", e2ee=False, direct=False, needs_api_key=False,
        can_delete=False, max_bytes=None,
        retention="Free transfers last up to 90 days. Delete them on transfer.it itself; Glossarion only keeps the link.",
        at_rest_note="transfer.it encrypts files on its servers but holds the keys (it is not zero-knowledge).",
        terms_url="https://transfer.it",
    ),
    "gofile": ProviderInfo(
        id="gofile", label="Gofile", operator="WOJTEK SAS", country="France", site="https://gofile.io",
        e2ee=False, direct=True, needs_api_key=False, can_delete=True, max_bytes=None,
        retention="Kept while people download it; Gofile removes it after about 10 days without downloads. "
                  "You can delete it from the Book page.",
        at_rest_note="Gofile says files are encrypted at rest on its servers.",
        terms_url="https://gofile.io/terms",
    ),
    "send": ProviderInfo(
        id="send", label="Send (send.vis.ee)", operator="a volunteer (a hobby instance of the open-source Send project)",
        country="", site="https://send.vis.ee", e2ee=True, direct=True, needs_api_key=False, can_delete=True,
        max_bytes=int(2.5 * GIB),
        retention="Expires after at most 3 days or 20 downloads, whichever comes first. You can delete it earlier.",
        terms_url="https://github.com/timvisee/send",
    ),
    "pixeldrain": ProviderInfo(
        id="pixeldrain", label="pixeldrain", operator="Fornaxian Technologies", country="",
        site="https://pixeldrain.com", e2ee=False, direct=True, needs_api_key=True, can_delete=True,
        max_bytes=10 * 1000 ** 3,
        retention="Removed 60 days after the last download. You can delete it from the Book page.",
        terms_url="https://pixeldrain.com/about",
    ),
}


def provider_info(provider_id: str) -> ProviderInfo:
    try:
        return _INFO[str(provider_id)]
    except KeyError:
        raise ShareError("unsupported", "Unknown sharing service") from None


def format_size(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover


@dataclass(frozen=True)
class ConsentText:
    provider: str
    version: int
    title: str
    lines: tuple  # paragraphs, in order
    checkbox: str  # the required confirmation
    confirm: str  # the button
    terms_url: str


def consent_text(provider_id: str, *, file_name: str = "", size: Optional[int] = None) -> ConsentText:
    """The consent sheet for a provider (neutral wording; says that the file leaves the phone, who can
    read it, what the service sees, how long it lasts and that it must be the user's to share)."""
    info = provider_info(provider_id)
    what = f"“{file_name}”" if file_name else "the file you pick"
    if size:
        what += f" ({format_size(size)})"
    run_by = (f" {info.label} is run by {info.operator}" + (f" ({info.country})." if info.country else ".")
              if info.operator else "")
    lines = []
    if info.direct:
        lines.append(f"Share file via link sends {what} from this phone to {info.label}, which gives you a link."
                     f"{run_by} Glossarion normally keeps everything on your phone.")
    else:
        lines.append(f"Share file via link opens {info.label} in the browser.{run_by} You pick {what} there and "
                     f"upload it yourself; Glossarion sends nothing to {info.label}.")
    lines.append("Anyone who has the link can download the file, and links can be forwarded. Share it only with "
                 "people you trust.")
    if info.e2ee:
        lines.append(f"End-to-end encrypted: the file is encrypted on this phone and the key is only in the link "
                     f"(after the #). {info.label} cannot read the file.")
    else:
        lines.append(f"Not end-to-end encrypted: {info.label} can read the file."
                     + (f" {info.at_rest_note}" if info.at_rest_note else ""))
    lines.append(f"{info.label} sees your IP address and when you upload. Glossarion's developers receive nothing: "
                 "the file goes straight from your phone to the service.")
    lines.append(info.retention)
    if info.needs_api_key:
        lines.append(f"Uploads use your own {info.label} account (your API key, stored encrypted on this phone).")
    lines.append(f"Only share files you have the right to share. {info.label} removes files reported as "
                 "infringing and can block your access.")
    if info.direct:
        lines.append("Uploading uses mobile data when you are not on Wi-Fi.")
    return ConsentText(
        provider=info.id, version=info.consent_version, title=f"Share file via link · {info.label}",
        lines=tuple(lines), checkbox="I have the right to share the files I upload",
        confirm=f"Use {info.label}" if info.direct else f"Open {info.label}", terms_url=info.terms_url,
    )


@dataclass
class UploadResult:
    url: str  # the share link (Send: with the #key)
    remote_id: str = ""
    delete_handle: Optional[dict] = None  # provider data needed to delete it (stored encrypted)
    expires: Optional[float] = None  # epoch seconds when known
    downloads_limit: Optional[int] = None
    account_token: Optional[str] = None  # Gofile: the guest token this upload created (store it)
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP (stdlib, streaming, cancellable)
# ---------------------------------------------------------------------------


def user_agent() -> str:
    try:
        import app_version  # shared (lightweight)

        version = str(getattr(app_version, "APP_VERSION", "") or "")
    except Exception:
        version = ""
    return f"Glossarion-Mobile/{version or '0'}"


def is_loopback_host(host: Any) -> bool:
    text = str(host or "").strip("[]").lower()
    return text in ("localhost", "::1") or text.startswith("127.")


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or None
    if not cafile:
        try:
            import certifi

            cafile = certifi.where()
        except Exception:
            cafile = None
    try:
        return ssl.create_default_context(cafile=cafile)
    except (OSError, ssl.SSLError):
        return ssl.create_default_context()


def parse_retry_after(value: Any, *, now: Optional[float] = None) -> Optional[float]:
    """Seconds from a Retry-After header (delta-seconds or an HTTP date), else None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - (time.time() if now is None else now)
    if seconds != seconds:  # NaN
        return None
    return max(0.0, seconds)


def _early_response(conn: Any) -> Optional[tuple]:
    """``(response, body)`` of an error the server sent before it stopped reading the upload (a wrong key,
    a file too large: it answers and closes while the body is still being sent), else None."""
    sock = getattr(conn, "sock", None)
    if sock is None:
        return None
    try:
        sock.settimeout(5.0)
        response = conn.getresponse()
        if response.status < 400:
            return None
        return response, response.read(MAX_RESPONSE_BYTES + 1)
    except Exception:
        return None


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict  # lower-case names
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None


class HttpClient:
    """One request per connection over ``http.client``; the body may be an iterator of chunks."""

    def __init__(self, *, connect_timeout: float = CONNECT_TIMEOUT, io_timeout: float = IO_TIMEOUT,
                 agent: Optional[str] = None, context_factory: Callable[[], ssl.SSLContext] = _ssl_context) -> None:
        self.connect_timeout = float(connect_timeout)
        self.io_timeout = float(io_timeout)
        self.agent = agent or user_agent()
        self._context_factory = context_factory
        self._context: Optional[ssl.SSLContext] = None

    def _connection(self, url: str) -> tuple:
        parts = urlsplit(url)
        scheme = (parts.scheme or "").lower()
        host = parts.hostname or ""
        if scheme == "https":
            if self._context is None:
                self._context = self._context_factory()
            conn = http.client.HTTPSConnection(host, parts.port or 443, timeout=self.connect_timeout,
                                               context=self._context)
        elif scheme == "http" and is_loopback_host(host):
            conn = http.client.HTTPConnection(host, parts.port or 80, timeout=self.connect_timeout)
        else:
            raise ShareError("protocol", "Only secure (https) connections are allowed")
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        return conn, path, host

    def request(self, method: str, url: str, *, headers: Optional[Mapping[str, str]] = None,
                body: Any = None, length: Optional[int] = None, cancel: Optional[CancelToken] = None) -> HttpResponse:
        """Send one request. ``body``: None, bytes, or an iterable of byte chunks (``length`` required).

        Raises ``ShareError``: ``cancelled``; ``network`` when the connection failed before the whole
        body was sent; ``maybe_uploaded`` when it timed out or broke after that.
        """
        cancel = cancel or CancelToken()
        cancel.check()
        conn, path, host = self._connection(url)
        chunks: Iterable[bytes]
        if body is None:
            chunks, length = (), 0 if method.upper() in ("POST", "PUT", "PATCH", "DELETE") else None
        elif isinstance(body, (bytes, bytearray)):
            chunks, length = (bytes(body),), len(body)
        else:
            if length is None:
                raise ValueError("length is required for a streamed body")
            chunks = body
        sent_all = False

        remove = cancel.on_cancel(lambda: abort_socket(getattr(conn, "sock", None)))
        try:
            try:
                conn.connect()
                if conn.sock is not None:
                    conn.sock.settimeout(self.io_timeout)
                conn.putrequest(method.upper(), path, skip_accept_encoding=True)
                conn.putheader("User-Agent", self.agent)
                conn.putheader("Accept", "application/json")
                for name, value in (headers or {}).items():
                    conn.putheader(name, value)
                if length is not None:
                    conn.putheader("Content-Length", str(int(length)))
                conn.endheaders()
                for chunk in chunks:
                    cancel.check()
                    if chunk:
                        conn.send(chunk)
                cancel.check()
                sent_all = True
                response = conn.getresponse()
                data = response.read(MAX_RESPONSE_BYTES + 1)
            except ShareError:
                raise
            except (socket.timeout, TimeoutError) as exc:
                if sent_all:  # the whole file reached the service: never "cancelled" (it may be stored there)
                    raise ShareError("maybe_uploaded", MAYBE_CANCELLED if cancel.cancelled else
                                     "The service did not answer in time. The file may have been uploaded "
                                     "without a link coming back; uploading again may leave a second copy "
                                     "there.") from exc
                cancel.check()
                raise ShareError("network", f"The connection to {host} timed out") from exc
            except (OSError, http.client.HTTPException) as exc:
                if sent_all:
                    raise ShareError("maybe_uploaded", MAYBE_CANCELLED if cancel.cancelled else
                                     "The connection broke after the file was sent. The file may have been "
                                     "uploaded without a link coming back; uploading again may leave a second "
                                     "copy there.") from exc
                cancel.check()
                early = _early_response(conn)
                if early is None:
                    raise ShareError("network", f"Could not reach {host} ({type(exc).__name__})") from exc
                response, data = early
        finally:
            remove()
            close_body = getattr(chunks, "close", None)
            if callable(close_body):  # a generator left mid-file closes its file now (Windows cannot delete it)
                try:
                    close_body()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass
        if len(data) > MAX_RESPONSE_BYTES:
            raise ShareError("protocol", "The service sent an unexpectedly large answer", status=response.status)
        return HttpResponse(response.status, {k.lower(): v for k, v in response.getheaders()}, data)


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


def file_chunks(path: str, size: int, *, progress: Optional[Progress] = None,
                cancel: Optional[CancelToken] = None, chunk_size: int = CHUNK) -> Iterator[bytes]:
    """The file's bytes in chunks, reporting ``progress(sent, size)``; stops at ``size`` bytes."""
    sent = 0
    if progress is not None:
        progress(0, size)
    with open(path, "rb") as handle:
        while sent < size:
            if cancel is not None:
                cancel.check()
            data = handle.read(min(chunk_size, size - sent))
            if not data:
                raise ShareError("missing_file", "The file got shorter while it was being uploaded")
            sent += len(data)
            yield data
            if progress is not None:
                progress(sent, size)


def _form_quote(text: str) -> str:
    """WHATWG multipart/form-data name escaping (what browsers send): raw UTF-8, ``"``, CR, LF escaped."""
    return str(text).replace('"', "%22").replace("\r", "%0D").replace("\n", "%0A")


def multipart_file_body(field_name: str, file_name: str, mime: str, path: str, size: int, *,
                        fields: Sequence[tuple] = (), progress: Optional[Progress] = None,
                        cancel: Optional[CancelToken] = None) -> tuple:
    """``(content_type, length, make_iterator)`` for a multipart/form-data upload of one file.

    ``make_iterator()`` builds a fresh chunk iterator (a retry re-reads the file).
    """
    boundary = "----GlossarionFormBoundary" + uuid.uuid4().hex
    head = bytearray()
    for name, value in fields:
        head += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{_form_quote(name)}\"\r\n\r\n"
                 f"{value}\r\n").encode("utf-8")
    head += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{_form_quote(field_name)}\"; "
             f"filename=\"{_form_quote(file_name)}\"\r\nContent-Type: {mime}\r\n\r\n").encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    length = len(head) + int(size) + len(tail)

    def make() -> Iterator[bytes]:
        yield bytes(head)
        yield from file_chunks(path, int(size), progress=progress, cancel=cancel)
        yield tail

    return f"multipart/form-data; boundary={boundary}", length, make


def basic_auth(user: str, password: str) -> str:
    raw = f"{user}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def quote_path_name(name: str) -> str:
    return quote(str(name), safe="")
