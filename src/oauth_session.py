"""Shared pieces of the begin/complete OAuth split used by the auth*_auth modules.

U1 split ``authgpt_auth.run_oauth_flow()`` into ``begin_oauth()`` /
``complete_from_redirect()`` for Glossarion Mobile: the app opens the browser
itself (Custom Tabs / SFSafariViewController) and offers a "paste the redirect
URL" fallback. U4 gives ``authgem_auth``, ``authcd_auth`` and ``authgrok_auth``
the same split. The provider-independent parts live here, written to the
behaviour of authgpt's U1 code:

* :class:`LoopbackOAuthSession` - the loopback callback listener(s) started by a
  ``begin_oauth()`` and the redirect they captured (authgpt's ``OAuthSession``
  with a ``provider`` name, provider-specific ``extra`` pending fields and a
  caller-driven :meth:`~LoopbackOAuthSession.handle_request` mode);
* :class:`PendingOAuthMixin` - the encrypted "pending sign-in" file kept next to
  a token file, so a pasted redirect still completes after the listener or the
  app was killed (authgpt's ``AuthGPTTokenStore.*_pending_oauth``);
* :func:`parse_oauth_redirect` - ``(code, state, error)`` from a pasted redirect
  URL, its query string, ``code#state`` or a bare code;
* :func:`resolve_completion` / :func:`finish_completion` - the common start and
  end of every ``complete_from_redirect()``;
* :func:`oauth_success_html` - the ``GLOSSARION_OAUTH_RETURN_URL`` seam of the
  loopback success pages;
* :func:`make_mobile_loopback_server` / :func:`send_page_quietly` - Glossarion
  Mobile's callback listeners (a thread per connection, a read timeout) and
  callback answers that tolerate a browser tab closed in the meantime;
* :func:`default_token_dir` - ``~/.glossarion``, or ``GLOSSARION_TOKEN_DIR``;
* :func:`is_mobile`.

Desktop sets neither ``GLOSSARION_OAUTH_RETURN_URL`` nor ``GLOSSARION_TOKEN_DIR``
and never asks for the mobile listeners, so its success pages, listeners and token
paths are unchanged.

Stdlib only (``token_encryption`` is imported lazily); Python 3.10; no Qt.
"""
import logging
import os
import socket
import sys
import threading
import time
from html import escape as _html_escape
from http.server import HTTPServer, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

__all__ = [
    "PENDING_OAUTH_MAX_AGE_SECONDS",
    "Completion",
    "LoopbackOAuthSession",
    "PendingOAuthMixin",
    "default_token_dir",
    "finish_completion",
    "is_mobile",
    "make_ipv6_loopback_server",
    "make_mobile_loopback_server",
    "oauth_return_page",
    "oauth_return_target",
    "oauth_return_url",
    "oauth_success_html",
    "parse_oauth_redirect",
    "resolve_completion",
    "send_page_quietly",
    "session_of",
]

PENDING_OAUTH_MAX_AGE_SECONDS = 3600  # saved PKCE/state for the paste fallback
#: Glossarion Mobile listeners: seconds a connection may take to send its request line.
MOBILE_REQUEST_TIMEOUT = 15.0


# ===========================================================================
# Runtime / paths
# ===========================================================================

def is_mobile() -> bool:
    """mobile_runtime.is_mobile(); False when the shared module is unavailable."""
    if sys.platform in ("ios", "android"):
        return True
    try:
        import mobile_runtime
        return mobile_runtime.is_mobile()
    except Exception:
        return False


def default_token_dir() -> str:
    """Folder of the auth token files: ``GLOSSARION_TOKEN_DIR``, else ``~/.glossarion``.

    Glossarion Mobile sets ``GLOSSARION_TOKEN_DIR`` to ``<data>/home/.glossarion``
    before any backend import: on a Windows dev run ``os.path.expanduser('~')``
    ignores the ``HOME`` override and would point at the real user profile (the
    desktop app's tokens). Desktop never sets it.
    """
    override = os.environ.get("GLOSSARION_TOKEN_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".glossarion")


# ===========================================================================
# Success-page seam (GLOSSARION_OAUTH_RETURN_URL)
# ===========================================================================

def oauth_return_url() -> str:
    """``GLOSSARION_OAUTH_RETURN_URL`` (Glossarion Mobile's return deep link), or ''."""
    return os.environ.get("GLOSSARION_OAUTH_RETURN_URL", "").strip()


def oauth_return_target(return_url: str, provider: str) -> str:
    """'<return_url>?p=<provider>' ('&p=<provider>' when it already has a query)."""
    separator = "&" if "?" in return_url else "?"
    return f"{return_url}{separator}p={provider}"


def oauth_return_page(return_url: str, provider: str, heading_html: str = "&#10004; Authenticated!") -> str:
    """The success page that sends the browser back to the app.

    A button plus an automatic redirect to '<return_url>?p=<provider>' (the
    page authgpt_auth serves when GLOSSARION_OAUTH_RETURN_URL is set). Nothing
    depends on that custom-scheme redirect: the app already has the code from
    the request this page answers and finishes the sign-in on its own, and the
    page says the tab can simply be closed (Android Custom Tabs may refuse a
    script-started jump to another app).
    """
    target = _html_escape(oauth_return_target(return_url, provider), quote=True)
    return (
        "<html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'></head>"
        "<body style='font-family:sans-serif;text-align:center;padding-top:60px'>"
        f"<h1>{heading_html}</h1>"
        "<p>Glossarion finishes the sign-in on its own. Returning to Glossarion&hellip;</p>"
        f"<p><a id='glossarion-return' href='{target}' "
        "style='display:inline-block;margin-top:16px;padding:12px 20px;border-radius:20px;"
        "background:#E18F98;color:#121826;text-decoration:none;font-weight:bold'>"
        "Return to Glossarion</a></p>"
        "<p style='color:#666'>If the app does not open, close this page "
        "(&#10005; or Done at the top) to go back to Glossarion.</p>"
        "<script>setTimeout(function(){location.href="
        "document.getElementById('glossarion-return').href;},300);</script>"
        "</body></html>"
    )


def send_page_quietly(handler: Any, status: int, body: str,
                      content_type: str = "text/html; charset=utf-8") -> bool:
    """Answer a loopback callback request; False when the browser was already gone.

    Glossarion Mobile: the answer can come late (the app was paused while the
    browser waited) and the tab may have been closed in the meantime. That is
    not an error - the caller records the callback *before* answering, so the
    sign-in still finishes.
    """
    data = body.encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(data)
        return True
    except OSError as exc:  # BrokenPipe / ConnectionReset / ConnectionAborted / timeout
        logger.info("OAuth callback: the browser closed the connection before the page was sent (%s)",
                    type(exc).__name__)
        handler.close_connection = True
        return False


def oauth_success_html(provider: str, default_html: str, heading_html: str = "&#10004; Authenticated!") -> str:
    """*default_html* (desktop), or the return-to-app page when GLOSSARION_OAUTH_RETURN_URL is set."""
    return_url = oauth_return_url()
    if not return_url:
        return default_html
    return oauth_return_page(return_url, provider, heading_html)


# ===========================================================================
# Pasted redirects
# ===========================================================================

def parse_oauth_redirect(value: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(code, state, error) from a pasted redirect URL, its query string,
    'code#state', or a bare authorization code."""
    text = str(value or "").strip()
    if not text:
        return None, None, None

    def from_query(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        qs = parse_qs(query.lstrip("?#"))
        return qs.get("code", [None])[0], qs.get("state", [None])[0], qs.get("error", [None])[0]

    if "?" in text:
        query, _, fragment = text.split("?", 1)[1].partition("#")
        code, state, error = from_query(query)
        if not code and not error and "=" in fragment:
            code, state, error = from_query(fragment)
        return code, state, error
    if "=" in text:
        if "://" in text and "#" in text:
            return from_query(text.split("#", 1)[1])
        return from_query(text)
    if "#" in text:
        code, _, state = text.partition("#")
        return code.strip() or None, state.strip() or None, None
    return text, None, None


# ===========================================================================
# Loopback session
# ===========================================================================

def _shutdown_quietly(server: HTTPServer) -> None:
    try:
        server.shutdown()
    except Exception:
        pass


class _IPv6LoopbackServer(HTTPServer):
    address_family = socket.AF_INET6


class _MobileLoopbackServer(ThreadingHTTPServer):
    """Glossarion Mobile's callback listener: one thread per connection, and a
    connection that has not sent its request line within
    :data:`MOBILE_REQUEST_TIMEOUT` is dropped.

    The desktop's single-threaded ``HTTPServer`` serves one connection at a time
    with no read timeout, so one idle socket (a browser preconnect, a load the
    user aborted) holds back the real redirect and the page just spins.
    """

    daemon_threads = True
    request_timeout = MOBILE_REQUEST_TIMEOUT

    def get_request(self):
        sock, address = super().get_request()
        try:
            sock.settimeout(self.request_timeout)
        except OSError:
            pass
        return sock, address


class _MobileIPv6LoopbackServer(_MobileLoopbackServer):
    address_family = socket.AF_INET6


def make_mobile_loopback_server(address: Tuple[str, int], handler_cls: Any, ipv6: bool = False) -> HTTPServer:
    """Glossarion Mobile's callback listener on *address* (see :class:`_MobileLoopbackServer`).

    Raises OSError when the address cannot be bound, like ``HTTPServer``.
    """
    server_cls = _MobileIPv6LoopbackServer if ipv6 else _MobileLoopbackServer
    return server_cls(address, handler_cls)


def make_ipv6_loopback_server(port: int, handler_cls: Any, timeout: Optional[float] = None,
                              mobile: bool = False) -> Optional[HTTPServer]:
    """A second callback listener on [::1]:*port*, or None when IPv6 (or the port) is unavailable.

    Mobile browsers may resolve 'localhost' to ::1 before 127.0.0.1. *mobile*
    gives it Glossarion Mobile's threaded listener (:func:`make_mobile_loopback_server`).
    """
    if not getattr(socket, "has_ipv6", False):
        return None
    try:
        if mobile:
            server = make_mobile_loopback_server(("::1", port), handler_cls, ipv6=True)
        else:
            server = _IPv6LoopbackServer(("::1", port), handler_cls)
    except Exception:
        return None
    server._auth_code = None
    server._returned_state = None
    server._error = None
    if timeout is not None:
        server.timeout = timeout
    return server


class LoopbackOAuthSession:
    """A browser sign-in started by a ``begin_oauth()``, waiting for its redirect.

    The caller opens ``auth_url`` in a browser. When the browser is sent to
    ``redirect_uri``, the provider's loopback handler calls
    :meth:`_record_callback`, which fills ``auth_code``, ``returned_state`` and
    ``error``; the provider's ``complete_from_redirect()`` then checks the state
    and exchanges the code. :meth:`pending_state` is what ``begin_oauth()``
    persists for the paste-the-redirect fallback.

    The listeners either serve on their own threads (:meth:`_start`, Glossarion
    Mobile) or are driven by the caller one request at a time
    (:meth:`handle_request`, the desktop loops that poll for cancellation).
    """

    def __init__(
        self,
        auth_url: str,
        code_verifier: Optional[str],
        state: str,
        redirect_uri: str,
        store: Any = None,
        account_id: Optional[int] = None,
        timeout: float = 300,
        provider: str = "oauth",
        extra: Optional[Dict] = None,
    ):
        self.auth_url = auth_url
        self.code_verifier = code_verifier
        self.state = state
        self.redirect_uri = redirect_uri
        self.store = store
        self.account_id = account_id
        self.timeout = timeout
        self.provider = provider
        self.extra: Dict = dict(extra or {})
        self.created_at = time.time()
        self.auth_code: Optional[str] = None
        self.returned_state: Optional[str] = None
        self.error: Optional[str] = None
        self._servers: List[HTTPServer] = []
        self._serving = 0
        self._closed = False
        self._watchdog: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._callback_event = threading.Event()
        self._done_event = threading.Event()
        self._close_finished = threading.Event()

    @property
    def server_running(self) -> bool:
        """True while a loopback listener is serving."""
        with self._lock:
            return self._serving > 0 and not self._closed

    @property
    def callback_received(self) -> bool:
        """True once the loopback server received the redirect (code or error)."""
        return self._callback_event.is_set()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def port(self) -> Optional[int]:
        """Port of the first listener (None without one)."""
        if not self._servers:
            return None
        return self._servers[0].server_address[1]

    def pending_state(self) -> Dict:
        """What ``complete_from_redirect()`` needs to finish this sign-in later."""
        pending = {
            "provider": self.provider,
            "auth_url": self.auth_url,
            "code_verifier": self.code_verifier,
            "state": self.state,
            "redirect_uri": self.redirect_uri,
            "account_id": self.account_id,
            "created_at": self.created_at,
        }
        pending.update(self.extra)
        return pending

    def wait_for_callback(self, timeout: Optional[float] = None) -> bool:
        """Block until the loopback server received the redirect; False on timeout."""
        return self._callback_event.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the success page was served or the listeners stopped."""
        return self._done_event.wait(timeout)

    def handle_request(self) -> None:
        """Serve one request on the first listener (caller-driven mode, see the class doc)."""
        if self._servers and not self.closed:
            self._servers[0].handle_request()

    def close(self) -> None:
        """Stop the loopback listener(s) and free the port. Safe to call repeatedly.

        A second caller returns once the first one has released the port.
        """
        with self._lock:
            already_closing = self._closed
            self._closed = True
            servers = list(self._servers)
            watchdog, self._watchdog = self._watchdog, None
        if already_closing:
            self._close_finished.wait()
            return
        if watchdog is not None:
            watchdog.cancel()
        # shutdown() waits up to one poll interval per listener; stop them in parallel.
        stoppers = [
            threading.Thread(target=_shutdown_quietly, args=(server,), daemon=True)
            for server in servers
            if getattr(server, "_glossarion_serving", False)
        ]
        for stopper in stoppers:
            stopper.start()
        for stopper in stoppers:
            stopper.join()
        for server in servers:
            try:
                server.server_close()
            except Exception:
                pass
        self._done_event.set()
        self._close_finished.set()

    def _add_server(self, server: HTTPServer) -> None:
        server._oauth_session = self
        self._servers.append(server)

    def _start(self, auto_close: bool) -> None:
        for server in self._servers:
            thread = threading.Thread(
                target=self._serve, args=(server,), daemon=True, name=f"{self.provider}-oauth-callback"
            )
            with self._lock:
                self._serving += 1
            server._glossarion_serving = True
            try:
                thread.start()
            except BaseException:
                server._glossarion_serving = False
                with self._lock:
                    self._serving -= 1
                raise
        if auto_close and self.timeout:
            watchdog = threading.Timer(self.timeout, self.close)
            watchdog.daemon = True
            with self._lock:
                self._watchdog = watchdog
            watchdog.start()

    def _serve(self, server: HTTPServer) -> None:
        try:
            server.serve_forever()
        finally:
            with self._lock:
                self._serving -= 1
                finished = self._serving <= 0
            if finished:
                self._done_event.set()

    def _record_callback(self, code: Optional[str], state: Optional[str], error: Optional[str]) -> None:
        with self._lock:
            self.auth_code = code
            self.returned_state = state
            self.error = error
        self._callback_event.set()


def session_of(server: Any) -> Optional[LoopbackOAuthSession]:
    """The session a loopback server belongs to (None for a bare desktop server)."""
    return getattr(server, "_oauth_session", None)


# ===========================================================================
# complete_from_redirect() helpers
# ===========================================================================

class Completion:
    """What a ``complete_from_redirect()`` resolved: the sign-in and the redirect to check."""

    def __init__(self, session, store, pending, code, returned_state, error, use_loopback, pasted):
        self.session: Optional[LoopbackOAuthSession] = session
        self.store = store
        self.pending: Dict = pending
        self.code: Optional[str] = code
        self.returned_state: Optional[str] = returned_state
        self.error: Optional[str] = error
        self.use_loopback: bool = use_loopback
        self.pasted: Optional[str] = pasted

    @property
    def expected_state(self) -> Optional[str]:
        return self.pending.get("state")

    @property
    def state_mismatch(self) -> bool:
        """The redirect carried a state (always, for the loopback) and it is not ours."""
        return (self.use_loopback or self.returned_state is not None) and self.returned_state != self.expected_state


def resolve_completion(
    session_or_saved_state: Any,
    redirect_url_or_code: Optional[str],
    store: Any,
    *,
    provider: str,
    label: str,
    require_verifier: bool = True,
) -> Completion:
    """The common start of ``complete_from_redirect()`` (authgpt's U1 semantics).

    *session_or_saved_state* is the :class:`LoopbackOAuthSession`, a saved
    pending dict (``pending_state()`` / ``store.load_pending_oauth()``), or a
    token store whose persisted pending sign-in is used. *redirect_url_or_code*
    is None to use what the loopback server captured (session only), else the
    pasted value, parsed with :func:`parse_oauth_redirect`.

    Raises RuntimeError when nothing is pending, the saved sign-in belongs to
    another provider or is incomplete, or a session-less call has nothing pasted.
    """
    session: Optional[LoopbackOAuthSession] = None
    if isinstance(session_or_saved_state, LoopbackOAuthSession):
        session = session_or_saved_state
        if store is None:
            store = session.store
        pending = session.pending_state()
    elif isinstance(session_or_saved_state, dict):
        pending = session_or_saved_state
    else:
        if session_or_saved_state is not None:
            store = session_or_saved_state
        if store is None or not hasattr(store, "load_pending_oauth"):
            raise RuntimeError(f"No pending {label} sign-in to complete. Start the sign-in again.")
        pending = store.load_pending_oauth()
        if not pending:
            raise RuntimeError(
                f"No pending {label} sign-in to complete (it may have expired). Start the sign-in again."
            )

    saved_provider = pending.get("provider")
    if saved_provider and saved_provider != provider:
        raise RuntimeError(f"The saved sign-in is not a {label} sign-in. Start the sign-in again.")
    if not (pending.get("state") and pending.get("redirect_uri")) or (
        require_verifier and not pending.get("code_verifier")
    ):
        raise RuntimeError(f"The saved {label} sign-in is incomplete. Start the sign-in again.")

    use_loopback = session is not None and redirect_url_or_code is None
    if use_loopback:
        code, returned_state, error = session.auth_code, session.returned_state, session.error
    elif redirect_url_or_code is None:
        raise RuntimeError(f"Paste the redirect URL (or the code) to finish the {label} sign-in.")
    else:
        code, returned_state, error = parse_oauth_redirect(redirect_url_or_code)
    return Completion(session, store, pending, code, returned_state, error, use_loopback, redirect_url_or_code)


def finish_completion(completion: Completion, tokens: Dict) -> Dict:
    """The common end of ``complete_from_redirect()``.

    Tokens are saved to the store (when there is one) and the pending sign-in is
    cleared; a pasted redirect also stops a still-running loopback listener.
    """
    store = completion.store
    if store is not None:
        store.save_tokens(tokens)
        if hasattr(store, "clear_pending_oauth"):
            store.clear_pending_oauth()
    if completion.session is not None and not completion.use_loopback:
        completion.session.close()
    return tokens


# ===========================================================================
# Pending sign-in storage (mixed into the token stores)
# ===========================================================================

class PendingOAuthMixin:
    """Encrypted pending sign-in next to a token store's file (authgpt's U1 methods).

    Needs ``self._token_file``, ``self._lock`` (an RLock) and ``self._ensure_dir()``,
    which every auth token store has. A saved dict expires after
    :data:`PENDING_OAUTH_MAX_AGE_SECONDS`, or earlier at its own ``expires_at``
    (a device code).
    """

    _pending_label = "OAuth"

    @property
    def pending_oauth_file(self) -> str:
        """Encrypted PKCE verifier/state of an unfinished sign-in, next to the token file."""
        return os.path.splitext(self._token_file)[0] + ".oauth_pending"

    def save_pending_oauth(self, pending: Dict) -> bool:
        """Save a pending sign-in encrypted like the tokens (never as plain text)."""
        path = self.pending_oauth_file
        with self._lock:
            try:
                from token_encryption import encrypt_tokens
                data = encrypt_tokens(dict(pending))
                self._ensure_dir()
                with open(path, "wb") as f:
                    f.write(data)
                if os.name != "nt":
                    try:
                        os.chmod(path, 0o600)
                    except OSError:
                        pass
                return True
            except Exception as exc:
                logger.warning("⚠️ %s: could not save the pending sign-in (%s)", self._pending_label, type(exc).__name__)
                return False

    def load_pending_oauth(self) -> Optional[Dict]:
        """Return the saved pending sign-in, or None when missing, unreadable or expired."""
        path = self.pending_oauth_file
        with self._lock:
            if not os.path.isfile(path):
                return None
            try:
                with open(path, "rb") as f:
                    data = f.read()
                from token_encryption import decrypt_tokens
                pending = decrypt_tokens(data)
            except Exception as exc:
                logger.warning(
                    "⚠️ %s: discarding unreadable pending sign-in (%s)", self._pending_label, type(exc).__name__
                )
                self.clear_pending_oauth()
                return None
            if not isinstance(pending, dict):
                self.clear_pending_oauth()
                return None
            now = time.time()
            try:
                expired = now - float(pending.get("created_at")) > PENDING_OAUTH_MAX_AGE_SECONDS
            except (TypeError, ValueError):
                expired = True
            expires_at = pending.get("expires_at")
            if not expired and expires_at is not None:
                try:
                    expired = now >= float(expires_at)
                except (TypeError, ValueError):
                    expired = True
            if expired:
                self.clear_pending_oauth()
                return None
            return pending

    def clear_pending_oauth(self) -> None:
        """Forget a pending sign-in."""
        path = self.pending_oauth_file
        with self._lock:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("Failed to remove pending sign-in file: %s", exc)
