"""Test double for the in-app browser page API (no Flet import needed; nothing touches the network).

``FakeWebPagesNative`` has ``GlossarionNative``'s ``web_*`` coroutines with the semantics of the native
side (``HiddenWebPages.kt`` / ``HiddenWebPages.swift``, contract in ``web_pages.py``):

* ``web_open``: ``bad_args`` without a ``page_id``, ``exists`` for an open id, ``too_many`` past the cap
  (gone pages count until they are closed); ``prevent_links`` and the viewport are recorded.
* ``web_load``: http / https only; the page gets a new document (``generation`` + 1, ``state`` cleared)
  and the events ``started`` -> ``url`` -> ``finished`` follow the answer on the loop, like platform
  events (``auto_finish=False`` sends only ``started``; :meth:`finish` completes it).
* ``web_eval``: the expression inside the app's eval wrapper (see :func:`expression_of`) goes to
  ``responder(page_id, expression)``; its value (or a raised exception) becomes the envelope
  ``JSON.stringify({ok, value})`` / ``{ok: false, error}`` the wrapper returns, as one string. A script
  that is not wrapped answers the responder's value when it is a string (the native completion-value
  rule). Held evals (:meth:`hold_evals`) wait until :meth:`release_evals`, and answer ``closed`` /
  ``gone`` when their page is closed / crashes first, exactly once.
* ``crash(page_id)``: the renderer stopped: event ``gone``, pending evals answer ``gone``, the id keeps
  answering ``gone`` until ``web_close``.
* ``get_platform_info()``: ``{"platform", "native": True, "web_pages": capability}`` (no ``web_pages`` key
  with ``capability=None``; ``hang_platform_info=True`` never answers).

Events go to ``emit(payload_dict)`` when given, else to ``on_web`` (``NativeBridge(native=fake)`` sets it)
as a real ``WebPageEvent`` when Flet is importable (the Flet dispatch path), a plain dict otherwise.
Every hook that changes state (``crash``, ``finish``, ``navigate``, ``release_evals`` ...) may be called
from any thread: it runs on the loop the fake's calls came from and returns a
``concurrent.futures.Future``. ``wait_held`` / ``wait_calls`` / ``wait_events`` block a test thread on a
condition (no sleeps). Use it like::

    fake = FakeWebPagesNative(lambda page_id, expression: 42)
    bridge = NativeBridge(native=fake)          # sets fake.on_web
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import re
import threading
from typing import Any, Callable, Optional

from flet_glossarion_native.web_pages import (
    NATIVE_MAX_PAGES,
    WEB_PAGES_VERSION,
    WebPageError,
    clamp_side,
    error_result,
    is_http_url,
)

__all__ = ["DEFAULT_CAPABILITY", "FakeJsError", "FakePage", "FakeWebPagesNative", "expression_of"]

#: ``get_platform_info()["web_pages"]`` of the fake.
DEFAULT_CAPABILITY = {"version": WEB_PAGES_VERSION, "max_pages": NATIVE_MAX_PAGES, "provider": "fake"}

# The app's wrapper puts the expression on its own lines between "var value = (" and ");" followed by
# "envelope = {ok: true" (services/native_webview.eval_script); indentation may differ.
_WRAPPED = re.compile(r"var value = \(\r?\n(.*)\r?\n[ \t]*\);\r?\n[ \t]*envelope = \{ok: true", re.S)


def expression_of(script: Any) -> Optional[str]:
    """The expression inside the app's eval wrapper, or None when ``script`` is not wrapped."""
    if not isinstance(script, str):
        return None
    match = _WRAPPED.search(script)
    if match is None:
        return None
    expression = match.group(1)
    return expression[:-1] if expression.endswith("\r") else expression


class FakeJsError(Exception):
    """Raise it (or any exception) from a responder: the expression threw; the envelope says ok: false."""


class FakePage:
    """One page of the fake: what the native side would hold, plus a ``state`` dict for responders
    (stands in for the document's JavaScript globals; a new document starts empty)."""

    def __init__(self, page_id: str, width: int, height: int, prevent_links: list[str]) -> None:
        self.page_id = page_id
        self.width = width
        self.height = height
        self.prevent_links = prevent_links
        self.url: Optional[str] = None
        self.title: Optional[str] = None
        self.generation = 0
        self.loads: list[str] = []
        self.state: dict[str, Any] = {}
        self.gone = False
        self.closed = False


class _Held:
    def __init__(self, method: str, page_id: Optional[str], script: Optional[str],
                 future: "asyncio.Future[dict]") -> None:
        self.method = method
        self.page_id = page_id
        self.script = script
        self.future = future


def _settle(future: "asyncio.Future[dict]") -> Callable[["asyncio.Future[dict]"], None]:
    """Done-callback: hand a finished evaluation to the held call's future (first answer wins)."""

    def done(task: "asyncio.Future[dict]") -> None:
        if future.done():
            return
        if task.cancelled():
            future.set_result(error_result(WebPageError.CLOSED, "the in-app page was closed"))
        elif task.exception() is not None:
            future.set_result(error_result(WebPageError.NATIVE_ERROR, type(task.exception()).__name__))
        else:
            future.set_result(task.result())

    return done


def _ok(**values: Any) -> dict:
    out: dict[str, Any] = {"ok": True, "error": None, "message": None}
    out.update(values)
    return out


class FakeWebPagesNative:
    """``GlossarionNative``'s in-app page methods answered like the native side (see the module doc)."""

    is_stub = False

    def __init__(
        self,
        responder: Optional[Callable[[str, str], Any]] = None,
        *,
        emit: Optional[Callable[[dict], Any]] = None,
        capability: Optional[dict] = DEFAULT_CAPABILITY,
        title_of: Optional[Callable[[str], Optional[str]]] = None,
        platform: str = "android",
        max_pages: int = NATIVE_MAX_PAGES,
        auto_finish: bool = True,
        hang_platform_info: bool = False,
    ) -> None:
        self.responder = responder
        self.emit = emit
        self.capability = dict(capability) if capability is not None else None
        self.title_of = title_of
        self.platform = platform
        self.max_pages = max_pages
        self.auto_finish = auto_finish
        self.hang_platform_info = hang_platform_info
        self.on_web: Optional[Callable[[Any], Any]] = None
        self.pages: dict[str, FakePage] = {}
        self.calls: list[tuple[str, dict]] = []
        self.events: list[dict] = []
        self.closed_ids: list[str] = []
        self.resets = 0
        self._held: list[_Held] = []
        self._hold_methods: set[str] = set()
        self._hold_pages: Optional[set[str]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._cond = threading.Condition()

    # ------------------------------------------------------------------ helpers

    def _record(self, method: str, **args: Any) -> None:
        self._loop = asyncio.get_running_loop()
        with self._cond:
            self.calls.append((method, args))
            self._cond.notify_all()

    def calls_of(self, method: str) -> list[dict]:
        """Arguments of every call of ``method``, oldest first."""
        return [args for name, args in list(self.calls) if name == method]

    def page(self, page_id: str) -> Optional[FakePage]:
        return self.pages.get(page_id)

    @property
    def open_pages(self) -> list[str]:
        """Ids of pages that are open and alive."""
        return [p.page_id for p in self.pages.values() if not p.gone]

    def _wait(self, predicate: Callable[[], bool], timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(predicate, timeout)

    def wait_calls(self, method: str, count: int = 1, timeout: float = 10.0) -> bool:
        """Block (a test thread) until ``method`` was called ``count`` times."""
        return self._wait(lambda: len(self.calls_of(method)) >= count, timeout)

    def wait_held(self, count: int = 1, timeout: float = 10.0, *, method: str = "web_eval") -> bool:
        """Block until ``count`` calls of ``method`` are held (see :meth:`hold`)."""
        return self._wait(lambda: sum(1 for h in self._held if h.method == method) >= count, timeout)

    def wait_events(self, predicate: Callable[[dict], bool], count: int = 1, timeout: float = 10.0) -> bool:
        """Block until ``count`` emitted events match ``predicate``."""
        return self._wait(lambda: sum(1 for e in list(self.events) if predicate(e)) >= count, timeout)

    def _on_loop(self, fn: Callable[..., Any], *args: Any) -> "concurrent.futures.Future[Any]":
        """Run ``fn(*args)`` on the fake's loop (now, when called on it) and return a concurrent Future."""
        done: "concurrent.futures.Future[Any]" = concurrent.futures.Future()
        loop = self._loop
        try:
            running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop:
            try:
                done.set_result(fn(*args))
            except BaseException as exc:  # handed to the caller
                done.set_exception(exc)
            return done

        def run() -> None:
            try:
                done.set_result(fn(*args))
            except BaseException as exc:
                done.set_exception(exc)

        loop.call_soon_threadsafe(run)
        return done

    # ------------------------------------------------------------------ events

    def _dispatch(self, payload: dict) -> None:
        with self._cond:
            self.events.append(dict(payload))
            self._cond.notify_all()
        if self.emit is not None:
            result = self.emit(dict(payload))
        else:
            handler = self.on_web
            if handler is None:
                return
            result = handler(self._event(payload))
        if inspect.isawaitable(result):
            asyncio.ensure_future(result)

    def _event(self, payload: dict) -> Any:
        try:
            from flet_glossarion_native.types import WebPageEvent
        except Exception:  # no Flet here: the plain payload
            event = dict(payload)
            event["name"] = "web"
            return event
        fields = {k: payload.get(k) for k in ("type", "page_id", "url", "ok", "error", "code", "main_frame",
                                              "crashed")}
        fields["type"] = fields["type"] or ""
        return WebPageEvent(name="web", control=self, **fields)

    def _schedule(self, payload: dict) -> None:
        """An event that follows the current answer (the platform sends it on its own)."""
        asyncio.get_running_loop().call_soon(self._dispatch, payload)

    def emit_event(self, payload: dict) -> "concurrent.futures.Future[Any]":
        """Inject a raw event (any thread)."""
        return self._on_loop(self._dispatch, dict(payload))

    # ------------------------------------------------------------------ holds

    def hold(self, method: str = "web_eval", page_id: Optional[str] = None) -> None:
        """Hold the answers of later ``method`` calls (of ``page_id`` only, when given) until :meth:`release`.
        Effects of ``web_open`` / ``web_load`` / ``web_close`` apply at once (the platform got the call);
        only the answer waits. A held ``web_eval`` computes its value when released."""
        self._hold_methods.add(method)
        if page_id is not None:
            self._hold_pages = (self._hold_pages or set()) | {page_id}

    def hold_evals(self, page_id: Optional[str] = None) -> None:
        self.hold("web_eval", page_id)

    def release(self, method: Optional[str] = None, page_id: Optional[str] = None,
                *, keep_holding: bool = False) -> "concurrent.futures.Future[int]":
        """Answer held calls (all, or of ``method`` / ``page_id``); stops holding unless ``keep_holding``."""
        if not keep_holding:
            if method is None:
                self._hold_methods.clear()
                self._hold_pages = None
            else:
                self._hold_methods.discard(method)
        return self._on_loop(self._release, method, page_id)

    def release_evals(self, page_id: Optional[str] = None, *, keep_holding: bool = False) -> "concurrent.futures.Future[int]":
        return self.release("web_eval", page_id, keep_holding=keep_holding)

    def _holding(self, method: str, page_id: Optional[str]) -> bool:
        if method not in self._hold_methods:
            return False
        return self._hold_pages is None or page_id in self._hold_pages

    async def _held_answer(self, method: str, page_id: Optional[str], script: Optional[str] = None) -> dict:
        future: "asyncio.Future[dict]" = asyncio.get_running_loop().create_future()
        with self._cond:
            self._held.append(_Held(method, page_id, script, future))
            self._cond.notify_all()
        return await future

    def _take_held(self, keep: Callable[[_Held], bool]) -> list[_Held]:
        with self._cond:
            taken = [held for held in self._held if keep(held)]
            self._held = [held for held in self._held if not keep(held)]
            self._cond.notify_all()
        return taken

    def _release(self, method: Optional[str], page_id: Optional[str]) -> int:
        taken = self._take_held(lambda held: (method is None or held.method == method)
                                and (page_id is None or held.page_id == page_id))
        released = 0
        for held in taken:
            if held.future.done():
                continue
            released += 1
            if held.method != "web_eval":
                held.future.set_result(_ok())
                continue
            page = self.pages.get(held.page_id or "")
            if page is None:
                held.future.set_result(error_result(WebPageError.NO_PAGE, "no such in-app page"))
            elif page.gone:
                held.future.set_result(error_result(WebPageError.GONE, "the in-app page's renderer stopped"))
            else:
                task = asyncio.ensure_future(self._evaluate(page, held.script or ""))
                task.add_done_callback(_settle(held.future))
        return released

    def _fail_held(self, page_id: Optional[str], code: WebPageError) -> None:
        message = "the in-app page's renderer stopped" if code == WebPageError.GONE else "the in-app page was closed"
        taken = self._take_held(lambda held: held.method == "web_eval"
                                and (page_id is None or held.page_id == page_id))
        for held in taken:
            if not held.future.done():
                held.future.set_result(error_result(code, message))

    # ------------------------------------------------------------------ platform hooks

    def crash(self, page_id: str, *, crashed: Optional[bool] = True) -> "concurrent.futures.Future[bool]":
        """The page's renderer stops (any thread): event ``gone``, pending evals answer ``gone``.
        ``crashed`` None is iOS (WebKit gives no reason)."""
        return self._on_loop(self._crash, page_id, crashed)

    def _crash(self, page_id: str, crashed: Optional[bool]) -> bool:
        page = self.pages.get(page_id)
        if page is None or page.gone:
            return False
        page.gone = True
        self._dispatch({"page_id": page_id, "type": "gone", "crashed": None if crashed is None else bool(crashed)})
        self._fail_held(page_id, WebPageError.GONE)
        return True

    def finish(self, page_id: str, *, ok: bool = True, error: Optional[str] = None) -> "concurrent.futures.Future[bool]":
        """Complete the page's current load (with ``auto_finish=False``), any thread."""
        return self._on_loop(self._finish, page_id, ok, error)

    def _finish(self, page_id: str, ok: bool, error: Optional[str]) -> bool:
        page = self.pages.get(page_id)
        if page is None or page.gone:
            return False
        if not ok:
            self._dispatch({"page_id": page_id, "type": "error", "url": page.url, "code": -2, "error": error,
                            "main_frame": True})
        self._dispatch({"page_id": page_id, "type": "finished", "url": page.url, "ok": bool(ok), "error": error})
        return True

    def navigate(self, page_id: str, url: str, *, title: Optional[str] = None) -> "concurrent.futures.Future[bool]":
        """The page navigates by itself (a reload, a redirect): a new document with the usual events."""
        return self._on_loop(self._navigate, page_id, url, title)

    def _navigate(self, page_id: str, url: str, title: Optional[str]) -> bool:
        page = self.pages.get(page_id)
        if page is None or page.gone:
            return False
        self._new_document(page, url, title)
        self._dispatch({"page_id": page_id, "type": "started", "url": url})
        self._dispatch({"page_id": page_id, "type": "url", "url": url})
        self._dispatch({"page_id": page_id, "type": "finished", "url": url, "ok": True, "error": None})
        return True

    def _new_document(self, page: FakePage, url: str, title: Optional[str] = None) -> None:
        page.url = url
        page.generation += 1
        page.state = {}
        if title is None and self.title_of is not None:
            title = self.title_of(url)
        page.title = title

    # ------------------------------------------------------------------ the API

    async def get_platform_info(self) -> dict:
        self._record("get_platform_info")
        if self.hang_platform_info:
            await asyncio.get_running_loop().create_future()  # never answers (cancelled by the caller)
        info: dict[str, Any] = {"platform": self.platform, "native": True, "extension_version": "0.1.0"}
        if self.capability is not None:
            info["web_pages"] = dict(self.capability)
        return info

    async def web_open(self, page_id: str, *, width: int, height: int, prevent_links: Any = ()) -> dict:
        links = [str(p) for p in (prevent_links or ()) if p]
        self._record("web_open", page_id=page_id, width=width, height=height, prevent_links=links)
        await asyncio.sleep(0)
        if not isinstance(page_id, str) or not page_id:
            return error_result(WebPageError.BAD_ARGS, "page_id is required")
        if page_id in self.pages:
            return error_result(WebPageError.EXISTS, "the page is already open")
        if len(self.pages) >= self.max_pages:
            return error_result(WebPageError.TOO_MANY, "too many in-app pages")
        self.pages[page_id] = FakePage(page_id, clamp_side(width, 480), clamp_side(height, 800), links)
        if self._holding("web_open", page_id):
            await self._held_answer("web_open", page_id)
        return _ok(page_id=page_id, attached=True)

    def _usable(self, page_id: Any) -> tuple[Optional[FakePage], Optional[dict]]:
        page = self.pages.get(page_id) if isinstance(page_id, str) else None
        if page is None:
            return None, error_result(WebPageError.NO_PAGE, "no such in-app page")
        if page.gone:
            return None, error_result(WebPageError.GONE, "the in-app page's renderer stopped")
        return page, None

    async def web_load(self, page_id: str, url: str) -> dict:
        self._record("web_load", page_id=page_id, url=url)
        await asyncio.sleep(0)
        page, failure = self._usable(page_id)
        if failure is not None:
            return failure
        assert page is not None
        if not is_http_url(url):
            return error_result(WebPageError.BAD_ARGS, "only http(s) pages can be opened")
        page.loads.append(url)
        self._new_document(page, url)
        self._schedule({"page_id": page_id, "type": "started", "url": url})
        self._schedule({"page_id": page_id, "type": "url", "url": url})
        if self.auto_finish:
            self._schedule({"page_id": page_id, "type": "finished", "url": url, "ok": True, "error": None})
        if self._holding("web_load", page_id):
            await self._held_answer("web_load", page_id)
        return _ok()

    async def web_eval(self, page_id: str, script: str, *, timeout: float = 30.0) -> dict:
        self._record("web_eval", page_id=page_id, script=script, timeout=timeout)
        await asyncio.sleep(0)
        page, failure = self._usable(page_id)
        if failure is not None:
            return failure
        assert page is not None
        if not isinstance(script, str):
            return error_result(WebPageError.BAD_ARGS, "script is required")
        if self._holding("web_eval", page_id):
            return await self._held_answer("web_eval", page_id, script)
        return await self._evaluate(page, script)

    async def _respond(self, page: FakePage, expression: str) -> Any:
        if self.responder is None:
            return None
        value = self.responder(page.page_id, expression)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _evaluate(self, page: FakePage, script: str) -> dict:
        if page.gone:
            return error_result(WebPageError.GONE, "the in-app page's renderer stopped")
        expression = expression_of(script)
        if expression is None:
            try:
                value = await self._respond(page, script)
            except Exception:
                value = None
            return _ok(result=value if isinstance(value, str) else None, url=page.url)
        try:
            envelope: dict[str, Any] = {"ok": True, "value": await self._respond(page, expression)}
        except Exception as exc:
            envelope = {"ok": False, "error": str(exc)}
        try:
            text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            text = json.dumps({"ok": False, "error": "the value is not JSON"})
        if page.gone:  # crashed while the responder ran
            return error_result(WebPageError.GONE, "the in-app page's renderer stopped")
        return _ok(result=text, url=page.url)

    async def web_title(self, page_id: str) -> dict:
        self._record("web_title", page_id=page_id)
        await asyncio.sleep(0)
        page, failure = self._usable(page_id)
        if failure is not None:
            return failure
        assert page is not None
        return _ok(title=page.title, url=page.url)

    async def web_close(self, page_id: str) -> dict:
        self._record("web_close", page_id=page_id)
        await asyncio.sleep(0)
        page = self.pages.pop(page_id, None) if isinstance(page_id, str) else None
        if page is not None:
            page.closed = True
            self.closed_ids.append(page_id)
            self._fail_held(page_id, WebPageError.CLOSED)
            with self._cond:
                self._cond.notify_all()
        if self._holding("web_close", page_id):
            await self._held_answer("web_close", page_id)
        return _ok(closed=page is not None)

    async def web_reset(self) -> dict:
        self._record("web_reset")
        await asyncio.sleep(0)
        closed = list(self.pages)
        for page_id in closed:
            self.pages[page_id].closed = True
            self.closed_ids.append(page_id)
        self.pages.clear()
        self.resets += 1
        self._fail_held(None, WebPageError.CLOSED)
        with self._cond:
            self._cond.notify_all()
        return _ok(closed=len(closed))

    async def web_status(self) -> dict:
        self._record("web_status")
        await asyncio.sleep(0)
        pending: dict[Optional[str], int] = {}
        for held in self._held:
            if held.method == "web_eval":
                pending[held.page_id] = pending.get(held.page_id, 0) + 1
        pages = [{"page_id": p.page_id, "url": p.url, "gone": p.gone, "width": p.width, "height": p.height,
                  "pending": pending.get(p.page_id, 0)} for p in self.pages.values()]
        live = any(not p.gone for p in self.pages.values())
        extra: dict[str, Any] = {"activity": True, "density": 2.0} if self.platform == "android" else {"window": True}
        return _ok(pages=pages, attached=live, **extra)
