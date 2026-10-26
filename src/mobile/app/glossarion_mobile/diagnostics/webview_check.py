"""On-device check of the native in-app browser (hotfix 3): smoke-suite check ``native_webview``.

The bug it guards: the flet-webview pages of the in-app browser routes (authchatgpt/, authnd/,
authds/, search/gemini) lived in ``page.overlay``, which Flet 1.0.3 draws only in the top View,
so every screen push / pop made Flutter build a new WebView that loaded its URL again: an
authchatgpt/ reply in progress was lost. The native pages (``services/native_webview.py``) are
owned by the extension outside the Flutter widget tree. Only CI's Android emulator and iOS
simulator compile the extension's Kotlin / Swift side, so this check proves it there:

1. a page of the registered native driver loads a loopback document (the Reader's
   ``ReaderServer``: 127.0.0.1, random port, token path) and reports its title (when browser-route
   requests hold every page slot for :data:`SLOT_WAIT_SECONDS`, :class:`PagesBusy`: a skip);
2. a script sets a JavaScript global; then, on the Flet loop, :func:`screen_churn` pushes a
   Settings screen over the one shown, goes back and opens and closes a sheet;
3. the global and the page's load generation survived (no new document), scripts still run,
   typing through ``execCommand('insertText')`` reaches a textarea of the non-focusable native
   WebView (authchatgpt/ types that way), and a 200 000-character result arrives in one piece;
4. closing the page leaves no native page (``web_status``).

``check_native_webview`` (``diagnostics.selftest``) skips on the host (desktop, host tests,
``GLOSSARION_SIMULATE_PLATFORM`` host smoke runs) and when ``GLOSSARION_NATIVE_WEBVIEW`` switches
the native browser off; on a device in strict mode a missing native driver fails the check.
Blocking: runs on the self-test worker thread, never on the Flet loop.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "BIG_SIZE",
    "OWNER",
    "SELFTEST_HTML",
    "TITLE",
    "PagesBusy",
    "churn_route",
    "run",
    "screen_churn",
    "wait_for_native_driver",
]

TITLE = "Glossarion in-app browser self-test"
OWNER = "selftest"
VIEWPORT = (360, 640)
SELFTEST_HTML = (
    "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    f"<title>{TITLE}</title></head>\n"
    "<body><p>Checking that hidden pages survive screen changes.</p>"
    "<textarea id=\"t\"></textarea></body></html>\n"
)
#: The check's scripts (route-style expressions; the host tests' fake page answers exactly these).
SET_TOKEN_SCRIPT = "(window.__glossarionSelfTest = {token}, window.__glossarionSelfTest)"
GET_TOKEN_SCRIPT = "window.__glossarionSelfTest"
MATH_SCRIPT = "6 * 7"
TYPE_SCRIPT = ("(() => {const t = document.getElementById('t'); t.focus();"
               " document.execCommand('insertText', false, 'ok'); return t.value;})()")
BIG_SIZE = 200000
BIG_SCRIPT = f"'x'.repeat({BIG_SIZE})"
#: Route names :func:`churn_route` opens: Settings from the chat home, else the first of these pages
#: that is not on the stack (opened in the app, it goes on top of the screens shown).
CHURN_ROUTE = "settings"
CHURN_ROUTES_ON_TOP = ("settings.about", "settings.appearance", "settings.storage")
#: Seconds Flutter gets to build / tear down each screen change (device only; tests pass 0).
SETTLE_SECONDS = 0.8
#: Seconds a device waits for the app to register the native driver (start-up still running).
DRIVER_WAIT_SECONDS = 30.0
#: Seconds the closed page may take to leave ``web_status``.
CLOSE_WAIT_SECONDS = 5.0
#: Seconds the check waits for a free page slot: browser-route requests (authchatgpt/, authnd/,
#: authds/, search/gemini) may hold every one of the driver's ``max_pages``.
SLOT_WAIT_SECONDS = 30.0


class PagesBusy(RuntimeError):
    """Every in-app page slot stayed in use by browser-route requests for :data:`SLOT_WAIT_SECONDS`:
    the check cannot run now (``check_native_webview`` reports a skip, like a check waiting for a job)."""


def _open_check_page(driver: Any) -> Any:
    """The check's page, waiting at most :data:`SLOT_WAIT_SECONDS` for a free slot. The deadline only
    applies while ``open_page`` waits: the page keeps the cancel check, which never fires afterwards."""
    import browser_driver

    deadline = time.monotonic() + SLOT_WAIT_SECONDS
    waiting = [True]

    def slot_wait_over() -> bool:
        return waiting[0] and time.monotonic() >= deadline

    try:
        page = driver.open_page(owner=OWNER, viewport=VIEWPORT, cancel_check=slot_wait_over)
    except browser_driver.BrowserCancelled as exc:
        raise PagesBusy(f"every in-app page ({getattr(driver, 'max_pages', '?')}) stayed in use by browser-route "
                        f"requests for {SLOT_WAIT_SECONDS:g} s; run the self-test again once they finish") from exc
    waiting[0] = False
    return page


def wait_for_native_driver(timeout: Optional[float] = None) -> Tuple[Any, str]:
    """``(driver, "")`` once the native driver is the registered ``browser_driver``, else
    ``(None, why)``: switched off, another driver installed, or nothing within ``timeout``
    (default :data:`DRIVER_WAIT_SECONDS`)."""
    import browser_driver

    from glossarion_mobile.services import native_webview, webview_bridge

    if native_webview.disabled_by_env():
        return None, f"switched off by {native_webview.ENV_SWITCH}"
    wait = DRIVER_WAIT_SECONDS if timeout is None else timeout
    deadline = time.monotonic() + max(0.0, float(wait))
    while True:
        driver = browser_driver.get_driver()
        if getattr(driver, "name", None) == native_webview.NATIVE_NAME:
            return driver, ""
        if driver is not None:
            return None, f"the {getattr(driver, 'name', None) or 'other'} in-app browser is installed instead"
        if time.monotonic() >= deadline:
            return None, webview_bridge.availability()[1] or "no in-app browser is installed"
        time.sleep(0.25)


def churn_route(shell: Any) -> str:
    """The screen :func:`screen_churn` opens: Settings over the chat home; over any other screen (the
    first-run Welcome of a fresh install, Settings › Logs & diagnostics where "Run self-test" lives) a
    Settings page that is not on the stack yet. An in-app navigation puts such a page on top of the
    screens shown (``AppShell._push_base``), so phones get exactly one new View and Back returns to the
    very same screens. Settings itself is a top-level destination: opened over other screens it would
    replace them instead (fewer Views, and the Logs screen running the check would be closed)."""
    from glossarion_mobile.ui.router import build_route

    on_stack = {str(getattr(entry, "route", entry)) for entry in (getattr(shell, "stack", None) or [])}
    if not on_stack:
        return build_route(CHURN_ROUTE)
    for name in CHURN_ROUTES_ON_TOP:
        route = build_route(name)
        if route not in on_stack:
            return route
    return build_route(CHURN_ROUTE)


async def screen_churn(app: Any, *, settle: Optional[float] = None) -> dict:
    """Flet loop: push a screen over the one shown (:func:`churn_route`; a new View on phones, the old
    bug's trigger), go back, then open and close a sheet. ``App.back()`` may reopen an unfinished
    first-run Welcome on the chat home (``_resume_welcome``); the screen shown before is then
    navigated to again (``restored``)."""
    from glossarion_mobile.ui.components.info_sheet import InfoSheet

    pause = SETTLE_SECONDS if settle is None else max(0.0, float(settle))
    shell, page = app.shell, app.page
    tablet = bool(getattr(shell, "tablet", False))
    before, views_before = shell.current_route, len(page.views)
    await app.navigate(churn_route(shell))
    await asyncio.sleep(pause)
    pushed, views_pushed = shell.current_route, len(page.views)
    app.back()
    await asyncio.sleep(pause)
    after = shell.current_route
    restored = False
    if after != before:
        await app.navigate(before)
        await asyncio.sleep(pause)
        restored, after = True, shell.current_route
    sheet = InfoSheet(title="In-app browser self-test", body="Checking that hidden pages survive screen changes.")
    sheet.show(page)
    await asyncio.sleep(pause)
    sheet.close()
    await asyncio.sleep(pause)
    return {"before": before, "pushed": pushed, "after": after, "restored": restored, "tablet": tablet,
            "views_before": views_before, "views_pushed": views_pushed, "sheet": True}


def _check_churn(churn: Any) -> dict:
    """The screen changes really happened (phones: a View was pushed); returns the detail."""
    if not isinstance(churn, dict):
        raise AssertionError("the screen change gave no report")
    if churn.get("pushed") == churn.get("before"):
        raise AssertionError(f"the screen change did not happen (still on {str(churn.get('before'))[:60]})")
    proved = int(churn.get("views_pushed") or 0) > int(churn.get("views_before") or 0)
    if not churn.get("tablet") and not proved:
        raise AssertionError("the screen change did not happen (no View was pushed for Settings)")
    return dict(churn, proved_view_change=proved)


def _native_page_ids(driver: Any, timeout: float) -> Optional[list]:
    future = driver.dispatcher.submit(driver.native_status)
    if future is None:
        return None
    status = future.result(timeout=timeout)
    pages = status.get("pages") if isinstance(status, dict) else None
    if not isinstance(pages, list):
        return None
    return [str(item.get("page_id")) for item in pages if isinstance(item, dict)]


def run(driver: Any, *, server_factory: Optional[Callable[[], Any]] = None, timeout: float = 60.0) -> dict:
    """The check against ``driver`` (a registered ``NativeWebDriver``); raises AssertionError
    with the reason, returns the detail dict (worker thread)."""
    import browser_driver

    if server_factory is None:
        from glossarion_mobile.services.reader_server import ReaderServer as server_factory
    churn_fn = getattr(driver, "screen_churn", None)
    if churn_fn is None:
        raise AssertionError("the native in-app browser has no screen-change hook (native_webview.install sets it)")
    started = time.monotonic()
    detail: dict = {"driver": getattr(driver, "name", None), "capability": dict(getattr(driver, "capability", {}) or {})}
    server = server_factory()
    page = None
    try:
        server.start()
        url = server.publish(SELFTEST_HTML, name="webview-selftest.html")
        page = _open_check_page(driver)
        page.load(url)
        if not browser_driver.wait_until_loaded(page, 30):
            state = page.load_state
            raise AssertionError("the self-test page did not load: "
                                 f"{str(state.get('error') or state.get('last_event') or 'no page event')[:120]}")
        title = page.title()
        if title != TITLE:
            raise AssertionError(f"the self-test page reports the title {title[:80]!r}")
        token = uuid.uuid4().hex
        if page.run_js(SET_TOKEN_SCRIPT.format(token=json.dumps(token)), 10000) != token:
            raise AssertionError("the self-test page ran no script")
        generation = int(page.load_state.get("generation", 0))
        future = driver.dispatcher.submit(churn_fn)
        if future is None:
            raise AssertionError("the app's UI loop is not running")
        detail["churn"] = _check_churn(future.result(timeout=timeout))
        survived = page.run_js(GET_TOKEN_SCRIPT, 10000)
        after = int(page.load_state.get("generation", 0))
        if survived != token or after != generation or getattr(page, "renderer_gone", False):
            raise AssertionError(f"the hidden page lost its state across a screen change (generation {generation} -> "
                                 f"{after}{', renderer gone' if getattr(page, 'renderer_gone', False) else ''})")
        if page.run_js(MATH_SCRIPT, 10000) != 42:
            raise AssertionError("the hidden page stopped running scripts after a screen change")
        typed = page.run_js(TYPE_SCRIPT, 10000)
        if typed != "ok":
            raise AssertionError(f"typing into the hidden page failed (the textarea holds {str(typed)[:40]!r})")
        big = page.run_js(BIG_SCRIPT, 20000)
        if not isinstance(big, str) or len(big) != BIG_SIZE:
            raise AssertionError(f"a {BIG_SIZE}-character result came back as "
                                 f"{len(big) if isinstance(big, str) else type(big).__name__}")
        page_id = page.id
        page.close()
        page = None
        deadline = time.monotonic() + CLOSE_WAIT_SECONDS
        while True:
            ids = _native_page_ids(driver, timeout=10.0)
            if ids is None:
                raise AssertionError("the extension gave no page status (web_status)")
            if page_id not in ids:
                break
            if time.monotonic() >= deadline:
                raise AssertionError("the closed self-test page is still open in the extension")
            time.sleep(0.1)
        if any(getattr(item, "id", None) == page_id for item in driver.pages()):
            raise AssertionError("the driver still lists the closed self-test page")
        detail.update({"title": title, "token_survived": True, "generation": generation, "typed": True,
                       "big_result": len(big), "native_closed": True})
    finally:
        if page is not None:
            page.close()
        server.stop()
    detail["secs"] = round(time.monotonic() - started, 2)
    return detail
