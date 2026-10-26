"""Hotfix 3 issue 1: "The close menu button and slide to close is not working in some edge cases, I have to
backspace on my android".

Run from src/mobile (Flet venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix3_sheets.py

The tests drive the real Flet session with ``FlutterModel``, a frame-by-frame model of what the Flet 1.0.3
Flutter client does with ``page.views`` and ``page._dialogs`` (flet v1.0.3 ``view.dart``, ``bottom_sheet.dart``,
``alert_dialog.dart``, ``page.dart``, ``utils/misc.dart closeModalRoute``):

* dialogs (sheets, alert dialogs) render only inside the LAST View (view.dart: ``control.id ==
  pageViews.last.id``), keyed by control id, so a change of the top View rebuilds their element in another
  parent: a new State;
* a sheet's State keeps the modal route it pushed (``_sheetRoute``); ``open: false`` closes that route, and
  a new State without it closes nothing (``open != lastOpen && lastOpen`` with ``route == null``);
* ``Navigator.pages`` is keyed by View route; a pageless modal route stays attached to the page it was
  pushed above, so a View pushed later covers it, and it comes back when that View is popped;
* swipe-down works only when the sheet was shown with ``enableDrag`` (``draggable``); Android back pops the
  topmost route; a completed route sends ``open: false`` and ``dismiss`` to Python;
* ``invoke_method`` on a control without a method listener waits until a frame built it (control.dart
  ``invokeMethod``; ScrollableControl, which wraps every Column, registers it in initState): the views guard's
  frame probe (``dialogs.FRAME_PROBE_METHOD``, a Column in the page overlay, built in the last View only).

Frames run after Python's burst (``settle``): what a phone does when the close patch and the next patches
arrive before the next vsync.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
import types
from pathlib import Path
from typing import Any

import pytest

_TF_SPEC = importlib.util.spec_from_file_location("_glossarion_tf_helpers_hotfix3_sheets",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
TF = importlib.util.module_from_spec(_TF_SPEC)
_TF_SPEC.loader.exec_module(TF)
storage = TF.storage
app_env = TF.app_env
needs_flet = pytest.mark.skipif(importlib.util.find_spec("flet") is None or importlib.util.find_spec("msgpack") is None,
                                reason="flet/msgpack not installed")

ROUTE_FREE = ("SnackBar", "Banner")  # the ScaffoldMessenger shows these, not a route


@pytest.fixture(autouse=True)
def _restore_flet_hooks():
    """``install_close_buttons`` patches flet classes for the whole process; undo it after each test."""
    try:
        import flet as ft
        from flet.controls.base_page import BasePage
    except Exception:
        yield
        return
    saved = [(ft.BottomSheet, "before_update", ft.BottomSheet.__dict__.get("before_update")),
             (BasePage, "show_dialog", BasePage.__dict__.get("show_dialog")),
             (BasePage, "_remove_dialog", BasePage.__dict__.get("_remove_dialog"))]
    yield
    for cls, name, value in saved:
        if value is None:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, value)


# ==========================================================================
# the Flutter client model
# ==========================================================================


class _Route:
    def __init__(self, dialog: Any, drag: bool) -> None:
        self.dialog = dialog
        self.drag = drag
        self.done = False


def _children(control: Any) -> list:
    from flet.controls.base_control import BaseControl

    out = []
    for name in ("content", "title", "subtitle", "leading", "trailing"):
        child = getattr(control, name, None)
        if isinstance(child, BaseControl):
            out.append(child)
    for name in ("controls", "actions"):
        items = getattr(control, name, None)
        if isinstance(items, list):
            out.extend(c for c in items if isinstance(c, BaseControl))
    return out


def _contains(root: Any, target: Any) -> bool:
    if root is target:
        return True
    return any(_contains(child, target) for child in _children(root))


def _find(root: Any, predicate) -> Any:
    if predicate(root):
        return root
    for child in _children(root):
        found = _find(child, predicate)
        if found is not None:
            return found
    return None


def close_button(dialog: Any) -> Any:
    button = _find(dialog.content, lambda c: getattr(c, "key", None) == "sheet-close")
    assert button is not None, "the sheet has no Close Menu button"
    return button


class FlutterModel:
    """What the Flet 1.0.3 Flutter client does with page views and dialogs, one frame at a time."""

    def __init__(self, session: Any, conn: Any, *, deliver_dismiss: bool = True) -> None:
        self.session = session
        self.page = session.page
        self.deliver_dismiss = deliver_dismiss
        conn.confirms_dialog_dismiss = True  # a client that reports every dismissed dialog (Flet's own do)
        self.conn = conn
        self.built: set = set()  # overlay control ids a frame built (their method listeners exist)
        self.probe_calls: list = []  # (control id, call id) of frame probes waiting for their listener
        self._wrap_send(conn)
        self.client_open: dict = {}  # control id -> the client-only "_open" property
        self.states: dict = {}  # (view key, control id) -> {"route": _Route | None}
        self.nav: list = []  # Navigator history: [[page key, [pageless modal routes]]]
        self.popped: set = set()  # view routes the client popped, until Python drops them
        self.completed: list = []
        self.frames = 0

    def _wrap_send(self, conn: Any) -> None:
        from flet.messaging.protocol import MessageAction

        from glossarion_mobile.ui.components.dialogs import FRAME_PROBE_METHOD

        send = conn.send_message

        def send_message(message: Any) -> None:
            body = getattr(message, "body", None)
            if message.action == MessageAction.INVOKE_METHOD and getattr(body, "name", None) == FRAME_PROBE_METHOD:
                conn.messages.append(message)
                self.probe_calls.append((body.control_id, body.call_id))
                self._answer_probes()
                return
            send(message)

        conn.send_message = send_message

    def _answer_probes(self) -> None:
        waiting = []
        for cid, call_id in self.probe_calls:
            if cid in self.built:
                self.conn.loop.call_soon(self.session.handle_invoke_method_results, cid, call_id, None, None)
            else:
                waiting.append((cid, call_id))
        self.probe_calls = waiting

    # ---- frames ----------------------------------------------------------------------------------------

    def _keys(self) -> list:
        return [getattr(v, "route", None) or str(v._i) for v in self.page.views]

    def _dialogs(self) -> list:
        return [d for d in self.page._dialogs.controls if type(d).__name__ not in ROUTE_FREE]

    def frame(self) -> None:
        keys = self._keys()
        self.popped &= set(keys)
        effective = [k for k in keys if k not in self.popped]
        # page.dart: Navigator.pages keyed by route; a removed page takes its pageless routes with it
        old = {entry[0]: entry for entry in self.nav}
        for entry in self.nav:
            if entry[0] not in effective:
                for route in list(entry[1]):
                    self._complete(route)
        self.nav = [old.get(k) or [k, []] for k in effective]
        # view.dart: the dialogs are built inside the last View only
        last = keys[-1] if keys else None
        dialogs = self._dialogs()
        present = {(last, d._i) for d in dialogs}
        for key in list(self.states):
            if key not in present:
                del self.states[key]  # element unmounted: its State and the route it held are gone
        post = []
        for d in dialogs:
            state = self.states.setdefault((last, d._i), {"route": None})
            is_open, last_open = bool(d.open), self.client_open.get(d._i, False)
            if is_open and not last_open:
                self.client_open[d._i] = True
                post.append(("open", d, state))
            elif is_open != last_open and last_open:
                self.client_open[d._i] = False
                route, state["route"] = state["route"], None
                if route is not None:
                    post.append(("close", route, None))
        for kind, item, state in post:  # addPostFrameCallback
            if kind == "open":
                drag = type(item).__name__ == "BottomSheet" and bool(getattr(item, "draggable", False))
                route = _Route(item, drag)
                state["route"] = route
                self.nav[-1][1].append(route)
            else:
                self._complete(item)  # closeModalRoute: pop (current) or removeRoute
        if last is not None and last not in self.popped:  # the overlay is built in the last View only
            self.built |= {c._i for c in self.page.overlay}
        self.frames += 1
        self._answer_probes()

    def _complete(self, route: _Route) -> None:
        if route.done:
            return
        route.done = True
        for entry in self.nav:
            if route in entry[1]:
                entry[1].remove(route)
        self.completed.append(route)

    async def deliver(self) -> None:
        """A completed route: ``_open``/``open`` false and the ``dismiss`` event (bottom_sheet.dart ``.then``)."""
        done, self.completed = self.completed, []
        for route in done:
            d = route.dialog
            self.client_open[d._i] = False
            if not self.deliver_dismiss:
                continue
            try:
                self.session.apply_patch(d._i, {"open": False})
            except Exception:
                pass
            await self.session.dispatch_event(d._i, "dismiss", None)
            await self.session.after_event(d)

    async def settle(self, rounds: int = 12) -> None:
        """Let Python run its burst (spawned tasks too), then frames, until the client is idle."""
        for _ in range(rounds):
            for _ in range(5):
                await asyncio.sleep(0)
            self.frame()
            await self.deliver()

    async def wait_until(self, predicate, timeout: float = 4.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await self.settle(2)
            if predicate():
                return True
            await asyncio.sleep(0.02)
        return predicate()

    # ---- what the user sees and does -----------------------------------------------------------------------

    def top(self) -> tuple:
        entry = self.nav[-1]
        if entry[1]:
            return ("modal", entry[1][-1].dialog)
        return ("page", entry[0])

    def modals(self) -> list:
        return [route.dialog for entry in self.nav for route in entry[1]]

    def showing(self, dialog: Any) -> bool:
        return self.top() == ("modal", dialog)

    async def tap(self, control: Any) -> None:
        kind, what = self.top()
        if kind == "modal":
            assert _contains(what, control), "the control is not on the sheet on top of the screen"
        await self.session.dispatch_event(control._i, "click", None)
        await self.session.after_event(control)

    async def drag_down(self, dialog: Any) -> None:
        assert self.showing(dialog), "the sheet is not on top"
        route = self.nav[-1][1][-1]
        assert route.drag, "the sheet cannot be dragged down (enableDrag off)"
        self._complete(route)
        await self.deliver()

    async def back(self) -> None:
        entry = self.nav[-1]
        if entry[1]:
            self._complete(entry[1][-1])
            await self.deliver()
            return
        if len(self.nav) == 1:
            return
        key = entry[0]
        for route in list(entry[1]):
            self._complete(route)
        self.nav.pop()
        self.popped.add(key)
        await self.session.dispatch_event(self.page._i, "view_pop", {"route": key})
        await self.session.after_event(self.page)
        await self.deliver()


# ==========================================================================
# helpers
# ==========================================================================


async def _start_app(width: int = 412):
    main_module, conn, session, page, app = await TF._start("android", width)
    model = FlutterModel(session, conn)
    await model.settle()
    return conn, session, page, app, model


def _phone_session():
    conn, session = TF._fake_session("android")
    session.apply_page_patch({"width": 412, "height": 860})
    return conn, session, session.page


def _in_dialogs(page, dialog) -> bool:
    return dialog in page._dialogs.controls


# ==========================================================================
# reproductions on the real app
# ==========================================================================


@needs_flet
def test_menu_row_that_navigates_leaves_no_sheet_behind(app_env):
    """Drawer › Help › "Logs & diagnostics": the ActionSheet closes and the screen opens in the same burst.
    The close used to reach a sheet State in the next top View, which never pushed the route, so the sheet
    stayed under the Logs screen, came back on Back, and neither Close Menu nor a swipe closed it."""

    async def scenario():
        conn, session, page, app, model = await _start_app()
        try:
            await TF._wait(lambda: app.state.engine_ready)
            sheet = app._on_help()
            await model.settle()
            assert model.showing(sheet.dialog)
            logs = next(t for t in sheet.tiles if getattr(t.title, "value", None) == "Logs & diagnostics")
            await model.tap(logs)
            assert await model.wait_until(lambda: model.top()[0] == "page" and model.top()[1] != "/")
            assert model.modals() == [] and not _in_dialogs(page, sheet.dialog)
            while model.top()[0] == "page" and model.top()[1] != "/":  # Logs, then Settings, then the chat
                await model.back()
                await model.settle()
            assert model.top() == ("page", "/") and model.modals() == []
        finally:
            await TF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_navigation_while_a_sheet_is_open_closes_it_first(app_env):
    """A screen opened while a sheet is still open (a deep link, a notification, a sheet button that
    navigates without closing): the sheet closes before the new top View is installed, so nothing is left
    under the new screen."""

    async def scenario():
        conn, session, page, app, model = await _start_app()
        try:
            await TF._wait(lambda: app.state.engine_ready)
            sheet = app._on_help()
            await model.settle()
            assert model.showing(sheet.dialog)
            app.navigate_to("settings.about")
            assert await model.wait_until(lambda: model.top()[0] == "page" and model.top()[1] != "/")
            assert model.modals() == [] and not sheet.dialog.open and not _in_dialogs(page, sheet.dialog)
        finally:
            await TF._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# every kind of sheet closes with Close Menu and with a swipe
# ==========================================================================


def _sheet_kinds():
    """name -> factory(page) for every way the app builds a bottom sheet; returns (dialog, cancelled list)."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
    from glossarion_mobile.ui.components.info_sheet import InfoSheet

    def action(page):
        cancelled: list = []
        sheet = ActionSheet([ActionItem("Copy", lambda: None), ActionItem("Share", lambda: None)], title="File",
                            on_cancel=lambda: cancelled.append(1))
        sheet.show(page)
        return sheet.dialog, cancelled

    def info(page):
        sheet = InfoSheet(title="Why", body="Because.")
        sheet.show(page)
        return sheet.dialog, None

    def scroll(page):
        dialog = s.scroll_sheet("Options", [ft.Text(f"row {i}") for i in range(30)], actions=[ft.TextButton("Save")])
        page.show_dialog(dialog)
        return dialog, None

    def factory(page):
        dialog = s.bottom_sheet(s.sheet_frame(ft.Column([ft.Text("x")])))
        page.show_dialog(dialog)
        return dialog, None

    def raw(page):  # built outside the factory (map glossaries, review, headers, reader sheets ...)
        dialog = ft.BottomSheet(content=ft.Container(content=ft.Column([ft.Text("raw")]), padding=16),
                                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        page.show_dialog(dialog)
        return dialog, None

    def scrolling_list(page):  # a ListView body (the model sheet, map glossaries)
        dialog = ft.BottomSheet(content=ft.ListView(controls=[ft.Text(str(i)) for i in range(50)], expand=True),
                                scrollable=True, show_drag_handle=True)
        page.show_dialog(dialog)
        return dialog, None

    def manual(page):  # the Manual glossary sheet (chat, Force Manual Glossary)
        try:
            from glossarion_mobile.ui.sheets.manual_glossary import ManualGlossarySheet
        except Exception as exc:  # pragma: no cover - a venv without the shared modules
            pytest.skip(f"manual glossary sheet not importable: {exc}")
        sheet = ManualGlossarySheet()
        sheet.show(page)
        return sheet.dialog, None

    def plus(page):  # the composer's ＋ sheet (already draggable)
        try:
            from glossarion_mobile.ui.sheets.plus_sheet import PlusSheet
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"plus sheet not importable: {exc}")
        sheet = PlusSheet()
        sheet.show(page)
        return sheet.dialog, None

    return {"action": action, "info": info, "scroll": scroll, "factory": factory, "raw": raw, "list": scrolling_list,
            "manual": manual, "plus": plus}


@needs_flet
@pytest.mark.parametrize("kind", ["action", "info", "scroll", "factory", "raw", "list", "manual", "plus"])
@pytest.mark.parametrize("how", ["close_menu", "swipe", "back"])
def test_every_kind_of_sheet_closes(kind, how):
    from glossarion_mobile.ui.components import sheet as s

    s.install_close_buttons()

    async def scenario():
        conn, session, page = _phone_session()
        model = FlutterModel(session, conn)
        model.frame()
        dialog, cancelled = _sheet_kinds()[kind](page)
        await model.settle()
        assert model.showing(dialog)
        if how == "close_menu":
            await model.tap(close_button(dialog))
        elif how == "swipe":
            await model.drag_down(dialog)
        else:
            await model.back()
        await model.settle()
        assert model.modals() == [] and not dialog.open and not _in_dialogs(page, dialog)
        if cancelled is not None:
            assert cancelled == [1]  # closed without a row: the awaited choice is answered once
        # and it opens (and closes) again
        page.show_dialog(dialog)
        await model.settle()
        assert model.showing(dialog)
        await model.tap(close_button(dialog))
        await model.settle()
        assert model.modals() == []

    asyncio.run(scenario())


@needs_flet
def test_close_menu_closes_a_keyed_sheet_shown_again_before_its_dismiss():
    """A keyed sheet opened again before the first one's dismiss arrived (a quick second tap, the Model
    manager's file sheet): Flet freezes the new one in its keyed diff, and Close Menu raised "Frozen controls
    cannot be updated" (which Flet reports as a crashed session) instead of closing it."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.dialogs import close_dialog

    s.install_close_buttons()

    async def scenario():
        conn, session, page = _phone_session()
        model = FlutterModel(session, conn)
        model.frame()
        first = s.scroll_sheet("Local file", [ft.Text("one")], key="local-file-sheet")
        page.show_dialog(first)
        await model.settle()
        close_dialog(page, first)  # its dismiss is still on the way
        second = s.scroll_sheet("Local file", [ft.Text("two")], key="local-file-sheet")
        page.show_dialog(second)
        assert hasattr(second, "_frozen")  # what Flet 1.0.3 does
        close_button(second).on_click(types.SimpleNamespace(control=None, page=page))  # never raises
        assert not second.open
        await model.settle()
        assert model.modals() == []

    asyncio.run(scenario())


@needs_flet
def test_nested_and_rebuilt_sheets_close():
    """A sheet opened from a row of another sheet sits on top of it: Close Menu and a swipe close the top
    one, then the one under it. A sheet whose content is replaced after it was shown keeps a working
    Close Menu. A sheet that closes while the next one opens (the model sheet's choice opening the Manual
    glossary sheet) leaves only the new one."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.dialogs import close_dialog

    s.install_close_buttons()

    async def scenario():
        conn, session, page = _phone_session()
        model = FlutterModel(session, conn)
        model.frame()
        lower = s.scroll_sheet("Chat settings", [ft.Text("a")])
        page.show_dialog(lower)
        await model.settle()
        upper = s.scroll_sheet("Choose", [ft.Text("b")])
        page.show_dialog(upper)
        await model.settle()
        assert model.modals() == [lower, upper] and model.showing(upper)
        await model.tap(close_button(upper))
        await model.settle()
        assert model.showing(lower)
        # rebuilt content keeps the strip, and the button closes the sheet
        lower.content = s.sheet_frame(ft.Column([ft.Text("rebuilt")]))
        lower.update()
        await model.settle()
        await model.tap(close_button(lower))
        await model.settle()
        assert model.modals() == []
        # close one and open the next in one burst
        first = s.scroll_sheet("Model", [ft.Text("m")])
        page.show_dialog(first)
        await model.settle()
        second = s.scroll_sheet("Manual glossary", [ft.Text("g")])
        close_dialog(page, first)
        page.show_dialog(second)
        await model.settle()
        assert model.modals() == [second]
        await model.drag_down(second)
        await model.settle()
        assert model.modals() == []

    asyncio.run(scenario())


# ==========================================================================
# view changes wait for the dialogs on the old top View
# ==========================================================================


def _views_page():
    """A page with a root View and a FlutterModel; returns (conn, session, page, model, root, other)."""
    import flet as ft

    conn, session, page = _phone_session()
    root = ft.View(route="/", controls=[ft.Text("home")])
    page.views.clear()
    page.views.append(root)
    page.update()
    model = FlutterModel(session, conn)
    model.frame()
    other = ft.View(route="/other", controls=[ft.Text("other")])
    return conn, session, page, model, root, other


def _push(page, view):
    """What the shell does for a new top View (``AppShell._install_views``)."""
    from glossarion_mobile.ui.components.dialogs import change_views

    def apply():
        page.views.append(view)

    applied = change_views(page, apply, top=view)
    page.update()
    return applied


@needs_flet
def test_confirm_dialog_whose_action_opens_a_screen_leaves_nothing_behind():
    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.dialogs import ConfirmDialog

    s.install_close_buttons()

    async def scenario():
        conn, session, page, model, root, other = _views_page()
        dialog = ConfirmDialog(title="Open the book?", body="", on_confirm=lambda: _push(page, other))
        dialog.show(page)
        await model.settle()
        assert model.showing(dialog.dialog)
        await model.tap(dialog.confirm_button)
        assert await model.wait_until(lambda: model.top() == ("page", "/other"))
        assert model.modals() == [] and dialog.state == "closed"

    asyncio.run(scenario())


@needs_flet
def test_a_dialog_shown_while_a_view_change_waits_opens_on_the_new_screen():
    """Close a menu, open a screen and a sheet on it in one burst: the new sheet waits for the screen and is
    built on it (not on the screen under it, where it would be stranded)."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.dialogs import close_dialog

    s.install_close_buttons()

    async def scenario():
        conn, session, page, model, root, other = _views_page()
        menu = s.scroll_sheet("Menu", [ft.Text("row")])
        page.show_dialog(menu)
        await model.settle()
        close_dialog(page, menu)
        assert _push(page, other) is False  # waits for the menu's dismiss
        assert page.views[-1] is root
        follow = s.scroll_sheet("Translate", [ft.Text("t")])
        page.show_dialog(follow)
        assert follow.open and not _in_dialogs(page, follow)  # queued until the new screen is on top
        assert await model.wait_until(lambda: model.top() == ("modal", follow))
        assert model.nav[-1][0] == "/other" and model.modals() == [follow]
        await model.tap(close_button(follow))
        await model.settle()
        assert model.top() == ("page", "/other") and model.modals() == []

    asyncio.run(scenario())


@needs_flet
def test_a_view_change_does_not_wait_forever_for_a_lost_dismiss(monkeypatch):
    import flet as ft

    from glossarion_mobile.ui.components import dialogs
    from glossarion_mobile.ui.components import sheet as s

    s.install_close_buttons()
    monkeypatch.setattr(dialogs, "VIEWS_WAIT", 0.2)

    async def scenario():
        conn, session, page, model, root, other = _views_page()
        model.deliver_dismiss = False  # the client never reports it
        menu = s.scroll_sheet("Menu", [ft.Text("row")])
        page.show_dialog(menu)
        await model.settle()
        sent = len(conn.messages)
        assert _push(page, other) is False
        assert await model.wait_until(lambda: page.views[-1] is other, timeout=3.0)
        assert len(conn.messages) > sent  # the late view change was sent to the client
        # a dialog that never got its dismiss does not hold the next view change
        third = ft.View(route="/third", controls=[ft.Text("3")])
        assert _push(page, third) is True and page.views[-1] is third

    asyncio.run(scenario())


@needs_flet
def test_dialogs_shown_in_the_same_burst_as_a_view_change_never_strand_or_stall(monkeypatch):
    """A dialog shown and closed in one burst, and one shown just before a view change: either may already
    stand on the old top View (the phone draws a frame between any two messages) or never have been built
    (then no dismiss ever comes). The change closes them and waits for one client frame, not for dismissals;
    nothing is left on any screen, and the never-built ones leave page._dialogs with their on_dismiss."""
    import flet as ft

    from glossarion_mobile.ui.components import dialogs
    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.dialogs import close_dialog

    s.install_close_buttons()
    monkeypatch.setattr(dialogs, "VIEWS_WAIT", 0.3)

    async def scenario():
        conn, session, page, model, root, other = _views_page()
        dismissed: list = []
        quick = s.scroll_sheet("Quick", [ft.Text("q")])
        quick.on_dismiss = lambda e: dismissed.append("quick")
        page.show_dialog(quick)
        close_dialog(page, quick)
        fresh = s.scroll_sheet("Fresh", [ft.Text("f")])
        fresh.on_dismiss = lambda e: dismissed.append("fresh")
        page.show_dialog(fresh)
        started = time.monotonic()
        assert _push(page, other) is False  # one client frame first
        assert not fresh.open and page.views[-1] is root
        assert await model.wait_until(lambda: page.views[-1] is other and model.nav[-1][0] == "/other")
        assert time.monotonic() - started < dialogs.VIEWS_WAIT  # the frame, not a dismiss timeout
        assert model.modals() == []
        assert await model.wait_until(lambda: not _in_dialogs(page, quick) and not _in_dialogs(page, fresh))
        assert sorted(dismissed) == ["fresh", "quick"] and not page.overlay  # the probe is gone too

    asyncio.run(scenario())


@needs_flet
def test_a_dialog_the_client_built_just_before_a_view_change_is_closed_first():
    """The slow-phone side of the same burst: the client built the sheet on the old top View before the
    change came. Its close runs in the frame the change waits for, so the route is gone before the new screen
    is pushed (no ghost sheet under it), and its dismiss arrives as usual."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s

    s.install_close_buttons()

    async def scenario():
        conn, session, page, model, root, other = _views_page()
        sheet = s.scroll_sheet("Job", [ft.Text("j")])
        page.show_dialog(sheet)
        model.frame()  # a vsync between the sheet and the screen change: built on "/"
        assert model.showing(sheet)
        assert _push(page, other) is False
        assert await model.wait_until(lambda: model.nav[-1][0] == "/other")
        assert model.modals() == [] and [entry[1] for entry in model.nav] == [[], []]
        assert await model.wait_until(lambda: not _in_dialogs(page, sheet))

    asyncio.run(scenario())


@needs_flet
def test_without_a_client_that_reports_dismissals_views_change_at_once():
    """Host-test sessions (and a disconnected client) never send dismiss events: the guard stays out of
    the way and every view change applies immediately, as before."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s

    s.install_close_buttons()

    async def scenario():
        conn, session, page = _phone_session()
        root = ft.View(route="/", controls=[ft.Text("home")])
        page.views[:] = [root]
        page.update()
        menu = s.scroll_sheet("Menu", [ft.Text("row")])
        page.show_dialog(menu)
        await asyncio.sleep(0)
        other = ft.View(route="/other")
        assert _push(page, other) is True and page.views[-1] is other and menu.open
        later = s.scroll_sheet("Later", [ft.Text("l")])
        page.show_dialog(later)
        assert _in_dialogs(page, later)

    asyncio.run(scenario())


@needs_flet
def test_close_menu_closes_a_sheet_shown_in_the_tablet_side_panel():
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components import surface
    from glossarion_mobile.ui.responsive import SizeClass

    class Host:
        size_class = SizeClass.TABLET

        def __init__(self):
            self.owner = None

        def present(self, content, *, title, owner, on_close):
            self.owner = owner
            return True

        def hosts(self, owner):
            return owner is self.owner

        def dismiss(self, owner):
            if owner is self.owner:
                self.owner = None
                return True
            return False

    async def scenario():
        conn, session, page = _phone_session()
        host = Host()
        surface.register(page, host)
        try:
            dialog = s.with_close_button(s.scroll_sheet("Chat settings", [ft.Text("x")]))
            assert surface.present_sheet(page, dialog, title="Chat settings")
            close_button(dialog).on_click(types.SimpleNamespace(control=None, page=page))
            assert host.owner is None
        finally:
            surface.unregister(page, host)

    asyncio.run(scenario())


@needs_flet
def test_sheets_with_their_own_gestures_keep_drag_off():
    """Pan / zoom inside a sheet (the Reader's image viewer) and full-screen editors (typed text a stray swipe
    would discard) keep their close buttons only; every other sheet slides closed."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s

    s.install_close_buttons()
    viewer = ft.BottomSheet(content=ft.Container(content=ft.InteractiveViewer(content=ft.Text("img")), height=400),
                            show_drag_handle=True, open=True)
    viewer.before_update()
    editor = ft.BottomSheet(content=ft.SafeArea(content=ft.Column([ft.TextField(multiline=True)])), fullscreen=True,
                            open=True)
    editor.before_update()
    pinned = s.no_drag_dismiss(ft.BottomSheet(content=ft.Text("p"), open=True))
    pinned.before_update()
    menu = ft.BottomSheet(content=ft.Text("menu"), open=True)
    menu.before_update()
    assert not viewer.draggable and not editor.draggable and not pinned.draggable
    assert menu.draggable
    for dialog in (viewer, editor, pinned, menu):
        close_button(dialog)  # every one has Close Menu
    # a web page (the DeepSeek sign-in sheet): flet_webview's platform view gets only the drags no Flutter
    # recognizer claims, so a draggable sheet would take every vertical swipe (no scrolling, closed mid-login)
    fwv = pytest.importorskip("flet_webview")
    page = ft.Container(height=500, content=ft.Column([ft.Text("Sign in"),
                                                       ft.Container(content=fwv.WebView(url="https://example.invalid"),
                                                                    expand=True)]))
    factory = s.bottom_sheet(page, key="web-sheet")
    raw = ft.BottomSheet(content=ft.Container(content=fwv.WebView(url="https://example.invalid")), open=True)
    raw.before_update()
    factory.open = True
    factory.before_update()
    assert not factory.draggable and not raw.draggable
    close_button(factory)


@needs_flet
def test_a_failing_confirm_action_keeps_the_dialog_usable(caplog):
    """An exception in a confirm action used to propagate out of the click handler: Flet then reports a
    crashed session and the whole app sits behind its error screen. The dialog stays open, both buttons
    work again, and Cancel closes it."""
    from glossarion_mobile.ui.components.dialogs import ConfirmDialog

    def boom():
        raise OSError("disk full")

    async def scenario():
        conn, session, page = _phone_session()
        model = FlutterModel(session, conn)
        model.frame()
        dialog = ConfirmDialog(title="Delete book", body="Delete?", destructive=True, on_confirm=boom)
        dialog.show(page)
        await model.settle()
        await model.tap(dialog.confirm_button)  # session.dispatch_event: an escaping exception is a crash
        await model.settle()
        assert not any(m.action.name == "SESSION_CRASHED" for m in conn.messages)
        assert dialog.dialog.open and dialog.state == "idle" and model.showing(dialog.dialog)
        assert not dialog.confirm_button.disabled and not dialog.cancel_button.disabled
        assert "disk full" in caplog.text
        await model.tap(dialog.cancel_button)
        await model.settle()
        assert model.modals() == [] and dialog.result is False

    asyncio.run(scenario())


@needs_flet
def test_close_menu_closes_its_own_sheet_under_a_snackbar():
    """A row of the sheet showed a SnackBar (also a dialog control in Flet 1.0.3, the newest one): Close Menu
    closes the sheet itself, never the SnackBar (``page.pop_dialog()`` would pick the SnackBar)."""
    import flet as ft

    from glossarion_mobile.ui.components import sheet as s
    from glossarion_mobile.ui.components.dialogs import show_snackbar

    s.install_close_buttons()

    async def scenario():
        conn, session, page = _phone_session()
        model = FlutterModel(session, conn)
        model.frame()
        dialog = s.scroll_sheet("Keys", [ft.Text("k")])
        page.show_dialog(dialog)
        await model.settle()
        bar = show_snackbar(page, "Copied")
        await model.settle()
        await model.tap(close_button(dialog))
        await model.settle()
        assert model.modals() == [] and not dialog.open and bar.open

    asyncio.run(scenario())
