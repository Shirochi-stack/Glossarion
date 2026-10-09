"""Acceptance test for the owner's request #17 (device report 2026-10-08, U8 APK):

    "Key settings must NOT be in Chat settings; add a dedicated Keys button next to the Settings
    button in the drawer / tablet sidebar footer (it opens the existing API keys / Multi-Key Manager
    screen); the sidebar footer (status chip, Settings, Keys, Help) is pinned to the bottom of the
    drawer and the tablet sidebar - only the chat list above it scrolls."

What the owner had on the U8 APK: a footer of status chip · Settings · Help only (no way to the keys
from the drawer), and a phone drawer whose whole content (footer included) scrolled with a drag,
because ``AppShell`` sized the drawer box to ``height - status bar`` inside Flutter's
``NavigationDrawer`` list (``SafeArea(bottom: false)`` + ``ListView`` padded by the navigation bar).

This test drives the REAL app (``main.main`` on the fake Flet session of tests_host, the real
``ChatFeature`` loading a desktop-format ``direct_text_chats.json`` of 200 chats through the shared
``direct_text_store``) the way the owner uses it:

* phone (3-button navigation, and a small phone with gesture navigation): open the drawer with 200
  chats; the footer is exactly status chip · Settings · API keys · Help, it is the last child of the
  drawer's non-scrolling column, outside the one scrolling part (the chat list holding all 200 rows),
  no ancestor of it scrolls, the ``NavigationDrawer``'s own list cannot scroll (the box is its
  viewport minus its end padding) and the fixed parts (header, search, chips, footer) fit in the box
  with room for the chat list, so the footer is on screen without scrolling;
* tap 🔑 (the click goes through the Flet session to the button's handler): the drawer closes and the
  Multi-Key Manager (``/settings/keys``, ``KeysScreen`` on the main pool) is on screen; Android back
  closes it (it never stays on the Keys screen and never leaves the app) and the back chain ends on
  the chat, which is unchanged;
* tablet (1000 x 800): the persistent sidebar shows the same pinned footer with 200 chats; 🔑 opens
  the Multi-Key Manager in the main area while the sidebar stays as it was; the system back
  (``on_confirm_pop`` of the one root View) pops back to the chat;
* "Android back closes the keys screen back to the chat" (the acceptance criterion of item 17): ONE
  back press on the Keys screen opened with 🔑 from the chat shows that chat again, on the phone (a
  ``view_pop`` of the Keys View) and on the tablet (``on_confirm_pop``). The 🔑 sits next to Settings
  as a destination of its own, and every other drawer destination (Library, Jobs, Glossaries, Tools,
  Settings) is one back press from the chat; Settings home, which the owner never opened, is not a
  stop on the way. (UI_SPEC §1.2's generic rule "a drawer destination ... start[s] from the route's
  static parents" and the builder's §1.3 line "Back returns through Settings, like the other
  drawer-opened settings pages" give the Keys -> Settings -> chat chain this criterion rejects; the
  two scenarios above accept either chain, so a failure here is about the extra stop only.)
* Chat settings (⋯ › Chat settings, as a bottom sheet on the phone and in the SidePanel on the
  tablet), in both scopes (This chat, All chats): no key control (KeyField, password field, key pool
  tile), no text about API keys / key pools / the Multi-Key Manager, no handler that opens
  ``settings.keys``, and none of the config keys it writes is an API key or key pool setting.

Real data is never touched: FLET_APP_STORAGE_* (so HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
GLOSSARION_DATA_DIR, CONFIG_FILE and the chat history) come from ``app_env`` under ``tmp_path``;
USERPROFILE, APPDATA and LOCALAPPDATA are pointed there too, and GLOSSARION_HTTP_LOG=0.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue17.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import os
import re
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed"),
    pytest.mark.skipif(not (_has("requests") and _has("bs4")), reason="the real chat feature drives the backend"),
]

# The real-app helpers (fake Flet session, fixtures, _wait / _routes / _walk) live in test_ui_foundations.py.
_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_devfix17",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
uf = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(uf)
storage = uf.storage
app_env = uf.app_env

CHATS = 200
STATUS_BAR = 24.0
TABLET = (1000.0, 800.0)
# (id, width, height, navigation bar): a 6.3" phone with 3-button navigation, a small one with gestures
PHONES = (("pixel-3-button", 412.0, 860.0, 48.0), ("small-gesture", 360.0, 740.0, 24.0))

# Upper bounds (dp) of the drawer's fixed rows at 100 % text, Material 3: the header row (56 dp minimum,
# 48 dp buttons), the dense filled search field (+ its 2 dp progress bar), the chip row (32 dp chips in
# 48 dp tap targets), the footer row (56 dp minimum).
HEADER_MAX, SEARCH_MAX, CHIPS_MAX = 56.0, 58.0, 48.0
#: The chat list must keep at least this many 48 dp rows on screen above the footer.
MIN_VISIBLE_ROWS = 4

KEY_TEXT = re.compile(r"api[\s_-]?keys?\b|multi[\s_-]?key|key[\s_-]?(pool|manager|rotation)|\bkeys?\s*✓|🔑",
                      re.IGNORECASE)
KEY_CONTROL_KEY = re.compile(r"api[-_]?key|multi[-_]?key|key[-_]?pool|(^|[-_.])keys?($|[-_.])", re.IGNORECASE)
KEY_ROUTES = ("settings.keys", "/settings/keys")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "winhome"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


# ==========================================================================
# Helpers
# ==========================================================================


def _seed_history(storage_dirs: dict, count: int) -> list:
    """A desktop ``direct_text_chats.json`` v2 history of ``count`` chats, one turn each, the newest
    first and three hours apart (Today, Yesterday, Previous 7 days and older month groups)."""
    from glossarion_mobile.state.chat_store_adapter import default_history_path

    path = Path(default_history_path())
    data = Path(storage_dirs["data"]).resolve()
    assert data in path.resolve().parents, f"the chat history must live in the test storage, not {path}"
    now = time.time()
    sessions = []
    for n in range(count):
        stamp = datetime.fromtimestamp(now - n * 3 * 3600, tz=timezone.utc).isoformat()
        sessions.append({
            "id": 100 + n,
            "title": f"Novel chat {n:03d}",
            "messages": [
                ["user", f"Translate line {n}"],
                ["assistant", f"Line {n}", "", "Token summary", "", "Request 1", {"created_at": stamp}],
            ],
            "draft": "",
            "attachment": None,
            "output_folder": "",
            "output_folder_name": "",
            "next_output_index": 1,
            "expanded": [],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 2, "current_chat_id": 100, "sessions": sessions}), encoding="utf-8")
    return [str(s["id"]) for s in sessions]


def _media(top: float, nav: float) -> dict:
    """``page.media`` as the Flutter client sends it (edge-to-edge: the system bars are insets)."""
    return {
        "padding": {"left": 0, "top": top, "right": 0, "bottom": nav},
        "view_padding": {"left": 0, "top": top, "right": 0, "bottom": nav},
        "view_insets": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        "device_pixel_ratio": 2.625,
        "orientation": "portrait",
    }


async def _launch(width: float, height: float, media: dict | None = None):
    """``main.main`` on a fake session that registered with this size and these insets."""
    main_module = uf._load_main_module()
    conn, session = uf._fake_session("android")
    patch = {"width": width, "height": height}
    if media is not None:
        patch["media"] = media
    session.apply_page_patch(patch)
    page = session.page
    await main_module.main(page)
    await session.after_event(page)
    return conn, session, page, page.data


async def _history_loaded(app, ids: list) -> None:
    from glossarion_mobile.state.chat_store_adapter import ChatStoreAdapter

    def loaded() -> bool:
        return isinstance(app.state.chats, ChatStoreAdapter) and len(app.drawer.chat_rows) == len(ids)

    assert await uf._wait(loaded, timeout=60), (type(app.state.chats).__name__, len(app.drawer.chat_rows))
    assert set(app.drawer.chat_rows) == set(ids)


def _child_controls(control) -> list:
    """Every control a Flet control holds (any dataclass field: content, controls, options, segments,
    leading, trailing, title, label, actions, ...), in field order."""
    from flet.controls.base_control import BaseControl

    out = []
    if not dataclasses.is_dataclass(control):
        return out
    for field in dataclasses.fields(control):
        if field.name.startswith("_") or field.name == "data":
            continue
        value = getattr(control, field.name, None)
        if isinstance(value, BaseControl):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(v for v in value if isinstance(v, BaseControl))
    return out


def _tree(root) -> list:
    out, stack, seen = [], [root], set()
    while stack:
        control = stack.pop()
        if control is None or id(control) in seen:
            continue
        seen.add(id(control))
        out.append(control)
        stack.extend(reversed(_child_controls(control)))
    return out


def _path(root, target, seen=None):
    """The controls from ``root`` down to ``target`` (both included), or None."""
    seen = set() if seen is None else seen
    if root is target:
        return [root]
    if id(root) in seen:
        return None
    seen.add(id(root))
    for child in _child_controls(root):
        found = _path(child, target, seen)
        if found is not None:
            return [root, *found]
    return None


def _has_control(controls, target) -> bool:
    """``target`` (this very object) is among ``controls``: Flet controls compare by value."""
    return any(c is target for c in controls)


def _scrolls(control) -> bool:
    import flet as ft

    if isinstance(control, (ft.ListView, ft.GridView)):
        return True
    return getattr(control, "scroll", None) not in (None, False)


def _strings(control) -> list:
    """The user-visible strings of one control (text, labels, hints, tooltips, string content)."""
    out = []
    if not dataclasses.is_dataclass(control):
        return out
    for field in dataclasses.fields(control):
        if field.name.startswith("_") or field.name in ("data", "key"):
            continue
        value = getattr(control, field.name, None)
        if isinstance(value, str):
            out.append(value)
    return out


def _handler_constants(fn, depth: int = 0) -> list:
    """String constants of an event handler's code (lambdas and the functions/methods they wrap)."""
    out: list = []
    if fn is None or depth > 3:
        return out
    fn = getattr(fn, "__func__", fn)
    code = getattr(fn, "__code__", None)
    if code is None:
        return out
    stack = [code]
    while stack:
        current = stack.pop()
        for const in current.co_consts:
            if isinstance(const, str):
                out.append(const)
            elif isinstance(const, types.CodeType):
                stack.append(const)
    return out


def _key_violations(root) -> list:
    """Every key control, key text or keys-route handler under ``root``."""
    import flet as ft

    problems = []
    for control in _tree(root):
        kind = type(control)
        if kind.__name__ in ("KeyField", "KeyEditor", "KeysScreen") or kind.__module__.endswith(
                ("ui.screens.keys", "ui.screens.key_editor")):
            problems.append(f"key control {kind.__name__}")
        if isinstance(control, ft.TextField) and (control.password or control.can_reveal_password):
            problems.append(f"password field {control.label or control.hint_text!r}")
        key = getattr(control, "key", None)
        key = getattr(key, "value", key)  # ScrollKey(value=<config key>) on settings tiles
        if isinstance(key, str) and KEY_CONTROL_KEY.search(key) and not key.startswith("setting-"):
            problems.append(f"control key {key!r}")
        for text in _strings(control):
            if KEY_TEXT.search(text):
                problems.append(f"{kind.__name__} text {text[:80]!r}")
        for field in dataclasses.fields(control) if dataclasses.is_dataclass(control) else ():
            if field.name.startswith("on_") and callable(getattr(control, field.name, None)):
                consts = _handler_constants(getattr(control, field.name))
                if any(c in KEY_ROUTES or c.startswith("/settings/keys") for c in consts):
                    problems.append(f"{kind.__name__}.{field.name} opens the keys screen")
    return problems


def _assert_pinned_footer(app, container) -> None:
    """status chip · Settings · API keys · Help, the last child of the drawer's non-scrolling column, outside
    the one scrolling part (the chat list with every chat row), and nothing between ``container`` (the
    NavigationDrawer / the tablet sidebar) and the footer scrolls."""
    import flet as ft

    drawer = app.drawer
    row = drawer.footer.content
    assert isinstance(row, ft.Row) and not _scrolls(row)
    buttons = [c for c in row.controls if isinstance(c, ft.IconButton)]
    assert row.controls[0].content is drawer.status_chip and row.controls[0].expand
    assert len(buttons) == 3 and all(b is want for b, want in
                                     zip(buttons, (drawer.settings_button, drawer.keys_button, drawer.help_button)))
    assert [b.tooltip for b in buttons] == ["Settings", "API keys", "Help"]
    assert drawer.keys_button.icon == ft.Icons.KEY and drawer.keys_button.visible is not False
    assert not drawer.keys_button.disabled and callable(drawer.keys_button.on_click)
    assert drawer.footer.visible is not False

    column = drawer.content
    assert isinstance(column, ft.Column) and not _scrolls(column)
    assert column.controls[-1] is drawer.footer
    scrolling = [c for c in column.controls if _scrolls(c)]
    assert len(scrolling) == 1 and scrolling[0] is drawer.body and drawer.body.expand
    in_body = {id(c) for c in _tree(drawer.body)}
    assert id(drawer.footer) not in in_body and id(drawer.keys_button) not in in_body
    assert all(id(row_) in in_body for row_ in drawer.chat_rows.values())  # every chat row is in the list

    path = _path(container, drawer.footer)
    assert path is not None, "the footer is not under the drawer / sidebar"
    assert not _has_control(path, drawer.body)
    assert [type(c).__name__ for c in path if _scrolls(c)] == [], "an ancestor of the footer scrolls"
    assert _path(container, drawer.keys_button) is not None


def _fixed_rows(drawer) -> float:
    """Height (upper bound) of the drawer column's fixed rows: header, search, chips, footer and the gaps."""
    from glossarion_mobile.ui import tokens

    gaps = drawer.content.spacing * (len(drawer.content.controls) - 1)
    return HEADER_MAX + SEARCH_MAX + CHIPS_MAX + tokens.SIZES["drawer_footer"] + gaps


def _assert_footer_fits_width(drawer, width: float) -> None:
    """The footer row fits the drawer / sidebar width: the 3 icon buttons plus a usable status chip."""
    from glossarion_mobile.ui import tokens

    padding = drawer.footer.padding
    fixed = 3 * tokens.SIZES["hit_target"] + float(padding.left or 0) + float(padding.right or 0)
    assert width - fixed >= 96, f"the status chip gets only {width - fixed} dp next to Settings · Keys · Help"


def _confirm_pops(conn) -> list:
    from flet.messaging.protocol import MessageAction

    return [(m.body.args or {}).get("should_pop") for m in conn.messages
            if m.action == MessageAction.INVOKE_METHOD and m.body.name == "confirm_pop"]


# ==========================================================================
# Phone: the drawer
# ==========================================================================


@pytest.mark.parametrize("name, width, height, nav", PHONES, ids=[p[0] for p in PHONES])
def test_phone_drawer_with_200_chats_footer_pinned_and_keys_open_the_key_manager(app_env, storage, name, width,
                                                                                  height, nav):
    import flet as ft

    from glossarion_mobile.ui.screens.keys import KeysScreen

    ids = _seed_history(storage, CHATS)

    async def scenario():
        conn, session, page, app = await _launch(width, height, _media(STATUS_BAR, nav))
        try:
            shell, drawer = app.shell, app.drawer
            assert not shell.tablet
            await _history_loaded(app, ids)
            await uf._wait(lambda: app.state.engine_ready, timeout=60)
            current = app.state.current_chat.value

            # the owner opens the drawer (☰)
            conn.messages.clear()
            await app.open_drawer()
            assert "show_drawer" in conn.invoked()
            nav_drawer = page.views[0].drawer
            assert isinstance(nav_drawer, ft.NavigationDrawer) and nav_drawer is shell.nav_drawer
            assert len(nav_drawer.controls) == 1 and nav_drawer.controls[0] is shell.drawer_box
            assert shell.drawer_box.content is drawer.content
            _assert_pinned_footer(app, nav_drawer)

            # Flutter's NavigationDrawer = SafeArea(bottom: false) > ListView padded at its end by the
            # navigation bar: its viewport is the screen minus the status bar. The drawer box plus that
            # padding must not exceed it, or a drag scrolls the whole drawer, footer included (the U8 box,
            # height - status bar, overshot it by exactly the navigation bar).
            viewport = height - STATUS_BAR
            box = float(shell.drawer_box.height)
            assert box + nav <= viewport, f"the NavigationDrawer list scrolls by {box + nav - viewport} dp"
            # ... and the fixed rows leave the chat list room, so the footer ends inside the box (on screen)
            room = box - _fixed_rows(drawer)
            assert room >= MIN_VISIBLE_ROWS * 48, f"only {room} dp left for the chat list above the footer"
            _assert_footer_fits_width(drawer, float(nav_drawer.width))

            # 🔑 API keys: one tap closes the drawer and opens the Multi-Key Manager
            conn.messages.clear()
            await session.dispatch_event(drawer.keys_button._i, "click", None)
            assert await uf._wait(lambda: isinstance(shell.top_screen, KeysScreen)), \
                f"🔑 did not open the Multi-Key Manager: {uf._routes(page)}"
            assert await uf._wait(lambda: uf._routes(page)[-1] == "/settings/keys"), uf._routes(page)
            screen = shell.top_screen
            assert screen.pool == "main" and screen.title == "API keys"
            assert "close_drawer" in conn.invoked()
            assert page.views[-1].route == "/settings/keys" and _has_control(uf._walk(page.views[-1]), screen.body)

            # Android back closes the Multi-Key Manager and the back chain ends on the chat (at most one stop,
            # Settings, in between: whether that stop may exist is test_one_android_back_on_the_keys_screen_...)
            chain = []
            while uf._routes(page) != ["/"] and len(chain) < 4:
                await session.dispatch_event(page._i, "view_pop", {"route": uf._routes(page)[-1]})
                chain.append(uf._routes(page)[-1])
                if len(chain) == 1:
                    assert not isinstance(shell.top_screen, KeysScreen), "back did not close the Keys screen"
                    assert "/settings/keys" not in uf._routes(page)
                    assert all(not isinstance(e.screen, KeysScreen) for e in shell.stack)
            assert uf._routes(page) == ["/"] and shell.stack == [], uf._routes(page)
            assert chain in (["/"], ["/settings", "/"]), f"back chain from the Keys screen: {chain}"
            assert page.views[0] is shell.root_view and shell.current_route == "/"
            assert app.state.current_chat.value == current  # the same chat, untouched
            # the footer is still pinned after the round trip
            _assert_pinned_footer(app, page.views[0].drawer)
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# Tablet: the persistent sidebar
# ==========================================================================


def test_tablet_sidebar_with_200_chats_footer_pinned_and_keys_open_in_the_main_area(app_env, storage):
    from glossarion_mobile.ui.screens.keys import KeysScreen

    ids = _seed_history(storage, CHATS)
    width, height = TABLET

    async def scenario():
        conn, session, page, app = await _launch(width, height, _media(STATUS_BAR, 48.0))
        try:
            shell, drawer = app.shell, app.drawer
            assert shell.tablet and page.views[0].drawer is None and uf._routes(page) == ["/"]
            await _history_loaded(app, ids)
            await uf._wait(lambda: app.state.engine_ready, timeout=60)
            current = app.state.current_chat.value
            _assert_pinned_footer(app, shell.sidebar)
            assert _has_control(uf._walk(page.views[0]), shell.sidebar)
            # the sidebar fills the screen height (Row STRETCH, SafeArea, expanding Column) and the
            # fixed rows plus the docked JobStrip slot leave the chat list room above the footer
            column = shell.sidebar.content.content
            assert not _scrolls(column) and column.expand and column.controls[0].expand
            room = height - STATUS_BAR - 48.0 - 60.0 - _fixed_rows(drawer)
            assert room >= MIN_VISIBLE_ROWS * 48, f"only {room} dp left for the chat list above the footer"
            _assert_footer_fits_width(drawer, float(shell.sidebar.width))

            sidebar_before = (shell.sidebar, drawer.footer, drawer.keys_button)
            await session.dispatch_event(drawer.keys_button._i, "click", None)
            assert await uf._wait(lambda: isinstance(shell.top_screen, KeysScreen)), \
                f"🔑 did not open the Multi-Key Manager: {[e.route for e in shell.stack]}"
            screen = shell.top_screen
            assert screen.pool == "main" and uf._routes(page) == ["/"]  # main area, not a pushed View
            assert _has_control(uf._walk(shell.main_area), screen.body)
            assert all(a is b for a, b in zip((shell.sidebar, drawer.footer, drawer.keys_button), sidebar_before))
            _assert_pinned_footer(app, shell.sidebar)  # the sidebar stays as it was

            # the system back on the one root View closes the Keys screen; the chain ends on the chat
            root = page.views[0]
            assert root.can_pop is False and callable(root.on_confirm_pop)
            conn.messages.clear()
            chain = []
            while shell.stack and len(chain) < 4:
                await session.dispatch_event(root._i, "confirm_pop", None)
                assert await uf._wait(lambda: len(_confirm_pops(conn)) == len(chain) + 1)
                chain.append(shell.current_route)
                if len(chain) == 1:
                    assert not isinstance(shell.top_screen, KeysScreen), "back did not close the Keys screen"
            assert shell.stack == [] and _confirm_pops(conn) == [False] * len(chain)  # never left the app
            assert chain in (["/"], ["/settings", "/"]), f"back chain from the Keys screen: {chain}"
            assert page.views[0].can_pop is True  # the chat is in the main area again
            assert shell._top_panel_entry() is None and shell.current_route == "/"
            assert app.state.current_chat.value == current
            _assert_pinned_footer(app, shell.sidebar)
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# Android back on the Keys screen: straight back to the chat
# ==========================================================================


@pytest.mark.parametrize("form", ["phone", "tablet"])
def test_one_android_back_on_the_keys_screen_returns_to_the_chat(app_env, storage, form):
    """The owner is on a chat, taps 🔑 in the drawer / sidebar footer, looks at the keys and presses the
    Android back button once: the Keys screen closes and the same chat is on screen again."""
    from glossarion_mobile.ui.screens.keys import KeysScreen

    ids = _seed_history(storage, CHATS)
    width, height = (412.0, 860.0) if form == "phone" else TABLET

    async def scenario():
        conn, session, page, app = await _launch(width, height, _media(STATUS_BAR, 48.0))
        try:
            shell, drawer = app.shell, app.drawer
            assert shell.tablet is (form == "tablet")
            await _history_loaded(app, ids)
            await uf._wait(lambda: app.state.engine_ready, timeout=60)
            current = app.state.current_chat.value
            assert current in ids and shell.stack == [] and shell.current_route == "/"

            if form == "phone":
                await app.open_drawer()
            await session.dispatch_event(drawer.keys_button._i, "click", None)
            assert await uf._wait(lambda: isinstance(shell.top_screen, KeysScreen)), \
                f"🔑 did not open the Multi-Key Manager: {[e.route for e in shell.stack]}"
            opened = [e.route for e in shell.stack]

            # ONE Android back
            conn.messages.clear()
            if form == "phone":
                assert await uf._wait(lambda: uf._routes(page)[-1] == "/settings/keys"), uf._routes(page)
                await session.dispatch_event(page._i, "view_pop", {"route": "/settings/keys"})
                assert await uf._wait(lambda: "/settings/keys" not in uf._routes(page)), \
                    "back did not close the Keys screen"
                shown = uf._routes(page)[-1]
            else:
                root = page.views[0]
                await session.dispatch_event(root._i, "confirm_pop", None)
                assert await uf._wait(lambda: len(_confirm_pops(conn)) == 1)
                assert _confirm_pops(conn) == [False], "back left the app"
                assert await uf._wait(lambda: not isinstance(shell.top_screen, KeysScreen)), \
                    "back did not close the Keys screen"
                shown = shell.current_route
            assert shown == "/" and shell.stack == [], (
                f"one Android back on the Keys screen (opened with 🔑 from the chat) shows {shown!r}, not the "
                f"chat: 🔑 opened the stack {opened} (drawer navigation restarts from the static parents of "
                f"settings.keys), so a second back is needed to get back to the chat")
            if form == "phone":
                assert uf._routes(page) == ["/"] and page.views[0] is shell.root_view
            else:
                assert shell._top_panel_entry() is None and page.views[0].can_pop is True
            assert app.state.current_chat.value == current  # the same chat, untouched
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# Chat settings: no key settings
# ==========================================================================


@pytest.mark.parametrize("form", ["phone", "tablet"])
def test_chat_settings_have_no_key_settings(app_env, storage, form):
    import key_pool_service
    from glossarion_mobile.ui.sheets.chat_settings import CHAT_SETTING_KEYS

    ids = _seed_history(storage, 3)
    width, height = (412.0, 860.0) if form == "phone" else TABLET

    # what Chat settings writes to config.json is no API key / key pool setting
    pool_keys = {"api_key"}
    for spec in key_pool_service.POOL_SPECS.values():
        for field in ("config_key", "toggle_key"):
            if spec.get(field):
                pool_keys.add(spec[field])
    assert "multi_api_keys" in pool_keys and "use_multi_api_keys" in pool_keys
    assert not pool_keys & set(CHAT_SETTING_KEYS.values())

    async def scenario():
        conn, session, page, app = await _launch(width, height, _media(STATUS_BAR, 48.0))
        try:
            await _history_loaded(app, ids)
            assert await uf._wait(lambda: app.chat_view.bound, timeout=30)
            # ⋯ › Chat settings, as the owner opens it
            item = app.chat_view.header.menu_items["chat_settings"]
            await session.dispatch_event(item._i, "click", None)
            assert await uf._wait(lambda: getattr(app.chat_view, "settings_sheet", None) is not None)
            sheet = app.chat_view.settings_sheet
            if form == "phone":
                assert sheet.dialog.open and not sheet.in_panel
                host = sheet.dialog
            else:
                assert sheet.in_panel and app.shell.side_panel.is_open
                host = app.shell.side_panel.control
            assert _has_control(_tree(host), sheet.column)

            for scope in ("chat", "global"):  # This chat, All chats
                if sheet.scope != scope:
                    sheet.scope_button.selected = [scope]  # the client sets ``selected``, then fires change
                    await session.dispatch_event(sheet.scope_button._i, "change", None)
                assert sheet.scope == scope
                tree = _tree(host)
                rows = [c.key for c in tree if isinstance(getattr(c, "key", None), str)
                        and c.key.startswith("setting-")]
                assert "setting-model" in rows and "setting-profile" in rows, rows  # the sheet is built
                assert _key_violations(host) == [], (scope, _key_violations(host))
                assert not [r for r in rows if r[len("setting-"):] in pool_keys], rows

            # the keys have their own place: the 🔑 button next to Settings in the drawer / sidebar footer
            footer = app.drawer.footer.content.controls
            assert footer.index(app.drawer.keys_button) == footer.index(app.drawer.settings_button) + 1
            sheet.close()

            # Settings › Direct Text (the Settings form of Chat settings › All chats) has none either
            await app.navigate("/settings/s/direct_text.settings")
            section = app.shell.top_screen
            assert type(section).__name__ == "SectionPage", type(section).__name__
            body = section.get_body()
            tiles = {getattr(c.key, "value", c.key) for c in _tree(body) if getattr(c, "key", None) is not None}
            assert "direct_text_output_mode" in tiles, sorted(map(str, tiles))  # the section is built
            assert not pool_keys & tiles and not getattr(section, "pool_tiles", [])
            assert _key_violations(body) == [], _key_violations(body)
        finally:
            await uf._stop(app)

    asyncio.run(scenario())
