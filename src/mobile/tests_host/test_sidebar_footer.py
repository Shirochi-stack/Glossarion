"""Owner request 17 (device report 2026-10-08, U8 APK): the drawer / sidebar footer.

    "Key settings must NOT live in Chat settings; add a dedicated Keys button next to the Settings
    button in the drawer / sidebar footer (it opens the API keys / Multi-Key Manager screen); the
    sidebar footer (status chip, Settings, Keys, Help) is pinned to the bottom of the drawer and the
    tablet sidebar - never scrolled away with the chat list."

What the owner saw on the phone: Flutter's ``NavigationDrawer`` (Flet 1.0.3 has no other phone drawer)
puts its children in its own ``ListView`` under ``SafeArea(bottom: false)``. That list is the screen
height minus the status bar and pads its end by the navigation bar. ``AppShell`` sized the drawer
content box to ``height - status bar``, so the list was one navigation bar taller than its viewport:
a drag on the drawer (with few chats the inner chat list cannot scroll, so the drag goes to the outer
list) moved the whole content, footer included, and at rest the footer sat under the navigation bar.

These tests drive the REAL app (``main.main`` on the fake Flet session of tests_host, the shared
``test_ui_foundations`` helpers) on a phone with a 3-button navigation bar and on a tablet:

* the footer row is exactly status chip · Settings · API keys · Help, below the drawer's only
  scrolling part (the chat list, ``ChatDrawer.body``), and no ancestor of the footer scrolls, on the
  phone drawer and on the tablet sidebar, with a chat list far longer than the screen;
* the phone drawer box is the ``NavigationDrawer`` list's viewport minus its end padding, so that list
  cannot scroll (pins the owner's complaint; it fails on the U1 box size), for 3-button and gesture
  navigation, with the keyboard up, after a navigation-mode change and after a rotation;
* the Keys button opens the Multi-Key Manager (``/settings/keys``, ``KeysScreen``) and closes the
  drawer; one Android back then returns to the chat (a drawer shortcut: the Multi-Key Manager is
  the whole stack), while Settings › API keys opened from Settings home still goes back to Settings;
  on a tablet the screen opens in the main area and the system back (``on_confirm_pop``) pops the
  same way.

Real data is never touched: FLET_APP_STORAGE_* (so HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
CONFIG_FILE) come from ``app_env`` under ``tmp_path``; USERPROFILE, APPDATA and LOCALAPPDATA are pointed
there too, and GLOSSARION_HTTP_LOG=0.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_sidebar_footer.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
needs_backend = pytest.mark.skipif(not (_has("requests") and _has("bs4")), reason="the real features drive the backend")

# The real-app helpers (fake Flet session, _start / _wait / _routes / _walk) live in test_ui_foundations.py.
_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_sidebar_footer",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
uf = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(uf)
storage = uf.storage
app_env = uf.app_env

PHONE_HEIGHT = 860.0
STATUS_BAR = 24.0
THREE_BUTTON_NAV = 48.0
GESTURE_NAV = 24.0
KEYBOARD = 300.0


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "winhome"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


def _media(top: float, nav: float, *, keyboard: float = 0.0, orientation: str = "portrait") -> dict:
    """``page.media`` as the Flutter client sends it: ``padding`` = ``view_padding`` minus the keyboard."""
    return {
        "padding": {"left": 0, "top": top, "right": 0, "bottom": max(0.0, nav - keyboard)},
        "view_padding": {"left": 0, "top": top, "right": 0, "bottom": nav},
        "view_insets": {"left": 0, "top": 0, "right": 0, "bottom": keyboard},
        "device_pixel_ratio": 2.625,
        "orientation": orientation,
    }


async def _media_change(session, media: dict) -> None:
    """What the client does on a media change: patch ``page.media``, then fire ``media_change``."""
    session.apply_page_patch({"media": media})
    await session.dispatch_event(session.page._i, "media_change", media)


def _outer_list_overflow(page, box_height: float) -> float:
    """How far Flutter's NavigationDrawer list can scroll its one child (the drawer box): the box plus
    the list's end padding (``MediaQuery.padding.bottom``, kept by ``SafeArea(bottom: false)``) minus
    the list's viewport (the page height minus ``padding.top``, which the SafeArea consumes)."""
    media = page.media
    return box_height + float(media.padding.bottom) - (float(page.height) - float(media.padding.top))


def _children(control):
    out = []
    for attr in ("content", "controls", "leading", "trailing", "title"):
        child = getattr(control, attr, None)
        if isinstance(child, list):
            out.extend(c for c in child if c is not None and hasattr(c, "_i"))
        elif child is not None and hasattr(child, "_i"):
            out.append(child)
    return out


def _path(root, target, depth: int = 0):
    """The controls from ``root`` down to ``target`` (both included), or None."""
    if root is target:
        return [root]
    if depth > 40:
        return None
    for child in _children(root):
        found = _path(child, target, depth + 1)
        if found is not None:
            return [root, *found]
    return None


def _scrolls(control) -> bool:
    import flet as ft

    if isinstance(control, (ft.ListView, ft.GridView)):
        return True
    return getattr(control, "scroll", None) not in (None, False)


def _many_chats(app, count: int = 80) -> None:
    """A chat list far longer than the screen, swapped in the way ChatFeature installs its index."""
    from glossarion_mobile.state.chat_index import ChatSummary, InMemoryChatIndex

    now = time.time()
    chats = [ChatSummary(cid=str(1000 + i), title=f"Chat {i}", updated_at=now - i * 3600) for i in range(count)]
    drawer = app.drawer
    drawer.detach()
    app.state.chats = InMemoryChatIndex(chats)
    drawer.attach()
    drawer._changed()


def _assert_pinned_footer(drawer, root) -> None:
    """The footer is status chip · Settings · API keys · Help, the last child of the drawer's
    non-scrolling Column, outside the chat list, and nothing between ``root`` and it scrolls."""
    import flet as ft

    row = drawer.footer.content
    assert isinstance(row, ft.Row)
    assert row.controls[0].content is drawer.status_chip
    assert row.controls[1:4] == [drawer.settings_button, drawer.keys_button, drawer.help_button]
    assert [b.tooltip for b in row.controls[1:4]] == ["Settings", "API keys", "Help"]
    content = drawer.content
    assert isinstance(content, ft.Column) and not _scrolls(content)
    assert content.controls[-1] is drawer.footer
    assert [c for c in content.controls if _scrolls(c)] == [drawer.body] and drawer.body.expand
    assert drawer.footer not in uf._walk(drawer.body)
    path = _path(root, drawer.footer)
    assert path is not None, "the footer is not under the drawer / sidebar"
    assert drawer.body not in path
    assert not [type(c).__name__ for c in path if _scrolls(c)], "an ancestor of the footer scrolls"


# ==========================================================================
# The drawer component alone
# ==========================================================================


def test_keys_button_sits_between_settings_and_help_and_navigates_like_a_destination():
    import flet as ft

    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.router import ROUTES_BY_NAME
    from glossarion_mobile.ui.shell.drawer import KEYS_ROUTE, ChatDrawer

    went = []
    drawer = ChatDrawer(state=AppState(), on_navigate=went.append, clock=lambda: 200.0)
    _assert_pinned_footer(drawer, drawer.content)
    assert drawer.keys_button.icon == ft.Icons.KEY and drawer.keys_button.key == "drawer-keys"
    assert KEYS_ROUTE == "settings.keys" and ROUTES_BY_NAME[KEYS_ROUTE].pattern == "/settings/keys"
    drawer.keys_button.on_click(None)
    assert went == [KEYS_ROUTE]

    # an explicit handler wins (the app may route it differently)
    calls = []
    custom = ChatDrawer(state=AppState(), on_navigate=went.append, on_keys=calls.append, clock=lambda: 200.0)
    custom.keys_button.on_click("event")
    assert calls == ["event"] and went == [KEYS_ROUTE]


# ==========================================================================
# Phone: the NavigationDrawer box (the owner's complaint)
# ==========================================================================


@needs_backend
def test_phone_drawer_footer_never_scrolls_with_the_chat_list(app_env, monkeypatch):
    """The owner's complaint: with a navigation bar, the drawer box must fit the NavigationDrawer's
    own list exactly, so only the chat list scrolls and the footer stays pinned above the bar."""
    import flet as ft

    real_session = uf._fake_session

    def session_with_insets(platform):  # the client registers with its size and media (3-button nav)
        conn, session = real_session(platform)
        session.apply_page_patch({"height": PHONE_HEIGHT, "media": _media(STATUS_BAR, THREE_BUTTON_NAV)})
        return conn, session

    monkeypatch.setattr(uf, "_fake_session", session_with_insets)

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=412)
        try:
            shell, drawer = app.shell, app.drawer
            _many_chats(app)
            assert await uf._wait(lambda: len(drawer.chat_rows) > 40)  # far more rows than the screen holds
            nav = page.views[0].drawer
            assert isinstance(nav, ft.NavigationDrawer) and nav.controls == [shell.drawer_box]
            assert shell.drawer_box.content is drawer.content
            _assert_pinned_footer(drawer, nav)

            # 3-button navigation: the box is the list viewport minus the list's end padding
            assert shell.drawer_box.height == PHONE_HEIGHT - STATUS_BAR - THREE_BUTTON_NAV
            assert _outer_list_overflow(page, shell.drawer_box.height) <= 0
            # (the U1 box, height - status bar, let the whole drawer scroll by the navigation bar)
            assert _outer_list_overflow(page, PHONE_HEIGHT - STATUS_BAR) == THREE_BUTTON_NAV

            # the keyboard (drawer search) changes padding.bottom, never the box
            box = shell.drawer_box.height
            await _media_change(session, _media(STATUS_BAR, THREE_BUTTON_NAV, keyboard=KEYBOARD))
            assert shell.drawer_box.height == box and _outer_list_overflow(page, box) <= 0
            await _media_change(session, _media(STATUS_BAR, THREE_BUTTON_NAV))
            assert shell.drawer_box.height == box

            # switching to gesture navigation re-fits the box without a resize event
            await _media_change(session, _media(STATUS_BAR, GESTURE_NAV))
            assert shell.drawer_box.height == PHONE_HEIGHT - STATUS_BAR - GESTURE_NAV
            assert _outer_list_overflow(page, shell.drawer_box.height) == 0

            # rotation: media first, then the size (as the client sends them); landscape keeps the drawer
            landscape = _media(STATUS_BAR, 0.0, orientation="landscape")
            session.apply_page_patch({"media": landscape, "width": PHONE_HEIGHT, "height": 412})
            await session.dispatch_event(page._i, "media_change", landscape)
            await session.dispatch_event(page._i, "resize", {"width": PHONE_HEIGHT, "height": 412})
            assert not shell.tablet and page.views[0].drawer is not None
            assert shell.drawer_box.height == 412 - STATUS_BAR
            assert _outer_list_overflow(page, shell.drawer_box.height) <= 0
            _assert_pinned_footer(drawer, page.views[0].drawer)
            assert conn.bytes_sent > 0
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


@needs_backend
def test_phone_keys_button_opens_the_key_manager_and_back_returns_to_the_chat(app_env):
    from glossarion_mobile.ui.screens.keys import KeysScreen

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=412)
        try:
            await uf._wait(lambda: app.state.engine_ready, timeout=60)
            conn.messages.clear()
            await session.dispatch_event(app.drawer.keys_button._i, "click", None)
            assert await uf._wait(lambda: uf._routes(page) == ["/", "/settings/keys"])
            screen = app.shell.top_screen
            assert isinstance(screen, KeysScreen) and screen.pool == "main"
            assert "close_drawer" in conn.invoked()  # one tap navigates and closes the drawer (§1.3)
            # Android back: Keys -> the chat (a shortcut: the Multi-Key Manager is the whole stack)
            await session.dispatch_event(page._i, "view_pop", {"route": "/settings/keys"})
            assert uf._routes(page) == ["/"] and app.shell.stack == []
            # Settings › API keys opened from Settings home still goes back to Settings
            await session.dispatch_event(app.drawer.settings_button._i, "click", None)
            assert await uf._wait(lambda: uf._routes(page) == ["/", "/settings"])
            app.navigate_to("settings.keys")
            assert await uf._wait(lambda: uf._routes(page) == ["/", "/settings", "/settings/keys"])
            await session.dispatch_event(page._i, "view_pop", {"route": "/settings/keys"})
            assert uf._routes(page) == ["/", "/settings"]
            await session.dispatch_event(page._i, "view_pop", {"route": "/settings"})
            assert uf._routes(page) == ["/"] and app.shell.stack == []
            # the Settings button next to it still opens the Settings home
            await session.dispatch_event(app.drawer.settings_button._i, "click", None)
            assert await uf._wait(lambda: uf._routes(page) == ["/", "/settings"])
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# Tablet: the persistent sidebar
# ==========================================================================


@needs_backend
def test_tablet_sidebar_footer_is_pinned_and_keys_open_in_the_main_area(app_env):
    from flet.messaging.protocol import MessageAction

    from glossarion_mobile.ui.screens.keys import KeysScreen

    def confirm_calls(conn):
        return [(m.body.args or {}).get("should_pop") for m in conn.messages
                if m.action == MessageAction.INVOKE_METHOD and m.body.name == "confirm_pop"]

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=1000)
        try:
            shell, drawer = app.shell, app.drawer
            assert shell.tablet and page.views[0].drawer is None
            await uf._wait(lambda: app.state.engine_ready, timeout=60)
            _many_chats(app)
            assert await uf._wait(lambda: len(drawer.chat_rows) > 40)
            _assert_pinned_footer(drawer, shell.sidebar)
            # the JobStrip slot docks below the footer, outside the drawer content (UI_SPEC §1.7)
            column = shell.sidebar.content.content
            assert column.controls[-1] is shell.sidebar_strip_slot and not _scrolls(column)

            await session.dispatch_event(drawer.keys_button._i, "click", None)
            assert await uf._wait(lambda: isinstance(shell.top_screen, KeysScreen))
            assert [e.route for e in shell.stack] == ["/settings/keys"] and uf._routes(page) == ["/"]
            assert shell.top_screen.body in uf._walk(shell.main_area)
            _assert_pinned_footer(drawer, shell.sidebar)  # the sidebar stays as it was
            # the system back on the one root View pops the main area: Keys -> the chat
            root = page.views[0]
            assert root.can_pop is False and callable(root.on_confirm_pop)
            conn.messages.clear()
            await session.dispatch_event(root._i, "confirm_pop", None)
            assert await uf._wait(lambda: confirm_calls(conn) == [False])
            assert shell.stack == [] and page.views[0].can_pop is True
        finally:
            await uf._stop(app)

    asyncio.run(scenario())
