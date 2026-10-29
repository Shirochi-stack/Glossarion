"""v9.15.7 hotfix 4, owner (2026-10-29, phone, Welcome): "it took 5 seconds to respond to my tap ... and after I
did login, the button vanished"; "the buttons should be animated during sign in and update when signed with a
checkmark, not stay silent and disappear".

Causes: the Welcome's DeepSeek tap first loaded DeepSeek in a hidden page and waited up to 10 s for a session
before the sheet opened, with no sign on the button; a signed-in provider's button was replaced by a full-width
"signed in · Change model" row.

Fix: the slow check runs only when the ``.signed_in`` marker says a sign-in was seen before
(``deepseek_session_marked``); the tapped button spins (and ignores taps) while its sign-in is under way, goes
back to normal when the sheet closes without one, and stays in its half-row cell with a checkmark once signed in.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_4.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ft = pytest.importorskip("flet")

from glossarion_mobile.ui.router import parse_route  # noqa: E402
from glossarion_mobile.ui.screens import accounts  # noqa: E402
from glossarion_mobile.ui.screens.welcome import WelcomeScreen  # noqa: E402


def _keys(root) -> dict:
    out, stack = {}, [root]
    while stack:
        node = stack.pop()
        if getattr(node, "key", None):
            out[node.key] = node
        for name in ("controls", "content"):
            child = getattr(node, name, None)
            if isinstance(child, list):
                stack.extend(c for c in child if isinstance(c, ft.Control))
            elif isinstance(child, ft.Control):
                stack.append(child)
    return out


def _state(button) -> str:
    return str(button.content.content.key).split(":", 1)[0]  # AnimatedSwitcher > Row keyed "<state>:<key>"


@pytest.fixture
def welcome(monkeypatch):
    from glossarion_mobile.services import webview_bridge

    monkeypatch.setattr(webview_bridge, "availability", lambda: (True, ""))
    screen = WelcomeScreen(parse_route("/welcome"), oauth=object())
    screen.get_body()
    screen.page = object()

    async def check_model(provider):  # the model poll is not under test here
        return None

    screen.check_model = check_model
    return screen


async def _tick():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def test_deepseek_tap_without_a_marker_opens_the_sheet_at_once_and_spins(welcome, monkeypatch):
    shown, checked = [], []

    async def present(run_io=None):
        checked.append(True)
        return False

    monkeypatch.setattr(accounts, "deepseek_session_marked", lambda: False)
    monkeypatch.setattr(accounts, "deepseek_session_present", present)
    monkeypatch.setattr(accounts, "show_deepseek_sign_in",
                        lambda page, **kw: shown.append(kw))

    async def scenario():
        await welcome._deepseek_sign_in(welcome.page)
        assert checked == [] and len(shown) == 1  # no hidden-page wait before the sheet
        button = _keys(welcome.page_area.content)["welcome-signin-authds"]
        assert _state(button) == "busy" and button.disabled and button.on_click is None
        assert isinstance(button.content, ft.AnimatedSwitcher)
        # a second tap while it is busy opens nothing more
        await welcome._deepseek_sign_in(welcome.page)
        assert len(shown) == 1
        # closed without signing in: tappable again
        shown[0]["on_closed"]()
        button = _keys(welcome.page_area.content)["welcome-signin-authds"]
        assert _state(button) == "idle" and not button.disabled and button.on_click is not None
        # signed in from the sheet: the same cell, now with a checkmark
        shown[0]["on_signed_in"]()
        await _tick()
        keys = _keys(welcome.page_area.content)
        assert "welcome-signin-authds" not in keys and _state(keys["welcome-use-authds"]) == "done"
        cell = next(c for c in keys["welcome-sign-ins"].controls if c.content.key == "welcome-signed-authds")
        assert cell.col == {"xs": 6}

    asyncio.run(scenario())


def test_deepseek_tap_with_a_marker_checks_first(welcome, monkeypatch):
    shown = []

    async def present(run_io=None):
        return True

    monkeypatch.setattr(accounts, "deepseek_session_marked", lambda: True)
    monkeypatch.setattr(accounts, "deepseek_session_present", present)
    monkeypatch.setattr(accounts, "show_deepseek_sign_in", lambda page, **kw: shown.append(kw))

    async def scenario():
        await welcome._deepseek_sign_in(welcome.page)
        await _tick()

    asyncio.run(scenario())
    assert shown == [] and "authds" in welcome.provider_status and not welcome.signing_in


def test_login_sheet_reports_a_close_once_and_not_after_a_sign_in(monkeypatch):
    class Page:
        def show_dialog(self, dialog):
            dialog.open = True

    monkeypatch.setattr(accounts, "LoginPanel", lambda oauth, **kw: ft.Text(""))  # the panel is not under test
    closed, done = [], []

    async def scenario():
        sheet = accounts.LoginSheet(object(), provider="authgem", autostart=False,
                                    on_done=done.append, on_closed=lambda: closed.append(True))
        sheet.show(Page())
        sheet.close()
        sheet.dialog.on_dismiss(None)  # swiped away / closed: reported once
        await _tick()
        assert closed == [True]
        other = accounts.LoginSheet(object(), provider="authgem", autostart=False,
                                    on_done=done.append, on_closed=lambda: closed.append(True))
        other.show(Page())
        other._done({"email": "a@b"})
        other.dialog.on_dismiss(None)  # a sign-in is not a close
        await _tick()
        assert closed == [True] and done == [{"email": "a@b"}]

    asyncio.run(scenario())


def test_welcome_oauth_button_spins_until_the_sheet_closes(welcome, monkeypatch):
    sheets = []

    class Sheet:
        def __init__(self, oauth, **kw):
            self.kw = kw
            sheets.append(self)

        def show(self, page):
            pass

    monkeypatch.setattr(accounts, "LoginSheet", Sheet)

    async def scenario():
        welcome.open_sign_in("authgem")
        assert _state(_keys(welcome.page_area.content)["welcome-signin-authgem"]) == "busy"
        sheets[0].kw["on_closed"]()
        assert _state(_keys(welcome.page_area.content)["welcome-signin-authgem"]) == "idle"
        welcome.open_sign_in("authgem")
        sheets[1].kw["on_done"]({"email": "gem@example.com"})
        await _tick()
        assert _state(_keys(welcome.page_area.content)["welcome-use-authgem"]) == "done"

    asyncio.run(scenario())
