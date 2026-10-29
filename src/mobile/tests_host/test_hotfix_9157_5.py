"""v9.15.7 hotfix 5, owner (2026-10-29, phone, a fresh install):

1. "The welcome screen takes 5 seconds to appear, before it the first thing I see is the chat window": the Welcome
   was opened after every feature had been installed (Library, Reader, Glossary, Tools, Manga, ...), with the chat
   on screen meanwhile.
2. "The permission requests for notification and disable battery optimization appear 1 second after [the
   Welcome]... they might mistakenly reject it": the launch dialogs were queued 2 s after the jobs feature
   installed, whether or not the Welcome was up.
3. "The grok login sometimes fails on my first attempt... my account is clearly logged, just not caught": the
   first page opened signs the xAI website out and then continues to the approval page; signing back in can end
   elsewhere, and "Open verification page" opened that sign-out page again.

Fixes: a "Setting up Glossarion" cover over the chat on a home launch without a finished Welcome, the Welcome
opened as soon as it can be built (after the pages feature), then the cover is lifted; the launch dialogs wait
while the Welcome is needed (it asks on its Permissions step) and come up when it closes; a device sign-in's
"Open verification page" opens the approval page itself.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_5.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

_TF_SPEC = importlib.util.spec_from_file_location("_glossarion_hf9157_5_foundations",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
TF = importlib.util.module_from_spec(_TF_SPEC)
_TF_SPEC.loader.exec_module(TF)
app_env = TF.app_env
storage = TF.storage
needs_flet = pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")


def _record_startup(monkeypatch, seen: dict) -> None:
    """What was on screen when the jobs feature and the Library installed."""
    from glossarion_mobile.app import GlossarionApp

    jobs, library = GlossarionApp._install_jobs, GlossarionApp._install_library

    async def install_jobs(self):
        seen["cover_at_jobs"] = getattr(self, "_first_run_cover", None) is not None
        view = self.shell.root_view
        seen["home_shown_at_jobs"] = any(c.visible for c in [view.appbar, *view.controls] if c is not None)
        await jobs(self)

    async def install_library(self):
        seen["route_at_library"] = TF._routes(self.page)[-1]
        seen["cover_at_library"] = getattr(self, "_first_run_cover", None) is not None
        await library(self)

    monkeypatch.setattr(GlossarionApp, "_install_jobs", install_jobs)
    monkeypatch.setattr(GlossarionApp, "_install_library", install_library)


@needs_flet
def test_first_run_covers_the_chat_and_opens_the_welcome_before_the_later_features(app_env, monkeypatch):
    seen: dict = {}
    _record_startup(monkeypatch, seen)

    async def scenario():
        _m, conn, session, page, app = await TF._start("android", first_run=True)
        try:
            assert seen["cover_at_jobs"] is True  # the chat never shows on its own first
            assert seen["home_shown_at_jobs"] is False  # nor under the Welcome's slide (the overlay is per view)
            assert seen["route_at_library"] == "/welcome" and seen["cover_at_library"] is False
            assert await TF._wait(lambda: not any(getattr(c, "key", None) == "first-run-cover" for c in page.overlay))
            assert TF._routes(page)[-1] == "/welcome"
            view = app.shell.root_view
            assert all(c.visible for c in [view.appbar, *view.controls] if c is not None)  # back under the Welcome
        finally:
            app.jobs.close()
            await TF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_returning_user_gets_no_cover_and_a_finished_welcome_is_remembered(app_env, monkeypatch):
    seen: dict = {}
    _record_startup(monkeypatch, seen)

    async def scenario():
        _m, conn, session, page, app = await TF._start("android")
        try:
            assert seen["cover_at_jobs"] is False and seen["route_at_library"] == "/"
            assert seen["home_shown_at_jobs"] is True
            assert TF._routes(page)[-1] == "/"
        finally:
            app.jobs.close()
            await TF._stop(app)

    asyncio.run(scenario())


class _Permissions:
    def __init__(self) -> None:
        self.asked: list = []

    async def request(self, name: str) -> str:
        self.asked.append(name)
        return "granted"

    async def status(self, name: str) -> str:
        return "denied"


class _Prefs(dict):
    def set(self, key, value):
        self[key] = value


def _background():
    from glossarion_mobile.services.background import BackgroundExecution

    permissions = _Permissions()
    return BackgroundExecution(object(), platform="android", prefs=_Prefs(), permissions=permissions), permissions


def test_launch_permission_dialogs_wait_for_the_welcome():
    background, permissions = _background()
    asyncio.run(background.ask_permissions_on_launch(0, defer=lambda: True))
    assert permissions.asked == []  # the Welcome is up: nothing over it
    asyncio.run(background.ask_permissions_on_launch(0, defer=lambda: False))
    assert permissions.asked == ["NOTIFICATION", "IGNORE_BATTERY_OPTIMIZATIONS"]


def test_jobs_feature_defers_while_the_welcome_is_needed():
    import types

    from glossarion_mobile.ui.screens.jobs import _welcome_pending

    app = types.SimpleNamespace(chat_feature=None, _first_run_cover=object())
    assert _welcome_pending(app)  # the chat feature is not installed yet, the first-run cover is up
    app._first_run_cover = None
    assert not _welcome_pending(app)
    app.chat_feature = types.SimpleNamespace(needs_welcome=lambda: True)
    assert _welcome_pending(app)
    app.chat_feature = types.SimpleNamespace(needs_welcome=lambda: False)
    assert not _welcome_pending(app)


def test_welcome_done_asks_the_held_back_dialogs(app_env):
    """Finishing (or skipping) the Welcome brings up the launch dialogs it held back."""
    from glossarion_mobile.ui.chat.integration import WELCOME_PERMISSIONS_DELAY

    async def scenario():
        _m, conn, session, page, app = await TF._start("android", first_run=True)
        try:
            asked: list = []

            async def ask(delay=0.0, defer=None):
                asked.append(delay)

            app.jobs.background.ask_permissions_on_launch = ask
            screen = app.chat_feature._welcome_screen
            screen.skip()
            assert await TF._wait(lambda: asked == [WELCOME_PERMISSIONS_DELAY])
        finally:
            app.jobs.close()
            await TF._stop(app)

    asyncio.run(scenario())


def test_grok_open_verification_page_opens_the_approval_page_not_the_sign_out(tmp_path):
    from glossarion_mobile.services.oauth import OAuthBridge, SignInState

    opened: list = []
    bridge = OAuthBridge(auth_modules={}, opener=types_opener(opened), timeout=20, safe_root=str(tmp_path))
    signed_out = "https://accounts.x.ai/sign-out?redirect=oauth2-provider&return_to=%2Foauth2%2Fdevice"
    bridge.state = SignInState(provider="authgrok", step="waiting", auth_url=signed_out, user_code="WXYZ-1234",
                               verification_uri="https://accounts.x.ai/oauth2/device",
                               approve_url="https://accounts.x.ai/oauth2/device?user_code=WXYZ-1234")
    assert bridge.reopen_browser()
    assert opened == ["https://accounts.x.ai/oauth2/device?user_code=WXYZ-1234"]
    # a loopback sign-in (no device code) still reopens its own page
    bridge.state = SignInState(provider="authgpt", step="waiting", auth_url="https://auth.example/authorize")
    assert bridge.reopen_browser() and opened[-1] == "https://auth.example/authorize"


def types_opener(opened: list):
    class Opener:
        def open(self, url):
            opened.append(url)
            return True

        def __call__(self, url):
            return self.open(url)

    return Opener()
