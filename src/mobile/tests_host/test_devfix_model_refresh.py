"""Owner issue 2 (model provider menu): "'Refresh this provider' and 'Refresh online models' don't have any
visual indicator and don't seem to work. The desktop app's search bar has an animated border indicator."

Diagnosis (checked on the real app below, before the fix): both items DID poll (``ModelCatalogService.refresh``
off the loop, the shared ``model_options`` poll, the list republished), but the user saw none of it:

* the only feedback was ``notify`` = a SnackBar, which Flutter draws in the page Scaffold UNDER the modal
  ModelSheet (BottomSheet) that stays open over most of the screen, so the "N models found" / error line
  never showed;
* the only progress hint was a Shimmer on the provider's group header, usually far down the list, and keyed
  by the group (``authgpt``) while the poll tag is the catalog provider (``authgpt:2``), so numbered account
  slots never shimmered; the ⋯ item and the row action stayed as they were while polling;
* an explicit refresh that met the sheet's own 24 h auto-poll of the same provider was dropped
  ("A catalog poll is already running", again only in the hidden SnackBar).

The fix (``ui/sheets/model_sheet.py``, ``ui/components/poll_border.py``): the desktop's animated poll border
(``translator_gui._ModelCatalogPollBorder``: cyan / violet sweep turning round the field) on the sheet's
search field while any catalog poll runs; the ⋯ item reads "Refreshing online models…" (disabled) and the
row action "Refreshing this provider…" (disabled, with its reason) while the provider polls; the status line
under the chips says what polls, then "✓ <provider>: N models found · M new" or "⚠ <provider>: <why>"; the
indicator stops in ``finally`` (errors included) and when the sheet closes; an explicit refresh waits for a
running poll of the same provider, then polls.

These tests start the REAL app on the fake Flet session (``test_ui_foundations._start``) and click the real
controls through ``session.dispatch_event``; only ``model_options.refresh_provider_model_catalogs`` is a fake
(offline, held on a ``threading.Event`` gate so every "while polling" check waits on the poll, never on time).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_model_refresh.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any

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


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_model_refresh",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
uf = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(uf)
storage = uf.storage
app_env = uf.app_env

MODEL = "gpt-4o"
NEW_MODEL = "gpt-refresh-new-1"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No traffic beyond loopback."""
    loopback = ("127.0.0.1", "::1", "localhost")
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def connect(self, address):
        if isinstance(address, tuple) and address and address[0] not in loopback:
            raise OSError(f"network disabled in tests ({address})")
        return real_connect(self, address)

    def create_connection(address, *args, **kwargs):
        if address and address[0] not in loopback:
            raise OSError(f"network disabled in tests ({address})")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)


class Polls:
    """``model_options.refresh_provider_model_catalogs`` offline. Each call waits on ``gate`` (a test step
    checks the "polling" state, then releases it); ``fail`` raises instead of answering."""

    def __init__(self, mo) -> None:
        self.mo = mo
        self.calls: list = []
        self.gate = threading.Event()
        self.started = threading.Event()
        self.fail: Any = None

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs.get("only_provider"))
        self.started.set()
        assert self.gate.wait(30), "the test never released the poll"
        if self.fail is not None:
            raise self.fail
        only = kwargs.get("only_provider") or "openai"
        models = list(self.mo.get_model_options()) + [NEW_MODEL]
        return self.mo.ModelCatalogRefreshResult(
            models=models, provider_models={only: [MODEL, NEW_MODEL]},
            statuses={only: "online (2 models)"}, requested_provider=kwargs.get("only_provider"))


@pytest.fixture
def polls(app_env, monkeypatch):
    import model_options as mo

    os.makedirs(os.path.dirname(os.environ["CONFIG_FILE"]), exist_ok=True)
    with open(os.environ["CONFIG_FILE"], "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL}, fh)
    fake = Polls(mo)
    monkeypatch.setattr(mo, "refresh_provider_model_catalogs", fake)
    yield fake
    fake.gate.set()


async def _open_sheet():
    _main, conn, session, page, app = await uf._start("android")
    feature = app.models_keys
    assert await uf._wait(lambda: feature.catalog.snapshot.loaded, 30), "the catalog never loaded"
    sheet = app.chat_view.open_model_sheet("model")
    assert await uf._wait(lambda: sheet.dialog.open and not sheet.closed, 10)
    assert sheet.current["model"] == MODEL
    return session, page, app, feature.catalog, sheet


async def _shutdown(app) -> None:
    from glossarion_mobile.services import model_catalog as mc
    from glossarion_mobile.ui.sheets import model_sheet as ms

    ms.install_sheet_env(None)
    mc.set_default_service(None)
    await uf._stop(app)


async def _until_started(polls: Polls) -> bool:
    return await uf._wait(polls.started.is_set, 10)


def _run(coro) -> None:
    asyncio.run(coro)


# ---- the ⋯ "Refresh online models" item ---------------------------------------------------------------


def test_menu_refresh_shows_the_animated_border_then_the_outcome(polls):
    from flet import SweepGradient

    async def scenario() -> None:
        session, page, app, catalog, sheet = await _open_sheet()
        try:
            border = sheet.search_border
            assert not border.active and border.control.gradient is None
            assert sheet.refresh_item.content == "Refresh online models" and not sheet.refresh_item.disabled
            # the owner's tap: ⋯ › Refresh online models (a real click event on the menu item)
            await session.dispatch_event(sheet.refresh_item._i, "click", None)
            assert await _until_started(polls)
            assert polls.calls == ["openai"]
            # while the poll runs: the search border turns, the item says so and is disabled, the line says what polls
            assert await uf._wait(lambda: border.active and border.ticks >= 3, 10), "the poll border never turned"
            assert border.spinning and isinstance(border.control.gradient, SweepGradient)
            assert sheet.refresh_item.content == "Refreshing online models…" and sheet.refresh_item.disabled
            assert sheet.status_text.visible and "Refreshing online models" in sheet.status_text.value
            assert catalog.snapshot.is_polling("openai")
            # a second tap meanwhile does not start another poll
            sheet.refresh_selected_provider()
            polls.gate.set()
            assert await uf._wait(lambda: not border.active, 10), "the poll border kept turning"
            assert await uf._wait(lambda: not border.spinning, 5)
            assert polls.calls == ["openai"]
            assert border.control.gradient is None
            assert sheet.refresh_item.content == "Refresh online models" and not sheet.refresh_item.disabled
            assert sheet.status_text.visible and "2 models found" in sheet.status_text.value
            assert re.search(r"· [12] new$", sheet.status_text.value), sheet.status_text.value
            # the result reached the open sheet: the new model is a row (searched for, like the owner would)
            sheet.set_query(NEW_MODEL)
            assert NEW_MODEL in sheet.rows
            assert NEW_MODEL in catalog.snapshot.models
        finally:
            polls.gate.set()
            await _shutdown(app)

    _run(scenario())


def test_a_failed_refresh_stops_the_border_and_says_why(polls):
    async def scenario() -> None:
        session, page, app, catalog, sheet = await _open_sheet()
        try:
            polls.fail = RuntimeError("provider unreachable (test)")
            await session.dispatch_event(sheet.refresh_item._i, "click", None)
            assert await _until_started(polls)
            assert await uf._wait(lambda: sheet.search_border.active, 5)
            polls.gate.set()
            assert await uf._wait(lambda: not sheet.search_border.active, 10), "the border kept turning after an error"
            assert await uf._wait(lambda: "⚠" in (sheet.status_text.value or ""), 5)
            assert "provider unreachable (test)" in sheet.status_text.value
            assert not sheet.refresh_item.disabled and not catalog.snapshot.polling
        finally:
            polls.gate.set()
            await _shutdown(app)

    _run(scenario())


# ---- long-press › "Refresh this provider" ----------------------------------------------------------------


def test_row_action_refresh_this_provider_shows_progress_and_result(polls):
    async def scenario() -> None:
        session, page, app, catalog, sheet = await _open_sheet()
        try:
            actions = sheet.open_row_actions(MODEL)
            tile = next(t for t in actions.tiles if getattr(t, "key", None) == "refresh-provider")
            assert await uf._wait(lambda: actions.dialog.open, 5)
            await session.dispatch_event(tile._i, "click", None)
            assert await _until_started(polls)
            assert polls.calls == ["openai"]
            assert await uf._wait(lambda: sheet.search_border.active, 5)
            # long-press again while it polls: the row action says so, disabled with its reason
            busy = sheet.open_row_actions(MODEL)
            item = next(i for i in busy.items if i.key == "refresh-provider")
            assert item.label == "Refreshing this provider…" and item.disabled_reason
            busy.close()
            polls.gate.set()
            assert await uf._wait(lambda: "2 models found" in (sheet.status_text.value or ""), 10)
            assert not sheet.search_border.active and not sheet.search_border.spinning
            again = sheet.open_row_actions(MODEL)
            assert next(i for i in again.items if i.key == "refresh-provider").label == "Refresh this provider"
            again.close()
        finally:
            polls.gate.set()
            await _shutdown(app)

    _run(scenario())


# ---- closing, reduce motion, a poll already running ----------------------------------------------------------


def test_closing_the_sheet_mid_poll_stops_the_border_and_reports_by_snackbar(polls):
    async def scenario() -> None:
        session, page, app, catalog, sheet = await _open_sheet()
        try:
            notes: list = []
            sheet.env.notify = lambda message, *a, **k: notes.append(message)
            sheet.refresh_selected_provider()
            assert await _until_started(polls)
            assert await uf._wait(lambda: sheet.search_border.spinning, 5)
            sheet.close()
            assert not sheet.search_border.active
            assert await uf._wait(lambda: not sheet.search_border.spinning, 5), "the border task outlived the sheet"
            polls.gate.set()
            assert await uf._wait(lambda: any("2 models found" in n for n in notes), 10), notes
            assert not sheet.search_border.active
        finally:
            polls.gate.set()
            await _shutdown(app)

    _run(scenario())


def test_reduce_motion_keeps_a_still_border(polls):
    from glossarion_mobile.ui import motion

    async def scenario() -> None:
        session, page, app, catalog, sheet = await _open_sheet()
        before = motion.reduced()
        motion.set_reduce_motion(True)
        try:
            sheet.refresh_selected_provider()
            assert await _until_started(polls)
            assert await uf._wait(lambda: sheet.search_border.active, 5)
            assert not sheet.search_border.spinning and sheet.search_border.control.gradient is not None
            polls.gate.set()
            assert await uf._wait(lambda: not sheet.search_border.active, 10)
        finally:
            motion.set_reduce_motion(before)
            polls.gate.set()
            await _shutdown(app)

    _run(scenario())


def test_an_explicit_refresh_waits_for_a_running_poll_of_the_provider(polls):
    """The sheet's 24 h auto-poll (explicit=False) holds the provider: the explicit refresh used to be dropped
    ("already running", in the hidden SnackBar); now it waits for it, then polls itself."""

    async def scenario() -> None:
        session, page, app, catalog, sheet = await _open_sheet()
        try:
            auto = app.models_keys.spawn(catalog.refresh("openai", explicit=False))
            assert await _until_started(polls)
            assert await uf._wait(lambda: sheet.search_border.active, 5)  # automatic polls show the border too
            assert "Checking online models" in sheet.status_text.value
            task = sheet.refresh_provider("openai")
            assert task is not None
            polls.gate.set()
            await auto
            await task
            assert polls.calls == ["openai", "openai"]
            assert "2 models found" in sheet.status_text.value
            assert await uf._wait(lambda: not sheet.search_border.active, 5)
        finally:
            polls.gate.set()
            await _shutdown(app)

    _run(scenario())


# ---- pure helpers ---------------------------------------------------------------------------------------------


def test_numbered_account_polls_light_their_group_and_outcome_text():
    from glossarion_mobile.services import model_catalog as mc
    from glossarion_mobile.ui.sheets.model_sheet import refresh_outcome_text

    snap = mc.CatalogSnapshot(polling=frozenset({"authgpt:2"}))
    assert snap.is_polling("authgpt") and snap.is_polling("authgpt:2") and not snap.is_polling("openai")
    assert not mc.CatalogSnapshot(polling=frozenset({"custom:lan/"})).is_polling("custom")
    ok = mc.RefreshOutcome("openai", True, statuses={"openai": "online (12 models)"}, new_models=3)
    assert refresh_outcome_text("openai", ok).endswith("12 models found · 3 new")
    bad = mc.RefreshOutcome("openai", False, statuses={"openai": "static fallback (no provider credential)"})
    assert refresh_outcome_text("openai", bad).startswith("⚠") and "no provider credential" in refresh_outcome_text("openai", bad)
    assert refresh_outcome_text("openai", None).startswith("⚠")


def test_an_unmounted_border_gives_up_and_can_turn_again(monkeypatch):
    """Review fix: a border never mounted stops turning after MAX_UNMOUNTED_TICKS and is reset (no gradient,
    not active), so a later start() - the sheet shown, the next catalog publish - turns it afresh instead of
    staying frozen behind the idempotent start()."""
    import flet as ft

    from glossarion_mobile.ui.components.poll_border import PollBorder

    async def scenario() -> None:
        border = PollBorder(ft.TextField())
        border.TICK_S = 0.01
        border.MAX_UNMOUNTED_TICKS = 3
        border.start()
        for _ in range(500):
            if not border.spinning:
                break
            await asyncio.sleep(0.01)
        assert not border.spinning and not border.active and border.control.gradient is None
        border.start()
        assert border.active and border.spinning and border.control.gradient is not None
        border.stop()
        await asyncio.sleep(0)
        assert not border.active and border.control.gradient is None

    _run(scenario())
