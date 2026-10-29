"""v9.15.7 hotfix 3, owner (2026-10-29, phone): "the sign in buttons are now all unresponsive and crash the mobile
app" ("Error running app: Frozen controls cannot be updated", raised in Flet's ``wrapped_on_dismiss`` ->
``_restore_dialog_on_dismiss``); "I can somehow get to the sign in page if I tap hard enough, but it just
crashes".

Cause: two taps on DeepSeek's "Sign in" before the first sheet was drawn showed two sheets with the same key
(``authds-signin-sheet``). Flet diffs a keyed pair of different objects as frozen and marks the second sheet
``_frozen``; its dismiss then raised, ``page._dialogs`` kept both sheets and every later tap went nowhere.

Fix: ``show_deepseek_sign_in`` returns the open sheet instead of a second one; ``AccountsScreen.open_login``
does the same for the other providers' sheets; the app's ``show_dialog`` hook never shows a dialog while an open
one with the same key is in ``page._dialogs`` (a closed twin waiting for its dismiss leaves first), and a frozen
dialog can still dismiss (the marker is stripped as Flet's own ``use_dialog`` does).

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_3.py
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_hf9157_3_sheets",
                                               Path(__file__).with_name("test_hotfix3_sheets.py"))
HS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(HS)
TF = HS.TF
storage = HS.storage
app_env = HS.app_env
_restore_flet_hooks = HS._restore_flet_hooks


def _walk(control):
    yield control
    for name in ("controls", "content", "actions", "leading", "trailing", "title", "subtitle"):
        value = getattr(control, name, None)
        for child in value if isinstance(value, list) else [value]:
            if child is not None and hasattr(child, "_i"):
                yield from _walk(child)


async def _accounts(app, page, model):
    await TF._wait(lambda: app.state.engine_ready)
    app.navigate_to("settings.accounts")
    await model.settle()
    return {getattr(c, "key", None): c for c in _walk(page.views[-1]) if getattr(c, "key", None)}


@HS.needs_flet
def test_two_quick_taps_on_deepseek_sign_in_open_one_sheet_that_closes(app_env):
    async def scenario():
        conn, session, page, app, model = await HS._start_app()
        errors: list = []
        report = session.error
        session.error = lambda message: (errors.append(message), report(message))[1]
        try:
            keyed = await _accounts(app, page, model)
            button = keyed["account-authds-signin"]
            for _ in range(2):  # both taps arrive before the client drew a frame
                await session.dispatch_event(button._i, "click", None)
                await session.after_event(button)
            await model.settle()
            sheets = [d for d in page._dialogs.controls if getattr(d, "key", None) == "authds-signin-sheet"]
            assert len(sheets) == 1 and not hasattr(sheets[0], "_frozen")
            await model.back()
            await model.settle()
            assert errors == [] and page._dialogs.controls == []
            # and it opens again afterwards
            await session.dispatch_event(button._i, "click", None)
            await session.after_event(button)
            await model.settle()
            assert model.top()[0] == "modal"
        finally:
            await TF._stop(app)

    asyncio.run(scenario())


@HS.needs_flet
def test_two_quick_taps_on_a_provider_sign_in_open_one_sheet(app_env):
    async def scenario():
        conn, session, page, app, model = await HS._start_app()
        try:
            await _accounts(app, page, model)
            button = next(c for c in _walk(page.views[-1]) if type(c).__name__.endswith("Button")
                          and getattr(c, "content", None) == "Sign in" and getattr(c, "key", None) is None)
            for _ in range(2):
                await session.dispatch_event(button._i, "click", None)
                await session.after_event(button)
            await model.settle()
            assert len(page._dialogs.controls) == 1
            await model.back()
            await model.settle()
            assert page._dialogs.controls == []
        finally:
            await TF._stop(app)

    asyncio.run(scenario())


@HS.needs_flet
def test_any_keyed_dialog_shown_twice_stays_one_and_a_frozen_dialog_still_dismisses():
    import flet as ft

    from glossarion_mobile.ui.components import dialogs, sheet as s

    dialogs.install_views_guard()

    async def scenario():
        conn, session, page = HS._phone_session()
        model = HS.FlutterModel(session, conn)
        errors: list = []
        report = session.error
        session.error = lambda message: (errors.append(message), report(message))[1]
        first = s.bottom_sheet(s.sheet_frame(ft.Column([ft.Text("a")])), key="twin-sheet")
        second = s.bottom_sheet(s.sheet_frame(ft.Column([ft.Text("b")])), key="twin-sheet")
        page.show_dialog(first)
        page.show_dialog(second)  # an open twin: ignored
        await model.settle()
        assert page._dialogs.controls == [first] and not hasattr(first, "_frozen")
        object.__setattr__(first, "_frozen", True)  # whatever froze it, its dismiss must still work
        await model.back()
        await model.settle()
        assert errors == [] and page._dialogs.controls == []

    asyncio.run(scenario())
