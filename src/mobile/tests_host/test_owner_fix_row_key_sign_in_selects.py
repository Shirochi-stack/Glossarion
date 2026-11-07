"""Owner issue (2026-11-07): tapping "Add key" or "Sign in with ..." on a Model menu row and finishing it did
not select that model; the user had to go back to the model manager and pick it again.

Now every row action that leads to a key / sign-in flow selects its model once the flow finishes (and
the sheet's own ``on_select`` runs, so the user stays where they were); a close without a key / sign-in
leaves the model as it was.
"""

from __future__ import annotations

import asyncio
import importlib.util
import types
from pathlib import Path

_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_rowfix",
                                                  Path(__file__).with_name("test_models_keys.py"))
_MK = importlib.util.module_from_spec(_MK_SPEC)
_MK_SPEC.loader.exec_module(_MK)

needs_flet = _MK.needs_flet


def _sheet(tmp_path, *, current="gemini-3.5-flash", signed=(), **env_fields):
    from glossarion_mobile.state.prefs import Prefs
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    conn, session = _MK._fake_session("android")
    page = session.page
    store = _MK._store(tmp_path, {"model": current})
    service = _MK._service(store, _MK.FakeOptions())
    service.load_blocking()
    prefs = Prefs(tmp_path / "mobile_state.json")
    prefs.load()
    selected: list = []
    env = SheetEnv(catalog=_MK.FakeCatalog(service), store=store, prefs=prefs, ctx=_MK._settings_ctx(store, page),
                   navigate=lambda name, params=None: selected.append(("nav", name)),
                   signed_in_keys=lambda: frozenset(signed), spawn=lambda coro: asyncio.ensure_future(coro),
                   **env_fields)
    sheet = ModelSheet(current_model=current, signed_in=lambda m: False, env=env,
                       on_select=lambda f, v, chat: selected.append((f, v, chat)))
    sheet.show(page)
    return sheet, selected, (prefs, store)


def _close(handles):
    prefs, store = handles
    prefs.close()
    store._saver.close()


@needs_flet
def test_row_sign_in_selects_the_model_once_signed_in(tmp_path):
    async def scenario():
        calls: list = []
        sheet, selected, handles = _sheet(tmp_path, sign_in=lambda *args: calls.append(args))
        row = sheet.rows["authgpt/gpt-6-luna"]
        assert row.trailing.content == "Sign in with ChatGPT"
        row.trailing.on_click(None)
        assert sheet.closed and len(calls) == 1
        route, account, on_done = calls[0]
        assert (route, account) == ("authgpt", 0) and callable(on_done)
        assert selected == []  # the LoginSheet is open: nothing selected yet (a close leaves the model)
        on_done({"signed_in": True})
        assert selected == [("model", "authgpt/gpt-6-luna", False)]
        _close(handles)

    asyncio.run(scenario())


@needs_flet
def test_row_add_key_opens_the_editor_and_selects_the_model_once_saved(tmp_path):
    async def scenario():
        calls: list = []
        sheet, selected, handles = _sheet(tmp_path, add_key=lambda *args: calls.append(args))
        row = sheet.rows["gpt-6"]
        assert row.trailing.content == "Add key"
        row.trailing.on_click(None)
        # the editor over the current screen, not the Multi-Key Manager page
        assert sheet.closed and ("nav", "settings.keys") not in selected and len(calls) == 1
        model, on_saved = calls[0]
        assert model == "gpt-6" and selected == []
        on_saved({"api_key": "sk-x", "model": "gpt-6"})
        assert selected == [("model", "gpt-6", False)]
        _close(handles)

    asyncio.run(scenario())


@needs_flet
def test_auth_menu_and_add_account_sign_ins_select_their_model(tmp_path):
    async def scenario():
        calls: list = []
        sheet, selected, handles = _sheet(tmp_path, current="authgpt/gpt-6-luna",
                                          sign_in=lambda *args: calls.append(args), next_slot=lambda route: 2)
        sheet._add_account("authgpt")
        route, account, on_done = calls[-1]
        assert (route, account) == ("authgpt", 2)
        on_done({"signed_in": True})
        assert selected == [("model", "authgpt2/gpt-6-luna", False)]
        # the route row's "Switch sign-in" menu: a sign-in that is not signed in opens it, then selects
        # that sign-in's most cost-efficient model (what the same item does once signed in)
        from glossarion_mobile.services import model_catalog as mc

        best = mc.recommended_model("authgpt", sheet.snapshot.models)
        assert best
        item = next(i for i in sheet.open_auth_menu("authgpt").items if i.key == "auth-menu-authgpt")
        item.on_select()
        route, account, on_done = calls[-1]
        assert (route, account) == ("authgpt", 0)
        on_done({"signed_in": True})
        assert selected[-1] == ("model", best, False)
        _close(handles)

    asyncio.run(scenario())


@needs_flet
def test_feature_add_key_editor_saves_a_key_for_the_model_and_reports_only_a_save(tmp_path):
    from glossarion_mobile.ui.screens.model_manager import ModelsKeysFeature

    async def scenario():
        conn, session = _MK._fake_session("android")
        store, keys = _MK._keys(tmp_path)
        ctx = _MK._settings_ctx(store, session.page)

        async def run_io(fn, *args):
            return fn(*args)

        feature = types.SimpleNamespace(ctx=ctx, keys=keys, run_io=run_io, app=types.SimpleNamespace())
        saved: list = []
        editor = ModelsKeysFeature.add_key(feature, "gpt-6", saved.append)
        assert editor.model_picker.value == "gpt-6" and editor.new
        editor.close()  # cancelled: nothing stored, nothing selected
        assert saved == [] and keys.count("main") == 0
        editor = ModelsKeysFeature.add_key(feature, "gpt-6", saved.append)
        editor.key_field.set_value("sk-new-key-99999999", notify=False)
        assert editor.save()
        assert keys.count("main") == 1 and keys.keys("main")[0]["model"] == "gpt-6"
        assert len(saved) == 1 and saved[0]["model"] == "gpt-6"
        store._saver.close()

    asyncio.run(scenario())


def test_feature_sign_in_hands_on_done_to_the_login_sheet(monkeypatch):
    from glossarion_mobile.ui.screens import accounts
    from glossarion_mobile.ui.screens.model_manager import ModelsKeysFeature

    made: list = []

    class FakeLoginSheet:
        def __init__(self, oauth, **kwargs):
            made.append(kwargs)

        def show(self, page):
            pass

    monkeypatch.setattr(accounts, "LoginSheet", FakeLoginSheet)
    feature = types.SimpleNamespace(_oauth=lambda: object(), page=None, app=types.SimpleNamespace())
    done = lambda status: None  # noqa: E731
    ModelsKeysFeature.sign_in(feature, "authgem", 1, on_done=done)
    assert made and made[0]["on_done"] is done and made[0]["provider"] == "authgem" and made[0]["account_id"] == 1
