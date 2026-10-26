"""Mobile hotfix 3, owner item 4: "Why did glossary default to off after switching out of an on device model
instead of balanced glossary".

Diagnosis (reproduced on f4753489 with the real ``MobileConfigStore`` and the shared ``settings_rules``):
choosing a small on-device ``local/`` model writes NO GLOSSARY (``setting_writes.ON_DEVICE_GLOSSARY_SETTINGS``)
and remembers the glossary mode and its enable flag in the On-device record (``ON_DEVICE_BEFORE_KEY``, its
``settings``); choosing a cloud model again put back the remembered values, but when no mode had been saved
before the on-device model, ``_restore_profiles_before_on_device`` removed both keys ("the default again").
An unset ``auto_glossary_mode`` runs and shows as "Off" (``run_env._current_auto_glossary_mode``: no mode and
``enable_auto_glossary`` off -> 'off'; the schema's display default is 'off' too), not the Balanced the
Welcome recommends. Nothing is saved before the on-device model on a fresh install whose Welcome picks the
on-device model in its first step (``ChatFeature.welcome_screen.set_model`` writes the model before Finish
writes the preselected NO GLOSSARY card), or whose Welcome was skipped.

The fix (``setting_writes._glossary_mode_after_on_device``): leaving an on-device model while its NO GLOSSARY
is still selected brings back Balanced (``GLOSSARY_MODE_AFTER_ON_DEVICE``) when no mode was saved before it or
the saved one was No glossary itself, written the way the glossary-mode selector writes it (the shared rules:
enable / append / auto-map flags); a saved mode comes back as it was; a mode picked while on the on-device
model stays; the 9B model, which never changed the mode, changes nothing. No new config key: the remembered
values are the On-device record's.

Every global model writer goes through ``write_setting(store, "model", ...)``; the app tests drive each one on
the real app (the drawer / Welcome ModelSheet, Chat settings › All chats, Model manager "Use", Series ›
All chats) and check that ``/model`` in an open chat (a model of that chat only) never touches the global mode.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix3_glossary_default.py
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _backend_error() -> str:
    try:
        import glossary_document  # noqa: F401
        import prompt_profiles
        import settings_rules  # noqa: F401

        prompt_profiles.profile_state_from_config({})
    except Exception as exc:  # the flet-only venv lacks the backend dependencies
        return f"the shared backend is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
pytestmark = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_hotfix3_glossary",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env

LOCAL = "local/qwen3-0.6b-q4"
OTHER_LOCAL = "local/gemma-3-1b-it-q4"
LOCAL_9B = "local/Qwen3.5-9B-Q4_K_M"
CLOUD = "authchatgpt/auto"
OTHER_CLOUD = "gemini-2.5-flash"
MODE = "auto_glossary_mode"
ENABLE = "enable_auto_glossary"
#: what the glossary-mode selector writes (``settings_rules.AUTO_GLOSSARY_MODE_KEYS``)
SELECTOR_KEYS = (MODE, ENABLE, "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "Library"),
                      ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_env" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "_env" / "Output"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    yield tmp_path


def _store(tmp_path: Path, config=None):
    from glossarion_mobile.state.config_store import MobileConfigStore

    folder = tmp_path / "cfg"
    folder.mkdir(exist_ok=True)
    path = folder / "config.json"
    path.write_text(json.dumps(config or {}), encoding="utf-8")
    store = MobileConfigStore(str(path), debounce=10,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


def _run_mode(store) -> str:
    """The mode a chat run / book job extracts with (``RunEnvMixin._current_auto_glossary_mode`` on the config)."""
    from glossarion_mobile.ui.chat.direct_text_rules import auto_glossary_mode_key

    return auto_glossary_mode_key(store.get)


def _selector_writes(config: dict, mode: str) -> dict:
    """What the glossary-mode selector (Settings › Glossary mode, the desktop shortcut combo) writes for
    ``mode`` on ``config``: the shared rules' changed keys."""
    import settings_rules

    changed, _env = settings_rules.apply_change(copy.deepcopy(config), MODE, mode)
    return dict(changed)


def _assert_balanced(store, before_leaving: dict) -> None:
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.state import setting_writes as sw

    expected = _selector_writes(before_leaving, "balanced")
    assert expected[MODE] == "balanced" and expected[ENABLE] is True
    for key, value in expected.items():
        assert store.get(key) == value, key
    assert _run_mode(store) == "balanced"
    assert GlossaryService(config=store).mode() == "balanced"  # the Glossary page's mode row
    assert not store.has(sw.ON_DEVICE_BEFORE_KEY)


def _on_device(store) -> None:
    assert store.get(MODE) == "no_glossary" and store.get(ENABLE) is False
    assert _run_mode(store) == "no_glossary"


# ---- the rule ------------------------------------------------------------------------------------------


def test_balanced_is_the_welcome_default_and_on_device_is_the_no_glossary_card():
    from glossarion_mobile.state import setting_writes as sw
    from glossarion_mobile.ui.screens.welcome_flow import DEFAULT_GLOSSARY_MODE, welcome_glossary_updates

    assert sw.GLOSSARY_MODE_AFTER_ON_DEVICE == DEFAULT_GLOSSARY_MODE == "balanced"
    assert sw.ON_DEVICE_GLOSSARY_SETTINGS == welcome_glossary_updates("no_glossary")
    assert sw.GLOSSARY_MODE_KEY == MODE


@pytest.mark.parametrize("welcome", [False, True], ids=["welcome-skipped", "welcome-no-glossary-card"])
def test_leaving_on_device_without_a_saved_mode_brings_back_balanced(tmp_path, welcome):
    """The owner's case: nothing saved before the on-device model -> Balanced, not the unset key's 'Off'."""
    from glossarion_mobile.state import setting_writes as sw
    from glossarion_mobile.ui.screens.welcome_flow import welcome_glossary_updates

    store = _store(tmp_path, {})
    sw.write_setting(store, "model", LOCAL)
    if welcome:  # the Welcome's Finish after its first step picked the on-device model (ChatFeature.welcome_screen)
        store.set_many(welcome_glossary_updates("no_glossary"))
    _on_device(store)
    before_leaving = store.snapshot()
    sw.write_setting(store, "model", CLOUD)
    _assert_balanced(store, before_leaving)
    store._saver.close()


def test_a_saved_no_glossary_mode_also_brings_back_balanced(tmp_path):
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, {MODE: "no_glossary", ENABLE: False})
    sw.write_setting(store, "model", LOCAL)
    _on_device(store)
    before_leaving = store.snapshot()
    sw.write_setting(store, "model", CLOUD)
    _assert_balanced(store, before_leaving)
    store._saver.close()


@pytest.mark.parametrize("saved", [
    {MODE: "minimal", ENABLE: True},
    {MODE: "balanced", ENABLE: True, "append_glossary": True, "append_glossary_auto_load": True},
    {MODE: "full", ENABLE: True},
    {MODE: "single_pass", ENABLE: True},
    {MODE: "off", ENABLE: False},
    {MODE: "off_fuzzy_automap", ENABLE: False},
    {MODE: "off_no_automap", ENABLE: False},
    {MODE: "full"},  # an imported desktop config without the legacy flag
], ids=lambda saved: "-".join(str(v) for v in saved.values()))
def test_a_saved_mode_comes_back_as_it_was(tmp_path, saved):
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, dict(saved))
    sw.write_setting(store, "model", LOCAL)
    _on_device(store)
    sw.write_setting(store, "model", CLOUD)
    for key in SELECTOR_KEYS:
        if key in saved:
            assert store.get(key) == saved[key], key
        else:
            assert not store.has(key), key
    store._saver.close()


@pytest.mark.parametrize("writer", ["settings", "glossary_page"])
@pytest.mark.parametrize("picked", ["minimal", "full", "off", "off_no_automap"])
@pytest.mark.parametrize("config", [{}, {MODE: "balanced", ENABLE: True}], ids=["fresh", "saved"])
def test_a_mode_picked_while_on_device_stays(tmp_path, writer, picked, config):
    """The user changed the mode on the on-device model (Settings › Glossary mode / the Glossary page's mode
    row): the cloud model keeps that choice."""
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, dict(config))
    sw.write_setting(store, "model", LOCAL)
    _on_device(store)
    if writer == "settings":
        sw.write_setting(store, MODE, picked)
    else:
        GlossaryService(config=store).set_mode(picked)
    picked_state = {key: store.get(key) for key in SELECTOR_KEYS if store.has(key)}
    sw.write_setting(store, "model", CLOUD)
    assert {key: store.get(key) for key in SELECTOR_KEYS if store.has(key)} == picked_state
    assert store.get(MODE) == picked and not store.has(sw.ON_DEVICE_BEFORE_KEY)
    store._saver.close()


def test_repeated_switching(tmp_path):
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, {})
    for _round in range(3):  # cloud -> on-device -> cloud, again and again: Balanced every time
        sw.write_setting(store, "model", LOCAL)
        _on_device(store)
        sw.write_setting(store, "model", OTHER_LOCAL)  # one on-device model to another: still No glossary
        _on_device(store)
        before_leaving = store.snapshot()
        sw.write_setting(store, "model", CLOUD)
        _assert_balanced(store, before_leaving)
        sw.write_setting(store, "model", OTHER_CLOUD)  # cloud to cloud: nothing changes
        assert store.get(MODE) == "balanced"
    # the user's own mode on a cloud model survives every round trip
    sw.write_setting(store, MODE, "minimal")
    minimal = {key: store.get(key) for key in SELECTOR_KEYS}
    for _round in range(2):
        sw.write_setting(store, "model", LOCAL)
        _on_device(store)
        sw.write_setting(store, "model", CLOUD)
        assert {key: store.get(key) for key in SELECTOR_KEYS} == minimal
    # a mode picked on the on-device model, then on-device -> cloud -> on-device -> cloud
    sw.write_setting(store, "model", LOCAL)
    sw.write_setting(store, MODE, "full")
    sw.write_setting(store, "model", CLOUD)
    assert store.get(MODE) == "full"
    sw.write_setting(store, "model", LOCAL)
    _on_device(store)
    sw.write_setting(store, "model", CLOUD)
    assert store.get(MODE) == "full" and store.get(ENABLE) is True
    store._saver.close()


@pytest.mark.parametrize("config", [{}, {MODE: "minimal", ENABLE: True}], ids=["fresh", "saved"])
def test_the_9b_model_never_changes_the_mode(tmp_path, config):
    """The 9B model extracts its own glossary (owner 2026-10-19): it keeps the mode, and leaving it writes
    no glossary mode either."""
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, dict(config))
    stored = {key: store.get(key) for key in SELECTOR_KEYS if store.has(key)}
    sw.write_setting(store, "model", LOCAL_9B)
    assert {key: store.get(key) for key in SELECTOR_KEYS if store.has(key)} == stored
    sw.write_setting(store, "model", CLOUD)
    assert {key: store.get(key) for key in SELECTOR_KEYS if store.has(key)} == stored
    store._saver.close()


def test_a_record_left_by_the_9b_model_remembers_the_mode_of_the_next_entry(tmp_path):
    """A record no cloud model restored (the model changed by a writer that does not restore it, e.g. a
    settings import) is kept on the next entry; a key that entry did not set (the 9B model leaves the mode
    alone) is remembered now, so the user's mode comes back instead of Balanced."""
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, {MODE: "full", ENABLE: True})
    sw.write_setting(store, "model", LOCAL_9B)
    record = store.get(sw.ON_DEVICE_BEFORE_KEY)
    assert record and MODE not in record["settings"] and MODE not in record["applied"]
    store.set("model", CLOUD)  # no restore: the record stays
    sw.write_setting(store, "model", LOCAL)
    _on_device(store)
    assert store.get(sw.ON_DEVICE_BEFORE_KEY)["settings"][MODE] == [True, "full"]
    sw.write_setting(store, "model", CLOUD)
    assert store.get(MODE) == "full" and store.get(ENABLE) is True
    store._saver.close()


def test_leaving_still_restores_the_other_on_device_settings(tmp_path):
    """The Balanced write sits next to the other restores (profiles, request size, OCR provider)."""
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, {"batch_size": 3, "translation_temperature": 0.3})
    sw.write_setting(store, "model", LOCAL)
    assert store.get("batch_size") == 1 and store.get("translation_temperature") == sw.ON_DEVICE_TEMPERATURE
    assert store.get("active_glossary_prompt_profiles") == {"balanced_full": sw.ON_DEVICE_PROFILE,
                                                            "minimal": sw.ON_DEVICE_PROFILE}
    sw.write_setting(store, "model", CLOUD)
    assert store.get("batch_size") == 3 and store.get("translation_temperature") == 0.3
    assert sw.ON_DEVICE_PROFILE not in (store.get("active_glossary_prompt_profiles") or {}).values()
    assert store.get(MODE) == "balanced" and not store.has(sw.ON_DEVICE_BEFORE_KEY)
    store._saver.close()


# ---- every model-switch path on the real app -----------------------------------------------------------


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_hotfix3_glossary",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forget_glossary_mode(store) -> None:
    """The owner's state before an on-device model: no glossary mode saved."""
    for key in SELECTOR_KEYS:
        store.unset(key)
    assert _run_mode(store) == "off"  # what an unset mode runs as (the bug's result)


@needs_flet
def test_every_global_model_switch_path_brings_back_balanced(app_env):
    """The drawer's model chip (``open_model_sheet(global_scope=True, follow_open_chat=True)``: the
    ModelSheet, which ``model_sheet_min.ModelSheetMin`` is), Chat settings › All chats (its model field and
    its Choose… sheet), Model manager "Use" and Series › All chats: on-device -> No glossary, back to a cloud
    model -> Balanced, twice each."""
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet
    from glossarion_mobile.ui.sheets.model_sheet_min import ModelSheetMin

    assert ModelSheetMin is ModelSheet
    tf = _foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android")
        try:
            store, view = app.config_store, app.chat_view
            assert view.bound
            cid = view._on_new_scratch()
            assert cid

            def drawer(model):
                sheet = view.open_model_sheet("model", global_scope=True, follow_open_chat=True)
                assert isinstance(sheet, ModelSheet)
                sheet.select("model", model)

            def chat_settings_field(model):
                view.open_chat_settings(scope="global").set_value("model", model)

            def chat_settings_choose(model):
                settings = view.open_chat_settings(scope="global")
                settings.on_choose_model(settings.scope).select("model", model)

            def model_manager(model):
                app.models_keys.use_model(model)

            def series_all_chats(model):
                app.series._choose_model("series-1", "global", store).select("model", model)

            for name, pick in (("drawer", drawer), ("chat settings", chat_settings_field),
                               ("chat settings › Choose", chat_settings_choose), ("model manager", model_manager),
                               ("series › All chats", series_all_chats)):
                _forget_glossary_mode(store)
                for _round in range(2):
                    pick(LOCAL)
                    assert store.get("model") == LOCAL, name
                    assert store.get(MODE) == "no_glossary" and store.get(ENABLE) is False, name
                    before_leaving = store.snapshot()
                    pick(CLOUD)
                    assert store.get("model") == CLOUD, name
                    for key, value in _selector_writes(before_leaving, "balanced").items():
                        assert store.get(key) == value, (name, key)
                    assert _run_mode(store) == "balanced", name

            # /model in an open chat picks that chat's own model (owner U13): the global mode never moves
            _forget_glossary_mode(store)
            saved_mode = store.get(MODE, None)
            assert view.run_slash("/model") == "model"
            view.model_sheet.select("model", LOCAL)
            assert (view.env.chats.overrides(cid) or {}).get("model") == LOCAL
            assert store.get("model") == CLOUD and store.get(MODE, None) == saved_mode
            assert view.run_slash("/model") == "model"
            view.model_sheet.select("model", CLOUD)
            assert store.get("model") == CLOUD and store.get(MODE, None) == saved_mode
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.parametrize("leave", ["after-finish", "in-the-welcome"])
def test_welcome_on_device_then_cloud_brings_back_balanced(app_env, leave):
    """First run: the Welcome's first step picks the on-device model (``on_set_model`` -> ``write_setting``)
    and Finish saves the preselected NO GLOSSARY card; a cloud model chosen later (the drawer) brings back
    Balanced. Changing to a cloud model inside the Welcome preselects Balanced again and restores it too."""
    tf = _foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", first_run=True)
        try:
            assert await tf._wait(lambda: tf._routes(page)[-1] == "/welcome")
            store = app.config_store
            screen = app.chat_feature._welcome_screen
            assert screen is not None and not store.has(MODE)
            screen._pick_model(LOCAL)
            assert store.get("model") == LOCAL
            _on_device(store)
            if leave == "in-the-welcome":
                screen._pick_model(CLOUD)
                assert store.get("model") == CLOUD and store.get(MODE) == "balanced"
                updates = screen.finish()
                assert updates[MODE] == "balanced"
            else:
                updates = screen.finish()
                assert updates[MODE] == "no_glossary"  # the card preselected for the on-device model
                _on_device(store)
                before_leaving = store.snapshot()
                app.chat_view.open_model_sheet("model", global_scope=True, follow_open_chat=True).select(
                    "model", CLOUD)
                assert store.get("model") == CLOUD
                for key, value in _selector_writes(before_leaving, "balanced").items():
                    assert store.get(key) == value, key
            assert _run_mode(store) == "balanced" and store.get(ENABLE) is True
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())
