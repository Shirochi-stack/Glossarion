"""Owner device fixes (manga translator + Multi-Key Manager):

* the selected key-pool chip scrolls into view (the Vision pool opened off-screen);
* the manga editor's cached MangaTranslator is rebuilt for each job (its multi-key pool lookup runs
  in the job's own env / key pools);
* the manga translator's own model + API key (Prefs, ``ENC:``) reach its jobs;
* a stop latched by an earlier run is cleared for a key test and for every editor step;
* the box texts come back from the page state (no blank OCR / translation fields);
* the box editors: an inline translation panel and a compact OCR sheet.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services import manga as svc  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
needs_editor = pytest.mark.skipif(not (_has("manga_editor_core") and _has("manga_env")),
                                  reason="the shared manga cores are not importable")


def _manga_ui():
    name = "_glossarion_devfix_manga_ui_helpers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("test_manga_ui.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    for name in ("Output", "home", "data"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


class _Prefs(dict):
    def set(self, key, value):
        self[key] = value


class _Box:
    """A reversible stand-in for the app's SecretBox (``ENC:`` + reversed text)."""

    def seal(self, value):
        return "ENC:" + str(value)[::-1]

    def open(self, token):
        assert str(token).startswith("ENC:")
        return str(token)[4:][::-1]


# ---- issue 2: the selected pool chip scrolls into view ------------------------------------------------


@needs_flet
def test_selected_pool_chip_scrolls_into_view():
    import flet as ft

    from glossarion_mobile.ui.screens.keys import KeysScreen

    calls = []

    class Row:
        page = object()

        async def scroll_to(self, **kwargs):
            calls.append(kwargs)

    screen = KeysScreen.__new__(KeysScreen)
    screen.pool = "qa_scan"
    screen.chip_row = Row()
    screen.pool_chips = {"main": ft.Chip(label=ft.Text("Translation"), key=ft.ScrollKey("pool-translation")),
                         "qa_scan": ft.Chip(label=ft.Text("Vision"), key=ft.ScrollKey("pool-vision"))}
    screen.spawn_fn = None
    task = asyncio.new_event_loop()
    try:
        screen.spawn = lambda coro: task.run_until_complete(coro)
        screen.scroll_chip_into_view(duration=200)
    finally:
        task.close()
    assert calls == [{"scroll_key": screen.pool_chips["qa_scan"].key, "duration": 200}]
    # not on a page yet: nothing to scroll (no error)
    screen.chip_row = types.SimpleNamespace()
    assert screen.scroll_chip_into_view() is None


@needs_flet
def test_pool_chips_carry_scroll_keys():
    import inspect

    from glossarion_mobile.ui.screens import keys

    assert 'key=ft.ScrollKey(f"pool-{spec.slug}")' in inspect.getsource(keys.KeysScreen.build_body)


# ---- issue 5: a latched stop is cleared for a key test -------------------------------------------------


def test_key_test_clears_a_latched_stop(monkeypatch):
    import unified_api_client

    from glossarion_mobile.ui.screens.endpoints import run_in_run_env

    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    unified_api_client.set_stop_flag(True)
    try:
        seen = run_in_run_env({}, lambda: (os.environ.get("TRANSLATION_CANCELLED"), os.environ.get("GRACEFUL_STOP"),
                                           unified_api_client.is_stop_requested()),
                              owner_factory=lambda *a, **k: object(), env_builder=lambda *a: {})
    finally:
        unified_api_client.set_stop_flag(False)
    assert seen == (None, "0", False)


# ---- issue 4: the manga translator's own model + API key ----------------------------------------------


def test_translation_model_and_key_reach_the_job(monkeypatch):
    monkeypatch.setattr(svc, "_secret_box", lambda: _Box())
    prefs = _Prefs()
    assert svc.translation_params(prefs) == {}
    svc.save_translation_model(prefs, " gemini-3.5-flash ")
    assert svc.save_translation_api_key(prefs, "sk-manga-123456")
    assert prefs[svc.PREF_API_KEY].startswith("ENC:") and "sk-manga" not in prefs[svc.PREF_API_KEY]
    assert svc.translation_api_key(prefs) == "sk-manga-123456"
    params = svc.translation_params(prefs)
    assert params == {"config_overrides": {"model": "gemini-3.5-flash"}, svc.PARAM_API_KEY: prefs[svc.PREF_API_KEY]}
    spec = svc.step_spec("translate", "tok", __file__, extra=params)
    assert spec.params["config_overrides"] == {"model": "gemini-3.5-flash"}
    # the job thread puts the key where the manga code looks it up
    from headless_owner import TextShim

    owner = types.SimpleNamespace(config={"api_key": "main-key"}, api_key_entry=TextShim("main-key"))
    assert svc.apply_translation_key(owner, spec.params)
    assert owner.config["api_key"] == "sk-manga-123456" and owner.api_key_entry.text() == "sk-manga-123456"
    # forgetting it: the main key again
    svc.save_translation_api_key(prefs, "")
    svc.save_translation_model(prefs, "")
    assert svc.translation_params(prefs) == {}
    assert svc.apply_translation_key(owner, {}) is None


def test_plain_key_is_never_stored(monkeypatch):
    from glossarion_mobile.services.share_links import SecretBox

    monkeypatch.setattr(svc, "_secret_box", lambda: SecretBox(ready=lambda: False))
    prefs = _Prefs()
    with pytest.raises(Exception):
        svc.save_translation_api_key(prefs, "sk-plain-0000")
    assert "sk-plain" not in repr(prefs)


# ---- issues 3, 5, 8: the editor session ---------------------------------------------------------------


@needs_editor
def test_editor_session_job_owner_texts_and_stop(tmp_path, monkeypatch):
    import manga_editor_core as core

    page = _manga_ui().write_png(tmp_path / "pages" / "p1.png", 100, 200)
    session = core.MangaEditorSession(state_file=str(tmp_path / "state.json"), image_paths=[page])
    monkeypatch.setattr(session, "apply_settings", lambda owner=None: None)
    # issue 3: a translator cached by an earlier job is dropped for a new job's owner
    first, second = object(), object()
    session.set_owner(first)
    session._manga_translator = "cached"
    session.set_owner(first)
    assert session._manga_translator == "cached"
    session.set_owner(second)
    assert session._manga_translator is None
    # issue 8: the box texts come from the page state when the in-memory maps lack them
    state = {"viewer_rectangles": [{"x": 1, "y": 2, "width": 30, "height": 40, "shape": "rect"}],
             "recognized_texts": [{"text": "こんにちは", "bbox": [1, 2, 30, 40], "region_index": 0}],
             "translated_texts": [{"original": {"text": "こんにちは", "region_index": 0}, "translation": "Hello",
                                   "bbox": [1, 2, 30, 40]}]}
    session.image_state_manager.set_state(os.path.abspath(page), state, save=True)
    snap = session.open_page(page)
    assert [(b["ocr_text"], b["translation"]) for b in snap["boxes"]] == [("こんにちは", "Hello")]
    session._recognition_data, session._translation_data = {}, {}
    snap = session.page_snapshot()
    assert [(b["ocr_text"], b["translation"]) for b in snap["boxes"]] == [("こんにちは", "Hello")]
    # issue 5: a step after a Stop is not "cancelled by user": the stop flag is cleared at its start
    session.stop(force=False)
    assert session.stop_flag.is_set()
    try:
        session._begin(second, page)
        assert not session.stop_flag.is_set() and os.environ.get("GRACEFUL_STOP") == "0"
    finally:
        os.environ["GRACEFUL_STOP"] = "0"
        os.environ["WAIT_FOR_CHUNKS"] = "0"


# ---- issue 8: the box editors --------------------------------------------------------------------------


@needs_flet
def test_box_panel_ocr_and_translated_tabs():
    """The box panel: OCR and Translated tabs, each text edited in place (no OCR sheet), the ⋯ menu
    in the header row, compact type."""
    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.tools.manga.box_sheet import BOX_TABS, NOT_RECOGNIZED, BoxPanel, box_signature

    actions = []
    box = {"ocr_text": "原文", "translation": "Hi", "free_text": False, "exclude_from_clean": False}
    panel = BoxPanel(box, 2, on_action=lambda *a: actions.append(a))
    assert [label for _name, label in BOX_TABS] == ["OCR", "Translated", "Move"]
    assert panel.translation_field.value == "Hi" and panel.ocr_field.value == "原文"
    assert not panel.ocr_field.read_only and not panel.translate_button.disabled
    assert panel.tab == "translated" and panel.translation_body.visible and not panel.ocr_body.visible
    header = panel.control.content.controls[0]
    assert header.controls[-1] is panel.close_button and panel.menu in header.controls and panel.tabs in header.controls
    assert not header.wrap
    assert panel.translation_field.text_style.size == tokens.TYPE_SCALE["body_small"].size
    assert panel.changes() == {}
    panel.translation_field.value = "Hello"
    panel.act("save_render")
    panel.set_flag("free_text", True)
    panel.select_tab("ocr")
    assert panel.tab == "ocr" and panel.ocr_body.visible and not panel.translation_body.visible
    assert panel.tabs.selected_index == 0
    panel.ocr_field.value = "原文!"
    panel.act("ocr")
    assert actions == [("save_render", 2, {"translation": "Hello"}),
                       ("save", 2, {"translation": "Hello", "free_text": True}),
                       ("ocr", 2, {"ocr_text": "原文!", "translation": "Hello"})]
    assert panel.signature == box_signature(box, 2) != box_signature({**box, "translation": "Hello"}, 2)
    again = BoxPanel(box, 2, on_action=lambda *a: None, tab="ocr")
    again.keep_typing(panel)
    assert again.tab == "ocr" and again.ocr_field.value == "原文!" and again.translation_field.value == "Hello"
    # not recognized: the OCR text is read-only (desktop), Translate this text waits for OCR
    empty = BoxPanel({}, 0, on_action=lambda *a: None)
    assert empty.translate_button.disabled and empty.ocr_field.read_only
    assert empty.ocr_field.hint_text == NOT_RECOGNIZED
    empty.ocr_field.value = "typed"
    assert empty.changes() == {}
