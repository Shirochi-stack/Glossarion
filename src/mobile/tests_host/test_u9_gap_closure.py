"""Host tests for the U9 feature-gap closures (audit round 2).

* global setting writes with the desktop control's side effects (``state.setting_writes``: output
  mode flags, glossary-mode toggles, thoughts lock, target-language fan-out, the profile's
  extraction method) from Settings tiles, the Plan card's run overlay, the ModelSheet and Chat
  settings; the Plan card's mode chip on "Save to: Library";
* the UI_SPEC §4.15 curated settings sections, KeyPoolTiles and section links;
* Notifications & background, the full-screen composer, ＋ › From Library;
* Multi-Key Manager: Copy current key, bulk per-key edits, live stats / Cooling (Ns);
* sign-in lines of every OAuth provider, Check environment, the ModelSheet 📊 Gemini status;
* Glossary: settings tabs' Search / ⋯, the wide-screen section hook, Request mode, Entry Type
  Configuration, Custom Fields' description flag, CBZ extraction;
* the desktop workspace-collision rename in the translate / glossary adapters;
* Library card Reader items on real ``library_core`` in-progress rows; SDLXLIFF tablet side list;
  Manga min / max font size, slider ranges and reason chips.

Real data is never touched: every test runs with HOME / USERPROFILE / GLOSSARION_LIBRARY_DIR /
OUTPUT_DIRECTORY / GLOSSARION_DATA_DIR in pytest's tmp dir (autouse fixture).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u9_gap_closure.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
import types
import zipfile
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


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_gap_closure",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No test reads or writes the user's Library, output folders, home or data folder."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    yield tmp_path / "_iso"


def _store(tmp_path, data=None):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    path = tmp_path / "config.json"
    path.write_text(json.dumps(data or {}), encoding="utf-8")
    schema = SchemaAccess()
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store, schema


# ==========================================================================
# setting writes with the desktop side effects
# ==========================================================================


def test_apply_change_fans_out_the_target_language():
    import settings_rules as sr

    config = {"output_language": "English", "ai_hunter_config": {"language_detection": {"target_language": "x"}}}
    changed, env = sr.apply_change(config, "output_language", "Korean")
    assert changed["output_language"] == changed["glossary_target_language"] == "Korean"
    assert config["manga_settings"]["manual_edit"]["translate_target_language"] == "Korean"
    assert config["ai_hunter_config"]["language_detection"]["target_language"] == "Korean"
    assert env == {"OUTPUT_LANGUAGE": "Korean", "GLOSSARY_TARGET_LANGUAGE": "Korean"}


def test_write_setting_applies_the_desktop_rules(tmp_path):
    from glossarion_mobile.state.setting_writes import output_mode_values, write_setting

    store, _schema = _store(tmp_path)
    write_setting(store, "output_mode", "image")
    assert store.get("output_mode") == "image"
    assert store.get("enable_image_translation") is True and store.get("enable_image_output_mode") is True
    assert store.get("enable_video_output_mode") is False
    write_setting(store, "output_mode", "text")
    assert store.get("enable_image_translation") is False and store.get("enable_image_output_mode") is False
    write_setting(store, "stream_thinking_logs", True)
    assert store.get("enable_thoughts") is True
    write_setting(store, "auto_glossary_mode", "minimal")
    assert store.get("auto_glossary_mode") == "minimal" and store.get("fuzzy_auto_mapping") is False
    write_setting(store, "output_language", "Japanese")
    assert store.get("glossary_target_language") == "Japanese"
    assert store.get("manga_settings")["manual_edit"]["translate_target_language"] == "Japanese"
    # the same value again rewrites stale legacy flags (a config written before U9)
    store.set("enable_image_translation", True)
    write_setting(store, "output_mode", "text")
    assert store.get("enable_image_translation") is False
    assert output_mode_values("vision")["enable_image_translation"] is True
    assert output_mode_values("vision")["enable_image_output_mode"] is False
    store._saver.close()


def test_run_overlay_store_writes_take_the_rules_too(tmp_path):
    from glossarion_mobile.state.setting_writes import write_setting
    from glossarion_mobile.ui.chat.plan_model import RunOverlayStore

    store, _schema = _store(tmp_path)
    overlay = RunOverlayStore(store)
    write_setting(overlay, "output_mode", "vision")
    assert overlay.values["output_mode"] == "vision" and overlay.values["enable_image_translation"] is True
    assert not store.has("output_mode")  # only this run
    store._saver.close()


def test_per_chat_profile_sets_the_extraction_method():
    from glossarion_mobile.ui.chat.run_request import config_overrides

    assert config_overrides({"profile": "Korean_html2text"}) == {"active_profile": "Korean_html2text",
                                                                 "text_extraction_method": "enhanced"}
    assert config_overrides({"profile": "Korean_BeautifulSoup"})["text_extraction_method"] == "standard"
    assert config_overrides({"profile": "Universal"}) == {"active_profile": "Universal"}


@needs_flet
def test_global_profile_and_language_selectors_use_the_shared_rules(tmp_path):
    from glossarion_mobile.state.setting_writes import write_setting
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    store, _schema = _store(tmp_path, {"prompt_profiles": {"Korean_BeautifulSoup": "BS prompt",
                                                            "Korean_html2text": "h2t prompt"}})
    write_setting(store, "active_profile", "Korean_html2text")
    assert store.get("active_profile") == "Korean_html2text" and store.get("text_extraction_method") == "enhanced"
    # ModelSheet › Profile / Language (All chats)
    applied: list = []
    fake = types.SimpleNamespace(bound=False, env=types.SimpleNamespace(store=store),
                                 apply_settings_changed=lambda: applied.append(True))
    ChatView._on_model_sheet_select(fake, "profile", "Korean_BeautifulSoup", False)
    assert store.get("text_extraction_method") == "standard" and applied
    ChatView._on_model_sheet_select(fake, "language", "French", False)
    assert store.get("glossary_target_language") == "French"
    # Chat settings › All chats
    sheet = types.SimpleNamespace(scope="global", config=store, rebuild=lambda: None, _changed=lambda: None,
                                  chats=None)
    ChatSettingsSheet.set_value(sheet, "target_language", "German")
    assert store.get("output_language") == store.get("glossary_target_language") == "German"
    store._saver.close()


@needs_flet
def test_settings_tile_writes_the_output_mode_flags(tmp_path):
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.tiles import make_tile

    store, schema = _store(tmp_path)
    ctx = SettingsContext(page=None, store=store, schema=schema)
    tile = make_tile(schema.spec("output_mode"), ctx)
    assert tile.apply("image")
    assert store.get("enable_image_translation") is True and store.get("enable_image_output_mode") is True
    flag = make_tile(schema.spec("enable_image_translation"), ctx)
    assert flag.readonly_reason == "Follows Output mode" and not flag.editable
    language = make_tile(schema.spec("output_language"), ctx)
    assert language.apply("Spanish") and store.get("glossary_target_language") == "Spanish"
    store._saver.close()


@needs_flet
def test_save_to_library_runs_with_the_plan_cards_mode_and_chat_overrides():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    submitted: list = []

    class Jobs:
        async def submit(self, kind, title, inputs, params, origin):
            submitted.append((kind, inputs, params))
            return "j1"

    chats = types.SimpleNamespace(
        session=lambda cid: {"title": "T"}, overrides=lambda cid: {"profile": "Korean_html2text", "model": "m1"},
        append_messages=lambda *a, **k: None, set_meta=lambda *a: None)
    fake = types.SimpleNamespace(env=types.SimpleNamespace(jobs=Jobs(), chats=chats), render_transcript=lambda **k: None,
                                 notify=lambda *a, **k: None, navigate=lambda *a: None)
    fake.run_overrides = lambda cid: ChatView.run_overrides(fake, cid)  # the chat's overrides as a run gets them
    plan = {"output_mode": "image"}
    asyncio.run(ChatView._submit_library(fake, "c1", plan, {"path": "/x/Book.epub", "name": "Book.epub"},
                                         {"translation_temperature": 0.5}))
    kind, inputs, params = submitted[0]
    overrides = params["config_overrides"]
    assert kind == "translate" and inputs == ("/x/Book.epub",)
    assert overrides["output_mode"] == "image" and overrides["enable_image_translation"] is True
    assert overrides["enable_image_output_mode"] is True
    assert overrides["model"] == "m1" and overrides["text_extraction_method"] == "enhanced"
    assert overrides["translation_temperature"] == 0.5


# ==========================================================================
# curated settings sections, pool tiles, links
# ==========================================================================


def test_curated_sections_regroup_without_losing_a_key():
    import settings_schema
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    sections = {s.id: s for s in schema.sections()}
    assert {"translation_defaults", "context_memory", "thinking", "provider_options", "pdf", "epub_output"} <= set(sections)
    td = sections["translation_defaults"]
    assert {"output_language", "output_mode", "delay", "thread_submission_delay", "multipass_mode"} <= set(td.keys)
    assert dict(td.headings)["delay"] == "Pacing" and dict(td.headings)["output_mode"] == "Language & output mode"
    assert {"use_rolling_summary", "rolling_summary_system_prompt", "contextual"} <= set(sections["context_memory"].keys)
    assert {"enable_gpt_thinking", "thinking_budget"} <= set(sections["thinking"].keys)
    assert "enable_thoughts" in sections["other.response"].keys  # devfix4: folded into the Streaming switch
    assert {"disable_gemini_safety", "gemini_service_tier"} <= set(sections["provider_options"].keys)
    assert {"pdf_extraction_workers", "enable_pdf_output"} <= set(sections["pdf"].keys)
    assert dict(sections["other.response"].headings)["max_retries"] == "Retries"
    assert sections["other.context"].group == "Advanced"
    assert sections["other.meta_data"].title == "Metadata, TOC & headers"
    # every schema key is still in exactly one section
    from glossarion_mobile.ui.settings.model import VIRTUAL_SPECS

    before = [k for s in settings_schema.sections() for k in s.keys]
    after = [k for s in schema.sections() for k in s.keys if k not in VIRTUAL_SPECS]  # U9: Context mode combo
    assert sorted(before) == sorted(after) and len(after) == len(set(after))
    # emptied desktop sections resolve; a key finds its new section
    assert schema.section("main.run").id == "translation_defaults"
    assert schema.section("other.output").id == "pdf"
    assert schema.resolve_section_id("other.processing.extraction", "disable_gemini_safety") == "provider_options"
    assert schema.section_for_key("output_mode").id == "translation_defaults"


@needs_flet
def test_section_page_pool_tiles_links_and_curated_headings(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.section_page import SectionPage

    async def scenario():
        _conn, session = _TB._fake_session("android")
        store, schema = _store(tmp_path, {"rolling_summary_keys": [{"api_key": "sk-1", "model": "m"}]})
        routes: list = []
        ctx = SettingsContext(page=session.page, store=store, schema=schema, navigate_route=routes.append)
        page = SectionPage(parse_route("/settings/s/context_memory"), ctx)
        session.page.views[0].controls.append(page.get_body())
        session.page.update()
        assert page.headings["use_rolling_summary"] == "Rolling summary"
        keys = [t.key for t in page.pool_tiles]
        assert keys == ["use_rolling_summary_keys", "rolling_summary_keys"]
        assert page.pool_tiles[1].summary() == "1 key"
        page.pool_tiles[1]._on_tap()
        assert routes[-1] == "/settings/keys/rolling_summary"
        response = SectionPage(parse_route("/settings/s/other.response"), ctx)
        response.get_body()
        link = next(c for c in response.list_view.controls if getattr(c, "key", "") == "link-settings.section-qa.ai_hunter")
        link.on_click(None)
        assert routes[-1] == "/settings/s/qa.ai_hunter"
        image = SectionPage(parse_route("/settings/s/other.image"), ctx)
        image.get_body()
        link = next(c for c in image.list_view.controls
                    if getattr(c, "key", "") == "link-settings.section-translation_defaults")
        link.on_click(None)
        assert routes[-1] == "/settings/s/translation_defaults#output_mode"
        legacy = SectionPage(parse_route("/settings/s/main.run#delay"), ctx)
        assert legacy.section_id == "translation_defaults" and legacy.focus_target == "delay"
        assert ctx.open_setting("other.processing.extraction", "disable_gemini_safety") == \
            "/settings/s/provider_options#disable_gemini_safety"
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# Notifications & background, full-screen composer, ＋ › From Library
# ==========================================================================


@needs_flet
def test_notifications_page_is_shipped_and_requests_permissions():
    from glossarion_mobile.ui.screens.notifications import NotificationsScreen
    from glossarion_mobile.ui.screens.pages_feature import IMPLEMENTED_ROUTES, SCREEN_ROUTES

    assert "settings.notifications" in SCREEN_ROUTES and "settings.notifications" in IMPLEMENTED_ROUTES
    prefs: dict = {}
    background = types.SimpleNamespace(is_android=True, is_ios=False, keep_screen_on=lambda: False,
                                       _pref=lambda key, default=None: prefs.get(key, default),
                                       _set_pref=prefs.__setitem__)
    asked: list = []

    async def request_notifications():
        asked.append("notifications")
        return "granted"

    screen = NotificationsScreen(None, types.SimpleNamespace(say=lambda m: None), background=background,
                                 request_notifications=request_notifications,
                                 request_battery=lambda: asked.append("battery") or "granted")
    screen.get_body()
    assert asyncio.run(screen.allow_notifications()) == "granted"
    assert asyncio.run(screen.allow_battery()) == "granted" and asked == ["notifications", "battery"]
    screen.keep_awake.value = True
    screen._on_keep_awake(types.SimpleNamespace(control=screen.keep_awake))
    from glossarion_mobile.services.background import PREF_KEEP_SCREEN_ON

    assert prefs[PREF_KEEP_SCREEN_ON] is True
    assert "30 seconds" in screen.get_body().controls[-1].content.controls[-1].value


@needs_flet
def test_full_screen_composer_writes_back_text_and_draft(monkeypatch):
    from glossarion_mobile.ui.chat import compose_screen
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.compose_screen import ComposeScreen
    from glossarion_mobile.ui.chat.integration import SCREEN_ROUTES

    counted: list = []
    monkeypatch.setattr(compose_screen, "count_tokens", lambda text, model: counted.append(model) or 42)

    assert "chat.compose" in SCREEN_ROUTES
    done: list = []
    closed: list = []
    screen = ComposeScreen(None, text="line 1\nline 2\nline 3", model="gpt-4o", on_done=done.append,
                           on_close=lambda: closed.append(True))
    screen.get_body()
    screen.field.value = "edited text " * 20
    assert asyncio.run(screen.update_tokens(debounce=0)) == "≈42 tok" and counted == ["gpt-4o"]
    assert screen.finish() == "edited text " * 20 and done == ["edited text " * 20] and closed == [True]
    # Done -> the composer text and the chat draft
    drafts: dict = {}
    composer = types.SimpleNamespace(text="", set_text=lambda t: setattr(composer, "text", t))
    fake = types.SimpleNamespace(composer=composer, bound=True, cid="5", _on_content_changed=lambda: None,
                                 _push=lambda *c: None,
                                 env=types.SimpleNamespace(chats=types.SimpleNamespace(
                                     set_draft=lambda cid, text: drafts.__setitem__(cid, text))))
    ChatView.apply_composed_text(fake, "new draft")
    assert composer.text == "new draft" and drafts == {"5": "new draft"}


@needs_flet
def test_plus_from_library_attaches_a_library_books_raw_file(tmp_path, monkeypatch):
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.tools import source_picker
    from glossarion_mobile.ui.tools.targets import ToolTarget

    raw = tmp_path / "Book.epub"
    raw.write_bytes(b"PK")
    shown: dict = {}

    class FakePicker:
        def __init__(self, ctx, **kwargs):
            shown.update(kwargs)

        def show(self, page):
            shown["shown"] = True
            return self

    monkeypatch.setattr(source_picker, "SourcePicker", FakePicker)
    attached: list = []
    fake = types.SimpleNamespace(env=types.SimpleNamespace(tools_context=lambda: object()), page=object(),
                                 navigate=lambda *a: attached.append(("nav",) + a),
                                 attach_library_book=attached.append)
    ChatView.open_library_picker(fake)
    # device fixes 2026-10-08: the in-chat Library picker (one segment, search, the Library's rows with
    # covers, the books without a raw file listed but disabled, "Open Library" in its header)
    assert shown["shown"] and shown["segment"] == "library" and shown["find_folder"] is False
    assert shown["segments"] == ("library",) and shown["searchable"] is True and shown["book_rows"] is True
    assert shown["include_unresolved"] is True and shown["query"] == "" and not shown["multi"]
    label, open_library = shown["header_action"]
    assert label == "Open Library"
    open_library()
    assert attached[-1] == ("nav", "library")
    eligible = shown["eligible"]
    assert eligible(ToolTarget(title="Book", source=str(raw), origin="library", kind="epub")) is None
    assert "No raw file" in eligible(ToolTarget(title="Gone", source=str(tmp_path / "gone.epub"), origin="library",
                                               kind="epub"))
    assert "No raw file" in eligible(ToolTarget(title="Shelf", origin="library", bid="b9"))  # unresolved row
    book = ToolTarget(title="Book", source=str(raw), origin="library", kind="epub", bid="b1")
    shown["on_done"]([book])
    assert attached[-1] is book
    ChatView.open_library_picker(fake, "novel")  # /library novel with several matches: prefilled
    assert shown["query"] == "novel"
    # without the Tools feature the Library opens instead
    fallback = types.SimpleNamespace(env=types.SimpleNamespace(tools_context=lambda: None), page=object(),
                                     navigate=lambda *a: attached.append(("nav",) + a))
    ChatView.open_library_picker(fallback)
    assert attached[-1] == ("nav", "library")


@needs_flet
def test_library_book_attaches_with_its_meta_and_forgets_it(tmp_path):
    """The picked Library book is attached and remembered as the chat's Library book (sidecar meta
    ``library_attachment``); removing it or attaching another file forgets it; Remove in the snackbar."""
    from glossarion_mobile.ui.chat.chat_view import LIBRARY_ATTACHMENT_META, ChatView
    from glossarion_mobile.ui.tools.targets import ToolTarget

    raw = tmp_path / "Book.epub"
    raw.write_bytes(b"PK")
    other = tmp_path / "Other.txt"
    other.write_text("hello", encoding="utf-8")
    meta: dict = {}
    notes: list = []
    removed: list = []

    class Chats:
        available = True

        def meta(self, cid):
            return dict(meta)

        def set_meta(self, cid, key, value):
            if value is None:
                meta.pop(key, None)
            else:
                meta[key] = value

        def set_attachment(self, cid, record):
            meta["_attachment"] = record

    attached: list = []
    view = types.SimpleNamespace(env=types.SimpleNamespace(chats=Chats()), bound=True, cid="3",
                                 notify=lambda message, **k: notes.append((message, k)),
                                 composer=types.SimpleNamespace(_remove_attachment=lambda: removed.append(True)))
    view.attach_file = lambda path: attached.append(path) or True
    target = ToolTarget(title="Book", source=str(raw), origin="library", kind="epub", bid="b1")
    assert ChatView.attach_library_book(view, target)
    assert attached == [str(raw)] and meta[LIBRARY_ATTACHMENT_META] == {"bid": "b1", "path": str(raw)}
    message, kwargs = notes[-1]
    assert message == "Attached Book.epub from the Library" and kwargs["action_label"] == "Remove"
    kwargs["on_action"]()
    assert removed == [True]
    assert not ChatView.attach_library_book(view, ToolTarget(title="Shelf", origin="library", bid="b2"))
    # another file (or the same one again) on the real view helpers
    view._forget_library_attachment = lambda keep_path="": ChatView._forget_library_attachment(view, keep_path)
    ChatView._forget_library_attachment(view, keep_path=str(raw))
    assert LIBRARY_ATTACHMENT_META in meta  # the same book stays the chat's Library book
    ChatView._forget_library_attachment(view, keep_path=str(other))
    assert LIBRARY_ATTACHMENT_META not in meta
    meta[LIBRARY_ATTACHMENT_META] = {"bid": "b1", "path": str(raw)}
    view.state = types.SimpleNamespace(output_mode=types.SimpleNamespace(value=types.SimpleNamespace(
        attachment_changed=lambda path: "mode")))
    view._set_mode = lambda mode: None
    view.refresh_send = lambda: None
    ChatView._on_attachment_removed(view)
    assert LIBRARY_ATTACHMENT_META not in meta and meta["_attachment"] is None


# ==========================================================================
# Multi-Key Manager
# ==========================================================================


def test_key_status_rule_is_shared_and_reads_the_live_pool(monkeypatch):
    import key_pool_service as kps
    from glossarion_mobile.ui.screens.keys import key_counts_text, key_overrides_text, key_status, key_status_info

    now = time.time()
    entry = {"api_key": "sk-abc", "model": "m", "enabled": True}
    live = {"api_key": "sk-abc", "model": "m", "is_cooling_down": True, "last_error_time": now - 10, "cooldown": 60,
            "success_count": 7, "error_count": 2, "times_used": 9}
    status, text = key_status_info(entry, live)
    assert status == "cooling" and text.startswith("Cooling (") and text.endswith("s)")
    assert key_status(entry) == "active"
    assert key_status({"api_key": "sk", "last_test_result": "timeout"}) == "failed"
    assert key_status_info({"api_key": "sk", "last_test_result": "rate_limited"})[1] == "Rate limited"
    assert key_status({"api_key": "ENC:x"}) == "encrypted" and key_status({"api_key": "k", "enabled": False}) == "disabled"
    assert key_counts_text(entry, live) == "✅ 7 · ❌ 2"
    assert key_overrides_text({"cooldown": 90, "individual_output_token_limit": 4096, "individual_key_temperature": 0.3,
                               "api_call_delay": 2}) == "⌛ 90s · limit 4096 · T 0.3 · delay 2s"
    assert key_overrides_text({"cooldown": 60}) == ""
    # the live stats come from the running client's pool (never imported here)
    pool_key = types.SimpleNamespace(api_key="sk-abc", model="m", success_count=3, error_count=1, times_used=4,
                                     is_cooling_down=False, last_error_time=None, cooldown=60)
    client = types.SimpleNamespace(_rolling_summary_key_pool=types.SimpleNamespace(keys=[pool_key]))
    monkeypatch.setitem(sys.modules, "unified_api_client", types.SimpleNamespace(UnifiedClient=client))
    rows = kps.live_key_stats("rolling_summary")
    assert rows[0]["success_count"] == 3 and kps.live_key_stats("fallback") == []
    assert kps.merge_live_stats([{"api_key": "x", "model": "m"}, entry], rows) == [None, rows[0]]


def test_keys_controller_copy_current_key_and_bulk_fields(tmp_path):
    from glossarion_mobile.ui.screens.keys import KeysController, parse_bulk_value

    store, _schema = _store(tmp_path, {"api_key": "sk-main-key-123456", "model": "gpt-4o",
                                       "rolling_summary_keys": [{"api_key": "sk-a1234567", "model": "m1"},
                                                                {"api_key": "sk-b1234567", "model": "m2"}]})
    controller = KeysController(store)
    entry = controller.current_key_entry("rolling_summary")
    assert entry["api_key"] == "sk-main-key-123456" and entry["model"] == "gpt-4o"
    changed, errors = controller.set_key_fields("rolling_summary", [0, 1], {"model": "gemini-2.5-pro"})
    assert changed == 2 and not errors
    assert [k["model"] for k in controller.keys("rolling_summary")] == ["gemini-2.5-pro"] * 2
    controller.set_key_fields("rolling_summary", [1], {"individual_key_temperature": 0.4, "api_call_delay": 3.0})
    keys = controller.keys("rolling_summary")
    assert keys[1]["individual_key_temperature"] == 0.4 and keys[1]["api_call_delay"] == 3.0
    assert "individual_key_temperature" not in keys[0] or keys[0]["individual_key_temperature"] is None
    assert parse_bulk_value("cooldown", "5")[1] == "Cooldown (seconds): 10–3600"
    assert parse_bulk_value("cooldown", "120") == ({"cooldown": 120}, None)
    assert parse_bulk_value("individual_key_temperature", "x")[1] == "Enter a number"
    store._saver.close()


@needs_flet
def test_bulk_field_sheet_values():
    from glossarion_mobile.ui.screens.keys import BulkFieldSheet

    applied: list = []
    sheet = BulkFieldSheet("api_call_delay", {"api_call_delay": 1.5}, count=2,
                           on_apply=lambda values: applied.append(values))
    sheet.page = types.SimpleNamespace(show_dialog=lambda d: None, pop_dialog=lambda: None)
    sheet.input.value = "4"
    assert sheet.apply() is None and applied[-1] == {"api_call_delay": 4.0}
    assert sheet.can_clear and sheet.values(clear=True) == ({"api_call_delay": 0.0}, None)
    params = BulkFieldSheet("request_parameters", {}, count=1, on_apply=applied.append)
    params.input.value = "[1]"
    assert params.values() == (None, "Enter a JSON object")
    params.input.value = '{"top_p": 0.9}'
    assert params.values() == ({"request_parameters": {"top_p": 0.9}}, None)
    endpoint = BulkFieldSheet("endpoint", {}, count=1, on_apply=applied.append)
    endpoint.endpoint_switch.value = True
    assert endpoint.values()[1] == "Enter the endpoint URL"
    assert endpoint.values(clear=True) == ({"use_individual_endpoint": False}, None)
    cooldown = BulkFieldSheet("cooldown", {}, count=1, on_apply=applied.append)
    assert not cooldown.can_clear


# ==========================================================================
# sign-in lines, Check environment, Gemini status
# ==========================================================================


def test_sign_in_lines_of_every_provider_name_the_slot():
    from glossarion_mobile.services.jobs import SIGN_IN_MARKERS, sign_in_event, sign_in_provider

    assert sign_in_provider("🔄 AuthCD: No valid token found – opening browser for login…") == "authcd"
    assert sign_in_provider("🔄 AuthGem: No valid token found – starting browser login…") == "authgem"
    assert sign_in_provider("AuthGrok: sign in to Grok from Glossarion's Accounts screen (the app ...)") == "authgrok"
    assert sign_in_provider("🔄 AuthGPT: No valid token found – starting browser login…") == "authgpt"
    assert sign_in_provider("Translated chapter 3") is None
    assert "AuthGPT: Session expired" in SIGN_IN_MARKERS
    assert sign_in_event("authcd2/claude-sonnet-4", "authcd")["account_id"] == 2
    assert sign_in_event("gpt-4o", "authgem")["account_id"] == 0


@needs_flet
def test_sign_in_event_opens_the_providers_login_sheet():
    from glossarion_mobile.ui.screens.jobs import JobsFeature, sign_in_label

    assert sign_in_label("authcd", 2) == "Claude #2" and sign_in_label("authgpt") == "ChatGPT"
    notes: list = []
    spawned: list = []
    opened: list = []

    async def sign_in_required(label):
        return label

    fake = types.SimpleNamespace(
        notifications=types.SimpleNamespace(sign_in_required=sign_in_required),
        spawn=lambda coro: spawned.append(coro) or coro.close(),
        _notify=lambda message, action=None, on_action=None: notes.append((message, action, on_action)),
        open_login_sheet=lambda provider, account: opened.append((provider, account)))
    JobsFeature._on_job_event(fake, "j1", "sign_in_required", {"provider": "authgem", "account_id": 3})
    assert notes[0][0] == "Sign-in required for Gemini #3" and notes[0][1] == "Sign in"
    notes[0][2]()
    assert opened == [("authgem", 3)] and spawned


def test_check_environment_runs_the_shared_methods_and_redacts():
    from glossarion_mobile.ui.screens.env_preview import redact_log_lines, run_env_check

    class Owner:
        def __init__(self, config, host):
            self.config = config
            self.host = host

        def initialize_environment_variables(self):
            assert self.config["show_debug_buttons"] is True  # the desktop button only exists in debug mode
            self.host.log("🚀 [INIT] Initializing all environment variables from config...")
            os.environ["U9_ENV_CHECK_PROBE"] = "1"
            return True

        def debug_environment_variables(self, show_all=False):
            assert show_all
            self.host.log('✅ [ENV_DEBUG] FALLBACK_KEYS: [{"api_key": "sk-secretsecret123", "model": "m"}]')
            self.host.log("✅ [ENV_DEBUG] TOP_P: 0.9")
            self.host.log("❌ [ENV_DEBUG] CRITICAL MISSING: OUTPUT_LANGUAGE - Target language")
            return False

    result = run_env_check({"api_key": "sk-secretsecret123"}, lock=threading.Lock(),
                           owner_factory=lambda config, host, api_key: Owner(config, host))
    assert result.ok and not result.passed and result.verdict.startswith("❌")
    text = "\n".join(result.lines)
    assert "sk-secretsecret123" not in text and "FALLBACK_KEYS: <REDACTED>" in text
    assert "TOP_P: 0.9" in text and "CRITICAL MISSING: OUTPUT_LANGUAGE" in text
    assert "U9_ENV_CHECK_PROBE" not in os.environ  # the check runs in the scoped process env
    assert redact_log_lines(["plain sk-secretsecret123 here"], ["sk-secretsecret123"])[0] == \
        "plain <REDACTED> (18 chars) here"


@needs_flet
def test_gemini_status_sheet_is_shared_by_accounts_and_the_model_sheet():
    from glossarion_mobile.ui.screens.accounts import gemini_status_sheet, open_gemini_status

    oauth = types.SimpleNamespace(open_url=lambda url: None,
                                  gemini_status=lambda account: {"verified": False, "verification_url": "https://v",
                                                                 "verification_message": "Verify your account"})
    sheet = gemini_status_sheet(oauth, oauth.gemini_status(2), 2)
    assert sheet.title == "Gemini #2 status" and "Verify your account" in sheet.body

    async def io(fn, *args):
        return fn(*args)

    shown = asyncio.run(open_gemini_status(oauth, 0, io=io, page=None))
    assert shown.title == "Gemini status"


# ==========================================================================
# Glossary
# ==========================================================================


def test_schema_overlay_for_the_refinement_request_mode_and_internal_flags():
    import settings_schema as s

    spec = s.spec("glossary_refinement_chunking_mode")
    assert spec.section == "glossary.refinement" and spec.label == "Request mode" and spec.type == "choice"
    assert [c[0] for c in spec.choices] == ["separate", "all"] and "token budget" in spec.tooltip
    assert s.spec("custom_field_description_removed").section == "internal.state"
    for key in ("enable_image_translation", "enable_image_output_mode", "enable_video_output_mode",
                "enable_audio_output_mode", "enable_refinement_output_mode"):
        assert s.spec(key).readonly == "Follows Output mode"


def test_entry_type_and_custom_field_rules_are_shared():
    import glossary_document as gd

    types_ = {"character": {"enabled": True, "has_gender": True}, "term": {"enabled": True, "has_gender": False}}
    gd.normalize_legacy_entry_types(types_)
    assert "terms" in types_ and "term" not in types_
    assert gd.add_entry_type(types_, "  Skills ", True) == ("skills", None)
    assert types_["skills"] == {"enabled": True, "has_gender": True}
    assert gd.add_entry_type(types_, "skills", False) == (None, ("Duplicate Type", "Type 'skills' already exists"))
    assert gd.add_entry_type(types_, "  ", False) == (None, ("Invalid Input", "Please enter a type name"))
    assert gd.entry_type_remove_warning("character") == ("Cannot Remove", "Built-in types cannot be removed")
    assert gd.entry_type_remove_warning("skills") is None
    assert [name for name, _c in gd.sorted_entry_types(types_)] == ["character", "terms", "skills"]
    assert gd.custom_fields_flag_updates(["description", "notes"], ["notes"]) == {"custom_field_description_removed": True}
    assert gd.custom_fields_flag_updates(["notes"], ["notes", "Description"]) == {"custom_field_description_removed": False}
    assert gd.custom_fields_flag_updates(["notes"], ["notes", "x"]) == {}


@needs_flet
def test_entry_type_editors_and_the_custom_fields_tile(tmp_path):
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.tiles import (
        CustomFieldsTile,
        EntryTypePickerTile,
        EntryTypesTile,
        make_tile,
    )

    store, schema = _store(tmp_path, {"custom_entry_types": {"character": {"enabled": True, "has_gender": True},
                                                             "term": {"enabled": True, "has_gender": False}},
                                      "custom_glossary_fields": ["description", "notes"]})
    page = types.SimpleNamespace(show_dialog=lambda d: None, pop_dialog=lambda: None, update=lambda *a: None)
    ctx = SettingsContext(page=page, store=store, schema=schema)
    tile = make_tile(schema.spec("custom_entry_types"), ctx)
    assert isinstance(tile, EntryTypesTile) and tile.summary() == "2 types · 2 enabled"
    editor = tile.activate()
    assert "terms" in editor.types  # legacy name normalised like the desktop
    assert editor.add("Items", False) == "items" and editor.add("items") is None
    assert editor.remove("character") is False and editor.remove("items") is True
    editor.switches["terms"].value = False
    assert editor.save() and store.get("custom_entry_types")["terms"]["enabled"] is False
    picker_tile = make_tile(schema.spec("glossary_refinement_selected_types"), ctx)
    assert isinstance(picker_tile, EntryTypePickerTile)
    picker = picker_tile.activate()
    assert picker.options[:2] == ["character", "terms"]
    picker.boxes["terms"].value = True
    assert picker.save() and store.get("glossary_refinement_selected_types") == ["terms"]
    fields = make_tile(schema.spec("custom_glossary_fields"), ctx)
    assert isinstance(fields, CustomFieldsTile)
    assert fields.apply(["notes"]) and store.get("custom_field_description_removed") is True
    assert fields.apply(["notes", "description"]) and store.get("custom_field_description_removed") is False
    store._saver.close()


@needs_flet
def test_glossary_settings_tabs_have_search_menu_and_focus(tmp_path):
    from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsScreen
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    store, schema = _store(tmp_path)
    settings = SettingsContext(page=None, store=store, schema=schema)
    gctx = types.SimpleNamespace(settings=settings, feature=None, say=lambda *a, **k: None,
                                 go=lambda *a, **k: None, service=None, post_ui=lambda fn: fn(), tablet=False)
    match = parse_route("/settings/s/glossary.general#compress_glossary_shadow_log")
    screen = GlossarySettingsScreen(match, gctx, "general")
    actions = screen.actions()  # the shell asks before the body
    assert len(actions) == 2 and screen.tab.page is not None
    assert screen.tab.page.focus_target == "compress_glossary_shadow_log"
    assert "compress_glossary_shadow_log" in screen.tab.page.keys
    assert any(getattr(c, "key", "") == "gs-book-title" for c in screen.tab._links())
    # Settings home's wide detail pane asks the section-screen hook first
    built: list = []
    settings.extras["section_screen"] = lambda m: built.append(m.params["section"]) or screen
    home = SettingsHome(parse_route("/settings"), settings)
    assert home._section_screen(match) is screen and built == ["glossary.general"]
    store._saver.close()


def test_cbz_is_expanded_with_the_shared_manga_extractor(tmp_path):
    from glossarion_mobile.services.glossary import GlossaryService

    cbz = tmp_path / "Vol 1.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        zf.writestr("p2.png", b"\x89PNG2")
        zf.writestr("p1.png", b"\x89PNG1")
        zf.writestr("notes.txt", "x")
    data = tmp_path / "data"
    service = GlossaryService(paths=types.SimpleNamespace(data=str(data)))
    images = service.expand_cbz(str(cbz))
    assert [os.path.basename(p) for p in images] == ["p1.png", "p2.png"]
    assert all(str(data / "Inbox" / "_cbz") in p for p in images)
    from glossarion_mobile.ui.glossary.sheets import SOURCE_EXTENSIONS

    assert "cbz" in SOURCE_EXTENSIONS


# ==========================================================================
# workspace-collision rename (desktop selection rule) in the adapters
# ==========================================================================


def _rename_owner():
    from run_env import RunEnvMixin

    class Owner(RunEnvMixin):
        def __init__(self):
            self.config = {}
            self.logs: list = []

        def append_log(self, message):
            self.logs.append(message)

    return Owner()


def test_translate_adapter_renames_app_owned_copies_on_a_workspace_collision(tmp_path, _isolated):
    import library_core as lc
    from glossarion_mobile.job_kinds.translate import check_inputs, resolve_workspace_collisions

    out = Path(os.environ["OUTPUT_DIRECTORY"])
    inbox = Path(os.environ["GLOSSARION_DATA_DIR"]) / "Inbox"
    inbox.mkdir(parents=True)
    raw_dir = Path(lc.library_root_path()) / "Raw"
    raw_dir.mkdir(parents=True)
    for stem in ("Novel", "Saga"):
        (out / stem).mkdir(parents=True)
        (out / stem / "source_epub.txt").write_text(str(tmp_path / f"{stem}.pdf"), encoding="utf-8")
    inbox_copy = inbox / "Novel.epub"
    inbox_copy.write_bytes(b"PK")
    library_copy = raw_dir / "Saga.epub"
    library_copy.write_bytes(b"PK")
    lc.record_library_raw_input(str(library_copy))
    original = tmp_path / "user" / "Novel.epub"
    original.parent.mkdir()
    original.write_bytes(b"PK")
    results: dict = {}
    ctx = types.SimpleNamespace(owner=_rename_owner(), log=lambda m: None, set_result=lambda **d: results.update(d))
    renamed = resolve_workspace_collisions(ctx, [str(inbox_copy), str(library_copy), str(original)])
    assert renamed[0] == str(inbox / "Novel_EPUB.epub") and os.path.isfile(renamed[0])
    assert renamed[1] == str(raw_dir / "Saga_EPUB.epub")
    assert renamed[2] == str(original) and original.exists()  # a user original is never renamed
    assert results["renamed_inputs"] == {str(inbox_copy): renamed[0], str(library_copy): renamed[1]}
    registered = {os.path.normcase(p) for p in lc.load_library_raw_inputs()}
    assert os.path.normcase(renamed[1]) in registered and os.path.normcase(str(library_copy)) not in registered
    assert any("workspace collision" in line for line in ctx.owner.logs)
    # a resumed job still names the old copy: it finds the renamed one
    assert check_inputs([str(inbox_copy)]) == [renamed[0]]


def test_rename_moved_into_run_env_and_desktop_calls_it():
    import run_env

    assert hasattr(run_env.RunEnvMixin, "_rename_input_for_existing_workspace_collision")
    source = (SRC_DIR / "translator_gui.py").read_text(encoding="utf-8-sig")
    assert "def _rename_input_for_existing_workspace_collision" not in source
    assert "self._rename_input_for_existing_workspace_collision(path)" in source


# ==========================================================================
# Library card Reader items, SDLXLIFF tablet, Manga
# ==========================================================================


def test_library_card_reader_items_on_real_in_progress_rows(tmp_path, monkeypatch):
    import library_core as lc
    from glossarion_mobile.ui.library.home import reader_actions

    out = tmp_path / "rows_out"
    monkeypatch.setattr(lc, "_resolve_output_roots", lambda config=None: [str(out)])
    progress = {"chapters": {"1": {"status": "completed", "output_file": "response_001.html"},
                             "2": {"status": "pending"}}, "version": "2.1"}
    raw_pdf = tmp_path / "Manual.pdf"
    raw_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    for stem, source in (("Manual", raw_pdf), ("Gone", tmp_path / "Gone.epub")):
        folder = out / stem
        folder.mkdir(parents=True)
        (folder / "source_epub.txt").write_text(str(source), encoding="utf-8")
        (folder / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
        (folder / "response_001.html").write_text("<p>x</p>", encoding="utf-8")
    with zipfile.ZipFile(out / "Gone" / "Gone.epub", "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
    rows = {row["folder_name"]: row for row in lc.scan_output_folders({})}
    assert rows["Manual"]["type"] == rows["Gone"]["type"] == "in_progress"
    assert [(label.split(" ", 1)[1], how) for label, how, _p in reader_actions(rows["Manual"])] == [
        ("Open in EPUB reader", "book")]
    gone = reader_actions(rows["Gone"])
    assert [(label.split(" ", 1)[1], how) for label, how, _p in gone] == [("Open Translated EPUB", "translated")]
    assert gone[0][2] == rows["Gone"]["output_epub_path"]
    # device fixes 2026-10-08: the Reader opens TXT books (text mode), so a TXT card has the Reader item
    assert reader_actions({"type": "txt", "path": "x.txt"}) == [("\U0001f4d6 Open in Reader", "book", "x.txt")]
    assert reader_actions({"type": "epub", "path": "b.epub"})[0][0].endswith("Open in Reader")


@needs_flet
def test_sdlxliff_tablet_side_list_and_size_class_switch():
    from glossarion_mobile.ui.tools.sdlxliff import SdlxliffScreen

    ctx = types.SimpleNamespace(tablet=True, prefs=None, spawn=lambda coro: coro.close(), say=lambda *a: None,
                                mono="monospace")
    screen = SdlxliffScreen(None, ctx)
    body = screen.get_body()
    assert screen.md is not None and body.content is screen.md.control
    pieces = [{"rows": [{"status": "red"}], "red_count": 1}, {"rows": [], "manual_green_override": True}]
    session = types.SimpleNamespace(piece_summary=lambda i: {"red_count": pieces[i].get("red_count", 0),
                                                             "completed": bool(pieces[i].get("manual_green_override"))})
    screen.binding = types.SimpleNamespace(pieces=pieces, books=[{"label": "A"}, {"label": "B"}], session=session,
                                           label=lambda i: f"Piece {i + 1}", select=lambda i: None,
                                           provider=lambda: "auto", provider_labels=lambda: {"auto": "Auto"},
                                           status="")
    screen.render()
    titles = [getattr(getattr(c, "title", None), "value", None) for c in screen.side_list.controls]
    assert "Piece 1" in titles and "Piece 2" in titles and "A" in titles
    piece_rows = [c for c in screen.side_list.controls if getattr(getattr(c, "title", None), "value", "").startswith("Piece")]
    assert piece_rows[0].leading.bgcolor == "#dc3545" and piece_rows[1].leading.bgcolor == "#28a745"
    assert not screen.piece_dropdown.visible
    screen.select_piece(1)
    assert screen.piece_index == 1
    screen.apply_size_class(types.SimpleNamespace(persistent_sidebar=False))
    assert screen.md is None and screen.root.content.content is screen.list_view and screen.piece_dropdown.visible


def test_manga_reason_chips_name_the_reason():
    from glossarion_mobile.services import manga as svc

    # U12 item 1: PyTorch-only rows are not listed on mobile; their reason still names it (value_reason)
    chips = {row.value: row.chip for row in svc.detector_rows(mobile=True)}
    assert "rtdetr" not in chips and "yolo" not in chips
    assert "hybrid" not in {row.value for row in svc.inpaint_method_rows(mobile=True)}
    assert svc.chip_text(svc.value_reason("manga_inpaint_method", "hybrid", mobile=True)) == "Needs PyTorch"
    assert svc.chip_text("Some long reason without any separator words at all here").endswith("…")
    assert svc.chip_text("RT-DETR (something)", "RT-DETR (PyTorch)") == "Not available on mobile"


@needs_flet
def test_manga_font_bounds_and_slider_ranges():
    from glossarion_mobile.services import manga as svc
    from glossarion_mobile.ui.tools.manga import settings as ms

    config: dict = {}

    def set_many(updates):
        for key, value in updates.items():
            if isinstance(key, tuple):
                node = config
                for part in key[:-1]:
                    node = node.setdefault(part, {})
                node[key[-1]] = value
            else:
                config[key] = value

    fake = types.SimpleNamespace(get=lambda key, default=None: svc.effective_setting(config, key, default),
                                 set_many=set_many, refresh=lambda: None, _font_bounds=ms.SettingsTab._font_bounds)
    ms.SettingsTab.set_font_bound(fake, "max", "60")
    assert config["manga_max_font_size"] == config["manga_settings"]["rendering"]["auto_max_size"] == 60
    assert config["manga_settings"]["font_sizing"]["max_size"] == 60
    low, high = ms.SettingsTab.set_font_bound(fake, "min", "75")  # above the maximum: lowered to it
    assert (low, high) == (60, 60) and config["manga_settings"]["font_sizing"]["min_size"] == 60
    assert ms.SettingsTab.set_font_bound(fake, "max", "5000")[1] == 999
    assert "manga_settings.rendering.auto_max_size" in ms.CURATED_KEYS
    source = (APP_DIR / "glossarion_mobile" / "ui" / "tools" / "manga" / "settings.py").read_text(encoding="utf-8")
    assert '"manga_settings", "font_sizing", "line_spacing"), 1.0, 2.0, 20' in source
    assert '"manga_safe_area_scale", 0.70, 1.10, 40' in source


def test_image_document_types_reach_the_app():
    import tomllib

    data = tomllib.loads((MOBILE_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    docs = data["tool"]["flet"]["ios"]["info"]["CFBundleDocumentTypes"]
    image = next(d for d in docs if d["CFBundleTypeName"] == "Image")
    assert image["LSItemContentTypes"] == ["public.image"] and image["CFBundleTypeRole"] == "Viewer"
    manifest = (MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src" / "flutter" / "flet_glossarion_native"
                / "android" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
    view = manifest.split('android.intent.action.VIEW', 1)[1].split("</intent-filter>", 1)[0]
    assert 'android:mimeType="image/*"' in view
