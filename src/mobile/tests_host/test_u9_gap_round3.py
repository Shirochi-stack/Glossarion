"""Host tests for the U9 feature-gap closures (audit round 3).

* chat job cards act on their own turn workspace: Open output (Files at the workspace), Compile ▾
  EPUB / PDF (``ChatRuns.compile(folder=)``), output chips (``chat_ops.workspace_outputs``), Share /
  Export; Jobs › job › Files of a chat job opens its persisted Attachments workspace;
* MessageMoreSheet: Glossary terms used, Copy as Markdown / HTML / plain text, the Add-term target picker;
* ModelSheet "Use “<typed id>”" (the desktop editable model box);
* settings: Refusal patterns link, mobile notes / partial-route chip, count-style keys are never secret,
  the compression-factor locks and recomputes, Configure All › Advanced + Reset, the TTS endpoint,
  image compression switch placement, junk / removed tiles, the Advanced filter chip, the QA emoticon
  phrase list, the manga custom-detector / experimental-tools locks, Single Pass header prompt default,
  the metadata translation mode choices;
* Library / Book page: "Metadata Already Exists", 🔊 Play audio in the MediaViewer, the Overview's
  landscape columns and hero skeleton, the Chapters banner's Full refresh, the glossary file card, the
  Output tab's workspace groups, Files › Open with › QA report viewer;
* Reader: imported fonts in Aa (served by ReaderServer), "Text size in Aa", the tablet chapters panel;
* Tools ErrorCards; manga fonts and the editor strip's skip toggle; the bundled user guide.

Real data is never touched: HOME / USERPROFILE / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY /
GLOSSARION_DATA_DIR point at pytest's tmp dir (autouse fixture).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u9_gap_round3.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
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


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


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


def _ctx(tmp_path, data=None, **kw):
    from glossarion_mobile.ui.settings.context import SettingsContext

    store, schema = _store(tmp_path, data)
    return SettingsContext(page=None, store=store, schema=schema, **kw)


# ==========================================================================
# settings: schema overlay and tiles
# ==========================================================================


def test_only_credentials_render_as_secret_tiles():
    import settings_schema as ss
    from glossarion_mobile.ui.settings import model

    secret = {spec.key for spec in ss.all_specs() if model.tile_kind(spec) == "secret"}
    allowed = {key for key in secret if key == "api_key" or key.endswith("api_key")}
    assert secret == allowed, sorted(secret - allowed)
    assert model.tile_kind(ss.spec("glossary_max_output_tokens")) == "number"
    assert ss.spec("glossary_max_output_tokens").type == "int"
    assert not ss.is_available("multi_api_key_tree_font_size")[0]
    assert not ss.is_available("multi_api_key_tree_heights")[0]


def test_emoticon_patterns_get_the_phrase_list_editor():
    import settings_schema as ss
    from glossarion_mobile.ui.settings import model

    spec = ss.spec("qa_scanner_settings.emoticon_patterns")
    assert model.sample_value(spec) is None
    assert model.tile_kind(spec) == "list"
    default = ss.effective_default("qa_scanner_settings.emoticon_patterns")
    assert isinstance(default, list) and default and all(isinstance(v, str) for v in default)


def test_schema_overlay_choices_labels_and_locks():
    import settings_schema as ss

    assert list(ss.choice_values("lang_prompt_behavior")) == ["auto", "never", "always"]
    assert ss.coerce("lang_prompt_behavior", "whatever") == "auto"
    assert list(ss.choice_values("metadata_translation_mode")) == ["together", "metadata_separate", "parallel"]
    assert ss.spec("lang_prompt_behavior").section == ss.spec("forced_source_lang").section == "other.meta_data"
    assert ss.spec("use_header_as_output").readonly.startswith("Disabled on desktop too")
    for key in ("has_gender", "legacy_structure"):
        assert ss.spec(key).section == "internal.state" and ss.spec(key).readonly
    for key in ("manga_settings.ocr.custom_model_path", "experimental_translate_all"):
        assert not ss.is_available(key)[0]
    assert "AuthZA / Arena / Antigravity / OcAgy" in ss.MOBILE_PARTIAL_REASONS["allow_authgpt_batch_stream_logs"]
    assert any("excluded on mobile" in note for note in ss.spec("allow_authgpt_batch_stream_logs").discrepancies)
    assert ss.spec("compression_factor").locked_if == "lock:compression_factor"
    assert ss.spec("glossary_compression_factor").locked_if == "lock:glossary_compression_factor"
    assert ss.spec("openai_tts_endpoint").section == "other.endpoints"
    assert "TTS" in ss.spec("openai_tts_endpoint").label


def test_single_pass_header_prompt_shows_the_built_in_text():
    import extract_glossary_from_epub as eg
    import settings_schema as ss

    spec = ss.spec("single_pass_glossary_header_prompt")
    assert spec.section == "glossary.balanced_full"
    assert ss.effective_default("single_pass_glossary_header_prompt") == eg.DEFAULT_SINGLE_PASS_GLOSSARY_HEADER_PROMPT
    assert any("built-in text" in note for note in spec.discrepancies)


def test_curated_sections_take_the_image_switch_and_the_refusal_limit():
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    epub = schema.section("epub_output")
    headings = dict(epub.headings)
    assert headings["enable_image_compression"] == headings["image_compression_quality"] == "Image compression"
    assert dict(schema.section("other.response").headings)["refusal_pattern_length_limit"] == "Safety checks"
    assert "enable_image_compression" not in schema.section("other.image").keys


def test_heading_overrides_fix_the_generator_groups():
    import settings_schema as ss
    from glossarion_mobile.ui.settings import model

    assert model.sub_heading(ss.spec("openai_base_url")) == "Custom OpenAI endpoint"
    assert model.sub_heading(ss.spec("openai_tts_endpoint")) == "Text-to-speech"
    assert model.sub_heading(ss.spec("output_directory")) == "Output folder"
    assert model.sub_heading(ss.spec("single_pass_glossary_header_prompt")) == model.sub_heading(
        ss.spec("manual_glossary_prompt3"))


def test_compression_factor_writes_recompute_and_lock(tmp_path):
    from glossarion_mobile.state.setting_writes import write_setting

    store, schema = _store(tmp_path)
    write_setting(store, "max_output_tokens", 20000)
    assert store.get("compression_factor") == 2.0
    write_setting(store, "glossary_max_output_tokens", -1)
    assert store.get("glossary_compression_factor") == 1.2
    write_setting(store, "glossary_max_output_tokens", 70000)
    assert store.get("glossary_compression_factor") == 1.5
    write_setting(store, "auto_compression_factor", False)
    assert store.get("manual_chunk_size")
    from glossarion_mobile.ui.settings.tiles import EffectiveConfig

    assert not schema.lock_reason(schema.spec("compression_factor"), EffectiveConfig(store))
    write_setting(store, "auto_compression_factor", True)
    assert schema.lock_reason(schema.spec("compression_factor"), EffectiveConfig(store))
    store._saver.close()


@needs_flet
def test_tiles_for_tokens_partial_chip_and_tts_follow(tmp_path):
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.settings.tiles import NumberTile, make_tile

    ctx = _ctx(tmp_path)
    tile = make_tile(ctx.schema.spec("glossary_max_output_tokens"), ctx)
    assert isinstance(tile, NumberTile) and tile.apply("4096") and ctx.store.get("glossary_max_output_tokens") == 4096
    stream = make_tile(ctx.schema.spec("allow_authgpt_batch_stream_logs"), ctx)
    chips = [c for c in stream.badges.controls if isinstance(c, ReasonChip)]
    assert chips and "excluded on mobile" in chips[0].reason
    base_url = make_tile(ctx.schema.spec("openai_base_url"), ctx)
    assert base_url.apply("http://10.0.0.2:8000/v1/audio/speech")
    assert ctx.store.get("openai_tts_endpoint") == "http://10.0.0.2:8000/v1/audio/speech"
    assert base_url.apply("http://10.0.0.2:11434/v1")
    assert ctx.store.get("openai_tts_endpoint") == "http://10.0.0.2:8000/v1/audio/speech"  # typing never clears it
    lang = make_tile(ctx.schema.spec("lang_prompt_behavior"), ctx)
    assert lang.kind == "dropdown"
    ctx.store._saver.close()


@needs_flet
def test_endpoints_quick_paste_and_clear_reset_the_tts_endpoint(tmp_path):
    from glossarion_mobile.ui.screens import endpoints as ep

    ctx = _ctx(tmp_path, {"openai_tts_endpoint": "http://old/audio/speech"})
    screen = ep.EndpointsScreen(None, ctx)
    screen.build_body()
    assert "openai_tts_endpoint" in screen.tiles
    base = screen.tiles["openai_base_url"]
    assert screen.quick_paste(base, "http://192.168.1.10:11434/v1")
    assert ctx.store.get("openai_tts_endpoint") == ""
    assert screen.quick_paste(base, "http://192.168.1.10:8000/audio/speech")
    assert ctx.store.get("openai_tts_endpoint") == "http://192.168.1.10:8000/audio/speech"
    assert screen.quick_paste(base, "")  # Clear
    assert ctx.store.get("openai_base_url") == "" and ctx.store.get("openai_tts_endpoint") == ""
    ctx.store._saver.close()


@needs_flet
def test_refusal_patterns_link_runs_the_registered_action(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.section_page import SectionPage

    ctx = _ctx(tmp_path)
    page = SectionPage(parse_route("/settings/s/other.response"), ctx)
    page.get_body()
    keys = [getattr(c, "key", "") for c in page.list_view.controls]
    assert "link-action-refusal_patterns" not in keys  # no feature registered it
    opened = []
    ctx.extras["actions"] = {"refusal_patterns": lambda: opened.append(True)}
    page = SectionPage(parse_route("/settings/s/other.response"), ctx)
    page.get_body()
    link = next(c for c in page.list_view.controls if getattr(c, "key", "") == "link-action-refusal_patterns")
    link.on_click(None)
    assert opened == [True]
    image = SectionPage(parse_route("/settings/s/other.image"), ctx)
    image.get_body()
    assert any(getattr(c, "key", "") == "link-settings.section-epub_output" for c in image.list_view.controls)
    ctx.store._saver.close()


def test_models_keys_feature_registers_the_refusal_action():
    from glossarion_mobile.ui.screens.model_manager import ModelsKeysFeature

    extras: dict = {}
    feature = ModelsKeysFeature.__new__(ModelsKeysFeature)
    feature.ctx = types.SimpleNamespace(extras=extras)
    # the attach step only (the rest needs a running app)
    actions = extras.setdefault("actions", {})
    actions["refusal_patterns"] = feature.open_refusal_patterns
    src = Path(sys.modules[ModelsKeysFeature.__module__].__file__).read_text(encoding="utf-8")
    assert 'extras.setdefault("actions", {})["refusal_patterns"] = self.open_refusal_patterns' in src


@needs_flet
def test_advanced_filter_chip_keeps_the_advanced_group(tmp_path):
    from glossarion_mobile.ui.settings.search import FILTERS, find_settings

    assert ("advanced", "Advanced") in FILTERS
    ctx = _ctx(tmp_path)
    hits = find_settings(ctx, "", ["advanced"], limit=500)
    assert hits and {h.group for h in hits} == {"Advanced"}
    assert {"lang_prompt_behavior", "has_gender"} & {h.key for h in hits} == {"has_gender"}
    ctx.store._saver.close()


def test_configure_all_has_the_advanced_tab_and_resets_like_desktop():
    from metadata_defaults import METADATA_PROMPT_RESET_KEYS
    from glossarion_mobile.ui.tools import headers_model as hm

    groups = dict(hm.PROMPT_GROUPS)
    assert groups["Advanced"] == ("lang_prompt_behavior", "forced_source_lang", "output_language")
    config = {key: "custom" for key in METADATA_PROMPT_RESET_KEYS}
    config["model"] = "m"
    removed, written = hm.reset_prompt_changes(config)
    assert set(removed) == set(METADATA_PROMPT_RESET_KEYS) - set(written)
    assert written["book_title_prompt"] == "" and "batch_header_prompt" in written and "model" not in written


# ==========================================================================
# ModelSheet: the typed model id
# ==========================================================================


@needs_flet
def test_model_sheet_offers_the_typed_model_id():
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    selected: list = []
    sheet = ModelSheet(current_model="gpt-6", models=["gpt-6", "gpt-6-mini"], env=SheetEnv(),
                       on_select=lambda *a: selected.append(a))
    sheet.query = "openrouter/new-vendor/brand-new-model"
    rows = sheet.model_rows()
    assert getattr(rows[0], "key", "") == "model-free-text"
    sheet.query = "gpt-6"
    assert getattr(sheet.model_rows()[0], "key", "") != "model-free-text"
    sheet.query = "ollamapull/qwen3:8b"
    assert sheet.model_rows()[0].disabled  # an excluded route is shown with its reason
    sheet.query = "chutes/some/model"
    sheet.submit_query()
    assert selected and selected[-1][:2] == ("model", "chutes/some/model")


# ==========================================================================
# chat: job card outputs, compile, open output, message sheet
# ==========================================================================


def _workspace(tmp_path):
    folder = tmp_path / "out" / "Direct Text" / "Chat 1" / "Attachments" / "story"
    (folder / "SDLXLIFF").mkdir(parents=True)
    (folder / "story.epub").write_bytes(b"PK")
    (folder / "story_translated.txt").write_text("x", encoding="utf-8")
    (folder / "story.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
    (folder / "SDLXLIFF" / "story.sdlxliff").write_text("<xliff/>", encoding="utf-8")
    (folder / "glossary.csv").write_text("type,raw_name,translated_name\ncharacter,김철수,Kim Cheolsu\n", encoding="utf-8")
    (folder / "response_001.html").write_text("<p>x</p>", encoding="utf-8")
    return folder


def test_workspace_outputs_and_turn_workspace(tmp_path):
    from glossarion_mobile.ui.chat.chat_ops import turn_workspace, workspace_outputs

    folder = _workspace(tmp_path)
    kinds = [(os.path.basename(p), k) for p, k in workspace_outputs(str(folder))]
    assert kinds[0] == ("story.epub", "epub")
    assert ("story_translated.txt", "txt") in kinds and ("story.srt", "subtitle") in kinds
    assert ("story.sdlxliff", "sdlxliff") in kinds and ("glossary.csv", "glossary") in kinds
    assert not any(name == "response_001.html" for name, _k in kinds)
    request_folder = folder / "story_output"
    request_folder.mkdir()
    messages = [("user_file", "", str(tmp_path / "story.epub")),
                ("assistant", "a", "", "", str(request_folder), "Request 1", {})]
    assert turn_workspace(messages, [1]) == str(folder)
    assert turn_workspace(messages, [0, None, 7]) == ""


def test_copy_text_for_prefers_the_desktop_copies(tmp_path):
    from glossarion_mobile.ui.chat.chat_ops import copy_text_for

    html_copy = tmp_path / "000002-content.html"
    html_copy.write_text("<p>saved <b>html</b></p>", encoding="utf-8")
    storage = {"content_html_path": "000002-content.html"}
    resolve = lambda ref: str(tmp_path / ref)  # noqa: E731
    assert copy_text_for("html", "**md**", storage, resolve) == "<p>saved <b>html</b></p>"
    assert copy_text_for("markdown", "**md** body", storage, resolve) == "**md** body"
    html = copy_text_for("html", "**bold** text", {}, resolve)
    assert "<strong>bold</strong>" in html
    assert copy_text_for("text", "**bold** text", {}, resolve) == "bold text"


def test_glossary_terms_markdown_uses_the_shared_footnote(tmp_path):
    from glossarion_mobile.ui.chat.chat_ops import glossary_terms_markdown, source_text_for

    glossary = tmp_path / "glossary.csv"
    glossary.write_text("type,raw_name,translated_name\ncharacter,김철수,Kim Cheolsu\nterm,마나,mana\n", encoding="utf-8")
    source = source_text_for(("user", "김철수가 웃었다."))
    markdown = glossary_terms_markdown(str(glossary), source, "Kim Cheolsu smiled.", label="Response 1")
    assert "Matched glossary entries" in markdown and "Kim Cheolsu" in markdown and "mana" not in markdown


@needs_flet
def test_job_card_output_chips_and_compile_label():
    from glossarion_mobile.ui.chat.cards import ATTACHMENT_ACTIONS, JobCard
    from glossarion_mobile.ui.chat.job_binding import CardPhase

    opened: list = []
    card = JobCard(attachment={"name": "story.epub", "extension": ".epub"}, phase=CardPhase("done"),
                   on_action=lambda a: None, on_open_output=lambda p, k: opened.append((p, k)))
    card.set_outputs([("/w/story.epub", "epub"), ("/w/story.srt", "subtitle")])
    assert card.outputs_row.visible and len(card.outputs_row.controls) == 2
    card.outputs_row.controls[1].on_click(None)
    assert opened == [("/w/story.srt", "subtitle")]
    assert dict((a, label) for a, label, _i, _m in ATTACHMENT_ACTIONS)["compile"].startswith("Compile")
    running = JobCard(attachment={"name": "x"}, phase=CardPhase("running"))
    running.set_outputs([("/w/a.epub", "epub")])
    assert not running.outputs_row.visible


def test_chat_runs_compile_uses_the_cards_folder(tmp_path):
    from glossarion_mobile.ui.chat.run_controller import ChatRuns

    folder = _workspace(tmp_path)
    submitted: list = []

    async def submit(kind, title, inputs, params, origin):
        submitted.append((kind, params["folder"], origin["cid"]))
        return "job-1"

    runs = types.SimpleNamespace(run_for=lambda cid: None, jobs=types.SimpleNamespace(submit=submit))
    assert asyncio.run(ChatRuns.compile(runs, "c1", "compile_pdf", folder=str(folder))) == "job-1"
    assert submitted == [("compile_pdf", str(folder), "c1")]
    assert asyncio.run(ChatRuns.compile(runs, "c1")) is None  # no run in this session, no folder


def test_open_output_navigates_to_the_workspace_folder(tmp_path):
    from glossarion_mobile.ui.chat.integration import ChatFeature

    folder = _workspace(tmp_path)
    output = tmp_path / "out"
    navigated: list = []
    prefs = types.SimpleNamespace(file_ref=lambda p: "f" + str(abs(hash(os.path.normcase(p))))[:11])
    jobs = types.SimpleNamespace(file_roots=lambda: {"output": str(output), "chats": str(output / "Direct Text")})
    feature = ChatFeature.__new__(ChatFeature)
    feature.app = types.SimpleNamespace(navigate_to=lambda *a: navigated.append(a), prefs=prefs, jobs=jobs)
    assert feature.open_output(str(folder)) == "tools.files.folder"
    assert navigated[-1] == ("tools.files.folder", {"root": "chats", "fid": prefs.file_ref(str(folder))})
    assert feature.open_output(str(tmp_path / "elsewhere")) == "tools.files"


def test_job_workspace_of_a_chat_job(tmp_path):
    from glossarion_mobile.ui.chat.integration import ChatFeature

    folder = _workspace(tmp_path)
    request_folder = folder / "story_output"
    request_folder.mkdir()
    messages = [("user", "hi"), ("assistant", "hello", "", "", "", "", {}),
                ("user_file", "", str(tmp_path / "story.epub")),
                ("assistant", "a", "", "", str(request_folder), "Request 1", {})]
    chats = types.SimpleNamespace(messages=lambda cid: messages, attachment_folders=lambda cid: [str(folder)])
    feature = ChatFeature.__new__(ChatFeature)
    feature.env = types.SimpleNamespace(chats=chats)
    snap = types.SimpleNamespace(spec=types.SimpleNamespace(origin={"type": "chat", "cid": "c1"},
                                                            params={"user_index": 2}))
    assert feature.job_workspace(snap) == str(folder)
    messages[3] = ("assistant", "a", "", "", "", "Request 1", {})  # no folder recorded: the attachment's stem
    assert feature.job_workspace(snap) == str(folder)


@needs_flet
def test_job_detail_files_open_the_chat_workspace(tmp_path):
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    folder = _workspace(tmp_path)
    output = tmp_path / "out"
    run_root = tmp_path / "data" / "direct_text_runs" / "r1" / "story"
    run_root.mkdir(parents=True)
    screen = JobDetailScreen.__new__(JobDetailScreen)
    screen.roots = lambda: {"output": str(output), "chats": str(output / "Direct Text")}
    screen.chat_workspace = lambda snap: str(folder)
    screen._files_key = screen._files_target = None
    snap = types.SimpleNamespace(id="j1", output_dir=str(run_root), output_dirs={}, state="done",
                                 spec=types.SimpleNamespace(origin={"type": "chat", "cid": "c1"}))
    assert screen.files_target(snap) == ("chats", str(folder))
    screen.chat_workspace = lambda snap: ""
    screen._files_key = None
    assert screen.files_target(snap) is None  # the run root is outside every root: no Files button
    library = types.SimpleNamespace(id="j2", output_dir=str(output / "Book"), output_dirs={}, state="done",
                                    spec=types.SimpleNamespace(origin={"type": "library"}))
    assert screen.files_target(library) == ("output", str(output / "Book"))


# ==========================================================================
# Library / Book page
# ==========================================================================


def _library_ctx(service, answers=None):
    from glossarion_mobile.ui.library.common import LibraryContext

    said: list = []
    ctx = LibraryContext(service=service, notify=lambda m, *a: said.append(m))
    if answers is not None:
        ctx.extras["answers"] = list(answers)
    ctx.said = said
    return ctx


class _MetaService:
    def __init__(self) -> None:
        self.submitted: list = []

    async def io(self, fn, *args):
        return fn(*args)

    def has_job_kind(self, kind):
        return kind == "metadata"

    def metadata_spec(self, books):
        return ("metadata", tuple(b["name"] for b in books))

    async def submit(self, spec):
        self.submitted.append(spec)
        return "job-m"


def test_library_metadata_asks_before_regenerating(tmp_path):
    out = tmp_path / "Book"
    out.mkdir()
    (out / "metadata.json").write_text("{}", encoding="utf-8")
    service = _MetaService()
    book = {"name": "Book", "output_folder": str(out)}
    ctx = _library_ctx(service, answers=["cancel"])
    assert asyncio.run(ctx.translate_metadata([book])) is None and not service.submitted
    assert ctx.extras["asked"][0][0] == "Metadata Already Exists"
    ctx = _library_ctx(service, answers=["yes"])
    assert asyncio.run(ctx.translate_metadata([book])) == "job-m"
    fresh = {"name": "Fresh", "output_folder": str(tmp_path / "Fresh")}
    ctx = _library_ctx(service, answers=[])
    assert asyncio.run(ctx.translate_metadata([fresh])) == "job-m" and "asked" not in ctx.extras


@needs_flet
def test_book_page_metadata_item_needs_a_raw_epub():
    from glossarion_mobile.ui.library.book_page import BookPageScreen

    fake = types.SimpleNamespace(book={"raw_source_path": "/x/book.pdf"}, service=_MetaService())
    assert BookPageScreen.metadata_reason(fake) == "Needs a raw EPUB"
    fake.book = {"raw_source_path": "/x/book.epub"}
    assert BookPageScreen.metadata_reason(fake) is None


def test_overview_two_columns_only_on_large_phone_landscape():
    from glossarion_mobile.ui.library.overview_tab import two_columns

    assert two_columns("large_phone", 880, 400)
    assert not two_columns("large_phone", 700, 900)
    assert not two_columns("compact", 800, 400)
    assert not two_columns("tablet", 1000, 700)


def test_glossary_file_summary_counts_types(tmp_path):
    from glossarion_mobile.ui.library.glossary_tab import glossary_file_summary, summary_line

    path = tmp_path / "Book_glossary.csv"
    path.write_text("type,raw_name,translated_name\ncharacter,A,a\ncharacter,B,b\nterm,C,c\n", encoding="utf-8")
    summary = glossary_file_summary(str(path))
    assert summary["entries"] == 3 and summary["types"][0] == ("character", 2)
    line = summary_line(summary)
    assert line.startswith("3 entries · character 2 · term 1") and "modified " in line
    assert glossary_file_summary(str(tmp_path / "missing.csv"))["entries"] is None


def test_output_tab_workspace_groups(tmp_path):
    from glossarion_mobile.ui.library.output_tab import workspace_groups

    folder = tmp_path / "Book"
    (folder / "text_to_speech").mkdir(parents=True)
    (folder / "images").mkdir()
    (folder / "Book_glossary.csv").write_text("x", encoding="utf-8")
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    (folder / "TOC.txt").write_text("x", encoding="utf-8")
    (folder / "ch1.sdlxliff").write_text("x", encoding="utf-8")
    (folder / "text_to_speech" / "ch1.mp3").write_bytes(b"ID3")
    (folder / "images" / "a.png").write_bytes(b"\x89PNG")
    report = folder / "Book_Scan Report"
    report.mkdir()
    (report / "validation_results.html").write_text("<html/>", encoding="utf-8")
    groups = workspace_groups(str(folder))
    assert [os.path.basename(p) for p in groups["glossary"]] == ["Book_glossary.csv"]
    assert [os.path.basename(p) for p in groups["metadata"]] == ["metadata.json", "TOC.txt"]
    assert groups["sdlxliff"] and groups["tts"] and groups["images"] == [str(folder / "images")]
    assert groups["qa"] and groups["qa"][0].endswith("validation_results.html")


def test_files_open_with_offers_the_qa_report_viewer():
    from glossarion_mobile.ui.screens.files import is_qa_report, root_for

    assert is_qa_report("/x/Book_Scan Report/validation_results.html")
    assert not is_qa_report("/x/report.html")
    roots = {"output": "/tmp/o", "chats": "/tmp/o/Direct Text"}
    assert root_for("/nowhere/at/all", roots) is None


# ==========================================================================
# Reader
# ==========================================================================


def test_reader_custom_fonts_and_font_face(tmp_path):
    from glossarion_mobile.ui.reader import model as rm

    (tmp_path / "Nanum Myeongjo.ttf").write_bytes(b"\x00\x01\x00\x00rest")
    (tmp_path / "Nanum Myeongjo.otf").write_bytes(b"OTTOrest")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    families = rm.custom_font_families(str(tmp_path))
    assert [f for f, _p in families] == ["Nanum Myeongjo"]
    settings = rm.ReaderSettings(font_family="Nanum Myeongjo")
    css = rm.override_css({"bg": "#000000", "fg": "#ffffff"}, settings, font_faces={"Nanum Myeongjo": "/t/font/abc"})
    assert css.startswith("@font-face { font-family: 'Nanum Myeongjo'; src: url('/t/font/abc')")
    assert "'Nanum Myeongjo'" in css.split("\n", 2)[2]
    assert "@font-face" not in rm.override_css({}, rm.ReaderSettings(font_family="Serif"), font_faces={"x": "/u"})


def test_reader_server_serves_fonts_only(tmp_path):
    import urllib.request

    from glossarion_mobile.services.reader_server import ReaderServer

    font = tmp_path / "f.ttf"
    font.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 60)
    fake = tmp_path / "f2.ttf"
    fake.write_bytes(b"not a font")
    server = ReaderServer()
    server.start()
    try:
        url = server.origin + server.register_font(str(font))
        with urllib.request.urlopen(url, timeout=5) as response:
            assert response.headers.get("Content-Type") == "font/ttf" and response.read()[:4] == b"\x00\x01\x00\x00"
        bad = server.origin + server.register_font(str(fake))
        with pytest.raises(Exception):
            urllib.request.urlopen(bad, timeout=5)
    finally:
        server.stop()


@needs_flet
def test_aa_sheet_lists_imported_fonts():
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader.aa_sheet import AaSheet

    sheet = AaSheet(rm.ReaderSettings(), [{"name": "Dark", "bg": "#000", "fg": "#fff"}],
                    extra_families=["Nanum Myeongjo"])
    sheet._text_tab()
    assert [o.key for o in sheet.family.options][-1] == "Nanum Myeongjo"


def test_text_size_hint_at_large_text(monkeypatch):
    from glossarion_mobile.ui import text_scale
    from glossarion_mobile.ui.reader.reader_view import TEXT_SIZE_HINT, ReaderScreen

    said: list = []
    state = types.SimpleNamespace(text_scale=types.SimpleNamespace(value=1.0))
    fake = types.SimpleNamespace(scale_hint_shown=False, open_aa=lambda: None,
                                 deps=types.SimpleNamespace(extras={"state": state},
                                                            notify=lambda *a: said.append(a[0])))
    assert ReaderScreen.text_size_hint(fake) is False
    monkeypatch.setitem(text_scale._state, "os", 2.0)
    assert ReaderScreen.text_size_hint(fake) is True and said == [TEXT_SIZE_HINT]
    assert ReaderScreen.text_size_hint(fake) is False  # once


# ==========================================================================
# manga
# ==========================================================================


def test_manga_font_catalog_and_mobile_default_font(tmp_path):
    import manga_models
    import manga_settings_defaults as msd

    system = tmp_path / "system_fonts"
    (system / "sub").mkdir(parents=True)
    (system / "Roboto-Regular.ttf").write_bytes(b"x")
    (system / "sub" / "NotoSansCJK-Regular.ttc").write_bytes(b"x")
    (system / "readme.txt").write_text("x", encoding="utf-8")
    custom = tmp_path / "Mine.otf"
    custom.write_bytes(b"x")
    fonts = msd.available_fonts({"custom_fonts": [{"name": "Mine", "path": str(custom)}]}, font_dirs=[str(system)])
    names = [os.path.basename(p) for p in fonts]
    assert names[:2] == ["NotoSansCJK-Regular.ttc", "Roboto-Regular.ttf"] and names[-1] == "Mine.otf"
    latin, cjk = str(system / "Roboto-Regular.ttf"), str(system / "sub" / "NotoSansCJK-Regular.ttc")
    assert manga_models.mobile_default_font("English", candidates=[latin, cjk]) == latin
    assert manga_models.mobile_default_font(None, candidates=[str(tmp_path / "none.ttf")]) is None


def _host_ttf():
    candidates = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/System/Library/Fonts/Supplemental/Arial.ttf"]
    try:
        import matplotlib

        candidates.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                                       "DejaVuSans.ttf"))
    except Exception:
        pass
    return next((p for p in candidates if os.path.isfile(p)), None)


def test_mobile_run_config_resolves_to_a_truetype_font(monkeypatch):
    import manga_models

    font = _host_ttf()
    if font is None:
        pytest.skip("no TrueType font on this host")
    pil = pytest.importorskip("PIL.ImageFont")
    monkeypatch.setattr(manga_models, "MOBILE_DEFAULT_FONTS", (font,))
    config = manga_models.apply_mobile_run_defaults({"output_language": "English"}, force=True)
    assert config["manga_font_path"] == font
    rendered = pil.truetype(config["manga_font_path"], 32)
    assert type(rendered).__name__ == "FreeTypeFont" and rendered.size == 32
    kept = manga_models.apply_mobile_run_defaults({"manga_font_path": "/mine.ttf"}, force=True)
    assert kept["manga_font_path"] == "/mine.ttf"
    desktop = manga_models.apply_mobile_run_defaults({"output_language": "English"})
    assert "manga_font_path" not in desktop  # desktop config untouched


@needs_flet
def test_manga_editor_thumb_long_press_toggles_skip():
    from glossarion_mobile.ui.tools.manga.editor import EditorTab

    skipped: set = set()

    def toggle(path):
        skipped.symmetric_difference_update({path})
        return path in skipped

    files = types.SimpleNamespace(is_skipped=lambda p: p in skipped, toggle_skip=toggle, files=["/p/1.png"])
    mutated: list = []

    async def mutate(fn, *args):
        mutated.append(args)
        return fn(*args)

    refreshed: list = []
    fake = types.SimpleNamespace(session=types.SimpleNamespace(files=files),
                                 screen=types.SimpleNamespace(files_tab=types.SimpleNamespace(_mutate=mutate)),
                                 refresh=lambda: refreshed.append(True),
                                 ctx=types.SimpleNamespace(extras={}, page=None, tablet=False))
    fake.is_skipped = lambda p: EditorTab.is_skipped(fake, p)
    sheet = EditorTab.thumb_menu(fake, "/p/1.png")
    assert sheet.items[0].label == "⏭️ Skip Processing"
    assert asyncio.run(EditorTab.toggle_skip(fake, "/p/1.png")) is True and mutated == [("/p/1.png",)] and refreshed
    assert EditorTab.thumb_menu(fake, "/p/1.png").items[0].label == "▶️ Process This Image"


# ==========================================================================
# Tools ErrorCard, user guide
# ==========================================================================


@needs_flet
def test_failed_tools_job_shows_an_error_card():
    from glossarion_mobile.ui.components.error_card import ErrorCard
    from glossarion_mobile.ui.tools.common import failed_job_card, job_failed

    went: list = []
    ctx = types.SimpleNamespace(go=lambda *a: went.append(a), copy_text=lambda t: None)
    snap = types.SimpleNamespace(id="j9", title="Compile EPUB · Book", error="RuntimeError: boom",
                                 state=types.SimpleNamespace(value="failed"), state_label="Failed")
    assert job_failed(snap) and not job_failed(types.SimpleNamespace(state=types.SimpleNamespace(value="done")))
    card = failed_job_card(ctx, snap, on_retry=lambda: None)
    assert isinstance(card, ErrorCard) and card.text == "RuntimeError: boom"
    card.on_view_log()
    assert went == [("jobs.detail", {"jid": "j9"})]


def test_user_guide_is_bundled_and_help_opens_it():
    from glossarion_mobile.ui.screens.about import USER_GUIDE_ASSET, load_user_guide

    text = load_user_guide(types.SimpleNamespace(assets_dir=str(APP_DIR / "assets")))
    assert text.startswith("# Glossarion Mobile") and "Use “…”" in text
    assert (APP_DIR / "assets" / USER_GUIDE_ASSET).is_file()
    app_source = (APP_DIR / "glossarion_mobile" / "app.py").read_text(encoding="utf-8")
    assert "Arrives with the bundled user guide" not in app_source
