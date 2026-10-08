"""Host tests for the U9 feature-gap closures (audit round 4).

* chat: a chat-origin job without a Job card (Extract glossary, Compile) keeps the JobStrip in its chat;
  a TXT / MD attachment over 20,000 characters gets a Plan card (its characters, not the message's);
  ``/retranslate N-M`` hands the range to the Chapters tab, which selects it;
* the Library review gate: ``job_kinds.translate.glossary_review_gate`` (the shared approval points, ■ No
  stops the job), the TranslateSheet switch, the Book page / Jobs approval sheet and its notification route;
* excluded routes: the job preflight (``model_catalog.job_model_block``, ``JobsFeature.submit``), the
  TranslateSheet / Extract sheet reasons, Keys cards, catalog statuses and ``model_options`` without
  ``ocagy_cli``;
* Keys: per-key endpoint and Google credential checks, Test all (enabled keys only), context presets;
  the ModelSheet GCP project picker;
* settings: Glossary mode locks on Settings › Glossary, the Unified / Endpoints section pages, extraction
  visibility rules, Image & vision key pools, EPUB contents, the Context mode combo, Direct Text read-only,
  Reader & Library types, the FFT / manga worker notes, the Vision OCR prepass choices;
* Library / Reader / Tools: the windowed Glossary Progress list, image-folder Refresh, 📂 Open file,
  Organize selected, the empty chapter, the Review ErrorCard, the manga worker cap;
* no ReasonChip under a disabled control (UI_SPEC §5.2).

Real data is never touched: HOME / USERPROFILE / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY /
GLOSSARION_DATA_DIR / GLOSSARION_MODEL_CATALOG_CACHE point at pytest's tmp dir (autouse fixture).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u9_gap_round4.py
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
    """No test reads or writes the user's Library, output folders, home, data folder or catalog cache."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data"), ("LOCALAPPDATA", "local")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_MODEL_CATALOG_CACHE", str(tmp_path / "_iso" / "catalog.json"))
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


def _walk(control, ancestors=()):
    """Every Flet control under ``control`` with its ancestors (content / controls / title / ... slots)."""
    import flet as ft

    yield control, ancestors
    for name in ("content", "controls", "title", "subtitle", "leading", "trailing", "actions", "tabs"):
        child = getattr(control, name, None)
        for kid in (child if isinstance(child, (list, tuple)) else [child]):
            if isinstance(kid, ft.Control):
                yield from _walk(kid, ancestors + (control,))


def _dead_chips(root) -> list:
    """ReasonChips under a disabled control (Flet disables every child: the chip could not open its reason)."""
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    return [chip.reason for chip, parents in _walk(root)
            if isinstance(chip, ReasonChip) and any(getattr(p, "disabled", False) for p in parents)]


# ==========================================================================
# chat: JobStrip, Plan card for long TXT, /retranslate N-M
# ==========================================================================


def _snap(params=None, origin=None, kind="extract_glossary"):
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    spec = JobSpec(kind, "Book.epub", ("a",), params=dict(params or {}),
                   origin=dict(origin or {"type": "chat", "cid": "7", "label": "Chat"}))
    return JobSnapshot(id="abc123abc123", spec=spec, state=JobState.RUNNING, created=0.0, started=1.0)


def test_strip_model_tells_a_chat_run_from_a_cardless_chat_job():
    from glossarion_mobile.services.jobs import strip_model_for

    extract = strip_model_for(_snap())
    assert extract.owner_chat == "7" and extract.chat_card is False
    run = strip_model_for(_snap({"chat_id": 7}, kind="direct_text"))
    assert run.owner_chat == "7" and run.chat_card is True
    other = strip_model_for(_snap({"chat_id": 3}, kind="direct_text"))
    assert other.chat_card is False


@needs_flet
def test_chat_keeps_the_strip_for_its_own_cardless_jobs():
    from glossarion_mobile.services.jobs import strip_model_for
    from glossarion_mobile.ui.chat.chat_view import ChatView

    shown: list = []
    fake = types.SimpleNamespace(state=types.SimpleNamespace(current_chat=types.SimpleNamespace(value="7")),
                                 job_strip=types.SimpleNamespace(set_model=shown.append))
    ChatView._apply_job_strip(fake, strip_model_for(_snap()))  # Extract glossary started from chat 7
    assert shown[-1] is not None and shown[-1].title.startswith("Extracting glossary")
    ChatView._apply_job_strip(fake, strip_model_for(_snap({"chat_id": 7}, kind="direct_text")))
    assert shown[-1] is None  # the chat's own run: its Job card is the progress UI


def test_attachment_characters_decide_the_plan(tmp_path):
    from glossarion_mobile.ui.chat import direct_text_rules as rules
    from glossarion_mobile.ui.chat.run_request import attachment_record

    big = tmp_path / "novel.txt"
    big.write_text("가" * 25_000, encoding="utf-8")  # 75,000 bytes: read, 25,000 characters
    short = tmp_path / "short.md"
    short.write_text("가" * 7_000, encoding="utf-8")  # 21,000 bytes but 7,000 characters
    huge = tmp_path / "huge.txt"
    huge.write_text("a" * 90_000, encoding="utf-8")  # over 4x the threshold: settled by the size
    assert rules.attachment_text_chars(attachment_record(big)) == 25_000
    assert rules.attachment_text_chars(attachment_record(short)) == 7_000
    assert rules.attachment_text_chars(attachment_record(huge)) > rules.PLAN_TEXT_THRESHOLD
    epub = tmp_path / "b.epub"
    epub.write_bytes(b"PK")
    assert rules.attachment_text_chars(attachment_record(epub)) == 0


@needs_flet
def test_send_puts_a_long_txt_attachment_behind_a_plan_card(tmp_path):
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings
    from glossarion_mobile.ui.chat.run_request import attachment_record

    meta: dict = {}
    submitted: list = []

    def fake_view():
        return types.SimpleNamespace(
            env=types.SimpleNamespace(chats=types.SimpleNamespace(
                record_user_turn=lambda cid, turn, title: 4, set_meta=lambda cid, k, v: meta.__setitem__(k, v))),
            cid="1", version_anchor=None, sent=[],
            state=types.SimpleNamespace(output_mode=types.SimpleNamespace(value=types.SimpleNamespace(mode="text"))),
            _link_version=lambda *a: None, _after_send_ui=lambda mode: None,
            _submit=lambda *a, **k: submitted.append(a), _spawn=lambda value: None)

    big = tmp_path / "novel.txt"
    big.write_text("가" * 25_000, encoding="utf-8")
    ChatView._send(fake_view(), "", attachment_record(big), DirectTextSettings(), None)
    plan = meta.get("pending_plan")
    assert plan and plan["user_index"] == 4 and plan["attachment"]["name"] == "novel.txt" and not submitted
    meta.clear()
    small = tmp_path / "short.txt"
    small.write_text("가" * 7_000, encoding="utf-8")
    ChatView._send(fake_view(), "", attachment_record(small), DirectTextSettings(), None)
    assert "pending_plan" not in meta and submitted  # straight to the run


def test_retranslate_range_parses_with_the_desktop_rule():
    from glossarion_mobile.ui.chat.slash import parse_chapter_range

    assert parse_chapter_range("5-10") == (5, 10) and parse_chapter_range(" 7 ") == (7, 7)
    assert parse_chapter_range("five") is None and parse_chapter_range("5-") is None


def test_chapters_tab_selects_a_requested_range():
    from glossarion_mobile.ui.library.chapters_tab import ChaptersTab, request_range_selection, take_range_request

    request_range_selection("b" * 12, 2, 3)
    assert take_range_request("b" * 12) == (2, 3) and take_range_request("b" * 12) is None

    def row(key, num, kind="chapter"):
        return types.SimpleNamespace(key=key, kind=kind, info={"num": num}, children=())

    said: list = []
    fake = types.SimpleNamespace(visible=[row("a", 1), row("b", 2), row("c", 3.0), row("d", 4), row("x", 2, "chunk")],
                                 selecting=False, selected=set(), window_start=0,
                                 _render_list=lambda rows, window_start=0: None, _sync_selection=lambda: None,
                                 ctx=types.SimpleNamespace(say=said.append))
    assert ChaptersTab.select_range(fake, 2, 3) == 2
    assert fake.selecting and fake.selected == {"b", "c"} and said[-1].startswith("Chapters 2–3: 2 selected")
    assert ChaptersTab.select_range(fake, 40, 50) == 0 and said[-1] == "No chapter 40–50 in this book's progress"


# ==========================================================================
# the Library glossary review gate
# ==========================================================================


class _GateCtx:
    def __init__(self, answer, params=None):
        self.params = dict(params if params is not None else {"review_glossary": True})
        self.answer = answer
        self.logs: list = []
        self.asked: list = []
        self.stops: list = []
        self.owner = types.SimpleNamespace(manual_glossary_path="")
        self.owner._auto_load_glossary_after_extraction = None  # set below (a bound-method stand-in)

    def log(self, text):
        self.logs.append(text)

    def ask(self, kind, **data):
        self.asked.append((kind, data))
        return self.answer

    def stop_requested(self):
        return bool(self.stops)

    def request_stop(self, reason=""):
        self.stops.append(reason)


def test_review_gate_asks_through_the_shared_approval_points(tmp_path, monkeypatch):
    from glossarion_mobile.job_kinds import translate as kind

    backend = types.SimpleNamespace(_direct_text_glossary_approval_callback=None)
    backend.set_direct_text_glossary_approval_callback = (
        lambda cb: setattr(backend, "_direct_text_glossary_approval_callback", cb))
    monkeypatch.setitem(sys.modules, "TransateKRtoEN", backend)
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")

    class Owner:
        manual_glossary_path = ""

        def _auto_load_glossary_after_extraction(self):
            return str(glossary)

    ctx = _GateCtx(answer=True)
    ctx.owner = Owner()
    assert kind.SUPPORTS_GLOSSARY_REVIEW is True
    with kind.glossary_review_gate(ctx) as approve:
        assert callable(approve) and backend._direct_text_glossary_approval_callback is approve
        assert ctx.owner._auto_load_glossary_after_extraction() == str(glossary)  # the pipeline's Balanced pre-pass
    assert ctx.asked == [("glossary_approval", {"path": str(glossary), "default": False})] and not ctx.stops
    assert "_auto_load_glossary_after_extraction" not in vars(ctx.owner)  # the shared method again
    assert backend._direct_text_glossary_approval_callback is None
    declined = _GateCtx(answer=False)
    declined.owner = Owner()
    with kind.glossary_review_gate(declined):
        assert backend._direct_text_glossary_approval_callback(str(glossary)) is False  # the backend's own phase
    assert declined.stops == [kind.GLOSSARY_REVIEW_DECLINED]
    off = _GateCtx(answer=True, params={})
    off.owner = Owner()
    with kind.glossary_review_gate(off) as approve:
        assert approve is None and backend._direct_text_glossary_approval_callback is None


def test_glossary_question_routes_and_kinds():
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState
    from glossarion_mobile.services.notifications import question_route
    from glossarion_mobile.ui.chat.cards import glossary_question

    spec = JobSpec("translate", "Book", ("a",), params={"review_glossary": True},
                   origin={"type": "library", "bid": "abcdefabcdef", "label": "Library · Book"})
    question = {"id": "q1", "kind": "glossary_approval", "data": {"path": "/x.csv"}}
    snap = JobSnapshot(id="j" * 12, spec=spec, state=JobState.RUNNING, created=0.0, question=question)
    assert glossary_question(snap) == question and question_route(snap) == "/library/book/abcdefabcdef"
    other = JobSnapshot(id="j" * 12, spec=spec, state=JobState.RUNNING, created=0.0,
                        question={"id": "q2", "kind": "async_batch_question", "data": {}})
    assert glossary_question(other) is None
    multi = JobSpec("translate", "2 books", ("a", "b"), origin={"type": "library", "label": "Library"})
    assert question_route(JobSnapshot(id="k" * 12, spec=multi, state=JobState.RUNNING, created=0.0)) == "/job/" + "k" * 12


@needs_flet
def test_review_sheet_answers_once_and_closes(tmp_path):
    from glossarion_mobile.ui.chat.cards import GlossaryReviewSheet, glossary_preview

    path = tmp_path / "g.csv"
    path.write_text("type,raw_name,translated_name\ncharacter,김철수,Kim\n", encoding="utf-8")
    answers: list = []
    sheet = GlossaryReviewSheet(path=str(path), info=glossary_preview(str(path)), on_answer=answers.append,
                                title="Book · glossary ready")
    assert sheet.card.info["entries"] == 1 and not sheet.card.edit_button.disabled
    sheet.card.answer(False)
    sheet.card.answer(True)  # a second tap is ignored
    assert answers == [False] and sheet.answered is False


@needs_flet
def test_translate_sheet_review_switch_and_excluded_model(tmp_path):
    from glossarion_mobile.ui.library.translate_sheet import TranslateSheet, review_gate_supported

    assert review_gate_supported()
    raw = tmp_path / "Book.epub"
    raw.write_bytes(b"PK")
    config = {"model": "ocz/big-pickle"}
    service = types.SimpleNamespace(cfg=lambda k, d=None: config.get(k, d), jobs=object(),
                                    has_job_kind=lambda kind: True)
    ctx = types.SimpleNamespace(service=service, intents=None, page=None)
    sheet = TranslateSheet(ctx, [{"name": "Book"}], [str(raw)])
    assert not sheet.review_switch.disabled
    assert sheet.start_reason == "ocz/ isn't available on mobile" and sheet.start_button.disabled
    assert any(getattr(c, "key", "") == "fact-model-excluded" for c in sheet.fact_chips)
    config["model"] = "gpt-6"
    ok = TranslateSheet(ctx, [{"name": "Book"}], [str(raw)])
    assert ok.start_reason is None and not ok.start_button.disabled


# ==========================================================================
# excluded routes: preflight, catalog statuses, model_options without ocagy_cli
# ==========================================================================


def test_job_model_block_covers_model_jobs_only():
    from glossarion_mobile.services import model_catalog as mc

    get = {"model": "ollamapull/qwen3:8b"}.get
    assert mc.job_model_block("translate", {}, get)[0] == "ollamapull/ isn't available on mobile"
    assert mc.job_model_block("extract_glossary", {}, get) is not None
    assert mc.job_model_block("compile_epub", {}, get) is None and mc.job_model_block("qa_scan", {}, get) is None
    # the job's own model wins over the config's
    assert mc.job_model_block("translate", {"config_overrides": {"model": "gpt-6"}}, get) is None
    assert mc.job_model_block("direct_text", {"model": "antigravity/x"}, {"model": "gpt-6"}.get) is not None
    assert mc.mobile_statuses({"ocagy": "static fallback (ModuleNotFoundError)", "openai": "online (2 models)"}) == {
        "ocagy": mc.EXCLUDED_STATUS, "openai": "online (2 models)"}
    assert "npm/bun" in mc.provider_excluded_detail("opencode-zen")


def test_jobs_feature_refuses_an_excluded_model_before_queueing():
    from glossarion_mobile.services.jobs import JobSpec
    from glossarion_mobile.ui.screens.jobs import JobsFeature

    notes: list = []
    feature = JobsFeature.__new__(JobsFeature)
    feature.app = types.SimpleNamespace(config_store={"model": "ocz/big-pickle"},
                                        notify=lambda *a: notes.append(a), chat_view=None)
    spec = JobSpec("translate", "Book", ("a.epub",))
    assert asyncio.run(feature.submit(spec)) is None
    assert notes[-1][0] == "ocz/ isn't available on mobile" and notes[-1][1] == "Choose model"
    assert feature.model_block(JobSpec("compile_epub", "Book", ("a",))) is None


def test_keys_cards_mark_excluded_models():
    from glossarion_mobile.ui.screens.keys import excluded_model_reason

    assert excluded_model_reason("authza/glm-5")[0] == "authza/ isn't available on mobile"
    assert excluded_model_reason("gpt-6") is None and excluded_model_reason("") is None


def test_model_options_reports_unbundled_ocagy_like_arena(monkeypatch):
    import model_options

    bundled = model_options._module_bundled
    monkeypatch.setattr(model_options, "_module_bundled", lambda name: False if name == "ocagy_cli" else bundled(name))
    result = model_options.refresh_provider_model_catalogs(only_provider="ocagy", timeout=0.1)
    assert result.statuses["ocagy"] == "unavailable in this build"
    result = model_options.refresh_provider_model_catalogs(only_provider="opencode-zen", timeout=0.1)
    assert result.statuses["opencode-zen"] == "unavailable in this build"
    assert model_options._module_bundled("json") and not model_options._module_bundled("no_such_module_xyz")


def test_catalog_refresh_returns_an_outcome_for_an_import_error():
    from glossarion_mobile.services import model_catalog as mc

    def refresh(**kwargs):
        raise ImportError("No module named 'autharena_proxy'")

    service = mc.ModelCatalogService({}, options=types.SimpleNamespace(refresh_provider_model_catalogs=refresh))
    service.credentials = lambda provider, representative=None: {
        "active_model": "", "active_api_key": "", "provider_keys": {}, "custom_routes": []}
    outcome = service._refresh_locked(None, explicit=True, timeout=0.1)
    assert outcome.ok is False and outcome.message.startswith("Catalog refresh failed: No module named")


# ==========================================================================
# Keys: endpoint / credential checks, Test all, presets, GCP picker
# ==========================================================================


def test_individual_endpoint_rules_are_the_desktop_dialog_rules():
    import key_pool_service as kps

    assert kps.individual_endpoint_error(False, "", "") is None
    assert kps.individual_endpoint_error(True, " ", "v") == "Endpoint Base URL is required when Enable is ON."
    assert kps.individual_endpoint_error(True, "192.168.1.5:11434/v1", "") == (
        "Endpoint URL must start with http:// or https://")
    assert kps.individual_endpoint_error(True, "https://r.openai.azure.com", "") == (
        "Azure API Version is required for Azure endpoints.")
    assert kps.individual_endpoint_error(True, "http://192.168.1.5:11434/v1", "") is None
    entry, error = kps.validate_entry({"model": "gpt-6", "use_individual_endpoint": True, "azure_endpoint": None})
    assert entry is None and error == "Endpoint Base URL is required when Enable is ON."
    assert [label for label, _url in kps.INDIVIDUAL_ENDPOINT_SHORTCUTS] == ["Ollama", "LM Studio", "TTS", "TTS v1"]


def test_google_credentials_check_is_the_desktop_picker_check(tmp_path):
    import settings_rules
    from glossarion_mobile.ui.screens.key_editor import google_credentials_problem

    good = tmp_path / "sa.json"
    good.write_text('{"type": "service_account", "project_id": "p1"}', encoding="utf-8")
    bad = tmp_path / "other.json"
    bad.write_text('{"installed": {}}', encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    assert google_credentials_problem(str(good)) is None
    assert google_credentials_problem(str(bad)) == settings_rules.INVALID_GOOGLE_CREDENTIALS
    assert google_credentials_problem(str(broken)).startswith("Failed to load credentials:")


@needs_flet
def test_settings_credentials_tile_refuses_a_non_service_account_file(tmp_path):
    from glossarion_mobile.ui.settings.tiles import PathTile, make_tile

    ctx = _ctx(tmp_path)
    tile = make_tile(ctx.schema.spec("google_cloud_credentials"), ctx)
    assert isinstance(tile, PathTile)
    bad = tmp_path / "x.json"
    bad.write_text("[]", encoding="utf-8")
    assert tile.save_path(str(bad)).startswith("Invalid Google Cloud credentials file")
    assert not ctx.store.has("google_cloud_credentials")
    good = tmp_path / "sa.json"
    good.write_text('{"type": "service_account", "project_id": "p"}', encoding="utf-8")
    assert tile.save_path(str(good)) is None and ctx.store.get("google_cloud_credentials") == str(good)
    ctx.store._saver.close()


@needs_flet
def test_key_editor_endpoint_checks_chips_and_context_presets(tmp_path):
    from glossarion_mobile.ui.screens.key_editor import KeyEditor

    ctx = _ctx(tmp_path)
    editor = KeyEditor(ctx, entry={"api_key": "k", "model": "gpt-6"}, pool_id="main",
                       contexts=("translation", "image_ocr", "inpainter", "tts"))
    editor.get_body() if hasattr(editor, "get_body") else editor.build_body()
    labels = [c.label.value for c in editor.endpoint_chips.controls]
    assert "Ollama on LAN" in labels and "TTS v1 on LAN" in labels and "Clear" not in labels
    editor.endpoint_switch.value = True
    editor.paste_endpoint("192.168.1.10:11434/v1")
    with pytest.raises(ValueError, match="must start with http"):
        editor.collect()
    editor.paste_endpoint("http://192.168.1.10:11434/v1")
    assert editor.collect()["azure_endpoint"] == "http://192.168.1.10:11434/v1"
    presets = [c.label.value for c in editor.context_presets.controls]
    assert presets == ["Enable all", "Disable all", "🖼️ Images only"]
    editor.apply_context_preset(set())
    assert not any(editor.context_enabled.values())
    import key_contexts

    images = dict(key_contexts.context_presets(editor.contexts))["🖼️ Images only"]
    editor.apply_context_preset(images)
    assert editor.context_enabled == {"translation": False, "image_ocr": True, "inpainter": True, "tts": False}
    ctx.store._saver.close()


@needs_flet
def test_context_sheet_presets_track_changes():
    from glossarion_mobile.ui.screens.keys import ContextSheet

    applied: list = []
    sheet = ContextSheet(states={"translation": True, "glossary": None, "image_ocr": False},
                         labels={}, on_apply=applied.append)
    assert [c.label.value for c in sheet.preset_row.controls] == ["Enable all", "Disable all", "🖼️ Images only"]
    sheet.apply_preset({"translation", "glossary", "image_ocr"})
    assert sheet.changes == {"glossary": True, "image_ocr": True}
    sheet.apply_preset(set())
    assert sheet.changes == {"translation": False, "glossary": False}
    no_images = ContextSheet(states={"translation": True}, labels={}, on_apply=applied.append)
    assert [c.label.value for c in no_images.preset_row.controls] == ["Enable all", "Disable all"]


@needs_flet
def test_test_all_only_tests_enabled_translation_keys():
    from glossarion_mobile.ui.screens.keys import KeysScreen

    said: list = []
    tested: list = []

    async def run_tests(indices):
        tested.append(list(indices))
        return []

    keys = {"main": [{"enabled": False}, {"enabled": True}, {}], "fallback": [{"enabled": False}, {}]}
    screen = KeysScreen.__new__(KeysScreen)
    screen.controller = types.SimpleNamespace(keys=lambda pool: keys[pool])
    screen.say = said.append
    screen.run_tests = run_tests
    screen.pool = "main"
    asyncio.run(screen.test_all())
    assert tested[-1] == [1, 2]
    screen.pool = "fallback"
    asyncio.run(screen.test_all())
    assert tested[-1] == [0, 1]  # the desktop fallback / glossary pools test every key
    screen.pool = "main"
    keys["main"] = [{"enabled": False}]
    assert asyncio.run(screen.test_all()) == [] and said[-1] == "No enabled keys to test"


class _FakeOAuth:
    def __init__(self):
        self.pushed: list = []

    def gemini_projects(self, account_id):
        return [("p-billed", "billed"), ("p-unbilled", "unbilled")]

    def gemini_project_choice(self, projects, current):
        return "p-billed"

    def set_gemini_project(self, project_id, account_id):
        self.pushed.append((project_id, account_id))


@needs_flet
def test_gcp_project_picker_is_shared_by_accounts_and_the_model_sheet():
    from glossarion_mobile.services.model_catalog import RouteInfo
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.accounts import AccountsScreen, GcpProjectPicker
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    config: dict = {}
    oauth = _FakeOAuth()

    async def io(fn, *args):
        return fn(*args)

    picker = GcpProjectPicker(oauth, config_get=config.get, config_set=config.update, io=io, slot=lambda: 2)
    picker.build()
    assert asyncio.run(picker.load_projects()) == [("p-billed", "billed"), ("p-unbilled", "unbilled")]
    assert config["authgem_project"] == "p-billed" and oauth.pushed[-1] == ("p-billed", 2)
    assert [o.text for o in picker.dropdown.options] == ["✅ p-billed", "⚠️ p-unbilled (no billing)"]
    assert picker.note.value == "Found 1 GCP project(s) with billing enabled"
    screen = AccountsScreen(parse_route("/settings/accounts"), oauth=oauth, config_get=lambda k, d=None: config.get(k, d),
                            config_set=config.update)
    assert isinstance(screen.project_picker, GcpProjectPicker) and screen.select_project("p-unbilled") == "p-unbilled"
    built: list = []

    def factory(model):
        built.append(model)
        return GcpProjectPicker(oauth, config_get=config.get, config_set=config.update, io=io, slot=lambda: 0,
                                key="route-gcp-project")

    sheet = ModelSheet(current_model="authgem-vertex/gemini-3-pro", models=["authgem-vertex/gemini-3-pro"],
                       env=SheetEnv(gcp_project_picker=factory), on_select=lambda *a: None)
    controls = sheet._route_controls(RouteInfo(model="authgem-vertex/gemini-3-pro", gcp_project=True,
                                               needs_key=False))
    assert built == ["authgem-vertex/gemini-3-pro"]
    assert any(getattr(c, "key", "") == "route-gcp-project-picker" for c in controls)
    plain = ModelSheet(current_model="authgem-vertex/gemini-3-pro", models=[], env=SheetEnv(), on_select=lambda *a: None)
    assert not any(getattr(c, "key", "") == "route-gcp-project-picker"
                   for c in plain._route_controls(RouteInfo(model="authgem-vertex/x", gcp_project=True, needs_key=False)))


# ==========================================================================
# settings
# ==========================================================================


@needs_flet
def test_glossary_settings_screen_runs_the_mode_lock_pass():
    from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsScreen

    passes: list = []
    observed: dict = {}
    def observe(key, callback):
        observed[key] = callback
        return lambda: observed.pop(key, None)

    store = types.SimpleNamespace(observe=observe)
    ctx = types.SimpleNamespace(service=types.SimpleNamespace(apply_mode_locks=lambda: passes.append(1) or {"x": 1}),
                                settings=types.SimpleNamespace(store=store), post_ui=lambda fn: fn())
    screen = GlossarySettingsScreen.__new__(GlossarySettingsScreen)
    screen.ctx = ctx
    screen.tab = types.SimpleNamespace(did_show=lambda: None, dispose=lambda: None)
    screen._lock_unsubs = None
    screen.did_show()
    screen.did_show()  # once per screen
    assert passes == [1] and "auto_glossary_mode" in observed
    observed["auto_glossary_mode"]("auto_glossary_mode", "no_glossary")
    assert passes == [1, 1]
    screen.dispose()
    assert screen._lock_unsubs is None and "auto_glossary_mode" not in observed


def test_glossary_mode_locks_fix_the_toggles_for_no_glossary():
    import settings_rules

    for mode in ("balanced", "off_fuzzy_automap", "off", "minimal"):
        config = {"auto_glossary_mode": mode, "append_glossary": True, "append_glossary_auto_load": True}
        settings_rules.apply_change(config, "auto_glossary_mode", "no_glossary")
        settings_rules.apply_glossary_mode_locks(config)
        locks = settings_rules.glossary_mode_locks("no_glossary")
        for key in ("append_glossary", "append_glossary_auto_load"):
            if locks[key].locked:
                assert config[key] == locks[key].value, (mode, key)


@needs_flet
def test_section_screens_registry_and_the_unified_and_endpoints_pages(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.settings_home import ROUTE_SECTIONS, section_screen_for

    built: list = []
    extras = {"section_screens": {"other.endpoints": lambda m: built.append(("endpoints", m.fragment)) or "E"},
              "section_screen": lambda m: built.append(("legacy", m.params["section"])) or None}
    ctx = types.SimpleNamespace(extras=extras)
    assert section_screen_for(ctx, parse_route("/settings/s/other.endpoints#openai_base_url")) == "E"
    assert built[-1] == ("endpoints", "openai_base_url")
    assert section_screen_for(ctx, parse_route("/settings/s/pdf")) is None and built[-1] == ("legacy", "pdf")
    assert ROUTE_SECTIONS["endpoints"] == "other.endpoints"
    # GlossaryFeature: Settings › Glossary › Unified Glossary is the Unified glossary page
    from glossarion_mobile.ui.glossary.feature import GlossaryFeature
    from glossarion_mobile.ui.glossary.unified import UnifiedGlossaryScreen

    feature = GlossaryFeature.__new__(GlossaryFeature)
    feature.app = types.SimpleNamespace(settings=object())
    feature.context = lambda: types.SimpleNamespace(service=object(), jobs=None, settings=None)
    assert isinstance(feature.section_screen(parse_route("/settings/s/glossary.unified")), UnifiedGlossaryScreen)
    assert feature.section_screen(parse_route("/settings/s/pdf")) is None
    # the Endpoints page focuses a key
    from glossarion_mobile.ui.screens.endpoints import EndpointsScreen

    sctx = _ctx(tmp_path)
    screen = EndpointsScreen(parse_route("/settings/s/other.endpoints#groq_base_url"), sctx)
    screen.build_body()
    assert screen.focus_target == "groq_base_url" and "groq_base_url" in screen.tiles
    sctx.store._saver.close()


def test_extraction_options_follow_the_text_extraction_method():
    import settings_schema as ss
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    standard = {"text_extraction_method": "standard", "extraction_mode": "smart"}
    enhanced = {"text_extraction_method": "enhanced"}
    assert ss.spec("fix_stray_p_gt_bs").visible_if == "extraction:standard"
    for key in ("enhanced_preserve_structure", "skip_markdown_to_html", "allow_ai_markdown_headers",
                "enhanced_single_line_break", "convert_br_to_paragraphs", "html2text_escape_snob",
                "preserve_asterisk_separator_lines", "use_markdown2_converter"):
        spec = ss.spec(key)
        assert schema.hidden_reason(spec, standard) and not schema.hidden_reason(spec, enhanced), key
    stray = ss.spec("fix_stray_p_gt_bs")
    assert not schema.hidden_reason(stray, standard) and schema.hidden_reason(stray, {"extraction_mode": "enhanced"})


def test_section_layout_pools_epub_contents_and_direct_text_link():
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess
    from glossarion_mobile.ui.settings.section_page import SECTION_POOL_KEYS, STATIC_LINKS

    assert SECTION_POOL_KEYS["other.image"] == ("qa_scan_keys", "inpainter_keys")
    schema = SchemaAccess()
    epub = dict(schema.section("epub_output").headings)
    for key in ("disable_epub_gallery", "disable_automatic_cover_creation", "skip_non_spine_special_files",
                "skip_unreferenced_epub_images"):
        assert epub[key] == "Contents"
        assert schema.resolve_section_id("other.processing.extraction", key) == "epub_output"
    assert STATIC_LINKS["direct_text.settings"][0][2] == "action:chat_settings_global"
    import settings_schema as ss

    for key in schema.section("direct_text.settings").keys:
        assert ss.spec(key).readonly == "Edited in Chat settings › All chats", key
    assert ss.spec("direct_text_force_simple_mode").label == "Force simple mode"


def test_context_mode_writes_the_flags_as_one_choice(tmp_path):
    from glossarion_mobile.state.setting_writes import context_mode_of, write_setting
    from glossarion_mobile.ui.chat.plan_model import RUN_OPTION_KEYS, run_options_summary

    store, _schema = _store(tmp_path, {"contextual": True, "use_rolling_summary": False, "batching_mode": "direct"})
    write_setting(store, "context_mode", "rolling_summary_append")
    assert store.get("contextual") is False and store.get("use_rolling_summary") is True
    assert store.get("rolling_summary_mode") == "append" and context_mode_of(store.snapshot()) == "rolling_summary_append"
    write_setting(store, "context_mode", "off")
    assert store.get("use_rolling_summary") is False and store.get("batching_mode") == "aggressive"
    store._saver.close()
    # a desktop rolling-summary config (contextual off) reads as Rolling summary in the Plan card's line
    desktop = {"contextual": False, "use_rolling_summary": True, "rolling_summary_mode": "replace"}
    assert "Rolling summary" in run_options_summary(desktop.get)
    assert "History 3" in run_options_summary({"contextual": True, "translation_history_limit": 3}.get)
    assert "context_mode" in RUN_OPTION_KEYS and "contextual" not in RUN_OPTION_KEYS


@needs_flet
def test_context_mode_tile_and_the_follow_tiles(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.tiles import ContextModeTile

    ctx = _ctx(tmp_path, {"contextual": True})
    page = SectionPage(parse_route("/settings/s/context_memory"), ctx)
    page.get_body()
    assert page.keys[0] == "context_mode" and page.headings["context_mode"] == "Context mode"
    tile = page.tile("context_mode")
    assert isinstance(tile, ContextModeTile) and tile.value() == "contextual_history" and tile.editable
    assert tile.apply("rolling_summary_replace")
    assert ctx.store.get("use_rolling_summary") is True and ctx.store.get("contextual") is False
    follow = page.tile("contextual")
    assert follow.readonly_reason == "Follows Context mode" and not follow.editable
    ctx.store._saver.close()


def test_reader_library_progress_and_mobile_notes_are_typed():
    import library_core
    import reader_doc
    import settings_schema as ss

    assert ss.coerce("retranslation_show_model_info", "false") is False
    assert ss.spec("epub_details_show_special_files").type == "bool"
    assert [v for v, _l in ss.CHOICES_OVERRIDES["epub_library_card_size"]] == list(library_core._ALL_SIZES)
    assert [v for v, _l in ss.CHOICES_OVERRIDES["epub_library_sort"]] == [
        library_core.SORT_DATE, library_core.SORT_NAME, library_core.SORT_SIZE]
    assert [v for v, _l in ss.CHOICES_OVERRIDES["epub_reader_layout"]] == list(reader_doc.READER_LAYOUTS)
    assert [label for _v, label in ss.CHOICES_OVERRIDES["epub_reader_theme"]] == list(reader_doc.READER_THEME_NAMES)
    assert ss.spec("temperature").type == "float" and "refinement" in ss.spec("temperature").label.lower()
    assert ss.spec("selected_files").section == "internal.state"
    assert ss.MOBILE_PARTIAL_REASONS["advanced_watermark_removal"] == "Advanced FFT is slow on phones"
    assert ss.MOBILE_PARTIAL_REASONS["manga_settings.advanced.panel_max_workers"] == "Capped to 2 on phones"
    assert list(ss.choice_values("vision_ocr_source_prepass")) == ["auto", "on", "off"]
    from glossarion_mobile.ui.chat.mode_options_sheet import MODE_OPTION_KEYS

    assert "vision_ocr_source_prepass" in MODE_OPTION_KEYS["vision"]


def test_vision_prepass_choices_are_values_the_backend_reads(monkeypatch):
    """The backend's rule (TransateKRtoEN ``_vision_ocr_source_prepass_enabled_for_mode``, read from source so the
    heavy module is not imported) turns the schema's On / Off into a forced prepass / direct OCR."""
    import ast

    source = (SRC_DIR / "TransateKRtoEN.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "_vision_ocr_source_prepass_enabled_for_mode")
    namespace = {"os": os, "_vision_ocr_mode_enabled": lambda: True}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<prepass>", "exec"), namespace)
    rule = namespace["_vision_ocr_source_prepass_enabled_for_mode"]
    for value, expected in (("on", True), ("off", False)):
        monkeypatch.setenv("VISION_OCR_SOURCE_PREPASS", value)
        assert rule({"vision_ocr_source_prepass": value}) is expected


# ==========================================================================
# Library / Reader / Tools
# ==========================================================================


def test_manga_jobs_cap_the_workers_on_a_phone(monkeypatch):
    from glossarion_mobile.services import manga as svc

    config = {"manga_settings": {"advanced": {"panel_max_workers": 8, "max_workers": "3", "parallel_panel_translation": True}}}
    notes = svc.cap_workers(config)
    assert config["manga_settings"]["advanced"]["panel_max_workers"] == svc.MOBILE_MANGA_MAX_WORKERS == 2
    assert config["manga_settings"]["advanced"]["max_workers"] == 2 and len(notes) == 2
    assert svc.cap_workers({"manga_settings": {"advanced": {"panel_max_workers": 1}}}) == []
    monkeypatch.setattr(svc, "_is_mobile", lambda: True)
    monkeypatch.setattr(svc, "apply_phone_defaults", lambda config: config)
    run = {"manga_settings": {"advanced": {"panel_max_workers": 8}}}
    assert any("panel_max_workers 8 → 2" in line for line in svc.prepare_run(run))
    assert run["manga_settings"]["advanced"]["panel_max_workers"] == 2


def test_empty_chapters_are_detected():
    from glossarion_mobile.ui.reader.blocks import Block
    from glossarion_mobile.ui.reader.document import blocks_empty, chapter_body_empty

    assert chapter_body_empty("<div>\n <p>&nbsp;</p></div><style>p{}</style>")
    assert not chapter_body_empty("<p>Hello</p>") and not chapter_body_empty('<div><img src="a.png"/></div>')
    assert blocks_empty([]) and blocks_empty([Block(kind="para", text="  ")])
    assert not blocks_empty([Block(kind="image", src="a.png")]) and not blocks_empty([Block(kind="para", text="x")])


def test_organize_selected_narrows_the_shelf_plan(tmp_path):
    from glossarion_mobile.services.library import LibraryService

    raw_dir = tmp_path / "Library" / "Raw"
    raw_dir.mkdir(parents=True)
    a, b = tmp_path / "a.epub", tmp_path / "b.epub"
    a.write_bytes(b"PK")
    b.write_bytes(b"PK")
    (raw_dir / "b.epub").write_bytes(b"PK")  # b collides
    book_a = {"name": "A", "raw_source_path": str(a)}
    book_b = {"name": "B", "raw_source_path": str(b)}

    class Shelf:
        def plan_organize(self):
            return {"raw_moves": [(book_a, str(a)), (book_b, str(b))], "translated_moves": [],
                    "raw_dir": str(raw_dir), "trans_dir": str(tmp_path / "Library" / "Translated"),
                    "preview": ["Raw → Library/Raw: 2 files"], "collisions": [(str(b), str(raw_dir / "b.epub"))]}

        @staticmethod
        def _organize_preview_lines(plan):
            return [f"Raw → Library/Raw: {len(plan['raw_moves'])} file"]

        @staticmethod
        def _organize_collisions(plan):
            return ([(src, os.path.join(plan["raw_dir"], os.path.basename(src))) for _b, src in plan["raw_moves"]
                     if os.path.isfile(os.path.join(plan["raw_dir"], os.path.basename(src)))], [])

    service = LibraryService.__new__(LibraryService)
    service._shelf_for = lambda config=None: Shelf()
    service.bid_for = lambda book: str(book.get("name"))
    plan = service.plan_organize_blocking([book_a])
    assert plan["raw_moves"] == [(book_a, str(a))] and plan["collisions"] == [] and plan["preview"] == [
        "Raw → Library/Raw: 1 file"]
    plan = service.plan_organize_blocking([{"name": "other", "raw_source_path": str(b)}])
    assert plan["raw_moves"] == [(book_b, str(b))] and len(plan["raw_collisions"]) == 1
    assert len(service.plan_organize_blocking()["raw_moves"]) == 2


@needs_flet
def test_open_file_uses_the_read_only_editor(tmp_path):
    from glossarion_mobile.ui.library.chapters_tab import ChaptersTab
    from glossarion_mobile.ui.tools import text_editor

    out = tmp_path / "ws"
    out.mkdir()
    (out / "response_001.html").write_text("<p>x</p>", encoding="utf-8")
    (out / "chapter.mp3").write_bytes(b"ID3")
    went: list = []
    said: list = []
    shared: list = []

    async def io(fn, *args):
        return fn(*args)

    async def share(paths):
        shared.append(paths)
        return True

    ctx = types.SimpleNamespace(prefs=types.SimpleNamespace(file_ref=lambda p: "f" * 12), io=io, say=said.append,
                                go=lambda *a: went.append(a), files=types.SimpleNamespace(share=share))
    tab = types.SimpleNamespace(ctx=ctx, view=types.SimpleNamespace(output_dir=str(out)))
    tab.output_path = lambda row: ChaptersTab.output_path(tab, row)
    tab.share_output = lambda row: ChaptersTab.share_output(tab, row)
    html_row = types.SimpleNamespace(output_file="response_001.html")
    assert asyncio.run(ChaptersTab.open_output(tab, html_row)) == "f" * 12
    assert went[-1][0] == "tools.text" and text_editor.take_request("f" * 12).read_only
    asyncio.run(ChaptersTab.open_output(tab, types.SimpleNamespace(output_file="chapter.mp3")))
    assert shared and shared[-1] == [str(out / "chapter.mp3")]
    asyncio.run(ChaptersTab.open_output(tab, types.SimpleNamespace(output_file="gone.html")))
    assert said[-1] == "The output file is not on disk"
    ctx.files = None
    assert not asyncio.run(ChaptersTab.share_output(tab, html_row)) and said[-1] == "Sharing files is not available here"


@needs_flet
def test_glossary_progress_rows_are_windowed_and_replaced_in_place():
    from glossarion_mobile.ui.library import progress_model as pm
    from glossarion_mobile.ui.library.glossary_tab import GlossaryTab

    def row(index, status="completed"):
        return pm.GlossaryRowVM(key=f"gp-{index}", kind="chapter", status=status, icon="✅", label="Done",
                                title=f"Chapter {index}")

    rows = tuple(row(i) for i in range(2000))
    view = pm.GlossaryView(rows=rows, path="/x/glossary_progress.json", book_title="Book")
    ctx = types.SimpleNamespace(spawn=lambda c: c.close() if hasattr(c, "close") else None, push=lambda *a: None,
                                io=None, dark=False, text_scale=1.0, tablet=False, page=None,
                                haptic=lambda *a: None, say=lambda *a: None)
    page = types.SimpleNamespace(ctx=ctx, glossary=None, book={"name": "Book"}, full_refresh=lambda: None,
                                 service=types.SimpleNamespace(
                                     has_job_kind=lambda k: True, glossary_hooks=None, cfg=lambda k, d=None: d,
                                     core=types.SimpleNamespace(available=lambda *a: False, fn=lambda *a: None,
                                                                has_module=lambda *a: False)),
                                 open_files=lambda: None)
    tab = GlossaryTab(page)
    tab.build()
    tab.apply(view)
    assert tab.renders == 1 and tab.rows_list.windowed and tab.rows_list.mounted_count <= 150
    assert any(getattr(c, "key", "") == "gp-refresh" for c in tab.path_row.controls)
    changed = view.rows[:3] + (row(3, "failed"),) + view.rows[4:]
    tab.apply(pm.GlossaryView(rows=changed, path=view.path, book_title="Book"))
    assert tab.renders == 1  # a poll change replaced the row, no rebuild
    tab.on_row_long_press(changed[5])
    tab.toggle_select(changed[6].key)
    assert tab.renders == 1 and tab.selected == {changed[5].key, changed[6].key}  # taps replace their rows
    tab.select_all()  # every mounted row changes: one rebuild of the mounted window, where it is
    assert tab.renders == 2 and len(tab.selected) == 2000
    tab.exit_selection()
    assert tab.renders == 3 and not tab.selected


@needs_flet
def test_review_tool_shows_an_error_card_and_view_log():
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState
    from glossarion_mobile.ui.tools.review import ReviewScreen

    went: list = []
    ctx = types.SimpleNamespace(tool_state={}, cfg=lambda k, d=None: d, dark=False, go=lambda *a: went.append(a),
                                spawn=lambda c: c.close(), copy_text=None, config_snapshot=lambda: {},
                                schema=None, store=None, say=lambda *a: None)
    screen = ReviewScreen.__new__(ReviewScreen)
    screen.ctx = ctx
    screen.watch = types.SimpleNamespace(active=lambda: None)
    screen.last_job = None
    import flet as ft

    screen.log_button = ft.TextButton(content="View log", visible=False)
    screen.error_holder = ft.Container(visible=False)
    screen._set_running = lambda running, text: None
    screen.load_current = lambda: _noop()
    failed = JobSnapshot(id="r" * 12, spec=JobSpec("review", "Book", ("a.epub",)), state=JobState.FAILED,
                         created=0.0, error="boom")
    ReviewScreen._on_job_end(screen, failed)
    assert screen.error_holder.visible and screen.log_button.visible
    assert getattr(screen.error_holder.content, "key", "") == "review-error-card"
    ReviewScreen.view_log(screen)
    assert went[-1] == ("jobs.detail", {"jid": "r" * 12})
    done = JobSnapshot(id="d" * 12, spec=JobSpec("review", "Book", ("a.epub",)), state=JobState.DONE, created=0.0)
    ReviewScreen._on_job_end(screen, done)
    assert not screen.error_holder.visible


async def _noop():
    return None


# ==========================================================================
# no ReasonChip under a disabled control (UI_SPEC §5.2)
# ==========================================================================


@needs_flet
def test_unavailable_rows_keep_their_reason_chips_tappable(tmp_path, monkeypatch):
    import flet as ft

    from glossarion_mobile.services import webview_bridge as wb
    from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.accounts import AccountsScreen
    from glossarion_mobile.ui.screens.endpoints import EndpointsScreen
    from glossarion_mobile.ui.screens.local_ai import LocalAiScreen
    from glossarion_mobile.ui.settings.section_page import static_row

    monkeypatch.setattr(wb, "availability", lambda: (False, wb.UNSUPPORTED_REASON))
    accounts = AccountsScreen(parse_route("/settings/accounts"), oauth=None)
    accounts._provider_card = lambda provider: ft.Container()
    body = accounts.build_body()
    assert _dead_chips(body) == [] and any(True for _c, _p in _walk(body))
    ctx = _ctx(tmp_path)
    endpoints = EndpointsScreen(None, ctx)
    assert _dead_chips(endpoints.build_body()) == []
    local = LocalAiScreen(None, store=ctx.store)
    assert _dead_chips(local.build_body()) == []
    assert _dead_chips(static_row("Lock mouse wheel", "Desktop only", "why")) == []
    sheet = ActionSheet([ActionItem("Argos", disabled_reason="Needs argostranslate")])
    assert _dead_chips(sheet.dialog) == [] and sheet.tiles[0].on_click is not None
    from glossarion_mobile.ui.screens.appearance import AppearanceScreen

    prefs = types.SimpleNamespace(get=lambda k, d=None: d, set=lambda k, v: None)
    appearance = AppearanceScreen(parse_route("/settings/appearance"),
                                  types.SimpleNamespace(prefs=prefs, page=None),
                                  state=types.SimpleNamespace(text_scale=types.SimpleNamespace(value=1.0, set=lambda v: None)),
                                  haptics=types.SimpleNamespace(enabled=True))
    assert _dead_chips(appearance.get_body()) == []
    ctx.store._saver.close()
