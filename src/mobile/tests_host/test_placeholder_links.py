"""Mobile placeholder links (owner 2026-11-14: "can you make it so mobile marks placeholders?").

The desktop marks AI prompt placeholders like links with a live hover tooltip; mobile marks them in the chips
(tap inserts, hold opens the value sheet), in a marked view of the prompt (tap a placeholder: the value sheet)
and the value sheet says what the placeholder says right now, worked out from the live config when it opens
(``ui.components.placeholder_links`` + ``services.prompt_placeholders``). Display only.

Run (src/mobile)::

    .venv/Scripts/python -m pytest tests_host/test_placeholder_links.py -q -p no:cacheprovider
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
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

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
needs_core = pytest.mark.skipif(not _has("extract_glossary_from_epub"), reason="the glossary core is not importable")


@pytest.fixture(autouse=True)
def _clean_glossary_env(monkeypatch):
    """The preview reads a setting the mapping lacks from os.environ: none may leak in from other tests."""
    for key in list(os.environ):
        if key.startswith("GLOSSARY_"):
            monkeypatch.delenv(key, raising=False)


class _Page:
    def __init__(self):
        self.dialogs: list = []
        self.width, self.height = 412, 900

    def show_dialog(self, dialog):
        dialog.open = True
        self.dialogs.append(dialog)

    def update(self, *a):
        pass


class _Ctx:
    """The SettingsContext surface the prompt editors use."""

    def __init__(self, store=None):
        self.store = store
        self.page = _Page()
        self.copied: list = []

    def show_dialog(self, dialog):
        self.page.show_dialog(dialog)

    def push(self, *controls):
        pass

    async def run_io(self, fn, *args):
        return fn(*args)

    def copy_text(self, text):
        self.copied.append(text)


def _walk(control, out=None):
    from flet.controls.base_control import BaseControl

    out = [] if out is None else out
    if isinstance(control, (list, tuple)):
        for item in control:
            _walk(item, out)
        return out
    if not isinstance(control, BaseControl):
        return out
    out.append(control)
    for attr in ("controls", "content", "title", "subtitle", "leading", "trailing", "label", "actions"):
        value = getattr(control, attr, None)
        if isinstance(value, (list, tuple, BaseControl)):
            _walk(value, out)
    return out


def _by_key(root, key):
    return next((c for c in _walk(root) if getattr(c, "key", None) == key), None)


def _types(**gender):
    """Three entry types; ``gender`` overrides has_gender per type."""
    types = {"character": {"enabled": True, "has_gender": True}, "terms": {"enabled": True, "has_gender": False},
             "titles": {"enabled": True, "has_gender": False}}
    for name, value in gender.items():
        types[name]["has_gender"] = value
    return types


# ==========================================================================
# the pure rules
# ==========================================================================


def test_marked_segments_mark_only_the_replaced_placeholders_and_keep_the_text():
    from glossarion_mobile.ui.components.placeholder_links import marked_segments

    text = "Translate to {target_lang}.\n{split_marker_instruction}\nKeep {words} and {target_lang} {x"
    segments = marked_segments(text, {"target_lang", "split_marker_instruction"})
    assert "".join(s for s, _n in segments) == text  # identical text
    assert [s for s, n in segments if n] == ["{target_lang}", "{split_marker_instruction}", "{target_lang}"]
    assert [n for _s, n in segments if n] == ["target_lang", "split_marker_instruction", "target_lang"]
    assert ("{words}" in "".join(s for s, n in segments if n is None))  # a literal {word} stays plain
    assert marked_segments("", {"x"}) == [("", None)]
    assert marked_segments("no placeholders", set()) == [("no placeholders", None)]


def test_value_lines_say_inactive_per_request_and_the_verbatim_text():
    from glossarion_mobile.ui.components.placeholder_links import value_lines

    raw = "Use `code` and <b>tags</b>\nverbatim"
    assert value_lines(raw) == (raw, [])
    assert value_lines({"active": False, "inactive": "Inactive right now: its line is removed from the prompt."}) == (
        "", ["Inactive right now: its line is removed from the prompt."])
    assert value_lines({"per_request": True, "text": "the raw text."}) == ("", ["Filled in for each request: the raw text."])
    assert value_lines({"error": "boom"}) == ("", ["The current value could not be read: boom"])


def test_links_for_setting_marks_only_what_each_request_replaces():
    from glossarion_mobile.services.prompt_placeholders import links_for_setting

    store = {"output_language": "German"}
    assert links_for_setting("refinement_system_prompt", store).names_now() == {
        "target_lang", "QA_Issues", "QA_ISSUES", "qa_issues"}
    assert {"html", "content"} <= links_for_setting("refinement_user_prompt", store).names_now()
    assert links_for_setting("rolling_summary_user_prompt", store).names_now() == {"translations"}
    assert links_for_setting("book_title_system_prompt", store).names_now() == {"target_lang"}
    assert links_for_setting("vision_ocr_translation_user_prompt", store).names_now() == {"context", "ocr_text"}
    assert links_for_setting("append_glossary_prompt", store) is None  # replaces nothing
    assert links_for_setting("manga_ocr_prompt", store) is None
    # Other › Stored system_prompt (async_batch_core fills both) and Other › Metadata system prompt (run_env)
    assert links_for_setting("system_prompt", store).names_now() == {"target_lang", "split_marker_instruction"}
    assert links_for_setting("metadata_system_prompt", store).names_now() == {"target_lang"}
    assert links_for_setting("metadata_system_prompt", store).value("target_lang", "") == "German"


def test_target_lang_shows_the_current_output_language():
    from glossarion_mobile.services.prompt_placeholders import links_for_setting, translation_links

    store = {"output_language": "German"}
    links = translation_links(store)
    assert links.value("target_lang", "") == "German"
    store["output_language"] = "Korean"  # changed since the editor opened
    assert links.value("target_lang", "") == "Korean"
    assert links_for_setting("review_system_prompt", store).value("target_lang", "") == "Korean"
    assert translation_links(store, target_lang=lambda: "French").value("target_lang", "") == "French"  # chat override
    off = links.value("split_marker_instruction", "")
    assert off["active"] is False and "Request Merging is off" in off["inactive"]
    store["request_merging_enabled"] = True
    assert links.value("split_marker_instruction", "")["text"].startswith("- CRITICAL Requirement")


# ==========================================================================
# the editors: toggle, chips, sheet
# ==========================================================================


def _sheet_editor(store, **kwargs):
    from glossarion_mobile.services.prompt_placeholders import translation_links
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

    ctx = _Ctx(store)
    editor = PromptEditorSheet(ctx, title="Profile", value="Translate to {target_lang}. Keep {braces}.",
                               links=translation_links(store), counter=lambda text, model: 7, **kwargs)
    return ctx, editor


@needs_flet
def test_marked_view_toggle_keeps_value_dirty_state_and_token_count():
    store = {"output_language": "German"}
    ctx, editor = _sheet_editor(store)
    editor.field.value = "Unsaved {target_lang} edit with {split_marker_instruction} and {not_one}"
    editor._on_change()
    before = (editor.field.value, editor.dirty, editor.counter.value, editor.collect())

    async def scenario():
        await editor.marked.set_marked(True)
        view = editor.marked
        assert view.marked and view.view.visible and not view.holder.visible
        assert view.toggle.tooltip == "Back to editing"
        assert "".join(span.text for span in view.text.spans) == editor.field.value  # the unsaved edit, verbatim
        linked = [span for span in view.text.spans if span.on_click is not None]
        assert [span.text for span in linked] == ["{target_lang}", "{split_marker_instruction}"]
        assert all(span.style.decoration is not None and span.style.color for span in linked)
        assert (editor.field.value, editor.dirty, editor.counter.value, editor.collect()) == before
        # a marked placeholder opens what it says now
        linked[0].on_click(None)
        sheet = editor.linker.sheet
        await sheet._task
        assert ctx.page.dialogs[-1] is sheet.dialog and sheet.title.value == "{target_lang}"
        assert sheet.text == "German" and sheet.value_text.value == "German"
        sheet.copy_value()
        assert ctx.copied == ["German"]
        await view.set_marked(False)
        assert not view.marked and view.holder.visible and not view.view.visible
        assert (editor.field.value, editor.dirty, editor.counter.value, editor.collect()) == before

    asyncio.run(scenario())


@needs_flet
def test_chip_tap_inserts_at_the_caret_and_a_long_press_opens_the_sheet():
    import flet as ft

    store = {"output_language": "Spanish"}
    ctx, editor = _sheet_editor(store)
    chip = next(c for c in editor.chips if c.key == "placeholder-target_lang")
    assert chip.label.style.decoration is not None and chip.label.style.color  # link-styled
    assert _by_key(editor.body, "prompt-ph-hint").value == "Tap to insert · hold to see what it says now"
    editor.cursor.on_selection_change(type("E", (), {"selection": ft.TextSelection(base_offset=0, extent_offset=0)})())
    chip.on_click(None)
    assert editor.field.value.startswith("{target_lang}Translate to")  # inserted at the caret, as before
    holder = _by_key(editor.body, "placeholder-target_lang-hold")
    assert isinstance(holder, ft.GestureDetector) and holder.content is chip

    async def scenario():
        holder.on_long_press(None)
        sheet = editor.linker.sheet
        assert ctx.page.dialogs[-1] is sheet.dialog  # opens at once
        await sheet._task
        assert sheet.text == "Spanish"

    asyncio.run(scenario())
    assert editor.field.value.startswith("{target_lang}Translate to")  # the long press inserted nothing


@needs_flet
def test_pane_chips_keep_their_keys_and_get_the_marked_view():
    from glossarion_mobile.services.prompt_placeholders import translation_links
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorPane

    pane = PromptEditorPane(value="Translate", counter=lambda text, model: 1, links=translation_links({}))
    assert [chip.key for chip in pane.chips] == ["placeholder-target_lang", "placeholder-split_marker_instruction"]
    assert pane.marked is not None and pane.marked.toggle in _walk(pane)
    pane.field.value = "x {target_lang}"

    async def scenario():
        await pane.marked.set_marked(True)
        assert "".join(s.text for s in pane.marked.text.spans) == "x {target_lang}"
        pane.insert("{split_marker_instruction}")  # a chip tap goes back to editing first
        assert not pane.marked.marked and pane.field.value.endswith("{split_marker_instruction}")

    asyncio.run(scenario())


@needs_flet
def test_prompt_tile_editor_gets_the_links_of_its_setting():
    from glossarion_mobile.services.prompt_placeholders import links_for_setting
    from glossarion_mobile.ui.settings.editors import PromptEditor

    ctx = _Ctx({})
    plain = PromptEditor(ctx, title="Append", value="x", links=links_for_setting("append_glossary_prompt", {}))
    assert plain.marked is None  # nothing to mark: no toggle
    marked = PromptEditor(ctx, title="Summary", value="{translations}",
                          links=links_for_setting("rolling_summary_user_prompt", {}))
    assert marked.marked.toggle in _walk(marked.header)  # by Save: the footer row is full at 360 dp
    assert marked.marked.toggle not in _walk(marked.footer)

    async def scenario():
        await marked.marked.set_marked(True)
        marked.linker.open("translations")
        await marked.linker.sheet._task
        assert marked.linker.sheet.text == ""
        assert marked.linker.sheet.notes[0].startswith("Filled in for each request: the translated text")

    asyncio.run(scenario())


@needs_flet
def test_parallel_pair_wrapper_chips_and_marked_views(tmp_path, monkeypatch):
    """The pair wrapper chips are link-styled (tap appends as before, hold: what it is filled with); the system
    prompt and the wrapper each have a marked view."""
    spec = importlib.util.spec_from_file_location("_glossarion_gui_helpers_ph", Path(__file__).with_name("test_glossary_ui.py"))
    GU = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GU)
    for name in ("HOME", "USERPROFILE", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(name, str(tmp_path / name))
    from glossarion_mobile.ui.glossary.parallel_pair import ParallelPairScreen

    service = GU.make_service({})
    service.default_wrapper_prompt = lambda: "RAW {raw_text} / TR {translated_text}"

    async def scenario():
        _conn, session = GU._TB._fake_session("android")
        ctx = GU._ctx(session.page, service)
        screen = ParallelPairScreen(None, ctx)
        GU._mount(session.page, screen.get_body())
        chip = next(c for c in screen.placeholder_chip_list if c.key == "pp-chip-raw_filename")
        assert chip.label.style.decoration is not None
        chip.on_click(None)
        assert screen.wrapper.value.endswith("{raw_filename}")  # appended, as before
        _by_key(screen.root, "pp-chip-raw_filename-hold").on_long_press(None)
        await screen.wrapper_linker.sheet._task
        assert screen.wrapper_linker.sheet.notes == ["Filled in for each request: the raw HTML file name of each "
                                                     "mapped pair."]
        await screen.wrapper_marked.set_marked(True)
        assert [s.text for s in screen.wrapper_marked.text.spans if s.on_click] == [
            "{raw_text}", "{translated_text}", "{raw_filename}"]
        assert screen.system_marked.toggle in _walk(screen.root)
        # off the loop through the context's io (a Library context has no run_io)
        assert screen.wrapper_linker.run_io is not None and screen.system_linker.run_io is not None
        # same font as the fields, their labels kept, and a toggle label per prompt
        assert screen.wrapper_marked.text.font_family is None and screen.system_marked.text.font_family is None
        assert _by_key(screen.root, "pp-wrapper-marked-label").value == "Pair wrapper prompt"
        assert screen.wrapper_marked.toggle.tooltip == "Back to editing the pair wrapper prompt"
        assert screen.system_marked.toggle.tooltip.startswith("Show system prompt placeholders marked")
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# glossary prompts: live settings, inactive, per request
# ==========================================================================


@needs_flet
@needs_core
def test_glossary_sheet_follows_the_live_entry_types_without_reopening_the_editor():
    from extract_glossary_from_epub import DEFAULT_GLOSSARY_PROMPT
    from glossarion_mobile.services.prompt_placeholders import glossary_links
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

    store = {"custom_entry_types": _types(), "glossary_enable_relations_tracker": True}
    ctx = _Ctx(store)
    editor = PromptEditorSheet(ctx, title="Default", value=DEFAULT_GLOSSARY_PROMPT,
                               placeholders=("{entries}", "{gender_instruction}", "{relations_instruction}"),
                               links=glossary_links("balanced", store), counter=lambda text, model: 1)

    async def says(name):
        editor.linker.open(name)
        await editor.linker.sheet._task
        return editor.linker.sheet.text

    async def scenario():
        gender, relations, entries = await says("gender_instruction"), await says("relations_instruction"), \
            await says("entries")
        assert "applies only to character entries" in gender
        assert "for character entries only" in relations
        assert entries == "Character, Terms, & Titles entries"
        # the live settings change while the editor stays open
        store["custom_entry_types"]["titles"]["has_gender"] = True
        assert "applies only to character, titles entries" in await says("gender_instruction")
        assert "for character, titles entries only" in await says("relations_instruction")
        store["custom_entry_types"]["terms"]["enabled"] = False
        assert await says("entries") == "Character & Titles entries"
        # the marked view marks what build_prompt replaces, and not a literal {word}
        editor.field.value = DEFAULT_GLOSSARY_PROMPT + "\n{not_a_placeholder}"
        await editor.marked.set_marked(True)
        linked = {span.text for span in editor.marked.text.spans if span.on_click is not None}
        assert "{entries}" in linked and "{not_a_placeholder}" not in linked

    asyncio.run(scenario())


@needs_flet
@needs_core
def test_inactive_and_per_request_glossary_placeholders_say_so():
    from extract_glossary_from_epub import DEFAULT_AUTO_GLOSSARY_PROMPT, DEFAULT_GLOSSARY_PROMPT
    from glossarion_mobile.services.prompt_placeholders import glossary_links
    from glossarion_mobile.ui.components.placeholder_links import PlaceholderLinker

    store = {"custom_entry_types": _types(), "glossary_enable_relations_tracker": False}

    async def says(mode, text, name):
        linker = PlaceholderLinker(glossary_links(mode, store), text=lambda: text, page=_Page())
        sheet = linker.open(name)
        await sheet._task
        return sheet

    async def scenario():
        off = await says("balanced", DEFAULT_GLOSSARY_PROMPT, "relations_instruction")
        assert off.text == "" and off.notes[0] == "Inactive right now: its line is removed from the prompt."
        assert not off.value_box.visible
        marker = await says("minimal", DEFAULT_AUTO_GLOSSARY_PROMPT, "marker")
        assert marker.text == "" and marker.notes[0].startswith("Filled in for each request: context window marker")

    asyncio.run(scenario())


def _fresh_store(tmp_path):
    """A first-run MobileConfigStore with the schema display defaults (nothing stored for the entry types)."""
    import json

    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    path = tmp_path / "config.json"
    store = MobileConfigStore(
        path, debounce=10, defaults=SchemaAccess().effective_default,
        reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {},
        writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


@needs_core
def test_unset_entry_types_list_in_the_order_the_job_sends(tmp_path):
    """Nothing stored: the schema display default has sorted keys, but the job's owner (and the core's own
    default for an empty value) uses the desktop order; the sheet must say what the request says."""
    from extract_glossary_from_epub import DEFAULT_CUSTOM_ENTRY_TYPES, DEFAULT_GLOSSARY_PROMPT, placeholder_value
    from glossarion_mobile.services.prompt_placeholders import glossary_refinement_links, glossary_settings

    store = _fresh_store(tmp_path)
    assert store.get("custom_entry_types", None) is None
    settings = glossary_settings(store)
    entries = placeholder_value("entries", mode="balanced", prompt=DEFAULT_GLOSSARY_PROMPT, settings=settings)
    assert entries["text"] == "Character, Term, Surnames, Titles, Locations, & Nicknames entries"
    store.set("glossary_refinement_chunking_mode", "all")
    assert glossary_refinement_links(store).value("entry_types", "") == ", ".join(DEFAULT_CUSTOM_ENTRY_TYPES)
    # a stored value is what the job sends
    store.set("custom_entry_types", {"titles": {"enabled": True, "has_gender": True},
                                     "character": {"enabled": True, "has_gender": True}})
    entries = placeholder_value("entries", mode="balanced", prompt=DEFAULT_GLOSSARY_PROMPT,
                                settings=glossary_settings(store))
    assert entries["text"] == "Titles & Character entries"


def test_settings_context_carries_the_clipboard_for_the_value_sheet():
    from glossarion_mobile.ui.settings.integration import SettingsFeature

    copied: list = []
    feature = SettingsFeature(page=None, store={}, prefs=None, copy_handler=copied.append)
    assert feature.ctx.copy_text is not None
    feature.ctx.copy_text("x")
    assert copied == ["x"]
