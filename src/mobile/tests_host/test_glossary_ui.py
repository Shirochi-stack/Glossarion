"""Host tests for the U6 Glossary Manager (UI_SPEC §4.1): service, editor, jobs, settings tabs, feature.

* Save round trip: edits made through ``GlossaryService`` (EntrySheet values, one undo step)
  and saved write the same bytes as the shared ``glossary_document.GlossaryDocument`` driven
  directly (what the desktop editor runs), for token CSV, JSON list and JSON dict glossaries;
  a repeated save is stable.
* The backup step: the desktop ``create_glossary_backup`` (``glossary_files``, a fake here)
  before destructive actions, and its "Backup Failed … Continue anyway?" question.
* The Editor over a 10,000-entry glossary in the fake Flet session: the WindowedList builds
  one step, windows of 1,500 rows, loads more, jumps across windows, search / column filters.
* EntrySheet edit → Save with the desktop "Update output files" question (declined: nothing
  written; accepted: the glossary and the output chapter are updated, byte-identical to the
  shared document); Find / Replace preview, Find Next, Replace, Replace All, the "No glossary
  match" output-file fallback and its undo; Delete / Backups / Restore / Filter / Trim.
* U6 review: the swipe delete's Undo, the view derived again after every reload (Hide unused,
  pruned column filters), "Info" boxes leaving edits unsaved, in-place selection toggles,
  debounced search, dismissed dialogs, a cancelled file switch, Rebuild Now while busy, and the
  profile bars over the shared ``GlossaryPromptProfiles`` / ``RefinementPromptProfiles``.
* U6 second review: the shell's leave guard for unsaved edits (sidebar, chat row, outside link, a
  popped full-screen View) and the title after a file switch in the running app; Add to glossary
  over the tablet Reader; Hide unused / search kept across a file switch; the file list and the
  Library input read off the UI loop; "From Library…" compiled EPUBs; reopened screens following
  their queued job; the "Use as manual glossary" sheet closed without a choice.
* Mode locks: ``set_mode`` / the lock pass through ``settings_rules``; the General tab's tiles
  show the 🔒 reason from the real settings schema.
* Job kinds (registration, the refinement / unified adapters over fake shared modules, the
  glossary stop protocol) and the feature wiring (routes, Library hooks, the blocking
  "Continue anyway?" bridge, delete / restore glossary files with the desktop texts).

Every test isolates HOME / USERPROFILE / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY under
``tmp_path``; nothing reads or writes the developer's Library, output folders or config.json.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_glossary_ui.py
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
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

from glossarion_mobile.services.glossary import (  # noqa: E402
    GlossaryService,
    ViewState,
    doc_count,
    translated_field,
)
from glossarion_mobile.services.library import SharedCore  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _core(name: str, *attrs: str):
    """The real shared module when it has every attribute, else None."""
    if not _has(name):
        return None
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return module if all(hasattr(module, a) for a in attrs) else None


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
gd = _core("glossary_document", "GlossaryDocument", "write_token_csv", "editor_row_specs", "EditorRow")
needs_gd = pytest.mark.skipif(gd is None, reason="glossary_document (U6 chain 1) not importable")
rules = _core("settings_rules", "apply_glossary_mode_locks", "glossary_mode_toggle_states", "apply_change",
              "auto_glossary_modes", "glossary_mode_label", "evaluate_locks")
needs_rules = pytest.mark.skipif(rules is None, reason="settings_rules (glossary mode locks) not importable")


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Scratch HOME / Library / output roots (the developer's folders are never read)."""
    home, library, output = tmp_path / "home", tmp_path / "Library", tmp_path / "Output"
    for folder in (home, library, output):
        folder.mkdir()
    for name, value in (("HOME", home), ("USERPROFILE", home), ("GLOSSARION_LIBRARY_DIR", library),
                        ("OUTPUT_DIRECTORY", output)):
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("GLOSSARY_SHARED_DIR", raising=False)
    return types.SimpleNamespace(root=tmp_path, home=home, library=library, output=output)


ENTRIES = [
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female", "description": "a witch"},
    {"type": "character", "raw_name": "카이", "translated_name": "Kai", "gender": "male"},
    {"type": "terms", "raw_name": "마나", "translated_name": "Mana"},
    {"type": "terms", "raw_name": "검", "translated_name": "Sword"},
]


def write_token_glossary(path: Path, entries=ENTRIES) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    gd.write_token_csv([dict(e) for e in entries], str(path), [], ["description"])
    return str(path)


def book_glossary(folder: Path) -> str:
    """<folder>/Book_glossary.csv next to one translated chapter (the editor's output folder)."""
    path = write_token_glossary(folder / "Book_glossary.csv")
    (folder / "response_001_ch1.html").write_text("<p>Luna met Kai.</p>", encoding="utf-8")
    return path


class FakeFiles:
    """``glossary_files`` stand-in: the desktop backup (a JSON snapshot in Backups/) + delete / restore."""

    def __init__(self) -> None:
        self.calls: list = []
        self.fail = None
        self.plan: list = []
        self.backup = (None, [])

    def create_glossary_backup(self, doc, operation_name="manual"):
        self.calls.append(("backup", os.path.basename(doc.path), operation_name))
        if self.fail:
            raise OSError(self.fail)
        folder = os.path.join(os.path.dirname(doc.path), "Backups")
        os.makedirs(folder, exist_ok=True)
        stem = os.path.splitext(os.path.basename(doc.path))[0]
        name = f"{stem}_{operation_name}_{time.strftime('%Y%m%d_%H%M%S')}_{len(self.calls)}.json"
        with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
            json.dump(doc.current_glossary_data, handle, ensure_ascii=False, indent=2)
        return True

    def collect_glossary_files_for_inputs(self, inputs, config=None):
        self.calls.append(("plan", [os.path.basename(p) for p in inputs]))
        return list(self.plan)

    @staticmethod
    def glossary_delete_display(all_files):
        lines = []
        for book in dict.fromkeys(b for b, _p in all_files):
            lines.append(f"[{book}]")
            lines.extend(f"  {os.path.basename(p)}" for b, p in all_files if b == book)
        return lines

    def delete_glossary_files(self, files, append_log=None):
        self.calls.append(("delete", [os.path.basename(p) for _b, p in files]))
        return [f"{book}/{os.path.basename(path)}" for book, path in files]

    def find_latest_glossary_backup(self, inputs, config=None):
        self.calls.append(("latest", [os.path.basename(p) for p in inputs]))
        return self.backup

    def restore_glossary_backup(self, backup_dir, backup_files, append_log=None):
        self.calls.append(("restore", os.path.basename(backup_dir), list(backup_files)))
        return list(backup_files)


gf = _core("glossary_files", "create_glossary_backup", "collect_glossary_files_for_inputs", "delete_glossary_files",
           "find_latest_glossary_backup", "restore_glossary_backup", "glossary_delete_display")
pec = _core("parallel_epub_core", "offset_parallel_epub_mapping", "validate_parallel_epub_pair",
            "rebuild_parallel_epub_pair_result", "build_parallel_epub_pair_artifact")


def make_service(config=None, **kwargs) -> GlossaryService:
    files = FakeFiles()
    service = GlossaryService(config={} if config is None else config, core=SharedCore({"glossary_files": files}),
                              **kwargs)
    service.fake_files = files  # type: ignore[attr-defined]
    return service


_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_glossary",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)


def _ctx(page, service, **kwargs):
    from glossarion_mobile.ui.glossary.common import GlossaryContext

    notes, navigated = [], []
    ctx = GlossaryContext(service=service, page=page, platform="android",
                          notify=lambda message, action=None, on_action=None: notes.append(message),
                          navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                          **kwargs)
    ctx.notes, ctx.navigated = notes, navigated
    return ctx


def _mount(page, body):
    page.views[0].controls.append(body)
    page.update()


def _texts(control, out=None):
    """Every ``Text.value`` / string label under a control."""
    import flet as ft

    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ft.Text) and control.value:
        out.append(str(control.value))
    for name in ("content", "controls", "label", "leading", "title", "subtitle", "trailing"):
        value = getattr(control, name, None)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ft.BaseControl):
                    _texts(item, out)
        elif isinstance(value, ft.BaseControl):
            _texts(value, out)
        elif isinstance(value, str) and name in ("content", "label"):
            out.append(value)
    return out


async def _open_editor(page, service, path):
    from glossarion_mobile.ui.glossary.glossary_view import GlossaryScreen

    ctx = _ctx(page, service)
    screen = GlossaryScreen(None, ctx, path=path)
    _mount(page, screen.get_body())
    doc = await screen.editor.open(path)
    page.update()
    assert doc is not None, screen.editor.error
    return ctx, screen, screen.editor


# ==========================================================================
# Service: save round trip, backups
# ==========================================================================


def _write_fixture(folder: Path, fmt: str) -> str:
    if fmt == "token_csv":
        return write_token_glossary(folder / "Book_glossary.csv")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "Book_glossary.json"
    if fmt == "list":
        data = [dict(e) for e in ENTRIES]
    else:
        data = {e["raw_name"]: e["translated_name"] for e in ENTRIES}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


@needs_gd
@pytest.mark.parametrize("fmt", ["token_csv", "list", "dict"])
def test_save_round_trip_is_byte_identical_to_the_shared_document(iso, fmt):
    path = _write_fixture(iso.output / "A", fmt)
    copy = iso.output / "B" / os.path.basename(path)
    copy.parent.mkdir()
    shutil.copyfile(path, copy)
    config = {"update_html_on_save": False}
    service = make_service(dict(config))
    doc = service.open_document(path)
    assert doc.current_glossary_format == fmt and doc_count(doc) == len(ENTRIES) and not doc.dirty
    specs = service.row_specs(doc)
    assert [s.source_idx for s in specs] == list(range(len(ENTRIES)))
    assert service.save_edits(doc, update_outputs=False)["saved"]
    first = Path(path).read_bytes()
    assert service.save_edits(doc, update_outputs=False)["saved"] and Path(path).read_bytes() == first  # stable
    field = translated_field(doc)
    values = {field: "Lunaria"}
    if fmt != "dict":
        values["description"] = "the witch"
    ref = specs[0].ref
    service.update_entry(doc, ref, values)
    assert doc.dirty and len(doc._undo_stack) == 1  # one undo step for the whole sheet
    assert service.translated_changes(doc) == [("Luna", "Lunaria")]
    title, text = service.update_prompt(service.translated_changes(doc))
    assert title == "Update output files" and text.endswith("Luna -> Lunaria")
    report = service.save_edits(doc, update_outputs=False)
    assert report == {"saved": True, "files_updated": 0, "replacements": 0} and not doc.dirty
    assert ("backup", os.path.basename(path), "before_save") in service.fake_files.calls

    reference = gd.GlossaryDocument(dict(config))  # the desktop editor's steps on the copy
    reference.load(str(copy))
    reference.save_edits(update_output_files=False)
    reference.save_edits(update_output_files=False)
    for key, value in values.items():
        reference.edit_cell(ref, key, value)
    reference.save_edits(update_output_files=False)
    assert Path(path).read_bytes() == copy.read_bytes()


@needs_gd
def test_backup_failure_asks_continue_anyway_and_auto_backup_setting(iso):
    path = write_token_glossary(iso.output / "Book" / "Book_glossary.csv")
    service = make_service({"update_html_on_save": False})
    doc = service.open_document(path)
    before = Path(path).read_bytes()
    service.fake_files.fail = "disk full"
    asked = []
    service.ask_continue = lambda title, text: asked.append((title, text)) or False
    assert service.delete(doc, [0]) is None and Path(path).read_bytes() == before and doc_count(doc) == 4
    assert asked == [("Backup Failed", "Failed to create backup: disk full\n\nContinue anyway?")]
    service.ask_continue = lambda title, text: True  # "Yes": the desktop deletes without the backup
    report = service.delete(doc, [0])
    assert report.ok and report.title == "Success" and report.message == "Deleted 1 entries" and doc_count(doc) == 3
    assert ("backup", "Book_glossary.csv", "before_delete_1") in service.fake_files.calls
    service.fake_files.fail = None
    service.config["glossary_auto_backup"] = False
    calls = len(service.fake_files.calls)
    assert service.backup_callback(doc, "before_trim_1") is True and len(service.fake_files.calls) == calls
    assert service.manual_backup(doc) and service.fake_files.calls[-1] == ("backup", "Book_glossary.csv", "manual")
    assert service.list_backups(doc)[0]["name"].startswith("Book_glossary_manual_")
    assert service.set_backup_settings(True, 7) == gd.backup_settings_message(True, 7)
    assert service.backup_settings() == (True, 7)


@needs_gd
def test_view_rows_search_filters_and_sort(iso):
    path = write_token_glossary(iso.output / "Book" / "Book_glossary.csv")
    service = make_service()
    doc = service.open_document(path)
    specs = service.row_specs(doc)
    state = ViewState(query="KAI")
    assert [s.source_idx for s in service.visible(doc, state, specs)] == [1]
    state = ViewState(filters={"type": frozenset({"character"})})
    assert [s.source_idx for s in service.visible(doc, state, specs)] == [0, 1]
    values = service.column_values(doc, "type", specs)
    assert [v for v, _n in values] == gd.column_filter_values([gd.editor_display_value(s.entry, "type") for s in specs])
    state = ViewState(sort_field="translated_name")
    assert [s.entry["translated_name"] for s in service.visible(doc, state, specs)] == ["Kai", "Luna", "Mana", "Sword"]
    state = ViewState(used_rows=frozenset({2, 3}), query="a")
    assert [s.source_idx for s in service.visible(doc, state, specs)] == [2]
    assert service.stats_text(doc).startswith("Total entries: 4")
    rows, occurrences = service.preview_replace(doc, "a", specs)
    assert rows == 3 and occurrences >= 3  # Luna / Mana / Sword (+ the section / type columns)


# ==========================================================================
# Editor (fake Flet session)
# ==========================================================================


@needs_flet
@needs_gd
def test_windowed_editor_over_a_10k_entry_glossary(iso):
    import flet as ft

    from glossarion_mobile.ui.glossary.windowed_list import STEP, WINDOW_ROWS

    entries = [{"type": "character" if i % 3 else "terms", "raw_name": f"이름{i}", "translated_name": f"Name {i}",
                "gender": ("male" if i % 2 else "female") if i % 3 else ""} for i in range(10_000)]
    path = write_token_glossary(iso.output / "Big" / "Big_glossary.csv", entries)
    service = make_service({"update_html_on_save": False})

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        started = time.perf_counter()
        ctx, screen, editor = await _open_editor(page, service, path)
        built = time.perf_counter() - started
        wl = editor.windowed
        assert doc_count(editor.doc) == 10_000 and len(editor.visible) == 10_000
        assert wl.mounted_count == STEP and wl.selector.visible
        assert wl.window_text.value == "Rows 1–1,500 of 10,000"
        assert editor.stats.value.startswith("Total entries: 10000")
        started = time.perf_counter()
        for _ in range(3):
            wl.load_more()
        page.update()
        assert wl.mounted_count == 4 * STEP
        target = editor.visible[9876].key
        assert await editor.jump_to(target)
        page.update()
        jumped = time.perf_counter() - started
        assert wl.window_start == 9000 and wl.rendered > 9876 and wl.jumps[-1] == target
        assert wl.mounted_count <= WINDOW_ROWS and wl.window_text.value == "Rows 9,001–10,000 of 10,000"
        assert editor.current_key == target and wl.controls[target] is not None
        wl.set_window(1500)
        assert wl.window_text.value == "Rows 1,501–3,000 of 10,000" and wl.mounted_count == STEP
        # search (debounced; filtered off the UI loop) narrows below one window (no selector)
        editor.search.value = "name 9"
        editor._on_search()
        first_search = editor._search_task
        editor.search.value = "name 99"
        editor._on_search()  # the next keystroke restarts the wait: one search runs, with the last text
        await asyncio.wait_for(editor._search_task, 10)
        page.update()
        assert first_search.cancelled() or first_search.done()
        assert len(editor.visible) == 111 and not wl.selector.visible and wl.mounted_count == 111
        assert "Search: 111/10000 shown" in editor.stats.value
        editor.clear_filters()
        types_ = dict(service.column_values(editor.doc, "type", editor.specs))
        editor.set_column_filter("type", frozenset({"character"}))
        assert len(editor.visible) == types_["character"] and editor.filter_row.visible
        editor.set_column_filter("type", None)
        assert len(editor.visible) == 10_000 and not editor.filter_row.visible
        # long-press selection over the window: entering it rebuilds the mounted rows once (the swipe
        # wrapper goes); a tap inside it changes that one row in place (UI_SPEC §7.3)
        first, second = editor.visible[0], editor.visible[1]
        editor.on_row_long_press(first)
        page.update()
        mounted = dict(wl.controls)
        row = wl.controls[second.key]
        started = time.perf_counter()
        editor.on_row_tap(second)
        toggled = time.perf_counter() - started
        assert second.source_idx in editor.selected and len(editor.selected) == 2
        assert all(wl.controls[key] is control for key, control in mounted.items())  # nothing rebuilt
        assert row.bgcolor == ft.Colors.SECONDARY_CONTAINER and row.data["check"].icon == ft.Icons.CHECK_CIRCLE
        assert editor.selection_bar.control.visible and editor.bulk_bar.control.visible
        editor.on_row_tap(second)
        assert second.source_idx not in editor.selected and row.bgcolor != ft.Colors.SECONDARY_CONTAINER
        assert row.data["check"].icon == ft.Icons.RADIO_BUTTON_UNCHECKED and wl.controls[second.key] is row
        editor.on_row_tap(first)  # the last selected row: selection mode ends, the rows get the swipe wrapper
        assert not editor.selecting and wl.controls[first.key] is not mounted[first.key]
        assert isinstance(wl.controls[first.key], ft.Dismissible)
        assert toggled < 1.0, toggled
        editor.on_row_long_press(editor.visible[0])
        editor.select_all()
        assert editor.selecting and len(editor.selected) == 10_000 and editor.bulk_bar.control.visible
        editor.exit_selection()
        assert not editor.selecting and not editor.selected
        assert built < 15 and jumped < 15, (built, jumped)
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_entry_sheet_save_updates_outputs_byte_identical_to_the_shared_document(iso):
    from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet

    path = book_glossary(iso.output / "Book")
    ref_path = book_glossary(iso.root / "ref" / "Book")
    original = Path(path).read_bytes()
    service = make_service({})  # update_html_on_save defaults on, like the desktop

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx, screen, editor = await _open_editor(page, service, path)
        spec = editor.specs[0]
        sheet = editor.open_entry(spec)
        assert isinstance(sheet, EntrySheet) and sheet.inputs["translated_name"].value == "Luna"
        assert "_section" not in sheet.inputs and sheet.inputs["gender"].value == "Female"
        sheet.inputs["translated_name"].value = "Lunaria"
        sheet.save()
        page.update()
        assert editor.doc.dirty and editor.save_button.badge is not None
        assert "edited" in _texts(editor.windowed.controls[spec.key]) and "→ Lunaria" in _texts(
            editor.windowed.controls[spec.key])
        assert not editor.undo_button.disabled
        # declining the desktop "Update output files" question saves nothing
        ctx.extras["answers"] = [False]
        assert await editor.save() is None
        kind, title, body = ctx.extras["asked"][-1]
        assert title == "Update output files" and body.endswith("Luna -> Lunaria")
        assert Path(path).read_bytes() == original
        ctx.extras["answers"] = [True]
        report = await editor.save()
        assert report == {"saved": True, "files_updated": 1, "replacements": 1}
        assert ctx.notes[-1] == "Glossary saved successfully · 1 output file(s) updated"
        assert (iso.output / "Book" / "response_001_ch1.html").read_text(encoding="utf-8") == "<p>Lunaria met Kai.</p>"
        assert not editor.doc.dirty and editor.save_button.badge is None
        # ＋ Entry: a new row (undoable), saved like an edit
        editor.add_entry({"type": "terms", "raw_name": "성", "translated_name": "Castle"})
        assert doc_count(editor.doc) == 5 and editor.doc.dirty
        assert await editor.undo() == "glossary" and doc_count(editor.doc) == 4

        reference = gd.GlossaryDocument({})
        reference.load(ref_path)
        reference.edit_cell(0, "translated_name", "Lunaria")
        reference.save_edits(update_output_files=True)
        assert Path(path).read_bytes() == Path(ref_path).read_bytes()
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_find_replace_preview_find_next_and_the_output_file_fallback(iso):
    path = book_glossary(iso.output / "Book")
    html = iso.output / "Book" / "response_001_ch1.html"
    service = make_service({"update_html_on_save": False})

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx, screen, editor = await _open_editor(page, service, path)
        sheet = editor.open_find_replace()
        sheet.find_field.value = "Kai"
        assert sheet.update_preview() == (1, 1) and sheet.preview.value == "1 row · 1 match"
        assert await sheet.find_next() == 1 and sheet.status.value == "Found at row 2"
        assert editor.current_key == editor.visible[1].key and editor.windowed.jumps[-1] == editor.visible[1].key
        sheet.replace_field.value = "Kyle"
        assert sheet.replace_current() == 1 and sheet.status.value == "Replaced 1 occurrence(s) in current row"
        assert editor.doc.current_glossary_data[1]["translated_name"] == "Kyle" and editor.doc.dirty
        sheet.find_field.value = "Mana"
        sheet.replace_field.value = "Manna"
        assert await sheet.replace_all() == 1
        assert sheet.status.value == "Replaced 1 occurrence(s) across all entries."
        # nothing in the glossary: "No glossary match" → the output files directly (an undoable step)
        sheet.find_field.value = "met"
        sheet.replace_field.value = "greeted"
        assert sheet.update_preview() == (0, 0) and sheet.preview.value == "No matches in the glossary"
        ctx.extras["answers"] = [True]
        assert await sheet.replace_all() == 0
        assert ctx.extras["asked"][-1][1] == gd.NO_GLOSSARY_MATCH_TITLE
        assert ctx.extras["asked"][-1][2] == f"{gd.NO_GLOSSARY_MATCH_TEXT}\n\nmet -> greeted"
        assert sheet.status.value == "Updated 1 files directly (1 replacements)."
        assert html.read_text(encoding="utf-8") == "<p>Luna greeted Kai.</p>"
        assert await editor.undo() == "html" and html.read_text(encoding="utf-8") == "<p>Luna met Kai.</p>"
        assert await editor.undo() == "glossary"  # restores (and saves, like the desktop) the rows before Replace All
        assert editor.doc.current_glossary_data[2]["translated_name"] == "Mana"
        assert await editor.undo(redo=True) == "glossary"
        assert editor.doc.current_glossary_data[2]["translated_name"] == "Manna"
        sheet.close()
        assert editor.last_find == "met" and editor.last_replace == "greeted"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_delete_backups_restore_filter_and_trim_tools(iso):
    path = write_token_glossary(iso.output / "Book" / "Book_glossary.csv")
    service = make_service({"update_html_on_save": False})

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx, screen, editor = await _open_editor(page, service, path)
        ctx.extras["answers"] = [True]
        report = await editor.delete_rows([editor.specs[3]])
        assert ctx.extras["asked"][-1] == ("ask", "Confirm Delete", "Delete 1 selected entries?")
        assert report.message == "Deleted 1 entries" and doc_count(editor.doc) == 3 and len(editor.specs) == 3
        assert ctx.notes[-1] == "Deleted 1 entries"
        backups = await editor.open_backups()
        assert backups.backups and backups.backups[0]["name"].startswith("Book_glossary_before_delete_1_")
        ctx.extras["answers"] = [True]
        restored = await editor.restore_backup(backups.backups[0]["path"])
        assert restored.ok and doc_count(editor.doc) == 4
        assert service.fake_files.calls[-1][2] == "before_restore"
        # Filter Entries: the shared filter_matcher choices from the sheet
        fsheet = editor.open_filter_entries()
        assert "term" in fsheet.type_checks
        fsheet.type_checks["term"].value = False
        choices = fsheet.conditions()
        assert choices["kept_types"]["term"] is False and choices["search_text"] == ""
        assert choices["gender_value"] == "all" and set(choices["type_limits"]) == set(fsheet.type_checks)
        preview = await fsheet.preview_filter()
        assert preview.message == "Filter matches: 2 entries (2 will be removed)"
        before = iso.root / "before_filter.csv"
        shutil.copyfile(path, before)
        report = await editor.run_tool("filter", choices)
        assert report.title == "Success" and report.message == "Filter applied!\n\nKept: 2 entries\nRemoved: 2 entries"
        reference = gd.GlossaryDocument({"update_html_on_save": False})
        reference.load(str(before))
        reference.apply_filter(**choices)
        assert Path(path).read_bytes() == before.read_bytes()
        # Trim: the shared preview text; About Format: the desktop box
        tsheet = editor.open_trim()
        tsheet.field.value = "1"
        assert tsheet.preview_changes() == gd.trim_preview_text(2, 1)
        info = editor.open_about_format()
        assert info.title == gd.DUPLICATE_DETECTION_INFO_TITLE
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_swipe_undo_reload_rederives_the_view_and_info_boxes_keep_edits_unsaved(iso):
    """U6 review: the swipe delete's Undo restores the row; every reload derives Hide unused and the
    column filters again (source indices shift); a box that saved nothing leaves the edits unsaved."""
    path = book_glossary(iso.output / "Book")  # Luna, Kai, Mana, Sword; the chapter says "Luna met Kai."
    service = make_service({"update_html_on_save": False})

    def names(rows):
        return [s.entry["translated_name"] for s in rows]

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx, screen, editor = await _open_editor(page, service, path)
        said = []
        ctx.notify = lambda message, action=None, on_action=None: said.append((message, action, on_action))
        assert await editor.toggle_hide_unused() and names(editor.visible) == ["Luna", "Kai"]
        # swipe Luna away: the file is re-read, Hide unused runs again on the new source indices
        report = await editor.swipe_delete(editor.visible[0])
        assert report.message == "Deleted 1 entries" and doc_count(editor.doc) == 3
        assert names(editor.visible) == ["Kai"] and names(editor.view_rows()) == ["Kai"]
        message, label, undo = said[-1]
        assert (message, label) == ("Deleted 1 entries", "Undo") and len(editor.doc._undo_stack) == 0
        assert await undo() is True
        assert doc_count(editor.doc) == 4 and "루나" in Path(path).read_text(encoding="utf-8")
        assert names(editor.visible) == ["Luna", "Kai"] and said[-1][0] == "Entry restored" and not editor.dirty
        # the Undo of a stale snapshot is refused: the glossary was edited after that delete
        await editor.swipe_delete(editor.visible[1])
        undo = said[-1][2]
        editor.save_entry(editor.specs[0], {"translated_name": "Lunaria"})
        assert editor.dirty and await undo() is False and doc_count(editor.doc) == 3
        assert said[-1][0].startswith("The glossary changed since the delete")
        await editor.reload(force=True)  # discard the edit
        # a column filter of a column the re-read file no longer has is dropped
        await editor.toggle_hide_unused()
        editor.set_column_filter("description", frozenset({"a witch"}))
        assert names(editor.visible) == ["Luna"]
        editor.state.filters["gone_column"] = frozenset({"x"})
        await editor.reload(force=True)
        assert "gone_column" not in editor.state.filters and names(editor.visible) == ["Luna"]
        screen.dispose()

        # a JSON list without empty fields: Clean Empty Fields answers "Info" and saves nothing
        listed = iso.output / "Other" / "Other_glossary.json"
        listed.parent.mkdir()
        listed.write_text(json.dumps([dict(e, gender=e.get("gender") or "male", description="d") for e in ENTRIES],
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        conn, session = _TB._fake_session("android")
        ctx, screen, editor = await _open_editor(session.page, service, str(listed))
        editor.save_entry(editor.specs[1], {"translated_name": "Kyle"})
        assert editor.dirty
        report = await editor.run_tool("clean")
        assert report.title == "Info" and editor.dirty and editor.save_button.badge is not None
        ctx.extras["answers"] = [False]  # "Keep editing"
        assert screen.handle_back() is True  # Back still asks about the unsaved edit
        await asyncio.sleep(0.05)
        assert ctx.extras["asked"][-1][1] == "Unsaved changes" and editor.dirty
        before = listed.read_bytes()
        report = await editor.run_tool("convert", str(iso.root / "export.csv"))
        assert report.title == "Success" and (iso.root / "export.csv").is_file()
        assert editor.dirty and listed.read_bytes() == before  # exported elsewhere: this file is still unsaved
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_dismissed_dialogs_answer_no_and_cancelled_switch_keeps_the_file(iso):
    """A ConfirmDialog / text dialog closed with the back gesture answers No / None (Save is not stuck), and
    "Unsaved changes" › Cancel leaves the screen on the open file."""
    from glossarion_mobile.ui.glossary.common import ask, prompt_text

    path = book_glossary(iso.output / "Book")
    other = write_token_glossary(iso.output / "Other" / "Other_glossary.csv")
    service = make_service({})  # update_html_on_save: the save asks first

    async def dismiss(ctx):
        for _ in range(50):
            dialog = ctx.extras.get("last_dialog")
            if dialog is not None:
                break
            await asyncio.sleep(0.01)
        ctx.extras.pop("last_dialog", None)
        control = getattr(dialog, "dialog", dialog)
        await control._trigger_event("dismiss", None)  # what the client sends after a back gesture

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx, screen, editor = await _open_editor(page, service, path)
        asking = asyncio.ensure_future(ask(ctx, title="Q", body="?"))
        await dismiss(ctx)
        assert await asyncio.wait_for(asking, 5) is False
        prompting = asyncio.ensure_future(prompt_text(ctx, title="Name", label="Name", value="x"))
        await dismiss(ctx)
        assert await asyncio.wait_for(prompting, 5) is None
        editor.save_entry(editor.specs[0], {"translated_name": "Lunaria"})
        saving = asyncio.ensure_future(editor.save())  # asks "Update output files"
        await dismiss(ctx)
        assert await asyncio.wait_for(saving, 5) is None and not editor._saving and editor.dirty
        # "Unsaved changes" › Cancel: title, gid and path stay the open file's
        gid, title = screen.gid, screen.title
        ctx.extras["answers"] = [False]
        assert await screen.switch_to(other) is False
        assert (screen.gid, screen.title, screen.path, editor.path) == (gid, title, path, path)
        ctx.extras["answers"] = [True]
        assert await screen.switch_to(other) is True
        assert screen.path == other and editor.path == other and screen.gid == service.gid_for(other)
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# Glossary mode locks
# ==========================================================================


@needs_rules
def test_mode_selector_runs_the_shared_lock_pass(iso):
    config = {"auto_glossary_mode": "balanced", "append_glossary": False}
    service = GlossaryService(config=config)
    changed = service.apply_mode_locks()
    forced = {k: v for k, (locked, v) in rules.glossary_mode_toggle_states("balanced").items() if locked}
    assert forced and all(config[k] == v for k, v in forced.items()) and changed["append_glossary"] is True
    assert service.apply_mode_locks() == {}  # idempotent
    service.set_mode("no_glossary")
    forced = {k: v for k, (locked, v) in rules.glossary_mode_toggle_states("no_glossary").items() if locked}
    assert config["auto_glossary_mode"] == "no_glossary" and all(config[k] == v for k, v in forced.items())
    assert config["append_glossary"] is False
    assert [m for m, _label in service.modes()] == list(rules.auto_glossary_modes())
    assert service.mode_label() == rules.glossary_mode_label("no_glossary")
    lock = rules.evaluate_locks(config, ["append_glossary"])["append_glossary"]
    assert lock.locked


@needs_flet
@needs_rules
@pytest.mark.skipif(not _has("settings_schema"), reason="settings_schema not importable")
def test_general_tab_shows_mode_lock_badges(iso, tmp_path):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsTab
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    if not schema.available:
        pytest.skip(f"settings schema unavailable: {schema.error}")
    config_path = tmp_path / "config.json"
    store = MobileConfigStore(config_path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk),
                                                                                        encoding="utf-8"))
    store.load()
    service = GlossaryService(config=store)

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        service.set_mode("no_glossary")
        assert store.get("auto_glossary_mode") == "no_glossary" and store.get("append_glossary") is False
        settings = SettingsContext(page=page, store=store, schema=schema)
        ctx = _ctx(page, service, settings=settings)
        tab = GlossarySettingsTab(ctx, "general")
        _mount(page, tab.build())
        tab.did_show()
        keys = tab.keys()
        assert keys[0] == "auto_glossary_mode" and "append_glossary" in keys
        tile = tab.page.tiles["append_glossary"]
        expected = schema.lock_reason(schema.spec("append_glossary"), store.snapshot())
        assert expected and tile.lock_reason == expected
        assert tab.page.tiles["fuzzy_auto_mapping"].lock_reason
        tab.dispose()
        store._saver.close()

    asyncio.run(scenario())


needs_profiles = pytest.mark.skipif(
    _core("prompt_profiles") is None or gd is None or not hasattr(gd, "GlossaryPromptProfiles"),
    reason="prompt_profiles / glossary_document.GlossaryPromptProfiles not importable")
_BALANCED_KEYS = ("manual_glossary_prompt3", "manual_glossary_prompt")


def _bucket_view(config: dict, bucket: str) -> dict:
    """The values one Balanced/Full or Minimal row owns in a config."""
    meta = gd.glossary_prompt_profile_meta(bucket)
    view = {name: (config.get(name) or {}).get(bucket, "<absent>") for name in (
        "glossary_prompt_profiles", "active_glossary_prompt_profiles", "glossary_prompt_profile_defaults")}
    for key in (meta["config_key"], meta.get("legacy_config_key")):
        if key:
            view[key] = config.get(key, "<absent>")
    return view


@needs_profiles
def test_prompt_profile_rows_are_the_shared_glossary_document_classes():
    """The settings tabs' profile rows are glossary_document's GlossaryPromptProfiles / RefinementPromptProfiles
    (the classes tests/test_glossary_document.py replays against the desktop controls): the same actions
    leave the same config, each row writes only its own entry of the dicts Balanced/Full and Minimal share."""
    import copy

    # the review probe: a stale Default text while the prompt was edited elsewhere (Settings › Glossary)
    config = {"glossary_prompt_profile_defaults": {"balanced_full": "OLD default"}, "manual_glossary_prompt3": "NEW"}
    service = GlossaryService(config=config)
    row = service.prompt_profiles("balanced_full", default_text="Default prompt")
    assert isinstance(row, gd.GlossaryPromptProfiles) and row.names == ["Default"]
    assert row.select("Default") == "NEW" and service.persist_prompt_profiles(row)
    assert config["manual_glossary_prompt3"] == "NEW" and config["glossary_prompt_profile_defaults"] == {
        "balanced_full": "NEW"}
    assert row.delete("Default") == ("Default Profile", "The Default glossary prompt profile cannot be deleted.")

    # differential: the service row (persisted after every action) vs the shared class on its own owner
    start = {"manual_glossary_prompt3": "Base prompt", "glossary_prompt_profiles": {"minimal": {"Keep": "m"}},
             "active_glossary_prompt_profiles": {"minimal": "Keep"}}
    config = copy.deepcopy(start)
    service = GlossaryService(config=config)
    row = service.prompt_profiles("balanced_full")
    reference_owner = types.SimpleNamespace(config=copy.deepcopy(start), manual_glossary_prompt="Base prompt")
    reference = gd.GlossaryPromptProfiles(reference_owner, "balanced_full", "Base prompt")
    steps = [("new",), ("stage", "Prompt A"), ("save", None, "Prompt A"), ("new",), ("save", "Renamed", "B text"),
             ("select", "Default"), ("stage", "Default edit"), ("select", "New Profile #1"),
             ("delete", None), ("select", "Gone"), ("delete", "Default")]
    for step in steps:
        boxes = []
        for target in (row, reference):
            name = step[1] if len(step) > 1 and step[1] is not None else target.selected_name()
            if step[0] == "new":
                boxes.append(target.new())
            elif step[0] == "stage":
                boxes.append(target.stage(target.selected_name(), step[1]))
            elif step[0] == "save":
                boxes.append(target.save(name, step[2]))
            elif step[0] == "select":
                boxes.append(target.select(name))
            elif step[0] == "delete":
                boxes.append(target.delete(name, confirm=lambda _n: True))
        assert service.persist_prompt_profiles(row)
        assert boxes[0] == boxes[1], step
        assert row.names == reference.names and row.text == reference.text, step
        assert _bucket_view(config, "balanced_full") == _bucket_view(reference_owner.config, "balanced_full"), step
        # the Minimal entries the Balanced/Full row never touches are kept
        assert config["glossary_prompt_profiles"]["minimal"] == {"Keep": "m"}, step
        assert config["active_glossary_prompt_profiles"]["minimal"] == "Keep", step

    # two rows open at once (two tabs): each persists only its own bucket
    config = {}
    service = GlossaryService(config=config)
    balanced, minimal = service.prompt_profiles("balanced_full"), service.prompt_profiles("minimal")
    minimal.new()
    balanced.new()
    assert service.persist_prompt_profiles(minimal) and service.persist_prompt_profiles(balanced)
    assert config["glossary_prompt_profiles"] == {"minimal": {"New Profile #1": ""},
                                                  "balanced_full": {"New Profile #1": ""}}
    assert config["active_glossary_prompt_profiles"] == {"minimal": "New Profile #1", "balanced_full": "New Profile #1"}

    # Refinement: the system + user pair row
    config = {"glossary_refinement_system_prompt": "SYS", "glossary_refinement_user_prompt": "USR"}
    service = GlossaryService(config=config)
    pair = service.prompt_profiles("refinement")
    assert isinstance(pair, gd.RefinementPromptProfiles) and pair.pair == {"system": "SYS", "user": "USR"}
    assert pair.new() is None and service.persist_prompt_profiles(pair)
    assert config["active_glossary_refinement_prompt_profile"] == "New Profile #1"
    assert config["glossary_refinement_prompt_profiles"] == {"New Profile #1": {"system": "", "user": ""}}
    assert pair.delete("Default") == ("Default Profile", "The Default refinement prompt profile cannot be deleted.")
    # persist failing (config.json not writable): the shared row rolls the action back
    broken = types.SimpleNamespace(snapshot=lambda: dict(config), get=config.get, set_many=lambda updates: 1 / 0)
    service = GlossaryService(config=broken)
    row = service.prompt_profiles("balanced_full")
    assert row.save("Mine", "text") == ("Save Failed", "Could not save the glossary prompt profile. Please try again.")
    assert row.names == ["Default"]


@needs_flet
@needs_profiles
def test_profile_bar_drives_the_shared_row(iso):
    from glossarion_mobile.ui.glossary.settings_tabs import ProfileBar

    config = {"manual_glossary_prompt3": "Base prompt"}
    service = GlossaryService(config=config)

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service, settings=None)
        bar = ProfileBar(ctx, "balanced_full", prompt_keys=("manual_glossary_prompt3",))
        _mount(page, bar.control)
        assert bar.names == ["Default"] and bar.dropdown.value == "Default" and not bar.dropdown.disabled
        name = bar.new()
        assert name == "New Profile #1" and config["active_glossary_prompt_profiles"] == {"balanced_full": name}
        assert config["glossary_prompt_profiles"]["balanced_full"] == {name: ""}
        assert [o.key for o in bar.dropdown.options] == ["Default", name] and bar.dropdown.value == name
        # a prompt tile edit stages into the selected profile (_auto_save_glossary_prompt_profile)
        config["manual_glossary_prompt3"] = "Edited prompt"
        bar.on_prompt_changed()
        assert config["glossary_prompt_profiles"]["balanced_full"][name] == "Edited prompt"
        assert await bar.save("Mine") is None and ctx.notes[-1] == "Saved profile “Mine”"
        assert config["glossary_prompt_profiles"]["balanced_full"] == {"Mine": "Edited prompt"}
        bar.select("Default")
        assert config["manual_glossary_prompt3"] == "Base prompt" and config["active_glossary_prompt_profiles"] == {}
        ctx.extras["answers"] = []
        await bar.delete()  # Default: the desktop box, nothing asked
        assert ctx.notes[-1] == "The Default glossary prompt profile cannot be deleted."
        assert not ctx.extras.get("asked")
        bar.select("Mine")
        ctx.extras["answers"] = [False]
        await bar.delete()  # "No" keeps it
        assert ctx.extras["asked"][-1] == ("ask", "Delete Profile", "Delete glossary prompt profile 'Mine'?")
        assert "Mine" in config["glossary_prompt_profiles"]["balanced_full"]
        ctx.extras["answers"] = [True]
        await bar.delete()
        assert config["glossary_prompt_profiles"]["balanced_full"] == {} and bar.names == ["Default"]
        # hidden, then shown again: the row is opened again on the current config
        bar.detach()
        config["glossary_prompt_profiles"]["balanced_full"] = {"Elsewhere": "x"}
        bar.attach()
        assert bar.names == ["Default", "Elsewhere"]
        bar.detach()

    asyncio.run(scenario())


# ==========================================================================
# Job kinds and the glossary stop protocol
# ==========================================================================


class _UnifiedJobs:
    """JobService surface the Unified screen uses: ``busy`` is a property, like JobService.busy."""

    def __init__(self, busy=False):
        self._busy = busy
        self.specs, self.listeners, self.snaps = [], [], {}

    @property
    def busy(self):
        return self._busy

    def has_kind(self, kind):
        return kind == "unified_glossary"

    def submit(self, spec):
        from glossarion_mobile.services.jobs import JobSnapshot, JobState

        self.specs.append(spec)
        job_id = f"job{len(self.specs)}"
        self.snaps[job_id] = JobSnapshot(id=job_id, spec=spec, state=JobState.QUEUED, created=1.0)
        return job_id

    def snapshot(self, job_id=None):
        return self.snaps.get(job_id)

    def view(self):
        """The JobsView shape (the jobs submitted here wait in the queue)."""
        return types.SimpleNamespace(active=None, queue=tuple(s for s in self.snaps.values() if not s.is_terminal))

    def on_transition(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback) if callback in self.listeners else None

    def finish(self, job_id):
        import dataclasses

        from glossarion_mobile.services.jobs import JobState

        snap = dataclasses.replace(self.snaps[job_id], state=JobState.DONE, started=2.0, finished=3.0)
        self.snaps[job_id] = snap
        for callback in list(self.listeners):
            callback(snap, JobState.RUNNING)


@needs_flet
def test_unified_rebuild_warns_while_a_run_is_busy_and_is_disabled_until_it_ends(iso):
    from glossarion_mobile.ui.glossary.unified import BUSY_WARNING, UnifiedGlossaryScreen

    jobs = _UnifiedJobs(busy=True)
    service = make_service({}, jobs=jobs)

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service, jobs=jobs)
        screen = UnifiedGlossaryScreen(None, ctx)
        _mount(page, screen.get_body())
        screen.did_show()
        assert not screen.rebuild_button.disabled
        job_id = await screen.rebuild()
        assert job_id == "job1" and ctx.notes[0] == BUSY_WARNING and jobs.specs[0].kind == "unified_glossary"
        assert screen.rebuild_button.disabled and screen.status.value.startswith("(Queued")
        assert await screen.rebuild() == "job1" and len(jobs.specs) == 1  # a second tap queues nothing
        jobs.finish(job_id)
        await asyncio.sleep(0.05)
        assert not screen.rebuild_button.disabled and "finished" in screen.status.value
        jobs._busy = False
        assert await screen.rebuild() == "job2" and ctx.notes[-1].startswith("📚 Unified glossary: Rebuild Now")
        assert BUSY_WARNING not in ctx.notes[1:]
        screen.dispose()

    asyncio.run(scenario())


class _JobCtx:
    def __init__(self, params, inputs=(), owner=None, config=None):
        self.params = dict(params)
        self.inputs = tuple(inputs)
        self.owner = owner if owner is not None else types.SimpleNamespace(config=dict(config or {}))
        self.config = dict(config or {})
        self.logs, self.phases, self.out_dirs = [], [], []

    def log(self, text):
        self.logs.append(str(text))

    def phase(self, label):
        self.phases.append(label)

    def set_output_dir(self, path):
        self.out_dirs.append(path)

    def stop_requested(self):
        return False


def test_glossary_job_kinds_are_registered():
    from glossarion_mobile import job_kinds
    from glossarion_mobile.job_kinds import glossary as gk
    from glossarion_mobile.services.jobs import JobKind

    for kind in ("extract_glossary", "glossary_refine", "unified_glossary", "parallel_pair"):
        assert job_kinds.KIND_MODULES[kind] == "glossary" and JobKind(kind).value == kind
        info = job_kinds.get_kind(kind)
        assert info.stop_kind == "glossary" and callable(info.run)
    assert job_kinds.get_kind("unified_glossary").resumable is False
    for kind in ("glossary_refine", "unified_glossary", "parallel_pair"):
        reason = gk.kind_ready(kind)
        assert reason is None or "not available" in reason


def test_specs_and_refine_gating(iso, monkeypatch):
    jobs = types.SimpleNamespace(kinds=set(), has_kind=lambda kind: kind in jobs.kinds)
    fake_gpc = types.SimpleNamespace()
    service = GlossaryService(config={}, jobs=jobs, core=SharedCore({"glossary_progress_core": fake_gpc}))
    images = [str(iso.root / "pages" / f"{i}.png") for i in range(3)]
    spec = service.extract_spec(images)
    assert spec.kind == "extract_glossary" and spec.title == "pages" and len(spec.inputs) == 3
    spec = service.extract_spec([str(iso.root / "Book.epub")], force_balanced_request_merging=True)
    assert spec.title == "Book" and spec.params == {"force_balanced_request_merging": True}
    with pytest.raises(ValueError):
        service.extract_spec([])
    assert "not available" in service.refine_supported()
    jobs.kinds.add("glossary_refine")
    assert "run_manual_glossary_refinement" in service.refine_supported()
    fake_gpc.run_manual_glossary_refinement = lambda *a, **k: None
    assert service.refine_supported() is None
    with pytest.raises(ValueError):
        service.refine_spec(glossary_path="g.csv", progress_path=None, source_path=None, selected_types=[])
    spec = service.refine_spec(glossary_path="g.csv", progress_path="p.json", source_path="b.epub",
                               selected_types=["character"], target_chunk_count=4)
    assert spec.kind == "glossary_refine" and spec.params["target_chunk_count"] == 4
    assert spec.params["selected_types"] == ["character"] and spec.params["progress_path"] == "p.json"
    assert service.unified_spec().params == {"shared_dir": os.path.join(str(iso.output), "Glossary")}
    with pytest.raises(ValueError):
        service.pair_spec({"raw_path": "a.epub"})


def test_refine_and_unified_adapters_call_the_shared_functions(iso, monkeypatch):
    from glossarion_mobile.job_kinds import glossary as gk
    from glossarion_mobile.services.jobs import JobError

    glossary = iso.output / "Book" / "Book_glossary.csv"
    glossary.parent.mkdir()
    glossary.write_text("Glossary Columns: raw_name, translated_name\n", encoding="utf-8")
    calls = []

    def plan(owner, glossary_path, progress_path, source_path, selected_types, target_chunk_count=None, log=None):
        calls.append(("plan", os.path.basename(glossary_path), progress_path, selected_types, target_chunk_count))
        return types.SimpleNamespace(selected_types=list(selected_types)), ["chunk-1"]

    def run(owner, glossary_path, progress_path, options, plan_):
        calls.append(("run", os.path.basename(glossary_path), options.selected_types, plan_))

    monkeypatch.setitem(sys.modules, "glossary_progress_core",
                        types.SimpleNamespace(plan_manual_glossary_refinement=plan))
    ctx = _JobCtx({"glossary_path": str(glossary), "selected_types": ["character"]})
    with pytest.raises(JobError):  # no shared runner yet
        gk.run_refine(ctx)
    monkeypatch.setitem(sys.modules, "glossary_progress_core",
                        types.SimpleNamespace(plan_manual_glossary_refinement=plan, run_manual_glossary_refinement=run))
    result = gk.run_refine(ctx)
    assert calls == [("plan", "Book_glossary.csv", None, ["character"], None),
                     ("run", "Book_glossary.csv", ["character"], ["chunk-1"])]
    assert result["outputs"] == [str(glossary)] and ctx.out_dirs == [str(glossary.parent)]
    with pytest.raises(JobError):
        gk.run_refine(_JobCtx({"glossary_path": str(glossary), "selected_types": []}))

    shared = iso.output / "Glossary"
    unified_csv = shared / "Unified Glossary" / "english" / "glossary_unified.csv"
    unified_csv.parent.mkdir(parents=True)
    unified_csv.write_text("x", encoding="utf-8")
    rebuilt = []
    fake_ug = types.SimpleNamespace(
        rebuild_now=lambda shared_dir=None, settings=None, log=print: rebuilt.append((shared_dir, settings)) or True,
        unified_root=lambda root: os.path.join(root, "Unified Glossary"),
        unified_paths=lambda root, key: (None, None, os.path.join(root, "Unified Glossary", key,
                                                                   "glossary_unified.csv")))
    monkeypatch.setitem(sys.modules, "unified_glossary", fake_ug)
    config = {"output_language": "Korean", "unified_glossary_combine_all_languages": True}
    ctx = _JobCtx({"shared_dir": str(shared)}, config=config)
    result = gk.run_unified(ctx)
    builder = getattr(gd, "unified_rebuild_settings", None) if gd is not None else None
    expected = builder(config, str(shared)) if callable(builder) else {
        "OUTPUT_LANGUAGE": "Korean", "UNIFIED_GLOSSARY_COMBINE_ALL_LANGUAGES": "1",
        "UNIFIED_GLOSSARY_EXCLUDE_GENDER_ENTRIES": "1", "GLOSSARY_SHARED_DIR": str(shared)}
    assert rebuilt == [(str(shared), expected)] and result["outputs"] == [str(unified_csv)]


def test_glossary_jobs_take_the_glossary_stop_protocol(monkeypatch):
    from glossarion_mobile.services.jobs import JobBackend

    calls = []
    hooks = []

    def request_glossary_stop(*, graceful, set_stop_requested, log, glossary_stop_flag=None, get_run_id=None):
        calls.append(("glossary", graceful))
        hooks.append((glossary_stop_flag, get_run_id))

    fake = types.SimpleNamespace(
        request_glossary_stop=request_glossary_stop,
        request_stop=lambda *a, **k: calls.append(("translation",)),
    )
    monkeypatch.setitem(sys.modules, "stop_control", fake)
    flags = []
    monkeypatch.setitem(sys.modules, "extract_glossary_from_epub",
                        types.SimpleNamespace(set_stop_flag=flags.append))
    owner = types.SimpleNamespace(graceful_stop_active=True, _last_stop_was_graceful=True,
                                  _glossary_run_id="glossary-abc")
    logs = []
    backend = JobBackend()
    backend.request_stop(graceful=True, wait_for_chunks=False, force=False, set_stop_requested=lambda: None,
                         log=logs.append, kind="glossary", owner=owner)
    backend.request_stop(graceful=True, wait_for_chunks=False, force=True, set_stop_requested=lambda: None,
                         log=logs.append, kind="glossary", owner=owner)
    assert calls == [("glossary", True), ("glossary", False)]
    assert owner.graceful_stop_active is False and logs == ["⚡ Double-click detected — forcing immediate stop!"]
    # the desktop's glossary_stop_flag (the loaded extractor's setter) and run-id guard
    stop_flag, get_run_id = hooks[-1]
    stop_flag(True)
    assert flags == [True] and get_run_id() == "glossary-abc"


# ==========================================================================
# The real shared cores behind the CONTRACT (Integrate U6)
# ==========================================================================


def test_contract_names_resolve_in_the_shared_modules():
    """Every CONTRACT operation names a real function (refinement: not shared yet, stays disabled)."""
    from glossarion_mobile.services.glossary import CONTRACT

    if gf is None or pec is None:
        pytest.skip("glossary_files / parallel_epub_core not importable")
    missing = []
    for op, (module, names) in CONTRACT.items():
        if op.startswith("refine_"):
            continue
        target = importlib.import_module(module)
        if not callable(getattr(target, names[0], None)):
            missing.append(f"{op}: {module}.{names[0]}")
    assert not missing, missing


@needs_gd
def test_delete_restore_and_backup_through_the_real_glossary_files(iso, monkeypatch):
    """The Library's delete / restore glossary files and the editor's backup run the desktop's
    glossary_files functions (no fakes): files move into Backups/<stamp>/ and come back."""
    if gf is None:
        pytest.skip("glossary_files not importable")
    service = GlossaryService(config={"glossary_auto_backup": True, "glossary_max_backups": 2})
    book_dir = iso.output / "Glossary" / "Book"
    glossary = write_token_glossary(book_dir / "Book_glossary.csv")
    epub = str(iso.root / "Book.epub")
    plan = service.delete_plan([epub])
    assert [(b, os.path.basename(p)) for b, p in plan] == [("Book", "Book_glossary.csv")]
    assert service.delete_prompt(plan) == "Delete the following files?\n\n[Book]\n  Book_glossary.csv"
    assert service.delete_files(plan) == ["Book/Book_glossary.csv"] and not os.path.exists(glossary)
    backup_dir, files = service.latest_backup([epub])
    assert files == ["Book_glossary.csv"] and os.path.dirname(backup_dir) == str(book_dir / "Backups")
    assert service.restore_files(backup_dir, files) == ["Book_glossary.csv"] and os.path.isfile(glossary)
    doc = gd.GlossaryDocument.open(glossary, {})
    stamps = iter(f"20260101_00000{i}" for i in range(9))
    monkeypatch.setattr(gf.time, "strftime", lambda _fmt: next(stamps))
    for _ in range(3):
        assert service.backup_callback(doc, "delete") is True
    snapshots = [n for n in os.listdir(book_dir / "Backups") if n.endswith(".json")]
    assert len(snapshots) == 2  # glossary_max_backups prunes the oldest


@needs_gd
def test_parallel_pair_mapping_runs_on_the_shared_core():
    """PairMapping keeps only the table cells; offset, unmap, restore, status and the Accept checks are
    parallel_epub_core's (the desktop dialog's own functions)."""
    if pec is None:
        pytest.skip("parallel_epub_core not importable")
    from glossarion_mobile.ui.glossary.parallel_pair import PairMapping

    raw = [{"filename": f"ch{i}.xhtml", "text": f"raw {i}"} for i in range(1, 6)]
    translated = [{"filename": f"t{i}.xhtml", "text": f"tr {i}"} for i in range(1, 7)]
    auto = pec.auto_map_epub_chapters(raw, translated, enable_auto_offset=True)
    mapping = PairMapping(raw, translated, auto, core=pec)
    selected = mapping.selected()
    assert mapping.status() == pec.parallel_epub_mapping_status(selected, 0, auto, 5, 6)
    mapping.apply_offset(1)
    rows = pec.offset_parallel_epub_mapping(auto, 1, translated, None)
    assert list(zip(mapping.assign, mapping.strategy)) == rows
    assert mapping.set_unmapped([0, 0, 9]) == 1 and mapping.strategy[0] == "Manual — Unmapped"
    assert mapping.unpaired_warning() == pec.unpaired_warning_text(mapping.selected(), 5, 6)
    restored, _skipped = pec.restore_parallel_epub_pairs(raw, translated, [
        {"raw_filename": "ch2.xhtml", "translated_filename": "t3.xhtml"}])
    mapping.restore(restored)
    assert mapping.selected() == [{"raw_index": 1, "translated_index": 2}] and mapping.offset == 0
    assert mapping.pairs() == pec.build_parallel_epub_pairs(mapping.selected(), raw, translated)
    mapping.set_row(3, 2)
    assert mapping.duplicate_count() == 1
    problem = pec.validate_parallel_epub_pair(loading=False, raw_path="a.epub", translated_path="b.epub",
                                              raw_chapters=raw, translated_chapters=translated,
                                              wrapper_prompt="{raw_text}{translated_text}", system_prompt="x",
                                              mapping=mapping.selected())
    assert problem == ("warning", "Duplicate Mapping", "Each translated HTML file can only be assigned once.")


@needs_flet
def test_parallel_pair_screen_loads_its_profiles_off_the_ui_loop(iso):
    """The built-in pair prompt imports the glossary extractor (seconds cold): the screen reads the profiles
    on the io pool after it opens, with the dropdown disabled until then."""
    import threading

    from glossarion_mobile.ui.glossary.parallel_pair import DEFAULT_PROFILE, ParallelPairScreen

    service = make_service({})
    calls = []

    def pair_profiles():
        calls.append(threading.get_ident())
        return {"Mine": "my prompt"}, "Mine"

    service.pair_profiles = pair_profiles
    service.default_pair_system_prompt = lambda: "built-in"

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service)
        screen = ParallelPairScreen(None, ctx)
        assert not calls and not screen.profiles_ready  # the screen factory (UI loop) reads nothing
        _mount(page, screen.get_body())
        assert screen.profile_dropdown.disabled
        assert screen.save_profile() is None and ctx.notes[-1] == "Loading profiles…"  # never persists {} early
        screen.did_show()
        await screen.load_profiles()
        assert calls and calls[0] != threading.get_ident()
        assert screen.profiles == {"Mine": "my prompt", DEFAULT_PROFILE: "built-in"} and screen.profile == "Mine"
        assert not screen.profile_dropdown.disabled and screen.profile_dropdown.value == "Mine"
        assert screen.system_prompt.value == "my prompt" and len(calls) == 1
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# Feature wiring
# ==========================================================================


class _FakeShell:
    def __init__(self):
        self.fallback, self.sheets, self.overlays = [], [], []
        self.tablet = False
        self.top_screen = None

    def screen_factory(self, match):
        self.fallback.append(match.name)
        return "fallback"

    def show_sheet(self, match):
        self.sheets.append(match.name)

    def push_overlay(self, view):
        self.overlays.append(view)


@needs_flet
def test_feature_routes_hooks_continue_bridge_and_glossary_files(iso):
    from glossarion_mobile.ui.glossary.feature import IMPLEMENTED_ROUTES, GlossaryFeature
    from glossarion_mobile.ui.glossary.home import GlossariesScreen
    from glossarion_mobile.ui.router import ROUTES_BY_NAME, parse_route

    assert all(name in ROUTES_BY_NAME for name in IMPLEMENTED_ROUTES)
    service = make_service({})

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        notes = []
        app = types.SimpleNamespace(page=page, shell=_FakeShell(), library=types.SimpleNamespace(), chat_view=None,
                                    dispatcher=types.SimpleNamespace(loop=asyncio.get_running_loop(), bound=False),
                                    notify=lambda message, *a: notes.append(message))
        feature = await GlossaryFeature.install(app, service=service)
        assert app.glossary is service and app.glossary_feature is feature and app.library.glossary_hooks is feature
        assert service.ask_continue == feature.ask_continue_blocking
        assert isinstance(app.shell.screen_factory(parse_route("/glossary")), GlossariesScreen)
        assert app.shell.screen_factory(parse_route("/settings")) == "fallback" and app.shell.fallback == ["settings"]
        # the blocking "Continue anyway?" bridge: asked on the UI loop from an io thread
        feature.extras["answers"] = [True]
        assert await asyncio.to_thread(feature.ask_continue_blocking, "Backup Failed", "Failed to create backup: x")
        assert feature.extras["asked"][-1] == ("ask", "Backup Failed", "Failed to create backup: x")
        assert feature.ask_continue_blocking("Backup Failed", "on the loop") is False  # never blocks the loop
        # Library selection bar / Book page: delete / restore glossary files with the desktop texts
        epub = str(iso.library / "Raw" / "Book.epub")
        glossary = str(iso.output / "Glossary" / "Book" / "Book_glossary.csv")
        service.fake_files.plan = [("Book", glossary)]
        feature.extras["answers"] = [True]
        assert await feature.delete_glossary_files([epub]) == ["Book/Book_glossary.csv"]
        assert feature.extras["asked"][-1] == ("ask", "Delete Glossary",
                                               "Delete the following files?\n\n[Book]\n  Book_glossary.csv")
        assert notes[-1] == "🗑️ Deleted (1 files backed up)"
        service.fake_files.plan = []
        feature.extras["answers"] = [True]
        assert await feature.delete_glossary_files([epub]) == []
        assert feature.extras["asked"][-1] == ("ask", "Nothing to Delete", "No glossary files found for: Book")
        backup_dir = str(iso.output / "Glossary" / "Book" / "Backups" / "20261006_101010")
        service.fake_files.backup = (backup_dir, ["Book_glossary.csv"])
        feature.extras["answers"] = [False]
        assert await feature.restore_glossary_backup([epub]) is None
        feature.extras["answers"] = [True]
        assert await feature.restore_glossary_backup([epub]) == ["Book_glossary.csv"]
        assert feature.extras["asked"][-1] == ("ask", "Restore Glossary",
                                               "Restore from backup (20261006_101010)?\n\nBook_glossary.csv")
        assert service.fake_files.calls[-1] == ("restore", "20261006_101010", ["Book_glossary.csv"])

    asyncio.run(scenario())


@needs_flet
def test_chat_and_reader_hand_offs_open_the_glossary_editor(iso):
    """Integrate (U6): the chat approval card's "Open in table editor", a chat response's / the Reader's
    "Add to glossary" open the Glossary Manager on the file (a new entry with the raw name filled in)."""
    from glossarion_mobile.ui.chat.cards import ATTACHMENT_ACTION_REASONS, GlossaryEditorView
    from glossarion_mobile.ui.glossary.feature import GlossaryFeature

    service = make_service({})
    glossary = book_glossary(iso.output / "Glossary" / "Book")

    async def scenario():
        conn, session = _TB._fake_session("android")
        went = []
        chat_view = types.SimpleNamespace(_on_tool=lambda tool_id: None, composer=types.SimpleNamespace(
            set_plus_open=lambda _open: None))
        app = types.SimpleNamespace(page=session.page, shell=_FakeShell(), library=None, chat_view=chat_view,
                                    dispatcher=types.SimpleNamespace(loop=asyncio.get_running_loop(), bound=False),
                                    navigate_to=lambda name, params=None, query=None: went.append((name, params)),
                                    notify=lambda *a: None)
        feature = await GlossaryFeature.install(app, service=service)
        gid = service.gid_for(glossary)
        # the chat's hooks
        assert chat_view.glossary_table_opener == feature.open_editor_for_path
        assert feature.open_editor_for_path(glossary) == gid and went[-1] == ("glossary.detail", {"gid": gid})
        opened = []
        editor = GlossaryEditorView(glossary, "", has_bom=False, on_close=lambda: opened.append("closed"),
                                    on_table=lambda: opened.append("table"))
        editor.open_table()
        assert opened == ["closed", "table"]
        # Add to glossary: the editor of that file, then its new-entry sheet with the raw name
        new_entries = []
        app.shell.top_screen = types.SimpleNamespace(gid=gid, editor=types.SimpleNamespace(
            doc=object(), open_new_entry=lambda raw_term="": new_entries.append(raw_term) or "sheet"))
        assert await feature.add_term("루나", glossary_path=glossary) == gid
        assert new_entries == ["루나"] and went[-1] == ("glossary.detail", {"gid": gid})
        # a book without a glossary file says so (nothing opens)
        before = list(went)
        assert await feature.add_term("x", book={"name": "Nobody", "output_folder": str(iso.output / "Nobody")}) is None
        assert went == before
        # the chat attachment card's QA scan stays disabled: chats are Direct Text workspaces
        assert "Direct Text" in ATTACHMENT_ACTION_REASONS["qa"]

    asyncio.run(scenario())


def test_library_selection_bar_hands_glossary_files_to_the_feature():
    from glossarion_mobile.ui.library.home import LibraryScreen

    seen = []

    class Hooks:
        service = types.SimpleNamespace(input_paths_for_books=lambda books: [f"/raw/{b['name']}.epub" for b in books])

        async def delete_glossary_files(self, inputs):
            seen.append(("delete", inputs))
            return ["Book/Book_glossary.csv"]

        async def restore_glossary_backup(self, inputs):
            seen.append(("restore", inputs))
            return []

    async def io(fn, *args):
        return fn(*args)

    exits = []
    stub = types.SimpleNamespace(service=types.SimpleNamespace(), ctx=types.SimpleNamespace(io=io, say=seen.append),
                                 exit_selection=lambda: exits.append(True))
    stub._glossary_hooks = lambda: LibraryScreen._glossary_hooks(stub)
    assert LibraryScreen._glossary_reason(stub)  # not installed: disabled with the reason
    assert asyncio.run(LibraryScreen.delete_glossary_files(stub, [{"name": "Book"}])) is None
    stub.service.glossary_hooks = Hooks()
    assert LibraryScreen._glossary_reason(stub) is None
    books = [{"name": "Book"}, {"name": "Other"}]
    assert asyncio.run(LibraryScreen.delete_glossary_files(stub, books)) == ["Book/Book_glossary.csv"]
    assert asyncio.run(LibraryScreen.restore_glossary_backup(stub, books)) == []
    assert seen[-2:] == [("delete", ["/raw/Book.epub", "/raw/Other.epub"]),
                         ("restore", ["/raw/Book.epub", "/raw/Other.epub"])]
    assert exits == [True, True]


@needs_gd
def test_load_as_manual_sets_the_desktop_keys(iso):
    path = write_token_glossary(iso.output / "Book" / "Book_glossary.csv")
    config = {"auto_glossary_mode": "balanced"}
    service = make_service(config)
    text, info = service.load_prompt(path, "balanced")
    assert text == "Load this glossary for translation?\n\nBook_glossary.csv" and "Current mode: Balanced" in info
    result = service.load_as_manual(path)
    assert config["manual_glossary_path"] == path and config["append_glossary"] is True
    assert result["copied_to"] is None and result["mode"] == "balanced"


# ==========================================================================
# U6 second review round
# ==========================================================================


def _remount(page, body):
    """Replace the mounted body (the previous visit's View left in an update of its own, as in the shell)."""
    page.views[0].controls.clear()
    page.update()
    _mount(page, body)


@needs_flet
@needs_gd
def test_use_as_manual_sheet_closed_without_a_choice_loads_nothing(iso):
    """With a chat open, "Use as manual glossary" asks where first; its Cancel row, Android back or an outside
    tap answer "nothing" (the call returns instead of waiting forever), and a row still goes on to the
    desktop "Load Glossary" question."""
    from glossarion_mobile.ui.glossary.feature import GlossaryFeature

    path = write_token_glossary(iso.output / "Book" / "Book_glossary.csv")
    config = {"auto_glossary_mode": "balanced"}
    service = make_service(config)

    async def scenario():
        conn, session = _TB._fake_session("android")
        chat_view = types.SimpleNamespace(bound=True, cid="5", _on_tool=lambda tool_id: None,
                                          composer=types.SimpleNamespace(set_plus_open=lambda _open: None))
        app = types.SimpleNamespace(page=session.page, shell=_FakeShell(), library=None, chat_view=chat_view,
                                    dispatcher=types.SimpleNamespace(loop=asyncio.get_running_loop(), bound=False),
                                    notify=lambda *a: None)
        feature = await GlossaryFeature.install(app, service=service)

        async def ask_where():
            count = len(feature.sheets)
            task = asyncio.ensure_future(feature.use_as_manual(path))
            for _ in range(200):
                if len(feature.sheets) > count and getattr(feature.sheets[-1].dialog, "open", False):
                    return task, feature.sheets[-1]
                assert not task.done(), task.result()
                await asyncio.sleep(0.01)
            raise AssertionError("the 'Use as manual glossary' sheet did not open")

        task, sheet = await ask_where()
        await sheet.dialog._trigger_event("dismiss", None)  # what the client sends after Android back
        assert await asyncio.wait_for(task, 5) is None
        task, sheet = await ask_where()
        sheet._on_cancel()  # the sheet's Cancel row
        assert await asyncio.wait_for(task, 5) is None
        assert "manual_glossary_path" not in config and "asked" not in feature.extras
        task, sheet = await ask_where()
        feature.extras["answers"] = [False]  # "Load Glossary" › Cancel
        sheet._on_select(None, sheet.item("For the next translation runs"))
        await sheet.dialog._trigger_event("dismiss", None)  # the dismiss that follows a picked row
        assert await asyncio.wait_for(task, 5) is None
        assert feature.extras["asked"][-1][1] == "Load Glossary" and "manual_glossary_path" not in config

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_add_to_glossary_over_a_tablet_full_screen_view_uses_the_entry_sheet(iso):
    """Tablet with the Reader (a full-screen View) on top: "Add to glossary" opens the new-entry sheet over it
    (UI_SPEC §3.11) - the editor would open in the main area behind the Reader - and Add writes the entry to
    the file with the editor's Save ("before_save" backup). Without a full-screen View the editor opens."""
    from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet
    from glossarion_mobile.ui.glossary.feature import GlossaryFeature

    glossary = write_token_glossary(iso.output / "Glossary" / "Book" / "Book_glossary.csv")
    service = make_service({})

    async def scenario():
        conn, session = _TB._fake_session("android")
        went, notes = [], []
        shell = _FakeShell()
        shell.tablet = True
        shell.stack = [types.SimpleNamespace(fullscreen=True, route="/reader/0123456789ab")]
        app = types.SimpleNamespace(page=session.page, shell=shell, library=None, chat_view=None,
                                    dispatcher=types.SimpleNamespace(loop=asyncio.get_running_loop(), bound=False),
                                    navigate_to=lambda name, params=None, query=None: went.append((name, params)),
                                    notify=lambda message, *a: notes.append(message))
        feature = await GlossaryFeature.install(app, service=service)
        gid = await feature.add_term("새단어", glossary_path=glossary)
        assert gid == service.gid_for(glossary) and went == []  # nothing pushed behind the Reader
        sheet = feature.sheets[-1]
        assert isinstance(sheet, EntrySheet) and sheet.new and sheet.dialog.open
        assert sheet.inputs["raw_name"].value == "새단어"
        sheet.inputs["translated_name"].value = "New Word"
        sheet.save()
        for _ in range(250):
            if notes:
                break
            await asyncio.sleep(0.02)
        assert notes == ["Added to Book_glossary.csv"]
        saved = service.open_document(glossary)
        names = [e.get("translated_name") for e in saved.current_glossary_data]
        assert sorted(names) == ["Kai", "Luna", "Mana", "New Word", "Sword"]  # its section keeps the type order
        assert ("backup", "Book_glossary.csv", "before_save") in service.fake_files.calls
        # no full-screen View on top (or a phone): the editor of that file opens with its new-entry sheet
        shell.stack[-1].fullscreen = False
        opened = []
        shell.top_screen = types.SimpleNamespace(gid=gid, editor=types.SimpleNamespace(
            doc=object(), open_new_entry=lambda raw_term="": opened.append(raw_term) or "sheet"))
        assert await feature.add_term("다른", glossary_path=glossary) == gid
        assert went == [("glossary.detail", {"gid": gid})] and opened == ["다른"]

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_switching_files_keeps_hide_unused_and_the_search(iso):
    """Like the desktop (Hide unused stays checked and is re-applied after every load), a file switch keeps
    Hide unused on; the search box's text keeps filtering the new file."""
    path = book_glossary(iso.output / "Book")  # the chapter says "Luna met Kai."
    other_dir = iso.output / "Other"
    other = write_token_glossary(other_dir / "Other_glossary.csv")
    (other_dir / "response_001_ch1.html").write_text("<p>Mana and Sword.</p>", encoding="utf-8")
    service = make_service({"update_html_on_save": False})

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx, screen, editor = await _open_editor(page, service, path)
        assert await editor.toggle_hide_unused()
        editor.search.value = "a"
        await editor.apply_search()
        assert [s.entry["translated_name"] for s in editor.visible] == ["Luna", "Kai"]
        assert await screen.switch_to(other)
        page.update()
        assert editor.hide_unused and editor.state.query == editor.search.value == "a"
        assert [s.entry["translated_name"] for s in editor.visible] == ["Mana"]  # used in Other and matching "a"
        assert {item.content: item.checked for item in editor.menu.items}["Hide unused entries"] is True
        editor.search.value = ""
        await editor.apply_search()
        assert [s.entry["translated_name"] for s in editor.visible] == ["Mana", "Sword"]
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_gd
def test_editor_reads_the_glossary_list_and_its_input_off_the_ui_loop(iso):
    """Opened directly (no Glossaries list yet), the editor reads the list for ◀ ▶ / file name ▾ and its
    Library input on the io pool - never while building or rendering - and resolves the input once."""
    import threading

    from glossarion_mobile.ui.glossary.glossary_view import GlossaryScreen

    path = book_glossary(iso.output / "Glossary" / "Book")
    other = write_token_glossary(iso.output / "Glossary" / "Other" / "Other_glossary.csv")
    raw = str(iso.library / "Raw" / "Book.epub")
    calls = []
    service = make_service({})
    real_list = service.list_glossaries

    def list_glossaries():
        calls.append(("list", threading.get_ident()))
        return real_list()

    def raw_source(book):
        calls.append(("raw_source", threading.get_ident()))
        return raw

    service.list_glossaries = list_glossaries
    service.library = types.SimpleNamespace(raw_source=raw_source, snapshot=types.SimpleNamespace(
        all_books=lambda: [{"name": "Book", "output_folder": str(iso.output / "Book")}]))

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        loop_thread = threading.get_ident()
        feature = types.SimpleNamespace(listing=[], open_extract_sheet=lambda source=None: calls.append(
            ("extract", source)))
        ctx = _ctx(page, service, feature=feature)
        screen = GlossaryScreen(None, ctx, path=path)
        _mount(page, screen.get_body())
        assert screen.sibling_files() == [] and not calls  # building the editor scanned nothing
        assert not screen.editor.prev_button.visible
        screen.did_show()
        await screen.listing_task
        for _ in range(250):
            if screen.editor.doc is not None:
                break
            await asyncio.sleep(0.02)
        assert screen.editor.doc is not None and screen.editor.doc.source_path == raw
        assert sorted(kind for kind, _thread in calls) == ["list", "raw_source"]
        assert all(thread != loop_thread for _kind, thread in calls)
        assert {os.path.normcase(f.path) for f in feature.listing} >= {os.path.normcase(path), os.path.normcase(other)}
        assert screen.editor.prev_button.visible and screen.editor.next_button.visible
        await screen.editor.reload(force=True)
        await screen.open_extract()
        assert [c[0] for c in calls].count("raw_source") == 1  # resolved once; Reload and Extract reuse it
        assert calls[-1] == ("extract", raw)
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_parallel_pair_from_library_offers_the_compiled_translation(iso):
    """"From Library…" lists the raw sources for the raw side and, for the translated side, a book's compiled
    EPUB (``compiled_outputs_blocking``, never the raw Library file it also lists) or a Completed-shelf EPUB;
    the lookup runs on the io pool."""
    import threading

    from glossarion_mobile.ui.glossary.parallel_pair import ParallelPairScreen

    raw = iso.library / "Raw" / "Novel.epub"
    compiled = iso.output / "Novel" / "Novel.epub"
    done = iso.library / "Completed" / "Done.epub"
    for epub in (raw, compiled, done):
        epub.parent.mkdir(parents=True, exist_ok=True)
        epub.write_bytes(b"PK")
    books = [{"name": "Novel", "path": str(raw), "raw_source_path": str(raw), "output_folder": str(compiled.parent),
              "type": "in_progress"},
             {"name": "Done", "path": str(done), "type": "completed"}]
    threads = []

    def compiled_outputs_blocking(book):
        threads.append(threading.get_ident())
        out = [(str(book["path"]), "epub")]  # like LibraryService: the Library file itself comes first
        if book.get("output_folder"):
            out.append((str(compiled), "epub"))
        return out

    library = types.SimpleNamespace(snapshot=types.SimpleNamespace(all_books=lambda: books),
                                    compiled_outputs_blocking=compiled_outputs_blocking)
    service = make_service({})
    service.pair_profiles = lambda: ({}, "")
    service.default_pair_system_prompt = lambda: "built-in"

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service, library=library)
        screen = ParallelPairScreen(None, ctx)
        _mount(page, screen.get_body())
        assert screen.library_epubs("raw") == [("Novel", str(raw))]
        assert screen.library_epubs("translated") == [("Novel", str(compiled)), ("Done", str(done))]
        threads.clear()
        sheet = await screen.pick_from_library("translated")
        assert [item.label for item in sheet.items] == ["Novel", "Done"]
        assert threads and threading.get_ident() not in threads
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_unified_screen_reopened_follows_the_queued_rebuild(iso):
    """Rebuild Now stays disabled on a reopened Unified glossary screen while the rebuild an earlier visit
    queued has not ended, and a second tap queues nothing."""
    from glossarion_mobile.ui.glossary.unified import UnifiedGlossaryScreen

    jobs = _UnifiedJobs(busy=True)
    service = make_service({}, jobs=jobs)

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service, jobs=jobs)
        first = UnifiedGlossaryScreen(None, ctx)
        _mount(page, first.get_body())
        first.did_show()
        job_id = await first.rebuild()
        first.dispose()
        again = UnifiedGlossaryScreen(None, ctx)
        _remount(page, again.get_body())
        again.did_show()
        assert again.rebuild_button.disabled and again.status.value.startswith("(Queued")
        assert await again.rebuild() == job_id and len(jobs.specs) == 1
        jobs.finish(job_id)
        await asyncio.sleep(0.05)
        assert not again.rebuild_button.disabled and "finished" in again.status.value
        again.dispose()

    asyncio.run(scenario())


storage = _TB.storage
app_env = _TB.app_env


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_glossary",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _open_in_app(app, uf, path):
    await app.navigate(f"/glossary/{app.glossary.gid_for(path)}")
    screen = app.shell.top_screen
    assert await uf._wait(lambda: getattr(screen, "editor", None) is not None and screen.editor.doc is not None)
    return screen


def _stack(app):
    return [entry.route for entry in app.shell.stack]


@needs_flet
@needs_gd
def test_unsaved_editor_edits_survive_any_navigation_that_would_dispose_them(app_env):
    """The running app: unsaved glossary edits ask "Unsaved changes" before a tablet sidebar destination, a
    chat row, an outside link (phone) or a full-screen View popped below the editor disposes the editor;
    Keep editing keeps it (and the edit), Discard goes on. The title follows a file switch."""
    uf = _foundations()

    def asked(extras):
        return [q for q in extras.get("asked", []) if q[1] == "Unsaved changes"]

    async def tablet():
        _m, conn, session, page, app = await uf._start("android", width=1000)
        try:
            out = Path(app.paths.output) / "Glossary"
            a = write_token_glossary(out / "Book" / "Book_glossary.csv")
            b = write_token_glossary(out / "Other" / "Other_glossary.csv")
            before = Path(a).read_bytes()
            screen = await _open_in_app(app, uf, a)
            assert app.shell.tablet and app.shell.tablet_title.value == "Book_glossary.csv"
            assert await screen.switch_to(b) and app.shell.tablet_title.value == "Other_glossary.csv"
            assert await screen.switch_to(a) and app.shell.tablet_title.value == "Book_glossary.csv"
            screen.editor.save_entry(screen.editor.specs[0], {"translated_name": "Lunaria"})
            extras = app.glossary_feature.extras
            kept = _stack(app)
            for leave in (lambda: app._drawer_navigate("library"), lambda: app._open_chat("1")):
                extras["answers"] = [False]  # Keep editing
                count = len(asked(extras))
                leave()
                assert await uf._wait(lambda: len(asked(extras)) == count + 1)
                await asyncio.sleep(0.2)
                assert _stack(app) == kept and app.shell.top_screen is screen and screen.editor.dirty
            extras["answers"] = [True]  # Discard
            app._drawer_navigate("library")
            assert await uf._wait(lambda: _stack(app) == ["/library"])
            assert Path(a).read_bytes() == before
            # a full-screen View with the dirty editor above it in the main area: Back on that View asks too
            await app.navigate("/tools/text/0123456789ab")
            screen = await _open_in_app(app, uf, a)
            full = next(e for e in app.shell.stack if e.fullscreen)
            assert _stack(app) == ["/library", full.route, screen.route]
            screen.editor.save_entry(screen.editor.specs[0], {"translated_name": "Lunaria"})
            extras["answers"] = [False]  # Keep editing: the View is gone, the editor stays in the main area
            await session.dispatch_event(page._i, "view_pop", {"route": full.route})
            assert await uf._wait(lambda: _stack(app) == ["/library", screen.route])
            assert app.shell.top_screen is screen and screen.editor.dirty and asked(extras)
        finally:
            await uf._stop(app)

    async def phone():
        _m, conn, session, page, app = await uf._start("android")
        try:
            out = Path(app.paths.output) / "Glossary"
            a = write_token_glossary(out / "Book" / "Book_glossary.csv")
            b = write_token_glossary(out / "Other" / "Other_glossary.csv")
            screen = await _open_in_app(app, uf, a)
            assert page.views[-1].appbar.title.value == "Book_glossary.csv"
            assert await screen.switch_to(b)
            page.update()
            assert page.views[-1].appbar.title.value == "Other_glossary.csv"
            screen.editor.save_entry(screen.editor.specs[0], {"translated_name": "Lunaria"})
            extras = app.glossary_feature.extras
            kept = _stack(app)
            extras["answers"] = [False]
            await _TB._route(session, "/jobs/abc123")  # a notification tap / deep link from outside the app
            await asyncio.sleep(0.2)
            assert _stack(app) == kept and asked(extras) and screen.editor.dirty
            extras["answers"] = [True]
            await _TB._route(session, "/jobs/abc124")
            assert await uf._wait(lambda: _stack(app) == ["/jobs", "/jobs/abc124"])
        finally:
            await uf._stop(app)

    asyncio.run(tablet())
    asyncio.run(phone())


@needs_flet
@needs_gd
def test_reader_add_to_glossary_on_a_tablet_keeps_the_reader_on_top(app_env):
    """The running app on a tablet: the Reader's "Add to glossary" shows the new-entry sheet over the Reader
    (no editor hidden behind it) and Add saves the entry to the book's glossary."""
    uf = _foundations()
    spec = importlib.util.spec_from_file_location("_glossarion_lu_helpers_glossary",
                                                  Path(__file__).with_name("test_library_ui.py"))
    lu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lu)
    lc = lu._core("library_core", "install_library_env", "scan_library")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=1000)
        try:
            home, book_page, bid = await lu._open_novel(app, uf)
            glossary = write_token_glossary(Path(app.paths.output) / "Glossary" / "Novel" / "Novel_glossary.csv")
            await app.navigate(f"/reader/{bid}")
            assert await uf._wait(lambda: app.shell.stack[-1].match.name == "reader")
            reader_route = app.shell.stack[-1].route
            feature = app.glossary_feature
            gid = await feature.add_term("새단어", book=dict(book_page.book))
            assert gid == app.glossary.gid_for(glossary)
            assert app.shell.stack[-1].route == reader_route and page.views[-1].route == reader_route
            assert not any(e.match.name == "glossary.detail" for e in app.shell.stack)
            sheet = feature.sheets[-1]
            assert isinstance(sheet, EntrySheet) and sheet.inputs["raw_name"].value == "새단어"
            sheet.inputs["translated_name"].value = "New Word"
            sheet.save()
            assert await uf._wait(lambda: "New Word" in Path(glossary).read_text(encoding="utf-8"), 10)
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())
