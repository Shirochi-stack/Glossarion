"""Hotfix 1, issues 10 and 1 (the owner's phone report on the current APK), Glossary editor.

Issue 10: "The glossary editor is missing an edit button next to each entry and there is no pop up menu for
the entries like male / female to resolve the issue. There should be a separate button for it next to the
edit." Every row now has ✎ Edit (the entry sheet), and a row whose tracked gender conflicts (the shared
``glossary_document.can_resolve_gender``; its gender shows "Male / Female") has the gender menu next to it:
the desktop Glossary Manager's choices with the desktop labels (the context menu's "Resolve Gender…" and the
dialog's decisions Auto / Male / Female, read from GlossaryManager_GUI.py here). A choice updates that row at
once (only that row is sent) and is written like the desktop's: on Save, to the glossary and the gender
tracker, byte for byte what the shared ``GlossaryDocument`` writes for the same decision. The rows fit a
320 dp phone.

Issue 1 ("Unable to use glossary editor during translation without freezing up the entire app"), the parts
that need no running job (``test_hotfix1_glossary_editor_freeze.py`` reproduces the owner's case with the real
pipeline): Flet 1.0.3 auto-updates the Page after an event handler that called no ``update()`` - before the
fix every scroll event (each 120 ms while scrolling), every keystroke in the search box and most taps of the
editor sent a whole-page diff, and every render sent the whole editor with all its mounted rows. Now no
editor event sends a page diff, a scroll step builds at most ``FULL_ROWS_PER_STEP`` full rows per loop step
(the rest follow a chunk per step), a one-row edit sends that row, a save sends no list, the auto-reload
takes a file a job rewrote without touching unsaved edits or a half-written file, and nothing in the editor
waits on ``job_runner.JOB_LOCK`` or on the shared gender-tracker lock a running job holds.

Every test isolates HOME / USERPROFILE / APPDATA / LOCALAPPDATA / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY /
GLOSSARION_DATA_DIR under ``tmp_path``; nothing reads or writes the developer's Library, output folders or
config.json.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_glossary_editor.py
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import gc
import importlib.util
import json
import os
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for entry in (str(APP_DIR), str(TESTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

G = _load("_glossarion_g_hotfix1_editor", "test_glossary_ui.py")  # make_service, _open_editor, write_token_glossary
gd = G.gd
needs_gd = pytest.mark.skipif(gd is None, reason="glossary_document not importable")

#: generous for slow CI runners (the host: 20-60 ms; before the fix a scroll step alone was 90-130 ms)
LOOP_GAP_S = 0.25


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Scratch HOME / Library / output / data folders and the Windows profile variables."""
    folders = {}
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY",
                 "GLOSSARION_DATA_DIR"):
        folder = tmp_path / name.lower()
        folder.mkdir()
        folders[name] = folder
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.delenv("GLOSSARY_SHARED_DIR", raising=False)
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    return types.SimpleNamespace(root=tmp_path, output=folders["OUTPUT_DIRECTORY"], **{k.lower(): v for k, v in
                                                                                      folders.items()})


# ==========================================================================
# Fixtures: a book glossary whose 루나 has a tracked gender conflict
# ==========================================================================

ENTRIES = [
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female", "description": "a witch"},
    {"type": "character", "raw_name": "카이", "translated_name": "Kai", "gender": "male"},
    {"type": "terms", "raw_name": "마나", "translated_name": "Mana"},
]


def _occurrence(raw: str, name: str, gender: str, chapter: int) -> dict:
    return {"raw_name": raw, "gender": gender, "chapter_index": chapter, "chapter_num": chapter,
            "chapter_file": f"chapter{chapter:04d}.html", "source_path": "Book.epub", "translated_name": name}


def conflict_book(folder: Path) -> str:
    """``<folder>/Book_glossary.csv`` + ``Book_gender_tracker.json``: 루나 seen female twice and male once (a tracked
    conflict, shown "Male / Female"), 카이 male twice (no conflict), 마나 a term (no tracker entry)."""
    path = G.write_token_glossary(folder / "Book_glossary.csv", [dict(e) for e in ENTRIES])
    tracked = {"루나": ("Luna", [("female", 0), ("female", 2), ("male", 5)]), "카이": ("Kai", [("male", 1), ("male", 3)])}
    entries = {}
    for raw, (name, seen) in tracked.items():
        genders = {}
        for gender, chapter in seen:
            genders.setdefault(gender, {"first_seen_chapter": chapter, "first_seen_file": f"chapter{chapter:04d}.html"})
        entries[raw] = {"raw_name": raw, "translated_name": name, "genders": genders,
                        "occurrences": [_occurrence(raw, name, g, c) for g, c in seen]}
    (folder / "Book_gender_tracker.json").write_text(
        json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def big_entries(count: int) -> list:
    return [{"type": "character" if i % 3 else "terms", "raw_name": f"이름{i}", "translated_name": f"Name {i}",
             "gender": ("male" if i % 2 else "female") if i % 3 else "", "description": f"desc {i}"}
            for i in range(count)]


# ==========================================================================
# Helpers
# ==========================================================================


def _walk(control):
    from host_tester import _children

    stack, seen = [control], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(_children(node))


def _key(control):
    key = getattr(control, "key", None)
    return getattr(key, "value", key)


def _find(root, key):
    return next((c for c in _walk(root) if _key(c) == key), None)


def _texts(root) -> list:
    import flet as ft

    out = []
    for node in _walk(root):
        if isinstance(node, ft.Text) and isinstance(node.value, str):
            out.append(node.value)
    return out


def _row(editor, key):
    return editor.windowed.controls[key]


def _is_placeholder(editor, row) -> bool:
    """A row still waiting for its card (``EditorPane._placeholder_row``)."""
    import flet as ft

    from glossarion_mobile.ui.glossary.editor import PLACEHOLDER_HEIGHT

    card = row.content if isinstance(row, ft.Dismissible) else row
    return getattr(card, "height", None) == PLACEHOLDER_HEIGHT and not _texts(card)


class PatchLog:
    """Every patch the session sends: (type of the patched control, its key)."""

    def __init__(self) -> None:
        self.sent: list = []

    @contextlib.contextmanager
    def watching(self):
        from flet.messaging import session as fs

        original = fs.Session.patch_control
        log = self

        def patch_control(self, control, *args, **kwargs):
            log.sent.append((type(control).__name__, str(_key(control))))
            return original(self, control, *args, **kwargs)

        fs.Session.patch_control = patch_control
        try:
            yield self
        finally:
            fs.Session.patch_control = original

    def clear(self) -> None:
        self.sent.clear()

    def kinds(self) -> set:
        return {kind for kind, _key_ in self.sent}


class Heartbeat:
    """The UI loop's largest gap while it runs (a 5 ms tick)."""

    def __init__(self) -> None:
        self.gaps: list = []
        self.task = None

    async def _beat(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.sleep(0.005)
            self.gaps.append(loop.time() - started - 0.005)

    def start(self) -> None:
        self.gaps = []
        self.task = asyncio.ensure_future(self._beat())

    async def stop(self) -> float:
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task
        return max(self.gaps or [0.0])


async def _settle(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


async def _open(iso, path, *, width: int = 320, config=None):
    conn, session = G._TB._fake_session("android")
    session.apply_page_patch({"width": width, "height": 640})
    service = G.make_service(dict(config or {"update_html_on_save": False}))
    ctx, screen, editor = await G._open_editor(session.page, service, path)
    return types.SimpleNamespace(conn=conn, session=session, page=session.page, service=service, ctx=ctx,
                                 screen=screen, editor=editor)


async def _click(env, control, event: str = "click", data=None) -> None:
    await env.session.dispatch_event(control._i, event, data)


def _desktop_gender_labels() -> tuple:
    """The desktop "Resolve Tracked Gender" decisions and the context-menu entry, read from GlossaryManager_GUI.py."""
    source = (SRC_DIR / "GlossaryManager_GUI.py").read_text(encoding="utf-8")
    match = re.search(r"for decision, label in (\(\(\"auto\".*?\)\)):", source)
    assert match, "the desktop decision radios moved (GlossaryManager_GUI._open_gender_resolution)"
    decisions = ast.literal_eval(match.group(1))
    menu = re.search(r"if current_item and _can_resolve_gender\(current_item\):\s*menu\.addAction\(\s*\"([^\"]+)\"",
                     source)
    assert menu, "the desktop context menu's Resolve Gender entry moved (show_tree_context_menu)"
    return tuple(decisions), menu.group(1)


# ==========================================================================
# Issue 10: ✎ Edit on every row, the gender menu next to it for a tracked conflict
# ==========================================================================


@needs_gd
def test_every_row_has_edit_and_a_conflict_row_has_the_desktop_gender_menu(iso):
    import flet as ft

    from glossarion_mobile.ui.glossary.entry_sheet import GENDER_DECISIONS, RESOLVE_GENDER_LABEL, EntrySheet
    from glossarion_mobile.ui.theme import HIT_TARGET

    decisions, menu_label = _desktop_gender_labels()
    assert GENDER_DECISIONS == decisions and RESOLVE_GENDER_LABEL == menu_label == "Resolve Gender…"
    path = conflict_book(iso.output / "Book")

    async def scenario():
        env = await _open(iso, path)
        editor = env.editor
        try:
            luna, kai, mana = editor.specs
            assert editor.service.gender_status(editor.doc, luna.entry)["label"] == "Male / Female"
            for spec in editor.specs:
                row = _row(editor, spec.key)
                edit = _find(row, f"ge-edit-{spec.key}")
                assert isinstance(edit, ft.IconButton) and edit.tooltip == "Edit" and edit.icon == ft.Icons.EDIT_OUTLINED
                assert edit.size_constraints == HIT_TARGET
                gender = _find(row, f"ge-gender-{spec.key}")
                if spec is luna:
                    assert isinstance(gender, ft.PopupMenuButton) and gender.tooltip == RESOLVE_GENDER_LABEL
                    assert gender.size_constraints == HIT_TARGET and gender.icon == ft.Icons.WC
                    assert [i.content for i in gender.items] == [label for _v, label in decisions] + [menu_label]
                    assert [i.checked for i in gender.items[:3]] == [True, False, False]  # Auto is the decision now
                    # next to Edit, in the row's action area (gender menu first, then Edit)
                    actions = row.content.content.controls
                    assert actions[-2:] == [gender, edit]
                else:
                    assert gender is None, f"{spec.entry['raw_name']} has no tracked conflict"
                assert "⚠ Male / Female" in _texts(row) if spec is luna else "⚠ Male / Female" not in _texts(row)
            # ✎ Edit opens the entry sheet of that row (the tap, as Flutter sends it)
            await _click(env, _find(_row(editor, kai.key), f"ge-edit-{kai.key}"))
            sheet = editor.sheets[-1]
            assert isinstance(sheet, EntrySheet) and sheet.inputs["translated_name"].value == "Kai"
            assert sheet.inputs["raw_name"].value == "카이" and editor.current_key == kai.key
            sheet.close()
        finally:
            env.screen.dispose()

    asyncio.run(scenario())


@needs_gd
def test_a_gender_choice_updates_its_row_at_once_and_saves_like_the_desktop(iso):
    """Male from the row's gender menu: the row shows the decision at once (only that row and the header are
    sent), the glossary is edited and undoable, nothing is written until Save; Save writes the glossary and the
    gender tracker exactly as the shared GlossaryDocument does for the same decision (desktop
    ``apply_resolution`` + ``save_edited_glossary``). "Resolve Gender…" opens the full sheet."""
    from glossarion_mobile.ui.glossary.editor import UNRESOLVED_GENDER_PINK, RowShell
    from glossarion_mobile.ui.glossary.entry_sheet import GenderSheet

    path = conflict_book(iso.output / "Book")
    ref_path = conflict_book(iso.root / "ref" / "Book")
    tracker = Path(path).with_name("Book_gender_tracker.json")
    before = (Path(path).read_bytes(), tracker.read_bytes())
    config = {"update_html_on_save": False}

    async def scenario():
        env = await _open(iso, path, config=config)
        editor = env.editor
        try:
            luna = editor.specs[0]
            row = _row(editor, luna.key)
            assert isinstance(row, RowShell) and UNRESOLVED_GENDER_PINK in str(row.content.bgcolor)
            male = next(i for i in _find(row, f"ge-gender-{luna.key}").items if i.content == "Male")
            log = PatchLog()
            with log.watching():
                await _click(env, male)
            assert _row(editor, luna.key) is row  # the same row object, its card replaced
            assert log.kinds() <= {"RowShell", "Container"} and ("RowShell", f"{luna.key}") in log.sent, log.sent
            assert not [k for k in log.sent if k[1] in ("ge-list", "ge-list-holder", "glossary-editor")], log.sent
            texts = _texts(row)
            assert "Male" in texts and "⚠ Male" in texts and "edited" in texts and "⚠ Male / Female" not in texts
            assert UNRESOLVED_GENDER_PINK not in str(row.content.bgcolor)  # resolved: no pink tint
            menu = _find(row, f"ge-gender-{luna.key}")
            assert [i.checked for i in menu.items[:3]] == [False, True, False]
            assert editor.doc.dirty and not editor.undo_button.disabled
            assert editor.doc.current_glossary_data[0]["gender"] == "Male"
            assert (Path(path).read_bytes(), tracker.read_bytes()) == before  # written on Save, like the desktop
            # "Resolve Gender…": the desktop dialog (tracking history, the decision radios)
            sheet_item = next(i for i in menu.items if i.content == "Resolve Gender…")
            await _click(env, sheet_item)
            sheet = editor.sheets[-1]
            assert isinstance(sheet, GenderSheet) and sheet.group.value == "male"
            assert any(line.startswith("Female: 2/3") for line in sheet.model["history"]), sheet.model["history"]
            sheet.host.close()
            report = await editor.save()
            assert report and report["saved"], report
        finally:
            env.screen.dispose()

    asyncio.run(scenario())
    reference = gd.GlossaryDocument(dict(config))
    reference.load(ref_path)
    assert reference.resolve_gender(0, "male")
    assert reference.save_edits(update_output_files=False)["saved"]
    assert Path(path).read_bytes() == Path(ref_path).read_bytes()
    saved = json.loads(tracker.read_text(encoding="utf-8"))
    expected = json.loads(Path(ref_path).with_name("Book_gender_tracker.json").read_text(encoding="utf-8"))
    assert saved == expected and saved["entries"]["루나"]["decision"] == "male"


@needs_gd
def test_rows_fit_a_320dp_phone(iso):
    """At 320 dp (the CI emulator) the row's names, description and chips get the width the 48 dp actions leave
    (expanding, ellipsized / wrapping), in normal and selection mode, for a long entry with a gender conflict."""
    import flet as ft

    flows = _load("_glossarion_uf_hotfix1_editor", "test_ui_flows.py")
    folder = iso.output / "Book"
    path = conflict_book(folder)
    long_entries = [dict(e) for e in ENTRIES]
    long_entries[0].update(translated_name="Lunaria the Everlasting Witch of the Northern Snowfields " * 3,
                           description="A very long description " * 20)
    G.write_token_glossary(folder / "Book_glossary.csv", long_entries)

    def check(editor):
        for spec in editor.specs:
            row = _row(editor, spec.key)
            card = row.content if isinstance(row, ft.Dismissible) else row
            line = card.content
            assert isinstance(line, ft.Row) and not line.wrap
            flexible = [c for c in line.controls if c.expand]
            rigid = [c for c in line.controls if not c.expand]
            assert len(flexible) == 1 and isinstance(flexible[0], ft.Column)
            widths = []
            for control in rigid:
                if isinstance(control, (ft.IconButton, ft.PopupMenuButton)):
                    widths.append(control.size_constraints.min_width)
                elif isinstance(control, ft.Icon):
                    widths.append(control.size or 24)
                else:
                    raise AssertionError(f"unexpected rigid {type(control).__name__} in the row")
            padding = card.padding
            available = 320 - 8 - 8 - padding.left - padding.right  # the list's side padding, the card's
            text_width = available - sum(widths) - line.spacing * (len(line.controls) - 1)
            assert text_width >= 150, (spec.entry["raw_name"], text_width, widths)
            for child in flexible[0].controls:
                if isinstance(child, ft.Text):
                    assert child.max_lines and child.overflow == ft.TextOverflow.ELLIPSIS
                else:
                    assert isinstance(child, ft.Row) and child.wrap  # the chip line wraps
        unflexed = {where: text for where, text in flows.unflexed_ellipsis_rows(editor.ctx.page).items()
                    if "glossary-editor" in where or "/RowShell" in where or "/RowCard" in where}
        assert not unflexed, unflexed

    async def scenario():
        env = await _open(iso, path)
        editor = env.editor
        try:
            assert _find(_row(editor, "g0"), "ge-gender-g0") is not None
            check(editor)
            editor.on_row_long_press(editor.specs[1])  # selection mode: the check icon joins the row
            assert editor.selecting
            check(editor)
        finally:
            env.screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# Issue 1 (no job needed): no whole-page diff, small steps, one-row sends
# ==========================================================================


@needs_gd
def test_no_editor_event_sends_a_whole_page_diff(iso):
    """Scroll events (with and without a step to append), keystrokes in the search box, a row tap, ✎ Edit, a sort,
    a filter chip and the settings tabs: Flet sends no patch of the Page after any of them (before the fix each
    one diffed every View, the chat behind the editor included)."""
    import flet as ft

    path = G.write_token_glossary(iso.output / "Big" / "Big_glossary.csv", big_entries(600))

    async def scenario():
        env = await _open(iso, path)
        editor, page = env.editor, env.page
        try:
            behind = ft.Column([ft.Text(f"chat line {i}") for i in range(200)])  # what a page diff would walk
            page.views[0].controls.append(behind)
            page.update()
            lv = editor.windowed.list_view
            event = {"event_type": "update", "pixels": 100.0, "max_scroll_extent": 99999.0, "min_scroll_extent": 0.0,
                     "viewport_dimension": 600.0}
            log = PatchLog()
            with log.watching():
                for i in range(6):
                    await env.session.dispatch_event(lv._i, "scroll", dict(event, pixels=100.0 + 40 * i))
                assert not log.sent, log.sent  # nothing to append: nothing sent
                mounted = editor.windowed.mounted_count
                await env.session.dispatch_event(lv._i, "scroll", dict(event, pixels=99800.0,
                                                                         max_scroll_extent=100000.0))
                assert editor.windowed.mounted_count == mounted + editor.windowed.step
                for text in ("N", "Na", "Nam"):
                    editor.search.value = text
                    await env.session.dispatch_event(editor.search._i, "change", text)
                await asyncio.wait_for(editor._search_task, 10)
                assert len(editor.visible) == 600
                editor.search.value = "Name 1"
                await env.session.dispatch_event(editor.search._i, "change", "Name 1")
                await asyncio.wait_for(editor._search_task, 10)
                spec = editor.visible[0]
                await _click(env, _row(editor, spec.key).content)  # a row tap opens its sheet
                editor.sheets[-1].close()
                await _click(env, _find(_row(editor, spec.key), f"ge-edit-{spec.key}"))
                editor.sheets[-1].close()
                await _click(env, editor.sort_menu.items[2])  # Translated name A–Z
                editor.set_column_filter("type", frozenset({"terms"}))
                await _click(env, editor.filter_row.controls[0], "delete")
                env.session.apply_patch(env.screen.tabs._i, {"selected_index": 1})  # the client's tab tap
                await env.session.dispatch_event(env.screen.tabs._i, "change", 1)  # General: built on first show
                await asyncio.sleep(0.3)
            pages = [entry for entry in log.sent if entry[0] in ("Page", "View")]
            assert not pages, pages
            assert env.screen.tab_slots["general"].content is not None
            assert all(slot.content is None for name, slot in env.screen.tab_slots.items() if name != "general")
        finally:
            env.screen.dispose()

    asyncio.run(scenario())


@needs_gd
def test_steps_edits_and_saves_send_only_what_changed(iso):
    """A 3,000-entry glossary: a scroll step builds at most ``FULL_ROWS_PER_STEP`` full rows at once and the rest
    a chunk per loop step (the UI loop never waits for a whole step); deep in the window an append does not
    walk the mounted rows again (isolated rows); a one-row edit through the entry sheet sends that row and the
    header; Save sends no list; the loop's largest gap stays small throughout."""
    from glossarion_mobile.ui.glossary.editor import FULL_ROWS_PER_STEP

    path = G.write_token_glossary(iso.output / "Big" / "Big_glossary.csv", big_entries(3000))

    async def scenario():
        env = await _open(iso, path)
        editor = env.editor
        wl = editor.windowed
        beat = Heartbeat()
        gc.collect()
        gc.freeze()  # a full GC of the test process's heap stalls the loop on its own
        try:
            await _settle(lambda: not editor._fill_queue)
            lv = wl.list_view
            event = {"event_type": "update", "pixels": 99800.0, "max_scroll_extent": 100000.0, "min_scroll_extent": 0.0,
                     "viewport_dimension": 600.0}
            beat.start()
            log = PatchLog()
            with log.watching():
                for step in range(6):
                    before = wl.mounted_count
                    await env.session.dispatch_event(lv._i, "scroll", event)
                    new_rows = [_row(editor, k) for k in wl.mounted_keys()[before:]]
                    assert len(new_rows) == wl.step
                    full = [r for r in new_rows if not _is_placeholder(editor, r)]
                    assert len(full) <= FULL_ROWS_PER_STEP, (step, len(full))
                    assert await _settle(lambda: not editor._fill_queue)
                    assert not [r for r in new_rows if _is_placeholder(editor, r)]
            assert ("ListView", "ge-list") in log.sent and ("Page", "None") not in log.sent
            assert wl.mounted_count == 7 * wl.step
            row_texts = _texts(_row(editor, wl.mounted_keys()[-1]))
            last = editor.visible[wl.mounted_count - 1].entry
            assert last["raw_name"] in row_texts and f"→ {last['translated_name']}" in row_texts
            # one row edited through its sheet: that row and the header go out, not the list
            spec = editor.visible[wl.mounted_count - 5]
            log.clear()
            with log.watching():
                await _click(env, _find(_row(editor, spec.key), f"ge-edit-{spec.key}"))
                sheet = editor.sheets[-1]
                sheet.inputs["translated_name"].value = "Renamed"
                sheet.save()
            assert ("RowShell", spec.key) in log.sent, log.sent
            assert not [k for k in log.sent if k[0] in ("ListView", "Page") or k[1] in ("ge-list-holder",
                                                                                         "glossary-editor")], log.sent
            assert "→ Renamed" in _texts(_row(editor, spec.key)) and "edited" in _texts(_row(editor, spec.key))
            log.clear()
            with log.watching():
                report = await editor.save()
            assert report and report["saved"]
            assert not [k for k in log.sent if k[0] in ("ListView", "Page") or k[1] == "ge-list-holder"], log.sent
            assert "edited" not in _texts(_row(editor, spec.key)) and "→ Renamed" in _texts(_row(editor, spec.key))
            gap = await beat.stop()
            assert gap < LOOP_GAP_S, gap
        finally:
            gc.unfreeze()
            env.screen.dispose()

    asyncio.run(scenario())


@needs_gd
def test_the_auto_reload_follows_a_job_writing_the_open_glossary(iso):
    """The file the editor shows is rewritten while it is open (a running job saving the book's glossary):
    a clean document is swapped for the file's new state and only the rows that changed are sent; unsaved
    edits keep the document (the "changed on disk" banner); a file caught empty or half-written leaves the
    document as it was and is read again only once it changes."""
    import glossary_document as shared

    folder = iso.output / "Book"
    path = str(folder / "Book_glossary.json")
    folder.mkdir(parents=True)
    entries = big_entries(400)
    Path(path).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def job_writes(rows, text=None) -> None:
        """What a job's save does to the file: new content and a new mtime."""
        data = text if text is not None else json.dumps(rows, ensure_ascii=False, indent=2)
        Path(path).write_text(data, encoding="utf-8")
        stamp = time.time() + 2 + job_writes.n
        job_writes.n += 2
        os.utime(path, (stamp, stamp))

    job_writes.n = 0

    async def scenario():
        env = await _open(iso, path)
        editor = env.editor
        reads: list = []
        original = env.service.reread_document

        def counting(doc):
            reads.append(time.monotonic())
            return original(doc)

        env.service.reread_document = counting
        try:
            await _settle(lambda: not editor._fill_queue)
            first = editor.doc
            rows = [dict(e) for e in entries]
            rows[3]["translated_name"] = "Written by the job"
            job_writes(rows)
            log = PatchLog()
            with log.watching():
                assert await editor.check_disk() == "reloaded"
                await _settle(lambda: not editor._fill_queue)
            assert editor.doc is not first and not editor.dirty
            assert "→ Written by the job" in _texts(_row(editor, "g3"))
            row_patches = [k for k in log.sent if k[0] in ("RowShell", "RowCard")]
            assert row_patches == [("RowShell", "g3")], log.sent
            assert not [k for k in log.sent if k[0] in ("ListView", "Page") or k[1] == "ge-list-holder"], log.sent
            # unsaved edits: the job's write does not replace them
            editor.save_entry(editor.specs[7], {"translated_name": "Mine"})
            rows[9]["translated_name"] = "Job again"
            job_writes(rows)
            assert await editor.check_disk() == "changed"
            assert editor.changed_on_disk and editor.changed_banner.visible and editor.dirty
            assert "→ Mine" in _texts(_row(editor, "g7")) and "→ Job again" not in _texts(_row(editor, "g9"))
            # the banner's Reload takes the file (the edit is dropped, like the desktop Reload)
            await editor.reload(force=True)
            assert not editor.dirty and "→ Job again" in _texts(_row(editor, "g9"))
            # caught empty (a job truncated the file to write it): kept, read again on the next check
            kept = editor.doc
            reads.clear()
            job_writes(rows, text="")
            assert await editor.check_disk() is None and editor.doc is kept
            assert reads and editor._unreadable_mtime is None
            # caught half-written: kept; that file state is not read again ...
            job_writes(rows, text=json.dumps(rows, ensure_ascii=False)[:500])
            with pytest.raises(shared.GlossaryEditorError):
                shared.GlossaryDocument({}).load(path)
            assert await editor.check_disk() is None and editor.doc is kept and len(editor.specs) == 400
            count = len(reads)
            assert await editor.check_disk() is None and len(reads) == count
            # ... until the job finished writing it
            rows.append({"type": "terms", "raw_name": "새", "translated_name": "New"})
            job_writes(rows)
            assert await editor.check_disk() == "reloaded" and len(editor.specs) == 401
        finally:
            env.screen.dispose()

    asyncio.run(scenario())


@needs_gd
def test_nothing_waits_on_the_locks_a_running_job_holds(iso):
    """A job holds ``job_runner.JOB_LOCK`` for its whole run, and the shared gender-tracker lock while it records
    genders. The editor opens, searches, edits and saves without waiting for JOB_LOCK (Remove Duplicates, which
    sets process-wide env, says the run is busy like the desktop); a save that needs the tracker lock waits on the
    io pool for the job's write to finish - the UI loop keeps running meanwhile - and then writes."""
    job_runner = pytest.importorskip("job_runner")
    efe = pytest.importorskip("extract_glossary_from_epub")
    path = conflict_book(iso.output / "Book")

    async def scenario():
        job_holds = threading.Event()
        let_go = threading.Event()

        def job(lock) -> None:
            with lock:
                job_holds.set()
                let_go.wait(20)

        runner = threading.Thread(target=job, args=(job_runner.JOB_LOCK,), daemon=True)
        runner.start()
        assert job_holds.wait(5)
        env = None
        try:
            env = await asyncio.wait_for(_open(iso, path), 20)
            editor = env.editor
            editor.search.value = "Kai"
            editor._on_search()
            await asyncio.wait_for(editor._search_task, 10)
            assert [s.entry["raw_name"] for s in editor.visible] == ["카이"]
            editor.save_entry(editor.visible[0], {"translated_name": "Kyle"})
            assert (await asyncio.wait_for(editor.save(), 30))["saved"]
            assert await asyncio.wait_for(editor.run_tool("dedupe"), 10) is None
            assert any("running job" in str(n) for n in env.ctx.notes), env.ctx.notes
            free = job_runner.JOB_LOCK.acquire(blocking=False)  # still the job's the whole time
            if free:
                job_runner.JOB_LOCK.release()
            assert not free and runner.is_alive()
        finally:
            let_go.set()
            runner.join(10)
        # the tracker lock: the save waits for it off the loop
        tracker_lock = efe._gender_tracker_lock
        editor.save_entry(editor.specs[0], {"description": "edited while the job writes genders"})
        resolved = editor.apply_gender(editor.specs[0], "female")
        assert resolved
        job_holds.clear()
        let_go.clear()
        writer = threading.Thread(target=job, args=(tracker_lock,), daemon=True)
        writer.start()
        assert job_holds.wait(5)
        beat = Heartbeat()
        beat.start()
        saving = asyncio.ensure_future(editor.save())
        await asyncio.sleep(0.6)
        assert not saving.done()  # waiting for the job's tracker write (desktop: the same lock)
        let_go.set()
        report = await asyncio.wait_for(saving, 30)
        gap = await beat.stop()
        writer.join(10)
        assert report["saved"] and gap < LOOP_GAP_S, (report, gap)
        tracker = json.loads(Path(path).with_name("Book_gender_tracker.json").read_text(encoding="utf-8"))
        assert tracker["entries"]["루나"]["decision"] == "female"
        env.screen.dispose()

    asyncio.run(scenario())
