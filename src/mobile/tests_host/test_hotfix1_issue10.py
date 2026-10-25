"""Hotfix 1, owner issue 10 - acceptance test on the real app.

What the owner reported from the phone (current APK): "The glossary editor is missing an edit button next to
each entry and there is no pop up menu for the entries like male / female to resolve the issue. There should
be a separate button for it next to the edit that was never added".

The scenario replays it on the real objects: ``GlossarionApp`` on the fake Flet session as an Android phone at
320 x 640 dp (the CI emulator's size), with the owner's book "[TS] I'm the Only Mechanical Genius Girl". Its
glossary sits where a run writes it (``<output>/Glossary/<book>/<book>_glossary.csv``) and its gender tracker
was written by the translation pipeline itself (``extract_glossary_from_epub.update_gender_tracker``, chapter
by chapter): 유나 (Yuna) was seen female, female, then male and 에델바이스 (a long name and description) male,
female, female - both are tracked conflicts the editor shows as "Male / Female"; 레온 is always male and
마도공학 is a term. The user walks the app like on the phone - drawer › Glossaries › the book's glossary - and:

* every row shows ✎ Edit; each conflicting row (and only those, like the desktop context menu) has a separate
  gender button right next to Edit;
* the gender button's popup lists the desktop Glossary Manager's choices with the desktop labels, read from
  GlossaryManager_GUI.py (the "Resolve Tracked Gender" decisions Auto / Male / Female and the context menu's
  "Resolve Gender…"), the tracker's current decision checked;
* picking Male updates that row at once like the desktop row (the label the desktop's
  ``entry_gender_status`` gives, no unresolved pink, marked updated), nothing is written before Save; the
  popup's "Resolve Gender…" opens the full sheet (desktop dialog: history lines, decision radios) where
  Female is applied for the second conflict; Save writes the glossary and the tracker exactly like the
  desktop's own calls for the same two choices (``parse_glossary_file`` / ``apply_parse_payload`` /
  ``apply_gender_decision`` / ``save_document``), and reopening the glossary shows the saved decisions;
* ✎ Edit opens the entry sheet of that row (its values; the conflict banner and "Resolve Gender…" on a
  conflicting row), an edit made there shows in the row and is saved;
* the rows fit 320 dp: no Flutter layout error in the editor's view (``test_progress_layout_constraints``'s
  constraint model), no unflexed ellipsis text, and the width the 48 dp buttons leave holds the names and
  every chip - in the normal and the selection layout.

Real data stays untouched: the app runs on temp storage (``test_devfix_issue12.iso``: FLET_APP_STORAGE_*,
HOME, USERPROFILE, APPDATA, LOCALAPPDATA, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR,
CONFIG_FILE under tmp; GLOSSARION_HTTP_LOG=0), the Glossaries list never scans the checkout's Glossary/
folder, the E2E network guard refuses every non-loopback connection, and src/ (config.json included) is
checked unchanged.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_issue10.py
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for _entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # dataclasses in the helper modules look their module up
    spec.loader.exec_module(module)
    return module


I12 = _load("_glossarion_devfix12_hotfix1_issue10", "test_devfix_issue12.py")  # the real-app harness
pytestmark = I12.pytestmark
storage = I12.storage
app_env = I12.app_env
iso = I12.iso
offline = I12.offline

PHONE = (320, 640)
BOOK = "[TS] I'm the Only Mechanical Genius Girl"
LONG_NAME = "Edelweiss von Hohenzollern-Sigmaringen, Grand Engineer of the Northern Mechanical Guild"
LONG_DESCRIPTION = " ".join(["A rival inventor whose clockwork knights guard the northern border."] * 6)
ENTRIES = [
    {"type": "character", "raw_name": "유나", "translated_name": "Yuna", "gender": "female",
     "description": "The only mechanical genius girl"},
    {"type": "character", "raw_name": "레온", "translated_name": "Leon", "gender": "male",
     "description": "Yuna's rival"},
    {"type": "character", "raw_name": "에델바이스", "translated_name": LONG_NAME, "gender": "male",
     "description": LONG_DESCRIPTION},
    {"type": "terms", "raw_name": "마도공학", "translated_name": "Magitech", "description": "Magic engineering"},
]
#: what each chapter's glossary extraction reported for the book's characters (the pipeline's tracker input)
SEEN = {1: {"유나": "female", "레온": "male"}, 2: {"유나": "female", "에델바이스": "male"}, 3: {"레온": "male"},
        4: {"에델바이스": "female"}, 5: {"유나": "male"}, 6: {"에델바이스": "female"}}
CONFLICTS = ("유나", "에델바이스")
#: the owner's choices: 유나 from the row's popup (Female, back to Auto, then Male), 에델바이스 Female through the
#: popup's full sheet; DECISIONS is what stays
YUNA_STEPS = ("female", "auto", "male")
DECISIONS = {"유나": "male", "에델바이스": "female"}
#: the desktop editor's colour for a row whose tracked conflict is unresolved
#: (GlossaryManager_GUI ``_apply_editor_gender_presentation``: QColor("#ec4899"))
DESKTOP_UNRESOLVED_PINK = "#ec4899"
#: Material 3 icon buttons (IconButton, a PopupMenuButton's icon button) take a 48 x 48 dp tap target on phones
HIT = 48.0
#: the narrowest the names / description column may get at 320 dp (two 48 dp buttons beside it)
MIN_TEXT_COLUMN = 140.0
TEXT_SCALES = (1.0, 1.3)


# ==========================================================================
# The desktop side (read from the desktop sources, run through the desktop's own calls)
# ==========================================================================


def _desktop_choices() -> tuple:
    """``((decision, label), ...)`` of the desktop "Resolve Tracked Gender" dialog's radios and the label of the
    editor context menu's entry that opens it, parsed from GlossaryManager_GUI.py."""
    tree = ast.parse((SRC_DIR / "GlossaryManager_GUI.py").read_text(encoding="utf-8"))
    decisions, menu = None, None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_open_gender_resolution":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.For) and isinstance(sub.target, ast.Tuple)
                        and [getattr(e, "id", None) for e in sub.target.elts] == ["decision", "label"]):
                    decisions = tuple(ast.literal_eval(sub.iter))
        elif node.name == "show_tree_context_menu":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "addAction"
                        and len(sub.args) == 2 and isinstance(sub.args[0], ast.Constant)
                        and isinstance(sub.args[1], ast.Lambda) and "_open_gender_resolution" in ast.dump(sub.args[1])):
                    menu = sub.args[0].value
    assert decisions and menu, "the desktop Resolve Gender dialog / context menu moved in GlossaryManager_GUI.py"
    return decisions, menu


def _desktop_editor(path: str, config: dict):
    """The desktop Glossary Manager's editor state after it loaded ``path`` (``_parse_glossary_file_for_editor_async``
    + ``apply_loaded_glossary_result``) and its row presentation helpers. The owner is the TranslatorGUI state for
    ``config`` (``glossary_document.EditorOwner``: the desktop start-up seeds ``custom_glossary_fields`` etc.); the
    editor keeps its document on that owner, as the desktop's ``self``."""
    import glossary_document as gdoc

    owner = gdoc.EditorOwner(config)
    config = owner.config
    gdoc.apply_parse_payload(owner, gdoc.parse_glossary_file(path, config, getattr(owner, "custom_entry_types", None)))
    assert owner.current_gender_tracker_data, "the desktop loader found no gender tracker"

    def settings():
        return gdoc.editor_gender_settings(owner, config)

    def entry(raw: str) -> dict:
        return next(e for e in owner.current_glossary_data if e.get("raw_name") == raw)

    def status(raw: str):
        return gdoc.entry_gender_status(entry(raw), owner.current_gender_tracker_data, settings)

    def tracker_entry(raw: str):
        return gdoc.tracker_entry_for_entry(entry(raw), owner.current_gender_tracker_data)

    def decide(raw: str, decision: str) -> None:
        """``_open_gender_resolution`` › Apply (``apply_resolution``)."""
        data_entry = entry(raw)
        assert gdoc.can_resolve_gender(status(raw)), raw
        gdoc.apply_gender_decision(owner, tracker_entry(raw), data_entry,
                                   str(data_entry.get("raw_name", "") or "").strip(), decision)

    def history(raw: str) -> list:
        threshold, bias = settings()
        summary = gdoc.gender_resolution_summary(tracker_entry(raw), entry(raw).get("gender", ""), threshold, bias)
        return [gdoc.gender_history_line(g, summary, summary["total"]) for g in ("male", "female")]

    return types.SimpleNamespace(owner=owner, entry=entry, status=status, tracker_entry=tracker_entry, decide=decide,
                                 history=history, save=lambda: gdoc.save_document(owner, path, config))


def _tracker_view(raw_bytes: bytes) -> dict:
    data = json.loads(raw_bytes.decode("utf-8"))
    data.pop("updated_at", None)  # the write time
    return data


# ==========================================================================
# The book on the phone
# ==========================================================================


def _write_book(shared: Path) -> Path:
    """The book's glossary where a run writes it, and its gender tracker as the pipeline records it."""
    import extract_glossary_from_epub as pipeline
    import gender_tracking
    import glossary_document as gdoc

    folder = shared / BOOK
    folder.mkdir(parents=True)
    path = folder / f"{BOOK}_glossary.csv"
    gdoc.write_token_csv([dict(e) for e in ENTRIES], str(path), [], [])
    by_raw = {e["raw_name"]: e for e in ENTRIES}
    for chapter, seen in SEEN.items():
        pipeline.update_gender_tracker([dict(by_raw[raw], gender=gender) for raw, gender in seen.items()], str(path),
                                       source_path=str(shared.parent / f"{BOOK}.epub"), chapter_index=chapter - 1,
                                       chapter_num=chapter, chapter_file=f"chapter{chapter:04d}.xhtml")
    tracker = Path(gender_tracking.tracker_path_for_glossary(str(path)))
    assert tracker == folder / f"{BOOK}_gender_tracker.json" and tracker.is_file()
    entries = json.loads(tracker.read_text(encoding="utf-8"))["entries"]
    assert {raw: sorted({o["gender"] for o in entries[raw]["occurrences"]}) for raw in ("유나", "레온", "에델바이스")} == {
        "유나": ["female", "male"], "레온": ["male"], "에델바이스": ["female", "male"]}
    return path


# ==========================================================================
# Tree helpers (what Flutter's finders see on the device: PyTester's walk)
# ==========================================================================


def _children(control):
    from host_tester import _children as children

    return list(children(control))


def _walk(control):
    stack, seen = [control], set()
    while stack:
        node = stack.pop()
        if id(node) in seen or getattr(node, "visible", True) is False:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(_children(node)))  # depth first, in the screen's order


def _texts(control) -> list:
    import flet as ft

    return [n.value for n in _walk(control) if isinstance(n, ft.Text) and isinstance(n.value, str)]


def _ancestors(control):
    node = getattr(control, "parent", None)
    while node is not None:
        yield node
        node = getattr(node, "parent", None)


def _row_of(editor, control):
    """The mounted editor row (list entry) holding ``control``, and its row key."""
    by_id = {id(row): key for key, row in editor.windowed.controls.items()}
    for node in [control, *_ancestors(control)]:
        if id(node) in by_id:
            return by_id[id(node)], node
    return None, None


def _raw_of(editor, key) -> str:
    spec = editor.spec_for_key(key)
    return str(spec.entry.get("raw_name")) if spec is not None else ""


def _card(row):
    import flet as ft

    return row.content if isinstance(row, ft.Dismissible) else row


def _rows_by_raw(editor) -> dict:
    return {_raw_of(editor, key): row for key, row in editor.windowed.controls.items()}


async def _found(tester, **spec) -> list:
    """The controls a Flutter finder matches on the screen (top view + open sheets / dialogs)."""
    if "tooltip" in spec:
        finder = await tester.find_by_tooltip(spec["tooltip"])
    elif "text" in spec:
        finder = await tester.find_by_text(spec["text"])
    else:
        finder = await tester.find_by_key(spec["key"])
    return list(tester._finders[finder.id])


async def _row_buttons(tester, editor, tooltip: str) -> dict:
    """``{raw name: [buttons]}``: the buttons (IconButton / PopupMenuButton) with ``tooltip`` on the screen, by row."""
    import flet as ft

    out: dict = {}
    for control in await _found(tester, tooltip=tooltip):
        if isinstance(control, (ft.IconButton, ft.PopupMenuButton)):
            key, _row = _row_of(editor, control)
            out.setdefault(_raw_of(editor, key), []).append(control)
    return out


async def _tap(tester, control) -> None:
    """A tap on ``control`` as Flutter delivers it (PyTester: the nearest enabled control with a handler)."""
    await tester.tap(tester._register([control]))
    await asyncio.sleep(0.05)


async def _settle(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


def _open_sheets(page) -> list:
    dialogs = getattr(getattr(page, "_dialogs", None), "controls", None) or []
    return [d for d in dialogs if getattr(d, "open", False)]


# ==========================================================================
# The 320 dp fit
# ==========================================================================


def _horizontal(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return 2.0 * float(value)
    return float(getattr(value, "left", 0) or 0) + float(getattr(value, "right", 0) or 0)


def _border(control) -> float:
    border = getattr(control, "border", None)
    if border is None:
        return 0.0
    return sum(float(getattr(getattr(border, side, None), "width", 0) or 0) for side in ("left", "right"))


def _width_of(control, screen_width: float) -> float:
    """The width a control is laid out in: the screen minus every ancestor's horizontal padding / margin / border
    (a fixed-width ancestor bounds it instead)."""
    width = screen_width
    insets = 0.0
    for node in _ancestors(control):
        fixed = getattr(node, "width", None)
        if isinstance(fixed, (int, float)) and not isinstance(fixed, bool):
            width = float(fixed)
            break
        insets += _horizontal(getattr(node, "padding", None)) + _horizontal(getattr(node, "margin", None)) + _border(node)
    return width - insets


def _glyph_em(ch: str) -> float:
    """A generous advance (em) per character: CJK / Hangul full width, emoji and symbols wide."""
    code = ord(ch)
    if code in (0xFE0F, 0x200D):
        return 0.0
    if 0x1100 <= code <= 0x11FF or 0x2E80 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF or 0xF900 <= code <= 0xFAFF:
        return 1.0
    if code >= 0x1F000 or 0x2600 <= code <= 0x27BF:
        return 1.3
    if ch == " ":
        return 0.3
    return 0.72 if ch.isupper() or ch in "mw" else 0.6


def _text_width(text: str, size: float, scale: float) -> float:
    return sum(_glyph_em(ch) * size * scale + 0.1 for ch in text)


def _fit_problems(editor, screen_width: float) -> list:
    """Why a mounted row would not fit the screen (empty: every row fits)."""
    import flet as ft

    problems: list = []
    for key, row in editor.windowed.controls.items():
        raw = _raw_of(editor, key)
        card = _card(row)
        line = card.content
        if not isinstance(line, ft.Row) or line.wrap or line.scroll is not None:
            problems.append(f"{raw}: the row's line is not a plain Row ({type(line).__name__})")
            continue
        children = [c for c in line.controls if c.visible is not False]
        flexible = [c for c in children if c.expand]
        if len(flexible) != 1 or not isinstance(flexible[0], ft.Column):
            problems.append(f"{raw}: expected one expanding text column, got {[type(c).__name__ for c in flexible]}")
            continue
        rigid = 0.0
        for child in children:
            if child is flexible[0]:
                continue
            if isinstance(child, (ft.IconButton, ft.PopupMenuButton)):
                constraints = getattr(child, "size_constraints", None) if isinstance(child, ft.IconButton) else None
                rigid += max(HIT, float(getattr(constraints, "min_width", 0) or 0))
            elif isinstance(child, ft.Icon):
                rigid += float(child.size or 24)
            else:
                problems.append(f"{raw}: no width model for {type(child).__name__} beside the text column")
        spacing = float(10 if line.spacing is None else line.spacing) * (len(children) - 1)
        inner = _width_of(card, screen_width) - _horizontal(card.padding) - _border(card)
        column = inner - rigid - spacing
        if column < MIN_TEXT_COLUMN:
            problems.append(f"{raw}: the names get {column:.0f} dp of {screen_width:.0f} (buttons {rigid:.0f} dp)")
        for child in flexible[0].controls:
            if child.visible is False:
                continue
            if isinstance(child, ft.Text):
                if child.no_wrap and child.overflow != ft.TextOverflow.ELLIPSIS:
                    problems.append(f"{raw}: unwrapped text without an ellipsis {child.value!r:.40}")
                if child.max_lines and child.overflow != ft.TextOverflow.ELLIPSIS:
                    problems.append(f"{raw}: clipped lines without an ellipsis {child.value!r:.40}")
            elif isinstance(child, ft.Row):
                if not child.wrap:
                    problems.append(f"{raw}: the chip line does not wrap")
                for chip in child.controls:
                    if chip.visible is False:
                        continue
                    label = chip.content if isinstance(chip, ft.Container) else chip
                    if not isinstance(label, ft.Text):
                        problems.append(f"{raw}: no width model for chip {type(chip).__name__}")
                        continue
                    size = float(label.size or 14)
                    pad = _horizontal(getattr(chip, "padding", None)) if isinstance(chip, ft.Container) else 0.0
                    for scale in TEXT_SCALES:
                        need = _text_width(str(label.value or ""), size, scale) + pad
                        if need > column:
                            problems.append(f"{raw}: chip {label.value!r} needs {need:.0f} dp of {column:.0f} at x{scale}")
            else:
                problems.append(f"{raw}: no width model for {type(child).__name__} in the text column")
    return problems


def _layout_problems(page) -> list:
    """Flutter layout errors in the editor's View (constraint model) and unflexed ellipsis texts in its rows."""
    layout = _load("_glossarion_layout_hotfix1_issue10", "test_progress_layout_constraints.py")
    flows = _load("_glossarion_flows_hotfix1_issue10", "test_ui_flows.py")
    view = page.views[-1]
    problems = list(layout.layout_errors(view))
    problems += [f"unflexed ellipsis {where}: {text!r}" for where, text in flows.unflexed_ellipsis_rows(page).items()
                 if "/RowShell" in where or "/RowCard" in where or "glossary-editor" in where or "[ge-" in where]
    return problems


# ==========================================================================
# The scenario
# ==========================================================================


def _md5(path: Path) -> str:
    # src/config.json is untracked (gitignored): a CI checkout has none, which is fine ("" before and after)
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


def test_issue10_edit_and_gender_buttons_on_the_phone(iso, offline, monkeypatch):
    import flet as ft

    import flows
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet, GenderSheet

    src_listing = sorted(os.listdir(SRC_DIR))
    config_md5 = _md5(SRC_DIR / "config.json")
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    monkeypatch.delenv("GLOSSARY_SHARED_DIR", raising=False)
    I12._assert_isolated(iso.tmp)
    data_dir = Path(rb.get_paths().data)
    # the Glossaries list scans <app dir>/Glossary too: on a Windows host that is the checkout's src/ (real data)
    monkeypatch.setattr(GlossaryService, "app_dir", lambda self: str(data_dir))
    output = Path(os.environ["OUTPUT_DIRECTORY"])
    assert I12._inside(output, iso.tmp)
    path = _write_book(output / "Glossary")
    tracker_path = path.with_name(f"{BOOK}_gender_tracker.json")
    original = {path: path.read_bytes(), tracker_path: tracker_path.read_bytes()}
    decisions, menu_label = _desktop_choices()
    choice_labels = [label for _value, label in decisions]
    results: dict = {}

    async def open_glossary(app, tester, driver):
        """Drawer › Glossaries › the book's glossary, as on the phone; returns the editor once its rows are up."""
        await flows.open_drawer(driver)
        await driver.tap(key="dest-glossary")
        gid = app.glossary.gid_for(str(path))
        await driver.wait(key=f"gh-row-{gid}", timeout=30)
        await driver.tap(key=f"gh-row-{gid}")
        screen = None

        def ready() -> bool:
            nonlocal screen
            screen = app.shell.top_screen
            editor = getattr(screen, "editor", None)
            return (editor is not None and editor.doc is not None and not editor.loading
                    and len(editor.windowed.controls) == len(ENTRIES) and not getattr(editor, "_fill_queue", None))

        assert await _settle(ready, 30), getattr(getattr(screen, "editor", None), "error", None)
        assert screen.editor.path == str(path)
        return screen.editor

    async def body():
        tf = I12._UF._foundations()
        app, tester, driver = await I12._UF._host_driver(tf, {}, width=PHONE[0])
        session, page = tester.session, tester.page
        try:
            await session.dispatch_event(page._i, "resize", {"width": PHONE[0], "height": PHONE[1]})
            await flows.wait_home(driver)
            config = dict(app.glossary.config_snapshot())
            results["config"] = config
            desktop = _desktop_editor(str(path), config)
            editor = await open_glossary(app, tester, driver)

            # ---- every row: ✎ Edit; a conflicting row: the gender button right next to it ------------------
            edits = await _row_buttons(tester, editor, "Edit")
            genders = await _row_buttons(tester, editor, menu_label)
            assert {raw: len(found) for raw, found in edits.items()} == {e["raw_name"]: 1 for e in ENTRIES}, edits
            expected_conflicts = sorted(raw for raw in (e["raw_name"] for e in ENTRIES)
                                        if (desktop.status(raw) or {}).get("conflict"))
            assert expected_conflicts == sorted(CONFLICTS)  # the desktop offers Resolve Gender… for these two only
            assert {raw: len(found) for raw, found in genders.items()} == {raw: 1 for raw in CONFLICTS}, genders
            buttons = {raw: {"edit": found[0]} for raw, found in edits.items()}
            for raw, found in genders.items():
                buttons[raw]["gender"] = found[0]
            for raw, found in buttons.items():
                edit = found["edit"]
                assert isinstance(edit, ft.IconButton) and edit.on_click is not None and not edit.disabled, raw
                if raw in CONFLICTS:
                    gender = found["gender"]
                    assert isinstance(gender, ft.PopupMenuButton) and not gender.disabled, raw
                    assert gender is not edit and gender.parent is edit.parent, f"{raw}: not beside Edit"
                    siblings = [c for c in edit.parent.controls if c.visible is not False]
                    assert abs(siblings.index(gender) - siblings.index(edit)) == 1, f"{raw}: not next to Edit"
            rows = _rows_by_raw(editor)
            for raw in CONFLICTS:  # what the owner saw: "male / female", the desktop's unresolved pink
                label = desktop.status(raw)["label"]
                assert label == "Male / Female" and desktop.status(raw)["unresolved"]
                assert any(label in t for t in _texts(rows[raw])), (raw, _texts(rows[raw]))
                assert DESKTOP_UNRESOLVED_PINK in str(_card(rows[raw]).bgcolor), (raw, _card(rows[raw]).bgcolor)

            # ---- the popup: the desktop choices, the tracker's decision checked ------------------------------
            for raw in CONFLICTS:
                items = [i for i in buttons[raw]["gender"].items if i.visible is not False]
                assert [i.content for i in items] == choice_labels + [menu_label], [i.content for i in items]
                current = desktop.tracker_entry(raw).get("decision", "auto")
                assert [bool(i.checked) for i in items[:len(decisions)]] == [v == current for v, _l in decisions]
                assert all(i.on_click is not None and not i.disabled for i in items), raw

            # ---- 유나: the owner changes their mind in the popup - Female, Auto, then Male ------------------------
            for decision in YUNA_STEPS:
                gender = (await _row_buttons(tester, editor, menu_label))["유나"][0]  # the row's button as it is now
                await _tap(tester, gender)  # opens the menu on the device; its items are in the tree
                await _tap(tester, next(i for i in gender.items if i.content == dict(decisions)[decision]))
                desktop.decide("유나", decision)
                want = desktop.status("유나")
                assert want["conflict"]  # the tracker keeps both genders: the desktop still offers Resolve Gender…
                row = _rows_by_raw(editor)["유나"]
                texts = _texts(row)
                assert any(want["label"] in t for t in texts), (decision, want["label"], texts)  # the gender column
                assert "edited" in texts, (decision, texts)  # desktop mark_row_updated(item, True) after each Apply
                if not want["unresolved"]:
                    assert not any("Male / Female" in t for t in texts), (decision, texts)
                    bgcolor = str(_card(row).bgcolor)
                    assert DESKTOP_UNRESOLVED_PINK not in bgcolor and "orange" in bgcolor.lower(), (decision, bgcolor)
                menus = (await _row_buttons(tester, editor, menu_label)).get("유나") or []
                assert len(menus) == 1, (decision, menus)
                assert [bool(i.checked) for i in menus[0].items[:len(decisions)]] == [v == decision for v, _l in decisions]
                assert editor.dirty and not editor.save_button.disabled
            assert desktop.status("유나")["label"] == "Male"
            assert {p: p.read_bytes() for p in original} == original, "written before Save (the desktop writes on Save)"

            # ---- 에델바이스: the popup's "Resolve Gender…" -> the full sheet -> Female ------------------------
            gender = (await _row_buttons(tester, editor, menu_label))["에델바이스"][0]
            await _tap(tester, gender)
            await _tap(tester, next(i for i in gender.items if i.content == menu_label))
            assert await _settle(lambda: editor.sheets and isinstance(editor.sheets[-1], GenderSheet)
                                 and editor.sheets[-1].dialog in _open_sheets(page), 10)
            sheet = editor.sheets[-1]
            assert [(r.value, r.label) for r in _walk(sheet.group) if isinstance(r, ft.Radio)] == list(decisions)
            assert sheet.group.value == "auto"
            assert [str(line) for line in sheet.model["history"]] == desktop.history("에델바이스")
            session.apply_patch(sheet.group._i, {"value": "female"})  # the user picks the Female radio
            await driver.tap(text="Apply")
            assert await _settle(lambda: sheet.dialog not in _open_sheets(page), 10)
            desktop.decide("에델바이스", "female")
            want = desktop.status("에델바이스")
            row = _rows_by_raw(editor)["에델바이스"]
            texts = _texts(row)
            assert want["label"] == "Female" and any(t in ("Female", "⚠ Female") for t in texts), texts
            assert not any("Male / Female" in t for t in texts) and "edited" in texts, texts
            assert DESKTOP_UNRESOLVED_PINK not in str(_card(row).bgcolor)

            # ---- Save: the glossary and the tracker as the desktop writes them -------------------------------
            await driver.tap(tooltip="Save")
            assert await _settle(lambda: not editor.dirty and not editor._saving, 30), editor.reports[-1:]
            assert editor.reports and editor.reports[-1].get("saved"), editor.reports[-1:]
            results["mobile"] = {p: p.read_bytes() for p in original}
            assert results["mobile"] != original
            for raw in CONFLICTS:  # desktop: highlights cleared after the save, the label stays
                texts = _texts(_rows_by_raw(editor)[raw])
                assert "edited" not in texts and not any("Male / Female" in t for t in texts), (raw, texts)

            # ---- reopened from disk: the saved decisions ------------------------------------------------------
            await flows.go_home(driver)
            editor = await open_glossary(app, tester, driver)
            for raw, decision in DECISIONS.items():
                menu = (await _row_buttons(tester, editor, menu_label))[raw][0]
                assert [bool(i.checked) for i in menu.items[:len(decisions)]] == [v == decision for v, _l in decisions]
                texts = _texts(_rows_by_raw(editor)[raw])
                assert any(t in (decision.title(), f"⚠ {decision.title()}") for t in texts), (raw, texts)
                assert DESKTOP_UNRESOLVED_PINK not in str(_card(_rows_by_raw(editor)[raw]).bgcolor)

            # ---- ✎ Edit: the entry sheet of that row ------------------------------------------------------------
            sheets_before = len(editor.sheets)
            edit = (await _row_buttons(tester, editor, "Edit"))["레온"][0]
            await _tap(tester, edit)
            assert await _settle(lambda: len(editor.sheets) == sheets_before + 1, 10)
            sheet = editor.sheets[-1]
            assert isinstance(sheet, EntrySheet) and _open_sheets(page) == [sheet.dialog]
            assert sheet.inputs["raw_name"].value == "레온" and sheet.inputs["translated_name"].value == "Leon"
            assert sheet.inputs["description"].value == "Yuna's rival"
            assert "entry-resolve" not in {getattr(c, "key", None) for c in _walk(sheet.dialog)}  # no conflict here
            await driver.enter("Leon Ashford", key="entry-translated_name")
            await driver.tap(text="Save")
            assert await _settle(lambda: sheet.dialog not in _open_sheets(page), 10)
            texts = _texts(_rows_by_raw(editor)["레온"])
            assert "→ Leon Ashford" in texts and "edited" in texts, texts
            assert editor.dirty
            # a conflicting row's sheet: the conflict and its own Resolve Gender…
            edit = (await _row_buttons(tester, editor, "Edit"))["에델바이스"][0]
            await _tap(tester, edit)
            assert await _settle(lambda: len(editor.sheets) == sheets_before + 2, 10)
            sheet = editor.sheets[-1]
            assert isinstance(sheet, EntrySheet) and sheet.inputs["translated_name"].value == LONG_NAME
            resolve = [c for c in _walk(sheet.dialog) if getattr(c, "key", None) == "entry-resolve"]
            assert resolve and resolve[0].content == menu_label
            assert any("Female" in t for t in _texts(sheet.dialog)), _texts(sheet.dialog)
            await driver.tap(text="Cancel")
            assert await _settle(lambda: sheet.dialog not in _open_sheets(page), 10)
            await driver.tap(tooltip="Save")

            async def saved_or_asked() -> bool:
                if not editor.dirty and not editor._saving:
                    return True
                if await driver.count(text="Yes"):  # the desktop "Update output files?" question: its default
                    await driver.tap(text="Yes")
                return False

            deadline = time.monotonic() + 30
            while not await saved_or_asked() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert not editor.dirty and editor.reports[-1].get("saved"), editor.reports[-1:]
            import glossary_document as gdoc

            saved = gdoc.parse_glossary_file(str(path), config, None)["current_data"]
            assert next(e for e in saved if e["raw_name"] == "레온")["translated_name"] == "Leon Ashford"
            tracker = json.loads(tracker_path.read_text(encoding="utf-8"))["entries"]
            assert {raw: tracker[raw]["decision"] for raw in DECISIONS} == DECISIONS

            # ---- 320 dp: the rows fit, in the normal and the selection layout ----------------------------------
            assert _fit_problems(editor, PHONE[0]) == []
            assert _layout_problems(page) == []
            spec = editor.spec_for_key(next(k for k in editor.windowed.controls if _raw_of(editor, k) == "마도공학"))
            editor.on_row_long_press(spec)  # selection mode: the check icon joins every row
            assert await _settle(lambda: editor.selecting and not getattr(editor, "_fill_queue", None), 10)
            assert sorted(await _row_buttons(tester, editor, "Edit")) == sorted(e["raw_name"] for e in ENTRIES)
            assert sorted(await _row_buttons(tester, editor, menu_label)) == sorted(CONFLICTS)
            assert _fit_problems(editor, PHONE[0]) == []
            assert _layout_problems(page) == []
            editor.exit_selection()
        finally:
            try:
                app.jobs.close()
            finally:
                await tf._stop(app)

    asyncio.run(body())
    I12._assert_offline(offline)
    assert sorted(os.listdir(SRC_DIR)) == src_listing, "the test wrote into the checkout's src/"
    assert _md5(SRC_DIR / "config.json") == config_md5

    # The desktop Glossary Manager's own calls for the same two choices, on the same files as the phone had them.
    mobile = results["mobile"]
    for file, content in original.items():
        file.write_bytes(content)
    desktop = _desktop_editor(str(path), results["config"])
    for decision in YUNA_STEPS:
        desktop.decide("유나", decision)
    desktop.decide("에델바이스", DECISIONS["에델바이스"])
    assert desktop.save()
    assert mobile[path].decode("utf-8") == path.read_text(encoding="utf-8"), "the saved glossary differs from the desktop's"
    assert _tracker_view(mobile[tracker_path]) == _tracker_view(tracker_path.read_bytes()), \
        "the saved gender tracker differs from the desktop's"
    assert {raw: _tracker_view(mobile[tracker_path])["entries"][raw]["decision"] for raw in DECISIONS} == DECISIONS
