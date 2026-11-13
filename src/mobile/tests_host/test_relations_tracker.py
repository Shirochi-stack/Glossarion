"""Relations Tracker on Glossarion Mobile (plan section 6).

Glossary › General: the schema rows (above Skip Gender Tracking, the dropdown locked while the toggle is
off), the toggle's implied Custom Fields write, the prompt check (decline -> the warning notice with the
copyable line; accept -> the prompt reset to default), the locked relations row of the Custom Fields
editor, the placeholder chip lists, EntrySheet's columns, and the relationship diagram (slider, node
history sheet, empty state). Every rule comes from the shared ``relations_tracking`` module.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
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


pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed"),
    pytest.mark.skipif(not _has("relations_tracking"), reason="relations_tracking not importable"),
]

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_relations",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)

TOGGLE = "glossary_enable_relations_tracker"
MODE = "glossary_relations_history_mode"
FIELD = "glossary_relations_field_name"
FIELDS = "custom_glossary_fields"
BALANCED = "manual_glossary_prompt3"
MINIMAL = "unified_auto_glosary_prompt3"


@pytest.fixture
def iso(tmp_path, monkeypatch):
    home, library, output = tmp_path / "home", tmp_path / "Library", tmp_path / "Output"
    for folder in (home, library, output):
        folder.mkdir()
    for name, value in (("HOME", home), ("USERPROFILE", home), ("GLOSSARION_LIBRARY_DIR", library),
                        ("OUTPUT_DIRECTORY", output)):
        monkeypatch.setenv(name, str(value))
    for name in ("GLOSSARY_ENABLE_RELATIONS_TRACKER", "GLOSSARY_RELATIONS_FIELD", "GLOSSARY_RELATIONS_HISTORY_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    return types.SimpleNamespace(root=tmp_path, home=home, library=library, output=output)


def _schema():
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    if not schema.available:
        pytest.skip(f"settings schema unavailable: {schema.error}")
    return schema


def _store(tmp_path: Path, schema, config=None):
    from glossarion_mobile.state.config_store import MobileConfigStore

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config or {}), encoding="utf-8")
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk),
                                                                                        encoding="utf-8"))
    store.load()
    return store


def _ctx(page, store, schema, **kwargs):
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.glossary.common import GlossaryContext
    from glossarion_mobile.ui.settings.context import SettingsContext

    service = GlossaryService(config=store)
    settings = SettingsContext(page=page, store=store, schema=schema)
    copied: list = []
    overlays: list = []
    ctx = GlossaryContext(service=service, page=page, platform="android", settings=settings,
                          notify=lambda message, action=None, on_action=None: None,
                          copy_text=lambda text: copied.append(text), push_overlay=overlays.append,
                          pop_overlay=lambda: overlays.pop() if overlays else None, extras={}, **kwargs)
    ctx.copied, ctx.overlays = copied, overlays
    return ctx


def _mount(page, body):
    page.views[0].controls.append(body)
    page.update()


def _open_dialogs(page) -> list:
    return [d for d in (getattr(getattr(page, "_dialogs", None), "controls", None) or []) if getattr(d, "open", False)]


async def _settle(rounds: int = 6) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)


# ==========================================================================
# Schema rows and the implied write
# ==========================================================================


def test_schema_rows_sit_above_skip_gender_tracking_and_the_dropdown_follows_the_toggle(iso, tmp_path):
    from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsTab

    schema = _schema()
    toggle, mode = schema.spec(TOGGLE), schema.spec(MODE)
    assert toggle.label == "Enable Relations Tracker" and toggle.type == "bool"
    assert mode.type == "choice" and [v for v, _l in mode.choices] == ["previous", "full", "current"]
    assert dict(mode.choices)["previous"] == "Previous one only"
    assert toggle.section == mode.section == "glossary.general"
    env = {(name, site) for name, site, *_rest in (tuple(b) for b in schema.spec(TOGGLE).env)} \
        if toggle.env and isinstance(toggle.env[0], (tuple, list)) else None
    if env is not None:
        assert ("GLOSSARY_ENABLE_RELATIONS_TRACKER", "translation") in env
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        tab = GlossarySettingsTab(ctx, "general")
        _mount(session.page, tab.build())
        tab.did_show()
        keys = tab.keys()
        at = keys.index("glossary_skip_gender_tracking")
        assert keys[at - 2:at] == [TOGGLE, MODE]
        mode_tile = tab.page.tiles[MODE]
        assert mode_tile.lock_reason == "Turn on Enable Relations Tracker first"
        tab.page.tiles[TOGGLE].apply(True)
        await _settle()
        assert store.get(TOGGLE) is True
        mode_tile.refresh(push=False)
        assert not mode_tile.lock_reason
        tab.dispose()
        store._saver.close()

    asyncio.run(scenario())


def test_toggle_write_adds_and_removes_the_relations_field(iso, tmp_path):
    from glossarion_mobile.state.setting_writes import relations_tracker_writes, write_setting

    schema = _schema()
    store = _store(tmp_path, schema)
    changed = write_setting(store, TOGGLE, True)
    assert store.get(TOGGLE) is True and store.get(FIELDS) == ["description", "relations"]
    assert FIELDS in changed
    write_setting(store, TOGGLE, False)
    assert store.get(TOGGLE) is False and store.get(FIELDS) == ["description"]
    # a renamed column and a list without "description"
    store.set_many({FIELD: "Ties", FIELDS: ["notes"], "custom_field_description_removed": True})
    write_setting(store, TOGGLE, True)
    assert store.get(FIELDS) == ["notes", "Ties"]
    write_setting(store, TOGGLE, False)
    assert store.get(FIELDS) == ["notes"]
    # nothing stored and the toggle off: only the toggle is written
    assert relations_tracker_writes({}, False) == {TOGGLE: False}
    store._saver.close()


def test_turning_on_keeps_a_users_own_relations_column(iso, tmp_path):
    from glossarion_mobile.state import setting_writes
    from glossarion_mobile.state.setting_writes import relations_field_name, write_setting

    rt = importlib.import_module("relations_tracking")
    assert setting_writes.RELATIONS_RESERVED_FIELDS == rt.RESERVED_FIELD_NAMES
    schema = _schema()
    store = _store(tmp_path, schema, {FIELDS: ["description", "relations"]})
    write_setting(store, TOGGLE, True)
    assert store.get(FIELDS) == ["description", "relations", "relations_tracker"]
    assert store.get(FIELD) == "relations_tracker"
    write_setting(store, TOGGLE, True)  # already on: nothing renamed again
    assert store.get(FIELDS) == ["description", "relations", "relations_tracker"]
    write_setting(store, TOGGLE, False)
    assert store.get(FIELDS) == ["description", "relations"]
    # a stored name that is a real glossary column is never used
    assert relations_field_name({FIELD: "gender"}) == "relations"
    store._saver.close()


def test_locked_row_cannot_take_a_glossary_column_name(iso, tmp_path):
    schema = _schema()
    store = _store(tmp_path, schema, {TOGGLE: True, FIELDS: ["relations"],
                                      "custom_field_description_removed": True})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        editor = _custom_fields_tile(ctx, store).activate()
        for name in ("gender", "raw_name", "translated_name", "type", "description"):
            editor._fields[0].value = name
            assert editor.save() is False and "built-in glossary column" in editor.error_text.value
        assert store.get(FIELDS) == ["relations"] and store.get(FIELD) in (None, "relations")
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# The prompt check under the toggle
# ==========================================================================


def _general_tab(ctx):
    from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsTab

    tab = GlossarySettingsTab(ctx, "general")
    return tab


def test_declining_the_reset_shows_the_notice_with_the_copyable_line(iso, tmp_path):
    schema = _schema()
    store = _store(tmp_path, schema, {BALANCED: "My own prompt {language} {fields}"})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        ctx.extras["answers"] = [False]
        tab = _general_tab(ctx)
        _mount(session.page, tab.build())
        tab.did_show()
        panel = tab.relations
        assert panel is not None and panel.warning.visible is False  # the toggle is off
        assert tab.page.after_tiles[TOGGLE] is panel.control
        assert panel.control in tab.page.list_view.controls
        tab.page.tiles[TOGGLE].apply(True)
        await _settle()
        asked = ctx.extras.get("asked") or []
        assert len(asked) == 1 and "Balanced/Full" in asked[0][2] and "{relations_instruction}" in asked[0][2]
        assert panel.missing == ["balanced_full"] and panel.warning.visible is True
        assert "Balanced/Full prompt is missing the relations line" in panel.notice_text.value
        text = next(c for c in panel.copy_row.content.controls if isinstance(c, type(panel.notice_text)))
        assert text.value == "- {relations_instruction}" and text.selectable
        copy = panel.copy_row.content.controls[-1]
        copy.on_click(None)
        assert ctx.copied == ["- {relations_instruction}"]
        assert store.get(BALANCED) == "My own prompt {language} {fields}"  # kept
        # the line added by hand: the notice goes away; the toggle off hides it too
        store.set(BALANCED, "My own prompt\n- {relations_instruction}")
        await _settle()
        assert panel.warning.visible is False
        store.set(BALANCED, "Again without it")
        await _settle()
        assert panel.warning.visible is True
        tab.page.tiles[TOGGLE].apply(False)
        await _settle()
        assert panel.warning.visible is False and panel.diagram_button.disabled is True
        # rebuilt with the toggle on: recomputed when the tab is built (no new question)
        tab.dispose()
        store.set(TOGGLE, True)
        again = _general_tab(ctx)
        _mount(session.page, again.build())
        again.did_show()
        await _settle()
        assert again.relations.warning.visible is True and len(ctx.extras["asked"]) == 1
        again.dispose()
        store._saver.close()

    asyncio.run(scenario())


def test_accepting_the_reset_restores_the_default_prompt(iso, tmp_path):
    schema = _schema()
    store = _store(tmp_path, schema, {BALANCED: "Custom balanced", MINIMAL: "Custom minimal"})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        ctx.extras["answers"] = [False, True]
        tab = _general_tab(ctx)
        _mount(session.page, tab.build())
        tab.did_show()
        tab.page.tiles[TOGGLE].apply(True)
        await _settle()
        await tab.relations.check_task
        # both prompts lack the line: ONE question that names both
        asked = ctx.extras["asked"]
        assert [title for _k, title, _b in asked] == ["Relations Tracker"]
        assert "The Balanced/Full and Minimal prompts have no {relations_instruction} line" in asked[0][2]
        assert asked[0][2].endswith("Reset both prompts to default?")
        # declined: both kept, both named in the notice
        assert store.get(BALANCED) == "Custom balanced" and store.get(MINIMAL) == "Custom minimal"
        assert tab.relations.missing == ["balanced_full", "minimal"]
        assert "Balanced/Full and Minimal prompts are missing" in tab.relations.notice_text.value
        # off and on again, accepted: the one answer resets both
        tab.page.tiles[TOGGLE].apply(False)
        await _settle()
        tab.page.tiles[TOGGLE].apply(True)
        await _settle()
        await tab.relations.check_task
        assert len(ctx.extras["asked"]) == 2
        # allow_import: the built-in text is a lazy default (None until its module is imported)
        assert store.get(BALANCED) == schema.effective_default(BALANCED, allow_import=True)
        assert store.get(MINIMAL) == schema.effective_default(MINIMAL, allow_import=True)
        assert "{relations_instruction}" in store.get(BALANCED) and "{relations_instruction}" in store.get(MINIMAL)
        assert tab.relations.missing == [] and tab.relations.warning.visible is False
        tab.dispose()
        store._saver.close()

    asyncio.run(scenario())


def test_the_on_device_profile_is_never_offered_a_reset(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import RelationsPanel

    schema = _schema()
    store = _store(tmp_path, schema, {BALANCED: "tiny prompt", TOGGLE: True,
                                      "active_glossary_prompt_profiles": {"balanced_full": "On-device"}})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        panel = RelationsPanel(ctx)
        missing = await panel.check_prompts()
        assert missing == ["balanced_full"] and not ctx.extras.get("asked")
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# Custom Fields: the locked relations row
# ==========================================================================


def _custom_fields_tile(ctx, store):
    from glossarion_mobile.ui.settings.tiles import CustomFieldsTile, make_tile

    tile = make_tile(ctx.settings.schema.spec(FIELDS), ctx.settings)
    assert isinstance(tile, CustomFieldsTile)
    return tile


def _icons(row):
    import flet as ft

    content = row.content if isinstance(row, ft.Container) else row
    return [getattr(c, "icon", None) for c in content.controls if isinstance(c, ft.IconButton)]


def test_locked_row_has_no_delete_and_renames_write_the_field_name(iso, tmp_path):
    import flet as ft

    schema = _schema()
    store = _store(tmp_path, schema, {TOGGLE: True, FIELDS: ["description", "relations"]})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        tile = _custom_fields_tile(ctx, store)
        editor = tile.activate()
        rows = editor.rows.controls
        assert editor.locked_index == 1
        assert ft.Icons.DELETE_OUTLINE in _icons(rows[0])
        assert rows[1].key == "row-locked" and ft.Icons.DELETE_OUTLINE not in _icons(rows[1])
        assert rows[1].bgcolor  # amber tint
        editor.remove(1)
        assert editor.items == ["description", "relations"]  # not removable
        # blank and duplicate names are refused
        editor._fields[1].value = "  "
        assert editor.save() is False and "needs a name" in editor.error_text.value
        editor._fields[1].value = "Description"
        assert editor.save() is False and "already a field" in editor.error_text.value
        assert store.get(FIELDS) == ["description", "relations"]
        editor._fields[1].value = "Ties"
        assert editor.save() is True
        assert store.get(FIELDS) == ["description", "Ties"] and store.get(FIELD) == "Ties"
        # reorder: the lock follows the row; the name stays
        moved = _custom_fields_tile(ctx, store).activate()
        assert moved.locked_index == 1
        moved.move(1, -1)
        assert moved.items == ["Ties", "description"] and moved.locked_index == 0
        assert moved.rows.controls[0].key == "row-locked"
        assert moved.save() is True
        assert store.get(FIELDS) == ["Ties", "description"] and store.get(FIELD) == "Ties"
        # the toggle off: a plain list again (the row can be removed)
        store.set(TOGGLE, False)
        plain = _custom_fields_tile(ctx, store).activate()
        assert plain.locked_index is None and all(ft.Icons.DELETE_OUTLINE in _icons(r) for r in plain.rows.controls)
        store._saver.close()

    asyncio.run(scenario())


def test_placeholder_chips_and_entry_sheet_columns(iso, tmp_path):
    from glossarion_mobile.services.glossary import CORE_MODULES, PROMPT_PLACEHOLDERS
    from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet, offered_fields

    assert "{relations_instruction}" in PROMPT_PLACEHOLDERS["balanced_full"]
    assert "{relations_instruction}" in PROMPT_PLACEHOLDERS["minimal"]
    assert "relations_tracking" in CORE_MODULES
    schema = _schema()
    store = _store(tmp_path, schema, {TOGGLE: True})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        fields = ["type", "raw_name", "translated_name", "gender", "Relations", "notes"]
        assert offered_fields(ctx, fields) == ["type", "raw_name", "translated_name", "gender", "notes"]
        sheet = EntrySheet(ctx, fields=fields, values={"raw_name": "A"})
        assert "Relations" not in sheet.fields
        store.set(TOGGLE, False)  # off: a user's own "relations" column is an ordinary field
        assert offered_fields(ctx, fields) == fields
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# The relationship diagram
# ==========================================================================


def _tracker():
    rt = importlib.import_module("relations_tracking")
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [
        {"raw_name": "서윤", "translated_name": "Seo-yun", "relations": "하린: Best Friend"},
        {"raw_name": "하린", "translated_name": "Ha-rin", "relations": "서윤: Best Friend"},
    ], field="relations", chapter_index=0, chapter_num=1, chapter_file="ch1.html")
    # The model writes, on X's row, what X is to the other character (stored on the other's entry).
    rt.record_relations(tracker, [
        {"raw_name": "서윤", "translated_name": "Seo-yun", "relations": "민준: Sister"},
        {"raw_name": "하린", "translated_name": "Ha-rin", "relations": "서윤: Enemy"},
        {"raw_name": "민준", "translated_name": "Min-jun", "relations": "서윤: Brother"},
    ], field="relations", chapter_index=4, chapter_num=5, chapter_file="ch5.html")
    return tracker


def test_diagram_builds_from_a_tracker_and_the_slider_changes_the_edges(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram, edge_label_text

    schema = _schema()
    store = _store(tmp_path, schema)
    entries = [{"raw_name": "서윤", "translated_name": "Seo-yun", "gender": "female"},
               {"raw_name": "하린", "translated_name": "Ha-rin", "gender": "female"},
               {"raw_name": "민준", "translated_name": "Min-jun", "gender": "male"}]

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=_tracker(), entries=entries)
        assert not diagram.empty and diagram.chapter == 5 and diagram.chapter_label.value == "Up to chapter 5"
        assert diagram.edge_count() == 2 and diagram.node_count() == 3
        lines = [s for s in diagram.canvas.shapes if type(s).__name__ == "Line"]
        assert len(lines) == 2
        texts = [s.value for s in diagram.canvas.shapes if type(s).__name__ == "Text"]
        assert any("Enemy [Best Friend]" in t for t in texts)
        positions = dict(diagram.positions)
        # every pill lies inside the Stack, which clips: a node at the layout's edge was cut off
        real = diagram.core

        class _Corners:  # a layout that puts the nodes on the corners of the area it was given
            def __getattr__(self, name):
                return getattr(real, name)

            def layout_graph(self, graph, width=1000.0, height=700.0, focus=None):
                corners = [(0.0, 0.0), (width, height), (width, 0.0), (0.0, height)]
                return {n["id"]: corners[i % 4] for i, n in enumerate(graph.get("nodes", []))}

        diagram.core = _Corners()
        diagram.set_focus(None)
        assert len(diagram.node_controls) == 3
        for pill in diagram.node_controls:
            assert pill.left >= 0 and pill.top >= 0
            assert pill.left + pill.width <= diagram.width and pill.top + pill.height <= diagram.height
        diagram.core = real
        diagram.set_focus(None)
        assert diagram.positions == positions
        diagram.set_chapter(1)
        assert diagram.edge_count() == 1 and diagram.chapter_label.value == "Up to chapter 1"
        assert diagram.positions == positions  # nodes stay put while the slider moves
        diagram.set_chapter(5)
        diagram.set_former(False)
        assert all("[" not in edge_label_text(e, False) for e in diagram.drawn_edges)
        diagram.set_focus(next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "서윤"))
        assert diagram.edge_count() == 2
        # tapping a name opens its relation history
        node = next(c for c in diagram.node_controls if c.key.endswith("서윤") or "Seo-yun" in c.content.value)
        node.on_click(None)
        sheet = diagram.history_sheet
        assert sheet is not None and sheet.key == "rel-history-sheet" and _open_dialogs(session.page)[-1] is sheet
        rows = diagram.history_rows(next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "서윤"))
        assert {r["current"] for r in rows} == {"Enemy", "Brother"}
        node.on_click(None)  # again: the previous sheet is closed first, one live sheet per key
        await _settle()
        live = [d for d in _open_dialogs(session.page) if d.key == "rel-history-sheet"]
        assert len(live) == 1 and live[0] is diagram.history_sheet
        store._saver.close()

    asyncio.run(scenario())


def test_history_sheet_shows_the_description_written_at_the_sliders_chapter(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt = importlib.import_module("relations_tracking")
    tracker = _tracker()
    entries = [{"raw_name": "서윤", "translated_name": "Seo-yun", "description": "The glossary's present text"},
               {"raw_name": "하린", "translated_name": "Ha-rin", "description": "A healer, as the glossary has it"},
               {"raw_name": "민준", "translated_name": "Min-jun"}]

    def texts(control):
        found = [control.value] if type(control).__name__ == "Text" else []
        for child in getattr(control, "controls", None) or []:
            found += texts(child)
        return found

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        # A tracker from before descriptions were recorded: the glossary's own, said to be undated.
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=tracker, entries=entries)
        seo = next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "서윤")
        written, earlier, source = diagram.description_rows(seo)
        assert (written["text"], earlier, source) == ("The glossary's present text", [], "glossary")
        for chapter, text in ((1, "The heir, first seen"), (5, "The heir who lost her friend")):
            rt.record_relations(tracker, [{"type": "character", "raw_name": "서윤", "description": text}],
                                chapter_index=chapter - 1, chapter_num=chapter, chapter_file=f"c{chapter}.xhtml")
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=tracker, entries=entries)
        assert diagram.node_count() == 3  # a name kept only for its descriptions adds no node
        sheet = diagram.show_history(seo)
        shown = [t for c in sheet.content.controls for t in texts(c)] if hasattr(sheet.content, "controls") else []
        column = diagram._description_controls(seo)[0]
        assert column.key == "rel-history-description"
        assert texts(column) == ["Description written at chapter 5", "The heir who lost her friend",
                                 "ch 1: The heir, first seen"]
        assert not shown or "The heir who lost her friend" in shown
        # 하린 has none recorded: the glossary's; 민준 has none anywhere: no description block
        harin = next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "하린")
        assert texts(diagram._description_controls(harin)[0])[1] == "A healer, as the glossary has it"
        minjun = next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "민준")
        assert diagram._description_controls(minjun) == []
        diagram.set_chapter(3)
        assert texts(diagram._description_controls(seo)[0]) == ["Description written at chapter 1",
                                                                "The heir, first seen"]
        store._saver.close()

    asyncio.run(scenario())


def test_diagram_without_a_tracker_is_an_empty_state_and_opens_as_an_overlay(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        assert diagram.empty and diagram.body.key == "rel-empty"
        assert ctx.overlays == [diagram.view]
        diagram.close()
        assert ctx.overlays == []
        # a tracker beside the glossary: the diagram has its relations
        rt = importlib.import_module("relations_tracking")
        Path(rt.tracker_path_for_glossary(str(glossary))).write_text(json.dumps(_tracker(), ensure_ascii=False),
                                                                     encoding="utf-8")
        diagram = await open_diagram(ctx, str(glossary))
        assert not diagram.empty and diagram.edge_count() == 0  # names not in the glossary are unlisted
        diagram.set_unlisted(True)
        assert diagram.edge_count() == 2
        glossary.write_text("type,raw_name,translated_name,gender\ncharacter,서윤,Seo-yun,female\n"
                            "character,하린,Ha-rin,female\ncharacter,민준,Min-jun,male\n", encoding="utf-8")
        diagram = await open_diagram(ctx, str(glossary))
        assert diagram.edge_count() == 2 and diagram.entries and len(ctx.overlays) == 2
        store._saver.close()

    asyncio.run(scenario())
