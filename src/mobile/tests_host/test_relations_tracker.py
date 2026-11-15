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
import math
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
MINIMAL_TOGGLE = "glossary_minimal_include_relations"


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
    # the Minimal prompt is checked only while Minimal's "Include Relations Column" is on
    store = _store(tmp_path, schema, {BALANCED: "Custom balanced", MINIMAL: "Custom minimal", MINIMAL_TOGGLE: True})

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


def test_minimal_include_relations_row_sits_under_include_description_and_needs_the_tracker(iso, tmp_path):
    from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsTab

    schema = _schema()
    spec = schema.spec(MINIMAL_TOGGLE)
    assert spec.label == "Include Relations Column" and spec.type == "bool"
    assert spec.section == "glossary.minimal"
    assert schema.effective_default(MINIMAL_TOGGLE) is False
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        tab = GlossarySettingsTab(ctx, "minimal")
        _mount(session.page, tab.build())
        tab.did_show()
        keys = tab.keys()
        at = keys.index("include_description")
        assert keys[at + 1] == MINIMAL_TOGGLE
        tile = tab.page.tiles[MINIMAL_TOGGLE]
        assert tile.lock_reason == "Turn on Enable Relations Tracker first"
        store.set(TOGGLE, True)
        await _settle()
        tile.refresh(push=False)
        assert not tile.lock_reason
        tab.dispose()
        store._saver.close()

    asyncio.run(scenario())


def test_the_minimal_prompt_is_checked_only_with_its_toggle_on(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import RelationsPanel, missing_prompts

    schema = _schema()
    store = _store(tmp_path, schema, {BALANCED: "Custom balanced\n- {relations_instruction}",
                                      MINIMAL: "Custom minimal", TOGGLE: True})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        ctx.extras["answers"] = [False]
        panel = RelationsPanel(ctx)
        panel.attach()
        assert missing_prompts(ctx) == [] and panel.warning.visible is False
        # turned on with the tracker on: the Minimal prompt is checked (one question) and named
        store.set(MINIMAL_TOGGLE, True)
        await _settle()
        await panel.check_task
        asked = ctx.extras["asked"]
        assert len(asked) == 1 and "The Minimal prompt has no {relations_instruction} line" in asked[0][2]
        assert panel.missing == ["minimal"] and panel.warning.visible is True
        assert store.get(MINIMAL) == "Custom minimal"  # declined: kept
        store.set(MINIMAL_TOGGLE, False)
        await _settle()
        assert panel.missing == [] and panel.warning.visible is False
        panel.detach()
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


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_the_chapter_number_box_sets_the_exact_chapter(iso, tmp_path, platform):
    """Owner: the slider is too coarse for a long book, so the chapter can be typed or stepped."""
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session(platform)
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=_tracker(), entries=None)
        field, slider = diagram.chapter_field, diagram.slider
        row = diagram.diagram_body.controls[0]
        assert row.controls == [diagram.chapter_label, diagram.chapter_minus, field, diagram.chapter_plus]
        assert diagram.diagram_body.controls[1].controls == [slider]
        assert (slider.min, slider.max, field.value, field.disabled) == (1, 5, "5", False)
        # typing a chapter and pressing Done moves the slider and redraws
        field.on_focus(None)
        field.value = "1"
        field.on_submit(None)
        assert diagram.chapter == 1 and slider.value == 1 and diagram.edge_count() == 1
        field.on_blur(None)  # the blur after Done commits nothing more
        assert diagram.chapter == 1
        # a tap outside commits too
        field.on_focus(None)
        field.value = "3"
        field.on_tap_outside(None)
        assert diagram.chapter == 3 and slider.value == 3
        # a number outside the tracked chapters is held to them; the box shows what was applied
        for typed, applied in (("2000", 5), ("0", 1), ("-4", 1), ("2.6", 3)):
            field.on_focus(None)
            field.value = typed
            field.on_blur(None)
            assert (diagram.chapter, slider.value, field.value) == (applied, applied, str(applied)), typed
        # not a number: the chapter stays and the box shows it again
        for typed in ("abc", "", "nan", "inf"):
            field.on_focus(None)
            field.value = typed
            field.on_blur(None)
            assert (diagram.chapter, field.value) == (3, "3"), typed
        # − and + step one chapter and stop at the ends
        diagram.chapter_minus.on_click(None)
        assert (diagram.chapter, field.value, slider.value) == (2, "2", 2)
        diagram.chapter_minus.on_click(None)
        diagram.chapter_minus.on_click(None)
        assert diagram.chapter == 1 and diagram.edge_count() == 1
        for _ in range(6):
            diagram.chapter_plus.on_click(None)
        assert (diagram.chapter, field.value) == (5, "5") and diagram.edge_count() == 2
        # dragging the slider updates the box
        slider.value = 2.0
        slider.on_change(types.SimpleNamespace(control=slider))
        assert field.value == "2"
        slider.on_change_end(types.SimpleNamespace(control=slider))
        assert diagram.chapter == 2 and field.value == "2"
        # a number being typed survives a redraw from new file contents
        field.on_focus(None)
        field.value = "4"
        diagram._apply_state(tracker=_tracker())
        assert field.value == "4" and diagram.chapter == 2
        field.on_blur(None)
        assert diagram.chapter == 4 and slider.value == 4
        store._saver.close()

    asyncio.run(scenario())


def test_the_chapter_number_box_is_off_for_a_single_chapter(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=_tracker(), entries=None)
        rt = diagram.core
        one = rt.empty_tracker()
        rt.record_relations(one, [{"raw_name": "서윤", "translated_name": "Seo-yun", "relations": "하린: Friend"},
                                  {"raw_name": "하린", "translated_name": "Ha-rin", "relations": "서윤: Friend"}],
                            field="relations", chapter_index=2, chapter_num=3, chapter_file="ch3.html")
        diagram._apply_state(tracker=one)
        assert diagram.chapter == 3 and diagram.chapter_field.value == "3"
        assert diagram.chapter_field.disabled and diagram.chapter_minus.disabled and diagram.chapter_plus.disabled
        diagram.step_chapter(1)
        assert diagram.chapter == 3
        diagram._apply_state(tracker=_tracker())  # more chapters again: the box works again
        assert not diagram.chapter_field.disabled and not diagram.chapter_plus.disabled
        store._saver.close()

    asyncio.run(scenario())


def test_diagram_opens_fitted_and_keeps_the_users_zoom(iso, tmp_path):
    """Owner 2026-11-15: "the relationship diagram in mobile should be zoomed out by default": the whole canvas
    fits the viewer when the screen opens and when the layout changes (Focus, Unlisted names); the update check,
    the slider and a later size change (the keyboard) leave the zoom alone."""
    from glossarion_mobile.ui.glossary.relations_diagram import FIT_PADDING, RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=_tracker(), entries=None)
        calls: list = []

        def recorder(name):
            async def call(*args):
                calls.append((name, *args))
            return call

        for name in ("reset", "pan", "zoom"):
            object.__setattr__(diagram.viewer, name, recorder(name))
        assert diagram.fit_due and diagram.fit_all() is None  # no size yet: waits for the first one
        diagram._on_viewer_size(types.SimpleNamespace(width=560.0, height=500.0))
        await _settle()
        assert [c[0] for c in calls] == ["reset", "pan", "zoom"] and not diagram.fit_due
        scale, dx, dy = calls[2][1], calls[1][1], calls[1][2]
        width, height = diagram.width, diagram.height
        assert width * scale <= 560 - 2 * FIT_PADDING + 0.01 and height * scale <= 500 - 2 * FIT_PADDING + 0.01
        assert scale < 1  # the canvas (at least 676 x 520 dp) is zoomed out to fit
        assert abs(dx - (560 - width * scale) / 2) < 0.01 and abs(dy - (500 - height * scale) / 2) < 0.01
        # Flutter stops zooming out where canvas + boundary margin no longer covers the viewport: widened
        margin = diagram.viewer.boundary_margin.left
        assert (width + 2 * margin) * scale >= 560 and (height + 2 * margin) * scale >= 500
        calls.clear()
        diagram.set_chapter(diagram.low)  # the slider
        diagram._apply_state(tracker=_tracker())  # the once-a-second update
        diagram._on_viewer_size(types.SimpleNamespace(width=560.0, height=260.0))  # the keyboard opened
        await _settle()
        assert calls == []
        diagram.set_focus(diagram.all_graph["nodes"][0]["id"])  # a new layout: fitted again
        await _settle()
        assert [c[0] for c in calls] == ["reset", "pan", "zoom"]
        calls.clear()
        diagram.set_unlisted(True)
        await _settle()
        assert [c[0] for c in calls] == ["reset", "pan", "zoom"]
        # a huge canvas: far out, the margin grows with it
        diagram._set_canvas(6000.0, 900.0)
        scale, _dx, _dy, margin = diagram.fit_transform()
        assert scale * 6000 <= 560 - 2 * FIT_PADDING + 0.01 and (900 + 2 * margin) * scale >= 260
        store._saver.close()

    asyncio.run(scenario())


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
        assert not diagram.empty and diagram.chapter == 5 and diagram.chapter_label.value == "Up to chapter"
        assert diagram.chapter_field.value == "5"
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

            def layout_graph(self, graph, width=1000.0, height=700.0, focus=None, **_sizes):
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
        assert diagram.edge_count() == 1 and diagram.chapter_field.value == "1"
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
        assert sheet is not None and sheet.key.startswith("rel-history-sheet-") and _open_dialogs(session.page)[-1] is sheet
        rows = diagram.history_rows(next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "서윤"))
        assert {r["current"] for r in rows} == {"Enemy", "Brother"}
        # the history stops at the slider's chapter, like the picture
        seo = next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "서윤")
        diagram.set_chapter(1)
        assert {r["current"] for r in diagram.history_rows(seo)} == {"Best Friend"}
        diagram.set_chapter(5)
        node = next(c for c in diagram.node_controls if c.key.endswith("서윤") or "Seo-yun" in c.content.value)
        node.on_click(None)  # again: the previous sheet is closed first, one live sheet per key
        await _settle()
        live = [d for d in _open_dialogs(session.page) if (d.key or "").startswith("rel-history-sheet-")]
        assert len(live) == 1 and live[0] is diagram.history_sheet
        store._saver.close()

    asyncio.run(scenario())


def test_diagram_hub_labels_do_not_overlap(iso, tmp_path):
    """The owner's book: a main character with dozens of relations made a knot of labels around it.
    Pure geometry of what the screen draws: the name pills and the label boxes (``label_box`` of the
    text at the point drawn) overlap nothing, each label sits ON its line, and the canvas holds every
    pill, also with that character focused."""
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram, label_box

    schema = _schema()
    store = _store(tmp_path, schema)
    rt = importlib.import_module("relations_tracking")
    tracker = rt.empty_tracker()
    words = [("Patient", "Healer"), ("Younger Brother", "Older Sister"), ("Student", "Mentor"), ("Rival", "Rival")]
    rows = [{"raw_name": f"Person {i}", "translated_name": f"Person {i}",
             "relations": f"Hero: {words[i % 4][1]}" + (f"; Person {i + 1}: Coworker" if i % 4 == 0 else "")}
            for i in range(32)]
    rows.insert(0, {"raw_name": "Hero", "translated_name": "Hero",
                    "relations": "; ".join(f"Person {i}: {words[i % 4][0]}" for i in range(32))})
    rt.record_relations(tracker, rows, field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")

    def overlaps(diagram):
        texts = [s for s in diagram.canvas.shapes if type(s).__name__ == "Text"]
        lines = [s for s in diagram.canvas.shapes if type(s).__name__ == "Line"]
        assert len(texts) == len(lines) == 40
        for text in texts:  # ON a line: on the segment between two drawn names
            assert any(rt._point_segment_distance((text.x, text.y), (l.x1, l.y1), (l.x2, l.y2)) < 1e-6
                       for l in lines)
        labels = []
        for text in texts:
            w, h = label_box(text.value)
            labels.append((text.x - w / 2, text.y - h / 2, text.x + w / 2, text.y + h / 2))
        names = [(c.left, c.top, c.left + c.width, c.top + c.height) for c in diagram.node_controls]
        for left, top, right, bottom in names:
            assert left >= 0 and top >= 0 and right <= diagram.width and bottom <= diagram.height

        def count(first, second, same):
            return sum(1 for i, a in enumerate(first) for j, b in enumerate(second)
                       if (not same or j > i) and min(a[2], b[2]) - max(a[0], b[0]) > 0.5
                       and min(a[3], b[3]) - max(a[1], b[1]) > 0.5)
        return count(labels, labels, True), count(labels, names, False), count(names, names, True)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=tracker, entries=None)
        assert diagram.node_count() == 33
        assert overlaps(diagram) == (0, 0, 0)
        diagram.set_focus(rt.tracker_key("Hero"))
        assert overlaps(diagram) == (0, 0, 0)
        store._saver.close()

    asyncio.run(scenario())


def _line_gaps(diagram):
    """From what the screen draws (canvas lines, label texts sized by ``label_box``, name pills):
    ``(labels a line that is not their own comes within 6 px of, beyond its half width; names a line
    not ending there comes that near; labels drawn)``. Own geometry, not the layout's bookkeeping."""
    from glossarion_mobile.ui.glossary.relations_diagram import label_box

    def to_segment(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
        return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)

    def meet(p1, p2, p3, p4):
        def side(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (side(p3, p4, p1) > 0) != (side(p3, p4, p2) > 0) and (side(p1, p2, p3) > 0) != (side(p1, p2, p4) > 0)

    def gap(a, b, box):
        x0, y0, x1, y1 = box
        if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (a, b)):
            return 0.0
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if any(meet(a, b, corners[i], corners[(i + 1) % 4]) for i in range(4)):
            return 0.0
        return min([to_segment(c, a, b) for c in corners]
                   + [math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1)) for p in (a, b)])

    lines = [((s.x1, s.y1), (s.x2, s.y2), (s.paint.stroke_width or 1.4) / 2)
             for s in diagram.canvas.shapes if type(s).__name__ == "Line"]
    texts = [s for s in diagram.canvas.shapes if type(s).__name__ == "Text"]
    touched = 0
    for text in texts:
        own = [i for i, (a, b, _half) in enumerate(lines) if to_segment((text.x, text.y), a, b) < 1e-6]
        assert len(own) == 1  # ON its line
        w, h = label_box(text.value)
        box = (text.x - w / 2, text.y - h / 2, text.x + w / 2, text.y + h / 2)
        touched += any(gap(a, b, box) - half < 6.0 for i, (a, b, half) in enumerate(lines) if i != own[0])
    crossed = 0
    for pill in diagram.node_controls:
        box = (pill.left, pill.top, pill.left + pill.width, pill.top + pill.height)
        centre = (pill.left + pill.width / 2, pill.top + pill.height / 2)
        crossed += any(gap(a, b, box) - half < 6.0 for a, b, half in lines
                       if math.dist(a, centre) > 1e-6 and math.dist(b, centre) > 1e-6)
    return touched, crossed, len(texts)


def test_diagram_lines_stay_clear_of_other_labels_and_names(iso, tmp_path):
    """The owner's "the text boxes here don't touch other lines" on the phone: no relation line comes
    within 6 px of another relation's label or of a name it does not end at, in the opening view,
    without former relations, at an earlier chapter and with the main character focused."""
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt = importlib.import_module("relations_tracking")
    tracker = rt.empty_tracker()
    words = [("Patient", "Healer"), ("Younger Brother", "Older Sister"), ("Student", "Mentor"), ("Rival", "Rival")]
    rows = [{"raw_name": f"Person {i}", "translated_name": f"Person {i}",
             "relations": f"Hero: {words[i % 4][1]}" + (f"; Person {i + 1}: Coworker" if i % 4 == 0 else "")}
            for i in range(32)]
    rows.insert(0, {"raw_name": "Hero", "translated_name": "Hero",
                    "relations": "; ".join(f"Person {i}: {words[i % 4][0]}" for i in range(32))})
    rt.record_relations(tracker, rows, field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    rt.record_relations(tracker, [{"raw_name": "Hero", "translated_name": "Hero",
                                   "relations": "; ".join(f"Person {i}: Sworn Enemy" for i in range(0, 32, 3))}],
                        field="relations", chapter_index=1, chapter_num=2, chapter_file="c2.xhtml")

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=tracker, entries=None)
        positions = dict(diagram.positions)
        assert _line_gaps(diagram) == (0, 0, 40)
        diagram.set_former(False)
        assert _line_gaps(diagram) == (0, 0, 40)
        diagram.set_chapter(1)
        assert diagram.positions == positions and _line_gaps(diagram) == (0, 0, 40)
        diagram.set_chapter(2)
        diagram.set_focus(rt.tracker_key("Hero"))
        focused = dict(diagram.positions)
        assert _line_gaps(diagram) == (0, 0, 40)
        diagram.set_former(True)
        diagram.set_chapter(1)
        assert diagram.positions == focused and _line_gaps(diagram) == (0, 0, 40)
        store._saver.close()

    asyncio.run(scenario())


def test_diagram_above_the_label_cap_labels_only_the_focus_character(iso, tmp_path):
    """Above ``ROOM_FOR_LABELS_MAX_EDGES`` lines the layout makes no room for labels, so (as on the
    desktop) only the Focus character's lines keep theirs, instead of a pile of text."""
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt = importlib.import_module("relations_tracking")
    tracker = rt.empty_tracker()
    count = rt.ROOM_FOR_LABELS_MAX_EDGES + 1
    rows = [{"raw_name": f"Spoke {i}", "translated_name": f"Spoke {i}", "relations": "Hub: Mentor"}
            for i in range(count)]
    rows[0]["relations"] += "; Spoke 1: Rival"
    rows.insert(0, {"raw_name": "Hub", "translated_name": "Hub",
                    "relations": "; ".join(f"Spoke {i}: Student" for i in range(count))})
    rt.record_relations(tracker, rows, field="relations", chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")

    def labels(diagram):
        return [s for s in diagram.canvas.shapes if type(s).__name__ == "Text"]

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=tracker, entries=None)
        assert diagram.edge_count() == count + 1
        assert labels(diagram) == [] and "Focus character only" in diagram.summary.value
        diagram.set_focus(rt.tracker_key("Spoke 0"))
        assert len(labels(diagram)) == 2  # its lines to Hub and to Spoke 1
        # the Hub alone has more lines than the cap: the layout made no room for its labels, so
        # none are drawn (drawn anyway, some sat on other lines)
        diagram.set_focus(rt.tracker_key("Hub"))
        assert not diagram.layout.label_spots and labels(diagram) == []
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
        assert texts(column) == ["Description written at chapter 5", "The heir who lost her friend"]
        assert not shown or "The heir who lost her friend" in shown
        # the earlier ones sit behind "Earlier descriptions (1)", collapsed
        toggle_line, listing = diagram._earlier_controls(seo)
        assert toggle_line.content == "Earlier descriptions (1)" and listing.visible is False
        assert texts(listing) == ["ch 1: The heir, first seen"]
        # 하린 has none recorded: the glossary's; 민준 has none anywhere: no description block
        harin = next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "하린")
        assert texts(diagram._description_controls(harin)[0])[1] == "A healer, as the glossary has it"
        minjun = next(n["id"] for n in diagram.graph["nodes"] if n["raw_name"] == "민준")
        assert diagram._description_controls(minjun) == []
        diagram.set_chapter(3)
        assert texts(diagram._description_controls(seo)[0]) == ["Description written at chapter 1",
                                                                "The heir, first seen"]
        assert diagram._earlier_controls(seo) == []  # nothing earlier: no line
        store._saver.close()

    asyncio.run(scenario())


def test_earlier_descriptions_are_collapsed_until_tapped_and_stay_open_across_updates(iso, tmp_path):
    import flet as ft

    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt = importlib.import_module("relations_tracking")
    tracker = _tracker()
    for chapter in (1, 2, 3):
        rt.record_relations(tracker, [{"type": "character", "raw_name": "서윤", "description": f"Seen at {chapter}"}],
                            chapter_index=chapter - 1, chapter_num=chapter, chapter_file=f"c{chapter}.xhtml")
    rt, glossary = _book(tmp_path, tracker=tracker)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        seo = _id(diagram, "서윤")
        sheet = diagram.show_history(seo)
        line, listing = _find(sheet, "rel-earlier-toggle"), _find(sheet, "rel-earlier-list")
        assert line.content == "Earlier descriptions (2)" and listing.visible is False
        assert "Seen at 3" in _texts(sheet) and _find(sheet, "rel-desc-edit-0")  # the current one stays
        line.on_click(None)
        assert diagram.earlier_open and listing.visible and line.icon == ft.Icons.EXPAND_LESS
        assert diagram.history_sheet is sheet  # shown in place, not rebuilt
        # another writer changes the tracker: the sheet is refreshed and the list stays open
        rt.edit_tracker(rt.tracker_path_for_glossary(str(glossary)),
                        lambda t: rt.set_description(t, "서윤", "Seen at 2, rewritten", 2, chapter_file="c2.xhtml"))
        assert await diagram.check_for_update()
        assert diagram.history_sheet is sheet and _find(sheet, "rel-earlier-list").visible
        assert "ch 2: Seen at 2, rewritten" in _texts(sheet)
        # edit an earlier one while open
        _find(sheet, "rel-desc-edit-2").on_click(None)
        assert _find(diagram.form_sheet, "rel-desc-text").value == "Seen at 1"
        _find(diagram.form_sheet, "rel-desc-text").value = "First seen"
        _find(diagram.form_sheet, "rel-desc-save").on_click(None)
        await diagram.task
        texts = {d["chapter_num"]: d["text"] for d in _saved(rt, glossary)["entries"][seo]["descriptions"]}
        assert texts[1] == "First seen"
        assert _find(diagram.history_sheet, "rel-earlier-list").visible
        # tapping again hides the list
        _find(diagram.history_sheet, "rel-earlier-toggle").on_click(None)
        assert not diagram.earlier_open and not _find(diagram.history_sheet, "rel-earlier-list").visible
        diagram.close()
        store._saver.close()

    asyncio.run(scenario())


def test_diagram_without_a_tracker_is_an_empty_state_and_opens_as_an_overlay(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import open_diagram
    from glossarion_mobile.ui.glossary.relations_diagram import EMPTY_BODY, EMPTY_BODY_ON

    schema = _schema()
    store = _store(tmp_path, schema)
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        store.set(TOGGLE, False)
        diagram = await open_diagram(ctx, str(glossary))
        assert diagram.empty and diagram.body.key == "rel-empty"
        assert EMPTY_BODY in _texts(diagram.body) and EMPTY_BODY_ON not in _texts(diagram.body)
        assert ctx.overlays == [diagram.view]
        diagram.close()
        assert ctx.overlays == []
        # the toggle already on: it does not say to turn it on
        store.set(TOGGLE, True)
        diagram = await open_diagram(ctx, str(glossary))
        assert EMPTY_BODY_ON in _texts(diagram.body) and EMPTY_BODY not in _texts(diagram.body)
        diagram.close()
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


# ==========================================================================
# Fresh configs: both toggles on, except while Sigma Q8 is selected (owner 2026-11-13)
# ==========================================================================

MINIMAL_TOGGLE = "glossary_minimal_include_relations"
FRESH = "glossary_relations_fresh_defaults"
SIGMA = "local/sigma-q8"
QWEN = "local/garbage"


@pytest.fixture
def engine_service(tmp_path, monkeypatch):
    """The app's on-device engine service with Sigma Q8, Sigma Q4 and a Qwen model downloaded."""
    from local_llm_fakes import FakePrefs
    from glossarion_mobile.services.local_llm import downloads as dl, engine, presets, service as svc

    registry = dl.Registry(str(tmp_path / "models"))
    for stem, repo, file in (("sigma-q8", presets.SIGMA_REPO, presets.SIGMA_Q8_FILE),
                             ("sigma-q4", presets.SIGMA_REPO, "gguf/sft3-Q4_K_M.gguf"),
                             ("garbage", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf")):
        registry.add(stem, {"repo": repo, "file": file, "size": 100})
        Path(registry.file_path(stem)).write_bytes(b"GGUF")
    service = svc.LocalEngineService(prefs=FakePrefs(), registry=registry, watch=False,
                                     engine_factory=engine.FakeEngine, platform="android",
                                     memory=lambda: (24 * 1024**3, 20 * 1024**3))
    monkeypatch.setattr(svc, "_DEFAULT", service)
    return service


def _plain_store(path: Path):
    from glossarion_mobile.state.config_store import MobileConfigStore

    store = MobileConfigStore(path, debounce=10,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk),
                                                                                        encoding="utf-8"))
    store.load()
    return store


def _toggles(store) -> tuple:
    return store.get(TOGGLE), store.get(MINIMAL_TOGGLE)


def test_sigma_q8_is_recognised_by_its_file(engine_service):
    from glossarion_mobile.state import setting_writes as sw

    assert sw.is_sigma_q8_model(SIGMA)
    assert not sw.is_sigma_q8_model("local/sigma-q4") and not sw.is_sigma_q8_model(QWEN)
    assert not sw.is_sigma_q8_model("authchatgpt/auto") and not sw.is_sigma_q8_model(None)
    assert sw.relations_default_for_model(None) is True and sw.relations_default_for_model(SIGMA) is False
    # owner 2026-11-15: off for every on-device model, on for cloud / remote ones
    assert sw.relations_default_for_model(QWEN) is False and sw.relations_default_for_model("local/sigma-q4") is False
    assert sw.relations_default_for_model("authchatgpt/auto") is True
    assert sw.relations_default_for_model("ollamapull/qwen3:8b") is True  # runs on the user's PC, not the phone


def test_fresh_config_follows_the_model_until_the_user_sets_a_toggle(engine_service, tmp_path):
    from glossarion_mobile.state import setting_writes as sw

    path = tmp_path / "config.json"
    store = _plain_store(path)
    assert not store.exists
    sw.apply_relations_fresh_defaults(store, fresh=True)
    # no model saved (the default is a cloud model): both on, the relations column in Custom Fields
    assert _toggles(store) == (True, True)
    assert store.get(FIELDS) == ["description", "relations"]
    assert store.get(FRESH) == [TOGGLE, MINIMAL_TOGGLE]
    assert not store.flush() and not path.exists()  # seeded like loaded values: nothing written by itself
    # the job gets the same values
    snapshot = sw.with_mobile_streaming_defaults(store.snapshot())
    assert snapshot[TOGGLE] is True and snapshot[MINIMAL_TOGGLE] is True and "relations" in snapshot[FIELDS]

    # the on-device Sigma Q8 model chosen first (e.g. on the Welcome): both off, the column removed
    sw.write_setting(store, "model", SIGMA)
    assert _toggles(store) == (False, False) and "relations" not in (store.get(FIELDS) or [])
    # another model: on again by themselves
    sw.write_setting(store, "model", "authchatgpt/auto")
    assert _toggles(store) == (True, True) and "relations" in store.get(FIELDS)
    sw.write_setting(store, "model", QWEN)  # every on-device model (owner 2026-11-15)
    assert _toggles(store) == (False, False) and "relations" not in (store.get(FIELDS) or [])
    sw.write_setting(store, "model", "authchatgpt/auto")
    assert _toggles(store) == (True, True)
    sw.write_setting(store, "model", SIGMA)
    assert _toggles(store) == (False, False)

    # the user turns the tracker on by hand on Sigma: it stays on; the Minimal toggle still follows
    sw.write_setting(store, TOGGLE, True)
    assert store.get(FRESH) == [MINIMAL_TOGGLE]
    sw.write_setting(store, "model", "authchatgpt/auto")
    assert _toggles(store) == (True, True)
    # ... and off by hand: it stays off whatever the model
    sw.write_setting(store, TOGGLE, False)
    for model in (SIGMA, "authchatgpt/auto", QWEN, SIGMA, "authchatgpt/auto"):
        sw.write_setting(store, "model", model)
        assert store.get(TOGGLE) is False
        assert store.get(MINIMAL_TOGGLE) is (not model.startswith("local/"))
    # Minimal set by hand too: nothing follows the model any more
    sw.write_setting(store, MINIMAL_TOGGLE, False)
    assert not store.has(FRESH)
    sw.write_setting(store, "model", QWEN)
    assert _toggles(store) == (False, False)
    store.flush()
    assert json.loads(path.read_text(encoding="utf-8"))[TOGGLE] is False
    store._saver.close()


def test_existing_config_without_the_keys_follows_the_model_too(engine_service, tmp_path):
    """Owner 2026-11-15 ("change the relations toggle defaults to be on. In mobile. Except for on device
    models"): an existing config whose toggles were never set (the mobile store writes keys sparsely, so they
    are absent) gets the mobile default too: seeded, so the file is not rewritten for it."""
    from glossarion_mobile.state import setting_writes as sw

    path = tmp_path / "config.json"
    original = json.dumps({"model": "authchatgpt/auto", "temperature": 0.5})
    path.write_text(original, encoding="utf-8")
    store = _plain_store(path)
    assert sorted(sw.apply_relations_fresh_defaults(store, fresh=False)) == sorted([TOGGLE, MINIMAL_TOGGLE, FIELDS])
    assert _toggles(store) == (True, True) and store.get(FIELDS) == ["description", "relations"]
    assert store.get(FRESH) == [TOGGLE, MINIMAL_TOGGLE]
    assert not store.flush() and path.read_text(encoding="utf-8") == original  # byte-identical until a real save
    for model, on in ((SIGMA, False), (QWEN, False), ("authchatgpt/auto", True)):
        sw.write_setting(store, "model", model)
        assert _toggles(store) == (on, on) and ("relations" in (store.get(FIELDS) or [])) is on
    # a second load of the saved file: nothing new (the toggles are stored now, the mark still lists them)
    store.flush()
    store._saver.close()
    again = _plain_store(path)
    assert sw.apply_relations_fresh_defaults(again, fresh=False) == [] and _toggles(again) == (True, True)
    again._saver.close()


def test_existing_config_on_an_on_device_model_starts_off(engine_service, tmp_path):
    from glossarion_mobile.state import setting_writes as sw

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": QWEN, "custom_glossary_fields": []}), encoding="utf-8")
    store = _plain_store(path)
    sw.apply_relations_fresh_defaults(store, fresh=False)
    assert _toggles(store) == (False, False) and store.get(FIELDS) == []  # no relations column
    sw.write_setting(store, "model", "authchatgpt/auto")
    assert _toggles(store) == (True, True) and "relations" in store.get(FIELDS)
    store._saver.close()


def test_a_stored_toggle_is_the_users_and_never_follows(engine_service, tmp_path):
    """A value in config.json (an explicit false too: set by hand, or a desktop config) is a choice: it stays;
    only the absent one follows the model."""
    from glossarion_mobile.state import setting_writes as sw

    path = tmp_path / "config.json"
    original = json.dumps({"model": "authchatgpt/auto", TOGGLE: False})
    path.write_text(original, encoding="utf-8")
    store = _plain_store(path)
    sw.apply_relations_fresh_defaults(store, fresh=False)
    assert store.get(FRESH) == [MINIMAL_TOGGLE] and _toggles(store) == (False, True)
    for model in (SIGMA, "authchatgpt/auto"):
        sw.write_setting(store, "model", model)
        assert store.get(TOGGLE) is False
    both = tmp_path / "both.json"
    both.write_text(json.dumps({TOGGLE: False, MINIMAL_TOGGLE: False}), encoding="utf-8")
    kept = _plain_store(both)
    assert sw.apply_relations_fresh_defaults(kept, fresh=False) == [] and not kept.has(FRESH)
    store._saver.close()
    kept._saver.close()


def test_reset_to_default_follows_the_model_again(engine_service, tmp_path):
    from glossarion_mobile.state import setting_writes as sw

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "authchatgpt/auto", TOGGLE: False, MINIMAL_TOGGLE: False}), encoding="utf-8")
    store = _plain_store(path)
    old = sw.reset_relations_toggle(store, TOGGLE)
    assert store.get(TOGGLE) is True and store.get(FRESH) == [TOGGLE] and "relations" in store.get(FIELDS)
    assert old == {TOGGLE: False, FRESH: None, FIELDS: None}
    sw.write_setting(store, "model", SIGMA)
    assert store.get(TOGGLE) is False and store.get(MINIMAL_TOGGLE) is False  # Minimal is still the user's
    sw.restore_relations_toggle(store, old)  # Undo
    assert store.get(TOGGLE) is False and not store.has(FRESH) and not store.has(FIELDS)
    store._saver.close()


def test_a_run_uses_the_default_of_its_own_model(engine_service, tmp_path):
    """A chat or the Translate sheet can run another model than the global one (config_overrides): the
    toggles the user never set follow the RUN's model in the job's config; config.json is untouched."""
    from glossarion_mobile.services.jobs import JobService
    from glossarion_mobile.state import setting_writes as sw

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "authchatgpt/auto"}), encoding="utf-8")
    store = _plain_store(path)
    sw.apply_relations_fresh_defaults(store)
    assert _toggles(store) == (True, True) and "relations" in store.get(FIELDS)

    def run_config(overrides, source=None):
        config = dict(source if source is not None else store.snapshot())
        config.update(overrides)
        return sw.with_mobile_relations_defaults(sw.with_mobile_streaming_defaults(config))

    local = run_config({"model": QWEN})
    assert (local[TOGGLE], local[MINIMAL_TOGGLE]) == (False, False) and "relations" not in local[FIELDS]
    cloud = run_config({"model": "authchatgpt/auto"})
    assert (cloud[TOGGLE], cloud[MINIMAL_TOGGLE]) == (True, True) and "relations" in cloud[FIELDS]
    # a config without the mark (a run before Settings loaded, a loader without the store): absent = following
    bare = run_config({"model": SIGMA}, source={"model": "authchatgpt/auto"})
    assert (bare[TOGGLE], bare[MINIMAL_TOGGLE]) == (False, False) and FRESH not in bare
    bare_cloud = run_config({}, source={"model": "authchatgpt/auto"})
    assert (bare_cloud[TOGGLE], bare_cloud[MINIMAL_TOGGLE]) == (True, True) and "relations" in bare_cloud[FIELDS]
    # the global model on-device, the chat on a cloud model: on for that run
    sw.write_setting(store, "model", QWEN)
    assert _toggles(store) == (False, False)
    assert run_config({"model": "authchatgpt/auto"})[TOGGLE] is True
    # a toggle the user set stays as set in every run
    sw.write_setting(store, TOGGLE, True)
    sw.write_setting(store, "model", "authchatgpt/auto")
    hand = run_config({"model": QWEN})
    assert hand[TOGGLE] is True and hand[MINIMAL_TOGGLE] is False and "relations" in hand[FIELDS]
    desktop = run_config({"model": "authchatgpt/auto"}, source={TOGGLE: False, MINIMAL_TOGGLE: False})
    assert (desktop[TOGGLE], desktop[MINIMAL_TOGGLE]) == (False, False)

    # the job manager builds a run's config that way (store snapshot + the run's overrides)
    manager = JobService.__new__(JobService)
    manager.config_store, manager._config_loader = store, None
    manager._note_job_running = lambda running: None
    sw.reset_relations_toggle(store, TOGGLE)  # following again
    job = types.SimpleNamespace(spec=types.SimpleNamespace(params={"config_overrides": {"model": QWEN}}),
                                instance=None)
    config = JobService._config_snapshot(manager, job)
    assert config["model"] == QWEN and config[TOGGLE] is False and "relations" not in config[FIELDS]
    assert store.get(TOGGLE) is True  # the store keeps the global model's value
    store._saver.close()


def test_restore_and_import_seed_the_untouched_toggles(engine_service, tmp_path, monkeypatch):
    """A backup restore / config import in the session gets what app start gives it."""
    from glossarion_mobile.ui.screens import backup

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "authchatgpt/auto", TOGGLE: False}), encoding="utf-8")
    store = _plain_store(path)
    saved = tmp_path / "backup.json"
    saved.write_text(json.dumps({"model": QWEN}), encoding="utf-8")
    fake = types.SimpleNamespace(
        restore_config_backup_file=lambda target, source, safety_backup=None: Path(target).write_text(
            Path(source).read_text(encoding="utf-8"), encoding="utf-8"),
        backup_config_file=lambda p: None)
    monkeypatch.setitem(sys.modules, "config_store", fake)
    backup.restore_backup(store, str(saved))
    assert _toggles(store) == (False, False) and store.get(FRESH) == [TOGGLE, MINIMAL_TOGGLE]
    from glossarion_mobile.state import setting_writes as sw

    sw.write_setting(store, "model", "authchatgpt/auto")
    assert _toggles(store) == (True, True)
    store._saver.close()


def test_settings_counts_a_following_toggle_as_unchanged(engine_service, tmp_path):
    """The section subtitle, the Modified search filter and the section reset agree with the tile: a toggle
    that follows the model is the default, not a change."""
    from glossarion_mobile.state import setting_writes as sw
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    schema = _schema()
    store = _store(tmp_path, schema, {"model": "authchatgpt/auto"})
    sw.apply_relations_fresh_defaults(store)
    assert store.is_modified((TOGGLE,))  # seeded on, the desktop default is off
    ctx = SettingsContext(page=None, store=store, schema=schema)
    section = next(s for s in schema.sections() if TOGGLE in s.keys)
    home = SettingsHome(parse_route("/settings"), ctx)
    assert "changed" not in home._section_tile(section).subtitle.value
    assert not schema.is_modified(store, TOGGLE) and not schema.is_stored(store, TOGGLE)
    # the relations column the toggle seeded into Custom Fields is part of that default too
    assert store.get(FIELDS) == ["description", "relations"] and store.is_modified((FIELDS,))
    fields_section = next(s for s in schema.sections() if FIELDS in s.keys)
    assert "changed" not in home._section_tile(fields_section).subtitle.value
    assert not schema.is_modified(store, FIELDS) and not schema.is_stored(store, FIELDS)
    fields_page = SectionPage(parse_route(f"/settings/s/{fields_section.id}"), ctx)
    fields_page.get_body()
    assert FIELDS not in fields_page.reset_section() and store.get(FIELDS) == ["description", "relations"]
    fields_tile = fields_page.tile(FIELDS)
    assert not fields_tile.modified_dot.visible and fields_tile.value_text.value.endswith("· default")
    page = SectionPage(parse_route(f"/settings/s/{section.id}"), ctx)
    page.get_body()
    assert TOGGLE not in page.reset_section() and store.get(TOGGLE) is True
    # set by hand: a change, and the section reset makes it follow the model again (on), not the desktop off
    sw.write_setting(store, TOGGLE, False)
    assert schema.is_modified(store, TOGGLE) is False  # off is the desktop display default
    assert schema.is_stored(store, TOGGLE)
    assert TOGGLE in page.reset_section()
    assert store.get(TOGGLE) is True and sw.relations_follows_model(store, TOGGLE)
    assert sw.relations_follows_model(store, FIELDS)  # the column came back with it
    # a column the user added is a change of their own
    store.set(FIELDS, ["description", "relations", "age"])
    assert schema.is_modified(store, FIELDS) and not sw.relations_follows_model(store, FIELDS)
    store._saver.close()


def test_settings_load_seeds_the_untouched_toggles(engine_service, tmp_path):
    from glossarion_mobile.ui.settings.integration import SettingsFeature

    fresh = _plain_store(tmp_path / "fresh" / "config.json")
    asyncio.run(SettingsFeature(page=None, store=fresh).load())
    assert _toggles(fresh) == (True, True) and fresh.get(FRESH) == [TOGGLE, MINIMAL_TOGGLE]
    old_path = tmp_path / "old" / "config.json"
    old_path.parent.mkdir()
    old_path.write_text(json.dumps({"model": "m"}), encoding="utf-8")
    old = _plain_store(old_path)
    asyncio.run(SettingsFeature(page=None, store=old).load())
    assert _toggles(old) == (True, True) and old.get(FRESH) == [TOGGLE, MINIMAL_TOGGLE]
    assert not old.flush() and json.loads(old_path.read_text(encoding="utf-8")) == {"model": "m"}
    fresh._saver.close()
    old._saver.close()


def test_info_sheet_states_the_mobile_default(engine_service, tmp_path):
    """The info sheet says what the phone does (on; off for on-device models), not the desktop tooltip's
    "Default OFF." (the desktop text itself is unchanged)."""
    from glossarion_mobile.ui.settings.tiles import SwitchTile

    schema = _schema()
    spec = schema.spec(TOGGLE)
    assert "Default OFF." in spec.tooltip  # the desktop tooltip
    store = _store(tmp_path, schema)  # with the schema's display defaults
    ctx = types.SimpleNamespace(store=store, schema=schema, page=None, push=lambda *a: None, say=lambda *a, **k: None)
    for key in (TOGGLE, MINIMAL_TOGGLE):
        body = SwitchTile(schema.spec(key), ctx).help_body()
        assert "Default OFF" not in body and "Default: Off" not in body
        assert "Default: On (Off while an on-device model is selected)" in body
    other = SwitchTile(schema.spec("glossary_skip_gender_tracking"), ctx).help_body()
    assert "Default: " in other and "on-device" not in other
    # a toggle that follows the model shows as the default (no "modified" dot, nothing to reset) ...
    import flet as ft
    from glossarion_mobile.state import setting_writes as sw

    sw.apply_relations_fresh_defaults(store)
    tile = SwitchTile(schema.spec(TOGGLE), ctx)
    assert tile.follows_mobile_default() and tile.switch.value is True
    assert tile.value_text.value == "On · default" and not tile.modified_dot.visible
    assert not any(isinstance(a, ft.TextButton) for a in tile.open_help().actions)
    # ... until the user sets it
    assert tile.apply(False) and not tile.follows_mobile_default()
    assert store.get(TOGGLE) is False and tile.value_text.value == "Off" and store.get(FRESH) == [MINIMAL_TOGGLE]
    store._saver.close()


def test_reset_settings_starts_a_fresh_config(engine_service, tmp_path):
    from glossarion_mobile.ui.screens.danger_zone import reset_settings

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": SIGMA, "temperature": 0.9, TOGGLE: True}), encoding="utf-8")
    store = _plain_store(path)
    store.backup_now = lambda: None
    kept = reset_settings(store)
    assert kept == {"model": SIGMA}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[FRESH] == [TOGGLE, MINIMAL_TOGGLE] and saved[TOGGLE] is False and saved[MINIMAL_TOGGLE] is False
    store._saver.close()


def test_fresh_config_shows_the_toggles_on_and_asks_no_prompt_question(engine_service, iso, tmp_path):
    from glossarion_mobile.state import setting_writes as sw
    from glossarion_mobile.ui.glossary.relations import RelationsPanel, missing_prompts

    schema = _schema()
    from glossarion_mobile.state.config_store import MobileConfigStore

    store = MobileConfigStore(tmp_path / "config.json", debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk),
                                                                                        encoding="utf-8"))
    store.load()
    sw.apply_relations_fresh_defaults(store, fresh=True)
    assert store.effective(TOGGLE) is True and store.effective(MINIMAL_TOGGLE) is True

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        panel = RelationsPanel(ctx)
        panel.attach()
        assert panel.enabled and missing_prompts(ctx) == []  # the built-in prompts hold the line
        assert await panel.check_prompts() == [] and not ctx.extras.get("asked")
        # Sigma, then another model: the tracker turns on by itself, and still no question
        sw.write_setting(store, "model", SIGMA)
        await _settle()
        sw.write_setting(store, "model", "authchatgpt/auto")
        await _settle()
        if panel.check_task is not None:
            await panel.check_task
        assert store.get(TOGGLE) is True and not ctx.extras.get("asked") and panel.warning.visible is False
        panel.detach()
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# Relations & Gender Tracker: name taps, auto-update, editing, gender, filters, no "?" (owner 2026-11-13)
# ==========================================================================

GLOSSARY_CSV = ("type,raw_name,translated_name,gender\ncharacter,서윤,Seo-yun,female\n"
                "character,하린,Ha-rin,female\ncharacter,민준,Min-jun,male\ncharacter,지우,Ji-woo,\n")


def _walk(control):
    """Every control under ``control`` (content / controls / leading / title / actions)."""
    if control is None:
        return
    yield control
    for name in ("content", "title", "leading"):
        child = getattr(control, name, None)
        if child is not None and hasattr(child, "__dict__") and not isinstance(child, str):
            yield from _walk(child)
    for name in ("controls", "actions"):
        for child in getattr(control, name, None) or []:
            yield from _walk(child)


def _find(control, key):
    found = [c for c in _walk(control) if getattr(c, "key", None) == key]
    assert found, f"no control {key!r}"
    return found[0]


def _keys(control) -> set:
    return {getattr(c, "key", None) for c in _walk(control)} - {None}


def _texts(control) -> list:
    out = []
    for c in _walk(control):
        if type(c).__name__ == "Text":
            out.append(c.value)
        elif type(c).__name__ in ("TextButton", "FilledButton") and isinstance(getattr(c, "content", None), str):
            out.append(c.content)
    return out


def _book(tmp_path, *, tracker=None, gender=None):
    """A glossary file with its relations tracker (and optionally gender tracker) beside it."""
    rt = importlib.import_module("relations_tracking")
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text(GLOSSARY_CSV, encoding="utf-8")
    Path(rt.tracker_path_for_glossary(str(glossary))).write_text(
        json.dumps(tracker if tracker is not None else _tracker(), ensure_ascii=False), encoding="utf-8")
    for raw, value, chapter in gender or ():
        rt.set_gender(str(glossary), raw, value, chapter)
    return rt, glossary


def _id(diagram, raw):
    return diagram.core.tracker_key(raw)


def _saved(rt, glossary):
    return json.loads(Path(rt.tracker_path_for_glossary(str(glossary))).read_text(encoding="utf-8"))


def test_one_sided_relations_have_no_question_mark(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations_diagram import RelationsDiagram, edge_label_text

    rt = importlib.import_module("relations_tracking")
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [{"raw_name": "서윤", "translated_name": "Seo-yun", "relations": "하린: Club Member"}],
                        field="relations", chapter_index=0, chapter_num=1, chapter_file="ch1.html")
    assert edge_label_text({"label_ab": "", "label_ba": ""}) == ""
    schema = _schema()
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=tracker)
        texts = [s.value for s in diagram.canvas.shapes if type(s).__name__ == "Text"]
        assert texts == ["Club Member"] and not any("?" in t for t in texts)
        sheet = diagram.show_history(_id(diagram, "하린"))
        shown = _texts(sheet)
        assert not any("?" in str(t) for t in shown)
        assert "Seo-yun is Club Member to 하린" in shown  # who is what to whom (하린 has no translated name)
        store._saver.close()

    asyncio.run(scenario())


def test_tapping_a_name_in_the_sheet_opens_that_characters_sheet(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations_diagram import TITLE, RelationsDiagram

    schema = _schema()
    store = _store(tmp_path, schema)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = RelationsDiagram(ctx, str(tmp_path / "Book_glossary.csv"), tracker=_tracker())
        assert TITLE == "Relations & Gender Tracker" and _find(diagram.view.appbar, "rel-title").value == TITLE
        seo, harin = _id(diagram, "서윤"), _id(diagram, "하린")
        first = diagram.show_history(seo)
        assert diagram.selected == seo
        _find(first, f"rel-open-{harin}").on_click(None)
        await _settle()
        assert diagram.selected == harin and diagram.history_sheet is not first
        assert diagram.history_sheet.key.startswith("rel-history-sheet-") and _open_dialogs(session.page)[-1] is diagram.history_sheet
        assert "Ha-rin" in _texts(diagram.history_sheet)
        live = [d for d in _open_dialogs(session.page) if (d.key or "").startswith("rel-history-sheet-")]
        assert len(live) == 1
        store._saver.close()

    asyncio.run(scenario())


def test_editing_from_the_sheet_writes_the_tracker_sticks_and_undoes(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        assert diagram.watch_task is not None and diagram.undo_button.disabled
        seo, harin, minjun = _id(diagram, "서윤"), _id(diagram, "하린"), _id(diagram, "민준")
        sheet = diagram.show_history(seo)
        assert {f"rel-edit-{harin}", f"rel-delete-{harin}", "rel-history-add"} <= _keys(sheet)
        # edit one side of a relation (what Ha-rin is to Seo-yun)
        _find(sheet, f"rel-edit-{harin}").on_click(None)
        form = diagram.form_sheet
        assert form.key.startswith("rel-form-sheet-") and diagram.editing == 1
        assert await diagram.check_for_update() is False  # no reload while a form is open
        assert _find(form, "rel-form-ab").value == "Enemy"
        _find(form, "rel-form-ab").value = "Rival"
        _find(form, "rel-form-save").on_click(None)
        await diagram.task
        assert diagram.editing == 0
        saved = _saved(rt, glossary)
        assert rt.relation_labels(saved, "서윤", "하린") == ("Rival", "Best Friend")
        assert diagram.stamps[0] == rt.file_stamp(rt.tracker_path_for_glossary(str(glossary)))
        assert await diagram.check_for_update() is False and diagram.reloads == 0  # its own write: no reload
        assert not diagram.undo_button.disabled
        assert any("Rival" in str(s.value) for s in diagram.canvas.shapes if type(s).__name__ == "Text")
        # the correction sticks: a later extraction that repeats "Enemy" records "Rival"
        rt.record_relations(saved, [{"raw_name": "하린", "translated_name": "Ha-rin", "relations": "서윤: Enemy"}],
                            field="relations", chapter_index=5, chapter_num=6, chapter_file="ch6.html")
        assert rt.relation_labels(saved, "서윤", "하린")[0] == "Rival"
        # undo
        assert await diagram.undo() is True
        assert rt.relation_labels(_saved(rt, glossary), "서윤", "하린") == ("Enemy", "Best Friend")
        assert diagram.undo_button.disabled
        # delete asks first: No keeps it, Yes deletes the pair
        ctx.extras["answers"] = [False]
        await diagram.delete_relation(seo, minjun)
        assert minjun in _saved(rt, glossary)["entries"][seo]["relations"]
        ctx.extras["answers"] = [True]
        await diagram.delete_relation(seo, minjun)
        saved = _saved(rt, glossary)
        assert minjun not in saved["entries"][seo]["relations"] and seo not in saved["entries"].get(minjun, {}).get("relations", {})
        assert [t for _k, t, _b in ctx.extras["asked"]] == ["Delete relation", "Delete relation"]
        assert diagram.edge_count() == 1
        # add a relation: pick the other name from the suggestions, one direction, chapter 3
        diagram.add_relation_form(seo)
        form = diagram.form_sheet
        assert _find(form, "rel-add-name-a").value == "Seo-yun" and _find(form, "rel-add-chapter").value == "5"
        field_b = _find(form, "rel-add-name-b")
        field_b.value = "min"
        field_b.on_change(None)
        _find(form, f"rel-add-pick-b-{minjun}").on_click(None)
        assert field_b.value == "Min-jun"
        assert _find(form, "rel-add-label-ab").label == "What Min-jun is to Seo-yun"
        _find(form, "rel-add-save").on_click(None)  # no label yet: the form stays open with an error
        assert diagram.form_sheet is form and _find(form, "rel-add-error").visible
        _find(form, "rel-add-label-ab").value = "Cousin"
        _find(form, "rel-add-chapter").value = "3"
        _find(form, "rel-add-save").on_click(None)
        await diagram.task
        rel = _saved(rt, glossary)["entries"][seo]["relations"][minjun]
        assert [(o["label"], o["chapter_num"], o.get("manual")) for o in rel["occurrences"]] == [("Cousin", 3, True)]
        # a junk name typed by hand, then removed from the tracker
        diagram.add_relation_form(seo)
        _find(diagram.form_sheet, "rel-add-name-b").value = "Older Sibling"
        _find(diagram.form_sheet, "rel-add-label-ab").value = "Older Sibling"
        _find(diagram.form_sheet, "rel-add-save").on_click(None)
        await diagram.task
        junk = rt.tracker_key("Older Sibling")
        assert junk in _saved(rt, glossary)["entries"][seo]["relations"]
        ctx.extras["answers"] = [True]
        await diagram.remove_character(junk)
        saved = _saved(rt, glossary)
        assert junk not in saved["entries"] and junk not in saved["entries"][seo]["relations"]
        assert junk in saved["edits"]["removed"]
        # one history entry: Best Friend (ch 1) of what Ha-rin is to Seo-yun
        diagram.set_chapter(5)
        sheet = diagram.show_history(seo)
        ctx.extras["answers"] = [True]
        _find(sheet, f"rel-run-delete-{harin}-0").on_click(None)
        await diagram.task
        labels = [o["label"] for o in _saved(rt, glossary)["entries"][seo]["relations"][harin]["occurrences"]]
        assert labels == ["Enemy"]
        # descriptions: write one at the slider's chapter, edit it, delete it (asked)
        diagram.description_form(seo)
        _find(diagram.form_sheet, "rel-desc-text").value = "The heir"
        _find(diagram.form_sheet, "rel-desc-save").on_click(None)
        await diagram.task
        assert [(d["text"], d["chapter_num"], d.get("edited")) for d in _saved(rt, glossary)["entries"][seo]["descriptions"]] \
            == [("The heir", 5, True)]
        sheet = diagram.history_sheet  # refreshed after the write
        assert "The heir" in _texts(sheet)
        _find(sheet, "rel-desc-edit-0").on_click(None)
        _find(diagram.form_sheet, "rel-desc-text").value = "The heir, older"
        _find(diagram.form_sheet, "rel-desc-save").on_click(None)
        await diagram.task
        assert _saved(rt, glossary)["entries"][seo]["descriptions"][0]["text"] == "The heir, older"
        ctx.extras["answers"] = [True]
        _find(diagram.history_sheet, "rel-desc-delete-0").on_click(None)
        await diagram.task
        assert not _saved(rt, glossary)["entries"][seo].get("descriptions")
        assert len(diagram.undo_stack) >= 5
        diagram.close()
        assert diagram.closed and diagram.watch_task is None and ctx.overlays == []
        store._saver.close()

    asyncio.run(scenario())


def test_long_press_menu_offers_the_name_actions(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        minjun = _id(diagram, "민준")
        node = next(c for c in diagram.node_controls if c.key == f"rel-node-{minjun}")
        node.on_long_press(None)
        menu = diagram.form_sheet
        assert menu.key.startswith("rel-menu-sheet-")
        assert {"rel-menu-add", "rel-menu-description", "rel-menu-gender", "rel-menu-hide", "rel-menu-remove"} <= _keys(menu)
        _find(menu, "rel-menu-gender").on_click(None)
        assert diagram.form_sheet.key.startswith("rel-gender-sheet-")
        _find(diagram.form_sheet, "rel-gender-cancel").on_click(None)
        assert diagram.form_sheet is None and diagram.editing == 0
        node.on_long_press(None)
        _find(diagram.form_sheet, "rel-menu-hide").on_click(None)
        await diagram.task
        assert _saved(rt, glossary)["view"]["hidden"] == [minjun]
        assert all(c.key != f"rel-node-{minjun}" for c in diagram.node_controls)
        diagram.close()
        store._saver.close()

    asyncio.run(scenario())


def test_gender_tints_history_gender_only_names_and_gender_edits(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import open_diagram
    from glossarion_mobile.ui.glossary.relations_diagram import GENDER_TINTS, NEUTRAL_TINT

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path, gender=[("민준", "male", 1), ("민준", "female", 5), ("지우", "female", None)])

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        minjun, jiwoo = _id(diagram, "민준"), _id(diagram, "지우")

        def tint(node_id):
            node = next(c for c in diagram.node_controls if c.key == f"rel-node-{node_id}")
            return node.border.top.color

        assert tint(minjun) == GENDER_TINTS["female"]  # the gender tracker at chapter 5 wins over the glossary
        diagram.set_chapter(1)
        assert tint(minjun) == GENDER_TINTS["male"]
        diagram.set_chapter(5)
        # Ji-woo has gender tracking and no relations: a name without lines
        node = next(n for n in diagram.graph["nodes"] if n["id"] == jiwoo)
        assert node["gender_only"] and node["degree"] == 0 and tint(jiwoo) == GENDER_TINTS["female"]
        sheet = diagram.show_history(minjun)
        assert _find(sheet, "rel-history-gender").value == "Gender: Male (Ch. 1), Female (Ch. 5)"
        # set the gender for all chapters through the gender tracker's writer; undo it
        _find(sheet, "rel-gender-edit").on_click(None)
        form = diagram.form_sheet
        assert _find(form, "rel-gender-value").value == "female"
        _find(form, "rel-gender-value").value = "male"
        _find(form, "rel-gender-all").value = True
        _find(form, "rel-gender-save").on_click(None)
        await diagram.task
        gender_file = Path(rt.gender_tracker_path_for_glossary(str(glossary)))
        assert json.loads(gender_file.read_text(encoding="utf-8"))["entries"][minjun]["decision"] == "male"
        assert diagram.stamps[1] == rt.file_stamp(str(gender_file))
        assert await diagram.check_for_update() is False
        assert tint(minjun) == GENDER_TINTS["male"]
        assert _find(diagram.history_sheet, "rel-history-gender").value == "Gender: Male (all chapters)"
        assert await diagram.undo() is True
        assert json.loads(gender_file.read_text(encoding="utf-8"))["entries"][minjun]["decision"] == "auto"
        # one chapter cleared: asked first
        diagram.gender_form(minjun)
        _find(diagram.form_sheet, "rel-gender-value").value = "auto"
        _find(diagram.form_sheet, "rel-gender-all").value = False
        ctx.extras["answers"] = [True]
        _find(diagram.form_sheet, "rel-gender-save").on_click(None)
        await diagram.task
        assert ctx.extras["asked"][-1][1] == "Clear gender"
        assert rt.gender_history(json.loads(gender_file.read_text(encoding="utf-8")), "민준")["timeline"][-1]["gender"] == "male"
        # Gender off: neutral tints, no gender-only names, no gender history
        diagram.set_gender_shown(False)
        await diagram.task
        assert tint(minjun) == NEUTRAL_TINT and all(c.key != f"rel-node-{jiwoo}" for c in diagram.node_controls)
        assert "rel-history-gender" not in _keys(diagram.show_history(minjun))
        diagram.close()
        store._saver.close()

    asyncio.run(scenario())


def test_filters_hide_names_and_are_stored_in_the_tracker_view(iso, tmp_path):
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        seo, harin, minjun = _id(diagram, "서윤"), _id(diagram, "하린"), _id(diagram, "민준")
        assert diagram.edge_count() == 2 and "view" not in _saved(rt, glossary)
        positions = dict(diagram.positions)
        relations_before = _saved(rt, glossary)["entries"]
        # Hide: a searchable checklist
        hide = diagram.hide_sheet()
        assert hide.key.startswith("rel-hide-sheet-") and diagram.editing == 0
        search = _find(hide, "rel-hide-search")
        search.value = "min"
        search.on_change(None)
        assert _find(hide, f"rel-hide-{minjun}").visible and not _find(hide, f"rel-hide-{seo}").visible
        box = _find(hide, f"rel-hide-{minjun}")
        box.value = True
        box.on_change(types.SimpleNamespace(control=box))
        await diagram.task
        assert _saved(rt, glossary)["view"] == {"hidden": [minjun], "show_relations": True, "show_gender": True}
        assert _saved(rt, glossary)["entries"] == relations_before  # relations data untouched
        assert diagram.node_count() == 2 and diagram.edge_count() == 1
        assert diagram.positions == positions  # the others keep their places
        assert diagram.hide_chip.label.value == "Hidden (1)"
        assert await diagram.check_for_update() is False and diagram.reloads == 0  # its own write: no redraw
        _find(hide, "rel-hide-all").on_click(None)
        await diagram.task
        assert "view" not in _saved(rt, glossary) and diagram.node_count() == 3
        # Relations off: no lines, no labels, no relation history
        diagram.relations_chip.on_select(None)
        await diagram.task
        assert _saved(rt, glossary)["view"]["show_relations"] is False
        assert diagram.edge_count() == 0 and not diagram.canvas.shapes
        assert not any(k and k.startswith("rel-relation-") for k in _keys(diagram.show_history(seo)))
        # a second screen on the same book starts from the stored view (desktop shares it)
        other = await open_diagram(ctx, str(glossary))
        assert other.view_choice["show_relations"] is False and other.relations_chip.selected is False
        other.close()
        diagram.relations_chip.on_select(None)
        await diagram.task
        assert "view" not in _saved(rt, glossary) and diagram.edge_count() == 2
        diagram.close()
        store._saver.close()

    asyncio.run(scenario())


def test_auto_update_reloads_external_changes_and_keeps_the_view(iso, tmp_path, monkeypatch):
    from glossarion_mobile.ui.glossary import relations_diagram as module
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path)
    tracker_file = Path(rt.tracker_path_for_glossary(str(glossary)))

    def extraction(chapter, relations):
        rt.edit_tracker(str(tracker_file), lambda t: rt.record_relations(
            t, relations, field="relations", chapter_index=chapter - 1, chapter_num=chapter,
            chapter_file=f"ch{chapter}.html"))

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        seo, harin = _id(diagram, "서윤"), _id(diagram, "하린")
        assert await diagram.check_for_update() is False  # nothing changed: stat only
        diagram.set_focus(seo)
        diagram.show_history(harin)
        viewer = diagram.viewer
        # at the last chapter: follows the new last chapter
        extraction(7, [{"raw_name": "민준", "translated_name": "Min-jun", "relations": "하린: Friend"}])
        assert await diagram.check_for_update() is True
        assert diagram.chapter == 7 and diagram.slider.max == 7 and diagram.chapter_field.value == "7"
        assert diagram.focus == seo and diagram.selected == harin and diagram.viewer is viewer
        assert "Min-jun: Friend" in _texts(diagram.history_sheet)  # the open sheet shows the update
        # not at the last chapter: keeps its chapter
        diagram.set_chapter(1)
        extraction(9, [{"raw_name": "서윤", "translated_name": "Seo-yun", "relations": "하린: Rival"}])
        assert await diagram.check_for_update() is True and diagram.chapter == 1 and diagram.slider.max == 9
        # a file mid-write is skipped (not shown as empty) and read on a later check
        edges = diagram.edge_count()
        good = tracker_file.read_bytes()
        tracker_file.write_bytes(good[: len(good) // 2])
        assert await diagram.check_for_update() is False and diagram.edge_count() == edges
        tracker_file.write_bytes(good + b" ")
        assert await diagram.check_for_update() is True and diagram.edge_count() == edges
        # the glossary changes too: a name renamed there is redrawn
        glossary.write_text(GLOSSARY_CSV.replace("Ha-rin", "Harin"), encoding="utf-8")
        assert await diagram.check_for_update() is True
        assert any(c.content.value == "Harin" for c in diagram.node_controls)
        # the loop: once a second (sped up here), stopped by close
        monkeypatch.setattr(module, "POLL_SECONDS", 0.01)
        diagram.close()
        again = await open_diagram(ctx, str(glossary))
        extraction(10, [{"raw_name": "지우", "translated_name": "Ji-woo", "relations": "서윤: Maid"}])
        for _ in range(100):
            if again.reloads:
                break
            await asyncio.sleep(0.02)
        assert again.reloads == 1 and again.chapter == 10
        task = again.watch_task
        again.close()
        await asyncio.sleep(0.05)
        assert task.done() and again.watch_task is None
        store._saver.close()

    asyncio.run(scenario())


def test_android_back_stops_the_watch_loop(iso, tmp_path, monkeypatch):
    """Review: the system back gesture pops the View through the shell without ``close``."""
    from glossarion_mobile.ui.glossary import relations_diagram as module
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path)
    monkeypatch.setattr(module, "POLL_SECONDS", 0.01)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        ctx.shell = types.SimpleNamespace(overlays=ctx.overlays)
        diagram = await open_diagram(ctx, str(glossary))
        task = diagram.watch_task
        await asyncio.sleep(0.05)
        assert not task.done()
        ctx.overlays.remove(diagram.view)  # shell.pop_view on Android back
        for _ in range(50):
            if task.done():
                break
            await asyncio.sleep(0.02)
        assert task.done() and diagram.closed and diagram.watch_task is None
        store._saver.close()

    asyncio.run(scenario())


def test_slider_range_covers_the_gender_trackers_chapters(iso, tmp_path):
    """Review: the same range as desktop (relations 1-5, gender up to chapter 8)."""
    from glossarion_mobile.ui.glossary.relations import open_diagram
    from glossarion_mobile.ui.glossary.relations_diagram import GENDER_TINTS

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path, gender=[("민준", "female", 6), ("민준", "male", 8)])

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        minjun = _id(diagram, "민준")
        assert (diagram.low, diagram.high, diagram.chapter) == (1, 8, 8)
        node = next(c for c in diagram.node_controls if c.key == f"rel-node-{minjun}")
        assert node.border.top.color == GENDER_TINTS["male"]
        diagram.close()
        store._saver.close()

    asyncio.run(scenario())


def test_undoing_an_edit_keeps_a_later_view_choice(iso, tmp_path):
    """Review: a filter changed after an edit is not part of that edit's undo."""
    from glossarion_mobile.ui.glossary.relations import open_diagram

    schema = _schema()
    store = _store(tmp_path, schema)
    rt, glossary = _book(tmp_path)

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, store, schema)
        diagram = await open_diagram(ctx, str(glossary))
        seo, harin = _id(diagram, "서윤"), _id(diagram, "하린")
        _find(diagram.show_history(seo), f"rel-edit-{harin}").on_click(None)
        _find(diagram.form_sheet, "rel-form-ab").value = "Rival"
        _find(diagram.form_sheet, "rel-form-save").on_click(None)
        await diagram.task
        diagram.set_gender_shown(False)
        await diagram.task
        assert _saved(rt, glossary)["view"]["show_gender"] is False
        assert await diagram.undo() is True
        saved = _saved(rt, glossary)
        assert rt.relation_labels(saved, "서윤", "하린")[0] == "Enemy"
        assert saved["view"]["show_gender"] is False and diagram.view_choice["show_gender"] is False
        diagram.close()
        store._saver.close()

    asyncio.run(scenario())
