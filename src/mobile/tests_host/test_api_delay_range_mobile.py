"""API call delay ranges on mobile (2026-10-21): the delay fields take "5" or a range "5-7", like the desktop.

The Settings tile (``delay`` and the QA dialog's AI-truncation delay, -1 = the global delay), the key
editor's and the bulk sheet's per-key delay, and Welcome's pacing field all parse and store through
``settings_rules`` (one copy, shared with the desktop): the min stays the numeric setting, the max goes to
``<key>_max`` ('' = a fixed delay).
"""

from __future__ import annotations

import asyncio
import importlib.util
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

pytest.importorskip("flet")

_MS_SPEC = importlib.util.spec_from_file_location("_glossarion_ms_helpers_delay",
                                                  Path(__file__).with_name("test_mobile_settings.py"))
_MS = importlib.util.module_from_spec(_MS_SPEC)
sys.modules[_MS_SPEC.name] = _MS  # its dataclasses look their module up
_MS_SPEC.loader.exec_module(_MS)

QA_KEY = "qa_scanner_settings.ai_truncation_api_call_delay"


def _tile(tmp_path, key, data):
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess
    from glossarion_mobile.ui.settings.tiles import DelayRangeTile, make_tile

    schema = SchemaAccess()
    assert schema.available, schema.error
    store = _MS._memory_store(tmp_path, data, schema)
    _conn, session = _MS._fake_session("android")
    ctx = _MS._ctx(store, schema, session.page)
    tile = make_tile(schema.spec(key), ctx)
    assert isinstance(tile, DelayRangeTile)
    return tile, store


def test_a_new_user_sees_3_to_7_and_a_saved_delay_stays_fixed(tmp_path):
    async def scenario():
        tile, store = _tile(tmp_path / "new", "delay", {})
        assert tile.field.value == "3-7" and not store.keys()     # display only, nothing written
        tile, store = _tile(tmp_path / "saved", "delay", {"delay": 5.0})
        assert tile.field.value == "5"
        tile, store = _tile(tmp_path / "range", "delay", {"delay": 5.0, "delay_max": 7.0})
        assert tile.field.value == "5-7"
    (tmp_path / "new").mkdir(), (tmp_path / "saved").mkdir(), (tmp_path / "range").mkdir()
    asyncio.run(scenario())


@pytest.mark.parametrize("typed,stored,shown", [
    ("6-8", (6.0, 8.0), "6-8"),
    ("6 – 8", (6.0, 8.0), "6-8"),
    ("6—8", (6.0, 8.0), "6-8"),
    ("6-", (6.0, ""), "6"),
    ("8-6", (8.0, ""), "8"),
    ("6-6", (6.0, ""), "6"),
    ("0.5-1.5", (0.5, 1.5), "0.5-1.5"),
    ("4", (4.0, ""), "4"),
])
def test_the_delay_tile_stores_the_min_and_the_max(tmp_path, typed, stored, shown):
    async def scenario():
        tile, store = _tile(tmp_path, "delay", {"delay": 5.0, "delay_max": 7.0})
        assert tile.apply(typed)
        assert (store.get(("delay",)), store.get(("delay_max",))) == stored
        assert tile.field.value == shown
    asyncio.run(scenario())


def test_the_delay_tile_rejects_text_that_is_no_delay(tmp_path):
    async def scenario():
        tile, store = _tile(tmp_path, "delay", {"delay": 5.0})
        assert not tile.apply("fast") and tile.error
        assert store.get(("delay",)) == 5.0 and not store.has(("delay_max",))
    asyncio.run(scenario())


def test_the_qa_delay_tile_keeps_minus_one_for_the_global_delay(tmp_path):
    async def scenario():
        tile, store = _tile(tmp_path, QA_KEY, {})
        assert tile.field.value == "-1"
        assert tile.apply("1-3")
        qa = store.get(("qa_scanner_settings",))
        assert (qa["ai_truncation_api_call_delay"], qa["ai_truncation_api_call_delay_max"]) == (1.0, 3.0)
        assert tile.field.value == "1-3"
        assert tile.apply("-1")
        qa = store.get(("qa_scanner_settings",))
        assert (qa["ai_truncation_api_call_delay"], qa["ai_truncation_api_call_delay_max"]) == (-1.0, "")
    asyncio.run(scenario())


def test_the_max_keys_have_no_tile_of_their_own():
    from glossarion_mobile.ui.settings.model import FOLDED_KEYS
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    assert FOLDED_KEYS["delay_max"] == "delay"
    assert FOLDED_KEYS[QA_KEY + "_max"] == QA_KEY
    schema = SchemaAccess()
    section = next(s for s in schema.sections() if "delay" in s.keys)
    assert "delay_max" in section.keys
    assert "delay_max" not in [s.key for s in schema.specs_for(section)]
    assert schema.display_key("delay_max") == "delay"


def test_bulk_and_summary_per_key_ranges():
    from glossarion_mobile.ui.screens import keys

    assert keys.parse_bulk_value("api_call_delay", "2-5") == ({"api_call_delay": 2.0, "api_call_delay_max": 5.0}, None)
    assert keys.parse_bulk_value("api_call_delay", "3") == ({"api_call_delay": 3.0, "api_call_delay_max": ""}, None)
    assert keys.parse_bulk_value("api_call_delay", "x")[1]
    assert keys.parse_bulk_value("api_call_delay", "1-9999")[1]       # above the 3600 s bound
    assert keys.BULK_CLEAR_EXTRA["api_call_delay"] == {"api_call_delay_max": ""}


def test_the_key_editor_takes_a_per_key_range(tmp_path):
    from glossarion_mobile.ui.screens.key_editor import KeyEditor

    async def scenario():
        _conn, session = _MS._fake_session("android")
        store = _MS._memory_store(tmp_path, {})
        ctx = _MS._ctx(store, None, session.page)
        saved = []
        entry = {"api_key": "k", "model": "gpt-6", "api_call_delay": 2.0, "api_call_delay_max": 5.0}

        def edit(text):
            editor = KeyEditor(ctx, entry=dict(entry), pool_id="fallback",
                               on_save=lambda e: saved.append(e) or None).show()
            field = editor.number_fields["api_call_delay"][0]
            assert field.value == "2-5"
            field.value = text
            return editor

        assert edit("3 – 4").save()
        assert (saved[-1]["api_call_delay"], saved[-1]["api_call_delay_max"]) == (3.0, 4.0)
        assert edit("6").save()
        assert (saved[-1]["api_call_delay"], saved[-1]["api_call_delay_max"]) == (6.0, "")
        assert edit("").save()                      # empty = the global delay
        assert (saved[-1]["api_call_delay"], saved[-1]["api_call_delay_max"]) == (0.0, "")
        editor = edit("soon")
        assert not editor.save() and "range" in editor.error_text.value
        store._saver.close()

    asyncio.run(scenario())


def test_welcome_takes_a_range():
    from glossarion_mobile.ui.screens.welcome_flow import WelcomeFlow

    flow = WelcomeFlow()
    flow.delay = "4-6"
    updates = flow.finish_updates()
    assert (updates["delay"], updates["delay_max"]) == (4.0, 6.0)
    flow = WelcomeFlow()
    flow.delay = "4"
    updates = flow.finish_updates()
    assert (updates["delay"], updates["delay_max"]) == (4.0, "")
