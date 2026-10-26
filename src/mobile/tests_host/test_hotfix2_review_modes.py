"""Hotfix 2 (owner, 2026-10-26): Tools › Review "Modes" took too much space (one full-width switch
row per mode). They are compact checkboxes in one wrapping row now, persisted as before.

Run from src/mobile:  python -m pytest -p no:cacheprovider tests_host/test_hotfix2_review_modes.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

ft = pytest.importorskip("flet")


def _u7():
    name = "_glossarion_hotfix2_review_u7"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "test_tools_u7.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_review_modes_are_one_compact_row_and_still_persist():
    u7 = _u7()  # puts src/ (the shared review_generator) on sys.path
    pytest.importorskip("review_generator")
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools import review as rv

    async def scenario():
        _conn, session = u7._tb()._fake_session("android")
        page = session.page
        store = {}
        ctx = u7._tools_ctx(page, store=store, jobs=u7.U7Jobs())
        screen = rv.ReviewScreen(parse_route("/tools/review"), ctx)
        body = screen.get_body()
        u7._tools()._mount(page, body)
        screen.did_show()

        modes = next(c for c in body.controls if getattr(c, "key", None) == "review-modes")
        rows = []

        def walk(control):
            if isinstance(control, ft.Row) and getattr(control, "key", None) == "review-modes-row":
                rows.append(control)
            for child in getattr(control, "controls", None) or []:
                walk(child)
            content = getattr(control, "content", None)
            if isinstance(content, ft.Control):
                walk(content)

        walk(modes)
        assert len(rows) == 1 and rows[0].wrap is True
        boxes = [c for c in rows[0].controls if isinstance(c, ft.Checkbox)]
        assert [b.label for b in boxes] == ["50/50 Split", "Full Review", "Wrap Chunks", "Volume"]
        assert not any(isinstance(c, ft.Switch) for c in rows[0].controls)
        assert all(b.visual_density == ft.VisualDensity.COMPACT for b in boxes)
        assert screen.order_button in rows[0].controls

        # toggles still persist under the dialog's config keys and drive Wrap Chunks' visibility
        assert screen.switches["wrap_chunks"].visible is False
        screen.set_mode("chunk_mode", True)
        assert store["review_chunk_mode"] is True and screen.switches["chunk_mode"].value is True
        assert screen.switches["wrap_chunks"].visible is True
        screen.set_mode("spoiler_mode", True)
        assert store["review_spoiler_mode"] is True and screen.switches["spoiler_mode"].value is True

    asyncio.run(scenario())
