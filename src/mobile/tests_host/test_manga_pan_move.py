"""Owner (2026-11-07): "also mobile has a separate tool for move and grab, but can't we merge it? since
the user is not tapping the text box unless they want to move it anyway".

The manga editor's Pan / zoom and Select / move tools are one default tool, Pan / move. Each gesture is
routed by where it starts: a hit area over every box (and the selected box's handles) takes taps,
long-presses and drags that start on it; a drag that starts on the empty page reaches only the
viewer (``InteractiveViewer``), which pans it, and a pinch zooms it.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_manga_pan_move.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_manga_ui_panmove", Path(__file__).with_name("test_manga_ui.py"))
mu = sys.modules.get("_glossarion_manga_ui_panmove")
if mu is None:
    mu = importlib.util.module_from_spec(_SPEC)
    sys.modules["_glossarion_manga_ui_panmove"] = mu
    _SPEC.loader.exec_module(mu)

iso = mu.iso  # every path the manga code may write to lives in tmp_path

needs_flet = pytest.mark.skipif(not mu._has("flet"), reason="flet not importable")


def _open(tmp_path, monkeypatch, *, jobs=None):
    """The editor on a 100 x 200 px page shown 200 x 400 dp (scale 2) with two boxes: box 0 at
    (10, 20) 40 x 60 px (page dp (20, 40) 80 x 120) and box 1 at (10, 150) 40 x 30 px."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga import editor as me
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    fake = mu.FakeEditorSession(pages)
    monkeypatch.setitem(sys.modules, "manga_editor_core",
                        mu._module("manga_editor_core", MangaEditorSession=lambda *a, **k: fake,
                                   default_state_file=lambda: str(tmp_path / "state.json")))
    monkeypatch.setattr(me, "image_size", lambda path: (100, 200))
    monkeypatch.setitem(sys.modules, "manga_models", None)
    _conn, session = mu._tb()._fake_session("android")
    ctx = mu._ctx(session.page, {}, jobs=jobs or mu.FakeJobs(), files=mu.FakeFiles(),
                  output_root=str(iso_root(tmp_path)))
    manga = mu._session(tmp_path, {})
    manga.files.host.selected_files = list(pages)
    manga.loaded = True
    screen = MangaScreen(parse_route("/tools/manga?tab=editor"), ctx, session=manga)
    mu._mount(session.page, screen.get_body())
    return me, fake, pages, session, screen, screen.editor_tab


def iso_root(tmp_path):
    root = tmp_path / "Output"
    root.mkdir(parents=True, exist_ok=True)
    return root


async def _page_with_boxes(tab, fake, pages):
    await tab.open_page(0)
    tab._on_viewport_size(types.SimpleNamespace(width=200.0, height=500.0))
    fake.add_box(10, 20, 40, 60)
    fake.add_box(10, 150, 40, 30)
    await tab._apply_snapshot(await tab._snapshot())
    assert tab.geometry.scale == 2.0


def _rect(control):
    return (control.left, control.top, control.width, control.height)


def _covers(control, x, y):
    left, top, width, height = _rect(control)
    return left <= x <= left + width and top <= y <= top + height


async def _fire(session, control, name, x, y):
    """A Flutter gesture event on ``control`` at its local point (x, y), through Flet's own dispatch."""
    point = {"x": float(x), "y": float(y)}
    await session.dispatch_event(control._i, name, {"k": "touch", "l": point, "g": point, "ld": point, "gd": point,
                                                     "v": {"x": 0.0, "y": 0.0}})


@needs_flet
def test_one_pan_move_tool_replaces_pan_and_select(iso, tmp_path, monkeypatch):
    async def scenario():
        me, _fake, _pages, _session, screen, tab = _open(tmp_path, monkeypatch)
        assert [t[0] for t in me.TOOLS] == ["pan", "box", "circle", "lasso"]
        assert me.TOOLS[0][1] == "Pan / move" and tab.tool == "pan"
        assert "select" not in tab.tool_buttons and tab.tool_buttons["pan"].tooltip == "Pan / move"
        # an old "select" (a caller of an earlier build) is the merged tool
        tab.set_tool("box")
        tab.set_tool("select")
        assert tab.tool == "pan" and tab.tool_buttons["pan"].selected
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_a_drag_on_a_box_moves_it_and_a_drag_on_the_empty_page_is_the_viewers(iso, tmp_path, monkeypatch):
    async def scenario():
        me, fake, pages, session, screen, tab = _open(tmp_path, monkeypatch)
        await _page_with_boxes(tab, fake, pages)
        viewer, surface = tab.interactive, tab.gestures
        # Pan / move: the viewer pans and pinches; the full-page surface has no drag or tap recognizer
        # (it would win every drag against the viewer), only long-press
        assert viewer.pan_enabled and viewer.scale_enabled
        assert (surface.on_pan_start, surface.on_tap_down) == (None, None)
        areas = list(tab.hit_areas)
        assert tab.stack.controls == [tab.page_image, tab.canvas, surface, *areas]  # the top children
        assert [_rect(a) for a in areas] == [(20, 40, 80, 120), (20, 300, 80, 60)]
        assert all(a.on_pan_start and a.on_pan_update and a.on_pan_end and a.on_tap_down and a.on_long_press_start
                   for a in areas)
        # a drag on the empty page: no hit area there, so only the viewer sees it; it pans / zooms the page
        # and no box moves
        calls = len(fake.calls)
        assert not [a for a in areas if _covers(a, 190, 10)]
        tab._on_view_start(types.SimpleNamespace(local_focal_point=types.SimpleNamespace(x=190, y=10), pointer_count=2))
        tab._on_view_update(types.SimpleNamespace(local_focal_point=types.SimpleNamespace(x=190, y=10), scale=2.0))
        tab._on_view_end(types.SimpleNamespace(velocity=types.SimpleNamespace(x=0, y=0)))
        assert tab.view_transform.scale == 2.0 and tab.drag is None
        assert fake.calls[calls:] == [] and [b["y"] for b in tab.boxes] == [20.0, 150.0]
        assert tab.selected is None
        # a drag that starts on box 1 moves it (and selects it), live while the finger moves
        area = areas[1]
        await _fire(session, area, "pan_start", 40, 30)  # page (60, 330) = image (30, 165)
        assert tab.drag["mode"] == "move" and tab.selected == 1
        assert area in tab.hit_areas  # the area under the finger stays (the drag goes on)
        await _fire(session, area, "pan_update", 60, 50)  # +20 dp = +10 px each way
        assert (tab.boxes[1]["x"], tab.boxes[1]["y"]) == (20.0, 160.0)
        assert area in tab.hit_areas
        await _fire(session, area, "pan_end", 60, 50)
        await mu._settle()
        assert fake.calls[-1] == ("update_box", 1, 20.0, 160.0, 40.0, 30.0, False) and tab.selected == 1
        # the areas follow the moved box; the selected box's handles get their own (2 x 22 dp at 2x zoom)
        rects = [_rect(a) for a in tab.hit_areas]
        assert rects[:2] == [(20, 40, 80, 120), (40, 320, 80, 60)]
        assert rects[2:] == [(40 - 11, 320 - 11, 22, 22), (120 - 11, 380 - 11, 22, 22)]
        # the resize handle resizes
        handle = tab.hit_areas[3]
        await _fire(session, handle, "pan_start", 11, 11)
        assert tab.drag["mode"] == "resize"
        await _fire(session, handle, "pan_update", 21, 31)
        await _fire(session, handle, "pan_end", 21, 31)
        await mu._settle()
        assert fake.calls[-1] == ("update_box", 1, 20.0, 160.0, 45.0, 40.0, False)
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_a_tap_on_a_box_selects_it_and_long_press_opens_its_panel(iso, tmp_path, monkeypatch):
    async def scenario():
        me, fake, pages, session, screen, tab = _open(tmp_path, monkeypatch)
        await _page_with_boxes(tab, fake, pages)
        first, second = tab.hit_areas
        await _fire(session, first, "tap_down", 5, 5)
        assert tab.selected == 0 and tab.panel is not None and tab.panel.index == 0
        assert [s for s in tab.canvas.shapes if isinstance(s, me.cv.Circle)]  # its handles show
        # a tap beside every box keeps the selection (only a box's area takes taps; the page has none)
        assert tab.gestures.on_tap_down is None
        await _fire(session, second, "long_press_start", 5, 5)
        assert tab.selected == 1 and tab.panel.index == 1
        # a drawing tool: no hit areas (the surface draws), the viewer holds still; back: areas again
        tab.set_tool("box")
        assert tab.hit_areas == [] and not tab.interactive.pan_enabled and tab.selected is None
        assert tab.stack.controls == [tab.page_image, tab.canvas, tab.gestures]  # the surface on top
        assert tab.gestures.on_pan_start is not None
        tab.set_tool("pan")
        assert len(tab.hit_areas) == 2 and tab.interactive.pan_enabled
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_no_hit_areas_while_a_step_runs(iso, tmp_path, monkeypatch):
    """While a step runs the boxes cannot be edited: a drag anywhere pans the page."""
    async def scenario():
        jobs = mu.FakeJobs()
        me, fake, pages, session, screen, tab = _open(tmp_path, monkeypatch, jobs=jobs)
        await _page_with_boxes(tab, fake, pages)
        assert len(tab.hit_areas) == 2
        job_id = await tab.run_step("translate")
        assert tab.busy and tab.hit_areas == []
        jobs.finish(job_id, result={"manga_step": "translate"})
        await mu._settle()
        assert not tab.busy and len(tab.hit_areas) == 2
        screen.dispose()

    asyncio.run(scenario())
