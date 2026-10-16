"""Owner's mobile manga editor reports (2026-10-16), issues 3, 4 and 5b.

3. "The box dialog in the manga editor keeps jumping up or down. Please just keep it always below the
   text box. You can assign more height if the text box is at the bottom of the image."
4. "The manga editor seems to sometimes default to the heavy model when clicking run. Fix that. Mobile
   should default to the 11 MB one."
5b. The editor's tool row runs off the right edge on a 412 dp phone (the exclude icon is cut off).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_manga_editor_r3.py
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_manga_ui_r3", Path(__file__).with_name("test_manga_ui.py"))
mu = sys.modules.get("_glossarion_manga_ui_r3")
if mu is None:
    mu = importlib.util.module_from_spec(_SPEC)
    sys.modules["_glossarion_manga_ui_r3"] = mu
    _SPEC.loader.exec_module(mu)

iso = mu.iso  # every path the manga code may write to lives in tmp_path

needs_models = pytest.mark.skipif(not mu._has("manga_models"), reason="manga_models not importable")
needs_flet = pytest.mark.skipif(not mu._has("flet"), reason="flet not importable")

#: A config imported from (or synced with) the desktop: its GUI defaults are stored values.
DESKTOP_CONFIG = {"manga_settings": {"ocr": {"rtdetr_onnx_variant": "detector.onnx", "detector_type": "rtdetr_onnx"},
                                     "inpainting": {"method": "local", "local_method": "anime_onnx"}},
                  "manga_local_inpaint_model": "anime_onnx", "manga_inpaint_method": "local"}


@pytest.fixture
def mobile(iso, monkeypatch):
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    return iso


# ================================================================ 4. the 11 MB detector by default


@needs_models
def test_a_run_from_an_imported_desktop_config_uses_the_phone_models(mobile, monkeypatch):
    """The heavy model came from two paths: (1) a config imported from the desktop stores
    ``detector.onnx`` / ``anime_onnx`` and a stored value won over the phone default (prepare_run,
    ensure_run_models, the Settings rows); (2) the editor's Detect loads the detector without a file
    name (manga_editor_core: ``load_rtdetr_onnx_model(model_source)``), so BubbleDetector fell back to
    ``RTDETR_ONNX_FILENAME`` or ``detector.onnx`` whatever the config said."""
    from glossarion_mobile.services import manga as svc
    import manga_models

    monkeypatch.delenv("RTDETR_ONNX_FILENAME", raising=False)
    run = copy.deepcopy(DESKTOP_CONFIG)
    before = dict(os.environ)
    try:
        notes = svc.prepare_run(run)
        exported = os.environ.get("RTDETR_ONNX_FILENAME")
    finally:
        os.environ.clear()
        os.environ.update(before)
    assert run["manga_settings"]["ocr"]["rtdetr_onnx_variant"] == "detector-v4-s_int8.onnx"
    assert run["manga_local_inpaint_model"] == run["manga_settings"]["inpainting"]["local_method"] == "aot_onnx"
    assert any("detector.onnx → detector-v4-s_int8.onnx" in n for n in notes)
    assert any("anime_onnx → aot_onnx" in n for n in notes)
    # the editor's Detect (no file name) resolves the run's export, like the desktop's env
    assert exported == "detector-v4-s_int8.onnx"
    bubble = sys.modules.get("bubble_detector")
    if bubble is not None or mu._has("bubble_detector"):
        try:
            import bubble_detector
        except Exception:
            bubble_detector = None
        if bubble_detector is not None:
            monkeypatch.setenv("RTDETR_ONNX_FILENAME", exported)
            assert bubble_detector.BubbleDetector.normalize_rtdetr_onnx_filename(None) == "detector-v4-s_int8.onnx"
    # the pre-download and the "Download models" prompt only ever fetch the light models
    downloads: list = []
    monkeypatch.setattr(manga_models, "is_installed", lambda spec: False)
    monkeypatch.setattr(manga_models, "download", lambda spec, **kw: downloads.append(spec.key) or "x")
    svc.ensure_run_models(copy.deepcopy(DESKTOP_CONFIG))
    assert downloads == ["rtdetr_v4_s_int8", "aot_onnx"]
    assert svc.ModelManager().required(copy.deepcopy(DESKTOP_CONFIG)) == ["rtdetr_v4_s_int8", "aot_onnx"]
    # Settings shows what the run loads
    assert svc.ModelManager().detector_id(DESKTOP_CONFIG) == "rtdetr_v4_s_int8"
    assert svc.current_inpaint_choice(DESKTOP_CONFIG) == ("local", "aot_onnx")
    # a fresh install (nothing stored) and a blank value: the phone defaults
    assert svc.ModelManager().required({}) == ["rtdetr_v4_s_int8", "aot_onnx"]
    blank = {"manga_settings": {"ocr": {"rtdetr_onnx_variant": ""}}, "manga_local_inpaint_model": None}
    assert svc.ModelManager().required(blank) == ["rtdetr_v4_s_int8", "aot_onnx"]


@needs_models
def test_a_model_picked_on_the_phone_is_kept(mobile):
    from glossarion_mobile.services import manga as svc

    run = copy.deepcopy(DESKTOP_CONFIG)
    for updates in (svc.model_choice_updates("detector", "detector.onnx"),
                    svc.model_choice_updates("inpaint", "anime_onnx")):
        for (top, kind), value in updates.items():
            run.setdefault(top, {})[kind] = value
    before = dict(os.environ)
    try:
        notes = svc.prepare_run(run)
    finally:
        os.environ.clear()
        os.environ.update(before)
    assert not [n for n in notes if "→" in n and ("detector" in n.lower() or "inpainting" in n.lower())]
    assert run["manga_settings"]["ocr"]["rtdetr_onnx_variant"] == "detector.onnx"
    assert run["manga_local_inpaint_model"] == "anime_onnx"
    assert svc.ModelManager().required(run) == ["rtdetr", "anime_onnx"]
    # the desktop never changes
    os.environ.pop("GLOSSARION_MOBILE", None)
    desktop = copy.deepcopy(DESKTOP_CONFIG)
    assert svc.prepare_run(desktop) == [] and desktop == DESKTOP_CONFIG


@needs_flet
@needs_models
@pytest.mark.skipif(not mu._has("settings_schema"), reason="settings_schema not importable")
def test_settings_picks_record_the_phone_choice(mobile, tmp_path):
    from glossarion_mobile.services import manga as svc
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    store, schema = mu._settings_store(tmp_path)
    for key, value in (("manga_local_inpaint_model", "anime_onnx"),
                       (("manga_settings", "ocr", "rtdetr_onnx_variant"), "detector.onnx")):
        store.set(key, value)  # an imported desktop config

    async def scenario():
        _conn, session = mu._tb()._fake_session("android")
        page = session.page
        settings = SettingsContext(page=page, store=store, schema=schema)
        ctx = mu._ctx(page, store, settings=settings, output_root=str(mobile["Output"]))
        screen = MangaScreen(parse_route("/tools/manga?tab=settings"), ctx, session=mu._session(tmp_path, {}))
        mu._mount(page, screen.get_body())
        tab = screen.settings_tab
        tab.did_show()
        # the rows show what a run loads (the light models), not the imported heavy ones
        assert tab.model_rows["detector"].model_id == "rtdetr_v4_s_int8"
        assert svc.current_inpaint_choice(tab.config())[1] == "aot_onnx"
        # picking the heavy ones on the phone keeps them for the phone's runs
        tab._select_variant("detector.onnx")
        assert tab.select_inpaint("local", "anime_onnx")
        assert store.get(("manga_mobile_model_choice", "detector")) == "detector.onnx"
        assert store.get(("manga_mobile_model_choice", "inpaint")) == "anime_onnx"
        assert tab.model_rows["detector"].model_id == "rtdetr"
        assert svc.current_inpaint_choice(tab.config())[1] == "anime_onnx"
        assert svc.ModelManager().required(tab.config()) == ["rtdetr", "anime_onnx"]
        tab.dispose() if hasattr(tab, "dispose") else None

    asyncio.run(scenario())


# ================================================================ 5b. every tool on screen


@needs_flet
def test_toolbar_fits_a_360_and_412_dp_phone(mobile, tmp_path, monkeypatch):
    from glossarion_mobile.ui.tools.manga import editor as me

    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    monkeypatch.setitem(sys.modules, "manga_models", None)

    async def scenario():
        _ctx, _manga, _screen, tab = _open_screen(tmp_path, pages)
        _check_toolbar(me, tab)

    asyncio.run(scenario())


def _check_toolbar(me, tab):
    buttons = [tab.view_button, *tab.tool_buttons.values(), tab.delete_button, tab.exclude_button,
               tab.clear_button, tab.more_menu]
    assert len(buttons) == 10 and tab.exclude_button in buttons
    for button in buttons:
        c = button.size_constraints
        assert c.max_width == me.TOOL_BUTTON and c.min_width == me.TOOL_BUTTON, button.key
    for button in buttons[1:-1]:
        assert button.icon_size == me.TOOL_ICON
    divider = next(c for c in tab.tools_row.controls if isinstance(c, me.ft.VerticalDivider))
    total = len(buttons) * me.TOOL_BUTTON + divider.width
    for phone in (360, 412):
        assert total <= phone - 2 * me.GUTTER, (phone, total)
    # no horizontal scrolling: every tool is laid out on screen
    assert tab.tools_row.scroll is None and tab.toolbar.scroll is None
    # the selected tool's highlight stays inside its own 34 dp square
    assert me.TOOL_STYLE.bgcolor[me.ft.ControlState.SELECTED] is not None
    assert tab.tool_buttons["pan"].selected and tab.tool_buttons["pan"].style is me.TOOL_STYLE
    # icons are not text: 200 % text scale does not widen the row (the emoji fits the 34 dp button)
    assert tab.view_emoji.size <= me.TOOL_BUTTON / 2


# ================================================================ 3. the box panel always under the box


def _open_screen(tmp_path, pages, *, jobs=None):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    _conn, session = mu._tb()._fake_session("android")
    page = session.page
    ctx = mu._ctx(page, {}, jobs=jobs, files=mu.FakeFiles(), output_root=str(tmp_path / "Output"))
    manga = mu._session(tmp_path, {})
    manga.files.host.selected_files = list(pages)
    manga.loaded = True
    screen = MangaScreen(parse_route("/tools/manga?tab=editor"), ctx, session=manga)
    mu._mount(page, screen.get_body())
    return ctx, manga, screen, screen.editor_tab


@needs_flet
def test_panel_placement_math():
    from glossarion_mobile.ui.tools.manga import editor as me

    # 412 x 600 dp viewer, a page 412 x 900 (taller: pans), plus 206 dp of panel room
    layout = me.PageLayout.fit(412, 600, 412 / 900, room=206)
    assert (layout.page_top, layout.extra, layout.content_h) == (0, 206, 1106)
    t = me.ViewTransform()
    t.resize(412, 600, 412, layout.content_h)
    assert me.panel_placement(t, 100, 200, 6) == (0.0, 106)  # fits: no pan
    pan, top = me.panel_placement(t, 880, 200, 6)  # the box at the page's bottom
    assert pan == -486 and top == 400 and top + 200 == 600
    # a page shorter than the viewer gets only what its bottom margin lacks
    short = me.PageLayout.fit(360, 600, 360 / 400, room=206)
    assert short.page_top == 100 and short.extra == 106 and short.content_h == 706
    assert me.PageLayout.fit(360, 600, 360 / 400).extra == 0
    # zoomed: the pan is clamped to what the content allows
    z = me.ViewTransform(scale=2.0)
    z.resize(412, 600, 412, 1106)
    pan, top = me.panel_placement(z, 1100, 200, 6)
    assert z.ty + pan >= 600 - 1106 * 2 and pan <= 0


@needs_flet
def test_panel_stays_under_the_box_and_never_jumps(mobile, tmp_path, monkeypatch):
    from glossarion_mobile.ui.tools.manga import editor as me

    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 300)]
    fake_session = mu.FakeEditorSession(pages)
    monkeypatch.setitem(sys.modules, "manga_editor_core",
                        mu._module("manga_editor_core", MangaEditorSession=lambda *a, **k: fake_session,
                                   default_state_file=lambda: str(tmp_path / "state.json")))
    monkeypatch.setattr(me, "image_size", lambda path: (100, 300))
    monkeypatch.setitem(sys.modules, "manga_models", None)
    monkeypatch.setattr(me, "PAN_SETTLE", 0.0)  # (no wait for Flutter's layout in a host test)
    pans: list = []

    async def scenario():
        _ctx, _manga, _screen, tab = _open_screen(tmp_path, pages)
        await tab.open_page(0)
        tab._on_viewport_size(types.SimpleNamespace(width=200.0, height=500.0))  # page 200 x 600 dp
        tab.interactive.pan = lambda dx, dy=0, dz=0: _record(pans, dy)
        fake_session.add_box(10, 20, 40, 30)    # near the top
        fake_session.add_box(10, 270, 40, 25)   # at the very bottom of the page (bottom 295 px = 590 dp)
        await tab._apply_snapshot(await tab._snapshot())
        assert tab.layout.extra == 0 and tab.frame.height == 600
        tab.selected = 0
        tab.refresh()
        assert tab.panel_holder.top == 100 + me.PANEL_GAP and tab.panel_holder.bottom is None
        assert tab.view_transform.ty == 0 and pans == []
        # the bottom box: blank room under the page, the page scrolls up, the panel stays under the box
        tab.selected = 1
        tab.refresh()
        await asyncio.sleep(0)
        assert tab.layout.extra == me.PANEL_GAP + me.PANEL_H and tab.frame.height == 600 + tab.layout.extra
        assert tab.frame.padding.top == 0  # the page did not move inside its frame
        box_bottom = 590 + tab.view_transform.ty
        assert tab.panel_holder.top == pytest.approx(box_bottom + me.PANEL_GAP)
        assert tab.panel_holder.top + me.PANEL_H <= 500 + 0.5
        assert pans and pans[-1] == pytest.approx(tab.view_transform.ty)
        placed = (tab.panel_holder.top, tab.view_transform.ty)
        # typing, a re-render and a refresh do not move it
        tab.panel.translation_field.value = "typed\nmore\nlines"
        tab.refresh()
        tab.refresh()
        assert (tab.panel_holder.top, tab.view_transform.ty) == placed and len(pans) == 1
        # the keyboard shrinks the viewer: box and panel scroll up together, the panel still under the box
        tab._on_viewport_size(types.SimpleNamespace(width=200.0, height=380.0))
        await asyncio.sleep(0)
        assert tab.panel_holder.top == pytest.approx(590 + tab.view_transform.ty + me.PANEL_GAP)
        assert tab.panel_holder.top + me.PANEL_H <= 380 + 0.5
        # the panel grows (a taller field / 200 % text): more room, still under the box and on screen
        tab._on_panel_size(types.SimpleNamespace(width=200.0, height=260.0))
        await asyncio.sleep(0)
        assert tab.panel_h == 260 and tab.layout.extra == me.PANEL_GAP + 260
        assert tab.panel_holder.top + 260 <= 380 + 0.5
        tab._on_panel_size(types.SimpleNamespace(width=200.0, height=180.0))  # never shrinks back
        assert tab.panel_h == 260
        # panning by hand: the panel follows the box
        tab.view_transform.ty += 40
        tab._on_view_update(types.SimpleNamespace(local_focal_point=None, scale=1.0))
        assert tab.panel_holder.top == pytest.approx(590 + tab.view_transform.ty + me.PANEL_GAP)
        # done: the room goes away
        panned = len(pans)
        tab.deselect()
        await asyncio.sleep(0)
        assert tab.layout.extra == 0 and tab.frame.height == 600 and not tab.panel_holder.visible
        # the page was panned into the room: the viewer is panned back with the clamp (no desync)
        assert tab.view_transform.ty == 380 - 600 and len(pans) == panned + 1 and pans[-1] > 0

    asyncio.run(scenario())


async def _noop():
    return None


def _record(pans, dy):
    pans.append(dy)
    return _noop()
