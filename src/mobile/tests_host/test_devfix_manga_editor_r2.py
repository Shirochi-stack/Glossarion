"""Owner's mobile manga editor reports (2026-10-16), issues 1, 10 and 11.

1. "Moving a text box in the manga editor is not updating the image preview in real time like the
   desktop app."
10. "The manga editor is missing the trash tool that deletes all OCR, clean and translation data from one
    manga panel (page). Add this to the triple-dots menu."
11. "The cancel button in the manga translator is not deleting selected text boxes."

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_manga_editor_r2.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import sys
import types
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_manga_ui_r2", Path(__file__).with_name("test_manga_ui.py"))
mu = sys.modules.get("_glossarion_manga_ui_r2")
if mu is None:
    mu = importlib.util.module_from_spec(_SPEC)
    sys.modules["_glossarion_manga_ui_r2"] = mu
    _SPEC.loader.exec_module(mu)

iso = mu.iso  # every path the manga code may write to lives in tmp_path

needs_real_core = pytest.mark.skipif(
    not (mu._has("flet") and mu._has("manga_editor_core") and mu._has("manga_files_core") and mu._has("PIL")),
    reason="flet or the shared manga editor core is not importable")


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


async def _draw(tab, x0, y0, x1, y1):
    tab.set_tool("box")
    tab.pan_start(x0, y0)
    tab.pan_update(x1, y1)
    await tab.pan_end()


def _color_png(path, color, size=(100, 200), box=None, box_color=None) -> str:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if box is not None:
        x, y, w, h = box
        image.paste(Image.new("RGB", (w, h), box_color), (x, y))
    image.save(path)
    return str(path)


def _colors(png: bytes) -> set:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        return {color for _count, color in image.convert("RGB").getcolors(1 << 16)}


RED, BLUE = (255, 0, 0), (0, 0, 255)
needs_flet_pil = pytest.mark.skipif(not (mu._has("flet") and mu._has("msgpack") and mu._has("PIL")),
                                    reason="flet / msgpack / PIL not installed")


def _fake_editor(monkeypatch, tmp_path, pages):
    """The editor screen over ``test_manga_ui.FakeEditorSession`` (jobs are scripted, nothing renders)."""
    from glossarion_mobile.ui.tools.manga import editor as me

    fake = mu.FakeEditorSession(pages)
    monkeypatch.setitem(sys.modules, "manga_editor_core",
                        mu._module("manga_editor_core", MangaEditorSession=lambda *a, **k: fake,
                                   default_state_file=lambda: str(tmp_path / "state.json")))
    monkeypatch.setattr(me, "image_size", lambda path: (100, 200))
    monkeypatch.setattr(me, "RENDER_DELAY", 0.2)
    monkeypatch.setitem(sys.modules, "manga_models", None)
    return fake


async def _translated_page(tab, fake, page, translated, cleaned):
    """Two translated boxes on the open page, its render and cleaned image (displayed 1:1)."""
    tab._on_viewport_size(types.SimpleNamespace(width=100.0, height=300.0))
    fake.add_box(10, 20, 40, 60)
    fake.add_box(10, 120, 40, 40)
    fake.boxes[page][0].update(ocr_text="原文", translation="Hi")
    fake.boxes[page][1].update(ocr_text="二", translation="Two")
    fake.rendered[page], fake.cleaned[page] = translated, cleaned
    fake.output_revision += 1
    await tab._apply_snapshot(await tab._snapshot())
    assert tab.translated_view and tab.background() == tab.translated_view


# ==========================================================================
# 1. A moved translated box: its text follows at once, the page re-renders it at its new place
# ==========================================================================


@needs_flet_pil
def test_moving_a_translated_box_carries_its_text_and_rerenders_it_where_it_went(iso, tmp_path, monkeypatch):
    """Root causes: (a) the live patches were swapped (``crop_patches`` returns (text, page under it), the
    editor took them as (hole, text)): the cleaned patch followed the finger and the rendered text stayed
    where it was; (b) the render after the move was Save & Update Overlay, which locks every region to
    its last render position, so the re-rendered page drew the text where it was before the move."""
    pages = [mu.write_png(tmp_path / "in" / f"{n}.png", 100, 200) for n in (1, 2)]
    # the translated render: red text in the box (10, 20, 40, 60) on the blue cleaned page
    translated = _color_png(tmp_path / "out" / "1_translated.png", BLUE, box=(10, 20, 40, 60), box_color=RED)
    cleaned = _color_png(tmp_path / "out" / "1_cleaned.png", BLUE)
    fake = _fake_editor(monkeypatch, tmp_path, pages)

    async def scenario():
        jobs = mu.FakeJobs()
        ctx, manga, screen, tab = _open_screen(tmp_path, pages, jobs=jobs)
        await tab.open_page(0)
        await _translated_page(tab, fake, pages[0], translated, cleaned)
        tab.set_tool("select")
        tab.pan_start(30, 50)  # inside box 0
        assert tab.drag["mode"] == "move"
        await mu._settle()
        tab.pan_update(60, 90)  # dragged by (+30, +40): live, before the finger lifts
        live = tab.live[0]
        assert live["text"] in tab.canvas.shapes and live["hole"] in tab.canvas.shapes
        # the rendered text (red) follows the box; the cleaned page (blue) covers where it was
        assert (live["text"].x, live["text"].y) == (40, 60) and _colors(live["text"].src) == {RED}
        assert (live["hole"].x, live["hole"].y) == (10, 20) and _colors(live["hole"].src) == {BLUE}
        await tab.pan_end()
        assert fake.calls[-1] == ("update_box", 0, 40, 60, 40, 60, False)
        # a moment after the last move: the desktop's rectangle-moved render of the moved box
        await asyncio.sleep(0.3)
        await mu._settle()
        spec = jobs.specs[-1]
        assert spec.params["step"] == "rerender_boxes" and spec.params["boxes"] == [0] and tab.busy
        jobs.finish(manga.step_job_id, result={"manga_step": "rerender_boxes"})
        await mu._settle()
        assert tab.live == {} and tab.view == "translated"
        screen.dispose()

    asyncio.run(scenario())


def _stub_renderer(monkeypatch, core):
    """``manga_translator.TextRegion`` + a recording ``_render_with_manga_translator`` (no model, no
    MangaTranslator import): the regions each render was asked to draw."""
    rendered: list = []

    class TextRegion:
        def __init__(self, text="", vertices=None, bounding_box=None, confidence=1.0, region_type="text_block"):
            self.text, self.vertices, self.bounding_box = text, vertices, bounding_box
            self.confidence, self.region_type, self.translated_text = confidence, region_type, ""

    monkeypatch.setitem(sys.modules, "manga_translator", mu._module("manga_translator", TextRegion=TextRegion))

    def render(self, base_image, regions, output_path=None, original_image_path=None, switch_tab=False, **kw):
        rendered.append(sorted((tuple(r.bounding_box), r.translated_text) for r in regions))
        state = self.image_state_manager.get_state(original_image_path) or {}
        state["rendered_image_path"] = output_path
        self.image_state_manager.set_state(original_image_path, state, save=True)

    monkeypatch.setattr(core, "_render_with_manga_translator", render)
    return rendered


@needs_real_core
def test_rerender_boxes_renders_a_moved_box_at_its_new_place_like_the_desktop(iso, tmp_path, monkeypatch):
    import manga_editor_core as core

    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.chdir(tmp_path)
    page = mu.write_png(tmp_path / "in" / "1.png", 100, 200)
    rendered = _stub_renderer(monkeypatch, core)
    session = core.MangaEditorSession(None, state_file=str(tmp_path / "state.json"), image_paths=[page])
    # (no owner / settings needed for a render of known texts)
    monkeypatch.setattr(session, "_begin", lambda owner, image_path: session.image_preview_widget.current_image_path)
    try:
        session.open_page(page)
        session.add_box(10, 20, 40, 60)
        session.add_box(10, 120, 40, 40)
        session.edit_box_text(0, translation="Hi")
        session.edit_box_text(1, translation="Two")
        state = session.image_state_manager.get_state(os.path.abspath(page))
        state["last_render_positions"] = {"0": [10, 20, 40, 60], "1": [10, 120, 40, 40]}  # the last render
        session.update_box(0, 50, 100, 40, 60, rerender=False)  # the drag
        # Save & Update Overlay keeps every region at its last render position (the old post-move render)
        session.save_and_update_overlay(page)
        assert ((10, 20, 40, 60), "Hi") in rendered[-1]
        # the desktop's rectangle-moved auto save: the moved box at its new place, the other one stays
        assert session.rerender_boxes([0], page)
        assert rendered[-1] == [((10, 120, 40, 40), "Two"), ((50, 100, 40, 60), "Hi")]
    finally:
        session.close()


# ==========================================================================
# 11. Delete box: the box goes, with its texts, and stays gone
# ==========================================================================


@needs_real_core
def test_a_deleted_box_stays_deleted_and_its_texts_never_move_to_the_next_box(iso, tmp_path, monkeypatch):
    """Root cause: ``MangaEditorSession.delete_box`` ran the context-menu handler only. The page state kept
    the deleted box's translated / recognized entry in its place, so after a reopen the next box showed
    the deleted box's text (and a re-render drew it), and the last box of a page came back on reopen
    (``viewer_rectangles`` was only rewritten while boxes remained; the desktop toolbar 🗑 persists it)."""
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.chdir(tmp_path)
    pages = [mu.write_png(tmp_path / "in" / f"{n}.png", 100, 200) for n in (1, 2)]

    async def wait_for(condition):
        for _ in range(300):
            await asyncio.sleep(0.01)
            if condition():
                return

    async def scenario():
        ctx, manga, screen, tab = _open_screen(tmp_path, pages)
        await tab.open_page(0)
        tab.geometry.display_w = 100.0
        await _draw(tab, 5, 5, 40, 40)
        await _draw(tab, 5, 100, 40, 150)
        await _draw(tab, 50, 100, 90, 150)
        for index, text in enumerate(("First", "Second", "Third")):
            await tab.box_action("save", index, {"translation": text})
        es = manga.editor
        key = os.path.abspath(pages[0])
        es.image_state_manager.get_state(key)["last_render_positions"] = {
            "0": [5, 5, 35, 35], "1": [5, 100, 35, 50], "2": [50, 100, 40, 50]}
        # the toolbar's Delete box comes right after the tools (on screen on a phone)
        controls = tab.tools_row.controls
        assert controls.index(tab.delete_button) < controls.index(tab.exclude_button)
        tab.set_tool("select")
        assert tab.tap_at(20, 20) == 0 and not tab.delete_button.disabled
        tab.delete_button.on_click(None)
        await wait_for(lambda: len(tab.boxes) == 2 and not tab._editing)
        assert [(b["x"], b["translation"]) for b in tab.boxes] == [(5.0, "Second"), (50.0, "Third")]
        state = es.image_state_manager.get_state(key)
        assert state["last_render_positions"] == {"0": [5, 100, 35, 50], "1": [50, 100, 40, 50]}
        # after a reopen the texts stay on their boxes, and the next edit lands on the right one
        await tab.step_page(1)
        await tab.step_page(-1)
        assert [(b["x"], b["translation"]) for b in tab.boxes] == [(5.0, "Second"), (50.0, "Third")]
        await tab.box_action("save", 1, {"translation": "Third!"})
        assert [b["translation"] for b in tab.boxes] == ["Second", "Third!"]
        # the box panel's ⋯ › Delete box, down to the last box: the page stays empty after a reopen
        for remaining in (1, 0):
            tab.long_press_at(10, 110)
            tab.panel.act("delete")
            await wait_for(lambda: len(tab.boxes) == remaining and not tab._editing)
        assert tab.boxes == []
        await tab.step_page(1)
        await tab.step_page(-1)
        assert tab.boxes == []
        screen.dispose()
        es.close()

    asyncio.run(scenario())


@needs_flet_pil
def test_deleting_a_translated_box_covers_its_text_and_rerenders_the_page(iso, tmp_path, monkeypatch):
    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    translated = _color_png(tmp_path / "out" / "1_translated.png", BLUE, box=(10, 20, 40, 60), box_color=RED)
    cleaned = _color_png(tmp_path / "out" / "1_cleaned.png", BLUE)
    fake = _fake_editor(monkeypatch, tmp_path, pages)

    async def scenario():
        jobs = mu.FakeJobs()
        ctx, manga, screen, tab = _open_screen(tmp_path, pages, jobs=jobs)
        await tab.open_page(0)
        await _translated_page(tab, fake, pages[0], translated, cleaned)
        tab.selected = 0
        assert await tab.delete_selected()
        await mu._settle()
        assert fake.calls[-1] == ("delete_box", 0) and len(tab.boxes) == 1
        hole = tab.holes[0]["hole"]
        assert hole in tab.canvas.shapes and (hole.x, hole.y, hole.width) == (10, 20, 40)
        assert _colors(hole.src) == {BLUE}  # the cleaned page covers the deleted text at once
        await asyncio.sleep(0.3)
        await mu._settle()
        spec = jobs.specs[-1]
        assert spec.params["step"] == "rerender_boxes" and spec.params["boxes"] == []
        jobs.finish(manga.step_job_id, result={"manga_step": "rerender_boxes"})
        await mu._settle()
        assert tab.holes == []
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# 10. ⋮ › Delete page data: every OCR, clean and translation result of the page
# ==========================================================================


@needs_real_core
def test_delete_page_data_in_the_more_menu_removes_ocr_clean_and_translation(iso, tmp_path, monkeypatch):
    import flet as ft

    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.chdir(tmp_path)
    pages = [mu.write_png(tmp_path / "in" / f"{n}.png", 100, 200) for n in (1, 2)]
    out = tmp_path / "in" / "1_translated"
    cleaned = mu.write_png(out / "1_cleaned.png", 100, 200)
    translated = mu.write_png(out / "1.png", 100, 200)
    other_page_output = mu.write_png(tmp_path / "in" / "2_translated" / "2_cleaned.png", 100, 200)

    async def scenario():
        ctx, manga, screen, tab = _open_screen(tmp_path, pages)
        await tab.open_page(0)
        es = manga.editor
        key = os.path.abspath(pages[0])
        tab.geometry.display_w = 100.0
        await _draw(tab, 5, 5, 40, 40)
        await tab.box_action("save", 0, {"translation": "Hello"})
        es._persist_recognized_text(0, "こんにちは")
        es.image_state_manager.update_state(key, {"cleaned_image_path": cleaned, "rendered_image_path": translated})
        await tab.step_page(1)
        await tab.step_page(-1)
        assert tab.boxes[0]["translation"] == "Hello" and tab.cleaned_view and tab.translated_view
        # the trash is in the ⋮ menu, enabled for an open page
        item = next(i for i in tab.more_menu.items if i.key == "me-delete-page")
        assert item is tab.delete_page_item and not item.disabled and item.icon == ft.Icons.DELETE_FOREVER
        ctx.extras["answers"] = ["no"]  # confirmed first: Cancel keeps everything
        assert not await tab.delete_page_data()
        assert ctx.extras["asked"][-1][0] == "Delete page data"
        assert len(tab.boxes) == 1 and os.path.isfile(cleaned) and os.path.isfile(translated)
        ctx.extras["answers"] = ["yes"]
        item.on_click(None)
        for _ in range(300):
            await asyncio.sleep(0.01)
            if not tab.boxes and not tab._editing:
                break
        await mu._settle()
        assert tab.boxes == [] and not tab.cleaned_view and not tab.translated_view
        assert not os.path.exists(cleaned) and not os.path.exists(translated)
        assert os.path.isfile(pages[0]) and os.path.isfile(other_page_output)  # the page and other pages stay
        state = es.image_state_manager.get_state(key)
        for name in ("recognized_texts", "translated_texts", "viewer_rectangles", "detection_regions",
                     "cleaned_image_path", "rendered_image_path", "last_render_positions"):
            assert not state.get(name), name
        await tab.step_page(1)
        await tab.step_page(-1)
        assert tab.boxes == [] and not tab.cleaned_view and not tab.translated_view
        screen.dispose()
        es.close()

    asyncio.run(scenario())
