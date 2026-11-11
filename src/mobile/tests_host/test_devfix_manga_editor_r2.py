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


def _open_screen(tmp_path, pages, *, jobs=None, platform="android"):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.manga.screen import MangaScreen

    _conn, session = mu._tb()._fake_session(platform)
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
        assert not jobs.specs and not tab.busy
        assert ("local_overlay", (0,), pages[0]) in fake.calls
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

    monkeypatch.setitem(sys.modules, "manga_translator",
                        mu._module("manga_translator", TextRegion=TextRegion, MangaTranslator=lambda **kw: object()))

    def render(self, base_image, regions, output_path=None, original_image_path=None, switch_tab=False, **kw):
        rendered.append(sorted((tuple(r.bounding_box), r.translated_text) for r in regions))
        state = self.image_state_manager.get_state(original_image_path) or {}
        state["rendered_image_path"] = output_path
        self.image_state_manager.set_state(original_image_path, state, save=True)
        return output_path

    monkeypatch.setattr(core, "_render_with_manga_translator", render)
    return rendered


@needs_flet_pil
@pytest.mark.parametrize("platform", ["android", "ios"])
def test_release_remains_interactive_during_render_and_latest_move_wins(iso, tmp_path, monkeypatch, platform):
    """A slow overlay must not remove touch targets or make release wait for the core's lock."""
    import threading
    from glossarion_mobile.ui.tools.manga import editor as me

    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    translated = _color_png(tmp_path / "out" / "translated.png", BLUE, box=(10, 20, 40, 60), box_color=RED)
    cleaned = _color_png(tmp_path / "out" / "cleaned.png", BLUE)
    fake = _fake_editor(monkeypatch, tmp_path, pages)
    entered, release = threading.Event(), threading.Event()
    original_render = fake.rerender_boxes_locally

    def slow_render(*args):
        entered.set()
        assert release.wait(10), "test did not release the renderer"
        return original_render(*args)

    monkeypatch.setattr(fake, "rerender_boxes_locally", slow_render)

    async def scenario():
        jobs = mu.FakeJobs()
        ctx, manga, screen, tab = _open_screen(tmp_path, pages, jobs=jobs, platform=platform)
        render_task = None
        try:
            await tab.open_page(0)
            await _translated_page(tab, fake, pages[0], translated, cleaned)
            old_image = tab.translated_view
            tab.pan_start(30, 50)
            tab.pan_update(35, 55)
            old_strip = list(tab.strip.controls)
            await tab.pan_end()
            # A geometry save does not replace/redecode the page or rebuild its thumbnails.
            assert tab.translated_view == old_image and tab.strip.controls == old_strip
            tab._render_token += 1  # run the render now, not through the debounce timer
            render_task = asyncio.create_task(tab._render_overlay())
            for _ in range(500):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set() and tab._overlay_rendering
            assert tab.hit_areas and tab._can_edit(geometry=True)
            assert not tab._can_edit()  # destructive actions still wait for the renderer
            updates_before = len([c for c in fake.calls if c[0] == "update_box"])
            for start, end in [((35, 55), (40, 60)), ((40, 60), (45, 65))]:
                tab.pan_start(*start)
                assert tab.drag and tab.drag["mode"] == "move"
                tab.pan_update(*end)
                assert await asyncio.wait_for(tab.pan_end(), timeout=0.5) == 0
            assert len([c for c in fake.calls if c[0] == "update_box"]) == updates_before
            assert (tab.boxes[0]["x"], tab.boxes[0]["y"]) == (25, 35)
            tab.pan_start(65, 95)  # the resize handle remains active too
            assert tab.drag["mode"] == "resize"
            tab.pan_update(70, 100)
            assert await asyncio.wait_for(tab.pan_end(), timeout=0.5) == 0
            assert tab._pulse_tokens and tab.hit_areas
            pulse_widths = lambda: [s.paint.stroke_width for s in tab.canvas.shapes
                                   if getattr(getattr(s, "paint", None), "color", None) == "#AA66FF"]
            widths = pulse_widths()
            assert widths and (5 in widths or 2 in widths)
            for _ in range(40):  # the UI's pulse continues while the renderer is held
                await asyncio.sleep(0.05)
                if pulse_widths() != widths:
                    break
            assert pulse_widths() != widths
            assert tab.translated_view == old_image and tab.live
            release.set()
            await asyncio.wait_for(render_task, timeout=3)
            assert (fake.boxes[pages[0]][0]["x"], fake.boxes[pages[0]][0]["y"]) == (25, 35)
            assert (fake.boxes[pages[0]][0]["width"], fake.boxes[pages[0]][0]["height"]) == (45, 65)
            assert len([c for c in fake.calls if c[0] == "update_box"]) == updates_before + 1
            assert tab.translated_view == old_image and tab.live  # stale render was not installed
            for _ in range(100):
                await asyncio.sleep(0.05)
                if not tab.live and not tab.busy:
                    break
            assert not tab.live and not tab.busy and not tab._deferred_geometry
            assert (tab.boxes[0]["x"], tab.boxes[0]["y"]) == (25, 35)
            assert not jobs.specs  # auto move renders are not jobs
            await asyncio.sleep(1.0)
            assert not tab._pulse_tokens
        finally:
            release.set()
            if render_task is not None:
                await render_task
            screen.dispose()

    asyncio.run(scenario())


@needs_flet_pil
@pytest.mark.parametrize("platform", ["android", "ios"])
def test_repeated_resizes_keep_transparent_preview_and_pulse_during_render(iso, tmp_path, monkeypatch, platform):
    """Grow/shrink releases stay responsive and keep the latest size while rendering is blocked."""
    import threading
    from PIL import Image

    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    translated = _color_png(tmp_path / "out" / "translated.png", BLUE,
                            box=(25, 35, 5, 15), box_color=RED)
    cleaned = _color_png(tmp_path / "out" / "cleaned.png", BLUE)
    fake = _fake_editor(monkeypatch, tmp_path, pages)
    entered, release = threading.Event(), threading.Event()
    original_render = fake.rerender_boxes_locally

    def slow_render(*args):
        entered.set()
        assert release.wait(10), "test did not release the renderer"
        return original_render(*args)

    monkeypatch.setattr(fake, "rerender_boxes_locally", slow_render)

    async def scenario():
        jobs = mu.FakeJobs()
        ctx, manga, screen, tab = _open_screen(tmp_path, pages, jobs=jobs, platform=platform)
        render_task = None
        try:
            await tab.open_page(0)
            await _translated_page(tab, fake, pages[0], translated, cleaned)
            tab.selected = 0
            old_image = tab.translated_view
            tab.pan_start(50, 80)
            assert tab.drag["mode"] == "resize"
            tab.pan_update(60, 90)
            assert await asyncio.wait_for(tab.pan_end(), timeout=0.5) == 0
            tab._render_token += 1
            render_task = asyncio.create_task(tab._render_overlay())
            for _ in range(500):
                if entered.is_set() and tab.live[0]["text"] is not None:
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set() and tab._overlay_rendering
            updates_before = len([c for c in fake.calls if c[0] == "update_box"])
            for start, end, size in [((60, 90), (80, 110), (70, 90)),
                                     ((80, 110), (75, 105), (65, 85))]:
                tab.pan_start(*start)
                assert tab.drag["mode"] == "resize"
                tab.pan_update(*end)
                live = tab.live[0]
                assert (live["text"].x, live["text"].y) == (10, 20)
                assert (live["text"].width, live["text"].height) == size
                assert (live["hole"].width, live["hole"].height) == (40, 60)
                with Image.open(io.BytesIO(live["text"].src)) as ink:
                    assert ink.mode == "RGBA"
                    assert ink.getpixel((5, 5))[3] == 0  # background must not stretch with the text
                    assert ink.getpixel((17, 20)) == (*RED, 255)
                assert await asyncio.wait_for(tab.pan_end(), timeout=0.5) == 0
                assert tab._pulse_tokens and tab.hit_areas
            assert len([c for c in fake.calls if c[0] == "update_box"]) == updates_before
            pulse_widths = lambda: [s.paint.stroke_width for s in tab.canvas.shapes
                                   if getattr(getattr(s, "paint", None), "color", None) == "#AA66FF"]
            widths = pulse_widths()
            assert widths and (5 in widths or 2 in widths)
            for _ in range(40):
                await asyncio.sleep(0.05)
                if pulse_widths() != widths:
                    break
            assert pulse_widths() != widths
            release.set()
            await asyncio.wait_for(render_task, timeout=3)
            assert tab.translated_view == old_image and tab.live  # reject the stale render
            assert len([c for c in fake.calls if c[0] == "update_box"]) == updates_before + 1
            assert tuple(fake.boxes[pages[0]][0][k] for k in ("x", "y", "width", "height")) == (10, 20, 65, 85)
            for _ in range(100):
                await asyncio.sleep(0.05)
                if not tab.live and not tab.busy and not tab._pulse_tokens:
                    break
            assert not tab.live and not tab.busy and not tab._pulse_tokens and not tab._deferred_geometry
            assert not jobs.specs
        finally:
            release.set()
            if render_task is not None:
                await render_task
            screen.dispose()

    asyncio.run(scenario())


@needs_flet_pil
def test_move_during_display_preparation_cannot_install_an_old_render(iso, tmp_path, monkeypatch):
    import threading
    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    translated = _color_png(tmp_path / "out" / "translated.png", BLUE, box=(10, 20, 40, 60), box_color=RED)
    cleaned = _color_png(tmp_path / "out" / "cleaned.png", BLUE)
    fake = _fake_editor(monkeypatch, tmp_path, pages)
    prepared, release = threading.Event(), threading.Event()

    async def scenario():
        ctx, manga, screen, tab = _open_screen(tmp_path, pages, jobs=mu.FakeJobs())
        pending = None
        try:
            await tab.open_page(0)
            await _translated_page(tab, fake, pages[0], translated, cleaned)
            old_image = tab.translated_view
            display = tab._display_translated

            def slow_display(snap=None):
                result = display(snap)
                prepared.set()
                assert release.wait(10)
                return result

            monkeypatch.setattr(tab, "_display_translated", slow_display)
            fake.output_revision += 1
            snap = fake.page_snapshot()
            pending = asyncio.create_task(tab._apply_snapshot(snap, render_token=tab._render_token))
            for _ in range(500):
                if prepared.is_set():
                    break
                await asyncio.sleep(0.01)
            assert prepared.is_set()
            assert Path(old_image).is_file()  # still needed for the next live ink crop
            tab.pan_start(30, 50)
            tab.pan_update(45, 65)
            release.set()
            await pending
            assert tab.translated_view == old_image
            assert (tab.boxes[0]["x"], tab.boxes[0]["y"]) == (25, 35)
            assert tab.live and tab.drag
        finally:
            release.set()
            if pending is not None:
                await pending
            screen.dispose()

    asyncio.run(scenario())


@needs_flet_pil
@pytest.mark.parametrize("platform", ["android", "ios"])
def test_a_slow_position_write_does_not_block_the_next_drag(iso, tmp_path, monkeypatch, platform):
    import threading
    pages = [mu.write_png(tmp_path / "in" / "1.png", 100, 200)]
    fake = _fake_editor(monkeypatch, tmp_path, pages)
    entered, release = threading.Event(), threading.Event()
    update = fake.update_box

    def slow_write(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return update(*args, **kwargs)

    monkeypatch.setattr(fake, "update_box", slow_write)

    async def scenario():
        ctx, manga, screen, tab = _open_screen(tmp_path, pages, jobs=mu.FakeJobs(), platform=platform)
        saving = None
        try:
            await tab.open_page(0)
            tab._on_viewport_size(types.SimpleNamespace(width=100.0, height=300.0))
            fake.add_box(10, 20, 40, 60)
            await tab._apply_snapshot(await tab._snapshot())
            tab.pan_start(30, 50)
            tab.pan_update(35, 55)
            saving = asyncio.create_task(tab.pan_end())
            for _ in range(500):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set() and tab._geometry_saving
            assert tab.hit_areas and tab._can_edit(geometry=True)
            for start, end in [((35, 55), (40, 60)), ((40, 60), (45, 65))]:
                tab.pan_start(*start)
                assert tab.drag is not None
                tab.pan_update(*end)
                assert await asyncio.wait_for(tab.pan_end(), timeout=0.5) == 0
            release.set()
            assert await asyncio.wait_for(saving, timeout=3) == 0
            assert not tab.busy and not tab._deferred_geometry
            assert (fake.boxes[pages[0]][0]["x"], fake.boxes[pages[0]][0]["y"]) == (25, 35)
            assert (tab.boxes[0]["x"], tab.boxes[0]["y"]) == (25, 35)
            assert len([c for c in fake.calls if c[0] == "update_box"]) == 2  # initial write + latest
        finally:
            release.set()
            if saving is not None:
                await saving
            screen.dispose()

    asyncio.run(scenario())


@needs_flet_pil
@pytest.mark.parametrize("output_scale", [1, 2])
def test_live_move_patch_contains_ink_without_the_rectangular_artwork(tmp_path, output_scale):
    from PIL import Image, ImageDraw
    from glossarion_mobile.ui.tools.manga.editor import crop_patches

    # A patterned panel behind a text glyph must stay on the page, not follow the glyph.
    base = Image.new("RGB", (100, 200), "white")
    draw = ImageDraw.Draw(base)
    for x in range(0, 100, 4):
        draw.line((x, 0, x, 199), fill=(100, 100, 100))
    cleaned = tmp_path / "cleaned.png"
    base.save(cleaned)
    rendered = base.resize((100 * output_scale, 200 * output_scale))
    ImageDraw.Draw(rendered).rectangle((25 * output_scale, 35 * output_scale,
                                      30 * output_scale, 50 * output_scale), fill="black")
    translated = tmp_path / "translated.png"
    rendered.save(translated)
    text, hole = crop_patches(str(translated), str(cleaned), (10, 20, 40, 60), (100, 200))
    with Image.open(io.BytesIO(text)) as ink, Image.open(io.BytesIO(hole)) as patch:
        assert ink.mode == "RGBA"
        assert ink.getpixel((17 * output_scale, 20 * output_scale)) == (0, 0, 0, 255)
        assert ink.getpixel((5 * output_scale, 5 * output_scale))[3] == 0
        assert ink.getpixel((10 * output_scale, 5 * output_scale))[3] == 0
        assert patch.convert("RGB").tobytes() == base.crop((10, 20, 50, 80)).tobytes()


@needs_real_core
def test_rerender_boxes_renders_a_moved_box_at_its_new_place_like_the_desktop(iso, tmp_path, monkeypatch):
    import manga_editor_core as core

    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.chdir(tmp_path)
    page = mu.write_png(tmp_path / "in" / "1.png", 100, 200)
    rendered = _stub_renderer(monkeypatch, core)
    session = core.MangaEditorSession(None, state_file=str(tmp_path / "state.json"), image_paths=[page])
    # (no owner / settings needed for a render of known texts)
    monkeypatch.setattr(session, "_begin",
                        lambda owner, image_path, **kw: session.image_preview_widget.current_image_path)
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
        assert session.rerender_boxes_locally([0], page, {})
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
        for remaining, (px, py) in ((1, (10, 110)), (0, (60, 110))):
            panel = tab.long_press_at(px, py)
            assert panel is not None and tab.selected == 0
            panel.act("delete")
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
        assert not jobs.specs and not tab.busy
        assert ("local_overlay", (), pages[0]) in fake.calls
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
