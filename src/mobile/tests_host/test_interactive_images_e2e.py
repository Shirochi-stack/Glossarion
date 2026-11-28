"""Interactive Images end to end in the real app (HeadlessOwner + JobService + the shared runner), offline.

Import an EPUB into the Library, translate it from the Book page (Chapters › Run translation › Start) against
the fake model server, then Book page › Images › ▶ Build pending: an ``interactive_images`` job plans each
translated chapter on the text model (the fake server's planner answer), draws one image per chapter on the
image model (``/images/generations``) and inserts it. The tab's rows turn Built, the chapter files carry the
figure with their text unchanged, and the images sit in ``Interactive_Images/`` under fixed names.

Run from src/mobile:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_interactive_images_e2e.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


I7 = _load("test_hotfix1_issue7.py", "_glossarion_i7_helpers_interactive_images")
pytestmark = I7.pytestmark
storage = I7.storage  # noqa: F811  (pytest fixtures)
app_env = I7.app_env
isolated = I7.isolated

IMAGE_MODEL = "gpt-image-1"  # an OpenAI image model: the custom endpoint's /images/generations


async def _translation(js):
    """The translate job Start queued (already in the history on a fast runner)."""
    snap = await I7._until(lambda: next((s for s in [js.snapshot(), *js.view().history]
                                         if s is not None and I7._kind(s) == "translate"), None), 60)
    assert snap is not None, "Start did not run a translation"
    return snap


def test_translate_then_build_interactive_images_from_the_images_tab(isolated):
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_PNG, FakeLLMServer

    picks, files = isolated
    tf = I7._load("test_ui_foundations.py", "_glossarion_tf_helpers_interactive_images")

    async def scenario(server):
        run = await I7._start(tf, server, picks, files)
        app, driver = run.app, run.driver
        js = app.job_service
        try:
            app.config_store.set_many({"interactive_images_image_model": IMAGE_MODEL, "auto_glossary_mode": "off"})
            await I7._open_library(run)
            book = await I7._import_book(run, f"{I7.BOOK_STEM}.epub")
            book_page, _bid = await I7._open_chapters(run, book)
            workspace = book["output_folder"]

            # 1. the translation
            await driver.tap(key="ch-run", timeout=10)
            await driver.wait(key="start", timeout=30)
            await driver.tap(key="start", timeout=10)
            if await driver.exists(text="Keep translations running", timeout=2):
                await driver.tap(text="Not now")
            snap = await _translation(js)
            done = await I7._wait_end(js, snap.id)
            assert I7._state(done) == "DONE", getattr(done, "error", done)

            # 2. Book page › Images: every translated chapter, nothing built yet
            await driver.tap(text="Images", timeout=30)
            assert await I7._until(lambda: book_page.images is not None and book_page.images.rows, 60), \
                book_page.images
            view = book_page.images
            chapters = len(view.rows)
            assert chapters >= I7.CHAPTERS and {r.status for r in view.rows} == {"not_built"}, view.rows
            assert book_page.images_tab.build_reason() is None

            # 3. ▶ Build pending: the job, then the rows turn Built
            before = len(server.requests)
            await driver.tap(key="ix-build", timeout=10)
            job = await I7._until(lambda: next((s for s in [js.snapshot(), *js.view().history]
                                                if s is not None and I7._kind(s) == "interactive_images"), None), 60)
            assert job is not None and tuple(I7._norm(p) for p in job.spec.inputs) == (I7._norm(workspace),)
            end = await I7._wait_end(js, job.id)
            assert I7._state(end) == "DONE", getattr(end, "error", end)
            assert end.result["interactive_images"]["completed"] == chapters, end.result
            kinds = [r.kind for r in server.requests[before:]]
            assert kinds.count("planner") == chapters and kinds.count("image_generation") == chapters, kinds
            assert await I7._until(lambda: book_page.images is not None
                                   and {r.status for r in book_page.images.rows} == {"completed"}, 30), \
                [r.status for r in book_page.images.rows]
            assert book_page.images.tab_count == f"{chapters}/{chapters}"

            # 4. on disk: the figures, the images (fixed names), the progress file
            import interactive_images_core as core
            import interactive_images_progress_core as ipc

            progress = json.loads(Path(core.progress_path(workspace)).read_text(encoding="utf-8"))
            assert {e["status"] for e in progress["chapters"].values()} == {"completed"}
            for chapter in ipc.translated_chapters(workspace):
                html = Path(chapter.path).read_text(encoding="utf-8")
                name = core.image_filename(chapter.position, 1)
                assert f'src="Interactive_Images/{name}"' in html, chapter.key
                assert Path(workspace, "Interactive_Images", name).read_bytes() == FAKE_PNG
            assert not [r for r in server.requests[before:] if r.kind == "translation"]
        except Exception:
            for row in run.tester.dump(200):
                print(row)
            raise
        finally:
            app.jobs.close()
            await tf._stop(app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))


def test_run_after_each_translation_illustrates_inside_the_translation_job(isolated):
    """With "Run after each translation" on, the translation job itself runs the step once the run ended normally
    (the shared ``translation_pipeline`` post-run hook, desktop and mobile alike); a failure there only logs."""
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    picks, files = isolated
    tf = I7._load("test_ui_foundations.py", "_glossarion_tf_helpers_interactive_images_auto")

    async def scenario(server):
        run = await I7._start(tf, server, picks, files)
        app, driver = run.app, run.driver
        js = app.job_service
        try:
            app.config_store.set_many({"interactive_images_image_model": IMAGE_MODEL, "auto_glossary_mode": "off",
                                       "interactive_images_auto_after_translation": True})
            await I7._open_library(run)
            book = await I7._import_book(run, f"{I7.BOOK_STEM}.epub")
            await I7._open_chapters(run, book)
            await driver.tap(key="ch-run", timeout=10)
            await driver.wait(key="start", timeout=30)
            await driver.tap(key="start", timeout=10)
            if await driver.exists(text="Keep translations running", timeout=2):
                await driver.tap(text="Not now")
            snap = await _translation(js)
            done = await I7._wait_end(js, snap.id)
            assert I7._state(done) == "DONE", getattr(done, "error", done)
            kinds = [r.kind for r in server.requests]
            first_planner = kinds.index("planner")
            assert "translation" not in kinds[first_planner:], kinds  # after the whole translation
            assert kinds.count("planner") == kinds.count("image_generation") >= I7.CHAPTERS
            assert [s for s in js.view().history if I7._kind(s) == "interactive_images"] == []  # no extra job

            import interactive_images_progress_core as ipc

            model = ipc.open_progress(book["output_folder"])
            assert {r.status for r in ipc.image_rows(model)} == {"completed"}
        except Exception:
            for row in run.tester.dump(200):
                print(row)
            raise
        finally:
            app.jobs.close()
            await tf._stop(app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))