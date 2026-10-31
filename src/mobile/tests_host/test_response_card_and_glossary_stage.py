"""Owner's device report of 2026-10-31.

1. A response card's dropdowns (Run options, Requests, the report, OCR) close on a tap away: the extension's
   ``TapAway`` (``ui.components.tap_away``), which only reports a tap on the page outside an open dropdown.
2. Glossary progress › Continue / Extract glossary waits exactly like Chapters › Run translation.
3. Glossary extraction reads "Indexing" on the notification, the Jobs page, the job strip and the card: an
   extraction job (its verb) and a book job in its glossary stage (``job_kinds.translate.indexing_phase``).

The real-app test drives a job through the app's own UI against the fake model, parked inside its glossary stage.

Run from src/mobile:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_response_card_and_glossary_stage.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _path in (APP_DIR, MOBILE_DIR / "tests", MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

ft = pytest.importorskip("flet")
DART = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src" / "flutter" / "flet_glossarion_native" / "lib" / "src"


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


I7 = _load("test_hotfix1_issue7.py", "_glossarion_i7_helpers_glossary_stage")
storage = I7.storage  # noqa: F811  (pytest fixtures)
app_env = I7.app_env
isolated = I7.isolated


@pytest.fixture
def tap_away_installed(monkeypatch):
    from flet_glossarion_native import TapAway

    from glossarion_mobile.ui.components import tap_away

    monkeypatch.setattr(tap_away, "_region_class", TapAway)
    return TapAway


def _event(value):
    return types.SimpleNamespace(data=value)


# ---- 1. dropdowns close on a tap away -----------------------------------------------------------------------


def test_dart_tap_away_reads_the_python_controls_properties():
    from flet_glossarion_native import TapAway

    assert 'case "GlossarionTapAway":' in (DART / "extension.dart").read_text(encoding="utf-8")
    dart = (DART / "tap_away.dart").read_text(encoding="utf-8")
    read = set(re.findall(r'(?:getBool|buildWidget)\("([a-z_]+)"', dart))
    own = {f.name for f in dataclasses.fields(TapAway)} - {f.name for f in dataclasses.fields(ft.LayoutControl)}
    assert read | {"on_tap_away"} == own == {"content", "enabled", "on_tap_away"}
    assert 'triggerEvent("tap_away")' in dart
    assert TapAway()._c == "GlossarionTapAway"


def test_a_client_without_the_control_keeps_the_bare_tile(monkeypatch):
    from glossarion_mobile.ui.components import tap_away

    monkeypatch.setattr(tap_away, "_region_class", None)
    tile = ft.ExpansionTile(title="Run options")
    assert tap_away.closes_on_tap_away(tile) is tile
    assert tile.on_change is None


def test_install_finds_the_control_on_the_phone_app_only(monkeypatch):
    from flet_glossarion_native import TapAway

    from glossarion_mobile.ui.components import tap_away

    monkeypatch.setattr(tap_away, "_region_class", None)
    monkeypatch.delenv("GLOSSARION_FORCE_NATIVE_STUB", raising=False)
    monkeypatch.delenv("GLOSSARION_NATIVE_DISABLE", raising=False)
    tap_away.install_tap_away(types.SimpleNamespace(platform="windows", web=False))
    assert tap_away._region_class is None
    tap_away.install_tap_away(types.SimpleNamespace(platform=types.SimpleNamespace(value="android"), web=False))
    assert tap_away._region_class is TapAway


def test_a_tile_the_user_opened_closes_on_a_tap_away(tap_away_installed):
    from glossarion_mobile.ui.components.tap_away import closes_on_tap_away

    seen = []
    tile = ft.ExpansionTile(title="Requests", on_change=lambda e: seen.append(e.data))
    region = closes_on_tap_away(tile)
    assert isinstance(region, tap_away_installed) and region.content is tile
    assert closes_on_tap_away(tile) is region  # a repaint re-adds the same region
    assert region.enabled is False  # shut: a tap anywhere sends nothing

    tile.on_change(_event(True))
    assert region.enabled is True and seen == [True]  # the tile's own handler still runs
    tile.expanded = True  # what the client reports back with the event
    region.on_tap_away(None)
    assert tile.expanded is False and region.enabled is False

    tile.on_change(_event("true"))
    assert region.enabled is True
    tile.on_change(_event(False))  # closed from its own header
    assert region.enabled is False and seen == [True, "true", False]


def test_a_list_that_opened_by_itself_stays_until_the_user_toggles_it(tap_away_installed):
    from glossarion_mobile.ui.chat.cards import JobCard

    card = JobCard(attachment={"name": "Book.epub"}, requests_expanded=True)
    region = next(c for c in card.body.controls if getattr(c, "content", None) is card.requests_tile)
    assert card.requests_tile.expanded is True and region.enabled is False
    card.requests_tile.on_change(_event(False))
    card.requests_tile.on_change(_event(True))
    assert region.enabled is True


def test_a_hidden_tile_leaves_no_gap_in_its_column(tap_away_installed):
    from glossarion_mobile.ui.components.tap_away import closes_on_tap_away

    tile = ft.ExpansionTile(title="OCR (0)", visible=False)
    region = closes_on_tap_away(tile)
    region.before_update()
    assert region.visible is False
    tile.visible = True
    region.before_update()
    assert region.visible is True


def test_every_dropdown_of_the_response_card_closes_on_a_tap_away(tap_away_installed):
    from glossarion_mobile.ui.chat.cards import JobCard

    card = JobCard(attachment={"name": "Book.epub"})
    held = [c.content for c in card.body.controls if isinstance(c, tap_away_installed)]
    assert held == [card.requests_tile, card.ocr_tile, card.report_tile]
    options = ft.ExpansionTile(title="Run options")
    card.set_run_options(options)
    assert isinstance(card.options_box.controls[0], tap_away_installed)
    assert card.options_box.controls[0].content is options
    card.set_run_options(None)
    assert card.options_box.controls == []


# ---- 3. "Indexing" ------------------------------------------------------------------------------------------


def _snap(kind="translate", **kwargs):
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState, Progress

    return JobSnapshot(id="j", spec=JobSpec(kind, "Book.epub"), state=JobState.RUNNING, created=0.0, started=0.0,
                       progress=kwargs.pop("progress", Progress()), in_flight=4, **kwargs)


def test_a_book_job_in_its_glossary_stage_reads_indexing_everywhere():
    from glossarion_mobile.services.jobs import (notification_lines, position_line, progress_line, progress_view,
                                                 strip_model_for)
    from glossarion_mobile.ui.chat.job_binding import running_label

    snap = _snap(phase="Indexing", position="[118/177] Ch. 116")
    assert notification_lines(snap)[0] == "Indexing • [118/177] Ch. 116 · 4 in flight"
    assert position_line(snap) == "Indexing • [118/177] Ch. 116"
    assert progress_line(snap, 0.0, verb=True).startswith("Indexing • [118/177] Ch. 116 · 4 in flight")
    assert strip_model_for(snap).title == "Indexing · Book.epub"
    assert progress_view(snap) == "glossary"
    assert running_label(snap) == "Indexing"


def test_the_stage_shows_even_with_translation_progress_from_an_earlier_run():
    from glossarion_mobile.services.jobs import Progress, position_line, progress_view

    snap = _snap(phase="Indexing", position="[9/177] Ch. 8", progress=Progress(total=177, completed=30))
    assert position_line(snap) == "Indexing • [9/177] Ch. 8"
    assert progress_view(snap) == "glossary"


def test_the_translation_stage_and_other_jobs_keep_their_words():
    from glossarion_mobile.services.jobs import position_line, progress_view, strip_model_for
    from glossarion_mobile.ui.chat.job_binding import running_label

    snap = _snap(phase="Translating", position="[9/177] Ch. 8")
    assert position_line(snap) == "Translating • [9/177] Ch. 8"
    assert strip_model_for(snap).title == "Translating · Book.epub"
    assert running_label(snap) == "Translating"
    assert progress_view(_snap(phase="Translating")) == "chapters"
    assert position_line(_snap("qa_scan", phase="Scanning")).startswith("QA scanning")


def test_an_extraction_job_reads_indexing():
    from glossarion_mobile.services.jobs import position_line, strip_model_for

    snap = _snap("extract_glossary", phase="Generating glossary…", position="[28/177] Ch. 26")
    assert position_line(snap) == "Indexing • [28/177] Ch. 26"
    assert strip_model_for(snap).title == "Indexing · Book.epub"


def test_the_phase_follows_the_glossary_pass_and_leaves_the_owner_clean():
    from glossarion_mobile.job_kinds.translate import indexing_phase

    phases = []
    calls = []

    class Owner:
        def run_glossary_extraction_direct(self, force_balanced_request_merging=False):
            calls.append((force_balanced_request_merging, phases[-1]))
            return "done"

    owner = Owner()
    ctx = types.SimpleNamespace(owner=owner, phase=phases.append)
    with indexing_phase(ctx):
        assert owner.run_glossary_extraction_direct(force_balanced_request_merging=True) == "done"
    assert calls == [(True, "Indexing")]
    assert phases == ["Indexing", "Translating"]
    assert "run_glossary_extraction_direct" not in vars(owner)

    def failing():
        raise RuntimeError("stopped")

    owner.run_glossary_extraction_direct = failing
    phases.clear()
    with indexing_phase(ctx):
        with pytest.raises(RuntimeError):
            owner.run_glossary_extraction_direct()
    assert phases == ["Indexing", "Translating"]  # a failed pass does not leave the job Indexing


def test_the_backends_own_glossary_pass_reports_its_stage():
    import TransateKRtoEN as backend

    from glossarion_mobile.job_kinds.translate import indexing_phase

    phases = []

    class Owner:
        def _current_auto_glossary_mode(self):
            return "minimal"

    with indexing_phase(types.SimpleNamespace(owner=Owner(), phase=phases.append)):
        backend._report_glossary_stage(True)
        backend._report_glossary_stage(False)
    assert phases == ["Indexing", "Translating"]
    assert backend._glossary_stage_callback is None
    backend._report_glossary_stage(True)  # no host: nothing to tell


# ---- 2 + 3 in the real app ----------------------------------------------------------------------------------


def test_a_real_run_reads_indexing_and_continue_waits_for_it(isolated):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import notification_lines, position_line, progress_view, strip_model_for
    from glossarion_mobile.ui.components.tap_away import closes_on_tap_away
    from glossarion_mobile.ui.library import book_page as bp

    picks, files = isolated
    tf = I7._load("test_ui_foundations.py", "_glossarion_tf_helpers_glossary_stage")

    async def scenario(server):
        gate = I7.ReplyGate(server)
        run = await I7._start(tf, server, picks, files)
        app, driver = run.app, run.driver
        js = app.job_service
        try:
            app.config_store.set_many({"auto_glossary_mode": "full", "extraction_workers": 1})
            await I7._open_library(run)
            book = await I7._import_book(run, f"{I7.BOOK_STEM}.epub")
            book_page, _bid = await I7._open_chapters(run, book)
            tab = book_page.glossary_tab
            assert tab.extract_button.disabled is False and tab.extract_button.tooltip is None

            # a dropdown on this (android) client: what the extension's events do, as the client would send them
            tile = ft.ExpansionTile(title="Run options", key="wire-tile")
            region = closes_on_tap_away(tile)
            assert type(region).__name__ == "TapAway"
            run.page.overlay.append(region)
            run.page.update()
            wire = len(run.conn.messages)
            await run.session.dispatch_event(tile._i, "change", True)
            assert region.enabled is True and run.patched_since(wire, region, "enabled")
            tile.expanded = True
            run.page.update()
            wire = len(run.conn.messages)
            await run.session.dispatch_event(region._i, "tap_away", None)
            assert tile.expanded is False and region.enabled is False
            assert run.patched_since(wire, tile, "expanded")
            wire = len(run.conn.messages)
            tile.visible = False
            region.update()  # a hidden dropdown: the region goes with it and leaves no gap
            assert region.visible is False and run.patched_since(wire, region, "visible")
            run.page.overlay.remove(region)
            run.page.update()

            # the job's glossary stage, parked after its first answer
            gate.hold_after("glossary", 1)
            mark = len(run.conn.messages)
            await driver.tap(key="ch-run", timeout=10)
            await driver.wait(key="start", timeout=30)
            await driver.tap(key="start", timeout=10)
            if await driver.exists(text="Keep translations running", timeout=2):
                await driver.tap(text="Not now")
            snap = await I7._until(lambda: js.snapshot(), 60)
            assert snap is not None and I7._kind(snap) == "translate"
            job_id = snap.id
            assert await I7._until(lambda: I7._settled(server), 120), js.snapshot(job_id)
            assert await I7._until(lambda: js.snapshot(job_id).position, 30), js.snapshot(job_id)
            snap = js.snapshot(job_id)
            assert snap.phase == "Indexing"
            assert position_line(snap).startswith("Indexing • ")
            assert notification_lines(snap)[0].startswith("Indexing • ")
            assert strip_model_for(snap).title.startswith("Indexing · ")
            assert progress_view(snap) == "glossary"

            # Continue waits like Run translation, on the screen and on the client
            waiting = (True, bp.RUN_WAIT_TRANSLATION)
            button = tab.extract_button
            assert await I7._until(lambda: (button.disabled, button.tooltip) == waiting, 10), button.tooltip
            assert I7._buttons(book_page)[0] == waiting
            assert bp.RUN_WAIT_TRANSLATION.encode() in run.sent_since(mark)
            assert run.patched_since(mark, button)
            assert await tab.extract() is None  # nothing is queued behind a stale tap
            assert not js.view().queue

            # the translation stage: back to "Translating", Continue still waits
            gate.release()
            gate.hold_after("translation", 1)
            assert await I7._until(lambda: js.snapshot(job_id).phase == "Translating", 120), js.snapshot(job_id)
            assert await I7._until(lambda: I7._settled(server), 120), js.snapshot(job_id)
            assert position_line(js.snapshot(job_id)).startswith("Translating")
            assert (button.disabled, button.tooltip) == waiting

            # the run ends: Continue is back
            gate.release()
            end = await I7._wait_end(js, job_id)
            assert I7._state(end) == "DONE", end
            assert await I7._until(lambda: button.disabled is False and button.tooltip is None, 20), button.tooltip
        except Exception:
            for row in run.tester.dump(200):
                print(row)
            raise
        finally:
            gate.release()
            app.jobs.close()
            await tf._stop(app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))
