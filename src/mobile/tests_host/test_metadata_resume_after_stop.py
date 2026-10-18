"""Regression: a graceful Stop during the metadata phase must keep the translated metadata.

The owner saw the book metadata translated again on every Resume. A graceful stop raised while the
metadata request ran (``TransateKRtoEN.main``'s metadata phase) returned before ``metadata.json`` was
written, while ``translation_progress.json`` already said the metadata phase was completed; Resume
read the untranslated ``metadata.json`` and sent the metadata request again. Now the stop records
the fields that completed and writes ``metadata.json`` with their ``*_translated`` flags before
stopping (``_stop_metadata_phase_keeping_results`` / ``_save_metadata_json``).

The real app on the fake OpenAI server (``diagnostics.fake_llm_server``, the helpers of
``test_devfix_issue4.py`` as ``test_attachment_continues_library.py`` uses them): New chat, ＋ › Files
with the self-test EPUB, Send, Start; then the composer's Stop (a graceful stop) at one of two moments:

1. ``test_stop_after_the_metadata_response_keeps_it_for_resume``: right after the metadata response
   was accepted (``MetadataTranslator.translate_metadata`` returned; the pipeline goes on only once
   the stop is raised), so the stop lands between the response and the end of the metadata phase.
2. ``test_stop_during_the_metadata_request_keeps_it_for_resume``: while the fake server holds the
   metadata request (``hold()`` from its ``on_request`` hook); the request is released once the stop
   is raised, and its response must be kept (``MetadataTranslator._send_with_retry`` discards a
   response only on a hard cancel, not on a graceful stop).

Either way the run ends Stopped; ``metadata.json`` in the run's workspace has ``title_translated`` and
the other ``*_translated`` flags; the card's Resume sends NO second metadata request and translates
the chapters.

Run from src/mobile (mobile venv; ``tools/collect_backend.py`` first, so app/backend has the fix)::

    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_metadata_resume_after_stop.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import threading
import time
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR), str(MOBILE_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _load("_glossarion_devfix4_metadata_resume", "test_devfix_issue4.py")  # the app-on-fake-server helpers
pytestmark = D.pytestmark
storage = D.storage
app_env = D.app_env
iso = D.iso

METADATA_PROMPT = "Fields to translate:"  # MetadataTranslator._translate_fields_together's user prompt


def _metadata_requests(server, since: int = 0) -> list:
    """The metadata requests the fake server received (any status)."""
    return [r for r in server.records(since=since, status=None)
            if r.kind == "translation" and str(r.preview or "").startswith(METADATA_PROMPT)]


def _translated_flags(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if k == "title_translated" or k.endswith("_translated")}


async def _until_answering(driver, predicate, timeout: float) -> bool:
    """``D._until`` that answers the first long job's "Keep translations running" with "Not now" (the
    Android battery prompt, asked once per fresh storage, as ``D._send`` / ``_job_done`` answer it)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        if await driver.count(text="Keep translations running"):
            await driver.tap(text="Not now")
        await asyncio.sleep(0.05)
    return bool(predicate())


async def _stop_and_resume(app, tester, driver, server, epub: Path, wait_for_metadata, release) -> None:
    """Translate ``epub`` in a new chat, tap Stop once ``wait_for_metadata`` says the metadata request
    reached the chosen moment, ``release`` the pipeline / request, then check metadata.json and Resume."""
    import flows

    await flows.wait_home(driver)
    D._configure(app, server)
    runs = app.chat_feature.runs
    chats = app.chat_feature.chats

    # ---- New chat, ＋ › Files, Send, Start ----------------------------------------------------------
    await driver.tap(tooltip="New chat")
    await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
    await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
    await driver.wait(contains=epub.stem, timeout=60)
    await driver.tap(key="send-idle_ready", timeout=60)
    await driver.wait(text="Ready to translate", timeout=60)
    await driver.tap(text="Start")
    cid = app.chat_view.cid
    assert await _until_answering(driver, wait_for_metadata, D.RUN_TIMEOUT), ("the metadata phase never reached the chosen moment", [(r.kind, r.status, r.parked, r.preview[:60]) for r in server.records(status=None)])
    assert len(_metadata_requests(server)) == 1, [(r.status, r.preview) for r in _metadata_requests(server)]

    # ---- the composer's Stop (graceful), then the metadata phase goes on ------------------------------
    try:
        await driver.tap(tooltip="Stop translation", timeout=15)
        assert await D._until(lambda: runs.run_for(cid) is not None and runs.run_for(cid).stop_requested, 10)
    finally:
        release()
    run = runs.run_for(cid)
    assert await D._until(lambda: not run.live, D.RUN_TIMEOUT), run.state
    await D._join_finish_threads(runs)
    snap = app.job_service.snapshot(run.job_id)
    assert run.state == "stopped", (run.state, snap.state, snap.stop_mode, snap.last_line)
    assert snap.stop_mode == "graceful", snap.stop_mode
    first = _metadata_requests(server)
    assert len(first) == 1 and first[0].status == "ok", [(r.status, r.preview) for r in first]
    folders = chats.attachment_folders(cid)
    assert len(folders) == 1, folders
    workspace = Path(folders[0])

    # ---- metadata.json kept what the response translated ---------------------------------------------
    metadata_file = workspace / "metadata.json"
    assert metadata_file.is_file(), sorted(p.name for p in workspace.iterdir())
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    flags = _translated_flags(metadata)
    assert metadata.get("title_translated") is True, metadata
    assert flags and all(v is True for v in flags.values()), flags
    assert D._progress(workspace)["completed"] < D.CHAPTERS

    # ---- Resume: no second metadata request, the chapters go on ---------------------------------------
    mark = server.mark()
    view = app.chat_view
    view.render_transcript()
    assert await D._until(lambda: any("resume" in c.action_buttons for c in D._job_cards(view)), 15)
    card = next(c for c in D._job_cards(view) if "resume" in c.action_buttons)
    await tester._dispatch(card.action_buttons["resume"], "click")
    assert await D._until(lambda: runs.run_for(cid) is not None and runs.run_for(cid).live, 30)
    resumed = runs.run_for(cid)
    await D._finish(app, resumed)
    assert resumed.state == "done", resumed.state
    again = _metadata_requests(server, since=mark)
    assert again == [], f"Resume translated the metadata again: {[r.preview for r in again]}"
    chapters = [r for r in server.records("translation", since=mark) if r.chapters]
    assert chapters, [(r.kind, r.preview) for r in server.records(since=mark, status=None)]


def _selftest_copy(iso) -> Path:
    epub = iso.picks / "Metadata Book.epub"
    shutil.copyfile(D.SELFTEST_EPUB, epub)
    return epub


def test_stop_after_the_metadata_response_keeps_it_for_resume(iso, monkeypatch):
    import metadata_batch_translator

    epub = _selftest_copy(iso)
    # the pipeline waits right after the metadata response until the Stop is raised
    answered = threading.Event()
    stop_raised = threading.Event()
    original = metadata_batch_translator.MetadataTranslator.translate_metadata

    def translate_metadata(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not answered.is_set():
            answered.set()
            stop_raised.wait(60)
        return result

    monkeypatch.setattr(metadata_batch_translator.MetadataTranslator, "translate_metadata", translate_metadata)

    async def scenario(server):
        tf, app, tester, driver = await D._start_app({epub.name: epub})
        try:
            await _stop_and_resume(app, tester, driver, server, epub, answered.is_set, stop_raised.set)
        finally:
            stop_raised.set()
            await D._stop_app(tf, app)

    D._run_scenario(scenario)


def test_stop_during_the_metadata_request_keeps_it_for_resume(iso):
    epub = _selftest_copy(iso)

    async def scenario(server):
        def hold_metadata(record) -> None:
            """The metadata request parks in the fake server (``hold``) until the Stop is raised."""
            if str(record.preview or "").startswith(METADATA_PROMPT) and not _metadata_requests(server)[:-1]:
                server.hold()

        tf, app, tester, driver = await D._start_app({epub.name: epub})
        try:
            server.on_request.append(hold_metadata)
            def parked() -> bool:
                return server.parked >= 1 and any(r.parked and r.status == "pending" for r in _metadata_requests(server))

            await _stop_and_resume(app, tester, driver, server, epub, parked, server.release)
        finally:
            server.release()
            await D._stop_app(tf, app)

    D._run_scenario(scenario)
