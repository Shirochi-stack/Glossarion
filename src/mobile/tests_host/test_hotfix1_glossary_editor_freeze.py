"""Hotfix 1 issue 1, reproduced with the real pipeline: "Unable to use glossary editor during translation
without freezing up the entire app".

Root cause (reproduced first on the base commit with this scenario, UI loop stall watchdog + patch log):
Flet 1.0.3 auto-updates the nearest isolated ancestor - the Page - after an event handler that called no
``update()``. The Glossary editor's handlers mostly spawned a coroutine or relied on that, so every scroll
event (the list reports one each 120 ms while the user scrolls: ``WindowedList`` without ``quiet_scroll``),
every keystroke in the search box and most taps sent a whole-page diff: every View, i.e. the chat behind the
editor with its live job card, the editor with all its mounted rows and its four eagerly built settings tabs
(25 scroll events: 25 Page patches, 0.6-1.0 s of diffing on the host; on a phone several times that, while
the events keep coming every 120 ms - the loop never catches up). On top of that every render sent the
whole editor root (all mounted rows walked and rebuilt), a one-row edit sent the root, each scroll step
walked every mounted row again, and the running job's own repaints and threads compete for the same loop
and GIL. Nothing waited on ``job_runner.JOB_LOCK`` (the editor's tools take it without waiting).

The fix (ui/glossary/editor.py, glossary_view.py, common.py): quiet handlers and ``quiet_scroll`` (no
auto-update), renders that send the header and the list (never the page), reused / isolated row objects
(a list update compares rows, not their cards; a changed row is sent alone), steps whose rows past
``FULL_ROWS_PER_STEP`` follow a chunk per loop step, settings tabs built on first show, and an auto-reload
that swaps in a new document read off the loop.

This test runs the owner's case end to end on the real app objects (``test_devfix_issue12``'s harness: the app
on a fake Flet session, the real JobService on the app's UiDispatcher, HeadlessOwner and the shared
translation pipeline; only the model is the offline fake OpenAI server, and the E2E network guard refuses
every non-loopback connection): a 120-chapter EPUB attached in a chat, Balanced glossary approved, and while
its chapters translate the book's Glossary editor is opened, scrolled (scroll events as Flutter sends them),
searched (keystrokes), edited through the entry sheet and saved; then another book's 3,000-entry glossary
(with a tracked gender conflict) gets the same plus a gender choice from the row's menu. Throughout: no
whole-page patch, a one-row edit sends that row, Save sends no list, JOB_LOCK stays the job's, and the UI
loop's largest gap stays small. The fake model holds after a share of the chapters (``HoldGate``), so the job
is still RUNNING at every check on any runner.

Run from src/mobile with the mobile env (on the dev box ``.venv``, Python 3.12 with the shell PYTHONPATH):
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_glossary_editor_freeze.py
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent), str(MOBILE_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


I12 = _load("_glossarion_devfix12_hotfix1_freeze", "test_devfix_issue12.py")  # the real-pipeline harness
E = _load("_glossarion_editor_hotfix1_freeze", "test_hotfix1_glossary_editor.py")  # patch log, heartbeat, fixtures
pytestmark = I12.pytestmark
storage = I12.storage
app_env = I12.app_env
iso = I12.iso
offline = I12.offline

CHAPTERS = 120
#: the fake model answers each chapter after this long (wall time: the job keeps translating meanwhile)
TRANSLATION_DELAY = 0.6
#: the fake model holds after this many translation replies (the job then waits, RUNNING, until teardown)
HOLD_AFTER = 45
#: generous for slow CI runners; on the host the editor's interactions measured 20-60 ms with the job running
LOOP_GAP_S = 0.3
SCROLL = {"event_type": "update", "pixels": 120.0, "max_scroll_extent": 99999.0, "min_scroll_extent": 0.0,
          "viewport_dimension": 600.0}
NEAR_END = dict(SCROLL, pixels=99800.0, max_scroll_extent=100000.0)
WHOLE = ("Page", "View")
LIST = ("ge-list", "ge-list-holder", "glossary-editor", "glossary-view")


def _server():
    """The fake model; its glossary answers include the two terms every chapter of the test book has."""
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer, GlossaryEntry

    server = FakeLLMServer()
    server.glossary = server.glossary + (GlossaryEntry("term", "마왕성", "Demon King's Castle", "", "The demon's keep"),
                                         GlossaryEntry("term", "검", "Sword", "", "A blade"))
    server.set_delay("translation", TRANSLATION_DELAY)
    return server


def _job_lock_free() -> bool:
    import job_runner

    free = job_runner.JOB_LOCK.acquire(blocking=False)
    if free:
        job_runner.JOB_LOCK.release()
    return free


async def _interact(env, editor, log, *, search: str, edit_text: str) -> None:
    """The owner's use of the editor: scroll, steps near the end, typing, a row's sheet, an edit, Save."""
    session = env.session
    lv = editor.windowed.list_view
    for i in range(10):
        await session.dispatch_event(lv._i, "scroll", dict(SCROLL, pixels=120.0 + 50 * i))
        await asyncio.sleep(0.12)
    for _ in range(3):
        before = editor.windowed.mounted_count
        await session.dispatch_event(lv._i, "scroll", NEAR_END)
        assert editor.windowed.mounted_count > before or editor.windowed.mounted_count == len(editor.visible)
        await asyncio.sleep(0.12)
    assert await E._settle(lambda: not editor._fill_queue, 30)
    for i in range(1, len(search) + 1):
        editor.search.value = search[:i]
        await session.dispatch_event(editor.search._i, "change", search[:i])
        await asyncio.sleep(0.08)
    await asyncio.wait_for(editor._search_task, 30)
    assert editor.visible and all(search.casefold() in json.dumps(s.entry, ensure_ascii=False).casefold()
                                  for s in editor.visible), [s.entry for s in editor.specs[:12]]
    editor.search.value = ""
    await session.dispatch_event(editor.search._i, "change", "")
    await asyncio.wait_for(editor._search_task, 30)
    spec = editor.visible[1]
    await session.dispatch_event(E._row(editor, spec.key).content._i, "click", None)  # the row's sheet
    sheet = editor.sheets[-1]
    log.clear()
    field = sheet.inputs["translated_name"]
    session.apply_patch(field._i, {"value": edit_text})  # the user types it (the client's patch; no handler)
    sheet.save()
    assert ("RowShell", spec.key) in log.sent, log.sent
    assert not [k for k in log.sent if k[0] in WHOLE or k[1] in LIST], log.sent
    assert f"→ {edit_text}" in E._texts(E._row(editor, spec.key)) and editor.dirty
    log.clear()
    editor.ctx.extras["answers"] = [True]  # the desktop "Update output files" question: Yes
    await session.dispatch_event(editor.save_button._i, "click", None)
    assert await E._settle(lambda: not editor.dirty and not editor._saving, 60)
    assert editor.reports and editor.reports[-1].get("saved"), editor.reports[-1:]
    assert not [k for k in log.sent if k[0] in WHOLE or k[1] in LIST], log.sent
    assert f"→ {edit_text}" in E._texts(E._row(editor, spec.key))


def _clear_client_stop() -> None:
    """The teardown's force stop leaves the shared client's process-wide stop flags set
    (``unified_api_client.global_stop_flag`` / ``UnifiedClient._global_cancelled``); the next job's start
    clears them (``stop_control.clear_client_cancellation``), but the tests after this one in the same
    process may start none (the on-device engine reads them as a Stop): clear them like a run start."""
    client = sys.modules.get("unified_api_client")
    if client is not None and hasattr(client, "set_stop_flag"):
        client.set_stop_flag(False)


def test_glossary_editor_stays_responsive_while_a_book_translates(iso, offline):
    async def body(server):
        import flows

        epub = I12._book(iso.picks, CHAPTERS)
        tf = I12._UF._foundations()
        app, tester, driver = await I12._UF._host_driver(tf, {I12.EPUB_NAME: epub}, width=320)
        app.job_service.watcher_interval = 0.2
        gate = I12.HoldGate(server)
        run = None
        beat = E.Heartbeat()
        log = E.PatchLog()
        try:
            await flows.wait_home(driver)
            I12._assert_isolated(iso.tmp)
            I12._configure(app, server, "balanced")
            data = Path(app.paths.data)
            app.glossary.app_dir = lambda: str(data)  # the glossary list never reads the checkout's Glossary/
            await flows.go_home(driver)
            await driver.tap(tooltip="New chat")
            await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
            await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
            await driver.wait(contains=epub.stem, timeout=60)
            await driver.tap(key="send-idle_ready", timeout=60)
            await driver.wait(text="Ready to translate", timeout=60)
            await driver.tap(text="Start")
            cid = app.chat_view.cid
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and run is None:
                if await driver.count(text="Keep translations running"):
                    await driver.tap(text="Not now")
                candidate = app.chat_feature.runs.run_for(cid)
                run = candidate if candidate is not None and candidate.job_id is not None else None
                await asyncio.sleep(0.05)
            assert run is not None, "Start did not submit the chat's job"
            probe = I12.Probe(app, tester, cid)
            assert await I12._until(lambda: run.awaiting_glossary, I12.RUN_TIMEOUT), probe.describe()
            await asyncio.sleep(0.6)
            gate.hold_after(HOLD_AFTER)
            await I12._tap(probe, app.chat_view.approval_card.yes_button)
            assert await I12._until(lambda: probe.job()["completed"] >= 3, I12.RUN_TIMEOUT), probe.describe()
            assert probe.job()["state"] == "RUNNING" and not _job_lock_free()

            book = next(iter(sorted(data.rglob(f"{epub.stem}_glossary.csv"))), None)
            assert book is not None, "the Balanced glossary of the book was not written"
            other = E.conflict_book(data / "Glossary" / "Other")
            E.G.write_token_glossary(Path(other), E.ENTRIES + E.big_entries(3000)[3:])  # 루나 keeps its conflict
            gc.collect()
            gc.freeze()  # a full GC of this long single-process run's heap stalls the loop on its own
            beat.start()
            with log.watching():
                # ---- the book being translated ------------------------------------------------------
                await app.navigate(f"/glossary/{app.glossary.gid_for(str(book))}")
                screen = app.shell.top_screen
                assert await E._settle(lambda: getattr(screen, "editor", None) is not None
                                       and screen.editor.doc is not None, 60)
                editor = screen.editor
                env = type("Env", (), {"session": tester.session})()
                assert all(slot.content is None for slot in screen.tab_slots.values())  # tabs: on first show
                log.clear()
                await _interact(env, editor, log, search="Castle", edit_text="Edited during the run")
                assert probe.job()["state"] == "RUNNING" and not _job_lock_free()
                # ---- another book's 3,000-entry glossary ------------------------------------------------
                assert await screen.switch_to(other)
                assert await E._settle(lambda: editor.doc is not None and len(editor.specs) == 3000, 60)
                await _interact(env, editor, log, search="Name 12", edit_text="Renamed while translating")
                luna = editor.specs[0]
                await E._settle(lambda: not editor._fill_queue, 30)
                menu = E._find(E._row(editor, luna.key), f"ge-gender-{luna.key}")
                assert menu is not None, "the conflict row has no gender menu"
                log.clear()
                male = next(item for item in menu.items if item.content == "Male")
                await tester.session.dispatch_event(male._i, "click", None)
                assert editor.dirty and "⚠ Male" in E._texts(E._row(editor, luna.key))
                assert not [k for k in log.sent if k[0] in WHOLE or k[1] in LIST], log.sent
                await tester.session.dispatch_event(editor.save_button._i, "click", None)
                assert await E._settle(lambda: not editor.dirty and not editor._saving, 60)
                assert probe.job()["state"] == "RUNNING" and not _job_lock_free()
            gap = await beat.stop()
            whole = [k for k in log.sent if k[0] in WHOLE]
            # the navigation to the editor and the file switch update the page once each (the shell's own
            # navigation); no editor event did
            assert len(whole) <= 1, whole
            assert gap < LOOP_GAP_S, f"the UI loop stalled {gap * 1000:.0f} ms"
            saved = json.loads(Path(other).with_name("Book_gender_tracker.json").read_text(encoding="utf-8"))
            assert saved["entries"]["루나"]["decision"] == "male"
        finally:
            gc.unfreeze()
            if beat.task is not None and not beat.task.done():
                beat.task.cancel()
            try:
                await I12._end_job(app, server, run)
            finally:
                try:
                    app.jobs.close()
                finally:
                    await tf._stop(app)
                    _clear_client_stop()

    with _server() as server:
        asyncio.run(body(server))
    I12._assert_offline(offline)
