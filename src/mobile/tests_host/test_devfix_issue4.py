"""Acceptance test for the owner's device report #4 on the U8 APK (2026-10-08):
"Chat translations auto-migrate into the Library; no manual option".

On the phone, an EPUB translated in the chat stayed in ``Output/Direct Text/<chat>/Attachments/<stem>``
until the owner tapped Migrate (job card or the Attachments manager), and the Library ⋯ offered
Organize / Undo. This file proves the fix end to end on the REAL app objects: the app started on a
fake Flet session (``test_ui_foundations._start`` through ``test_ui_flows._host_driver``, as the other
tests_host UI tests), the real JobService / ChatFeature / ChatRuns / LibraryService, the shared
desktop pipeline and ``direct_text_store.ChatStore``; only the model is replaced by the offline fake
OpenAI server (``diagnostics.fake_llm_server``, the E2E fixture) on 127.0.0.1.

1. ``test_chat_book_joins_the_library_without_a_tap``: the owner's flow through the UI (new chat,
   ＋ › Files, Send, Start). When the run is Done the workspace moves into the Library output root
   with no further tap (one call of the desktop ``ChatStore.migrate_attachment``, made with the real
   ``OUTPUT_DIRECTORY`` while no job runs), the stored chat paths follow, the book is on the
   Library's Completed shelf (Book page › Chapters 12/12), the job card offers "Open in Library"
   (enabled, opens that Book page) and no Migrate, the Library ⋯ has no Organize / Undo and the
   Attachments manager has no Migrate.
2. ``test_the_next_queued_job_never_receives_the_move``: the race a skeptic probe showed (the shared
   Migrate reads the live ``OUTPUT_DIRECTORY``, which the next job points at its own temporary run
   root): chat B's book is queued behind chat A's and parks on the model while A finishes. A's move
   waits (nothing appears in B's run root), and both books land in the real output root once the
   queue is idle.
3. ``test_name_clash_same_book_merges_other_book_asks``: the same book sent again from another chat
   merges into its Library folder silently; a different book with the same file name stays in the
   chat, the snackbar's "Merge…" opens the desktop "Attachment folder already exists" dialog and
   "Merge and replace" moves it. ``test_a_different_book_at_the_same_inbox_path_is_not_merged_silently``:
   a different book that lands at the same Inbox path (the first copy was deleted) is a different
   book too: it must ask, not replace the Library book's translation.
4. ``test_scratch_chat_and_stopped_run_stay_put``: a scratch chat's book never leaves its scratch
   folder; a run stopped with the composer's Stop stays in Attachments (Resume, "Open in Library"
   off with its reason, idle sweeps leave it, the Attachments manager lists it without Migrate)
   and joins the Library by itself once Resume finishes it.
5. ``test_books_left_by_the_previous_version_join_on_the_next_launch``: the books the owner already
   translated with the U8 APK (no auto-migrate) sit in Attachments; the next launch of the fixed
   app moves them into the Library by itself (startup sweep).

Real data stays untouched: every test runs in temp FLET_APP_STORAGE_* dirs (the bootstrap points
HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR and CONFIG_FILE there) and USERPROFILE / APPDATA /
LOCALAPPDATA point into tmp_path too.

Run from src/mobile (3.13 venv, ``unset PYTHONPATH``)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue4.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_NEEDED = ("flet", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml")
pytestmark = pytest.mark.skipif(not all(_has(m) for m in _NEEDED),
                                reason=f"needs {', '.join(_NEEDED)} (the 3.13 project venv)")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_UF = _load("_glossarion_uiflows_devfix4", "test_ui_flows.py")  # _host_driver / _foundations
storage = _UF.storage
app_env = _UF.app_env

SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py
CHAPTERS = 12
RUN_TIMEOUT = 120.0


# ==========================================================================
# Fixtures and helpers
# ==========================================================================


@pytest.fixture
def iso(tmp_path, monkeypatch, request):
    """The app on temp storage, with the Windows profile variables redirected as well."""
    if not SELFTEST_EPUB.is_file():
        pytest.skip("app/assets/selftest is generated by tools/prepare_assets.py")
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    request.getfixturevalue("app_env")  # FLET_APP_STORAGE_* + bootstrap (HOME, OUTPUT_DIRECTORY, Library, config)
    picks = tmp_path / "picks"
    picks.mkdir()
    return types.SimpleNamespace(tmp=tmp_path, picks=picks, user=user)


class Calls(list):
    """The recorded Migrate calls; ``probe.service`` is the app's JobService once it runs."""

    probe: types.SimpleNamespace


@pytest.fixture
def migrate_calls(iso, monkeypatch):
    """Every call of the desktop Migrate (``direct_text_store.ChatStore.migrate_attachment``, the
    one the chat reuses), with what the process looked like at that moment."""
    import direct_text_store
    import job_runner

    calls = Calls()
    probe = types.SimpleNamespace(service=None)
    original = direct_text_store.ChatStore.migrate_attachment

    def spy(self, session, source_folder, *, confirm_merge=None):
        free = job_runner.JOB_LOCK.acquire(blocking=False)  # reentrant: True when free or held by this thread
        if free:
            job_runner.JOB_LOCK.release()
        active = None
        if probe.service is not None:
            try:
                active = probe.service.view().active
            except Exception:
                active = "?"
        entry = {
            "chat": str(session.get("id")),
            "folder": str(source_folder),
            "output_directory": os.environ.get("OUTPUT_DIRECTORY"),
            "lock_free_or_ours": free,
            "active_job": getattr(active, "id", active),
            "merge_callback": confirm_merge is not None,
            "thread": threading.current_thread().name,
        }
        result = original(self, session, source_folder, confirm_merge=confirm_merge)
        entry["result"] = result
        calls.append(entry)
        return result

    monkeypatch.setattr(direct_text_store.ChatStore, "migrate_attachment", spy)
    calls.probe = probe
    return calls


def _same(a, b) -> bool:
    return bool(a) and bool(b) and os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _inside(path, root) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _threads() -> str:
    """Every thread's name and stack (a wait that timed out says where things stand)."""
    import traceback

    frames = sys._current_frames()
    out = []
    for thread in threading.enumerate():
        frame = frames.get(thread.ident)
        stack = "".join(traceback.format_stack(frame)[-6:]) if frame is not None else ""
        out.append(f"--- {thread.name}\n{stack}")
    return "\n".join(out)


async def _until(predicate, timeout: float = 30.0, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return bool(predicate())


async def _join_finish_threads(runs) -> None:
    for thread in list(runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)


def _progress(folder) -> dict:
    """Chapter rows of a workspace's ``translation_progress.json`` (the E2E's reading: chapter rows
    whose response file exists count as completed)."""
    from glossarion_mobile.diagnostics import e2e

    chapters = e2e._progress_chapters(str(folder))
    return {"total": len(chapters), "completed": len(e2e._completed(chapters))}


def _configure(app, server) -> None:
    """The settings the device flows import from desktop (``flows.ui_config``): the fake OpenAI
    endpoint, a dummy key, glossary off, no request spacing."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    store = app.config_store
    store.set_many(flows.ui_config(server.url, FAKE_MODEL))
    store.flush()
    assert store.save_error is None, store.save_error


def _record_notes(app) -> list:
    """Wrap ``app.notify`` (snackbars): ``[(message, action_label, on_action, bar)]``."""
    notes: list = []
    original = app.notify

    def notify(message, action_label=None, on_action=None):
        bar = original(message, action_label, on_action)
        notes.append((str(message), action_label, on_action, bar))
        return bar

    app.notify = notify
    return notes


def _import(app, source: Path, name: str = "") -> str:
    """What ＋ › Files does with a picked file: FileBridge copies it into the Inbox."""
    imported = app.files.import_paths([str(source)], names=[name or source.name])
    assert imported, f"FileBridge did not import {source}"
    return imported[0].path


async def _send(app, driver, cid, path):
    """The chat's send of an attachment turn (``ChatRuns.send`` with the chat view's effective
    settings, as the E2E's chat scenario sends). The first long job on Android asks "Keep
    translations running" first (BackgroundExecution): "Not now", as the device flow answers it."""
    from glossarion_mobile.ui.chat.run_request import attachment_record

    record = attachment_record(path)
    assert record is not None, path
    settings = app.chat_view.settings(str(cid))
    task = asyncio.ensure_future(app.chat_feature.runs.send(cid, text="", attachment=record, settings=settings,
                                                            output_mode="text"))
    deadline = time.monotonic() + 60
    while not task.done():
        if await driver.count(text="Keep translations running"):
            await driver.tap(text="Not now")
        if time.monotonic() > deadline:
            task.cancel()
            raise AssertionError(f"ChatRuns.send did not return for {path}")
        await asyncio.sleep(0.05)
    return task.result()


async def _finish(app, run, timeout: float = RUN_TIMEOUT) -> None:
    runs = app.chat_feature.runs
    assert await _until(lambda: not run.live, timeout), f"the chat run did not end (state {run.state})"
    await _join_finish_threads(runs)
    await asyncio.sleep(0.2)  # the posted announcements


def _job_cards(view) -> list:
    """The job cards on screen: the rendered turns and the live tail (a running job's card)."""
    shown = list(view.transcript.cards) + [getattr(c, "card", c) for c in view.transcript.tail]
    return [c for c in shown if type(c).__name__ == "JobCard"]


def _texts(tester) -> list:
    out = []
    for _kind, _key, tip, texts in tester.dump(5000):
        out.extend(texts)
        if tip:
            out.append(tip)
    return out


def _no_migrate_words(texts) -> list:
    return [t for t in texts if "migrate" in str(t).lower() or "organize" in str(t).lower()
            or str(t).lower().startswith("undo (")]


def _scan_root_for(root: Path, name: str) -> list:
    if not root or not Path(root).is_dir():
        return []
    return [str(p) for p in Path(root).rglob(name) if p.is_dir()]


def _run_scenario(scenario, server_kwargs: dict = None):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    with FakeLLMServer(**(server_kwargs or {})) as server:
        asyncio.run(scenario(server))


async def _start_app(files: dict):
    tf = _UF._foundations()
    app, tester, driver = await _UF._host_driver(tf, files)
    return tf, app, tester, driver


async def _stop_app(tf, app) -> None:
    try:
        app.jobs.close()
    finally:
        await tf._stop(app)


# ==========================================================================
# 1. The owner's flow: no tap after Start, the book is in the Library
# ==========================================================================


def test_chat_book_joins_the_library_without_a_tap(iso, migrate_calls):
    import flows

    epub = iso.picks / flows.EPUB_NAME
    shutil.copyfile(SELFTEST_EPUB, epub)
    stem = epub.stem

    async def scenario(server):
        tf, app, tester, driver = await _start_app({flows.EPUB_NAME: epub})
        try:
            await flows.wait_home(driver)
            _configure(app, server)
            migrate_calls.probe.service = app.job_service
            notes = _record_notes(app)
            out_root = Path(app.paths.output)
            assert _same(os.environ.get("OUTPUT_DIRECTORY"), out_root)
            assert _inside(out_root, iso.tmp) and _inside(os.environ["GLOSSARION_LIBRARY_DIR"], iso.tmp)

            # ---- the owner's taps: New chat, ＋ › Files, Send, Start; then only waiting -----------------
            await flows.chat_translate_and_migrate(driver, epub=flows.EPUB_NAME, timeout=RUN_TIMEOUT)
            taps = [s for s in driver.steps if s.startswith("tap ")]
            assert not [s for s in taps if "migrate" in s.lower() or "library" in s.lower()], taps
            assert not [n for n in notes if "migrate" in n[0].lower()], notes
            feature = app.chat_feature
            chats, runs = feature.chats, feature.runs
            cid = app.chat_view.cid
            await _join_finish_threads(runs)

            # ---- the workspace left the chat's Attachments for the output root -------------------
            target = out_root / stem
            chat_folder = Path(chats.output_folder(cid))
            assert _inside(chat_folder, out_root / "Direct Text"), chat_folder
            assert chats.attachment_folders(cid) == []
            assert not (chat_folder / "Attachments" / stem).exists()
            assert (target / "translation_progress.json").is_file(), sorted(p.name for p in out_root.iterdir())
            assert _progress(target) == {"total": CHAPTERS, "completed": CHAPTERS}
            assert list(target.glob("*.epub")), sorted(p.name for p in target.iterdir())
            assert not [p for p in out_root.iterdir() if p.name.startswith(f"{stem} (")]
            run = runs.run_for(cid)
            assert run is not None and run.state == "done" and _same(run.output_folder, target)
            folders = {m[4] for m in chats.messages(cid) if m[0] == "assistant" and len(m) > 4 and m[4]}
            assert any(_same(f, target) for f in folders), folders
            assert not any("Attachments" in Path(f).parts for f in folders), folders

            # ---- the desktop Migrate did it once, with the real output root and no job running ---------
            mine = [c for c in migrate_calls if Path(c["folder"]).name == stem]
            assert len(mine) == 1, migrate_calls
            call = mine[0]
            assert call["result"].get("ok") is True, call
            assert _same(call["output_directory"], out_root), call
            assert call["lock_free_or_ours"] and call["active_job"] is None, call
            assert not call["merge_callback"], call  # a fresh name: no merge question at all

            # ---- the raw stays in the Inbox and is registered for the Library -----------------------
            inbox_raw = [m[2] for m in chats.messages(cid) if m[0] == "user_file"][0]
            assert os.path.isfile(inbox_raw) and Path(inbox_raw).name == flows.EPUB_NAME
            registry = Path(os.environ["GLOSSARION_LIBRARY_DIR"]) / "library_raw_inputs.txt"
            assert registry.is_file() and any(_same(line.strip(), inbox_raw)
                                              for line in registry.read_text(encoding="utf-8").splitlines())

            # ---- the snackbar said so, with "Open book" -----------------------------------------
            added = [n for n in notes if n[0] == "Added to the Library"]
            assert added and added[-1][1] == "Open book", notes
            library = app.library
            await library.refresh(quiet=True, reason="test")
            snap = library.snapshot
            completed = [b for b in snap.completed if _same(b.get("output_folder"), target)]
            assert completed, [(b.get("name"), b.get("output_folder")) for b in snap.all_books()]
            assert not [b for b in snap.in_progress if _same(b.get("output_folder"), target)]
            bid = library.bid_for(completed[0])
            assert library.raw_source(completed[0]) and _same(library.raw_source(completed[0]), inbox_raw)

            # ---- the job card: "Open in Library" (enabled) and no Migrate ---------------------------
            view = app.chat_view
            view.render_transcript()
            assert await _until(lambda: any(c.action_reasons.get("library", "?") is None for c in _job_cards(view)), 15), \
                [c.action_reasons for c in _job_cards(view)]
            card = next(c for c in _job_cards(view) if c.action_reasons.get("library", "?") is None)
            assert "migrate" not in card.action_buttons and "library" in card.action_buttons
            assert not _no_migrate_words(_texts(tester)), _no_migrate_words(_texts(tester))
            found = await driver.find(text="Open in Library")
            assert found.count >= 1
            button = tester.control(found.first)
            assert not tester._disabled(button) and type(button).__name__ != "OutlinedButton"
            await driver.tap(text="Open in Library")
            book_route = f"/library/book/{bid}"
            assert await _until(lambda: app.shell.current_route == book_route, 15), app.shell.current_route
            assert getattr(app.shell.top_screen, "bid", None) == bid
            # the snackbar's "Open book" goes to the same Book page
            await driver.back()
            await _until(lambda: app.shell.current_route != book_route, 5)
            await tester._dispatch(added[-1][3], "action")  # the snackbar's "Open book" tap
            assert await _until(lambda: app.shell.current_route == book_route, 15), app.shell.current_route

            # ---- Library: the Completed shelf -> the book -> Chapters 12/12 -------------------------
            await flows.library_book_chapters(driver)
            # ---- Library ⋯: Scan for raw · Refresh · Library settings; no Organize / Undo ---------------
            app.navigate_to("library", reset=True)
            assert await _until(lambda: type(app.shell.top_screen).__name__ == "LibraryScreen", 15)
            screen = app.shell.top_screen
            assert await _until(lambda: getattr(screen, "menu_items", None), 10)
            assert set(screen.menu_items) == {"scan", "refresh", "settings"}, list(screen.menu_items)
            labels = [str(i.content) for i in screen.menu.items]
            assert not [t for t in labels if "organize" in t.lower() or "undo" in t.lower()], labels
            primary, more = screen.bulk_actions()
            ids = {a.id for a in list(primary) + list(more)}
            assert not {i for i in ids if "organize" in i or "undo" in i}, ids
            assert not _no_migrate_words(_texts(tester)), _no_migrate_words(_texts(tester))
            sheet_items = [i.label for i in screen.card_actions(completed[0]).items]
            assert not [t for t in sheet_items if "organize" in t.lower() or "migrate" in t.lower()], sheet_items

            # ---- Attachments manager: nothing waiting, and no Migrate anywhere ----------------------------
            app.navigate_to("chat.attachments", {"cid": cid})
            assert await _until(lambda: type(app.shell.top_screen).__name__ == "AttachmentsScreen", 15)
            attachments = app.shell.top_screen
            assert await _until(lambda: attachments.folders == [] and attachments.list.controls
                                and type(attachments.list.controls[0]).__name__ != "ProgressRing", 15)
            await driver.wait(key="attachments-empty", timeout=10)
            assert not _no_migrate_words(_texts(tester)), _no_migrate_words(_texts(tester))
            assert not hasattr(attachments, "migrate_button")
        finally:
            await _stop_app(tf, app)

    _run_scenario(scenario)


# ==========================================================================
# 2. The OUTPUT_DIRECTORY race: the next queued job must never receive the move
# ==========================================================================


def test_the_next_queued_job_never_receives_the_move(iso, migrate_calls):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.ui.chat.integration import MIGRATE_DEFERRED

    source_a = iso.picks / "Race A.epub"
    shutil.copyfile(SELFTEST_EPUB, source_a)
    source_b = fixtures.build_tiny_epub(iso.picks / "Race B.epub", chapters=3)

    async def scenario(server):
        tf, app, tester, driver = await _start_app({})
        try:
            await _until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            _configure(app, server)
            migrate_calls.probe.service = app.job_service
            feature = app.chat_feature
            chats, runs, service = feature.chats, feature.runs, app.job_service
            out_root = Path(app.paths.output)
            path_a, path_b = _import(app, source_a), _import(app, source_b)
            cid_a = chats.new_chat()
            state: dict = {}

            def park_b(record) -> None:
                """B's first model request parks (B is then the running job) once A's job has ended."""
                job_a = state.get("job_a")
                snap = service.snapshot(job_a) if job_a else None
                if snap is not None and snap.is_terminal and "held" not in state:
                    state["held"] = record.id
                    server.hold()

            server.on_request.append(park_b)
            run_a = await _send(app, driver, cid_a, path_a)
            state["job_a"] = run_a.job_id
            cid_b = chats.new_chat()  # after A's turn: the desktop reuses an empty chat
            assert cid_b != cid_a
            run_b = await _send(app, driver, cid_b, path_b)
            assert run_b.job_id != run_a.job_id
            if not await _until(lambda: not run_a.live, RUN_TIMEOUT):
                records = [(r.id, r.kind, r.status, r.parked, r.chapters, r.preview[:40]) for r in server.records()]
                raise AssertionError(f"A did not end: {run_a.state} {state} view={service.view()} "
                                     f"snapA={service.snapshot(run_a.job_id)} records={records}\n{_threads()}")
            await _join_finish_threads(runs)  # A's finish ran its auto-migrate attempt
            assert await _until(lambda: server.parked >= 1, 60), (state, server.parked)
            assert run_a.state == "done", run_a.state

            # B is the running job and owns the process environment right now
            view = service.view()
            assert view.active is not None and view.active.id == run_b.job_id, view.active
            live_output = os.environ.get("OUTPUT_DIRECTORY", "")
            state["live_output"] = live_output
            folders_a = chats.attachment_folders(cid_a)
            assert len(folders_a) == 1 and Path(folders_a[0]).name == "Race A", folders_a
            folder_a = folders_a[0]
            assert (Path(folder_a) / "translation_progress.json").is_file()
            assert not (out_root / "Race A").exists()
            assert not [c for c in migrate_calls if c["chat"] == str(cid_a)], migrate_calls
            # the race is real: the live OUTPUT_DIRECTORY is not the output root while B runs
            assert not _same(live_output, out_root), (live_output, out_root)
            assert not _scan_root_for(Path(live_output), "Race A"), live_output
            outcome = await asyncio.to_thread(feature.auto_migrate_blocking, cid_a, folder_a)
            assert outcome["status"] == MIGRATE_DEFERRED and outcome.get("jobs_running"), outcome
            assert Path(folder_a).is_dir() and not _scan_root_for(Path(live_output), "Race A")

            # B finishes: the queue is idle, both books move into the real output root
            server.release()
            assert await _until(lambda: not run_b.live, RUN_TIMEOUT), run_b.state
            await _join_finish_threads(runs)
            assert run_b.state == "done", run_b.state
            assert await _until(lambda: not chats.attachment_folders(cid_a) and not chats.attachment_folders(cid_b), 60), \
                (chats.attachment_folders(cid_a), chats.attachment_folders(cid_b))
            assert _progress(out_root / "Race A") == {"total": CHAPTERS, "completed": CHAPTERS}
            assert (out_root / "Race B" / "translation_progress.json").is_file()
            assert _same(runs.run_for(cid_a).output_folder, out_root / "Race A")
            for name, cid in (("Race A", cid_a), ("Race B", cid_b)):
                mine = [c for c in migrate_calls if c["chat"] == str(cid)]
                assert len(mine) == 1 and mine[0]["result"].get("ok"), (name, migrate_calls)
            for call in migrate_calls:
                assert _same(call["output_directory"], out_root), call
                assert call["lock_free_or_ours"] and call["active_job"] is None, call
            assert not _scan_root_for(Path(runs.temp_dir), "Race A"), runs.temp_dir
            assert _same(os.environ.get("OUTPUT_DIRECTORY"), out_root)
        finally:
            server.release(abort=True)
            await _stop_app(tf, app)

    _run_scenario(scenario)


# ==========================================================================
# 3. Name clashes: the same book merges, another book asks first
# ==========================================================================


def test_name_clash_same_book_merges_other_book_asks(iso, migrate_calls):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.ui.chat.attachments import MERGE_TITLE

    source = iso.picks / "Clash.epub"
    shutil.copyfile(SELFTEST_EPUB, source)
    stranger = fixtures.build_tiny_epub(iso.picks / "elsewhere" / "Clash.epub", chapters=3)

    async def scenario(server):
        tf, app, tester, driver = await _start_app({})
        try:
            await _until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            _configure(app, server)
            migrate_calls.probe.service = app.job_service
            notes = _record_notes(app)
            feature = app.chat_feature
            chats = feature.chats
            out_root = Path(app.paths.output)
            target = out_root / "Clash"
            raw = _import(app, source)

            # first chat: a fresh name, moved
            cid1 = chats.new_chat()
            await _finish(app, await _send(app, driver, cid1, raw))
            assert chats.attachment_folders(cid1) == [] and (target / "translation_progress.json").is_file()

            # the same book from another chat: the Library folder of that name is its own -> merged
            cid2 = chats.new_chat()
            assert cid2 != cid1
            again = _import(app, source)  # ＋ › Files picks the same book again
            assert _same(again, raw)  # FileBridge reuses the identical Inbox copy
            await _finish(app, await _send(app, driver, cid2, again))
            assert await _until(lambda: chats.attachment_folders(cid2) == [], 30), chats.attachment_folders(cid2)
            assert sorted(p.name for p in out_root.iterdir() if p.name.startswith("Clash")) == ["Clash"]
            assert _progress(target) == {"total": CHAPTERS, "completed": CHAPTERS}
            merged = [c for c in migrate_calls if c["chat"] == str(cid2)]
            assert len(merged) == 1 and merged[0]["merge_callback"] and merged[0]["result"].get("ok"), migrate_calls
            assert not [n for n in notes if n[0].startswith("A Library book named")], notes
            folders = {m[4] for m in chats.messages(cid2) if m[0] == "assistant" and len(m) > 4 and m[4]}
            assert any(_same(f, target) for f in folders), folders

            # a different book with the same file name: it stays in the chat until the user decides
            cid3 = chats.new_chat()
            assert cid3 not in (cid1, cid2)
            before = (target / "translation_progress.json").read_bytes()
            calls_before = len(migrate_calls)
            run3 = await _send(app, driver, cid3, str(stranger))
            await _finish(app, run3)
            assert run3.state == "done", run3.state
            folders3 = chats.attachment_folders(cid3)
            assert len(folders3) == 1 and Path(folders3[0]).name == "Clash", folders3
            assert (target / "translation_progress.json").read_bytes() == before
            assert len(migrate_calls) == calls_before, migrate_calls[calls_before:]
            clash = [n for n in notes if n[0] == "A Library book named Clash already exists"]
            assert clash and clash[-1][1] == "Merge…", [n[:2] for n in notes]
            # a later sweep (queue idle) leaves it and does not nag again
            await feature.sweep("test")
            assert chats.attachment_folders(cid3) == folders3
            assert len([n for n in notes if n[0].startswith("A Library book named")]) == 1
            # the snackbar action -> the desktop dialog -> Merge and replace
            await tester._dispatch(clash[-1][3], "action")
            await driver.wait(text=MERGE_TITLE, timeout=10)
            await driver.tap(text="Merge and replace")
            assert await _until(lambda: chats.attachment_folders(cid3) == [], 30), chats.attachment_folders(cid3)
            assert await _until(lambda: len([n for n in notes if n[0] == "Added to the Library"]) >= 3, 10), \
                [n[:2] for n in notes]
            assert (target / "source_epub.txt").is_file()
            assert _same((target / "source_epub.txt").read_text(encoding="utf-8").strip(), stranger)
            last = migrate_calls[-1]
            assert last["chat"] == str(cid3) and last["merge_callback"] and last["result"].get("ok"), last
            assert _same(last["output_directory"], out_root) and last["active_job"] is None, last
        finally:
            await _stop_app(tf, app)

    _run_scenario(scenario)


def test_a_different_book_at_the_same_inbox_path_is_not_merged_silently(iso, migrate_calls):
    """The same-book rule must look at the book, not only at the path ``source_epub.txt`` names: the
    Inbox copy of a translated book is deleted (Files › Inbox) and later a DIFFERENT book with the
    same file name is picked, so FileBridge puts it at the very same Inbox path. Its finished chat
    workspace must stay in the chat and ask (the name-clash snackbar), never replace the Library
    book's translation without a word."""
    from glossarion_mobile.diagnostics import fixtures

    source = iso.picks / "Clash.epub"
    shutil.copyfile(SELFTEST_EPUB, source)
    other = fixtures.build_tiny_epub(iso.picks / "other" / "Clash.epub", chapters=3)

    async def scenario(server):
        tf, app, tester, driver = await _start_app({})
        try:
            await _until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            _configure(app, server)
            migrate_calls.probe.service = app.job_service
            notes = _record_notes(app)
            chats = app.chat_feature.chats
            target = Path(app.paths.output) / "Clash"
            raw = _import(app, source)
            cid1 = chats.new_chat()
            await _finish(app, await _send(app, driver, cid1, raw))
            assert _progress(target) == {"total": CHAPTERS, "completed": CHAPTERS}
            before = (target / "translation_progress.json").read_bytes()

            os.remove(raw)  # Files › Inbox › Delete
            raw2 = _import(app, other)  # another book called Clash.epub
            assert _same(raw2, raw)  # FileBridge reuses the free name
            cid2 = chats.new_chat()
            assert cid2 != cid1
            run2 = await _send(app, driver, cid2, raw2)
            await _finish(app, run2)
            assert run2.state == "done", run2.state
            folders = chats.attachment_folders(cid2)
            assert len(folders) == 1 and Path(folders[0]).name == "Clash", \
                (f"the other book was merged into the Library book silently: attachments {folders}, "
                 f"Library progress now {_progress(target)}, snackbars {[n[:2] for n in notes]}")
            assert (target / "translation_progress.json").read_bytes() == before
            assert not [c for c in migrate_calls if c["chat"] == str(cid2)], migrate_calls
            assert [n for n in notes if n[0] == "A Library book named Clash already exists"], [n[:2] for n in notes]
        finally:
            await _stop_app(tf, app)

    _run_scenario(scenario)


# ==========================================================================
# 4. Scratch chats and stopped runs stay where they are
# ==========================================================================


def test_scratch_chat_and_stopped_run_stay_put(iso, migrate_calls):
    import flows
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.ui.chat.cards import LIBRARY_WAIT_REASON

    scratch_src = fixtures.build_tiny_epub(iso.picks / "Scratch Book.epub", chapters=3)
    epub = iso.picks / "Stopped Book.epub"
    shutil.copyfile(SELFTEST_EPUB, epub)

    async def scenario(server):
        tf, app, tester, driver = await _start_app({epub.name: epub})
        try:
            await flows.wait_home(driver)
            _configure(app, server)
            migrate_calls.probe.service = app.job_service
            notes = _record_notes(app)
            feature = app.chat_feature
            chats, runs = feature.chats, feature.runs
            out_root = Path(app.paths.output)

            # ---- a scratch chat: never saved implicitly, so its book stays in the scratch folder ------
            scid = chats.new_scratch()
            run = await _send(app, driver, scid, _import(app, scratch_src))
            await _finish(app, run)
            assert run.state == "done", run.state
            scratch_ws = Path(run.output_folder)
            assert scratch_ws.is_dir() and (scratch_ws / "translation_progress.json").is_file()
            assert _inside(scratch_ws, chats.scratch_dir), (scratch_ws, chats.scratch_dir)
            assert not (out_root / "Scratch Book").exists()
            outcomes = await feature.sweep("test")
            assert not [o for o in outcomes if o.get("cid") == scid], outcomes
            assert scratch_ws.is_dir() and not (out_root / "Scratch Book").exists()
            assert not migrate_calls, migrate_calls
            assert not [n for n in notes if n[0] == "Added to the Library"], notes

            # ---- a run stopped from the job card -------------------------------------------------------
            def park_after_first_chapter(record) -> None:
                """The model stops answering after the first chapter (the job keeps running until Stop)."""
                if record.kind == "translation" and len(record.chapters) == 1 and not server.holding:
                    server.hold()

            server.on_response.append(park_after_first_chapter)
            await driver.tap(tooltip="New chat")
            await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
            await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
            await driver.wait(contains=epub.stem, timeout=60)
            await driver.tap(key="send-idle_ready", timeout=60)
            await driver.wait(text="Ready to translate", timeout=60)
            await driver.tap(text="Start")
            cid = app.chat_view.cid
            assert await _until(lambda: server.parked >= 1, 120), "the run never reached the model"
            server.on_response.remove(park_after_first_chapter)
            view = app.chat_view
            await driver.tap(tooltip="Stop translation", timeout=15)  # the composer's Stop while it runs
            assert await _until(lambda: runs.run_for(cid) is not None and runs.run_for(cid).stop_requested, 10)
            server.release()  # the requests in flight finish; the graceful stop ends the run after them
            run = runs.run_for(cid)
            assert await _until(lambda: not run.live, RUN_TIMEOUT), run.state
            await _join_finish_threads(runs)
            snap = app.job_service.snapshot(run.job_id)
            assert run.state == "stopped", (run.state, snap.state, snap.stop_mode, snap.progress, snap.last_line)
            assert snap.state.value == "CANCELLED" and snap.stop_mode == "graceful", (snap.state, snap.stop_mode)
            folders = chats.attachment_folders(cid)
            assert len(folders) == 1 and Path(folders[0]).name == epub.stem, folders
            workspace = Path(folders[0])
            progress = _progress(workspace)
            assert 0 < progress["completed"] < CHAPTERS, progress
            assert not (out_root / epub.stem).exists()
            # idle sweeps leave it: its card still offers Resume
            outcomes = await feature.sweep("idle")
            mine = [o for o in outcomes if o.get("cid") == str(cid)]
            assert mine and all(o["status"] == "skipped" for o in mine), mine
            assert workspace.is_dir() and not (out_root / epub.stem).exists()
            assert not [c for c in migrate_calls if c["chat"] == str(cid)], migrate_calls
            view.render_transcript()
            assert await _until(lambda: any("resume" in c.action_buttons for c in _job_cards(view)), 15)
            card = next(c for c in _job_cards(view) if "resume" in c.action_buttons)
            assert await _until(lambda: card.action_reasons.get("library") == LIBRARY_WAIT_REASON, 15), card.action_reasons
            assert not _no_migrate_words(_texts(tester)), _no_migrate_words(_texts(tester))
            # the Attachments manager lists it, without Migrate
            app.navigate_to("chat.attachments", {"cid": cid})
            assert await _until(lambda: type(app.shell.top_screen).__name__ == "AttachmentsScreen", 15)
            attachments = app.shell.top_screen
            assert await _until(lambda: attachments.folders == [str(workspace)] or
                                [os.path.normcase(f) for f in attachments.folders] == [os.path.normcase(str(workspace))], 15)
            assert not _no_migrate_words(_texts(tester)), _no_migrate_words(_texts(tester))
            labels = [i.label for i in attachments.more_items(str(workspace))]
            assert not [t for t in labels if "migrate" in t.lower() or "merge" in t.lower()], labels
            await driver.back()
            assert await _until(lambda: type(app.shell.top_screen).__name__ != "AttachmentsScreen", 10)

            # ---- Resume finishes it, and only then it joins the Library by itself -------------------------
            view.render_transcript()
            card = next(c for c in _job_cards(view) if "resume" in c.action_buttons)
            await tester._dispatch(card.action_buttons["resume"], "click")
            assert await _until(lambda: runs.run_for(cid) is not None and runs.run_for(cid).live, 30)
            resumed = runs.run_for(cid)
            await _finish(app, resumed)
            assert resumed.state == "done", resumed.state
            assert await _until(lambda: chats.attachment_folders(cid) == [], 30), chats.attachment_folders(cid)
            assert _progress(out_root / epub.stem) == {"total": CHAPTERS, "completed": CHAPTERS}
            mine = [c for c in migrate_calls if c["chat"] == str(cid)]
            assert len(mine) == 1 and mine[0]["result"].get("ok") and _same(mine[0]["output_directory"], out_root), mine
            assert scratch_ws.is_dir() and not (out_root / "Scratch Book").exists()
        finally:
            await _stop_app(tf, app)

    _run_scenario(scenario)


# ==========================================================================
# 5. The owner's phone today: books the U8 APK left in Attachments
# ==========================================================================


def test_books_left_by_the_previous_version_join_on_the_next_launch(iso, migrate_calls, monkeypatch):
    """The chat books the owner already translated with the U8 APK sit in ``Attachments/<stem>``. The
    first app session here stands for the U8 app (no auto-migrate: the move is switched off); the
    next launch of the fixed app moves them into the Library by itself (the startup sweep), with no
    tap, and they are on the Completed shelf."""
    import flows
    from glossarion_mobile.ui.chat.integration import MIGRATE_SKIPPED, ChatFeature

    epub = iso.picks / "Earlier Book.epub"
    shutil.copyfile(SELFTEST_EPUB, epub)
    seen: dict = {}

    async def scenario(server):
        # ---- session 1: the U8 app (no auto-migrate) translates a book in the chat -----------------
        with monkeypatch.context() as patch:
            patch.setattr(ChatFeature, "auto_migrate_blocking",
                          lambda self, cid, folder: {"cid": str(cid), "folder": str(folder), "status": MIGRATE_SKIPPED,
                                                     "reason": "U8 had no auto-migrate", "target": "", "source": ""})
            tf, app, tester, driver = await _start_app({})
            try:
                await flows.wait_home(driver)
                _configure(app, server)
                chats = app.chat_feature.chats
                cid = chats.new_chat()
                run = await _send(app, driver, cid, _import(app, epub))
                await _finish(app, run)
                assert run.state == "done", run.state
                folders = chats.attachment_folders(cid)
                assert len(folders) == 1 and Path(folders[0]).name == epub.stem, folders
                chats.flush()
                seen.update(cid=cid, workspace=folders[0], out_root=Path(app.paths.output))
            finally:
                await _stop_app(tf, app)
        assert not migrate_calls, migrate_calls
        workspace, out_root = Path(seen["workspace"]), seen["out_root"]
        assert workspace.is_dir() and not (out_root / epub.stem).exists()

        # ---- session 2: the fixed app starts; nobody taps anything ------------------------------
        tf, app, tester, driver = await _start_app({})
        try:
            migrate_calls.probe.service = app.job_service
            await flows.wait_home(driver)
            chats = app.chat_feature.chats
            assert await _until(lambda: not workspace.exists(), 60), "the startup sweep did not move the book"
            assert await _until(lambda: chats.attachment_folders(seen["cid"]) == [], 15)
            target = out_root / epub.stem
            assert _progress(target) == {"total": CHAPTERS, "completed": CHAPTERS}
            mine = [c for c in migrate_calls if c["chat"] == str(seen["cid"])]
            assert len(mine) == 1 and mine[0]["result"].get("ok"), migrate_calls
            assert _same(mine[0]["output_directory"], out_root) and mine[0]["active_job"] is None, mine
            folders = {m[4] for m in chats.messages(seen["cid"]) if m[0] == "assistant" and len(m) > 4 and m[4]}
            assert any(_same(f, target) for f in folders) and not any("Attachments" in Path(f).parts for f in folders)
            await driver.wait(contains="Added to the Library", timeout=15)
            library = app.library
            await library.refresh(quiet=True, reason="test")
            assert [b for b in library.snapshot.completed if _same(b.get("output_folder"), target)], \
                [(b.get("name"), b.get("output_folder")) for b in library.snapshot.all_books()]
            # a later sweep has nothing left to do
            assert [o for o in await app.chat_feature.sweep("again") if o.get("status") == "moved"] == []
        finally:
            await _stop_app(tf, app)

    _run_scenario(scenario)
