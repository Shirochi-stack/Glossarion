"""Acceptance test for hotfix 3, owner issue 2 (Android phone, 2026-10-26): "Please add a Library button in my
jobs menu, so I can jump to the attachments library page quickly from it".

The owner's screenshots are the Job page (``/jobs/<jid>``) of a running chat EPUB translation (Job card with the
Running chip, "Open Chat · <book>", progress, Stop + Progress, Requests) with the floating job bar
"Translating · <book> · Ch 14/107 · 1 in flight" pinned at the bottom of the screen. This file replays the owner's
use end to end on the REAL app objects: the app on a fake Android Flet session (``test_ui_foundations._start``
through ``test_ui_flows._host_driver``, 412 dp), the real JobService on the app's UiDispatcher, ``HeadlessOwner``
and the shared translation pipeline, the real ChatFeature / ChatRuns / ChatView (with its automatic move of a
finished chat book into the Library), the real LibraryService (scans), the Library screens, Tools › QA Scanner,
the Jobs list, the floating job bar and the shell's navigation stack. Only the model is the offline fake OpenAI
server on 127.0.0.1 (``diagnostics.fake_llm_server``); the E2E network guard refuses every non-loopback
connection for the whole test. Every RUNNING step is held by a gate (the fake model's own reply hook stops
answering mid-book, ``test_devfix_issue12.HoldGate``; the QA scan waits at a gate in its job kind's run before
the real scan), never by sleeps.

What the owner sees is checked twice: in the Python control tree of the top view (what the tester walks and a tap
can reach) and in what the session last SENT the client (``ClientMirror``: the phone shows that, not the Python
tree).

1. ``test_chat_epub_job_library_button_running_and_finished`` (the owner's screenshot): New chat, ＋ › Files (an
   EPUB), Send, Start. Mid-book (the model holds) the chat's running Job card › Log opens the Job page: Stop,
   Progress and the floating job bar are there, and so is **Library**, enabled (hotfix 5): the run recorded the
   attached raw file, so a tap opens its Library page already and Back returns to the Job page.
   The model answers and the job ends Done while the chat's move into the Library is held at a gate: Library is
   enabled and a tap says "This book is not in the Library yet" (no navigation). The chat moves the book: a tap
   opens that book's Book page (the moved workspace, pushed over the Job page) and Back returns to the Job page.
   Drawer › Jobs › the job's row opens the Job page again: Library opens the same Book page, Back returns.
2. ``test_library_epub_job_library_button_running_and_finished``: Library › Import EPUB, the book's card, Book page ›
   Translate… › Start. Mid-book the floating job bar (on the Library shelf) opens the Job page: Library is enabled
   at once and opens that book's Book page; Back returns to the Job page. The same after the job ended Done, and
   from Drawer › Jobs › the job's row.
3. ``test_text_chat_and_tools_jobs_have_no_library_button``: a typed-text chat run and a Tools › QA Scanner run over
   a folder that is no Library book: the Job page shows no Library button (or reason), neither while the job runs
   nor after it ended, on screen or on the client. A Tools › QA Scanner run over a Library book never opens
   another book: either it has no button, or the button opens that book.
4. ``test_chat_save_to_library_job_library_button_running_and_finished``: a chat attachment sent with the Plan
   card's "Save to: Library" is a ``translate`` job whose workspace is in the output root, so the book is on the
   Library's In progress shelf while it runs: mid-book the chat card's Job opens the Job page and Library opens
   that book's Book page (the job's own workspace); Back returns; the same after the job ended Done.

Real data stays untouched: the app runs in temp FLET_APP_STORAGE_* dirs (HOME, OUTPUT_DIRECTORY,
GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR, CONFIG_FILE asserted inside tmp), USERPROFILE / APPDATA /
LOCALAPPDATA point into tmp, HTTP logging is off, and src/config.json must be byte-identical afterwards. A failing
scenario force-stops its jobs before the app closes.

Run from src/mobile with the mobile env (on the dev box ``.venv``, Python 3.12, keep the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests_host/test_hotfix3_issue2.py
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MOBILE_DIR.parent
for entry in (str(MOBILE_DIR / "app"), str(SRC_DIR), str(MOBILE_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load(alias: str, file_name: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


I12 = _load("_glossarion_devfix12_hotfix3_issue2", "test_devfix_issue12.py")  # the real-pipeline harness
pytestmark = I12.pytestmark
storage = I12.storage
app_env = I12.app_env
iso = I12.iso
offline = I12.offline

CHAPTERS = 12
RUN_TIMEOUT = 300.0
#: translation replies (the EPUB metadata request, then chapters) before the fake model stops answering: the job
#: is held mid-book, like the owner's "Ch 14/107 · 1 in flight"
HELD_AFTER = 3
WAIT_REASON = "Added to the Library when the translation finishes"
COMPOSER_HINT = "Message to translate…"
LIBRARY_BOOK = "library_novel.epub"
TYPED_TEXT = "제1장\n마왕성의 문이 열렸다. 기사는 검을 들었다."


@pytest.fixture(autouse=True)
def config_untouched():
    """src/config.json (the developer's real settings) is byte-identical after every test."""
    config = SRC_DIR / "config.json"

    def md5():
        try:
            return hashlib.md5(config.read_bytes()).hexdigest()
        except OSError:
            return None

    before = md5()
    yield
    assert md5() == before, "src/config.json changed during the test"


# ==========================================================================
# What the client was sent
# ==========================================================================


def _chain(control) -> list:
    out, node = [], control
    while node is not None:
        out.append(node)
        node = getattr(node, "parent", None)
    return out


class ClientMirror:
    """What the session last SENT the client about the Library buttons (the phone shows that). Flet sends a
    control's changed properties when it or one of its ancestors is patched (``Session.patch_control``), so after
    every patch each watched control under the patched one records ``visible`` / ``disabled`` of every node from it
    up to the patched control. A node never sent since the watch began counts with its current value; the positive
    checks also require ``sent_once``."""

    def __init__(self) -> None:
        self.watched = lambda: []
        self.sent: dict = {}
        self.errors: list = []

    @contextlib.contextmanager
    def watching(self):
        from flet.messaging import session as fs

        original = fs.Session.patch_control
        mirror = self

        def patch_control(session_self, control, *args, **kwargs):
            result = original(session_self, control, *args, **kwargs)
            try:
                mirror._after(control)
            except Exception as exc:  # the mirror must never break the app's patch
                mirror.errors.append(f"{type(exc).__name__}: {exc}")
            return result

        fs.Session.patch_control = patch_control
        try:
            yield self
        finally:
            fs.Session.patch_control = original

    def _after(self, patched) -> None:
        for control in self.watched() or ():
            chain = _chain(control)
            index = next((i for i, node in enumerate(chain) if node is patched), None)
            if index is None:
                continue
            if any(getattr(node, "is_isolated", lambda: False)() for node in chain[1:index]):
                continue  # Flet does not diff into an isolated control's children from a parent's patch
            for node in chain[:index + 1]:
                self.sent[id(node)] = (getattr(node, "visible", True) is not False,
                                       bool(getattr(node, "disabled", False)))

    def _state(self, node) -> tuple:
        return self.sent.get(id(node), (getattr(node, "visible", True) is not False,
                                        bool(getattr(node, "disabled", False))))

    def sent_once(self, control) -> bool:
        return id(control) in self.sent

    def shown(self, control) -> bool:
        return all(self._state(node)[0] for node in _chain(control))

    def disabled(self, control) -> bool:
        return any(self._state(node)[1] for node in _chain(control))


def _watched(app) -> list:
    """The Library button and its reason chip of every Job page on the shell's stack."""
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    out = []
    for entry in list(getattr(app.shell, "stack", None) or ()):
        screen = getattr(entry, "screen", None)
        if isinstance(screen, JobDetailScreen) and getattr(screen, "library_button", None) is not None:
            out.extend((screen.library_button, screen.library_reason))
    return out


# ==========================================================================
# Helpers
# ==========================================================================


TRACE = bool(os.environ.get("GLOSSARION_TEST_TRACE"))


def _trace(*parts) -> None:
    """Debug aid (GLOSSARION_TEST_TRACE=1 + ``-s``): to the real stdout (a running job captures ``sys.stdout``)."""
    if TRACE:
        print(time.strftime("%H:%M:%S"), "[issue2]", *parts, flush=True, file=sys.__stdout__)


def _on_screen(tester, control) -> bool:
    """``control`` is in the visible tree of the top view (what a tap can reach)."""
    return any(c is control for c in tester._walk())


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _inside(path, root) -> bool:
    return I12._inside(path, root)


def _library_ui(tester, mirror, detail) -> dict:
    """The Job page's Library button as the owner sees it: ``hidden`` / ``waiting`` (disabled, its reason shown) /
    ``disabled`` (no reason) / ``enabled``, in the walked tree (``tree``) and on the client (``client``)."""
    button, chip = getattr(detail, "library_button", None), getattr(detail, "library_reason", None)
    if button is None or chip is None:  # a Job page without a Library button
        return {"tree": "hidden", "client": "hidden", "sent": False, "label": "", "reason": "",
                "reason_on_screen": False, "reason_on_client": False, "found": None, "missing": True}
    if not _on_screen(tester, button):
        tree = "hidden"
    elif tester._disabled(button):
        tree = "waiting" if _on_screen(tester, chip) else "disabled"
    else:
        tree = "enabled"
    if not mirror.shown(button):
        client = "hidden"
    elif mirror.disabled(button):
        client = "waiting" if mirror.shown(chip) else "disabled"
    else:
        client = "enabled"
    return {"tree": tree, "client": client, "sent": mirror.sent_once(button), "label": str(button.content),
            "reason": str(getattr(chip, "reason", "") or ""), "reason_on_screen": _on_screen(tester, chip),
            "reason_on_client": mirror.shown(chip), "found": getattr(detail, "_library_found", None)}


def _lookup_done(detail) -> bool:
    """The Job page's Library lookup for its current job state has answered (or none runs)."""
    task = getattr(detail, "_library_task", None)
    return task is None or task.done()


def _expect(ui: dict, state: str) -> bool:
    if ui["tree"] != state or ui["client"] != state:
        return False
    if state == "hidden":
        return not ui["reason_on_screen"]
    if not ui["sent"] or ui["label"] != "Library":
        return False
    if state == "waiting":
        return ui["reason"] == WAIT_REASON and ui["reason_on_screen"] and ui["reason_on_client"]
    return not ui["reason_on_screen"] and not ui["reason_on_client"]


def _detail(app, jid):
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    screen = app.shell.top_screen
    if isinstance(screen, JobDetailScreen) and screen.job_id == jid and screen.snap is not None:
        return screen
    return None


def _book_page(app, bid=None):
    from glossarion_mobile.ui.library.book_page import BookPageScreen

    screen = app.shell.top_screen
    if isinstance(screen, BookPageScreen) and (bid is None or screen.bid == bid):
        return screen
    return None


def _stack(app) -> list:
    return [entry.match.name for entry in app.shell.stack]


async def _wait_detail(app, jid, timeout: float = 30.0):
    found = await I12._until(lambda: _detail(app, jid) is not None, timeout)
    assert found, f"the Job page of {jid} did not open: {_stack(app)} {app.shell.current_route}"
    return _detail(app, jid)


async def _wait_state(tester, mirror, detail, state: str, timeout: float = 30.0) -> dict:
    ok = await I12._until(lambda: _expect(_library_ui(tester, mirror, detail), state), timeout)
    ui = _library_ui(tester, mirror, detail)
    assert ok, f"the Job page's Library button is not {state!r}: {ui}"
    return ui


async def _tap_library(app, driver, detail, bid=None):
    """Library on the Job page: the Book page of ``bid`` (None: of whichever book; the caller checks it), pushed
    over the Job page, its Overview on screen."""
    stack_before = _stack(app)
    await driver.tap(key="job-library", timeout=15)
    ok = await I12._until(lambda: _book_page(app, bid) is not None, 30)
    assert ok, (f"Library did not open the Book page of {bid}: top={type(app.shell.top_screen).__name__} "
                f"{app.shell.current_route} {_stack(app)} found={getattr(detail, '_library_found', None)}")
    page = _book_page(app, bid)
    assert _stack(app) == stack_before + ["library.book"], (stack_before, _stack(app))
    assert app.shell.current_route.startswith(f"/library/book/{page.bid}"), app.shell.current_route
    await driver.wait(key="ov-title", timeout=30)  # the Book page's Overview: what the owner sees
    return page


async def _back_to_job(app, driver, jid, detail):
    """Android Back (the app-bar back arrow pops the same route): the Job page again, the same screen."""
    await driver.back()
    ok = await I12._until(lambda: _detail(app, jid) is not None, 30)
    assert ok, f"Back did not return to the Job page: {_stack(app)} {app.shell.current_route}"
    assert _detail(app, jid) is detail, "Back rebuilt the Job page instead of returning to it"
    assert app.shell.current_route == f"/jobs/{jid}", app.shell.current_route


def _card_button(card, label: str):
    from host_tester import _children

    stack = [card]
    while stack:
        control = stack.pop()
        if getattr(control, "content", None) == label and getattr(control, "on_click", None) is not None:
            return control
        stack.extend(_children(control))
    return None


async def _answer_dialogs(driver) -> None:
    """The first long job on Android asks "Keep translations running" ("Not now", as the device flow answers);
    an "Output Folder Mismatch" question is answered Yes; Tools › QA Scanner's "no source" question: continue
    without the word count."""
    if await driver.count(text="Keep translations running"):
        await driver.tap(text="Not now")
    if await driver.count(key="lib-output-root"):
        await driver.tap(text="Yes")
    if await driver.count(text="Continue without word count"):
        await driver.tap(text="Continue without word count")


async def _wait_terminal(app, jid, timeout: float = RUN_TIMEOUT):
    service = app.job_service

    def ended():
        snap = service.snapshot(jid)
        return snap if snap is not None and snap.is_terminal else None

    assert await I12._until(lambda: ended() is not None, timeout, 0.1), f"job {jid} did not end: {service.snapshot(jid)}"
    return ended()


def _new_job(service, before: set, kind: str):
    view = service.view()
    for snap in ((view.active,) if view.active is not None else ()) + tuple(view.queue) + tuple(view.history):
        if snap is not None and snap.kind == kind and snap.id not in before:
            return snap.id
    return None


def _known(service) -> set:
    view = service.view()
    return {s.id for s in ((view.active,) if view.active is not None else ()) + tuple(view.queue) + tuple(view.history)
            if s is not None}


async def _tap_nowait(tester, control):
    """A tap whose handler awaits a question (the dispatch returns once the handler did)."""
    assert _on_screen(tester, control), f"{type(control).__name__} is not on screen"
    task = asyncio.ensure_future(tester._dispatch(control, "click"))
    await asyncio.sleep(0)
    return task


async def _open_from_strip(app, tester, jid):
    """The floating job bar (``AppShell.global_strip``) on the current screen: a tap opens the running job."""
    strip = app.shell.global_strip
    ok = await I12._until(lambda: _on_screen(tester, strip) and bool(strip.title_text.value), 30)
    assert ok, f"the floating job bar is not on screen: visible={strip.visible} model={strip.model}"
    _trace("strip:", strip.title_text.value, "|", strip.subtitle_text.value)
    await tester._dispatch(strip, "click")
    return await _wait_detail(app, jid)


async def _open_from_jobs_list(app, driver, jid):
    """Drawer › Jobs › the job's row."""
    import flows

    await flows.open_drawer(driver)
    await driver.tap(key="dest-jobs", timeout=15)
    await driver.tap(key=f"job-row-{jid}", timeout=30)
    return await _wait_detail(app, jid)


def _configure(app, server) -> None:
    """The owner's phone: the fake OpenAI endpoint + a dummy key (``flows.ui_config``: no glossary, English, no
    request spacing)."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    store = app.config_store
    store.set_many(flows.ui_config(server.url, FAKE_MODEL))
    store.flush()
    assert store.save_error is None, store.save_error


class KindGate:
    """A job kind's ``run`` waits at a gate (the test releases it, or the job is stopped), then runs for real."""

    def __init__(self, monkeypatch, kind: str) -> None:
        from glossarion_mobile import job_kinds

        info = job_kinds.get_kind(kind)
        self.entered = threading.Event()
        self.gate = threading.Event()
        gate = self

        def run(ctx):
            gate.entered.set()
            while not gate.gate.wait(0.02):
                if ctx.stop_requested():
                    break
            return info.run(ctx)

        monkeypatch.setitem(job_kinds._CACHE, kind, dataclasses.replace(info, run=run))

    def release(self) -> None:
        self.gate.set()


async def _end_jobs(app, server, jids, gates=()) -> None:
    """Teardown: release the gates and force-stop a job a failing scenario left running."""
    for gate in gates:
        gate.release()
    server.release()
    service = getattr(app, "job_service", None)
    if service is None:
        return
    for jid in [j for j in jids if j]:
        snap = service.snapshot(jid)
        if snap is not None and not snap.is_terminal:
            server.release(abort=True)
            for _attempt in range(3):
                service.request_stop(jid, force=True, reason="test teardown")
                if await I12._until(lambda: service.snapshot(jid).is_terminal, 30):
                    break
    await asyncio.to_thread(service.wait_idle, 60)
    for thread in list(getattr(getattr(app, "chat_feature", None), "runs", types.SimpleNamespace(finish_threads=[]))
                       .finish_threads):
        await asyncio.to_thread(thread.join, 60)


async def _session(iso, server, files: dict, body) -> None:
    """Start the app on the fake Android session (412 dp) with the owner's settings, run ``body(ctx)``, then stop
    every job it left running and close the app."""
    import flows

    tf = I12._UF._foundations()
    app, tester, driver = await I12._UF._host_driver(tf, files)
    ctx = types.SimpleNamespace(app=app, tester=tester, driver=driver, server=server, session=tester.session,
                                page=tester.page, jobs=[], gates=[])
    try:
        await flows.wait_home(driver)
        I12._assert_isolated(iso.tmp)
        _configure(app, server)
        await body(ctx)
    finally:
        try:
            await _end_jobs(app, server, ctx.jobs, ctx.gates)
        finally:
            try:
                for name in ("config_store", "prefs"):
                    target = getattr(app, name, None)
                    if target is not None:
                        try:
                            target.flush()
                        except Exception:
                            pass
            finally:
                app.jobs.close()
                await tf._stop(app)


# ==========================================================================
# 1. The owner's screenshot: a chat EPUB translation
# ==========================================================================


def test_chat_epub_job_library_button_running_and_finished(iso, offline):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import JobState

    mirror = ClientMirror()

    async def body(ctx):
        app, tester, driver, server, run, probe = ctx.app, ctx.tester, ctx.driver, ctx.server, ctx.run, ctx.probe
        service, chats, cid = app.job_service, app.chat_feature.chats, probe.cid
        mirror.watched = lambda: _watched(app)
        jid = run.job_id
        raw = next(Path(m[2]) for m in chats.messages(cid) if m[0] == "user_file")

        # ---- mid-book: the model holds, the job is RUNNING and cannot end ---------------------------------------
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), I12._held_state(probe, server)
        snap = service.snapshot(jid)
        assert snap.state is JobState.RUNNING and snap.kind == "direct_text", (snap.state, snap.kind)
        assert (snap.spec.origin or {}).get("type") == "chat" and not (snap.spec.origin or {}).get("bid"), snap.spec.origin

        # the chat's running Job card › Log: the Job page
        card = app.chat_view.live_job_card
        assert card is not None, probe.describe()
        log_button = _card_button(card, "Log")
        assert log_button is not None, "the running Job card has no Log button"
        await I12._tap(probe, log_button)
        detail = await _wait_detail(app, jid)
        assert _stack(app)[-1] == "jobs.detail", _stack(app)
        assert await I12._until(lambda: detail.snap.state is JobState.RUNNING and _on_screen(tester, detail.stop_button)
                                and _on_screen(tester, detail.progress_button), 30), "no Stop / Progress on the Job page"
        strip = app.shell.global_strip
        assert await I12._until(lambda: _on_screen(tester, strip)
                                and str(strip.title_text.value or "").startswith("Translating"), 30), \
            f"the floating job bar is not on the Job page: {strip.visible} {strip.title_text.value!r}"
        _trace("job bar on the Job page:", strip.title_text.value, "|", strip.subtitle_text.value)
        # what the owner saw, plus the Library button: enabled while the book is translated in the chat too
        # (hotfix 5: "regardless of the attachment being compiled or not")
        ui = await _wait_state(tester, mirror, detail, "enabled")
        _trace("running chat book:", ui)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        ui = _library_ui(tester, mirror, detail)
        assert _expect(ui, "enabled"), f"the running chat book's lookup changed the button: {ui}"
        # a tap opens the book's Library page while it is still translated: the run recorded its raw file at the
        # start, so it is on the Library's shelf already (the owner: "it exists for in progress attachments")
        await driver.tap(key="job-library", timeout=10)
        assert await I12._until(lambda: _stack(app)[-1:] == ["library.book"], 30), _stack(app)
        await _back_to_job(app, driver, jid, detail)
        assert I12._settled(server) and service.snapshot(jid).state is JobState.RUNNING  # still held mid-book

        # ---- the model answers: Done; the chat's move into the Library is held at a gate -----------------------
        feature = app.chat_feature
        move_entered, move_gate = threading.Event(), threading.Event()
        move = feature.auto_migrate_blocking

        def held_move(*args, **kwargs):  # every move (the finished run's, the idle sweep's) goes through it
            move_entered.set()
            move_gate.wait(120)
            return move(*args, **kwargs)

        feature.auto_migrate_blocking = held_move
        ctx.release_move = move_gate.set
        server.release()
        done = await _wait_terminal(app, jid)
        assert done.state is JobState.DONE, (done.state, done.error)
        assert await I12._until(move_entered.is_set, 60), "the finished chat run never tried to move its book"
        assert chats.attachment_folders(cid), "the book left the chat's Attachments while the move was held"
        # Done, not moved yet: Library is enabled (no reason), a tap says why it cannot open the book yet
        await _wait_state(tester, mirror, detail, "enabled")
        assert await I12._until(lambda: _lookup_done(detail), 30)
        stack = _stack(app)
        await driver.tap(key="job-library", timeout=10)
        await driver.wait(text="This book is not in the Library yet", timeout=30)  # the snackbar
        assert await I12._until(lambda: _lookup_done(detail), 30)
        assert _detail(app, jid) is detail and _stack(app) == stack, (stack, _stack(app))

        # ---- the chat moves the book into the Library by itself -------------------------------------------------
        move_gate.set()
        await I12._finished(app, run)

        def moved():
            if chats.attachment_folders(cid):
                return None
            for message in chats.messages(cid):
                if message[0] == "assistant" and len(message) > 4 and message[4] \
                        and "Attachments" not in Path(message[4]).parts and Path(message[4]).is_dir():
                    return str(message[4])
            return None

        assert await I12._until(lambda: moved() is not None, 60), \
            f"the chat did not move the book into the Library: {chats.attachment_folders(cid)}"
        workspace = moved()
        assert _inside(workspace, iso.tmp), workspace
        _trace("moved to", workspace)

        # the same Job page: Library enabled now, and it opens the moved book's Book page
        assert _detail(app, jid) is detail, f"the Job page went away when the job ended: {_stack(app)}"
        ui = await _wait_state(tester, mirror, detail, "enabled")
        _trace("finished chat book:", ui)
        # (the done state's own lookup may have run before the move: then the tap looks again)
        book_page = await _tap_library(app, driver, detail)
        bid = book_page.bid
        assert _same(book_page.workspace, workspace), (book_page.workspace, workspace)
        rows = [row for row in app.library.snapshot.all_books()
                if _same(row.get("output_folder") or app.library.workspace_for(row) or "", workspace)]
        assert len(rows) == 1 and app.library.bid_for(rows[0]) == bid, (bid, [r.get("name") for r in rows])
        _trace("book row:", {k: rows[0].get(k) for k in ("name", "type", "raw_source_path", "output_folder")}, raw)
        assert detail._library_found == bid

        # ---- Back returns to the Job page; Library opens the same Book page again ---------------------------------
        await _back_to_job(app, driver, jid, detail)
        await _wait_state(tester, mirror, detail, "enabled")
        await _tap_library(app, driver, detail, bid)
        await _back_to_job(app, driver, jid, detail)

        # ---- Drawer › Jobs › the job's row: a new Job page finds the book by itself -----------------------------
        again = await _open_from_jobs_list(app, driver, jid)
        assert again is not detail and _stack(app)[-2:] == ["jobs", "jobs.detail"], _stack(app)
        await _wait_state(tester, mirror, again, "enabled")
        assert await I12._until(lambda: again._library_found == bid, 30), again._library_found
        await _tap_library(app, driver, again, bid)
        await _back_to_job(app, driver, jid, again)
        assert not mirror.errors, mirror.errors

    async def guarded(ctx):
        try:
            await body(ctx)
        finally:
            release = getattr(ctx, "release_move", None)
            if release is not None:
                release()

    with FakeLLMServer() as server, mirror.watching():
        I12.HoldGate(server, HELD_AFTER)  # the model's reply hook: it holds mid-book
        asyncio.run(I12._owner_flow(iso, server, guarded, chapters=CHAPTERS, glossary="off"))
    I12._assert_offline(offline)


# ==========================================================================
# 2. A Library EPUB translation
# ==========================================================================


async def _import_library_book(app, driver, iso) -> types.SimpleNamespace:
    """Library › Import EPUB (the FAB; the picker returns the file) › the book's card: its Book page."""
    import flows

    service = app.library
    await flows.open_drawer(driver)
    await driver.tap(key="dest-library")
    await driver.wait(key="lib-search", timeout=60)
    await driver.pick_file(LIBRARY_BOOK, lambda: driver.tap(key="lib-fab"))
    assert await I12._until(lambda: len(service.snapshot.all_books()) >= 1, 60), "the import added no book"
    rows = [dict(r) for r in service.snapshot.all_books()]
    assert len(rows) == 1, [r.get("name") for r in rows]
    row = rows[0]
    raw = str(row.get("raw_source_path") or "")
    assert raw and Path(raw).name == LIBRARY_BOOK and _inside(raw, iso.tmp), row
    bid = service.bid_for(row)
    await driver.tap(key=f"book-{bid}", timeout=60)
    assert await I12._until(lambda: _book_page(app, bid) is not None, 30), app.shell.current_route
    return types.SimpleNamespace(bid=bid, row=row, raw=raw, page=_book_page(app, bid))


async def _start_library_translation(app, driver, tester, book) -> str:
    """Book page › Overview › Translate… › the Translate sheet's Start: the book's ``translate`` job."""
    service = app.job_service
    before = _known(service)
    await driver.tap(key="ov-translate", timeout=60)
    found = await driver.wait(key="start", timeout=60)
    buttons = [tester.control(found.at(i)) for i in range(found.count)]
    buttons = [b for b in buttons if type(b).__name__ == "FilledButton"]
    assert len(buttons) == 1, f"{len(buttons)} Start buttons"
    assert await I12._until(lambda: not buttons[0].disabled, 30), "the Translate sheet's Start stays disabled"
    tap = await _tap_nowait(tester, buttons[0])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        await _answer_dialogs(driver)
        jid = _new_job(service, before, "translate")
        if jid:
            await asyncio.wait_for(tap, 30)
            return jid
        await asyncio.sleep(0.05)
    raise AssertionError(f"Start submitted no translate job (tap handler done: {tap.done()})")


def test_library_epub_job_library_button_running_and_finished(iso, offline):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import JobState

    mirror = ClientMirror()
    epub = fixtures.build_tiny_epub(iso.picks / LIBRARY_BOOK, chapters=CHAPTERS)

    async def body(ctx):
        app, tester, driver, server = ctx.app, ctx.tester, ctx.driver, ctx.server
        service = app.job_service
        mirror.watched = lambda: _watched(app)
        book = await _import_library_book(app, driver, iso)
        I12.HoldGate(server, HELD_AFTER)  # the model's reply hook: it holds mid-book
        jid = await _start_library_translation(app, driver, tester, book)
        ctx.jobs.append(jid)
        snap = service.snapshot(jid)
        assert (snap.spec.origin or {}).get("bid") == book.bid, snap.spec.origin

        # ---- mid-book: the floating job bar on the Library shelf opens the Job page ------------------------------
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), (server.holding, server.parked,
                                                                             service.snapshot(jid).state)
        assert service.snapshot(jid).state is JobState.RUNNING
        await driver.back()  # Book page -> Library shelf
        assert await I12._until(lambda: _stack(app)[-1:] == ["library"], 30), _stack(app)
        detail = await _open_from_strip(app, tester, jid)
        assert _stack(app)[-2:] == ["library", "jobs.detail"], _stack(app)
        ui = await _wait_state(tester, mirror, detail, "enabled", 10)
        _trace("running Library book:", ui)
        assert detail._library_found == book.bid and detail._library_task is None  # its origin: no lookup
        await _tap_library(app, driver, detail, book.bid)
        await _back_to_job(app, driver, jid, detail)
        assert I12._settled(server) and service.snapshot(jid).state is JobState.RUNNING  # still held mid-book

        # ---- the job ends Done: the same Job page, Library still opens the book -----------------------------------
        server.release()
        done = await _wait_terminal(app, jid)
        assert done.state is JobState.DONE, (done.state, done.error)
        assert _detail(app, jid) is detail, _stack(app)
        await _wait_state(tester, mirror, detail, "enabled")
        await _tap_library(app, driver, detail, book.bid)
        await _back_to_job(app, driver, jid, detail)

        # ---- Drawer › Jobs › the job's row ---------------------------------------------------------------------
        again = await _open_from_jobs_list(app, driver, jid)
        await _wait_state(tester, mirror, again, "enabled", 10)
        await _tap_library(app, driver, again, book.bid)
        await _back_to_job(app, driver, jid, again)
        assert not mirror.errors, mirror.errors

    with FakeLLMServer() as server, mirror.watching():
        asyncio.run(_session(iso, server, {LIBRARY_BOOK: epub}, body))
    I12._assert_offline(offline)


# ==========================================================================
# 3. Jobs without a Library book: a typed-text chat, a Tools job
# ==========================================================================


def _scan_folder(root: Path) -> Path:
    """A translated output folder that is no Library book (outside the output root and the Library)."""
    folder = root / "elsewhere" / "Scan"
    folder.mkdir(parents=True)
    for index in (1, 2):
        (folder / f"response_{index:04d}_chapter{index:04d}.html").write_text(
            f"<html><head><title>Chapter {index}</title></head><body><h1>Chapter {index}</h1>"
            f"<p>The knight raised the sword at dawn, chapter {index}.</p></body></html>", encoding="utf-8")
    return folder


async def _start_qa(app, driver, tester, targets) -> str:
    """Tools › QA Scanner with ``targets`` chosen, then its Start (questions answered): the ``qa_scan`` job."""
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

    service = app.job_service
    before = _known(service)
    await app.navigate("/tools/qa")
    assert await I12._until(lambda: isinstance(app.shell.top_screen, QaScannerScreen), 30), app.shell.current_route
    screen = app.shell.top_screen
    screen.set_targets(targets)
    assert screen.targets, "the QA Scanner refused the target"
    start = await driver.wait(key="qa-start", timeout=30)
    tap = await _tap_nowait(tester, tester.control(start.first))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        await _answer_dialogs(driver)
        jid = _new_job(service, before, "qa_scan")
        if jid:
            await asyncio.wait_for(tap, 30)
            return jid
        await asyncio.sleep(0.05)
    raise AssertionError(f"QA Start submitted no qa_scan job (tap handler done: {tap.done()})")


def test_text_chat_and_tools_jobs_have_no_library_button(iso, offline, monkeypatch):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.tools import targets as tg

    mirror = ClientMirror()
    epub = fixtures.build_tiny_epub(iso.picks / LIBRARY_BOOK, chapters=3)
    folder = _scan_folder(iso.tmp)

    async def body(ctx):
        app, tester, driver, server = ctx.app, ctx.tester, ctx.driver, ctx.server
        service = app.job_service
        mirror.watched = lambda: _watched(app)

        # ---- a typed-text chat run, held at the model ----------------------------------------------------------
        server.hold()
        before = _known(service)
        await driver.tap(tooltip="New chat")
        await driver.enter(TYPED_TEXT, text=COMPOSER_HINT)
        await driver.tap(key="send-idle_ready", timeout=30)
        jid = None
        deadline = time.monotonic() + 60
        while jid is None and time.monotonic() < deadline:
            await _answer_dialogs(driver)
            jid = _new_job(service, before, "direct_text")
            await asyncio.sleep(0.05)
        assert jid, "Send submitted no chat job"
        ctx.jobs.append(jid)
        assert await I12._until(lambda: server.parked >= 1 and service.snapshot(jid).state is JobState.RUNNING, 60), \
            (server.parked, service.snapshot(jid).state)
        detail = await _open_from_jobs_list(app, driver, jid)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        ui = _library_ui(tester, mirror, detail)
        assert _expect(ui, "hidden"), f"a typed-text chat's running job shows Library: {ui}"
        assert await driver.count(key="job-library") == 0 and await driver.count(key="job-library-reason") == 0
        server.release()
        done = await _wait_terminal(app, jid)
        assert done.state is JobState.DONE, (done.state, done.error)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        await asyncio.sleep(0)  # the done state's repaint
        ui = _library_ui(tester, mirror, detail)
        assert _expect(ui, "hidden"), f"a typed-text chat's finished job shows Library: {ui}"
        assert await driver.count(key="job-library") == 0

        # ---- Tools › QA Scanner over a folder that is no Library book, held at its run -------------------------
        gate = KindGate(monkeypatch, "qa_scan")
        ctx.gates.append(gate)
        target = tg.ToolTarget(title=folder.name, folder=str(folder), origin="browse", kind="epub")
        qa = await _start_qa(app, driver, tester, [target])
        ctx.jobs.append(qa)
        assert await I12._until(lambda: gate.entered.is_set() and service.snapshot(qa).state is JobState.RUNNING, 60)
        detail = await _open_from_strip(app, tester, qa)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        ui = _library_ui(tester, mirror, detail)
        assert _expect(ui, "hidden"), f"a Tools job without a Library book shows Library while it runs: {ui}"
        gate.release()
        await _wait_terminal(app, qa)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        ui = _library_ui(tester, mirror, detail)
        assert _expect(ui, "hidden"), f"a Tools job without a Library book shows Library after it ended: {ui}"

        # ---- Tools › QA Scanner over a Library book: never another book ----------------------------------------
        book = await _import_library_book(app, driver, iso)
        library_target = await asyncio.to_thread(tg.target_for_book, app.library, book.row)
        assert library_target is not None and library_target.folder, library_target
        qa_book = await _start_qa(app, driver, tester, [library_target])
        ctx.jobs.append(qa_book)
        snap = await _wait_terminal(app, qa_book)
        assert not (snap.spec.origin or {}).get("bid"), snap.spec.origin  # a Tools job: no Library origin
        await app.navigate(f"/jobs/{qa_book}")
        detail = await _wait_detail(app, qa_book)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        ui = _library_ui(tester, mirror, detail)
        _trace("QA over a Library book:", ui)
        assert ui["tree"] == ui["client"] and ui["tree"] in ("hidden", "enabled"), ui
        if ui["tree"] == "enabled":
            await _tap_library(app, driver, detail, book.bid)
            await _back_to_job(app, driver, qa_book, detail)
        assert not mirror.errors, mirror.errors

    with FakeLLMServer() as server, mirror.watching():
        asyncio.run(_session(iso, server, {LIBRARY_BOOK: epub}, body))
    I12._assert_offline(offline)


# ==========================================================================
# 4. A chat attachment sent with "Save to: Library"
# ==========================================================================


CHAT_LIBRARY_BOOK = "chat_library_novel.epub"


def test_chat_save_to_library_job_library_button_running_and_finished(iso, offline):
    """New chat, ＋ › Files (an EPUB), Send, the Plan card's "Save to: Library", Start: a ``translate`` job whose
    workspace is in the output root (the book is on the Library's In progress shelf while it runs). Mid-book the
    chat's "Sent to the Library" card › Job opens the Job page: Library opens the book's Book page (the job's own
    workspace) and Back returns; the same after the job ended Done."""
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import JobState

    mirror = ClientMirror()
    epub = fixtures.build_tiny_epub(iso.picks / CHAT_LIBRARY_BOOK, chapters=CHAPTERS)

    async def body(ctx):
        import flows

        app, tester, driver, server = ctx.app, ctx.tester, ctx.driver, ctx.server
        service = app.job_service
        mirror.watched = lambda: _watched(app)
        I12.HoldGate(server, HELD_AFTER)  # the model's reply hook: it holds mid-book
        before = _known(service)
        await flows.go_home(driver)
        await driver.tap(tooltip="New chat")
        await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
        await driver.pick_file(CHAT_LIBRARY_BOOK, lambda: driver.tap(key="attach-files"))
        await driver.wait(contains=Path(CHAT_LIBRARY_BOOK).stem, timeout=60)  # the composer pill
        await driver.tap(key="send-idle_ready", timeout=60)
        await driver.wait(text="Ready to translate", timeout=60)  # the Plan card
        await driver.tap(key="plan-destination", timeout=30)
        await driver.tap(key="dest-library", timeout=30)
        await driver.wait(text="Save to: Library", timeout=30)
        await driver.tap(text="Start")
        jid = None
        deadline = time.monotonic() + 60
        while jid is None and time.monotonic() < deadline:
            await _answer_dialogs(driver)
            jid = _new_job(service, before, "translate")
            await asyncio.sleep(0.05)
        assert jid, "Start submitted no Library translate job"
        ctx.jobs.append(jid)
        snap = service.snapshot(jid)
        assert (snap.spec.origin or {}).get("type") == "chat", snap.spec.origin
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), (server.holding, server.parked)
        assert service.snapshot(jid).state is JobState.RUNNING
        workspaces = [str(d) for d in (service.snapshot(jid).output_dirs or {}).values()] + \
            [str(service.snapshot(jid).output_dir or "")]
        _trace("save-to-library job dirs:", workspaces)

        # the chat's "Sent to the Library" card › Job: the Job page
        await driver.tap(key="libjob-job", timeout=30)
        detail = await _wait_detail(app, jid)
        assert await I12._until(lambda: _lookup_done(detail), 30)
        ui = _library_ui(tester, mirror, detail)
        _trace("running save-to-library chat book:", ui)
        ui = await _wait_state(tester, mirror, detail, "enabled", 30)
        page = await _tap_library(app, driver, detail)
        assert any(w and _same(page.workspace, w) for w in workspaces), (page.workspace, workspaces)
        bid = page.bid
        await _back_to_job(app, driver, jid, detail)
        assert I12._settled(server) and service.snapshot(jid).state is JobState.RUNNING  # still held mid-book

        server.release()
        done = await _wait_terminal(app, jid)
        assert done.state is JobState.DONE, (done.state, done.error)
        await _wait_state(tester, mirror, detail, "enabled")
        await _tap_library(app, driver, detail, bid)
        await _back_to_job(app, driver, jid, detail)
        assert not mirror.errors, mirror.errors

    with FakeLLMServer() as server, mirror.watching():
        asyncio.run(_session(iso, server, {CHAT_LIBRARY_BOOK: epub}, body))
    I12._assert_offline(offline)


def test_sent_to_library_card_progress_opens_the_chapters_whatever_exists(iso, offline):
    """The chat's "Sent to the Library" card › Progress (owner 2026-10-31): the book's Chapters in the Progress
    manager, also with no translation progress yet (the job parked in its glossary stage, whose Glossary progress
    the Job card's Progress would open instead), and again after the job ended Done."""
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import JobState

    epub = fixtures.build_tiny_epub(iso.picks / CHAT_LIBRARY_BOOK, chapters=CHAPTERS)

    def book_page(app):
        top = app.shell.top_screen
        return top if type(top).__name__ == "BookPageScreen" else None

    async def open_progress(app, driver):
        await driver.tap(key="libjob-progress", timeout=30)
        assert await I12._until(lambda: book_page(app) is not None, 30), app.shell.current_route
        page = book_page(app)
        assert app.shell.current_route.startswith("/tools/progress"), app.shell.current_route
        assert page.title == "Progress manager" and page.current_tab == "chapters"
        assert await I12._until(lambda: list(page.chapters.visible), 60), "the Chapters tab listed nothing"
        return page

    async def body(ctx):
        import flows

        app, driver, server = ctx.app, ctx.driver, ctx.server
        service = app.job_service
        app.config_store.set_many({"auto_glossary_mode": "full", "extraction_workers": 1})
        server.hold()  # the job's first request parks: its glossary stage, before any translation progress
        before = _known(service)
        await flows.go_home(driver)
        await driver.tap(tooltip="New chat")
        await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
        await driver.pick_file(CHAT_LIBRARY_BOOK, lambda: driver.tap(key="attach-files"))
        await driver.wait(contains=Path(CHAT_LIBRARY_BOOK).stem, timeout=60)
        await driver.tap(key="send-idle_ready", timeout=60)
        await driver.wait(text="Ready to translate", timeout=60)
        await driver.tap(key="plan-destination", timeout=30)
        await driver.tap(key="dest-library", timeout=30)
        await driver.wait(text="Save to: Library", timeout=30)
        await driver.tap(text="Start")
        jid = None
        deadline = time.monotonic() + 60
        while jid is None and time.monotonic() < deadline:
            await _answer_dialogs(driver)
            jid = _new_job(service, before, "translate")
            await asyncio.sleep(0.05)
        assert jid, "Start submitted no Library translate job"
        ctx.jobs.append(jid)
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), (server.holding, server.parked)
        snap = service.snapshot(jid)
        assert snap.state is JobState.RUNNING and snap.phase == "Indexing", (snap.state, snap.phase)
        assert not any(os.path.isfile(os.path.join(str(d), "translation_progress.json"))
                       and os.path.getsize(os.path.join(str(d), "translation_progress.json")) > 200
                       for d in (snap.output_dirs or {}).values()), "the job already has translation progress"

        page = await open_progress(app, driver)
        assert page.job is not None and page.job.id == jid  # the Book page follows the running job
        await driver.back()
        assert await I12._until(lambda: book_page(app) is None, 30), app.shell.current_route
        assert app.shell.top_screen is None, type(app.shell.top_screen).__name__  # one Back: the chat, not Tools

        server.release()
        done = await _wait_terminal(app, jid)
        assert done.state is JobState.DONE, (done.state, done.error)
        page = await open_progress(app, driver)
        chapters = [row for row in page.chapters.visible if row.kind == "chapter"]
        assert chapters and all(row.status == "completed" for row in chapters), [r.status for r in chapters]

    with FakeLLMServer() as server:
        asyncio.run(_session(iso, server, {CHAT_LIBRARY_BOOK: epub}, body))
    I12._assert_offline(offline)
