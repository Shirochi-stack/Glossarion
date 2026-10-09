"""U10 review fixes: regression tests for the findings the acceptance tests do not cover by themselves.

* a recompile / a format turned on while the book is being copied is copied afterwards (queue ``gen``);
* Tools › Headers & metadata (``output_roots`` maps raw source -> output root) never uploads the raw EPUB, and
  nothing in Library/Raw is ever a book to copy;
* changing the destination while a write runs: the new destination gets the book without a resume, and no
  record is left under the retired destination;
* Stop / a foreground-service timeout while the private snapshot is made: nothing is written afterwards;
* one "Couldn't save" notification per drain that counts the books; the save-locations destination's label;
* save locations mode gives a persisted grant back when its record stops pointing at it;
* a Result card re-binds only when its state changed (progress repaints only the card of the book copied);
* chat books the startup sweep moved before the cloud sync existed / while it loaded its records are queued;
* the Library delete asks before it drops the deletable share-link uploads of the books it deletes;
* Cancel after the whole file was sent is "maybe uploaded", never "cancelled";
* the Output tab lists the PDF an EPUB book's Compile PDF writes ("<Title>.pdf");
* the share sheets get per-build dialog keys (a sheet shown again before the previous one's dismiss arrived
  stays usable).

Everything lives under ``tmp_path``; nothing reaches the network (local fakes only).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_u10_fixes.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _load(name: str, alias: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # its dataclasses resolve their module while the file runs
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(not _has("flet"), reason="flet not installed")

if _has("flet"):
    TCS = _load("test_cloud_sync.py", "_glossarion_tcs_helpers_u10fixes")
    _isolate = TCS._isolate  # autouse: Library / output / home / data under tmp_path
    _no_network = TCS._no_network  # autouse: only loopback connects
    Env = TCS.Env
    EPUB1, EPUB2, PDF1 = TCS.EPUB1, TCS.EPUB2, TCS.PDF1

    from glossarion_mobile.services import cloud_sync as cs
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState
    from glossarion_mobile.state.cloud_records import book_key


def run(coro, timeout: float = 120.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))  # a hang fails the test instead of the whole run


async def until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


def _writes_started(env, before: int) -> bool:
    return any(c[0] == "write_file" for c in env.cloud.calls[before:])


# ---------------------------------------------------------------------------------------------------
# a trigger that arrives while the book is copied is never lost (queue generations)
# ---------------------------------------------------------------------------------------------------


def test_a_recompile_during_the_copy_is_copied_after_it(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookG", {"BookG.epub": EPUB1})
        await env.link()
        assert env.cloud.files() == {"BookG/BookG.epub": EPUB1}
        env.cloud.gate = asyncio.Event()
        env.recompile(folder, "BookG.epub", EPUB2)
        mark = len(env.cloud.calls)
        env.done(folder, jid="g1")
        assert await until(lambda: _writes_started(env, mark))
        v3 = b"PK-epub-version-3-written-while-v2-was-copied" * 30
        env.recompile(folder, "BookG.epub", v3, later=20)
        env.done(folder, jid="g2")  # the recompile finishes while v2 is being copied
        gate, env.cloud.gate = env.cloud.gate, None
        gate.set()
        await env.settle()
        assert env.cloud.files() == {"BookG/BookG.epub": v3}, "the newer recompile was dropped with the old entry"
        assert env.store.queue() == []
        assert env.service.file_entries(folder)["epub"]["status"] == "ok"

    run(scenario())


def test_a_format_turned_on_during_the_copy_is_copied(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookF", {"BookF.epub": EPUB1, "BookF.pdf": PDF1})
        env.service.set_kind_enabled("pdf", False)
        await env.link()
        assert env.cloud.files() == {"BookF/BookF.epub": EPUB1}
        env.cloud.gate = asyncio.Event()
        env.recompile(folder, "BookF.epub", EPUB2)
        mark = len(env.cloud.calls)
        env.done(folder, jid="f1")
        assert await until(lambda: _writes_started(env, mark))
        env.service.set_kind_enabled("pdf", True)  # the PDF switch goes on while the EPUB is written
        await asyncio.sleep(0.05)
        gate, env.cloud.gate = env.cloud.gate, None
        gate.set()
        await env.settle()
        assert env.cloud.files() == {"BookF/BookF.epub": EPUB2, "BookF/BookF.pdf": PDF1}
        assert env.store.queue() == []

    run(scenario())


def test_queue_generations_keep_a_newer_trigger(tmp_path):
    from glossarion_mobile.state.cloud_records import CloudRecordStore

    store = CloudRecordStore(tmp_path / "c.json", save_delay=0.0)
    taken = store.enqueue(tmp_path / "Book", "job:1")
    store.enqueue(tmp_path / "Book", "job:2")  # a new trigger while the first is processed
    assert not store.finish(taken["key"], taken["gen"]) and store.queued(tmp_path / "Book") is not None
    assert store.reschedule(taken["key"], 1e12, attempts=1, gen=taken["gen"]) is None
    entry = store.queued(tmp_path / "Book")
    assert entry["attempts"] == 0 and entry["next_at"] < 1e12  # still due with a fresh start
    assert store.finish(entry["key"], entry["gen"]) and store.queue() == []
    store.close()


# ---------------------------------------------------------------------------------------------------
# untranslated sources never leave
# ---------------------------------------------------------------------------------------------------


def test_headers_tool_job_never_uploads_the_raw_source(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("Novel", {"Novel.epub": EPUB1})
        raw_dir = env.library_dir / "Raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_dir / "Novel.epub"
        raw.write_bytes(b"PK untranslated raw source " * 20)
        await env.link()
        bid = env.library.bid_for(env.books[0])
        # Tools › Headers & metadata: params.output_roots maps {raw source: output root} (headers_model)
        spec = JobSpec(kind="metadata", title="Headers & metadata", inputs=(str(raw),),
                       params={"output_roots": {str(raw): str(env.output)}},
                       origin={"type": "library", "bid": bid})
        assert [b[0] for b in env.service._job_books(SimpleSnap(spec))] == [folder]
        env.jobs.deliver(JobSnapshot(id="meta1", spec=spec, state=JobState.DONE, created=0.0), JobState.RUNNING)
        await env.settle()
        assert all(b"untranslated" not in data for data in env.cloud.files().values()), env.cloud.files()
        assert sorted(env.cloud.files()) == ["Novel/Novel.epub"]
        # nothing in Library/Raw is a book to copy (Send now refuses it, the state says so)
        assert env.service._library_state_blocking(str(raw)) == "outside"
        answer = await env.service.send_now(str(raw))
        assert not answer["ok"]

    run(scenario())


class SimpleSnap:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.outputs = ()
        self.output_dir = None
        self.output_dirs = {}


# ---------------------------------------------------------------------------------------------------
# a destination changed while a write runs
# ---------------------------------------------------------------------------------------------------


def test_changing_the_destination_mid_write_continues_on_the_new_one(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookR", {"BookR.epub": EPUB1, "BookR.pdf": PDF1})
        await env.link()
        old_id = env.service.destination().id
        other = env.cloud.add(None, "Other", "folder")
        target = env.cloud.ref(other)
        target.update(uri=f"content://com.fake.docs/tree/{other}", id=f"other-{other}")
        env.cloud.gate = asyncio.Event()
        env.recompile(folder, "BookR.epub", EPUB2)
        mark = len(env.cloud.calls)
        env.done(folder, jid="r1")
        assert await until(lambda: _writes_started(env, mark))
        env.cloud.pick_queue.append({"ok": True, "error": None, "target": target, "persisted": True})
        answer = await env.service.pick_folder()  # its cancel releases the held write (the fake sets the gate)
        assert answer["ok"], answer
        env.cloud.gate = None
        await env.settle()
        await asyncio.sleep(0.05)
        await env.settle()
        book_folder = next(n for n, node in env.cloud.nodes.items()
                           if node["kind"] == "folder" and node["parent"] == other and node["name"] == "BookR")
        assert env.cloud.files(book_folder) == {"BookR.epub": EPUB2, "BookR.pdf": PDF1}, \
            "the new destination waited for a resume"
        assert env.store.books(old_id) == {}, "a record came back under the retired destination"

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# Stop / timeout while the private snapshot is made
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("event", [{"type": "timeout"}, {"type": "destroyed"}])
def test_a_foreground_timeout_during_the_snapshot_writes_nothing(tmp_path, event):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookS", {"BookS.epub": EPUB1})
        await env.link(enable=False)
        started = threading.Event()
        real = env.service._snapshot_blocking

        def slow_snapshot(source):
            started.set()
            time.sleep(0.4)
            return real(source)

        env.service._snapshot_blocking = slow_snapshot
        writes = len(env.cloud.writes)
        await env.service.send_now(folder)
        assert await until(started.is_set)
        assert env.service.keepalive.held
        await env.service.on_foreground_event(event)
        await env.settle()
        assert len(env.cloud.writes) == writes, "the drain wrote after the foreground service ended"
        assert env.store.queued(folder) is not None  # the book waits for the next resume / job
        env.service._snapshot_blocking = real
        await env.service.drain_now("resume")
        assert env.cloud.files() == {"BookS/BookS.epub": EPUB1}

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# failure notifications, labels
# ---------------------------------------------------------------------------------------------------


def test_several_failing_books_post_one_notification_that_counts_them(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folders = [env.book(f"Book{i}", {f"Book{i}.epub": EPUB1}) for i in range(3)]
        await env.link()
        env.cloud.offline = True
        for i, folder in enumerate(folders):
            env.recompile(folder, f"Book{i}.epub", EPUB2)
            env.done(folder, jid=f"n{i}")
        await env.settle()
        for _ in range(cs.FAIL_AFTER + 1):
            env.clock.now += cs.BACKOFF_MAX + 10
            await env.service.drain_now("retry")
        notes = [n for n in env.bridge.notifications if n["title"].startswith("Couldn't save")]
        assert len(notes) == 1, notes
        assert notes[0]["body"] == "3 books waiting · tap for details"
        assert env.service.ui_state()["failed"] == 3

    run(scenario())


def test_save_locations_destination_label():
    dest = cs.Destination(id=cs.FILES_TARGET_ID, mode=cs.MODE_FILES)
    assert dest.display == "your save locations"
    assert "chosen files" not in cs._progress_text("Book.epub", dest.display)


# ---------------------------------------------------------------------------------------------------
# save locations mode gives grants back
# ---------------------------------------------------------------------------------------------------


def test_save_locations_release_the_grant_a_record_stops_using(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        env.cloud.modes = {"w"}  # this cloud app only takes 'w' (a shorter file cannot be written in place)
        folder = env.book("BookP", {"BookP.epub": EPUB2})
        await env.service.use_save_locations()
        env.service.set_enabled(True)
        await env.settle()
        assert (await env.service.choose_save_location(folder, "epub"))["ok"]
        doc = env.store.record(env.service.destination().id, book_key(folder), "epub")["doc"]
        assert doc["document"] in env.cloud.grants
        env.recompile(folder, "BookP.epub", EPUB1)  # shorter: needs a new save location
        env.done(folder, jid="p1")
        await env.settle()
        entry = env.store.record(env.service.destination().id, book_key(folder), "epub")
        assert entry["status"] == "needs_pick" and not entry.get("doc")
        assert doc["document"] in env.cloud.releases and doc["document"] not in env.cloud.grants
        # "Choose cloud file…" over a record that has a document gives the old one back too
        assert (await env.service.choose_save_location(folder, "epub"))["ok"]
        second = env.store.record(env.service.destination().id, book_key(folder), "epub")["doc"]
        other = env.cloud.add("picked", "Elsewhere.epub", data=b"old")
        env.cloud.grants.add(f"content://com.fake.docs/document/{other}")
        env.cloud.pick_queue.append({"ok": True, "document": env.cloud.ref(other)})
        assert (await env.service.choose_existing_file(folder, "epub"))["ok"]
        await env.settle()
        assert second["document"] in env.cloud.releases and second["document"] not in env.cloud.grants

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# chat Result cards
# ---------------------------------------------------------------------------------------------------


def test_result_cards_rebind_only_when_their_state_changed(tmp_path):
    TUI = _load("test_u10_ui.py", "_glossarion_tui_helpers_u10fixes")
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.job_binding import CardPhase

    ws = [tmp_path / "Library" / f"Novel{i}" for i in range(2)]
    for folder in ws:
        folder.mkdir(parents=True)
        (folder / f"{folder.name}.epub").write_bytes(b"epub")

    class Cloud(TUI.FakeCloud):
        state_seq = 1
        inflight = ""

        def inflight_key(self):
            return self.inflight

    async def scenario():
        cloud, shares, page = Cloud(), TUI.FakeShares(), TUI.FakePage()
        cloud.dest = TUI.Dest("folder-abc", "folder", label="Drive › Glossarion", provider_label="Drive")
        cloud.enabled = True
        app = types.SimpleNamespace(page=page, cloud_sync=cloud, share_links=shares, files=None, opener=None,
                                    clipboard=None, shell=None, notify=lambda *a, **k: None,
                                    navigate_to=lambda *a, **k: None, _copy_text=None)
        feature = TUI._chat_feature(app)
        cards = [JobCard(attachment={"name": f.name, "extension": ".epub"}, phase=CardPhase("done")) for f in ws]
        for card, folder in zip(cards, ws):
            await feature.bind_u10_card(card, lambda f=folder: (str(f), False))
        builds = [c.u10_builds for c in cards]
        feature._u10_full, feature._u10_seq = False, cloud.state_seq  # (a fresh feature re-reads all at first)
        reads = []
        real_state = feature._u10_card_state
        feature._u10_card_state = lambda resolve: reads.append(resolve()[0]) or real_state(resolve)
        # write progress of the first book: only its card is read again, and nothing visible changed
        cloud.inflight = book_key(ws[0])
        cloud.changed()
        await asyncio.sleep(1.3)
        await TUI.settle()
        assert reads == [str(ws[0])]
        assert [c.u10_builds for c in cards] == builds  # same state: rows and buttons kept
        # a real change (the record of the second book): every card is read, the changed one rebuilt
        cloud.state_seq += 1
        cloud.files[str(ws[1])] = {"epub": {"status": "ok", "synced_at": time.time()}}
        cloud.changed()
        await asyncio.sleep(1.3)
        await TUI.settle()
        assert sorted(reads[1:]) == sorted(str(f) for f in ws)
        assert cards[0].u10_builds == builds[0] and cards[1].u10_builds == builds[1] + 1
        # a share upload's progress changes nothing a card shows
        shares.emit("upload")
        await asyncio.sleep(1.3)
        await TUI.settle()
        assert len(reads) == 3
        for unsub in feature._u10_unsubs:
            unsub()

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# chat books moved before the cloud sync existed / while it loaded
# ---------------------------------------------------------------------------------------------------


def test_moves_before_install_or_while_loading_are_queued(tmp_path):
    from glossarion_mobile.ui.chat.integration import ChatFeature

    async def scenario():
        env = Env(tmp_path)
        env.book("Story", {"Story.epub": EPUB1})
        await env.link(enable=True)
        dest_id = env.service.destination().id
        chat = tmp_path / "data" / "chat1" / "Attachments" / "Story2"
        target = env.output / "Story2"
        # the chat book's record: its cloud file, saved before the app was closed
        book_nid = env.cloud.add("root", "Story2", "folder")
        file_nid = env.cloud.add(book_nid, "Story2.epub", data=EPUB1)
        env.store.update_record(dest_id, book_key(chat), "epub", chat, doc=env.cloud.ref(file_nid),
                                parent=env.cloud.ref(book_nid), name="Story2.epub", status="ok")
        await asyncio.to_thread(env.store.flush)
        # the chat's startup sweep moves a book before the cloud sync is installed
        feature = ChatFeature.__new__(ChatFeature)
        feature.app = types.SimpleNamespace(cloud_sync=None)
        feature._cloud_moves, feature._cloud_moves_lock = [], threading.Lock()
        feature._cloud_moved(str(chat), str(target), False)
        assert feature._cloud_moves == [(str(chat), str(target), False)]
        # install: the service exists but loads its records; the handed-over move waits for the load
        env.service._loading = True
        feature.app.cloud_sync = env.service
        assert feature.attach_cloud_sync(env.service) == 1
        assert env.store.record(dest_id, book_key(target), "epub") is None  # not applied before the load
        target.mkdir(parents=True, exist_ok=True)
        (target / "Story2.epub").write_bytes(EPUB2)
        env.books.append({"name": "Story2", "path": str(target), "output_folder": str(target)})
        env.library.set_snapshot(TCS.ScanSnapshot(in_progress=tuple(env.books)))
        await env.service.start()
        await env.settle()
        moved = env.store.record(dest_id, book_key(target), "epub")
        assert moved is not None and moved.get("name") == "Story2.epub"
        assert env.store.queue() == []  # queued after the load, then saved in place
        assert env.cloud.files(book_nid) == {"Story2.epub": EPUB2}

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# the Library delete and share-link uploads
# ---------------------------------------------------------------------------------------------------


class FakeShareLinks:
    def __init__(self, live: int) -> None:
        self.live = live
        self.calls: list = []

    def deletable_links_blocking(self, identities):
        self.calls.append(("deletable", list(identities)))
        return [object()] * self.live

    def forget_books_blocking(self, identities, *, delete_remote=False):
        self.calls.append(("forget", list(identities), delete_remote))
        return [object()] if delete_remote and self.live > 1 else []


def test_library_delete_asks_before_dropping_deletable_uploads(tmp_path):
    from glossarion_mobile.ui.library.delete_confirm import DeleteFlow

    async def scenario():
        env = Env(tmp_path)
        folder = env.book("Doomed", {"Doomed.epub": EPUB1})
        shares = FakeShareLinks(live=2)
        env.library.share_links = shares
        row = env.books[0]
        assert env.library.deletable_links_blocking([row]) == 2
        shown, said = [], []

        async def io(fn, *args):
            return await asyncio.to_thread(fn, *args)

        ctx = types.SimpleNamespace(service=env.library, io=io, show=shown.append, say=said.append,
                                    push_overlay=shown.append, pop_overlay=lambda: None, shell=None,
                                    dispatcher=None, haptic=lambda *a: None, tablet=False, mono="monospace")
        flow = DeleteFlow(ctx)
        sheet = await flow.start([row])
        assert sheet is flow.links_sheet and "shared via a link" in sheet.title
        keys = {item.key: item for item in sheet.items}
        assert set(keys) == {"delete-links-remote", "delete-links-keep"}
        confirm = keys["delete-links-remote"].on_select()  # "Delete these uploads too" -> the usual confirm
        assert confirm is not None and flow.delete_remote_links is True
        report = await flow.run(None)
        assert not os.path.exists(folder)
        assert any(c[0] == "forget" and c[2] is True and folder in c[1] for c in shares.calls), shares.calls
        assert "still online until the link expires" in report.summary  # one of two could not be deleted
        # the default path keeps them online and says so
        folder2 = env.book("Kept", {"Kept.epub": EPUB1})
        plan = env.library.plan_delete_blocking([env.books[-1]])
        report = env.library.execute_delete_blocking(plan)
        assert not os.path.exists(folder2)
        assert any(c[0] == "forget" and c[2] is False and folder2 in c[1] for c in shares.calls), shares.calls
        assert "2 files shared via a link are still online until the link expires." in report.summary

    run(scenario())


def test_wipe_text_says_shared_uploads_stay_until_they_expire():
    source = (APP_DIR / "glossarion_mobile" / "ui" / "screens" / "danger_zone.py").read_text(encoding="utf-8")
    assert "stay on that service until" in source and "functools.partial(wipe_app_data" in source


# ---------------------------------------------------------------------------------------------------
# share links: Cancel after the whole file was sent
# ---------------------------------------------------------------------------------------------------


def test_cancel_after_the_file_was_sent_is_maybe_uploaded():
    from glossarion_mobile.services import share_providers as sp

    received = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            received.set()
            release.wait(5)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = sp.CancelToken()
    outcome: dict = {}

    def upload():
        try:
            sp.HttpClient(io_timeout=10).request("POST", f"http://127.0.0.1:{server.server_port}/up",
                                                 body=b"x" * 4096, cancel=token)
            outcome["result"] = "answered"
        except sp.ShareError as exc:
            outcome["error"] = exc

    client = threading.Thread(target=upload)
    client.start()
    try:
        assert received.wait(5)
        time.sleep(0.1)
        token.cancel()  # the user taps Cancel while the service is answering
        client.join(5)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
    error = outcome.get("error")
    assert error is not None and error.code == "maybe_uploaded", outcome
    assert error.message == sp.MAYBE_CANCELLED


# ---------------------------------------------------------------------------------------------------
# the Output tab's rows
# ---------------------------------------------------------------------------------------------------


def test_compiled_outputs_list_the_pdf_of_an_epub_book(tmp_path):
    env = Env(tmp_path)
    folder = env.book("Moon", {"Moon.epub": EPUB1, "Moon.pdf": PDF1, "Moon_translated.txt": b"text"})
    raw = Path(folder) / "source.pdf"
    raw.write_bytes(b"%PDF raw")
    rows = env.library.compiled_outputs_blocking({"output_folder": folder, "raw_source_path": str(raw)})
    assert [(os.path.basename(p), k) for p, k in rows] == [("Moon.epub", "epub"), ("Moon.pdf", "pdf"),
                                                           ("Moon_translated.txt", "txt")]


# ---------------------------------------------------------------------------------------------------
# share sheets: per-build dialog keys
# ---------------------------------------------------------------------------------------------------


def test_share_sheets_get_per_build_dialog_keys():
    TUI = _load("test_u10_ui.py", "_glossarion_tui_helpers_u10fixes")
    from glossarion_mobile.ui.screens import cloud_sync as u10

    page = TUI.FakePage()
    actions = u10.U10Actions(shares=TUI.FakeShares(), page=page, spawn=lambda coro: asyncio.ensure_future(coro))
    provider = {"id": "gofile", "label": "Gofile"}
    answers: list = []

    async def scenario():
        first = actions.consent_sheet(provider, None, answers.append)
        first["cancel"].on_click(None)  # Cancel; the client's dismiss has not arrived yet
        second = actions.consent_sheet(provider, None, answers.append)  # the switch flipped again at once
        return first, second

    first, second = run(scenario())
    assert first["dialog"].key != second["dialog"].key  # never diffed under the closing sheet's key
    assert first["dialog"].content.key == second["dialog"].content.key == "share-consent"
    assert page.by_key("share-consent") is second["dialog"]
    assert answers == [False]


def test_a_drain_with_nothing_to_write_starts_no_foreground_service(tmp_path):
    """Save locations mode, every output still waiting for its place: the drain posts "needs a save location"
    and writes nothing, so it never starts (and at once stops) the foreground service."""
    async def scenario():
        env = Env(tmp_path)
        env.book("BookN", {"BookN.epub": EPUB1})
        await env.service.use_save_locations()
        env.service.set_enabled(True)
        await env.settle()
        assert env.service.ui_state()["needs_pick"] == 1
        assert "start_job_service" not in env.bridge.methods(), env.bridge.methods()
        assert any("save location" in n["title"] for n in env.bridge.notifications)

    run(scenario())
