"""The owner's report: "do attachments not check the library menu? If I reuse the same attachment again, I
expect it to continue from the progress file that same attachment has in the library ... I see the metadata
being retranslated over and over again when the same attachment already translated it."

A chat attachment run translates in a fresh run root (``ChatRuns.send`` -> ``prepare_direct_text_input``),
so the same EPUB attached again started over: every chapter and the metadata were sent again. Now a file
attached in the chat that IS a Library book (same bytes as the book's raw file: ``library_book_for_file``,
the identity ``workspace_sync.same_book`` uses) is attached as that book (``ChatView.attach_library_book``):
Send continues it in its own Library workspace ("Save to: Library", the book's ``translate`` job), where
the pipeline's own resume skips the finished chapters and the translated metadata, and the run is the
book's (job origin ``bid``). A different file with the same name stays a chat attachment.

1. ``library_book_for_file`` on a fake Library: same content at another path matches, the same name with
   other bytes does not, a non-book type is never looked up, a row with a workspace wins.
2. ``test_same_attachment_twice_continues_its_library_book``: the real app on the fake OpenAI server - the
   first chat run's book joins the Library; ＋ › Files with the same file in a new chat attaches it as that
   Library book, Start runs the book's job in its workspace and the model gets no request at all
   (chapters done, metadata already translated); a different ``Again.epub`` is not matched.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

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


D = _load("_glossarion_devfix4_attach_again", "test_devfix_issue4.py")  # the app-on-fake-server helpers
pytestmark = D.pytestmark
storage = D.storage
app_env = D.app_env
iso = D.iso
migrate_calls = D.migrate_calls


# ==========================================================================
# 1. library_book_for_file
# ==========================================================================


def _fake_library(books: list):
    """What ``targets.library_targets`` reads of a LibraryService: the snapshot, raw_source, bid_for."""
    snapshot = types.SimpleNamespace(all_books=lambda: tuple(books))
    return types.SimpleNamespace(snapshot=snapshot, raw_source=lambda b: b.get("raw_source_path") or "",
                                 bid_for=lambda b: "bid-" + str(b.get("name")))


def test_library_book_for_file_matches_the_bytes_not_the_name(tmp_path):
    from glossarion_mobile.ui.chat.chat_view import library_book_for_file

    raw = tmp_path / "inbox" / "Book.epub"
    raw.parent.mkdir()
    raw.write_bytes(b"PK the book" * 100)
    workspace = tmp_path / "Output" / "Book"
    workspace.mkdir(parents=True)
    service = _fake_library([
        {"name": "Raw only", "raw_source_path": str(raw), "mtime": 1.0},
        {"name": "Book", "raw_source_path": str(raw), "output_folder": str(workspace), "mtime": 2.0},
    ])

    # the same file, and the same bytes picked again from elsewhere: the book with the workspace
    for picked in (raw, tmp_path / "picks" / "Book copy.epub"):
        if not picked.exists():
            picked.parent.mkdir()
            shutil.copyfile(raw, picked)
        target = library_book_for_file(service, str(picked))
        assert target is not None and target.bid == "bid-Book" and Path(target.folder) == workspace

    # a different file with the same name (and the same size) is not that book
    stranger = tmp_path / "elsewhere" / "Book.epub"
    stranger.parent.mkdir()
    stranger.write_bytes(b"PK the bool" * 100)
    assert library_book_for_file(service, str(stranger)) is None

    # not a book type, or gone: never looked up
    image = tmp_path / "picks" / "Book.png"
    image.write_bytes(raw.read_bytes())
    assert library_book_for_file(service, str(image)) is None
    assert library_book_for_file(service, str(tmp_path / "missing.epub")) is None


# ==========================================================================
# 2. The owner's flow on the real app
# ==========================================================================


async def _job_done(app, driver, job_id: str, timeout: float = D.RUN_TIMEOUT):
    """The job's snapshot once it ended (the first long job may ask "Keep translations running")."""
    jobs = app.chat_view.env.jobs
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await driver.count(text="Keep translations running"):
            await driver.tap(text="Not now")
        snap = jobs.snapshot_of(job_id)
        state = str(getattr(getattr(snap, "state", None), "name", getattr(snap, "state", "")) or "")
        if state in ("DONE", "FAILED", "STOPPED", "CANCELLED", "INTERRUPTED"):
            return snap
        await asyncio.sleep(0.1)
    raise AssertionError(f"the Library job {job_id} did not end")


def test_same_attachment_twice_continues_its_library_book(iso, migrate_calls):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.ui.chat.chat_view import LIBRARY_ATTACHMENT_META, TOOL_JOBS_META, library_book_for_file

    source = iso.picks / "Again.epub"
    shutil.copyfile(D.SELFTEST_EPUB, source)
    stranger = fixtures.build_tiny_epub(iso.picks / "elsewhere" / "Again.epub", chapters=3)

    async def scenario(server):
        tf, app, tester, driver = await D._start_app({})
        try:
            await D._until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            D._configure(app, server)
            migrate_calls.probe.service = app.job_service
            chats = app.chat_feature.chats
            out_root = Path(app.paths.output)
            target = out_root / "Again"
            raw = D._import(app, source)

            # ---- the first chat run: the book is translated and joins the Library ------------------
            cid1 = chats.new_chat()
            await D._finish(app, await D._send(app, driver, cid1, raw))
            assert (target / "translation_progress.json").is_file()
            assert D._progress(target) == {"total": D.CHAPTERS, "completed": D.CHAPTERS}
            library = app.library
            await library.refresh(quiet=True, reason="test")
            books = [b for b in library.snapshot.all_books() if D._same(b.get("output_folder"), target)]
            assert len(books) == 1, [(b.get("name"), b.get("output_folder")) for b in library.snapshot.all_books()]
            bid = library.bid_for(books[0])
            first_requests = server.count(status=None)
            assert first_requests > D.CHAPTERS  # chapters + metadata went to the model once

            # ---- a new chat, ＋ › Files with the same file: attached as that Library book ------------
            cid2 = chats.new_chat()
            await _open_chat(app, cid2)
            view = app.chat_view
            again = D._import(app, source)
            assert view.attach_file(again)
            view.begin_send()  # at once: Send waits for the attachment's Library lookup
            view.begin_send()  # a second tap meanwhile sends nothing more
            assert await D._until(lambda: isinstance(chats.meta(cid2).get("pending_plan"), dict), 30)
            assert (chats.meta(cid2).get(LIBRARY_ATTACHMENT_META) or {}).get("bid") == bid, chats.meta(cid2)
            assert len([m for m in chats.messages(cid2) if m[0] == "user_file"]) == 1
            plan = chats.meta(cid2)["pending_plan"]
            assert plan.get("destination") == "library" and plan.get("library_bid") == bid, plan

            # ---- Start: the book's own job in its workspace; nothing is sent to the model again -----
            mark = server.mark()
            view.start_plan()
            assert await D._until(lambda: chats.meta(cid2).get(TOOL_JOBS_META), 30), chats.meta(cid2)
            data = next(iter(chats.meta(cid2)[TOOL_JOBS_META].values()))
            assert data.get("bid") == bid, data
            snap = await _job_done(app, driver, str(data["library_job"]))
            assert snap.state.name == "DONE", (snap.state, getattr(snap, "error", None))
            assert snap.spec.origin.get("bid") == bid, snap.spec.origin
            assert D._same(snap.spec.inputs[0], raw)
            sent = [(r.kind, r.path) for r in server.records(since=mark, status=None)]
            assert sent == [], f"the continued book sent requests again: {sent}"
            assert D._progress(target) == {"total": D.CHAPTERS, "completed": D.CHAPTERS}
            assert sorted(p.name for p in out_root.iterdir() if p.name.startswith("Again")) == ["Again"]
            await library.refresh(quiet=True, reason="test")
            assert [library.bid_for(b) for b in library.snapshot.all_books()
                    if D._same(b.get("output_folder"), target)] == [bid]

            # ---- a different Again.epub is not that book: it stays a chat attachment ----------------
            service = view._library_service()
            assert await app.chat_view.env.run_io(library_book_for_file, service, str(stranger)) is None
            cid3 = chats.new_chat()
            await _open_chat(app, cid3)
            other = D._import(app, Path(stranger))
            assert view.attach_file(other)
            lookup = view._library_lookup
            if lookup is not None:
                await lookup[1]
            assert not chats.meta(cid3).get(LIBRARY_ATTACHMENT_META), chats.meta(cid3)
            view.begin_send()
            assert await D._until(lambda: isinstance(chats.meta(cid3).get("pending_plan"), dict), 15)
            assert chats.meta(cid3)["pending_plan"].get("destination", "chat") != "library"
        finally:
            await D._stop_app(tf, app)

    D._run_scenario(scenario)


async def _open_chat(app, cid: str) -> None:
    app._open_chat(cid)
    assert await D._until(lambda: app.chat_view.cid == cid, 10), (app.chat_view.cid, cid)
