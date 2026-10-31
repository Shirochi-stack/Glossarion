"""Owner issues 2 and 7 (mobile fixes 7, 2026-10-31), on a real chat attachment job in its glossary stage.

The real app objects and the real pipeline, only the model is the offline fake server (the harness of
``test_devfix_issue12``). The book's workspace does not exist while the glossary runs: the pipeline creates
``<run root>/<book>`` after it, and the glossary sits in ``<run root>/Glossary/<book>``.

Issue 2: Progress on the Job screen said "No translation progress in this chat's workspace yet", because the
lookup gave up on the missing folder. It opens the Glossary Progress view now, from the Job screen and the chat
card; the toast is left for a book with no progress of any kind.

Issue 7: the job line had no "[n/total]" during the glossary stage (the spine came from the workspace's OPF
alone), and it lost the chapter between two requests ("Translating · 1 in flight") because only an unfinished
card counted. Each step below waits on the fake model's hold, never on a sleep.

Run from src/mobile::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix7_progress_labels.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import threading
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_devfix12_for_7", Path(__file__).with_name("test_devfix_issue12.py"))
_DF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DF)
storage = _DF.storage
app_env = _DF.app_env
iso = _DF.iso
offline = _DF.offline
pytestmark = _DF.pytestmark

CHAPTERS = 6
SPINE = CHAPTERS + 1  # the fixture book's nav page is in its spine: the "/total" is the book's own spine
CHUNKS = 3


def _split_into_three(self, chapter_html, max_tokens, *args, **kwargs):
    """Every chapter in three chunks: the fixture's text is too short for the real splitter's budget."""
    return [(chapter_html, index, CHUNKS) for index in range(1, CHUNKS + 1)]


@pytest.fixture
def full_glossary(monkeypatch):
    """Full mode, chapter by chapter, in chunks (what the owner's on-device model ran)."""
    import chapter_splitter
    import unified_api_client
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    configure = _DF._configure

    def configured(app, server, glossary="balanced"):
        configure(app, server, "balanced")
        app.config_store.set_many({"auto_glossary_mode": "full", "extraction_workers": 1})
        app.config_store.flush()

    monkeypatch.setattr(_DF, "_configure", configured)
    monkeypatch.setattr(_DF, "_book", _long_book)
    monkeypatch.setattr(chapter_splitter.ChapterSplitter, "split_chapter", _split_into_three)
    unified_api_client.remember_context_window(FAKE_MODEL, 3000)  # splitting is on for a known window
    yield
    with unified_api_client._KNOWN_CONTEXT_WINDOWS_LOCK:
        unified_api_client._KNOWN_CONTEXT_WINDOWS.pop(FAKE_MODEL.lower(), None)


def _long_book(picks, chapters=CHAPTERS):
    """Chapters long enough for the glossary's chunk path (``_split_into_three`` does the splitting)."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("devfix7")
    book.set_title("Long")
    book.set_language("ko")
    body = "<p>" + "기사단장 전 빌리는 마왕성의 문을 열었다. " * 40 + "</p>"
    items = []
    for number in range(1, chapters + 1):
        item = epub.EpubHtml(title=f"제{number}장", file_name=f"chapter{number:04d}.xhtml", lang="ko")
        item.content = f"<html><head><title>제{number}장</title></head><body><h1>제{number}장</h1>{body * 40}</body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path = Path(picks) / _DF.EPUB_NAME
    epub.write_epub(str(path), book)
    return path


class _GlossaryHold:
    """Holds the fake model once ``count`` glossary answers went out: the job stays in its glossary stage."""

    def __init__(self, server, count: int) -> None:
        self.server = server
        self.left = count
        self.lock = threading.Lock()
        server.on_response.append(self)

    def __call__(self, record) -> None:
        if record.kind != "glossary":
            return
        with self.lock:
            self.left -= 1
            if self.left != 0:
                return
        self.server.hold()


async def _held(ctx, positions: list = None) -> None:
    """The fake model parked the job's next glossary request; ``positions``: every job line position until then."""
    service, job_id = ctx.app.job_service, ctx.run.job_id

    def settled() -> bool:
        if positions is not None:
            position = service.snapshot(job_id).position
            if not positions or positions[-1] != position:
                positions.append(position)
        return _DF._settled(ctx.server) and bool(service.snapshot(job_id).position) \
            and not service.snapshot(job_id).position.endswith(("(1/3)",)) is None

    assert await _DF._until(settled, 120, 0.005), ctx.probe.describe()


def test_issue7_every_surface_shows_the_spine_position_in_the_glossary_stage(iso, offline, full_glossary):
    from glossarion_mobile.services.jobs import notification_lines, position_line, progress_line, strip_model_for
    from glossarion_mobile.ui.chat.job_binding import progress_line as card_line

    shown = re.compile(r"\[(\d+)/%d\] Ch\. (\d+) \((\d)/%d\)" % (SPINE, CHUNKS))

    async def body(ctx):
        service, job_id = ctx.app.job_service, ctx.run.job_id
        positions = []
        await _held(ctx, positions)
        snap = service.snapshot(job_id)
        assert snap.state.value == "RUNNING" and snap.question is None
        assert not os.path.isdir(str(snap.output_dir)), "the workspace exists: not the owner's case"
        position = snap.position
        found = shown.fullmatch(position)
        assert found, position
        assert int(found.group(1)) == int(found.group(2)) + 1  # the nav page comes first in the spine
        assert position_line(snap) == f"Translating • {position}"
        assert progress_line(snap, verb=True).startswith(f"Translating • {position}")
        assert notification_lines(snap)[0].startswith(f"Translating • {position}")
        assert strip_model_for(snap, 0, None).subtitle.startswith(position)
        assert card_line(snap).startswith(position)
        # the chapter never blinked away between two requests, and each position has its spine number
        named = positions[next(i for i, p in enumerate(positions) if p):]
        assert all(shown.fullmatch(p) for p in named), positions
        ctx.server.release()

    server = _DF._server(0.0)
    _GlossaryHold(server, 2)
    with server:
        asyncio.run(_DF._owner_flow(iso, server, body, chapters=CHAPTERS))
    _DF._assert_offline(offline)


def test_stopping_in_the_glossary_stage_leaves_progress_working(iso, offline, full_glossary):
    """A run stopped before any translation: Progress on the Job screen still opens its glossary."""

    async def body(ctx):
        app = ctx.app
        await _held(ctx)
        job_id = ctx.run.job_id
        app.job_service.request_stop(job_id, force=True, reason="test")
        ctx.server.release()
        await _DF._finished(app, ctx.run)
        snap = app.job_service.snapshot(job_id)
        notes, routes = [], []
        navigate, notify = app.navigate_to, app.notify
        app.navigate_to = lambda *args: routes.append(args)
        app.notify = lambda *args, **kw: notes.append(args)
        try:
            bid = await app.chat_feature.open_progress(snap.output_dir)
        finally:
            app.navigate_to, app.notify = navigate, notify
        assert snap.state.value == "CANCELLED" and notes == []
        assert bid and routes == [("tools.progress.glossary", None, {"out": bid})]

    server = _DF._server(0.0)
    _GlossaryHold(server, 2)
    with server:
        asyncio.run(_DF._owner_flow(iso, server, body, chapters=CHAPTERS))
    _DF._assert_offline(offline)


def test_issue2_progress_opens_the_glossary_view_in_the_glossary_stage(iso, offline, full_glossary):
    async def body(ctx):
        app = ctx.app
        await _held(ctx)
        snap = app.job_service.snapshot(ctx.run.job_id)
        assert not os.path.isdir(str(snap.output_dir))

        notes, routes = [], []
        navigate, notify = app.navigate_to, app.notify
        app.navigate_to = lambda *args: routes.append(args)
        app.notify = lambda *args, **kw: notes.append(args)
        try:
            # the Job screen (``JobsFeature._open_progress_folder`` -> ``ChatFeature.open_progress``)
            bid = await app.chat_feature.open_progress(snap.output_dir)
            # the chat card (``ChatView._open_progress``: the run's folders, the attachment)
            view_bid = await app.chat_view._open_progress_async(ctx.probe.turn_item())
        finally:
            app.navigate_to, app.notify = navigate, notify
        assert notes == []
        assert bid and view_bid == bid
        assert routes == [("tools.progress.glossary", None, {"out": bid})] * 2

        # the screen itself: the Book page's Glossary tab reads the live progress file
        from glossarion_mobile.ui.library import progress_model as pm

        book = app.library.book_for_bid(bid)
        path = pm.find_glossary_progress_path(app.library, book)
        assert path and path.endswith("big_novel_glossary_progress.json"), path
        navigate("tools.progress.glossary", None, {"out": bid})
        await ctx.driver.wait(text="Glossary progress", timeout=30)
        await ctx.driver.wait(contains="chapter0001", timeout=30)
        ctx.server.release()

    server = _DF._server(0.0)
    _GlossaryHold(server, 2)
    with server:
        asyncio.run(_DF._owner_flow(iso, server, body, chapters=CHAPTERS))
    _DF._assert_offline(offline)


@pytest.mark.parametrize("kind", ["extract_glossary", "translate"])
def test_a_jobs_own_requests_name_their_spine_position(iso, offline, full_glossary, kind):
    """A Glossary extraction job and a Library translation run the same stage outside a chat."""
    import flows
    from glossarion_mobile.services.jobs import JobSpec

    async def scenario(server):
        epub = _long_book(iso.picks)
        tf = _DF._UF._foundations()
        app, tester, driver = await _DF._UF._host_driver(tf, {epub.name: epub})
        job_id = None
        try:
            await flows.wait_home(driver)
            _DF._configure(app, server)
            job_id = app.job_service.submit(JobSpec(kind, epub.name, (str(epub),)))
            seen = []

            def settled() -> bool:
                snap = app.job_service.snapshot(job_id)
                seen.append((snap.state.value, snap.position, [s.get("label") for s in app.job_service.request_segments(job_id)][-3:]))
                return _DF._settled(server) and bool(snap.position)

            assert await _DF._until(settled, 120), seen[-5:]
            snap = app.job_service.snapshot(job_id)
            assert re.fullmatch(r"\[\d+/%d\] Ch\. \d+ \(\d/%d\)" % (SPINE, CHUNKS), snap.position), seen[-5:]
        finally:
            if job_id:
                server.release(abort=True)
                app.job_service.request_stop(job_id, force=True, reason="test teardown")
                await asyncio.to_thread(app.job_service.wait_idle, 60)
            app.jobs.close()
            await tf._stop(app)

    server = _DF._server(0.0)
    _GlossaryHold(server, 2)
    with server:
        asyncio.run(scenario(server))


# ---- without a run -----------------------------------------------------------------------------------------


def _seg(label: str, complete: bool = False) -> dict:
    return {"label": label, "complete": complete}


def test_position_keeps_the_newest_finished_chapter_between_requests():
    from glossarion_mobile.services.jobs import request_position

    finished = [_seg("Chapter 2 (chunk 1/14) · ch2.xhtml", complete=True)]
    assert request_position(finished, lambda name: 3, 15) == "[3/15] Ch. 2 (1/14)"
    # an unfinished card that names a chapter wins; a generic request does not hide the chapter
    assert request_position(finished + [_seg("Chapter 3 · ch3.xhtml")], lambda name: 4, 15) == "[4/15] Ch. 3"
    assert request_position(finished + [_seg("Request 9")], lambda name: 3, 15) == "[3/15] Ch. 2 (1/14)"
    assert request_position([_seg("Request 9"), _seg("Metadata translation", complete=True)]) == ""
    assert request_position([_seg("Merged 1-99 (chunk 2/3) · ch1.xhtml")], lambda name: 2, 100) == "[2/100] Merged 1-99 (2/3)"


def test_the_spine_comes_from_the_jobs_own_epub_while_no_workspace_exists(tmp_path, monkeypatch):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.services.jobs import JOB_TARGET_PARAM, JobService, JobSpec, _job_epub

    monkeypatch.delenv("EPUB_PATH", raising=False)
    epub = fixtures.build_tiny_epub(tmp_path / "Book.epub", chapters=3)
    spec = JobSpec("direct_text", "Book", (str(tmp_path / "input.txt"),), params={JOB_TARGET_PARAM: [str(epub)]})
    assert _job_epub(type("J", (), {"spec": spec})) == str(epub)
    job = type("J", (), {"spec": spec, "spine": {}, "spine_files": None, "output_dir": str(tmp_path / "Book")})()
    assert JobService._spine_position(job, "chapter0002.xhtml") == 3 and len(job.spine_files) == 4


def test_a_process_wide_epub_path_of_another_book_is_not_used(tmp_path, monkeypatch):
    from glossarion_mobile.diagnostics import fixtures
    from TransateKRtoEN import _single_pass_spine_files

    other = fixtures.build_tiny_epub(tmp_path / "Other.epub", chapters=9)
    mine = fixtures.build_tiny_epub(tmp_path / "Mine.epub", chapters=2)
    monkeypatch.setenv("EPUB_PATH", str(other))
    assert len(_single_pass_spine_files(str(tmp_path / "none"), str(mine))) == 3
    assert len(_single_pass_spine_files(str(tmp_path / "none"))) == 10


def _snap(kind, position="", total=None, state=None, **fields):
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState, Progress

    return JobSnapshot(id="1", spec=JobSpec(kind, "Book", ("/b/Book.epub",), origin=fields.pop("origin", None)),
                       state=state or JobState.RUNNING, created=1.0, position=position,
                       progress=Progress(total=total, completed=0), **fields)


def test_progress_view_of_a_job():
    from glossarion_mobile.services.jobs import progress_view

    assert progress_view(_snap("extract_glossary")) == "glossary"
    assert progress_view(_snap("glossary_refine")) == "glossary"
    assert progress_view(_snap("translate", "[3/15] Ch. 2")) == "glossary"
    assert progress_view(_snap("translate", "[3/15] Ch. 2", total=15)) == "chapters"
    assert progress_view(_snap("translate")) == "chapters"


def test_glossary_folder_lookups(tmp_path):
    from glossarion_mobile.ui.chat.integration import _glossary_workspace

    root = tmp_path / "run"
    book_dir = root / "Glossary" / "Book"
    book_dir.mkdir(parents=True)
    (book_dir / "Book_glossary_progress.json").write_text("{}", encoding="utf-8")
    # the workspace the pipeline has not created yet: its name and place
    assert _glossary_workspace(str(root / "Book")) == str(root / "Book")
    # the run root with the attachment
    assert _glossary_workspace(str(root), str(tmp_path / "Book.epub")) == str(root / "Book")
    # a refinement job's output folder: the book's glossary folder
    assert _glossary_workspace(str(book_dir)) == str(root / "Book")
    stash = tmp_path / "Attachments" / "Book" / "Glossary"
    stash.mkdir(parents=True)
    (stash / "Book_glossary_progress.json").write_text("{}", encoding="utf-8")
    assert _glossary_workspace(str(stash)) == str(stash.parent)
    # nothing there at all
    assert _glossary_workspace(str(tmp_path / "gone" / "Book")) == ""
    assert _glossary_workspace("") == ""


def test_no_progress_of_any_kind_keeps_the_toast_in_plain_words(tmp_path):
    from glossarion_mobile.ui.chat.integration import ChatFeature

    notes, routes = [], []

    class Library:
        def bid_for(self, book):
            return "bid"

    feature = ChatFeature.__new__(ChatFeature)
    feature.app = type("App", (), {"library": Library(), "navigate_to": staticmethod(lambda *a: routes.append(a)),
                                   "notify": staticmethod(lambda *a: notes.append(a))})()

    async def run_io(function, *args):
        return function(*args)

    feature._run_io = run_io
    assert asyncio.run(feature.open_progress(str(tmp_path / "gone" / "Book"))) is None
    assert routes == [] and notes == [("No progress to show yet: nothing has been translated or extracted for this book",)]


def test_the_job_screen_asks_for_the_glossary_view():
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    def screen(kind, origin=None, position=""):
        calls = []
        item = JobDetailScreen.__new__(JobDetailScreen)
        item.snap = _snap(kind, position, origin=origin, output_dir="/o/Book")
        item.navigate = lambda *a: calls.append(a)
        item.open_progress = lambda *a, **kw: calls.append((a, kw))
        return item, calls

    item, calls = screen("extract_glossary")
    item._on_progress()
    assert calls == [(("/o/Book",), {"glossary": True})]
    item, calls = screen("translate", {"type": "library", "bid": "b1"}, "[3/15] Ch. 2")
    item._on_progress()
    assert calls == [("library.book", {"bid": "b1"}, {"tab": "glossary"})]
    item, calls = screen("translate", {"type": "library", "bid": "b1"}, "[3/15] Ch. 2")
    item._on_progress(failed=True)
    assert calls == [("library.book", {"bid": "b1"}, {"tab": "chapters", "filter": "failed"})]
    item, calls = screen("translate")
    item._on_progress()
    assert calls == [(("/o/Book",), {})]


def test_a_long_request_still_moves_the_elapsed_time():
    """The Jobs row and the job strip showed different times (3:39 / 2:25): each repaints only on a job change,
    and nothing changes while one long request runs."""
    import types

    from glossarion_mobile.services.jobs import JobService

    repainted = threading.Event()
    ticks = iter(range(0, 1000, 5))
    service = types.SimpleNamespace(stream_drain_interval=0.01, elapsed_tick_interval=1.0, clock=lambda: next(ticks),
                                    _track_position=lambda job: None, _changed=repainted.set)
    job = types.SimpleNamespace(stream=types.SimpleNamespace(drain=lambda final=False: None))
    with JobService._draining(service, job):
        assert repainted.wait(30)


def test_the_chat_cards_line_leads_with_the_position():
    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.chat.job_binding import progress_line

    running = _snap("direct_text", "[17/182] Ch. 13 (1/2)", 48, in_flight=3)
    ended = _snap("direct_text", "[17/182] Ch. 13 (1/2)", 48, JobState.DONE, in_flight=3)
    assert progress_line(running) == "[17/182] Ch. 13 (1/2) · Chapter 0/48 · 3 in flight"
    assert progress_line(ended) == "Chapter 0/48 · 3 in flight"


# ---- review findings -----------------------------------------------------------------------------------------


def test_a_card_that_names_no_file_takes_its_chapters_file_or_the_chapter_number():
    """The client's own "Chapter 2 ... API call in progress" line starts a card with no file under another thread
    key; it was the newest card, so the "[n/total]" went missing on every surface."""
    from glossarion_mobile.services.jobs import request_position

    def spine_of(name):
        return {"ch2.xhtml": 3}.get(name, 0)

    # no card of the chapter names its file: by the chapter's number
    assert request_position([_seg("Chapter 2")], spine_of, 15, lambda number: 3 if number == "2" else 0) == "[3/15] Ch. 2"
    # an unfinished card of the chapter that names it, whichever came first
    dispatched = _seg("Chapter 2 (chunk 1/14) · ch2.xhtml")
    bare = _seg("Chapter 2 (chunk 1/14)")
    assert request_position([dispatched, bare], spine_of, 15) == "[3/15] Ch. 2 (1/14)"
    assert request_position([bare, dispatched], spine_of, 15) == "[3/15] Ch. 2 (1/14)"
    # a finished chunk of that chapter names it as well
    assert request_position([_seg("Chapter 2 (chunk 1/14) · ch2.xhtml", complete=True), _seg("Chapter 2 (chunk 2/14)")],
                            spine_of, 15) == "[3/15] Ch. 2 (2/14)"
    # another chapter's file is not this chapter's: "Chapter 12", "Section 2"
    assert request_position([_seg("Chapter 12 · ch2.xhtml"), _seg("Chapter 2")], spine_of, 15) == "Ch. 2"
    assert request_position([_seg("Section 2 · ch2.xhtml"), _seg("Chapter 2")], spine_of, 15) == "Ch. 2"
    # still nothing known: the chapter alone
    assert request_position([_seg("Chapter 2")], spine_of, 15) == "Ch. 2"


def _spine_job(tmp_path, chapters=3):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.services.jobs import JOB_TARGET_PARAM, JobSpec, _Job

    epub = fixtures.build_tiny_epub(tmp_path / "Book.epub", chapters=chapters)
    spec = JobSpec("translate", "Book", (str(epub),), params={JOB_TARGET_PARAM: [str(epub)]})
    return _Job(id="1", spec=spec, created=1.0, output_dir=str(tmp_path / "Book"))


def test_the_job_line_of_a_chapter_label_without_a_file(tmp_path, monkeypatch):
    """The real spine of the job's EPUB, its nav page first: "Chapter 2" is its third entry."""
    from glossarion_mobile.services.jobs import JobService

    monkeypatch.delenv("EPUB_PATH", raising=False)
    job = _spine_job(tmp_path)
    job.stream = type("Stream", (), {"segments": staticmethod(lambda drain=False: [_seg("Chapter 2")])})()
    service = JobService.__new__(JobService)
    service._lock = threading.Lock()
    service._changed = lambda: None
    service._track_position(job)
    assert job.position == "[3/4] Ch. 2"


def test_the_chapter_number_finds_its_spine_entry_or_nothing(tmp_path, monkeypatch):
    from glossarion_mobile.services.jobs import JobService

    monkeypatch.delenv("EPUB_PATH", raising=False)
    job = _spine_job(tmp_path)
    assert JobService._chapter_spine_position(job, "2") == 3
    assert JobService._chapter_spine_position(job, "1-99") == 2  # a merged request: its first chapter
    for unknown in ("9", "0", "2.5", "12", "-1", ""):
        assert JobService._chapter_spine_position(job, unknown) == 0, unknown
    # the pipeline's own numbering: a sequence that restarts carries on, the nav and cover pages have no number
    job.spine_files, job.chapter_spine = ["cover.xhtml", "a01.xhtml", "a02.xhtml", "b01.xhtml"], None
    assert [JobService._chapter_spine_position(job, n) for n in "123"] == [2, 3, 4]
    # two files with one number: no position beats a wrong one
    job.spine_files, job.chapter_spine = ["a01.xhtml", "b01.xhtml", "b02.xhtml"], None
    assert [JobService._chapter_spine_position(job, n) for n in "12"] == [0, 3]
    # no spine yet: asked again on the next request
    job.spine_files, job.chapter_spine = None, None
    job.spec = type("Spec", (), {"params": {}, "inputs": ()})()
    assert JobService._chapter_spine_position(job, "2") == 0 and job.chapter_spine is None


@pytest.fixture
def glossary_only_book(tmp_path, monkeypatch):
    """The real Library and Glossary services over a folder that holds only a book's extracted glossary."""
    import types

    import library_core as lc

    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.services.library import LibraryService

    library, output = tmp_path / "Library", tmp_path / "out"
    (library / "Raw").mkdir(parents=True)
    book_dir = output / "Glossary" / "Novel"
    book_dir.mkdir(parents=True)
    (book_dir / "Novel_glossary_progress.json").write_text("{}", encoding="utf-8")
    (book_dir / "Novel_glossary.csv").write_text("type,raw_name,translated_name\n", encoding="utf-8")
    for key, value in (("GLOSSARION_LIBRARY_DIR", library), ("OUTPUT_DIRECTORY", output),
                       ("HOME", tmp_path), ("USERPROFILE", tmp_path)):
        monkeypatch.setenv(key, str(value))
    paths = types.SimpleNamespace(library=library, output=output, cache=tmp_path / "cache")
    service = LibraryService(paths=paths, config={})
    service.ensure_env()
    service.scan_blocking()
    try:
        yield types.SimpleNamespace(library=service, glossary=GlossaryService(paths=paths, config={}, library=service),
                                    book_dir=book_dir, root=output)
    finally:
        try:
            lc.uninstall_library_env()
        except Exception:
            pass


def _glossary_feature(library, glossary, chat=True):
    """The Glossaries feature and the routes it opens, over the real ``ChatFeature.open_progress``."""
    import types

    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.glossary.feature import GlossaryFeature

    routes = []
    app = types.SimpleNamespace(library=library, navigate_to=lambda *a: routes.append(a), notify=routes.append)
    if chat:
        app.chat_feature = ChatFeature.__new__(ChatFeature)
        app.chat_feature.app = app

        async def run_io(function, *args):
            return function(*args)

        app.chat_feature._run_io = run_io
    feature = GlossaryFeature.__new__(GlossaryFeature)
    feature.app, feature.dispatcher, feature.service = app, None, glossary
    feature.context = lambda: types.SimpleNamespace(go=lambda *a: routes.append(a))
    return feature, routes


def _book_row(env):
    rows = [r for r in env.glossary.list_glossaries() if r.kind == "book" and r.book == "Novel"]
    assert rows, "the Glossaries home lists the book's glossary"
    return rows[0]


def test_glossary_progress_of_a_book_that_is_not_on_the_shelf(glossary_only_book):
    """Both Glossary screen entries (the row's action and the editor's menu) of a glossary-only book: its manager,
    never the SourcePicker's bare route. The row carries its book to the manager's own file lookup."""
    from glossarion_mobile.ui.library import progress_model as pm

    env = glossary_only_book
    row = _book_row(env)
    assert env.glossary.book_for_glossary(row) is None
    feature, routes = _glossary_feature(env.library, env.glossary)

    async def from_row():
        return await feature.open_progress_for(row)

    bid = asyncio.run(from_row())
    assert routes == [("tools.progress.glossary", None, {"out": bid})]
    book = env.library._by_bid[bid]
    assert book["name"] == "Novel" and book["output_folder"] == str(env.root / "Novel")
    assert os.path.samefile(pm.find_glossary_progress_path(env.library, book),
                            env.book_dir / "Novel_glossary_progress.json")

    routes.clear()

    async def from_menu():
        return await feature.open_progress_for_path(str(env.book_dir / "Novel_glossary.csv"))

    assert asyncio.run(from_menu()) == bid
    assert routes == [("tools.progress.glossary", None, {"out": bid})]


def test_glossary_progress_of_a_book_on_the_shelf_keeps_the_librarys_row(glossary_only_book):
    env = glossary_only_book
    feature, routes = _glossary_feature(env.library, env.glossary)
    row = _book_row(env)
    env.glossary.book_for_glossary = lambda glossary: {"name": "Novel", "output_folder": str(env.root / "Novel")}
    bid = feature.open_progress_for(row)
    assert routes == [("tools.progress.glossary", None, {"out": bid})] and isinstance(bid, str)


def test_glossary_progress_without_the_chat_feature_keeps_the_picker(glossary_only_book):
    env = glossary_only_book
    feature, routes = _glossary_feature(env.library, env.glossary, chat=False)
    assert feature.open_progress_for(_book_row(env)) is None
    assert routes == [("tools.progress.glossary",)]
