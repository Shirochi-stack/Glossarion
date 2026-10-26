"""Hotfix 2, owner device report (Tools › Review "Review books" picker, 2026-10-26):

1. "It is showing library and chat workspaces as 2 separate tabs. Merge them" - the Tools SourcePicker
   offered Recent outputs · Library books · Chat workspaces · Browse. Now: Recent outputs · Books · Browse;
   Books is one list of the Library's books and the chat books, one row per book
   (``targets.merge_book_rows``: a Library book the chat translated again is one row), each labelled
   where it was found (📚 Library · 💬 Chat). ``segment="library"`` (every Tools caller) opens Books; the
   chat's ＋ › From Library keeps its Library-only ``segments=("library",)``.
2. "It is unable to detect my attachment and is showing the same one twice" - the chat tab walked
   ``Output/Direct Text`` and listed every folder holding HTML: the chat's ``Chat Messages`` (its text
   replies) as a book, and the book itself (``<chat>/Attachments/<book>``) without a raw file, so both
   were disabled with "No raw source file" (``targets.chat_workspace_targets`` never resolved one; a chat
   whose own folder held a ``Direct Text N`` reply hid its books altogether). Now the chat's own folder
   and ``Chat Messages`` are never books, and a chat book's raw file is the attachment, resolved by the
   shared resolvers (``LibraryService.raw_source``: the translator's ``source_epub.txt`` names it).
3. "Recent outputs tab is also empty" - Recent outputs read only the Library's In progress shelf, and a
   translation that has just finished is on the Completed shelf (a chat book that joined the Library
   too); chat books were never there. Now it lists every Library and chat workspace, newest translated
   first (``translation_progress.json`` of any of the book's workspaces), one row per book with the
   Books row's key; Library imports nobody translated ("Not started") stay out.

``test_rows_merge_and_order`` checks the row building on a synthetic tree (pure; no Flet).
``test_review_and_async_pickers_on_real_chat_and_library_books`` is the owner's flow on the real app
objects (``test_devfix_issue4`` harness: fake Flet session, real JobService / ChatFeature / LibraryService,
the shared pipeline, the offline fake OpenAI server): a chat book that joined the Library, the owner's
chat (a typed message, then the EPUB attachment; its book stays in ``Attachments``), the Library book
translated again in another chat, a Library import translated from the Library; then Tools › Review and
Tools › Async batch, what each picker lists and what the jobs receive.

Real data stays untouched: the harness runs in temp FLET_APP_STORAGE_* dirs (HOME, OUTPUT_DIRECTORY,
GLOSSARION_LIBRARY_DIR, CONFIG_FILE there) with USERPROFILE / APPDATA / LOCALAPPDATA in tmp_path.

Run from src/mobile with the mobile venv::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix2_source_picker.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
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

from glossarion_mobile.ui.tools import targets as tg  # noqa: E402


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _load("_glossarion_devfix4_hotfix2_picker", "test_devfix_issue4.py")  # the app-on-fake-server helpers
storage = D.storage
app_env = D.app_env
iso = D.iso

OWNER_NAME = "[211151] [TS]나 혼자 메카닉 천재 소녀.epub"
PROGRESS = "translation_progress.json"


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _age(path: Path, seconds: float, now: float) -> None:
    os.utime(path, (now - seconds, now - seconds))


# ==========================================================================
# 1. The rows (pure)
# ==========================================================================


class FakeLibrary:
    """What the targets read of a LibraryService: the snapshot (books + card views), ``raw_source``
    (recorded: a chat book's workspace row goes through it) and ``bid_for``."""

    def __init__(self, books, views, sources):
        self.snapshot = types.SimpleNamespace(all_books=lambda: list(books), views=dict(views))
        self.sources = {_norm(k): str(v) for k, v in sources.items()}
        self.asked: list = []

    def raw_source(self, book):
        self.asked.append(dict(book))
        if book.get("raw_source_path"):
            return book["raw_source_path"]
        return self.sources.get(_norm(book.get("output_folder") or ""), "")

    def bid_for(self, book):
        return "bid-" + str(book.get("name"))


def test_rows_merge_and_order(tmp_path):
    from glossarion_mobile.services.library import book_key

    now = time.time()
    out = tmp_path / "Output"
    chats = out / "Direct Text"
    inbox = tmp_path / "Inbox"
    lib_raw = _write(tmp_path / "Library" / "Raw" / "Lib Book.epub")
    owner_raw = _write(inbox / OWNER_NAME)
    import_raw = _write(tmp_path / "Library" / "Raw" / "Fresh Import.epub")
    shelf_raw = _write(tmp_path / "Library" / "Raw" / "Shelf Only.epub")
    # the Library: a translated workspace, an import nobody translated, a book without a workspace
    lib_ws = out / "Lib Book"
    _write(lib_ws / PROGRESS, "{}")
    _write(lib_ws / "response_001.html")
    fresh_ws = out / "Fresh Import"
    _write(fresh_ws / PROGRESS, "{}")
    lib_book = {"name": "Lib Book", "output_folder": str(lib_ws), "raw_source_path": str(lib_raw), "mtime": 1.0}
    fresh = {"name": "Fresh Import", "output_folder": str(fresh_ws), "raw_source_path": str(import_raw),
             "mtime": 2.0, "is_in_progress": True}
    shelf = {"name": "Shelf Only", "path": str(tmp_path / "Library" / "Translated" / "Shelf Only.epub"),
             "raw_source_path": str(shelf_raw), "mtime": 3.0, "in_library": True}
    # the owner's chat: a typed reply in the chat's own folder, its message bodies, the attached book
    owner_chat = chats / f"{OWNER_NAME} - 20261026_041614_e68060c2"
    _write(owner_chat / "Direct Text 1.html")
    _write(owner_chat / "Chat Messages" / "000002-response.html")
    owner_ws = owner_chat / "Attachments" / Path(OWNER_NAME).stem
    _write(owner_ws / PROGRESS, "{}")
    _write(owner_ws / "response_001.html")
    # another chat translated the Library book again; a third one the book that has no workspace
    again_chat = chats / "Lib Book.epub - 20261026_050000_aaaaaaaa"
    _write(again_chat / "Chat Messages" / "000002-response.html")
    again_ws = again_chat / "Attachments" / "Lib Book"
    _write(again_ws / PROGRESS, "{}")
    shelf_ws = chats / "Shelf Only.epub - 20261026_060000_bbbbbbbb" / "Attachments" / "Shelf Only"
    _write(shelf_ws / PROGRESS, "{}")
    for folder, age in ((owner_ws, 30), (again_ws, 20), (shelf_ws, 10)):
        _age(folder, age, now)
    service = FakeLibrary([lib_book, fresh, shelf], {book_key(fresh): {"state": "not_started"}},
                          {owner_ws: owner_raw, again_ws: lib_raw, shelf_ws: shelf_raw})

    chat_rows = tg.chat_workspace_targets(str(chats), service=service)
    # one row per chat book, newest first; never Chat Messages nor a chat's own folder
    assert [Path(t.folder).name for t in chat_rows] == ["Shelf Only", "Lib Book", Path(OWNER_NAME).stem]
    assert [t.title for t in chat_rows] == ["Shelf Only", "Lib Book", Path(OWNER_NAME).stem]
    assert all(t.origin == "chat" and t.direct_text and tg.found_in(t) == ("chat",) for t in chat_rows)
    owner_row = chat_rows[2]
    assert owner_row.source == str(owner_raw) and owner_row.kind == "epub" and owner_row.extra["chat"] == owner_chat.name
    # the raw file comes from the Library's resolvers, asked with a workspace row
    asked = {_norm(b["output_folder"]): b for b in service.asked}
    assert set(asked) == {_norm(owner_ws), _norm(again_ws), _norm(shelf_ws)}
    assert asked[_norm(owner_ws)]["type"] == "in_progress" and _norm(asked[_norm(owner_ws)]["path"]) == _norm(owner_ws)
    # no service: the rows are listed, nothing is resolved (the pre-fix rows had no raw file either)
    assert [t.source for t in tg.chat_workspace_targets(str(chats))] == ["", "", ""]

    library_rows = tg.library_targets(service)
    books = tg.merge_book_rows(library_rows, chat_rows)
    # the Library rows in their order, then the chat book that is in no Library row
    assert [t.title for t in books] == ["Lib Book", "Fresh Import", "Shelf Only", Path(OWNER_NAME).stem]
    by_title = {t.title: t for t in books}
    merged = by_title["Lib Book"]  # same raw file: one row, the Library's workspace, both workspaces kept
    assert tg.found_in(merged) == ("library", "chat") and merged.folder == str(lib_ws) and merged.bid == "bid-Lib Book"
    assert [_norm(w) for w in merged.extra["workspaces"]] == [_norm(lib_ws), _norm(again_ws)]
    assert merged.key == library_rows[0].key
    shelf_row = by_title["Shelf Only"]  # a Library book without a workspace takes the chat's
    assert tg.found_in(shelf_row) == ("library", "chat") and _norm(shelf_row.folder) == _norm(shelf_ws)
    assert shelf_row.source == str(shelf_raw) and shelf_row.direct_text
    assert tg.found_in(by_title["Fresh Import"]) == ("library",)
    assert tg.found_in(by_title[Path(OWNER_NAME).stem]) == ("chat",)
    assert len({t.key for t in books}) == len(books)
    # the same chat folder listed by the Library (a chat book filed by the desktop): one row
    filed = tg.ToolTarget("Filed", folder=str(owner_ws), source="", origin="library")
    filed_rows = tg.merge_book_rows([filed], [owner_row])
    assert [tg.found_in(t) for t in filed_rows] == [("library", "chat")] and filed_rows[0].source == str(owner_raw)

    # Recent outputs: newest translated first over every workspace of a book; "Not started" stays out
    _age(lib_ws / PROGRESS, 500, now)
    _age(again_ws / PROGRESS, 100, now)  # the Library book was translated again in a chat, after the others
    _age(owner_ws / PROGRESS, 300, now)
    _age(shelf_ws / PROGRESS, 400, now)
    recent = tg.recent_output_targets(service, library_rows=library_rows, chat_rows=chat_rows)
    assert [t.title for t in recent] == ["Lib Book", Path(OWNER_NAME).stem, "Shelf Only"]
    assert [t.mtime for t in recent] == sorted((t.mtime for t in recent), reverse=True)
    assert abs(recent[0].mtime - (now - 100)) < 2
    assert {t.key for t in recent} <= {t.key for t in books}
    assert all(t.origin == "recent" for t in recent)
    assert [tg.found_in(t) for t in recent] == [("library", "chat"), ("chat",), ("library", "chat")]
    # without chat books (the old call) the Library workspaces are still there, Completed ones too
    assert [t.title for t in tg.recent_output_targets(service, library_rows=library_rows)] == ["Lib Book"]


def test_picker_segments_and_labels(tmp_path):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.tools import source_picker as sp
    from glossarion_mobile.ui.tools.common import ToolsContext

    ctx = ToolsContext(service=None, page=None, store={})
    picker = sp.SourcePicker(ctx, title="Review books", multi=True, segment="library")
    assert sp.SEGMENTS == (("recent", "Recent outputs"), ("books", "Books"), ("browse", "Browse"))
    assert picker.segments == ("recent", "books", "browse") and picker.segment == "books"
    assert [s.value for s in picker.segmented.segments] == ["recent", "books", "browse"]
    assert [s.label.value for s in picker.segmented.segments] == ["Recent outputs", "Books", "Browse"]
    picker.set_segment("chat")
    assert picker.segment == "books"
    picker.set_segment("nonsense")
    assert picker.segment == "recent"
    # the chat's ＋ › From Library keeps one Library-only list; the chat books alone stay opt-in
    assert sp.SourcePicker(ctx, title="x", segments=("library",), segment="library").segments == ("library",)
    assert sp.SourcePicker(ctx, title="x", segments=("chat", "browse"), segment="library").segment == "chat"
    row = tg.ToolTarget("Book", folder=str(tmp_path / "Book"), source=str(tmp_path / "Book.epub"), origin="library",
                        extra={"found_in": ("library", "chat")})
    assert sp.found_in_label(row) == "📚 Library · 💬 Chat"
    assert sp._subtitle(row).startswith("📚 Library · 💬 Chat · 📁 Book · 📖 Book.epub")
    assert sp.found_in_label(tg.ToolTarget("Picked", source="x.epub", origin="browse")) == ""


# ==========================================================================
# 2. The owner's flow on the real app
# ==========================================================================


def _titled_epub(path: Path, title: str, chapters: int = 3) -> Path:
    """A tiny Korean EPUB (``fixtures.build_tiny_epub``'s shape) with its own title: the Library pairs
    workspaces and compiled EPUBs by title, so the books must not share one."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"hotfix2-{abs(hash(title))}")
    book.set_title(title)
    book.set_language("ko")
    items = []
    for index in range(1, chapters + 1):
        chapter = epub.EpubHtml(title=f"제{index}장", file_name=f"chapter{index:04d}.xhtml", lang="ko")
        chapter.content = (f"<html><head><title>제{index}장</title></head><body><h1>제{index}장</h1>"
                           f"<p>{title} {index}번째 단락입니다. 오늘은 맑은 날이었다.</p></body></html>")
        book.add_item(chapter)
        items.append(chapter)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def kept_chats(monkeypatch):
    """The chats whose books stay in their ``Attachments`` (the owner's book did: a run that can still be
    resumed, a job still running, ...): the auto-migrate skips them."""
    from glossarion_mobile.ui.chat.integration import MIGRATE_SKIPPED, ChatFeature

    kept: set = set()
    original = ChatFeature.auto_migrate_blocking

    def migrate(self, cid, folder):
        if str(cid) in kept:
            return {"cid": str(cid), "folder": str(folder), "status": MIGRATE_SKIPPED, "reason": "kept by the test",
                    "target": "", "source": ""}
        return original(self, cid, folder)

    monkeypatch.setattr(ChatFeature, "auto_migrate_blocking", migrate)
    return kept


async def _send_text(app, driver, cid, text):
    """A typed chat message (``ChatRuns.send`` without an attachment): its reply lands in the chat's own
    folder (``Direct Text N``) and ``Chat Messages``."""
    settings = app.chat_view.settings(str(cid))
    task = asyncio.ensure_future(app.chat_feature.runs.send(cid, text=text, attachment=None, settings=settings,
                                                            output_mode="text"))
    deadline = time.monotonic() + 60
    while not task.done():
        if await driver.count(text="Keep translations running"):
            await driver.tap(text="Not now")
        if time.monotonic() > deadline:
            task.cancel()
            raise AssertionError("ChatRuns.send did not return for the typed message")
        await asyncio.sleep(0.05)
    return task.result()


async def _job_done(app, driver, job_id, timeout: float = D.RUN_TIMEOUT):
    service = app.job_service

    def ended():
        snap = service.snapshot(job_id)
        return snap if snap is not None and snap.is_terminal else None

    deadline = time.monotonic() + timeout
    while ended() is None and time.monotonic() < deadline:
        if await driver.count(text="Keep translations running"):
            await driver.tap(text="Not now")
        await asyncio.sleep(0.05)
    return ended()


async def _idle_and_scanned(app) -> None:
    """No job active or queued, then a Library scan of what is on disk now (a scan already running when
    ``refresh`` is called makes it return at once, maybe before it saw the newest files)."""
    service = app.job_service

    def idle() -> bool:
        view = service.view()
        return view.active is None and not tuple(getattr(view, "queue", ()) or ())

    assert await D._until(idle, D.RUN_TIMEOUT), service.view()
    library = app.library
    assert await D._until(lambda: not library.scanning, 60)
    await library.refresh(quiet=True, reason="hotfix2 test")
    assert library.snapshot.scanned_at


async def _screen(app, route: str, name: str):
    app.navigate_to(route)
    assert await D._until(lambda: type(app.shell.top_screen).__name__ == name, 15), app.shell.current_route
    return app.shell.top_screen


async def _opened(picker):
    assert await D._until(lambda: picker.loaded, 60), "the picker did not load"
    return picker


def _texts(control) -> str:
    value = getattr(control, "value", None)
    out = [str(value)] if isinstance(value, str) else []
    for name in ("content", "title", "subtitle", "leading", "trailing"):
        child = getattr(control, name, None)
        if child is not None and not isinstance(child, (str, int, float, bool)):
            out.append(_texts(child))
    for child in getattr(control, "controls", None) or ():
        out.append(_texts(child))
    return " ".join(t for t in out if t)


@D.pytestmark
def test_review_and_async_pickers_on_real_chat_and_library_books(iso, kept_chats):
    from glossarion_mobile.ui.tools.source_picker import SourcePicker

    alpha_src = _titled_epub(iso.picks / "Alpha Book.epub", "알파의 책")
    owner_src = _titled_epub(iso.picks / "owner" / OWNER_NAME, "나 혼자 메카닉 천재 소녀")
    gamma_src = _titled_epub(iso.picks / "Gamma Library.epub", "감마 서재")
    owner_stem = Path(OWNER_NAME).stem

    async def scenario(server):
        tf, app, tester, driver = await D._start_app({})
        try:
            await D._until(lambda: getattr(app, "chat_feature", None) is not None and app.library is not None, 30)
            D._configure(app, server)
            chats = app.chat_feature.chats
            library = app.library
            out_root = Path(app.paths.output)

            # ---- a chat book that joined the Library by itself (Completed shelf) ------------------------
            alpha_raw = D._import(app, alpha_src)
            cid1 = chats.new_chat()
            run1 = await D._send(app, driver, cid1, alpha_raw)
            await D._finish(app, run1)
            assert run1.state == "done", run1.state
            alpha_ws = out_root / "Alpha Book"
            assert (alpha_ws / PROGRESS).is_file() and chats.attachment_folders(cid1) == []

            # ---- the owner's chat: a typed message, then the EPUB; the book stays in Attachments ----------
            cid2 = chats.new_chat()
            kept_chats.add(str(cid2))
            text_run = await _send_text(app, driver, cid2, "안녕하세요. 오늘은 맑은 날이었다.")
            await D._finish(app, text_run)
            owner_raw = D._import(app, owner_src)
            run2 = await D._send(app, driver, cid2, owner_raw)
            await D._finish(app, run2)
            assert run2.state == "done", run2.state
            owner_chat = Path(chats.output_folder(cid2))
            assert (owner_chat / "Chat Messages").is_dir(), sorted(p.name for p in owner_chat.iterdir())
            assert list(owner_chat.glob("Direct Text *")), sorted(p.name for p in owner_chat.iterdir())
            folders = chats.attachment_folders(cid2)
            assert [Path(f).name for f in folders] == [owner_stem], folders
            owner_ws = Path(folders[0])
            attached = [m[2] for m in chats.messages(cid2) if m[0] == "user_file"]
            assert len(attached) == 1 and D._same(attached[0], owner_raw)

            # ---- the Library book translated again in another chat (stays in that chat) ----------------
            cid3 = chats.new_chat()
            kept_chats.add(str(cid3))
            run3 = await D._send(app, driver, cid3, alpha_raw)
            await D._finish(app, run3)
            assert run3.state == "done", run3.state
            alpha_chat_ws = Path(chats.attachment_folders(cid3)[0])
            assert alpha_chat_ws.name == "Alpha Book" and (alpha_chat_ws / PROGRESS).is_file()

            # ---- a Library import: Not started (no Recent output), then translated from the Library -----
            await asyncio.to_thread(library.import_paths_blocking, [str(gamma_src)])
            await _idle_and_scanned(app)
            gamma = next(b for b in library.snapshot.all_books() if Path(str(b.get("output_folder") or "")).name
                         == "Gamma Library")
            gamma_ws = Path(gamma["output_folder"])
            gamma_raw = library.raw_source(gamma)
            assert gamma_raw and Path(gamma_raw).parent.name == "Raw"
            ctx = app.tools.context()
            library_rows = tg.library_targets(library)
            chat_rows = tg.chat_workspace_targets(ctx.chats_root, service=library)
            assert _norm(gamma_ws) in {_norm(t.folder) for t in tg.merge_book_rows(library_rows, chat_rows)}
            recent = tg.recent_output_targets(library, library_rows=library_rows, chat_rows=chat_rows)
            assert _norm(gamma_ws) not in {_norm(t.folder) for t in recent}, "a Not started import is no output"
            job_id = await library.submit(library.translate_spec([gamma]))
            snap = await _job_done(app, driver, job_id)
            assert snap is not None and str(getattr(snap.state, "value", snap.state)).lower() == "done", snap
            await _idle_and_scanned(app)
            translated = next(b for b in library.snapshot.all_books() if _norm(b.get("output_folder") or "") ==
                              _norm(gamma_ws))
            assert (library.card_view(translated) or {}).get("state") != "not_started", translated

            # the order the books were translated in (pinned: no reliance on the clock's resolution)
            now = time.time()
            for folder, age in ((alpha_ws, 400), (owner_ws, 300), (alpha_chat_ws, 200), (gamma_ws, 100)):
                _age(folder / PROGRESS, age, now)

            # ---- Tools › Review › Choose…: Recent outputs · Books · Browse ----------------------------
            review = await _screen(app, "tools.review", "ReviewScreen")
            picker = await _opened(review.open_picker())
            assert picker.sheet.open and picker.segment == "books"
            assert [s.label.value for s in picker.segmented.segments] == ["Recent outputs", "Books", "Browse"]
            books = picker.rows["books"]
            by_folder = {_norm(t.folder): t for t in books}
            # one row per book: the Library's two, the owner's chat book; the re-translated copy is the Library row
            assert set(by_folder) == {_norm(alpha_ws), _norm(gamma_ws), _norm(owner_ws)}, [t.folder for t in books]
            assert len(books) == 3 and picker.status.value == "3 items"
            every = [t for rows in picker.rows.values() for t in rows]
            assert not [t for t in every if Path(t.folder).name == "Chat Messages"], "Chat Messages listed as a book"
            chat_folders = {_norm(chats.output_folder(c)) for c in (cid1, cid2, cid3)}
            assert not [t for t in every if _norm(t.folder) in chat_folders], "a chat's own folder listed as a book"
            owner = by_folder[_norm(owner_ws)]
            alpha = by_folder[_norm(alpha_ws)]
            gamma_row = by_folder[_norm(gamma_ws)]
            # the chat attachment is detected: its raw file is the EPUB the chat stored, so it can be reviewed
            assert D._same(owner.source, owner_raw) and picker.reasons[owner.key] is None
            assert tg.found_in(owner) == ("chat",) and owner.title == owner_stem
            assert D._same(alpha.source, alpha_raw) and tg.found_in(alpha) == ("library", "chat")
            assert D._same(gamma_row.source, gamma_raw) and tg.found_in(gamma_row) == ("library",)
            assert all(picker.reasons[t.key] is None for t in books)
            picker.render()
            assert "💬 Chat" in _texts(picker.tiles[owner.key]) and "📚 Library" not in _texts(picker.tiles[owner.key])
            assert "📚 Library · 💬 Chat" in _texts(picker.tiles[alpha.key])
            # Recent outputs: newest translated first, the Books rows' keys
            recent = picker.rows["recent"]
            assert [_norm(t.folder) for t in recent] == [_norm(gamma_ws), _norm(alpha_ws), _norm(owner_ws)], \
                [(t.title, t.mtime) for t in recent]
            assert [t.key for t in recent] == [gamma_row.key, alpha.key, owner.key]
            picker.set_segment("recent")
            assert picker.status.value == "3 items" and len(picker.list_view.controls) == 3
            picker.set_segment("books")

            # selection -> what the review job receives
            picker.toggle(owner)
            picker.toggle(alpha)
            picker._on_done()
            assert not picker.sheet.open
            assert [t.key for t in review.targets] == [owner.key, alpha.key]
            submitted: list = []

            async def submit(spec):
                submitted.append(spec)
                return f"captured-{len(submitted)}"

            review.ctx.submit = submit
            assert await review.start("single") == "captured-1"
            spec = submitted[-1]
            assert spec.kind == "review" and spec.params["mode"] == "single" and len(spec.inputs) == 1
            assert D._same(spec.inputs[0], owner_raw)
            review.step(1)
            await review.start("single")
            assert submitted[-1].kind == "review" and D._same(submitted[-1].inputs[0], alpha_raw)

            # ---- Tools › Async batch › Create batch from…: the same Books rows, one tap picks ---------
            batch = await _screen(app, "tools.async", "AsyncBatchScreen")
            apicker = await _opened(batch.open_picker())
            assert apicker.segment == "books" and [t.key for t in apicker.rows["books"]] == [t.key for t in books]
            apicker.toggle(next(t for t in apicker.rows["books"] if t.key == owner.key))
            assert not apicker.sheet.open and batch.target is not None and D._same(batch.target.source, owner_raw)
            batch.ctx.submit = submit
            await batch.start("submit")
            spec = submitted[-1]
            assert spec.kind == "async_batch" and spec.params["action"] == "submit"
            assert len(spec.inputs) == 1 and D._same(spec.inputs[0], owner_raw)

            # ---- the chat's ＋ › From Library stays the Library alone -----------------------------------
            only = SourcePicker(app.tools.context(), title="Attach from Library", segment="library",
                                segments=("library",), find_folder=False)
            await only.load()
            assert {_norm(t.folder) for t in only.rows["library"]} == {_norm(alpha_ws), _norm(gamma_ws)}
            assert only.rows["books"] == [] and only.rows["chat"] == []
        finally:
            await D._stop_app(tf, app)

    D._run_scenario(scenario)
