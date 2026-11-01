"""Owner issue 1 (mobile batch 9.15.11): Progress on an old card after an app restart said "Book not found".

Every test runs the real ``Prefs`` (a real ``mobile_state.json``), the real ``LibraryService`` and the real
``ChatFeature.open_progress`` over a scratch Library, and "restarts" by building new objects over the same disk:
nothing survives but the files. A card's route is then opened on the real ``LibraryFeature.progress_tool_screen``.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_m11_progress_restart.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services.library import LibraryService  # noqa: E402
from glossarion_mobile.state.prefs import Prefs  # noqa: E402
from glossarion_mobile.ui.chat.integration import ChatFeature  # noqa: E402
from glossarion_mobile.ui.library.common import LibraryContext  # noqa: E402
from glossarion_mobile.ui.library.feature import LibraryFeature  # noqa: E402
from glossarion_mobile.ui.router import build_route, parse_route  # noqa: E402

_TB_SPEC = importlib.util.spec_from_file_location("_m11_bootstrap", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)

CHAPTERS = ("ch001.xhtml", "ch002.xhtml", "ch003.xhtml")


def _epub(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                    'container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-'
                    'package+xml"/></rootfiles></container>')
        manifest = "".join(f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
                           for i, n in enumerate(CHAPTERS))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(CHAPTERS)))
        zf.writestr("OEBPS/content.opf",
                    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                    'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Raw</dc:title></metadata>'
                    f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for name in CHAPTERS:
            zf.writestr(f"OEBPS/{name}", f"<html><body><h1>{name}</h1><p>text</p></body></html>")
    return str(path)


def _workspace(folder: Path, raw: str = "") -> Path:
    """A translated book's output folder: two finished chapters in ``translation_progress.json``."""
    folder.mkdir(parents=True, exist_ok=True)
    if raw:
        (folder / "source_epub.txt").write_text(raw, encoding="utf-8")
    chapters = {}
    for number, name in enumerate(CHAPTERS[:2], start=1):
        output = f"response_{Path(name).stem}.html"
        (folder / output).write_text(f"<html><body>{name}</body></html>", encoding="utf-8")
        chapters[str(number)] = {"actual_num": number, "status": "completed", "output_file": output,
                                 "original_basename": name, "content_hash": f"h{number}"}
    (folder / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": chapters}),
                                                      encoding="utf-8")
    return folder


class Life:
    """One run of the app: everything in memory is new, the disk is the same."""

    def __init__(self, env: types.SimpleNamespace) -> None:
        self.prefs = Prefs(env.state)
        self.prefs.load()
        paths = types.SimpleNamespace(library=env.library, output=env.output, cache=env.cache)
        self.service = LibraryService(paths=paths, config={}, prefs=self.prefs)
        self.routes: list = []
        self.notes: list = []
        self.app = types.SimpleNamespace(library=self.service, notify=lambda *a, **kw: self.notes.append(a[0]),
                                         navigate_to=self._navigate)
        self.chat = types.SimpleNamespace(app=self.app, _run_io=self._run_io)

    @staticmethod
    async def _run_io(fn, *args):
        return fn(*args)

    def _navigate(self, name, params=None, query=None, **options):
        self.routes.append((name, params, query))

    def end(self) -> None:
        self.prefs.close()

    def tap(self, folder="", source="", **options):
        """A card's Progress: the real opener; the route it navigated to."""
        bid = asyncio.run(ChatFeature.open_progress(self.chat, str(folder), str(source), **options))
        return bid, (self.routes[-1] if self.routes else None)

    def screen(self, route):
        async def build():
            return self.build_screen(route)

        return asyncio.run(build())

    def build_screen(self, route):
        """The route on the real Progress manager factory, built like the shell builds it (on a running loop)."""
        name, params, query = route
        match = parse_route(build_route(name, params, query))
        conn, session = _TB._fake_session("android")
        ctx = LibraryContext(service=self.service, page=session.page, prefs=self.prefs, platform="android",
                             navigate=lambda *a, **kw: None, notify=lambda *a, **kw: None)
        screen = LibraryFeature.progress_tool_screen(types.SimpleNamespace(), match, ctx)
        screen.get_body()
        return screen


def _missing(screen) -> bool:
    return getattr(screen.get_body(), "key", None) == "book-missing"


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    library, output = tmp_path / "Library", tmp_path / "Output"
    library.mkdir()
    output.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
    return types.SimpleNamespace(tmp=tmp_path, library=library, output=output, cache=tmp_path / "cache",
                                 state=tmp_path / "mobile_state.json")


def _first_life_bid(env, identity) -> str:
    """The id a card stored in an earlier run: the Library's row id of ``identity`` at that time."""
    life = Life(env)
    row = ({"path": str(identity), "name": Path(identity).stem} if str(identity).endswith(".epub")
           else {"output_folder": str(identity), "name": Path(identity).name})
    bid = life.service.bid_for(row)
    life.end()
    return bid


# ---- what already worked, and stays that way ------------------------------------------------------------


def test_a_response_card_opens_its_folder_after_a_restart(env):
    """Job card › Progress re-registers the card's folder on every tap: not the failing path."""
    folder = _workspace(env.output / "Book")
    first = Life(env)
    bid, _route = first.tap(folder)
    first.end()
    life = Life(env)
    new_bid, route = life.tap(folder)
    assert new_bid == bid and route == ("tools.progress", None, {"out": bid})
    assert not _missing(life.screen(route))


def test_a_stored_id_resolves_before_any_library_scan(env):
    """Candidate "the registry is not loaded yet": Prefs answers without a scan while the folder is there."""
    folder = _workspace(env.output / "Book")
    bid = _first_life_bid(env, folder)
    life = Life(env)
    assert not life.service.snapshot.all_books()  # nothing scanned in this run
    stored_bid, route = life.tap("", "", chapters=True, bid=bid)
    assert stored_bid == bid
    screen = life.screen(route)
    assert not _missing(screen) and _same(screen.workspace, folder)


def test_another_spelling_of_a_live_folder_resolves(env):
    """Candidate /data/user/0 vs /data/data: both spell one folder, so the stored path still resolves."""
    real = _workspace(env.output / "Book")
    alias_root = env.tmp / "alias"
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(env.output), str(alias_root))
        else:
            os.symlink(env.output, alias_root)
    except OSError as exc:
        pytest.skip(f"cannot make a second spelling here: {exc}")
    try:
        alias = alias_root / "Book"
        assert alias.is_dir() and not _same(alias, real)
        bid = _first_life_bid(env, alias)
        life = Life(env)
        assert not _missing(life.screen(("tools.progress", None, {"out": bid})))
    finally:
        if sys.platform == "win32":
            os.rmdir(alias_root)
        else:
            alias_root.unlink()


# ---- the stored id named a book that has moved since ---------------------------------------------------------


def test_a_book_joined_at_the_startup_sweep_opens_after_a_restart(env):
    """The owner's case. A chat's spelling of the book's folder (cut to 120 characters here; "X" for "X." on
    Android, which Windows cannot create) is what the card's id names; the next start's sweep joins it into the
    pipeline's folder (a2d9a3d8). Chats, cloud records and share links follow the move; the id does not."""
    from glossarion_mobile.services.workspace_sync import chat_spelling, join_workspace, split_workspaces

    stem = "b" * 125
    raw = _epub(env.library / "Raw" / f"{stem}.epub")
    spelled = _workspace(env.output / chat_spelling(stem), raw)
    assert spelled.name != stem
    bid = _first_life_bid(env, spelled)

    life = Life(env)  # the next start: the sweep's own calls, then the card
    moves = split_workspaces(env.output)
    assert [Path(target).name for _f, target, _r in moves] == [stem]
    assert join_workspace(*moves[0][:2])
    assert not spelled.exists() and (env.output / stem / "translation_progress.json").is_file()

    # the card, without the source it knows: the stored id alone has to lead to the book
    _opened, route = life.tap("", "", chapters=True, bid=bid)
    screen = life.screen(route)
    assert not _missing(screen)
    assert _same(screen.workspace, env.output / stem)


def test_the_trailing_dot_spelling_of_the_owners_book(env):
    """"[213524] X..epub": the pipeline's folder is "[213524] X.", the chat's "[213524] X". Only a file system that
    keeps a trailing dot can hold the pair (Android, Linux CI; not Windows)."""
    joined = env.output / "[213524] X."
    joined.mkdir()
    if "[213524] X." not in os.listdir(env.output):
        pytest.skip("this file system drops a trailing dot")
    _workspace(joined)
    old = env.output / "[213524] X"
    assert not old.exists()
    life = Life(env)
    assert life.service.moved_workspace(str(old)) == str(joined)


def test_a_move_in_this_run_did_not_say_book_not_found_and_now_opens_the_book(env):
    """Why the owner saw it only after a restart: this run's ``_by_bid`` still holds the row of a book that moved,
    so the page opened (on the gone folder, with no chapters); a new run has only the id's path, which is gone."""
    folder = _workspace(env.output / "Book")
    life = Life(env)
    bid = life.service.bid_for({"output_folder": str(folder), "name": "Book"})
    moved = env.output / "Book two"
    os.replace(folder, moved)
    assert not _missing(life.screen(("tools.progress", None, {"out": bid})))  # the row in memory
    life.end()
    assert _missing(Life(env).screen(("tools.progress", None, {"out": bid})))  # a different name: a different book

    # the same name under today's output folder: this run's Progress finds it although its row is stale
    other = env.tmp / "Elsewhere"
    book = _workspace(other / "Saga")
    life = Life(env)
    bid = life.service.bid_for({"output_folder": str(book), "name": "Saga"})
    shutil.move(str(book), str(env.output / "Saga"))
    _opened, route = life.tap("", "", chapters=True, bid=bid)
    assert _same(life.screen(route).workspace, env.output / "Saga")


def test_the_sent_to_the_library_card_opens_the_book_through_its_own_handler(env):
    """The card's handler (``ChatView._open_library_progress``) with what it has after a restart: its stored id
    and source, and the job's record, whose output folder went stale in the move."""
    from glossarion_mobile.services.workspace_sync import chat_spelling
    from glossarion_mobile.ui.chat.chat_view import ChatView

    stem = "d" * 125
    raw = _epub(env.library / "Raw" / f"{stem}.epub")
    spelled = _workspace(env.output / chat_spelling(stem), raw)
    bid = _first_life_bid(env, spelled)
    shutil.move(str(spelled), str(env.output / stem))
    life = Life(env)
    record = types.SimpleNamespace(output_dirs={"translate": str(spelled)}, output_dir=str(spelled))

    async def passthrough(value):
        return await value if asyncio.iscoroutine(value) else value

    view = types.SimpleNamespace(
        env=types.SimpleNamespace(open_progress=lambda *a, **kw: ChatFeature.open_progress(life.chat, *a, **kw),
                                  jobs=types.SimpleNamespace(snapshot_of=lambda job_id: record)),
        notify=life.notes.append, _await=passthrough)
    assert asyncio.run(ChatView._open_library_progress(view, "job1", raw, bid))
    assert life.notes == []
    assert _same(life.screen(life.routes[-1]).workspace, env.output / stem)


def test_a_stored_id_from_an_old_app_container_opens_the_book_in_the_new_one(env):
    """iOS gives an update a new container path, and a user can point the app at another output folder: the id
    names the old absolute path, the book's folder is under today's output root."""
    old = _workspace(env.tmp / "OldContainer" / "Output" / "Book")
    bid = _first_life_bid(env, old)
    shutil.rmtree(env.tmp / "OldContainer")
    _workspace(env.output / "Book")
    life = Life(env)
    screen = life.screen(("tools.progress", None, {"out": bid}))
    assert not _missing(screen)
    assert _same(screen.workspace, env.output / "Book")


def test_a_stored_id_of_a_raw_file_that_moved_finds_the_books_folder(env):
    """A Library book without a workspace at attach time: the id names its raw file, a run then wrote the
    folder, and the file was renamed for a workspace collision or moved."""
    raw = _epub(env.library / "Raw" / "Book.epub")
    bid = _first_life_bid(env, raw)
    _workspace(env.output / "Book")
    os.replace(raw, env.library / "Raw" / "Book (1).epub")
    life = Life(env)
    _opened, route = life.tap("", raw, chapters=True, bid=bid)  # the card still knows the old source path
    screen = life.screen(route)
    assert not _missing(screen)
    assert _same(screen.workspace, env.output / "Book")


def test_a_stored_id_of_a_raw_file_opens_the_run_folder_not_the_raw_chapters(env):
    """The raw file is still there, so the id resolves; but it has no workspace of its own, and the manager over
    it would list untranslated chapters while the run's progress sits in the output folder."""
    raw = _epub(env.library / "Raw" / "Book.epub")
    bid = _first_life_bid(env, raw)
    _workspace(env.output / "Book", raw)
    life = Life(env)
    _opened, route = life.tap("", raw, chapters=True, bid=bid)
    screen = life.screen(route)
    assert _same(screen.workspace, env.output / "Book")
    _opened, route = Life(env).tap("", "", chapters=True, bid=bid)  # a card with no source: the id's file names it
    assert _same(life.screen(route).workspace, env.output / "Book")


def test_a_card_whose_id_prefs_lost_still_opens_the_folder(env):
    """mobile_state.json was unreadable (Prefs starts fresh) or the id fell out of the LRU: the card's own job
    folder and source still name the book."""
    raw = _epub(env.library / "Raw" / "Book.epub")
    folder = _workspace(env.output / "Book", raw)
    bid = _first_life_bid(env, folder)
    os.remove(env.state)
    life = Life(env)
    assert life.service.book_for_bid(bid) is None
    _opened, route = life.tap(folder, raw, chapters=True, bid=bid)
    screen = life.screen(route)
    assert not _missing(screen)
    assert _same(screen.workspace, folder)


def test_a_stale_job_folder_on_a_card_without_an_id_follows_the_book(env):
    """A chat file sent to the Library has no id: its job's recorded output folder went stale in the move."""
    from glossarion_mobile.services.workspace_sync import chat_spelling

    stem = "c" * 125
    raw = _epub(env.library / "Raw" / f"{stem}.epub")
    _workspace(env.output / stem, raw)
    gone = env.output / chat_spelling(stem)
    life = Life(env)
    _opened, route = life.tap(gone, raw, chapters=True)
    assert _same(life.screen(route).workspace, env.output / stem)


def test_a_response_card_whose_folder_moved_opens_the_book(env):
    """Job card › Progress after a move that left the card's folder behind: not "No progress to show yet"."""
    _workspace(env.output / "Book")
    gone = env.tmp / "Attachments" / "Book"
    gone.parent.mkdir()
    life = Life(env)
    opened, route = life.tap(gone)
    assert opened and route[0] == "tools.progress" and life.notes == []
    assert _same(life.screen(route).workspace, env.output / "Book")


# ---- the real page over the moved book ------------------------------------------------------------------


def test_the_moved_book_lists_its_chapters_on_the_real_page(env):
    lc = pytest.importorskip("library_core")
    if not hasattr(lc, "uninstall_library_env"):
        pytest.skip("library_core (U5 API) not importable")
    raw = _epub(env.library / "Raw" / "Book.epub")
    old = _workspace(env.tmp / "OldContainer" / "Output" / "Book", raw)
    bid = _first_life_bid(env, old)
    shutil.rmtree(env.tmp / "OldContainer")
    _workspace(env.output / "Book", raw)
    life = Life(env)
    life.service.ensure_env()

    async def scenario():
        screen = life.build_screen(("tools.progress", None, {"out": bid}))
        assert screen.title == "Progress manager" and not _missing(screen)
        await screen.reload_progress()
        return [row.status for row in screen.chapters.visible if row.kind == "chapter"]

    try:
        statuses = asyncio.run(scenario())
    finally:
        lc.uninstall_library_env()
    assert statuses.count("completed") == 2, statuses


# ---- what must still say so ---------------------------------------------------------------------------------


def test_book_not_found_stays_when_the_folder_is_really_gone(env):
    folder = _workspace(env.output / "Book")
    bid = _first_life_bid(env, folder)
    shutil.rmtree(folder)
    _workspace(env.output / "Another Book")  # another book's folder is no match
    life = Life(env)
    opened, route = life.tap("", "", chapters=True, bid=bid)
    assert opened == bid and _missing(life.screen(route))


def test_a_name_alone_without_progress_does_not_claim_the_book(env):
    folder = _workspace(env.tmp / "Old" / "Book")
    bid = _first_life_bid(env, folder)
    shutil.rmtree(env.tmp / "Old")
    (env.output / "Book").mkdir()  # no translation_progress.json
    assert Life(env).service.book_for_bid(bid) is None


def test_a_library_book_before_its_run_wrote_anything_keeps_its_own_page(env):
    """Sent to the Library › Progress before the first chapter (v9.15.10): the stored id, its EPUB's chapters."""
    raw = _epub(env.library / "Raw" / "Book.epub")
    life = Life(env)
    bid = life.service.bid_for({"path": raw, "name": "Book"})
    opened, route = life.tap("", raw, chapters=True, bid=bid)
    assert opened == bid and route == ("tools.progress", None, {"out": bid})
    assert not _missing(life.screen(route))
