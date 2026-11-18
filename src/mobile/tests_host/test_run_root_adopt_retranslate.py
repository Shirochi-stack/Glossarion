"""Owner reports (2026-11): "Output Folder Mismatch" after Stop + Start, and retranslated chapters that came back.

1. After Stop on a translation and Start again, the app asked "Output Folder Mismatch" (current ``.../Output``,
   "will switch to" ``.../direct_text_runs/glossarion_input_output_XXXX``): a Library row named the workspace a
   stopped chat run left in its run root. The owner: "Auto resolve this. Don't disturb the user." Now the run never
   asks, never points the process at a run root, and the run-root workspace joins the Library's output folder
   (moved, or merged with a real progress merge, its glossary carried); what cannot move stays, and the run goes on
   in the Library's folder.
2. After Retranslate in the Progress manager, Run reused the old translation: a merge with another copy of the
   book (the chat's, a stopped run's) preferred the copy that "got further". Now the retranslate job marks the rows
   it reset and every merge keeps that reset over an older copy (its old file and chunk results do not come back);
   rows an interrupted run left still lose to a finished copy.
3. The Book page's Progress view follows the book's workspace when it moved, instead of refreshing the old folder.

Synthetic fixtures only (no book content, no network).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.services import workspace_sync as ws  # noqa: E402
from glossarion_mobile.services.library import LibraryService, SharedCore  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

RUN_ROOT_NAME = "glossarion_input_output_abc12345"


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _row(status: str, source: str, output: str, content_hash: str, updated: float = 100.0, **extra) -> dict:
    return {"status": status, "original_basename": source, "output_file": output, "content_hash": content_hash,
            "last_updated": updated, **extra}


def _workspace(folder: Path, rows: dict, *, raw: Path | None = None, chunks: dict | None = None,
               label: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        _write(folder / "source_epub.txt", str(raw))
    _write(folder / "translation_progress.json", json.dumps({"chapters": rows, "chapter_chunks": chunks or {}}))
    for entry in rows.values():
        if entry.get("output_file") and str(entry.get("status") or "").startswith("completed"):
            _write(folder / entry["output_file"], f"{label or folder.name}:{entry['original_basename']}")
    return folder


def _progress(folder: Path) -> dict:
    return json.loads((folder / "translation_progress.json").read_text(encoding="utf-8"))


def _by_source(folder: Path) -> dict:
    return {e["original_basename"]: e for e in _progress(folder)["chapters"].values() if e.get("original_basename")}


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


class _Env:
    """The app's folders: ``<data>/Output`` (the Library's output folder) and a stopped chat run's run root."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.data = tmp_path / "data"
        self.output = self.data / "Output"
        self.output.mkdir(parents=True)
        self.run_root = self.data / ws.RUN_ROOTS_DIR / RUN_ROOT_NAME
        self.raw = _write(tmp_path / "Inbox" / "Book.epub", "PK the book")
        monkeypatch.setenv("OUTPUT_DIRECTORY", str(self.output))
        self.service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}), config={},
                                      paths=types.SimpleNamespace(output=str(self.output), data=str(self.data)))
        self.moved: list = []
        self.service.workspace_moved.append(lambda old, new, merged: self.moved.append((old, new, merged)))

    def stopped_chat_run(self) -> Path:
        """What a stopped chat run leaves: ``<run root>/Book`` (2 chapters) + ``<run root>/Glossary/Book``."""
        book = _workspace(self.run_root / "Book", {
            "k1": _row("completed", "c1.xhtml", "response_1.html", "h1"),
            "k2": _row("completed", "c2.xhtml", "response_2.html", "h2"),
        }, raw=self.raw, label="run")
        _write(self.run_root / "Glossary" / "Book" / "Book_glossary_progress.json", '{"from": "run"}')
        return book

    def row(self, folder: Path) -> dict:
        return {"name": "Book", "output_folder": str(folder), "type": "in_progress", "is_in_progress": True,
                "raw_source_path": str(self.raw)}


def _confirm(env: _Env, rows: list, monkeypatch) -> tuple:
    """``confirm_output_root`` as the TranslateSheet's Start calls it, with the shared shelf's mismatch check
    simulated (a row outside the Library's output folder is a mismatch whose switch names that folder's root)."""
    from glossarion_mobile.ui.library.translate_sheet import confirm_output_root
    from glossarion_mobile.ui.tools import common

    asked: list = []
    applied: list = []

    async def ask(*args, **kwargs):
        asked.append(args)
        return "yes"

    monkeypatch.setattr(common, "ask", ask)

    def mismatch(books):
        for book in books:
            parent = os.path.dirname(str(book.get("output_folder") or ""))
            if parent and not _same(parent, env.output):
                return {"prompt": "The selected translation is saved under a different folder …",
                        "new_override": parent}
        return None

    env.service.output_root_mismatch_blocking = mismatch
    env.service.apply_output_override_blocking = lambda root: applied.append(root) or root
    env.said = []
    ctx = types.SimpleNamespace(service=env.service, io=env.service.io, say=lambda *a, **k: env.said.append(a[0]),
                                extras={})
    result = asyncio.run(confirm_output_root(ctx, rows))
    return result, asked, applied


# ---- 1: Stop + Start: the run-root workspace joins the Library, no question ---------------------------------------


@needs_flet
@pytest.mark.parametrize("library_copy", ["none", "without_pointer"])
def test_start_brings_a_stopped_chat_runs_workspace_into_the_library_without_asking(tmp_path, monkeypatch,
                                                                                    library_copy):
    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    # the stopped Library job got as far as its glossary stage: Output/Glossary/Book, newer than the run's
    _write(env.output / "Glossary" / "Book" / "Book_glossary.csv", "type,raw_name,translated_name\n")
    if library_copy == "without_pointer":
        # an earlier attempt's folder without source_epub.txt (the pipeline writes it only in its translation stage)
        _workspace(env.output / "Book", {
            "k1": _row("completed", "c1.xhtml", "response_1.html", "h1"),
            "k3": _row("completed", "c3.xhtml", "response_3.html", "h3"),
        }, label="library")
    row = env.row(folder)
    bid = env.service.bid_for(row)  # the Book page / job origin of the row the scan made

    result, asked, applied = _confirm(env, [row], monkeypatch)

    assert result is True and asked == [] and applied == []
    assert os.environ["OUTPUT_DIRECTORY"] == str(env.output)  # never a run root
    target = env.output / "Book"
    assert not folder.exists() and target.is_dir()
    rows = _by_source(target)
    assert rows["c1.xhtml"]["status"] == rows["c2.xhtml"]["status"] == "completed"
    assert (target / "response_2.html").read_text(encoding="utf-8") == "run:c2.xhtml"
    if library_copy == "without_pointer":
        # a real progress merge: the Library's own rows and texts stay, the run's finished chapter comes over
        assert rows["c3.xhtml"]["status"] == "completed"
        assert (target / "response_1.html").read_text(encoding="utf-8") == "library:c1.xhtml"
    # the glossary came along with its workspace, next to the Library job's
    assert json.loads((env.output / "Glossary" / "Book" / "Book_glossary_progress.json").read_text(
        encoding="utf-8")) == {"from": "run"}
    assert (env.output / "Glossary" / "Book" / "Book_glossary.csv").is_file()
    # the row and its id now name the Library's workspace; the chat was told so its cards follow
    assert _same(env.service.book_for_bid(bid)["output_folder"], target)
    assert [(o, n) for o, n, _m in env.moved] == [(str(folder), str(target))]
    assert env.moved[0][2] is (library_copy == "without_pointer")


@needs_flet
def test_a_different_book_of_that_name_stays_and_the_run_does_not_start(tmp_path, monkeypatch):
    """Review finding: the run would write into the other book's folder (``OUTPUT_DIRECTORY/<stem>``), overwriting
    its chapters and its source pointer. It does not start; a short message says why."""
    from glossarion_mobile.ui.library.translate_sheet import NAME_TAKEN

    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    other_raw = _write(tmp_path / "Elsewhere" / "Book.epub", "PK another book")
    _workspace(env.output / "Book", {"x1": _row("completed", "x1.xhtml", "response_x1.html", "other-1")},
               raw=other_raw, label="other")
    row = env.row(folder)
    bid = env.service.bid_for(row)

    result, asked, applied = _confirm(env, [row], monkeypatch)

    assert result is False and asked == [] and applied == []
    assert env.said == [NAME_TAKEN.format(name="Book")]
    assert os.environ["OUTPUT_DIRECTORY"] == str(env.output)
    assert (folder / "translation_progress.json").is_file() and (folder / "response_1.html").is_file()  # untouched
    assert set(_progress(env.output / "Book")["chapters"]) == {"x1"}  # the other book is left as it was
    # the row is not pointed at the other book's folder
    assert _same(env.service.book_for_bid(bid)["output_folder"], folder)
    assert env.moved == []


def test_a_gone_run_root_row_never_stands_for_another_book_of_that_name(tmp_path, monkeypatch):
    """Review finding: a cached row of a chat run root that is gone resolves to ``<Output>/<name>`` only when that
    folder is the same book (its raw pointer), never another book's workspace."""
    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    bid = env.service.bid_for(env.row(folder))
    other_raw = _write(tmp_path / "Elsewhere" / "Book.epub", "PK another book")
    _workspace(env.output / "Book", {"x1": _row("completed", "x1.xhtml", "response_x1.html", "other-1")},
               raw=other_raw, label="other")
    shutil.rmtree(env.run_root)  # the chat run finished elsewhere and its run root was deleted
    assert _same(env.service.book_for_bid(bid)["output_folder"], folder)
    assert env.service.workspace_for(env.row(folder)) == ""
    _write(env.output / "Book" / "source_epub.txt", str(env.raw))  # the same book: it stands for it
    assert _same(env.service.book_for_bid(bid)["output_folder"], env.output / "Book")


def test_a_merge_keeps_the_librarys_glossary(tmp_path, monkeypatch):
    """Review finding: a stopped chat run's newer but shorter glossary does not replace the glossary of the book the
    Library already had; the run only adds the files the Library lacks."""
    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    _write(env.run_root / "Glossary" / "Book" / "Book_glossary.csv", "run: 10 entries")
    library = _workspace(env.output / "Book", {"k1": _row("completed", "c1.xhtml", "response_1.html", "h1")},
                         raw=env.raw, label="library")
    csv = _write(env.output / "Glossary" / "Book" / "Book_glossary.csv", "library: 300 entries")
    os.utime(csv, (1_000_000, 1_000_000))  # older than the run's copy
    out = ws.adopt_workspace(str(folder), str(env.output), str(env.raw))
    assert out["status"] == "merged" and library.is_dir()
    assert csv.read_text(encoding="utf-8") == "library: 300 entries"
    assert json.loads((env.output / "Glossary" / "Book" / "Book_glossary_progress.json").read_text(
        encoding="utf-8")) == {"from": "run"}  # what the Library lacked came along


def test_no_run_root_workspace_moves_while_a_job_is_active(tmp_path, monkeypatch):
    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    env.service.jobs = types.SimpleNamespace(view=lambda: types.SimpleNamespace(active=object()))
    result = env.service.adopt_workspaces_blocking([env.row(folder)])
    assert result["moved"] == [] and folder.is_dir() and not (env.output / "Book").exists()
    assert result["kept"] and "job" in result["kept"][0][1]
    assert _same(result["books"][0]["output_folder"], env.output / "Book")  # the run never goes to the run root


def test_output_roots_and_bids_leave_a_chat_run_root_out(tmp_path, monkeypatch):
    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(env.run_root))  # what a running chat job sets
    assert env.service.output_roots() == [str(env.output)]
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Another"))
    assert env.service.output_roots() == [str(env.output), str(tmp_path / "Another")]
    assert ws.run_root_of(folder / "sub") == str(env.run_root) and ws.run_root_of(env.output / "Book") == ""
    # a row the scan made while the chat job ran: while its run-root copy is on disk it stays (the chat's Resume /
    # Progress of that run); once the copy moved into the Library, its id resolves to the Library's workspace
    bid = env.service.bid_for(env.row(folder))
    assert _same(env.service.book_for_bid(bid)["output_folder"], folder)
    _workspace(env.output / "Book", {"k1": _row("completed", "c1.xhtml", "response_1.html", "h1")})
    assert _same(env.service.book_for_bid(bid)["output_folder"], folder)
    shutil.rmtree(folder)
    assert _same(env.service.book_for_bid(bid)["output_folder"], env.output / "Book")
    assert _same(env.service.workspace_for(env.row(folder)), env.output / "Book")


# ---- 2: a Retranslate reset survives every merge ------------------------------------------------------------------


def _retranslate(folder: Path, key: str, monkeypatch, plan=None) -> None:
    """The mobile retranslate job over ``folder``, with the shared reset reduced to what it does to one chapter row
    (status pending, its file deleted, its chunk results reset)."""
    from glossarion_mobile.job_kinds import retranslate

    progress_file = folder / "translation_progress.json"

    def apply_retranslation(book, plan, linked_choice=None):
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        entry = data["chapters"][key]
        if (folder / entry["output_file"]).is_file():
            os.remove(folder / entry["output_file"])
        entry["status"] = "pending"
        chunk = data.get("chapter_chunks", {}).get(entry["content_hash"])
        if isinstance(chunk, dict):
            chunk["completed"] = []
        progress_file.write_text(json.dumps(data), encoding="utf-8")
        return types.SimpleNamespace(as_dict=lambda: {"status_reset": 1})

    fake = types.SimpleNamespace(apply_retranslation=apply_retranslation,
                                 retranslation_result_message=lambda result: ("info", "Success", "reset"))
    monkeypatch.setattr(retranslate, "_progress_actions", lambda: fake)
    book = types.SimpleNamespace(progress_file=str(progress_file), output_dir=str(folder),
                                 data={"progress_file": str(progress_file), "output_dir": str(folder)})
    token = retranslate.stash(book, plan or types.SimpleNamespace(count=1))
    results: dict = {}
    ctx = types.SimpleNamespace(params={"plan": token}, phase=lambda *a: None, log=lambda *a: None,
                                set_result=lambda **kw: results.update(kw))
    assert retranslate.run(ctx)["ok"]


def _book_rows(label: str) -> dict:
    return {
        "k1": _row("completed", "c1.xhtml", "response_1.html", "h1"),
        "k2": _row("completed", "c2.xhtml", "response_2.html", "h2"),
        "k3": _row("in_progress" if label == "chat" else "completed", "c3.xhtml", "response_3.html", "h3"),
    }


def test_the_retranslate_job_marks_its_rows_and_the_chat_migrate_keeps_them(tmp_path, monkeypatch):
    chunks = {"h2": {"total_chunks": 2, "completed": [0, 1]}}
    library = _workspace(tmp_path / "Output" / "Book", _book_rows("library"), chunks=chunks, label="library")
    chat = _workspace(tmp_path / "Chat" / "Attachments" / "Book", _book_rows("chat"), chunks=dict(chunks), label="chat")
    _write(chat / "response_3.html", "chat:partial")

    _retranslate(library, "k2", monkeypatch)
    rows = _progress(library)["chapters"]
    assert rows["k2"]["status"] == "pending" and rows["k2"][ws.RESET_MARK] == rows["k2"]["last_updated"]
    assert ws.RESET_MARK not in rows["k1"] and ws.RESET_MARK not in rows["k3"]

    # the chat's Migrate: prepare, copy the chat's tree over the Library folder, keep the book's state
    state = ws.prepare_merge(str(chat), str(library))
    shutil.copytree(chat, library, dirs_exist_ok=True)
    ws.keep_book_state(str(library), state)

    merged = _progress(library)
    rows = _by_source(library)
    assert rows["c2.xhtml"]["status"] == "pending"  # the reset stands: the next run translates it again
    assert not (library / "response_2.html").exists()  # the old text did not come back
    assert merged["chapter_chunks"]["h2"]["completed"] == []  # nor its chunk results
    assert rows["c1.xhtml"]["status"] == rows["c3.xhtml"]["status"] == "completed"
    assert (library / "response_3.html").read_text(encoding="utf-8") == "library:c3.xhtml"


@pytest.mark.parametrize("reset_in", ["target", "folder"])
def test_a_join_keeps_a_retranslate_reset_from_either_folder(tmp_path, reset_in):
    marked = _row("pending", "c2.xhtml", "response_2.html", "h2", updated=500.0, **{ws.RESET_MARK: 500.0})
    reset_rows = dict(_book_rows("library"), k2=marked)
    old_rows = _book_rows("library")  # finished at 100.0, before the reset
    target = _workspace(tmp_path / "out" / "Book", reset_rows if reset_in == "target" else old_rows, label="target",
                        chunks={"h2": {"completed": [] if reset_in == "target" else [0]}})
    folder = _workspace(tmp_path / "out" / "Book2", old_rows if reset_in == "target" else reset_rows, label="folder",
                        chunks={"h2": {"completed": [0] if reset_in == "target" else []}})
    assert ws.join_workspace(str(folder), str(target)) == "merged"
    rows = _by_source(target)
    assert rows["c2.xhtml"]["status"] == "pending" and not (target / "response_2.html").exists()
    assert _progress(target)["chapter_chunks"]["h2"]["completed"] == []
    assert rows["c1.xhtml"]["status"] == rows["c3.xhtml"]["status"] == "completed"


def test_adopting_an_older_copy_keeps_the_librarys_reset(tmp_path, monkeypatch):
    env = _Env(tmp_path, monkeypatch)
    folder = env.stopped_chat_run()
    library = _workspace(env.output / "Book", {
        "k1": _row("completed", "c1.xhtml", "response_1.html", "h1"),
        "k2": _row("pending", "c2.xhtml", "response_2.html", "h2", updated=500.0, **{ws.RESET_MARK: 500.0}),
    }, raw=env.raw, label="library")
    out = ws.adopt_workspace(str(folder), str(env.output), str(env.raw))
    assert out["status"] == "merged" and not folder.exists()
    rows = _by_source(library)
    assert rows["c2.xhtml"]["status"] == "pending" and not (library / "response_2.html").exists()
    assert (library / "response_1.html").read_text(encoding="utf-8") == "library:c1.xhtml"


def test_interrupted_rows_and_older_resets_still_lose_to_a_finished_copy():
    own = {"chapters": {
        "k1": _row("in_progress", "c1.xhtml", "response_1.html", "h1", updated=900.0),  # interrupted, no mark
        "k2": _row("pending", "c2.xhtml", "response_2.html", "h2", updated=200.0, **{ws.RESET_MARK: 200.0}),
        "k3": _row("pending", "c3.xhtml", "response_3.html", "h3", updated=900.0),  # plain pending
    }, "chapter_chunks": {}}
    other = {"chapters": {
        "o1": _row("completed", "c1.xhtml", "response_1.html", "h1", updated=100.0),
        "o2": _row("completed", "c2.xhtml", "response_2.html", "h2", updated=300.0),  # translated after the reset
        "o3": _row("completed", "c3.xhtml", "response_3.html", "h3", updated=100.0),
    }}
    merged, taken = ws.merge_progress(own, other)
    statuses = {e["original_basename"]: e["status"] for e in merged["chapters"].values()}
    assert statuses == {"c1.xhtml": "completed", "c2.xhtml": "completed", "c3.xhtml": "completed"}
    assert len(taken) == 3
    # the other way round, a reset newer than the finished copy wins, an unmarked pending row still loses
    merged, _taken = ws.merge_progress(other, {"chapters": {
        "r2": _row("pending", "c2.xhtml", "response_2.html", "h2", updated=400.0, **{ws.RESET_MARK: 400.0}),
        "r3": _row("pending", "c3.xhtml", "response_3.html", "h3", updated=900.0),
    }})
    statuses = {e["original_basename"]: e["status"] for e in merged["chapters"].values()}
    assert statuses == {"c1.xhtml": "completed", "c2.xhtml": "pending", "c3.xhtml": "completed"}
    assert ws.reset_outputs(merged) == ["response_2.html"]


def test_a_row_retranslated_while_pending_is_marked_too(tmp_path, monkeypatch):
    """Review finding: the plan's rows are marked even when they were pending before (an interrupted run), so a
    stopped run's older finished copy does not come back over the Retranslate."""
    rows = dict(_book_rows("library"), k2=_row("pending", "c2.xhtml", "response_2.html", "h2"))
    library = _workspace(tmp_path / "Output" / "Book", rows, label="library")
    _write(library / "response_2.html", "library:partial")
    plan = types.SimpleNamespace(count=1, selected_chapters=[{"progress_key": "k2", "output_file": "response_2.html"}])
    _retranslate(library, "k2", monkeypatch, plan)
    assert ws.RESET_MARK in _progress(library)["chapters"]["k2"]
    stopped = _workspace(tmp_path / "runs" / "Book2", _book_rows("library"), label="run")
    assert ws.join_workspace(str(stopped), str(library)) == "merged"
    assert _by_source(library)["c2.xhtml"]["status"] == "pending" and not (library / "response_2.html").exists()


def test_the_plan_marks_a_chunk_reset_as_partial(tmp_path):
    progress = tmp_path / "translation_progress.json"
    before = {"chapters": {"k1": _row("completed", "c1.xhtml", "response_1.html", "h1"),
                           "k2": _row("completed", "c2.xhtml", "response_2.html", "h2")}}
    after = json.loads(json.dumps(before))
    for key in ("k1", "k2"):
        after["chapters"][key]["status"] = "pending"
    _write(progress, json.dumps(after))
    assert ws.mark_reset_rows(str(progress), before, 500.0, rows={"k1"}, partial={"k2"}, since=490.0) == 2
    rows = json.loads(progress.read_text(encoding="utf-8"))["chapters"]
    assert rows["k1"][ws.RESET_MARK] == rows["k2"][ws.RESET_MARK] == 500.0
    assert ws.RESET_PARTIAL not in rows["k1"] and rows["k2"][ws.RESET_PARTIAL] == 490.0


@pytest.mark.parametrize("reset_in", ["target", "folder"])
def test_a_chunk_reset_keeps_the_other_chunks_through_a_join(tmp_path, reset_in):
    """Review finding: a chunk-level Retranslate keeps the chapter's other chunks in its file; a join removes only
    an older copy of the whole old chapter, never the file that reset left."""
    import time

    since = time.time() - 5.0
    mark = time.time()
    reset_row = _row("pending", "c2.xhtml", "response_2.html", "h2", updated=mark,
                     **{ws.RESET_MARK: mark, ws.RESET_PARTIAL: since})
    old_rows = _book_rows("library")  # finished at 100.0
    target = _workspace(tmp_path / "out" / "Book", dict(old_rows, k2=reset_row) if reset_in == "target" else old_rows,
                        label="target")
    folder = _workspace(tmp_path / "out" / "Book2", old_rows if reset_in == "target" else dict(old_rows, k2=reset_row),
                        label="folder")
    reset_side, old_side = (target, folder) if reset_in == "target" else (folder, target)
    _write(reset_side / "response_2.html", "chunks 1-2 and 4-8")  # what the reset left, written after `since`
    os.utime(old_side / "response_2.html", (1_000_000, 1_000_000))  # the old whole chapter
    assert ws.join_workspace(str(folder), str(target)) == "merged"
    assert _by_source(target)["c2.xhtml"]["status"] == "pending"
    assert (target / "response_2.html").read_text(encoding="utf-8") == "chunks 1-2 and 4-8"
    assert ws.reset_outputs(_progress(target)) == []  # without a folder a partial reset's file is kept


def test_the_pipeline_translates_a_marked_row_again(tmp_path):
    """The mark is an extra key in a row: the pipeline reads rows selectively, and a reset (pending) row is
    translated again, even with its old file still on disk."""
    if not (_has("TransateKRtoEN") and _has("tiktoken")):
        pytest.skip("the translation pipeline is not available here")
    try:
        from TransateKRtoEN import FileUtilities, ProgressManager
    except Exception as exc:  # pragma: no cover - a build without the pipeline's dependencies
        pytest.skip(f"the translation pipeline cannot be imported ({exc})")
    folder = tmp_path / "Book"
    folder.mkdir()
    manager = ProgressManager(str(folder))
    chapters = []
    for index in range(2):
        chapter = {"num": index + 1, "original_basename": f"c{index + 1}.xhtml", "filename": f"c{index + 1}.xhtml",
                   "content_hash": f"hash{index + 1}", "title": f"T{index + 1}"}
        output = FileUtilities.create_chapter_filename(chapter, chapter["num"])
        _write(folder / output, "old text")
        manager.update(index, chapter["num"], chapter["content_hash"], output, status="completed", chapter_obj=chapter)
        chapters.append(chapter)
    manager.save()
    before = _progress(folder)
    data = _progress(folder)
    key = next(k for k, e in data["chapters"].items() if e.get("content_hash") == "hash2")
    data["chapters"][key]["status"] = "pending"
    _write(folder / "translation_progress.json", json.dumps(data))
    assert ws.mark_reset_rows(str(folder / "translation_progress.json"), before) == 1
    again = ProgressManager(str(folder))
    assert again.check_chapter_status(1, 2, "hash2", str(folder), chapters[1])[0] is True
    assert again.check_chapter_status(0, 1, "hash1", str(folder), chapters[0])[0] is False


# ---- 3: the Book page's Progress view follows a moved workspace ---------------------------------------------------


def test_the_progress_view_is_built_again_when_the_workspace_moved(tmp_path):
    from glossarion_mobile.ui.library import progress_model as pm

    old, new = tmp_path / "runs" / "Book", tmp_path / "Output" / "Book"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    calls: list = []

    class Core:
        STATUS_GROUPS = None

        @staticmethod
        def ProgressOwner(config, save_config=None):  # noqa: N802 - the shared class name
            return types.SimpleNamespace(config=config, take_notices=lambda: [])

        @staticmethod
        def build_book_progress(source, config, **kwargs):
            calls.append(("build", kwargs.get("fixed_output_dir")))
            return None

        @staticmethod
        def refresh_book_progress(state, read_only=True, force=False):
            calls.append(("refresh", state.output_dir))
            return None

    where = {"folder": str(old)}

    class Service:
        core = types.SimpleNamespace(module=lambda name: Core if name == "progress_core" else None)

        @staticmethod
        def config_snapshot():
            return {}

        @staticmethod
        def save_owner_config(config):
            return []

        @staticmethod
        def workspace_for(book):
            return where["folder"]

        @staticmethod
        def raw_source(book):
            return str(tmp_path / "Inbox" / "Book.epub")

    book = {"name": "Book", "type": "in_progress", "is_in_progress": True}
    previous = pm.ProgressView(output_dir=str(old), state=types.SimpleNamespace(output_dir=str(old),
                                                                               owner=types.SimpleNamespace()))
    try:
        pm.load_progress_view(Service(), book, previous=previous)
    except Exception:
        pass  # the fake refresh returns no BookProgress: only which call was made matters
    assert calls == [("refresh", str(old))]  # same folder: the pinned view is refreshed
    calls.clear()
    where["folder"] = str(new)
    pm.load_progress_view(Service(), book, previous=previous)
    assert calls == [("build", str(new))]  # moved: a fresh view of the folder the next run uses
