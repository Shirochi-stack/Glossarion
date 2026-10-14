"""Owner fixes: glossary approval card, glossary progress, refinement alerts and Continue reusing a book's folders.

* Approval card: ✏️ Edit opens the table editor (24), a per-type totals row (25), ✓ Yes and ■ No on one row
  (26), "Always accept" needs a second tap within 2 s (27).
* Chat card › Progress opens the Glossary Progress manager when only the glossary was extracted (11).
* The gate's "action required" alerts are hidden while its glossary is refined (12); the refinement records
  its entry types in the book's ``*_glossary_progress.json`` (14).
* Glossary Progress manager: "Continue" button, the Total label next to the title (13).
* Continue: a chat run's glossary folder is kept with the workspace and filed under ``<root>/Glossary/<book>``
  when the book moves into the Library; a Library book outside the output root is moved there instead of the
  "Output Folder Mismatch" switch (18-20).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

GLOSSARY = ("Glossary Columns: type,raw_name,translated_name,gender,description\n"
            "=== CHARACTERS ===\n* 김철수 = Kim Cheolsu\n* 이영희 = Lee Younghee\n"
            "=== TERMS ===\n* 검 = Sword\n")


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------
# 24-27: approval card
# ---------------------------------------------------------------------------------------------------------


def test_glossary_preview_counts_entries_per_type(tmp_path):
    from glossarion_mobile.ui.chat.cards import glossary_preview, type_counts_text

    path = _write(tmp_path / "Book_glossary.csv", GLOSSARY)
    info = glossary_preview(str(path))
    assert info["entries"] == 3
    assert info["types"] == [("Characters", 2), ("Terms", 1)]
    assert type_counts_text(info["types"]) == "Characters 2 · Terms 1"


@needs_flet
def test_approval_card_rows_types_and_double_tap_always(tmp_path):
    import flet as ft

    from glossarion_mobile.ui.chat.cards import (
        ALWAYS_ACCEPT_LABEL,
        ALWAYS_CONFIRM_LABEL,
        GlossaryApprovalCard,
        glossary_preview,
    )

    path = _write(tmp_path / "Book_glossary.csv", GLOSSARY)
    always: list = []
    card = GlossaryApprovalCard(path=str(path), info=glossary_preview(str(path)), on_always=lambda: always.append(1))
    rows = [c for c in card.content.controls if isinstance(c, ft.Row)]
    answer_row = next(r for r in rows if r.key == "approval-answer-row")
    assert answer_row.controls == [card.yes_button, card.no_button] and not answer_row.wrap  # 26: one row
    assert card.types_text.value == "Characters 2 · Terms 1" and card.types_text.visible  # 25
    # 27: one tap only arms; a second tap within 2 s accepts; a late second tap arms again
    assert card.always_tap(now=100.0) is False and always == [] and card.always_button.content == ALWAYS_CONFIRM_LABEL
    assert card.always_tap(now=103.0) is False and always == []
    assert card.always_tap(now=104.5) is True and always == [1] and card.always_accepted
    assert card.always_tap(now=105.0) is False and always == [1]
    fresh = GlossaryApprovalCard(path=str(path), on_always=lambda: None)
    assert fresh.always_button.content == ALWAYS_ACCEPT_LABEL


@needs_flet
def test_approval_card_edit_opens_the_table_editor(tmp_path):
    """24: the chat's ✏️ Edit goes straight to the Glossary Manager's table editor (no raw text editor)."""
    from glossarion_mobile.ui.chat.chat_view import ChatView

    path = _write(tmp_path / "Book_glossary.csv", GLOSSARY)
    opened: list = []
    overlays: list = []

    async def run_io(fn, *args):
        return fn(*args)

    view = ChatView.__new__(ChatView)
    view.env = types.SimpleNamespace(run_io=run_io, push_overlay=overlays.append, mono=None)
    view.glossary_table_opener = opened.append
    view.notify = lambda *a, **k: None
    tasks: list = []
    view._spawn = lambda coro: tasks.append(coro)
    view.open_glossary_editor(str(path))
    asyncio.run(tasks[0])
    assert opened == [str(path)] and overlays == []


# ---------------------------------------------------------------------------------------------------------
# 11: Progress -> Glossary Progress manager
# ---------------------------------------------------------------------------------------------------------


def test_progress_falls_back_to_the_glossary_workspace(tmp_path):
    from glossarion_mobile.ui.chat.integration import _glossary_workspace, _has_translation_progress

    run_root = tmp_path / "run"
    workspace = run_root / "Book"
    workspace.mkdir(parents=True)
    assert _glossary_workspace(str(run_root), str(tmp_path / "Book.epub")) == str(workspace)
    assert _glossary_workspace(str(workspace), str(tmp_path / "Book.epub")) == str(workspace)
    _write(workspace / "translation_progress.json", json.dumps({"chapters": {}, "chapter_chunks": {}}))
    assert not _has_translation_progress(str(workspace))  # written before the glossary pass: empty
    _write(workspace / "translation_progress.json", json.dumps({"chapters": {"1": {"status": "completed"}}}))
    assert _has_translation_progress(str(workspace))


def test_glossary_progress_beside_a_chat_workspace_is_found(tmp_path, monkeypatch):
    """The Book page's lookup also searches the folder holding the workspace (a chat run root / its
    Attachments folder), so a chat book's glossary progress opens in the Glossary Progress manager."""
    from glossarion_mobile.services.library import LibraryService, SharedCore
    from glossarion_mobile.ui.library import progress_model as pm

    output = tmp_path / "Output"
    output.mkdir()
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
    run_root = tmp_path / "runs" / "r1"
    workspace = run_root / "Book"
    workspace.mkdir(parents=True)
    raw = _write(tmp_path / "Book.epub", "PK")
    gp = _write(run_root / "Glossary" / "Book" / "Book_glossary_progress.json", "{}")
    service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}), config={})
    book = {"name": "Book", "output_folder": str(workspace), "raw_source_path": str(raw)}
    assert os.path.normcase(pm.find_glossary_progress_path(service, book) or "") == os.path.normcase(str(gp))
    # a book in the output root itself gets no extra root
    assert pm.workspace_root_owner(service, {"name": "B", "output_folder": str(output / "B")}) is None


# ---------------------------------------------------------------------------------------------------------
# 12 / 14: refinement at the gate
# ---------------------------------------------------------------------------------------------------------


def test_refinement_finds_the_glossary_progress_file(tmp_path):
    from glossarion_mobile.job_kinds.glossary import progress_for_glossary

    folder = tmp_path / "Output" / "Glossary" / "Book"
    glossary = _write(folder / "Book_glossary.csv", GLOSSARY)
    assert progress_for_glossary(str(glossary)) is None
    gp = _write(folder / "Book_glossary_progress.json", "{}")
    assert progress_for_glossary(str(glossary)) == str(gp)
    # a workspace's canonical glossary.csv: the book's Glossary folder beside the workspace
    canonical = _write(tmp_path / "Output" / "Book" / "glossary.csv", GLOSSARY)
    assert progress_for_glossary(str(canonical)) == str(gp)


def test_alerts_hidden_while_the_gate_glossary_is_refined():
    from glossarion_mobile.services import jobs as jobs_mod
    from glossarion_mobile.services.jobs import (
        GLOSSARY_REFINING_TEXT,
        QUESTION_SUSPENDED_EVENT,
        JobService,
        JobSnapshot,
        JobSpec,
        JobState,
        progress_line,
        strip_model_for,
    )
    from glossarion_mobile.ui.chat.cards import glossary_question

    question = {"id": "q1", "kind": "glossary_approval", "data": {}}
    spec = JobSpec(kind="translate", title="Book", origin={"type": "library", "bid": "b"})

    class Job:
        id = "j1"
        warning = True
        refining = False

        def __init__(self):
            self.question = dict(question)

        def snapshot(self):
            return JobSnapshot(id="j1", spec=spec, state=JobState.RUNNING, created=0.0, question=self.question,
                               warning=self.warning, refining=self.refining)

    events: list = []
    host = types.SimpleNamespace(_lock=threading.RLock(), _changed=lambda: None)
    host._post = lambda fn, *args: fn(*args)
    host._deliver_event = lambda job_id, kind, data: events.append(("event", kind, data["id"]))
    host._deliver_question = lambda snap, q: events.append(("question", q["id"], snap.warning))
    job = Job()
    suspended = JobService._suspend_question(host, job)
    snap = job.snapshot()
    assert suspended["id"] == "q1" and events == [("event", QUESTION_SUSPENDED_EVENT, "q1")]
    assert not strip_model_for(snap).warning and progress_line(snap) == GLOSSARY_REFINING_TEXT
    assert glossary_question(snap) is None  # the Book page's review button / sheet stay away
    JobService._resume_question(host, job, suspended)
    snap = job.snapshot()
    assert events[-1] == ("question", "q1", True)  # the alerts come back: still waiting
    assert strip_model_for(snap).warning and glossary_question(snap) is not None
    # answered meanwhile: nothing is delivered again
    events.clear()
    JobService._suspend_question(host, job)
    job.question = None
    JobService._resume_question(host, job, question)
    assert [e[0] for e in events] == ["event"]
    job.question = dict(question)
    job.refining = True
    assert jobs_mod.notification_text(job.snapshot()).endswith(": refining the glossary")


# ---------------------------------------------------------------------------------------------------------
# 18-20: Continue reuses the book's folders
# ---------------------------------------------------------------------------------------------------------


def test_run_glossary_is_kept_and_filed_with_the_book(tmp_path):
    from glossarion_mobile.services.workspace_sync import book_glossary_dir, carry_glossary, stash_run_glossary

    run_root = tmp_path / "run"
    _write(run_root / "Glossary" / "Book" / "Book_glossary_progress.json", '{"chapters": {}}')
    _write(run_root / "Glossary" / "Book" / ".cache" / "epub_text.json", "{}")
    workspace = tmp_path / "Chat" / "Attachments" / "Book"
    workspace.mkdir(parents=True)
    stash = stash_run_glossary(str(run_root), "Book", str(workspace))
    assert os.path.isfile(os.path.join(stash, "Book_glossary_progress.json"))
    assert os.path.isfile(os.path.join(stash, ".cache", "epub_text.json"))
    # the book moved into the Library's output root: its glossary joins <root>/Glossary/Book
    root = tmp_path / "Output"
    moved = root / "Book"
    os.makedirs(root)
    os.replace(workspace, moved)
    target = carry_glossary(str(moved), str(root))
    assert target == book_glossary_dir(str(root), "Book")
    assert os.path.isfile(os.path.join(target, "Book_glossary_progress.json"))
    assert os.path.isfile(os.path.join(target, ".cache", "epub_text.json"))
    assert not os.path.isdir(moved / "Glossary")  # the stash is gone once filed


def test_adopt_workspace_moves_merges_or_keeps(tmp_path, monkeypatch):
    from glossarion_mobile.services import workspace_sync as ws

    other = tmp_path / "Other"
    root = tmp_path / "Output"
    book = other / "Book"
    _write(book / "translation_progress.json", json.dumps({"chapters": {"1": {"content_hash": "h1"}}}))
    _write(book / "chapters_full.json", "[]")
    _write(other / "Glossary" / "Book" / "Book_glossary_progress.json", "{}")
    out = ws.adopt_workspace(str(book), str(root))
    assert out["status"] == "moved" and not book.exists()
    assert (root / "Book" / "chapters_full.json").is_file()  # the extraction cache came along
    assert (root / "Glossary" / "Book" / "Book_glossary_progress.json").is_file()
    assert ws.adopt_workspace(str(root / "Book"), str(root))["status"] == "same"
    # the root already holds a folder of that name: merged only for the same book
    second = other / "Book"
    _write(second / "translation_progress.json", "{}")
    monkeypatch.setattr(ws, "same_book", lambda target, folder, source="": False)
    assert ws.adopt_workspace(str(second), str(root))["status"] == "kept" and second.exists()
    monkeypatch.setattr(ws, "same_book", lambda target, folder, source="": True)
    _write(second / "extra.html", "<p/>")
    assert ws.adopt_workspace(str(second), str(root))["status"] == "merged" and not second.exists()
    assert (root / "Book" / "extra.html").is_file()


def test_library_continue_adopts_the_workspace_instead_of_asking(tmp_path, monkeypatch):
    from glossarion_mobile.services.library import LibraryService, SharedCore

    root = tmp_path / "Output"
    root.mkdir()
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(root))
    elsewhere = tmp_path / "Other" / "Book"
    _write(elsewhere / "translation_progress.json", "{}")
    attachment = tmp_path / "Chat" / "Attachments" / "Novel"
    _write(attachment / "translation_progress.json", "{}")
    _write(root / "Here" / "translation_progress.json", "{}")
    service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}), config={})
    result = service.adopt_workspaces_blocking([
        {"name": "Book", "output_folder": str(elsewhere)},
        {"name": "Novel", "output_folder": str(attachment)},
        {"name": "Here", "output_folder": str(root / "Here")},  # already in the Library's root
    ])
    assert result["moved"] == [(str(elsewhere), str(root / "Book"))]
    assert result["books"][0]["output_folder"] == str(root / "Book")
    assert result["kept"] == [("Novel", "a chat attachment")] and attachment.exists()
