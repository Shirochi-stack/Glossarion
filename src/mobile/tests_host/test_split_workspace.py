"""Owner report (2026-10-31): after an update the Library showed a book twice and its Progress manager had
"lost" the progress file. Nothing was lost: "[213524] … 되었다..epub" has two workspaces.

The pipeline names a book's folder after the raw file's whole stem (``<root>/… 되었다.``); a chat names its copy
with ``direct_text_store.attachment_workspace_name`` (``Attachments/… 되었다``: trailing dots dropped, cut to 120
characters), and that copy moved into the Library under the chat's name. The sweep at the next start made the
move: the guard for a run that can still be resumed compared the folder with the stem as text and missed it. The
new card, its Reader and its Progress manager showed the old chat run's few chapters, while every run of the book
(Run translation, Resume, the Reader's live translate: all from the raw file) went on in the full folder, so the
progress strip counted a book the Reader was not showing.

Fixed: the move goes to the pipeline's folder, a merge keeps the chapters only the Library's progress file knew,
the guard knows the chat's spelling, and folders already split become one at the next sweep.

A trailing dot cannot exist on NTFS, so the tests that need one skip on Windows (they run on CI's Linux); the same
paths run on every system with the other difference between the two names, the 120-character cut.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[2]
for folder in (str(SRC), str(SRC / "mobile" / "app")):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from glossarion_mobile.services import workspace_sync as ws  # noqa: E402

DOT_NAME = "[213524] Hero World."  # the stem of "[213524] Hero World..epub"
LONG_NAME = "[TS] Alone Mecha Genius " + "x" * 110  # the chat cuts the stem to 120 characters


def _trailing_dots_exist(root: Path) -> bool:
    (root / "probe.").mkdir()
    return "probe." in os.listdir(root)


@pytest.fixture(params=["long", "dot"])
def name(request, tmp_path) -> str:
    """Each test runs with both ways a chat's spelling differs from the stem."""
    if request.param == "long":
        return LONG_NAME
    if not _trailing_dots_exist(tmp_path):
        pytest.skip("this file system drops trailing dots")
    return DOT_NAME


def _row(status: str, source: str, output: str = "", content_hash: str = "") -> dict:
    return {"status": status, "original_basename": source, "output_file": output,
            "content_hash": content_hash or f"hash-{source}"}


def _workspace(folder: Path, raw: Path, rows: dict, *, chunks: dict | None = None, title: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    (folder / "translation_progress.json").write_text(
        json.dumps({"chapters": rows, "chapter_chunks": chunks or {}, "version": "2.1"}), encoding="utf-8")
    for entry in rows.values():
        if entry.get("output_file"):
            (folder / entry["output_file"]).write_text(f"{folder.name}:{entry['original_basename']}", encoding="utf-8")
    if title:
        (folder / "metadata.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    return folder


def _raw(tmp_path: Path, stem: str) -> Path:
    inbox = tmp_path / "Inbox"
    inbox.mkdir(exist_ok=True)
    raw = inbox / f"{stem}.epub"
    raw.write_bytes(b"PK the book")
    return raw


def _progress(folder: Path) -> dict:
    return json.loads((folder / "translation_progress.json").read_text(encoding="utf-8"))


def test_the_chat_spelling_is_the_one_the_chat_uses():
    from direct_text_store import attachment_workspace_name

    assert attachment_workspace_name(DOT_NAME) == "[213524] Hero World" == ws.chat_spelling(DOT_NAME)
    assert len(ws.chat_spelling(LONG_NAME)) == 120
    assert ws.chat_spelling("...") == "Attachment"
    assert ws.named_after("[213524] hero world", DOT_NAME) and ws.named_after(DOT_NAME, DOT_NAME)
    assert not ws.named_after("[213524] Hero", DOT_NAME) and not ws.named_after("", DOT_NAME)


def test_a_split_book_becomes_one_and_keeps_the_longer_translation(tmp_path, name):
    """The owner's pair: 168 chapters in the pipeline's folder, a stopped chat run's few in the chat's."""
    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, name)
    full = _workspace(root / name, raw, {
        "metadata": {"status": "completed", "title": "Became a Succubus"},
        "k1": _row("completed", "chapter0001.xhtml", "response_1.html"),
        "k2": _row("completed", "chapter0002.xhtml", "response_2.html"),
        "k3": _row("qa_failed", "chapter0003.xhtml", "response_3.html"),
        "k4": _row("in_progress", "chapter0004.xhtml", "response_4.html"),
    }, title="Became a Succubus")
    split = _workspace(root / ws.chat_spelling(name), raw, {
        "metadata": {"status": "completed", "title": "A Fallen Necromancer"},
        "s1": _row("completed", "chapter0001.xhtml", "response_1.html", "other-1"),
        "s3": _row("completed", "chapter0003.xhtml", "response_3.html", "other-3"),
        "s4": _row("completed", "chapter0004.xhtml", "response_4.html", "other-4"),
        "s5": _row("completed", "chapter0005.xhtml", "response_5.html", "other-5"),
        "s6": _row("in_progress", "chapter0006.xhtml", "response_6.html"),
    }, chunks={"other-5": {"total_chunks": 2, "chunks": {}}}, title="A Fallen Necromancer")
    (split / "chat_only.txt").write_text("kept", encoding="utf-8")

    assert ws.split_workspaces(root) == [(str(split), str(full), str(raw))]
    assert ws.join_workspace(str(split), str(full)) == "merged"

    assert not split.exists() and ws.split_workspaces(root) == []
    chapters = _progress(full)["chapters"]
    by_source = {e.get("original_basename"): e for e in chapters.values() if e.get("original_basename")}
    assert by_source["chapter0001.xhtml"]["content_hash"] == "hash-chapter0001.xhtml"  # the book's own stays
    assert by_source["chapter0003.xhtml"]["status"] == "completed"  # QA failed there, finished here
    assert by_source["chapter0004.xhtml"]["status"] == "completed"  # only started there
    assert by_source["chapter0005.xhtml"]["status"] == "completed"  # only here
    assert "chapter0006.xhtml" not in by_source  # an interrupted run's row: nothing is running it
    assert _progress(full)["chapter_chunks"] == {"other-5": {"total_chunks": 2, "chunks": {}}}
    assert chapters["metadata"]["title"] == "Became a Succubus"
    assert json.loads((full / "metadata.json").read_text(encoding="utf-8"))["title"] == "Became a Succubus"
    # each chapter's text is the one its row belongs to
    assert (full / "response_1.html").read_text(encoding="utf-8").startswith(f"{name}:")
    assert (full / "response_3.html").read_text(encoding="utf-8").startswith(f"{ws.chat_spelling(name)}:")
    assert (full / "response_5.html").is_file() and (full / "chat_only.txt").is_file()


def test_a_lone_folder_under_the_chats_spelling_takes_the_pipelines_name(tmp_path, name):
    """No second folder yet: the next run would have started the book over in ``<root>/<stem>``."""
    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, name)
    split = _workspace(root / ws.chat_spelling(name), raw, {"k1": _row("completed", "c1.xhtml", "response_1.html")})
    target = root / name
    assert ws.split_workspaces(root) == [(str(split), str(target), str(raw))]
    assert ws.join_workspace(str(split), str(target)) == "moved"
    assert not split.exists() and _progress(target)["chapters"]["k1"]["status"] == "completed"


def test_folders_that_are_not_one_book_stay_apart(tmp_path, name):
    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, name)
    other = tmp_path / "other" / f"{name}.epub"
    other.parent.mkdir()
    other.write_bytes(b"PK another book")
    _workspace(root / name, other, {"k1": _row("completed", "c1.xhtml", "response_1.html")})
    _workspace(root / ws.chat_spelling(name), raw, {"k2": _row("completed", "c2.xhtml", "response_2.html")})
    assert ws.split_workspaces(root) == []  # the same name, other bytes
    _workspace(root / "Plain book", _raw(tmp_path, "Plain book"), {})
    assert ws.split_workspaces(root) == []  # a folder named after its raw file


def test_a_progress_file_that_cannot_be_read_stops_the_join(tmp_path, name):
    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, name)
    full = _workspace(root / name, raw, {"k1": _row("completed", "c1.xhtml", "response_1.html")})
    split = _workspace(root / ws.chat_spelling(name), raw, {"k2": _row("completed", "c2.xhtml", "response_2.html")})
    (full / "translation_progress.json").write_text("{ cut off", encoding="utf-8")
    assert ws.join_workspace(str(split), str(full)) == ""
    assert (split / "translation_progress.json").is_file() and (split / "response_2.html").is_file()


def test_a_folder_the_file_system_folds_onto_its_target_is_not_a_split(tmp_path, monkeypatch):
    """vfat / NTFS drop the trailing dot: ``X`` and ``X.`` are one folder, and "merging" would delete it."""
    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, LONG_NAME)
    folder = _workspace(root / ws.chat_spelling(LONG_NAME), raw, {})
    (root / LONG_NAME).mkdir()
    monkeypatch.setattr(os.path, "samefile", lambda a, b: True)
    assert ws.split_workspaces(root) == [] and folder.is_dir()


def test_a_merge_into_the_library_keeps_the_books_chapters_and_title(tmp_path):
    """The shared Migrate replaces files of the same name: a chat's 2 chapters replaced the book's progress file,
    its title and, with a chapter the chat only started, that chapter's text."""
    import shutil

    raw = _raw(tmp_path, "book")
    library = _workspace(tmp_path / "output" / "book", raw, {
        "k1": _row("completed", "c1.xhtml", "response_1.html"),
        "k2": _row("completed", "c2.xhtml", "response_2.html"),
        "k3": _row("completed", "c3.xhtml", "response_3.html"),
    }, title="Became a Succubus")
    chat = _workspace(tmp_path / "chat" / "book", raw, {
        "c1": _row("completed", "c1.xhtml", "response_1.html", "chat-1"),
        "c2": _row("in_progress", "c2.xhtml", "response_2.html", "chat-2"),
    }, title="A Fallen Necromancer")

    state = ws.prepare_merge(str(chat), str(library))
    assert not (chat / "response_2.html").exists()  # the partial text
    shutil.copytree(chat, library, dirs_exist_ok=True)  # what the Migrate does
    ws.keep_book_state(str(library), state)

    rows = {e["original_basename"]: e for e in _progress(library)["chapters"].values()}
    assert rows["c1.xhtml"]["content_hash"] == "chat-1"  # both finished it: the chat's text is the file there now
    assert rows["c2.xhtml"]["status"] == "completed" and rows["c3.xhtml"]["status"] == "completed"
    assert (library / "response_2.html").is_file() and (library / "response_3.html").is_file()
    assert json.loads((library / "metadata.json").read_text(encoding="utf-8"))["title"] == "Became a Succubus"
    ws.keep_book_state(str(library), None)  # no merge: nothing to do


# ---- the sweep and the move, on the chat feature's own code ----------------------------------------------------


def _migrate_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_glossarion_devfix4_split",
                                                  Path(__file__).with_name("test_devfix_issue4.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_resume_guard_knows_the_chats_spelling():
    """A stopped chat run still offers Resume: after a restart only the file name tells its folder."""
    from glossarion_mobile.ui.chat.integration import ChatFeature

    feature = SimpleNamespace(
        runs=SimpleNamespace(last_job_ending=lambda cid: ("stopped", None, "", f"/inbox/{LONG_NAME}.epub")),
        job_workspace=lambda snap: "")
    folder = f"/chats/1/Attachments/{ws.chat_spelling(LONG_NAME)}"
    assert ChatFeature._resumable_here(feature, "1", folder)
    assert not ChatFeature._resumable_here(feature, "1", "/chats/1/Attachments/Another book")


def test_a_chat_book_moves_into_the_pipelines_folder_and_a_split_pair_joins_at_the_sweep(tmp_path, monkeypatch):
    from glossarion_mobile.ui.chat import integration

    chat_folder = tmp_path / "Direct Text" / "Chat" / "Attachments"
    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, LONG_NAME)
    workspace = _workspace(chat_folder / ws.chat_spelling(LONG_NAME), raw,
                           {"k1": _row("completed", "c1.xhtml", "response_1.html")})

    moved: list = []

    class Chats:
        def session(self, cid):
            return {"id": cid}

        def binding_for(self, cid):
            return SimpleNamespace(is_managed_attachment_workspace=lambda session, folder: True)

        def messages(self, cid):
            return [("user_file", f"{LONG_NAME}.epub", str(raw))]

        def migration_target(self, cid, folder):
            return str(root / os.path.basename(os.path.normpath(folder)))

        def rename_attachment_workspace(self, cid, folder, new):
            target = os.path.join(os.path.dirname(folder), new)
            os.rename(folder, target)
            return target

        def migrate_attachment(self, cid, folder, confirm):
            target = root / os.path.basename(folder)
            os.rename(folder, target)
            moved.append(str(target))
            return {"ok": True, "notices": []}

        def relocate_workspace(self, folder, target):
            moved.append(("relocated", folder, target))

    import threading

    feature = integration.ChatFeature.__new__(integration.ChatFeature)
    feature.chats = Chats()
    feature._migrate_lock = threading.RLock()
    feature._merge_approved = set()
    feature._cloud_moves_lock = threading.Lock()
    feature._cloud_moves = []
    feature._temp_dir = None
    feature.app = SimpleNamespace(cloud_sync=None, share_links=None)
    feature.runs = SimpleNamespace(temp_dir=None, relocate_output=lambda *a: None, live_run=lambda cid: None,
                                   last_job_ending=lambda cid: ("", None, "", ""))
    feature.jobs = SimpleNamespace(jobs=None)
    monkeypatch.setattr(feature, "_job_lock", lambda: None, raising=False)
    monkeypatch.setattr(feature, "_active_job", lambda: None, raising=False)
    monkeypatch.setattr(feature, "_record_raw_blocking", lambda source: None, raising=False)
    monkeypatch.setattr(feature, "library_service", lambda: SimpleNamespace(output_roots=lambda: [str(root)]),
                        raising=False)

    outcome = feature.auto_migrate_blocking("1", str(workspace))
    assert outcome["status"] == integration.MIGRATE_MOVED
    assert moved == [str(root / LONG_NAME)] and not workspace.exists()
    assert ws.split_workspaces(root) == []

    # an earlier build's move: the chat's spelling beside the pipeline's folder
    split = _workspace(root / ws.chat_spelling(LONG_NAME), raw, {"k2": _row("completed", "c2.xhtml", "response_2.html")})
    outcomes = feature.join_split_books_blocking()
    assert [o["status"] for o in outcomes] == [integration.MIGRATE_MOVED] and outcomes[0]["reason"] == "merged"
    assert outcomes[0]["target"] == str(root / LONG_NAME) and not split.exists()
    assert ("relocated", str(split), str(root / LONG_NAME)) in moved
    rows = {e["original_basename"] for e in _progress(root / LONG_NAME)["chapters"].values()}
    assert rows == {"c1.xhtml", "c2.xhtml"}


def test_no_join_while_a_job_runs(tmp_path, monkeypatch):
    from glossarion_mobile.ui.chat import integration
    import threading

    root = tmp_path / "output"
    root.mkdir()
    raw = _raw(tmp_path, LONG_NAME)
    _workspace(root / LONG_NAME, raw, {})
    split = _workspace(root / ws.chat_spelling(LONG_NAME), raw, {})
    feature = integration.ChatFeature.__new__(integration.ChatFeature)
    feature._migrate_lock = threading.RLock()
    monkeypatch.setattr(feature, "_job_lock", lambda: None, raising=False)
    monkeypatch.setattr(feature, "_active_job", lambda: object(), raising=False)
    monkeypatch.setattr(feature, "library_service", lambda: SimpleNamespace(output_roots=lambda: [str(root)]),
                        raising=False)
    assert feature.join_split_books_blocking() == [] and split.is_dir()


# ---- the Reader's live panel ----------------------------------------------------------------------------------


def _panel():
    pytest.importorskip("flet")
    pytest.importorskip("live_stream")
    from glossarion_mobile.ui.reader import live as rl
    from glossarion_mobile.ui.reader.live_panel import LivePanel

    return LivePanel(chapter_file="chapter0004.xhtml", feed=rl.LiveFeed("chapter0004.xhtml"), clock=lambda: 100.0)


def _snap(state, **fields):
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, Progress

    spec = JobSpec(kind="single_chapter", title="Book · chapter0004.xhtml", inputs=("/raw.epub",))
    values = dict(id="j1", spec=spec, state=state, created=1.0, started=1.0, progress=Progress(total=177, completed=168,
                                                                                               failed=9))
    values.update(fields)
    return JobSnapshot(**values)


def test_the_live_panel_has_no_empty_space_until_text_streams():
    from glossarion_mobile.ui.components.sheet import sheet_body

    panel = _panel()
    holder = sheet_body(panel.sheet)
    assert not panel.content_box.visible and holder.height is None and panel.body_column.tight

    class Page:
        window = None

        def show_dialog(self, dialog):
            self.shown = dialog

    panel.show(Page(), 1000.0)  # the screen's height sets the share it takes later, not its height now
    holder = sheet_body(panel.sheet)
    assert holder.height is None and panel.full_height == 550.0 and not panel.content_box.visible

    panel.add_lines(["🚀 Starting translation", "🧠 [gpt] Thinking...", "    planning the chapter"])
    panel.render(force=True)
    assert panel.state == "waiting" and not panel.content_box.visible and holder.height is None
    assert panel.feed.side_count  # the thinking pane fills while the chapter area stays away

    panel.add_lines(["🧠 [gpt] Thinking complete", "📡 text streaming started", "<h1>Chapter 4</h1>"])
    panel.render(force=True)
    assert panel.state == "streaming" and panel.content_box.visible and holder.height == 550.0
    assert panel.body_column.expand and not panel.body_column.tight
    assert panel.status.value == "\U0001f6f0️ Translating “chapter0004.xhtml”"  # no longer "waiting for stream…"


def test_the_live_panel_says_what_the_jobs_page_says():
    from glossarion_mobile.services.jobs import JobState, progress_line

    panel = _panel()
    assert not panel.job_line.visible
    running = _snap(JobState.RUNNING)
    panel.sync_job(running)
    assert panel.job_line.visible and panel.job_line.value == "Translating • 168/177 · 9 failed"
    assert panel.job_line.value == progress_line(running, verb=True, elapsed=False)
    assert panel.status.value.endswith("waiting for stream…")

    panel.sync_job(_snap(JobState.QUEUED, started=None))
    assert panel.status.value == "\U0001f6f0️ “chapter0004.xhtml” — queued"
    panel.sync_job(running)  # it started
    assert panel.status.value.endswith("waiting for stream…")

    panel.sync_job(_snap(JobState.STOPPING))  # a Stop from the Jobs page
    assert panel.status.value.startswith("⏹") and panel.stop_button.disabled

    panel.finish("⏹ Translation stopped — incomplete output cleared.", stopped=True)
    panel.sync_job(running)  # a late event changes nothing
    assert panel.status.value.startswith("⏹ Translation stopped")
