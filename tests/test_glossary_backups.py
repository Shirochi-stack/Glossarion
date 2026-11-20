"""Glossary Editor backups: manual and automatic kept apart, each with its own limit, and Restore Backup.

* Core (``glossary_files`` / ``glossary_document``, no Qt): a backup's kind is read from its file name
  (the names every version wrote, nothing renamed or moved); ``create_glossary_backup`` prunes only the
  kind it just wrote, to that kind's limit (``glossary_max_backups`` automatic, ``glossary_max_manual_backups``
  manual), oldest first by the time in the name and never by file age; a config from before the manual
  limit keeps its one number for both kinds; the restore list is newest first with kind and operation.
* Desktop (PySide6): the editor's "Restore Backup" button sits above "About Format" and restores a manual
  or an automatic backup into the open glossary (a ``before_restore`` backup first, Undo brings the
  previous entries back); the Backup Settings window is made without a Qt parent (no stylesheet rule
  reaches its spin boxes), has the Glossary Manager's background and saves both limits; the backup browser
  (``qt_widget_helpers.BackupBrowserDialog``, also the Config Backup Manager) lists, restores and deletes.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_glossary_backups.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import glossary_document as gd  # noqa: E402
import glossary_files as gf  # noqa: E402

HAS_QT = importlib.util.find_spec("PySide6") is not None
needs_qt = pytest.mark.skipif(not HAS_QT, reason="needs PySide6")

ENTRIES = [
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female", "description": "a witch"},
    {"type": "character", "raw_name": "카이", "translated_name": "Kai", "gender": "male", "description": "a knight"},
    {"type": "character", "raw_name": "세라", "translated_name": "Sera", "gender": "female", "description": ""},
    {"type": "term", "raw_name": "마나", "translated_name": "Mana", "gender": "", "description": "magic power"},
    {"type": "term", "raw_name": "은빛 숲", "translated_name": "Silver Forest", "gender": "", "description": ""},
]


class Stamps:
    """``time`` stand-in for glossary_files: one stamp per backup, one second apart, from ``start``."""

    def __init__(self, start="20260301_000000"):
        self.now = time.mktime(time.strptime(start, "%Y%m%d_%H%M%S"))

    def strftime(self, fmt):
        self.now += 1
        return time.strftime(fmt, time.localtime(self.now))


@pytest.fixture
def stamps(monkeypatch):
    clock = Stamps()
    monkeypatch.setattr(gf, "time", clock)
    return clock


def old_backup(folder, stem, operation, stamp, entries=1, mtime=None):
    """A backup as every version wrote it: ``<stem>_<operation>_<stamp>.json`` in ``Backups``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}_{operation}_{stamp}.json"
    path.write_text(json.dumps(ENTRIES[:entries], ensure_ascii=False, indent=2), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def backup(glossary, operation, config, log=None):
    return gf.create_glossary_backup(str(glossary), [dict(e) for e in ENTRIES], operation, config=config,
                                     append_log=(log.append if log is not None else lambda _m: None))


def names(folder, kind=None, stem="book_glossary"):
    out = []
    for name in sorted(os.listdir(folder)):
        parsed = gf.parse_backup_name(stem + ".csv", name)
        if kind is None or (parsed is not None and gf.backup_kind(parsed[0]) == kind):
            out.append(name)
    return out


# =============================================================================================
# core: kinds, limits, pruning
# =============================================================================================

def test_kind_is_read_from_the_file_name():
    parse = gf.parse_backup_name
    assert parse("book_glossary.csv", "book_glossary_manual_20260101_101500.json") == ("manual", "20260101_101500")
    assert parse("C:/x/book_glossary.csv", "book_glossary_before_delete_5_20240115_143052.json") == (
        "before_delete_5", "20240115_143052")
    assert parse("a.b.json", "a.b_before_save_20260101_000001.json") == ("before_save", "20260101_000001")
    # not backups of this glossary: no operation, no stamp, another extension, another glossary
    for name in ("book_glossary_20260101_101500.json", "book_glossary_manual.json", "book_glossary.csv",
                 "book_glossary_manual_20260101_101500.txt", "other_manual_20260101_101500.json",
                 "book_glossaryx_manual_20260101_101500.json", "book_glossary_notes.json"):
        assert parse("book_glossary.csv", name) is None, name
    assert gf.backup_kind("manual") == "manual"
    for operation in ("before_save", "before_delete_3", "before_restore", "delete", "anything"):
        assert gf.backup_kind(operation) == "auto"


def test_limits_default_to_50_each_and_an_old_config_keeps_its_one_number():
    limit = gf.glossary_backup_limit
    assert (limit({}, "manual"), limit({}, "before_save")) == (50, 50)
    # a config from before the manual limit existed: its number applies to both kinds
    old = {"glossary_auto_backup": True, "glossary_max_backups": 20}
    assert (limit(old, "manual"), limit(old, "before_save")) == (20, 20)
    assert (limit({"glossary_max_backups": 0}, "manual"), limit({"glossary_max_backups": 0}, "before_save")) == (0, 0)
    new = {"glossary_max_backups": 7, "glossary_max_manual_backups": 3}
    assert (limit(new, "manual"), limit(new, "before_save"), limit(new, "before_trim_2")) == (3, 7, 7)


def test_each_kind_is_held_to_its_own_limit(tmp_path, stamps):
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    config = {"glossary_max_backups": 3, "glossary_max_manual_backups": 2}
    log = []
    for index in range(5):
        assert backup(glossary, "manual", config, log) is True
        assert backup(glossary, f"before_delete_{index}", config, log) is True
    manual, automatic = names(folder, "manual"), names(folder, "auto")
    assert len(manual) == 2 and len(automatic) == 3
    # the newest of each kind are the ones kept
    assert [n.split("_before_")[1][:8] for n in automatic] == ["delete_2", "delete_3", "delete_4"]
    assert sum("Removed old backup" in line for line in log) == 5


def test_default_limits_keep_50_manual_and_50_automatic(tmp_path, stamps):
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    for _ in range(55):
        backup(glossary, "manual", {})
        backup(glossary, "before_save", {})
    assert len(names(folder, "manual")) == 50 and len(names(folder, "auto")) == 50


def test_pruning_automatic_backups_never_removes_manual_ones_and_the_reverse(tmp_path, stamps):
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    # old manual backups, older than everything else by name AND by file time
    manual = [old_backup(folder, "book_glossary", "manual", f"2024010{i}_120000", mtime=1_600_000_000 + i).name
              for i in range(1, 6)]
    config = {"glossary_max_backups": 4}  # an old config: one number, both kinds
    for index in range(30):
        backup(glossary, "before_save" if index % 2 else f"before_delete_{index}", config)
    assert names(folder, "manual") == manual  # 5 > 4, but only a manual backup prunes manual ones
    assert len(names(folder, "auto")) == 4

    # the reverse: old automatic backups survive a run of manual backups
    folder2 = tmp_path / "two" / "Backups"
    glossary2 = tmp_path / "two" / "book_glossary.csv"
    automatic = [old_backup(folder2, "book_glossary", f"before_delete_{i}", f"2024010{i}_120000",
                            mtime=1_600_000_000 + i).name for i in range(1, 6)]
    for _ in range(30):
        backup(glossary2, "manual", config)
    assert names(folder2, "auto") == sorted(automatic)
    assert len(names(folder2, "manual")) == 4


def test_pruning_goes_by_the_time_in_the_name_not_by_file_age(tmp_path, stamps):
    """A copied or synced Backups folder has new (or equal) file times: the count per kind and the
    names decide, so the newest backups by name are kept whatever their file times say."""
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    now = time.time()
    # the NEWEST names carry the OLDEST file times (far older than any age limit), the oldest names are new files
    for i in range(1, 9):
        old_backup(folder, "book_glossary", "before_save", f"2025010{i}_000000", mtime=now - i * 400 * 86400
                   if i > 4 else now)
    backup(glossary, "before_save", {"glossary_max_backups": 4})
    kept = names(folder, "auto")
    assert len(kept) == 4
    assert [n[-20:-5] for n in kept[:3]] == ["20250106_000000", "20250107_000000", "20250108_000000"]
    # under the limit nothing is removed, however old the files are
    folder2 = tmp_path / "old" / "Backups"
    for i in range(1, 9):
        old_backup(folder2, "book_glossary", "manual", f"2019010{i}_000000", mtime=1_000_000_000)
        old_backup(folder2, "book_glossary", "before_save", f"2019010{i}_000000", mtime=1_000_000_000)
    backup(tmp_path / "old" / "book_glossary.csv", "manual", {})
    backup(tmp_path / "old" / "book_glossary.csv", "before_save", {})
    assert len(os.listdir(folder2)) == 18


def test_an_old_folder_and_an_old_config_work_unchanged(tmp_path, stamps):
    """Before the split one limit covered both kinds, so a folder holds at most that many files. With
    the old config nothing is removed on the first backups after the update, no file is renamed or
    moved, and every old backup is in the restore list with its kind."""
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    old = []
    for i in range(1, 10):
        operation = "manual" if i % 3 == 0 else ("before_save" if i % 2 else f"before_delete_{i}")
        old.append(old_backup(folder, "book_glossary", operation, f"2025020{i}_101010", entries=i % 5 + 1))
    before = {p.name: p.read_bytes() for p in old}
    config = {"glossary_auto_backup": True, "glossary_max_backups": 9}  # no glossary_max_manual_backups
    backup(glossary, "manual", config)
    backup(glossary, "before_save", config)
    after = {name: (folder / name).read_bytes() for name in os.listdir(folder)}
    assert {name: after[name] for name in before} == before  # nothing removed, renamed or rewritten
    assert len(after) == len(before) + 2
    rows = gd.editor_backup_rows(str(glossary))
    assert len(rows) == 11
    assert {row["name"] for row in rows} == set(after)
    by_name = {row["name"]: row for row in rows}
    for path in old:
        operation, _stamp = gf.parse_backup_name("book_glossary.csv", path.name)
        assert by_name[path.name]["kind"] == ("manual" if operation == "manual" else "auto")
        assert by_name[path.name]["operation"] == operation
    assert gd.editor_backup_counts(str(glossary)) == {"manual": 4, "auto": 7}


def test_files_that_are_not_backups_are_never_pruned(tmp_path, stamps):
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    folder.mkdir()
    strays = ["book_glossary_my_copy.json", "book_glossary.json", "other_glossary_manual_20200101_000000.json",
              "book_glossary_manual_20200101_000000.txt", "book_glossary2_before_save_20200101_000000.json"]
    for name in strays:
        (folder / name).write_text("[]", encoding="utf-8")
        os.utime(folder / name, (1_000_000_000, 1_000_000_000))
    config = {"glossary_max_backups": 1, "glossary_max_manual_backups": 1}
    for _ in range(4):
        backup(glossary, "manual", config)
        backup(glossary, "before_save", config)
    left = sorted(os.listdir(folder))
    assert all(name in left for name in strays)
    assert len(left) == len(strays) + 2
    # the restore list still shows the files that carry this glossary's name, marked as not a backup
    kinds = {row["name"]: row["kind"] for row in gd.editor_backup_rows(str(glossary))}
    assert kinds["book_glossary_my_copy.json"] == "other" and kinds["book_glossary.json"] == "other"
    assert "other_glossary_manual_20200101_000000.json" not in kinds


def test_zero_keeps_every_backup(tmp_path, stamps):
    glossary = tmp_path / "book_glossary.csv"
    for _ in range(60):
        backup(glossary, "manual", {"glossary_max_backups": 0})
        backup(glossary, "before_save", {"glossary_max_backups": 5, "glossary_max_manual_backups": 0})
    assert len(names(tmp_path / "Backups", "manual")) == 60
    assert len(names(tmp_path / "Backups", "auto")) == 5


def test_automatic_backups_follow_the_setting_and_manual_ones_do_not(tmp_path, stamps):
    glossary = tmp_path / "book_glossary.csv"
    config = {"glossary_auto_backup": False}
    assert backup(glossary, "before_save", config) is True
    assert not (tmp_path / "Backups").exists()
    assert backup(glossary, "manual", config) is True
    assert len(names(tmp_path / "Backups", "manual")) == 1


def test_clean_old_backups_without_a_kind_holds_each_kind_to_the_limit(tmp_path):
    folder = tmp_path / "Backups"
    for i in range(1, 6):
        old_backup(folder, "book_glossary", "manual", f"2025010{i}_000000")
        old_backup(folder, "book_glossary", "before_save", f"2025010{i}_000000")
    gf.clean_old_backups(str(folder), "book_glossary.csv", 2, lambda _m: None)
    assert len(names(folder, "manual")) == 2 and len(names(folder, "auto")) == 2


def test_restore_list_is_newest_first_with_kind_and_operation(tmp_path):
    glossary = tmp_path / "book_glossary.csv"
    folder = tmp_path / "Backups"
    old_backup(folder, "book_glossary", "manual", "20250101_080000")
    old_backup(folder, "book_glossary", "before_delete_3", "20250103_090000", mtime=1_000_000_000)
    old_backup(folder, "book_glossary", "before_save", "20250102_070000")
    rows = gd.editor_backup_rows(str(glossary))
    assert [(row["kind"], row["operation"]) for row in rows] == [
        ("auto", "before_delete_3"), ("auto", "before_save"), ("manual", "manual")]
    # the moment in the name, not the file's time
    assert time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rows[0]["time"])) == "2025-01-03 09:00:00"
    assert all(row["size"] > 0 and os.path.isfile(row["path"]) for row in rows)
    assert gd.editor_backup_rows(str(tmp_path / "none" / "x.csv")) == []
    label = gd.backup_operation_label
    assert label("manual") == "Backup Now" and label("before_save") == "Before Save"
    assert label("before_delete_3") == "Before deleting 3 entries"
    assert label("before_remove_2_dupes") == "Before removing 2 duplicates"
    assert label("before_filter_remove_7") == "Before filtering out 7 entries"
    assert label("before_restore") == "Before Restore Backup" and label("custom_op") == "custom_op"


@pytest.mark.parametrize("operation", ["manual", "before_delete_2"])
def test_document_restores_a_backup_of_either_kind(tmp_path, stamps, operation):
    """GlossaryDocument.restore_backup (the mobile editor and the desktop button share its checks)."""
    glossary = tmp_path / "book_glossary.csv"
    gd.write_token_csv([dict(e) for e in ENTRIES], str(glossary), [], ["description"])
    saved = old_backup(tmp_path / "Backups", "book_glossary", operation, "20250101_080000", entries=2)
    config = {"update_html_on_save": False}
    doc = gd.GlossaryDocument.open(
        str(glossary), config,
        backup=lambda d, op: gf.create_glossary_backup(d.path, d.current_glossary_data, op, config=d.config,
                                                       append_log=lambda _m: None))
    assert len(doc.current_glossary_data) == 5
    assert doc.restore_backup(str(saved)) == ("Success", f"Restored {saved.name}")
    reopened = gd.GlossaryDocument.open(str(glossary), config)
    assert [e["raw_name"] for e in reopened.current_glossary_data] == ["루나", "카이"]
    # the state before the restore is an automatic backup
    safety = [row for row in gd.editor_backup_rows(str(glossary)) if row["operation"] == "before_restore"]
    assert len(safety) == 1 and safety[0]["kind"] == "auto"
    assert len(json.loads(Path(safety[0]["path"]).read_text(encoding="utf-8"))) == 5


def test_a_backup_of_another_format_or_a_broken_one_is_refused(tmp_path):
    folder = tmp_path / "Backups"
    folder.mkdir()
    as_dict = folder / "g_manual_20250101_000000.json"
    as_dict.write_text(json.dumps({"entries": {"a": "b"}}), encoding="utf-8")
    as_list = folder / "g_manual_20250102_000000.json"
    as_list.write_text(json.dumps(ENTRIES[:1], ensure_ascii=False), encoding="utf-8")
    broken = folder / "g_manual_20250103_000000.json"
    broken.write_text("{not json", encoding="utf-8")
    scalar = folder / "g_manual_20250104_000000.json"
    scalar.write_text("3", encoding="utf-8")
    assert gd.load_restorable_backup(str(as_list), "token_csv") == ENTRIES[:1]
    assert gd.load_restorable_backup(str(as_list), "list") == ENTRIES[:1]
    assert gd.load_restorable_backup(str(as_dict), "dict") == {"entries": {"a": "b"}}
    for path, fmt in ((as_dict, "token_csv"), (as_list, "dict"), (broken, "list"), (scalar, "list"),
                      (scalar, "dict"), (folder / "missing.json", "list")):
        with pytest.raises(gd.GlossaryEditorError):
            gd.load_restorable_backup(str(path), fmt)


def test_changed_sources_keep_uniform_line_endings():
    for name in ("glossary_files.py", "glossary_document.py", "GlossaryManager_GUI.py", "qt_widget_helpers.py",
                 "config_backup.py", "settings_schema_data.py"):
        data = (SRC / name).read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name


# =============================================================================================
# desktop: the editor tab, Restore Backup, Backup Settings, the backup browser
# =============================================================================================

#: The Glossary Manager's rules that reach a child window: no background, restyled spin boxes.
MANAGER_SHEET = """
    QDialog { background-color: #1e1e1e; color: white; }
    QWidget { background-color: transparent; }
    QLabel { color: white; background-color: transparent; border: none; }
    QSpinBox, QDoubleSpinBox { background-color: transparent; color: white; border: 1px solid #4a5568;
        border-radius: 3px; padding: 4px; }
"""


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qt_cleanup(qapp, monkeypatch, tmp_path):
    """Process state the editor touches (environment, cwd), no save sound, and every window a test
    made deleted before the next one (windows left to the end of the session crash Linux CI at exit)."""
    from PySide6.QtCore import QCoreApplication, QEvent
    import qt_widget_helpers
    monkeypatch.setattr(qt_widget_helpers, "saved_beep", lambda: None)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    environ = dict(os.environ)
    cwd = os.getcwd()
    before = list(qapp.topLevelWidgets())
    yield
    monkeypatch.undo()
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)
    for widget in qapp.topLevelWidgets():
        if any(widget is known for known in before):
            continue
        widget.close()
        widget.deleteLater()
    for _ in range(3):
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class Editor:
    """The real editor tab of the Glossary Manager on a scratch glossary (the main window stubbed)."""

    BOOK = "Demo Book"

    def __init__(self, qapp, root, config=None, sheet=MANAGER_SHEET):
        from PySide6 import QtWidgets
        import GlossaryManager_GUI as gm

        self.qapp, self.root = qapp, root
        (root / "in").mkdir(parents=True)
        (root / "in" / f"{self.BOOK}.epub").write_bytes(b"PK")
        folder = root / "out" / "Glossary" / self.BOOK
        folder.mkdir(parents=True)
        (root / "out" / self.BOOK).mkdir()
        self.glossary = folder / f"{self.BOOK}_glossary.csv"
        gd.write_token_csv([dict(e) for e in ENTRIES], str(self.glossary), [], ["description"])
        self.backups = folder / "Backups"
        os.environ["OUTPUT_DIRECTORY"] = str(root / "out")
        self.logs = logs = []
        self.boxes = boxes = []
        values = gd.EditorOwner(dict({"update_html_on_save": False}, **(config or {})))

        def create_glossary_backup(self_, operation_name="manual"):  # TranslatorGUI.create_glossary_backup
            if operation_name != "manual" and not self_.config.get("glossary_auto_backup", True):
                return True
            if not self_.current_glossary_data or not hasattr(self_, "editor_file_entry"):
                return True
            return gf.create_glossary_backup(self_.editor_file_entry.text(), self_.current_glossary_data,
                                             operation_name, config=self_.config, append_log=self_.append_log)

        cls = type("BackupTestOwner", (gm.GlossaryManagerMixin,), {
            "create_glossary_backup": create_glossary_backup,
            "append_log": lambda self_, message: logs.append(message),
            "save_config": lambda self_, show_message=True: None,
            "_show_message": lambda self_, *a, **k: boxes.append(("message",) + tuple(map(str, a))),
        })
        owner = cls()
        owner.config = values.config
        for attr in ("custom_entry_types", "glossary_gender_noise_threshold_var", "glossary_gender_tracking_bias_var"):
            setattr(owner, attr, getattr(values, attr))
        owner.enable_parallel_extraction_var = False
        owner.extraction_workers_var = 1
        owner.selected_files = [str(root / "in" / f"{self.BOOK}.epub")]
        self.window = QtWidgets.QDialog()
        self.window.setStyleSheet(sheet)
        owner.dialog = self.window
        layout = QtWidgets.QVBoxLayout(self.window)
        self.page = QtWidgets.QWidget()
        layout.addWidget(self.page)
        owner._setup_glossary_editor_tab(self.page)
        owner._editor_auto_reload_timer.stop()
        owner._editor_auto_reload_timer.start = lambda *args: None
        self.owner = owner
        self.settle()

    def settle(self, timeout=60.0):
        owner = self.owner
        deadline = time.monotonic() + timeout
        quiet = 0
        while time.monotonic() < deadline:
            self.qapp.processEvents()
            busy = (getattr(owner, "_editor_load_token", None) is not None
                    or owner.stats_label.text().startswith("Loading glossary"))
            quiet = 0 if busy else quiet + 1
            if quiet >= 8:
                return
            time.sleep(0.003)
        raise AssertionError(f"editor did not settle: {owner.stats_label.text()!r}")

    def button(self, text, inside=None):
        from PySide6.QtWidgets import QPushButton
        for candidate in (inside or self.page).findChildren(QPushButton):
            if candidate.text() == text:
                return candidate
        raise AssertionError(f"no button {text!r}")

    def raw_names(self):
        return [entry["raw_name"] for entry in self.owner.current_glossary_data]

    def names_on_disk(self):
        return [e["raw_name"] for e in gd.GlossaryDocument.open(str(self.glossary), {}).current_glossary_data]


def patch_boxes(monkeypatch, editor, answers=None):
    """Message boxes answer from ``answers`` (default: Yes) and are recorded in ``editor.boxes``."""
    from PySide6.QtWidgets import QMessageBox
    answers = list(answers or [])

    def box_exec(box):
        answer = answers.pop(0) if answers else "yes"
        editor.boxes.append((box.windowTitle(), box.text(), answer))
        return QMessageBox.Yes if answer == "yes" else QMessageBox.No

    def static(kind):
        def show(*args, **kwargs):
            editor.boxes.append((kind,) + tuple(a for a in args if isinstance(a, str)))
            return QMessageBox.Ok
        return show

    monkeypatch.setattr(QMessageBox, "exec", box_exec)
    for kind in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(static(kind)))


def on_dialog(monkeypatch, handler):
    """``QDialog.exec`` runs ``handler(dialog)`` instead of an event loop."""
    from PySide6.QtWidgets import QDialog
    seen = []

    def fake_exec(dialog):
        seen.append(dialog.windowTitle())
        handler(dialog)
        return dialog.result()

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    return seen


@needs_qt
def test_restore_backup_button_sits_above_about_format(qapp, qt_cleanup, tmp_path):
    from PySide6.QtWidgets import QGridLayout
    editor = Editor(qapp, tmp_path / "e")
    restore, about = editor.button("Restore Backup"), editor.button("About Format")
    grid = restore.parentWidget().layout()
    assert isinstance(grid, QGridLayout) and about.parentWidget() is restore.parentWidget()
    position = {name: grid.getItemPosition(grid.indexOf(editor.button(name)))[:2]
                for name in ("Reload", "Backup Settings", "Restore Backup", "Trim Entries", "About Format")}
    assert position == {"Reload": (0, 0), "Backup Settings": (0, 3), "Restore Backup": (0, 4),
                        "Trim Entries": (1, 0), "About Format": (1, 4)}


@needs_qt
@pytest.mark.parametrize("operation", ["manual", "before_delete_3"])
def test_restore_backup_restores_either_kind_and_undo_brings_the_entries_back(
        qapp, qt_cleanup, tmp_path, monkeypatch, stamps, operation):
    from PySide6.QtWidgets import QTreeWidget
    editor = Editor(qapp, tmp_path / "e")
    stem = editor.glossary.stem
    chosen = old_backup(editor.backups, stem, operation, "20250105_101500", entries=2)
    old_backup(editor.backups, stem, "manual", "20250101_101500", entries=3)
    old_backup(editor.backups, stem, "before_save", "20250109_101500", entries=4)
    patch_boxes(monkeypatch, editor)
    listed = []

    def restore(dialog):
        tree = dialog.findChild(QTreeWidget)
        rows = [[tree.topLevelItem(i).text(c) for c in range(tree.columnCount())]
                for i in range(tree.topLevelItemCount())]
        listed.extend(rows)
        for i in range(tree.topLevelItemCount()):
            if tree.topLevelItem(i).text(3) == chosen.name:
                tree.setCurrentItem(tree.topLevelItem(i))
        editor.button("✅ Restore Selected", inside=dialog).click()

    seen = on_dialog(monkeypatch, restore)
    assert editor.raw_names() == [e["raw_name"] for e in ENTRIES]
    editor.button("Restore Backup").click()
    editor.settle()
    assert seen == ["Restore Glossary Backup"]
    # newest first, both kinds told apart, with date and operation
    assert [row[0] for row in listed] == ["2025-01-09 10:15:00", "2025-01-05 10:15:00", "2025-01-01 10:15:00"]
    assert [row[1] for row in listed] == ["Automatic", "Manual" if operation == "manual" else "Automatic", "Manual"]
    assert listed[1][2] == ("Backup Now" if operation == "manual" else "Before deleting 3 entries")
    # restored into the editor and into the file
    assert editor.raw_names() == ["루나", "카이"] == editor.names_on_disk()
    assert editor.owner.glossary_tree.topLevelItemCount() == 2
    titles = [box[0] for box in editor.boxes]
    assert titles == ["Confirm Restore", "Restore Complete"]
    assert chosen.name in editor.boxes[0][1] and "(2 entries)" in editor.boxes[0][1]
    # a backup of the state before the restore (an automatic one), the other backups untouched
    safety = [row for row in gd.editor_backup_rows(str(editor.glossary)) if row["operation"] == "before_restore"]
    assert len(safety) == 1 and safety[0]["kind"] == "auto"
    assert [e["raw_name"] for e in json.loads(Path(safety[0]["path"]).read_text(encoding="utf-8"))] == [
        e["raw_name"] for e in ENTRIES]
    assert len(os.listdir(editor.backups)) == 4 and chosen.exists()
    # Undo brings the previous entries back (in the editor and in the file), Redo restores again
    assert editor.owner._undo_btn.isEnabled() and len(editor.owner._undo_stack) == 1
    editor.owner._undo_btn.click()
    editor.settle()
    assert editor.raw_names() == [e["raw_name"] for e in ENTRIES] == editor.names_on_disk()
    editor.owner._redo_btn.click()
    editor.settle()
    assert editor.raw_names() == ["루나", "카이"] == editor.names_on_disk()


@needs_qt
def test_restore_backup_declined_refused_or_empty_changes_nothing(qapp, qt_cleanup, tmp_path, monkeypatch, stamps):
    from PySide6.QtWidgets import QTreeWidget
    editor = Editor(qapp, tmp_path / "e")
    before = editor.glossary.read_bytes()
    # no backups yet: a message, no window
    patch_boxes(monkeypatch, editor)
    seen = on_dialog(monkeypatch, lambda dialog: None)
    editor.button("Restore Backup").click()
    assert seen == [] and editor.boxes[-1][:2] == ("information", "No Backups")
    assert not editor.backups.exists()

    stem = editor.glossary.stem
    good = old_backup(editor.backups, stem, "manual", "20250105_101500", entries=2)
    wrong = editor.backups / f"{stem}_before_save_20250106_101500.json"
    wrong.write_text(json.dumps({"entries": {"a": "b"}}), encoding="utf-8")  # a dict backup for a list glossary

    def pick(name):
        def handler(dialog):
            tree = dialog.findChild(QTreeWidget)
            for i in range(tree.topLevelItemCount()):
                if tree.topLevelItem(i).text(3) == name:
                    tree.setCurrentItem(tree.topLevelItem(i))
            editor.button("✅ Restore Selected", inside=dialog).click()
        return handler

    # declined at the question
    editor.boxes.clear()
    patch_boxes(monkeypatch, editor, answers=["no"])
    on_dialog(monkeypatch, pick(good.name))
    editor.button("Restore Backup").click()
    editor.settle()
    assert [box[0] for box in editor.boxes] == ["Confirm Restore"]
    # a backup that does not fit the glossary's format
    editor.boxes.clear()
    patch_boxes(monkeypatch, editor)
    on_dialog(monkeypatch, pick(wrong.name))
    editor.button("Restore Backup").click()
    editor.settle()
    assert [box[0] for box in editor.boxes] == ["Restore Failed"]
    assert editor.glossary.read_bytes() == before and len(editor.owner._undo_stack) == 0
    assert sorted(os.listdir(editor.backups)) == sorted([good.name, wrong.name])  # no before_restore either


@needs_qt
def test_restore_with_automatic_backups_off_still_restores_and_can_be_undone(
        qapp, qt_cleanup, tmp_path, monkeypatch, stamps):
    from PySide6.QtWidgets import QTreeWidget
    editor = Editor(qapp, tmp_path / "e", config={"glossary_auto_backup": False})
    good = old_backup(editor.backups, editor.glossary.stem, "manual", "20250105_101500", entries=1)
    patch_boxes(monkeypatch, editor)
    on_dialog(monkeypatch, lambda dialog: (dialog.findChild(QTreeWidget).setCurrentItem(
        dialog.findChild(QTreeWidget).topLevelItem(0)),
        editor.button("✅ Restore Selected", inside=dialog).click()))
    editor.button("Restore Backup").click()
    editor.settle()
    assert "Automatic backups are off" in editor.boxes[0][1]
    assert editor.names_on_disk() == ["루나"] and os.listdir(editor.backups) == [good.name]
    editor.owner._undo_btn.click()
    editor.settle()
    assert editor.names_on_disk() == [e["raw_name"] for e in ENTRIES]


@needs_qt
def test_backup_browser_creates_and_deletes_backups(qapp, qt_cleanup, tmp_path, monkeypatch, stamps):
    from PySide6.QtWidgets import QTreeWidget
    editor = Editor(qapp, tmp_path / "e")
    stem = editor.glossary.stem
    old_backup(editor.backups, stem, "before_save", "20250105_101500")
    doomed = old_backup(editor.backups, stem, "manual", "20250101_101500")
    patch_boxes(monkeypatch, editor)
    counts = []
    opened = []
    import subprocess
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)  # Windows
    monkeypatch.setattr(subprocess, "Popen", lambda command, *a, **k: opened.append(command[-1]))  # others

    def handler(dialog):
        tree = dialog.findChild(QTreeWidget)
        counts.append(tree.topLevelItemCount())
        editor.button("📁 Open Folder", inside=dialog).click()  # the editor's own "Backups" folder action
        editor.button("💾 Create New Backup", inside=dialog).click()  # Backup Now: a manual backup
        counts.append(tree.topLevelItemCount())
        assert tree.topLevelItem(0).text(1) == "Manual"  # the new one is the newest
        for i in range(tree.topLevelItemCount()):
            if tree.topLevelItem(i).text(3) == doomed.name:
                tree.setCurrentItem(tree.topLevelItem(i))
        editor.button("🗑️ Delete Selected", inside=dialog).click()
        counts.append(tree.topLevelItemCount())
        editor.button("❌ Close", inside=dialog).click()

    on_dialog(monkeypatch, handler)
    editor.button("Restore Backup").click()
    assert counts == [2, 3, 2]
    assert opened == [str(editor.backups)]
    assert not doomed.exists() and gd.editor_backup_counts(str(editor.glossary)) == {"manual": 1, "auto": 1}
    assert editor.raw_names() == [e["raw_name"] for e in ENTRIES]  # nothing restored


@needs_qt
def test_backup_settings_window_has_the_managers_background_and_plain_spin_boxes(
        qapp, qt_cleanup, tmp_path, monkeypatch, stamps):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractSpinBox, QLabel, QSpinBox
    import config_store
    config_file = tmp_path / "config.json"
    monkeypatch.setitem(sys.modules, "translator_gui", types.SimpleNamespace(CONFIG_FILE=str(config_file)))
    monkeypatch.setattr(config_store, "backup_config_file", lambda *a, **k: None, raising=False)
    # an old config: one limit, no manual one
    editor = Editor(qapp, tmp_path / "e", config={"glossary_auto_backup": True, "glossary_max_backups": 20})
    old_backup(editor.backups, editor.glossary.stem, "manual", "20250101_101500")
    old_backup(editor.backups, editor.glossary.stem, "before_save", "20250102_101500")
    old_backup(editor.backups, editor.glossary.stem, "before_save", "20250103_101500")
    patch_boxes(monkeypatch, editor)
    seen = {}

    def handler(dialog):
        seen["title"] = dialog.windowTitle()
        # no Qt parent: the Glossary Manager's QSpinBox rule cannot reach the spin boxes ...
        seen["parent"] = dialog.parent()
        seen["modal"] = dialog.windowModality() == Qt.ApplicationModal
        spins = dialog.findChildren(QSpinBox)
        seen["spins"] = [(type(s) is QSpinBox, s.value(), s.minimum(), s.maximum(), s.styleSheet(),
                          s.focusPolicy() == Qt.StrongFocus) for s in spins]
        seen["spinbox_rules"] = "SpinBox" in dialog.styleSheet()
        seen["all_spins"] = len(dialog.findChildren(QAbstractSpinBox))
        # ... and the window has the Glossary Manager's background instead of none
        dialog.setGraphicsEffect(None)  # the fade-in starts at zero opacity; no event loop runs it here
        dialog.resize(600, 500)
        seen["background"] = dialog.grab().toImage().pixelColor(3, 3).name()
        seen["labels"] = [label.text() for label in dialog.findChildren(QLabel)]
        seen["buttons"] = [b.text() for b in dialog.findChildren(type(editor.button("Reload")))]
        spins[0].setValue(30)
        spins[1].setValue(5)
        editor.button("💾 Save", inside=dialog).click()
        editor.button("Backup Now", inside=dialog).click()
        seen["labels_after"] = [label.text() for label in dialog.findChildren(QLabel)]
        editor.button("Close", inside=dialog).click()

    on_dialog(monkeypatch, handler)
    editor.button("Backup Settings").click()
    assert seen["title"] == "Backup Settings" and seen["parent"] is None and seen["modal"]
    assert seen["spins"] == [(True, 20, 0, 999, "", True), (True, 20, 0, 999, "", True)]  # the old number for both
    assert seen["all_spins"] == 2 and not seen["spinbox_rules"]
    assert seen["background"] == "#1e1e1e"
    assert "Automatic backups to keep:" in seen["labels"] and "Manual backups to keep:" in seen["labels"]
    assert "This glossary: 2 automatic, 1 manual backup(s)" in seen["labels"]
    assert "This glossary: 2 automatic, 2 manual backup(s)" in seen["labels_after"]
    assert seen["buttons"] == ["💾 Save", "Backup Now", "Close"]
    # saved: in memory and in the config file, both limits
    assert editor.owner.config["glossary_max_backups"] == 30
    assert editor.owner.config["glossary_max_manual_backups"] == 5
    saved = json.loads(config_file.read_text(encoding="utf-8"))
    assert (saved["glossary_auto_backup"], saved["glossary_max_backups"], saved["glossary_max_manual_backups"]) == (
        True, 30, 5)
    assert ("information", "Success", "Manual backup created successfully!") in editor.boxes


@needs_qt
def test_backup_browser_dialog_rows_selection_and_callbacks(qapp, qt_cleanup, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from qt_widget_helpers import BackupBrowserDialog, format_backup_size
    assert [format_backup_size(n) for n in (0, 1023, 1024, 5 * 1024 * 1024)] == ["0 B", "1023 B", "1 KB", "5 MB"]
    calls = []
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: (shown.append(box.windowTitle()), QMessageBox.Yes)[1])
    rows = [{"cells": ["2026-01-02", "b.json", "2 KB"], "key": "b", "colors": {1: "#7bb3e0"}},
            {"cells": ["2026-01-01", "a.json", "1 KB"], "key": "a"}]
    dialog = BackupBrowserDialog(
        None, window_title="Backups", heading="Heading", description="Description", location="C:/x/Backups",
        columns=[("Date", 100), ("File", 200), ("Size", 60)], rows=rows,
        on_restore=lambda key: calls.append(("restore", key)),
        on_delete=lambda key: (calls.append(("delete", key)), key == "a")[1])
    assert dialog.tree.topLevelItemCount() == 2 and dialog.selected_key() == "b"  # newest first, first selected
    assert "Total Backups: 2" in dialog.info_label.text() and "C:/x/Backups" in dialog.info_label.text()
    # buttons without a callback are not offered
    assert dialog.create_button.isHidden() and dialog.folder_button.isHidden()
    assert not dialog.restore_button.isHidden() and not dialog.delete_button.isHidden()
    dialog.restore_button.click()
    dialog.delete_button.click()  # the callback keeps "b": its row stays
    assert dialog.tree.topLevelItemCount() == 2
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    dialog.delete_button.click()
    assert calls == [("restore", "b"), ("delete", "b"), ("delete", "a")]
    assert dialog.tree.topLevelItemCount() == 1 and "Total Backups: 1" in dialog.info_label.text()
    dialog.tree.setCurrentItem(None)
    dialog.restore_button.click()
    dialog.delete_button.click()
    assert shown == ["No Selection", "No Selection"] and len(calls) == 3
    assert dialog.confirm("Sure?", "Really?") is True and shown[-1] == "Sure?"
    dialog.set_rows(rows[:1])
    assert dialog.tree.topLevelItemCount() == 1 and dialog.selected_key() == "b"
