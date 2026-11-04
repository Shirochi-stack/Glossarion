"""instance_runs: the main window's side of a multi-instance translation (desktop, Qt).

A run is split into instances when Multi API Keys is on, the enabled Translation keys carry at
least two instance numbers and at least two input files are selected
(``instance_plan.is_multi_instance``). Otherwise the Run button runs exactly as before.

The Run button's handler (``TranslatorGUI._run_translation_clicked``) calls :func:`start_run`, which

* asks which instance translates which file (:class:`InstanceAssignmentDialog`, prefilled from the
  saved ``instance_file_map``, else round-robin) and saves the answer through ``save_config``;
* records every input of the run in the Library registry once (the instances do not);
* starts instances 2..N as processes of this program (``instance_worker``), each with a config
  snapshot holding only its own keys of every key pool (never the main GUI key, which is instance
  1's), its own stop file, log and watchdog subfolder, and a reader thread that feeds its output
  into that instance's log;
* runs instance 1 in this window exactly like a normal Run (``run_translation_thread``: same thread,
  same log, same dialogs) on instance 1's files. The saved config is never narrowed.

While any instance of the run is alive this window uses instance 1's API keys only, in every key
pool (``instance_worker.restrict_keys``): instance 1's run and everything else started here
meanwhile (a review, the Manga translator, a QA scan, glossary tools ...). The QA scan an instance
asks for after its translation runs here once every instance has finished, under that instance's
own keys and config, one instance at a time.

Instance 1's run reads (and renames) ``gui.selected_files``, which holds only its own files while
it runs; every other action of this window asks :func:`user_selection` for the user's selection,
which is then every file of the run.

The main window shows one log per instance (instance 1's is ``log_text``) with an "Instance"
switcher above the log (:func:`log_area`) and an instance filter next to the API watchdog bar
(:func:`watchdog_row`, :func:`watchdog_view`, :func:`watchdog_files`); both are visible only while
the current or last run used more than one instance and follow each other. Stop reaches every
instance (:func:`stop_children`: a stop file each; a second Stop or a double click kills the
process trees), and when the last instance has finished one summary line per instance goes into
instance 1's log. Closing the window ends the instance processes (:func:`shutdown`).
"""

import collections
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import instance_plan
import instance_worker

__all__ = [
    "ALL_INSTANCES",
    "InstanceAssignmentDialog",
    "InstanceRun",
    "children_alive",
    "file_instance",
    "log_area",
    "show_single_instance",
    "shutdown",
    "start_run",
    "stop_children",
    "user_selection",
    "visible_log",
    "watchdog_files",
    "watchdog_row",
    "watchdog_view",
]

#: Watchdog filter value for "All instances".
ALL_INSTANCES = 0
#: Subfolder of the main window's watchdog folder an instance process writes its file into.
WATCHDOG_SUBDIR = "instance_"
#: Child output lines kept while the log is not drained (a flood keeps the newest).
PENDING_LINES_MAX = 50_000
#: How often the end of a run checks whether a QA scan thread is still going (ms).
QA_POLL_MS = 1000

_RESULT_BY_EXIT = {
    instance_worker.EXIT_OK: "completed",
    instance_worker.EXIT_FAILED: "failed",
    instance_worker.EXIT_STOPPED: "stopped",
}

_COMPACT_COMBO_STYLE = """
    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #4a5568;
        border-radius: 3px;
        color: white;
        font-size: 8pt;
        padding: 0px 6px;
    }
    QComboBox::drop-down { border: none; width: 28px; }
    QComboBox::down-arrow { image: none; }
"""


# ---------------------------------------------------------------------------
# Main window widgets (built by TranslatorGUI._create_log_section)
# ---------------------------------------------------------------------------


def watchdog_row(gui):
    """Row 10: the instance switcher (hidden until needed; left of the bar, away from the Stop
    button) and the API watchdog bar. The switcher filters the bar and picks the log shown."""
    row = QWidget()
    row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    combo = QComboBox()
    combo.setFixedHeight(19)
    combo.setFocusPolicy(Qt.NoFocus)
    combo.setStyleSheet(_COMPACT_COMBO_STYLE)
    combo.setToolTip("API calls of every instance, or one instance's API calls and log")
    _style_combo(gui, combo)
    combo.currentIndexChanged.connect(lambda index, g=gui: _on_watchdog_instance_changed(g, index))
    combo.hide()
    layout.addWidget(combo)
    layout.addWidget(gui.api_watchdog_bar, 1)
    gui.watchdog_instance_combo = combo
    return row


def log_area(gui):
    """Row 11: one log per instance (the watchdog row's switcher picks which one shows);
    ``gui.log_text`` is instance 1's log and the stack's first page."""
    stack = QStackedWidget()
    stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    stack.setMinimumHeight(40)
    stack.addWidget(gui.log_text)
    gui.log_stack = stack
    gui._instance_logs = {1: gui.log_text}
    return stack


def _style_combo(gui, combo):
    """The main window's combo look (no wheel scrolling, Halgakos arrow) when it has the helpers."""
    for name in ("_disable_combobox_mousewheel", "_add_combobox_arrow"):
        helper = getattr(gui, name, None)
        if callable(helper):
            try:
                helper(combo)
            except Exception:
                pass


def visible_log(gui):
    """The log widget on screen (instance 1's ``log_text`` unless another instance is shown)."""
    stack = getattr(gui, "log_stack", None)
    widget = stack.currentWidget() if stack is not None else None
    return widget if widget is not None else gui.log_text


def _log_widget(gui, number):
    """Instance *number*'s log, made like ``log_text`` the first time it is needed."""
    logs = gui._instance_logs
    widget = logs.get(number)
    if widget is None:
        base = gui.log_text
        widget = QPlainTextEdit()
        widget.setReadOnly(True)
        widget.document().setDocumentMargin(0)
        widget.setMaximumBlockCount(base.maximumBlockCount())
        widget.setMinimumHeight(40)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setFont(base.font())
        widget.setStyleSheet(base.styleSheet())
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(gui._show_context_menu)
        gui.log_stack.addWidget(widget)
        logs[number] = widget
    return widget


def _on_watchdog_instance_changed(gui, index):
    """"Instance N" shows instance N's API calls and log; "All instances" every instance's API
    calls (the log shown stays)."""
    number = gui.watchdog_instance_combo.itemData(index)
    widget = gui._instance_logs.get(int(number)) if number else None
    if widget is not None:
        gui.log_stack.setCurrentWidget(widget)
    try:
        gui._update_api_watchdog()
    except Exception:
        pass


def _prepare_views(gui, numbers):
    """The switcher listing *numbers* (instance 1, the main log, always among them); shown when
    there is more than one."""
    numbers = sorted(set(numbers) | {1})
    combo = gui.watchdog_instance_combo
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("All instances", ALL_INSTANCES)
    for number in numbers:
        combo.addItem(f"Instance {number}", number)
    combo.setCurrentIndex(0)
    combo.blockSignals(False)
    for number in numbers:
        if number != 1:
            _log_widget(gui, number).clear()
    gui.log_stack.setCurrentWidget(gui.log_text)
    combo.setVisible(len(numbers) > 1)


def show_single_instance(gui):
    """A single-instance run: back to instance 1's log, switcher hidden."""
    if children_alive(gui) or getattr(gui, "log_stack", None) is None:
        return
    gui.log_stack.setCurrentWidget(gui.log_text)
    gui.watchdog_instance_combo.hide()


def watchdog_view(gui):
    """The watchdog filter: None for every instance, else the instance number to show."""
    combo = getattr(gui, "watchdog_instance_combo", None)
    if combo is None or combo.isHidden():
        return None
    number = combo.currentData()
    return int(number) if number else None


def watchdog_files(watchdog_dir):
    """The watchdog files the main window aggregates: its own folder's (this process and its helper
    processes) and every instance process's subfolder."""
    import glob

    return (glob.glob(os.path.join(watchdog_dir, "api_watchdog_*.json"))
            + glob.glob(os.path.join(watchdog_dir, f"{WATCHDOG_SUBDIR}*", "api_watchdog_*.json")))


def file_instance(state, path):
    """The instance a watchdog file belongs to: its ``instance`` field, else the ``instance_N``
    subfolder it is in, else 1 (this window and its helper processes)."""
    value = state.get("instance") if isinstance(state, dict) else None
    if value is not None:
        return instance_plan.coerce_instance(value)
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if folder.startswith(WATCHDOG_SUBDIR):
        return instance_plan.coerce_instance(folder[len(WATCHDOG_SUBDIR):])
    return 1


# ---------------------------------------------------------------------------
# Run / Stop entry points (TranslatorGUI)
# ---------------------------------------------------------------------------


def children_alive(gui):
    """True while an instance process of the current run is alive."""
    run = getattr(gui, "_instance_run", None)
    return bool(run is not None and run.children_alive())


def user_selection(gui):
    """The input files the user selected, as an action of this window should see them.

    While instance 1 of a multi-instance run translates, ``gui.selected_files`` holds only its own
    files (its run reads and renames that list); every file of the run is the user's selection
    then, instance 1's inputs under the names its run gave them. Otherwise, and for the QA scan
    instance 1's run asks for, ``gui.selected_files`` itself."""
    current = list(getattr(gui, "selected_files", None) or [])
    run = getattr(gui, "_instance_run", None)
    if run is None or run.narrowed is None or run.own_scan:
        return current
    selection, own = run.narrowed
    return _full_selection(selection, own, current)


def _full_selection(selection, own, current):
    """*selection* (the user's, at Run) with instance 1's files (*own*) under their current names
    (*current*: instance 1's list now, renamed in place by its run, e.g. an archive converted to an
    EPUB). A list of another length means the user picked new files meanwhile: that one."""
    if len(current) != len(own):
        return current
    renamed = dict(zip(own, current))
    return [renamed.get(path, path) for path in selection]


def stop_children(gui, *, graceful, force=False):
    """The Stop button for the instance processes: a stop file each (graceful or immediate, as
    the in-process Stop); a force stop or a second Stop ends their process trees."""
    run = getattr(gui, "_instance_run", None)
    if run is not None:
        run.stop(graceful=graceful, force=force)


def shutdown(gui):
    """Window closing: end every instance process tree now."""
    run = getattr(gui, "_instance_run", None)
    if run is not None:
        run.kill_children()
        run.release_scopes()
        shutil.rmtree(run.run_dir, ignore_errors=True)


def _thread_alive(gui, thread_name, future_name):
    thread = getattr(gui, thread_name, None)
    future = getattr(gui, future_name, None)
    return bool((thread is not None and thread.is_alive())
                or (future is not None and not future.done()))


def start_run(gui):
    """The Run button's multi-instance branch.

    Returns False when this is not a multi-instance run (the caller runs
    ``run_translation_thread`` as before, which also reports why a run cannot start), True when it
    handled the click: a multi-instance run started, or the user cancelled it.
    """
    previous = getattr(gui, "_instance_run", None)
    if previous is not None and previous.finished and not _thread_alive(gui, "qa_thread", "qa_future"):
        previous.release_scopes()  # a new Run ends the last run's QA turns that have not started
    files = [str(path) for path in (getattr(gui, "selected_files", None) or []) if path]
    config = getattr(gui, "config", None)
    if not isinstance(config, dict) or not instance_plan.is_multi_instance(config, files):
        show_single_instance(gui)
        return False
    if (children_alive(gui) or _thread_alive(gui, "translation_thread", "translation_future")
            or _thread_alive(gui, "glossary_thread", "glossary_future")):
        return False
    # runs prepared by another entry point keep their own single-instance path
    if (getattr(gui, "_metadata_only_run", False) or getattr(gui, "_input_output_run_active", False)
            or getattr(gui, "_single_chapter_filter", None) or getattr(gui, "_single_qa_resolution_request", None)):
        return False
    try:
        if gui._parallel_epub_pair_is_selected():
            return False
    except Exception:
        pass
    # run_translation_thread turns a click right after a graceful stop into a force stop
    if (getattr(gui, "_last_stop_was_graceful", False)
            and time.time() - float(getattr(gui, "_last_stop_translation_ts", 0) or 0) < 2.0):
        return False
    if not gui._require_model_selection("starting translation"):
        return True

    instances = instance_plan.available_instances(config)
    assignment = instance_plan.default_assignment(files, instances, config.get(instance_plan.CONFIG_FILE_MAP))
    dialog = InstanceAssignmentDialog(gui, instances, assignment)
    if dialog.exec() != QDialog.Accepted:
        gui.append_log("⏹️ Multi-instance translation cancelled")
        return True
    assignment = dialog.assignment()
    saved = config.get(instance_plan.CONFIG_FILE_MAP)
    saved = dict(saved) if isinstance(saved, dict) else {}
    for path, number in assignment.items():
        saved[os.path.abspath(path)] = number
    config[instance_plan.CONFIG_FILE_MAP] = saved
    # also brings the window's current settings into gui.config, which the snapshots copy
    gui.save_config(show_message=False)

    plan = instance_plan.plan(files, assignment)
    try:
        snapshots = {number: instance_plan.config_for_instance(gui.config, number) for number in plan}
    except ValueError as exc:
        QMessageBox.warning(gui, "Multi-Instance Translation", str(exc))
        return True
    previous = getattr(gui, "_instance_run", None)
    if previous is not None:
        previous.deleteLater()  # its processes and readers are done (children_alive above)
    run = InstanceRun(gui, plan)
    gui._instance_run = run
    run.start(snapshots)
    return True


# ---------------------------------------------------------------------------
# Assignment dialog
# ---------------------------------------------------------------------------


class InstanceAssignmentDialog(QDialog):
    """Which instance translates which input file (one row per file, an instance dropdown each)."""

    def __init__(self, gui, instances, assignment):
        super().__init__(gui if isinstance(gui, QWidget) else None)
        self.setWindowTitle("Multi-Instance Translation")
        self._files = list(assignment)
        self._combos = []
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each input file is translated by one instance. The instances run at the same time as "
            "separate processes, each with its own log and only the API keys of its own number, in "
            "every key pool (Multi API Key Manager › Instance). Instance 1 runs in this window and "
            "is the only one that uses the main GUI key."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        table = QTableWidget(len(self._files), 2)
        table.setHorizontalHeaderLabels(["Input file", "Instance"])
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for row, path in enumerate(self._files):
            item = QTableWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            table.setItem(row, 0, item)
            combo = QComboBox()
            for number in instances:
                combo.addItem(f"Instance {number}", number)
            combo.setCurrentIndex(max(0, combo.findData(assignment[path])))
            _style_combo(gui, combo)
            table.setCellWidget(row, 1, combo)
            table.setRowHeight(row, 34)
            self._combos.append(combo)
        layout.addWidget(table)

        # centred, spaced apart and wide
        buttons = QHBoxLayout()
        buttons.setSpacing(24)
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(150)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        start_btn = QPushButton("Start")
        start_btn.setMinimumWidth(150)
        start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        start_btn.setDefault(True)
        start_btn.clicked.connect(self.accept)
        buttons.addWidget(start_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.resize(640, min(520, 170 + 34 * len(self._files)))

    def assignment(self):
        """{file: instance} as chosen."""
        return {path: int(combo.currentData()) for path, combo in zip(self._files, self._combos)}


# ---------------------------------------------------------------------------
# One multi-instance run
# ---------------------------------------------------------------------------


class _Child:
    """One instance process."""

    def __init__(self, number, files, folder):
        self.number = number
        self.files = files
        self.folder = folder
        self.stop_file = os.path.join(folder, "stop.flag")
        self.watchdog_dir = None
        self.proc = None
        self.killed = False

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


class InstanceRun(QObject):
    """The instances of one Run click: instance 1 in this window, the others as processes."""

    #: a reader thread queued output lines (coalesced: at most one pending emission)
    lines_ready = Signal()
    #: (instance, exit code) of an instance process that ended
    child_finished = Signal(int, int)
    #: instance 1's worker returned
    instance_one_done = Signal()
    #: instance 1's run asked for its post-translation QA scan
    instance_one_qa = Signal()

    def __init__(self, gui, plan):
        super().__init__(gui if isinstance(gui, QObject) else None)
        self.gui = gui
        self.plan = {int(number): list(files) for number, files in plan.items()}
        self.children = {}
        self.results = {}
        self.stop_requests = 0
        self.qa_requested = set()
        self.instance_one_running = False
        self.finished = False
        #: (the user's selection at Run, instance 1's files) while gui.selected_files holds only
        #: instance 1's files (:func:`user_selection`)
        self.narrowed = None
        #: True while instance 1's own post-translation QA scan starts (it scans its own files)
        self.own_scan = False
        self.run_dir = os.path.join(tempfile.gettempdir(), "glossarion_instances",
                                    f"{os.getpid()}_{int(time.time() * 1000)}")
        self._undo = []
        self._key_owners = {}
        self._scope_undo = None  # this window's instance-1 key scope while the run is alive
        self._qa_queue = []
        self._qa_scope_undo = None  # the key scope of the instance whose QA scan runs now
        self._entered = threading.Event()
        self._outcome = None
        self._pending = collections.deque(maxlen=PENDING_LINES_MAX)
        self._pending_lock = threading.Lock()
        self._drain_scheduled = False
        self.lines_ready.connect(self._drain)
        self.child_finished.connect(self._on_child_finished)
        self.instance_one_done.connect(self._on_instance_one_done)
        self.instance_one_qa.connect(self._on_instance_one_qa)

    # -- state ---------------------------------------------------------------
    def children_alive(self):
        return any(child.alive() for child in self.children.values())

    def release_scopes(self):
        """End this run's key scopes in this window (and the QA scans not started yet)."""
        self._qa_queue = []
        for name in ("_qa_scope_undo", "_scope_undo"):
            undo = getattr(self, name)
            setattr(self, name, None)
            if undo is not None:
                try:
                    undo()
                except Exception as exc:
                    print(f"[instance_runs] key scope restore failed: {exc}")

    def _log(self, number, text):
        """A line in instance *number*'s log (instance 1: the main log)."""
        if number == 1:
            self.gui.append_log(text)
        else:
            self._append(_log_widget(self.gui, number), [text])

    # -- start ---------------------------------------------------------------
    def start(self, snapshots):
        gui = self.gui
        os.makedirs(self.run_dir, exist_ok=True)
        _prepare_views(gui, self.plan)
        # whose key is whose (fingerprints): every instance's request guard refuses the others' keys
        self._key_owners = instance_plan.key_owners(gui.config)
        try:
            # while any instance of the run is alive, everything this window starts (instance 1's
            # run, a review, the Manga translator, a QA scan ...) uses instance 1's keys only
            self._scope_undo = instance_worker.restrict_keys(1, gui.config, key_owners=self._key_owners)
        except Exception as exc:
            gui.append_log(f"❌ Multi-instance translation could not start: instance 1's API keys "
                           f"could not be kept to it: {exc}")
            self.results = {number: "did not start" for number in self.plan}
            self.finished = True
            shutil.rmtree(self.run_dir, ignore_errors=True)
            return
        every_file = [path for files in self.plan.values() for path in files]
        if not getattr(gui, "_input_output_run_active", False):
            try:
                gui._record_library_raw_inputs(every_file)
            except Exception:
                pass
        gui.append_log(f"🧩 Multi-instance translation: {len(every_file)} file(s) in {len(self.plan)} instance(s)")
        for number, files in self.plan.items():
            gui.append_log(f"   Instance {number}: " + ", ".join(os.path.basename(p) for p in files))
        for number, files in self.plan.items():
            if number == 1:
                continue
            try:
                self._spawn(number, files, snapshots[number])
            except Exception as exc:
                self.results[number] = "did not start"
                gui.append_log(f"❌ Instance {number} could not start: {exc}")
        if 1 in self.plan:
            self._start_instance_one(snapshots[1])
        gui.update_run_button()
        self._finish_if_done()

    def _spawn(self, number, files, snapshot):
        from app_paths import _atomic_json_write
        from config_store import save_config_file
        from shutdown_utils import subprocess_no_window_kwargs

        child = _Child(number, files, os.path.join(self.run_dir, f"instance_{number}"))
        os.makedirs(child.folder, exist_ok=True)
        config_path = os.path.join(child.folder, "config.json")
        save_config_file(snapshot, config_path, backup=False)  # API keys encrypted like config.json
        watchdog_root = os.environ.get("GLOSSARION_WATCHDOG_DIR")
        if watchdog_root:
            child.watchdog_dir = os.path.join(watchdog_root, f"{WATCHDOG_SUBDIR}{number}")
            shutil.rmtree(child.watchdog_dir, ignore_errors=True)
        log_dir = os.environ.get("GLOSSARION_LOG_DIR") or child.folder
        # values over the platform's environment limit live in large_env's store, not os.environ
        # (never this window's API keys: the instance exports its own from its snapshot)
        large_env = sys.modules.get("large_env")
        overflow = dict(getattr(large_env, "_store", {}) or {}) if large_env is not None else {}
        overflow = {name: value for name, value in overflow.items() if instance_worker.inheritable(name)}
        job = {
            "version": instance_worker.JOB_VERSION,
            "instance": number,
            "files": list(files),
            "config_path": config_path,
            "env": overflow,
            "watchdog_dir": child.watchdog_dir,
            "stop_file": child.stop_file,
            "log_path": os.path.join(log_dir, f"run_instance_{number}.log"),
            "parent_pid": os.getpid(),
            # fingerprints (never the keys) of every other instance's keys and the main GUI key
            "key_owners": {fingerprint: owners for fingerprint, owners in self._key_owners.items()
                           if number not in owners},
        }
        job_path = os.path.join(child.folder, "job.json")
        _atomic_json_write(job_path, job)
        _log_widget(self.gui, number).appendPlainText(
            f"▶ Instance {number}: " + ", ".join(os.path.basename(p) for p in files))
        child.proc = subprocess.Popen(
            instance_worker.command_line(job_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=instance_worker.child_environment(os.environ, number),
            cwd=os.getcwd(),
            **subprocess_no_window_kwargs(),
        )
        # an instance 1 Stop / the Manga Stop that ends every child of this window spares it
        instance_plan.register_instance_process(child.proc.pid)
        self.children[number] = child
        threading.Thread(target=self._read, args=(child,), name=f"instance-{number}-reader", daemon=True).start()

    def _start_instance_one(self, snapshot):
        """Instance 1's files through the normal Run (``run_translation_thread``), its keys only
        (the run's key scope, :meth:`start`); its helper processes read instance 1's snapshot and
        ``gui.selected_files`` holds only its files until its worker returns."""
        gui = self.gui
        from config_store import save_config_file

        try:
            # the run's helper processes (minimal glossary worker) read their keys from CONFIG_FILE
            folder = os.path.join(self.run_dir, "instance_1")
            os.makedirs(folder, exist_ok=True)
            config_path = os.path.join(folder, "config.json")
            save_config_file(snapshot, config_path, backup=False)
            previous = os.environ.get("CONFIG_FILE")
            os.environ["CONFIG_FILE"] = config_path

            def undo_config_file():
                if previous is None:
                    os.environ.pop("CONFIG_FILE", None)
                else:
                    os.environ["CONFIG_FILE"] = previous

            self._undo.append(undo_config_file)
        except Exception as exc:
            self._restore_instance_one()
            self.results[1] = "did not start"
            gui.append_log(f"❌ Instance 1 could not start: {exc}")
            return

        selection = list(getattr(gui, "selected_files", None) or [])
        own = list(self.plan[1])
        gui.selected_files = list(own)
        self.narrowed = (selection, own)
        self._undo.append(lambda: self._restore_selection(selection, own))
        self._forward_own_qa_scan()

        before = getattr(gui, "translation_thread", None)
        self.instance_one_running = True
        gui._translation_worker = self._instance_one_worker  # the thread target records the outcome
        try:
            gui.run_translation_thread()
        except Exception as exc:
            gui.append_log(f"❌ Instance 1 could not start: {exc}")
        finally:
            try:
                del gui._translation_worker
            except AttributeError:
                pass
        if getattr(gui, "translation_thread", None) is before and not self._entered.is_set():
            # run_translation_thread refused (it logged why)
            self.instance_one_running = False
            self.results[1] = "did not start"
            self._restore_instance_one()

    def _instance_one_worker(self, request):
        self._entered.set()
        outcome = None
        try:
            outcome = type(self.gui)._translation_worker(self.gui, request)
            return outcome
        finally:
            self._outcome = outcome
            self.instance_one_done.emit()

    def _forward_own_qa_scan(self):
        """Instance 1's post-translation scan request starts its scan through
        :meth:`_on_instance_one_qa` (still on its own files); every other ``_ui_request`` kind goes
        to the window's own method. Undone with instance 1's run."""
        gui = self.gui
        original = getattr(gui, "_ui_request", None)
        if not callable(original):
            return
        own = getattr(gui, "__dict__", {})
        had_own, previous = "_ui_request" in own, own.get("_ui_request")

        def ui_request(kind, *args, **data):
            if kind == "trigger_qa_scan":
                self.instance_one_qa.emit()
                return None
            return original(kind, *args, **data)

        gui._ui_request = ui_request

        def undo():
            if had_own:
                gui._ui_request = previous
            else:
                try:
                    del gui._ui_request
                except AttributeError:
                    pass

        self._undo.append(undo)

    def _on_instance_one_qa(self):
        """Instance 1's own post-translation QA scan: its files (:func:`user_selection` reads
        ``gui.selected_files`` for it), instance 1's keys (the run's scope)."""
        self.own_scan = True
        try:
            self.gui._trigger_qa_scan_on_main_thread()
        finally:
            self.own_scan = False

    def _restore_selection(self, selection, own):
        """The full selection again; inputs instance 1's run renamed (archives converted to EPUB)
        keep their new names."""
        gui = self.gui
        self.narrowed = None
        current = list(getattr(gui, "selected_files", None) or [])
        if len(current) != len(own):
            return  # the selection changed meanwhile: leave it
        restored = _full_selection(selection, own, current)
        gui.selected_files = restored
        apply = getattr(gui, "_apply_selected_files_to_input_field", None)
        if callable(apply):
            apply(restored)

    def _restore_instance_one(self):
        undo, self._undo = self._undo, []
        for step in reversed(undo):
            try:
                step()
            except Exception as exc:
                print(f"[instance_runs] restore failed: {exc}")

    def _on_instance_one_done(self):
        self.instance_one_running = False
        if self.stop_requests and not self._outcome:
            self.results[1] = "stopped"
        else:
            self.results[1] = "completed" if self._outcome else "failed"
        self._restore_instance_one()
        self.gui.update_run_button()
        self._finish_if_done()

    # -- instance process output ---------------------------------------------------
    def _read(self, child):
        proc = child.proc
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\r\n").split("\r")[-1]
                if line == instance_worker.QA_SCAN_MARKER:
                    self.qa_requested.add(child.number)
                    continue
                with self._pending_lock:
                    self._pending.append((child.number, line))
                    schedule = not self._drain_scheduled
                    self._drain_scheduled = True
                if schedule:
                    self.lines_ready.emit()
        except Exception:
            pass
        finally:
            try:
                code = proc.wait()
            except Exception:
                code = instance_worker.EXIT_FAILED
            self.child_finished.emit(child.number, int(code))

    def _drain(self):
        with self._pending_lock:
            items = list(self._pending)
            self._pending.clear()
            self._drain_scheduled = False
        grouped = collections.OrderedDict()
        for number, line in items:
            grouped.setdefault(number, []).append(line)
        for number, lines in grouped.items():
            self._append(_log_widget(self.gui, number), lines)

    @staticmethod
    def _append(widget, lines):
        bar = widget.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        widget.appendPlainText("\n".join(lines))
        if at_bottom:
            bar.setValue(bar.maximum())

    def _on_child_finished(self, number, code):
        self._drain()
        child = self.children.get(number)
        if child is not None and child.proc is not None:
            instance_plan.unregister_instance_process(child.proc.pid)
        result = _RESULT_BY_EXIT.get(code, f"ended (exit code {code})")
        if child is not None and child.killed:
            result = "stopped"
        self.results[number] = result
        if child is not None and child.watchdog_dir:
            shutil.rmtree(child.watchdog_dir, ignore_errors=True)  # the process is gone: no live calls
        self._log(number, f"■ Instance {number} {result}")
        self.gui.update_run_button()
        self._finish_if_done()

    # -- stop ----------------------------------------------------------------
    def stop(self, *, graceful, force=False):
        self.stop_requests += 1
        alive = [child for child in self.children.values() if child.alive()]
        if not alive:
            return
        if force or self.stop_requests > 1:
            self.kill_children()
            self.gui.append_log("⚡ Force stop: ended instance process(es) "
                                + ", ".join(str(child.number) for child in alive))
        else:
            mode = instance_worker.STOP_GRACEFUL if graceful else instance_worker.STOP_IMMEDIATE
            for child in alive:
                try:
                    instance_worker.write_stop_request(child.stop_file, mode)
                    self._log(child.number, f"⏹️ Stop sent to instance {child.number} ({mode})")
                except OSError as exc:
                    self._log(child.number, f"⚠️ Could not send Stop to instance {child.number}: {exc}")
            # a second click ends the processes, so Stop stays clickable while one still runs
            button = getattr(self.gui, "run_button", None)
            if button is not None:
                button.setEnabled(True)

    def kill_children(self):
        from shutdown_utils import terminate_subprocess_tree

        for child in self.children.values():
            if child.alive():
                child.killed = True
                terminate_subprocess_tree(child.proc, kill=True)

    # -- end of the run ------------------------------------------------------------
    def _finish_if_done(self):
        if self.finished or self.instance_one_running or self.children_alive():
            return
        if any(number not in self.results for number in self.plan):
            return  # a reader has not reported its process's end yet
        self.finished = True
        self.gui.append_log("🏁 Multi-instance translation finished")
        for number in sorted(self.plan):
            names = ", ".join(os.path.basename(p) for p in self.plan[number])
            self.gui.append_log(f"   Instance {number} — {self.results.get(number, 'did not start')}: {names}")
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.gui.update_run_button()
        self._end_scope_when_idle()

    def _end_scope_when_idle(self):
        """The run's instance-1 key scope ends once no QA scan runs in this window (instance 1's
        post-translation scan, or one started meanwhile, keeps instance 1's keys to its end); then
        the scans the instance processes asked for (:meth:`_next_qa_scan`)."""
        if self._scope_undo is None:
            return  # released already (a new Run, the window closing)
        if _thread_alive(self.gui, "qa_thread", "qa_future"):
            from PySide6.QtCore import QTimer

            QTimer.singleShot(QA_POLL_MS, self._end_scope_when_idle)
            return
        undo, self._scope_undo = self._scope_undo, None
        try:
            undo()
        except Exception as exc:
            print(f"[instance_runs] key scope restore failed: {exc}")
        if self.qa_requested and not self.stop_requests:
            self._qa_queue = [number for number in sorted(self.qa_requested) if self.plan.get(number)]
            self._next_qa_scan()

    def _next_qa_scan(self):
        """The post-translation QA scan the next instance process asked for, over its files, under
        that instance's own keys and config (``instance_worker.restrict_keys`` with its
        ``config_for_instance`` snapshot, which the QA scanner's AI checks read) until the scan's
        thread has ended; one instance at a time."""
        from PySide6.QtCore import QTimer

        gui = self.gui
        if _thread_alive(gui, "qa_thread", "qa_future"):
            QTimer.singleShot(QA_POLL_MS, self._next_qa_scan)  # this instance's scan, or the user's
            return
        undo, self._qa_scope_undo = self._qa_scope_undo, None
        if undo is not None:
            try:
                undo()
            except Exception as exc:
                print(f"[instance_runs] key scope restore failed: {exc}")
        if self.stop_requests:
            self._qa_queue = []  # a Stop meanwhile ends the scans not started yet
        if not self._qa_queue:
            return
        if _thread_alive(gui, "translation_thread", "translation_future") or children_alive(gui):
            files = [path for number in self._qa_queue for path in self.plan.get(number, [])]
            self._qa_queue = []
            gui.append_log("ℹ️ Run QA Scan for: " + ", ".join(os.path.basename(p) for p in files))
            return
        number = self._qa_queue.pop(0)
        files = list(self.plan[number])
        try:
            snapshot = instance_plan.config_for_instance(gui.config, number)
            self._qa_scope_undo = instance_worker.restrict_keys(number, snapshot, key_owners=self._key_owners)
        except Exception as exc:
            gui.append_log(f"⚠️ Instance {number}: the QA scan did not start: {exc}")
            QTimer.singleShot(0, self._next_qa_scan)
            return
        gui.append_log(f"🧪 Instance {number}: QA scan with instance {number}'s API keys — "
                       + ", ".join(os.path.basename(p) for p in files))
        selection = list(getattr(gui, "selected_files", None) or [])
        gui.selected_files = files
        try:
            gui._trigger_qa_scan_on_main_thread()
        finally:
            gui.selected_files = selection
        QTimer.singleShot(QA_POLL_MS, self._next_qa_scan)
