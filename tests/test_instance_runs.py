"""Multi-instance translation: the main window's side (src/instance_runs.py + translator_gui wiring).

* a single file, multi-keys off or one instance number: the Run button runs exactly as before
  (``start_run`` returns False, no dialog, no process; ``_run_translation_clicked`` calls
  ``run_translation_thread`` once);
* a multi-instance run: the assignment is saved as ``instance_file_map``; instance 1 runs through
  ``run_translation_thread`` on its own files with only its Translation keys (and a config snapshot
  for its helper processes), everything restored afterwards; instance 2 is a real process whose
  output reaches its own log; the summary goes to instance 1's log; the QA scan an instance asked
  for runs over its files at the end;
* Stop: a stop file per instance process, a second Stop kills the process tree;
* the per-instance logs, the watchdog filter (``_update_api_watchdog``) and the context menu
  actions on the visible log; the 14 specs ship the modules.

The instance processes are small stand-in scripts (``instance_worker.command_line`` patched);
tests/test_instance_worker.py covers the real process. Runs offscreen. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_instance_runs.py
"""

import ast
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit, QProgressBar, QPushButton  # noqa: E402

APP = QApplication.instance() or QApplication([])

import instance_plan  # noqa: E402
import instance_runs  # noqa: E402
import instance_worker  # noqa: E402

KEYS = [
    {"api_key": "sk-one", "enabled": True},
    {"api_key": "sk-two", "enabled": True, "instance": 2},
]

#: Stand-in instance process: reports what it got, then behaves as FAKE_CHILD_MODE says.
CHILD_SCRIPT = r'''
import json, os, sys, time
job = json.load(open(sys.argv[1], encoding="utf-8"))
print("child instance", os.environ.get("GLOSSARION_INSTANCE"), "stop-file-env", os.environ.get("GLOSSARY_STOP_FILE"),
      "files", ",".join(os.path.basename(p) for p in job["files"]), flush=True)
mode = os.environ.get("FAKE_CHILD_MODE", "ok")
if job.get("kind") == "glossary":
    print("glossary job", flush=True)
if mode == "ok":
    if job.get("kind") != "glossary":
        print("##GLOSSARION_INSTANCE_QA_SCAN##", flush=True)
    print("translated", flush=True)
    sys.exit(0)
deadline = time.time() + 30
while time.time() < deadline:
    if mode == "obey" and os.path.exists(job["stop_file"]):
        print("stopping:", open(job["stop_file"], encoding="utf-8").read(), flush=True)
        sys.exit(2)
    time.sleep(0.05)
sys.exit(1)
'''


def _wait(condition, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    APP.processEvents()
    return condition()


class FakeGui:
    """The parts of TranslatorGUI instance_runs uses (real Qt widgets for the log and the bar)."""

    def __init__(self, config, files):
        self.config = config
        self.selected_files = list(files)
        self.log_text = QPlainTextEdit()
        self.log_text.setMaximumBlockCount(5000)
        self.api_watchdog_bar = QProgressBar()
        self.run_button = QPushButton()
        self.main_lines = []
        self.saved = 0
        self.recorded = []
        self.translation_thread = None
        self.button_updates = 0
        self.worker_seen = []
        self.qa_scans = []
        self.qa_scopes = []
        self.qa_thread = self.qa_future = None
        self.glossary_thread = self.glossary_future = self.translation_future = None
        self.keep_worker_running = threading.Event()
        self.keep_worker_running.set()
        self.watchdog_row = instance_runs.watchdog_row(self)
        self.log_area = instance_runs.log_area(self)

    def append_log(self, text):
        self.main_lines.append(str(text))
        self.log_text.appendPlainText(str(text))

    def save_config(self, show_message=True):
        self.saved += 1
        return True

    def _require_model_selection(self, why):
        return True

    def _parallel_epub_pair_is_selected(self):
        return False

    def update_run_button(self):
        self.button_updates += 1

    def _show_context_menu(self, pos):
        pass

    def _record_library_raw_inputs(self, files):
        self.recorded.append(list(files))

    def _apply_selected_files_to_input_field(self, paths):
        self.selected_files = list(paths)

    def _trigger_qa_scan_on_main_thread(self):
        self.qa_scans.append(list(self.selected_files))
        self.qa_scopes.append(instance_plan.key_scope())  # the keys the scan's requests may use

    def _update_api_watchdog(self):
        pass

    def run_translation_thread(self):
        # like the desktop: the thread target is self._translation_worker, read at start
        self.translation_thread = threading.Thread(target=self._translation_worker,
                                                   args=({"files": list(self.selected_files)},), daemon=True)
        self.translation_thread.start()

    def _translation_worker(self, request):
        try:
            import unified_api_client

            unified_api_client.UnifiedClient.setup_multi_key_pool(self.config["multi_api_keys"])
            snapshot = None
            if os.environ.get("CONFIG_FILE"):
                from config_store import load_config

                snapshot = load_config(os.environ["CONFIG_FILE"])
            self.worker_seen.append({
                "files": list(self.selected_files),
                "pool": list(unified_api_client.UnifiedClient._api_key_pool),
                "snapshot_keys": [k["api_key"] for k in (snapshot or {}).get("multi_api_keys", [])],
                "snapshot_fallback": [k["api_key"] for k in (snapshot or {}).get("fallback_keys", [])],
                "key_scope": instance_plan.key_scope(),
            })
            self.keep_worker_running.wait(10)
            return True
        finally:
            self.translation_thread = None


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake client module, watchdog/log folders, patched key-pool apply and stand-in processes.

    Like the real client, the fake narrows every key list to the key scope
    (``instance_plan.scope_keys``) and keeps its pool as a class attribute, which
    ``job_runner.isolated_key_pools`` detaches for instance 1's run and puts back afterwards."""
    monkeypatch.delenv(instance_plan.ENV_KEY_INSTANCE, raising=False)
    monkeypatch.delenv(instance_plan.ENV_INSTANCE, raising=False)

    class UnifiedClient:
        _api_key_pool = None

        @classmethod
        def set_in_memory_multi_keys(cls, keys_list, force_rotation=True, rotation_frequency=1):
            cls.setup_multi_key_pool(keys_list)

        @classmethod
        def setup_multi_key_pool(cls, keys_list, force_rotation=True, rotation_frequency=1):
            cls._api_key_pool = [k["api_key"] for k in instance_plan.scope_keys(keys_list)]

    UnifiedClient.setup_multi_key_pool(KEYS)  # the pool the window had before the run
    pool = UnifiedClient._api_key_pool
    module = types.ModuleType("unified_api_client")
    module.UnifiedClient = UnifiedClient
    monkeypatch.setitem(sys.modules, "unified_api_client", module)
    import key_pools

    applied = []

    def fake_apply(config, *, export_env=True):
        applied.append(export_env)
        UnifiedClient.set_in_memory_multi_keys(config.get("multi_api_keys", []))

    monkeypatch.setattr(key_pools, "apply_key_pools_to_runtime", fake_apply)
    script = tmp_path / "fake_instance.py"
    script.write_text(CHILD_SCRIPT, encoding="utf-8")
    monkeypatch.setattr(instance_worker, "command_line", lambda job_path: [sys.executable, str(script), job_path])
    watchdog = tmp_path / "watchdog"
    (watchdog / "instance_2").mkdir(parents=True)
    (watchdog / "instance_2" / "api_watchdog_999999.json").write_text("{}", encoding="utf-8")  # stale
    monkeypatch.setenv("GLOSSARION_WATCHDOG_DIR", str(watchdog))
    monkeypatch.setenv("GLOSSARION_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("GLOSSARY_STOP_FILE", str(tmp_path / "gui_stop.flag"))
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("GLOSSARION_INSTANCE", raising=False)
    monkeypatch.setattr(instance_runs.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    dialogs = []

    def accept(self):
        dialogs.append(self.assignment())
        return QDialog.Accepted

    monkeypatch.setattr(instance_runs.InstanceAssignmentDialog, "exec", accept)
    files = []
    for name in ("a.epub", "b.epub"):
        path = tmp_path / name
        path.write_text("x", encoding="utf-8")
        files.append(str(path))
    return types.SimpleNamespace(tmp=tmp_path, pool=pool, applied=applied, client=UnifiedClient, files=files,
                                 watchdog=watchdog, dialogs=dialogs, monkeypatch=monkeypatch)


def _config(**extra):
    config = {"use_multi_api_keys": True, "multi_api_keys": [dict(k) for k in KEYS]}
    config.update(extra)
    return config


# --------------------------------------------------------------------------
# single-instance runs stay as they were
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config, count", [
    (_config(), 1),                                                        # one file
    (_config(use_multi_api_keys=False), 2),                                # multi-keys off
    (_config(multi_api_keys=[{"api_key": "a"}, {"api_key": "b", "instance": 1}]), 2),  # one instance number
    (_config(multi_api_keys=[{"api_key": "a"}, {"api_key": "b", "instance": 2, "enabled": False}]), 2),
])
def test_single_instance_runs_never_enter_the_multi_instance_branch(env, monkeypatch, config, count):
    def boom(*_a, **_k):
        raise AssertionError("multi-instance branch entered")

    monkeypatch.setattr(instance_runs, "InstanceAssignmentDialog", boom)
    monkeypatch.setattr(instance_runs, "InstanceRun", boom)
    gui = FakeGui(config, env.files[:count])
    assert instance_runs.start_run(gui) is False
    assert gui.saved == 0 and gui.recorded == [] and gui.main_lines == []
    assert instance_plan.CONFIG_FILE_MAP not in gui.config
    assert getattr(gui, "_instance_run", None) is None


@pytest.fixture(scope="module")
def translator_gui_module():
    try:
        import translator_gui
    except Exception as exc:  # pragma: no cover - a build without the GUI's dependencies
        pytest.skip(f"translator_gui is not importable here: {exc}")
    return translator_gui


def test_run_button_runs_run_translation_thread_once_for_a_single_instance(env, translator_gui_module):
    calls = []
    gui = FakeGui(_config(), env.files[:1])
    gui.run_translation_thread = lambda: calls.append("run")
    translator_gui_module.TranslatorGUI._run_translation_clicked(gui, False)
    assert calls == ["run"]


# --------------------------------------------------------------------------
# a multi-instance run
# --------------------------------------------------------------------------

def test_extract_glossary_button_assigns_files_and_stops_child(env, translator_gui_module):
    gui = FakeGui(_config(), env.files)
    gui._has_epub_conversion_inputs = lambda: False
    gui.glossary_button = QPushButton()
    gui.graceful_stop_var = True
    gui._start_glossary_extraction = lambda worker: worker()
    gui.run_glossary_extraction_direct = lambda: gui._translation_worker(None)
    env.monkeypatch.setenv("FAKE_CHILD_MODE", "obey")

    translator_gui_module.TranslatorGUI.run_glossary_extraction_thread(gui)
    run = gui._instance_run
    assert env.dialogs == [{env.files[0]: 1, env.files[1]: 2}]
    assert gui.worker_seen[0]["files"] == env.files[:1]
    assert gui.worker_seen[0]["snapshot_keys"] == ["sk-one"]
    assert _wait(lambda: run.results.get(1) == "completed")
    assert instance_runs.run_active(gui, "glossary")
    assert not instance_runs.run_active(gui, "translation")
    child_log = instance_runs._log_widget(gui, 2)
    assert _wait(lambda: "glossary job" in child_log.toPlainText())

    translator_gui_module.TranslatorGUI.stop_glossary_extraction(gui)
    assert _wait(lambda: run.finished)
    assert run.results == {1: "completed", 2: "stopped"}
    assert "stopping: graceful" in child_log.toPlainText()
    assert gui.selected_files == env.files
    assert instance_plan.key_scope() is None


def test_multi_instance_run(env):
    import config_store

    snapshots = {}
    real_save = config_store.save_config_file

    def recording_save(config, path=None, *, backup=True):
        snapshots[Path(path).parent.name] = [k["api_key"] for k in config.get("multi_api_keys", [])]
        snapshots[Path(path).parent.name + " fallback"] = [k["api_key"] for k in config.get("fallback_keys", [])]
        snapshots[Path(path).parent.name + " main key"] = config.get("api_key")
        return real_save(config, path, backup=backup)

    env.monkeypatch.setattr(config_store, "save_config_file", recording_save)
    gui = FakeGui(_config(api_key="sk-main", fallback_keys=[{"api_key": "fb-one", "model": "m"},
                                                             {"api_key": "fb-two", "model": "m", "instance": 2}],
                          use_fallback_keys=True), env.files)
    saved_config = json.loads(json.dumps(gui.config))
    gui.keep_worker_running.clear()  # instance 1 runs until the child has finished
    assert instance_runs.start_run(gui) is True
    run = gui._instance_run
    a, b = env.files
    assert env.dialogs == [{a: 1, b: 2}]  # prefilled round-robin
    assert gui.config[instance_plan.CONFIG_FILE_MAP] == {os.path.abspath(a): 1, os.path.abspath(b): 2}
    assert gui.saved == 1 and gui.recorded == [[a, b]]
    # every pool split by instance; the main GUI key is instance 1's only
    assert snapshots == {"instance_2": ["sk-two"], "instance_2 fallback": ["fb-two"], "instance_2 main key": "sk-two",
                         "instance_1": ["sk-one"], "instance_1 fallback": ["fb-one"], "instance_1 main key": "sk-main"}
    assert not gui.watchdog_instance_combo.isHidden() and not hasattr(gui, "log_instance_combo")
    # instance 2: a process whose output reaches its own log; a stale watchdog file was removed
    assert _wait(lambda: run.results.get(2) == "completed")
    child_log = gui._instance_logs[2].toPlainText()
    assert "child instance 2 stop-file-env None files b.epub" in child_log
    assert "translated" in child_log and instance_worker.QA_SCAN_MARKER not in child_log
    assert "■ Instance 2 completed" in child_log
    assert not (env.watchdog / "instance_2").exists()
    assert instance_runs.children_alive(gui) is False and run.qa_requested == {2}
    # instance 1 meanwhile: its own files, keys (every pool) and snapshot
    assert _wait(lambda: gui.worker_seen)
    assert gui.worker_seen == [{"files": [a], "pool": ["sk-one"], "snapshot_keys": ["sk-one"],
                                "snapshot_fallback": ["fb-one"], "key_scope": 1}]
    assert env.client._api_key_pool is not env.pool and env.pool == ["sk-one", "sk-two"]  # not narrowed in place
    assert 1 not in run.results and not run.finished
    gui.keep_worker_running.set()
    assert _wait(lambda: run.finished)
    assert run.results == {1: "completed", 2: "completed"}
    # the QA scan instance 2 asked for runs with instance 2's keys only; its key scope ends with the scan
    assert gui.qa_scopes == [2]
    assert _wait(lambda: instance_plan.key_scope() is None)
    # everything instance 1 narrowed is back: the very pool the window had, the saved config untouched
    assert gui.selected_files == [a, b] and "CONFIG_FILE" not in os.environ
    assert env.client._api_key_pool is env.pool and env.pool == ["sk-one", "sk-two"]
    # the pools were built for instance 1's scope (the run) and instance 2's (its QA scan), never exported
    assert env.applied == [False, False] and instance_plan.key_scope() is None
    assert {k: v for k, v in gui.config.items() if k != instance_plan.CONFIG_FILE_MAP} == saved_config
    env.client.setup_multi_key_pool(KEYS)
    assert env.client._api_key_pool == ["sk-one", "sk-two"]  # the key filter is gone
    assert gui._translation_worker.__func__ is FakeGui._translation_worker
    assert gui.main_lines[-4:] == ["🏁 Multi-instance translation finished",
                                   "   Instance 1 — completed: a.epub", "   Instance 2 — completed: b.epub",
                                   "🧪 Instance 2: QA scan with instance 2's API keys — b.epub"]
    assert not os.path.exists(run.run_dir)
    # the QA scan instance 2 asked for, over its own file
    assert gui.qa_scans == [[b]] and gui.selected_files == [a, b]
    # the views stay for the last run; the one switcher picks the API calls and the log
    gui.watchdog_instance_combo.setCurrentIndex(gui.watchdog_instance_combo.findData(2))
    assert instance_runs.visible_log(gui) is gui._instance_logs[2] and instance_runs.watchdog_view(gui) == 2
    gui.watchdog_instance_combo.setCurrentIndex(gui.watchdog_instance_combo.findData(instance_runs.ALL_INSTANCES))
    assert instance_runs.visible_log(gui) is gui._instance_logs[2] and instance_runs.watchdog_view(gui) is None
    gui.watchdog_instance_combo.setCurrentIndex(gui.watchdog_instance_combo.findData(1))
    assert instance_runs.visible_log(gui) is gui.log_text and instance_runs.watchdog_view(gui) == 1
    # a later single-file run goes back to the plain log
    gui.selected_files = [a]
    assert instance_runs.start_run(gui) is False
    assert gui.watchdog_instance_combo.isHidden() and instance_runs.visible_log(gui) is gui.log_text
    assert instance_runs.watchdog_view(gui) is None


def test_cancelled_assignment_starts_nothing(env, monkeypatch):
    monkeypatch.setattr(instance_runs.InstanceAssignmentDialog, "exec", lambda self: QDialog.Rejected)
    gui = FakeGui(_config(), env.files)
    assert instance_runs.start_run(gui) is True
    assert gui.saved == 0 and getattr(gui, "_instance_run", None) is None
    assert gui.main_lines == ["⏹️ Multi-instance translation cancelled"]


def _children_only_run(env, mode):
    env.monkeypatch.setenv("FAKE_CHILD_MODE", mode)
    a, b = env.files
    gui = FakeGui(_config(instance_file_map={a: 2, b: 2}), env.files)
    assert instance_runs.start_run(gui) is True
    run = gui._instance_run
    assert run.plan == {2: [a, b]} and gui.worker_seen == []
    assert _wait(lambda: "child instance 2" in gui._instance_logs[2].toPlainText())
    return gui, run


def test_stop_reaches_the_instance_processes(env):
    gui, run = _children_only_run(env, "obey")
    gui.run_button.setEnabled(False)  # stop_translation greys the button out
    instance_runs.stop_children(gui, graceful=False)
    assert gui.run_button.isEnabled()  # a second click can still end the processes
    assert _wait(lambda: run.finished)
    assert run.results == {2: "stopped"} and run.children[2].killed is False
    assert "stopping: immediate" in gui._instance_logs[2].toPlainText()
    assert gui.main_lines[-1] == "   Instance 2 — stopped: a.epub, b.epub"


def test_second_stop_kills_the_process_tree(env):
    gui, run = _children_only_run(env, "ignore")
    instance_runs.stop_children(gui, graceful=True)
    assert instance_worker.read_stop_request(run.children[2].stop_file) == instance_worker.STOP_GRACEFUL
    started = time.time()
    instance_runs.stop_children(gui, graceful=True)
    assert _wait(lambda: run.finished, timeout=15) and time.time() - started < 15
    assert run.results == {2: "stopped"} and run.children[2].killed is True
    assert any("Force stop: ended instance process(es) 2" in line for line in gui.main_lines)


def test_force_stop_and_shutdown_kill_at_once(env):
    gui, run = _children_only_run(env, "ignore")
    instance_runs.shutdown(gui)
    assert _wait(lambda: run.finished, timeout=15)
    assert run.children[2].killed is True and not os.path.exists(run.run_dir)


# --------------------------------------------------------------------------
# watchdog filter, logs, context menu
# --------------------------------------------------------------------------

def test_one_instance_switcher_left_of_the_watchdog_bar(env):
    """One switcher (no second one above the log), left of the bar (away from the Stop button),
    styled like the main window's combos (the Halgakos arrow helper)."""
    styled = []

    class StyledGui(FakeGui):
        def _add_combobox_arrow(self, combo):
            styled.append(combo)

    gui = StyledGui(_config(), env.files)
    layout = gui.watchdog_row.layout()
    assert layout.itemAt(0).widget() is gui.watchdog_instance_combo
    assert layout.itemAt(1).widget() is gui.api_watchdog_bar
    assert styled == [gui.watchdog_instance_combo]
    assert gui.log_area is gui.log_stack and not hasattr(gui, "log_instance_combo")


# --------------------------------------------------------------------------
# while an instance runs: instance 1's keys in the window, the user's whole selection
# --------------------------------------------------------------------------

def test_main_window_keeps_instance_one_keys_until_the_last_instance_ends(env):
    """Owner rule: instance 1 has finished, instance 2 still runs: everything the window starts now (a review,
    the Manga translator, a QA scan builds its key pool) gets instance 1's keys only. Once instance 2 has ended,
    every key again and the very pool the window had."""
    env.monkeypatch.setenv("FAKE_CHILD_MODE", "obey")
    gui = FakeGui(_config(api_key="sk-main"), env.files)
    assert instance_runs.start_run(gui) is True
    run = gui._instance_run
    assert _wait(lambda: run.results.get(1) == "completed")
    assert run.children_alive() and not run.finished
    assert instance_plan.key_scope() == 1
    env.client.setup_multi_key_pool(KEYS)  # a feature started now builds its pool
    assert env.client._api_key_pool == ["sk-one"]
    assert gui.selected_files == env.files and run.narrowed is None  # instance 1's own narrowing is over
    instance_worker.write_stop_request(run.children[2].stop_file, instance_worker.STOP_GRACEFUL)
    assert _wait(lambda: run.finished)
    assert run.results[2] == "stopped"
    assert _wait(lambda: instance_plan.key_scope() is None)
    assert env.client._api_key_pool is env.pool
    env.client.setup_multi_key_pool(KEYS)
    assert env.client._api_key_pool == ["sk-one", "sk-two"]


class _PMDialog:
    def __init__(self):
        from PySide6.QtWidgets import QDialog

        self.widget = QDialog()
        self.shown = 0

    def __getattr__(self, name):
        return getattr(self.widget, name)

    def show(self):
        self.shown += 1


def test_actions_see_the_whole_selection_while_instance_one_runs(env, translator_gui_module):
    """The owner's bug: Progress Manager during a multi-instance run said "Please select a valid EPUB, text file,
    or image folder first." Instance 1's run holds only its own file in ``selected_files`` (it reads and renames
    that list); the Progress Manager and the window's other actions ask ``user_selection``: every file of the run,
    instance 1's under the name its run gave it. Instance 1's own post-translation QA scan keeps its own file."""
    import types as _types

    from QA_Scanner_GUI import QAScannerMixin
    from Retranslation_GUI import RetranslationMixin

    env.monkeypatch.setenv("FAKE_CHILD_MODE", "obey")
    gui = FakeGui(_config(), env.files)
    gui.keep_worker_running.clear()
    a, b = env.files
    assert instance_runs.start_run(gui) is True
    run = gui._instance_run
    try:
        assert _wait(lambda: gui.worker_seen)
        assert gui.selected_files == [a] and instance_runs.user_selection(gui) == [a, b]
        gui.selected_files[0] = a + ".epub"  # instance 1's run renamed its input in place
        assert instance_runs.user_selection(gui) == [a + ".epub", b]
        gui.selected_files[0] = a

        # Progress Manager: the multi-file manager with both books, never the single-file error
        opened, messages = [], []
        gui._force_retranslation_multiple_files = lambda selected_files=None: opened.append(list(selected_files))
        gui._show_message = lambda *args, **kwargs: messages.append(args)
        gui._selected_subtitle_bundle_progress_target = _types.MethodType(
            RetranslationMixin._selected_subtitle_bundle_progress_target, gui)
        RetranslationMixin.force_retranslation(gui)
        assert opened == [[a, b]] and messages == []
        # its cached multi-file dialog (built for both books) is shown again, not thrown away as stale
        dialog = _PMDialog()
        gui._multi_file_retranslation_dialog = dialog
        gui._multi_file_selection_key = tuple(sorted([a, b]))
        assert translator_gui_module.TranslatorGUI._show_cached_progress_manager_if_any(gui) is True
        assert dialog.shown == 1 and gui._multi_file_retranslation_dialog is dialog
        # the QA Scan button's auto-search looks for both books' output folders
        sources, _stored, _direct = QAScannerMixin._qa_auto_search_sources(gui)
        assert sources == [a, b]
        # ... except instance 1's own post-translation scan, which belongs to its run: its own book
        run.own_scan = True
        assert QAScannerMixin._qa_auto_search_sources(gui)[0] == [a]
        run.own_scan = False
        assert gui.selected_files == [a]  # instance 1's run still reads only its own file
    finally:
        gui.keep_worker_running.set()
        instance_worker.write_stop_request(run.children[2].stop_file, instance_worker.STOP_GRACEFUL)
        assert _wait(lambda: run.finished)
        assert _wait(lambda: instance_plan.key_scope() is None)
    assert gui.selected_files == [a, b] and instance_runs.user_selection(gui) == [a, b]
    assert run.narrowed is None


def test_instance_one_post_translation_scan_goes_through_the_run(env):
    """Instance 1's worker asks for its post-translation QA scan (``_ui_request('trigger_qa_scan')``): the run
    starts it on instance 1's own file under instance 1's keys; other request kinds reach the window as before,
    and the window's own ``_ui_request`` is back once instance 1 has finished."""
    calls = []

    class Gui(FakeGui):
        def _ui_request(self, kind, *args, **data):
            calls.append(kind)

        def _translation_worker(self, request):
            self._ui_request("input_files_updated", list(self.selected_files))
            self._ui_request("trigger_qa_scan")
            return True

    gui = Gui(_config(), env.files)
    assert instance_runs.start_run(gui) is True
    run = gui._instance_run
    assert _wait(lambda: run.finished)
    assert calls == ["input_files_updated"] and gui.qa_scans[0] == [env.files[0]] and gui.qa_scopes[0] == 1
    assert "_ui_request" not in vars(gui)


def test_every_rerouted_action_reads_the_user_selection():
    """The main window's actions that read the selection at click time go through ``user_selection``."""
    def body(path, name):
        source = (SRC_DIR / path).read_text(encoding="utf-8")
        start = source.index(f"def {name}(")
        end = source.find("\n    def ", start + 10)
        return source[start:end if end > 0 else None]

    for path, name in [
        ("Retranslation_GUI.py", "force_retranslation"),
        ("Retranslation_GUI.py", "_force_retranslation_multiple_files"),
        ("Retranslation_GUI.py", "_progress_manager_current_input_signature"),
        ("translator_gui.py", "_show_cached_progress_manager_if_any"),
        ("translator_gui.py", "open_output_folder"),
        ("translator_gui.py", "_open_review_dialog"),
        ("translator_gui.py", "_update_review_indicator"),
        ("translator_gui.py", "load_glossary"),
        ("QA_Scanner_GUI.py", "_qa_auto_search_sources"),
        ("review_dialog.py", "_build_ui"),
        ("review_dialog.py", "refresh_epub_list"),
        ("other_settings.py", "validate_epub_structure_gui"),
        ("other_settings.py", "_delete_translation_cache_files"),
    ]:
        text = body(path, name)
        assert "user_selection(" in text, (path, name)
        assert "self.selected_files" not in text.replace("self.selected_files = selection", ""), (path, name)
    # the Progress Manager rebuild hands its selection over instead of overwriting selected_files
    assert "self.selected_files = selection" not in (SRC_DIR / "Retranslation_GUI.py").read_text(encoding="utf-8")


def test_watchdog_file_helpers(tmp_path):
    (tmp_path / "instance_3").mkdir()
    own = tmp_path / "api_watchdog_1.json"
    child = tmp_path / "instance_3" / "api_watchdog_2.json"
    for path in (own, child, tmp_path / "other.json"):
        path.write_text("{}", encoding="utf-8")
    assert sorted(instance_runs.watchdog_files(str(tmp_path))) == sorted([str(own), str(child)])
    assert instance_runs.file_instance({}, str(own)) == 1
    assert instance_runs.file_instance({}, str(child)) == 3
    assert instance_runs.file_instance({"instance": 2}, str(own)) == 2
    assert instance_runs.file_instance(None, str(child)) == 3


def _watchdog_gui(translator_gui_module, env):
    gui = FakeGui(_config(), env.files)
    gui._elide_watchdog_bar_text = lambda text, max_chars=140: text
    instance_runs._prepare_views(gui, [1, 2])
    folder = env.watchdog
    (folder / "instance_2" / "api_watchdog_999999.json").unlink()
    soon = time.time() + 60  # no "last change Ns ago" in the label
    entries = lambda *chapters: [{"chapter": c, "status": "in_flight", "request_id": f"r{c}"} for c in chapters]
    (folder / "api_watchdog_111.json").write_text(json.dumps(
        {"pid": 111, "instance": 1, "in_flight": 1, "in_flight_entries": entries(7), "last_change_ts": soon}), "utf-8")
    (folder / "instance_2" / "api_watchdog_222.json").write_text(json.dumps(
        {"pid": 222, "instance": 2, "in_flight": 2, "in_flight_entries": entries(3, 4), "last_change_ts": soon}), "utf-8")
    module = sys.modules["unified_api_client"]
    module.get_api_watchdog_state = lambda: {}
    return gui


@pytest.mark.parametrize("view, expected", [
    (instance_runs.ALL_INSTANCES, "API calls: 3"), (1, "API calls: 1"), (2, "API calls: 2")])
def test_update_api_watchdog_filters_by_instance(env, translator_gui_module, view, expected):
    gui = _watchdog_gui(translator_gui_module, env)
    combo = gui.watchdog_instance_combo
    combo.setCurrentIndex(combo.findData(view))
    translator_gui_module.TranslatorGUI._update_api_watchdog(gui)
    label = gui.api_watchdog_bar.format()
    assert label.startswith(expected), label
    assert ("7" in label) == (view in (instance_runs.ALL_INSTANCES, 1))
    assert ("3, 4" in label) == (view in (instance_runs.ALL_INSTANCES, 2))


def test_watchdog_shows_everything_while_the_filter_is_hidden(env, translator_gui_module):
    gui = _watchdog_gui(translator_gui_module, env)
    instance_runs.show_single_instance(gui)
    translator_gui_module.TranslatorGUI._update_api_watchdog(gui)
    assert gui.api_watchdog_bar.format().startswith("API calls: 3")


def test_log_actions_use_the_visible_log(env, translator_gui_module, monkeypatch):
    copied = []
    clipboard = types.SimpleNamespace(setText=copied.append)
    monkeypatch.setattr(translator_gui_module, "QApplication", types.SimpleNamespace(clipboard=lambda: clipboard))
    gui = FakeGui(_config(), env.files)
    instance_runs._prepare_views(gui, [1, 2])
    gui.log_text.setPlainText("main log")
    instance_runs._log_widget(gui, 2).setPlainText("instance two")
    gui.watchdog_instance_combo.setCurrentIndex(gui.watchdog_instance_combo.findData(2))
    translator_gui_module.TranslatorGUI.select_all_log(gui)
    translator_gui_module.TranslatorGUI.copy_selection(gui)
    assert copied == ["instance two"]
    assert not gui.log_text.textCursor().hasSelection()


# --------------------------------------------------------------------------
# wiring and packaging
# --------------------------------------------------------------------------

def _tg_methods():
    tree = ast.parse((SRC_DIR / "translator_gui.py").read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
    return {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_translator_gui_wiring():
    methods = _tg_methods()
    assert "instance_runs.start_run" not in methods["run_translation_thread"]  # local worker never opens the dialog
    clicked = methods["_run_translation_clicked"]
    assert "if instance_runs.start_run(self):" in clicked and "self.run_translation_thread()" in clicked
    assert "self.run_button.clicked.connect(self._run_translation_clicked)" in methods["update_run_button"]
    assert any("self.run_button.clicked.connect(self._run_translation_clicked)" in body  # the button is built
               for name, body in methods.items() if name != "update_run_button")
    assert not any("clicked.connect(self.run_translation_thread)" in body for body in methods.values())
    for name in ("update_run_button", "_is_any_process_running"):
        for kind in ("translation", "glossary"):
            assert f"instance_runs.run_active(self, {kind!r})" in methods[name], name
    assert "instance_runs.stop_children(self, graceful=graceful_stop, force=force_stop)" in methods["stop_translation"]
    assert "instance_runs.shutdown(self)" in methods["closeEvent"]
    log_section = methods["_create_log_section"]
    assert "instance_runs.watchdog_row(self)" in log_section and "instance_runs.log_area(self)" in log_section
    assert "self.log_text = QPlainTextEdit()" in log_section  # instance 1's log is the same widget
    watchdog = methods["_update_api_watchdog"]
    assert "instance_runs.watchdog_files(watchdog_dir)" in watchdog and "instance_runs.file_instance(st, fp)" in watchdog
    assert "state['instance'] = instance_plan.current_instance()" in methods["_create_watchdog_snapshot"]


def _spec_list(source, name):
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = ["))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].rstrip() == "]")
    return ast.literal_eval("\n".join(lines[start:end + 1]).split("=", 1)[1].strip())


def test_every_spec_ships_the_multi_instance_modules():
    specs = sorted(SRC_DIR.glob("translator*.spec"))
    assert len(specs) == 14
    for spec in specs:
        source = spec.read_text(encoding="utf-8")
        files = [Path(entry[0]).stem for entry in _spec_list(source, "app_files")]
        modules = _spec_list(source, "app_modules")
        for name in ("instance_plan", "instance_worker", "instance_runs"):
            assert files.count(name) == 1 and modules.count(name) == 1, (spec.name, name)
