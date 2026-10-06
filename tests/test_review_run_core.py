"""U7: the Review run orchestration (review_generator) against the frozen ReviewDialog.

``ReviewDialog._on_start_review`` / ``_on_generate_all`` (frozen at ``U7_BASE_SHA`` through
``git show``) gathered the run parameters (API key, model, endpoint, temperature, the frozen
config with the live multi-key flag, input token limit, prompts with ``{target_lang}``), reset the
stop flags, applied the streaming toggle, routed the log (``[Review]`` prefix, stdout redirect)
and called ``generate_review`` / ``generate_chunked_review`` (one review, or every input
sequentially / in parallel with per-input output folders). Those steps moved verbatim into
``review_generator`` (``review_run_params``, ``reset_review_stop_flags``,
``apply_review_streaming_env``, ``review_log_fn``, ``ReviewStdoutWriter``, ``run_review``,
``run_all_reviews``, ``review_output_dir_for_file``, ``review_paths_for``, ...); the dialog calls
them, and Glossarion Mobile runs ``run_review_session`` / ``run_all_reviews``.

Tier D: for ``PARITY_REVIEW_STATES`` (default 24) random GUI states per handler, the frozen and the
rewired dialog (offscreen) make the same generator calls (every argument but the callables), log
the same lines (dialog log, main log replay), export the same ``ENABLE_STREAMING`` and leave
``sys.stdout`` restored. Tier M: ``run_review_session`` (mobile) makes the dialog's call.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_review_run_core.py
"""

from __future__ import annotations

import ast
import functools
import os
import random
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import review_generator  # noqa: E402

U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"
STATES = max(1, int(os.environ.get("PARITY_REVIEW_STATES", "24")))
SEED = int(os.environ.get("PARITY_REVIEW_SEED", "717"))
ENV_KEYS = ("ENABLE_STREAMING", "ENDPOINT", "MODEL", "OUTPUT_DIRECTORY")


@functools.lru_cache(maxsize=None)
def _legacy_text():
    try:
        data = subprocess.run(["git", "show", f"{U7_BASE_SHA}:src/review_dialog.py"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, str(exc)
    return data.decode("utf-8-sig").replace("\r\n", "\n"), None


def legacy_module():
    text, error = _legacy_text()
    if text is None:
        pytest.skip(f"review_dialog.py@{U7_BASE_SHA[:8]} unavailable: {error}")
    module = types.ModuleType("review_dialog")
    module.__file__ = str(SRC / "review_dialog.py")
    exec(compile(text, str(SRC / "review_dialog.py"), "exec"), module.__dict__)
    return module


# =============================================================================================
# moved code is the frozen code
# =============================================================================================

def _frozen_method(name):
    text, error = _legacy_text()
    if text is None:
        pytest.skip(error)
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "ReviewDialog")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(text.split("\n")[node.lineno - 1:node.end_lineno])


def _live_function(name):
    text = (SRC / "review_generator.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
    node = next(n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return "\n".join(text.split("\n")[node.lineno - 1:node.end_lineno])


def _norm_lines(src):
    return [line.strip() for line in src.split("\n") if line.strip() and not line.strip().startswith("#")]


@pytest.mark.parametrize("handler", ["_on_start_review", "_on_generate_all"])
@pytest.mark.parametrize("helper,expected_lines", [
    ("review_run_params", (
        "api_key = gui.api_key_entry.text().strip() if hasattr(gui, 'api_key_entry') else ''",
        "model = getattr(gui, 'model_var', os.getenv('MODEL', 'gemini-2.0-flash'))",
        "endpoint = os.environ.get('ENDPOINT', '') or gui.config.get('endpoint', '')",
        "temperature = float(gui.config.get('translation_temperature', 0.3))",
        "config = dict(gui.config)",
        "token_limit = int(gui.token_limit_entry.text().replace(',', '').strip())",
        "token_limit = 200000",
        "output_lang = getattr(gui, 'lang_var', 'English')",
    )),
    ("reset_review_stop_flags", (
        "UnifiedClient.set_global_cancellation(False)", "unified_api_client.global_stop_flag = False",
        "extract_glossary_from_epub.set_stop_flag(False)", "TransateKRtoEN.set_stop_flag(False)",
    )),
])
def test_shared_statements_come_from_both_handlers(handler, helper, expected_lines):
    frozen = _norm_lines(_frozen_method(handler))
    live = _norm_lines(_live_function(helper))
    for line in expected_lines:
        assert line in frozen, (handler, line)
        assert line in live, (helper, line)


def test_streaming_log_and_batch_rules_are_the_frozen_closures():
    start = _norm_lines(_frozen_method("_on_start_review"))
    every = _norm_lines(_frozen_method("_on_generate_all"))
    stream = _norm_lines(_live_function("apply_review_streaming_env"))
    for line in ("'enable_streaming_var',", "os.environ['ENABLE_STREAMING'] = '1' if stream_on else '0'"):
        assert line in start and line in every and line in stream, line
    log = _norm_lines(_live_function("review_log_fn"))
    for line in ("if not stripped or all(c in '─═' for c in stripped):", 'prefixed.append(f"[Review] {line}")',
                 "full = '\\n'.join(prefixed)", "print(full)"):
        assert line in start and line in log, line
    single = _norm_lines(_live_function("single_review_batch_size"))
    every_size = _norm_lines(_live_function("review_all_batch_size"))
    for line in ("chunk_batch_size = int(getattr(gui, 'batch_size_var', 1))", "chunk_batch_size = 1"):
        assert line in start and line in single, line
    for line in ("batch_size = int(getattr(gui, 'batch_size_var', 1))", "batch_size = 1  # Sequential when batch mode is OFF"):
        assert line in every and line in every_size, line
    run_all = _norm_lines(_live_function("run_all_reviews"))
    for line in ("workers = min(batch_size, total)", "with ThreadPoolExecutor(max_workers=workers) as executor:",
                 "put(('all_done', None))"):
        frozen_line = line.replace("put(", "self._review_queue.put(")
        assert frozen_line in every and line in run_all, line


def test_review_modules_stay_gui_free_and_python310():
    for name in ("review_generator.py",):
        data = (SRC / name).read_bytes()
        tree = ast.parse(data.decode("utf-8-sig"), feature_version=(3, 10))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        assert not top & {"PySide6", "translator_gui", "dpi_setup", "review_dialog"}, top
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name


# =============================================================================================
# tier D: frozen vs rewired dialog
# =============================================================================================

class _Entry:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class FakeGui:
    """The TranslatorGUI surface the Review dialog reads."""

    def __init__(self, files, state):
        self.selected_files = [str(p) for p in files]
        self.config = dict(state["config"])
        self.base_dir = str(files[0].parent)
        self.logs = []
        if state["api_key"] is not None:
            self.api_key_entry = _Entry(state["api_key"])
        if state["model"] is not None:
            self.model_var = state["model"]
        if state["token_limit"] is not None:
            self.token_limit_entry = _Entry(state["token_limit"])
        if state["lang"] is not None:
            self.lang_var = state["lang"]
        self.batch_translation_var = state["batch_on"]
        self.batch_size_var = state["batch_size"]
        if state["streaming_var"] is not None:
            self.enable_streaming_var = state["streaming_var"]
        self.graceful_stop_var = True

    def append_log(self, message):
        self.logs.append(str(message))

    def save_config(self, show_message=False):
        return None


class Recorder:
    """generate_review / generate_chunked_review stand-ins."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)
        self.lock = threading.Lock()

    def make(self, kind):
        def generate(**kwargs):
            log_fn = kwargs.pop("log_fn")
            stop = kwargs.pop("stop_check_fn")
            record = {"kind": kind, "stop": bool(stop()) if stop else None}
            record.update({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kwargs.items()})
            with self.lock:
                self.calls.append(record)
            name = os.path.basename(str(kwargs["epub_path"] if not isinstance(kwargs["epub_path"], list)
                                        else kwargs["epub_path"][0]))
            log_fn(f"Reviewing {name}\n──────\n\n  indented line")
            print(f"stdout from {kind} {name}")
            if name in self.fail_on:
                raise RuntimeError(f"boom {name}")
            return f"# Review of {name}"
        return generate


def _random_state(rng, tmp_path):
    return {
        "api_key": rng.choice([None, "", "  sk-live-key  "]),
        "model": rng.choice([None, "gpt-4o", "gemini-2.5-pro"]),
        "token_limit": rng.choice([None, "150,000", " 64000 ", "abc", ""]),
        "lang": rng.choice([None, "Japanese", "Spanish"]),
        "batch_on": rng.choice([False, True, 1, 0]),
        "batch_size": rng.choice([1, 3, "2", 0, -4, "x", None]),
        "streaming_var": rng.choice([None, True, False]),
        "config": {
            "translation_temperature": rng.choice([0.3, "0.7", 1]),
            "endpoint": rng.choice(["", "https://example.invalid/v1"]),
            "use_multi_api_keys": rng.choice([True, False, 0, "yes", None]),
            "enable_streaming": rng.choice([True, False]),
            "output_directory": rng.choice(["", str(tmp_path / "cfg_out")]),
            "review_system_prompt": rng.choice(["", "Review in {target_lang}, please.  "]),
            "review_final_prompt": rng.choice(["", "Final in {target_lang}"]),
            "review_spoiler_mode": rng.choice([True, False]),
            "review_chunk_mode": rng.choice([True, False]),
            "review_chunk_wrap": rng.choice([True, False]),
            "review_volume_mode": rng.choice([True, False]),
            "other": {"nested": [1, 2]},
        },
        "env": {"ENDPOINT": rng.choice([None, "https://env.invalid"]),
                "MODEL": rng.choice([None, "env-model"]),
                "OUTPUT_DIRECTORY": rng.choice([None, str(tmp_path / "env_out")])},
        "fail": rng.choice([(), ("Novel 2.txt",)]),
    }


def _drain(app, dialog, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        timer = getattr(dialog, "_review_poll_timer", None)
        thread = getattr(dialog, "_review_thread", None)
        if timer is not None and not timer.isActive() and (thread is None or not thread.is_alive()):
            break
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


def _run(module, handler, state, files, app, monkeypatch, gen_target):
    recorder = Recorder(state["fail"])
    monkeypatch.setattr(gen_target, "generate_review", recorder.make("single"))
    monkeypatch.setattr(gen_target, "generate_chunked_review", recorder.make("chunked"))
    monkeypatch.setattr(module, "count_epub_tokens", lambda _path: 7)
    monkeypatch.setattr(module, "count_review_tokens", lambda paths: 11)
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in state["env"].items():
        if value is not None:
            monkeypatch.setenv(key, value)
    gui = FakeGui(files, state)
    stdout = sys.stdout
    dialog = module.ReviewDialog(None, gui, str(files[0]))
    try:
        getattr(dialog, handler)()
        _drain(app, dialog)
        log_field = dialog.log_field.toPlainText()
    finally:
        dialog.hide()
        dialog.deleteLater()
        app.processEvents()
    restored = sys.stdout is stdout
    sys.stdout = stdout
    calls = sorted(recorder.calls, key=lambda c: str(c["epub_path"]))
    return {"calls": calls, "log_field": sorted(log_field.split("\n")), "main_log": sorted(gui.logs),
            "streaming": os.environ.get("ENABLE_STREAMING"), "stdout_restored": restored}


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("handler", ["_on_start_review", "_on_generate_all"])
def test_rewired_dialog_matches_the_frozen_dialog(handler, qapp, tmp_path, monkeypatch):
    import review_dialog
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Yes)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "library"))
    files = []
    for name in ("Novel 2.txt", "Novel 10.txt", "Novel 1.txt"):
        path = tmp_path / "books" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"{name} text", encoding="utf-8")
        files.append(path)
    legacy = legacy_module()
    rng = random.Random(SEED + (0 if handler == "_on_start_review" else 1))
    seen_kinds, seen_parallel = set(), False
    saved_env = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        for i in range(STATES):
            state = _random_state(rng, tmp_path)
            n_files = 1 if (handler == "_on_start_review" and rng.random() < 0.3) else 3
            use = files[:n_files]
            old = _run(legacy, handler, state, use, qapp, monkeypatch, legacy)
            new = _run(review_dialog, handler, state, use, qapp, monkeypatch, review_generator)
            assert old == new, (i, state)
            assert new["stdout_restored"]
            seen_kinds |= {c["kind"] for c in new["calls"]}
            seen_parallel |= any("in parallel" in line for line in new["log_field"])
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert seen_kinds == {"single", "chunked"}
    if handler == "_on_generate_all" and STATES >= 12:
        assert seen_parallel


# =============================================================================================
# tier M: the mobile entry makes the dialog's call
# =============================================================================================

def test_run_review_session_makes_the_dialogs_call(qapp, tmp_path, monkeypatch):
    import review_dialog

    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "library"))
    files = []
    for name in ("Vol 2.txt", "Vol 1.txt"):
        path = tmp_path / "books" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name, encoding="utf-8")
        files.append(path)
    rng = random.Random(SEED + 9)
    for _ in range(8):
        state = _random_state(rng, tmp_path)
        state["fail"] = ()
        desk = _run(review_dialog, "_on_start_review", state, files, qapp, monkeypatch, review_generator)
        recorder = Recorder()
        monkeypatch.setattr(review_generator, "generate_review", recorder.make("single"))
        monkeypatch.setattr(review_generator, "generate_chunked_review", recorder.make("chunked"))
        gui = FakeGui(files, state)
        cfg = gui.config
        volume = bool(cfg.get("review_volume_mode"))
        volume_paths = sorted([str(p) for p in files], key=review_dialog._natural_path_key)
        stdout = sys.stdout
        result = review_generator.run_review_session(
            gui,
            prompt=cfg.get("review_system_prompt") or review_generator.DEFAULT_REVIEW_PROMPT,
            spoiler_mode=bool(cfg.get("review_spoiler_mode")),
            chunk_mode=bool(cfg.get("review_chunk_mode")),
            wrap_chunks=bool(cfg.get("review_chunk_wrap", True)),
            final_review_prompt=cfg.get("review_final_prompt") or review_generator.DEFAULT_FINAL_REVIEW_PROMPT,
            file_path=str(files[0]),
            volume_paths=volume_paths,
            volume_mode=volume,
            stop_check_fn=lambda: False,
        )
        assert sys.stdout is stdout
        assert recorder.calls == desk["calls"]
        assert result.startswith("# Review of ")
        assert os.environ.get("ENABLE_STREAMING") == desk["streaming"]
        assert any(line.startswith("[Review] Reviewing") for line in gui.logs)


def test_run_all_reviews_reports_every_input_and_honours_stop(tmp_path, monkeypatch):
    monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
    recorder = Recorder(("b.txt",))
    monkeypatch.setattr(review_generator, "generate_review", recorder.make("single"))
    gui = FakeGui([tmp_path / "a.txt"], {"api_key": "k", "model": "m", "token_limit": "1,000", "lang": None,
                                         "batch_on": True, "batch_size": 2, "streaming_var": None,
                                         "config": {"output_directory": str(tmp_path / "out")}})
    params = review_generator.review_run_params(gui, "Prompt {target_lang}", False)
    assert params.as_dict()["system_prompt"] == "Prompt English" and params.token_limit == 1000
    messages = []
    done = review_generator.run_all_reviews(params, [tmp_path / "a.txt", tmp_path / "b.txt"], chunk_mode=False,
                                            wrap_chunks=True, batch_size=2, put=messages.append, stop_check=lambda: False)
    assert done == (2, 1)
    assert messages[-1] == ("all_done", None)
    assert sorted(m[1] for m in messages if m[0] == "nav") == [0, 1]
    assert {c["output_dir"] for c in recorder.calls} == {str(tmp_path / "out" / "a"), str(tmp_path / "out" / "b")}
    stopped = []
    assert review_generator.run_all_reviews(params, [tmp_path / "a.txt"], chunk_mode=False, wrap_chunks=True,
                                            batch_size=1, put=stopped.append, stop_check=lambda: True) == (0, 0)
    assert stopped[-1] == ("all_done", None)


# =============================================================================================
# 🗑️ Delete / ↩️ Restore: the frozen handlers vs the rewired ones (+ the mobile helpers)
# =============================================================================================

class _Widget:
    """A Qt widget stand-in that records every call (text() / styleSheet() return its state)."""

    def __init__(self, name, log):
        self._name, self._log, self._text, self._style = name, log, f"{name} text", ""

    def text(self):
        return self._text

    def styleSheet(self):
        return self._style

    def setText(self, value):
        self._text = value
        self._log.append((self._name, "setText", value))

    def setStyleSheet(self, value):
        self._style = value
        self._log.append((self._name, "setStyleSheet", value))

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return lambda *args, **kwargs: self._log.append((self._name, attr) + tuple(args))


def _review_handlers(text, globals_):
    """``_on_delete``, ``_get_backups_dir``, ``_on_restore`` of ``text`` compiled against ``globals_``."""
    import textwrap

    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "ReviewDialog")
    ns = dict(globals_)
    for name in ("_on_delete", "_get_backups_dir", "_on_restore", "_update_restore_btn_visibility"):
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(textwrap.dedent("\n".join(text.split("\n")[node.lineno - 1:node.end_lineno])),
                     f"<{name}>", "exec"), ns)
    return ns


class _FakeBox:
    """``QMessageBox`` for the restore question: records the text, answers from ``ANSWER``."""

    Warning, Yes, No = 3, 1, 2
    ANSWER = 1
    LOG = []

    def __init__(self, parent=None):
        pass

    def setText(self, text):
        _FakeBox.LOG.append(("box", text))

    def exec(self):
        return _FakeBox.ANSWER

    def __getattr__(self, attr):
        return lambda *args, **kwargs: None


def _review_dialog_side(handlers, root, paths, events):
    owner = types.SimpleNamespace()
    for widget in ("delete_btn", "restore_btn", "start_btn", "generate_all_btn", "log_field"):
        setattr(owner, widget, _Widget(widget, events))
    owner._get_review_paths = lambda: list(paths)
    owner._get_review_path = lambda: paths[0] if paths else None
    owner._all_epub_paths = ["a.epub"]
    owner._is_volume_mode = lambda: False
    owner.translator_gui = types.SimpleNamespace(_update_review_indicator=lambda: events.append(("indicator",)))
    owner._safe_delayed_reset = lambda button, *a, **k: events.append(("reset", button._name) + a[1:2])
    owner._append_log = lambda message: events.append(("log", message))
    owner._md_to_html = lambda content, **kw: f"<html>{content}</html>"
    owner._get_font_kwargs = lambda: {}
    owner._load_remote_images = lambda: events.append(("images",))
    for name in ("_on_delete", "_get_backups_dir", "_on_restore", "_update_restore_btn_visibility"):
        setattr(owner, name, types.MethodType(handlers[name], owner))
    return owner


def _review_tree(root):
    import re as _re

    out = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            rel = _re.sub(r"review_\d{8}_\d{6}_\d{6}\.md$", "review_<stamp>.md", path.relative_to(root).as_posix())
            out.setdefault(rel, []).append(path.read_text(encoding="utf-8"))
    return out


@pytest.mark.parametrize("layout", ["single", "volume", "no_review", "only_backup", "declined"])
def test_delete_and_restore_match_the_frozen_handlers(layout, tmp_path, monkeypatch):
    """Frozen (U7 base) and rewired ``_on_delete`` / ``_on_restore`` leave the same files, make the same
    widget calls and ask the same question; the mobile helpers do the same file moves."""
    import PySide6.QtWidgets as qtw

    frozen_text, error = _legacy_text()
    if frozen_text is None:
        pytest.skip(error)
    live_text = (SRC / "review_dialog.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    import review_dialog

    monkeypatch.setattr(qtw, "QMessageBox", _FakeBox)
    monkeypatch.setitem(sys.modules, "winsound", types.SimpleNamespace(MessageBeep=lambda *a: None, MB_OK=0))
    sides = {}
    for side, text, globals_ in (("frozen", frozen_text, {"os": os}), ("live", live_text, vars(review_dialog))):
        root = tmp_path / side
        names = ["Book"] if layout != "volume" else ["Vol1", "Vol2"]
        paths = [str(root / n / "review" / ("combined_review/review.md" if layout == "volume" else "review.md"))
                 for n in names]
        for index, path in enumerate(paths):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if layout != "no_review" and layout != "only_backup":
                Path(path).write_text(f"# Review {index}\n", encoding="utf-8")
        if layout in ("only_backup", "declined"):
            backups = Path(paths[0]).parent / "backups"
            backups.mkdir(parents=True, exist_ok=True)
            (backups / "review_20260101_000000_000001.md").write_text("old 1\n", encoding="utf-8")
            (backups / "review_20260102_000000_000002.md").write_text("old 2\n", encoding="utf-8")
        events = []
        _FakeBox.LOG = []
        _FakeBox.ANSWER = _FakeBox.No if layout == "declined" else _FakeBox.Yes
        owner = _review_dialog_side(_review_handlers(text, globals_), root, paths, events)
        if layout != "declined":
            owner._on_delete()
            after_delete = _review_tree(root)
        else:
            after_delete = None
        owner._on_restore()
        sides[side] = {"events": events, "boxes": list(_FakeBox.LOG), "after_delete": after_delete,
                       "after_restore": _review_tree(root), "raw": getattr(owner, "_raw_review_md", None),
                       "backups_dir": os.path.relpath(owner._get_backups_dir(), root) if owner._get_backups_dir() else None}
    assert sides["frozen"] == sides["live"]
    if layout == "declined":
        assert sides["live"]["boxes"] and sides["live"]["boxes"][0][1] == review_generator.review_restore_question(
            "review_20260102_000000_000002.md")

    # the mobile helpers: the same moves without the dialog
    root = tmp_path / "mobile"
    paths = [str(root / "Book" / "review" / "review.md")]
    Path(paths[0]).parent.mkdir(parents=True)
    Path(paths[0]).write_text("# Review 0\n", encoding="utf-8")
    assert review_generator.latest_review_backup(paths) is None and review_generator.restore_review_backup(paths) is False
    name = review_generator.move_review_to_backups(paths, timestamp="20260103_000000_000003")
    assert name == "review_20260103_000000_000003.md" and not Path(paths[0]).exists()
    assert review_generator.latest_review_backup(paths) == (str(Path(paths[0]).parent / "backups"), name)
    assert review_generator.restore_review_backup(paths) == "# Review 0\n" and Path(paths[0]).is_file()
