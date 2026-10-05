"""Tier D for the U4 Other Settings / Glossary Manager moves: frozen legacy vs working tree.

Moved in milestone U4 (chain 2) out of the desktop dialogs into GUI-free shared modules,
with the dialog functions left as thin wrappers:

* other_settings ``on_profile_select`` / ``save_profile`` / ``delete_profile`` /
  ``save_profiles`` / ``import_profiles`` / ``export_profiles`` -> ``prompt_profiles``;
* other_settings ``_sync_thoughts_lock_state`` -> ``settings_rules.thoughts_lock_state``;
* other_settings ``_set_output_mode`` and the ``_update_output_mode_sub_settings`` closure of
  ``_create_image_translation_section`` -> ``settings_rules.output_mode_flags`` /
  ``output_mode_sub_settings``;
* the GlossaryManager_GUI ``update_auto_glossary_state`` closure of ``_setup_auto_glossary_tab``
  (mode mapping + the 🔒 toggle locks) -> ``settings_rules.glossary_mode_*``;
* (U4 review fixes) the TranslatorGUI "Assistant Prompt" dialog closures
  (``show_assistant_prompt_dialog``) -> ``prompt_profiles.prefill_*``, ``_quick_new_profile`` ->
  ``prompt_profiles.new_profile``, the worker of ``_fetch_authgem_projects`` ->
  ``authgem_auth.list_gcp_projects``, and the preservation block of the other_settings
  ``_reset_config_to_defaults`` closure -> ``config_store.reset_preserved_keys``.

The oracle is the source of each function / closure at ``U4_BASE_SHA`` (the U4 parent commit,
``git show``), executed against the live module globals; the new side is the working tree.
Both run on identical harnesses built from real Qt widgets (offscreen) for at least
``PARITY_U4_STATES`` (default 500) random states per moved function, and every observable is
compared after every step: owner attributes, config, the config.json bytes, combo box /
editor / radio / checkbox / label / slider state, message boxes (titles, texts, answers),
logs, hook calls, exceptions and the exported environment.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/parity/test_u4_dialog_parity.py
"""

from __future__ import annotations

import ast
import contextlib
import copy
import functools
import io
import json
import logging
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QFileDialog, QLabel, QMessageBox, QPushButton, QRadioButton,
    QSlider, QTextEdit, QWidget,
)

#: Handlers that raise inside Qt signal slots (e.g. on_profile_select without an editor) raise
#: identically on both sides; the comparison checks their effects, so pytest-qt must not
#: turn them into failures.
pytestmark = pytest.mark.qt_no_exception_capture

#: U4 parent commit (src identical to the U3 freeze a7aa4a75 outside src/mobile).
U4_BASE_SHA = "9f06f8b3efa6195a0486bddeed1c07e4af719b3c"
STATES = max(1, int(os.environ.get("PARITY_U4_STATES", "500")))
SEED = int(os.environ.get("PARITY_U4_SEED", "4242"))

PROFILE_FUNCTIONS = ("on_profile_select", "save_profile", "delete_profile", "save_profiles",
                     "import_profiles", "export_profiles")
ENV_KEYS = ("ENABLE_THOUGHTS", "ENABLE_IMAGE_OUTPUT_MODE", "ENABLE_VIDEO_OUTPUT_MODE", "ENABLE_AUDIO_OUTPUT_MODE",
            "ENABLE_REFINEMENT_OUTPUT_MODE", "ENABLE_IMAGE_TRANSLATION", "OUTPUT_MODE")
_ABSENT = "<absent>"


# =============================================================================================
# legacy sources
# =============================================================================================

def _git_text(relpath):
    try:
        data = subprocess.run(["git", "show", f"{U4_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"legacy source {relpath}@{U4_BASE_SHA[:8]} unavailable: {exc}")
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def _top_functions(text, names, filename):
    tree = ast.parse(text, filename=filename)
    found = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = set(names) - set(found)
    assert not missing, f"{filename}@{U4_BASE_SHA[:8]} lacks {sorted(missing)}"
    return [found[name] for name in names]


def _nested_function(text, outer_class, outer, inner, filename):
    tree = ast.parse(text, filename=filename)
    scope = tree
    if outer_class:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == outer_class)
    outer_fn = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == outer)
    node = next(n for n in ast.walk(outer_fn) if isinstance(n, ast.FunctionDef) and n.name == inner)
    lines = text.split("\n")
    return textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))


def _closure_factory(source, params, namespace, name):
    """``factory(*params) -> closure`` for a nested function's source (free variables as params)."""
    body = textwrap.indent(source, "    ")
    code = f"def _factory({', '.join(params)}):\n{body}\n    return {name}\n"
    ns = dict(namespace)
    exec(compile(code, f"<{name}>", "exec"), ns)
    return ns["_factory"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def other_settings_mod(qapp):
    import other_settings
    return other_settings


@pytest.fixture(scope="module")
def legacy_profile_functions(other_settings_mod):
    text = _git_text("src/other_settings.py")
    nodes = _top_functions(text, PROFILE_FUNCTIONS + ("_sync_thoughts_lock_state", "_set_output_mode"),
                           "other_settings.py")
    return text, nodes


def _bind_legacy(nodes, module, overrides):
    ns = dict(vars(module))
    ns.update(overrides)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<legacy other_settings>", "exec"), ns)
    return ns


# =============================================================================================
# recorders shared by both sides
# =============================================================================================

class Boxes:
    """QMessageBox / QFileDialog stand-ins: record every box, answer from a script."""

    def __init__(self):
        self.log = []
        self.answers = []
        self.open_path = ""
        self.save_path = ""

    def install(self, monkeypatch):
        boxes = self

        def static(kind):
            def show(*args, **kwargs):
                texts = [a for a in args if isinstance(a, str)]
                boxes.log.append((kind, tuple(texts)))
                return QMessageBox.Ok
            return show

        def exec_(box):
            answer = boxes.answers.pop(0) if boxes.answers else QMessageBox.No
            boxes.log.append(("exec", box.windowTitle(), box.text(), box.informativeText(),
                              int(box.defaultButton() is not None), "yes" if answer == QMessageBox.Yes else "no"))
            return answer

        for kind in ("critical", "warning", "information", "question"):
            monkeypatch.setattr(QMessageBox, kind, staticmethod(static(kind)))
        monkeypatch.setattr(QMessageBox, "exec", exec_)
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (boxes.open_path, "")))
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (boxes.save_path, "")))


def _env_snapshot():
    return {key: os.environ.get(key, _ABSENT) for key in ENV_KEYS}


class _env_guard:
    def __enter__(self):
        self.saved = {key: os.environ.get(key) for key in ENV_KEYS}
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _run(fn):
    try:
        return ("ok", repr(fn()))
    except Exception as exc:  # the legacy code may raise; the new code must raise the same
        return ("raise", type(exc).__name__, str(exc))


def _normalize_dir(obj, directory):
    """The trace with the side's temp folder replaced by <DIR> (messages quote paths)."""
    text = json.dumps(obj, default=repr)
    raw = str(directory)
    for variant in sorted({raw, raw.replace("\\", "\\\\"), raw.replace("\\", "\\\\\\\\")}, key=len, reverse=True):
        text = text.replace(variant, "<DIR>")
    return json.loads(text)


def _compare(left, right, where):
    if left != right:
        keys = sorted(set(left) | set(right)) if isinstance(left, dict) else []
        diff = {k: (left.get(k), right.get(k)) for k in keys if left.get(k) != right.get(k)}
        raise AssertionError(f"{where}: legacy and new differ: {diff or (left, right)}")


# =============================================================================================
# 1. prompt profiles (other_settings profile actions)
# =============================================================================================

BUILT_INS = {
    "Universal": "Universal built-in {target_lang}",
    "Refinement": "Refine built-in",
    "Korean_BeautifulSoup": "KO BS built-in",
    "Japanese_html2text": "JA h2t built-in",
    "Subtitle Translation": "Subtitle built-in",
}
CUSTOM_NAMES = ("Alpha", "Beta", "Gamma beautifulsoup", "delta_HTML2TEXT", " Spaced ", "New Profile #1")
NAME_POOL = tuple(BUILT_INS) + CUSTOM_NAMES + ("", "   ", "Missing", "Universal copy", "Renamed")


def _random_profile_state(rng):
    names = [n for n in BUILT_INS if rng.random() < 0.8]
    names += [n for n in CUSTOM_NAMES if rng.random() < 0.5]
    rng.shuffle(names)
    if not names and rng.random() < 0.8:
        names = ["Universal"]
    profiles = {n: (BUILT_INS.get(n, f"{n} saved") if rng.random() < 0.7 else f"{n} edited {rng.randint(0, 9)}")
                for n in names}
    active = rng.choice(names + [""]) if names else ""
    state = {
        "profiles": profiles,
        "active": active,
        "autosave": rng.choice([_ABSENT, None, active, rng.choice(NAME_POOL)]),
        "originals": (_ABSENT if rng.random() < 0.3 else
                      {n: f"{n} original" for n in names if rng.random() < 0.4}),
        "editor": rng.choice(["", "Draft text", profiles.get(active, "x"), "  padded  "]),
        "menu_text": rng.choice([active, rng.choice(NAME_POOL), active.upper()]),
        "has_menu": rng.random() < 0.93,
        "has_editor": rng.random() < 0.97,
        "radios": rng.random() < 0.5,
        "method": rng.choice(["standard", "enhanced"]),
        "protected": rng.choice(["mixin", "mixin", "raise", "absent"]),
        "restore_pending": rng.random() < 0.05,
        "save_btn": rng.random() < 0.3,
        "same_dict": rng.random() < 0.7,
        "config_extra": {"unrelated": "kept", "profile_name_autofill": rng.choice([True, False, 0, "yes"])},
        "file": rng.choice(["absent", "json", "json", "corrupt"]),
        "open_fails": rng.random() < 0.06,
        "refresh_btn": rng.random() < 0.5,
    }
    if rng.random() < 0.3:
        state["config_extra"]["profile_mousewheel_locked"] = rng.choice([True, False])
    return state


class ProfileOwner(QObject):
    """TranslatorGUI's profile surface: real widgets, the real autosave and mixin helpers."""

    def __init__(self, functions, state, boxes, auto_save, mixin):
        super().__init__()
        self.boxes = boxes
        self.logs = []
        self.calls = []
        profiles = dict(state["profiles"])
        self.config = {"prompt_profiles": profiles if state["same_dict"] else dict(profiles),
                       "active_profile": state["active"]}
        self.config.update(copy.deepcopy(state["config_extra"]))
        self.prompt_profiles = profiles
        self.profile_var = state["active"]
        self.default_prompts = dict(BUILT_INS)
        self.text_extraction_method_var = state["method"]
        if state["autosave"] != _ABSENT:
            self._active_profile_for_autosave = state["autosave"]
        if state["originals"] != _ABSENT:
            self._original_profile_content = dict(state["originals"])
        if state["restore_pending"]:
            self._config_restore_pending = True
        for name, fn in functions.items():
            setattr(self, name, types.MethodType(fn, self))
        self._auto_save_system_prompt = types.MethodType(auto_save, self)
        self._reset_prompt_profile_to_default = types.MethodType(mixin._reset_prompt_profile_to_default, self)
        if state["protected"] == "mixin":
            self._get_protected_prompt_profiles = types.MethodType(mixin._get_protected_prompt_profiles, self)
        elif state["protected"] == "raise":
            def _broken(_self):
                raise RuntimeError("protected list unavailable")
            self._get_protected_prompt_profiles = types.MethodType(_broken, self)
        self.holder = QWidget()
        if state["has_menu"]:
            self.profile_menu = QComboBox(self.holder)
            self.profile_menu.setEditable(True)
            self.profile_menu.addItems(list(profiles))
            index = self.profile_menu.findText(state["active"])
            if index >= 0:
                self.profile_menu.setCurrentIndex(index)
            if state["menu_text"] != self.profile_menu.currentText():
                self.profile_menu.setEditText(state["menu_text"])
            self.profile_menu.currentIndexChanged.connect(lambda: self.on_profile_select())
        if state["has_editor"]:
            self.prompt_text = QTextEdit(self.holder)
            self.prompt_text.setPlainText(state["editor"])
            self.prompt_text.textChanged.connect(self._auto_save_system_prompt)
        if state["radios"]:
            self.standard_extraction_radio = QRadioButton("BS", self.holder)
            self.enhanced_extraction_radio = QRadioButton("h2t", self.holder)
            (self.standard_extraction_radio if state["method"] == "standard"
             else self.enhanced_extraction_radio).setChecked(True)
        if state["save_btn"]:
            self._save_profile_btn = QPushButton("Save Profile", self.holder)
        if state["refresh_btn"]:
            self._refresh_progress_text_analysis_button = lambda: self.calls.append(("refresh_review",))

    def append_log(self, message):
        self.logs.append(message)

    def _update_profile_delete_button_label(self, name=None):
        self.calls.append(("label", name))

    def snapshot(self, config_file):
        menu = getattr(self, "profile_menu", None)
        editor = getattr(self, "prompt_text", None)
        btn = getattr(self, "_save_profile_btn", None)
        return {
            "profiles": list(self.prompt_profiles.items()),
            "config": json.dumps(self.config, sort_keys=False, default=repr),
            "same_dict": self.config.get("prompt_profiles") is self.prompt_profiles,
            "profile_var": self.profile_var,
            "autosave": getattr(self, "_active_profile_for_autosave", _ABSENT),
            "originals": list(getattr(self, "_original_profile_content", {_ABSENT: _ABSENT}).items()),
            "method": self.text_extraction_method_var,
            "menu": (None if menu is None else
                     ([menu.itemText(i) for i in range(menu.count())], menu.currentIndex(), menu.currentText())),
            "editor": None if editor is None else editor.toPlainText(),
            "radios": (None if not hasattr(self, "standard_extraction_radio") else
                       (self.standard_extraction_radio.isChecked(), self.enhanced_extraction_radio.isChecked())),
            "btn": None if btn is None else (btn.text(), btn.styleSheet(), btn.isEnabled()),
            "timer": hasattr(self, "_save_profile_timer"),
            "file": config_file.read_bytes() if config_file.exists() else None,
            "boxes": list(self.boxes.log),
            "logs": list(self.logs),
            "calls": list(self.calls),
        }


def _write_config_file(path, state, rng_text):
    if path.exists():
        path.unlink()
    if state["file"] == "json":
        path.write_text(json.dumps({"api_key": "ENC:secret", "prompt_profiles": {"Old": "o"}, "zz": 1,
                                    "active_profile": "Old", "note": rng_text}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    elif state["file"] == "corrupt":
        path.write_text("{not json", encoding="utf-8")


def _profile_ops(rng, focus):
    ops = [rng.choice(PROFILE_FUNCTIONS + ("select_combo", "type_name", "edit_text"))
           for _ in range(rng.randint(0, 3))]
    ops.append(focus)
    return ops


def _apply_profile_op(owner, op, rng_values, boxes, import_file, export_dir):
    """Apply one op with pre-drawn random values (identical on both sides)."""
    kind, value = rng_values
    if op == "select_combo":
        menu = getattr(owner, "profile_menu", None)
        if menu is not None and menu.count():
            return _run(lambda: menu.setCurrentIndex(value % menu.count()))
        return ("skip",)
    if op == "type_name":
        menu = getattr(owner, "profile_menu", None)
        if menu is not None:
            return _run(lambda: menu.setEditText(NAME_POOL[value % len(NAME_POOL)]))
        return ("skip",)
    if op == "edit_text":
        editor = getattr(owner, "prompt_text", None)
        if editor is not None:
            return _run(lambda: editor.setPlainText(f"typed {value}"))
        return ("skip",)
    if op == "delete_profile":
        boxes.answers = [QMessageBox.Yes if kind else QMessageBox.No]
    if op == "import_profiles":
        choice = value % 5
        if choice == 0:
            boxes.open_path = ""
        else:
            boxes.open_path = str(import_file)
            payload = [{"Imported": "imp", "Alpha": "replaced"}, {"Beta": "b2"}, "{broken", [["Pair", "p"]]][choice - 1]
            import_file.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    if op == "export_profiles":
        choice = value % 4
        boxes.save_path = ["", str(export_dir / "out"), str(export_dir / "out.json"),
                           str(export_dir / "missing_dir" / "x.json")][choice]
    return _run(getattr(owner, op))


@pytest.fixture(scope="module")
def profile_sides(other_settings_mod, legacy_profile_functions):
    from _src_corpus import find_method
    from owner_state import ConfigStateMixin

    _text, nodes = legacy_profile_functions
    # TranslatorGUI's real autosave (the editor's textChanged handler)
    ns = {"os": os}
    exec(compile(ast.Module(body=[find_method("_auto_save_system_prompt")], type_ignores=[]),
                 "<translator_gui>", "exec"), ns)
    return nodes, ns["_auto_save_system_prompt"], ConfigStateMixin


@pytest.mark.parametrize("focus", PROFILE_FUNCTIONS)
def test_profile_actions_match_legacy(qapp, tmp_path, monkeypatch, other_settings_mod, profile_sides, focus):
    nodes, auto_save, mixin = profile_sides
    boxes = Boxes()
    boxes.install(monkeypatch)
    try:  # the save confirmation beeps on Windows
        import winsound
        monkeypatch.setattr(winsound, "MessageBeep", lambda *_a: None)
    except ImportError:
        pass
    rng = random.Random(f"{SEED}:{focus}")
    legacy_file = tmp_path / "legacy" / "config.json"
    new_file = tmp_path / "new" / "config.json"
    for path in (legacy_file, new_file):
        path.parent.mkdir()
    legacy_ns = _bind_legacy(nodes, other_settings_mod, {"CONFIG_FILE": str(legacy_file)})
    legacy_functions = {name: legacy_ns[name] for name in PROFILE_FUNCTIONS}
    new_functions = {name: getattr(other_settings_mod, name) for name in PROFILE_FUNCTIONS}
    monkeypatch.setattr(other_settings_mod, "CONFIG_FILE", str(new_file))
    counts = {"states": 0, "focus_ok": 0, "focus_raise": 0}

    def failing_open(*args, **kwargs):
        raise OSError("simulated disk error")

    for index in range(STATES):
        state = _random_profile_state(rng)
        ops = _profile_ops(rng, focus)
        values = [(rng.random() < 0.6, rng.randint(0, 50)) for _ in ops]
        note = f"state {index}"
        results = []
        for side, functions, cfg_file in (("legacy", legacy_functions, legacy_file),
                                          ("new", new_functions, new_file)):
            _write_config_file(cfg_file, state, note)
            import_file = cfg_file.parent / "import.json"
            export_dir = cfg_file.parent / "export"
            export_dir.mkdir(exist_ok=True)
            for leftover in export_dir.glob("*"):
                leftover.unlink()
            boxes.log, boxes.answers = [], []
            if state["open_fails"]:
                if side == "legacy":
                    legacy_ns["open"] = failing_open
                else:
                    monkeypatch.setattr(other_settings_mod, "open", failing_open, raising=False)
            owner = ProfileOwner(functions, state, boxes, auto_save, mixin)
            trace = []
            for op, value in zip(ops, values):
                outcome = _apply_profile_op(owner, op, value, boxes, import_file, export_dir)
                exported = sorted((p.name, p.read_bytes().decode("utf-8", "replace")) for p in export_dir.glob("*"))
                snap = owner.snapshot(cfg_file)
                snap["file"] = None if snap["file"] is None else snap["file"].decode("utf-8", "replace")
                trace.append((op, outcome, snap, exported))
            results.append(_normalize_dir(trace, cfg_file.parent))
            legacy_ns.pop("open", None)
            monkeypatch.delattr(other_settings_mod, "open", raising=False)
            timer = getattr(owner, "_save_profile_timer", None)
            if timer is not None:
                timer.stop()
            owner.holder.deleteLater()
        for step, (left, right) in enumerate(zip(*results)):
            _compare({"op": left[0], "outcome": left[1], "export": left[3], **left[2]},
                     {"op": right[0], "outcome": right[1], "export": right[3], **right[2]},
                     f"{focus} {note} step {step} ops={ops} state={state}")
        counts["states"] += 1
        last = results[0][-1][1]
        counts["focus_ok" if last[0] == "ok" else "focus_raise"] += 1
        if index % 50 == 0:
            qapp.processEvents()
    assert counts["states"] >= STATES
    assert counts["focus_ok"] >= counts["states"] // 2, counts


# =============================================================================================
# 2. stream thinking -> Enable thoughts lock
# =============================================================================================

class ThoughtsOwner:
    def __init__(self, fn, state):
        self.config = {"enable_thoughts": state["initial"], "other": 1}
        self.enable_thoughts_var = state["initial"]
        self._sync_thoughts_lock_state = types.MethodType(fn, self)
        self.holder = QWidget()
        if state["checkbox"]:
            cb = QCheckBox("Enable thoughts (include model reasoning metadata)", self.holder)
            cb.setChecked(state["checked"])
            if state["tick"]:
                tick = QLabel("✓", cb)
                tick.setVisible(state["checked"])
            if state["locked_before"]:
                cb._mode_locked = True
                cb._original_text = "Enable thoughts (include model reasoning metadata)"
                cb.setText("🔒 Enable thoughts (include model reasoning metadata)")
                cb.setEnabled(False)
            self.enable_thoughts_checkbox = cb

    def snapshot(self):
        cb = getattr(self, "enable_thoughts_checkbox", None)
        out = {"config": dict(self.config), "var": self.enable_thoughts_var, "env": _env_snapshot()}
        if cb is not None:
            ticks = [(c.text(), c.isVisibleTo(cb), c.geometry().getRect()) for c in cb.findChildren(QLabel)]
            out["cb"] = (cb.text(), cb.isChecked(), cb.isEnabled(), cb.styleSheet(),
                         getattr(cb, "_mode_locked", _ABSENT), getattr(cb, "_original_text", _ABSENT), ticks)
        return out


def test_thoughts_lock_matches_legacy(qapp, other_settings_mod, legacy_profile_functions):
    _text, nodes = legacy_profile_functions
    legacy_fn = _bind_legacy(nodes, other_settings_mod, {})["_sync_thoughts_lock_state"]
    new_fn = other_settings_mod._sync_thoughts_lock_state
    rng = random.Random(f"{SEED}:thoughts")
    for index in range(STATES):
        state = {"initial": rng.choice([True, False]), "checkbox": rng.random() < 0.9,
                 "checked": rng.choice([True, False]), "tick": rng.random() < 0.8,
                 "locked_before": rng.random() < 0.4}
        calls = [rng.choice([True, False, 1, 0, None, "x", ""]) for _ in range(rng.randint(1, 3))]
        traces = []
        for fn in (legacy_fn, new_fn):
            with _env_guard():
                owner = ThoughtsOwner(fn, state)
                trace = []
                for value in calls:
                    trace.append((_run(lambda: owner._sync_thoughts_lock_state(value)), owner.snapshot()))
                traces.append(trace)
                owner.holder.deleteLater()
        _compare({"t": traces[0]}, {"t": traces[1]}, f"_sync_thoughts_lock_state state {index} {state} {calls}")


# =============================================================================================
# 3. output mode setter and its sub-settings visibility
# =============================================================================================

MODE_INPUTS = ("text", "vision", "image", "video", "audio", "refinement", "refine", " Image ", "VIDEO",
               "unknown", "", None, 3)


class OutputOwner:
    def __init__(self, fn, state):
        self.calls = []
        self.config = None if state["config_none"] else {"output_mode": "text", "kept": 1}
        self._set_output_mode = types.MethodType(fn, self)
        self.holder = QWidget()
        if state["toggle"] == "ok":
            self.toggle_image_translation_section = lambda: self.calls.append("toggle")
        elif state["toggle"] == "raise":
            def _raise():
                raise RuntimeError("toggle failed")
            self.toggle_image_translation_section = _raise
        if state["radios"]:
            self._output_mode_radios = {}
            for mode in ("text", "vision", "image", "video", "audio", "refinement"):
                if state["radios"] == "all" or mode in ("text", "image"):
                    rb = QRadioButton(mode, self.holder)
                    rb.setAutoExclusive(False)
                    self._output_mode_radios[mode] = rb
        if state["sub"]:
            self._update_output_mode_sub_settings = lambda mode: self.calls.append(("sub", mode))
        if state["combo"]:
            self._output_mode_combo = QComboBox(self.holder)
            self._output_mode_combo.addItems(["T", "V", "I", "Vi", "A", "R"])
            self._output_mode_combo.setCurrentIndex(state["combo_index"])
            self._output_mode_combo.currentIndexChanged.connect(lambda i: self.calls.append(("combo", i)))

    def snapshot(self):
        attrs = {name: getattr(self, name, _ABSENT) for name in (
            "enable_image_output_mode_var", "enable_video_output_mode_var", "enable_audio_output_mode_var",
            "enable_refinement_output_mode_var", "enable_image_translation_var", "output_mode_var")}
        radios = {m: rb.isChecked() for m, rb in getattr(self, "_output_mode_radios", {}).items()}
        combo = getattr(self, "_output_mode_combo", None)
        return {"attrs": attrs, "config": copy.deepcopy(self.config), "env": _env_snapshot(), "radios": radios,
                "combo": None if combo is None else combo.currentIndex(), "calls": list(self.calls)}


def test_set_output_mode_matches_legacy(qapp, other_settings_mod, legacy_profile_functions):
    _text, nodes = legacy_profile_functions
    legacy_fn = _bind_legacy(nodes, other_settings_mod, {})["_set_output_mode"]
    new_fn = other_settings_mod._set_output_mode
    rng = random.Random(f"{SEED}:output_mode")
    for index in range(STATES):
        state = {"config_none": rng.random() < 0.05, "toggle": rng.choice(["ok", "ok", "raise", "absent"]),
                 "radios": rng.choice(["all", "all", "some", None]), "sub": rng.random() < 0.8,
                 "combo": rng.random() < 0.8, "combo_index": rng.randint(0, 5)}
        modes = [rng.choice(MODE_INPUTS) for _ in range(rng.randint(1, 3))]
        traces = []
        for fn in (legacy_fn, new_fn):
            with _env_guard():
                owner = OutputOwner(fn, state)
                traces.append([(_run(lambda: owner._set_output_mode(m)), owner.snapshot()) for m in modes])
                owner.holder.deleteLater()
        _compare({"t": traces[0]}, {"t": traces[1]}, f"_set_output_mode state {index} {state} {modes}")


class _Vis:
    def __init__(self, checked=False):
        self.visible = None
        self.checked = checked

    def setVisible(self, value):
        self.visible = bool(value)

    def isChecked(self):
        return self.checked


SUB_PARAMS = ("rb_image", "rb_video", "rb_vision", "img_sub", "vid_sub", "vision_sub", "vision_only_widgets",
              "vision_batch_row", "vision_batch_desc")


def test_output_mode_sub_settings_closure_matches_legacy(other_settings_mod):
    legacy_src = _nested_function(_git_text("src/other_settings.py"), None, "_create_image_translation_section",
                                  "_update_output_mode_sub_settings", "other_settings.py")
    new_src = _nested_function((SRC / "other_settings.py").read_text(encoding="utf-8").replace("\r\n", "\n"), None,
                               "_create_image_translation_section", "_update_output_mode_sub_settings",
                               "other_settings.py")
    assert "settings_rules.output_mode_sub_settings" in new_src
    factories = [_closure_factory(src, SUB_PARAMS, vars(other_settings_mod), "_update_output_mode_sub_settings")
                 for src in (legacy_src, new_src)]
    rng = random.Random(f"{SEED}:sub_settings")
    for index in range(STATES):
        radios = [rng.random() < 0.3 for _ in range(3)]
        mode = rng.choice((None,) + MODE_INPUTS[:-1])
        results = []
        for factory in factories:
            widgets = {name: _Vis(radios[i] if i < 3 else False) for i, name in enumerate(SUB_PARAMS)
                       if name != "vision_only_widgets"}
            only = tuple(_Vis() for _ in range(6))
            closure = factory(*[only if n == "vision_only_widgets" else widgets[n] for n in SUB_PARAMS])
            outcome = _run(lambda: closure(mode))
            results.append((outcome, {n: w.visible for n, w in widgets.items()}, [w.visible for w in only]))
        assert results[0] == results[1], (index, mode, radios, results)


# =============================================================================================
# 4. Glossary Manager mode locks (update_auto_glossary_state)
# =============================================================================================

GM_LABELS = ("Off", "Off (Fuzzy Mapping)", "Manual Glossary Only", "No Glossary", "Minimal", "Balanced", "Full",
             "Single Pass")


class GlossaryOwner:
    """The Glossary Manager surface the mode-lock pass touches (General + Minimal tabs)."""

    TOGGLES = (("append_glossary_checkbox", "append_glossary", "Append Glossary to System Prompt"),
               ("append_glossary_auto_load_checkbox", "append_glossary_auto_load", "Auto-Mapping (Auto-Fill)"),
               ("fuzzy_auto_mapping_checkbox", "fuzzy_auto_mapping", "Fuzzy Auto-Mapping"))

    def __init__(self, state):
        self.config = {"glossary_x": 1}
        self.calls = []
        self.logs = []
        self.holder = QWidget()
        if state["combo"]:
            self.auto_glossary_mode_combo = QComboBox(self.holder)
            self.auto_glossary_mode_combo.addItems(list(GM_LABELS) + ["Off (No Auto-Mapping)", "Weird Mode"])
            self.auto_glossary_mode_combo.setCurrentIndex(state["mode_index"])
        self.auto_prompt_text = QTextEdit(self.holder)
        for attr, key, text in self.TOGGLES:
            if not state["toggles"][key]:
                continue
            parent = QWidget(self.holder)
            cb = QCheckBox(text, parent)
            cb.setChecked(state["checked"][key])
            if state["pre_locked"][key]:
                cb._mode_locked = True
                cb._original_text = text
            setattr(self, attr, cb)
            if key == "fuzzy_auto_mapping" and state["fuzzy_hint"]:
                QLabel("Map similar names automatically", parent)
            if state["vars"][key]:
                setattr(self, f"{key}_var", state["checked"][key])
            cb.toggled.connect(lambda checked, k=key: self._on_toggle(k, checked))
        for attr, present in (("_auto_load_desc_label", state["auto_load_label"]),
                              ("_append_glossary_desc_label", state["append_label"])):
            if present:
                setattr(self, attr, QLabel("hint", self.holder))
        if state["slider"]:
            slider_parent = QWidget(self.holder)
            self.fuzzy_mapping_slider = QSlider(slider_parent)
            QLabel("Similarity threshold", slider_parent)
            self.fuzzy_mapping_value_label = QLabel("50%", slider_parent)
        self._sources = state["sources"]

    def _on_toggle(self, key, checked):
        self.calls.append(("toggled", key, checked))
        self.config[key] = checked

    # path-switching collaborators (recorded)
    def _glossary_editor_input_sources(self):
        return list(self._sources)

    def auto_load_glossary_for_file(self, path):
        self.calls.append(("auto_load", path))

    def _autofill_glossary_for_current_selection(self):
        self.calls.append(("autofill",))

    def append_log(self, message):
        self.logs.append(message)

    def snapshot(self):
        out = {"config": dict(self.config), "calls": list(self.calls), "logs": list(self.logs),
               "prompt_enabled": self.auto_prompt_text.isEnabled()}
        for attr, key, _text in self.TOGGLES:
            cb = getattr(self, attr, None)
            out[key] = None if cb is None else (cb.text(), cb.isChecked(), cb.isEnabled(), cb.styleSheet(),
                                                getattr(cb, "_mode_locked", _ABSENT))
            out[key + "_var"] = getattr(self, f"{key}_var", _ABSENT)
            hints = [] if cb is None else [(lbl.text(), lbl.styleSheet(), getattr(lbl, "_mode_locked", _ABSENT))
                                          for lbl in cb.parentWidget().findChildren(QLabel)]
            out[key + "_hints"] = hints
        for attr in ("_auto_load_desc_label", "_append_glossary_desc_label"):
            lbl = getattr(self, attr, None)
            out[attr] = None if lbl is None else (lbl.styleSheet(), getattr(lbl, "_mode_locked", _ABSENT))
        slider = getattr(self, "fuzzy_mapping_slider", None)
        if slider is not None:
            out["slider"] = (slider.isEnabled(), slider.styleSheet(), getattr(slider, "_mode_locked", _ABSENT),
                             self.fuzzy_mapping_value_label.styleSheet(),
                             [(c.text(), c.styleSheet(), getattr(c, "_mode_locked", _ABSENT))
                              for c in slider.parentWidget().findChildren(QLabel)])
        return out

    def click_hints(self):
        """Click every hint label whose handler the pass replaced (lock: no-op, unlock: toggle)."""
        labels = [getattr(self, a, None) for a in ("_auto_load_desc_label", "_append_glossary_desc_label")]
        cb = getattr(self, "fuzzy_auto_mapping_checkbox", None)
        if cb is not None:
            labels += cb.parentWidget().findChildren(QLabel)
        for lbl in labels:
            handler = getattr(lbl, "__dict__", {}).get("mousePressEvent") if lbl is not None else None
            if handler is not None:
                handler(None)


class _Grid:
    def __init__(self, widgets):
        self.widgets = widgets

    def count(self):
        return len(self.widgets)

    def itemAt(self, i):
        return types.SimpleNamespace(widget=lambda w=self.widgets[i]: w)


def test_glossary_mode_lock_pass_matches_legacy(qapp):
    import GlossaryManager_GUI as gm

    legacy_src = _nested_function(_git_text("src/GlossaryManager_GUI.py"), "GlossaryManagerMixin",
                                  "_setup_auto_glossary_tab", "update_auto_glossary_state", "GlossaryManager_GUI.py")
    new_src = _nested_function((SRC / "GlossaryManager_GUI.py").read_text(encoding="utf-8").replace("\r\n", "\n"),
                               "GlossaryManagerMixin", "_setup_auto_glossary_tab", "update_auto_glossary_state",
                               "GlossaryManager_GUI.py")
    assert "settings_rules.glossary_mode_toggle_steps(mode)" in new_src
    params = ("self", "settings_label_frame", "extraction_grid")
    factories = [_closure_factory(src, params, vars(gm), "update_auto_glossary_state") for src in (legacy_src, new_src)]
    rng = random.Random(f"{SEED}:glossary_mode")
    keys = ("append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping")
    for index in range(STATES):
        state = {
            "combo": rng.random() < 0.95, "mode_index": rng.randint(0, len(GM_LABELS) + 1),
            "toggles": {k: rng.random() < 0.9 for k in keys}, "checked": {k: rng.random() < 0.5 for k in keys},
            "pre_locked": {k: rng.random() < 0.3 for k in keys}, "vars": {k: rng.random() < 0.7 for k in keys},
            "fuzzy_hint": rng.random() < 0.8, "auto_load_label": rng.random() < 0.8,
            "append_label": rng.random() < 0.8, "slider": rng.random() < 0.85,
            "sources": rng.choice([[], [], ["book.epub"], ["a.epub", "b.epub"]]),
        }
        passes = rng.randint(1, 2)
        results = []
        for factory in factories:
            owner = GlossaryOwner(state)
            frame = QWidget(owner.holder)
            grid_widgets = [QWidget(frame) for _ in range(3)]
            for widget in grid_widgets:
                QLabel("child", widget)
            closure = factory(owner, frame, _Grid(grid_widgets))
            trace = []
            for _ in range(passes):
                outcome = _run(closure)
                snap = owner.snapshot()
                snap["frame"] = frame.isEnabled()
                snap["grid"] = [(w.isEnabled(), [c.isEnabled() for c in w.findChildren(QWidget)]) for w in grid_widgets]
                trace.append((outcome, snap))
                owner.click_hints()
                trace.append(("clicked", owner.snapshot()))
            results.append(trace)
            owner.holder.deleteLater()
        _compare({"t": results[0]}, {"t": results[1]}, f"update_auto_glossary_state state {index} {state}")
        if index % 50 == 0:
            qapp.processEvents()


# =============================================================================================
# 5. TranslatorGUI handlers rewired by the U4 review fixes
# =============================================================================================

@functools.lru_cache(maxsize=None)
def _tg_methods(side):
    """{name: FunctionDef} of TranslatorGUI at U4_BASE_SHA ("legacy") or in the working tree ("new")."""
    if side == "legacy":
        text = _git_text("src/translator_gui.py")
    else:
        text = (SRC / "translator_gui.py").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    tree = ast.parse(text, filename="translator_gui.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def _tg_function(side, name, namespace):
    ns = dict(namespace)
    exec(compile(ast.Module(body=[_tg_methods(side)[name]], type_ignores=[]), f"<translator_gui {side}>", "exec"), ns)
    return ns[name]


# ---- 5a. "Assistant Prompt" dialog (show_assistant_prompt_dialog -> prompt_profiles.prefill_*) -------

PREFILL_TYPED = ("Default", "Alpha", "Beta", "New Profile #1", "New Profile #2", "  Alpha  ", "default", "",
                 "   ", "Gamma", "DEFAULT", " Spaced ")


def _assistant_dialog_class(side):
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtWidgets import QMainWindow

    from _src_corpus import find_method

    gm_path = SRC / "GlossaryManager_GUI.py"
    tree = ast.parse(gm_path.read_text(encoding="utf-8-sig"))
    gm = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlossaryManagerMixin")
    helpers = [n for n in gm.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_disable_combobox_mousewheel", "_apply_halgakos_combo_icons")]
    ns = {}
    for module in (QtCore, QtGui, QtWidgets):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("_")})
    ns.update({"os": os, "__file__": str(gm_path)})
    body = ([_tg_methods(side)["show_assistant_prompt_dialog"]]
            + [find_method(n) for n in ("_add_combobox_arrow", "_count_assistant_prompt_tokens")] + helpers)
    exec(compile(ast.Module(body=body, type_ignores=[]), f"<assistant dialog {side}>", "exec"), ns)

    class AssistantOwner(QMainWindow):
        def __init__(self, config, prompt):
            super().__init__()
            self.config = copy.deepcopy(config)
            self.assistant_prompt = prompt
            self.logs = []
            self.fail_save = False

        def save_config(self, show_message=True):
            self.logs.append(("save_config", show_message, self.assistant_prompt))
            if self.fail_save:
                return False
            self.config["assistant_prompt"] = self.assistant_prompt
            return None

        def append_log(self, message):
            self.logs.append(message)

        def _update_assistant_prompt_button_style(self):
            self.logs.append(("button_style", self.assistant_prompt))

    for name in ("show_assistant_prompt_dialog", "_add_combobox_arrow", "_count_assistant_prompt_tokens",
                 "_disable_combobox_mousewheel", "_apply_halgakos_combo_icons"):
        setattr(AssistantOwner, name, ns[name])
    return AssistantOwner


def _random_prefill_config(rng):
    config = {"unrelated": "kept"}
    roll = rng.random()
    if roll < 0.8:
        profiles = {name: f"{name} saved" for name in ("Alpha", "Beta", "New Profile #1", "Gamma")
                    if rng.random() < 0.5}
        if rng.random() < 0.15:
            profiles["Default"] = "ignored"
        if rng.random() < 0.1:
            profiles["  "] = "blank name"
        if rng.random() < 0.1:
            profiles["Delta"] = 5                    # non-text values are dropped
        if rng.random() < 0.15:
            profiles[" Spaced "] = "spaced name"
        config["assistant_prompt_profiles"] = profiles
    elif roll < 0.88:
        config["assistant_prompt_profiles"] = ["not", "a", "dict"]
    if rng.random() < 0.6:
        config["assistant_prompt_profile_default"] = rng.choice(["Default saved", "", None])
    if rng.random() < 0.8:
        config["assistant_prompt"] = rng.choice(["", "Live prompt", "Alpha saved", "  padded  "])
    if rng.random() < 0.75:
        config["active_assistant_prompt_profile"] = rng.choice(["Alpha", "Beta", "Gone", "", 7, " Spaced "])
    prompt = config.get("assistant_prompt", "") if rng.random() < 0.9 else rng.choice([None, "Runtime only"])
    return config, prompt


def _assistant_ops(rng):
    ops = []
    for _ in range(rng.randint(3, 10)):
        op = rng.choice(["new", "select", "activate", "type", "edit", "rename", "save", "delete", "delete",
                         "clear", "rename_save"])
        ops.append((op, rng.randint(0, 99), rng.random() < 0.7, rng.random() < 0.12,
                    rng.choice(["edited", "  spaced draft  ", "", "line1\nline2", "Alpha saved"]),
                    rng.choice(PREFILL_TYPED)))
    end = rng.choice([None, None, "Save", "Cancel"])
    if end:
        ops.append((end, 0, False, rng.random() < 0.15, "", ""))
    return ops


def _assistant_snapshot(window, boxes):
    from PySide6.QtWidgets import QComboBox, QLabel, QTextEdit

    dialog = window._assistant_prompt_dialog
    snap = {"config": json.dumps(window.config, sort_keys=True, default=repr), "prompt": window.assistant_prompt,
            "logs": json.dumps(window.logs, default=repr), "boxes": list(boxes.log), "open": dialog is not None}
    if dialog is not None:
        combo = dialog.findChildren(QComboBox)[0]
        editor = dialog.findChildren(QTextEdit)[0]
        snap.update(items=[combo.itemText(i) for i in range(combo.count())], index=combo.currentIndex(),
                    text=combo.currentText(), editor=editor.toPlainText(),
                    tokens=[label.text() for label in dialog.findChildren(QLabel) if label.text().startswith("Tokens")])
    return snap


def _apply_assistant_op(window, op, boxes):
    from PySide6.QtWidgets import QComboBox, QPushButton, QTextEdit

    kind, value, yes, fail, text, typed = op
    dialog = window._assistant_prompt_dialog
    if dialog is None:
        return ("closed",)
    window.fail_save = fail
    combo = dialog.findChildren(QComboBox)[0]
    editor = dialog.findChildren(QTextEdit)[0]

    def click(label):
        button = next(b for b in dialog.findChildren(QPushButton) if b.text() == label)
        return _run(button.click)

    if kind == "new":
        return click("+ New Profile")
    if kind == "select":
        return _run(lambda: combo.setCurrentIndex(value % combo.count()))
    if kind == "activate":
        return _run(lambda: combo.activated.emit(value % combo.count()))
    if kind == "type":
        combo.setEditText(typed)
        return _run(combo.lineEdit().returnPressed.emit)
    if kind == "edit":
        return _run(lambda: editor.setPlainText(text))
    if kind == "rename":
        return _run(lambda: combo.setEditText(typed))
    if kind == "rename_save":
        combo.setEditText(typed)
        return click("💾 Save Profile")
    if kind == "save":
        return click("💾 Save Profile")
    if kind == "delete":
        boxes.answers = [QMessageBox.Yes if yes else QMessageBox.No]
        return click("🗑 Delete Profile")
    if kind == "clear":
        return click("Clear")
    return click(kind)  # "Save" / "Cancel" close the dialog


def test_assistant_prompt_dialog_matches_legacy(qapp, monkeypatch):
    assert "prompt_profiles.prefill_" in ast.unparse(_tg_methods("new")["show_assistant_prompt_dialog"])
    boxes = Boxes()
    boxes.install(monkeypatch)
    classes = {side: _assistant_dialog_class(side) for side in ("legacy", "new")}
    rng = random.Random(f"{SEED}:assistant_dialog")
    counts = {"states": 0, "ops": 0, "saved": 0, "deleted": 0, "boxes": 0}
    for index in range(STATES):
        config, prompt = _random_prefill_config(rng)
        ops = _assistant_ops(rng)
        traces = []
        for side in ("legacy", "new"):
            boxes.log, boxes.answers = [], []
            window = classes[side](config, prompt)
            trace = [("open", _run(window.show_assistant_prompt_dialog), _assistant_snapshot(window, boxes))]
            for op in ops:
                trace.append((op[0], _apply_assistant_op(window, op, boxes), _assistant_snapshot(window, boxes)))
            traces.append(trace)
            if window._assistant_prompt_dialog is not None:
                window._assistant_prompt_dialog.reject()
            window.deleteLater()
        for step, (left, right) in enumerate(zip(*traces)):
            _compare({"op": left[0], "outcome": left[1], **left[2]}, {"op": right[0], "outcome": right[1], **right[2]},
                     f"assistant dialog state {index} step {step} config={config} prompt={prompt!r} ops={ops}")
        last = traces[0][-1][2]
        counts["states"] += 1
        counts["ops"] += len(ops)
        counts["saved"] += "Saved assistant prompt profile" in last["logs"]
        counts["deleted"] += "Deleted assistant prompt profile" in last["logs"]
        counts["boxes"] += bool(last["boxes"])
        if index % 25 == 0:
            qapp.processEvents()
    qapp.processEvents()
    assert counts["states"] >= STATES
    # the random walk reaches the save, delete and message-box branches
    assert min(counts["saved"], counts["deleted"], counts["boxes"]) >= counts["states"] // 20, counts


# ---- 5b. _quick_new_profile -> prompt_profiles.new_profile -----------------------------------------------

def test_quick_new_profile_matches_legacy(qapp, tmp_path, monkeypatch, other_settings_mod, profile_sides):
    _nodes, auto_save, mixin = profile_sides
    assert "new_profile(" in ast.unparse(_tg_methods("new")["_quick_new_profile"])
    boxes = Boxes()
    boxes.install(monkeypatch)
    try:  # save_profiles beeps on Windows
        import winsound
        monkeypatch.setattr(winsound, "MessageBeep", lambda *_a: None)
    except ImportError:
        pass
    quick = {side: _tg_function(side, "_quick_new_profile", {"os": os}) for side in ("legacy", "new")}
    functions = {name: getattr(other_settings_mod, name) for name in PROFILE_FUNCTIONS}
    rng = random.Random(f"{SEED}:quick_new_profile")

    def failing_open(*args, **kwargs):
        raise OSError("simulated disk error")

    for index in range(STATES):
        state = _random_profile_state(rng)
        if rng.random() < 0.3:
            state["profiles"]["New Profile #1"] = ""
        ops = [rng.choice(("select_combo", "type_name", "edit_text")) for _ in range(rng.randint(0, 2))]
        ops += ["quick_new"] * rng.randint(1, 2) + [rng.choice(("edit_text", "select_combo", "save_profiles"))]
        values = [(rng.random() < 0.6, rng.randint(0, 50)) for _ in ops]
        note = f"state {index}"
        results = []
        for side in ("legacy", "new"):
            cfg_file = tmp_path / side / "config.json"
            cfg_file.parent.mkdir(exist_ok=True)
            monkeypatch.setattr(other_settings_mod, "CONFIG_FILE", str(cfg_file))
            _write_config_file(cfg_file, state, note)
            boxes.log, boxes.answers = [], []
            if state["open_fails"]:
                monkeypatch.setattr(other_settings_mod, "open", failing_open, raising=False)
            owner = ProfileOwner(functions, state, boxes, auto_save, mixin)
            owner._quick_new_profile = types.MethodType(quick[side], owner)
            trace = []
            for op, value in zip(ops, values):
                if op == "quick_new":
                    outcome = _run(owner._quick_new_profile)
                else:
                    outcome = _apply_profile_op(owner, op, value, boxes, cfg_file.parent / "import.json",
                                                cfg_file.parent)
                snap = owner.snapshot(cfg_file)
                snap["file"] = None if snap["file"] is None else snap["file"].decode("utf-8", "replace")
                trace.append((op, outcome, snap))
            results.append(_normalize_dir(trace, cfg_file.parent))
            monkeypatch.delattr(other_settings_mod, "open", raising=False)
            timer = getattr(owner, "_save_profile_timer", None)
            if timer is not None:
                timer.stop()
            owner.holder.deleteLater()
        for step, (left, right) in enumerate(zip(*results)):
            _compare({"op": left[0], "outcome": left[1], **left[2]}, {"op": right[0], "outcome": right[1], **right[2]},
                     f"_quick_new_profile {note} step {step} ops={ops} state={state}")
        if index % 50 == 0:
            qapp.processEvents()


# ---- 5c. _fetch_authgem_projects worker -> authgem_auth.list_gcp_projects ---------------------------------

_PROJECTS_URL = "https://cloudresourcemanager.googleapis.com/v1/projects"


class _FakeResponse:
    def __init__(self, ok, payload=None, bad_json=False):
        self.ok = ok
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class _FakeRequests(types.ModuleType):
    """``requests`` stand-in answering from a per-state table; records every call (thread-safe)."""

    def __init__(self, table):
        super().__init__("requests")
        self.table = table
        self.calls = []
        self._lock = threading.Lock()

    def get(self, url, headers=None, params=None, timeout=None):
        with self._lock:
            self.calls.append((url, json.dumps(headers, sort_keys=True), json.dumps(params, sort_keys=True), timeout))
        kind, payload = self.table.get(url, ("fail", None))
        if kind == "raise":
            raise ConnectionError(f"network down: {url}")
        if kind == "bad_json":
            return _FakeResponse(True, bad_json=True)
        return _FakeResponse(kind == "ok", payload)


def _random_projects_table(rng):
    pids = [f"proj-{i}" for i in range(rng.randint(0, 7))]
    projects = [{"projectId": pid, "name": pid.upper()} for pid in pids]
    if rng.random() < 0.15:
        projects.append({"name": "no id"})
    if rng.random() < 0.05:
        projects = [{"name": "no id"}, {"projectId": ""}]       # listed, but nothing usable
    listing = rng.choice([("ok", {"projects": projects})] * 6 + [("ok", {}), ("fail", None), ("bad_json", None),
                                                                  ("raise", None)])
    table = {_PROJECTS_URL: listing}
    for pid in pids:
        billing = f"https://cloudbilling.googleapis.com/v1/projects/{pid}/billingInfo"
        vertex = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{pid}/locations/us-central1"
        table[billing] = rng.choice([("ok", {"billingEnabled": True}), ("ok", {"billingEnabled": False}), ("ok", {}),
                                     ("fail", None), ("fail", None), ("raise", None), ("bad_json", None)])
        table[vertex] = rng.choice([("ok", {}), ("fail", None), ("raise", None)])
    return table


class _FakeMeta:
    """QMetaObject stand-in: records each queued ``_authgem_projects_loaded`` with the lists it would show."""

    def __init__(self):
        self.published = []

    def invokeMethod(self, obj, name, connection):
        self.published.append((name, connection, sorted(obj._authgem_billed_projects),
                               sorted(obj._authgem_unbilled_projects), sorted(obj._authgem_unknown_projects)))


class _CaptureLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def test_fetch_authgem_projects_matches_legacy(monkeypatch):
    import authgem_auth  # imported before requests is faked (it binds the module at import)

    assert "list_gcp_projects(" in ast.unparse(_tg_methods("new")["_fetch_authgem_projects"])
    capture = _CaptureLog()
    logger = logging.getLogger("translator_gui")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(capture)
    rng = random.Random(f"{SEED}:authgem_projects")
    counts = {"states": 0, "phases2": 0}
    try:
        for index in range(STATES):
            table = _random_projects_table(rng)
            token = rng.choice(["tok"] * 5 + ["raise"])
            guard = rng.choice([None, False, False, False, False, True])
            results = []
            for side in ("legacy", "new"):
                fake = _FakeRequests(table)
                monkeypatch.setitem(sys.modules, "requests", fake)
                meta = _FakeMeta()
                capture.records = []
                fetch = _tg_function(side, "_fetch_authgem_projects", {
                    "QMetaObject": meta, "Qt": types.SimpleNamespace(QueuedConnection="queued"),
                    "__name__": "translator_gui"})
                owner = types.SimpleNamespace()
                if guard is not None:
                    owner._authgem_fetching_projects = guard

                def store(_token=token):
                    def get_valid_access_token(auto_login=True):
                        if _token == "raise":
                            raise RuntimeError("Not logged in")
                        return f"{_token}-{auto_login}"
                    return types.SimpleNamespace(get_valid_access_token=get_valid_access_token)

                owner._get_authgem_store_for_current_model = store
                outcome = _run(lambda: fetch(owner))
                deadline = time.monotonic() + 20
                while guard is not True and getattr(owner, "_authgem_fetching_projects", False):
                    if time.monotonic() > deadline:
                        raise AssertionError(f"{side}: the worker did not finish")
                    time.sleep(0.001)
                results.append({
                    "outcome": outcome, "published": meta.published, "calls": sorted(fake.calls),
                    "guard": getattr(owner, "_authgem_fetching_projects", _ABSENT),
                    "lists": [sorted(getattr(owner, a, [])) for a in ("_authgem_billed_projects",
                                                                      "_authgem_unbilled_projects",
                                                                      "_authgem_unknown_projects")],
                    "log": list(capture.records),
                })
            _compare(results[0], results[1], f"_fetch_authgem_projects state {index} token={token} guard={guard} "
                                             f"table={table}")
            counts["states"] += 1
            counts["phases2"] += len(results[0]["published"]) == 2
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
    assert counts["phases2"] >= counts["states"] // 5, counts
    # the shared function on its own: phase-1 callback, then the checked lists
    fake = _FakeRequests({_PROJECTS_URL: ("ok", {"projects": [{"projectId": "a"}, {"projectId": "b"}]}),
                          "https://cloudbilling.googleapis.com/v1/projects/a/billingInfo": ("ok", {"billingEnabled": True}),
                          "https://cloudbilling.googleapis.com/v1/projects/b/billingInfo": ("fail", None),
                          "https://us-central1-aiplatform.googleapis.com/v1/projects/b/locations/us-central1":
                              ("fail", None)})
    listed = []
    assert authgem_auth.list_gcp_projects("t", on_listed=lambda *lists: listed.append(lists), http=fake) == \
        (["a"], ["b"], [])
    assert listed == [([], [], ["a", "b"])]
    assert authgem_auth.list_gcp_projects("t", http=_FakeRequests({_PROJECTS_URL: ("fail", None)})) is None


# ---- 5d. Reset Settings to Defaults (other_settings closure) -> config_store.reset_preserved_keys ------

_RESET_KEYS = (
    "api_key", "multi_api_keys", "fallback_keys", "glossary_keys", "glossary_refinement_keys", "metadata_keys",
    "qa_scan_keys", "ai_truncation_detection_keys", "rolling_summary_keys", "truncation_retry_keys",
    "inpainter_keys", "tts_keys", "replicate_api_key", "model", "azure_vision_key", "azure_vision_endpoint",
    "azure_document_intelligence_key", "azure_document_intelligence_endpoint", "google_vision_credentials",
    "google_cloud_credentials", "prompt_profiles", "active_profile", "use_multi_api_keys", "use_fallback_keys",
    "use_glossary_keys", "use_glossary_refinement_keys", "use_metadata_keys", "use_qa_scan_keys",
    "use_ai_truncation_detection_keys", "use_rolling_summary_keys", "use_truncation_retry_keys",
    "use_inpainter_keys", "use_tts_keys",
)


class _SettingsDialog(QWidget):
    def __init__(self, calls):
        super().__init__()
        self._calls = calls

    def close(self):
        self._calls.append("settings_dialog.close")
        return super().close()


def _random_reset_config(rng):
    values = ["", "sk-live-1234", None, 0, True, False, ["a", {"api_key": "k", "model": "m"}], {"nested": [1, 2]}, "★"]
    config = {key: copy.deepcopy(rng.choice(values)) for key in _RESET_KEYS if rng.random() < 0.45}
    for key in ("temperature", "batch_size", "openai_base_url", "custom_model_list"):
        if rng.random() < 0.5:
            config[key] = rng.choice(values)
    roll = rng.random()
    if roll < 0.35:
        config["qa_scanner_settings"] = {"excluded_characters": rng.choice(["★", "", None, ["x"]]), "min_file_length": 9}
    elif roll < 0.5:
        config["qa_scanner_settings"] = {"min_file_length": 2}
    elif roll < 0.6:
        config["qa_scanner_settings"] = rng.choice(["broken", None, ["excluded_characters"]])
    items = list(config.items())
    rng.shuffle(items)
    return dict(items)


def test_reset_to_defaults_matches_legacy(qapp, tmp_path, monkeypatch, other_settings_mod):
    legacy_src = _nested_function(_git_text("src/other_settings.py"), None, "_create_danger_zone_section",
                                  "_reset_config_to_defaults", "other_settings.py")
    new_src = _nested_function((SRC / "other_settings.py").read_text(encoding="utf-8").replace("\r\n", "\n"), None,
                               "_create_danger_zone_section", "_reset_config_to_defaults", "other_settings.py")
    assert "reset_preserved_keys(current_config)" in new_src and "RESET_PRESERVED_TEXT" in new_src
    boxes = Boxes()
    boxes.install(monkeypatch)
    calls = []
    # the Yes branch restarts the app: Popen and sys.exit only record here
    monkeypatch.setattr(subprocess, "Popen", lambda args, *a, **k: calls.append(("Popen", args == [sys.executable] + sys.argv)))
    monkeypatch.setattr(sys, "exit", lambda code=0: calls.append(("exit", code)))
    factories = {}
    for side, src in (("legacy", legacy_src), ("new", new_src)):
        folder = tmp_path / side
        folder.mkdir()
        overrides = {"_get_app_dir": (lambda d=str(folder): d), "CONFIG_FILE": "config.json"}
        factories[side] = (_closure_factory(src, ("self",), {**vars(other_settings_mod), **overrides},
                                            "_reset_config_to_defaults"), folder / "config.json")
    rng = random.Random(f"{SEED}:reset_defaults")
    counts = {"states": 0, "written": 0}
    for index in range(STATES):
        config = _random_reset_config(rng)
        shape = rng.choice(["dict"] * 9 + ["none", "missing"])
        dialog_kind = rng.choice(["widget", "widget", "none", "absent"])
        answer = QMessageBox.Yes if rng.random() < 0.6 else QMessageBox.No
        file_exists = rng.random() < 0.85
        results = []
        for side in ("legacy", "new"):
            factory, path = factories[side]
            if path.exists():
                path.unlink()
            if file_exists:
                path.write_text('{"old": true}', encoding="utf-8")
            boxes.log, boxes.answers = [], [answer]
            calls.clear()
            owner = types.SimpleNamespace()
            if shape == "dict":
                owner.config = copy.deepcopy(config)
            elif shape == "none":
                owner.config = None
            if dialog_kind == "widget":
                owner._other_settings_dialog = _SettingsDialog(calls)
            elif dialog_kind == "none":
                owner._other_settings_dialog = None
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                outcome = _run(factory(owner))
            results.append({"outcome": outcome, "stdout": out.getvalue(), "boxes": list(boxes.log),
                            "calls": list(calls), "file": path.read_text(encoding="utf-8") if path.exists() else None,
                            "config": json.dumps(getattr(owner, "config", _ABSENT), sort_keys=True, default=repr)})
            dialog = getattr(owner, "_other_settings_dialog", None)
            if dialog is not None:
                dialog.deleteLater()
        _compare(results[0], results[1], f"_reset_config_to_defaults state {index} shape={shape} dialog={dialog_kind} "
                                         f"config={config}")
        counts["states"] += 1
        counts["written"] += results[0]["file"] not in (None, '{"old": true}')
        if index % 50 == 0:
            qapp.processEvents()
    assert counts["written"] >= counts["states"] // 5, counts


# =============================================================================================
# harness self-test: a rare-branch change in the shared code must be caught
# =============================================================================================

def _inject(module_name, attr, wrap):
    import importlib

    module = importlib.import_module(module_name)
    return module, attr, wrap(getattr(module, attr))


def _no_gamma_extraction(original):
    def injected(name):
        return None if "gamma" in name.lower() else original(name)
    return injected


def _drop_wheel_lock(original):
    def injected(prompt_profiles, config, active_profile):
        data = original(prompt_profiles, config, active_profile)
        data.pop("profile_mousewheel_locked", None)
        return data
    return injected


def _minimal_keeps_automap(original):
    def injected(mode):
        steps = original(mode)
        return steps[:-1] if mode == "minimal" else steps
    return injected


def _thoughts_env_text(original):
    def injected(stream_thinking_on):
        enabled, env_value, locked = original(stream_thinking_on)
        return enabled, ("true" if env_value == "1" else env_value), locked
    return injected


def _audio_translates_images(original):
    def injected(mode, config=None):
        flags = original(mode, config)
        if flags.mode == "audio":
            flags.var_values["enable_image_translation_var"] = True
        return flags
    return injected


def _vision_only_for_image(original):
    def injected(mode):
        out = original(mode)
        if mode == "image":
            out["vision_only"] = True
        return out
    return injected


def _beta_delete_selects_default(original):
    def injected(state, name, **kwargs):
        outcome = original(state, name, **kwargs)
        if outcome.ok and name.strip() == "Beta":
            state.active_name = ""
        return outcome
    return injected


def _second_new_profile_drops_autosave(original):
    def injected(state, config=None, **kwargs):
        name = original(state, config, **kwargs)
        if name == "New Profile #2":
            state._active_profile_for_autosave = None
        return name
    return injected


def _one_unbilled_reported_unknown(original):
    def injected(access_token, **kwargs):
        result = original(access_token, **kwargs)
        if result and len(result[1]) >= 2:
            billed, unbilled, unknown = result
            return billed, unbilled[1:], unknown + unbilled[:1]
        return result
    return injected


def _reset_drops_tts_toggle(original):
    def injected(current_config):
        kept = original(current_config)
        kept.pop("use_tts_keys", None)
        return kept
    return injected


INJECTIONS = {
    "on_profile_select": ("prompt_profiles", "extraction_method_for_profile", _no_gamma_extraction),
    "save_profiles": ("prompt_profiles", "_profile_file_updates", _drop_wheel_lock),
    "glossary_mode": ("settings_rules", "glossary_mode_toggle_steps", _minimal_keeps_automap),
    "thoughts": ("settings_rules", "thoughts_lock_state", _thoughts_env_text),
    "output_mode": ("settings_rules", "output_mode_flags", _audio_translates_images),
    "sub_settings": ("settings_rules", "output_mode_sub_settings", _vision_only_for_image),
    "assistant_dialog": ("prompt_profiles", "prefill_delete", _beta_delete_selects_default),
    "quick_new_profile": ("prompt_profiles", "new_profile", _second_new_profile_drops_autosave),
    "authgem_projects": ("authgem_auth", "list_gcp_projects", _one_unbilled_reported_unknown),
    "reset_defaults": ("config_store", "reset_preserved_keys", _reset_drops_tts_toggle),
}


@pytest.mark.parametrize("name", sorted(INJECTIONS))
def test_harness_catches_injected_differences(request, qapp, tmp_path, monkeypatch, other_settings_mod,
                                              legacy_profile_functions, profile_sides, name):
    module, attr, injected = _inject(*INJECTIONS[name])
    monkeypatch.setattr(module, attr, injected)
    monkeypatch.setattr(sys.modules[__name__], "STATES", min(STATES, 150))
    with pytest.raises(AssertionError):
        if name in PROFILE_FUNCTIONS:
            test_profile_actions_match_legacy(qapp, tmp_path, monkeypatch, other_settings_mod, profile_sides, name)
        elif name == "glossary_mode":
            test_glossary_mode_lock_pass_matches_legacy(qapp)
        elif name == "thoughts":
            test_thoughts_lock_matches_legacy(qapp, other_settings_mod, legacy_profile_functions)
        elif name == "output_mode":
            test_set_output_mode_matches_legacy(qapp, other_settings_mod, legacy_profile_functions)
        elif name == "assistant_dialog":
            test_assistant_prompt_dialog_matches_legacy(qapp, monkeypatch)
        elif name == "quick_new_profile":
            test_quick_new_profile_matches_legacy(qapp, tmp_path, monkeypatch, other_settings_mod, profile_sides)
        elif name == "authgem_projects":
            test_fetch_authgem_projects_matches_legacy(monkeypatch)
        elif name == "reset_defaults":
            test_reset_to_defaults_matches_legacy(qapp, tmp_path, monkeypatch, other_settings_mod)
        else:
            test_output_mode_sub_settings_closure_matches_legacy(other_settings_mod)
