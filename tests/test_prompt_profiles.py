"""prompt_profiles: the shared GUI-free core of the prompt-profile actions (U4).

What is checked:

* hygiene: GUI-free (PySide6 blocked), cheap import, Python 3.10 syntax;
* ``profile_state_from_config`` gives the profiles / active profile / extraction method the
  desktop start-up gives (HeadlessOwner replays it) and never touches the config;
* the functions without widgets (the mobile path) end in the state the desktop handlers
  (other_settings, with their Qt combo box, editor and message boxes) end in, for select /
  save (rename, built-in copy, errors) / delete / reset / import / export / save_profiles;
* ``new_profile`` matches TranslatorGUI._quick_new_profile;
* the assistant prefill functions match the real "Assistant Prompt" dialog
  (TranslatorGUI.show_assistant_prompt_dialog) over random operation sequences: drafts, the
  combo text, message boxes and every persisted config value;
* the config.json writer only changes the four profile keys (ENC: secrets kept as read).

tests/parity/test_u4_dialog_parity.py proves the desktop handlers themselves unchanged.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_prompt_profiles.py
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prompt_profiles as pp  # noqa: E402

HAS_QT = importlib.util.find_spec("PySide6") is not None
needs_qt = pytest.mark.skipif(not HAS_QT, reason="needs PySide6")
#: Desktop handlers raising inside Qt slots are part of the behaviour under comparison.
pytestmark = pytest.mark.qt_no_exception_capture


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------

def test_module_is_gui_free_cheap_and_python_310():
    source = (SRC / "prompt_profiles.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import prompt_profiles as pp; "
        "heavy = [m for m in ('owner_state', 'translator_gui', 'dpi_setup', 'other_settings') if m in sys.modules]; "
        "state = pp.profile_state_from_config({'prompt_profiles': {'Mine': 'x'}, 'active_profile': 'Mine'}); "
        "pp.save_profile(state, 'Mine2', 'text'); "
        "print(heavy, list(state.prompt_profiles)[-1], state.profile_var, 'translator_gui' in sys.modules)" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[] Mine2 Mine2 False"


# ---------------------------------------------------------------------------
# start-up state
# ---------------------------------------------------------------------------

START_CONFIGS = (
    {},
    {"prompt_profiles": {"Mine": "m", "Universal": "u"}, "active_profile": "Mine"},
    {"prompt_profiles": {"Mine": "m"}, "active_profile": "Gone", "extraction_mode": "enhanced"},
    {"prompt_profiles": {"Korean_html2text": "k", "Universal": "u", "Refinement": "r"},
     "text_extraction_method": "enhanced"},
    {"active_profile": "Korean_BeautifulSoup"},
)


@pytest.mark.parametrize("index", range(len(START_CONFIGS)))
def test_profile_state_matches_the_desktop_start_up(tmp_path, monkeypatch, index):
    from _headless_env import headless_owner

    config = copy.deepcopy(START_CONFIGS[index])
    snapshot = copy.deepcopy(config)
    state = pp.profile_state_from_config(config)
    assert config == snapshot, "profile_state_from_config must not touch the config"
    with headless_owner(tmp_path, monkeypatch, copy.deepcopy(START_CONFIGS[index])) as owner:
        expected = (list(owner.prompt_profiles.items()), owner.profile_var, owner.text_extraction_method_var,
                    dict(owner.default_prompts))
    assert (list(state.prompt_profiles.items()), state.profile_var, state.text_extraction_method_var,
            state.default_prompts) == expected
    assert state.config is config
    assert "Universal" in state.protected and "Mine" not in state.protected


def test_built_in_reset_and_protection_use_the_desktop_helpers():
    state = pp.profile_state_from_config({"prompt_profiles": {"Universal": "edited", "Mine": "m"},
                                          "active_profile": "Mine"})
    default = state.default_prompts["Universal"]
    outcome = pp.delete_or_reset_profile(state, "Universal")
    assert outcome.kind == "reset" and state.prompt_profiles["Universal"] == default
    # the desktop refreshes the selection to the reset profile (on_profile_select)
    assert state.profile_var == "Universal" and state.config["active_profile"] == "Universal"
    assert pp.delete_or_reset_profile(state, "Nope").error == "Profile 'Nope' not found."


# ---------------------------------------------------------------------------
# desktop handlers (Qt) vs the same functions without widgets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    if not HAS_QT:
        pytest.skip("needs PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _Boxes:
    def __init__(self, monkeypatch):
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        self.log = []
        self.answers = []
        self.open_path = self.save_path = ""

        def static(kind):
            def show(*args, **kwargs):
                self.log.append((kind,) + tuple(a for a in args if isinstance(a, str)))
                return QMessageBox.Ok
            return show

        def exec_(box):
            answer = self.answers.pop(0) if self.answers else QMessageBox.No
            self.log.append(("exec", box.windowTitle(), box.text()))
            return answer

        for kind in ("critical", "warning", "information"):
            monkeypatch.setattr(QMessageBox, kind, staticmethod(static(kind)))
        monkeypatch.setattr(QMessageBox, "exec", exec_)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (self.open_path, "")))
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (self.save_path, "")))


def _desktop_owner(config, boxes):
    """A TranslatorGUI profile surface: other_settings handlers, real autosave, Qt widgets."""
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QComboBox, QTextEdit, QWidget

    import other_settings
    from _src_corpus import find_method
    from owner_state import ConfigStateMixin

    ns = {"os": os}
    exec(compile(ast.Module(body=[find_method("_auto_save_system_prompt")], type_ignores=[]), "<tg>", "exec"), ns)

    class Owner(QObject):
        pass

    owner = Owner()
    owner.boxes = boxes
    owner.config = config
    holder = pp.profile_state_from_config(config)
    owner.prompt_profiles = holder.prompt_profiles
    owner.profile_var = holder.profile_var
    owner.default_prompts = holder.default_prompts
    owner.text_extraction_method_var = holder.text_extraction_method_var
    owner._active_profile_for_autosave = holder.profile_var
    for name in ("on_profile_select", "save_profile", "delete_profile", "save_profiles", "import_profiles",
                 "export_profiles"):
        setattr(owner, name, types.MethodType(getattr(other_settings, name), owner))
    owner._auto_save_system_prompt = types.MethodType(ns["_auto_save_system_prompt"], owner)
    owner._get_protected_prompt_profiles = types.MethodType(ConfigStateMixin._get_protected_prompt_profiles, owner)
    owner._reset_prompt_profile_to_default = types.MethodType(
        ConfigStateMixin._reset_prompt_profile_to_default, owner)
    owner.logs = []
    owner.append_log = owner.logs.append
    owner._update_profile_delete_button_label = lambda name=None: None
    owner.holder = QWidget()
    owner.profile_menu = QComboBox(owner.holder)
    owner.profile_menu.setEditable(True)
    owner.profile_menu.addItems(list(owner.prompt_profiles))
    owner.profile_menu.setCurrentIndex(max(0, owner.profile_menu.findText(owner.profile_var)))
    owner.profile_menu.currentIndexChanged.connect(lambda: owner.on_profile_select())
    owner.prompt_text = QTextEdit(owner.holder)
    owner.prompt_text.setPlainText(owner.prompt_profiles.get(owner.profile_var, ""))
    owner.prompt_text.textChanged.connect(owner._auto_save_system_prompt)
    return owner


PROFILE_KEYS = ("prompt_profiles", "active_profile", "text_extraction_method")


def _profile_view(owner_or_state):
    config = owner_or_state.config
    return {
        "profiles": list(owner_or_state.prompt_profiles.items()),
        "active": owner_or_state.profile_var,
        "method": owner_or_state.text_extraction_method_var,
        "config": {k: copy.deepcopy(config[k]) for k in PROFILE_KEYS if k in config},
    }


START = {"prompt_profiles": {"Universal": "U", "Refinement": "R", "Korean_BeautifulSoup": "KBS",
                             "Japanese_html2text": "JH", "Alpha": "A", "Beta": "B"},
         "active_profile": "Alpha", "unrelated": 1}


def _pick(owner, name):
    """Choose a profile in the combo box (re-choosing the current one re-runs the handler)."""
    index = owner.profile_menu.findText(name)
    if index == owner.profile_menu.currentIndex():
        owner.profile_menu.setEditText(name)
        owner.on_profile_select()
    else:
        owner.profile_menu.setCurrentIndex(index)


def _desktop_steps(owner, steps, boxes):
    for op, *args in steps:
        if op == "select":
            _pick(owner, args[0])
        elif op == "save":
            source, name, content = args
            _pick(owner, source)
            owner.prompt_text.setPlainText(content)
            owner.profile_menu.setEditText(name)
            owner.save_profile()
        elif op == "delete":
            _pick(owner, args[0])
            from PySide6.QtWidgets import QMessageBox
            boxes.answers = [QMessageBox.Yes]
            owner.delete_profile()
        elif op == "import":
            path, = args
            boxes.open_path = str(path)
            owner.import_profiles()


def _core_steps(state, steps):
    errors = []
    for op, *args in steps:
        if op == "select":
            pp.select_profile(state, args[0])
        elif op == "save":
            source, name, content = args
            pp.select_profile(state, source)
            outcome = pp.save_profile(state, name, content, source_name=source)
            if outcome.error:
                errors.append((outcome.title, outcome.error))
        elif op == "delete":
            outcome = pp.delete_or_reset_profile(state, args[0])
            if outcome.error:
                errors.append((outcome.title, outcome.error))
        elif op == "import":
            pp.merge_imported_profiles(state, pp.read_profiles_file(args[0]))
    return errors


SCENARIOS = {
    "select_extraction_profiles": [("select", "Korean_BeautifulSoup"), ("select", "Japanese_html2text"),
                                   ("select", "Beta")],
    "rename_custom_keeps_order": [("save", "Alpha", "Renamed", "  new text  ")],
    "built_in_saved_under_new_name_is_a_copy": [("save", "Universal", "My Universal", "custom")],
    # rejected saves: the desktop editor keeps the edit staged in memory (prompt autosave),
    # so only the errors and what reaches config.json are compared
    "rename_onto_existing_is_rejected": [("save", "Alpha", "Beta", "x")],
    "empty_name_is_rejected": [("save", "Alpha", "   ", "x")],
    "delete_custom_selects_first": [("delete", "Beta"), ("delete", "Alpha")],
    "reset_built_in": [("save", "Universal", "Universal", "edited"), ("delete", "Universal")],
    "import_then_select": [("import", "<file>"), ("select", "Imported")],
    "mixed": [("select", "Beta"), ("save", "Beta", "Beta 2", "b2"), ("delete", "Beta 2"),
              ("save", "Korean_BeautifulSoup", "KBS copy", "copy"), ("select", "KBS copy")],
}


@needs_qt
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_core_without_widgets_matches_the_desktop_handlers(qapp, tmp_path, monkeypatch, name):
    import other_settings

    boxes = _Boxes(monkeypatch)
    monkeypatch.setattr(other_settings, "CONFIG_FILE", str(tmp_path / "config.json"))
    import_file = tmp_path / "import.json"
    import_file.write_text(json.dumps({"Imported": "imp", "Alpha": "A2"}), encoding="utf-8")
    steps = [tuple(str(import_file) if a == "<file>" else a for a in step) for step in SCENARIOS[name]]

    owner = _desktop_owner(copy.deepcopy(START), boxes)
    _desktop_steps(owner, steps, boxes)
    desktop = _profile_view(owner)
    desktop_errors = [entry[1:] for entry in boxes.log if entry[0] in ("critical", "warning")]

    state = pp.profile_state_from_config(copy.deepcopy(START))
    core_errors = _core_steps(state, steps)
    if not name.endswith("_rejected"):
        assert _profile_view(state) == desktop
    assert core_errors == desktop_errors
    # the persisted key set equals other_settings.save_profiles' file write
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) if (tmp_path / "config.json").exists() else None
    if on_disk is not None:
        updates = pp.profiles_config_updates(state)
        assert {k: on_disk[k] for k in updates} == json.loads(json.dumps(updates))
    owner.holder.deleteLater()


@needs_qt
def test_new_profile_matches_quick_new_profile(qapp, monkeypatch, tmp_path):
    import other_settings
    from _src_corpus import find_method

    boxes = _Boxes(monkeypatch)
    monkeypatch.setattr(other_settings, "CONFIG_FILE", str(tmp_path / "config.json"))
    ns = {"os": os}
    exec(compile(ast.Module(body=[find_method("_quick_new_profile")], type_ignores=[]), "<tg>", "exec"), ns)
    start = copy.deepcopy(START)
    start["prompt_profiles"]["New Profile #1"] = ""
    owner = _desktop_owner(copy.deepcopy(start), boxes)
    owner.profile_menu.addItem("New Profile #2")      # a combo-only name is skipped too
    owner._quick_new_profile = types.MethodType(ns["_quick_new_profile"], owner)
    owner._quick_new_profile()
    state = pp.profile_state_from_config(copy.deepcopy(start))
    name = pp.new_profile(state, existing_names=["New Profile #2"])
    assert name == "New Profile #3" == owner.profile_var
    assert _profile_view(state) == _profile_view(owner)
    assert state._original_profile_content[name] == owner._original_profile_content[name] == ""
    owner.holder.deleteLater()


def test_config_file_writer_changes_only_the_profile_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"api_key": "ENC:abc", "prompt_profiles": {"Old": "o"}, "z": [1, 2]}),
                    encoding="utf-8")
    state = pp.ProfileState({"profile_mousewheel_locked": False}, {"Ünïcode": "テキスト"}, "Ünïcode")
    assert pp.write_profiles_to_config_file(str(path), state.prompt_profiles, state.profile_var, state.config)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"api_key": "ENC:abc", "prompt_profiles": {"Ünïcode": "テキスト"}, "z": [1, 2],
                    "profile_name_autofill": True, "profile_mousewheel_locked": False, "active_profile": "Ünïcode"}
    assert list(data)[:3] == ["api_key", "prompt_profiles", "z"]
    assert "テキスト" in path.read_text(encoding="utf-8")       # ensure_ascii=False, as desktop
    assert pp.apply_profiles_to_config(state) == {
        "prompt_profiles": {"Ünïcode": "テキスト"}, "profile_name_autofill": True,
        "profile_mousewheel_locked": False, "active_profile": "Ünïcode"}
    assert pp.export_profiles_json({"a": "é"}) == json.dumps({"a": "é"}, ensure_ascii=False, indent=2)
    assert pp.json_export_path("x") == "x.json" and pp.json_export_path("x.json") == "x.json"


# ---------------------------------------------------------------------------
# assistant prefill: the real dialog vs the prefill functions
# ---------------------------------------------------------------------------

def _dialog_harness_class():
    from PySide6.QtWidgets import QMainWindow

    from _src_corpus import find_method

    path = SRC / "GlossaryManager_GUI.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    gm = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlossaryManagerMixin")
    gm_methods = [n for n in gm.body if isinstance(n, ast.FunctionDef)
                  and n.name in ("_disable_combobox_mousewheel", "_apply_halgakos_combo_icons")]
    from PySide6 import QtCore, QtGui, QtWidgets
    ns = {}
    for module in (QtCore, QtGui, QtWidgets):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("_")})
    ns.update({"os": os, "__file__": str(path)})
    body = [find_method(n) for n in ("show_assistant_prompt_dialog", "_add_combobox_arrow",
                                     "_count_assistant_prompt_tokens")] + gm_methods
    exec(compile(ast.Module(body=body, type_ignores=[]), "<dialog>", "exec"), ns)

    class Harness(QMainWindow):
        def __init__(self, config):
            super().__init__()
            self.config = copy.deepcopy(config)
            self.assistant_prompt = self.config.get("assistant_prompt", "")
            self.logs = []
            self.fail_save = False

        def save_config(self, show_message=True):
            if self.fail_save:
                return False
            self.config["assistant_prompt"] = self.assistant_prompt
            return None

        def append_log(self, message):
            self.logs.append(message)

        def _update_assistant_prompt_button_style(self):
            pass

    for name in ("show_assistant_prompt_dialog", "_add_combobox_arrow", "_count_assistant_prompt_tokens",
                 "_disable_combobox_mousewheel", "_apply_halgakos_combo_icons"):
        setattr(Harness, name, ns[name])
    return Harness


PREFILL_NAMES = ("Default", "Alpha", "Beta", "New Profile #1", "  Alpha  ", "default", "", "Gamma")


def _prefill_start(rng):
    config = {}
    if rng.random() < 0.8:
        config["assistant_prompt_profiles"] = {n: f"{n} saved" for n in ("Alpha", "Beta") if rng.random() < 0.7}
        if rng.random() < 0.2:
            config["assistant_prompt_profiles"]["Default"] = "ignored"
            config["assistant_prompt_profiles"]["  "] = "ignored"
    if rng.random() < 0.7:
        config["assistant_prompt_profile_default"] = "Default saved"
    if rng.random() < 0.8:
        config["assistant_prompt"] = rng.choice(["", "Live prompt", "Alpha saved"])
    if rng.random() < 0.7:
        config["active_assistant_prompt_profile"] = rng.choice(["Alpha", "Beta", "Gone", ""])
    return config


@needs_qt
def test_prefill_functions_match_the_assistant_prompt_dialog(qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QMessageBox, QPushButton, QTextEdit

    log = []
    answers = []

    def warning(*args, **kwargs):
        log.append(tuple(a for a in args if isinstance(a, str)))
        return QMessageBox.Ok

    def exec_(box):
        return answers.pop(0) if answers else QMessageBox.No

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(warning))
    monkeypatch.setattr(QMessageBox, "exec", exec_)
    harness_class = _dialog_harness_class()
    rng = random.Random(77)
    for round_index in range(50):        # 600 dialog operations
        config = _prefill_start(rng)
        window = harness_class(config)
        window.show_assistant_prompt_dialog()
        state = pp.prefill_state_from_config(config, window.assistant_prompt)
        core_config = copy.deepcopy(config)
        combo_text = state.active_name or pp.PREFILL_DEFAULT_NAME
        core_log = []

        def widgets():
            dialog = window._assistant_prompt_dialog
            return dialog, dialog.findChildren(QComboBox)[0], dialog.findChildren(QTextEdit)[0]

        def click(label):
            dialog = window._assistant_prompt_dialog
            next(b for b in dialog.findChildren(QPushButton) if b.text() == label).click()

        def persist(ok):
            if ok:
                core_config.update(pp.prefill_config_updates(state))
            else:
                core_log.append(("Save Failed", "Could not save the assistant prompt profiles. Your edits are "
                                                "still open; please try saving again."))

        history = []
        for step in range(12):
            op = rng.choice(["new", "select", "type", "edit", "rename", "save", "delete", "clear"])
            history.append(op)
            fail = rng.random() < 0.1
            window.fail_save = fail
            dialog, combo, editor = widgets()
            if op == "new":
                click("+ New Profile")
                pp.prefill_new(state)
                combo_text = state.active_name
                persist(not fail)
            elif op == "select":
                index = rng.randrange(combo.count())
                before = combo.currentIndex()
                combo.setCurrentIndex(index)
                if index != before:            # currentIndexChanged -> the dialog's select_profile
                    pp.prefill_select(state, combo.itemText(index))
                    combo_text = state.active_name or pp.PREFILL_DEFAULT_NAME
                else:                          # no signal; Qt shows the item text again
                    combo_text = combo.itemText(index)
            elif op == "type":
                name = rng.choice(PREFILL_NAMES)
                combo.setEditText(name)
                combo.lineEdit().returnPressed.emit()
                combo_text = name
                if pp.prefill_select(state, name) is not None:
                    combo_text = state.active_name or pp.PREFILL_DEFAULT_NAME
            elif op == "edit":
                text = rng.choice(["edited", "  spaced draft  ", "", "line1\nline2"])
                editor.setPlainText(text)
                pp.prefill_stage(state, combo_text, text)
            elif op == "rename":
                combo_text = rng.choice(PREFILL_NAMES)
                combo.setEditText(combo_text)
            elif op == "save":
                click("💾 Save Profile")
                outcome = pp.prefill_save(state, combo_text, state.text)
                if outcome.ok:
                    combo_text = state.active_name or pp.PREFILL_DEFAULT_NAME
                    persist(not fail)
                else:
                    core_log.append((outcome.title, outcome.error))
            elif op == "delete":
                answers[:] = [QMessageBox.Yes if rng.random() < 0.7 else QMessageBox.No]
                yes = answers[0] == QMessageBox.Yes
                click("🗑 Delete Profile")
                name = combo_text.strip()
                if name.casefold() == "default" or not name or name not in state.profiles:
                    core_log.append((pp.prefill_delete(state, name).title, pp.prefill_delete(state, name).error))
                elif yes:
                    pp.prefill_delete(state, name)
                    combo_text = state.active_name or pp.PREFILL_DEFAULT_NAME
                    persist(not fail)
            elif op == "clear":
                click("Clear")
                pp.prefill_stage(state, combo_text, "")
            dialog, combo, editor = widgets()
            where = (round_index, step, history, config, state)
            assert [combo.itemText(i) for i in range(combo.count())] == state.options, where
            assert combo.currentText() == combo_text, where
            assert editor.toPlainText().strip() == state.text.strip(), where
            assert log == core_log, where
            assert {k: window.config.get(k) for k in pp.PREFILL_CONFIG_KEYS} == \
                {k: core_config.get(k) for k in pp.PREFILL_CONFIG_KEYS}, where
        window._assistant_prompt_dialog.reject()
        window.deleteLater()
        log.clear()
        qapp.processEvents()


def test_prefill_contract_used_by_the_mobile_screens():
    state = pp.prefill_state_from_config({"assistant_prompt": "legacy prefill"})
    assert (state.profiles, state.default_prompt, state.active_name, state.text) == ({}, "legacy prefill", "",
                                                                                     "legacy prefill")
    assert pp.prefill_new(state) == "New Profile #1" and state.options == ["Default", "New Profile #1"]
    assert pp.prefill_save(state, "Style", "  be terse ").ok and list(state.profiles) == ["Style"]
    assert pp.prefill_save(state, "Default", "x").error == "Default is reserved. Choose another profile name."
    assert pp.prefill_select(state, "default") == "legacy prefill" and state.active_name == ""
    assert pp.prefill_delete(state, "Default").error.startswith("The Default assistant prompt profile")
    config = {}
    assert pp.prefill_apply_to_config(state, config, "typed") == config == {
        "assistant_prompt_profiles": {"Style": "be terse"}, "assistant_prompt_profile_default": "legacy prefill",
        "active_assistant_prompt_profile": "", "assistant_prompt": "typed"}
