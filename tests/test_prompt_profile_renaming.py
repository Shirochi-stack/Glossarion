"""Profile renames through the glossary controls and runtime-bound main saver."""

import ast
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QCompleter, QListWidget, QMainWindow, QMessageBox, QPushButton, QTextEdit

import other_settings
from GlossaryManager_GUI import GlossaryManagerMixin


class ProfileHarness(GlossaryManagerMixin, QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.logs = []
        self.config = {
            "unrelated_setting": "preserved",
            "prompt_profiles": {"Universal": "Built-in", "Alpha": "Alpha saved", "Beta": "Beta saved"},
            "active_profile": "Alpha",
            "glossary_prompt_profiles": {
                key: {"Alpha": f"{key} Alpha", "Beta": f"{key} Beta"}
                for key in ("balanced_full", "minimal")
            },
            "active_glossary_prompt_profiles": {"balanced_full": "Alpha", "minimal": "Alpha"},
            "glossary_prompt_profile_defaults": {"balanced_full": "Full default", "minimal": "Minimal default"},
        }
        self.prompt_profiles = self.config["prompt_profiles"]
        self.profile_var = "Alpha"
        self._active_profile_for_autosave = "Alpha"
        self._original_profile_content = dict(self.prompt_profiles)
        other_settings.setup_other_settings_methods(self)

        self.profile_menu = QComboBox(self)
        self.profile_menu.setEditable(True)
        self.profile_menu.addItems(list(self.prompt_profiles))
        self.profile_menu.setCurrentIndex(self.profile_menu.findText("Alpha"))
        self._apply_profile_name_autofill()
        self.prompt_text = QTextEdit(self)
        self.profile_menu.currentIndexChanged.connect(lambda _: self.on_profile_select())
        self.prompt_text.textChanged.connect(self._auto_save_system_prompt)
        self.on_profile_select()
        self.save_main_button = QPushButton("Save Profile", self)
        self.save_main_button.clicked.connect(self.save_profile)

        self.rows = {}
        for key in ("balanced_full", "minimal"):
            editor = QTextEdit(self)
            row = self._create_glossary_prompt_profile_controls(key, editor)
            row.setParent(self)
            self.rows[key] = row
            self._apply_active_glossary_prompt_profile(key)
        self.save_config()

    def save_config(self, show_message=False):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        return True

    def append_log(self, text):
        self.logs.append(text)

    def _update_profile_delete_button_label(self, name=None):
        pass


# Only omit TranslatorGUI's unrelated application startup; these are the real
# autosave, quick-create, and built-in protection methods used by its saver.
source_path = Path(__file__).resolve().parents[1] / "src" / "translator_gui.py"
tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
gui_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
method_names = {
    "_auto_save_system_prompt", "_quick_new_profile", "_get_protected_prompt_profiles",
    "_apply_profile_name_autofill", "_open_profile_manager", "_save_profile_order",
    "_get_list_order", "_restore_list_order", "_move_profile_in_list",
}
methods = [n for n in gui_class.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
namespace = {"Qt": Qt, "os": os}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), "exec"), namespace)
for method_name in method_names:
    setattr(ProfileHarness, method_name, namespace[method_name])


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: QMessageBox.Ok)
    path = tmp_path / "config.json"
    monkeypatch.setattr(other_settings, "CONFIG_FILE", str(path))
    window = ProfileHarness(path)
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def saved(gui):
    return json.loads(gui.path.read_text(encoding="utf-8"))


def save_glossary(gui, key):
    button = next(b for b in gui.rows[key].findChildren(QPushButton) if b.text() == "💾 Save Profile")
    button.click()


def manager_controls(gui):
    gui._open_profile_manager()
    dialog = gui._profile_manager_dialog
    checkbox = dialog.findChild(QCheckBox, "profile_name_autofill_checkbox")
    buttons = {b.text(): b for b in dialog.findChildren(QPushButton)}
    return dialog, checkbox, buttons


def type_profile_name(gui, text):
    gui.show()
    gui.activateWindow()
    gui.profile_menu.show()
    gui.profile_menu.setFocus()
    QApplication.processEvents()
    editor = gui.profile_menu.lineEdit()
    editor.selectAll()
    QTest.keyClicks(editor, text)
    return editor


def test_main_autofill_off_by_default_and_cancel_preserves_it(gui):
    editor = type_profile_name(gui, "al")
    assert editor.text() == "al"
    assert not editor.hasSelectedText()
    dialog, checkbox, buttons = manager_controls(gui)
    assert not checkbox.isChecked()
    checkbox.setChecked(True)
    buttons["Cancel"].click()
    assert gui.profile_menu.completer() is None
    assert not saved(gui).get("profile_name_autofill", False)


def test_main_autofill_opt_in_persists_and_can_be_disabled(gui):
    dialog, checkbox, buttons = manager_controls(gui)
    checkbox.setChecked(True)
    buttons["Save Changes"].click()
    assert saved(gui)["profile_name_autofill"] is True
    assert gui.profile_menu.completer().completionMode() == QCompleter.InlineCompletion
    editor = type_profile_name(gui, "al")
    assert editor.text().lower() == "alpha"
    assert editor.selectedText().lower() == "pha"

    # Simulate rebuilding the field after loading the saved configuration.
    gui.config = saved(gui)
    gui.profile_menu.setCompleter(None)
    gui._apply_profile_name_autofill()
    assert type_profile_name(gui, "be").text().lower() == "beta"
    dialog, checkbox, buttons = manager_controls(gui)
    assert checkbox.isChecked()
    checkbox.setChecked(False)
    buttons["Save Changes"].click()
    assert saved(gui)["profile_name_autofill"] is False
    assert type_profile_name(gui, "al").text() == "al"


def test_main_autofill_tracks_renamed_and_reordered_profiles(gui):
    dialog, checkbox, buttons = manager_controls(gui)
    checkbox.setChecked(True)
    profiles = dialog.findChild(QListWidget)
    gui._restore_list_order(profiles, ["Beta", "Alpha", "Universal"])
    buttons["Save Changes"].click()
    assert gui.profile_menu.currentText() == "Alpha"
    assert gui.profile_menu.currentIndex() == 1
    gui.profile_menu.setEditText("Renamed")
    gui.save_main_button.click()
    assert list(saved(gui)["prompt_profiles"]) == ["Beta", "Renamed", "Universal"]
    assert type_profile_name(gui, "re").text().lower() == "renamed"
    assert type_profile_name(gui, "al").text() == "al"


def test_failed_manager_save_keeps_autofill_setting_and_allows_retry(gui, monkeypatch):
    before = saved(gui)
    dialog, checkbox, buttons = manager_controls(gui)
    checkbox.setChecked(True)
    profiles = dialog.findChild(QListWidget)
    gui._restore_list_order(profiles, ["Beta", "Alpha", "Universal"])
    with monkeypatch.context() as patch:
        patch.setattr(gui, "save_profiles", lambda: False)
        buttons["Save Changes"].click()
    assert dialog.isVisible()
    assert gui.profile_menu.completer() is None
    assert not gui.config.get("profile_name_autofill", False)
    assert list(gui.prompt_profiles) == ["Universal", "Alpha", "Beta"]
    assert saved(gui) == before
    buttons["Save Changes"].click()
    assert saved(gui)["profile_name_autofill"] is True
    assert list(saved(gui)["prompt_profiles"]) == ["Beta", "Alpha", "Universal"]


@pytest.mark.parametrize("edit_before_name", [True, False])
def test_main_rename_preserves_order_content_and_active_state(gui, edit_before_name):
    if edit_before_name:
        gui.prompt_text.setPlainText("Renamed content")
    gui.profile_menu.setEditText("  Renamed  ")
    if not edit_before_name:
        gui.prompt_text.setPlainText("Renamed content")
    gui.save_main_button.click()
    assert list(gui.prompt_profiles) == ["Universal", "Renamed", "Beta"]
    assert "Alpha" not in gui._original_profile_content
    assert gui._original_profile_content["Renamed"] == "Renamed content"
    assert gui.profile_var == gui._active_profile_for_autosave == "Renamed"
    assert gui.profile_menu.currentIndex() == gui.profile_menu.findText("Renamed")
    assert saved(gui)["active_profile"] == "Renamed"

    gui.prompt_text.setPlainText("Saved after rename")
    assert gui.prompt_profiles["Renamed"] == "Saved after rename"
    gui.save_main_button.click()
    gui.profile_menu.setCurrentIndex(gui.profile_menu.findText("Beta"))
    gui.profile_menu.setCurrentIndex(gui.profile_menu.findText("Renamed"))
    assert gui.prompt_text.toPlainText() == "Saved after rename"
    gui.save_config()
    assert "Alpha" not in saved(gui)["prompt_profiles"]
    assert saved(gui)["active_profile"] == "Renamed"
    assert saved(gui)["unrelated_setting"] == "preserved"


@pytest.mark.parametrize("name", ["Beta", "Universal"])
def test_main_rename_rejects_existing_destination(gui, name):
    before = saved(gui)
    gui.profile_menu.setEditText(name)
    gui.prompt_text.setPlainText("Do not overwrite")
    gui.save_main_button.click()
    assert saved(gui) == before
    assert gui.prompt_profiles[name] == before["prompt_profiles"][name]
    assert gui.profile_var == "Alpha"


def test_main_builtin_can_be_copied_without_removing_required_profile(gui):
    gui.profile_menu.setCurrentIndex(gui.profile_menu.findText("Universal"))
    gui.prompt_text.setPlainText("Custom built-in copy")
    gui.profile_menu.setEditText("My Universal")
    gui.save_main_button.click()
    assert gui.prompt_profiles["Universal"] == "Built-in"
    assert gui.prompt_profiles["My Universal"] == "Custom built-in copy"
    assert gui.profile_var == "My Universal"


def test_main_new_profile_renames_and_repeated_renames_do_not_accumulate(gui):
    gui._quick_new_profile()
    for name in ("First name", "Second name"):
        gui.profile_menu.setEditText(name)
        gui.save_main_button.click()
        assert list(gui.prompt_profiles) == ["Universal", "Alpha", "Beta", name]
        assert gui._active_profile_for_autosave == name


def test_main_failed_disk_save_restores_rename_state_for_retry(gui, monkeypatch):
    before = saved(gui)
    gui.profile_menu.setEditText("Renamed")
    with monkeypatch.context() as patch:
        def fail_open(*args, **kwargs):
            raise OSError("simulated disk error")
        patch.setattr(other_settings, "open", fail_open, raising=False)
        gui.save_main_button.click()
    assert saved(gui) == before
    assert list(gui.prompt_profiles) == ["Universal", "Alpha", "Beta"]
    assert gui.profile_var == gui._active_profile_for_autosave == "Alpha"
    assert "Renamed" not in gui._original_profile_content
    gui.save_main_button.click()
    assert list(saved(gui)["prompt_profiles"]) == ["Universal", "Renamed", "Beta"]


@pytest.mark.parametrize("key", ["balanced_full", "minimal"])
@pytest.mark.parametrize("edit_before_name", [True, False])
def test_glossary_rename_keeps_bucket_order_and_prompt(gui, key, edit_before_name):
    other_key = "minimal" if key == "balanced_full" else "balanced_full"
    before_other = copy.deepcopy(gui.config["glossary_prompt_profiles"][other_key])
    widgets = gui._glossary_prompt_profile_widgets[key]
    if edit_before_name:
        widgets["editor"].setPlainText("New glossary prompt")
    widgets["combo"].setEditText("  Renamed  ")
    if not edit_before_name:
        widgets["editor"].setPlainText("New glossary prompt")
    save_glossary(gui, key)
    assert list(gui.config["glossary_prompt_profiles"][key]) == ["Renamed", "Beta"]
    assert gui.config["glossary_prompt_profiles"][key]["Renamed"] == "New glossary prompt"
    assert gui.config["active_glossary_prompt_profiles"][key] == "Renamed"
    assert gui.config["glossary_prompt_profiles"][other_key] == before_other
    meta = gui._glossary_prompt_profile_meta(key)
    assert saved(gui)[meta["config_key"]] == "New glossary prompt"
    gui.config = saved(gui)
    gui._apply_active_glossary_prompt_profile(key)
    assert widgets["combo"].currentText() == "Renamed"
    assert widgets["combo"].findText("Alpha") == -1
    assert widgets["editor"].toPlainText() == "New glossary prompt"


@pytest.mark.parametrize("key", ["balanced_full", "minimal"])
@pytest.mark.parametrize("name", ["Beta", "dEfAuLt"])
def test_glossary_rename_preserves_other_profiles_and_default(gui, key, name):
    before = saved(gui)
    widgets = gui._glossary_prompt_profile_widgets[key]
    widgets["combo"].setEditText(name)
    widgets["editor"].setPlainText("Do not overwrite")
    save_glossary(gui, key)
    assert saved(gui) == before
    assert gui.config["glossary_prompt_profile_defaults"] == before["glossary_prompt_profile_defaults"]
    assert gui.config["glossary_prompt_profiles"][key]["Beta"] == before["glossary_prompt_profiles"][key]["Beta"]


@pytest.mark.parametrize("key", ["balanced_full", "minimal"])
def test_glossary_failed_rename_can_retry_without_duplicate(gui, key, monkeypatch):
    before = saved(gui)
    widgets = gui._glossary_prompt_profile_widgets[key]
    widgets["combo"].setEditText("Renamed")
    with monkeypatch.context() as patch:
        patch.setattr(gui, "save_config", lambda **kwargs: False)
        save_glossary(gui, key)
    assert saved(gui) == before
    assert list(gui.config["glossary_prompt_profiles"][key]) == ["Alpha", "Beta"]
    assert gui.config["active_glossary_prompt_profiles"][key] == "Alpha"
    save_glossary(gui, key)
    assert list(saved(gui)["glossary_prompt_profiles"][key]) == ["Renamed", "Beta"]
