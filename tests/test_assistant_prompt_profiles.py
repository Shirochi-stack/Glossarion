"""Exercise the real assistant prompt dialog without booting the translator app."""

import ast
import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)


SRC = Path(__file__).resolve().parents[1] / "src"


def _production_methods(filename, class_name, names):
    """Load actual methods while avoiding app imports and their startup side effects."""
    path = SRC / filename
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {method.name for method in methods} == set(names)
    namespace = dict(globals(), __file__=str(path))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _PromptHarness(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.assistant_prompt = self.config.get("assistant_prompt", "")
        self.saved_configs = []
        self.logs = []
        self.style_updates = 0
        self.base_dir = str(SRC)
        self.assistant_prompt_button = QPushButton("Asst. Prompt", self)
        self.assistant_prompt_button.clicked.connect(self.show_assistant_prompt_dialog)

    def save_config(self, show_message=True):
        self.config["assistant_prompt"] = self.assistant_prompt
        self.saved_configs.append(copy.deepcopy(self.config))

    def append_log(self, message):
        self.logs.append(message)

    def _update_assistant_prompt_button_style(self):
        self.style_updates += 1


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module", autouse=True)
def _bind_production_methods():
    for name, method in _production_methods(
        "translator_gui.py", "TranslatorGUI",
        ["show_assistant_prompt_dialog", "_add_combobox_arrow"],
    ).items():
        setattr(_PromptHarness, name, method)
    for name, method in _production_methods(
        "GlossaryManager_GUI.py",
        "GlossaryManagerMixin",
        ["_disable_combobox_mousewheel", "_apply_halgakos_combo_icons"],
    ).items():
        setattr(_PromptHarness, name, method)


@pytest.fixture(params=["class_helpers", "runtime_helpers"])
def make_harness(qapp, monkeypatch, request):
    windows = []
    # Profile feedback must not start a modal event loop in a headless test.
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)

    def make(config=None):
        window = _PromptHarness(config or {})
        if request.param == "runtime_helpers":
            from other_settings import setup_other_settings_methods
            setup_other_settings_methods(window)
        windows.append(window)
        window.assistant_prompt_button.click()
        qapp.processEvents()
        return window

    yield make

    for window in windows:
        if getattr(window, "_assistant_prompt_dialog", None) is not None:
            window._assistant_prompt_dialog.reject()
        window.close()
        window.deleteLater()
    qapp.processEvents()


def _widgets(window):
    dialog = window._assistant_prompt_dialog
    combos = dialog.findChildren(QComboBox)
    editors = dialog.findChildren(QTextEdit)
    assert len(combos) == len(editors) == 1
    return dialog, combos[0], editors[0]


def _click(window, text):
    buttons = [
        button for button in window._assistant_prompt_dialog.findChildren(QPushButton)
        if button.text() == text
    ]
    assert len(buttons) == 1, text
    buttons[0].click()


def _select(combo, name):
    index = combo.findText(name)
    assert index >= 0, name
    combo.setCurrentIndex(index)


def _assert_count(window, expected):
    labels = [
        label.text() for label in window._assistant_prompt_dialog.findChildren(QLabel)
        if label.text().startswith("Characters:")
    ]
    assert labels == [f"Characters: {expected}"]


def _profiles_config():
    return {
        "assistant_prompt": "Alpha saved",
        "assistant_prompt_profiles": {"Alpha": "Alpha saved", "Beta": "Beta saved"},
        "active_assistant_prompt_profile": "Alpha",
        "assistant_prompt_profile_default": "Default saved",
    }


def _click_profile_option(combo, name, qapp):
    QTest.mouseClick(
        combo, Qt.LeftButton, pos=QPoint(combo.width() - 12, combo.height() // 2),
    )
    qapp.processEvents()
    view = combo.view()
    assert view.isVisible()
    index = combo.model().index(combo.findText(name), combo.modelColumn())
    position = view.visualRect(index).center()
    QTest.mouseMove(view.viewport(), position)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=position)
    qapp.processEvents()
    assert not view.isVisible()


def test_typed_selection_keeps_dropdown_index_in_sync(make_harness, qapp):
    window = make_harness(_profiles_config())
    _, combo, editor = _widgets(window)
    combo.setEditText("  Beta  ")
    QTest.keyClick(combo.lineEdit(), Qt.Key_Return)
    qapp.processEvents()

    assert editor.toPlainText() == "Beta saved"
    assert combo.currentIndex() == combo.findText("Beta")
    _click_profile_option(combo, "Alpha", qapp)
    assert combo.currentText() == "Alpha"
    assert editor.toPlainText() == "Alpha saved"


def test_popup_selection_applies_on_first_click_after_editing_name(make_harness, qapp):
    window = make_harness(_profiles_config())
    _, combo, editor = _widgets(window)
    editor.setPlainText("Alpha draft")
    combo.lineEdit().selectAll()
    QTest.keyClicks(combo.lineEdit(), "New name")
    qapp.processEvents()

    for name, expected in (("Beta", "Beta saved"), ("Alpha", "Alpha draft"),
                           ("Default", "Default saved"), ("Beta", "Beta saved")):
        _click_profile_option(combo, name, qapp)
        assert combo.currentText() == name
        assert editor.toPlainText() == expected
        assert combo.currentIndex() == combo.findText(name)
    assert window.saved_configs == []


def test_popup_clicks_to_and_from_default_survive_suppressed_mouse_release(make_harness, qapp):
    # Qt's combo popup can suppress a release during its opening guard. Model
    # that condition rather than relying on native popup timing in headless CI.
    class SuppressRelease(QObject):
        def eventFilter(self, watched, event):
            return event.type() == QEvent.MouseButtonRelease

    window = make_harness(_profiles_config())
    _, combo, editor = _widgets(window)
    editor.setPlainText("Alpha draft")
    release_filter = SuppressRelease(combo)
    combo.view().viewport().installEventFilter(release_filter)

    for name, expected in (("Default", "Default saved"), ("Beta", "Beta saved"),
                           ("Default", "Default saved"), ("Alpha", "Alpha draft")):
        _click_profile_option(combo, name, qapp)
        assert combo.currentText() == name
        assert editor.toPlainText() == expected
        assert combo.currentIndex() == combo.findText(name)
    assert window.saved_configs == []


def test_legacy_prompt_opens_as_plain_text_default_and_reuses_nonmodal_dialog(make_harness):
    prompt = "<analysis>Keep & preserve <b>literal tags</b></analysis>\nSecond line"
    config = {"assistant_prompt": prompt}
    window = make_harness(config)
    dialog, combo, editor = _widgets(window)

    assert dialog.windowTitle() == "Assistant Prompt (Optional)"
    assert not dialog.isModal()
    assert combo.isEditable()
    assert combo.itemText(0) == combo.currentText() == "Default"
    assert editor.toPlainText() == prompt
    _assert_count(window, len(prompt))
    assert window.config == config
    assert window.saved_configs == []

    editor.setPlainText("Draft survives opening the same dialog again")
    window.show_assistant_prompt_dialog()
    assert window._assistant_prompt_dialog is dialog
    assert editor.toPlainText() == "Draft survives opening the same dialog again"


@pytest.mark.parametrize("close_action", ["cancel", "window_close"])
def test_profile_switching_preserves_drafts_but_cancel_discards_them(make_harness, close_action):
    config = _profiles_config()
    window = make_harness(config)
    dialog, combo, editor = _widgets(window)
    assert combo.currentText() == "Alpha"

    editor.setPlainText("Alpha draft")
    _select(combo, "Beta")
    assert editor.toPlainText() == "Beta saved"
    _assert_count(window, len("Beta saved"))
    editor.setPlainText("Beta draft with <tags>")
    _select(combo, "Default")
    assert editor.toPlainText() == "Default saved"
    editor.setPlainText("Default draft")
    _select(combo, "Alpha")
    assert editor.toPlainText() == "Alpha draft"
    _select(combo, "Beta")
    assert editor.toPlainText() == "Beta draft with <tags>"
    _assert_count(window, len("Beta draft with <tags>"))
    assert window.config == config
    assert window.assistant_prompt == "Alpha saved"
    assert window.saved_configs == []

    if close_action == "cancel":
        _click(window, "Cancel")
    else:
        dialog.close()
    assert window._assistant_prompt_dialog is None
    window.show_assistant_prompt_dialog()
    _, combo, editor = _widgets(window)
    assert combo.currentText() == "Alpha"
    assert editor.toPlainText() == "Alpha saved"
    _select(combo, "Beta")
    assert editor.toPlainText() == "Beta saved"
    _select(combo, "Default")
    assert editor.toPlainText() == "Default saved"


def test_save_profile_creates_typed_name_and_cancel_keeps_saved_checkpoint(make_harness):
    window = make_harness({"assistant_prompt": "Legacy default"})
    dialog, combo, editor = _widgets(window)
    combo.setEditText("  Reasoning  ")
    prompt = "<think>Keep the <b>raw markup</b> & continue.</think>"
    editor.setPlainText(prompt)
    _click(window, "💾 Save Profile")

    assert window._assistant_prompt_dialog is dialog
    assert combo.currentText() == "Reasoning"
    assert window.config["assistant_prompt_profiles"]["Reasoning"] == prompt
    assert window.config["active_assistant_prompt_profile"] == "Reasoning"
    assert window.config["assistant_prompt_profile_default"] == "Legacy default"
    assert window.saved_configs[-1]["assistant_prompt"] == window.assistant_prompt == prompt
    assert window.style_updates > 0

    editor.setPlainText("Unsaved subsequent edit")
    _click(window, "Cancel")
    window.show_assistant_prompt_dialog()
    _, combo, editor = _widgets(window)
    assert combo.currentText() == "Reasoning"
    assert editor.toPlainText() == prompt

    editor.setPlainText("Replacement text")
    _click(window, "💾 Save Profile")
    assert window.config["assistant_prompt_profiles"] == {"Reasoning": "Replacement text"}
    assert window.saved_configs[-1]["assistant_prompt"] == "Replacement text"


def test_new_profile_uses_first_unused_number_and_saves_empty_prompt(make_harness):
    window = make_harness({
        "assistant_prompt": "Default text",
        "assistant_prompt_profiles": {"New Profile #1": "One", "New Profile #3": "Three"},
        "assistant_prompt_profile_default": "Default text",
    })
    dialog, combo, editor = _widgets(window)
    _click(window, "+ New Profile")

    assert window._assistant_prompt_dialog is dialog
    assert combo.currentText() == "New Profile #2"
    assert editor.toPlainText() == ""
    _assert_count(window, 0)
    assert window.config["assistant_prompt_profiles"] == {
        "New Profile #1": "One", "New Profile #2": "", "New Profile #3": "Three",
    }
    assert window.config["active_assistant_prompt_profile"] == "New Profile #2"
    assert window.saved_configs[-1]["assistant_prompt"] == window.assistant_prompt == ""


def test_delete_confirmation_and_fallback_update_runtime_prompt(make_harness, monkeypatch):
    window = make_harness(_profiles_config())
    _, combo, editor = _widgets(window)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
    _click(window, "🗑 Delete Profile")
    assert window.config == _profiles_config()
    assert window.saved_configs == []

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    _click(window, "🗑 Delete Profile")
    assert combo.currentText() == "Beta"
    assert editor.toPlainText() == "Beta saved"
    assert window.config["assistant_prompt_profiles"] == {"Beta": "Beta saved"}
    assert window.saved_configs[-1]["assistant_prompt"] == "Beta saved"
    assert window.config["active_assistant_prompt_profile"] == "Beta"

    _click(window, "🗑 Delete Profile")
    assert combo.currentText() == "Default"
    assert editor.toPlainText() == "Default saved"
    assert window.config["assistant_prompt_profiles"] == {}
    assert window.config["active_assistant_prompt_profile"] == ""
    assert window.saved_configs[-1]["assistant_prompt"] == "Default saved"


def test_default_name_is_case_insensitive_and_cannot_be_deleted(make_harness, monkeypatch):
    window = make_harness({"assistant_prompt": "Original"})
    _, combo, editor = _widgets(window)
    confirmations = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: confirmations.append(args) or QMessageBox.Yes,
    )
    combo.setEditText("  dEfAuLt  ")
    editor.setPlainText("Edited default")
    _click(window, "💾 Save Profile")
    assert combo.currentText() == "Default"
    assert window.config["assistant_prompt_profiles"] == {}
    assert window.config["assistant_prompt_profile_default"] == "Edited default"
    assert window.config["active_assistant_prompt_profile"] == ""

    saved = copy.deepcopy(window.config)
    combo.setEditText("dEfAuLt")
    _click(window, "🗑 Delete Profile")
    assert window.config == saved
    assert confirmations == []


def test_bottom_save_persists_typed_profile_and_reopens_it(make_harness):
    window = make_harness({"assistant_prompt": "Legacy default"})
    _, combo, editor = _widgets(window)
    combo.setEditText("XML output")
    prompt = "<response>\nExact & literal markup\n</response>"
    editor.setPlainText(prompt)
    _click(window, "Save")

    assert window._assistant_prompt_dialog is None
    assert window.config["assistant_prompt_profiles"] == {"XML output": prompt}
    assert window.config["active_assistant_prompt_profile"] == "XML output"
    assert window.saved_configs[-1]["assistant_prompt"] == prompt
    window.show_assistant_prompt_dialog()
    _, combo, editor = _widgets(window)
    assert combo.currentText() == "XML output"
    assert editor.toPlainText() == prompt
    _select(combo, "Default")
    assert editor.toPlainText() == "Legacy default"


def test_clear_is_staged_until_save_and_disables_runtime_prefill(make_harness):
    window = make_harness(_profiles_config())
    _, _, editor = _widgets(window)
    _click(window, "Clear")
    assert editor.toPlainText() == ""
    _assert_count(window, 0)
    assert window.assistant_prompt == "Alpha saved"
    assert window.saved_configs == []

    _click(window, "Save")
    assert window._assistant_prompt_dialog is None
    assert window.assistant_prompt == ""
    assert window.config["assistant_prompt_profiles"]["Alpha"] == ""
    assert window.config["active_assistant_prompt_profile"] == "Alpha"
    assert window.saved_configs[-1]["assistant_prompt"] == ""


def test_failed_save_preserves_runtime_and_config_and_keeps_edits_open(make_harness, monkeypatch):
    window = make_harness(_profiles_config())
    dialog, _, editor = _widgets(window)
    editor.setPlainText("New text awaiting successful save")
    monkeypatch.setattr(window, "save_config", lambda **kwargs: False)
    _click(window, "Save")

    assert window._assistant_prompt_dialog is dialog
    assert window.config == _profiles_config()
    assert window.assistant_prompt == "Alpha saved"
    assert editor.toPlainText() == "New text awaiting successful save"
    assert window.style_updates == 0
    assert window.logs == []

    monkeypatch.setattr(window, "save_config", _PromptHarness.save_config.__get__(window))
    _click(window, "Save")
    assert window._assistant_prompt_dialog is None
    assert window.config["assistant_prompt_profiles"]["Alpha"] == "New text awaiting successful save"
    assert window.saved_configs[-1]["assistant_prompt"] == "New text awaiting successful save"


@pytest.mark.parametrize("active", ["Alpha", "Missing"])
def test_saved_profile_text_wins_over_stale_flat_runtime_prompt(make_harness, active):
    config = _profiles_config()
    config["assistant_prompt"] = "Stale runtime text"
    config["assistant_prompt_profile_default"] = ""
    config["active_assistant_prompt_profile"] = active
    window = make_harness(config)
    _, combo, editor = _widgets(window)

    expected = "Alpha saved" if active == "Alpha" else ""
    assert combo.currentText() == ("Alpha" if active == "Alpha" else "Default")
    assert editor.toPlainText() == expected
    _assert_count(window, len(expected))
    assert window.config == config
    assert window.saved_configs == []
