"""Headless test of the manga editor's Format toolbar (manga_format_toolbar) on a real preview widget.

The row sits at the top of MangaImagePreviewWidget (under the title), is shown with Enable Manual
Editing, is disabled without a selected box, shows the selected box's style, stores every change in
the page state (``region_styles``) and re-renders once through the preview's Save & Update Overlay.

A box on Auto shows the size it was rendered at (``region_font_sizes``) and steps 0.5 px from it; the
size box is greyed when disabled and as tall as the row; the box menu's − / + row; Enter never presses
a push button of the manga window ("Add Files").
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QObject, QRectF, Qt, Signal  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402

FONT_PATH = TESTS_DIR / "fixtures" / "manga_editor" / "DejaVuSans-Latin.ttf"


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _MainGui(QObject):
    """TranslatorGUI's signal a finished render emits (the preview then reloads the page)."""
    refresh_preview_signal = Signal()


class _States:
    def __init__(self):
        self.states = {}

    def get_state(self, path):
        return self.states.get(path, {})

    def set_state(self, path, state, save=True):
        self.states[path] = state


def _pump(app, seconds=0.0):
    deadline = time.monotonic() + seconds
    while True:
        app.processEvents()
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)


@pytest.fixture
def editor(qapp, tmp_path, monkeypatch):
    import manga_image_preview
    from manga_format_toolbar import install_format_toolbar

    # The Manga Panel Translator window: a QDialog whose "Add Files" button is built before the preview
    dialog = QtWidgets.QDialog()
    dialog_layout = QtWidgets.QVBoxLayout(dialog)
    add_files = QtWidgets.QPushButton("Add Files")
    add_files_clicks = []
    add_files.clicked.connect(lambda: add_files_clicks.append(1))
    dialog_layout.addWidget(add_files)
    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    dialog_layout.addWidget(widget)
    font_combo = QtWidgets.QComboBox()
    font_combo.addItems(["Default", "DejaVu Sans", "Browse Custom Font..."])
    tab = types.SimpleNamespace(image_state_manager=_States(), font_combo=font_combo,
                                font_mapping={"DejaVu Sans": str(FONT_PATH)}, dialog=dialog, main_gui=_MainGui(),
                                text_color_r_value=0, text_color_g_value=0, text_color_b_value=0)
    widget.manga_integration = tab
    widget.current_image_path = page = str(tmp_path / "001.png")
    renders = []
    monkeypatch.setattr(widget, "_on_save_overlay_clicked", lambda: renders.append(page))
    viewer = widget.viewer
    for index, (x, y, w, h) in enumerate(((10, 10, 100, 60), (150, 10, 100, 60), (10, 120, 200, 60))):
        item = manga_image_preview.MoveableRectItem(QRectF(x, y, w, h))
        item.region_index = index
        viewer.scene().addItem(item)
        viewer.rectangles.append(item)
    toolbar = install_format_toolbar(widget, tab)
    dialog.show()
    monkeypatch.setattr("manga_format_toolbar.RENDER_DELAY_MS", 0)
    toolbar._render_timer.setInterval(0)
    yield types.SimpleNamespace(widget=widget, tab=tab, toolbar=toolbar, page=page, renders=renders,
                                rects=viewer.rectangles, app=qapp, dialog=dialog, add_files=add_files,
                                add_files_clicks=add_files_clicks)
    _pump(qapp, 0.6)  # let the Manual Editing icon's spin (~0.5 s) finish before its label goes
    for item in list(viewer.rectangles):
        viewer.scene().removeItem(item)
    dialog.close()
    dialog.deleteLater()
    _pump(qapp)


def _select(env, *indexes):
    scene = env.widget.viewer.scene()
    scene.clearSelection()
    for index in indexes:
        env.rects[index].setSelected(True)
    _pump(env.app)


def _styles(env):
    return env.tab.image_state_manager.get_state(env.page).get('region_styles', {})


def test_toolbar_sits_at_the_top_and_follows_manual_editing(editor):
    layout = editor.widget.layout()
    assert layout.indexOf(editor.toolbar.frame) == 1  # right under the title row, above the viewer
    assert editor.widget.format_toolbar is editor.toolbar
    toggle = editor.widget.manual_editing_toggle
    toggle.setChecked(True)
    assert not editor.toolbar.frame.isHidden()
    toggle.setChecked(False)
    assert editor.toolbar.frame.isHidden()
    # the Rendering Settings font list (without its "Default" / "Browse" entries)
    combo = editor.toolbar.font_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["Default font", "DejaVu Sans"]
    # the preview's own tool buttons
    assert editor.toolbar.bold_btn.styleSheet() == editor.widget.fit_btn.styleSheet()


def test_disabled_without_a_selection(editor):
    _select(editor)
    assert editor.toolbar.indexes() == []
    assert not any(control.isEnabled() for control in editor.toolbar.controls)
    editor.toolbar.bold_btn.click()  # nothing to format
    assert _styles(editor) == {} and editor.renders == []
    _select(editor, 1)
    assert editor.toolbar.indexes() == [1]
    assert all(control.isEnabled() for control in editor.toolbar.controls)
    assert editor.toolbar.selection_label.text() == "Box 2"


def test_changes_are_stored_per_box_and_render_once(editor):
    toolbar = editor.toolbar
    _select(editor, 0, 2)
    toolbar.bold_btn.click()
    toolbar.italic_btn.click()
    toolbar.underline_btn.click()
    toolbar.size_spin.setValue(28)
    toolbar.align_buttons['right'].click()
    toolbar.font_combo.setCurrentIndex(1)
    expected = {'font': str(FONT_PATH), 'font_size': 28, 'bold': True, 'italic': True, 'underline': True,
                'align': 'right'}
    assert _styles(editor) == {'0': expected, '2': expected}
    _pump(editor.app, 0.1)
    assert editor.renders == [editor.page]  # one render-only pass for the burst of changes

    # colour through the colour dialog
    from PySide6 import QtWidgets as W
    original = W.QColorDialog.getColor
    W.QColorDialog.getColor = staticmethod(lambda *a, **k: QColor(10, 200, 30))
    try:
        toolbar.color_btn.click()
    finally:
        W.QColorDialog.getColor = original
    assert _styles(editor)['0']['color'] == [10, 200, 30]

    # back to the defaults, one entry at a time and with Reset
    toolbar.align_buttons['center'].click()
    toolbar.size_spin.setValue(0)
    toolbar.bold_btn.click()
    assert 'align' not in _styles(editor)['0'] and 'font_size' not in _styles(editor)['0']
    assert 'bold' not in _styles(editor)['2']
    toolbar.reset_btn.click()
    assert _styles(editor) == {}
    _pump(editor.app, 0.1)
    assert len(editor.renders) == 2


def test_a_page_left_before_its_render_is_rendered_from_its_state(editor, monkeypatch):
    """Switching images inside the debounce must not re-render the newly shown page in its place."""
    import ImageRenderer
    calls = []
    # ImageRenderer's persisted-state render (on main_gui.executor when there is one; none here)
    monkeypatch.setattr(ImageRenderer, "render_persisted_translation_state",
                        lambda tab, page, refresh_preview=True: calls.append((tab, page, refresh_preview)))
    _select(editor, 0)
    editor.toolbar.bold_btn.click()
    editor.widget.current_image_path = editor.page.replace("001", "002")
    _pump(editor.app, 0.1)
    assert calls == [(editor.tab, editor.page, False)] and editor.renders == []


def test_no_change_while_a_batch_runs(editor):
    """The viewer blocks box edits during a batch: the toolbar neither stores nor re-renders then."""
    _select(editor, 0)
    editor.tab._batch_mode_active = True
    editor.widget.viewer.manga_integration = editor.tab
    editor.toolbar.bold_btn.click()
    _pump(editor.app, 0.1)
    assert _styles(editor) == {} and editor.renders == [] and not editor.toolbar.bold_btn.isChecked()
    editor.tab._batch_mode_active = False
    editor.toolbar.bold_btn.click()
    assert _styles(editor) == {'0': {'bold': True}}


def test_shows_the_selected_box_style(editor):
    editor.tab.image_state_manager.set_state(editor.page, {'region_styles': {
        '1': {'bold': True, 'underline': True, 'font_size': 33, 'align': 'left', 'font': str(FONT_PATH)}}})
    toolbar = editor.toolbar
    _select(editor, 1)
    assert toolbar.bold_btn.isChecked() and toolbar.underline_btn.isChecked() and not toolbar.italic_btn.isChecked()
    assert toolbar.size_spin.value() == 33 and toolbar.font_combo.currentText() == "DejaVu Sans"
    assert toolbar.align_buttons['left'].isChecked() and not toolbar.align_buttons['center'].isChecked()
    _select(editor, 0)
    assert not toolbar.bold_btn.isChecked() and toolbar.size_spin.value() == 0
    assert toolbar.size_spin.text() == "Auto" and toolbar.font_combo.currentText() == "Default font"
    assert toolbar.align_buttons['center'].isChecked()
    assert editor.renders == []  # showing a selection changes nothing


def test_keyboard_shortcuts_and_clear_boxes(editor):
    toolbar = editor.toolbar
    editor.widget.manual_editing_toggle.setChecked(True)
    _select(editor, 2)
    for shortcut in toolbar._shortcuts:
        if shortcut.key().toString() == "Ctrl+B":
            shortcut.activated.emit()
    assert _styles(editor) == {'2': {'bold': True}}
    toolbar._on_boxes_cleared()  # Clear Boxes (connected after the preview's own handler)
    assert _styles(editor) == {}


def _menu_actions(menu):
    return {action.text(): action for action in menu.actions() if action.text()}


def test_right_click_menu_formats_the_box(editor, monkeypatch):
    """The box's right-click "Text format" submenu applies the toolbar's changes to that box."""
    from PySide6 import QtWidgets as W
    toolbar = editor.toolbar
    _select(editor, 0)
    menu = W.QMenu()
    sub = toolbar.add_context_menu(menu, 1)  # right-clicked box 2, not part of the selection
    assert sub.title() == "🔤 Text format" and sub.isEnabled()
    actions = _menu_actions(sub)
    assert {"Bold", "Italic", "Underline", "Align left", "Center", "Align right", "Font size: Auto", "Font size…",
            "Auto size (fit the box)", "Font", "Text colour…", "⟲ Reset to default"} <= set(actions)
    assert actions["Center"].isChecked() and not actions["Bold"].isChecked()
    actions["Bold"].trigger()
    actions["Align right"].trigger()
    assert _styles(editor) == {'1': {'bold': True, 'align': 'right'}}

    monkeypatch.setattr(W.QInputDialog, "getDouble", staticmethod(lambda *a, **k: (30, True)))
    actions["Font size…"].trigger()
    fonts = _menu_actions(actions["Font"].menu())
    assert fonts["Default font"].isChecked()
    fonts["DejaVu Sans"].trigger()
    monkeypatch.setattr(W.QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor(1, 2, 3)))
    actions["Text colour…"].trigger()
    assert _styles(editor) == {'1': {'bold': True, 'align': 'right', 'font_size': 30, 'font': str(FONT_PATH),
                                     'color': [1, 2, 3]}}

    # a fresh menu shows the stored style; on a selected box it acts on the whole selection
    _select(editor, 1, 2)
    menu = W.QMenu()
    sub = toolbar.add_context_menu(menu, 1)
    assert sub.title() == "🔤 Text format (2 boxes)"
    actions = _menu_actions(sub)
    assert actions["Bold"].isChecked() and actions["Align right"].isChecked() and "Font size: 30 px" in actions
    actions["Italic"].trigger()
    assert _styles(editor)['2'] == {'italic': True} and _styles(editor)['1']['italic'] is True
    actions["⟲ Reset to default"].trigger()
    assert _styles(editor) == {}
    _pump(editor.app, 0.1)
    assert editor.renders and set(editor.renders) == {editor.page}  # render-only passes of this page


def test_box_context_menu_has_the_text_format_submenu(editor, monkeypatch):
    """ImageRenderer's box right-click menu (pinned code, approved edit) adds the submenu."""
    import ImageRenderer
    from PySide6 import QtWidgets as W
    from PySide6.QtCore import QPointF, QTimer
    from PySide6.QtGui import QMouseEvent
    shown = []

    def read_and_close_popup():
        popup = W.QApplication.activePopupWidget()
        if popup is None:
            QTimer.singleShot(20, read_and_close_popup)
            return
        shown.append([action.text() for action in popup.actions()])
        popup.close()

    QTimer.singleShot(20, read_and_close_popup)  # the menu runs its own event loop (QMenu.exec)
    editor.tab.image_preview_widget = editor.widget
    rect = editor.rects[1]
    ImageRenderer._add_context_menu_to_rectangle(editor.tab, rect, 1)
    event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
                        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
    rect.mousePressEvent(event)
    assert shown and "🔤 Text format" in shown[0]
    assert shown[0].index("🔤 Text format") < shown[0].index("🗑️ Delete Selected")


# ===========================================================================
# Owner requests 2026-11-06: the real size, 0.5 px steps, look, menu − / +, Enter
# ===========================================================================

def _set_page_state(env, **state):
    env.tab.image_state_manager.set_state(env.page, dict(state))


def test_size_box_shows_the_rendered_size_of_a_box_on_auto(editor):
    """A box on Auto shows the size it was last rendered at ("28 px (auto)"), not "Auto"; the arrows
    give it a fixed size 0.5 px up / down from there."""
    _set_page_state(editor, region_font_sizes={'0': 28, '1': 31}, region_styles={'1': {'font_size': 31.5}})
    spin = editor.toolbar.size_spin
    _select(editor, 0)
    assert spin.value() == 28 and spin.text() == "28 px (auto)" and spin.font().italic()
    assert "rendered at 28 px" in spin.toolTip()
    spin.stepBy(1)
    assert _styles(editor) == {'0': {'font_size': 28.5}, '1': {'font_size': 31.5}}
    assert spin.text() == "28.5 px" and not spin.font().italic()
    spin.stepBy(-1)
    spin.stepBy(-1)
    assert _styles(editor)['0'] == {'font_size': 27.5}
    _select(editor, 1)  # its own size (the rendered size is not shown for a fixed size)
    assert spin.text() == "31.5 px"
    _select(editor, 2)  # never rendered: Auto
    assert spin.value() == 0 and spin.text() == "Auto"
    # down from the auto size too
    _select(editor, 0)
    editor.toolbar.reset_btn.click()
    assert '0' not in _styles(editor) and spin.text() == "28 px (auto)"
    spin.stepBy(-1)
    assert _styles(editor)['0'] == {'font_size': 27.5}
    _pump(editor.app, 0.1)
    assert editor.renders and set(editor.renders) == {editor.page}


def test_size_box_typed_value_and_auto(editor):
    """Typing a size (Enter commits it) sets it, in 0.5 px steps; 0 goes back to Auto."""
    from PySide6.QtTest import QTest
    _set_page_state(editor, region_font_sizes={'0': 22})
    spin = editor.toolbar.size_spin
    _select(editor, 0)
    spin.setFocus()
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "30.5")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert _styles(editor) == {'0': {'font_size': 30.5}} and spin.text() == "30.5 px"
    spin.setValue(0)
    assert _styles(editor) == {} and spin.text() == "22 px (auto)"
    assert editor.add_files_clicks == []  # Enter in the size box commits it, nothing else


def test_size_box_with_several_boxes_shows_the_first(editor):
    _set_page_state(editor, region_font_sizes={'0': 20, '2': 26})
    _select(editor, 0, 2)
    spin = editor.toolbar.size_spin
    assert spin.text() == "20 px (auto)"
    spin.stepBy(1)
    assert _styles(editor) == {'0': {'font_size': 20.5}, '2': {'font_size': 20.5}}


def test_size_box_refreshes_after_a_render(editor):
    """A render records new sizes and emits TranslatorGUI's refresh_preview_signal: the row follows."""
    _select(editor, 0)
    assert editor.toolbar.size_spin.text() == "Auto"
    _set_page_state(editor, region_font_sizes={'0': 19})
    editor.tab.main_gui.refresh_preview_signal.emit()  # the render saved the page
    _pump(editor.app)
    assert editor.toolbar.size_spin.text() == "19 px (auto)"


def test_size_box_disabled_look_and_height(editor):
    """Disabled like the other controls (greyed text, under the window's white-text style sheet);
    as tall as the row's tool buttons and font box; no box-model style sheet (native arrows)."""
    from PySide6.QtGui import QPalette
    editor.dialog.setStyleSheet("QWidget { background-color: #1e1e1e; color: white; }")  # the main GUI's rule
    _pump(editor.app)
    spin = editor.toolbar.size_spin
    _select(editor)
    assert not spin.isEnabled()
    assert spin.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text) == QColor("#555555")
    _select(editor, 0)
    assert spin.isEnabled()
    assert spin.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text) == QColor("white")
    sheet = spin.styleSheet()
    assert ":disabled" in sheet and not any(rule in sheet for rule in ("border", "padding", "background", "height"))
    _pump(editor.app)
    toolbar = editor.toolbar
    assert spin.height() == toolbar.bold_btn.height() == toolbar.font_combo.height() == toolbar.reset_btn.height()
    assert spin.wheelEvent(None) is None  # the mouse wheel stays off


def test_menu_size_row_steps_and_stays_open(editor):
    """The box menu shows the real size and its − / + step it by 0.5 px (the menu stays open: a
    widget row); Font size… takes 0.5 px values; Auto size goes back to Auto."""
    from PySide6 import QtWidgets as W
    _set_page_state(editor, region_font_sizes={'0': 28, '1': 40})
    toolbar = editor.toolbar
    _select(editor, 0)
    menu = W.QMenu()
    sub = toolbar.add_context_menu(menu, 0)
    actions = _menu_actions(sub)
    row = actions["Font size: 28 px (auto)"]
    assert isinstance(row, W.QWidgetAction) and row.size_label.text() == "28 px (auto)"
    assert actions["Auto size (fit the box)"].isChecked()
    row.plus_button.click()
    assert _styles(editor) == {'0': {'font_size': 28.5}}
    assert row.size_label.text() == "28.5 px" and row.text() == "Font size: 28.5 px"
    assert not actions["Auto size (fit the box)"].isChecked()
    row.minus_button.click()
    row.minus_button.click()
    assert _styles(editor) == {'0': {'font_size': 27.5}}
    actions["Auto size (fit the box)"].trigger()
    assert _styles(editor) == {}

    original = W.QInputDialog.getDouble
    try:
        calls = []
        W.QInputDialog.getDouble = staticmethod(lambda *a, **k: calls.append(a) or (33.5, True))
        actions["Font size…"].trigger()
    finally:
        W.QInputDialog.getDouble = original
    assert _styles(editor) == {'0': {'font_size': 33.5}}
    assert calls[0][3:7] == (28.0, 0, 200, 1) and calls[0][-1] == 0.5  # from the shown size, 0.5 px steps

    # a box never rendered and on Auto: nothing to step from
    menu = W.QMenu()
    sub = toolbar.add_context_menu(menu, 2)
    row = _menu_actions(sub)["Font size: Auto"]
    assert not row.plus_button.isEnabled() and not row.minus_button.isEnabled()
    _pump(editor.app, 0.1)


def test_enter_never_presses_a_window_button(editor):
    """In the Manga Panel Translator window Enter does not click "Add Files" (the QDialog's first
    auto-default button), from a spin box, from the focused button itself, or for a button added
    after the window was built; Space still presses a focused button."""
    from PySide6.QtTest import QTest
    dialog, add_files = editor.dialog, editor.add_files
    _pump(editor.app)
    assert not add_files.autoDefault() and not add_files.isDefault()
    _select(editor, 0)
    for target in (editor.toolbar.size_spin, add_files):
        target.setFocus()
        _pump(editor.app)
        QTest.keyClick(target, Qt.Key.Key_Return)
        QTest.keyClick(target, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
    QTest.keyClick(dialog, Qt.Key.Key_Return)
    assert editor.add_files_clicks == []
    later_clicks = []
    later = QtWidgets.QPushButton("Load Folder")
    later.clicked.connect(lambda: later_clicks.append(1))
    dialog.layout().addWidget(later)
    later.show()
    dialog.activateWindow()
    later.setFocus()
    _pump(editor.app)
    later.setFocus(Qt.FocusReason.TabFocusReason)
    editor.app.focusChanged.emit(None, later)  # offscreen: focus changes are not always reported
    QTest.keyClick(later, Qt.Key.Key_Return)
    assert later_clicks == [] and not later.autoDefault() and not later.isDefault()
    QTest.keyClick(add_files, Qt.Key.Key_Space)
    assert editor.add_files_clicks == [1]
    # a dialog shown again: still no default button
    dialog.hide()
    dialog.show()
    _pump(editor.app)
    assert not any(button.isDefault() or button.autoDefault() for button in dialog.findChildren(QtWidgets.QPushButton))


def test_guard_enter_key_is_installed_once():
    from manga_format_toolbar import guard_enter_key
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert guard_enter_key(None) is None and guard_enter_key(QtWidgets.QWidget()) is None
    dialog = QtWidgets.QDialog()
    first = guard_enter_key(dialog)
    assert guard_enter_key(dialog) is first
    dialog.deleteLater()
