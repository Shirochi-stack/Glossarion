"""Headless test of the manga editor's Format toolbar (manga_format_toolbar) on a real preview widget.

The row sits at the top of MangaImagePreviewWidget (under the title), is shown with Enable Manual
Editing, is disabled without a selected box, shows the selected box's style, stores every change in
the page state (``region_styles``) and re-renders once through the preview's Save & Update Overlay.
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
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402

FONT_PATH = TESTS_DIR / "fixtures" / "manga_editor" / "DejaVuSans-Latin.ttf"


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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

    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    font_combo = QtWidgets.QComboBox()
    font_combo.addItems(["Default", "DejaVu Sans", "Browse Custom Font..."])
    tab = types.SimpleNamespace(image_state_manager=_States(), font_combo=font_combo,
                                font_mapping={"DejaVu Sans": str(FONT_PATH)},
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
    widget.show()
    monkeypatch.setattr("manga_format_toolbar.RENDER_DELAY_MS", 0)
    toolbar._render_timer.setInterval(0)
    yield types.SimpleNamespace(widget=widget, tab=tab, toolbar=toolbar, page=page, renders=renders,
                                rects=viewer.rectangles, app=qapp)
    _pump(qapp, 0.6)  # let the Manual Editing icon's spin (~0.5 s) finish before its label goes
    for item in list(viewer.rectangles):
        viewer.scene().removeItem(item)
    widget.close()
    widget.deleteLater()
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
    assert {"Bold", "Italic", "Underline", "Align left", "Center", "Align right", "Font size: Auto…",
            "Font", "Text colour…", "⟲ Reset to default"} <= set(actions)
    assert actions["Center"].isChecked() and not actions["Bold"].isChecked()
    actions["Bold"].trigger()
    actions["Align right"].trigger()
    assert _styles(editor) == {'1': {'bold': True, 'align': 'right'}}

    monkeypatch.setattr(W.QInputDialog, "getInt", staticmethod(lambda *a, **k: (30, True)))
    actions["Font size: Auto…"].trigger()
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
    assert actions["Bold"].isChecked() and actions["Align right"].isChecked() and "Font size: 30 px…" in actions
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
