"""Ctrl zoom of the manga editor's image preview (manga_preview_zoom), with real key and wheel events.

Owner report: Ctrl + / - and Ctrl+mouse wheel did not zoom the (webtoon) preview, only the zoom buttons
did. ``manga_format_toolbar.install_format_toolbar`` now installs ``manga_preview_zoom``: Ctrl+= / Ctrl++
(numpad too) / Ctrl+- (numpad too) press the preview's Zoom In / Zoom Out, Ctrl+0 its Fit to View,
Ctrl+wheel (and a pinch) zoom one button step per notch about the mouse; a plain wheel still scrolls,
a focused text field keeps its keys and the Format row's Ctrl+B still works.
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
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, Signal  # noqa: E402
from PySide6.QtGui import QColor, QPixmap, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

STEP = 1.15
CTRL = Qt.KeyboardModifier.ControlModifier
KEYPAD = Qt.KeyboardModifier.KeypadModifier


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _MainGui(QObject):
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
def window(qapp, tmp_path, monkeypatch):
    """The Manga Panel Translator window: a line edit, then the preview showing a tall webtoon strip."""
    import manga_image_preview
    from manga_format_toolbar import install_format_toolbar

    dialog = QtWidgets.QDialog()
    dialog.resize(900, 700)
    layout = QtWidgets.QVBoxLayout(dialog)
    line_edit = QtWidgets.QLineEdit()
    layout.addWidget(line_edit)
    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    layout.addWidget(widget, 1)
    font_combo = QtWidgets.QComboBox()
    font_combo.addItems(["Default"])
    tab = types.SimpleNamespace(image_state_manager=_States(), font_combo=font_combo, font_mapping={},
                                dialog=dialog, main_gui=_MainGui(), text_color_r_value=0,
                                text_color_g_value=0, text_color_b_value=0)
    widget.manga_integration = tab
    widget.current_image_path = page = str(tmp_path / "001.png")
    monkeypatch.setattr(widget, "_on_save_overlay_clicked", lambda: None)
    toolbar = install_format_toolbar(widget, tab)
    viewer = widget.viewer
    strip = QPixmap(800, 8000)
    strip.fill(QColor(200, 200, 200))
    viewer.setPhoto(strip)
    item = manga_image_preview.MoveableRectItem(QRectF(100, 300, 200, 100))
    item.region_index = 0
    viewer.scene().addItem(item)
    viewer.rectangles.append(item)
    dialog.show()
    dialog.activateWindow()
    QTest.qWaitForWindowActive(dialog, 2000)
    _pump(qapp, 0.05)  # setPhoto's fit timer
    viewer.fitInView()
    viewer.setFocus()
    _pump(qapp)
    yield types.SimpleNamespace(app=qapp, dialog=dialog, widget=widget, viewer=viewer, line_edit=line_edit,
                                toolbar=toolbar, tab=tab, page=page, rect=item)
    for rect in list(viewer.rectangles):
        viewer.scene().removeItem(rect)
    dialog.close()
    dialog.deleteLater()
    _pump(qapp)


def _scale(viewer):
    return viewer.transform().m11()


def _wheel(env, dy, modifiers=Qt.KeyboardModifier.NoModifier, pos=None):
    viewport = env.viewer.viewport()
    pos = QPointF(pos if pos is not None else QPointF(viewport.rect().center()))
    event = QWheelEvent(pos, QPointF(viewport.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, dy),
                        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(viewport, event)
    env.app.processEvents()
    return event


def _key(env, key, modifiers=CTRL, target=None):
    QTest.keyClick(target or env.viewer, key, modifiers)
    env.app.processEvents()


def test_ctrl_wheel_zooms_one_button_step_per_notch(window):
    viewer = window.viewer
    fit = _scale(viewer)
    _wheel(window, 120, CTRL)
    assert _scale(viewer) == pytest.approx(fit * STEP)
    _wheel(window, 240, CTRL)  # two notches at once
    assert _scale(viewer) == pytest.approx(fit * STEP ** 3)
    _wheel(window, -120, CTRL)
    assert _scale(viewer) == pytest.approx(fit * STEP ** 2)
    assert viewer.user_zoomed  # a resize keeps the zoom, as after the buttons


def test_ctrl_wheel_adds_up_small_touchpad_deltas(window):
    viewer = window.viewer
    fit = _scale(viewer)
    for _ in range(3):
        _wheel(window, 30, CTRL)
    assert _scale(viewer) == pytest.approx(fit)
    _wheel(window, 30, CTRL)
    assert _scale(viewer) == pytest.approx(fit * STEP)
    _wheel(window, 90, CTRL)
    _wheel(window, -30, CTRL)  # the direction turned: the 90 left over is dropped
    assert _scale(viewer) == pytest.approx(fit * STEP)


def test_ctrl_wheel_keeps_the_point_under_the_mouse(window):
    viewer = window.viewer
    viewer.resetTransform()
    viewer.scale(2, 2)  # the strip is larger than the view both ways (both scroll bars shown)
    viewer.user_zoomed = True  # no refit on a late resize
    window.app.processEvents()
    viewer.centerOn(400, 4000)
    window.app.processEvents()
    pos = QPoint(viewer.viewport().width() // 4, viewer.viewport().height() // 3)
    before = viewer.mapToScene(pos)
    _wheel(window, 120, CTRL, QPointF(pos))
    assert _scale(viewer) == pytest.approx(2 * STEP)
    after = viewer.mapToScene(pos)
    assert abs(after.x() - before.x()) <= 1 and abs(after.y() - before.y()) <= 1
    _wheel(window, -240, CTRL, QPointF(pos))
    after = viewer.mapToScene(pos)
    assert abs(after.x() - before.x()) <= 2 and abs(after.y() - before.y()) <= 2


def test_plain_wheel_still_scrolls_the_strip(window):
    viewer = window.viewer
    viewer.resetTransform()
    viewer.user_zoomed = True
    window.app.processEvents()
    viewer.centerOn(400, 0)
    window.app.processEvents()
    bar = viewer.verticalScrollBar()
    start, scale = bar.value(), _scale(viewer)
    _wheel(window, -120)
    assert bar.value() > start
    assert _scale(viewer) == pytest.approx(scale)


def test_shift_wheel_keeps_the_viewer_zoom(window):
    viewer = window.viewer
    fit = _scale(viewer)
    _wheel(window, 120, Qt.KeyboardModifier.ShiftModifier)
    assert _scale(viewer) == pytest.approx(fit * STEP)


def test_ctrl_keys_zoom_like_the_buttons(window):
    viewer = window.viewer
    fit = _scale(viewer)
    _key(window, Qt.Key.Key_Equal)
    assert _scale(viewer) == pytest.approx(fit * STEP)
    _key(window, Qt.Key.Key_Plus)
    assert _scale(viewer) == pytest.approx(fit * STEP ** 2)
    _key(window, Qt.Key.Key_Plus, CTRL | KEYPAD)
    assert _scale(viewer) == pytest.approx(fit * STEP ** 3)
    _key(window, Qt.Key.Key_Minus)
    assert _scale(viewer) == pytest.approx(fit * STEP ** 2)
    _key(window, Qt.Key.Key_Minus, CTRL | KEYPAD)
    assert _scale(viewer) == pytest.approx(fit * STEP)
    _key(window, Qt.Key.Key_0)
    assert _scale(viewer) == pytest.approx(fit)
    assert not viewer.user_zoomed  # Fit to View


def test_keys_work_from_elsewhere_in_the_window_but_not_in_a_text_field(window):
    viewer = window.viewer
    fit = _scale(viewer)
    window.widget.zoom_in_btn.setFocus()
    _key(window, Qt.Key.Key_Equal, target=window.widget.zoom_in_btn)
    assert _scale(viewer) == pytest.approx(fit * STEP)
    window.line_edit.setFocus()
    window.app.processEvents()
    _key(window, Qt.Key.Key_Minus, target=window.line_edit)
    _key(window, Qt.Key.Key_0, target=window.line_edit)
    assert _scale(viewer) == pytest.approx(fit * STEP)
    viewer.setFocus()
    window.app.processEvents()
    _key(window, Qt.Key.Key_Minus)
    assert _scale(viewer) == pytest.approx(fit)


def test_format_row_ctrl_b_still_works(window):
    window.widget.manual_editing_toggle.setChecked(True)
    window.rect.setSelected(True)
    window.app.processEvents()
    _key(window, Qt.Key.Key_B)
    assert window.tab.image_state_manager.get_state(window.page).get('region_styles') == {'0': {'bold': True}}
    _pump(window.app, 0.6)  # the Manual Editing icon's spin


def test_pinch_gesture_zooms_about_the_fingers(window):
    from PySide6.QtGui import QNativeGestureEvent, QPointingDevice

    viewer = window.viewer
    fit = _scale(viewer)
    viewport = viewer.viewport()
    pos = QPointF(viewport.rect().center())
    glob = QPointF(viewport.mapToGlobal(pos.toPoint()))
    try:
        event = QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(),
                                    2, pos, pos, glob, 0.2, QPointF(0, 0))
    except TypeError:
        pytest.skip("this PySide6 has another QNativeGestureEvent constructor")
    QtWidgets.QApplication.sendEvent(viewport, event)
    assert _scale(viewer) == pytest.approx(fit * STEP)  # ln(1.2) is one ln(1.15) step


def _gesture(viewport, kind, value=0.0):
    from PySide6.QtGui import QNativeGestureEvent, QPointingDevice

    pos = QPointF(viewport.rect().center())
    glob = QPointF(viewport.mapToGlobal(pos.toPoint()))
    try:
        return QNativeGestureEvent(kind, QPointingDevice.primaryPointingDevice(), 2, pos, pos, glob, value,
                                   QPointF(0, 0))
    except TypeError:
        pytest.skip("this PySide6 has another QNativeGestureEvent constructor")


def test_a_new_pinch_does_not_inherit_the_last_pinchs_part_step(window):
    viewer = window.viewer
    viewport = viewer.viewport()
    fit = _scale(viewer)
    gesture = Qt.NativeGestureType
    # A pinch that ends just short of one step zooms nothing
    QtWidgets.QApplication.sendEvent(viewport, _gesture(viewport, gesture.BeginNativeGesture))
    QtWidgets.QApplication.sendEvent(viewport, _gesture(viewport, gesture.ZoomNativeGesture, 0.14))
    QtWidgets.QApplication.sendEvent(viewport, _gesture(viewport, gesture.EndNativeGesture))
    assert _scale(viewer) == pytest.approx(fit)
    # ... and the next pinch starts from zero: a small spread is still no step
    QtWidgets.QApplication.sendEvent(viewport, _gesture(viewport, gesture.BeginNativeGesture))
    QtWidgets.QApplication.sendEvent(viewport, _gesture(viewport, gesture.ZoomNativeGesture, 0.02))
    assert _scale(viewer) == pytest.approx(fit)
    QtWidgets.QApplication.sendEvent(viewport, _gesture(viewport, gesture.ZoomNativeGesture, 0.14))
    assert _scale(viewer) == pytest.approx(fit * STEP)


def test_buttons_name_the_keys(window):
    assert "Ctrl+" in window.widget.zoom_in_btn.toolTip()
    assert "Ctrl+" in window.widget.zoom_out_btn.toolTip()
    assert "Ctrl+0" in window.widget.fit_btn.toolTip()


def test_installed_once():
    from manga_preview_zoom import install_preview_zoom

    preview = types.SimpleNamespace(preview_zoom="installed")
    assert install_preview_zoom(preview) == "installed"
