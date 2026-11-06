# manga_preview_zoom.py
"""Ctrl zoom of the manga editor's image preview (keyboard, Ctrl+wheel, touchpad pinch).

The preview's zoom buttons (``MangaImagePreviewWidget._zoom_in`` / ``_zoom_out`` / ``_fit_active_view``)
were the only way to zoom: Ctrl+wheel scrolled the page (a tall webtoon strip a page at a time) and
Ctrl + / - / 0 did nothing. ``install_preview_zoom(preview, window)`` adds, through those same three
methods (same 1.15 step, same fit):

* Ctrl+= / Ctrl++ (also the numpad +) zoom in, Ctrl+- (also the numpad -) zooms out, Ctrl+0 fits
  (the Fit to View button). They act anywhere in the manga window except while a text field has
  focus (line edit, spin box, editable combo box, text view), which keeps its keys.
* Ctrl+wheel over the image zooms one step per wheel notch (a high-resolution wheel or touchpad adds
  up its small deltas), keeping the point under the mouse where it is. A plain wheel still scrolls the
  strip and Shift+wheel keeps the viewer's own zoom.
* A touchpad pinch zooms the same way: Windows sends it as Ctrl+wheel, macOS / Linux as a native zoom
  gesture.

The widget class itself stays as it is (tests/test_manga_editor_core.py pins it); this is installed
from outside, by ``manga_format_toolbar.install_format_toolbar``.
"""

import math

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit,
                               QWidget)

#: One wheel notch (``QWheelEvent.angleDelta``, 1/8 degree units): one press of a zoom button.
WHEEL_NOTCH = 120
#: A pinch zooms one step each time its scale grows / shrinks by the buttons' factor.
BUTTON_FACTOR = 1.15
ZOOM_IN_KEYS = ("Ctrl+=", "Ctrl++")
ZOOM_OUT_KEYS = ("Ctrl+-",)
FIT_KEYS = ("Ctrl+0",)
ZOOM_IN_TIP = "Zoom In (Ctrl++ / Ctrl+Wheel)"
ZOOM_OUT_TIP = "Zoom Out (Ctrl+- / Ctrl+Wheel)"
FIT_TIP = "Fit to View (Ctrl+0)"


def is_text_field(widget):
    """True for a widget that takes typed text (and so keeps Ctrl + / - / 0 for itself)."""
    if isinstance(widget, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)):
        return True
    return isinstance(widget, QComboBox) and widget.isEditable()


class PreviewZoom(QObject):
    """The zoom keys of ``preview`` (shortcuts of ``window``) and its Ctrl+wheel / pinch filter."""

    def __init__(self, preview, window=None):
        super().__init__(preview)
        self.preview = preview
        self._wheel_rest = 0
        self._pinch_rest = 0.0
        # Window-wide when the manga window is known, else while the focus is inside the preview
        self.host = window if isinstance(window, QWidget) else preview
        context = (Qt.ShortcutContext.WindowShortcut if self.host is not preview
                   else Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.shortcuts = []
        for keys, slot in ((ZOOM_IN_KEYS, preview._zoom_in), (ZOOM_OUT_KEYS, preview._zoom_out),
                           (FIT_KEYS, preview._fit_active_view)):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self.host)
                shortcut.setContext(context)
                shortcut.activated.connect(slot)
                self.shortcuts.append(shortcut)
        self.viewport = preview.viewer.viewport()
        self.viewport.installEventFilter(self)
        for button, tip in ((getattr(preview, 'zoom_in_btn', None), ZOOM_IN_TIP),
                            (getattr(preview, 'zoom_out_btn', None), ZOOM_OUT_TIP),
                            (getattr(preview, 'fit_btn', None), FIT_TIP)):
            if button is not None:
                button.setToolTip(tip)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)
            self._on_focus_changed(None, app.focusWidget())

    # ---- keys ------------------------------------------------------------------------------------------

    def _on_focus_changed(self, _old, new):
        """A text field keeps Ctrl + / - / 0: the shortcuts are off while one has focus."""
        try:
            enabled = not is_text_field(new)
            for shortcut in self.shortcuts:
                shortcut.setEnabled(enabled)
        except RuntimeError:
            pass  # the preview is gone

    # ---- wheel / pinch -----------------------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.viewport:
            kind = event.type()
            if kind == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._on_ctrl_wheel(event)
                return True
            if kind == QEvent.Type.NativeGesture and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self._on_pinch(event)
                return True
        return False

    def _on_ctrl_wheel(self, event):
        delta = event.angleDelta()
        amount = delta.y() or delta.x()
        if (amount > 0) != (self._wheel_rest > 0):
            self._wheel_rest = 0  # the direction turned: start a new notch
        self._wheel_rest += amount
        steps = int(self._wheel_rest / WHEEL_NOTCH)
        self._wheel_rest -= steps * WHEEL_NOTCH
        self.zoom_at(steps, event.position())
        event.accept()

    def _on_pinch(self, event):
        try:
            self._pinch_rest += math.log1p(float(event.value()))
        except ValueError:
            return  # a scale of -100 % or less
        step = math.log(BUTTON_FACTOR)
        steps = int(self._pinch_rest / step)
        self._pinch_rest -= steps * step
        self.zoom_at(steps, event.position())
        event.accept()

    def zoom_at(self, steps, position):
        """Press Zoom In (``steps`` > 0) or Zoom Out ``abs(steps)`` times, keeping the image point at
        ``position`` (viewport coordinates) under it."""
        if not steps:
            return
        viewer = self.preview._active_viewer()
        if not viewer.hasPhoto():
            return
        point = QPoint(round(position.x()), round(position.y()))
        anchor = viewer.mapToScene(point)
        # The buttons scale about the mouse only while Qt sees it over the view; hold the scroll
        # position instead and put the point back under the mouse below
        old_anchor = viewer.transformationAnchor()
        viewer.setTransformationAnchor(viewer.ViewportAnchor.NoAnchor)
        try:
            press = self.preview._zoom_in if steps > 0 else self.preview._zoom_out
            for _ in range(abs(steps)):
                press()
        finally:
            viewer.setTransformationAnchor(old_anchor)
        moved = viewer.mapFromScene(anchor) - point
        viewer.horizontalScrollBar().setValue(viewer.horizontalScrollBar().value() + moved.x())
        viewer.verticalScrollBar().setValue(viewer.verticalScrollBar().value() + moved.y())


def install_preview_zoom(preview, window=None):
    """Install the Ctrl zoom of ``preview`` once (see the module docstring) and return it; ``window`` is
    the manga window its keys work in (None: only while the focus is inside the preview)."""
    zoom = getattr(preview, 'preview_zoom', None)
    if zoom is None:
        zoom = PreviewZoom(preview, window)
        preview.preview_zoom = zoom
    return zoom
