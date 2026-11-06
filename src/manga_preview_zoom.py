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

It also keeps the zoom and scroll position when the page shown is reloaded (``KeepViewOnReload``): an
edit of a box (Edit Translation, the Format row, moving / resizing a box, Save & Update Overlay, a
single-box render, Clean) re-renders the page and the preview loads it again, and
``CompactImageViewer.setPhoto`` fitted every load (a zoomed webtoon strip jumped back to the whole
strip, scrolled to the top). A reload of the same page (its cleaned or rendered image included) now
keeps the view; another page still fits as before.

The widget class itself stays as it is (tests/test_manga_editor_core.py pins it); this is installed
from outside, by ``manga_format_toolbar.install_format_toolbar``.
"""

import math
import os

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
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
        self.keep_view = KeepViewOnReload(preview)

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
            if kind == QEvent.Type.NativeGesture:
                gesture = event.gestureType()
                if gesture == Qt.NativeGestureType.BeginNativeGesture:
                    self._pinch_rest = 0.0  # a new pinch: no part step left over from the last one
                elif gesture == Qt.NativeGestureType.ZoomNativeGesture:
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


def page_base(path):
    """(base name, derived) of an image the preview shows for a page: page ``001.png``'s cleaned
    image ``001_translated/001_cleaned.png`` and rendered image ``001_translated/001.png`` (also under
    the output folder) have base ``001`` and derived True, as in ``manga_editor_core``."""
    folder, name = os.path.split(os.path.normpath(path))
    stem = os.path.splitext(name)[0]
    base = stem[:-len('_cleaned')] if stem.lower().endswith('_cleaned') else stem
    derived = base != stem or os.path.basename(folder).lower().endswith('_translated')
    return os.path.normcase(base), derived


def same_page(path, other):
    """True when the two image paths are the same page: the same file, or a page and its cleaned /
    rendered image (the preview's ``current_image_path`` is either, by the caller)."""
    if not path or not other:
        return False
    if os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(other)):
        return True
    (base, derived), (other_base, other_derived) = page_base(path), page_base(other)
    return (derived or other_derived) and base == other_base


class KeepViewOnReload(QObject):
    """Keep the preview's zoom and scroll position when the page it shows is loaded again.

    ``CompactImageViewer.load_image`` emits ``image_loading`` before it sets the new image (the old one
    is still shown) and ``image_loaded`` after; ``setPhoto`` clears ``user_zoomed`` and schedules a
    fit. When the reload is of the page shown (``same_page``), the user had zoomed and nothing else
    replaced the image in between, the view is put back on ``image_loaded`` and that one scheduled fit
    is skipped. An image of another size (a page rendered at another scale) keeps the same part of the
    page in view at the same size on screen."""

    def __init__(self, preview):
        super().__init__(preview)
        self.preview = preview
        self.viewer = viewer = preview.viewer
        self._shown = None  # (page path, pixmap cacheKey) of the image the viewer got by its last load
        self._view = None  # the view to put back when the image being loaded is set
        # how many of the fits setPhoto scheduled are for kept views (skipped): two reloads in one event
        # loop turn (Clean queues a preview update and the button restore together) schedule two
        self._skip_fit = 0
        fit = viewer.fitInView

        def fit_in_view():
            if self._skip_fit:
                self._skip_fit -= 1
                return
            fit()

        # setPhoto schedules self.fitInView: the instance attribute is what its timer calls (as do
        # Fit to View and a resize, which run the fit as before)
        viewer.fitInView = fit_in_view
        viewer.image_loading.connect(self._on_loading)
        viewer.image_loaded.connect(self._on_loaded)

    def _pixmap(self):
        return self.viewer.photo.pixmap()

    def _on_loading(self, _path):
        viewer = self.viewer
        self._view = None
        try:
            shown = self._shown
            if not (shown and viewer.hasPhoto() and viewer.user_zoomed):
                return
            pixmap = self._pixmap()
            if pixmap.cacheKey() != shown[1] or not same_page(shown[0], self.preview.current_image_path):
                return
            center = viewer.mapToScene(viewer.viewport().rect().center())
            self._view = (viewer.transform(), viewer.horizontalScrollBar().value(),
                          viewer.verticalScrollBar().value(), pixmap.size(), center, viewer.zoom_level)
        except RuntimeError:
            pass  # the preview is gone

    def _on_loaded(self, _path):
        viewer = self.viewer
        view, self._view = self._view, None
        try:
            pixmap = self._pixmap()
            self._shown = (self.preview.current_image_path, pixmap.cacheKey())
            if view is None:
                return
            transform, x, y, size, center, zoom_level = view
            new = pixmap.size()
            if new == size:
                if viewer.transform() != transform:
                    viewer.setTransform(transform)
                viewer.horizontalScrollBar().setValue(x)
                viewer.verticalScrollBar().setValue(y)
            else:
                viewer.setSceneRect(viewer.photo.boundingRect())
                viewer.setTransform(transform)
                ratio = size.width() / new.width() if new.width() else 1.0
                viewer.scale(ratio, ratio)
                viewer.centerOn(QPointF(center.x() * new.width() / max(size.width(), 1),
                                        center.y() * new.height() / max(size.height(), 1)))
            viewer.user_zoomed = True
            viewer.zoom_level = zoom_level
            self._skip_fit += 1
        except RuntimeError:
            pass


def install_preview_zoom(preview, window=None):
    """Install the Ctrl zoom of ``preview`` once (see the module docstring) and return it; ``window`` is
    the manga window its keys work in (None: only while the focus is inside the preview)."""
    zoom = getattr(preview, 'preview_zoom', None)
    if zoom is None:
        zoom = PreviewZoom(preview, window)
        preview.preview_zoom = zoom
    return zoom
