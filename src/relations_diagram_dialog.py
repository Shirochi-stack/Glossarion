"""Relations & Gender Tracker: who is what to whom, and each name's gender, as known up to a chapter.

The window over a glossary's ``*_relations_tracker.json`` and ``*_gender_tracker.json``. The
graph, its layout, every edit and the per-book view choices come from ``relations_tracking``
(build_graph / layout_graph, edit_tracker and its pure edits, set_gender, save_view, undo_edit);
this module draws them and asks for the edits (PySide6 only, no manga-tier imports).

While it is open it checks the two tracker files and the glossary once a second (one os.stat
each) and reloads on the loading thread when one of them changed.
"""

import html
import math
import os
import threading

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QCompleter, QDialog, QFormLayout, QFrame,
                               QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QMenu, QPlainTextEdit,
                               QPushButton, QSlider, QSpinBox, QSplitter, QStackedWidget, QTextBrowser,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

import relations_tracking
from qt_widget_helpers import CONTEXT_MENU_STYLE, CheckmarkDelegate, HalgakosLoadingWidget

WINDOW_TITLE = "Relations & Gender Tracker"
ALL_CHARACTERS = "All characters"
LOADING_TEXT = "Loading relations…"
HISTORY_HINT = "Click a character to see their relation history."
EMPTY_TEXT = ("No relations tracked for this glossary yet.\n\n"
              "Turn on Enable Relations Tracker in Glossary Manager → General Settings, "
              "then run Balanced or Full glossary extraction.")
#: How often the open window checks whether the tracker files or the glossary changed.
POLL_MS = 1000
#: How many edits Undo can take back while the window is open.
UNDO_LIMIT = 30
#: How long a notice ("Undone: ...", "Could not save ...") stays beside the Undo button.
NOTICE_MS = 8000

#: Node tint by gender. Female uses the pink the Glossary Editor gives tracked-gender rows,
#: male the dialog accent blue; anything else stays neutral grey.
GENDER_TINTS = {
    'female': QColor("#ec4899"),
    'male': QColor("#5a9fd4"),
}
NEUTRAL_TINT = QColor("#6b7280")
BACKGROUND = QColor("#1b1d21")
EDGE_COLOR = QColor("#8b95a5")
SELECTED_EDGE_COLOR = QColor("#e5e7eb")
CHANGED_COLOR = QColor("#f59e0b")  # amber: the relation changed
LABEL_BG = QColor("#1b1d21")
#: Above this many visible edges, labels are drawn only for the selected character's edges.
MAX_LABELLED_EDGES = 120
#: How long a name stays outlined after a jump from the side panel.
FLASH_MS = 1600
#: Lines and names that do not belong to the selected character fade to this.
DIMMED_OPACITY = 0.22
#: Where along its line a label may sit (middle first), so it never floats beside the line.
LABEL_SPOTS = (0.5, 0.42, 0.58, 0.35, 0.65, 0.28, 0.72)
#: What a diagram item is (its data(0)): "node<SEP>id" or "edge<SEP>a<SEP>b".
_KEY_SEP = "\x1f"

SMALL_BUTTON_STYLE = ("QPushButton { background-color: #2d2d2d; color: white; border: 1px solid #4a5568;"
                      " border-radius: 3px; padding: 3px 10px; }"
                      " QPushButton:hover { border-color: #5a9fd4; }"
                      " QPushButton:disabled { color: #6b7280; border-color: #3a3f47; }")

#: The edit dialogs: the Glossary Manager's dark dialogs (inputs #2d2d2d, borders #4a5568, accent #5a9fd4).
EDIT_DIALOG_STYLE = (
    "QDialog { background-color: #1e1e1e; color: white; }"
    " QLabel { color: white; }"
    " QLineEdit, QPlainTextEdit, QSpinBox { background-color: #2d2d2d; color: white;"
    "  border: 1px solid #4a5568; border-radius: 3px; padding: 4px; }"
    " QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus { border-color: #5a9fd4; }"
    " QComboBox { background-color: #2d2d2d; color: white; border: 1px solid #4a5568;"
    "  border-radius: 3px; padding: 4px 6px; }"
    " QComboBox QAbstractItemView { background-color: #2d2d2d; color: white;"
    "  selection-background-color: #5a9fd4; }"
    " QPushButton { background-color: #3a3a3a; color: white; border: 1px solid #4a5568;"
    "  border-radius: 3px; padding: 5px 16px; }"
    " QPushButton:hover { border-color: #5a9fd4; }"
    " QPushButton#primary { background-color: #3a5f8a; border-color: #5a9fd4; }"
)

LINK_STYLE = "color:#7bb3e0;text-decoration:none;font-size:8pt"


def _over_background(tint, strength):
    """``tint`` painted over the diagram background at ``strength``: an opaque fill, so the
    lines running behind a name do not show through it."""
    return QColor(
        round(BACKGROUND.red() + (tint.red() - BACKGROUND.red()) * strength),
        round(BACKGROUND.green() + (tint.green() - BACKGROUND.green()) * strength),
        round(BACKGROUND.blue() + (tint.blue() - BACKGROUND.blue()) * strength),
    )


def _styled_checkbox(text):
    """The app's checkbox with the drawn tick (the Library's shared factory)."""
    try:
        from epub_library import _create_styled_checkbox
        return _create_styled_checkbox(text)
    except Exception:
        return QCheckBox(text)


def _style_combo(combo):
    """The Glossary Manager's profile dropdown look (drop-down area and the ▼ overlay)."""
    try:
        from GlossaryManager_GUI import GlossaryManagerMixin
        # Both helpers it calls are static, so the class itself can stand in for ``self``.
        GlossaryManagerMixin._style_glossary_profile_combo(GlossaryManagerMixin, combo)
    except Exception:
        pass


def _gender_key(value):
    text = str(value or '').strip().casefold()
    if text in ('female', 'f', 'woman', 'girl'):
        return 'female'
    if text in ('male', 'm', 'man', 'boy'):
        return 'male'
    return ''


def _item_key(item):
    """``("node", id)`` / ``("edge", a, b)`` for a diagram item (or its text), else None."""
    while item is not None:
        value = item.data(0)
        if isinstance(value, str) and _KEY_SEP in value:
            return tuple(value.split(_KEY_SEP))
        item = item.parentItem()
    return None


def _chapter_text(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number == int(number) else str(number)


#: ``"Father / Daughter"`` edge labels (shared with the mobile diagram).
edge_label_text = relations_tracking.edge_label_text


class _GraphView(QGraphicsView):
    """Wheel zoom around the cursor, drag to pan.

    The whole diagram is fitted to the window until the user zooms or pans; from then on
    the view is theirs. A click on empty space calls ``on_background_click``; a right-click
    on a name or a relation calls ``on_item_menu(key, global_pos)``, a double-click on a
    relation ``on_item_double(key)`` (both after the event, since they may rebuild the scene).
    """

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.on_background_click = None
        self.on_item_menu = None
        self.on_item_double = None
        self._auto_fit = True
        self._press_pos = None
        self._content_rect = QRectF()
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setBackgroundBrush(QBrush(BACKGROUND))
        # Panning is by drag; scroll bars only added two grey bands around the picture.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QGraphicsView { border: 1px solid #4a5568; }")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            return super().wheelEvent(event)
        factor = 1.15 if delta > 0 else 1 / 1.15
        current = self.transform().m11()
        if (factor > 1 and current > 6.0) or (factor < 1 and current < 0.05):
            return
        self._auto_fit = False
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        pressed, self._press_pos = self._press_pos, None
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton or pressed is None:
            return
        released = event.position().toPoint()
        if (released - pressed).manhattanLength() > 4:
            self._auto_fit = False  # a drag: the user moved the picture
        elif self.itemAt(released) is None and self.on_background_click is not None:
            self.on_background_click()

    def mouseDoubleClickEvent(self, event):
        key = _item_key(self.itemAt(event.position().toPoint()))
        if key is not None:
            # A relation opens its editor; a name keeps the selection its first click made.
            if key[0] == 'edge' and self.on_item_double is not None:
                handler = self.on_item_double
                QTimer.singleShot(0, lambda: handler(key))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        key = _item_key(self.itemAt(event.pos()))
        if key is not None and self.on_item_menu is not None:
            handler, where = self.on_item_menu, QPoint(event.globalPos())
            QTimer.singleShot(0, lambda: handler(key, where))
            event.accept()
            return
        super().contextMenuEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_all()

    def showEvent(self, event):
        super().showEvent(event)
        if self._auto_fit:
            QTimer.singleShot(0, self.fit_all)  # after the first layout pass gave the real size

    def set_content_rect(self, rect):
        """The area that must stay in sight (every name of the layout, not only those drawn now)."""
        self._content_rect = QRectF(rect)

    def fit_all(self):
        """Show the whole diagram and keep it fitted until the user zooms or pans."""
        self._auto_fit = True
        rect = QRectF(self._content_rect)
        if rect.isEmpty():
            rect = self.scene().itemsBoundingRect()
        if rect.isValid() and not rect.isEmpty():
            self.fitInView(rect.adjusted(-24, -24, 24, 24), Qt.KeepAspectRatio)
            if self.transform().m11() > 1.4:  # do not blow a few nodes up to giant size
                self.resetTransform()
                self.scale(1.4, 1.4)
            self.centerOn(rect.center())

    def bring_into_view(self, item):
        """Centre ``item`` when it is not wholly in sight (the user zoomed in or panned away).
        A diagram that already shows it is left where it is. True when the view moved."""
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        if visible.contains(item.sceneBoundingRect()):
            return False
        self._auto_fit = False
        self.centerOn(item)
        return True


class _NodeItem(QGraphicsPathItem):
    """A rounded name label; a click shows that character's history."""

    def __init__(self, node, on_click, font):
        super().__init__()
        self.node_id = node['id']
        self._on_click = on_click
        label = str(node.get('label') or node.get('raw_name') or node['id'])
        self.text_item = QGraphicsSimpleTextItem(label, self)
        self.text_item.setFont(font)
        self.text_item.setBrush(QBrush(QColor("#ffffff")))
        text_rect = self.text_item.boundingRect()
        width, height = text_rect.width() + 18, text_rect.height() + 10
        path = QPainterPath()
        path.addRoundedRect(QRectF(-width / 2, -height / 2, width, height), 8, 8)
        self.setPath(path)
        self.text_item.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
        self.tint = GENDER_TINTS.get(_gender_key(node.get('gender')), NEUTRAL_TINT)
        listed = node.get('listed', True)
        self.setBrush(QBrush(_over_background(self.tint, 0.6 if listed else 0.28)))
        self._border = self.tint.lighter(130)
        style = Qt.SolidLine if listed else Qt.DashLine
        if node.get('gender_only'):
            style = Qt.DotLine  # tracked for gender, no relation (yet)
        self.setPen(QPen(self._border, 1.4, style))
        tip = str(node.get('raw_name') or '')
        if not listed:
            tip += " (not in the glossary)"
        if node.get('gender_only'):
            tip += " (gender tracked, no relations)"
        self.setToolTip(tip)
        self.setData(0, f"node{_KEY_SEP}{self.node_id}")
        self.setZValue(10)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def set_selected_look(self, selected):
        self._selected_look = bool(selected)
        pen = self.pen()
        pen.setWidthF(3.0 if selected else 1.4)
        pen.setColor(QColor("#ffffff") if selected else self._border)
        self.setPen(pen)

    def set_flash(self, on):
        """A thick amber outline for a moment: "here it is" after a jump from the side panel."""
        if on:
            pen = self.pen()
            pen.setWidthF(4.5)
            pen.setColor(CHANGED_COLOR)
            self.setPen(pen)
        else:
            self.set_selected_look(getattr(self, '_selected_look', False))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._on_click is not None:
            # After this handler returns: the click may rebuild the scene, which deletes
            # this item, and Qt must not be left holding it as the mouse grabber.
            on_click, node_id = self._on_click, self.node_id
            QTimer.singleShot(0, lambda: on_click(node_id))
            event.accept()
            return
        super().mousePressEvent(event)


class _EditDialog(QDialog):
    """A small dark form (Save / Cancel) for one edit."""

    def __init__(self, parent, title, intro=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setStyleSheet(EDIT_DIALOG_STYLE)
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        if intro:
            heading = QLabel(intro)
            heading.setWordWrap(True)
            heading.setStyleSheet("font-weight: bold; color: #5a9fd4;")
            layout.addWidget(heading)
        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(8)
        layout.addLayout(self.form)
        self._notes = QVBoxLayout()
        layout.addLayout(self._notes)
        #: The input widgets in the order they were added.
        self.fields = []
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("primary")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def line(self, label, text="", placeholder=""):
        widget = QLineEdit(str(text or ""))
        widget.setPlaceholderText(placeholder)
        widget.setClearButtonEnabled(True)
        self.form.addRow(label, widget)
        self.fields.append(widget)
        return widget

    def text(self, label, text=""):
        widget = QPlainTextEdit(str(text or ""))
        widget.setMinimumHeight(110)
        self.form.addRow(label, widget)
        self.fields.append(widget)
        return widget

    def spin(self, label, value, low=0, high=100000):
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setValue(int(value) if value is not None else low)
        self.form.addRow(label, widget)
        self.fields.append(widget)
        return widget

    def combo(self, label, items, current=None, editable=False, placeholder=""):
        """``items``: ``[(text, data)]``. Editable: type to search (contains), or a new name."""
        widget = QComboBox()
        for text, data in items:
            widget.addItem(text, data)
        if editable:
            widget.setEditable(True)
            widget.setInsertPolicy(QComboBox.NoInsert)
            completer = QCompleter(widget.model(), widget)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            widget.setCompleter(completer)
            widget.setCurrentIndex(-1)
            widget.lineEdit().setPlaceholderText(placeholder)
        if current is not None:
            index = widget.findData(current)
            if index >= 0:
                widget.setCurrentIndex(index)
        _style_combo(widget)
        self.form.addRow(label, widget)
        self.fields.append(widget)
        return widget

    def note(self, text):
        widget = QLabel(text)
        widget.setWordWrap(True)
        widget.setStyleSheet("color: #9ca3af;")
        self._notes.addWidget(widget)
        return widget


class _HidePopup(QFrame):
    """The Hide control's searchable checklist: a checked name is not drawn. "Show all" unchecks all."""

    hidden_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background-color: #252525; border: 1px solid #5a9fd4; }"
            " QLabel { color: #9ca3af; border: none; }"
            " QLineEdit { background-color: #2d2d2d; color: white; border: 1px solid #4a5568;"
            "  border-radius: 3px; padding: 4px; }"
            " QTreeWidget { background-color: #252525; color: white; border: none; }"
            " QTreeWidget::item { padding: 2px 0; }"
            " QTreeWidget::item:hover { background-color: #2f3a48; }" + SMALL_BUTTON_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search names…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 26)
        self.tree.setItemDelegateForColumn(0, CheckmarkDelegate(self.tree))
        self.tree.itemChanged.connect(lambda *_args: self._emit())
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree, 1)
        bottom = QHBoxLayout()
        hint = QLabel("Checked names are hidden")
        bottom.addWidget(hint)
        bottom.addStretch()
        self.show_all_button = QPushButton("Show all")
        self.show_all_button.clicked.connect(self.show_all)
        bottom.addWidget(self.show_all_button)
        layout.addLayout(bottom)
        self.resize(280, 360)
        self._ids = []

    def set_names(self, names, hidden):
        """``names``: ``[{"id", "label", "raw_name"}]`` in the order listed; ``hidden``: ids."""
        hidden = set(hidden or ())
        blocked = self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self._ids = []
            for name in names:
                text = str(name.get('label') or name['id'])
                raw = str(name.get('raw_name') or '')
                if raw and raw != text:
                    text = f"{text} ({raw})"
                item = QTreeWidgetItem(["", text])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if name['id'] in hidden else Qt.Unchecked)
                item.setData(1, Qt.UserRole, name['id'])
                self.tree.addTopLevelItem(item)
                self._ids.append(name['id'])
        finally:
            self.tree.blockSignals(blocked)
        self._filter(self.search.text())

    def hidden_ids(self):
        result = []
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.checkState(0) == Qt.Checked:
                result.append(item.data(1, Qt.UserRole))
        return result

    def set_checked(self, node_id, hidden):
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.data(1, Qt.UserRole) == node_id:
                item.setCheckState(0, Qt.Checked if hidden else Qt.Unchecked)  # emits
                return True
        return False

    def show_all(self):
        blocked = self.tree.blockSignals(True)
        try:
            for index in range(self.tree.topLevelItemCount()):
                self.tree.topLevelItem(index).setCheckState(0, Qt.Unchecked)
        finally:
            self.tree.blockSignals(blocked)
        self._emit()

    def _on_item_clicked(self, item, column):
        if column == 1:  # the name toggles too, not only the box
            item.setCheckState(0, Qt.Unchecked if item.checkState(0) == Qt.Checked else Qt.Checked)

    def _filter(self, text):
        wanted = str(text or '').strip().casefold()
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            item.setHidden(bool(wanted) and wanted not in item.text(1).casefold())

    def _emit(self):
        self.hidden_changed.emit(self.hidden_ids())

    def popup_below(self, widget):
        self.move(widget.mapToGlobal(QPoint(0, widget.height() + 2)))
        self.show()
        self.search.setFocus()


class RelationsDiagramDialog(QDialog):
    """Graph of the tracked relations and genders with a chapter slider, focus search,
    filters, a history panel and hand edits.

    ``load_async=True`` (the Glossary Manager's buttons): the window is built empty with a
    loading screen and returns at once; the trackers, the glossary and the layout are worked
    out on a thread and the diagram appears when they are ready. Otherwise everything is
    ready when the constructor returns.

    With a ``glossary_path`` the window edits the tracker files (each edit is one locked
    write through ``relations_tracking``), remembers the view choices in the tracker, and
    reloads by itself when an extraction (or anything else) changes the files.
    """

    #: A loading-thread result (a dict: the initial load, a reload, or a new layout).
    _loaded = Signal(object)

    def __init__(self, parent=None, glossary_path=None, tracker=None, glossary_entries=None, load_async=False):
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1200, 780)
        self.setStyleSheet(
            "QDialog { background-color: #1e1e1e; color: white; }"
            " QLabel, QCheckBox { color: white; }"
            " QTextBrowser { background-color: #252525; color: #e5e7eb; border: 1px solid #4a5568; }"
            " QComboBox { background-color: #2d2d2d; color: white; border: 1px solid #4a5568;"
            "  border-radius: 3px; padding: 3px 6px; }"
            " QComboBox QAbstractItemView { background-color: #2d2d2d; color: white;"
            "  selection-background-color: #5a9fd4; }"
        )
        try:
            from PySide6.QtGui import QIcon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass

        self.glossary_path = glossary_path or ''
        # Path arithmetic only: nothing is read before the window is shown.
        self.tracker_path = relations_tracking.tracker_path_for_glossary(self.glossary_path) if self.glossary_path else ''
        self.gender_path = relations_tracking.gender_tracker_path_for_glossary(self.glossary_path) if self.glossary_path else ''
        # Reading the files and laying the graph out is the slow part (over a second on a
        # 15-name book before the layout was sped up): with load_async it happens after the
        # window is on screen.
        self._loading = bool(load_async) and tracker is None and bool(self.glossary_path)
        self._stamps = {}
        self.gender_tracker = None
        if self._loading:
            tracker, glossary_entries = relations_tracking.empty_tracker(), None
        else:
            if self.glossary_path:
                self._stamps = self._stamps_now()
            loaded_from_file = tracker is None
            tracker, glossary_entries = self._read_data(self.glossary_path, tracker, glossary_entries)
            if loaded_from_file and self.glossary_path:
                self.gender_tracker = self._read_gender(self.glossary_path)
        self.tracker = tracker
        self.glossary_entries = glossary_entries
        self._glossary_index = self._index_glossary(glossary_entries)

        self._node_items = {}
        self._edge_count = 0
        self._positions = {}
        self._layout_key = None
        self._refit = False
        self._label_boxes = []
        self._history_links = []
        self._history_actions = []
        self._selected = None
        self._node_info = {}
        self._reloading = False
        self._data_gen = 0
        self._dialogs_open = 0
        self._undo = []
        self._notice_token = 0
        self._hide_popup = None
        self._node_font = QFont("Segoe UI", 10)
        self._node_font.setBold(True)
        self._label_font = QFont("Segoe UI", 8)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._loaded.connect(self._on_loaded)

        self._build_shell()
        if self._loading:
            self._stack.setCurrentWidget(self.loading_widget)
            threading.Thread(target=self._load_in_background, name="RelationsDiagramLoad", daemon=True).start()
        else:
            self._build_diagram_page()
            self.rebuild()

    # ── data ────────────────────────────────────────────────────────────
    @classmethod
    def _read_data(cls, glossary_path, tracker=None, glossary_entries=None):
        """``(tracker, glossary entries)``: what was passed in, else read from beside ``glossary_path``."""
        if tracker is None:
            tracker_path = relations_tracking.tracker_path_for_glossary(glossary_path) if glossary_path else ''
            tracker = relations_tracking.load_tracker(tracker_path) if tracker_path else relations_tracking.empty_tracker()
        if not isinstance(tracker, dict):
            tracker = relations_tracking.empty_tracker()
        if glossary_entries is None and glossary_path:
            glossary_entries = cls._load_glossary_entries(glossary_path)
        return tracker, (list(glossary_entries) if glossary_entries is not None else None)

    @staticmethod
    def _read_gender(glossary_path):
        """The book's gender tracker (empty when there is none), None when it cannot be read now."""
        return relations_tracking.read_gender_tracker(relations_tracking.gender_tracker_path_for_glossary(glossary_path))

    @staticmethod
    def _index_glossary(entries):
        index = {}
        for entry in entries or []:
            if isinstance(entry, dict) and str(entry.get('raw_name') or '').strip():
                index.setdefault(relations_tracking.tracker_key(entry.get('raw_name')), entry)
        return index

    def _stamps_now(self):
        """``{path: (mtime_ns, size) or None}`` of the two trackers and the glossary: three os.stat calls."""
        return {path: relations_tracking.file_stamp(path)
                for path in (self.tracker_path, self.gender_path, self.glossary_path) if path}

    @staticmethod
    def _layout_size(node_count):
        width = max(900.0, 230.0 * math.sqrt(max(1, node_count)) * 1.6)
        return width, width * 0.68

    @staticmethod
    def _full_graph(tracker, glossary_entries, gender_tracker, show_unlisted):
        """Every name of every chapter (gender-only names included): what the layout is made of."""
        return relations_tracking.build_graph(tracker, glossary_entries, None, show_unlisted=show_unlisted,
                                              gender_tracker=gender_tracker,
                                              include_gender_only=gender_tracker is not None)

    @classmethod
    def _compute_layout(cls, tracker, glossary_entries, show_unlisted, focus, gender_tracker=None):
        """Positions from the graph of ALL chapters, so nodes stay put while the slider moves
        and the filters hide names."""
        full = cls._full_graph(tracker, glossary_entries, gender_tracker, show_unlisted)
        width, height = cls._layout_size(len(full.get('nodes') or []))
        return relations_tracking.layout_graph(full, width, height, focus=focus)

    def _load_in_background(self):
        """The loading thread: plain data only, no Qt objects. The result goes back by signal."""
        try:
            stamps = self._stamps_now()  # before reading: a write meanwhile is seen at the next check
            tracker, entries = self._read_data(self.glossary_path)
            gender = self._read_gender(self.glossary_path)
            key = (False, None)  # the layout the diagram opens with
            payload = {'initial': True, 'stamps': stamps, 'tracker': tracker, 'entries': entries,
                       'gender': gender, 'key': key,
                       'positions': self._compute_layout(tracker, entries, *key, gender_tracker=gender)}
        except Exception:
            payload = {'initial': True, 'stamps': {}, 'tracker': relations_tracking.empty_tracker(),
                       'entries': None, 'gender': None, 'key': None, 'positions': {}}
        self._emit_loaded(payload)

    def _emit_loaded(self, payload):
        try:
            self._loaded.emit(payload)
        except RuntimeError:  # the window was closed before the data was ready
            pass

    def _on_loaded(self, payload):
        if not payload.get('initial'):
            self._on_job_done(payload)
            return
        self.tracker, self.glossary_entries = payload['tracker'], payload['entries']
        self.gender_tracker = payload.get('gender')
        self._glossary_index = self._index_glossary(self.glossary_entries)
        self._stamps = dict(payload.get('stamps') or {})
        self._loading = False
        self._build_diagram_page()
        if payload.get('key') is not None:
            self._install_layout(payload['key'], payload['positions'])
        self.rebuild()
        if self.isVisible():
            self._poll_timer.start()

    def is_loading(self):
        return self._loading

    def is_busy(self):
        """Loading, or a reload / new layout is being worked out on the loading thread."""
        return self._loading or self._reloading

    @staticmethod
    def _load_glossary_entries(path):
        try:
            import glossary_document
            parsed = glossary_document.parse_glossary_file(path, {}, None)
            entries = parsed.get('entries') or []
            return [e for e in entries if isinstance(e, dict)]
        except Exception:
            return None

    def _has_relations(self):
        entries = self.tracker.get('entries') if isinstance(self.tracker, dict) else None
        if not isinstance(entries, dict):
            return False
        return any(isinstance(item, dict) and item.get('relations') for item in entries.values())

    def _has_content(self):
        """Something to draw: relations, or names with gender tracking."""
        if self._has_relations():
            return True
        entries = self.gender_tracker.get('entries') if isinstance(self.gender_tracker, dict) else None
        return bool(entries) and bool(self._node_info)

    # ── auto-update ─────────────────────────────────────────────────────
    def showEvent(self, event):
        super().showEvent(event)
        if self.glossary_path and not self._loading:
            self._poll_timer.start()
            QTimer.singleShot(0, self._poll)  # catch up on what changed while it was hidden

    def hideEvent(self, event):
        self._poll_timer.stop()
        super().hideEvent(event)

    def _poll(self):
        """The once-a-second check: reload when a tracker file or the glossary changed since the
        last load (or this window's own last write). True when a reload was started."""
        if self._loading or self._reloading or self._dialogs_open or not self.glossary_path or self._page is None:
            return False
        stamps = self._stamps_now()
        if stamps == self._stamps:
            return False
        self._start_job(stamps)
        return True

    def _start_job(self, stamps=None):
        """Reload the files whose stamp changed (``stamps``), or only lay the current data out
        again (None), on the loading thread. The picture on screen stays until the result is in."""
        if self._reloading:
            return False
        stamps = stamps or {}
        job = {'initial': False, 'gen': self._data_gen, 'stamps': stamps,
               'read_tracker': bool(stamps) and stamps.get(self.tracker_path) != self._stamps.get(self.tracker_path),
               'read_gender': bool(stamps) and stamps.get(self.gender_path) != self._stamps.get(self.gender_path),
               'read_glossary': bool(stamps) and stamps.get(self.glossary_path) != self._stamps.get(self.glossary_path),
               'tracker': self.tracker, 'gender': self.gender_tracker, 'entries': self.glossary_entries,
               'key': (self.show_unlisted_checkbox.isChecked(), self.focus_id())}
        self._reloading = True
        threading.Thread(target=self._job_in_background, args=(job,), name="RelationsDiagramReload",
                         daemon=True).start()
        return True

    def _job_in_background(self, job):
        """The loading thread for a reload: a file that cannot be read now comes back None (the
        window keeps what it shows and tries again at the next check)."""
        out = dict(job)
        try:
            if job['read_tracker']:
                out['new_tracker'] = relations_tracking.read_tracker(self.tracker_path)
            if job['read_gender']:
                out['new_gender'] = self._read_gender(self.glossary_path)
            if job['read_glossary']:
                out['new_entries'] = self._load_glossary_entries(self.glossary_path)
            tracker = out.get('new_tracker') or job['tracker']
            gender = out['new_gender'] if out.get('new_gender') is not None else job['gender']
            entries = out['new_entries'] if out.get('new_entries') is not None else job['entries']
            usable = (not (job['read_tracker'] or job['read_gender'] or job['read_glossary'])
                      or out.get('new_tracker') is not None or out.get('new_gender') is not None
                      or out.get('new_entries') is not None)
            out['positions'] = self._compute_layout(tracker, entries, *job['key'], gender_tracker=gender) if usable else None
        except Exception:
            out['positions'] = None
        self._emit_loaded(out)

    def _on_job_done(self, payload):
        self._reloading = False
        current = payload.get('gen') == self._data_gen
        positions = payload.get('positions')
        if current:
            stamps = payload.get('stamps') or {}
            for flag, field, path in (('read_tracker', 'new_tracker', self.tracker_path),
                                      ('read_gender', 'new_gender', self.gender_path),
                                      ('read_glossary', 'new_entries', self.glossary_path)):
                if not payload.get(flag):
                    continue
                value = payload.get(field)
                if value is None:
                    continue  # mid-write or unreadable: keep what is shown, try again next time
                if field == 'new_tracker':
                    self.tracker = value
                elif field == 'new_gender':
                    self.gender_tracker = value
                else:
                    self.glossary_entries = value
                    self._glossary_index = self._index_glossary(value)
                self._stamps[path] = stamps.get(path)
        else:
            positions = None  # this window wrote meanwhile: the layout was made for older data
        self._after_data_change(payload.get('key'), positions)

    def _after_data_change(self, key=None, positions=None):
        """The trackers or the glossary changed (a reload, an edit, an undo): redraw, keeping the
        slider, the focus, the filters, the selection and the user's zoom and pan."""
        if self._page is None:
            return
        self._update_chapter_range()
        self._fill_focus_combo()
        self._sync_view_controls()
        if self._selected is not None and self._selected not in self._node_info:
            self._selected = None
            self.history_title.setText(HISTORY_HINT)
            self.history_panel.clear()
        current_key = (self.show_unlisted_checkbox.isChecked(), self.focus_id())
        if self._layout_key == current_key:
            full = self._full_graph(self.tracker, self.glossary_entries, self.gender_tracker, current_key[0])
            ids = {n['id'] for n in full.get('nodes') or []}
            if ids != set(self._positions):
                if key == current_key and positions is not None and set(positions) == ids:
                    self._install_layout(current_key, positions, refit=self._view._auto_fit)
                else:
                    self._start_job()  # a name came or went: lay out again; the names shown stay put meanwhile
        self.rebuild()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_shell(self):
        """What the window needs to appear: the empty-state text and the loading screen."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        self.empty_label = QLabel(EMPTY_TEXT)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet("color: #9ca3af; font-size: 11pt;")
        self._stack.addWidget(self.empty_label)

        self.loading_widget = HalgakosLoadingWidget(LOADING_TEXT)
        self._stack.addWidget(self.loading_widget)
        self._page = None

    def _build_diagram_page(self):
        """The controls, the diagram and the history panel (needs the tracker: chapter range, names)."""
        page = self._page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(page)

        controls = QHBoxLayout()
        page_layout.addLayout(controls)
        self._chapter_low = self._chapter_high = None
        self.chapter_label = QLabel("")
        self.chapter_label.setMinimumWidth(130)
        controls.addWidget(self.chapter_label)
        self.chapter_slider = QSlider(Qt.Horizontal)
        self.chapter_slider.setMinimumWidth(220)
        self.chapter_slider.setToolTip("Show only what was known up to this chapter")
        self._update_chapter_range(initial=True)
        self.chapter_slider.valueChanged.connect(self._on_controls_changed)
        controls.addWidget(self.chapter_slider, 1)

        controls.addWidget(QLabel("Focus:"))
        self.focus_combo = QComboBox()
        self.focus_combo.setEditable(True)
        self.focus_combo.setInsertPolicy(QComboBox.NoInsert)
        self.focus_combo.setMinimumWidth(200)
        self.focus_combo.setToolTip("Type to search; pick a character to place them in the centre")
        completer = QCompleter(self.focus_combo.model(), self.focus_combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.focus_combo.setCompleter(completer)
        self.focus_combo.currentIndexChanged.connect(self._on_controls_changed)
        controls.addWidget(self.focus_combo)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setStyleSheet(SMALL_BUTTON_STYLE)
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Nothing to undo")
        self.undo_button.clicked.connect(self.undo)
        controls.addWidget(self.undo_button)
        self._undo_shortcut = QShortcut(QKeySequence(QKeySequence.Undo), self)
        self._undo_shortcut.activated.connect(self.undo)

        filters = QHBoxLayout()
        page_layout.addLayout(filters)
        self.show_former_checkbox = _styled_checkbox("Show former relations")
        self.show_former_checkbox.setToolTip("Add earlier labels in brackets, e.g. Enemy [Best Friend]")
        self.show_former_checkbox.toggled.connect(self._on_controls_changed)
        filters.addWidget(self.show_former_checkbox)
        self.show_unlisted_checkbox = _styled_checkbox("Show unlisted names")
        self.show_unlisted_checkbox.setToolTip("Also show names that are not (or no longer) in the glossary")
        self.show_unlisted_checkbox.toggled.connect(self._on_controls_changed)
        filters.addWidget(self.show_unlisted_checkbox)
        filters.addSpacing(12)
        self.show_relations_checkbox = _styled_checkbox("Relations")
        self.show_relations_checkbox.setToolTip("Draw the relation lines and labels and list the relation history")
        self.show_relations_checkbox.toggled.connect(lambda on: self.set_view(show_relations=bool(on)))
        filters.addWidget(self.show_relations_checkbox)
        self.show_gender_checkbox = _styled_checkbox("Gender")
        self.show_gender_checkbox.setToolTip("Tint names by gender at the slider's chapter, show the gender "
                                             "history and the names tracked for gender only")
        self.show_gender_checkbox.toggled.connect(lambda on: self.set_view(show_gender=bool(on)))
        filters.addWidget(self.show_gender_checkbox)
        self.hide_button = QPushButton("Hide…")
        self.hide_button.setStyleSheet(SMALL_BUTTON_STYLE)
        self.hide_button.setToolTip("Pick names to leave out of the diagram (remembered for this book)")
        self.hide_button.clicked.connect(self.open_hide_popup)
        filters.addWidget(self.hide_button)
        filters.addStretch(1)
        self.notice_label = QLabel("")
        self.notice_label.setStyleSheet("color: #f59e0b;")
        filters.addWidget(self.notice_label)

        splitter = QSplitter(Qt.Horizontal)
        page_layout.addWidget(splitter, 1)
        self.scene = QGraphicsScene(self)
        self._view = _GraphView(self.scene)
        self._view.on_background_click = self.clear_selection
        self._view.on_item_menu = self._on_item_menu
        self._view.on_item_double = self._on_item_double
        splitter.addWidget(self._view)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(6, 0, 0, 0)
        self.history_title = QLabel(HISTORY_HINT)
        self.history_title.setWordWrap(True)
        self.history_title.setStyleSheet("font-weight: bold; color: #5a9fd4;")
        side_layout.addWidget(self.history_title)
        self.history_panel = QTextBrowser()
        self.history_panel.setOpenLinks(False)
        # A name in the list selects that character; the small links edit or delete a row.
        self.history_panel.anchorClicked.connect(self._on_history_link)
        self.history_panel.setToolTip("Click a name to show that character; edit / delete change the tracker")
        side_layout.addWidget(self.history_panel, 1)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([880, 300])

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #9ca3af;")
        page_layout.addWidget(self.status_label)

        self._fill_focus_combo()
        self._sync_view_controls()

    def _chapter_bounds(self):
        """The slider's range: the relations tracker's chapters and the gender tracker's."""
        low, high = relations_tracking.chapter_bounds(self.tracker)
        numbers = [n for n in (low, high) if isinstance(n, (int, float))]
        entries = self.gender_tracker.get('entries') if isinstance(self.gender_tracker, dict) else None
        for item in (entries.values() if isinstance(entries, dict) else ()):
            for occurrence in (item.get('occurrences') or []) if isinstance(item, dict) else []:
                try:
                    numbers.append(float(occurrence.get('chapter_num')))
                except (TypeError, ValueError, AttributeError):
                    pass
        if not numbers:
            return None, None
        return int(math.floor(min(numbers))), int(math.ceil(max(numbers)))

    def _update_chapter_range(self, initial=False):
        """Set the slider's range from the data. At the last chapter it stays at the (new) last
        chapter; elsewhere it keeps its chapter."""
        slider = self.chapter_slider
        at_end = initial or not slider.isEnabled() or self._chapter_high is None or slider.value() >= self._chapter_high
        old = slider.value()
        low, high = self._chapter_bounds()
        self._chapter_low, self._chapter_high = low, high
        blocked = slider.blockSignals(True)
        try:
            if low is not None and high is not None:
                slider.setRange(low, high)
                slider.setValue(high if at_end else min(max(old, low), high))
                slider.setEnabled(True)
            else:
                slider.setEnabled(False)
        finally:
            slider.blockSignals(blocked)

    def _fill_focus_combo(self):
        """The Focus list (every name of every chapter, unlisted ones too); keeps the current pick."""
        graph = self._full_graph(self.tracker, self.glossary_entries, self.gender_tracker, True)
        nodes = sorted(graph.get('nodes') or [], key=lambda n: str(n.get('label') or '').casefold())
        self._node_info = {n['id']: n for n in nodes}
        current = self.focus_combo.currentData() if self.focus_combo.count() else None
        blocked = self.focus_combo.blockSignals(True)
        try:
            self.focus_combo.clear()
            self.focus_combo.addItem(ALL_CHARACTERS, None)
            for node in nodes:
                text = str(node.get('label') or node['id'])
                raw = str(node.get('raw_name') or '')
                if raw and raw != text:
                    text = f"{text} ({raw})"
                self.focus_combo.addItem(text, node['id'])
            index = self.focus_combo.findData(current) if current is not None else 0
            self.focus_combo.setCurrentIndex(max(0, index))
        finally:
            self.focus_combo.blockSignals(blocked)

    # ── state ───────────────────────────────────────────────────────────
    def up_to_chapter(self):
        if not self.chapter_slider.isEnabled():
            return None
        value = self.chapter_slider.value()
        return None if value >= (self._chapter_high or value) else value

    def edit_chapter(self):
        """The chapter a hand edit goes to by default: the slider's."""
        chapter = self.up_to_chapter()
        return chapter if chapter is not None else self._chapter_high

    def focus_id(self):
        return self.focus_combo.currentData()

    def edge_count(self):
        return self._edge_count

    def node_count(self):
        return len(self._node_items)

    def view_settings(self):
        """``{"hidden", "show_relations", "show_gender"}`` (kept in the tracker file)."""
        return relations_tracking.view_settings(self.tracker)

    def can_edit(self):
        return bool(self.tracker_path)

    def _on_controls_changed(self, *_args):
        self.rebuild()

    def _install_layout(self, key, positions, refit=True):
        self._positions = positions
        self._layout_key = key
        self._refit = bool(refit)
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            # Room for the name boxes around the outermost positions.
            self._view.set_content_rect(QRectF(min(xs) - 90, min(ys) - 26,
                                               max(xs) - min(xs) + 180, max(ys) - min(ys) + 52))

    def _layout_for(self, show_unlisted, focus):
        """Positions from the graph of ALL chapters, so nodes stay put while the slider moves."""
        key = (bool(show_unlisted), focus)
        if key != self._layout_key:
            self._install_layout(key, self._compute_layout(self.tracker, self.glossary_entries, *key,
                                                           gender_tracker=self.gender_tracker))
        return self._positions

    def _sync_view_controls(self):
        """The Relations / Gender boxes and the Hide button show the tracker's view choices."""
        if self._page is None:
            return
        view = self.view_settings()
        for box, on in ((self.show_relations_checkbox, view['show_relations']),
                        (self.show_gender_checkbox, view['show_gender'])):
            blocked = box.blockSignals(True)
            try:
                box.setChecked(bool(on))
            finally:
                box.blockSignals(blocked)
        count = len(view['hidden'])
        self.hide_button.setText(f"Hide ({count})…" if count else "Hide…")

    def rebuild(self):
        """Redraw the scene for the current controls and data."""
        if self._loading or self._page is None:
            return  # nothing to draw yet: the loading screen is up
        if not self._has_content():
            self._stack.setCurrentWidget(self.empty_label)
            self._edge_count = 0
            self._node_items = {}
            return
        self._stack.setCurrentWidget(self._page)
        chapter = self.up_to_chapter()
        shown = chapter if chapter is not None else self._chapter_high
        self.chapter_label.setText(f"Up to chapter {shown}" if shown is not None else "All chapters")
        show_former = self.show_former_checkbox.isChecked()
        show_unlisted = self.show_unlisted_checkbox.isChecked()
        focus = self.focus_id()
        view = self.view_settings()
        graph = relations_tracking.build_graph(self.tracker, self.glossary_entries, chapter,
                                               show_unlisted=show_unlisted,
                                               mode='full' if show_former else 'current',
                                               gender_tracker=self.gender_tracker,
                                               include_gender_only=self.gender_tracker is not None,
                                               hidden=view['hidden'], show_relations=view['show_relations'],
                                               show_gender=view['show_gender'])
        positions = self._layout_for(show_unlisted, focus)

        self.scene.clear()
        self._node_items = {}
        nodes = {n['id']: n for n in graph.get('nodes') or []}
        for node_id, node in nodes.items():
            if node_id not in positions:
                continue
            item = _NodeItem(node, self._on_node_clicked, self._node_font)
            written, _earlier, _source = self.descriptions_shown(node_id, str(node.get('raw_name') or node_id))
            if written is not None:
                item.setToolTip(f"<qt><b>{html.escape(item.toolTip())}</b><br>{html.escape(written['text'])}</qt>")
            x, y = positions[node_id]
            item.setPos(x, y)
            self.scene.addItem(item)
            self._node_items[node_id] = item

        edges = [e for e in graph.get('edges') or [] if e['a'] in self._node_items and e['b'] in self._node_items]
        self._edge_count = len(edges)
        label_all = len(edges) <= MAX_LABELLED_EDGES
        # A selected character keeps its own lines and labels; everything else fades back.
        selected = self._selected if self._selected in self._node_items else None
        related = {selected}
        # Labels steer clear of the name boxes and of each other.
        placed_labels = [item.sceneBoundingRect().adjusted(-3, -3, 3, 3) for item in self._node_items.values()]
        to_label = []
        self._label_boxes = []
        for edge in edges:
            a, b = self._node_items[edge['a']], self._node_items[edge['b']]
            pa, pb = a.pos(), b.pos()
            mine = selected is not None and selected in (edge['a'], edge['b'])
            if mine:
                related.update((edge['a'], edge['b']))
            color = CHANGED_COLOR if edge.get('changed') else (SELECTED_EDGE_COLOR if mine else EDGE_COLOR)
            width = 2.2 if edge.get('changed') else (2.0 if mine else 1.4)
            line = self.scene.addLine(pa.x(), pa.y(), pb.x(), pb.y(), QPen(color, width))
            line.setData(0, f"edge{_KEY_SEP}{edge['a']}{_KEY_SEP}{edge['b']}")
            line.setToolTip("\n".join(self._edge_tip_lines(edge)))
            line.setZValue(2 if mine else 1)
            if selected is not None and not mine:
                line.setOpacity(DIMMED_OPACITY)
            if mine or (selected is None and label_all):
                to_label.append((math.hypot(pb.x() - pa.x(), pb.y() - pa.y()), edge, pa, pb, color))
        # Short edges have the least room, so they choose their label spot first.
        for _length, edge, pa, pb, color in sorted(to_label, key=lambda item: item[0]):
            self._add_edge_label(edge, pa, pb, color, show_former, placed_labels)

        if selected is not None:
            for node_id, item in self._node_items.items():
                if node_id not in related:
                    item.setOpacity(0.4)
            self._node_items[selected].set_selected_look(True)
        hidden = len(view['hidden'])
        self.status_label.setText(
            f"{len(self._node_items)} characters, {len(edges)} relations"
            + (f", {hidden} hidden" if hidden else "")
            + ("" if label_all else f" (labels shown for the selected character only above {MAX_LABELLED_EDGES})")
            + " · amber = changed · click a name to show only its relations"
            + (" · right-click to edit" if self.can_edit() else "")
            + " · scroll to zoom, drag to pan")
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-200, -200, 200, 200))
        if self._refit:
            self._refit = False
            self._view.fit_all()
        if selected is not None:
            self.show_history(selected)  # the slider moved: the description written at that chapter

    @staticmethod
    def _overlap(box, others):
        total = 0.0
        for other in others:
            shared = box.intersected(other)
            if not shared.isEmpty():
                total += shared.width() * shared.height()
        return total

    def _edge_tip_lines(self, edge):
        lines = relations_tracking.edge_description_lines(edge, self._name)
        if edge.get('since') is not None:
            lines.append(f"Since chapter {edge['since']}")
        if self.can_edit():
            lines.append("Double-click or right-click to edit")
        return lines

    def _add_edge_label(self, edge, pa, pb, color, show_former, placed):
        """One pill ON its line: the line runs into both ends of the pill, so it is plain
        which relation the words belong to. It slides along the line to a free spot.
        A relation with no side known has no words: no pill."""
        text = edge_label_text(edge, show_former)
        if not text:
            return None
        label = QGraphicsSimpleTextItem(text)
        label.setFont(self._label_font)
        label.setBrush(QBrush(color.lighter(125) if edge.get('changed') else QColor("#e5e7eb")))
        rect = label.boundingRect()
        width, height = rect.width() + 12, rect.height() + 6
        box, least = None, None
        for spot in LABEL_SPOTS:
            cx = pa.x() + (pb.x() - pa.x()) * spot
            cy = pa.y() + (pb.y() - pa.y()) * spot
            candidate = QRectF(cx - width / 2, cy - height / 2, width, height)
            overlap = self._overlap(candidate, placed)
            if least is None or overlap < least:
                box, least = candidate, overlap
            if overlap == 0.0:
                break
        placed.append(box)
        self._label_boxes.append(box)
        pill = QPainterPath()
        pill.addRoundedRect(box, height / 2, height / 2)
        border = QColor(color)
        border.setAlpha(150)
        background = self.scene.addPath(pill, QPen(border, 1.0), QBrush(LABEL_BG))
        background.setZValue(5)
        label.setPos(box.x() + 6, box.y() + 3)
        label.setZValue(6)
        self.scene.addItem(label)
        key = f"edge{_KEY_SEP}{edge['a']}{_KEY_SEP}{edge['b']}"
        tip = "\n".join(self._edge_tip_lines(edge))
        for item in (background, label):
            item.setData(0, key)
            item.setToolTip(tip)
        return label

    def _name(self, node_id):
        item = self._node_items.get(node_id)
        if item is not None:
            return item.text_item.text()
        info = self._node_info.get(node_id)
        return str(info.get('label') or info.get('raw_name') or node_id) if info else str(node_id)

    def _node(self, node_id):
        """``{"id", "raw_name", "label", "translated"}`` of a name (drawn or not)."""
        info = self._node_info.get(node_id) or {}
        entries = self.tracker.get('entries') if isinstance(self.tracker, dict) else None
        item = entries.get(node_id) if isinstance(entries, dict) else None
        raw = str(info.get('raw_name') or (item.get('raw_name') if isinstance(item, dict) else '') or node_id)
        label = str(info.get('label') or (item.get('translated_name') if isinstance(item, dict) else '') or raw)
        return {'id': node_id, 'raw_name': raw, 'label': label, 'translated': label if label != raw else ''}

    # ── selection and the history panel ─────────────────────────────────
    def _on_node_clicked(self, node_id):
        if node_id == self._selected:  # a second click on the same name lets go of it
            self.clear_selection()
            return
        self._selected = node_id
        self.rebuild()
        self.show_history(node_id)

    def select(self, node_id):
        """Show ``node_id``: the panel switches to it, the diagram shows its relations, and the
        name is brought into sight and outlined for a moment (a name clicked in the panel)."""
        self._selected = node_id
        self.rebuild()
        self.show_history(node_id)
        self.jump_to(node_id)

    def _on_history_link(self, url):
        """A link in the side panel: ``node:<i>`` selects a listed name, ``act:<i>`` runs a row's
        edit or delete (after the click: it may open a dialog and rewrite the panel)."""
        scheme = url.scheme()
        try:
            index = int(url.path())
        except ValueError:
            return
        if scheme == 'node':
            if 0 <= index < len(self._history_links):
                self.select(self._history_links[index])
        elif scheme == 'act':
            if 0 <= index < len(self._history_actions):
                action = self._history_actions[index][1]
                QTimer.singleShot(0, action)

    def history_action(self, text):
        """The panel link whose description starts with ``text`` (``(description, callable)``), else None."""
        for entry in self._history_actions:
            if entry[0].startswith(text):
                return entry
        return None

    def jump_to(self, node_id):
        """Show where ``node_id`` is in the diagram: bring it into view when it is out of
        sight and outline it for a moment."""
        item = self._node_items.get(node_id)
        if item is None:
            return False
        self._view.bring_into_view(item)
        item.set_flash(True)
        QTimer.singleShot(FLASH_MS, lambda: self._end_flash(node_id, item))
        return True

    def _end_flash(self, node_id, item):
        # Only while that very item is still drawn: a rebuild in between deleted it.
        if self._node_items.get(node_id) is item:
            try:
                item.set_flash(False)
            except RuntimeError:  # the window was closed meanwhile
                pass

    def clear_selection(self):
        """Back to the whole picture (a click on empty space, or on the selected name again)."""
        if self._selected is None:
            return
        self._selected = None
        self.rebuild()
        self.history_title.setText(HISTORY_HINT)
        self.history_panel.clear()

    def descriptions_shown(self, node_id, raw):
        """The description written at the slider's chapter, the earlier ones, and its source
        (``relations_tracking.descriptions_shown``, shared with the mobile diagram)."""
        return relations_tracking.descriptions_shown(self.tracker, raw, self.up_to_chapter(), self.glossary_entries)

    def _link(self, text, description, action):
        """A small panel link running ``action`` (kept for ``_on_history_link`` / ``history_action``)."""
        self._history_actions.append((description, action))
        return f"<a href='act:{len(self._history_actions) - 1}' style='{LINK_STYLE}'>{html.escape(text)}</a>"

    def _row_links(self, edit=None, delete=None):
        parts = []
        if edit is not None:
            parts.append(self._link("edit", *edit))
        if delete is not None:
            parts.append(self._link("delete", *delete))
        return ("&nbsp;&nbsp;" + " · ".join(parts)) if parts else ""

    def _description_html(self, node_id, raw):
        editable = self.can_edit()
        written, earlier, source = self.descriptions_shown(node_id, raw)
        if written is None:
            if source != 'chapter':
                return []
            shown = self.up_to_chapter()
            return [f"<p style='color:#9ca3af;margin:0 0 10px 0'>No description written up to chapter {shown}.</p>"]
        heading = (f"Description written at chapter {written.get('chapter_num')}" if source == 'chapter'
                   else "Description (from the glossary; not recorded per chapter for this book yet)")
        links = ""
        if editable and source == 'chapter':
            chapter, file = written.get('chapter_num'), written.get('chapter_file', '')
            links = self._row_links(
                (f"edit description ch {chapter}", lambda: self.edit_description_for(node_id, chapter, file)),
                (f"delete description ch {chapter}", lambda: self.delete_description_for(node_id, chapter, file)))
        parts = [f"<p style='color:#9ca3af;margin:0 0 2px 0'>{html.escape(heading)}{links}</p>",
                 f"<p style='margin:0 0 10px 0'>{html.escape(written['text'])}</p>"]
        if earlier:
            parts.append("<p style='color:#9ca3af;margin:0 0 2px 0'>Earlier descriptions</p><ul style='margin:0 0 10px 0'>")
            for old in earlier:
                links = ""
                if editable:
                    chapter, file = old.get('chapter_num'), old.get('chapter_file', '')
                    links = self._row_links(
                        (f"edit description ch {chapter}", lambda c=chapter, f=file: self.edit_description_for(node_id, c, f)),
                        (f"delete description ch {chapter}", lambda c=chapter, f=file: self.delete_description_for(node_id, c, f)))
                parts.append(f"<li style='color:#9ca3af'>Ch. {html.escape(str(old.get('chapter_num')))}: "
                             f"{html.escape(old['text'])}{links}</li>")
            parts.append("</ul>")
        return parts

    def _gender_html(self, node_id, raw):
        entry = self._glossary_index.get(relations_tracking.tracker_key(raw))
        history = relations_tracking.gender_history(self.gender_tracker, raw, self.up_to_chapter(),
                                                    glossary_gender=entry.get('gender') if entry else '')
        text = relations_tracking.gender_history_text(history) or "unknown"
        links = self._row_links((f"set gender {raw}", lambda: self.set_gender_for(node_id))) if self.can_edit() else ""
        return [f"<p style='margin:0 0 8px 0'><span style='color:#9ca3af'>Gender:</span> "
                f"{html.escape(text)}{links}</p>"]

    def show_history(self, node_id):
        info = self._node(node_id)
        raw, name = info['raw_name'], self._name(node_id)
        self.history_title.setText(f"{name} ({raw})" if raw != name else name)
        view = self.view_settings()
        editable = self.can_edit()
        # Like the picture beside it, the history stops at the slider's chapter.
        chapter = self.up_to_chapter()
        history = relations_tracking.character_history(self.tracker, raw, chapter)
        self._history_links = []
        self._history_actions = []
        actions = []
        if editable:
            actions.append(self._link("add relation", f"add relation {raw}", lambda: self.add_relation_for(node_id)))
            actions.append(self._link("add description", f"add description {raw}", lambda: self.edit_description_for(node_id)))
        actions.append(self._link("hide", f"hide {raw}", lambda: self.hide_name(node_id)))
        if editable:
            actions.append(self._link("remove from tracker", f"remove {raw}", lambda: self.remove_character_for(node_id)))
        parts = [f"<p style='margin:0 0 8px 0'>{' · '.join(actions)}</p>"]
        if view['show_gender']:
            parts += self._gender_html(node_id, raw)
        parts += self._description_html(node_id, raw)
        if not view['show_relations']:
            parts.append("<p style='color:#9ca3af'>Relations are switched off (the Relations box).</p>")
            self.history_panel.setHtml("".join(parts))
            return history
        if not history:
            upto = f" up to chapter {chapter}" if chapter is not None else ""
            parts.append(f"<p style='color:#9ca3af'>No relations recorded from this character's side{upto}.</p>")
            self.history_panel.setHtml("".join(parts))
            return history
        for rel in history:
            tkey = rel.get('target_key')
            target_name = str(rel.get('target_translated') or rel.get('target_raw') or '')
            target = html.escape(target_name)
            if tkey in self._node_info or tkey in self._node_items:  # a known name: clicking shows it
                self._history_links.append(tkey)
                target = (f"<a href='node:{len(self._history_links) - 1}' "
                          f"style='color:#e5e7eb'>{target}</a>")
            links = ""
            if editable:
                links = self._row_links(
                    (f"edit relation {target_name}", lambda t=tkey: self.edit_relation_between(node_id, t)),
                    (f"delete relation {target_name}", lambda t=tkey: self.delete_relation_between(node_id, t)))
            parts.append(f"<p style='margin:8px 0 2px 0'><b>{target}</b>: "
                         f"<span style='color:#f59e0b'>{html.escape(str(rel.get('current') or ''))}</span>{links}</p>"
                         "<ul style='margin:0'>")
            for run in rel.get('timeline') or []:
                start, end = run.get('from_chapter'), run.get('to_chapter')
                span = f"Ch. {start}" if start == end else f"Ch. {start}–{end}"
                label = str(run.get('label') or '')
                links = ""
                if editable:
                    links = self._row_links(
                        (f"edit {label} {target_name} {span}",
                         lambda t=tkey, lab=label: self.edit_history_label(node_id, t, lab)),
                        (f"delete {label} {target_name} {span}",
                         lambda t=tkey, lab=label, s=start, e=end: self.delete_history_row(node_id, t, lab, s, e)))
                parts.append(f"<li>{html.escape(label)} <span style='color:#9ca3af'>({span})</span>{links}</li>")
            for blip in rel.get('blips') or []:
                label = str(blip.get('label') or '')
                links = ""
                if editable:
                    number, file = blip.get('chapter_num'), blip.get('chapter_file', '')
                    links = self._row_links(None, (
                        f"delete {label} {target_name} Ch. {number}",
                        lambda t=tkey, lab=label, n=number, f=file: self.delete_history_row(node_id, t, lab, n, n, f)))
                parts.append(f"<li style='color:#9ca3af'>{'ignored reversed' if blip.get('flip') else 'ignored one-off'}: "
                             f"{html.escape(label)} (Ch. {blip.get('chapter_num')}){links}</li>")
            parts.append("</ul>")
        self.history_panel.setHtml("".join(parts))
        return history

    # ── menus ───────────────────────────────────────────────────────────
    def _menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(CONTEXT_MENU_STYLE)
        return menu

    @staticmethod
    def _add_action(menu, text, handler, enabled=True):
        action = menu.addAction(text)
        action.setEnabled(bool(enabled))
        action.triggered.connect(lambda _checked=False: handler())
        return action

    def node_menu(self, node_id):
        """A name's right-click menu."""
        menu = self._menu()
        editable = self.can_edit()
        self._add_action(menu, "Add relation…", lambda: self.add_relation_for(node_id), editable)
        self._add_action(menu, "Edit description…", lambda: self.edit_description_for(node_id), editable)
        self._add_action(menu, "Set gender…", lambda: self.set_gender_for(node_id), editable)
        menu.addSeparator()
        self._add_action(menu, "Hide from view", lambda: self.hide_name(node_id))
        menu.addSeparator()
        self._add_action(menu, "Remove from tracker…", lambda: self.remove_character_for(node_id), editable)
        return menu

    def edge_menu(self, a, b):
        """A relation's right-click menu."""
        menu = self._menu()
        editable = self.can_edit()
        self._add_action(menu, "Edit relation…", lambda: self.edit_relation_between(a, b), editable)
        self._add_action(menu, "Delete relation", lambda: self.delete_relation_between(a, b), editable)
        return menu

    def _on_item_menu(self, key, global_pos):
        if key[0] == 'node' and len(key) == 2:
            menu = self.node_menu(key[1])
        elif key[0] == 'edge' and len(key) == 3:
            menu = self.edge_menu(key[1], key[2])
        else:
            return
        self._dialogs_open += 1
        try:
            menu.exec(global_pos)
        finally:
            self._dialogs_open -= 1
            menu.deleteLater()

    def _on_item_double(self, key):
        if key[0] == 'edge' and len(key) == 3 and self.can_edit():
            self.edit_relation_between(key[1], key[2])

    # ── hiding ──────────────────────────────────────────────────────────
    def open_hide_popup(self):
        """The Hide control: a searchable checklist of the names (checked = hidden)."""
        popup = self.hide_popup()
        popup.popup_below(self.hide_button)
        return popup

    def hide_popup(self):
        if self._hide_popup is None:
            self._hide_popup = _HidePopup(self)
            self._hide_popup.hidden_changed.connect(self.set_hidden)
        hidden = self.view_settings()['hidden']
        names = [{'id': n['id'], 'label': n.get('label'), 'raw_name': n.get('raw_name')}
                 for n in self._node_info.values()]
        names += [{'id': h, 'label': h, 'raw_name': ''} for h in hidden if h not in self._node_info]
        self._hide_popup.set_names(names, hidden)
        return self._hide_popup

    def set_hidden(self, node_ids):
        if self._selected in set(node_ids or ()):
            self.clear_selection()
        return self.set_view(hidden=list(node_ids or ()))

    def hide_name(self, node_id):
        hidden = self.view_settings()['hidden']
        if node_id in hidden:
            return None
        return self.set_hidden(hidden + [node_id])

    def set_view(self, hidden=None, show_relations=None, show_gender=None):
        """Change the view choices; kept in the tracker file's ``view`` (relations data untouched)."""
        changes = {k: v for k, v in (('hidden', hidden), ('show_relations', show_relations),
                                     ('show_gender', show_gender)) if v is not None}
        if not changes:
            return None
        what = ", ".join(self._view_change_text(k, v) for k, v in changes.items())
        result = None
        if self.can_edit():
            try:
                result = relations_tracking.save_view(self.tracker_path, **changes)
            except (ValueError, OSError) as exc:
                self.notify(f"Could not save the view choice: {exc}")
        if result is not None:
            self._after_write('relations', result, what)
        else:
            relations_tracking.set_view(self.tracker, **changes)  # this window only
            self._sync_view_controls()
            self.rebuild()
        if self._selected is not None and self._selected not in self._node_items:
            self.clear_selection()
        return result

    def _view_change_text(self, key, value):
        if key == 'hidden':
            return f"hide {len(value)} name(s)" if value else "show all names"
        return f"{'show' if value else 'hide'} {'relations' if key == 'show_relations' else 'gender'}"

    # ── writing ─────────────────────────────────────────────────────────
    def _exec(self, dialog):
        """Run an edit dialog (the auto-update waits while one is open). True when saved."""
        self._dialogs_open += 1
        try:
            return dialog.exec() == QDialog.Accepted
        finally:
            self._dialogs_open -= 1

    def _confirm(self, title, text):
        import review_dialog
        self._dialogs_open += 1
        try:
            return bool(review_dialog.confirm_prompt_profile_action(self, title, text))
        finally:
            self._dialogs_open -= 1

    def notify(self, text):
        """A short notice beside the filters (it goes away by itself)."""
        if self._page is None:
            return
        self._notice_token += 1
        token = self._notice_token
        self.notice_label.setText(text)
        QTimer.singleShot(NOTICE_MS, lambda: self._clear_notice(token))

    def _clear_notice(self, token):
        if token == self._notice_token:
            try:
                self.notice_label.setText("")
            except RuntimeError:
                pass

    def _after_write(self, kind, result, what):
        """This window wrote a tracker file: remember its stamp (so the check does not reload
        our own write), take the written data, offer Undo and redraw."""
        path = self.tracker_path if kind == 'relations' else self.gender_path
        self._stamps[path] = result.get('stamp')
        if not result.get('changed'):
            return False
        self._data_gen += 1
        if kind == 'relations':
            self.tracker = result['tracker']
        else:
            self.gender_tracker = result['tracker']
        self._undo.append({'kind': kind, 'edit': result, 'text': what})
        del self._undo[:-UNDO_LIMIT]
        self._update_undo_button()
        self._after_data_change()
        return True

    def _edit(self, mutate, what):
        """One locked load -> ``mutate`` -> write of the relations tracker."""
        if not self.can_edit():
            return None
        try:
            result = relations_tracking.edit_tracker(self.tracker_path, mutate)
        except (ValueError, OSError) as exc:
            self.notify(f"Could not save: {exc}")
            return None
        if not self._after_write('relations', result, what):
            self.notify("Nothing changed.")
        return result

    def _update_undo_button(self):
        if self._page is None:
            return
        if self._undo:
            self.undo_button.setEnabled(True)
            self.undo_button.setToolTip(f"Undo: {self._undo[-1]['text']} (Ctrl+Z)")
        else:
            self.undo_button.setEnabled(False)
            self.undo_button.setToolTip("Nothing to undo")

    def undo(self):
        """Take back the last edit made in this window. Refused (with a notice) when someone
        else changed the same characters since."""
        if not self._undo:
            return False
        entry = self._undo.pop()
        try:
            if entry['kind'] == 'relations':
                result = relations_tracking.undo_edit(self.tracker_path, entry['edit'])
            else:
                result = relations_tracking.undo_gender_edit(self.glossary_path, entry['edit'])
        except (ValueError, OSError) as exc:
            result = {'undone': False, 'reason': str(exc)}
        self._update_undo_button()
        if not result.get('undone'):
            reason = result.get('reason')
            why = {'changed': "the tracker was changed since (an extraction?)",
                   'unreadable': "the tracker file cannot be read",
                   'nothing': "nothing to take back"}.get(reason, reason)
            self.notify(f"Cannot undo “{entry['text']}”: {why}.")
            return False
        path = self.tracker_path if entry['kind'] == 'relations' else self.gender_path
        self._stamps[path] = result.get('stamp')
        self._data_gen += 1
        if entry['kind'] == 'relations':
            self.tracker = result['tracker']
        else:
            self.gender_tracker = result['tracker']
        self._after_data_change()
        self.notify(f"Undone: {entry['text']}")
        return True

    # ── edits ───────────────────────────────────────────────────────────
    def _name_choices(self, exclude=()):
        names = relations_tracking.known_names(self.tracker, self.glossary_entries, self.gender_tracker)
        return [n for n in names if n['id'] not in set(exclude)]

    @staticmethod
    def _choice_text(name):
        label, raw = str(name.get('label') or ''), str(name.get('raw_name') or '')
        return f"{label} ({raw})" if raw and raw != label else (label or raw)

    def _picked_name(self, combo, names):
        """``(raw_name, translated)`` of the picker's choice; a name typed that is not listed is new."""
        text = combo.currentText().strip()
        if not text:
            return '', ''
        index = combo.currentIndex()
        if 0 <= index < len(names) and combo.itemText(index) == text:
            name = names[index]
        else:
            wanted = text.casefold()
            name = next((n for n in names if wanted in (self._choice_text(n).casefold(),
                                                         str(n.get('label') or '').casefold(),
                                                         str(n.get('raw_name') or '').casefold())), None)
        if name is None:
            return text, ''
        raw, label = str(name['raw_name']), str(name.get('label') or '')
        return raw, (label if label != raw else '')

    def add_relation_for(self, node_id, other=None):
        """Add a relation between ``node_id`` and a known name (or a new one typed in)."""
        me = self._node(node_id)
        names = self._name_choices(exclude=(node_id,))
        dialog = _EditDialog(self, "Add relation", f"New relation for {me['label']}")
        picker = dialog.combo("Other character:", [(self._choice_text(n), n['raw_name']) for n in names],
                              editable=True, placeholder="Pick a name, or type a new one")
        if other:
            index = picker.findData(self._node(other)['raw_name'])
            picker.setCurrentIndex(index)
        label = dialog.line(f"What they are to {me['label']}:", placeholder="e.g. Older Brother")
        reverse = dialog.line(f"What {me['label']} is to them:", placeholder="e.g. Younger Sister")
        chapter = dialog.spin("From chapter:", self.edit_chapter() or 1)
        dialog.note("Fill one or both labels.")
        self._last_dialog = dialog
        if not self._exec(dialog):
            return None
        other_raw, other_translated = self._picked_name(picker, names)
        forward, backward = label.text().strip(), reverse.text().strip()
        if not other_raw or not (forward or backward):
            self.notify("Nothing added: pick a name and write at least one label.")
            return None
        number = chapter.value()
        result = self._edit(lambda t: relations_tracking.add_relation(
            t, me['raw_name'], other_raw, forward or None, backward or None, chapter=number,
            translated_name=me['translated'], target_translated=other_translated),
            f"add relation {me['label']} – {other_translated or other_raw}")
        if (result and result.get('changed') and self.glossary_entries is not None
                and not self.show_unlisted_checkbox.isChecked()
                and relations_tracking.tracker_key(other_raw) not in self._glossary_index):
            self.notify(f"{other_translated or other_raw} is not in the glossary: "
                        "turn on “Show unlisted names” to see it.")
        return result

    def edit_relation_between(self, a, b):
        """Edit both directions of the relation between ``a`` and ``b``; a cleared box deletes that side."""
        na, nb = self._node(a), self._node(b)
        old_ab, old_ba = relations_tracking.relation_labels(self.tracker, na['raw_name'], nb['raw_name'],
                                                            self.up_to_chapter())
        dialog = _EditDialog(self, "Edit relation", f"{na['label']} and {nb['label']}")
        line_ab = dialog.line(f"What {nb['label']} is to {na['label']}:", old_ab)
        line_ba = dialog.line(f"What {na['label']} is to {nb['label']}:", old_ba)
        chapter = dialog.spin("Chapter (for a new side):", self.edit_chapter() or 1)
        dialog.note("A changed label also replaces the AI's wording in later extractions. "
                    "Clear a box to delete that side.")
        self._last_dialog = dialog
        if not self._exec(dialog):
            return None
        new_ab, new_ba = line_ab.text().strip(), line_ba.text().strip()
        label_ab = None if new_ab == old_ab else new_ab
        label_ba = None if new_ba == old_ba else new_ba
        if label_ab is None and label_ba is None:
            return None
        cleared = [text for text, old, new in ((f"what {nb['label']} is to {na['label']}", old_ab, label_ab),
                                               (f"what {na['label']} is to {nb['label']}", old_ba, label_ba))
                   if old and new == ""]
        if cleared and not self._confirm("Delete relation side",
                                         f"Delete {' and '.join(cleared)}?\n\n"
                                         "Later extractions will not add that label back."):
            return None
        number = chapter.value()
        return self._edit(lambda t: relations_tracking.edit_relation(
            t, na['raw_name'], nb['raw_name'], label_ab, label_ba, old_ab=old_ab or None, old_ba=old_ba or None,
            chapter=number, translated_a=na['translated'], translated_b=nb['translated']),
            f"edit relation {na['label']} – {nb['label']}")

    def delete_relation_between(self, a, b):
        na, nb = self._node(a), self._node(b)
        if not self._confirm("Delete relation",
                             f"Delete the relation between {na['label']} and {nb['label']}?\n\n"
                             "Both directions and their history are deleted, and later extractions "
                             "will not add it back."):
            return None
        return self._edit(lambda t: relations_tracking.delete_relation(t, na['raw_name'], nb['raw_name']),
                          f"delete relation {na['label']} – {nb['label']}")

    def edit_history_label(self, owner, target, label):
        """Rename one label of "what ``target`` is to ``owner``" (every record of it)."""
        me, other = self._node(owner), self._node(target)
        dialog = _EditDialog(self, "Edit relation label", f"What {other['label']} is to {me['label']}")
        line = dialog.line("Label:", label)
        dialog.note("Every record of this label becomes the new one, also in later extractions. "
                    "Clear it to delete the label.")
        self._last_dialog = dialog
        if not self._exec(dialog):
            return None
        new = line.text().strip()
        if new == label:
            return None
        if not new and not self._confirm("Delete label",
                                         f"Delete “{label}” (what {other['label']} is to {me['label']})?\n\n"
                                         "Later extractions will not add it back."):
            return None
        return self._edit(lambda t: relations_tracking.set_relation_label(
            t, me['raw_name'], other['raw_name'], new, old_label=label,
            translated_name=me['translated'], target_translated=other['translated']),
            f"rename {label} → {new or '(deleted)'}")

    def delete_history_row(self, owner, target, label, from_chapter, to_chapter, chapter_file=''):
        """Delete one history row (a timeline run, or a one-off at one chapter)."""
        me, other = self._node(owner), self._node(target)
        span = (f"chapter {_chapter_text(from_chapter)}" if from_chapter == to_chapter
                else f"chapters {_chapter_text(from_chapter)}–{_chapter_text(to_chapter)}")
        if not self._confirm("Delete history entry",
                             f"Delete “{label}” ({span}) from what {other['label']} is to {me['label']}?\n\n"
                             "Re-extracting those chapters will not add it back."):
            return None
        return self._edit(lambda t: relations_tracking.delete_history_entry(
            t, me['raw_name'], other['raw_name'], label, from_chapter, to_chapter, chapter_file),
            f"delete {label} ({span})")

    def edit_description_for(self, node_id, chapter=None, chapter_file=''):
        """Write (or rewrite) a character's description at a chapter (default: the slider's)."""
        me = self._node(node_id)
        fixed = chapter is not None
        number = chapter if fixed else self.edit_chapter()
        current = ''
        for description in relations_tracking.description_history(self.tracker, me['raw_name']):
            if number is not None and _chapter_text(description.get('chapter_num')) == _chapter_text(number):
                current = description.get('text') or ''
        dialog = _EditDialog(self, "Edit description", f"Description of {me['label']}")
        spin = dialog.spin("Chapter:", int(float(number)) if number is not None else 1)
        spin.setEnabled(not fixed)
        text = dialog.text("Description:", current)
        dialog.note("A description written here is kept when that chapter is extracted again. "
                    "Clear it to delete it.")
        self._last_dialog = dialog
        if not self._exec(dialog):
            return None
        new = " ".join(text.toPlainText().split())
        target_chapter = number if fixed else spin.value()
        if not new:
            if not current:
                return None
            return self.delete_description_for(node_id, target_chapter, chapter_file)
        return self._edit(lambda t: relations_tracking.set_description(
            t, me['raw_name'], new, target_chapter, chapter_file=chapter_file, translated_name=me['translated']),
            f"description of {me['label']} (Ch. {_chapter_text(target_chapter)})")

    def delete_description_for(self, node_id, chapter, chapter_file=''):
        me = self._node(node_id)
        if not self._confirm("Delete description",
                             f"Delete the description of {me['label']} written at chapter {_chapter_text(chapter)}?\n\n"
                             "Re-extracting that chapter will not add it back."):
            return None
        return self._edit(lambda t: relations_tracking.delete_description(t, me['raw_name'], chapter,
                                                                           chapter_file=chapter_file),
                          f"delete description of {me['label']} (Ch. {_chapter_text(chapter)})")

    def set_gender_for(self, node_id):
        """Set a name's gender for one chapter or for all chapters (the book's gender tracker)."""
        if not self.can_edit():
            return None
        me = self._node(node_id)
        chapter = self.edit_chapter()
        entry = self._glossary_index.get(relations_tracking.tracker_key(me['raw_name']))
        history = relations_tracking.gender_history(self.gender_tracker, me['raw_name'], self.up_to_chapter(),
                                                    glossary_gender=entry.get('gender') if entry else '')
        dialog = _EditDialog(self, "Set gender", f"Gender of {me['label']}")
        gender = dialog.combo("Gender:", [("Male", "male"), ("Female", "female"), ("Unknown / automatic", "auto")],
                              current=history.get('current') or 'auto')
        scopes = ([(f"Chapter {_chapter_text(chapter)} only", 'chapter')] if chapter is not None else []) + \
            [("All chapters", 'all')]
        scope = dialog.combo("For:", scopes, current='all' if history.get('source') == 'decision' or chapter is None
                             else 'chapter')
        known = relations_tracking.gender_history_text(history)
        dialog.note(f"Now: {known}." if known else "No gender known yet.")
        self._last_dialog = dialog
        if not self._exec(dialog):
            return None
        value, where = gender.currentData(), scope.currentData()
        target = chapter if where == 'chapter' else None
        try:
            result = relations_tracking.set_gender(self.glossary_path, me['raw_name'], value, target,
                                                   translated_name=me['translated'])
        except (ValueError, OSError) as exc:
            self.notify(f"Could not save the gender: {exc}")
            return None
        span = f"Ch. {_chapter_text(target)}" if target is not None else "all chapters"
        if not self._after_write('gender', result, f"gender of {me['label']} ({span})"):
            self.notify("Nothing changed.")
        return result

    def remove_character_for(self, node_id):
        """Remove a (junk) name from the tracker: its entry and every relation pointing at it."""
        me = self._node(node_id)
        if not self._confirm("Remove from tracker",
                             f"Remove {me['label']} from the relations tracker?\n\n"
                             "Its entry and every relation that points at it are deleted, and later "
                             "extractions will not add the name again."):
            return None
        if self._selected == node_id:
            self.clear_selection()
        return self._edit(lambda t: relations_tracking.remove_character(t, me['raw_name']),
                          f"remove {me['label']}")
