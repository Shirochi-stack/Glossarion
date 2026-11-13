"""Relations diagram: who is what to whom, as known up to a chapter.

View-only window over a glossary's ``*_relations_tracker.json``. The graph and
its layout come from ``relations_tracking.build_graph`` / ``layout_graph``;
this module only draws them (PySide6 only, no manga-tier imports).
"""

import html
import math
import os
import threading

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QCompleter, QDialog, QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QHBoxLayout, QLabel, QSlider, QSplitter,
                               QStackedWidget, QTextBrowser, QVBoxLayout, QWidget)

import relations_tracking
from qt_widget_helpers import HalgakosLoadingWidget

ALL_CHARACTERS = "All characters"
LOADING_TEXT = "Loading relations…"
HISTORY_HINT = "Click a character to see their relation history."
EMPTY_TEXT = ("No relations tracked for this glossary yet.\n\n"
              "Turn on Enable Relations Tracker in Glossary Manager → General Settings, "
              "then run Balanced or Full glossary extraction.")

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


def _gender_key(value):
    text = str(value or '').strip().casefold()
    if text in ('female', 'f', 'woman', 'girl'):
        return 'female'
    if text in ('male', 'm', 'man', 'boy'):
        return 'male'
    return ''


#: ``"Father / Daughter"`` edge labels (shared with the mobile diagram).
edge_label_text = relations_tracking.edge_label_text


class _GraphView(QGraphicsView):
    """Wheel zoom around the cursor, drag to pan.

    The whole diagram is fitted to the window until the user zooms or pans; from then on
    the view is theirs. A click on empty space calls ``on_background_click``.
    """

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.on_background_click = None
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
        tint = GENDER_TINTS.get(_gender_key(node.get('gender')), NEUTRAL_TINT)
        self.setBrush(QBrush(_over_background(tint, 0.6 if node.get('listed', True) else 0.28)))
        self._border = tint.lighter(130)
        self.setPen(QPen(self._border, 1.4, Qt.SolidLine if node.get('listed', True) else Qt.DashLine))
        tip = str(node.get('raw_name') or '')
        if not node.get('listed', True):
            tip += " (not in the glossary)"
        self.setToolTip(tip)
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


class RelationsDiagramDialog(QDialog):
    """Graph of the tracked relations with a chapter slider, focus search and history panel.

    ``load_async=True`` (the Glossary Manager's buttons): the window is built empty with a
    loading screen and returns at once; the tracker, the glossary and the layout are worked
    out on a thread and the diagram appears when they are ready. Otherwise everything is
    ready when the constructor returns.
    """

    #: (tracker, glossary entries, layout key, positions) from the loading thread
    _loaded = Signal(object)

    def __init__(self, parent=None, glossary_path=None, tracker=None, glossary_entries=None, load_async=False):
        super().__init__(parent)
        self.setWindowTitle("Relations Diagram")
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
        # Reading the files and laying the graph out is the slow part (over a second on a
        # 15-name book before the layout was sped up): with load_async it happens after the
        # window is on screen.
        self._loading = bool(load_async) and tracker is None and bool(self.glossary_path)
        if self._loading:
            tracker, glossary_entries = relations_tracking.empty_tracker(), None
        else:
            tracker, glossary_entries = self._read_data(self.glossary_path, tracker, glossary_entries)
        self.tracker = tracker
        self.glossary_entries = glossary_entries

        self._node_items = {}
        self._edge_count = 0
        self._positions = {}
        self._layout_key = None
        self._refit = False
        self._label_boxes = []
        self._history_links = []
        self._selected = None
        self._node_font = QFont("Segoe UI", 10)
        self._node_font.setBold(True)
        self._label_font = QFont("Segoe UI", 8)

        self._build_shell()
        if self._loading:
            self._stack.setCurrentWidget(self.loading_widget)
            self._loaded.connect(self._on_loaded)
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
    def _layout_size(node_count):
        width = max(900.0, 230.0 * math.sqrt(max(1, node_count)) * 1.6)
        return width, width * 0.68

    @classmethod
    def _compute_layout(cls, tracker, glossary_entries, show_unlisted, focus):
        """Positions from the graph of ALL chapters, so nodes stay put while the slider moves."""
        full = relations_tracking.build_graph(tracker, glossary_entries, None, show_unlisted=show_unlisted)
        width, height = cls._layout_size(len(full.get('nodes') or []))
        return relations_tracking.layout_graph(full, width, height, focus=focus)

    def _load_in_background(self):
        """The loading thread: plain data only, no Qt objects. The result goes back by signal."""
        try:
            tracker, entries = self._read_data(self.glossary_path)
            key = (False, None)  # the layout the diagram opens with
            payload = (tracker, entries, key, self._compute_layout(tracker, entries, *key))
        except Exception:
            payload = (relations_tracking.empty_tracker(), None, None, {})
        try:
            self._loaded.emit(payload)
        except RuntimeError:  # the window was closed before the data was ready
            pass

    def _on_loaded(self, payload):
        tracker, entries, key, positions = payload
        self.tracker, self.glossary_entries = tracker, entries
        self._loading = False
        self._build_diagram_page()
        if key is not None:
            self._install_layout(key, positions)
        self.rebuild()

    def is_loading(self):
        return self._loading

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
        low, high = relations_tracking.chapter_bounds(self.tracker)
        self._chapter_low = int(math.floor(low)) if isinstance(low, (int, float)) else None
        self._chapter_high = int(math.ceil(high)) if isinstance(high, (int, float)) else None
        self.chapter_label = QLabel("")
        self.chapter_label.setMinimumWidth(130)
        controls.addWidget(self.chapter_label)
        self.chapter_slider = QSlider(Qt.Horizontal)
        self.chapter_slider.setMinimumWidth(220)
        if self._chapter_low is not None and self._chapter_high is not None:
            self.chapter_slider.setRange(self._chapter_low, self._chapter_high)
            self.chapter_slider.setValue(self._chapter_high)
        else:
            self.chapter_slider.setEnabled(False)
        self.chapter_slider.setToolTip("Show only what was known up to this chapter")
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

        self.show_former_checkbox = _styled_checkbox("Show former relations")
        self.show_former_checkbox.setToolTip("Add earlier labels in brackets, e.g. Enemy [Best Friend]")
        self.show_former_checkbox.toggled.connect(self._on_controls_changed)
        controls.addWidget(self.show_former_checkbox)
        self.show_unlisted_checkbox = _styled_checkbox("Show unlisted names")
        self.show_unlisted_checkbox.setToolTip("Also show names that are not (or no longer) in the glossary")
        self.show_unlisted_checkbox.toggled.connect(self._on_controls_changed)
        controls.addWidget(self.show_unlisted_checkbox)

        splitter = QSplitter(Qt.Horizontal)
        page_layout.addWidget(splitter, 1)
        self.scene = QGraphicsScene(self)
        self._view = _GraphView(self.scene)
        self._view.on_background_click = self.clear_selection
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
        # A name in the list is a link to that name in the diagram.
        self.history_panel.anchorClicked.connect(self._on_history_link)
        self.history_panel.setToolTip("Click a name to find it in the diagram")
        side_layout.addWidget(self.history_panel, 1)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([880, 300])

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #9ca3af;")
        page_layout.addWidget(self.status_label)

        self._fill_focus_combo()

    def _fill_focus_combo(self):
        graph = relations_tracking.build_graph(self.tracker, self.glossary_entries, None, show_unlisted=True)
        nodes = sorted(graph.get('nodes') or [], key=lambda n: str(n.get('label') or '').casefold())
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
            self.focus_combo.setCurrentIndex(0)
        finally:
            self.focus_combo.blockSignals(blocked)

    # ── state ───────────────────────────────────────────────────────────
    def up_to_chapter(self):
        if not self.chapter_slider.isEnabled():
            return None
        value = self.chapter_slider.value()
        return None if value >= (self._chapter_high or value) else value

    def focus_id(self):
        return self.focus_combo.currentData()

    def edge_count(self):
        return self._edge_count

    def node_count(self):
        return len(self._node_items)

    def _on_controls_changed(self, *_args):
        self.rebuild()

    def _install_layout(self, key, positions):
        self._positions = positions
        self._layout_key = key
        self._refit = True
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
            self._install_layout(key, self._compute_layout(self.tracker, self.glossary_entries, *key))
        return self._positions

    def rebuild(self):
        """Redraw the scene for the current controls (only called when a control changes)."""
        if self._loading:
            return  # nothing to draw yet: the loading screen is up
        if not self._has_relations():
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
        graph = relations_tracking.build_graph(self.tracker, self.glossary_entries, chapter,
                                               show_unlisted=show_unlisted,
                                               mode='full' if show_former else 'current')
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
        self.status_label.setText(
            f"{len(self._node_items)} characters, {len(edges)} relations"
            + ("" if label_all else f" (labels shown for the selected character only above {MAX_LABELLED_EDGES})")
            + " · amber = changed · click a name to show only its relations · scroll to zoom, drag to pan")
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

    def _add_edge_label(self, edge, pa, pb, color, show_former, placed):
        """One pill ON its line: the line runs into both ends of the pill, so it is plain
        which relation the words belong to. It slides along the line to a free spot."""
        text = edge_label_text(edge, show_former)
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
        tip = f"{self._name(edge['b'])} is {edge.get('label_ab') or '?'} to {self._name(edge['a'])}\n" \
              f"{self._name(edge['a'])} is {edge.get('label_ba') or '?'} to {self._name(edge['b'])}"
        if edge.get('since') is not None:
            tip += f"\nSince chapter {edge['since']}"
        background.setToolTip(tip)
        label.setToolTip(tip)

    def _name(self, node_id):
        item = self._node_items.get(node_id)
        return item.text_item.text() if item is not None else str(node_id)

    # ── history panel ───────────────────────────────────────────────────
    def _on_node_clicked(self, node_id):
        if node_id == self._selected:  # a second click on the same name lets go of it
            self.clear_selection()
            return
        self._selected = node_id
        self.rebuild()
        self.show_history(node_id)

    def _on_history_link(self, url):
        """A name clicked in the side panel (``node:<index into the names listed>``)."""
        if url.scheme() != 'node':
            return
        try:
            node_id = self._history_links[int(url.path())]
        except (ValueError, IndexError):
            return
        self.jump_to(node_id)

    def jump_to(self, node_id):
        """Show where ``node_id`` is in the diagram: bring it into view when it is out of
        sight and outline it for a moment. The selection and the panel stay as they are."""
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

    def _description_html(self, node_id, raw):
        written, earlier, source = self.descriptions_shown(node_id, raw)
        if written is None:
            if source != 'chapter':
                return []
            shown = self.up_to_chapter()
            return [f"<p style='color:#9ca3af;margin:0 0 10px 0'>No description written up to chapter {shown}.</p>"]
        heading = (f"Description written at chapter {written.get('chapter_num')}" if source == 'chapter'
                   else "Description (from the glossary; not recorded per chapter for this book yet)")
        parts = [f"<p style='color:#9ca3af;margin:0 0 2px 0'>{html.escape(heading)}</p>",
                 f"<p style='margin:0 0 10px 0'>{html.escape(written['text'])}</p>"]
        if earlier:
            parts.append("<p style='color:#9ca3af;margin:0 0 2px 0'>Earlier descriptions</p><ul style='margin:0 0 10px 0'>")
            for old in earlier:
                parts.append(f"<li style='color:#9ca3af'>Ch. {html.escape(str(old.get('chapter_num')))}: "
                             f"{html.escape(old['text'])}</li>")
            parts.append("</ul>")
        return parts

    def show_history(self, node_id):
        entries = self.tracker.get('entries') or {}
        item = entries.get(node_id) if isinstance(entries, dict) else None
        raw = str(item.get('raw_name') or node_id) if isinstance(item, dict) else str(node_id)
        name = self._name(node_id)
        self.history_title.setText(f"{name} ({raw})" if raw != name else name)
        # Like the picture beside it, the history stops at the slider's chapter.
        chapter = self.up_to_chapter()
        history = relations_tracking.character_history(self.tracker, raw, chapter)
        self._history_links = []
        parts = self._description_html(node_id, raw)
        if not history:
            upto = f" up to chapter {chapter}" if chapter is not None else ""
            parts.append(f"<p style='color:#9ca3af'>No relations recorded from this character's side{upto}.</p>")
            self.history_panel.setHtml("".join(parts))
            return history
        for rel in history:
            target = html.escape(str(rel.get('target_translated') or rel.get('target_raw') or ''))
            if rel.get('target_key') in self._node_items:  # drawn: the name jumps to it
                self._history_links.append(rel['target_key'])
                target = (f"<a href='node:{len(self._history_links) - 1}' "
                          f"style='color:#e5e7eb'>{target}</a>")
            parts.append(f"<p style='margin:8px 0 2px 0'><b>{target}</b>: "
                         f"<span style='color:#f59e0b'>{html.escape(str(rel.get('current') or ''))}</span></p><ul style='margin:0'>")
            for run in rel.get('timeline') or []:
                start, end = run.get('from_chapter'), run.get('to_chapter')
                span = f"Ch. {start}" if start == end else f"Ch. {start}–{end}"
                parts.append(f"<li>{html.escape(str(run.get('label') or ''))} <span style='color:#9ca3af'>({span})</span></li>")
            for blip in rel.get('blips') or []:
                parts.append(f"<li style='color:#9ca3af'>{'ignored reversed' if blip.get('flip') else 'ignored one-off'}: {html.escape(str(blip.get('label') or ''))} "
                             f"(Ch. {blip.get('chapter_num')})</li>")
            parts.append("</ul>")
        self.history_panel.setHtml("".join(parts))
        return history
