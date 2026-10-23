"""Small Qt widgets / helpers shared by several dialogs (light: PySide6 only, no app imports).

* ``NavComboRow`` - a labelled ◀ combo box ▶ row (the Glossary Manager's "Glossary File:" switcher,
  the QA scanner's glossary name review "Output folder:").
* ``CheckmarkDelegate`` - paints Other Settings' checkbox (``_create_styled_checkbox``) in a tree
  column (EPUB Library raw-file matching, the QA scanner's glossary name review).
* ``CONTEXT_MENU_STYLE`` - the Progress Manager's right-click menu style (no empty icon gutter).
* ``run_with_progress`` - run work on a thread behind a progress window with a centered Cancel (the QA
  scanner's glossary name review, Other Settings' payload summary).
* ``center_messagebox_buttons`` - centered, optionally larger message-box buttons (Other Settings,
  the QA scanner's glossary name check).
"""

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QStyle, QStyledItemDelegate, QStyleOptionViewItem, QWidget)

# Context menus without the empty icon gutter (Progress Manager rows, the QA scanner's glossary
# name review)
CONTEXT_MENU_STYLE = (
    "QMenu {"
    "  padding: 4px;"
    "  background-color: #2b2b2b;"
    "  color: white;"
    "  border: 1px solid #5a9fd4;"
    "} "
    "QMenu::icon { width: 0px; } "
    "QMenu::item {"
    "  padding: 6px 12px;"
    "  background-color: transparent;"
    "} "
    "QMenu::item:selected {"
    "  background-color: #17a2b8;"
    "  color: white;"
    "} "
    "QMenu::item:pressed {"
    "  background-color: #138496;"
    "}"
)

ARROW_STYLE = ("QPushButton { background: #444; color: white; border-radius: 4px; font-size: 10pt; padding: 0; } "
               "QPushButton:hover { background: #666; }")


class NavComboRow(QWidget):
    """``label`` ◀ combo ▶: the arrows wrap around and show only when there is more than one item;
    the mouse wheel does not change the selection."""

    def __init__(self, label, previous_tip, next_tip, *, tooltip=None, parent=None):
        super().__init__(parent)
        tooltip = tooltip or (lambda text: text)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        layout.addWidget(self.label)
        self.prev_button = self._arrow("◀", tooltip(previous_tip), -1)
        layout.addWidget(self.prev_button)
        self.combo = QComboBox()
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo.setStyleSheet("QComboBox { padding: 4px; }")
        self.combo.wheelEvent = lambda e: e.ignore()
        layout.addWidget(self.combo)
        self.next_button = self._arrow("▶", tooltip(next_tip), 1)
        layout.addWidget(self.next_button)
        self.update_arrows()

    def _arrow(self, text, tip, direction):
        button = QPushButton(text)
        button.setFixedSize(24, 24)
        button.setStyleSheet(ARROW_STYLE)
        button.setToolTip(tip)
        button.clicked.connect(lambda: self.step(direction))
        return button

    def step(self, direction):
        """Select the previous (-1) / next (1) item, wrapping around."""
        count = self.combo.count()
        if count <= 1:
            return
        self.combo.setCurrentIndex((self.combo.currentIndex() + direction) % count)

    def update_arrows(self):
        """Show ◀ ▶ only when there is something to switch to."""
        show = self.combo.count() > 1
        self.prev_button.setVisible(show)
        self.next_button.setVisible(show)


class CheckmarkDelegate(QStyledItemDelegate):
    """Paint an Other-Settings-styled checkbox on column 0.

    Draws a 14×14 rounded square with a #5a9fd4 border and
    #2d2d2d unchecked fill / #5a9fd4 checked fill, plus a
    white “✓” stroke inside the checked state. Rows whose
    flags lack ``ItemIsUserCheckable`` (the “no match”
    rows) render as an empty column so they don’t look
    interactive. Click handling is left to the base class
    — Qt’s default ``editorEvent`` still toggles the check
    state on the model when the user clicks the cell.
    (QTreeView::indicator stylesheets do not draw the tick on Windows.)
    """

    def paint(self, painter, option, index):  # type: ignore[override]
        if index.column() != 0:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        painter.save()
        try:
            # Row-level background / selection highlight
            # (mirrors the QTreeWidget::item:selected rule).
            if opt.state & QStyle.State_Selected:
                painter.fillRect(opt.rect, QColor("#3a3a5e"))
            flags = index.flags()
            if not (flags & Qt.ItemIsUserCheckable):
                return
            # Read the check state through BOTH ``opt.checkState``
            # (populated by ``initStyleOption``) AND the raw
            # ``CheckStateRole`` data, normalising the value to
            # a plain int (``Qt.Checked`` = 2). PySide6 6.4+
            # ships ``Qt.CheckState`` as a strict Python enum
            # that doesn’t support ``int(enum_value)`` AND on
            # some Windows builds returns the enum from
            # ``data()`` while ``==`` against ``Qt.Checked``
            # silently yields False. Pulling ``.value`` (or
            # falling back to ``int()``) dodges both pitfalls,
            # and comparing against the literal ``2`` (the
            # wire value of ``Qt.Checked``) means we never
            # have to call ``int()`` on the enum constant
            # itself — which is what crashed the previous
            # iteration with ``TypeError: int() argument must
            # be a string, a bytes-like object or a real
            # number, not 'CheckState'``.
            def _cs_int(value):
                if value is None:
                    return 0
                val = getattr(value, "value", value)
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return 0
            opt_cs = _cs_int(opt.checkState)
            raw_cs_int = _cs_int(index.data(Qt.CheckStateRole))
            # 2 == ``Qt.CheckState.Checked.value`` in every Qt
            # build shipped so far; safe to hardcode.
            is_checked = (opt_cs == 2 or raw_cs_int == 2)
            rect = opt.rect
            size = 14
            box_x = rect.x() + max(
                4, (rect.width() - size) // 2)
            box_y = rect.y() + max(
                2, (rect.height() - size) // 2)
            box = QRect(box_x, box_y, size, size)
            painter.setRenderHint(QPainter.Antialiasing, True)
            hovered = bool(opt.state & QStyle.State_MouseOver)
            border_color = (
                QColor("#7bb3e0") if hovered
                else QColor("#5a9fd4"))
            painter.setPen(QPen(border_color, 1))
            if is_checked:
                painter.setBrush(QBrush(QColor("#5a9fd4")))
            else:
                painter.setBrush(QBrush(QColor("#2d2d2d")))
            painter.drawRoundedRect(box, 2, 2)
            if is_checked:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(
                    QColor("white"), 2,
                    Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                # Classic “✓”: short leg from (3,7) to (6,10),
                # long leg from (6,10) to (11,4) inside the box.
                x = box.x()
                y = box.y()
                painter.drawLine(x + 3, y + 7, x + 6, y + 10)
                painter.drawLine(x + 6, y + 10, x + 11, y + 4)
        finally:
            painter.restore()


def center_messagebox_buttons(
    msg_box,
    *,
    min_button_width=None,
    min_button_height=None,
):
    """Center message-box buttons and optionally enlarge their click targets."""
    button_box = msg_box.findChild(QDialogButtonBox)
    if button_box:
        button_box.setCenterButtons(True)
        for button in button_box.buttons():
            if min_button_width is not None:
                button.setMinimumWidth(min_button_width)
            if min_button_height is not None:
                button.setMinimumHeight(min_button_height)


def run_with_progress(parent, title, work, *, text="Working...", on_tick=None, thread_name="ProgressWork"):
    """Run ``work(state, cancelled)`` on a worker thread behind a modal progress window, so the GUI
    keeps responding. ``work`` updates ``state`` (``text``; ``value`` / ``maximum``, maximum 0 = busy)
    and checks ``cancelled()``. ``on_tick()`` runs on the GUI thread every 100 ms (e.g. to flush log
    lines). Returns what ``work`` returned; None when cancelled; re-raises ``work``'s error."""
    import threading
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QDialog, QProgressBar, QVBoxLayout

    cancel = threading.Event()
    state = {"text": text, "value": 0, "maximum": 0}
    outcome = {"done": False, "result": None, "error": None}

    def run():
        try:
            outcome["result"] = work(state, cancel.is_set)
        except Exception as e:  # noqa: BLE001 - re-raised on the GUI thread
            outcome["error"] = e
        finally:
            outcome["done"] = True

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumWidth(460)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 14)
    layout.setSpacing(10)
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    label.setWordWrap(True)
    layout.addWidget(label)
    bar = QProgressBar()
    bar.setRange(0, 0)
    layout.addWidget(bar)
    # a centered, wider Cancel (QProgressDialog keeps its button at the right)
    cancel_row = QHBoxLayout()
    cancel_button = QPushButton("Cancel")
    cancel_button.setMinimumWidth(120)
    cancel_button.setMinimumHeight(36)
    cancel_row.addStretch()
    cancel_row.addWidget(cancel_button)
    cancel_row.addStretch()
    layout.addLayout(cancel_row)

    def request_cancel(*_args):
        cancel.set()
        cancel_button.setEnabled(False)
        label.setText("Stopping...")

    cancel_button.clicked.connect(request_cancel)
    dialog.rejected.connect(request_cancel)  # Esc / the window's close button
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(100)

    def tick():
        if on_tick is not None:
            on_tick()
        if not cancel.is_set():
            label.setText(state["text"])
            bar.setRange(0, state["maximum"])
            bar.setValue(state["value"])
        if outcome["done"]:
            timer.stop()
            loop.quit()

    timer.timeout.connect(tick)
    threading.Thread(target=run, name=thread_name, daemon=True).start()
    timer.start()
    dialog.show()
    loop.exec()
    dialog.rejected.disconnect(request_cancel)
    dialog.close()
    if on_tick is not None:
        on_tick()
    if cancel.is_set():
        return None
    if outcome["error"] is not None:
        raise outcome["error"]
    return outcome["result"]
