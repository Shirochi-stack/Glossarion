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
* ``HalgakosLoadingWidget`` - the spinning Halgakos over one line of text (the EPUB Library's
  loading screen as a widget; the Relations Diagram shows it while its data is read).
* ``flash_saved`` - the Multi API Key Manager's saved confirmation for any Save button: green,
  "💾 Saved!", the Windows notification sound (the Placeholder Text editor, Manage Models).
"""

import os
import time

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QStyle, QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout, QWidget)

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


class HalgakosLoadingWidget(QWidget):
    """A spinning Halgakos over one line of text, centred: "something is loading".

    The same look as the EPUB Library's loading screen (64 px icon, one turn in 1.2 s,
    grey 11 pt text). The frames are rendered once; each tick shows the frame for the time
    that has passed, so the icon keeps its pace when the UI thread is busy between ticks.
    It spins while it is visible.
    """

    ICON_SIZE = 64
    PERIOD_S = 1.2
    STEP_DEGREES = 6

    def __init__(self, text="Loading…", parent=None, icon_path=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(8)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        # The icon turned 45 degrees is 64 * sqrt(2) = 91 px across: room for every frame.
        self.icon_label.setFixedSize(96, 96)
        if icon_path is None:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        self._frames = []
        pixmap = QPixmap(icon_path) if icon_path and os.path.exists(icon_path) else QPixmap()
        if not pixmap.isNull():
            base = pixmap.scaled(self.ICON_SIZE, self.ICON_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(base)
            self._frames = [base.transformed(QTransform().rotate(angle), Qt.SmoothTransformation)
                            for angle in range(0, 360, self.STEP_DEGREES)]
        layout.addWidget(self.icon_label, 0, Qt.AlignCenter)
        self.text_label = QLabel(text)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color: #888; font-size: 11pt; padding-top: 4px;")
        layout.addWidget(self.text_label, 0, Qt.AlignCenter)
        self._frame = -1
        self._started = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(lambda: self.tick())

    def set_text(self, text):
        self.text_label.setText(text)

    def is_spinning(self):
        return self._timer.isActive()

    def tick(self):
        """Show the frame that matches the time since the widget was shown."""
        if not self._frames:
            return
        index = int((time.perf_counter() - self._started) / self.PERIOD_S * len(self._frames)) % len(self._frames)
        if index != self._frame:
            self._frame = index
            self.icon_label.setPixmap(self._frames[index])

    def showEvent(self, event):
        super().showEvent(event)
        self._started = time.perf_counter()
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)


SAVED_FLASH_COLOR = "#2e7d32"  # the Multi API Key Manager's "Saved!" green


def saved_beep():
    """The Windows notification sound of a save (nothing elsewhere)."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        pass


def flash_saved(button, text="💾 Saved!", ms=1500, then=None, beep=True):
    """The saved confirmation of a Save button: it turns green and reads ``text`` for ``ms`` milliseconds,
    with the Windows notification sound, then gets its own text and style back and ``then()`` runs (e.g.
    closing the dialog). A second call while it shows restarts the wait. The timer belongs to the button, so
    nothing runs after the button is gone."""
    original = getattr(button, "_saved_flash_original", None)
    if original is None:
        original = (button.text(), button.styleSheet())
        button._saved_flash_original = original
    style = original[1]
    if "{" in style:  # a sheet with selectors: a later rule wins
        style += (f" QPushButton, QPushButton:hover {{ background-color: {SAVED_FLASH_COLOR}; "
                  f"border-color: {SAVED_FLASH_COLOR}; color: white; }}")
    else:  # bare declarations (or none)
        style = f"{style.rstrip().rstrip(';')}; background-color: {SAVED_FLASH_COLOR}; color: white;".lstrip("; ")
    button.setText(text)
    button.setStyleSheet(style)
    if beep:
        saved_beep()
    timer = getattr(button, "_saved_flash_timer", None)
    if timer is None:
        timer = QTimer(button)
        timer.setSingleShot(True)
        button._saved_flash_timer = timer
    else:
        timer.stop()
        try:
            timer.timeout.disconnect()
        except (RuntimeError, TypeError):
            pass

    def restore():
        try:
            button.setText(original[0])
            button.setStyleSheet(original[1])
            button._saved_flash_original = None
        except RuntimeError:  # the button was deleted
            return
        if then is not None:
            then()

    timer.timeout.connect(restore)
    timer.start(int(ms))


def saved_flash_active(button) -> bool:
    """True while ``flash_saved`` shows on ``button``."""
    return getattr(button, "_saved_flash_original", None) is not None
