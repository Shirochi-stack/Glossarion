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
* ``dialog_button`` - the Placeholder Text editor's blue Save / red Reset / grey Close bottom-row
  buttons (also the Refinement Prompt dialog).
* ``link_placeholders`` / ``link_placeholders_in_label`` - AI prompt placeholders ({name}) shown like links:
  blue and underlined in a prompt editor, an "Available placeholders" line or a label, with a tooltip
  that says what the placeholder expands to at the moment of the hover (display only).
"""

import html
import os
import re
import time

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QRect, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QCursor, QInputMethodEvent, QPainter, QPen, QPixmap,
                           QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTransform)
from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QToolTip,
                               QVBoxLayout, QWidget)

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


# background, border, hover of a dialog's bottom-row buttons (the Placeholder Text editor's row)
DIALOG_BUTTON_COLORS = {
    "save": ("#2f5f8f", "#4a78a8", "#3f78ad"),   # blue
    "reset": ("#6f2f2f", "#944545", "#8a3a3a"),  # red
    "close": ("#3a3a3a", "#555555", "#4a4a4a"),  # grey
}


def dialog_button(text, kind):
    """A bottom-row button of the Placeholder Text editor's style: blue ``"save"``, red ``"reset"`` or grey
    ``"close"``. Its minimum width is its resting width, so a shorter ``flash_saved`` text (e.g. "✅ Reset!")
    does not move the row."""
    background, border, hover = DIALOG_BUTTON_COLORS[kind]
    button = QPushButton(text)
    button.setAutoDefault(False)
    button.setFixedHeight(28)
    button.setStyleSheet(f"""
        QPushButton {{ background-color: {background}; color: #ffffff; font-size: 11px;
            font-weight: bold; border: 1px solid {border}; border-radius: 4px; padding: 0px 12px; }}
        QPushButton:hover {{ background-color: {hover}; }}
    """)
    button.ensurePolished()
    button.setMinimumWidth(button.sizeHint().width())
    return button


# ── AI prompt placeholders shown like links ──────────────────────────────────────────────────────
# Display only: nothing here changes a prompt's text, its undo history or its modified flag.

#: The app's link blue (the "Available placeholders" lines, focus borders, the context menu border).
PLACEHOLDER_LINK_COLOR = "#5a9fd4"
#: The same link on a light input background, where #5a9fd4 reads poorly.
PLACEHOLDER_LINK_COLOR_ON_LIGHT = "#1f5fa8"
#: One ``{name}`` token.
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PLACEHOLDER_HREF = "placeholder:"
_TOOLTIP_WIDTH = 520
_TOOLTIP_MUTED = "#9aa0a6"


def _placeholder_names(names):
    """The names valid now: ``names`` is a callable (called now) or a collection."""
    try:
        value = names() if callable(names) else names
    except Exception:
        return frozenset()
    return value if isinstance(value, frozenset) else frozenset(value or ())


def placeholder_spans(text, names):
    """``[(start, end, name)]`` of each ``{name}`` in ``text`` whose name is valid (``names``)."""
    if not text or "{" not in text:
        return []
    valid = _placeholder_names(names)
    if not valid:
        return []
    return [(m.start(), m.end(), m.group(1)) for m in PLACEHOLDER_RE.finditer(text) if m.group(1) in valid]


def placeholder_at(text, index, names):
    """The ``(start, end, name)`` of the valid placeholder at caret position ``index`` (from its first
    character's left edge to its last character's right edge), or None."""
    for start, end, name in placeholder_spans(text, names):
        if start <= index <= end:
            return start, end, name
    return None


def placeholder_link_color(widget=None):
    """The link blue for ``widget``'s input background: #5a9fd4 on the app's dark inputs, a darker blue on a
    light one."""
    try:
        if widget is not None and widget.palette().base().color().lightness() > 160:
            return PLACEHOLDER_LINK_COLOR_ON_LIGHT
    except Exception:
        pass
    return PLACEHOLDER_LINK_COLOR


def _tooltip_lines(lines, muted=False):
    text = html.escape("\n".join(lines), quote=False).replace("\n", "<br>")
    return f"<span style='color: {_TOOLTIP_MUTED};'>{text}</span>" if muted else text


def _placeholder_tooltip_body(value):
    """``(html, long)`` for one value (see placeholder_tooltip)."""
    if isinstance(value, dict):
        text = str(value.get("text") or "")
        note = str(value.get("note") or "")
        if value.get("error"):
            body = _tooltip_lines([f"The current value could not be read: {value['error']}"], muted=True)
            return body, False
        if value.get("per_request"):
            lines = ["Filled in for each request: " + (text or note)]
            return _tooltip_lines(lines, muted=True), len(lines[0]) > 70
        if value.get("active") is False:
            lines = [str(value.get("inactive") or "Inactive right now: removed from the prompt.")]
            if note:
                lines.append(note)
            return _tooltip_lines(lines, muted=True), any(len(line) > 70 for line in lines)
        extra = [str(value["extra"])] if value.get("extra") else []
    else:
        text, extra = str(value if value is not None else ""), []
    body = _tooltip_lines([text]) if text else _tooltip_lines(["(empty)"], muted=True)
    if extra:
        body += "<br>" + _tooltip_lines(extra, muted=True)
    return body, len(text) > 300 or any(len(line) > 70 for line in [*text.split("\n"), *extra])


def placeholder_tooltip(name, value):
    """The tooltip of ``{name}`` (rich text, the value shown verbatim and wrapped).

    ``value`` is what the placeholder says now: a str (its expansion), a dict with ``text`` and the optional
    keys ``active`` (False: removed from the prompt now), ``inactive`` (what removed means there),
    ``per_request`` (``text`` describes data of each request), ``note`` (shown under an inactive value),
    ``extra`` (a short muted line under the value), ``error``; or a list of ``(label, value)`` pairs, one
    section each (e.g. Balanced/Full and Minimal).
    """
    sections = value if isinstance(value, list) else [(None, value)]
    rows = [f"<b>{html.escape('{' + str(name) + '}', quote=False)}</b>"]
    wide = False
    for label, item in sections:
        body, long_body = _placeholder_tooltip_body(item)
        wide = wide or long_body
        if label:
            rows.append(f"<i>{html.escape(str(label), quote=False)}</i>")
        rows.append(body)
    width = f" width='{_TOOLTIP_WIDTH}'" if wide else ""
    cells = "".join(f"<tr><td style='white-space: pre-wrap; padding-top: 2px;'>{row}</td></tr>" for row in rows)
    return f"<qt><table{width} cellspacing='0' cellpadding='0'>{cells}</table></qt>"


def _resolve_placeholder(resolver, name):
    try:
        return resolver(name)
    except Exception as exc:  # a broken resolver must not break the hover
        return {"error": str(exc) or type(exc).__name__}


class PlaceholderHighlighter(QSyntaxHighlighter):
    """Formats each valid ``{name}`` of a QTextEdit / QPlainTextEdit like a link (blue, underlined).

    ``names``: a callable returning the names valid in that editor now (or a collection); call
    ``rehighlight()`` after it changes without a text change. Formatting only (QTextLayout formats): the
    text, the undo stack and the modified flag are untouched.
    """

    def __init__(self, editor, names):
        super().__init__(editor.document())
        self._editor = editor
        self._names = names
        self._color = None
        self._format = QTextCharFormat()

    def set_names(self, names):
        self._names = names
        self.rehighlight()

    def _link_format(self):
        color = placeholder_link_color(self._editor)
        if color != self._color:
            self._color = color
            self._format = QTextCharFormat()
            self._format.setForeground(QColor(color))
            self._format.setFontUnderline(True)
            self._format.setUnderlineColor(QColor(color))
        return self._format

    def refresh_color(self):
        """Re-format when the editor's background changed between dark and light."""
        if self._color is not None and placeholder_link_color(self._editor) != self._color:
            self.rehighlight()

    def highlightBlock(self, text):
        spans = placeholder_spans(text, self._names)
        if spans:
            link = self._link_format()
            for start, end, _name in spans:
                self.setFormat(start, end - start, link)


class _PlaceholderLinks(QObject):
    """Link look + live hover tooltip for the placeholders of one editor or QLineEdit (link_placeholders)."""

    _LINE_REFORMAT_EVENTS = (QEvent.Show, QEvent.FontChange, QEvent.StyleChange, QEvent.PaletteChange)

    def __init__(self, widget, names, resolver):
        self._widget = widget
        self._line = isinstance(widget, QLineEdit)
        self._target = widget if self._line else widget.viewport()
        super().__init__(self._target)
        self._names = names
        self._resolver = resolver
        self._shown = None
        self._hand = False
        self._rest_cursor = None
        self.highlighter = None if self._line else PlaceholderHighlighter(widget, self._current_names)
        self._target.setMouseTracking(True)
        self._target.installEventFilter(self)
        if self._line:
            self.refresh()

    def _current_names(self):
        return _placeholder_names(self._names)

    def update(self, names=None, resolver=None):
        if names is not None:
            self._names = names
        if resolver is not None:
            self._resolver = resolver
        self.refresh()

    def refresh(self):
        """Re-apply the link look (after the valid names changed without a text change)."""
        if self.highlighter is not None:
            self.highlighter.rehighlight()
            return
        line = self._widget
        link = QTextCharFormat()
        color = QColor(placeholder_link_color(line))
        link.setForeground(color)
        link.setFontUnderline(True)
        link.setUnderlineColor(color)
        cursor = line.cursorPosition()
        # A QLineEdit takes extra character formats only from an input method event (an empty preedit:
        # the text, the cursor and the undo history stay as they are).
        attributes = [QInputMethodEvent.Attribute(QInputMethodEvent.TextFormat, start - cursor, end - start, link)
                      for start, end, _name in placeholder_spans(line.text(), self._current_names)]
        QCoreApplication.sendEvent(line, QInputMethodEvent("", attributes))

    def _hit(self, pos):
        """``(name, rect or None)`` of the placeholder under ``pos`` (target coordinates), or None."""
        widget = self._widget
        if self._line:
            span = placeholder_at(widget.text(), widget.cursorPositionAt(pos), self._current_names)
            return (span[2], None) if span else None
        cursor = widget.cursorForPosition(pos)
        block = cursor.block()
        span = placeholder_at(block.text(), cursor.positionInBlock(), self._current_names)
        if span is None:
            return None
        first, last = QTextCursor(block), QTextCursor(block)
        first.setPosition(block.position() + span[0])
        last.setPosition(block.position() + span[1])
        left, right = widget.cursorRect(first), widget.cursorRect(last)
        if left.top() == right.top():  # on one line: the pointer must be on the token itself
            rect = QRect(left.left(), left.top(), max(1, right.left() - left.left()), left.height())
            return (span[2], rect) if rect.contains(pos) else None
        return (span[2], None) if left.top() <= pos.y() <= right.bottom() else None  # wrapped token

    def _set_hand(self, on):
        if on == self._hand:
            return
        self._hand = on
        if on:
            self._rest_cursor = self._target.cursor()
            self._target.setCursor(Qt.PointingHandCursor)
        elif self._rest_cursor is not None:
            self._target.setCursor(self._rest_cursor)
        else:
            self._target.unsetCursor()

    def _show(self, name, rect, global_pos):
        tip = placeholder_tooltip(name, _resolve_placeholder(self._resolver, name))
        if rect is not None:
            QToolTip.showText(global_pos, tip, self._target, rect)
        else:
            QToolTip.showText(global_pos, tip, self._target)
        self._shown = name

    def _hide(self):
        self._shown = None
        QToolTip.hideText()

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.MouseMove:
            hit = None if event.buttons() != Qt.NoButton else self._hit(event.position().toPoint())
            self._set_hand(hit is not None)
            if self._shown is not None and (hit is None or hit[0] != self._shown):
                if hit is None:
                    self._hide()
                else:
                    self._show(hit[0], hit[1], event.globalPosition().toPoint())
        elif kind == QEvent.ToolTip:
            hit = self._hit(event.pos())
            if hit is not None:
                self._show(hit[0], hit[1], event.globalPos())
                return True
            if self._shown is not None:
                self._hide()
        elif kind == QEvent.Leave:
            self._set_hand(False)
            if self._shown is not None:
                self._hide()
        elif self._line and kind in self._LINE_REFORMAT_EVENTS:
            self.refresh()
        elif kind == QEvent.Show and self.highlighter is not None:
            self.highlighter.refresh_color()
        return False


def link_placeholders(widget, names, resolver):
    """Show the AI prompt placeholders of a prompt editor (QTextEdit / QPlainTextEdit) or a line (QLineEdit)
    like links, with a tooltip of what each says at the moment of the hover.

    ``names``: a callable returning the names the code replaces in that prompt now (or a collection); only
    those are marked. ``resolver(name)``: called at hover time, returns the value (see placeholder_tooltip).
    The pointer becomes a pointing hand over a placeholder; clicking still places the caret. Calling it
    again on the same widget replaces ``names`` / ``resolver``. Returns the helper (``refresh()``).
    """
    links = getattr(widget, "_placeholder_links", None)
    if links is not None:
        links.update(names, resolver)
        return links
    links = _PlaceholderLinks(widget, names, resolver)
    widget._placeholder_links = links
    return links


def link_described_placeholders(widget, descriptions):
    """link_placeholders for a prompt whose placeholders hold data of each request: ``descriptions`` maps
    each name to what it is filled with, which the tooltip shows."""
    return link_placeholders(widget, *described_placeholder_resolver(descriptions))


def placeholder_label_html(text, names, color=PLACEHOLDER_LINK_COLOR):
    """``text`` as label rich text, each valid placeholder a blue, underlined link; reads the same."""
    def plain(part):
        return html.escape(part, quote=False).replace("\n", "<br>")

    out, pos = [], 0
    for start, end, name in placeholder_spans(text or "", names):
        out.append(plain(text[pos:start]))
        out.append(f"<a href='{_PLACEHOLDER_HREF}{name}' style='color: {color}; text-decoration: underline;'>"
                   f"{html.escape(text[start:end], quote=False)}</a>")
        pos = end
    out.append(plain((text or "")[pos:]))
    return "<span style='white-space: pre-wrap;'>" + "".join(out) + "</span>"


class _LabelPlaceholderLinks(QObject):
    """The live hover tooltip of a label's placeholder links (link_placeholders_in_label)."""

    def __init__(self, label):
        super().__init__(label)
        self._label = label
        self.names = ()
        self.resolver = None
        self.hovered = None
        label.linkHovered.connect(self._link_hovered)
        label.installEventFilter(self)

    def _show(self):
        QToolTip.showText(QCursor.pos(), placeholder_tooltip(
            self.hovered, _resolve_placeholder(self.resolver, self.hovered)), self._label)

    def _link_hovered(self, href):
        if href.startswith(_PLACEHOLDER_HREF):
            self.hovered = href[len(_PLACEHOLDER_HREF):]
            self._show()
        elif self.hovered is not None:
            self.hovered = None
            QToolTip.hideText()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ToolTip and self.hovered is not None:
            self._show()  # the label's own tooltip (if any) waits until the pointer leaves the link
            return True
        if event.type() == QEvent.Leave and self.hovered is not None:
            self.hovered = None
            QToolTip.hideText()
        return False


def link_placeholders_in_label(label, names, resolver, text=None):
    """Show the placeholders of a QLabel's text like links with the live tooltip of link_placeholders.

    ``text``: the label's plain text (default: the text it was given before, or its current text); call
    this again instead of ``setText`` when the text changes. Word wrap, selection, the context menu and
    the visible text stay as they were; clicking a link opens nothing.
    """
    links = getattr(label, "_placeholder_label_links", None)
    if text is None:
        text = links.text if links is not None else label.text()
    if links is None:
        links = _LabelPlaceholderLinks(label)
        label._placeholder_label_links = links
    links.text, links.names, links.resolver = text, names, resolver
    label.setTextFormat(Qt.RichText)
    label.setOpenExternalLinks(False)
    label.setTextInteractionFlags(label.textInteractionFlags() | Qt.LinksAccessibleByMouse)
    label.setText(placeholder_label_html(text, names))
    return links


class _PlaceholderNameTooltip(QObject):
    def __init__(self, widget, name, resolver):
        super().__init__(widget)
        self.name = name
        self.resolver = resolver
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ToolTip:
            name = self.name() if callable(self.name) else self.name
            if name:
                QToolTip.showText(event.globalPos(), placeholder_tooltip(
                    name, _resolve_placeholder(self.resolver, name)), obj)
                return True
        return False


def placeholder_tooltip_for(widget, name, resolver):
    """``widget``'s tooltip becomes the live value of placeholder ``name`` (a callable: the name now, e.g.
    the one a combo box has selected), resolved at hover time."""
    helper = _PlaceholderNameTooltip(widget, name, resolver)
    widget._placeholder_name_tooltip = helper
    return helper


def described_placeholder_resolver(descriptions):
    """The resolver of placeholders that hold data of each request: ``descriptions`` maps each name (braces
    optional) to what it is filled with. Returns ``(names, resolver)``."""
    descriptions = {str(name).strip("{}"): text for name, text in dict(descriptions).items()}
    return frozenset(descriptions), lambda name: {"per_request": True, "text": descriptions[name]}


def link_described_placeholders_in_label(label, descriptions, text=None):
    """link_placeholders_in_label for placeholders that hold data of each request (see
    described_placeholder_resolver)."""
    names, resolver = described_placeholder_resolver(descriptions)
    return link_placeholders_in_label(label, names, resolver, text=text)


def target_language_now(owner):
    """What ``{target_lang}`` becomes now for ``owner`` (the main window: its target language selector,
    see TranslatorGUI._prompt_placeholder_value), else its saved output language."""
    resolver = getattr(owner, "_prompt_placeholder_value", None)
    if callable(resolver):
        value = resolver("target_lang")
        if isinstance(value, str) and value.strip():
            return value
    config = getattr(owner, "config", None) or {}
    return str(config.get("output_language") or "English")
