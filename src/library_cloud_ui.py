"""The desktop Library's cloud windows (Qt): Library settings, Share and "Sync to Drive now".

* :class:`LibrarySettingsDialog` - the ⚙ Settings button in the Library header: Google Drive sync at
  the top (sign in / out, the folder name, the sync mode, the switch, the last sync), then "Share via
  link" with the hosting sites, each with its switch and the first-use consent (the same wording as
  mobile, worded for a computer).
* :class:`ShareDialog` - "Share…" in a card's context menu and the Share button in Book Details:
  uploads the book's compiled file to an enabled site in the background, shows the link with Copy
  and lists the book's saved links (Copy / Delete). With no compiled file it offers the Library's
  own Compile; with no site turned on it points to Library settings.
* :func:`sync_book_now` - "Sync to Drive now" in the card context menu.

All logic is in the GUI-free ``library_cloud`` (settings file, sign-in, uploads), ``drive_sync_core``
and ``share_hosting``; this module only shows it. Blocking work (the browser sign-in wait, uploads,
deleting a link at the site) runs on worker threads and reports back through Qt signals.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

import drive_sync_core as dsc
import library_cloud
from epub_library import (
    _LIBRARY_FORM_STYLE,
    _create_styled_checkbox,
    _find_translator_gui,
    _open_folder_in_explorer,
    _resolve_book_output_folder,
    _resolve_book_translated_file,
    _workspace_compile_kind,
)
from qt_widget_helpers import dialog_button, flash_saved, style_combo
from share_hosting import CancelToken, ShareError, format_size, provider_info
from share_hosting_litterbox import TIME_CHOICES as LITTERBOX_TIME_CHOICES

#: The extra rules the cloud windows add to the Library's form style.
CLOUD_STYLE = _LIBRARY_FORM_STYLE + """
            QFrame#cloudSection { background: #1c1c2c; border: 1px solid #2e2e48; border-radius: 8px; }
            QLabel#sectionTitle { color: #e0e0e0; font-size: 11.5pt; font-weight: bold; }
            QLabel#note { color: #8a90a8; font-size: 8.5pt; }
            QLabel#status { color: #c8cbe0; font-size: 9.5pt; }
            QLabel#ok { color: #7bd88f; font-size: 9.5pt; font-weight: bold; }
            QLabel#warn { color: #ffb347; font-size: 9pt; }
            QComboBox {
                background: #202033; color: #f0f0f5; border: 1px solid #3a3a5e; border-radius: 5px;
                padding: 5px 8px; font-size: 9.5pt;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #17a2b8;
                border: 1px solid #3a3a5e;
            }
            QComboBox:disabled, QLineEdit:disabled { color: #666a80; }
            QPushButton:disabled { color: #666a80; background: #1e1e2c; border-color: #2a2a3e; }
            QProgressBar { background: #202033; border: 1px solid #3a3a5e; border-radius: 4px; height: 10px;
                text-align: center; color: #c8cbe0; font-size: 8pt; }
            QProgressBar::chunk { background: #6c63ff; border-radius: 3px; }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { width: 10px; background: #12121e; }
            QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 5px; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """

#: Added to a ``qt_widget_helpers.dialog_button`` whose own sheet would otherwise look enabled.
DISABLED_BUTTON_RULE = (" QPushButton:disabled { background-color: #2a2a3a; color: #6f7388; "
                        "border-color: #34344a; }")

#: Send (send.vis.ee) choices shown in Library settings.
SEND_EXPIRY_LABELS = ((300, "5 minutes"), (3600, "1 hour"), (86400, "1 day"), (3 * 86400, "3 days"))
SEND_DOWNLOAD_CHOICES = (1, 2, 3, 5, 10, 20)
#: The consent window's button when Library settings turns a site on (nothing is opened or uploaded
#: there; the Share window does that later). Mobile's settings switch says the same.
TURN_ON_LABEL = "Turn on"
TURN_ON_AND_SIGN_IN_LABEL = "Turn on and sign in"


# ---------------------------------------------------------------------------------------------------
# the shared controller and its change signal
# ---------------------------------------------------------------------------------------------------


class _CloudSignals(QObject):
    """``changed(what)``: "drive" | "sync" | "links" from library_cloud (worker threads too)."""

    changed = Signal(str)


_SIGNALS: Optional[_CloudSignals] = None


def cloud_signals() -> _CloudSignals:
    global _SIGNALS
    if _SIGNALS is None:
        _SIGNALS = _CloudSignals()
        app = QApplication.instance()
        if app is not None:
            _SIGNALS.moveToThread(app.thread())
    return _SIGNALS


def _emit_changed(what: str) -> None:
    try:
        cloud_signals().changed.emit(str(what))
    except RuntimeError:  # the app is closing
        pass


def controller(widget: Any = None) -> library_cloud.LibraryCloud:
    """The app's one :class:`library_cloud.LibraryCloud`, its log going to the main window's log."""
    gui = _find_translator_gui(widget) if widget is not None else None
    log_line = getattr(gui, "append_log", None)
    return library_cloud.instance(log_line=log_line, on_change=_emit_changed)


class _Task(QObject):
    """Run ``work(task)`` on a daemon thread; ``done(result, error)`` and ``progress(sent, total)`` arrive
    on the GUI thread (connect them to a QObject's methods)."""

    done = Signal(object, object)
    progress = Signal(object, object)
    _running: set = set()

    def __init__(self, work: Callable[["_Task"], Any], name: str = "library-cloud") -> None:
        super().__init__()
        self._work = work
        self._name = name
        app = QApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self.done.connect(self._finished)

    def _finished(self, _result: Any, _error: Any) -> None:  # GUI thread
        QTimer.singleShot(0, lambda: _Task._running.discard(self))

    def start(self) -> "_Task":
        _Task._running.add(self)  # kept alive until it reports
        threading.Thread(target=self._run, name=self._name, daemon=True).start()
        return self

    def report(self, sent: Any, total: Any) -> None:
        try:
            self.progress.emit(sent, total)
        except RuntimeError:
            pass

    def _run(self) -> None:
        result, error = None, None
        try:
            result = self._work(self)
        except Exception as exc:  # shown to the user by the receiver
            error = exc
        try:
            self.done.emit(result, error)
        except RuntimeError:
            pass


def _error_text(error: Any) -> str:
    if isinstance(error, ShareError):
        return error.message
    return f"{type(error).__name__}: {error}" if error else ""


def _ago(timestamp: Any) -> str:
    try:
        seconds = max(0.0, time.time() - float(timestamp))
    except (TypeError, ValueError):
        return ""
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(timestamp)))


def last_sync_text(record: Any) -> str:
    """"Last sync: Book · 1 new, 0 updated, 3 unchanged · 5 min ago" (or "No sync yet")."""
    if not isinstance(record, dict):
        return "No sync yet."
    mark = "" if record.get("ok") else "⚠️ "
    parts = [str(record.get("label") or "book"), str(record.get("summary") or ""), _ago(record.get("time"))]
    return f"{mark}Last sync: " + " · ".join(p for p in parts if p)


def _label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def _section(title: str) -> tuple:
    frame = QFrame()
    frame.setObjectName("cloudSection")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)
    layout.addWidget(_label(title, "sectionTitle"))
    return frame, layout


def _combo() -> QComboBox:
    combo = QComboBox()
    combo.setFocusPolicy(Qt.StrongFocus)
    combo.setCursor(Qt.PointingHandCursor)
    style_combo(combo)
    return combo


def open_in_browser(url: str, parent: Any = None) -> bool:
    """Open ``url`` in the default browser (Qt first, then Python's ``webbrowser``); says so when neither
    could, with the address to open by hand. True when a browser was asked to open it."""
    opened = False
    try:
        opened = QDesktopServices.openUrl(QUrl(url)) is not False  # Qt says False when no handler took it
    except Exception:
        opened = False
    if not opened:
        try:
            import webbrowser

            opened = bool(webbrowser.open(url))
        except Exception:
            opened = False
    if not opened:
        QMessageBox.warning(parent, "Open in browser", f"Could not open your browser. Open this address in it:\n{url}")
    return opened


def _small_button(text: str, tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setCursor(Qt.PointingHandCursor)
    button.setAutoDefault(False)
    if tooltip:
        button.setToolTip(tooltip)
    return button


# ---------------------------------------------------------------------------------------------------
# the consent window (first use of a hosting site)
# ---------------------------------------------------------------------------------------------------


class ConsentDialog(QDialog):
    """``share_hosting.consent_text`` (the mobile sheet's wording) as a dialog: the confirmation
    checkbox unlocks the confirm button."""

    def __init__(self, consent: Any, parent: Any = None, *, confirm_label: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(consent.title)
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet(CLOUD_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        layout.addWidget(_label(consent.title, "sectionTitle", wrap=True))
        for line in consent.lines:
            layout.addWidget(_label(line, "status", wrap=True))
        if consent.terms_url:
            terms = _label(f'<a href="{consent.terms_url}" style="color:#5a9fd4;">Terms and privacy</a>')
            terms.setTextFormat(Qt.RichText)
            terms.setOpenExternalLinks(True)  # opened only when the user clicks it
            layout.addWidget(terms)
        self.checkbox = _create_styled_checkbox(consent.checkbox)
        layout.addWidget(self.checkbox)
        row = QHBoxLayout()
        row.addStretch()
        cancel = dialog_button("Cancel", "close")
        cancel.clicked.connect(self.reject)
        self.confirm = dialog_button(confirm_label or consent.confirm, "save")
        # greyed until the box is ticked (the button's own sheet looks enabled either way)
        self.confirm.setStyleSheet(self.confirm.styleSheet() + DISABLED_BUTTON_RULE)
        self.confirm.setEnabled(False)
        self.confirm.clicked.connect(self.accept)
        self.checkbox.toggled.connect(self.confirm.setEnabled)
        row.addWidget(cancel)
        row.addWidget(self.confirm)
        layout.addLayout(row)


def ask_consent(parent: Any, cloud: library_cloud.LibraryCloud, provider_id: str, path: str = "", *,
                confirm_label: str = "") -> bool:
    """Show the consent window; True when the user ticked and confirmed (the site is then turned on).
    ``confirm_label``: the button when confirming only turns the site on (the consent text's own
    "Open transfer.it" / "Use Gofile" promise the share itself)."""
    dialog = ConsentDialog(cloud.consent_text(provider_id, path or None), parent, confirm_label=confirm_label)
    if dialog.exec() != QDialog.Accepted:
        return False
    cloud.store.set_provider_enabled(provider_id, True, consent=True)
    return True


# ---------------------------------------------------------------------------------------------------
# Library settings
# ---------------------------------------------------------------------------------------------------


class LibrarySettingsDialog(QDialog):
    """Library settings: Google Drive sync, then Share via link (decisions D and E)."""

    def __init__(self, parent: Any = None, cloud: Optional[library_cloud.LibraryCloud] = None) -> None:
        super().__init__(parent)
        self.cloud = cloud or controller(parent)
        self._sign_in: Any = None
        self._sign_in_cancel: Optional[CancelToken] = None
        self._task: Optional[_Task] = None
        self.setWindowTitle("Library settings")
        self.setMinimumWidth(640)
        self.resize(700, 820)
        self.setStyleSheet(CLOUD_STYLE)
        self._build()
        self.load()
        cloud_signals().changed.connect(self._on_cloud_changed)

    # ---- layout ----------------------------------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)
        heading = _label("⚙️  Library settings", "sectionTitle")
        heading.setStyleSheet("font-size: 14pt;")
        outer.addWidget(heading)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(12)
        column.addWidget(self._build_drive())
        column.addWidget(self._build_share())
        column.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.save_button = dialog_button("💾 Save", "save")
        self.save_button.clicked.connect(self.save)
        close = dialog_button("Close", "close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.save_button)
        buttons.addWidget(close)
        outer.addLayout(buttons)

    def _build_drive(self) -> QFrame:
        frame, layout = _section("☁️  Google Drive sync")
        self.drive_state = _label("", "status", wrap=True)
        layout.addWidget(self.drive_state)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.sign_in_button = _small_button("Sign in with Google",
                                            "Opens Google's sign-in page in your browser.")
        self.sign_in_button.clicked.connect(self._on_sign_in_clicked)
        self.sign_out_button = _small_button("Sign out", "Disconnects Google Drive and deletes the saved sign-in.")
        self.sign_out_button.clicked.connect(self._on_sign_out_clicked)
        row.addWidget(self.sign_in_button)
        row.addWidget(self.sign_out_button)
        row.addStretch()
        layout.addLayout(row)
        self.sign_in_reason = _label("", "warn", wrap=True)
        layout.addWidget(self.sign_in_reason)

        self.own_client = _create_styled_checkbox("Use my own Google OAuth client")
        self.own_client.setToolTip("A Desktop-type OAuth client you created in the Google Cloud console.")
        self.own_client.toggled.connect(self._update_drive_enabled_state)
        layout.addWidget(self.own_client)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(22, 0, 0, 0)
        self.client_id = QLineEdit()
        self.client_id.setPlaceholderText("…apps.googleusercontent.com")
        self.client_secret = QLineEdit()
        self.client_secret.setEchoMode(QLineEdit.Password)
        grid.addWidget(_label("Client ID"), 0, 0)
        grid.addWidget(self.client_id, 0, 1)
        grid.addWidget(_label("Client secret"), 1, 0)
        grid.addWidget(self.client_secret, 1, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.client_id.textChanged.connect(self._update_drive_enabled_state)
        self.client_secret.textChanged.connect(self._update_drive_enabled_state)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #2e2e48; background: #2e2e48;")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        self.drive_enabled = _create_styled_checkbox("Upload to Google Drive after each compile")
        layout.addWidget(self.drive_enabled)
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText(dsc.DEFAULT_FOLDER)
        self.folder.setToolTip("A folder name, or a path such as Books/Glossarion.")
        self.mode = _combo()
        for mode in dsc.MODES:
            self.mode.addItem(dsc.MODE_LABELS[mode], mode)
        self.mode.setItemData(0, "The compiled EPUB / PDF; compiling again replaces the same Drive file.",
                              Qt.ToolTipRole)
        self.mode.setItemData(1, "Every file of the book's output folder, keeping its folders (no zip).",
                              Qt.ToolTipRole)
        self.mode.setItemData(2, "As the full output folder, plus the raw source file in a RAW subfolder.",
                              Qt.ToolTipRole)
        form.addWidget(_label("Drive folder"), 0, 0)
        form.addWidget(self.folder, 0, 1)
        form.addWidget(_label(dsc.FOLDER_NOTE, "note", wrap=True), 1, 1)
        form.addWidget(_label("What to upload"), 2, 0)
        form.addWidget(self.mode, 2, 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)
        self.last_sync = _label("", "note", wrap=True)
        layout.addWidget(self.last_sync)
        return frame

    def _build_share(self) -> QFrame:
        frame, layout = _section("📤  Share via link")
        layout.addWidget(_label("Share… on a book card or in Book Details uploads the book's compiled file to "
                                "the site you choose and gives you a link. Turn on the sites you want to use.",
                                "note", wrap=True))
        self.provider_boxes = {}
        self.provider_reasons = {}
        for provider_id in library_cloud.DESKTOP_PROVIDERS:
            info = provider_info(provider_id)
            row = QHBoxLayout()
            row.setSpacing(8)
            box = _create_styled_checkbox(info.label)
            box.setToolTip(info.site)
            box.clicked.connect(lambda checked, pid=provider_id: self._on_provider_clicked(pid, checked))
            reason = _label("", "note")
            row.addWidget(box)
            row.addWidget(reason, 1)
            layout.addLayout(row)
            self.provider_boxes[provider_id] = box
            self.provider_reasons[provider_id] = reason
            if provider_id == "transferit":
                layout.addWidget(self._indent(_label(
                    "Share… opens transfer.it in your browser; you upload the file there and paste the link back.",
                    "note", wrap=True)))
            elif provider_id == "litterbox":
                litter_row = QHBoxLayout()
                litter_row.setSpacing(8)
                self.litterbox_time = _combo()
                for value, text, _seconds in LITTERBOX_TIME_CHOICES:
                    self.litterbox_time.addItem(text, value)
                litter_row.addWidget(_label("Expires after"))
                litter_row.addWidget(self.litterbox_time)
                litter_row.addWidget(_label("(cannot be deleted earlier)", "note"))
                litter_row.addStretch()
                layout.addWidget(self._indent(litter_row))
            elif provider_id == "pixeldrain":
                key_row = QHBoxLayout()
                key_row.setSpacing(8)
                self.pixeldrain_key = QLineEdit()
                self.pixeldrain_key.setEchoMode(QLineEdit.Password)
                self.pixeldrain_key.setPlaceholderText("pixeldrain API key")
                self.pixeldrain_clear = _small_button("Remove key")
                self.pixeldrain_clear.clicked.connect(self._on_pixeldrain_clear)
                key_row.addWidget(self.pixeldrain_key, 1)
                key_row.addWidget(self.pixeldrain_clear)
                layout.addWidget(self._indent(key_row))
            elif provider_id == "send":
                send_row = QHBoxLayout()
                send_row.setSpacing(8)
                self.send_expire = _combo()
                for seconds, text in SEND_EXPIRY_LABELS:
                    self.send_expire.addItem(text, seconds)
                self.send_downloads = _combo()
                for count in SEND_DOWNLOAD_CHOICES:
                    self.send_downloads.addItem(f"{count} download" + ("" if count == 1 else "s"), count)
                send_row.addWidget(_label("Expires after"))
                send_row.addWidget(self.send_expire)
                send_row.addWidget(_label("or"))
                send_row.addWidget(self.send_downloads)
                send_row.addStretch()
                layout.addWidget(self._indent(send_row))
        return frame

    @staticmethod
    def _indent(content: Any) -> QWidget:
        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(22, 0, 0, 2)
        if isinstance(content, QWidget):
            layout.addWidget(content)
        else:
            layout.addLayout(content)
        return holder

    # ---- state -----------------------------------------------------------------------------------------

    def load(self) -> None:
        store = self.cloud.store
        settings = store.drive_settings()
        own = store.own_client()
        self.own_client.setChecked(own["use_own"])
        self.client_id.setText(own["client_id"])
        self.client_secret.clear()
        self.client_secret.setPlaceholderText("Saved (leave empty to keep it)" if own["has_secret"] else
                                              "Client secret")
        self.drive_enabled.setChecked(settings.enabled)
        self.folder.setText(settings.folder)
        index = self.mode.findData(settings.mode)
        self.mode.setCurrentIndex(max(0, index))
        send = store.send_options()
        self._select_nearest(self.send_expire, send["expire"])
        self._select_nearest(self.send_downloads, send["downloads"])
        self.litterbox_time.setCurrentIndex(max(0, self.litterbox_time.findData(store.litterbox_options()["time"])))
        self.pixeldrain_key.clear()
        self.refresh()

    @staticmethod
    def _select_nearest(combo: QComboBox, value: int) -> None:
        values = [combo.itemData(i) for i in range(combo.count())]
        best = min(range(len(values)), key=lambda i: abs(int(values[i]) - int(value)))
        combo.setCurrentIndex(best)

    def refresh(self) -> None:
        """Signed-in state, last sync and the sites' states (the editable fields are left as typed)."""
        store = self.cloud.store
        settings = store.drive_settings()
        signing_in = self._sign_in is not None
        if signing_in:
            self.drive_state.setObjectName("status")
            self.drive_state.setText("Finish the sign-in in your browser…")
        elif settings.signed_in:
            self.drive_state.setObjectName("ok")
            self.drive_state.setText("● Connected to Google Drive")
        else:
            self.drive_state.setObjectName("status")
            self.drive_state.setText("Not connected. Nothing is sent to Google until you sign in and turn "
                                     "uploading on.")
        self.drive_state.style().unpolish(self.drive_state)
        self.drive_state.style().polish(self.drive_state)
        self.sign_out_button.setVisible(settings.signed_in and not signing_in)
        self.last_sync.setText(last_sync_text(store.last_sync()))
        self._update_drive_enabled_state()
        for state in store.provider_states():
            box = self.provider_boxes[state.id]
            # No blockSignals: the box's ✓ overlay follows stateChanged (a blocked reset left the tick drawn
            # on a site the user had just cancelled), and only ``clicked`` reaches _on_provider_clicked.
            box.setChecked(state.enabled)
            reason = "" if not state.enabled else ("Ready" if state.ready else state.reason)
            self.provider_reasons[state.id].setText(reason)
        has_key = store.has_pixeldrain_key()
        self.pixeldrain_key.setPlaceholderText("Saved (type a new key to replace it)" if has_key else
                                               "pixeldrain API key")
        self.pixeldrain_clear.setVisible(has_key)

    def _client_typed(self) -> bool:
        return bool(self.own_client.isChecked() and self.client_id.text().strip() and (
            self.client_secret.text().strip() or self.cloud.store.own_client()["has_secret"]))

    def _update_drive_enabled_state(self, *_args: Any) -> None:
        own = self.own_client.isChecked()
        self.client_id.setEnabled(own)
        self.client_secret.setEnabled(own)
        signing_in = self._sign_in is not None
        available, reason = self.cloud.sign_in_status()
        if not available and self._client_typed():
            available, reason = True, ""  # applied when Sign in is clicked
        signed_in = self.cloud.store.drive_settings().signed_in
        self.sign_in_button.setVisible(not signed_in or signing_in)
        self.sign_in_button.setText("Cancel sign-in" if signing_in else "Sign in with Google")
        self.sign_in_button.setEnabled(signing_in or available)
        self.sign_in_reason.setText("" if (available or signed_in) else reason)
        self.sign_in_reason.setVisible(bool(self.sign_in_reason.text()))

    def _on_cloud_changed(self, what: str) -> None:
        if what in ("drive", "sync", "links"):
            self.refresh()

    # ---- saving ----------------------------------------------------------------------------------------

    def _apply_client(self) -> None:
        secret = self.client_secret.text().strip()
        self.cloud.set_own_client(use_own=self.own_client.isChecked(), client_id=self.client_id.text().strip(),
                                  client_secret=secret or None)
        self.client_secret.clear()

    def save(self) -> bool:
        store = self.cloud.store
        try:
            self._apply_client()
            store.set_drive_options(enabled=self.drive_enabled.isChecked(),
                                    mode=self.mode.currentData(), folder=self.folder.text())
            store.set_send_options(expire=int(self.send_expire.currentData()),
                                   downloads=int(self.send_downloads.currentData()))
            store.set_litterbox_options(time=str(self.litterbox_time.currentData()))
            key = self.pixeldrain_key.text().strip()
            if key:
                store.set_pixeldrain_key(key)
        except ShareError as exc:
            QMessageBox.warning(self, "Library settings", exc.message)
            return False
        self.load()
        flash_saved(self.save_button)
        _emit_changed("share")
        return True

    def _on_pixeldrain_clear(self) -> None:
        self.cloud.store.clear_pixeldrain_key()
        self.refresh()
        _emit_changed("share")

    def _on_provider_clicked(self, provider_id: str, checked: bool) -> None:
        store = self.cloud.store
        try:
            if not checked:
                store.set_provider_enabled(provider_id, False)
            elif store.provider_state(provider_id).consented:
                store.set_provider_enabled(provider_id, True)
            else:
                sign_in = (provider_id == "gdrive" and not store.drive_settings().signed_in
                           and self._sign_in is None and self.sign_in_button.isEnabled())
                accepted = ask_consent(self, self.cloud, provider_id,
                                       confirm_label=TURN_ON_AND_SIGN_IN_LABEL if sign_in else TURN_ON_LABEL)
                if accepted and sign_in:
                    self._on_sign_in_clicked()  # opens Google's sign-in page in the browser
        except ShareError as exc:
            QMessageBox.warning(self, "Share via link", exc.message)
        self.refresh()
        _emit_changed("share")

    # ---- Google sign-in --------------------------------------------------------------------------------

    def _on_sign_in_clicked(self) -> None:
        if self._sign_in is not None:
            self._cancel_sign_in()
            return
        try:
            self._apply_client()
            sign_in = self.cloud.begin_sign_in(timeout=300)
        except ShareError as exc:
            QMessageBox.warning(self, "Google Drive", exc.message)
            self.refresh()
            return
        self._sign_in = sign_in
        self._sign_in_cancel = CancelToken()
        cancel = self._sign_in_cancel
        self._task = _Task(lambda task: self.cloud.finish_sign_in(sign_in, cancel=cancel), "drive-sign-in")
        self._task.done.connect(self._on_sign_in_done)
        self._task.start()
        open_in_browser(sign_in.auth_url, self)
        self.refresh()

    def _cancel_sign_in(self) -> None:
        if self._sign_in_cancel is not None:
            self._sign_in_cancel.cancel()
        if self._sign_in is not None:
            try:
                self._sign_in.cancel()
            except Exception:
                pass

    def _on_sign_in_done(self, _result: Any, error: Any) -> None:
        self._sign_in = None
        self._sign_in_cancel = None
        self._task = None
        if error is not None and not (isinstance(error, ShareError) and error.code == "cancelled"):
            QMessageBox.warning(self, "Google Drive", _error_text(error) or "The sign-in did not complete.")
        self.refresh()

    def _on_sign_out_clicked(self) -> None:
        answer = QMessageBox.question(
            self, "Google Drive", "Sign out of Google Drive? Uploads stop and the saved sign-in is deleted.")
        if answer != QMessageBox.Yes:
            return
        self.sign_out_button.setEnabled(False)
        self._task = _Task(lambda task: self.cloud.sign_out(), "drive-sign-out")
        self._task.done.connect(self._on_sign_out_done)
        self._task.start()

    def _on_sign_out_done(self, _result: Any, _error: Any) -> None:
        self._task = None
        self.sign_out_button.setEnabled(True)
        self.refresh()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        self._cancel_sign_in()
        try:
            cloud_signals().changed.disconnect(self._on_cloud_changed)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)

    def reject(self) -> None:  # Esc
        self.close()


# ---------------------------------------------------------------------------------------------------
# Share
# ---------------------------------------------------------------------------------------------------


def book_output_folder(book: dict) -> str:
    if book.get("type") == "in_progress":
        folder = book.get("output_folder") or book.get("path") or ""
    else:
        try:
            folder = _resolve_book_output_folder(book) or ""
        except Exception:
            folder = ""
    return folder if folder and os.path.isdir(folder) else ""


def shareable_files(book: dict) -> list:
    """The book's compiled files (translated file first, then compiled EPUB / PDF in the output folder)."""
    found = []
    try:
        translated = _resolve_book_translated_file(book)
    except Exception:
        translated = ""
    if translated:
        found.append(translated)
    found.extend(dsc.compiled_files_in(book_output_folder(book)))
    path = book.get("path") or ""
    if book.get("type") != "in_progress" and path:
        found.append(path)
    unique = []
    for candidate in found:
        if (candidate and os.path.isfile(candidate) and candidate.lower().endswith(library_cloud.ELIGIBLE_EXTENSIONS)
                and os.path.normcase(os.path.abspath(candidate)) not in
                [os.path.normcase(os.path.abspath(u)) for u in unique]):
            unique.append(candidate)
    return unique


def book_key(book: dict) -> str:
    """The output folder, or (no output folder: an organized / imported book on the shared Library
    shelf) the book's own file, so two such books never share their links."""
    files = shareable_files(book)
    return library_cloud.shelf_book_key(book, book_output_folder(book), files[0] if files else "")


def book_can_compile(book: dict) -> str:
    """"epub" / "pdf" when the book's workspace can be compiled, else ""."""
    folder = book_output_folder(book)
    if not folder or not os.path.isfile(os.path.join(folder, "translation_progress.json")):
        return ""
    try:
        return "pdf" if _workspace_compile_kind(book, folder) == "pdf" else "epub"
    except Exception:
        return "epub"


class ShareDialog(QDialog):
    """Share one book's compiled file by link (non-modal; the upload runs on a worker thread)."""

    def __init__(self, book: dict, parent: Any = None, *, cloud: Optional[library_cloud.LibraryCloud] = None,
                 compile_cb: Optional[Callable[[], Any]] = None,
                 open_settings: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(parent)
        self.book = book
        self.cloud = cloud or controller(parent)
        self.compile_cb = compile_cb
        self.open_settings = open_settings
        self.book_key = book_key(book)
        self._upload: Optional[_Task] = None
        self._cancel: Optional[CancelToken] = None
        self._plan: Any = None
        self._compiling_since = 0.0
        title = str(book.get("name") or book.get("title") or os.path.basename(book.get("path") or "") or "Book")
        self.setWindowTitle(f"Share · {title}")
        self.setMinimumWidth(620)
        self.setStyleSheet(CLOUD_STYLE)
        self._build(title)
        self._poll = QTimer(self)
        self._poll.setInterval(1500)
        self._poll.timeout.connect(self._poll_compile)
        self.reload()
        cloud_signals().changed.connect(self._on_cloud_changed)

    # ---- layout ----------------------------------------------------------------------------------------

    def _build(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        heading = _label("📤  Share via link", "sectionTitle")
        layout.addWidget(heading)
        name = _label(title, "status", wrap=True)
        layout.addWidget(name)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.file_combo = _combo()
        self.file_combo.setPlaceholderText("No compiled file")
        self.file_combo.currentIndexChanged.connect(lambda _i: self._update_buttons())
        grid.addWidget(_label("File"), 0, 0)
        grid.addWidget(self.file_combo, 0, 1)
        self.site_combo = _combo()
        self.site_combo.setPlaceholderText("No site turned on")
        self.site_combo.currentIndexChanged.connect(lambda _i: self._update_buttons())
        grid.addWidget(_label("Site"), 1, 0)
        grid.addWidget(self.site_combo, 1, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        # no compiled file yet
        self.compile_row = QWidget()
        compile_layout = QHBoxLayout(self.compile_row)
        compile_layout.setContentsMargins(0, 0, 0, 0)
        self.compile_note = _label("This book has no compiled file yet.", "warn", wrap=True)
        self.compile_button = _small_button("📘  Compile EPUB")
        self.compile_button.clicked.connect(self._on_compile_clicked)
        compile_layout.addWidget(self.compile_note, 1)
        compile_layout.addWidget(self.compile_button)
        layout.addWidget(self.compile_row)

        # no site turned on
        self.no_site_row = QWidget()
        no_site = QHBoxLayout(self.no_site_row)
        no_site.setContentsMargins(0, 0, 0, 0)
        no_site.addWidget(_label("No sharing site is turned on. Turn one on in Library settings.", "warn",
                                 wrap=True), 1)
        self.settings_button = _small_button("⚙️  Library settings")
        self.settings_button.clicked.connect(self._on_open_settings)
        no_site.addWidget(self.settings_button)
        layout.addWidget(self.no_site_row)

        self.site_reason = _label("", "warn", wrap=True)
        layout.addWidget(self.site_reason)

        actions = QHBoxLayout()
        self.upload_button = dialog_button("Upload and get link", "save")
        self.upload_button.setStyleSheet(self.upload_button.styleSheet() + DISABLED_BUTTON_RULE)
        self.upload_button.clicked.connect(self._on_upload_clicked)
        self.cancel_button = dialog_button("Cancel upload", "reset")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.hide()
        actions.addWidget(self.upload_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.progress, 1)
        actions.addStretch(0)
        layout.addLayout(actions)
        self.status = _label("", "note", wrap=True)
        layout.addWidget(self.status)

        # the new link
        self.link_row = QWidget()
        link_layout = QHBoxLayout(self.link_row)
        link_layout.setContentsMargins(0, 0, 0, 0)
        self.link = QLineEdit()
        self.link.setReadOnly(True)
        self.copy_button = _small_button("📋  Copy")
        self.copy_button.clicked.connect(lambda: self._copy(self.link.text(), self.copy_button))
        open_button = _small_button("Open", "Open the link in your browser")
        open_button.clicked.connect(lambda: open_in_browser(self.link.text(), self))
        link_layout.addWidget(self.link, 1)
        link_layout.addWidget(self.copy_button)
        link_layout.addWidget(open_button)
        layout.addWidget(self.link_row)
        self.link_row.hide()

        # transfer.it handoff
        self.handoff = QWidget()
        handoff = QVBoxLayout(self.handoff)
        handoff.setContentsMargins(0, 0, 0, 0)
        handoff.setSpacing(6)
        self.handoff_hint = _label("", "status", wrap=True)
        handoff.addWidget(self.handoff_hint)
        paste_row = QHBoxLayout()
        self.folder_button = _small_button("📂  Show the file")
        self.folder_button.clicked.connect(self._on_show_file)
        self.paste = QLineEdit()
        self.paste.setPlaceholderText("Paste the transfer.it link here")
        self.save_link_button = _small_button("Save link")
        self.save_link_button.clicked.connect(self._on_save_pasted)
        paste_row.addWidget(self.folder_button)
        paste_row.addWidget(self.paste, 1)
        paste_row.addWidget(self.save_link_button)
        handoff.addLayout(paste_row)
        layout.addWidget(self.handoff)
        self.handoff.hide()

        layout.addWidget(_label("Links of this book", "sectionTitle"))
        self.links_box = QVBoxLayout()
        self.links_box.setSpacing(4)
        layout.addLayout(self.links_box)

        bottom = QHBoxLayout()
        bottom.addStretch()
        close = dialog_button("Close", "close")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        layout.addLayout(bottom)

    # ---- state -----------------------------------------------------------------------------------------

    def reload(self) -> None:
        current_file = self.file_combo.currentData()
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for path in shareable_files(self.book):
            try:
                size = format_size(os.path.getsize(path))
            except OSError:
                size = ""
            self.file_combo.addItem(f"{os.path.basename(path)}  ({size})" if size else os.path.basename(path), path)
        index = self.file_combo.findData(current_file) if current_file else -1
        self.file_combo.setCurrentIndex(max(0, index))
        self.file_combo.blockSignals(False)

        current_site = self.site_combo.currentData()
        self.site_combo.blockSignals(True)
        self.site_combo.clear()
        for state in self.cloud.store.provider_states():
            if state.enabled:
                self.site_combo.addItem(state.label, state.id)
        index = self.site_combo.findData(current_site) if current_site else -1
        if index < 0:  # the first ready site
            ready = [i for i in range(self.site_combo.count())
                     if self.cloud.store.provider_state(self.site_combo.itemData(i)).ready]
            index = ready[0] if ready else 0
        self.site_combo.setCurrentIndex(index)
        self.site_combo.blockSignals(False)
        self._refresh_links()
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_file = self.file_combo.count() > 0
        has_site = self.site_combo.count() > 0
        busy = self._upload is not None
        kind = book_can_compile(self.book)
        self.compile_row.setVisible(not has_file)
        compiling = bool(self._compiling_since)
        self.compile_button.setVisible(bool(kind and self.compile_cb) and not compiling)
        self.compile_button.setText("📄  Compile PDF" if kind == "pdf" else "📘  Compile EPUB")
        if compiling:
            self.compile_note.setText("Compiling… the file appears here when it is ready.")
        elif kind and self.compile_cb:
            self.compile_note.setText("This book has no compiled file yet. Compile it first.")
        else:
            self.compile_note.setText("This book has no compiled file to share.")
        self.file_combo.setEnabled(has_file and not busy)
        self.no_site_row.setVisible(not has_site)
        self.site_combo.setEnabled(has_site and not busy)
        site = self.site_combo.currentData()
        reason = ""
        ready = False
        if site:
            state = self.cloud.store.provider_state(site)
            ready, reason = state.ready, state.reason
        self.site_reason.setText(reason if site and not ready else "")
        self.site_reason.setVisible(bool(self.site_reason.text()))
        self.upload_button.setText("Open transfer.it" if site == "transferit" else "Upload and get link")
        self.upload_button.setEnabled(has_file and ready and not busy)
        self.upload_button.setVisible(not busy)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        if site != "transferit":
            self.handoff.hide()

    def _refresh_links(self) -> None:
        while self.links_box.count():
            item = self.links_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        records = self.cloud.links_for(self.book_key)
        if not records:
            self.links_box.addWidget(_label("No links yet.", "note"))
            return
        for record in records[:20]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            try:
                site = provider_info(record.get("provider")).label
            except ShareError:
                site = str(record.get("provider") or "")
            text = f"{site} · {record.get('name') or ''} · {_ago(record.get('created'))}"
            label = _label(text, "status")
            label.setToolTip(record.get("url") or "")
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            copy = _small_button("Copy")
            copy.setEnabled(bool(record.get("url")))
            copy.clicked.connect(lambda _c=False, url=record.get("url") or "", b=copy: self._copy(url, b))
            delete = _small_button("Delete", "Delete the upload at the site (when it allows it) and forget the "
                                             "link." if record.get("can_delete") else "Forget the link.")
            delete.clicked.connect(lambda _c=False, rid=record["id"]: self._on_delete_link(rid))
            row_layout.addWidget(label, 1)
            row_layout.addWidget(copy)
            row_layout.addWidget(delete)
            self.links_box.addWidget(row)

    def _on_cloud_changed(self, what: str) -> None:
        if what == "links":
            self._refresh_links()
        elif what in ("drive", "share"):
            if self._upload is None:
                self.reload()
            else:
                self._update_buttons()

    @staticmethod
    def _copy(text: str, button: Optional[QPushButton] = None) -> None:
        if not text:
            return
        QApplication.clipboard().setText(text)
        if button is not None:
            flash_saved(button, "✅ Copied", ms=1200, beep=False)

    # ---- compile first -----------------------------------------------------------------------------------

    def _on_compile_clicked(self) -> None:
        if self.compile_cb is None:
            return
        try:
            self.compile_cb()
        except Exception as exc:
            QMessageBox.warning(self, "Share", f"Could not start the compile:\n{exc}")
            return
        self._compiling_since = time.monotonic()
        self._poll.start()
        self._update_buttons()

    def _poll_compile(self) -> None:
        if shareable_files(self.book):
            self._poll.stop()
            self._compiling_since = 0.0
            self.reload()
        elif time.monotonic() - self._compiling_since > 3600:
            self._poll.stop()
            self._compiling_since = 0.0
            self._update_buttons()

    def _on_open_settings(self) -> None:
        if self.open_settings is not None:
            self.open_settings()
        else:
            open_library_settings(self.parent() or self)

    # ---- upload ------------------------------------------------------------------------------------------

    def _on_upload_clicked(self) -> None:
        path = self.file_combo.currentData()
        site = self.site_combo.currentData()
        if not path or not site:
            return
        if site == "transferit":
            self._start_handoff(path)
            return
        self.link_row.hide()
        self._cancel = CancelToken()
        cancel = self._cancel
        key = self.book_key

        def work(task: _Task) -> Any:
            return self.cloud.share_upload(site, path, book_key=key, progress=task.report, cancel=cancel)

        self._upload = _Task(work, "share-upload")
        self._upload.progress.connect(self._on_progress)
        self._upload.done.connect(self._on_upload_done)
        self.progress.setValue(0)
        self.status.setText(f"Uploading to {provider_info(site).label}…")
        self._upload.start()
        self._update_buttons()

    def _on_progress(self, sent: Any, total: Any) -> None:
        try:
            self.progress.setValue(int(100 * int(sent) / max(1, int(total))))
        except (TypeError, ValueError):
            pass

    def _on_cancel_clicked(self) -> None:
        if self._cancel is not None:
            self._cancel.cancel()
            self.status.setText("Stopping…")

    def _on_upload_done(self, record: Any, error: Any) -> None:
        self._upload = None
        self._cancel = None
        if error is not None:
            if isinstance(error, ShareError) and error.code == "cancelled":
                self.status.setText("Upload cancelled.")
            else:
                self.status.setText("⚠️ " + (_error_text(error) or "The upload failed."))
        elif isinstance(record, dict):
            self.show_link(record.get("url") or "")
            self.status.setText("Anyone with the link can download the file.")
        self._update_buttons()
        self._refresh_links()

    def show_link(self, url: str) -> None:
        self.link.setText(url)
        self.link.setCursorPosition(0)
        self.link_row.setVisible(bool(url))

    # ---- transfer.it -------------------------------------------------------------------------------------

    def _start_handoff(self, path: str) -> None:
        try:
            self._plan = self.cloud.transferit_plan(path)
        except Exception as exc:  # shown, never a silent click
            self.status.setText("⚠️ " + (_error_text(exc) or "Could not prepare transfer.it."))
            return
        self._handoff_path = path
        self.handoff_hint.setText(self._plan.hint)
        self.handoff.show()
        open_in_browser(self._plan.url, self)

    def _on_show_file(self) -> None:
        path = getattr(self, "_handoff_path", "") or self.file_combo.currentData() or ""
        if path:
            _open_folder_in_explorer(path)

    def _on_save_pasted(self) -> None:
        path = getattr(self, "_handoff_path", "") or self.file_combo.currentData() or ""
        try:
            record = self.cloud.add_pasted_link(path, self.paste.text(), book_key=self.book_key)
        except ShareError as exc:
            self.status.setText("⚠️ " + exc.message)
            return
        self.paste.clear()
        self.handoff.hide()
        self.show_link(record.get("url") or "")
        self.status.setText("Link saved.")
        self._refresh_links()

    # ---- links -------------------------------------------------------------------------------------------

    def _on_delete_link(self, link_id: str) -> None:
        answer = QMessageBox.question(self, "Share", "Delete this link? Where the site allows it, the upload "
                                                     "is deleted there too.")
        if answer != QMessageBox.Yes:
            return
        task = _Task(lambda _t: self.cloud.delete_link(link_id), "share-delete")
        task.done.connect(self._on_deleted)
        task.start()

    def _on_deleted(self, _deleted: Any, error: Any) -> None:
        if error is not None:
            self.status.setText("⚠️ " + (_error_text(error) or "The link could not be deleted."))
        self._refresh_links()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt
        if self._cancel is not None:  # closing the window stops its upload
            self._cancel.cancel()
        self._poll.stop()
        try:
            cloud_signals().changed.disconnect(self._on_cloud_changed)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)

    def reject(self) -> None:  # Esc
        self.close()


# ---------------------------------------------------------------------------------------------------
# entry points used by epub_library
# ---------------------------------------------------------------------------------------------------


def _show(dialog: QDialog, attr: str, parent: Any) -> QDialog:
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    if parent is not None:
        setattr(parent, attr, dialog)  # keep it alive while open
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


def open_library_settings(parent: Any = None) -> QDialog:
    existing = getattr(parent, "_library_settings_dialog", None) if parent is not None else None
    try:
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing
    except RuntimeError:
        pass
    return _show(LibrarySettingsDialog(parent), "_library_settings_dialog", parent)


def open_share_dialog(parent: Any, book: dict, compile_cb: Optional[Callable[[], Any]] = None) -> QDialog:
    return _show(ShareDialog(book, parent, compile_cb=compile_cb), "_share_dialog", parent)


def drive_sync_on(widget: Any = None) -> bool:
    """Whether "Sync to Drive now" is offered: Drive sync switched on and signed in."""
    try:
        return controller(widget).store.drive_settings().active
    except Exception:
        return False


def sync_book_now(parent: Any, book: dict) -> bool:
    """Queue one book for Drive sync (mode from Library settings); progress goes to the main log."""
    cloud = controller(parent)
    if cloud.sync_book(book):
        QToolTip.showText(QCursor.pos(), "☁️ Sending to Google Drive… (progress is in the main log)")
        return True
    QMessageBox.information(parent, "Google Drive",
                            "Sign in to Google Drive in Library settings first, and check that this book has an "
                            "output folder.")
    return False
