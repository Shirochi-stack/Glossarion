"""Interactive Images: the desktop (PySide6) layer over the shared cores.

- ``add_interactive_styles_rows``: the rows under Attach CSS in Other Settings (EPUB section).
- ``InteractiveImagesDialog``: profiles, images per chapter, prompts, models (+ the Image keys
  preview), max reference images, run options and the per-book reference images.
- ``InteractiveStylesDialog``: the 22 built-in CSS presets and the user's own, with a live preview
  (Qt WebEngine when the build has it, else a QTextBrowser).
- ``PlaceholderTextDialog``: the editable texts the placeholders expand to (right-click a prompt editor >
  "Edit placeholder text…").
- ``InteractiveImagesGalleryDialog``: the book's images as cards with multi-select delete.
- ``build_images_page``: the Progress Manager's Interactive Images tab.

All logic lives in ``interactive_images_core`` / ``interactive_images_progress_core`` /
``interactive_images_runner`` / ``interactive_css_presets``; this module only shows it. It never
imports ``epub_library`` (Lite builds exclude it and Qt WebEngine).
"""

import os
import threading
import time
import traceback

from PySide6.QtCore import (QEvent, QFileSystemWatcher, QItemSelectionModel, QObject, QRect, QSize, Qt, QTimer,
                            Signal)
from PySide6.QtGui import QColor, QFont, QIcon, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QFileDialog, QFrame, QGroupBox,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListView, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QStackedWidget, QStyle, QStyledItemDelegate,
                               QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

import interactive_css_presets as presets
import interactive_images_core as core
import interactive_images_progress_core as ipc

# Qt WebEngine for the style preview (animations, hover, <details>); Lite builds have none.
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    _HAS_WEBENGINE = True
except ImportError:
    _HAS_WEBENGINE = False

FALLBACK_NOTE = "Animations, hover and collapsibles need Qt WebEngine (not in Lite builds)"

#: The dialogs' own sheet (no QSpinBox / QLineEdit rule: the spin box stays native).
DIALOG_STYLE = """
QDialog { background-color: #1e1e1e; color: white; }
QLabel { color: #e2e8f0; }
QGroupBox { color: #5a9fd4; border: 1px solid #3a4555; border-radius: 4px; margin-top: 10px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #5a9fd4; }
QPlainTextEdit, QTextBrowser { background-color: #2d2d2d; color: white; border: 1px solid #4a5568;
    border-radius: 3px; padding: 4px; }
QPushButton { background-color: #3d3d3d; color: white; border: 1px solid #4a5568; border-radius: 3px;
    padding: 5px 10px; }
QPushButton:hover { background-color: #4d4d4d; border-color: #5a9fd4; }
QPushButton:disabled { color: #777777; }
QComboBox { background-color: #2d2d2d; color: white; border: 1px solid #4a5568; border-radius: 3px; padding: 4px; }
QComboBox QAbstractItemView { background-color: #2d2d2d; color: white; selection-background-color: #5a9fd4; }
QTabWidget::pane { border: 1px solid #3a4555; }
QTabBar::tab { background: #2d2d2d; color: #cbd5e1; padding: 5px 12px; border: 1px solid #3a4555; }
QTabBar::tab:selected { background: #3a4555; color: white; }
QCheckBox { color: white; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #5a9fd4; border-radius: 2px;
    background-color: #2d2d2d; }
QCheckBox::indicator:checked { background-color: #5a9fd4; border-color: #5a9fd4; }
QCheckBox::indicator:hover { border-color: #7bb3e0; }
QTableWidget, QListWidget { background-color: #252526; color: white; border: 1px solid #3a4555; }
QHeaderView::section { background-color: #2d2d2d; color: #cbd5e1; border: 1px solid #3a4555; padding: 3px; }
"""

_LINE_STYLE = ("QLineEdit { background-color: #2d2d2d; color: white; border: 1px solid #4a5568; "
               "border-radius: 3px; padding: 4px; }")


def _house_combo(combo, extra=""):
    """The app's dropdown (Halgakos arrow, mouse wheel off): ``qt_widget_helpers.style_combo``."""
    from qt_widget_helpers import style_combo
    style_combo(combo, extra)
    return combo


def _line(text='', placeholder=''):
    edit = QLineEdit(text)
    edit.setStyleSheet(_LINE_STYLE)
    if placeholder:
        edit.setPlaceholderText(placeholder)
    return edit


def _button(text, kind='default', tooltip=''):
    colors = {
        'primary': ('#2f5f8f', '#4a78a8', '#3f78ad'),
        'success': ('#2d6a4f', '#40916c', '#40916c'),
        'danger': ('#6f2f2f', '#944545', '#8a3a3a'),
        'accent': ('#5b3a8a', '#7a52b3', '#6d47a3'),
        'info': ('#17a2b8', '#138496', '#138496'),
        'default': ('#3d3d3d', '#4a5568', '#4d4d4d'),
    }
    background, border, hover = colors.get(kind, colors['default'])
    button = QPushButton(text)
    button.setAutoDefault(False)
    button.setCursor(Qt.PointingHandCursor)
    button.setStyleSheet(
        f"QPushButton {{ background-color: {background}; color: white; border: 1px solid {border}; "
        f"border-radius: 4px; padding: 4px 10px; font-weight: bold; }} "
        f"QPushButton:hover {{ background-color: {hover}; }} QPushButton:disabled {{ color: #888888; }}")
    if tooltip:
        button.setToolTip(tooltip)
    return button


def _create_styled_checkbox(owner, text):
    """Other Settings' styled checkbox (the same name, so the settings schema generator reads its label)."""
    try:
        from other_settings import _create_styled_checkbox
        return _create_styled_checkbox(owner, text)
    except Exception:
        from PySide6.QtWidgets import QCheckBox
        return QCheckBox(text)


def _config(owner):
    config = getattr(owner, 'config', None)
    if not isinstance(config, dict):
        config = {}
        try:
            owner.config = config
        except Exception:
            pass
    return config


def _persist(owner):
    try:
        result = owner.save_config(show_message=False)
        return result is not False
    except Exception:
        return False


def _log(owner, message):
    try:
        owner.append_log(message)
    except Exception:
        print(message)


def _stay_above(window, owner_widget):
    """A parentless window kept above ``owner_widget`` (its spin box must not inherit an ancestor sheet)."""
    if owner_widget is None:
        return
    try:
        window.winId()
        owner_widget.window().winId()
        handle = window.windowHandle()
        parent_handle = owner_widget.window().windowHandle()
        if handle is not None and parent_handle is not None:
            handle.setTransientParent(parent_handle)
    except Exception:
        pass


def toggle_window(owner, attr, factory):
    """Show the window stored as ``owner.<attr>`` (made by ``factory()``), or close it when it is open:
    the button that opens it closes it on the second click."""
    window = getattr(owner, attr, None)
    try:
        if window is not None and window.isVisible():
            window.close()
            return None
    except RuntimeError:
        window = None
    if window is None:
        window = factory()
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda *_a, o=owner, a=attr: setattr(o, a, None) if getattr(o, a, None) is window
                                 else None)
        setattr(owner, attr, window)
    window.show()
    window.raise_()
    window.activateWindow()
    return window


PLACEHOLDER_TEXT_MENU_ACTION = "Edit placeholder text…"


def _placeholder_menu(editor, names, owner=None):
    """Right-click "Available placeholders" on a prompt editor (inserts the clicked one), plus -- with an
    ``owner`` -- "Edit placeholder text…" (the Glossary Manager's wording): the editable texts the step expands
    placeholders to (gender, image subjects, reference lines ...)."""
    def show_menu(pos):
        menu = editor.createStandardContextMenu()
        menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: white; } "
                           "QMenu::item:selected { background-color: #17a2b8; }")
        if names:
            sub = menu.addMenu("Available placeholders")
            for name in names:
                action = sub.addAction("{" + name + "}")
                action.triggered.connect(lambda _c=False, n=name: editor.insertPlainText("{" + n + "}"))
        if owner is not None:
            menu.addSeparator()
            edit = menu.addAction(PLACEHOLDER_TEXT_MENU_ACTION)
            edit.triggered.connect(lambda _c=False: open_placeholder_dialog(owner, editor.window()))
        menu.exec(editor.mapToGlobal(pos))
        menu.deleteLater()
    editor.setContextMenuPolicy(Qt.CustomContextMenu)
    editor.customContextMenuRequested.connect(show_menu)


def open_path(path, make=False):
    """Open a file or folder with the system's default application."""
    if make:
        os.makedirs(path, exist_ok=True)
    if not os.path.exists(path):
        return False
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
    except Exception:
        return False


def _model_names():
    try:
        from model_options import get_model_options
        return list(get_model_options())
    except Exception:
        return []


def _model_combo(combo, placeholder):
    """An editable model dropdown (the model list, house style); the caller sets its text."""
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    _house_combo(combo)
    combo.addItems(_model_names())
    combo.lineEdit().setPlaceholderText(placeholder)
    combo.setMinimumWidth(320)
    return combo


def _book_output_dirs(owner, extra=None):
    """``[(label, output_dir)]`` for the books the user has open (selected files) plus ``extra``."""
    found = []

    def add(label, folder):
        if folder and os.path.isdir(folder) and all(os.path.normcase(folder) != os.path.normcase(f) for _l, f in found):
            found.append((label, folder))

    for label, folder in extra or ():
        add(label, folder)
    sources = []
    for attr in ('selected_files',):
        value = getattr(owner, attr, None)
        if isinstance(value, (list, tuple)):
            sources.extend(str(v) for v in value if v)
    last = _config(owner).get('last_epub_path')
    if last:
        sources.append(str(last))
    try:
        from progress_core import resolve_output_dir
    except Exception:
        resolve_output_dir = None
    for source in sources:
        if resolve_output_dir is None:
            break
        try:
            folder = resolve_output_dir(owner, source)
        except Exception:
            folder = None
        add(os.path.splitext(os.path.basename(source))[0], folder)
    return found


# =============================================================================================
# Style preview (Qt WebEngine, else QTextBrowser)
# =============================================================================================

class StylePreview(QWidget):
    """A live preview of the sample chapter under a stylesheet."""

    def __init__(self, parent=None, *, force_fallback=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.uses_webengine = bool(_HAS_WEBENGINE and not force_fallback)
        self._css = None
        if self.uses_webengine:
            self.view = QWebEngineView(self)
            try:
                self.view.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, False)
            except Exception:
                pass
            self.note = None
        else:
            self.view = QTextBrowser(self)
            self.view.setOpenLinks(False)
            self.note = QLabel(FALLBACK_NOTE)
            self.note.setStyleSheet("color: #f6ad55; font-size: 9pt;")
            layout.addWidget(self.note)
        layout.addWidget(self.view, 1)

    def set_css(self, css):
        if css == self._css:
            return
        self._css = css
        document = presets.preview_document(css or '')
        if self.uses_webengine:
            self.view.setHtml(document)
        else:
            self.view.setHtml(document)


# =============================================================================================
# Styles dialog
# =============================================================================================

class InteractiveStylesDialog(QDialog):
    """The CSS presets: pick one, duplicate a built-in, write one from a prompt, edit, save, delete, apply to a book."""

    preset_changed = Signal(str)

    def __init__(self, owner, parent=None, *, output_dir=None, force_fallback=False):
        super().__init__(parent)
        self.owner = owner
        self.output_dir = output_dir
        self.setWindowTitle("Interactive Styles")
        self.setStyleSheet(DIALOG_STYLE)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.62), int(screen.height() * 0.72))
        layout = QVBoxLayout(self)

        intro = QLabel("CSS-only styles for the translated chapters (EPUB readers strip scripts). Every rule is scoped "
                       "under a .glx- class, so a style only touches what Interactive Images marked. The book's "
                       "stylesheet is glossarion_interactive.css, linked after your Attach CSS file.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #a0aec0;")
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Style:"))
        self.combo = _house_combo(QComboBox())
        self.combo.setMinimumWidth(260)
        row.addWidget(self.combo)
        self.duplicate_btn = _button("⧉ Duplicate", 'default', "Make an editable copy of this style")
        self.prompt_btn = _button("✨ New from prompt…", 'accent', "Describe a style; the text model writes the CSS")
        self.edit_btn = _button("✏ Edit CSS", 'default', "Show or hide the CSS editor")
        self.save_btn = _button("💾 Save", 'primary', "Save this style's CSS")
        self.delete_btn = _button("🗑 Delete", 'danger', "Delete this user style")
        self.apply_btn = _button("📌 Apply to book…", 'success', "Write this style as a book's glossarion_interactive.css")
        for button in (self.duplicate_btn, self.prompt_btn, self.edit_btn, self.save_btn, self.delete_btn,
                       self.apply_btn):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color: #f6ad55; font-size: 9pt;")
        self.warnings.hide()
        layout.addWidget(self.warnings)

        splitter = QSplitter(Qt.Horizontal)
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont('Consolas', 9))
        self.editor.hide()
        splitter.addWidget(self.editor)
        self.preview = StylePreview(force_fallback=force_fallback)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(lambda: self.preview.set_css(self.editor.toPlainText()))
        self.editor.textChanged.connect(self._debounce.start)

        self.combo.currentTextChanged.connect(self._on_select)
        self.duplicate_btn.clicked.connect(self._duplicate)
        self.prompt_btn.clicked.connect(self._new_from_prompt)
        self.edit_btn.clicked.connect(lambda: self.editor.setVisible(not self.editor.isVisible()))
        self.save_btn.clicked.connect(self._save)
        self.delete_btn.clicked.connect(self._delete)
        self.apply_btn.clicked.connect(self._apply_to_book)
        self._fill(presets.active_preset_name(_config(owner)))

    # ---- helpers --------------------------------------------------------------------------------
    def _fill(self, selected):
        blocked = self.combo.blockSignals(True)
        try:
            self.combo.clear()
            self.combo.addItems(presets.preset_names(_config(self.owner)))
            index = self.combo.findText(selected)
            self.combo.setCurrentIndex(max(0, index))
        finally:
            self.combo.blockSignals(blocked)
        self._on_select(self.combo.currentText())

    def current_name(self):
        return self.combo.currentText()

    def _on_select(self, name):
        config = _config(self.owner)
        css = presets.preset_css(config, name) or ''
        builtin = presets.is_builtin_preset(name)
        blocked = self.editor.blockSignals(True)
        self.editor.setPlainText(css)
        self.editor.blockSignals(blocked)
        self.editor.setReadOnly(builtin)
        self.save_btn.setEnabled(not builtin)
        self.delete_btn.setEnabled(not builtin)
        self.preview.set_css(css)
        self._show_warnings([])
        if name and config.get(presets.PRESET_KEY) != name:
            config[presets.PRESET_KEY] = name
            _persist(self.owner)
        self.preset_changed.emit(name)

    def _show_warnings(self, warnings):
        if warnings:
            self.warnings.setText("Adjusted: " + "; ".join(warnings[:6]) + (" …" if len(warnings) > 6 else ""))
            self.warnings.show()
        else:
            self.warnings.hide()

    def _store(self, name, css):
        config = _config(self.owner)
        name, warnings = presets.save_user_preset(config, name, css)
        config[presets.PRESET_KEY] = name
        _persist(self.owner)
        self._fill(name)
        self._show_warnings(warnings)
        return name

    # ---- actions --------------------------------------------------------------------------------
    def _duplicate(self):
        config = _config(self.owner)
        base = self.current_name()
        name = presets.duplicate_name(config, base)
        self._store(name, self.editor.toPlainText() or presets.preset_css(config, base) or '')
        self.editor.show()

    def _save(self):
        name = self.current_name()
        if presets.is_builtin_preset(name):
            return
        self._store(name, self.editor.toPlainText())
        try:
            from qt_widget_helpers import flash_saved
            flash_saved(self.save_btn)
        except Exception:
            pass

    def _delete(self):
        name = self.current_name()
        if presets.is_builtin_preset(name):
            return
        try:
            from review_dialog import confirm_prompt_profile_action
            if not confirm_prompt_profile_action(self, "Delete Style", f"Delete the style '{name}'?"):
                return
        except Exception:
            pass
        presets.delete_user_preset(_config(self.owner), name)
        _persist(self.owner)
        self._fill(presets.active_preset_name(_config(self.owner)))

    def _new_from_prompt(self):
        request, ok = QInputDialog.getMultiLineText(
            self, "New Style from Prompt", "Describe the style you want (it becomes {user_request} in the CSS prompt):")
        if not ok or not str(request).strip():
            return
        from interactive_images_runner import generate_stylesheet
        try:
            from qt_widget_helpers import run_with_progress
            css, warnings = run_with_progress(self, "Writing the style", lambda: generate_stylesheet(
                _config(self.owner), request), text="Asking the text model for a stylesheet…")
        except Exception as exc:
            QMessageBox.warning(self, "New Style", f"The text model did not return a stylesheet:\n{exc}")
            return
        if not str(css or '').strip():
            QMessageBox.warning(self, "New Style", "The text model returned no CSS.")
            return
        name = presets.preset_name_for_request(_config(self.owner), request)
        self._store(name, css)
        self._show_warnings(warnings)
        self.editor.show()

    def _apply_to_book(self):
        folder = self.output_dir
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Select the book's output folder")
        if not folder:
            return
        css = self.editor.toPlainText() if not presets.is_builtin_preset(self.current_name()) else \
            presets.preset_css(_config(self.owner), self.current_name())
        path, warnings = presets.write_book_stylesheet(folder, css or '')
        self._show_warnings(warnings)
        _log(self.owner, f"🎨 Interactive styles '{self.current_name()}' written to {path}")
        QMessageBox.information(self, "Interactive Styles",
                                f"Wrote {os.path.basename(path)} for this book.\nIt is linked when the EPUB is compiled.")


# =============================================================================================
# Placeholder texts (right-click a prompt > "Edit placeholder text...") and the Auto ratio picker
# =============================================================================================

class PlaceholderTextDialog(QDialog):
    """Pick a placeholder, edit the text it expands to, Save / Reset. The texts the step puts into a request
    (gender wording, image subjects, reference lines) are all here, never in code; an empty text expands to
    nothing."""

    def __init__(self, owner, parent=None, *, select=None):
        super().__init__(parent)
        self.owner = owner
        self.setWindowTitle("Placeholder Text")
        self.setObjectName("interactive_placeholder_text_dialog")
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(720, 480)
        self.items = core.placeholder_text_defs()
        self.defaults = {item['name']: item['default'] for item in self.items}
        self.notes = {item['name']: item['note'] for item in self.items}
        self.drafts = {}
        self.name = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Placeholder:"))
        self.combo = _house_combo(QComboBox())
        self.combo.setMinimumWidth(280)
        for item in self.items:
            self.combo.addItem("{" + item['name'] + "}", item['name'])
        row.addWidget(self.combo)
        row.addStretch(1)
        layout.addLayout(row)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
        layout.addWidget(self.note)
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont('Consolas', 9))
        layout.addWidget(self.editor, 1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #9aa0a6; font-size: 9pt;")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.save_btn = _button("💾 Save", 'primary', "Save this placeholder's text")
        self.reset_btn = _button("Reset to Default", 'default')
        self.reset_all_btn = _button("Reset All to Defaults", 'danger')
        close_btn = _button("Close")
        for button in (self.save_btn, self.reset_btn, self.reset_all_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        self.combo.currentIndexChanged.connect(self._on_select)
        self.editor.textChanged.connect(self._refresh_status)
        self.save_btn.clicked.connect(self.save)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_all_btn.clicked.connect(self.reset_all)
        close_btn.clicked.connect(self.close)
        index = self.combo.findData(select) if select else 0
        self.combo.setCurrentIndex(max(0, index))
        self._on_select(self.combo.currentIndex())

    def effective(self, name):
        return core.placeholder_text(_config(self.owner), name)

    def _remember(self):
        if self.name is None:
            return
        text = self.editor.toPlainText()
        if text != self.effective(self.name):
            self.drafts[self.name] = text
        else:
            self.drafts.pop(self.name, None)

    def _on_select(self, index):
        if index < 0:
            return
        self._remember()
        self.name = self.combo.itemData(index)
        blocked = self.editor.blockSignals(True)
        try:
            self.editor.setPlainText(self.drafts.get(self.name, self.effective(self.name)))
        finally:
            self.editor.blockSignals(blocked)
        self.note.setText(self.notes[self.name])
        self._refresh_status()

    def _refresh_status(self):
        if self.name is None:
            return
        text = self.editor.toPlainText()
        label = "Edited" if self.effective(self.name) != self.defaults[self.name] else "Default text"
        if text != self.effective(self.name):
            label += " (unsaved changes)"
        self.status.setText(label)

    def save(self):
        if self.name is None:
            return
        core.set_placeholder_text(_config(self.owner), self.name, self.editor.toPlainText())
        _persist(self.owner)
        self.drafts.pop(self.name, None)
        self._refresh_status()
        try:
            from qt_widget_helpers import flash_saved
            flash_saved(self.save_btn)
        except Exception:
            pass

    def reset(self):
        if self.name is None:
            return
        self.editor.setPlainText(self.defaults[self.name])

    def reset_all(self):
        try:
            from review_dialog import confirm_prompt_profile_action
            if not confirm_prompt_profile_action(self, "Reset Placeholder Texts",
                                                 "Reset every placeholder text to its default?"):
                return
        except Exception:
            pass
        config = _config(self.owner)
        for name, default in self.defaults.items():
            core.set_placeholder_text(config, name, default)
        _persist(self.owner)
        self.drafts.clear()
        self._on_select(self.combo.currentIndex())


def open_placeholder_dialog(owner, anchor=None, select=None):
    """The placeholder text window (toggles: a second call closes it)."""
    def make():
        dialog = PlaceholderTextDialog(owner, None, select=select)
        _stay_above(dialog, anchor)
        return dialog
    return toggle_window(owner, '_interactive_placeholder_dialog', make)


class RatioPopup(QFrame):
    """The Auto aspect ratio's checklist: the ratios an image may get, one picked at random for each image."""

    changed = Signal(list)
    #: A click on the button within this many seconds of the list closing is the click that closed it.
    TOGGLE_SECONDS = 0.3

    def __init__(self, owner, selected, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #252525; border: 1px solid #5a9fd4; } "
                           "QLabel { color: #9ca3af; border: none; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        hint = QLabel("Auto picks one of the ticked ratios for each image")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.boxes = {}
        for ratio in core.ASPECT_CHOICES:
            box = _create_styled_checkbox(owner, ratio)
            box.setChecked(ratio in selected)
            box.toggled.connect(lambda _c=False: self.changed.emit(self.selected()))
            layout.addWidget(box)
            self.boxes[ratio] = box
        #: when the list last closed (time.monotonic)
        self.closed_at = 0.0

    def selected(self):
        return [ratio for ratio in core.ASPECT_CHOICES if self.boxes[ratio].isChecked()]

    def set_selected(self, selected):
        for ratio, box in self.boxes.items():
            blocked = box.blockSignals(True)
            box.setChecked(ratio in selected)
            box.blockSignals(blocked)

    def hideEvent(self, event):
        self.closed_at = time.monotonic()
        super().hideEvent(event)

    def popup_below(self, button):
        self.adjustSize()
        self.move(button.mapToGlobal(button.rect().bottomLeft()))
        self.show()


# =============================================================================================
# Settings dialog
# =============================================================================================

class InteractiveImagesDialog(QDialog):
    """Interactive Images settings: profile, images per chapter, prompts, models, references."""

    def __init__(self, owner, parent=None, *, output_dir=None):
        super().__init__(parent)
        self.owner = owner
        self.setWindowTitle("Interactive Images")
        self.setStyleSheet(DIALOG_STYLE)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.6), int(screen.height() * 0.78))
        config = _config(owner)
        settings = core.settings_from_config(config)
        layout = QVBoxLayout(self)

        intro = QLabel("A step after translation: a text model picks where illustrations go in each translated chapter "
                       "(and which style classes paragraphs get), an image model draws them, and Glossarion inserts "
                       "them. A chapter whose text would change is rejected; translation output is never touched "
                       "otherwise.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #a0aec0;")
        layout.addWidget(intro)

        # ---- profile row --------------------------------------------------------------------
        from review_dialog import build_prompt_profile_row, refresh_prompt_profile_row, confirm_prompt_profile_action
        self._refresh_row = refresh_prompt_profile_row
        self._confirm = confirm_prompt_profile_action
        self.profiles = core.InteractiveImagesProfiles(owner, persist=lambda: _persist(owner),
                                                       log=lambda message: _log(owner, message))
        row, self.profile_combo, self.profile_buttons = build_prompt_profile_row(
            "Profile:", "interactive_images",
            "Select an Interactive Images profile, or edit its name and save to rename it.\n"
            "Each profile stores the planner, image and CSS prompts and the images per chapter.",
            {'new': self._new_profile, 'save': self._save_profile, 'delete': self._delete_profile},
            self._select_profile)
        layout.addWidget(row)

        per_row = QHBoxLayout()
        per_row.addWidget(QLabel("Images per chapter:"))
        self.per_chapter = _house_combo(QComboBox())
        self.per_chapter.addItems([str(n) for n in range(0, core.MAX_IMAGES_PER_CHAPTER + 1)])
        self.per_chapter.setFixedWidth(80)
        self.per_chapter.setToolTip("0 = styles only (classes, no images)")
        per_row.addWidget(self.per_chapter)
        hint = QLabel("0 = styles only: classes from the book's interactive stylesheet, no image model needed")
        hint.setStyleSheet("color: gray; font-size: 9pt;")
        per_row.addWidget(hint)
        per_row.addStretch(1)
        layout.addLayout(per_row)

        # The widgets are bound with ``config.get(key, default)`` (defaults = interactive_images_core.DEFAULTS)
        # so the settings schema generator reads these keys, labels and defaults for the mobile settings.
        subject_row = QHBoxLayout()
        self.gender_cb = _create_styled_checkbox(owner, "Include gender with character names")
        self.gender_cb.setChecked(bool(config.get('interactive_images_include_gender', True)))
        self.gender_cb.setToolTip("Sends a gender with a character's name only when it is confident: the glossary "
                                  "and the Relations & Gender Tracker agree, or one has it and the other has "
                                  "nothing against it. A conflicting or unknown gender is never sent.\n"
                                  "The wording is the {character_gender} placeholder text.")
        subject_row.addWidget(self.gender_cb)
        subject_row.addSpacing(12)
        subject_row.addWidget(QLabel("Image subjects:"))
        self.subjects_combo = QComboBox()
        self.subjects = _house_combo(self.subjects_combo)
        self.subjects_combo.addItems(list(core.SUBJECTS))
        self.subjects_combo.setCurrentText(core.coerce_setting(
            'interactive_images_subjects', config.get('interactive_images_subjects', 'characters')))
        self.subjects.setFixedWidth(130)
        self.subjects.setToolTip("characters: scenes with the book's characters (as before)\n"
                                 "backgrounds: scenery and establishing moments, no characters\n"
                                 "both: the planner may pick either\nThe planner is told through {image_subjects}.")
        subject_row.addWidget(self.subjects)
        subject_row.addSpacing(12)
        self.placeholders_btn = _button("✏ Placeholder texts…", 'default',
                                        "Edit the texts the placeholders expand to (gender, image subjects, "
                                        "reference lines)")
        self.placeholders_btn.clicked.connect(lambda: open_placeholder_dialog(self.owner, self))
        subject_row.addWidget(self.placeholders_btn)
        self.gallery_btn = _button("🖼 Gallery…", 'accent', "Browse and delete the book's generated images")
        self.gallery_btn.clicked.connect(self._open_gallery)
        subject_row.addWidget(self.gallery_btn)
        subject_row.addStretch(1)
        layout.addLayout(subject_row)

        tabs = QTabWidget()
        self.editors = {}
        for field_name, label, names in (
                ('planner', "Planner", core.PLANNER_PLACEHOLDERS),
                ('image', "Image", core.IMAGE_PLACEHOLDERS),
                ('image_system', "Image system", ()),
                ('css', "CSS", core.CSS_PLACEHOLDERS)):
            editor = QPlainTextEdit()
            editor.setFont(QFont('Consolas', 9))
            _placeholder_menu(editor, names, owner)
            tabs.addTab(editor, label)
            self.editors[field_name] = editor
        self.editors['image_system'].setPlaceholderText("Optional system prompt for the image request")
        layout.addWidget(tabs, 2)

        # ---- models -------------------------------------------------------------------------
        models = QGroupBox("Models")
        grid = QVBoxLayout(models)
        text_row = QHBoxLayout()
        # The widgets are bound with ``config.get(key, default)`` (defaults = interactive_images_core.DEFAULTS)
        # so the settings schema generator reads these keys, labels and defaults for the mobile settings.
        text_row.addWidget(QLabel("Text model:"))
        self.text_model_combo = QComboBox()
        _model_combo(self.text_model_combo, "Main translation model")
        self.text_model_combo.setEditText(str(config.get('interactive_images_text_model', '') or ''))
        self.text_model = self.text_model_combo
        text_row.addWidget(self.text_model, 1)
        grid.addLayout(text_row)
        image_row = QHBoxLayout()
        image_row.addWidget(QLabel("Image model:"))
        self.image_model_combo = QComboBox()
        _model_combo(self.image_model_combo, "Image Gen / Edit key pool model")
        self.image_model_combo.setEditText(str(config.get('interactive_images_image_model', '') or ''))
        self.image_model = self.image_model_combo
        image_row.addWidget(self.image_model, 1)
        try:
            from multi_api_key_manager import create_preview_pool_button
            self.keys_btn = create_preview_pool_button(
                self, owner, 'inpainter', "🔑 Image Keys",
                "Open the Multi API Key Manager focused on the Image Gen/Edit key pool.")
            image_row.addWidget(self.keys_btn)
        except Exception:
            self.keys_btn = None
        grid.addLayout(image_row)
        self.ref_warning = QLabel("")
        self.ref_warning.setStyleSheet("color: #f6ad55; font-size: 9pt;")
        self.ref_warning.setWordWrap(True)
        grid.addWidget(self.ref_warning)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution = _house_combo(self.resolution_combo)
        self.resolution_combo.addItems(list(core.RESOLUTIONS))
        self.resolution_combo.setCurrentText(core.coerce_setting(
            'interactive_images_resolution', config.get('interactive_images_resolution', '1K')))
        self.resolution.setFixedWidth(84)
        out_row.addWidget(self.resolution)
        out_row.addWidget(QLabel("Aspect ratio:"))
        self.aspect_combo = QComboBox()
        self.aspect = _house_combo(self.aspect_combo)
        self.aspect_combo.addItems(list(core.ASPECT_RATIOS))
        self.aspect_combo.setCurrentText(core.coerce_setting(
            'interactive_images_aspect_ratio', config.get('interactive_images_aspect_ratio', '3:4')))
        self.aspect.setFixedWidth(96)
        self.aspect.setToolTip("auto: each image gets one of the ticked ratios, picked at random "
                               "(the same for a chapter's image number on every rerun)")
        out_row.addWidget(self.aspect)
        self.aspect_choices = core.coerce_setting('interactive_images_aspect_choices',
                                                  config.get('interactive_images_aspect_choices',
                                                             ['1:1', '3:4', '4:3', '16:9']))
        self.ratios_btn = _button("Ratios…", 'default', "The ratios Auto picks from")
        self.ratios_popup = None
        self.ratios_btn.clicked.connect(self.toggle_ratios)
        out_row.addWidget(self.ratios_btn)
        self.aspect_combo.currentTextChanged.connect(self._sync_ratios_button)
        out_row.addWidget(QLabel("Max reference images sent:"))
        self.max_refs_spin = QSpinBox()   # native: no stylesheet rule reaches it
        self.max_refs = self.max_refs_spin
        self.max_refs_spin.setRange(0, core.MAX_REFERENCES)
        self.max_refs_spin.setValue(core.coerce_setting('interactive_images_max_references',
                                                        config.get('interactive_images_max_references', 5)))
        self.max_refs.setFixedWidth(96)
        self.max_refs.wheelEvent = lambda event: event.ignore()
        out_row.addWidget(self.max_refs)
        out_row.addStretch(1)
        grid.addLayout(out_row)
        run_row = QHBoxLayout()
        self.auto_cb = _create_styled_checkbox(owner, "Run after each translation")
        self.auto_cb.setChecked(bool(config.get('interactive_images_auto_after_translation', False)))
        self.recompile_cb = _create_styled_checkbox(owner, "Recompile the EPUB/PDF afterwards (if it was compiled)")
        self.recompile_cb.setChecked(bool(config.get('interactive_images_recompile_after', True)))
        run_row.addWidget(self.auto_cb)
        run_row.addWidget(self.recompile_cb)
        run_row.addStretch(1)
        grid.addLayout(run_row)
        layout.addWidget(models)
        self.image_model.currentTextChanged.connect(self._update_reference_warning)

        # ---- references ---------------------------------------------------------------------
        self.references = ReferencesEditor(owner, output_dir=output_dir, parent=self)
        layout.addWidget(self.references, 2)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.save_btn = _button("💾 Save", 'primary', "Save the settings, this profile and the book's references")
        self.save_btn.clicked.connect(self._save_all)
        close_btn = _button("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._show_selected()
        self._update_reference_warning()
        self._sync_ratios_button()

    # ---- Auto aspect ratio / gallery --------------------------------------------------------------
    def _sync_ratios_button(self, *_args):
        self.ratios_btn.setVisible(self.aspect.currentText() == 'auto')
        self.ratios_btn.setText(f"Ratios ({len(self.aspect_choices)})…")

    def toggle_ratios(self):
        """The Ratios button: opens the checklist, and closes it on the second click (the click outside the list
        has just closed it: it must not reopen)."""
        popup = self.ratios_popup
        if popup is not None:
            if popup.isVisible():
                popup.hide()
                return popup
            if time.monotonic() - popup.closed_at < RatioPopup.TOGGLE_SECONDS:
                return popup
        if popup is None:
            popup = self.ratios_popup = RatioPopup(self.owner, self.aspect_choices, self)
            popup.changed.connect(self._on_ratios)
        popup.set_selected(self.aspect_choices)
        popup.popup_below(self.ratios_btn)
        return popup

    def _on_ratios(self, selected):
        self.aspect_choices = list(selected)
        self._sync_ratios_button()

    def _open_gallery(self):
        folder = self.references.output_dir
        if not folder:
            QMessageBox.information(self, "Gallery", "Pick a book (its output folder) in Reference images first.")
            return None
        return open_gallery_dialog(self.owner, self, folder)

    # ---- profiles -----------------------------------------------------------------------------
    def editor_texts(self):
        texts = {name: editor.toPlainText() for name, editor in self.editors.items()}
        texts['per_chapter'] = self.per_chapter.currentText()
        return texts

    def _show_selected(self):
        texts = self.profiles.texts
        for name, editor in self.editors.items():
            editor.setPlainText(texts.get(name, ''))
        self.per_chapter.setCurrentText(str(core.parse_per_chapter(texts.get('per_chapter', '1'))))
        self._refresh_row(self.profile_combo, self.profile_buttons, self.profiles,
                          "Restore the built-in '{name}' Interactive Images prompts",
                          "Delete the selected Interactive Images profile")

    def _box(self, box):
        if box:
            QMessageBox.warning(self, box[0], box[1])
        return box

    def _select_profile(self, name):
        name = str(name or '').strip()
        if not name or name == self.profiles.selected_name() or self.profiles.profile_texts(name) is None:
            return
        self.profiles.stage(self.profiles.selected_name(), self.editor_texts())
        if self.profiles.select(name) is None:
            return
        self._show_selected()
        _persist(self.owner)

    def _new_profile(self):
        self.profiles.stage(self.profiles.selected_name(), self.editor_texts())
        box = self.profiles.new()
        self._show_selected()
        self._box(box)

    def _save_profile(self):
        box = self.profiles.save(self.profile_combo.currentText(), self.editor_texts())
        if self._box(box):
            return False
        self._show_selected()
        return True

    def _delete_profile(self):
        name = self.profiles.selected_name()
        if self.profiles.is_builtin(name):
            if not self._confirm(self, "Reset Profile",
                                 f"Reset the built-in '{name}' Interactive Images profile to its shipped prompts?"):
                return
            box = self.profiles.reset(name)
        else:
            box = self.profiles.delete(name, confirm=lambda n: self._confirm(
                self, "Delete Profile", f"Delete Interactive Images profile '{n}'?"))
        self._show_selected()
        self._box(box)

    # ---- the rest -----------------------------------------------------------------------------
    def _update_reference_warning(self, *_args):
        level, text = core.reference_note(self.image_model.currentText())
        self.ref_warning.setText((("⚠ " if level == 'warning' else "ℹ ") + text) if text else "")
        self.ref_warning.setVisible(bool(self.ref_warning.text()))

    def apply_settings(self):
        config = _config(self.owner)
        config['interactive_images_text_model'] = self.text_model.currentText().strip()
        config['interactive_images_image_model'] = self.image_model.currentText().strip()
        config['interactive_images_max_references'] = int(self.max_refs.value())
        config['interactive_images_resolution'] = self.resolution.currentText()
        config['interactive_images_aspect_ratio'] = self.aspect.currentText()
        config['interactive_images_aspect_choices'] = list(core.coerce_aspect_choices(self.aspect_choices))
        config['interactive_images_include_gender'] = bool(self.gender_cb.isChecked())
        config['interactive_images_subjects'] = self.subjects.currentText()
        config['interactive_images_auto_after_translation'] = bool(self.auto_cb.isChecked())
        config['interactive_images_recompile_after'] = bool(self.recompile_cb.isChecked())

    def _save_all(self):
        self.apply_settings()
        if not self._save_profile():
            return
        self.references.save()
        if not _persist(self.owner):
            QMessageBox.warning(self, "Save Failed", "Could not save the settings. Please try again.")
            return
        try:
            from qt_widget_helpers import flash_saved
            flash_saved(self.save_btn)
        except Exception:
            pass
        _log(self.owner, "✅ Saved the Interactive Images settings")


class ReferencesEditor(QGroupBox):
    """Reference images per book: character name -> images, plus art-style images (``references.json``)."""

    def __init__(self, owner, *, output_dir=None, parent=None):
        super().__init__("Reference images", parent)
        self.owner = owner
        self.refs = core.empty_references()
        self.output_dir = None
        layout = QVBoxLayout(self)
        book_row = QHBoxLayout()
        book_row.addWidget(QLabel("Book:"))
        self.book_combo = _house_combo(QComboBox())
        self.book_combo.setMinimumWidth(280)
        book_row.addWidget(self.book_combo, 1)
        browse = _button("📁 Output folder…", 'default', "Pick a book's output folder")
        browse.clicked.connect(self._browse)
        book_row.addWidget(browse)
        layout.addLayout(book_row)
        note = QLabel("Images are copied into Interactive_Images/references/ of the book. The scene's characters "
                      "fill the reference slots first, then the art-style images.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(note)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Character", "Images", "", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, 2)
        char_row = QHBoxLayout()
        self.name_edit = _line(placeholder="Character name")
        add_char = _button("+ Character", 'success')
        add_char.clicked.connect(self._add_character)
        load_names = _button("Load names from glossary", 'info')
        load_names.clicked.connect(self._load_names)
        char_row.addWidget(self.name_edit, 1)
        char_row.addWidget(add_char)
        char_row.addWidget(load_names)
        layout.addLayout(char_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Art style:"))
        self.style_list = QListWidget()
        self.style_list.setMaximumHeight(64)
        self.style_list.setFlow(QListWidget.LeftToRight)
        style_row.addWidget(self.style_list, 1)
        add_style = _button("Add…")
        add_style.clicked.connect(self._add_style)
        remove_style = _button("Remove")
        remove_style.clicked.connect(self._remove_style)
        style_row.addWidget(add_style)
        style_row.addWidget(remove_style)
        layout.addLayout(style_row)

        extra = [(os.path.basename(output_dir), output_dir)] if output_dir else []
        for label, folder in _book_output_dirs(owner, extra):
            self.book_combo.addItem(label, folder)
            self.book_combo.setItemData(self.book_combo.count() - 1, folder, Qt.ToolTipRole)
        self.book_combo.currentIndexChanged.connect(lambda _i: self._load(self.book_combo.currentData()))
        self._load(self.book_combo.currentData())

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select the book's output folder")
        if not folder:
            return
        self.book_combo.addItem(os.path.basename(folder), folder)
        self.book_combo.setCurrentIndex(self.book_combo.count() - 1)

    def _load(self, folder):
        self.output_dir = folder or None
        self.refs = core.load_references(folder) if folder else core.empty_references()
        self._render()

    def _render(self):
        enabled = bool(self.output_dir)
        self.table.setEnabled(enabled)
        self.table.setRowCount(0)
        for name, paths in core.reference_rows(self.refs):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            names = ', '.join(os.path.basename(p) for p in paths) or '(none)'
            item = QTableWidgetItem(f"{len(paths)}: {names}")
            item.setToolTip('\n'.join(paths))
            self.table.setItem(row, 1, item)
            add = _button("Add…")
            add.clicked.connect(lambda _c=False, n=name: self._add_images(n))
            clear = _button("Clear")
            clear.clicked.connect(lambda _c=False, n=name: self._clear(n))
            self.table.setCellWidget(row, 2, add)
            self.table.setCellWidget(row, 3, clear)
        self.style_list.clear()
        for path in self.refs.get('style', []):
            self.style_list.addItem(os.path.basename(path))

    def _need_book(self):
        if self.output_dir:
            return True
        QMessageBox.information(self, "Reference images", "Pick a book (its output folder) first.")
        return False

    def _pick_images(self):
        files, _f = QFileDialog.getOpenFileNames(self, "Select reference images", "",
                                                 "Images (*.png *.jpg *.jpeg *.webp)")
        bad = [path for path in files if not core.is_reference_image(path)]
        if bad:
            QMessageBox.warning(self, "Reference images", "Not a PNG, JPEG or WebP image:\n" + "\n".join(bad[:5]))
        return [path for path in files if path not in bad]

    def _add_character(self):
        name = ' '.join(self.name_edit.text().split())
        if not name or not self._need_book():
            return
        core.add_reference_names(self.refs, [name])
        self.name_edit.clear()
        self._render()

    def _load_names(self):
        if not self._need_book():
            return
        entries = core.load_glossary_entries(self.output_dir)
        names = core.character_names(entries)
        if not names:
            QMessageBox.information(self, "Reference images", "No character names found in this book's glossary.")
            return
        core.add_reference_names(self.refs, names)
        self._render()

    def _add_images(self, name):
        if not self._need_book():
            return
        for path in self._pick_images():
            try:
                core.add_reference_image(self.output_dir, self.refs, path, name)
            except OSError as exc:
                QMessageBox.warning(self, "Reference images", f"Could not copy {path}:\n{exc}")
        self._render()

    def _clear(self, name):
        core.clear_reference_character(self.output_dir, self.refs, name)
        self._render()

    def _add_style(self):
        if not self._need_book():
            return
        for path in self._pick_images():
            try:
                core.add_reference_image(self.output_dir, self.refs, path, None)
            except OSError as exc:
                QMessageBox.warning(self, "Reference images", f"Could not copy {path}:\n{exc}")
        self._render()

    def _remove_style(self):
        row = self.style_list.currentRow()
        styles = self.refs.get('style', [])
        if 0 <= row < len(styles):
            core.remove_reference_image(self.output_dir, self.refs, styles[row])
            self._render()

    def save(self):
        if self.output_dir:
            core.save_references(self.output_dir, self.refs)


def open_settings_dialog(owner, anchor=None, output_dir=None):
    """The settings window (toggles: a second click on its button closes it). Parentless, tied above
    ``anchor``'s window: Other Settings' sheet styles QSpinBox, which would break the native spin box."""
    def make():
        dialog = InteractiveImagesDialog(owner, None, output_dir=output_dir)
        _stay_above(dialog, anchor)
        return dialog
    return toggle_window(owner, '_interactive_images_dialog', make)


def open_styles_dialog(owner, anchor=None, output_dir=None, on_preset=None):
    def make():
        dialog = InteractiveStylesDialog(owner, None, output_dir=output_dir)
        _stay_above(dialog, anchor)
        if on_preset is not None:
            dialog.preset_changed.connect(on_preset)
        return dialog
    return toggle_window(owner, '_interactive_styles_dialog', make)


# =============================================================================================
# Other Settings rows (EPUB section, under Attach CSS)
# =============================================================================================

def add_interactive_styles_rows(owner, section_layout, parent=None):
    """The "Interactive styles" row (checkbox, style dropdown, Styles & Preview, Clear) and the
    "Interactive Images…" row, under the Attach CSS row. The two stylesheets stay separate files:
    Clear here only turns interactive styles off; it never touches the Attach CSS override."""
    config = _config(owner)
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 5)
    h.setSpacing(8)
    enabled_cb = _create_styled_checkbox(owner, "Interactive styles")
    enabled_cb.setChecked(bool(config.get('interactive_css_enabled', False)))
    enabled_cb.setToolTip("Link the book's glossarion_interactive.css (after your Attach CSS file) and mark the "
                          "chapter bodies with glx-book when Interactive Images runs.\nCSS only: every rule is "
                          "scoped under a .glx- class.")
    h.addWidget(enabled_cb)
    combo = _house_combo(QComboBox())
    combo.addItems(presets.preset_names(config))
    combo.setCurrentText(presets.active_preset_name(config))
    combo.setMinimumWidth(200)
    combo.setToolTip("The style the next Interactive Images run writes as the book's stylesheet")
    h.addWidget(combo)
    styles_btn = _button("🎨 Styles & Preview…", 'info', "Preview, duplicate, edit or write styles")
    h.addWidget(styles_btn)
    clear_btn = QPushButton("Clear")
    clear_btn.setToolTip("Turn interactive styles off (the Attach CSS file is not touched)")
    clear_btn.setMinimumWidth(60)
    clear_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 4px 8px; "
                            "border-radius: 4px; font-weight: bold; } QPushButton:hover { background-color: #5a6268; }")
    h.addWidget(clear_btn)
    h.addStretch(1)
    section_layout.addWidget(row)

    row2 = QWidget()
    h2 = QHBoxLayout(row2)
    h2.setContentsMargins(0, 0, 0, 5)
    h2.setSpacing(8)
    images_btn = _button("🖼 Interactive Images…", 'accent',
                         "Illustrations and style classes for translated chapters (a separate step after translation)")
    h2.addWidget(images_btn)
    images_note = QLabel("Illustrations after translation, from the Progress Manager's Interactive Images tab")
    images_note.setStyleSheet("color: gray; font-size: 9pt;")
    h2.addWidget(images_note)
    h2.addStretch(1)
    section_layout.addWidget(row2)

    def on_toggle(checked):
        _config(owner)[presets.ENABLED_KEY] = bool(checked)

    def on_preset(name):
        if name:
            _config(owner)[presets.PRESET_KEY] = name

    def sync_combo(name):
        try:
            blocked = combo.blockSignals(True)
            combo.clear()
            combo.addItems(presets.preset_names(_config(owner)))
            combo.setCurrentText(name)
            combo.blockSignals(blocked)
        except RuntimeError:
            pass

    enabled_cb.toggled.connect(on_toggle)
    combo.currentTextChanged.connect(on_preset)
    clear_btn.clicked.connect(lambda: enabled_cb.setChecked(False))
    styles_btn.clicked.connect(lambda: open_styles_dialog(owner, parent or row, on_preset=sync_combo))
    images_btn.clicked.connect(lambda: open_settings_dialog(owner, parent or row))
    owner.interactive_css_enabled_checkbox = enabled_cb
    owner.interactive_css_preset_combo = combo
    return row, row2


# =============================================================================================
# Gallery: the book's AI images, with multi-select delete (the model is interactive_images_progress_core's)
# =============================================================================================

THUMB_SIZE = 168
ALL_CHARACTERS = "All characters"
ALL_CHAPTERS = "All chapters"


def _thumbnail(path, size=THUMB_SIZE):
    """A ``size`` x ``size`` icon of an image file (read scaled; an unreadable file gives an empty icon)."""
    from PySide6.QtGui import QPainter

    screen = QApplication.primaryScreen()
    side = int(size * (screen.devicePixelRatio() if screen else 1.0))
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    original = reader.size()
    if original.isValid() and original.width() and original.height():
        scale = min(side / original.width(), side / original.height(), 1.0)
        reader.setScaledSize(QSize(max(1, int(original.width() * scale)), max(1, int(original.height() * scale))))
    image = reader.read()
    if image.isNull():
        return QIcon()
    # a square card with the picture centred (portrait and landscape images line up in the grid)
    card = QPixmap(side, side)
    card.fill(QColor('#2a2d33'))
    picture = QPixmap.fromImage(image).scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    painter = QPainter(card)
    painter.drawPixmap((side - picture.width()) // 2, (side - picture.height()) // 2, picture)
    painter.end()
    return QIcon(card)


class _CardDelegate(QStyledItemDelegate):
    """The gallery cards: the default item look with Other Settings' checkbox (a 16 px rounded square, #5a9fd4
    border and fill when ticked, a white tick) drawn at the top-left corner of the card. Qt's own indicator is
    left out (it shows a lone tick with no box on this theme); a click on the box ticks it without selecting the
    card, a click anywhere else selects as usual."""

    BOX = 18

    def box_rect(self, option):
        return QRect(option.rect.left() + 6, option.rect.top() + 6, self.BOX, self.BOX)

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.features &= ~QStyleOptionViewItem.HasCheckIndicator
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        if not (index.flags() & Qt.ItemIsUserCheckable):
            return
        checked = index.data(Qt.CheckStateRole) in (Qt.Checked, 2, Qt.Checked.value)
        painter.save()
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            box = self.box_rect(option)
            painter.setPen(QPen(QColor('#5a9fd4'), 1.5))
            painter.setBrush(QColor('#5a9fd4') if checked else QColor(45, 45, 45, 220))
            painter.drawRoundedRect(box, 3, 3)
            if checked:
                painter.setPen(QPen(QColor('white'), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                x, y, s = box.left(), box.top(), box.width()
                painter.drawLine(x + s * 0.24, y + s * 0.52, x + s * 0.43, y + s * 0.72)
                painter.drawLine(x + s * 0.43, y + s * 0.72, x + s * 0.77, y + s * 0.30)
        finally:
            painter.restore()

    def editorEvent(self, event, model, option, index):
        inside = False
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            inside = self.box_rect(option).contains(event.position().toPoint())
        if inside and index.flags() & Qt.ItemIsUserCheckable:
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                checked = index.data(Qt.CheckStateRole) in (Qt.Checked, 2, Qt.Checked.value)
                model.setData(index, Qt.Unchecked if checked else Qt.Checked, Qt.CheckStateRole)
            return True
        return False


class ImageViewerDialog(QDialog):
    """One image at full size, with its details; Delete and Go to chapter."""

    deleted = Signal(str)   # the image id

    def __init__(self, gallery, image, parent=None):
        super().__init__(parent)
        self.gallery = gallery
        self.image = image
        self.setWindowTitle(f"{image.file} - {image.caption}")
        self.setStyleSheet(DIALOG_STYLE)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.8))
        layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.picture = QLabel()
        self.picture.setPixmap(QPixmap(image.path))
        self.scroll.setWidget(self.picture)
        layout.addWidget(self.scroll, 1)
        detail = [image.caption]
        if image.aspect_ratio:
            detail.append(f"Ratio {image.aspect_ratio}")
        if image.prompt:
            detail.append(image.prompt)
        info = QLabel(" | ".join(detail))
        info.setWordWrap(True)
        info.setStyleSheet("color: #a0aec0;")
        layout.addWidget(info)
        buttons = QHBoxLayout()
        self.chapter_btn = _button("📄 Go to chapter", 'info', "Open the chapter file")
        self.delete_btn = _button("🗑 Delete", 'danger', "Delete this image and its figure in the chapter")
        close_btn = _button("Close")
        self.chapter_btn.clicked.connect(lambda: gallery.go_to_chapter(image))
        self.delete_btn.clicked.connect(self._delete)
        close_btn.clicked.connect(self.close)
        for button in (self.chapter_btn, self.delete_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _delete(self):
        if self.gallery.delete_images([self.image]):
            self.deleted.emit(self.image.id)
            self.close()


class InteractiveImagesGalleryDialog(QDialog):
    """A scrollable grid of the book's AI images; each card shows the image, its chapter and the characters
    present. Filter by character and chapter; select with ctrl / shift click or the checkboxes; Delete selected
    removes the images and their figures through the shared, text-checked delete logic."""

    def __init__(self, owner, output_dir, parent=None, *, confirm=True):
        super().__init__(parent)
        self.owner = owner
        self.output_dir = output_dir
        self.confirm = confirm
        self.model = ipc.open_progress(output_dir)
        self.images = []
        self._by_id = {}
        self._viewers = []
        self.setWindowTitle("Interactive Images Gallery")
        self.setStyleSheet(DIALOG_STYLE)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.62), int(screen.height() * 0.78))
        layout = QVBoxLayout(self)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Character:"))
        self.character_combo = _house_combo(QComboBox())
        self.character_combo.setMinimumWidth(180)
        filters.addWidget(self.character_combo)
        filters.addWidget(QLabel("Chapter:"))
        self.chapter_combo = _house_combo(QComboBox())
        self.chapter_combo.setMinimumWidth(200)
        filters.addWidget(self.chapter_combo)
        filters.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #a0aec0;")
        filters.addWidget(self.count_label)
        layout.addLayout(filters)

        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setWrapping(True)
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(True)
        self.list.setSpacing(8)
        self.list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.setGridSize(QSize(THUMB_SIZE + 28, THUMB_SIZE + 62))
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)   # ctrl / shift click
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setItemDelegate(_CardDelegate(self.list))
        self.list.setStyleSheet("QListWidget::item { color: #e2e8f0; border-radius: 4px; } "
                                "QListWidget::item:hover { background-color: #333c4a; } "
                                "QListWidget::item:selected { background-color: #2f5f8f; color: white; }")
        layout.addWidget(self.list, 1)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #f6ad55; font-size: 9pt;")
        self.note.hide()
        layout.addWidget(self.note)

        buttons = QHBoxLayout()
        self.all_btn = _button("Select all", 'default')
        self.none_btn = _button("Select none", 'default')
        self.view_btn = _button("🔍 Open full size", 'info', "Open the highlighted image at full size")
        self.chapter_btn = _button("📄 Go to chapter", 'info', "Open the highlighted image's chapter")
        self.delete_btn = _button("🗑 Delete selected", 'danger',
                                  "Delete the ticked / highlighted images and their figures in the chapters")
        close_btn = _button("Close")
        for button in (self.all_btn, self.none_btn, self.view_btn, self.chapter_btn, self.delete_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self.character_combo.currentIndexChanged.connect(lambda _i: self.render())
        self.chapter_combo.currentIndexChanged.connect(lambda _i: self.render())
        self.list.itemChanged.connect(lambda _item: self._update_buttons())
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.itemDoubleClicked.connect(lambda item: self.open_viewer(self._image_for(item)))
        self.all_btn.clicked.connect(self.select_all)
        self.none_btn.clicked.connect(self.select_none)
        self.view_btn.clicked.connect(lambda: self.open_viewer(self._current_image()))
        self.chapter_btn.clicked.connect(lambda: self.go_to_chapter(self._current_image()))
        self.delete_btn.clicked.connect(self.delete_selected)
        close_btn.clicked.connect(self.close)
        self.reload()

    # ---- data -----------------------------------------------------------------------------------
    def reload(self):
        """Read the progress file again and rebuild the filters and the grid."""
        ipc.reload_progress(self.model)
        self.images = ipc.gallery_images(self.model)
        self._by_id = {image.id: image for image in self.images}
        names = core.character_names(core.load_glossary_entries(self.output_dir))
        self._fill_combo(self.character_combo, [(ALL_CHARACTERS, '')] +
                         [(name, name) for name in ipc.gallery_characters(self.images, names)])
        self._fill_combo(self.chapter_combo, [(ALL_CHAPTERS, '')] +
                         [(label, key) for key, label in ipc.gallery_chapters(self.images)])
        self.render()

    @staticmethod
    def _fill_combo(combo, rows):
        current = combo.currentData()
        blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for label, data in rows:
                combo.addItem(label, data)
            index = combo.findData(current) if current else 0
            combo.setCurrentIndex(max(0, index))
        finally:
            combo.blockSignals(blocked)

    def visible_images(self):
        return ipc.gallery_filter(self.images, self.character_combo.currentData() or '',
                                  self.chapter_combo.currentData() or '')

    def render(self):
        keep = set(self.selected_ids())
        blocked = self.list.blockSignals(True)
        try:
            self.list.clear()
            for image in self.visible_images():
                item = QListWidgetItem(_thumbnail(image.path), image.caption)
                item.setData(Qt.UserRole, image.id)
                item.setToolTip(f"{image.file}\n{image.prompt}" if image.prompt else image.file)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if image.id in keep else Qt.Unchecked)
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
                self.list.addItem(item)
        finally:
            self.list.blockSignals(blocked)
        self._update_buttons()

    # ---- selection --------------------------------------------------------------------------------
    def _image_for(self, item):
        return self._by_id.get(item.data(Qt.UserRole)) if item is not None else None

    def _current_image(self):
        selected = self.selected_images()
        if selected:
            return selected[0]
        return self._image_for(self.list.currentItem())

    def selected_ids(self):
        """Ticked or highlighted images (ctrl / shift click highlights, the checkbox ticks)."""
        ids = []
        for index in range(self.list.count()):
            item = self.list.item(index)
            if item.checkState() == Qt.Checked or item.isSelected():
                ids.append(item.data(Qt.UserRole))
        return ids

    def selected_images(self):
        return [self._by_id[i] for i in self.selected_ids() if i in self._by_id]

    def select_all(self):
        blocked = self.list.blockSignals(True)
        try:
            for index in range(self.list.count()):
                self.list.item(index).setCheckState(Qt.Checked)
        finally:
            self.list.blockSignals(blocked)
        self._update_buttons()

    def select_none(self):
        blocked = self.list.blockSignals(True)
        try:
            for index in range(self.list.count()):
                item = self.list.item(index)
                item.setCheckState(Qt.Unchecked)
                item.setSelected(False)
        finally:
            self.list.blockSignals(blocked)
        self._update_buttons()

    def _update_buttons(self):
        count = len(self.selected_ids())
        self.count_label.setText(f"{self.list.count()} of {len(self.images)} images | {count} selected")
        self.delete_btn.setEnabled(count > 0)
        self.view_btn.setEnabled(count > 0 or self.list.currentItem() is not None)
        self.chapter_btn.setEnabled(self.view_btn.isEnabled())
        self.delete_btn.setText(f"🗑 Delete selected ({count})" if count else "🗑 Delete selected")

    # ---- actions ------------------------------------------------------------------------------------
    def go_to_chapter(self, image):
        if image is None:
            return False
        return open_path(os.path.join(self.output_dir, image.key))

    def open_viewer(self, image):
        if image is None:
            return None
        viewer = ImageViewerDialog(self, image, self)
        viewer.setAttribute(Qt.WA_DeleteOnClose, True)
        viewer.destroyed.connect(lambda *_a, v=viewer: self._viewers.remove(v) if v in self._viewers else None)
        self._viewers.append(viewer)
        viewer.show()
        return viewer

    def delete_selected(self):
        return self.delete_images(self.selected_images())

    def delete_images(self, images):
        """Delete ``images`` after a confirmation: the shared ``delete_gallery_images`` (text-checked)."""
        images = [image for image in images or () if image is not None]
        if not images:
            return 0
        if self.confirm:
            try:
                from review_dialog import confirm_prompt_profile_action
                if not confirm_prompt_profile_action(self, "Delete Images", ipc.gallery_delete_confirmation(len(images))):
                    return 0
            except Exception:
                pass
        result = ipc.delete_gallery_images(self.output_dir, images)
        _log(self.owner, f"🗑 Interactive Images: deleted {result.deleted} image(s) from {result.chapters} chapter(s)")
        if result.skipped:
            self.note.setText(f"{len(result.skipped)} image(s) were kept: " + "; ".join(
                f"{image.file}: {reason}" for image, reason in result.skipped[:3]))
            self.note.show()
        else:
            self.note.hide()
        self.reload()
        return result.deleted


def open_gallery_dialog(owner, anchor=None, output_dir=None):
    """The gallery window of one book (toggles: the button that opens it closes it on the second click)."""
    existing = getattr(owner, '_interactive_gallery_dialog', None)
    try:
        if existing is not None and existing.isVisible() and \
                os.path.normcase(existing.output_dir) != os.path.normcase(str(output_dir)):
            existing.close()                      # another book's gallery: replace it
            owner._interactive_gallery_dialog = None
    except RuntimeError:
        owner._interactive_gallery_dialog = None

    def make():
        dialog = InteractiveImagesGalleryDialog(owner, output_dir, None)
        _stay_above(dialog, anchor)
        return dialog
    return toggle_window(owner, '_interactive_gallery_dialog', make)


# =============================================================================================
# Runs (one at a time; a worker thread with a stop event)
# =============================================================================================

class _RunBridge(QObject):
    progress = Signal(str, str, str)   # output_dir, key, status
    finished = Signal(str, object)     # output_dir, RunResult or exception


def run_bridge(owner):
    bridge = getattr(owner, '_interactive_images_bridge', None)
    if bridge is None:
        bridge = _RunBridge()
        owner._interactive_images_bridge = bridge
    return bridge


def current_run(owner):
    run = getattr(owner, '_interactive_images_run', None)
    if run and run['thread'].is_alive():
        return run
    return None


def start_run(owner, output_dir, keys=None):
    """Start the Interactive Images step for one book on a worker thread. Returns False when one is running."""
    if current_run(owner):
        _log(owner, "⚠️ Interactive Images is already running")
        return False
    from interactive_images_runner import owner_recompiler, run_interactive_images

    bridge = run_bridge(owner)
    stop = threading.Event()

    def work():
        try:
            result = run_interactive_images(
                _config(owner), output_dir, keys, log=lambda message: _log(owner, message), stop_event=stop,
                on_progress=lambda key, status: bridge.progress.emit(output_dir, key, status),
                recompile=owner_recompiler(owner))
        except Exception as exc:
            _log(owner, f"❌ Interactive Images failed: {exc}\n{traceback.format_exc()}")
            result = exc
        bridge.finished.emit(output_dir, result)

    thread = threading.Thread(target=work, name='interactive-images', daemon=True)
    owner._interactive_images_run = {'thread': thread, 'stop': stop, 'output_dir': output_dir}
    thread.start()
    return True


def stop_run(owner):
    run = current_run(owner)
    if run:
        run['stop'].set()
        _log(owner, "⏹ Interactive Images: stopping after the current request")
        return True
    return False


# =============================================================================================
# Progress Manager tab
# =============================================================================================

def _pm_key(path):
    return os.path.normcase(os.path.abspath(str(path or '')))


class ImagesFilePanel(QWidget):
    """One book's Interactive Images progress (the PM tab page of the file in the shared dropdown)."""

    def __init__(self, owner, window, output_dir, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.window_ref = window
        self.output_dir = output_dir
        self.model = ipc.open_progress(output_dir)
        self._rows = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.stats_labels = {}
        stats = QHBoxLayout()
        stats.setContentsMargins(0, 3, 0, 3)
        self.total_label = QLabel("Total: 0 | ")
        self.total_label.setStyleSheet("font-size: 9pt;")
        stats.addWidget(self.total_label)
        for group, text, color in ipc.STAT_CHIPS:
            label = QLabel(f"{text}: 0")
            label.setStyleSheet(f"color: {color}; font-size: 9pt;")
            label.setCursor(Qt.PointingHandCursor)
            label.setToolTip("Click to select the next row with this status")
            label.mousePressEvent = self._make_cycle(ipc.STATUS_GROUPS[group])
            stats.addWidget(label)
            if group != 'not_built':
                stats.addWidget(QLabel("|"))
            self.stats_labels[group] = (label, text)
        stats.addStretch(1)
        layout.addLayout(stats)

        self.list = QListWidget()
        try:
            owner._apply_compact_inline_list_style(self.list, QFont('Courier', 10))
        except Exception:
            self.list.setFont(QFont('Courier', 10))
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.itemDoubleClicked.connect(lambda item: self._show_plan([self._row_for(item)]))
        layout.addWidget(self.list, 1)

        path_row = QHBoxLayout()
        self.path_label = QLabel(f"📁 {core.interactive_dir(output_dir)}")
        self.path_label.setStyleSheet("color: #666; font-size: 8pt;")
        self.path_label.setWordWrap(True)
        path_row.addWidget(self.path_label, 1)
        open_folder = _button("📂 Open Folder", 'default')
        open_folder.clicked.connect(lambda: self._open_path(core.interactive_dir(self.output_dir), make=True))
        path_row.addWidget(open_folder)
        layout.addLayout(path_row)

        actions = QHBoxLayout()
        self.build_btn = _button("▶ Build pending", 'success', "Plan, draw and insert images for chapters that need it")
        self.stop_btn = _button("⏹ Stop", 'danger', "Stop after the current request")
        settings_btn = _button("⚙ Settings", 'default', "Interactive Images settings and this book's references")
        styles_btn = _button("🎨 Styles", 'info', "Interactive styles and the preview")
        gallery_btn = _button("🖼 Gallery", 'accent', "Browse this book's images; delete them with multi-selection")
        try:
            from Retranslation_GUI import AnimatedRefreshButton
            self.refresh_btn = AnimatedRefreshButton("  Refresh")
        except Exception:
            self.refresh_btn = QPushButton("  Refresh")
        self.build_btn.clicked.connect(lambda: self._build(None))
        self.stop_btn.clicked.connect(lambda: stop_run(self.owner))
        settings_btn.clicked.connect(lambda: open_settings_dialog(self.owner, self, self.output_dir))
        styles_btn.clicked.connect(lambda: open_styles_dialog(self.owner, self, self.output_dir))
        gallery_btn.clicked.connect(lambda: open_gallery_dialog(self.owner, self, self.output_dir))
        self.gallery_btn = gallery_btn
        self.refresh_btn.clicked.connect(self.full_refresh)
        for button in (self.build_btn, self.stop_btn, settings_btn, styles_btn, gallery_btn, self.refresh_btn):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        try:
            from progress_core import _PROGRESS_WATCH_DEBOUNCE_MS
        except Exception:
            _PROGRESS_WATCH_DEBOUNCE_MS = 500
        self._debounce.setInterval(_PROGRESS_WATCH_DEBOUNCE_MS)
        self._debounce.timeout.connect(self.refresh)
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_watch)
        self.watcher.directoryChanged.connect(self._on_watch)
        self._watch()
        self._signature = None
        bridge = run_bridge(owner)
        bridge.progress.connect(self._on_run_progress)
        bridge.finished.connect(self._on_run_finished)
        self.refresh(force=True)
        self._update_buttons()

    # ---- refresh --------------------------------------------------------------------------------
    def _watch(self):
        paths = [core.progress_path(self.output_dir), os.path.join(self.output_dir, 'translation_progress.json'),
                 core.interactive_dir(self.output_dir)]
        present = [p for p in paths if os.path.exists(p)]
        missing = [p for p in present if p not in self.watcher.files() + self.watcher.directories()]
        if missing:
            self.watcher.addPaths(missing)

    def _on_watch(self, *_args):
        if self.isVisible():
            self._debounce.start()

    def _row_for(self, item):
        index = self.list.row(item)
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def refresh(self, force=False):
        signature = ipc.book_signature(self.output_dir)
        if not force and signature == self._signature:
            return
        self._signature = signature
        self._watch()
        ipc.reload_progress(self.model)
        rows = ipc.image_rows(self.model)
        selected = {self._rows[self.list.row(i)].key for i in self.list.selectedItems()
                    if 0 <= self.list.row(i) < len(self._rows)}
        if [r.key for r in rows] != [r.key for r in self._rows]:
            self.list.clear()
            for row in rows:
                item = QListWidgetItem(row.display)
                try:
                    self.owner._set_compact_inline_item_size(self.list, item)
                except Exception:
                    pass
                self.list.addItem(item)
        for index, row in enumerate(rows):
            item = self.list.item(index)
            if item.text() != row.display:
                item.setText(row.display)
            item.setForeground(QColor(row.color))
            item.setData(Qt.UserRole, row.status)
            item.setToolTip(row.error or ipc.STATUS_LABELS.get(row.status, row.status))
            item.setSelected(row.key in selected)
        self._rows = rows
        stats = ipc.image_stats(self.model, rows)
        self.total_label.setText(f"Total: {stats['total']} | Images: {stats['images']} | ")
        for group, (label, text) in self.stats_labels.items():
            label.setText(f"{text}: {stats.get(group, 0)}")
        self.stats = stats
        callback = getattr(self, 'on_stats', None)
        if callable(callback):
            callback(self, stats)

    def full_refresh(self):
        try:
            self.refresh_btn.start_animation()
        except Exception:
            pass
        self.refresh(force=True)
        QTimer.singleShot(800, lambda: self._finish_refresh_animation())

    def _finish_refresh_animation(self):
        try:
            self.refresh_btn.stop_animation()
        except Exception:
            pass

    def _make_cycle(self, statuses):
        statuses = set(statuses)

        def handler(_event=None):
            indices = [i for i in range(self.list.count()) if self.list.item(i).data(Qt.UserRole) in statuses]
            if not indices:
                return
            current = self.list.currentRow()
            nxt = next((i for i in indices if i > current), indices[0])
            self.list.setCurrentRow(nxt, QItemSelectionModel.ClearAndSelect)
            self.list.scrollToItem(self.list.item(nxt), QListWidget.PositionAtCenter)
        return handler

    # ---- runs -----------------------------------------------------------------------------------
    def _update_buttons(self):
        run = current_run(self.owner)
        running_here = bool(run and _pm_key(run['output_dir']) == _pm_key(self.output_dir))
        self.build_btn.setEnabled(run is None)
        self.stop_btn.setEnabled(running_here)

    def _on_run_progress(self, output_dir, _key, _status):
        if _pm_key(output_dir) == _pm_key(self.output_dir):
            self.refresh(force=True)

    def _on_run_finished(self, output_dir, _result):
        self._update_buttons()
        if _pm_key(output_dir) == _pm_key(self.output_dir):
            self.refresh(force=True)

    def _build(self, keys):
        if start_run(self.owner, self.output_dir, keys):
            self._update_buttons()

    # ---- context menu ---------------------------------------------------------------------------
    def selected_rows(self):
        return [row for row in (self._row_for(item) for item in self.list.selectedItems()) if row is not None]

    def _context_menu(self, pos):
        rows = self.selected_rows()
        if not rows:
            item = self.list.itemAt(pos)
            if item is None:
                return
            item.setSelected(True)
            rows = self.selected_rows()
        try:
            from qt_widget_helpers import CONTEXT_MENU_STYLE
        except Exception:
            CONTEXT_MENU_STYLE = ''
        menu = QMenu(self)
        menu.setStyleSheet(CONTEXT_MENU_STYLE)
        running = current_run(self.owner) is not None
        actions = {}
        for action, _label in ipc.ROW_ACTIONS:   # the shared table (the mobile Images tab shows the same)
            if action in ('delete', 'open_images'):
                menu.addSeparator()
            entry = menu.addAction(ipc.action_label(action, rows))
            entry.setEnabled(ipc.action_reason(action, rows, running=running) is None)
            actions[entry] = action
        chosen = menu.exec(self.list.mapToGlobal(pos))
        menu.deleteLater()
        if chosen is None or chosen not in actions:
            return
        self.run_action(actions[chosen], rows)

    def run_action(self, action, rows):
        targets = ipc.action_targets(action, rows)
        if action in ('build', 'retry'):
            self._build([r.key for r in targets])
        elif action == 'mark':
            ipc.mark_for_rerun(core.progress_path(self.output_dir), targets)
            self.refresh(force=True)
        elif action == 'delete':
            self._delete(targets)
        elif action == 'open_images':
            for row in targets:
                for path in ipc.open_targets(self.output_dir, row)['images']:
                    self._open_path(path)
        elif action == 'open_chapter':
            for row in targets[:5]:
                self._open_path(ipc.open_targets(self.output_dir, row)['chapter'])
        elif action == 'show_plan':
            self._show_plan(targets)
        elif action == 'open_folder':
            self._open_path(core.interactive_dir(self.output_dir), make=True)

    def _delete(self, rows):
        try:
            from review_dialog import confirm_prompt_profile_action
            ok = confirm_prompt_profile_action(self, "Delete Images", ipc.delete_confirmation(len(rows)))
        except Exception:
            ok = True
        if not ok:
            return
        count = ipc.delete_entries(self.output_dir, rows)
        _log(self.owner, f"🗑 Interactive Images: deleted the images of {count} chapter(s)")
        self.refresh(force=True)

    def _show_plan(self, rows):
        rows = [r for r in rows if r is not None]
        if not rows:
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setWindowTitle("Interactive Images plan")
        dialog.setStyleSheet(DIALOG_STYLE)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText('\n\n'.join(f"── {row.key} ──\n{ipc.plan_text(row)}" for row in rows))
        layout.addWidget(text)
        close = _button("Close")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, 0, Qt.AlignRight)
        dialog.resize(720, 520)
        dialog.show()

    def _open_path(self, path, make=False):
        open_path(path, make)


def build_images_page(owner, window, host):
    """The Progress Manager's Interactive Images tab (``window._pm_tab_factories['images']``): a per-file stack
    built lazily for the file selected in the shared dropdown, a 2 s fallback refresh that runs only while the
    window is visible and this tab is current, and the tab label count."""
    layout = host.layout()
    title = QLabel("Interactive Images")
    title.setStyleSheet("color: #c084fc; font-size: 11pt; font-weight: bold;")
    layout.addWidget(title)
    note = QLabel("ℹ️ Illustrations and style classes added after translation, per chapter. A chapter shows as stale "
                  "after it is retranslated; rebuild it from the context menu.")
    note.setWordWrap(True)
    note.setStyleSheet("color: #7a8a9e; font-size: 9pt;")
    layout.addWidget(note)
    stack = QStackedWidget(host)
    layout.addWidget(stack, 1)
    empty = QLabel("No translated output for this file yet")
    empty.setAlignment(Qt.AlignCenter)
    empty.setStyleSheet("color: #7a8a9e; font-size: 10pt; font-weight: bold;")
    stack.addWidget(empty)
    panels = {}
    host._ix_panels = panels
    host._ix_stack = stack

    def output_dir_for(fp):
        dirs = getattr(window, '_pm_images_output_dirs', {}) or {}
        return dirs.get(_pm_key(fp)) if fp is not None else None

    def on_stats(panel, stats):
        try:
            if getattr(window, '_pm_current_file', lambda: None)() is not None and \
                    panels.get(_pm_key(window._pm_current_file())) is panel:
                window._pm_set_tab_count('images', ipc.tab_count_text(stats))
        except RuntimeError:
            pass

    def show_file(fp=None):
        fp = fp if fp is not None else window._pm_current_file()
        folder = output_dir_for(fp)
        if not folder or not os.path.isdir(folder):
            stack.setCurrentWidget(empty)
            window._pm_set_tab_count('images')
            return None
        key = _pm_key(fp)
        panel = panels.get(key)
        if panel is None:
            panel = ImagesFilePanel(owner, window, folder, stack)
            panel.on_stats = on_stats
            panels[key] = panel
            stack.addWidget(panel)
        stack.setCurrentWidget(panel)
        panel.refresh()
        on_stats(panel, getattr(panel, 'stats', {}))
        return panel

    timer = QTimer(host)
    timer.setInterval(2000)

    def tick():
        try:
            if window.isVisible() and window._pm_tabs.currentWidget() is host:
                panel = stack.currentWidget()
                if isinstance(panel, ImagesFilePanel):
                    panel.refresh()
                    panel._update_buttons()
        except RuntimeError:
            timer.stop()

    timer.timeout.connect(tick)

    def on_shown():
        show_file()
        timer.start()

    def on_hidden():
        timer.stop()

    window._pm_tab_callbacks['images'] = (on_shown, on_hidden)

    def on_file_changed(fp):
        try:
            if window._pm_current_tab_name() == 'images':
                show_file(fp)
        except RuntimeError:
            pass

    window._pm_file_changed_callbacks.append(on_file_changed)
    host._ix_show_file = show_file
    host._ix_timer = timer
    show_file()
    if window.isVisible() and window._pm_tabs.currentWidget() is host:
        timer.start()
    return host


__all__ = [
    'DIALOG_STYLE', 'FALLBACK_NOTE', 'ImageViewerDialog', 'ImagesFilePanel', 'InteractiveImagesDialog',
    'InteractiveImagesGalleryDialog', 'InteractiveStylesDialog', 'PlaceholderTextDialog', 'RatioPopup',
    'ReferencesEditor', 'StylePreview', 'add_interactive_styles_rows', 'build_images_page', 'current_run',
    'open_gallery_dialog', 'open_path', 'open_placeholder_dialog', 'open_settings_dialog', 'open_styles_dialog',
    'run_bridge', 'start_run', 'stop_run', 'toggle_window',
]
