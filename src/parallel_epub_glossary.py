"""Parallel raw/translated EPUB pairing for glossary extraction.

The dialog in this module deliberately stops at preparing a paired EPUB.  The
main window then sends that temporary book through Glossarion's existing EPUB
glossary pipeline, so chapter batching, progress recovery, parsing, refinement,
and output handling all continue to use the established implementation.

The mapping, prompt and EPUB-writing logic lives in the GUI-free
``parallel_epub_core`` (shared with the mobile app); every name it had here is
re-exported below, and the dialog keeps only widgets and calls into it.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from PySide6.QtCore import QRect, QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


from parallel_epub_core import (  # noqa: F401 - moved to the GUI-free core (U6); re-exported
    DEFAULT_PARALLEL_EPUB_PROFILE,
    DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT,
    PARALLEL_EPUB_SELECTION_CONFIG_KEY,
    PARALLEL_EPUB_SYSTEM_INSTRUCTIONS,
    _chapter_special_flags,
    _has_positive_member_number,
    _member_number_signature,
    _nonpositive_member_layout,
    _normalized_member_stem,
    active_parallel_epub_profile,
    apply_parallel_epub_wrapper,
    auto_map_epub_chapters,
    build_parallel_epub_pairs,
    chapter_filename,
    chapter_text,
    compact_parallel_epub_selection,
    default_parallel_epub_system_prompt,
    load_parallel_epub_chapters,
    load_parallel_epub_documents,
    offset_parallel_epub_mapping,
    parallel_epub_mapping_status,
    parallel_epub_profiles,
    parallel_epub_prompt_settings,
    parallel_epub_selection_matches,
    parallel_epub_working_filename,
    persisted_parallel_epub_rows,
    prepare_persisted_parallel_epub_selection,
    restore_parallel_epub_pairs,
    selected_parallel_epub_mapping,
    translated_mapping_label,
    unpaired_file_counts,
    unpaired_warning_text,
    valid_parallel_epub_rows,
    validate_parallel_epub_pair,
    write_parallel_epub,
)


class _EpubDropZone(QFrame):
    epubDropped = Signal(str)

    def __init__(self, heading: str, side_hint: str, accent: str, parent=None):
        super().__init__(parent)
        self._accent = accent
        self._side_hint = side_hint
        self._visual_state = "idle"
        self._hovered = False
        self.setAcceptDrops(True)
        self.setMinimumHeight(132)
        self.setObjectName("parallelEpubDropZone")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(5)
        title = QLabel(heading)
        title.setStyleSheet(f"font-weight: bold; color: {accent}; font-size: 11pt;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.hint_label = QLabel(side_hint)
        self.hint_label.setStyleSheet("color: #b8bec9;")
        self.hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hint_label)
        self.path_label = QLabel("Drop an .epub here")
        self.path_label.setWordWrap(True)
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self.path_label, 1)
        self.count_label = QLabel("")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("color: #8f98a8;")
        layout.addWidget(self.count_label)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setTextVisible(False)
        self.loading_bar.setFixedHeight(6)
        self.loading_bar.setStyleSheet(
            f"QProgressBar {{ border: 0; border-radius: 3px; background: #343a44; }}"
            f"QProgressBar::chunk {{ border-radius: 3px; background: {accent}; }}"
        )
        self.loading_bar.hide()
        layout.addWidget(self.loading_bar)
        self._refresh_visuals()

    def _refresh_visuals(self):
        if self._hovered:
            border_style = "solid"
            border_color = "#8bc7ff"
            background = "#29394a"
            self.hint_label.setText("Release to load this EPUB")
            self.hint_label.setStyleSheet(
                "color: #d9efff; font-weight: bold; font-size: 10pt;"
            )
        else:
            border_style = "dashed" if self._visual_state == "idle" else "solid"
            border_color = self._accent
            background = "#242424"
            if self._visual_state == "loading":
                background = "#282b32"
            elif self._visual_state == "loaded":
                background = "#242b31"
            elif self._visual_state == "error":
                border_color = "#e46767"
                background = "#322525"
            self.hint_label.setText(self._side_hint)
            self.hint_label.setStyleSheet("color: #b8bec9;")
        self.setStyleSheet(
            f"QFrame#parallelEpubDropZone {{ border: 3px {border_style} {border_color}; "
            f"border-radius: 8px; background: {background}; }}"
        )

    def set_hovered(self, hovered: bool):
        self._hovered = bool(hovered)
        self._refresh_visuals()

    @staticmethod
    def _epub_from_event(event) -> str:
        if not event.mimeData().hasUrls():
            return ""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path) and path.lower().endswith(".epub"):
                return os.path.abspath(path)
        return ""

    def dragEnterEvent(self, event):
        if self._epub_from_event(event):
            self.set_hovered(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.set_hovered(False)
        event.accept()

    def dragMoveEvent(self, event):
        if self._epub_from_event(event):
            self.set_hovered(True)
            event.acceptProposedAction()
        else:
            self.set_hovered(False)
            event.ignore()

    def dropEvent(self, event):
        path = self._epub_from_event(event)
        self.set_hovered(False)
        if path:
            self.epubDropped.emit(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def set_epub(self, path: str, chapter_count: int):
        self._visual_state = "loaded"
        self.path_label.setText(os.path.basename(path))
        self.path_label.setToolTip(path)
        self.count_label.setText(f"{chapter_count} eligible HTML file(s)")
        self.loading_bar.hide()
        self._refresh_visuals()

    def set_loading(self, path: str, *, queued: bool = False):
        self._visual_state = "loading"
        self.path_label.setText(os.path.basename(path))
        self.path_label.setToolTip(path)
        self.count_label.setText(
            "Waiting for the other EPUB…"
            if queued
            else "Reading HTML files in the background…"
        )
        self.loading_bar.show()
        self._refresh_visuals()

    def set_error(self, path: str):
        self._visual_state = "error"
        self.path_label.setText(os.path.basename(path))
        self.path_label.setToolTip(path)
        self.count_label.setText("Could not read this EPUB")
        self.loading_bar.hide()
        self._refresh_visuals()

    def clear_epub(self):
        """Return the drop zone to its empty state."""

        self._visual_state = "idle"
        self._hovered = False
        self.path_label.setText("Drop an .epub here")
        self.path_label.setToolTip("")
        self.count_label.setText("")
        self.loading_bar.hide()
        self._refresh_visuals()


class _MappingComboDelegate(QStyledItemDelegate):
    """Paint lightweight dropdown cells and create one combo only while editing."""

    def __init__(self, dialog):
        super().__init__(dialog.mapping_table)
        self.dialog = dialog
        icon_path = Path(__file__).with_name("Halgakos.ico")
        self.arrow_icon = QIcon(str(icon_path)) if icon_path.is_file() else QIcon()

    def paint(self, painter, option, index):
        text_option = QStyleOptionViewItem(option)
        text_option.rect.adjust(0, 0, -26, 0)
        super().paint(painter, text_option, index)
        painter.save()
        divider_x = option.rect.right() - 25
        painter.setPen(QColor("#4a5568"))
        painter.drawLine(
            divider_x,
            option.rect.top() + 2,
            divider_x,
            option.rect.bottom() - 2,
        )
        if not self.arrow_icon.isNull():
            # QIcon.pixmap() may return a high-DPI backing pixmap whose
            # physical height is larger than its rendered logical height.
            # Center a logical target rect instead and let QIcon paint it.
            self.arrow_icon.paint(
                painter,
                self._arrow_rect(option.rect, divider_x),
                Qt.AlignCenter,
            )
        painter.restore()

    @staticmethod
    def _arrow_rect(cell_rect: QRect, divider_x: int) -> QRect:
        """Return a DPI-independent icon rectangle centered in the table row."""

        icon_size = min(16, max(1, cell_rect.height() - 2))
        arrow_area_width = 24
        x = divider_x + 1 + (arrow_area_width - icon_size) // 2
        y = cell_rect.top() + (cell_rect.height() - icon_size) // 2
        return QRect(x, y, icon_size, icon_size)

    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        self.dialog._configure_mapping_combo(editor)
        editor.setModel(self.dialog._translated_mapping_model)
        editor.activated.connect(lambda _value: self._commit_and_close(editor))
        QTimer.singleShot(0, lambda: self._show_popup(editor))
        return editor

    @staticmethod
    def _show_popup(editor):
        """Open a newly installed combo editor on the initiating click."""

        try:
            editor.showPopup()
        except RuntimeError:
            pass

    def setEditorData(self, editor, index):
        translated_index = index.data(Qt.UserRole)
        try:
            translated_index = int(translated_index)
        except (TypeError, ValueError):
            translated_index = -1
        editor.setCurrentIndex(translated_index + 1)

    def setModelData(self, editor, model, index):
        translated_index = editor.currentIndex() - 1
        model.setData(index, translated_index, Qt.UserRole)
        model.setData(
            index,
            self.dialog._translated_mapping_label(translated_index),
            Qt.DisplayRole,
        )
        self.dialog._mapping_changed(index.row())

    @staticmethod
    def updateEditorGeometry(editor, option, _index):
        editor.setGeometry(option.rect)

    def _commit_and_close(self, editor):
        self.commitData.emit(editor)
        self.closeEditor.emit(editor)


class ParallelEpubPairDialog(QDialog):
    """Map a raw EPUB's HTML documents to an existing translated EPUB."""

    epubLoadFinished = Signal(str, str, int, object, str)

    def __init__(
        self,
        parent=None,
        *,
        config: Optional[dict] = None,
        chapter_loader: Optional[Callable[[str], Sequence]] = None,
        special_file_predicate: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Parallel EPUB Pair")
        self.resize(1520, 860)
        self.setMinimumSize(1120, 700)
        self.config = config if isinstance(config, dict) else {}
        self.chapter_loader = chapter_loader or self._default_chapter_loader
        self.special_file_predicate = special_file_predicate
        self.raw_path = ""
        self.translated_path = ""
        self.raw_chapters: List[Dict[str, str]] = []
        self.translated_chapters: List[Dict[str, str]] = []
        self.raw_reading_order: List[str] = []
        self.translated_reading_order: List[str] = []
        self._auto_mapping: List[Dict[str, object]] = []
        self._mapping_offset = 0
        self._mapping_build_serial = 0
        self._mapping_building = False
        self._pending_persisted_selection: Optional[dict] = None
        self._loaded_profile = ""
        self.result_data: Optional[dict] = None
        self._load_serial = 0
        self._latest_load_serial = {"raw": 0, "translated": 0}
        self._active_load = None
        self._pending_loads = []
        self._translated_mapping_model = QStringListModel(self)
        self.epubLoadFinished.connect(self._finish_epub_load)

        self.profiles = parallel_epub_profiles(self.config)

        self._build_ui()
        active = active_parallel_epub_profile(self.config, self.profiles)
        self.profile_combo.setCurrentText(active)
        self._load_profile(active)
        self._refresh_load_controls()

    @staticmethod
    def _default_chapter_loader(path: str) -> Sequence:
        return load_parallel_epub_chapters(path)

    def _build_ui(self):
        mapping_combo_style = ""
        icon_path = Path(__file__).with_name("Halgakos.ico")
        if icon_path.is_file():
            icon_url = str(icon_path).replace("\\", "/")
            mapping_combo_style = f"""
                QComboBox#parallelMappingCombo,
                QComboBox#parallelPromptProfileCombo {{ padding-right: 4px; }}
                QComboBox#parallelMappingCombo::drop-down,
                QComboBox#parallelPromptProfileCombo::drop-down {{
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 18px;
                    border-left: 1px solid #4a5568;
                }}
                QComboBox#parallelMappingCombo::down-arrow,
                QComboBox#parallelPromptProfileCombo::down-arrow {{
                    image: url({icon_url});
                    width: 16px;
                    height: 16px;
                    border: none;
                }}
                QComboBox#parallelMappingCombo::down-arrow:on,
                QComboBox#parallelPromptProfileCombo::down-arrow:on {{ top: 1px; }}
            """
        self.setStyleSheet(
            "QDialog { background: #1f1f1f; color: white; }"
            "QGroupBox { border: 1px solid #49515e; border-radius: 6px; "
            "margin-top: 10px; padding-top: 8px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
            "QTextEdit, QComboBox, QTableWidget { background: #282828; color: white; "
            "border: 1px solid #505866; border-radius: 4px; }"
            "QHeaderView::section { background: #343a44; color: white; padding: 6px; "
            "border: 0; border-right: 1px solid #4b5360; }"
            "QPushButton { padding: 6px 12px; border-radius: 4px; background: #3b424d; color: white; }"
            "QPushButton:hover { background: #4a5361; }"
            + mapping_combo_style
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        intro = QLabel(
            "Pair the source novel with the translation you want to continue. "
            "Glossarion will cross-check each mapped HTML file while extracting the glossary."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #cbd2dc; font-size: 10pt;")
        root.addWidget(intro)

        # Use the dialog's width: controls remain on the left while the mapping
        # gets a dedicated, full-height pane on the right. The splitter lets
        # users choose the balance without making the table compete vertically
        # with both prompt editors.
        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.controls_panel = QWidget()
        self.controls_panel.setMinimumWidth(500)
        controls_layout = QVBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(0, 0, 6, 0)
        controls_layout.setSpacing(10)
        self.mapping_panel = QWidget()
        self.mapping_panel.setMinimumWidth(480)
        mapping_layout = QVBoxLayout(self.mapping_panel)
        mapping_layout.setContentsMargins(6, 0, 0, 0)
        mapping_layout.setSpacing(10)
        self.content_splitter.addWidget(self.controls_panel)
        self.content_splitter.addWidget(self.mapping_panel)
        self.content_splitter.setStretchFactor(0, 4)
        self.content_splitter.setStretchFactor(1, 6)
        root.addWidget(self.content_splitter, 1)

        wrapper_group = QGroupBox("Pair Wrapper Prompt")
        wrapper_layout = QVBoxLayout(wrapper_group)
        wrapper_help = QLabel(
            "Available placeholders: {raw_text}, {translated_text}, "
            "{raw_filename}, {translated_filename}"
        )
        wrapper_help.setWordWrap(True)
        wrapper_help.setStyleSheet("color: #62a9e8; font-weight: normal;")
        wrapper_layout.addWidget(wrapper_help)
        self.wrapper_edit = QTextEdit()
        self.wrapper_edit.setAcceptRichText(False)
        self.wrapper_edit.setMaximumHeight(145)
        self.wrapper_edit.setPlainText(
            str(
                self.config.get("parallel_epub_glossary_wrapper_prompt")
                or DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT
            )
        )
        wrapper_layout.addWidget(self.wrapper_edit)
        controls_layout.addWidget(wrapper_group)

        epub_row = QHBoxLayout()
        raw_column = QVBoxLayout()
        self.raw_drop = _EpubDropZone(
            "RAW EPUB", "Drag the source-language EPUB to the left", "#4aa3ff"
        )
        self.raw_drop.epubDropped.connect(lambda path: self._load_epub("raw", path))
        raw_column.addWidget(self.raw_drop)
        raw_browse = QPushButton("Browse Raw EPUB…")
        raw_browse.clicked.connect(lambda: self._browse_epub("raw"))
        raw_column.addWidget(raw_browse)
        epub_row.addLayout(raw_column, 1)

        translated_column = QVBoxLayout()
        self.translated_drop = _EpubDropZone(
            "TRANSLATED EPUB", "Drag the existing translation to the right", "#9b78ff"
        )
        self.translated_drop.epubDropped.connect(
            lambda path: self._load_epub("translated", path)
        )
        translated_column.addWidget(self.translated_drop)
        translated_browse = QPushButton("Browse Translated EPUB…")
        translated_browse.clicked.connect(lambda: self._browse_epub("translated"))
        translated_column.addWidget(translated_browse)
        epub_row.addLayout(translated_column, 1)
        controls_layout.addLayout(epub_row)

        mapping_header = QHBoxLayout()
        mapping_title = QLabel("HTML File Mapping")
        mapping_title.setStyleSheet("font-weight: bold; font-size: 10pt;")
        mapping_header.addWidget(mapping_title)
        self.mapping_status = QLabel("Load both EPUBs to create a map.")
        self.mapping_status.setStyleSheet("color: #9ba4b3; font-size: 8pt;")
        mapping_header.addWidget(self.mapping_status, 1)
        self.auto_offset_checkbox = self._create_auto_offset_checkbox()
        self.auto_offset_checkbox.setChecked(
            bool(self.config.get("parallel_epub_auto_offset_enabled", True))
        )
        self.auto_offset_checkbox.setToolTip(
            "Automatically keep unnumbered and zero-only files from shifting "
            "positive-numbered chapters. Turn off for plain reading-order mapping."
        )
        mapping_header.addWidget(self.auto_offset_checkbox)
        self.auto_map_button = QPushButton("Auto-map Again")
        self.auto_map_button.clicked.connect(self._rebuild_mapping)
        mapping_header.addWidget(self.auto_map_button)
        self.offset_down_button = QPushButton("\N{MINUS SIGN} Offset")
        self.offset_down_button.setToolTip(
            "Move every automatic mapped entry one raw row up."
        )
        self.offset_down_button.clicked.connect(
            lambda: self._apply_mapping_offset(-1)
        )
        mapping_header.addWidget(self.offset_down_button)
        self.offset_up_button = QPushButton("+ Offset")
        self.offset_up_button.setToolTip(
            "Move every automatic mapped entry one raw row down."
        )
        self.offset_up_button.clicked.connect(lambda: self._apply_mapping_offset(1))
        mapping_header.addWidget(self.offset_up_button)
        mapping_layout.addLayout(mapping_header)

        self.mapping_table = QTableWidget(0, 3)
        self.mapping_table.setHorizontalHeaderLabels(
            ["Raw HTML (reading order)", "Translated HTML", "Match"]
        )
        self.mapping_table.setAlternatingRowColors(True)
        self.mapping_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mapping_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.mapping_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mapping_table.customContextMenuRequested.connect(
            self._show_mapping_context_menu
        )
        self.mapping_table.setToolTip(
            "Select one or more rows, then right-click the Raw HTML column "
            "to set them all as unmapped."
        )
        self.mapping_table.verticalHeader().setVisible(False)
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._mapping_delegate = _MappingComboDelegate(self)
        self.mapping_table.setItemDelegateForColumn(1, self._mapping_delegate)
        self.mapping_table.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.mapping_table.cellClicked.connect(self._mapping_cell_clicked)
        mapping_layout.addWidget(self.mapping_table, 1)
        self.auto_offset_checkbox.toggled.connect(self._auto_offset_toggled)

        prompt_group = QGroupBox("Parallel EPUB Glossary System Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("parallelPromptProfileCombo")
        self.profile_combo.wheelEvent = lambda event: event.ignore()
        self.profile_combo.setToolTip(
            "Click to choose a prompt profile; the mouse wheel will not change it."
        )
        self.profile_combo.addItems(list(self.profiles))
        self.profile_combo.currentTextChanged.connect(self._load_profile)
        profile_row.addWidget(self.profile_combo, 1)
        new_profile = QPushButton("+ New Profile")
        new_profile.clicked.connect(self._new_profile)
        profile_row.addWidget(new_profile)
        save_profile = QPushButton("Save Profile")
        save_profile.clicked.connect(self._save_profile)
        profile_row.addWidget(save_profile)
        self.delete_profile_button = QPushButton("Reset Profile")
        self.delete_profile_button.clicked.connect(self._delete_or_reset_profile)
        profile_row.addWidget(self.delete_profile_button)
        prompt_layout.addLayout(profile_row)
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setAcceptRichText(False)
        self.system_prompt_edit.setMinimumHeight(190)
        prompt_layout.addWidget(self.system_prompt_edit)
        controls_layout.addWidget(prompt_group, 1)
        self.content_splitter.setSizes([570, 910])

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        button_row.addWidget(cancel)
        self.use_pair_button = QPushButton("Use Mapped Pair")
        self.use_pair_button.setStyleSheet(
            "QPushButton { background: #1878d1; color: white; font-weight: bold; padding: 8px 16px; }"
            "QPushButton:hover { background: #258ce8; }"
            "QPushButton:disabled { background: #343a44; color: #777f8d; }"
        )
        self.use_pair_button.clicked.connect(self._accept_pair)
        self.use_pair_button.setEnabled(False)
        button_row.addWidget(self.use_pair_button)
        root.addLayout(button_row)

    def _browse_epub(self, side: str):
        title = "Select Raw EPUB" if side == "raw" else "Select Translated EPUB"
        path, _ = QFileDialog.getOpenFileName(self, title, "", "EPUB files (*.epub)")
        if path:
            self._load_epub(side, path)

    def _create_auto_offset_checkbox(self) -> QCheckBox:
        """Reuse the app's standard checkmark toggle, with a standalone fallback."""

        parent = self.parent()
        factory = getattr(parent, "_create_styled_checkbox", None)
        if callable(factory):
            return factory("Auto Offset")

        checkbox = QCheckBox("Auto Offset")
        checkbox.setStyleSheet(
            """
            QCheckBox { color: white; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4;
                border-color: #5a9fd4;
            }
            QCheckBox::indicator:hover { border-color: #7bb3e0; }
            QCheckBox:disabled { color: #666666; }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
            """
        )
        checkmark = QLabel("\N{CHECK MARK}", checkbox)
        checkmark.setStyleSheet(
            "color: white; background: transparent; font-weight: bold; font-size: 11px;"
        )
        checkmark.setAlignment(Qt.AlignCenter)
        checkmark.hide()
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)

        def update_checkmark():
            try:
                checkmark.setGeometry(2, 1, 14, 14)
                checkmark.setVisible(checkbox.isChecked())
                if checkbox.isChecked():
                    checkmark.raise_()
            except RuntimeError:
                pass

        checkbox._checkmark_label = checkmark
        checkbox._update_checkmark = update_checkmark
        checkbox.stateChanged.connect(update_checkmark)
        QTimer.singleShot(0, update_checkmark)
        return checkbox

    def _load_epub(self, side: str, path: str):
        path = os.path.abspath(path)
        if not os.path.isfile(path) or not path.lower().endswith(".epub"):
            QMessageBox.warning(self, "Invalid EPUB", "Please choose an existing .epub file.")
            return
        pending = self._pending_persisted_selection
        if isinstance(pending, dict):
            saved_path = os.path.abspath(str(pending.get(f"{side}_path") or ""))
            if saved_path and os.path.normcase(path) != os.path.normcase(saved_path):
                # A newly chosen EPUB starts a new mapping. Do not apply an old
                # sidecar after its asynchronous table build finishes.
                self._pending_persisted_selection = None
        self._load_serial += 1
        serial = self._load_serial
        self._latest_load_serial[side] = serial
        self._pending_loads = [
            job for job in self._pending_loads if job[0] != side
        ]
        if self._active_load is not None:
            self._pending_loads.append((side, path, serial))
            self._drop_zone(side).set_loading(path, queued=True)
            self._refresh_load_controls()
            return
        self._start_epub_load(side, path, serial)

    def _drop_zone(self, side: str) -> _EpubDropZone:
        return self.raw_drop if side == "raw" else self.translated_drop

    def _start_epub_load(self, side: str, path: str, serial: int):
        self._active_load = (side, path, serial)
        self._drop_zone(side).set_loading(path)
        self.mapping_status.setText(
            f"Reading {'raw' if side == 'raw' else 'translated'} EPUB HTML in the background…"
        )
        self.mapping_status.setStyleSheet("color: #62a9e8; font-size: 8pt;")
        self._refresh_load_controls()

        def load_in_background():
            chapters, reading_order, error = load_parallel_epub_documents(
                self.chapter_loader, path
            )
            try:
                self.epubLoadFinished.emit(
                    side, path, serial,
                    {"chapters": chapters, "reading_order": reading_order}, error,
                )
            except RuntimeError:
                # The dialog was closed while the daemon loader was finishing.
                pass

        threading.Thread(
            target=load_in_background,
            name=f"ParallelEpubLoad-{side}-{serial}",
            daemon=True,
        ).start()

    def _finish_epub_load(
        self,
        side: str,
        path: str,
        serial: int,
        chapters,
        error: str,
    ):
        completed_active_load = self._active_load == (side, path, serial)
        if completed_active_load:
            self._active_load = None
        is_latest = self._latest_load_serial.get(side) == serial
        if is_latest and error:
            existing_path = self.raw_path if side == "raw" else self.translated_path
            existing_chapters = (
                self.raw_chapters if side == "raw" else self.translated_chapters
            )
            if existing_path and existing_chapters:
                self._drop_zone(side).set_epub(existing_path, len(existing_chapters))
            else:
                self._drop_zone(side).set_error(path)
            QMessageBox.warning(
                self,
                "Could not read EPUB",
                f"{path}\n\n{error}",
            )
        elif is_latest:
            if isinstance(chapters, dict):
                self._apply_loaded_epub(
                    side, path, list(chapters.get("chapters") or []),
                    reading_order=chapters.get("reading_order"),
                )
            else:
                self._apply_loaded_epub(side, path, list(chapters or []))

        # Only the loader that currently owns the serialized queue may advance
        # it. A stale loader invalidated by Clear Selection must not start a
        # newly queued job while another new loader is already active.
        if completed_active_load and self._pending_loads:
            next_side, next_path, next_serial = self._pending_loads.pop(0)
            self._start_epub_load(next_side, next_path, next_serial)
        elif completed_active_load or self._active_load is None:
            self._refresh_load_controls()

    def _apply_loaded_epub(self, side: str, path: str, chapters, reading_order=None):
        ordered_filenames = list(reading_order) if reading_order is not None else [
            chapter_filename(chapter) for chapter in chapters
        ]
        if side == "raw":
            self.raw_path = path
            self.raw_chapters = chapters
            self.raw_reading_order = ordered_filenames
            self.raw_drop.set_epub(path, len(chapters))
        else:
            self.translated_path = path
            self.translated_chapters = chapters
            self.translated_reading_order = ordered_filenames
            self.translated_drop.set_epub(path, len(chapters))
        self._rebuild_mapping()

    def clear_selection(self):
        """Clear both EPUBs and invalidate any background mapping work."""

        # The loaders are daemon threads and may already be reading ZIP data.
        # Advancing each serial makes their eventual signal stale, while
        # clearing the queue prevents the second old load from starting.
        self._load_serial += 1
        self._latest_load_serial = {
            "raw": self._load_serial,
            "translated": self._load_serial,
        }
        self._active_load = None
        self._pending_loads = []
        self._mapping_build_serial += 1
        self._mapping_building = False
        self._pending_persisted_selection = None
        self._mapping_offset = 0
        self._auto_mapping = []
        self.raw_path = ""
        self.translated_path = ""
        self.raw_chapters = []
        self.translated_chapters = []
        self.raw_reading_order = []
        self.translated_reading_order = []
        self.result_data = None
        self._translated_mapping_model.setStringList(["— Unmapped —"])
        self.mapping_table.setUpdatesEnabled(False)
        self.mapping_table.setRowCount(0)
        self.mapping_table.setUpdatesEnabled(True)
        self.mapping_table.verticalScrollBar().setValue(0)
        self.raw_drop.clear_epub()
        self.translated_drop.clear_epub()
        self.mapping_status.setText("Load both EPUBs to create a map.")
        self.mapping_status.setStyleSheet("color: #9ba4b3; font-size: 8pt;")
        self._refresh_load_controls()

    def restore_persisted_selection(self, selection: dict) -> bool:
        """Load both real EPUBs and restore a text-free saved HTML mapping."""

        pending = prepare_persisted_parallel_epub_selection(selection)
        if pending is None:
            return False
        raw_path = pending["raw_path"]
        translated_path = pending["translated_path"]

        self._pending_persisted_selection = pending

        wrapper_prompt = str(selection.get("wrapper_prompt") or "")
        if wrapper_prompt:
            self.wrapper_edit.setPlainText(wrapper_prompt)
        profile_name = str(selection.get("profile_name") or "").strip()
        if profile_name and self.profile_combo.findText(profile_name) >= 0:
            self.profile_combo.setCurrentText(profile_name)
        system_prompt = str(selection.get("system_prompt") or "")
        if system_prompt:
            # Restore the exact saved prompt. Profiles are not migrated or
            # rewritten while loading a mapping sidecar.
            self.system_prompt_edit.setPlainText(system_prompt)

        raw_loaded = (
            bool(self.raw_chapters)
            and os.path.normcase(os.path.abspath(self.raw_path))
            == os.path.normcase(raw_path)
        )
        translated_loaded = (
            bool(self.translated_chapters)
            and os.path.normcase(os.path.abspath(self.translated_path))
            == os.path.normcase(translated_path)
        )
        if raw_loaded and translated_loaded:
            self._rebuild_mapping()
        else:
            if not raw_loaded:
                self._load_epub("raw", raw_path)
            if not translated_loaded:
                self._load_epub("translated", translated_path)
        return True

    def _refresh_load_controls(self):
        loading = self._active_load is not None or bool(self._pending_loads)
        ready = bool(self.raw_chapters and self.translated_chapters)
        available = ready and not loading and not self._mapping_building
        self.use_pair_button.setText(
            "Mapping..." if self._mapping_building else "Use Mapped Pair"
        )
        self.use_pair_button.setEnabled(available)
        self.auto_map_button.setEnabled(available)
        self.offset_down_button.setEnabled(available)
        self.offset_up_button.setEnabled(available)
        self.auto_offset_checkbox.setEnabled(not loading and not self._mapping_building)

    def _rebuild_mapping(self):
        self._mapping_build_serial += 1
        build_serial = self._mapping_build_serial
        self._mapping_building = False
        self._mapping_offset = 0
        self.mapping_table.setUpdatesEnabled(False)
        self.mapping_table.setRowCount(0)
        if not self.raw_chapters or not self.translated_chapters:
            self.mapping_status.setText("Load both EPUBs to create a map.")
            self.mapping_status.setStyleSheet("color: #9ba4b3; font-size: 8pt;")
            self.mapping_table.setUpdatesEnabled(True)
            self._refresh_load_controls()
            return
        self._auto_mapping = auto_map_epub_chapters(
            self.raw_chapters,
            self.translated_chapters,
            enable_auto_offset=self.auto_offset_checkbox.isChecked(),
            special_file_predicate=self.special_file_predicate,
            protect_interior_special_files=bool(
                self.config.get('never_consider_in_between_files_as_special', True)
            ),
            raw_reading_order=self.raw_reading_order,
            translated_reading_order=self.translated_reading_order,
        )
        self._translated_mapping_model.setStringList(
            ["— Unmapped —"]
            + [chapter["filename"] for chapter in self.translated_chapters]
        )
        self.mapping_table.setRowCount(len(self.raw_chapters))
        # ResizeToContents recalculates the Match column after every inserted
        # row and makes large mappings needlessly slow. Freeze it throughout
        # population and perform one content-based resize after the last row.
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Fixed
        )
        self.mapping_table.setUpdatesEnabled(True)
        self._mapping_building = True
        self.mapping_status.setText(
            f"Building HTML mapping… 0/{len(self.raw_chapters)}"
        )
        self.mapping_status.setStyleSheet("color: #62a9e8; font-size: 8pt;")
        self._refresh_load_controls()
        QTimer.singleShot(
            0, lambda: self._populate_mapping_rows(build_serial, start_row=0)
        )

    def _populate_mapping_rows(self, build_serial: int, start_row: int):
        """Build lightweight mapping rows in short repaint-friendly slices."""

        if build_serial != self._mapping_build_serial:
            return
        deadline = time.perf_counter() + 0.012
        row = start_row
        total = len(self.raw_chapters)
        self.mapping_table.setUpdatesEnabled(False)
        while row < total and time.perf_counter() < deadline:
            raw = self.raw_chapters[row]
            raw_item = QTableWidgetItem(raw["filename"])
            raw_item.setFlags(raw_item.flags() & ~Qt.ItemIsEditable)
            raw_item.setToolTip(raw["filename"])
            self.mapping_table.setItem(row, 0, raw_item)

            mapped_index = self._auto_mapping[row]["translated_index"]
            mapped_index = -1 if mapped_index is None else int(mapped_index)
            translated_item = QTableWidgetItem(
                self._translated_mapping_label(mapped_index)
            )
            translated_item.setData(Qt.UserRole, mapped_index)
            translated_item.setToolTip(
                "Click to choose a translated HTML file."
            )
            self.mapping_table.setItem(row, 1, translated_item)

            strategy_item = QTableWidgetItem(str(self._auto_mapping[row]["strategy"]))
            strategy_item.setFlags(strategy_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(row, 2, strategy_item)
            row += 1
        self.mapping_table.setUpdatesEnabled(True)
        if row < total:
            self.mapping_status.setText(f"Building HTML mapping… {row}/{total}")
            QTimer.singleShot(
                0,
                lambda next_row=row: self._populate_mapping_rows(
                    build_serial, next_row
                ),
            )
            return
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._mapping_building = False
        self._apply_pending_persisted_mapping()
        self._update_mapping_status()
        self._refresh_load_controls()

    def _apply_pending_persisted_mapping(self) -> bool:
        """Apply saved choices after auto-mapping has finished building rows."""

        selection = self._pending_persisted_selection
        if not isinstance(selection, dict):
            return False
        if not parallel_epub_selection_matches(
            selection, self.raw_path, self.translated_path
        ):
            return False

        restored, _skipped = restore_parallel_epub_pairs(
            self.raw_chapters,
            self.translated_chapters,
            selection.get("mapping") or [],
        )
        rows = persisted_parallel_epub_rows(self.mapping_table.rowCount(), restored)
        header = self.mapping_table.horizontalHeader()
        self.mapping_table.setUpdatesEnabled(False)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        try:
            for row, (translated_index, strategy) in enumerate(rows):
                translated_item = self.mapping_table.item(row, 1)
                if translated_item is not None:
                    translated_item.setData(Qt.UserRole, translated_index)
                    translated_item.setText(
                        self._translated_mapping_label(translated_index)
                    )
                strategy_item = self.mapping_table.item(row, 2)
                if strategy_item is not None:
                    strategy_item.setText(strategy)
        finally:
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.mapping_table.setUpdatesEnabled(True)
            self.mapping_table.viewport().update()
        self._mapping_offset = 0
        self._pending_persisted_selection = None
        return True

    def _auto_offset_toggled(self, enabled: bool):
        """Persist the automatic offset preference and rebuild the mapping."""

        self.config["parallel_epub_auto_offset_enabled"] = bool(enabled)
        parent = self.parent()
        if parent is not None and hasattr(parent, "save_config"):
            try:
                parent.save_config(show_message=False)
            except Exception:
                pass
        self._rebuild_mapping()

    def _mapping_cell_clicked(self, row: int, column: int):
        """Open a translated-file dropdown immediately on a single click."""

        if column != 1 or self._mapping_building:
            return
        item = self.mapping_table.item(row, column)
        if item is not None:
            self.mapping_table.editItem(item)

    @staticmethod
    def _configure_mapping_combo(combo: QComboBox):
        """Apply Glossarion's mapping-combo wheel lock and arrow treatment."""

        # Ignoring the wheel event lets the containing mapping table continue
        # scrolling without silently changing the selected translated file.
        combo.setObjectName("parallelMappingCombo")
        combo.wheelEvent = lambda event: event.ignore()
        combo.setToolTip(
            "Click to select a translated HTML file; the mouse wheel scrolls the table."
        )

    def _translated_mapping_label(self, translated_index: int) -> str:
        return translated_mapping_label(self.translated_chapters, translated_index)

    def _apply_mapping_offset(self, delta: int):
        """Shift every automatic translated index, keeping overflow unmapped."""

        if not self._auto_mapping or not self.translated_chapters:
            return
        self._mapping_offset += int(delta)
        rows = offset_parallel_epub_mapping(
            self._auto_mapping,
            self._mapping_offset,
            self.translated_chapters,
            self.special_file_predicate,
            protect_interior=bool(
                self.config.get('never_consider_in_between_files_as_special', True)
            ),
            reading_order=self.translated_reading_order,
        )
        # Suspend table painting for the whole batch so hundreds of mapping
        # cells can be updated with a single final repaint.
        header = self.mapping_table.horizontalHeader()
        self.mapping_table.setUpdatesEnabled(False)
        # The Match column normally sizes itself to its contents. Temporarily
        # freeze it so changing every row does not trigger hundreds of full
        # column-width recalculations.
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        try:
            for row, (translated_index, strategy) in enumerate(rows):
                translated_item = self.mapping_table.item(row, 1)
                if translated_item is None:
                    continue
                translated_item.setData(Qt.UserRole, translated_index)
                translated_item.setText(
                    self._translated_mapping_label(translated_index)
                )
                strategy_item = self.mapping_table.item(row, 2)
                if strategy_item is not None:
                    strategy_item.setText(strategy)
        finally:
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.mapping_table.setUpdatesEnabled(True)
            self.mapping_table.viewport().update()
        self._update_mapping_status()

    def _mapping_changed(self, row: int):
        item = self.mapping_table.item(row, 2)
        if item is not None:
            item.setText("Manual")
        self._update_mapping_status()

    def _show_mapping_context_menu(self, position):
        """Offer batch actions when the Raw HTML column is right-clicked."""

        if self._mapping_building:
            return
        index = self.mapping_table.indexAt(position)
        if not index.isValid() or index.column() != 0:
            return
        clicked_row = index.row()
        selected_rows = sorted(
            {
                selected.row()
                for selected in self.mapping_table.selectionModel().selectedRows(0)
            }
        )
        if clicked_row not in selected_rows:
            self.mapping_table.clearSelection()
            self.mapping_table.selectRow(clicked_row)
            selected_rows = [clicked_row]

        menu = QMenu(self.mapping_table)
        label = (
            f"Set {len(selected_rows)} Selected Rows as Unmapped"
            if len(selected_rows) > 1
            else "Set This Row as Unmapped"
        )
        unmap_action = menu.addAction(label)
        unmap_action.triggered.connect(
            lambda: self._set_rows_unmapped(selected_rows)
        )
        menu.exec(self.mapping_table.viewport().mapToGlobal(position))

    def _set_rows_unmapped(self, rows: Iterable[int]):
        """Set several mapping cells to Unmapped in one repaint-safe batch."""

        valid_rows = valid_parallel_epub_rows(rows, self.mapping_table.rowCount())
        if not valid_rows:
            return
        header = self.mapping_table.horizontalHeader()
        self.mapping_table.setUpdatesEnabled(False)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        try:
            for row in valid_rows:
                translated_item = self.mapping_table.item(row, 1)
                if translated_item is None:
                    continue
                translated_item.setData(Qt.UserRole, -1)
                translated_item.setText(self._translated_mapping_label(-1))
                strategy_item = self.mapping_table.item(row, 2)
                if strategy_item is not None:
                    strategy_item.setText("Manual — Unmapped")
        finally:
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.mapping_table.setUpdatesEnabled(True)
            self.mapping_table.viewport().update()
        self._update_mapping_status()

    def _selected_mapping(self) -> List[Dict[str, int]]:
        translated_indexes = []
        for row in range(self.mapping_table.rowCount()):
            translated_item = self.mapping_table.item(row, 1)
            translated_indexes.append(
                translated_item.data(Qt.UserRole)
                if translated_item is not None
                else -1
            )
        return selected_parallel_epub_mapping(translated_indexes)

    def _unpaired_file_counts(self, mapping: Sequence[Dict[str, int]]) -> tuple:
        """Return unmatched raw and unused translated document counts."""

        return unpaired_file_counts(
            mapping, len(self.raw_chapters), len(self.translated_chapters)
        )

    def _unpaired_warning_text(self, mapping: Sequence[Dict[str, int]]) -> str:
        """Explain every individual HTML document excluded from the pair."""

        return unpaired_warning_text(
            mapping, len(self.raw_chapters), len(self.translated_chapters)
        )

    def _create_centered_question_box(self, title: str, text: str) -> QMessageBox:
        """Build a consistent centered Yes/No confirmation dialog."""

        message_box = QMessageBox(self)
        message_box.setWindowTitle(title)
        message_box.setIcon(QMessageBox.Question)
        message_box.setText(text)
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        message_box.setDefaultButton(QMessageBox.No)
        message_box.setEscapeButton(QMessageBox.No)
        button_box = message_box.findChild(QDialogButtonBox)
        if button_box is not None:
            button_box.setCenterButtons(True)
            if button_box.layout() is not None:
                button_box.layout().setSpacing(20)
        for standard_button in (QMessageBox.Yes, QMessageBox.No):
            button = message_box.button(standard_button)
            if button is not None:
                button.setMinimumSize(120, 46)
        return message_box

    def _create_unpaired_warning_box(
        self, mapping: Sequence[Dict[str, int]]
    ) -> QMessageBox:
        """Build the centered unpaired-files confirmation."""

        return self._create_centered_question_box(
            "Unmapped HTML Files",
            self._unpaired_warning_text(mapping),
        )

    def _update_mapping_status(self):
        mapping = self._selected_mapping()
        status_text, duplicate_count = parallel_epub_mapping_status(
            mapping,
            self._mapping_offset,
            self._auto_mapping,
            len(self.raw_chapters),
            len(self.translated_chapters),
        )
        if duplicate_count:
            self.mapping_status.setStyleSheet("color: #ff7b7b; font-size: 8pt;")
        else:
            self.mapping_status.setStyleSheet("color: #9ba4b3; font-size: 8pt;")
        self.mapping_status.setText(status_text)
        self.mapping_status.setToolTip(status_text)

    def _load_profile(self, name: str):
        if not name or name not in self.profiles:
            return
        self._loaded_profile = name
        self.system_prompt_edit.setPlainText(str(self.profiles.get(name) or ""))
        is_default = name == DEFAULT_PARALLEL_EPUB_PROFILE
        self.delete_profile_button.setText("Reset Profile" if is_default else "Delete Profile")

    def _new_profile(self):
        name, accepted = QInputDialog.getText(self, "New Profile", "Profile name:")
        name = str(name or "").strip()
        if not accepted or not name:
            return
        if name in self.profiles:
            QMessageBox.warning(self, "Profile Exists", f"A profile named '{name}' already exists.")
            return
        self.profiles[name] = self.system_prompt_edit.toPlainText()
        self.profile_combo.addItem(name)
        self.profile_combo.setCurrentText(name)
        self._persist_prompt_settings()

    def _save_profile(self):
        name = self.profile_combo.currentText().strip()
        if not name:
            return
        self.profiles[name] = self.system_prompt_edit.toPlainText()
        self._loaded_profile = name
        self._persist_prompt_settings()

    def _delete_or_reset_profile(self):
        name = self.profile_combo.currentText().strip()
        if not name:
            return
        if name == DEFAULT_PARALLEL_EPUB_PROFILE:
            answer = self._create_centered_question_box(
                "Reset Profile",
                "Reset the built-in Parallel EPUB Glossary profile?\n\n"
                "The current prompt text will be replaced with the default "
                "pair-specific and glossary extraction instructions.",
            ).exec()
            if answer != QMessageBox.Yes:
                return
            self.profiles[name] = default_parallel_epub_system_prompt()
            self.system_prompt_edit.setPlainText(self.profiles[name])
        else:
            answer = QMessageBox.question(
                self,
                "Delete Profile",
                f"Delete the profile '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            del self.profiles[name]
            index = self.profile_combo.findText(name)
            if index >= 0:
                self.profile_combo.removeItem(index)
            self.profile_combo.setCurrentText(DEFAULT_PARALLEL_EPUB_PROFILE)
        self._persist_prompt_settings()

    def _persist_prompt_settings(self):
        self.config.update(
            parallel_epub_prompt_settings(
                self.profiles,
                self.profile_combo.currentText(),
                self.wrapper_edit.toPlainText(),
            )
        )
        parent = self.parent()
        if parent is not None and hasattr(parent, "save_config"):
            try:
                parent.save_config(show_message=False)
            except Exception:
                pass

    def _accept_pair(self):
        wrapper = self.wrapper_edit.toPlainText()
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        mapping = self._selected_mapping()
        problem = validate_parallel_epub_pair(
            loading=self._active_load is not None or bool(self._pending_loads),
            raw_path=self.raw_path,
            translated_path=self.translated_path,
            raw_chapters=self.raw_chapters,
            translated_chapters=self.translated_chapters,
            wrapper_prompt=wrapper,
            system_prompt=system_prompt,
            mapping=mapping,
        )
        if problem is not None:
            kind, title, text = problem
            if kind == "information":
                QMessageBox.information(self, title, text)
            else:
                QMessageBox.warning(self, title, text)
            return
        unmatched_raw, unused_translated = self._unpaired_file_counts(mapping)
        if unmatched_raw or unused_translated:
            answer = self._create_unpaired_warning_box(mapping).exec()
            if answer != QMessageBox.Yes:
                return

        pairs = build_parallel_epub_pairs(
            mapping, self.raw_chapters, self.translated_chapters
        )

        profile_name = self.profile_combo.currentText().strip() or DEFAULT_PARALLEL_EPUB_PROFILE
        self.profiles[profile_name] = system_prompt
        self.config["parallel_epub_glossary_last_raw_epub"] = self.raw_path
        self.config["parallel_epub_glossary_last_translated_epub"] = self.translated_path
        self._persist_prompt_settings()
        self.result_data = {
            "raw_path": self.raw_path,
            "translated_path": self.translated_path,
            "pairs": pairs,
            "wrapper_prompt": wrapper,
            "system_prompt": system_prompt,
            "profile_name": profile_name,
        }
        self.accept()
