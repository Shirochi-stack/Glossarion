"""Parallel raw/translated EPUB pairing for glossary extraction.

The dialog in this module deliberately stops at preparing a paired EPUB.  The
main window then sends that temporary book through Glossarion's existing EPUB
glossary pipeline, so chapter batching, progress recovery, parsing, refinement,
and output handling all continue to use the established implementation.
"""

from __future__ import annotations

import html
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ebooklib import epub
from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_PARALLEL_EPUB_PROFILE = "Parallel EPUB Glossary"

DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT = """\
[RAW EPUB START — {raw_filename}]
{raw_text}
[RAW EPUB END]

[TRANSLATED EPUB START — {translated_filename}]
{translated_text}
[TRANSLATED EPUB END]"""

PARALLEL_EPUB_SYSTEM_INSTRUCTIONS = """\
PAIR-SPECIFIC INSTRUCTIONS:
- You are cross-checking aligned HTML chapters from a raw/source-language EPUB and an existing translated EPUB.
- Every user input contains one or more mapped raw/translated chapter pairs. Treat each RAW EPUB section as the authority for raw_name and its matching TRANSLATED EPUB section as the authority for established translated_name spellings.
- Cross-check both sections before creating each entry. When the translated section contains the matching name or term, copy its established rendering exactly; only translate or transliterate it yourself when no usable rendering exists there.
- Use the paired context to recover entries that one edition makes implicit, but never invent an entry or translation unsupported by either section.
- The pair-specific rules above take priority if the general glossary rules below would otherwise make you ignore the supplied translated edition."""


def default_parallel_epub_system_prompt() -> str:
    """Return pair instructions followed by the canonical prompt verbatim."""
    from extract_glossary_from_epub import DEFAULT_GLOSSARY_PROMPT

    return f"{PARALLEL_EPUB_SYSTEM_INSTRUCTIONS}\n\n{DEFAULT_GLOSSARY_PROMPT}"


def chapter_filename(chapter) -> str:
    """Read a chapter filename from extractor tuples or dialog dictionaries."""
    if isinstance(chapter, dict):
        return str(chapter.get("filename") or "")
    if isinstance(chapter, (tuple, list)) and len(chapter) >= 2:
        return str(chapter[1] or "")
    return ""


def chapter_text(chapter) -> str:
    """Read chapter text from extractor tuples or dialog dictionaries."""
    if isinstance(chapter, dict):
        return str(chapter.get("text") or "")
    if isinstance(chapter, (tuple, list)) and chapter:
        return str(chapter[0] or "")
    return str(chapter or "")


def _normalized_member_stem(filename: str) -> str:
    stem = Path(str(filename or "")).stem.casefold()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _member_number_signature(filename: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", Path(str(filename or "")).stem))


def auto_map_epub_chapters(
    raw_chapters: Sequence,
    translated_chapters: Sequence,
) -> List[Dict[str, object]]:
    """Create a stable one-to-one map using name, number, then reading order."""
    mappings: List[Dict[str, object]] = [
        {"raw_index": index, "translated_index": None, "strategy": "Unmatched"}
        for index in range(len(raw_chapters))
    ]
    available = set(range(len(translated_chapters)))

    def assign_unique(key_func: Callable[[str], object], strategy: str) -> None:
        raw_keys: Dict[object, List[int]] = {}
        translated_keys: Dict[object, List[int]] = {}
        for raw_index, mapping in enumerate(mappings):
            if mapping["translated_index"] is not None:
                continue
            key = key_func(chapter_filename(raw_chapters[raw_index]))
            if key:
                raw_keys.setdefault(key, []).append(raw_index)
        for translated_index in sorted(available):
            key = key_func(chapter_filename(translated_chapters[translated_index]))
            if key:
                translated_keys.setdefault(key, []).append(translated_index)
        for key, raw_indexes in raw_keys.items():
            translated_indexes = translated_keys.get(key, [])
            if len(raw_indexes) != 1 or len(translated_indexes) != 1:
                continue
            raw_index = raw_indexes[0]
            translated_index = translated_indexes[0]
            mappings[raw_index]["translated_index"] = translated_index
            mappings[raw_index]["strategy"] = strategy
            available.discard(translated_index)

    assign_unique(_normalized_member_stem, "Exact filename")
    assign_unique(_member_number_signature, "Chapter number")

    unmatched_raw = [
        index for index, mapping in enumerate(mappings)
        if mapping["translated_index"] is None
    ]
    for raw_index, translated_index in zip(unmatched_raw, sorted(available)):
        mappings[raw_index]["translated_index"] = translated_index
        mappings[raw_index]["strategy"] = "Reading order"
        available.discard(translated_index)

    return mappings


def apply_parallel_epub_wrapper(
    template: str,
    *,
    raw_text: str,
    translated_text: str,
    raw_filename: str,
    translated_filename: str,
) -> str:
    """Expand only supported placeholders, leaving unrelated braces intact."""
    result = str(template or "")
    replacements = {
        "{raw_text}": str(raw_text or ""),
        "{translated_text}": str(translated_text or ""),
        "{raw_filename}": str(raw_filename or ""),
        "{translated_filename}": str(translated_filename or ""),
    }
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def write_parallel_epub(
    output_path: str,
    pairs: Iterable[Dict[str, str]],
    wrapper_prompt: str,
    *,
    title: str = "Parallel EPUB Pair",
) -> str:
    """Write mapped chapter pairs as one valid EPUB for the shared extractor."""
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("At least one mapped HTML pair is required.")
    if "{raw_text}" not in wrapper_prompt or "{translated_text}" not in wrapper_prompt:
        raise ValueError(
            "The wrapper prompt must contain {raw_text} and {translated_text}."
        )

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    book = epub.EpubBook()
    book.set_identifier(f"glossarion-parallel-{uuid.uuid4().hex}")
    book.set_title(str(title or "Parallel EPUB Pair"))
    book.set_language("und")

    epub_chapters = []
    for index, pair in enumerate(pair_list, start=1):
        wrapped = apply_parallel_epub_wrapper(
            wrapper_prompt,
            raw_text=pair.get("raw_text", ""),
            translated_text=pair.get("translated_text", ""),
            raw_filename=pair.get("raw_filename", ""),
            translated_filename=pair.get("translated_filename", ""),
        )
        # A preformatted element preserves wrapper boundaries while escaping any
        # markup that appeared in source prose. The established EPUB extractor
        # will turn it back into plain text before sending it to the model.
        content = (
            "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head>"
            f"<title>Mapped pair {index}</title></head><body>"
            f"<pre style=\"white-space: pre-wrap\">{html.escape(wrapped)}</pre>"
            "</body></html>"
        )
        chapter = epub.EpubHtml(
            title=f"Mapped pair {index}",
            file_name=f"pair_{index:04d}.xhtml",
            lang="und",
        )
        chapter.content = content
        book.add_item(chapter)
        epub_chapters.append(chapter)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    # Keep the required EPUB navigation document out of the reading spine so a
    # user who enables "translate special files" still sends only mapped pairs
    # through glossary extraction.
    book.spine = list(epub_chapters)
    epub.write_epub(output_path, book, {})
    return output_path


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


class ParallelEpubPairDialog(QDialog):
    """Map a raw EPUB's HTML documents to an existing translated EPUB."""

    epubLoadFinished = Signal(str, str, int, object, str)

    def __init__(
        self,
        parent=None,
        *,
        config: Optional[dict] = None,
        chapter_loader: Optional[Callable[[str], Sequence]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Parallel EPUB Pair")
        self.resize(1440, 860)
        self.setMinimumSize(1050, 700)
        self.config = config if isinstance(config, dict) else {}
        self.chapter_loader = chapter_loader or self._default_chapter_loader
        self.raw_path = ""
        self.translated_path = ""
        self.raw_chapters: List[Dict[str, str]] = []
        self.translated_chapters: List[Dict[str, str]] = []
        self._auto_mapping: List[Dict[str, object]] = []
        self._loaded_profile = ""
        self.result_data: Optional[dict] = None
        self._load_serial = 0
        self._latest_load_serial = {"raw": 0, "translated": 0}
        self._active_load = None
        self._pending_loads = []
        self._translated_mapping_model = QStringListModel(self)
        self.epubLoadFinished.connect(self._finish_epub_load)

        saved_profiles = self.config.get("parallel_epub_glossary_profiles", {})
        self.profiles = dict(saved_profiles) if isinstance(saved_profiles, dict) else {}
        default_prompt = default_parallel_epub_system_prompt()
        if DEFAULT_PARALLEL_EPUB_PROFILE not in self.profiles:
            self.profiles[DEFAULT_PARALLEL_EPUB_PROFILE] = default_prompt

        self._build_ui()
        active = str(
            self.config.get("parallel_epub_glossary_active_profile")
            or DEFAULT_PARALLEL_EPUB_PROFILE
        )
        if active not in self.profiles:
            active = DEFAULT_PARALLEL_EPUB_PROFILE
        self.profile_combo.setCurrentText(active)
        self._load_profile(active)

    @staticmethod
    def _default_chapter_loader(path: str) -> Sequence:
        from extract_glossary_from_epub import extract_chapters_from_epub

        return extract_chapters_from_epub(path, return_metadata=True)

    def _build_ui(self):
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
        self.mapping_status.setStyleSheet("color: #9ba4b3;")
        mapping_header.addWidget(self.mapping_status, 1)
        self.auto_map_button = QPushButton("Auto-map Again")
        self.auto_map_button.clicked.connect(self._rebuild_mapping)
        mapping_header.addWidget(self.auto_map_button)
        mapping_layout.addLayout(mapping_header)

        self.mapping_table = QTableWidget(0, 3)
        self.mapping_table.setHorizontalHeaderLabels(
            ["Raw HTML (reading order)", "Translated HTML", "Match"]
        )
        self.mapping_table.setAlternatingRowColors(True)
        self.mapping_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mapping_table.verticalHeader().setVisible(False)
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        mapping_layout.addWidget(self.mapping_table, 1)

        prompt_group = QGroupBox("Parallel EPUB Glossary System Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
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
        self.content_splitter.setSizes([570, 830])

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

    def _load_epub(self, side: str, path: str):
        path = os.path.abspath(path)
        if not os.path.isfile(path) or not path.lower().endswith(".epub"):
            QMessageBox.warning(self, "Invalid EPUB", "Please choose an existing .epub file.")
            return
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
        self.mapping_status.setStyleSheet("color: #62a9e8;")
        self._refresh_load_controls()

        def load_in_background():
            chapters = []
            error = ""
            try:
                extracted = list(self.chapter_loader(path) or [])
                for index, item in enumerate(extracted, start=1):
                    text = chapter_text(item)
                    if not text.strip():
                        continue
                    chapters.append(
                        {
                            "text": text,
                            "filename": chapter_filename(item) or f"HTML {index}",
                        }
                    )
                if not chapters:
                    error = "The EPUB has no eligible HTML files with readable text."
            except Exception as exc:
                error = str(exc)
            try:
                self.epubLoadFinished.emit(side, path, serial, chapters, error)
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
        if self._active_load == (side, path, serial):
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
            self._apply_loaded_epub(side, path, list(chapters or []))

        if self._pending_loads:
            next_side, next_path, next_serial = self._pending_loads.pop(0)
            self._start_epub_load(next_side, next_path, next_serial)
        else:
            self._refresh_load_controls()

    def _apply_loaded_epub(self, side: str, path: str, chapters):
        if side == "raw":
            self.raw_path = path
            self.raw_chapters = chapters
            self.raw_drop.set_epub(path, len(chapters))
        else:
            self.translated_path = path
            self.translated_chapters = chapters
            self.translated_drop.set_epub(path, len(chapters))
        self._rebuild_mapping()

    def _refresh_load_controls(self):
        loading = self._active_load is not None or bool(self._pending_loads)
        ready = bool(self.raw_chapters and self.translated_chapters)
        self.use_pair_button.setEnabled(ready and not loading)
        self.auto_map_button.setEnabled(ready and not loading)

    def _rebuild_mapping(self):
        self.mapping_table.setUpdatesEnabled(False)
        self.mapping_table.setRowCount(0)
        if not self.raw_chapters or not self.translated_chapters:
            self.mapping_status.setText("Load both EPUBs to create a map.")
            self.mapping_status.setStyleSheet("color: #9ba4b3;")
            self.mapping_table.setUpdatesEnabled(True)
            self._refresh_load_controls()
            return
        self._auto_mapping = auto_map_epub_chapters(
            self.raw_chapters, self.translated_chapters
        )
        self._translated_mapping_model.setStringList(
            ["— Unmapped —"]
            + [chapter["filename"] for chapter in self.translated_chapters]
        )
        self.mapping_table.setRowCount(len(self.raw_chapters))
        for row, raw in enumerate(self.raw_chapters):
            raw_item = QTableWidgetItem(raw["filename"])
            raw_item.setFlags(raw_item.flags() & ~Qt.ItemIsEditable)
            raw_item.setToolTip(raw["filename"])
            self.mapping_table.setItem(row, 0, raw_item)

            combo = QComboBox()
            # Every row shares one model. Creating a full copy of every
            # translated filename in every combo is O(n²) and was another
            # source of visible stalls for books with many HTML files.
            combo.setModel(self._translated_mapping_model)
            mapped_index = self._auto_mapping[row]["translated_index"]
            combo.setCurrentIndex(0 if mapped_index is None else int(mapped_index) + 1)
            combo.currentIndexChanged.connect(
                lambda _index, mapped_row=row: self._mapping_changed(mapped_row)
            )
            self.mapping_table.setCellWidget(row, 1, combo)

            strategy_item = QTableWidgetItem(str(self._auto_mapping[row]["strategy"]))
            strategy_item.setFlags(strategy_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(row, 2, strategy_item)
        self.mapping_table.setUpdatesEnabled(True)
        self._update_mapping_status()
        self._refresh_load_controls()

    def _mapping_changed(self, row: int):
        item = self.mapping_table.item(row, 2)
        if item is not None:
            item.setText("Manual")
        self._update_mapping_status()

    def _selected_mapping(self) -> List[Dict[str, int]]:
        selected = []
        for row in range(self.mapping_table.rowCount()):
            combo = self.mapping_table.cellWidget(row, 1)
            translated_index = combo.currentIndex() - 1 if combo is not None else -1
            if translated_index >= 0:
                selected.append({"raw_index": row, "translated_index": translated_index})
        return selected

    def _update_mapping_status(self):
        mapping = self._selected_mapping()
        used = [item["translated_index"] for item in mapping]
        duplicate_count = len(used) - len(set(used))
        unmatched_raw = len(self.raw_chapters) - len(mapping)
        unused_translated = len(self.translated_chapters) - len(set(used))
        parts = [f"{len(mapping)} mapped"]
        if unmatched_raw:
            parts.append(f"{unmatched_raw} raw unmatched")
        if unused_translated:
            parts.append(f"{unused_translated} translated unused")
        if duplicate_count:
            parts.append(f"{duplicate_count} duplicate assignment(s)")
            self.mapping_status.setStyleSheet("color: #ff7b7b;")
        else:
            self.mapping_status.setStyleSheet("color: #9ba4b3;")
        self.mapping_status.setText(" • ".join(parts))

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
        self.config["parallel_epub_glossary_profiles"] = dict(self.profiles)
        self.config["parallel_epub_glossary_active_profile"] = (
            self.profile_combo.currentText().strip() or DEFAULT_PARALLEL_EPUB_PROFILE
        )
        self.config["parallel_epub_glossary_wrapper_prompt"] = (
            self.wrapper_edit.toPlainText()
        )
        parent = self.parent()
        if parent is not None and hasattr(parent, "save_config"):
            try:
                parent.save_config(show_message=False)
            except Exception:
                pass

    def _accept_pair(self):
        if self._active_load is not None or self._pending_loads:
            QMessageBox.information(
                self,
                "EPUB Still Loading",
                "Wait for both EPUBs to finish loading before using the mapped pair.",
            )
            return
        if not self.raw_chapters or not self.translated_chapters:
            QMessageBox.warning(self, "EPUBs Required", "Load both the raw and translated EPUB.")
            return
        if os.path.normcase(os.path.abspath(self.raw_path)) == os.path.normcase(
            os.path.abspath(self.translated_path)
        ):
            QMessageBox.warning(
                self,
                "Two EPUBs Required",
                "Choose the source EPUB on the left and its translated edition on the right.",
            )
            return
        wrapper = self.wrapper_edit.toPlainText()
        if "{raw_text}" not in wrapper or "{translated_text}" not in wrapper:
            QMessageBox.warning(
                self,
                "Wrapper Placeholders Required",
                "The wrapper prompt must contain both {raw_text} and {translated_text}.",
            )
            return
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        if not system_prompt:
            QMessageBox.warning(self, "System Prompt Required", "The system prompt cannot be empty.")
            return
        mapping = self._selected_mapping()
        if not mapping:
            QMessageBox.warning(self, "Mapping Required", "Map at least one HTML file pair.")
            return
        translated_indexes = [item["translated_index"] for item in mapping]
        if len(translated_indexes) != len(set(translated_indexes)):
            QMessageBox.warning(
                self,
                "Duplicate Mapping",
                "Each translated HTML file can only be assigned once.",
            )
            return
        if len(mapping) < len(self.raw_chapters):
            answer = QMessageBox.question(
                self,
                "Unmatched Raw HTML",
                f"{len(self.raw_chapters) - len(mapping)} raw HTML file(s) are unmatched and will be skipped. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        pairs = []
        for item in mapping:
            raw = self.raw_chapters[item["raw_index"]]
            translated = self.translated_chapters[item["translated_index"]]
            pairs.append(
                {
                    "raw_filename": raw["filename"],
                    "raw_text": raw["text"],
                    "translated_filename": translated["filename"],
                    "translated_text": translated["text"],
                }
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
