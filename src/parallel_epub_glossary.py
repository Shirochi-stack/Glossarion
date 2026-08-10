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
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ebooklib import epub
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


DEFAULT_PARALLEL_EPUB_PROFILE = "Parallel EPUB Glossary"
PARALLEL_EPUB_SELECTION_CONFIG_KEY = "parallel_epub_pair_selection"

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
- Cross-check both sections before creating each entry. Only output an entry when its matching established rendering is present in the translated section, and copy that rendering exactly. If the translated section does not provide a verifiable matching rendering, skip the entry entirely; never invent one yourself.
- Use the paired context to recover entries that one edition makes implicit, but never invent an entry or translation unsupported by either section.
- The pair-specific rules above take priority if the general glossary rules below would otherwise make you ignore the supplied translated edition."""


def parallel_epub_working_filename(raw_path: str) -> str:
    """Keep the raw EPUB basename so glossary output uses its normal folder."""

    source_name = os.path.basename(str(raw_path or "").strip())
    stem, extension = os.path.splitext(source_name)
    if not stem:
        stem = "raw_epub"
    if extension.lower() != ".epub":
        return f"{stem}.epub"
    return source_name


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


def compact_parallel_epub_selection(result: dict) -> dict:
    """Return the persistent, text-free representation of a mapped pair."""

    mappings = []
    for ordinal, pair in enumerate(result.get("pairs") or []):
        if not isinstance(pair, dict):
            continue
        raw_filename = str(pair.get("raw_filename") or "")
        translated_filename = str(pair.get("translated_filename") or "")
        if not raw_filename or not translated_filename:
            continue
        try:
            raw_index = int(pair.get("raw_index", ordinal))
        except (TypeError, ValueError):
            raw_index = ordinal
        try:
            translated_index = int(pair.get("translated_index", ordinal))
        except (TypeError, ValueError):
            translated_index = ordinal
        mappings.append(
            {
                "raw_index": raw_index,
                "translated_index": translated_index,
                "raw_filename": raw_filename,
                "translated_filename": translated_filename,
            }
        )

    return {
        "version": 1,
        "raw_path": os.path.abspath(str(result.get("raw_path") or "")),
        "translated_path": os.path.abspath(
            str(result.get("translated_path") or "")
        ),
        "mapping": mappings,
        "wrapper_prompt": str(result.get("wrapper_prompt") or ""),
        "system_prompt": str(result.get("system_prompt") or ""),
        "profile_name": str(
            result.get("profile_name") or DEFAULT_PARALLEL_EPUB_PROFILE
        ),
    }


def restore_parallel_epub_pairs(
    raw_chapters: Sequence,
    translated_chapters: Sequence,
    stored_mapping: Sequence,
) -> tuple[List[Dict[str, object]], int]:
    """Reattach saved filename mappings to freshly extracted chapter text.

    Stored indexes are used only when the filename at that index still agrees.
    Filename lookup is the fallback, so a harmless EPUB reading-order change does
    not destroy the saved mapping. Missing or duplicate references are skipped
    and returned as the second value.
    """

    raw_used = set()
    translated_used = set()

    def resolve_index(chapters, saved_index, saved_filename, used):
        expected = str(saved_filename or "")
        expected_key = expected.replace("\\", "/").casefold()
        try:
            candidate = int(saved_index)
        except (TypeError, ValueError):
            candidate = -1
        if (
            0 <= candidate < len(chapters)
            and candidate not in used
            and chapter_filename(chapters[candidate])
            .replace("\\", "/")
            .casefold()
            == expected_key
        ):
            return candidate
        for index, chapter in enumerate(chapters):
            if index in used:
                continue
            if (
                chapter_filename(chapter).replace("\\", "/").casefold()
                == expected_key
            ):
                return index
        return None

    restored = []
    skipped = 0
    for entry in stored_mapping or []:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        raw_index = resolve_index(
            raw_chapters,
            entry.get("raw_index"),
            entry.get("raw_filename"),
            raw_used,
        )
        translated_index = resolve_index(
            translated_chapters,
            entry.get("translated_index"),
            entry.get("translated_filename"),
            translated_used,
        )
        if raw_index is None or translated_index is None:
            skipped += 1
            continue
        raw_used.add(raw_index)
        translated_used.add(translated_index)
        restored.append(
            {
                "raw_index": raw_index,
                "translated_index": translated_index,
                "raw_filename": chapter_filename(raw_chapters[raw_index]),
                "raw_text": chapter_text(raw_chapters[raw_index]),
                "translated_filename": chapter_filename(
                    translated_chapters[translated_index]
                ),
                "translated_text": chapter_text(
                    translated_chapters[translated_index]
                ),
            }
        )
    return restored, skipped


def _normalized_member_stem(filename: str) -> str:
    stem = Path(str(filename or "")).stem.casefold()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _member_number_signature(filename: str) -> tuple:
    numbers = tuple(
        int(part) for part in re.findall(r"\d+", Path(str(filename or "")).stem)
    )
    # Zero-only names such as 0000_Information are front-matter offset
    # candidates, not chapter-number anchors.
    return numbers if any(number > 0 for number in numbers) else ()


def _has_positive_member_number(filename: str) -> bool:
    """Return whether a filename contains any numeric value greater than zero."""

    return bool(_member_number_signature(filename))


def _nonpositive_member_layout(chapters: Sequence) -> tuple:
    """Describe where no-number/zero-only files occur among numbered files."""

    positive_members_seen = 0
    layout = []
    for chapter in chapters:
        if _has_positive_member_number(chapter_filename(chapter)):
            positive_members_seen += 1
        else:
            layout.append(positive_members_seen)
    return tuple(layout)


def auto_map_epub_chapters(
    raw_chapters: Sequence,
    translated_chapters: Sequence,
    *,
    enable_auto_offset: bool = True,
) -> List[Dict[str, object]]:
    """Map names/numbers, isolating unnumbered and zero-only offset files."""
    mappings: List[Dict[str, object]] = [
        {
            "raw_index": index,
            "translated_index": None,
            "strategy": "Unmatched",
            "auto_offset": 0,
        }
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

    if enable_auto_offset:
        # No-number and zero-only documents deliberately stay out of every
        # automatic assignment, even when their stems match. They remain in
        # the UI for explicit manual selection.
        assign_unique(
            lambda filename: (
                _normalized_member_stem(filename)
                if _has_positive_member_number(filename)
                else ""
            ),
            "Exact filename",
        )
    else:
        assign_unique(_normalized_member_stem, "Exact filename")

    def assign_reading_group(numbered: bool, strategy: str) -> None:
        raw_indexes = [
            index
            for index, mapping in enumerate(mappings)
            if mapping["translated_index"] is None
            and _has_positive_member_number(chapter_filename(raw_chapters[index]))
            is numbered
        ]
        translated_indexes = [
            index
            for index in sorted(available)
            if _has_positive_member_number(
                chapter_filename(translated_chapters[index])
            )
            is numbered
        ]
        for raw_index, translated_index in zip(raw_indexes, translated_indexes):
            visual_offset = raw_index - translated_index if numbered else 0
            mappings[raw_index]["translated_index"] = translated_index
            mappings[raw_index]["auto_offset"] = visual_offset
            mappings[raw_index]["strategy"] = (
                f"Auto offset {visual_offset:+d}" if visual_offset else strategy
            )
            available.discard(translated_index)

    if enable_auto_offset:
        # When zero-only/unnumbered files occur at different positions, they
        # are the offset signal. Align positive-numbered reading sequences
        # first; raw_0002 can legitimately correspond to translated_0001.
        has_nonpositive_offset = _nonpositive_member_layout(
            raw_chapters
        ) != _nonpositive_member_layout(translated_chapters)
        if has_nonpositive_offset:
            assign_reading_group(True, "Numbered order")
        assign_unique(_member_number_signature, "Chapter number")
        if not has_nonpositive_offset:
            assign_reading_group(True, "Numbered order")

        # No-number/zero-only raw rows stay Unmatched and their translated
        # counterparts stay unused. This prevents front matter from entering
        # the paired glossary unless the user selects it manually.
    else:
        assign_unique(_member_number_signature, "Chapter number")
        unmatched_raw = [
            index
            for index, mapping in enumerate(mappings)
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
        self._auto_mapping: List[Dict[str, object]] = []
        self._mapping_offset = 0
        self._mapping_build_serial = 0
        self._mapping_building = False
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
        self._refresh_load_controls()

    @staticmethod
    def _default_chapter_loader(path: str) -> Sequence:
        from extract_glossary_from_epub import extract_chapters_from_epub

        return extract_chapters_from_epub(path, return_metadata=True)

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
            chapters = []
            error = ""
            try:
                extracted = list(self.chapter_loader(path) or [])
                for index, item in enumerate(extracted, start=1):
                    filename = chapter_filename(item) or f"HTML {index}"
                    if self.special_file_predicate and self.special_file_predicate(
                        filename
                    ):
                        continue
                    text = chapter_text(item)
                    if not text.strip():
                        continue
                    chapters.append(
                        {
                            "text": text,
                            "filename": filename,
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
        self._update_mapping_status()
        self._refresh_load_controls()

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
        if 0 <= translated_index < len(self.translated_chapters):
            return str(self.translated_chapters[translated_index]["filename"])
        return "— Unmapped —"

    def _apply_mapping_offset(self, delta: int):
        """Shift every automatic translated index, keeping overflow unmapped."""

        if not self._auto_mapping or not self.translated_chapters:
            return
        self._mapping_offset += int(delta)
        translated_count = len(self.translated_chapters)
        # Offset direction follows what the user sees in the raw-row table:
        # +1 moves the existing assignments down one raw row, so each row must
        # select the translated index that was previously one row above it.
        # Suspend table painting for the whole batch so hundreds of mapping
        # cells can be updated with a single final repaint.
        header = self.mapping_table.horizontalHeader()
        self.mapping_table.setUpdatesEnabled(False)
        # The Match column normally sizes itself to its contents. Temporarily
        # freeze it so changing every row does not trigger hundreds of full
        # column-width recalculations.
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        try:
            for row, automatic in enumerate(self._auto_mapping):
                translated_item = self.mapping_table.item(row, 1)
                if translated_item is None:
                    continue
                base_index = automatic.get("translated_index")
                shifted_index = (
                    None
                    if base_index is None
                    else int(base_index) - self._mapping_offset
                )
                if shifted_index is None or not 0 <= shifted_index < translated_count:
                    translated_index = -1
                    strategy = f"Offset {self._mapping_offset:+d} (unmapped)"
                else:
                    translated_index = shifted_index
                    strategy = f"Offset {self._mapping_offset:+d}"
                translated_item.setData(Qt.UserRole, translated_index)
                translated_item.setText(
                    self._translated_mapping_label(translated_index)
                )
                strategy_item = self.mapping_table.item(row, 2)
                if strategy_item is not None:
                    if self._mapping_offset:
                        strategy_item.setText(strategy)
                    else:
                        strategy_item.setText(
                            str(automatic.get("strategy") or "Unmatched")
                        )
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

        valid_rows = sorted(
            {
                int(row)
                for row in rows
                if 0 <= int(row) < self.mapping_table.rowCount()
            }
        )
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
        selected = []
        for row in range(self.mapping_table.rowCount()):
            translated_item = self.mapping_table.item(row, 1)
            translated_index = (
                translated_item.data(Qt.UserRole)
                if translated_item is not None
                else -1
            )
            try:
                translated_index = int(translated_index)
            except (TypeError, ValueError):
                translated_index = -1
            if translated_index >= 0:
                selected.append({"raw_index": row, "translated_index": translated_index})
        return selected

    def _unpaired_file_counts(self, mapping: Sequence[Dict[str, int]]) -> tuple:
        """Return unmatched raw and unused translated document counts."""

        used_translated = {item["translated_index"] for item in mapping}
        unmatched_raw = len(self.raw_chapters) - len(mapping)
        unused_translated = len(self.translated_chapters) - len(used_translated)
        return unmatched_raw, unused_translated

    def _unpaired_warning_text(self, mapping: Sequence[Dict[str, int]]) -> str:
        """Explain every individual HTML document excluded from the pair."""

        unmatched_raw, unused_translated = self._unpaired_file_counts(mapping)
        excluded_total = unmatched_raw + unused_translated
        return (
            f"{excluded_total} HTML file(s) are not part of a mapped pair and "
            "will be skipped.\n\n"
            f"Mapped raw/translated pairs: {len(mapping)}\n"
            f"Unmatched raw HTML files: {unmatched_raw}\n"
            f"Unused translated HTML files: {unused_translated}\n\n"
            "Unused translated files are the extra files that overflow beyond "
            "the available raw rows, or files that no raw row currently selects.\n\n"
            "Continue with only the mapped pairs?"
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
        used = [item["translated_index"] for item in mapping]
        duplicate_count = len(used) - len(set(used))
        unmatched_raw, unused_translated = self._unpaired_file_counts(mapping)
        parts = [f"{len(mapping)} mapped"]
        if self._mapping_offset:
            parts.append(f"offset {self._mapping_offset:+d}")
        else:
            automatic_offsets = sorted(
                {
                    int(item.get("auto_offset") or 0)
                    for item in self._auto_mapping
                    if int(item.get("auto_offset") or 0)
                }
            )
            if len(automatic_offsets) == 1:
                parts.append(f"auto offset {automatic_offsets[0]:+d}")
            elif automatic_offsets:
                parts.append("automatic numbering offsets")
        if unmatched_raw:
            parts.append(f"{unmatched_raw} raw unmatched")
        if unused_translated:
            parts.append(f"{unused_translated} translated unused")
        if duplicate_count:
            parts.append(f"{duplicate_count} duplicate assignment(s)")
            self.mapping_status.setStyleSheet("color: #ff7b7b; font-size: 8pt;")
        else:
            self.mapping_status.setStyleSheet("color: #9ba4b3; font-size: 8pt;")
        status_text = " • ".join(parts)
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
        unmatched_raw, unused_translated = self._unpaired_file_counts(mapping)
        if unmatched_raw or unused_translated:
            answer = self._create_unpaired_warning_box(mapping).exec()
            if answer != QMessageBox.Yes:
                return

        pairs = []
        for item in mapping:
            raw = self.raw_chapters[item["raw_index"]]
            translated = self.translated_chapters[item["translated_index"]]
            pairs.append(
                {
                    "raw_index": item["raw_index"],
                    "translated_index": item["translated_index"],
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
