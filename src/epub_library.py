# epub_library.py
"""
EPUB Library & Reader for Glossarion Desktop (Windows / macOS).

Provides:
  - scan_for_epubs(): recursively find .epub files across output dirs
  - EpubLibraryDialog: grid-card browser with cover thumbnails
  - EpubReaderDialog: simple in-app EPUB reader with TOC navigation
"""

import os
import re
import sys
import hashlib
import logging
import shutil
import subprocess
import tempfile
import platform
import traceback
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QLineEdit, QFrame, QSplitter, QTextBrowser,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy, QToolButton,
    QApplication, QMenu, QComboBox, QStackedWidget, QStyledItemDelegate,
    QStyle, QStyleOptionViewItem, QLayout, QFormLayout, QDialogButtonBox,
    QPlainTextEdit
)
from PySide6.QtCore import Qt, QSize, QRect, QRectF, Signal, Slot, QThread, QTimer, QSizeF, QPoint, QPointF, QUrl, QEventLoop, QAbstractAnimation, QPropertyAnimation, QEasingCurve, QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QPixmap, QFont, QFontMetrics, QIcon, QImage, QImageReader, QMovie, QCursor, QShortcut, QKeySequence, QTransform, QTextLayout, QTextOption, QPainter, QColor, QPen

from metadata_progress import is_metadata_progress_entry
from translation_artifacts import is_translation_artifact_progress_entry
from html_tag_entities import unescape_valid_html_tag_entities
from epub_package import find_epub_opf_member
from chapter_chunk_progress import (
    chunk_failure_summary,
    chunk_status_summary_text,
    effective_parent_status,
    ensure_chunk_entry_schema,
    is_multi_chunk_entry,
    reset_chunks_for_retranslation,
    sorted_chunk_items,
)
from chapter_display_numbering import (
    filename_chapter_number,
    nonreset_chapter_display_numbers,
)
# The GUI-free Library / Reader core (shared with Glossarion Mobile): registries, scans,
# resolvers and Book Details (library_core), covers (library_covers), the reader
# document (reader_doc) and the live stream (live_stream). Moved names are re-exported
# here so callers are unchanged; the Qt classes below inherit the moved methods from
# the mixins (listed first in their bases).
from library_core import (
    BookDetailsLoaderMixin,
    BookDetailsMixin,
    DualScanMixin,
    FORMAT_ALL,
    FORMAT_EPUB,
    FORMAT_HTML,
    FORMAT_IMAGE,
    FORMAT_PDF,
    FORMAT_TXT,
    LibraryDeleteMixin,
    LibraryShelfMixin,
    RawScanMixin,
    SIZE_2XL,
    SIZE_2XS,
    SIZE_3XL,
    SIZE_4XL,
    SIZE_5XL,
    SIZE_6XL,
    SIZE_COMPACT,
    SIZE_LARGE,
    SIZE_NORMAL,
    SIZE_XL,
    SIZE_XS,
    SORT_DATE,
    SORT_NAME,
    SORT_SIZE,
    ScanForRawMixin,
    _ALL_SIZES,
    _CARD_TYPE_BADGES,
    _CHAPTER_BADGE_STYLES,
    _CHAPTER_BADGE_TEXT,
    _CHAPTER_PRIMARY_STYLES,
    _DEFAULT_SPECIAL_FILE_EXACT,
    _DEFAULT_SPECIAL_FILE_KEYWORDS,
    _EDITABLE_BOOK_METADATA_FIELDS,
    _EPUB_CACHE_SCHEMA,
    _EPUB_SEARCH_METADATA_CACHE,
    _FILENAME_STRIP_CHARS,
    _LIBRARY_TAG_SEARCH_KEYS,
    _LIBRARY_TRACKING_FILENAMES,
    _MetadataEditError,
    _PROGRESS_SIDECAR_FILENAMES,
    _RE_HTML_HEADING,
    _RE_HTML_STRIP_TAGS,
    _RE_HTML_TITLE,
    _RE_HTML_WS,
    _SIZE_PRESETS,
    _SPINE_COUNT_CACHE,
    _attach_cross_location_duplicates,
    _book_library_tag_values,
    _book_library_title_values,
    _book_matches_library_query,
    _card_progress_view,
    _card_raw_title,
    _card_size_text,
    _card_type_badge,
    _chapter_completed_in_progress,
    _cleanup_incomplete_chapter_output,
    _count_epub_spine_items,
    _count_translated_response_files,
    _default_output_root,
    _detect_workspace_kind,
    _epub_cache_key,
    _expected_output_root_for_book,
    _extract_epub_search_metadata,
    _extract_epub_subjects,
    _extract_epub_titles,
    _extract_html_title_fast,
    _find_in_progress_novels,
    _find_raw_source_for_folder,
    _find_raw_source_for_library_epub,
    _folder_has_compiled_output,
    _folder_has_output_epub,
    _has_number_in_filename,
    _is_configured_special_file,
    _is_gallery_filename,
    _is_progress_sidecar_entry,
    _is_special_spine_item,
    _iter_library_search_values,
    _library_io_worker_count,
    _list_compiled_outputs,
    _load_origins,
    _mark_chapter_pending_for_retranslation,
    _merge_manual_metadata_edits,
    _metadata_changed_values,
    _metadata_subject_values,
    _migrate_legacy_library_layout,
    _norm_book_key,
    _origins_file,
    _origins_raw_sources_for_stem,
    _output_paths_equal,
    _page_bounds,
    _page_label,
    _parse_epub_details,
    _parse_special_file_list,
    _prepare_chapter_row_spec,
    _read_progress_summary,
    _read_source_epub_pointer,
    _read_translated_chapter_title,
    _reader_worker_count,
    _resolve_book_metadata_source,
    _resolve_book_output_folder,
    _resolve_book_source_file,
    _resolve_book_translated_file,
    _resolve_output_roots,
    _resolve_show_special_files,
    _resolve_special_file_lists,
    _resolve_translate_all_numbered,
    _resolve_translate_special_files,
    _save_origins,
    _special_file_settings_signature,
    _special_file_stem,
    _unique_dest,
    _validate_source_epub_for_workspace,
    _workspace_compile_kind,
    get_library_dir,
    get_library_raw_dir,
    get_library_raw_inputs_file,
    get_library_translated_dir,
    get_library_translated_inputs_file,
    load_library_raw_inputs,
    load_library_translated_inputs,
    record_library_raw_input,
    record_library_translated_input,
    remove_library_raw_input,
    remove_library_translated_input,
    scan_for_epubs,
    scan_library_completed,
    scan_output_folders,
    split_output_folders_by_status,
)
from library_covers import (
    CoverLoaderMixin,
    _PDF_COVER_SCAN_PAGE_LIMIT,
    _cover_cache_dir,
    _download_remote_cover_image,
    _extract_cover,
    _extract_pdf_cover,
    _find_cover_in_dir,
    _find_folder_cover,
    _find_halgakos_icon,
)
from reader_doc import (
    EpubCacheLoaderMixin,
    EpubLoaderMixin,
    EpubSearchMixin,
    LAYOUT_ALL,
    LAYOUT_DOUBLE,
    LAYOUT_SCROLL,
    LAYOUT_SINGLE,
    OverlayMergeMixin,
    ReaderDocMixin,
    ReaderImagePreloadMixin,
    WorkspaceReaderLoaderMixin,
    _LAZY_EPUB_IMAGE_TAG,
    _READER_GT_LANG_CODES,
    _READER_IMAGE_EXTS,
    _READER_THEMES,
    _chapter_display_numbers,
    _define_url,
    _discover_epub_image_members,
    _epub_cache_dir,
    _epub_plain_chapter_text,
    _epub_search_excerpt,
    _find_reader_sidecar,
    _google_translate_url,
    _lazy_epub_image,
    _lazy_epub_image_member,
    _load_epub_cache,
    _load_reader_native_toc,
    _map_native_toc_to_chapters,
    _native_toc_target_key,
    _parse_native_toc_ncx,
    _parse_native_toc_txt,
    _read_epub_member_from_zip,
    _reader_file_image_resource,
    _reader_image_cache_path,
    _reader_image_candidates,
    _reader_image_is_sizeable,
    _reader_image_map_signature,
    _reader_image_resource,
    _reader_overlay_signature,
    _save_epub_cache,
    _target_lang_to_google_code,
    _url_scheme,
    _workspace_reader_placeholder,
    _write_reader_image_cache,
)
from live_stream import (
    LiveStreamMixin,
    live_outcome_text,
)

try:
    import dpi_setup
    dpi_setup.install_qt_message_filter()
except Exception:
    pass

# Use QWebEngineView for full CSS support (images, block layout, etc.)
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    _HAS_WEBENGINE = True
except ImportError:
    _HAS_WEBENGINE = False

logger = logging.getLogger(__name__)

_ORPHANED_QTHREADS: set[QThread] = set()


def _configure_epub_reader_web_settings(view) -> bool:
    """Allow local reader pages to render remote HTTP(S) images."""
    if not _HAS_WEBENGINE or view is None:
        return False
    try:
        settings = view.settings()
        settings.setAttribute(QWebEngineSettings.AutoLoadImages, True)
        settings.setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls,
            True,
        )
        return True
    except Exception:
        logger.debug(
            "Could not enable remote images for EPUB reader: %s",
            traceback.format_exc(),
        )
        return False


class _FlowLayout(QLayout):
    """Compact wrapping layout used by metadata tag chips."""

    def __init__(self, parent=None, horizontal_spacing=8, vertical_spacing=8):
        super().__init__(parent)
        self._items = []
        self._horizontal_spacing = int(horizontal_spacing)
        self._vertical_spacing = int(vertical_spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, max(0, int(width)), 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _do_layout(self, rect, test_only):
        left, top, right, bottom = self.getContentsMargins()
        available = rect.adjusted(left, top, -right, -bottom)
        x = available.x()
        y = available.y()
        line_height = 0

        for item in self._items:
            hint = item.sizeHint().expandedTo(item.minimumSize())
            next_x = x + hint.width()
            if (line_height > 0
                    and next_x > available.right() + 1):
                x = available.x()
                y += line_height + self._vertical_spacing
                next_x = x + hint.width()
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x + self._horizontal_spacing
            line_height = max(line_height, hint.height())

        return max(0, y + line_height - rect.y() + bottom)


def _disconnect_thread_signals(thread, signal_names=()) -> None:
    """Best-effort disconnect for worker signals targeting closing dialogs."""
    for name in signal_names or ():
        signal = getattr(thread, name, None)
        if signal is None:
            continue
        try:
            signal.disconnect()
        except Exception:
            pass


def _orphan_running_qthread(thread) -> None:
    """Keep a still-running QThread alive after its owner dialog closes."""
    if thread is None:
        return
    try:
        if not thread.isRunning():
            return
    except Exception:
        pass
    try:
        thread.setParent(None)
    except Exception:
        pass
    _ORPHANED_QTHREADS.add(thread)
    if getattr(thread, "_glossarion_orphaned", False):
        return
    try:
        setattr(thread, "_glossarion_orphaned", True)
    except Exception:
        pass

    def _cleanup(*_args, t=thread):
        _ORPHANED_QTHREADS.discard(t)
        try:
            t.deleteLater()
        except Exception:
            pass

    try:
        thread.finished.connect(_cleanup)
    except Exception:
        pass


def _stop_qthread_safely(thread, timeout_ms: int = 1200,
                         signal_names=(), cancel: bool = True) -> bool:
    """Request a QThread stop; detach it if it cannot finish immediately."""
    if thread is None:
        return True
    _disconnect_thread_signals(thread, signal_names)
    if cancel:
        cancel_fn = getattr(thread, "cancel", None)
        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:
                pass
        try:
            thread.requestInterruption()
        except Exception:
            pass
    try:
        thread.quit()
    except Exception:
        pass
    try:
        if not thread.isRunning():
            return True
    except Exception:
        pass
    try:
        stopped = thread.wait(max(0, int(timeout_ms)))
    except TypeError:
        thread.wait()
        stopped = True
    except Exception:
        stopped = False
    if stopped is False:
        _orphan_running_qthread(thread)
        return False
    return True

# -- Module-level render caches ---------------------------------------------
# Flipping the library's View dropdown (S / M / L / XL / …) invalidates the
# per-card ``preset_key`` and forces cards to rebuild. The rebuild itself is
# necessary, but the two hottest sub-operations are pure and cacheable:
#   1. scaling the same cover/fallback pixmap to the same (w, h)
#   2. fitting the same title / pill text into the same box
# Caching both removes most of the remaining latency after the cover-loader
# thread fix, especially on large libraries where many cards share a rebuild.
_RENDER_RESULT_CACHE_LIMIT = 4096
_TITLE_FIT_CACHE: dict[tuple, tuple[str, float]] = {}
_PILL_FONT_CACHE: dict[tuple, float] = {}
_SCALED_PIXMAP_CACHE_LIMIT = 512
_SCALED_PIXMAP_CACHE: dict[tuple[str, int, int], QPixmap] = {}
_BASE_PIXMAP_CACHE_LIMIT = 256
_BASE_PIXMAP_CACHE: dict[str, QPixmap] = {}

# _DEFAULT_SPECIAL_FILE_KEYWORDS, _DEFAULT_SPECIAL_FILE_EXACT, _parse_special_file_list, _resolve_special_file_lists, _special_file_settings_signature, _special_file_stem, _has_number_in_filename, _resolve_translate_all_numbered, _is_configured_special_file moved verbatim to library_core (imported above).

# _epub_plain_chapter_text, _epub_search_excerpt moved verbatim to reader_doc (imported above).


def _cache_put_bounded(cache: dict, key, value, limit: int) -> None:
    """Insert *key*/*value* into *cache*, evicting the oldest entry if full."""
    try:
        if key in cache:
            cache.pop(key, None)
        elif len(cache) >= int(limit):
            cache.pop(next(iter(cache)))
    except Exception:
        pass
    cache[key] = value


def _font_cache_key(font: QFont | None) -> str:
    """Return a stable, hashable key for *font* suitable for dict caches."""
    if font is None:
        return ""
    try:
        return font.toString()
    except Exception:
        try:
            return font.family()
        except Exception:
            return ""


def _cached_scaled_pixmap(path: str, width: int, height: int) -> QPixmap | None:
    """Return a cached smooth-scaled pixmap for *(path, width, height)*.

    Caches both the base loaded ``QPixmap`` and the scaled result so a card
    resize doesn't re-hit disk or re-run smooth scaling for covers / the
    fallback Halgakos icon that were already used once.
    """
    if not path or width <= 0 or height <= 0:
        return None
    try:
        norm_path = os.path.normcase(os.path.abspath(path))
    except Exception:
        norm_path = path
    scaled_key = (norm_path, int(width), int(height))
    cached_scaled = _SCALED_PIXMAP_CACHE.get(scaled_key)
    if cached_scaled is not None:
        return cached_scaled
    base = _BASE_PIXMAP_CACHE.get(norm_path)
    if base is None:
        try:
            base = QPixmap(norm_path)
        except Exception:
            return None
        if base is None or base.isNull():
            return None
        _cache_put_bounded(
            _BASE_PIXMAP_CACHE, norm_path, base, _BASE_PIXMAP_CACHE_LIMIT)
    scaled = base.scaled(
        int(width), int(height),
        Qt.KeepAspectRatio, Qt.SmoothTransformation)
    _cache_put_bounded(
        _SCALED_PIXMAP_CACHE, scaled_key, scaled,
        _SCALED_PIXMAP_CACHE_LIMIT)
    return scaled


def _animated_image_reader(path: str) -> QImageReader | None:
    """Return a reader only when *path* contains a genuinely animated image.

    Detection is content-based rather than extension-based because EPUB cover
    bytes are stored in Glossarion's legacy ``.jpg`` cache even when the
    embedded resource is actually a GIF.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        reader = QImageReader(path)
        if not reader.canRead() or not reader.supportsAnimation():
            return None
        # Some plugins report animation support for a single-frame resource.
        # ``-1`` means the count is not cheaply known, so allow that through.
        if reader.imageCount() == 1:
            return None
        return reader
    except Exception:
        return None


def _build_cover_movie(
    image_path: str,
    parent: QLabel,
    width: int,
    height: int,
) -> QMovie | None:
    """Build a movie whose frames are smooth-scaled into a cover label.

    ``QMovie.setScaledSize()`` delegates to the image plugin's inexpensive
    scaler, which makes downscaled GIF covers visibly jagged/blurry. Keep the
    decoder at native resolution and smooth-scale each decoded frame instead.
    """
    if width <= 0 or height <= 0:
        return None
    reader = _animated_image_reader(image_path)
    if reader is None:
        return None
    movie = None
    try:
        movie = QMovie(image_path, reader.format(), parent)
        if not movie.isValid():
            movie.deleteLater()
            return None
        target_size = QSize(int(width), int(height))

        def _paint_frame(_frame_number: int) -> None:
            try:
                frame = movie.currentImage()
                if frame.isNull():
                    return
                scaled = frame.scaled(
                    target_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                parent.setPixmap(QPixmap.fromImage(scaled))
                parent.setText("")
            except RuntimeError:
                # The label/movie can disappear together while a queued frame
                # notification is still being delivered during dialog close.
                pass

        movie.frameChanged.connect(_paint_frame)
        # Keep an explicit Python reference for the lifetime of the C++ movie.
        movie._glossarion_frame_renderer = _paint_frame
        return movie
    except Exception:
        if movie is not None:
            movie.deleteLater()
        logger.debug("Animated cover setup failed for %s: %s",
                     image_path, traceback.format_exc())
        return None


def _dispose_cover_movie(movie: QMovie | None) -> None:
    """Stop and release a movie that is no longer attached to a cover label."""
    if movie is None:
        return
    try:
        movie.stop()
        movie.deleteLater()
    except RuntimeError:
        pass


class _NoWheelComboBox(QComboBox):
    """QComboBox that ignores mouse-wheel scroll to prevent accidental changes.

    A plain QComboBox changes its current selection whenever the user
    scrolls the mouse wheel while the widget has focus (or just sits
    under the cursor). That's hostile inside a scrollable toolbar where
    the user wants to scroll the page or zoom cards — a stray wheel
    tick silently flips the active filter. Overriding ``wheelEvent`` to
    ignore the event keeps the popup + keyboard navigation intact but
    locks the combo against wheel-driven changes.
    """

    def wheelEvent(self, event):  # type: ignore[override]
        event.ignore()


# ---------------------------------------------------------------------------
# Paths & Utilities
# ---------------------------------------------------------------------------

# get_library_dir / get_library_raw_dir / get_library_translated_dir, the legacy layout
# migration, the raw / translated input registries and the origins mapping
# (_LIBRARY_TRACKING_FILENAMES .. _save_origins) moved verbatim to library_core
# (imported above; shared with Glossarion Mobile).


# _cover_cache_dir, _PDF_COVER_SCAN_PAGE_LIMIT, _extract_pdf_cover, _download_remote_cover_image moved verbatim to library_covers (imported above).


# _epub_cache_dir moved verbatim to reader_doc (imported above).


# _EPUB_CACHE_SCHEMA moved verbatim to library_core (imported above).
# _LAZY_EPUB_IMAGE_TAG, _READER_IMAGE_EXTS, _discover_epub_image_members, _lazy_epub_image, _lazy_epub_image_member, _url_scheme, _reader_image_candidates, _reader_image_resource, _reader_file_image_resource, _read_epub_member_from_zip, _reader_image_is_sizeable, _reader_image_cache_path, _write_reader_image_cache, _reader_image_map_signature moved verbatim to reader_doc (imported above).
# (_url_scheme replaces QUrl(src).scheme(); DISCREPANCIES U5 "Qt replacements".)


# _epub_cache_key moved verbatim to library_core (imported above).


# _load_epub_cache, _save_epub_cache moved verbatim to reader_doc (imported above).


# _find_halgakos_icon moved verbatim to library_covers (imported above).


def _find_translator_gui(widget):
    """Walk up the Qt parent chain to the main TranslatorGUI instance.

    Both :class:`BookDetailsDialog` (Library → Details) and
    :class:`EpubReaderDialog` ultimately hang off the main translator
    window, which exposes ``start_single_chapter_translation`` /
    ``run_translation_thread``. Returns ``None`` when no such ancestor
    exists (e.g. the reader was opened standalone).
    """
    w = widget
    for _ in range(16):
        if w is None:
            return None
        if (hasattr(w, "start_single_chapter_translation")
                and hasattr(w, "append_log")):
            return w
        try:
            w = w.parent()
        except Exception:
            return None
    return None


def _epub_converter_running(gui) -> bool:
    """True while the main GUI's EPUB or PDF compiler worker is active."""
    if gui is None:
        return False
    for future_name in ("epub_future", "pdf_future"):
        fut = getattr(gui, future_name, None)
        try:
            if fut is not None and not fut.done():
                return True
        except Exception:
            pass
    for thread_name in ("epub_thread", "pdf_thread"):
        th = getattr(gui, thread_name, None)
        try:
            if th is not None and th.is_alive():
                return True
        except Exception:
            pass
    return False


# _workspace_compile_kind, _mark_chapter_pending_for_retranslation, _chapter_completed_in_progress, _cleanup_incomplete_chapter_output moved verbatim to library_core (imported above).


# _extract_cover moved verbatim to library_covers (imported above).


def _open_folder_in_explorer(path: str):
    """Reveal *path* in the OS file manager (non-blocking).

    The actual spawn (``subprocess.Popen`` / ``os.startfile``) runs on
    a short-lived daemon thread because ``CreateProcess`` /
    ``ShellExecuteEx`` on Windows can stall the caller for 0.5–1 s
    while explorer.exe initialises COM and resolves the target. On
    the Qt main thread that shows up as a visible freeze between
    clicking a "Reveal source file" / "Open Output Folder" action and
    the Explorer window actually appearing. Off-loading the spawn
    means the click handler returns immediately and Qt can paint the
    next frame while Windows is still bringing Explorer up.

    On Windows the subprocess call also passes ``CREATE_NO_WINDOW``
    so the spawned helper doesn't flash a console window beside the
    cursor before Explorer paints its pane.
    """
    if not path:
        return
    # Snapshot the values the worker needs so it doesn't poke at
    # shared state from a background thread.
    _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        is_file = os.path.isfile(path)
    except OSError:
        is_file = False
    target_path = os.path.abspath(path)
    folder = os.path.dirname(target_path) if is_file else target_path
    normalized = os.path.normpath(target_path) if is_file else target_path
    system_name = platform.system()

    def _run_checked(command):
        """Run a desktop opener and report whether it accepted the path."""
        try:
            completed = subprocess.run(
                list(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if completed.returncode == 0:
                return True, ""
            return False, completed.stderr.strip() or f"{command[0]} exited with {completed.returncode}"
        except Exception as exc:
            return False, str(exc)

    def _open_linux():
        last_error = ""

        if is_file:
            reveal_commands = [
                ("dolphin", "--select", normalized),
                ("nautilus", "--select", normalized),
                ("thunar", normalized),
                ("nemo", normalized),
            ]
            for command in reveal_commands:
                if not shutil.which(command[0]):
                    continue
                ok, error = _run_checked(command)
                if ok:
                    return
                last_error = error

        desktop_openers = [
            ("xdg-open", folder),
            ("gio", "open", folder),
            ("kioclient6", "exec", folder),
            ("kioclient5", "exec", folder),
            ("kioclient", "exec", folder),
        ]
        for command in desktop_openers:
            if not shutil.which(command[0]):
                continue
            ok, error = _run_checked(command)
            if ok:
                return
            last_error = error

        for manager in ("dolphin", "thunar", "nemo", "nautilus", "pcmanfm", "caja"):
            if not shutil.which(manager):
                continue
            subprocess.Popen(
                [manager, folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        raise RuntimeError(last_error or "No Linux file manager opener was found. Install xdg-utils or a file manager.")

    def _worker():
        try:
            if system_name == "Windows":
                if is_file:
                    subprocess.Popen(
                        ["explorer", "/select,", normalized],
                        creationflags=_no_window,
                    )
                else:
                    os.startfile(folder)
            elif system_name == "Darwin":
                subprocess.Popen(
                    ["open", "-R", target_path] if is_file else ["open", folder]
                )
            else:
                _open_linux()
        except Exception as exc:
            logger.warning("Failed to open folder: %s\n%s",
                           exc, traceback.format_exc())

    import threading as _threading
    _threading.Thread(
        target=_worker,
        name="OpenFolderInExplorer",
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

# _library_io_worker_count, _reader_worker_count, _resolve_output_roots, _default_output_root, _expected_output_root_for_book, _output_paths_equal, _FILENAME_STRIP_CHARS, _EPUB_SEARCH_METADATA_CACHE, _extract_epub_search_metadata, _extract_epub_titles, _extract_epub_subjects, _LIBRARY_TAG_SEARCH_KEYS, _iter_library_search_values, _book_library_tag_values, _book_library_title_values, _book_matches_library_query, _norm_book_key, _is_gallery_filename, _PROGRESS_SIDECAR_FILENAMES, _is_progress_sidecar_entry, _read_progress_summary, _SPINE_COUNT_CACHE, _count_epub_spine_items, _resolve_translate_special_files, _resolve_show_special_files, _is_special_spine_item, _count_translated_response_files, _folder_has_output_epub, _folder_has_compiled_output, _list_compiled_outputs, _detect_workspace_kind moved verbatim to library_core (imported above).


# _read_source_epub_pointer, _origins_raw_sources_for_stem,
# _validate_source_epub_for_workspace and _find_raw_source_for_folder moved verbatim
# to library_core (imported above; the EPUB compile env resolves the source EPUB
# through them on desktop and mobile).


# _resolve_book_output_folder, _resolve_book_source_file, _resolve_book_metadata_source, _resolve_book_translated_file, _find_raw_source_for_library_epub moved verbatim to library_core (imported above).


# ---------------------------------------------------------------------------
# Cover helpers
# ---------------------------------------------------------------------------

# _find_cover_in_dir moved verbatim to library_covers (imported above).


# ---------------------------------------------------------------------------
# Tab scanners: Completed (Library) and In Progress (output folders)
# ---------------------------------------------------------------------------

# scan_library_completed, scan_output_folders, split_output_folders_by_status, _find_in_progress_novels, scan_for_epubs moved verbatim to library_core (imported above).


# ---------------------------------------------------------------------------
# Library Dialog — constants & helpers
# ---------------------------------------------------------------------------

# SORT_DATE, SORT_NAME, SORT_SIZE, FORMAT_ALL, FORMAT_EPUB, FORMAT_TXT, FORMAT_PDF, FORMAT_HTML, FORMAT_IMAGE, SIZE_2XS, SIZE_XS, SIZE_COMPACT, SIZE_NORMAL, SIZE_LARGE, SIZE_XL, SIZE_2XL, SIZE_3XL, SIZE_4XL, SIZE_5XL, SIZE_6XL, _ALL_SIZES, _SIZE_PRESETS moved verbatim to library_core (imported above).


def _parse_pt(pt_str) -> float:
    """Parse a CSS-like point-size string (e.g. ``"8.5pt"``) to a float."""
    try:
        return float(str(pt_str).replace("pt", "").strip())
    except (ValueError, TypeError):
        return 9.0


def _create_styled_checkbox(text: str = "", parent=None):
    """Build a QCheckBox styled identically to Other Settings / main GUI.

    Mirrors :meth:`TranslatorGUI._create_styled_checkbox` so every
    checkbox in the Library dialogs reads as a first-class citizen of
    the same app — same indicator border / fill colours, same “✓”
    label overlay that paints the checkmark (Qt’s default indicator
    rendering on Windows drops the tick at small sizes). The overlay
    is a click-through QLabel child positioned inside the indicator,
    shown / hidden in sync with the checkbox state.
    """
    from PySide6.QtWidgets import QCheckBox, QLabel

    checkbox = QCheckBox(text, parent)
    # Let the row/dialog background paint through the checkbox widget.
    # Without ``WA_TranslucentBackground`` Qt fills the QCheckBox's
    # bounding rect with the widget's base brush (a solid grey on
    # Windows' Fusion/Vista styles), so the extension toggles in the
    # Scan for Raw Sources dialog rendered as opaque boxes sitting on
    # top of the row. Combined with ``background: transparent;`` in
    # the stylesheet below this makes the label + indicator area
    # genuinely transparent — only the 14×14 indicator keeps a fill.
    checkbox.setAttribute(Qt.WA_TranslucentBackground)
    checkbox.setStyleSheet("""
        QCheckBox {
            color: white;
            spacing: 6px;
            background: transparent;
        }
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
        QCheckBox::indicator:hover {
            border-color: #7bb3e0;
        }
        QCheckBox:disabled {
            color: #666666;
        }
        QCheckBox::indicator:disabled {
            background-color: #1a1a1a;
            border-color: #3a3a3a;
        }
    """)

    checkmark = QLabel("\u2713", checkbox)
    checkmark.setStyleSheet(
        "QLabel { color: white; background: transparent; "
        "font-weight: bold; font-size: 11px; }"
    )
    checkmark.setAlignment(Qt.AlignCenter)
    checkmark.hide()
    checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)

    def _position_checkmark():
        try:
            checkmark.setGeometry(2, 1, 14, 14)
        except RuntimeError:
            # Underlying widget was destroyed
            pass

    def _update_checkmark():
        try:
            if checkbox.isChecked():
                _position_checkmark()
                checkmark.show()
            else:
                checkmark.hide()
        except RuntimeError:
            pass

    checkbox.stateChanged.connect(_update_checkmark)
    # Defer the first positioning until the widget has actually laid
    # out — otherwise ``setGeometry`` runs before the style sheet
    # has sized the indicator and the “✓” can be drawn outside it.
    QTimer.singleShot(0, lambda: (_position_checkmark(), _update_checkmark()))
    return checkbox


def _fit_pill_font_pt(text: str, max_width: int,
                      base_pt: float = 7.0, min_pt: float = 5.0,
                      step: float = 0.5,
                      horiz_padding: int = 14,
                      bold: bool = True,
                      base_font: QFont | None = None) -> float:
    """Pick a font point size that renders *text* inside *max_width* pixels.

    Pills on the flash cards (“Ready to compile…”, “⚠ missing raw”,
    “⏳ N/M”, …) live inside a fixed-width card and can easily overflow
    horizontally at the preset’s base font size once the card shrinks
    to Compact / Normal or the text happens to be unusually long
    (“Ready to compile (1589/1589)” is ~140 px wide at 7 pt bold).
    Shrinking the font in 0.5 pt increments until it fits mirrors
    :func:`_fit_title_text`’s shrink loop and keeps the short-text
    case rendered at full size.

    *horiz_padding* covers the pill’s CSS padding (5 px left + 5 px
    right), border (1 px × 2), and a small safety buffer so a 1-2 px
    measurement error from ``QFontMetrics.horizontalAdvance`` doesn’t
    let a marginal case clip on render. Falls back to *min_pt* when
    even that doesn’t fit (callers can still truncate if needed).
    """
    if not text:
        return float(base_pt)
    max_width = max(1, int(max_width))
    pt = float(base_pt)
    min_pt = float(min_pt)
    step = float(step)
    cache_key = (
        text,
        max_width,
        round(pt, 3),
        round(min_pt, 3),
        round(step, 3),
        int(horiz_padding),
        bool(bold),
        _font_cache_key(base_font),
    )
    cached = _PILL_FONT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    while pt > min_pt:
        f = QFont(base_font) if base_font is not None else QFont()
        f.setPointSizeF(pt)
        f.setBold(bold)
        fm = QFontMetrics(f)
        if fm.horizontalAdvance(text) + horiz_padding <= max_width:
            _cache_put_bounded(
                _PILL_FONT_CACHE, cache_key, pt, _RENDER_RESULT_CACHE_LIMIT)
            return pt
        pt = max(min_pt, pt - step)
    _cache_put_bounded(
        _PILL_FONT_CACHE, cache_key, min_pt, _RENDER_RESULT_CACHE_LIMIT)
    return min_pt


def _fit_title_text(
    text: str,
    avail_width: int,
    max_height: int,
    base_pt: float,
    base_font: QFont | None = None,
    min_pt: float = 6.5,
    step: float = 0.5,
) -> tuple[str, float]:
    """Return (rendered_text, font_pt) that fits inside the given box.

    Strategy:
      1. Render *text* at *base_pt* with word-wrap inside ``avail_width``.
      2. If the wrapped text overflows ``max_height`` vertically, shrink the
         font size in ``step`` pt increments down to ``min_pt``.
      3. If even at ``min_pt`` the text still overflows, truncate it with
         an ellipsis at the longest prefix that does fit (binary search).

    No hard character cap — short titles always render at full size, and
    long titles gracefully scale / trim as needed.
    """
    text = text or ""
    avail_width = max(1, int(avail_width))
    max_height = max(1, int(max_height))
    base_pt = float(base_pt)
    min_pt = min(float(min_pt), base_pt)
    step = float(step)
    cache_key = (
        text,
        avail_width,
        max_height,
        round(base_pt, 3),
        round(min_pt, 3),
        round(step, 3),
        _font_cache_key(base_font),
    )
    cached = _TITLE_FIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Measure via ``QTextLayout`` with ``WrapAtWordBoundaryOrAnywhere``.
    # Plain ``QFontMetrics.boundingRect(TextWordWrap)`` under-measures
    # long Hangul / filename-style runs (it doesn't break between CJK
    # ideographs the way QLabel's renderer actually does), so a 5-line
    # title was being reported as "fits in 3 lines at base_pt" and the
    # shrink loop exited immediately. ``QTextLayout`` is the same
    # engine ``QLabel`` uses to paint word-wrapped text — when we also
    # switch the flash-card and Book-Details renderers to
    # ``_FittedTitleLabel`` (which paints with identical layout
    # options), measurement and render are byte-identical.
    _text_option = QTextOption()
    _text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
    _text_option.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    _height_cache: dict[tuple[str, float], int] = {}

    def _height(s: str, pt_size: float) -> int:
        if not s:
            return 0
        height_key = (s, round(float(pt_size), 3))
        cached_height = _height_cache.get(height_key)
        if cached_height is not None:
            return cached_height
        f = QFont(base_font) if base_font is not None else QFont()
        f.setPointSizeF(pt_size)
        f.setBold(True)
        layout = QTextLayout(s, f)
        layout.setTextOption(_text_option)
        layout.beginLayout()
        y = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(float(avail_width))
            line.setPosition(QPointF(0.0, y))
            y += line.height()
        layout.endLayout()
        # Round up so a fractional overflow of a pixel still trips the
        # shrink loop — better to shrink an extra half-step than to let
        # the descender of the last line get clipped.
        out = int(y + 0.999)
        _height_cache[height_key] = out
        return out

    pt = base_pt
    while pt > min_pt and _height(text, pt) > max_height:
        pt = max(min_pt, pt - step)

    if _height(text, pt) <= max_height:
        result = (text, pt)
        _cache_put_bounded(
            _TITLE_FIT_CACHE, cache_key, result, _RENDER_RESULT_CACHE_LIMIT)
        return result

    # Overflows even at the minimum size — truncate with an ellipsis.
    ellipsis = "\u2026"
    lo, hi, best = 1, max(1, len(text)), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if _height(candidate, pt) <= max_height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    result = (text[:best].rstrip() + ellipsis, pt)
    _cache_put_bounded(
        _TITLE_FIT_CACHE, cache_key, result, _RENDER_RESULT_CACHE_LIMIT)
    return result


class _FittedTitleLabel(QWidget):
    """Self-painting label that wraps text at *any* point if needed.

    ``QLabel.setWordWrap(True)`` only breaks at Unicode word boundaries,
    which isn't enough for long Hangul / CJK filename-style titles —
    the renderer keeps a single long ideograph run on one line and
    overflows horizontally. This widget paints the text using the same
    ``QTextLayout`` + ``QTextOption.WrapAtWordBoundaryOrAnywhere`` combo
    that :func:`_fit_title_text` uses for measurement, so the measured
    height and the rendered height are byte-identical — the shrink
    loop's chosen font size always lands cleanly inside the title box.

    Public API mirrors the subset of QLabel that the card code needs:
    :meth:`setText`, :meth:`setFont`, :meth:`setToolTip`,
    :meth:`setFixedHeight`, :meth:`setAlignment`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._color = "#e0e0e0"
        self._option = QTextOption()
        self._option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self._option.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setAttribute(Qt.WA_TranslucentBackground)

    # -- public, QLabel-compatible API -------------------------------------
    def setText(self, text: str) -> None:
        self._text = text or ""
        self.update()

    def text(self) -> str:
        return self._text

    def setAlignment(self, alignment) -> None:
        self._option.setAlignment(alignment)
        self.update()

    def setTextColor(self, color: str) -> None:
        """Set the foreground color as a CSS-style string."""
        self._color = color or "#e0e0e0"
        self.update()

    # -- QWidget overrides -------------------------------------------------
    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        if not self._text:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setFont(self.font())
            painter.setPen(QColor(self._color))
            layout = QTextLayout(self._text, self.font())
            layout.setTextOption(self._option)
            width = float(max(1, self.width()))
            layout.beginLayout()
            y = 0.0
            max_h = float(self.height())
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                line.setPosition(QPointF(0.0, y))
                if y + line.height() > max_h:
                    # No room for another line — bail out so we don't
                    # paint a half-clipped descender.
                    break
                y += line.height()
            layout.endLayout()
            layout.draw(painter, QPointF(0.0, 0.0))
        finally:
            painter.end()

    def sizeHint(self) -> QSize:
        # Enough rows for a 2-line tooltip at the current font; the card
        # layout owns the real height via setFixedHeight anyway.
        fm = QFontMetrics(self.font())
        return QSize(100, fm.height() * 2)


# _find_folder_cover moved verbatim to library_covers (imported above).


class _LibraryScannerThread(QThread):
    """Run scan_for_epubs() off the UI thread so the loading spinner animates.

    Retained for backward compatibility — external callers still depend
    on the merged scan. The tabbed dialog uses
    :class:`_DualScannerThread` instead.
    """
    scan_finished = Signal(list)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setObjectName("LibraryScannerThread")
        self._config = config or {}

    def run(self):
        try:
            results = scan_for_epubs(self._config)
        except Exception:
            logger.debug("Library scan failed: %s", traceback.format_exc())
            results = []
        self.scan_finished.emit(results)


# _attach_cross_location_duplicates moved verbatim to library_core (imported above).


class _DualScannerThread(DualScanMixin, QThread):
    """Scan both library and output roots, partitioning by completion status.

    Emits ``(in_progress_list, completed_list)`` where:
      * in_progress_list — output folders with a progress file but no compiled EPUB yet.
      * completed_list — Library EPUBs + output folders whose compiled EPUB exists.

    Entries whose compiled EPUB already lives in the Library folder are
    deduped (library entry wins) so the same book can't appear twice.
    """
    scan_finished = Signal(list, list)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setObjectName("DualLibraryScannerThread")
        self._config = config or {}

    # run moved verbatim to library_core.DualScanMixin (inherited).


class _LibraryDeleteThread(LibraryDeleteMixin, QThread):
    """Delete top-level library targets off the UI thread.

    Each target is still one logical item from the confirmation dialog:
    either a file or a whole output workspace folder. The thread fans
    those top-level targets out across a small pool so selecting several
    workspaces does not wait for ``shutil.rmtree`` one folder at a time.
    """
    progress = Signal(int, int, str)
    delete_finished = Signal(list)

    def __init__(self, targets: list[tuple[str, str, bool]], parent=None):
        super().__init__(parent)
        self.setObjectName("LibraryDeleteThread")
        self._targets = list(targets or [])

    # _delete_one, run moved verbatim to library_core.LibraryDeleteMixin (inherited).


class _CoverLoader(CoverLoaderMixin, QThread):
    result_ready = Signal(str, str)

    def __init__(self, file_path: str, file_type: str = "epub", config: dict | None = None,
                 original_path: str | None = None, raw_source_path: str | None = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("LibraryCoverLoader")
        self._file_path = file_path
        self._file_type = file_type
        self._config = config or {}
        self._original_path = original_path
        # For in-progress cards the output folder may have no images yet, so
        # the thumbnail comes directly from the resolved raw EPUB/PDF source.
        self._raw_source_path = raw_source_path or ""
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        try:
            self.requestInterruption()
        except Exception:
            pass

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to library_covers.CoverLoaderMixin (inherited).


class _SelectableGrid(QWidget):
    """Grid container that supports click-drag rubber-band selection.

    Emits :attr:`rubber_band_selection` once the user finishes a drag.
    Plain clicks on empty space emit :attr:`empty_clicked` so the parent
    can clear the current selection.

    Dragging starts from **anywhere** in the grid — including on top of
    a :class:`_BookCard` — because the slivers of empty space between
    cards are too narrow to reliably grab. A mouse press on a card is
    initially handled as a normal card click (the card still emits
    :attr:`_BookCard.select_requested` so single-click selection keeps
    working); as soon as the pointer moves past :data:`_DRAG_THRESHOLD`,
    the grid promotes the gesture to a rubber-band drag starting from
    the original press location. The card's just-fired selection is
    overwritten by the rubber-band result on mouse release (a plain
    drag clears + replaces; Ctrl / Shift extends), so the user never
    sees a stuck "single card selected" state after a drag.
    """
    # (list_of_books_inside_band, modifiers)
    rubber_band_selection = Signal(list, object)
    # (modifiers) — fired on a plain click with no drag
    empty_clicked = Signal(object)

    # Minimum pointer movement (Manhattan distance in px) before we start
    # showing the rubber band. Prevents a stray twitch on mouse press from
    # accidentally replacing the current multi-selection with nothing.
    _DRAG_THRESHOLD = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rubber_band = None
        self._drag_origin = None
        self._drag_modifiers = None
        self._drag_started = False
        # Filter-based press tracking for drags that start on a card.
        # The card handles the initial press itself (so click-select
        # still works); we watch for subsequent MouseMove on the card
        # — via an event filter installed on every descendant — and
        # promote the gesture to a rubber-band once it passes the
        # drag threshold. Coords are always translated into grid-
        # local space via :func:`QWidget.mapTo` so the band geometry
        # matches what a press on empty space would have produced.
        self._card_drag_origin = None
        self._card_drag_modifiers = None
        # Install the filter on any children that happen to exist at
        # construction time (usually none — the grid is populated
        # later). :meth:`childEvent` handles everything added after.
        self._install_filter_on_descendants()

    # -- event-filter plumbing ---------------------------------------------
    def childEvent(self, event):
        """Install the rubber-band filter on newly added descendants.

        ``ChildAdded`` fires once per direct child, so we handle the
        direct child here and recurse via :meth:`_install_filter_on_descendants`
        to catch the card's own sub-widgets (labels, layout spacers) —
        otherwise a press on a card's title label would slip past the
        filter entirely and the drag-through-card gesture would feel
        unreliable depending on exactly where the user clicked.
        """
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.ChildAdded:
            child = event.child()
            try:
                if isinstance(child, QWidget):
                    child.installEventFilter(self)
                    # Recurse so labels / spacers inside cards are
                    # covered too — a press on ``title_lbl`` otherwise
                    # never reaches the filter because QLabel consumes
                    # mouse events by default.
                    for sub in child.findChildren(QWidget):
                        try:
                            sub.installEventFilter(self)
                        except Exception:
                            pass
            except Exception:
                pass
        super().childEvent(event)

    def _install_filter_on_descendants(self):
        for c in self.findChildren(QWidget):
            try:
                c.installEventFilter(self)
            except Exception:
                pass

    def _card_ancestor(self, obj):
        """Return the :class:`_BookCard` ancestor of *obj*, or None."""
        if not isinstance(obj, QWidget):
            return None
        node = obj
        while node is not None and node is not self:
            if isinstance(node, _BookCard):
                return node
            try:
                node = node.parent()
            except Exception:
                return None
        return None

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent, QRect
        card = self._card_ancestor(obj)
        if card is None:
            return super().eventFilter(obj, event)
        etype = event.type()
        if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            # Record the press origin in grid-local coords but let
            # the card keep the event so ``select_requested`` still
            # fires for a plain click. ``mapTo`` handles the nested
            # widget case (press on a label inside the card).
            try:
                self._card_drag_origin = obj.mapTo(self, event.pos())
            except Exception:
                self._card_drag_origin = None
            self._card_drag_modifiers = event.modifiers()
            self._drag_started = False
            return False
        if etype == QEvent.MouseMove and (event.buttons() & Qt.LeftButton):
            if (self._card_drag_origin is not None
                    and not self._drag_started):
                try:
                    current = obj.mapTo(self, event.pos())
                except Exception:
                    return False
                delta = current - self._card_drag_origin
                if abs(delta.x()) + abs(delta.y()) > self._DRAG_THRESHOLD:
                    self._start_rubber_band(
                        self._card_drag_origin,
                        self._card_drag_modifiers or Qt.NoModifier)
                    self._rubber_band.setGeometry(
                        QRect(self._drag_origin, current).normalized())
                    return True
            if self._drag_started and self._rubber_band is not None:
                try:
                    current = obj.mapTo(self, event.pos())
                except Exception:
                    return False
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, current).normalized())
                return True
        if etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._drag_started:
                # Finalize: emit selection + reset state. Consume the
                # release so the card's own handler doesn't also fire
                # and re-emit ``select_requested`` for a now-stale
                # single-card selection.
                self._finish_rubber_band()
                self._card_drag_origin = None
                self._card_drag_modifiers = None
                return True
            self._card_drag_origin = None
            self._card_drag_modifiers = None
            return False
        return super().eventFilter(obj, event)

    # -- shared start / finish helpers -------------------------------------
    def _start_rubber_band(self, origin, modifiers):
        from PySide6.QtCore import QRect, QSize
        from PySide6.QtWidgets import QRubberBand
        self._drag_origin = origin
        self._drag_modifiers = modifiers
        self._drag_started = True
        if self._rubber_band is None:
            self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber_band.setGeometry(QRect(origin, QSize()))
        self._rubber_band.show()

    def _finish_rubber_band(self):
        if self._rubber_band is None or not self._drag_started:
            return
        rect = self._rubber_band.geometry()
        self._rubber_band.hide()
        books: list = []
        for child in self.children():
            if isinstance(child, _BookCard):
                if rect.intersects(child.geometry()):
                    books.append(child.book)
        self.rubber_band_selection.emit(
            books, self._drag_modifiers or Qt.NoModifier)
        self._drag_origin = None
        self._drag_modifiers = None
        self._drag_started = False

    # -- direct mouse handling (presses landing on empty grid space) ------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.pos()
            self._drag_modifiers = event.modifiers()
            self._drag_started = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and (event.buttons() & Qt.LeftButton):
            from PySide6.QtCore import QRect
            delta = event.pos() - self._drag_origin
            distance = abs(delta.x()) + abs(delta.y())
            if not self._drag_started and distance > self._DRAG_THRESHOLD:
                self._start_rubber_band(
                    self._drag_origin,
                    self._drag_modifiers or Qt.NoModifier)
            if self._drag_started and self._rubber_band is not None:
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, event.pos()).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if self._drag_started and self._rubber_band is not None:
                self._finish_rubber_band()
            elif (event.button() == Qt.LeftButton
                    and self._drag_origin is not None):
                # Plain click on empty space (no drag) — let the parent
                # decide whether to clear the selection.
                self.empty_clicked.emit(self._drag_modifiers or Qt.NoModifier)
        finally:
            self._drag_origin = None
            self._drag_modifiers = None
            self._drag_started = False
        super().mouseReleaseEvent(event)


# _card_raw_title moved verbatim to library_core (imported above). _unique_dest is the nested
# helper of EpubLibraryDialog._organize_into_library lifted to library_core; _page_bounds /
# _page_label were extracted from the library and chapter pager methods and _CARD_TYPE_BADGES,
# _card_type_badge, _card_size_text, _card_progress_view from _BookCard.__init__ (library_core,
# imported above; see DISCREPANCIES U5 "Phase-1 splits").


class _BookCard(QFrame):
    clicked = Signal(dict)
    context_menu_requested = Signal(dict, object)
    hover_changed = Signal(object, bool)
    # Emitted on left-click so the parent dialog can manage multi-selection
    # (Ctrl-click toggles, plain click replaces). Payload: (book, modifiers).
    select_requested = Signal(dict, object)

    # Selectors target the widget by object name (rather than its Python
    # class name via a type selector) because Qt's metaobject className for
    # PySide6-subclassed QFrames isn't always reliable — an ID selector is
    # the one form guaranteed to match exactly this widget. The border is
    # kept at 2 px in both states so the content area doesn't reflow on
    # selection (which previously caused visually stuck hover rendering).
    _BASE_STYLE = (
        "QFrame#bookCard { background: #1e1e2e; border: 2px solid #2a2a3e;"
        " border-radius: 6px; }"
    )
    _HOVER_STYLE = (
        "QFrame#bookCard { background: #252540; border: 2px solid #6c63ff;"
        " border-radius: 6px; }"
    )
    _SELECTED_STYLE = (
        "QFrame#bookCard { background: #2a2d5a; border: 2px solid #a097ff;"
        " border-radius: 6px; }"
    )
    _SELECTED_HOVER_STYLE = (
        "QFrame#bookCard { background: #343670; border: 2px solid #c0b8ff;"
        " border-radius: 6px; }"
    )

    def __init__(self, book: dict, preset: dict | None = None, parent=None,
                 show_raw_title: bool = False):
        super().__init__(parent)
        self.book = book
        p = preset or _SIZE_PRESETS[SIZE_NORMAL]
        self._card_w = p["card_w"]
        self._cover_h = p["cover_h"]
        # Smaller badge typography on the tiny presets (XS / 2XS) so the
        # corner ribbon and the size / type badges don't dominate the
        # shrunken cards. Preset is identified via its title size — the
        # card width itself can be stretched by the grid's column fitting.
        _title_pt = _parse_pt(p.get("title_size", "9pt"))
        if _title_pt <= 7.5:      # 2XS
            self._ribbon_pt = 5.0
            self._badge_pt = 5.5
            self._size_lbl_pt = 6.0
        elif _title_pt <= 8.0:    # XS
            self._ribbon_pt = 5.5
            self._badge_pt = 6.0
            self._size_lbl_pt = 6.5
        else:
            self._ribbon_pt = 6.5
            self._badge_pt = 7.0
            self._size_lbl_pt = 7.5
        self._has_cover = False
        self._cover_movie = None
        self._selected = False
        self._hovered = False
        self._applied_style = ""
        self._show_raw_title = bool(show_raw_title)

        self.setObjectName("bookCard")
        self.setFixedWidth(self._card_w)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_card_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        # Tight inter-row spacing so the "size/badge" row and the
        # progress pill sit closer together. QVBoxLayout applies this
        # between every pair of widgets, including cover ↔ title, but
        # 1 px stays visually clean and mirrors the card's balance.
        layout.setSpacing(1)

        self.cover_label = QLabel()
        self.cover_label.setFixedSize(self._card_w - 8, self._cover_h)
        self.cover_label.setAlignment(Qt.AlignCenter)
        self.cover_label.setStyleSheet("background: #2a2a3e; border-radius: 4px; color: #555; font-size: 28pt;")
        icon_path = _find_halgakos_icon()
        if icon_path:
            self._set_fallback_icon(icon_path)
        else:
            self.cover_label.setText("\U0001f4d6")
        layout.addWidget(self.cover_label)
        # Breathing room between the cover thumbnail and the title so
        # the two don’t read as a single block. ``layout.setSpacing(1)``
        # above keeps every OTHER inter-widget gap tight (size / pill /
        # warning rows) — we only want the extra air right here where
        # the image meets the text. 5 px lands comfortably between
        # “touching” (1 px) and “floating” (8+ px).
        layout.addSpacing(5)

        # Title: try to render the full name at the preset font size; shrink
        # the font (down to ``title_min_size``) if it wraps past the max
        # height, and fall back to an ellipsis only as a last resort. See
        # :func:`_fit_title_text`. When the library's "Show raw titles"
        # toggle is on, :func:`_card_raw_title` picks the source-language
        # title instead of whatever ``book['name']`` resolved to.
        if self._show_raw_title:
            full_title = _card_raw_title(book) or book.get("name", "")
        else:
            full_title = book["name"]
        base_pt = _parse_pt(p.get("title_size", "9pt"))
        min_pt = _parse_pt(p.get("title_min_size", "6.5pt"))
        max_title_h = p.get("title_max_h", 36)
        # Cards WITHOUT any warning badge inherit the 16 px slot
        # we reserve at the bottom (see ``reserved_h += 16`` below)
        # as extra title room — otherwise that space just sits
        # empty, the card looks bottom-heavy, and longer titles
        # ellipsize unnecessarily. Cards WITH a warning keep the
        # slot for the badge and the title stays at its preset
        # height. Either way the total card height is identical so
        # the grid rows line up. Both ``missing_raw_file`` and
        # ``compiled_conflicts`` render on the same dedicated
        # ``warnings_row`` below, so either one counts as a warning
        # for the purposes of the title-height reservation.
        _has_warning_preview = bool(
            book.get("missing_raw_file")
            or book.get("compiled_conflicts"))
        effective_max_title_h = max_title_h + (0 if _has_warning_preview else 16)
        # Custom-paint widget whose wrap mode matches :func:`_fit_title_text`'s
        # measurement, so the shrink loop's chosen font size always fits
        # the box even for long Hangul / CJK filename-style raw titles.
        title_lbl = _FittedTitleLabel()
        title_lbl.setFixedWidth(self._card_w - 10)
        title_lbl.setFixedHeight(effective_max_title_h)
        title_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        title_lbl.setToolTip(full_title)
        fitted_text, fitted_pt = _fit_title_text(
            full_title,
            avail_width=self._card_w - 10,
            max_height=effective_max_title_h,
            base_pt=base_pt,
            base_font=title_lbl.font(),
            min_pt=min_pt,
        )
        # Apply the fitted point size via ``setFont`` directly on the
        # custom widget. The widget's paintEvent uses the same QTextLayout
        # + WrapAtWordBoundaryOrAnywhere pipeline :func:`_fit_title_text`
        # measures with, so measured height == rendered height.
        title_font = QFont(title_lbl.font())
        title_font.setPointSizeF(fitted_pt)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setText(fitted_text)
        title_lbl.setTextColor("#e0e0e0")
        layout.addWidget(title_lbl)

        # Size + file type badge on same row. For in-progress folders the
        # badge reflects the *source* workspace kind (epub/txt/pdf/image)
        # so users can distinguish a TXT translation's progress file from
        # an EPUB's at a glance.
        badge_text, badge_color = _card_type_badge(book)
        size_str = _card_size_text(book["size"])
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(4)
        size_lbl = QLabel(size_str)
        size_lbl.setAttribute(Qt.WA_TranslucentBackground)
        size_lbl.setStyleSheet(f"color: #888; font-size: {self._size_lbl_pt}pt; background: transparent;")
        info_row.addWidget(size_lbl)
        badge_lbl = QLabel(badge_text)
        badge_lbl.setAttribute(Qt.WA_TranslucentBackground)
        badge_lbl.setStyleSheet(f"color: {badge_color}; font-size: {self._badge_pt}pt; font-weight: bold; background: transparent;")
        info_row.addWidget(badge_lbl)
        info_row.addStretch()
        layout.addLayout(info_row)

        # Warnings row: every warning badge (both ``missing_raw_file``
        # and ``compiled_conflicts``) is stacked onto its own dedicated
        # row below the size / badge row so the Completed tab's
        # “⚠ +N” conflicts badge reads as a peer of In Progress's
        # “⚠ missing raw” — both sit beneath ``info_row`` instead of
        # being crammed next to the size label. Stacking them beside
        # the size / badge on ``info_row`` used to clip long labels
        # (“⚠ missing raw” → “⚠ …”) once the card was Compact /
        # Normal, because the size label + EPUB badge already claimed
        # most of the row’s width.
        #
        # Multi-output (``compiled_conflicts``): the scanner detected
        # more than one compiled artefact in this book's output folder
        # (e.g. two ``.epub`` files from successive recompiles, or a
        # stale ``*_translated.html`` sitting next to a fresh
        # ``.epub``). The primary compiled output is still chosen
        # deterministically (EPUB > PDF > TXT > HTML) but this warns
        # the user that the deterministic pick may not be what they
        # expected.
        #
        # Missing-raw (``missing_raw_file``): the workspace still has
        # a compiled / progress / response trail but the ORIGINAL raw
        # source EPUB can't be resolved on disk anymore. Actions that
        # need the raw source (Reveal source, Read raw, Load for
        # translation) are disabled while this flag is set.
        conflicts = list(book.get("compiled_conflicts") or [])
        has_missing_raw = bool(book.get("missing_raw_file"))
        has_warnings_row = bool(conflicts) or has_missing_raw
        if has_warnings_row:
            warnings_row = QHBoxLayout()
            warnings_row.setContentsMargins(0, 0, 0, 0)
            warnings_row.setSpacing(4)
            if has_missing_raw:
                missing_lbl = QLabel("\u26a0 missing raw")
                missing_lbl.setAttribute(Qt.WA_TranslucentBackground)
                _missing_budget = max(40, int(self._card_w - 12))
                _missing_pt = _fit_pill_font_pt(
                    missing_lbl.text(), _missing_budget,
                    base_pt=7.0, min_pt=5.5, horiz_padding=14,
                )
                missing_lbl.setStyleSheet(
                    f"color: #ff9e6d; font-size: {_missing_pt}pt; font-weight: bold; "
                    "background: rgba(255, 158, 109, 0.15); "
                    "border: 1px solid #ff9e6d; border-radius: 3px; "
                    "padding: 0 4px;"
                )
                missing_lbl.setToolTip(
                    "The raw source file for this book can't be found on "
                    "disk \u2014 Library/Raw, source_epub.txt, and the "
                    "raw-inputs registry all came up empty. The compiled "
                    "output is still readable, but actions that need the "
                    "raw source (Reveal source, Read raw, Load for "
                    "translation) will be disabled."
                )
                warnings_row.addWidget(missing_lbl)
            if conflicts:
                conflict_lbl = QLabel(f"\u26a0 +{len(conflicts)}")
                conflict_lbl.setAttribute(Qt.WA_TranslucentBackground)
                # Dynamic font size: the conflict badge usually reads
                # “⚠ +2” / “⚠ +12” (short). Now that it lives on its
                # own row the whole card width is available, but we
                # still clamp so a runaway “+999” can't blow the row
                # out on a Compact card.
                _conflict_budget = max(40, int(self._card_w - 12))
                _conflict_pt = _fit_pill_font_pt(
                    conflict_lbl.text(), _conflict_budget,
                    base_pt=7.0, min_pt=5.5, horiz_padding=12,
                )
                conflict_lbl.setStyleSheet(
                    f"color: #ffb347; font-size: {_conflict_pt}pt; font-weight: bold; "
                    "background: rgba(255, 179, 71, 0.15); "
                    "border: 1px solid #ffb347; border-radius: 3px; "
                    "padding: 0 4px;"
                )
                # Lightweight human-readable summary for the tooltip:
                # "Extra compiled files in this folder:\n  • foo.epub (EPUB)".
                tip_lines = ["Extra compiled files in this folder:"]
                for name, kind in conflicts[:8]:
                    tip_lines.append(f"  \u2022 {name} ({kind.upper()})")
                if len(conflicts) > 8:
                    tip_lines.append(f"  \u2026 and {len(conflicts) - 8} more.")
                tip_lines.append(
                    "\nThe card displays only the primary output (EPUB > PDF > "
                    "TXT > HTML); delete the stale artefacts to avoid "
                    "confusion on the next scan."
                )
                conflict_lbl.setToolTip("\n".join(tip_lines))
                warnings_row.addWidget(conflict_lbl)
            warnings_row.addStretch()
            layout.addLayout(warnings_row)

        # In-progress indicator: small status pill + overlay ribbon on the cover
        has_progress_row = False
        progress_view = _card_progress_view(book)
        if progress_view is not None:
            has_progress_row = True
            total = progress_view["total"]
            state = progress_view["state"]
            progress_row = QHBoxLayout()
            progress_row.setContentsMargins(0, 0, 0, 0)
            progress_row.setSpacing(4)
            # Budget for the pill so :func:`_fit_pill_font_pt` can
            # shrink a too-long label (e.g. “Ready to compile
            # (1589/1589)”) down to a size that still fits inside
            # the card’s fixed width. We subtract the card’s own
            # left/right padding (8 px), the ~30 px the “NN%”
            # label to the right takes when the “in_progress”
            # branch shows it, and a generous buffer for the
            # row’s 4 px spacing, border, and Qt’s emoji-width
            # under-measurement (✨, ⏳, ⚠, 🆕 all render a
            # few px wider than ``QFontMetrics.horizontalAdvance``
            # predicts on Windows, which is why “Ready to compile
            # (15/15)” clipped at 7 pt even though the measurement
            # said it fit).
            _needs_pct_lbl = bool(total) and state not in (
                "outdated_progress", "not_started", "ready_to_compile",
            )
            _pct_reservation = 30 if _needs_pct_lbl else 0
            _pill_budget = max(40, int(self._card_w - 16 - _pct_reservation))
            # Horizontal padding used by every call below covers
            # 10 px CSS padding (5 + 5), 2 px border, and 10 px of
            # safety buffer for emoji-width drift / antialiasing
            # so the shrink loop’s “it fits” result actually fits
            # on every system font. Pair with ``min_pt=4.5`` so
            # the worst-case label (“Ready to compile (1589/1589)”
            # on a Compact card) still lands without truncation.
            _pill_horiz_padding = 22
            _pill_base_font = QFont(self.font())
            _pill_base_font.setBold(True)
            pill = QLabel(progress_view["pill_text"])
            pill.setToolTip(progress_view["pill_tooltip"])
            _pill_pt = _fit_pill_font_pt(
                pill.text(), _pill_budget,
                base_pt=7.0, min_pt=4.5,
                horiz_padding=_pill_horiz_padding,
                base_font=_pill_base_font,
            )
            pill.setStyleSheet(
                f"color: {progress_view['pill_color']}; "
                f"background: {progress_view['pill_background']}; "
                f"border: 1px solid {progress_view['pill_border']}; border-radius: 3px; "
                f"font-size: {_pill_pt}pt; font-weight: bold; "
                "padding: 0 5px 2px 5px;"
            )
            progress_row.addWidget(pill)
            if progress_view["show_pct"]:
                pct_lbl = QLabel(progress_view["pct_text"])
                pct_lbl.setAttribute(Qt.WA_TranslucentBackground)
                pct_lbl.setStyleSheet(f"color: #8ab4d0; font-size: {self._badge_pt}pt; font-weight: bold; background: transparent;")
                progress_row.addWidget(pct_lbl)
            ribbon_text = progress_view["ribbon_text"]
            ribbon_bg = progress_view["ribbon_background"]
            progress_row.addStretch()
            # Pin the progress row to the BOTTOM of the card by
            # inserting a vertical stretch above it. Without this
            # stretch the pill floats mid-card (between info_row /
            # warnings_row and the ``layout.addStretch()`` below),
            # so a card without a warning row sits with a dead
            # band below the pill while another card WITH a
            # warning has the pill tucked in the middle. Pinning
            # the pill to the bottom lines every card’s pill up
            # along the same baseline across the grid.
            layout.addStretch()
            layout.addLayout(progress_row)

            # Corner ribbon on the cover label (absolutely positioned child)
            self._progress_ribbon = QLabel(ribbon_text, self.cover_label)
            self._progress_ribbon.setStyleSheet(
                f"color: #fff; background: {ribbon_bg}; "
                f"font-size: {self._ribbon_pt}pt; font-weight: bold; "
                "padding: 1px 5px; "
                "border-bottom-right-radius: 3px;"
            )
            self._progress_ribbon.move(0, 0)
            self._progress_ribbon.show()

        # Trailing stretch + fixed card height so every card within the
        # same tab occupies a uniform footprint. Completed cards skip the
        # progress-pill reservation so they don't render with a dead band
        # of empty space at the bottom where the pill would have been on
        # the In Progress tab.
        layout.addStretch()
        # Shrunk alongside ``layout.setSpacing(1)`` / bottom-margin 4
        # so the tighter inter-row gaps don't just migrate into a
        # bigger empty band below the progress pill.
        reserved_h = 24 + p.get("spacing", 4)
        # Match the ``layout.addSpacing(5)`` inserted between the cover
        # and the title above — without this, ``setFixedHeight`` stays
        # the same and the title box loses 5 px off its bottom.
        reserved_h += 5
        if has_progress_row:
            reserved_h += 20  # pill row height + extra inter-widget spacing
        # Always reserve the warnings-row slot so every card lands at
        # the same fixed height, whether or not it currently carries
        # a “⚠ missing raw” badge. Without this constant, cards
        # with and without the badge drift by the badge’s height and
        # the grid looks ragged row-to-row. 16 px covers the badge
        # (~12 px) + ``layout.setSpacing(1)`` + a bit of breathing
        # room below the progress pill so the card doesn’t feel
        # bottom-cramped.
        reserved_h += 16
        self.setFixedHeight(self._cover_h + max_title_h + reserved_h)

    def set_selected(self, selected: bool):
        """Toggle the card's "selected" visual state.

        Used by :class:`EpubLibraryDialog` to render multi-selection for
        batch actions like "Load N for translation". Stylesheet-only
        change — no layout recomputation, so this is cheap.
        """
        new_value = bool(selected)
        if new_value == self._selected:
            return
        self._selected = new_value
        self._apply_card_style()
        # Force an immediate repaint so the stylesheet swap is painted on
        # the current tick rather than waiting for the next synthetic event.
        self.update()

    @property
    def selected(self) -> bool:
        return self._selected

    def set_hovered(self, hovered: bool) -> None:
        new_value = bool(hovered)
        if new_value == self._hovered:
            return
        self._hovered = new_value
        self._apply_card_style()
        self.update()

    def _apply_card_style(self) -> None:
        if self._selected:
            style = self._SELECTED_HOVER_STYLE if self._hovered else self._SELECTED_STYLE
        else:
            style = self._HOVER_STYLE if self._hovered else self._BASE_STYLE
        if style == self._applied_style:
            return
        self._applied_style = style
        self.setStyleSheet(style)

    def enterEvent(self, event):
        self.set_hovered(True)
        self.hover_changed.emit(self, True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.set_hovered(False)
        self.hover_changed.emit(self, False)
        super().leaveEvent(event)

    def _set_fallback_icon(self, icon_path: str):
        try:
            target_w = int(self._cover_h * 0.5)
            target_h = int(self._cover_h * 0.6)
            scaled = _cached_scaled_pixmap(icon_path, target_w, target_h)
            if scaled is not None and not scaled.isNull():
                self.cover_label.setPixmap(scaled)
                self.cover_label.setText("")
        except Exception:
            logger.debug("Fallback icon failed: %s", traceback.format_exc())
            self.cover_label.setText("📖")

    def set_cover(self, image_path: str):
        try:
            movie = _build_cover_movie(
                image_path,
                self.cover_label,
                self._card_w - 8,
                self._cover_h,
            )
            if movie is not None:
                _dispose_cover_movie(self._cover_movie)
                self._cover_movie = movie
                self.cover_label.setText("")
                self._has_cover = True
                movie.start()
                return
            scaled = _cached_scaled_pixmap(
                image_path, self._card_w - 8, self._cover_h)
            if scaled is not None and not scaled.isNull():
                _dispose_cover_movie(self._cover_movie)
                self._cover_movie = None
                self.cover_label.setPixmap(scaled)
                self.cover_label.setText("")
                self._has_cover = True
        except Exception:
            logger.debug("Set cover failed: %s", traceback.format_exc())

    def set_compiling(self, active: bool) -> None:
        """Swap the corner ribbon to a "COMPILING…" badge while the EPUB
        converter runs for this card's workspace, restoring the original
        ribbon (or removing the temporary one) when the run ends."""
        ribbon = getattr(self, "_progress_ribbon", None)
        if active:
            if ribbon is None:
                cover = getattr(self, "cover_label", None)
                if cover is None:
                    return
                ribbon = QLabel("", cover)
                ribbon.move(0, 0)
                self._progress_ribbon = ribbon
                self._compiling_ribbon_created = True
            if not getattr(self, "_compiling_active", False):
                self._compiling_active = True
                self._saved_ribbon_state = (
                    ribbon.text(), ribbon.styleSheet(), ribbon.isVisible())
            ribbon.setText("⚙ COMPILING…")
            ribbon.setStyleSheet(
                "color: #1e1616; background: rgba(255, 209, 102, 0.95); "
                f"font-size: {getattr(self, '_ribbon_pt', 6.5)}pt; "
                "font-weight: bold; padding: 1px 5px; "
                "border-bottom-right-radius: 3px;")
            ribbon.adjustSize()
            ribbon.show()
            ribbon.raise_()
        else:
            if not getattr(self, "_compiling_active", False):
                return
            self._compiling_active = False
            if getattr(self, "_compiling_ribbon_created", False):
                # We created this ribbon just for the compile state —
                # remove it instead of restoring an empty label.
                self._compiling_ribbon_created = False
                self._progress_ribbon = None
                try:
                    ribbon.hide()
                    ribbon.deleteLater()
                except Exception:
                    pass
            elif ribbon is not None:
                text, style, visible = getattr(
                    self, "_saved_ribbon_state", ("", "", False))
                ribbon.setText(text)
                ribbon.setStyleSheet(style)
                ribbon.adjustSize()
                ribbon.setVisible(visible)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.book)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Forward to the parent dialog so it can update multi-selection
            # state. Ctrl/Shift modifiers extend or toggle the selection;
            # plain click replaces it with just this card. We MUST accept
            # the event so it doesn't propagate to the enclosing
            # :class:`_SelectableGrid`, which would otherwise record a
            # drag origin and fire ``empty_clicked`` on the matching
            # release — clearing the selection we just set.
            self.select_requested.emit(self.book, event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Consume the release for the same reason as mousePressEvent: if
        # the event reaches :class:`_SelectableGrid` with no drag movement
        # it triggers the "empty space click" path and wipes the selection.
        if event.button() == Qt.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        self.context_menu_requested.emit(self.book, event.globalPos())
        event.accept()


# ---------------------------------------------------------------------------
# Scan-for-Raw dialog
# ---------------------------------------------------------------------------

class _RawScanWorker(RawScanMixin, QThread):
    """Walk + match raw source files off the UI thread.

    Both the ``os.walk`` and the per-workspace matching run here so
    the only main-thread work after a scan is building the tree
    rows. Previously the matching pass (especially Fuzzy mode's
    ``difflib.SequenceMatcher.ratio()`` across every candidate)
    ran back on the UI thread and stalled the dialog for seconds
    on big folders.

    The worker accepts an optional ``prewalked`` candidate list so
    mode / threshold changes don't have to re-walk the folder —
    the dialog caches the last walk result and hands it back for
    fast re-matching.

    A :class:`concurrent.futures.ThreadPoolExecutor` inside the
    worker fans per-directory classification out across 4 threads
    so the normalization pass doesn't bottleneck on the walking
    thread. Cancellable via :meth:`cancel` so a rapid UI toggle
    can stop a stale scan at the next directory / workspace
    boundary.
    """

    # (scan_folder, candidates, matches_dict)
    results = Signal(str, list, dict)

    def __init__(self, scan_folder: str, ext_suffixes: tuple[str, ...],
                 tracking_names: frozenset,
                 books: list[dict], mode: str, threshold: int,
                 prewalked: list | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("RawScanWorker")
        self._folder = scan_folder
        self._suffixes = ext_suffixes
        self._tracking = tracking_names
        self._books = list(books or [])
        self._mode = mode
        self._threshold = int(threshold)
        self._prewalked = prewalked
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        try:
            self.requestInterruption()
        except Exception:
            pass

    # _classify, _walk, _book_keys, _KIND_ALLOWED_EXTS, _compute_matches, run moved verbatim to library_core.RawScanMixin (inherited).


class _StyledCheckDelegate:
    """Marker module-level symbol so the real delegate can live below.

    PySide6 requires importing the ``QStyledItemDelegate`` base class
    from :mod:`PySide6.QtWidgets`, which we do lazily inside
    :class:`_ScanForRawDialog._setup_ui` anyway to match the rest of
    this file’s lazy-import pattern. The actual delegate is
    constructed there via a nested helper class, keeping the import
    footprint of this module small when ``epub_library`` is consumed
    purely for its scanner helpers (e.g. by headless tools).
    """
    pass


class _ScanForRawDialog(ScanForRawMixin, QDialog):
    """Pair every In Progress workspace to a raw source file on disk.

    The user points this dialog at a directory, picks Exact or Fuzzy
    matching (with a slider for the similarity threshold), previews
    the candidate pairings in a table, tweaks the per-row selection
    if they want, and clicks Apply. Each accepted pairing writes:

      * ``<workspace>/source_epub.txt`` — the authoritative pointer
        the scanner consults first, so subsequent scans resolve the
        raw directly without going back through title matching.
      * An entry in ``library_raw_inputs.txt`` for the resolved raw
        so :func:`_find_raw_source_for_folder`'s registry lookup
        picks it up even if the sidecar is lost later.

    Matching heuristic:

      * Exact: the workspace's normalized-key set intersects the
        candidate raw's normalized filename stem.
      * Fuzzy: the highest ``difflib.SequenceMatcher.ratio()`` among
        the workspace's keys vs. the candidate stem, above the
        user-chosen threshold. Ratios are computed against the
        normalized key so the comparison survives Windows filename
        sanitization quirks.
    """

    applied = Signal(int)  # number of pairings written

    # MATCH_EXACT, MATCH_FUZZY, _SUPPORTED_EXTS moved verbatim to library_core.ScanForRawMixin (inherited).

    def __init__(self,
                 in_progress_books: list[dict],
                 config: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self._init_scan_state(in_progress_books, config)
        # Background worker thread that walks the scan folder.
        # Replaced / cancelled whenever the user changes folder or
        # extension selection so we never chew CPU on a stale scan.
        self._scan_worker: _RawScanWorker | None = None
        self.setWindowTitle("\U0001f50d  Scan for Raw Sources")
        self.setMinimumSize(760, 480)
        # Dialog background matches the rest of the Glossarion shell so
        # child widgets (radios / checkboxes / slider / labels) don't
        # paint on top of a default-grey / black fill. Each of those
        # widgets also sets ``background: transparent`` in its own
        # stylesheet below so the dialog colour shows through cleanly.
        self.setStyleSheet("QDialog { background: #12121e; }")
        self._setup_ui()
        # Paint a first preview immediately if the persisted folder
        # still exists on disk — otherwise the user lands on an empty
        # dialog that reads as "nothing found" even though they haven't
        # picked a folder yet.
        if self._scan_folder and os.path.isdir(self._scan_folder):
            QTimer.singleShot(0, self._rescan_folder)

    # _derive_auto_exts moved verbatim to library_core.ScanForRawMixin (inherited).
    # _init_scan_state was extracted from __init__ into library_core.ScanForRawMixin (see DISCREPANCIES U5 "Phase-1 splits").

    # -- UI -----------------------------------------------------------------
    def _setup_ui(self):
        from PySide6.QtWidgets import (
            QSlider, QRadioButton, QButtonGroup, QTreeWidget,
            QTreeWidgetItem, QHeaderView, QDialogButtonBox,
            QFileDialog, QCheckBox, QProgressBar,
        )
        self._QTreeWidgetItem = QTreeWidgetItem
        self._QFileDialog = QFileDialog
        # Tracks whether a background walk / match is currently in
        # flight so UI affordances (progress bar, placeholder row in
        # the tree, Browse tooltip) can stay in sync without every
        # caller having to flip them individually.
        self._scanning = False
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "Pick a folder that contains your raw source files. "
            "Glossarion will try to match each in-progress workspace "
            "to one of those files. Matched pairings get written to "
            "each workspace's <code>source_epub.txt</code> so future "
            "scans resolve the raw automatically."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        intro.setAttribute(Qt.WA_TranslucentBackground)
        intro.setStyleSheet(
            "color: #c8cbe0; font-size: 9.5pt; background: transparent;")
        root.addWidget(intro)

        # Folder picker row
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self._folder_edit = QLineEdit(self._scan_folder)
        self._folder_edit.setPlaceholderText(
            "Folder containing raw EPUB / TXT / PDF / HTML files \u2026")
        self._folder_edit.setStyleSheet(
            "background: #1e1e2e; border: 1px solid #3a3a5e; "
            "border-radius: 4px; padding: 4px 8px; color: #e0e0e0; "
            "font-size: 9.5pt;")
        self._folder_edit.editingFinished.connect(self._on_folder_edit_done)
        folder_row.addWidget(self._folder_edit, 1)
        self._browse_btn = QPushButton("\U0001f4c2  Browse\u2026")
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }"
            "QPushButton:disabled { background: #2a2a3e; color: #6a6d80; }")
        self._browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._browse_btn)
        root.addLayout(folder_row)

        # Mode + threshold row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_lbl = QLabel("Match:")
        mode_lbl.setAttribute(Qt.WA_TranslucentBackground)
        mode_lbl.setStyleSheet(
            "color: #888; font-size: 8.5pt; background: transparent;")
        mode_row.addWidget(mode_lbl)
        self._mode_group = QButtonGroup(self)
        self._rb_exact = QRadioButton("Exact")
        self._rb_fuzzy = QRadioButton("Fuzzy")
        self._rb_exact.setAttribute(Qt.WA_TranslucentBackground)
        self._rb_fuzzy.setAttribute(Qt.WA_TranslucentBackground)
        radio_css = (
            "QRadioButton { color: #c8cbe0; font-size: 9pt; spacing: 6px; "
            "background: transparent; padding: 2px 4px; }"
            "QRadioButton::indicator { width: 14px; height: 14px; "
            "border: 1px solid #5a9fd4; border-radius: 8px; "
            "background-color: #1e1e2e; }"
            "QRadioButton::indicator:checked { background-color: "
            "qradialgradient(cx:0.5, cy:0.5, radius:0.5, "
            "fx:0.5, fy:0.5, stop:0 #20b2cc, stop:0.55 #17a2b8, "
            "stop:0.6 #1e1e2e, stop:1 #1e1e2e); "
            "border-color: #20b2cc; }"
            "QRadioButton::indicator:hover { border-color: #7bb3e0; }"
        )
        self._rb_exact.setStyleSheet(radio_css)
        self._rb_fuzzy.setStyleSheet(radio_css)
        self._rb_exact.setChecked(self._mode == self.MATCH_EXACT)
        self._rb_fuzzy.setChecked(self._mode == self.MATCH_FUZZY)
        self._mode_group.addButton(self._rb_exact)
        self._mode_group.addButton(self._rb_fuzzy)
        self._rb_exact.toggled.connect(self._on_mode_changed)
        self._rb_fuzzy.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self._rb_exact)
        mode_row.addWidget(self._rb_fuzzy)

        mode_row.addSpacing(16)
        thr_lbl = QLabel("Similarity:")
        thr_lbl.setAttribute(Qt.WA_TranslucentBackground)
        thr_lbl.setStyleSheet(
            "color: #888; font-size: 8.5pt; background: transparent;")
        mode_row.addWidget(thr_lbl)
        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setMinimum(40)
        self._threshold_slider.setMaximum(95)
        self._threshold_slider.setValue(self._threshold)
        self._threshold_slider.setTickPosition(QSlider.TicksBelow)
        self._threshold_slider.setTickInterval(5)
        self._threshold_slider.setFixedWidth(200)
        self._threshold_slider.setAttribute(Qt.WA_TranslucentBackground)
        self._threshold_slider.setStyleSheet(
            "QSlider { background: transparent; }"
            "QSlider::groove:horizontal { background: #2a2a3e; "
            "height: 4px; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #17a2b8; "
            "height: 4px; border-radius: 2px; }"
            "QSlider::add-page:horizontal { background: #2a2a3e; "
            "height: 4px; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #17a2b8; "
            "width: 14px; margin: -6px 0; border-radius: 7px; "
            "border: 1px solid #20b2cc; }")
        self._threshold_slider.valueChanged.connect(
            self._on_threshold_changed)
        mode_row.addWidget(self._threshold_slider)
        self._threshold_value_lbl = QLabel(f"{self._threshold}%")
        self._threshold_value_lbl.setAttribute(Qt.WA_TranslucentBackground)
        self._threshold_value_lbl.setStyleSheet(
            "color: #c8cbe0; font-size: 9pt; font-weight: bold; "
            "background: transparent;")
        self._threshold_value_lbl.setFixedWidth(40)
        mode_row.addWidget(self._threshold_value_lbl)

        mode_row.addStretch()
        root.addLayout(mode_row)

        # Extension checkboxes — default to Auto (derived from each
        # missing-raw workspace's known kind). The user can opt out
        # of Auto to pick extensions manually via the individual
        # checkboxes.
        ext_row = QHBoxLayout()
        ext_row.setSpacing(6)
        ext_lbl = QLabel("Extensions:")
        ext_lbl.setAttribute(Qt.WA_TranslucentBackground)
        ext_lbl.setStyleSheet(
            "color: #888; font-size: 8.5pt; background: transparent;")
        ext_row.addWidget(ext_lbl)
        # Every checkbox in this row goes through
        # :func:`_create_styled_checkbox` so the visual language matches
        # the Other Settings dialog exactly (same indicator border /
        # fill / hover colours, same “✓” overlay). The previous
        # bespoke ``ext_css`` stylesheet used teal accent colours and
        # drew a slightly different indicator, so flipping between
        # Other Settings and this dialog made the checkboxes look
        # like they came from two different apps.
        self._auto_cb = _create_styled_checkbox("Auto")
        self._auto_cb.setToolTip(
            "Automatically pick the extensions to scan for based on "
            "each missing-raw card's workspace kind. Turn this off "
            "to override with the checkboxes to the right."
        )
        self._auto_cb.setChecked(self._auto_mode)
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        ext_row.addWidget(self._auto_cb)
        self._ext_cbs: dict[str, QCheckBox] = {}
        for ext, label in (
            ("epub", ".epub"),
            ("txt",  ".txt"),
            ("pdf",  ".pdf"),
            ("html", ".html"),
        ):
            cb = _create_styled_checkbox(label)
            cb.setChecked(ext in self._selected_exts)
            cb.toggled.connect(
                lambda checked, e=ext: self._on_ext_toggled(e, checked))
            self._ext_cbs[ext] = cb
            ext_row.addWidget(cb)
        # Auto mode disables the individual toggles since their
        # values are derived from the books. The checkboxes still
        # *display* the auto-derived state so the user sees which
        # extensions will be scanned.
        self._apply_auto_mode_ui()
        ext_row.addStretch()
        root.addLayout(ext_row)

        # Preview table
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels([
            "", "Workspace", "Matched file", "Similarity"
        ])
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QTreeWidget.SingleSelection)
        # The tree stylesheet handles background / row selection /
        # header; the per-cell checkbox indicator is painted by a
        # custom :class:`QStyledItemDelegate` installed below. We
        # tried styling the indicator purely via
        # ``QTreeView::indicator`` + an inline SVG ``image:`` URL,
        # but Qt’s stylesheet engine doesn’t reliably resolve
        # ``data:`` URIs in ``url()`` — the indicator rendered as a
        # solid blue square with no “✓” glyph on Windows. Painting
        # the indicator manually via a delegate matches
        # :func:`_create_styled_checkbox` (same colours, same tick)
        # byte-for-byte without the data-URL round-trip.
        self._tree.setStyleSheet(
            "QTreeWidget { background: #1a1a2a; color: #e0e0e0; "
            "border: 1px solid #2a2a3e; border-radius: 6px; "
            "font-size: 9pt; }"
            "QTreeWidget::item { padding: 4px; }"
            "QTreeWidget::item:selected { background: #3a3a5e; }"
            "QHeaderView::section { background: #1e1e2e; color: #b0b0c0; "
            "padding: 4px 8px; border: none; border-bottom: "
            "1px solid #2a2a3e; font-weight: bold; font-size: 8.5pt; }")
        # Install the custom delegate on column 0 so the check
        # indicator matches Other Settings. Other columns keep their
        # default rendering.
        from PySide6.QtWidgets import (
            QStyledItemDelegate, QStyle, QStyleOptionViewItem,
        )
        from PySide6.QtGui import QColor, QPen, QBrush, QPainter
        from PySide6.QtCore import QRect as _QRect, Qt as _Qt

        class _CheckmarkDelegate(QStyledItemDelegate):
            """Paint an Other-Settings-styled checkbox on column 0.

            Draws a 14×14 rounded square with a #5a9fd4 border and
            #2d2d2d unchecked fill / #5a9fd4 checked fill, plus a
            white “✓” stroke inside the checked state. Rows whose
            flags lack ``ItemIsUserCheckable`` (the “no match”
            rows) render as an empty column so they don’t look
            interactive. Click handling is left to the base class
            — Qt’s default ``editorEvent`` still toggles the check
            state on the model when the user clicks the cell.
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
                    if not (flags & _Qt.ItemIsUserCheckable):
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
                    raw_cs_int = _cs_int(index.data(_Qt.CheckStateRole))
                    # 2 == ``Qt.CheckState.Checked.value`` in every Qt
                    # build shipped so far; safe to hardcode.
                    is_checked = (opt_cs == 2 or raw_cs_int == 2)
                    rect = opt.rect
                    size = 14
                    box_x = rect.x() + max(
                        4, (rect.width() - size) // 2)
                    box_y = rect.y() + max(
                        2, (rect.height() - size) // 2)
                    box = _QRect(box_x, box_y, size, size)
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
                        painter.setBrush(_Qt.NoBrush)
                        painter.setPen(QPen(
                            QColor("white"), 2,
                            _Qt.SolidLine, _Qt.RoundCap, _Qt.RoundJoin))
                        # Classic “✓”: short leg from (3,7) to (6,10),
                        # long leg from (6,10) to (11,4) inside the box.
                        x = box.x()
                        y = box.y()
                        painter.drawLine(x + 3, y + 7, x + 6, y + 10)
                        painter.drawLine(x + 6, y + 10, x + 11, y + 4)
                finally:
                    painter.restore()

        self._check_delegate = _CheckmarkDelegate(self._tree)
        self._tree.setItemDelegateForColumn(0, self._check_delegate)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        root.addWidget(self._tree, 1)

        # Status line + indeterminate progress bar. The bar is only
        # shown while a background scan is running so the user has a
        # clear visual signal that work is in flight \u2014 the rest
        # of the dialog stays interactive, but widgets that would
        # implicitly restart the scan (Browse button, folder edit)
        # are disabled so a rapid double-click doesn't kick off a
        # second scan before the first finishes.
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        self._status_lbl = QLabel("")
        self._status_lbl.setAttribute(Qt.WA_TranslucentBackground)
        self._status_lbl.setStyleSheet(
            "color: #8ab4d0; font-size: 9pt; background: transparent;")
        status_row.addWidget(self._status_lbl, 1)
        self._scan_progress = QProgressBar()
        self._scan_progress.setRange(0, 0)  # indeterminate
        self._scan_progress.setTextVisible(False)
        self._scan_progress.setFixedHeight(6)
        self._scan_progress.setFixedWidth(160)
        self._scan_progress.setStyleSheet(
            "QProgressBar { background: #1a1a2a; border: 1px solid "
            "#2a2a3e; border-radius: 3px; }"
            "QProgressBar::chunk { background: #17a2b8; "
            "border-radius: 3px; }")
        self._scan_progress.hide()
        status_row.addWidget(self._scan_progress, 0)
        root.addLayout(status_row)

        # Dialog buttons
        bbox = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self._apply_btn = bbox.button(QDialogButtonBox.Apply)
        self._apply_btn.setText("Apply Pairings")
        self._apply_btn.setCursor(Qt.PointingHandCursor)
        self._apply_btn.clicked.connect(self._apply_matches)
        bbox.rejected.connect(self.reject)
        bbox.setStyleSheet(
            "QDialogButtonBox { background: transparent; }"
            "QPushButton { background: #3a5a7a; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; min-width: 120px; }"
            "QPushButton:hover { background: #4a6a8a; }"
            "QPushButton:disabled { background: #2a2a3e; color: #555; }")
        root.addWidget(bbox)

        # Fuzzy slider only meaningful in fuzzy mode — grey it out in
        # exact mode so the UI reads as "similarity doesn't apply".
        self._update_threshold_enabled()

    # -- Scanning-state helper ---------------------------------------------
    def _set_scanning_state(self, active: bool, message: str = ""):
        """Flip every \"scan in progress\" UI affordance at once.

        *active* True shows the progress bar, styles the status
        label as a teal \"scanning\" pill, inserts a placeholder
        row in the tree, and locks the Browse button + folder edit
        so the user can't accidentally restart the scan while the
        previous one is still walking the disk. *active* False
        reverts everything and leaves the tree ready to be
        repainted with real results.
        """
        active = bool(active)
        self._scanning = active
        if active:
            self._scan_progress.show()
            self._status_lbl.setText(
                message or "\u23f3 Scanning\u2026")
            self._status_lbl.setStyleSheet(
                "color: #20b2cc; font-size: 9pt; font-weight: bold; "
                "background: rgba(32, 178, 204, 0.12); "
                "border: 1px solid #17a2b8; border-radius: 4px; "
                "padding: 2px 8px;")
            # Lock the inputs that would implicitly restart the
            # scan. Mode / slider / extension toggles stay live so
            # the user can still tweak them \u2014 those call
            # :meth:`_rematch_only` / :meth:`_rescan_folder` which
            # cancel the in-flight worker first.
            if hasattr(self, "_browse_btn"):
                self._browse_btn.setEnabled(False)
                self._browse_btn.setToolTip(
                    "Scan in progress \u2014 please wait\u2026")
            self._folder_edit.setEnabled(False)
            self._apply_btn.setEnabled(False)
            # Show a clear placeholder row in the tree so stale
            # results from a previous scan don't linger on screen.
            self._tree.blockSignals(True)
            self._tree.clear()
            placeholder = self._QTreeWidgetItem([
                "",
                "\u23f3  Scanning folder\u2026",
                "Matching workspaces against raw sources\u2026",
                "",
            ])
            placeholder.setFlags(Qt.ItemIsEnabled)
            from PySide6.QtGui import QColor
            placeholder.setForeground(1, QColor("#20b2cc"))
            placeholder.setForeground(2, QColor("#8ab4d0"))
            self._tree.addTopLevelItem(placeholder)
            self._tree.blockSignals(False)
        else:
            self._scan_progress.hide()
            self._status_lbl.setStyleSheet(
                "color: #8ab4d0; font-size: 9pt; background: transparent;")
            if hasattr(self, "_browse_btn"):
                self._browse_btn.setEnabled(True)
                self._browse_btn.setToolTip("")
            self._folder_edit.setEnabled(True)

    # -- Folder + mode handlers --------------------------------------------
    def _browse_folder(self):
        # Defence-in-depth: the button is disabled while a scan is
        # running, but guard the handler as well in case the click
        # arrives between state transitions.
        if getattr(self, "_scanning", False):
            return
        start = self._folder_edit.text().strip() or str(Path.home())
        if not os.path.isdir(start):
            start = str(Path.home())
        folder = self._QFileDialog.getExistingDirectory(
            self, "Pick a folder containing raw source files", start)
        if folder:
            self._folder_edit.setText(folder)
            self._on_folder_edit_done()

    def _on_folder_edit_done(self):
        new_folder = self._folder_edit.text().strip()
        if new_folder == self._scan_folder:
            return
        self._scan_folder = new_folder
        try:
            self._config["epub_library_scan_raw_folder"] = new_folder
        except Exception:
            pass
        self._rescan_folder()

    def _on_mode_changed(self, _checked: bool):
        mode = (self.MATCH_EXACT if self._rb_exact.isChecked()
                else self.MATCH_FUZZY)
        if mode == self._mode:
            return
        self._mode = mode
        try:
            self._config["epub_library_scan_raw_mode"] = mode
        except Exception:
            pass
        self._update_threshold_enabled()
        # Mode change is a re-match only, so reuse the cached
        # candidates from the last walk (no folder rescan).
        self._rematch_only()

    def _on_threshold_changed(self, value: int):
        self._threshold = int(value)
        self._threshold_value_lbl.setText(f"{self._threshold}%")
        try:
            self._config["epub_library_scan_raw_threshold"] = self._threshold
        except Exception:
            pass
        if self._mode == self.MATCH_FUZZY:
            self._rematch_only()

    def _update_threshold_enabled(self):
        enabled = self._mode == self.MATCH_FUZZY
        self._threshold_slider.setEnabled(enabled)
        self._threshold_value_lbl.setEnabled(enabled)

    def _on_ext_toggled(self, ext: str, checked: bool):
        """Toggle one of the extension filters and re-scan.

        No-op when Auto is on: the checkbox states are derived from
        the missing-raw workspaces and can't be changed manually
        without first unticking Auto. Refuses to leave the selection
        empty in manual mode — unticking the last checkbox re-ticks
        itself so the user can't put the dialog into a "nothing will
        ever match" state.
        """
        if self._auto_mode:
            return
        ext = ext.lower().lstrip(".")
        if checked:
            self._manual_exts.add(ext)
        else:
            self._manual_exts.discard(ext)
            if not self._manual_exts:
                self._manual_exts.add(ext)
                cb = self._ext_cbs.get(ext)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(True)
                    cb.blockSignals(False)
                return
        self._selected_exts = set(self._manual_exts)
        try:
            self._config["epub_library_scan_raw_exts"] = sorted(
                self._manual_exts)
        except Exception:
            pass
        self._rescan_folder()

    def _on_auto_toggled(self, checked: bool):
        """Flip the Auto extension mode on / off."""
        new_value = bool(checked)
        if new_value == self._auto_mode:
            return
        self._auto_mode = new_value
        try:
            self._config["epub_library_scan_raw_auto"] = new_value
        except Exception:
            pass
        if self._auto_mode:
            self._selected_exts = self._derive_auto_exts()
        else:
            self._selected_exts = set(self._manual_exts)
        self._apply_auto_mode_ui()
        self._rescan_folder()

    def _apply_auto_mode_ui(self):
        """Sync the extension checkboxes to the current mode.

        In Auto mode the checkboxes reflect the derived set but are
        disabled so the user can see what WILL be scanned without
        accidentally tweaking it. In manual mode the checkboxes are
        editable and reflect ``_manual_exts``.
        """
        enabled = not self._auto_mode
        display = (self._derive_auto_exts() if self._auto_mode
                   else self._manual_exts)
        for ext, cb in self._ext_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(ext in display)
            cb.setEnabled(enabled)
            cb.blockSignals(False)

    # -- Scanning + matching -----------------------------------------------
    def _cancel_worker(self):
        if self._scan_worker is not None:
            try:
                self._scan_worker.cancel()
                self._scan_worker.results.disconnect()
            except Exception:
                pass
            self._scan_worker = None

    # _ext_suffixes moved verbatim to library_core.ScanForRawMixin (inherited).

    def _rescan_folder(self):
        """Kick off a background walk + match of the current folder.

        Walk AND match both run on :class:`_RawScanWorker` so the
        dialog stays responsive even for huge folders with heavy
        Fuzzy matching. Any previously running worker is cancelled
        first so rapid folder / extension toggles don't leave
        multiple scans racing.
        """
        self._candidates.clear()
        self._cancel_worker()
        if not self._scan_folder or not os.path.isdir(self._scan_folder):
            self._set_scanning_state(False)
            self._status_lbl.setText(
                "\u26a0 Pick a folder that exists on disk.")
            self._matches.clear()
            self._tree.clear()
            self._apply_btn.setEnabled(False)
            return
        # Paint a "scanning\u2026" placeholder so the UI reflects the
        # in-flight state rather than silently showing stale rows.
        self._set_scanning_state(
            True,
            f"\u23f3 Scanning {self._scan_folder}\u2026")
        self._scan_worker = _RawScanWorker(
            self._scan_folder, self._ext_suffixes(),
            _LIBRARY_TRACKING_FILENAMES,
            books=self._books, mode=self._mode,
            threshold=self._threshold,
            parent=self,
        )
        self._scan_worker.results.connect(self._on_scan_results)
        self._scan_worker.start()

    def _rematch_only(self):
        """Re-run matching against the cached candidate list.

        Used by mode / slider changes so we don't re-walk the
        folder every time the user drags the Similarity slider.
        Falls back to a full rescan when no cached candidates exist
        yet (e.g. the user hasn't picked a folder).
        """
        if not self._candidates:
            self._rescan_folder()
            return
        self._cancel_worker()
        self._set_scanning_state(
            True, "\u23f3 Re-matching\u2026")
        self._scan_worker = _RawScanWorker(
            self._scan_folder, self._ext_suffixes(),
            _LIBRARY_TRACKING_FILENAMES,
            books=self._books, mode=self._mode,
            threshold=self._threshold,
            prewalked=list(self._candidates),
            parent=self,
        )
        self._scan_worker.results.connect(self._on_scan_results)
        self._scan_worker.start()

    @Slot(str, list, dict)
    def _on_scan_results(self, scan_folder: str,
                         candidates: list,
                         matches: dict) -> None:
        """Worker callback \u2014 runs on the main thread via Qt signal.

        The worker did both the walk and the matching, so all we
        need to do is cache candidates and paint the tree.
        """
        if scan_folder != self._scan_folder:
            return
        self._candidates = list(candidates)
        self._matches = dict(matches)
        # Clear the scanning-state UI BEFORE painting results so the
        # progress bar / placeholder row aren't briefly visible
        # alongside the real rows.
        self._set_scanning_state(False)
        self._populate_tree()

    def _populate_tree(self):
        self._tree.blockSignals(True)
        self._tree.clear()
        hits = 0
        for ws_folder, info in self._matches.items():
            book = info["book"]
            # Column 1 shows the workspace's ON-DISK folder basename
            # so the user can compare it directly to the matched raw
            # filename in column 2. The metadata-driven ``book['name']``
            # is the translated title (e.g. "The Slaves I Kicked Out")
            # which is useless for filename matching — the folder name
            # (e.g. "[393761] ㄝㅇㅏㄴㄴㄴ...") is what the
            # scanner actually pairs against.
            workspace_label = (book.get("folder_name")
                               or os.path.basename(ws_folder)
                               or book.get("name")
                               or "")
            item = self._QTreeWidgetItem([
                "",
                str(workspace_label),
                (os.path.basename(info["path"])
                 if info["path"] else "\u2014 no match"),
                (f"{info['ratio'] * 100:.0f}%"
                 if info["ratio"] > 0 else "\u2014"),
            ])
            # Tooltip on the workspace column reveals the full folder
            # path for users who want to verify which workspace on
            # disk the row points at.
            if ws_folder:
                item.setToolTip(1, ws_folder)
            item.setData(0, Qt.UserRole, ws_folder)
            if info["path"]:
                item.setToolTip(2, info["path"])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.Checked if info["accepted"] else Qt.Unchecked)
                hits += 1
            else:
                # No candidate — disable the checkbox row and paint
                # the workspace name in a dimmed color so it reads
                # as unreachable. ``QPalette.Disabled`` /
                # ``QPalette.Text`` are *class-level* enums on
                # ``QPalette`` itself; accessing them through a
                # palette *instance* (as in
                # ``QApplication.palette().Disabled``) raises
                # ``AttributeError`` on PySide6 because the instance
                # doesn't re-export the enum values.
                from PySide6.QtGui import QPalette, QColor
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable
                              & ~Qt.ItemIsSelectable)
                try:
                    item.setForeground(
                        1, QApplication.palette().brush(
                            QPalette.Disabled, QPalette.Text))
                except Exception:
                    # Palette brush lookup failed on some Qt builds
                    # (themed stylesheets sometimes strip the
                    # Disabled group). Fall back to a hardcoded
                    # dimmed grey so the row still reads as
                    # unreachable.
                    item.setForeground(1, QColor("#6a6d80"))
            self._tree.addTopLevelItem(item)
        self._tree.blockSignals(False)
        self._status_lbl.setText(self._scan_status_text(hits))
        self._apply_btn.setEnabled(hits > 0)

    # _scan_status_text was extracted from _populate_tree into library_core.ScanForRawMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _on_tree_item_changed(self, item, column: int):
        if column != 0:
            return
        ws_folder = item.data(0, Qt.UserRole)
        if not ws_folder or ws_folder not in self._matches:
            return
        self._matches[ws_folder]["accepted"] = (
            item.checkState(0) == Qt.Checked)

    # -- Qt lifecycle ------------------------------------------------------
    def closeEvent(self, event):
        """Cancel the background walk before the dialog is torn down.

        Without this the ``QThread`` can outlive the Python object;
        when it finally emits ``results`` it'd try to poke at a
        deleted C++ widget and crash the interpreter.
        """
        if self._scan_worker is not None:
            _stop_qthread_safely(
                self._scan_worker,
                timeout_ms=1200,
                signal_names=("results",),
            )
            self._scan_worker = None
        super().closeEvent(event)

    # -- Apply -------------------------------------------------------------
    def _apply_matches(self):
        written = self._write_raw_pairings()
        self.applied.emit(written)
        QMessageBox.information(
            self, "Scan for Raw",
            f"Linked {written} workspace"
            f"{'s' if written != 1 else ''} to a raw source file. "
            "The library will refresh in a moment."
            if written else
            "No pairings were applied."
        )
        if written:
            self.accept()

    # _write_raw_pairings was extracted from _apply_matches into library_core.ScanForRawMixin (see DISCREPANCIES U5 "Phase-1 splits").


# ---------------------------------------------------------------------------
# Library Dialog
# ---------------------------------------------------------------------------

class EpubLibraryDialog(LibraryShelfMixin, QDialog):
    # Emitted when the user imports a new EPUB from the "In Progress" tab.
    # Parents (e.g. TranslatorGUI) can connect to set it as the input file.
    import_epub_requested = Signal(str)
    # Emitted when the user triggers a multi-card "Load N for translation"
    # action. Payload is a list of absolute paths.
    import_epubs_requested = Signal(list)
    # Emitted after an Organize / Undo Move operation relocates files on
    # disk. Payload is a list of ``(old_abs_path, new_abs_path)`` tuples.
    # Parents (e.g. TranslatorGUI) use it to update any stale paths in
    # their own state — selected_files, entry_epub text, etc. — so the
    # user doesn't end up clicking Run on a path that no longer exists
    # after the raw was moved into Library/Raw.
    files_reorganized = Signal(list)

    _CARD_STREAM_BATCH_SIZE = 2
    _CARD_STREAM_TICK_MS = 16
    _CARD_STREAM_FRAME_BUDGET_SEC = 0.007
    _COVER_APPLY_BATCH_SIZE = 1
    _COVER_APPLY_TICK_MS = 16
    _COVER_PUMP_IDLE_MS = 450
    _COVER_PUMP_BUSY_MS = 140
    _INACTIVE_TAB_IDLE_MS = 1200
    _CARD_HOVER_RECONCILE_MS = 50
    _CARD_HOVER_SETTLE_MS = 6000
    _GRID_SCROLLBAR_RESERVE = 8
    _WHEEL_SCROLL_PIXELS = 110
    _WHEEL_SCROLL_DURATION_MS = 145

    def __init__(self, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\U0001f4da Glossarion Library")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        # Ratio-based sizing relative to screen
        screen = self.screen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.55), int(avail.height() * 0.65))
            self.setMinimumSize(int(avail.width() * 0.35), int(avail.height() * 0.4))
        else:
            self.resize(900, 650)
            self.setMinimumSize(500, 400)
        self._config = config or {}
        # Per-tab data. "In Progress" comes from the output root (strict
        # OUTPUT_DIRECTORY override); "Completed" comes from the Library folder.
        self._in_progress_books: list[dict] = []
        self._completed_books: list[dict] = []
        self._ip_cards: list[_BookCard] = []
        self._comp_cards: list[_BookCard] = []
        # Multi-selection state, keyed by ``book['path']``. One set per tab
        # so switching tabs doesn't clobber the other tab's selection. The
        # sets are consulted every time ``_populate_grid_common`` rebuilds
        # the cards (which happens on scan / auto-refresh / sort / filter)
        # so selection survives refreshes as long as the book path is
        # still present in the scan result.
        self._selected_paths_ip: set[str] = set()
        self._selected_paths_comp: set[str] = set()
        self._library_pages: dict[str, int] = {"ip": 0, "comp": 0}
        self._library_pagers: dict[str, dict] = {}
        self._cover_threads: list[_CoverLoader] = []
        # Cover-path cache: maps ``book_path \u2192 resolved_cover_path``.
        # Populated by :meth:`_on_cover_loaded` the first time a
        # :class:`_CoverLoader` finishes for a book. Subsequent card
        # rebuilds (e.g. when the user flips the View size dropdown,
        # which invalidates the per-card ``preset_key`` cache) consult
        # this map in :meth:`_populate_grid_common` and call
        # :meth:`_BookCard.set_cover` with the cached path instead of
        # spawning another ``_CoverLoader`` QThread \u2014 so switching
        # between S / M / L / XL doesn't pay an N-thread
        # ``_extract_cover`` re-scan per card. Entries that previously
        # resolved to no cover (empty string) are cached too so the
        # loader doesn't re-run for known-failing books either.
        self._cover_path_cache: dict[str, str] = {}
        self._card_stream_generation: dict[str, int] = {}
        self._card_stream_states: dict[str, dict] = {}
        self._card_stream_timers: dict[str, QTimer] = {}
        self._active_card_stream_key: str | None = None
        self._dirty_card_tabs: set[str] = {"ip", "comp"}
        self._pending_card_reflow: set[str] = set()
        self._grid_layout_widths: dict[str, int] = {}
        self._cover_queue = deque()
        self._cover_queued_paths: set[str] = set()
        self._cover_active_loaders: dict[_CoverLoader, str] = {}
        self._cover_active_paths: set[str] = set()
        self._cover_apply_queue = deque()
        self._cover_apply_pending_paths: set[str] = set()
        self._cover_generation = 0
        self._library_scroll_animations: dict[str, QPropertyAnimation] = {}
        self._library_scroll_targets: dict[str, int] = {}
        self._hovered_card: _BookCard | None = None
        self._card_hover_reconcile_until = 0.0
        # Restore persisted settings
        self._sort_mode = self._config.get('epub_library_sort', SORT_DATE)
        self._card_size = self._config.get('epub_library_card_size', SIZE_COMPACT)
        self._current_tab = self._config.get('epub_library_tab', 0)
        # File-format filter (EPUB / TXT / PDF / HTML / Image / All).
        # Persisted per-user so the chosen chip survives across
        # library-dialog opens.
        self._format_filter = self._config.get(
            'epub_library_format_filter', FORMAT_ALL)
        if self._format_filter not in {
            FORMAT_ALL, FORMAT_EPUB, FORMAT_TXT,
            FORMAT_PDF, FORMAT_HTML, FORMAT_IMAGE,
        }:
            self._format_filter = FORMAT_ALL
        # When True, every flash card renders the raw source-language title
        # instead of the (possibly translated) ``book['name']``. See
        # :func:`_card_raw_title` for the resolution rules.
        self._show_raw_titles = bool(
            self._config.get('epub_library_show_raw_titles', False)
        )
        self._scanner_thread: _DualScannerThread | None = None
        self._delete_thread: _LibraryDeleteThread | None = None
        self._delete_progress = None
        self._shutdown_requested = False
        self._active_details = None
        self._active_reader = None
        self._initial_scan_started = False
        # Enable drag-and-drop of EPUB / TXT / PDF / HTML files onto the
        # dialog. Drops are routed through the same import pipeline as the
        # "Import EPUB" button (see :meth:`_import_paths_into_library`).
        self.setAcceptDrops(True)
        self._setup_ui()
        self._grid_reflow_timer = QTimer(self)
        self._grid_reflow_timer.setSingleShot(True)
        self._grid_reflow_timer.timeout.connect(self._run_pending_grid_reflow)
        self._inactive_tab_populate_timer = QTimer(self)
        self._inactive_tab_populate_timer.setSingleShot(True)
        self._inactive_tab_populate_timer.timeout.connect(
            self._populate_inactive_tab_if_idle)
        self._cover_apply_timer = QTimer(self)
        self._cover_apply_timer.setSingleShot(False)
        self._cover_apply_timer.setInterval(self._COVER_APPLY_TICK_MS)
        self._cover_apply_timer.timeout.connect(self._apply_cover_batch)
        self._cover_pump_timer = QTimer(self)
        self._cover_pump_timer.setSingleShot(True)
        self._cover_pump_timer.timeout.connect(self._pump_cover_queue)
        self._card_hover_reconcile_timer = QTimer(self)
        self._card_hover_reconcile_timer.setSingleShot(False)
        self._card_hover_reconcile_timer.setInterval(
            self._CARD_HOVER_RECONCILE_MS)
        self._card_hover_reconcile_timer.timeout.connect(
            self._reconcile_card_hover)
        # Flip the dialog into its loading state BEFORE the caller
        # shows it. Previously ``_show_loading`` was called by the
        # deferred :meth:`_load_books` tick, which meant the dialog's
        # first paint rendered empty tabs (no spinner, no cards) for
        # one event-loop iteration — visible to the user as a blank
        # window for a noticeable fraction of a second. Doing the
        # widget swap here guarantees the very first paint already
        # shows the Halgakos spinner + "Scanning library…" strip.
        self._show_loading()
        # The filesystem scan is intentionally visible-demand: hidden
        # startup prewarm should polish the widgets, not enumerate every
        # output folder. showEvent starts the worker thread when the user
        # actually opens the Library.
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self._auto_refresh)

    def _warm_library_widgets(self) -> None:
        """Polish/layout library tabs and currently built cards."""
        app = QApplication.instance()
        try:
            self.ensurePolished()
            layout = self.layout()
            if layout is not None:
                layout.activate()
            for widget in (
                getattr(self, "_tabs", None),
                getattr(self, "_ip_scroll", None),
                getattr(self, "_comp_scroll", None),
                getattr(self, "_ip_grid_container", None),
                getattr(self, "_comp_grid_container", None),
            ):
                if widget is None:
                    continue
                try:
                    widget.ensurePolished()
                    widget.updateGeometry()
                    widget_layout = widget.layout()
                    if widget_layout is not None:
                        widget_layout.activate()
                except Exception:
                    pass
            for card in [*getattr(self, "_ip_cards", []), *getattr(self, "_comp_cards", [])]:
                try:
                    card.ensurePolished()
                    card.updateGeometry()
                    card_layout = card.layout()
                    if card_layout is not None:
                        card_layout.activate()
                except Exception:
                    pass
            if app is not None:
                app.processEvents(QEventLoop.ExcludeUserInputEvents)
        except Exception:
            pass

    def _center_on_screen(self) -> None:
        try:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is None:
                return
            geo = screen.availableGeometry()
            self.move(
                geo.x() + max(0, (geo.width() - self.width()) // 2),
                geo.y() + max(0, (geo.height() - self.height()) // 2),
            )
        except Exception:
            pass

    def _book_card_from_widget(self, widget) -> _BookCard | None:
        while widget is not None:
            if isinstance(widget, _BookCard):
                return widget
            if widget is self:
                break
            try:
                widget = widget.parentWidget()
            except Exception:
                break
        return None

    def _set_hovered_card(self, card: _BookCard | None) -> None:
        if card is not None and not card.isVisible():
            card = None
        old = getattr(self, "_hovered_card", None)
        if old is card:
            return
        self._hovered_card = card
        if old is not None:
            try:
                old.set_hovered(False)
            except RuntimeError:
                pass
        if card is not None:
            try:
                card.set_hovered(True)
            except RuntimeError:
                self._hovered_card = None

    def _on_card_hover_changed(self, card: _BookCard, hovered: bool) -> None:
        if hovered:
            self._set_hovered_card(card)
        elif getattr(self, "_hovered_card", None) is card:
            self._set_hovered_card(None)

    def _request_card_hover_reconcile(self, settle_ms: int | None = None) -> None:
        if not self.isVisible():
            return
        ms = self._CARD_HOVER_SETTLE_MS if settle_ms is None else max(0, int(settle_ms))
        self._card_hover_reconcile_until = max(
            getattr(self, "_card_hover_reconcile_until", 0.0),
            time.perf_counter() + (ms / 1000.0),
        )
        timer = getattr(self, "_card_hover_reconcile_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _reconcile_card_hover(self) -> None:
        card = None
        if self.isVisible():
            try:
                widget = QApplication.widgetAt(QCursor.pos())
            except Exception:
                widget = None
            try:
                if widget is not None and (widget is self or self.isAncestorOf(widget)):
                    card = self._book_card_from_widget(widget)
            except Exception:
                card = None
        self._set_hovered_card(card)

        busy = (
            self._is_card_stream_active()
            or bool(getattr(self, "_cover_queue", None))
            or bool(getattr(self, "_cover_active_loaders", None))
            or bool(getattr(self, "_cover_apply_queue", None))
        )
        now = time.perf_counter()
        if busy:
            self._card_hover_reconcile_until = max(
                getattr(self, "_card_hover_reconcile_until", 0.0),
                now + 0.35,
            )
        if now >= getattr(self, "_card_hover_reconcile_until", 0.0):
            timer = getattr(self, "_card_hover_reconcile_timer", None)
            if timer is not None:
                timer.stop()

    def _clear_card_hover(self) -> None:
        self._card_hover_reconcile_until = 0.0
        timer = getattr(self, "_card_hover_reconcile_timer", None)
        if timer is not None:
            timer.stop()
        self._set_hovered_card(None)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.modifiers() & Qt.ControlModifier:
            sizes = _ALL_SIZES
            idx = sizes.index(self._card_size) if self._card_size in sizes else 0
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                if idx < len(sizes) - 1:
                    self._set_card_size(sizes[idx + 1])
            elif event.key() == Qt.Key_Minus:
                if idx > 0:
                    self._set_card_size(sizes[idx - 1])
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Ctrl+Wheel to zoom card size."""
        if event.modifiers() & Qt.ControlModifier:
            self._handle_ctrl_wheel(event.angleDelta().y())
            event.accept()
        else:
            super().wheelEvent(event)

    def _handle_ctrl_wheel(self, delta_y: int) -> None:
        """Step the card-size preset up / down based on wheel direction."""
        if not delta_y:
            return
        sizes = _ALL_SIZES
        idx = sizes.index(self._card_size) if self._card_size in sizes else 0
        if delta_y > 0 and idx < len(sizes) - 1:
            self._set_card_size(sizes[idx + 1])
        elif delta_y < 0 and idx > 0:
            self._set_card_size(sizes[idx - 1])

    def _library_scroll_area_for_event_object(self, obj):
        """Return ``(key, area)`` when *obj* belongs to a card-grid scroller."""
        for key, area, grid in (
            ("ip", getattr(self, "_ip_scroll", None),
             getattr(self, "_ip_grid_container", None)),
            ("comp", getattr(self, "_comp_scroll", None),
             getattr(self, "_comp_grid_container", None)),
        ):
            if area is None:
                continue
            try:
                if obj is area or obj is area.viewport() or obj is grid:
                    return key, area
            except RuntimeError:
                continue
        return None, None

    def _animate_library_scroll(
        self, key: str, area: QScrollArea, delta_y: int
    ) -> bool:
        """Smooth one angle-wheel movement while accumulating rapid ticks."""
        if not delta_y:
            return False
        bar = area.verticalScrollBar()
        if bar.maximum() <= bar.minimum():
            return False

        animation = self._library_scroll_animations.get(key)
        running = bool(
            animation is not None
            and animation.state() == QAbstractAnimation.Running
        )
        base_target = (
            self._library_scroll_targets.get(key, bar.value())
            if running else bar.value()
        )
        wheel_ticks = float(delta_y) / 120.0
        movement = int(round(-wheel_ticks * self._WHEEL_SCROLL_PIXELS))
        if movement == 0:
            movement = -1 if delta_y > 0 else 1
        target = max(bar.minimum(), min(bar.maximum(), base_target + movement))
        if target == bar.value() and not running:
            return True

        if animation is None:
            animation = QPropertyAnimation(bar, b"value", self)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            self._library_scroll_animations[key] = animation
        else:
            animation.stop()
        self._library_scroll_targets[key] = target
        animation.setDuration(self._WHEEL_SCROLL_DURATION_MS)
        animation.setStartValue(bar.value())
        animation.setEndValue(target)
        animation.start()
        return True

    def eventFilter(self, obj, event):
        """Route Ctrl+Wheel on the grid scroll areas into the card zoom.

        Without this filter :class:`QScrollArea` swallows every wheel
        event to drive its own vertical scrolling, so the dialog's
        :meth:`wheelEvent` never runs when the cursor is over the grid
        \u2014 which is where the user naturally scrolls. Intercepting
        the wheel event here, checking for Ctrl, and forwarding it to
        :meth:`_handle_ctrl_wheel` lets the same gesture zoom cards
        regardless of where the cursor lives. Plain wheel events fall
        through so normal grid scrolling keeps working.
        """
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Resize:
            # The dialog can finish resizing before a hidden/newly shown
            # QScrollArea viewport receives its final geometry. Those viewport
            # resize events are the authoritative signal that the frozen
            # column count may be stale.
            tab_key = None
            for key, area in (
                ("ip", getattr(self, "_ip_scroll", None)),
                ("comp", getattr(self, "_comp_scroll", None)),
            ):
                try:
                    if area is not None and obj is area.viewport():
                        tab_key = key
                        break
                except Exception:
                    continue
            if tab_key is not None:
                try:
                    raw_width_changed = (
                        event.oldSize().width() != event.size().width()
                    )
                except Exception:
                    raw_width_changed = True
                grid_widget = (
                    getattr(self, "_comp_grid_container", None)
                    if tab_key == "comp"
                    else getattr(self, "_ip_grid_container", None)
                )
                current_width = self._effective_grid_width(area, grid_widget)
                previous_width = self._grid_layout_widths.get(tab_key)
                if (
                    raw_width_changed
                    and hasattr(self, "_grid_reflow_timer")
                    and current_width > 0
                    and current_width != previous_width
                    and (
                        bool(self._cards_for_tab_key(tab_key))
                        or self._is_card_stream_active(tab_key)
                    )
                ):
                    self._schedule_grid_reflow(tab_key)
        if event.type() in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop, QEvent.DragLeave):
            return self._handle_library_drop_filter_event(event)
        if event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                try:
                    self._handle_ctrl_wheel(event.angleDelta().y())
                except Exception:
                    pass
                event.accept()
                return True
            # High-resolution touchpads provide pixelDelta and already scroll
            # fluidly through QScrollArea. Only animate coarse angle-wheel
            # ticks; replacing pixel scrolling would make touchpads feel less
            # direct and introduce needless latency.
            try:
                pixel_delta_y = event.pixelDelta().y()
            except Exception:
                pixel_delta_y = 0
            if not pixel_delta_y:
                key, area = self._library_scroll_area_for_event_object(obj)
                try:
                    handled = bool(
                        area is not None
                        and self._animate_library_scroll(
                            key, area, event.angleDelta().y()))
                except Exception:
                    handled = False
                if handled:
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    def _make_sort_btn(self, text, tooltip, mode):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(26)
        btn.setMinimumWidth(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(mode == self._sort_mode)
        btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 8px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        btn.clicked.connect(lambda: self._set_sort(mode))
        return btn

    def _make_format_btn(self, text, tooltip, fmt_key):
        """Button factory for the file-format filter chips.

        Styled like the sort chips so the grouped toolbar reads as a
        single filter strip. Uses a teal "checked" colour to stay
        visually distinct from the purple sort chips.
        """
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(26)
        btn.setMinimumWidth(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(fmt_key == self._format_filter)
        btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 8px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #17a2b8; border-color: #20b2cc; color: #fff; }
        """)
        btn.clicked.connect(lambda: self._set_format_filter(fmt_key))
        return btn

    def _make_size_btn(self, text, tooltip, size_key):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(28, 26)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(size_key == self._card_size)
        btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 9pt; font-weight: bold; padding: 0; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #17a2b8; border-color: #20b2cc; color: #fff; }
        """)
        btn.clicked.connect(lambda: self._set_card_size(size_key))
        return btn

    def _make_library_pager(self, tab_key: str) -> QWidget:
        """Create Book-Details-style pagination controls for one card grid."""
        pager = QWidget()
        pager.setObjectName("library-pager")
        layout = QHBoxLayout(pager)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        layout.addStretch()

        def _button(text: str, role: str, tooltip: str, action: str):
            button = QPushButton(text)
            button.setObjectName(f"library-page-{role}")
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, key=tab_key, act=action:
                    self._on_library_page_action(key, act))
            layout.addWidget(button)
            return button

        first = _button("\u00ab", "first", "First library page", "first")
        previous = _button(
            "\u2039", "prev", "Previous library page", "previous")
        page_label = QLabel("Page 1 / 1 \u00b7 0 of 0")
        page_label.setObjectName("library-page-label")
        layout.addWidget(page_label)
        next_button = _button(
            "\u203a", "next", "Next library page", "next")
        last = _button("\u00bb", "last", "Last library page", "last")

        page_size = _NoWheelComboBox()
        page_size.setObjectName("library-page-size")
        page_size.setToolTip("Library cards per page")
        for label, value in (
            ("20 / page", 20),
            ("50 / page", 50),
            ("100 / page", 100),
            ("250 / page", 250),
            ("500 / page", 500),
            ("All", "all"),
        ):
            page_size.addItem(label, value)
        stored_page_size = self._config.get("epub_library_page_size", 20)
        if str(stored_page_size).strip().lower() == "all":
            page_size_index = page_size.findData("all")
        else:
            try:
                page_size_index = page_size.findData(int(stored_page_size))
            except (TypeError, ValueError):
                page_size_index = page_size.findData(20)
        page_size.setCurrentIndex(
            page_size_index if page_size_index >= 0 else 0)
        page_size.currentIndexChanged.connect(
            lambda _index, key=tab_key:
                self._on_library_page_size_changed(key))
        layout.addWidget(page_size)
        layout.addStretch()

        icon_path = _find_halgakos_icon()
        icon_rule = ""
        if icon_path:
            icon_rule = (
                'QComboBox#library-page-size::down-arrow {'
                f' image: url("{icon_path.replace(chr(92), "/")}");'
                ' width: 16px; height: 16px; }'
            )
        pager.setStyleSheet("""
            QLabel#library-page-label {
                color: #9aa2b8; font-size: 9pt;
            }
            QPushButton#library-page-first, QPushButton#library-page-prev,
            QPushButton#library-page-next, QPushButton#library-page-last {
                background: #202036; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 0px 10px; font-size: 11pt; font-weight: bold;
                min-width: 28px; min-height: 34px; max-height: 34px;
            }
            QPushButton#library-page-first:hover,
            QPushButton#library-page-prev:hover,
            QPushButton#library-page-next:hover,
            QPushButton#library-page-last:hover {
                border-color: #6c63ff; background: #282848;
            }
            QPushButton#library-page-first:disabled,
            QPushButton#library-page-prev:disabled,
            QPushButton#library-page-next:disabled,
            QPushButton#library-page-last:disabled {
                color: #555a70; background: #171724; border-color: #28283d;
            }
            QComboBox#library-page-size {
                background: #1e1e2e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 5px 30px 5px 9px; font-size: 9pt;
                min-height: 24px; min-width: 112px;
            }
            QComboBox#library-page-size:hover {
                border-color: #6c63ff; background: #24243a;
            }
            QComboBox#library-page-size::drop-down {
                subcontrol-origin: padding; subcontrol-position: top right;
                width: 28px; border-left: 1px solid #3a3a5e;
                border-top-right-radius: 6px; border-bottom-right-radius: 6px;
                background: #202036;
            }
            QComboBox#library-page-size QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0;
                border: 1px solid #3a3a5e;
                selection-background-color: #2a2d5a;
                selection-color: #ffffff;
            }
        """ + icon_rule)
        self._library_pagers[tab_key] = {
            "widget": pager,
            "first": first,
            "previous": previous,
            "label": page_label,
            "next": next_button,
            "last": last,
            "page_size": page_size,
        }
        return pager

    def _setup_ui(self):
        from PySide6.QtWidgets import QTabWidget
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        # Halgakos icon next to title (HiDPI-aware)
        icon_path = _find_halgakos_icon()
        if icon_path:
            icon_lbl = QLabel()
            pm = QPixmap(icon_path)
            if not pm.isNull():
                dpr = self.devicePixelRatio() or 1.0
                logical_size = 28
                raw = int(logical_size * dpr)
                scaled = pm.scaled(raw, raw, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                scaled.setDevicePixelRatio(dpr)
                icon_lbl.setPixmap(scaled)
                icon_lbl.setFixedSize(logical_size + 2, logical_size + 2)
                header.addWidget(icon_lbl)
        title = QLabel("\U0001f4da  Glossarion Library")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #e0e0e0;")
        header.addWidget(title)
        header.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Filter title or tag…")
        self._search.setFixedWidth(200)
        self._search.setStyleSheet(
            "background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 6px; "
            "padding: 4px 10px; color: #e0e0e0; font-size: 9.5pt;"
        )
        self._search.textChanged.connect(self._apply_filter)
        header.addWidget(self._search)

        refresh_btn = QPushButton("\U0001f504  Refresh")
        refresh_btn.setToolTip("Refresh library")
        refresh_btn.setFixedHeight(28)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 9pt; "
            "color: #888; padding: 2px 8px; }"
            "QPushButton:hover { color: #e0e0e0; }")
        refresh_btn.clicked.connect(self._load_books)
        header.addWidget(refresh_btn)

        root.addLayout(header)
        root.addSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(sort_lbl)
        self._sort_btns = {}
        for text, tip, mode in [
            ("🕐 Date", "Sort by date (newest first)", SORT_DATE),
            ("A-Z", "Sort by name (alphabetical)", SORT_NAME),
            ("📏 Size", "Sort by file size (largest first)", SORT_SIZE),
        ]:
            btn = self._make_sort_btn(text, tip, mode)
            self._sort_btns[mode] = btn
            toolbar.addWidget(btn)
        toolbar.addSpacing(16)
        # Format filter dropdown (replaces the previous row of chip
        # buttons). :class:`_NoWheelComboBox` prevents the filter from
        # accidentally flipping when the user scrolls the toolbar. Each
        # item carries its FORMAT_* key as ``itemData`` so the handler
        # never has to map a display label back to a filter key.
        format_lbl = QLabel("Format:")
        format_lbl.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(format_lbl)
        self._format_combo = _NoWheelComboBox()
        self._format_combo.setFixedHeight(26)
        # Narrow footprint: the longest label is "HTML" (4 chars). The
        # width only needs to cover that + the drop-down arrow + a
        # small padding budget, so the combo doesn't hog toolbar space
        # that the Raw titles / Open Library Folder buttons want.
        self._format_combo.setFixedWidth(72)
        self._format_combo.setCursor(Qt.PointingHandCursor)
        self._format_combo.setFocusPolicy(Qt.StrongFocus)
        self._format_combo.setToolTip(
            "Filter library cards by file format (All / EPUB / TXT / "
            "PDF / HTML / image). Mouse wheel is intentionally ignored.")
        self._format_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; font-weight: bold; padding: 2px 8px;
            }
            QComboBox:hover { border-color: #20b2cc; color: #fff; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: url(noimg); width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #17a2b8;
                border: 1px solid #3a3a5e; font-size: 8.5pt;
            }
        """)
        self._format_options = [
            (FORMAT_ALL,   "All",  "Show every file type"),
            (FORMAT_EPUB,  "EPUB", "Show only EPUB books / workspaces"),
            (FORMAT_TXT,   "TXT",  "Show only TXT translations"),
            (FORMAT_PDF,   "PDF",  "Show only PDF translations"),
            (FORMAT_HTML,  "HTML", "Show only HTML translations"),
            (FORMAT_IMAGE, "IMG",  "Show only image-based (manga / comic) workspaces"),
        ]
        for key, label, tip in self._format_options:
            self._format_combo.addItem(label, key)
            self._format_combo.setItemData(
                self._format_combo.count() - 1, tip, Qt.ToolTipRole)
        # Restore persisted selection before wiring the signal so the
        # initial ``setCurrentIndex`` doesn't spuriously refresh the view.
        for i, (key, _label, _tip) in enumerate(self._format_options):
            if key == self._format_filter:
                self._format_combo.setCurrentIndex(i)
                break
        self._format_combo.currentIndexChanged.connect(
            lambda idx: self._set_format_filter(
                self._format_combo.itemData(idx)))
        toolbar.addWidget(self._format_combo)
        toolbar.addSpacing(16)
        # View / thumbnail-size dropdown (replaces the previous row of
        # S / M / L / XL… chip buttons). Same :class:`_NoWheelComboBox`
        # pattern as the Format filter so a mouse wheel can't silently
        # resize every card while the user is trying to scroll.
        size_lbl = QLabel("View:")
        size_lbl.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(size_lbl)
        self._size_combo = _NoWheelComboBox()
        self._size_combo.setFixedHeight(26)
        # Narrow footprint: the longest label is "6XL" (3 chars).
        self._size_combo.setFixedWidth(62)
        self._size_combo.setCursor(Qt.PointingHandCursor)
        self._size_combo.setFocusPolicy(Qt.StrongFocus)
        self._size_combo.setToolTip(
            "Thumbnail size for the library grid. Mouse wheel is "
            "intentionally ignored \u2014 use Ctrl+Wheel / Ctrl+= / Ctrl+- "
            "to zoom cards instead.")
        self._size_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; font-weight: bold; padding: 2px 8px;
            }
            QComboBox:hover { border-color: #20b2cc; color: #fff; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox::down-arrow { image: url(noimg); width: 10px; height: 10px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #17a2b8;
                border: 1px solid #3a3a5e; font-size: 8.5pt;
            }
        """)
        self._size_options = [
            (SIZE_2XS,     "2XS", "2XS thumbnails"),
            (SIZE_XS,      "XS",  "Extra small thumbnails"),
            (SIZE_COMPACT, "S",   "Compact thumbnails"),
            (SIZE_NORMAL,  "M",   "Normal thumbnails"),
            (SIZE_LARGE,   "L",   "Large thumbnails"),
            (SIZE_XL,      "XL",  "Extra large thumbnails"),
            (SIZE_2XL,     "2XL", "2XL thumbnails"),
            (SIZE_3XL,     "3XL", "3XL thumbnails"),
            (SIZE_4XL,     "4XL", "4XL thumbnails"),
            (SIZE_5XL,     "5XL", "5XL thumbnails"),
            (SIZE_6XL,     "6XL", "6XL thumbnails"),
        ]
        for key, label, tip in self._size_options:
            self._size_combo.addItem(label, key)
            self._size_combo.setItemData(
                self._size_combo.count() - 1, tip, Qt.ToolTipRole)
        for i, (key, _label, _tip) in enumerate(self._size_options):
            if key == self._card_size:
                self._size_combo.setCurrentIndex(i)
                break
        self._size_combo.currentIndexChanged.connect(
            lambda idx: self._set_card_size(
                self._size_combo.itemData(idx)))
        toolbar.addWidget(self._size_combo)
        toolbar.addSpacing(16)
        # "Show raw titles" toggle: when checked, every flash card displays
        # the raw source-language title instead of the translated one. Same
        # visual language as the sort buttons so it reads as a persistent
        # filter rather than a one-shot action.
        self._raw_titles_btn = QPushButton("\U0001f524  Raw titles")
        self._raw_titles_btn.setToolTip(
            "Show the raw source filename on every card instead of the \n"
            "translated / compiled-EPUB title. Falls back to the source-\n"
            "language title from metadata when no source filename is \n"
            "available. Useful for finding a book by the name it has \n"
            "on disk."
        )
        self._raw_titles_btn.setFixedHeight(26)
        self._raw_titles_btn.setCursor(Qt.PointingHandCursor)
        self._raw_titles_btn.setCheckable(True)
        self._raw_titles_btn.setChecked(self._show_raw_titles)
        self._raw_titles_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        self._raw_titles_btn.toggled.connect(self._on_raw_titles_toggled)
        toolbar.addWidget(self._raw_titles_btn)
        toolbar.addStretch()
        # "Open Library Folder" lives on the shared toolbar above the
        # tabs so it's reachable from either tab without having to
        # swap over to Completed first.
        self._open_library_btn = QPushButton(
            "\U0001f4c1  Open Library Folder")
        self._open_library_btn.setCursor(Qt.PointingHandCursor)
        self._open_library_btn.setToolTip(
            f"Open {get_library_dir()} in the system file explorer.")
        self._open_library_btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }"
        )
        self._open_library_btn.clicked.connect(self._open_library_folder)
        toolbar.addWidget(self._open_library_btn)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        toolbar.addWidget(self._count_label)
        root.addLayout(toolbar)

        # Per-tab action bar: each tab gets its own extra button row with
        # context-specific actions. Organize / Undo are duplicated across
        # both tabs so the user doesn't have to hop over to In Progress to
        # organize a Library entry. Every button label carries a live
        # counter that reflects how many files the action would act on
        # (see :meth:`_update_organize_counts`).
        self._tabs = QTabWidget()
        # Base stylesheet: only the In Progress / Completed *content*
        # tabs. The Scan-for-Raw action tab gets its teal-button
        # styling grafted on by :meth:`_apply_tab_stylesheet` only
        # when it's actually visible \u2014 otherwise ``QTabBar::tab:last``
        # would match whichever tab is currently the last visible one
        # (i.e. Completed when Scan-for-Raw is hidden) and the wrong
        # tab would render as a button.
        self._tabs_base_qss = """
            QTabWidget::pane { border: 1px solid #2a2a3e; border-radius: 6px;
                                background: #12121e; top: -1px; }
            QTabBar::tab { background: #1e1e2e; color: #b0b0c0;
                            border: 1px solid #2a2a3e; border-bottom: none;
                            border-top-left-radius: 6px; border-top-right-radius: 6px;
                            padding: 6px 16px; font-size: 9.5pt; font-weight: bold;
                            margin-right: 2px; min-width: 110px; }
            QTabBar::tab:selected { background: #12121e; color: #e0e0e0;
                                     border-color: #2a2a3e; }
            QTabBar::tab:hover:!selected { background: #252540; color: #e0e0e0; }
        """
        # Appended only while the scan tab is visible. When hidden,
        # ``:last`` would bleed onto Completed \u2014 so the rule must
        # be removed, not just ignored.
        self._tabs_scan_button_qss = """
            QTabBar::tab:last { background: #17a2b8; color: white;
                                 border: none; border-radius: 4px;
                                 padding: 6px 14px; font-size: 9pt;
                                 font-weight: bold; margin-left: 8px;
                                 margin-right: 2px; margin-top: 2px;
                                 margin-bottom: 2px; min-width: 0; }
            QTabBar::tab:last:selected { background: #17a2b8; color: white;
                                          border: none; }
            QTabBar::tab:last:hover { background: #20b2cc; color: white;
                                       border: none; }
        """
        self._tabs.setStyleSheet(self._tabs_base_qss)


        # --- In Progress tab ---
        self._ip_tab = QWidget()
        self._ip_tab.setStyleSheet("background: #12121e;")
        ip_layout = QVBoxLayout(self._ip_tab)
        ip_layout.setContentsMargins(6, 8, 6, 6)
        ip_layout.setSpacing(6)
        ip_action_row = QHBoxLayout()
        ip_action_row.setSpacing(6)
        self._ip_count_label = QLabel("")
        self._ip_count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        ip_action_row.addWidget(self._ip_count_label)
        ip_action_row.addStretch()
        self._ip_organize_btn = self._make_organize_button(kind="ip")
        ip_action_row.addWidget(self._ip_organize_btn)
        self._ip_undo_btn = self._make_undo_button(kind="ip")
        ip_action_row.addWidget(self._ip_undo_btn)
        self._import_btn = QPushButton("\u2795  Import EPUB")
        self._import_btn.setCursor(Qt.PointingHandCursor)
        self._import_btn.setToolTip(
            "Register an EPUB with the Library and scaffold a new output "
            "folder so you can translate it later. The source file stays "
            "exactly where it is on disk \u2014 click Organize when you're "
            "ready to move it into Library/Raw."
        )
        self._import_btn.setStyleSheet(
            "QPushButton { background: #6c63ff; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #8078ff; }")
        self._import_btn.clicked.connect(self._import_epub)
        ip_action_row.addWidget(self._import_btn)
        ip_layout.addLayout(ip_action_row)
        self._ip_scroll = QScrollArea()
        self._ip_scroll.setWidgetResizable(True)
        self._ip_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 8px; background: #1a1a2e; }"
            "QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 4px; }")
        # Use :class:`_SelectableGrid` so the user can click-drag a
        # rubber-band rectangle across multiple cards to select them.
        self._ip_grid_container = _SelectableGrid()
        self._ip_grid_container.rubber_band_selection.connect(
            self._on_rubber_band_selection)
        self._ip_grid_container.empty_clicked.connect(self._on_empty_area_clicked)
        self._ip_grid_layout = QGridLayout(self._ip_grid_container)
        self._ip_grid_layout.setContentsMargins(1, 1, 1, 1)
        self._ip_grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._ip_scroll.setWidget(self._ip_grid_container)
        ip_layout.addWidget(self._ip_scroll, 1)
        self._ip_pager = self._make_library_pager("ip")
        ip_layout.addWidget(self._ip_pager)
        # QScrollArea eats wheel events for its own scrolling, so the
        # dialog-level :meth:`wheelEvent` never fires when the cursor is
        # over the grid. Installing this event filter on the scroll
        # area + its viewport lets Ctrl+Wheel zoom the card size instead
        # of scrolling the grid (plain wheel still scrolls normally).
        self._ip_scroll.installEventFilter(self)
        self._ip_scroll.viewport().installEventFilter(self)
        self._ip_grid_container.installEventFilter(self)
        self._ip_empty_label = QLabel(
            "No translations in progress.\nUse \u201cImport EPUB\u201d to start one.")
        self._ip_empty_label.setAlignment(Qt.AlignCenter)
        self._ip_empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._ip_empty_label.hide()
        ip_layout.addWidget(self._ip_empty_label)

        # --- Completed (Library) tab ---
        self._comp_tab = QWidget()
        self._comp_tab.setStyleSheet("background: #12121e;")
        comp_layout = QVBoxLayout(self._comp_tab)
        comp_layout.setContentsMargins(6, 8, 6, 6)
        comp_layout.setSpacing(6)
        comp_action_row = QHBoxLayout()
        comp_action_row.setSpacing(6)
        self._comp_count_label = QLabel("")
        self._comp_count_label.setStyleSheet("color: #888; font-size: 8.5pt;")
        comp_action_row.addWidget(self._comp_count_label)
        comp_action_row.addStretch()
        # Completed tab gets its own copy of the Organize + Undo buttons so
        # the user can move a compiled EPUB into the Library without
        # having to bounce back to In Progress. Wired to the same
        # handlers — the actions operate on both tabs' books regardless
        # of which button was clicked; only the *label counters* are
        # tab-specific.
        self._comp_organize_btn = self._make_organize_button(kind="comp")
        comp_action_row.addWidget(self._comp_organize_btn)
        self._comp_undo_btn = self._make_undo_button(kind="comp")
        comp_action_row.addWidget(self._comp_undo_btn)
        # "Add Translation" mirrors the In Progress tab's Import EPUB
        # slot — it sits next to Organize / Undo so registering a
        # compiled EPUB is reachable directly from the Completed tab.
        # Uses the same import pipeline as the Completed tab's
        # drag-drop (``target="translated"``): files are registered in
        # place via ``library_translated_inputs.txt`` and surface as
        # ``registered_translated=True`` cards. Nothing is moved until
        # Organize fires.
        self._add_translation_btn = QPushButton(
            "\U0001f4d5  Add Translation")
        self._add_translation_btn.setCursor(Qt.PointingHandCursor)
        self._add_translation_btn.setToolTip(
            "Pick one or more compiled .epub files to register with "
            "the Library's Completed tab. Files stay where they are "
            "on disk \u2014 same behaviour as dropping them onto "
            "this tab. Click Organize later to move them into "
            "Library/Translated."
        )
        self._add_translation_btn.setStyleSheet(
            "QPushButton { background: #6c63ff; color: white; "
            "border-radius: 4px; padding: 6px 14px; font-size: 9pt; "
            "font-weight: bold; border: none; }"
            "QPushButton:hover { background: #8078ff; }")
        self._add_translation_btn.clicked.connect(self._add_translation)
        comp_action_row.addWidget(self._add_translation_btn)
        comp_layout.addLayout(comp_action_row)
        self._comp_scroll = QScrollArea()
        self._comp_scroll.setWidgetResizable(True)
        self._comp_scroll.setStyleSheet(self._ip_scroll.styleSheet())
        self._comp_grid_container = _SelectableGrid()
        self._comp_grid_container.rubber_band_selection.connect(
            self._on_rubber_band_selection)
        self._comp_grid_container.empty_clicked.connect(self._on_empty_area_clicked)
        self._comp_grid_layout = QGridLayout(self._comp_grid_container)
        self._comp_grid_layout.setContentsMargins(1, 1, 1, 1)
        self._comp_grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._comp_scroll.setWidget(self._comp_grid_container)
        comp_layout.addWidget(self._comp_scroll, 1)
        self._comp_pager = self._make_library_pager("comp")
        comp_layout.addWidget(self._comp_pager)
        # Same Ctrl+Wheel interception as the In Progress scroll area.
        self._comp_scroll.installEventFilter(self)
        self._comp_scroll.viewport().installEventFilter(self)
        self._comp_grid_container.installEventFilter(self)
        self._comp_empty_label = QLabel(
            "Your Library is empty.\n\n"
            "Drop finished .epub files here \u2014 or click\n"
            "\u201cAdd Translation\u201d above \u2014 to see them here.")
        self._comp_empty_label.setAlignment(Qt.AlignCenter)
        self._comp_empty_label.setStyleSheet("color: #555; font-size: 12pt; padding: 40px;")
        self._comp_empty_label.hide()
        comp_layout.addWidget(self._comp_empty_label)

        self._install_library_drop_targets()

        self._tabs.addTab(self._ip_tab, "\u23f3  In Progress")
        self._tabs.addTab(self._comp_tab, "\u2705  Completed")
        # "Scan for Raw" needs to read as a real third tab after
        # Completed, not a corner widget. This placeholder page is
        # never meant to stay selected: :meth:`_on_tab_changed`
        # immediately snaps back to the previous content tab and
        # opens the dialog instead.
        self._scan_tab = QWidget()
        self._scan_tab.setStyleSheet("background: #12121e;")
        self._install_library_drop_target(self._scan_tab)
        self._scan_tab_index = self._tabs.addTab(
            self._scan_tab, "\U0001f50d  Scan for Raw")
        try:
            self._tabs.setTabVisible(self._scan_tab_index, False)
        except AttributeError:
            self._tabs.setTabEnabled(self._scan_tab_index, False)
        try:
            initial_idx = int(self._current_tab) or 0
        except (TypeError, ValueError):
            initial_idx = 0
        if initial_idx == self._scan_tab_index or initial_idx < 0:
            initial_idx = 0
        self._tabs.setCurrentIndex(initial_idx)
        self._prev_content_tab = initial_idx
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)

        # ── Loading overlay (shared spinner shown over the current tab) ──
        self._loading_widget = QWidget()
        loading_layout = QVBoxLayout(self._loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        loading_layout.setContentsMargins(0, 20, 0, 20)
        loading_layout.setSpacing(8)
        self._spin_label = QLabel()
        icon_path = _find_halgakos_icon()
        self._spin_pixmap = None
        # Pre-rendered rotation frames. Rotating the pixmap every tick with
        # ``transformed()`` re-rasterises AND reallocates a new QPixmap on the
        # UI thread each frame, which stutters and competes with the
        # card-building work. Instead we render every frame ONCE up front with
        # high-quality (Smooth) sampling and just swap the cached pixmap per
        # tick — O(1), allocation-free, and no aliasing "crawl".
        self._spin_frames: list = []
        self._spin_frame_idx = 0
        self._spin_period_s = 1.2  # seconds per full revolution
        self._spin_t0 = 0.0        # perf_counter() baseline, set on show
        if icon_path:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                self._spin_pixmap = pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._spin_label.setPixmap(self._spin_pixmap)
                self._spin_frames = [
                    self._spin_pixmap.transformed(
                        QTransform().rotate(a), Qt.SmoothTransformation)
                    for a in range(0, 360, 6)  # 60 frames → smooth 6° steps
                ]
        if not self._spin_pixmap:
            self._spin_label.setText("\U0001f4da")
            self._spin_label.setStyleSheet("font-size: 32pt; color: #e0e0e0;")
        self._spin_label.setAlignment(Qt.AlignCenter)
        # A 64px square rotated to 45° spans 64·√2 ≈ 91px. Size the label to
        # fit the largest rotated frame so corners are never clipped — a
        # clipped/re-cropped pixmap is what makes the icon appear to pulse and
        # wobble as it turns.
        self._spin_label.setFixedSize(96, 96)
        loading_layout.addWidget(self._spin_label, 0, Qt.AlignCenter)
        self._spin_angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(16)  # ~60 fps
        # Wrap in a lambda so PySide6 routes the call directly instead
        # of going through Qt's meta-object slot-lookup — certain
        # PySide6 builds raise ``AttributeError: Slot 'EpubLibraryDialog::
        # _rotate_spinner()' not found`` on ``QTimer.timeout`` dispatch
        # even when the method exists and is ``@Slot()``-decorated.
        self._spin_timer.timeout.connect(lambda: self._rotate_spinner())
        self._loading_text = QLabel("Scanning library\u2026")
        self._loading_text.setAlignment(Qt.AlignCenter)
        self._loading_text.setStyleSheet("color: #888; font-size: 11pt; padding-top: 4px;")
        loading_layout.addWidget(self._loading_text, 0, Qt.AlignCenter)
        from PySide6.QtWidgets import QProgressBar
        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)  # indeterminate
        self._loading_bar.setFixedWidth(220)
        self._loading_bar.setFixedHeight(6)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setStyleSheet("""
            QProgressBar { background: #2a2a2a; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #6c63ff; border-radius: 3px; }
        """)
        loading_layout.addWidget(self._loading_bar, 0, Qt.AlignCenter)
        self._loading_widget.setStyleSheet("background: transparent;")
        self._loading_widget.hide()
        root.addWidget(self._loading_widget, 1)

        self.setStyleSheet("QDialog { background: #12121e; }")

        # Drag-over overlay: semi-transparent purple panel with a centered
        # "Drop to import" message. Shown in :meth:`dragEnterEvent` and
        # hidden in :meth:`dragLeaveEvent` / :meth:`dropEvent`. It's a
        # plain child widget positioned manually so it covers the whole
        # dialog without participating in the root layout.
        self._drop_overlay = QLabel(self)
        self._drop_overlay.setObjectName("drop-overlay")
        self._drop_overlay.setAlignment(Qt.AlignCenter)
        self._drop_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._drop_overlay.setText(
            "\U0001f4e5\n\nDrop files to register them with the Library\n"
            "(files stay where they are \u2014 click Organize to move)\n\n"
            "EPUB \u00b7 PDF \u00b7 TXT \u00b7 HTML"
        )
        self._drop_overlay.setStyleSheet(
            "QLabel#drop-overlay {"
            " background: rgba(108, 99, 255, 0.22);"
            " color: #e8e4ff;"
            " border: 3px dashed #8078ff;"
            " border-radius: 12px;"
            " font-size: 14pt;"
            " font-weight: bold;"
            "}"
        )
        self._drop_overlay.hide()

        # Toast widget: animated non-modal status line that appears at the
        # bottom of the dialog. Used instead of QMessageBox for drag-drop
        # imports so the flow stays click-free.
        self._toast = QLabel(self)
        self._toast.setObjectName("toast")
        self._toast.setAlignment(Qt.AlignCenter)
        self._toast.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._toast.setStyleSheet(
            "QLabel#toast {"
            " background: rgba(42, 45, 90, 0.96);"
            " color: #e8e4ff;"
            " border: 1px solid #8078ff;"
            " border-radius: 10px;"
            " padding: 10px 18px;"
            " font-size: 10pt;"
            " font-weight: bold;"
            "}"
        )
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._toast_opacity = QGraphicsOpacityEffect(self._toast)
        self._toast_opacity.setOpacity(0.0)
        self._toast.setGraphicsEffect(self._toast_opacity)
        self._toast.hide()
        self._toast_hide_timer = QTimer(self)
        self._toast_hide_timer.setSingleShot(True)
        self._toast_hide_timer.timeout.connect(self._fade_out_toast)
        self._toast_anim = None  # current QPropertyAnimation (if any)

    # -- Tab / scan / render helpers ----------------------------------------

    @Slot()
    def _rotate_spinner(self):
        """Timer-driven spinner tick — delegates to the time-based updater.

        Decorated with ``@Slot()`` so PySide6 registers it in the Qt
        meta-object system. Without the decorator, ``QTimer.timeout`` can fail
        to resolve the slot by name at invocation time (``AttributeError: Slot
        'EpubLibraryDialog::_rotate_spinner()' not found``).
        """
        self._tick_spinner()

    def _tick_spinner(self) -> None:
        """Set the spinner to the frame matching *wall-clock* elapsed time.

        The angle is derived from how long the overlay has been visible rather
        than incremented per call. This is what kills the "spin, freeze, spin"
        stutter: while the UI thread is busy building cards the 16 ms timer
        can't fire on schedule, but every time we DO get a slice of CPU (via
        :meth:`_pump_loading_events`) the icon jumps straight to the angle it
        should be at for *now* — so it reads as continuous motion instead of
        sticking at a stale angle until the next batch boundary.
        """
        frames = self._spin_frames
        if not frames:
            return
        n = len(frames)
        elapsed = time.perf_counter() - self._spin_t0
        idx = int(elapsed / self._spin_period_s * n) % n
        if idx != self._spin_frame_idx:
            self._spin_frame_idx = idx
            self._spin_label.setPixmap(frames[idx])

    def _show_loading(self):
        """Show the spinner overlay and hide both tab grids."""
        self._set_loading_text("Scanning library\u2026")
        self._spin_frame_idx = 0
        self._spin_t0 = time.perf_counter()
        self._spin_timer.start()
        self._tabs.hide()
        self._ip_empty_label.hide()
        self._comp_empty_label.hide()
        self._loading_widget.show()

    def _set_loading_text(self, text: str) -> None:
        label = getattr(self, "_loading_text", None)
        if label is not None:
            label.setText(text)

    def _hide_loading(self):
        """Hide the spinner overlay and restore the tab widget."""
        self._spin_timer.stop()
        self._loading_widget.hide()
        self._tabs.show()

    def _pump_loading_events(self) -> None:
        """Advance the spinner and let it repaint while cards are being built.

        We explicitly tick the spinner here rather than relying on the 16 ms
        QTimer — Qt coalesces it to at most one pending timeout per
        ``processEvents`` pass, so on a busy UI thread the timer alone would
        only nudge the icon one step per batch. Ticking by wall-clock time
        keeps it tracking real motion.
        """
        loading = getattr(self, "_loading_widget", None)
        if loading is None or not loading.isVisible():
            return
        self._tick_spinner()
        app = QApplication.instance()
        if app is not None:
            app.processEvents(QEventLoop.ExcludeUserInputEvents)

    def _tab_key_for_index(self, index: int | None = None) -> str:
        if index is None:
            index = int(getattr(self, "_current_tab", 0) or 0)
        return "comp" if int(index) == 1 else "ip"

    def _active_card_tab_key(self) -> str:
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return self._tab_key_for_index(getattr(self, "_current_tab", 0))
        try:
            index = int(tabs.currentIndex())
        except Exception:
            index = int(getattr(self, "_current_tab", 0) or 0)
        scan_tab_index = int(getattr(self, "_scan_tab_index", -1) or -1)
        if index == scan_tab_index:
            index = int(getattr(self, "_prev_content_tab", 0) or 0)
        return self._tab_key_for_index(index)

    @staticmethod
    def _other_card_tab_key(tab_key: str) -> str:
        return "comp" if tab_key == "ip" else "ip"

    def _cards_for_tab_key(self, tab_key: str) -> list[_BookCard]:
        return self._comp_cards if tab_key == "comp" else self._ip_cards

    def _effective_grid_width(
        self,
        scroll_area: QScrollArea | None,
        grid_widget: QWidget | None = None,
    ) -> int:
        """Return a stable card-grid width, including for hidden tabs.

        A hidden ``QScrollArea`` commonly keeps its pre-layout/default width
        (about 640 px on Windows). The same can happen for a newly revealed
        viewport for one event-loop turn. Both values are valid positive
        integers, so falling back only when the width is zero or tiny still
        lets the grid lock in too few columns. The Library tabs occupy the
        dialog's full content width, therefore the visible sibling scroll area
        and a conservative dialog-width floor are safe geometry sources while
        the target viewport is settling.
        """
        widths: list[int] = []

        def _collect(area: QScrollArea | None) -> None:
            if area is None:
                return
            try:
                widths.append(int(area.viewport().width()))
            except Exception:
                pass
            try:
                # ``contentsRect`` stays constant when the vertical scrollbar
                # appears; the viewport itself becomes narrower by exactly the
                # reserved scrollbar width. Using both values lets card
                # streaming toggle the scrollbar without changing the layout
                # width and recursively triggering another full rebuild.
                widths.append(int(area.contentsRect().width()))
            except Exception:
                pass

        # Inactive tabs are deliberately pre-populated for fast switching, but
        # Qt does not lay their scroll areas out to the final width until they
        # become visible. Borrow the already-settled sibling width instead.
        try:
            target_visible = bool(scroll_area and scroll_area.isVisible())
        except Exception:
            target_visible = False
        if target_visible:
            _collect(scroll_area)
        else:
            for sibling in (
                getattr(self, "_ip_scroll", None),
                getattr(self, "_comp_scroll", None),
            ):
                if sibling is None or sibling is scroll_area:
                    continue
                try:
                    if sibling.isVisible():
                        _collect(sibling)
                        break
                except Exception:
                    continue

        try:
            widths.append(max(0, int(self.width()) - 40))
        except Exception:
            pass
        if not any(width > 0 for width in widths):
            _collect(scroll_area)
        if not any(width > 0 for width in widths) and grid_widget is not None:
            try:
                widths.append(int(grid_widget.width()))
            except Exception:
                pass

        measured = max((width for width in widths if width > 0), default=0)
        return max(0, measured - self._GRID_SCROLLBAR_RESERVE)

    def _sync_card_size_from_combo(self) -> None:
        combo = getattr(self, "_size_combo", None)
        if combo is not None:
            try:
                size_key = combo.itemData(combo.currentIndex())
                if size_key in _SIZE_PRESETS:
                    self._card_size = size_key
                    return
            except Exception:
                pass
        if self._card_size not in _SIZE_PRESETS:
            self._card_size = SIZE_COMPACT

    def _start_visible_card_refresh(self) -> None:
        self._sync_card_size_from_combo()
        for widget in (
            getattr(self, "_tabs", None),
            getattr(self, "_ip_scroll", None),
            getattr(self, "_comp_scroll", None),
            getattr(self, "_ip_grid_container", None),
            getattr(self, "_comp_grid_container", None),
        ):
            if widget is None:
                continue
            try:
                widget.updateGeometry()
                layout = widget.layout()
                if layout is not None:
                    layout.activate()
            except Exception:
                pass
        self._refresh_view()

    def _books_for_tab_key(self, tab_key: str) -> list[dict]:
        return self._completed_books if tab_key == "comp" else self._in_progress_books

    def _library_page_size(self, tab_key: str) -> int:
        pager = self._library_pagers.get(tab_key, {})
        combo = pager.get("page_size")
        value = combo.currentData() if combo is not None else 20
        if value == "all":
            return 0
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 20

    def _library_page_bounds(
        self, tab_key: str, total: int
    ) -> tuple[int, int, int]:
        start, end, page_count, page = _page_bounds(
            total,
            self._library_pages.get(tab_key, 0),
            self._library_page_size(tab_key),
        )
        self._library_pages[tab_key] = page
        return start, end, page_count

    def _update_library_pagination_controls(
        self, tab_key: str, filtered_count: int
    ) -> tuple[int, int]:
        pager = self._library_pagers.get(tab_key, {})
        start, end, page_count = self._library_page_bounds(
            tab_key, filtered_count)
        page = int(self._library_pages.get(tab_key, 0) or 0)
        label = pager.get("label")
        if label is not None:
            label.setText(_page_label(
                page, page_count, start, end, filtered_count,
                self._library_page_size(tab_key)))
        can_go_back = filtered_count > 0 and page > 0
        can_go_forward = filtered_count > 0 and page < page_count - 1
        for name in ("first", "previous"):
            button = pager.get(name)
            if button is not None:
                button.setEnabled(can_go_back)
        for name in ("next", "last"):
            button = pager.get(name)
            if button is not None:
                button.setEnabled(can_go_forward)
        widget = pager.get("widget")
        if widget is not None:
            widget.setVisible(filtered_count > 0)
        return start, end

    def _reset_library_scroll_position(self, tab_key: str) -> None:
        animation = self._library_scroll_animations.get(tab_key)
        if animation is not None:
            animation.stop()
        area = (
            getattr(self, "_comp_scroll", None)
            if tab_key == "comp" else getattr(self, "_ip_scroll", None)
        )
        if area is not None:
            area.verticalScrollBar().setValue(0)
        self._library_scroll_targets[tab_key] = 0

    def _reset_library_pages(self) -> None:
        self._library_pages["ip"] = 0
        self._library_pages["comp"] = 0

    def _on_library_page_action(self, tab_key: str, action: str) -> None:
        filtered_count = len(self._filtered(self._books_for_tab_key(tab_key)))
        _, _, page_count = self._library_page_bounds(tab_key, filtered_count)
        current = int(self._library_pages.get(tab_key, 0) or 0)
        if action == "first":
            target = 0
        elif action == "previous":
            target = max(0, current - 1)
        elif action == "next":
            target = min(page_count - 1, current + 1)
        else:
            target = max(0, page_count - 1)
        if target == current:
            return
        self._library_pages[tab_key] = target
        self._reset_library_scroll_position(tab_key)
        self._populate_tab(tab_key)

    def _on_library_page_size_changed(self, tab_key: str) -> None:
        pager = self._library_pagers.get(tab_key, {})
        combo = pager.get("page_size")
        if combo is None:
            return
        value = combo.currentData()
        self._config["epub_library_page_size"] = value
        # Both tabs intentionally share one Library page-size preference.
        for other_key, other_pager in self._library_pagers.items():
            other_combo = other_pager.get("page_size")
            if other_key == tab_key or other_combo is None:
                continue
            index = other_combo.findData(value)
            if index >= 0 and other_combo.currentIndex() != index:
                other_combo.blockSignals(True)
                other_combo.setCurrentIndex(index)
                other_combo.blockSignals(False)
        self._reset_library_pages()
        self._reset_library_scroll_position("ip")
        self._reset_library_scroll_position("comp")
        self._refresh_view()

    def _populate_tab(self, tab_key: str) -> None:
        filtered = self._filtered(self._books_for_tab_key(tab_key))
        start, end = self._update_library_pagination_controls(
            tab_key, len(filtered))
        page_books = filtered[start:end]
        if tab_key == "comp":
            self._populate_completed(
                page_books, filtered_count=len(filtered))
        else:
            self._populate_in_progress(
                page_books, filtered_count=len(filtered))
            tab_key = "ip"
        self._dirty_card_tabs.discard(tab_key)

    def _is_card_stream_active(self, tab_key: str | None = None) -> bool:
        states = getattr(self, "_card_stream_states", {}) or {}
        if tab_key:
            return tab_key in states
        return bool(states)

    def _schedule_grid_reflow(self, tab_key: str | None = None):
        """Debounce grid reflow without restarting active card streams."""
        key = tab_key or self._active_card_tab_key()
        self._pending_card_reflow.add(key)
        if self._is_card_stream_active(key):
            return
        self._grid_reflow_timer.start(90)

    def _schedule_grid_reflow_if_width_changed(
        self, tab_key: str | None = None
    ) -> None:
        """Queue a reflow only when the tab's usable width really changed."""
        key = tab_key or self._active_card_tab_key()
        if not (
            self._cards_for_tab_key(key)
            or self._is_card_stream_active(key)
        ):
            return
        scroll_area = (
            getattr(self, "_comp_scroll", None)
            if key == "comp" else getattr(self, "_ip_scroll", None)
        )
        grid_widget = (
            getattr(self, "_comp_grid_container", None)
            if key == "comp" else getattr(self, "_ip_grid_container", None)
        )
        current_width = self._effective_grid_width(scroll_area, grid_widget)
        if (
            current_width > 0
            and current_width != self._grid_layout_widths.get(key)
        ):
            self._schedule_grid_reflow(key)

    def _run_pending_grid_reflow(self) -> None:
        pending = list(getattr(self, "_pending_card_reflow", set()) or set())
        if not pending:
            return
        for key in pending:
            if self._is_card_stream_active():
                break
            self._pending_card_reflow.discard(key)
            self._populate_tab(key)
            break
        if self._pending_card_reflow:
            self._grid_reflow_timer.start(90)
        else:
            self._schedule_inactive_tab_population()

    def _schedule_inactive_tab_population(self) -> None:
        if not self.isVisible():
            return
        if self._is_card_stream_active():
            return
        if self._pending_card_reflow:
            self._inactive_tab_populate_timer.start(self._INACTIVE_TAB_IDLE_MS)
            return
        if (self._cover_queue or self._cover_active_loaders
                or self._cover_apply_queue):
            self._inactive_tab_populate_timer.start(self._INACTIVE_TAB_IDLE_MS)
            return
        inactive = self._other_card_tab_key(self._active_card_tab_key())
        if inactive in self._dirty_card_tabs:
            self._inactive_tab_populate_timer.start(self._INACTIVE_TAB_IDLE_MS)

    def _populate_inactive_tab_if_idle(self) -> None:
        if not self.isVisible():
            return
        if self._is_card_stream_active():
            self._schedule_inactive_tab_population()
            return
        if self._pending_card_reflow:
            self._schedule_inactive_tab_population()
            return
        if (self._cover_queue or self._cover_active_loaders
                or self._cover_apply_queue):
            self._schedule_inactive_tab_population()
            return
        inactive = self._other_card_tab_key(self._active_card_tab_key())
        if inactive in self._dirty_card_tabs:
            self._populate_tab(inactive)

    def _set_sort(self, mode):
        self._sort_mode = mode
        for k, btn in self._sort_btns.items():
            btn.setChecked(k == mode)
        self._reset_library_pages()
        self._reset_library_scroll_position("ip")
        self._reset_library_scroll_position("comp")
        self._refresh_view()

    def _set_format_filter(self, fmt_key: str):
        """Change the active file-format filter and refresh both tabs.

        Driven by the ``Format`` dropdown (:class:`_NoWheelComboBox`).
        The combo's current selection is the source of truth \u2014 we
        sync it defensively in case the change came from a keyboard
        shortcut or an external call rather than the combo itself.
        """
        if not fmt_key or fmt_key == self._format_filter:
            return
        self._format_filter = fmt_key
        self._reset_library_pages()
        self._reset_library_scroll_position("ip")
        self._reset_library_scroll_position("comp")
        try:
            self._config["epub_library_format_filter"] = fmt_key
        except Exception:
            pass
        combo = getattr(self, "_format_combo", None)
        if combo is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == fmt_key:
                    if combo.currentIndex() != i:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(i)
                        combo.blockSignals(False)
                    break
        self._refresh_view()

    # _format_of_book moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _set_card_size(self, size_key):
        """Change the active card size and refresh both tabs.

        Driven by the ``View`` dropdown (:class:`_NoWheelComboBox`) and
        also by Ctrl+Wheel / Ctrl+= / Ctrl+- shortcuts. The combo is
        synced defensively so every entry point keeps the dropdown
        in lockstep with :attr:`_card_size`.
        """
        if not size_key:
            return
        self._card_size = size_key
        combo = getattr(self, "_size_combo", None)
        if combo is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == size_key:
                    if combo.currentIndex() != i:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(i)
                        combo.blockSignals(False)
                    break
        self._refresh_view()

    def _on_raw_titles_toggled(self, checked: bool):
        """Flip every card between raw and translated title rendering."""
        new_value = bool(checked)
        if new_value == self._show_raw_titles:
            return
        self._show_raw_titles = new_value
        try:
            self._config["epub_library_show_raw_titles"] = self._show_raw_titles
        except Exception:
            pass
        # Cheapest reliable path: rebuild the cards so each :class:`_BookCard`
        # picks the title through its constructor. Cards are lightweight and
        # we already do this on sort / size changes.
        self._refresh_view()

    # _sorted_books moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _on_tab_changed(self, index: int):
        scan_tab_index = getattr(self, "_scan_tab_index", -1)
        if index == scan_tab_index:
            fallback = getattr(self, "_prev_content_tab", 0)
            if fallback == scan_tab_index or fallback < 0:
                fallback = 0
            # Defer both actions so the clicked tab visibly behaves as
            # a tab item, but the dialog still opens without leaving
            # the QTabWidget parked on an empty placeholder page.
            QTimer.singleShot(
                0, lambda fb=fallback: self._tabs.setCurrentIndex(fb))
            QTimer.singleShot(0, self._open_scan_for_raw)
            return
        self._prev_content_tab = index
        self._current_tab = index
        self._config["epub_library_tab"] = index
        tab_key = self._tab_key_for_index(index)
        if tab_key in getattr(self, "_dirty_card_tabs", set()):
            self._populate_tab(tab_key)
        elif self._cards_for_tab_key(tab_key):
            self._request_card_hover_reconcile(1500)
            QTimer.singleShot(0, self._queue_visible_missing_covers)

    # -- Actions ------------------------------------------------------------

    def _open_library_folder(self):
        """Open the Glossarion Library folder in the system file explorer."""
        lib = get_library_dir()
        try:
            os.makedirs(lib, exist_ok=True)
        except OSError:
            pass
        _open_folder_in_explorer(lib)

    def _open_scan_for_raw(self):
        """Open the Scan-for-Raw dialog.

        Feeds the dialog every workspace-backed card with the
        ``missing_raw_file`` warning (In Progress AND Completed tab)
        since a completed workspace whose compiled output still
        points at a lost raw benefits from the pairing too. When
        the user applies pairings the dialog emits ``applied`` with
        the count; we trigger a library reload so the scanner picks
        up the freshly written ``source_epub.txt`` files.
        """
        candidates = [
            b for b in (
                list(self._in_progress_books)
                + list(self._completed_books))
            if b.get("output_folder")
        ]
        dlg = _ScanForRawDialog(
            candidates,
            config=self._config,
            parent=self,
        )
        dlg.applied.connect(lambda _n: QTimer.singleShot(
            0, self._load_books))
        dlg.exec()

    def _add_translation(self):
        """Pick compiled EPUB(s) and register them with the Completed tab.

        Mirrors the Completed tab's drag-drop flow — files stay on disk
        where they are; the Library merely tracks them via
        ``library_translated_inputs.txt`` and surfaces a
        ``registered_translated=True`` card on the Completed tab. The
        user can later promote the file(s) into ``Library/Translated``
        via the Organize button, which is also what the drag-drop path
        expects.

        Only ``.epub`` is accepted here (same contract as the drag-drop
        target) since non-EPUB compiled outputs aren't shelf artefacts.
        """
        from PySide6.QtWidgets import QFileDialog
        start_dir = str(Path.home() / "Downloads")
        if not os.path.isdir(start_dir):
            start_dir = str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add translated EPUB(s) to Library",
            start_dir,
            "EPUB files (*.epub);;All files (*.*)",
        )
        if not paths:
            return
        self._import_paths_into_library(
            paths, source="picker", target="translated")

    # -- Organize / Undo button factory + counters --------------------------

    def _make_organize_button(self, kind: str) -> QPushButton:
        """Create an "Organize N into Library" button for the given tab kind.

        *kind* is either ``"ip"`` (In Progress — counts raw sources that
        can move into ``Library/Raw``) or ``"comp"`` (Completed — counts
        compiled EPUBs that can move into ``Library/Translated``). The
        label is refreshed by :meth:`_update_organize_counts` after every
        scan.
        """
        btn = QPushButton("\U0001f4e5  Organize (0)")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: #3a5a7a; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #4a6a8a; }"
            "QPushButton:disabled { background: #2a2a3e; color: #555; }"
        )
        btn.setToolTip(
            "Move every resolvable raw source into Library/Raw and every "
            "compiled EPUB into Library/Translated. Each move is recorded "
            "so it can be reversed by Undo Move."
        )
        btn.clicked.connect(self._organize_into_library)
        btn.setProperty("_organize_kind", kind)
        return btn

    def _make_undo_button(self, kind: str) -> QPushButton:
        """Create an "Undo N" button for the given tab kind.

        IP tab shows the count of raw moves that can be undone; Completed
        tab shows the translated bucket's count. Clicking either opens
        the same 3-way prompt (Raw / Translated / All).
        """
        btn = QPushButton("\u21a9  Undo (0)")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: #6f42c1; color: white; border-radius: 4px; "
            "padding: 6px 14px; font-size: 9pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #5a32a3; }"
            "QPushButton:disabled { background: #2a2a3e; color: #555; }"
        )
        btn.setToolTip(
            "Restore previously organized files back to their original "
            "locations. You'll be asked whether to undo Raw, Translated, "
            "or All."
        )
        btn.clicked.connect(self._undo_organize_prompt)
        btn.setProperty("_undo_kind", kind)
        return btn

    # _count_raw_movable, _count_trans_movable, _build_workspace_title_index moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _apply_tab_stylesheet(self, scan_visible: bool) -> None:
        """Attach / detach the teal-button last-tab rule on the tab bar.

        ``QTabBar::tab:last`` matches whichever tab is currently the
        last visible one, so the styling only makes sense while the
        Scan-for-Raw tab is actually present \u2014 otherwise it would
        bleed onto the Completed tab.
        """
        qss = self._tabs_base_qss
        if scan_visible:
            qss = qss + self._tabs_scan_button_qss
        try:
            if self._tabs.styleSheet() != qss:
                self._tabs.setStyleSheet(qss)
        except Exception:
            pass

    def _update_organize_counts(self):
        """Refresh the Organize + Undo button labels with live counts.

        Called after every scan (initial + auto-refresh). Disables each
        button when its count is zero so the user can see at a glance
        that there's nothing to do.
        """
        counts = self._organize_counts()
        raw_count = counts["raw_count"]
        trans_count = counts["trans_count"]
        raw_undo = counts["raw_undo"]
        trans_undo = counts["trans_undo"]
        try:
            self._ip_organize_btn.setText(
                f"\U0001f4e5  Organize ({raw_count})")
            self._ip_organize_btn.setEnabled(raw_count > 0)
            self._ip_undo_btn.setText(f"\u21a9  Undo ({raw_undo})")
            self._ip_undo_btn.setEnabled(raw_undo > 0)
            self._comp_organize_btn.setText(
                f"\U0001f4e5  Organize ({trans_count})")
            self._comp_organize_btn.setEnabled(trans_count > 0)
            self._comp_undo_btn.setText(f"\u21a9  Undo ({trans_undo})")
            self._comp_undo_btn.setEnabled(trans_undo > 0)
        except Exception:
            pass
        # Toggle the Scan-for-Raw TAB based on whether any visible
        # card has the ``missing_raw_file`` warning. Hidden entirely
        # otherwise \u2014 there's nothing actionable for the dialog to
        # do when every workspace's raw is already resolved. The
        # teal-button tab styling is also attached / detached here
        # so the Completed tab never inherits it when the scan tab is
        # hidden (``QTabBar::tab:last`` targets whichever tab is
        # currently the last visible one).
        try:
            missing_raw_count = self._missing_raw_count()
            self._apply_tab_stylesheet(missing_raw_count > 0)
            scan_tab_index = getattr(self, "_scan_tab_index", -1)
            if scan_tab_index >= 0:
                self._tabs.setTabText(
                    scan_tab_index,
                    (f"\U0001f50d  Scan for Raw ({missing_raw_count})"
                     if missing_raw_count > 0
                     else "\U0001f50d  Scan for Raw"))
                if missing_raw_count <= 0:
                    fallback = getattr(self, "_prev_content_tab", 0)
                    if (self._tabs.currentIndex() == scan_tab_index
                            and fallback != scan_tab_index):
                        self._tabs.setCurrentIndex(max(0, fallback))
                try:
                    self._tabs.setTabVisible(
                        scan_tab_index, missing_raw_count > 0)
                except AttributeError:
                    self._tabs.setTabEnabled(
                        scan_tab_index, missing_raw_count > 0)
        except Exception:
            pass

    # _count_library_files moved verbatim to library_core.LibraryShelfMixin (inherited).
    # _organize_counts, _missing_raw_count were extracted from _update_organize_counts into
    # library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _import_epub(self):
        """Pick raw source EPUB(s), copy them into Library/Raw, and scaffold
        output folders in the configured output root with a source_epub.txt
        pointer. This is how the In Progress tab picks each book up later.

        The import does NOT push the files into the translator's input field
        — that's what the context-menu "Load for translation" action is for.
        """
        from PySide6.QtWidgets import QFileDialog
        start_dir = str(Path.home() / "Downloads")
        if not os.path.isdir(start_dir):
            start_dir = str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import file(s) into Library", start_dir,
            "EPUB files (*.epub);;All supported (*.epub *.txt *.pdf *.html);;All files (*.*)",
        )
        if not paths:
            return
        self._import_paths_into_library(paths, source="picker", target="raw")

    def _prompt_duplicate_policy(self, collisions: list[tuple[str, str]],
                                 dest_label: str) -> str | None:
        """Ask the user how to handle one or more duplicate filenames.

        *collisions* is a list of ``(source_path, existing_dest_path)``
        tuples — one entry per file whose basename already exists in the
        destination folder. Previously the import code silently resolved
        collisions by appending a counter suffix (``name (2).epub``),
        which meant a raw EPUB dropped twice produced two separate
        copies and, on the translator side, two unrelated output
        folders. The user never got a chance to say "this is the same
        book, just replace it".

        Returns one of: ``"replace"``, ``"keep_both"``, ``"skip"``,
        or ``None`` when the user cancels (the whole import is then
        aborted). The returned policy is applied to *every* colliding
        file in the batch — per-file prompting would be much noisier
        for multi-drop imports and matches what Explorer / Finder do.
        """
        if not collisions:
            return "keep_both"
        n = len(collisions)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Duplicate files")
        preview_lines = []
        for src, existing in collisions[:6]:
            preview_lines.append(f"  \u2022 {os.path.basename(existing)}")
        if n > 6:
            preview_lines.append(f"  \u2026 and {n - 6} more.")
        if n == 1:
            msg.setText(
                f"A file named \u201c{os.path.basename(collisions[0][1])}\u201d "
                f"already exists in {dest_label}.\n\n"
                f"What would you like to do?"
            )
        else:
            msg.setText(
                f"{n} files being imported already exist in {dest_label}:\n\n"
                + "\n".join(preview_lines)
                + "\n\nWhat would you like to do with the duplicates?"
            )
        btn_replace = msg.addButton(
            "Replace" + (" All" if n > 1 else ""), QMessageBox.AcceptRole)
        btn_keep = msg.addButton(
            "Keep Both" + (" All" if n > 1 else ""), QMessageBox.AcceptRole)
        btn_skip = msg.addButton(
            "Skip" + (" All" if n > 1 else ""), QMessageBox.AcceptRole)
        btn_cancel = msg.addButton(QMessageBox.Cancel)
        btn_replace.setToolTip(
            "Overwrite the existing file(s) with the new one(s). "
            "Cannot be undone."
        )
        btn_keep.setToolTip(
            "Keep both copies. The new file(s) get a counter suffix "
            "like \u201cname (2).epub\u201d."
        )
        btn_skip.setToolTip("Don't import the duplicate(s).")
        msg.setDefaultButton(btn_keep)
        msg.exec()
        chosen = msg.clickedButton()
        if chosen is btn_replace:
            return "replace"
        if chosen is btn_keep:
            return "keep_both"
        if chosen is btn_skip:
            return "skip"
        return None  # cancel

    def _import_paths_into_library(self, paths, source: str = "picker",
                                   target: str = "raw"):
        """Core import pipeline used by both the file picker and drag-drop.

        Both ``target`` modes now register the source file **in place**
        on disk. Nothing is copied or moved by Import / drag-drop:

          * ``"raw"`` (default): appends the absolute path to
            ``Library/Raw/library_raw_inputs.txt`` and scaffolds an
            output folder under the configured output root with a
            ``source_epub.txt`` sidecar pointing at the original
            location. Surfaces as a Not Started card on the In
            Progress tab. Used by the "Import EPUB" button and by
            drag-drop while the **In Progress** tab is active.
          * ``"translated"``: appends the absolute path to
            ``Library/library_translated_inputs.txt``. Surfaces as a
            card on the Completed tab. Only ``.epub`` is accepted;
            other types are reported as skipped. Used by drag-drop
            while the **Completed** tab is active.

        The Organize button is the deliberate move step for both
        modes: it moves the file into ``Library/Raw`` or
        ``Library/Translated`` and records an entry in
        ``library_origins.txt`` so Undo Move can reverse it. Because
        nothing is relocated during Import itself, no collision dialog
        is shown for either target — duplicate registrations simply
        dedupe by normalized path in the registry files.

        The library is refreshed and a summary dialog is shown after
        processing so batch imports don't spam modal messages.
        """
        if not paths:
            return
        imported, skipped, errors = self._run_import(paths, target)
        if imported:
            QTimer.singleShot(0, self._load_books)
        if not (imported or skipped or errors):
            return
        # Drag-drop imports get animated, non-modal status feedback. File-
        # picker imports keep the detailed summary dialog so users who
        # explicitly chose "Import EPUB" still see per-file diagnostics.
        if source == "drop":
            self._show_toast(
                self._import_toast_text(imported, skipped, errors, target))
            return
        title, body = self._import_summary(imported, skipped, errors, target)
        if imported and not errors:
            QMessageBox.information(self, title, body)
        else:
            QMessageBox.warning(self, title, body)

    # _import_single_file moved verbatim to library_core.LibraryShelfMixin (inherited).
    # _run_import, _import_toast_text, _import_summary were extracted from _import_paths_into_library
    # into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _ensure_output_override_matches(self, books: list) -> bool:
        """Warn the user + switch the output-directory override when it
        doesn't match the folder a flash card's translation lives under.

        Flash cards get scanned from BOTH the configured
        ``OUTPUT_DIRECTORY`` override and the implicit default fallback
        (see :func:`_resolve_output_roots`), so the Library surfaces
        books whose workspaces live in either location. Loading a card
        "for translation" without aligning the override means a
        subsequent Run would write the new output under the current
        override root — splitting the progress across two different
        folders instead of resuming inside the card's existing
        workspace. This helper prompts the user whenever that drift is
        detected, and (on confirmation) rewrites the override in
        ``config['output_directory']``, ``os.environ``, the live
        ``other_settings`` UI entry, and the persisted ``config.json``
        — matching what :func:`other_settings._on_output_dir_changed`
        does when the user edits the field by hand.

        Returns True when loading can proceed (no mismatch, or the user
        accepted the switch); False when the user cancels so the caller
        can abort the emit cleanly.
        """
        info = self._output_override_mismatch(books)
        if info is None:
            return True
        new_override = info["new_override"]

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Output Folder Mismatch")
        msg.setText(self._output_override_prompt_text(info))
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec() != QMessageBox.Yes:
            return False

        self._apply_output_override_config(new_override)

        # Sync the Other Settings dialog's live UI entry if present —
        # the field is bound to the same config key via
        # ``_on_output_dir_changed``, so leaving it stale would let the
        # user see the "old" path in the settings dialog until they
        # next edited it. Walking the parent chain mirrors
        # :func:`_persist_config_via_parent`'s traversal so we find
        # whichever ancestor hosts the translator GUI's attributes.
        try:
            parent = self.parent() if hasattr(self, "parent") else None
        except Exception:
            parent = None
        while parent is not None:
            entry = getattr(parent, "output_dir_entry", None)
            if entry is not None and hasattr(entry, "setText"):
                try:
                    entry.blockSignals(True)
                    entry.setText(new_override)
                    entry.blockSignals(False)
                except Exception:
                    logger.debug(
                        "Failed to sync output_dir_entry: %s",
                        traceback.format_exc(),
                    )
                break
            try:
                parent = parent.parent()
            except Exception:
                break

        # Persist to ``config.json`` so the change survives a restart.
        try:
            _persist_config_via_parent(self)
        except Exception:
            logger.debug(
                "Failed to persist config after override switch: %s",
                traceback.format_exc(),
            )

        return True

    # _output_override_mismatch, _output_override_prompt_text, _apply_output_override_config were
    # extracted from _ensure_output_override_matches into library_core.LibraryShelfMixin
    # (see DISCREPANCIES U5 "Phase-1 splits").

    def _compile_epub_for_folder(self, folder: str):
        """Run the EPUB converter on *folder* via the main translator GUI.

        Shared by the flash-card context menu's "Compile EPUB" action —
        same behavior as the Book Details button: the converter runs in the
        main window's worker with progress in the main log, and its own
        busy-state guards apply.
        """
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Compile EPUB",
                "Could not resolve this card's output folder.")
            return
        gui = _find_translator_gui(self)
        if gui is None or not hasattr(gui, "epub_converter"):
            QMessageBox.warning(
                self, "Compile EPUB",
                "The main translator window is not available.")
            return
        try:
            gui.epub_converter(folder=folder)
        except Exception as exc:
            QMessageBox.warning(self, "Compile EPUB",
                                f"Could not start the EPUB converter:\n{exc}")
            return
        # Flip the matching flash cards' corner ribbon to "COMPILING…" and
        # watch the worker so the badge restores when the run ends.
        self._set_cards_compiling(folder, True)
        self._watch_compile_for_folder(folder, gui)

    def _compile_pdf_for_folder(self, folder: str):
        """Rebuild the translated PDF for a PDF output workspace."""
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Compile PDF",
                "Could not resolve this card's output folder.")
            return
        gui = _find_translator_gui(self)
        if gui is None or not hasattr(gui, "pdf_converter"):
            QMessageBox.warning(
                self, "Compile PDF",
                "The main translator window is not available.")
            return
        try:
            gui.pdf_converter(folder=folder)
        except Exception as exc:
            QMessageBox.warning(self, "Compile PDF",
                                f"Could not start the PDF compiler:\n{exc}")
            return
        self._set_cards_compiling(folder, True)
        self._watch_compile_for_folder(folder, gui)

    def _translate_metadata_for_books(self, books: list[dict]) -> bool:
        """Run the normal metadata phase for every resolvable selected EPUB."""
        entries: list[tuple[dict, str, str]] = []
        seen_sources: set[str] = set()
        for book in books or []:
            source_path = _resolve_book_metadata_source(book)
            if not source_path:
                continue
            source_key = os.path.normcase(os.path.abspath(source_path))
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            if book.get("type") == "in_progress":
                output_folder = (
                    book.get("output_folder") or book.get("path", "")
                )
                if not os.path.isdir(output_folder):
                    output_folder = ""
            else:
                output_folder = _resolve_book_output_folder(book)
            entries.append((book, source_path, output_folder))

        if not entries:
            QMessageBox.warning(
                self,
                "Translate Metadata",
                "Could not resolve an original EPUB source for the selection.",
            )
            return False

        # A single output root can be synchronized with the main override in
        # the usual way. Selections spanning multiple roots are routed per
        # book below, so changing the global override would be misleading.
        alignment_books: list[dict] = []
        alignment_roots: set[str] = set()
        for book, _source_path, output_folder in entries:
            alignment_book = book
            if output_folder and not book.get("output_folder"):
                alignment_book = dict(book)
                alignment_book["output_folder"] = output_folder
            alignment_books.append(alignment_book)
            if output_folder:
                alignment_roots.add(os.path.normcase(os.path.abspath(
                    os.path.dirname(output_folder)
                )))
        if len(alignment_roots) <= 1:
            if not self._ensure_output_override_matches(alignment_books):
                return False

        existing_metadata: list[str] = []
        output_roots: dict[str, str] = {}
        for _book, source_path, output_folder in entries:
            if not output_folder:
                continue
            output_roots[os.path.abspath(source_path)] = os.path.dirname(
                os.path.abspath(output_folder)
            )
            metadata_path = os.path.join(output_folder, "metadata.json")
            if os.path.isfile(metadata_path):
                existing_metadata.append(metadata_path)

        if existing_metadata:
            count = len(existing_metadata)
            total = len(entries)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Metadata Already Exists")
            if total == 1:
                warning_text = (
                    "metadata.json already exists for this EPUB.\n\n"
                    "Continuing will regenerate the selected translated "
                    "metadata fields and replace their current translated "
                    "values."
                )
            else:
                warning_text = (
                    f"metadata.json already exists for {count} of the {total} "
                    "selected EPUBs.\n\n"
                    "Continuing will regenerate the selected translated "
                    "metadata fields and replace their current translated "
                    "values."
                )
            msg.setText(warning_text)
            preview = existing_metadata[:8]
            detail = "\n".join(preview)
            if count > len(preview):
                detail += f"\n… and {count - len(preview)} more"
            msg.setInformativeText(detail)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Cancel)
            if msg.exec() != QMessageBox.Yes:
                return False

        gui = _find_translator_gui(self)
        if gui is None or not hasattr(gui, "start_metadata_translation"):
            QMessageBox.warning(
                self,
                "Translate Metadata",
                "The main translator window is not available.",
            )
            return False
        try:
            return bool(gui.start_metadata_translation(
                [source_path for _book, source_path, _folder in entries],
                output_roots=output_roots,
            ))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Translate Metadata",
                f"Could not start metadata translation:\n{exc}",
            )
            return False

    def _translate_metadata_for_book(self, book: dict) -> bool:
        """Book Details compatibility wrapper for a single EPUB."""
        return self._translate_metadata_for_books([book])

    def _cards_for_output_folder(self, folder: str) -> list:
        """Return every flash card whose workspace is *folder*."""
        if not folder:
            return []
        target = os.path.normcase(os.path.normpath(os.path.abspath(folder)))
        matches = []
        for card in (*getattr(self, "_ip_cards", []),
                     *getattr(self, "_comp_cards", [])):
            try:
                book = getattr(card, "book", None) or {}
                if (book.get("type") or "").lower() == "in_progress":
                    cand = book.get("output_folder") or book.get("path", "")
                else:
                    cand = _resolve_book_output_folder(book)
                if cand and os.path.normcase(os.path.normpath(
                        os.path.abspath(cand))) == target:
                    matches.append(card)
            except Exception:
                continue
        return matches

    def _set_cards_compiling(self, folder: str, active: bool):
        for card in self._cards_for_output_folder(folder):
            try:
                card.set_compiling(active)
            except Exception:
                pass

    def _watch_compile_for_folder(self, folder: str, gui):
        """Poll the converter worker; restore the cards' badge when done."""
        self._compile_watch_folder = folder
        self._compile_watch_gui = gui
        self._compile_watch_started = time.monotonic()
        timer = getattr(self, "_compile_watch_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(300)
            timer.timeout.connect(lambda: self._compile_watch_tick())
            self._compile_watch_timer = timer
        timer.start()

    def _compile_watch_tick(self):
        if _epub_converter_running(getattr(self, "_compile_watch_gui", None)):
            return
        # Grace period: don't restore the badge before the executor has
        # even surfaced the worker.
        if (time.monotonic()
                - getattr(self, "_compile_watch_started", 0)) < 1.2:
            return
        timer = getattr(self, "_compile_watch_timer", None)
        if timer is not None:
            timer.stop()
        self._set_cards_compiling(
            getattr(self, "_compile_watch_folder", ""), False)
        self._compile_watch_gui = None

    def _load_for_translation(self, book: dict):
        """Push the card's raw source path into the translator's input field.

        Emits :attr:`import_epub_requested` which :class:`TranslatorGUI`
        handles. Falls back to a warning when no raw path is available.
        Consults :meth:`_ensure_output_override_matches` first so a
        card whose workspace lives under a different output root than
        the current override triggers a warning + automatic switch.
        """
        raw = book.get("raw_source_path") or book.get("path") or ""
        if not raw or not os.path.isfile(raw):
            QMessageBox.warning(self, "Load for translation",
                                "No raw source file is available for this card.")
            return
        if not self._ensure_output_override_matches([book]):
            return
        try:
            self.import_epub_requested.emit(raw)
            record_library_raw_input(raw)
        except Exception:
            logger.debug("Failed to emit import_epub_requested: %s",
                         traceback.format_exc())

    def _load_multi_for_translation(self, paths: list,
                                    books: list | None = None):
        """Push one-or-many raw source paths into the translator's input.

        Deduplicates by normalized path, emits ``import_epubs_requested``
        when more than one file is loaded (so the receiver can show a
        "N files selected" summary), or ``import_epub_requested`` for the
        single-file case. Every loaded path is recorded in the raw-inputs
        registry so subsequent library scans can resolve it.

        When *books* is provided (the context-menu caller does so) we
        first consult :meth:`_ensure_output_override_matches` so any
        card whose workspace lives under a different output root than
        the current override prompts the user to switch the override.
        Omitted ``books`` keeps the legacy "just emit the paths"
        behavior for any future caller that doesn't have the source
        dicts on hand.
        """
        if not paths:
            return
        seen: set[str] = set()
        uniq: list[str] = []
        for p in paths:
            if not p or not os.path.isfile(p):
                continue
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        if not uniq:
            QMessageBox.warning(self, "Load for translation",
                                "No resolvable raw source files in the selection.")
            return
        if books and not self._ensure_output_override_matches(books):
            return
        try:
            if len(uniq) == 1:
                self.import_epub_requested.emit(uniq[0])
            else:
                self.import_epubs_requested.emit(uniq)
            for p in uniq:
                record_library_raw_input(p)
        except Exception:
            logger.debug("Failed to emit multi import: %s",
                         traceback.format_exc())

    def _organize_into_library(self):
        """Move raw sources into Library/Raw *and* compiled EPUBs into
        Library/Translated, recording every move in a reversible origins
        file (``Library/library_origins.txt``). Per-file policy:

          * **Raw** sources are MOVED from their current location into
            Library/Raw and the card's ``source_epub.txt`` pointer is
            rewritten so future scans resolve to the library copy.
          * **Translated** compiled EPUBs are MOVED from their output
            folder into Library/Translated (the output folder itself
            stays on disk, minus the EPUB).

        The operation is idempotent: files already inside their
        destination folder are skipped silently.
        """
        plan = self._plan_organize()
        raw_moves = plan["raw_moves"]
        translated_moves = plan["translated_moves"]

        if not raw_moves and not translated_moves:
            QMessageBox.information(
                self, "Organize Files into Library",
                "All resolvable files are already in Library/Raw or "
                "Library/Translated. Nothing to move.")
            return

        preview = self._organize_preview_lines(plan)

        msg = QMessageBox(self)
        msg.setWindowTitle("Organize Files into Library")
        msg.setText(
            "Move the following files into the Library?\n\n"
            + "\n".join("  \u2022 " + line for line in preview)
            + "\n\nThis is reversible via the Undo Move button."
        )
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec() != QMessageBox.Yes:
            return

        raw_collisions, trans_collisions = self._organize_collisions(plan)
        all_collisions = raw_collisions + trans_collisions
        collision_policy = "keep_both"
        if all_collisions:
            # Combined dest label for the prompt so a mixed
            # raw+translated batch reads naturally.
            if raw_collisions and trans_collisions:
                dest_label = "Library/Raw and Library/Translated"
            elif raw_collisions:
                dest_label = "Library/Raw"
            else:
                dest_label = "Library/Translated"
            collision_policy = self._prompt_duplicate_policy(
                all_collisions, dest_label)
            if collision_policy is None:
                # User cancelled the organize entirely.
                return
        result = self._execute_organize(plan, collision_policy, all_collisions)
        path_moves = result["path_moves"]
        summary = self._organize_summary(result)
        # Notify listeners (the translator GUI) BEFORE the summary
        # modal so when the user dismisses the dialog their input
        # field already reflects the new library location — clicking
        # Run immediately after Organize no longer points at a stale
        # Downloads path.
        if path_moves:
            try:
                self.files_reorganized.emit(list(path_moves))
            except Exception:
                logger.debug("files_reorganized emit failed: %s",
                             traceback.format_exc())
        QMessageBox.information(self, "Organize Files into Library", summary)
        self._load_books()

    # _plan_organize, _organize_preview_lines, _organize_collisions, _execute_organize,
    # _organize_summary were extracted from _organize_into_library into
    # library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _undo_organize_prompt(self):
        """Ask the user which category to undo, then reverse those moves.

        Prompts with three buttons: Raw, Translated, All. Each button
        moves the affected files back to their original locations — by
        origins registry lookup when possible, otherwise falling back
        to a best-guess workspace target derived from title matching
        (same rules the scanner uses to dedup cross-tab duplicates).
        Also rewrites any stale ``source_epub.txt`` pointers.

        Files actually sitting in ``Library/Raw`` / ``Library/Translated``
        are enumerated *on disk* as well as from the origins registry,
        so orphan files dropped in through any non-organize route
        (manual copy, legacy install with a missing origins.txt, etc.)
        still get processed instead of being silently ignored.
        """
        plan = self._plan_undo()
        raw_map = plan["raw_map"]
        trans_map = plan["trans_map"]

        if not raw_map and not trans_map:
            QMessageBox.information(
                self, "Undo Move",
                "No files to undo — Library/Raw and Library/Translated "
                "are both empty and the origins registry is clean.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Undo Move")
        msg.setText(self._undo_prompt_text(plan))
        msg.setIcon(QMessageBox.Question)
        btn_raw = msg.addButton("Raw", QMessageBox.AcceptRole)
        btn_trans = msg.addButton("Translated", QMessageBox.AcceptRole)
        btn_all = msg.addButton("All", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(btn_all if (raw_map and trans_map) else (btn_raw or btn_trans))
        msg.exec()
        chosen = msg.clickedButton()
        if chosen is None or chosen is btn_cancel:
            return
        restore_raw = chosen is btn_raw or chosen is btn_all
        restore_trans = chosen is btn_trans or chosen is btn_all

        undo_collisions = self._undo_collisions(plan, restore_raw, restore_trans)
        undo_policy = "keep_both"
        if undo_collisions:
            undo_policy = self._prompt_duplicate_policy(
                undo_collisions, "their original location")
            if undo_policy is None:
                # User cancelled the undo entirely.
                return
        result = self._execute_undo(
            plan, restore_raw, restore_trans, undo_policy, undo_collisions)
        path_moves = result["path_moves"]
        summary = self._undo_summary(plan, restore_raw, restore_trans, result)
        # Notify listeners (the translator GUI) before the summary
        # modal so stale input paths get rewritten to the restored
        # originals BEFORE the user can click Run again.
        if path_moves:
            try:
                self.files_reorganized.emit(list(path_moves))
            except Exception:
                logger.debug("files_reorganized emit failed: %s",
                             traceback.format_exc())
        QMessageBox.information(self, "Undo Move", summary)
        self._load_books()

    # _plan_undo, _undo_prompt_text, _undo_collisions, _execute_undo, _undo_summary were
    # extracted from _undo_organize_prompt into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _refresh_view(self):
        """Re-filter + render the active tab first; defer the other tab."""
        self._sync_card_size_from_combo()
        active = self._active_card_tab_key()
        inactive = self._other_card_tab_key(active)
        self._dirty_card_tabs.add(inactive)
        self._populate_tab(active)

    # _filtered moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _auto_refresh(self):
        """Lightweight auto-refresh: only reload if either tab changed."""
        if not self.isVisible():
            return
        if (self._delete_thread is not None
                and self._delete_thread.isRunning()):
            return
        loading = getattr(self, "_loading_widget", None)
        if loading is not None and loading.isVisible():
            return
        if self._scanner_thread and self._scanner_thread.isRunning():
            return
        self._scanner_thread = _DualScannerThread(self._config, self)
        self._scanner_thread.scan_finished.connect(self._on_auto_scan_done)
        self._scanner_thread.start()

    def _on_auto_scan_done(self, in_progress: list[dict], completed: list[dict]):
        """Apply a background scan without needlessly rebuilding the shelf.

        Translation normally changes files *inside* one existing output
        workspace. That advances the progress/size/mtime rendered by one card,
        but it does not change the Library's structure. Keep the existing grid
        mounted and replace only cards whose payload changed. A full refresh
        remains the fallback when a workspace is added/removed, moves between
        tabs, or crosses the active search/format filter.

        Date/size sorting is intentionally not re-applied for content-only
        updates: an output folder changes mtime and size for every translated
        chapter. Explicit refreshes, pagination, and sort changes still
        calculate the latest order.
        """
        structure_changed, changed_by_tab, new_by_tab = self._scan_diff(
            in_progress, completed)

        self._in_progress_books = in_progress
        self._completed_books = completed
        if structure_changed:
            self._refresh_view()
        else:
            for tab_key in ("ip", "comp"):
                fresh_by_path = new_by_tab[tab_key]
                # A scan can finish while a page is being streamed. Update its
                # not-yet-mounted payloads instead of restarting the stream.
                state = self._card_stream_states.get(tab_key)
                if state is not None:
                    state["books"] = [
                        fresh_by_path.get(book.get("path", ""), book)
                        for book in state.get("books", [])
                    ]
                for card in self._cards_for_tab_key(tab_key):
                    card_path = str(card.book.get("path", "") or "")
                    if card_path not in changed_by_tab[tab_key]:
                        card.book = fresh_by_path.get(card_path, card.book)
                for path in changed_by_tab[tab_key]:
                    self._replace_mounted_library_card(
                        tab_key, fresh_by_path[path])
        # Always refresh the Organize / Undo counters — origins registry may
        # have changed even when the card list didn't (e.g. after undo).
        self._update_organize_counts()

    # _books_by_path, _book_matches_current_filters moved verbatim to library_core.LibraryShelfMixin (inherited).
    # _scan_diff was extracted from _on_auto_scan_done into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _replace_mounted_library_card(self, tab_key: str, book: dict) -> bool:
        """Replace one mounted card in its existing grid cell.

        Off-page cards have no widget under the paginated cache and therefore
        require no UI work; the fresh payload is already in the backing list.
        """
        path = str(book.get("path", "") or "")
        cache = getattr(
            self, "_comp_card_cache" if tab_key == "comp" else "_ip_card_cache",
            None,
        )
        cards = self._comp_cards if tab_key == "comp" else self._ip_cards
        grid_layout = (
            self._comp_grid_layout if tab_key == "comp" else self._ip_grid_layout
        )
        selected_paths = (
            self._selected_paths_comp if tab_key == "comp"
            else self._selected_paths_ip
        )
        if not path or not cache or path not in cache:
            return False

        cached = cache[path]
        old_card = cached[0]
        try:
            card_index = cards.index(old_card)
        except ValueError:
            # The cache can be detached while this tab is midway through a
            # stream. The caller updated the stream's pending payloads.
            return False
        layout_index = grid_layout.indexOf(old_card)
        if layout_index < 0:
            return False
        row, column, row_span, column_span = grid_layout.getItemPosition(
            layout_index)

        try:
            preset_key = cached[2]
            size_key, card_width = preset_key
        except (IndexError, TypeError, ValueError):
            size_key = self._card_size
            card_width = _SIZE_PRESETS[size_key]["card_w"]
            preset_key = (size_key, card_width)
        preset = dict(_SIZE_PRESETS.get(size_key, _SIZE_PRESETS[self._card_size]))
        preset["card_w"] = int(card_width)
        show_raw_title = bool(cached[3]) if len(cached) > 3 else bool(
            self._show_raw_titles)

        card = _BookCard(
            book, preset=preset, show_raw_title=show_raw_title)
        card.clicked.connect(self._on_card_clicked)
        card.context_menu_requested.connect(self._show_context_menu)
        card.select_requested.connect(self._on_card_select_requested)
        card.hover_changed.connect(self._on_card_hover_changed)
        card._library_hover_connected = True
        self._install_library_drop_target(card, recursive=True)
        card.set_selected(path in selected_paths)

        cached_cover = self._cover_path_cache.get(path)
        if (
            cached_cover
            and cached_cover != "_none_"
            and os.path.isfile(cached_cover)
        ):
            # A single card can reuse its resolved thumbnail immediately; the
            # batching queue is only needed when populating a whole page.
            card.set_cover(cached_cover)
        elif cached_cover is None:
            self._queue_cover_load(
                book, priority=(tab_key == self._active_card_tab_key()))

        was_hovered = (
            getattr(self, "_hovered_card", None) is old_card
            or bool(getattr(old_card, "_hovered", False))
        )
        grid_widget = grid_layout.parentWidget()
        updates_were_enabled = None
        if grid_widget is not None:
            updates_were_enabled = grid_widget.updatesEnabled()
            grid_widget.setUpdatesEnabled(False)
        try:
            grid_layout.removeWidget(old_card)
            grid_layout.addWidget(
                card, row, column, row_span, column_span,
                Qt.AlignTop | Qt.AlignLeft,
            )
            cards[card_index] = card
            cache[path] = (
                card, self._card_signature(book), preset_key, show_raw_title)
            if was_hovered:
                self._hovered_card = card
                card.set_hovered(True)
            old_card.setParent(None)
            old_card.deleteLater()
            card.show()
        finally:
            if grid_widget is not None and updates_were_enabled is not None:
                grid_widget.setUpdatesEnabled(updates_were_enabled)
                grid_widget.update()
        return True

    def _load_books(self):
        """Kick off an async scan of both tabs and show the loading spinner.

        Safe to call whether the loading state is already active (first
        open — :meth:`__init__` primes it before ``show()``) or not
        (later refreshes from the "Refresh" button) — ``_show_loading``
        is idempotent.
        """
        if (self._delete_thread is not None
                and self._delete_thread.isRunning()):
            return
        self._clear_cover_work(stop_active=True)
        if self._scanner_thread and self._scanner_thread.isRunning():
            return
        self._show_loading()
        self._initial_scan_started = True
        self._scanner_thread = _DualScannerThread(self._config, self)
        self._scanner_thread.scan_finished.connect(self._on_initial_scan_done)
        self._scanner_thread.start()

    def _on_initial_scan_done(self, in_progress: list[dict], completed: list[dict]):
        self._in_progress_books = in_progress
        self._completed_books = completed
        self._set_loading_text("Loading books\u2026")
        self._pump_loading_events()
        self._update_organize_counts()

        def _finish_initial_render():
            self._hide_loading()
            # Card columns depend on QScrollArea viewport width. Start the
            # stream after the tabs are visible so startup does not compute
            # a one-column grid from the hidden loading-overlay geometry.
            QTimer.singleShot(40, self._start_visible_card_refresh)

        QTimer.singleShot(0, _finish_initial_render)

    # _card_signature moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _stop_card_stream(self, stream_key: str | None = None) -> None:
        keys = (
            [stream_key]
            if stream_key
            else list(getattr(self, "_card_stream_timers", {}).keys())
        )
        for key in keys:
            timer = getattr(self, "_card_stream_timers", {}).get(key)
            if timer is not None and timer.isActive():
                timer.stop()
            getattr(self, "_card_stream_states", {}).pop(key, None)
            if getattr(self, "_active_card_stream_key", None) == key:
                self._active_card_stream_key = None

    def _card_stream_timer(self, stream_key: str) -> QTimer:
        timers = getattr(self, "_card_stream_timers", None)
        if timers is None:
            self._card_stream_timers = {}
            timers = self._card_stream_timers
        timer = timers.get(stream_key)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(False)
            timer.setInterval(self._CARD_STREAM_TICK_MS)
            timer.timeout.connect(
                lambda key=stream_key: self._populate_card_grid_batch_tick(key))
            timers[stream_key] = timer
        return timer

    def _cover_worker_limit(self) -> int:
        total = (
            len(getattr(self, "_cover_queue", ()) or ())
            + len(getattr(self, "_cover_active_loaders", {}) or {})
        )
        return max(1, _reader_worker_count(max(1, total), self._config))

    def _schedule_cover_pump(self, delay_ms: int | None = None) -> None:
        if not getattr(self, "_cover_queue", None):
            return
        timer = getattr(self, "_cover_pump_timer", None)
        if timer is None:
            self._pump_cover_queue()
            return
        if delay_ms is None:
            delay_ms = (
                self._COVER_PUMP_BUSY_MS
                if self._is_card_stream_active()
                else self._COVER_PUMP_IDLE_MS
            )
        timer.start(max(0, int(delay_ms)))

    def _queue_cover_load(self, book: dict, *, priority: bool = False) -> None:
        path = book.get("path", "") or ""
        if not path:
            return
        if path in self._cover_queued_paths or path in self._cover_active_paths:
            return
        job = {
            "generation": self._cover_generation,
            "path": path,
            "file_type": book.get("type", "epub"),
            "original_path": book.get("original_path"),
            "raw_source_path": book.get("raw_source_path"),
            "priority": bool(priority),
        }
        if priority:
            insert_at = len(self._cover_queue)
            for idx, queued in enumerate(self._cover_queue):
                if not bool(queued.get("priority", False)):
                    insert_at = idx
                    break
            self._cover_queue.insert(insert_at, job)
        else:
            self._cover_queue.append(job)
        self._cover_queued_paths.add(path)
        self._schedule_cover_pump()

    def _pump_cover_queue(self) -> None:
        if self._is_card_stream_active():
            self._schedule_cover_pump(self._COVER_PUMP_BUSY_MS)
            return
        limit = self._cover_worker_limit()
        while self._cover_queue and len(self._cover_active_loaders) < limit:
            job = self._cover_queue.popleft()
            path = job.get("path", "") or ""
            self._cover_queued_paths.discard(path)
            if not path or path in self._cover_active_paths:
                continue
            loader = _CoverLoader(
                path,
                file_type=job.get("file_type", "epub"),
                config=self._config,
                original_path=job.get("original_path"),
                raw_source_path=job.get("raw_source_path"),
                parent=self,
            )
            loader._library_cover_generation = job.get(
                "generation", self._cover_generation)
            loader.result_ready.connect(
                lambda book_path, cover_path, ldr=loader: self._on_cover_loaded(
                    book_path, cover_path, ldr))
            loader.finished.connect(
                lambda ldr=loader: self._on_cover_thread_finished(ldr))
            self._cover_active_loaders[loader] = path
            self._cover_active_paths.add(path)
            self._cover_threads.append(loader)
            loader.start()

    def _queue_cover_apply(
        self,
        book_path: str,
        cover_path: str,
        *,
        priority: bool = False,
    ) -> None:
        if not book_path or not cover_path:
            return
        if book_path in self._cover_apply_pending_paths:
            return
        item = (book_path, cover_path, bool(priority))
        if priority:
            insert_at = len(self._cover_apply_queue)
            for idx, queued in enumerate(self._cover_apply_queue):
                queued_priority = bool(queued[2]) if len(queued) > 2 else False
                if not queued_priority:
                    insert_at = idx
                    break
            self._cover_apply_queue.insert(insert_at, item)
        else:
            self._cover_apply_queue.append(item)
        self._cover_apply_pending_paths.add(book_path)
        if not self._cover_apply_timer.isActive():
            self._cover_apply_timer.start()

    def _queue_visible_missing_covers(self) -> None:
        if not self.isVisible():
            return
        for card in list(self._cards_for_tab_key(self._active_card_tab_key())):
            if getattr(card, "_has_cover", False):
                continue
            book = getattr(card, "book", {}) or {}
            path = book.get("path", "") or ""
            if not path:
                continue
            cached_cover = self._cover_path_cache.get(path)
            if cached_cover is not None:
                if (cached_cover and cached_cover != "_none_"
                        and os.path.isfile(cached_cover)):
                    self._queue_cover_apply(path, cached_cover, priority=True)
                elif (cached_cover == ""
                        and book.get("type", "epub") in ("epub", "in_progress")):
                    self._cover_path_cache[path] = "_none_"
                    self._queue_cover_load(book, priority=True)
            else:
                self._queue_cover_load(book, priority=True)

    def _cover_apply_is_active_priority(self, book_path: str) -> bool:
        active_key = self._active_card_tab_key()
        for card in self._cards_for_tab_key(active_key):
            if card.book.get("path") == book_path:
                return True
        return False

    def _apply_cover_batch(self) -> None:
        if not self._cover_apply_queue:
            self._cover_apply_timer.stop()
            self._schedule_inactive_tab_population()
            return
        processed = 0
        while self._cover_apply_queue and processed < self._COVER_APPLY_BATCH_SIZE:
            book_path, cover_path, _priority = self._cover_apply_queue.popleft()
            self._cover_apply_pending_paths.discard(book_path)
            for card in (*self._ip_cards, *self._comp_cards):
                if card.book.get("path") == book_path:
                    card.set_cover(cover_path)
            processed += 1
        self._request_card_hover_reconcile(750)
        if not self._cover_apply_queue:
            self._cover_apply_timer.stop()
            self._schedule_inactive_tab_population()

    def _clear_cover_work(self, *, stop_active: bool = False) -> None:
        self._cover_generation += 1
        self._cover_queue.clear()
        self._cover_queued_paths.clear()
        self._cover_apply_queue.clear()
        self._cover_apply_pending_paths.clear()
        if self._cover_apply_timer.isActive():
            self._cover_apply_timer.stop()
        cover_pump_timer = getattr(self, "_cover_pump_timer", None)
        if cover_pump_timer is not None and cover_pump_timer.isActive():
            cover_pump_timer.stop()
        if not stop_active:
            return
        for thread in list(getattr(self, "_cover_threads", []) or []):
            _stop_qthread_safely(
                thread, timeout_ms=500, signal_names=("result_ready",))
        self._cover_threads.clear()
        self._cover_active_loaders.clear()
        self._cover_active_paths.clear()

    def _prune_pending_cover_work(self, allowed_paths: set[str]) -> None:
        """Drop queued cover work for cards no longer on either visible page."""
        allowed_paths = {path for path in allowed_paths if path}
        self._cover_queue = deque(
            job for job in self._cover_queue
            if job.get("path", "") in allowed_paths
        )
        self._cover_queued_paths = {
            job.get("path", "") for job in self._cover_queue
            if job.get("path", "")
        }
        self._cover_apply_queue = deque(
            item for item in self._cover_apply_queue
            if item[0] in allowed_paths
        )
        self._cover_apply_pending_paths = {
            item[0] for item in self._cover_apply_queue
        }
        if not self._cover_apply_queue and self._cover_apply_timer.isActive():
            self._cover_apply_timer.stop()

    def _add_streamed_card(self, state: dict, idx: int, book: dict) -> None:
        path = book.get("path", "") or ""
        card_cache = state["card_cache"]
        selected_paths = state["selected_paths"]
        preset_key = state["preset_key"]
        show_raw_title = state["show_raw_title"]
        sig = self._card_signature(book)
        cached = card_cache.get(path)
        reuse = bool(
            cached
            and cached[1] == sig
            and cached[2] == preset_key
            and cached[3] == show_raw_title
        )
        priority_cover = state.get("stream_key") == self._active_card_tab_key()
        if reuse:
            card = cached[0]
            try:
                card.book = book
            except Exception:
                pass
            cached_cover = self._cover_path_cache.get(path)
            if (cached_cover and cached_cover != "_none_"
                    and os.path.isfile(cached_cover)
                    and not getattr(card, "_has_cover", False)):
                self._queue_cover_apply(
                    path, cached_cover, priority=priority_cover)
        else:
            if cached:
                try:
                    cached[0].setParent(None)
                    cached[0].deleteLater()
                except Exception:
                    pass
            card = _BookCard(
                book, preset=state["preset"], show_raw_title=show_raw_title)
            card.clicked.connect(self._on_card_clicked)
            card.context_menu_requested.connect(self._show_context_menu)
            card.select_requested.connect(self._on_card_select_requested)
            self._install_library_drop_target(card, recursive=True)
            cached_cover = self._cover_path_cache.get(path)
            run_loader = False
            if cached_cover is not None:
                if cached_cover and cached_cover != "_none_" and os.path.isfile(cached_cover):
                    self._queue_cover_apply(
                        path, cached_cover, priority=priority_cover)
                elif cached_cover == "" and book.get("type", "epub") in ("epub", "in_progress"):
                    self._cover_path_cache[path] = "_none_"
                    run_loader = True
            else:
                run_loader = True
            if run_loader and path:
                self._queue_cover_load(book, priority=priority_cover)
        if not getattr(card, "_library_hover_connected", False):
            try:
                card.hover_changed.connect(self._on_card_hover_changed)
                card._library_hover_connected = True
            except Exception:
                pass
        card_cache[path] = (card, sig, preset_key, show_raw_title)
        card.set_selected(path in selected_paths)
        state["card_list"].append(card)
        row, col = divmod(idx, state["cols"])
        state["grid_layout"].addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
        card.show()

    def _populate_card_grid_batch_tick(self, stream_key: str) -> None:
        state = getattr(self, "_card_stream_states", {}).get(stream_key)
        if not state:
            timer = getattr(self, "_card_stream_timers", {}).get(stream_key)
            if timer is not None:
                timer.stop()
            return
        if state.get("generation") != self._card_stream_generation.get(stream_key):
            self._stop_card_stream(stream_key)
            return
        books = state["books"]
        start = int(state.get("idx", 0) or 0)
        if start >= len(books):
            self._finish_card_grid_stream(stream_key)
            return

        end_limit = min(start + self._CARD_STREAM_BATCH_SIZE, len(books))
        deadline = time.perf_counter() + self._CARD_STREAM_FRAME_BUDGET_SEC
        end = start
        grid_widget = state.get("grid_widget")
        updates_were_enabled = None
        if grid_widget is not None:
            try:
                updates_were_enabled = grid_widget.updatesEnabled()
                grid_widget.setUpdatesEnabled(False)
            except Exception:
                updates_were_enabled = None
        try:
            for i in range(start, end_limit):
                self._add_streamed_card(state, i, books[i])
                end = i + 1
                if end > start and time.perf_counter() >= deadline:
                    break
        finally:
            if grid_widget is not None and updates_were_enabled is not None:
                try:
                    grid_widget.setUpdatesEnabled(updates_were_enabled)
                    grid_widget.update()
                except Exception:
                    pass
        state["idx"] = end
        if state.get("loading_visible"):
            self._pump_loading_events()
        self._request_card_hover_reconcile(750)
        if end >= len(books):
            self._finish_card_grid_stream(stream_key)

    def _finish_card_grid_stream(self, stream_key: str) -> None:
        state = getattr(self, "_card_stream_states", {}).pop(stream_key, None)
        timer = getattr(self, "_card_stream_timers", {}).get(stream_key)
        if timer is not None:
            timer.stop()
        if getattr(self, "_active_card_stream_key", None) == stream_key:
            self._active_card_stream_key = None
        if not state:
            return
        grid_layout = state["grid_layout"]
        books = state["books"]
        cols = max(1, int(state.get("cols", 1) or 1))
        for c in range(max(grid_layout.columnCount(), cols + 2, 64)):
            grid_layout.setColumnStretch(c, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        last_row = (len(books) - 1) // cols + 1
        try:
            row_limit = max(grid_layout.rowCount(), last_row + 2)
        except Exception:
            row_limit = last_row + 2
        for r in range(row_limit):
            grid_layout.setRowStretch(r, 0)
        grid_layout.setRowStretch(last_row, 1)
        grid_widget = state.get("grid_widget")
        if grid_widget is not None:
            try:
                grid_widget.updateGeometry()
                grid_widget.update()
            except Exception:
                pass
        if state.get("loading_visible"):
            self._pump_loading_events()
        self._request_card_hover_reconcile(1500)
        self._reconcile_card_hover()
        if stream_key in getattr(self, "_pending_card_reflow", set()):
            self._grid_reflow_timer.start(90)
        if stream_key == self._active_card_tab_key():
            self._schedule_cover_pump(self._COVER_PUMP_IDLE_MS)
            self._schedule_inactive_tab_population()

    def _populate_grid_common(
        self,
        books: list[dict],
        grid_layout: QGridLayout,
        card_list: list[_BookCard],
        count_label: QLabel,
        empty_label: QLabel,
        count_word: str,
        selected_paths: set[str] | None = None,
        card_cache: dict | None = None,
        full_books: list[dict] | None = None,
        scroll_area: QScrollArea | None = None,
        stream_key: str = "grid",
        displayed_count: int | None = None,
        retain_only_visible_cache: bool = False,
    ):
        """Shared render pipeline used by both tabs.

        Cards are cached per path in *card_cache* (one dict per tab,
        owned by the dialog). Each entry maps ``path → (card,
        signature, preset_key, show_raw_title)`` so the next call can
        tell at a glance whether the cached widget is still valid.
        Valid cards are detached from the grid (not deleted), re-laid
        out in the new order, and re-shown. Only new / stale entries
        pay the full ``_BookCard`` construction cost (title-fit loop +
        cover-loader thread spawn).

        *books* is the list that actually renders (already sorted /
        search-filtered / format-filtered). *full_books* is the raw
        scan result for this tab — used to distinguish "filtered
        out" from "genuinely gone" so a Format chip toggle never
        deletes cards it's about to need again on the next click.
        Callers that don't pass *full_books* fall back to the
        previous behaviour (any path not in *books* is treated as
        gone).
        """
        stream_key = stream_key or "grid"
        self._set_hovered_card(None)
        self._request_card_hover_reconcile()
        for other_key in list(getattr(self, "_card_stream_states", {}).keys()):
            if other_key != stream_key:
                self._stop_card_stream(other_key)
                self._dirty_card_tabs.add(other_key)
        self._stop_card_stream(stream_key)
        self._card_stream_generation[stream_key] = (
            int(self._card_stream_generation.get(stream_key, 0) or 0) + 1
        )
        generation = self._card_stream_generation[stream_key]
        selected_paths = selected_paths if selected_paths is not None else set()
        card_cache = card_cache if card_cache is not None else {}
        loading_visible = bool(
            getattr(self, "_loading_widget", None) is not None
            and self._loading_widget.isVisible()
        )
        grid_widget = grid_layout.parentWidget()
        _grid_updates_were_enabled = None
        if grid_widget is not None:
            try:
                _grid_updates_were_enabled = grid_widget.updatesEnabled()
                grid_widget.setUpdatesEnabled(False)
            except Exception:
                _grid_updates_were_enabled = None
        visible_paths = {b.get("path", "") for b in books}
        # "Known" paths for the underlying data. When the caller
        # hands us the full unfiltered list we use it; otherwise we
        # assume ``books`` IS the full set (back-compat behaviour).
        full_paths = (
            {b.get("path", "") for b in full_books}
            if full_books is not None else visible_paths)
        cache_keep_paths = (
            visible_paths if retain_only_visible_cache else full_paths)
        if retain_only_visible_cache:
            other_visible_paths = {
                card.book.get("path", "")
                for card in self._cards_for_tab_key(
                    self._other_card_tab_key(stream_key))
            }
            self._prune_pending_cover_work(
                visible_paths | other_visible_paths)
        # Prune the selection set of paths that no longer exist in
        # the scan result (NOT the filtered view) so switching
        # Format chips doesn't silently wipe a multi-selection that
        # only some of whose cards match the current filter.
        selected_paths &= full_paths

        # Detach every cached card from the grid before we rebuild it.
        # Detaching (``setParent(None)``) is cheap and preserves the
        # widget for reuse; deleting is what makes filter toggles
        # slow, so we only delete cards whose path is gone from the
        # underlying scan result — filter misses stay in the cache
        # so the next toggle can re-use them without another
        # ``_BookCard`` constructor pass.
        for path, cached in list(card_cache.items()):
            cached_card = cached[0]
            try:
                grid_layout.removeWidget(cached_card)
                cached_card.setParent(None)
                cached_card.hide()
            except Exception:
                pass

        for stale_path in [p for p in card_cache if p not in cache_keep_paths]:
            try:
                cached_card = card_cache[stale_path][0]
                cached_card.setParent(None)
                cached_card.deleteLater()
            except Exception:
                pass
            del card_cache[stale_path]

        # Also clear stragglers the grid might still hold (e.g.
        # widgets we never tracked, or layout items left over from
        # a size-preset change).
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            w = item.widget()
            if w and w not in (c[0] for c in card_cache.values()):
                w.setParent(None)

        card_list.clear()

        if not books:
            empty_label.show()
            count_label.setText("")
            if grid_widget is not None and _grid_updates_were_enabled is not None:
                try:
                    grid_widget.setUpdatesEnabled(_grid_updates_were_enabled)
                    grid_widget.update()
                except Exception:
                    pass
            if loading_visible:
                self._pump_loading_events()
            return

        empty_label.hide()
        total_for_label = (
            len(books) if displayed_count is None else int(displayed_count))
        count_label.setText(
            f"{total_for_label} {count_word}"
            f"{'s' if total_for_label != 1 else ''}"
        )

        preset = _SIZE_PRESETS[self._card_size]
        preset_card_w = preset["card_w"]
        spacing = preset["spacing"]
        grid_layout.setHorizontalSpacing(spacing)
        grid_layout.setVerticalSpacing(spacing + 2)
        # Resolve geometry through the dialog/visible sibling as well as the
        # target viewport. This rejects stale-but-plausible first-layout and
        # hidden-tab widths, the two cases that previously left half the
        # Library blank until zoom forced a second render.
        viewport_w = self._effective_grid_width(scroll_area, grid_widget)
        self._grid_layout_widths[stream_key] = viewport_w
        cols = max(1, (viewport_w + spacing) // (preset_card_w + spacing))
        # Redistribute the leftover horizontal space across the cards
        # themselves so the grid fills the viewport instead of leaving
        # a dead column to the right. Each card grows by up to
        # ``(viewport_w - cols*preset_card_w - (cols-1)*spacing) //
        # cols`` pixels, never shrinks below the preset minimum, and
        # caps a few pixels tight of the raw math to stay inside the
        # viewport even when Qt's QGridLayout rounds fractional widths
        # up. The resulting ``card_w`` is propagated through a shallow
        # preset copy so :class:`_BookCard` renders at the new width
        # (cover + title labels + fitted title font all derive from
        # ``preset['card_w']``).
        inner_w = max(preset_card_w * cols,
                      viewport_w - (cols - 1) * spacing)
        card_w = max(preset_card_w, inner_w // cols)
        if card_w != preset_card_w:
            preset = dict(preset)
            preset["card_w"] = card_w
        # Include the effective card width in the cache key so a
        # window resize that changes the expansion factor forces
        # dependent cards to rebuild with the new width. Otherwise
        # the cache would keep serving cards sized for the previous
        # viewport and the new space would re-appear as a dead column.
        preset_key = (self._card_size, card_w)

        show_raw_title = bool(getattr(self, "_show_raw_titles", False))
        if grid_widget is not None and _grid_updates_were_enabled is not None:
            try:
                grid_widget.setUpdatesEnabled(_grid_updates_were_enabled)
                grid_widget.update()
            except Exception:
                pass

        self._active_card_stream_key = stream_key
        self._card_stream_states[stream_key] = {
            "generation": generation,
            "books": list(books),
            "idx": 0,
            "grid_layout": grid_layout,
            "grid_widget": grid_widget,
            "card_list": card_list,
            "card_cache": card_cache,
            "selected_paths": selected_paths,
            "preset": preset,
            "preset_key": preset_key,
            "show_raw_title": show_raw_title,
            "cols": cols,
            "loading_visible": loading_visible,
            "stream_key": stream_key,
        }
        self._card_stream_timer(stream_key).start()
        if loading_visible:
            self._pump_loading_events()

    def _populate_in_progress(
        self, books: list[dict], filtered_count: int | None = None
    ):
        if not hasattr(self, "_ip_card_cache"):
            self._ip_card_cache: dict = {}
        self._populate_grid_common(
            books, self._ip_grid_layout, self._ip_cards,
            self._ip_count_label, self._ip_empty_label, "novel",
            selected_paths=self._selected_paths_ip,
            card_cache=self._ip_card_cache,
            scroll_area=self._ip_scroll,
            # Pass the full unfiltered scan result so the cache's
            # stale-entry sweep only removes cards whose underlying
            # book is genuinely gone. Filter misses stay in the
            # cache, so toggling Format / search is an O(visible)
            # detach + re-add pass instead of a full rebuild.
            full_books=self._in_progress_books,
            stream_key="ip",
            displayed_count=filtered_count,
            retain_only_visible_cache=True,
        )

    def _populate_completed(
        self, books: list[dict], filtered_count: int | None = None
    ):
        if not hasattr(self, "_comp_card_cache"):
            self._comp_card_cache: dict = {}
        self._populate_grid_common(
            books, self._comp_grid_layout, self._comp_cards,
            self._comp_count_label, self._comp_empty_label, "book",
            selected_paths=self._selected_paths_comp,
            card_cache=self._comp_card_cache,
            full_books=self._completed_books,
            scroll_area=self._comp_scroll,
            stream_key="comp",
            displayed_count=filtered_count,
            retain_only_visible_cache=True,
        )

    def _active_selection(self) -> tuple[set[str], list[_BookCard]]:
        """Return (selection_set, card_list) for the currently active tab."""
        if self._tabs.currentIndex() == 0:
            return self._selected_paths_ip, self._ip_cards
        return self._selected_paths_comp, self._comp_cards

    def _on_rubber_band_selection(self, books: list, modifiers):
        """Apply rubber-band drag selection to the active tab.

        Ctrl / Shift preserves the existing selection and adds to it;
        a plain drag replaces the selection with whatever landed in the
        rubber-band rectangle.
        """
        selected, cards = self._active_selection()
        paths = {b.get("path", "") for b in books if b.get("path", "")}
        extend = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))
        if not extend:
            selected.clear()
        selected.update(paths)
        for c in cards:
            c.set_selected(c.book.get("path", "") in selected)

    def _on_empty_area_clicked(self, modifiers):
        """Clear the active-tab selection when user clicks empty grid space.

        Skipped when the user is modifying the existing selection with
        Ctrl / Shift — otherwise a fat-fingered click would wipe a
        carefully-built multi-selection.
        """
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            return
        selected, cards = self._active_selection()
        if not selected:
            return
        selected.clear()
        for c in cards:
            c.set_selected(False)

    def _on_card_select_requested(self, book: dict, modifiers):
        """Handle left-click selection with modifier semantics.

        - Ctrl-click toggles this card in the active-tab selection set.
        - Shift-click adds without clearing.
        - Plain click replaces the selection with just this card.

        Visual updates are applied to every card in the active tab so
        multi-selection reads correctly (highlighted borders).
        """
        selected, cards = self._active_selection()
        path = book.get("path", "") or ""
        if not path:
            return
        if modifiers & Qt.ControlModifier:
            if path in selected:
                selected.discard(path)
            else:
                selected.add(path)
        elif modifiers & Qt.ShiftModifier:
            selected.add(path)
        else:
            selected.clear()
            selected.add(path)
        for c in cards:
            c.set_selected(c.book.get("path", "") in selected)

    def _on_cover_loaded(self, book_path: str, cover_path: str, loader=None):
        if loader is not None:
            if getattr(loader, "_library_cover_generation", None) != self._cover_generation:
                self._schedule_cover_pump(self._COVER_PUMP_BUSY_MS)
                return
        # Cache every loader result, including empty strings for books
        # with no cover, so subsequent card rebuilds can skip extraction.
        if cover_path:
            self._cover_path_cache[book_path] = cover_path
        elif self._cover_path_cache.get(book_path) != "_none_":
            self._cover_path_cache[book_path] = ""
        if not cover_path:
            self._schedule_cover_pump(self._COVER_PUMP_BUSY_MS)
            self._schedule_inactive_tab_population()
            return
        self._queue_cover_apply(
            book_path, cover_path,
            priority=self._cover_apply_is_active_priority(book_path))
        self._schedule_cover_pump(self._COVER_PUMP_BUSY_MS)

    def _on_cover_thread_finished(self, loader=None) -> None:
        if loader is None:
            return
        path = None
        try:
            path = self._cover_active_loaders.pop(loader, None)
        except Exception:
            path = None
        if path:
            try:
                self._cover_active_paths.discard(path)
            except Exception:
                pass
        try:
            if loader in self._cover_threads:
                self._cover_threads.remove(loader)
        except Exception:
            pass
        try:
            loader.deleteLater()
        except Exception:
            pass
        self._schedule_cover_pump(self._COVER_PUMP_BUSY_MS)

    def _apply_filter(self, text):
        self._reset_library_pages()
        self._reset_library_scroll_position("ip")
        self._reset_library_scroll_position("comp")
        self._refresh_view()

    @staticmethod
    def _card_has_reader_workspace(book) -> bool:
        folder = book.get("output_folder") or _resolve_book_output_folder(book)
        return bool(
            folder
            and os.path.isdir(folder)
            and os.path.isfile(os.path.join(folder, "translation_progress.json"))
        )

    def _open_card_file_direct(self, book) -> None:
        """Open the concrete card file, bypassing Book Details."""
        try:
            from PySide6.QtGui import QDesktopServices

            path = str(book.get("path") or "")
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(path or "No file path is available")
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
                raise OSError(f"The system could not open {path}")
        except Exception as exc:
            logger.error("Could not open file: %s\n%s", exc, traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")

    def _on_card_clicked(self, book):
        file_type = book.get("type", "epub")
        if file_type == "txt" or (
                file_type == "pdf" and not self._card_has_reader_workspace(book)):
            # Open with best available editor/viewer (cross-platform)
            try:
                path = book["path"]
                _no_window = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                if sys.platform == 'win32':
                    if file_type == "txt":
                        # Prefer Notepad++ for text files
                        _npp_paths = [
                            r'C:\Program Files\Notepad++\notepad++.exe',
                            r'C:\Program Files (x86)\Notepad++\notepad++.exe',
                        ]
                        _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                        if _npp:
                            subprocess.Popen([_npp, path], creationflags=_no_window)
                        else:
                            subprocess.Popen(['notepad.exe', path], creationflags=_no_window)
                    else:
                        os.startfile(path)  # PDF → default viewer
                elif sys.platform == 'darwin':
                    if file_type == "txt" and shutil.which('code'):
                        subprocess.Popen(['code', path])
                    else:
                        subprocess.Popen(['open', path])
                else:
                    if file_type == "txt":
                        _editors = ['gedit', 'kate', 'code', 'mousepad', 'xed', 'pluma']
                        _editor = next((e for e in _editors if shutil.which(e)), 'xdg-open')
                        subprocess.Popen([_editor, path])
                    else:
                        subprocess.Popen(['xdg-open', path])
            except Exception as exc:
                logger.error("Could not open file: %s\n%s", exc, traceback.format_exc())
                QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")
            return
        # EPUB or translated PDF workspace: open the web-like Book Details page
        # instead of jumping into the reader. Users who want the direct behavior
        # can use the "Open in Reader" context menu action.
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            details_book = dict(book or {})
            book_path = details_book.get("path", "") or ""
            cached_cover = self._cover_path_cache.get(book_path)
            if (cached_cover and cached_cover != "_none_"
                    and os.path.isfile(cached_cover)):
                details_book["_cached_cover_path"] = cached_cover
            dialog = BookDetailsDialog(details_book, config=self._config, parent=self)
            QApplication.restoreOverrideCursor()
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            self._active_details = dialog  # prevent GC
            dialog.show()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            tb = traceback.format_exc()
            logger.error("Could not open book details: %s\n%s", exc, tb)
            QMessageBox.warning(self, "Error", f"Could not open book details:\n{exc}\n\nDetails:\n{tb}")

    def _open_reader_direct(self, book):
        """Bypass the details page and open the integrated HTML reader."""
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            workspace = book.get("output_folder") or _resolve_book_output_folder(book)
            if workspace and self._card_has_reader_workspace(book):
                source = book.get("raw_source_path") or _read_source_epub_pointer(workspace) or ""
                if source.lower().endswith(".pdf") and os.path.isfile(source):
                    reader = EpubReaderDialog(
                        source,
                        config=self._config,
                        parent=self,
                        workspace_dir=workspace,
                        initial_show_raw=False,
                        window_title=f"{book.get('name') or os.path.basename(workspace)} (Translated)",
                    )
                    QApplication.restoreOverrideCursor()
                    reader.setModal(False)
                    reader.setAttribute(Qt.WA_DeleteOnClose)
                    self._active_reader = reader
                    reader.show()
                    return
            # When the card carries a distinct raw source (Completed-tab
            # EPUB that was organized alongside its Library/Raw pair, or
            # an in-progress promotion) pass it as the reader's alt so
            # the Show-raw pill can flip between translated and source.
            book_path = book.get("path", "")
            alt_path = book.get("raw_source_path", "") or ""
            if alt_path:
                try:
                    if os.path.normcase(os.path.abspath(alt_path)) == \
                            os.path.normcase(os.path.abspath(book_path)):
                        alt_path = ""
                except Exception:
                    alt_path = ""
                if alt_path and not os.path.isfile(alt_path):
                    alt_path = ""
            reader = EpubReaderDialog(
                book_path,
                config=self._config,
                parent=self,
                alt_epub_path=alt_path or None,
                toc_output_dir=workspace or None,
            )
            QApplication.restoreOverrideCursor()
            reader.setModal(False)
            reader.setAttribute(Qt.WA_DeleteOnClose)
            self._active_reader = reader  # prevent GC
            reader.show()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            tb = traceback.format_exc()
            logger.error("Could not open EPUB: %s\n%s", exc, tb)
            QMessageBox.warning(self, "Error", f"Could not open EPUB:\n{exc}\n\nDetails:\n{tb}")

    def _show_context_menu(self, book, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 9pt; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background: #3a3a5e; }
        """)
        # Selection-aware: if the right-clicked card isn't part of the
        # current selection, treat this as a single-card action (replace
        # selection with just this card). Otherwise keep the existing
        # multi-selection intact so "Load N for translation" can act on it.
        selected, cards = self._active_selection()
        path = book.get("path", "") or ""
        if path and path not in selected:
            selected.clear()
            selected.add(path)
            for c in cards:
                c.set_selected(c.book.get("path", "") in selected)
        selected_books = [c.book for c in cards
                          if c.book.get("path", "") in selected]
        if not selected_books:
            selected_books = [book]
        # Pre-resolve ``raw_source_path`` on every selected book
        # BEFORE any gate runs. :func:`_resolve_book_source_file`
        # caches the result on the book dict via the same
        # ``_find_raw_source_for_library_epub`` fallback that Book
        # Details uses, so Load / Reveal / Clear all see the same
        # resolved path. Without this, a library-filed card whose
        # scan left ``raw_source_path`` empty would show Reveal
        # (the resolver's first call) while Load and Clear silently
        # missed \u2014 exactly the \"Book Details finds it fine,
        # context menu can't\" inconsistency the user hit.
        for _b in selected_books:
            try:
                _resolve_book_source_file(_b)
            except Exception:
                logger.debug(
                    "Context-menu raw-source prefetch failed: %s",
                    traceback.format_exc())
        file_type = book.get("type", "epub")
        if file_type == "in_progress":
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            # Always expose "Open in Reader" when an EPUB source is
            # resolvable — either the raw source (Not Started / mid-
            # translation cards) or a compiled translated EPUB.
            # NOTE: ``QAction.triggered`` emits a ``bool checked`` argument
            # which Qt binds to lambda *default* parameters, so we MUST close
            # over the loop-free locals without using ``lambda x=local: …``
            # (otherwise ``x`` gets overwritten with the checked bool and
            # subsequent ``x.get(…)`` calls crash with AttributeError).
            raw_src = book.get("raw_source_path") or ""
            out_epub = book.get("output_epub_path") or ""
            if (raw_src and raw_src.lower().endswith(".epub")
                    and os.path.isfile(raw_src)):
                raw_reader_action = menu.addAction("\U0001f4d6  Open in Reader")
                raw_reader_action.triggered.connect(
                    lambda: self._open_reader_direct({"path": raw_src, "type": "epub"})
                )
            if out_epub and os.path.isfile(out_epub):
                reader_action = menu.addAction("\U0001f4d6  Open Translated EPUB")
                reader_action.triggered.connect(
                    lambda: self._open_reader_direct({"path": out_epub, "type": "epub"})
                )
            if (raw_src and raw_src.lower().endswith(".pdf")
                    and os.path.isfile(raw_src)
                    and self._card_has_reader_workspace(book)):
                pdf_reader_action = menu.addAction(
                    "\U0001f4d6  Open in EPUB reader"
                )
                pdf_reader_action.triggered.connect(
                    lambda: self._open_reader_direct(book)
                )
        elif file_type == "epub":
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            reader_action = menu.addAction("\U0001f4d6  Open in Reader")
            reader_action.triggered.connect(lambda: self._open_reader_direct(book))
        elif file_type == "pdf" and self._card_has_reader_workspace(book):
            details_action = menu.addAction("\U0001f4d1  Open Book Details")
            details_action.triggered.connect(lambda: self._on_card_clicked(book))
            reader_action = menu.addAction("\U0001f4d6  Open in EPUB reader")
            reader_action.triggered.connect(lambda: self._open_reader_direct(book))
            open_action = menu.addAction("\U0001f4c2  Open File")
            open_action.triggered.connect(lambda: self._open_card_file_direct(book))
        else:
            open_action = menu.addAction("\U0001f4c2  Open File")
            if file_type == "txt":
                open_action.triggered.connect(lambda: self._on_card_clicked(book))
            else:
                open_action.triggered.connect(lambda: self._open_card_file_direct(book))
        # "Load for translation" — pushes the card's raw source into the
        # translator's input field (moved out of the Import EPUB button).
        # When multiple cards are selected, the label becomes
        # "Load N files for translation" and the action emits all resolvable
        # raw paths at once via :attr:`import_epubs_requested`.
        def _resolve_raw(b: dict) -> str:
            r = b.get("raw_source_path") or ""
            if (not r and b.get("type") == "epub"
                    and os.path.isfile(b.get("path", ""))):
                r = b["path"]
            return r if r and os.path.isfile(r) else ""

        raw_candidates = [p for p in (_resolve_raw(b) for b in selected_books) if p]
        if raw_candidates:
            menu.addSeparator()
            if len(raw_candidates) > 1:
                label = f"\U0001f501  Load {len(raw_candidates)} files for translation"
                tooltip = ("Set these files as the translator's current input "
                           "(batch translation).")
            else:
                label = "\U0001f501  Load for translation"
                tooltip = "Set this file as the translator's current input."
            load_action = menu.addAction(label)
            load_action.setToolTip(tooltip)
            # See "NOTE" above — close over ``raw_candidates`` without a
            # default arg so Qt's ``checked`` bool can't overwrite it.
            # Snapshot ``selected_books`` so
            # :meth:`_ensure_output_override_matches` can consult each
            # book's ``output_folder`` before the emit \u2014 without
            # the dicts we'd only have paths and no way to check
            # whether any card's workspace lives under a mismatched
            # output root.
            load_books_snapshot = list(selected_books)
            load_action.triggered.connect(
                lambda: self._load_multi_for_translation(
                    raw_candidates, books=load_books_snapshot))
        menu.addSeparator()
        folder_action = menu.addAction("\U0001f4c2  Open Output Folder")
        # Resolution rules:
        #   * In-progress cards: the card ``path`` IS the output folder.
        #   * Anything else: use :func:`_resolve_book_output_folder` so
        #     Library/Translated entries fall back to the origins
        #     registry's original output folder (mirrors the Book
        #     Details 📁 button). Previously this branch used
        #     ``os.path.dirname(book['path'])`` as the fallback, which
        #     for library-organized EPUBs pointed at ``Library/
        #     Translated`` instead of the workspace that actually
        #     contains ``translation_progress.json`` / response_*.
        #   * Last-resort fallback: the book's own containing folder,
        #     same as before, so non-library files without any
        #     registered origin still get *some* folder opened.
        if file_type == "in_progress":
            output_folder = book.get("output_folder") or book.get("path", "")
        else:
            output_folder = (_resolve_book_output_folder(book)
                             or os.path.dirname(book.get("path", "")))
        folder_action.triggered.connect(lambda: _open_folder_in_explorer(output_folder))
        # Run the normal title/metadata phase for every selected EPUB. The
        # translator processes them in order, or concurrently when its Batch
        # Translation toggle is enabled.
        metadata_books = [
            selected_book for selected_book in selected_books
            if _resolve_book_metadata_source(selected_book)
        ]
        if metadata_books:
            if len(metadata_books) > 1:
                metadata_label = (
                    f"\U0001f310  Translate Metadata for "
                    f"{len(metadata_books)} EPUBs"
                )
            else:
                metadata_label = "\U0001f310  Translate Metadata"
            metadata_action = menu.addAction(metadata_label)
            metadata_action.setToolTip(
                "Translate the configured metadata fields for the selected "
                "EPUBs only."
            )
            metadata_books_snapshot = list(metadata_books)
            metadata_action.triggered.connect(
                lambda *_a, books=metadata_books_snapshot:
                    self._translate_metadata_for_books(books)
            )
        # "Compile EPUB" — explicitly runs the EPUB converter phase on the
        # card's output workspace (same action as the Book Details button;
        # single-chapter Translate runs never compile on their own). Only
        # offered when the workspace actually holds translation progress.
        compile_folder = output_folder
        if not (compile_folder and os.path.isfile(os.path.join(
                compile_folder, "translation_progress.json"))):
            compile_folder = _resolve_book_output_folder(book)
        if compile_folder and os.path.isfile(os.path.join(
                compile_folder, "translation_progress.json")):
            compile_kind = _workspace_compile_kind(book, compile_folder)
            if compile_kind == "pdf":
                compile_action = menu.addAction("\U0001f4c4  Compile PDF")
                compile_action.setToolTip(
                    "Build the output PDF from this workspace's translated "
                    f"sections:\n{compile_folder}")
                compile_action.triggered.connect(
                    lambda *_a, f=compile_folder:
                    self._compile_pdf_for_folder(f))
            else:
                compile_action = menu.addAction("\U0001f4d8  Compile EPUB")
                compile_action.setToolTip(
                    "Build the output EPUB from this workspace's translated "
                    f"chapters:\n{compile_folder}")
                compile_action.triggered.connect(
                    lambda *_a, f=compile_folder:
                    self._compile_epub_for_folder(f))
        # Reveal the raw source file \u2014 identical to the Book Details
        # \U0001f517 button. Resolves the same way (raw_source_path, then
        # origins lookup for Library/Translated entries, then book path).
        # The action is only added when a real source file exists on
        # disk; when it can't be resolved (e.g. compiled EPUB with no
        # matching raw) we omit it rather than surface a dead entry.
        source_target = _resolve_book_source_file(book)
        if source_target and os.path.isfile(source_target):
            source_action = menu.addAction("\U0001f517  Reveal source file")
            source_action.setToolTip(
                f"Reveal source file:\n{source_target}"
            )
            source_action.triggered.connect(
                lambda: _open_folder_in_explorer(source_target))
        # Reveal the compiled / translated EPUB itself. Resolution
        # lives in :func:`_resolve_book_translated_file` so this menu
        # action and the Book Details 📕 button share one code path.
        # Omitted entirely when no translated EPUB exists on disk.
        translation_target = _resolve_book_translated_file(book)
        if translation_target and os.path.isfile(translation_target):
            translation_action = menu.addAction(
                "\U0001f4d5  Reveal Translated File")
            translation_action.setToolTip(
                f"Reveal translated file:\n{translation_target}"
            )
            translation_action.triggered.connect(
                lambda: _open_folder_in_explorer(translation_target))
        menu.addSeparator()
        copy_path_action = menu.addAction("\U0001f4cb  Copy Path")
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(book["path"]))
        # "Clear saved raw link" \u2014 lets the user reset a bad
        # Scan-for-Raw pairing (e.g. an EPUB workspace that got
        # auto-paired to a ``.pdf`` via an earlier broken match).
        # Deletes the ``source_epub.txt`` sidecar and unregisters
        # the stale raw from the raw-inputs registry so the next
        # library scan re-derives ``workspace_kind`` from the real
        # folder contents. Only offered for workspace-backed cards
        # whose sidecar actually exists on disk \u2014 a card without
        # a saved link has nothing to clear. Sits right above Delete
        # because it's a "corrective" action in the same destructive-
        # but-recoverable family.
        clear_targets = [
            b for b in selected_books
            if self._card_has_saved_raw_link(b)
        ]
        if clear_targets:
            if len(clear_targets) > 1:
                clear_label = (
                    f"\u2702\ufe0f  Clear saved raw link for "
                    f"{len(clear_targets)} items")
                clear_tip = (
                    "Remove the saved raw-source pointer "
                    "(source_epub.txt) from each selected "
                    "workspace. Useful when a previous auto-scan "
                    "matched the wrong file \u2014 the next library "
                    "scan will re-detect the workspace kind from "
                    "the folder contents.")
            else:
                clear_label = "\u2702\ufe0f  Clear saved raw link"
                clear_tip = (
                    "Remove this workspace's saved raw-source "
                    "pointer (source_epub.txt). Use this if the "
                    "auto-scan matched a .pdf to an EPUB (or any "
                    "other wrong-kind pairing) \u2014 the library "
                    "will re-scan with the real folder kind.")
            clear_action = menu.addAction(clear_label)
            clear_action.setToolTip(clear_tip)
            clear_action.triggered.connect(
                lambda: self._clear_saved_raw_link(clear_targets))
        # Delete \u2014 tab-specific semantics:
        #   * In Progress \u2192 delete the output folder (recursive)
        #   * Completed   \u2192 delete the .epub / compiled file
        # Works with multi-selection too ("Delete N items"). Always asks
        # for confirmation first.
        menu.addSeparator()
        delete_targets = list(selected_books) if len(selected_books) > 1 else [book]
        n_del = len(delete_targets)
        if n_del > 1:
            delete_label = f"\U0001f5d1\ufe0f  Delete {n_del} items"
        else:
            delete_label = "\U0001f5d1\ufe0f  Delete"
        delete_action = menu.addAction(delete_label)
        delete_action.setToolTip(
            "In Progress \u2192 deletes the output folder.  "
            "Completed \u2192 deletes the EPUB file.  "
            "(Asks for confirmation first.)"
        )
        delete_action.triggered.connect(
            lambda: self._delete_books_prompt(delete_targets)
        )
        menu.exec(pos)

    # _raw_is_in_library_raw, _library_raw_match_for_book, _card_has_saved_raw_link moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _clear_saved_raw_link(self, books: list):
        """Delete the ``source_epub.txt`` sidecar(s) after confirmation.

        Also unregisters the previously-pointed-at raw path from the
        raw-inputs registry when it's no longer referenced by any
        other workspace, so the Scan-for-Raw dialog won't try to
        re-apply the same stale pairing on its next run. On success
        the library is reloaded so the scanner re-derives
        ``workspace_kind`` from the real folder contents (which
        fixes the \"EPUB workspace wearing a PDF badge\" symptom).
        """
        if not books:
            return
        targets = self._plan_clear_raw_link(books)
        if not targets:
            return
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Clear saved raw link")
        msg.setText(self._clear_raw_link_prompt_text(targets))
        msg.setStandardButtons(
            QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec() != QMessageBox.Yes:
            return
        cleared = self._execute_clear_raw_link(targets)
        if cleared:
            # Reload so the scanner re-classifies workspace_kind from
            # the folder contents \u2014 that's what flips the card
            # badge back from PDF to EPUB for mis-paired entries.
            self._load_books()
        else:
            QMessageBox.warning(
                self, "Clear saved raw link",
                "None of the selected workspaces could be updated. "
                "Check that the folders still exist on disk.")

    # _plan_clear_raw_link, _clear_raw_link_prompt_text, _execute_clear_raw_link were extracted
    # from _clear_saved_raw_link into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _delete_books_prompt(self, books: list):
        """Confirm + delete the backing file / folder for each book.

        Target resolution:

          * **Library entries** (``in_library=True``) — delete
            ``book['path']`` as a single file. These live in
            ``Library/Translated`` and don't own a surrounding workspace.
          * **Output-folder cards** (``type == "in_progress"`` OR any
            completed card with a resolvable ``output_folder``) —
            delete the folder recursively so every sibling artefact
            goes with it.
          * Everything else — delete ``book['path']`` as a plain file.

        Confirmation policy:

          * When every selected card is "Not Started" (no translated
            chapters on disk yet), a plain Yes / Cancel dialog is
            enough — there's no meaningful work to lose.
          * Otherwise the user has to type the word "Halgakos"
            (case-insensitive) into a confirmation field before the
            Delete button enables. This is an intentional speed bump
            because the delete is recursive and unrecoverable: a
            completed translation's ``response_*.html`` files,
            ``translation_progress.json``, ``images/``, and the
            compiled EPUB all go together when the card gets removed.
        """
        if not books:
            return
        if (self._delete_thread is not None
                and self._delete_thread.isRunning()):
            QMessageBox.information(
                self, "Delete",
                "A delete operation is already running. "
                "Wait for it to finish before starting another one.",
            )
            return
        targets, unregister_cards = self._plan_delete(books)
        # Perform silent unregistrations for unsafe cards (the "Add
        # Translation from Downloads" flow). These operate directly on
        # the tracking files — no confirmation, no message box, no
        # summary line — since the physical file is left untouched.
        def _unregister_unsafe_cards() -> int:
            return self._unregister_cards(unregister_cards)

        if not targets:
            # No disk deletes to perform — still honor the user's
            # intent by unregistering any unsafe card(s) so the flash
            # card disappears. Silent: no dialog, just refresh.
            if unregister_cards:
                _unregister_unsafe_cards()
                QTimer.singleShot(0, self._load_books)
                return
            QMessageBox.information(
                self, "Delete",
                "Nothing to delete \u2014 none of the selected cards "
                "point at a file or folder on disk.",
            )
            return

        all_not_started = self._all_targets_not_started(targets)

        if all_not_started:
            confirmed_targets = self._confirm_delete_simple(targets)
        else:
            confirmed_targets = self._confirm_delete_with_keyword(targets)
        if not confirmed_targets:
            return
        # The typed-keyword dialog returns the *filtered* subset the
        # user chose to keep enabled in the checkbox list. The simple
        # dialog always returns the full batch (Not Started cards don't
        # offer per-target opt-out since there's nothing to lose).
        targets = list(confirmed_targets)

        delete_targets = [
            (label, pth, is_folder) for label, pth, is_folder, _b in targets
        ]
        self._start_delete_worker(
            delete_targets,
            len(targets),
            _unregister_unsafe_cards,
        )
        return

    # _plan_delete, _unregister_cards, _all_targets_not_started were extracted from
    # _delete_books_prompt into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _start_delete_worker(
            self,
            targets: list[tuple[str, str, bool]],
            target_count: int,
            unregister_callback) -> None:
        """Start a background, parallel delete batch."""
        if not targets:
            return
        if (self._delete_thread is not None
                and self._delete_thread.isRunning()):
            QMessageBox.information(
                self, "Delete",
                "A delete operation is already running. "
                "Wait for it to finish before starting another one.",
            )
            return

        from PySide6.QtWidgets import QProgressDialog

        total = len(targets)
        progress = QProgressDialog("Deleting selected items...", "", 0,
                                   total, self)
        progress.setWindowTitle("Delete")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        self._delete_progress = progress

        auto_refresh_was_active = False
        try:
            auto_refresh_was_active = self._auto_refresh_timer.isActive()
            if auto_refresh_was_active:
                self._auto_refresh_timer.stop()
        except Exception:
            auto_refresh_was_active = False

        worker = _LibraryDeleteThread(targets, self)
        self._delete_thread = worker
        worker.progress.connect(self._on_delete_progress)
        worker.delete_finished.connect(
            lambda results, w=worker, count=target_count,
                   cb=unregister_callback, resume=auto_refresh_was_active:
            self._on_delete_finished(w, results, count, cb, resume)
        )
        worker.finished.connect(worker.deleteLater)
        progress.show()
        worker.start()

    @Slot(int, int, str)
    def _on_delete_progress(self, done: int, total: int, label: str) -> None:
        progress = getattr(self, "_delete_progress", None)
        if progress is None:
            return
        progress.setLabelText(
            f"Deleting selected items... ({done}/{total})\n{label}"
        )
        progress.setMaximum(max(1, total))
        progress.setValue(done)

    def _on_delete_finished(
            self,
            worker: _LibraryDeleteThread,
            results: list,
            target_count: int,
            unregister_callback,
            resume_auto_refresh: bool) -> None:
        progress = getattr(self, "_delete_progress", None)
        if progress is not None:
            progress.close()
        self._delete_progress = None
        if self._delete_thread is worker:
            self._delete_thread = None

        deleted, errors, summary = self._delete_result_summary(
            results, target_count)

        # Unsafe cards are registry-only removals. Keep them out of the
        # visible summary, matching the old behavior.
        try:
            unregister_callback()
        except Exception:
            logger.debug("Silent unregister after delete failed: %s",
                         traceback.format_exc())

        if errors:
            QMessageBox.warning(self, "Delete", summary)
        else:
            QMessageBox.information(self, "Delete", summary)

        if resume_auto_refresh:
            try:
                self._auto_refresh_timer.start()
            except Exception:
                pass
        QTimer.singleShot(0, self._load_books)

    # _delete_result_summary was extracted from _on_delete_finished into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    # -- Delete-confirmation helpers ----------------------------------------
    # _DELETE_KEYWORDS, _summarize_folder_contents, _format_delete_detail moved verbatim to library_core.LibraryShelfMixin (inherited).

    def _build_target_row_widget(
        self, target: tuple[str, str, bool, dict], checkbox
    ) -> QWidget:
        """Render a single delete target as a checkbox + detail block.

        The checkbox is created by the caller so it can track the
        per-target include/exclude state; this method just owns the
        layout and the label that sits next to it (path, size, and for
        folders the artefact breakdown).
        """
        label, pth, is_folder, _book = target
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)
        row_layout.setSpacing(8)

        checkbox.setChecked(True)
        # The caller now constructs the checkbox via
        # :func:`_create_styled_checkbox`, so its stylesheet already
        # matches Other Settings exactly \u2014 don't overwrite it here.
        checkbox.setToolTip(
            "Uncheck to exclude this item from the delete batch."
        )
        row_layout.addWidget(checkbox, 0, Qt.AlignTop)

        # Text block describing what this target is and what's inside.
        lines: list[str] = []
        if is_folder:
            lines.append(f"{label}  \u2014  output folder")
            lines.append(pth)
            contents = self._summarize_folder_contents(pth)
            if contents:
                lines.extend(contents)
            else:
                lines.append("    \u00b7 (folder is empty)")
        else:
            try:
                size = os.path.getsize(pth)
                if size >= 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size / 1024:.0f} KB"
            except OSError:
                size_str = "?"
            lines.append(f"{label}  \u2014  file ({size_str})")
            lines.append(pth)

        detail = QLabel("\n".join(lines))
        detail.setTextFormat(Qt.PlainText)
        detail.setWordWrap(True)
        detail.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        detail.setStyleSheet(
            "QLabel { color: #e0e0e0; "
            "font-family: Consolas, Menlo, monospace; font-size: 9pt; }"
        )
        row_layout.addWidget(detail, 1)
        return row

    def _confirm_delete_simple(
        self, targets: list[tuple[str, str, bool, dict]]
    ) -> list[tuple[str, str, bool, dict]] | None:
        """Yes / Cancel confirmation for Not-Started-only batches.

        Not-Started cards have no translated work on disk so the
        simple prompt doesn't bother with per-target checkboxes; the
        whole batch goes or nothing does. Returns the full targets
        list on accept, ``None`` on cancel — matching the shape of
        :meth:`_confirm_delete_with_keyword`.
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Delete")
        msg.setText(self._delete_simple_prompt_text(targets))
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        if msg.exec() != QMessageBox.Yes:
            return None
        return list(targets)

    # _delete_simple_prompt_text was extracted from _confirm_delete_simple into library_core.LibraryShelfMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _confirm_delete_with_keyword(
        self, targets: list[tuple[str, str, bool, dict]]
    ) -> list[tuple[str, str, bool, dict]] | None:
        """Typed-keyword confirmation for in-progress / completed cards.

        Shows a modal dialog with a scrollable, per-target checkbox
        list so the user can untick anything they don't actually want
        removed (common case: keep the Library/Raw copy when
        deleting a compiled Library/Translated entry). The Delete
        button is disabled until:

          * at least one checkbox is still ticked, AND
          * the user types :attr:`_DELETE_KEYWORDS` (case-insensitive,
            leading/trailing whitespace forgiven) into the entry field.

        The speed-bump is deliberately harder to bypass than a Yes/No
        because each target wipes real translation work (response
        HTMLs, progress JSON, images, compiled EPUBs).

        Returns the filtered list of targets still checked when the
        user clicked Delete, or ``None`` if the dialog was cancelled.
        """
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QCheckBox, QFrame,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Delete \u2014 confirmation required")
        dlg.setModal(True)
        dlg.setMinimumWidth(620)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        headline = QLabel(
            f"\u26a0  Permanent delete \u2014 {len(targets)} "
            f"item{'s' if len(targets) != 1 else ''}"
        )
        headline.setStyleSheet(
            "color: #ff8080; font-size: 13pt; font-weight: bold;"
        )
        layout.addWidget(headline)

        # Count folders vs files in the batch for the subtitle.
        folder_count = sum(1 for _l, _p, f, _b in targets if f)
        file_count = len(targets) - folder_count
        subtitle_bits = []
        if folder_count:
            subtitle_bits.append(
                f"{folder_count} output folder"
                f"{'s' if folder_count != 1 else ''}"
            )
        if file_count:
            subtitle_bits.append(
                f"{file_count} file{'s' if file_count != 1 else ''}"
            )
        subtitle = QLabel(
            "Uncheck anything you want to keep. Everything still "
            "checked below will be removed from disk ("
            + " + ".join(subtitle_bits) + "):"
        )
        subtitle.setStyleSheet("color: #c8cbe0; font-size: 10pt;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Scrollable list of per-target rows so arbitrarily large batches
        # (multi-select + auto-queued Library/Raw copies) don't blow the
        # dialog off the bottom of the screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: #1a1a2a; border: 1px solid #3a3a5e; "
            "border-radius: 4px; }"
            "QScrollBar:vertical { width: 10px; background: #12121e; }"
            "QScrollBar::handle:vertical { background: #3a3a5e; "
            "border-radius: 5px; min-height: 24px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
            "{ height: 0; }"
        )
        scroll.setMinimumHeight(220)
        scroll.setMaximumHeight(360)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 6, 10, 6)
        inner_layout.setSpacing(2)

        checkboxes: list[QCheckBox] = []
        for idx, target in enumerate(targets):
            if idx > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setStyleSheet("color: #2a2a3e; background: #2a2a3e;")
                sep.setFixedHeight(1)
                inner_layout.addWidget(sep)
            cb = _create_styled_checkbox()
            checkboxes.append(cb)
            row_widget = self._build_target_row_widget(target, cb)
            inner_layout.addWidget(row_widget)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        warning = QLabel(
            "This removes the checked item(s) permanently \u2014 for "
            "output folders that's recursive: every translated chapter, "
            "progress tracker, image, and compiled EPUB inside goes "
            "with it. This cannot be undone."
        )
        warning.setStyleSheet(
            "color: #ffb347; font-size: 9.5pt; "
            "background: rgba(255, 179, 71, 0.12); "
            "border: 1px solid #ffb347; border-radius: 4px; "
            "padding: 6px 10px;"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        # Typed-keyword prompt. Showing both expected words in the
        # instruction + one of them in the placeholder makes it obvious
        # the user needs to type EXACTLY that — no discovery required.
        pretty = " or ".join(
            f"<b>{kw.capitalize()}</b>" for kw in self._DELETE_KEYWORDS
        )
        instr = QLabel(
            f"Type {pretty} below to unlock the Delete button "
            f"(case-insensitive):"
        )
        instr.setTextFormat(Qt.RichText)
        instr.setStyleSheet("color: #c8cbe0; font-size: 10pt;")
        layout.addWidget(instr)

        entry = QLineEdit()
        entry.setPlaceholderText(self._DELETE_KEYWORDS[0].capitalize())
        entry.setStyleSheet(
            "QLineEdit { background: #1e1e2e; color: #e0e0e0; "
            "border: 1px solid #3a3a5e; border-radius: 4px; "
            "padding: 6px 10px; font-size: 10pt; }"
            "QLineEdit:focus { border-color: #6c63ff; }"
        )
        layout.addWidget(entry)

        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        delete_btn = QPushButton("\U0001f5d1\ufe0f  Delete")
        delete_btn.setEnabled(False)
        delete_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border: none; "
            "border-radius: 4px; padding: 6px 18px; font-weight: bold; }"
            "QPushButton:hover:enabled { background: #e74c3c; }"
            "QPushButton:disabled { background: #3a3a3a; color: #888; }"
        )
        btns.addButton(delete_btn, QDialogButtonBox.AcceptRole)
        btns.rejected.connect(dlg.reject)
        delete_btn.clicked.connect(dlg.accept)
        layout.addWidget(btns)

        def _recompute_enabled(*_args) -> None:
            keyword_ok = (
                entry.text().strip().lower() in self._DELETE_KEYWORDS
            )
            any_checked = any(cb.isChecked() for cb in checkboxes)
            delete_btn.setEnabled(keyword_ok and any_checked)

        entry.textChanged.connect(_recompute_enabled)
        for cb in checkboxes:
            cb.toggled.connect(_recompute_enabled)
        entry.setFocus()
        if dlg.exec() != QDialog.Accepted:
            return None
        return [
            targets[i] for i, cb in enumerate(checkboxes) if cb.isChecked()
        ]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the drag overlay + toast glued to the current geometry — they
        # don't participate in the root layout since they float above it.
        self._reposition_overlays()
        # Debounce: re-flow cards once the user stops dragging (300ms). An
        # active stream may have captured its column count before adding its
        # first card, so an empty card list is not enough reason to ignore the
        # resize.
        active_key = self._active_card_tab_key()
        if not (
            self._cards_for_tab_key(active_key)
            or self._is_card_stream_active(active_key)
        ):
            return
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(
                self._schedule_grid_reflow_if_width_changed)
        self._resize_timer.start(300)

    def _reposition_overlays(self):
        """Re-layout the drop overlay + toast after a resize / show event."""
        if getattr(self, "_drop_overlay", None) is not None:
            margin = 16
            self._drop_overlay.setGeometry(
                margin, margin,
                max(0, self.width() - margin * 2),
                max(0, self.height() - margin * 2),
            )
            self._drop_overlay.raise_()
        if getattr(self, "_toast", None) is not None:
            self._toast.adjustSize()
            tw = min(self._toast.width(), max(240, self.width() - 40))
            th = self._toast.height()
            self._toast.setGeometry(
                (self.width() - tw) // 2,
                self.height() - th - 24,
                tw, th,
            )
            self._toast.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition_overlays()
        self._request_card_hover_reconcile()
        if not getattr(self, "_initial_scan_started", False):
            QTimer.singleShot(0, self._load_books)
        else:
            QTimer.singleShot(0, self._auto_refresh)
        try:
            if not self._auto_refresh_timer.isActive():
                self._auto_refresh_timer.start()
        except Exception:
            pass
        active_key = self._active_card_tab_key()
        if (active_key in getattr(self, "_dirty_card_tabs", set())
                and (self._in_progress_books or self._completed_books)):
            QTimer.singleShot(
                0, lambda key=active_key: self._populate_tab(key))
        elif self._ip_cards or self._comp_cards:
            self._request_card_hover_reconcile(1500)
            QTimer.singleShot(0, self._queue_visible_missing_covers)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._clear_card_hover()
        try:
            self._auto_refresh_timer.stop()
        except Exception:
            pass
        interrupted = set(getattr(self, "_card_stream_states", {}) or {})
        interrupted.update(getattr(self, "_pending_card_reflow", set()) or set())
        try:
            self._grid_reflow_timer.stop()
        except Exception:
            pass
        try:
            self._inactive_tab_populate_timer.stop()
        except Exception:
            pass
        self._pending_card_reflow.clear()
        self._stop_card_stream()
        self._dirty_card_tabs.update(interrupted)
        self._clear_cover_work(stop_active=True)

    def shutdown(self) -> None:
        """Close owned dialogs and stop workers before the app exits."""
        self._shutdown_requested = True
        for attr in ("_active_reader", "_active_details"):
            child = getattr(self, attr, None)
            if child is None:
                continue
            try:
                child.close()
                QApplication.processEvents()
            except Exception:
                pass
            setattr(self, attr, None)
        try:
            self._auto_refresh_timer.stop()
        except Exception:
            pass
        self._clear_card_hover()
        self._stop_card_stream()
        self._clear_cover_work(stop_active=True)
        _stop_qthread_safely(
            getattr(self, "_scanner_thread", None),
            timeout_ms=1500,
            signal_names=("scan_finished",),
        )
        self._scanner_thread = None
        _stop_qthread_safely(
            getattr(self, "_delete_thread", None),
            timeout_ms=1500,
            signal_names=("progress", "delete_finished"),
        )
        self._delete_thread = None
        _persist_config_via_parent(self)

    def closeEvent(self, event):
        """Hide the dialog instead of closing \u2014 persist settings.

        Also asks the translator parent to flush the in-memory config
        dict to ``config.json`` via :func:`_persist_config_via_parent`,
        so library settings survive app crashes / force-quits instead
        of only persisting when the main window's own save runs.
        """
        self._config['epub_library_sort'] = self._sort_mode
        self._config['epub_library_card_size'] = self._card_size
        self._config['epub_library_tab'] = self._current_tab
        _persist_config_via_parent(self)
        if getattr(self, "_shutdown_requested", False):
            event.accept()
            return
        event.ignore()
        self.hide()

    # -- Drag & drop import -------------------------------------------------

    # Extensions accepted by each tab's drop zone. The In Progress tab
    # lands everything on the raw source pipeline (EPUB / TXT / PDF /
    # HTML) while the Completed tab only makes sense for finished
    # compiled EPUBs, so it filters down to just ``.epub``.
    _DND_RAW_EXTS = (".epub", ".txt", ".pdf", ".html", ".htm")
    _DND_TRANSLATED_EXTS = (".epub",)

    def _drop_target_kind(self) -> str:
        """Return ``"translated"`` when the Completed tab is active, else ``"raw"``."""
        try:
            return "translated" if self._tabs.currentIndex() == 1 else "raw"
        except Exception:
            return "raw"

    def _allowed_drop_exts(self) -> tuple:
        """Return the tuple of file extensions currently accepted by the drop zone."""
        return (self._DND_TRANSLATED_EXTS
                if self._drop_target_kind() == "translated"
                else self._DND_RAW_EXTS)

    def _install_library_drop_target(self, widget, recursive: bool = False) -> None:
        """Make a child widget participate in the library-level drop zone."""
        if widget is None:
            return
        try:
            widget.setAcceptDrops(True)
        except Exception:
            pass
        try:
            widget.installEventFilter(self)
        except Exception:
            pass
        if not recursive:
            return
        try:
            for child in widget.findChildren(QWidget):
                try:
                    child.setAcceptDrops(True)
                except Exception:
                    pass
                try:
                    child.installEventFilter(self)
                except Exception:
                    pass
        except Exception:
            pass

    def _install_library_drop_targets(self) -> None:
        """Register the visible library canvas as a drop target.

        Relying only on the top-level dialog works on some systems, but
        QScrollArea viewports/cards can own the cursor area during drag
        negotiation on others, producing a forbidden cursor before the
        dialog-level handler sees the event.
        """
        for widget in (
            getattr(self, "_tabs", None),
            getattr(self, "_ip_tab", None),
            getattr(self, "_comp_tab", None),
            getattr(self, "_scan_tab", None),
            getattr(self, "_ip_scroll", None),
            getattr(self, "_comp_scroll", None),
            getattr(self, "_ip_grid_container", None),
            getattr(self, "_comp_grid_container", None),
            getattr(self, "_ip_empty_label", None),
            getattr(self, "_comp_empty_label", None),
        ):
            self._install_library_drop_target(widget)
        for scroll in (getattr(self, "_ip_scroll", None), getattr(self, "_comp_scroll", None)):
            try:
                self._install_library_drop_target(scroll.viewport())
            except Exception:
                pass

    def _dnd_accept_event(self, event) -> bool:
        """Return True iff the drag payload contains at least one file accepted
        by the currently active tab (EPUB-only on Completed, EPUB/TXT/PDF/HTML
        on In Progress)."""
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            return False
        allowed = self._allowed_drop_exts()
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if path and path.lower().endswith(allowed):
                return True
        return False

    def _handle_library_drop_filter_event(self, event) -> bool:
        from PySide6.QtCore import QEvent
        etype = event.type()
        if etype in (QEvent.DragEnter, QEvent.DragMove):
            if self._dnd_accept_event(event):
                event.setDropAction(Qt.CopyAction)
                event.acceptProposedAction()
                if etype == QEvent.DragEnter:
                    self._show_drop_overlay(True)
                return True
            return False
        if etype == QEvent.DragLeave:
            self._show_drop_overlay(False)
            return False
        if etype == QEvent.Drop:
            self.dropEvent(event)
            return event.isAccepted()
        return False

    def dragEnterEvent(self, event):
        if self._dnd_accept_event(event):
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
            self._show_drop_overlay(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._dnd_accept_event(event):
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._show_drop_overlay(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._show_drop_overlay(False)
        mime = event.mimeData()
        if not mime or not mime.hasUrls():
            event.ignore()
            return
        target = self._drop_target_kind()
        allowed = self._allowed_drop_exts()
        paths: list[str] = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if p and p.lower().endswith(allowed):
                paths.append(p)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        # Stay on the active tab: drops on Completed mean "file these
        # compiled EPUBs into Library/Translated" and the user should see
        # the Completed list refresh in-place. Drops on In Progress stay
        # on that tab so the new Raw import card appears where expected.
        dest_label = "Library/Translated" if target == "translated" else "Library/Raw"
        # Immediate toast so the user sees something happened even before
        # the import pipeline finishes. The pipeline will replace this with
        # a final "Imported N into <dest>" status in
        # :meth:`_import_paths_into_library`.
        self._show_toast(
            f"\U0001f4e5  Importing {len(paths)} file"
            f"{'s' if len(paths) != 1 else ''} into {dest_label}\u2026",
            auto_hide_ms=0,
        )
        self._import_paths_into_library(paths, source="drop", target=target)

    # -- Overlay / toast helpers --------------------------------------------

    def _show_drop_overlay(self, visible: bool):
        """Show / hide the purple "drop to import" overlay panel.

        The overlay's text is refreshed on every show so the user sees
        which library subfolder the active tab will route drops into
        (Raw for In Progress, Translated for Completed).
        """
        if getattr(self, "_drop_overlay", None) is None:
            return
        if visible:
            if self._drop_target_kind() == "translated":
                self._drop_overlay.setText(
                    "\U0001f4e5\n\nDrop EPUBs to import into Library/Translated\n\n"
                    "EPUB only"
                )
            else:
                self._drop_overlay.setText(
                    "\U0001f4e5\n\nDrop files to import into Library/Raw\n\n"
                    "EPUB \u00b7 PDF \u00b7 TXT \u00b7 HTML"
                )
            self._reposition_overlays()
            self._drop_overlay.show()
            self._drop_overlay.raise_()
        else:
            self._drop_overlay.hide()

    def _show_toast(self, text: str, auto_hide_ms: int = 2600):
        """Fade in a short status line at the bottom of the dialog.

        *auto_hide_ms* == 0 keeps the toast visible until the next
        :meth:`_show_toast` call or an explicit :meth:`_fade_out_toast`.
        Used by the drag-drop pipeline to stream a two-stage "Importing…"
        → "Imported N" status update without any modal interruption.
        """
        if getattr(self, "_toast", None) is None:
            return
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        self._toast_hide_timer.stop()
        self._toast.setText(text)
        self._reposition_overlays()
        self._toast.show()
        self._toast.raise_()
        # Cancel any in-flight fade so a rapid second call doesn't race
        # the previous animation to zero opacity.
        if self._toast_anim is not None:
            try:
                self._toast_anim.stop()
            except Exception:
                pass
        anim = QPropertyAnimation(self._toast_opacity, b"opacity", self)
        anim.setDuration(180)
        anim.setStartValue(float(self._toast_opacity.opacity()))
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._toast_anim = anim
        if auto_hide_ms > 0:
            self._toast_hide_timer.start(auto_hide_ms)

    def _fade_out_toast(self):
        """Fade the toast back to 0 opacity and hide it on completion."""
        if getattr(self, "_toast", None) is None or not self._toast.isVisible():
            return
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        if self._toast_anim is not None:
            try:
                self._toast_anim.stop()
            except Exception:
                pass
        anim = QPropertyAnimation(self._toast_opacity, b"opacity", self)
        anim.setDuration(260)
        anim.setStartValue(float(self._toast_opacity.opacity()))
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self._toast.hide)
        anim.start()
        self._toast_anim = anim


# ---------------------------------------------------------------------------
# Book Details — metadata / TOC parser
# ---------------------------------------------------------------------------

# _RE_HTML_TITLE, _RE_HTML_HEADING, _RE_HTML_STRIP_TAGS, _RE_HTML_WS moved verbatim to library_core (imported above).


# _parse_native_toc_txt, _parse_native_toc_ncx, _find_reader_sidecar, _load_reader_native_toc, _native_toc_target_key, _map_native_toc_to_chapters moved verbatim to reader_doc (imported above).


# _extract_html_title_fast, _parse_epub_details, _read_translated_chapter_title moved verbatim to library_core (imported above).


class _BookDetailsLoader(BookDetailsLoaderMixin, QThread):
    """Parse EPUB metadata + TOC + translation status off the UI thread.

    Emits two signals during the lifetime of a single ``run()`` call:

    * :attr:`preview_ready` — fires as soon as the OPF metadata + cover
      are extracted (fast path, no per-chapter HTML parsing). The details
      dialog renders the hero (cover, title, synopsis, metadata grid)
      immediately from this payload so the user isn't staring at a
      Halgakos placeholder while ~400 chapter HTML blobs are decoded.
    * :attr:`done` — fires once the full chapters_info list (including
      per-chapter raw titles, translated titles, and on-disk status)
      has been built. Drives the chapter list + progress strip.
    """
    preview_ready = Signal(dict)
    done = Signal(dict)
    error = Signal(str)

    def __init__(self, book: dict, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("BookDetailsLoader")
        self._book = book
        self._config = config or {}
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to library_core.BookDetailsLoaderMixin (inherited).


# _CHAPTER_PRIMARY_STYLES, _CHAPTER_BADGE_STYLES, _CHAPTER_BADGE_TEXT, _prepare_chapter_row_spec moved verbatim to library_core (imported above).


class _ChapterRowPrepThread(QThread):
    """Prepare chapter-row display specs away from the Qt UI thread."""
    batch_ready = Signal(int, object, bool)

    def __init__(self, generation: int, infos, show_raw_title: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("ChapterRowPrepThread")
        self._generation = int(generation)
        self._infos = [dict(info or {}) for info in (infos or [])]
        self._show_raw_title = bool(show_raw_title)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def run(self):
        batch: list[dict] = []
        prepared_count = 0
        try:
            for info in self._infos:
                if self._cancelled or self.isInterruptionRequested():
                    return
                batch.append(_prepare_chapter_row_spec(
                    info, self._show_raw_title))
                prepared_count += 1
                if len(batch) >= 80:
                    self.batch_ready.emit(self._generation, batch, False)
                    batch = []
            if not (self._cancelled or self.isInterruptionRequested()):
                self.batch_ready.emit(self._generation, batch, True)
        except Exception:
            logger.debug("Chapter row prep failed: %s", traceback.format_exc())
            if not (self._cancelled or self.isInterruptionRequested()):
                fallback = [
                    _prepare_chapter_row_spec(info, self._show_raw_title)
                    for info in self._infos[prepared_count:]
                ]
                self.batch_ready.emit(self._generation, fallback, True)


class _ChapterVirtualList(QWidget):
    """Paint a large chapter list without creating one QWidget per row."""

    clicked = Signal(int)
    activated = Signal(int)
    # Right-click → (chapter index, global QPoint); menu built by the dialog.
    menu_requested = Signal(int, object)

    _ROW_HEIGHT = 66
    _ROW_GAP = 6
    _SIDE_MARGIN = 4
    _BADGE_PALETTE = {
        "completed": (QColor(126, 200, 126), QColor(126, 200, 126, 31),
                      QColor(126, 200, 126)),
        "failed": (QColor(255, 158, 109), QColor(255, 158, 109, 31),
                   QColor(255, 158, 109)),
        "qa_failed": (QColor(255, 158, 109), QColor(255, 158, 109, 31),
                      QColor(255, 158, 109)),
        "in_progress": (QColor(255, 209, 102), QColor(255, 209, 102, 31),
                        QColor(255, 209, 102)),
        "pending": (QColor("#7a8599"), QColor("#2a2a3e"),
                    QColor("#3a3a5e")),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_specs: list[dict] = []
        self._reserved_height = 0
        self._selected_idx: int | None = None
        self._hover_row = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def count(self) -> int:
        return len(self._row_specs)

    def clear(self) -> None:
        self._row_specs = []
        self._hover_row = -1
        self._selected_idx = None
        self._sync_height()
        self.update()

    def append_specs(self, specs, selected_idx: int | None = None) -> None:
        if selected_idx is not None:
            self._selected_idx = selected_idx
        if specs:
            self._row_specs.extend(list(specs))
        self._sync_height()
        self.update()

    def set_selected_index(self, idx: int | None) -> None:
        self._selected_idx = idx
        self.update()

    def set_reserved_height(self, height: int) -> None:
        """Reserve blank result space without stretching the details hero."""
        height = max(0, int(height or 0))
        if height == self._reserved_height:
            return
        self._reserved_height = height
        self._sync_height()

    def reserved_height(self) -> int:
        return self._reserved_height

    def _sync_height(self) -> None:
        height = max(
            len(self._row_specs) * self._ROW_HEIGHT,
            self._reserved_height,
        )
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        return QSize(
            640,
            max(
                len(self._row_specs) * self._ROW_HEIGHT,
                self._reserved_height,
            ),
        )

    def _row_at(self, y: int) -> int:
        row = int(y // self._ROW_HEIGHT)
        if 0 <= row < len(self._row_specs):
            return row
        return -1

    def _chapter_index_for_row(self, row: int) -> int:
        if not (0 <= row < len(self._row_specs)):
            return 0
        info = self._row_specs[row].get("info", {}) or {}
        return int(info.get("index", 0) or 0)

    def mouseMoveEvent(self, event):
        row = self._row_at(int(event.position().y()))
        if row != self._hover_row:
            old = self._hover_row
            self._hover_row = row
            for candidate in (old, row):
                if candidate >= 0:
                    self.update(0, candidate * self._ROW_HEIGHT,
                                self.width(), self._ROW_HEIGHT)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        old = self._hover_row
        self._hover_row = -1
        if old >= 0:
            self.update(0, old * self._ROW_HEIGHT,
                        self.width(), self._ROW_HEIGHT)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            row = self._row_at(int(event.position().y()))
            if row >= 0:
                idx = self._chapter_index_for_row(row)
                self._selected_idx = idx
                self.clicked.emit(idx)
                self.update()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            row = self._row_at(int(event.position().y()))
            if row >= 0:
                self.activated.emit(self._chapter_index_for_row(row))
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        row = self._row_at(int(event.pos().y()))
        if row >= 0:
            idx = self._chapter_index_for_row(row)
            # Focus the row under the cursor so the menu visibly applies to it.
            self._selected_idx = idx
            self.clicked.emit(idx)
            self.update()
            self.menu_requested.emit(idx, event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        visible = event.rect()
        start = max(0, int(visible.top() // self._ROW_HEIGHT) - 1)
        end = min(
            len(self._row_specs),
            int(visible.bottom() // self._ROW_HEIGHT) + 2,
        )
        title_font = QFont(self.font())
        title_font.setPointSizeF(10.0)
        meta_font = QFont(self.font())
        meta_font.setPointSizeF(8.5)
        badge_font = QFont(self.font())
        badge_font.setPointSizeF(8.0)
        badge_font.setBold(True)
        row_width = max(10, self.width() - (self._SIDE_MARGIN * 2))

        for row in range(start, end):
            spec = self._row_specs[row]
            info = spec.get("info", {}) or {}
            chapter_idx = int(info.get("index", 0) or 0)
            selected = chapter_idx == self._selected_idx
            hovered = row == self._hover_row

            y = row * self._ROW_HEIGHT + (self._ROW_GAP // 2)
            rect = QRect(
                self._SIDE_MARGIN,
                y,
                row_width,
                self._ROW_HEIGHT - self._ROW_GAP,
            )
            bg = QColor("#2a2d5a" if selected else
                        "#232340" if hovered else "#1a1a2a")
            border = QColor("#a097ff" if selected else
                            "#6c63ff" if hovered else "#242438")
            painter.setPen(QPen(border, 2))
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, 6, 6)

            text_rect = rect.adjusted(18, 10, -18, -8)
            badge = str(spec.get("badge_text", "") or "")
            badge_rect = QRect()
            if badge:
                status = str(info.get("status", "") or "")
                palette = self._BADGE_PALETTE.get(status)
                if palette is None and status == "qa_failed":
                    palette = self._BADGE_PALETTE.get("failed")
                if palette is None:
                    palette = self._BADGE_PALETTE.get("pending")
                badge_metrics = QFontMetrics(badge_font)
                badge_w = min(
                    max(74, badge_metrics.horizontalAdvance(badge) + 22),
                    max(74, rect.width() // 3),
                )
                badge_h = 24
                badge_rect = QRect(
                    rect.right() - badge_w - 14,
                    rect.center().y() - (badge_h // 2),
                    badge_w,
                    badge_h,
                )
                text_rect.setRight(max(text_rect.left() + 40,
                                       badge_rect.left() - 14))

            painter.setFont(title_font)
            title_color = "#ffffff" if selected else (
                "#e0e0e0" if spec.get("primary_class") == "translated"
                else "#c8cbe0"
            )
            painter.setPen(QColor(title_color))
            title = str(spec.get("primary_text", "") or "")
            title_metrics = QFontMetrics(title_font)
            painter.drawText(
                text_rect,
                Qt.AlignLeft | Qt.AlignTop,
                title_metrics.elidedText(title, Qt.ElideRight,
                                         max(20, text_rect.width())),
            )

            filename = str(spec.get("filename", "") or "")
            meta = filename
            if meta:
                painter.setFont(meta_font)
                painter.setPen(QColor("#e5e7ff" if selected else "#9aa2b8"))
                meta_metrics = QFontMetrics(meta_font)
                painter.drawText(
                    text_rect.adjusted(0, 28, 0, 0),
                    Qt.AlignLeft | Qt.AlignTop,
                    meta_metrics.elidedText(meta, Qt.ElideRight,
                                            max(20, text_rect.width())),
                )
            if badge and not badge_rect.isNull():
                text_color, badge_bg, badge_border = palette
                painter.setFont(badge_font)
                painter.setPen(QPen(badge_border, 1))
                painter.setBrush(badge_bg)
                painter.drawRoundedRect(badge_rect, 10, 10)
                painter.setPen(text_color)
                painter.drawText(badge_rect, Qt.AlignCenter, badge)


# ---------------------------------------------------------------------------
# Book Details Dialog
# ---------------------------------------------------------------------------

# _EDITABLE_BOOK_METADATA_FIELDS, _metadata_subject_values, _merge_manual_metadata_edits moved verbatim to library_core (imported above).
# _metadata_changed_values was extracted from _BookMetadataEditDialog.changed_values and
# _MetadataEditError added for BookDetailsDialog._on_edit_metadata_clicked's error texts
# (library_core, imported above; see DISCREPANCIES U5 "Phase-1 splits").


class _BookMetadataEditDialog(QDialog):
    """Compact editor for the output workspace's metadata.json."""

    def __init__(self, values: dict, parent=None):
        super().__init__(parent)
        self._initial_values = dict(values or {})
        self.setWindowTitle("Edit EPUB Metadata")
        self.setModal(True)
        self.setMinimumWidth(620)
        self.setStyleSheet("""
            QDialog { background: #161622; color: #e0e0e0; }
            QLabel { color: #aeb3c7; font-size: 9pt; }
            QLineEdit, QPlainTextEdit {
                background: #202033; color: #f0f0f5;
                border: 1px solid #3a3a5e; border-radius: 5px;
                padding: 6px 8px; font-size: 9.5pt;
            }
            QLineEdit:focus, QPlainTextEdit:focus {
                border-color: #6c63ff;
            }
            QPushButton {
                background: #2a2a3e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 5px;
                padding: 6px 16px; font-weight: bold;
            }
            QPushButton:hover {
                background: #3a3a5e; border-color: #6c63ff;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(14)

        note = QLabel(
            "Changes are saved to the output workspace's metadata.json. "
            "The original EPUB is not modified."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        self._title_edit = QLineEdit(str(values.get("title") or ""))
        self._creator_edit = QLineEdit(str(values.get("creator") or ""))
        self._publisher_edit = QLineEdit(str(values.get("publisher") or ""))
        self._language_edit = QLineEdit(str(values.get("language") or ""))
        self._date_edit = QLineEdit(str(values.get("date") or ""))
        self._subject_edit = QPlainTextEdit(
            str(values.get("subject") or "")
        )
        self._subject_edit.setPlaceholderText(
            "Separate tags with commas or new lines"
        )
        self._subject_edit.setFixedHeight(72)
        self._description_edit = QPlainTextEdit(
            str(values.get("description") or "")
        )
        self._description_edit.setFixedHeight(150)

        form.addRow("Title", self._title_edit)
        form.addRow("Author", self._creator_edit)
        form.addRow("Publisher", self._publisher_edit)
        form.addRow("Language", self._language_edit)
        form.addRow("Date", self._date_edit)
        form.addRow("Tags", self._subject_edit)
        form.addRow("Synopsis", self._description_edit)
        outer.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def values(self) -> dict:
        return {
            "title": self._title_edit.text(),
            "creator": self._creator_edit.text(),
            "publisher": self._publisher_edit.text(),
            "language": self._language_edit.text(),
            "date": self._date_edit.text(),
            "subject": self._subject_edit.toPlainText(),
            "description": self._description_edit.toPlainText(),
        }

    def changed_values(self) -> dict:
        return _metadata_changed_values(self._initial_values, self.values())


class BookDetailsDialog(BookDetailsMixin, QDialog):
    """Web-like book page: cover, metadata, synopsis, and collapsible TOC.

    Clicking a chapter launches the EPUB reader positioned at that chapter.
    Opens instead of jumping straight into the reader when a library card is
    activated.
    """

    def __init__(self, book: dict, config: dict | None = None, parent=None):
        super().__init__(parent)
        self._book = book
        self._config = config or {}
        self._details: dict = {}
        self._chapters_info: list[dict] = []
        self._metadata_json: dict = {}
        self._loader: _BookDetailsLoader | None = None
        self._active_reader: QDialog | None = None
        # Gate chapter-row activations until Phase 2 has actually built the
        # list. Prevents stray double-clicks during the "Loading chapters…"
        # phase from being dispatched to a half-populated chapter list.
        self._chapters_loaded = False
        self._populate_generation = 0
        self._chapter_row_prep_threads: list[_ChapterRowPrepThread] = []
        self._chapter_row_prep_thread: _ChapterRowPrepThread | None = None
        self._chapter_page = 0
        self._chapter_list_defer_clear = False
        self._chapter_list_transition_pending = False
        self._show_qa_failures_only = False
        self._details_config_persist_timer: QTimer | None = None
        # Single-selected chapter row (by spine index). ``None`` means no
        # row currently has focus.
        self._selected_chapter_idx: int | None = None
        # Whether the "show special files" toggle is on. Its initial state is
        # coupled to the global "Translate Special Files" setting (Other
        # Settings): if the translator is configured to handle special-file
        # keywords, the dialog defaults to showing them too and counting
        # them in the progress percentage. The user can still override the
        # toggle per-dialog — the override is persisted in config.
        _translate_special = _resolve_translate_special_files(self._config)
        _stored_show = self._config.get("epub_details_show_special_files", None)
        if _stored_show is None:
            self._show_special_files = _translate_special
        else:
            # User has an explicit preference — honor it, but also respect
            # the translate-special toggle so turning it ON auto-propagates.
            self._show_special_files = bool(_stored_show) or _translate_special
        # "Show raw titles" overrides the default "translated when completed,
        # raw otherwise" rule used by :class:`_ChapterRow` so every row shows
        # the source-language title regardless of translation status. Only
        # meaningful for in-progress EPUBs (where translated titles exist);
        # the toggle is hidden otherwise.
        self._show_raw_titles = bool(
            self._config.get("epub_details_show_raw_titles", False)
        )

        epub_path = book["path"]
        pretty = book.get("name") or os.path.splitext(os.path.basename(epub_path))[0]
        self.setWindowTitle(pretty)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        screen = self.screen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.62), int(avail.height() * 0.84))
            self.setMinimumSize(int(avail.width() * 0.42), int(avail.height() * 0.5))
        else:
            self.resize(1100, 830)
            self.setMinimumSize(700, 500)

        icon_path = _find_halgakos_icon()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self._setup_ui()
        self._start_loading()
        # Auto-refresh details every 2s so chapter status + progress
        # strip catch up with the translator as it writes new
        # ``response_*.html`` files. The timer pauses while a loader
        # is already running or the dialog is hidden — see
        # :meth:`_auto_refresh_details`.
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self._auto_refresh_details)
        QTimer.singleShot(2500, self._auto_refresh_timer.start)

    # -- UI construction ----------------------------------------------------

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background: #12121e; }
            QLabel#title { color: #e0e0e0; font-size: 22pt; font-weight: bold; }
            QLabel#author { color: #9aa2b8; font-size: 11pt; }
            QLabel#section { color: #b0b0c0; font-size: 10pt; font-weight: bold;
                             letter-spacing: 1px; }
            QLabel.meta-k { color: #7a8599; font-size: 8.5pt; }
            QLabel.meta-v { color: #e0e0e0; font-size: 9.5pt; font-weight: bold; }
            QLabel.tag { color: #c8cbe0; background: #2a2a3e;
                         border: 1px solid #3a3a5e; border-radius: 10px;
                         padding: 3px 9px; font-size: 8.5pt; }
            QLabel.pending { color: #7a8599; font-size: 8pt; font-style: italic; }
            QLabel.filename { color: #8a8fa8; font-size: 8.5pt; font-family: 'Consolas','Menlo',monospace; }
            QLabel.translated { color: #e0e0e0; font-size: 10pt; font-weight: bold; }
            QLabel.raw { color: #c8cbe0; font-size: 10pt; font-weight: bold; }
            QPushButton#start {
                background: #ff8a3d; color: #1e1616; font-weight: bold;
                font-size: 11pt; padding: 8px 18px; border-radius: 6px; border: none;
            }
            QPushButton#start:hover { background: #ffa05c; }
            QPushButton.icon-btn {
                background: #2a2a3e; color: #e0e0e0; border: 1px solid #3a3a5e;
                border-radius: 6px; padding: 6px 10px; font-size: 10pt;
            }
            QPushButton.icon-btn:hover { background: #3a3a5e; }
            QPushButton.icon-btn:disabled {
                background: #16161f; color: #2f2f3a;
                border: 1px dashed #2a2a3e;
            }
            QPushButton#toc-toggle {
                background: transparent; color: #b0b0c0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 6px 12px; font-size: 10pt; font-weight: bold;
                text-align: left;
            }
            QPushButton#toc-toggle:hover { color: #e0e0e0; border-color: #6c63ff; }
            QPushButton#toc-title {
                color: #e0e0e0; background: #17172a;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 6px 12px; font-size: 10pt; font-weight: bold;
                text-align: left;
            }
            QPushButton#toc-title:hover {
                background: #20203a; border-color: #6c63ff;
            }
            QPushButton#toc-title:checked {
                background: #2a2d5a; border-color: #a097ff;
            }
            QLabel#toc-page-label { color: #9aa2b8; font-size: 9pt; }
            QPushButton#toc-page-first, QPushButton#toc-page-prev,
            QPushButton#toc-page-next, QPushButton#toc-page-last {
                background: #202036; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 0px 10px; font-size: 11pt; font-weight: bold;
                min-width: 28px;
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton#toc-page-first:hover, QPushButton#toc-page-prev:hover,
            QPushButton#toc-page-next:hover, QPushButton#toc-page-last:hover {
                border-color: #6c63ff; background: #282848;
            }
            QPushButton#toc-page-first:disabled, QPushButton#toc-page-prev:disabled,
            QPushButton#toc-page-next:disabled, QPushButton#toc-page-last:disabled {
                color: #555a70; background: #171724; border-color: #28283d;
            }
            QComboBox#toc-page-size {
                background: #1e1e2e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 5px 9px; font-size: 9pt;
            }
            QLineEdit#toc-search {
                background: #1e1e2e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 6px 10px; font-size: 9.5pt;
            }
            QScrollArea { border: none; background: transparent; }
            QListWidget#chapterList {
                background: transparent;
                border: none;
                outline: none;
                padding: 0px;
            }
            QListWidget#chapterList::item {
                background: #1a1a2a;
                color: #c8cbe0;
                border: 2px solid #242438;
                border-radius: 6px;
                margin: 2px 4px;
                padding: 8px 12px;
            }
            QListWidget#chapterList::item:hover {
                background: #232340;
                border-color: #6c63ff;
            }
            QListWidget#chapterList::item:selected {
                background: #2a2d5a;
                border-color: #a097ff;
                color: #ffffff;
            }
            QScrollBar:vertical { width: 10px; background: #12121e; }
            QScrollBar::handle:vertical { background: #3a3a5e; border-radius: 5px;
                                          min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar with a back / close button
        topbar = QHBoxLayout()
        topbar.setContentsMargins(14, 10, 14, 6)
        topbar.setSpacing(8)
        back_btn = QPushButton("\u2190  Back to Library")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8a8fa8; border: none;"
            " font-size: 9.5pt; padding: 4px 6px; }"
            "QPushButton:hover { color: #e0e0e0; }"
        )
        back_btn.clicked.connect(self.close)
        topbar.addWidget(back_btn)
        topbar.addStretch()
        outer.addLayout(topbar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        body = QWidget()
        self._scroll.setWidget(body)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(32, 14, 48, 32)
        body_layout.setSpacing(18)

        # ── Hero row: cover + title + actions + metadata ──
        hero = QHBoxLayout()
        hero.setSpacing(28)

        # Cover
        self._cover_lbl = QLabel()
        self._cover_lbl.setFixedSize(240, 340)
        self._cover_lbl.setAlignment(Qt.AlignCenter)
        self._cover_lbl.setStyleSheet(
            "background: #2a2a3e; border-radius: 6px; color: #555; font-size: 32pt;"
        )
        # Use the already-resolved card thumbnail when the library has one.
        # This prevents the details page from flashing the fallback before the
        # background metadata loader emits its preview payload.
        self._current_cover_path = ""
        self._cover_movie = None
        if not self._apply_cover_path(self._book.get("_cached_cover_path", "")):
            self._apply_halgakos_fallback()
        hero.addWidget(self._cover_lbl, 0, Qt.AlignTop)

        # Center column: title, author, actions, synopsis
        center = QVBoxLayout()
        center.setSpacing(8)
        self._title_lbl = QLabel(self._book.get("name", ""))
        self._title_lbl.setObjectName("title")
        self._title_lbl.setWordWrap(True)
        center.addWidget(self._title_lbl)
        self._author_lbl = QLabel("")
        self._author_lbl.setObjectName("author")
        self._author_lbl.setWordWrap(True)
        center.addWidget(self._author_lbl)

        # Progress strip (only for in-progress novels)
        self._progress_strip = QLabel("")
        self._progress_strip.setStyleSheet(
            "color: #ffd166; background: rgba(108, 99, 255, 0.14);"
            " border: 1px solid #6c63ff; border-radius: 6px;"
            " padding: 6px 10px; font-size: 9.5pt; font-weight: bold;"
        )
        self._progress_strip.hide()
        center.addWidget(self._progress_strip)

        # Action row
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._start_btn = QPushButton("\U0001f4d6  Start reading")
        self._start_btn.setObjectName("start")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.clicked.connect(lambda: self._open_reader())
        actions.addWidget(self._start_btn)

        # Secondary action (only shown for in-progress novels) to open the
        # untouched source EPUB without the translated overlay.
        self._raw_btn = QPushButton("\U0001f4dc  Read raw source")
        self._raw_btn.setProperty("class", "icon-btn")
        self._raw_btn.setCursor(Qt.PointingHandCursor)
        self._raw_btn.setToolTip("Open the source EPUB without the translated overlay")
        self._raw_btn.clicked.connect(lambda: self._open_reader(raw_only=True))
        self._raw_btn.hide()
        actions.addWidget(self._raw_btn)

        # Icon buttons carry a QGraphicsOpacityEffect so the *emoji itself*
        # dims when the button is disabled — the stylesheet `color:` rule has
        # no effect on emoji glyphs, which render from their own color table.
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._folder_btn = QPushButton("\U0001f4c2")
        self._folder_btn.setProperty("class", "icon-btn")
        self._folder_btn.setToolTip("Open output folder in file explorer")
        self._folder_btn.setCursor(Qt.PointingHandCursor)
        self._folder_btn.clicked.connect(self._open_output_folder)
        self._folder_btn_opacity = QGraphicsOpacityEffect(self._folder_btn)
        self._folder_btn_opacity.setOpacity(1.0)
        self._folder_btn.setGraphicsEffect(self._folder_btn_opacity)
        actions.addWidget(self._folder_btn)

        self._source_btn = QPushButton("\U0001f517")
        self._source_btn.setProperty("class", "icon-btn")
        self._source_btn.setToolTip("Reveal source file")
        self._source_btn.setCursor(Qt.PointingHandCursor)
        self._source_btn.clicked.connect(self._reveal_source)
        self._source_btn_opacity = QGraphicsOpacityEffect(self._source_btn)
        self._source_btn_opacity.setOpacity(1.0)
        self._source_btn.setGraphicsEffect(self._source_btn_opacity)
        actions.addWidget(self._source_btn)

        # "Reveal Translated File" — matches the card context menu's
        # equivalent action. Uses :func:`_resolve_book_translated_file`
        # so this button and the menu item resolve the same target.
        self._translated_btn = QPushButton("\U0001f4d5")
        self._translated_btn.setProperty("class", "icon-btn")
        self._translated_btn.setToolTip("Reveal translated file")
        self._translated_btn.setCursor(Qt.PointingHandCursor)
        self._translated_btn.clicked.connect(self._reveal_translated)
        self._translated_btn_opacity = QGraphicsOpacityEffect(
            self._translated_btn)
        self._translated_btn_opacity.setOpacity(1.0)
        self._translated_btn.setGraphicsEffect(
            self._translated_btn_opacity)
        actions.addWidget(self._translated_btn)

        self._translate_metadata_btn = QPushButton(
            "\U0001f310  Translate Metadata"
        )
        self._translate_metadata_btn.setCursor(Qt.PointingHandCursor)
        self._translate_metadata_btn.setMaximumWidth(145)
        self._translate_metadata_btn.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self._translate_metadata_btn.setToolTip(
            "Run this EPUB's configured title/metadata translation phase only."
        )
        self._translate_metadata_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 5px;
                padding: 4px 8px; font-size: 8.25pt; font-weight: bold; }
            QPushButton:hover { background: #3a3a5e; border-color: #6c63ff; }
        """)
        self._translate_metadata_btn.clicked.connect(
            self._on_translate_metadata_clicked
        )
        try:
            _metadata_source = _resolve_book_metadata_source(self._book)
            self._translate_metadata_btn.setVisible(bool(_metadata_source))
        except Exception:
            self._translate_metadata_btn.setVisible(False)
        actions.addWidget(self._translate_metadata_btn)

        actions.addStretch()
        center.addLayout(actions)

        # Synopsis block
        synopsis_heading = QLabel("SYNOPSIS")
        synopsis_heading.setObjectName("section")
        center.addSpacing(6)
        center.addWidget(synopsis_heading)
        self._synopsis_lbl = QLabel("Loading\u2026")
        self._synopsis_lbl.setWordWrap(True)
        self._synopsis_lbl.setStyleSheet("color: #c8cbe0; font-size: 10pt; line-height: 1.55;")
        center.addWidget(self._synopsis_lbl)
        center.addStretch()
        hero.addLayout(center, 1)

        # Metadata column
        meta_col = QVBoxLayout()
        meta_col.setSpacing(14)
        meta_header = QHBoxLayout()
        meta_header.setSpacing(8)
        meta_heading = QLabel("METADATA")
        meta_heading.setObjectName("section")
        meta_header.addWidget(meta_heading)
        meta_header.addStretch()
        self._edit_metadata_btn = QPushButton("\u270f\ufe0f  Edit")
        self._edit_metadata_btn.setCursor(Qt.PointingHandCursor)
        self._edit_metadata_btn.setToolTip(
            "Edit the output workspace's metadata.json"
        )
        self._edit_metadata_btn.setStyleSheet("""
            QPushButton {
                background: #242438; color: #c8cbe0;
                border: 1px solid #3a3a5e; border-radius: 5px;
                padding: 3px 8px; font-size: 8pt; font-weight: bold;
            }
            QPushButton:hover {
                background: #343454; border-color: #6c63ff;
            }
            QPushButton:disabled {
                background: #181824; color: #555a70;
                border-color: #28283d;
            }
        """)
        self._edit_metadata_btn.clicked.connect(
            self._on_edit_metadata_clicked
        )
        meta_header.addWidget(self._edit_metadata_btn)
        meta_col.addLayout(meta_header)
        self._meta_grid = QGridLayout()
        self._meta_grid.setContentsMargins(0, 0, 0, 0)
        self._meta_grid.setHorizontalSpacing(14)
        # Tight uniform vertical spacing — only the Title row gets extra
        # breathing room, and that’s provided by :func:`_fit_title_text`
        # below (dynamically shrinking the Title font instead of padding
        # every row with 36 px of whitespace).
        self._meta_grid.setVerticalSpacing(8)
        # Let the second column stretch so long titles wrap across more
        # horizontal space instead of clipping vertically.
        self._meta_grid.setColumnStretch(0, 0)
        self._meta_grid.setColumnStretch(1, 1)
        meta_col.addLayout(self._meta_grid)

        # Genres / Tags containers (filled in populate step)
        self._genres_heading = QLabel("GENRES")
        self._genres_heading.setObjectName("section")
        meta_col.addWidget(self._genres_heading)
        self._genres_row = QWidget()
        self._genres_layout = QHBoxLayout(self._genres_row)
        self._genres_layout.setContentsMargins(0, 0, 0, 0)
        self._genres_layout.setSpacing(6)
        self._genres_layout.addStretch()
        meta_col.addWidget(self._genres_row)

        self._tags_heading = QLabel("TAGS")
        self._tags_heading.setObjectName("section")
        meta_col.addWidget(self._tags_heading)
        self._tags_row = QWidget()
        self._tags_row.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self._tags_layout = _FlowLayout(
            self._tags_row, horizontal_spacing=8, vertical_spacing=8
        )
        meta_col.addWidget(self._tags_row)
        meta_col.addStretch()

        meta_wrapper = QWidget()
        meta_wrapper.setLayout(meta_col)
        # Wider column so long titles / author names don't wrap aggressively
        # and collide with the next metadata row. Bumped further so the
        # common CJK novel title fits on one line in the Title row.
        meta_wrapper.setMinimumWidth(400)
        meta_wrapper.setMaximumWidth(700)
        meta_wrapper.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self._meta_wrapper = meta_wrapper
        hero.addWidget(meta_wrapper, 1, Qt.AlignTop)
        body_layout.addLayout(hero)

        # ── Chapters section ──
        chap_header = QHBoxLayout()
        chap_header.setSpacing(10)
        self._toc_title = QPushButton("Chapters")
        self._toc_title.setObjectName("toc-title")
        self._toc_title.setCursor(Qt.PointingHandCursor)
        self._toc_title.setCheckable(True)
        self._toc_title.setChecked(False)
        self._toc_title.setToolTip("Show QA failures only")
        self._toc_title.toggled.connect(self._on_qa_failures_only_toggled)
        # Keep the old attribute name for helpers that only need setText().
        self._toc_toggle = self._toc_title
        chap_header.addWidget(self._toc_title)

        # "Compile EPUB" — explicitly runs the EPUB converter phase on this
        # book's output workspace. Single-chapter Translate actions (context
        # menu / reader) intentionally never compile, so this is the way to
        # build the output EPUB when you're ready.
        try:
            _out_folder = _resolve_book_output_folder(self._book)
        except Exception:
            _out_folder = ""
        self._compile_output_kind = _workspace_compile_kind(
            self._book, _out_folder)
        compile_text = (
            "\U0001f4c4  Compile PDF"
            if self._compile_output_kind == "pdf"
            else "\U0001f4d8  Compile EPUB"
        )
        self._compile_epub_btn = QPushButton(compile_text)
        self._compile_epub_btn.setCursor(Qt.PointingHandCursor)
        self._compile_epub_btn.setToolTip(
            "Build the output EPUB from this book's translated chapters.\n"
            "Single-chapter Translate runs skip the converter phase — use\n"
            "this button when you want the compiled EPUB.")
        if self._compile_output_kind == "pdf":
            self._compile_epub_btn.setToolTip(
                "Build the output PDF from this book's translated sections.")
        self._compile_epub_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 6px 14px; font-size: 9.5pt; font-weight: bold; }
            QPushButton:hover { background: #3a3a5e; border-color: #6c63ff; }
        """)
        self._compile_epub_btn.clicked.connect(self._on_compile_epub_clicked)
        try:
            self._compile_epub_btn.setVisible(
                bool(_out_folder and os.path.isdir(_out_folder)))
        except Exception:
            pass
        chap_header.addWidget(self._compile_epub_btn)

        chap_header.addStretch()

        self._toc_first_btn = QPushButton("\u00ab")
        self._toc_first_btn.setObjectName("toc-page-first")
        self._toc_first_btn.setCursor(Qt.PointingHandCursor)
        self._toc_first_btn.setToolTip("First chapter page")
        self._toc_first_btn.clicked.connect(self._on_chapter_first_page)
        chap_header.addWidget(self._toc_first_btn)

        self._toc_prev_btn = QPushButton("\u2039")
        self._toc_prev_btn.setObjectName("toc-page-prev")
        self._toc_prev_btn.setCursor(Qt.PointingHandCursor)
        self._toc_prev_btn.setToolTip("Previous chapter page")
        self._toc_prev_btn.clicked.connect(self._on_chapter_prev_page)
        chap_header.addWidget(self._toc_prev_btn)

        self._toc_page_label = QLabel("Page 1 / 1")
        self._toc_page_label.setObjectName("toc-page-label")
        chap_header.addWidget(self._toc_page_label)

        self._toc_next_btn = QPushButton("\u203a")
        self._toc_next_btn.setObjectName("toc-page-next")
        self._toc_next_btn.setCursor(Qt.PointingHandCursor)
        self._toc_next_btn.setToolTip("Next chapter page")
        self._toc_next_btn.clicked.connect(self._on_chapter_next_page)
        chap_header.addWidget(self._toc_next_btn)

        self._toc_last_btn = QPushButton("\u00bb")
        self._toc_last_btn.setObjectName("toc-page-last")
        self._toc_last_btn.setCursor(Qt.PointingHandCursor)
        self._toc_last_btn.setToolTip("Last chapter page")
        self._toc_last_btn.clicked.connect(self._on_chapter_last_page)
        chap_header.addWidget(self._toc_last_btn)

        self._toc_page_size_combo = QComboBox()
        self._toc_page_size_combo.setObjectName("toc-page-size")
        self._toc_page_size_combo.setToolTip("Chapter rows per page")
        for label, value in (
            ("20 / page", 20),
            ("50 / page", 50),
            ("100 / page", 100),
            ("250 / page", 250),
            ("500 / page", 500),
            ("All", "all"),
        ):
            self._toc_page_size_combo.addItem(label, value)
        stored_page_size = self._config.get("epub_details_chapter_page_size", 20)
        if str(stored_page_size).strip().lower() == "all":
            page_size_index = self._toc_page_size_combo.findData("all")
        else:
            try:
                page_size_index = self._toc_page_size_combo.findData(
                    int(stored_page_size))
            except (TypeError, ValueError):
                page_size_index = 0
        self._toc_page_size_combo.setCurrentIndex(
            page_size_index if page_size_index >= 0 else 0)
        self._toc_page_size_combo.currentIndexChanged.connect(
            self._on_chapter_page_size_changed)
        self._style_chapter_page_size_combo()
        chap_header.addWidget(self._toc_page_size_combo)

        # "Show special files" toggle mirrors the Progress Manager's behavior.
        # Hidden by default for EPUBs so configured files like nav/toc/title
        # don't clutter the list; user state is persisted via config. Built via
        # :func:`_create_styled_checkbox` so the Book Details header reads
        # as the same visual vocabulary as Other Settings (identical
        # indicator border / fill / hover colours + “✓” overlay).
        self._special_cb = _create_styled_checkbox(
            "Show skipped files")
        self._special_cb.setToolTip(
            "When enabled, shows files that would be skipped during translation\n"
            "(matching the special-file keywords configured in Other Settings)."
        )
        self._special_cb.setChecked(self._show_special_files)
        self._special_cb.toggled.connect(self._on_special_files_toggled)
        chap_header.addWidget(self._special_cb)
        # "Show raw titles" toggle: when checked, every chapter row displays
        # the source-language title instead of the translated one (if any).
        # Only useful for in-progress EPUBs where translated titles exist —
        # hidden otherwise via :meth:`_on_details_ready`. Shares the Other
        # Settings styling via the same helper as ``_special_cb``.
        self._raw_titles_cb = _create_styled_checkbox("Show raw titles")
        self._raw_titles_cb.setToolTip(
            "Show the source-language chapter titles instead of the "
            "translated titles, even for chapters that have already been "
            "translated."
        )
        self._raw_titles_cb.setChecked(self._show_raw_titles)
        self._raw_titles_cb.toggled.connect(self._on_raw_titles_toggled)
        # Hidden by default; the details-ready handler flips it on for
        # in-progress EPUBs that actually have translated chapters to
        # distinguish from.
        self._raw_titles_cb.hide()
        chap_header.addWidget(self._raw_titles_cb)
        self._toc_search = QLineEdit()
        self._toc_search.setObjectName("toc-search")
        self._toc_search.setPlaceholderText("\U0001f50d  Search chapters\u2026")
        self._toc_search.setFixedWidth(260)
        self._toc_search.textChanged.connect(self._apply_chapter_filter)
        chap_header.addWidget(self._toc_search)
        body_layout.addLayout(chap_header)

        self._chap_container = QWidget()
        self._chap_layout = QVBoxLayout(self._chap_container)
        self._chap_layout.setContentsMargins(4, 4, 4, 4)
        self._chap_layout.setSpacing(4)
        # Visible by default — the TOC is the main reading entry point.
        body_layout.addWidget(self._chap_container)

        self._chap_list = _ChapterVirtualList()
        self._chap_list.setObjectName("chapterList")
        self._chap_list.clicked.connect(self._on_chapter_virtual_clicked)
        self._chap_list.activated.connect(self._on_chapter_activated)
        self._chap_list.menu_requested.connect(self._show_chapter_context_menu)
        self._chap_list_opacity = QGraphicsOpacityEffect(self._chap_list)
        self._chap_list_opacity.setOpacity(1.0)
        self._chap_list.setGraphicsEffect(self._chap_list_opacity)
        self._chap_list_anim = None
        self._chap_list.hide()
        body_layout.addWidget(self._chap_list)

        self._toc_bottom_pager = QWidget()
        self._toc_bottom_pager.setObjectName("toc-bottom-pager")
        bottom_pager_layout = QHBoxLayout(self._toc_bottom_pager)
        bottom_pager_layout.setContentsMargins(4, 4, 4, 4)
        bottom_pager_layout.setSpacing(10)
        bottom_pager_layout.addStretch()

        self._toc_bottom_first_btn = QPushButton("\u00ab")
        self._toc_bottom_first_btn.setObjectName("toc-page-first")
        self._toc_bottom_first_btn.setCursor(Qt.PointingHandCursor)
        self._toc_bottom_first_btn.setToolTip("First chapter page")
        self._toc_bottom_first_btn.clicked.connect(self._on_chapter_first_page)
        bottom_pager_layout.addWidget(self._toc_bottom_first_btn)

        self._toc_bottom_prev_btn = QPushButton("\u2039")
        self._toc_bottom_prev_btn.setObjectName("toc-page-prev")
        self._toc_bottom_prev_btn.setCursor(Qt.PointingHandCursor)
        self._toc_bottom_prev_btn.setToolTip("Previous chapter page")
        self._toc_bottom_prev_btn.clicked.connect(self._on_chapter_prev_page)
        bottom_pager_layout.addWidget(self._toc_bottom_prev_btn)

        self._toc_bottom_page_label = QLabel("Page 1 / 1")
        self._toc_bottom_page_label.setObjectName("toc-page-label")
        bottom_pager_layout.addWidget(self._toc_bottom_page_label)

        self._toc_bottom_next_btn = QPushButton("\u203a")
        self._toc_bottom_next_btn.setObjectName("toc-page-next")
        self._toc_bottom_next_btn.setCursor(Qt.PointingHandCursor)
        self._toc_bottom_next_btn.setToolTip("Next chapter page")
        self._toc_bottom_next_btn.clicked.connect(self._on_chapter_next_page)
        bottom_pager_layout.addWidget(self._toc_bottom_next_btn)

        self._toc_bottom_last_btn = QPushButton("\u00bb")
        self._toc_bottom_last_btn.setObjectName("toc-page-last")
        self._toc_bottom_last_btn.setCursor(Qt.PointingHandCursor)
        self._toc_bottom_last_btn.setToolTip("Last chapter page")
        self._toc_bottom_last_btn.clicked.connect(self._on_chapter_last_page)
        bottom_pager_layout.addWidget(self._toc_bottom_last_btn)

        bottom_pager_layout.addStretch()
        self._toc_bottom_pager.hide()
        body_layout.addWidget(self._toc_bottom_pager)

        # The chapter section is no longer collapsible; the placeholder can
        # temporarily hide the container while metadata is loading, so this
        # flag remains as the stable "intended visible" state for helpers.
        self._chap_section_expanded = True

        # Standalone "⏳  Loading chapters…" placeholder shown only while
        # the background details loader is still collecting chapter info.
        # Once rows are available, :meth:`_populate_chapters` hides this
        # label and streams the batched row widgets directly into view.
        self._chap_loading_lbl = QLabel("\u23f3  Loading chapters\u2026")
        self._chap_loading_lbl.setAlignment(Qt.AlignCenter)
        self._chap_loading_lbl.setMinimumHeight(60)
        self._chap_loading_lbl.setStyleSheet(
            "color: #7a8599; font-size: 10pt; padding: 28px 22px 24px 22px;"
        )
        self._chap_loading_lbl.hide()
        body_layout.addWidget(self._chap_loading_lbl)

        body_layout.addStretch()
        outer.addWidget(self._scroll, 1)

    # -- Data population ----------------------------------------------------

    def _start_loading(self):
        self._synopsis_lbl.setText("Loading book details\u2026")
        # Install a placeholder in the chapters section so the user sees
        # activity while the (slower) spine parse is running in the
        # background. Replaced in :meth:`_on_details_ready` with real rows.
        self._show_chapter_placeholder()
        self._is_auto_refreshing = False
        self._loader = _BookDetailsLoader(self._book, self._config, self)
        # ``preview_ready`` fires first with cover + metadata so the hero
        # paints immediately; ``done`` follows with the full chapter list.
        self._loader.preview_ready.connect(self._on_preview_ready)
        self._loader.done.connect(self._on_details_ready)
        self._loader.error.connect(self._on_details_error)
        self._loader.start()

    def _auto_refresh_details(self):
        """Re-run the details loader so per-chapter status catches up.

        Skipped when the dialog is hidden (no point paying for disk I/O
        when the user can't see it) or a loader is already mid-flight.
        The refresh runs a silent :meth:`_BookDetailsLoader` pass that
        does NOT show the loading placeholder — we reuse the existing
        hero + chapter rows and swap them in place only when a
        per-chapter status signature actually changed.
        """
        try:
            if not self.isVisible():
                return
        except Exception:
            pass
        if self._loader is not None:
            try:
                if self._loader.isRunning():
                    return
            except Exception:
                pass
        self._is_auto_refreshing = True
        self._loader = _BookDetailsLoader(self._book, self._config, self)
        # Only subscribe to ``done`` — ``preview_ready`` is only useful
        # for the initial paint and would re-flash the cover + synopsis
        # with every tick otherwise.
        self._loader.done.connect(self._on_details_ready)
        self._loader.error.connect(self._on_details_error)
        self._loader.start()

    def _show_chapter_placeholder(self):
        """Show the "⏳  Loading chapters…" placeholder while the chapter
        list isn't ready to display.

        Toggles the standalone :attr:`_chap_loading_lbl` sibling widget
        instead of stuffing a label inside ``_chap_layout``. This is only
        for the background metadata/spine load; once chapter row data is
        ready, rows stream into the real container in batches.
        """
        # Clear any stale rows (e.g. leftover from a previous open)
        # without resurrecting a per-layout placeholder label.
        while self._chap_layout.count():
            item = self._chap_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        # Hide the chapter container (which is currently empty) and show
        # the sibling placeholder in its slot.
        if getattr(self, "_chap_container", None) is not None:
            self._chap_container.hide()
        chap_list = getattr(self, "_chap_list", None)
        if chap_list is not None:
            self._chapter_list_defer_clear = False
            self._chapter_list_transition_pending = False
            self._stop_chapter_list_animation()
            self._set_chapter_list_opacity(1.0)
            chap_list.set_reserved_height(0)
            chap_list.clear()
            chap_list.hide()
        if getattr(self, "_toc_bottom_pager", None) is not None:
            self._toc_bottom_pager.hide()
        if getattr(self, "_chap_loading_lbl", None) is not None:
            self._chap_loading_lbl.show()

    def _apply_cover_path(self, cover_path: str) -> bool:
        """Paint a real cover path into the hero cover label."""
        cover_path = str(cover_path or "")
        if not cover_path:
            return False
        try:
            movie = _build_cover_movie(
                cover_path,
                self._cover_lbl,
                self._cover_lbl.width(),
                self._cover_lbl.height(),
            )
            if movie is not None:
                _dispose_cover_movie(self._cover_movie)
                self._cover_movie = movie
                self._cover_lbl.setText("")
                self._current_cover_path = cover_path
                movie.start()
                return True
            pm = QPixmap(cover_path)
            if pm.isNull():
                return False
            scaled = pm.scaled(
                self._cover_lbl.width(),
                self._cover_lbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            _dispose_cover_movie(self._cover_movie)
            self._cover_movie = None
            self._cover_lbl.setPixmap(scaled)
            self._cover_lbl.setText("")
            self._current_cover_path = cover_path
            return True
        except Exception:
            logger.debug("Cover pixmap load failed for %s: %s",
                         cover_path, traceback.format_exc())
            return False

    def _apply_halgakos_fallback(self):
        """Render the Halgakos brand icon into the cover label.

        Scaled to fit the full cover-label footprint so freshly imported
        Not Started cards (which have no extractable EPUB cover on disk
        yet) still render a visible thumbnail instead of an empty dark
        box. Falls back to a book emoji if the icon file fails to load.
        """
        icon_path = _find_halgakos_icon()
        applied = False
        if icon_path:
            try:
                pm = QPixmap(icon_path)
                if not pm.isNull():
                    # Fill the full label footprint (240×340) rather than a
                    # small 160×160 centered icon so the fallback doesn't
                    # read as "no thumbnail" against the dark background.
                    scaled = pm.scaled(
                        self._cover_lbl.width(),
                        self._cover_lbl.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    _dispose_cover_movie(self._cover_movie)
                    self._cover_movie = None
                    self._cover_lbl.setPixmap(scaled)
                    self._cover_lbl.setText("")
                    applied = True
            except Exception:
                logger.debug("Halgakos pixmap failed: %s",
                             traceback.format_exc())
        if not applied:
            _dispose_cover_movie(self._cover_movie)
            self._cover_movie = None
            self._cover_lbl.setPixmap(QPixmap())
            self._cover_lbl.setText("\U0001f4d6")

    def _apply_hero_payload(self, payload: dict):
        """Paint cover + title + author + synopsis + metadata grid.

        Shared between :meth:`_on_preview_ready` (fast pass, no chapter
        data yet) and :meth:`_on_details_ready` (final pass). Safe to
        call repeatedly — the grid is cleared + rebuilt each invocation.
        """
        self._details = payload.get("details", self._details) or self._details
        self._metadata_json = payload.get("metadata_json", self._metadata_json) or self._metadata_json
        cover_path = payload.get("cover", "")
        cover_applied = self._apply_cover_path(cover_path)
        if not cover_applied:
            cover_applied = self._apply_cover_path(
                self._book.get("_cached_cover_path", ""))
        if not cover_applied:
            # No real cover could be extracted — keep the Halgakos branding
            # in place so the details dialog never shows an empty box.
            self._apply_halgakos_fallback()

        title = (self._metadata_json.get("title") or self._details.get("title")
                 or self._book.get("name", ""))
        self._title_lbl.setText(title or self._book.get("name", ""))
        self.setWindowTitle(title or self._book.get("name", ""))
        authors = self._metadata_author_values()
        self._author_lbl.setText(", ".join(authors) if authors else "")

        synopsis = (self._metadata_json.get("description")
                    or self._details.get("description") or "").strip()
        if synopsis:
            # Collapse any run of blank lines (i.e. a newline followed by
            # one or more whitespace-only lines) down to a single newline
            # so the synopsis renders compactly instead of with huge gaps
            # between sentences that happen to be paragraph-separated in
            # the source metadata.
            import re as _re
            synopsis = _re.sub(r"\n\s*\n+", "\n", synopsis)
        self._synopsis_lbl.setText(synopsis if synopsis else "No synopsis available.")

        # Metadata grid
        while self._meta_grid.count():
            item = self._meta_grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        publisher = self._metadata_json.get("publisher") or self._details.get("publisher") or "\u2014"
        language = self._metadata_json.get("language") or self._details.get("language") or "\u2014"
        year = (self._metadata_json.get("date") or self._details.get("date") or "").strip()
        year = year[:4] if year and year[:4].isdigit() else (year or "\u2014")
        rows = [
            ("\U0001f4d8 Title", title or "\u2014"),
            ("\u270d\ufe0f Author", ", ".join(authors) if authors else "\u2014"),
            ("\U0001f3db\ufe0f Publisher", publisher or "\u2014"),
            ("\U0001f310 Language", language or "\u2014"),
            ("\U0001f4c5 Year", year or "\u2014"),
        ]
        # Estimate the horizontal space available to the value column.
        # ``meta_wrapper`` is between 400–700 px wide; the key column is
        # ~100 px ("\U0001f3db\ufe0f Publisher" is the widest) and the grid
        # has 14 px horizontal spacing. If the wrapper has already been laid
        # out we use its real width; otherwise fall back to the conservative
        # minimum so shrink-to-fit kicks in even on first paint.
        wrapper_w = self._meta_wrapper.width() if getattr(self, "_meta_wrapper", None) else 0
        if wrapper_w < 300:
            wrapper_w = 400
        val_avail_w = max(200, wrapper_w - 100 - 14 - 4)
        for i, (key, val) in enumerate(rows):
            k_lbl = QLabel(key)
            k_lbl.setProperty("class", "meta-k")
            k_lbl.setStyleSheet("color: #7a8599; font-size: 8.5pt;")
            v_lbl = QLabel()
            v_lbl.setProperty("class", "meta-v")
            v_lbl.setWordWrap(True)
            if i == 0:
                # Title row — dynamically shrink the font if the title would
                # otherwise wrap past the reserved box, and only fall back
                # to an ellipsis as a last resort. Matches the flash card
                # title behavior.
                title_box_h = 52
                v_lbl.setFixedHeight(title_box_h)
                v_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                fitted_text, fitted_pt = _fit_title_text(
                    str(val),
                    avail_width=val_avail_w,
                    max_height=title_box_h,
                    base_pt=9.5,
                    base_font=v_lbl.font(),
                    min_pt=7.5,
                )
                # Same rationale as the flash-card title: apply the fitted
                # size via ``setFont`` so the rendered font exactly matches
                # what :func:`_fit_title_text` measured. Raw CJK / filename
                # titles otherwise slip past the stylesheet font-size
                # resolver and overflow horizontally without triggering
                # the shrink loop.
                title_font = QFont(v_lbl.font())
                title_font.setPointSizeF(fitted_pt)
                title_font.setBold(True)
                v_lbl.setFont(title_font)
                v_lbl.setText(fitted_text)
                v_lbl.setToolTip(str(val))
                v_lbl.setStyleSheet("color: #e0e0e0; font-weight: bold;")
            else:
                v_lbl.setText(str(val))
                v_lbl.setStyleSheet(
                    "color: #e0e0e0; font-size: 9.5pt; font-weight: bold;"
                )
            self._meta_grid.addWidget(k_lbl, i, 0, Qt.AlignTop | Qt.AlignLeft)
            self._meta_grid.addWidget(v_lbl, i, 1, Qt.AlignTop | Qt.AlignLeft)

        tags = self._display_tag_values()
        self._genres_heading.hide()
        self._genres_row.hide()
        self._tags_heading.setVisible(bool(tags))
        self._tags_row.setVisible(bool(tags))
        self._fill_chip_row(self._tags_layout, tags)

        # Button availability — also dim the emoji + update tooltip so it's
        # obvious why the action isn't usable. (Previously only computed in
        # _on_details_ready; moved to the shared hero pass so the preview
        # already enables / disables the actions correctly.)
        resolved_out = self._resolve_output_folder_target()
        folder_ok = bool(resolved_out) and os.path.isdir(resolved_out)
        self._edit_metadata_btn.setEnabled(folder_ok)
        self._edit_metadata_btn.setCursor(
            Qt.PointingHandCursor if folder_ok else Qt.ForbiddenCursor
        )
        self._edit_metadata_btn.setToolTip(
            "Edit the output workspace's metadata.json"
            if folder_ok
            else "An output workspace is required to edit metadata"
        )
        self._folder_btn.setEnabled(folder_ok)
        self._folder_btn.setCursor(Qt.PointingHandCursor if folder_ok else Qt.ForbiddenCursor)
        self._folder_btn_opacity.setOpacity(1.0 if folder_ok else 0.35)
        self._folder_btn.setToolTip(
            f"Open output folder:\n{resolved_out}" if folder_ok
            else "Output folder not available for this book"
        )
        resolved_src = self._resolve_source_file_target()
        source_ok = bool(resolved_src) and os.path.isfile(resolved_src)
        self._source_btn.setEnabled(source_ok)
        self._source_btn.setCursor(Qt.PointingHandCursor if source_ok else Qt.ForbiddenCursor)
        self._source_btn_opacity.setOpacity(1.0 if source_ok else 0.35)
        self._source_btn.setToolTip(
            f"Reveal source file:\n{resolved_src}" if source_ok
            else "Source file not found on disk"
        )
        resolved_trans = self._resolve_translated_file_target()
        trans_ok = bool(resolved_trans) and os.path.isfile(resolved_trans)
        self._translated_btn.setEnabled(trans_ok)
        self._translated_btn.setCursor(
            Qt.PointingHandCursor if trans_ok else Qt.ForbiddenCursor)
        self._translated_btn_opacity.setOpacity(
            1.0 if trans_ok else 0.35)
        self._translated_btn.setToolTip(
            f"Reveal translated file:\n{resolved_trans}" if trans_ok
            else "Translated file not found on disk"
        )

    @Slot(dict)
    def _on_preview_ready(self, payload: dict):
        """Render the hero row from Phase 1 (cover + metadata) immediately.

        Called before ``_on_details_ready`` so the user sees a fully
        populated cover + title + synopsis in milliseconds, without
        waiting for the spine's per-chapter HTML parse.
        """
        self._apply_hero_payload(payload)

    @Slot(dict)
    def _on_details_ready(self, payload: dict):
        new_chapters_info = payload.get("chapters_info", []) or []
        auto = bool(getattr(self, "_is_auto_refreshing", False))
        new_metadata_json = payload.get("metadata_json", {}) or {}
        metadata_changed = new_metadata_json != self._metadata_json
        # On an auto-refresh, skip the expensive TOC rebuild when every
        # per-chapter signal is unchanged — otherwise we'd pointlessly
        # tear down + re-add hundreds of rows every 2 seconds.
        old_sig = [
            (c.get("index"), c.get("status"),
             c.get("translated_path") or "",
             c.get("translated_title") or "",
             c.get("chunk_status_text") or "")
            for c in self._chapters_info
        ]
        new_sig = [
            (c.get("index"), c.get("status"),
             c.get("translated_path") or "",
             c.get("translated_title") or "",
             c.get("chunk_status_text") or "")
            for c in new_chapters_info
        ]
        chapters_changed = old_sig != new_sig
        self._chapters_info = new_chapters_info
        # Re-apply the hero with the richer details returned by Phase 2
        # (OPF now includes real chapter titles, metadata_json may have
        # been re-loaded). Hero widgets are idempotent so this is safe.
        if not auto or metadata_changed:
            self._apply_hero_payload(payload)

        # In-progress strip + reader-entry-point relabeling
        has_translated = any(c.get("translated_path") for c in self._chapters_info)
        raw_source = str(self._book.get("raw_source_path") or "")
        is_pdf_workspace = bool(
            raw_source.lower().endswith(".pdf")
            or str(self._book.get("workspace_kind") or "").lower() == "pdf"
            or str(self._book.get("compiled_output_kind") or "").lower() == "pdf"
        )
        if self._book.get("is_in_progress"):
            self._update_progress_strip()
            # When any chapter has been translated, the primary reader shows
            # the translated content (raw fallback for pending chapters). The
            # secondary "Read raw source" button remains for the raw view.
            if has_translated:
                self._start_btn.setText("\U0001f4d6  Read translated")
                self._raw_btn.show()
            else:
                self._start_btn.setText("\U0001f4d6  Read raw source")
                self._raw_btn.hide()
        elif is_pdf_workspace:
            self._progress_strip.hide()
            if has_translated:
                self._start_btn.setText("\U0001f4d6  Read translated")
                self._raw_btn.setToolTip(
                    "Read raw bookmarked PDF sections (extracted and cached on demand)"
                )
                self._raw_btn.show()
            else:
                self._start_btn.setText("\U0001f4d6  Read raw source")
                self._raw_btn.hide()
        else:
            self._progress_strip.hide()
            self._start_btn.setText("\U0001f4d6  Start reading")
            self._raw_btn.hide()

        # "Show raw titles" toggle is meaningful whenever at least one
        # chapter has BOTH a raw title and a distinct translated title
        # — covers in-progress books (titles harvested from source spine
        # + response files) AND Completed-tab library entries that have
        # a paired raw EPUB (titles harvested from both EPUBs by the
        # loader's library-raw pass).
        has_distinct_titles = any(
            (c.get("raw_title") or "")
            and (c.get("translated_title") or "")
            and (c.get("raw_title") or "") != (c.get("translated_title") or "")
            for c in self._chapters_info
        )
        raw_titles_applicable = has_distinct_titles
        # Skip toggling checkbox visibility during a silent auto-refresh
        # (the user may be interacting with it right now).
        if not auto:
            self._raw_titles_cb.setVisible(raw_titles_applicable)
        # Keep the flag accurate: if the toggle becomes inapplicable after
        # a refresh we don't want the checkbox state to stay "checked"
        # invisibly and force raw rendering on an unrelated book.
        if not raw_titles_applicable:
            if self._show_raw_titles:
                self._show_raw_titles = False
                self._raw_titles_cb.setChecked(False)

        # Chapter rows: on the initial load always rebuild; on a silent
        # auto-refresh only rebuild if the per-chapter signature drifted.
        if not auto or chapters_changed:
            self._populate_chapters(silent=auto)
        # Button availability is handled inside :meth:`_apply_hero_payload`
        # so both the preview and the final pass enable / disable actions
        # consistently. No extra work needed here.
        # Reset the auto-refresh flag so the next loader call (triggered
        # by anything other than :meth:`_auto_refresh_details`) goes
        # through the full reveal path.
        self._is_auto_refreshing = False

    def _update_progress_strip(self):
        """Refresh the "Translation in progress" strip.

        Uses ``self._show_special_files`` as the counting toggle so the
        user-visible "Show special files" checkbox drives the percentage
        (in addition to the chapter list filter). Rules:

        - Checkbox ON  → count every spine chapter, including configured
          special files, toward the denominator.
        - Checkbox OFF → exclude special files so a book doesn't sit at
          98/100 forever when the only untranslated entries are files
          the translator skips by design.

        The checkbox itself is initialized from the global
        ``translate_special_files`` setting (see :meth:`__init__`), so
        enabling that toggle in Other Settings cascades into the dialog.
        """
        text = self._progress_strip_text()
        if text is None:
            self._progress_strip.hide()
            return
        self._progress_strip.setText(text)
        self._progress_strip.show()

    # _progress_strip_text was extracted from _update_progress_strip into library_core.BookDetailsMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _fill_chip_row(self, layout: QLayout, values: list[str]):
        # Flow vertically as needed instead of compressing every tag into a
        # single row. All tags are shown; the old implementation capped the
        # list at 12 and then squeezed those labels below their size hints.
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for v in values:
            if not v:
                continue
            text = str(v).strip()
            if not text:
                continue
            chip = QLabel(text)
            chip.setProperty("class", "tag")
            chip.setStyleSheet(
                "color: #c8cbe0; background: #2a2a3e; "
                "border: 1px solid #3a3a5e; border-radius: 10px; "
                "padding: 5px 11px; font-size: 9pt;"
            )
            chip.setToolTip(text)
            chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            chip.ensurePolished()
            chip.setMinimumSize(chip.sizeHint())
            layout.addWidget(chip)
        layout.invalidate()
        self._tags_row.updateGeometry()

    # _collect_tag_values, _metadata_author_values, _display_tag_values moved verbatim to library_core.BookDetailsMixin (inherited).

    def _style_chapter_page_size_combo(self) -> None:
        icon_path = _find_halgakos_icon()
        if not icon_path:
            return
        icon_url = icon_path.replace("\\", "/")
        self._toc_page_size_combo.setStyleSheet(f"""
            QComboBox#toc-page-size {{
                background: #1e1e2e; color: #e0e0e0;
                border: 1px solid #3a3a5e; border-radius: 6px;
                padding: 5px 30px 5px 9px; font-size: 9pt;
            }}
            QComboBox#toc-page-size:hover {{
                border-color: #6c63ff; background: #24243a;
            }}
            QComboBox#toc-page-size::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid #3a3a5e;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: #202036;
            }}
            QComboBox#toc-page-size::down-arrow {{
                image: url("{icon_url}");
                width: 16px;
                height: 16px;
            }}
            QComboBox#toc-page-size QAbstractItemView {{
                background: #1e1e2e;
                color: #e0e0e0;
                border: 1px solid #3a3a5e;
                selection-background-color: #2a2d5a;
                selection-color: #ffffff;
            }}
        """)

    def _schedule_details_config_persist(self) -> None:
        timer = getattr(self, "_details_config_persist_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: _persist_config_via_parent(self))
            self._details_config_persist_timer = timer
        timer.start(750)

    def _stop_chapter_list_animation(self) -> None:
        anim = getattr(self, "_chap_list_anim", None)
        if anim is not None:
            try:
                anim.stop()
            except Exception:
                pass
        self._chap_list_anim = None

    def _set_chapter_list_opacity(self, opacity: float) -> None:
        effect = getattr(self, "_chap_list_opacity", None)
        if effect is not None:
            effect.setOpacity(max(0.0, min(1.0, float(opacity))))

    def _begin_chapter_page_transition(self) -> None:
        self._stop_chapter_list_animation()
        self._set_chapter_list_opacity(0.62)

    def _commit_chapter_page_transition_swap(self) -> None:
        if not bool(getattr(self, "_chapter_list_defer_clear", False)):
            return
        chap_list = getattr(self, "_chap_list", None)
        if chap_list is not None:
            chap_list.clear()
            chap_list.show()
        self._chapter_list_defer_clear = False
        self._chapter_list_transition_pending = True
        self._set_chapter_list_opacity(0.0)
        # Switch the title, page summary, navigation state, and bottom pager
        # in the same event-loop turn as the first replacement rows.  Keeping
        # these deferred avoids a transient hybrid frame such as
        # "Failures (0)" beside "Page 1 / 14 · 1-20 of 261".
        self._update_toc_toggle_label()

    def _finish_chapter_page_transition(self) -> None:
        if not bool(getattr(self, "_chapter_list_transition_pending", False)):
            self._set_chapter_list_opacity(1.0)
            return
        effect = getattr(self, "_chap_list_opacity", None)
        if effect is None:
            self._chapter_list_transition_pending = False
            return
        self._stop_chapter_list_animation()
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(110)
        anim.setStartValue(max(0.0, min(1.0, float(effect.opacity()))))
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(
            lambda: setattr(self, "_chapter_list_transition_pending", False))
        self._chap_list_anim = anim
        anim.start()

    # Per-tick batch size for :meth:`_populate_chapters` — small enough
    # The virtual list path is used for every page-size option so
    # 250 / 500 / All don't regress into hundreds of live child widgets.
    _POPULATE_BATCH_SIZE = 10
    _POPULATE_LIST_BATCH_SIZE = 180
    _POPULATE_TICK_MS = 16
    _POPULATE_FRAME_BUDGET_SEC = 0.006
    _POPULATE_LIST_FRAME_BUDGET_SEC = 0.008
    _CHAPTER_GEOMETRY_REFRESH_INTERVAL_SEC = 0.12

    def _cancel_chapter_row_prep(self) -> None:
        for thread in list(getattr(self, "_chapter_row_prep_threads", []) or []):
            try:
                thread.cancel()
            except Exception:
                pass

    def _populate_chapters(self, silent: bool = False):
        """Rebuild the chapter row list, yielding between batches.

        A compiled EPUB with hundreds of chapters used to freeze the
        UI for multiple seconds while every row widget was constructed
        synchronously. We now prepare row display specs on a worker
        thread, then stream them into a lightweight painted list so the
        event loop keeps dispatching in between.

        In **initial** (non-silent) mode, the "⏳  Loading chapters…"
        placeholder is hidden as soon as row data is ready, then each
        batch is painted into the visible chapter container immediately.

        In **silent** mode (auto-refresh), the loading placeholder is
        NOT shown — the existing rows stay visible while the new
        batches render on top. This avoids a visible flash every 2
        seconds when the auto-refresh timer ticks, and the user's
        scroll position / selection stays intact.

        Any pre-existing batch timer is cancelled first so a toggle
        flip (e.g. Show raw titles) doesn't double-render.
        """
        # Tear down before rebuilding and disable activation for the window
        # of time while we're mutating the list — this flag is read from
        # :meth:`_on_chapter_activated` to squash spurious clicks that
        # might land while Phase 2 is still populating rows.
        self._chapters_loaded = False
        self._populate_generation = (
            int(getattr(self, "_populate_generation", 0) or 0) + 1
        )
        generation = self._populate_generation
        self._cancel_chapter_row_prep()
        # Cancel any in-flight batch timer from a previous call. Users
        # can retrigger this via the Show-raw-titles toggle or a scan
        # refresh while the previous batch is still rendering.
        populate_timer = getattr(self, "_populate_timer", None)
        if populate_timer is not None and populate_timer.isActive():
            populate_timer.stop()
        # Read user intent from the toggle flag rather than the
        # container's live visibility — the placeholder may hide the
        # container while the background loader is still running, so
        # isVisible() would incorrectly report "collapsed" on every load.
        target_visible = True
        use_list_view = True
        chap_list = getattr(self, "_chap_list", None)
        # Remember the toolbar's viewport anchor before a filter/page swap.
        # Without this, an empty Failures result shrinks the scroll body by
        # ~1300 px; Qt clamps the scrollbar and the toolbar jumps downward.
        # Any blank height needed to retain this anchor is added to the chapter
        # list itself in _restore_chapter_scroll_anchor(), never to the whole
        # body (which would cause the hero layout to stretch vertically).
        chapter_scroll_anchor_y = None
        if silent and chap_list is not None and chap_list.isVisible():
            try:
                scroll = self._scroll
                viewport = scroll.viewport()
                chapter_scroll_anchor_y = self._toc_title.mapTo(
                    viewport, QPoint(0, 0)).y()
            except Exception:
                chapter_scroll_anchor_y = None
        self._chapter_scroll_anchor_y = chapter_scroll_anchor_y
        defer_list_clear = bool(
            silent
            and use_list_view
            and chap_list is not None
            and chap_list.isVisible()
        )
        self._chapter_list_defer_clear = defer_list_clear
        self._chapter_list_transition_pending = False
        # Initial loads used to hide the container until the final batch,
        # which made large books feel stuck. Now the placeholder disappears
        # once row data exists and every batch paints as soon as it is
        # constructed. Silent refreshes keep the same no-placeholder path.
        if not silent:
            if getattr(self, "_chap_loading_lbl", None) is not None:
                self._chap_loading_lbl.hide()
            if getattr(self, "_chap_container", None) is not None:
                self._chap_container.setVisible(
                    bool(target_visible and not use_list_view))
            if getattr(self, "_chap_list", None) is not None:
                self._chap_list.setVisible(
                    bool(target_visible and use_list_view))
            if getattr(self, "_toc_bottom_pager", None) is not None:
                self._toc_bottom_pager.setVisible(bool(target_visible))
        # Clear previous rows (placeholder now lives outside this layout).
        while self._chap_layout.count():
            item = self._chap_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        if chap_list is not None:
            if defer_list_clear:
                self._begin_chapter_page_transition()
            else:
                self._stop_chapter_list_animation()
                chap_list.clear()
                self._set_chapter_list_opacity(1.0)
            chap_list.setVisible(bool(target_visible and use_list_view))
        if getattr(self, "_chap_container", None) is not None:
            self._chap_container.setVisible(
                bool(target_visible and not use_list_view))

        # Snapshot the state the worker uses so a mid-flight toggle
        # change can't cross-contaminate the rendering flavor.
        filtered_infos = self._filtered_chapter_infos()
        if not defer_list_clear:
            self._update_chapter_pagination_controls(len(filtered_infos))
        populate_infos = [
            dict(info or {})
            for info in self._current_chapter_page_infos(filtered_infos)
        ]
        self._populate_row_specs = []
        self._populate_idx = 0
        self._populate_prep_done = False
        self._populate_show_raw_title = bool(self._show_raw_titles)
        self._populate_target_visible = bool(target_visible)
        self._populate_silent = bool(silent)
        self._populate_use_list = bool(use_list_view)
        self._last_chapter_geometry_refresh = 0.0

        if not populate_infos:
            self._commit_chapter_page_transition_swap()
            self._populate_prep_done = True
            self._finish_chapter_population()
            return

        if populate_timer is None:
            populate_timer = QTimer(self)
            populate_timer.setSingleShot(False)
            populate_timer.setInterval(self._POPULATE_TICK_MS)
            populate_timer.timeout.connect(self._populate_chapters_batch_tick)
            self._populate_timer = populate_timer

        prep_thread = _ChapterRowPrepThread(
            generation,
            populate_infos,
            bool(self._populate_show_raw_title),
            self,
        )
        prep_thread.batch_ready.connect(self._on_chapter_row_specs_batch)
        prep_thread.finished.connect(
            lambda t=prep_thread: self._on_chapter_row_prep_finished(t))
        self._chapter_row_prep_thread = prep_thread
        self._chapter_row_prep_threads.append(prep_thread)
        prep_thread.start()

    def _on_chapter_row_specs_batch(self, generation: int, specs, done: bool):
        if generation != getattr(self, "_populate_generation", 0):
            return
        if specs:
            self._populate_row_specs.extend(list(specs))
        if done:
            self._populate_prep_done = True

        populate_timer = getattr(self, "_populate_timer", None)
        if populate_timer is not None and not populate_timer.isActive():
            populate_timer.start()

    def _on_chapter_row_prep_finished(self, thread) -> None:
        try:
            threads = getattr(self, "_chapter_row_prep_threads", [])
            if thread in threads:
                threads.remove(thread)
        except Exception:
            pass
        if getattr(self, "_chapter_row_prep_thread", None) is thread:
            self._chapter_row_prep_thread = None
        try:
            thread.deleteLater()
        except Exception:
            pass

    def _refresh_chapter_stream_geometry(self, final: bool = False) -> None:
        if getattr(self, "_refreshing_chapter_geometry", False):
            return
        self._refreshing_chapter_geometry = True
        try:
            layout = getattr(self, "_chap_layout", None)
            if layout is not None:
                layout.invalidate()
                layout.activate()

            container = getattr(self, "_chap_container", None)
            if container is not None:
                container.updateGeometry()
                if final:
                    container.adjustSize()

            chap_list = getattr(self, "_chap_list", None)
            if chap_list is not None:
                chap_list.updateGeometry()
                viewport_fn = getattr(chap_list, "viewport", None)
                viewport = viewport_fn() if callable(viewport_fn) else None
                if viewport is not None:
                    viewport.update()
                else:
                    chap_list.update()

            scroll = getattr(self, "_scroll", None)
            body = scroll.widget() if scroll is not None else None
            if body is not None:
                body_layout = body.layout()
                if body_layout is not None:
                    body_layout.invalidate()
                    body_layout.activate()
                body.updateGeometry()
                if final:
                    body.adjustSize()

            if scroll is not None:
                scroll.updateGeometry()
                viewport = scroll.viewport()
                if viewport is not None:
                    viewport.update()
        except Exception:
            pass
        finally:
            self._refreshing_chapter_geometry = False

    def _maybe_refresh_chapter_stream_geometry(self, force: bool = False) -> None:
        now = time.perf_counter()
        last = float(getattr(self, "_last_chapter_geometry_refresh", 0.0) or 0.0)
        if not force and now - last < self._CHAPTER_GEOMETRY_REFRESH_INTERVAL_SEC:
            return
        self._last_chapter_geometry_refresh = now
        self._refresh_chapter_stream_geometry(final=False)

    def _finish_chapter_population(self):
        if self._chapters_loaded:
            return
        timer = getattr(self, "_populate_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        use_list_view = bool(getattr(self, "_populate_use_list", False))
        if not use_list_view:
            self._chap_layout.addStretch()
        self._update_toc_toggle_label()
        if not bool(getattr(self, "_populate_silent", False)):
            if getattr(self, "_chap_loading_lbl", None) is not None:
                self._chap_loading_lbl.hide()
            if getattr(self, "_populate_target_visible", True):
                if use_list_view and getattr(self, "_chap_list", None) is not None:
                    self._chap_list.show()
                    self._chap_container.hide()
                else:
                    self._chap_container.show()
                    if getattr(self, "_chap_list", None) is not None:
                        self._chap_list.hide()
                if getattr(self, "_toc_bottom_pager", None) is not None:
                    self._toc_bottom_pager.show()
        self._chapters_loaded = True
        if use_list_view:
            self._finish_chapter_page_transition()
        self._refresh_chapter_stream_geometry(final=True)
        self._restore_chapter_scroll_anchor()

    def _restore_chapter_scroll_anchor(self) -> None:
        """Restore the chapter toolbar's viewport Y after a row-count swap."""
        anchor_y = getattr(self, "_chapter_scroll_anchor_y", None)
        self._chapter_scroll_anchor_y = None
        if anchor_y is None:
            return
        try:
            viewport = self._scroll.viewport()
            body = self._scroll.widget()
            scrollbar = self._scroll.verticalScrollBar()
            chap_list = self._chap_list

            # Discard a reservation left by the preceding page, then grow the
            # result slot only by each measured scrollbar shortfall.  Usually
            # two passes are enough: the first exposes the compact layout's
            # fixed height and the second reaches the exact saved offset.  The
            # extra passes cover a horizontal scrollbar appearing/disappearing.
            chap_list.set_reserved_height(0)
            for _ in range(4):
                body_layout = body.layout()
                if body_layout is not None:
                    body_layout.invalidate()
                    body_layout.activate()
                body.adjustSize()
                toolbar_body_y = self._toc_title.mapTo(
                    body, QPoint(0, 0)).y()
                desired_value = max(
                    0, int(toolbar_body_y) - int(anchor_y))
                shortfall = desired_value - int(scrollbar.maximum())
                if shortfall <= 0:
                    break
                scrollbar_extent = max(
                    1,
                    int(QApplication.style().pixelMetric(
                        QStyle.PM_ScrollBarExtent)),
                )
                chap_list.set_reserved_height(
                    chap_list.reserved_height()
                    + shortfall
                    + scrollbar_extent)

            body_layout = body.layout()
            if body_layout is not None:
                body_layout.invalidate()
                body_layout.activate()
            body.adjustSize()
            toolbar_body_y = self._toc_title.mapTo(body, QPoint(0, 0)).y()
            scrollbar.setValue(
                max(0, int(toolbar_body_y) - int(anchor_y)))
        except Exception:
            pass

    def _on_chapter_virtual_clicked(self, idx: int) -> None:
        self._selected_chapter_idx = int(idx)

    def _populate_chapters_batch_tick(self):
        """Render the next :data:`_POPULATE_BATCH_SIZE` chapter rows.

        Stops the driving ``QTimer`` and finalizes the TOC state once
        every row has been appended to the layout.
        """
        specs = getattr(self, "_populate_row_specs", None) or []
        start = int(getattr(self, "_populate_idx", 0) or 0)
        if start >= len(specs):
            if bool(getattr(self, "_populate_prep_done", False)):
                self._finish_chapter_population()
            return

        use_list_view = bool(getattr(self, "_populate_use_list", False))
        batch_size = (self._POPULATE_LIST_BATCH_SIZE if use_list_view
                      else self._POPULATE_BATCH_SIZE)
        frame_budget = (self._POPULATE_LIST_FRAME_BUDGET_SEC if use_list_view
                        else self._POPULATE_FRAME_BUDGET_SEC)
        end_limit = min(start + batch_size, len(specs))
        selected_idx = self._selected_chapter_idx
        end = start
        deadline = time.perf_counter() + frame_budget
        if use_list_view:
            chap_list = getattr(self, "_chap_list", None)
            if chap_list is None:
                self._finish_chapter_population()
                return
            first_transition_batch = bool(
                getattr(self, "_chapter_list_defer_clear", False))
            self._commit_chapter_page_transition_swap()
            batch = []
            for i in range(start, end_limit):
                batch.append(specs[i])
                end = i + 1
                if end > start and time.perf_counter() >= deadline:
                    break
            chap_list.append_specs(batch, selected_idx)
            if first_transition_batch:
                self._finish_chapter_page_transition()
        else:
            for i in range(start, end_limit):
                row_spec = specs[i]
                info = row_spec.get("info", {})
                row = _ChapterRow(
                    info,
                    parent=self._chap_container,
                    row_spec=row_spec,
                )
                row.activated.connect(self._on_chapter_activated)
                row.clicked.connect(self._on_chapter_clicked)
                row.menu_requested.connect(self._show_chapter_context_menu)
                if info.get("index") == selected_idx:
                    row.set_selected(True)
                self._chap_layout.addWidget(row)
                end = i + 1
                if end > start and time.perf_counter() >= deadline:
                    break
        self._populate_idx = end
        self._maybe_refresh_chapter_stream_geometry()
        if end >= len(specs) and bool(getattr(self, "_populate_prep_done", False)):
            self._finish_chapter_population()

    def _on_chapter_clicked(self, idx: int):
        """Update the single-select focus to the clicked chapter row."""
        self._selected_chapter_idx = idx
        chap_list = getattr(self, "_chap_list", None)
        if chap_list is not None:
            chap_list.set_selected_index(idx)
        for i in range(self._chap_layout.count()):
            w = self._chap_layout.itemAt(i).widget()
            if isinstance(w, _ChapterRow):
                w.set_selected(w.info.get("index") == idx)

    def _on_chapter_activated(self, idx: int):
        """Open the reader at *idx* only if the chapter list is fully loaded.

        Chapter rows emit ``activated`` on double-click (see
        :class:`_ChapterRow`), but if a click manages to arrive while the
        spine is still being populated we silently ignore it rather than
        opening a reader positioned into a half-built list.
        """
        if not self._chapters_loaded:
            return
        self._open_reader(initial_chapter=idx)

    def _on_compile_epub_clicked(self):
        """Run the compiler appropriate for this book's output workspace."""
        try:
            out_folder = _resolve_book_output_folder(self._book)
        except Exception:
            out_folder = ""
        compile_kind = _workspace_compile_kind(self._book, out_folder)
        label = "PDF" if compile_kind == "pdf" else "EPUB"
        if not out_folder or not os.path.isdir(out_folder):
            QMessageBox.warning(
                self, f"Compile {label}",
                "Could not resolve this book's output folder.\n"
                "Run a translation first so a workspace exists.")
            return
        gui = _find_translator_gui(self)
        converter_name = (
            "pdf_converter" if compile_kind == "pdf" else "epub_converter")
        if gui is None or not hasattr(gui, converter_name):
            QMessageBox.warning(
                self, f"Compile {label}",
                "The main translator window is not available.")
            return
        try:
            getattr(gui, converter_name)(folder=out_folder)
        except Exception as exc:
            QMessageBox.warning(
                self, f"Compile {label}",
                f"Could not start the {label} compiler:\n{exc}")
            return
        # Animate immediately for instant click feedback; the poll ends the
        # animation quickly if a guard rejected the start (worker never ran).
        self._begin_compile_animation()
        # Also flip this book's flash card ribbon to "COMPILING…" in the
        # Library grid behind this dialog.
        parent_lib = self.parent()
        if parent_lib is not None and hasattr(parent_lib,
                                              "_watch_compile_for_folder"):
            try:
                parent_lib._set_cards_compiling(out_folder, True)
                parent_lib._watch_compile_for_folder(out_folder, gui)
            except Exception:
                pass

    # _metadata_editor_values, _source_metadata_values moved verbatim to library_core.BookDetailsMixin (inherited).

    def _on_edit_metadata_clicked(self):
        """Edit and atomically save this workspace's metadata.json."""
        output_folder = self._resolve_output_folder_target()
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(
                self,
                "Edit Metadata",
                "Could not resolve this book's output workspace.",
            )
            return

        dialog = _BookMetadataEditDialog(
            self._metadata_editor_values(),
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        edits = dialog.changed_values()
        if not edits:
            return

        try:
            updated = self._save_metadata_edits(output_folder, edits)
        except _MetadataEditError as exc:
            QMessageBox.warning(self, "Edit Metadata", str(exc))
            return
        if updated is None:
            return

        self._metadata_json = updated
        self._book["metadata_json"] = dict(updated)
        self._apply_hero_payload({
            "details": self._details,
            "metadata_json": updated,
            "cover": self._current_cover_path,
        })

        parent = self.parent()
        if parent is not None and hasattr(parent, "_auto_refresh"):
            QTimer.singleShot(0, parent._auto_refresh)

    # _save_metadata_edits was extracted from _on_edit_metadata_clicked into library_core.BookDetailsMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _on_translate_metadata_clicked(self):
        """Run this book's metadata phase through the parent Library."""
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "_translate_metadata_for_book"):
                parent._translate_metadata_for_book(self._book)
                return
            try:
                parent = parent.parent()
            except Exception:
                parent = None
        QMessageBox.warning(
            self,
            "Translate Metadata",
            "The EPUB Library window is not available.",
        )

    def _compile_epub_running(self) -> bool:
        """True while the main GUI's EPUB or PDF compiler is active."""
        gui = _find_translator_gui(self)
        return _epub_converter_running(gui)

    def _compile_spin_frames(self) -> list:
        """Build (and cache) pre-rendered high-DPI spinner frames.

        Frames are rendered once from the highest-resolution icon source
        available (PNG preferred over .ico) at the button's device pixel
        ratio, so the spinning icon stays crisp on scaled displays and each
        tick is just a cached ``setIcon`` — no per-frame painting.
        """
        dpr = 1.0
        try:
            dpr = float(self.devicePixelRatioF() or 1.0)
        except Exception:
            pass
        cached = getattr(self, "_compile_spin_cache", None)
        if cached is not None and cached.get("dpr") == dpr:
            return cached["frames"]

        # Pick the sharpest source: the PNG sibling of the resolved icon
        # first (large raster), then whatever _find_halgakos_icon returned.
        src = None
        icon_path = _find_halgakos_icon() or ""
        candidates = []
        if icon_path:
            candidates.append(os.path.splitext(icon_path)[0] + ".png")
            candidates.append(icon_path)
        for cand in candidates:
            if cand and os.path.isfile(cand):
                pm = QPixmap(cand)
                if not pm.isNull():
                    src = pm
                    break
        frames: list = []
        if src is not None:
            # Downscale the (potentially huge) source once to ~4x target so
            # per-frame rotation+scale stays cheap but keeps detail.
            work = src
            if work.width() > 96 or work.height() > 96:
                work = work.scaled(96, 96, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            logical = 18.0
            size_px = max(1, int(round(logical * dpr)))
            # Fit the source into the 18×18 logical box WITHOUT distorting
            # it — scale by the limiting dimension and center the result.
            sw = max(1, work.width())
            sh = max(1, work.height())
            fit = min(logical / sw, logical / sh)
            tw = sw * fit
            th = sh * fit
            target = QRectF(-tw / 2.0, -th / 2.0, tw, th)
            for step in range(15):  # 24° per frame
                angle = step * 24
                frame = QPixmap(size_px, size_px)
                frame.setDevicePixelRatio(dpr)
                frame.fill(Qt.transparent)
                painter = QPainter(frame)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                half = logical / 2.0
                painter.translate(half, half)
                painter.rotate(angle)
                painter.drawPixmap(target, work, QRectF(work.rect()))
                painter.end()
                frames.append(QIcon(frame))
        self._compile_spin_cache = {"dpr": dpr, "frames": frames}
        return frames

    def _begin_compile_animation(self):
        if getattr(self, "_compile_anim_active", False):
            return
        btn = getattr(self, "_compile_epub_btn", None)
        if btn is None:
            return
        self._compile_anim_active = True
        self._compile_anim_step = 0
        self._compile_anim_started = time.monotonic()
        self._compile_btn_text = btn.text()
        btn.setEnabled(False)
        btn.setText("  Compiling…")
        if getattr(self, "_compile_anim_timer", None) is None:
            self._compile_anim_timer = QTimer(self)
            self._compile_anim_timer.setInterval(70)
            self._compile_anim_timer.timeout.connect(
                lambda: self._compile_anim_tick())
            self._compile_poll_timer = QTimer(self)
            self._compile_poll_timer.setInterval(150)
            self._compile_poll_timer.timeout.connect(
                lambda: self._compile_anim_poll())
        self._compile_anim_tick()  # first frame immediately, no 70ms wait
        self._compile_anim_timer.start()
        self._compile_poll_timer.start()

    def _compile_anim_tick(self):
        btn = getattr(self, "_compile_epub_btn", None)
        if btn is None:
            return
        step = getattr(self, "_compile_anim_step", 0)
        self._compile_anim_step = step + 1
        frames = self._compile_spin_frames()
        if frames:
            btn.setIcon(frames[step % len(frames)])
            btn.setIconSize(QSize(18, 18))
        else:
            dots = step % 4
            icon = (
                "\U0001f4c4"
                if getattr(self, "_compile_output_kind", "epub") == "pdf"
                else "\U0001f4d8"
            )
            btn.setText(f"{icon}  Compiling" + "." * dots + " " * (3 - dots))

    def _compile_anim_poll(self):
        if self._compile_epub_running():
            return
        # Grace period: the executor needs a moment to surface the worker —
        # don't kill the animation before it ever registers as running.
        if (time.monotonic() - getattr(self, "_compile_anim_started", 0)) < 1.2:
            return
        self._end_compile_animation()

    def _end_compile_animation(self):
        if not getattr(self, "_compile_anim_active", False):
            return
        self._compile_anim_active = False
        for timer_name in ("_compile_anim_timer", "_compile_poll_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        btn = getattr(self, "_compile_epub_btn", None)
        if btn is not None:
            btn.setIcon(QIcon())
            btn.setText(getattr(self, "_compile_btn_text",
                                "\U0001f4d8  Compile EPUB"))
            btn.setEnabled(True)

    def _show_chapter_context_menu(self, idx: int, global_pos):
        """Right-click menu for a chapter entry (Translate / Open in Reader)."""
        if not self._chapters_loaded:
            return
        if not (0 <= idx < len(self._chapters_info)):
            return
        info = self._chapters_info[idx] or {}
        filename = info.get("filename") or ""
        status = (info.get("status") or "").strip()

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 9pt; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background: #3a3a5e; }
            QMenu::item:disabled { color: #666; }
        """)
        label = ("\U0001f310  Retranslate this chapter" if status == "completed"
                 else "\U0001f310  Translate this chapter")
        translate_action = menu.addAction(label)
        translate_action.setToolTip(
            "Translate only this HTML file — skips the full extraction "
            "pass and jumps straight to the translation phase.")
        if not filename or bool(info.get("is_gallery")):
            translate_action.setEnabled(False)
        menu.addSeparator()
        reader_action = menu.addAction("\U0001f4d6  Open in Reader")

        translate_action.triggered.connect(
            lambda *_a, i=idx: self._translate_single_chapter(i))
        reader_action.triggered.connect(
            lambda *_a, i=idx: self._on_chapter_activated(i))
        menu.exec(global_pos)

    def _on_qa_failures_only_toggled(self, checked: bool):
        """Switch the Chapters button between all rows and QA failures."""
        checked = bool(checked)
        if checked == self._show_qa_failures_only:
            return
        self._show_qa_failures_only = checked
        self._chapter_page = 0
        if self._chapters_info:
            self._populate_chapters(silent=True)
        else:
            self._update_toc_toggle_label()

    def _translate_single_chapter(self, idx: int):
        """Translate exactly one chapter entry via the main translator GUI.

        Skips the full extraction pass — only the HTML file behind this
        entry is pulled out of the source EPUB (numbering preserved) and
        the run jumps straight to the translation phase.
        """
        if not (0 <= idx < len(self._chapters_info)):
            return
        info = self._chapters_info[idx] or {}
        filename = info.get("filename") or ""
        if not filename:
            QMessageBox.warning(self, "Translate chapter",
                                "This entry has no source filename to translate.")
            return

        # Resolve the raw source EPUB the chapter lives in.
        def _is_epub_file(p: str) -> bool:
            return bool(p) and p.lower().endswith(".epub") and os.path.isfile(p)

        raw_source = self._book.get("raw_source_path", "") or ""
        book_path = self._book.get("path", "") or ""
        epub_path = next((p for p in (raw_source, book_path) if _is_epub_file(p)), "")
        if not epub_path:
            QMessageBox.warning(
                self, "Translate chapter",
                "Could not resolve the raw source EPUB for this book.")
            return

        gui = _find_translator_gui(self)
        if gui is None:
            QMessageBox.warning(
                self, "Translate chapter",
                "The main translator window is not available.")
            return
        thread = getattr(gui, "translation_thread", None)
        if thread is not None and thread.is_alive():
            QMessageBox.warning(
                self, "Translate chapter",
                "A translation is already running.\n"
                "Please wait for it to finish (or stop it) first.")
            return

        title = (info.get("translated_title") or info.get("raw_title")
                 or filename)
        status = (info.get("status") or "").strip()
        if status == "completed":
            if QMessageBox.question(
                    self, "Retranslate chapter",
                    f"“{title}” is already translated.\n\n"
                    "Delete its current translation and retranslate it now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return

        # Keep the output-directory override aligned with this book's
        # workspace (same guard "Load for translation" uses).
        parent_lib = self.parent()
        if parent_lib is not None and hasattr(parent_lib,
                                              "_ensure_output_override_matches"):
            try:
                if not parent_lib._ensure_output_override_matches([self._book]):
                    return
            except Exception:
                logger.debug("Output override check failed: %s",
                             traceback.format_exc())

        # Reset any existing progress entry so the pipeline re-translates
        # instead of skipping a chapter it considers done.
        if status:
            try:
                out_folder = _resolve_book_output_folder(self._book)
                if out_folder and os.path.isdir(out_folder):
                    _mark_chapter_pending_for_retranslation(out_folder, filename)
            except Exception:
                logger.debug("Progress reset failed: %s", traceback.format_exc())

        try:
            started = bool(gui.start_single_chapter_translation(
                epub_path, os.path.basename(filename)))
        except Exception as exc:
            QMessageBox.warning(self, "Translate chapter",
                                f"Could not start translation:\n{exc}")
            return
        if started:
            # The 2s auto-refresh timer will flip this row's badge to
            # "in progress" as soon as the pipeline picks the chapter up.
            try:
                gui.append_log(
                    f"\U0001f4d6 Library: translating single chapter "
                    f"“{title}” ({os.path.basename(filename)})")
            except Exception:
                pass

    # _visible_counts, _has_progress_context moved verbatim to library_core.BookDetailsMixin (inherited).

    def _update_toc_toggle_label(self):
        text, tooltip = self._toc_toggle_state()
        self._toc_toggle.setText(text)
        self._toc_toggle.setToolTip(tooltip)
        self._update_chapter_pagination_controls()

    # _chapter_base_infos, _filtered_chapter_infos moved verbatim to library_core.BookDetailsMixin (inherited).
    # _toc_toggle_state was extracted from _update_toc_toggle_label into library_core.BookDetailsMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _chapter_page_size(self) -> int:
        combo = getattr(self, "_toc_page_size_combo", None)
        value = combo.currentData() if combo is not None else 20
        if value == "all":
            return 0
        try:
            size = int(value)
        except (TypeError, ValueError):
            size = 20
        return max(1, size)

    def _chapter_page_bounds(self, total: int) -> tuple[int, int, int]:
        start, end, page_count, page = _page_bounds(
            total,
            getattr(self, "_chapter_page", 0),
            self._chapter_page_size(),
        )
        self._chapter_page = page
        return start, end, page_count

    def _current_chapter_page_infos(self, filtered_infos=None) -> list[dict]:
        if filtered_infos is None:
            filtered_infos = self._filtered_chapter_infos()
        start, end, _ = self._chapter_page_bounds(len(filtered_infos))
        return list(filtered_infos[start:end])

    def _update_chapter_pagination_controls(self, filtered_count: int | None = None):
        if filtered_count is None:
            filtered_count = len(self._filtered_chapter_infos())
        start, end, page_count = self._chapter_page_bounds(filtered_count)
        page = int(getattr(self, "_chapter_page", 0) or 0)

        label_text = _page_label(
            page, page_count, start, end, filtered_count,
            self._chapter_page_size())
        for label_name in ("_toc_page_label", "_toc_bottom_page_label"):
            label = getattr(self, label_name, None)
            if label is not None:
                label.setText(label_text)

        can_go_back = filtered_count > 0 and page > 0
        can_go_forward = filtered_count > 0 and page < page_count - 1

        for button_name in ("_toc_first_btn", "_toc_bottom_first_btn"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(can_go_back)
        for button_name in ("_toc_prev_btn", "_toc_bottom_prev_btn"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(can_go_back)
        for button_name in ("_toc_next_btn", "_toc_bottom_next_btn"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(can_go_forward)
        for button_name in ("_toc_last_btn", "_toc_bottom_last_btn"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(can_go_forward)

        bottom_pager = getattr(self, "_toc_bottom_pager", None)
        if bottom_pager is not None:
            bottom_pager.setVisible(filtered_count > 0)

    def _on_chapter_first_page(self):
        if int(getattr(self, "_chapter_page", 0) or 0) <= 0:
            return
        self._chapter_page = 0
        self._populate_chapters(silent=True)

    def _on_chapter_prev_page(self):
        if int(getattr(self, "_chapter_page", 0) or 0) <= 0:
            return
        self._chapter_page -= 1
        self._populate_chapters(silent=True)

    def _on_chapter_next_page(self):
        filtered_count = len(self._filtered_chapter_infos())
        _, _, page_count = self._chapter_page_bounds(filtered_count)
        if int(getattr(self, "_chapter_page", 0) or 0) >= page_count - 1:
            return
        self._chapter_page += 1
        self._populate_chapters(silent=True)

    def _on_chapter_last_page(self):
        filtered_count = len(self._filtered_chapter_infos())
        _, _, page_count = self._chapter_page_bounds(filtered_count)
        last_page = max(0, page_count - 1)
        if int(getattr(self, "_chapter_page", 0) or 0) >= last_page:
            return
        self._chapter_page = last_page
        self._populate_chapters(silent=True)

    def _on_chapter_page_size_changed(self, _index=None):
        combo = getattr(self, "_toc_page_size_combo", None)
        if combo is not None:
            try:
                value = combo.currentData()
                self._config["epub_details_chapter_page_size"] = value
                self._schedule_details_config_persist()
            except Exception:
                pass
        self._chapter_page = 0
        if self._chapters_info:
            self._populate_chapters(silent=True)
        else:
            self._update_toc_toggle_label()

    def _apply_chapter_filter(self, text: str):
        self._chapter_page = 0
        if self._chapters_info:
            self._populate_chapters(silent=True)
        else:
            self._update_toc_toggle_label()

    def _on_special_files_toggled(self, checked: bool):
        self._show_special_files = bool(checked)
        # Persist for next time this dialog (or another book) opens.
        try:
            self._config["epub_details_show_special_files"] = self._show_special_files
        except Exception:
            pass
        self._chapter_page = 0
        if self._chapters_info:
            self._populate_chapters(silent=True)
        else:
            self._update_toc_toggle_label()
        # The "Translation in progress — X/Y" strip now reflects this
        # toggle's state too, so refresh its fraction whenever the user
        # flips the checkbox.
        self._update_progress_strip()

    def _on_raw_titles_toggled(self, checked: bool):
        """Swap every chapter row between translated-title and raw-title mode."""
        new_value = bool(checked)
        if new_value == self._show_raw_titles:
            return
        self._show_raw_titles = new_value
        try:
            self._config["epub_details_show_raw_titles"] = self._show_raw_titles
        except Exception:
            pass
        # Re-populate the chapter list so every _ChapterRow is rebuilt with
        # the new primary-title policy. Cheaper than retrofitting the rows
        # in place and matches the pattern used for other toggles.
        if self._chapters_loaded:
            self._populate_chapters()

    def _toggle_chapters(self):
        # Kept as a compatibility no-op for older signal paths. The chapter
        # section is paginated now, not collapsible.
        self._chap_section_expanded = True
        self._chap_container.setVisible(False)
        if getattr(self, "_chap_list", None) is not None:
            self._chap_list.setVisible(True)
        if getattr(self, "_chap_loading_lbl", None) is not None:
            self._chap_loading_lbl.hide()
        self._toc_search.setVisible(True)
        self._special_cb.setVisible(True)
        # Mirror the applicability rule used in :meth:`_on_details_ready`
        # so the toggle surfaces for any book — in-progress OR library
        # — whose chapter rows carry distinct raw + translated titles.
        has_distinct_titles = any(
            (c.get("raw_title") or "")
            and (c.get("translated_title") or "")
            and (c.get("raw_title") or "") != (c.get("translated_title") or "")
            for c in self._chapters_info
        )
        self._raw_titles_cb.setVisible(has_distinct_titles)
        self._update_toc_toggle_label()

    @Slot(str)
    def _on_details_error(self, message: str):
        self._synopsis_lbl.setText(f"Failed to load book details: {message}")

    # -- Actions ------------------------------------------------------------

    # _build_translated_overlay, _translated_css_dirs moved verbatim to library_core.BookDetailsMixin (inherited).

    def _open_reader(self, initial_chapter: int | None = None, raw_only: bool = False):
        """Dispatch to the appropriate viewer based on the resolved source type.

        EPUB sources — whether ``book['path']`` is the EPUB itself (completed
        cards) or the output folder that holds one (in-progress cards) — use
        the integrated :class:`EpubReaderDialog` with the optional translated
        overlay. Only TXT / PDF / HTML / image workspaces fall through to the
        OS default viewer.
        """
        def _busy():
            # Same place as before the split: entering the workspace / EPUB
            # branch, before the translated overlay is built.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()

        try:
            plan = self._plan_open_reader(initial_chapter, raw_only, busy=_busy)
            if plan["mode"] in ("workspace", "epub"):
                reader = EpubReaderDialog(
                    plan["source"],
                    config=self._config,
                    parent=self,
                    **plan["kwargs"],
                )
                QApplication.restoreOverrideCursor()
                reader.setModal(False)
                reader.setAttribute(Qt.WA_DeleteOnClose)
                self._active_reader = reader
                reader.show()
                return

            target = plan.get("target") or ""
            if not target or not os.path.isfile(target):
                QMessageBox.warning(
                    self, "Error",
                    "No readable file is available for this book.",
                )
                return

            self._open_with_system_viewer(target)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            logger.error("Could not open reader from details: %s\n%s", exc, traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")

    # _plan_open_reader was extracted from _open_reader into library_core.BookDetailsMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _open_with_system_viewer(self, path: str):
        """Open *path* with the OS default handler.

        Mirrors the PDF/TXT branch inside
        :meth:`EpubLibraryDialog._on_card_clicked` so the two entry points
        stay behaviorally consistent.
        """
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            if sys.platform == "win32":
                if ext == "txt":
                    _npp_paths = [
                        r"C:\Program Files\Notepad++\notepad++.exe",
                        r"C:\Program Files (x86)\Notepad++\notepad++.exe",
                    ]
                    _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                    if _npp:
                        subprocess.Popen([_npp, path], creationflags=_no_window)
                    else:
                        subprocess.Popen(["notepad.exe", path], creationflags=_no_window)
                else:
                    os.startfile(path)
            elif sys.platform == "darwin":
                if ext == "txt" and shutil.which("code"):
                    subprocess.Popen(["code", path])
                else:
                    subprocess.Popen(["open", path])
            else:
                if ext == "txt":
                    _editors = ["gedit", "kate", "code", "mousepad", "xed", "pluma"]
                    _editor = next((e for e in _editors if shutil.which(e)), "xdg-open")
                    subprocess.Popen([_editor, path])
                else:
                    subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            logger.error("Could not open file %s: %s\n%s", path, exc, traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Could not open file:\n{exc}")

    # _resolve_output_folder_target, _resolve_source_file_target, _resolve_translated_file_target moved verbatim to library_core.BookDetailsMixin (inherited).

    def _open_output_folder(self):
        """Open the book's output folder in the system file explorer."""
        folder = self._resolve_output_folder_target()
        if folder and os.path.isdir(folder):
            _open_folder_in_explorer(folder)

    def _reveal_source(self):
        """Reveal the raw source file in the system file explorer."""
        path = self._resolve_source_file_target()
        if path and os.path.isfile(path):
            _open_folder_in_explorer(path)

    def _reveal_translated(self):
        """Reveal the compiled / translated EPUB in the file explorer."""
        path = self._resolve_translated_file_target()
        if path and os.path.isfile(path):
            _open_folder_in_explorer(path)

    # -- Qt lifecycle --------------------------------------------------------

    def closeEvent(self, event):
        active_reader = getattr(self, "_active_reader", None)
        if active_reader is not None:
            try:
                active_reader.close()
                QApplication.processEvents()
            except Exception:
                pass
            self._active_reader = None
        persist_timer = getattr(self, "_details_config_persist_timer", None)
        if persist_timer is not None and persist_timer.isActive():
            persist_timer.stop()
            _persist_config_via_parent(self)
        populate_timer = getattr(self, "_populate_timer", None)
        if populate_timer is not None and populate_timer.isActive():
            populate_timer.stop()
        self._cancel_chapter_row_prep()
        for thread in list(getattr(self, "_chapter_row_prep_threads", []) or []):
            _stop_qthread_safely(
                thread, timeout_ms=1000,
                signal_names=("batch_ready", "finished"),
            )
        self._chapter_row_prep_threads.clear()
        if self._loader is not None:
            _stop_qthread_safely(
                self._loader, timeout_ms=1500,
                signal_names=("preview_ready", "done", "error"),
            )
            self._loader = None
        super().closeEvent(event)


class _ChapterRow(QFrame):
    """One row in the Book Details chapter list.

    Displays translated title + filename when the chapter has been translated;
    otherwise shows the raw source title with a muted "pending" label.
    A single click selects the row (visual focus only); a double click
    activates it and opens the reader. This mirrors the flash-card UX in
    :class:`EpubLibraryDialog`.
    """
    activated = Signal(int)
    # Emitted on a single left-click so the parent dialog can update the
    # "currently focused" chapter row. Separate from :attr:`activated` so
    # selecting a row doesn't also open the reader.
    clicked = Signal(int)
    # Right-click → (chapter index, global QPoint). The parent dialog builds
    # the actual QMenu (Translate, Open in Reader, …).
    menu_requested = Signal(int, object)

    # Object-name selectors for the same reason as :class:`_BookCard`:
    # they match this exact widget reliably across Qt/PySide versions.
    # Borders are 2 px in both states to avoid layout shift on selection.
    _BASE_STYLE = (
        "QFrame#chapterRow { background: #1a1a2a; border: 2px solid #242438; border-radius: 6px; }"
    )
    _HOVER_STYLE = (
        "QFrame#chapterRow { background: #232340; border: 2px solid #6c63ff; border-radius: 6px; }"
    )
    _SELECTED_STYLE = (
        "QFrame#chapterRow { background: #2a2d5a; border: 2px solid #a097ff; border-radius: 6px; }"
    )
    _SELECTED_HOVER_STYLE = (
        "QFrame#chapterRow { background: #343670; border: 2px solid #c0b8ff; border-radius: 6px; }"
    )

    def __init__(self, info: dict, parent=None, show_raw_title: bool = False,
                 row_spec: dict | None = None):
        super().__init__(parent)
        if row_spec is None:
            row_spec = _prepare_chapter_row_spec(info, show_raw_title)
        self.info = row_spec.get("info", info)
        self._selected = False
        self._hovered = False
        self._applied_style = ""
        self.setObjectName("chapterRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Click to select — double-click to open this chapter in the reader"
        )
        self._apply_row_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        primary = QLabel(row_spec.get("primary_text", ""))
        primary.setProperty("class", row_spec.get("primary_class", "raw"))
        primary.setStyleSheet(row_spec.get("primary_style", ""))
        primary.setWordWrap(True)
        if row_spec.get("primary_tooltip"):
            primary.setToolTip(row_spec["primary_tooltip"])
        text_col.addWidget(primary)

        sub = QLabel(row_spec.get("filename", ""))
        sub.setProperty("class", "filename")
        sub.setStyleSheet("color: #8a8fa8; font-size: 8.5pt; font-family: 'Consolas','Menlo',monospace;")
        text_col.addWidget(sub)
        layout.addLayout(text_col, 1)

        badge_text = row_spec.get("badge_text", "")
        if badge_text:
            badge = QLabel(badge_text)
            badge.setStyleSheet(row_spec.get("badge_style", ""))
            layout.addWidget(badge, 0, Qt.AlignRight)

    def set_selected(self, selected: bool) -> None:
        """Toggle the row's "focused" visual state (purple border/background)."""
        new_value = bool(selected)
        if new_value == self._selected:
            return
        self._selected = new_value
        self._apply_row_style()
        self.update()

    @property
    def selected(self) -> bool:
        return self._selected

    def _apply_row_style(self) -> None:
        if self._selected:
            style = self._SELECTED_HOVER_STYLE if self._hovered else self._SELECTED_STYLE
        else:
            style = self._HOVER_STYLE if self._hovered else self._BASE_STYLE
        if style == self._applied_style:
            return
        self._applied_style = style
        self.setStyleSheet(style)

    def enterEvent(self, event):
        self._hovered = True
        self._apply_row_style()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_row_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Single left-click = focus/select. Actual activation happens on
        # double-click via :meth:`mouseDoubleClickEvent`.
        if event.button() == Qt.LeftButton:
            self.clicked.emit(int(self.info.get("index", 0)))
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Opening the reader requires a full double-click so accidental
        # single clicks (e.g. while scrolling) don't launch anything.
        if event.button() == Qt.LeftButton:
            self.activated.emit(int(self.info.get("index", 0)))
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        # Right-click — also focus the row so the menu visibly applies to it.
        idx = int(self.info.get("index", 0))
        self.clicked.emit(idx)
        self.menu_requested.emit(idx, event.globalPos())
        event.accept()


# ---------------------------------------------------------------------------
# EPUB Reader — background loader thread
# ---------------------------------------------------------------------------

if _HAS_WEBENGINE:
    class _WheelCapturingView(QWebEngineView):
        """QWebEngineView subclass that surfaces wheel events as a signal.

        Out of the box wheel events on a QWebEngineView are consumed
        by the internal Chromium render widget before they ever reach
        Python. The usual Qt workaround is to install an event filter
        on every child widget the view spawns (they host the actual
        render surface). We do that here plus re-run the install for
        any children added after the first page load (Qt creates those
        lazily). Emits ``wheel_scrolled(delta_y, modifiers)`` for the
        reader dialog to translate into page-turn / zoom behavior.
        """
        wheel_scrolled = Signal(int, object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._install_filter_on_children()

        def childEvent(self, event):
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.ChildAdded:
                child = event.child()
                try:
                    if hasattr(child, "installEventFilter"):
                        child.installEventFilter(self)
                except Exception:
                    pass
            super().childEvent(event)

        def _install_filter_on_children(self):
            from PySide6.QtWidgets import QWidget
            for c in self.findChildren(QWidget):
                try:
                    c.installEventFilter(self)
                except Exception:
                    pass

        def eventFilter(self, obj, event):
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta != 0:
                    self.wheel_scrolled.emit(delta, event.modifiers())
                    # Consume the event so Chromium doesn't also
                    # scroll / zoom the page in parallel.
                    return True
            return super().eventFilter(obj, event)


_EPUB_READER_WEBENGINE_WARMUP_VIEW = None
_EPUB_READER_WEBENGINE_CLEANUP_CONNECTED = False


def prewarm_epub_reader_webengine() -> bool:
    """Start Chromium in a hidden view before the user opens a reader.

    This must run on the Qt GUI thread. Keeping one invisible view alive is
    intentional: Chromium otherwise tears its renderer back down before the
    real reader is constructed, bringing the same cold-start window flash
    back on the context-menu click.
    """
    global _EPUB_READER_WEBENGINE_WARMUP_VIEW
    global _EPUB_READER_WEBENGINE_CLEANUP_CONNECTED
    if not _HAS_WEBENGINE:
        return False
    existing = _EPUB_READER_WEBENGINE_WARMUP_VIEW
    if existing is not None:
        try:
            existing.page()
            return True
        except RuntimeError:
            _EPUB_READER_WEBENGINE_WARMUP_VIEW = None

    app = QApplication.instance()
    if app is None:
        return False
    try:
        view = _WheelCapturingView()
        _configure_epub_reader_web_settings(view)
        view.resize(1, 1)
        view.setUrl(QUrl("about:blank"))
        view.hide()
        _EPUB_READER_WEBENGINE_WARMUP_VIEW = view
        if not _EPUB_READER_WEBENGINE_CLEANUP_CONNECTED:
            app.aboutToQuit.connect(_release_epub_reader_webengine_warmup)
            _EPUB_READER_WEBENGINE_CLEANUP_CONNECTED = True
        return True
    except Exception:
        logger.debug("EPUB reader WebEngine prewarm failed: %s",
                     traceback.format_exc())
        _EPUB_READER_WEBENGINE_WARMUP_VIEW = None
        return False


def _epub_reader_webengine_is_warmed() -> bool:
    """Return whether the retained hidden Chromium view is still valid."""
    view = _EPUB_READER_WEBENGINE_WARMUP_VIEW
    if view is None:
        return False
    try:
        view.page()
        return True
    except RuntimeError:
        return False


def _release_epub_reader_webengine_warmup() -> None:
    """Release the retained warmup view during application shutdown."""
    global _EPUB_READER_WEBENGINE_WARMUP_VIEW
    view = _EPUB_READER_WEBENGINE_WARMUP_VIEW
    _EPUB_READER_WEBENGINE_WARMUP_VIEW = None
    if view is None:
        return
    try:
        view.stop()
    except Exception:
        pass
    try:
        view.deleteLater()
    except Exception:
        pass


class _EpubCacheLoaderThread(EpubCacheLoaderMixin, QThread):
    """Read the pickled EPUB cache off the UI thread.

    ``pickle.load`` on a large cache (hundreds of chapters + embedded
    image bytes) blocks the Qt event loop for up to a second, visibly
    freezing the Halgakos spinner that's supposed to be animating while
    the user waits. Running the read in a dedicated QThread keeps the
    spinner smooth and lets the main thread dispatch paint events
    throughout.

    Emits ``hit(chapters, images, filenames)`` on a successful cache
    read, or ``miss()`` when the cache is absent / invalid / empty and
    the caller should fall back to the full :class:`_EpubLoaderThread`
    re-parse.
    """
    hit = Signal(object, object, list)
    miss = Signal()

    def __init__(self, epub_path: str, show_special_files: bool = True,
                 config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("EpubCacheLoaderThread")
        self._epub_path = epub_path
        self._show_special_files = bool(show_special_files)
        self._config = config or {}
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to reader_doc.EpubCacheLoaderMixin (inherited).


# _reader_overlay_signature moved verbatim to reader_doc (imported above).


class _OverlayMergeThread(OverlayMergeMixin, QThread):
    """Off-UI-thread merge of the reader's loaded chapters against the
    translated-chapter overlay.

    Reading per-chapter translated HTML from disk used to run synchronously in
    :meth:`EpubReaderDialog._on_epub_loaded_from_cache`, making an
    in-progress book with hundreds of translated chapters visibly lag
    the Qt event loop on open. Running them in a dedicated QThread —
    plus a ``ThreadPoolExecutor`` inside for concurrent small reads —
    keeps the UI responsive while the worker hits the disk.

    Emits ``done(overlaid_chapters, merged_images, overlay_applied)``
    once all reads finish. The container payloads ride as ``object``
    (plain Python references) rather than ``list`` / ``dict`` so
    PySide6 doesn't try to marshal them into ``QVariantList`` /
    ``QVariantMap`` — that pathway fails slot lookup on the main
    thread with ``AttributeError: Slot 'EpubReaderDialog::
    _on_overlay_merge_done(QVariantList,QVariantMap,bool)' not
    found``, even though the matching Python method exists. Matches
    the pattern used by :class:`_EpubCacheLoaderThread.hit`.
    """
    done = Signal(object, object, bool)

    def __init__(self, raw_chapters, images, filenames, overlay_map,
                 extra_image_dirs, config: dict | None = None, parent=None,
                 previous_chapters=None):
        super().__init__(parent)
        self.setObjectName("OverlayMergeThread")
        self._raw_chapters = list(raw_chapters or [])
        self._images = dict(images or {})
        self._filenames = list(filenames or [])
        self._overlay = dict(overlay_map or {})
        self._extra_dirs = list(extra_image_dirs or [])
        self._config = dict(config or {})
        self._previous_chapters = list(previous_chapters or [])
        self._read_signature = None
        self._retry_required = False
        self._result_ready = False
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to reader_doc.OverlayMergeMixin (inherited).


class _ReaderImagePreloadThread(ReaderImagePreloadMixin, QThread):
    """Materialize the next chapter's image resources off the GUI thread."""

    done = Signal(str, object)

    def __init__(self, preload_key: str, html_content: str, images: dict,
                 extra_image_dirs, epub_path: str, temp_dir: str, parent=None):
        super().__init__(parent)
        self.setObjectName("EpubReaderImagePreloadThread")
        self._preload_key = str(preload_key or "")
        self._html_content = str(html_content or "")
        self._images = dict(images or {})
        self._extra_dirs = list(extra_image_dirs or [])
        self._epub_path = str(epub_path or "")
        self._temp_dir = str(temp_dir or "")
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to reader_doc.ReaderImagePreloadMixin (inherited).


# _workspace_reader_placeholder moved verbatim to reader_doc (imported above).


class _WorkspaceReaderLoaderThread(WorkspaceReaderLoaderMixin, QThread):
    """Load translated HTML chapter files without requiring an EPUB zip."""

    done = Signal(object, object, list)
    error = Signal(str)

    def __init__(self, manifest: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkspaceReaderLoaderThread")
        self._manifest = dict(manifest or {})
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        try:
            return self._cancelled or self.isInterruptionRequested()
        except RuntimeError:
            return True

    # run moved verbatim to reader_doc.WorkspaceReaderLoaderMixin (inherited).


class _PdfRawSectionLoaderThread(QThread):
    """Materialize one raw bookmark range for a PDF workspace."""

    done = Signal(int, str)
    error = Signal(int, str)

    def __init__(self, row: int, manifest: dict, entry: dict,
                 mode: str, extract_images: bool, parent=None):
        super().__init__(parent)
        self.setObjectName("PdfRawSectionLoaderThread")
        self._row = int(row)
        self._manifest = dict(manifest or {})
        self._entry = dict(entry or {})
        self._mode = str(mode or "fast_semantic")
        self._extract_images = bool(extract_images)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def run(self):
        try:
            if self._cancelled or self.isInterruptionRequested():
                return
            from workspace_reader import ensure_pdf_raw_section

            html_path = ensure_pdf_raw_section(
                self._manifest,
                self._entry,
                mode=self._mode,
                extract_images=self._extract_images,
            )
            if self._cancelled or self.isInterruptionRequested():
                return
            with open(html_path, "r", encoding="utf-8", errors="replace") as stream:
                content = stream.read()
            if not self._cancelled and not self.isInterruptionRequested():
                self.done.emit(self._row, content)
        except Exception as exc:
            if not self._cancelled:
                self.error.emit(self._row, str(exc))


class _EpubSearchThread(EpubSearchMixin, QThread):
    """Build the Search EPUB match list away from the Qt UI thread."""
    results_ready = Signal(int, str, object)
    results_batch_ready = Signal(int, str, object, bool)

    def __init__(self, search_id: int, query: str, chapters,
                 config: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("EpubSearchThread")
        self._search_id = int(search_id)
        self._query = query or ""
        self._chapters = list(chapters or [])
        self._config = dict(config or {})
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to reader_doc.EpubSearchMixin (inherited).


class _EpubSearchLineEdit(QLineEdit):
    """Search input that gives Return/Enter to the EPUB find panel."""
    enterPressed = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.enterPressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _EpubSearchResultDelegate(QStyledItemDelegate):
    """Paint search result rows with lightweight query highlighting."""

    _HIGHLIGHT_BG = QColor("#ffd966")
    _HIGHLIGHT_FG = QColor("#101010")

    def sizeHint(self, option, index):
        return QSize(0, 42)

    def _draw_highlighted_line(self, painter, rect: QRect, text: str,
                               query: str, font: QFont, color: QColor) -> None:
        painter.setFont(font)
        metrics = QFontMetrics(font)
        line = metrics.elidedText(str(text or ""), Qt.ElideRight,
                                  max(0, rect.width()))
        if not line:
            return
        needle = str(query or "").strip()
        if not needle:
            painter.setPen(color)
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, line)
            return

        pattern = re.compile(re.escape(needle), re.IGNORECASE)
        x = rect.left()
        pos = 0
        matched = False
        for match in pattern.finditer(line):
            matched = True
            before = line[pos:match.start()]
            if before:
                width = metrics.horizontalAdvance(before)
                painter.setPen(color)
                painter.drawText(
                    QRect(x, rect.top(), width, rect.height()),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    before,
                )
                x += width
            hit = line[match.start():match.end()]
            width = metrics.horizontalAdvance(hit)
            if width > 0:
                bg_rect = QRect(x - 1, rect.top() + 2,
                                width + 2, max(4, rect.height() - 4))
                painter.setPen(Qt.NoPen)
                painter.setBrush(self._HIGHLIGHT_BG)
                painter.drawRoundedRect(bg_rect, 2, 2)
                painter.setPen(self._HIGHLIGHT_FG)
                painter.drawText(
                    QRect(x, rect.top(), width, rect.height()),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    hit,
                )
                x += width
            pos = match.end()
        tail = line[pos:]
        if tail:
            painter.setPen(color)
            painter.drawText(
                QRect(x, rect.top(), rect.right() - x + 1, rect.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                tail,
            )
        elif not matched:
            painter.setPen(color)
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, line)

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text or ""
        opt.text = ""
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()

        painter.save()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        lines = text.splitlines()
        title = lines[0] if lines else ""
        excerpt = lines[1] if len(lines) > 1 else ""
        data = index.data(Qt.UserRole)
        query = data.get("text", "") if isinstance(data, dict) else ""

        selected = bool(option.state & QStyle.State_Selected)
        base_color = (
            option.palette.highlightedText().color()
            if selected else option.palette.text().color()
        )
        muted = QColor(base_color)
        muted.setAlpha(215 if selected else 225)

        title_font = QFont(option.font)
        excerpt_font = QFont(option.font)
        if excerpt_font.pointSizeF() > 0:
            excerpt_font.setPointSizeF(max(7.5, excerpt_font.pointSizeF() - 0.2))

        rect = option.rect.adjusted(6, 2, -6, -2)
        title_h = QFontMetrics(title_font).height()
        excerpt_h = QFontMetrics(excerpt_font).height()
        title_rect = QRect(rect.left(), rect.top(),
                           rect.width(), title_h + 2)
        excerpt_rect = QRect(rect.left(), rect.top() + title_h + 2,
                             rect.width(), excerpt_h + 2)
        self._draw_highlighted_line(
            painter, title_rect, title, query, title_font, base_color)
        self._draw_highlighted_line(
            painter, excerpt_rect, excerpt, query, excerpt_font, muted)
        painter.restore()


class _EpubSearchResultsList(QListWidget):
    """Search result list where Return/Enter means activate next result."""
    enterPressed = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.enterPressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _EpubLoaderThread(EpubLoaderMixin, QThread):
    """Load the EPUB in a background thread and write result to cache.

    Emitting large binary data (images) through Qt signals across threads
    can crash the GUI.  Instead, write to a cache file and emit a
    lightweight success signal.
    """
    done = Signal()          # success — data available via cache
    error = Signal(str)

    def __init__(self, epub_path: str, parent=None,
                 show_special_files: bool = True,
                 config: dict | None = None):
        super().__init__(parent)
        self.setObjectName("EpubLoaderThread")
        self._epub_path = epub_path
        # When False, spine items flagged by the configured special-file
        # keywords are excluded from the
        # chapter list so the TOC mirrors what the translator would act
        # on with ``translate_special_files`` off. Baked into the cache
        # key so on / off variants don't collide.
        self._show_special_files = bool(show_special_files)
        self._config = config or {}
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        if self._cancelled:
            return True
        try:
            return bool(self.isInterruptionRequested())
        except RuntimeError:
            return True

    # run moved verbatim to reader_doc.EpubLoaderMixin (inherited).


# ---------------------------------------------------------------------------
# EPUB Reader Dialog
# ---------------------------------------------------------------------------

# LAYOUT_SCROLL, LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_ALL, _READER_GT_LANG_CODES, _target_lang_to_google_code, _google_translate_url, _define_url, _chapter_display_numbers moved verbatim to reader_doc (imported above).
# (_google_translate_url / _define_url / _chapter_display_numbers were extracted from
# _open_google_translate / _open_web_define / _finalize_post_load; see DISCREPANCIES U5 "Phase-1 splits".)


def _persist_config_via_parent(widget) -> bool:
    """Walk *widget*'s parent chain looking for a ``save_config`` method and call it.

    Both :class:`EpubLibraryDialog` and :class:`EpubReaderDialog` mutate
    the shared ``config`` dict on close (layout mode, font size, sort
    order, etc.), but that dict only lives in memory until the
    translator's main window persists it to ``config.json``. On a normal
    quit the main window's own save runs first, but users who close the
    app by killing the tab — or who simply want their reader settings
    to outlive a crash — would lose the just-written values. Walking up
    to the :class:`TranslatorGUI` (or whichever parent exposes
    ``save_config``) and triggering a silent save here makes the in-memory
    changes durable as soon as the dialog closes.

    Returns True when a save ran, False when no eligible parent was
    found or the save itself raised. Uses ``show_message=False`` when
    the signature accepts it so the silent-save path doesn't pop an
    unexpected message box.
    """
    try:
        parent = widget.parent() if hasattr(widget, "parent") else None
    except Exception:
        parent = None
    while parent is not None:
        save = getattr(parent, "save_config", None)
        if callable(save):
            try:
                try:
                    save(show_message=False)
                except TypeError:
                    save()
                return True
            except Exception:
                logger.debug("Parent save_config failed: %s",
                             traceback.format_exc())
                return False
        try:
            parent = parent.parent()
        except Exception:
            break
    return False


# _READER_THEMES moved verbatim to reader_doc (imported above).


class EpubReaderDialog(ReaderDocMixin, LiveStreamMixin, QDialog):
    """EPUB reader with chapter navigation, layout modes, and theme support."""

    _SEARCH_DEBOUNCE_MS = 650
    # Let Chromium paint the raw-PDF placeholder before starting MuPDF work.
    # Some image-heavy sections spend long stretches in native extraction;
    # without this small handoff the UI can appear frozen on translated text.
    _RAW_PDF_WORKER_PAINT_DELAY_MS = 120
    # The startup transition keeps a native snapshot above Chromium while its
    # first visible compositor buffers settle, then fades that snapshot away.
    _STARTUP_TRANSITION_SETTLE_MS = 100
    _STARTUP_TRANSITION_FADE_MS = 160

    def __init__(self, epub_path: str, config: dict | None = None, parent=None,
                 initial_chapter: int | None = None,
                 initial_chapter_filename: str | None = None,
                 translated_overlay: dict | None = None,
                 extra_image_dirs: list[str] | None = None,
                 translated_css_dirs: list[str] | None = None,
                 window_title: str | None = None,
                 show_special_files: bool | None = None,
                 alt_epub_path: str | None = None,
                 overlay_provider=None,
                 auto_refresh_interval_ms: int = 3000,
                 workspace_dir: str | None = None,
                 initial_show_raw: bool | None = None,
                 toc_output_dir: str | None = None):
        super().__init__(parent)
        self._config = config or {}
        self._workspace_manifest: dict = {}
        self._workspace_mode = False
        self._workspace_has_raw = False
        if workspace_dir:
            from workspace_reader import build_workspace_reader_manifest

            self._workspace_manifest = build_workspace_reader_manifest(
                workspace_dir,
                source_path=epub_path or None,
            )
            self._workspace_mode = True
            self._workspace_has_raw = (
                self._workspace_manifest.get("source_format") == "pdf"
            )
            epub_path = (
                self._workspace_manifest.get("source_path")
                or self._workspace_manifest.get("workspace")
            )
        self._epub_path = str(epub_path or "")
        # Dual-path mode: when the caller passes ``alt_epub_path`` (the raw
        # source EPUB that pairs with the compiled one), the Show-raw pill
        # swaps ``_epub_path`` between the two and reloads. Used by the
        # Completed tab so users can flip between a compiled translation
        # and its original without dropping out of the reader.
        #
        # The primary file (``_translated_epub_path``) is whatever the
        # caller opened; the alt (``_raw_epub_alt_path``) is the raw
        # counterpart. Either field is "" when not available.
        self._translated_epub_path = str(epub_path or "")
        self._raw_epub_alt_path = ""
        if alt_epub_path:
            alt_abs = os.path.abspath(str(alt_epub_path))
            cur_abs = os.path.abspath(str(epub_path or ""))
            if (os.path.isfile(alt_abs)
                    and os.path.normcase(alt_abs) != os.path.normcase(cur_abs)):
                self._raw_epub_alt_path = alt_abs
        self._chapters: list[tuple[str, str]] = []
        # Parallel (untouched) chapter list kept alongside the possibly-
        # overlaid ``_chapters`` so the Show-raw toolbar toggle can flip
        # between them without re-parsing the EPUB. ``_chapters_overlaid``
        # is the result of applying ``_translated_overlay`` to the raw
        # chapters — when there's no overlay the two lists are identical.
        self._chapters_raw: list[tuple[str, str]] = []
        self._chapters_overlaid: list[tuple[str, str]] = []
        self._chapter_filenames: list[str] = []
        self._chapter_display_numbers: list[int] = []
        self._images: dict[str, bytes] = {}
        # Whether to surface configured non-chapter spine items in the TOC.
        # Caller can force the flag; otherwise we
        # resolve it the same way BookDetailsDialog does, so opening a
        # reader directly from a library card picks up the user's last
        # toggle state instead of silently diverging.
        if show_special_files is None:
            self._show_special_files = _resolve_show_special_files(self._config)
        else:
            self._show_special_files = bool(show_special_files)
        # Show-raw toggle: when True, the TOC + content pane render the
        # raw source chapters instead of the translated overlay. Two
        # modes back this flag:
        #   * Overlay mode: a ``_translated_overlay`` dict was attached
        #     (typical for the In Progress tab) — flipping the toggle
        #     swaps ``_chapters`` between raw and overlaid copies.
        #   * Dual-path mode: a ``alt_epub_path`` was attached (typical
        #     for the Completed tab when the raw source is resolvable)
        #     — flipping the toggle swaps ``_epub_path`` between the
        #     compiled and raw EPUBs and reloads.
        # The pill is hidden when neither applies. Persisted in config.
        self._show_raw = bool(self._config.get('epub_reader_show_raw', False))
        if initial_show_raw is not None:
            self._show_raw = bool(initial_show_raw)
        # If we can satisfy the persisted "raw" state via the dual-path
        # alt (no overlay scenario), start the reader on the raw EPUB so
        # the user sees what they last chose.
        if (self._show_raw and self._raw_epub_alt_path
                and not (translated_overlay or {})):
            self._epub_path = self._raw_epub_alt_path
        # Restore persisted reader settings
        self._font_size = self._config.get('epub_reader_font_size', 14)
        self._line_spacing = self._config.get('epub_reader_line_spacing', 1.8)
        self._theme_index = self._config.get('epub_reader_theme', 0)
        self._font_family = self._config.get('epub_reader_font_family', 'Embedded CSS')
        self._native_toc_enabled = bool(
            self._config.get('epub_reader_native_toc', False))
        layout_key = self._config.get('epub_reader_layout', LAYOUT_SINGLE)
        self._layout_mode = layout_key if layout_key in (LAYOUT_SCROLL, LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_ALL) else LAYOUT_SINGLE
        # Give the native dialog a real backing colour before Windows maps it.
        # Otherwise DWM can briefly expose its default black erase buffer while
        # the first child-widget paint is still being assembled.
        from PySide6.QtGui import QPalette
        initial_palette = self.palette()
        initial_palette.setColor(
            QPalette.Window, QColor(self._get_theme()['bg']))
        self.setPalette(initial_palette)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Optional — caller can request opening at a specific chapter index.
        # The index is clamped to the available chapter range once the EPUB
        # has finished loading (see _on_epub_loaded_from_cache). When the
        # ``initial_chapter_filename`` is provided it takes precedence (index
        # is resolved by matching the filename basename, so the selection is
        # correct regardless of spine/manifest ordering differences).
        self._initial_chapter = initial_chapter if isinstance(initial_chapter, int) and initial_chapter >= 0 else None
        self._initial_chapter_filename = (
            os.path.basename(str(initial_chapter_filename)).lower()
            if initial_chapter_filename else None
        )
        # Optional translated-content overlay for in-progress novels. Keyed by
        # the lowercased basename of the source chapter filename (e.g.
        # ``"chapter0001.xhtml"``) so overlay mapping is robust across the
        # various spine/manifest ordering differences between the reader's
        # loader and the BookDetailsDialog's spine parser. Each value is a
        # dict of the form ``{"path": str, "title": Optional[str]}``.
        self._translated_overlay: dict[str, dict] = {}
        if isinstance(translated_overlay, dict):
            for k, v in translated_overlay.items():
                if not k:
                    continue
                key = os.path.basename(str(k)).lower()
                if not key:
                    continue
                if isinstance(v, str):
                    self._translated_overlay[key] = {"path": v}
                elif isinstance(v, dict) and v.get("path"):
                    self._translated_overlay[key] = dict(v)
        # Extra image search directories (e.g. <output_folder>/images,
        # <output_folder>/translated_images) used to augment the EPUB's own
        # image table so translated chapters can resolve their assets.
        workspace_image_dirs = self._workspace_manifest.get("image_dirs", []) or []
        self._extra_image_dirs: list[str] = list(
            extra_image_dirs or workspace_image_dirs
        )
        # CSS directories from the translated output folder. In overlay mode
        # the active EPUB path still points at the raw source, so translated
        # Embedded CSS must be supplied explicitly.
        workspace_css_dirs = self._workspace_manifest.get("css_dirs", []) or []
        self._translated_css_dirs: list[str] = [
            str(p) for p in (translated_css_dirs or workspace_css_dirs) if p
        ]
        # Native TOC sidecars belong to the translated output workspace, not
        # necessarily the active EPUB path (overlay mode reads the raw EPUB).
        # Prefer the caller's explicit folder, then derive it from the workspace
        # or translated assets, and finally fall back to the EPUB's parent.
        toc_dir = str(toc_output_dir or "").strip()
        if not toc_dir:
            toc_dir = str(self._workspace_manifest.get("workspace") or "").strip()
        if not toc_dir and self._translated_overlay:
            first_overlay = next(iter(self._translated_overlay.values()), {})
            overlay_path = str(first_overlay.get("path") or "")
            if overlay_path:
                toc_dir = os.path.dirname(overlay_path)
        if not toc_dir and self._extra_image_dirs:
            toc_dir = os.path.dirname(str(self._extra_image_dirs[0]))
        if not toc_dir and self._translated_css_dirs:
            css_dir = str(self._translated_css_dirs[0])
            if os.path.isfile(css_dir):
                toc_dir = os.path.dirname(css_dir)
            elif os.path.basename(os.path.normpath(css_dir)).casefold() in {
                "css", "styles"
            }:
                toc_dir = os.path.dirname(css_dir)
            else:
                toc_dir = css_dir
        if not toc_dir and self._epub_path and os.path.isfile(self._epub_path):
            toc_dir = os.path.dirname(self._epub_path)
        self._toc_output_dir = os.path.abspath(toc_dir) if toc_dir else ""
        self._native_toc_source_entries: list[dict[str, str]] = \
            _load_reader_native_toc(self._toc_output_dir, self._epub_path)
        self._native_toc_entries: list[dict] = []
        self._toc_row_to_chapter: list[int] = []
        # Auto-refresh: when a ``overlay_provider`` callable is supplied
        # (typically :meth:`BookDetailsDialog._build_translated_overlay`)
        # the reader ticks a :class:`QTimer` on a short interval, re-runs
        # the provider, and \u2014 if the returned overlay differs from the
        # currently-merged one \u2014 fires a fresh :class:`_OverlayMergeThread`
        # to pick up newly translated chapters. This lets users read an
        # in-progress novel while the translator fills chapters in behind
        # them. The timer is started after the initial load lands (see
        # :meth:`_finalize_post_load`) so auto-refresh can't race the
        # first paint, and stops in :meth:`closeEvent`.
        self._overlay_provider = overlay_provider if callable(overlay_provider) else None
        self._auto_refresh_interval_ms = max(500, int(auto_refresh_interval_ms or 0))
        self._overlay_refresh_timer: QTimer | None = None
        self._overlay_loaded_signature = None
        self._overlay_retry_required = False
        self._overlay_merge_pending = False
        self._current_row = 0
        self._current_page = 0  # viewport-based page for single/double page modes
        # Page-count cache + currently-rendered-chapter sentinel need to
        # exist BEFORE ``_setup_ui`` runs — the QWebEngineView widgets
        # created there kick off an initial ``setUrl("about:blank")``
        # asynchronously, which can fire ``loadFinished`` and reach
        # :meth:`_finalize_single_page` before
        # :meth:`_finalize_post_load` initializes these attributes.
        # Leaving them out causes the about:blank load to crash with
        # ``AttributeError: 'EpubReaderDialog' object has no attribute
        # '_chapter_page_cache'``.
        self._chapter_page_cache: dict[int, int] = {}
        self._loaded_chapter: int = -1
        self._closing = False
        self._cache_loader_thread: _EpubCacheLoaderThread | None = None
        self._loader_thread: _EpubLoaderThread | None = None
        self._overlay_thread: _OverlayMergeThread | None = None
        self._workspace_loader_thread: _WorkspaceReaderLoaderThread | None = None
        self._workspace_raw_thread: _PdfRawSectionLoaderThread | None = None
        self._image_preload_thread: _ReaderImagePreloadThread | None = None
        self._epub_load_image_preload_thread: _ReaderImagePreloadThread | None = None
        self._pending_epub_load = None
        self._pending_image_chapter_activation: tuple[str, int] | None = None
        self._image_preload_active_key = ""
        self._image_preload_pending = False
        self._preloaded_chapter_keys: set[str] = set()
        self._preloaded_image_resources: dict[str, dict] = {}
        self._processed_html_cache: dict[str, str] = {}
        self._image_sizeable_cache: dict[str, bool] = {}
        self._image_cache_generation = 0
        self._image_resource_signature: tuple = ()
        self._epub_image_zip = None
        self._epub_image_zip_path = ""
        self._epub_image_zip_names: dict[str, str] = {}
        # Completed books can switch between two standalone EPUB files. Keep
        # the already-loaded state for each path so returning to a flavor does
        # not re-read its cache, rebuild image metadata, or re-process chapter
        # HTML. The key includes the same source/settings fingerprint as the
        # persistent EPUB cache, so replacing either file naturally misses.
        self._dual_path_reader_states: dict[str, dict] = {}
        self._workspace_raw_ready: set[int] = set()
        self._workspace_raw_pending_row: int | None = None
        self._raw_toggle_in_flight = False
        self._reader_render_generation = 0
        # QWebEngineView starts a Chromium renderer the first time it is
        # constructed. On packaged builds that can take several seconds, so
        # keep it out of __init__: callers can show the lightweight loading
        # shell immediately and the web views are attached after its first
        # native paint.
        self._reader_views_ready = False
        self._reader_init_queued = False
        # Keep the lightweight loading shell on screen until Chromium has
        # completed the first real chapter render (including the hidden
        # scroll-to-paginated prime pass). Revealing the content any earlier
        # exposes the intermediate layout and causes an opening flicker.
        self._reader_startup_pending = True
        self._reader_reveal_queued = False
        self._reader_transition_overlay = None
        self._reader_transition_animation = None

        # ── Live single-chapter translation state ──────────────────────
        # The toolbar Translate button streams the chapter's translation
        # straight into the reader: raw streamed text replaces the page
        # while thinking/status lines collect in a discreet collapsible
        # pane. ``_live_log_queue`` is filled from worker threads via the
        # main GUI's log-listener hook and drained on a GUI-side QTimer.
        self._live_translate_active = False
        self._live_log_queue: deque = deque()
        self._live_drain_timer: QTimer | None = None
        self._live_poll_timer: QTimer | None = None
        self._live_panel: QWidget | None = None
        self._live_content_view = None
        self._live_think_view = None
        self._live_think_toggle = None
        self._live_status_label = None
        self._live_content_buf = ""
        self._live_think_pending = ""
        self._live_log_pending = ""
        self._live_dirty = False
        self._live_in_thinking = False
        self._live_streaming_text = False
        self._live_gui_ref = None
        self._live_listener = None
        self._live_target_row: int | None = None
        self._live_epub_path = ""
        self._live_chapter_file = ""
        self._live_follow_stream = True
        self._live_scroll_guard = False

        title_text = (
            window_title
            or self._workspace_manifest.get("title")
            or os.path.splitext(os.path.basename(str(epub_path or "")))[0]
        )
        self._window_title_text = title_text
        self.setWindowTitle(f"\U0001f4d6 {title_text}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        # Ratio-based sizing relative to screen
        screen = self.screen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.60), int(avail.height() * 0.76))
            self.setMinimumSize(int(avail.width() * 0.4), int(avail.height() * 0.4))
        else:
            self.resize(1020, 760)
            self.setMinimumSize(600, 400)

        # Chromium must be initialized while this dialog is still hidden.
        # Attaching the first QWebEngineView from showEvent temporarily unmaps
        # the native dialog on Windows, which looks like the reader opens,
        # closes, and then opens again. The retained 1x1 view starts the engine
        # without exposing a native surface to the user.
        if _HAS_WEBENGINE and not _epub_reader_webengine_is_warmed():
            prewarm_epub_reader_webengine()

        self._setup_ui()
        # Restore combo positions after UI is built
        modes = [LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_SCROLL, LAYOUT_ALL]
        if self._layout_mode in modes:
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(modes.index(self._layout_mode))
            self._layout_combo.blockSignals(False)
        if 0 <= self._theme_index < len(_READER_THEMES):
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(self._theme_index)
            self._theme_combo.blockSignals(False)
        # Restore spacing combo
        spacing_str = str(self._line_spacing)
        idx = self._spacing_combo.findText(spacing_str)
        self._spacing_combo.blockSignals(True)
        if idx >= 0:
            self._spacing_combo.setCurrentIndex(idx)
        else:
            self._spacing_combo.setCurrentText(spacing_str)
        self._spacing_combo.blockSignals(False)
        # Restore font family combo
        fam_idx = self._font_combo.findText(self._font_family)
        self._font_combo.blockSignals(True)
        if fam_idx >= 0:
            self._font_combo.setCurrentIndex(fam_idx)
        else:
            self._font_combo.setCurrentText(self._font_family)
        self._font_combo.blockSignals(False)
        # The EPUB loader starts together with the deferred reader views from
        # showEvent. Starting the spinner here gives the lightweight shell
        # useful feedback during a Chromium cold start.
        self._toolbar_widget.hide()
        self._loading_widget.show()
        self._content_widget.hide()
        self._spin_timer.start()
        # Always install WebEngine-backed panes before the native reader window
        # is shown. The deferred showEvent path remains only for the lightweight
        # QTextBrowser fallback used when QtWebEngine is unavailable.
        if _HAS_WEBENGINE or _epub_reader_webengine_is_warmed():
            self._initialize_reader_views()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._root_layout = root

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.setSpacing(6)

        title_lbl = QLabel(f"\U0001f4d6 {getattr(self, '_window_title_text', os.path.splitext(os.path.basename(self._epub_path))[0])}")
        title_lbl.setStyleSheet("font-size: 11pt; font-weight: bold; color: #e0e0e0;")
        title_lbl.setMaximumWidth(400)
        toolbar.addWidget(title_lbl)

        # TOC toggle button
        self._toc_btn = self._make_toolbar_btn("📑", "Toggle table of contents")
        self._toc_btn.clicked.connect(self._toggle_toc)
        toolbar.addWidget(self._toc_btn)

        self._native_toc_btn = QPushButton("📑 TOC")
        self._native_toc_btn.setToolTip(
            "Use TOC.txt or toc.ncx entries in the sidebar instead of\n"
            "titles extracted from each chapter's HTML."
        )
        self._native_toc_btn.setFixedHeight(26)
        self._native_toc_btn.setCursor(Qt.PointingHandCursor)
        self._native_toc_btn.setCheckable(True)
        self._native_toc_btn.setChecked(self._native_toc_enabled)
        self._native_toc_btn.setEnabled(bool(self._native_toc_source_entries))
        self._native_toc_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
            QPushButton:disabled { color: #666675; border-color: #303044; }
        """)
        self._native_toc_btn.setAutoDefault(False)
        self._native_toc_btn.setDefault(False)
        self._native_toc_btn.toggled.connect(self._on_native_toc_toggled)
        toolbar.addWidget(self._native_toc_btn)

        toolbar.addStretch()

        # Translate-current-chapter button: keep this action at the left edge
        # of the reader controls, immediately before the page-layout selector.
        # It runs a single-chapter translation through the main GUI and
        # replaces the page with raw streaming output while it runs. While a
        # run is active the button flips between the live and normal reader.
        self._translate_btn = QPushButton("\U0001f310  Translate")
        self._translate_btn.setToolTip(
            "Translate the current chapter now.\n"
            "Only this chapter's HTML is extracted (full extraction is\n"
            "skipped) and the live streaming output replaces the page\n"
            "while the translation runs."
        )
        self._translate_btn.setFixedHeight(26)
        self._translate_btn.setCursor(Qt.PointingHandCursor)
        self._translate_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
        """)
        self._translate_btn.setAutoDefault(False)
        self._translate_btn.setDefault(False)
        self._translate_btn.clicked.connect(self._on_translate_current_chapter)
        toolbar.addWidget(self._translate_btn)

        toolbar.addSpacing(6)

        # Layout mode dropdown
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["📄 Single Page", "📖 Double Page", "📜 Scroll", "📃 Scroll All"])
        self._layout_combo.setFixedWidth(100)
        self._layout_combo.setCursor(Qt.PointingHandCursor)
        self._layout_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 8px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 0px; }
            QComboBox::down-arrow { width: 0px; height: 0px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e; font-size: 8.5pt;
            }
        """)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        self._layout_combo.setFocusPolicy(Qt.StrongFocus)
        self._layout_combo.installEventFilter(self)
        toolbar.addWidget(self._layout_combo)

        toolbar.addSpacing(6)

        # Show-raw toggle: flips the reader between translated overlay
        # and raw source content on the fly. Hidden until the loader
        # confirms a translated overlay exists (there's nothing to
        # toggle against for a plain EPUB open). Visually matches the
        # "Raw titles" pill on the Library toolbar so the two surfaces
        # read as the same control in different locations.
        self._raw_btn = QPushButton("\U0001f524  Raw")
        self._raw_btn.setToolTip(
            "Show the raw source-language content instead of the translated \n"
            "overlay. Useful for comparing a chapter against its original."
        )
        self._raw_btn.setFixedHeight(26)
        self._raw_btn.setCursor(Qt.PointingHandCursor)
        self._raw_btn.setCheckable(True)
        self._raw_btn.setChecked(self._show_raw)
        self._raw_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 2px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        self._raw_btn.setAutoDefault(False)
        self._raw_btn.setDefault(False)
        self._raw_btn.toggled.connect(self._on_show_raw_toggled)
        # Hidden until we know whether there's a translated overlay to flip
        # against — :meth:`_on_epub_loaded_from_cache` reveals it when the
        # overlaid list actually differs from the raw one.
        self._raw_btn.hide()
        toolbar.addWidget(self._raw_btn)

        toolbar.addSpacing(8)

        # Line spacing dropdown
        spacing_lbl = QLabel("↕")
        spacing_lbl.setStyleSheet("color: #888; font-size: 11pt;")
        spacing_lbl.setToolTip("Line spacing")
        toolbar.addWidget(spacing_lbl)

        self._spacing_combo = QComboBox()
        self._spacing_combo.setEditable(True)
        self._spacing_combo.addItems(["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2", "2.4", "2.6", "2.8", "3.0"])
        self._spacing_combo.setFixedWidth(66)
        self._spacing_combo.setCursor(Qt.PointingHandCursor)
        _spacing_icon_path = (_find_halgakos_icon() or "").replace("\\", "/")
        self._spacing_combo.setStyleSheet(f"""
            QComboBox {{
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 2px 4px; padding-right: 18px;
            }}
            QComboBox:hover {{ border-color: #6c63ff; }}
            QComboBox::drop-down {{
                border: none; width: 18px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }}
            QComboBox::down-arrow {{
                image: url("{_spacing_icon_path}");
                width: 12px; height: 12px;
            }}
            QComboBox QAbstractItemView {{
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }}
        """)
        self._spacing_combo.activated.connect(lambda idx: self._on_spacing_changed(self._spacing_combo.itemText(idx)))
        self._spacing_combo.lineEdit().editingFinished.connect(lambda: self._on_spacing_changed(self._spacing_combo.currentText()))
        self._spacing_combo.setFocusPolicy(Qt.StrongFocus)
        self._spacing_combo.installEventFilter(self)
        toolbar.addWidget(self._spacing_combo)

        toolbar.addSpacing(8)

        # Font family dropdown (system fonts, editable for custom families)
        font_family_lbl = QLabel("𝑨")
        font_family_lbl.setStyleSheet("color: #888; font-size: 12pt; padding: 0 2px;")
        font_family_lbl.setToolTip("Font family")
        toolbar.addWidget(font_family_lbl)

        self._font_combo = QComboBox()
        self._font_combo.setEditable(True)
        self._font_combo.setInsertPolicy(QComboBox.NoInsert)
        self._font_combo.setFixedWidth(130)
        self._font_combo.setCursor(Qt.PointingHandCursor)
        self._font_combo.setToolTip("Text font family")
        # First item: use the EPUB's own embedded CSS (fonts, layout, etc.)
        self._font_combo.addItem("Embedded CSS")
        self._font_combo.insertSeparator(self._font_combo.count())
        # Populate with smoothly-scalable system font families (same pattern
        # used in review_dialog.py). Pin common reading fonts to the top.
        try:
            from PySide6.QtGui import QFontDatabase
            families = sorted({
                f for f in QFontDatabase.families()
                if QFontDatabase.isSmoothlyScalable(f)
            })
        except Exception:
            families = []
        _pinned = ["Georgia", "Noto Serif", "Segoe UI", "Cambria", "Times New Roman",
                   "Garamond", "Palatino Linotype", "Arial", "Verdana", "Tahoma",
                   "Consolas"]
        _added: set[str] = set()
        for fam in _pinned:
            if not families or fam in families:
                self._font_combo.addItem(fam)
                _added.add(fam)
        if families:
            self._font_combo.insertSeparator(self._font_combo.count())
            for fam in families:
                if fam not in _added:
                    self._font_combo.addItem(fam)
        _font_icon_path = (_find_halgakos_icon() or "").replace("\\", "/")
        self._font_combo.setStyleSheet(f"""
            QComboBox {{
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 2px 4px 2px 6px; padding-right: 20px;
            }}
            QComboBox:hover {{ border-color: #6c63ff; }}
            QComboBox::drop-down {{
                border: none; background: transparent;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: url("{_font_icon_path}");
                width: 12px; height: 12px;
            }}
            QComboBox QLineEdit {{
                background: transparent; color: #e0e0e0; border: none;
                padding: 0px; margin: 0px; font-size: 8.5pt;
            }}
            QComboBox QAbstractItemView {{
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }}
        """)
        self._font_combo.setFixedWidth(140)
        self._font_combo.activated.connect(
            lambda idx: self._on_font_family_changed(self._font_combo.itemText(idx)))
        self._font_combo.lineEdit().editingFinished.connect(
            lambda: self._on_font_family_changed(self._font_combo.currentText()))
        self._font_combo.setFocusPolicy(Qt.StrongFocus)
        self._font_combo.installEventFilter(self)
        toolbar.addWidget(self._font_combo)

        toolbar.addSpacing(8)

        # Font size controls
        font_down = self._make_toolbar_btn("A−", "Decrease font size", width=42)
        font_down.clicked.connect(lambda: self._change_font_size(-1))
        toolbar.addWidget(font_down)

        self._font_label = QLabel(f"{self._font_size}pt")
        self._font_label.setStyleSheet("color: #888; font-size: 8.5pt; padding: 0 2px;")
        toolbar.addWidget(self._font_label)

        font_up = self._make_toolbar_btn("A+", "Increase font size", width=42)
        font_up.clicked.connect(lambda: self._change_font_size(1))
        toolbar.addWidget(font_up)

        toolbar.addSpacing(4)

        # Theme dropdown
        self._theme_combo = QComboBox()
        self._theme_combo.addItems([t["name"] for t in _READER_THEMES])
        self._theme_combo.setCurrentIndex(0)
        self._theme_combo.setFixedWidth(68)
        self._theme_combo.setCursor(Qt.PointingHandCursor)
        self._theme_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 8.5pt; padding: 3px 8px;
            }
            QComboBox:hover { border-color: #6c63ff; }
            QComboBox::drop-down { border: none; width: 0px; }
            QComboBox::down-arrow { width: 0px; height: 0px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e0e0e0; selection-background-color: #3a3a5e;
                border: 1px solid #3a3a5e;
            }
        """)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self._theme_combo.setFocusPolicy(Qt.StrongFocus)
        self._theme_combo.installEventFilter(self)
        toolbar.addWidget(self._theme_combo)

        self._toolbar_widget = QWidget()
        self._toolbar_widget.setLayout(toolbar)
        self._toolbar_widget.setStyleSheet("background: #1e1e1e; border-bottom: 1px solid #333333;")
        root.addWidget(self._toolbar_widget)

        # ── Loading indicator ──
        self._loading_widget = QWidget()
        loading_layout = QVBoxLayout(self._loading_widget)
        loading_layout.setAlignment(Qt.AlignCenter)
        # Spinning icon via QTimer rotation
        self._spin_label = QLabel()
        icon_path = _find_halgakos_icon()
        if icon_path:
            pm = QPixmap(icon_path)
            if not pm.isNull():
                self._spin_pixmap = pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._spin_label.setPixmap(self._spin_pixmap)
            else:
                self._spin_pixmap = None
        else:
            self._spin_pixmap = None
        if not self._spin_pixmap:
            self._spin_label.setText("📖")
            self._spin_label.setStyleSheet("font-size: 32pt;")
        self._spin_label.setAlignment(Qt.AlignCenter)
        self._spin_label.setFixedSize(72, 72)
        loading_layout.addWidget(self._spin_label, 0, Qt.AlignCenter)
        self._spin_angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(25)  # ~40 fps
        # Wrap in a lambda so PySide6 routes the call directly instead
        # of going through Qt's meta-object slot-lookup — certain
        # PySide6 builds raise ``AttributeError: Slot 'EpubReaderDialog::
        # _rotate_spinner()' not found`` on ``QTimer.timeout`` dispatch
        # even when the method exists and is ``@Slot()``-decorated.
        self._spin_timer.timeout.connect(lambda: self._rotate_spinner())
        loading_text = QLabel(
            "Loading PDF sections\u2026"
            if self._workspace_mode else "Loading EPUB\u2026"
        )
        loading_text.setAlignment(Qt.AlignCenter)
        loading_text.setStyleSheet("color: #888; font-size: 11pt; padding-top: 8px;")
        loading_layout.addWidget(loading_text)
        # Indeterminate progress bar
        from PySide6.QtWidgets import QProgressBar
        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)  # indeterminate
        self._loading_bar.setFixedWidth(220)
        self._loading_bar.setFixedHeight(6)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setStyleSheet("""
            QProgressBar { background: #2a2a2a; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #6c63ff; border-radius: 3px; }
        """)
        loading_layout.addWidget(self._loading_bar, 0, Qt.AlignCenter)
        self._loading_widget.setStyleSheet("background: #1e1e1e;")
        root.addWidget(self._loading_widget)

        # ── Main content (hidden until loaded) ──
        self._content_widget = QWidget()
        content_layout = QHBoxLayout(self._content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._reader_column = QWidget()
        reader_layout = QVBoxLayout(self._reader_column)
        reader_layout.setContentsMargins(0, 0, 0, 0)
        reader_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        # TOC sidebar
        self._toc_list = QListWidget()
        self._toc_list.setMinimumWidth(100)
        self._toc_list.resize(220, self._toc_list.height())
        # Word-wrap long chapter titles onto multiple lines so the TOC
        # uses the extra vertical space instead of eliding with "\u2026".
        # Combined with ``ElideNone``, titles that don't fit the sidebar
        # width fall onto a second / third line (and the stylesheet
        # below adds a row divider so wrapped entries still read as
        # separate rows). ``Adjust`` resize mode + non-uniform item
        # sizes lets Qt recompute each row's height when the splitter
        # is dragged narrower or wider.
        from PySide6.QtWidgets import QListView
        self._toc_list.setWordWrap(True)
        self._toc_list.setTextElideMode(Qt.ElideNone)
        self._toc_list.setUniformItemSizes(False)
        self._toc_list.setResizeMode(QListView.Adjust)
        self._toc_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._toc_list.currentRowChanged.connect(self._on_chapter_selected)
        splitter.addWidget(self._toc_list)

        # Reader area — single browser for scroll/single, an HBox for
        # double-page. The actual browser widgets are installed after the
        # dialog is visible; placeholders keep the final layout and stack
        # indexes stable meanwhile.
        self._reader_stack = QStackedWidget()

        # Page 0: single reader (for scroll, single page, all-scroll)
        self._single_reader_container = QWidget()
        self._single_reader_layout = QVBoxLayout(self._single_reader_container)
        self._single_reader_layout.setContentsMargins(0, 0, 0, 0)
        self._single_reader_placeholder = QWidget()
        self._single_reader_layout.addWidget(self._single_reader_placeholder)
        self._reader = None
        self._reader_stack.addWidget(self._single_reader_container)

        # Page 1: double-page layout (two browsers side by side)
        double_widget = QWidget()
        self._double_widget = double_widget  # keep ref for styling
        double_layout = QHBoxLayout(double_widget)
        double_layout.setContentsMargins(0, 0, 0, 0)
        double_layout.setSpacing(2)
        self._double_layout = double_layout
        self._reader_left_placeholder = QWidget()
        self._reader_right_placeholder = QWidget()
        self._reader_left = None
        self._reader_right = None
        double_layout.addWidget(self._reader_left_placeholder)
        double_layout.addWidget(self._reader_right_placeholder)
        self._reader_stack.addWidget(double_widget)

        splitter.addWidget(self._reader_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 680])
        self._splitter = splitter  # keep ref for styling
        reader_layout.addWidget(splitter, 1)

        # Bottom nav bar for single-page mode
        self._nav_bar = QWidget()
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(10, 4, 10, 6)
        nav_layout.setSpacing(8)
        self._prev_btn = QPushButton("◀  Previous")
        self._prev_btn.setCursor(Qt.PointingHandCursor)
        self._prev_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #c0c0d0; font-size: 9pt; padding: 6px 16px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:disabled { color: #555; }
        """)
        self._prev_btn.clicked.connect(self._prev_chapter)
        nav_layout.addWidget(self._prev_btn)
        nav_layout.addStretch()
        self._page_label = QLabel("")
        self._page_label.setStyleSheet("color: #888; font-size: 9pt;")
        nav_layout.addWidget(self._page_label)
        nav_layout.addStretch()
        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setCursor(Qt.PointingHandCursor)
        self._next_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #c0c0d0; font-size: 9pt; padding: 6px 16px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:disabled { color: #555; }
        """)
        self._next_btn.clicked.connect(self._next_chapter)
        nav_layout.addWidget(self._next_btn)
        self._nav_bar.hide()
        reader_layout.addWidget(self._nav_bar)
        content_layout.addWidget(self._reader_column, 1)

        self._search_detached = bool(
            self._config.get('epub_reader_search_detached', False))
        self._search_panel = QFrame()
        self._search_panel.setObjectName("readerSearchPanel")
        self._search_panel.setMinimumWidth(300)
        self._search_panel.setMaximumWidth(360)
        self._search_panel_layout = QVBoxLayout(self._search_panel)
        self._search_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._search_panel_layout.setSpacing(0)
        self._search_panel.hide()
        content_layout.addWidget(self._search_panel)

        self._content_widget.hide()
        root.addWidget(self._content_widget, 1)

        # Search bar (hidden by default)
        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("Search across EPUB... (Enter = next, Esc = close)")
        self._search_bar.hide()
        self._search_bar.textChanged.connect(self._on_search_text_changed)
        self._search_bar.returnPressed.connect(self._on_search_next)
        self._search_chapter_idx = 0  # track which chapter we last searched
        self._search_dialog_generation = 0
        self._search_workers = []
        self._search_pending_render_rows = []
        self._search_pending_render_query = ""
        self._search_debounce_timer = QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.timeout.connect(
            self._start_search_dialog_worker)
        self._search_render_timer = QTimer(self)
        self._search_render_timer.setInterval(0)
        self._search_render_timer.timeout.connect(
            self._render_search_result_batch)
        root.addWidget(self._search_bar)

        self._apply_reader_style()

        # Shortcuts (work regardless of child focus)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_chapter)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_chapter)
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self._change_font_size(1))
        QShortcut(QKeySequence("Ctrl++"), self, lambda: self._change_font_size(1))
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self._change_font_size(-1))
        QShortcut(QKeySequence("Ctrl+F"), self, self._toggle_search)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._close_search)

    def _make_reader_widget(self):
        """Create one reader pane after the loading shell is visible."""
        if _HAS_WEBENGINE:
            # Use the wheel-capturing subclass so mouse wheel events can drive
            # page navigation / font zoom instead of being swallowed by the
            # internal Chromium scroller.
            widget = _WheelCapturingView()
            _configure_epub_reader_web_settings(widget)
            # Set the compositor clear colour before the first navigation.
            # Doing this after about:blank allowed one default-black surface
            # to be presented during initial attachment on Windows.
            widget.page().setBackgroundColor(QColor(self._get_theme()['bg']))
            widget.setUrl(QUrl("about:blank"))
            widget.wheel_scrolled.connect(self._on_reader_wheel_scrolled)
        else:
            widget = QTextBrowser()
            widget.setOpenExternalLinks(False)
            widget.setOpenLinks(False)
        # Right-click menu: Google Translate + Web Search for the current
        # selection. Applies to every reader pane so double-page mode matches.
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos, browser=widget:
                self._show_reader_context_menu(browser, pos))
        return widget

    @Slot()
    def _initialize_reader_views(self):
        """Attach expensive browser panes, then begin the threaded EPUB load."""
        self._reader_init_queued = False
        if self._closing or self._reader_views_ready:
            return

        self._reader = self._make_reader_widget()
        if _HAS_WEBENGINE:
            # The current double-page layout uses two CSS columns in this one
            # view, so only the primary pane needs to exist or receive load
            # callbacks. The legacy left/right stack page remains a cheap
            # placeholder for index compatibility.
            self._reader.loadFinished.connect(self._on_reader_load_finished)

        self._single_reader_layout.replaceWidget(
            self._single_reader_placeholder,
            self._reader,
        )
        self._single_reader_placeholder.hide()
        self._single_reader_placeholder.deleteLater()

        self._reader_views_ready = True
        # The shell was fully themed and put into its loading state before it
        # was shown. Reapplying the top-level stylesheet or hide/showing these
        # widgets here forces a second native repaint that looks like the
        # reader closed and reopened. The reader content is styled when the
        # first chapter is rendered.
        self._start_loading(preserve_shell=True)

    # ── Event filter (block wheel scroll in paginated modes) ──────────────

    def eventFilter(self, obj, event):
        """Block wheel on combos, handle Ctrl+Wheel for font zoom."""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                if obj is getattr(self, "_search_dialog_query", None):
                    self._search_enter_next()
                    return True
                results = getattr(self, "_search_dialog_results", None)
                results_viewport = results.viewport() if results is not None else None
                if obj is results or obj is results_viewport:
                    self._search_enter_next()
                    return True
        if event.type() == QEvent.Wheel:
            if isinstance(obj, QComboBox):
                return True  # block wheel on all toolbar combos
            # Ctrl+Wheel = font zoom
            if event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self._change_font_size(1)
                else:
                    self._change_font_size(-1)
                return True
        return super().eventFilter(obj, event)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _make_toolbar_btn(self, text, tooltip, width=36):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(width, 28)
        btn.setCursor(Qt.PointingHandCursor)
        # Prevent Enter key from activating toolbar buttons (conflicts
        # with the search bar's returnPressed signal in QDialog).
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setStyleSheet(
            "QPushButton { background: #2a2a3e; border-radius: 4px; color: #e0e0e0; "
            "font-size: 10pt; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #3a3a5e; }")
        return btn

    @Slot()
    def _rotate_spinner(self):
        """Rotate the Halgakos icon by 15° each tick.

        Decorated with ``@Slot()`` so PySide6 registers it in the Qt
        meta-object system. Without the decorator, ``QTimer.timeout``
        can fail to resolve the slot by name at invocation time
        (``AttributeError: Slot 'EpubReaderDialog::_rotate_spinner()'
        not found``).
        """
        if self._spin_pixmap:
            self._spin_angle = (self._spin_angle + 15) % 360
            t = QTransform().rotate(self._spin_angle)
            rotated = self._spin_pixmap.transformed(t, Qt.FastTransformation)
            self._spin_label.setPixmap(rotated)

    # ── Loading ───────────────────────────────────────────────────

    def _start_loading(self, preserve_shell=False):
        if getattr(self, "_closing", False):
            return
        if not preserve_shell:
            self._toolbar_widget.hide()
            self._loading_widget.show()
            self._content_widget.hide()
            self._spin_angle = 0
            self._spin_timer.start()
        if self._workspace_mode:
            previous = getattr(self, "_workspace_loader_thread", None)
            if previous is not None:
                _stop_qthread_safely(
                    previous,
                    timeout_ms=250,
                    signal_names=("done", "error"),
                )
            thread = _WorkspaceReaderLoaderThread(
                self._workspace_manifest,
                parent=self,
            )
            thread.done.connect(
                lambda raw, translated, filenames:
                    self._on_workspace_loaded(raw, translated, filenames)
            )
            thread.error.connect(lambda message: self._on_epub_error(message))
            self._workspace_loader_thread = thread
            thread.start()
            return
        # Probe the cache in a worker thread so the pickle.load doesn't
        # freeze the spinner. The worker emits ``hit`` with the cached
        # data on success, or ``miss`` to fall through to the full
        # :class:`_EpubLoaderThread` re-parse.
        prev_cache = getattr(self, "_cache_loader_thread", None)
        if prev_cache is not None:
            _stop_qthread_safely(
                prev_cache, timeout_ms=250,
                signal_names=("hit", "miss"),
            )
        self._cache_loader_thread = _EpubCacheLoaderThread(
            self._epub_path,
            show_special_files=self._show_special_files,
            config=self._config,
            parent=self,
        )
        self._cache_loader_thread.hit.connect(self._on_cache_hit)
        self._cache_loader_thread.miss.connect(self._on_cache_miss)
        self._cache_loader_thread.start()

    @Slot(object, object, list)
    def _on_workspace_loaded(self, raw_chapters, translated_chapters,
                             filenames):
        """Install a translation-folder HTML manifest in the shared reader."""
        if getattr(self, "_closing", False):
            return
        self._finalize_post_load(
            list(raw_chapters or []),
            list(translated_chapters or []),
            {},
            bool(self._workspace_has_raw),
            list(filenames or []),
        )

    @Slot(object, object, list)
    def _on_cache_hit(self, chapters, images, filenames):
        if getattr(self, "_closing", False):
            return
        """Cache worker delivered hit — finalize or fall back to reparse."""
        filenames = list(filenames or [])
        # Older caches may not carry filenames. If an overlay is requested
        # but filenames are missing we MUST reparse the EPUB so the overlay
        # can map correctly — otherwise we'd silently keep the raw text.
        if self._translated_overlay and not filenames:
            self._on_cache_miss()
            return
        self._on_epub_loaded_from_cache(chapters, images, filenames)

    @Slot()
    def _on_cache_miss(self):
        if getattr(self, "_closing", False):
            return
        """Cache worker had nothing usable — kick off the full reparse."""
        self._loader_thread = _EpubLoaderThread(
            self._epub_path, self,
            show_special_files=self._show_special_files,
            config=self._config,
        )
        self._loader_thread.done.connect(lambda: self._on_loader_done())
        self._loader_thread.error.connect(lambda msg: self._on_epub_error(msg))
        self._loader_thread.start()

    @Slot()
    def _on_loader_done(self):
        if getattr(self, "_closing", False):
            return
        """Loader finished — re-read data from cache in a worker so the
        pickle.load doesn't re-freeze the spinner right before the
        reader is about to render.
        """
        prev_cache = getattr(self, "_cache_loader_thread", None)
        if prev_cache is not None:
            _stop_qthread_safely(
                prev_cache, timeout_ms=250,
                signal_names=("hit", "miss"),
            )
        reader_thread = _EpubCacheLoaderThread(
            self._epub_path,
            show_special_files=self._show_special_files,
            config=self._config,
            parent=self,
        )
        # Post-reparse cache-miss shouldn't re-trigger the parser (that
        # would loop forever) — surface it as an error instead.
        reader_thread.hit.connect(self._on_cache_hit)
        reader_thread.miss.connect(
            lambda: self._on_epub_error(
                "Failed to read EPUB cache after loading."))
        self._cache_loader_thread = reader_thread
        reader_thread.start()

    def _on_epub_loaded_from_cache(self, chapters, images, filenames=None):
        if getattr(self, "_closing", False):
            return
        filenames = list(filenames or [])
        raw_chapters = list(chapters or [])
        # Fast path: nothing to merge. Extra image directories are resolved
        # lazily per chapter and no longer require a startup worker.
        if not self._translated_overlay:
            self._overlay_loaded_signature = ()
            self._overlay_retry_required = False
            self._overlay_merge_pending = False
            if self._start_epub_load_image_preload(
                    raw_chapters, images or {}, filenames):
                return
            self._finalize_post_load(raw_chapters, raw_chapters, images or {},
                                     False, filenames)
            return
        # Heavy path: hand the per-chapter file reads + image-dir scan to
        # :class:`_OverlayMergeThread` so the Qt event loop stays free.
        # Stash raw + filenames on self so the done signal's slot can
        # seed the finalizer without having to round-trip them through
        # the worker (images are modified there, chapters are produced).
        self._pending_raw_chapters = raw_chapters
        self._pending_filenames = filenames
        # Cancel any in-flight worker from a previous load (e.g. the
        # user closed + reopened quickly, or the Raw toggle fired a
        # dual-path reload before the previous merge finished).
        prev = getattr(self, "_overlay_thread", None)
        if prev is not None:
            _stop_qthread_safely(
                prev, timeout_ms=250,
                signal_names=("done",),
            )
        self._overlay_thread = _OverlayMergeThread(
            raw_chapters=raw_chapters,
            images=images or {},
            filenames=filenames,
            overlay_map=self._translated_overlay,
            extra_image_dirs=self._extra_image_dirs,
            config=self._config,
            parent=self,
        )
        worker = self._overlay_thread
        self._overlay_merge_pending = True
        # Lambda-wrap so PySide6 routes the delivery directly instead
        # of going through Qt's meta-object slot-lookup — the same
        # build quirk that hits ``_rotate_spinner`` also surfaces here
        # as ``AttributeError: Slot 'EpubReaderDialog::
        # _on_overlay_merge_done(...)' not found``.
        self._overlay_thread.done.connect(
            lambda overlaid, imgs, applied, worker=worker:
                self._on_overlay_merge_done(overlaid, imgs, applied, worker=worker)
        )
        self._overlay_thread.start()

    @Slot(object, object, bool)
    def _on_overlay_merge_done(self, overlaid_chapters, merged_images,
                               overlay_applied: bool, worker=None):
        if getattr(self, "_closing", False):
            return
        """Merge worker finished: hand off to the main-thread finalizer.

        ``@Slot(object, object, bool)`` matches the updated signature
        of :attr:`_OverlayMergeThread.done` — both sides use ``object``
        so the payload is a raw Python reference and PySide6 doesn't
        try (and fail) to look up a ``(QVariantList, QVariantMap, bool)``
        slot on ``EpubReaderDialog``.
        """
        if worker is not None:
            if worker is not self._overlay_thread:
                return
            self._overlay_loaded_signature = worker._read_signature
            self._overlay_retry_required = worker._retry_required
        self._overlay_merge_pending = False
        raw_chapters = getattr(self, "_pending_raw_chapters", []) or []
        filenames = getattr(self, "_pending_filenames", []) or []
        self._pending_raw_chapters = []
        self._pending_filenames = []
        self._finalize_post_load(
            raw_chapters, overlaid_chapters,
            merged_images, bool(overlay_applied), filenames)

    def _begin_toc_width_lock(self) -> list[int]:
        """Pin the TOC pane width during transient reader reload states.

        Hiding the reader stack during the paginated prime pass leaves the
        TOC as the only visible splitter child. Qt then happily expands it
        to fill the splitter until the reader is shown again, which reads
        as a full-screen TOC flash during raw/translated swaps.
        """
        sizes = list(self._splitter.sizes())
        total = sum(sizes) if sizes else 0
        if total > 0:
            width = max(0, int(sizes[0]))
        else:
            width = max(0, int(getattr(self, '_toc_saved_width', 220) or 220))
            sizes = [width, max(1, int(self.width()) - width)]

        if not hasattr(self, '_toc_width_lock_state'):
            self._toc_width_lock_state = (
                self._toc_list.minimumWidth(),
                self._toc_list.maximumWidth(),
            )
        self._toc_list.setMinimumWidth(width)
        self._toc_list.setMaximumWidth(width)
        if total > 0:
            self._splitter.setSizes(sizes)
        return sizes

    def _end_toc_width_lock(self, sizes: list[int] | None = None) -> None:
        """Restore normal TOC resizing after a transient lock."""
        state = getattr(self, '_toc_width_lock_state', None)
        if state is None:
            return
        min_width, max_width = state
        self._toc_list.setMinimumWidth(min_width)
        self._toc_list.setMaximumWidth(max_width)
        try:
            del self._toc_width_lock_state
        except AttributeError:
            pass
        if sizes and any(s > 0 for s in sizes):
            self._splitter.setSizes(sizes)

    def _reveal_initial_reader_shell(self) -> None:
        """Crossfade the loading shell into the first rendered reader frame.

        QWebEngine's Chromium surface is a native compositor layer, so an
        ordinary child QWidget cannot cover its first black buffers. Instead,
        snapshot the loading shell into a separate owned tool window, expose
        Chromium behind it, then fade that native overlay away.
        """
        if not getattr(self, "_reader_startup_pending", False):
            return
        if (_HAS_WEBENGINE
                and not getattr(self, "_reader_reveal_queued", False)):
            self._reader_reveal_queued = True
            self._begin_initial_reader_transition()
            return
        if getattr(self, "_reader_reveal_queued", False):
            return
        self._commit_initial_reader_shell()

    def _begin_initial_reader_transition(self) -> None:
        """Place a native loading-shell snapshot above the reader window."""
        if (getattr(self, "_closing", False)
                or not getattr(self, "_reader_startup_pending", False)):
            self._reader_reveal_queued = False
            return
        try:
            snapshot = self.grab()
            if snapshot.isNull():
                raise RuntimeError("reader transition snapshot is empty")
            flags = (
                Qt.Tool
                | Qt.FramelessWindowHint
                | Qt.NoDropShadowWindowHint
                | Qt.WindowDoesNotAcceptFocus
            )
            overlay = QLabel(parent=self, f=flags)
            overlay.setObjectName("readerStartupTransition")
            overlay.setAttribute(Qt.WA_ShowWithoutActivating, True)
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            overlay.setPixmap(snapshot)
            overlay.setScaledContents(True)
            top_left = self.mapToGlobal(QPoint(0, 0))
            overlay.setGeometry(
                top_left.x(), top_left.y(), self.width(), self.height())
            overlay.setWindowOpacity(1.0)
            overlay.show()
            overlay.raise_()
            self._reader_transition_overlay = overlay
            # Give the owned native overlay one event-loop turn to paint before
            # the underlying WebEngine surface is made visible.
            QTimer.singleShot(0, self._commit_initial_reader_shell)
        except Exception:
            logger.debug("Reader startup transition unavailable: %s",
                         traceback.format_exc())
            # The themed loading shell still stays up during the compositor
            # settle interval when a platform cannot create the overlay.
            QTimer.singleShot(
                self._STARTUP_TRANSITION_SETTLE_MS,
                self._commit_initial_reader_shell,
            )

    def _commit_initial_reader_shell(self) -> None:
        """Expose the reader behind the startup-transition snapshot."""
        if (getattr(self, "_closing", False)
                or not getattr(self, "_reader_startup_pending", False)):
            self._cleanup_initial_reader_transition()
            return
        self._reader_startup_pending = False
        self._spin_timer.stop()
        self.setUpdatesEnabled(False)
        try:
            self._reader_stack.show()
            self._toolbar_widget.show()
            self._content_widget.show()
            self._loading_widget.hide()
        finally:
            self.setUpdatesEnabled(True)
        self.update()
        if getattr(self, "_reader_transition_overlay", None) is not None:
            QTimer.singleShot(
                self._STARTUP_TRANSITION_SETTLE_MS,
                self._fade_initial_reader_transition,
            )
        else:
            self._reader_reveal_queued = False

    def _fade_initial_reader_transition(self) -> None:
        """Fade the loading snapshot away after Chromium is visibly stable."""
        overlay = getattr(self, "_reader_transition_overlay", None)
        if getattr(self, "_closing", False) or overlay is None:
            self._cleanup_initial_reader_transition()
            return
        animation = QPropertyAnimation(overlay, b"windowOpacity", self)
        animation.setDuration(self._STARTUP_TRANSITION_FADE_MS)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.finished.connect(self._cleanup_initial_reader_transition)
        self._reader_transition_animation = animation
        animation.start()

    def _cleanup_initial_reader_transition(self) -> None:
        """Destroy transition resources and leave the reader fully interactive."""
        animation = getattr(self, "_reader_transition_animation", None)
        self._reader_transition_animation = None
        if animation is not None:
            try:
                animation.stop()
            except RuntimeError:
                pass
            animation.deleteLater()
        overlay = getattr(self, "_reader_transition_overlay", None)
        self._reader_transition_overlay = None
        if overlay is not None:
            try:
                overlay.hide()
                overlay.deleteLater()
            except RuntimeError:
                pass
        self._reader_reveal_queued = False

    def _finalize_post_load(self, raw_chapters, overlaid_chapters, images,
                            overlay_applied: bool, filenames):
        """Install the loaded chapters + images into the reader UI.

        Split out from :meth:`_on_epub_loaded_from_cache` so the
        translator-overlay merge + extra-image-dir scan can run in
        :class:`_OverlayMergeThread` without blocking the event loop.
        For the fast path (plain EPUB, no overlay) the loader calls this
        directly; for the heavy path it's invoked via the worker's
        ``done`` signal.
        """
        # Persist both flavors + pick the one matching the current toggle.
        self._chapters_raw = raw_chapters
        self._chapters_overlaid = overlaid_chapters
        chapters = raw_chapters if (self._show_raw and overlay_applied) else overlaid_chapters
        # Expose the Show-raw pill when EITHER a translated overlay
        # landed on at least one chapter (overlay mode) OR a dual-path
        # alt EPUB was wired up by the caller (Completed tab with a
        # resolved raw source). Otherwise there's nothing to toggle
        # against and the pill stays hidden.
        has_dual_path = bool(getattr(self, "_raw_epub_alt_path", ""))
        has_workspace_raw = bool(getattr(self, "_workspace_has_raw", False))
        if getattr(self, "_raw_btn", None) is not None:
            self._raw_btn.setVisible(
                bool(overlay_applied) or has_dual_path or has_workspace_raw
            )
            # Sync the pill's checked state with the actually-rendered
            # flavor — this can diverge from the persisted default when
            # overlay mode couldn't satisfy the "raw" preference (no
            # chapter had an overlay entry) or when a dual-path reload
            # already swapped ``_epub_path`` to the raw file.
            try:
                raw_alt = getattr(self, "_raw_epub_alt_path", "") or ""
                currently_raw = bool(
                    (overlay_applied and self._show_raw)
                    or (has_workspace_raw and self._show_raw)
                    or (has_dual_path and raw_alt
                        and os.path.normcase(os.path.abspath(
                            self._epub_path))
                        == os.path.normcase(os.path.abspath(raw_alt)))
                )
            except Exception:
                currently_raw = bool(
                    (overlay_applied or has_workspace_raw) and self._show_raw)
            self._raw_btn.blockSignals(True)
            self._raw_btn.setChecked(currently_raw)
            self._raw_btn.blockSignals(False)
        self._chapters = chapters
        self._set_reader_images(images)
        self._refresh_search_for_active_chapters()
        # Keep filenames around so callers can resolve chapter indices by
        # source filename (used by initial_chapter_filename lookup + TOC jumps).
        self._chapter_filenames = [os.path.basename(f or "").lower() for f in filenames]
        self._chapter_display_numbers = _chapter_display_numbers(
            self._chapter_filenames, getattr(self, "_config", None))
        self._chapter_page_cache = {}  # {chapter_index: page_count}
        self._loaded_chapter = -1  # track which chapter's HTML is loaded

        # Freeze splitter sizes around the TOC rebuild and any hidden
        # paginated-prime render. Otherwise the TOC can momentarily fill
        # the splitter when the reader stack is hidden.
        _saved_sizes = self._begin_toc_width_lock()
        self._refresh_native_toc_entries()
        self._rebuild_toc_sidebar(current_chapter=0)
        if _saved_sizes and any(s > 0 for s in _saved_sizes):
            self._splitter.setSizes(_saved_sizes)

        # During the initial open, keep the loading shell visible until the
        # browser finishes its real chapter render. Later reloads retain the
        # already-visible reader, matching the existing seamless swap path.
        if not getattr(self, "_reader_startup_pending", False):
            self._toolbar_widget.show()
            self._loading_widget.hide()
            self._content_widget.show()

        if self._chapters:
            # Honor an explicit initial chapter if one was requested by the
            # caller; otherwise default to the first chapter. The index is
            # clamped to the loaded chapter range so stale requests (e.g. a
            # chapter that was removed between sessions) fall back cleanly.
            # Filename takes priority — BookDetailsDialog passes the source
            # chapter filename so the selection is stable across ordering
            # differences between its spine-based list and the reader's
            # manifest-ordered list.
            initial = 0
            # A reload carrying a position hint (Show-raw toggle in
            # dual-path mode) takes the highest precedence so the user
            # lands back on the same chapter in the new flavor — the
            # finalizer then consumes the page portion of the hint.
            reload_hint = getattr(self, '_reload_position_hint', None)
            if reload_hint:
                initial = max(0, min(int(reload_hint.get("row", 0) or 0),
                                     len(self._chapters) - 1))
                # Promote to a page hint for the finalizer and clear the
                # reload slot so a later organic open doesn't inherit it.
                self._pending_page_hint = {
                    "row": initial,
                    "was_last_page": bool(reload_hint.get("was_last_page")),
                    "proportion": float(reload_hint.get("proportion") or 0.0),
                }
                self._reload_position_hint = None
            elif self._initial_chapter_filename:
                try:
                    initial = self._chapter_filenames.index(self._initial_chapter_filename)
                except ValueError:
                    initial = 0
            elif self._initial_chapter is not None:
                initial = max(0, min(self._initial_chapter, len(self._chapters) - 1))
            # Select initial chapter silently — the priming sequence below
            # drives the initial render so we don't want setCurrentRow to
            # also fire _on_chapter_selected.
            self._set_toc_selection_for_chapter(initial)
            self._current_row = initial
            self._current_page = 0

            # Prime the reader: if we're opening in a paginated mode
            # (single/double), first render in scroll mode and then switch
            # back to the configured mode. This replicates the manual
            # layout-toggle workaround — without it, QtWebEngine caches the
            # GPU-composited #columns layer at the wrong DPR and text
            # stays blurry until the user toggles modes.
            saved_mode = self._layout_mode
            if (_HAS_WEBENGINE
                    and saved_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE)):
                # Hide the reader stack during the prime swap so the
                # brief scroll-mode render never becomes visible. The
                # TOC width remains locked until the real paginated view
                # is shown, preventing a full-width sidebar flash.
                self._reader_stack.hide()
                self._priming_initial_render = True
                self._prime_toc_sizes = _saved_sizes
                self._prime_saved_mode = saved_mode
                self._layout_mode = LAYOUT_SCROLL
                # The swap-back is triggered event-driven the moment the
                # scroll-mode load finishes (see _on_reader_load_finished),
                # avoiding an arbitrary fixed delay.
                self._render_current()
            else:
                self._end_toc_width_lock(_saved_sizes)
                self._render_current()
                if not _HAS_WEBENGINE:
                    self._reveal_initial_reader_shell()
                    self._finish_raw_toggle()
        else:
            self._end_toc_width_lock(_saved_sizes)
            self._reader_stack.setCurrentIndex(0)
            self._reader.setHtml(
                "<div style='text-align:center; padding: 60px; color: #888;'>"
                "<p style='font-size: 16pt;'>📭</p>"
                "<p>No readable content found for this book.</p></div>")
            self._reveal_initial_reader_shell()
            self._finish_raw_toggle()

        # Kick off the auto-refresh timer now that the initial load has
        # finished. Only started once per dialog \u2014 subsequent finalize
        # calls (from :meth:`_on_show_raw_toggled` reloads or from
        # :meth:`_on_auto_refresh_merge_done` itself) skip this branch.
        self._ensure_overlay_refresh_timer()

    def _ensure_overlay_refresh_timer(self):
        """Start the translated-overlay auto-refresh timer (idempotent).

        No-ops when no ``overlay_provider`` was supplied (e.g. plain
        EPUB opens, dual-path Completed-tab opens). Also skipped when
        the timer has already been created by a previous finalize \u2014
        a single ``QTimer`` drives every subsequent tick.
        """
        if not callable(getattr(self, "_overlay_provider", None)):
            return
        if getattr(self, "_overlay_refresh_timer", None) is not None:
            return
        timer = QTimer(self)
        timer.setInterval(self._auto_refresh_interval_ms)
        # Lambda-wrap so PySide6 routes the callback directly instead of
        # going through Qt's meta-object slot-lookup \u2014 same quirk that
        # forces ``_rotate_spinner`` to be dispatched via a lambda.
        timer.timeout.connect(lambda: self._on_overlay_refresh_tick())
        self._overlay_refresh_timer = timer
        timer.start()

    @Slot()
    def _on_overlay_refresh_tick(self):
        if getattr(self, "_closing", False):
            return
        """Poll the overlay provider and re-merge when the overlay changed.

        The translator writes ``response_*.html`` files as it finishes
        each chapter. The provider (typically
        :meth:`BookDetailsDialog._build_translated_overlay`) rescans
        that state and returns a fresh ``(overlay_map, extra_dirs)``
        tuple. When the returned overlay differs from the one currently
        merged into :attr:`_chapters_overlaid` (either new chapter
        entries or a rewritten response file with a newer mtime) we
        fire another :class:`_OverlayMergeThread` off the UI thread
        and swap its result in via :meth:`_on_auto_refresh_merge_done`.

        Safety rails keep the tick cheap and non-disruptive:
          * Hidden dialog \u2192 bail out (no point paying for disk I/O).
          * Active load / merge thread \u2192 bail out (don't pile on).
          * Empty raw chapter list \u2192 bail out (initial load hasn't
            landed yet).
          * Identical overlay signature \u2192 bail out (nothing changed).
        """
        provider = getattr(self, "_overlay_provider", None)
        if not callable(provider):
            return
        try:
            if not self.isVisible():
                return
        except Exception:
            pass
        # Don't stack refresh merges on top of each other or on top of
        # the initial loader.
        if getattr(self, "_overlay_merge_pending", False):
            worker = getattr(self, "_overlay_thread", None)
            try:
                if worker is not None and (
                        worker.isRunning() or worker._result_ready):
                    # A finished worker can still have its result queued for
                    # the GUI thread. Do not start a second merge in that gap.
                    return
            except RuntimeError:
                pass
            # A cancelled/failed worker that emitted no result must not leave
            # refresh permanently disabled.
            self._overlay_merge_pending = False
            self._overlay_retry_required = True
        for attr in ("_overlay_thread", "_loader_thread",
                     "_cache_loader_thread"):
            t = getattr(self, attr, None)
            if t is None:
                continue
            try:
                if t.isRunning():
                    return
            except Exception:
                pass
        if not self._chapters_raw:
            return
        try:
            result = provider()
        except Exception:
            logger.debug("Overlay provider raised: %s",
                         traceback.format_exc())
            return
        # Accept either a bare overlay dict or a (overlay, extra_dirs)
        # tuple so the provider stays flexible.
        if isinstance(result, tuple) and len(result) == 2:
            raw_overlay, new_extra_dirs = result
        elif isinstance(result, dict):
            raw_overlay, new_extra_dirs = result, self._extra_image_dirs
        else:
            return
        # Normalise keys the same way ``__init__`` does so the diff
        # below compares like-for-like.
        normalised: dict[str, dict] = {}
        for k, v in (raw_overlay or {}).items():
            if not k:
                continue
            key = os.path.basename(str(k)).lower()
            if not key:
                continue
            if isinstance(v, str):
                normalised[key] = {"path": v}
            elif isinstance(v, dict) and v.get("path"):
                normalised[key] = dict(v)

        new_sig = _reader_overlay_signature(normalised)
        old_sig = getattr(self, "_overlay_loaded_signature", None)
        extras_changed = (list(new_extra_dirs or [])
                          != list(self._extra_image_dirs or []))
        if (new_sig == old_sig and not extras_changed
                and not getattr(self, "_overlay_retry_required", False)):
            return
        # Swap in the new overlay + extra dirs, then kick a merge thread
        # that reuses the already-loaded raw chapters + images.
        self._translated_overlay = normalised
        self._extra_image_dirs = list(new_extra_dirs or [])
        raw_chapters = list(self._chapters_raw)
        filenames = list(self._chapter_filenames)
        self._pending_raw_chapters = raw_chapters
        self._pending_filenames = filenames
        prev = getattr(self, "_overlay_thread", None)
        if prev is not None:
            _stop_qthread_safely(
                prev, timeout_ms=250,
                signal_names=("done",),
            )
        self._overlay_thread = _OverlayMergeThread(
            raw_chapters=raw_chapters,
            images=dict(self._images),
            filenames=filenames,
            overlay_map=self._translated_overlay,
            extra_image_dirs=self._extra_image_dirs,
            config=self._config,
            parent=self,
            previous_chapters=self._chapters_overlaid,
        )
        worker = self._overlay_thread
        self._overlay_merge_pending = True
        # Route the ``done`` signal to the in-place updater rather than
        # :meth:`_on_overlay_merge_done` \u2014 the latter re-runs the
        # full finalize pass (with its priming-mode render), which
        # would blank the reader on every tick.
        self._overlay_thread.done.connect(
            lambda overlaid, imgs, applied, worker=worker:
                self._on_auto_refresh_merge_done(
                    overlaid, imgs, applied, worker=worker)
        )
        self._overlay_thread.start()

    @Slot(object, object, bool)
    def _on_auto_refresh_merge_done(self, overlaid_chapters,
                                    merged_images,
                                    overlay_applied: bool, worker=None):
        if getattr(self, "_closing", False):
            return
        """Swap refreshed chapters into the UI without a full re-render.

        Differs from :meth:`_on_overlay_merge_done` in two key ways:
          * Backing chapter + image state is replaced, the TOC is
            rebuilt, and the Show-raw pill visibility is refreshed,
            but the reader pane itself is only re-rendered when the
            currently-viewed chapter's content actually changed. New
            chapters that the user hasn't opened yet therefore land
            silently in the TOC; the active chapter keeps its scroll
            / page position.
          * Reading position is preserved via the existing page-hint
            plumbing \u2014 if the current chapter's content did change,
            the finalizer rehydrates it to the same proportional
            page rather than jumping back to page 1.
        """
        if worker is not None:
            if worker is not self._overlay_thread:
                return
            self._overlay_loaded_signature = worker._read_signature
            self._overlay_retry_required = worker._retry_required
        self._overlay_merge_pending = False
        pending_raw = getattr(self, "_pending_raw_chapters", []) or []
        self._pending_raw_chapters = []
        self._pending_filenames = []
        previous_chapters = self._chapters
        self._chapters_raw = pending_raw or self._chapters_raw
        self._chapters_overlaid = overlaid_chapters or []
        self._set_reader_images(merged_images or self._images)
        if self._show_raw and overlay_applied:
            new_chapters = self._chapters_raw
        else:
            new_chapters = self._chapters_overlaid or self._chapters_raw
        changed_rows = {
            idx for idx in range(max(len(previous_chapters), len(new_chapters)))
            if idx >= len(previous_chapters) or idx >= len(new_chapters)
            or previous_chapters[idx] != new_chapters[idx]
        }
        render_changed = self._current_row in changed_rows or (
            self._layout_mode == LAYOUT_ALL and bool(changed_rows)
        )
        # Capture the old position before invalidating counts or swapping text.
        position_hint = self._capture_position_hint() if render_changed else None
        for idx in changed_rows:
            self._chapter_page_cache.pop(idx, None)
        prev_row = self._current_row
        self._chapters = new_chapters
        if changed_rows:
            self._refresh_search_for_active_chapters()
        # Rebuild the TOC, preserving the current selection. Chapter
        # count can grow (new translated chapters surface with
        # translated titles) but never shrinks during a refresh.
        _saved_sizes = self._begin_toc_width_lock()
        if new_chapters:
            new_row = max(0, min(prev_row, len(new_chapters) - 1))
            self._current_row = new_row
        else:
            self._current_row = 0
        self._refresh_native_toc_entries()
        self._rebuild_toc_sidebar(current_chapter=self._current_row)
        self._end_toc_width_lock(_saved_sizes)
        # Refresh the Show-raw pill visibility \u2014 overlay_applied can
        # flip from False \u2192 True when the first translated chapter
        # lands mid-read.
        has_dual_path = bool(getattr(self, "_raw_epub_alt_path", ""))
        if getattr(self, "_raw_btn", None) is not None:
            self._raw_btn.setVisible(
                bool(overlay_applied) or has_dual_path)
            if overlay_applied:
                # The first translation can arrive after an all-raw startup,
                # when the hidden pill had no translated view to compare.
                self._raw_btn.blockSignals(True)
                self._raw_btn.setChecked(bool(self._show_raw))
                self._raw_btn.blockSignals(False)
        # If the active chapter's content changed, re-render it with a
        # position hint so the user lands back near where they were.
        # If unchanged, leave the viewport alone \u2014 this keeps
        # scroll position + paginated page rock steady across ticks.
        if render_changed and new_chapters:
            self._pending_page_hint = position_hint
            self._loaded_chapter = -1
            self._render_current()
        else:
            self._update_nav_buttons()
        # Overlay state may have changed for the current chapter (e.g. its
        # translation just landed, or a partial one was cleaned up).
        try:
            self._update_translate_btn_visibility()
        except Exception:
            pass

    def _get_chapter_pages(self, chapter_idx):
        """Get page count for a chapter (cached or live-compute for current)."""
        if chapter_idx in self._chapter_page_cache:
            return self._chapter_page_cache[chapter_idx]
        return 1  # fallback until chapter is actually rendered

    def _get_global_page_info(self):
        """Return (current_global_page_1based, total_pages)."""
        offset = sum(self._get_chapter_pages(i) for i in range(self._current_row))
        total = sum(self._get_chapter_pages(i) for i in range(len(self._chapters)))
        return (offset + self._current_page + 1, total)

    def _on_epub_error(self, error_msg):
        self._spin_timer.stop()
        self._reader_stack.setCurrentIndex(0)
        source_label = "PDF workspace" if self._workspace_mode else "EPUB"
        self._reader.setHtml(
            f"<div style='text-align:center; padding: 60px; color: #ff6b6b;'>"
            f"<p style='font-size: 16pt;'>⚠️</p>"
            f"<p>Failed to load {source_label}:<br>{error_msg}</p></div>")
        if getattr(self, "_reader_startup_pending", False):
            self._reveal_initial_reader_shell()
        else:
            self._toolbar_widget.show()
            self._loading_widget.hide()
            self._content_widget.show()
        self._finish_raw_toggle()

    # ── Theme / Font / Spacing ─────────────────────────────────────────────

    # _get_theme moved verbatim to reader_doc.ReaderDocMixin (inherited).

    def _apply_reader_style(self):
        t = self._get_theme()
        bg = t['bg']
        fg = t['fg']
        border = t['border']
        muted = t.get('muted', '#888888')
        selection = t.get('selection', border)
        button_bg = t.get('button_bg', t['code_bg'])
        button_hover = t.get('button_hover', border)
        if _HAS_WEBENGINE:
            # For QWebEngineView, styling is done via CSS in _wrap_html.
            # We style surrounding containers only.
            if self._reader is not None:
                self._reader.setStyleSheet(f"background: {bg}; border: none;")
            if self._reader_left is not None:
                self._reader_left.setStyleSheet(f"background: {bg}; border: none; border-right: 1px solid {border};")
            if self._reader_right is not None:
                self._reader_right.setStyleSheet(f"background: {bg}; border: none;")
        else:
            css = f"""
                QTextBrowser {{
                    background: {bg}; color: {fg}; border: none;
                    padding: 20px 30px 32px 30px; font-size: {self._font_size}pt;
                }}
                QTextBrowser a {{ color: {t['link']}; }}
                QScrollBar:vertical {{ width: 8px; background: {bg}; }}
                QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 20px; }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            """
            if self._reader is not None:
                self._reader.setStyleSheet(css)
            if self._reader_left is not None:
                self._reader_left.setStyleSheet(
                    css + f"QTextBrowser {{ border-right: 1px solid {border}; }}")
            if self._reader_right is not None:
                self._reader_right.setStyleSheet(css)
        # Theme all surrounding containers
        self.setStyleSheet(f"QDialog {{ background: {bg}; }}")
        self._reader_stack.setStyleSheet(f"QStackedWidget {{ background: {bg}; border: none; }}")
        self._double_widget.setStyleSheet(f"background: {bg};")
        # Row styling notes:
        #   * Extra vertical padding (8/10 px) so a wrapped title
        #     doesn't collide with the next row's text baseline.
        #   * ``margin-bottom: 2px`` + a 1 px ``border-bottom``
        #     draws a subtle divider between rows so the TOC reads
        #     as a list of discrete entries even when the user shrinks
        #     the sidebar enough to force wrapping.
        #   * ``background-clip: padding-box`` keeps the rounded-
        #     rect selection highlight inside the padding and off the
        #     divider line.
        self._toc_list.setStyleSheet(f"""
            QListWidget {{ background: {bg}; border: none;
                color: {fg}; font-size: 9pt; padding: 4px; outline: 0; }}
            QListWidget::item {{
                padding: 8px 10px;
                margin-bottom: 2px;
                border-radius: 4px;
                border-bottom: 1px solid {border};
                background-clip: padding-box;
            }}
            QListWidget::item:selected {{ background: {border}; color: {t['heading']}; }}
            QListWidget::item:hover {{ background: {t['code_bg']}; }}
            QScrollBar:vertical {{ width: 8px; background: {bg}; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{ height: 8px; background: {bg}; }}
            QScrollBar::handle:horizontal {{ background: {border}; border-radius: 4px; min-width: 20px; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)
        # Re-layout so Qt recomputes each item's sizeHint against the
        # new stylesheet padding + word-wrap width. Without this, an
        # ``_apply_reader_style`` call after the TOC is populated
        # leaves existing rows at their old height and the divider
        # lines overlap the text of wrapped entries.
        try:
            self._toc_list.doItemsLayout()
        except Exception:
            pass
        self._toc_list.viewport().setStyleSheet(f"background: {bg};")
        self._nav_bar.setStyleSheet(f"background: {bg}; border-top: 1px solid {border};")
        self._loading_widget.setStyleSheet(f"background: {bg};")
        self._splitter.setStyleSheet(f"QSplitter::handle {{ background: transparent; width: 0px; }}")
        self._search_bar.setStyleSheet(f"""
            QLineEdit {{
                background: {t['code_bg']}; color: {fg}; border: 1px solid {border};
                border-radius: 4px; padding: 6px 12px; font-size: 10pt;
                margin: 4px 40px;
            }}
        """)
        panel = getattr(self, "_search_panel", None)
        if panel is not None:
            search_style = f"""
                QFrame#readerSearchPanel, QFrame#readerSearchContent {{
                    background: {bg}; border-left: 1px solid {border};
                }}
                QLabel#readerSearchTitle {{
                    color: {fg}; font-weight: 600; font-size: 9pt;
                }}
                QLabel#readerSearchCount {{
                    color: {muted}; font-size: 8pt;
                }}
                QLineEdit#readerSearchQuery {{
                    background: {t['code_bg']}; color: {fg}; border: 1px solid {border};
                    border-radius: 4px; padding: 3px 8px; font-size: 9pt;
                }}
                QListWidget#readerSearchResults {{
                    background: {t['bg']}; color: {fg}; border: 1px solid {border};
                    border-radius: 4px; outline: none;
                }}
                QListWidget#readerSearchResults::item {{
                    border-bottom: 1px solid {border}; padding: 0;
                }}
                QListWidget#readerSearchResults::item:selected {{
                    background: {selection}; color: {fg};
                }}
                QToolButton#readerSearchTool {{
                    background: {button_bg}; color: {fg}; border: 1px solid {border};
                    border-radius: 4px; padding: 2px 8px; font-size: 8.5pt;
                }}
                QToolButton#readerSearchTool:hover {{
                    background: {button_hover};
                }}
            """
            panel.setStyleSheet(search_style)
            content = getattr(self, "_search_content", None)
            if content is not None:
                content.setStyleSheet(search_style)
            dlg = getattr(self, "_search_dialog", None)
            if dlg is not None:
                dlg.setStyleSheet(search_style)

    def _toggle_search(self):
        if self._search_ui_is_open():
            self._close_search()
        else:
            self._show_search_dialog()

    def _close_search(self):
        self._search_bar.hide()
        self._search_bar.clear()
        self._search_dialog_generation = (
            int(getattr(self, "_search_dialog_generation", 0) or 0) + 1)
        self._cancel_search_dialog_workers()
        panel = getattr(self, "_search_panel", None)
        if panel is not None:
            panel.hide()
        self._restore_attached_search_geometry()
        dlg = getattr(self, "_search_dialog", None)
        if dlg is not None:
            try:
                dlg.hide()
            except Exception:
                pass
        if _HAS_WEBENGINE:
            self._clear_reader_find_highlights()
            for w in [self._reader, self._reader_left, self._reader_right]:
                if hasattr(w, 'findText'):
                    w.findText("")

    def _clear_reader_find_highlights(self):
        if not _HAS_WEBENGINE:
            return
        js = (
            "try { if (window.CSS && CSS.highlights) "
            "CSS.highlights.delete('glossarion-find'); } catch (e) {}"
            "try { window.getSelection().removeAllRanges(); } catch (e) {}"
        )
        for w in [self._reader, self._reader_left, self._reader_right]:
            try:
                if hasattr(w, 'page'):
                    w.page().runJavaScript(js)
            except Exception:
                pass

    def _on_search_text_changed(self, text):
        """Live highlight in current chapter as user types."""
        if not _HAS_WEBENGINE:
            return
        self._search_chapter_idx = self._current_row
        self._search_last_row = self._current_row
        if not text:
            self._search_current_text = ""
            self._search_match_index = -1
            self._clear_reader_find_highlights()
            for w in [self._reader, self._reader_left, self._reader_right]:
                if hasattr(w, 'findText'):
                    w.findText("")
            return
        self._search_current_text = text
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            count = self._count_chapter_matches(self._current_row, text)
            if count <= 0:
                self._search_match_index = -1
                self._search_match_count = 0
                return
            self._select_match_in_chapter(self._current_row, text, 0)
            return
        self._search_match_index = 0
        self._reader.findText(text)

    def _ensure_search_content(self):
        """Create the reusable search widget shared by docked/detached modes."""
        content = getattr(self, "_search_content", None)
        if content is not None:
            return content

        content = QFrame()
        content.setObjectName("readerSearchContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel("Search EPUB")
        title.setObjectName("readerSearchTitle")
        header.addWidget(title)
        count_label = QLabel("Type to search")
        count_label.setObjectName("readerSearchCount")
        header.addWidget(count_label)
        header.addStretch()

        detach_btn = QToolButton()
        detach_btn.setObjectName("readerSearchTool")
        detach_btn.setAutoRaise(False)
        detach_btn.setText("Detach")
        detach_btn.setToolTip("Detach search into a floating window")
        detach_btn.setFixedHeight(24)
        detach_btn.clicked.connect(self._toggle_search_detached)
        header.addWidget(detach_btn)

        close_btn = QToolButton()
        close_btn.setObjectName("readerSearchTool")
        close_btn.setAutoRaise(False)
        close_btn.setText("Close")
        close_btn.setToolTip("Close search")
        close_btn.setFixedHeight(24)
        close_btn.clicked.connect(self._close_search)
        header.addWidget(close_btn)
        layout.addLayout(header)

        query = _EpubSearchLineEdit()
        query.setObjectName("readerSearchQuery")
        query.setPlaceholderText("Search text... (Enter = next)")
        query.setFixedHeight(28)
        query.textChanged.connect(self._populate_search_results)
        query.enterPressed.connect(self._search_enter_next)
        layout.addWidget(query)

        results = _EpubSearchResultsList()
        results.setObjectName("readerSearchResults")
        results.setItemDelegate(_EpubSearchResultDelegate(results))
        results.setWordWrap(False)
        results.setUniformItemSizes(True)
        results.setSpacing(0)
        results.setMinimumHeight(80)
        results.enterPressed.connect(self._search_enter_next)
        results.itemActivated.connect(self._activate_search_result)
        results.itemClicked.connect(self._activate_search_result)
        layout.addWidget(results, 1)

        self._search_content = content
        self._search_dialog_query = query
        self._search_dialog_count = count_label
        self._search_dialog_results = results
        self._search_detach_btn = detach_btn
        self._apply_reader_style()
        return content

    def _set_search_content_parent(self, detached: bool) -> None:
        content = self._ensure_search_content()
        self._search_detached = bool(detached)
        self._config['epub_reader_search_detached'] = self._search_detached
        _persist_config_via_parent(self)

        if self._search_detached:
            self._restore_attached_search_geometry()
            panel = getattr(self, "_search_panel", None)
            if panel is not None:
                panel.hide()
            dlg = getattr(self, "_search_dialog", None)
            if dlg is None:
                dlg = QDialog(self)
                dlg.setWindowTitle("Search EPUB")
                dlg.resize(620, 320)
                dlg.setModal(False)
                dlg_layout = QVBoxLayout(dlg)
                dlg_layout.setContentsMargins(0, 0, 0, 0)
                dlg_layout.setSpacing(0)
                try:
                    dlg.rejected.connect(self._close_search)
                except Exception:
                    pass
                self._search_dialog = dlg
            if content.parent() is not dlg:
                dlg.layout().addWidget(content)
            btn = getattr(self, "_search_detach_btn", None)
            if btn is not None:
                btn.setText("Attach")
                btn.setToolTip("Attach search to the reader")
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return

        dlg = getattr(self, "_search_dialog", None)
        if dlg is not None:
            try:
                dlg.hide()
            except Exception:
                pass
        panel = getattr(self, "_search_panel", None)
        if panel is not None:
            if content.parent() is not panel:
                self._search_panel_layout.addWidget(content)
            self._expand_for_attached_search(require_visible=False)
            panel.show()
            QTimer.singleShot(0, self._resync_page_count)
        btn = getattr(self, "_search_detach_btn", None)
        if btn is not None:
            btn.setText("Detach")
            btn.setToolTip("Detach search into a floating window")
        self._expand_for_attached_search()
        QTimer.singleShot(0, self._expand_for_attached_search)
        self._schedule_search_realign()

    def _toggle_search_detached(self):
        self._set_search_content_parent(
            not bool(getattr(self, "_search_detached", False)))

    def _show_search_dialog(self):
        """Open the EPUB search UI docked to the reader or detached."""
        self._set_search_content_parent(
            bool(getattr(self, "_search_detached", False)))
        query = getattr(self, "_search_dialog_query", None)
        if query is not None:
            query.setFocus()
            query.selectAll()
            self._populate_search_results(query.text())

    def _available_reader_screen_geometry(self):
        try:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                return screen.availableGeometry()
        except Exception:
            pass
        return None

    def _expand_for_attached_search(self, require_visible: bool = True) -> None:
        """Grow the reader window so the attached search panel adds space."""
        if bool(getattr(self, "_search_detached", False)):
            return
        panel = getattr(self, "_search_panel", None)
        if panel is None:
            return
        if require_visible and not panel.isVisible():
            return
        target_panel_w = 360
        if not bool(getattr(self, "_search_window_expanded", False)):
            self._search_original_geometry = self.geometry()
            extra = target_panel_w
            self._search_window_expanded = True
        else:
            return
        if extra <= 0:
            return
        geo = self.geometry()
        avail = self._available_reader_screen_geometry()
        if avail is None:
            self.resize(self.width() + extra, self.height())
            return
        right_room = max(0, avail.right() - geo.right())
        left_room = max(0, geo.left() - avail.left())
        grow = min(extra, right_room + left_room)
        if grow <= 0:
            return
        shift_left = min(left_room, max(0, grow - right_room))
        if shift_left:
            self.move(max(avail.left(), geo.left() - shift_left), geo.top())
        self.resize(min(avail.width(), self.width() + grow), self.height())

    def _restore_attached_search_geometry(self) -> None:
        if not bool(getattr(self, "_search_window_expanded", False)):
            return
        original = getattr(self, "_search_original_geometry", None)
        self._search_window_expanded = False
        self._search_original_geometry = None
        if isinstance(original, QRect):
            self.setGeometry(original)

    def _search_excerpt(self, plain: str, start: int, end: int, radius: int = 60) -> str:
        return _epub_search_excerpt(plain, start, end, radius=radius)

    def _highlight_search_excerpt(self, excerpt: str, query: str) -> str:
        import html as html_lib
        safe = html_lib.escape(excerpt or "")
        needle = html_lib.escape(query or "")
        if not needle:
            return safe
        pattern = re.compile(re.escape(needle), re.IGNORECASE)
        return pattern.sub(
            lambda m: (
                "<span style='background:#ffd966; color:#101010; "
                "border-radius:2px; padding:0 2px;'>"
                f"{m.group(0)}</span>"
            ),
            safe,
        )

    def _set_search_result_widget(self, item, title: str, excerpt: str,
                                  query: str, match_count: int = 1) -> None:
        clean_title = " ".join(str(title or "").split())
        clean_excerpt = " ".join(str(excerpt or "").split())
        item.setText(f"{clean_title}\n{clean_excerpt}" if clean_excerpt else clean_title)
        item.setToolTip(clean_excerpt or clean_title)
        item.setSizeHint(QSize(0, 42))

    def _cancel_search_dialog_workers(self) -> None:
        timer = getattr(self, "_search_debounce_timer", None)
        if timer is not None:
            timer.stop()
        render_timer = getattr(self, "_search_render_timer", None)
        if render_timer is not None:
            render_timer.stop()
        self._search_pending_render_rows = []
        self._search_pending_render_query = ""
        self._search_pending_render_index = 0
        for worker in list(getattr(self, "_search_workers", []) or []):
            try:
                worker.cancel()
            except Exception:
                pass

    def _populate_search_results(self, text):
        results = getattr(self, "_search_dialog_results", None)
        count_label = getattr(self, "_search_dialog_count", None)
        if results is None or count_label is None:
            return
        self._search_dialog_generation = (
            int(getattr(self, "_search_dialog_generation", 0) or 0) + 1)
        self._cancel_search_dialog_workers()
        results.clear()
        query = (text or "").strip()
        self._search_pending_render_rows = []
        self._search_pending_render_query = query
        self._search_pending_render_index = 0
        self._search_dialog_total_matches = 0
        self._search_dialog_total_results = 0
        self._search_scan_done = False
        if not query:
            count_label.setText("Type to search")
            return
        count_label.setText("Waiting...")
        timer = getattr(self, "_search_debounce_timer", None)
        if timer is not None:
            timer.start(self._SEARCH_DEBOUNCE_MS)
        else:
            self._start_search_dialog_worker()

    def _refresh_search_for_active_chapters(self) -> None:
        """Re-run open search UI after raw/translated chapter swaps."""
        self._search_current_text = ""
        self._search_match_index = -1
        self._search_match_count = 0
        self._search_last_row = getattr(self, "_current_row", 0)
        self._search_dialog_active_text = ""
        self._search_dialog_active_chapter = -1
        self._search_dialog_active_occurrence = -1
        self._search_dialog_active_row = -1
        self._search_realign_generation = (
            int(getattr(self, "_search_realign_generation", 0) or 0) + 1)
        self._search_selection_generation = (
            int(getattr(self, "_search_selection_generation", 0) or 0) + 1)
        if _HAS_WEBENGINE:
            self._clear_reader_find_highlights()

        if not self._search_ui_is_open():
            return
        query = getattr(self, "_search_dialog_query", None)
        text = query.text().strip() if query is not None else ""
        if text:
            self._populate_search_results(text)
        else:
            self._populate_search_results("")

    def _start_search_dialog_worker(self):
        query_widget = getattr(self, "_search_dialog_query", None)
        count_label = getattr(self, "_search_dialog_count", None)
        if query_widget is None or count_label is None:
            return
        query = query_widget.text().strip()
        search_id = int(getattr(self, "_search_dialog_generation", 0) or 0)
        if not query:
            count_label.setText("Type to search")
            return
        count_label.setText("Searching...")
        worker = _EpubSearchThread(
            search_id,
            query,
            list(getattr(self, "_chapters", []) or []),
            config=self._config,
            parent=self,
        )
        self._search_workers.append(worker)
        worker.results_batch_ready.connect(
            self._on_search_dialog_results_batch)
        worker.results_ready.connect(self._on_search_dialog_results_ready)
        worker.finished.connect(lambda w=worker: self._on_search_worker_finished(w))
        worker.start()

    def _on_search_worker_finished(self, worker):
        try:
            self._search_workers.remove(worker)
        except (AttributeError, ValueError):
            pass
        try:
            worker.deleteLater()
        except Exception:
            pass

    @Slot(int, str, object)
    def _on_search_dialog_results_ready(self, search_id: int, query: str, rows):
        current_id = int(getattr(self, "_search_dialog_generation", 0) or 0)
        query_widget = getattr(self, "_search_dialog_query", None)
        results = getattr(self, "_search_dialog_results", None)
        count_label = getattr(self, "_search_dialog_count", None)
        if query_widget is None or results is None or count_label is None:
            return
        if search_id != current_id or query != query_widget.text().strip():
            return
        rows = list(rows or [])
        self._search_pending_render_rows = rows
        self._search_pending_render_query = query
        self._search_pending_render_index = 0
        self._search_scan_done = True
        total = int(rows[0].get("total_matches", len(rows)) or 0) if rows else 0
        self._search_dialog_total_matches = total
        self._search_dialog_total_results = len(rows)
        results.clear()
        result_count = len(rows)
        if total and result_count != total:
            text = (
                f"{total} match{'es' if total != 1 else ''} in "
                f"{result_count} result{'s' if result_count != 1 else ''}"
            )
        else:
            text = f"{total} match{'es' if total != 1 else ''}"
        count_label.setText(text + (" - rendering..." if result_count else ""))
        if result_count:
            self._search_render_timer.start()

    @Slot(int, str, object, bool)
    def _on_search_dialog_results_batch(self, search_id: int, query: str,
                                        rows, done: bool):
        current_id = int(getattr(self, "_search_dialog_generation", 0) or 0)
        query_widget = getattr(self, "_search_dialog_query", None)
        results = getattr(self, "_search_dialog_results", None)
        count_label = getattr(self, "_search_dialog_count", None)
        if query_widget is None or results is None or count_label is None:
            return
        if search_id != current_id or query != query_widget.text().strip():
            return
        rows = list(rows or [])
        pending = getattr(self, "_search_pending_render_rows", None)
        if pending is None or getattr(self, "_search_pending_render_query", "") != query:
            pending = []
            self._search_pending_render_rows = pending
            self._search_pending_render_query = query
            self._search_pending_render_index = 0
        if rows:
            pending.extend(rows)
        self._search_scan_done = bool(done)
        total = len(pending)
        self._search_dialog_total_matches = total
        self._search_dialog_total_results = total
        rendered = int(getattr(self, "_search_pending_render_index", 0) or 0)
        if total:
            suffix = ""
            if rendered < total:
                suffix = " - rendering..."
            elif not done:
                suffix = " - searching..."
            count_label.setText(
                f"{total} match{'es' if total != 1 else ''}{suffix}")
            timer = getattr(self, "_search_render_timer", None)
            if timer is not None and rows and not timer.isActive():
                timer.start()
        elif done:
            count_label.setText("No matches")
        else:
            count_label.setText("Searching...")

    def _render_search_result_batch(self):
        results = getattr(self, "_search_dialog_results", None)
        count_label = getattr(self, "_search_dialog_count", None)
        if results is None or count_label is None:
            return
        rows = getattr(self, "_search_pending_render_rows", []) or []
        query = getattr(self, "_search_pending_render_query", "") or ""
        idx = int(getattr(self, "_search_pending_render_index", 0) or 0)
        end = min(len(rows), idx + 75)
        results.setUpdatesEnabled(False)
        try:
            for row in rows[idx:end]:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, {
                    "chapter_idx": int(row.get("chapter_idx", 0) or 0),
                    "local_occurrence": int(row.get("local_occurrence", 0) or 0),
                    "global_occurrence": int(row.get("global_occurrence", 0) or 0),
                    "match_count": int(row.get("match_count", 1) or 1),
                    "text": row.get("text", query),
                })
                self._set_search_result_widget(
                    item,
                    str(row.get("title", "") or ""),
                    str(row.get("excerpt", "") or ""),
                    query,
                    int(row.get("match_count", 1) or 1),
                )
                results.addItem(item)
        finally:
            results.setUpdatesEnabled(True)
        self._search_pending_render_index = end
        total = int(getattr(self, "_search_dialog_total_matches", len(rows)) or 0)
        result_count = int(getattr(self, "_search_dialog_total_results", len(rows)) or 0)
        scan_done = bool(getattr(self, "_search_scan_done", False))
        if total and result_count != total:
            label_text = (
                f"{total} match{'es' if total != 1 else ''} in "
                f"{result_count} result{'s' if result_count != 1 else ''}"
            )
        else:
            label_text = f"{total} match{'es' if total != 1 else ''}"
        if end >= len(rows):
            self._search_render_timer.stop()
            if scan_done:
                count_label.setText(label_text if total else "No matches")
            else:
                count_label.setText(label_text + " - searching...")
        else:
            count_label.setText(
                f"{label_text} - showing {end}")

    def _search_ui_is_open(self) -> bool:
        panel = getattr(self, "_search_panel", None)
        if panel is not None and panel.isVisible():
            return True
        dlg = getattr(self, "_search_dialog", None)
        return bool(dlg is not None and dlg.isVisible())

    def _focus_is_in_search_ui(self) -> bool:
        content = getattr(self, "_search_content", None)
        focus = QApplication.focusWidget()
        while focus is not None:
            if focus is content:
                return True
            focus = focus.parentWidget()
        return False

    def _search_enter_next(self) -> bool:
        query = getattr(self, "_search_dialog_query", None)
        results = getattr(self, "_search_dialog_results", None)
        if query is None:
            return False
        if not query.text().strip():
            return False
        debounce_timer = getattr(self, "_search_debounce_timer", None)
        if debounce_timer is not None and debounce_timer.isActive():
            debounce_timer.stop()
            self._start_search_dialog_worker()
            return True
        if results is not None and results.count() > 0:
            row = results.currentRow()
            row = 0 if row < 0 else (row + 1) % results.count()
            results.setCurrentRow(row)
            item = results.item(row)
            if item is not None:
                results.scrollToItem(item)
                self._activate_search_result(item)
                return True
        self._find_next_from_search_dialog()
        return True

    def _find_next_from_search_dialog(self):
        query = getattr(self, "_search_dialog_query", None)
        if query is None:
            return
        text = query.text().strip()
        if not text:
            return
        targets = self._search_exact_targets()
        if not targets:
            return
        active_text = str(getattr(self, "_search_dialog_active_text", "") or "")
        active_chapter = int(getattr(self, "_search_dialog_active_chapter", -1) or -1)
        active_occurrence = int(getattr(self, "_search_dialog_active_occurrence", -1) or -1)
        next_target = targets[0]
        if active_text == text and active_chapter >= 0 and active_occurrence >= 0:
            for target in targets:
                if (target["chapter_idx"], target["local_occurrence"]) > (
                        active_chapter, active_occurrence):
                    next_target = target
                    break
        self._activate_search_target(next_target)

    def _activate_current_search_result(self):
        results = getattr(self, "_search_dialog_results", None)
        if results is None or results.count() <= 0:
            return
        row = results.currentRow()
        if row < 0:
            row = 0
        self._activate_search_row(row)

    def _activate_search_row(self, row: int):
        results = getattr(self, "_search_dialog_results", None)
        if results is None or results.count() <= 0:
            return
        row = max(0, min(int(row or 0), results.count() - 1))
        results.setCurrentRow(row)
        item = results.item(row)
        if item is None:
            return
        results.scrollToItem(item)
        self._search_dialog_active_row = row
        self._activate_search_result(item)

    def _search_exact_targets(self):
        results = getattr(self, "_search_dialog_results", None)
        targets = []

        rows = getattr(self, "_search_pending_render_rows", None)
        query = getattr(self, "_search_pending_render_query", "") or ""
        if rows and query == self._active_search_text():
            source = []
            for row, data in enumerate(rows):
                if isinstance(data, dict):
                    source.append((row, data))
        elif results is not None:
            source = []
            for row in range(results.count()):
                item = results.item(row)
                data = item.data(Qt.UserRole) if item is not None else None
                if isinstance(data, dict):
                    source.append((row, data))
        else:
            source = []

        for row, data in source:
            if not isinstance(data, dict):
                continue
            count = max(1, int(data.get("match_count", 1) or 1))
            base_local = int(data.get("local_occurrence", 0) or 0)
            base_global = int(data.get("global_occurrence", 0) or 0)
            for offset in range(count):
                targets.append({
                    "row": row,
                    "chapter_idx": int(data.get("chapter_idx", 0) or 0),
                    "local_occurrence": base_local + offset,
                    "global_occurrence": base_global + offset,
                    "text": data.get("text", ""),
                })
        return targets

    def _activate_search_target(self, target):
        if not isinstance(target, dict):
            return
        results = getattr(self, "_search_dialog_results", None)
        row = int(target.get("row", 0) or 0)
        if results is not None and 0 <= row < results.count():
            results.setCurrentRow(row)
            item = results.item(row)
            if item is not None:
                results.scrollToItem(item)
        text = target.get("text", "")
        chapter_idx = int(target.get("chapter_idx", 0) or 0)
        local_occurrence = int(target.get("local_occurrence", 0) or 0)
        global_occurrence = int(target.get("global_occurrence", 0) or 0)
        self._search_dialog_active_text = text
        self._search_dialog_active_chapter = chapter_idx
        self._search_dialog_active_occurrence = local_occurrence
        self._search_dialog_active_row = row
        if self._layout_mode == LAYOUT_ALL:
            self._select_text_occurrence(self._reader, text, global_occurrence)
            return
        self._select_match_in_chapter(chapter_idx, text, local_occurrence)

    def _active_search_text(self) -> str:
        query = getattr(self, "_search_dialog_query", None)
        if query is not None:
            try:
                text = query.text().strip()
            except RuntimeError:
                text = ""
            if text:
                return text
        return str(getattr(self, "_search_current_text", "") or "").strip()

    def _schedule_search_realign(self) -> None:
        """Re-select the active search result after column geometry settles."""
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if not self._active_search_text():
            return
        generation = int(getattr(self, "_search_realign_generation", 0) or 0) + 1
        self._search_realign_generation = generation

        def _run(expected=generation):
            if expected != int(getattr(self, "_search_realign_generation", 0) or 0):
                return
            self._realign_active_search_result()

        for delay in (80, 260, 620):
            QTimer.singleShot(delay, _run)

    def _realign_active_search_result(self) -> None:
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        text = self._active_search_text()
        if not text or not self._chapters:
            return
        occurrence = int(getattr(self, "_search_match_index", -1) or -1)
        if occurrence < 0:
            occurrence = 0
        count = self._count_chapter_matches(self._current_row, text)
        if count <= 0:
            return
        self._select_match_in_chapter(
            self._current_row, text, min(occurrence, count - 1))

    def _active_search_browser(self):
        """Return the visible pane that should drive in-chapter search."""
        return self._reader

    def _schedule_search_selection_retry(self, chapter_idx: int, text: str,
                                         occurrence: int) -> None:
        """Re-apply a paginated search jump as Chromium's columns settle."""
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if not text:
            return
        generation = int(
            getattr(self, "_search_selection_generation", 0) or 0) + 1
        self._search_selection_generation = generation
        chapter_idx = int(chapter_idx)
        occurrence = max(0, int(occurrence or 0))
        needle = text.casefold()

        def _retry(expected=generation):
            if expected != int(
                    getattr(self, "_search_selection_generation", 0) or 0):
                return
            if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
                return
            if int(getattr(self, "_current_row", -1) or -1) != chapter_idx:
                return
            if int(getattr(self, "_search_match_index", -1) or -1) != occurrence:
                return
            if self._active_search_text().casefold() != needle:
                return
            self._select_text_occurrence(
                self._active_search_browser(), text, occurrence)

        for delay in (180, 520, 900):
            QTimer.singleShot(delay, _retry)

    def _activate_search_result(self, item):
        data = item.data(Qt.UserRole) if item is not None else None
        if not isinstance(data, dict):
            return
        results = getattr(self, "_search_dialog_results", None)
        if results is not None:
            row = results.row(item)
            if row >= 0:
                self._search_dialog_active_row = row
        self._search_dialog_active_text = data.get("text", "")
        self._search_dialog_active_chapter = int(data.get("chapter_idx", 0) or 0)
        self._search_dialog_active_occurrence = int(data.get("local_occurrence", 0) or 0)
        text = data.get("text", "")
        chapter_idx = int(data.get("chapter_idx", 0) or 0)
        local_occurrence = int(data.get("local_occurrence", 0) or 0)
        global_occurrence = int(data.get("global_occurrence", 0) or 0)
        if self._layout_mode == LAYOUT_ALL:
            self._select_text_occurrence(self._reader, text, global_occurrence)
            return
        self._select_match_in_chapter(chapter_idx, text, local_occurrence)

    def _select_text_occurrence(self, browser, text: str, occurrence: int = 0,
                                update_reader: bool = True):
        """Select a concrete text occurrence and align the active layout."""
        if not _HAS_WEBENGINE or not text:
            return
        try:
            import json
            needle = json.dumps(text)
            wanted = max(0, int(occurrence or 0))
            visible_pages = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
        except Exception:
            return
        js = f"""(function() {{
  var query = {needle};
  var wanted = {wanted};
  var visiblePages = {visible_pages};
  var root = document.getElementById('content') || document.body;
  if (!query || !root) return {{found: false}};
  if (typeof _setupColumns === 'function') _setupColumns();
  var lowerQuery = query.toLocaleLowerCase();
  var ignored = {{script: true, style: true, noscript: true, img: true}};
  var nodeList = [];
  var flat = '';
  function addNode(node) {{
    if (node.nodeType === Node.TEXT_NODE) {{
      var value = node.nodeValue || '';
      if (value.length) {{
        nodeList.push({{node: node, offset: flat.length, length: value.length}});
        flat += value;
      }}
      return;
    }}
    if (node.nodeType !== Node.ELEMENT_NODE && node.nodeType !== Node.DOCUMENT_FRAGMENT_NODE) return;
    var tag = node.tagName ? node.tagName.toLowerCase() : '';
    if (ignored[tag]) return;
    var children = node.childNodes || [];
    for (var i = 0; i < children.length; i++) addNode(children[i]);
  }}
  function endpointForIndex(idx, isEnd) {{
    if (!nodeList.length) return null;
    idx = Math.max(0, Math.min(idx, flat.length));
    for (var i = 0; i < nodeList.length; i++) {{
      var item = nodeList[i];
      var start = item.offset;
      var end = item.offset + item.length;
      if ((isEnd && idx >= start && idx <= end) || (!isEnd && idx >= start && idx < end)) {{
        return {{node: item.node, offset: idx - start}};
      }}
    }}
    var last = nodeList[nodeList.length - 1];
    return {{node: last.node, offset: last.length}};
  }}
  addNode(root);
  function clearFindHighlights() {{
    try {{
      if (window.CSS && CSS.highlights) CSS.highlights.delete('glossarion-find');
    }} catch (e) {{}}
  }}
  function ensureFindHighlightStyle() {{
    if (document.getElementById('glossarion-find-highlight-style')) return;
    var style = document.createElement('style');
    style.id = 'glossarion-find-highlight-style';
    style.textContent = '::highlight(glossarion-find) {{ background: #ffd966; color: #101010; }}';
    document.head.appendChild(style);
  }}
  function rangeForIndex(idx) {{
    var startPoint = endpointForIndex(idx, false);
    var endPoint = endpointForIndex(idx + query.length, true);
    if (!startPoint || !endPoint) return null;
    var r = document.createRange();
    r.setStart(startPoint.node, startPoint.offset);
    r.setEnd(endPoint.node, endPoint.offset);
    return r;
  }}
  function scrollColumnsToRange(range, columns, span, pageCount) {{
    var rects = range.getClientRects();
    var rect = rects && rects.length ? rects[0] : range.getBoundingClientRect();
    var columnsRect = columns.getBoundingClientRect();
    var absoluteLeft = (rect.left - columnsRect.left) + columns.scrollLeft;
    var page = Math.min(pageCount - 1, Math.max(0, Math.floor(absoluteLeft / span)));
    columns.scrollLeft = Math.round(page * span);
    return page;
  }}
  function installFindHighlights() {{
    clearFindHighlights();
    if (!(window.CSS && CSS.highlights && window.Highlight)) return;
    ensureFindHighlightStyle();
    var ranges = [];
    var p = 0;
    while ((p = lowerFlat.indexOf(lowerQuery, p)) !== -1) {{
      var r = rangeForIndex(p);
      if (r) ranges.push(r);
      p += Math.max(1, query.length);
    }}
    var highlight = new Highlight();
    ranges.forEach(function(r) {{ highlight.add(r); }});
    CSS.highlights.set('glossarion-find', highlight);
  }}
  var lowerFlat = flat.toLocaleLowerCase();
  clearFindHighlights();
  var seen = 0;
  var pos = 0;
  while ((pos = lowerFlat.indexOf(lowerQuery, pos)) !== -1) {{
      if (seen === wanted) {{
        var range = rangeForIndex(pos);
        if (!range) return {{found: false}};
        installFindHighlights();
        var columns = document.getElementById('columns');
        if (columns) {{
          var gap = (typeof _PAGE_GAP !== 'undefined') ? _PAGE_GAP : 0;
          var w = (typeof _pageWidthFor === 'function')
            ? _pageWidthFor(columns)
            : Math.max(1, Math.floor((columns.clientWidth || window.innerWidth || 1) / Math.max(1, visiblePages)));
          var span = Math.max(1, w + gap);
          var pageCount = (typeof _pageCountFor === 'function')
            ? _pageCountFor(columns)
            : Math.max(1, Math.ceil((columns.scrollWidth + gap) / span));
          var rawPage = scrollColumnsToRange(range, columns, span, pageCount);
          var targetPage = rawPage;
          if (visiblePages > 1) {{
            targetPage = Math.max(0, targetPage - (targetPage % visiblePages));
            columns.scrollLeft = Math.round(targetPage * span);
          }}
          if (typeof _CURRENT_PAGE !== 'undefined') _CURRENT_PAGE = targetPage;
          var sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          return {{
            found: true,
            page: targetPage,
            rawPage: rawPage,
            pageCount: pageCount
          }};
        }}
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        var rect = range.getBoundingClientRect();
        window.scrollBy({{ top: rect.top - Math.floor(window.innerHeight * 0.25), left: 0, behavior: 'smooth' }});
        return {{found: true}};
      }}
      seen += 1;
      pos += Math.max(1, query.length);
  }}
  return {{found: false}};
}})();"""
        def _after_select(result):
            if not isinstance(result, dict) or not result.get("found"):
                return
            page = result.get("page", None)
            if page is None or self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
                return
            try:
                page_num = int(page)
            except (TypeError, ValueError):
                return
            page_count = result.get("pageCount", None)
            if page_count is not None:
                try:
                    self._chapter_page_cache[self._current_row] = int(page_count)
                except (TypeError, ValueError):
                    page_count = None
            self._current_page = self._clamp_page_for_layout(
                page_num, page_count)
            self._update_nav_buttons()

        if update_reader:
            browser.page().runJavaScript(js, _after_select)
        else:
            browser.page().runJavaScript(js)

    def _plain_chapter_text(self, html: str) -> str:
        return _epub_plain_chapter_text(html)

    def _count_chapter_matches(self, chapter_idx: int, text: str) -> int:
        if not text or chapter_idx < 0 or chapter_idx >= len(self._chapters):
            return 0
        plain = self._plain_chapter_text(self._chapters[chapter_idx][1])
        return plain.lower().count(text.lower())

    def _select_match_in_chapter(self, chapter_idx: int, text: str,
                                 occurrence: int = 0) -> None:
        """Select *occurrence* of *text*, loading the chapter if needed."""
        occurrence = max(0, int(occurrence or 0))
        count = self._count_chapter_matches(chapter_idx, text)
        if count <= 0:
            return
        occurrence = min(occurrence, count - 1)
        self._search_chapter_idx = chapter_idx
        self._search_last_row = chapter_idx
        self._search_match_index = occurrence
        self._search_match_count = count
        if chapter_idx != self._current_row:
            self._pending_search_text = text
            self._pending_search_index = occurrence
            self._set_toc_selection_for_chapter(chapter_idx)
            self._activate_chapter_index(chapter_idx)
        else:
            self._select_text_occurrence(
                self._active_search_browser(), text, occurrence)
            self._schedule_search_selection_retry(
                chapter_idx, text, occurrence)

    def _on_search_next(self):
        """Enter pressed: find next match across all chapters."""
        text = self._search_bar.text().strip()
        if not text or not self._chapters:
            return
        if self._layout_mode == LAYOUT_ALL:
            # Scroll All is one rendered document containing every chapter.
            # Re-rendering via the chapter-hop path resets WebEngine's find
            # state to the first match, so repeated Enter should just advance
            # the browser's own in-document search.
            self._reader.findText(text)
            return
        n = len(self._chapters)
        browser = self._active_search_browser()

        # In paginated layouts QWebEngine's native find can select text
        # in an off-screen CSS column without turning the page. Drive the
        # same explicit occurrence -> page jump path used by the search
        # results dialog instead, like Calibre's viewer does.
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            if getattr(self, "_search_current_text", "") != text:
                self._search_current_text = text
                self._search_match_index = -1
                self._search_last_row = self._current_row
            current_count = self._count_chapter_matches(
                self._current_row, text)
            next_occurrence = int(
                getattr(self, "_search_match_index", -1) or -1) + 1
            if next_occurrence < current_count:
                self._select_match_in_chapter(
                    self._current_row, text, next_occurrence)
                return
            start = (self._current_row + 1) % n
            for offset in range(n):
                idx = (start + offset) % n
                if self._count_chapter_matches(idx, text) > 0:
                    self._select_match_in_chapter(idx, text, 0)
                    return
            self._search_bar.setStyleSheet(
                self._search_bar.styleSheet() + " QLineEdit { border-color: #c04040; }")
            QTimer.singleShot(800, lambda: self._apply_reader_style())
            return

        # Enter means "next matching HTML/chapter", not "next occurrence
        # inside this same file". Live typing already selects the first
        # match in the active chapter; repeated Enter hops between files.
        last_row = int(getattr(self, '_search_last_row', self._current_row) or 0)
        start = (last_row + 1) % n
        for offset in range(n):
            idx = (start + offset) % n
            count = self._count_chapter_matches(idx, text)
            if count > 0:
                self._search_chapter_idx = idx
                self._search_last_row = idx
                self._search_match_index = 0
                self._search_match_count = count
                if idx != self._current_row:
                    self._pending_search_text = text
                    self._pending_search_index = 0
                    self._set_toc_selection_for_chapter(idx)
                    self._activate_chapter_index(idx)
                else:
                    browser.findText(text)
                return
        self._search_bar.setStyleSheet(
            self._search_bar.styleSheet() + " QLineEdit { border-color: #c04040; }")
        QTimer.singleShot(800, lambda: self._apply_reader_style())
        return

    def _refresh_native_toc_entries(self) -> None:
        """Reload sidecar/embedded TOC data and map it to loaded chapters."""
        self._native_toc_source_entries = _load_reader_native_toc(
            getattr(self, "_toc_output_dir", ""),
            getattr(self, "_epub_path", ""),
        )
        self._native_toc_entries = _map_native_toc_to_chapters(
            self._native_toc_source_entries,
            self._chapter_filenames,
        )
        button = getattr(self, "_native_toc_btn", None)
        if button is not None:
            available = bool(self._native_toc_entries)
            button.setEnabled(available)
            if available:
                source = str(self._native_toc_entries[0].get("source") or "")
                button.setToolTip(
                    "Use the book's native table of contents in the sidebar.\n"
                    f"Source: {source}"
                )
            else:
                button.setToolTip(
                    "No TOC.txt or toc.ncx entries could be matched to this "
                    "book's loaded chapters."
                )

    def _rebuild_toc_sidebar(self, current_chapter: int | None = None) -> None:
        """Populate the sidebar from HTML headings or mapped native TOC rows."""
        if current_chapter is None:
            current_chapter = self._current_row
        old_blocked = self._toc_list.blockSignals(True)
        try:
            self._toc_list.clear()
            self._toc_row_to_chapter = []
            use_native = bool(
                self._native_toc_enabled and self._native_toc_entries)
            if use_native:
                for entry in self._native_toc_entries:
                    chapter_index = int(entry.get("chapter_index", 0))
                    item = QListWidgetItem(str(entry.get("title") or "Section"))
                    item.setData(Qt.UserRole, chapter_index)
                    item.setData(Qt.UserRole + 1, str(entry.get("fragment") or ""))
                    self._toc_list.addItem(item)
                    self._toc_row_to_chapter.append(chapter_index)
            else:
                for chapter_index, (title, _content) in enumerate(self._chapters):
                    item = QListWidgetItem(title)
                    item.setData(Qt.UserRole, chapter_index)
                    self._toc_list.addItem(item)
                    self._toc_row_to_chapter.append(chapter_index)
            self._set_toc_selection_for_chapter(
                int(current_chapter), signals_already_blocked=True)
        finally:
            self._toc_list.blockSignals(old_blocked)

    def _set_toc_selection_for_chapter(
        self,
        chapter_index: int,
        *,
        signals_already_blocked: bool = False,
    ) -> None:
        """Highlight the sidebar row mapped to *chapter_index*, if present."""
        try:
            toc_row = self._toc_row_to_chapter.index(int(chapter_index))
        except (ValueError, TypeError):
            toc_row = -1
        if signals_already_blocked:
            self._toc_list.setCurrentRow(toc_row)
            return
        old_blocked = self._toc_list.blockSignals(True)
        try:
            self._toc_list.setCurrentRow(toc_row)
        finally:
            self._toc_list.blockSignals(old_blocked)

    def _toc_chapter_index_for_row(self, toc_row: int) -> int:
        """Resolve a visible sidebar row to its underlying spine index."""
        if toc_row < 0 or toc_row >= len(self._toc_row_to_chapter):
            return -1
        return int(self._toc_row_to_chapter[toc_row])

    def _on_native_toc_toggled(self, checked: bool) -> None:
        """Switch sidebar sources and retain the choice in reader config."""
        self._native_toc_enabled = bool(checked)
        self._config['epub_reader_native_toc'] = self._native_toc_enabled
        _persist_config_via_parent(self)
        self._refresh_native_toc_entries()
        saved_sizes = self._begin_toc_width_lock()
        self._rebuild_toc_sidebar(current_chapter=self._current_row)
        self._end_toc_width_lock(saved_sizes)

    def _toggle_toc(self):
        """Show or hide the TOC sidebar using splitter sizes."""
        sizes = self._splitter.sizes()
        if sizes[0] > 0:
            # Hide TOC: remember width, set to 0
            self._toc_saved_width = sizes[0]
            self._splitter.setSizes([0, sizes[1] + sizes[0]])
        else:
            # Show TOC: restore saved width
            w = getattr(self, '_toc_saved_width', 220)
            self._splitter.setSizes([w, max(1, sizes[1] - w)])
        # The reader viewport width just changed. The in-page _setupColumns
        # resize handler re-anchors the transform to _CURRENT_PAGE atomically,
        # so no opacity flash is needed here — we only need to refresh the
        # Python-side page-count cache that drives nav-button state.
        self._resync_page_count()

    def _resync_page_count(self):
        """Requery page count after the paginated viewport changes.

        Qt can report one or more transitional WebEngine geometries while a
        hidden reader stack is being shown or a splitter is moving.  Require
        two matching measurements before committing the count so those
        short-lived layouts cannot strand a real final column outside the
        Python navigation bounds.
        """
        if not hasattr(self, '_chapter_page_cache'):
            return
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if not self._chapters:
            return
        self._chapter_page_cache.clear()
        token = int(getattr(self, '_pagination_resync_token', 0) or 0) + 1
        self._pagination_resync_token = token
        expected_row = self._current_row
        expected_layout = self._layout_mode
        state = {"last": None, "matches": 0, "attempts": 0}

        def _still_current():
            return (
                not getattr(self, '_closing', False)
                and getattr(self, '_pagination_resync_token', 0) == token
                and self._current_row == expected_row
                and self._layout_mode == expected_layout
            )

        def _commit(count):
            if not _still_current():
                return
            count = max(1, int(count))
            self._chapter_page_cache[expected_row] = count
            # Clamp if the current page is now past the end (e.g. column
            # count shrank) and silently jump to the clamped position.
            clamped_page = self._clamp_page_for_layout(
                self._current_page, count)
            if self._current_page != clamped_page:
                self._current_page = clamped_page
                self._js_scroll_to(
                    self._reader, self._current_page, animate=False)
            self._update_nav_buttons()
            self._schedule_search_realign()

        def _measure():
            if not _still_current():
                return

            def _on_count(count):
                if not _still_current():
                    return
                count = max(1, int(count))
                state["attempts"] += 1
                if count == state["last"]:
                    state["matches"] += 1
                else:
                    state["last"] = count
                    state["matches"] = 1
                # Two consecutive equal counts indicate that Chromium has
                # finished applying the visible viewport.  The retry cap is
                # a safety valve for pages with continuously resizing media.
                if state["matches"] >= 2 or state["attempts"] >= 8:
                    _commit(count)
                    return
                QTimer.singleShot(80, _measure)

            self._js_page_count(self._reader, _on_count)

        # Give the browser a tick to process the splitter/visibility change
        # before beginning the stability measurements.
        QTimer.singleShot(60, _measure)

    def _change_font_size(self, delta):
        self._font_size = max(8, min(32, self._font_size + delta))
        self._font_label.setText(f"{self._font_size}pt")
        self._apply_reader_style()
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
        # Hide content before re-render to prevent flash
        _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
        if _HAS_WEBENGINE:
            self._reader.page().runJavaScript(_hide)
        self._render_current()

    def _on_reader_wheel_scrolled(self, delta_y: int, modifiers) -> None:
        """Route a wheel event from a reader pane to the right action.

        :class:`_WheelCapturingView` consumes every wheel event its
        Chromium child emits and routes it here. We dispatch based on
        modifier + current layout mode:

          * **Ctrl + wheel**  → font size zoom (matches Ctrl+= / Ctrl+-
            shortcuts). Works in every layout.
          * **Plain wheel in paginated modes** (single / double page)
            → previous / next page. Scrolling up turns back, scrolling
            down turns forward — matches what Chromium would do to a
            scrollable page.
          * **Plain wheel in scroll modes** (single-chapter scroll,
            all-scroll) → fall back to a manual ``window.scrollBy``
            inside the page so the normal scrolling behaviour is
            preserved (we swallowed the event at the filter level, so
            Chromium won't do it for us).
        """
        if not delta_y:
            return
        # Ctrl + wheel: font zoom regardless of layout.
        try:
            ctrl = bool(modifiers & Qt.ControlModifier)
        except Exception:
            ctrl = False
        if ctrl:
            self._change_font_size(1 if delta_y > 0 else -1)
            return
        # Paginated modes: turn pages.
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            if delta_y > 0:
                self._prev_chapter()
            else:
                self._next_chapter()
            return
        # Scroll modes: replicate the browser's own wheel-scroll since
        # the event filter already swallowed the native behaviour. We
        # invert the sign because angleDelta y>0 means "wheel up" and
        # scrollBy expects positive values to scroll the page DOWN.
        if not _HAS_WEBENGINE:
            return
        pixels = int(-delta_y)
        js = f"window.scrollBy({{top: {pixels}, left: 0, behavior: 'auto'}});"
        try:
            self._reader.page().runJavaScript(js)
        except Exception:
            logger.debug("Wheel scroll passthrough failed: %s",
                         traceback.format_exc())

    def _on_font_family_changed(self, family):
        """Update reader font family and re-render."""
        family = (family or "").strip()
        if not family or family == self._font_family:
            return
        self._font_family = family
        # Persist immediately so the choice survives crashes / force-closes
        self._config['epub_reader_font_family'] = family
        _persist_config_via_parent(self)
        # Clear embedded CSS cache if switching away from or to Embedded CSS
        if hasattr(self, '_embedded_css_cache'):
            del self._embedded_css_cache
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
        # Hide content before re-render to prevent flash
        _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
        if _HAS_WEBENGINE:
            self._reader.page().runJavaScript(_hide)
        self._render_current()

    def _on_spacing_changed(self, text):
        try:
            val = float(text)
            self._line_spacing = max(1.0, min(3.0, val))
        except ValueError:
            self._line_spacing = 1.8
        self._chapter_page_cache.clear()
        self._loaded_chapter = -1
        # Hide content before re-render to prevent flash
        _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
        if _HAS_WEBENGINE:
            self._reader.page().runJavaScript(_hide)
        self._render_current()

    def _on_theme_changed(self, index):
        self._theme_index = index
        # Update QWebEngineView page backgrounds to prevent flash
        if _HAS_WEBENGINE:
            from PySide6.QtGui import QColor
            t = self._get_theme()
            bg = QColor(t['bg'])
            for w in [self._reader, self._reader_left, self._reader_right]:
                if hasattr(w, 'page'):
                    w.page().setBackgroundColor(bg)
        self._apply_reader_style()
        self._loaded_chapter = -1  # force re-render for inline styles
        self._render_current()

    # ── Layout mode ────────────────────────────────────────────────────────

    def _on_layout_changed(self, index):
        modes = [LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_SCROLL, LAYOUT_ALL]
        old_mode = self._layout_mode
        old_page = self._current_page
        old_count = self._chapter_page_cache.get(self._current_row, 0)
        new_mode = modes[index] if index < len(modes) else LAYOUT_SINGLE
        self._layout_mode = new_mode
        if old_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE) and new_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            self._current_page = self._clamp_page_for_layout(old_page, old_count)
        else:
            self._current_page = 0
        self._loaded_chapter = -1  # force re-render on layout change
        self._chapter_page_cache.clear()
        self._render_current()

    def _ensure_workspace_raw_chapter(self, row: int) -> bool:
        """Start the one-section PDF raw cache on demand.

        Returns ``True`` when the requested raw chapter is already available.
        A different in-flight range is allowed to finish, after which the most
        recently requested row is started.  At no point is the entire PDF
        walked merely because the reader was opened or Raw was toggled.
        """
        if not (self._workspace_mode and self._workspace_has_raw):
            return True
        entries = self._workspace_manifest.get("entries", []) or []
        if not (0 <= row < len(entries)):
            return False
        if row in self._workspace_raw_ready:
            return True

        running = getattr(self, "_workspace_raw_thread", None)
        if running is not None:
            try:
                if running.isRunning():
                    if getattr(running, "_row", None) != row:
                        self._workspace_raw_pending_row = row
                    return False
            except RuntimeError:
                pass

        title = self._chapters_raw[row][0]
        self._chapters_raw[row] = (
            title,
            _workspace_reader_placeholder(
                title,
                "Loading raw PDF pages for this bookmark... "
                "Image-heavy sections may take a moment.",
            ),
        )
        mode = str(self._config.get("pdf_render_mode", "fast_semantic") or "")
        if mode not in ("fast_semantic", "fast_layout"):
            mode = "fast_semantic"
        extract_images = str(os.environ.get("PDF_EXTRACT_IMAGES", "1")).strip().lower() \
            not in ("0", "false", "no", "off")
        thread = _PdfRawSectionLoaderThread(
            row,
            self._workspace_manifest,
            entries[row],
            mode,
            extract_images,
            parent=self,
        )
        thread.done.connect(
            lambda loaded_row, content:
                self._on_workspace_raw_loaded(loaded_row, content)
        )
        thread.error.connect(
            lambda failed_row, message:
                self._on_workspace_raw_error(failed_row, message)
        )
        self._workspace_raw_thread = thread
        # Rendering the placeholder and starting MuPDF in the same event-loop
        # turn can prevent the message from becoming visible until extraction
        # has already finished. Give the web view one paint opportunity first.
        QTimer.singleShot(
            self._RAW_PDF_WORKER_PAINT_DELAY_MS,
            lambda pending=thread: self._start_workspace_raw_thread(pending),
        )
        return False

    def _start_workspace_raw_thread(self, thread) -> None:
        """Start the still-current delayed raw-PDF worker."""
        if getattr(self, "_closing", False):
            return
        if thread is not getattr(self, "_workspace_raw_thread", None):
            return
        try:
            if not thread.isRunning():
                thread.start()
        except RuntimeError:
            pass

    @Slot(int, str)
    def _on_workspace_raw_loaded(self, row: int, content: str) -> None:
        if getattr(self, "_closing", False):
            return
        if 0 <= row < len(self._chapters_raw):
            title = self._chapters_raw[row][0]
            self._chapters_raw[row] = (title, str(content or ""))
            self._workspace_raw_ready.add(row)
        self._workspace_raw_thread = None
        if self._show_raw and row == self._current_row:
            self._chapters = self._chapters_raw
            self._chapter_page_cache.pop(row, None)
            self._loaded_chapter = -1
            self._render_current()
        pending = self._workspace_raw_pending_row
        self._workspace_raw_pending_row = None
        if (isinstance(pending, int)
                and pending not in self._workspace_raw_ready
                and self._show_raw):
            QTimer.singleShot(0, lambda r=pending:
                              self._ensure_workspace_raw_chapter(r))

    @Slot(int, str)
    def _on_workspace_raw_error(self, row: int, message: str) -> None:
        if getattr(self, "_closing", False):
            return
        logger.warning("Could not load raw PDF section %s: %s", row, message)
        if 0 <= row < len(self._chapters_raw):
            title = self._chapters_raw[row][0]
            self._chapters_raw[row] = (
                title,
                _workspace_reader_placeholder(
                    title,
                    f"Could not load this raw PDF section: {message}",
                ),
            )
            self._workspace_raw_ready.add(row)
        self._workspace_raw_thread = None
        if self._show_raw and row == self._current_row:
            self._chapters = self._chapters_raw
            self._loaded_chapter = -1
            self._render_current()
        pending = self._workspace_raw_pending_row
        self._workspace_raw_pending_row = None
        if (isinstance(pending, int)
                and pending not in self._workspace_raw_ready
                and self._show_raw):
            QTimer.singleShot(0, lambda r=pending:
                              self._ensure_workspace_raw_chapter(r))

    def _render_current(self):
        """Re-render the current chapter in the active layout mode."""
        if not self._chapters:
            return
        # Async WebEngine callbacks from an older render must never finalize
        # pagination or visibility after a newer chapter/flavor has won.
        self._reader_render_generation = int(getattr(
            self, "_reader_render_generation", 0
        ) or 0) + 1
        if self._workspace_mode and self._show_raw:
            self._ensure_workspace_raw_chapter(self._current_row)
        self._apply_reader_style()  # refresh theme
        # Translate button only applies to not-yet-translated chapters.
        try:
            self._update_translate_btn_visibility()
        except Exception:
            pass

        row = self._current_row

        def _set_html(browser, html):
            """Set HTML on either QWebEngineView or QTextBrowser.

            QWebEngineView.setHtml() has a ~2MB limit — base64 images
            easily exceed this.  Write to a temp file and load via URL.
            """
            if _HAS_WEBENGINE:
                tmp_dir = _epub_cache_dir()
                tmp_path = os.path.join(tmp_dir, f"_reader_{id(browser)}.html")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(html)
                serial = int(getattr(self, "_reader_html_serial", 0) or 0) + 1
                self._reader_html_serial = serial
                url = QUrl.fromLocalFile(tmp_path)
                url.setQuery(f"v={serial}")
                browser.setUrl(url)
            else:
                browser.setHtml(html)

        if self._layout_mode == LAYOUT_ALL:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.hide()
            all_html = self._all_chapters_html()
            _set_html(self._reader, self._wrap_html(all_html, paginated=False))
            self._loaded_chapter = -1
            self._toc_list.blockSignals(True)
            self._toc_list.setCurrentRow(-1)
            self._toc_list.blockSignals(False)

        elif self._layout_mode == LAYOUT_DOUBLE:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.show()
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                _set_html(self._reader, self._wrap_html(
                    html, paginated=True, spread_pages=2))
                self._loaded_chapter = row
                # finalization happens via loadFinished signal
            self._update_nav_buttons()

        elif self._layout_mode == LAYOUT_SINGLE:
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.show()
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                _set_html(self._reader, self._wrap_html(html, paginated=True))
                self._loaded_chapter = row
                # finalization happens via loadFinished signal
            self._update_nav_buttons()

        else:  # LAYOUT_SCROLL
            self._reader_stack.setCurrentIndex(0)
            self._nav_bar.hide()
            if row < len(self._chapters):
                html = self._process_html(self._chapters[row][1])
                _set_html(self._reader, self._wrap_html(html, paginated=False))
                self._loaded_chapter = row

        if not _HAS_WEBENGINE and self._layout_mode in (LAYOUT_SCROLL, LAYOUT_ALL):
            # QTextBrowser installs its document synchronously; WebEngine uses
            # the loadFinished hook below after the replacement page is ready.
            self._apply_pending_scroll_hint()

        # Let Chromium begin the current page load before starting background
        # work for the following chapter.
        QTimer.singleShot(0, self._schedule_next_chapter_image_preload)

    # _all_chapters_html was extracted from _render_current into reader_doc.ReaderDocMixin (see DISCREPANCIES U5 "Phase-1 splits").

    # ── Pagination helpers (CSS column-based) ───────────────────────────────

    # ── Right-click context menu on the reader pane ────────────────────

    def _get_reader_selection(self, browser) -> str:
        """Return the currently-selected text in *browser*, or ``""``.

        ``QWebEngineView.selectedText`` is synchronous (Qt >= 5.7) so it's
        safe to call from the context-menu handler. ``QTextBrowser``
        exposes the selection via its cursor instead. Both paths are
        wrapped in try/except so a Qt oddity can't block the menu.
        """
        try:
            if _HAS_WEBENGINE and hasattr(browser, "selectedText"):
                return browser.selectedText() or ""
            if hasattr(browser, "textCursor"):
                return browser.textCursor().selectedText() or ""
        except Exception:
            logger.debug("Reader selection read failed: %s",
                         traceback.format_exc())
        return ""

    def _show_reader_context_menu(self, browser, pos):
        """Populate + show the right-click menu for *browser* at *pos*.

        Menu contents are scoped to the reader's current flavor so the
        verb matches what the user is looking at:

          * **Raw** mode — the selection is source-language text, so the
            sole action is **Google Translate** (``translate.google.com``
            with ``sl=auto`` + ``tl=<target_language>``). This is the
            manual-MT sanity-check flow the reader was built for.
          * **Translated** mode — the selection is already in the target
            language, so instead we offer **Define on web**, a Google
            search with the ``define`` operator that surfaces the
            dictionary card for the selected word / phrase.

        Both entries are disabled when no text is selected so the menu
        reads clearly instead of silently doing nothing.
        """
        selected = (self._get_reader_selection(browser) or "").strip()
        has_selection = bool(selected)
        target_lang = (self._config.get("output_language") or "English").strip() or "English"
        target_code = _target_lang_to_google_code(target_lang)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 9pt; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background: #3a3a5e; }
            QMenu::item:disabled { color: #555; }
        """)

        if getattr(self, "_show_raw", False):
            gt_action = menu.addAction(
                f"\U0001f310  Google Translate \u2192 {target_lang}")
            gt_action.setToolTip(
                f"Open translate.google.com in your default browser with the "
                f"selection machine-translated into {target_lang} "
                f"(source language auto-detected)."
            )
            gt_action.setEnabled(has_selection)
            gt_action.triggered.connect(
                lambda: self._open_google_translate(selected, target_code))
        else:
            def_action = menu.addAction("\U0001f4d6  Define on web")
            def_action.setToolTip(
                "Look up the selected word / phrase on Google using the "
                "'define' operator — surfaces the dictionary card above "
                "the regular search results."
            )
            def_action.setEnabled(has_selection)
            def_action.triggered.connect(
                lambda: self._open_web_define(selected))

        menu.exec(browser.mapToGlobal(pos))

    def _open_google_translate(self, text: str, target_code: str) -> None:
        """Hand off *text* to translate.google.com via the default browser."""
        url = _google_translate_url(text, target_code)
        if not url:
            return
        from PySide6.QtGui import QDesktopServices
        try:
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            logger.debug("Google Translate open failed: %s",
                         traceback.format_exc())

    def _open_web_define(self, text: str) -> None:
        """Open Google's ``define:`` card for *text* in the default browser.

        Uses the ``define`` query prefix rather than a bare search so the
        dictionary entry (with pronunciation, part of speech, and
        definitions) renders at the top of the results page. For
        multi-word selections Google still surfaces the best-matching
        dictionary card; when no dictionary hit exists Google silently
        degrades to normal results.
        """
        url = _define_url(text)
        if not url:
            return
        from PySide6.QtGui import QDesktopServices
        try:
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            logger.debug("Web define open failed: %s",
                         traceback.format_exc())

    def _on_reader_load_finished(self, ok):
        """Called when QWebEngineView finishes loading HTML."""
        if not ok:
            # Never strand a raw/translated transition with the browser pane
            # hidden merely because WebEngine rejected/cancelled a load.
            self._finish_raw_toggle()
            return
        # Ignore stray load events that fire before a real chapter was
        # ever queued up. ``_make_reader_widget`` calls
        # ``setUrl("about:blank")`` to prime the view, which asynchronously
        # triggers ``loadFinished`` after :meth:`_setup_ui` has connected
        # this handler but before any chapter data is present. Without
        # this guard, :meth:`_finalize_single_page` runs against an
        # empty state and crashes the Python callback pipeline.
        if not self._chapters:
            self._finish_raw_toggle()
            return
        # Prime phase: the scroll-mode render has just completed. Swap
        # back to the configured paginated mode immediately — this is the
        # event-driven replacement for the previous fixed-delay timer.
        if (getattr(self, '_priming_initial_render', False)
                and self._layout_mode == LAYOUT_SCROLL
                and self.sender() is self._reader):
            self._layout_mode = getattr(self, '_prime_saved_mode', LAYOUT_SINGLE)
            self._loaded_chapter = -1
            self._chapter_page_cache.clear()
            self._render_current()
            return
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            if self.sender() is self._reader:
                self._apply_pending_scroll_hint()
                self._reveal_initial_reader_shell()
            if (self._layout_mode in (LAYOUT_SCROLL, LAYOUT_ALL)
                    and self.sender() is self._reader
                    and getattr(self, '_pending_search_text', None)):
                self._consume_pending_search(self._reader)
            if self.sender() is self._reader:
                self._finish_raw_toggle()
            return
        # The signal fires from whichever browser finished loading; route
        # based on the sender rather than layout alone so stale loads from
        # the single-reader don't confuse the double-page pending counter.
        sender = self.sender()
        if self._layout_mode == LAYOUT_SINGLE:
            if sender is self._reader:
                self._finalize_single_page()
        elif self._layout_mode == LAYOUT_DOUBLE:
            if sender is not self._reader:
                return
            self._finalize_double_page()

    def _clamp_page_for_layout(self, page_num, page_count=None) -> int:
        """Clamp a paginated target to a valid single page or spread start."""
        try:
            page = int(page_num)
        except (TypeError, ValueError):
            page = 0
        if page_count is None:
            page_count = self._chapter_page_cache.get(self._current_row, 0)
        try:
            count = int(page_count)
        except (TypeError, ValueError):
            count = 0
        count = max(1, count)
        if self._layout_mode == LAYOUT_DOUBLE:
            page = max(0, page - (page % 2))
            last_start = max(0, count - 1)
            last_start -= last_start % 2
            return max(0, min(page, last_start))
        return max(0, min(page, count - 1))

    def _js_scroll_to(self, browser, page_num, animate: bool = True):
        """Navigate to a CSS column page using native inline scrolling.

        When *animate* is False, the CSS transition is suppressed for the
        jump (used right after a load so the page doesn't visibly slide in
        from position 0). Translate values are rounded to whole pixels and
        use translate3d to keep text on a stable GPU layer — otherwise
        fractional offsets produce subpixel LCD-antialiasing fringing
        that looks like a red/colored shift on the text.
        """
        if _HAS_WEBENGINE:
            if animate:
                js = (
                    "var c = document.getElementById('columns');"
                    "if (c) {"
                    "  if (typeof _setupColumns==='function') _setupColumns();"
                    "  var gap = (typeof _PAGE_GAP!=='undefined')?_PAGE_GAP:0;"
                    "  var w = (typeof _pageWidthFor==='function')"
                    "    ? _pageWidthFor(c)"
                    "    : Math.max(1, Math.floor(c.clientWidth || window.innerWidth || 1));"
                    "  var span = Math.max(1, w + gap);"
                    # Record the target page so _setupColumns can re-anchor
                    # the transform on the next viewport-width change.
                    f"  _CURRENT_PAGE = {page_num};"
                    "  c.style.transition = 'none';"
                    "  c.style.transform = 'none';"
                    f"  c.scrollLeft = Math.round({page_num} * span);"
                    "}"
                )
            else:
                js = (
                    "var c = document.getElementById('columns');"
                    "if (c) {"
                    "  if (typeof _setupColumns==='function') _setupColumns();"
                    "  var gap = (typeof _PAGE_GAP!=='undefined')?_PAGE_GAP:0;"
                    "  var w = (typeof _pageWidthFor==='function')"
                    "    ? _pageWidthFor(c)"
                    "    : Math.max(1, Math.floor(c.clientWidth || window.innerWidth || 1));"
                    "  var span = Math.max(1, w + gap);"
                    f"  _CURRENT_PAGE = {page_num};"
                    "  c.style.transition = 'none';"
                    "  c.style.transform = 'none';"
                    f"  c.scrollLeft = Math.round({page_num} * span);"
                    "  void c.offsetHeight;"  # force reflow so the jump is committed w/o transition
                    "}"
                )
            browser.page().runJavaScript(js)
        else:
            vp = browser.viewport()
            h = vp.height()
            if h > 0:
                browser.verticalScrollBar().setValue(page_num * h)

    def _js_page_count(self, browser, callback):
        """Get page count from CSS column layout (#columns scrollWidth / page width)."""
        if _HAS_WEBENGINE:
            js = (
                "var c = document.getElementById('columns');"
                "if (typeof _setupColumns==='function') _setupColumns();"
                "if (!c) 1;"
                "else if (typeof _pageCountFor==='function') _pageCountFor(c);"
                "else {"
                "  var gap = (typeof _PAGE_GAP!=='undefined')?_PAGE_GAP:0;"
                "  var w = Math.max(1, Math.floor(c.clientWidth || window.innerWidth || 1));"
                "  var span = Math.max(1, w + gap);"
                # ``scrollWidth`` is integer-rounded by Chromium.  Ceiling
                # preserves a partially measured final column; floor could
                # make its content unreachable at fractional Windows DPI.
                "  Math.max(1, Math.ceil((c.scrollWidth + gap) / span));"
                "}"
            )
            browser.page().runJavaScript(js, callback)
        else:
            doc = browser.document()
            vp = browser.viewport()
            w, h = vp.width(), vp.height()
            if w <= 0 or h <= 0:
                callback(1)
                return
            doc.setPageSize(QSizeF(w, h))
            callback(max(1, doc.pageCount()))

    def _apply_pending_page_hint(self, count: int) -> None:
        """Resolve ``_pending_page_hint`` (if any) against *count* pages.

        Called by the single / double finalizers after the fresh chapter's
        page count is known. The hint expresses "I was on the last page"
        or a proportional position so the Show-raw toggle can leave the
        user on an equivalent page in the new flavor (page counts drift
        between translations).
        """
        hint = getattr(self, '_pending_page_hint', None)
        if not hint:
            return
        # Only apply the hint if it was recorded against the chapter
        # we're now paginating — otherwise a stale hint from a prior
        # chapter could hijack the position.
        if hint.get("row") not in (None, self._current_row):
            self._pending_page_hint = None
            return
        c = max(1, int(count))
        if hint.get("was_last_page"):
            self._current_page = c - 1
        else:
            prop = float(hint.get("proportion") or 0.0)
            target = round(prop * (c - 1))
            self._current_page = max(0, min(int(target), c - 1))
        self._current_page = self._clamp_page_for_layout(self._current_page, c)
        self._pending_page_hint = None

    def _consume_pending_search(self, browser):
        """If a cross-chapter search stored ``_pending_search_text``,
        select the requested occurrence after the fresh chapter has loaded.
        """
        text = getattr(self, '_pending_search_text', None)
        if not text:
            return
        self._pending_search_text = None
        occurrence = int(getattr(self, '_pending_search_index', 0) or 0)
        self._pending_search_index = 0
        self._search_match_index = occurrence
        self._select_text_occurrence(browser, text, occurrence)
        self._schedule_search_selection_retry(
            self._current_row, text, occurrence)

    def _finalize_single_page(self):
        """After HTML load: get page count and scroll to current page."""
        generation = int(getattr(self, "_reader_render_generation", 0) or 0)

        def on_count(count):
            if generation != int(getattr(
                    self, "_reader_render_generation", 0) or 0):
                return
            count = int(count)
            self._chapter_page_cache[self._current_row] = count
            # Restore the pre-swap reading position when a Show-raw
            # toggle (or equivalent reload) staged a hint.
            self._apply_pending_page_hint(count)
            self._current_page = self._clamp_page_for_layout(
                self._current_page, count)
            # animate=False: jump instantly so the reader doesn't visibly
            # slide from page 1 to the current page on theme/chapter change.
            self._js_scroll_to(self._reader, self._current_page, animate=False)
            def after_reveal():
                if generation != int(getattr(
                        self, "_reader_render_generation", 0) or 0):
                    return
                self._update_nav_buttons()
                self._reveal_reader_stack_after_prime()
                self._consume_pending_search(self._reader)
                self._finish_raw_toggle()
            self._js_reveal(self._reader, after_reveal)
        self._js_page_count(self._reader, on_count)

    def _finalize_double_page(self):
        """After HTML load: get page count and position both panes."""
        generation = int(getattr(self, "_reader_render_generation", 0) or 0)

        def on_count(count):
            if generation != int(getattr(
                    self, "_reader_render_generation", 0) or 0):
                return
            count = int(count)
            self._chapter_page_cache[self._current_row] = count
            self._apply_pending_page_hint(count)
            self._current_page = self._clamp_page_for_layout(
                self._current_page, count)
            self._js_scroll_to(self._reader, self._current_page, animate=False)
            def after_reveal():
                if generation != int(getattr(
                        self, "_reader_render_generation", 0) or 0):
                    return
                self._update_nav_buttons()
                self._reveal_reader_stack_after_prime()
                self._consume_pending_search(self._reader)
                self._finish_raw_toggle()
            self._js_reveal(self._reader, after_reveal)
        self._js_page_count(self._reader, on_count)

    def _reveal_reader_stack_after_prime(self):
        """Reveal the reader stack after the post-prime paginated render
        completes. No-op if the prime sequence wasn't used.
        """
        if not getattr(self, '_priming_initial_render', False):
            return
        self._priming_initial_render = False
        self._reader_stack.show()
        sizes = getattr(self, '_prime_toc_sizes', None)
        self._prime_toc_sizes = None
        self._end_toc_width_lock(sizes)
        self._reveal_initial_reader_shell()
        # The first paginated page count is measured while ``_reader_stack``
        # is hidden.  Showing it gives QWebEngine its real viewport height and
        # can reflow the chapter into a different number of CSS columns.  The
        # browser-side resize handler updates the columns themselves, but the
        # Python cache that drives Next/Previous used to retain the hidden
        # geometry's count.  When that count was one page short, the final
        # column existed in the DOM but could never be reached.  Requery on
        # the next event-loop turn, after Qt has laid out the visible stack.
        QTimer.singleShot(0, self._resync_page_count)

    def _scroll_to_page_single(self):
        """Navigate single-page reader to current page."""
        self._js_scroll_to(self._reader, self._current_page)
        self._update_nav_buttons()

    def _scroll_to_page_double(self):
        """Navigate double-page panes to current page."""
        self._js_scroll_to(self._reader, self._current_page)
        self._update_nav_buttons()

    def _js_reveal(self, browser, callback=None):
        """Show the positioned columns, then notify the shell handoff."""
        if _HAS_WEBENGINE:
            js = (
                "var c = document.getElementById('columns'); "
                "if (c) c.style.opacity = '1';"
            )
            if callback:
                browser.page().runJavaScript(
                    js, lambda _result: callback()
                )
            else:
                browser.page().runJavaScript(js)
        elif callback:
            callback()

    def resizeEvent(self, event):
        """Invalidate page cache on resize, preserving reading position."""
        super().resizeEvent(event)
        self._refresh_pagination_viewport(delay=100)

    def _refresh_pagination_viewport(self, delay: int = 120):
        """Recompute paginated layout after viewport width changes.

        Used by resizeEvent and TOC toggle — anything that changes the
        reader widget's width without rebuilding the HTML.
        """
        if not hasattr(self, '_chapter_page_cache'):
            return
        if self._layout_mode not in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            return
        if not self._chapters:
            return
        # Save scroll proportion before clearing cache
        old_count = self._chapter_page_cache.get(self._current_row, 0)
        old_page = self._current_page
        self._chapter_page_cache.clear()
        # Proportion-based: map old position to new page count
        proportion = old_page / max(1, old_count) if old_count > 0 else 0
        # Hide content immediately to prevent image flash during resize
        _hide_js = "var c = document.getElementById('columns'); if (c) { c.style.transition = 'none'; c.style.opacity = '0'; }"
        _reveal_js = "var c = document.getElementById('columns'); if (c) { c.style.transition = 'transform 0.25s ease'; c.style.opacity = '1'; }"
        self._reader.page().runJavaScript(_hide_js)

        def _on_resize_recount():
            if self._layout_mode == LAYOUT_SINGLE:
                def on_count(count):
                    count = int(count)
                    self._chapter_page_cache[self._current_row] = count
                    self._current_page = self._clamp_page_for_layout(
                        round(proportion * count), count)
                    self._js_scroll_to(self._reader, self._current_page)
                    # Reveal after scroll
                    QTimer.singleShot(30, lambda: self._reader.page().runJavaScript(_reveal_js))
                    self._update_nav_buttons()
                    self._schedule_search_realign()
                self._js_page_count(self._reader, on_count)
            else:
                def on_count(count):
                    count = int(count)
                    self._chapter_page_cache[self._current_row] = count
                    self._current_page = self._clamp_page_for_layout(
                        round(proportion * count), count)
                    self._js_scroll_to(self._reader, self._current_page)
                    QTimer.singleShot(30, lambda: self._reader.page().runJavaScript(_reveal_js))
                    self._update_nav_buttons()
                    self._schedule_search_realign()
                self._js_page_count(self._reader, on_count)
        QTimer.singleShot(delay, _on_resize_recount)

    # _reader_chapter_display_number moved verbatim to reader_doc.ReaderDocMixin (inherited).

    def _update_nav_buttons(self):
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            ch_pages = self._get_chapter_pages(self._current_row)
            step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
            self._prev_btn.setEnabled(self._current_page > 0 or self._current_row > 0)
            self._next_btn.setEnabled(
                self._current_page + step < ch_pages or
                self._current_row < len(self._chapters) - 1
            )
            cur_global, total_global = self._get_global_page_info()
            self._page_label.setText(f"Page {cur_global}/{total_global}")
        else:
            self._prev_btn.setEnabled(self._current_row > 0)
            self._next_btn.setEnabled(self._current_row < len(self._chapters) - 1)
            chapter_number = self._reader_chapter_display_number(self._current_row)
            self._page_label.setText(
                f"Chapter {chapter_number} of {len(self._chapters)}"
            )

    def _prev_chapter(self):
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
            if self._current_page >= step:
                # Same chapter, just scroll
                self._current_page -= step
                if self._layout_mode == LAYOUT_SINGLE:
                    self._scroll_to_page_single()
                else:
                    self._scroll_to_page_double()
            elif self._current_row > 0:
                # Go to previous chapter's last page
                self._current_row -= 1
                self._set_toc_selection_for_chapter(self._current_row)
                # Set page to last page of previous chapter (will be clamped in finalize)
                pages = self._get_chapter_pages(self._current_row)
                self._current_page = self._clamp_page_for_layout(pages - 1, pages)
                self._render_current()
        else:
            new_row = max(0, self._current_row - 1)
            self._current_row = new_row
            self._set_toc_selection_for_chapter(new_row)
            self._render_current()

    def _next_chapter(self):
        if self._layout_mode in (LAYOUT_SINGLE, LAYOUT_DOUBLE):
            if _HAS_WEBENGINE:
                # Recheck the live column layout before deciding that this is
                # the chapter's last page.  This prevents a stale or rounded
                # cache entry from skipping a final column and jumping
                # directly to the next chapter.
                expected_row = self._current_row
                expected_page = self._current_page
                expected_layout = self._layout_mode

                def _on_count(count):
                    if (self._current_row != expected_row
                            or self._current_page != expected_page
                            or self._layout_mode != expected_layout):
                        return
                    count = max(1, int(count))
                    self._chapter_page_cache[expected_row] = count
                    self._advance_paginated_next(count)

                self._js_page_count(self._reader, _on_count)
            else:
                self._advance_paginated_next(
                    self._get_chapter_pages(self._current_row))
        else:
            new_row = min(len(self._chapters) - 1, self._current_row + 1)
            self._set_toc_selection_for_chapter(new_row)
            self._activate_chapter_index(new_row)

    def _advance_paginated_next(self, ch_pages):
        """Advance one page/spread using a freshly resolved chapter count."""
        step = 2 if self._layout_mode == LAYOUT_DOUBLE else 1
        ch_pages = max(1, int(ch_pages))
        if self._current_page + step < ch_pages:
            self._current_page += step
            if self._layout_mode == LAYOUT_SINGLE:
                self._scroll_to_page_single()
            else:
                self._scroll_to_page_double()
        elif self._current_row < len(self._chapters) - 1:
            new_row = self._current_row + 1
            self._set_toc_selection_for_chapter(new_row)
            self._activate_chapter_index(new_row)

    # ── Live single-chapter translation ──────────────────────────────────

    # _LIVE_STATUS_CHARS moved verbatim to live_stream.LiveStreamMixin (inherited).

    def _on_translate_current_chapter(self):
        """Toolbar Translate: stream-translate the chapter being read."""
        # While a live run is active the button toggles between the live
        # view and the normal reader page.
        if self._live_translate_active:
            if self._reader_stack.currentWidget() is self._live_panel:
                self._render_current()
            elif self._live_panel is not None:
                self._reader_stack.setCurrentWidget(self._live_panel)
            return

        if not self._chapters or not self._chapter_filenames:
            QMessageBox.information(self, "Translate chapter",
                                    "The EPUB is still loading — try again in a moment.")
            return
        row = max(0, min(self._current_row, len(self._chapter_filenames) - 1))
        chapter_file = os.path.basename(str(self._chapter_filenames[row] or ""))
        if not chapter_file:
            QMessageBox.warning(self, "Translate chapter",
                                "Could not resolve this chapter's source filename.")
            return

        # Source EPUB: in overlay mode ``_epub_path`` IS the raw source; in
        # dual-path (compiled) mode prefer the raw counterpart.
        epub_path = self._epub_path
        if not self._translated_overlay and self._raw_epub_alt_path:
            epub_path = self._raw_epub_alt_path
        if not (epub_path and os.path.isfile(epub_path)
                and epub_path.lower().endswith(".epub")):
            QMessageBox.warning(self, "Translate chapter",
                                "Could not resolve the source EPUB for this chapter.")
            return

        gui = _find_translator_gui(self)
        if gui is None:
            QMessageBox.warning(
                self, "Translate chapter",
                "The main translator window is not available — open the "
                "reader from within Glossarion to use live translation.")
            return
        thread = getattr(gui, "translation_thread", None)
        if thread is not None and thread.is_alive():
            QMessageBox.warning(
                self, "Translate chapter",
                "A translation is already running.\n"
                "Please wait for it to finish (or stop it) first.")
            return

        # Already-translated chapter → confirm + reset its progress entry
        # so the pipeline doesn't skip it as completed. QA-failed/failed
        # outputs are already known to need another attempt, so do not show the
        # misleading "already translated" confirmation for those chapters.
        overlay_entry = (self._translated_overlay or {}).get(chapter_file.lower())
        if overlay_entry and overlay_entry.get("path"):
            entry_status = str(
                overlay_entry.get("status") or "").strip().lower()
            if entry_status in ("", "completed"):
                title = (self._chapters[row][0]
                         if row < len(self._chapters) else chapter_file)
                if QMessageBox.question(
                        self, "Retranslate chapter",
                        f"“{title}” is already translated.\n\n"
                        "Delete its current translation and retranslate it live?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No) != QMessageBox.Yes:
                    return
            out_dir = os.path.dirname(str(overlay_entry["path"]))
            if out_dir and os.path.isdir(out_dir):
                _mark_chapter_pending_for_retranslation(out_dir, chapter_file)

        # Wire the live view up BEFORE starting so no early lines are lost.
        self._ensure_live_panel()
        self._reset_live_state()
        self._live_gui_ref = gui
        listener = self._on_live_log_line
        self._live_listener = listener
        try:
            gui.add_log_listener(listener)
        except Exception:
            self._live_listener = None

        started = False
        try:
            started = bool(gui.start_single_chapter_translation(
                epub_path, chapter_file, force_stream_all=True))
        except Exception as exc:
            QMessageBox.warning(self, "Translate chapter",
                                f"Could not start translation:\n{exc}")
        if not started:
            self._teardown_live_listener()
            return

        self._live_translate_active = True
        self._live_target_row = row
        self._live_epub_path = epub_path
        self._live_chapter_file = chapter_file
        self._translate_btn.setText("\U0001f6f0️  Live view")
        self._live_status_label.setText(
            f"\U0001f6f0️ Translating “{os.path.basename(chapter_file)}” — waiting for stream…")
        self._reader_stack.setCurrentWidget(self._live_panel)
        self._nav_bar.hide()

        if self._live_drain_timer is None:
            self._live_drain_timer = QTimer(self)
            self._live_drain_timer.setInterval(90)
            self._live_drain_timer.timeout.connect(lambda: self._drain_live_queue())
        self._live_drain_timer.start()
        if self._live_poll_timer is None:
            self._live_poll_timer = QTimer(self)
            self._live_poll_timer.setInterval(700)
            self._live_poll_timer.timeout.connect(lambda: self._poll_live_done())
        # Give the worker a moment to spin up before polling for liveness.
        QTimer.singleShot(2500, lambda: (
            self._live_poll_timer.start()
            if self._live_translate_active and self._live_poll_timer else None))

    def _update_translate_btn_visibility(self):
        """Hide the Translate button for chapters that are already translated.

        Rules:
          * While a live run is active the button stays visible (it toggles
            the live view).
          * Overlay mode (in-progress books): hidden when the current chapter
            has a completed translated response file on disk. Failed and
            QA-failed responses remain translatable even though their output
            file still exists.
          * Dual-path mode (compiled output + raw source): hidden while the
            compiled/translated EPUB is the active view — every chapter in
            it is already translated. Flipping to Raw shows it again.
          * Plain raw EPUBs: always visible.
        """
        btn = getattr(self, "_translate_btn", None)
        if btn is None:
            return
        if self._workspace_mode:
            # PDF workspace entries are translated through Progress Manager;
            # the EPUB-spine single-chapter translator cannot address them.
            btn.setVisible(False)
            return
        if self._live_translate_active:
            btn.setVisible(True)
            return
        visible = True
        try:
            overlay = self._translated_overlay or {}
            if overlay:
                row = self._current_row
                if 0 <= row < len(self._chapter_filenames):
                    base = os.path.basename(
                        str(self._chapter_filenames[row] or "")).lower()
                    entry = overlay.get(base)
                    if entry and entry.get("path") and os.path.isfile(
                            str(entry["path"])):
                        status = str(entry.get("status") or "").strip().lower()
                        # Status-less overlay entries predate status propagation
                        # and retain the original "file means completed"
                        # behavior. Explicit non-completed statuses must keep
                        # the action available for another attempt.
                        visible = status not in ("", "completed")
            elif self._raw_epub_alt_path:
                # Compiled↔raw dual mode: the primary path is the
                # translated output; only the raw flavor is translatable.
                try:
                    active = os.path.normcase(os.path.abspath(
                        str(self._epub_path or "")))
                    translated = os.path.normcase(os.path.abspath(
                        str(self._translated_epub_path or "")))
                    if active == translated:
                        visible = False
                except Exception:
                    pass
        except Exception:
            logger.debug("Translate-button visibility check failed: %s",
                         traceback.format_exc())
        btn.setVisible(visible)

    def _ensure_live_panel(self):
        """Create the streaming page (reader-stack index 2) on first use."""
        if self._live_panel is not None:
            return
        from PySide6.QtWidgets import QPlainTextEdit

        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(8)
        self._live_status_label = QLabel("")
        self._live_status_label.setStyleSheet(
            "color: #ffd166; font-size: 9pt; font-weight: bold;")
        h.addWidget(self._live_status_label, 1)

        self._live_think_toggle = QPushButton("\U0001f9e0  Thinking")
        self._live_think_toggle.setCheckable(True)
        self._live_think_toggle.setCursor(Qt.PointingHandCursor)
        self._live_think_toggle.setToolTip(
            "Show the model's thinking / pipeline log for this run.\n"
            "Kept out of the page so the stream stays readable.")
        self._live_think_toggle.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; font-weight: bold; padding: 3px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
            QPushButton:checked { background: #6c63ff; border-color: #7c73ff; color: #fff; }
        """)
        self._live_think_toggle.toggled.connect(self._on_live_think_toggled)
        h.addWidget(self._live_think_toggle)

        stop_btn = QPushButton("⏹  Stop")
        stop_btn.setCursor(Qt.PointingHandCursor)
        stop_btn.setToolTip("Stop the translation run.")
        stop_btn.setStyleSheet("""
            QPushButton { background: #3e2a2a; border: 1px solid #5e3a3a; border-radius: 4px;
                color: #d0a0a0; font-size: 8.5pt; font-weight: bold; padding: 3px 10px; }
            QPushButton:hover { background: #5e3a3a; color: #f0c0c0; }
        """)
        stop_btn.clicked.connect(self._stop_live_translation)
        h.addWidget(stop_btn)

        back_btn = QPushButton("✕  Hide")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setToolTip(
            "Return to the reader without stopping the translation.\n"
            "The Translate button re-opens this live view.")
        back_btn.setStyleSheet("""
            QPushButton { background: #2a2a3e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #b0b0c0; font-size: 8.5pt; padding: 3px 10px; }
            QPushButton:hover { background: #3a3a5e; color: #e0e0e0; }
        """)
        back_btn.clicked.connect(lambda: self._render_current())
        h.addWidget(back_btn)
        header.setStyleSheet("background: #16161f; border-bottom: 1px solid #2a2a3e;")
        v.addWidget(header)

        split = QSplitter(Qt.Vertical)
        split.setHandleWidth(2)

        # Main streamed-content view. A QTextBrowser (not a webengine view)
        # so the accumulated HTML can be cheaply re-set every drain tick —
        # Qt's rich-text engine parses tags/CSS in real time and tolerates
        # the half-open tags that mid-stream HTML inevitably has.
        self._live_content_view = QTextBrowser()
        self._live_content_view.setOpenExternalLinks(False)
        self._live_content_view.setOpenLinks(False)
        self._live_content_view.setFrameShape(QFrame.NoFrame)
        # Follow-the-stream tracking: user scrolls flip the flag, guarded
        # programmatic scrolls (re-renders) don't. See _on_live_scroll_changed.
        self._live_content_view.verticalScrollBar().valueChanged.connect(
            lambda v: self._on_live_scroll_changed(v))
        split.addWidget(self._live_content_view)

        # Discreet thinking / pipeline-log pane (hidden until toggled).
        self._live_think_view = QPlainTextEdit()
        self._live_think_view.setReadOnly(True)
        self._live_think_view.setFrameShape(QFrame.NoFrame)
        self._live_think_view.setStyleSheet(
            "QPlainTextEdit { background: #131318; color: #8a8fa8;"
            " font-family: 'Consolas','Menlo',monospace; font-size: 8.5pt;"
            " padding: 8px; }")
        self._live_think_view.hide()
        split.addWidget(self._live_think_view)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        self._live_splitter = split
        v.addWidget(split, 1)

        self._live_panel = panel
        self._reader_stack.addWidget(panel)

    def _reset_live_state(self):
        self._live_log_queue.clear()
        self._live_content_buf = ""
        self._live_think_pending = ""
        self._live_log_pending = ""
        self._live_dirty = False
        self._live_in_thinking = False
        self._live_streaming_text = False
        self._live_follow_stream = True
        self._live_scroll_guard = False
        if self._live_content_view is not None:
            self._live_content_view.clear()
        if self._live_think_view is not None:
            self._live_think_view.clear()
        if self._live_think_toggle is not None:
            self._live_think_toggle.setChecked(False)
            self._live_think_toggle.setText("\U0001f9e0  Thinking")

    def _on_live_think_toggled(self, checked: bool):
        if self._live_think_view is not None:
            self._live_think_view.setVisible(bool(checked))

    def _on_live_log_line(self, message):
        """Main-GUI log listener — called from worker threads. Only queue."""
        try:
            self._live_log_queue.append(str(message))
        except Exception:
            pass

    # _classify_live_line moved verbatim to live_stream.LiveStreamMixin (inherited).

    def _drain_live_queue(self):
        """GUI-thread timer: drain queued log lines into the live views."""
        drained, content_added = self._drain_live_lines()
        if not drained:
            return

        # Thinking + pipeline log share the discreet pane; thinking text is
        # appended bare, status lines keep their prefixes.
        if self._live_think_pending or self._live_log_pending:
            view = self._live_think_view
            if view is not None:
                cursor = view.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                if self._live_think_pending:
                    cursor.insertText(self._live_think_pending)
                if self._live_log_pending:
                    cursor.insertText(self._live_log_pending)
                view.setTextCursor(cursor)
                sb = view.verticalScrollBar()
                sb.setValue(sb.maximum())
                # Surface activity on the collapsed toggle.
                if self._live_think_toggle is not None and not self._live_think_toggle.isChecked():
                    n = self._live_think_view.blockCount()
                    self._live_think_toggle.setText(f"\U0001f9e0  Thinking ({n})")
            self._live_think_pending = ""
            self._live_log_pending = ""

        if content_added:
            self._render_live_content()

    # _drain_live_lines was extracted from _drain_live_queue into live_stream.LiveStreamMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _on_live_scroll_changed(self, value: int):
        """Track whether the user is following the stream.

        Only USER-initiated scrolls may flip the follow flag —
        ``setHtml()`` resets the scrollbar to 0 on every re-render, which
        must not be mistaken for "the user scrolled to the top".
        """
        if getattr(self, "_live_scroll_guard", False):
            return
        view = self._live_content_view
        if view is None:
            return
        sb = view.verticalScrollBar()
        self._live_follow_stream = value >= sb.maximum() - 8

    def _render_live_content(self):
        """Re-render the accumulated stream with live HTML/CSS parsing and
        follow the output (auto page switching) unless the user scrolled up."""
        view = self._live_content_view
        if view is None:
            return
        follow = getattr(self, "_live_follow_stream", True)
        sb = view.verticalScrollBar()
        saved_pos = sb.value()
        self._live_scroll_guard = True
        try:
            view.setHtml(self._wrap_live_html(self._live_content_buf))
            if follow:
                # Force the document layout to the end so maximum() is
                # computed against the NEW content height, not the stale one.
                cursor = view.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                view.setTextCursor(cursor)
                view.ensureCursorVisible()
                sb.setValue(sb.maximum())
            else:
                sb.setValue(min(saved_pos, sb.maximum()))
        finally:
            self._live_scroll_guard = False
        if follow:
            # Late layout passes (images, long reflows) can still grow the
            # document after this tick — snap again once the event loop has
            # settled.
            def _snap_bottom():
                v = self._live_content_view
                if v is None or not getattr(self, "_live_follow_stream", True):
                    return
                self._live_scroll_guard = True
                try:
                    bar = v.verticalScrollBar()
                    bar.setValue(bar.maximum())
                finally:
                    self._live_scroll_guard = False
            QTimer.singleShot(0, _snap_bottom)

    # _wrap_live_html moved verbatim to live_stream.LiveStreamMixin (inherited).

    def _stop_live_translation(self):
        gui = self._live_gui_ref
        if gui is not None and hasattr(gui, "stop_translation"):
            try:
                gui.stop_translation()
                if self._live_status_label is not None:
                    self._live_status_label.setText("⏹ Stopping…")
            except Exception:
                logger.debug("Live stop failed: %s", traceback.format_exc())

    def _poll_live_done(self):
        """Watch the main GUI's worker; when it ends, wrap the live run up."""
        if not self._live_translate_active:
            return
        gui = self._live_gui_ref
        thread = getattr(gui, "translation_thread", None) if gui else None
        if thread is not None and thread.is_alive():
            return
        self._finish_live_translation()

    # _resolve_live_output_folder moved verbatim to live_stream.LiveStreamMixin (inherited).

    def _finish_live_translation(self):
        if not self._live_translate_active:
            return
        self._live_translate_active = False
        if self._live_poll_timer is not None:
            self._live_poll_timer.stop()
        # Final drain so trailing chunks aren't lost, then stop ticking.
        try:
            self._drain_live_queue()
        except Exception:
            pass
        if self._live_drain_timer is not None:
            self._live_drain_timer.stop()
        gui = self._live_gui_ref
        was_stopped = bool(getattr(gui, "stop_requested", False)) if gui else False
        self._teardown_live_listener()
        self._translate_btn.setText("\U0001f310  Translate")

        # Outcome check: a stopped/failed run leaves a partially-written
        # response file and an in_progress progress entry behind — clear
        # both so the reader doesn't keep rendering the half-translated
        # chapter (and the chapter drops back to "pending").
        chapter_file, out_dir, completed, cleaned = self._live_outcome()
        if cleaned:
            if gui is not None:
                try:
                    gui.append_log(
                        f"\U0001f9f9 Live view: cleared incomplete translation "
                        f"for {chapter_file}")
                except Exception:
                    pass

        if self._live_status_label is not None:
            self._live_status_label.setText(
                live_outcome_text(completed, was_stopped))
        # Re-merge the overlay: picks the fresh response file up on success,
        # or drops the deleted partial chapter after a stop/failure. Then
        # swap back to the normal reader page at the same chapter.
        try:
            self._on_overlay_refresh_tick()
        except Exception:
            pass

        def _back_to_reader():
            if self._closing or self._live_translate_active:
                return
            target = self._live_target_row
            if isinstance(target, int) and 0 <= target < len(self._chapters):
                self._current_row = target
            if self._reader_stack.currentWidget() is self._live_panel:
                self._render_current()
            self._update_translate_btn_visibility()

        QTimer.singleShot(2200 if completed else 1200, _back_to_reader)

    # _live_outcome was extracted from _finish_live_translation into live_stream.LiveStreamMixin (see DISCREPANCIES U5 "Phase-1 splits").

    def _teardown_live_listener(self):
        gui = self._live_gui_ref
        listener = self._live_listener
        if gui is not None and listener is not None:
            try:
                gui.remove_log_listener(listener)
            except Exception:
                pass
        self._live_listener = None
        self._live_gui_ref = None

    def showEvent(self, event):
        """Paint the loading shell before starting Chromium-backed panes."""
        super().showEvent(event)
        # Keep QObject ownership (lifetime, config persistence, and access to
        # the translator) without making the reader an OS-owned window.  On
        # Windows an owned top-level window is automatically hidden whenever
        # its owner is minimized, which previously made minimizing Library or
        # Book Details minimize the reader as well.
        self._clear_transient_window_parent()
        # Some platform plugins finish creating the native window immediately
        # after showEvent.  Repeat once after that hand-off so the native owner
        # cannot be restored as part of first-show window creation.
        QTimer.singleShot(0, self._clear_transient_window_parent)
        if (self._closing or self._reader_views_ready
                or self._reader_init_queued):
            return
        self._reader_init_queued = True
        # A short delay lets Windows map and paint the native dialog. A zero-
        # timeout can run before the first paint and recreate the original
        # symptom: the context menu disappears but no reader window is visible
        # while Chromium performs its cold start.
        QTimer.singleShot(40, self._initialize_reader_views)

    def _clear_transient_window_parent(self):
        """Make the reader minimize independently of its QObject parent."""
        if getattr(self, "_closing", False):
            return
        try:
            handle = self.windowHandle()
            if handle is not None and handle.transientParent() is not None:
                handle.setTransientParent(None)
        except Exception:
            logger.debug(
                "Could not detach EPUB reader from its transient window owner: %s",
                traceback.format_exc(),
            )

    def closeEvent(self, event):
        """Persist reader settings back into config and flush to disk.

        The priming pass in :meth:`_on_epub_loaded_from_cache` temporarily
        flips ``_layout_mode`` to :data:`LAYOUT_SCROLL` while the first
        render happens (swapping back via ``loadFinished``). If the user
        closes the dialog before that swap-back lands, persisting the
        transient ``_layout_mode`` would poison the config with
        ``"scroll"`` — and since Scroll never triggers priming, every
        subsequent open would open in Scroll and stay there. Fall back
        to ``_prime_saved_mode`` whenever the priming flag is still set.

        After writing into the in-memory ``_config`` dict we also ask
        the translator parent to flush to ``config.json`` via
        :func:`_persist_config_via_parent` so the values actually
        survive to the next session — the main window's own save only
        runs on an orderly quit, and users' library / reader tweaks
        shouldn't hinge on that.
        """
        self._config['epub_reader_font_size'] = self._font_size
        self._config['epub_reader_line_spacing'] = self._line_spacing
        self._config['epub_reader_theme'] = self._theme_index
        if getattr(self, '_priming_initial_render', False):
            self._config['epub_reader_layout'] = getattr(
                self, '_prime_saved_mode', self._layout_mode)
        else:
            self._config['epub_reader_layout'] = self._layout_mode
        self._config['epub_reader_font_family'] = self._font_family
        self._config['epub_reader_show_raw'] = self._show_raw
        self._config['epub_reader_native_toc'] = self._native_toc_enabled
        self._closing = True
        self._cleanup_initial_reader_transition()
        # Stop timers and browser loads before worker shutdown so no
        # queued callbacks can restart work while the dialog is closing.
        # Live-translation cleanup: detach from the main GUI's log stream
        # (the translation itself keeps running in the main window).
        try:
            self._live_translate_active = False
            self._teardown_live_listener()
        except Exception:
            pass
        for timer_name in (
            "_spin_timer",
            "_overlay_refresh_timer",
            "_search_debounce_timer",
            "_search_render_timer",
            "_live_drain_timer",
            "_live_poll_timer",
        ):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
                if timer_name == "_overlay_refresh_timer":
                    self._overlay_refresh_timer = None
        for browser_name in ("_reader", "_reader_left", "_reader_right"):
            browser = getattr(self, browser_name, None)
            if browser is not None:
                try:
                    browser.stop()
                except Exception:
                    pass
        self._search_dialog_generation = (
            int(getattr(self, "_search_dialog_generation", 0) or 0) + 1)
        self._search_realign_generation = (
            int(getattr(self, "_search_realign_generation", 0) or 0) + 1)
        self._search_selection_generation = (
            int(getattr(self, "_search_selection_generation", 0) or 0) + 1)
        self._cancel_search_dialog_workers()
        for worker in list(getattr(self, "_search_workers", []) or []):
            _stop_qthread_safely(
                worker, timeout_ms=1200,
                signal_names=("results_ready", "results_batch_ready",
                              "finished"),
            )
        self._search_workers = []
        _stop_qthread_safely(
            getattr(self, "_overlay_thread", None),
            timeout_ms=1500,
            signal_names=("done",),
        )
        self._overlay_thread = None
        _stop_qthread_safely(
            getattr(self, "_loader_thread", None),
            timeout_ms=2000,
            signal_names=("done", "error"),
        )
        self._loader_thread = None
        _stop_qthread_safely(
            getattr(self, "_cache_loader_thread", None),
            timeout_ms=2000,
            signal_names=("hit", "miss"),
        )
        self._cache_loader_thread = None
        _stop_qthread_safely(
            getattr(self, "_workspace_loader_thread", None),
            timeout_ms=1200,
            signal_names=("done", "error"),
        )
        self._workspace_loader_thread = None
        _stop_qthread_safely(
            getattr(self, "_workspace_raw_thread", None),
            timeout_ms=1200,
            signal_names=("done", "error"),
        )
        self._workspace_raw_thread = None
        _stop_qthread_safely(
            getattr(self, "_image_preload_thread", None),
            timeout_ms=800,
            signal_names=("done", "finished"),
        )
        self._image_preload_thread = None
        _stop_qthread_safely(
            getattr(self, "_epub_load_image_preload_thread", None),
            timeout_ms=800,
            signal_names=("done", "finished"),
        )
        self._epub_load_image_preload_thread = None
        self._pending_epub_load = None
        self._pending_image_chapter_activation = None
        self._close_epub_image_zip()
        _persist_config_via_parent(self)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        """Suppress Enter/Return from activating auto-default buttons.

        In a QDialog, Enter presses the auto-default QPushButton —
        which randomly toggles whichever button Qt chose as the default
        (TOC, raw/translated, etc.).  We eat the key unless a QLineEdit
        (e.g. the search bar) currently has focus, in which case we let
        it handle returnPressed normally.
        """
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if (self._search_ui_is_open()
                    and self._focus_is_in_search_ui()
                    and self._search_enter_next()):
                event.accept()
                return
            from PySide6.QtWidgets import QLineEdit
            fw = self.focusWidget()
            if isinstance(fw, QLineEdit):
                # Let the search bar's returnPressed signal fire
                super().keyPressEvent(event)
            else:
                # Swallow — do nothing
                event.accept()
            return
        super().keyPressEvent(event)

    # ── Chapter rendering ───────────────────────────────────────────────

    def _set_raw_toggle_busy(self, busy: bool) -> None:
        """Serialize raw/translated swaps.

        A swap may span an EPUB cache load, a hidden scroll-mode prime, and a
        second paginated WebEngine load.  Letting the button start another
        swap during that chain used to leave ``_reader_stack`` hidden when the
        callbacks completed out of order.
        """
        self._raw_toggle_in_flight = bool(busy)
        button = getattr(self, "_raw_btn", None)
        if button is not None:
            button.setEnabled(not busy)

    def _finish_raw_toggle(self) -> None:
        """Finish a raw swap and recover every temporarily-hidden UI part."""
        if not getattr(self, "_raw_toggle_in_flight", False):
            return

        # A failed/cancelled load can strand the hidden pagination-prime
        # sequence before ``_reveal_reader_stack_after_prime`` runs.  Restore
        # the saved layout and splitter state here as a final safety net.
        if getattr(self, "_priming_initial_render", False):
            self._priming_initial_render = False
            if self._layout_mode == LAYOUT_SCROLL:
                self._layout_mode = getattr(
                    self, "_prime_saved_mode", LAYOUT_SINGLE)
            sizes = getattr(self, "_prime_toc_sizes", None)
            self._prime_toc_sizes = None
            self._end_toc_width_lock(sizes)

        stack = getattr(self, "_reader_stack", None)
        if stack is not None:
            stack.show()
        if _HAS_WEBENGINE and getattr(self, "_reader", None) is not None:
            try:
                self._reader.page().runJavaScript(
                    "var c=document.getElementById('columns');"
                    "if(c)c.style.opacity='1';"
                )
            except RuntimeError:
                pass
        self._set_raw_toggle_busy(False)

    def _restore_raw_toggle_value(self, value: bool) -> None:
        """Roll the pill and persisted state back after a rejected swap."""
        self._show_raw = bool(value)
        button = getattr(self, "_raw_btn", None)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(self._show_raw)
            button.blockSignals(False)
        try:
            self._config['epub_reader_show_raw'] = self._show_raw
        except Exception:
            pass
        self._finish_raw_toggle()

    def _on_show_raw_toggled(self, checked: bool):
        """Swap between raw and translated content.

        Two backing modes:
          * **Overlay mode** — both raw and overlaid chapter lists were
            precomputed at load time, so this just flips ``_chapters``
            between them, rebuilds the TOC (titles differ between
            flavors), invalidates the paginated-page cache (page counts
            differ between translations) and re-renders. Cheap, no I/O.
          * **Dual-path mode** — the caller attached an alt EPUB (raw
            counterpart of a compiled Completed-tab book). Flipping the
            toggle swaps ``_epub_path`` between the compiled and raw
            files and restarts the loader via
            :meth:`_reload_epub_from_active_path`.

        Reading position survives the swap in both modes via
        :attr:`_pending_page_hint` / :attr:`_reload_position_hint`:
        the finalizer after pagination consumes the hint and positions
        the reader either at the same proportional page or — if the
        user was on the last page before the swap — at the last page
        of the new flavor (page counts differ between translations).
        """
        new_value = bool(checked)
        if getattr(self, "_raw_toggle_in_flight", False):
            # Normally impossible because the pill is disabled during the
            # transition, but also guard programmatic toggles and queued Qt
            # click events that were posted before it became disabled.
            button = getattr(self, "_raw_btn", None)
            if button is not None:
                button.blockSignals(True)
                button.setChecked(bool(self._show_raw))
                button.blockSignals(False)
            return
        if new_value == self._show_raw:
            return
        previous_value = bool(self._show_raw)
        self._set_raw_toggle_busy(True)
        # Save the currently-active standalone EPUB before changing the
        # flavor flag. Besides chapters and image descriptors, this retains
        # the processed HTML and materialized-image caches accumulated while
        # reading it. Overlay mode is intentionally ignored by the helper.
        self._remember_dual_path_reader_state()
        self._show_raw = new_value
        # Invalidate embedded CSS cache so it re-reads from the correct EPUB
        if hasattr(self, '_embedded_css_cache'):
            del self._embedded_css_cache
        try:
            self._config['epub_reader_show_raw'] = self._show_raw
        except Exception:
            pass
        # Snapshot the reading position so the finalizer can restore it
        # against the (potentially differently-paginated) new content.
        position_hint = self._capture_position_hint()
        # Dual-path mode takes precedence: it's the only meaningful
        # interpretation when no overlay was supplied. We also fall
        # through to it when overlay mode has nothing loaded yet (e.g.
        # toggle clicked mid-load before chapters exist).
        has_overlay = bool(self._chapters_overlaid) and any(
            (self._chapters_overlaid[i] != self._chapters_raw[i])
            for i in range(min(len(self._chapters_overlaid),
                               len(self._chapters_raw)))
        )
        if self._raw_epub_alt_path and not has_overlay:
            target = (self._raw_epub_alt_path if self._show_raw
                      else self._translated_epub_path)
            if not target or not os.path.isfile(target):
                self._restore_raw_toggle_value(previous_value)
                return
            self._epub_path = target
            # Reload pipeline consumes this hint to pick the same row
            # and hand the page portion to the finalizer.
            self._reload_position_hint = position_hint
            if self._restore_dual_path_reader_state():
                return
            self._reload_epub_from_active_path()
            return
        # Overlay mode — in-memory chapter swap.
        if not self._chapters_raw and not self._chapters_overlaid:
            self._restore_raw_toggle_value(previous_value)
            return
        self._chapters = (self._chapters_raw if self._show_raw
                          else self._chapters_overlaid)
        self._refresh_search_for_active_chapters()
        # Rebuild TOC entries so titles reflect the active flavor. Hold
        # the current row so we don't lose the reading position.
        current_row = max(0, min(self._current_row, len(self._chapters) - 1))
        _saved_sizes = self._begin_toc_width_lock()
        self._refresh_native_toc_entries()
        self._rebuild_toc_sidebar(current_chapter=current_row)
        self._end_toc_width_lock(_saved_sizes)
        self._current_row = current_row
        # Force a fresh paginated render: page-count caches differ
        # between translations and the currently-loaded chapter needs to
        # be re-set from the new source. The finalizer will consult
        # ``_pending_page_hint`` to pick a sensible page number once
        # the new pagination lands.
        self._chapter_page_cache = {}
        self._loaded_chapter = -1
        self._pending_page_hint = position_hint
        if self._chapters:
            # Hide content before re-render to prevent flash (same pattern
            # as font/theme changes). The _js_reveal in the paginated
            # finalizer will set opacity back to 1 after positioning.
            _hide = "var c = document.getElementById('columns'); if (c) c.style.opacity = '0';"
            if _HAS_WEBENGINE:
                self._reader.page().runJavaScript(_hide)
            self._render_current()
            if not _HAS_WEBENGINE:
                self._finish_raw_toggle()
        else:
            self._finish_raw_toggle()

    def _dual_path_reader_state_key(self, epub_path: str | None = None) -> str:
        """Return a file/settings fingerprint for an in-memory reader state."""
        raw_path = str(epub_path or self._epub_path or "")
        if not raw_path:
            return ""
        path = os.path.abspath(raw_path)
        cache_key = _epub_cache_key(
            path,
            show_special_files=self._show_special_files,
            config=self._config,
        )
        return f"{os.path.normcase(path)}|{cache_key}"

    def _remember_dual_path_reader_state(self) -> None:
        """Retain the active standalone EPUB for a later instant swap-back."""
        if (not getattr(self, "_raw_epub_alt_path", "")
                or getattr(self, "_translated_overlay", None)
                or not getattr(self, "_chapters_raw", None)):
            return
        key = self._dual_path_reader_state_key()
        if not key:
            return
        state = {
            "raw_chapters": list(self._chapters_raw),
            "overlaid_chapters": list(self._chapters_overlaid),
            "images": dict(self._images),
            "filenames": list(self._chapter_filenames),
            "processed_html_cache": dict(self._processed_html_cache),
            "image_sizeable_cache": dict(self._image_sizeable_cache),
            "preloaded_chapter_keys": set(self._preloaded_chapter_keys),
            "preloaded_image_resources": dict(
                self._preloaded_image_resources),
            "image_cache_generation": int(self._image_cache_generation),
            "image_resource_signature": self._image_resource_signature,
            "img_temp_dir": getattr(self, "_img_temp_dir", ""),
        }
        if hasattr(self, "_embedded_css_cache"):
            state["embedded_css_cache"] = self._embedded_css_cache
        self._dual_path_reader_states[key] = state

    def _restore_dual_path_reader_state(self) -> bool:
        """Restore a previously visited standalone EPUB without disk loading."""
        key = self._dual_path_reader_state_key()
        state = self._dual_path_reader_states.get(key)
        if not state:
            return False

        _stop_qthread_safely(
            getattr(self, "_image_preload_thread", None),
            timeout_ms=500,
            signal_names=("done", "finished"),
        )
        self._image_preload_thread = None
        self._image_preload_active_key = ""
        self._image_preload_pending = False
        _stop_qthread_safely(
            getattr(self, "_epub_load_image_preload_thread", None),
            timeout_ms=500,
            signal_names=("done", "finished"),
        )
        self._epub_load_image_preload_thread = None
        self._pending_epub_load = None
        self._pending_image_chapter_activation = None
        self._close_epub_image_zip()

        self._processed_html_cache = dict(
            state.get("processed_html_cache") or {})
        self._image_sizeable_cache = dict(
            state.get("image_sizeable_cache") or {})
        self._preloaded_chapter_keys = set(
            state.get("preloaded_chapter_keys") or set())
        self._preloaded_image_resources = dict(
            state.get("preloaded_image_resources") or {})
        self._image_cache_generation = int(
            state.get("image_cache_generation") or 0)
        self._image_resource_signature = tuple(
            state.get("image_resource_signature") or ())
        temp_dir = str(state.get("img_temp_dir") or "")
        if temp_dir:
            self._img_temp_dir = temp_dir
        elif hasattr(self, "_img_temp_dir"):
            del self._img_temp_dir
        if "embedded_css_cache" in state:
            self._embedded_css_cache = state["embedded_css_cache"]
        elif hasattr(self, "_embedded_css_cache"):
            del self._embedded_css_cache

        self._native_toc_source_entries = _load_reader_native_toc(
            self._toc_output_dir, self._epub_path)
        raw_chapters = list(state.get("raw_chapters") or [])
        overlaid = list(state.get("overlaid_chapters") or raw_chapters)
        self._finalize_post_load(
            raw_chapters,
            overlaid,
            dict(state.get("images") or {}),
            False,
            list(state.get("filenames") or []),
        )
        return True

    def _start_epub_load_image_preload(
            self, chapters, images, filenames) -> bool:
        """Prepare the first visible chapter's images before EPUB rendering.

        Cache/parsing work already runs outside the GUI thread. Image
        materialization used to re-enter the GUI thread during the first
        chapter render, though, which was especially visible for standalone
        raw EPUBs containing large scans. The loading shell stays responsive
        on an initial open; raw/translated swaps keep the old page visible.
        """
        is_initial_open = bool(
            getattr(self, "_reader_startup_pending", False)
            and not getattr(self, "_chapters", None)
        )
        is_flavor_swap = bool(getattr(self, "_raw_toggle_in_flight", False))
        if not chapters or not (is_initial_open or is_flavor_swap):
            return False
        hint = getattr(self, "_reload_position_hint", None) or {}
        if hint:
            row = int(hint.get("row", 0) or 0)
        else:
            row = 0
            initial_filename = getattr(
                self, "_initial_chapter_filename", None)
            if initial_filename:
                normalized = [
                    os.path.basename(str(name or "")).lower()
                    for name in (filenames or [])
                ]
                try:
                    row = normalized.index(initial_filename)
                except ValueError:
                    row = 0
            elif getattr(self, "_initial_chapter", None) is not None:
                row = int(self._initial_chapter)
        row = max(0, min(row, len(chapters) - 1))
        if getattr(self, "_layout_mode", LAYOUT_SINGLE) == LAYOUT_ALL:
            html_content = "\n".join(
                str(content or "") for _title, content in chapters)
        else:
            html_content = str(chapters[row][1] or "")
        if not re.search(r"(?:<|&lt;)\s*(?:img|image)\b", html_content, re.I):
            return False

        previous = getattr(self, "_epub_load_image_preload_thread", None)
        if previous is not None:
            _stop_qthread_safely(
                previous, timeout_ms=500,
                signal_names=("done", "finished"),
            )
        state_key = self._dual_path_reader_state_key()
        preload_key = f"epub-load|{state_key}|{row}"
        self._pending_epub_load = (
            state_key,
            list(chapters),
            dict(images or {}),
            list(filenames or []),
        )
        thread = _ReaderImagePreloadThread(
            preload_key,
            html_content,
            dict(images or {}),
            list(getattr(self, "_extra_image_dirs", []) or []),
            self._epub_path,
            self._ensure_reader_image_temp_dir(),
            parent=self,
        )
        self._epub_load_image_preload_thread = thread
        thread.done.connect(self._on_epub_load_images_preloaded)
        thread.finished.connect(
            lambda pending=thread:
                self._on_epub_load_image_preload_finished(pending))
        thread.start()
        return True

    @Slot(str, object)
    def _on_epub_load_images_preloaded(
            self, preload_key: str, resources: dict) -> None:
        """Install first-chapter image results and complete the EPUB load."""
        if getattr(self, "_closing", False):
            return
        pending = getattr(self, "_pending_epub_load", None)
        if not pending:
            return
        state_key, chapters, images, filenames = pending
        if state_key != self._dual_path_reader_state_key():
            return
        self._pending_epub_load = None
        self._set_reader_images(images)
        self._preloaded_image_resources.update(dict(resources or {}))
        for identity, info in (resources or {}).items():
            if isinstance(info, dict) and "sizeable" in info:
                self._image_sizeable_cache[str(identity)] = bool(
                    info["sizeable"])
        self._finalize_post_load(
            chapters, chapters, images, False, filenames)

    def _on_epub_load_image_preload_finished(self, thread) -> None:
        if thread is getattr(self, "_epub_load_image_preload_thread", None):
            self._epub_load_image_preload_thread = None
        try:
            thread.deleteLater()
        except Exception:
            pass

    def _capture_position_hint(self) -> dict:
        """Snapshot the current reading position for later restoration.

        Returned dict carries:
          * ``row``          — the current chapter index.
          * ``was_last_page`` — True when the user was on the final page
            of the chapter (so the finalizer can jump to the last page
            of the new flavor regardless of page-count drift).
          * ``proportion``   — relative progress through the chapter
            [0, 1], used when ``was_last_page`` is False to pick the
            closest equivalent page in the new pagination.
          * ``scroll_x`` / ``scroll_y`` and ``layout`` — viewport position
            for scroll layouts, whose page counters do not track scrolling.
        """
        row = int(getattr(self, '_current_row', 0) or 0)
        page = int(getattr(self, '_current_page', 0) or 0)
        pages = int(self._get_chapter_pages(row)) if self._chapters else 0
        was_last = pages > 0 and page >= pages - 1
        if pages > 1:
            proportion = max(0.0, min(1.0, page / (pages - 1)))
        else:
            proportion = 0.0
        hint = {
            "row": row,
            "was_last_page": bool(was_last),
            "proportion": float(proportion),
        }
        layout = getattr(self, "_layout_mode", LAYOUT_SINGLE)
        if layout in (LAYOUT_SCROLL, LAYOUT_ALL):
            try:
                if _HAS_WEBENGINE:
                    position = self._reader.page().scrollPosition()
                    x, y = position.x(), position.y()
                else:
                    x = self._reader.horizontalScrollBar().value()
                    y = self._reader.verticalScrollBar().value()
                hint.update(layout=layout, scroll_x=float(x), scroll_y=float(y))
            except (AttributeError, RuntimeError):
                pass
        return hint

    def _apply_pending_scroll_hint(self) -> None:
        """Restore a refreshed scroll document without overriding navigation."""
        hint = getattr(self, "_pending_page_hint", None)
        if not hint or self._layout_mode not in (LAYOUT_SCROLL, LAYOUT_ALL):
            return
        self._pending_page_hint = None
        if (hint.get("row") != self._current_row
                or hint.get("layout") != self._layout_mode
                or "scroll_y" not in hint):
            return
        x = max(0.0, float(hint.get("scroll_x") or 0.0))
        y = max(0.0, float(hint.get("scroll_y") or 0.0))
        if _HAS_WEBENGINE:
            self._reader.page().runJavaScript(
                f"window.scrollTo({{left: {x}, top: {y}, behavior: 'instant'}});")
        else:
            self._reader.horizontalScrollBar().setValue(round(x))
            self._reader.verticalScrollBar().setValue(round(y))

    def _reload_epub_from_active_path(self):
        if getattr(self, "_closing", False):
            return
        """Restart the loader pipeline against the current ``_epub_path``.

        Called by the Show-raw toggle in dual-path mode: after swapping
        ``_epub_path`` to the raw (or back to the compiled) EPUB, we need
        to reset chapter / image state and re-run the spinner + loader
        exactly like the initial open sequence. The temp image directory
        is left behind on purpose — it's keyed by the active epub_path
        and will be re-created on the first ``_process_html`` call.

        If ``_reload_position_hint`` is set (by the Show-raw toggle), it
        is preserved across the reset so :meth:`_on_epub_loaded_from_cache`
        can seed the initial row / page from the user's pre-swap position
        instead of defaulting to the first chapter.
        """
        self._chapters = []
        self._chapters_raw = []
        self._chapters_overlaid = []
        self._chapter_filenames = []
        self._chapter_display_numbers = []
        _stop_qthread_safely(
            getattr(self, "_image_preload_thread", None),
            timeout_ms=500,
            signal_names=("done", "finished"),
        )
        self._image_preload_thread = None
        self._image_preload_active_key = ""
        self._image_preload_pending = False
        _stop_qthread_safely(
            getattr(self, "_epub_load_image_preload_thread", None),
            timeout_ms=500,
            signal_names=("done", "finished"),
        )
        self._epub_load_image_preload_thread = None
        self._pending_epub_load = None
        self._pending_image_chapter_activation = None
        self._close_epub_image_zip()
        self._images = {}
        self._image_resource_signature = ()
        self._invalidate_processed_reader_cache()
        self._preloaded_image_resources.clear()
        self._chapter_page_cache = {}
        self._loaded_chapter = -1
        self._current_page = 0
        # Clear embedded CSS cache so it re-reads from the new EPUB path
        if hasattr(self, '_embedded_css_cache'):
            del self._embedded_css_cache
        self._native_toc_source_entries = _load_reader_native_toc(
            self._toc_output_dir, self._epub_path)
        if hasattr(self, "_img_temp_dir"):
            # Clear so _process_html picks a fresh per-EPUB temp dir.
            try:
                del self._img_temp_dir
            except AttributeError:
                pass
        # NOTE: do NOT clear _toc_list here — it causes a visible flash
        # where the splitter momentarily resizes the empty TOC to full
        # width. The TOC is rebuilt in _finalize_post_load when the new
        # data arrives.
        # Start loading in the background WITHOUT showing the loading
        # spinner — keeps the current content visible for a seamless
        # raw↔translated transition instead of a jarring flash.
        prev_cache = getattr(self, "_cache_loader_thread", None)
        if prev_cache is not None:
            _stop_qthread_safely(
                prev_cache, timeout_ms=250,
                signal_names=("hit", "miss"),
            )
        self._cache_loader_thread = _EpubCacheLoaderThread(
            self._epub_path,
            show_special_files=self._show_special_files,
            config=self._config,
            parent=self,
        )
        self._cache_loader_thread.hit.connect(self._on_cache_hit)
        self._cache_loader_thread.miss.connect(self._on_cache_miss)
        self._cache_loader_thread.start()

    def _on_chapter_selected(self, toc_row):
        chapter_index = self._toc_chapter_index_for_row(toc_row)
        if chapter_index < 0:
            return
        self._activate_chapter_index(chapter_index)

    def _activate_chapter_index(self, chapter_index: int) -> None:
        """Open an underlying spine chapter independently of sidebar shape."""
        if chapter_index < 0 or chapter_index >= len(self._chapters):
            return
        if self._defer_chapter_activation_for_images(chapter_index):
            return
        self._current_row = chapter_index
        self._current_page = 0  # reset pagination when selecting a new chapter
        if not getattr(self, '_pending_search_text', None):
            self._search_match_index = 0
        self._render_current()

    # _ensure_reader_image_temp_dir, _close_epub_image_zip, _load_reader_image_resource, _invalidate_processed_reader_cache, _set_reader_images, _processed_reader_html_key, _chapter_image_preload_key moved verbatim to reader_doc.ReaderDocMixin (inherited).

    def _defer_chapter_activation_for_images(self, chapter_index: int) -> bool:
        """Keep the current page usable while a jumped-to chapter is warmed."""
        if (getattr(self, "_closing", False)
                or self._layout_mode == LAYOUT_ALL
                or not (0 <= chapter_index < len(self._chapters))):
            return False
        html_content = str(self._chapters[chapter_index][1] or "")
        preload_key = self._chapter_image_preload_key(
            chapter_index, html_content)
        if preload_key in self._preloaded_chapter_keys:
            return False
        if not re.search(r"(?:<|&lt;)\s*(?:img|image)\b", html_content, re.I):
            self._preloaded_chapter_keys.add(preload_key)
            return False

        self._pending_image_chapter_activation = (
            preload_key, chapter_index)
        active = getattr(self, "_image_preload_thread", None)
        if active is not None:
            try:
                if active.isRunning():
                    if self._image_preload_active_key == preload_key:
                        return True
                    _stop_qthread_safely(
                        active, timeout_ms=500,
                        signal_names=("done", "finished"),
                    )
            except RuntimeError:
                pass
        self._image_preload_thread = None
        self._image_preload_active_key = ""
        self._image_preload_pending = False

        thread = _ReaderImagePreloadThread(
            preload_key,
            html_content,
            dict(getattr(self, "_images", {}) or {}),
            list(getattr(self, "_extra_image_dirs", []) or []),
            self._epub_path,
            self._ensure_reader_image_temp_dir(),
            parent=self,
        )
        self._image_preload_thread = thread
        self._image_preload_active_key = preload_key
        thread.done.connect(self._on_next_chapter_images_preloaded)
        thread.finished.connect(
            lambda pending=thread:
                self._on_image_preload_thread_finished(pending))
        thread.start()
        return True

    def _schedule_next_chapter_image_preload(self) -> None:
        """Warm the following chapter's local image files in the background."""
        if getattr(self, "_closing", False):
            return
        if self._layout_mode == LAYOUT_ALL:
            return
        next_row = int(getattr(self, "_current_row", 0) or 0) + 1
        chapters = getattr(self, "_chapters", []) or []
        if not (0 <= next_row < len(chapters)):
            return
        html_content = str(chapters[next_row][1] or "")
        preload_key = self._chapter_image_preload_key(next_row, html_content)
        if preload_key in getattr(self, "_preloaded_chapter_keys", set()):
            return
        if not re.search(r"(?:<|&lt;)\s*(?:img|image)\b", html_content, re.I):
            self._preloaded_chapter_keys.add(preload_key)
            return
        active = getattr(self, "_image_preload_thread", None)
        if active is not None:
            try:
                if active.isRunning():
                    if getattr(self, "_image_preload_active_key", "") == preload_key:
                        return
                    self._image_preload_pending = True
                    return
            except RuntimeError:
                pass
        temp_dir = self._ensure_reader_image_temp_dir()
        thread = _ReaderImagePreloadThread(
            preload_key,
            html_content,
            dict(getattr(self, "_images", {}) or {}),
            list(getattr(self, "_extra_image_dirs", []) or []),
            self._epub_path,
            temp_dir,
            parent=self,
        )
        self._image_preload_thread = thread
        self._image_preload_active_key = preload_key
        self._image_preload_pending = False
        thread.done.connect(self._on_next_chapter_images_preloaded)
        thread.finished.connect(
            lambda pending=thread: self._on_image_preload_thread_finished(pending))
        thread.start()

    @Slot(str, object)
    def _on_next_chapter_images_preloaded(self, preload_key: str,
                                          resources: dict) -> None:
        if getattr(self, "_closing", False):
            return
        try:
            generation = int(str(preload_key).split(":", 1)[0])
        except (TypeError, ValueError):
            generation = -1
        pending_activation = getattr(
            self, "_pending_image_chapter_activation", None)
        if generation != int(getattr(self, "_image_cache_generation", 0) or 0):
            if pending_activation and pending_activation[0] == preload_key:
                self._pending_image_chapter_activation = None
                QTimer.singleShot(
                    0,
                    lambda row=pending_activation[1]:
                        self._activate_chapter_index(row),
                )
            return
        self._preloaded_chapter_keys.add(str(preload_key))
        self._preloaded_image_resources.update(dict(resources or {}))
        for identity, info in (resources or {}).items():
            if isinstance(info, dict) and "sizeable" in info:
                self._image_sizeable_cache[str(identity)] = bool(info["sizeable"])
        if pending_activation and pending_activation[0] == preload_key:
            self._pending_image_chapter_activation = None
            QTimer.singleShot(
                0,
                lambda row=pending_activation[1]:
                    self._activate_chapter_index(row),
            )

    def _on_image_preload_thread_finished(self, thread) -> None:
        if thread is getattr(self, "_image_preload_thread", None):
            self._image_preload_thread = None
            self._image_preload_active_key = ""
        try:
            thread.deleteLater()
        except Exception:
            pass
        if (not getattr(self, "_closing", False)
                and getattr(self, "_image_preload_pending", False)):
            self._image_preload_pending = False
            QTimer.singleShot(0, self._schedule_next_chapter_image_preload)

    def _reader_file_url(self, path: str) -> str:
        """Shared-core hook (``reader_doc.ReaderDocMixin``): local file URL for WebEngine."""
        return QUrl.fromLocalFile(path).toString()

    # _process_html, _get_embedded_css, _resolve_attach_css_to_chapters, _wrap_html moved verbatim to reader_doc.ReaderDocMixin (inherited).


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = EpubLibraryDialog()
    dlg.exec()
