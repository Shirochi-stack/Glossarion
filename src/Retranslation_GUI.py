"""
Retranslation GUI Module
Force retranslation functionality for EPUB, text, and image files
"""

import os
import sys
import json
import re
import importlib.util
import html as html_lib
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import unquote
from PySide6.QtWidgets import (QWidget, QDialog, QLabel, QFrame, QListWidget, 
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QMessageBox, QFileDialog, QTabWidget, QListWidgetItem,
                                QScrollArea, QSizePolicy, QMenu, QAbstractItemView,
                                QPlainTextEdit, QTextBrowser, QStackedWidget, QComboBox, QInputDialog,
                                QLineEdit, QProgressBar, QGraphicsOpacityEffect, QWidgetAction,
                                QApplication, QSpinBox, QDialogButtonBox)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve, Property, QEventLoop, QUrl, QItemSelectionModel, QSize, QPoint, QEvent, QObject, QFileSystemWatcher
from PySide6.QtGui import QFont, QFontMetricsF, QColor, QTransform, QIcon, QPixmap, QDesktopServices, QPalette, QKeySequence, QShortcut
import xml.etree.ElementTree as ET
import zipfile
import shutil
import traceback
import subprocess
import platform
import time
import threading
import queue
import hashlib
import unicodedata
from sdlxliff_sidecar_writer import _write_html_sdlxliff_sidecar
from sdlxliff_sidecar_writer import (
    _SIDECAR_FRESHNESS_MANIFEST_LOCK,
    _SIDECAR_FRESHNESS_MANIFEST_TYPE,
    USER_ADDED_BREAK_POSITIONS_ATTRIBUTE,
    USER_ADDED_TARGET_INDEXES_ATTRIBUTE,
    _blank_manual_untranslated_sdlxliff_target,
    _clear_manual_untranslated_sdlxliff,
    _is_manual_editing_sdlxliff,
    _is_manual_untranslated_sdlxliff,
    _reset_sdlxliff_target_for_manual_retranslation,
)
from glossary_usage import (
    CHECK_PREFIX,
    WARNING_PREFIX,
    _is_special_basename,
    build_chapter_footnote,
    parse_glossary_file,
    read_translated_output_text,
    read_epub_spine_chapters,
    write_completed_summary,
)
from metadata_progress import (
    METADATA_PROGRESS_KEY,
    build_metadata_progress_plan,
    is_metadata_progress_entry,
    metadata_field_complete,
    resolve_metadata_field_settings,
)
from pdf_output_naming import (
    move_pdf_output_to_readable_name,
    readable_pdf_section_filename,
)
from translation_artifacts import (
    TRANSLATION_ARTIFACT_SPECS,
    is_translation_artifact_progress_entry,
    reset_translation_artifact_progress_entries,
    translation_artifact_path,
    translation_artifacts_are_recycled_linked,
    translation_artifact_spec_for_filename,
    translation_artifact_spec_for_kind,
)

from progress_core import (  # noqa: F401 -- moved in U5; re-exported for desktop code and tests
    ProgressViewMixin,
    _ARTIFACT_ROW_STATUS_RANK,
    _CHUNK_QA_MIRROR_FIELDS,
    _IS_MACOS,
    _LLM_TOKEN_QA_RE,
    _MISSING_IMAGE_QA_RE,
    _NON_CHAPTER_OUTPUT_FILENAMES,
    _PROGRESS_DIRECT_ROW_UPDATE_LIMIT,
    _PROGRESS_LIVE_REFRESH_MIN_INTERVAL_SECONDS,
    _PROGRESS_READER_HTML_EXTENSIONS,
    _PROGRESS_SIDECAR_FILENAMES,
    _PROGRESS_WATCH_DEBOUNCE_MS,
    _QA_MARK_FIELDS,
    _RAW_FOREIGN_TEXT_QA_RE,
    _RETRANSLATION_PROGRESS_LOCKS,
    _RETRANSLATION_PROGRESS_LOCKS_GUARD,
    _artifact_row_sort_rank,
    _chunk_ledger_for_progress_entry,
    _clear_refinement_progress_fields,
    _format_qa_issue_for_progress_display,
    _index_epub_html_members,
    _is_progress_sidecar_entry,
    _match_epub_html_member_basename,
    _merge_and_write_retranslation_progress,
    _merge_retranslation_progress_changes,
    _normalize_progress_match_name,
    _normalize_progress_match_text,
    _pending_mark_chunk_blocks,
    _pending_mark_output_path,
    _persist_progress_manager_source_link,
    _progress_entry_has_llm_token_qa,
    _progress_entry_has_meaningful_tts_state,
    _progress_entry_has_missing_image_qa,
    _progress_entry_has_raw_foreign_text_qa,
    _progress_entry_is_completed_image_only_for_display,
    _progress_entry_model_for_display,
    _progress_entry_refined_for_display,
    _progress_entry_refinement_failed_for_display,
    _progress_item_is_html,
    _progress_path_signature,
    _progress_snapshot_listing,
    _progress_status_hides_model_for_display,
    _progress_total_label,
    _qa_value_has_llm_token_issue,
    _qa_value_has_missing_image_issue,
    _restore_pending_mark_record,
    _retranslation_progress_lock,
    _select_progress_entry_for_display,
    _snapshot_progress_output_dir,
    _sync_parent_chunk_qa_summary,
    _write_progress_snapshot_atomic,
    delete_image_folder_items,
    image_folder_delete_confirmation,
    image_folder_delete_message,
    image_folder_mark_skipped_message,
    image_folder_not_found_message,
    image_folder_output_dir,
    image_folder_row_text,
    mark_image_folder_items_skipped,
    mutate_progress,
    scan_image_folder,
    progress_entry_has_qa_mark,
    progress_stats_labels,
    progress_status_color,
)
from progress_actions import (  # noqa: F401 -- moved in U5; re-exported for desktop code and tests
    _bulk_retranslation_sidecar_updates,
    _clear_llm_token_qa_markers,
    _clear_missing_image_qa_markers,
    _partial_b_request,
    _partial_b_target,
    _qa_mapping_is_missing_image_issue,
    _qa_scalar_is_missing_image_issue,
    _recover_pending_marks,
    _remove_pending_marks,
    _repair_empty_attribute_qa_file,
    _without_llm_token_qa,
    _without_missing_image_qa,
    clear_chunk_row_qa_mark,
    clear_progress_entry_qa_mark,
    find_row_audio,
    insert_missing_images,
    plan_remove_qa_marks,
    refinement_status_keys,
    remove_qa_marks,
    reset_tts,
    resolve_llm_token_qa,
    restore_in_progress,
    apply_retranslation,
    plan_retranslation,
    prepare_single_qa_resolution,
    retranslation_result_message,
    _find_progress_entry as _progress_find_progress_entry,
    _normalize_filename as _progress_normalize_filename,
    _reset_tts_progress_for_output as _progress_reset_tts_for_output,
    remove_refinement_status as _progress_remove_refinement_status,
)
from glossary_progress_core import (  # noqa: F401 -- moved in U5; re-exported for desktop code and tests
    _combine_glossary_progress_legend_stats,
    _derive_glossary_refinement_aggregate_status,
    _filter_glossary_source_chapter_map,
    _find_matching_glossary_refinement_aggregate,
    _glossary_progress_filename_keys,
    _glossary_refinement_manual_completion_info,
    _glossary_refinement_row_detail,
    _glossary_refinement_type_key,
    _map_zero_based_glossary_progress_index,
    _merge_glossary_refinement_row_info,
    _normalize_glossary_refinement_selection,
    _parallel_glossary_progress_filename_aliases,
    finish_manual_glossary_refinement,
    glossary_progress_locator,
    make_glossary_progress_model,
    prepare_manual_glossary_refinement,
    run_manual_glossary_refinement,
)
from sdlxliff_review_core import (  # noqa: F401 -- moved in U7; re-exported for desktop code and tests
    SdlxliffAutogenMixin,
    SdlxliffReviewCoreMixin,
    _MACHINE_TRANSLATION_DIR,
    _SDLXLIFF_SIDECAR_MANIFEST_LOCK,
    _SDLXLIFF_SIDECAR_MANIFEST_TYPE,
    _SDLXLIFF_SIDECAR_MANIFEST_VERSION,
    _existing_sdlxliff_sidecars_by_logical_output,
    _get_app_dir,
    _manual_editing_output_filename,
    _read_sdlxliff_sidecar_manifest,
    _sdlxliff_decode_html_bytes,
    _sdlxliff_file_stat_record,
    _sdlxliff_logical_output_key,
    _sdlxliff_machine_translation_output_name,
    _sdlxliff_machine_translation_path,
    _sdlxliff_manifest_freshness_record,
    _sdlxliff_manifest_record_current,
    _sdlxliff_sha256_bytes,
    _sdlxliff_sha256_file,
    _sdlxliff_sidecar_manifest_path,
    _sdlxliff_source_record,
    _sdlxliff_source_record_payload,
    _update_sdlxliff_sidecar_manifest,
)
from epub_package import (
    find_epub_opf_member,
    find_opf_path as find_workspace_opf_path,
)
from chapter_chunk_progress import (
    chunk_failure_summary,
    effective_parent_status,
    ensure_chunk_entry_schema,
    extract_marked_chunks,
    extract_marked_chunks_for_entry,
    is_multi_chunk_entry,
    remove_chunk_segments_from_file,
    reset_chunks_for_retranslation,
    set_chunk_qa,
    sorted_chunk_items,
)
from chapter_display_numbering import nonreset_chapter_display_numbers


class _KeepOpenActionMenu(QMenu):
    """QMenu that leaves explicitly persistent actions open after activation."""

    KEEP_OPEN_PROPERTY = "sdl_keep_menu_open"

    @classmethod
    def _keeps_open(cls, action):
        return bool(
            action is not None
            and action.isEnabled()
            and action.property(cls.KEEP_OPEN_PROPERTY)
        )

    def mouseReleaseEvent(self, event):
        try:
            position = event.position().toPoint()
        except AttributeError:
            position = event.pos()
        action = self.actionAt(position)
        if self._keeps_open(action):
            self.setActiveAction(action)
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if (
            event.key() in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space}
            and self._keeps_open(self.activeAction())
        ):
            self.activeAction().trigger()
            event.accept()
            return
        super().keyPressEvent(event)


# --- Non-blocking access to TransateKRtoEN.ProgressManager ------------------
# TransateKRtoEN is a very heavy import chain (tiktoken, ebooklib, bs4,
# unified_api_client, GlossaryManager, ...). In frozen (PyInstaller onefile)
# builds the first import can take 10-20s. When the translation worker thread
# starts that import, any plain `from TransateKRtoEN import ProgressManager`
# executed on the GUI thread (e.g. by this dialog's 2s auto-refresh timer)
# blocks on Python's per-module import lock until the worker finishes —
# freezing the entire GUI for the whole import. These helpers only ever do a
# lock-free sys.modules/getattr lookup on the GUI thread and, if the module
# isn't ready yet, warm it once in a background thread instead of blocking.
_TRANSLATE_MODULE_WARMUP_STARTED = threading.Event()
_EPUB_LIBRARY_WARMUP_STARTED = threading.Event()
_EPUB_READER_ENGINE_PREWARM_SCHEDULED = False


def _warm_translate_module_in_background():
    """Kick off the heavy TransateKRtoEN import on a daemon thread (once)."""
    if _TRANSLATE_MODULE_WARMUP_STARTED.is_set():
        return
    _TRANSLATE_MODULE_WARMUP_STARTED.set()

    def _warm():
        try:
            import TransateKRtoEN  # noqa: F401
        except Exception:
            pass

    threading.Thread(target=_warm, name="translate-module-warmup", daemon=True).start()


def _get_progress_manager_nonblocking():
    """Return TransateKRtoEN.ProgressManager without touching the import lock.

    Returns None when TransateKRtoEN is not fully imported yet (e.g. it is
    being imported right now by the translation worker in a frozen build).
    Callers on the GUI thread should treat None as "skip this tick" rather
    than importing the module themselves.
    """
    mod = sys.modules.get('TransateKRtoEN')
    if mod is not None:
        pm = getattr(mod, 'ProgressManager', None)
        if pm is not None:
            return pm
    _warm_translate_module_in_background()
    return None


def _warm_epub_library_in_background():
    """Warm the reader's lazy module before a Progress Manager menu click.

    Frozen builds can spend seconds unpacking/importing QtWebEngine on the
    first ``epub_library`` import. Progress Manager already exists by the time
    this helper runs, so importing on a daemon thread hides that one-time cost
    behind the user's normal interaction with the dialog.
    """
    if 'epub_library' in sys.modules or _EPUB_LIBRARY_WARMUP_STARTED.is_set():
        return
    _EPUB_LIBRARY_WARMUP_STARTED.set()

    def _warm():
        try:
            import epub_library  # noqa: F401
        except Exception:
            pass

    threading.Thread(
        target=_warm,
        name="epub-reader-module-warmup",
        daemon=True,
    ).start()


def _schedule_epub_reader_engine_prewarm():
    """Create the hidden Chromium warmup view once the lazy import finishes."""
    global _EPUB_READER_ENGINE_PREWARM_SCHEDULED
    if _EPUB_READER_ENGINE_PREWARM_SCHEDULED:
        return
    _EPUB_READER_ENGINE_PREWARM_SCHEDULED = True
    _warm_epub_library_in_background()

    def _try_prewarm(attempt=0):
        module = sys.modules.get('epub_library')
        prewarm = getattr(module, 'prewarm_epub_reader_webengine', None)
        if callable(prewarm):
            try:
                prewarm()
            except Exception:
                pass
            return
        # A frozen-build import can take several seconds. Poll without ever
        # touching its import lock from the GUI thread.
        if attempt < 200:
            QTimer.singleShot(
                50,
                lambda: _try_prewarm(attempt + 1),
            )

    # Let the Progress Manager shell and first list batch paint before the
    # one-time Chromium startup runs on the GUI thread.
    QTimer.singleShot(250, _try_prewarm)
# -----------------------------------------------------------------------------




def _resolve_dialog_window_parent(parent):
    """Return the top-level window that owns a dialog-triggering child widget."""
    if parent is None:
        return None
    try:
        owner = parent.window()
        if owner is not None:
            return owner
    except (AttributeError, RuntimeError):
        pass
    return parent


class _GlossaryProgressAsyncBridge(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, on_finished=None, on_failed=None, on_progress=None, parent=None):
        super().__init__(parent)
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_progress = on_progress
        self.progress.connect(self._handle_progress)
        self.finished.connect(self._handle_finished)
        self.failed.connect(self._handle_failed)

    @Slot(object)
    def _handle_progress(self, payload):
        if callable(self._on_progress):
            self._on_progress(payload)

    @Slot(object)
    def _handle_finished(self, payload):
        if callable(self._on_finished):
            self._on_finished(payload)

    @Slot(str)
    def _handle_failed(self, message):
        if callable(self._on_failed):
            self._on_failed(message)





# WindowManager and UIHelper removed - not needed in PySide6
# Qt handles window management and UI utilities automatically


class AnimatedRefreshButton(QPushButton):
    """Custom QPushButton with rotation animation for refresh action using Halgakos.ico"""
    
    def __init__(self, text="Refresh", parent=None):
        super().__init__(text, parent)
        self._rotation = 0
        self._animation = None
        self._original_text = text
        self._timer = None
        self._animation_step = 0
        self._original_icon = None
        
        # Try to load Halgakos.ico
        try:
            # Get base directory
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                self._original_icon = QIcon(ico_path)
                self.setIcon(self._original_icon)
                self.setIconSize(self.iconSize() * 1.2)  # Make icon slightly larger
        except Exception as e:
            print(f"Could not load Halgakos.ico for refresh button: {e}")
        
    def get_rotation(self):
        return self._rotation
    
    def set_rotation(self, angle):
        self._rotation = angle
        self.update()  # Trigger repaint
    
    # Define rotation as a Qt Property for animation
    rotation = Property(float, get_rotation, set_rotation)
    
    def start_animation(self):
        """Start the spinning animation"""
        if self._timer and self._timer.isActive():
            return  # Already animating
        
        self.setProperty("refreshActive", True)
        self.style().unpolish(self)
        self.style().polish(self)
        
        # Start timer-based animation for icon rotation
        self._animation_step = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation_frame)
        self._timer.start(50)  # Update every 50ms for smooth rotation
    
    def _update_animation_frame(self):
        """Update animation frame by rotating the icon"""
        if self._original_icon:
            # Increment rotation angle (30 degrees per frame for smooth spinning)
            self._rotation = (self._rotation + 30) % 360
            
            # Create a rotated version of the icon
            pixmap = self._original_icon.pixmap(self.iconSize())
            transform = QTransform().rotate(self._rotation)
            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
            
            # Set the rotated icon
            self.setIcon(QIcon(rotated_pixmap))
    
    def stop_animation(self):
        """Stop the spinning animation"""
        if self._timer:
            self._timer.stop()
            self._timer = None
            self._rotation = 0
            self._animation_step = 0
            
            # Restore original icon (unrotated)
            if self._original_icon:
                self.setIcon(self._original_icon)
            
            self.setProperty("refreshActive", False)
            self.style().unpolish(self)
            self.style().polish(self)
            
            self.update()


class SDLXLIFFReviewDialog(SdlxliffReviewCoreMixin, QDialog):
    """Internal source/output reviewer for generated SDLXLIFF HTML sidecars."""

    _tooltip_translation_finished = Signal(int, object, str)
    _tooltip_translation_progress = Signal(int, int)
    _tooltip_translation_status = Signal(int, object, str)
    _tooltip_translation_batch_finished = Signal(int, int, str)
    _review_data_preload_finished = Signal(int, object)
    _review_refresh_scan_finished = Signal(int, object)
    _review_piece_reload_finished = Signal(int, object)
    _review_generation_progress = Signal(object)
    # Constants and the widget-free methods: sdlxliff_review_core.SdlxliffReviewCoreMixin.

    def __init__(
        self,
        output_dir,
        current_path=None,
        parent=None,
        config=None,
        autogen_owner=None,
        autogen_file_path=None,
        autogen_progress_data=None,
        autogen_output_files=None,
        autogen_manual_entries=None,
    ):
        super().__init__(parent)
        self._init_review_state(
            output_dir,
            current_path,
            parent,
            config,
            autogen_owner,
            autogen_file_path,
            autogen_progress_data,
            autogen_output_files,
            autogen_manual_entries,
        )
        self._edit_save_timer = QTimer(self)
        self._edit_save_timer.setSingleShot(True)
        self._edit_save_timer.timeout.connect(self._flush_target_edits)
        self._render_token = 0
        self._active_render_timer = None
        self._rows_rebuild_active = False
        self._first_show_render_started = False
        self._initial_piece_row = 0
        self._piece_pages = {}
        self._piece_render_complete = set()
        self._piece_scroll_positions = {}
        self._current_scroll_piece_row = None
        self._restoring_review_scroll = False
        self._active_render_row = None
        self._active_render_page = None
        self._preload_render_timer = None
        self._preload_render_queue = []
        self._preload_render_row = None
        self._preload_render_page = None
        self._preload_render_state = None
        self._preload_start_queued = False
        self._review_page_cache_trim_queued = False
        self._last_review_selection_change = 0.0
        self._review_data_preload_token = 0
        self._review_data_preload_running = False
        self._review_data_preload_requested = False
        self._review_data_preload_queued = False
        self._sdl_review_loading_icon_timer = None
        self._sdl_review_loading_icon = None
        self._sdl_review_loading_original_pixmap = None
        self._sdl_review_loading_angle = 0
        self._review_loading_minimum_ms = 10
        self._review_dirty_preview_refresh_queued = False
        self._status_jump_indices = {}
        self._highlighted_status_frame = None
        self._book_nav_combo = None
        self._book_nav_prev = None
        self._book_nav_next = None
        self._book_nav_counter = None
        self._book_nav_updating = False
        self._last_review_signature = None
        self._last_machine_translation_signature = None
        self._review_image_assets_output = ""
        self._notepad_mode_supported = self._detect_notepad_mode_support()
        self._two_column_layout_enabled = self._review_two_column_layout_enabled()
        if not self._notepad_mode_supported:
            # A full-build Notepad preference may be shared with a Lite build.
            # Keep it intact in config, but never enter the unavailable mode.
            self._two_column_layout_enabled = True
        self._auto_refresh_timer = None
        self._refreshing_review_data = False
        self._review_data_loaded = False
        self._initial_review_load_started = False
        self._queued_review_refresh = False
        self._review_refresh_scan_token = 0
        self._review_refresh_scan_running = False
        self._last_review_scan_finished_at = 0.0
        self._review_refresh_scan_requested = False
        self._review_refresh_scan_queued = False
        self._review_refresh_scan_force = False
        self._review_refresh_scan_validate = False
        self._review_refresh_scan_current_path = None
        self._review_piece_reload_token = 0
        self._review_piece_reload_running = False
        self._review_piece_reload_requested = False
        self._seamless_review_old_page = None
        self._review_transition_overlay = None
        self._review_transition_anim = None
        self._review_transition_held = False
        self._pending_review_transition_pixmap = None
        self._review_context_menu_open = False
        self._review_text_context_menu = None
        self._piece_list_context_menu = None
        self._machine_translation_provider_menu = None
        self._generation_stream_pending_pieces = []
        self._generation_stream_finished_message = ""
        self._generation_stream_preserve_after_finish = False
        self._generation_stream_flush_timer = QTimer(self)
        self._generation_stream_flush_timer.setSingleShot(True)
        self._generation_stream_flush_timer.timeout.connect(self._flush_generated_sidecar_stream_pieces)
        self._manual_refresh_shortcut = None
        self._full_screen_shortcut = None
        self._review_was_maximized_before_full_screen = False
        self._refresh_button_timer = None
        self._refresh_button_stop_timer = None
        self._refresh_button_frame = 0
        self._flag_accuracy_button_timer = None
        self._flag_accuracy_button_stop_timer = None
        self._flag_accuracy_button_frame = 0
        self._tooltip_translation_running = False
        self._tooltip_translation_batch_active = False
        self._tooltip_translation_finished.connect(self._apply_tooltip_translations)
        self._tooltip_translation_progress.connect(self._update_tooltip_translation_progress)
        self._tooltip_translation_status.connect(self._apply_tooltip_translation_status)
        self._tooltip_translation_batch_finished.connect(self._finish_piece_list_tooltip_translations)
        self._review_data_preload_finished.connect(self._apply_review_data_preload)
        self._review_refresh_scan_finished.connect(self._apply_review_refresh_scan)
        self._review_piece_reload_finished.connect(self._apply_async_review_piece_reload)
        self._review_generation_progress.connect(self._apply_review_generation_progress)
        self.setWindowTitle("SDLXLIFF Source -> Output Review - Credits: OMORIO")
        self.setObjectName("SDLXLIFFReviewDialog")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, False)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setWindowModality(Qt.NonModal)
        self.resize(1500, 900)
        self._full_screen_shortcut = QShortcut(QKeySequence("F11"), self)
        self._full_screen_shortcut.setContext(Qt.WindowShortcut)
        self._full_screen_shortcut.activated.connect(self._toggle_review_full_screen)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(self.THEME["bg"]))
        self.setPalette(palette)
        self._apply_translator_theme(parent)
        try:
            if parent is not None and not parent.windowIcon().isNull():
                self.setWindowIcon(parent.windowIcon())
        except Exception:
            pass

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        book_nav = self._create_review_book_navigation()
        if book_nav is not None:
            main_layout.addWidget(book_nav)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(6)
        main_layout.addLayout(content_row, 1)

        self.piece_list = QListWidget()
        self.piece_list.setObjectName("SdlReviewPieceList")
        self.piece_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.piece_list.setContextMenuPolicy(Qt.NoContextMenu)
        self.piece_list.installEventFilter(self)
        self.piece_list.viewport().installEventFilter(self)
        self.piece_list.setUniformItemSizes(True)
        self.piece_list.setMinimumWidth(242)
        self.piece_list.setMaximumWidth(286)
        self.piece_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.piece_list.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        content_row.addWidget(self.piece_list, 0)

        detail = QWidget()
        detail.setObjectName("SdlReviewDetail")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)
        content_row.addWidget(detail, 1)

        self.header_label = QLabel()
        self.header_label.setTextFormat(Qt.PlainText)
        self.header_label.setWordWrap(True)
        self.header_label.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {self.THEME['accent']};")
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self.header_label, 1)
        self.translate_tooltips_btn = QPushButton(self.TRANSLATE_TOOLTIPS_BUTTON_TEXT)
        self.translate_tooltips_btn.setCursor(Qt.PointingHandCursor)
        self.translate_tooltips_btn.setToolTip("Generate source-row machine translation previews. Right-click to choose the provider.")
        self.translate_tooltips_btn.setStyleSheet(
            "QPushButton { background-color:#2b4f6f; color:#d7ecff; border:1px solid #5a9fd4; "
            "border-radius:4px; padding:4px 10px; font-size:9pt; font-weight:bold; }"
            "QPushButton:hover { background-color:#356b96; border-color:#7bb3e0; }"
            "QPushButton:disabled { color:#94a3b8; background-color:#2a3b4d; border-color:#4a5568; }"
        )
        self.translate_tooltips_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self.translate_tooltips_btn.customContextMenuRequested.connect(self._show_machine_translation_provider_menu)
        self.translate_tooltips_btn.clicked.connect(self._translate_current_piece_tooltips)
        self._update_machine_translation_button_tooltip()
        self.flag_accuracy_btn = QPushButton(self.FLAG_ACCURACY_BUTTON_TEXT)
        self.flag_accuracy_btn.setCursor(Qt.PointingHandCursor)
        self.flag_accuracy_btn.setToolTip("Mark rows that fall below the machine translation preview accuracy threshold.")
        self.flag_accuracy_btn.setStyleSheet(
            "QPushButton { background-color:#3b2450; color:#f0d9ff; border:1px solid #9c63d8; "
            "border-radius:4px; padding:4px 10px; font-size:9pt; font-weight:bold; }"
            "QPushButton:hover { background-color:#4c2d67; border-color:#b982f0; }"
            "QPushButton:disabled { color:#a891b8; background-color:#2d2338; border-color:#604275; }"
        )
        self.flag_accuracy_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self.flag_accuracy_btn.customContextMenuRequested.connect(self._show_flag_accuracy_context_menu)
        self.flag_accuracy_btn.clicked.connect(self._flag_current_piece_inaccurate_translations)
        self.two_column_layout_btn = QPushButton(self.TWO_COLUMN_LAYOUT_BUTTON_TEXT)
        self.two_column_layout_btn.setCursor(Qt.PointingHandCursor)
        self.two_column_layout_btn.setCheckable(True)
        self.two_column_layout_btn.setChecked(bool(self._two_column_layout_enabled))
        self._update_review_layout_button()
        self.two_column_layout_btn.setStyleSheet(
            "QPushButton { background-color:#253241; color:#d7ecff; border:1px solid #547596; "
            "border-radius:4px; padding:4px 10px; font-size:9pt; font-weight:bold; }"
            "QPushButton:hover { background-color:#324d68; border-color:#7bb3e0; }"
            "QPushButton:checked { background-color:#205f74; color:#e9fbff; border-color:#26a6c8; }"
            "QPushButton:checked:hover { background-color:#26738c; border-color:#5bc4df; }"
        )
        self.two_column_layout_btn.toggled.connect(self._set_review_two_column_layout)
        self.refresh_review_btn = QPushButton(self.MANUAL_REFRESH_BUTTON_TEXT)
        self.refresh_review_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_review_btn.setToolTip("Run the SDLXLIFF auto-refresh check now (F5).")
        self.refresh_review_btn.setStyleSheet(
            "QPushButton { background-color:#28394c; color:#d7ecff; border:1px solid #547596; "
            "border-radius:4px; padding:4px 10px; font-size:9pt; font-weight:bold; }"
            "QPushButton:hover { background-color:#324d68; border-color:#7bb3e0; }"
            "QPushButton:disabled { color:#94a3b8; background-color:#253241; border-color:#4a5568; }"
        )
        self.refresh_review_btn.clicked.connect(self._manual_review_refresh)
        detail_layout.addLayout(header_row)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(10)
        legend_row.addWidget(self._legend_status_label("green ok", "green"))
        legend_row.addWidget(self._legend_status_label("yellow density/tag-level", "yellow"))
        legend_row.addWidget(self._legend_status_label("purple MT inaccurate", "purple"))
        legend_row.addWidget(self._legend_status_label("red dropped/added/empty/untranslated", "red"))
        legend_row.addSpacing(24)
        legend_row.addWidget(self.translate_tooltips_btn, 0, Qt.AlignVCenter)
        legend_row.addWidget(self.flag_accuracy_btn, 0, Qt.AlignVCenter)
        legend_row.addWidget(self.two_column_layout_btn, 0, Qt.AlignVCenter)
        legend_row.addStretch(1)
        legend_row.addWidget(self.refresh_review_btn, 0, Qt.AlignRight | Qt.AlignVCenter)
        detail_layout.addLayout(legend_row)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("SdlReviewScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().setAutoFillBackground(True)
        viewport_palette = self.scroll.viewport().palette()
        viewport_palette.setColor(QPalette.Window, QColor(self.THEME["bg"]))
        self.scroll.viewport().setPalette(viewport_palette)
        self.scroll.viewport().setStyleSheet(f"background-color: {self.THEME['bg']};")
        self.rows_stack = QStackedWidget()
        self.rows_stack.setObjectName("SdlReviewRowsStack")
        self.rows_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_review_widget_background(self.rows_stack)
        self.loading_page = self._create_review_loading_page()
        self.rows_stack.addWidget(self.loading_page)
        self.rows_widget = self.loading_page
        self.rows_layout = self.loading_page.layout()
        self.scroll.setWidget(self.rows_stack)
        self.scroll.verticalScrollBar().valueChanged.connect(self._remember_current_review_scroll)
        detail_layout.addWidget(self.scroll, 1)

        close_row = QHBoxLayout()
        self.save_status_label = QLabel("")
        self.save_status_label.setTextFormat(Qt.PlainText)
        self.save_status_label.setStyleSheet(f"color: {self.THEME['muted']}; background: transparent;")
        close_row.addWidget(self.save_status_label)
        self.generation_progress_bar = QProgressBar()
        self.generation_progress_bar.setTextVisible(True)
        self.generation_progress_bar.setFixedHeight(16)
        self.generation_progress_bar.setMinimumWidth(280)
        self.generation_progress_bar.setMaximumWidth(420)
        self.generation_progress_bar.setStyleSheet(
            "QProgressBar { background-color:#202936; color:#d7ecff; border:1px solid #4a5568; "
            "border-radius:4px; text-align:center; font-size:8pt; font-weight:bold; }"
            "QProgressBar::chunk { background-color:#5a9fd4; border-radius:3px; }"
        )
        self.generation_progress_bar.hide()
        close_row.addWidget(self.generation_progress_bar, 0, Qt.AlignVCenter)
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        main_layout.addLayout(close_row)

        self.header_label.setText("Loading SDLXLIFF review...")
        try:
            self.loading_label.setText("Loading SDLXLIFF...")
        except Exception:
            pass
        self._manual_refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        self._manual_refresh_shortcut.setContext(Qt.WindowShortcut)
        self._manual_refresh_shortcut.activated.connect(self._manual_review_refresh)

    def _toggle_review_full_screen(self):
        """Toggle F11 full screen while restoring the preceding window state."""
        if self.isFullScreen():
            if self._review_was_maximized_before_full_screen:
                self.showMaximized()
            else:
                self.showNormal()
            return
        self._review_was_maximized_before_full_screen = self.isMaximized()
        self.showFullScreen()

    def showEvent(self, event):
        super().showEvent(event)
        if self._first_show_render_started:
            self._start_review_auto_refresh()
            # Force an immediate refresh on re-show instead of waiting for
            # the next auto-refresh tick and its cooldown window.
            try:
                if self._review_data_loaded and not getattr(self, '_review_refresh_scan_running', False):
                    self._queue_review_refresh_scan(force=False, current_path=self.current_path, delay_ms=50)
            except Exception:
                pass
            return
        self._first_show_render_started = True
        try:
            self.ensurePolished()
            layout = self.layout()
            if layout is not None:
                layout.activate()
            self._refresh_review_stream_geometry(final=True)
        except Exception:
            pass
        self._start_review_auto_refresh()
        if not self._review_data_loaded and not self._initial_review_load_started:
            self._initial_review_load_started = True
            self._show_review_loading_page()
            self._queue_review_refresh_scan(force=False, current_path=self.current_path, delay_ms=25)
            return
        if self.pieces:
            QTimer.singleShot(0, lambda: self._render_piece(self._initial_piece_row))

    def hideEvent(self, event):
        super().hideEvent(event)
        try:
            if self._auto_refresh_timer is not None and self._auto_refresh_timer.isActive():
                self._auto_refresh_timer.stop()
        except Exception:
            pass

    def _queue_review_refresh(self, force=True, current_path=None, signature=None, seamless=False, delay_ms=25):
        if self._queued_review_refresh:
            return
        self._queued_review_refresh = True
        if not seamless:
            self._show_review_loading_page()

        def _run_refresh():
            self._queued_review_refresh = False
            self.refresh_review_data(
                force=force,
                current_path=current_path,
                signature=signature,
                seamless=seamless,
            )

        QTimer.singleShot(max(0, int(delay_ms)), _run_refresh)

    def _queue_review_refresh_scan(self, force=False, current_path=None, delay_ms=350, validate=False):
        try:
            if not bool(getattr(self, "_review_data_loaded", False)):
                try:
                    self._show_review_loading_page()
                    self.save_status_label.setText("Checking SDLXLIFF sidecars...")
                except Exception:
                    pass
            self._review_refresh_scan_force = bool(getattr(self, "_review_refresh_scan_force", False) or force)
            self._review_refresh_scan_validate = bool(getattr(self, "_review_refresh_scan_validate", False) or validate)
            if current_path is not None:
                self._review_refresh_scan_current_path = current_path
            elif not getattr(self, "_review_refresh_scan_current_path", None):
                self._review_refresh_scan_current_path = self.current_path
            if getattr(self, "_review_refresh_scan_running", False):
                self._review_refresh_scan_requested = True
                return
            if getattr(self, "_review_refresh_scan_queued", False):
                self._review_refresh_scan_requested = True
                return
            self._review_refresh_scan_queued = True
            QTimer.singleShot(max(0, int(delay_ms)), self._start_review_refresh_scan)
        except Exception:
            pass

    def _start_review_refresh_scan(self):
        try:
            self._review_refresh_scan_queued = False
            if getattr(self, "_review_refresh_scan_running", False):
                self._review_refresh_scan_requested = True
                return
            self._review_refresh_scan_token = int(getattr(self, "_review_refresh_scan_token", 0)) + 1
            token = self._review_refresh_scan_token
            force = bool(getattr(self, "_review_refresh_scan_force", False))
            validate = bool(getattr(self, "_review_refresh_scan_validate", False))
            current_path = getattr(self, "_review_refresh_scan_current_path", None) or self.current_path
            self._review_refresh_scan_force = False
            self._review_refresh_scan_validate = False
            self._review_refresh_scan_current_path = None
            self._review_refresh_scan_requested = False
            self._review_refresh_scan_running = True
            last_review_signature = getattr(self, "_last_review_signature", None)
            last_mt_signature = getattr(self, "_last_machine_translation_signature", None)
            last_autogen_signature = getattr(self, "_last_autogen_signature", None)

            def _worker():
                self._review_refresh_scan_finished.emit(
                    token,
                    self._build_review_refresh_scan_result(
                        force=force,
                        validate=validate,
                        current_path=current_path,
                        last_review_signature=last_review_signature,
                        last_mt_signature=last_mt_signature,
                        last_autogen_signature=last_autogen_signature,
                    ),
                )

            threading.Thread(target=_worker, name="sdlxliff-review-refresh-scan", daemon=True).start()
        except Exception:
            self._review_refresh_scan_running = False
            self._last_review_scan_finished_at = time.monotonic()
            self._queue_stop_refresh_button_animation(150)

    def _emit_review_generation_progress(self, payload):
        try:
            self._review_generation_progress.emit(payload if isinstance(payload, dict) else {"message": str(payload)})
        except Exception:
            pass

    def _prepare_generation_streaming_piece_list(self, total=0):
        try:
            if getattr(self, "_generation_streaming_active", False):
                return True
            if self.pieces or self.piece_list.count():
                return False
            if self._sdlxliff_sidecar_paths_for_output_dir(self.output_dir):
                return False
            try:
                self.piece_list.currentRowChanged.disconnect(self._request_render_piece)
            except Exception:
                pass
            self.piece_list.clear()
            self.pieces = []
            self._piece_pages.clear()
            self._piece_render_complete.clear()
            self._generation_streaming_active = True
            self._generation_stream_seen_paths = set()
            self._generation_stream_skipped = 0
            self._generation_stream_pending_pieces = []
            self._generation_stream_finished_message = ""
            self._generation_stream_preserve_after_finish = False
            self._generation_stream_total = max(0, int(total or 0))
            self._generation_stream_progress_map = self._read_progress_metadata()
            self._generation_stream_spine_positions = self._read_spine_positions(allow_deep_search=False)
            self._streamed_piece_list_populated = True
            try:
                self.piece_list.currentRowChanged.connect(self._request_render_piece)
            except Exception:
                pass
            self._spin_review_loading_icon()
            QTimer.singleShot(0, self.piece_list.update)
            return True
        except Exception:
            return False

    def _flush_generated_sidecar_stream_pieces(self):
        trace_started = time.perf_counter()
        try:
            pending = getattr(self, "_generation_stream_pending_pieces", None)
            if not isinstance(pending, list):
                self._generation_stream_pending_pieces = []
                pending = self._generation_stream_pending_pieces
            pending_before = len(pending)
            flushed = 0
            while pending and flushed < 4:
                queued = pending.pop(0)
                path, progress_index = queued[:2]
                opf_position = queued[2] if len(queued) > 2 else None
                if self._append_generated_sidecar_stream_piece(
                    path,
                    progress_index=progress_index,
                    opf_position=opf_position,
                ):
                    flushed += 1
            if pending:
                self._trace_review_perf(
                    "flush_generated_sidecar_stream_batch",
                    trace_started,
                    pending_before=pending_before,
                    flushed=flushed,
                    remaining=len(pending),
                )
                self._schedule_generation_stream_flush(delay_ms=18)
                return
            message = str(getattr(self, "_generation_stream_finished_message", "") or "")
            if message:
                self._generation_stream_finished_message = ""
                self._finish_generation_streaming(message)
            self._trace_review_perf(
                "flush_generated_sidecar_stream_done",
                trace_started,
                force=bool(message),
                pending_before=pending_before,
                flushed=flushed,
            )
        except Exception:
            pass

    def _resort_generated_review_pieces(self):
        """Put streamed pieces into content.opf order without reparsing them."""
        if len(getattr(self, "pieces", []) or []) < 2:
            return False
        indexed = list(enumerate(self.pieces))
        ordered = sorted(
            indexed,
            key=lambda item: (
                item[1].get("opf_position") is None,
                item[1].get("opf_position")
                if item[1].get("opf_position") is not None
                else item[0],
                item[0],
            ),
        )
        if [old_index for old_index, _piece in ordered] == list(range(len(indexed))):
            return False
        selected_path = ""
        try:
            current_row = self.piece_list.currentRow()
            if 0 <= current_row < len(self.pieces):
                selected_path = self.pieces[current_row].get("path") or ""
        except Exception:
            pass
        self._render_token += 1
        self._clear_cached_review_pages()
        self.pieces = [piece for _old_index, piece in ordered]
        display_numbers = nonreset_chapter_display_numbers(
            piece.get("raw_chapter_num", piece.get("chapter_num"))
            for piece in self.pieces
        )
        for row, (piece, display_chapter_num) in enumerate(
            zip(self.pieces, display_numbers)
        ):
            piece["index"] = row
            piece["display_chapter_num"] = display_chapter_num
            display_position = (
                int(piece["opf_position"]) + 1
                if piece.get("opf_position") is not None
                else row + 1
            )
            piece["review_label"] = self._review_label_from_metadata({
                "display_position": display_position,
                "chapter_num": piece.get("chapter_num"),
                "display_chapter_num": display_chapter_num,
            })
        if selected_path:
            self.current_path = selected_path
        self._populate_piece_list()
        return True

    def _finish_generation_streaming(self, message=""):
        trace_started = time.perf_counter()
        try:
            was_streaming = bool(getattr(self, "_generation_streaming_active", False))
            self._generation_streaming_active = False
            self._resort_generated_review_pieces()
            self._review_data_loaded = bool(self.pieces)
            self._generation_stream_preserve_after_finish = bool(was_streaming and self.pieces)
            if self.pieces:
                row = self.piece_list.currentRow()
                if row < 0:
                    row = 0
                    previous_block = self.piece_list.blockSignals(True)
                    self.piece_list.setCurrentRow(row)
                    self.piece_list.blockSignals(previous_block)
                    QTimer.singleShot(0, lambda: self._render_piece(row, show_loading=False))
                else:
                    self._refresh_piece_header(row)
            if message:
                try:
                    self.save_status_label.setText(message)
                except Exception:
                    pass
            QTimer.singleShot(1200, self._hide_generation_progress)
        except Exception:
            pass
        finally:
            self._trace_review_perf(
                "finish_generation_streaming",
                trace_started,
                force=True,
                pieces=len(getattr(self, "pieces", []) or []),
                message=message[:80] if message else "",
            )

    def _append_generated_sidecar_stream_piece(self, path, progress_index=0, opf_position=None):
        trace_started = time.perf_counter()
        try:
            if not getattr(self, "_generation_streaming_active", False):
                return False
            if not path or not os.path.isfile(path):
                return False
            path_norm = os.path.normcase(os.path.abspath(path))
            seen = getattr(self, "_generation_stream_seen_paths", set())
            if path_norm in seen:
                return False
            seen.add(path_norm)
            self._generation_stream_seen_paths = seen

            row = len(self.pieces)
            progress_map = getattr(self, "_generation_stream_progress_map", {}) or {}
            spine_positions = getattr(self, "_generation_stream_spine_positions", {}) or {}
            metadata = self._sidecar_metadata(path, row, progress_map, spine_positions)
            if opf_position is not None:
                try:
                    metadata["opf_position"] = int(opf_position)
                    metadata["sort_key"] = (
                        int(opf_position),
                        self._chapter_number_from_name(metadata.get("output_name")),
                        str(metadata.get("output_name") or "").lower(),
                    )
                except (TypeError, ValueError):
                    pass
            if metadata.get("opf_position") is None:
                metadata["display_position"] = int(progress_index or row + 1)
            else:
                metadata["display_position"] = int(metadata["opf_position"]) + 1
            metadata["raw_chapter_num"] = metadata.get("chapter_num")
            try:
                raw_display_number = max(
                    0,
                    int(metadata.get("chapter_num")),
                )
            except (TypeError, ValueError, OverflowError):
                raw_display_number = 0
            if self.pieces:
                previous_piece = self.pieces[-1]
                try:
                    previous_display_number = max(
                        0,
                        int(
                            previous_piece.get(
                                "display_chapter_num",
                                previous_piece.get(
                                    "raw_chapter_num",
                                    previous_piece.get("chapter_num", 0),
                                ),
                            )
                        ),
                    )
                except (TypeError, ValueError, OverflowError):
                    previous_display_number = 0
                if (
                    previous_display_number > 0
                    and raw_display_number < previous_display_number
                ):
                    raw_display_number = previous_display_number + 1
            metadata["display_chapter_num"] = raw_display_number
            metadata["label"] = self._review_label_from_metadata(metadata)
            piece = self._build_piece(path, row, metadata)
            total = int(getattr(self, "_generation_stream_total", 0) or 0)
            if self._review_piece_is_empty_sidecar(piece):
                # Empty sidecars (e.g. the cover page, which has no text units)
                # are never shown, so drop them from the streamed denominator —
                # otherwise the bar rests at e.g. 15/16 and looks stuck.
                self._generation_stream_skipped = int(getattr(self, "_generation_stream_skipped", 0)) + 1
                if total:
                    effective = max(len(self.pieces), total - self._generation_stream_skipped)
                    self._set_loading_progress(len(self.pieces), effective, f"{len(self.pieces)}/{effective} SDLXLIFF entries")
                return False
            piece["index"] = row
            self.pieces.append(piece)
            self.piece_list.addItem(self._piece_list_item_for_piece(piece, row))
            if total:
                effective = max(len(self.pieces), total - int(getattr(self, "_generation_stream_skipped", 0)))
                self.save_status_label.setText(f"Loaded SDLXLIFF entry {len(self.pieces)}/{effective}")
                self._set_loading_progress(len(self.pieces), effective, f"{len(self.pieces)}/{effective} SDLXLIFF entries")
            if row == 0:
                previous_block = self.piece_list.blockSignals(True)
                self.piece_list.setCurrentRow(0)
                self.piece_list.blockSignals(previous_block)
                self._initial_piece_row = 0
                QTimer.singleShot(0, lambda: self._render_piece(0, show_loading=False))
            self.piece_list.update()
            self._trace_review_perf(
                "append_generated_sidecar_piece",
                trace_started,
                force=bool(row == 0 or (total and len(self.pieces) >= total)),
                row=row,
                visible=len(self.pieces),
                total=total or None,
                path=os.path.basename(str(path or "")),
            )
            return True
        except Exception:
            self._trace_review_perf(
                "append_generated_sidecar_piece_failed",
                trace_started,
                force=True,
                path=os.path.basename(str(path or "")),
            )
            return False

    def _apply_review_generation_progress(self, payload):
        try:
            if not isinstance(payload, dict):
                return
            stage = str(payload.get("stage") or "").lower()
            output_name = str(payload.get("output_name") or "")
            index = int(payload.get("index") or 0)
            total = int(payload.get("total") or 0)
            stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else None
            if stage == "start" and total:
                self._prepare_generation_streaming_piece_list(total=total)
                self._set_generation_progress(0, total, f"0/{total} SDLXLIFF")
            elif stage == "created":
                self._queue_generated_sidecar_stream_piece(
                    str(payload.get("path") or ""),
                    progress_index=index,
                    opf_position=payload.get("opf_position"),
                )
                self._set_generation_progress(index, total, f"{index}/{total} SDLXLIFF")
            elif stage in {"checking", "missing_source", "missing_output", "failed", "skipped"} and total:
                self._set_generation_progress(index, total, f"{index}/{total} SDLXLIFF")
            if stage == "finished":
                message = self._review_generation_summary(stats) or "SDLXLIFF generation finished"
                self._set_generation_progress(total, total, f"{total}/{total} SDLXLIFF")
                if getattr(self, "_generation_stream_pending_pieces", None):
                    self._generation_stream_finished_message = message
                    self._schedule_generation_stream_flush(delay_ms=16)
                else:
                    self._finish_generation_streaming(message)
            elif stage == "start":
                message = f"Generating SDLXLIFF sidecars for {total} completed entr{'y' if total == 1 else 'ies'}..."
            elif output_name and total:
                detail = str(payload.get("error") or payload.get("message") or "")
                label = {
                    "created": "Generated",
                    "missing_source": "Missing source for",
                    "missing_output": "Missing output for",
                    "failed": "Failed",
                    "skipped": "Skipped",
                }.get(stage, "Generating")
                message = f"{label} SDLXLIFF {index}/{total}: {output_name}"
                if detail and stage in {"failed", "missing_source", "missing_output"}:
                    message = f"{message} ({detail})"
            else:
                message = str(payload.get("message") or "Generating SDLXLIFF sidecars...")
            try:
                self.save_status_label.setText(message)
            except Exception:
                pass
        except Exception:
            pass

    def _queue_async_review_piece_reload(
        self,
        current_path=None,
        signature=None,
        autogen_signature=None,
        mt_signature=None,
        stats=None,
        changed_paths=None,
    ):
        try:
            if getattr(self, "_review_piece_reload_running", False):
                self._review_piece_reload_requested = True
                return True
            selected_path = os.path.abspath(current_path or self.current_path or "")
            try:
                row = self.piece_list.currentRow()
                if 0 <= row < len(self.pieces):
                    selected_path = os.path.abspath(self.pieces[row].get("path") or selected_path)
            except Exception:
                pass
            self._save_current_review_scroll()
            self._flush_target_edits()
            if selected_path and os.path.isfile(selected_path):
                self.current_path = selected_path
            changed_path_set = set()
            for path in changed_paths or []:
                try:
                    changed_path_set.add(os.path.normcase(os.path.abspath(path)))
                except Exception:
                    continue
            piece_snapshot = []
            for row, piece in enumerate(list(self.pieces or [])):
                try:
                    path = os.path.normcase(os.path.abspath(piece.get("path") or ""))
                except Exception:
                    path = ""
                piece_snapshot.append({
                    "row": row,
                    "path": piece.get("path") or "",
                    "path_norm": path,
                    "review_label": piece.get("review_label"),
                    "opf_position": piece.get("opf_position"),
                    "chapter_num": piece.get("chapter_num"),
                    "raw_chapter_num": piece.get(
                        "raw_chapter_num", piece.get("chapter_num")
                    ),
                    "display_chapter_num": piece.get("display_chapter_num"),
                    "output_name": piece.get("output_name"),
                })
            self._review_piece_reload_token = int(getattr(self, "_review_piece_reload_token", 0)) + 1
            token = self._review_piece_reload_token
            self._review_piece_reload_running = True
            self._review_piece_reload_requested = False
            self._refreshing_review_data = True
            try:
                self.save_status_label.setText("Refreshing SDLXLIFF entries...")
            except Exception:
                pass
            self._hide_generation_progress()

            def _worker():
                result = {
                    "pieces": [],
                    "current_path": selected_path,
                    "review_signature": signature,
                    "autogen_signature": autogen_signature,
                    "machine_translation_signature": mt_signature,
                    "stats": stats,
                    "partial": bool(changed_path_set),
                    "pieces_by_path": {},
                    "removed_paths": [],
                    "error": "",
                }
                try:
                    if changed_path_set:
                        progress_map = self._read_progress_metadata()
                        spine_positions = self._read_spine_positions(allow_deep_search=False)
                        snapshot_by_path = {
                            item.get("path_norm"): item
                            for item in piece_snapshot
                            if item.get("path_norm")
                        }
                        for path_norm in sorted(changed_path_set):
                            snapshot = snapshot_by_path.get(path_norm)
                            if not snapshot:
                                result["partial"] = False
                                result["pieces"] = self._load_pieces(stream_sidebar=False)
                                break
                            path = snapshot.get("path") or path_norm
                            if not os.path.isfile(path):
                                result["removed_paths"].append(path_norm)
                                continue
                            metadata = self._sidecar_metadata(
                                path,
                                int(snapshot.get("row") or 0),
                                progress_map,
                                spine_positions,
                            )
                            if snapshot.get("review_label"):
                                metadata["label"] = snapshot.get("review_label")
                            if snapshot.get("opf_position") is not None:
                                metadata["opf_position"] = snapshot.get("opf_position")
                            if snapshot.get("chapter_num") is not None:
                                metadata["chapter_num"] = snapshot.get("chapter_num")
                            if snapshot.get("raw_chapter_num") is not None:
                                metadata["raw_chapter_num"] = snapshot.get(
                                    "raw_chapter_num"
                                )
                            if snapshot.get("display_chapter_num") is not None:
                                metadata["display_chapter_num"] = snapshot.get(
                                    "display_chapter_num"
                                )
                            if snapshot.get("output_name"):
                                metadata["output_name"] = snapshot.get("output_name")
                            piece = self._build_piece(path, int(snapshot.get("row") or 0), metadata)
                            if self._review_piece_is_empty_sidecar(piece):
                                result["removed_paths"].append(path_norm)
                            else:
                                result["pieces_by_path"][path_norm] = piece
                    else:
                        result["pieces"] = self._load_pieces(stream_sidebar=False)
                except Exception as exc:
                    result["error"] = str(exc)
                self._review_piece_reload_finished.emit(token, result)

            threading.Thread(target=_worker, name="sdlxliff-review-piece-reload", daemon=True).start()
            return True
        except Exception as exc:
            self._review_piece_reload_running = False
            self._refreshing_review_data = False
            try:
                self.save_status_label.setText(f"SDLXLIFF refresh failed: {exc}")
            except Exception:
                pass
            return False

    def _apply_async_review_piece_reload(self, token, result):
        try:
            if int(token) != int(getattr(self, "_review_piece_reload_token", -1)):
                return
            if not isinstance(result, dict):
                result = {"pieces": [], "error": "Invalid SDLXLIFF reload result"}
            if result.get("error"):
                try:
                    self.save_status_label.setText(f"SDLXLIFF refresh failed: {result.get('error')}")
                except Exception:
                    pass
                return
            if result.get("partial"):
                pieces_by_path = result.get("pieces_by_path") if isinstance(result.get("pieces_by_path"), dict) else {}
                removed_paths = set(result.get("removed_paths") or [])
                path_to_row = {}
                for row, piece in enumerate(self.pieces or []):
                    try:
                        path_to_row[os.path.normcase(os.path.abspath(piece.get("path") or ""))] = row
                    except Exception:
                        continue
                changed_rows = []
                rows_to_remove = []
                for path_norm in removed_paths:
                    row = path_to_row.get(path_norm)
                    if row is not None:
                        rows_to_remove.append(row)
                for path_norm, piece in pieces_by_path.items():
                    row = path_to_row.get(path_norm)
                    if row is None:
                        continue
                    piece["index"] = row
                    self.pieces[row] = piece
                    self._invalidate_piece_page_for_refresh(row)
                    self._refresh_piece_list_item(row)
                    changed_rows.append(row)
                if rows_to_remove:
                    for row in sorted(rows_to_remove, reverse=True):
                        self._discard_piece_page(row)
                        try:
                            self.pieces.pop(row)
                            self.piece_list.takeItem(row)
                        except Exception:
                            pass
                    for row, piece in enumerate(self.pieces):
                        piece["index"] = row
                    current_path = os.path.abspath(result.get("current_path") or self.current_path or "")
                    self._populate_piece_list()
                    if current_path:
                        self._select_piece_for_path(current_path)
                self._last_review_signature = result.get("review_signature")
                self._last_machine_translation_signature = result.get("machine_translation_signature")
                self._last_autogen_signature = result.get("autogen_signature")
                summary = self._review_generation_summary(result.get("stats"))
                if summary:
                    try:
                        self.save_status_label.setText(summary)
                    except Exception:
                        pass
                elif changed_rows:
                    try:
                        self.save_status_label.setText(f"Refreshed {len(changed_rows)} SDLXLIFF entr{'y' if len(changed_rows) == 1 else 'ies'}")
                    except Exception:
                        pass
                current_row = self.piece_list.currentRow()
                if current_row in changed_rows and self.isVisible():
                    self._refresh_piece_header(current_row)
                    QTimer.singleShot(0, lambda row=current_row: self._render_piece(row, show_loading=False))
                return
            pieces = result.get("pieces") if isinstance(result.get("pieces"), list) else []
            selected_path = os.path.abspath(result.get("current_path") or self.current_path or "")
            if selected_path and os.path.isfile(selected_path):
                self.current_path = selected_path
            self._render_token += 1
            self._clear_cached_review_pages()
            self._streamed_piece_list_populated = False
            self.pieces = pieces
            self._review_data_loaded = True
            self._populate_piece_list()
            self._last_review_signature = result.get("review_signature")
            self._last_machine_translation_signature = result.get("machine_translation_signature")
            self._last_autogen_signature = result.get("autogen_signature")
            self._queue_review_data_preload(delay_ms=220)
            summary = self._review_generation_summary(result.get("stats"))
            if summary:
                try:
                    self.save_status_label.setText(summary)
                except Exception:
                    pass
            elif self.pieces:
                try:
                    self.save_status_label.setText(f"Refreshed {len(self.pieces)} SDLXLIFF entr{'y' if len(self.pieces) == 1 else 'ies'}")
                except Exception:
                    pass
            if self.pieces and self.isVisible():
                row = self.piece_list.currentRow()
                if row < 0:
                    row = self._initial_piece_row
                if not self._review_row_rendered_or_rendering(row):
                    QTimer.singleShot(0, lambda row=row: self._render_piece(row, show_loading=False))
            elif not self.pieces:
                self._show_review_empty_state("No SDLXLIFF review files found")
        except Exception:
            pass
        finally:
            self._refreshing_review_data = False
            self._review_piece_reload_running = False
            self._hide_generation_progress()
            self._queue_stop_refresh_button_animation(150)
            if getattr(self, "_review_piece_reload_requested", False):
                self._review_piece_reload_requested = False
                self._queue_review_refresh_scan(force=False, validate=True, current_path=self.current_path, delay_ms=350)

    def _apply_review_refresh_scan(self, token, result):
        defer_stop_refresh_animation = False
        try:
            if int(token) != int(getattr(self, "_review_refresh_scan_token", -1)):
                return
            if not isinstance(result, dict):
                return
            if result.get("error"):
                try:
                    self.save_status_label.setText(f"SDLXLIFF refresh failed: {result.get('error')}")
                except Exception:
                    pass
                return
            signature = result.get("review_signature")
            mt_signature = result.get("machine_translation_signature")
            autogen_signature = result.get("autogen_signature")
            stats = result.get("stats")

            # A background scan may start immediately before an editor save.
            # The save updates the active piece and advances both signatures
            # in place, but the scan's booleans were calculated against its
            # older snapshot. Do not let that stale result rebuild the page
            # and flash "Refreshed 1 SDLXLIFF entry" when its final state is
            # already exactly the state the editor has integrated.
            already_integrated = bool(
                not result.get("force")
                and not result.get("sidecars_generated")
                and signature == getattr(self, "_last_review_signature", None)
                and autogen_signature == getattr(self, "_last_autogen_signature", None)
            )
            if already_integrated:
                if (
                    result.get("machine_translation_changed")
                    and mt_signature != getattr(
                        self, "_last_machine_translation_signature", None
                    )
                ):
                    if self._tooltip_translation_running:
                        self._last_machine_translation_signature = mt_signature
                    else:
                        self._reload_machine_translation_previews(
                            signature=mt_signature
                        )
                return

            if stats:
                summary = self._review_generation_summary(stats)
                if summary:
                    try:
                        self.save_status_label.setText(summary)
                    except Exception:
                        pass
            if (
                result.get("force")
                or result.get("sidecar_changed")
                or result.get("autogen_changed")
                or result.get("sidecars_generated")
            ):
                preserve_generated_stream = bool(
                    result.get("sidecars_generated")
                    and (
                        getattr(self, "_generation_streaming_active", False)
                        or getattr(self, "_generation_stream_pending_pieces", None)
                        or getattr(self, "_generation_stream_preserve_after_finish", False)
                    )
                )
                if preserve_generated_stream:
                    self._generation_stream_preserve_after_finish = False
                    self._last_review_signature = signature
                    self._last_machine_translation_signature = mt_signature
                    self._last_autogen_signature = autogen_signature
                    if self.pieces:
                        self._review_data_loaded = True
                    return
                initial_load = not bool(getattr(self, "_review_data_loaded", False) and self.pieces)
                if initial_load:
                    try:
                        self.save_status_label.setText("Loading SDLXLIFF review entries...")
                    except Exception:
                        pass
                else:
                    # A settings-only change (e.g. the dedupe toggle) doesn't map
                    # to any changed sidecar path, so force a full reload (None)
                    # instead of the incremental path - otherwise it would
                    # short-circuit below and rebuild nothing.
                    force_full_reload = bool(
                        result.get("force")
                        or result.get("sidecar_path_set_changed")
                        or result.get("settings_changed")
                    )
                    changed_paths = None if force_full_reload else result.get("changed_sidecar_paths")
                    if changed_paths is not None:
                        changed_paths = self._refresh_visible_sidecar_paths(changed_paths)
                        if not changed_paths:
                            self._last_review_signature = signature
                            self._last_machine_translation_signature = mt_signature
                            self._last_autogen_signature = autogen_signature
                            return
                    defer_stop_refresh_animation = self._queue_async_review_piece_reload(
                        current_path=result.get("current_path") or self.current_path,
                        signature=signature,
                        autogen_signature=autogen_signature,
                        mt_signature=mt_signature,
                        stats=stats,
                        changed_paths=changed_paths,
                    )
                    if defer_stop_refresh_animation:
                        return
                self.refresh_review_data(
                    force=False,
                    current_path=result.get("current_path") or self.current_path,
                    signature=signature,
                    seamless=not initial_load,
                    skip_autogen=True,
                    autogen_signature=autogen_signature,
                    mt_signature=mt_signature,
                )
                if initial_load and not getattr(self, "_initial_sidecar_validation_queued", False):
                    # The initial scan skipped the expensive per-sidecar
                    # parse-validation to get entries on screen fast. Run it
                    # once now, in the background, a moment after the initial
                    # load settles - any invalid sidecars get regenerated and
                    # refreshed seamlessly.
                    self._initial_sidecar_validation_queued = True
                    QTimer.singleShot(
                        2500,
                        lambda: self._queue_review_refresh_scan(
                            validate=True,
                            current_path=self.current_path,
                            delay_ms=0,
                        ),
                    )
                if stats:
                    summary = self._review_generation_summary(stats)
                    if summary and not self.pieces:
                        try:
                            self.save_status_label.setText(summary)
                        except Exception:
                            pass
            elif result.get("machine_translation_changed"):
                if self._tooltip_translation_running:
                    self._last_machine_translation_signature = mt_signature
                else:
                    self._reload_machine_translation_previews(signature=mt_signature)
            else:
                self._last_review_signature = signature
                self._last_machine_translation_signature = mt_signature
                self._last_autogen_signature = autogen_signature
        except Exception:
            pass
        finally:
            self._sdlxliff_autogen_output_files = None
            self._review_refresh_scan_running = False
            self._last_review_scan_finished_at = time.monotonic()
            if not defer_stop_refresh_animation:
                self._queue_stop_refresh_button_animation(150)
            if getattr(self, "_review_refresh_scan_requested", False):
                force = bool(getattr(self, "_review_refresh_scan_force", False))
                validate = bool(getattr(self, "_review_refresh_scan_validate", False))
                current_path = getattr(self, "_review_refresh_scan_current_path", None) or self.current_path
                self._review_refresh_scan_requested = False
                self._queue_review_refresh_scan(force=force, validate=validate, current_path=current_path, delay_ms=350)

    def _refresh_visible_sidecar_paths(self, changed_paths, max_background_updates=24):
        paths = []
        for path in changed_paths or []:
            try:
                paths.append(os.path.normcase(os.path.abspath(path)))
            except Exception:
                continue
        if not paths:
            return []
        if len(paths) <= int(max_background_updates):
            return paths
        visible_paths = set()
        try:
            row = self.piece_list.currentRow()
            if 0 <= row < len(self.pieces):
                visible_paths.add(os.path.normcase(os.path.abspath(self.pieces[row].get("path") or "")))
        except Exception:
            pass
        try:
            if self.current_path:
                visible_paths.add(os.path.normcase(os.path.abspath(self.current_path)))
        except Exception:
            pass
        return [path for path in paths if path in visible_paths]

    def _start_review_auto_refresh(self):
        try:
            if self._auto_refresh_timer is not None:
                if not self._auto_refresh_timer.isActive():
                    self._auto_refresh_timer.start()
                return
            timer = QTimer(self)
            timer.setInterval(2000)
            timer.timeout.connect(self._silent_review_refresh)
            timer.start()
            self._auto_refresh_timer = timer
        except Exception:
            pass

    def _tick_refresh_button_animation(self):
        try:
            frames = ("⟳", "◴", "◷", "◶", "◵")
            self._refresh_button_frame = (int(self._refresh_button_frame or 0) + 1) % len(frames)
            if self.refresh_review_btn is not None:
                self.refresh_review_btn.setText(f"{frames[self._refresh_button_frame]} Refreshing")
        except Exception:
            pass

    def _start_refresh_button_animation(self):
        try:
            if self._refresh_button_stop_timer is not None and self._refresh_button_stop_timer.isActive():
                self._refresh_button_stop_timer.stop()
            if self._refresh_button_timer is None:
                timer = QTimer(self)
                timer.setInterval(90)
                timer.timeout.connect(self._tick_refresh_button_animation)
                self._refresh_button_timer = timer
            self._refresh_button_frame = -1
            self._tick_refresh_button_animation()
            if not self._refresh_button_timer.isActive():
                self._refresh_button_timer.start()
            if self.refresh_review_btn is not None:
                self.refresh_review_btn.setEnabled(False)
        except Exception:
            pass

    def _stop_refresh_button_animation(self):
        try:
            if self._refresh_button_timer is not None and self._refresh_button_timer.isActive():
                self._refresh_button_timer.stop()
            if self.refresh_review_btn is not None:
                self.refresh_review_btn.setEnabled(True)
                self.refresh_review_btn.setText(self.MANUAL_REFRESH_BUTTON_TEXT)
        except Exception:
            pass

    def _queue_stop_refresh_button_animation(self, delay_ms=350):
        try:
            if self._refresh_button_stop_timer is None:
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self._stop_refresh_button_animation)
                self._refresh_button_stop_timer = timer
            self._refresh_button_stop_timer.start(max(0, int(delay_ms)))
        except Exception:
            self._stop_refresh_button_animation()

    def _tick_flag_accuracy_button_animation(self):
        try:
            frames = ("🟣", "🟪", "💜", "🟪")
            self._flag_accuracy_button_frame = (int(self._flag_accuracy_button_frame or 0) + 1) % len(frames)
            if self.flag_accuracy_btn is not None:
                self.flag_accuracy_btn.setText(f"{frames[self._flag_accuracy_button_frame]} Flagging")
        except Exception:
            pass

    def _start_flag_accuracy_button_animation(self):
        try:
            if self._flag_accuracy_button_stop_timer is not None and self._flag_accuracy_button_stop_timer.isActive():
                self._flag_accuracy_button_stop_timer.stop()
            if self._flag_accuracy_button_timer is None:
                timer = QTimer(self)
                timer.setInterval(90)
                timer.timeout.connect(self._tick_flag_accuracy_button_animation)
                self._flag_accuracy_button_timer = timer
            self._flag_accuracy_button_frame = -1
            self._tick_flag_accuracy_button_animation()
            if not self._flag_accuracy_button_timer.isActive():
                self._flag_accuracy_button_timer.start()
            if self.flag_accuracy_btn is not None:
                self.flag_accuracy_btn.setEnabled(False)
        except Exception:
            pass

    def _stop_flag_accuracy_button_animation(self):
        try:
            if self._flag_accuracy_button_timer is not None and self._flag_accuracy_button_timer.isActive():
                self._flag_accuracy_button_timer.stop()
            if self.flag_accuracy_btn is not None:
                self.flag_accuracy_btn.setEnabled(True)
                self.flag_accuracy_btn.setText(self.FLAG_ACCURACY_BUTTON_TEXT)
        except Exception:
            pass

    def _queue_stop_flag_accuracy_button_animation(self, delay_ms=650):
        try:
            if self._flag_accuracy_button_stop_timer is None:
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self._stop_flag_accuracy_button_animation)
                self._flag_accuracy_button_stop_timer = timer
            self._flag_accuracy_button_stop_timer.start(max(0, int(delay_ms)))
        except Exception:
            self._stop_flag_accuracy_button_animation()

    def _reject_unavailable_notepad_mode(self):
        """Restore Compact immediately without triggering a page rebuild."""
        self._two_column_layout_enabled = True
        try:
            button = getattr(self, "two_column_layout_btn", None)
            if button is not None and not button.isChecked():
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)
        except Exception:
            pass
        self._update_review_layout_button()
        try:
            self.save_status_label.setText(
                "Notepad mode is unavailable in this Lite package because Qt WebEngine is not included; Compact mode remains active."
            )
        except Exception:
            pass

    def _set_review_two_column_layout(self, enabled):
        enabled = bool(enabled)
        if not enabled and not self._review_notepad_mode_is_available():
            self._reject_unavailable_notepad_mode()
            return
        try:
            if self._edit_save_timer.isActive():
                self._edit_save_timer.stop()
            self._flush_target_edits()
        except Exception:
            pass
        self._two_column_layout_enabled = enabled
        try:
            if getattr(self, "two_column_layout_btn", None) is not None and self.two_column_layout_btn.isChecked() != enabled:
                self.two_column_layout_btn.blockSignals(True)
                self.two_column_layout_btn.setChecked(enabled)
                self.two_column_layout_btn.blockSignals(False)
        except Exception:
            pass
        self._update_review_layout_button()
        self._persist_review_config_value(self.TWO_COLUMN_LAYOUT_CONFIG_KEY, enabled)
        try:
            # Freeze only the visible viewport for the transition. Keeping the
            # live Compact page here also kept its full document-height stack;
            # the new QWebEngineView then inherited that height and Qt tried to
            # allocate backing textures taller than the GPU's 16384px limit.
            transition_pixmap = self._capture_review_transition_snapshot()
            self._hold_review_page_transition(transition_pixmap)
            self._cancel_active_review_render()
            self._cancel_review_preload(discard_page=True)
            for row, page in list(self._piece_pages.items()):
                self._discard_piece_page(row, page)
            self._piece_pages.clear()
            self._piece_render_complete.clear()
            self._finish_seamless_review_swap(self.loading_page)
            self._reset_review_stack_for_mode_transition()
            for piece in self.pieces:
                if isinstance(piece, dict):
                    piece.pop("_render_model", None)
            self._review_data_preload_token = int(getattr(self, "_review_data_preload_token", 0)) + 1
            current_row = self.piece_list.currentRow() if getattr(self, "piece_list", None) is not None else -1
            if 0 <= current_row < len(self.pieces):
                QTimer.singleShot(0, lambda row=current_row: self._render_piece(row, show_loading=True))
        except Exception:
            pass

    def _update_machine_translation_button_tooltip(self):
        try:
            provider_label = self._machine_translation_provider_label()
            self.translate_tooltips_btn.setToolTip(
                f"Generate source-row machine translation previews. Provider: {provider_label}. Right-click to choose."
            )
        except Exception:
            pass

    def _prompt_secret_text(self, title, label, current=""):
        value, ok = QInputDialog.getText(
            self,
            title,
            label,
            QLineEdit.Password,
            str(current or ""),
        )
        if not ok:
            return None
        return str(value or "").strip()

    def _prompt_plain_text(self, title, label, current=""):
        value, ok = QInputDialog.getText(
            self,
            title,
            label,
            QLineEdit.Normal,
            str(current or ""),
        )
        if not ok:
            return None
        return str(value or "").strip()

    def _prompt_machine_translation_credentials(self, provider, force=False):
        provider = self._normalize_machine_translation_provider(provider)
        if provider == "deepl":
            current = self._machine_translation_config_value(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY)
            if current and not force:
                return True
            key = self._prompt_secret_text("DeepL API Key", "DeepL API key:", current)
            if not key:
                self.save_status_label.setText("DeepL requires an API key")
                return False
            self._persist_review_config_value(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY, key)
            return True
        if provider == "bing":
            current = self._machine_translation_config_value(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY)
            if not current or force:
                key = self._prompt_secret_text("Bing / Microsoft Translator API Key", "Microsoft Translator API key:", current)
                if not key:
                    self.save_status_label.setText("Bing requires a Microsoft Translator API key")
                    return False
                self._persist_review_config_value(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY, key)
            current_region = self._machine_translation_config_value(self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY)
            if force:
                region = self._prompt_plain_text("Bing / Microsoft Translator Region", "Azure region (optional):", current_region)
                if region is not None:
                    self._persist_review_config_value(self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY, region)
            return True
        if provider == "yandex":
            current_key = self._machine_translation_config_value(self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY)
            current_folder = self._machine_translation_config_value(self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY)
            if current_key and current_folder and not force:
                return True
            key = self._prompt_secret_text("Yandex Translate API Key", "Yandex Cloud API key:", current_key)
            if not key:
                self.save_status_label.setText("Yandex requires an API key")
                return False
            folder_id = self._prompt_plain_text("Yandex Translate Folder ID", "Yandex Cloud folder ID:", current_folder)
            if not folder_id:
                self.save_status_label.setText("Yandex requires a folder ID")
                return False
            self._persist_review_config_value(self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY, key)
            self._persist_review_config_value(self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY, folder_id)
            return True
        return True

    def _show_machine_translation_provider_menu(self, pos):
        try:
            current = self._machine_translation_provider()
            menu = QMenu(self)
            menu.setStyleSheet(
                "QMenu { padding: 4px 8px 4px 4px; }"
                "QMenu::item { padding: 6px 24px 6px 12px; }"
            )
            for provider, label in self.MACHINE_TRANSLATION_PROVIDER_LABELS.items():
                action = menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(provider == current)
                action.triggered.connect(lambda _checked=False, p=provider: self._set_machine_translation_provider(p))
            menu.addSeparator()
            deepl_action = menu.addAction("🔑 Configure DeepL API Key...")
            deepl_action.triggered.connect(lambda _checked=False: self._prompt_machine_translation_credentials("deepl", force=True))
            bing_action = menu.addAction("🔑 Configure Bing API Key...")
            bing_action.triggered.connect(lambda _checked=False: self._prompt_machine_translation_credentials("bing", force=True))
            yandex_action = menu.addAction("🔑 Configure Yandex API Key...")
            yandex_action.triggered.connect(lambda _checked=False: self._prompt_machine_translation_credentials("yandex", force=True))
            self._machine_translation_provider_menu = menu
            menu.aboutToHide.connect(lambda m=menu: self._clear_machine_translation_provider_menu(m))
            menu.popup(self.translate_tooltips_btn.mapToGlobal(pos))
        except Exception:
            pass

    def _clear_machine_translation_provider_menu(self, menu):
        try:
            if getattr(self, "_machine_translation_provider_menu", None) is menu:
                self._machine_translation_provider_menu = None
            menu.deleteLater()
        except Exception:
            pass

    def _show_flag_accuracy_context_menu(self, pos):
        try:
            current = self._machine_translation_inaccuracy_threshold()
            menu = QMenu(self)
            menu.setStyleSheet(
                "QMenu { padding: 4px 6px 4px 4px; }"
                "QMenu::item { padding: 6px 20px 6px 12px; }"
            )
            set_action = menu.addAction(f"🟣 Set Score Threshold... ({current:g})")
            reset_action = menu.addAction(f"↺ Reset Threshold ({self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD:g})")
            set_action.triggered.connect(self._prompt_machine_translation_threshold)
            reset_action.triggered.connect(self._reset_machine_translation_threshold)
            menu.popup(self.flag_accuracy_btn.mapToGlobal(pos))
        except Exception:
            pass

    def _prompt_machine_translation_threshold(self):
        current = self._machine_translation_inaccuracy_threshold()
        value, ok = QInputDialog.getDouble(
            self,
            "Flag Inaccurate Threshold",
            "Score threshold (lower flags more rows, higher flags fewer):",
            current,
            1.0,
            1000.0,
            1,
        )
        if not ok:
            return
        self._apply_machine_translation_threshold(value)

    def _manual_review_refresh(self):
        self._start_refresh_button_animation()
        self._queue_review_refresh_scan(force=True, validate=False, current_path=self.current_path, delay_ms=0)

    def _silent_review_refresh(self):
        try:
            if not self.isVisible() or self._refreshing_review_data:
                return
            if not self._review_data_loaded:
                return
            # Cooldown: skip if the last scan finished less than 4s ago
            _now = time.monotonic()
            _last = getattr(self, '_last_review_scan_finished_at', 0.0)
            if _now - _last < 4.0:
                return
            if getattr(self, '_review_refresh_scan_running', False):
                return
            if self._review_context_menu_is_open():
                return
            if self._active_render_timer is not None or self._active_render_page is not None:
                return
            if (
                self._edit_save_timer.isActive()
                or self._pending_target_edits
                or self._pending_notepad_edits
            ):
                return
            if self._tooltip_translation_running:
                return
            try:
                from PySide6.QtWidgets import QApplication
                focus = QApplication.focusWidget()
                if focus is not None and focus.window() is self and isinstance(focus, QPlainTextEdit):
                    return
                try:
                    row = self.piece_list.currentRow()
                    page = self._piece_pages.get(row)
                    browser = (
                        page.findChild(QWidget, "SdlReviewNotepadBrowser")
                        if page is not None else None
                    )
                    if browser is not None and (
                        browser.hasFocus()
                        or focus is browser
                        or (focus is not None and browser.isAncestorOf(focus))
                    ):
                        return
                except (AttributeError, RuntimeError):
                    pass
            except Exception:
                pass
            self._queue_review_refresh_scan(force=False, current_path=self.current_path, delay_ms=350)
        except Exception:
            pass

    def refresh_review_data(
        self,
        force=False,
        current_path=None,
        signature=None,
        seamless=False,
        skip_autogen=False,
        autogen_signature=None,
        mt_signature=None,
    ):
        if self._refreshing_review_data:
            return
        trace_started = time.perf_counter()
        try:
            phase_started = time.perf_counter()
            autogen_changed = False if skip_autogen else self._maybe_regenerate_review_sidecars(force=force)
            self._trace_review_perf(
                "refresh_review_data_autogen",
                phase_started,
                force=bool(autogen_changed or force),
                force_refresh=bool(force),
                changed=bool(autogen_changed),
            )
            if signature is None or autogen_changed or force:
                phase_started = time.perf_counter()
                signature = self._current_review_signature()
                self._trace_review_perf(
                    "refresh_review_data_signature",
                    phase_started,
                    force=bool(force or autogen_changed),
                )
                if not force and not autogen_changed and signature == self._last_review_signature:
                    self._trace_review_perf(
                        "refresh_review_data_no_change",
                        trace_started,
                        force=True,
                    )
                    return
            old_visible_page = None
            if seamless:
                try:
                    old_visible_page = self.rows_stack.currentWidget()
                    if old_visible_page is self.loading_page:
                        old_visible_page = None
                except Exception:
                    old_visible_page = None
            self._refreshing_review_data = True
            self._save_current_review_scroll()
            self._flush_target_edits()
            selected_path = os.path.abspath(current_path or self.current_path or "")
            try:
                row = self.piece_list.currentRow()
                if 0 <= row < len(self.pieces):
                    selected_path = os.path.abspath(self.pieces[row].get("path") or selected_path)
            except Exception:
                pass
            self.current_path = selected_path if selected_path and os.path.isfile(selected_path) else self.current_path
            self._render_token += 1
            if seamless and old_visible_page is not None:
                self._cancel_active_review_render()
                for _row, page in list(self._piece_pages.items()):
                    if page is old_visible_page:
                        continue
                    self._remove_review_page_widget(page)
                self._piece_pages.clear()
                self._piece_render_complete.clear()
                self._highlighted_status_frame = None
                self._status_jump_indices.clear()
                self._seamless_review_old_page = old_visible_page
            else:
                self._seamless_review_old_page = None
            self._clear_cached_review_pages()
            self._streamed_piece_list_populated = False
            phase_started = time.perf_counter()
            self.pieces = self._load_pieces(stream_sidebar=not seamless)
            self._trace_review_perf(
                "refresh_review_data_load_pieces",
                phase_started,
                force=True,
                pieces=len(self.pieces or []),
                seamless=bool(seamless),
            )
            self._review_data_loaded = True
            if not self._streamed_piece_list_populated:
                phase_started = time.perf_counter()
                self._populate_piece_list()
                self._trace_review_perf(
                    "refresh_review_data_populate_piece_list",
                    phase_started,
                    force=True,
                    pieces=len(self.pieces or []),
                )
            self._queue_review_data_preload(delay_ms=220)
            self._last_review_signature = signature if signature is not None else self._current_review_signature()
            try:
                self._last_machine_translation_signature = (
                    mt_signature if mt_signature is not None else self._current_machine_translation_signature()
                )
            except Exception:
                pass
            try:
                self._last_autogen_signature = (
                    autogen_signature if autogen_signature is not None else self._current_review_autogen_signature()
                )
            except Exception:
                pass
            if self.pieces and self.isVisible():
                show_loading = not (seamless and old_visible_page is not None)
                render_row = self.piece_list.currentRow()
                if render_row < 0 or render_row >= len(self.pieces):
                    render_row = self._initial_piece_row
                render_row = max(0, min(render_row, len(self.pieces) - 1))
                self._initial_piece_row = render_row
                if not self._review_row_rendered_or_rendering(render_row):
                    QTimer.singleShot(
                        0,
                        lambda row=render_row, show_loading=show_loading: self._render_piece(row, show_loading=show_loading),
                    )
            self._trace_review_perf(
                "refresh_review_data_done",
                trace_started,
                force=True,
                pieces=len(self.pieces or []),
                seamless=bool(seamless),
            )
        except Exception:
            self._trace_review_perf("refresh_review_data_failed", trace_started, force=True)
            pass
        finally:
            self._refreshing_review_data = False
            self._queue_stop_refresh_button_animation(150)

    def reopen_for_path(self, output_dir=None, current_path=None):
        output_changed = False
        if output_dir:
            try:
                old_output = os.path.normcase(os.path.abspath(self.output_dir or ""))
                new_output = os.path.normcase(os.path.abspath(output_dir))
                output_changed = old_output != new_output
            except Exception:
                output_changed = bool(output_dir != self.output_dir)
            self.output_dir = output_dir
        if current_path:
            self.current_path = os.path.abspath(current_path)
        self.show()
        self.raise_()
        self.activateWindow()
        self._start_review_auto_refresh()
        if (
            output_changed
            or not self._review_data_loaded
            or not self.pieces
            or (self.current_path and not os.path.isfile(self.current_path))
        ):
            self._initial_review_load_started = True
            self._queue_review_refresh_scan(
                force=False,
                current_path=self.current_path,
                delay_ms=25,
            )
            return
        if self.current_path:
            if self._select_piece_for_path(self.current_path):
                return
        try:
            row = self.piece_list.currentRow()
        except Exception:
            row = self._initial_piece_row
        if row < 0:
            row = self._initial_piece_row
        QTimer.singleShot(0, lambda row=row: self._render_piece(row))

    def _create_review_book_navigation(self):
        if len(self._book_entries) <= 1:
            return None

        nav = QWidget()
        nav.setObjectName("SdlReviewBookNav")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 4)
        nav_layout.setSpacing(6)

        button_style = (
            "QPushButton { background-color:#3a3a3a; color:white; font-weight:bold; "
            "font-size:13pt; border:1px solid #5a9fd4; border-radius:4px; padding:4px; }"
            "QPushButton:hover { background-color:#4a8fc4; }"
            "QPushButton:disabled { color:#666; background-color:#2a2a2a; }"
        )

        self._book_nav_prev = QPushButton("◀")
        self._book_nav_prev.setFixedWidth(46)
        self._book_nav_prev.setStyleSheet(button_style)

        self._book_nav_combo = QComboBox()
        self._book_nav_combo.setStyleSheet(
            "QComboBox { background-color:#3a3a3a; color:white; font-weight:bold; "
            "font-size:11pt; padding:6px 10px; border:1px solid #5a9fd4; border-radius:4px; }"
            "QComboBox::drop-down { border:none; }"
            "QComboBox QAbstractItemView { background-color:#2d2d2d; color:white; "
            "selection-background-color:#5a9fd4; }"
        )

        self._book_nav_counter = QLabel("1 / 1")
        self._book_nav_counter.setStyleSheet(f"color:{self.THEME['muted']}; font-size:10pt; font-weight:bold;")
        self._book_nav_counter.setFixedWidth(70)
        self._book_nav_counter.setAlignment(Qt.AlignCenter)

        self._book_nav_next = QPushButton("▶")
        self._book_nav_next.setFixedWidth(46)
        self._book_nav_next.setStyleSheet(button_style)

        for entry in self._book_entries:
            self._book_nav_combo.addItem(entry.get("label") or "SDLXLIFF")
        self._book_nav_combo.setCurrentIndex(self._book_index)

        nav_layout.addWidget(self._book_nav_prev)
        nav_layout.addWidget(self._book_nav_combo, 1)
        nav_layout.addWidget(self._book_nav_counter)
        nav_layout.addWidget(self._book_nav_next)

        self._book_nav_combo.currentIndexChanged.connect(self._switch_review_book)
        self._book_nav_prev.clicked.connect(lambda: self._switch_review_book(self._book_index - 1))
        self._book_nav_next.clicked.connect(lambda: self._switch_review_book(self._book_index + 1))
        self._update_review_book_nav()
        return nav

    def _update_review_book_nav(self):
        if not self._book_nav_combo:
            return
        total = len(self._book_entries)
        index = max(0, min(self._book_index, total - 1)) if total else 0
        self._book_nav_updating = True
        try:
            if self._book_nav_combo.currentIndex() != index:
                self._book_nav_combo.setCurrentIndex(index)
            if self._book_nav_prev:
                self._book_nav_prev.setEnabled(index > 0)
            if self._book_nav_next:
                self._book_nav_next.setEnabled(index < total - 1)
            if self._book_nav_counter:
                self._book_nav_counter.setText(f"{index + 1} / {total}")
        finally:
            self._book_nav_updating = False

    def _clear_cached_review_pages(self):
        self._review_data_preload_token += 1
        self._review_data_preload_requested = False
        self._cancel_active_review_render()
        self._cancel_review_preload(discard_page=True)
        for row, page in list(self._piece_pages.items()):
            self._remove_review_page_widget(page)
        self._piece_pages.clear()
        self._piece_render_complete.clear()
        self._piece_scroll_positions.clear()
        self._current_scroll_piece_row = None
        self._highlighted_status_frame = None
        self._status_jump_indices.clear()
        self._seamless_review_old_page = None

    def _switch_review_book(self, index):
        if self._book_nav_updating:
            return
        if index < 0 or index >= len(self._book_entries):
            return
        if index == self._book_index and self.pieces:
            self._update_review_book_nav()
            return

        self._save_current_review_scroll()
        self._render_token += 1
        self._clear_cached_review_pages()
        self._book_index = index
        entry = self._book_entries[index]
        self.output_dir = entry.get("output_dir") or self.output_dir
        self.current_path = entry.get("current_path") or ""
        self.pieces = []
        self._review_data_loaded = False
        self._update_review_book_nav()
        self._show_review_loading_page()
        self.piece_list.clear()
        self.header_label.setText("Loading SDLXLIFF review...")
        self._start_review_auto_refresh()
        self._queue_review_refresh_scan(force=False, current_path=self.current_path, delay_ms=25)

    def _apply_translator_theme(self, parent):
        base_style = ""
        try:
            if parent is not None and hasattr(parent, "styleSheet"):
                base_style = parent.styleSheet() or ""
        except Exception:
            base_style = ""
        if not base_style:
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                active = app.activeWindow() if app else None
                if active is not None and hasattr(active, "styleSheet"):
                    base_style = active.styleSheet() or ""
            except Exception:
                base_style = ""

        fallback = f"""
            QWidget {{
                background-color: {self.THEME['bg']};
                color: {self.THEME['text']};
            }}
            QLabel {{
                color: {self.THEME['text']};
                background-color: transparent;
            }}
            QPushButton {{
                background-color: #3d3d3d;
                color: {self.THEME['text']};
                border: 1px solid {self.THEME['border']};
                border-radius: 3px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #4d4d4d;
                border-color: {self.THEME['accent']};
            }}
        """
        local = f"""
            QDialog#SDLXLIFFReviewDialog {{
                background-color: {self.THEME['bg']};
                color: {self.THEME['text']};
            }}
            QWidget#SdlReviewDetail,
            QWidget#SdlReviewRows,
            QWidget#SdlReviewLoadingPage,
            QStackedWidget#SdlReviewRowsStack {{
                background-color: {self.THEME['bg']};
            }}
            QListWidget#SdlReviewPieceList {{
                background-color: {self.THEME['panel']};
                color: {self.THEME['text']};
                border: 1px solid {self.THEME['border']};
                border-radius: 3px;
                padding: 4px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 10pt;
            }}
            QListWidget#SdlReviewPieceList::item {{
                padding: 7px 8px;
                border-radius: 3px;
            }}
            QListWidget#SdlReviewPieceList::item:hover:!selected {{
                background-color: #334155;
                border: 1px solid #5a6f8c;
            }}
            QListWidget#SdlReviewPieceList::item:selected {{
                background-color: {self.THEME['accent']};
            }}
            QListWidget#SdlReviewPieceList::item:selected:hover {{
                background-color: #6cb4e8;
            }}
            QScrollArea#SdlReviewScroll {{
                border: 1px solid {self.THEME['border']};
                border-radius: 3px;
                background-color: {self.THEME['bg']};
            }}
            QScrollArea#SdlReviewScroll > QWidget > QWidget {{
                background-color: {self.THEME['bg']};
            }}
            QPlainTextEdit#SdlReviewTargetEdit {{
                background-color: {self.THEME['panel_alt']};
                color: {self.THEME['text']};
                border: 1px solid {self.THEME['border']};
                border-radius: 3px;
                padding: 4px;
                selection-background-color: {self.THEME['accent']};
            }}
            QPlainTextEdit#SdlReviewTargetEdit:focus {{
                border-color: {self.THEME['accent']};
            }}
        """
        self.setStyleSheet((base_style or fallback) + "\n" + local)

    def _apply_review_widget_background(self, widget, color=None):
        if widget is None:
            return
        background = QColor(color or self.THEME["bg"])
        try:
            widget.setAutoFillBackground(True)
            widget.setAttribute(Qt.WA_StyledBackground, True)
            palette = widget.palette()
            palette.setColor(QPalette.Window, background)
            widget.setPalette(palette)
            widget.setStyleSheet(f"background-color: {background.name()};")
        except Exception:
            pass

    def _create_review_rows_page(self):
        page = QWidget()
        page.setObjectName("SdlReviewRows")
        self._apply_review_widget_background(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4 if bool(getattr(self, "_two_column_layout_enabled", True)) else 0)
        return page, layout

    def _create_review_loading_page(self):
        loading_widget = QWidget()
        loading_widget.setObjectName("SdlReviewLoadingPage")
        self._apply_review_widget_background(loading_widget)
        loading_layout = QVBoxLayout(loading_widget)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        loading_layout.setSpacing(10)
        loading_layout.addStretch(1)

        try:
            try:
                from spinning import create_icon_label
            except Exception:
                from .spinning import create_icon_label
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            loading_icon = create_icon_label(52, base_dir)
            loading_icon.setFixedSize(52, 52)
            loading_icon.hide()
            loading_layout.addWidget(loading_icon, 0, Qt.AlignCenter)
            self._sdl_review_loading_icon = loading_icon
            pixmap = loading_icon.pixmap()
            if pixmap is not None and not pixmap.isNull():
                self._sdl_review_loading_original_pixmap = pixmap.copy()

            spin_timer = QTimer(self)
            spin_timer.timeout.connect(self._spin_review_loading_icon)
            spin_timer.start(45)
            QTimer.singleShot(0, self._spin_review_loading_icon)
            self._sdl_review_loading_icon_timer = spin_timer
        except Exception:
            pass

        loading_label = QLabel("Loading SDLXLIFF...")
        loading_label.setTextFormat(Qt.PlainText)
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("color: #94a3b8; font-size: 12pt; font-weight: bold; padding: 24px;")
        loading_label.hide()
        self.loading_label = loading_label
        loading_layout.addWidget(loading_label)
        loading_progress = QProgressBar()
        loading_progress.setTextVisible(True)
        loading_progress.setFixedHeight(18)
        loading_progress.setMinimumWidth(560)
        loading_progress.setMaximumWidth(720)
        loading_progress.setStyleSheet(
            "QProgressBar { background-color:#202936; color:#d7ecff; border:1px solid #4a5568; "
            "border-radius:5px; text-align:center; font-size:9pt; font-weight:bold; }"
            "QProgressBar::chunk { background-color:#5a9fd4; border-radius:4px; }"
        )
        loading_progress.hide()
        self.loading_progress_bar = loading_progress
        loading_layout.addWidget(loading_progress, 0, Qt.AlignCenter)
        loading_layout.addStretch(1)
        return loading_widget

    def _set_generation_progress(self, value=0, total=0, text=""):
        try:
            total = max(0, int(total or 0))
            value = max(0, int(value or 0))
        except Exception:
            total = 0
            value = 0
        if total <= 0:
            total = max(1, value, 1)
        value = min(value, total)
        progress_text = text or f"{value}/{total} SDLXLIFF"
        for bar in (getattr(self, "generation_progress_bar", None),):
            if bar is None:
                continue
            try:
                bar.setRange(0, total)
                bar.setValue(value)
                bar.setFormat(progress_text)
                bar.show()
            except RuntimeError:
                pass
            except Exception:
                pass

    def _set_loading_progress(self, value=0, total=0, text=""):
        try:
            total = max(0, int(total or 0))
            value = max(0, int(value or 0))
        except Exception:
            total = 0
            value = 0
        if total <= 0:
            total = max(1, value, 1)
        value = min(value, total)
        progress_text = text or f"{value}/{total} SDLXLIFF entries"
        try:
            status = getattr(self, "save_status_label", None)
            if status is not None:
                status.setText(progress_text)
        except RuntimeError:
            pass
        except Exception:
            pass
        bar = getattr(self, "generation_progress_bar", None)
        if bar is None:
            return
        try:
            bar.setRange(0, total)
            bar.setValue(value)
            bar.setFormat(progress_text)
            bar.show()
        except RuntimeError:
            pass
        except Exception:
            pass

    def _hide_generation_progress(self):
        for bar in (
            getattr(self, "generation_progress_bar", None),
            getattr(self, "loading_progress_bar", None),
        ):
            if bar is None:
                continue
            try:
                bar.hide()
            except RuntimeError:
                pass
            except Exception:
                pass

    def _hide_review_loading_page_widgets(self):
        for widget in (
            getattr(self, "_sdl_review_loading_icon", None),
            getattr(self, "loading_label", None),
            getattr(self, "loading_progress_bar", None),
        ):
            if widget is None:
                continue
            try:
                widget.hide()
            except RuntimeError:
                pass
            except Exception:
                pass

    def _spin_review_loading_icon(self):
        icon = getattr(self, "_sdl_review_loading_icon", None)
        original = getattr(self, "_sdl_review_loading_original_pixmap", None)
        if icon is None or original is None or original.isNull():
            return
        try:
            self._sdl_review_loading_angle = (self._sdl_review_loading_angle + 24) % 360
            transform = QTransform().rotate(self._sdl_review_loading_angle)
            rotated = original.transformed(transform, Qt.SmoothTransformation)
            if rotated.isNull():
                return
            scaled = rotated.scaled(
                icon.size().width(),
                icon.size().height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            icon.setPixmap(scaled)
            icon.update()
        except RuntimeError:
            timer = getattr(self, "_sdl_review_loading_icon_timer", None)
            if timer is not None:
                timer.stop()
        except Exception:
            pass

    def _legend_status_label(self, text, status):
        color = {
            "green": self.THEME["success"],
            "yellow": self.THEME["warning"],
            "purple": self.THEME["purple"],
            "red": self.THEME["danger"],
        }.get(status, self.THEME["muted"])
        label = QLabel(text)
        label.setTextFormat(Qt.PlainText)
        label.setCursor(Qt.PointingHandCursor)
        label.setToolTip(f"Jump to next {status} row")
        label.setStyleSheet(
            f"color: {color}; font-size: 9pt; font-weight: bold; "
            "background: transparent; text-decoration: underline;"
        )

        def _jump(event, target_status=status):
            self._jump_to_status(target_status)
            try:
                event.accept()
            except Exception:
                pass

        label.mousePressEvent = _jump
        return label

    def _jump_to_status(self, status):
        page = self.rows_widget
        if page is None or page is self.loading_page:
            return
        try:
            browser = page.findChild(QWidget, "SdlReviewNotepadBrowser")
            if browser is not None:
                self._jump_to_notepad_status(browser, status)
                return
            frames = [
                frame for frame in page.findChildren(QFrame, "SdlReviewRow")
                if frame.property("sdl_status") == status
            ]
            frames.sort(key=lambda frame: self._review_row_index_property(frame, default=0))
            if not frames:
                return
            try:
                piece_row = self.piece_list.currentRow()
            except Exception:
                piece_row = -1
            jump_key = (piece_row, status)
            next_index = (int(self._status_jump_indices.get(jump_key, -1)) + 1) % len(frames)
            self._status_jump_indices[jump_key] = next_index
            target = frames[next_index]
            self._highlight_review_row(target)
            self.scroll.ensureWidgetVisible(target, 0, 18)
            try:
                scrollbar = self.scroll.verticalScrollBar()
                scrollbar.setValue(max(0, min(scrollbar.maximum(), target.y() - 18)))
            except Exception:
                pass
            QTimer.singleShot(0, lambda target=target: self.scroll.ensureWidgetVisible(target, 0, 18))
        except Exception:
            pass

    def _jump_to_notepad_status(self, browser, status):
        """Cycle status legend navigation inside the browser-backed Notepad page."""
        try:
            piece_row = self.piece_list.currentRow()
        except Exception:
            piece_row = -1
        jump_key = (piece_row, str(status))
        requested_index = int(self._status_jump_indices.get(jump_key, -1)) + 1
        wanted_json = json.dumps(str(status))
        script = f"""
            (() => {{
                const wanted = {wanted_json};
                const marker = 'data-sdl-notepad-jump-highlight';
                const targets = Array.from(document.querySelectorAll(
                    '[data-sdl-notepad-status]'
                )).filter(element =>
                    element.getAttribute('data-sdl-notepad-status') === wanted
                );
                document.querySelectorAll('[' + marker + ']').forEach(
                    element => element.removeAttribute(marker)
                );
                if (!targets.length) return null;
                const index = {requested_index} % targets.length;
                const target = targets[index];
                target.setAttribute(marker, '1');
                target.scrollIntoView({{block: 'center', inline: 'nearest'}});
                window.clearTimeout(window.__sdlNotepadJumpHighlightTimer);
                window.__sdlNotepadJumpHighlightTimer = window.setTimeout(() => {{
                    if (target.isConnected) target.removeAttribute(marker);
                }}, 1400);
                return {{index, count: targets.length}};
            }})();
        """

        def _jumped(result):
            try:
                if isinstance(result, dict) and int(result.get("count", 0)) > 0:
                    self._status_jump_indices[jump_key] = int(result.get("index", 0))
            except Exception:
                pass

        try:
            browser.page().runJavaScript(script, _jumped)
        except RuntimeError:
            pass

    def _clear_review_row_highlight(self):
        frame = self._highlighted_status_frame
        self._highlighted_status_frame = None
        if frame is None:
            return
        try:
            base_style = frame.property("sdl_base_style")
            if base_style:
                frame.setStyleSheet(str(base_style))
        except RuntimeError:
            pass
        except Exception:
            pass

    def _highlight_review_row(self, frame):
        self._clear_review_row_highlight()
        if frame is None:
            return
        try:
            base_style = str(frame.property("sdl_base_style") or frame.styleSheet() or "")
            frame.setProperty("sdl_base_style", base_style)
            frame.setStyleSheet(
                base_style
                + f"\nQFrame#SdlReviewRow {{ border: 3px solid {self.THEME['accent']}; }}"
            )
            frame.raise_()
            frame.update()
            self._highlighted_status_frame = frame
        except Exception:
            pass

    def _apply_piece_list_item_style(self, item, piece):
        if item is None or not isinstance(piece, dict):
            return
        if piece.get("manual_untranslated"):
            color = QColor(self.THEME["muted"])
            color.setAlpha(115)
            item.setForeground(color)
        elif piece.get("mismatch"):
            item.setForeground(QColor(self.THEME["danger"]))
        elif piece.get("purple_count"):
            item.setForeground(QColor(self.THEME["purple"]))
        elif piece.get("yellow_count"):
            item.setForeground(QColor(self.THEME["warning"]))
        else:
            item.setForeground(QColor(self.THEME["success"]))

    def _piece_list_item_for_piece(self, piece, row):
        label = self._sidebar_label_for_piece(piece, row)
        output_name = self._output_name_for_piece(piece)
        item = QListWidgetItem(label)
        manual_hint = (
            "\nUntranslated manual sidecar; edit a target row to create its HTML output."
            if piece.get("manual_untranslated")
            else ""
        )
        item.setToolTip(
            f"{output_name}\nsource {piece['source_count']} -> output {piece['target_count']}\n{piece.get('path', '')}{manual_hint}"
        )
        item.setData(Qt.UserRole, row)
        self._apply_piece_list_item_style(item, piece)
        return item

    def _prepare_streaming_piece_list(self, work_items):
        trace_started = time.perf_counter()
        if not work_items:
            return False
        try:
            self.piece_list.currentRowChanged.disconnect(self._request_render_piece)
        except Exception:
            pass
        try:
            self.piece_list.clear()
            self.pieces = []
            self._streaming_piece_selected_path = (
                os.path.normcase(os.path.abspath(self.current_path)) if self.current_path else ""
            )
            self._streaming_piece_selected_row = 0
            self._streaming_piece_rendered_row = None
            self._streaming_piece_visible_count = 0
            self._streaming_piece_skipped = 0
            self._streaming_piece_last_pump = time.monotonic()
            self._set_loading_progress(0, len(work_items), f"0/{len(work_items)} SDLXLIFF entries")
        except Exception:
            return False
        try:
            self.piece_list.currentRowChanged.connect(self._request_render_piece)
        except Exception:
            pass
        self.piece_list.update()
        self._pump_review_loading_events(max_ms=5)
        self._trace_review_perf(
            "prepare_streaming_piece_list",
            trace_started,
            force=True,
            total=len(work_items),
        )
        return True

    def _stream_piece_list_item(self, original_index, piece):
        trace_started = time.perf_counter()
        try:
            total = int(getattr(self, "_streaming_piece_total", 0) or 0)
            if self._review_piece_is_empty_sidecar(piece):
                # Empty sidecars (e.g. the cover page) are never shown; drop them
                # from the denominator so the bar can reach 100% instead of
                # resting at e.g. 15/16 and looking stuck.
                self._streaming_piece_skipped = int(getattr(self, "_streaming_piece_skipped", 0)) + 1
                visible_count = int(getattr(self, "_streaming_piece_visible_count", 0) or 0)
                if total:
                    effective = max(visible_count, total - self._streaming_piece_skipped)
                    self._set_loading_progress(visible_count, effective, f"{visible_count}/{effective} SDLXLIFF entries")
                return
            visible_row = int(getattr(self, "_streaming_piece_visible_count", 0) or 0)
            piece = dict(piece)
            piece["index"] = visible_row
            self.pieces.append(piece)
            item = self._piece_list_item_for_piece(piece, visible_row)
            self.piece_list.addItem(item)
            self._streaming_piece_visible_count = visible_row + 1
            try:
                if total:
                    effective = max(visible_row + 1, total - int(getattr(self, "_streaming_piece_skipped", 0)))
                    self.save_status_label.setText(f"Loaded SDLXLIFF entry {visible_row + 1}/{effective}")
                    self._set_loading_progress(
                        visible_row + 1,
                        effective,
                        f"{visible_row + 1}/{effective} SDLXLIFF entries",
                    )
            except Exception:
                pass
            try:
                piece_norm = os.path.normcase(os.path.abspath(piece.get("path") or ""))
                selected_match = bool(self._streaming_piece_selected_path and piece_norm == self._streaming_piece_selected_path)
                if selected_match:
                    self._streaming_piece_selected_row = visible_row
                if selected_match or self.piece_list.currentRow() < 0:
                    previous_block = self.piece_list.blockSignals(True)
                    self.piece_list.setCurrentRow(visible_row)
                    self.piece_list.blockSignals(previous_block)
                    self._initial_piece_row = visible_row
                    self._streaming_piece_rendered_row = visible_row
                    QTimer.singleShot(0, lambda row=visible_row: self._render_piece(row, show_loading=False))
            except Exception:
                pass
            self.piece_list.update()
            last_pump = float(getattr(self, "_streaming_piece_last_pump", 0.0) or 0.0)
            if time.monotonic() - last_pump >= 0.012:
                self._streaming_piece_last_pump = time.monotonic()
                self._pump_review_loading_events(max_ms=2)
            total = int(getattr(self, "_streaming_piece_total", 0) or 0)
            visible_count = int(getattr(self, "_streaming_piece_visible_count", 0) or 0)
            self._trace_review_perf(
                "stream_piece_list_item",
                trace_started,
                force=bool(visible_count in {1, total} or (visible_count and visible_count % 50 == 0)),
                original_index=original_index,
                visible=visible_count,
                total=total or None,
                output=piece.get("output_name") or piece.get("name"),
            )
        except Exception:
            pass

    def _finish_streaming_piece_list(self):
        trace_started = time.perf_counter()
        try:
            self.piece_list.currentRowChanged.disconnect(self._request_render_piece)
        except Exception:
            pass
        try:
            self.piece_list.currentRowChanged.connect(self._request_render_piece)
        except Exception:
            pass
        try:
            selected_row = int(getattr(self, "_streaming_piece_selected_row", 0) or 0)
            if self.piece_list.count() > 0:
                current_row = self.piece_list.currentRow()
                if 0 <= current_row < self.piece_list.count():
                    selected_row = current_row
                else:
                    selected_row = max(0, min(selected_row, self.piece_list.count() - 1))
                    previous_block = self.piece_list.blockSignals(True)
                    self.piece_list.setCurrentRow(selected_row)
                    self.piece_list.blockSignals(previous_block)
                self._initial_piece_row = selected_row
                if (
                    getattr(self, "_streaming_piece_rendered_row", None) is None
                    and 0 <= selected_row < len(self.pieces)
                    and not self._review_row_rendered_or_rendering(selected_row)
                ):
                    self._streaming_piece_rendered_row = selected_row
                    QTimer.singleShot(0, lambda row=selected_row: self._render_piece(row, show_loading=False))
            else:
                self.header_label.setText("No SDLXLIFF review files found")
                self._show_review_empty_state("No SDLXLIFF review files found")
            self._streamed_piece_list_populated = True
            self.piece_list.update()
            self._trace_review_perf(
                "finish_streaming_piece_list",
                trace_started,
                force=True,
                pieces=len(self.pieces or []),
                selected=self.piece_list.currentRow(),
            )
            return True
        except Exception:
            self._trace_review_perf(
                "finish_streaming_piece_list_failed",
                trace_started,
                force=True,
                pieces=len(getattr(self, "pieces", []) or []),
            )
            return False

    def _populate_piece_list(self):
        try:
            self.piece_list.currentRowChanged.disconnect(self._request_render_piece)
        except Exception:
            pass
        self.piece_list.clear()
        selected_row = 0
        current_norm = os.path.normcase(os.path.abspath(self.current_path)) if self.current_path else ""
        for row, piece in enumerate(self.pieces):
            item = self._piece_list_item_for_piece(piece, row)
            self.piece_list.addItem(item)
            if current_norm and os.path.normcase(os.path.abspath(piece["path"])) == current_norm:
                selected_row = row

        self.piece_list.currentRowChanged.connect(self._request_render_piece)
        if self.pieces:
            previous_block = self.piece_list.blockSignals(True)
            self.piece_list.setCurrentRow(selected_row)
            self.piece_list.blockSignals(previous_block)
            self._initial_piece_row = selected_row
        else:
            self.header_label.setText("No SDLXLIFF review files found")
            self._show_review_empty_state("No SDLXLIFF review files found")

    def _refresh_piece_list_item(self, piece_index):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            piece = self.pieces[piece_index]
            item = self.piece_list.item(piece_index)
            if item is None:
                return
            output_name = self._output_name_for_piece(piece)
            item.setText(self._sidebar_label_for_piece(piece, piece_index))
            item.setToolTip(
                f"{output_name}\nsource {piece['source_count']} -> output {piece['target_count']}\n{piece.get('path', '')}"
            )
            self._apply_piece_list_item_style(item, piece)
        except Exception:
            pass

    def _refresh_piece_header(self, piece_index):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            if self.piece_list.currentRow() != piece_index:
                return
            self.header_label.setText(self._piece_header_text(piece_index))
        except Exception:
            pass

    def _select_piece_for_path(self, path):
        row = self._row_for_piece_path(path)
        if row is None:
            return False
        try:
            self.current_path = os.path.abspath(path)
            self._initial_piece_row = row
            if 0 <= self._book_index < len(self._book_entries):
                self._book_entries[self._book_index]["current_path"] = self.current_path
        except Exception:
            pass
        try:
            current_row = self.piece_list.currentRow()
        except Exception:
            current_row = -1
        if current_row != row:
            self.piece_list.setCurrentRow(row)
        else:
            QTimer.singleShot(0, lambda row=row: self._render_piece(row))
        return True

    def _clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _clear_rows(self, layout=None):
        self._clear_layout(layout or self.rows_layout)

    def _show_review_loading_page(self):
        try:
            status = getattr(self, "save_status_label", None)
            if status is not None:
                status.setText("Loading SDLXLIFF...")
            self._hide_review_loading_page_widgets()
            if self.rows_stack.currentWidget() is not self.loading_page and not self.pieces:
                self.rows_widget = self.loading_page
                self.rows_layout = self.loading_page.layout()
                self._current_scroll_piece_row = None
                self.rows_stack.setCurrentWidget(self.loading_page)
                self._sync_review_scroll_range(self.loading_page)
        except Exception:
            pass

    def _show_review_empty_state(self, message="No SDLXLIFF review files found"):
        try:
            self.rows_widget = self.loading_page
            self.rows_layout = self.loading_page.layout()
            self._current_scroll_piece_row = None
            if self.rows_stack.currentWidget() is not self.loading_page:
                self.rows_stack.setCurrentWidget(self.loading_page)
            self._sync_review_scroll_range(self.loading_page)
            try:
                icon = getattr(self, "_sdl_review_loading_icon", None)
                if icon is not None:
                    icon.show()
            except Exception:
                pass
            try:
                self.loading_label.setText(str(message or "No SDLXLIFF review files found"))
                self.loading_label.show()
            except Exception:
                pass
            try:
                progress = getattr(self, "loading_progress_bar", None)
                if progress is not None:
                    progress.hide()
            except Exception:
                pass
        except Exception:
            pass

    def _pump_review_loading_events(self, max_ms=8):
        if getattr(self, "_review_event_pump_active", False):
            return
        try:
            self._review_event_pump_active = True
            self._spin_review_loading_icon()
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents(QEventLoop.AllEvents, max(1, int(max_ms)))
        except Exception:
            pass
        finally:
            self._review_event_pump_active = False

    def _remove_review_page_widget(self, page):
        if page is None or page is self.loading_page:
            return
        try:
            self.rows_stack.removeWidget(page)
        except Exception:
            pass
        try:
            page.hide()
            page.setParent(None)
            page.deleteLater()
        except Exception:
            pass

    def _finish_seamless_review_swap(self, new_page):
        old_page = self._seamless_review_old_page
        self._seamless_review_old_page = None
        if old_page is None or old_page is new_page:
            return
        self._remove_review_page_widget(old_page)

    def _capture_review_transition_snapshot(self):
        """Grab what the user currently sees so the next page can fade in over it."""
        try:
            if not self.isVisible():
                return None
            viewport = self.scroll.viewport()
            if viewport is None or viewport.width() < 2 or viewport.height() < 2:
                return None
            return viewport.grab()
        except Exception:
            return None

    def _cancel_review_page_transition(self):
        anim = getattr(self, "_review_transition_anim", None)
        self._review_transition_anim = None
        self._review_transition_held = False
        self._pending_review_transition_pixmap = None
        if anim is not None:
            try:
                anim.stop()
            except Exception:
                pass
        overlay = getattr(self, "_review_transition_overlay", None)
        self._review_transition_overlay = None
        if overlay is not None:
            try:
                overlay.hide()
                overlay.setParent(None)
                overlay.deleteLater()
            except Exception:
                pass

    def _create_review_transition_overlay(self, pixmap):
        if pixmap is None or pixmap.isNull():
            return None, None
        viewport = self.scroll.viewport()
        if viewport is None:
            return None, None
        overlay = QLabel(viewport)
        overlay.setObjectName("SdlReviewTransitionOverlay")
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.setPixmap(pixmap)
        overlay.setScaledContents(False)
        overlay.setGeometry(0, 0, viewport.width(), viewport.height())
        effect = QGraphicsOpacityEffect(overlay)
        effect.setOpacity(1.0)
        overlay.setGraphicsEffect(effect)
        overlay.show()
        overlay.raise_()
        self._review_transition_overlay = overlay
        return overlay, effect

    def _hold_review_page_transition(self, pixmap):
        """Hold a viewport-sized snapshot while a replacement mode is built."""
        try:
            self._cancel_review_page_transition()
            overlay, _effect = self._create_review_transition_overlay(pixmap)
            if overlay is None:
                return False
            self._review_transition_held = True
            self._pending_review_transition_pixmap = pixmap
            # Paint the snapshot before the zero-delay rebuild callback runs.
            overlay.repaint()
            return True
        except Exception:
            self._cancel_review_page_transition()
            return False

    def _take_review_transition_snapshot(self, show_loading=True):
        if not show_loading:
            return None
        pixmap = getattr(self, "_pending_review_transition_pixmap", None)
        self._pending_review_transition_pixmap = None
        try:
            if pixmap is not None and not pixmap.isNull():
                return pixmap
        except Exception:
            pass
        return self._capture_review_transition_snapshot()

    def _reset_review_stack_for_mode_transition(self):
        """Release the old document height before creating a browser-backed page."""
        try:
            viewport = self.scroll.viewport()
            viewport_width = max(1, int(viewport.width()))
            viewport_height = max(1, int(viewport.height()))
            self.rows_stack.setCurrentWidget(self.loading_page)
            self.rows_widget = self.loading_page
            self.rows_layout = self.loading_page.layout()
            for widget in (self.loading_page, self.rows_stack):
                widget.setMinimumHeight(viewport_height)
                widget.setMaximumHeight(viewport_height)
                widget.resize(viewport_width, viewport_height)
                widget.updateGeometry()
            overlay = getattr(self, "_review_transition_overlay", None)
            if overlay is not None:
                overlay.setGeometry(0, 0, viewport_width, viewport_height)
                overlay.raise_()
        except Exception:
            pass

    def _start_review_page_transition(self, pixmap, duration_ms=140):
        """Crossfade entry switches: fade a static snapshot of the old content
        out over the new page. Only the snapshot is animated, so the live rows
        underneath render at full speed."""
        try:
            if pixmap is None or pixmap.isNull():
                self._cancel_review_page_transition()
                return
            overlay = getattr(self, "_review_transition_overlay", None)
            held = bool(getattr(self, "_review_transition_held", False))
            effect = overlay.graphicsEffect() if held and overlay is not None else None
            if not held or overlay is None or not isinstance(effect, QGraphicsOpacityEffect):
                self._cancel_review_page_transition()
                overlay, effect = self._create_review_transition_overlay(pixmap)
                if overlay is None or effect is None:
                    return
            else:
                viewport = self.scroll.viewport()
                overlay.setGeometry(0, 0, viewport.width(), viewport.height())
                effect.setOpacity(1.0)
                overlay.show()
                overlay.raise_()
            self._review_transition_held = False
            anim = QPropertyAnimation(effect, b"opacity", overlay)
            anim.setDuration(max(35, int(duration_ms)))
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._review_transition_overlay = overlay
            self._review_transition_anim = anim

            def _cleanup():
                if getattr(self, "_review_transition_overlay", None) is overlay:
                    self._review_transition_overlay = None
                    self._review_transition_anim = None
                    self._review_transition_held = False
                try:
                    overlay.hide()
                    overlay.setParent(None)
                    overlay.deleteLater()
                except Exception:
                    pass

            anim.finished.connect(_cleanup)
            anim.start()
        except Exception:
            self._cancel_review_page_transition()

    def _discard_piece_page(self, row, page=None):
        if row == self._preload_render_row:
            self._cancel_review_preload(discard_page=False)
        page = page or self._piece_pages.get(row)
        if page is None:
            return
        try:
            if self.rows_stack.currentWidget() is page:
                self._show_review_loading_page()
            self.rows_stack.removeWidget(page)
        except Exception:
            pass
        try:
            if self._piece_pages.get(row) is page:
                self._piece_pages.pop(row, None)
            self._piece_render_complete.discard(row)
        except Exception:
            pass
        try:
            page.hide()
            page.setParent(None)
            page.deleteLater()
        except Exception:
            pass

    def _cancel_active_review_render(self, defer_visible_discard=False):
        timer = self._active_render_timer
        row = self._active_render_row
        page = self._active_render_page
        self._active_render_timer = None
        self._active_render_row = None
        self._active_render_page = None
        if timer is not None:
            try:
                timer.stop()
                timer.deleteLater()
            except Exception:
                pass
        if row is not None and row not in self._piece_render_complete:
            page_is_current = False
            if defer_visible_discard and page is not None:
                try:
                    page_is_current = self.rows_stack.currentWidget() is page
                except Exception:
                    page_is_current = False
            if page_is_current:
                # Keep the partially rendered page on screen until the next
                # page's first batch is ready, then swap straight to it.
                # Discarding it here would flash the loading page on every
                # entry switch even though streaming makes it unnecessary.
                try:
                    if self._piece_pages.get(row) is page:
                        self._piece_pages.pop(row, None)
                    self._piece_render_complete.discard(row)
                except Exception:
                    pass
                previous = self._seamless_review_old_page
                if previous is not None and previous is not page:
                    self._remove_review_page_widget(previous)
                self._seamless_review_old_page = page
            else:
                self._discard_piece_page(row, page)

    def _cancel_review_preload(self, discard_page=True):
        timer = self._preload_render_timer
        row = self._preload_render_row
        page = self._preload_render_page
        self._preload_render_timer = None
        self._preload_start_queued = False
        self._preload_render_queue = []
        self._preload_render_row = None
        self._preload_render_page = None
        self._preload_render_state = None
        if timer is not None:
            try:
                timer.stop()
                timer.deleteLater()
            except Exception:
                pass
        if discard_page and row is not None and row not in self._piece_render_complete and page is not None:
            try:
                if self._piece_pages.get(row) is page:
                    self._piece_pages.pop(row, None)
                self._piece_render_complete.discard(row)
            except Exception:
                pass
            self._remove_review_page_widget(page)

    def _pause_review_preload_for_context_menu(self):
        timer = self._preload_render_timer
        self._preload_render_timer = None
        if timer is not None:
            try:
                timer.stop()
                timer.deleteLater()
            except Exception:
                pass

    def _resume_review_background_after_context_menu(self):
        try:
            if self._review_dirty_preview_refresh_queued:
                QTimer.singleShot(0, self._refresh_current_visible_dirty_source_previews)
            if self._preload_render_row is not None and self._preload_render_state is not None:
                if self._preload_render_timer is None:
                    timer = QTimer(self)
                    timer.setSingleShot(True)
                    timer.timeout.connect(self._run_review_preload_batch)
                    self._preload_render_timer = timer
                    timer.start(self.REVIEW_PRELOAD_STEP_MS)
                return
            current_row = self._displayed_piece_row()
            if 0 <= current_row < len(self.pieces):
                QTimer.singleShot(90, lambda row=current_row: self._queue_review_page_preloads(row))
        except Exception:
            pass

    def _invalidate_piece_page_for_refresh(self, row):
        if row == self._preload_render_row:
            self._cancel_review_preload(discard_page=False)
        try:
            if 0 <= row < len(self.pieces):
                self.pieces[row].pop("_render_model", None)
        except Exception:
            pass
        try:
            page = self._piece_pages.pop(row, None)
            self._piece_render_complete.discard(row)
        except Exception:
            page = None
        if page is None:
            return
        try:
            if self.rows_stack.currentWidget() is page:
                self._seamless_review_old_page = page
                return
        except Exception:
            pass
        self._remove_review_page_widget(page)

    def _set_review_context_menu_open(self, open_):
        self._review_context_menu_open = bool(open_)
        if self._review_context_menu_open:
            self._pause_review_preload_for_context_menu()
        else:
            self._resume_review_background_after_context_menu()

    def _review_render_viewport_width(self):
        try:
            return max(700, int(self.scroll.viewport().width()))
        except Exception:
            return 1200

    def _review_piece_render_model(self, piece):
        rows = self._review_rows_for_current_layout(piece)
        viewport_width = self._review_render_viewport_width()
        two_column_layout = bool(getattr(self, "_two_column_layout_enabled", True))
        model = piece.get("_render_model") if isinstance(piece, dict) else None
        if (
            isinstance(model, dict)
            and int(model.get("row_count", -1)) == len(rows)
            and int(model.get("viewport_width", -1)) == viewport_width
            and bool(model.get("two_column_layout", model.get("one_column_layout", model.get("one_row_layout", False)))) == two_column_layout
        ):
            return model
        model = self._build_review_piece_render_model_from_rows(
            self._piece_render_snapshot(piece, rows=rows),
            viewport_width,
            two_column_layout=two_column_layout,
        )
        piece["_render_model"] = model
        return model

    def _queue_review_data_preload(self, delay_ms=220):
        try:
            if not getattr(self, "_review_data_loaded", False) or not self.pieces:
                return
            self._review_data_preload_requested = True
            if getattr(self, "_review_data_preload_running", False) or getattr(self, "_review_data_preload_queued", False):
                return
            self._review_data_preload_queued = True

            def _run():
                self._review_data_preload_queued = False
                if not getattr(self, "_review_data_preload_requested", False):
                    return
                self._start_review_data_preload()

            QTimer.singleShot(max(0, int(delay_ms or 0)), _run)
        except Exception:
            self._review_data_preload_queued = False

    def _start_review_data_preload(self):
        if not getattr(self, "_review_data_loaded", False) or not self.pieces:
            return
        if not bool(getattr(self, "_two_column_layout_enabled", True)):
            # Notepad rows depend on the parsed DOM, not only the text-unit
            # snapshots used by this background size-model preloader.
            return
        if getattr(self, "_review_data_preload_running", False):
            self._review_data_preload_requested = True
            return
        self._review_data_preload_queued = False
        self._review_data_preload_token = int(getattr(self, "_review_data_preload_token", 0)) + 1
        token = self._review_data_preload_token
        viewport_width = self._review_render_viewport_width()
        two_column_layout = bool(getattr(self, "_two_column_layout_enabled", True))
        pieces_for_preload = [
            (piece_index, piece.get("rows") or [])
            for piece_index, piece in enumerate(self.pieces)
            if isinstance(piece, dict) and not piece.get("error")
        ]
        if not pieces_for_preload:
            return
        self._review_data_preload_running = True
        self._review_data_preload_requested = False
        cls = type(self)

        def _worker():
            models = {}
            try:
                for ordinal, (piece_index, rows) in enumerate(pieces_for_preload, start=1):
                    row_snapshot = [cls._review_row_snapshot(row_data) for row_data in rows]
                    models[piece_index] = cls._build_review_piece_render_model_from_rows(
                        row_snapshot,
                        viewport_width,
                        two_column_layout=two_column_layout,
                    )
                    if ordinal % 50 == 0:
                        time.sleep(0.001)
            except Exception:
                models = {}
            self._review_data_preload_finished.emit(token, models)

        threading.Thread(target=_worker, name="sdlxliff-review-data-preload", daemon=True).start()

    def _apply_review_data_preload(self, token, models):
        self._review_data_preload_running = False
        try:
            if int(token) != int(getattr(self, "_review_data_preload_token", -1)):
                if getattr(self, "_review_data_preload_requested", False):
                    self._review_data_preload_requested = False
                    self._queue_review_data_preload(delay_ms=250)
                return
            if not isinstance(models, dict):
                return
            for piece_index, model in models.items():
                try:
                    piece_index = int(piece_index)
                except Exception:
                    continue
                if 0 <= piece_index < len(self.pieces) and isinstance(model, dict):
                    rows = self.pieces[piece_index].get("rows") or []
                    if (
                        int(model.get("row_count", -1)) == len(rows)
                        and bool(model.get("two_column_layout", model.get("one_column_layout", model.get("one_row_layout", False)))) == bool(getattr(self, "_two_column_layout_enabled", True))
                    ):
                        self.pieces[piece_index]["_render_model"] = model
        finally:
            if getattr(self, "_review_data_preload_requested", False):
                self._queue_review_data_preload(delay_ms=250)

    def _queue_review_page_preloads(self, current_row):
        try:
            if not self.isVisible() or not self._review_data_loaded:
                return
            if not bool(getattr(self, "_two_column_layout_enabled", True)):
                return
            if self._review_context_menu_is_open():
                return
            self._queue_review_page_cache_trim(current_row)
            if self._preload_render_timer is not None or self._preload_render_row is not None:
                return
            if self._preload_start_queued:
                return
            self._preload_render_queue = self._review_preload_order(current_row)
            if self._preload_render_queue:
                self._preload_start_queued = True
                delay_ms = self.REVIEW_PRELOAD_IDLE_MS if self._review_selection_recently_changed() else 150
                QTimer.singleShot(delay_ms, self._start_next_review_preload)
        except Exception:
            pass

    def _start_next_review_preload(self):
        self._preload_start_queued = False
        try:
            if not self.isVisible() or not self._review_data_loaded:
                return
            if self._review_context_menu_is_open():
                self._queue_review_page_preloads(self._displayed_piece_row())
                return
            if self._review_selection_recently_changed():
                self._queue_review_page_preloads(self._displayed_piece_row())
                return
            if self._active_render_timer is not None or self._active_render_page is not None:
                self._queue_review_page_preloads(self._displayed_piece_row())
                return
            try:
                current_row = self.piece_list.currentRow()
            except Exception:
                current_row = -1
            while self._preload_render_queue:
                row = self._preload_render_queue.pop(0)
                if row == current_row or row in self._piece_render_complete or row in self._piece_pages:
                    continue
                if 0 <= row < len(self.pieces):
                    self._start_review_preload_row(row)
                    return
        except Exception:
            self._cancel_review_preload(discard_page=True)

    def _start_review_preload_row(self, row):
        if row < 0 or row >= len(self.pieces):
            return
        if self._review_selection_recently_changed():
            self._queue_review_page_preloads(self._displayed_piece_row())
            return
        piece = self.pieces[row]
        page, layout = self._create_review_rows_page()
        self._piece_pages[row] = page
        self._piece_render_complete.discard(row)
        self.rows_stack.addWidget(page)

        if piece.get("error"):
            error = QLabel(f"Could not parse SDLXLIFF:\n{piece['error']}")
            error.setTextFormat(Qt.PlainText)
            error.setStyleSheet(f"color: {self.THEME['danger']}; font-size: 11pt; padding: 12px;")
            layout.addWidget(error)
            layout.addStretch(1)
            self._piece_render_complete.add(row)
            QTimer.singleShot(60, self._start_next_review_preload)
            return

        rows = self._review_rows_for_current_layout(piece)
        if not rows:
            empty = QLabel("No p/h1-h6 text units found in this sidecar.")
            empty.setTextFormat(Qt.PlainText)
            empty.setStyleSheet(f"color: {self.THEME['muted']}; padding: 12px;")
            layout.addWidget(empty)
            layout.addStretch(1)
            self._piece_render_complete.add(row)
            QTimer.singleShot(60, self._start_next_review_preload)
            return

        render_model = self._review_piece_render_model(piece)
        max_len = int(render_model.get("max_len", 1))
        self._preload_render_row = row
        self._preload_render_page = page
        self._preload_render_state = {
            "idx": 0,
            "layout": layout,
            "rows": rows,
            "row_models": render_model.get("rows") or [],
            "max_len": max_len,
            "colors": self._review_status_colors(),
        }

        self._run_review_preload_batch()

    def _run_review_preload_batch(self):
        current_timer = self._preload_render_timer
        self._preload_render_timer = None
        if current_timer is not None:
            try:
                current_timer.deleteLater()
            except Exception:
                pass
        row = self._preload_render_row
        page = self._preload_render_page
        state = self._preload_render_state
        if row is None or page is None or not isinstance(state, dict):
            self._cancel_review_preload(discard_page=True)
            return
        try:
            if self._review_context_menu_is_open():
                self._preload_render_timer = QTimer(self)
                self._preload_render_timer.setSingleShot(True)
                self._preload_render_timer.timeout.connect(self._run_review_preload_batch)
                self._preload_render_timer.start(max(220, self.REVIEW_PRELOAD_IDLE_MS))
                return
            if self._review_selection_recently_changed():
                self._preload_render_timer = QTimer(self)
                self._preload_render_timer.setSingleShot(True)
                self._preload_render_timer.timeout.connect(self._run_review_preload_batch)
                self._preload_render_timer.start(self.REVIEW_PRELOAD_IDLE_MS)
                return
            if self._review_scroll_recently_active(0.35):
                # Don't build background pages while the user is actively
                # scrolling the visible page - it competes for the GUI thread.
                self._preload_render_timer = QTimer(self)
                self._preload_render_timer.setSingleShot(True)
                self._preload_render_timer.timeout.connect(self._run_review_preload_batch)
                self._preload_render_timer.start(max(200, self.REVIEW_PRELOAD_STEP_MS))
                return
            if not self.isVisible() or self._active_render_timer is not None or self._active_render_page is not None:
                self._preload_render_timer = QTimer(self)
                self._preload_render_timer.setSingleShot(True)
                self._preload_render_timer.timeout.connect(self._run_review_preload_batch)
                self._preload_render_timer.start(max(180, self.REVIEW_PRELOAD_STEP_MS))
                return
            try:
                current_row = self.piece_list.currentRow()
            except Exception:
                current_row = -1
            if row == current_row:
                self._cancel_review_preload(discard_page=True)
                return

            rows = state.get("rows") or []
            row_models = state.get("row_models") or []
            layout = state.get("layout")
            old_widget, old_layout = self.rows_widget, self.rows_layout
            self.rows_widget, self.rows_layout = page, layout
            try:
                start = int(state.get("idx", 0))
                end = min(len(rows), start + self.REVIEW_PRELOAD_BATCH_SIZE)
                piece = self.pieces[row]
                for idx in range(start, end):
                    row_model = row_models[idx] if idx < len(row_models) else None
                    self._add_review_row(
                        piece,
                        rows[idx],
                        idx,
                        state.get("max_len", 1),
                        state.get("colors") or self._review_status_colors(),
                        row_model=row_model,
                    )
                state["idx"] = end
            finally:
                self.rows_widget, self.rows_layout = old_widget, old_layout

            if state["idx"] >= len(rows):
                if layout is not None:
                    layout.addStretch(1)
                self._piece_render_complete.add(row)
                self._preload_render_timer = None
                self._preload_render_row = None
                self._preload_render_page = None
                self._preload_render_state = None
                QTimer.singleShot(self.REVIEW_PRELOAD_STEP_MS, self._start_next_review_preload)
                return

            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_review_preload_batch)
            self._preload_render_timer = timer
            timer.start(self.REVIEW_PRELOAD_STEP_MS)
        except Exception:
            self._cancel_review_preload(discard_page=True)

    def _queue_review_page_cache_trim(self, current_row):
        if getattr(self, "_review_page_cache_trim_queued", False):
            return
        self._review_page_cache_trim_queued = True
        QTimer.singleShot(250, lambda row=current_row: self._trim_review_page_cache(row))

    def _trim_review_page_cache(self, current_row):
        self._review_page_cache_trim_queued = False
        try:
            try:
                current_row = int(current_row)
            except Exception:
                current_row = self._displayed_piece_row()
            current_widget = self.rows_stack.currentWidget()
            active_page = self._active_render_page
            preload_page = self._preload_render_page
            cached_rows = list(self._piece_pages.items())
            if len(cached_rows) <= self.REVIEW_MAX_CACHED_PAGES:
                farthest_allowed = self.REVIEW_PRELOAD_RADIUS
            else:
                farthest_allowed = 0

            removable = []
            for row, page in cached_rows:
                if row == current_row or page is current_widget or page is active_page or page is preload_page:
                    continue
                distance = abs(row - current_row) if current_row >= 0 else self.REVIEW_PRELOAD_RADIUS + 1
                if distance > farthest_allowed or len(cached_rows) > self.REVIEW_MAX_CACHED_PAGES:
                    removable.append((distance, row, page))

            if not removable:
                return
            removable.sort(reverse=True)
            for _distance, row, page in removable[:4]:
                try:
                    if self._piece_pages.get(row) is page:
                        self._piece_pages.pop(row, None)
                    self._piece_render_complete.discard(row)
                    self._remove_review_page_widget(page)
                except Exception:
                    pass
            if len(removable) > 4:
                self._queue_review_page_cache_trim(current_row)
        except Exception:
            pass

    def _request_render_piece(self, row):
        if row < 0 or row >= len(self.pieces):
            return
        self._trace_review_perf(
            "request_render_piece",
            None,
            force=True,
            row=row,
        )
        self._last_review_selection_change = time.monotonic()
        self._cancel_review_preload(discard_page=True)
        self._save_current_review_scroll()
        QTimer.singleShot(0, lambda row=row: self._render_piece(row))

    def _set_rows_rebuild_active(self, active):
        self._rows_rebuild_active = bool(active)
        enabled = not self._rows_rebuild_active
        widgets = [self.rows_widget]
        try:
            if self.rows_stack.currentWidget() is self.rows_widget:
                widgets.append(self.scroll.viewport())
        except Exception:
            pass
        for widget in widgets:
            try:
                widget.setUpdatesEnabled(enabled)
            except Exception:
                pass

    def _finish_rows_rebuild(self, final=True):
        trace_started = time.perf_counter()
        self._refresh_review_stream_geometry(final=final)
        self._set_rows_rebuild_active(False)
        for widget in (self.rows_widget, self.rows_stack, self.scroll.viewport(), self.scroll):
            try:
                widget.update()
            except Exception:
                pass
        self._trace_review_perf(
            "finish_rows_rebuild",
            trace_started,
            final=bool(final),
        )

    def _sync_review_scroll_range(self, page=None):
        try:
            page = page or self.rows_stack.currentWidget()
            if page is None:
                return
            layout = page.layout()
            viewport_height = max(1, self.scroll.viewport().height())
            hint_height = 0
            if layout is not None:
                layout.invalidate()
                try:
                    hint_height = int(layout.totalSizeHint().height())
                except Exception:
                    hint_height = 0
            if hint_height <= 0:
                hint_height = max(page.sizeHint().height(), page.minimumSizeHint().height())
            content_height = max(viewport_height, hint_height)
            # IMPORTANT: lock the container heights BEFORE activating the layout.
            # Activating first lays the fixed-height rows out inside the stale
            # (smaller) page height, which briefly paints them overlapping while
            # a render stream is in flight. The stack must be resized in the
            # same synchronous step as the page, otherwise the stacked layout
            # squashes the page back to the old stack height until the scroll
            # area's async layout pass catches up (visible vertical bouncing).
            page.setMinimumHeight(content_height)
            page.setMaximumHeight(content_height)
            self.rows_stack.setMinimumHeight(content_height)
            self.rows_stack.setMaximumHeight(content_height)
            self.rows_stack.resize(max(1, self.rows_stack.width()), content_height)
            page.resize(max(1, self.rows_stack.width()), content_height)
            if layout is not None:
                layout.activate()
            page.updateGeometry()
            self.rows_stack.updateGeometry()
        except Exception:
            pass

    def _save_current_review_scroll(self):
        row = self._current_scroll_piece_row
        if row is None:
            return
        try:
            page = self._piece_pages.get(row)
            if page is None or self.rows_stack.currentWidget() is not page:
                return
            self._piece_scroll_positions[row] = self.scroll.verticalScrollBar().value()
        except Exception:
            pass

    def _remember_current_review_scroll(self, value):
        if self._restoring_review_scroll:
            return
        self._last_review_scroll_activity = time.monotonic()
        row = self._current_scroll_piece_row
        if row is None:
            return
        try:
            page = self._piece_pages.get(row)
            if page is None or self.rows_stack.currentWidget() is not page:
                return
            self._piece_scroll_positions[row] = int(value)
            self._queue_refresh_current_visible_dirty_source_previews()
        except Exception:
            pass

    def _restore_review_scroll(self, row):
        target_value = int(self._piece_scroll_positions.get(row, 0) or 0)

        def _apply_saved_scroll():
            try:
                page = self._piece_pages.get(row)
                if page is None or self.rows_stack.currentWidget() is not page:
                    return
                self._current_scroll_piece_row = row
                self._sync_review_scroll_range()
                bar = self.scroll.verticalScrollBar()
                value = max(0, min(bar.maximum(), target_value))
                self._restoring_review_scroll = True
                try:
                    bar.setValue(value)
                finally:
                    self._restoring_review_scroll = False
            except Exception:
                self._restoring_review_scroll = False

        _apply_saved_scroll()
        QTimer.singleShot(0, _apply_saved_scroll)
        QTimer.singleShot(25, _apply_saved_scroll)

    @classmethod
    def _tag_label_font_point_size(cls, text, label_width=None):
        text = str(text or "")
        maximum = float(cls.REVIEW_TAG_LABEL_MAX_FONT_PT)
        minimum = float(cls.REVIEW_TAG_LABEL_MIN_FONT_PT)
        available_width = max(
            12.0,
            float(label_width or cls.REVIEW_TAG_LABEL_WIDTH) - 6.0,
        )
        if not text:
            return maximum

        if QApplication.instance() is not None:
            font = QFont("Consolas")
            font.setStyleHint(QFont.Monospace)
            ordinal_font = QFont(font)
            ordinal_parts = re.findall(r"\(\d+\)", text)
            base_text = re.sub(r"\(\d+\)", "", text)
            half_steps = int(round((maximum - minimum) * 2))
            for step in range(half_steps + 1):
                point_size = maximum - (step * 0.5)
                font.setPointSizeF(point_size)
                ordinal_font.setPointSizeF(
                    cls._tag_label_ordinal_font_point_size(point_size)
                )
                rendered_width = QFontMetricsF(font).horizontalAdvance(base_text)
                rendered_width += sum(
                    QFontMetricsF(ordinal_font).horizontalAdvance(part)
                    for part in ordinal_parts
                )
                if rendered_width <= available_width:
                    return point_size
            return minimum

        estimated_width = max(1.0, len(text) * maximum * 0.62)
        fitted = min(maximum, maximum * available_width / estimated_width)
        fitted = int(fitted * 2) / 2.0
        return max(minimum, fitted)

    def _apply_tag_label_display(self, label, text, status):
        font_point_size = self._tag_label_font_point_size(
            text,
            label.width() or self.REVIEW_TAG_LABEL_WIDTH,
        )
        label.setText(self._tag_label_rich_text(text, font_point_size))
        label.setProperty("sdl_tag_font_point_size", font_point_size)
        color = {
            "green": self.THEME["success"],
            "yellow": self.THEME["warning"],
            "purple": self.THEME["purple"],
            "red": self.THEME["danger"],
        }.get(status, self.THEME["muted"])
        label.setStyleSheet(
            f"color: {color}; background: transparent; "
            f"font: {font_point_size:g}pt Consolas, 'Courier New', monospace; "
            "padding: 0px 2px;"
        )

    def _tag_label(self, source_tag, target_tag, status, source_label=None, target_label=None):
        text = self._tag_label_text(source_tag, target_tag, source_label, target_label)
        label = QLabel()
        label.setObjectName("SdlReviewTagLabel")
        label.setTextFormat(Qt.RichText)
        label.setAlignment(Qt.AlignCenter)
        label.setFixedWidth(self.REVIEW_TAG_LABEL_WIDTH)
        self._apply_tag_label_display(label, text, status)
        return label

    @staticmethod
    def _refresh_notepad_browser_row_statuses(browser, status_by_row):
        """Patch Notepad DOM markers without reloading the edited document."""
        normalized = {
            str(int(row_index)): {
                "status": str((row_state or {}).get("status") or "green"),
                "reason": str((row_state or {}).get("reason") or ""),
            }
            for row_index, row_state in (status_by_row or {}).items()
        }
        status_by_row_json = json.dumps(normalized)
        script = f"""
            (() => {{
                const statusByRow = {status_by_row_json};
                let changed = 0;
                document.querySelectorAll('[data-sdl-notepad-row-index]').forEach(element => {{
                    const rowState = statusByRow[
                        element.getAttribute('data-sdl-notepad-row-index')
                    ];
                    if (!rowState) return;
                    element.setAttribute('data-sdl-notepad-status', rowState.status);
                    if (rowState.reason) element.setAttribute(
                        'data-sdl-notepad-status-reason', rowState.reason
                    );
                    else element.removeAttribute('data-sdl-notepad-status-reason');
                    changed += 1;
                }});
                return changed;
            }})();
        """
        try:
            browser.page().runJavaScript(script)
        except RuntimeError:
            pass

    @classmethod
    def _refresh_notepad_browser_row_status(cls, browser, row_index, status, reason=""):
        cls._refresh_notepad_browser_row_statuses(
            browser,
            {row_index: {"status": status, "reason": reason}},
        )

    def _refresh_visible_review_row_status(self, piece_index, row_index):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            if self.piece_list.currentRow() != piece_index:
                return
            piece = self.pieces[piece_index]
            rows = piece.get("rows") or []
            if row_index < 0 or row_index >= len(rows):
                return
            row_data = rows[row_index]
            page = self._piece_pages.get(piece_index) or self.rows_widget
            if page is None:
                return
            browser = page.findChild(QWidget, "SdlReviewNotepadBrowser")
            if browser is not None:
                self._refresh_notepad_browser_row_status(
                    browser,
                    row_index,
                    row_data.get("status", "green"),
                    row_data.get("reason", ""),
                )
                self._status_jump_indices.clear()
                return
            frame = None
            for candidate in page.findChildren(QFrame, "SdlReviewRow"):
                try:
                    if self._review_row_index_property(candidate) == row_index:
                        frame = candidate
                        break
                except Exception:
                    continue
            if frame is None:
                return
            colors = self._review_status_colors()
            bg, _source_bar, _target_bar, dot_color, border_color = colors.get(row_data["status"], colors["green"])
            if bool(frame.property("sdl_notepad_layout")):
                status_color = {
                    "green": self.THEME["success"],
                    "yellow": self.THEME["warning"],
                    "purple": self.THEME["purple"],
                    "red": self.THEME["danger"],
                }.get(row_data["status"], self.THEME["border"])
                row_style = (
                    f"QFrame#SdlReviewRow {{ background-color: {self.THEME['panel_alt']}; border: 0; "
                    f"border-left: 3px solid {status_color}; border-bottom: 1px solid #363c46; "
                    "border-radius: 0; }}"
                )
                frame.setProperty("sdl_status", row_data["status"])
                frame.setProperty("sdl_base_style", row_style)
                frame.setStyleSheet(row_style)
                tag_label = frame.findChild(QLabel, "SdlReviewNotepadTag")
                if tag_label is not None:
                    tag_label.setStyleSheet(
                        f"color: {status_color}; background-color: #20242a; border: 0; "
                        "border-right: 1px solid #404854; padding: 3px 8px; "
                        "font: 9pt Consolas, 'Courier New', monospace;"
                    )
                self._status_jump_indices.clear()
                frame.update()
                return
            row_style = f"QFrame#SdlReviewRow {{ background-color: {bg}; border: 1px solid {border_color}; border-radius: 3px; }}"
            frame.setProperty("sdl_status", row_data["status"])
            frame.setProperty("sdl_base_style", row_style)
            frame.setStyleSheet(row_style)

            tag_label = frame.findChild(QLabel, "SdlReviewTagLabel")
            if tag_label is not None:
                # Recompute the label TEXT too — editing an "Empty" row
                # creates the target node (row_data["target_tag"] is set in
                # _target_html_with_edit), so the stale "Empty" caption must
                # be replaced, not just recolored.
                source_tag_label = row_data.get("source_tag_label")
                target_tag_label = row_data.get("target_tag_label")
                compact_tn_label = self._compact_translator_note_display_label(
                    piece, row_index
                )
                if compact_tn_label:
                    source_tag_label = compact_tn_label
                    target_tag_label = compact_tn_label
                tag_text = self._tag_label_text(
                    row_data.get("source_tag"),
                    row_data.get("target_tag"),
                    source_tag_label,
                    target_tag_label,
                )
                self._apply_tag_label_display(
                    tag_label, tag_text, row_data["status"]
                )
                tag_label.setToolTip(row_data.get("reason", ""))

            dot = frame.findChild(QLabel, "SdlReviewStatusDot")
            if dot is not None:
                dot.setStyleSheet(f"color: {dot_color}; background: transparent; font-size: 13pt;")
                dot.setToolTip(row_data.get("reason", ""))

            self._status_jump_indices.clear()
            frame.update()
        except Exception:
            pass

    def _refresh_notepad_page_after_save(self, piece_index, rebuilt, saved_html, html_text):
        page = self._piece_pages.get(piece_index)
        browser = page.findChild(QWidget, "SdlReviewNotepadBrowser") if page is not None else None
        if browser is not None and saved_html != html_text and hasattr(browser, "setHtml"):
            # Saved HTML intentionally has all editor-only source metadata
            # stripped. Rebuild the rendered document from the refreshed
            # piece so hover/context-menu source text survives a normalization
            # reload. Do not refill a row the user deliberately emptied.
            self._set_notepad_browser_html(
                browser,
                rebuilt,
                self._notepad_initial_document_html(
                    rebuilt, fill_untranslated=False
                ),
            )
        elif browser is not None:
            self._refresh_notepad_browser_row_statuses(
                browser,
                {
                    row_index: {
                        "status": row_data.get("status", "green"),
                        "reason": row_data.get("reason", ""),
                    }
                    for row_index, row_data in enumerate(rebuilt.get("rows") or [])
                },
            )

    def _insert_into_review_editor(self, editor, text):
        try:
            if isinstance(editor, QPlainTextEdit) and not editor.isReadOnly():
                self._replace_editor_text_preserving_undo(editor, text)
                editor.setFocus(Qt.OtherFocusReason)
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _replace_editor_text_preserving_undo(editor, text):
        if not isinstance(editor, QPlainTextEdit):
            return False
        cursor = editor.textCursor()
        cursor.beginEditBlock()
        try:
            editor.selectAll()
            editor.insertPlainText(str(text or ""))
        finally:
            cursor.endEditBlock()
        return True

    def _review_row_frame(self, piece_index, row_index):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return None
            page = self._piece_pages.get(piece_index)
            if page is None and self.piece_list.currentRow() == piece_index:
                page = self.rows_widget
            if page is None:
                return None
            for candidate in page.findChildren(QFrame, "SdlReviewRow"):
                try:
                    if self._review_row_index_property(candidate) == row_index:
                        return candidate
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _review_row_frames_by_index(self, piece_index):
        frames = {}
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return frames
            page = self._piece_pages.get(piece_index)
            if page is None and self.piece_list.currentRow() == piece_index:
                page = self.rows_widget
            if page is None:
                return frames
            for candidate in page.findChildren(QFrame, "SdlReviewRow"):
                try:
                    frames[self._review_row_index_property(candidate)] = candidate
                except Exception:
                    continue
        except Exception:
            return frames
        frames.pop(-1, None)
        return frames

    def _review_row_frame_is_near_viewport(self, frame, margin=180):
        try:
            top = int(frame.y())
            bottom = top + int(frame.height())
            visible_top = int(self.scroll.verticalScrollBar().value())
            visible_bottom = visible_top + int(self.scroll.viewport().height())
            return bottom >= visible_top - margin and top <= visible_bottom + margin
        except Exception:
            return True

    def _refresh_visible_review_row_source_preview(self, piece_index, row_index, frame=None, sync_geometry=True):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return False
            piece = self.pieces[piece_index]
            rows = piece.get("rows") or []
            if row_index < 0 or row_index >= len(rows):
                return False
            row_data = rows[row_index]
            frame = frame or self._review_row_frame(piece_index, row_index)
            if frame is None:
                return False
            grid = frame.layout()
            if not isinstance(grid, QGridLayout):
                return False

            target_widget = self._review_row_target_widget(frame)

            source_text = row_data.get("source", "")
            target_text = row_data.get("target", "")
            tooltip_translation = self._row_tooltip_translation(piece, row_data)
            row_snapshot = self._review_row_snapshot(row_data)
            tooltip_preview = self._row_machine_translation_preview_from_snapshot(row_snapshot)
            tooltip_state = self._row_machine_translation_preview_state(row_snapshot)
            tooltip_pending = tooltip_state == "pending"
            tooltip_detail = str(
                row_data.get("tooltip_translation_error_detail")
                or row_data.get("tooltip_translation_status")
                or ""
            )
            if bool(frame.property("sdl_notepad_layout")):
                preview_text = str(tooltip_preview or "").strip()
                details = f"Source:\n{source_text}\n\nOutput:\n{target_text}"
                if preview_text:
                    details += f"\n\nMachine translation preview:\n{preview_text}"
                if target_widget is not None:
                    target_widget.setToolTip(self._wrapped_tooltip(details))
                row_data.pop("_source_preview_dirty", None)
                return True
            row_height = self._review_row_height(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                tooltip_preview_text=tooltip_preview,
            )
            source_missing = bool(row_data.get(
                "source_missing", not row_data.get("source_tag")
            ))
            target_missing = bool(row_data.get(
                "target_missing", not row_data.get("target_tag")
            ))
            target_editable = not source_missing or not target_missing

            source_label = self._text_label(
                source_text,
                missing=source_missing,
                empty_placeholder=(
                    "[Translator Note]"
                    if row_data.get("translator_note") else None
                ),
                tooltip_translation=tooltip_preview,
                tooltip_pending=tooltip_pending,
                tooltip_state=tooltip_state,
                tooltip_detail=tooltip_detail,
                translate_tooltip_callback=(
                    lambda pi=piece_index, ri=row_index: self._translate_single_row_tooltip(pi, ri)
                ) if source_text else None,
                inject_machine_translation_callback=(
                    lambda pi=piece_index, ri=row_index, text=tooltip_translation, ed=target_widget:
                        self._inject_machine_translation_to_target(pi, ri, text, ed)
                ) if tooltip_translation and target_editable and tooltip_state == "translation" else None,
            )
            source_lines, target_lines, tooltip_lines = self._review_row_line_counts_for_width(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                self._review_render_viewport_width(),
                two_column_layout=bool(frame.property("sdl_two_column_layout")),
                tooltip_preview_text=tooltip_preview,
            )
            source_label.setToolTip(self._wrapped_tooltip(source_text))

            if not self._replace_review_row_source_widget(frame, source_label):
                return False

            frame.setFixedHeight(row_height)
            if target_widget is not None:
                self._apply_review_row_text_geometry(
                    frame,
                    source_label,
                    target_widget,
                    row_height,
                    source_lines=source_lines,
                    target_lines=target_lines,
                    tooltip_lines=tooltip_lines,
                )
            frame.updateGeometry()
            frame.update()
            try:
                grid.invalidate()
                grid.activate()
            except Exception:
                pass
            if sync_geometry:
                self._refresh_review_stream_geometry(final=False)
            row_data.pop("_source_preview_dirty", None)
            return True
        except Exception:
            return False

    def _patch_review_row_machine_translation_preview(self, piece_index, row_index, frame=None, sync_geometry=True):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return False
            piece = self.pieces[piece_index]
            rows = piece.get("rows") or []
            if row_index < 0 or row_index >= len(rows):
                return False
            row_data = rows[row_index]
            frame = frame or self._review_row_frame(piece_index, row_index)
            if frame is None:
                return False
            grid = frame.layout()
            if not isinstance(grid, QGridLayout):
                return False
            source_widget = self._review_row_source_widget(frame)
            if source_widget is None or source_widget.objectName() != "SdlReviewSourceText":
                return False
            translated_label = source_widget.findChild(QLabel, "SdlReviewMachineTranslationPending")
            if translated_label is None:
                translated_label = source_widget.findChild(QLabel, "SdlReviewMachineTranslation")
            if translated_label is None:
                return False

            target_widget = self._review_row_target_widget(frame)
            source_text = row_data.get("source", "")
            target_text = row_data.get("target", "")
            tooltip_translation = self._row_tooltip_translation(piece, row_data)
            row_snapshot = self._review_row_snapshot(row_data)
            tooltip_preview = self._row_machine_translation_preview_from_snapshot(row_snapshot)
            tooltip_state = self._row_machine_translation_preview_state(row_snapshot)
            tooltip_pending = tooltip_state == "pending"
            tooltip_detail = str(
                row_data.get("tooltip_translation_error_detail")
                or row_data.get("tooltip_translation_status")
                or ""
            )
            preview_text = tooltip_preview
            if not str(preview_text or "").strip():
                return False
            translated_label.setObjectName(
                "SdlReviewMachineTranslationPending" if tooltip_state in {"pending", "error"} else "SdlReviewMachineTranslation"
            )
            translated_label.setText(preview_text)
            if tooltip_state in {"pending", "error"}:
                translated_label.setToolTip(
                    self._wrapped_tooltip(tooltip_detail)
                    if tooltip_detail
                    else f"Machine translation preview is being generated with {self._machine_translation_provider_label()}."
                )
                translated_label.setStyleSheet(
                    "QLabel#SdlReviewMachineTranslationPending { "
                    "color: #d8c99b; background: rgba(54, 45, 23, 180); "
                    "border: 1px dashed #8a6f2a; border-left: 3px solid #d39e00; "
                    "border-radius: 4px; padding: 5px 8px; font-size: 8pt; "
                    "}"
                )
            else:
                translated_label.setToolTip(self._wrapped_tooltip(tooltip_translation))
                translated_label.setStyleSheet(
                    "QLabel#SdlReviewMachineTranslation { "
                    "color: #b6c7dc; background: rgba(23, 37, 54, 185); "
                    "border: 1px solid #37536d; border-left: 3px solid #5aa7d8; "
                    "border-radius: 4px; padding: 3px 7px; font-size: 7pt; "
                    "}"
                )

            target_editable = bool(row_data.get("source_tag")) or bool(row_data.get("target_tag"))
            inject_callback = (
                lambda pi=piece_index, ri=row_index, text=tooltip_translation, ed=target_widget:
                    self._inject_machine_translation_to_target(pi, ri, text, ed)
            ) if tooltip_translation and target_editable and tooltip_state == "translation" else None
            raw_label = source_widget.findChild(QLabel, "SdlReviewSourceRawText")
            self._wire_source_preview_context_menu(
                raw_label or translated_label,
                [source_widget, raw_label, translated_label],
                translate_tooltip_callback=(
                    lambda pi=piece_index, ri=row_index: self._translate_single_row_tooltip(pi, ri)
                ) if source_text else None,
                inject_machine_translation_callback=inject_callback,
            )

            source_lines, target_lines, tooltip_lines = self._review_row_line_counts_for_width(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                self._review_render_viewport_width(),
                two_column_layout=bool(frame.property("sdl_two_column_layout")),
                tooltip_preview_text=tooltip_preview,
            )
            row_height = self._review_row_height(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                tooltip_preview_text=tooltip_preview,
            )
            frame.setFixedHeight(row_height)
            if target_widget is not None:
                self._apply_review_row_text_geometry(
                    frame,
                    source_widget,
                    target_widget,
                    row_height,
                    source_lines=source_lines,
                    target_lines=target_lines,
                    tooltip_lines=tooltip_lines,
                )
            self._update_review_row_inject_button(
                frame,
                bool(tooltip_translation) and bool(target_editable) and tooltip_state == "translation",
            )
            frame.updateGeometry()
            frame.update()
            try:
                grid.invalidate()
                grid.activate()
            except Exception:
                pass
            if sync_geometry:
                self._refresh_review_stream_geometry(final=False)
            row_data.pop("_source_preview_dirty", None)
            return True
        except Exception:
            return False

    def _update_review_row_source_previews(self, piece_index, row_indices, visible_only=True):
        try:
            frames = self._review_row_frames_by_index(piece_index)
            if not frames:
                return False
            refreshed = False
            for row_index in sorted(set(row_indices or [])):
                frame = frames.get(row_index)
                if frame is None:
                    continue
                if visible_only and not self._review_row_frame_is_near_viewport(frame):
                    continue
                if self._patch_review_row_machine_translation_preview(
                    piece_index,
                    row_index,
                    frame=frame,
                    sync_geometry=False,
                ) or self._refresh_visible_review_row_source_preview(
                    piece_index,
                    row_index,
                    frame=frame,
                    sync_geometry=False,
                ):
                    refreshed = True
            if refreshed:
                self._refresh_review_stream_geometry(final=False)
            return refreshed
        except Exception:
            return False

    def _queue_refresh_current_visible_dirty_source_previews(self):
        if self._review_dirty_preview_refresh_queued:
            return
        self._review_dirty_preview_refresh_queued = True
        if self._review_context_menu_is_open():
            return
        QTimer.singleShot(0, self._refresh_current_visible_dirty_source_previews)

    def _refresh_current_visible_dirty_source_previews(self):
        if self._review_context_menu_is_open():
            self._review_dirty_preview_refresh_queued = True
            return
        self._review_dirty_preview_refresh_queued = False
        try:
            piece_index = self._displayed_piece_row()
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            rows = self.pieces[piece_index].get("rows") or []
            dirty_rows = [idx for idx, row_data in enumerate(rows) if row_data.get("_source_preview_dirty")]
            if dirty_rows:
                self._update_review_row_source_previews(piece_index, dirty_rows, visible_only=True)
        except Exception:
            pass

    def _refresh_visible_review_row_source_previews(self, piece_index, row_indices, visible_only=False):
        try:
            frames = self._review_row_frames_by_index(piece_index)
            if not frames:
                return False
            refreshed = False
            for row_index in sorted(set(row_indices or [])):
                frame = frames.get(row_index)
                if frame is None:
                    continue
                if visible_only and not self._review_row_frame_is_near_viewport(frame):
                    continue
                if self._refresh_visible_review_row_source_preview(
                    piece_index,
                    row_index,
                    frame=frame,
                    sync_geometry=False,
                ):
                    refreshed = True
            if refreshed:
                self._refresh_review_stream_geometry(final=False)
            return refreshed
        except Exception:
            return False

    def _current_piece_row(self):
        try:
            return self.piece_list.currentRow()
        except Exception:
            return -1

    def _displayed_piece_row(self):
        try:
            current_widget = self.rows_stack.currentWidget()
            for row, page in self._piece_pages.items():
                if page is current_widget:
                    return row
        except Exception:
            pass
        return self._current_piece_row()

    def _piece_list_viewport_pos_from_event(self, obj, event):
        try:
            if hasattr(event, "position"):
                pos = event.position().toPoint()
            else:
                pos = event.pos()
            if obj is self.piece_list:
                return self.piece_list.viewport().mapFrom(self.piece_list, pos)
            return pos
        except Exception:
            try:
                return self.piece_list.viewport().rect().center()
            except Exception:
                return QPoint()

    def eventFilter(self, obj, event):
        try:
            if obj is self.piece_list and event.type() == QEvent.KeyPress:
                if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
                    self.piece_list.selectAll()
                    event.accept()
                    return True
            piece_list_obj = obj is self.piece_list
            piece_list_viewport = False
            try:
                piece_list_viewport = obj is self.piece_list.viewport()
            except Exception:
                piece_list_viewport = False
            if piece_list_obj or piece_list_viewport:
                event_type = event.type()
                if event_type == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                    self._translate_piece_list_context_selection(
                        self._piece_list_viewport_pos_from_event(obj, event)
                    )
                    event.accept()
                    return True
                if event_type == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                    event.accept()
                    return True
                if event_type == QEvent.ContextMenu:
                    self._translate_piece_list_context_selection(
                        self._piece_list_viewport_pos_from_event(obj, event)
                    )
                    event.accept()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _selected_piece_rows(self):
        try:
            rows = [self.piece_list.row(item) for item in self.piece_list.selectedItems()]
        except Exception:
            rows = []
        return sorted({row for row in rows if 0 <= row < len(self.pieces)})

    def _translate_piece_list_context_selection(self, pos):
        try:
            item = self.piece_list.itemAt(pos)
            if item is None:
                return
            clicked_row = self.piece_list.row(item)
            if not item.isSelected():
                previous_signal_state = self.piece_list.blockSignals(True)
                try:
                    self.piece_list.clearSelection()
                    item.setSelected(True)
                finally:
                    self.piece_list.blockSignals(previous_signal_state)
            rows = self._selected_piece_rows() or [clicked_row]
            menu = QMenu(self)
            menu.setStyleSheet(
                "QMenu { padding: 4px 6px 4px 4px; }"
                "QMenu::item { padding: 6px 18px 6px 12px; }"
                "QMenu::item:disabled { color: #6f7782; background: transparent; }"
                "QMenu::item:selected:disabled { color: #6f7782; background: transparent; }"
            )
            entry_count = len(rows)
            action_text = (
                f"🌐 Generate Machine Translation Preview ({entry_count} entries)"
                if entry_count != 1
                else "🌐 Generate Machine Translation Preview"
            )
            translate_action = menu.addAction(action_text)
            translate_action.setEnabled(not self._tooltip_translation_running)
            translate_action.triggered.connect(
                lambda _checked=False, selected_rows=list(rows): self._translate_piece_rows_tooltips(selected_rows)
            )
            eligible_rows = [
                row for row in rows
                if 0 <= row < len(self.pieces) and self._piece_needs_manual_green_override(self.pieces[row])
            ]
            undo_rows = [
                row for row in rows
                if 0 <= row < len(self.pieces) and self.pieces[row].get("manual_green_override")
            ]
            menu.addSeparator()
            completed_text = (
                f"\u2705  Mark as Completed ({len(eligible_rows)} entries)"
                if len(rows) != 1
                else "\u2705  Mark as Completed"
            )
            completed_action = menu.addAction(completed_text)
            completed_action.setEnabled(bool(eligible_rows))
            completed_action.triggered.connect(
                lambda _checked=False, selected_rows=list(rows): self._mark_review_sidecars_completed(selected_rows)
            )
            undo_text = (
                f"\u21a9\ufe0f  Undo Completed Mark ({len(undo_rows)} entries)"
                if len(rows) != 1
                else "\u21a9\ufe0f  Undo Completed Mark"
            )
            undo_action = menu.addAction(undo_text)
            undo_action.setEnabled(bool(undo_rows))
            undo_action.triggered.connect(
                lambda _checked=False, selected_rows=list(rows): self._undo_review_sidecars_completed(selected_rows)
            )
            self._piece_list_context_menu = menu
            self._set_review_context_menu_open(True)
            menu.aboutToHide.connect(lambda m=menu: self._clear_piece_list_context_menu(m))
            menu.popup(self.piece_list.viewport().mapToGlobal(pos))
        except Exception:
            pass

    def _update_review_layout_button(self):
        button = getattr(self, "two_column_layout_btn", None)
        if button is None:
            return
        compact = bool(getattr(self, "_two_column_layout_enabled", True))
        button.setText(
            self.TWO_COLUMN_LAYOUT_BUTTON_TEXT
            if compact
            else self.NOTEPAD_LAYOUT_BUTTON_TEXT
        )
        button.setToolTip(
            (
                "Compact: source and output cards with row actions. Click for Notepad."
                if self._review_notepad_mode_is_available()
                else "Compact mode. Notepad is unavailable because this Lite package does not include Qt WebEngine."
            )
            if compact
            else "Notepad: edit the complete raw output HTML document in one code buffer. Click for Compact."
        )

    def _clear_piece_list_context_menu(self, menu):
        try:
            if getattr(self, "_piece_list_context_menu", None) is menu:
                self._piece_list_context_menu = None
            self._set_review_context_menu_open(False)
            menu.deleteLater()
        except Exception:
            pass

    def _start_tooltip_translation(self, row, work, ready_text="Preview Ready"):
        if self._tooltip_translation_running:
            return False
        if row < 0 or row >= len(self.pieces):
            return False
        if not work:
            try:
                self.translate_tooltips_btn.setText(ready_text)
                QTimer.singleShot(
                    1200,
                    lambda: self.translate_tooltips_btn.setText(self.TRANSLATE_TOOLTIPS_BUTTON_TEXT),
                )
            except Exception:
                pass
            return False

        self._tooltip_translation_running = True
        target_code = self._review_target_language_code()
        try:
            self.translate_tooltips_btn.setEnabled(False)
            self.translate_tooltips_btn.setText("Translating...")
        except Exception:
            pass
        self._mark_tooltip_translation_pending(row, work)

        def _worker():
            translations = {}
            error = ""
            try:
                work_keys = [item[1] for item in work]
                def _status(message):
                    self._tooltip_translation_status.emit(row, work_keys, str(message or ""))

                translator = self._machine_translation_translator(target_code, status_callback=_status)
                translations, error = self._translate_tooltip_work(translator, work)
                self._tooltip_translation_progress.emit(1, 1)
            except Exception as exc:
                error = str(exc)
            self._tooltip_translation_finished.emit(row, translations, error)

        threading.Thread(target=_worker, name="sdlxliff-machine-translation-preview", daemon=True).start()
        return True

    def _start_piece_list_tooltip_translation(self, jobs):
        if self._tooltip_translation_running:
            return False
        jobs = [(row, work) for row, work in (jobs or []) if 0 <= row < len(self.pieces) and work]
        if not jobs:
            try:
                self.translate_tooltips_btn.setText("Preview Ready")
                QTimer.singleShot(
                    1200,
                    lambda: self.translate_tooltips_btn.setText(self.TRANSLATE_TOOLTIPS_BUTTON_TEXT),
                )
            except Exception:
                pass
            return False

        self._tooltip_translation_running = True
        self._tooltip_translation_batch_active = True
        target_code = self._review_target_language_code()
        total_jobs = len(jobs)
        try:
            self.translate_tooltips_btn.setEnabled(False)
            self.translate_tooltips_btn.setText(f"Translating 0/{total_jobs}...")
        except Exception:
            pass
        current_row = self._displayed_piece_row()
        for piece_index, work in jobs:
            self._mark_tooltip_translation_pending(piece_index, work, refresh=piece_index == current_row)

        def _worker():
            translated_count = 0
            last_error = ""
            status_context = {
                "piece_index": jobs[0][0] if jobs else -1,
                "keys": [],
            }
            def _status(message):
                self._tooltip_translation_status.emit(
                    int(status_context.get("piece_index", -1)),
                    list(status_context.get("keys") or []),
                    str(message or ""),
                )

            try:
                translator = self._machine_translation_translator(target_code, status_callback=_status)
            except Exception as exc:
                self._tooltip_translation_batch_finished.emit(0, total_jobs, str(exc))
                return

            for done, (piece_index, work) in enumerate(jobs, start=1):
                status_context["piece_index"] = piece_index
                status_context["keys"] = [item[1] for item in work]
                translations = {}
                error = ""
                try:
                    translations, error = self._translate_tooltip_work(translator, work)
                    translated_count += len(translations)
                except Exception as exc:
                    error = str(exc)
                if error:
                    last_error = error
                self._tooltip_translation_finished.emit(piece_index, translations, error)
                self._tooltip_translation_progress.emit(done, total_jobs)
            self._tooltip_translation_batch_finished.emit(translated_count, total_jobs, last_error)

        threading.Thread(target=_worker, name="sdlxliff-piece-list-machine-translation-preview", daemon=True).start()
        return True

    def _translate_piece_rows_tooltips(self, piece_rows):
        if self._tooltip_translation_running:
            return
        rows = sorted({row for row in (piece_rows or []) if 0 <= row < len(self.pieces)})
        jobs = [(row, self._piece_tooltip_work(row)) for row in rows]
        jobs = [(row, work) for row, work in jobs if work]
        if len(jobs) == 1:
            row, work = jobs[0]
            self._start_tooltip_translation(row, work, ready_text="Preview Ready")
        elif jobs:
            self._start_piece_list_tooltip_translation(jobs)
        else:
            try:
                self.translate_tooltips_btn.setText("Preview Ready")
                QTimer.singleShot(
                    1200,
                    lambda: self.translate_tooltips_btn.setText(self.TRANSLATE_TOOLTIPS_BUTTON_TEXT),
                )
            except Exception:
                pass

    def _translate_current_piece_tooltips(self):
        if self._tooltip_translation_running:
            return
        row = self._current_piece_row()
        if row < 0 or row >= len(self.pieces):
            return
        work = self._piece_tooltip_work(row)
        self._start_tooltip_translation(row, work, ready_text="Preview Ready")

    def _translate_single_row_tooltip(self, piece_index, row_index):
        if self._tooltip_translation_running:
            return
        if piece_index < 0 or piece_index >= len(self.pieces):
            return
        piece = self.pieces[piece_index]
        rows = piece.get("rows") or []
        if row_index < 0 or row_index >= len(rows):
            return
        row_data = rows[row_index]
        source_text = str(row_data.get("source", "") or "").strip()
        if not source_text:
            return
        work = [(
            row_index,
            self._tooltip_translation_key(piece, row_data),
            source_text,
            row_data.get("source_tag"),
        )]
        self._start_tooltip_translation(piece_index, work, ready_text="Preview Ready")

    def _update_tooltip_translation_progress(self, done, total):
        try:
            if self._tooltip_translation_running:
                if int(total or 0) > 1:
                    self.translate_tooltips_btn.setText(f"Translating {int(done or 0)}/{int(total)}...")
                else:
                    self.translate_tooltips_btn.setText("Translating...")
        except Exception:
            pass

    def _apply_tooltip_translations(self, row, translations, error):
        batch_active = bool(getattr(self, "_tooltip_translation_batch_active", False))
        if not batch_active:
            self._tooltip_translation_running = False
            try:
                self.translate_tooltips_btn.setEnabled(True)
                self.translate_tooltips_btn.setText(self.TRANSLATE_TOOLTIPS_BUTTON_TEXT)
            except Exception:
                pass
        if 0 <= row < len(self.pieces):
            piece = self.pieces[row]
            changed_rows = self._store_tooltip_translations(row, translations, error)
            if changed_rows:
                if row == self._displayed_piece_row():
                    if self._review_context_menu_is_open():
                        for row_index in changed_rows:
                            try:
                                piece.get("rows", [])[row_index]["_source_preview_dirty"] = True
                            except Exception:
                                pass
                        self._queue_refresh_current_visible_dirty_source_previews()
                    else:
                        self._update_review_row_source_previews(row, changed_rows, visible_only=True)
            if translations:
                try:
                    self._last_machine_translation_signature = self._current_machine_translation_signature()
                except Exception:
                    pass
            self._refresh_open_notepad_machine_translation_context()
        if batch_active:
            return
        message = self._tooltip_translation_result_message(translations, error)
        if error and not translations:
            try:
                self.save_status_label.setText(message)
            except Exception:
                pass
        elif translations:
            try:
                self.save_status_label.setText(message)
                QTimer.singleShot(2500, self._clear_review_save_status)
            except Exception:
                pass

    def _clear_review_save_status(self):
        try:
            self.save_status_label.setText("")
        except RuntimeError:
            pass

    def _finish_piece_list_tooltip_translations(self, translated_count, piece_count, error):
        self._tooltip_translation_batch_active = False
        self._tooltip_translation_running = False
        self._refresh_open_notepad_machine_translation_context()
        try:
            self.translate_tooltips_btn.setEnabled(True)
            self.translate_tooltips_btn.setText("Preview Ready")
            QTimer.singleShot(
                1200,
                lambda: self.translate_tooltips_btn.setText(self.TRANSLATE_TOOLTIPS_BUTTON_TEXT),
            )
        except Exception:
            pass
        try:
            if int(translated_count or 0) > 0:
                message = (
                    f"Generated {int(translated_count)} {self._machine_translation_provider_label()} machine translation preview(s) across {int(piece_count or 0)} entries"
                )
                if error:
                    message = f"{message}. {self._compact_machine_translation_error(error)}"
                self.save_status_label.setText(message)
                QTimer.singleShot(2500, self._clear_review_save_status)
            elif error:
                self.save_status_label.setText(
                    f"Machine translation preview failed: {self._compact_machine_translation_error(error)}"
                )
        except Exception:
            pass

    def _selected_text_for_widget(self, widget):
        try:
            if isinstance(widget, QPlainTextEdit):
                return (widget.textCursor().selectedText() or "").replace("\u2029", "\n")
            if hasattr(widget, "selectedText"):
                return widget.selectedText() or ""
        except Exception:
            return ""
        return ""

    def _all_text_for_widget(self, widget):
        try:
            if isinstance(widget, QPlainTextEdit):
                return widget.toPlainText() or ""
            if hasattr(widget, "text"):
                return widget.text() or ""
        except Exception:
            return ""
        return ""

    def _copy_review_text(self, text):
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(str(text or ""))
        except Exception:
            pass

    def _paste_review_text(self, widget):
        try:
            if isinstance(widget, QPlainTextEdit) and not widget.isReadOnly():
                widget.paste()
        except Exception:
            pass

    def _select_all_review_text(self, widget):
        try:
            if isinstance(widget, QPlainTextEdit):
                widget.selectAll()
            elif hasattr(widget, "setSelection"):
                widget.setSelection(0, len(self._all_text_for_widget(widget)))
        except Exception:
            pass

    def _show_review_text_context_menu(
        self,
        widget,
        pos,
        edit_callback=None,
        translate_tooltip_callback=None,
        inject_machine_translation_callback=None,
        popup_widget=None,
        popup_pos=None,
    ):
        selected = self._selected_text_for_widget(widget).strip()
        has_selection = bool(selected)
        is_editable_editor = isinstance(widget, QPlainTextEdit) and not widget.isReadOnly()
        clipboard_text = ""
        if is_editable_editor:
            try:
                from PySide6.QtWidgets import QApplication
                clipboard_text = QApplication.clipboard().text() or ""
            except Exception:
                clipboard_text = ""
        target_lang = self._review_target_language()

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #1e1e2e; border: 1px solid #3a3a5e; border-radius: 4px;
                color: #e0e0e0; font-size: 9pt; padding: 4px; }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background: #3a3a5e; }
            QMenu::item:disabled { color: #555; }
        """)

        copy_action = menu.addAction("Copy")
        copy_action.setShortcut("Ctrl+C")
        copy_action.setEnabled(has_selection)
        copy_action.triggered.connect(lambda: self._copy_review_text(selected))

        if is_editable_editor:
            paste_action = menu.addAction("Paste")
            paste_action.setShortcut("Ctrl+V")
            paste_action.setEnabled(bool(clipboard_text))
            paste_action.triggered.connect(lambda: self._paste_review_text(widget))

        select_all_action = menu.addAction("Select All")
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.setEnabled(bool(self._all_text_for_widget(widget)))
        select_all_action.triggered.connect(lambda: self._select_all_review_text(widget))

        if edit_callback is not None:
            menu.addSeparator()
            edit_action = menu.addAction("Edit Output")
            edit_action.triggered.connect(edit_callback)

        if translate_tooltip_callback is not None:
            menu.addSeparator()
            tooltip_action = menu.addAction(f"\U0001f310  Machine Translation \u2192 {target_lang}")
            tooltip_action.setEnabled(not self._tooltip_translation_running and bool(self._all_text_for_widget(widget).strip()))
            tooltip_action.triggered.connect(lambda _checked=False: translate_tooltip_callback())

        if inject_machine_translation_callback is not None:
            if translate_tooltip_callback is None:
                menu.addSeparator()
            inject_action = menu.addAction("\U0001f4e5  Inject Machine Translation")
            inject_action.triggered.connect(lambda _checked=False: inject_machine_translation_callback())

        self._review_text_context_menu = menu
        self._set_review_context_menu_open(True)
        menu.aboutToHide.connect(lambda m=menu: self._clear_review_text_context_menu(m))
        anchor_widget = popup_widget or widget
        anchor_pos = popup_pos if popup_pos is not None else pos
        menu.popup(anchor_widget.mapToGlobal(anchor_pos))

    def _clear_review_text_context_menu(self, menu):
        try:
            if getattr(self, "_review_text_context_menu", None) is menu:
                self._review_text_context_menu = None
            self._set_review_context_menu_open(False)
            menu.deleteLater()
        except Exception:
            pass

    def _target_editor(self, piece_index, row_index, text, editable=True, height=None):
        editor = QPlainTextEdit(str(text or ""))
        editor.setObjectName("SdlReviewTargetEdit")
        editor.setFrameShape(QFrame.NoFrame)
        editor.setFocusPolicy(Qt.StrongFocus)
        editor.setTabChangesFocus(True)
        editor_height = max(28, int(height or 38))
        editor.setFixedHeight(editor_height)
        editor.setReadOnly(not editable)
        editor.setMinimumWidth(0)
        editor.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        try:
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        except Exception:
            try:
                editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
            except Exception:
                pass
        try:
            editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        try:
            editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        editor.setToolTip(self._wrapped_tooltip(text))
        try:
            editor.viewport().setCursor(Qt.IBeamCursor)
        except Exception:
            pass
        editor.setContextMenuPolicy(Qt.CustomContextMenu)
        editor.customContextMenuRequested.connect(
            lambda pos, ed=editor: self._show_review_text_context_menu(ed, pos)
        )
        if editable:
            editor.textChanged.connect(
                lambda pi=piece_index, ri=row_index, ed=editor: self._schedule_target_edit(pi, ri, ed.toPlainText())
            )
        return editor

    def _target_display_widget(self, piece_index, row_index, text, editable=True, height=None):
        container_height = max(30, int(height or 38))
        return self._target_editor(piece_index, row_index, text, editable=editable, height=container_height)

    def closeEvent(self, event):
        try:
            self._save_current_review_scroll()
            current_row = self.piece_list.currentRow()
            current_page = self._piece_pages.get(current_row)
            browser = (
                current_page.findChild(QWidget, "SdlReviewNotepadBrowser")
                if current_page is not None else None
            )
            if browser is not None:
                self._capture_notepad_browser_html(
                    browser, current_row, only_dirty=False, flush=True
                )
            if self._edit_save_timer.isActive():
                self._edit_save_timer.stop()
            self._flush_target_edits()
        except Exception:
            pass
        try:
            event.ignore()
        except Exception:
            pass
        self.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._sync_review_scroll_range()
            row = self._current_scroll_piece_row
            if row is not None:
                QTimer.singleShot(0, lambda row=row: self._restore_review_scroll(row))
        except Exception:
            pass

    def _refresh_review_stream_geometry(self, final=False):
        try:
            # _sync_review_scroll_range() already invalidates + activates the
            # current page layout and sizes the page/stack, so avoid doing the
            # same O(rows) work twice here.
            if final and self.rows_widget is not None:
                self.rows_widget.updateGeometry()
            self._sync_review_scroll_range()
            viewport = self.scroll.viewport()
            if viewport is not None:
                viewport.update()
        except Exception:
            pass

    def _apply_review_stream_geometry(self, page, layout, content_height):
        """Cheap geometry sync used while review rows stream in.

        All review rows have fixed heights, so the content height can be
        tracked incrementally by the caller instead of re-measuring the whole
        page layout on every batch (which is O(rows) and made streaming
        quadratic). Container heights are locked BEFORE the layout is
        activated so freshly added rows are never laid out into a stale,
        too-small page (the cause of the brief row-overlap flicker).
        """
        try:
            viewport_height = max(1, self.scroll.viewport().height())
            # Never shrink while streaming: alternating between the tracked
            # height and a measured height (e.g. from a machine-translation
            # preview patch landing mid-stream) would see-saw the page height
            # and visibly jiggle the scrollbar/rows.
            content_height = max(viewport_height, int(content_height or 0), page.minimumHeight())
            # Resize the stack AND the page synchronously, in the same tick.
            # Constraining only min/max leaves the stack at its old height
            # until the scroll area's asynchronous layout pass runs; in that
            # window the QStackedLayout squashes the (taller) page back down,
            # then it springs up again - which paints as vertical bouncing
            # between stream batches.
            page.setMinimumHeight(content_height)
            page.setMaximumHeight(content_height)
            self.rows_stack.setMinimumHeight(content_height)
            self.rows_stack.setMaximumHeight(content_height)
            self.rows_stack.resize(max(1, self.rows_stack.width()), content_height)
            page.resize(max(1, self.rows_stack.width()), content_height)
            if layout is not None:
                layout.activate()
            viewport = self.scroll.viewport()
            if viewport is not None:
                viewport.update()
        except Exception:
            pass

    def _bar_widget(self, length, max_length, color, align_right=False, width=180, max_bar_width=170):
        container = QWidget()
        container.setObjectName("SdlReviewBarContainer")
        container.setFixedWidth(max(12, int(width)))
        container.setStyleSheet("QWidget#SdlReviewBarContainer { background: transparent; }")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar_width = max(8, min(max(8, int(max_bar_width)), int(max(8, int(max_bar_width)) * (length / max(1, max_length)))))
        bar = QFrame()
        bar.setObjectName("SdlReviewLengthBar")
        bar.setFixedSize(bar_width, 10)
        bar.setStyleSheet(f"QFrame#SdlReviewLengthBar {{ background-color: {color}; border-radius: 5px; }}")
        if align_right:
            layout.addStretch(1)
            layout.addWidget(bar)
        else:
            layout.addWidget(bar)
            layout.addStretch(1)
        return container

    def _review_row_action_button(self, text, tooltip, callback=None, enabled=True, action_name=""):
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setEnabled(bool(enabled))
        button.setMinimumWidth(0)
        button.setProperty("sdlRowAction", True)
        if action_name:
            button.setProperty("sdl_action", action_name)
        button.setToolTip(tooltip)
        button.setStyleSheet(
            "QPushButton { background-color: #213447; color: #d7ecff; border: 1px solid #4d6f91; "
            "border-radius: 4px; padding: 4px 7px; font-size: 8pt; font-weight: bold; text-align: left; }"
            "QPushButton:hover { background-color: #2b4b66; border-color: #72acd9; }"
            "QPushButton:disabled { background-color: #1f2933; color: #718096; border-color: #354658; }"
        )
        if callback is not None:
            button.clicked.connect(lambda _checked=False: callback())
        return button

    def _apply_review_row_text_geometry(
        self,
        frame,
        source_widget,
        target_widget,
        row_height,
        source_lines=1,
        target_lines=1,
        tooltip_lines=0,
    ):
        two_column_layout = bool(frame.property("sdl_two_column_layout")) if frame is not None else False
        try:
            if two_column_layout:
                source_height, target_height = self._review_row_text_heights(
                    row_height,
                    source_lines=source_lines,
                    target_lines=target_lines,
                    tooltip_lines=tooltip_lines,
                )
                if source_widget is not None:
                    source_widget.setMaximumHeight(source_height)
                if target_widget is not None:
                    target_widget.setFixedHeight(target_height)
            else:
                if source_widget is not None:
                    source_widget.setMaximumHeight(max(24, row_height - 14))
                if target_widget is not None:
                    target_widget.setFixedHeight(max(30, row_height - 14))
        except Exception:
            pass

    def _review_row_controls_widget(
        self,
        piece,
        row_data,
        idx,
        row_model,
        max_len,
        source_bar,
        target_bar,
        dot,
        target_widget,
        target_editable,
        tooltip_translation,
        tooltip_pending,
    ):
        controls = QWidget()
        controls.setObjectName("SdlReviewTwoColumnControls")
        controls.setFixedWidth(250)
        controls.setStyleSheet("QWidget#SdlReviewTwoColumnControls { background: transparent; }")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(5)

        piece_index = piece["index"]
        controls_layout.addWidget(
            self._review_row_action_button(
                "↩️ Undo All Edits",
                "Restore this row to the target text loaded from the SDLXLIFF sidecar.",
                callback=lambda pi=piece_index, ri=idx, ed=target_widget: self._undo_all_target_edits(pi, ri, ed),
                enabled=target_editable,
                action_name="undo_all_edits",
            )
        )
        controls_layout.addWidget(
            self._review_row_action_button(
                "🌐 Generate Preview",
                "Generate a machine translation preview for this row.",
                callback=lambda pi=piece_index, ri=idx: self._translate_single_row_tooltip(pi, ri),
                enabled=bool(row_data.get("source")),
                action_name="generate_preview",
            )
        )
        inject_enabled = bool(tooltip_translation) and target_editable and not tooltip_pending
        controls_layout.addWidget(
            self._review_row_action_button(
                "📥 Inject Machine Translation",
                "Replace this row's output with the machine translation preview.",
                callback=lambda pi=piece_index, ri=idx, ed=target_widget: self._inject_current_machine_translation_to_target(pi, ri, ed),
                enabled=inject_enabled,
                action_name="inject_machine_translation",
            )
        )

        metrics = QWidget()
        metrics.setObjectName("SdlReviewTwoColumnMetrics")
        metrics.setStyleSheet("QWidget#SdlReviewTwoColumnMetrics { background: transparent; }")
        metrics_layout = QHBoxLayout(metrics)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(8)
        metrics_layout.addWidget(
            self._bar_widget(
                row_model.get("source_len", len(row_data.get("source", ""))),
                max_len,
                source_bar,
                align_right=True,
                width=84,
                max_bar_width=74,
            )
        )
        metrics_layout.addWidget(dot, 0, Qt.AlignVCenter)
        metrics_layout.addWidget(
            self._bar_widget(
                row_model.get("target_len", len(row_data.get("target", ""))),
                max_len,
                target_bar,
                align_right=False,
                width=84,
                max_bar_width=74,
            )
        )
        controls_layout.addWidget(metrics)
        return controls

    def _update_review_row_inject_button(self, frame, enabled):
        try:
            if frame is None:
                return
            for button in frame.findChildren(QPushButton):
                if button.property("sdl_action") == "inject_machine_translation":
                    button.setEnabled(bool(enabled))
        except Exception:
            pass

    def _text_label(
        self,
        text,
        missing=False,
        empty_placeholder=None,
        tooltip_translation=None,
        tooltip_pending=False,
        tooltip_state=None,
        tooltip_detail=None,
        translate_tooltip_callback=None,
        inject_machine_translation_callback=None,
    ):
        label = QLabel(
            text if text else (
                str(empty_placeholder)
                if empty_placeholder is not None
                else ("[missing]" if missing else "[empty]")
            )
        )
        label.setObjectName("SdlReviewSourceRawText")
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet(f"color: #cbd5e1; background: transparent; font-size: 10pt;")
        label.setToolTip(self._wrapped_tooltip(text))

        tooltip_translation = str(tooltip_translation or "").strip()
        tooltip_pending = bool(tooltip_pending)
        tooltip_state = str(tooltip_state or ("pending" if tooltip_pending else ("translation" if tooltip_translation else ""))).strip().lower()
        if tooltip_state not in {"pending", "error", "translation"}:
            tooltip_state = "pending" if tooltip_pending else ("translation" if tooltip_translation else "")
        tooltip_detail = str(tooltip_detail or "").strip()
        if not tooltip_translation and tooltip_state not in {"pending", "error"}:
            self._wire_source_preview_context_menu(
                label,
                [label],
                translate_tooltip_callback=translate_tooltip_callback,
                inject_machine_translation_callback=inject_machine_translation_callback,
            )
            return label

        container = QWidget()
        container.setObjectName("SdlReviewSourceText")
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        container.setToolTip(self._wrapped_tooltip(text))
        container.setStyleSheet("QWidget#SdlReviewSourceText { background: transparent; }")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addStretch(1)
        layout.addWidget(label)

        preview_text = tooltip_translation or self._machine_translation_pending_text()
        preview_label_text = preview_text
        translated_label = QLabel(preview_label_text)
        translated_label.setObjectName(
            "SdlReviewMachineTranslationPending" if tooltip_state in {"pending", "error"} else "SdlReviewMachineTranslation"
        )
        translated_label.setTextFormat(Qt.PlainText)
        translated_label.setWordWrap(True)
        translated_label.setMinimumWidth(0)
        translated_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        translated_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        translated_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if tooltip_state in {"pending", "error"}:
            translated_label.setToolTip(
                self._wrapped_tooltip(tooltip_detail)
                if tooltip_detail
                else f"Machine translation preview is being generated with {self._machine_translation_provider_label()}."
            )
            translated_label.setStyleSheet(
                "QLabel#SdlReviewMachineTranslationPending { "
                "color: #d8c99b; background: rgba(54, 45, 23, 180); "
                "border: 1px dashed #8a6f2a; border-left: 3px solid #d39e00; "
                "border-radius: 4px; padding: 5px 8px; font-size: 8pt; "
                "}"
            )
        else:
            translated_label.setToolTip(self._wrapped_tooltip(tooltip_translation))
            translated_label.setStyleSheet(
                "QLabel#SdlReviewMachineTranslation { "
                "color: #b6c7dc; background: rgba(23, 37, 54, 185); "
                "border: 1px solid #37536d; border-left: 3px solid #5aa7d8; "
                "border-radius: 4px; padding: 3px 7px; font-size: 7pt; "
                "}"
            )
        layout.addWidget(translated_label)
        layout.addStretch(1)
        self._wire_source_preview_context_menu(
            label,
            [container, label, translated_label],
            translate_tooltip_callback=translate_tooltip_callback,
            inject_machine_translation_callback=None if tooltip_state in {"pending", "error"} else inject_machine_translation_callback,
        )
        return container

    def _wire_source_preview_context_menu(
        self,
        text_widget,
        anchors,
        translate_tooltip_callback=None,
        inject_machine_translation_callback=None,
    ):
        for anchor in anchors or []:
            if anchor is None:
                continue
            if anchor.property("sdl_source_context_menu_wired"):
                try:
                    anchor.customContextMenuRequested.disconnect()
                except Exception:
                    pass
            anchor.setContextMenuPolicy(Qt.CustomContextMenu)
            anchor.customContextMenuRequested.connect(
                lambda pos, text_widget=text_widget, anchor=anchor: self._show_review_text_context_menu(
                    text_widget,
                    pos,
                    translate_tooltip_callback=translate_tooltip_callback,
                    inject_machine_translation_callback=inject_machine_translation_callback,
                    popup_widget=anchor,
                    popup_pos=pos,
                )
            )
            anchor.setProperty("sdl_source_context_menu_wired", True)

    def _review_row_height(self, source_text, target_text, tooltip_translation=None, tooltip_pending=False, tooltip_preview_text=None):
        return self._review_row_height_for_width(
            source_text,
            target_text,
            tooltip_translation,
            tooltip_pending,
            self._review_render_viewport_width(),
            two_column_layout=bool(getattr(self, "_two_column_layout_enabled", True)),
            tooltip_preview_text=tooltip_preview_text,
        )

    def _review_row_source_widget(self, frame):
        try:
            if frame is None:
                return None
            source_container = frame.findChild(QWidget, "SdlReviewSourceText")
            if source_container is not None:
                return source_container
            source_label = frame.findChild(QLabel, "SdlReviewSourceRawText")
            if source_label is not None:
                return source_label
            grid = frame.layout()
            if isinstance(grid, QGridLayout):
                item = grid.itemAtPosition(0, 1)
                return item.widget() if item is not None else None
        except Exception:
            return None
        return None

    def _review_row_target_widget(self, frame):
        try:
            if frame is None:
                return None
            target = frame.findChild(QPlainTextEdit, "SdlReviewTargetEdit")
            if target is not None:
                return target
            grid = frame.layout()
            if isinstance(grid, QGridLayout):
                item = grid.itemAtPosition(0, 5)
                return item.widget() if item is not None else None
        except Exception:
            return None
        return None

    def _replace_review_row_source_widget(self, frame, source_widget):
        try:
            if frame is None or source_widget is None:
                return False
            if bool(frame.property("sdl_two_column_layout")):
                content = frame.findChild(QWidget, "SdlReviewTwoColumnText")
                content_layout = content.layout() if content is not None else None
                if content_layout is None:
                    return False
                old_widget = self._review_row_source_widget(frame)
                if old_widget is not None:
                    content_layout.removeWidget(old_widget)
                    old_widget.hide()
                    old_widget.setParent(None)
                    old_widget.deleteLater()
                content_layout.insertWidget(0, source_widget)
                return True
            grid = frame.layout()
            if not isinstance(grid, QGridLayout):
                return False
            old_item = grid.itemAtPosition(0, 1)
            old_widget = old_item.widget() if old_item is not None else None
            if old_widget is not None:
                grid.removeWidget(old_widget)
                old_widget.hide()
                old_widget.setParent(None)
                old_widget.deleteLater()
            grid.addWidget(source_widget, 0, 1)
            return True
        except Exception:
            return False

    def _save_notepad_editor_now(self, browser=None, piece_index=None):
        if browser is not None and piece_index is not None:
            self._capture_notepad_browser_html(
                browser, int(piece_index), only_dirty=False, flush=True
            )
            return
        try:
            if self._edit_save_timer.isActive():
                self._edit_save_timer.stop()
            self._flush_target_edits()
        except Exception as exc:
            self.save_status_label.setText(f"Save failed: {exc}")

    def _notepad_browser_base_url(self, piece):
        output_path = self._output_path_for_piece(piece)
        base_dir = os.path.dirname(output_path) if output_path else (self.output_dir or "")
        return QUrl.fromLocalFile(os.path.join(os.path.abspath(base_dir or "."), ""))

    def _notepad_renderable_document_html(self, piece, html_text):
        """Resolve workspace image references for display without changing saved HTML."""
        document = str(html_text or "")
        try:
            from bs4 import BeautifulSoup

            rename_map = {}
            try:
                with open(
                    os.path.join(self.output_dir or "", "image_rename_map.json"),
                    "r",
                    encoding="utf-8",
                ) as handle:
                    loaded_map = json.load(handle)
                if isinstance(loaded_map, dict):
                    rename_map = {
                        os.path.basename(str(original)).casefold(): os.path.basename(str(renamed))
                        for original, renamed in loaded_map.items()
                        if original and renamed
                    }
            except (OSError, ValueError, TypeError):
                pass

            output_path = self._output_path_for_piece(piece) or ""
            document_dir = os.path.dirname(os.path.abspath(output_path)) if output_path else ""
            output_dir = os.path.abspath(self.output_dir) if self.output_dir else document_dir
            image_dirs = [
                os.path.join(output_dir, "images"),
                os.path.join(output_dir, "Images"),
                os.path.join(output_dir, "translated_images"),
            ]

            def _resolved_file_url(reference):
                raw_reference = html_lib.unescape(str(reference or "")).strip()
                if not raw_reference or raw_reference.startswith("//"):
                    return ""
                is_windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", raw_reference))
                if not is_windows_absolute:
                    scheme = QUrl(raw_reference).scheme().casefold()
                    if scheme in {"http", "https", "data", "blob", "file", "qrc"}:
                        return ""

                clean_reference = unquote(re.split(r"[?#]", raw_reference, maxsplit=1)[0])
                filesystem_reference = clean_reference.replace("/", os.sep)
                basename = os.path.basename(filesystem_reference)
                renamed_basename = rename_map.get(basename.casefold(), "")
                basenames = [name for name in (renamed_basename, basename) if name]
                candidates = []
                if os.path.isabs(filesystem_reference) or is_windows_absolute:
                    candidates.append(os.path.normpath(filesystem_reference))
                else:
                    for root in (document_dir, output_dir):
                        if root:
                            candidates.append(os.path.normpath(os.path.join(root, filesystem_reference)))
                for image_dir in image_dirs:
                    for name in basenames:
                        candidates.append(os.path.join(image_dir, name))

                seen = set()
                for candidate in candidates:
                    normalized = os.path.normcase(os.path.abspath(candidate))
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    if os.path.isfile(candidate):
                        return QUrl.fromLocalFile(os.path.abspath(candidate)).toString()
                return ""

            soup = BeautifulSoup(document, "html.parser")
            media_attributes = (
                ("img", "src"),
                ("image", "href"),
                ("image", "xlink:href"),
                ("object", "data"),
                ("video", "poster"),
            )
            for tag_name, attribute in media_attributes:
                marker = "data-sdl-notepad-original-" + attribute.replace(":", "-")
                for element in soup.find_all(tag_name):
                    original = element.get(attribute)
                    resolved = _resolved_file_url(original)
                    if not resolved:
                        continue
                    element[marker] = original
                    element[attribute] = resolved
            return str(soup)
        except Exception:
            return document

    def _set_notepad_browser_html(self, browser, piece, html_text):
        browser._sdl_document_prefix = self._notepad_document_prefix(html_text)
        browser.setHtml(
            self._notepad_renderable_document_html(piece, html_text),
            self._notepad_browser_base_url(piece),
        )

    def _install_notepad_browser_editing(self, browser):
        """Expose text-only edit islands while keeping the HTML tree immutable."""
        script = r"""
            (() => {
                if (!document.body) return false;
                document.designMode = 'off';
                window.__sdlNotepadDirty = false;
                if (window.__sdlNotepadGuardsInstalled) return true;
                window.__sdlNotepadGuardsInstalled = true;

                const EDIT_ATTR = 'data-sdl-notepad-text';
                const PLACEHOLDER_ATTR = 'data-sdl-notepad-placeholder';
                const BREAK_ATTR = 'data-sdl-notepad-break';
                const USER_TAG_ATTR = 'data-sdl-notepad-user-tag';
                const LOADED_EXTRA_BREAK_ATTR = 'data-sdl-notepad-loaded-extra-break';
                const USER_TAG_CONTAINER_ATTR = 'data-sdl-notepad-user-tag-container';
                const USER_BLOCK_ATTR = 'data-sdl-notepad-user-block';
                const SOURCE_EMPTY_ATTR = 'data-sdl-notepad-source-empty';
                const ORIGINAL_TEXT_CONTAINER_ATTR = 'data-sdl-notepad-original-had-text';
                const USER_EMPTY_CONTAINER_ATTR = 'data-sdl-notepad-user-empty-container';
                const SOURCE_ATTR = 'data-sdl-notepad-source';
                const ROW_ATTR = 'data-sdl-notepad-row-index';
                const STATUS_ATTR = 'data-sdl-notepad-status';
                const ACTIVE_CONTAINER_ATTR = 'data-sdl-notepad-active-container';
                const MULTILINE_CONTAINER_ATTR = 'data-sdl-notepad-multiline-container';
                const WHOLE_SELECTION_ATTR = 'data-sdl-notepad-whole-selection';
                const ORIGINAL_EDITABLE_ATTR = 'data-sdl-notepad-original-editable';
                const JUMP_HIGHLIGHT_ATTR = 'data-sdl-notepad-jump-highlight';
                const NORMALIZED_PLACEHOLDER_ATTR = 'data-sdl-notepad-normalized-placeholder';
                const TEXT_UNIT_SELECTOR = 'h1,h2,h3,h4,h5,h6,p,li,hr,div.u';
                const NESTED_TEXT_UNIT_SELECTOR = 'h1,h2,h3,h4,h5,h6,p,li,hr';
                const ALLOWED_INLINE_TAGS = new Set(['STRONG', 'EM', 'U', 'B', 'I', 'BR']);
                const blockedParents = 'script,style,noscript,template,textarea,select,option,svg,math';
                const reviewTextElements = () => Array.from(
                    document.body.querySelectorAll(TEXT_UNIT_SELECTOR)
                ).filter(candidate => candidate.tagName.toUpperCase() !== 'DIV'
                    || !candidate.querySelector(NESTED_TEXT_UNIT_SELECTOR));
                const reviewTextElement = node => {
                    const element = node && node.nodeType === Node.ELEMENT_NODE
                        ? node : node && node.parentElement;
                    if (!element) return null;
                    const candidate = element.closest(TEXT_UNIT_SELECTOR);
                    if (!candidate) return null;
                    if (candidate.tagName.toUpperCase() === 'DIV'
                            && candidate.querySelector(NESTED_TEXT_UNIT_SELECTOR)) return null;
                    return candidate;
                };
                const editableHost = node => {
                    const element = node && node.nodeType === Node.ELEMENT_NODE
                        ? node : node && node.parentElement;
                    return element ? element.closest('[' + EDIT_ATTR + ']') : null;
                };
                const selectionInside = host => {
                    const selection = window.getSelection();
                    return !!(selection && selection.rangeCount && host
                        && host.contains(selection.anchorNode)
                        && host.contains(selection.focusNode));
                };
                const selectionOffsets = host => {
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount || !host
                            || editableHost(selection.anchorNode) !== host
                            || editableHost(selection.focusNode) !== host
                            || !selectionInside(host)) return null;
                    const range = selection.getRangeAt(0);
                    const before = document.createRange();
                    before.selectNodeContents(host);
                    before.setEnd(range.startContainer, range.startOffset);
                    const start = before.toString().length;
                    return {
                        host,
                        start,
                        end: start + range.toString().length
                    };
                };
                const restoreSelection = snapshot => {
                    const host = snapshot && snapshot.host;
                    if (!host || !host.isConnected) return false;
                    const nodes = [];
                    const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
                    while (walker.nextNode()) nodes.push(walker.currentNode);
                    if (!nodes.length) {
                        // A break-only host already has a firstChild (`<br>`),
                        // whose nodeValue is null. Keep the actual caret text
                        // node instead of accidentally treating that element
                        // like text during Up/Down row navigation.
                        const caretNode = document.createTextNode('');
                        host.appendChild(caretNode);
                        nodes.push(caretNode);
                    }
                    const boundaryAt = rawOffset => {
                        let offset = Math.max(0, Number(rawOffset) || 0);
                        for (const node of nodes) {
                            const length = String(node.nodeValue || '').length;
                            if (offset <= length) return [node, offset];
                            offset -= length;
                        }
                        const last = nodes[nodes.length - 1];
                        return [last, String(last.nodeValue || '').length];
                    };
                    const start = boundaryAt(snapshot.start);
                    const end = boundaryAt(snapshot.end);
                    const range = document.createRange();
                    range.setStart(start[0], start[1]);
                    range.setEnd(end[0], end[1]);
                    try { host.focus({preventScroll: true}); }
                    catch (_error) { host.focus(); }
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    return true;
                };
                const selectionBoundary = host => {
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount || !selection.isCollapsed
                            || !selectionInside(host)) {
                        return { collapsed: false, atStart: false, atEnd: false };
                    }
                    const range = selection.getRangeAt(0);
                    const before = range.cloneRange();
                    before.selectNodeContents(host);
                    before.setEnd(range.startContainer, range.startOffset);
                    const after = range.cloneRange();
                    after.selectNodeContents(host);
                    after.setStart(range.endContainer, range.endOffset);
                    return {
                        collapsed: true,
                        atStart: before.toString().length === 0,
                        atEnd: after.toString().length === 0
                    };
                };
                const selectionAtContainerEnd = (host, container) => {
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount || !selection.isCollapsed
                            || !selectionInside(host) || !container) return false;
                    try {
                        const selected = selection.getRangeAt(0);
                        const tail = selected.cloneRange();
                        tail.setEnd(container, container.childNodes.length);
                        const remaining = tail.cloneContents();
                        if (String(remaining.textContent || '').trim()) return false;
                        return !remaining.querySelector(
                            'br,img,image,object,video,audio,svg,math,hr,iframe'
                        );
                    } catch (_error) {
                        return false;
                    }
                };
                const markDirty = () => { window.__sdlNotepadDirty = true; };
                const userTagContainer = node => {
                    const element = node && node.nodeType === Node.ELEMENT_NODE
                        ? node : node && node.parentElement;
                    if (!element) return null;
                    const candidate = element.closest(
                        'p,h1,h2,h3,h4,h5,h6,li,td,th,caption,figcaption,blockquote,div.u'
                    );
                    if (!candidate) return null;
                    if (candidate.tagName.toUpperCase() === 'DIV'
                            && candidate.querySelector(NESTED_TEXT_UNIT_SELECTOR)) return null;
                    return candidate;
                };
                const navigationContainer = host => {
                    if (!host) return null;
                    const rowContainer = host.closest(
                        'p,h1,h2,h3,h4,h5,h6,li,td,th,caption,figcaption,blockquote'
                    );
                    if (rowContainer) return rowContainer;
                    const broadContainer = host.closest('div,section,article');
                    if (broadContainer && !broadContainer.querySelector(
                            'p,h1,h2,h3,h4,h5,h6,li,td,th,caption,' +
                            'figcaption,blockquote')) {
                        return broadContainer;
                    }
                    return host;
                };
                const navigationContainers = () => {
                    const containers = [];
                    const seen = new Set();
                    for (const host of document.body.querySelectorAll(
                            '[' + EDIT_ATTR + ']')) {
                        const container = navigationContainer(host);
                        if (container && !seen.has(container)) {
                            seen.add(container);
                            containers.push(container);
                        }
                    }
                    return containers;
                };
                // Navigation treats every <br> as a visual line boundary. Only
                // explicitly inserted and legacy target-only breaks remain
                // deletable; source EPUB breaks stay
                // protected even though Up/Down can move across them.
                const navigationBreaksIn = container => Array.from(
                    container.querySelectorAll('br')
                );
                const trailingUserBreakIn = container => {
                    if (!container) return null;
                    const breaks = Array.from(container.querySelectorAll(
                        'br[' + USER_TAG_ATTR + '],br['
                            + LOADED_EXTRA_BREAK_ATTR + ']'
                    ));
                    const lineBreak = breaks.length ? breaks[breaks.length - 1] : null;
                    if (!lineBreak) return null;
                    try {
                        const afterBreak = document.createRange();
                        afterBreak.selectNodeContents(container);
                        afterBreak.setStartAfter(lineBreak);
                        return afterBreak.toString().trim() ? null : lineBreak;
                    } catch (_error) {
                        return null;
                    }
                };
                const visualLineCount = container => {
                    if (!container || !container.isConnected) return 0;
                    try {
                        const range = document.createRange();
                        range.selectNodeContents(container);
                        const lineTops = [];
                        for (const rect of Array.from(range.getClientRects())) {
                            // Ignore zero-width BR/caret fragments. A user BR
                            // is handled explicitly below, including a trailing
                            // break whose new line is still empty.
                            if (rect.width < 0.5 || rect.height < 0.5) continue;
                            if (!lineTops.some(top => Math.abs(top - rect.top) < 2)) {
                                lineTops.push(rect.top);
                            }
                        }
                        return lineTops.length;
                    } catch (_error) {
                        return 0;
                    }
                };
                const refreshUnderlineStart = container => {
                    if (!container || !container.isConnected) return;
                    const hosts = Array.from(container.querySelectorAll(
                        '[' + EDIT_ATTR + ']'
                    ));
                    for (const host of hosts) {
                        host.style.removeProperty('--sdl-notepad-leading-space-width');
                    }
                    const firstHost = hosts.find(host => host.textContent.length) || null;
                    const leading = firstHost
                        ? String(firstHost.textContent || '').match(/^\s+/u)
                        : null;
                    if (!firstHost || !leading || !leading[0].length) return;
                    try {
                        let remaining = leading[0].length;
                        let endNode = null;
                        let endOffset = 0;
                        const walker = document.createTreeWalker(
                            firstHost, NodeFilter.SHOW_TEXT
                        );
                        while (walker.nextNode()) {
                            const node = walker.currentNode;
                            const length = String(node.nodeValue || '').length;
                            if (remaining <= length) {
                                endNode = node;
                                endOffset = remaining;
                                break;
                            }
                            remaining -= length;
                        }
                        if (!endNode) return;
                        const range = document.createRange();
                        range.setStart(firstHost, 0);
                        range.setEnd(endNode, endOffset);
                        const width = Math.max(0, range.getBoundingClientRect().width);
                        if (width > 0) {
                            firstHost.style.setProperty(
                                '--sdl-notepad-leading-space-width', width + 'px'
                            );
                        }
                    } catch (_error) {}
                };
                const refreshNavigationContainerStyle = container => {
                    if (!container || !container.isConnected) return false;
                    refreshUnderlineStart(container);
                    const hasUserLineBreak = !!container.querySelector(
                        'br[' + USER_TAG_ATTR + ']'
                    );
                    const multiline = hasUserLineBreak || visualLineCount(container) > 1;
                    if (multiline) {
                        container.setAttribute(MULTILINE_CONTAINER_ATTR, '1');
                    } else {
                        container.removeAttribute(MULTILINE_CONTAINER_ATTR);
                    }
                    return multiline;
                };
                let activeNavigationContainer = null;
                let wholeSelectionContainer = null;
                let settingWholeSelection = false;
                const clearWholeContainerSelection = () => {
                    const container = wholeSelectionContainer;
                    wholeSelectionContainer = null;
                    if (!container || !container.isConnected
                            || !container.hasAttribute(WHOLE_SELECTION_ATTR)) return;
                    const original = container.getAttribute(WHOLE_SELECTION_ATTR);
                    container.removeAttribute(WHOLE_SELECTION_ATTR);
                    if (original === '__sdl_missing__') {
                        container.removeAttribute('contenteditable');
                    } else {
                        container.setAttribute('contenteditable', original);
                    }
                };
                const selectionCoversContainer = (selection, container) => {
                    if (!selection || !selection.rangeCount || !container) return false;
                    const range = selection.getRangeAt(0);
                    return range.startContainer === container && range.startOffset === 0
                        && range.endContainer === container
                        && range.endOffset === container.childNodes.length;
                };
                const setActiveNavigationContainer = node => {
                    const host = editableHost(node);
                    const container = navigationContainer(host);
                    if (container === activeNavigationContainer) {
                        refreshNavigationContainerStyle(container);
                        return container;
                    }
                    if (activeNavigationContainer && activeNavigationContainer.isConnected) {
                        activeNavigationContainer.removeAttribute(ACTIVE_CONTAINER_ATTR);
                    }
                    activeNavigationContainer = container || null;
                    if (activeNavigationContainer) {
                        activeNavigationContainer.setAttribute(ACTIVE_CONTAINER_ATTR, '1');
                        refreshNavigationContainerStyle(activeNavigationContainer);
                    }
                    return activeNavigationContainer;
                };
                document.addEventListener('focusin', event => {
                    const focusedContainer = navigationContainer(editableHost(event.target));
                    if (wholeSelectionContainer
                            && focusedContainer !== wholeSelectionContainer) {
                        clearWholeContainerSelection();
                    }
                    setActiveNavigationContainer(event.target);
                }, true);
                document.addEventListener('mousedown', event => {
                    if (wholeSelectionContainer && !settingWholeSelection) {
                        clearWholeContainerSelection();
                    }
                    const element = event.target && event.target.nodeType === Node.ELEMENT_NODE
                        ? event.target : event.target && event.target.parentElement;
                    if (!element) return;
                    const directHost = editableHost(element);
                    const container = navigationContainer(directHost)
                        || userTagContainer(element);
                    const lineBreak = trailingUserBreakIn(container);
                    const owner = editableHost(lineBreak);
                    if (!container || !lineBreak || !owner) return;
                    const containerRect = container.getBoundingClientRect();
                    const ownerStyle = window.getComputedStyle(owner);
                    const fontSize = Number.parseFloat(ownerStyle.fontSize) || 16;
                    const parsedLineHeight = Number.parseFloat(ownerStyle.lineHeight);
                    const lineHeight = Number.isFinite(parsedLineHeight)
                        ? parsedLineHeight : fontSize * 1.2;
                    const inTrailingBlankLine = event.clientX >= containerRect.left
                        && event.clientX <= containerRect.right
                        && event.clientY >= containerRect.bottom - (lineHeight * 1.25)
                        && event.clientY <= containerRect.bottom + 2;
                    if (!inTrailingBlankLine) return;
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    try { owner.focus({preventScroll: true}); }
                    catch (_error) { owner.focus(); }
                    const range = document.createRange();
                    range.setStartAfter(lineBreak);
                    range.collapse(true);
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    setActiveNavigationContainer(owner);
                }, true);
                document.addEventListener('selectionchange', () => {
                    const selection = window.getSelection();
                    if (!settingWholeSelection && wholeSelectionContainer
                            && !selectionCoversContainer(
                                selection, wholeSelectionContainer
                            )) {
                        clearWholeContainerSelection();
                    }
                    if (selection && selection.rangeCount) {
                        const selectionHost = editableHost(selection.anchorNode);
                        // Programmatic focus can briefly leave Chromium's old
                        // selection behind. Do not let that stale range move
                        // the paragraph marker away from the focused host.
                        if (selectionHost && document.activeElement === selectionHost) {
                            setActiveNavigationContainer(selectionHost);
                        }
                    }
                });
                document.addEventListener('dblclick', event => {
                    const host = editableHost(event.target);
                    const container = navigationContainer(host);
                    if (!host || !container) return;
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    setActiveNavigationContainer(host);
                    clearWholeContainerSelection();
                    const originalEditable = container.hasAttribute('contenteditable')
                        ? container.getAttribute('contenteditable') : '__sdl_missing__';
                    container.setAttribute(WHOLE_SELECTION_ATTR, originalEditable);
                    container.setAttribute('contenteditable', 'true');
                    wholeSelectionContainer = container;
                    const range = document.createRange();
                    range.selectNodeContents(container);
                    const selection = window.getSelection();
                    settingWholeSelection = true;
                    selection.removeAllRanges();
                    selection.addRange(range);
                    settingWholeSelection = false;
                    window.__sdlNotepadFormatSelection = null;
                }, true);
                const navigationPosition = (host, container) => {
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount || !container) {
                        return {segment: 0, column: 0};
                    }
                    const caret = selection.getRangeAt(0).cloneRange();
                    caret.collapse(true);
                    const breaks = navigationBreaksIn(container);
                    let segment = 0;
                    try {
                        const beforeCaret = document.createRange();
                        beforeCaret.selectNodeContents(container);
                        beforeCaret.setEnd(caret.startContainer, caret.startOffset);
                        for (const lineBreak of breaks) {
                            if (beforeCaret.intersectsNode(lineBreak)) segment += 1;
                        }
                    } catch (_error) {
                        segment = 0;
                    }
                    let column = 0;
                    try {
                        const lineRange = document.createRange();
                        if (segment > 0) lineRange.setStartAfter(breaks[segment - 1]);
                        else lineRange.setStart(container, 0);
                        lineRange.setEnd(caret.startContainer, caret.startOffset);
                        column = lineRange.toString().length;
                    } catch (_error) {
                        const snapshot = selectionOffsets(host);
                        column = snapshot ? snapshot.start : 0;
                    }
                    return {segment, column};
                };
                const restoreNavigationPosition = (container, rawSegment, rawColumn) => {
                    if (!container || !container.isConnected) return false;
                    const breaks = navigationBreaksIn(container);
                    const segment = Math.max(
                        0, Math.min(breaks.length, Number(rawSegment) || 0)
                    );
                    const segmentRange = document.createRange();
                    segmentRange.selectNodeContents(container);
                    if (segment > 0) segmentRange.setStartAfter(breaks[segment - 1]);
                    if (segment < breaks.length) segmentRange.setEndBefore(breaks[segment]);

                    const textNodes = [];
                    const walker = document.createTreeWalker(
                        container, NodeFilter.SHOW_TEXT
                    );
                    while (walker.nextNode()) {
                        const node = walker.currentNode;
                        if (!editableHost(node)) continue;
                        try {
                            if (segmentRange.intersectsNode(node)) textNodes.push(node);
                        } catch (_error) {}
                    }

                    let targetNode = null;
                    let targetOffset = 0;
                    let remaining = Math.max(0, Number(rawColumn) || 0);
                    for (const node of textNodes) {
                        const length = String(node.nodeValue || '').length;
                        targetNode = node;
                        targetOffset = Math.min(remaining, length);
                        if (remaining <= length) break;
                        remaining -= length;
                    }

                    const range = document.createRange();
                    let targetHost = editableHost(targetNode);
                    if (targetNode) {
                        range.setStart(targetNode, targetOffset);
                    } else if (segment > 0) {
                        const previousBreak = breaks[segment - 1];
                        targetHost = editableHost(previousBreak);
                        range.setStartAfter(previousBreak);
                    } else if (breaks.length) {
                        targetHost = editableHost(breaks[0]);
                        let caretNode = breaks[0].previousSibling;
                        if (!caretNode || caretNode.nodeType !== Node.TEXT_NODE) {
                            caretNode = document.createTextNode('');
                            breaks[0].parentNode.insertBefore(caretNode, breaks[0]);
                        }
                        range.setStart(
                            caretNode,
                            String(caretNode.nodeValue || '').length
                        );
                    } else {
                        targetHost = container.matches('[' + EDIT_ATTR + ']')
                            ? container
                            : container.querySelector('[' + EDIT_ATTR + ']');
                        if (!targetHost) return false;
                        const caretNode = document.createTextNode('');
                        targetHost.insertBefore(caretNode, targetHost.firstChild);
                        range.setStart(caretNode, 0);
                    }
                    if (!targetHost) return false;
                    range.collapse(true);
                    setActiveNavigationContainer(targetHost);
                    try { targetHost.focus({preventScroll: true}); }
                    catch (_error) { targetHost.focus(); }
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    return true;
                };
                const moveVertical = (host, upwards) => {
                    const container = navigationContainer(host);
                    if (!container) return false;
                    const containers = navigationContainers();
                    const containerIndex = containers.indexOf(container);
                    if (containerIndex < 0) return false;
                    const position = navigationPosition(host, container);
                    const breaks = navigationBreaksIn(container);
                    let targetContainer = container;
                    let targetSegment = position.segment + (upwards ? -1 : 1);
                    if (targetSegment < 0) {
                        targetContainer = containers[containerIndex - 1];
                        if (!targetContainer) return false;
                        targetSegment = navigationBreaksIn(targetContainer).length;
                    } else if (targetSegment > breaks.length) {
                        targetContainer = containers[containerIndex + 1];
                        if (!targetContainer) return false;
                        targetSegment = 0;
                    }
                    return restoreNavigationPosition(
                        targetContainer, targetSegment, position.column
                    );
                };
                const refreshUserTagIndicator = container => {
                    if (!container || !container.isConnected) return;
                    const userCreatedBlock = container.hasAttribute(USER_BLOCK_ATTR);
                    const filledSourceEmptySlot = container.hasAttribute(SOURCE_EMPTY_ATTR)
                        && !!container.textContent.trim();
                    if (userCreatedBlock
                            || filledSourceEmptySlot
                            || container.querySelector('br[' + USER_TAG_ATTR + ']')) {
                        container.setAttribute(USER_TAG_CONTAINER_ATTR, '1');
                    } else {
                        container.removeAttribute(USER_TAG_CONTAINER_ATTR);
                    }
                };
                const removeFilledEmptySourcePlaceholders = container => {
                    if (!container || !container.isConnected
                            || !container.hasAttribute(SOURCE_EMPTY_ATTR)
                            || !container.textContent.trim()) return false;
                    const placeholders = Array.from(container.querySelectorAll(
                        'br:not([' + USER_TAG_ATTR + ']):not(['
                            + LOADED_EXTRA_BREAK_ATTR + '])'
                    ));
                    placeholders.forEach(lineBreak => lineBreak.remove());
                    return placeholders.length > 0;
                };
                const refreshUserEmptyIndicator = container => {
                    if (!container || !container.isConnected) return;
                    if (container.hasAttribute(ORIGINAL_TEXT_CONTAINER_ATTR)
                            && !container.textContent.trim()) {
                        container.setAttribute(USER_EMPTY_CONTAINER_ATTR, '1');
                    } else {
                        container.removeAttribute(USER_EMPTY_CONTAINER_ATTR);
                    }
                };
                let breakUndoStack = [];
                let breakRedoStack = [];
                let breakUndoReady = false;
                let breakRedoReady = false;
                let enterKeyInsertionHost = null;
                let injectionUndoStack = [];
                let injectionRedoStack = [];
                let injectionUndoReady = false;
                let injectionRedoReady = false;
                const recordInjectionEdit = action => {
                    if (!injectionUndoReady) injectionUndoStack = [];
                    injectionUndoStack.push(action);
                    injectionRedoStack = [];
                    injectionUndoReady = true;
                    injectionRedoReady = false;
                    breakUndoReady = false;
                    breakRedoReady = false;
                };
                const replayInjectionHistory = redo => {
                    const source = redo ? injectionRedoStack : injectionUndoStack;
                    if (!source.length) return false;
                    const action = source[source.length - 1];
                    if (!action.container || !action.container.isConnected) return false;
                    source.pop();
                    action.container.innerHTML = redo
                        ? action.afterHtml : action.beforeHtml;
                    (redo ? injectionUndoStack : injectionRedoStack).push(action);
                    injectionUndoReady = injectionUndoStack.length > 0;
                    injectionRedoReady = injectionRedoStack.length > 0;
                    refreshUserTagIndicator(action.container);
                    refreshUserEmptyIndicator(action.container);
                    refreshNavigationContainerStyle(action.container);
                    const hosts = action.container.querySelectorAll(
                        '[' + EDIT_ATTR + ']'
                    );
                    const host = hosts.length ? hosts[hosts.length - 1] : null;
                    if (host) {
                        try { host.focus({preventScroll: true}); }
                        catch (_error) { host.focus(); }
                        const selection = window.getSelection();
                        const range = document.createRange();
                        range.selectNodeContents(host);
                        range.collapse(false);
                        selection.removeAllRanges();
                        selection.addRange(range);
                    }
                    markDirty();
                    return true;
                };
                const recordBreakEdit = action => {
                    if (!breakUndoReady) breakUndoStack = [];
                    breakUndoStack.push(action);
                    breakRedoStack = [];
                    breakUndoReady = true;
                    breakRedoReady = false;
                    injectionUndoReady = false;
                    injectionRedoReady = false;
                };
                const insertRecordedBreak = action => {
                    if (!action.parent || !action.parent.isConnected) return false;
                    const reference = action.nextSibling
                        && action.nextSibling.parentNode === action.parent
                        ? action.nextSibling : null;
                    action.parent.insertBefore(action.node, reference);
                    return true;
                };
                const focusBreakAction = (action, after=true) => {
                    if (action.block) {
                        const targetHost = after ? action.host : action.focusBefore;
                        if (!targetHost || !targetHost.isConnected) return;
                        try { targetHost.focus({preventScroll: true}); }
                        catch (_error) { targetHost.focus(); }
                        const selection = window.getSelection();
                        const range = document.createRange();
                        range.selectNodeContents(targetHost);
                        range.collapse(false);
                        selection.removeAllRanges();
                        selection.addRange(range);
                        setActiveNavigationContainer(targetHost);
                        return;
                    }
                    const host = action.host && action.host.isConnected
                        ? action.host : editableHost(action.node);
                    if (!host) return;
                    try { host.focus({preventScroll: true}); }
                    catch (_error) { host.focus(); }
                    const selection = window.getSelection();
                    const range = document.createRange();
                    if (action.node.isConnected) {
                        if (after) range.setStartAfter(action.node);
                        else range.setStartBefore(action.node);
                    } else if (action.nextSibling
                            && action.nextSibling.parentNode === action.parent) {
                        range.setStartBefore(action.nextSibling);
                    } else {
                        range.selectNodeContents(action.parent);
                        range.collapse(false);
                    }
                    range.collapse(true);
                    selection.removeAllRanges();
                    selection.addRange(range);
                };
                const replayBreakHistory = redo => {
                    const source = redo ? breakRedoStack : breakUndoStack;
                    if (!source.length) return false;
                    const action = source.pop();
                    const shouldInsert = redo
                        ? action.kind === 'insert' : action.kind === 'delete';
                    if (shouldInsert) insertRecordedBreak(action);
                    else if (action.node.isConnected) action.node.remove();
                    (redo ? breakUndoStack : breakRedoStack).push(action);
                    breakUndoReady = breakUndoStack.length > 0;
                    breakRedoReady = breakRedoStack.length > 0;
                    refreshUserTagIndicator(action.indicatorContainer);
                    refreshUserEmptyIndicator(action.indicatorContainer);
                    refreshNavigationContainerStyle(action.indicatorContainer);
                    focusBreakAction(action, shouldInsert);
                    markDirty();
                    return true;
                };
                const insertBreak = host => {
                    const selection = window.getSelection();
                    if (!host || !selection || !selection.rangeCount
                            || !selectionInside(host)) return false;
                    const range = selection.getRangeAt(0);
                    range.deleteContents();
                    const lineBreak = document.createElement('br');
                    lineBreak.setAttribute(USER_TAG_ATTR, 'br');
                    range.insertNode(lineBreak);
                    const indicatorContainer = userTagContainer(lineBreak);
                    const action = {
                        kind: 'insert',
                        node: lineBreak,
                        parent: lineBreak.parentNode,
                        nextSibling: lineBreak.nextSibling,
                        host,
                        indicatorContainer
                    };
                    recordBreakEdit(action);
                    refreshUserTagIndicator(indicatorContainer);
                    refreshUserEmptyIndicator(indicatorContainer);
                    refreshNavigationContainerStyle(indicatorContainer);
                    range.setStartAfter(lineBreak);
                    range.collapse(true);
                    selection.removeAllRanges();
                    selection.addRange(range);
                    host.removeAttribute(PLACEHOLDER_ATTR);
                    markDirty();
                    return true;
                };
                const insertEmptyParagraphAfter = (host, paragraph) => {
                    if (!host || !paragraph || paragraph.tagName.toUpperCase() !== 'P'
                            || !paragraph.parentNode) return false;
                    const block = paragraph.cloneNode(false);
                    for (const attribute of Array.from(block.attributes)) {
                        const name = String(attribute.name || '').toLowerCase();
                        if (name === 'id' || name.startsWith('data-sdl-notepad-')) {
                            block.removeAttribute(attribute.name);
                        }
                    }
                    block.setAttribute(USER_BLOCK_ATTR, '1');
                    const newHost = makeHost(null, true, '');
                    block.appendChild(newHost);
                    const parent = paragraph.parentNode;
                    const nextSibling = paragraph.nextSibling;
                    parent.insertBefore(block, nextSibling);
                    refreshUserTagIndicator(block);
                    const action = {
                        kind: 'insert',
                        block: true,
                        node: block,
                        parent,
                        nextSibling,
                        host: newHost,
                        focusBefore: host,
                        indicatorContainer: null
                    };
                    recordBreakEdit(action);
                    focusBreakAction(action, true);
                    markDirty();
                    return true;
                };
                const insertEnter = host => {
                    const container = navigationContainer(host);
                    if (container && container.tagName.toUpperCase() === 'P'
                            && selectionAtContainerEnd(host, container)) {
                        return insertEmptyParagraphAfter(host, container);
                    }
                    return insertBreak(host);
                };
                const adjacentBreakAtCaret = (host, backwards) => {
                    const selection = window.getSelection();
                    if (!host || !selection || !selection.rangeCount
                            || !selection.isCollapsed || !selectionInside(host)) return null;
                    const range = selection.getRangeAt(0);
                    let current = range.startContainer;
                    let candidate = null;
                    if (current.nodeType === Node.TEXT_NODE) {
                        const textLength = current.nodeValue.length;
                        if ((backwards && range.startOffset > 0)
                                || (!backwards && range.startOffset < textLength)) return null;
                    } else if (current.nodeType === Node.ELEMENT_NODE) {
                        candidate = backwards
                            ? current.childNodes[range.startOffset - 1]
                            : current.childNodes[range.startOffset];
                    }
                    while (!candidate && current && current !== host) {
                        candidate = backwards ? current.previousSibling : current.nextSibling;
                        current = current.parentNode;
                    }
                    while (candidate && candidate.nodeType === Node.ELEMENT_NODE
                            && candidate.tagName.toUpperCase() !== 'BR'
                            && candidate.childNodes.length) {
                        candidate = backwards
                            ? candidate.lastChild : candidate.firstChild;
                    }
                    return candidate && candidate.nodeType === Node.ELEMENT_NODE
                            && candidate.tagName.toUpperCase() === 'BR'
                            && host.contains(candidate)
                        ? candidate : null;
                };
                const adjacentEditableBreak = (host, backwards) => {
                    const lineBreak = adjacentBreakAtCaret(host, backwards);
                    return lineBreak && (
                            lineBreak.hasAttribute(USER_TAG_ATTR)
                            || lineBreak.hasAttribute(LOADED_EXTRA_BREAK_ATTR))
                        ? lineBreak : null;
                };
                const adjacentProtectedBreak = (host, backwards) => {
                    const lineBreak = adjacentBreakAtCaret(host, backwards);
                    return lineBreak
                            && !lineBreak.hasAttribute(USER_TAG_ATTR)
                            && !lineBreak.hasAttribute(LOADED_EXTRA_BREAK_ATTR)
                        ? lineBreak : null;
                };
                const selectionContainsProtectedBreak = host => {
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount || selection.isCollapsed
                            || !selectionInside(host)) return false;
                    const range = selection.getRangeAt(0);
                    return Array.from(host.querySelectorAll(
                        'br:not([' + USER_TAG_ATTR + ']):not(['
                            + LOADED_EXTRA_BREAK_ATTR + '])'
                    ))
                        .some(lineBreak => {
                            try { return range.intersectsNode(lineBreak); }
                            catch (_error) { return false; }
                        });
                };
                const fallbackEditableBreak = (host, backwards) => {
                    const boundary = selectionBoundary(host);
                    if (!backwards || !boundary.collapsed || !boundary.atStart) return null;
                    const hosts = Array.from(
                        document.body.querySelectorAll('[' + EDIT_ATTR + ']')
                    );
                    const previousHost = hosts[hosts.indexOf(host) - 1];
                    if (!previousHost
                            || userTagContainer(previousHost) !== userTagContainer(host)) {
                        return null;
                    }
                    const breaks = previousHost.querySelectorAll(
                        'br[' + USER_TAG_ATTR + '],br['
                            + LOADED_EXTRA_BREAK_ATTR + ']'
                    );
                    const lineBreak = breaks.length ? breaks[breaks.length - 1] : null;
                    if (!lineBreak) return null;
                    try {
                        const afterBreak = document.createRange();
                        afterBreak.selectNodeContents(previousHost);
                        afterBreak.setStartAfter(lineBreak);
                        if (afterBreak.toString().trim()) return null;
                    } catch (_error) {
                        return null;
                    }
                    return lineBreak;
                };
                const deleteAdjacentEditableBreak = (host, backwards) => {
                    const lineBreak = adjacentEditableBreak(host, backwards)
                        || fallbackEditableBreak(host, backwards);
                    if (!lineBreak) return false;
                    const owner = editableHost(lineBreak);
                    const action = {
                        kind: 'delete',
                        node: lineBreak,
                        parent: lineBreak.parentNode,
                        nextSibling: lineBreak.nextSibling,
                        host: owner || host,
                        indicatorContainer: userTagContainer(lineBreak)
                    };
                    recordBreakEdit(action);
                    lineBreak.remove();
                    refreshUserTagIndicator(action.indicatorContainer);
                    refreshUserEmptyIndicator(action.indicatorContainer);
                    refreshNavigationContainerStyle(action.indicatorContainer);
                    focusBreakAction(action, false);
                    markDirty();
                    return true;
                };
                const deleteEmptyUserParagraph = host => {
                    const block = navigationContainer(host);
                    if (!block || block.tagName.toUpperCase() !== 'P'
                            || block.textContent.trim()
                            || block.querySelector('br,img,image,object,video,audio,svg,math,hr')
                            || (!block.hasAttribute(USER_BLOCK_ATTR)
                                && (block.hasAttribute(SOURCE_ATTR)
                                    || block.hasAttribute(ROW_ATTR)))) return false;
                    const hosts = Array.from(
                        document.body.querySelectorAll('[' + EDIT_ATTR + ']')
                    );
                    const firstBlockHost = block.querySelector('[' + EDIT_ATTR + ']');
                    const hostIndex = hosts.indexOf(firstBlockHost || host);
                    const previousHost = hostIndex > 0 ? hosts[hostIndex - 1] : null;
                    if (!previousHost) return false;
                    const action = {
                        kind: 'delete',
                        block: true,
                        node: block,
                        parent: block.parentNode,
                        nextSibling: block.nextSibling,
                        host: firstBlockHost || host,
                        focusBefore: previousHost,
                        indicatorContainer: null
                    };
                    recordBreakEdit(action);
                    block.remove();
                    focusBreakAction(action, false);
                    markDirty();
                    return true;
                };
                const sanitizeHost = host => {
                    if (!host) return;
                    for (const element of Array.from(host.querySelectorAll('*'))) {
                        const tagName = element.tagName.toUpperCase();
                        if (!ALLOWED_INLINE_TAGS.has(tagName)) {
                            // Chromium sometimes emits styled spans even with
                            // styleWithCSS disabled. Preserve all three active
                            // styles instead of unwrapping whichever was added
                            // last and silently losing it.
                            const style = element.style || {};
                            const weight = String(style.fontWeight || '').toLowerCase();
                            const numericWeight = Number.parseInt(weight, 10);
                            const decoration = String(
                                style.textDecorationLine || style.textDecoration || ''
                            ).toLowerCase();
                            const wrappers = [];
                            if (weight === 'bold'
                                    || (Number.isFinite(numericWeight) && numericWeight >= 600)) {
                                wrappers.push('strong');
                            }
                            if (/^(italic|oblique)$/.test(
                                    String(style.fontStyle || '').toLowerCase())) {
                                wrappers.push('em');
                            }
                            if (decoration.includes('underline')) wrappers.push('u');
                            if (!wrappers.length) {
                                element.replaceWith(...Array.from(element.childNodes));
                                continue;
                            }
                            const content = document.createDocumentFragment();
                            while (element.firstChild) content.appendChild(element.firstChild);
                            let replacement = content;
                            for (const wrapperName of wrappers.slice().reverse()) {
                                const wrapper = document.createElement(wrapperName);
                                wrapper.appendChild(replacement);
                                replacement = wrapper;
                            }
                            element.replaceWith(replacement);
                            continue;
                        }
                        let canonicalName = '';
                        if (tagName === 'B') canonicalName = 'strong';
                        else if (tagName === 'I') canonicalName = 'em';
                        if (canonicalName) {
                            const replacement = document.createElement(canonicalName);
                            while (element.firstChild) replacement.appendChild(element.firstChild);
                            element.replaceWith(replacement);
                        } else {
                            for (const attribute of Array.from(element.attributes)) {
                                if (tagName === 'BR'
                                        && (attribute.name === USER_TAG_ATTR
                                            || attribute.name === LOADED_EXTRA_BREAK_ATTR)) {
                                    continue;
                                }
                                element.removeAttribute(attribute.name);
                            }
                        }
                    }
                    host.normalize();
                };
                const applyInlineFormat = (command, preferRemembered=false) => {
                    if (preferRemembered && window.__sdlNotepadFormatSelection) {
                        restoreSelection(window.__sdlNotepadFormatSelection);
                    }
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount) return false;
                    const host = editableHost(selection.anchorNode);
                    if (!host || editableHost(selection.focusNode) !== host
                            || !selectionInside(host)) return false;
                    const browserCommand = {
                        bold: 'bold', italic: 'italic', underline: 'underline'
                    }[String(command || '').toLowerCase()];
                    if (!browserCommand) return false;
                    const snapshot = selectionOffsets(host);
                    document.execCommand('styleWithCSS', false, false);
                    const changed = document.execCommand(browserCommand, false, null);
                    sanitizeHost(host);
                    if (snapshot) {
                        restoreSelection(snapshot);
                        window.__sdlNotepadFormatSelection = snapshot;
                    }
                    markDirty();
                    return !!changed;
                };
                window.__sdlApplyInlineFormat = applyInlineFormat;
                window.__sdlNotepadFormatSelection = null;
                window.__sdlNotepadContextSnapshot = null;
                document.addEventListener('contextmenu', event => {
                    const element = event.target && event.target.nodeType === Node.ELEMENT_NODE
                        ? event.target : event.target && event.target.parentElement;
                    const sourceElement = element
                        ? element.closest('[' + SOURCE_ATTR + ']') : null;
                    const rowElement = element
                        ? element.closest('[' + ROW_ATTR + ']') : null;
                    const targetElement = reviewTextElement(element);
                    const commandState = command => {
                        try { return !!document.queryCommandState(command); }
                        catch (_error) { return false; }
                    };
                    const tagged = selector => {
                        try { return !!(element && element.closest(selector)); }
                        catch (_error) { return false; }
                    };
                    const computed = element ? window.getComputedStyle(element) : null;
                    const contextHost = editableHost(element);
                    const formatSelection = selectionOffsets(contextHost);
                    if (formatSelection) {
                        window.__sdlNotepadFormatSelection = formatSelection;
                    }
                    const fontWeight = computed
                        ? String(computed.fontWeight || '').toLowerCase() : '';
                    const numericWeight = Number.parseInt(fontWeight, 10);
                    const visuallyBold = fontWeight === 'bold'
                        || (Number.isFinite(numericWeight) && numericWeight >= 600);
                    const visuallyItalic = computed
                        ? /^(italic|oblique)$/.test(String(computed.fontStyle || '').toLowerCase())
                        : false;
                    const visuallyUnderlined = computed
                        ? String(computed.textDecorationLine || '').toLowerCase().includes('underline')
                        : false;
                    window.__sdlNotepadContextSnapshot = {
                        source: sourceElement
                            ? sourceElement.getAttribute(SOURCE_ATTR) || '' : '',
                        rowIndex: rowElement
                            ? Number.parseInt(rowElement.getAttribute(ROW_ATTR), 10) : -1,
                        targetIndex: targetElement
                            ? reviewTextElements().indexOf(targetElement) : -1,
                        formats: {
                            bold: tagged('strong,b') || commandState('bold') || visuallyBold,
                            italic: tagged('em,i') || commandState('italic') || visuallyItalic,
                            underline: tagged('u') || commandState('underline')
                                || visuallyUnderlined
                        }
                    };
                }, true);
                const originallyEditable = Array.from(
                    document.body.querySelectorAll('[contenteditable]')
                );
                if (!originallyEditable.includes(document.body)) originallyEditable.unshift(document.body);
                for (const element of originallyEditable) {
                    element.setAttribute(
                        ORIGINAL_EDITABLE_ATTR,
                        element.hasAttribute('contenteditable')
                            ? element.getAttribute('contenteditable') : '__sdl_missing__'
                    );
                    element.setAttribute('contenteditable', 'false');
                }
                const makeHost = (textNode, placeholder=false, sourceText='') => {
                    const host = document.createElement('span');
                    host.setAttribute(EDIT_ATTR, '1');
                    // Rich editing is restricted to the three inline tags
                    // accepted by sanitizeHost; surrounding EPUB markup stays
                    // contenteditable=false and cannot be altered.
                    host.setAttribute('contenteditable', 'true');
                    host.setAttribute('spellcheck', 'false');
                    host.addEventListener('focus', () => {
                        setActiveNavigationContainer(host);
                    });
                    if (placeholder) host.setAttribute(PLACEHOLDER_ATTR, '1');
                    if (sourceText) host.setAttribute(SOURCE_ATTR, sourceText);
                    if (textNode) textNode.parentNode.replaceChild(host, textNode);
                    if (textNode) host.appendChild(textNode);
                    if (String(sourceText || '').trim()) {
                        const container = userTagContainer(host);
                        if (container) container.setAttribute(ORIGINAL_TEXT_CONTAINER_ATTR, '1');
                    }
                    return host;
                };

                const textNodes = [];
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                while (walker.nextNode()) textNodes.push(walker.currentNode);
                for (const node of textNodes) {
                    const parent = node.parentElement;
                    if (!parent || parent.closest(blockedParents) || parent.closest('[' + EDIT_ATTR + ']')) continue;
                    if (!node.nodeValue.trim() && !parent.closest('pre')) continue;
                    const sourceContainer = parent.closest('[' + SOURCE_ATTR + ']');
                    makeHost(
                        node,
                        false,
                        sourceContainer ? sourceContainer.getAttribute(SOURCE_ATTR) || '' : ''
                    );
                }

                // Put existing breaks inside an editable island. When a text
                // island is immediately before the break, absorb it there so
                // no separate zero-width hit target sits on the sentence edge.
                for (const lineBreak of Array.from(document.body.querySelectorAll('br'))) {
                    if (lineBreak.closest('[' + EDIT_ATTR + ']')
                            || lineBreak.closest(blockedParents)) continue;
                    let preceding = lineBreak.previousSibling;
                    while (preceding && preceding.nodeType === Node.TEXT_NODE
                            && !preceding.nodeValue.trim()) {
                        preceding = preceding.previousSibling;
                    }
                    if (preceding && preceding.nodeType === Node.ELEMENT_NODE
                            && preceding.hasAttribute(EDIT_ATTR)) {
                        preceding.appendChild(lineBreak);
                        continue;
                    }
                    const sourceContainer = lineBreak.parentElement
                        ? lineBreak.parentElement.closest('[' + SOURCE_ATTR + ']') : null;
                    const host = makeHost(
                        null,
                        false,
                        sourceContainer ? sourceContainer.getAttribute(SOURCE_ATTR) || '' : ''
                    );
                    host.setAttribute(BREAK_ATTR, '1');
                    lineBreak.parentNode.replaceChild(host, lineBreak);
                    host.appendChild(lineBreak);
                }

                // Empty rendered text containers still need a caret target.
                document.body.querySelectorAll(
                    'p,h1,h2,h3,h4,h5,h6,li,td,th,caption,figcaption,blockquote'
                ).forEach(element => {
                    if (!element.querySelector('[' + EDIT_ATTR + ']')) {
                        const display = String(window.getComputedStyle(element).display)
                            .toLowerCase();
                        if (display.startsWith('inline')) return;
                        element.insertBefore(
                            makeHost(null, true, element.getAttribute(SOURCE_ATTR) || ''),
                            element.firstChild
                        );
                    }
                });

                // Apply persisted marker state immediately. Previously these
                // container attributes were only refreshed by edit/history
                // handlers, so their yellow/red edge styles appeared only
                // after the user interacted with the loaded document.
                const initialIndicatorContainers = new Set();
                document.body.querySelectorAll(
                    '[' + EDIT_ATTR + '],[' + USER_TAG_ATTR + ']'
                ).forEach(element => {
                    const container = userTagContainer(element);
                    if (!container) return;
                    if (element.hasAttribute(EDIT_ATTR)
                            && String(element.getAttribute(SOURCE_ATTR) || '').trim()) {
                        container.setAttribute(ORIGINAL_TEXT_CONTAINER_ATTR, '1');
                    }
                    initialIndicatorContainers.add(container);
                });
                initialIndicatorContainers.forEach(container => {
                    removeFilledEmptySourcePlaceholders(container);
                    refreshUserTagIndicator(container);
                    refreshUserEmptyIndicator(container);
                    refreshNavigationContainerStyle(container);
                });

                window.__sdlInjectMachineTranslation = (rowIndex, translated) => {
                    const wanted = String(rowIndex);
                    const container = Array.from(
                        document.body.querySelectorAll('[' + ROW_ATTR + ']')
                    ).find(element => element.getAttribute(ROW_ATTR) === wanted);
                    if (!container) return false;
                    const beforeHtml = container.innerHTML;
                    const sourceText = container.getAttribute(SOURCE_ATTR) || '';
                    while (container.firstChild) container.removeChild(container.firstChild);
                    const textNode = document.createTextNode(String(translated || ''));
                    container.appendChild(textNode);
                    const host = makeHost(textNode, false, sourceText);
                    recordInjectionEdit({
                        container,
                        beforeHtml,
                        afterHtml: container.innerHTML
                    });
                    refreshUserTagIndicator(container);
                    refreshUserEmptyIndicator(container);
                    try { host.focus({preventScroll: true}); }
                    catch (_error) { host.focus(); }
                    const selection = window.getSelection();
                    const range = document.createRange();
                    range.selectNodeContents(host);
                    range.collapse(false);
                    selection.removeAllRanges();
                    selection.addRange(range);
                    markDirty();
                    return true;
                };

                const style = document.createElement('style');
                style.id = 'sdl-notepad-guard-style';
                style.textContent =
                    ':root { color-scheme: dark !important; background: #1e1e1e !important; }' +
                    'html, body { background-color: #1e1e1e !important; color: #e8edf2 !important; }' +
                    'body { caret-color: #7dd3fc !important; }' +
                    'body, body p, body h1, body h2, body h3, body h4, body h5, body h6,' +
                    'body li, body div, body span, body strong, body b, body em, body i,' +
                    'body ruby, body rb, body rt, body td, body th, body blockquote,' +
                    'body figcaption { color: #e8edf2 !important; }' +
                    'body article, body section, body main, body header, body footer, body nav,' +
                    'body aside { background-color: transparent !important; }' +
                    'body a, body a * { color: #69b7ff !important; }' +
                    'body pre, body code, body kbd, body samp {' +
                    ' background-color: #242424 !important; color: #d8dee9 !important;' +
                    ' border-color: #4a5568 !important; }' +
                    'body table, body td, body th, body hr { border-color: #4a5568 !important; }' +
                    'body img, body video, body object, body svg {' +
                    ' max-width: 100% !important; height: auto; }' +
                    'body input, body textarea, body select, body button {' +
                    ' background: #242424 !important; color: #e8edf2 !important;' +
                    ' border-color: #4a5568 !important; }' +
                    'body mark { background: #806b16 !important; color: #fff !important; }' +
                    '::selection { background: #315d82 !important; color: #fff !important; }' +
                    '::-webkit-scrollbar { width: 12px; height: 12px; background: #1e1e1e; }' +
                    '::-webkit-scrollbar-thumb { background: #56606d; border-radius: 6px;' +
                    ' border: 2px solid #1e1e1e; }' +
                    '::-webkit-scrollbar-thumb:hover { background: #6b7787; }' +
                    '[' + EDIT_ATTR + '] { outline: none; cursor: text; }' +
                    '[' + EDIT_ATTR + '][' + PLACEHOLDER_ATTR + '] {' +
                    ' display: inline-block; min-width: .65em; min-height: 1em; vertical-align: baseline; }' +
                    '[' + EDIT_ATTR + ']:focus { box-shadow: none; }' +
                    '[' + ACTIVE_CONTAINER_ATTR + ']:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'p:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'h1:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'h2:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'h3:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'h4:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'h5:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'h6:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'li:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'td:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'th:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'caption:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'figcaption:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '],' +
                    'blockquote:focus-within:not([' + MULTILINE_CONTAINER_ATTR + ']) [' + EDIT_ATTR + '] {' +
                    ' box-shadow: none !important;' +
                    ' background-image: linear-gradient(rgba(70,150,220,.72),' +
                    ' rgba(70,150,220,.72)) !important;' +
                    ' background-position: right bottom !important;' +
                    ' background-repeat: no-repeat !important;' +
                    ' background-size: calc(100% - var(--sdl-notepad-leading-space-width, 0px))' +
                    ' 1px !important; }' +
                    '[' + ACTIVE_CONTAINER_ATTR + '][' + MULTILINE_CONTAINER_ATTR + '],' +
                    'p[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'h1[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'h2[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'h3[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'h4[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'h5[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'h6[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'li[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'td[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'th[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'caption[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'figcaption[' + MULTILINE_CONTAINER_ATTR + ']:focus-within,' +
                    'blockquote[' + MULTILINE_CONTAINER_ATTR + ']:focus-within {' +
                    ' outline: 1px solid rgba(70,150,220,.72) !important;' +
                    ' outline-offset: 2px; border-radius: 2px; }' +
                    '[' + EDIT_ATTR + ']:has(br[' + USER_TAG_ATTR + ']:last-child)::after {' +
                    ' content: "\\00a0"; opacity: 0; }' +
                    '[' + EDIT_ATTR + '][' + PLACEHOLDER_ATTR + ']:focus,' +
                    '[' + EDIT_ATTR + '][' + BREAK_ATTR + ']:focus {' +
                    ' box-shadow: none; background-image: none !important; }' +
                    '[' + USER_TAG_CONTAINER_ATTR + '] {' +
                    ' box-sizing: border-box !important; border-left: 4px solid #d7a800 !important;' +
                    ' padding-left: 4px !important; box-shadow: none !important; }' +
                    '[' + USER_EMPTY_CONTAINER_ATTR + '] {' +
                    ' min-height: 1em !important;' +
                    ' box-sizing: border-box !important; border-left: 4px solid #dc3545 !important;' +
                    ' padding-left: 4px !important; box-shadow: none !important; }' +
                    '[' + USER_EMPTY_CONTAINER_ATTR + '] [' + EDIT_ATTR + '] {' +
                    ' display: inline-block; min-width: .65em; min-height: 1em; }' +
                    '[' + STATUS_ATTR + '="purple"] {' +
                    ' box-sizing: border-box !important; border-left: 4px solid #b967ff !important;' +
                    ' padding-left: 4px !important; box-shadow: none !important; }' +
                    '[' + JUMP_HIGHLIGHT_ATTR + '] {' +
                    ' outline: 2px solid #69b7ff !important; outline-offset: 2px;' +
                    ' border-radius: 2px; }' +
                    '#sdl-notepad-source-tooltip { position: fixed; display: none; z-index: 2147483647;' +
                    ' max-width: min(560px, calc(100vw - 32px)); padding: 9px 12px;' +
                    ' border: 1px solid #526173; border-radius: 6px; background: #15191f;' +
                    ' color: #eef2f7 !important; font: 13px/1.45 sans-serif; white-space: pre-wrap;' +
                    ' overflow-wrap: anywhere; box-shadow: 0 5px 18px rgba(0,0,0,.55);' +
                    ' pointer-events: none; }';
                (document.head || document.documentElement).appendChild(style);

                const sourceTooltip = document.createElement('div');
                sourceTooltip.id = 'sdl-notepad-source-tooltip';
                document.body.appendChild(sourceTooltip);
                const tooltipSource = target => {
                    const element = target && target.nodeType === Node.ELEMENT_NODE
                        ? target : target && target.parentElement;
                    if (!element) return '';
                    const sourceElement = element.closest('[' + SOURCE_ATTR + ']');
                    return sourceElement ? sourceElement.getAttribute(SOURCE_ATTR) || '' : '';
                };
                const positionTooltip = event => {
                    const margin = 12;
                    const left = Math.min(
                        event.clientX + 14,
                        window.innerWidth - sourceTooltip.offsetWidth - margin
                    );
                    const top = Math.min(
                        event.clientY + 18,
                        window.innerHeight - sourceTooltip.offsetHeight - margin
                    );
                    sourceTooltip.style.left = Math.max(margin, left) + 'px';
                    sourceTooltip.style.top = Math.max(margin, top) + 'px';
                };
                document.addEventListener('mouseover', event => {
                    const source = tooltipSource(event.target);
                    if (!source) return;
                    sourceTooltip.textContent = source;
                    sourceTooltip.style.display = 'block';
                    positionTooltip(event);
                }, true);
                document.addEventListener('mousemove', event => {
                    if (sourceTooltip.style.display === 'block') positionTooltip(event);
                }, true);
                document.addEventListener('mouseout', event => {
                    const nextSource = tooltipSource(event.relatedTarget);
                    if (!nextSource) sourceTooltip.style.display = 'none';
                }, true);

                document.addEventListener('beforeinput', event => {
                    const inputType = String(event.inputType || '');
                    if (inputType === 'insertParagraph' || inputType === 'insertLineBreak') {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        const host = editableHost(event.target);
                        if (host && selectionInside(host)) {
                            // Some Chromium builds emit beforeinput even after
                            // the Enter keydown was cancelled. Consume that
                            // companion event instead of inserting a second BR.
                            if (enterKeyInsertionHost === host) {
                                enterKeyInsertionHost = null;
                            } else {
                                insertEnter(host);
                            }
                        }
                        return;
                    }
                    const host = editableHost(event.target);
                    if (!host || !selectionInside(host)) {
                        event.preventDefault();
                        return;
                    }
                    if (inputType.startsWith('delete')
                            && deleteEmptyUserParagraph(host)) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    if (inputType.startsWith('delete')
                            && selectionContainsProtectedBreak(host)) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    const boundary = selectionBoundary(host);
                    const deletesBackward = inputType.endsWith('Backward');
                    const deletesForward = inputType.endsWith('Forward');
                    if (boundary.collapsed
                            && ((deletesBackward
                                    && deleteAdjacentEditableBreak(host, true))
                                || (deletesForward
                                    && deleteAdjacentEditableBreak(host, false)))) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    if (boundary.collapsed
                            && ((deletesBackward && adjacentProtectedBreak(host, true))
                                || (deletesForward && adjacentProtectedBreak(host, false)))) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    if (boundary.collapsed
                            && ((deletesBackward && boundary.atStart)
                                || (deletesForward && boundary.atEnd))) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    if (inputType === 'insertText' && typeof event.data === 'string'
                            && /[\r\n\u2028\u2029]/.test(event.data)) {
                        event.preventDefault();
                        document.execCommand(
                            'insertText', false,
                            event.data.replace(/[\r\n\u2028\u2029]+/g, ' ')
                        );
                        markDirty();
                    } else if ((inputType.startsWith('format')
                                && !['formatBold', 'formatItalic', 'formatUnderline'].includes(inputType))
                            || inputType === 'insertOrderedList'
                            || inputType === 'insertUnorderedList' || inputType === 'insertHorizontalRule') {
                        event.preventDefault();
                    }
                }, true);

                document.addEventListener('keydown', event => {
                    const shortcutKey = String(event.key || '').toLowerCase();
                    const shortcutHost = editableHost(event.target);
                    const historyCommand = (event.ctrlKey || event.metaKey) && !event.altKey
                        ? (shortcutKey === 'z' && !event.shiftKey
                            ? 'undo'
                            : (shortcutKey === 'y' || (shortcutKey === 'z' && event.shiftKey)
                                ? 'redo' : ''))
                        : '';
                    if (historyCommand && shortcutHost && selectionInside(shortcutHost)) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        if (historyCommand === 'undo' && injectionUndoReady
                                && replayInjectionHistory(false)) return;
                        if (historyCommand === 'redo' && injectionRedoReady
                                && replayInjectionHistory(true)) return;
                        if (historyCommand === 'undo' && breakUndoReady
                                && replayBreakHistory(false)) return;
                        if (historyCommand === 'redo' && breakRedoReady
                                && replayBreakHistory(true)) return;
                        document.execCommand(historyCommand, false, null);
                        breakUndoReady = false;
                        breakRedoReady = false;
                        injectionUndoReady = false;
                        injectionRedoReady = false;
                        markDirty();
                        return;
                    }
                    const formatCommand = (event.ctrlKey || event.metaKey) && !event.altKey
                        ? ({b: 'bold', i: 'italic', u: 'underline'}[shortcutKey] || '')
                        : '';
                    if (formatCommand && shortcutHost && selectionInside(shortcutHost)) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        applyInlineFormat(formatCommand);
                        return;
                    }
                    if ((event.key === 'ArrowUp' || event.key === 'ArrowDown')
                            && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey
                            && shortcutHost && selectionInside(shortcutHost)) {
                        if (moveVertical(shortcutHost, event.key === 'ArrowUp')) {
                            event.preventDefault();
                            event.stopImmediatePropagation();
                            return;
                        }
                    }
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        const current = editableHost(event.target);
                        if (current) {
                            enterKeyInsertionHost = current;
                            window.setTimeout(() => {
                                if (enterKeyInsertionHost === current) {
                                    enterKeyInsertionHost = null;
                                }
                            }, 0);
                            insertEnter(current);
                        }
                        return;
                    }
                    if (event.key !== 'Backspace' && event.key !== 'Delete') return;
                    const host = editableHost(event.target);
                    if (!host || !selectionInside(host)) {
                        event.preventDefault();
                        return;
                    }
                    const selection = window.getSelection();
                    if (!selection) return;
                    if (!selection.isCollapsed) {
                        if (selectionContainsProtectedBreak(host)) {
                            event.preventDefault();
                            event.stopImmediatePropagation();
                        }
                        return;
                    }
                    if (deleteEmptyUserParagraph(host)) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    if (deleteAdjacentEditableBreak(host, event.key === 'Backspace')) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    if (adjacentProtectedBreak(host, event.key === 'Backspace')) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                        return;
                    }
                    const boundary = selectionBoundary(host);
                    const crossesBoundary = event.key === 'Backspace'
                        ? boundary.atStart : boundary.atEnd;
                    if (crossesBoundary) {
                        event.preventDefault();
                        event.stopImmediatePropagation();
                    }
                }, true);

                document.addEventListener('paste', event => {
                    const host = editableHost(event.target);
                    if (!host || !selectionInside(host)) {
                        event.preventDefault();
                        return;
                    }
                    event.preventDefault();
                    const pastedText = event.clipboardData
                        ? event.clipboardData.getData('text/plain') : '';
                    document.execCommand(
                        'insertText', false,
                        pastedText.replace(/[\r\n\u2028\u2029]+/g, ' ')
                    );
                    markDirty();
                }, true);
                document.addEventListener('drop', event => event.preventDefault(), true);
                document.addEventListener('input', event => {
                    const host = editableHost(event.target);
                    if (!host) return;
                    breakUndoReady = false;
                    breakRedoReady = false;
                    injectionUndoReady = false;
                    injectionRedoReady = false;
                    // Keep only intentional strong/em/u markup. Browser-created
                    // blocks, spans, links, and other tags are unwrapped before
                    // the document enters the save pipeline.
                    sanitizeHost(host);
                    const indicatorContainer = userTagContainer(host);
                    removeFilledEmptySourcePlaceholders(indicatorContainer);
                    if (host.textContent || host.querySelector('br')) {
                        host.removeAttribute(PLACEHOLDER_ATTR);
                    } else {
                        host.setAttribute(PLACEHOLDER_ATTR, '1');
                    }
                    refreshUserTagIndicator(indicatorContainer);
                    refreshUserEmptyIndicator(indicatorContainer);
                    refreshNavigationContainerStyle(navigationContainer(host));
                    markDirty();
                }, true);

                window.addEventListener('resize', () => {
                    refreshNavigationContainerStyle(activeNavigationContainer);
                });

                const firstHost = document.body.querySelector('[' + EDIT_ATTR + ']');
                // Persist one-time repairs of stale trailing placeholder BRs
                // even when the user only opens and closes Notepad.
                if (document.body.querySelector('[' + NORMALIZED_PLACEHOLDER_ATTR + ']')) {
                    markDirty();
                }
                if (firstHost) firstHost.focus();
                else document.body.focus();
                return true;
            })();
        """
        try:
            browser.page().runJavaScript(script)
        except RuntimeError:
            pass

    def _notepad_browser_load_finished(self, browser):
        # A JavaScript callback issued during navigation may be discarded by
        # Chromium. Reset the in-flight guard before starting dirty polling.
        browser._sdl_capture_in_progress = False
        self._install_notepad_browser_editing(browser)
        poll_timer = getattr(browser, "_sdl_edit_poll_timer", None)
        if poll_timer is not None and not poll_timer.isActive():
            poll_timer.start()

    def _poll_notepad_browser_html(self, browser, piece_index):
        # Cached and hidden browser pages must not keep doing Chromium work.
        try:
            if not self.isVisible() or not browser.isVisible():
                return
        except RuntimeError:
            return
        self._capture_notepad_browser_html(browser, piece_index, only_dirty=True)

    def _capture_notepad_browser_html(
        self, browser, piece_index, *, only_dirty=True, flush=False
    ):
        """Pull the editable browser DOM back into the normal save pipeline."""
        if browser is None:
            return
        if bool(getattr(browser, "_sdl_capture_in_progress", False)):
            if flush:
                QTimer.singleShot(
                    40,
                    lambda b=browser, pi=piece_index: self._capture_notepad_browser_html(
                        b, pi, only_dirty=False, flush=True
                    ),
                )
            return
        browser._sdl_capture_in_progress = True

        def _received(document_html):
            try:
                browser._sdl_capture_in_progress = False
                if document_html is None:
                    return
                prefix = str(getattr(browser, "_sdl_document_prefix", "") or "")
                user_added_target_indexes = (
                    self._notepad_user_added_target_indexes(document_html)
                )
                user_added_break_positions = (
                    self._notepad_user_added_break_positions(document_html)
                )
                document = self._clean_notepad_browser_html(document_html)
                if prefix:
                    html_start = re.search(r"<html(?:\s|>)", document, flags=re.IGNORECASE)
                    document = prefix + (document[html_start.start():] if html_start else document)
                self._schedule_notepad_document_edit(
                    piece_index,
                    document,
                    user_added_target_indexes=user_added_target_indexes,
                    user_added_break_positions=user_added_break_positions,
                )
                if flush:
                    if self._edit_save_timer.isActive():
                        self._edit_save_timer.stop()
                    self._flush_target_edits()
            except RuntimeError:
                pass

        def _export_html():
            try:
                browser.page().toHtml(_received)
            except RuntimeError:
                browser._sdl_capture_in_progress = False

        def _dirty_received(is_dirty):
            if not bool(is_dirty):
                browser._sdl_capture_in_progress = False
                return
            _export_html()

        try:
            if only_dirty:
                browser.page().runJavaScript(
                    "(() => { const dirty = !!window.__sdlNotepadDirty; "
                    "window.__sdlNotepadDirty = false; return dirty; })();",
                    _dirty_received,
                )
            else:
                browser.page().runJavaScript("window.__sdlNotepadDirty = false;")
                _export_html()
        except RuntimeError:
            browser._sdl_capture_in_progress = False

    def _find_in_notepad_browser(self, browser):
        query, accepted = QInputDialog.getText(
            self,
            "Find in page",
            "Find:",
            text=str(browser.property("sdl_last_find") or ""),
        )
        if not accepted or not query:
            return
        browser.setProperty("sdl_last_find", query)
        browser.findText("")
        browser.findText(query)

    def _show_notepad_browser_context_menu(
        self,
        browser,
        pos,
        source_text="",
        format_states=None,
        piece_index=None,
        row_index=None,
        target_index=None,
    ):
        """Show browser edit actions plus source and machine-preview context."""
        from PySide6.QtWebEngineCore import QWebEnginePage

        format_states = format_states if isinstance(format_states, dict) else {}
        try:
            piece_index = int(piece_index)
            if piece_index < 0:
                raise IndexError
            piece = self.pieces[piece_index]
            rows = piece.get("rows") or []
            resolved_row_index = None
            try:
                wanted_target_index = int(target_index)
            except (TypeError, ValueError):
                wanted_target_index = -1
            if wanted_target_index >= 0:
                resolved_row_index = next(
                    (
                        candidate_index
                        for candidate_index, candidate in enumerate(rows)
                        if candidate.get("target_index") == wanted_target_index
                    ),
                    None,
                )
            if resolved_row_index is None:
                fallback_row_index = int(row_index)
                if fallback_row_index < 0:
                    raise IndexError
                resolved_row_index = fallback_row_index
            row_index = resolved_row_index
            row_data = rows[row_index]
        except (TypeError, ValueError, IndexError):
            piece = None
            row_data = None

        machine_translation = ""
        machine_preview = "No machine translation preview is available for this row."
        machine_state = ""
        machine_detail = ""
        if row_data is not None:
            machine_translation = self._row_tooltip_translation(piece, row_data)
            row_snapshot = self._review_row_snapshot(row_data)
            machine_state = self._row_machine_translation_preview_state(row_snapshot)
            machine_preview = (
                self._row_machine_translation_preview_from_snapshot(row_snapshot)
                or machine_preview
            )
            machine_detail = str(
                row_data.get("tooltip_translation_error_detail")
                or row_data.get("tooltip_translation_status")
                or ""
            )

        menu = _KeepOpenActionMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #1e1e1e; color: #eef2f7; border: 1px solid #4a5568;
                border-radius: 7px; padding: 6px 18px 6px 6px;
            }
            QMenu::item {
                padding: 7px 34px 7px 10px; margin: 1px 0;
                border-radius: 4px;
            }
            QMenu::item:selected { background: #34465a; }
            QMenu::item:disabled { color: #657180; }
            QMenu::indicator { width: 0px; height: 0px; }
            QMenu::separator { height: 1px; background: #3f4956; margin: 6px 5px; }
        """)

        source_panel = QWidget(menu)
        source_layout = QVBoxLayout(source_panel)
        source_layout.setContentsMargins(10, 7, 24, 8)
        source_layout.setSpacing(4)
        source_header = QLabel("Source text", source_panel)
        source_header.setStyleSheet("color: #7dd3fc; font-weight: bold; font-size: 9pt;")
        source_label = QLabel(str(source_text or "No source text is available for this element."), source_panel)
        source_label.setObjectName("SdlNotepadContextSourceText")
        source_label.setTextFormat(Qt.PlainText)
        source_label.setWordWrap(True)
        source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_label.setMinimumWidth(320)
        source_label.setMaximumWidth(560)
        source_label.setStyleSheet(
            "color: #e8edf2; background: #171b21; border: 1px solid #354252; "
            "border-radius: 4px; padding: 8px 14px 8px 9px;"
        )
        source_layout.addWidget(source_header)
        source_layout.addWidget(source_label)

        machine_header = QLabel("Machine translation preview", source_panel)
        machine_header.setObjectName("SdlNotepadContextMachineTranslationHeader")
        machine_label = QLabel(str(machine_preview), source_panel)
        machine_label.setObjectName("SdlNotepadContextMachineTranslationText")
        machine_label.setTextFormat(Qt.PlainText)
        machine_label.setWordWrap(True)
        machine_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        machine_label.setMinimumWidth(320)
        machine_label.setMaximumWidth(560)
        self._update_notepad_machine_translation_context_widgets(
            machine_header,
            machine_label,
            machine_state,
            machine_preview,
            machine_detail,
            machine_translation,
        )
        source_layout.addSpacing(5)
        source_layout.addWidget(machine_header)
        source_layout.addWidget(machine_label)
        source_action = QWidgetAction(menu)
        source_action.setDefaultWidget(source_panel)
        menu.addAction(source_action)
        menu.addSeparator()

        action_specs = (
            ("Bold", "format:bold"),
            ("Italic", "format:italic"),
            ("Underline", "format:underline"),
            (None, None),
            ("\U0001f310  Generate Machine Translation Preview", "generate:machine-translation"),
            ("\U0001f4e5  Inject Machine Translation", "inject:machine-translation"),
            (None, None),
            ("Undo", QWebEnginePage.WebAction.Undo),
            ("Redo", QWebEnginePage.WebAction.Redo),
            (None, None),
            ("Cut", QWebEnginePage.WebAction.Cut),
            ("Copy", QWebEnginePage.WebAction.Copy),
            ("Paste", QWebEnginePage.WebAction.Paste),
            ("Paste and match style", QWebEnginePage.WebAction.PasteAndMatchStyle),
            (None, None),
            ("Select all", QWebEnginePage.WebAction.SelectAll),
        )
        for label, web_action in action_specs:
            if label is None:
                menu.addSeparator()
                continue
            action = menu.addAction(label)
            if web_action == "generate:machine-translation":
                action.setObjectName("SdlNotepadGenerateMachineTranslationAction")
                action.setProperty(_KeepOpenActionMenu.KEEP_OPEN_PROPERTY, True)
                action.setEnabled(
                    not self._tooltip_translation_running
                    and piece is not None
                    and row_data is not None
                    and bool(str(row_data.get("source") or "").strip())
                )
                action.setToolTip(
                    "Generate a machine translation preview for this row."
                )
                action.triggered.connect(
                    lambda _checked=False, pi=piece_index, ri=row_index:
                        self._translate_single_row_tooltip(pi, ri)
                )
                continue
            if web_action == "inject:machine-translation":
                action.setObjectName("SdlNotepadInjectMachineTranslationAction")
                action.setEnabled(
                    bool(machine_translation) and machine_state == "translation"
                    and piece is not None and row_data is not None
                )
                action.setToolTip(
                    "Replace this row's output with the machine translation preview."
                )
                action.triggered.connect(
                    lambda _checked=False, view=browser, pi=piece_index, ri=row_index:
                        self._inject_notepad_machine_translation(view, pi, ri)
                )
                continue
            if isinstance(web_action, str) and web_action.startswith("format:"):
                command = web_action.split(":", 1)[1]
                action.setObjectName(
                    f"SdlNotepadFormat{command.title()}Action"
                )
                action.setCheckable(True)
                is_active = bool(format_states.get(command, False))
                action.setChecked(is_active)
                if is_active:
                    # The app's dark QMenu stylesheet can suppress the native
                    # platform check indicator, so keep an explicit visible
                    # mark in addition to QAction's checked state.
                    action.setText(f"✓ {label}")
                shortcut = {
                    "bold": "Ctrl+B",
                    "italic": "Ctrl+I",
                    "underline": "Ctrl+U",
                }.get(command)
                if shortcut:
                    action.setShortcut(QKeySequence(shortcut))
                    action.setShortcutVisibleInContextMenu(True)
                action.triggered.connect(
                    lambda _checked=False, selected_command=command, view=browser:
                        self._apply_notepad_inline_format(view, selected_command)
                )
                continue
            try:
                action.setEnabled(browser.page().action(web_action).isEnabled())
            except Exception:
                pass
            action.triggered.connect(
                lambda _checked=False, selected_action=web_action, view=browser:
                    view.page().triggerAction(selected_action)
            )

        self._review_text_context_menu = menu
        menu.setProperty("sdl_notepad_piece_index", piece_index)
        menu.setProperty("sdl_notepad_row_index", row_index)
        self._set_review_context_menu_open(True)
        menu.aboutToHide.connect(lambda m=menu: self._clear_review_text_context_menu(m))
        menu.popup(browser.mapToGlobal(pos))
        return menu

    def _update_notepad_machine_translation_context_widgets(
        self,
        machine_header,
        machine_label,
        machine_state,
        machine_preview,
        machine_detail="",
        machine_translation="",
    ):
        """Update the machine-preview panel without replacing its open menu."""
        machine_state = str(machine_state or "").strip().lower()
        machine_header.setStyleSheet(
            "color: #d8c99b; font-weight: bold; font-size: 9pt;"
            if machine_state in {"pending", "error"}
            else "color: #93c5fd; font-weight: bold; font-size: 9pt;"
        )
        machine_label.setText(str(machine_preview or ""))
        machine_label.setToolTip(
            self._wrapped_tooltip(machine_detail or machine_translation)
        )
        machine_label.setStyleSheet(
            "color: #d8c99b; background: #2a2518; border: 1px dashed #8a6f2a; "
            "border-left: 3px solid #d39e00; border-radius: 4px; padding: 8px 14px 8px 9px;"
            if machine_state in {"pending", "error"}
            else "color: #dbeafe; background: #172536; border: 1px solid #37536d; "
            "border-left: 3px solid #5aa7d8; border-radius: 4px; padding: 8px 14px 8px 9px;"
        )

    def _refresh_open_notepad_machine_translation_context(self):
        """Refresh a persistent Notepad context menu's preview and actions."""
        menu = getattr(self, "_review_text_context_menu", None)
        if menu is None or not isinstance(menu, _KeepOpenActionMenu):
            return
        try:
            piece_index = int(menu.property("sdl_notepad_piece_index"))
            row_index = int(menu.property("sdl_notepad_row_index"))
            piece = self.pieces[piece_index]
            row_data = (piece.get("rows") or [])[row_index]
        except (TypeError, ValueError, IndexError, RuntimeError):
            return
        machine_translation = self._row_tooltip_translation(piece, row_data)
        row_snapshot = self._review_row_snapshot(row_data)
        machine_state = self._row_machine_translation_preview_state(row_snapshot)
        machine_preview = (
            self._row_machine_translation_preview_from_snapshot(row_snapshot)
            or "No machine translation preview is available for this row."
        )
        machine_detail = str(
            row_data.get("tooltip_translation_error_detail")
            or row_data.get("tooltip_translation_status")
            or ""
        )
        machine_header = menu.findChild(
            QLabel, "SdlNotepadContextMachineTranslationHeader"
        )
        machine_label = menu.findChild(
            QLabel, "SdlNotepadContextMachineTranslationText"
        )
        if machine_header is not None and machine_label is not None:
            self._update_notepad_machine_translation_context_widgets(
                machine_header,
                machine_label,
                machine_state,
                machine_preview,
                machine_detail,
                machine_translation,
            )
        actions = {
            action.objectName(): action
            for action in menu.actions()
            if action.objectName()
        }
        generate_action = actions.get("SdlNotepadGenerateMachineTranslationAction")
        if generate_action is not None:
            generate_action.setEnabled(
                not self._tooltip_translation_running
                and bool(str(row_data.get("source") or "").strip())
            )
        inject_action = actions.get("SdlNotepadInjectMachineTranslationAction")
        if inject_action is not None:
            inject_action.setEnabled(
                bool(machine_translation) and machine_state == "translation"
            )

    def _inject_notepad_machine_translation(self, browser, piece_index, row_index):
        """Inject a row preview into the live Notepad DOM, then save normally."""
        try:
            piece_index = int(piece_index)
            row_index = int(row_index)
            if piece_index < 0 or row_index < 0:
                raise IndexError
            piece = self.pieces[piece_index]
            row_data = (piece.get("rows") or [])[row_index]
            translated = self._row_tooltip_translation(piece, row_data).strip()
        except (TypeError, ValueError, IndexError):
            translated = ""
        if not translated:
            try:
                self.save_status_label.setText("No machine translation preview")
            except Exception:
                pass
            return

        script = (
            "window.__sdlInjectMachineTranslation ? "
            f"window.__sdlInjectMachineTranslation({json.dumps(str(row_index))}, "
            f"{json.dumps(translated)}) : false;"
        )

        def _injected(changed):
            try:
                if not bool(changed):
                    self.save_status_label.setText("Machine translation injection failed")
                    return
                self.save_status_label.setText("Saving machine translation…")
                self._capture_notepad_browser_html(
                    browser, piece_index, only_dirty=True, flush=True
                )
            except RuntimeError:
                pass

        try:
            browser.page().runJavaScript(script, _injected)
        except RuntimeError:
            pass

    @staticmethod
    def _apply_notepad_inline_format(browser, command):
        """Apply one whitelisted inline HTML format to the browser selection."""
        command_json = json.dumps(str(command or "").strip().lower())
        try:
            browser.page().runJavaScript(
                f"window.__sdlApplyInlineFormat"
                f" ? window.__sdlApplyInlineFormat({command_json}, true) : false;"
            )
        except RuntimeError:
            pass

    def _request_notepad_browser_context_menu(self, browser, pos):
        x = int(pos.x())
        y = int(pos.y())
        script = f"""
            (() => {{
                const contextSnapshot = window.__sdlNotepadContextSnapshot;
                if (contextSnapshot) {{
                    window.__sdlNotepadContextSnapshot = null;
                    return JSON.stringify(contextSnapshot);
                }}
                const element = document.elementFromPoint({x}, {y});
                const sourceElement = element
                    ? element.closest('[data-sdl-notepad-source]') : null;
                const rowElement = element
                    ? element.closest('[data-sdl-notepad-row-index]') : null;
                const textUnitSelector = 'h1,h2,h3,h4,h5,h6,p,li,hr,div.u';
                const nestedTextUnitSelector = 'h1,h2,h3,h4,h5,h6,p,li,hr';
                const targetElement = element ? element.closest(textUnitSelector) : null;
                const targetElements = Array.from(
                    document.body.querySelectorAll(textUnitSelector)
                ).filter(candidate => candidate.tagName.toUpperCase() !== 'DIV'
                    || !candidate.querySelector(nestedTextUnitSelector));
                const targetIndex = targetElement
                    && (targetElement.tagName.toUpperCase() !== 'DIV'
                        || !targetElement.querySelector(nestedTextUnitSelector))
                    ? targetElements.indexOf(targetElement) : -1;
                const commandState = command => {{
                    try {{ return !!document.queryCommandState(command); }}
                    catch (_error) {{ return false; }}
                }};
                const pointTagState = selector => {{
                    try {{ return !!(element && element.closest(selector)); }}
                    catch (_error) {{ return false; }}
                }};
                const selectionTagState = tagNames => {{
                    const selection = window.getSelection();
                    if (!selection || !selection.rangeCount) return false;
                    const tags = new Set(tagNames.map(name => name.toUpperCase()));
                    const insideTag = node => {{
                        let current = node && node.nodeType === Node.ELEMENT_NODE
                            ? node : node && node.parentElement;
                        while (current && current !== document.body) {{
                            if (tags.has(current.tagName.toUpperCase())) return true;
                            current = current.parentElement;
                        }}
                        return false;
                    }};
                    if (selection.isCollapsed) return insideTag(selection.anchorNode);
                    const range = selection.getRangeAt(0);
                    const selectedTextNodes = [];
                    const walker = document.createTreeWalker(
                        document.body, NodeFilter.SHOW_TEXT
                    );
                    while (walker.nextNode()) {{
                        const node = walker.currentNode;
                        if (!node.nodeValue || !node.nodeValue.trim()) continue;
                        try {{
                            if (range.intersectsNode(node)) selectedTextNodes.push(node);
                        }} catch (_error) {{}}
                    }}
                    return selectedTextNodes.length > 0
                        && selectedTextNodes.every(insideTag);
                }};
                return JSON.stringify({{
                    source: sourceElement
                        ? sourceElement.getAttribute('data-sdl-notepad-source') || '' : '',
                    rowIndex: rowElement
                        ? Number.parseInt(rowElement.getAttribute('data-sdl-notepad-row-index'), 10) : -1,
                    targetIndex,
                    formats: {{
                        bold: commandState('bold')
                            || pointTagState('strong,b')
                            || selectionTagState(['strong', 'b']),
                        italic: commandState('italic')
                            || pointTagState('em,i')
                            || selectionTagState(['em', 'i']),
                        underline: commandState('underline')
                            || pointTagState('u')
                            || selectionTagState(['u'])
                    }}
                }});
            }})();
        """

        def _show(context):
            try:
                if isinstance(context, str):
                    try:
                        context = json.loads(context)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        context = {}
                context = context if isinstance(context, dict) else {}
                self._show_notepad_browser_context_menu(
                    browser,
                    pos,
                    context.get("source", ""),
                    context.get("formats", {}),
                    browser.property("sdl_piece_index"),
                    context.get("rowIndex", -1),
                    context.get("targetIndex", -1),
                )
            except RuntimeError:
                pass

        try:
            browser.page().runJavaScript(script, _show)
        except RuntimeError:
            pass

    def _add_notepad_document_editor(self, piece, layout):
        """Add one rendered, editable browser document—never visible markup."""
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWebEngineCore import QWebEngineSettings

        browser = QWebEngineView()
        browser.setObjectName("SdlReviewNotepadBrowser")
        browser.setProperty("sdl_piece_index", piece["index"])
        try:
            browser.settings().setAttribute(QWebEngineSettings.AutoLoadImages, True)
            browser.settings().setAttribute(
                QWebEngineSettings.LocalContentCanAccessRemoteUrls, True
            )
            browser.settings().setAttribute(
                QWebEngineSettings.LocalContentCanAccessFileUrls, True
            )
        except Exception:
            pass
        try:
            viewport_height = int(self.scroll.viewport().height())
        except Exception:
            viewport_height = 640
        browser.setMinimumHeight(max(420, viewport_height - 18))
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setStyleSheet(
            "QWebEngineView#SdlReviewNotepadBrowser { border: 1px solid #4a5568; "
            "border-radius: 2px; background: #1e1e1e; }"
        )
        try:
            browser.page().setBackgroundColor(QColor(self.THEME["bg"]))
        except Exception:
            pass
        # Source tooltips are rendered inside the page; a widget-level tooltip
        # would cover them with the old generic editor help text.
        browser.setToolTip("")
        browser.setContextMenuPolicy(Qt.CustomContextMenu)
        browser.customContextMenuRequested.connect(
            lambda pos, view=browser: self._request_notepad_browser_context_menu(view, pos)
        )
        poll_timer = QTimer(browser)
        poll_timer.setInterval(350)
        poll_timer.timeout.connect(
            lambda view=browser, pi=piece["index"]: self._poll_notepad_browser_html(
                view, pi
            )
        )
        browser._sdl_edit_poll_timer = poll_timer
        browser.loadFinished.connect(
            lambda _ok, view=browser: self._notepad_browser_load_finished(view)
        )

        find_shortcut = QShortcut(QKeySequence("Ctrl+F"), browser)
        find_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        find_shortcut.activated.connect(
            lambda view=browser: self._find_in_notepad_browser(view)
        )
        browser._sdl_find_shortcut = find_shortcut
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), browser)
        save_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        save_shortcut.activated.connect(
            lambda view=browser, pi=piece["index"]: self._save_notepad_editor_now(view, pi)
        )
        browser._sdl_save_shortcut = save_shortcut
        self._set_notepad_browser_html(
            browser, piece, self._notepad_initial_document_html(piece)
        )
        layout.addWidget(browser, 1)
        return browser

    def _add_notepad_review_row(self, piece, row_data, idx, updates_enabled=True):
        """Render one DOM-ordered line in the continuous Notepad layout."""
        structural = bool(row_data.get("notepad_structural"))
        review_row_index = int(row_data.get("review_row_index", -1))
        status = str(row_data.get("status") or "structural")
        status_color = {
            "green": self.THEME["success"],
            "yellow": self.THEME["warning"],
            "purple": self.THEME["purple"],
            "red": self.THEME["danger"],
        }.get(status, self.THEME["border"])
        target_text = str(row_data.get("target") or "")
        source_text = str(row_data.get("source") or "")
        if structural:
            row_height = 29
        else:
            wrapped_lines = max(
                1,
                target_text.count("\n") + 1,
                (len(target_text) // 105) + 1,
            )
            row_height = max(38, min(320, wrapped_lines * 21 + 15))

        frame = QFrame()
        frame.setObjectName("SdlReviewNotepadStructuralRow" if structural else "SdlReviewRow")
        frame.setProperty("sdl_notepad_layout", True)
        frame.setProperty("sdl_status", status if not structural else "structural")
        frame.setProperty("sdl_row_index", review_row_index)
        frame.setFixedHeight(row_height)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setUpdatesEnabled(bool(updates_enabled))
        row_style = (
            f"QFrame#{frame.objectName()} {{ background-color: {self.THEME['panel_alt']}; "
            f"border: 0; border-left: 3px solid {status_color}; "
            f"border-bottom: 1px solid #363c46; border-radius: 0; }}"
        )
        frame.setProperty("sdl_base_style", row_style)
        frame.setStyleSheet(row_style)

        grid = QGridLayout(frame)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        grid.setColumnMinimumWidth(0, 280)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        tag_label = QLabel(str(row_data.get("notepad_tag_caption") or ""))
        tag_label.setObjectName("SdlReviewNotepadTag")
        tag_label.setTextFormat(Qt.PlainText)
        tag_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tag_label.setFixedWidth(280)
        tag_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tag_label.setToolTip(
            self._wrapped_tooltip(
                f"DOM position {int(row_data.get('notepad_dom_order', idx)) + 1}\n"
                f"{row_data.get('notepad_tag_caption') or ''}"
            )
        )
        tag_label.setStyleSheet(
            f"color: {status_color if not structural else self.THEME['muted']}; "
            "background-color: #20242a; border: 0; border-right: 1px solid #404854; "
            "padding: 3px 8px; font: 9pt Consolas, 'Courier New', monospace;"
        )
        grid.addWidget(tag_label, 0, 0)

        if structural:
            inline_text = str(row_data.get("notepad_inline_text") or "")
            content = QLabel(inline_text)
            content.setObjectName("SdlReviewNotepadStructuralText")
            content.setTextFormat(Qt.PlainText)
            content.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            content.setTextInteractionFlags(Qt.TextSelectableByMouse)
            content.setToolTip(self._wrapped_tooltip(inline_text))
            content.setStyleSheet(
                f"color: {self.THEME['muted']}; background: transparent; border: 0; "
                "padding: 2px 10px; font: 9pt Consolas, 'Courier New', monospace;"
            )
            grid.addWidget(content, 0, 1)
        else:
            editor = self._target_display_widget(
                piece["index"],
                review_row_index,
                target_text,
                editable=True,
                height=row_height,
            )
            editor.setPlaceholderText("Type translation…")
            editor.setToolTip(
                self._wrapped_tooltip(
                    f"Source:\n{source_text}\n\nOutput:\n{target_text}"
                )
            )
            editor.setStyleSheet(
                "QPlainTextEdit#SdlReviewTargetEdit { background: transparent; color: #f8fafc; "
                "border: 0; border-radius: 0; padding: 6px 10px; "
                "font: 10pt Consolas, 'Courier New', monospace; "
                "selection-background-color: #5a9fd4; }"
                "QPlainTextEdit#SdlReviewTargetEdit:focus { background-color: #202b36; "
                "border: 0; border-bottom: 1px solid #5a9fd4; }"
            )
            try:
                editor.customContextMenuRequested.disconnect()
            except Exception:
                pass
            tooltip_translation = self._row_tooltip_translation(piece, row_data)
            editor.customContextMenuRequested.connect(
                lambda pos, ed=editor, pi=piece["index"], ri=review_row_index,
                translated=tooltip_translation: self._show_review_text_context_menu(
                    ed,
                    pos,
                    translate_tooltip_callback=lambda: self._translate_single_row_tooltip(pi, ri),
                    inject_machine_translation_callback=(
                        lambda: self._inject_current_machine_translation_to_target(pi, ri, ed)
                    ) if translated else None,
                )
            )
            grid.addWidget(editor, 0, 1)

        stretch_index = self._review_layout_trailing_stretch_index(self.rows_layout)
        if stretch_index >= 0:
            self.rows_layout.insertWidget(stretch_index, frame)
        else:
            self.rows_layout.addWidget(frame)
        return frame

    def _add_review_row(self, piece, row_data, idx, max_len, colors, row_model=None, updates_enabled=True):
        row_model = row_model if isinstance(row_model, dict) else {}
        two_column_layout = bool(row_model.get("two_column_layout", row_model.get("one_column_layout", row_model.get("one_row_layout", getattr(self, "_two_column_layout_enabled", True)))))
        if not two_column_layout:
            return self._add_notepad_review_row(
                piece,
                row_data,
                idx,
                updates_enabled=updates_enabled,
            )
        if not self._compact_review_row_visible(row_data):
            return None
        bg, source_bar, target_bar, dot_color, border_color = colors.get(row_data["status"], colors["green"])
        source_text = row_model.get("source_text", row_data.get("source", ""))
        target_text = row_model.get("target_text", row_data.get("target", ""))
        tooltip_translation = row_model.get("tooltip_translation", self._row_tooltip_translation(piece, row_data))
        fallback_snapshot = self._review_row_snapshot(row_data)
        tooltip_preview = row_model.get(
            "tooltip_preview",
            self._row_machine_translation_preview_from_snapshot(fallback_snapshot),
        )
        tooltip_state = row_model.get(
            "tooltip_state",
            self._row_machine_translation_preview_state(fallback_snapshot),
        )
        tooltip_detail = row_model.get(
            "tooltip_detail",
            str(row_data.get("tooltip_translation_error_detail") or row_data.get("tooltip_translation_status") or ""),
        )
        tooltip_pending = str(tooltip_state or "").strip().lower() == "pending"
        row_height = int(
            row_model.get("row_height")
            or self._review_row_height(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                tooltip_preview_text=tooltip_preview,
            )
        )
        frame = QFrame()
        frame.setObjectName("SdlReviewRow")
        frame.setProperty("sdl_status", row_data["status"])
        frame.setProperty("sdl_row_index", idx)
        frame.setProperty("sdl_two_column_layout", two_column_layout)
        frame.setFixedHeight(row_height)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setUpdatesEnabled(bool(updates_enabled))
        row_style = f"QFrame#SdlReviewRow {{ background-color: {bg}; border: 1px solid {border_color}; border-radius: 3px; }}"
        frame.setProperty("sdl_base_style", row_style)
        frame.setStyleSheet(row_style)
        grid = QGridLayout(frame)
        grid.setContentsMargins(4, 5, 6, 5)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(0)
        grid.setColumnMinimumWidth(0, 100)
        if two_column_layout:
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 0)
            grid.setColumnMinimumWidth(2, 250)
        else:
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 0)
            grid.setColumnStretch(3, 0)
            grid.setColumnStretch(4, 0)
            grid.setColumnStretch(5, 1)
            grid.setColumnMinimumWidth(2, 180)
            grid.setColumnMinimumWidth(3, 26)
            grid.setColumnMinimumWidth(4, 180)

        source_missing = bool(row_model.get("source_missing", not row_data.get("source_tag")))
        target_missing = bool(row_model.get("target_missing", not row_data.get("target_tag")))
        target_editable = bool(row_model.get("target_editable", not source_missing or not target_missing))

        source_tag_label = row_data.get("source_tag_label")
        target_tag_label = row_data.get("target_tag_label")
        compact_tn_label = self._compact_translator_note_display_label(
            piece, idx
        )
        if compact_tn_label:
            source_tag_label = compact_tn_label
            target_tag_label = compact_tn_label
        tag_label = self._tag_label(
            row_data.get("source_tag"),
            row_data.get("target_tag"),
            row_data.get("status"),
            source_label=source_tag_label,
            target_label=target_tag_label,
        )
        tag_label.setToolTip(row_data.get("reason", ""))
        target_widget = self._target_display_widget(
            piece["index"],
            idx,
            target_text,
            editable=target_editable,
            height=max(30, row_height - 14),
        )

        source_label = self._text_label(
            source_text,
            missing=source_missing,
            empty_placeholder=(
                "[Translator Note]"
                if row_model.get("translator_note") else None
            ),
            tooltip_translation=tooltip_preview,
            tooltip_pending=tooltip_pending,
            tooltip_state=tooltip_state,
            tooltip_detail=tooltip_detail,
            translate_tooltip_callback=(
                lambda pi=piece["index"], ri=idx: self._translate_single_row_tooltip(pi, ri)
            ) if source_text else None,
            inject_machine_translation_callback=(
                lambda pi=piece["index"], ri=idx, text=tooltip_translation, ed=target_widget:
                    self._inject_machine_translation_to_target(pi, ri, text, ed)
            ) if tooltip_translation and target_editable and tooltip_state == "translation" else None,
        )
        source_label.setMaximumHeight(max(24, row_height - 14))
        source_label.setToolTip(self._wrapped_tooltip(source_text))

        dot = QLabel("●")
        dot.setObjectName("SdlReviewStatusDot")
        dot.setAlignment(Qt.AlignCenter)
        dot.setStyleSheet(f"color: {dot_color}; background: transparent; font-size: 13pt;")
        dot.setToolTip(row_data.get("reason", ""))

        grid.addWidget(tag_label, 0, 0)
        if two_column_layout:
            content = QWidget()
            content.setObjectName("SdlReviewTwoColumnText")
            content.setMinimumWidth(0)
            content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            content.setStyleSheet("QWidget#SdlReviewTwoColumnText { background: transparent; }")
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(7)
            self._apply_review_row_text_geometry(
                frame,
                source_label,
                target_widget,
                row_height,
                source_lines=row_model.get("source_lines", 1),
                target_lines=row_model.get("target_lines", 1),
                tooltip_lines=row_model.get("tooltip_lines", 0),
            )
            content_layout.addWidget(source_label)
            content_layout.addWidget(target_widget)
            grid.addWidget(content, 0, 1)
            controls = self._review_row_controls_widget(
                piece,
                row_data,
                idx,
                row_model,
                max_len,
                source_bar,
                target_bar,
                dot,
                target_widget,
                target_editable,
                tooltip_translation,
                tooltip_pending,
            )
            grid.addWidget(controls, 0, 2, Qt.AlignTop)
        else:
            grid.addWidget(source_label, 0, 1)
            grid.addWidget(self._bar_widget(row_model.get("source_len", len(source_text)), max_len, source_bar, align_right=True), 0, 2)
            grid.addWidget(dot, 0, 3)
            grid.addWidget(self._bar_widget(row_model.get("target_len", len(target_text)), max_len, target_bar, align_right=False), 0, 4)
            grid.addWidget(target_widget, 0, 5)
        # Insert above the trailing stretch when one is already present (the
        # streaming path adds it up-front so rows stay top-anchored while the
        # rest of the page streams in - without it, a box layout spreads any
        # surplus space BETWEEN the rows, making them visibly bounce whenever
        # the page height is transiently out of sync with the row total).
        stretch_index = self._review_layout_trailing_stretch_index(self.rows_layout)
        if stretch_index >= 0:
            self.rows_layout.insertWidget(stretch_index, frame)
        else:
            self.rows_layout.addWidget(frame)
        return frame

    def _render_piece(self, row, show_loading=True):
        trace_started = time.perf_counter()
        if row < 0 or row >= len(self.pieces):
            return
        if self._preload_render_row is not None:
            self._cancel_review_preload(discard_page=True)
        try:
            if self.piece_list.currentRow() != row:
                return
        except Exception:
            pass
        self._save_current_review_scroll()
        self._render_token += 1
        self._clear_review_row_highlight()
        self._status_jump_indices.clear()
        render_token = self._render_token
        self._cancel_active_review_render(defer_visible_discard=True)
        # A mode switch installs its viewport snapshot before destroying the
        # old page. Normal entry changes capture here before geometry churn.
        transition_pixmap = self._take_review_transition_snapshot(show_loading)

        piece = self.pieces[row]
        try:
            self.current_path = os.path.abspath(piece.get("path") or self.current_path or "")
            self._initial_piece_row = row
            if 0 <= self._book_index < len(self._book_entries):
                self._book_entries[self._book_index]["current_path"] = self.current_path
        except Exception:
            pass
        warning_count = piece.get("yellow_count", 0) + piece.get("purple_count", 0)
        status_text = "MISMATCH" if piece["mismatch"] else ("WARN" if warning_count else "OK")
        flagged = piece["red_count"] + warning_count
        output_name = self._output_name_for_piece(piece)
        review_label = piece.get("review_label") or f"[{row + 1:03d}] Ch.{self._format_chapter_number(piece.get('chapter_num'))} |"
        self.header_label.setText(
            f"{review_label} {output_name}  -  source {piece['source_count']} text units "
            f"- output {piece['target_count']} - {status_text} - {flagged} flagged rows "
            f"(ratio ~= {piece['count_ratio']:.2f})"
        )

        cached_page = self._piece_pages.get(row)
        if cached_page is not None and row in self._piece_render_complete:
            self.rows_widget = cached_page
            self.rows_layout = cached_page.layout()
            self.rows_stack.setCurrentWidget(cached_page)
            self._finish_seamless_review_swap(cached_page)
            self._finish_rows_rebuild(final=True)
            self._queue_refresh_current_visible_dirty_source_previews()
            self._restore_review_scroll(row)
            self._start_review_page_transition(transition_pixmap)
            QTimer.singleShot(0, self._queue_refresh_current_visible_dirty_source_previews)
            self._queue_review_page_preloads(row)
            self._trace_review_perf(
                "render_piece_cached_done",
                trace_started,
                force=True,
                row=row,
                rows=len(piece.get("rows") or []),
            )
            return

        if cached_page is not None:
            self._discard_piece_page(row, cached_page)

        page, layout = self._create_review_rows_page()
        self._piece_pages[row] = page
        self._piece_render_complete.discard(row)
        self.rows_stack.addWidget(page)
        self.rows_widget = page
        self.rows_layout = layout
        self._active_render_row = row
        self._active_render_page = page

        if piece.get("error"):
            error = QLabel(f"Could not parse SDLXLIFF:\n{piece['error']}")
            error.setTextFormat(Qt.PlainText)
            error.setStyleSheet(f"color: {self.THEME['danger']}; font-size: 11pt; padding: 12px;")
            layout.addWidget(error)
            layout.addStretch(1)
            self._piece_render_complete.add(row)
            self._active_render_row = None
            self._active_render_page = None
            self.rows_stack.setCurrentWidget(page)
            self._finish_seamless_review_swap(page)
            self._finish_rows_rebuild(final=True)
            self._restore_review_scroll(row)
            self._start_review_page_transition(transition_pixmap)
            self._queue_review_page_preloads(row)
            self._trace_review_perf(
                "render_piece_error_done",
                trace_started,
                force=True,
                row=row,
            )
            return

        if not bool(getattr(self, "_two_column_layout_enabled", True)):
            try:
                self._add_notepad_document_editor(piece, layout)
                self._piece_render_complete.add(row)
                self._active_render_row = None
                self._active_render_page = None
                self.rows_stack.setCurrentWidget(page)
                self._finish_seamless_review_swap(page)
                self._finish_rows_rebuild(final=True)
                self._restore_review_scroll(row)
                self._start_review_page_transition(transition_pixmap)
                self._trace_review_perf(
                    "render_piece_notepad_done",
                    trace_started,
                    force=True,
                    row=row,
                    chars=len(str(piece.get("target_html") or "")),
                )
                return
            except Exception as exc:
                self._clear_rows(layout)
                error = QLabel(f"Could not open rendered HTML editor:\n{exc}")
                error.setTextFormat(Qt.PlainText)
                error.setStyleSheet(f"color: {self.THEME['danger']}; font-size: 11pt; padding: 12px;")
                layout.addWidget(error)
                layout.addStretch(1)
                self._piece_render_complete.add(row)
                self._active_render_row = None
                self._active_render_page = None
                self.rows_stack.setCurrentWidget(page)
                self._finish_seamless_review_swap(page)
                self._finish_rows_rebuild(final=True)
                self._restore_review_scroll(row)
                self._start_review_page_transition(transition_pixmap)
                return

        rows = self._review_rows_for_current_layout(piece)
        if not rows:
            empty = QLabel("No reviewable HTML elements found in this sidecar.")
            empty.setTextFormat(Qt.PlainText)
            empty.setStyleSheet(f"color: {self.THEME['muted']}; padding: 12px;")
            layout.addWidget(empty)
            layout.addStretch(1)
            self._piece_render_complete.add(row)
            self._active_render_row = None
            self._active_render_page = None
            self.rows_stack.setCurrentWidget(page)
            self._finish_seamless_review_swap(page)
            self._finish_rows_rebuild(final=True)
            self._restore_review_scroll(row)
            self._start_review_page_transition(transition_pixmap)
            self._queue_review_page_preloads(row)
            self._trace_review_perf(
                "render_piece_empty_done",
                trace_started,
                force=True,
                row=row,
            )
            return

        try:
            render_model = self._review_piece_render_model(piece)
            max_len = int(render_model.get("max_len", 1))
            row_models = render_model.get("rows") or []
            colors = self._review_status_colors()
        except Exception as exc:
            self._clear_rows(layout)
            error = QLabel(f"Could not prepare SDLXLIFF review rows:\n{exc}")
            error.setTextFormat(Qt.PlainText)
            error.setStyleSheet(f"color: {self.THEME['danger']}; font-size: 11pt; padding: 12px;")
            layout.addWidget(error)
            layout.addStretch(1)
            self._piece_render_complete.add(row)
            self._active_render_row = None
            self._active_render_page = None
            self.rows_stack.setCurrentWidget(page)
            self._finish_seamless_review_swap(page)
            self._finish_rows_rebuild(final=True)
            self._restore_review_scroll(row)
            self._start_review_page_transition(transition_pixmap)
            self._queue_review_page_preloads(row)
            self._trace_review_perf(
                "render_piece_model_failed_done",
                trace_started,
                force=True,
                row=row,
            )
            return

        if len(rows) <= self.REVIEW_SYNC_RENDER_ROW_LIMIT:
            sync_started = time.perf_counter()
            try:
                for idx, row_data in enumerate(rows):
                    row_model = row_models[idx] if idx < len(row_models) else None
                    self._add_review_row(piece, row_data, idx, max_len, colors, row_model=row_model)
                layout.addStretch(1)
                self._piece_render_complete.add(row)
                self._active_render_row = None
                self._active_render_page = None
                self.rows_stack.setCurrentWidget(page)
                self._finish_seamless_review_swap(page)
                self._finish_rows_rebuild(final=True)
                self._restore_review_scroll(row)
                self._start_review_page_transition(transition_pixmap)
                self._queue_review_page_preloads(row)
                self._trace_review_perf(
                    "render_piece_sync_done",
                    trace_started,
                    force=True,
                    row=row,
                    rows=len(rows),
                    build_ms=f"{(time.perf_counter() - sync_started) * 1000.0:.1f}",
                )
                return
            except Exception as exc:
                self._clear_rows(layout)
                error = QLabel(f"Could not render SDLXLIFF review rows:\n{exc}")
                error.setTextFormat(Qt.PlainText)
                error.setStyleSheet(f"color: {self.THEME['danger']}; font-size: 11pt; padding: 12px;")
                layout.addWidget(error)
                layout.addStretch(1)
                self._piece_render_complete.add(row)
                self._active_render_row = None
                self._active_render_page = None
                self.rows_stack.setCurrentWidget(page)
                self._finish_seamless_review_swap(page)
                self._finish_rows_rebuild(final=True)
                self._restore_review_scroll(row)
                self._start_review_page_transition(transition_pixmap)
                self._queue_review_page_preloads(row)
                self._trace_review_perf(
                    "render_piece_sync_failed_done",
                    trace_started,
                    force=True,
                    row=row,
                    rows=len(rows),
                    error=str(exc)[:160],
                )
                return

        render_timer = QTimer(self)
        render_timer.setSingleShot(True)
        row_state = {"idx": 0, "content_height": 0}
        batch_size = 12
        try:
            layout_spacing = layout.spacing()
            if layout_spacing < 0:
                layout_spacing = 4
        except Exception:
            layout_spacing = 4
        # Anchor rows to the top for the whole stream: with the stretch in
        # place from the start, any surplus page height collects BELOW the
        # rows instead of being distributed between them, so already-placed
        # rows never move while later entries stream in.
        try:
            layout.addStretch(1)
        except Exception:
            pass

        # Don't swap to the (still empty) page or to a loading screen here.
        # The previous content stays visible and the swap happens inside the
        # first batch tick, once enough rows exist to fill the viewport - so
        # switching entries never flashes a blank or loading page.
        swap_state = {"pending": bool(show_loading)}
        self._trace_review_perf(
            "render_piece_async_start",
            trace_started,
            force=True,
            row=row,
            rows=len(rows),
        )

        def _finish_active_render_timer():
            if self._active_render_timer is render_timer:
                self._active_render_timer = None
            try:
                render_timer.stop()
                render_timer.deleteLater()
            except Exception:
                pass

        def _discard_active_render_page():
            self._discard_piece_page(row, page)
            if self._active_render_page is page:
                self._active_render_row = None
                self._active_render_page = None

        def _run_render_batch():
            if self._active_render_timer is not render_timer:
                return
            if render_token != self._render_token:
                _finish_active_render_timer()
                _discard_active_render_page()
                return
            try:
                visible_stream = swap_state["pending"] or self.rows_stack.currentWidget() is page
                stream_widgets = tuple(
                    widget for widget in (page, self.rows_stack, self.scroll.viewport(), self.scroll)
                    if widget is not None
                ) if visible_stream else ()
                try:
                    for widget in stream_widgets:
                        try:
                            widget.setUpdatesEnabled(False)
                        except Exception:
                            pass
                    self.rows_widget = page
                    self.rows_layout = layout
                    start = row_state["idx"]
                    total_rows = len(rows)
                    batch_frames = []
                    # Time-boxed batches keep each tick short enough that the
                    # event loop stays responsive (scrolling, painting, input).
                    scrolling = self._review_scroll_recently_active()
                    batch_budget_s = 0.004 if scrolling else 0.008
                    batch_started_s = time.monotonic()
                    idx = start
                    while idx < total_rows:
                        row_model = row_models[idx] if idx < len(row_models) else None
                        frame = self._add_review_row(
                            piece,
                            rows[idx],
                            idx,
                            max_len,
                            colors,
                            row_model=row_model,
                            updates_enabled=False,
                        )
                        idx += 1
                        if frame is not None:
                            batch_frames.append(frame)
                            if row_state["content_height"] > 0:
                                row_state["content_height"] += layout_spacing
                            row_state["content_height"] += max(0, frame.minimumHeight())
                        if len(batch_frames) >= batch_size:
                            break
                        if (time.monotonic() - batch_started_s) >= batch_budget_s:
                            break
                    row_state["idx"] = idx
                    if visible_stream:
                        # Incremental geometry: rows have fixed heights, so the
                        # content height is tracked as rows are added instead of
                        # re-measuring the entire page layout every batch.
                        self._apply_review_stream_geometry(page, layout, row_state["content_height"])
                    if swap_state["pending"]:
                        # First batch rendered: swap from the old content to
                        # the new page in the same updates-disabled tick, so
                        # the switch paints exactly once with real rows.
                        swap_state["pending"] = False
                        self.rows_stack.setCurrentWidget(page)
                        self._finish_seamless_review_swap(page)
                        self._restore_review_scroll(row)
                        self._start_review_page_transition(transition_pixmap)
                        self._trace_review_perf(
                            "render_piece_async_first_swap",
                            trace_started,
                            force=True,
                            row=row,
                            rendered=row_state["idx"],
                            total=len(rows),
                        )
                    for frame in batch_frames:
                        try:
                            frame.setUpdatesEnabled(True)
                        except Exception:
                            pass
                finally:
                    for widget in stream_widgets:
                        try:
                            widget.setUpdatesEnabled(True)
                            widget.update()
                        except Exception:
                            pass

                if row_state["idx"] < len(rows):
                    # Yield a full frame between batches so scrolling/painting
                    # stays smooth while the rest of the rows stream in.
                    render_timer.start(16)
                    return

                if render_token != self._render_token:
                    _finish_active_render_timer()
                    _discard_active_render_page()
                    return
                if self._review_layout_trailing_stretch_index(layout) < 0:
                    layout.addStretch(1)
                self._piece_render_complete.add(row)
                self.rows_stack.setCurrentWidget(page)
                self._finish_seamless_review_swap(page)
                self._finish_rows_rebuild(final=True)
                self._restore_review_scroll(row)
                if self._active_render_page is page:
                    self._active_render_row = None
                    self._active_render_page = None
                _finish_active_render_timer()
                self._queue_review_page_preloads(row)
                self._trace_review_perf(
                    "render_piece_async_done",
                    trace_started,
                    force=True,
                    row=row,
                    rows=len(rows),
                )
            except Exception as exc:
                _finish_active_render_timer()
                if render_token == self._render_token:
                    self._clear_rows(layout)
                    error = QLabel(f"Could not render SDLXLIFF review rows:\n{exc}")
                    error.setTextFormat(Qt.PlainText)
                    error.setStyleSheet(f"color: {self.THEME['danger']}; font-size: 11pt; padding: 12px;")
                    layout.addWidget(error)
                    layout.addStretch(1)
                    self._piece_render_complete.add(row)
                    self.rows_stack.setCurrentWidget(page)
                    self._finish_seamless_review_swap(page)
                    self._finish_rows_rebuild(final=True)
                    self._restore_review_scroll(row)
                    self._start_review_page_transition(transition_pixmap)
                    self._queue_review_page_preloads(row)
                if self._active_render_page is page:
                    self._active_render_row = None
                    self._active_render_page = None
                self._trace_review_perf(
                    "render_piece_async_failed",
                    trace_started,
                    force=True,
                    row=row,
                    rows=len(rows),
                    error=str(exc)[:160],
                )

        render_timer.timeout.connect(_run_render_batch)
        self._active_render_timer = render_timer
        render_timer.start(0)


class RetranslationMixin(ProgressViewMixin, SdlxliffAutogenMixin):
    """Mixin class containing retranslation methods for TranslatorGUI"""

    # -- ProgressViewMixin hooks: the original desktop behaviour ----------------

    def _progress_reload_error(self, data, title, message):
        """A refresh could not recreate the progress file: warn in the dialog."""
        self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), title, message)

    def _progress_cleanup_ready(self):
        """Clean up only once TransateKRtoEN has finished importing (never block on its
        import lock from the GUI thread; this also warms the module in the background)."""
        return _get_progress_manager_nonblocking() is not None

    def _progress_raw_input_recorder(self):
        """Keep the epub_library registry import (builds without epub_library skip it)."""
        return None

    def _start_manual_glossary_refinement(
        self,
        glossary_path,
        progress_path,
        options,
        plan,
    ):
        """Start an explicit refinement pass in the normal glossary worker slot."""
        if hasattr(self, '_is_any_process_running') and self._is_any_process_running():
            self._show_message(
                'warning',
                'Process Running',
                'Please wait for the current translation, glossary, or refinement process to finish.',
                parent=self,
            )
            return False
        if not glossary_path or not os.path.isfile(glossary_path):
            self._show_message(
                'warning',
                'Glossary Not Found',
                'No saved glossary file was found for this progress entry.',
                parent=self,
            )
            return False

        self.stop_requested = False
        self.graceful_stop_active = False
        try:
            import extract_glossary_from_epub as extractor
            extractor.set_stop_flag(False)
        except Exception:
            extractor = None

        def _worker():
            try:
                self._run_manual_glossary_refinement(
                    glossary_path,
                    progress_path,
                    options,
                    plan,
                )
            except Exception as exc:
                if hasattr(self, 'append_log'):
                    self.append_log(f"❌ Manual glossary refinement failed: {exc}")
            finally:
                if getattr(self, 'stop_requested', False):
                    self._glossary_stop_was_requested = True
                self.stop_requested = False
                try:
                    import extract_glossary_from_epub as _manual_refinement_extractor
                    _manual_refinement_extractor.set_stop_flag(False)
                except Exception:
                    pass
                self.glossary_thread = None
                if hasattr(self, 'glossary_future'):
                    self.glossary_future = None
                try:
                    self.thread_complete_signal.emit()
                except Exception:
                    pass

        if hasattr(self, 'append_log'):
            scope = ', '.join(options.selected_types or [])
            self.append_log(f"\n✨ Starting manual glossary refinement: {scope}")
        try:
            if hasattr(self, '_ensure_executor'):
                self._ensure_executor()
        except Exception:
            pass
        executor = getattr(self, 'executor', None)
        if executor is not None:
            self.glossary_future = executor.submit(_worker)
        else:
            self.glossary_thread = threading.Thread(
                target=_worker,
                name=f"GlossaryRefinementThread_{int(time.time())}",
                daemon=True,
            )
            self.glossary_thread.start()
        try:
            self.update_run_button()
        except Exception:
            pass
        return True

    def _run_manual_glossary_refinement(
        self,
        glossary_path,
        progress_path,
        options,
        plan,
    ):
        """Execute and atomically persist one explicit refinement plan."""
        run_manual_glossary_refinement(
            self,
            glossary_path,
            progress_path,
            options,
            plan,
        )

    def _show_no_sdlxliff_review_available(self, parent=None):
        self._show_message(
            'info',
            "Text Analysis Unavailable",
            "No SDLXLIFF sidecars were found for this output folder.",
            parent=parent or self,
        )

    def _open_or_reuse_sdlxliff_review(
        self,
        output_dir,
        review_path=None,
        parent=None,
        autogen_file_path=None,
        autogen_progress_data=None,
        autogen_output_files=None,
        autogen_manual_entries=None,
    ):
        try:
            key = os.path.normcase(os.path.abspath(output_dir))
        except Exception:
            key = str(output_dir or "")

        manual_entries = list(autogen_manual_entries or [])
        manual_editing_enabled = self._get_retranslation_manual_editing_state()
        if (
            not self._output_dir_has_sdlxliff_sidecars(output_dir)
            and not self._output_dir_has_sdlxliff_generatable_html(
                output_dir,
                progress_data=autogen_progress_data,
                output_files=autogen_output_files,
            )
            and not (manual_editing_enabled and manual_entries)
        ):
            self._show_no_sdlxliff_review_available(parent or self)
            return None

        cache = getattr(self, "_sdlxliff_review_dialog_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_sdlxliff_review_dialog_cache", cache)

        review_dialog = cache.get(key)
        if review_dialog is not None:
            try:
                review_dialog.isVisible()
            except RuntimeError:
                review_dialog = None
                cache.pop(key, None)

        if review_dialog is None:
            review_dialog = SDLXLIFFReviewDialog(
                output_dir,
                review_path,
                parent or self,
                config=getattr(self, 'config', {}),
                autogen_owner=self,
                autogen_file_path=autogen_file_path,
                autogen_progress_data=autogen_progress_data,
                autogen_output_files=autogen_output_files,
                autogen_manual_entries=manual_entries,
            )
            review_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
            cache[key] = review_dialog

            def _forget_dialog():
                try:
                    cache.pop(key, None)
                except Exception:
                    pass

            review_dialog.destroyed.connect(_forget_dialog)
        else:
            try:
                review_dialog._sdlxliff_autogen_owner = self
                review_dialog._sdlxliff_autogen_file_path = autogen_file_path
                review_dialog._sdlxliff_autogen_progress_data = autogen_progress_data
                review_dialog._sdlxliff_autogen_output_files = list(autogen_output_files or []) or None
                review_dialog._sdlxliff_autogen_manual_entries = manual_entries
            except Exception:
                pass
            review_dialog.reopen_for_path(output_dir, review_path)
            try:
                if autogen_output_files or not self._output_dir_has_sdlxliff_sidecars(output_dir):
                    review_dialog._queue_review_refresh_scan(
                        force=False,
                        current_path=review_path or review_dialog.current_path,
                        delay_ms=25,
                    )
            except Exception:
                pass

        review_dialog.show()
        review_dialog.raise_()
        review_dialog.activateWindow()
        return review_dialog


    def _sync_metadata_progress_toggle(self, enabled):
        """Immediately refresh open/cached Progress Managers after a toggle change."""
        enabled = bool(enabled)
        self.translate_book_title_var = enabled
        if isinstance(getattr(self, 'config', None), dict):
            self.config['translate_book_title'] = enabled

        cached_data = []
        cache = getattr(self, '_retranslation_dialog_cache', {})
        if isinstance(cache, dict):
            cached_data.extend(value for value in cache.values() if isinstance(value, dict))

        multi_dialog = getattr(self, '_multi_file_retranslation_dialog', None)
        if multi_dialog is not None:
            cached_data.extend(
                value for value in getattr(multi_dialog, '_tab_data', [])
                if isinstance(value, dict)
            )

        seen = set()
        for data in cached_data:
            data_id = id(data)
            if data_id in seen or not data.get('progress_file'):
                continue
            seen.add(data_id)
            # Discard any background-prefetched snapshot because it reflects
            # the old setting and would otherwise restore the removed row.
            data.pop('_prefetched_prog', None)
            data.pop('_prefetched_prog_path', None)
            data['_refresh_read_only'] = False

            def _refresh(progress_data=data):
                try:
                    if self._is_data_valid(progress_data):
                        self._refresh_retranslation_data(progress_data)
                except Exception as exc:
                    print(f"⚠️ Could not sync metadata progress visibility: {exc}")

            QTimer.singleShot(0, _refresh)

    @staticmethod
    def _normalize_progress_manager_input_path(path):
        value = str(path or '').strip()
        if not value or value == '__generative_mode__':
            return ''
        try:
            return os.path.normcase(os.path.normpath(os.path.abspath(value)))
        except (OSError, TypeError, ValueError):
            return value

    def _progress_manager_current_input_signature(self):
        """Return the stable input selection represented by the main file field."""
        selected = [
            str(path).strip()
            for path in (getattr(self, 'selected_files', None) or [])
            if str(path or '').strip() and str(path).strip() != '__generative_mode__'
        ]
        try:
            entry_text = self.entry_epub.text().strip()
        except (AttributeError, RuntimeError):
            entry_text = ''

        # A real path typed into the single-file field is more current than a
        # stale selected_files value. Multi-file summary text is never a path.
        if (
            entry_text
            and 'files selected' not in entry_text
            and not entry_text.startswith('No file selected')
            and (os.path.isfile(entry_text) or os.path.isdir(entry_text))
        ):
            selected = [entry_text]

        normalized = {
            self._normalize_progress_manager_input_path(path)
            for path in selected
        }
        normalized.discard('')
        return tuple(sorted(normalized))

    def _stamp_progress_manager_input_signature(self, dialog):
        if dialog is not None:
            dialog._progress_manager_input_signature = (
                self._progress_manager_current_input_signature()
            )

    def _schedule_progress_manager_input_path_refresh(self, *_args):
        """Debounce input edits before rebuilding any visible Progress Manager."""
        timer = getattr(self, '_progress_input_refresh_timer', None)
        if timer is None:
            try:
                timer = QTimer(self)
            except TypeError:
                timer = QTimer()
            timer.setSingleShot(True)
            timer.setInterval(125)
            timer.timeout.connect(self._refresh_open_progress_managers_for_input_change)
            self._progress_input_refresh_timer = timer
        timer.start()

    def _progress_manager_dialogs(self):
        dialogs = []
        seen = set()

        def _add(dialog):
            if dialog is None or id(dialog) in seen:
                return
            seen.add(id(dialog))
            dialogs.append(dialog)

        cache = getattr(self, '_retranslation_dialog_cache', None)
        if isinstance(cache, dict):
            for value in cache.values():
                if isinstance(value, dict):
                    _add(value.get('dialog'))
        _add(getattr(self, '_multi_file_retranslation_dialog', None))

        image_cache = getattr(self, '_image_retranslation_dialog_cache', None)
        if isinstance(image_cache, dict):
            for dialog in image_cache.values():
                _add(dialog)
        return dialogs

    def _reopen_glossary_progress_after_input_change(self, attempt=0):
        for dialog in self._progress_manager_dialogs():
            try:
                if not dialog.isVisible():
                    continue
            except RuntimeError:
                continue
            opener = getattr(dialog, '_show_glossary_progress', None)
            if not callable(opener):
                continue
            opener()

            def _force_glossary_refresh(target_dialog=dialog):
                gp_dialog = getattr(target_dialog, '_glossary_progress_dialog', None)
                full_refresh = getattr(gp_dialog, '_gp_full_refresh', None)
                if callable(full_refresh):
                    full_refresh()

            QTimer.singleShot(0, _force_glossary_refresh)
            return
        if attempt < 30:
            QTimer.singleShot(
                100,
                lambda: self._reopen_glossary_progress_after_input_change(attempt + 1),
            )

    def _refresh_open_progress_managers_for_input_change(self):
        """Fully rebuild visible progress dialogs when the selected input changes."""
        if getattr(self, '_progress_input_rebuild_running', False):
            return
        current_signature = self._progress_manager_current_input_signature()
        if not current_signature:
            return

        changed_dialogs = []
        reopen_glossary = False
        for dialog in self._progress_manager_dialogs():
            try:
                if not dialog.isVisible():
                    continue
            except RuntimeError:
                continue
            previous_signature = getattr(
                dialog, '_progress_manager_input_signature', None
            )
            if previous_signature is None:
                dialog._progress_manager_input_signature = current_signature
                continue
            if tuple(previous_signature) == current_signature:
                continue
            changed_dialogs.append(dialog)
            try:
                gp_dialog = getattr(dialog, '_glossary_progress_dialog', None)
                reopen_glossary = reopen_glossary or bool(
                    gp_dialog is not None and gp_dialog.isVisible()
                )
            except RuntimeError:
                pass

        if not changed_dialogs:
            return

        # Keep the underlying selection synchronized when the user typed a
        # valid single path directly into the input field. force_retranslation()
        # consults selected_files before the line edit for folders and bundles.
        try:
            entry_text = self.entry_epub.text().strip()
            normalized_entry = self._normalize_progress_manager_input_path(entry_text)
            if (
                len(current_signature) == 1
                and normalized_entry == current_signature[0]
                and (os.path.isfile(entry_text) or os.path.isdir(entry_text))
            ):
                self.selected_files = [entry_text]
                self.file_path = entry_text
        except (AttributeError, RuntimeError):
            pass

        changed_ids = {id(dialog) for dialog in changed_dialogs}
        cache = getattr(self, '_retranslation_dialog_cache', None)
        if isinstance(cache, dict):
            for key, value in list(cache.items()):
                if isinstance(value, dict) and id(value.get('dialog')) in changed_ids:
                    cache.pop(key, None)

        multi_dialog = getattr(self, '_multi_file_retranslation_dialog', None)
        if id(multi_dialog) in changed_ids:
            self._multi_file_retranslation_dialog = None
            self._multi_file_selection_key = None

        image_cache = getattr(self, '_image_retranslation_dialog_cache', None)
        if isinstance(image_cache, dict):
            for key, dialog in list(image_cache.items()):
                if id(dialog) in changed_ids:
                    image_cache.pop(key, None)

        pending = getattr(self, '_subtitle_bundle_progress_shell', None)
        if isinstance(pending, dict) and id(pending.get('dialog')) in changed_ids:
            self._subtitle_bundle_progress_shell = None

        self._progress_input_rebuild_running = True
        for dialog in changed_dialogs:
            dialog._progress_input_retired = True
            try:
                gp_dialog = getattr(dialog, '_glossary_progress_dialog', None)
                if gp_dialog is not None:
                    gp_dialog.hide()
                    gp_dialog.deleteLater()
            except RuntimeError:
                pass
            try:
                dialog.hide()
                dialog.deleteLater()
            except RuntimeError:
                pass

        def _rebuild_for_new_input():
            try:
                self.force_retranslation()
                if reopen_glossary:
                    self._reopen_glossary_progress_after_input_change()
            except Exception as exc:
                print(f"⚠️ Could not rebuild Progress Manager for new input: {exc}")
            finally:
                self._progress_input_rebuild_running = False

        QTimer.singleShot(0, _rebuild_for_new_input)

    def _sync_translation_artifact_progress_toggle(self, kind, enabled):
        """Refresh Progress Manager rows after TOC/header toggle changes."""
        spec = translation_artifact_spec_for_kind(kind)
        if not spec:
            return
        enabled = bool(enabled)
        setattr(self, spec['toggle_attr'], enabled)
        if isinstance(getattr(self, 'config', None), dict):
            self.config[spec['toggle_config']] = enabled
            fallback_key = spec.get('toggle_fallback_config')
            if fallback_key:
                self.config[fallback_key] = enabled
        os.environ[spec['toggle_env']] = '1' if enabled else '0'
        if spec['kind'] == 'toc':
            self.translate_toc_ncx_var = enabled
            os.environ['TRANSLATE_TOC_NCX'] = '1' if enabled else '0'

        cached_data = []
        cache = getattr(self, '_retranslation_dialog_cache', {})
        if isinstance(cache, dict):
            cached_data.extend(
                value for value in cache.values() if isinstance(value, dict)
            )
        multi_dialog = getattr(self, '_multi_file_retranslation_dialog', None)
        if multi_dialog is not None:
            cached_data.extend(
                value for value in getattr(multi_dialog, '_tab_data', [])
                if isinstance(value, dict)
            )

        seen = set()
        for data in cached_data:
            data_id = id(data)
            if data_id in seen or not data.get('progress_file'):
                continue
            seen.add(data_id)
            data.pop('_prefetched_prog', None)
            data.pop('_prefetched_prog_path', None)
            data['_refresh_read_only'] = False

            def _refresh(progress_data=data):
                try:
                    if self._is_data_valid(progress_data):
                        self._refresh_retranslation_data(progress_data)
                except Exception as exc:
                    print(
                        f"⚠️ Could not sync translation artifact progress "
                        f"visibility: {exc}"
                    )

            QTimer.singleShot(0, _refresh)

    def _start_single_progress_qa_resolution(self, data, display_info):
        """Run Partial.b for exactly one foreign-text QA progress entry."""
        data = data if isinstance(data, dict) else {}
        display_info = display_info if isinstance(display_info, dict) else {}
        # Target lookup, refusals and the run state are the shared preflight
        # (progress_actions.prepare_single_qa_resolution).
        preflight = prepare_single_qa_resolution(self, data, display_info)
        if preflight['refusal']:
            refusal_kind, refusal_title, refusal_message = preflight['refusal']
            self._show_message(
                refusal_kind,
                refusal_title,
                refusal_message,
                parent=data.get('dialog', self),
            )
            if preflight['refresh']:
                try:
                    self._refresh_retranslation_data(data)
                except Exception:
                    pass
            return False

        source_path = preflight['source_path']
        try:
            if hasattr(self, 'entry_epub') and self.entry_epub is not None:
                self.entry_epub.setText(source_path)
        except Exception:
            pass

        try:
            self.append_log(preflight['log'])
        except Exception:
            pass
        self.run_translation_thread()
        started = bool(
            getattr(self, 'translation_thread', None)
            and self.translation_thread.is_alive()
        )
        if not started:
            self._single_qa_resolution_request = None
        return started



    def _apply_compact_inline_list_style(self, listbox, font=None, extra_row_px=0):
        """Use dense row spacing for inline status/list views."""
        try:
            if font is not None:
                listbox.setFont(font)
            listbox.setProperty("_compact_inline_extra_row_px", max(0, int(extra_row_px or 0)))
            listbox.setSpacing(0)
            listbox.setUniformItemSizes(True)
            listbox.setStyleSheet("""
                QListWidget {
                    outline: 0;
                }
                QListWidget::item {
                    margin: 0px;
                    padding: 0px 2px;
                }
            """)
        except Exception:
            pass

    def _set_compact_inline_item_size(self, listbox, item):
        try:
            extra_row_px = int(listbox.property("_compact_inline_extra_row_px") or 0)
            height = max(18, listbox.fontMetrics().lineSpacing() + 2) + extra_row_px
            item.setSizeHint(QSize(0, height))
        except Exception:
            pass
        return item

    def _add_compact_inline_list_item(self, listbox, item_or_text):
        item = item_or_text if isinstance(item_or_text, QListWidgetItem) else QListWidgetItem(str(item_or_text))
        self._set_compact_inline_item_size(listbox, item)
        listbox.addItem(item)
        return item
    
    def _ui_yield(self, ms=5):
        """Let the Qt event loop process pending events briefly."""
        try:
            if getattr(self, '_suspend_yield', False):
                return
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents(QEventLoop.AllEvents, ms)
        except Exception:
            pass
    
    def _clear_layout(self, layout):
        """Safely clear all items from a layout"""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
                elif item.layout():
                    self._clear_layout(item.layout())
    
    def _get_dialog_size(self, width_ratio=0.5, height_ratio=0.5):
        """Calculate dialog size as a ratio of screen size (default 50% width, 50% height)"""
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtGui import QScreen
            
            # Get primary screen
            screen = QApplication.primaryScreen()
            if screen:
                geometry = screen.availableGeometry()
                width = int(geometry.width() * width_ratio)
                height = int(geometry.height() * height_ratio)
                return width, height
        except:
            pass
        
        # Fallback to reasonable defaults if screen info unavailable
        return int(1920 * width_ratio), int(1080 * height_ratio)
    
    def _show_message(self, msg_type, title, message, parent=None):
        """Show message using PySide6 QMessageBox with Halgakos icon"""
        try:
            # Create message box
            msg_box = QMessageBox(parent)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            
            # Set icon based on message type
            if msg_type == 'info':
                msg_box.setIcon(QMessageBox.Information)
            elif msg_type == 'warning':
                msg_box.setIcon(QMessageBox.Warning)
            elif msg_type == 'error':
                msg_box.setIcon(QMessageBox.Critical)
            elif msg_type == 'question':
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
            # Center buttons
            msg_box.setStyleSheet("""
                QPushButton {
                    min-width: 80px;
                    min-height: 30px;
                    padding: 6px 20px;
                    font-size: 10pt;
                }
                QDialogButtonBox {
                    qproperty-centerButtons: true;
                }
            """)
            
            # Try to set Halgakos window icon
            try:
                from PySide6.QtGui import QIcon
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    import sys
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    msg_box.setWindowIcon(QIcon(ico_path))
            except:
                pass
            
            # Show message box
            if msg_type == 'question':
                # Ensure dialog stays on top if it's a critical question
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                return msg_box.exec() == QMessageBox.Yes
            else:
                msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                msg_box.exec()
                return True
                
        except Exception as e:
            # Fallback to console if dialog fails
            print(f"{title}: {message}")
            if msg_type == 'question':
                return False
            return False

    @staticmethod
    def _styled_msgbox(icon, parent, title, message, buttons=None):
        """Create a QMessageBox with centered buttons and return the result.
        
        Usage (replaces static convenience methods):
            QMessageBox.information(p, t, m)  →  self._styled_msgbox(QMessageBox.Information, p, t, m)
            QMessageBox.warning(p, t, m)      →  self._styled_msgbox(QMessageBox.Warning, p, t, m)
            QMessageBox.question(p, t, m, b)  →  self._styled_msgbox(QMessageBox.Question, p, t, m, b)
            QMessageBox.critical(p, t, m)     →  self._styled_msgbox(QMessageBox.Critical, p, t, m)
        """
        msg = QMessageBox(parent)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(message)
        if buttons is not None:
            msg.setStandardButtons(buttons)
        msg.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                padding: 6px 20px;
                font-size: 10pt;
            }
            QDialogButtonBox {
                qproperty-centerButtons: true;
            }
        """)
        result = msg.exec()
        return result

    @staticmethod
    def _recycled_artifact_retranslation_choice(
        parent,
        message,
        counterpart_filename,
    ):
        """Ask whether a RECYCLED artifact reset should include its source."""
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Confirm Linked Retranslation")
        msg.setText(message)
        delete_both_button = msg.addButton(
            "Delete Both Linked Files",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        keep_counterpart_button = msg.addButton(
            f"Keep {counterpart_filename}",
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(cancel_button)
        msg.setStyleSheet("""
            QPushButton {
                min-width: 170px;
                min-height: 38px;
                padding: 7px 18px;
                font-size: 10pt;
            }
            QDialogButtonBox {
                qproperty-centerButtons: true;
            }
        """)
        msg.exec()
        clicked_button = msg.clickedButton()
        if clicked_button is delete_both_button:
            return "both"
        if clicked_button is keep_counterpart_button:
            return "selected_only"
        return "cancel"
 
    def _flash_pm_button_green(self, folder_path=None):
        """Flash the Progress Manager button green to indicate a new folder was created.
        Also plays a Windows sound and stores the folder path for the dialog status row."""
        try:
            pm_btn = getattr(self, 'pm_button', None)
            if pm_btn is None:
                return

            # Remember the definitive original style (first call wins)
            # This prevents re-capturing an already-green style on rapid re-calls
            if not hasattr(self, '_pm_original_style') or not self._pm_original_style:
                self._pm_original_style = pm_btn.styleSheet()

            # Flash to green using the definitive original as base
            import re as _re
            green_style = _re.sub(
                r'background-color:\s*#[0-9a-fA-F]+',
                'background-color: #27ae60',
                self._pm_original_style,
                count=1
            )
            pm_btn.setStyleSheet(green_style)

            # Restore using the definitive original style after 1.5 seconds
            def _restore_pm_style():
                try:
                    pm_btn.setStyleSheet(self._pm_original_style)
                except Exception:
                    pass
            QTimer.singleShot(1500, _restore_pm_style)

            # Play Windows system sound
            try:
                import platform
                if platform.system() == 'Windows':
                    import winsound
                    winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass

            # Store the created folder path so the dialog stats row can show it
            if folder_path:
                self._pm_created_folder = folder_path
        except Exception as e:
            print(f"⚠️ Could not flash PM button: {e}")

    def _create_retranslation_shell_dialog(self, title="Progress Manager", width_ratio=0.44, height_ratio=0.4):
        from PySide6.QtWidgets import QApplication
        if not QApplication.instance():
            QApplication(sys.argv)

        parent_widget = self if isinstance(self, QWidget) else None
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.NonModal)
        width, height = self._get_dialog_size(width_ratio, height_ratio)
        dialog.resize(width, height)
        dialog.setMinimumSize(width, height)

        try:
            if parent_widget is not None:
                ss = parent_widget.styleSheet()
                if ss:
                    dialog.setStyleSheet(ss)
        except Exception:
            pass

        base_dir = None
        ico_path = None
        try:
            if hasattr(self, 'base_dir'):
                base_dir = self.base_dir
            else:
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                dialog.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Failed to load icon: {e}")

        dialog_layout = QVBoxLayout(dialog)
        loading_widget = QWidget(dialog)
        loading_layout = QVBoxLayout(loading_widget)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        loading_layout.setSpacing(10)
        loading_layout.addStretch(1)

        try:
            try:
                from spinning import create_icon_label
            except Exception:
                from .spinning import create_icon_label
            loading_icon = create_icon_label(52, base_dir)
            loading_icon.setFixedSize(52, 52)
            loading_layout.addWidget(loading_icon, 0, Qt.AlignCenter)

            dialog._loading_icon_label = loading_icon
            dialog._loading_icon_angle = 0
            pixmap = loading_icon.pixmap()
            dialog._loading_icon_original_pixmap = pixmap.copy() if pixmap and not pixmap.isNull() else None

            spin_timer = QTimer(dialog)

            def _spin_loading_icon():
                icon = getattr(dialog, '_loading_icon_label', None)
                original = getattr(dialog, '_loading_icon_original_pixmap', None)
                if icon is None or original is None or original.isNull():
                    return
                try:
                    dialog._loading_icon_angle = (getattr(dialog, '_loading_icon_angle', 0) + 24) % 360
                    rotated = original.transformed(QTransform().rotate(dialog._loading_icon_angle), Qt.SmoothTransformation)
                    if rotated.isNull():
                        return
                    icon.setPixmap(rotated.scaled(
                        icon.size().width(),
                        icon.size().height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    ))
                    icon.update()
                except RuntimeError:
                    spin_timer.stop()
                except Exception:
                    pass

            spin_timer.timeout.connect(_spin_loading_icon)
            spin_timer.start(45)
            QTimer.singleShot(0, _spin_loading_icon)
            dialog._loading_icon_timer = spin_timer
            dialog._advance_loading_icon = _spin_loading_icon
        except Exception:
            pass

        loading_label = QLabel("Loading progress...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("color: #94a3b8; font-size: 12pt; font-weight: bold; padding: 24px;")
        loading_layout.addWidget(loading_label)
        loading_layout.addStretch(1)
        dialog_layout.addWidget(loading_widget)
        return dialog, dialog_layout, loading_widget, loading_label

    def _show_retranslation_shell_then_build(
        self,
        file_path,
        show_special_files_state=False,
        resolved_output_dir=None,
        cache_key=None,
        glossary_progress_source_path=None,
        glossary_progress_source_filenames=None,
    ):
        dialog, dialog_layout, loading_widget, loading_label = self._create_retranslation_shell_dialog("Progress Manager")
        self._stamp_progress_manager_input_signature(dialog)
        file_key = cache_key or os.path.abspath(file_path)

        def closeEvent(event):
            event.ignore()
            dialog.hide()

        dialog.closeEvent = closeEvent
        dialog.show()
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents(QEventLoop.AllEvents, 50)
        except Exception:
            pass

        def _build_dialog_contents():
            try:
                if getattr(dialog, '_progress_input_retired', False):
                    return
                content = QWidget(dialog)
                content_layout = QVBoxLayout(content)
                content_layout.setContentsMargins(0, 0, 0, 0)

                result = self._force_retranslation_epub_or_text(
                    file_path,
                    parent_dialog=dialog,
                    tab_frame=content,
                    show_special_files_state=show_special_files_state,
                    _loading_label=loading_label,
                    resolved_output_dir=resolved_output_dir,
                    glossary_progress_source_path=glossary_progress_source_path,
                    glossary_progress_source_filenames=(
                        glossary_progress_source_filenames
                    ),
                )
                if not result:
                    dialog.hide()
                    pending = getattr(self, '_subtitle_bundle_progress_shell', None)
                    if isinstance(pending, dict) and pending.get('dialog') is dialog:
                        self._subtitle_bundle_progress_shell = None
                    return

                timer = getattr(dialog, '_loading_icon_timer', None)
                if timer:
                    timer.stop()
                loading_widget.hide()
                loading_widget.deleteLater()
                dialog_layout.addWidget(content)

                dialog.setWindowTitle("Progress Manager - OPF Based" if result.get('spine_chapters') else "Progress Manager")
                if not hasattr(self, '_retranslation_dialog_cache'):
                    self._retranslation_dialog_cache = {}
                self._retranslation_dialog_cache[file_key] = result
                pending = getattr(self, '_subtitle_bundle_progress_shell', None)
                if isinstance(pending, dict) and pending.get('dialog') is dialog:
                    self._subtitle_bundle_progress_shell = None
                QTimer.singleShot(50, lambda: self._populate_progress_listbox_streamed(result))
            except Exception as e:
                print(f"Failed to build progress manager contents: {e}")
                import traceback
                traceback.print_exc()
                try:
                    loading_label.setText(f"Failed to load progress:\n{e}")
                    loading_label.show()
                except Exception:
                    pass
                pending = getattr(self, '_subtitle_bundle_progress_shell', None)
                if isinstance(pending, dict) and pending.get('dialog') is dialog:
                    self._subtitle_bundle_progress_shell = None

        QTimer.singleShot(50, _build_dialog_contents)
        return dialog

    def _selected_subtitle_bundle_progress_target(self, selected_files=None):
        """Resolve extracted members from one subtitle ZIP to one PM target."""
        files = list(
            selected_files
            if selected_files is not None
            else (getattr(self, 'selected_files', None) or [])
        )
        if not files:
            return None

        output_info_for = getattr(self, '_subtitle_zip_output_info', None)
        if not callable(output_info_for):
            return None

        infos = []
        for file_path in files:
            info = output_info_for(file_path)
            if not isinstance(info, dict):
                return None
            infos.append(info)

        def _normalized(path):
            return os.path.normcase(os.path.abspath(str(path or "")))

        bundle_ids = {
            _normalized(info.get('bundle_id') or info.get('archive_path'))
            for info in infos
            if info.get('bundle_id') or info.get('archive_path')
        }
        if len(bundle_ids) != 1:
            return None

        expected_members = {
            _normalized(member)
            for info in infos
            for member in (info.get('bundle_files') or [])
            if member
        }
        selected_members = {_normalized(file_path) for file_path in files}
        # Do not collapse an intentional subset or a mixed selection. Translation
        # expands a ZIP to the complete bundle, so an exact match identifies the
        # race that previously produced one Progress Manager tab per subtitle.
        if expected_members and selected_members != expected_members:
            return None

        output_dirs = {
            _normalized(info.get('output_dir'))
            for info in infos
            if info.get('output_dir')
        }
        if len(output_dirs) != 1:
            return None

        first = infos[0]
        archive_path = first.get('archive_path') or first.get('bundle_id')
        if not archive_path:
            return None
        bundle_files = list(first.get('bundle_files') or files)
        return {
            'bundle_id': next(iter(bundle_ids)),
            'archive_path': os.path.abspath(str(archive_path)),
            'output_dir': os.path.abspath(str(first.get('output_dir'))),
            'group_name': str(first.get('group_name') or ''),
            'bundle_files': [os.path.abspath(str(path)) for path in bundle_files],
        }

    def _open_subtitle_bundle_progress_manager(self, target):
        """Open or refresh the single Progress Manager for a subtitle ZIP."""
        archive_path = os.path.abspath(str(target['archive_path']))
        output_dir = os.path.abspath(str(target['output_dir']))
        file_key = archive_path

        # A multi-file window may have been created immediately before the ZIP
        # extraction mapping became available. Retire it so it cannot reappear
        # with one tab per extracted subtitle.
        old_multi_dialog = getattr(self, '_multi_file_retranslation_dialog', None)
        if old_multi_dialog is not None:
            try:
                old_multi_dialog.hide()
            except Exception:
                pass
            try:
                old_multi_dialog.deleteLater()
            except Exception:
                pass
            self._multi_file_retranslation_dialog = None
            self._multi_file_selection_key = None

        cache = getattr(self, '_retranslation_dialog_cache', None)
        cached_data = cache.get(file_key) if isinstance(cache, dict) else None
        if isinstance(cached_data, dict) and cached_data.get('dialog'):
            cached_output = cached_data.get('output_dir')
            if cached_output and os.path.normcase(os.path.abspath(cached_output)) == os.path.normcase(output_dir):
                dialog = cached_data['dialog']
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()

                def _refresh_cached_bundle():
                    refresh_func = cached_data.get('refresh_func')
                    if callable(refresh_func):
                        try:
                            refresh_func()
                            return
                        except Exception:
                            pass
                    self._refresh_retranslation_data(cached_data)

                QTimer.singleShot(50, _refresh_cached_bundle)
                return dialog
            try:
                cached_data.get('dialog').hide()
                cached_data.get('dialog').deleteLater()
            except Exception:
                pass
            cache.pop(file_key, None)
            self._subtitle_bundle_progress_shell = None

        pending = getattr(self, '_subtitle_bundle_progress_shell', None)
        if isinstance(pending, dict) and pending.get('file_key') == file_key:
            pending_dialog = pending.get('dialog')
            if pending_dialog is not None:
                try:
                    pending_dialog.show()
                    pending_dialog.raise_()
                    pending_dialog.activateWindow()
                    return pending_dialog
                except RuntimeError:
                    self._subtitle_bundle_progress_shell = None

        dialog = self._show_retranslation_shell_then_build(
            archive_path,
            show_special_files_state=True,
            resolved_output_dir=output_dir,
        )
        self._subtitle_bundle_progress_shell = {
            'file_key': file_key,
            'dialog': dialog,
            'output_dir': output_dir,
        }
        return dialog

    def force_retranslation(self):
        """Force retranslation of specific chapters or images with improved display"""

        parallel_context_getter = getattr(
            self, "_parallel_epub_progress_manager_context", None
        )
        parallel_context = (
            parallel_context_getter()
            if callable(parallel_context_getter)
            else None
        )
        if isinstance(parallel_context, dict):
            raw_path = str(parallel_context.get("raw_path") or "")
            if raw_path and os.path.isfile(raw_path):
                self._show_retranslation_shell_then_build(
                    raw_path,
                    show_special_files_state=False,
                    cache_key=parallel_context.get("cache_key"),
                    glossary_progress_source_path=parallel_context.get(
                        "generated_path"
                    ),
                    glossary_progress_source_filenames=parallel_context.get(
                        "raw_filenames"
                    ),
                )
                return

        subtitle_bundle_target = self._selected_subtitle_bundle_progress_target()
        if subtitle_bundle_target:
            self._open_subtitle_bundle_progress_manager(subtitle_bundle_target)
            return

        # Check for multiple file selection first
        if hasattr(self, 'selected_files') and len(self.selected_files) > 1:
            self._force_retranslation_multiple_files()
            return
        
        # Check if it's a folder selection (for images)
        if hasattr(self, 'selected_files') and len(self.selected_files) > 0:
            # Check if the first selected file is actually a folder
            first_item = self.selected_files[0]
            if os.path.isdir(first_item):
                self._force_retranslation_images_folder(first_item)
                return
        
        # Original logic for single files
        # Get input path from QLineEdit widget
        if hasattr(self.entry_epub, 'text'):
            # PySide6 QLineEdit widget
            input_path = self.entry_epub.text()
        else:
            input_path = ""
        
        if not input_path or not os.path.isfile(input_path):
            self._show_message('error', "Error", "Please select a valid EPUB, text file, or image folder first.")
            return
        
        # Check if it's an image file
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        if input_path.lower().endswith(image_extensions):
            # For single image, pass the image file path itself
            self._force_retranslation_images_folder(input_path)
            return
        
        # Check if dialog already exists for this file and is just hidden
        file_key = os.path.abspath(input_path)
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            # Reuse existing dialog - just show it and refresh data
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data and cached_data.get('dialog'):
                # Recompute output directory (override path can change, or cache can be stale)
                epub_base = os.path.splitext(os.path.basename(input_path))[0]
                override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
                if not override_dir and hasattr(self, 'config'):
                    try:
                        override_dir = self.config.get('output_directory')
                    except Exception:
                        override_dir = None
                expected_output_dir = os.path.join(override_dir, epub_base) if override_dir else epub_base
                # On macOS .app bundles, cwd can be '/' (read-only root).
                # Resolve relative output paths against the input file's directory.
                # Only on macOS — on Windows this would change the output dir and break progress tracking.
                if _IS_MACOS and not os.path.isabs(expected_output_dir):
                    expected_output_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), expected_output_dir)

                output_dir = cached_data.get('output_dir')
                progress_file = cached_data.get('progress_file')

                # If cache points at a different location than current override, force a rebuild.
                if output_dir and expected_output_dir and os.path.abspath(output_dir) != os.path.abspath(expected_output_dir):
                    del self._retranslation_dialog_cache[file_key]
                else:
                    # Check if output folder still exists before trying to refresh
                    if not output_dir:
                        output_dir = expected_output_dir
                        cached_data['output_dir'] = output_dir
                        cached_data['progress_file'] = os.path.join(output_dir, "translation_progress.json")
                        progress_file = cached_data['progress_file']

                    if not os.path.exists(output_dir):
                        # Output folder doesn't exist - create it with an empty progress file
                        try:
                            os.makedirs(output_dir, exist_ok=True)
                            empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                            pf = os.path.join(output_dir, "translation_progress.json")
                            with open(pf, 'w', encoding='utf-8') as f:
                                json.dump(empty_prog, f, ensure_ascii=False, indent=2)
                            cached_data['output_dir'] = output_dir
                            cached_data['progress_file'] = pf
                            print(f"📁 Created output folder: {output_dir}")
                            # Flash the PM button green to signal folder creation
                            self._flash_pm_button_green(output_dir)
                        except Exception as e:
                            self._show_message('error', "Error", f"Could not create output folder: {e}")
                            del self._retranslation_dialog_cache[file_key]
                            return
                        del self._retranslation_dialog_cache[file_key]

                    if not progress_file or not os.path.exists(progress_file):
                        # Progress file was deleted - show message and remove from cache,
                        # but DO NOT return. Fall through so we rebuild the dialog and
                        # auto-discover completed chapters in a single click.
                        self._show_message('info', "Info", "No progress tracking found. Existing translations will be auto-discovered.")
                        del self._retranslation_dialog_cache[file_key]
                    else:
                        _persist_progress_manager_source_link(
                            input_path, output_dir)
                        dialog = cached_data['dialog']
                        dialog.show()
                        dialog.raise_()
                        dialog.activateWindow()

                        # Trigger refresh after the dialog is visible so reopening
                        # a large progress file does not block the first paint.
                        def _refresh_cached_single_dialog():
                            _rf = cached_data.get('refresh_func')
                            if callable(_rf):
                                try:
                                    _rf()
                                except Exception:
                                    self._refresh_retranslation_data(cached_data)
                            else:
                                self._refresh_retranslation_data(cached_data)

                        QTimer.singleShot(50, _refresh_cached_single_dialog)
                        return
        
        # For EPUB/text files, use the shared logic
        # Get current toggle state if it exists, or default based on file type
        # Subtitle files and subtitle ZIPs are non-EPUB progress sources.
        show_special_extensions = (
            '.txt', '.pdf', '.csv', '.json', '.srt', '.ass', '.lrc', '.zip'
        )
        show_special = input_path.lower().endswith(show_special_extensions)
        
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data:
                show_special = cached_data.get('show_special_files_state', show_special)
        
        self._show_retranslation_shell_then_build(input_path, show_special_files_state=show_special)


    def _force_retranslation_epub_or_text(
        self,
        file_path,
        parent_dialog=None,
        tab_frame=None,
        show_special_files_state=False,
        _loading_label=None,
        resolved_output_dir=None,
        glossary_progress_source_path=None,
        glossary_progress_source_filenames=None,
    ):
        """
        Shared logic for force retranslation of EPUB/text files with OPF support
        Can be used standalone or embedded in a tab
        
        Args:
            file_path: Path to the EPUB/text file
            parent_dialog: If provided, won't create its own dialog
            tab_frame: If provided, will render into this frame instead of creating dialog
            show_special_files_state: Initial state for showing special files toggle
            _loading_label: Optional QLabel to update with progress messages during loading
            resolved_output_dir: Optional authoritative output directory
            glossary_progress_source_path: Optional paired EPUB path whose
                glossary progress/output files should be displayed.
            glossary_progress_source_filenames: Optional mapped raw filenames
                used to restrict and order the raw EPUB chapter list.
        
        Returns:
            dict: Contains all the UI elements and data for external access
        """
        
        def _pump_loading(msg=None):
            """Update loading label text and spin the icon."""
            try:
                if msg and _loading_label is not None:
                    _loading_label.setText(msg)
                dlg = parent_dialog
                if dlg is not None:
                    advance = getattr(dlg, '_advance_loading_icon', None)
                    if callable(advance):
                        advance()
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents(QEventLoop.AllEvents, 15)
            except Exception:
                pass

        built = self._build_progress_view_data(
            file_path,
            parent_dialog=parent_dialog,
            resolved_output_dir=resolved_output_dir,
            _pump_loading=_pump_loading,
        )
        if built is None:
            return None
        output_dir = built['output_dir']
        progress_file = built['progress_file']
        prog = built['prog']
        progress_source_is_subtitle = built['progress_source_is_subtitle']
        spine_chapters = built['spine_chapters']
        opf_chapter_order = built['opf_chapter_order']
        chapter_display_info = built['chapter_display_info']
        _existing_output_files = built['existing_output_files']

        # State variables for title-row toggles (lists allow nested handlers to mutate them)
        show_special_files = [show_special_files_state]
        show_model_info_state = self._get_retranslation_show_model_info_state(file_path)
        show_model_info = [show_model_info_state]
        
        _pump_loading(f"Preparing UI ({len(chapter_display_info)} chapters)...")
        
        # =====================================================
        # CREATE UI
        # =====================================================
        
        # If no parent dialog or tab frame, create standalone dialog
        if not parent_dialog and not tab_frame:
            # Ensure QApplication exists for standalone PySide6 dialog
            from PySide6.QtWidgets import QApplication
            if not QApplication.instance():
                # Create QApplication if it doesn't exist
                import sys
                QApplication(sys.argv)

            # Create standalone PySide6 dialog.
            # IMPORTANT: If created without a parent, it will NOT inherit the main window's
            # dark stylesheet and will fall back to the OS theme (white on some Win10 setups).
            parent_widget = self if isinstance(self, QWidget) else None
            dialog = QDialog(parent_widget)
            dialog.setWindowTitle("Progress Manager - OPF Based" if spine_chapters else "Progress Manager")
            # Keep above the translator window but allow interaction with it
            # Parent-child windowing keeps this above the translator GUI
            dialog.setWindowModality(Qt.NonModal)
            # Use 42% width, 40% height for 1920x1080
            width, height = self._get_dialog_size(0.42, 0.4)
            dialog.resize(width, height)

            # Inherit/copy the main window stylesheet when available (ensures consistent dark theme).
            try:
                if parent_widget is not None:
                    ss = parent_widget.styleSheet()
                    if ss:
                        dialog.setStyleSheet(ss)
            except Exception:
                pass
            
            # Set icon
            try:
                from PySide6.QtGui import QIcon
                # Try to get base_dir from self (TranslatorGUI), fallback to calculating it
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    dialog.setWindowIcon(QIcon(ico_path))
            except Exception as e:
                print(f"Failed to load icon: {e}")
            dialog_layout = QVBoxLayout(dialog)
            container = QWidget(dialog)
            container_layout = QVBoxLayout(container)
            dialog_layout.addWidget(container)
        else:
            container = tab_frame or parent_dialog
            if not hasattr(container, 'layout') or container.layout() is None:
                container_layout = QVBoxLayout(container)
            else:
                container_layout = container.layout()
            dialog = parent_dialog
        
        # Title and toggle row
        title_row = QWidget()
        title_layout = QHBoxLayout(title_row)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        if spine_chapters:
            title_text = "Chapters from the OPF package (in reading order):"
        elif (
            str(file_path).lower().endswith('.pdf')
            and any(
                (row.get('info') or {}).get('pdf_toc_section')
                for row in chapter_display_info
            )
        ):
            title_text = "PDF bookmark sections:"
        else:
            title_text = "Select chapters to retranslate:"
        title_label = QLabel(title_text)
        title_font = QFont('Arial', 12 if not tab_frame else 11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Add toggle for showing special files
        from PySide6.QtWidgets import QCheckBox
        show_special_files_cb = QCheckBox("Show special files")
        show_special_files_cb.setChecked(show_special_files[0])  # Preserve the current state
        show_special_files_cb.setToolTip(
            "When enabled, shows files skipped by the special-file rules in Other Settings.\n"
            "Active metadata.json, TOC.txt, and translated_headers.txt rows are always visible;\n"
            "disabled rows appear here as skipped."
        )
        
        # Register this checkbox and checkmark with parent dialog for cross-tab syncing
        if parent_dialog and not hasattr(parent_dialog, '_all_toggle_checkboxes'):
            parent_dialog._all_toggle_checkboxes = []
            parent_dialog._all_checkmark_labels = []
            parent_dialog._tab_file_paths = {}  # Map file_path to index
        if parent_dialog:
            # Store the index for this file
            file_key = os.path.abspath(file_path)
            if file_key not in parent_dialog._tab_file_paths:
                parent_dialog._tab_file_paths[file_key] = len(parent_dialog._all_toggle_checkboxes)
                parent_dialog._all_toggle_checkboxes.append(show_special_files_cb)
            else:
                # Replace the old checkbox at this index
                idx = parent_dialog._tab_file_paths[file_key]
                if idx < len(parent_dialog._all_toggle_checkboxes):
                    parent_dialog._all_toggle_checkboxes[idx] = show_special_files_cb
        
        # Apply blue checkbox stylesheet (matching Other Settings dialog)
        show_special_files_cb.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 6px;
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
        
        # Create checkmark overlay for the check symbol
        checkmark = QLabel("✓", show_special_files_cb)
        checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        checkmark.setAlignment(Qt.AlignCenter)
        checkmark.hide()
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        def position_checkmark():
            try:
                if checkmark:
                    checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                pass
        
        def update_checkmark():
            try:
                if show_special_files_cb and checkmark:
                    if show_special_files_cb.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                pass
        
        show_special_files_cb.stateChanged.connect(update_checkmark)
        
        def safe_init():
            try:
                position_checkmark()
                update_checkmark()
            except RuntimeError:
                pass
        
        QTimer.singleShot(0, safe_init)
        
        # Register checkmark for cross-tab syncing
        if parent_dialog:
            file_key = os.path.abspath(file_path)
            if file_key in parent_dialog._tab_file_paths:
                idx = parent_dialog._tab_file_paths[file_key]
                # Append if new, replace if exists
                if idx >= len(parent_dialog._all_checkmark_labels):
                    parent_dialog._all_checkmark_labels.append(checkmark)
                else:
                    parent_dialog._all_checkmark_labels[idx] = checkmark
        
        title_layout.addWidget(show_special_files_cb)

        show_model_info_cb = QCheckBox("Show Model Info")
        show_model_info_cb.setChecked(show_model_info[0])
        show_model_info_cb.setToolTip("When enabled, replaces the output-file column with the model used for each request.")
        show_model_info_cb.setStyleSheet(show_special_files_cb.styleSheet())

        model_checkmark = QLabel("\u2713", show_model_info_cb)
        model_checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        model_checkmark.setAlignment(Qt.AlignCenter)
        model_checkmark.hide()
        model_checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)

        def position_model_checkmark():
            try:
                if model_checkmark:
                    model_checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                pass

        def update_model_checkmark():
            try:
                if show_model_info_cb and model_checkmark:
                    if show_model_info_cb.isChecked():
                        position_model_checkmark()
                        model_checkmark.show()
                    else:
                        model_checkmark.hide()
            except RuntimeError:
                pass

        show_model_info_cb.stateChanged.connect(update_model_checkmark)

        def safe_init_model_checkmark():
            try:
                position_model_checkmark()
                update_model_checkmark()
            except RuntimeError:
                pass

        QTimer.singleShot(0, safe_init_model_checkmark)
        title_layout.addWidget(show_model_info_cb)
        
        # ── Glossary Progress button ──
        # Find the glossary progress file based on automapping settings
        _gp_locator = glossary_progress_locator(
            self, file_path, glossary_progress_source_path
        )
        _glossary_progress_search_dirs = _gp_locator._glossary_progress_search_dirs
        _find_progress_in_dir = _gp_locator._find_progress_in_dir
        _find_glossary_progress_file = _gp_locator._find_glossary_progress_file
        _find_gp_for_file = _gp_locator._find_gp_for_file
        _bool_setting = _gp_locator._bool_setting
        _refinement_type_key = _gp_locator._refinement_type_key
        _active_glossary_refinement_types = _gp_locator._active_glossary_refinement_types
        _glossary_refinement_expected_entries = _gp_locator._glossary_refinement_expected_entries
        _find_glossary_for_refinement = _gp_locator._find_glossary_for_refinement
        
        glossary_progress_btn = QPushButton("📊 Glossary Progress")
        glossary_progress_btn.setCursor(Qt.PointingHandCursor)
        glossary_progress_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d6a4f;
                color: #d8f3dc;
                border: 1px solid #40916c;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #40916c;
                border-color: #52b788;
            }
        """)
        # Always show; the dialog has an empty-state panel until progress exists.
        _initial_glossary_progress_file = _find_glossary_progress_file()
        glossary_progress_btn.setVisible(True)
        if _initial_glossary_progress_file:
            glossary_progress_btn.setToolTip(f"View glossary extraction progress\n{_initial_glossary_progress_file}")
        else:
            glossary_progress_btn.setToolTip("View glossary extraction progress")

        def _sdlxliff_sidecar_paths_for_output_dir(_output_dir):
            sidecar_dir = os.path.join(_output_dir or "", "SDLXLIFF")
            paths = []
            try:
                if os.path.isdir(sidecar_dir):
                    paths = [
                        os.path.join(sidecar_dir, name)
                        for name in os.listdir(sidecar_dir)
                        if str(name).lower().endswith(".sdlxliff")
                    ]
            except Exception:
                paths = []
            return sorted(paths, key=lambda path: os.path.basename(path).lower())

        def _text_analysis_sidecars():
            return _sdlxliff_sidecar_paths_for_output_dir(output_dir)

        text_analysis_btn = QPushButton("🔍 Edit Translation")
        text_analysis_btn.setCursor(Qt.PointingHandCursor)
        text_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b4f6f;
                color: #d7ecff;
                border: 1px solid #5a9fd4;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #356b96;
                border-color: #7bb3e0;
            }
        """)
        text_analysis_btn.setVisible(True)

        manual_editing_cb = QCheckBox("Manual editing")
        manual_editing_cb.setChecked(self._get_retranslation_manual_editing_state())
        manual_editing_cb.setStyleSheet(show_special_files_cb.styleSheet())
        manual_editing_cb.setToolTip(
            "Create source-only SDLXLIFF sidecars for the Progress Manager's Not Translated entries.\n"
            "No HTML output is created until a target row is edited."
        )
        manual_checkmark = QLabel("\u2713", manual_editing_cb)
        manual_checkmark.setAlignment(Qt.AlignCenter)
        manual_checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
        manual_checkmark.setStyleSheet(
            "color: white; background: transparent; font-weight: bold; font-size: 11px;"
        )

        def _update_manual_checkmark():
            try:
                manual_checkmark.setGeometry(2, 1, 14, 14)
                manual_checkmark.setVisible(manual_editing_cb.isChecked())
            except RuntimeError:
                pass

        manual_editing_cb.stateChanged.connect(_update_manual_checkmark)
        QTimer.singleShot(0, _update_manual_checkmark)

        def _progress_manager_untranslated_entries():
            progress_data = getattr(manual_editing_cb, '_progress_data_ref', None)
            current_spine_chapters = (
                progress_data.get('spine_chapters', spine_chapters)
                if isinstance(progress_data, dict)
                else spine_chapters
            )
            status_data = progress_data if isinstance(progress_data, dict) else {'prog': prog}
            current_output_dir = (
                progress_data.get('output_dir', output_dir)
                if isinstance(progress_data, dict)
                else output_dir
            )
            untranslated_entries = []
            for entry in current_spine_chapters:
                if not isinstance(entry, dict):
                    continue
                if self._progress_display_status(entry, status_data) not in {
                    'not_translated', 'pending',
                }:
                    continue
                manual_entry = dict(entry)
                manual_entry['status'] = 'not_translated'
                untranslated_entries.append(manual_entry)
            return untranslated_entries

        manual_generation_state = {
            'running': False,
            'callbacks': [],
            'signature': None,
            'last_stats': None,
            'button_text': text_analysis_btn.text(),
            'persist_token': 0,
        }

        def _manual_generation_signature(entries, current_output_dir):
            output_names = tuple(sorted(
                str(entry.get('output_file') or '').replace('\\', '/').lower()
                for entry in (entries or [])
                if isinstance(entry, dict) and entry.get('output_file')
            ))
            return (
                os.path.normcase(os.path.abspath(current_output_dir or '')),
                len(output_names),
                hash(output_names),
            )

        def _show_generation_progress(payload):
            try:
                total = int(payload.get('total') or 0)
                index = int(payload.get('index') or 0)
                stage = str(payload.get('stage') or '')
                if stage in ('checking', 'created', 'skipped', 'missing_source', 'failed') and total:
                    text_analysis_btn.setText(f"Creating sidecars… {min(index, total)}/{total}")
            except (RuntimeError, TypeError, ValueError):
                pass

        def _finish_manual_generation(stats):
            stats = stats if isinstance(stats, dict) else {}
            manual_generation_state['running'] = False
            manual_generation_state['last_stats'] = stats
            if not stats.get('errors') and not stats.get('failed') and not stats.get('missing_source'):
                manual_generation_state['signature'] = stats.pop('_generation_signature', None)
            else:
                stats.pop('_generation_signature', None)
                manual_generation_state['signature'] = None
            try:
                text_analysis_btn.setText(manual_generation_state['button_text'])
            except RuntimeError:
                pass

            created = int(stats.get('created') or 0)
            skipped_sidecars = int(stats.get('skipped') or 0)
            missing_source = int(stats.get('missing_source') or 0)
            errors = list(stats.get('errors') or [])
            try:
                if errors and not created:
                    manual_editing_cb.setToolTip(f"Manual sidecar creation failed: {errors[0]}")
                elif created:
                    manual_editing_cb.setToolTip(
                        f"Created {created} source-only manual sidecar{'s' if created != 1 else ''}.\n"
                        "No HTML output is created until a target row is edited."
                    )
                elif skipped_sidecars:
                    manual_editing_cb.setToolTip(
                        "Manual sidecars already exist. No HTML output is created until a target row is edited."
                    )
                elif missing_source:
                    manual_editing_cb.setToolTip(
                        f"Could not locate source HTML for {missing_source} Not Translated entr{'ies' if missing_source != 1 else 'y'}."
                    )
            except RuntimeError:
                pass
            _queue_text_analysis_button_update()

            callbacks = list(manual_generation_state['callbacks'])
            manual_generation_state['callbacks'].clear()
            for callback in callbacks:
                try:
                    callback(stats)
                except Exception:
                    pass

        def _fail_manual_generation(message):
            stats = {
                'created': 0,
                'paths': [],
                'missing_source': 0,
                'failed': 1,
                'errors': [str(message or 'Unknown sidecar generation error')],
            }
            _finish_manual_generation(stats)

        manual_generation_bridge = _GlossaryProgressAsyncBridge(
            on_finished=_finish_manual_generation,
            on_failed=_fail_manual_generation,
            on_progress=_show_generation_progress,
            parent=dialog,
        )
        manual_editing_cb._manual_sidecar_generation_bridge = manual_generation_bridge

        def _generate_manual_editing_sidecars(on_finished=None):
            if callable(on_finished):
                manual_generation_state['callbacks'].append(on_finished)
            if manual_generation_state['running']:
                return

            untranslated_entries = _progress_manager_untranslated_entries()
            progress_data = getattr(manual_editing_cb, '_progress_data_ref', None)
            current_output_dir = (
                progress_data.get('output_dir', output_dir)
                if isinstance(progress_data, dict)
                else output_dir
            )
            current_file_path = (
                progress_data.get('file_path', file_path)
                if isinstance(progress_data, dict)
                else file_path
            )
            signature = _manual_generation_signature(
                untranslated_entries,
                current_output_dir,
            )
            if manual_generation_state['signature'] == signature:
                callbacks = list(manual_generation_state['callbacks'])
                manual_generation_state['callbacks'].clear()
                cached_stats = dict(manual_generation_state['last_stats'] or {})
                for callback in callbacks:
                    QTimer.singleShot(0, lambda cb=callback, result=cached_stats: cb(result))
                return

            manual_generation_state['running'] = True
            try:
                text_analysis_btn.setText("Creating sidecars…")
            except RuntimeError:
                pass

            def _worker():
                try:
                    last_progress_emit = [0.0]

                    def _emit_progress(payload):
                        now = time.monotonic()
                        stage = str((payload or {}).get('stage') or '')
                        if (
                            stage not in {'start', 'finished'}
                            and now - last_progress_emit[0] < 0.1
                        ):
                            return
                        last_progress_emit[0] = now
                        manual_generation_bridge.progress.emit(payload)

                    stats = self._generate_sdlxliff_sidecars_from_untranslated_entries(
                        current_output_dir,
                        untranslated_entries,
                        file_path=current_file_path,
                        progress_callback=_emit_progress,
                    )
                    stats = stats if isinstance(stats, dict) else {}
                    stats['_generation_signature'] = signature
                    manual_generation_bridge.finished.emit(stats)
                except Exception as exc:
                    try:
                        manual_generation_bridge.failed.emit(str(exc))
                    except RuntimeError:
                        pass

            threading.Thread(
                target=_worker,
                name="manual-sdlxliff-sidecar-generation",
                daemon=True,
            ).start()

        def _on_manual_editing_toggled(enabled):
            enabled = bool(enabled)
            try:
                if not hasattr(self, 'config') or not isinstance(self.config, dict):
                    self.config = {}
                self.config[self._RETRANSLATION_MANUAL_EDITING_CONFIG_KEY] = enabled
            except Exception:
                pass
            manual_generation_state['persist_token'] += 1
            persist_token = manual_generation_state['persist_token']

            def _persist_after_repaint():
                if persist_token != manual_generation_state['persist_token']:
                    return
                self._persist_retranslation_manual_editing_state(enabled)

            QTimer.singleShot(25, _persist_after_repaint)
            progress_data = getattr(manual_editing_cb, '_progress_data_ref', None)
            if isinstance(progress_data, dict):
                progress_data['manual_editing_state'] = enabled
            if enabled:
                # Return to Qt first so the checkmark paints immediately. The
                # expensive EPUB scan and sidecar writes run on the worker.
                QTimer.singleShot(
                    0,
                    lambda: (
                        _generate_manual_editing_sidecars()
                        if manual_editing_cb.isChecked()
                        else None
                    ),
                )

        def _update_text_analysis_button():
            try:
                sidecars = _text_analysis_sidecars()
                text_analysis_btn.setVisible(True)
                text_analysis_btn.setEnabled(True)
                if sidecars:
                    if len(sidecars) == 1:
                        text_analysis_btn.setToolTip(f"Review source/output text analysis\n{sidecars[0]}")
                    else:
                        text_analysis_btn.setToolTip(
                            f"Review source/output text analysis ({len(sidecars)} SDLXLIFF sidecars)"
                        )
                else:
                    text_analysis_btn.setToolTip(
                        "Review source/output text analysis. SDLXLIFF sidecars will be generated from completed entries if needed."
                    )
            except RuntimeError:
                pass
            except Exception:
                text_analysis_btn.setVisible(True)
                text_analysis_btn.setEnabled(True)

        def _show_text_analysis():
            def _open_after_manual_generation(_stats=None):
                try:
                    progress_data = getattr(manual_editing_cb, '_progress_data_ref', None)
                    current_output_dir = (
                        progress_data.get('output_dir', output_dir)
                        if isinstance(progress_data, dict)
                        else output_dir
                    )
                    current_progress = (
                        progress_data.get('prog', prog)
                        if isinstance(progress_data, dict)
                        else prog
                    )
                    review_dialog = self._open_or_reuse_sdlxliff_review(
                        current_output_dir,
                        None,
                        dialog,
                        autogen_file_path=file_path,
                        autogen_progress_data=current_progress,
                        autogen_manual_entries=_progress_manager_untranslated_entries(),
                    )
                    if review_dialog is None:
                        return
                    try:
                        review_dialog.save_status_label.setText("Checking SDLXLIFF sidecars...")
                    except Exception:
                        pass
                    _queue_text_analysis_button_update()
                except Exception as e:
                    self._show_message('error', "Open Failed", str(e), parent=dialog)

            if manual_editing_cb.isChecked():
                _generate_manual_editing_sidecars(
                    on_finished=_open_after_manual_generation,
                )
                return
            _open_after_manual_generation()

        text_analysis_btn.clicked.connect(_show_text_analysis)
        manual_editing_cb.toggled.connect(_on_manual_editing_toggled)
        _update_text_analysis_button()

        text_analysis_update_queued = [False]

        def _queue_text_analysis_button_update():
            if text_analysis_update_queued[0]:
                return
            text_analysis_update_queued[0] = True

            def _apply_queued_update():
                text_analysis_update_queued[0] = False
                _update_text_analysis_button()

            QTimer.singleShot(0, _apply_queued_update)
        try:
            self._refresh_progress_text_analysis_button = _queue_text_analysis_button_update
        except Exception:
            pass

        try:
            if hasattr(self, "profile_menu") and self.profile_menu is not None:
                self.profile_menu.currentTextChanged.connect(lambda *_: _queue_text_analysis_button_update())
                self.profile_menu.currentIndexChanged.connect(lambda *_: _queue_text_analysis_button_update())
                self.profile_menu.activated.connect(lambda *_: _queue_text_analysis_button_update())
        except Exception:
            pass
        for _radio_name in ("standard_extraction_radio", "enhanced_extraction_radio"):
            try:
                _radio = getattr(self, _radio_name, None)
                if _radio is not None:
                    _radio.toggled.connect(lambda *_: _queue_text_analysis_button_update())
            except Exception:
                pass


        def _confirm_manual_glossary_refinement(
            parent,
            source_path,
            progress_path,
            selected_types,
            completed_types=None,
        ):
            parent = _resolve_dialog_window_parent(parent)
            if hasattr(self, '_is_any_process_running') and self._is_any_process_running():
                self._show_message(
                    'warning',
                    'Process Running',
                    'Please wait for the current translation, glossary, or refinement process to finish.',
                    parent=parent,
                )
                return False
            require_model = getattr(self, '_require_model_selection', None)
            if callable(require_model):
                model = require_model('refining glossary entries', parent=parent)
            else:
                model = str(getattr(self, 'model_var', '') or '').strip()
                if not model:
                    self._show_message(
                        'warning',
                        'No Model Selected',
                        'Select or enter a model before refining glossary entries.',
                        parent=parent,
                    )
            if not model:
                return False
            # Glossary lookup, entry types and the automatic plan are shared
            # (glossary_progress_core.prepare_manual_glossary_refinement).
            preview_state = prepare_manual_glossary_refinement(
                self,
                source_path,
                progress_path,
                selected_types,
                model,
                find_glossary=_find_glossary_for_refinement,
                active_types_fn=_active_glossary_refinement_types,
            )
            if preview_state.refusal is not None:
                refusal_kind, refusal_title, refusal_message = preview_state.refusal
                self._show_message(refusal_kind, refusal_title, refusal_message, parent=parent)
                return False
            glossary_path = preview_state.glossary_path
            entries = preview_state.entries
            active_types = preview_state.active_types
            selected_types = preview_state.selected_types
            entry_counts = preview_state.entry_counts
            non_empty_types = preview_state.non_empty_types
            request_mode = preview_state.request_mode
            splitter = preview_state.splitter
            safe_budget = preview_state.safe_budget
            system_prompt = preview_state.system_prompt
            user_prompt = preview_state.user_prompt
            refinement_type_config = preview_state.refinement_type_config
            automatic_plan = preview_state.automatic_plan
            from glossary_refinement import plan_refinement

            from PySide6.QtWidgets import QAbstractSpinBox, QStyle, QStyleOptionButton, QToolButton

            selection_state = {
                'types': list(selected_types),
                'plan': automatic_plan,
            }
            preview = QDialog(parent)
            preview.setObjectName('glossaryRefinementPreview')
            preview.setWindowTitle('Confirm Glossary Refinement')
            preview.setWindowModality(Qt.WindowModal)

            # Keep the dialog proportional across resolutions and DPI settings.
            screen = parent.screen() if parent is not None and hasattr(parent, 'screen') else None
            screen = screen or QApplication.primaryScreen()
            if screen is not None:
                available_geometry = screen.availableGeometry()
                minimum_width = max(1, round(available_geometry.width() * 0.20))
                minimum_height = max(1, round(available_geometry.height() * 0.26))
                dialog_width = max(
                    minimum_width,
                    round(available_geometry.width() * 0.26),
                )
                dialog_height = max(
                    minimum_height,
                    round(available_geometry.height() * 0.42),
                )
                preview.setMinimumSize(minimum_width, minimum_height)
                preview.resize(dialog_width, dialog_height)
                preview_frame = preview.frameGeometry()
                if parent is not None:
                    preview_frame.moveCenter(parent.window().frameGeometry().center())
                else:
                    preview_frame.moveCenter(available_geometry.center())
                target_x = min(
                    max(preview_frame.left(), available_geometry.left()),
                    available_geometry.right() - preview_frame.width() + 1,
                )
                target_y = min(
                    max(preview_frame.top(), available_geometry.top()),
                    available_geometry.bottom() - preview_frame.height() + 1,
                )
                preview.move(target_x, target_y)

            preview.setStyleSheet("""
                QDialog#glossaryRefinementPreview {
                    background-color: #151a21;
                    color: #e7edf5;
                }
                QWidget#glossaryRefinementBody {
                    background-color: transparent;
                }
                QScrollArea#glossaryRefinementScroll {
                    background-color: transparent;
                    border: none;
                }
                QScrollArea#glossaryRefinementScroll > QWidget > QWidget {
                    background-color: transparent;
                }
                QFrame#refinementHero {
                    background-color: #1c2632;
                    border: 1px solid #33475d;
                    border-radius: 10px;
                }
                QLabel#refinementHeroIcon {
                    background-color: #263c4b;
                    border: 1px solid #3f6574;
                    border-radius: 8px;
                    padding: 6px;
                }
                QLabel#refinementHeroTitle {
                    color: #f3f8fb;
                    font-size: 12pt;
                    font-weight: 700;
                }
                QLabel#refinementHeroSubtitle,
                QLabel#refinementSectionHint,
                QLabel#refinementMetricHint,
                QLabel#refinementOverrideHint,
                QLabel#refinementNote {
                    color: #91a0b3;
                }
                QLabel#refinementSectionTitle {
                    color: #dce8f2;
                    font-size: 10pt;
                    font-weight: 700;
                }
                QFrame#refinementScopeCard,
                QFrame#refinementOverrideCard {
                    background-color: #1a2029;
                    border: 1px solid #303b49;
                    border-radius: 9px;
                }
                QFrame#refinementScopeRow {
                    background-color: #202936;
                    border: 1px solid #334052;
                    border-radius: 7px;
                }
                QLabel#refinementTypeName {
                    color: #e8eef5;
                    font-weight: 650;
                }
                QLabel#refinementTypeStats {
                    color: #9fb2c8;
                }
                QLabel#refinementTypeStatsEmpty {
                    color: #737f8e;
                }
                QToolButton#refinementTypePicker {
                    color: #e8eef5;
                    background-color: #202936;
                    border: 1px solid #415168;
                    border-radius: 7px;
                    padding: 7px 34px 7px 10px;
                    text-align: left;
                    font-weight: 650;
                }
                QToolButton#refinementTypePicker:hover,
                QToolButton#refinementTypePicker:pressed {
                    background-color: #293646;
                    border-color: #60758a;
                }
                QToolButton#refinementTypePicker::menu-indicator {
                    subcontrol-origin: padding;
                    subcontrol-position: center right;
                    right: 8px;
                }
                QFrame#refinementMetricCard {
                    background-color: #202b38;
                    border: 1px solid #34465a;
                    border-radius: 8px;
                }
                QLabel#refinementMetricLabel {
                    color: #8ea2b8;
                    font-size: 8pt;
                    font-weight: 700;
                }
                QLabel#refinementMetricValue {
                    color: #f0f6fb;
                    font-size: 13pt;
                    font-weight: 700;
                }
                QFrame#refinementWarningCard {
                    background-color: #302918;
                    border: 1px solid #755d20;
                    border-radius: 8px;
                }
                QLabel#refinementWarningText {
                    color: #f2c75c;
                    font-weight: 600;
                }
                QCheckBox#refinementOverrideCheckbox {
                    color: #edf3f8;
                    background-color: transparent;
                    border: none;
                    font-weight: 650;
                    spacing: 8px;
                }
                QCheckBox#refinementOverrideCheckbox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 1px solid #5f7388;
                    border-radius: 4px;
                    background-color: #12171d;
                }
                QCheckBox#refinementOverrideCheckbox::indicator:checked {
                    background-color: #52b788;
                    border-color: #74c69d;
                }
                QLabel#refinementOverrideCheckmark {
                    color: #07150f;
                    background-color: transparent;
                    font-size: 12pt;
                    font-weight: 900;
                }
                QSpinBox#refinementChunkCount {
                    color: #edf3f8;
                    background-color: #11171e;
                    border: 1px solid #435267;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
                QSpinBox#refinementChunkCount:disabled {
                    color: #687687;
                    background-color: #171d25;
                    border-color: #2b3440;
                }
                QToolButton#refinementStepButton {
                    color: #dce8f2;
                    background-color: #27313d;
                    border: 1px solid #465668;
                    border-radius: 6px;
                    padding: 5px 10px;
                    font-size: 12pt;
                    font-weight: 700;
                }
                QToolButton#refinementStepButton:hover {
                    color: #ffffff;
                    background-color: #344252;
                    border-color: #60758a;
                }
                QToolButton#refinementStepButton:pressed {
                    background-color: #1d2630;
                }
                QToolButton#refinementStepButton:disabled {
                    color: #596574;
                    background-color: #1b222b;
                    border-color: #2b3541;
                }
                QLabel#refinementOneRunBadge {
                    color: #76cfa4;
                    background-color: #1d3b32;
                    border: 1px solid #315e4e;
                    border-radius: 7px;
                    padding: 3px 7px;
                    font-size: 8pt;
                    font-weight: 700;
                }
                QPushButton#refinementPrimaryButton {
                    color: #07150f;
                    background-color: #74c69d;
                    border: 1px solid #8bd3ae;
                    border-radius: 7px;
                    padding: 10px 26px;
                    font-weight: 700;
                }
                QPushButton#refinementPrimaryButton:hover {
                    background-color: #8bd3ae;
                }
                QPushButton#refinementPrimaryButton:pressed {
                    background-color: #52b788;
                }
                QPushButton#refinementCancelButton {
                    color: #dce5ee;
                    background-color: #252d38;
                    border: 1px solid #465363;
                    border-radius: 7px;
                    padding: 10px 26px;
                }
                QPushButton#refinementCancelButton:hover {
                    background-color: #313b48;
                    border-color: #5b6a7d;
                }
                QScrollBar:vertical {
                    background: #151a21;
                    width: 9px;
                    margin: 2px;
                }
                QScrollBar::handle:vertical {
                    background: #445267;
                    border-radius: 4px;
                    min-height: 28px;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)

            preview_layout = QVBoxLayout(preview)
            preview_layout.setContentsMargins(10, 10, 10, 9)
            preview_layout.setSpacing(6)

            hero = QFrame(preview)
            hero.setObjectName('refinementHero')
            hero_layout = QHBoxLayout(hero)
            hero_layout.setContentsMargins(10, 7, 10, 7)
            hero_layout.setSpacing(8)
            hero_icon = QLabel('✨')
            hero_icon.setObjectName('refinementHeroIcon')
            hero_icon.setAlignment(Qt.AlignCenter)
            hero_layout.addWidget(hero_icon, 0, Qt.AlignTop)
            hero_copy = QVBoxLayout()
            hero_copy.setSpacing(3)
            hero_title = QLabel('Ready to refine glossary entries')
            hero_title.setObjectName('refinementHeroTitle')
            hero_subtitle = QLabel('Confirm the scope and request plan.')
            hero_subtitle.setObjectName('refinementHeroSubtitle')
            hero_subtitle.setWordWrap(True)
            hero_copy.addWidget(hero_title)
            hero_copy.addWidget(hero_subtitle)
            hero_layout.addLayout(hero_copy, 1)
            preview_layout.addWidget(hero)

            content_scroll = QScrollArea(preview)
            content_scroll.setObjectName('glossaryRefinementScroll')
            content_scroll.setWidgetResizable(True)
            content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content_scroll.setFrameShape(QFrame.NoFrame)
            content = QWidget(content_scroll)
            content.setObjectName('glossaryRefinementBody')
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 4, 0)
            content_layout.setSpacing(6)

            scope_card = QFrame(content)
            scope_card.setObjectName('refinementScopeCard')
            scope_layout = QVBoxLayout(scope_card)
            scope_layout.setContentsMargins(9, 7, 9, 8)
            scope_layout.setSpacing(5)
            scope_title = QLabel('SELECTED ENTRY TYPES')
            scope_title.setObjectName('refinementSectionTitle')
            scope_hint = QLabel('Choose one or more active types. Empty types are skipped.')
            scope_hint.setObjectName('refinementSectionHint')
            scope_hint.setWordWrap(True)
            scope_layout.addWidget(scope_title)
            scope_layout.addWidget(scope_hint)

            type_picker = QToolButton(scope_card)
            type_picker.setObjectName('refinementTypePicker')
            type_picker.setPopupMode(QToolButton.InstantPopup)
            type_picker.setToolButtonStyle(Qt.ToolButtonTextOnly)
            type_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            type_selection_menu = QMenu(type_picker)
            type_selection_menu.setObjectName('refinementTypeSelectionMenu')
            type_selection_menu.setStyleSheet("""
                QMenu {
                    color: #e7edf5;
                    background-color: #1b222c;
                    border: 1px solid #465568;
                    border-radius: 6px;
                    padding: 5px;
                }
                QCheckBox {
                    color: #e7edf5;
                    spacing: 8px;
                    padding: 6px 10px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 1px solid #5f7388;
                    border-radius: 4px;
                    background-color: #12171d;
                }
                QCheckBox::indicator:checked {
                    background-color: #52a8dc;
                    border-color: #6fbae7;
                }
                QCheckBox::indicator:indeterminate {
                    background-color: #3d789b;
                    border-color: #5da5d0;
                }
                QCheckBox#refinementSelectAllCheckbox {
                    color: #8bd3ae;
                    font-weight: 700;
                }
                QLabel#refinementTypeChoiceCheckmark {
                    color: #07131b;
                    background-color: transparent;
                    font-size: 12pt;
                    font-weight: 900;
                }
                QCheckBox:hover {
                    background-color: #293646;
                    border-radius: 4px;
                }
            """)
            type_choice_widgets = {}
            type_choice_checkmark_syncs = []

            def _attach_styled_checkmark(checkbox, object_name):
                checkmark = QLabel('✓', checkbox)
                checkmark.setObjectName(object_name)
                checkmark.setAlignment(Qt.AlignCenter)
                checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
                checkmark.hide()

                def _sync_checkmark(*_args):
                    try:
                        style_option = QStyleOptionButton()
                        checkbox.initStyleOption(style_option)
                        indicator_rect = checkbox.style().subElementRect(
                            QStyle.SE_CheckBoxIndicator,
                            style_option,
                            checkbox,
                        )
                        check_state = checkbox.checkState()
                        checkmark.setText('−' if check_state == Qt.PartiallyChecked else '✓')
                        checkmark.setGeometry(indicator_rect)
                        checkmark.setVisible(check_state != Qt.Unchecked)
                    except RuntimeError:
                        pass

                checkbox.toggled.connect(_sync_checkmark)
                QTimer.singleShot(0, _sync_checkmark)
                return _sync_checkmark

            total_active_entry_count = sum(
                entry_counts.get(_refinement_type_key(entry_type), 0)
                for entry_type in active_types
            )
            select_all_types_checkbox = QCheckBox(
                f'Select all — {total_active_entry_count:,} entries',
                type_selection_menu,
            )
            select_all_types_checkbox.setObjectName('refinementSelectAllCheckbox')
            select_all_types_checkbox.setTristate(True)
            select_all_widget_action = QWidgetAction(type_selection_menu)
            select_all_widget_action.setDefaultWidget(select_all_types_checkbox)
            type_selection_menu.addAction(select_all_widget_action)
            type_choice_checkmark_syncs.append(
                _attach_styled_checkmark(
                    select_all_types_checkbox,
                    'refinementTypeChoiceCheckmark',
                )
            )
            type_selection_menu.addSeparator()

            selected_type_keys = {_refinement_type_key(name) for name in selected_types}
            for entry_type in active_types:
                entry_count = entry_counts.get(_refinement_type_key(entry_type), 0)
                choice_suffix = f'{entry_count:,} entries' if entry_count else 'no entries'
                type_choice = QCheckBox(f'{entry_type}   —   {choice_suffix}', type_selection_menu)
                type_choice.setChecked(_refinement_type_key(entry_type) in selected_type_keys)
                choice_action = QWidgetAction(type_selection_menu)
                choice_action.setDefaultWidget(type_choice)
                type_selection_menu.addAction(choice_action)
                type_choice_widgets[entry_type] = type_choice
                type_choice_checkmark_syncs.append(
                    _attach_styled_checkmark(type_choice, 'refinementTypeChoiceCheckmark')
                )

            def _sync_select_all_checkbox():
                checked_count = sum(
                    type_choice.isChecked()
                    for type_choice in type_choice_widgets.values()
                )
                if checked_count == 0:
                    master_state = Qt.Unchecked
                elif checked_count == len(type_choice_widgets):
                    master_state = Qt.Checked
                else:
                    master_state = Qt.PartiallyChecked
                select_all_types_checkbox.blockSignals(True)
                select_all_types_checkbox.setCheckState(master_state)
                select_all_types_checkbox.blockSignals(False)
                type_choice_checkmark_syncs[0]()

            def _sync_type_choice_checkmarks():
                _sync_select_all_checkbox()
                for sync_checkmark in type_choice_checkmark_syncs:
                    sync_checkmark()

            type_selection_menu.aboutToShow.connect(
                lambda: QTimer.singleShot(0, _sync_type_choice_checkmarks)
            )
            type_picker.setMenu(type_selection_menu)
            scope_layout.addWidget(type_picker)
            selection_stats = QLabel()
            selection_stats.setObjectName('refinementTypeStats')
            selection_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
            scope_layout.addWidget(selection_stats)
            content_layout.addWidget(scope_card)

            mode_name = 'Send all selected types together' if request_mode == 'all' else 'Send each type separately'
            metric_grid = QGridLayout()
            metric_grid.setContentsMargins(0, 0, 0, 0)
            metric_grid.setHorizontalSpacing(8)
            metric_grid.setVerticalSpacing(8)
            metric_grid.setColumnStretch(0, 1)
            metric_grid.setColumnStretch(1, 1)
            metric_values = (
                ('payload_tokens', 'PAYLOAD TOKENS', f'{automatic_plan.total_payload_tokens:,}', 'Tokenizer estimate'),
                ('automatic_chunks', 'AUTO CHUNKS', f'{automatic_plan.total_chunks:,}', 'Using the saved split logic'),
                ('safe_budget', 'SAFE BUDGET', f'{safe_budget:,}', 'Input tokens per chunk'),
                ('request_mode', 'REQUEST MODE', request_mode.upper(), mode_name),
            )
            metric_value_widgets = {}
            for metric_index, (metric_key, metric_label_text, metric_value_text, metric_tooltip) in enumerate(metric_values):
                metric_card = QFrame(content)
                metric_card.setObjectName('refinementMetricCard')
                metric_card.setToolTip(metric_tooltip)
                metric_layout = QVBoxLayout(metric_card)
                metric_layout.setContentsMargins(9, 6, 9, 7)
                metric_layout.setSpacing(1)
                metric_label = QLabel(metric_label_text)
                metric_label.setObjectName('refinementMetricLabel')
                metric_value = QLabel(metric_value_text)
                metric_value.setObjectName('refinementMetricValue')
                metric_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                metric_layout.addWidget(metric_label)
                metric_layout.addWidget(metric_value)
                metric_value_widgets[metric_key] = metric_value
                metric_grid.addWidget(metric_card, metric_index // 2, metric_index % 2)
            content_layout.addLayout(metric_grid)

            completed_lc = {_refinement_type_key(name) for name in completed_types or []}
            warning_card = QFrame(content)
            warning_card.setObjectName('refinementWarningCard')
            warning_layout = QHBoxLayout(warning_card)
            warning_layout.setContentsMargins(8, 6, 8, 6)
            warning = QLabel()
            warning.setObjectName('refinementWarningText')
            warning.setWordWrap(True)
            warning_layout.addWidget(warning)
            warning_card.hide()
            content_layout.addWidget(warning_card)

            override_card = QFrame(content)
            override_card.setObjectName('refinementOverrideCard')
            override_layout = QVBoxLayout(override_card)
            override_layout.setContentsMargins(9, 7, 9, 8)
            override_layout.setSpacing(4)
            override_header = QHBoxLayout()
            override_checkbox = QCheckBox('Override automatic split', override_card)
            override_checkbox.setObjectName('refinementOverrideCheckbox')
            override_checkbox.setChecked(False)

            override_checkmark = QLabel('✓', override_checkbox)
            override_checkmark.setObjectName('refinementOverrideCheckmark')
            override_checkmark.setAlignment(Qt.AlignCenter)
            override_checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
            override_checkmark.hide()

            def _sync_override_checkmark(*_args):
                try:
                    style_option = QStyleOptionButton()
                    override_checkbox.initStyleOption(style_option)
                    indicator_rect = override_checkbox.style().subElementRect(
                        QStyle.SE_CheckBoxIndicator,
                        style_option,
                        override_checkbox,
                    )
                    override_checkmark.setGeometry(indicator_rect)
                    override_checkmark.setVisible(override_checkbox.isChecked())
                except RuntimeError:
                    pass

            override_checkbox.toggled.connect(_sync_override_checkmark)
            QTimer.singleShot(0, _sync_override_checkmark)
            override_header.addWidget(override_checkbox)
            override_header.addStretch()
            one_run_badge = QLabel('ONE RUN ONLY')
            one_run_badge.setObjectName('refinementOneRunBadge')
            override_header.addWidget(one_run_badge)
            override_layout.addLayout(override_header)
            override_hint = QLabel(
                'Off uses automatic splitting. Turn on to choose an exact chunk count.'
            )
            override_hint.setObjectName('refinementOverrideHint')
            override_hint.setWordWrap(True)
            override_layout.addWidget(override_hint)
            chunk_row = QHBoxLayout()
            chunk_row.addWidget(QLabel('Exact total chunk count:'))
            chunk_count = QSpinBox(override_card)
            chunk_count.setObjectName('refinementChunkCount')
            chunk_count.setButtonSymbols(QAbstractSpinBox.NoButtons)
            chunk_count.setAlignment(Qt.AlignCenter)
            minimum_chunks = 1 if request_mode == 'all' else len(non_empty_types)
            maximum_chunks = sum(automatic_plan.per_type_counts.get(name, 0) for name in non_empty_types)
            chunk_count.setRange(max(1, minimum_chunks), max(1, maximum_chunks))
            chunk_count.setValue(max(chunk_count.minimum(), min(automatic_plan.total_chunks, chunk_count.maximum())))
            chunk_count.setToolTip('This one-run override never changes the saved Refinement settings.')
            chunk_row.addWidget(chunk_count)

            decrement_chunk = QToolButton(override_card)
            decrement_chunk.setObjectName('refinementStepButton')
            decrement_chunk.setText('−')
            decrement_chunk.setToolTip('Use one fewer chunk')
            decrement_chunk.setAutoRepeat(True)
            decrement_chunk.clicked.connect(chunk_count.stepDown)
            chunk_row.addWidget(decrement_chunk)

            increment_chunk = QToolButton(override_card)
            increment_chunk.setObjectName('refinementStepButton')
            increment_chunk.setText('+')
            increment_chunk.setToolTip('Use one more chunk')
            increment_chunk.setAutoRepeat(True)
            increment_chunk.clicked.connect(chunk_count.stepUp)
            chunk_row.addWidget(increment_chunk)

            def _set_manual_chunk_controls_enabled(enabled):
                chunk_count.setEnabled(enabled)
                decrement_chunk.setEnabled(enabled and chunk_count.value() > chunk_count.minimum())
                increment_chunk.setEnabled(enabled and chunk_count.value() < chunk_count.maximum())

            override_checkbox.toggled.connect(_set_manual_chunk_controls_enabled)
            chunk_count.valueChanged.connect(
                lambda _value: _set_manual_chunk_controls_enabled(override_checkbox.isChecked())
            )
            _set_manual_chunk_controls_enabled(False)
            chunk_row.addStretch()
            override_layout.addLayout(chunk_row)
            chunk_range = QLabel(f'Allowed range: {chunk_count.minimum():,}–{chunk_count.maximum():,}')
            chunk_range.setObjectName('refinementSectionHint')
            override_layout.addWidget(chunk_range)
            content_layout.addWidget(override_card)

            note = QLabel('ℹ️ Chunk boundaries always fall between complete glossary entries; CSV rows are never divided.')
            note.setObjectName('refinementNote')
            note.setWordWrap(True)
            content_layout.addWidget(note)
            content_layout.addStretch()
            content_scroll.setWidget(content)
            preview_layout.addWidget(content_scroll, 1)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=preview)
            buttons.setCenterButtons(True)
            buttons.layout().setSpacing(18)
            refine_button = buttons.button(QDialogButtonBox.Ok)
            refine_button.setObjectName('refinementPrimaryButton')
            refine_button.setText('✨  Refine Now')
            refine_button.setDefault(True)
            cancel_button = buttons.button(QDialogButtonBox.Cancel)
            cancel_button.setObjectName('refinementCancelButton')

            def _selected_preview_types():
                return [
                    entry_type for entry_type in active_types
                    if type_choice_widgets[entry_type].isChecked()
                ]

            def _refresh_refinement_preview(*_args):
                current_types = _selected_preview_types()
                try:
                    current_plan = plan_refinement(
                        entries,
                        selected_types=current_types,
                        chunking_mode=request_mode,
                        chapter_splitter=splitter,
                        available_tokens=safe_budget,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        custom_entry_types=refinement_type_config,
                    )
                except Exception as exc:
                    selection_stats.setText(f'Unable to calculate refinement plan: {exc}')
                    refine_button.setEnabled(False)
                    return

                selection_state['types'] = list(current_types)
                selection_state['plan'] = current_plan
                non_empty_current = [
                    entry_type for entry_type in current_types
                    if current_plan.per_type_counts.get(entry_type, 0) > 0
                ]
                total_entries = sum(
                    current_plan.per_type_counts.get(entry_type, 0)
                    for entry_type in current_types
                )

                if not current_types:
                    type_picker.setText('Choose entry types…')
                    selection_stats.setText('No entry types selected.')
                else:
                    if len(current_types) <= 2:
                        picker_summary = ', '.join(current_types)
                    else:
                        picker_summary = f'{current_types[0]} + {len(current_types) - 1} more'
                    type_picker.setText(f'{picker_summary}  ({len(current_types)} selected)')
                    empty_count = len(current_types) - len(non_empty_current)
                    empty_suffix = f'  •  {empty_count} empty' if empty_count else ''
                    selection_stats.setText(
                        f'{total_entries:,} entries  •  '
                        f'{current_plan.total_payload_tokens:,} tokens{empty_suffix}'
                    )

                metric_value_widgets['payload_tokens'].setText(
                    f'{current_plan.total_payload_tokens:,}'
                )
                metric_value_widgets['automatic_chunks'].setText(
                    f'{current_plan.total_chunks:,}'
                )

                already_refined = [
                    name for name in current_types
                    if _refinement_type_key(name) in completed_lc
                ]
                if already_refined:
                    warning.setText(
                        '⚠️ Already refined: ' + ', '.join(already_refined)
                        + '. This run will replace them with fresh results.'
                    )
                    warning_card.show()
                else:
                    warning_card.hide()

                minimum_chunks = 1 if request_mode == 'all' else max(1, len(non_empty_current))
                maximum_chunks = max(1, total_entries)
                previous_chunks = chunk_count.value()
                chunk_count.blockSignals(True)
                chunk_count.setRange(minimum_chunks, maximum_chunks)
                chunk_count.setValue(max(
                    minimum_chunks,
                    min(previous_chunks or current_plan.total_chunks or minimum_chunks, maximum_chunks),
                ))
                chunk_count.blockSignals(False)
                chunk_range.setText(
                    f'Allowed range: {chunk_count.minimum():,}–{chunk_count.maximum():,}'
                )
                has_refinable_entries = bool(non_empty_current)
                refine_button.setEnabled(has_refinable_entries)
                _set_manual_chunk_controls_enabled(
                    override_checkbox.isChecked() and has_refinable_entries
                )
                _sync_select_all_checkbox()

            for type_choice in type_choice_widgets.values():
                type_choice.toggled.connect(_refresh_refinement_preview)

            def _toggle_all_preview_types():
                select_everything = not all(
                    type_choice.isChecked()
                    for type_choice in type_choice_widgets.values()
                )
                for type_choice in type_choice_widgets.values():
                    type_choice.blockSignals(True)
                    type_choice.setChecked(select_everything)
                    type_choice.blockSignals(False)
                _sync_type_choice_checkmarks()
                _refresh_refinement_preview()

            select_all_types_checkbox.clicked.connect(_toggle_all_preview_types)
            _refresh_refinement_preview()
            buttons.accepted.connect(preview.accept)
            buttons.rejected.connect(preview.reject)
            preview_layout.addWidget(buttons)
            if preview.exec() != QDialog.Accepted:
                return False

            finished = finish_manual_glossary_refinement(
                preview_state,
                list(selection_state.get('types') or []),
                chunk_count.value() if override_checkbox.isChecked() else None,
                selection_state.get('plan') or automatic_plan,
            )
            if not finished:
                return False
            options, execution_plan = finished
            return self._start_manual_glossary_refinement(
                glossary_path,
                progress_path,
                options,
                execution_plan,
            )
        
        def _build_gp_panel(fp, gp_path, parent_widget, pump_loading=None):
            """Build a glossary progress panel for a single EPUB. Returns (panel_widget, refresh_func)."""
            from PySide6.QtWidgets import QStackedWidget, QComboBox

            _gp_model = make_glossary_progress_model(
                self,
                fp,
                gp_path,
                pump_loading=pump_loading,
                glossary_progress_source_path=glossary_progress_source_path,
                glossary_progress_source_filenames=glossary_progress_source_filenames,
                output_dir=output_dir,
                prog=prog,
                find_gp_for_file=_find_gp_for_file,
                glossary_refinement_expected_entries=_glossary_refinement_expected_entries,
                refinement_type_key=_refinement_type_key,
                current_gp_data=lambda: gp_data,
            )
            _pump_loading_frame = _gp_model._pump_loading_frame
            _gp_load_progress_dict = _gp_model._gp_load_progress_dict
            _gp_int_list = _gp_model._gp_int_list
            _gp_safe_int = _gp_model._gp_safe_int
            gp_data = _gp_model.gp_data
            completed_indices = _gp_model.completed_indices
            skipped_indices = _gp_model.skipped_indices
            failed_indices = _gp_model.failed_indices
            merged_indices = _gp_model.merged_indices
            book_title = _gp_model.book_title
            _gp_qa_issue_map = _gp_model._gp_qa_issue_map
            _gp_filename_chapter_num = _gp_model._gp_filename_chapter_num
            _gp_source_chapter_num = _gp_model._gp_source_chapter_num
            _gp_display_chapter_num = _gp_model._gp_display_chapter_num
            _gp_filename_keys = _gp_model._gp_filename_keys
            _rebuild_reverse_lookups = _gp_model._rebuild_reverse_lookups
            _gp_index_for_actual_num = _gp_model._gp_index_for_actual_num
            _gp_index_for_entry = _gp_model._gp_index_for_entry
            _gp_index_for_progress_value = _gp_model._gp_index_for_progress_value
            _gp_sets = _gp_model._gp_sets
            _gp_skipped_set = _gp_model._gp_skipped_set
            _gp_in_progress_set = _gp_model._gp_in_progress_set
            _gp_status_cache = _gp_model._gp_status_cache
            _gp_status_for = _gp_model._gp_status_for
            _gp_entries_for = _gp_model._gp_entries_for
            _gp_entry_for = _gp_model._gp_entry_for
            _gp_model_for = _gp_model._gp_model_for
            _gp_display_for = _gp_model._gp_display_for
            _gp_glossary_entries = _gp_model._gp_glossary_entries
            _gp_minimal_pass_toggle_enabled = _gp_model._gp_minimal_pass_toggle_enabled
            _gp_minimal_pass_row = _gp_model._gp_minimal_pass_row
            _gp_minimal_pass_status_counts = _gp_model._gp_minimal_pass_status_counts
            _gp_refinement_rows = _gp_model._gp_refinement_rows
            _gp_refinement_status_counts = _gp_model._gp_refinement_status_counts
            _gp_extra_row_status_counts = _gp_model._gp_extra_row_status_counts
            _gp_color_for = _gp_model._gp_color_for
            _gp_restore_in_progress_entry = _gp_model._gp_restore_in_progress_entry
            _read_spine_map = _gp_model._read_spine_map
            _ts_init = _gp_model._ts_init
            _cmap_init = _gp_model._cmap_init
            _total_init = _gp_model._total_init
            _spine_idx_init = _gp_model._spine_idx_init
            panel_state = _gp_model.panel_state
            _gp_file_signature = _gp_model._gp_file_signature
            _gp_pending_result = _gp_model._gp_pending_result
            _invalidate_gp_refresh = _gp_model._invalidate_gp_refresh
            _finish_gp_refresh_callbacks = _gp_model._finish_gp_refresh_callbacks
            _gp_stats_for_dict = _gp_model._gp_stats_for_dict
            _gp_remove_mapped_indices_from_list = _gp_model._gp_remove_mapped_indices_from_list
            _gp_progress_uses_chapter_entries = _gp_model._gp_progress_uses_chapter_entries
            _gp_progress_value_for_index = _gp_model._gp_progress_value_for_index
            _gp_chapter_entry_map = _gp_model._gp_chapter_entry_map
            _gp_row_updates_for_targets = _gp_model._gp_row_updates_for_targets
            _gp_apply_mark_completed_to_progress = _gp_model._gp_apply_mark_completed_to_progress
            _gp_load_usage_inputs = _gp_model._gp_load_usage_inputs
            _gp_footnote_markdown_to_html = _gp_model._gp_footnote_markdown_to_html
            _gp_source_chapter_maps = _gp_model._gp_source_chapter_maps
            _gp_source_chapter_for_row = _gp_model._gp_source_chapter_for_row
            _gp_output_text_for_row = _gp_model._gp_output_text_for_row
            _gp_chapter_for_footnote = _gp_model._gp_chapter_for_footnote
            skip_unmatched_config_key = _gp_model.skip_unmatched_config_key
            _gp_skip_unmatched_entries = _gp_model._gp_skip_unmatched_entries
            _gp_set_skip_unmatched_entries = _gp_model._gp_set_skip_unmatched_entries
            _gp_write_completed_summary = _gp_model._gp_write_completed_summary
            _gp_collect_footnotes = _gp_model._gp_collect_footnotes
            _find_glossary_file = _gp_model._find_glossary_file
            _gp_apply_remove_from_progress = _gp_model._gp_apply_remove_from_progress
            
            panel = QWidget(parent_widget)
            p_layout = QVBoxLayout(panel)
            p_layout.setContentsMargins(4, 4, 4, 4)
            
            if book_title:
                bt_label = QLabel(f"📖 {book_title}")
                bt_label.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 10pt;")
                p_layout.addWidget(bt_label)
            else:
                bt_label = None

            missing_progress_label = QLabel(
                "⚠️ Progress file was deleted. Waiting for a new glossary progress file…"
            )
            missing_progress_label.setStyleSheet(
                "color: #f59e0b; font-size: 9pt; font-weight: bold; padding: 2px 0;"
            )
            missing_progress_label.setWordWrap(True)
            missing_progress_label.hide()
            p_layout.addWidget(missing_progress_label)
            
            # Stats row (clickable)
            _status_cache_init = _gp_status_cache(gp_data)
            _comp_set_init = _status_cache_init['completed']
            _skip_set_init = _status_cache_init['skipped']
            _fail_set_init = _status_cache_init['failed']
            _merg_set_init = _status_cache_init['merged']
            _in_prog_set_init = _status_cache_init['in_progress']
            # Completed count excludes chapters that are also merged
            n_completed = len(
                _comp_set_init - _skip_set_init - _merg_set_init - _fail_set_init
            )
            n_skipped = len(_skip_set_init)
            n_failed = len(_fail_set_init)
            n_merged = len(_merg_set_init)
            n_in_progress = len(_in_prog_set_init)
            n_remaining = max(0, panel_state['total'] - len(_comp_set_init | _skip_set_init | _fail_set_init | _merg_set_init | _in_prog_set_init))
            _gp_ref_counts_init = _gp_extra_row_status_counts(gp_data)
            _legend_stats_init = _combine_glossary_progress_legend_stats(
                {
                    'total': panel_state['total'],
                    'completed': n_completed,
                    'skipped': n_skipped,
                    'in_progress': n_in_progress,
                    'failed': n_failed,
                    'merged': n_merged,
                    'remaining': n_remaining,
                },
                _gp_ref_counts_init,
            )
            n_completed = _legend_stats_init['completed']
            n_skipped = _legend_stats_init['skipped']
            n_in_progress = _legend_stats_init['in_progress']
            n_not_refined = _legend_stats_init['not_refined']
            n_refine_failed = _legend_stats_init['refine_failed']
            
            gp_stats_frame = QWidget()
            gp_stats_layout = QHBoxLayout(gp_stats_frame)
            gp_stats_layout.setContentsMargins(0, 5, 0, 5)
            gp_stats_font = QFont('Arial', 9)
            
            lbl_total = QLabel(f"Total: {_legend_stats_init['total']} | ")
            lbl_total.setFont(gp_stats_font)
            gp_stats_layout.addWidget(lbl_total)
            
            lbl_gp_completed = QLabel(f"✅ Completed: {n_completed} | ")
            lbl_gp_completed.setFont(gp_stats_font)
            lbl_gp_completed.setStyleSheet("color: #27ae60;")
            lbl_gp_completed.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_completed)

            lbl_gp_skipped = QLabel(f"⏭️ Skipped: {n_skipped} | ")
            lbl_gp_skipped.setFont(gp_stats_font)
            lbl_gp_skipped.setStyleSheet("color: #94a3b8;")
            lbl_gp_skipped.setCursor(Qt.PointingHandCursor)
            lbl_gp_skipped.setVisible(True)
            gp_stats_layout.addWidget(lbl_gp_skipped)

            lbl_gp_in_progress = QLabel(f"🔄 In Progress: {n_in_progress} | ")
            lbl_gp_in_progress.setFont(gp_stats_font)
            lbl_gp_in_progress.setStyleSheet("color: #f59e0b;")
            lbl_gp_in_progress.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_in_progress)
            
            lbl_gp_failed = QLabel(f"❌ Failed: {n_failed} | ")
            lbl_gp_failed.setFont(gp_stats_font)
            lbl_gp_failed.setStyleSheet("color: #e74c3c;")
            lbl_gp_failed.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_failed)
            
            lbl_gp_merged = QLabel(f"🔗 Merged: {n_merged} | ")
            lbl_gp_merged.setFont(gp_stats_font)
            lbl_gp_merged.setStyleSheet("color: #17a2b8;")
            lbl_gp_merged.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_merged)
            if n_merged == 0:
                lbl_gp_merged.setVisible(False)
            
            lbl_gp_remaining = QLabel(f"⬜ Not Translated: {n_remaining}{' | ' if (n_not_refined or n_refine_failed) else ''}")
            lbl_gp_remaining.setFont(gp_stats_font)
            lbl_gp_remaining.setStyleSheet("color: #5a9fd4;")
            lbl_gp_remaining.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_remaining)

            lbl_gp_not_refined = QLabel(f"✨ Not Refined: {n_not_refined}")
            lbl_gp_not_refined.setFont(gp_stats_font)
            lbl_gp_not_refined.setStyleSheet("color: #5a9fd4;")
            lbl_gp_not_refined.setCursor(Qt.PointingHandCursor)
            lbl_gp_not_refined.setVisible(n_not_refined > 0)
            gp_stats_layout.addWidget(lbl_gp_not_refined)

            lbl_gp_refine_failed = QLabel(f"💀 Refine Failed: {n_refine_failed}")
            lbl_gp_refine_failed.setFont(gp_stats_font)
            lbl_gp_refine_failed.setStyleSheet("color: #7f5f00;")
            lbl_gp_refine_failed.setCursor(Qt.PointingHandCursor)
            lbl_gp_refine_failed.setVisible(n_refine_failed > 0)
            gp_stats_layout.addWidget(lbl_gp_refine_failed)
            
            gp_stats_layout.addStretch()
            p_layout.addWidget(gp_stats_frame)
            _pump_loading_frame()
            
            # Chapter list
            gp_listbox = QListWidget()
            self._apply_compact_inline_list_style(gp_listbox, QFont('Courier', 10))
            gp_listbox.setUniformItemSizes(True)
            gp_listbox.setContextMenuPolicy(Qt.CustomContextMenu)
            gp_listbox.setSelectionMode(QListWidget.ExtendedSelection)
            
            completed_set, failed_set, merged_set = _gp_sets(gp_data)
            
            chapter_map = panel_state['chapter_map']
            total_epub_chapters = 0
            
            for ci in range(total_epub_chapters):
                fname = chapter_map.get(ci, f'chapter {ci + 1}')
                ch_num = _gp_display_chapter_num(ci, fname)
                
                if ci in merged_set:
                    icon, status, color = '🔗', 'merged', '#17a2b8'
                elif ci in completed_set:
                    icon, status, color = '✅', 'completed', '#27ae60'
                elif ci in failed_set:
                    icon, status, color = '❌', 'failed', '#e74c3c'
                else:
                    icon, status, color = '⬜', 'not_completed', '#5a9fd4'
                
                display = f"Ch.{ch_num:03d} | {icon} {status.replace('_', ' ').title():14s} | {fname}"
                display, status = _gp_display_for(ci, fname, gp_data)
                color = _gp_color_for(status)
                item = QListWidgetItem(display)
                item.setForeground(QColor(color))
                item.setData(Qt.UserRole, status)
                item.setData(Qt.UserRole + 1, ci)  # Store chapter index for deletion
                self._add_compact_inline_list_item(gp_listbox, item)

            def _refresh_minimal_pass_row(_d, keep_updates_disabled=False):
                """Keep the Minimal-pass row as the first entry in the list."""
                row_data = _gp_minimal_pass_row(_d)
                fingerprint = tuple(row_data) if row_data else None
                if fingerprint == panel_state.get('_minimal_pass_fingerprint'):
                    return False
                if not keep_updates_disabled:
                    gp_listbox.setUpdatesEnabled(False)
                try:
                    for row in range(gp_listbox.count() - 1, -1, -1):
                        item = gp_listbox.item(row)
                        if item and item.data(Qt.UserRole + 4):
                            gp_listbox.takeItem(row)
                    if row_data:
                        key, display, status = row_data
                        item = QListWidgetItem(display)
                        item.setForeground(QColor(_gp_color_for(status)))
                        item.setData(Qt.UserRole, status)
                        item.setData(Qt.UserRole + 1, None)
                        item.setData(Qt.UserRole + 4, key)
                        self._set_compact_inline_item_size(gp_listbox, item)
                        gp_listbox.insertItem(0, item)
                    panel_state['_minimal_pass_fingerprint'] = fingerprint
                finally:
                    if not keep_updates_disabled:
                        gp_listbox.setUpdatesEnabled(True)
                        gp_listbox.viewport().update()
                return True

            def _refresh_refinement_rows(_d, keep_updates_disabled=False):
                refinement_rows = _gp_refinement_rows(_d)
                refinement_fingerprint = tuple(refinement_rows)
                if refinement_fingerprint == panel_state.get('_refinement_fingerprint'):
                    return False
                selected_ref_keys = {
                    it.data(Qt.UserRole + 3)
                    for it in gp_listbox.selectedItems()
                    if it and it.data(Qt.UserRole + 3)
                }
                if not keep_updates_disabled:
                    gp_listbox.setUpdatesEnabled(False)
                try:
                    for row in range(gp_listbox.count() - 1, -1, -1):
                        item = gp_listbox.item(row)
                        if item and item.data(Qt.UserRole + 3):
                            gp_listbox.takeItem(row)

                    for ref_key, ref_display, ref_status in refinement_rows:
                        item = QListWidgetItem(ref_display)
                        item.setForeground(QColor(_gp_color_for(ref_status)))
                        item.setData(Qt.UserRole, ref_status)
                        item.setData(Qt.UserRole + 1, None)
                        item.setData(Qt.UserRole + 3, ref_key)
                        self._add_compact_inline_list_item(gp_listbox, item)
                        if ref_key in selected_ref_keys:
                            item.setSelected(True)
                    panel_state['_refinement_fingerprint'] = refinement_fingerprint
                finally:
                    if not keep_updates_disabled:
                        gp_listbox.setUpdatesEnabled(True)
                        gp_listbox.viewport().update()
                return True
            
            def _populate_gp_listbox(_d, chunk_size=150):
                panel_state['populate_generation'] = panel_state.get('populate_generation', 0) + 1
                generation = panel_state['populate_generation']
                cache = _gp_status_cache(_d)
                total = panel_state['total']
                chapter_map = panel_state['chapter_map']
                saved_scroll = gp_listbox.verticalScrollBar().value()
                selected_chapters = {
                    item.data(Qt.UserRole + 1)
                    for item in gp_listbox.selectedItems()
                    if item and item.data(Qt.UserRole + 1) is not None
                }
                new_chapter_items = {}
                new_fingerprints = {}
                state = {'ci': 0}
                # Placed before the chapter rows are laid out so the index
                # offset below is stable for the whole population pass.
                _refresh_minimal_pass_row(_d, keep_updates_disabled=True)

                def _gp_leading_rows():
                    """Rows pinned above the chapters (currently the Minimal pass).

                    Chapter rows are addressed by list index, so every index
                    below has to be shifted by however many rows sit above
                    them or the list silently mismatches chapter data.
                    """
                    first = gp_listbox.item(0) if gp_listbox.count() else None
                    return 1 if (first is not None and first.data(Qt.UserRole + 4)) else 0

                def _add_chunk():
                    if generation != panel_state.get('populate_generation'):
                        return
                    signals_were_blocked = gp_listbox.signalsBlocked()
                    updates_were_enabled = gp_listbox.updatesEnabled()
                    start_ci = state['ci']
                    end_ci = min(start_ci + chunk_size, total)
                    try:
                        gp_listbox.blockSignals(True)
                        gp_listbox.setUpdatesEnabled(False)
                        lead = _gp_leading_rows()
                        for ci in range(start_ci, end_ci):
                            row_index = ci + lead
                            fname = chapter_map.get(ci, f'chapter {ci + 1}')
                            display, status = _gp_display_for(ci, fname, _d, cache)
                            color = _gp_color_for(status)
                            item = (
                                gp_listbox.item(row_index)
                                if row_index < gp_listbox.count()
                                else None
                            )
                            if item is None or item.data(Qt.UserRole + 3) or item.data(Qt.UserRole + 4):
                                item = QListWidgetItem(display)
                                self._set_compact_inline_item_size(gp_listbox, item)
                                gp_listbox.insertItem(row_index, item)
                            elif item.text() != display:
                                item.setText(display)
                            self._set_compact_inline_item_size(gp_listbox, item)
                            item.setForeground(QColor(color))
                            item.setData(Qt.UserRole, status)
                            item.setData(Qt.UserRole + 1, ci)
                            item.setData(Qt.UserRole + 3, None)
                            new_chapter_items[ci] = item
                            new_fingerprints[ci] = (status, color, display)
                            if panel_state.get('select_all_visible'):
                                item.setSelected(not item.isHidden())
                            else:
                                item.setSelected(ci in selected_chapters)
                        state['ci'] = end_ci
                    finally:
                        gp_listbox.blockSignals(signals_were_blocked)
                        gp_listbox.setUpdatesEnabled(updates_were_enabled)
                        gp_listbox.viewport().update()

                    if end_ci < total:
                        QTimer.singleShot(0, _add_chunk)
                    else:
                        signals_were_blocked = gp_listbox.signalsBlocked()
                        updates_were_enabled = gp_listbox.updatesEnabled()
                        try:
                            gp_listbox.blockSignals(True)
                            gp_listbox.setUpdatesEnabled(False)
                            # Remove obsolete chapter rows without clearing the
                            # live list or collapsing its scrollbar range.
                            lead = _gp_leading_rows()
                            for row in range(gp_listbox.count() - 1, total + lead - 1, -1):
                                item = gp_listbox.item(row)
                                if item and not item.data(Qt.UserRole + 3) and not item.data(Qt.UserRole + 4):
                                    gp_listbox.takeItem(row)
                            panel_state['_chapter_items'] = new_chapter_items
                            panel_state['_row_fingerprints'] = new_fingerprints
                            panel_state['_refinement_fingerprint'] = None
                            panel_state['_minimal_pass_fingerprint'] = None
                            _refresh_refinement_rows(_d, keep_updates_disabled=True)
                            _refresh_minimal_pass_row(_d, keep_updates_disabled=True)
                            sb = gp_listbox.verticalScrollBar()
                            sb.setValue(min(saved_scroll, sb.maximum()))
                        finally:
                            gp_listbox.blockSignals(signals_were_blocked)
                            gp_listbox.setUpdatesEnabled(updates_were_enabled)
                            gp_listbox.viewport().update()

                QTimer.singleShot(0, _add_chunk)

            _populate_gp_listbox(gp_data)


            def _apply_gp_stats(stats):
                if not isinstance(stats, dict):
                    return
                lbl_total.setText(f"Total: {stats.get('total', 0)} | ")
                lbl_gp_completed.setText(f"✅ Completed: {stats.get('completed', 0)} | ")
                skipped_count = stats.get('skipped', 0)
                lbl_gp_skipped.setText(f"⏭️ Skipped: {skipped_count} | ")
                lbl_gp_skipped.setVisible(True)
                lbl_gp_in_progress.setText(f"🔄 In Progress: {stats.get('in_progress', 0)} | ")
                lbl_gp_in_progress.setVisible(True)
                lbl_gp_failed.setText(f"❌ Failed: {stats.get('failed', 0)} | ")
                merged_count = stats.get('merged', 0)
                lbl_gp_merged.setText(f"🔗 Merged: {merged_count} | ")
                lbl_gp_merged.setVisible(merged_count > 0)
                not_refined = stats.get('not_refined', 0)
                refine_failed = stats.get('refine_failed', 0)
                lbl_gp_remaining.setText(f"⬜ Not Translated: {stats.get('remaining', 0)}{' | ' if (not_refined or refine_failed) else ''}")
                lbl_gp_not_refined.setText(f"✨ Not Refined: {not_refined}")
                lbl_gp_not_refined.setVisible(not_refined > 0)
                lbl_gp_refine_failed.setText(f"💀 Refine Failed: {refine_failed}")
                lbl_gp_refine_failed.setVisible(refine_failed > 0)

            # Helper to refresh stats labels from a loaded progress dict
            def _refresh_stats_from_dict(_d):
                _apply_gp_stats(_gp_stats_for_dict(_d))

            def _gp_context_target_label(item, target):
                target_kind, target_value = target
                display_text = item.text() if item else ""
                if target_kind == 'refinement':
                    return display_text.split('|')[-1].strip() or str(target_value)
                label_match = re.search(r'\bCh\.\d+(?:\.\d+)?\b', display_text or "")
                return label_match.group(0) if label_match else f"Ch.{target_value + 1}"


            def _apply_gp_mark_completed_result(payload):
                nonlocal gp_data
                if not isinstance(payload, dict):
                    return
                job_id = payload.get('job_id')
                targets = (panel_state.setdefault('_mark_completed_jobs', {})).pop(job_id, [])
                _d = payload.get('data') if isinstance(payload.get('data'), dict) else {}
                if _d:
                    gp_data = _d
                row_updates = payload.get('row_updates') if isinstance(payload.get('row_updates'), dict) else {}
                stats = payload.get('stats')
                updates_were_enabled = {'value': True}

                def _finish_row_updates():
                    try:
                        if payload.get('refresh_refinement_rows') and _d:
                            _refresh_refinement_rows(_d, keep_updates_disabled=True)
                            _refresh_minimal_pass_row(_d, keep_updates_disabled=True)
                        gp_listbox.setUpdatesEnabled(updates_were_enabled['value'])
                        gp_listbox.viewport().update()
                    except RuntimeError:
                        pass
                    if isinstance(stats, dict):
                        _apply_gp_stats(stats)
                    elif _d:
                        _refresh_stats_from_dict(_d)

                def _apply_row_update_chunk(start=0, chunk_size=200):
                    if start == 0:
                        try:
                            updates_were_enabled['value'] = gp_listbox.updatesEnabled()
                            gp_listbox.setUpdatesEnabled(False)
                        except RuntimeError:
                            return
                    end = min(start + chunk_size, len(targets))
                    for item, target in targets[start:end]:
                        update = row_updates.get(tuple(target))
                        if not update:
                            continue
                        try:
                            item.setText(update.get('display', item.text()))
                            item.setForeground(QColor(update.get('color') or _gp_color_for(update.get('status'))))
                            item.setData(Qt.UserRole, update.get('status'))
                        except RuntimeError:
                            continue
                    if end < len(targets):
                        QTimer.singleShot(0, lambda: _apply_row_update_chunk(end, chunk_size))
                    else:
                        _finish_row_updates()

                if targets:
                    _apply_row_update_chunk()
                else:
                    _finish_row_updates()

            def _apply_gp_mark_completed_error(message):
                try:
                    self._show_message(
                        'error',
                        "Glossary Progress",
                        f"Failed to mark glossary progress as completed:\n{message}",
                        parent=parent_widget,
                    )
                except Exception:
                    print(f"⚠️ Error marking glossary progress complete: {message}")

            gp_mark_completed_bridge = _GlossaryProgressAsyncBridge(
                _apply_gp_mark_completed_result,
                _apply_gp_mark_completed_error,
                gp_listbox,
            )

            def _gp_mark_targets_completed(targets):
                _rp = _find_gp_for_file(fp)
                if not _rp or not os.path.isfile(_rp):
                    return
                target_specs = [target for _, target in targets]
                job_id = f"{time.time():.6f}:{id(targets)}"
                panel_state.setdefault('_mark_completed_jobs', {})[job_id] = list(targets)

                def _worker():
                    try:
                        mark_lock = panel_state.get('mark_completed_lock')
                        if mark_lock:
                            with mark_lock:
                                payload = _gp_apply_mark_completed_to_progress(_rp, target_specs)
                        else:
                            payload = _gp_apply_mark_completed_to_progress(_rp, target_specs)
                        if not isinstance(payload, dict):
                            payload = {'changed': False, 'data': {}}
                        payload['job_id'] = job_id
                        gp_mark_completed_bridge.finished.emit(payload)
                    except Exception as e:
                        panel_state.setdefault('_mark_completed_jobs', {}).pop(job_id, None)
                        try:
                            gp_mark_completed_bridge.failed.emit(str(e))
                        except RuntimeError:
                            pass

                threading.Thread(target=_worker, name="glossary-progress-mark-completed", daemon=True).start()

            def _gp_usage_inputs(show_errors=True, progress_callback=None):
                loaded, error = _gp_load_usage_inputs(progress_callback)
                if error is not None:
                    if show_errors:
                        kind, title, message = error
                        if kind == 'critical':
                            QMessageBox.critical(gp_listbox, title, message)
                        else:
                            QMessageBox.information(gp_listbox, title, message)
                    return None
                return loaded


            def _gp_summary_loading_html(message, detail=""):
                detail_html = f"<p>{html_lib.escape(detail)}</p>" if detail else ""
                return (
                    "<html><head><style>"
                    "body { background: #2d2d2d; color: #f2f2f2; font-family: Segoe UI, Arial, sans-serif; font-size: 15px; }"
                    "h2 { margin: 0 0 16px 0; font-size: 24px; font-weight: 700; }"
                    ".loading { color: #d8d8d8; font-size: 18px; font-style: italic; }"
                    "p { color: #bfc5cf; }"
                    "</style></head><body>"
                    "<h2>Generating Completed Glossary Summary</h2>"
                    f'<p class="loading">{html_lib.escape(message)}</p>'
                    f"{detail_html}"
                    "</body></html>"
                )

            def _show_gp_footnote_dialog(text, title="Glossary Footnote"):
                def _split_entry_line(line):
                    content = line[2:].strip()
                    status = ""
                    for marker, marker_status in ((WARNING_PREFIX, "warning"), (CHECK_PREFIX, "confirmed")):
                        marker_prefix = f"{marker} "
                        if content.startswith(marker_prefix):
                            status = marker_status
                            content = content[len(marker_prefix):].strip()
                            break
                    label = content
                    details = ""
                    if content.endswith(")"):
                        details_start = content.rfind(" (")
                        if details_start > -1:
                            label = content[:details_start].strip()
                            details = content[details_start + 2:-1].strip()
                    return status, label, details

                def _entry_details_html(details):
                    if not details:
                        return ""
                    parts = [part.strip() for part in details.split(";") if part.strip()]
                    tags = []
                    desc_parts = []
                    if parts:
                        tags.append(parts[0])
                    if len(parts) > 1 and parts[1].casefold() in {"male", "female", "unknown", "nonbinary", "non-binary", "other"}:
                        tags.append(parts[1])
                        desc_parts = parts[2:]
                    else:
                        desc_parts = parts[1:]
                    blocks = ['<div class="entry-meta">']
                    if tags:
                        tag_text = " | ".join(html_lib.escape(part) for part in tags)
                        blocks.append(f'<div class="entry-tags">{tag_text}</div>')
                    if desc_parts:
                        desc_text = html_lib.escape("; ".join(desc_parts))
                        blocks.append(f'<div class="entry-desc">{desc_text}</div>')
                    blocks.append("</div>")
                    return "".join(blocks)

                def _footnote_markdown_to_html(markdown_text):
                    html_parts = [
                        "<html><head><style>",
                        "body { background: #2d2d2d; color: #f2f2f2; font-family: Segoe UI, Arial, sans-serif; font-size: 15px; line-height: 1.35; }",
                        "h1, h2 { margin: 0 0 12px 0; font-size: 26px; font-weight: 700; }",
                        "h3 { margin: 18px 0 10px 0; font-size: 20px; font-weight: 700; }",
                        "p { margin: 4px 0 10px 0; }",
                        "ul.meta-list { margin: 0 0 14px 28px; }",
                        "ul.entry-list { margin: 6px 0 0 28px; }",
                        "li { margin: 5px 0; }",
                        "li.entry { margin: 0 0 12px 0; }",
                        ".entry-label { color: #ffffff; font-weight: 600; }",
                        ".entry-warning { color: #ffca66; font-weight: 700; }",
                        ".entry-confirmed { color: #8bdc81; font-weight: 700; }",
                        ".entry-meta { margin-top: 2px; color: #c5c9d1; font-size: 13px; line-height: 1.35; }",
                        ".entry-tags { color: #9ecbff; font-weight: 600; }",
                        ".entry-desc { color: #c7cbd2; }",
                        ".saved-path { color: #d8d8d8; font-family: Consolas, monospace; }",
                        "</style></head><body>",
                    ]
                    list_kind = None
                    in_entries = False

                    def close_list():
                        nonlocal list_kind
                        if list_kind:
                            html_parts.append("</ul>")
                            list_kind = None

                    def open_list(kind):
                        nonlocal list_kind
                        if list_kind == kind:
                            return
                        close_list()
                        class_name = "entry-list" if kind == "entries" else "meta-list"
                        html_parts.append(f'<ul class="{class_name}">')
                        list_kind = kind

                    for raw_line in markdown_text.splitlines():
                        line = raw_line.strip()
                        if not line:
                            close_list()
                            continue
                        if line.startswith("### "):
                            close_list()
                            heading = line[4:].strip()
                            in_entries = heading.casefold() == "matched glossary entries"
                            html_parts.append(f"<h3>{html_lib.escape(heading)}</h3>")
                        elif line.startswith("## "):
                            close_list()
                            in_entries = False
                            html_parts.append(f"<h2>{html_lib.escape(line[3:].strip())}</h2>")
                        elif line.startswith("# "):
                            close_list()
                            in_entries = False
                            html_parts.append(f"<h1>{html_lib.escape(line[2:].strip())}</h1>")
                        elif line.startswith("- "):
                            if in_entries:
                                open_list("entries")
                                status, label, details = _split_entry_line(line)
                                marker = ""
                                if status == "warning":
                                    marker = f'<span class="entry-warning">{html_lib.escape(WARNING_PREFIX)}</span> '
                                elif status == "confirmed":
                                    marker = f'<span class="entry-confirmed">{html_lib.escape(CHECK_PREFIX)}</span> '
                                html_parts.append(
                                    '<li class="entry">'
                                    f'<div class="entry-label">{marker}{html_lib.escape(label)}</div>'
                                    f"{_entry_details_html(details)}"
                                    "</li>"
                                )
                            else:
                                open_list("meta")
                                html_parts.append(f"<li>{html_lib.escape(line[2:].strip())}</li>")
                        else:
                            close_list()
                            escaped = html_lib.escape(line)
                            if os.path.isabs(line):
                                html_parts.append(f'<p class="saved-path">{escaped}</p>')
                            else:
                                html_parts.append(f"<p>{escaped}</p>")
                    close_list()
                    html_parts.append("</body></html>")
                    return "".join(html_parts)

                footnote_dialog = QDialog(gp_listbox)
                footnote_dialog.setWindowTitle(title)
                footnote_dialog.resize(760, 560)
                layout = QVBoxLayout(footnote_dialog)
                viewer = QTextBrowser()
                viewer.setReadOnly(True)
                viewer.setHtml(_gp_footnote_markdown_to_html(text))
                layout.addWidget(viewer)
                buttons = QHBoxLayout()
                copy_btn = QPushButton("Copy")
                close_btn = QPushButton("Close")
                copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(text))
                close_btn.clicked.connect(footnote_dialog.accept)
                buttons.addStretch()
                buttons.addWidget(copy_btn)
                buttons.addWidget(close_btn)
                layout.addLayout(buttons)
                footnote_dialog.exec()


            def _gp_show_footnotes(targets):
                loaded = _gp_usage_inputs(show_errors=True)
                if not loaded:
                    return
                entries, chapters, progress_data = loaded
                parts, missing, missing_output = _gp_collect_footnotes(
                    entries,
                    chapters,
                    progress_data,
                    [value for _item, (_kind, value) in targets],
                )
                if not parts:
                    if missing_output:
                        QMessageBox.information(
                            gp_listbox,
                            "Glossary Footnote",
                            "Could not resolve translated output file for:\n"
                            + "\n".join(
                                f"- Ch.{_gp_display_chapter_num(ci, (panel_state.get('chapter_map') or {}).get(ci, ''))}: {path}"
                                for ci, path in missing_output
                            ),
                        )
                    else:
                        detail = f" Missing chapter indices: {missing}" if missing else ""
                        QMessageBox.information(gp_listbox, "Glossary Footnote", f"No selected chapters could be mapped to source text.{detail}")
                    return
                if missing_output:
                    QMessageBox.warning(
                        gp_listbox,
                        "Glossary Footnote",
                        "Skipped rows with unresolved translated output files:\n"
                        + "\n".join(
                            f"- Ch.{_gp_display_chapter_num(ci, (panel_state.get('chapter_map') or {}).get(ci, ''))}: {path}"
                            for ci, path in missing_output
                        ),
                    )
                _show_gp_footnote_dialog("\n\n".join(parts) + "\n", "Glossary Footnote")

            def _gp_generate_completed_summary():
                summary_dialog = QDialog(gp_listbox)
                summary_dialog.setWindowTitle("Completed Glossary Footnote Summary")
                summary_dialog.resize(760, 560)
                layout = QVBoxLayout(summary_dialog)
                viewer = QTextBrowser()
                viewer.setReadOnly(True)
                viewer.setHtml(_gp_summary_loading_html("Starting background summary worker..."))
                layout.addWidget(viewer)

                text_holder = {"text": ""}
                buttons = QHBoxLayout()
                copy_btn = QPushButton("Copy")
                close_btn = QPushButton("Close")
                copy_btn.setEnabled(False)
                copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(text_holder.get("text", "")))
                close_btn.clicked.connect(summary_dialog.accept)
                buttons.addStretch()
                buttons.addWidget(copy_btn)
                buttons.addWidget(close_btn)
                layout.addLayout(buttons)

                def _on_summary_progress(payload):
                    payload = payload if isinstance(payload, dict) else {"message": str(payload)}
                    stage = str(payload.get("stage") or "")
                    if stage == "generating":
                        current = int(payload.get("current") or 0)
                        total = int(payload.get("total") or 0)
                        chapter_num = payload.get("chapter_num", "")
                        status = str(payload.get("status") or "working")
                        detail = f"Chapter {chapter_num} - {status}" if chapter_num not in ("", None) else status
                        viewer.setHtml(
                            _gp_summary_loading_html(
                                f"Generating summary entries: {current}/{total}",
                                detail,
                            )
                        )
                    else:
                        viewer.setHtml(_gp_summary_loading_html(str(payload.get("message") or "Preparing summary...")))

                def _on_summary_finished(payload):
                    summary_path = payload.get("summary_path", "")
                    content = payload.get("content", "")
                    text = f"Saved to:\n{summary_path}\n\n{content}"
                    text_holder["text"] = text
                    copy_btn.setEnabled(True)
                    viewer.setHtml(_gp_footnote_markdown_to_html(text))

                def _on_summary_failed(message):
                    text_holder["text"] = f"Could not write completed summary:\n{message}"
                    copy_btn.setEnabled(True)
                    viewer.setHtml(
                        _gp_summary_loading_html(
                            "Could not write completed summary.",
                            str(message),
                        )
                    )

                summary_queue = queue.Queue()
                summary_timer = QTimer(summary_dialog)
                summary_timer.setInterval(50)

                def _drain_summary_queue():
                    handled = False
                    while True:
                        try:
                            kind, payload = summary_queue.get_nowait()
                        except queue.Empty:
                            break
                        handled = True
                        if kind == "progress":
                            _on_summary_progress(payload)
                        elif kind == "finished":
                            summary_timer.stop()
                            _on_summary_finished(payload if isinstance(payload, dict) else {})
                        elif kind == "failed":
                            summary_timer.stop()
                            _on_summary_failed(str(payload))
                    return handled

                summary_timer.timeout.connect(_drain_summary_queue)

                def _emit_summary_progress(payload):
                    summary_queue.put(("progress", payload))

                def _summary_worker():
                    try:
                        _emit_summary_progress({"stage": "loading", "message": "Loading glossary and source chapters..."})
                        loaded = _gp_usage_inputs(
                            show_errors=False,
                            progress_callback=lambda message: _emit_summary_progress(
                                {"stage": "loading", "message": str(message)}
                            ),
                        )
                        if not loaded:
                            raise RuntimeError("Could not load glossary entries, source chapters, or progress data.")
                        entries, chapters, progress_data = loaded
                        _emit_summary_progress(
                            {
                                "stage": "loading",
                                "message": f"Preparing {len(entries)} glossary entries for completed and merged chapters...",
                            }
                        )

                        def _progress_callback(payload):
                            payload = dict(payload or {})
                            payload["stage"] = "generating"
                            _emit_summary_progress(payload)

                        summary_path, content = _gp_write_completed_summary(
                            entries,
                            chapters,
                            progress_data,
                            progress_callback=_progress_callback,
                            skip_unmatched_entries=_gp_skip_unmatched_entries(),
                        )
                        summary_queue.put(("finished", {"summary_path": summary_path, "content": content}))
                    except Exception as exc:
                        summary_queue.put(("failed", str(exc)))

                summary_dialog.show()
                summary_timer.start()
                summary_thread = threading.Thread(target=_summary_worker, name="GlossaryCompletedSummary", daemon=True)
                summary_thread.start()
                QApplication.processEvents(QEventLoop.AllEvents, 50)
                _drain_summary_queue()
                summary_dialog.exec()
            
            # Right-click context menu to delete entries from progress
            def _gp_context_menu(pos):
                nonlocal gp_data
                # Gather selected items that are deletable
                clicked_item = gp_listbox.itemAt(pos)
                if clicked_item is not None and not clicked_item.isSelected():
                    gp_listbox.clearSelection()
                    clicked_item.setSelected(True)
                selected = gp_listbox.selectedItems()
                deletable_statuses = (
                    'completed',
                    'skipped',
                    'skipped_empty',
                    'skipped_image_only',
                    'skipped_title_header_only',
                    'merged',
                    'in_progress',
                    'partially_in_progress',
                    'failed',
                    'qa_failed',
                    'not_refined',
                )
                removable_targets = []
                mark_completed_targets = []
                footnote_targets = []
                refinement_targets = []
                for it in selected:
                    status = it.data(Qt.UserRole)
                    refinement_key = it.data(Qt.UserRole + 3)
                    chapter_index = it.data(Qt.UserRole + 1)
                    target = None
                    minimal_pass_key = it.data(Qt.UserRole + 4)
                    if minimal_pass_key:
                        # Removing the record resets the row to Not Translated
                        # and lets the next run execute the pass again.
                        if status in deletable_statuses:
                            removable_targets.append((it, ('minimal_pass', minimal_pass_key)))
                        continue
                    if refinement_key:
                        target = ('refinement', refinement_key)
                        refinement_targets.append((it, refinement_key, status))
                    elif chapter_index is not None:
                        target = ('chapter', chapter_index)
                    if not target:
                        continue
                    if target[0] == 'chapter':
                        footnote_targets.append((it, target))
                    if status in deletable_statuses:
                        removable_targets.append((it, target))
                    if status != 'completed':
                        mark_completed_targets.append((it, target))
                if not removable_targets and not mark_completed_targets and not footnote_targets and not refinement_targets:
                    return
                
                from PySide6.QtWidgets import QMenu
                menu = QMenu(gp_listbox)
                menu.setStyleSheet(
                    "QMenu { background-color: #2d2d2d; color: white; border: 1px solid #555; padding: 4px 8px 4px 4px; }"
                    "QMenu::item { padding: 6px 24px 6px 6px; }"
                    "QMenu::item:selected { background-color: #c0392b; }"
                    "QMenu::item:disabled { color: #6b7280; background-color: transparent; }"
                )
                
                refine_action = None
                normalized_refinement_types = []
                if refinement_targets:
                    active_refinement_types = _active_glossary_refinement_types()
                    aggregate_selected = any(str(key).startswith('all::') for _it, key, _status in refinement_targets)
                    normalized_refinement_types = _normalize_glossary_refinement_selection(
                        [key for _it, key, _status in refinement_targets],
                        active_refinement_types,
                    )
                    current_counts = Counter(
                        _refinement_type_key(entry.get('type'))
                        for entry in _gp_glossary_entries(gp_data)
                        if isinstance(entry, dict)
                    )
                    eligible_count = sum(
                        current_counts.get(_refinement_type_key(entry_type), 0)
                        for entry_type in normalized_refinement_types
                    )
                    if aggregate_selected or len(refinement_targets) == 1:
                        refine_action = menu.addAction('✨ Refine this')
                    else:
                        refine_action = menu.addAction(
                            f'✨ Refine selected entry types ({len(normalized_refinement_types)})'
                        )
                    refine_action.setEnabled(eligible_count > 0)

                mark_action = None
                if mark_completed_targets:
                    if refine_action is not None:
                        menu.addSeparator()
                    mark_action = menu.addAction("✅ Mark as Completed")

                footnote_action = None
                summary_action = None
                skip_unmatched_action = None
                if footnote_targets:
                    if mark_action is not None:
                        menu.addSeparator()
                    label = "📝 Show Glossary Footnote" if len(footnote_targets) == 1 else f"📝 Show Glossary Footnotes ({len(footnote_targets)})"
                    footnote_action = menu.addAction(label)
                    summary_action = menu.addAction("📄 Generate completed summary")

                    skip_unmatched_action = menu.addAction(
                        "✅ Skip unmatched entries"
                        if _gp_skip_unmatched_entries()
                        else "❌ Skip unmatched entries"
                    )

                remove_action = None
                if removable_targets:
                    if mark_action is not None or footnote_action is not None:
                        menu.addSeparator()
                    n = len(removable_targets)
                    if n == 1:
                        status = removable_targets[0][0].data(Qt.UserRole)
                        chapter_label = (
                            "Minimal Pass"
                            if removable_targets[0][1][0] == 'minimal_pass'
                            else _gp_context_target_label(*removable_targets[0])
                        )
                        remove_action = menu.addAction(f"🗑️ Remove {chapter_label} from progress ({status})")
                    else:
                        remove_action = menu.addAction(f"🗑️ Remove {n} chapters from progress")
                
                chosen = menu.exec(gp_listbox.viewport().mapToGlobal(pos))
                if refine_action is not None and chosen == refine_action:
                    refinement_progress = gp_data.get('refinement', {}) if isinstance(gp_data, dict) else {}
                    completed_types = []
                    if isinstance(refinement_progress, dict):
                        for key, info in refinement_progress.items():
                            if (
                                str(key).startswith('type::')
                                and isinstance(info, dict)
                                and str(info.get('status') or '').lower() == 'completed'
                            ):
                                completed_types.append(str(info.get('entry_type') or str(key).split('::', 1)[1]))
                    _confirm_manual_glossary_refinement(
                        panel,
                        fp,
                        _find_gp_for_file(fp) or gp_path,
                        normalized_refinement_types,
                        completed_types=completed_types,
                    )
                    return
                if mark_action is not None and chosen == mark_action:
                    _gp_mark_targets_completed(mark_completed_targets)
                    return
                if footnote_action is not None and chosen == footnote_action:
                    _gp_show_footnotes(footnote_targets)
                    return
                if summary_action is not None and chosen == summary_action:
                    _gp_generate_completed_summary()
                    return
                if skip_unmatched_action is not None and chosen == skip_unmatched_action:
                    _gp_set_skip_unmatched_entries(not _gp_skip_unmatched_entries())
                    return
                if remove_action is None or chosen != remove_action:
                    return
                
                # Remove from progress JSON
                try:
                    _rp = _find_gp_for_file(fp)
                    if not _rp or not os.path.isfile(_rp):
                        return
                    changed, _d, remove_minimal_pass = _gp_apply_remove_from_progress(
                        _rp,
                        [target for _item, target in removable_targets],
                    )
                    if changed:
                        gp_data = _d
                        _invalidate_gp_refresh()
                        # Update all affected items
                        _cmap = panel_state['chapter_map']
                        if remove_minimal_pass:
                            panel_state['_minimal_pass_fingerprint'] = None
                            _refresh_minimal_pass_row(_d)
                        for it, (kind, value) in removable_targets:
                            if kind == 'minimal_pass':
                                continue
                            if kind == 'refinement':
                                row = next((r for r in _gp_refinement_rows(_d) if r[0] == value), None)
                                if row:
                                    _rk, display3, restored_status = row
                                    it.setText(display3)
                                    it.setForeground(QColor(_gp_color_for(restored_status)))
                                    it.setData(Qt.UserRole, restored_status)
                                else:
                                    gp_listbox.takeItem(gp_listbox.row(it))
                            else:
                                ci = value
                                fname = _cmap.get(ci, f'chapter {ci + 1}')
                                display3, restored_status = _gp_display_for(ci, fname, _d)
                                it.setText(display3)
                                it.setForeground(QColor(_gp_color_for(restored_status)))
                                it.setData(Qt.UserRole, restored_status)
                        _refresh_stats_from_dict(_d)
                except Exception as e:
                    print(f"⚠️ Error removing chapters from progress: {e}")
            
            gp_listbox.customContextMenuRequested.connect(_gp_context_menu)
            
            # Cycle handler
            def _gp_make_cycle(target_statuses, lb_ref):
                target_statuses = set(target_statuses)
                def _handler(_event=None):
                    lb = lb_ref
                    if not lb:
                        return
                    indices = []
                    for i in range(lb.count()):
                        item = lb.item(i)
                        if not item or item.isHidden():
                            continue
                        status = item.data(Qt.UserRole)
                        if isinstance(status, str):
                            status = status.lower().replace(' ', '_')
                        if status in target_statuses:
                            indices.append(i)
                    if not indices:
                        return
                    selected_rows = [lb.row(item) for item in lb.selectedItems()]
                    current = lb.currentRow()
                    if selected_rows and current not in selected_rows:
                        current = max(selected_rows)
                    nxt = next((i for i in indices if i > current), indices[0])
                    lb.setCurrentRow(nxt, QItemSelectionModel.ClearAndSelect)
                    lb.scrollToItem(lb.item(nxt), QListWidget.PositionAtCenter)
                return _handler
            
            lbl_gp_completed.mousePressEvent = _gp_make_cycle(('completed',), gp_listbox)
            lbl_gp_skipped.mousePressEvent = _gp_make_cycle((
                'skipped',
                'skipped_empty',
                'skipped_image_only',
                'skipped_title_header_only',
            ), gp_listbox)
            lbl_gp_in_progress.mousePressEvent = _gp_make_cycle(
                ('in_progress', 'partially_in_progress'),
                gp_listbox,
            )
            lbl_gp_failed.mousePressEvent = _gp_make_cycle(('failed', 'qa_failed'), gp_listbox)
            lbl_gp_merged.mousePressEvent = _gp_make_cycle(('merged',), gp_listbox)
            lbl_gp_remaining.mousePressEvent = _gp_make_cycle(('not_completed', 'not_translated', 'no_tts'), gp_listbox)
            lbl_gp_not_refined.mousePressEvent = _gp_make_cycle(('not_refined',), gp_listbox)
            lbl_gp_refine_failed.mousePressEvent = _gp_make_cycle(('refine_failed',), gp_listbox)
            
            p_layout.addWidget(gp_listbox)
            
            # Progress file path + open folder button + open glossary button
            path_row = QHBoxLayout()
            path_label = QLabel(f"📁 {gp_path}")
            _normal_path_label_style = "color: #666; font-size: 8pt;"
            _missing_path_label_style = "color: #f59e0b; font-size: 8pt;"
            path_label.setStyleSheet(_normal_path_label_style)
            path_label.setWordWrap(True)
            path_row.addWidget(path_label, stretch=1)

            select_all_btn = QPushButton("Select All")
            select_all_btn.setCursor(Qt.PointingHandCursor)
            select_all_btn.setStyleSheet(
                "QPushButton { background-color: #263445; color: #dbeafe; border: 1px solid #64748b; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #334155; }"
            )
            select_all_btn.setFixedHeight(22)

            def _select_all_gp_visible(_checked=False):
                panel_state['select_all_visible'] = True
                first_selected = None
                gp_listbox.blockSignals(True)
                try:
                    gp_listbox.clearSelection()
                    for row in range(gp_listbox.count()):
                        item = gp_listbox.item(row)
                        if not item or item.isHidden():
                            continue
                        item.setSelected(True)
                        if first_selected is None:
                            first_selected = row
                    if first_selected is not None:
                        gp_listbox.setCurrentRow(first_selected, QItemSelectionModel.Select)
                finally:
                    gp_listbox.blockSignals(False)
                gp_listbox.viewport().update()

            select_all_btn.clicked.connect(_select_all_gp_visible)
            path_row.addWidget(select_all_btn)

            refinement_btn = QPushButton("✨ Refinement")
            refinement_btn.setCursor(Qt.PointingHandCursor)
            refinement_btn.setStyleSheet(
                "QPushButton { background-color: #2d2645; color: #e9d5ff; border: 1px solid #8b5cf6; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #4c3575; } "
                "QPushButton:disabled { background-color: #24242a; color: #666; border-color: #444; }"
            )
            refinement_btn.setFixedHeight(22)
            refinement_btn.setToolTip(
                "Refine selected refinement rows. If none are selected, refine all active entry types."
            )

            def _run_gp_refinement(_checked=False):
                active_refinement_types = _active_glossary_refinement_types()
                selected_refinement_keys = [
                    str(item.data(Qt.UserRole + 3))
                    for item in gp_listbox.selectedItems()
                    if item.data(Qt.UserRole + 3)
                ]
                refinement_types = (
                    _normalize_glossary_refinement_selection(
                        selected_refinement_keys,
                        active_refinement_types,
                    )
                    if selected_refinement_keys
                    else active_refinement_types
                )
                refinement_progress = (
                    gp_data.get('refinement', {})
                    if isinstance(gp_data, dict)
                    else {}
                )
                completed_types = []
                if isinstance(refinement_progress, dict):
                    for refinement_key, refinement_info in refinement_progress.items():
                        if (
                            str(refinement_key).startswith('type::')
                            and isinstance(refinement_info, dict)
                            and str(refinement_info.get('status') or '').lower() == 'completed'
                        ):
                            completed_types.append(
                                str(
                                    refinement_info.get('entry_type')
                                    or str(refinement_key).split('::', 1)[1]
                                )
                            )
                _confirm_manual_glossary_refinement(
                    panel,
                    fp,
                    _find_gp_for_file(fp) or gp_path,
                    refinement_types,
                    completed_types=completed_types,
                )

            refinement_btn.clicked.connect(_run_gp_refinement)
            path_row.addWidget(refinement_btn)
            
            _gp_folder = os.path.dirname(gp_path)
            open_folder_btn = QPushButton("📂 Open Folder")
            open_folder_btn.setCursor(Qt.PointingHandCursor)
            open_folder_btn.setStyleSheet(
                "QPushButton { background-color: #3a3a3a; color: #d8f3dc; border: 1px solid #40916c; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #40916c; }"
            )
            open_folder_btn.setFixedHeight(22)
            def _open_gp_folder(_checked=False, folder=_gp_folder):
                import subprocess, sys
                if sys.platform == 'win32':
                    os.startfile(folder)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', folder])
                else:
                    subprocess.Popen(['xdg-open', folder])
            open_folder_btn.clicked.connect(_open_gp_folder)
            path_row.addWidget(open_folder_btn)
            
            # ── Open Glossary button ──
            open_glossary_btn = QPushButton("✏️ Open Glossary")
            open_glossary_btn.setCursor(Qt.PointingHandCursor)
            open_glossary_btn.setStyleSheet(
                "QPushButton { background-color: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; "
                "border-radius: 3px; padding: 2px 8px; font-size: 8pt; } "
                "QPushButton:hover { background-color: #1e40af; }"
            )
            open_glossary_btn.setFixedHeight(22)
            
            
            def _open_glossary_file(_checked=False):
                """Open the glossary file in the best available text editor."""
                import subprocess, shutil, sys
                
                glossary_path = _find_glossary_file()
                if not glossary_path or not os.path.isfile(glossary_path):
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        panel, "No Glossary Found",
                        f"No glossary file found in:\n{_gp_folder}\n\n"
                        "Expected: glossary.csv, glossary.json, or <book>_glossary.csv"
                    )
                    return
                
                try:
                    if sys.platform == 'win32':
                        _npp_paths = [
                            r'C:\Program Files\Notepad++\notepad++.exe',
                            r'C:\Program Files (x86)\Notepad++\notepad++.exe',
                        ]
                        _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                        if _npp:
                            subprocess.Popen([_npp, glossary_path])
                        else:
                            subprocess.Popen(['notepad.exe', glossary_path])
                    elif sys.platform == 'darwin':
                        if shutil.which('code'):
                            subprocess.Popen(['code', glossary_path])
                        else:
                            subprocess.Popen(['open', '-t', glossary_path])
                    else:
                        if shutil.which('gedit'):
                            subprocess.Popen(['gedit', glossary_path])
                        elif shutil.which('kate'):
                            subprocess.Popen(['kate', glossary_path])
                        elif shutil.which('code'):
                            subprocess.Popen(['code', glossary_path])
                        else:
                            _linux_editors = ['mousepad', 'xed', 'pluma', 'nano', 'xdg-open']
                            _editor = next((e for e in _linux_editors if shutil.which(e)), 'xdg-open')
                            subprocess.Popen([_editor, glossary_path])
                except Exception as _e:
                    print(f"⚠️ Could not open glossary editor: {_e}")
            
            open_glossary_btn.clicked.connect(_open_glossary_file)
            # Show tooltip with glossary path if found
            _initial_glossary = _find_glossary_file()
            if _initial_glossary:
                open_glossary_btn.setToolTip(f"Open in text editor:\n{_initial_glossary}")
            else:
                open_glossary_btn.setToolTip("No glossary file found yet")
            path_row.addWidget(open_glossary_btn)
            
            p_layout.addLayout(path_row)
            
            # Helper to fully rebuild the listbox when chapter_map changes
            def _rebuild_listbox(_d):
                _populate_gp_listbox(_d)

            def _apply_missing_progress_state():
                """Clear cached rows once when the progress file disappears."""
                nonlocal gp_data
                if panel_state.get('_progress_missing'):
                    return False

                _invalidate_gp_refresh()
                panel_state['_progress_missing'] = True
                gp_data = {}
                missing_progress_label.show()
                path_label.setText(f"📁 Progress file not found (waiting for recreation)\n{gp_path}")
                path_label.setStyleSheet(_missing_path_label_style)
                _refresh_stats_from_dict(gp_data)
                _populate_gp_listbox(gp_data)
                return True

            def _clear_missing_progress_state(current_path):
                if not panel_state.get('_progress_missing'):
                    return
                panel_state['_progress_missing'] = False
                missing_progress_label.hide()
                path_label.setText(f"📁 {current_path}")
                path_label.setStyleSheet(_normal_path_label_style)
            
            def _apply_pending_gp_result():
                """Main-thread: apply any result left by a background refresh."""
                nonlocal gp_data
                if not _gp_pending_result:
                    return False
                result = _gp_pending_result.pop()
                _gp_pending_result.clear()  # discard any older stale results
                if result.get('generation') != panel_state.get('_refresh_generation'):
                    return False
                result_signature = result.get('signature')
                if result_signature != _gp_file_signature(result.get('path')):
                    panel_state['_last_signature'] = None
                    # The extractor replaced the file again between the read
                    # and GUI application. Retry immediately instead of
                    # waiting for another timer tick; frequent atomic saves
                    # could otherwise starve the progress view indefinitely.
                    QTimer.singleShot(0, _refresh)
                    return False
                try:
                    _d = result['d']
                    gp_data = _d
                    panel_state['_last_signature'] = result_signature
                    _clear_missing_progress_state(result.get('path') or gp_path)
                    
                    if result.get('toggle_changed'):
                        panel_state['translate_special'] = result['cur_ts']
                        sr = result.get('spine_result')
                        if sr:
                            new_cmap, new_total, new_spine_idx = sr
                            if new_total > 0:
                                panel_state['chapter_map'] = new_cmap
                                panel_state['spine_index_map'] = new_spine_idx
                                panel_state['total'] = new_total
                            elif new_total == 0:
                                _all_idx = (
                                    result['comp']
                                    | result.get('skip', set())
                                    | result['fail']
                                    | result['merg']
                                )
                                panel_state['total'] = (max(_all_idx, default=0) + 1) if _all_idx else 1
                        _rebuild_reverse_lookups()
                        _rebuild_listbox(_d)
                        _refresh_stats_from_dict(_d)
                        return True
                    
                    _apply_gp_stats(result['legend_stats'])
                    
                    if result.get('bt') and bt_label:
                        bt_label.setText(f"📖 {result['bt']}")

                    if result.get('full_rebuild'):
                        panel_state['_full_rebuild_pending'] = False
                        _populate_gp_listbox(_d)
                        return True
                    
                    chapter_items = panel_state.get('_chapter_items') or {}
                    row_fingerprints = panel_state.setdefault('_row_fingerprints', {})

                    item_updates = result.get('item_updates', [])
                    if item_updates:
                        gp_listbox.setUpdatesEnabled(False)
                        try:
                            for ci, new_status, new_color, display_text, _issues in item_updates:
                                item = chapter_items.get(ci)
                                if item is None:
                                    continue
                                if display_text:
                                    item.setText(display_text)
                                item.setForeground(QColor(new_color))
                                item.setData(Qt.UserRole, new_status)
                                row_fingerprints[ci] = (new_status, new_color, display_text)
                        finally:
                            gp_listbox.setUpdatesEnabled(True)
                            gp_listbox.viewport().update()
                    _refresh_refinement_rows(_d)
                    _refresh_minimal_pass_row(_d)
                except Exception as e:
                    print(f"⚠️ Could not apply glossary progress refresh: {e}")
                finally:
                    _finish_gp_refresh_callbacks()
                return True

            def _deliver_gp_refresh_result(result):
                """Apply a worker snapshot immediately on Qt's GUI thread."""
                _ensure_gp_watch_paths(result.get('path'))
                if result.get('retry'):
                    panel_state['_last_signature'] = None
                    panel_state['_refresh_after_running'] = False
                    QTimer.singleShot(0, _refresh)
                    return
                _gp_pending_result.append(result)
                _apply_pending_gp_result()
                if panel_state.pop('_refresh_after_running', False):
                    QTimer.singleShot(0, _refresh)

            # A Python worker cannot safely update QListWidgetItems directly.
            # Deliver completed snapshots through a Qt signal so live progress
            # paints as soon as the disk read finishes, without depending on a
            # later polling-timer tick.
            def _gp_refresh_failed(message):
                print(f"Glossary progress refresh failed: {message}")
                panel_state['_gp_bg_running'] = False
                if panel_state.pop('_refresh_after_running', False):
                    QTimer.singleShot(0, _refresh)
                else:
                    _finish_gp_refresh_callbacks()

            gp_refresh_bridge = _GlossaryProgressAsyncBridge(
                on_finished=_deliver_gp_refresh_result,
                on_failed=_gp_refresh_failed,
                parent=panel,
            )
            panel._gp_refresh_bridge = gp_refresh_bridge

            def _emit_gp_refresh_finished(payload):
                try:
                    gp_refresh_bridge.finished.emit(payload)
                except RuntimeError:
                    pass

            def _emit_gp_refresh_failed(message):
                try:
                    gp_refresh_bridge.failed.emit(message)
                except RuntimeError:
                    pass

            panel_state['_gp_bg_running'] = False

            def _refresh(force=False, on_complete=None):
                try:
                    if callable(on_complete):
                        panel_state.setdefault('_refresh_callbacks', []).append(on_complete)
                    if force:
                        _invalidate_gp_refresh()
                        panel_state['_full_rebuild_pending'] = True
                    # First, apply any pending result from a previous background scan
                    _apply_pending_gp_result()

                    _rp = _find_gp_for_file(fp)
                    if not _rp or not os.path.isfile(_rp):
                        # Missing is a real state transition, not "nothing changed".
                        # Clear stale cached statuses and invalidate any worker that
                        # may still be carrying a snapshot of the deleted file.
                        _apply_missing_progress_state()
                        _finish_gp_refresh_callbacks()
                        return

                    # Skip if a background scan is still in flight. This check is
                    # deliberately after the missing-file check so deletion is
                    # detected even while an old disk read is finishing.
                    if panel_state.get('_gp_bg_running'):
                        if force or panel_state.get('_refresh_callbacks'):
                            panel_state['_refresh_after_running'] = True
                        return
                    
                    # Dirty-check: skip if file hasn't changed
                    try:
                        _cur_signature = _gp_file_signature(_rp)
                    except OSError:
                        _cur_signature = None
                    _cur_ts = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
                    if (_cur_signature == panel_state.get('_last_signature')
                            and _cur_ts == panel_state.get('translate_special')):
                        # The file is unchanged, but the Minimal row also
                        # depends on the live "Add minimal pass" toggle, which
                        # never touches the file. Fingerprint-gated: no-op
                        # unless the row's text actually changes.
                        try:
                            _cached_d = gp_data if isinstance(gp_data, dict) else {}
                            if _refresh_minimal_pass_row(_cached_d):
                                _refresh_stats_from_dict(_cached_d)
                        except Exception:
                            pass
                        _finish_gp_refresh_callbacks()
                        return  # Nothing changed — skip
                    panel_state['_gp_bg_running'] = True

                    # Snapshot values the background thread needs
                    _snap_total = panel_state['total']
                    _snap_cmap = dict(panel_state['chapter_map'])
                    _snap_ts = panel_state.get('translate_special')
                    _snap_generation = panel_state.get('_refresh_generation', 0)
                    _snap_fingerprints = dict(panel_state.get('_row_fingerprints') or {})
                    _snap_full_rebuild = bool(panel_state.get('_full_rebuild_pending', False))

                    def _bg_work():
                        try:
                            _before_signature = _gp_file_signature(_rp)
                            _d = _gp_load_progress_dict(_rp)
                            _after_signature = _gp_file_signature(_rp)
                            if _before_signature != _after_signature:
                                panel_state['_gp_bg_running'] = False
                                _emit_gp_refresh_finished({
                                    'retry': True,
                                    'path': _rp,
                                    'generation': _snap_generation,
                                })
                                return
                            _toggle_changed = (_cur_ts != _snap_ts)
                            _spine_result = None
                            if _toggle_changed:
                                _spine_result = _read_spine_map(fp, _cur_ts)

                            _cache = _gp_status_cache(_d)
                            _comp = _cache['completed']
                            _skip = _cache['skipped']
                            _fail = _cache['failed']
                            _merg = _cache['merged']
                            _prog = _cache['in_progress']
                            _ref_counts = _gp_extra_row_status_counts(_d)

                            _item_updates = []
                            for ci in range(_snap_total):
                                new_status, _issues = _gp_status_for(ci, _d, _cache)
                                new_color = _gp_color_for(new_status)
                                fname = _snap_cmap.get(ci, f'chapter {ci + 1}')
                                display_text, _ = _gp_display_for(ci, fname, _d, _cache)
                                fingerprint = (new_status, new_color, display_text)
                                if _snap_full_rebuild or _snap_fingerprints.get(ci) != fingerprint:
                                    _item_updates.append((ci, new_status, new_color, display_text, _issues))

                            _nr = max(0, _snap_total - len(_comp | _skip | _fail | _merg | _prog))
                            _legend_stats = _combine_glossary_progress_legend_stats(
                                {
                                    'total': _snap_total,
                                    'completed': len(_comp - _skip - _merg),
                                    'skipped': len(_skip),
                                    'in_progress': len(_prog),
                                    'failed': len(_fail),
                                    'merged': len(_merg),
                                    'remaining': _nr,
                                },
                                _ref_counts,
                            )
                            _emit_gp_refresh_finished({
                                'd': _d,
                                'path': _rp,
                                'signature': _after_signature,
                                'generation': _snap_generation,
                                'toggle_changed': _toggle_changed,
                                'spine_result': _spine_result,
                                'cur_ts': _cur_ts,
                                'comp': _comp, 'skip': _skip, 'fail': _fail, 'merg': _merg, 'prog': _prog,
                                'legend_stats': _legend_stats,
                                'nr': _nr, 'total': _snap_total,
                                'item_updates': _item_updates,
                                'bt': _d.get('book_title', ''),
                                'full_rebuild': _snap_full_rebuild,
                            })
                        except Exception as e:
                            _emit_gp_refresh_failed(str(e))
                        finally:
                            panel_state['_gp_bg_running'] = False

                    import threading
                    threading.Thread(target=_bg_work, name="gp-refresh-bg", daemon=True).start()
                except Exception as e:
                    panel_state['_gp_bg_running'] = False
                    print(f"⚠️ Could not schedule glossary progress refresh: {e}")
                    _finish_gp_refresh_callbacks()

            # Atomic progress saves replace the file, so watch both the file and
            # its directory. The short debounce coalesces temp-file/replace event
            # bursts into one background snapshot. A slow timer remains as a
            # fallback for platforms that occasionally drop filesystem events.
            gp_watcher = QFileSystemWatcher(panel)
            gp_watch_debounce = QTimer(panel)
            gp_watch_debounce.setSingleShot(True)
            gp_watch_debounce.setInterval(_PROGRESS_WATCH_DEBOUNCE_MS)

            def _ensure_gp_watch_paths(path=None):
                target = os.path.abspath(path or _find_gp_for_file(fp) or gp_path)
                watch_paths = []
                parent_dir = os.path.dirname(target)
                if parent_dir and os.path.isdir(parent_dir):
                    watch_paths.append(parent_dir)
                if os.path.isfile(target):
                    watch_paths.append(target)
                current = set(gp_watcher.files()) | set(gp_watcher.directories())
                missing = [watch_path for watch_path in watch_paths if watch_path not in current]
                if missing:
                    gp_watcher.addPaths(missing)

            def _refresh_watched_gp():
                _ensure_gp_watch_paths()
                if panel.isVisible():
                    _refresh()

            def _gp_watch_changed(_path):
                _ensure_gp_watch_paths()
                if panel.isVisible():
                    gp_watch_debounce.start()

            gp_watch_debounce.timeout.connect(_refresh_watched_gp)
            gp_watcher.fileChanged.connect(_gp_watch_changed)
            gp_watcher.directoryChanged.connect(_gp_watch_changed)
            _ensure_gp_watch_paths(gp_path)
            panel._gp_progress_watcher = gp_watcher
            panel._gp_progress_watch_debounce = gp_watch_debounce

            return panel, _refresh
        
        def _show_glossary_progress():
            """Show glossary extraction progress for all EPUBs (with or without progress files)."""
            try:
                # Reuse cached dialog if it still exists
                _cached = getattr(dialog, '_glossary_progress_dialog', None)
                if _cached is not None:
                    try:
                        _cached.show()
                        _cached.raise_()
                        _cached.activateWindow()
                        # Restart auto-refresh timer if it was stopped on hide
                        _gpt = getattr(_cached, '_gp_auto_refresh_timer', None)
                        if _gpt is not None and not _gpt.isActive():
                            _gpt.start()

                        def _refresh_cached_gp_panels():
                            refresh_visible = getattr(_cached, '_gp_refresh_visible', None)
                            if callable(refresh_visible):
                                refresh_visible()

                        QTimer.singleShot(0, _refresh_cached_gp_panels)
                        return
                    except RuntimeError:
                        # Widget was deleted
                        dialog._glossary_progress_dialog = None
                        dialog._gp_refresh_funcs = []
                
                # Gather all EPUB paths from multi-file dialog or just this file
                all_files = [file_path]
                if parent_dialog and hasattr(parent_dialog, '_epub_files_in_dialog'):
                    all_files = [f for f in parent_dialog._epub_files_in_dialog if str(f).lower().endswith('.epub')]
                if not all_files:
                    all_files = [file_path]
                
                # Build entries for ALL EPUBs — gp_path is None when no progress file exists
                all_file_entries = []
                for fp in all_files:
                    gp = _find_gp_for_file(fp)
                    if gp and os.path.isfile(gp):
                        all_file_entries.append((fp, gp))
                    else:
                        all_file_entries.append((fp, None))
                
                # Create dialog
                gp_dialog = QDialog(dialog)
                gp_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
                n_files = len(all_file_entries)
                if n_files == 1:
                    gp_dialog.setWindowTitle(f"Glossary Extraction Progress — {os.path.basename(all_file_entries[0][0])}")
                else:
                    gp_dialog.setWindowTitle(f"Glossary Extraction Progress — {n_files} files")
                gp_dialog.setWindowModality(Qt.NonModal)
                gp_title_text = gp_dialog.windowTitle()
                try:
                    gp_dialog.deleteLater()
                except Exception:
                    pass
                gp_dialog, gp_main_layout, loading_widget, loading_label = self._create_retranslation_shell_dialog(
                    gp_title_text,
                    width_ratio=0.39,
                    height_ratio=0.45,
                )
                gp_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
                gp_dialog.setWindowModality(Qt.NonModal)
                loading_label.setText("Loading glossary progress...")
                # Cache and show the shell before building the heavier panels.
                dialog._glossary_progress_dialog = gp_dialog
                dialog._gp_refresh_funcs = []
                gp_dialog.show()
                try:
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.AllEvents, 50)
                except Exception:
                    pass

                def _pump_gp_loading(msg=None):
                    try:
                        if msg and loading_label is not None:
                            loading_label.setText(msg)
                        advance = getattr(gp_dialog, '_advance_loading_icon', None)
                        if callable(advance):
                            advance()
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents(QEventLoop.AllEvents, 15)
                    except Exception:
                        pass

                gp_content = QWidget(gp_dialog)
                gp_content.hide()
                gp_content_layout = QVBoxLayout(gp_content)
                gp_content_layout.setContentsMargins(0, 0, 0, 0)
                gp_content_layout.setSpacing(6)
                gp_main_layout.addWidget(gp_content)
                
                # Title + note
                gp_title = QLabel("Glossary Extraction Progress")
                gp_title_font = QFont('Arial', 12)
                gp_title_font.setBold(True)
                gp_title.setFont(gp_title_font)
                gp_title.setStyleSheet("color: #52b788;")
                gp_content_layout.addWidget(gp_title)
                
                gp_note = QLabel("ℹ️ Tracks Balanced / Full auto glossary modes (Extract Glossary logic)")
                gp_note.setStyleSheet("color: #7a8a9e; font-size: 8pt; font-style: italic;")
                gp_note.setWordWrap(True)
                gp_content_layout.addWidget(gp_note)
                _pump_gp_loading()
                
                # Build panels and collect refresh functions
                all_refresh_funcs = []
                gp_navigation_widget = None
                
                def _build_gp_empty_panel(fp, parent_widget):
                    """Build a placeholder panel for an EPUB without glossary progress yet.
                    Returns (panel_widget, refresh_func) — refresh auto-upgrades to full panel."""
                    epub_base = os.path.splitext(os.path.basename(fp))[0]
                    
                    panel = QWidget(parent_widget)
                    p_layout = QVBoxLayout(panel)
                    p_layout.setContentsMargins(12, 20, 12, 20)
                    
                    empty_icon = QLabel("📊")
                    empty_icon.setAlignment(Qt.AlignCenter)
                    empty_icon.setStyleSheet("font-size: 36pt;")
                    p_layout.addWidget(empty_icon)

                    glossary_path = _find_glossary_for_refinement(fp)
                    try:
                        empty_glossary_entries = parse_glossary_file(glossary_path) if glossary_path else []
                    except Exception:
                        empty_glossary_entries = []
                    expected_refinement = _glossary_refinement_expected_entries(empty_glossary_entries)
                    empty_label = QLabel(f"No glossary extraction progress found for:\n{epub_base}")
                    empty_label.setAlignment(Qt.AlignCenter)
                    empty_label.setStyleSheet("color: #7a8a9e; font-size: 11pt;")
                    empty_label.setWordWrap(True)
                    p_layout.addWidget(empty_label)
                    
                    hint_text = "Run glossary extraction to see chapter progress. Refinement entry types are listed below."
                    hint_label = QLabel(hint_text)
                    hint_label.setAlignment(Qt.AlignCenter)
                    hint_label.setStyleSheet("color: #555; font-size: 9pt; font-style: italic;")
                    p_layout.addWidget(hint_label)

                    if expected_refinement:
                        ref_list = QListWidget(panel)
                        ref_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
                        ref_list.setContextMenuPolicy(Qt.CustomContextMenu)
                        ref_list.setSpacing(0)
                        ref_list.setUniformItemSizes(True)
                        ref_list.setStyleSheet("""
                            QListWidget {
                                background-color: #1f1f1f;
                                color: white;
                                border: 1px solid #4a5568;
                                border-radius: 4px;
                                padding: 4px;
                            }
                            QListWidget::item {
                                margin: 0px;
                                padding: 0px 4px;
                            }
                        """)
                        ref_list.setMaximumHeight(160)
                        for _ref_key, ref_info in expected_refinement.items():
                            entry_type = str(ref_info.get('entry_type') or _ref_key.replace('type::', '')).strip() or 'entry type'
                            ref_status = str(ref_info.get('status') or 'not_refined')
                            if ref_status == 'skipped':
                                display = f"Refinement | ⏭️ Skipped - No Entries | {entry_type}"
                                color = '#94a3b8'
                            else:
                                display = f"Refinement | \u2728 Not Refined          | {entry_type}"
                                color = '#5a9fd4'
                            item = QListWidgetItem(display)
                            item.setForeground(QColor(color))
                            item.setData(Qt.UserRole, ref_status)
                            item.setData(Qt.UserRole + 3, _ref_key)
                            self._add_compact_inline_list_item(ref_list, item)

                        def _empty_refinement_menu(pos):
                            clicked = ref_list.itemAt(pos)
                            if clicked is not None and not clicked.isSelected():
                                ref_list.clearSelection()
                                clicked.setSelected(True)
                            selected_items = [
                                item for item in ref_list.selectedItems()
                                if item.data(Qt.UserRole + 3)
                            ]
                            if not selected_items:
                                return
                            selected_keys = [str(item.data(Qt.UserRole + 3)) for item in selected_items]
                            aggregate_selected = any(key.startswith('all::') for key in selected_keys)
                            active_types = _active_glossary_refinement_types()
                            selected_types = _normalize_glossary_refinement_selection(
                                selected_keys,
                                active_types,
                            )
                            counts = Counter(
                                _refinement_type_key(entry.get('type'))
                                for entry in empty_glossary_entries
                                if isinstance(entry, dict)
                            )
                            eligible = sum(counts.get(_refinement_type_key(name), 0) for name in selected_types)
                            context_menu = QMenu(ref_list)
                            context_menu.setStyleSheet(
                                "QMenu { background-color: #2d2d2d; color: white; border: 1px solid #555; padding: 4px 8px 4px 4px; }"
                                "QMenu::item { padding: 6px 24px 6px 6px; }"
                                "QMenu::item:selected { background-color: #c0392b; }"
                                "QMenu::item:disabled { color: #6b7280; background-color: transparent; }"
                            )
                            if aggregate_selected or len(selected_items) == 1:
                                refine_action = context_menu.addAction('✨ Refine this')
                            else:
                                refine_action = context_menu.addAction(
                                    f'✨ Refine selected entry types ({len(selected_types)})'
                                )
                            refine_action.setEnabled(eligible > 0)
                            if context_menu.exec(ref_list.viewport().mapToGlobal(pos)) != refine_action:
                                return
                            target_progress = os.path.join(
                                os.path.dirname(glossary_path),
                                f'{epub_base}_glossary_progress.json',
                            ) if glossary_path else None
                            _confirm_manual_glossary_refinement(
                                panel,
                                fp,
                                target_progress,
                                selected_types,
                            )

                        ref_list.customContextMenuRequested.connect(_empty_refinement_menu)
                        p_layout.addWidget(ref_list)
                    
                    p_layout.addStretch()
                    
                    # State container for upgrade tracking
                    _state = {'upgraded': False}
                    
                    def _empty_refresh(force=False, on_complete=None):
                        """Check if a progress file appeared and upgrade the panel in-place."""
                        if _state['upgraded']:
                            if callable(on_complete):
                                on_complete()
                            return
                        gp = _find_gp_for_file(fp)
                        if gp and os.path.isfile(gp):
                            _state['upgraded'] = True
                            # Clear placeholder content
                            while p_layout.count():
                                item = p_layout.takeAt(0)
                                if item and item.widget():
                                    item.widget().deleteLater()
                            # Build the real panel content inside this existing panel
                            real_panel, real_refresh = _build_gp_panel(fp, gp, panel)
                            p_layout.addWidget(real_panel)
                            # Replace this refresh func in the parent list
                            try:
                                idx = all_refresh_funcs.index(_empty_refresh)
                                all_refresh_funcs[idx] = real_refresh
                            except ValueError:
                                all_refresh_funcs.append(real_refresh)
                            if force:
                                real_refresh(force=True, on_complete=on_complete)
                                return
                        if callable(on_complete):
                            on_complete()
                    
                    return panel, _empty_refresh
                
                if n_files == 1:
                    # Single file — no tabs needed
                    _pump_gp_loading("Building glossary panel...")
                    fp, gp = all_file_entries[0]
                    if gp:
                        panel, refresh_fn = _build_gp_panel(fp, gp, gp_dialog, pump_loading=_pump_gp_loading)
                    else:
                        panel, refresh_fn = _build_gp_empty_panel(fp, gp_dialog)
                    gp_content_layout.addWidget(panel)
                    all_refresh_funcs.append(refresh_fn)
                    _pump_gp_loading()
                
                elif n_files <= 3:
                    # Tabs for ≤3 files
                    _pump_gp_loading(f"Building glossary panels ({n_files} files)...")
                    notebook = QTabWidget()
                    notebook.setStyleSheet("""
                        QTabWidget::pane {
                            border: 2px solid #40916c;
                            border-radius: 4px;
                            background-color: #2d2d2d;
                        }
                        QTabBar::tab {
                            background-color: #3a3a3a;
                            color: white;
                            padding: 8px 16px;
                            margin-right: 2px;
                            border: 1px solid #40916c;
                            border-bottom: none;
                            border-top-left-radius: 4px;
                            border-top-right-radius: 4px;
                            font-size: 10pt;
                        }
                        QTabBar::tab:selected {
                            background-color: #2d6a4f;
                            color: #d8f3dc;
                            font-weight: bold;
                        }
                        QTabBar::tab:hover { background-color: #40916c; }
                    """)
                    for fp, gp in all_file_entries:
                        epub_base = os.path.splitext(os.path.basename(fp))[0]
                        if gp:
                            panel, refresh_fn = _build_gp_panel(fp, gp, notebook, pump_loading=_pump_gp_loading)
                        else:
                            panel, refresh_fn = _build_gp_empty_panel(fp, notebook)
                        notebook.addTab(panel, epub_base)
                        all_refresh_funcs.append(refresh_fn)
                        _pump_gp_loading()
                    gp_content_layout.addWidget(notebook)
                    gp_navigation_widget = notebook
                
                else:
                    # Dropdown navigation for >3 files
                    _pump_gp_loading(f"Building glossary panels ({n_files} files)...")
                    
                    nav_row = QHBoxLayout()
                    nav_row.setSpacing(6)
                    
                    nav_prev = QPushButton("◀")
                    nav_prev.setFixedWidth(36)
                    nav_prev.setStyleSheet(
                        "QPushButton { background-color:#3a3a3a; color:white; font-weight:bold; "
                        "font-size:13pt; border:1px solid #5a9fd4; border-radius:4px; padding:4px; }"
                        "QPushButton:hover { background-color:#4a8fc4; }"
                        "QPushButton:disabled { color:#666; background-color:#2a2a2a; }"
                    )
                    
                    combo = QComboBox()
                    combo.setStyleSheet(
                        "QComboBox { background-color:#3a3a3a; color:white; font-weight:bold; "
                        "font-size:11pt; padding:6px 10px; border:1px solid #5a9fd4; border-radius:4px; }"
                        "QComboBox::drop-down { border:none; }"
                        "QComboBox QAbstractItemView { background-color:#2d2d2d; color:white; "
                        "selection-background-color:#5a9fd4; }"
                    )
                    
                    nav_counter = QLabel("1 / 1")
                    nav_counter.setStyleSheet("color:#94a3b8; font-size:10pt; font-weight:bold;")
                    nav_counter.setFixedWidth(60)
                    nav_counter.setAlignment(Qt.AlignCenter)
                    
                    nav_next = QPushButton("▶")
                    nav_next.setFixedWidth(36)
                    nav_next.setStyleSheet(nav_prev.styleSheet())
                    
                    nav_row.addWidget(nav_prev)
                    nav_row.addWidget(combo, stretch=1)
                    nav_row.addWidget(nav_counter)
                    nav_row.addWidget(nav_next)
                    gp_content_layout.addLayout(nav_row)
                    
                    stack = QStackedWidget()
                    
                    for fp, gp in all_file_entries:
                        epub_base = os.path.splitext(os.path.basename(fp))[0]
                        if gp:
                            panel, refresh_fn = _build_gp_panel(fp, gp, stack, pump_loading=_pump_gp_loading)
                        else:
                            panel, refresh_fn = _build_gp_empty_panel(fp, stack)
                        stack.addWidget(panel)
                        combo.addItem(epub_base)
                        all_refresh_funcs.append(refresh_fn)
                        _pump_gp_loading()
                    
                    def _update_nav():
                        idx = combo.currentIndex()
                        n = combo.count()
                        nav_prev.setEnabled(idx > 0)
                        nav_next.setEnabled(idx < n - 1)
                        nav_counter.setText(f"{idx + 1} / {n}")
                        stack.setCurrentIndex(idx)
                    
                    combo.currentIndexChanged.connect(lambda _: _update_nav())
                    nav_prev.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() - 1))
                    nav_next.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() + 1))
                    _update_nav()
                    
                    gp_content_layout.addWidget(stack)
                    gp_navigation_widget = stack

                def _gp_active_refresh_index():
                    if gp_navigation_widget is None:
                        return 0
                    try:
                        return max(0, gp_navigation_widget.currentIndex())
                    except (AttributeError, RuntimeError):
                        return 0

                def _invoke_gp_refresher(refresh_func, force=False, on_complete=None):
                    refresh_func(force=force, on_complete=on_complete)

                def _gp_refresh_visible(force=False, on_complete=None):
                    if not all_refresh_funcs:
                        if callable(on_complete):
                            on_complete()
                        return
                    idx = min(_gp_active_refresh_index(), len(all_refresh_funcs) - 1)
                    _invoke_gp_refresher(
                        all_refresh_funcs[idx],
                        force=force,
                        on_complete=on_complete,
                    )

                if isinstance(gp_navigation_widget, QTabWidget):
                    gp_navigation_widget.currentChanged.connect(
                        lambda _idx: QTimer.singleShot(0, _gp_refresh_visible)
                    )
                elif isinstance(gp_navigation_widget, QStackedWidget):
                    gp_navigation_widget.currentChanged.connect(
                        lambda _idx: QTimer.singleShot(0, _gp_refresh_visible)
                    )

                # Explicit refresh bypasses signatures and rebuilds the
                # visible panel. Hidden books remain untouched so large
                # multi-book dialogs stay responsive.
                action_row = QHBoxLayout()
                action_row.addStretch()
                refresh_btn = AnimatedRefreshButton("  Refresh")
                refresh_btn.setCursor(Qt.PointingHandCursor)
                refresh_btn.setMinimumHeight(32)
                refresh_btn.setToolTip("Reload and rebuild the visible glossary progress panel from disk")
                refresh_btn.setStyleSheet(
                    "QPushButton { "
                    "background-color: #17a2b8; "
                    "color: white; "
                    "padding: 6px 16px; "
                    "font-weight: bold; "
                    "font-size: 10pt; "
                    "}"
                    "QPushButton[refreshActive=\"true\"] { "
                    "background-color: #138496; "
                    "}"
                )

                def _gp_full_refresh():
                    if not refresh_btn.isEnabled():
                        return
                    refresh_btn.start_animation()
                    refresh_btn.setEnabled(False)
                    start_time = time.time()
                    min_animation_duration = 0.8

                    def _finish_refresh_animation():
                        elapsed = time.time() - start_time
                        remaining = max(0, min_animation_duration - elapsed)

                        def _stop_animation():
                            try:
                                refresh_btn.stop_animation()
                                refresh_btn.setEnabled(True)
                            except RuntimeError:
                                pass

                        if remaining > 0:
                            QTimer.singleShot(int(remaining * 1000), _stop_animation)
                        else:
                            _stop_animation()

                    QTimer.singleShot(
                        50,
                        lambda: _gp_refresh_visible(
                            force=True,
                            on_complete=_finish_refresh_animation,
                        ),
                    )

                refresh_btn.clicked.connect(_gp_full_refresh)
                action_row.addWidget(refresh_btn)

                close_btn = QPushButton("Close")
                close_btn.setStyleSheet(
                    "QPushButton { background-color: #555; color: white; padding: 6px 20px; "
                    "border-radius: 4px; font-size: 10pt; } "
                    "QPushButton:hover { background-color: #666; }"
                )
                close_btn.clicked.connect(gp_dialog.hide)
                action_row.addWidget(close_btn)
                action_row.addStretch()
                gp_content_layout.addLayout(action_row)

                try:
                    loading_timer = getattr(gp_dialog, '_loading_icon_timer', None)
                    if loading_timer is not None:
                        loading_timer.stop()
                except Exception:
                    pass
                try:
                    gp_main_layout.removeWidget(loading_widget)
                    loading_widget.hide()
                    loading_widget.setParent(None)
                    loading_widget.deleteLater()
                except Exception:
                    pass
                gp_content.show()
                try:
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.AllEvents, 50)
                except Exception:
                    pass
                
                # Slow reliability fallback. Filesystem watcher events provide
                # the real-time path and normal ticks refresh only the active page.
                def _gp_refresh_all():
                    try:
                        if not gp_dialog.isVisible():
                            return
                        _gp_refresh_visible()
                    except Exception:
                        pass

                _gp_timer = QTimer(gp_dialog)
                _gp_timer.setInterval(2000)
                _gp_timer.timeout.connect(_gp_refresh_all)
                _gp_timer.start()
                gp_dialog._gp_auto_refresh_timer = _gp_timer
                gp_dialog._gp_refresh_visible = _gp_refresh_visible
                gp_dialog._gp_full_refresh = _gp_full_refresh

                # Stop auto-refresh timer when dialog is hidden to avoid
                # main-thread JSON parsing lag during translation
                _original_gp_hide_event = gp_dialog.hideEvent
                def _gp_hide_event(event):
                    try:
                        if _gp_timer.isActive():
                            _gp_timer.stop()
                    except Exception:
                        pass
                    _original_gp_hide_event(event)
                gp_dialog.hideEvent = _gp_hide_event

                # Restart the timer and refresh the visible panel whenever the
                # cached dialog becomes visible again. Without this the auto-
                # refresh stayed dead after the first hide (hideEvent stops it
                # and nothing restarted it), and re-shown panels were stale.
                _original_gp_show_event = gp_dialog.showEvent
                def _gp_show_event(event):
                    try:
                        _original_gp_show_event(event)
                    except Exception:
                        pass
                    try:
                        if not _gp_timer.isActive():
                            _gp_timer.start()
                        QTimer.singleShot(0, _gp_refresh_all)
                    except Exception:
                        pass
                gp_dialog.showEvent = _gp_show_event
                
                # Cache on parent dialog so all tabs share the same instance
                dialog._glossary_progress_dialog = gp_dialog
                dialog._gp_refresh_funcs = all_refresh_funcs
                
                gp_dialog.show()
            
            except Exception as e:
                print(f"⚠️ Error showing glossary progress: {e}")
                import traceback
                traceback.print_exc()
        
        dialog._show_glossary_progress = _show_glossary_progress
        glossary_progress_btn.clicked.connect(_show_glossary_progress)
        title_layout.addWidget(glossary_progress_btn)
        
        # Periodic check: show/hide button based on file existence (3s)
        # Uses single-pass caching to avoid redundant filesystem scans
        def _check_glossary_btn_visibility():
            try:
                # Skip if parent dialog is not visible (no point scanning filesystem)
                if hasattr(dialog, 'isVisible') and not dialog.isVisible():
                    return
                # In a multi-file dialog, only the active tab should run this
                # all-EPUB filesystem check.
                if tab_frame is not None and not container.isVisible():
                    return
                
                # Check all EPUBs from multi-file dialog, or just this file
                all_epubs = [file_path]
                if parent_dialog and hasattr(parent_dialog, '_epub_files_in_dialog'):
                    all_epubs = [f for f in parent_dialog._epub_files_in_dialog if str(f).lower().endswith('.epub')]
                if not all_epubs:
                    all_epubs = [file_path]
                
                is_multi = len(all_epubs) > 1
                # Single pass: resolve all paths once, cache results
                gp_results = {fp: _find_gp_for_file(fp) for fp in all_epubs}
                found_paths = {fp: gp for fp, gp in gp_results.items() if gp}
                any_exists = bool(found_paths)
                glossary_progress_btn.setVisible(True)
                if any_exists:
                    count = len(found_paths)
                    if count == 1:
                        gp = next(iter(found_paths.values()))
                        glossary_progress_btn.setToolTip(f"View glossary extraction progress\n{gp}")
                    else:
                        glossary_progress_btn.setToolTip(f"View glossary extraction progress ({count}/{len(all_epubs)} files)")
                elif is_multi:
                    glossary_progress_btn.setToolTip(f"View glossary extraction progress ({len(all_epubs)} files)")
                else:
                    glossary_progress_btn.setToolTip("View glossary extraction progress")
                _update_text_analysis_button()
            except RuntimeError:
                # Widget was deleted
                _gp_vis_timer.stop()
        
        _gp_vis_timer = QTimer()
        _gp_vis_timer.setInterval(3000)
        _gp_vis_timer.timeout.connect(_check_glossary_btn_visibility)
        _gp_vis_timer.start()
        # Parent timer to container so it dies with the dialog
        _gp_vis_timer.setParent(container)
        
        container_layout.addWidget(title_row)
        
        # Store reference to the listbox (will be created later)
        listbox_ref = [None]
        
        # Function to handle toggle change - will be defined after UI is created
        def on_toggle_special_files(state):
            """Filter the chapter list when the special files toggle is changed"""
            # Update the state variable
            show_special_files[0] = show_special_files_cb.isChecked()
            
            # Store the state persistently
            file_key = os.path.abspath(file_path)
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            if file_key not in self._retranslation_dialog_cache:
                self._retranslation_dialog_cache[file_key] = {}
            self._retranslation_dialog_cache[file_key]['show_special_files_state'] = show_special_files[0]
            
            # For tabs in multi-file dialog, sync toggle state across tabs
            if tab_frame and parent_dialog:
                # Update cache for all files in the current selection
                if hasattr(parent_dialog, '_epub_files_in_dialog'):
                    for f_path in parent_dialog._epub_files_in_dialog:
                        f_key = os.path.abspath(f_path)
                        if f_key not in self._retranslation_dialog_cache:
                            self._retranslation_dialog_cache[f_key] = {}
                        self._retranslation_dialog_cache[f_key]['show_special_files_state'] = show_special_files[0]
                
                # Sync ALL toggle checkboxes and checkmarks in ALL tabs
                if hasattr(parent_dialog, '_all_toggle_checkboxes'):
                    for idx, other_checkbox in enumerate(parent_dialog._all_toggle_checkboxes):
                        if other_checkbox is None or other_checkbox == show_special_files_cb:
                            continue
                        
                        try:
                            other_checkbox.isChecked()
                            other_checkbox.blockSignals(True)
                            other_checkbox.setChecked(show_special_files[0])
                            other_checkbox.blockSignals(False)
                            
                            if hasattr(parent_dialog, '_all_checkmark_labels') and idx < len(parent_dialog._all_checkmark_labels):
                                other_checkmark = parent_dialog._all_checkmark_labels[idx]
                                if other_checkmark is not None:
                                    try:
                                        other_checkmark.isVisible()
                                        if show_special_files[0]:
                                            other_checkmark.setGeometry(2, 1, 14, 14)
                                            other_checkmark.show()
                                        else:
                                            other_checkmark.hide()
                                    except RuntimeError:
                                        parent_dialog._all_checkmark_labels[idx] = None
                        except (RuntimeError, AttributeError):
                            parent_dialog._all_toggle_checkboxes[idx] = None
            
            # Filter list items instead of rebuilding entire UI
            if listbox_ref[0]:
                listbox = listbox_ref[0]
                for i in range(listbox.count()):
                    item = listbox.item(i)
                    if item:
                        # Check if this item is marked as special
                        item_data = item.data(Qt.UserRole)
                        if item_data and isinstance(item_data, dict):
                            # Dynamically re-evaluate is_special to respect current
                            # translate_all_numbered_html setting.
                            is_skipped_special = self._progress_entry_needs_special_visibility(
                                item_data.get('info') or item_data
                            )
                            # Show all items if toggled on; otherwise hide special-only rows.
                            item.setHidden(is_skipped_special and not show_special_files[0])
        
        # Connect the checkbox to the handler
        show_special_files_cb.stateChanged.connect(on_toggle_special_files)

        def on_toggle_model_info(state):
            show_model_info[0] = show_model_info_cb.isChecked()
            file_key = os.path.abspath(file_path)
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            if file_key not in self._retranslation_dialog_cache:
                self._retranslation_dialog_cache[file_key] = {}
            self._retranslation_dialog_cache[file_key]['show_model_info_state'] = show_model_info[0]
            self._persist_retranslation_show_model_info_state(show_model_info[0])

            if tab_frame and parent_dialog and hasattr(parent_dialog, '_epub_files_in_dialog'):
                for f_path in parent_dialog._epub_files_in_dialog:
                    f_key = os.path.abspath(f_path)
                    if f_key not in self._retranslation_dialog_cache:
                        self._retranslation_dialog_cache[f_key] = {}
                    self._retranslation_dialog_cache[f_key]['show_model_info_state'] = show_model_info[0]

            data = getattr(show_model_info_cb, '_progress_data_ref', None)
            if isinstance(data, dict):
                data['show_model_info_state'] = show_model_info[0]
                self._update_listbox_display(data)

        show_model_info_cb.stateChanged.connect(on_toggle_model_info)
        
        # Statistics - always show for both OPF and non-OPF files
        stats_frame = QWidget()
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 5, 0, 5)
        container_layout.addWidget(stats_frame)
        
        # Calculate stats from the appropriate source. Skipped special
        # files (the rows the "Show skipped files" toggle reveals) get
        # their own legend status and are excluded from the regular
        # status counts so they aren't double-counted as Not Translated.
        (total_chapters, chunk_count, completed, merged, in_progress, pending,
         missing, failed, skipped) = self._progress_initial_statistics(
            prog, chapter_display_info, spine_chapters
        )

        # Create labels (outside the if/else so they always appear)
        stats_font = QFont('Arial', 9)
        
        lbl_total = QLabel(_progress_total_label(total_chapters, chunk_count))
        lbl_total.setFont(stats_font)
        title_layout.insertWidget(1, lbl_total)
        
        lbl_completed = QLabel(f"✅ Completed: {completed} | ")
        lbl_completed.setFont(stats_font)
        lbl_completed.setStyleSheet("color: green;")
        lbl_completed.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_completed)
        
        # Merged: chapters combined into parent request (always create, hide if 0)
        lbl_merged = QLabel(f"🔗 Merged: {merged} | ")
        lbl_merged.setFont(stats_font)
        lbl_merged.setStyleSheet("color: #17a2b8;")  # Cyan/teal
        stats_layout.addWidget(lbl_merged)
        if merged == 0:
            lbl_merged.setVisible(False)
        
        # In Progress: currently being translated (always create, hide if 0)
        lbl_in_progress = QLabel(f"🔄 In Progress: {in_progress} | ")
        lbl_in_progress.setFont(stats_font)
        lbl_in_progress.setStyleSheet("color: orange;")
        lbl_in_progress.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_in_progress)
        if in_progress == 0:
            lbl_in_progress.setVisible(False)
        
        # Pending: marked for retranslation (always create, hide if 0)
        lbl_pending = QLabel(f"❓ Pending: {pending} | ")
        lbl_pending.setFont(stats_font)
        lbl_pending.setStyleSheet("color: white;")
        lbl_pending.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_pending)
        if pending == 0:
            lbl_pending.setVisible(False)
        
        # Not Translated: unique emoji/color (distinct from failures)
        _current_output_mode = self._current_progress_output_mode({'prog': prog})
        _missing_label_text, _failed_label_icon, _failed_label_text, _failed_label_color = (
            progress_stats_labels(_current_output_mode)
        )
        lbl_missing = QLabel(f"{_missing_label_text}: {missing} | ")
        lbl_missing.setFont(stats_font)
        lbl_missing.setStyleSheet("color: #2b6cb0;")
        lbl_missing.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_missing)
        
        # Match list status: failed/qa_failed use ❌ and red (clickable — jumps to next failure)
        lbl_failed = QLabel(f"{_failed_label_icon} {_failed_label_text}: {failed} | ")
        lbl_failed.setFont(stats_font)
        lbl_failed.setStyleSheet(f"color: {_failed_label_color};")
        lbl_failed.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_failed)

        # Skipped: special files excluded from translation — the rows the
        # "Show skipped files" toggle reveals. Hidden when none exist.
        lbl_skipped = QLabel(f"⏭️ Skipped: {skipped}")
        lbl_skipped.setFont(stats_font)
        lbl_skipped.setStyleSheet("color: #9aa0a6;")
        lbl_skipped.setCursor(Qt.PointingHandCursor)
        lbl_skipped.setToolTip(
            "Special files the translation pipeline skips\n"
            "(toggle visibility with “Show skipped files”).")
        stats_layout.addWidget(lbl_skipped)
        if skipped == 0:
            lbl_skipped.setVisible(False)
        
        
        stats_layout.addStretch()
        stats_layout.addWidget(manual_editing_cb)
        stats_layout.addWidget(text_analysis_btn)
        
        # Show temporary "folder created" label in the stats row if a folder was just created
        created_folder = getattr(self, '_pm_created_folder', None)
        if created_folder:
            display_name = os.path.basename(created_folder) or created_folder
            lbl_created = QLabel(f"📁 Created: {display_name}")
            lbl_created.setFont(stats_font)
            lbl_created.setStyleSheet("color: #27ae60; font-weight: bold;")
            stats_layout.addWidget(lbl_created)
            # Auto-hide after 2000ms
            QTimer.singleShot(2000, lbl_created.hide)
            # Clear the stored path so it doesn't re-appear on refresh
            self._pm_created_folder = None
        
        # Main frame for listbox
        main_frame = QWidget()
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(10 if not tab_frame else 5, 5, 10 if not tab_frame else 5, 5)
        container_layout.addWidget(main_frame)
        
        # Create listbox (QListWidget has built-in scrollbars)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        listbox.setUniformItemSizes(True)
        listbox_font = QFont('Courier', 10)  # Fixed-width font for better alignment
        self._apply_compact_inline_list_style(listbox, listbox_font, extra_row_px=2)
        # Use 36% of screen width
        min_width, _ = self._get_dialog_size(0.36, 0)
        listbox.setMinimumWidth(min_width)
        main_layout.addWidget(listbox)
        
        # Store listbox reference for toggle handler
        listbox_ref[0] = listbox
        
        # Helper: cycle to next item matching given statuses
        def _make_cycle_handler(statuses):
            def _handler(_event=None):
                lb = listbox_ref[0]
                if not lb:
                    return
                status_data = {'prog': prog}
                indices = []
                for i in range(lb.count()):
                    item = lb.item(i)
                    if not item or item.isHidden():
                        continue
                    display_status = item.data(Qt.UserRole + 2)
                    if not display_status:
                        payload = item.data(Qt.UserRole) or {}
                        display_status = self._progress_display_status(payload.get('info', {}), status_data)
                    if display_status in statuses:
                        indices.append(i)
                if not indices:
                    return
                selected_rows = [lb.row(item) for item in lb.selectedItems()]
                current = lb.currentRow()
                if selected_rows and current not in selected_rows:
                    current = max(selected_rows)
                nxt = next((i for i in indices if i > current), indices[0])
                lb.setCurrentRow(nxt, QItemSelectionModel.ClearAndSelect)
                lb.scrollToItem(lb.item(nxt), QListWidget.PositionAtCenter)
            return _handler

        lbl_completed.mousePressEvent   = _make_cycle_handler(('completed',))
        lbl_in_progress.mousePressEvent = _make_cycle_handler(('in_progress',))
        lbl_pending.mousePressEvent     = _make_cycle_handler(('pending',))
        lbl_missing.mousePressEvent     = _make_cycle_handler(('not_translated', 'not_refined', 'no_tts'))
        lbl_failed.mousePressEvent      = _make_cycle_handler(('failed', 'qa_failed', 'refine_failed'))
        lbl_skipped.mousePressEvent     = _make_cycle_handler(('skipped',))
        
        # Large progress lists are populated after result setup so the dialog can paint first.
        
        # Selection count label
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 10 if not tab_frame else 9)
        selection_count_label.setFont(selection_font)
        container_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)
        
        # Return data structure for external access
        result = {
            'file_path': file_path,
            'output_dir': output_dir,
            'progress_file': progress_file,
            'prog': prog,
            'progress_source_is_subtitle': progress_source_is_subtitle,
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': chapter_display_info,
            'listbox': listbox,
            'selection_count_label': selection_count_label,
            'dialog': dialog,
            'container': container,
            'fixed_output_dir': (
                os.path.abspath(str(resolved_output_dir))
                if resolved_output_dir
                else None
            ),
            'show_special_files_state': show_special_files[0],  # Store current toggle state
            'show_special_files_cb': show_special_files_cb,  # Store checkbox reference
            'show_model_info_state': show_model_info[0],
            'show_model_info_cb': show_model_info_cb,
            'manual_editing_cb': manual_editing_cb,
            'manual_editing_state': manual_editing_cb.isChecked(),
            'generate_manual_editing_sidecars': _generate_manual_editing_sidecars,
            'manual_untranslated_entries_provider': _progress_manager_untranslated_entries,
        }
        result['_last_applied_snapshot_signatures'] = (
            _progress_path_signature(progress_file),
            (len(_existing_output_files), hash(frozenset(_existing_output_files))),
            None,
        )
        show_model_info_cb._progress_data_ref = result
        manual_editing_cb._progress_data_ref = result
        
        # If standalone (no parent), add buttons and show dialog
        if not parent_dialog and not tab_frame:
            self._add_retranslation_buttons_opf(result)
            
            # Override close event to hide instead of destroy
            def closeEvent(event):
                event.ignore()  # Ignore the close event
                dialog.hide()   # Just hide the dialog
            
            dialog.closeEvent = closeEvent
            
            # Cache the dialog for reuse
            if not hasattr(self, '_retranslation_dialog_cache'):
                self._retranslation_dialog_cache = {}
            
            file_key = os.path.abspath(file_path)
            self._retranslation_dialog_cache[file_key] = result
            
            # Show the dialog (non-modal to allow interaction with other windows)
            dialog.show()
            QTimer.singleShot(50, lambda: self._populate_progress_listbox_streamed(result))
        elif not parent_dialog or tab_frame:
            # Embedded in tab - just add buttons
            self._add_retranslation_buttons_opf(result)
        
        return result


    def _add_retranslation_buttons_opf(self, data, button_frame=None):
        """Add the standard button set for retranslation dialogs with OPF support"""
        if any(
            _progress_item_is_html(info)
            for info in (data.get('chapter_display_info', []) or [])
        ):
            _schedule_epub_reader_engine_prewarm()
        
        if not button_frame:
            button_frame = QWidget()
            button_layout = QGridLayout(button_frame)
            # Get container layout and add button frame
            container = data['container']
            if hasattr(container, 'layout') and container.layout():
                container.layout().addWidget(button_frame)
        else:
            button_layout = button_frame.layout() if button_frame.layout() else QGridLayout(button_frame)
        
        # Helper functions that work with the data dict
        def select_all():
            data['listbox'].clearSelection()
            selected_count = 0
            for idx in range(data['listbox'].count()):
                item = data['listbox'].item(idx)
                if item and not item.isHidden():
                    item.setSelected(True)
                    selected_count += 1
            data['selection_count_label'].setText(f"Selected: {selected_count}")
        
        def clear_selection():
            data['listbox'].clearSelection()
            data['selection_count_label'].setText("Selected: 0")
        
        def select_status(status_to_select):
            data['listbox'].clearSelection()
            for idx in range(data['listbox'].count()):
                item = data['listbox'].item(idx)
                if not item or item.isHidden():
                    continue
                display_status = item.data(Qt.UserRole + 2)
                if not display_status:
                    payload = item.data(Qt.UserRole) or {}
                    display_status = self._progress_display_status(payload.get('info', {}), data)
                if status_to_select == 'failed':
                    matched = display_status in ['failed', 'qa_failed', 'refine_failed']
                elif status_to_select == 'qa_failed':
                    matched = display_status == 'qa_failed'
                else:
                    matched = display_status == status_to_select
                if matched:
                    item.setSelected(True)
            count = len(data['listbox'].selectedItems())
            data['selection_count_label'].setText(f"Selected: {count}")

        def _sdlxliff_sidecar_path_for_output_file(output_file):
            if not output_file:
                return None
            output_name = os.path.basename(str(output_file).replace("\\", "/"))
            if not output_name:
                return None
            return os.path.join(data['output_dir'], "SDLXLIFF", f"{output_name}.sdlxliff")


        def restore_in_progress_marks():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [
                data['chapter_display_info'][i] for i in selected_indices
            ]
            in_progress_chapters = [ch for ch in selected_chapters if ch.get('status') == 'in_progress']

            if not in_progress_chapters:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No In Progress Chapters",
                                     "None of the selected chapters have 'in_progress' status.")
                return

            result = restore_in_progress(
                data['progress_file'], data['output_dir'], in_progress_chapters
            )
            restored_count = result['restored']
            deleted_count = result['deleted']
            failed_count = result['failed']
            if result['progress_updated']:
                self._refresh_retranslation_data(data)

            message_parts = []
            if restored_count:
                message_parts.append(f"restored {restored_count}")
            if deleted_count:
                message_parts.append(f"removed {deleted_count} not-translated placeholder(s)")
            if failed_count:
                message_parts.append(f"marked {failed_count} as failed")
            message = "Successfully " + ", ".join(message_parts) + "." if message_parts else "No in-progress marks were changed."
            self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "In Progress Restored", message)
        
        def remove_pending_marks(selected_infos):
            try:
                result = _remove_pending_marks(
                    data['progress_file'], data['output_dir'], selected_infos
                )
            except (OSError, ValueError, TypeError) as error:
                self._styled_msgbox(
                    QMessageBox.Warning, data.get('dialog', self),
                    "Remove Pending Mark", f"Could not update progress: {error}",
                )
                return
            self._refresh_retranslation_data(data)
            self._styled_msgbox(
                QMessageBox.Information, data.get('dialog', self),
                "Remove Pending Mark",
                f"Restored {result['recovered']} pending entries. "
                f"Skipped {result['skipped']} ineligible selections. "
                "Existing QA findings were preserved.",
            )

        def remove_qa_failed_mark():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            # Skip dedup here to avoid merging distinct chapters that share filenames
            

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [
                data['chapter_display_info'][i] for i in selected_indices
            ]

            failed_chapters = plan_remove_qa_marks(data['prog'], selected_chapters)

            if not failed_chapters:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Failed Chapters",
                                     "None of the selected chapters have 'qa_failed' or 'failed' status.")
                return
            
            count = len(failed_chapters)
            reply = self._styled_msgbox(QMessageBox.Question, data.get('dialog', self), "Confirm Remove Failed Mark", 
                                      f"Remove failed mark from {count} chapters?",
                                      QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Remove marks
            result = remove_qa_marks(
                data['progress_file'], data['output_dir'], failed_chapters
            )
            cleared_count = result['cleared']
            
            # Auto-refresh the display
            self._refresh_retranslation_data(data)
            
            self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Success", f"Removed failed mark from {cleared_count} chapters.")

        def remove_refinement_status():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(
                    QMessageBox.Warning,
                    data.get('dialog', self),
                    "No Selection",
                    "Please select at least one chapter.",
                )
                return

            selected_indices = [
                data['listbox'].row(item) for item in selected_items
            ]
            selected_chapters = [
                data['chapter_display_info'][index]
                for index in selected_indices
            ]
            keys_with_refinement = refinement_status_keys(
                data.get('prog', {}), selected_chapters
            )

            if not keys_with_refinement:
                self._styled_msgbox(
                    QMessageBox.Information,
                    data.get('dialog', self),
                    "No Refinement Status",
                    "None of the selected chapters have refinement status.",
                )
                return

            reply = self._styled_msgbox(
                QMessageBox.Question,
                data.get('dialog', self),
                "Confirm Remove Refinement Status",
                (
                    "Remove refinement status from "
                    f"{len(keys_with_refinement)} chapter(s)?"
                ),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

            cleared_count = _progress_remove_refinement_status(
                data['progress_file'], keys_with_refinement
            )

            if cleared_count:
                self._refresh_retranslation_data(data)

            self._styled_msgbox(
                QMessageBox.Information,
                data.get('dialog', self),
                "Success",
                f"Removed refinement status from {cleared_count} chapter(s).",
            )
        
        def retranslate_selected():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            # Do NOT dedup here; it can collapse distinct chapters sharing filenames

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            # Selection normalisation, guards and the confirmation copy are the
            # shared plan (progress_actions.plan_retranslation).
            plan = plan_retranslation(
                data,
                selected_indices,
                {'manual_editing': _manual_editing_enabled()},
                owner=self,
            )
            if plan.mode == 'refused':
                refusal_kind, refusal_title, refusal_message = plan.refusal
                self._styled_msgbox(
                    QMessageBox.Warning if refusal_kind == 'warning' else QMessageBox.Information,
                    data.get('dialog', self),
                    refusal_title,
                    refusal_message,
                )
                return

            if plan.mode == 'reset_tts':
                selected_chapters = plan.selected_chapters
                reply = self._styled_msgbox(
                    QMessageBox.Question,
                    data.get('dialog', self),
                    plan.confirm_title,
                    plan.confirm_message,
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

                _tts_result = reset_tts(
                    self, data['progress_file'], data['output_dir'], selected_chapters
                )
                deleted_count = _tts_result.get('deleted', 0)
                status_reset_count = _tts_result.get('status_reset', 0)
                missing_audio_count = _tts_result.get('missing_audio', 0)
                if _tts_result.get('error'):
                    print(f"Failed to update progress file: {_tts_result['error']}")
                elif _tts_result.get('progress_updated'):
                    print(f"Updated progress tracking file - reset {status_reset_count} TTS statuses to no_tts")

                data['skip_cleanup'] = True
                self._refresh_retranslation_data(data)

                success_parts = []
                if deleted_count > 0:
                    success_parts.append(f"deleted {deleted_count} TTS file(s)")
                if status_reset_count > 0:
                    success_parts.append(f"marked {status_reset_count} chapter(s) as No TTS")
                if missing_audio_count > 0:
                    success_parts.append(f"{missing_audio_count} chapter(s) had no audio file on disk")
                message = "Successfully " + ", ".join(success_parts) + "." if success_parts else "No TTS changes made."
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "TTS Reset", message)
                return

            linked_choice = None
            if plan.needs_linked_choice:
                linked_choice = self._recycled_artifact_retranslation_choice(
                    data.get('dialog', self),
                    plan.confirm_message,
                    plan.counterpart_filename,
                )
                if linked_choice == "cancel":
                    return
            else:
                reply = self._styled_msgbox(
                    QMessageBox.Question,
                    data.get('dialog', self),
                    plan.confirm_title,
                    plan.confirm_message,
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            # Everything above this point reads Qt selection state and displays
            # confirmation UI.  The caller resumes the generator below on a
            # worker thread so bulk file resets and progress merging cannot
            # block the GUI event loop.
            yield "run_background"

            # Deletes, resets and the merge-write of the progress file
            # (progress_actions.apply_retranslation).
            result = apply_retranslation(data, plan, linked_choice, owner=self)
            merged_progress = result.merged_progress

            # File/progress work is complete. Resume the remaining UI-only
            # refresh and result dialogs on the Qt thread.
            yield "apply_ui"

            if isinstance(merged_progress, dict):
                data['prog'].clear()
                data['prog'].update(merged_progress)

            # Auto-refresh the display to show updated status
            data['skip_cleanup'] = True  # Disable cleanup for this dialog after retranslate to avoid deleting pending/failed
            data['_prefetch_dirty'] = True
            progress_debounce = data.get('_progress_watch_debounce')
            if progress_debounce is not None:
                progress_debounce.start()
            else:
                QTimer.singleShot(0, lambda: self._refresh_retranslation_data(data))

            result_kind, result_title, result_message = retranslation_result_message(result)
            self._styled_msgbox(
                QMessageBox.Warning if result_kind == 'warning' else QMessageBox.Information,
                data.get('dialog', self),
                result_title,
                result_message,
            )

        def _launch_retranslate_selected():
            """Confirm on Qt, execute the reset off-thread, then apply on Qt."""
            if data.get('_retranslate_selected_active'):
                return

            steps = retranslate_selected()
            try:
                marker = next(steps)
            except StopIteration:
                return
            except Exception as exc:
                self._styled_msgbox(
                    QMessageBox.Critical,
                    data.get('dialog', self),
                    "Retranslation Reset Failed",
                    str(exc),
                )
                return
            if marker != "run_background":
                return

            data['_retranslate_selected_active'] = True
            try:
                btn_retranslate.setEnabled(False)
                data['listbox'].setEnabled(False)
            except (RuntimeError, KeyError):
                pass

            def _finish_busy_state():
                data['_retranslate_selected_active'] = False
                try:
                    btn_retranslate.setEnabled(True)
                    data['listbox'].setEnabled(True)
                except (RuntimeError, KeyError):
                    pass
                if data.pop('_prefetch_dirty', False):
                    progress_debounce = data.get('_progress_watch_debounce')
                    if progress_debounce is not None:
                        progress_debounce.start()

            def _apply_retranslation_result(marker):
                try:
                    if marker != "apply_ui":
                        raise RuntimeError(
                            "Retranslation reset ended before its UI update."
                        )
                    _finish_busy_state()
                    try:
                        next(steps)
                    except StopIteration:
                        pass
                except Exception as exc:
                    _finish_busy_state()
                    self._styled_msgbox(
                        QMessageBox.Critical,
                        data.get('dialog', self),
                        "Retranslation Reset Failed",
                        str(exc),
                    )
                finally:
                    data.pop('_retranslate_selected_bridge', None)

            def _apply_retranslation_error(message):
                _finish_busy_state()
                data.pop('_retranslate_selected_bridge', None)
                self._styled_msgbox(
                    QMessageBox.Critical,
                    data.get('dialog', self),
                    "Retranslation Reset Failed",
                    message,
                )

            bridge = _GlossaryProgressAsyncBridge(
                on_finished=_apply_retranslation_result,
                on_failed=_apply_retranslation_error,
                parent=data.get('dialog') or self,
            )
            data['_retranslate_selected_bridge'] = bridge

            def _run_retranslation_reset():
                try:
                    bridge.finished.emit(next(steps))
                except StopIteration:
                    bridge.failed.emit(
                        "Retranslation reset ended before completion."
                    )
                except Exception as exc:
                    bridge.failed.emit(str(exc))

            threading.Thread(
                target=_run_retranslation_reset,
                name="progress-retranslate-selected",
                daemon=True,
            ).start()
        
        # Add buttons - First row
        btn_select_all = QPushButton("Select All")
        btn_select_all.setMinimumHeight(32)
        btn_select_all.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_all.clicked.connect(select_all)
        button_layout.addWidget(btn_select_all, 0, 0)
        
        btn_clear = QPushButton("Clear")
        btn_clear.setMinimumHeight(32)
        btn_clear.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_clear.clicked.connect(clear_selection)
        button_layout.addWidget(btn_clear, 0, 1)
        
        btn_select_completed = QPushButton("Select Completed")
        btn_select_completed.setMinimumHeight(32)
        btn_select_completed.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_completed.clicked.connect(lambda: select_status('completed'))
        button_layout.addWidget(btn_select_completed, 0, 2)
        
        btn_select_qa_failed = QPushButton("Select QA Failed")
        btn_select_qa_failed.setMinimumHeight(32)
        # Use red for QA Failed
        btn_select_qa_failed.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_qa_failed.clicked.connect(lambda: select_status('qa_failed'))
        button_layout.addWidget(btn_select_qa_failed, 0, 3)
        
        btn_select_failed = QPushButton("Select Failed")
        btn_select_failed.setMinimumHeight(32)
        # Use red for Failed / QA Failed
        btn_select_failed.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_select_failed.clicked.connect(lambda: select_status('failed'))
        button_layout.addWidget(btn_select_failed, 0, 4)
        
        # Second row
        btn_retranslate = QPushButton("Reset TTS Selected" if self._current_progress_output_mode(data) == 'audio' else "Retranslate Selected")
        btn_retranslate.setMinimumHeight(32)
        btn_retranslate.setStyleSheet("QPushButton { background-color: #d39e00; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_retranslate.clicked.connect(_launch_retranslate_selected)
        button_layout.addWidget(btn_retranslate, 1, 0, 1, 2)
        
        btn_remove_qa = QPushButton("Remove QA Failed Mark")
        btn_remove_qa.setMinimumHeight(32)
        btn_remove_qa.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_remove_qa.clicked.connect(remove_qa_failed_mark)
        button_layout.addWidget(btn_remove_qa, 1, 2, 1, 1)
        
        # Add animated refresh button
        btn_refresh = AnimatedRefreshButton("  Refresh")  # Double space for icon padding
        btn_refresh.setMinimumHeight(32)
        btn_refresh.setStyleSheet(
            "QPushButton { "
            "background-color: #17a2b8; "
            "color: white; "
            "padding: 6px 16px; "
            "font-weight: bold; "
            "font-size: 10pt; "
            "}"
            "QPushButton[refreshActive=\"true\"] { "
            "background-color: #138496; "
            "}"
        )
        
        # Create refresh handler with animation
        def animated_refresh():
            import time

            # Invalidate any older background snapshot so it cannot paint over
            # the explicit refresh after finishing.
            data['_prefetch_generation'] = int(data.get('_prefetch_generation', 0)) + 1
            btn_refresh.start_animation()
            btn_refresh.setEnabled(False)

            # Track start time for minimum animation duration
            start_time = time.time()
            min_animation_duration = 0.8  # 800ms minimum

            # A token to prevent older timers from firing after a newer refresh click
            refresh_token = time.time()
            data['_last_refresh_token'] = refresh_token

            def _rebuild_gui_from_refresh():
                """Recreate the retranslation GUI if refresh appears to have failed to render."""
                try:
                    dlg = data.get('dialog')

                    # Best-effort capture current toggle state
                    show_special = data.get('show_special_files_state', False)
                    cb = data.get('show_special_files_cb')
                    if cb:
                        try:
                            show_special = cb.isChecked()
                        except RuntimeError:
                            pass

                    # Multi-file dialog: destroy and recreate the whole multi-tab window
                    if dlg and hasattr(dlg, '_tab_data'):
                        selection = None
                        if hasattr(self, '_multi_file_selection_key') and self._multi_file_selection_key:
                            try:
                                selection = list(self._multi_file_selection_key)
                            except Exception:
                                selection = None

                        def do_multi_rebuild():
                            try:
                                # Clear cached multi-file dialog so the recreate path is taken
                                if hasattr(self, '_multi_file_retranslation_dialog'):
                                    self._multi_file_retranslation_dialog = None
                                if hasattr(self, '_multi_file_selection_key'):
                                    self._multi_file_selection_key = None

                                try:
                                    dlg.hide()
                                except Exception:
                                    pass
                                try:
                                    dlg.deleteLater()
                                except Exception:
                                    pass

                                if selection is not None:
                                    self.selected_files = selection
                                    self._force_retranslation_multiple_files()
                            except Exception as e:
                                print(f"Error during multi-file rebuild: {e}")

                        QTimer.singleShot(0, do_multi_rebuild)
                        return

                    # Single-file dialog: remove cached entry and recreate
                    file_path = data.get('file_path')
                    if not file_path:
                        return

                    file_key = os.path.abspath(file_path)
                    if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                        try:
                            del self._retranslation_dialog_cache[file_key]
                        except Exception:
                            pass

                    old_dlg = dlg

                    def do_single_rebuild():
                        try:
                            if old_dlg:
                                try:
                                    old_dlg.hide()
                                except Exception:
                                    pass
                                try:
                                    old_dlg.deleteLater()
                                except Exception:
                                    pass
                            self._show_retranslation_shell_then_build(
                                file_path,
                                show_special_files_state=show_special,
                                resolved_output_dir=data.get('fixed_output_dir'),
                            )
                        except Exception as e:
                            print(f"Error during rebuild: {e}")

                    QTimer.singleShot(0, do_single_rebuild)

                except Exception as e:
                    print(f"Error during rebuild: {e}")

            # Use QTimer to run refresh after animation starts
            def do_refresh():
                try:
                    # Always refresh only this tab's data (not all tabs)
                    self._refresh_retranslation_data(data)

                    # Schedule watchdog: if after 3 seconds there are still no visible entries,
                    # but our data says there should be, rebuild the GUI.
                    def watchdog_check():
                        try:
                            if data.get('_last_refresh_token') != refresh_token:
                                return  # superseded by a newer refresh

                            expected_total = len(data.get('chapter_display_info', []) or [])
                            if expected_total <= 0:
                                return

                            listbox = data.get('listbox')
                            if not listbox:
                                _rebuild_gui_from_refresh()
                                return

                            try:
                                count = listbox.count()
                            except RuntimeError:
                                _rebuild_gui_from_refresh()
                                return

                            visible = 0
                            try:
                                for i in range(count):
                                    item = listbox.item(i)
                                    if item is not None and not item.isHidden():
                                        visible += 1
                            except RuntimeError:
                                _rebuild_gui_from_refresh()
                                return

                            if visible > 0:
                                return

                            # Don't rebuild if everything is hidden purely due to the special-files filter.
                            try:
                                show_special = data.get('show_special_files_state', False)
                                cb = data.get('show_special_files_cb')
                                if cb:
                                    show_special = cb.isChecked()
                                if not show_special:
                                    infos = data.get('chapter_display_info', []) or []
                                    if infos and all(bool(info.get('is_special', False)) for info in infos):
                                        return
                            except Exception:
                                pass

                            _rebuild_gui_from_refresh()
                        except Exception as e:
                            print(f"Watchdog check error: {e}")

                    QTimer.singleShot(3000, watchdog_check)

                    # Calculate remaining time to meet minimum animation duration
                    elapsed = time.time() - start_time
                    remaining = max(0, min_animation_duration - elapsed)

                    # Schedule animation stop after remaining time
                    def finish_animation():
                        btn_refresh.stop_animation()
                        btn_refresh.setEnabled(True)

                    if remaining > 0:
                        QTimer.singleShot(int(remaining * 1000), finish_animation)
                    else:
                        finish_animation()

                except Exception as e:
                    print(f"Error during refresh: {e}")
                    btn_refresh.stop_animation()
                    btn_refresh.setEnabled(True)

            QTimer.singleShot(50, do_refresh)  # Small delay to let animation start
        
        btn_refresh.clicked.connect(animated_refresh)
        button_layout.addWidget(btn_refresh, 1, 3, 1, 1)

        # Expose refresh handler for external triggers (e.g., Progress Manager reopen)
        data['refresh_func'] = animated_refresh
        if data.get('dialog'):
            setattr(data['dialog'], '_refresh_func', animated_refresh)

        # Filesystem events trigger a debounced background snapshot. The worker
        # emits its immutable result back to Qt immediately, eliminating the old
        # extra timer tick while keeping every disk operation off the GUI thread.
        def _progress_page_is_visible():
            dlg = data.get('dialog')
            if not (dlg and dlg.isVisible()):
                return False
            page = data.get('container')
            return page is None or page is dlg or page.isVisible()

        def _clear_prefetched_progress_data():
            for key in (
                '_prefetched_prog',
                '_prefetched_prog_path',
                '_prefetched_output_listing',
                '_prefetched_tts_listing',
                '_prefetched_signatures',
            ):
                data.pop(key, None)

        def _apply_prefetched_progress_payload(payload):
            """Apply one already-prepared snapshot without GUI-thread rematching."""
            if not isinstance(payload, dict) or not _progress_page_is_visible():
                return

            prepared_spine = payload.get('prepared_spine_chapters')
            prepared_rows = payload.get('prepared_chapter_display_info')
            if prepared_spine is None or prepared_rows is None:
                data['_prefetched_prog'] = payload.get('prog')
                data['_prefetched_prog_path'] = payload.get('progress_file')
                data['_prefetched_output_listing'] = payload.get('listing', set())
                if payload.get('tts_listing') is not None:
                    data['_prefetched_tts_listing'] = payload.get('tts_listing')
                data['_prefetched_signatures'] = payload.get('signatures')
                try:
                    data['_refresh_read_only'] = True
                    self._refresh_retranslation_data(data)
                    data['_last_applied_snapshot_signatures'] = payload.get('signatures')
                finally:
                    data['_refresh_read_only'] = False
                    _clear_prefetched_progress_data()
                return

            prog = payload.get('prog')
            if not isinstance(prog, dict):
                return
            data['prog'] = prog
            data['_last_good_prog'] = prog
            data['spine_chapters'] = prepared_spine
            chapter_display_info = prepared_rows

            # These small auxiliary rows may consult live Qt settings, so only
            # the pure OPF/progress matching is performed in the worker.
            self._append_chunk_progress_display_info(data, chapter_display_info)
            self._append_metadata_display_info(data, chapter_display_info)
            self._append_translation_artifact_display_info(
                data,
                chapter_display_info,
            )
            self._append_pdf_ocr_display_info(data, chapter_display_info)
            self._append_image_gen_display_info(data, chapter_display_info)
            data['chapter_display_info'] = chapter_display_info
            self._update_listbox_display(data)
            self._update_statistics_display(data)
            self._progress_list_show_special(data)
            data['_last_applied_snapshot_signatures'] = payload.get('signatures')

        data['_apply_prefetched_progress_payload'] = (
            _apply_prefetched_progress_payload
        )

        def _queue_silent_refresh(delay_ms=0):
            """Keep at most one trailing live-refresh callback queued."""
            if data.get('_prefetch_scheduled'):
                return
            data['_prefetch_scheduled'] = True

            def _run_queued_refresh():
                data['_prefetch_scheduled'] = False
                _silent_refresh()

            QTimer.singleShot(max(0, int(delay_ms)), _run_queued_refresh)

        def _schedule_dirty_prefetch():
            if not data.pop('_prefetch_dirty', False) or not _progress_page_is_visible():
                return
            elapsed = time.monotonic() - float(
                data.get('_last_prefetch_started_at', 0.0) or 0.0
            )
            remaining = max(
                0.0,
                _PROGRESS_LIVE_REFRESH_MIN_INTERVAL_SECONDS - elapsed,
            )
            _queue_silent_refresh(round(remaining * 1000))

        def _on_prefetch_finished(payload):
            data['_prefetch_running'] = False
            if not isinstance(payload, dict):
                _schedule_dirty_prefetch()
                return
            if payload.get('generation') != data.get('_prefetch_generation'):
                _schedule_dirty_prefetch()
                return

            _ensure_translation_watch_paths(payload.get('progress_file'))
            if payload.get('retry'):
                data['_prefetch_dirty'] = True
                _schedule_dirty_prefetch()
                return
            if payload.get('unchanged') or not _progress_page_is_visible():
                _schedule_dirty_prefetch()
                return

            if data.get('_listbox_populate_active'):
                # A newer snapshot supersedes an older one. Never restart the
                # streamed 1,000+ row reconcile from row zero while it is still
                # running; apply only the newest snapshot after it finishes.
                data['_deferred_prefetched_progress_payload'] = payload
            else:
                _apply_prefetched_progress_payload(payload)
            _schedule_dirty_prefetch()

        def _on_prefetch_failed(message):
            data['_prefetch_running'] = False
            print(f"Progress Manager background refresh failed: {message}")
            _schedule_dirty_prefetch()

        prefetch_bridge = _GlossaryProgressAsyncBridge(
            on_finished=_on_prefetch_finished,
            on_failed=_on_prefetch_failed,
            parent=data.get('container') or data.get('dialog') or self,
        )
        data['_prefetch_bridge'] = prefetch_bridge

        def _emit_prefetch_finished(payload):
            try:
                prefetch_bridge.finished.emit(payload)
            except RuntimeError:
                pass

        def _emit_prefetch_failed(message):
            try:
                prefetch_bridge.failed.emit(message)
            except RuntimeError:
                pass

        def _silent_refresh():
            try:
                if not btn_refresh.isEnabled() or not _progress_page_is_visible():
                    return
                if data.get('_retranslate_selected_active'):
                    # The reset worker owns the mutable progress snapshot until
                    # it commits. Keep only one trailing refresh request.
                    data['_prefetch_dirty'] = True
                    return
                if data.get('_prefetch_running'):
                    return

                elapsed = time.monotonic() - float(
                    data.get('_last_prefetch_started_at', 0.0) or 0.0
                )
                remaining = (
                    _PROGRESS_LIVE_REFRESH_MIN_INTERVAL_SECONDS - elapsed
                )
                if remaining > 0:
                    _queue_silent_refresh(round(remaining * 1000))
                    return

                data['_prefetch_running'] = True
                data['_prefetch_dirty'] = False
                data['_last_prefetch_started_at'] = time.monotonic()
                generation = int(data.get('_prefetch_generation', 0)) + 1
                data['_prefetch_generation'] = generation
                progress_file = data.get('progress_file')
                output_dir = data.get('output_dir')
                last_signatures = data.get('_last_applied_snapshot_signatures')
                # This snapshot is read-only. Avoid cloning every chapter on
                # the GUI thread for every watcher event; copy only in the
                # worker if a failed JSON read actually needs the fallback.
                fallback_prog = data.get('prog') or {}
                audio_mode = self._current_progress_output_mode(data) == 'audio'
                refresh_file_path = str(data.get('file_path') or '')
                spine_snapshot = None
                if (
                    not audio_mode
                    and refresh_file_path.lower().endswith('.epub')
                    and data.get('spine_chapters')
                ):
                    # Copy only the small spine dictionaries on the GUI thread.
                    # All progress matching and row-model construction then
                    # happens in the worker instead of blocking Qt.
                    spine_snapshot = [
                        dict(chapter)
                        for chapter in data.get('spine_chapters', ())
                        if isinstance(chapter, dict)
                    ]

                def _bg_prefetch():
                    try:
                        prefetch_tts = audio_mode or any(
                            _progress_entry_has_meaningful_tts_state(entry)
                            for entry in fallback_prog.get('chapters', {}).values()
                        )
                        signatures, listing, tts_listing = _progress_snapshot_listing(
                            progress_file, output_dir, prefetch_tts
                        )
                        progress_signature = signatures[0]
                        payload = {
                            'generation': generation,
                            'progress_file': progress_file,
                            'signatures': signatures,
                        }
                        if signatures == last_signatures:
                            payload['unchanged'] = True
                            _emit_prefetch_finished(payload)
                            return

                        try:
                            with open(progress_file, 'r', encoding='utf-8') as progress_stream:
                                loaded = json.load(progress_stream)
                            prog = (
                                loaded
                                if isinstance(loaded, dict)
                                else copy.deepcopy(fallback_prog)
                            )
                        except Exception:
                            prog = copy.deepcopy(fallback_prog)

                        # If an atomic replacement landed during the read, discard
                        # this snapshot and immediately queue one more pass.
                        if _progress_path_signature(progress_file) != progress_signature:
                            payload['retry'] = True
                            _emit_prefetch_finished(payload)
                            return

                        if spine_snapshot is not None:
                            prepared_data = {
                                'prog': prog,
                                'output_dir': output_dir,
                                'file_path': refresh_file_path,
                                'spine_chapters': spine_snapshot,
                                '_prefetched_output_listing': listing,
                                '_refresh_read_only': True,
                            }
                            self._rematch_spine_chapters(
                                prepared_data,
                                append_auxiliary=False,
                            )
                            payload['prepared_spine_chapters'] = prepared_data[
                                'spine_chapters'
                            ]
                            payload['prepared_chapter_display_info'] = prepared_data[
                                'chapter_display_info'
                            ]

                        payload.update({
                            'prog': prog,
                            'listing': listing,
                            'tts_listing': tts_listing,
                        })
                        _emit_prefetch_finished(payload)
                    except Exception as exc:
                        _emit_prefetch_failed(str(exc))

                threading.Thread(
                    target=_bg_prefetch,
                    name="retrans-refresh-prefetch",
                    daemon=True,
                ).start()
            except Exception as exc:
                data['_prefetch_running'] = False
                print(f"Could not schedule Progress Manager refresh: {exc}")

        progress_watcher = QFileSystemWatcher(data.get('container') or data.get('dialog') or self)
        progress_watch_debounce = QTimer(data.get('container') or data.get('dialog') or self)
        progress_watch_debounce.setSingleShot(True)
        progress_watch_debounce.setInterval(_PROGRESS_WATCH_DEBOUNCE_MS)

        def _ensure_translation_watch_paths(progress_file=None):
            raw_target = progress_file or data.get('progress_file') or ''
            target = os.path.abspath(raw_target) if raw_target else ''
            raw_output_dir = data.get('output_dir') or (os.path.dirname(target) if target else '')
            output_dir = os.path.abspath(raw_output_dir) if raw_output_dir else ''
            watch_paths = []
            for directory in {output_dir, os.path.dirname(target)}:
                if directory and os.path.isdir(directory):
                    watch_paths.append(directory)
            if target and os.path.isfile(target):
                watch_paths.append(target)
            current = set(progress_watcher.files()) | set(progress_watcher.directories())
            missing = [watch_path for watch_path in watch_paths if watch_path not in current]
            if missing:
                progress_watcher.addPaths(missing)

        def _translation_progress_changed(_path):
            _ensure_translation_watch_paths()
            data['_prefetch_dirty'] = True
            if _progress_page_is_visible():
                progress_watch_debounce.start()

        progress_watch_debounce.timeout.connect(_silent_refresh)
        progress_watcher.fileChanged.connect(_translation_progress_changed)
        progress_watcher.directoryChanged.connect(_translation_progress_changed)
        _ensure_translation_watch_paths()
        data['_progress_watcher'] = progress_watcher
        data['_progress_watch_debounce'] = progress_watch_debounce

        # Slow polling is retained only as a fallback for dropped filesystem
        # notifications and watcher re-registration after atomic replacements.
        _auto_refresh_timer = QTimer(data.get('dialog') or self)
        _auto_refresh_timer.setInterval(2000)
        _auto_refresh_timer.timeout.connect(_silent_refresh)
        _auto_refresh_timer.start()
        data['_auto_refresh_timer'] = _auto_refresh_timer

        # Force-refresh whenever the (cached, hidden-on-close) Progress Manager
        # becomes visible again: kick a background snapshot immediately and
        # apply it as soon as it lands. Uses the same prefetch path, so no disk
        # I/O runs on the GUI thread.
        _pm_dialog = data.get('dialog')
        if _pm_dialog is not None:
            if not hasattr(_pm_dialog, '_progress_show_refreshers'):
                _pm_dialog._progress_show_refreshers = []
            _pm_dialog._progress_show_refreshers.append(_silent_refresh)

            # Install one show handler per dialog, not one nested wrapper per
            # EPUB tab.  The refreshers themselves ignore inactive pages.
            if not getattr(_pm_dialog, '_progress_show_handler_installed', False):
                _pm_dialog._progress_show_handler_installed = True
                _original_pm_show_event = _pm_dialog.showEvent

                def _run_visible_refreshers():
                    for refresher in list(getattr(_pm_dialog, '_progress_show_refreshers', ())):
                        try:
                            refresher()
                        except Exception:
                            pass

                def _pm_show_event(event, _orig=_original_pm_show_event):
                    try:
                        _orig(event)
                    except Exception:
                        pass
                    _run_visible_refreshers()

                _pm_dialog.showEvent = _pm_show_event

        # ==== Context menu on listbox ====
        listbox = data['listbox']
        listbox.setContextMenuPolicy(Qt.CustomContextMenu)

        def _exact_output_path_for_file(output_file):
            if not output_file:
                return None, None
            normalized = str(output_file).replace("\\", "/")
            path = normalized if os.path.isabs(normalized) else os.path.join(data['output_dir'], normalized)
            path = os.path.normpath(path)
            return path if os.path.isfile(path) else None, path

        def _exact_output_path_for_item(display_info):
            progress_entry = display_info.get('info', {}) or {}
            return _exact_output_path_for_file(display_info.get('output_file') or progress_entry.get('output_file'))

        def _source_candidates_for_item(display_info):
            progress_entry = display_info.get('info', {}) or {}
            raw_candidates = [
                display_info.get('original_filename'),
                display_info.get('original_basename'),
                display_info.get('key'),
                progress_entry.get('original_basename'),
                progress_entry.get('original_filename'),
                progress_entry.get('chapter_file'),
                progress_entry.get('source_filename'),
                progress_entry.get('filename'),
            ]
            candidates = []
            seen = set()
            for candidate in raw_candidates:
                if not candidate:
                    continue
                text = str(candidate).replace("\\", "/")
                variants = [text, os.path.basename(text)]
                stem, ext = os.path.splitext(text)
                if stem and not ext:
                    variants.extend([f"{text}.xhtml", f"{text}.html", f"{text}.htm"])
                    base = os.path.basename(text)
                    variants.extend([f"{base}.xhtml", f"{base}.html", f"{base}.htm"])
                for variant in variants:
                    if variant and variant not in seen:
                        seen.add(variant)
                        candidates.append(variant)
            return candidates

        def _source_path_for_item(display_info):
            for candidate in _source_candidates_for_item(display_info):
                text = str(candidate).replace("\\", "/")
                variants = [text]
                basename = os.path.basename(text)
                if basename and basename != text:
                    variants.append(basename)
                for variant in variants:
                    path = variant if os.path.isabs(variant) else os.path.join(data['output_dir'], variant)
                    path = os.path.normpath(path)
                    if os.path.isfile(path):
                        return path
            return None

        def _source_epub_candidates():
            candidates = []
            file_path = data.get('file_path')
            try:
                preferred = self._sdlxliff_preferred_input_epub(data['output_dir'], file_path)
                if preferred:
                    candidates.append(preferred)
                    self._sdlxliff_update_source_epub_ref(data['output_dir'], preferred)
            except Exception:
                if file_path:
                    candidates.append(file_path)
            source_ref = os.path.join(data['output_dir'], "source_epub.txt")
            try:
                if os.path.isfile(source_ref):
                    with open(source_ref, 'r', encoding='utf-8', errors='ignore') as f:
                        ref = f.read().strip()
                    if ref:
                        candidates.append(ref)
            except Exception:
                pass
            try:
                candidates.extend(self._sdlxliff_exact_input_epub_candidates(data['output_dir']))
            except Exception:
                pass
            try:
                for fname in os.listdir(data['output_dir']):
                    if str(fname).lower().endswith(".epub"):
                        candidates.append(os.path.join(data['output_dir'], fname))
            except Exception:
                pass
            seen = set()
            resolved = []
            for candidate in candidates:
                path = self._sdlxliff_valid_epub_path(data['output_dir'], candidate)
                if not path:
                    continue
                norm = os.path.normcase(os.path.abspath(path))
                if norm in seen:
                    continue
                seen.add(norm)
                resolved.append(path)
            return resolved

        def _source_exists_in_epub(display_info):
            candidates = _source_candidates_for_item(display_info)
            if not candidates:
                return False
            candidate_names = {str(c).replace("\\", "/").lower().strip("/") for c in candidates if c}
            candidate_basenames = {os.path.basename(str(c).replace("\\", "/")).lower() for c in candidates if c}
            candidate_names.discard("")
            candidate_basenames.discard("")
            if not candidate_names and not candidate_basenames:
                return False
            for epub_path in _source_epub_candidates():
                try:
                    with zipfile.ZipFile(epub_path, 'r') as zf:
                        for name in zf.namelist():
                            normalized = str(name).replace("\\", "/").lower().strip("/")
                            if normalized in candidate_names or os.path.basename(normalized) in candidate_basenames:
                                return True
                except Exception:
                    continue
            return False

        def _source_exists_for_item(display_info):
            return bool(_source_path_for_item(display_info) or _source_exists_in_epub(display_info))

        def _sdlxliff_review_path_for_item(display_info):
            progress_entry = display_info.get('info', {}) or {}
            path = _sdlxliff_sidecar_path_for_output_file(display_info.get('output_file') or progress_entry.get('output_file'))
            return path if os.path.isfile(path) else None

        def _sdlxliff_expected_review_path_for_item(display_info):
            progress_entry = display_info.get('info', {}) or {}
            return _sdlxliff_sidecar_path_for_output_file(display_info.get('output_file') or progress_entry.get('output_file'))

        def _manual_editing_enabled():
            checkbox = data.get('manual_editing_cb')
            try:
                return bool(checkbox and checkbox.isChecked())
            except RuntimeError:
                return False

        def _open_sdlxliff_review_for_item(display_info):
            def _finish_open(_stats=None):
                progress_entry = display_info.get('info', {}) or {}
                output_file = display_info.get('output_file') or progress_entry.get('output_file')
                review_path = _sdlxliff_review_path_for_item(display_info) or _sdlxliff_expected_review_path_for_item(display_info)
                try:
                    review_dialog = self._open_or_reuse_sdlxliff_review(
                        data['output_dir'],
                        review_path,
                        data.get('dialog', self),
                        autogen_file_path=data.get('file_path'),
                        autogen_progress_data=data.get('prog'),
                        autogen_output_files=[output_file] if output_file else None,
                        autogen_manual_entries=(
                            data.get('manual_untranslated_entries_provider')()
                            if callable(data.get('manual_untranslated_entries_provider'))
                            else []
                        ),
                    )
                    if review_dialog is None:
                        return
                    try:
                        review_dialog.save_status_label.setText("Checking SDLXLIFF sidecar for selected entry...")
                    except Exception:
                        pass
                except Exception as e:
                    self._show_message('error', "Open Failed", str(e), parent=data.get('dialog', self))

            if _manual_editing_enabled() and not _sdlxliff_review_path_for_item(display_info):
                generate_sidecars = data.get('generate_manual_editing_sidecars')
                if callable(generate_sidecars):
                    generate_sidecars(on_finished=_finish_open)
                    return
            _finish_open()

        def _open_file_for_item(display_info):
            """Open the output file for a chapter. Accepts pre-extracted display_info dict."""
            output_file = display_info.get('output_file')
            if not output_file:
                self._show_message('error', "File Missing", "No output file recorded for this entry.", parent=data.get('dialog', self))
                return
            path, missing_path = _exact_output_path_for_item(display_info)
            if not path:
                self._show_message('error', "File Missing", f"File not found:\n{missing_path}", parent=data.get('dialog', self))
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=data.get('dialog', self))

        def _open_epub_reader_for_item(display_info):
            """Open an HTML progress entry as an overlay in the EPUB reader."""
            output_path, missing_path = _exact_output_path_for_item(display_info)
            if not output_path:
                self._show_message(
                    'error',
                    "File Missing",
                    f"File not found:\n{missing_path}",
                    parent=data.get('dialog', self),
                )
                return
            if not _progress_item_is_html(display_info):
                return

            # PDF translations are ordered HTML workspaces too.  They do not
            # have (and should not need) a source EPUB zip; the integrated
            # reader consumes response_*.html directly and lazily extracts
            # only the selected raw bookmark range when Raw is requested.
            workspace_source = ""
            try:
                from output_workspace import read_workspace_source_path

                workspace_source = read_workspace_source_path(data['output_dir'])
            except Exception:
                workspace_source = ""
            if workspace_source and not os.path.isabs(workspace_source):
                workspace_source = os.path.abspath(os.path.join(
                    data['output_dir'], workspace_source
                ))
            if not workspace_source:
                candidate = str(data.get('file_path') or "")
                if candidate.lower().endswith(".pdf"):
                    workspace_source = candidate
            if (workspace_source.lower().endswith(".pdf")
                    and os.path.isfile(workspace_source)):
                parent_dialog = data.get('dialog') or self
                try:
                    from epub_library import EpubReaderDialog

                    progress_entry = display_info.get('info', {}) or {}
                    initial_filename = os.path.basename(str(
                        display_info.get('output_file')
                        or progress_entry.get('output_file')
                        or output_path
                    ))
                    reader = EpubReaderDialog(
                        workspace_source,
                        config=getattr(self, 'config', {}) or {},
                        parent=parent_dialog,
                        initial_chapter_filename=initial_filename,
                        workspace_dir=data['output_dir'],
                        initial_show_raw=False,
                        window_title=(
                            f"{os.path.basename(data['output_dir'])} (Translated)"
                        ),
                    )
                    reader.setModal(False)
                    reader.setAttribute(Qt.WA_DeleteOnClose)
                    active_readers = getattr(
                        parent_dialog,
                        '_progress_epub_readers',
                        None,
                    )
                    if not isinstance(active_readers, list):
                        active_readers = []
                        parent_dialog._progress_epub_readers = active_readers
                    active_readers.append(reader)

                    def _forget_pdf_reader(
                            *_args, _reader=reader, _readers=active_readers):
                        try:
                            _readers.remove(_reader)
                        except ValueError:
                            pass

                    reader.destroyed.connect(_forget_pdf_reader)
                    reader.show()
                except Exception as exc:
                    self._show_message(
                        'error',
                        "Open Failed",
                        str(exc),
                        parent=parent_dialog,
                    )
                return

            source_epubs = _source_epub_candidates()
            if not source_epubs:
                self._show_message(
                    'error',
                    "Source EPUB Missing",
                    "Could not resolve the source EPUB for this HTML entry.",
                    parent=data.get('dialog', self),
                )
                return
            epub_path = source_epubs[0]

            try:
                with zipfile.ZipFile(epub_path, 'r') as source_zip:
                    member_names = source_zip.namelist()
            except Exception as exc:
                self._show_message(
                    'error',
                    "Open Failed",
                    f"Could not read the source EPUB:\n{exc}",
                    parent=data.get('dialog', self),
                )
                return

            member_index = _index_epub_html_members(member_names)

            def _reader_member_for_item(item_info):
                progress_entry = item_info.get('info', {}) or {}
                candidates = _source_candidates_for_item(item_info)
                output_name = (
                    item_info.get('output_file')
                    or progress_entry.get('output_file')
                )
                if output_name:
                    candidates.append(output_name)
                return _match_epub_html_member_basename(
                    member_names,
                    candidates,
                    member_index=member_index,
                )

            initial_filename = _reader_member_for_item(display_info)
            if not initial_filename:
                self._show_message(
                    'error',
                    "Chapter Not Found",
                    "Could not match this HTML entry to a chapter in the source EPUB.",
                    parent=data.get('dialog', self),
                )
                return

            overlay = {}
            chapter_infos = list(data.get('chapter_display_info', []) or [])
            if display_info not in chapter_infos:
                chapter_infos.append(display_info)
            for chapter_info in chapter_infos:
                if not isinstance(chapter_info, dict) or not _progress_item_is_html(chapter_info):
                    continue
                translated_path, _ = _exact_output_path_for_item(chapter_info)
                if not translated_path:
                    continue
                source_filename = _reader_member_for_item(chapter_info)
                if source_filename:
                    overlay[source_filename.lower()] = {
                        "path": translated_path,
                    }
            overlay[initial_filename.lower()] = {"path": output_path}

            extra_image_dirs = [
                path for path in (
                    os.path.join(data['output_dir'], 'images'),
                    os.path.join(data['output_dir'], 'translated_images'),
                )
                if os.path.isdir(path)
            ]
            translated_css_dirs = []
            css_dir = os.path.join(data['output_dir'], 'css')
            if os.path.isdir(css_dir):
                translated_css_dirs.append(css_dir)
            try:
                if any(
                    entry.is_file() and entry.name.lower().endswith('.css')
                    for entry in os.scandir(data['output_dir'])
                ):
                    translated_css_dirs.append(data['output_dir'])
            except OSError:
                pass

            parent_dialog = data.get('dialog') or self
            try:
                from epub_library import EpubReaderDialog
                from reader_overlay import make_epub_overlay_provider

                # Readers outlive the visible Progress Manager page. Poll the
                # workspace directly so newly completed chapters appear even
                # while the manager's own row refresh is paused or hidden.
                overlay_provider = make_epub_overlay_provider(
                    data['output_dir'], member_names, initial_overlay=overlay,
                )
                refreshed_overlay = overlay_provider()
                if refreshed_overlay is not None:
                    overlay, extra_image_dirs = refreshed_overlay

                reader = EpubReaderDialog(
                    epub_path,
                    config=getattr(self, 'config', {}) or {},
                    parent=parent_dialog,
                    initial_chapter_filename=initial_filename,
                    translated_overlay=overlay,
                    overlay_provider=overlay_provider,
                    extra_image_dirs=extra_image_dirs or None,
                    translated_css_dirs=translated_css_dirs or None,
                    window_title=(
                        f"{os.path.splitext(os.path.basename(epub_path))[0]} "
                        "(Translated)"
                    ),
                )
                reader.setModal(False)
                reader.setAttribute(Qt.WA_DeleteOnClose)
                active_readers = getattr(
                    parent_dialog,
                    '_progress_epub_readers',
                    None,
                )
                if not isinstance(active_readers, list):
                    active_readers = []
                    parent_dialog._progress_epub_readers = active_readers
                active_readers.append(reader)

                def _forget_reader(*_args, _reader=reader, _readers=active_readers):
                    try:
                        _readers.remove(_reader)
                    except ValueError:
                        pass

                reader.destroyed.connect(_forget_reader)
                reader.show()
            except Exception as exc:
                self._show_message(
                    'error',
                    "Open Failed",
                    str(exc),
                    parent=parent_dialog,
                )

        def _find_audio_file_for_item(display_info):
            """Return the generated TTS file path associated with an HTML row, if one exists."""
            return find_row_audio(self, data, display_info)

        def _reset_tts_progress_for_output(output_file):
            return mutate_progress(
                data['progress_file'],
                lambda prog: _progress_reset_tts_for_output(prog, output_file),
            )

        def _open_audio_file_for_item(display_info):
            audio_path = _find_audio_file_for_item(display_info)
            if not audio_path:
                self._show_message('error', "Audio Missing", "No generated audio file was found for this HTML entry.", parent=data.get('dialog', self))
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(audio_path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=data.get('dialog', self))

        def _delete_audio_file_for_item(display_info):
            audio_path = _find_audio_file_for_item(display_info)
            if not audio_path:
                self._show_message('info', "Audio Missing", "No generated audio file was found for this HTML entry.", parent=data.get('dialog', self))
                return
            reply = self._styled_msgbox(
                QMessageBox.Question,
                data.get('dialog', self),
                "Delete Audio File",
                f"Delete this generated audio file?\n\n{audio_path}",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                self._show_message('error', "Delete Failed", str(e), parent=data.get('dialog', self))
                return
            _reset_tts_progress_for_output(display_info.get('output_file'))
            self._refresh_retranslation_data(data)
            self._show_message('info', "Audio Deleted", "Audio file deleted and TTS status reset to No TTS.", parent=data.get('dialog', self))

        def _show_llm_token_repair_comparison(summary, repairs, total_repaired):
            """Show the exact malformed and repaired tag previews side by side."""
            parent_dialog = data.get('dialog', self)
            dialog = QDialog(parent_dialog)
            dialog.setWindowTitle("QA Issue Resolved — Before / After")
            dialog.setModal(True)
            dialog.resize(1000, 560)

            layout = QVBoxLayout(dialog)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(summary_label)

            preview_count = len(repairs)
            preview_label = QLabel(
                f"Showing {preview_count} of {total_repaired} repaired tag(s)."
            )
            preview_label.setStyleSheet("color: #9aa0a6;")
            layout.addWidget(preview_label)

            comparison_layout = QHBoxLayout()
            before_layout = QVBoxLayout()
            after_layout = QVBoxLayout()
            before_layout.addWidget(QLabel("Before — malformed LLM token tag"))
            after_layout.addWidget(QLabel("After — safe visible text in HTML"))

            before_text = QPlainTextEdit()
            after_text = QPlainTextEdit()
            before_text.setReadOnly(True)
            after_text.setReadOnly(True)
            comparison_font = QFont("Consolas", 10)
            before_text.setFont(comparison_font)
            after_text.setFont(comparison_font)

            before_blocks = []
            after_blocks = []
            for index, repair in enumerate(repairs, 1):
                before_blocks.append(
                    f"[{index}]\n{str(repair.get('before') or '')}"
                )
                after_blocks.append(
                    f"[{index}]\n{str(repair.get('after') or '')}"
                )
            before_text.setPlainText("\n\n".join(before_blocks))
            after_text.setPlainText("\n\n".join(after_blocks))
            before_layout.addWidget(before_text)
            after_layout.addWidget(after_text)
            comparison_layout.addLayout(before_layout, 1)
            comparison_layout.addLayout(after_layout, 1)
            layout.addLayout(comparison_layout, 1)

            close_button = QPushButton("Close")
            close_button.setMinimumHeight(34)
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            dialog.exec()

        def _resolve_llm_token_qa_issue(display_info, output_path):
            """Repair one output file and clear only its LLM-token QA markers."""
            parent_dialog = data.get('dialog', self)
            outcome = resolve_llm_token_qa(
                data['progress_file'], display_info, output_path
            )
            result = outcome['repair']
            if not result.get('resolved'):
                message = result.get('error') or (
                    "The empty-attribute repair did not remove the LLM token issue."
                )
                try:
                    self.append_log(f"❌ LLM token QA issue was not resolved: {message}")
                except Exception:
                    pass
                self._show_message(
                    'error',
                    "QA Issue Not Resolved",
                    message,
                    parent=parent_dialog,
                )
                return

            remaining_other_qa = outcome['remaining_other_qa']
            if outcome['error']:
                exc = outcome['error']
                message = (
                    "The malformed tags were repaired, but the progress file "
                    f"could not be updated:\n{exc}"
                )
                try:
                    self.append_log(f"❌ LLM token QA progress update failed: {exc}")
                except Exception:
                    pass
                self._show_message(
                    'error',
                    "QA Issue Not Fully Resolved",
                    message,
                    parent=parent_dialog,
                )
                return

            self._refresh_retranslation_data(data)
            repaired_count = int(result.get('repaired') or 0)
            if repaired_count:
                summary = (
                    f"Repaired {repaired_count} empty-attribute LLM token "
                    f"tag(s) in {os.path.basename(output_path)}."
                )
            else:
                summary = (
                    "No malformed empty-attribute tags remain; the stale "
                    "LLM token QA mark was cleared."
                )
            if remaining_other_qa:
                summary += "\n\nOther QA issues remain on this entry."
            else:
                summary += "\n\nThe QA issue has been resolved."
            try:
                self.append_log(f"✅ {summary.replace(chr(10), ' ')}")
            except Exception:
                pass
            repairs = list(result.get('repairs') or [])
            if repaired_count and repairs:
                _show_llm_token_repair_comparison(
                    summary,
                    repairs,
                    repaired_count,
                )
            else:
                self._show_message(
                    'info',
                    "QA Issue Resolved",
                    summary,
                    parent=parent_dialog,
                )

        def show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            if not item.isSelected():
                listbox.clearSelection()
                item.setSelected(True)
                listbox.setCurrentItem(item)
            
            # IMPORTANT: Extract ALL data from the item BEFORE menu.exec() blocks.
            # The auto-refresh timer (2s) can rebuild the listbox and delete C++ objects
            # while the context menu is open, making the item reference stale.
            try:
                info_wrapper = item.data(Qt.UserRole)
                if not info_wrapper:
                    return
                display_info = info_wrapper.get('info', {})
                item_text = item.text()
            except RuntimeError:
                # C++ object already deleted
                return
            
            # The actual progress entry is nested inside 'info' key of display_info
            progress_entry = display_info.get('info', {})
            
            # qa_issues is a boolean flag; the actual list is qa_issues_found
            qa_issues = progress_entry.get('qa_issues_found', [])
            if not isinstance(qa_issues, list):
                qa_issues = []
                
            has_missing_images = _progress_entry_has_missing_image_qa(
                progress_entry
            )
            has_raw_foreign_text_qa = (
                _progress_entry_has_raw_foreign_text_qa(progress_entry)
            )
            has_llm_token_qa = _progress_entry_has_llm_token_qa(
                progress_entry
            )
            
            # Fallback: Check item text directly as it definitely contains the issue if visible
            if not has_missing_images and 'missing_images' in item_text:
                has_missing_images = True
                print("DEBUG: Detected missing_images via list item text")
            
            # Determine file path for Notepad action
            _output_file = display_info.get('output_file')
            qa_file_path, _missing_output_path = _exact_output_path_for_item(display_info)
            
            menu = QMenu(listbox)
            # Remove extra left gutter reserved for icons to avoid empty space
            menu.setStyleSheet(
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
            # Skipped special files get a single-purpose menu: the only
            # meaningful action for them is "Do not skip" (removes the
            # Other Settings keyword that caused the skip).
            act_do_not_skip = None
            _skip_keyword = None
            try:
                _skip_keyword = self._special_skip_keyword_for_progress_info(
                    display_info
                )
            except Exception:
                _skip_keyword = None

            act_open = act_review_sdlxliff = act_open_audio = None
            act_delete_audio = act_notepad_qa = act_retranslate = None
            act_resolve_qa = None
            act_insert_img = act_remove_qa = act_remove_refinement = None
            act_remove_pending = None
            act_restore_in_progress = None
            act_copy_qa = act_open_epub_reader = None
            selected_infos = []
            pending_selected_infos = []

            if _skip_keyword:
                act_do_not_skip = menu.addAction(
                    f"⏭️ Do not skip (remove keyword '{_skip_keyword}')")
            else:
                act_open = menu.addAction("📂 Open File")
                if (
                    (_manual_editing_enabled() and _progress_item_is_html(display_info))
                    or (_source_exists_for_item(display_info) and qa_file_path)
                ):
                    act_review_sdlxliff = menu.addAction("🔍 Edit Translation")
                if _find_audio_file_for_item(display_info):
                    act_open_audio = menu.addAction("🔊 Open Audio File")
                    act_delete_audio = menu.addAction("🗑️ Delete Audio File")
                if qa_file_path:
                    _label = "✏️ Edit File (find QA issue)" if qa_issues else "✏️ Edit File"
                    act_notepad_qa = menu.addAction(_label)
                if qa_issues:
                    act_copy_qa = menu.addAction("📋 Copy QA issue")
                if _progress_item_is_html(display_info):
                    act_open_epub_reader = menu.addAction(
                        "📖 Open in EPUB reader"
                    )
                act_retranslate = menu.addAction("🔁 Retranslate Selected")
                if has_raw_foreign_text_qa or has_llm_token_qa:
                    act_resolve_qa = menu.addAction("⚠️ Resolve QA issue")

                if has_missing_images:
                    act_insert_img = menu.addAction("🖼️ Insert Missing Image")

                act_remove_qa = menu.addAction("🧹 Remove QA Failed Mark")
                if _pending_mark_output_path(display_info, data['output_dir']):
                    act_remove_pending = menu.addAction("🧽 Remove Pending Mark")
                act_remove_refinement = menu.addAction(
                    "⭐ Remove refinement status"
                )

                try:
                    for selected_item in listbox.selectedItems():
                        wrapper = selected_item.data(Qt.UserRole) or {}
                        selected_infos.append(wrapper.get('info', {}))
                except RuntimeError:
                    selected_infos = [display_info]
                pending_selected_infos = [
                    {key: selected.get(key) for key in (
                        'progress_key', 'key', 'output_file', 'is_chunk_progress',
                        'chunk_progress_key', 'parent_progress_key', 'chunk_index',
                    )}
                    for selected in selected_infos
                ]
                if any((info or {}).get('status') == 'in_progress' for info in selected_infos):
                    act_restore_in_progress = menu.addAction("Restore In Progress Status")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen is None:
                # Menu dismissed — must bail before the equality checks,
                # otherwise `None == <unset action var>` would match.
                return
            if chosen == act_open:
                _open_file_for_item(display_info)
            elif act_review_sdlxliff and chosen == act_review_sdlxliff:
                _open_sdlxliff_review_for_item(display_info)
            elif act_open_audio and chosen == act_open_audio:
                _open_audio_file_for_item(display_info)
            elif act_delete_audio and chosen == act_delete_audio:
                _delete_audio_file_for_item(display_info)
            elif act_copy_qa and chosen == act_copy_qa:
                try:
                    from PySide6.QtWidgets import QApplication
                    lines = []
                    for sel in listbox.selectedItems():
                        try:
                            w = sel.data(Qt.UserRole) or {}
                            di = w.get('info', {})
                            pe = di.get('info', {})
                            issues_list = pe.get('qa_issues_found', [])
                            if not isinstance(issues_list, list):
                                issues_list = []
                            issues_list = [str(i) for i in issues_list if str(i).strip()]
                            if not issues_list:
                                continue
                            fname = (di.get('output_file') or di.get('original_filename')
                                     or di.get('key') or '')
                            if fname:
                                lines.append(f"{fname}: " + ", ".join(issues_list))
                            else:
                                lines.append(", ".join(issues_list))
                        except RuntimeError:
                            continue
                    if not lines and qa_issues:
                        # Fall back to the right-clicked item's issues
                        lines = [", ".join(str(i) for i in qa_issues)]
                    QApplication.clipboard().setText("\n".join(lines))
                except Exception as ex:
                    print(f"Copy QA issue failed: {ex}")
            elif act_open_epub_reader and chosen == act_open_epub_reader:
                _open_epub_reader_for_item(display_info)
            elif chosen == act_retranslate:
                _launch_retranslate_selected()
            elif act_resolve_qa and chosen == act_resolve_qa:
                if has_llm_token_qa:
                    _resolve_llm_token_qa_issue(
                        display_info,
                        qa_file_path,
                    )
                else:
                    self._start_single_progress_qa_resolution(
                        data, display_info
                    )
            elif act_insert_img and chosen == act_insert_img:
                # IN-PLACE RESTORATION LOGIC using ContentProcessor
                kind, title, message, refreshed = insert_missing_images(
                    data, display_info
                )
                if refreshed:
                    self._refresh_retranslation_data(data)
                self._show_message(kind, title, message)
            elif act_restore_in_progress and chosen == act_restore_in_progress:
                restore_in_progress_marks()
            elif act_do_not_skip and chosen == act_do_not_skip:
                if self._remove_special_skip_keyword(_skip_keyword):
                    try:
                        self._update_listbox_display(data)
                    except Exception:
                        pass
                    try:
                        self._update_statistics_display(data)
                    except Exception:
                        pass
                    self._show_message(
                        'info', "Do not skip",
                        f"Removed special-file keyword '{_skip_keyword}' "
                        "from Other Settings.\nMatching files will no longer "
                        "be skipped during translation.",
                        parent=data.get('dialog', self))
            elif act_remove_pending and chosen == act_remove_pending:
                remove_pending_marks(pending_selected_infos)
            elif chosen == act_remove_qa:
                remove_qa_failed_mark()
            elif chosen == act_remove_refinement:
                remove_refinement_status()
            elif act_notepad_qa and chosen == act_notepad_qa:
                if not qa_file_path or not os.path.isfile(qa_file_path):
                    self._show_message(
                        'error',
                        "File Missing",
                        f"File not found:\n{_missing_output_path}",
                        parent=data.get('dialog', self)
                    )
                    return
                search_term = None
                _line_num = 1
                if qa_issues:
                    # Extract a meaningful search term from the QA issue strings
                    # Try all common delimiter styles in order
                    _QUOTE_PATTERNS = [
                        r"'([^']+)'",                    # single quotes: 'text'
                        r'"([^"]+)"',                   # double quotes: "text"
                        r"\u201c([^\u201d]+)\u201d",    # curly double quotes: “text”
                        r"\u2018([^\u2019]+)\u2019",    # curly single quotes: ‘text’
                        r"\u300c([^\u300d]+)\u300d",    # Japanese corner brackets: 「text」
                        r"\u300e([^\u300f]+)\u300f",    # Japanese white corner brackets: 『text』
                        r"\uff62([^\uff63]+)\uff63",    # Halfwidth corner brackets
                        r"\[([^\]]+)\]",              # square brackets: [text]
                        r"\(([^)]+)\)",               # parentheses: (text)
                    ]
                    for _issue in qa_issues:
                        _s = str(_issue)
                        for _pat in _QUOTE_PATTERNS:
                            _m = re.search(_pat, _s)
                            if _m and _m.group(1).strip():
                                search_term = _m.group(1)
                                break
                        if search_term:
                            break
                    # Fallback: scan file for any non-ASCII sequence
                    if not search_term:
                        try:
                            with open(qa_file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                                _content = _f.read()
                            _m = re.search(r'[^\x00-\x7f]{1,30}', _content)
                            if _m:
                                search_term = _m.group(0)
                        except Exception:
                            pass
                    # Find line number of search term in file
                    # Try progressively shorter prefixes in case the QA term is truncated
                    if search_term and os.path.exists(qa_file_path):
                        try:
                            with open(qa_file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                                _lines = _f.readlines()
                            # Strip surrounding quote/bracket chars so we search raw content
                            _STRIP_QUOTES = '\'"「」『』“”‘’｢｣《》〈〉（）'
                            _bare = search_term.strip(_STRIP_QUOTES)
                            _base = _bare if _bare else search_term
                            # Build candidates: full bare term, then shrinking prefixes (min 1 char)
                            _candidates = [_base[:_l] for _l in range(len(_base), 0, -1)]
                            for _cand in _candidates:
                                for _i, _ln in enumerate(_lines, 1):
                                    if _cand in _ln:
                                        _line_num = _i
                                        break
                                if _line_num > 1:
                                    break
                        except Exception:
                            pass
                    # Copy search term to clipboard
                    if search_term:
                        from PySide6.QtWidgets import QApplication
                        QApplication.clipboard().setText(search_term)
                # Open file in best available editor, jumping to line if supported
                try:
                    if sys.platform == 'win32':
                        _npp_paths = [
                            r'C:\Program Files\Notepad++\notepad++.exe',
                            r'C:\Program Files (x86)\Notepad++\notepad++.exe',
                        ]
                        _npp = next((p for p in _npp_paths if os.path.exists(p)), None)
                        if _npp:
                            subprocess.Popen([_npp, f'-n{_line_num}', qa_file_path])
                        else:
                            subprocess.Popen(['notepad.exe', qa_file_path])
                    elif sys.platform == 'darwin':
                        # Try TextEdit alternatives that support line jumping
                        if shutil.which('code'):
                            subprocess.Popen(['code', '--goto', f'{qa_file_path}:{_line_num}'])
                        else:
                            subprocess.Popen(['open', '-t', qa_file_path])
                    else:
                        # Linux: try editors with line-jump support first
                        if shutil.which('gedit'):
                            subprocess.Popen(['gedit', f'+{_line_num}', qa_file_path])
                        elif shutil.which('kate'):
                            subprocess.Popen(['kate', '-l', str(_line_num), qa_file_path])
                        elif shutil.which('code'):
                            subprocess.Popen(['code', '--goto', f'{qa_file_path}:{_line_num}'])
                        else:
                            _linux_editors = ['mousepad', 'xed', 'pluma', 'nano', 'xdg-open']
                            _editor = next((e for e in _linux_editors if shutil.which(e)), 'xdg-open')
                            subprocess.Popen([_editor, qa_file_path])
                except Exception as _e:
                    self._show_message('error', "Open Failed", f"Could not open editor:\n{_e}",
                                       parent=data.get('dialog', self))

        listbox.customContextMenuRequested.connect(show_context_menu)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setMinimumHeight(32)
        btn_cancel.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 6px 16px; font-weight: bold; font-size: 10pt; }")
        btn_cancel.clicked.connect(lambda: data['dialog'].close() if data.get('dialog') else None)
        button_layout.addWidget(btn_cancel, 1, 4, 1, 1)

        # The builder has just read and matched this progress snapshot, so an
        # immediate second full refresh only duplicates large-EPUB work.  The
        # silent change-detecting timer handles subsequent updates.

    def _refresh_all_tabs(self, tab_data_list):
        """Refresh all tabs in a multi-file retranslation dialog"""
        try:
            print(f"🔄 Refreshing all {len(tab_data_list)} tabs...")
            
            refreshed_count = 0
            skipped_count = 0
            for idx, data in enumerate(tab_data_list):
                if data and data.get('type') != 'image_folder' and data.get('type') != 'individual_images':
                    # Only refresh EPUB/text tabs
                    try:
                        # Check if widgets are still valid before attempting refresh
                        if not self._is_data_valid(data):
                            print(f"[DEBUG] Skipping tab {idx + 1}/{len(tab_data_list)} - widgets deleted")
                            skipped_count += 1
                            continue
                        
                        print(f"[DEBUG] Refreshing tab {idx + 1}/{len(tab_data_list)}")
                        self._refresh_retranslation_data(data)
                        refreshed_count += 1
                    except RuntimeError as e:
                        # Widget was deleted
                        print(f"[WARN] Skipping tab {idx + 1} - widget deleted: {e}")
                        skipped_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to refresh tab {idx + 1}: {e}")
            
            if skipped_count > 0:
                print(f"✅ Successfully refreshed {refreshed_count} tab(s), skipped {skipped_count} deleted tab(s)")
            else:
                print(f"✅ Successfully refreshed {refreshed_count} tab(s)")
            
        except Exception as e:
            print(f"❌ Failed to refresh all tabs: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_data_valid(self, data):
        """Check if the data structure has valid (non-deleted) widgets"""
        try:
            if not data:
                return False
            
            # Check if listbox exists and is still valid
            listbox = data.get('listbox')
            if not listbox:
                return False
            
            # Try to access a simple property to check if widget is still alive
            # This will raise RuntimeError if the C++ object was deleted
            listbox.count()
            return True
            
        except (RuntimeError, AttributeError):
            return False
    
    def _refresh_retranslation_data(self, data):
        """Refresh the retranslation dialog data by reloading progress and updating display"""
        updates_were_enabled = True
        signals_were_blocked = False
        try:
            # First check if widgets are still valid
            if not self._is_data_valid(data):
                print("⚠️ Cannot refresh - widgets have been deleted")
                return

            # If the output override directory changed while the dialog is open,
            # re-resolve output_dir/progress_file so we don't keep reading the old progress JSON.
            self._resolve_progress_view_output_dir(data)
            
            # Freeze painting while the current snapshot is reconciled. Preserve
            # one scrollbar value; competing anchor/queued restores caused the
            # viewport to visibly bounce on every live status transition.
            saved_scroll = None
            updates_were_enabled = True
            signals_were_blocked = False
            self._suspend_yield = True
            if 'listbox' in data and data['listbox']:
                try:
                    saved_scroll = data['listbox'].verticalScrollBar().value()
                    updates_were_enabled = data['listbox'].updatesEnabled()
                    signals_were_blocked = data['listbox'].signalsBlocked()
                    data['listbox'].blockSignals(True)
                    data['listbox'].setUpdatesEnabled(False)
                except Exception:
                    saved_scroll = None
            
            if not self._reload_progress_view_data(data):
                return

            # Note: chapter_display_info is already rebuilt/updated above
            # For OPF mode: _update_chapter_status_info updated existing entries
            # For fallback mode: _rebuild_chapter_display_info rebuilt from scratch
            
            # Update the listbox display
            self._update_listbox_display(data)
            
            # Update statistics if available
            self._update_statistics_display(data)

            # _update_listbox_display already applies the current filter while
            # updating item metadata; avoid a second all-rows Qt pass here.
            self._progress_list_show_special(data)
            
            # Restore once, before painting is re-enabled. The streamed
            # reconciler performs its own final restore if it is still running.
            if 'listbox' in data and data['listbox']:
                try:
                    sb = data['listbox'].verticalScrollBar()
                    if saved_scroll is not None:
                        target = min(saved_scroll, sb.maximum())
                        if sb.value() != target:
                            sb.setValue(target)
                    data['listbox'].setUpdatesEnabled(updates_were_enabled)
                    data['listbox'].blockSignals(signals_were_blocked)
                    data['listbox'].viewport().update()
                except Exception:
                    try:
                        data['listbox'].setUpdatesEnabled(updates_were_enabled)
                        data['listbox'].blockSignals(signals_were_blocked)
                    except Exception:
                        pass
            self._suspend_yield = False
            
            # print("✅ Retranslation data refreshed successfully")
            
        except RuntimeError as e:
            print(f"❌ Failed to refresh data - widget deleted: {e}")
        except FileNotFoundError as e:
            print(f"❌ Failed to refresh data - file not found: {e}")
            try:
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Output Folder Not Found", 
                                      f"The output folder appears to have been deleted or moved.\n\n"
                                      f"File not found: {os.path.basename(str(e))}")
            except (RuntimeError, AttributeError):
                print(f"[WARN] Could not show error dialog - dialog was deleted")
        except PermissionError as e:
            # Refresh runs periodically. If the translator is writing/replacing
            # the progress JSON, skip this tick instead of interrupting the user.
            print(f"⚠️ Progress file locked during refresh; will retry on next refresh tick: {e}")
        except Exception as e:
            print(f"❌ Failed to refresh data: {e}")
            import traceback
            traceback.print_exc()
            try:
                # Show friendlier error message for common cases
                error_msg = str(e)
                if "No such file or directory" in error_msg or "cannot find the path" in error_msg:
                    self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Output Folder Not Found", 
                                          f"The output folder appears to have been deleted or moved.\n\n"
                                          f"Error: {error_msg}")
                else:
                    self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "Refresh Failed", 
                                      f"Failed to refresh data: {error_msg}")
            except (RuntimeError, AttributeError):
                # Dialog was also deleted, just print to console
                print(f"[WARN] Could not show error dialog - dialog was deleted")
        finally:
            self._suspend_yield = False
            try:
                listbox = data.get('listbox') if isinstance(data, dict) else None
                if listbox:
                    listbox.setUpdatesEnabled(updates_were_enabled)
                    listbox.blockSignals(signals_were_blocked)
                    listbox.viewport().update()
            except Exception:
                pass
    

    def _progress_list_show_special(self, data):
        show_special_files = data.get('show_special_files_state', False) if isinstance(data, dict) else False
        cb = data.get('show_special_files_cb') if isinstance(data, dict) else None
        if cb:
            try:
                show_special_files = cb.isChecked()
            except RuntimeError:
                pass
        if isinstance(data, dict):
            data['show_special_files_state'] = show_special_files
        return show_special_files

    def _progress_list_sync_model_toggle(self, data):
        cb = data.get('show_model_info_cb') if isinstance(data, dict) else None
        if cb:
            try:
                data['show_model_info_state'] = cb.isChecked()
            except RuntimeError:
                pass


    def _apply_progress_list_item_visuals(self, item, status):
        item.setForeground(QColor(progress_status_color(status)))

    def _set_progress_list_item_metadata(self, item, info, status, show_special_files):
        is_special = info.get('is_special', False)
        is_skipped_special = self._progress_entry_needs_special_visibility(info)
        item.setToolTip("")
        item.setData(Qt.UserRole, {
            'is_special': is_special,
            'info': info,
            'progress_key': info.get('progress_key'),
            'item_key': self._progress_list_item_key(info),
            'payload_revision': self._progress_list_payload_revision(info),
        })
        item.setData(Qt.UserRole + 2, status)
        chapter_info = info.get('info') or info.get('progress_entry') or {}
        if info.get('pdf_toc_section') or chapter_info.get('pdf_toc_section'):
            full_title = str(
                info.get('pdf_toc_title_translated')
                or info.get('translated_title')
                or chapter_info.get('pdf_toc_title_translated')
                or chapter_info.get('translated_title')
                or info.get('pdf_toc_title')
                or chapter_info.get('pdf_toc_title')
                or chapter_info.get('title')
                or ''
            ).strip()
            if full_title:
                item.setToolTip(full_title)
        item.setHidden(is_skipped_special and not show_special_files)

    def _populate_progress_listbox_streamed(self, data, chunk_size=150, preserve_selection=False, preserve_scroll=False):
        """Reconcile large progress lists in chunks without clearing the viewport."""
        if not self._is_data_valid(data):
            return

        listbox = data.get('listbox')
        if not listbox:
            return

        self._progress_list_sync_model_toggle(data)
        infos = list(data.get('chapter_display_info') or [])
        max_original_len, max_output_len = self._progress_list_column_widths(infos, data)
        data['_progress_list_column_widths'] = (max_original_len, max_output_len)
        data['_progress_list_view_revision'] = (
            self._progress_list_show_special(data),
            bool(data.get('show_model_info_state')),
        )

        selected_keys = set()
        if preserve_selection:
            try:
                for item in listbox.selectedItems():
                    payload = item.data(Qt.UserRole) or {}
                    key = payload.get('item_key') or self._progress_list_item_key(payload.get('info') or {})
                    if key:
                        selected_keys.add(key)
            except RuntimeError:
                selected_keys = set()

        saved_scroll = None
        if preserve_scroll:
            try:
                saved_scroll = listbox.verticalScrollBar().value()
            except RuntimeError:
                saved_scroll = None

        generation = int(data.get('_listbox_populate_generation', 0)) + 1
        data['_listbox_populate_generation'] = generation
        data['_listbox_populate_active'] = True

        state = {'idx': 0}

        def _finish():
            if generation != data.get('_listbox_populate_generation'):
                return
            data['_listbox_populate_active'] = False
            signals_were_blocked = False
            updates_were_enabled = True
            try:
                signals_were_blocked = listbox.signalsBlocked()
                updates_were_enabled = listbox.updatesEnabled()
                listbox.blockSignals(True)
                listbox.setUpdatesEnabled(False)
                while listbox.count() > len(infos):
                    listbox.takeItem(listbox.count() - 1)

                if preserve_selection:
                    for row in range(listbox.count()):
                        item = listbox.item(row)
                        payload = item.data(Qt.UserRole) or {}
                        item_key = payload.get('item_key') if isinstance(payload, dict) else None
                        item.setSelected(bool(item_key and item_key in selected_keys))

                if saved_scroll is not None:
                    sb = listbox.verticalScrollBar()
                    sb.setValue(min(saved_scroll, sb.maximum()))
                label = data.get('selection_count_label')
                if label:
                    label.setText(f"Selected: {len(listbox.selectedItems())}")
            except RuntimeError:
                pass
            finally:
                try:
                    listbox.blockSignals(signals_were_blocked)
                    listbox.setUpdatesEnabled(updates_were_enabled)
                    listbox.viewport().update()
                except RuntimeError:
                    pass

            deferred_payload = data.pop(
                '_deferred_prefetched_progress_payload',
                None,
            )
            deferred_applier = data.get('_apply_prefetched_progress_payload')
            if deferred_payload is not None and callable(deferred_applier):
                QTimer.singleShot(
                    0,
                    lambda payload=deferred_payload: deferred_applier(payload),
                )

        def _add_chunk():
            if generation != data.get('_listbox_populate_generation'):
                return
            if not self._is_data_valid(data):
                data['_listbox_populate_active'] = False
                return
            signals_were_blocked = False
            updates_were_enabled = True
            try:
                signals_were_blocked = listbox.signalsBlocked()
                updates_were_enabled = listbox.updatesEnabled()
                listbox.blockSignals(True)
                listbox.setUpdatesEnabled(False)
                show_special_files = self._progress_list_show_special(data)
                end_idx = min(state['idx'] + chunk_size, len(infos))
                for idx in range(state['idx'], end_idx):
                    info = infos[idx]
                    display, status = self._progress_list_display_text(
                        info,
                        data,
                        max_original_len,
                        max_output_len,
                    )
                    item = listbox.item(idx) if idx < listbox.count() else None
                    if item is None:
                        item = QListWidgetItem(display)
                        self._add_compact_inline_list_item(listbox, item)
                    elif item.text() != display:
                        item.setText(display)
                    self._set_compact_inline_item_size(listbox, item)
                    self._apply_progress_list_item_visuals(item, status)
                    self._set_progress_list_item_metadata(item, info, status, show_special_files)
                    item.setData(
                        Qt.UserRole + 4,
                        (
                            display,
                            status,
                            self._progress_entry_needs_special_visibility(info)
                            and not show_special_files,
                        ),
                    )
                    if preserve_selection:
                        item.setSelected(self._progress_list_item_key(info) in selected_keys)
                state['idx'] = end_idx
            except RuntimeError:
                if generation == data.get('_listbox_populate_generation'):
                    data['_listbox_populate_active'] = False
                return
            finally:
                try:
                    listbox.blockSignals(signals_were_blocked)
                    listbox.setUpdatesEnabled(updates_were_enabled)
                    listbox.viewport().update()
                except RuntimeError:
                    pass

            if state['idx'] < len(infos):
                QTimer.singleShot(0, _add_chunk)
            else:
                _finish()

        QTimer.singleShot(0, _add_chunk)

    def _update_listbox_display(self, data):
        """Update the listbox display with current chapter information"""
        if not self._is_data_valid(data):
            print("⚠️ Cannot update listbox display - widgets have been deleted")
            return

        listbox = data['listbox']
        self._progress_list_sync_model_toggle(data)
        count_existing = listbox.count()
        count_new = len(data.get('chapter_display_info') or [])
        if data.get('_listbox_populate_active') or count_existing != count_new:
            self._populate_progress_listbox_streamed(
                data,
                preserve_selection=True,
                preserve_scroll=True,
            )
            return

        # Row objects can be safely reused even when a genuine reorder changes
        # their identity; metadata and selection are reconciled by stable keys.
        # Avoiding clear/rebuild keeps the scrollbar range and viewport stable.
        infos = data.get('chapter_display_info') or []
        show_special_files = self._progress_list_show_special(data)
        view_revision = (
            show_special_files,
            bool(data.get('show_model_info_state')),
        )

        # Reject unchanged rows using only compact Python tuples before doing
        # any display formatting or status/model resolution.  The former order
        # formatted every one of 1,000+ rows on every progress-file update and
        # only then discovered that almost all QListWidgetItems were unchanged.
        candidate_updates = []
        identity_changed = False
        view_changed = data.get('_progress_list_view_revision') != view_revision
        for idx, info in enumerate(infos):
            item = listbox.item(idx)
            if not item:
                continue
            old_payload = item.data(Qt.UserRole) or {}
            payload_revision = self._progress_list_payload_revision(info)
            item_key = self._progress_list_item_key(info)
            payload_changed = (
                not isinstance(old_payload, dict)
                or old_payload.get('payload_revision')
                != payload_revision
            )
            row_identity_changed = (
                not isinstance(old_payload, dict)
                or old_payload.get('item_key') != item_key
            )
            if row_identity_changed:
                identity_changed = True
            if view_changed or row_identity_changed or payload_changed:
                candidate_updates.append((item, info))

        if not candidate_updates:
            return

        # Width changes affect the padding of every OPF row. Recompute widths
        # only when at least one row changed, and fall back to a streamed
        # all-row reconcile if the shared layout actually changed.
        max_original_len, max_output_len = self._progress_list_column_widths(
            infos,
            data,
        )
        column_widths = (max_original_len, max_output_len)
        if data.get('_progress_list_column_widths') != column_widths:
            candidate_updates = [
                (listbox.item(idx), info)
                for idx, info in enumerate(infos)
                if listbox.item(idx) is not None
            ]

        # A large status transition (for example queueing hundreds of title-only
        # chapters) must yield between Qt batches. Reuse the existing streamed
        # reconciler instead of executing hundreds of item mutations in one GUI
        # callback.
        if len(candidate_updates) > _PROGRESS_DIRECT_ROW_UPDATE_LIMIT:
            self._populate_progress_listbox_streamed(
                data,
                chunk_size=_PROGRESS_DIRECT_ROW_UPDATE_LIMIT,
                preserve_selection=True,
                preserve_scroll=True,
            )
            return

        row_updates = []
        for item, info in candidate_updates:
            display, display_status = self._progress_list_display_text(
                info,
                data,
                max_original_len,
                max_output_len,
            )
            hidden = (
                self._progress_entry_needs_special_visibility(info)
                and not show_special_files
            )
            fingerprint = (display, display_status, hidden)
            row_updates.append(
                (item, info, display, display_status, fingerprint)
            )

        selected_keys = set()
        if identity_changed:
            try:
                for selected_item in listbox.selectedItems():
                    payload = selected_item.data(Qt.UserRole) or {}
                    if isinstance(payload, dict) and payload.get('item_key'):
                        selected_keys.add(payload['item_key'])
            except RuntimeError:
                selected_keys = set()

        updates_were_enabled = listbox.updatesEnabled()
        signals_were_blocked = listbox.signalsBlocked()
        listbox.setUpdatesEnabled(False)
        listbox.blockSignals(True)
        try:
            for item, info, display, display_status, fingerprint in row_updates:
                old_fingerprint = item.data(Qt.UserRole + 4)
                if item.text() != display:
                    item.setText(display)
                if (
                    not isinstance(old_fingerprint, tuple)
                    or len(old_fingerprint) < 2
                    or old_fingerprint[1] != display_status
                ):
                    self._apply_progress_list_item_visuals(item, display_status)
                self._set_progress_list_item_metadata(item, info, display_status, show_special_files)
                if old_fingerprint != fingerprint:
                    item.setData(Qt.UserRole + 4, fingerprint)
            if identity_changed:
                for row in range(listbox.count()):
                    item = listbox.item(row)
                    payload = item.data(Qt.UserRole) or {}
                    item_key = payload.get('item_key') if isinstance(payload, dict) else None
                    item.setSelected(bool(item_key and item_key in selected_keys))
                label = data.get('selection_count_label')
                if label:
                    label.setText(f"Selected: {len(listbox.selectedItems())}")
        finally:
            listbox.blockSignals(signals_were_blocked)
            listbox.setUpdatesEnabled(updates_were_enabled)
        data['_progress_list_column_widths'] = column_widths
        data['_progress_list_view_revision'] = view_revision

    def _update_statistics_display(self, data):
        """Update statistics display for both OPF and non-OPF files"""
        # Find statistics labels in the container
        container = data['container']
        
        # Search for statistics labels by traversing the widget hierarchy
        def find_stats_labels(widget):
            labels = {}
            if hasattr(widget, 'children'):
                for child in widget.children():
                    if hasattr(child, 'text'):
                        text = child.text()
                        if text.startswith('Total:'):
                            labels['total'] = child
                        elif text.startswith('✅ Completed:'):
                            labels['completed'] = child
                        elif text.startswith('🔗 Merged:'):
                            labels['merged'] = child
                        elif text.startswith('🔄 In Progress:'):
                            labels['in_progress'] = child
                        elif text.startswith('❓ Pending:'):
                            labels['pending'] = child
                        elif text.startswith('⬜ Not Translated:') or text.startswith('✨ Not Refined:') or text.startswith('🔊 No TTS:'):
                            labels['missing'] = child
                        elif text.startswith('❌ Failed:') or text.startswith('💀 Refine Failed:'):
                            labels['failed'] = child
                        elif text.startswith('⏭️ Skipped:'):
                            labels['skipped'] = child
                    
                    # Recursively search children
                    labels.update(find_stats_labels(child))
            return labels
        
        stats_labels = data.get('_stats_labels')
        if not isinstance(stats_labels, dict):
            stats_labels = find_stats_labels(container)
            data['_stats_labels'] = stats_labels
        
        if stats_labels:
            # Recalculate statistics from chapter_display_info (works for both OPF and non-OPF)
            (total_chapters, chunk_count, completed, merged, in_progress, pending,
             missing, failed, skipped, mode) = self._progress_statistics(data)
            stats_fingerprint = (
                total_chapters,
                chunk_count,
                completed,
                merged,
                in_progress,
                pending,
                missing,
                failed,
                skipped,
                mode,
            )
            if stats_fingerprint == data.get('_stats_fingerprint'):
                return
            data['_stats_fingerprint'] = stats_fingerprint

            # Update labels
            if 'total' in stats_labels:
                stats_labels['total'].setText(_progress_total_label(total_chapters, chunk_count))
            if 'completed' in stats_labels:
                stats_labels['completed'].setText(f"✅ Completed: {completed} | ")
            if 'merged' in stats_labels:
                if merged > 0:
                    stats_labels['merged'].setText(f"🔗 Merged: {merged} | ")
                    stats_labels['merged'].setVisible(True)
                else:
                    stats_labels['merged'].setVisible(False)
            if 'in_progress' in stats_labels:
                if in_progress > 0:
                    stats_labels['in_progress'].setText(f"🔄 In Progress: {in_progress} | ")
                    stats_labels['in_progress'].setVisible(True)
                else:
                    stats_labels['in_progress'].setVisible(False)
            if 'pending' in stats_labels:
                if pending > 0:
                    stats_labels['pending'].setText(f"❓ Pending: {pending} | ")
                    stats_labels['pending'].setVisible(True)
                else:
                    stats_labels['pending'].setVisible(False)
            missing_label, failed_icon, failed_label, failed_color = progress_stats_labels(mode)
            if 'missing' in stats_labels:
                stats_labels['missing'].setText(f"{missing_label}: {missing} | ")
            if 'failed' in stats_labels:
                stats_labels['failed'].setText(f"{failed_icon} {failed_label}: {failed} | ")
                stats_labels['failed'].setStyleSheet(f"color: {failed_color};")
            if 'skipped' in stats_labels:
                stats_labels['skipped'].setText(f"⏭️ Skipped: {skipped}")
                stats_labels['skipped'].setVisible(skipped > 0)

    def _refresh_image_folder_data(self, data):
        """Refresh the image folder retranslation dialog data by rescanning files"""
        try:
            # Validate that widgets still exist
            if not self._is_data_valid(data):
                print("⚠️ Cannot refresh - widgets have been deleted")
                return
            
            # Save current selections to restore after refresh
            selected_indices = []
            try:
                selected_indices = [data['listbox'].row(item) for item in data['listbox'].selectedItems()]
            except RuntimeError:
                print("⚠️ Could not save selection state - widget was deleted")
                return
            
            output_dir = data['output_dir']
            progress_file = data['progress_file']
            folder_path = data['folder_path']
            
            scanned = scan_image_folder(output_dir, progress_file)
            file_info = scanned['file_info']
            progress_data = scanned['progress_data']
            html_files = scanned['html_files']
            image_files = scanned['image_files']
            
            # Update data dictionary with fresh data
            data['file_info'] = file_info
            data['progress_data'] = progress_data
            
            # IMPORTANT: Also update the original refresh_data dict so future operations use fresh data
            # This ensures delete operations after refresh work with current state
            if 'progress_data' in data:
                # Update the reference in the closure
                data['progress_data'] = progress_data
            
            # Clear and rebuild listbox
            listbox = data['listbox']
            listbox.clear()
            
            # Add all tracked files to display
            for info in file_info:
                display = image_folder_row_text(info)
                
                self._add_compact_inline_list_item(listbox, display)
            
            # Restore selections
            try:
                if selected_indices:
                    for idx in selected_indices:
                        if idx < listbox.count():
                            listbox.item(idx).setSelected(True)
                    # Update selection count
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText(f"Selected: {len(selected_indices)}")
                else:
                    listbox.clearSelection()
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText("Selected: 0")
            except RuntimeError:
                print("⚠️ Could not restore selection state - widget was deleted during refresh")
            
            print(f"✅ Image folder data refreshed: {len(html_files)} HTML files, {len(image_files)} cover images")
            
        except Exception as e:
            print(f"❌ Failed to refresh image folder data: {e}")
            import traceback
            traceback.print_exc()

    def _force_retranslation_multiple_files(self):
        """Handle force retranslation when multiple files are selected - now uses shared logic"""
        try:
            print(f"[DEBUG] _force_retranslation_multiple_files called with {len(self.selected_files)} files")

            subtitle_bundle_target = self._selected_subtitle_bundle_progress_target()
            if subtitle_bundle_target:
                self._open_subtitle_bundle_progress_manager(subtitle_bundle_target)
                return

            # First, check if all selected files are images from the same folder
            # This handles the case where folder selection results in individual file selections
            if len(self.selected_files) > 1:
                all_images = True
                parent_dirs = set()
                
                image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
                
                for file_path in self.selected_files:
                    if os.path.isfile(file_path) and file_path.lower().endswith(image_extensions):
                        parent_dirs.add(os.path.dirname(file_path))
                    else:
                        all_images = False
                        break
                
                # If all files are images from the same directory, treat it as a folder selection
                if all_images and len(parent_dirs) == 1:
                    folder_path = parent_dirs.pop()
                    print(f"[DEBUG] Detected {len(self.selected_files)} images from same folder: {folder_path}")
                    print(f"[DEBUG] Treating as folder selection")
                    self._force_retranslation_images_folder(folder_path)
                    return
            
            # Otherwise, continue with normal categorization
            epub_files = []
            text_files = []
            image_files = []
            folders = []
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
            
            for file_path in self.selected_files:
                if os.path.isdir(file_path):
                    folders.append(file_path)
                elif file_path.lower().endswith('.epub'):
                    epub_files.append(file_path)
                elif file_path.lower().endswith(('.txt', '.srt', '.ass', '.lrc', '.zip')):
                    text_files.append(file_path)
                elif file_path.lower().endswith(image_extensions):
                    image_files.append(file_path)
            
            # Build summary
            summary_parts = []
            if epub_files:
                summary_parts.append(f"{len(epub_files)} EPUB file(s)")
            if text_files:
                summary_parts.append(f"{len(text_files)} text file(s)")
            if image_files:
                summary_parts.append(f"{len(image_files)} image file(s)")
            if folders:
                summary_parts.append(f"{len(folders)} folder(s)")
            
            if not summary_parts:
                self._styled_msgbox(QMessageBox.Information, self, "Info", "No valid files selected.")
                return
            
            # Create a unique key for the current selection
            selection_key = tuple(sorted(self.selected_files))
            
            # Check if we already have a cached dialog for this exact selection
            if (hasattr(self, '_multi_file_retranslation_dialog') and 
                self._multi_file_retranslation_dialog and 
                hasattr(self, '_multi_file_selection_key') and 
                self._multi_file_selection_key == selection_key):
                # Reuse existing dialog - show first, then refresh tabs without blocking open.
                cached_dialog = self._multi_file_retranslation_dialog
                cached_dialog.show()
                cached_dialog.raise_()
                cached_dialog.activateWindow()
                if getattr(cached_dialog, '_multi_file_tabs_building', False):
                    return
                return
            
            # If there's an existing dialog for a different selection, destroy it first
            if hasattr(self, '_multi_file_retranslation_dialog') and self._multi_file_retranslation_dialog:
                self._multi_file_retranslation_dialog.close()
                self._multi_file_retranslation_dialog.deleteLater()
                self._multi_file_retranslation_dialog = None
            
            # Create main dialog
            dialog = QDialog(self)
            self._stamp_progress_manager_input_signature(dialog)
            dialog.setWindowTitle("Progress Manager - Multiple Files")
            # Parent-child windowing keeps this above the translator GUI
            dialog.setWindowModality(Qt.NonModal)
            # Store the list of EPUBs in the dialog for cross-tab state updates
            dialog._epub_files_in_dialog = epub_files + text_files
            # Increased height from 18% to 25% for better visibility
            width, height = self._get_dialog_size(0.25, 0.45)
            dialog.resize(width, height)
            
            # Set icon
            try:
                from PySide6.QtGui import QIcon
                if hasattr(self, 'base_dir'):
                    base_dir = self.base_dir
                else:
                    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                ico_path = os.path.join(base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    dialog.setWindowIcon(QIcon(ico_path))
            except Exception as e:
                print(f"Failed to load icon: {e}")
            
            dialog_layout = QVBoxLayout(dialog)
            
            # Summary label
            summary_label = QLabel(f"Selected: {', '.join(summary_parts)}")
            summary_font = QFont('Arial', 12)
            summary_font.setBold(True)
            summary_label.setFont(summary_font)
            dialog_layout.addWidget(summary_label)
            
            # Count total files for UI decision
            total_files = len(epub_files) + len(text_files) + len(folders) + len(image_files)
            use_dropdown = total_files > 3

            if use_dropdown:
                # ── Dropdown + arrows for many files ──
                from PySide6.QtWidgets import QComboBox, QStackedWidget

                nav_row = QHBoxLayout()
                nav_row.setSpacing(6)

                nav_prev = QPushButton("◀")
                nav_prev.setFixedWidth(36)
                nav_prev.setStyleSheet(
                    "QPushButton { background-color:#3a3a3a; color:white; font-weight:bold; "
                    "font-size:13pt; border:1px solid #5a9fd4; border-radius:4px; padding:4px; }"
                    "QPushButton:hover { background-color:#4a8fc4; }"
                    "QPushButton:disabled { color:#666; background-color:#2a2a2a; }"
                )

                combo = QComboBox()
                combo.setStyleSheet(
                    "QComboBox { background-color:#3a3a3a; color:white; font-weight:bold; "
                    "font-size:11pt; padding:6px 10px; border:1px solid #5a9fd4; border-radius:4px; }"
                    "QComboBox::drop-down { border:none; }"
                    "QComboBox QAbstractItemView { background-color:#2d2d2d; color:white; "
                    "selection-background-color:#5a9fd4; }"
                )

                nav_counter = QLabel("1 / 1")
                nav_counter.setStyleSheet("color:#94a3b8; font-size:10pt; font-weight:bold;")
                nav_counter.setFixedWidth(60)
                nav_counter.setAlignment(Qt.AlignCenter)

                nav_next = QPushButton("▶")
                nav_next.setFixedWidth(36)
                nav_next.setStyleSheet(nav_prev.styleSheet())

                nav_row.addWidget(nav_prev)
                nav_row.addWidget(combo, stretch=1)
                nav_row.addWidget(nav_counter)
                nav_row.addWidget(nav_next)
                dialog_layout.addLayout(nav_row)

                stack = QStackedWidget()
                dialog_layout.addWidget(stack)

                def _update_nav():
                    idx = combo.currentIndex()
                    n = combo.count()
                    nav_prev.setEnabled(idx > 0)
                    nav_next.setEnabled(idx < n - 1)
                    nav_counter.setText(f"{idx + 1} / {n}")
                    stack.setCurrentIndex(idx)
                    populate_current = getattr(dialog, '_populate_current_progress_tab', None)
                    if callable(populate_current):
                        QTimer.singleShot(0, populate_current)

                combo.currentIndexChanged.connect(lambda _: _update_nav())
                nav_prev.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() - 1))
                nav_next.clicked.connect(lambda: combo.setCurrentIndex(combo.currentIndex() + 1))

                # Wrap stack+combo to behave like QTabWidget for the rest of the code
                class _DropdownNotebook:
                    """Thin adapter so addTab() works the same as QTabWidget."""
                    def __init__(self, stack, combo):
                        self._stack = stack
                        self._combo = combo
                    def addTab(self, widget, label):
                        self._stack.addWidget(widget)
                        self._combo.addItem(label)
                    def currentIndex(self):
                        return self._combo.currentIndex()
                    def setCurrentIndex(self, idx):
                        self._combo.setCurrentIndex(idx)

                notebook = _DropdownNotebook(stack, combo)
                dialog._dropdown_update_nav = _update_nav
            else:
                # ── Standard tabs for ≤7 files ──
                notebook = QTabWidget()
                notebook.setStyleSheet("""
                    QTabWidget::pane {
                        border: 2px solid transparent;
                        border-radius: 4px;
                        background-color: #2d2d2d;
                    }
                    QTabBar::tab {
                        background-color: #3a3a3a;
                        color: white;
                        padding: 8px 16px;
                        margin-right: 2px;
                        border: 1px solid #5a9fd4;
                        border-bottom: none;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                        font-weight: bold;
                        font-size: 11pt;
                    }
                    QTabBar::tab:selected {
                        background-color: #5a9fd4;
                        color: white;
                    }
                    QTabBar::tab:hover {
                        background-color: #4a8fc4;
                    }
                    QTabBar QToolButton {
                        background-color: #3a3a3a;
                        border: 1px solid #5a9fd4;
                        border-radius: 3px;
                        color: white;
                        font-weight: bold;
                        font-size: 14pt;
                        width: 36px;
                        padding: 4px;
                        margin: 2px 4px;
                    }
                    QTabBar QToolButton:hover {
                        background-color: #4a8fc4;
                    }
                    QTabBar::scroller {
                        width: 52px;
                    }
                """)
                dialog_layout.addWidget(notebook)
                notebook.currentChanged.connect(
                    lambda _idx: QTimer.singleShot(
                        0,
                        getattr(dialog, '_populate_current_progress_tab', lambda: None),
                    )
                )
            
            # Track all tab data
            tab_data = []
            tabs_created = False
            
            # Store tab_data reference on the dialog for cross-tab operations
            dialog._tab_data = tab_data

            def _populate_current_progress_tab():
                """Materialize QListWidgetItems only for the visible page."""
                for tab_result in list(tab_data):
                    if not tab_result or tab_result.get('_initial_population_requested'):
                        continue
                    page = tab_result.get('container')
                    try:
                        if page is not None and not page.isVisible():
                            continue
                    except RuntimeError:
                        continue
                    tab_result['_initial_population_requested'] = True
                    self._populate_progress_listbox_streamed(tab_result)
                    break

            dialog._populate_current_progress_tab = _populate_current_progress_tab

            # Paint the full-size multi-file shell before scanning/building every EPUB tab.
            dialog.show()
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception:
                pass
            
            # Get the global show_special state from the first file that has it cached
            # Default to True if any text files are present, False otherwise
            global_show_special = True if text_files else False
            
            for file_path in epub_files + text_files:
                file_key = os.path.abspath(file_path)
                if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                    cached_data = self._retranslation_dialog_cache[file_key]
                    if cached_data and 'show_special_files_state' in cached_data:
                        global_show_special = cached_data['show_special_files_state']
                        break  # Use the first one we find
            
            # Determine output directory override (matches single-file logic)
            override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
            if not override_dir and hasattr(self, 'config'):
                try:
                    override_dir = self.config.get('output_directory')
                except Exception:
                    override_dir = None

            # Stream tab creation: add lightweight tab shells immediately, then build
            # each EPUB/text tab on its own event-loop turn.
            self._add_multi_file_buttons(dialog, notebook, tab_data)

            def closeEvent(event):
                event.ignore()
                dialog.hide()

            dialog.closeEvent = closeEvent
            self._multi_file_retranslation_dialog = dialog
            self._multi_file_selection_key = selection_key

            def _update_dropdown_nav_safe():
                if hasattr(dialog, '_dropdown_update_nav'):
                    try:
                        dialog._dropdown_update_nav()
                    except RuntimeError:
                        pass

            build_tasks = []
            for file_path in epub_files + text_files:
                file_base = os.path.splitext(os.path.basename(file_path))[0]
                print(f"[DEBUG] Queueing EPUB/text tab: {file_base}")

                output_dir = os.path.join(override_dir, file_base) if override_dir else file_base
                if not os.path.exists(output_dir):
                    print(f"[DEBUG] Output folder missing for {file_base}; will create via tab builder: {output_dir}")

                tab_frame = QWidget()
                tab_layout = QVBoxLayout(tab_frame)
                tab_layout.setContentsMargins(0, 0, 0, 0)
                loading_label = QLabel(f"Loading {file_base}...")
                loading_label.setAlignment(Qt.AlignCenter)
                loading_label.setStyleSheet("color: #94a3b8; font-size: 10pt; font-weight: bold; padding: 18px;")
                tab_layout.addWidget(loading_label)

                tab_name = file_base if use_dropdown else (file_base[:20] + "..." if len(file_base) > 20 else file_base)
                notebook.addTab(tab_frame, tab_name)
                _update_dropdown_nav_safe()
                build_tasks.append(('epub_text', file_path, file_base, tab_frame, loading_label))

            for folder_path in folders:
                build_tasks.append(('folder', folder_path, os.path.basename(folder_path) or folder_path, None, None))

            build_state = {'idx': 0, 'tabs_created': False}
            dialog._multi_file_tabs_building = True

            def _refresh_tabs_streamed(idx=0):
                if idx >= len(tab_data):
                    return
                _td = tab_data[idx]
                _rf = _td.get('refresh_func') if _td else None
                if callable(_rf):
                    try:
                        _rf()
                    except Exception as _e:
                        print(f"[WARN] Auto-refresh failed for a tab: {_e}")
                QTimer.singleShot(25, lambda: _refresh_tabs_streamed(idx + 1))

            def _finish_streamed_tabs():
                dialog._multi_file_tabs_building = False

                if image_files and not build_state['tabs_created']:
                    image_tab_result = self._create_individual_images_tab(
                        image_files,
                        notebook,
                        dialog
                    )
                    if image_tab_result:
                        tab_data.append(image_tab_result)
                        build_state['tabs_created'] = True
                        _update_dropdown_nav_safe()

                if not build_state['tabs_created'] and folders:
                    scanned_images = []
                    for folder_path in folders:
                        if os.path.isdir(folder_path):
                            try:
                                for file in os.listdir(folder_path):
                                    file_path = os.path.join(folder_path, file)
                                    if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                                        scanned_images.append(file_path)
                            except Exception:
                                pass

                    if scanned_images:
                        image_tab_result = self._create_individual_images_tab(
                            scanned_images,
                            notebook,
                            dialog
                        )
                        if image_tab_result:
                            tab_data.append(image_tab_result)
                            build_state['tabs_created'] = True
                            _update_dropdown_nav_safe()

                if not build_state['tabs_created']:
                    self._styled_msgbox(QMessageBox.Information, self, "Info",
                        "No translation output found for any of the selected files.\n\n"
                        "Make sure the output folders exist in your script directory.")
                    dialog.hide()
                    return

                _update_dropdown_nav_safe()
                if not tab_data:
                    print(f"[WARN] No tab data to refresh on dialog open")

            def _build_next_tab():
                if getattr(dialog, '_progress_input_retired', False):
                    return
                if build_state['idx'] >= len(build_tasks):
                    _finish_streamed_tabs()
                    return

                kind, path, label, tab_frame, tab_loading_label = build_tasks[build_state['idx']]
                build_state['idx'] += 1

                if kind == 'epub_text':
                    print(f"[DEBUG] Creating streamed tab for {label}")
                    try:
                        if tab_frame.layout():
                            self._clear_layout(tab_frame.layout())
                        tab_result = self._force_retranslation_epub_or_text(
                            path,
                            parent_dialog=dialog,
                            tab_frame=tab_frame,
                            show_special_files_state=global_show_special,
                            _loading_label=tab_loading_label,
                        )
                        if tab_result:
                            cdi = tab_result.get('chapter_display_info', [])
                            completed = sum(1 for info in cdi if info.get('status') == 'completed')
                            in_progress = sum(1 for info in cdi if info.get('status') == 'in_progress')
                            tab_data.append(tab_result)
                            build_state['tabs_created'] = True
                            QTimer.singleShot(0, _populate_current_progress_tab)
                            print(f"[DEBUG] Successfully created tab for {label} (progress: {completed} done, {in_progress} in-progress)")
                        else:
                            if tab_frame.layout():
                                self._clear_layout(tab_frame.layout())
                                failed_label = QLabel(f"Failed to load {label}")
                                failed_label.setAlignment(Qt.AlignCenter)
                                failed_label.setStyleSheet("color: #e74c3c; font-size: 10pt; font-weight: bold; padding: 18px;")
                                tab_frame.layout().addWidget(failed_label)
                            print(f"[DEBUG] Failed to create content for {label}")
                    except Exception as _e:
                        print(f"[WARN] Failed to create streamed tab for {label}: {_e}")
                elif kind == 'folder':
                    folder_result = self._create_image_folder_tab(
                        path,
                        notebook,
                        dialog
                    )
                    if folder_result:
                        tab_data.append(folder_result)
                        build_state['tabs_created'] = True
                        _update_dropdown_nav_safe()

                QTimer.singleShot(0, _build_next_tab)

            QTimer.singleShot(50, _build_next_tab)
            return

            # Create tabs for EPUB/text files using shared logic
            pending_tabs = []  # Collect before sorting
            for file_path in epub_files + text_files:
                file_base = os.path.splitext(os.path.basename(file_path))[0]
                
                print(f"[DEBUG] Checking EPUB/text: {file_base}")
                
                # Quick check if output exists (respect override output directory)
                # NOTE: For multi-file, don't skip when missing — the tab builder will
                # create the output folder (same as single-file behavior).
                output_dir = os.path.join(override_dir, file_base) if override_dir else file_base
                if not os.path.exists(output_dir):
                    print(f"[DEBUG] Output folder missing for {file_base}; will create via tab builder: {output_dir}")
                
                print(f"[DEBUG] Creating tab for {file_base}")
                
                # Create tab
                tab_frame = QWidget()
                tab_layout = QVBoxLayout(tab_frame)
                tab_name = file_base if use_dropdown else (file_base[:20] + "..." if len(file_base) > 20 else file_base)
                
                # Use shared logic to populate the tab with global state
                tab_result = self._force_retranslation_epub_or_text(
                    file_path, 
                    parent_dialog=dialog, 
                    tab_frame=tab_frame,
                    show_special_files_state=global_show_special
                )
                
                # Only keep the tab if content was successfully created
                if tab_result:
                    # Count progress for sorting
                    cdi = tab_result.get('chapter_display_info', [])
                    completed = sum(1 for info in cdi if info.get('status') == 'completed')
                    in_progress = sum(1 for info in cdi if info.get('status') == 'in_progress')
                    progress_score = completed + in_progress
                    pending_tabs.append((progress_score, tab_frame, tab_name, tab_result))
                    print(f"[DEBUG] Successfully created tab for {file_base} (progress: {completed} done, {in_progress} in-progress)")
                else:
                    print(f"[DEBUG] Failed to create content for {file_base}")
            
            # Sort tabs: most progress first
            pending_tabs.sort(key=lambda t: t[0], reverse=True)
            for _score, tab_frame, tab_name, tab_result in pending_tabs:
                notebook.addTab(tab_frame, tab_name)
                tab_data.append(tab_result)
                tabs_created = True
            
            # Create tabs for image folders (keeping existing logic for now)
            for folder_path in folders:
                folder_result = self._create_image_folder_tab(
                    folder_path, 
                    notebook, 
                    dialog
                )
                if folder_result:
                    tab_data.append(folder_result)
                    tabs_created = True
            
            # If only individual image files selected and no tabs created yet
            if image_files and not tabs_created:
                # Create a single tab for all individual images
                image_tab_result = self._create_individual_images_tab(
                    image_files,
                    notebook,
                    dialog
                )
                if image_tab_result:
                    tab_data.append(image_tab_result)
                    tabs_created = True
            
            # If no tabs were created from folders, try scanning folders for individual images
            if not tabs_created and folders:
                # Scan folders for individual image files
                scanned_images = []
                for folder_path in folders:
                    if os.path.isdir(folder_path):
                        try:
                            for file in os.listdir(folder_path):
                                file_path = os.path.join(folder_path, file)
                                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                                    scanned_images.append(file_path)
                        except:
                            pass
                
                # If we found images, create a tab for them
                if scanned_images:
                    image_tab_result = self._create_individual_images_tab(
                        scanned_images,
                        notebook,
                        dialog
                    )
                    if image_tab_result:
                        tab_data.append(image_tab_result)
                        tabs_created = True
            
            # If still no tabs were created, show error
            if not tabs_created:
                self._styled_msgbox(QMessageBox.Information, self, "Info", 
                    "No translation output found for any of the selected files.\n\n"
                    "Make sure the output folders exist in your script directory.")
                dialog.close()
                return
        
            # Add unified button bar that works across all tabs
            self._add_multi_file_buttons(dialog, notebook, tab_data)
            
            # Override close event to minimize instead of destroy
            def closeEvent(event):
                event.ignore()  # Ignore the close event
                dialog.hide()   # Just hide (minimize) the dialog
            
            dialog.closeEvent = closeEvent
            
            # Cache the dialog and selection key for reuse
            self._multi_file_retranslation_dialog = dialog
            self._multi_file_selection_key = selection_key
            
            # Update dropdown nav state after all tabs are added
            if hasattr(dialog, '_dropdown_update_nav'):
                dialog._dropdown_update_nav()

            # Show the dialog (non-modal to allow interaction with other windows)
            dialog.show()

            def _populate_tabs_after_show():
                for _idx, _td in enumerate(tab_data):
                    if _td:
                        QTimer.singleShot(
                            _idx * 10,
                            lambda td=_td: self._populate_progress_listbox_streamed(td),
                        )

            # Trigger refresh after the dialog has painted so large tabs do not block opening.
            def _refresh_tabs_after_show():
                if tab_data:
                    print(f"[DEBUG] Auto-clicking refresh on all {len(tab_data)} tabs on dialog open...")
                    for _td in tab_data:
                        _rf = _td.get('refresh_func') if _td else None
                        if callable(_rf):
                            try:
                                _rf()
                            except Exception as _e:
                                print(f"[WARN] Auto-refresh failed for a tab: {_e}")
                else:
                    print(f"[WARN] No tab data to refresh on dialog open")

            QTimer.singleShot(50, _populate_tabs_after_show)
            QTimer.singleShot(150, _refresh_tabs_after_show)
            
        except Exception as e:
            print(f"[ERROR] _force_retranslation_multiple_files failed: {e}")
            import traceback
            traceback.print_exc()
            self._styled_msgbox(QMessageBox.Critical, self, "Error", f"Failed to open retranslation dialog:\n{str(e)}")

    def _add_multi_file_buttons(self, dialog, notebook, tab_data):
        """Placeholder for future multi-file button functionality"""
        # No buttons needed - dialog has standard close button
        pass
              
    def _create_individual_images_tab(self, image_files, notebook, parent_dialog):
        """Create a tab for individual image files"""
        # Create tab
        tab_frame = QWidget()
        tab_layout = QVBoxLayout(tab_frame)
        notebook.addTab(tab_frame, "Individual Images")
        
        # Instructions
        instruction_label = QLabel(f"Selected {len(image_files)} individual image(s):")
        instruction_font = QFont('Arial', 11)
        instruction_label.setFont(instruction_font)
        tab_layout.addWidget(instruction_label)
        
        # Listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self._apply_compact_inline_list_style(listbox)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        tab_layout.addWidget(listbox)
        
        # File info
        file_info = []
        script_dir = _get_app_dir()
        
        # Check each image for translations
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Look for translations in various possible locations
            found_translations = []
            
            # Check in script directory with base name
            possible_dirs = [
                os.path.join(script_dir, base_name),
                os.path.join(script_dir, f"{base_name}_translated"),
                base_name,
                f"{base_name}_translated"
            ]
            
            for output_dir in possible_dirs:
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    # Look for HTML files
                    for file in os.listdir(output_dir):
                        if file.lower().endswith(('.html', '.xhtml', '.htm')) and base_name in file:
                            found_translations.append((output_dir, file))
            
            if found_translations:
                for output_dir, html_file in found_translations:
                    display = f"📄 {img_name} → {html_file} | ✅ Translated"
                    self._add_compact_inline_list_item(listbox, display)
                    
                    file_info.append({
                        'type': 'translated',
                        'source_image': img_path,
                        'output_dir': output_dir,
                        'file': html_file,
                        'path': os.path.join(output_dir, html_file)
                    })
            else:
                display = f"🖼️ {img_name} | ❌ No translation found"
                self._add_compact_inline_list_item(listbox, display)
        
        # Selection count
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 9)
        selection_count_label.setFont(selection_font)
        tab_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)

        # Right-click context menu to open translated/cover files
        def _open_file_for_row(row):
            if row < 0 or row >= len(file_info):
                return
            info = file_info[row]
            path = info.get('path')
            if not path or not os.path.exists(path):
                self._show_message('error', "File Missing", f"File not found:\n{path}", parent=parent_dialog)
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=parent_dialog)

        def _show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            row = listbox.row(item)
            menu = QMenu(listbox)
            menu.setStyleSheet(
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
            act_open = menu.addAction("📂 Open File")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_row(row)

        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        listbox.customContextMenuRequested.connect(_show_context_menu)
        
        return {
            'type': 'individual_images',
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _create_image_folder_tab(self, folder_path, notebook, parent_dialog):
        """Create a tab for image folder retranslation"""
        folder_name = os.path.basename(folder_path)
        output_dir = f"{folder_name}_translated"
        
        if not os.path.exists(output_dir):
            return None
        
        # Create tab
        tab_frame = QWidget()
        tab_layout = QVBoxLayout(tab_frame)
        tab_name = "📁 " + (folder_name[:17] + "..." if len(folder_name) > 17 else folder_name)
        notebook.addTab(tab_frame, tab_name)
        
        # Instructions
        instruction_label = QLabel("Select images to retranslate:")
        instruction_font = QFont('Arial', 11)
        instruction_label.setFont(instruction_font)
        tab_layout.addWidget(instruction_label)
        
        # Listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self._apply_compact_inline_list_style(listbox)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        tab_layout.addWidget(listbox)
        
        # Find files
        file_info = []
        
        # Add HTML files (any .html/.xhtml/.htm, not just response_*)
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.html', '.xhtml', '.htm')):
                match = re.match(r'^response_(\d+)_([^.]*).(?:html?|xhtml|htm)(?:\.xhtml)?$', file, re.IGNORECASE)
                if match:
                    index = match.group(1)
                    base_name = match.group(2)
                    display = f"📄 Image {index} | {base_name} | ✅ Completed"
                else:
                    display = f"📄 {file} | ✅ Completed"
                
                self._add_compact_inline_list_item(listbox, display)
                file_info.append({
                    'type': 'translated',
                    'file': file,
                    'path': os.path.join(output_dir, file)
                })
        
        # Add cover images
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for file in sorted(os.listdir(images_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    display = f"🖼️ Cover | {file} | ⏭️ Skipped"
                    self._add_compact_inline_list_item(listbox, display)
                    file_info.append({
                        'type': 'cover',
                        'file': file,
                        'path': os.path.join(images_dir, file)
                    })
        
        # Selection count
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 9)
        selection_count_label.setFont(selection_font)
        tab_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)

        # Right-click context menu (Open File)
        def _open_file_for_row(row):
            if row < 0 or row >= len(file_info):
                return
            info = file_info[row]
            path = info.get('path')
            if not path or not os.path.exists(path):
                self._show_message('error', "File Missing", f"File not found:\n{path}", parent=parent_dialog)
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=parent_dialog)

        def _show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            row = listbox.row(item)
            menu = QMenu(listbox)
            menu.setStyleSheet(
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
            act_open = menu.addAction("📂 Open File")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_row(row)

        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        listbox.customContextMenuRequested.connect(_show_context_menu)
        
        return {
            'type': 'image_folder',
            'folder_path': folder_path,
            'output_dir': output_dir,
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _force_retranslation_images_folder(self, folder_path):
        """Handle force retranslation for image folders"""
        # If folder_path is actually a file (single image), get its directory
        if os.path.isfile(folder_path):
            # Single image file - use basename without extension
            folder_name = os.path.splitext(os.path.basename(folder_path))[0]
        else:
            # Folder - use folder name as-is
            folder_name = os.path.basename(folder_path)
        
        # Check if we already have a cached dialog for this folder
        folder_key = os.path.abspath(folder_path)
        if hasattr(self, '_image_retranslation_dialog_cache') and folder_key in self._image_retranslation_dialog_cache:
            cached_dialog = self._image_retranslation_dialog_cache[folder_key]
            if cached_dialog:
                # Reuse existing dialog - just show it
                try:
                    # Click stored refresh button or call stored refresh func on reuse
                    if hasattr(cached_dialog, '_refresh_button') and cached_dialog._refresh_button:
                        QTimer.singleShot(0, cached_dialog._refresh_button.click)
                    elif hasattr(cached_dialog, '_refresh_func'):
                        QTimer.singleShot(0, cached_dialog._refresh_func)
                except Exception:
                    pass
                cached_dialog.show()
                cached_dialog.raise_()
                cached_dialog.activateWindow()
                return
        
        # Look for output folder in the SCRIPT'S directory, not relative to the selected folder
        script_dir = _get_app_dir()  # Application directory where output is generated
        
        output_dir, possible_output_dirs = image_folder_output_dir(self, folder_name, script_dir)
        
        if not output_dir:
            self._styled_msgbox(QMessageBox.Information, self, "Info", image_folder_not_found_message(
                folder_name, folder_path, script_dir, possible_output_dirs))
            return
        
        print(f"Using output directory: {output_dir}")
        
        # Check for progress tracking file
        progress_file = os.path.join(output_dir, "translation_progress.json")
        has_progress_tracking = os.path.exists(progress_file)
        
        print(f"Progress tracking: {has_progress_tracking} at {progress_file}")
        
        # Find all HTML files in the output directory
        html_files = []
        _html_seen = set()
        image_files = []
        progress_data = None
        
        if has_progress_tracking:
            # Load progress data for image translations
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    print(f"Loaded progress data with {len(progress_data)} entries")
                    
                # Extract files from progress data
                # The structure appears to use hash keys at the root level
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = value['output_file']
                        if not output_file:
                            continue
                        # Normalize path
                        output_norm = os.path.normpath(str(output_file))
                        # If absolute and under output_dir, store as relative
                        try:
                            if os.path.isabs(output_norm) and output_dir:
                                outdir_norm = os.path.normpath(output_dir)
                                if output_norm.startswith(outdir_norm):
                                    output_norm = os.path.relpath(output_norm, outdir_norm)
                        except Exception:
                            pass
                        if output_norm in _html_seen:
                            continue
                        _html_seen.add(output_norm)
                        html_files.append(output_norm)
                        print(f"Found tracked file: {output_norm}")
            except Exception as e:
                print(f"Error loading progress file: {e}")
                import traceback
                traceback.print_exc()
                has_progress_tracking = False
        
            # Also scan directory for any HTML files not in progress
            # Include all .html/.xhtml/.htm files plus generated image files
        try:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                # Include HTML files (any name)
                if (os.path.isfile(file_path) and 
                    file.lower().endswith(('.html', '.xhtml', '.htm')) and 
                    file not in html_files and file not in _html_seen):
                    _html_seen.add(file)
                    html_files.append(file)
                    print(f"Found HTML file: {file}")
                # Also include generated image files (not in images/ subdirectory)
                elif (os.path.isfile(file_path) and 
                      file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')) and
                      file not in html_files):
                    html_files.append(file)  # Add to html_files for now, will be handled separately
                    print(f"Found generated image file: {file}")
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        # Check for images subdirectory (cover images)
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            try:
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        image_files.append(file)
            except Exception as e:
                print(f"Error scanning images directory: {e}")
        
        print(f"Total files found: {len(html_files)} HTML, {len(image_files)} images")
        
        if not html_files and not image_files:
            self._styled_msgbox(QMessageBox.Information, self, "Info", 
                f"No translated files found in: {output_dir}\n\n"
                f"Progress tracking: {'Yes' if has_progress_tracking else 'No'}")
            return
        
        # Create dialog
        dialog = QDialog(self)
        self._stamp_progress_manager_input_signature(dialog)
        dialog.setWindowTitle("Progress Manager - Images")
        # Parent-child windowing keeps this above the translator GUI
        dialog.setWindowModality(Qt.NonModal)
        # Decreased width to 18%, increased height to 25% for better vertical space
        width, height = self._get_dialog_size(0.18, 0.25)
        dialog.resize(width, height)
        
        # Set icon
        try:
            from PySide6.QtGui import QIcon
            if hasattr(self, 'base_dir'):
                base_dir = self.base_dir
            else:
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                dialog.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Failed to load icon: {e}")
        
        dialog_layout = QVBoxLayout(dialog)
        
        # Create listbox (QListWidget has built-in scrolling)
        listbox = QListWidget()
        listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self._apply_compact_inline_list_style(listbox)
        # Use 16% of screen width (half of original ~31% for 1920px screen)
        min_width, _ = self._get_dialog_size(0.16, 0)
        listbox.setMinimumWidth(min_width)
        dialog_layout.addWidget(listbox)
        
        # Keep track of file info
        file_info = []
        
        progress_data_current = progress_data
        
        # Add translated HTML files
        for html_file in sorted(set(html_files)):  # Use set to avoid duplicates
            display_name = os.path.basename(html_file)
            # Extract original image name from HTML filename
            # Expected format: response_001_imagename.html
            match = re.match(r'response_(\d+)_(.+)\.html', display_name)
            if match:
                index = match.group(1)
                base_name = match.group(2)
                display = f"📄 Image {index} | {base_name} | ✅ Completed"
            else:
                display = f"📄 {display_name} | ✅ Completed"
            
            self._add_compact_inline_list_item(listbox, display)
            
            # Find the hash key for this file if progress tracking exists
            hash_key = None
            if progress_data_current:
                for key, value in progress_data_current.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        outp = str(value.get('output_file') or '')
                        if html_file == outp or display_name == os.path.basename(outp) or html_file in outp:
                            hash_key = key
                            break
            
            # Build absolute path (preserve subfolders if present)
            if os.path.isabs(html_file):
                abs_path = html_file
            else:
                abs_path = os.path.join(output_dir, html_file)
            file_info.append({
                'type': 'translated',
                'file': html_file,  # may include subfolders relative to output_dir
                'path': abs_path,
                'hash_key': hash_key,
                'output_dir': output_dir  # Store for later use
            })
        
        # Add cover images
        for img_file in sorted(image_files):
            display = f"🖼️ Cover | {img_file} | ⏭️ Skipped (cover)"
            self._add_compact_inline_list_item(listbox, display)
            file_info.append({
                'type': 'cover',
                'file': img_file,
                'path': os.path.join(images_dir, img_file),
                'hash_key': None,
                'output_dir': output_dir
            })
        
        # Selection count label
        selection_count_label = QLabel("Selected: 0")
        selection_font = QFont('Arial', 10)
        selection_count_label.setFont(selection_font)
        dialog_layout.addWidget(selection_count_label)
        
        def update_selection_count():
            count = len(listbox.selectedItems())
            selection_count_label.setText(f"Selected: {count}")
        
        listbox.itemSelectionChanged.connect(update_selection_count)

        # ==== Context menu for image list ====
        def _open_file_for_index(idx):
            info_list = refresh_data.get('file_info', file_info)
            if idx < 0 or idx >= len(info_list):
                return
            info = info_list[idx]
            path = info.get('path')
            if not path or not os.path.exists(path):
                self._show_message('error', "File Missing", f"File not found:\n{path}", parent=dialog)
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=dialog)

        def _show_context_menu(pos):
            item = listbox.itemAt(pos)
            if not item:
                return
            row = listbox.row(item)
            menu = QMenu(listbox)
            menu.setStyleSheet(
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
            act_open = menu.addAction("📂 Open File")
            act_delete = menu.addAction("🔁 Delete / Retranslate")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_index(row)
            elif chosen == act_delete:
                retranslate_selected()

        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        listbox.customContextMenuRequested.connect(_show_context_menu)
        
        # Button frame
        button_frame = QWidget()
        button_layout = QGridLayout(button_frame)
        dialog_layout.addWidget(button_frame)
        
        def select_all():
            listbox.selectAll()
            update_selection_count()
        
        def clear_selection():
            listbox.clearSelection()
            update_selection_count()
        
        def select_translated():
            listbox.clearSelection()
            info_list = refresh_data.get('file_info', file_info)
            for idx, info in enumerate(info_list):
                if info['type'] == 'translated':
                    listbox.item(idx).setSelected(True)
            update_selection_count()
        
        def mark_as_skipped():
            """Move selected images to the images folder to be skipped"""
            selected_items = listbox.selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, dialog, "No Selection", "Please select at least one image to mark as skipped.")
                return
            
            # Get all selected items
            selected_indices = [listbox.row(item) for item in selected_items]
            info_list = refresh_data.get('file_info', file_info)
            items_with_info = [(i, info_list[i]) for i in selected_indices]
            progress_data_current = refresh_data.get('progress_data', progress_data)
            
            # Filter out items already in images folder (covers)
            items_to_move = [(i, item) for i, item in items_with_info if item['type'] != 'cover']
            
            if not items_to_move:
                self._styled_msgbox(QMessageBox.Information, dialog, "Info", "Selected items are already in the images folder (skipped).")
                return
            
            count = len(items_to_move)
            reply = self._styled_msgbox(QMessageBox.Question, dialog, "Confirm Mark as Skipped", 
                                      f"Move {count} translated image(s) to the images folder?\n\n"
                                      "This will:\n"
                                      "• Delete the translated HTML files\n"
                                      "• Copy source images to the images folder\n"
                                      "• Skip these images in future translations",
                                      QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            result = mark_image_folder_items_skipped(
                folder_path, output_dir, progress_file, progress_data_current, items_to_move, info_list,
            )
            for idx, display in result['displays'].items():
                listbox.item(idx).setText(display)
            if result['moved']:
                refresh_data['file_info'] = info_list
            
            # Auto-refresh the display to show updated status
            if 'refresh_data' in locals():
                self._refresh_image_folder_data(refresh_data)
            
            # Update selection count
            update_selection_count()
            
            # Show result
            result_kind, result_title, result_message = image_folder_mark_skipped_message(result)
            self._styled_msgbox(
                QMessageBox.Warning if result_kind == 'warning' else QMessageBox.Information,
                dialog, result_title, result_message)
        
        def retranslate_selected():
            selected_items = listbox.selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, dialog, "No Selection", "Please select at least one file.")
                return
            
            selected_indices = [listbox.row(item) for item in selected_items]
            info_list = refresh_data.get('file_info', file_info)
            progress_data_current = refresh_data.get('progress_data', progress_data)
            
            confirm_msg = image_folder_delete_confirmation(info_list, selected_indices)
            
            reply = self._styled_msgbox(QMessageBox.Question, dialog, "Confirm Deletion", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            deleted_count = delete_image_folder_items(
                progress_file, progress_data_current, info_list, selected_indices
            )
            
            # Auto-refresh the display to show updated status
            if 'refresh_data' in locals():
                self._refresh_image_folder_data(refresh_data)
            
            result_kind, result_title, result_message = image_folder_delete_message(deleted_count)
            self._styled_msgbox(QMessageBox.Information, dialog, result_title, result_message)

            dialog.close()
        
        # Add buttons in grid layout (similar to EPUB/text retranslation)
        # Row 0: Selection buttons
        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_select_all.clicked.connect(select_all)
        button_layout.addWidget(btn_select_all, 0, 0)
        
        btn_clear_selection = QPushButton("Clear Selection")
        btn_clear_selection.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_clear_selection.clicked.connect(clear_selection)
        button_layout.addWidget(btn_clear_selection, 0, 1)
        
        btn_select_translated = QPushButton("Select Translated")
        btn_select_translated.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_select_translated.clicked.connect(select_translated)
        button_layout.addWidget(btn_select_translated, 0, 2)
        
        btn_mark_skipped = QPushButton("Mark as Skipped")
        btn_mark_skipped.setStyleSheet("QPushButton { background-color: #e0a800; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_mark_skipped.clicked.connect(mark_as_skipped)
        button_layout.addWidget(btn_mark_skipped, 0, 3)
        
        # Row 1: Action buttons
        btn_delete = QPushButton("Delete Selected")
        btn_delete.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_delete.clicked.connect(retranslate_selected)
        button_layout.addWidget(btn_delete, 1, 0, 1, 1)
        
        # Add animated refresh button
        btn_refresh = AnimatedRefreshButton("  Refresh")  # Double space for icon padding
        btn_refresh.setStyleSheet(
            "QPushButton { "
            "background-color: #17a2b8; "
            "color: white; "
            "padding: 5px 15px; "
            "font-weight: bold; "
            "}"
            "QPushButton[refreshActive=\"true\"] { "
            "background-color: #138496; "
            "}"
        )
        
        # Create data dict for refresh function
        refresh_data = {
            'type': 'image_folder',
            'listbox': listbox,
            'file_info': file_info,
            'progress_file': progress_file,
            'progress_data': progress_data,
            'output_dir': output_dir,
            'folder_path': folder_path,
            'selection_count_label': selection_count_label,
            'dialog': dialog
        }
        
        # Create refresh handler with animation
        def animated_refresh():
            import time
            btn_refresh.start_animation()
            btn_refresh.setEnabled(False)
            
            # Track start time for minimum animation duration
            start_time = time.time()
            min_animation_duration = 0.8  # 800ms minimum
            
            # Use QTimer to run refresh after animation starts
            def do_refresh():
                try:
                    self._refresh_image_folder_data(refresh_data)
                    
                    # Calculate remaining time to meet minimum animation duration
                    elapsed = time.time() - start_time
                    remaining = max(0, min_animation_duration - elapsed)
                    
                    # Schedule animation stop after remaining time
                    def finish_animation():
                        btn_refresh.stop_animation()
                        btn_refresh.setEnabled(True)
                    
                    if remaining > 0:
                        QTimer.singleShot(int(remaining * 1000), finish_animation)
                    else:
                        finish_animation()
                        
                except Exception as e:
                    print(f"Error during refresh: {e}")
                    btn_refresh.stop_animation()
                    btn_refresh.setEnabled(True)
            
            QTimer.singleShot(50, do_refresh)  # Small delay to let animation start
        
        btn_refresh.clicked.connect(animated_refresh)
        button_layout.addWidget(btn_refresh, 1, 1, 1, 1)
        # Store for reuse-trigger
        dialog._refresh_button = btn_refresh

        # Auto-refresh every 3 seconds (silent, no animation)
        def _silent_refresh_images():
            try:
                # Skip if a manual refresh is already in progress
                if not btn_refresh.isEnabled():
                    return
                if dialog.isVisible():
                    self._refresh_image_folder_data(refresh_data)
            except Exception:
                pass

        _auto_refresh_timer = QTimer(dialog)
        _auto_refresh_timer.setInterval(1000)
        _auto_refresh_timer.timeout.connect(_silent_refresh_images)
        _auto_refresh_timer.start()
        dialog._auto_refresh_timer = _auto_refresh_timer

        # Force-refresh whenever the cached dialog is re-shown (close only
        # hides it), instead of waiting for the next timer tick.
        _original_img_show_event = dialog.showEvent

        def _img_show_event(event, _orig=_original_img_show_event):
            try:
                _orig(event)
            except Exception:
                pass
            try:
                QTimer.singleShot(0, _silent_refresh_images)
            except Exception:
                pass

        dialog.showEvent = _img_show_event
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; font-weight: bold; }")
        btn_cancel.clicked.connect(dialog.close)
        button_layout.addWidget(btn_cancel, 1, 2, 1, 2)
        
        # Override close event to hide instead of destroy
        def closeEvent(event):
            event.ignore()  # Ignore the close event
            dialog.hide()   # Just hide the dialog
        
        dialog.closeEvent = closeEvent
        
        # Cache the dialog for reuse
        if not hasattr(self, '_image_retranslation_dialog_cache'):
            self._image_retranslation_dialog_cache = {}
        
        folder_key = os.path.abspath(folder_path)
        self._image_retranslation_dialog_cache[folder_key] = dialog
        
        # Programmatically click the Refresh button once on open to ensure latest data (fires same slot)
        QTimer.singleShot(0, btn_refresh.click)

        # Show the dialog (non-modal to allow interaction with other windows)
        dialog.show()
