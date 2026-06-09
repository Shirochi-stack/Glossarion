"""
Retranslation GUI Module
Force retranslation functionality for EPUB, text, and image files
"""

import os
import sys
import json
import re
import html as html_lib
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from difflib import SequenceMatcher
from urllib.parse import unquote
from PySide6.QtWidgets import (QWidget, QDialog, QLabel, QFrame, QListWidget, 
                                QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QMessageBox, QFileDialog, QTabWidget, QListWidgetItem,
                                QScrollArea, QSizePolicy, QMenu, QAbstractItemView,
                                QPlainTextEdit, QStackedWidget, QComboBox, QInputDialog,
                                QLineEdit)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property, QEventLoop, QUrl, QItemSelectionModel, QSize, QPoint, QEvent
from PySide6.QtGui import QFont, QColor, QTransform, QIcon, QPixmap, QDesktopServices, QPalette, QKeySequence, QShortcut
import xml.etree.ElementTree as ET
import zipfile
import shutil
import traceback
import subprocess
import platform
import time
import threading
import hashlib
import unicodedata

_IS_MACOS = (sys.platform == 'darwin')
_MACHINE_TRANSLATION_DIR = "Machine_Translation"


def _sdlxliff_machine_translation_output_name(path_or_name):
    name = os.path.basename(str(path_or_name or "").replace("\\", "/"))
    suffix = ".sdlxliff"
    if name.lower().endswith(suffix):
        name = name[:-len(suffix)]
    return name


def _sdlxliff_machine_translation_path(output_dir, output_or_sidecar):
    output_name = _sdlxliff_machine_translation_output_name(output_or_sidecar)
    if not output_name:
        return None
    base_dir = str(output_dir or "")
    if not base_dir:
        sidecar_path = str(output_or_sidecar or "")
        sidecar_dir = os.path.dirname(sidecar_path)
        if os.path.basename(sidecar_dir).lower() == "sdlxliff":
            base_dir = os.path.dirname(sidecar_dir)
    if not base_dir:
        return None
    safe_name = re.sub(r'[^A-Za-z0-9._ -]+', '_', output_name).strip(" .")
    if not safe_name:
        safe_name = hashlib.sha256(output_name.encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(base_dir, "SDLXLIFF", _MACHINE_TRANSLATION_DIR, f"{safe_name}.json")

def _get_app_dir() -> str:
    """Return the application's base directory (Windows-safe)."""
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


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


class SDLXLIFFReviewDialog(QDialog):
    """Internal source/output reviewer for generated SDLXLIFF HTML sidecars."""

    _tooltip_translation_finished = Signal(int, object, str)
    _tooltip_translation_progress = Signal(int, int)
    _tooltip_translation_status = Signal(int, object, str)
    _tooltip_translation_batch_finished = Signal(int, int, str)
    _review_data_preload_finished = Signal(int, object)
    _review_refresh_scan_finished = Signal(int, object)

    TEXT_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p")
    THEME = {
        "bg": "#1e1e1e",
        "panel": "#2d2d2d",
        "panel_alt": "#242424",
        "border": "#4a5568",
        "accent": "#5a9fd4",
        "info": "#17a2b8",
        "success": "#28a745",
        "warning": "#d39e00",
        "purple": "#b967ff",
        "danger": "#dc3545",
        "text": "#ffffff",
        "muted": "#94a3b8",
    }
    REVIEW_ROW_MIN_HEIGHT = 96
    REVIEW_ROW_MAX_HEIGHT = 1600
    REVIEW_PRELOAD_RADIUS = 2
    REVIEW_PRELOAD_BATCH_SIZE = 8
    REVIEW_PRELOAD_IDLE_MS = 350
    REVIEW_PRELOAD_STEP_MS = 90
    REVIEW_MAX_CACHED_PAGES = 7
    REVIEW_SYNC_RENDER_ROW_LIMIT = 80
    TRANSLATE_TOOLTIPS_BUTTON_TEXT = "🌐 Generate Machine Translation Preview"
    FLAG_ACCURACY_BUTTON_TEXT = "🟣 Flag Inaccurate"
    TWO_COLUMN_LAYOUT_BUTTON_TEXT = "Compact"
    TWO_COLUMN_LAYOUT_CONFIG_KEY = "sdlxliff_two_column_layout"
    LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY = "sdlxliff_one_column_layout"
    LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY = "sdlxliff_one_row_layout"
    MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY = "sdlxliff_machine_translation_provider"
    MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY = "sdlxliff_machine_translation_deepl_api_key"
    MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY = "sdlxliff_machine_translation_bing_api_key"
    MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY = "sdlxliff_machine_translation_bing_region"
    MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY = "sdlxliff_machine_translation_yandex_api_key"
    MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY = "sdlxliff_machine_translation_yandex_folder_id"
    MACHINE_TRANSLATION_PROVIDER_LABELS = {
        "auto": "Auto",
        "google": "Google",
        "deepl": "DeepL",
        "bing": "Bing",
        "argos": "Argos Translate",
        "yandex": "Yandex",
    }
    MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY = "sdlxliff_machine_translation_inaccuracy_threshold"
    MACHINE_TRANSLATION_INACCURACY_THRESHOLD = 150.0
    MACHINE_TRANSLATION_SHORT_TEXT_MAX_TOKENS = 1
    MACHINE_TRANSLATION_SHORT_TEXT_MAX_CHARS = 24
    MACHINE_TRANSLATION_CONTENT_STOPWORDS = frozenset("""
        a an and are as at be been being but by for from had has have he her hers him his i if in into is it
        its me my of on or our ours she so some that the their theirs them then there they this to was were
        while who whom whose will with would you your yours
    """.split())
    MACHINE_TRANSLATION_PENDING_TEXT = "⏳ Generating machine translation preview..."
    MANUAL_REFRESH_BUTTON_TEXT = "↻ Refresh"
    _SDLXLIFF_AUTOGEN_STATUSES = {
        "completed",
        "qa_failed",
        "completed_empty",
        "completed_image_only",
    }

    def __init__(self, output_dir, current_path=None, parent=None, config=None, autogen_owner=None):
        super().__init__(parent)
        self.output_dir = output_dir
        self.current_path = os.path.abspath(current_path) if current_path else ""
        parent_config = getattr(parent, "config", None)
        self._config = config if isinstance(config, dict) else (parent_config if isinstance(parent_config, dict) else {})
        self._context_parent = parent
        self._sdlxliff_autogen_owner = autogen_owner
        self._last_autogen_signature = None
        self._book_entries = self._discover_review_books(parent)
        self._book_index = self._initial_review_book_index()
        if self._book_entries:
            current_book = self._book_entries[self._book_index]
            self.output_dir = current_book.get("output_dir") or self.output_dir
            self.current_path = current_book.get("current_path") or self.current_path
        self.pieces = []
        self._pending_target_edits = {}
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
        self._two_column_layout_enabled = self._review_two_column_layout_enabled()
        self._auto_refresh_timer = None
        self._refreshing_review_data = False
        self._review_data_loaded = False
        self._initial_review_load_started = False
        self._queued_review_refresh = False
        self._review_refresh_scan_token = 0
        self._review_refresh_scan_running = False
        self._review_refresh_scan_requested = False
        self._review_refresh_scan_queued = False
        self._review_refresh_scan_force = False
        self._review_refresh_scan_current_path = None
        self._seamless_review_old_page = None
        self._review_context_menu_open = False
        self._review_text_context_menu = None
        self._piece_list_context_menu = None
        self._machine_translation_provider_menu = None
        self._manual_refresh_shortcut = None
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
        self.setWindowTitle("SDLXLIFF Source -> Output Review - Credits: OMORIO")
        self.setObjectName("SDLXLIFFReviewDialog")
        self.setWindowModality(Qt.NonModal)
        self.resize(1500, 900)
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
        self.two_column_layout_btn.setToolTip("Show text in the first column and row actions/status markers in the second column.")
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

    def showEvent(self, event):
        super().showEvent(event)
        if self._first_show_render_started:
            self._start_review_auto_refresh()
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
            self._queue_review_refresh(force=False, current_path=self.current_path, delay_ms=25)
            return
        if self.pieces:
            QTimer.singleShot(0, lambda: self._render_piece(self._initial_piece_row))

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

    def _queue_review_refresh_scan(self, force=False, current_path=None, delay_ms=350):
        try:
            self._review_refresh_scan_force = bool(getattr(self, "_review_refresh_scan_force", False) or force)
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
            current_path = getattr(self, "_review_refresh_scan_current_path", None) or self.current_path
            self._review_refresh_scan_force = False
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
                        current_path=current_path,
                        last_review_signature=last_review_signature,
                        last_mt_signature=last_mt_signature,
                        last_autogen_signature=last_autogen_signature,
                    ),
                )

            threading.Thread(target=_worker, name="sdlxliff-review-refresh-scan", daemon=True).start()
        except Exception:
            self._review_refresh_scan_running = False
            self._queue_stop_refresh_button_animation(150)

    def _build_review_refresh_scan_result(
        self,
        force=False,
        current_path=None,
        last_review_signature=None,
        last_mt_signature=None,
        last_autogen_signature=None,
    ):
        result = {
            "force": bool(force),
            "current_path": current_path,
            "review_signature": last_review_signature,
            "machine_translation_signature": last_mt_signature,
            "autogen_signature": last_autogen_signature,
            "sidecar_changed": False,
            "machine_translation_changed": False,
            "autogen_changed": False,
            "sidecars_generated": False,
            "stats": None,
            "error": "",
        }
        try:
            review_signature = self._current_review_signature()
            mt_signature = self._current_machine_translation_signature()
            autogen_signature = self._current_review_autogen_signature()
            sidecar_changed = review_signature != last_review_signature
            mt_changed = mt_signature != last_mt_signature
            autogen_changed = autogen_signature != last_autogen_signature
            stats = None
            generated = False
            if force or sidecar_changed or autogen_changed:
                stats = self._regenerate_review_sidecars_for_refresh_scan(
                    force=force,
                    previous_signature=last_autogen_signature,
                    current_signature=autogen_signature,
                )
                generated = bool(stats and (stats.get("created") or stats.get("paths")))
                if force or generated:
                    review_signature = self._current_review_signature()
                    mt_signature = self._current_machine_translation_signature()
                    autogen_signature = self._current_review_autogen_signature()
                    sidecar_changed = True
                    mt_changed = mt_signature != last_mt_signature
                    autogen_changed = autogen_signature != last_autogen_signature
            result.update({
                "review_signature": review_signature,
                "machine_translation_signature": mt_signature,
                "autogen_signature": autogen_signature,
                "sidecar_changed": sidecar_changed,
                "machine_translation_changed": mt_changed,
                "autogen_changed": autogen_changed,
                "sidecars_generated": generated,
                "stats": stats,
            })
        except Exception as exc:
            result["error"] = str(exc)
        return result

    def _regenerate_review_sidecars_for_refresh_scan(self, force=False, previous_signature=None, current_signature=None):
        signature = current_signature or ()
        missing_outputs = self._missing_review_sidecar_outputs(self.output_dir, signature)
        invalid_outputs = []
        if force or previous_signature is not None or missing_outputs:
            invalid_outputs = self._invalid_review_sidecar_outputs(self.output_dir)
        if not force and signature == previous_signature and not invalid_outputs and not missing_outputs:
            return None

        owner = getattr(self, "_sdlxliff_autogen_owner", None)
        generator = getattr(owner, "_generate_sdlxliff_sidecars_from_completed_entries", None)
        if not callable(generator):
            return None

        file_path = None
        try:
            if 0 <= self._book_index < len(self._book_entries):
                file_path = self._book_entries[self._book_index].get("epub_path") or None
        except Exception:
            file_path = None

        output_files = None
        if not force:
            changed_outputs = self._changed_review_autogen_outputs(previous_signature, signature)
            output_files = sorted(set((changed_outputs or []) + (invalid_outputs or []) + (missing_outputs or []))) or None
            if not output_files:
                return None

        return generator(
            self.output_dir,
            file_path=file_path,
            progress_data=None,
            output_files=output_files,
            overwrite=True,
        )

    def _apply_review_refresh_scan(self, token, result):
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
            if (
                result.get("force")
                or result.get("sidecar_changed")
                or result.get("autogen_changed")
                or result.get("sidecars_generated")
            ):
                self.refresh_review_data(
                    force=False,
                    current_path=result.get("current_path") or self.current_path,
                    signature=signature,
                    seamless=True,
                    skip_autogen=True,
                    autogen_signature=autogen_signature,
                    mt_signature=mt_signature,
                )
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
            self._review_refresh_scan_running = False
            self._queue_stop_refresh_button_animation(150)
            if getattr(self, "_review_refresh_scan_requested", False):
                force = bool(getattr(self, "_review_refresh_scan_force", False))
                current_path = getattr(self, "_review_refresh_scan_current_path", None) or self.current_path
                self._review_refresh_scan_requested = False
                self._queue_review_refresh_scan(force=force, current_path=current_path, delay_ms=350)

    def _current_review_signature(self):
        signature = []
        for output_dir in self._review_output_dirs():
            for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
                try:
                    stat = os.stat(path)
                    signature.append(("sdlxliff", os.path.normcase(os.path.abspath(path)), stat.st_size, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))))
                except Exception:
                    signature.append(("sdlxliff", os.path.normcase(os.path.abspath(path)), -1, -1))
        return tuple(sorted(signature))

    def _review_output_dirs(self):
        dirs = []
        if self._book_entries:
            dirs.extend(entry.get("output_dir") for entry in self._book_entries if entry.get("output_dir"))
        elif self.output_dir:
            dirs.append(self.output_dir)
        seen = set()
        output_dirs = []
        for output_dir in dirs:
            try:
                norm_dir = os.path.normcase(os.path.abspath(output_dir))
            except Exception:
                continue
            if norm_dir in seen:
                continue
            seen.add(norm_dir)
            output_dirs.append(output_dir)
        return output_dirs

    def _current_machine_translation_signature(self):
        signature = []
        for output_dir in self._review_output_dirs():
            mt_dir = os.path.join(output_dir, "SDLXLIFF", _MACHINE_TRANSLATION_DIR)
            try:
                mt_dir_norm = os.path.normcase(os.path.abspath(mt_dir))
            except Exception:
                mt_dir_norm = str(mt_dir or "")
            signature.append(("machine_translation_dir",) + self._review_file_signature(mt_dir))
            if os.path.isdir(mt_dir):
                try:
                    for name in sorted(os.listdir(mt_dir)):
                        if not str(name).lower().endswith(".json"):
                            continue
                        mt_path = os.path.join(mt_dir, name)
                        if os.path.isfile(mt_path):
                            signature.append(("machine_translation",) + self._review_file_signature(mt_path))
                except Exception:
                    signature.append(("machine_translation_scan_failed", mt_dir_norm, -1, -1))
        return tuple(sorted(signature))

    @staticmethod
    def _review_file_signature(path):
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            norm = str(path or "")
        try:
            stat = os.stat(path)
            mtime = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))
            return (norm, stat.st_size, mtime)
        except Exception:
            return (norm, -1, -1)

    def _current_review_autogen_signature(self):
        output_dir = self.output_dir or ""
        signature = []
        progress_path = os.path.join(output_dir, "translation_progress.json")
        source_ref_path = os.path.join(output_dir, "source_epub.txt")
        signature.append(("progress",) + self._review_file_signature(progress_path))
        signature.append(("source_epub_ref",) + self._review_file_signature(source_ref_path))

        epub_paths = []
        try:
            if os.path.isfile(source_ref_path):
                with open(source_ref_path, "r", encoding="utf-8", errors="ignore") as f:
                    ref = f.read().strip()
                if ref:
                    epub_paths.append(ref if os.path.isabs(ref) else os.path.join(output_dir, ref))
        except Exception:
            pass
        try:
            if 0 <= self._book_index < len(self._book_entries):
                epub_path = self._book_entries[self._book_index].get("epub_path")
                if epub_path:
                    epub_paths.append(epub_path)
        except Exception:
            pass
        try:
            owner = getattr(self, "_sdlxliff_autogen_owner", None)
            exact_candidates = getattr(owner, "_sdlxliff_exact_input_epub_candidates", None)
            if callable(exact_candidates):
                epub_paths.extend(exact_candidates(output_dir))
        except Exception:
            pass
        seen_epubs = set()
        for epub_path in epub_paths:
            try:
                norm = os.path.normcase(os.path.abspath(epub_path))
            except Exception:
                continue
            if norm in seen_epubs:
                continue
            seen_epubs.add(norm)
            signature.append(("source_epub",) + self._review_file_signature(epub_path))

        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except Exception:
            progress_data = {}
        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}
        if isinstance(chapters, dict):
            for progress_key, entry in chapters.items():
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status", "") or "").lower()
                if status not in self._SDLXLIFF_AUTOGEN_STATUSES:
                    continue
                output_file = entry.get("output_file")
                output_name = os.path.basename(str(output_file or "").replace("\\", "/"))
                if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
                    continue
                normalized = str(output_file).replace("\\", "/")
                output_path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
                signature.append((
                    "output_html",
                    str(progress_key),
                    output_name.lower(),
                    status,
                    str(entry.get("original_basename") or ""),
                    str(entry.get("original_filename") or ""),
                    str(entry.get("chapter_file") or ""),
                    str(entry.get("source_filename") or ""),
                    str(entry.get("filename") or ""),
                ) + self._review_file_signature(os.path.normpath(output_path)))
        return tuple(sorted(signature))

    @staticmethod
    def _changed_review_autogen_outputs(previous_signature, current_signature):
        previous_signature = previous_signature or ()
        current_signature = current_signature or ()

        def _output_rows(signature):
            rows = {}
            for entry in signature:
                if not entry or entry[0] != "output_html":
                    continue
                output_name = entry[2] if len(entry) > 2 else ""
                key = (entry[1] if len(entry) > 1 else "", output_name)
                if output_name:
                    rows[key] = entry
            return rows

        previous = _output_rows(previous_signature)
        current = _output_rows(current_signature)
        changed = []
        for key, entry in current.items():
            if previous.get(key) != entry:
                changed.append(key[1])
        return sorted(set(changed))

    @staticmethod
    def _review_autogen_output_names(autogen_signature):
        output_names = []
        for entry in autogen_signature or ():
            if not entry or entry[0] != "output_html":
                continue
            output_name = entry[2] if len(entry) > 2 else ""
            if output_name:
                output_names.append(output_name)
        return sorted(set(output_names))

    def _missing_review_sidecar_outputs(self, output_dir, autogen_signature):
        expected = self._review_autogen_output_names(autogen_signature)
        if not expected:
            return []
        existing = set()
        for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
            output_name = self._sidecar_output_name(path)
            if output_name:
                existing.add(output_name.lower())
        return sorted(name for name in expected if name.lower() not in existing)

    @staticmethod
    def _review_normalized_unit_text(text):
        return " ".join(str(text or "").split())

    def _sdlxliff_sidecar_needs_source_regeneration(self, path):
        try:
            source_html, target_html = self._read_sdlxliff_html_pair(path)
            source_texts = [
                self._review_normalized_unit_text(unit.get("text"))
                for unit in self._extract_text_units(source_html)
            ]
            target_texts = [
                self._review_normalized_unit_text(unit.get("text"))
                for unit in self._extract_text_units(target_html)
            ]
            source_non_empty = [text for text in source_texts if text]
            target_non_empty = [text for text in target_texts if text]
            if target_non_empty and not source_non_empty:
                return True
            if (
                source_non_empty
                and target_non_empty
                and len(source_non_empty) == len(target_non_empty)
                and source_non_empty == target_non_empty
            ):
                return True
        except Exception:
            return False
        return False

    def _invalid_review_sidecar_outputs(self, output_dir):
        invalid = []
        for path in self._sdlxliff_sidecar_paths_for_output_dir(output_dir):
            if not self._sdlxliff_sidecar_needs_source_regeneration(path):
                continue
            output_name = self._sidecar_output_name(path)
            if output_name:
                invalid.append(output_name)
        return sorted(set(invalid))

    def _invalid_review_sidecar_regen_key(self, output_files, autogen_signature):
        return (
            tuple(sorted(str(name or "").lower() for name in (output_files or []) if name)),
            autogen_signature or (),
            self._current_review_signature(),
        )

    def _maybe_regenerate_review_sidecars(self, force=False):
        previous_signature = getattr(self, "_last_autogen_signature", None)
        try:
            signature = self._current_review_autogen_signature()
        except Exception:
            signature = ()

        missing_outputs = self._missing_review_sidecar_outputs(self.output_dir, signature)
        if not force and previous_signature is None:
            self._last_autogen_signature = signature
            if not missing_outputs:
                return False
            previous_signature = signature
            invalid_outputs = []
        else:
            invalid_outputs = self._invalid_review_sidecar_outputs(self.output_dir)
        invalid_regen_key = None
        if invalid_outputs:
            invalid_regen_key = self._invalid_review_sidecar_regen_key(invalid_outputs, signature)
            if not force and invalid_regen_key == getattr(self, "_last_invalid_sidecar_regen_key", None):
                invalid_outputs = []

        if not force and signature == previous_signature and not invalid_outputs and not missing_outputs:
            return False
        self._last_autogen_signature = signature

        owner = getattr(self, "_sdlxliff_autogen_owner", None)
        generator = getattr(owner, "_generate_sdlxliff_sidecars_from_completed_entries", None)
        if not callable(generator):
            return False

        file_path = None
        try:
            if 0 <= self._book_index < len(self._book_entries):
                file_path = self._book_entries[self._book_index].get("epub_path") or None
        except Exception:
            file_path = None

        output_files = None
        if not force:
            changed_outputs = self._changed_review_autogen_outputs(previous_signature, signature)
            output_files = sorted(set((changed_outputs or []) + (invalid_outputs or []) + (missing_outputs or []))) or None

        try:
            stats = generator(
                self.output_dir,
                file_path=file_path,
                progress_data=None,
                output_files=output_files,
                overwrite=True,
            )
            if invalid_regen_key is not None and invalid_outputs:
                self._last_invalid_sidecar_regen_key = self._invalid_review_sidecar_regen_key(
                    invalid_outputs or output_files,
                    signature,
                )
            return bool(stats and (stats.get("created") or stats.get("paths")))
        except Exception:
            return False
        finally:
            try:
                self._last_autogen_signature = self._current_review_autogen_signature()
            except Exception:
                pass

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

    def _machine_translation_inaccuracy_threshold(self):
        try:
            value = (self._config or {}).get(
                self.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY,
                self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD,
            )
            value = float(value)
            if value <= 0:
                raise ValueError
            return max(1.0, min(1000.0, value))
        except Exception:
            return float(self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD)

    def _review_two_column_layout_enabled(self):
        try:
            config = self._config or {}
            if self.TWO_COLUMN_LAYOUT_CONFIG_KEY in config:
                value = config.get(self.TWO_COLUMN_LAYOUT_CONFIG_KEY)
            elif self.LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY in config:
                value = config.get(self.LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY)
            elif self.LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY in config:
                value = config.get(self.LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY)
            else:
                return True
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        except Exception:
            return True

    def _set_review_two_column_layout(self, enabled):
        enabled = bool(enabled)
        self._two_column_layout_enabled = enabled
        try:
            if getattr(self, "two_column_layout_btn", None) is not None and self.two_column_layout_btn.isChecked() != enabled:
                self.two_column_layout_btn.blockSignals(True)
                self.two_column_layout_btn.setChecked(enabled)
                self.two_column_layout_btn.blockSignals(False)
        except Exception:
            pass
        self._persist_review_config_value(self.TWO_COLUMN_LAYOUT_CONFIG_KEY, enabled)
        try:
            self._cancel_active_review_render()
            self._cancel_review_preload(discard_page=True)
            for row, page in list(self._piece_pages.items()):
                self._discard_piece_page(row, page)
            self._piece_pages.clear()
            self._piece_render_complete.clear()
            for piece in self.pieces:
                if isinstance(piece, dict):
                    piece.pop("_render_model", None)
            self._review_data_preload_token = int(getattr(self, "_review_data_preload_token", 0)) + 1
            current_row = self.piece_list.currentRow() if getattr(self, "piece_list", None) is not None else -1
            if 0 <= current_row < len(self.pieces):
                QTimer.singleShot(0, lambda row=current_row: self._render_piece(row, show_loading=True))
        except Exception:
            pass

    def _persist_review_config_value(self, key, value):
        if isinstance(self._config, dict):
            self._config[key] = value
        parent = getattr(self, "_context_parent", None)
        try:
            parent_config = getattr(parent, "config", None)
            if isinstance(parent_config, dict):
                parent_config[key] = value
        except Exception:
            pass
        try:
            save_config = getattr(parent, "save_config", None)
            if callable(save_config):
                save_config(show_message=False)
                return True
        except TypeError:
            try:
                save_config(getattr(parent, "config", self._config), show_message=False)
                return True
            except Exception:
                pass
        except Exception:
            pass

        config_path = None
        for attr in ("config_file_path", "config_file"):
            try:
                candidate = getattr(parent, attr, None)
            except Exception:
                candidate = None
            if candidate:
                config_path = candidate
                break
        if not config_path:
            config_path = os.path.join(_get_app_dir(), "config.json")
        try:
            config_data = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
            config_data[key] = value
            os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
            tmp_path = f"{config_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, config_path)
            return True
        except Exception:
            return False

    @classmethod
    def _normalize_machine_translation_provider(cls, provider):
        value = str(provider or "auto").strip().lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "": "auto",
            "machine": "auto",
            "machine-translation": "auto",
            "google-free": "google",
            "google-translate": "google",
            "google-translate-free": "google",
            "argos": "argos",
            "argos-translate": "argos",
            "argostranslate": "argos",
            "microsoft": "bing",
            "microsoft-translator": "bing",
            "azure": "bing",
            "azure-translator": "bing",
            "yandex-translate": "yandex",
        }
        value = aliases.get(value, value)
        return value if value in cls.MACHINE_TRANSLATION_PROVIDER_LABELS else "auto"

    def _machine_translation_provider(self):
        try:
            return self._normalize_machine_translation_provider(
                (self._config or {}).get(self.MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY, "auto")
            )
        except Exception:
            return "auto"

    def _machine_translation_provider_label(self, provider=None):
        provider = self._normalize_machine_translation_provider(provider or self._machine_translation_provider())
        return self.MACHINE_TRANSLATION_PROVIDER_LABELS.get(provider, "Auto")

    def _machine_translation_pending_text(self):
        return f"⏳ Translating with {self._machine_translation_provider_label()}..."

    @staticmethod
    def _compact_machine_translation_error(error):
        text = str(error or "").strip()
        if not text:
            return "Machine translation preview failed"
        if "All Google Translate endpoints failed" in text:
            endpoint_lines = [
                line.strip().lstrip("•").strip()
                for line in text.splitlines()
                if "translate" in line and ": " in line
            ]
            reasons = []
            for line in endpoint_lines:
                match = re.match(r"^https?://[^\s]+:\s*(.+)$", line)
                reason = (match.group(1) if match else line.rsplit(": ", 1)[-1]).strip()
                if reason:
                    reasons.append(reason)
            count = len(endpoint_lines) or len(reasons)
            if reasons and len(set(reasons)) == 1:
                reason = reasons[0].replace(": ", " ")
                return f"Google failed: {reason} on {count} endpoints"
            if count:
                return f"Google failed on {count} endpoints; see tooltip for details"
            return "Google failed on all endpoints; see tooltip for details"
        if text.startswith("Auto fell back") and "Google endpoints failed:" in text:
            endpoints = [
                item.strip()
                for item in text.split("Google endpoints failed:", 1)[-1].split(",")
                if item.strip()
            ]
            if endpoints:
                return f"Auto fell back after Google failed on {len(endpoints)} endpoints"
        first_line = text.splitlines()[0].strip()
        if len(first_line) > 180:
            first_line = first_line[:177].rstrip() + "..."
        return first_line

    @classmethod
    def _row_machine_translation_preview_from_snapshot(cls, row):
        row = row if isinstance(row, dict) else {}
        if row.get("tooltip_translation_pending"):
            return str(row.get("tooltip_translation_status") or cls.MACHINE_TRANSLATION_PENDING_TEXT)
        error = str(row.get("tooltip_translation_error") or "").strip()
        if error:
            return error
        return str(row.get("tooltip_translation") or "").strip()

    @staticmethod
    def _row_machine_translation_preview_state(row):
        row = row if isinstance(row, dict) else {}
        if row.get("tooltip_translation_pending"):
            return "pending"
        if str(row.get("tooltip_translation_error") or "").strip():
            return "error"
        if str(row.get("tooltip_translation") or "").strip():
            return "translation"
        return ""

    @staticmethod
    def _machine_translation_result_note(result):
        if not isinstance(result, dict):
            return ""
        note = str(result.get("fallback_note") or "").strip()
        if note:
            return note
        endpoints = result.get("fallback_failed_endpoints")
        if isinstance(endpoints, (list, tuple)) and endpoints:
            return "Auto fell back after Google endpoints failed: " + ", ".join(str(item) for item in endpoints if item)
        return ""

    @staticmethod
    def _append_machine_translation_note(message, note):
        message = str(message or "").strip()
        note = str(note or "").strip()
        if not note:
            return message
        if not message:
            return note
        return f"{message} {note}"

    def _update_machine_translation_button_tooltip(self):
        try:
            provider_label = self._machine_translation_provider_label()
            self.translate_tooltips_btn.setToolTip(
                f"Generate source-row machine translation previews. Provider: {provider_label}. Right-click to choose."
            )
        except Exception:
            pass

    def _set_machine_translation_provider(self, provider):
        provider = self._normalize_machine_translation_provider(provider)
        if provider in {"deepl", "bing", "yandex"} and not self._prompt_machine_translation_credentials(provider):
            return
        saved = self._persist_review_config_value(self.MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY, provider)
        self._update_machine_translation_button_tooltip()
        try:
            suffix = "" if saved else " (not saved to config.json)"
            self.save_status_label.setText(f"Machine translation provider: {self._machine_translation_provider_label(provider)}{suffix}")
        except Exception:
            pass

    def _machine_translation_api_options(self):
        config = self._config or {}
        options = {}
        deepl_key = str(config.get(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY, "") or "").strip()
        if deepl_key:
            options["deepl"] = {"api_key": deepl_key}
        bing_key = str(config.get(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY, "") or "").strip()
        if bing_key:
            bing_options = {"api_key": bing_key}
            region = str(config.get(self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY, "") or "").strip()
            if region:
                bing_options["region"] = region
            options["bing"] = bing_options
        yandex_key = str(config.get(self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY, "") or "").strip()
        yandex_folder_id = str(config.get(self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY, "") or "").strip()
        if yandex_key or yandex_folder_id:
            options["yandex"] = {
                "api_key": yandex_key,
                "folder_id": yandex_folder_id,
            }
        return options

    def _machine_translation_translator(self, target_code, status_callback=None):
        from google_free_translate import GoogleFreeTranslateNew
        return GoogleFreeTranslateNew(
            "auto",
            target_code,
            provider=self._machine_translation_provider(),
            api_keys=self._machine_translation_api_options(),
            honor_global_stop=False,
            endpoint_status_callback=status_callback,
        )

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
        config = self._config or {}
        if provider == "deepl":
            current = str(config.get(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY, "") or "").strip()
            if current and not force:
                return True
            key = self._prompt_secret_text("DeepL API Key", "DeepL API key:", current)
            if not key:
                self.save_status_label.setText("DeepL requires an API key")
                return False
            self._persist_review_config_value(self.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY, key)
            return True
        if provider == "bing":
            current = str(config.get(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY, "") or "").strip()
            if not current or force:
                key = self._prompt_secret_text("Bing / Microsoft Translator API Key", "Microsoft Translator API key:", current)
                if not key:
                    self.save_status_label.setText("Bing requires a Microsoft Translator API key")
                    return False
                self._persist_review_config_value(self.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY, key)
            current_region = str(config.get(self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY, "") or "").strip()
            if force:
                region = self._prompt_plain_text("Bing / Microsoft Translator Region", "Azure region (optional):", current_region)
                if region is not None:
                    self._persist_review_config_value(self.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY, region)
            return True
        if provider == "yandex":
            current_key = str(config.get(self.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY, "") or "").strip()
            current_folder = str(config.get(self.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY, "") or "").strip()
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

    def _set_machine_translation_inaccuracy_threshold(self, threshold):
        try:
            value = float(threshold)
        except Exception:
            value = float(self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD)
        value = max(1.0, min(1000.0, value))
        if abs(value - round(value)) < 0.05:
            value = float(round(value))
        return value, self._persist_review_config_value(self.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY, value)

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
        threshold, saved = self._set_machine_translation_inaccuracy_threshold(value)
        try:
            suffix = "" if saved else " (not saved to config.json)"
            self.save_status_label.setText(f"MT inaccuracy threshold set to {threshold:g}{suffix}")
        except Exception:
            pass

    def _reset_machine_translation_threshold(self):
        threshold, saved = self._set_machine_translation_inaccuracy_threshold(self.MACHINE_TRANSLATION_INACCURACY_THRESHOLD)
        try:
            suffix = "" if saved else " (not saved to config.json)"
            self.save_status_label.setText(f"MT inaccuracy threshold reset to {threshold:g}{suffix}")
        except Exception:
            pass

    def _manual_review_refresh(self):
        self._start_refresh_button_animation()
        self._queue_review_refresh_scan(force=True, current_path=self.current_path, delay_ms=0)

    def _silent_review_refresh(self):
        try:
            if not self.isVisible() or self._refreshing_review_data:
                return
            if not self._review_data_loaded:
                return
            if self._review_context_menu_is_open():
                return
            if self._active_render_timer is not None or self._active_render_page is not None:
                return
            if self._edit_save_timer.isActive() or self._pending_target_edits:
                return
            if self._tooltip_translation_running:
                return
            try:
                from PySide6.QtWidgets import QApplication
                focus = QApplication.focusWidget()
                if focus is not None and focus.window() is self and isinstance(focus, QPlainTextEdit):
                    return
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
        try:
            autogen_changed = False if skip_autogen else self._maybe_regenerate_review_sidecars(force=force)
            if signature is None or autogen_changed or force:
                signature = self._current_review_signature()
                if not force and not autogen_changed and signature == self._last_review_signature:
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
            self.pieces = self._load_pieces(stream_sidebar=not seamless)
            self._review_data_loaded = True
            if not self._streamed_piece_list_populated:
                self._populate_piece_list()
            self._start_review_data_preload()
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
                QTimer.singleShot(0, lambda show_loading=show_loading: self._render_piece(self._initial_piece_row, show_loading=show_loading))
        except Exception:
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
        ):
            self._initial_review_load_started = True
            self._queue_review_refresh(
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

    def _candidate_epub_paths_from_context(self, parent):
        candidates = []

        def _add_many(values):
            if not values:
                return
            for value in values:
                try:
                    path = str(value)
                except Exception:
                    continue
                if (
                    path
                    and (
                        (path.lower().endswith(".epub") and os.path.isfile(path))
                        or RetranslationMixin._sdlxliff_is_extracted_epub_dir(path)
                    )
                ):
                    candidates.append(path)

        widget = parent
        seen_widgets = set()
        while widget is not None and id(widget) not in seen_widgets:
            seen_widgets.add(id(widget))
            _add_many(getattr(widget, "_epub_files_in_dialog", None))
            _add_many(getattr(widget, "selected_files", None))
            try:
                cfg = getattr(widget, "config", None)
                if isinstance(cfg, dict):
                    _add_many(cfg.get("last_input_files"))
                    _add_many(cfg.get("selected_files"))
            except Exception:
                pass
            try:
                widget = widget.parent()
            except Exception:
                widget = None

        if isinstance(self._config, dict):
            _add_many(self._config.get("last_input_files"))
            _add_many(self._config.get("selected_files"))

        seen = set()
        resolved = []
        for path in candidates:
            try:
                norm = os.path.normcase(os.path.abspath(path))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(os.path.abspath(path))
        return resolved

    def _review_output_dirs_for_epub(self, epub_path):
        base = os.path.splitext(os.path.basename(epub_path))[0]
        candidates = []

        def _add(path):
            if path:
                candidates.append(os.path.normpath(path))

        override_dir = os.environ.get("OUTPUT_DIRECTORY") or os.environ.get("OUTPUT_DIR")
        if not override_dir and isinstance(self._config, dict):
            override_dir = self._config.get("output_directory")

        if override_dir:
            _add(os.path.join(override_dir, base))
        else:
            _add(base)

        try:
            current_parent = os.path.dirname(os.path.abspath(self.output_dir))
            if current_parent:
                _add(os.path.join(current_parent, base))
        except Exception:
            pass

        try:
            input_parent = os.path.dirname(os.path.abspath(epub_path))
            _add(os.path.join(input_parent, base))
        except Exception:
            pass

        if _IS_MACOS and candidates:
            mac_candidates = []
            for path in candidates:
                if not os.path.isabs(path):
                    mac_candidates.append(os.path.join(os.path.dirname(os.path.abspath(epub_path)), path))
            candidates.extend(mac_candidates)

        seen = set()
        resolved = []
        for path in candidates:
            try:
                norm = os.path.normcase(os.path.abspath(path))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(path)
        return resolved

    def _sdlxliff_sidecar_paths_for_output_dir(self, output_dir):
        sidecar_dir = os.path.join(output_dir or "", "SDLXLIFF")
        paths = []
        try:
            if os.path.isdir(sidecar_dir):
                for fname in os.listdir(sidecar_dir):
                    if fname.lower().endswith(".sdlxliff"):
                        paths.append(os.path.join(sidecar_dir, fname))
        except Exception:
            return []
        return sorted(paths, key=lambda path: os.path.basename(path).lower())

    @staticmethod
    def _output_dir_has_sdlxliff_sidecars(output_dir):
        sidecar_dir = os.path.join(output_dir or "", "SDLXLIFF")
        try:
            if not os.path.isdir(sidecar_dir):
                return False
            with os.scandir(sidecar_dir) as entries:
                return any(entry.is_file() and entry.name.lower().endswith(".sdlxliff") for entry in entries)
        except Exception:
            return False

    def _discover_review_books(self, parent):
        entries = []
        entries_by_dir = {}
        seen_dirs = set()
        initial_output_dir = os.path.abspath(self.output_dir) if self.output_dir else ""
        initial_current_path = os.path.abspath(self.current_path) if self.current_path else ""

        def _add_entry(output_dir, epub_path=None, current_path=None):
            if not output_dir:
                return
            if not self._output_dir_has_sdlxliff_sidecars(output_dir):
                return
            try:
                abs_dir = os.path.abspath(output_dir)
                norm = os.path.normcase(abs_dir)
            except Exception:
                return
            if norm in seen_dirs:
                existing = entries_by_dir.get(norm)
                if existing is not None:
                    if epub_path and not existing.get("epub_path"):
                        existing["epub_path"] = epub_path
                        existing["label"] = os.path.splitext(os.path.basename(os.path.normpath(epub_path)))[0] or existing.get("label") or "SDLXLIFF"
                    if current_path and os.path.isfile(current_path) and not existing.get("current_path"):
                        existing["current_path"] = current_path
                return
            seen_dirs.add(norm)
            label = os.path.splitext(os.path.basename(os.path.normpath(epub_path)))[0] if epub_path else os.path.basename(abs_dir)
            selected_path = current_path if current_path and os.path.isfile(current_path) else ""
            entry = {
                "epub_path": epub_path or "",
                "output_dir": output_dir,
                "label": label or "SDLXLIFF",
                "current_path": selected_path,
            }
            entries.append(entry)
            entries_by_dir[norm] = entry

        if initial_output_dir:
            _add_entry(initial_output_dir, current_path=initial_current_path)

        for epub_path in self._candidate_epub_paths_from_context(parent):
            for output_dir in self._review_output_dirs_for_epub(epub_path):
                current_path = ""
                try:
                    if initial_current_path:
                        sidecar_root = os.path.join(os.path.abspath(output_dir), "SDLXLIFF")
                        if os.path.commonpath([sidecar_root, initial_current_path]) == sidecar_root:
                            current_path = initial_current_path
                except Exception:
                    current_path = ""
                _add_entry(output_dir, epub_path=epub_path, current_path=current_path)

        entries.sort(key=lambda entry: str(entry.get("label", "")).lower())
        return entries

    def _initial_review_book_index(self):
        if not self._book_entries:
            return 0
        current_dir = os.path.normcase(os.path.abspath(self.output_dir)) if self.output_dir else ""
        current_path = os.path.normcase(os.path.abspath(self.current_path)) if self.current_path else ""
        for index, entry in enumerate(self._book_entries):
            try:
                output_dir = os.path.normcase(os.path.abspath(entry.get("output_dir") or ""))
                if current_dir and output_dir == current_dir:
                    return index
                if current_path:
                    sidecar_dir = os.path.normcase(os.path.abspath(os.path.join(entry.get("output_dir") or "", "SDLXLIFF")))
                    if os.path.commonpath([sidecar_dir, current_path]) == sidecar_dir:
                        return index
            except Exception:
                continue
        return 0

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
        self._queue_review_refresh(force=False, current_path=self.current_path, delay_ms=25)

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
        layout.setSpacing(4)
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
        self.loading_label = loading_label
        loading_layout.addWidget(loading_label)
        loading_layout.addStretch(1)
        return loading_widget

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

    @staticmethod
    def _local_name(tag):
        return str(tag).rsplit("}", 1)[-1].lower()

    @staticmethod
    def _inner_xml_or_text(element):
        if element is None:
            return ""
        if len(list(element)) == 0:
            return element.text or ""
        parts = []
        if element.text:
            parts.append(element.text)
        for child in list(element):
            parts.append(ET.tostring(child, encoding="unicode"))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _read_sdlxliff_html_pair(self, path):
        source_parts = []
        target_parts = []
        root = ET.parse(path).getroot()
        for element in root.iter():
            name = self._local_name(element.tag)
            if name == "source":
                source_parts.append(self._inner_xml_or_text(element))
            elif name == "target":
                target_parts.append(self._inner_xml_or_text(element))
        return "\n".join(source_parts), "\n".join(target_parts)

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

    @staticmethod
    def _review_row_index_property(widget, default=-1):
        try:
            value = widget.property("sdl_row_index")
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    def _jump_to_status(self, status):
        page = self.rows_widget
        if page is None or page is self.loading_page:
            return
        try:
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

    def _extract_text_units(self, html_text):
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return []
        text = html_lib.unescape(str(html_text or ""))
        soup = BeautifulSoup(text, "html.parser")
        units = []
        text_tags = set(self.TEXT_TAGS)

        def _is_text_unit(tag):
            name = str(getattr(tag, "name", "") or "").lower()
            if name in text_tags:
                return True
            if name != "div":
                return False
            classes = tag.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            if "u" not in {str(cls).lower() for cls in classes}:
                return False
            return not tag.find(self.TEXT_TAGS)

        for index, tag in enumerate(soup.find_all(_is_text_unit)):
            value = tag.get_text(" ", strip=True)
            tag_name = tag.name.lower()
            if tag_name == "div":
                tag_name = "p"
            units.append({
                "index": index,
                "tag": tag_name,
                "text": value,
            })
        return units

    @staticmethod
    def _non_empty_text_unit_count(units):
        try:
            return sum(1 for unit in (units or []) if str(unit.get("text", "") or "").strip())
        except Exception:
            return 0

    @staticmethod
    def _non_empty_text_units(units):
        try:
            return [
                unit for unit in (units or [])
                if str(unit.get("text", "") or "").strip()
            ]
        except Exception:
            return []

    @staticmethod
    def _has_linguistic_letters(text):
        return any(ch.isalpha() for ch in str(text or ""))

    def _row_status(self, source_text, target_text, source_missing=False, target_missing=False):
        source_text = str(source_text or "").strip()
        target_text = str(target_text or "").strip()
        if source_missing or target_missing:
            return "red", "dropped/added"
        if not target_text:
            return "red", "empty"
        if source_text and source_text == target_text and self._has_linguistic_letters(source_text):
            return "red", "untranslated"
        if source_text:
            source_len = len(source_text)
            target_len = len(target_text)
            ratio = target_len / max(1, source_len)
            if source_len < 12:
                if target_len > 180:
                    return "yellow", f"density-off ({ratio:.1f}x)"
            elif source_len < 30:
                if target_len > max(220, source_len * 8) or target_len < max(2, int(source_len * 0.10)):
                    return "yellow", f"density-off ({ratio:.1f}x)"
            elif ratio < 0.12 or ratio > 6.0:
                return "yellow", f"density-off ({ratio:.1f}x)"
        return "green", "ok"

    @staticmethod
    def _clear_top_skew_promotions(rows):
        for row in rows or []:
            if row.pop("_top_skew_promoted", False):
                if row.get("_machine_accuracy_promoted"):
                    continue
                row["status"] = "green"
                row["reason"] = "ok"

    @staticmethod
    def _row_expected_comparison_text(row, use_machine_translation=False):
        source_text = str((row or {}).get("source", "") or "").strip()
        if use_machine_translation and not (row or {}).get("tooltip_translation_pending"):
            source_text = str((row or {}).get("tooltip_translation", "") or "").strip() or source_text
        return source_text

    @staticmethod
    def _comparison_tokens(text):
        return [
            token for token in re.findall(r"[a-z0-9']+|[\uac00-\ud7a3]+", str(text or "").lower())
            if re.search(r"[a-z0-9\uac00-\ud7a3]", token)
        ]

    @classmethod
    def _latin_token_overlap(cls, source_text, target_text):
        source_tokens = [
            token for token in cls._comparison_tokens(source_text)
            if re.search(r"[a-z0-9]", token)
        ]
        target_tokens = [
            token for token in cls._comparison_tokens(target_text)
            if re.search(r"[a-z0-9]", token)
        ]
        if not source_tokens or not target_tokens:
            return False, 0.0
        common_tokens = sum((Counter(source_tokens) & Counter(target_tokens)).values())
        return True, common_tokens / max(1, len(source_tokens))

    @classmethod
    def _row_skew_metrics(cls, row, use_machine_translation=False):
        source_text = cls._row_expected_comparison_text(row, use_machine_translation=use_machine_translation)
        target_text = str((row or {}).get("target", "") or "").strip()
        if not source_text or not target_text:
            return None
        if not (row or {}).get("source_tag") or not (row or {}).get("target_tag"):
            return None
        source_len = len(source_text)
        target_len = len(target_text)
        ratio = target_len / max(1, source_len)
        ratio_skew = max(ratio, 1.0 / max(ratio, 0.0001))
        source_tokens = cls._comparison_tokens(source_text)
        target_tokens = cls._comparison_tokens(target_text)
        common_tokens = 0
        if source_tokens and target_tokens:
            common_tokens = sum((Counter(source_tokens) & Counter(target_tokens)).values())
        comparable_token_overlap, source_token_overlap = cls._latin_token_overlap(source_text, target_text)
        unmatched_tokens = max(len(source_tokens), len(target_tokens)) - common_tokens
        similarity = SequenceMatcher(None, source_text.lower(), target_text.lower()).ratio()
        source_weight = max(0.25, min(1.0, source_len / 80.0))
        score = (
            ratio_skew * source_weight
            + min(12.0, abs(target_len - source_len) / 35.0)
            + min(12.0, unmatched_tokens / 2.5)
            + (1.0 - similarity) * 2.0
        )
        return {
            "ratio": ratio,
            "source_len": source_len,
            "source_token_count": len(source_tokens),
            "target_len": target_len,
            "target_token_count": len(target_tokens),
            "comparable_token_overlap": comparable_token_overlap,
            "source_token_overlap": source_token_overlap,
            "score": score,
        }

    @classmethod
    def _row_length_ratio(cls, row, use_machine_translation=False):
        metrics = cls._row_skew_metrics(row, use_machine_translation=use_machine_translation)
        return None if metrics is None else metrics["ratio"]

    def _promote_top_skewed_row_for_count_mismatch(self, rows, source_count, target_count, use_machine_translation=False):
        if source_count == target_count:
            return False
        if not any(row.get("status") == "red" for row in rows or []):
            return False

        best_row = None
        best_ratio = None
        best_score = None
        candidates = []
        for row_index, row in enumerate(rows or []):
            if row.get("status") not in {"green", "yellow"}:
                continue
            metrics = self._row_skew_metrics(row, use_machine_translation=use_machine_translation)
            if metrics is None:
                continue
            candidates.append((row_index, row, metrics))

        if not candidates:
            return False

        target_lengths = sorted(max(1, int(metrics.get("target_len") or 0)) for _index, _row, metrics in candidates)
        target_tokens = sorted(max(1, int(metrics.get("target_token_count") or 0)) for _index, _row, metrics in candidates)
        median_target_len = target_lengths[len(target_lengths) // 2]
        median_target_tokens = target_tokens[len(target_tokens) // 2]
        source_missing_from_output = source_count > target_count
        output_added = target_count > source_count

        for row_index, row, metrics in candidates:
            ratio = metrics["ratio"]
            source_len = max(1, int(metrics.get("source_len") or 0))
            source_token_count = max(1, int(metrics.get("source_token_count") or 0))
            target_len = max(1, int(metrics.get("target_len") or 0))
            target_token_count = max(1, int(metrics.get("target_token_count") or 0))
            target_len_outlier = target_len / max(1, median_target_len)
            target_token_outlier = target_token_count / max(1, median_target_tokens)
            if source_missing_from_output:
                directional_len_delta = max(0, target_len - source_len)
                directional_token_delta = max(0, target_token_count - source_token_count)
                directional_ratio = max(0.0, ratio - 1.0)
                directional_len_ratio = directional_len_delta / max(1, source_len)
                directional_token_ratio = directional_token_delta / max(1, source_token_count)
            elif output_added:
                directional_len_delta = max(0, source_len - target_len)
                directional_token_delta = max(0, source_token_count - target_token_count)
                directional_ratio = max(0.0, (1.0 / max(ratio, 0.0001)) - 1.0)
                directional_len_ratio = directional_len_delta / max(1, target_len)
                directional_token_ratio = directional_token_delta / max(1, target_token_count)
            else:
                directional_len_delta = abs(target_len - source_len)
                directional_token_delta = abs(target_token_count - source_token_count)
                directional_ratio = max(ratio, 1.0 / max(ratio, 0.0001)) - 1.0
                directional_len_ratio = directional_len_delta / max(1, min(source_len, target_len))
                directional_token_ratio = directional_token_delta / max(1, min(source_token_count, target_token_count))
            output_column_weight = (
                target_len_outlier * 6.0
                + target_token_outlier * 4.0
                + min(8.0, target_len / 80.0)
            )
            ratio_multiplier = 1.0 + min(5.0, directional_ratio * 2.0)
            anchor_factor = 1.0
            if metrics.get("comparable_token_overlap") and directional_ratio > 0.0:
                source_token_overlap = float(metrics.get("source_token_overlap") or 0.0)
                anchor_factor = min(1.0, 0.10 + source_token_overlap * 1.20)
            downstream_shift_factor = 1.0
            if source_missing_from_output and metrics.get("comparable_token_overlap") and directional_ratio > 0.0:
                source_token_overlap = float(metrics.get("source_token_overlap") or 0.0)
                next_source_overlap = 0.0
                for next_row in (rows or [])[row_index + 1:row_index + 3]:
                    next_source_text = self._row_expected_comparison_text(
                        next_row,
                        use_machine_translation=use_machine_translation,
                    )
                    comparable, overlap = self._latin_token_overlap(next_source_text, row.get("target"))
                    if comparable:
                        next_source_overlap = max(next_source_overlap, overlap)
                if next_source_overlap >= 0.45 and source_token_overlap < 0.35:
                    downstream_shift_factor = 0.05
            ratio_sensitive_score = (
                min(6.0, directional_len_ratio) * 130.0
                + min(6.0, directional_token_ratio) * 110.0
                + min(6.0, directional_ratio) * 70.0
                + min(240, directional_len_delta) * 0.35
                + min(80, directional_token_delta) * 2.5
                + output_column_weight * ratio_multiplier
            )
            score = (
                ratio_sensitive_score * anchor_factor
                + metrics["score"] * 0.15
            ) * downstream_shift_factor
            if best_score is None or score > best_score:
                best_row = row
                best_ratio = ratio
                best_score = score

        if not best_row or best_score is None or best_row.get("status") != "green":
            return False
        best_row["status"] = "yellow"
        best_row["reason"] = f"top translated-column skew ({best_ratio:.2f}x)"
        best_row["_top_skew_promoted"] = True
        return True

    @staticmethod
    def _clear_machine_accuracy_promotions(rows):
        for row in rows or []:
            if row.pop("_machine_accuracy_promoted", False):
                row["status"] = row.pop("_machine_accuracy_previous_status", "green")
                row["reason"] = row.pop("_machine_accuracy_previous_reason", "ok")
            elif row.get("status") == "purple" and str(row.get("reason") or "").startswith("machine translation inaccurate"):
                row["status"] = "green"
                row["reason"] = "ok"
            row.pop("_machine_accuracy_previous_status", None)
            row.pop("_machine_accuracy_previous_reason", None)

    def _machine_translation_accuracy_score(self, rows, row_index):
        try:
            row = rows[row_index]
        except Exception:
            return None
        if row.get("status") == "red":
            return None
        if row.get("tooltip_translation_pending"):
            return None
        expected = str(row.get("tooltip_translation") or "").strip()
        target = str(row.get("target") or "").strip()
        if not expected or not target:
            return None
        if not row.get("source_tag") or not row.get("target_tag"):
            return None
        expected_normalized = self._normalized_machine_translation_text(expected)
        target_normalized = self._normalized_machine_translation_text(target)
        if expected_normalized == target_normalized:
            return 0.0
        if self._compact_machine_translation_text(expected_normalized) == self._compact_machine_translation_text(target_normalized):
            return 0.0
        expected = expected_normalized
        target = target_normalized

        expected_tokens = self._comparison_tokens(expected)
        target_tokens = self._comparison_tokens(target)
        if self._machine_translation_text_too_short_for_accuracy(expected, target, expected_tokens, target_tokens):
            return 0.0
        if expected_tokens and target_tokens:
            common = sum((Counter(expected_tokens) & Counter(target_tokens)).values())
            token_f1 = (2.0 * common) / max(1, len(expected_tokens) + len(target_tokens))
        else:
            token_f1 = 0.0
        expected_content_tokens = self._machine_translation_content_tokens(expected_tokens)
        target_content_tokens = self._machine_translation_content_tokens(target_tokens)
        content_penalty = 0.0
        if len(expected_content_tokens) >= 4 and len(target_content_tokens) >= 4:
            content_common = sum((Counter(expected_content_tokens) & Counter(target_content_tokens)).values())
            content_f1 = (2.0 * content_common) / max(1, len(expected_content_tokens) + len(target_content_tokens))
            if content_f1 < 0.45:
                content_penalty = (0.45 - content_f1) * 165.0
        similarity = SequenceMatcher(None, expected.lower(), target.lower()).ratio()
        expected_len = max(1, len(expected))
        target_len = max(1, len(target))
        length_ratio = max(expected_len, target_len) / max(1, min(expected_len, target_len))
        expected_token_count = max(1, len(expected_tokens))
        target_token_count = max(1, len(target_tokens))
        token_ratio = max(expected_token_count, target_token_count) / max(1, min(expected_token_count, target_token_count))
        return (
            (1.0 - token_f1) * 120.0
            + (1.0 - similarity) * 70.0
            + min(6.0, length_ratio - 1.0) * 55.0
            + min(6.0, token_ratio - 1.0) * 45.0
            + content_penalty
        )

    @classmethod
    def _machine_translation_text_too_short_for_accuracy(cls, expected, target, expected_tokens=None, target_tokens=None):
        expected_tokens = expected_tokens if expected_tokens is not None else cls._comparison_tokens(expected)
        target_tokens = target_tokens if target_tokens is not None else cls._comparison_tokens(target)
        if not expected_tokens or not target_tokens:
            return False
        if (
            len(expected_tokens) != cls.MACHINE_TRANSLATION_SHORT_TEXT_MAX_TOKENS
            or len(target_tokens) != cls.MACHINE_TRANSLATION_SHORT_TEXT_MAX_TOKENS
        ):
            return False
        expected_len = len(cls._compact_machine_translation_text(expected))
        target_len = len(cls._compact_machine_translation_text(target))
        return max(expected_len, target_len) <= cls.MACHINE_TRANSLATION_SHORT_TEXT_MAX_CHARS

    @classmethod
    def _machine_translation_content_tokens(cls, tokens):
        return [
            token for token in (tokens or [])
            if token not in cls.MACHINE_TRANSLATION_CONTENT_STOPWORDS
        ]

    def _promote_inaccurate_machine_translation_rows(self, piece, threshold=None):
        rows = piece.get("rows") or []
        if not rows:
            return None
        threshold = float(threshold if threshold is not None else self._machine_translation_inaccuracy_threshold())
        self._clear_machine_accuracy_promotions(rows)
        self._clear_top_skew_promotions(rows)

        scored_rows = []
        promoted_indices = []
        for row_index, _row in enumerate(rows):
            score = self._machine_translation_accuracy_score(rows, row_index)
            if score is None:
                continue
            scored_rows.append((row_index, score))
            if score >= threshold:
                promoted_indices.append((row_index, score))

        if not scored_rows:
            piece.pop("_machine_accuracy_review_active", None)
            return None

        piece["_machine_accuracy_review_active"] = True
        for row_index, score in promoted_indices:
            row = rows[row_index]
            row["_machine_accuracy_previous_status"] = row.get("status", "green")
            row["_machine_accuracy_previous_reason"] = row.get("reason", "ok")
            row["_machine_accuracy_promoted"] = True
            row["status"] = "purple"
            row["reason"] = f"machine translation inaccurate ({score:.0f} >= {threshold:.0f})"
        return [row_index for row_index, _score in promoted_indices]

    def _flag_current_piece_inaccurate_translations(self):
        row = self._displayed_piece_row()
        if row < 0 or row >= len(self.pieces):
            return
        self._start_flag_accuracy_button_animation()
        try:
            piece = self.pieces[row]
            rows = piece.get("rows") or []
            before = [
                (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
                for row_data in rows
            ]
            promoted_indices = self._promote_inaccurate_machine_translation_rows(piece)
            if promoted_indices is None:
                try:
                    self.save_status_label.setText("Generate Machine Translation Preview first")
                except Exception:
                    pass
                return

            self._refresh_piece_summary(piece)
            changed_rows = [
                row_index for row_index, row_data in enumerate(rows)
                if row_index >= len(before)
                or before[row_index] != (str(row_data.get("status") or ""), str(row_data.get("reason") or ""))
            ]
            self._invalidate_piece_render_model(piece, restart_preload=False)
            self._refresh_piece_list_item(row)
            self._refresh_piece_header(row)
            for row_index in changed_rows:
                self._refresh_visible_review_row_status(row, row_index)
            try:
                if promoted_indices:
                    self.save_status_label.setText(f"Flagged {len(promoted_indices)} inaccurate machine translation row(s)")
                else:
                    self.save_status_label.setText("No inaccurate machine translation rows found")
            except Exception:
                pass
        finally:
            self._queue_stop_flag_accuracy_button_animation()

    @staticmethod
    def _heading_tag_level_changed(source_tag, target_tag):
        source_tag = str(source_tag or "").strip().lower()
        target_tag = str(target_tag or "").strip().lower()
        if source_tag == target_tag:
            return False
        return bool(re.fullmatch(r"h[1-6]", source_tag) and re.fullmatch(r"h[1-6]", target_tag))

    @staticmethod
    def _heading_paragraph_tag_changed(source_tag, target_tag):
        source_tag = str(source_tag or "").strip().lower()
        target_tag = str(target_tag or "").strip().lower()
        if source_tag == target_tag:
            return False
        source_heading = bool(re.fullmatch(r"h[1-6]", source_tag))
        target_heading = bool(re.fullmatch(r"h[1-6]", target_tag))
        return (source_heading and target_tag == "p") or (source_tag == "p" and target_heading)

    def _tag_mismatch_status(self, source_tag, target_tag):
        if self._heading_tag_level_changed(source_tag, target_tag):
            return "yellow", "heading level changed"
        if self._heading_paragraph_tag_changed(source_tag, target_tag):
            return "yellow", "heading/paragraph tag changed"
        return "red", "tag mismatch"

    @staticmethod
    def _review_unit_is_heading(unit):
        tag = str((unit or {}).get("tag", "") or "").strip().lower()
        return bool(re.fullmatch(r"h[1-6]", tag))

    @staticmethod
    def _review_unit_is_paragraph(unit):
        return str((unit or {}).get("tag", "") or "").strip().lower() == "p"

    def _review_units_are_compatible(self, source_unit, target_unit):
        source_tag = str((source_unit or {}).get("tag", "") or "").strip().lower()
        target_tag = str((target_unit or {}).get("tag", "") or "").strip().lower()
        if source_tag == target_tag:
            return True
        return (
            self._heading_tag_level_changed(source_tag, target_tag)
            or self._heading_paragraph_tag_changed(source_tag, target_tag)
        )

    def _align_review_units(self, source_units, target_units):
        source_units = list(source_units or [])
        target_units = list(target_units or [])
        rows = []
        i = 0
        j = 0
        while i < len(source_units) or j < len(target_units):
            src = source_units[i] if i < len(source_units) else None
            tgt = target_units[j] if j < len(target_units) else None
            if src is None:
                rows.append((None, tgt))
                j += 1
                continue
            if tgt is None:
                rows.append((src, None))
                i += 1
                continue
            next_src = source_units[i + 1] if i + 1 < len(source_units) else None
            next_tgt = target_units[j + 1] if j + 1 < len(target_units) else None
            source_remaining = len(source_units) - i
            target_remaining = len(target_units) - j

            if self._review_unit_is_heading(src) and self._review_unit_is_paragraph(tgt):
                if source_remaining > target_remaining and next_src is not None and self._review_units_are_compatible(next_src, tgt):
                    rows.append((src, None))
                    i += 1
                    continue

            if self._review_unit_is_paragraph(src) and self._review_unit_is_heading(tgt):
                if target_remaining > source_remaining and next_tgt is not None and self._review_units_are_compatible(src, next_tgt):
                    rows.append((None, tgt))
                    j += 1
                    continue

            if self._review_units_are_compatible(src, tgt):
                rows.append((src, tgt))
                i += 1
                j += 1
                continue

            rows.append((src, tgt))
            i += 1
            j += 1
        return rows

    @staticmethod
    def _review_piece_non_empty_count(rows, side):
        text_key = "source" if side == "source" else "target"
        tag_key = "source_tag" if side == "source" else "target_tag"
        try:
            return sum(
                1 for row in (rows or [])
                if row.get(tag_key) and str(row.get(text_key, "") or "").strip()
            )
        except Exception:
            return 0

    def _refresh_piece_summary(self, piece):
        rows = piece.get("rows") or []
        source_count = self._review_piece_non_empty_count(rows, "source")
        target_count = self._review_piece_non_empty_count(rows, "target")
        self._clear_top_skew_promotions(rows)
        manual_accuracy_active = (
            bool(piece.get("_machine_accuracy_review_active"))
            or any(row.get("_machine_accuracy_promoted") for row in rows)
        )
        if not manual_accuracy_active:
            self._promote_top_skewed_row_for_count_mismatch(rows, source_count, target_count)
        red_count = sum(1 for row in rows if row.get("status") == "red")
        yellow_count = sum(1 for row in rows if row.get("status") == "yellow")
        purple_count = sum(1 for row in rows if row.get("status") == "purple")
        piece["source_count"] = source_count
        piece["target_count"] = target_count
        piece["red_count"] = red_count
        piece["yellow_count"] = yellow_count
        piece["purple_count"] = purple_count
        piece["count_ratio"] = (target_count / source_count) if source_count else (1.0 if not target_count else 0.0)
        piece["mismatch"] = source_count != target_count or red_count > 0
        return piece

    @staticmethod
    def _canonical_basename(name):
        return os.path.basename(str(name or "").replace("\\", "/")).lower()

    @staticmethod
    def _canonical_review_path(name):
        text = str(name or "").replace("\\", "/").strip().lower()
        text = text.lstrip("/")
        while text.startswith("./"):
            text = text[2:]
        while text.startswith("../"):
            text = text[3:]
        return text

    @staticmethod
    def _sidecar_output_name(path_or_name):
        sidecar_name = os.path.basename(str(path_or_name or "").replace("\\", "/"))
        suffix = ".sdlxliff"
        if sidecar_name.lower().endswith(suffix):
            return sidecar_name[:-len(suffix)]
        return sidecar_name

    @staticmethod
    def _chapter_number_from_name(name):
        matches = re.findall(r"(\d+)", str(name or ""))
        if not matches:
            return 0
        try:
            return int(matches[-1])
        except Exception:
            return 0

    @staticmethod
    def _format_chapter_number(value):
        try:
            if isinstance(value, float):
                return f"{int(value):03d}" if value.is_integer() else f"{value:06.1f}"
            return f"{int(value):03d}"
        except Exception:
            return "000"

    def _read_progress_metadata(self):
        progress_path = os.path.join(self.output_dir or "", "translation_progress.json")
        output_map = {}
        if not os.path.isfile(progress_path):
            return output_map
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except Exception:
            return output_map

        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}

        for key, entry in chapters.items():
            if not isinstance(entry, dict):
                continue
            output_file = entry.get("output_file")
            if not output_file:
                continue
            normalized = self._canonical_basename(output_file)
            if not normalized:
                continue
            copied = dict(entry)
            copied["_progress_key"] = key
            output_map[normalized] = copied
        return output_map

    def _find_opf_path(self, allow_deep_search=True):
        output_dir = self.output_dir or ""
        if not output_dir or not os.path.isdir(output_dir):
            return None
        direct_candidates = [
            os.path.join(output_dir, "content.opf"),
            os.path.join(output_dir, "OEBPS", "content.opf"),
            os.path.join(output_dir, "EPUB", "content.opf"),
        ]
        for path in direct_candidates:
            if os.path.isfile(path):
                return path
        if not allow_deep_search:
            return None
        try:
            for root_dir, _dirs, files in os.walk(output_dir):
                for fname in files:
                    if fname.lower().endswith(".opf"):
                        return os.path.join(root_dir, fname)
        except Exception:
            return None
        return None

    def _read_spine_positions(self, allow_deep_search=True):
        opf_path = self._find_opf_path(allow_deep_search=allow_deep_search)
        if not opf_path:
            return {}
        try:
            root = ET.parse(opf_path).getroot()
        except Exception:
            return {}

        id_to_href = {}
        for element in root.iter():
            if self._local_name(element.tag) != "item":
                continue
            item_id = element.attrib.get("id")
            href = element.attrib.get("href")
            if item_id and href:
                id_to_href[item_id] = unquote(href)

        spine_positions = {}
        position = 0
        for element in root.iter():
            if self._local_name(element.tag) != "itemref":
                continue
            idref = element.attrib.get("idref")
            href = id_to_href.get(idref)
            if not href:
                continue
            href_key = self._canonical_review_path(href)
            if href_key:
                spine_positions[href_key] = position
                no_ext, _ext = os.path.splitext(href_key)
                if no_ext:
                    spine_positions[no_ext] = position
            href_base = self._canonical_basename(href)
            if not href_base:
                continue
            spine_positions[href_base] = position
            no_ext, _ext = os.path.splitext(href_base)
            if no_ext:
                spine_positions[no_ext] = position
            position += 1
        return spine_positions

    def _original_name_from_output(self, output_name):
        name = os.path.basename(str(output_name or ""))
        if name.lower().startswith("response_"):
            name = name[len("response_"):]
        root, ext = os.path.splitext(name)
        if ext.lower() in (".html", ".htm"):
            return f"{root}.xhtml"
        return name

    def _sidecar_metadata(self, path, fallback_index, progress_map, spine_positions):
        output_name = self._sidecar_output_name(path)
        progress_entry = progress_map.get(self._canonical_basename(output_name), {})
        original_name = (
            progress_entry.get("original_basename")
            or progress_entry.get("original_filename")
            or self._original_name_from_output(output_name)
        )

        opf_position = progress_entry.get("opf_position")
        if opf_position is None:
            opf_position = progress_entry.get("position")
        try:
            opf_position = int(opf_position) if opf_position is not None else None
        except Exception:
            opf_position = None
        for candidate in (original_name, output_name, self._original_name_from_output(output_name)):
            if opf_position is not None:
                break
            normalized_keys = [
                self._canonical_review_path(candidate),
                self._canonical_basename(candidate),
            ]
            for normalized in normalized_keys:
                if not normalized:
                    continue
                if normalized in spine_positions:
                    opf_position = spine_positions[normalized]
                    break
                no_ext, _ext = os.path.splitext(normalized)
                if no_ext in spine_positions:
                    opf_position = spine_positions[no_ext]
                    break
            if opf_position is not None:
                break

        chapter_num = progress_entry.get("actual_num", progress_entry.get("chapter_num"))
        if chapter_num is None:
            chapter_num = self._chapter_number_from_name(output_name)

        sort_position = opf_position if opf_position is not None else 100000 + fallback_index
        metadata = {
            "output_name": output_name,
            "progress_entry": progress_entry,
            "original_name": original_name,
            "opf_position": opf_position,
            "chapter_num": chapter_num,
            "sort_key": (sort_position, self._chapter_number_from_name(output_name), output_name.lower()),
        }
        return metadata

    def _review_label_from_metadata(self, metadata):
        display_position = metadata.get("display_position")
        if display_position is None:
            opf_position = metadata.get("opf_position")
            display_position = (int(opf_position) + 1) if opf_position is not None else 1
        chapter_label = self._format_chapter_number(metadata.get("chapter_num"))
        return f"[{int(display_position):03d}] Ch.{chapter_label} |"

    def _sidebar_label_for_piece(self, piece, row):
        prefix = piece.get("review_label") or f"[{row + 1:03d}] Ch.{self._format_chapter_number(piece.get('chapter_num'))} |"
        return f"{prefix} {piece.get('source_count', 0)} -> {piece.get('target_count', 0)}"

    def _apply_piece_list_item_style(self, item, piece):
        if item is None or not isinstance(piece, dict):
            return
        if piece.get("mismatch"):
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
        item.setToolTip(
            f"{output_name}\nsource {piece['source_count']} -> output {piece['target_count']}\n{piece.get('path', '')}"
        )
        item.setData(Qt.UserRole, row)
        self._apply_piece_list_item_style(item, piece)
        return item

    def _prepare_streaming_piece_list(self, work_items):
        if not work_items:
            return False
        try:
            self.piece_list.currentRowChanged.disconnect(self._request_render_piece)
        except Exception:
            pass
        try:
            self.piece_list.clear()
            self._streaming_piece_selected_path = (
                os.path.normcase(os.path.abspath(self.current_path)) if self.current_path else ""
            )
            self._streaming_piece_selected_row = 0
            self._streaming_piece_visible_count = 0
            self._streaming_piece_last_pump = time.monotonic()
        except Exception:
            return False
        self.piece_list.update()
        self._pump_review_loading_events(max_ms=5)
        return True

    def _stream_piece_list_item(self, original_index, piece):
        try:
            if self._review_piece_is_empty_sidecar(piece):
                return
            visible_row = int(getattr(self, "_streaming_piece_visible_count", 0) or 0)
            item = self._piece_list_item_for_piece(piece, visible_row)
            self.piece_list.addItem(item)
            self._streaming_piece_visible_count = visible_row + 1
            try:
                piece_norm = os.path.normcase(os.path.abspath(piece.get("path") or ""))
                if self._streaming_piece_selected_path and piece_norm == self._streaming_piece_selected_path:
                    self._streaming_piece_selected_row = visible_row
            except Exception:
                pass
            self.piece_list.update()
            last_pump = float(getattr(self, "_streaming_piece_last_pump", 0.0) or 0.0)
            if time.monotonic() - last_pump >= 0.012:
                self._streaming_piece_last_pump = time.monotonic()
                self._pump_review_loading_events(max_ms=2)
        except Exception:
            pass

    def _finish_streaming_piece_list(self):
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
                selected_row = max(0, min(selected_row, self.piece_list.count() - 1))
                previous_block = self.piece_list.blockSignals(True)
                self.piece_list.setCurrentRow(selected_row)
                self.piece_list.blockSignals(previous_block)
                self._initial_piece_row = selected_row
            else:
                self.header_label.setText("No SDLXLIFF review files found")
                try:
                    self.loading_label.setText("No SDLXLIFF review files found")
                except Exception:
                    pass
            self._streamed_piece_list_populated = True
            self.piece_list.update()
            return True
        except Exception:
            return False

    def _build_piece(self, path, index, metadata=None):
        metadata = dict(metadata or {})
        metadata.setdefault("output_name", self._sidecar_output_name(path))
        metadata.setdefault("display_position", index + 1)
        metadata.setdefault("label", self._review_label_from_metadata(metadata))
        try:
            source_html, target_html = self._read_sdlxliff_html_pair(path)
            source_units = self._extract_text_units(source_html)
            target_units = self._extract_text_units(target_html)
            source_review_units = self._non_empty_text_units(source_units)
            target_review_units = self._non_empty_text_units(target_units)
            rows = []
            red_count = 0
            yellow_count = 0
            purple_count = 0
            aligned_units = self._align_review_units(source_review_units, target_review_units)
            for row_idx, (src, tgt) in enumerate(aligned_units):
                status, reason = self._row_status(
                    src.get("text") if src else "",
                    tgt.get("text") if tgt else "",
                    source_missing=src is None,
                    target_missing=tgt is None,
                )
                if src is not None and tgt is not None and src.get("tag") != tgt.get("tag"):
                    status, reason = self._tag_mismatch_status(src.get("tag"), tgt.get("tag"))
                if status == "red":
                    red_count += 1
                elif status == "yellow":
                    yellow_count += 1
                rows.append({
                    "row_index": row_idx,
                    "source_tag": src.get("tag", "") if src else "",
                    "source": src.get("text", "") if src else "",
                    "source_index": src.get("index") if src else None,
                    "target_tag": tgt.get("tag", "") if tgt else "",
                    "target": tgt.get("text", "") if tgt else "",
                    "target_original": tgt.get("text", "") if tgt else "",
                    "target_index": tgt.get("index") if tgt else None,
                    "status": status,
                    "reason": reason,
                })
            source_count = len(source_review_units)
            target_count = len(target_review_units)
            count_ratio = (target_count / source_count) if source_count else (1.0 if not target_count else 0.0)
            piece = {
                "path": path,
                "index": index,
                "name": os.path.basename(path),
                "output_name": metadata.get("output_name"),
                "review_label": metadata.get("label"),
                "opf_position": metadata.get("opf_position"),
                "chapter_num": metadata.get("chapter_num"),
                "source_html": source_html,
                "target_html": target_html,
                "source_count": source_count,
                "target_count": target_count,
                "count_ratio": count_ratio,
                "red_count": red_count,
                "yellow_count": yellow_count,
                "purple_count": purple_count,
                "mismatch": source_count != target_count or red_count > 0,
                "rows": rows,
            }
            self._load_machine_translation_file_for_piece(piece)
            self._refresh_piece_summary(piece)
            return piece
        except Exception as exc:
            return {
                "path": path,
                "index": index,
                "name": os.path.basename(path),
                "output_name": metadata.get("output_name"),
                "review_label": metadata.get("label"),
                "opf_position": metadata.get("opf_position"),
                "chapter_num": metadata.get("chapter_num"),
                "source_count": 0,
                "target_count": 0,
                "count_ratio": 0.0,
                "red_count": 1,
                "yellow_count": 0,
                "purple_count": 0,
                "mismatch": True,
                "error": str(exc),
                "rows": [],
            }

    @staticmethod
    def _review_piece_is_empty_sidecar(piece):
        if not isinstance(piece, dict) or piece.get("error"):
            return False
        return int(piece.get("source_count") or 0) == 0 and int(piece.get("target_count") or 0) == 0

    def _filter_review_pieces(self, pieces):
        filtered = [
            piece for piece in (pieces or [])
            if piece is not None and not self._review_piece_is_empty_sidecar(piece)
        ]
        for index, piece in enumerate(filtered):
            piece["index"] = index
        return filtered

    def _load_pieces(self, stream_sidebar=False):
        sidecar_dir = os.path.join(self.output_dir or "", "SDLXLIFF")
        paths = []
        try:
            if os.path.isdir(sidecar_dir):
                for fname in os.listdir(sidecar_dir):
                    if fname.lower().endswith(".sdlxliff"):
                        paths.append(os.path.join(sidecar_dir, fname))
        except Exception:
            paths = []
        if self.current_path and os.path.isfile(self.current_path):
            current_norm = os.path.normcase(os.path.abspath(self.current_path))
            if all(os.path.normcase(os.path.abspath(path)) != current_norm for path in paths):
                paths.insert(0, self.current_path)

        progress_map = self._read_progress_metadata()
        spine_positions = self._read_spine_positions(allow_deep_search=not stream_sidebar)
        work_items = []
        for fallback_index, path in enumerate(paths):
            metadata = self._sidecar_metadata(path, fallback_index, progress_map, spine_positions)
            work_items.append((path, metadata))
        work_items.sort(key=lambda item: item[1].get("sort_key", (999999, 999999, "")))

        for index, (_path, metadata) in enumerate(work_items):
            if metadata.get("opf_position") is None:
                metadata["display_position"] = index + 1
            else:
                metadata["display_position"] = int(metadata["opf_position"]) + 1
            metadata["label"] = self._review_label_from_metadata(metadata)

        stream_sidebar = bool(stream_sidebar and self.isVisible())
        if stream_sidebar:
            stream_sidebar = self._prepare_streaming_piece_list(work_items)

        def flush_streamed_pieces(limit=None):
            if not stream_sidebar:
                return 0
            flushed = 0
            nonlocal next_stream_index
            while next_stream_index < len(pieces) and pieces[next_stream_index] is not None:
                self._stream_piece_list_item(next_stream_index, pieces[next_stream_index])
                next_stream_index += 1
                flushed += 1
                if limit is not None and flushed >= limit:
                    break
            if flushed:
                self._pump_review_loading_events(max_ms=4)
            return flushed

        pieces = []
        next_stream_index = 0
        if len(work_items) <= 1:
            pieces = [
                self._build_piece(path, idx, metadata)
                for idx, (path, metadata) in enumerate(work_items)
            ]
            if stream_sidebar:
                flush_streamed_pieces()
                self._finish_streaming_piece_list()
            return self._filter_review_pieces(pieces)

        pieces = [None] * len(work_items)
        max_workers = max(1, min(8, len(work_items)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._build_piece, path, idx, metadata): idx
                for idx, (path, metadata) in enumerate(work_items)
            }
            pending = set(futures)
            while pending or (stream_sidebar and next_stream_index < len(pieces)):
                if stream_sidebar and flush_streamed_pieces(limit=12):
                    continue
                if not pending:
                    break
                done, pending = wait(pending, timeout=0.025, return_when=FIRST_COMPLETED)
                if not done:
                    self._pump_review_loading_events(max_ms=4)
                    continue
                for future in done:
                    idx = futures[future]
                    try:
                        pieces[idx] = future.result()
                    except Exception as exc:
                        path, metadata = work_items[idx]
                        pieces[idx] = self._build_piece(path, idx, metadata)
                        pieces[idx]["error"] = str(exc)
                flush_streamed_pieces(limit=12)
                self._pump_review_loading_events(max_ms=4)
        if stream_sidebar:
            flush_streamed_pieces()
            self._finish_streaming_piece_list()
        return self._filter_review_pieces(pieces)

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
            try:
                self.loading_label.setText("No SDLXLIFF review files found")
            except Exception:
                pass

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
            piece = self.pieces[piece_index]
            warning_count = piece.get("yellow_count", 0) + piece.get("purple_count", 0)
            status_text = "MISMATCH" if piece["mismatch"] else ("WARN" if warning_count else "OK")
            flagged = piece["red_count"] + warning_count
            output_name = self._output_name_for_piece(piece)
            review_label = piece.get("review_label") or f"[{piece_index + 1:03d}] Ch.{self._format_chapter_number(piece.get('chapter_num'))} |"
            self.header_label.setText(
                f"{review_label} {output_name}  -  source {piece['source_count']} text units "
                f"- output {piece['target_count']} - {status_text} - {flagged} flagged rows "
                f"(ratio ~= {piece['count_ratio']:.2f})"
            )
        except Exception:
            pass

    def _row_for_piece_path(self, path):
        if not path:
            return None
        try:
            target_norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            return None
        for row, piece in enumerate(self.pieces):
            try:
                piece_norm = os.path.normcase(os.path.abspath(piece.get("path") or ""))
            except Exception:
                continue
            if piece_norm == target_norm:
                return row
        return None

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
            self.loading_label.setText("Loading SDLXLIFF...")
            self.rows_widget = self.loading_page
            self.rows_layout = self.loading_page.layout()
            self._current_scroll_piece_row = None
            if self.rows_stack.currentWidget() is not self.loading_page:
                self.rows_stack.setCurrentWidget(self.loading_page)
            self._sync_review_scroll_range(self.loading_page)
            self.loading_page.show()
            self.loading_page.raise_()
            self.scroll.viewport().update()
            self.rows_stack.update()
            self.rows_stack.repaint()
            self._spin_review_loading_icon()
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents(QEventLoop.AllEvents, 20)
            except Exception:
                pass
        except Exception:
            pass

    def _pump_review_loading_events(self, max_ms=8):
        try:
            self._spin_review_loading_icon()
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents(QEventLoop.AllEvents, max(1, int(max_ms)))
        except Exception:
            pass

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

    def _cancel_active_review_render(self):
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

    def _review_context_menu_is_open(self):
        return bool(getattr(self, "_review_context_menu_open", False))

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

    @staticmethod
    def _review_row_snapshot(row_data):
        row_data = row_data if isinstance(row_data, dict) else {}
        return {
            "source": str(row_data.get("source", "") or ""),
            "target": str(row_data.get("target", "") or ""),
            "source_tag": str(row_data.get("source_tag", "") or ""),
            "target_tag": str(row_data.get("target_tag", "") or ""),
            "status": str(row_data.get("status", "green") or "green"),
            "tooltip_translation": str(row_data.get("tooltip_translation", "") or ""),
            "tooltip_translation_pending": bool(row_data.get("tooltip_translation_pending")),
            "tooltip_translation_status": str(row_data.get("tooltip_translation_status", "") or ""),
            "tooltip_translation_error": str(row_data.get("tooltip_translation_error", "") or ""),
            "tooltip_translation_error_detail": str(row_data.get("tooltip_translation_error_detail", "") or ""),
        }

    @staticmethod
    def _review_wrapped_lines(value, line_chars):
        text = str(value or "")
        if not text:
            return 1
        line_chars = max(1, int(line_chars or 1))
        wrapped_lines = 0
        for part in text.splitlines() or [text]:
            wrapped_lines += max(1, (len(part) + line_chars - 1) // line_chars)
        return wrapped_lines

    @classmethod
    def _review_chars_per_line_for_width(cls, viewport_width=1200, two_column_layout=False):
        viewport_width = max(700, int(viewport_width or 1200))
        fixed_width = 92 + 180 + 26 + 180 + 20 + 50
        if two_column_layout:
            fixed_width = 92 + 250 + 46
        text_column_width = max(180, (viewport_width - fixed_width) // 2)
        if two_column_layout:
            text_column_width = max(320, viewport_width - fixed_width)
        return max(24, int(text_column_width / 9))

    @classmethod
    def _review_row_line_counts_for_width(
        cls,
        source_text,
        target_text,
        tooltip_translation=None,
        tooltip_pending=False,
        viewport_width=1200,
        two_column_layout=False,
        tooltip_preview_text=None,
    ):
        chars_per_line = cls._review_chars_per_line_for_width(
            viewport_width,
            two_column_layout=two_column_layout,
        )
        source_lines = cls._review_wrapped_lines(source_text, chars_per_line)
        target_lines = cls._review_wrapped_lines(target_text, chars_per_line)
        tooltip_preview = (
            str(tooltip_preview_text or "").strip()
            if tooltip_preview_text is not None
            else (cls.MACHINE_TRANSLATION_PENDING_TEXT if tooltip_pending else str(tooltip_translation or "").strip())
        )
        if tooltip_preview:
            translated_chars_per_line = max(30, int(chars_per_line * 1.35))
            tooltip_lines = min(18, cls._review_wrapped_lines(tooltip_preview, translated_chars_per_line))
        else:
            tooltip_lines = 0
        return source_lines, target_lines, tooltip_lines

    @classmethod
    def _review_row_height_for_width(
        cls,
        source_text,
        target_text,
        tooltip_translation=None,
        tooltip_pending=False,
        viewport_width=1200,
        two_column_layout=False,
        tooltip_preview_text=None,
    ):
        source_lines, target_lines, tooltip_lines = cls._review_row_line_counts_for_width(
            source_text,
            target_text,
            tooltip_translation,
            tooltip_pending,
            viewport_width,
            two_column_layout=two_column_layout,
            tooltip_preview_text=tooltip_preview_text,
        )
        max_lines = max(1, source_lines, target_lines)
        tooltip_preview = (
            str(tooltip_preview_text or "").strip()
            if tooltip_preview_text is not None
            else (cls.MACHINE_TRANSLATION_PENDING_TEXT if tooltip_pending else str(tooltip_translation or "").strip())
        )
        if tooltip_lines:
            max_lines = max(max_lines, source_lines + tooltip_lines)
        if two_column_layout:
            source_height = max(34, (source_lines + tooltip_lines) * 22 + (28 if tooltip_lines else 10))
            target_height = max(38, target_lines * 22 + 28)
            text_height = 10 + source_height + 7 + target_height
            controls_height = 3 * 34 + 2 * 5 + 24
            height = min(
                cls.REVIEW_ROW_MAX_HEIGHT,
                max(cls.REVIEW_ROW_MIN_HEIGHT, text_height, controls_height),
            )
            if tooltip_preview:
                height = min(cls.REVIEW_ROW_MAX_HEIGHT, max(height, cls.REVIEW_ROW_MIN_HEIGHT + 40))
            return height
        extra_lines = max(0, min(12, max_lines - 1))
        height = min(cls.REVIEW_ROW_MAX_HEIGHT, cls.REVIEW_ROW_MIN_HEIGHT + extra_lines * 22)
        if tooltip_preview:
            height = min(cls.REVIEW_ROW_MAX_HEIGHT, max(height, cls.REVIEW_ROW_MIN_HEIGHT + 30))
        return height

    @classmethod
    def _build_review_piece_render_model_from_rows(cls, row_snapshots, viewport_width, two_column_layout=False):
        rows = list(row_snapshots or [])
        max_len = max(
            [len(row.get("source", "")) for row in rows]
            + [len(row.get("target", "")) for row in rows]
            + [1]
        )
        row_models = []
        for row in rows:
            source_text = row.get("source", "")
            target_text = row.get("target", "")
            tooltip_translation = row.get("tooltip_translation", "")
            tooltip_pending = bool(row.get("tooltip_translation_pending"))
            tooltip_preview = cls._row_machine_translation_preview_from_snapshot(row)
            tooltip_state = cls._row_machine_translation_preview_state(row)
            source_lines, target_lines, tooltip_lines = cls._review_row_line_counts_for_width(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                viewport_width,
                two_column_layout=two_column_layout,
                tooltip_preview_text=tooltip_preview,
            )
            source_missing = not row.get("source_tag")
            target_missing = not row.get("target_tag")
            row_models.append({
                "source_text": source_text,
                "target_text": target_text,
                "source_len": len(source_text),
                "target_len": len(target_text),
                "source_missing": source_missing,
                "target_missing": target_missing,
                "target_editable": (not source_missing or not target_missing),
                "tooltip_translation": tooltip_translation,
                "tooltip_pending": tooltip_pending,
                "tooltip_preview": tooltip_preview,
                "tooltip_state": tooltip_state,
                "tooltip_detail": str(row.get("tooltip_translation_error_detail") or row.get("tooltip_translation_status") or ""),
                "two_column_layout": bool(two_column_layout),
                "source_lines": source_lines,
                "target_lines": target_lines,
                "tooltip_lines": tooltip_lines,
                "row_height": cls._review_row_height_for_width(
                    source_text,
                    target_text,
                    tooltip_translation,
                    tooltip_pending,
                    viewport_width,
                    two_column_layout=two_column_layout,
                    tooltip_preview_text=tooltip_preview,
                ),
            })
        return {
            "viewport_width": int(max(700, viewport_width or 1200)),
            "two_column_layout": bool(two_column_layout),
            "row_count": len(rows),
            "max_len": max_len,
            "rows": row_models,
        }

    def _piece_render_snapshot(self, piece):
        return [
            self._review_row_snapshot(row_data)
            for row_data in (piece.get("rows") or [])
        ]

    def _review_piece_render_model(self, piece):
        rows = piece.get("rows") or []
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
            self._piece_render_snapshot(piece),
            viewport_width,
            two_column_layout=two_column_layout,
        )
        piece["_render_model"] = model
        return model

    def _invalidate_piece_render_model(self, piece=None, restart_preload=True):
        try:
            if isinstance(piece, dict):
                piece.pop("_render_model", None)
            self._review_data_preload_token = int(getattr(self, "_review_data_preload_token", 0)) + 1
            if restart_preload and getattr(self, "_review_data_loaded", False):
                self._review_data_preload_requested = True
                if not getattr(self, "_review_data_preload_running", False):
                    QTimer.singleShot(250, self._start_review_data_preload)
        except Exception:
            pass

    def _start_review_data_preload(self):
        if not getattr(self, "_review_data_loaded", False) or not self.pieces:
            return
        if getattr(self, "_review_data_preload_running", False):
            self._review_data_preload_requested = True
            return
        self._review_data_preload_token = int(getattr(self, "_review_data_preload_token", 0)) + 1
        token = self._review_data_preload_token
        viewport_width = self._review_render_viewport_width()
        two_column_layout = bool(getattr(self, "_two_column_layout_enabled", True))
        snapshots = [
            (piece_index, self._piece_render_snapshot(piece))
            for piece_index, piece in enumerate(self.pieces)
            if isinstance(piece, dict) and not piece.get("error")
        ]
        if not snapshots:
            return
        self._review_data_preload_running = True
        self._review_data_preload_requested = False

        def _worker():
            models = {}
            try:
                for ordinal, (piece_index, row_snapshot) in enumerate(snapshots, start=1):
                    models[piece_index] = self._build_review_piece_render_model_from_rows(
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
                    self._start_review_data_preload()
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
                self._review_data_preload_requested = False
                QTimer.singleShot(0, self._start_review_data_preload)

    def _review_selection_recently_changed(self):
        try:
            return (time.monotonic() - float(self._last_review_selection_change or 0.0)) < (
                self.REVIEW_PRELOAD_IDLE_MS / 1000.0
            )
        except Exception:
            return False

    def _review_preload_order(self, current_row):
        if not self.pieces:
            return []
        try:
            current_row = int(current_row)
        except Exception:
            current_row = 0
        rows = []
        for distance in range(1, self.REVIEW_PRELOAD_RADIUS + 1):
            for row in (current_row + distance, current_row - distance):
                if (
                    0 <= row < len(self.pieces)
                    and row not in self._piece_render_complete
                    and row not in self._piece_pages
                    and row not in rows
                ):
                    rows.append(row)
        return rows

    def _queue_review_page_preloads(self, current_row):
        try:
            if not self.isVisible() or not self._review_data_loaded:
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

        rows = piece.get("rows") or []
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
        self._refresh_review_stream_geometry(final=final)
        self._set_rows_rebuild_active(False)
        for widget in (self.rows_widget, self.rows_stack, self.scroll.viewport(), self.scroll):
            try:
                widget.update()
            except Exception:
                pass

    def _sync_review_scroll_range(self, page=None):
        try:
            page = page or self.rows_stack.currentWidget()
            if page is None:
                return
            layout = page.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
            page.updateGeometry()
            viewport_height = max(1, self.scroll.viewport().height())
            content_height = max(viewport_height, page.sizeHint().height(), page.minimumSizeHint().height())
            page.setMinimumHeight(content_height)
            page.setMaximumHeight(content_height)
            page.resize(max(1, page.width()), content_height)
            self.rows_stack.setMinimumHeight(content_height)
            self.rows_stack.setMaximumHeight(content_height)
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

    def _tag_label(self, source_tag, target_tag, status):
        source_tag = str(source_tag or "").strip()
        target_tag = str(target_tag or "").strip()
        if source_tag and target_tag:
            text = source_tag if source_tag == target_tag else f"{source_tag} -> {target_tag}"
        elif source_tag:
            text = f"{source_tag} -> missing"
        elif target_tag:
            text = f"+ {target_tag}"
        else:
            text = "-"
        label = QLabel(text)
        label.setObjectName("SdlReviewTagLabel")
        label.setTextFormat(Qt.PlainText)
        label.setAlignment(Qt.AlignCenter)
        label.setFixedWidth(36)
        color = {
            "green": self.THEME["success"],
            "yellow": self.THEME["warning"],
            "purple": self.THEME["purple"],
            "red": self.THEME["danger"],
        }.get(status, self.THEME["muted"])
        label.setStyleSheet(
            f"color: {color}; background: transparent; font: 11pt Consolas, 'Courier New', monospace;"
        )
        return label

    @staticmethod
    def _wrapped_tooltip(text, width=560):
        text = str(text or "").strip()
        if not text:
            return ""
        escaped = html_lib.escape(text).replace("\n", "<br>")
        return f'<div style="white-space: normal; width: {int(width)}px;">{escaped}</div>'

    def _review_status_colors(self):
        return {
            "green": (self.THEME["panel"], self.THEME["accent"], self.THEME["info"], self.THEME["success"], self.THEME["border"]),
            "yellow": ("#3d3320", self.THEME["accent"], self.THEME["warning"], self.THEME["warning"], self.THEME["warning"]),
            "purple": ("#32243f", self.THEME["accent"], self.THEME["purple"], self.THEME["purple"], self.THEME["purple"]),
            "red": ("#3a2428", self.THEME["accent"], self.THEME["danger"], self.THEME["danger"], self.THEME["danger"]),
        }

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
            row_style = f"QFrame#SdlReviewRow {{ background-color: {bg}; border: 1px solid {border_color}; border-radius: 3px; }}"
            frame.setProperty("sdl_status", row_data["status"])
            frame.setProperty("sdl_base_style", row_style)
            frame.setStyleSheet(row_style)

            tag_label = frame.findChild(QLabel, "SdlReviewTagLabel")
            if tag_label is not None:
                tag_color = {
                    "green": self.THEME["success"],
                    "yellow": self.THEME["warning"],
                    "purple": self.THEME["purple"],
                    "red": self.THEME["danger"],
                }.get(row_data["status"], self.THEME["muted"])
                tag_label.setStyleSheet(
                    f"color: {tag_color}; background: transparent; font: 11pt Consolas, 'Courier New', monospace;"
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

    def _inject_machine_translation_to_target(self, piece_index, row_index, translated, editor=None):
        translated = str(translated or "").strip()
        if not translated:
            return
        try:
            if isinstance(editor, QPlainTextEdit) and not editor.isReadOnly():
                self._replace_editor_text_preserving_undo(editor, translated)
                editor.setFocus(Qt.OtherFocusReason)
                return
        except Exception:
            pass
        self._apply_target_edit(piece_index, row_index, translated)

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

    def _undo_all_target_edits(self, piece_index, row_index, editor=None):
        original = ""
        try:
            if 0 <= piece_index < len(self.pieces):
                rows = self.pieces[piece_index].get("rows") or []
                if 0 <= row_index < len(rows):
                    original = str(rows[row_index].get("target_original", "") or "")
        except Exception:
            original = ""
        try:
            if isinstance(editor, QPlainTextEdit) and not editor.isReadOnly():
                self._replace_editor_text_preserving_undo(editor, original)
                editor.setFocus(Qt.OtherFocusReason)
                return
        except Exception:
            pass
        self._apply_target_edit(piece_index, row_index, original)

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
            row_height = self._review_row_height(
                source_text,
                target_text,
                tooltip_translation,
                tooltip_pending,
                tooltip_preview_text=tooltip_preview,
            )
            source_missing = not row_data.get("source_tag")
            target_missing = not row_data.get("target_tag")
            target_editable = not source_missing or not target_missing

            source_label = self._text_label(
                source_text,
                missing=source_missing,
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

    _GT_LANG_CODES = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "chinese": "zh-CN",
        "chinese (simplified)": "zh-CN",
        "simplified chinese": "zh-CN",
        "chinese (traditional)": "zh-TW",
        "traditional chinese": "zh-TW",
        "japanese": "ja",
        "korean": "ko",
        "turkish": "tr",
        "vietnamese": "vi",
    }

    def _review_target_language(self):
        language = ""
        try:
            language = str((self._config or {}).get("output_language") or "").strip()
        except Exception:
            language = ""
        return language or "English"

    def _review_target_language_code(self):
        return self._GT_LANG_CODES.get(self._review_target_language().lower(), "en")

    @staticmethod
    def _machine_translation_source_hash(text):
        return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()

    def _machine_translation_row_key(self, row_data, target_code=None):
        target_code = target_code or self._review_target_language_code()
        source_index = row_data.get("source_index")
        if source_index is None:
            source_index = row_data.get("row_index")
        source_tag = str(row_data.get("source_tag", "") or "").strip().lower()
        source_hash = self._machine_translation_source_hash(row_data.get("source", ""))
        return f"{target_code}|{source_index}|{source_tag}|{source_hash}"

    def _machine_translation_path_for_piece(self, piece):
        return _sdlxliff_machine_translation_path(
            getattr(self, "output_dir", "") or "",
            piece.get("path") or piece.get("output_name") or piece.get("name"),
        )

    def _read_machine_translation_file(self, piece):
        path = self._machine_translation_path_for_piece(piece)
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") if isinstance(data, dict) else {}
            return entries if isinstance(entries, dict) else {}
        except Exception:
            return {}

    def _load_machine_translation_file_for_piece(self, piece):
        rows = piece.get("rows") or []
        if not rows:
            return
        entries = self._read_machine_translation_file(piece)
        if not entries:
            return
        target_code = self._review_target_language_code()
        for row_data in rows:
            key = self._machine_translation_row_key(row_data, target_code)
            entry = entries.get(key)
            if not isinstance(entry, dict):
                continue
            if entry.get("target_language") != target_code:
                continue
            if entry.get("source_hash") != self._machine_translation_source_hash(row_data.get("source", "")):
                continue
            translated = str(entry.get("translation") or "").strip()
            if not translated:
                continue
            row_data["tooltip_translation"] = translated

    def _reload_machine_translation_previews(self, signature=None):
        changed_by_piece = {}
        try:
            for piece_index, piece in enumerate(self.pieces or []):
                rows = piece.get("rows") or []
                if not rows:
                    continue
                before = [
                    (
                        str(row_data.get("tooltip_translation") or ""),
                        str(row_data.get("status") or ""),
                        str(row_data.get("reason") or ""),
                    )
                    for row_data in rows
                ]
                for row_data in rows:
                    row_data.pop("tooltip_translation", None)
                self._load_machine_translation_file_for_piece(piece)
                self._refresh_piece_summary(piece)
                changed_rows = []
                for row_index, row_data in enumerate(rows):
                    after = (
                        str(row_data.get("tooltip_translation") or ""),
                        str(row_data.get("status") or ""),
                        str(row_data.get("reason") or ""),
                    )
                    if before[row_index] != after:
                        row_data["_source_preview_dirty"] = True
                        changed_rows.append(row_index)
                if changed_rows:
                    self._invalidate_piece_render_model(piece, restart_preload=False)
                    self._refresh_piece_list_item(piece_index)
                    changed_by_piece[piece_index] = changed_rows

            current_row = self._displayed_piece_row()
            for piece_index, changed_rows in changed_by_piece.items():
                if piece_index == current_row:
                    self._refresh_piece_header(piece_index)
                    for row_index in changed_rows:
                        self._refresh_visible_review_row_status(piece_index, row_index)
                if piece_index == current_row and not self._review_context_menu_is_open():
                    self._update_review_row_source_previews(piece_index, changed_rows, visible_only=True)
                elif piece_index == current_row:
                    self._queue_refresh_current_visible_dirty_source_previews()
        except Exception:
            pass
        try:
            self._last_machine_translation_signature = (
                signature if signature is not None else self._current_machine_translation_signature()
            )
        except Exception:
            pass

    def _write_machine_translation_entries(self, piece, row_translations):
        row_translations = [
            (row_data, str(translated or "").strip())
            for row_data, translated in (row_translations or [])
            if str(translated or "").strip()
        ]
        if not row_translations:
            return
        path = self._machine_translation_path_for_piece(piece)
        if not path:
            return
        target_code = self._review_target_language_code()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {}
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
                except Exception:
                    data = {}
            entries = data.get("entries")
            if not isinstance(entries, dict):
                entries = {}

            for row_data, translated in row_translations:
                entry_key = self._machine_translation_row_key(row_data, target_code)
                source_index = row_data.get("source_index")
                if source_index is None:
                    source_index = row_data.get("row_index")
                source_tag = str(row_data.get("source_tag", "") or "").strip().lower()
                source_hash = self._machine_translation_source_hash(row_data.get("source", ""))
                entries[entry_key] = {
                    "source_index": source_index,
                    "source_tag": source_tag,
                    "source_hash": source_hash,
                    "target_language": target_code,
                    "translation": translated,
                }

            data.update({
                "version": 1,
                "sidecar": os.path.basename(str(piece.get("path") or piece.get("output_name") or "")),
                "entries": entries,
            })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _write_machine_translation_entry(self, piece, row_data, translated):
        translated = str(translated or "").strip()
        if not translated:
            return
        self._write_machine_translation_entries(piece, [(row_data, translated)])

    def _tooltip_translation_key(self, piece, row_data):
        piece_path = ""
        try:
            piece_path = os.path.normcase(os.path.abspath(piece.get("path") or piece.get("output_name") or ""))
        except Exception:
            piece_path = str(piece.get("path") or piece.get("output_name") or "")
        source_index = row_data.get("source_index")
        if source_index is None:
            source_index = row_data.get("row_index")
        return (
            piece_path,
            self._review_target_language_code(),
            source_index,
            str(row_data.get("source_tag", "") or "").strip().lower(),
            self._machine_translation_source_hash(row_data.get("source", "")),
        )

    def _row_tooltip_translation(self, piece, row_data):
        stored = row_data.get("tooltip_translation")
        if stored:
            return str(stored)
        return ""

    def _set_row_tooltip_translation(self, piece, row_data, translated, persist=True):
        translated = str(translated or "").strip()
        if not translated:
            return
        try:
            from google_free_translate import GoogleFreeTranslateNew
            translated = GoogleFreeTranslateNew._sanitize_argos_text_tag_fragments(row_data.get("source", ""), translated)
        except Exception:
            pass
        translated = str(translated or "").strip()
        if not translated:
            return
        if self._normalized_machine_translation_text(translated) == self._normalized_machine_translation_text(row_data.get("source", "")):
            return
        row_data["tooltip_translation"] = translated
        row_data["_source_preview_dirty"] = True
        self._invalidate_piece_render_model(piece, restart_preload=False)
        if persist:
            self._write_machine_translation_entry(piece, row_data, translated)

    @staticmethod
    def _tooltip_batch_tag_name(tag_name):
        tag_name = str(tag_name or "").strip().lower()
        if re.fullmatch(r"h[1-6]", tag_name) or tag_name == "p":
            return tag_name
        return "p"

    def _tooltip_batch_html(self, work):
        parts = []
        for pos, (_row_idx, _key, source_text, tag_name) in enumerate(work):
            tag_name = self._tooltip_batch_tag_name(tag_name)
            escaped = html_lib.escape(str(source_text or ""), quote=False)
            parts.append(f'<{tag_name} data-sdl-tip="{pos}">{escaped}</{tag_name}>')
        return "\n".join(parts)

    def _extract_tooltip_batch_translations(self, translated_html, work):
        translated_html = html_lib.unescape(str(translated_html or ""))
        if not translated_html.strip() or not work:
            return {}
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(translated_html, "html.parser")
            nodes = [
                node for node in soup.find_all(self.TEXT_TAGS)
                if node.get_text(" ", strip=True)
            ]
            by_position = {}
            for node in nodes:
                raw_pos = node.get("data-sdl-tip")
                if raw_pos is None:
                    raw_pos = node.get("data-sdl-tip".lower())
                try:
                    pos = int(raw_pos)
                except (TypeError, ValueError):
                    continue
                if 0 <= pos < len(work):
                    by_position[pos] = node.get_text(" ", strip=True)
            if len(by_position) < len(work):
                ordered_texts = [
                    node.get_text(" ", strip=True)
                    for node in nodes
                    if node.get_text(" ", strip=True)
                ]
                for pos, text in enumerate(ordered_texts[:len(work)]):
                    by_position.setdefault(pos, text)
            return {
                key: by_position[pos]
                for pos, (_row_idx, key, _source_text, _tag_name) in enumerate(work)
                if str(by_position.get(pos, "")).strip()
            }
        except Exception:
            return {}

    @staticmethod
    def _normalized_machine_translation_text(text):
        text = unicodedata.normalize("NFKC", html_lib.unescape(str(text or ""))).strip().lower()
        return re.sub(r"\s+", " ", text)

    @classmethod
    def _compact_machine_translation_text(cls, text):
        return re.sub(r"\s+", "", cls._normalized_machine_translation_text(text))

    def _validate_tooltip_batch_translations(self, translations, work):
        if not translations:
            return {}, "Machine translation returned no parseable preview translations."
        source_by_key = {
            key: str(source_text or "")
            for _row_idx, key, source_text, _tag_name in (work or [])
        }
        valid = {}
        unchanged = []
        for key, translated in (translations or {}).items():
            translated_text = str(translated or "").strip()
            source_text = source_by_key.get(key, "")
            if not translated_text:
                continue
            if self._normalized_machine_translation_text(translated_text) == self._normalized_machine_translation_text(source_text):
                unchanged.append(key)
                continue
            valid[key] = translated_text
        if not valid:
            if unchanged:
                return {}, (
                    "Machine translation returned source text unchanged for "
                    f"{len(unchanged)}/{len(work or [])} row(s); refusing to save raw source as preview."
                )
            return {}, "Machine translation returned no usable preview translations."
        if unchanged:
            return valid, (
                "Machine translation returned source text unchanged for "
                f"{len(unchanged)}/{len(work or [])} row(s); skipped those rows."
            )
        return valid, ""

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
            self._piece_list_context_menu = menu
            self._set_review_context_menu_open(True)
            menu.aboutToHide.connect(lambda m=menu: self._clear_piece_list_context_menu(m))
            menu.popup(self.piece_list.viewport().mapToGlobal(pos))
        except Exception:
            pass

    def _clear_piece_list_context_menu(self, menu):
        try:
            if getattr(self, "_piece_list_context_menu", None) is menu:
                self._piece_list_context_menu = None
            self._set_review_context_menu_open(False)
            menu.deleteLater()
        except Exception:
            pass

    def _piece_tooltip_work(self, piece_index):
        if piece_index < 0 or piece_index >= len(self.pieces):
            return []
        piece = self.pieces[piece_index]
        work = []
        for row_idx, row_data in enumerate(piece.get("rows") or []):
            source_text = str(row_data.get("source", "") or "").strip()
            if not source_text:
                continue
            work.append((
                row_idx,
                self._tooltip_translation_key(piece, row_data),
                source_text,
                row_data.get("source_tag"),
            ))
        return work

    def _mark_tooltip_translation_pending(self, piece_index, work, refresh=True):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            rows = self.pieces[piece_index].get("rows") or []
            pending_rows = []
            for row_idx, _key, _source_text, _tag_name in work:
                if 0 <= row_idx < len(rows):
                    rows[row_idx]["tooltip_translation_pending"] = True
                    rows[row_idx]["tooltip_translation_status"] = self._machine_translation_pending_text()
                    rows[row_idx].pop("tooltip_translation_error", None)
                    rows[row_idx].pop("tooltip_translation_error_detail", None)
                    rows[row_idx]["_source_preview_dirty"] = True
                    pending_rows.append(row_idx)
            if pending_rows:
                self._invalidate_piece_render_model(self.pieces[piece_index], restart_preload=False)
            if refresh and pending_rows:
                if self._review_context_menu_is_open():
                    self._queue_refresh_current_visible_dirty_source_previews()
                else:
                    self._update_review_row_source_previews(piece_index, pending_rows, visible_only=True)
        except Exception:
            pass

    def _apply_tooltip_translation_status(self, piece_index, keys, message):
        try:
            if piece_index < 0 or piece_index >= len(self.pieces):
                return
            message = str(message or "").strip()
            if not message:
                return
            key_set = set(keys or [])
            piece = self.pieces[piece_index]
            rows = piece.get("rows") or []
            changed_rows = []
            for row_index, row_data in enumerate(rows):
                if key_set and self._tooltip_translation_key(piece, row_data) not in key_set:
                    continue
                if not row_data.get("tooltip_translation_pending"):
                    continue
                row_data["tooltip_translation_status"] = message
                row_data["_source_preview_dirty"] = True
                changed_rows.append(row_index)
            if not changed_rows:
                return
            self._invalidate_piece_render_model(piece, restart_preload=False)
            if piece_index == self._displayed_piece_row():
                if self._review_context_menu_is_open():
                    self._queue_refresh_current_visible_dirty_source_previews()
                else:
                    self._update_review_row_source_previews(piece_index, changed_rows, visible_only=True)
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
                batch_html = self._tooltip_batch_html(work)
                result = translator.translate(batch_html)
                result_error = str(result.get("error") or "").strip() if isinstance(result, dict) else ""
                result_note = self._machine_translation_result_note(result)
                translated_html = str(result.get("translatedText") or "").strip()
                translations = self._extract_tooltip_batch_translations(translated_html, work)
                translations, validation_error = self._validate_tooltip_batch_translations(translations, work)
                if result_error:
                    error = result_error
                    translations = {}
                elif validation_error:
                    error = validation_error
                error = self._append_machine_translation_note(error, result_note)
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
                    batch_html = self._tooltip_batch_html(work)
                    result = translator.translate(batch_html)
                    result_error = str(result.get("error") or "").strip() if isinstance(result, dict) else ""
                    result_note = self._machine_translation_result_note(result)
                    translated_html = str(result.get("translatedText") or "").strip()
                    translations = self._extract_tooltip_batch_translations(translated_html, work)
                    translations, validation_error = self._validate_tooltip_batch_translations(translations, work)
                    if result_error:
                        error = result_error
                        translations = {}
                    elif validation_error:
                        error = validation_error
                    error = self._append_machine_translation_note(error, result_note)
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
            changed_rows = set()
            rows_to_persist = []
            for row_index, row_data in enumerate(piece.get("rows") or []):
                key = self._tooltip_translation_key(piece, row_data)
                was_pending = bool(row_data.pop("tooltip_translation_pending", False))
                if was_pending:
                    row_data.pop("tooltip_translation_status", None)
                    changed_rows.add(row_index)
                if key in translations:
                    translated = str(translations[key] or "").strip()
                    row_data.pop("tooltip_translation_error", None)
                    row_data.pop("tooltip_translation_error_detail", None)
                    if self._normalized_machine_translation_text(translated) == self._normalized_machine_translation_text(row_data.get("source", "")):
                        changed_rows.add(row_index)
                        continue
                    self._set_row_tooltip_translation(piece, row_data, translated, persist=False)
                    rows_to_persist.append((row_data, translated))
                    changed_rows.add(row_index)
                elif error and was_pending and not translations:
                    row_data["tooltip_translation_error"] = self._compact_machine_translation_error(error)
                    row_data["tooltip_translation_error_detail"] = str(error or "")
                    row_data["_source_preview_dirty"] = True
                    changed_rows.add(row_index)
            if rows_to_persist:
                self._write_machine_translation_entries(piece, rows_to_persist)
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
        if batch_active:
            return
        if error and not translations:
            try:
                self.save_status_label.setText(
                    f"Machine translation preview failed: {self._compact_machine_translation_error(error)}"
                )
            except Exception:
                pass
        elif translations:
            try:
                provider_label = self._machine_translation_provider_label()
                message = f"Generated {len(translations)} {provider_label} machine translation preview(s)"
                if error:
                    message = f"{message}. {self._compact_machine_translation_error(error)}"
                self.save_status_label.setText(message)
                QTimer.singleShot(2500, lambda: self.save_status_label.setText(""))
            except Exception:
                pass

    def _finish_piece_list_tooltip_translations(self, translated_count, piece_count, error):
        self._tooltip_translation_batch_active = False
        self._tooltip_translation_running = False
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
                QTimer.singleShot(2500, lambda: self.save_status_label.setText(""))
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

    def _schedule_target_edit(self, piece_index, row_index, text):
        if piece_index < 0 or row_index < 0:
            return
        self._pending_target_edits[(piece_index, row_index)] = text
        self.save_status_label.setText("Unsaved")
        self._edit_save_timer.start(500)

    def _flush_target_edits(self):
        pending = dict(self._pending_target_edits)
        self._pending_target_edits.clear()
        if not pending:
            return
        saved = 0
        try:
            for (piece_index, row_index), text in pending.items():
                if self._apply_target_edit(piece_index, row_index, text):
                    saved += 1
            self.save_status_label.setText("Saved" if saved else "")
        except Exception as exc:
            self.save_status_label.setText(f"Save failed: {exc}")

    def _output_name_for_piece(self, piece):
        output_name = piece.get("output_name") or self._sidecar_output_name(piece.get("path") or piece.get("name") or "")
        return output_name or piece.get("name") or "SDLXLIFF output"

    def _output_path_for_piece(self, piece):
        output_name = self._output_name_for_piece(piece)
        if not output_name:
            return None
        return os.path.join(self.output_dir or "", output_name)

    def _target_html_with_edit(self, piece, row_data, text):
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return piece.get("target_html", "")
        target_html = html_lib.unescape(str(piece.get("target_html") or ""))
        soup = BeautifulSoup(target_html, "html.parser")
        tag_nodes = list(soup.find_all(self.TEXT_TAGS))
        target_index = row_data.get("target_index")
        node = None
        if isinstance(target_index, int) and 0 <= target_index < len(tag_nodes):
            node = tag_nodes[target_index]
        elif row_data.get("target_tag"):
            editable_nodes = [tag for tag in tag_nodes if tag.get_text(" ", strip=True)]
            fallback_index = min(len(editable_nodes) - 1, max(0, int(row_data.get("row_index", 0)))) if editable_nodes else -1
            if fallback_index >= 0:
                node = editable_nodes[fallback_index]

        if node is None:
            tag_name = row_data.get("source_tag") or row_data.get("target_tag") or "p"
            node = soup.new_tag(tag_name)
            if soup.body:
                soup.body.append(node)
            else:
                soup.append(node)
            row_data["target_tag"] = tag_name
            row_data["target_index"] = len(list(soup.find_all(self.TEXT_TAGS))) - 1

        node.clear()
        node.append(str(text or ""))
        return str(soup)

    def _write_piece_target_html(self, piece, target_html):
        sidecar_path = piece.get("path")
        if not sidecar_path:
            return
        tree = ET.parse(sidecar_path)
        root = tree.getroot()
        target_element = None
        for element in root.iter():
            if self._local_name(element.tag) == "target":
                target_element = element
                break
        if target_element is None:
            raise ValueError("SDLXLIFF target element not found")
        for child in list(target_element):
            target_element.remove(child)
        target_element.text = target_html
        try:
            ET.register_namespace("", "urn:oasis:names:tc:xliff:document:1.2")
            ET.register_namespace("sdl", "http://sdl.com/FileTypes/SdlXliff/1.0")
        except Exception:
            pass
        tree.write(sidecar_path, encoding="utf-8", xml_declaration=True)
        try:
            self._last_review_signature = self._current_review_signature()
        except Exception:
            pass

        output_path = self._output_path_for_piece(piece)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(target_html)

    def _apply_target_edit(self, piece_index, row_index, text):
        if piece_index < 0 or piece_index >= len(self.pieces):
            return False
        piece = self.pieces[piece_index]
        rows = piece.get("rows") or []
        if row_index < 0 or row_index >= len(rows):
            return False
        row_data = rows[row_index]
        row_data["row_index"] = row_index
        target_html = self._target_html_with_edit(piece, row_data, text)
        self._write_piece_target_html(piece, target_html)
        piece["target_html"] = target_html
        row_data["target"] = str(text or "")
        status, reason = self._row_status(row_data.get("source", ""), row_data.get("target", ""))
        if row_data.get("source_tag") and row_data.get("target_tag") and row_data.get("source_tag") != row_data.get("target_tag"):
            status, reason = self._tag_mismatch_status(row_data.get("source_tag"), row_data.get("target_tag"))
        row_data["status"] = status
        row_data["reason"] = reason
        self._invalidate_piece_render_model(piece, restart_preload=False)
        self._refresh_piece_summary(piece)
        self._refresh_piece_list_item(piece_index)
        self._refresh_piece_header(piece_index)
        self._refresh_visible_review_row_status(piece_index, row_index)
        return True

    def closeEvent(self, event):
        try:
            self._save_current_review_scroll()
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
            if self.rows_layout is not None:
                self.rows_layout.invalidate()
                self.rows_layout.activate()
            if self.rows_widget is not None:
                self.rows_widget.updateGeometry()
            if final:
                if self.rows_widget is not None:
                    self.rows_widget.adjustSize()
                self.rows_stack.adjustSize()
            self._sync_review_scroll_range()
            self.rows_stack.updateGeometry()
            self.scroll.updateGeometry()
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

    def _review_row_text_heights(self, row_height, source_lines=1, target_lines=1, tooltip_lines=0):
        source_lines = max(1, int(source_lines or 1))
        target_lines = max(1, int(target_lines or 1))
        tooltip_lines = max(0, int(tooltip_lines or 0))
        source_height = max(34, (source_lines + tooltip_lines) * 22 + (28 if tooltip_lines else 10))
        target_height = max(38, target_lines * 22 + 28)
        available = max(38, int(row_height or self.REVIEW_ROW_MIN_HEIGHT) - 10)
        return min(source_height, available), min(target_height, available)

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

    def _inject_current_machine_translation_to_target(self, piece_index, row_index, editor=None):
        try:
            piece = self.pieces[piece_index]
            row_data = (piece.get("rows") or [])[row_index]
            translated = self._row_tooltip_translation(piece, row_data)
            if not str(translated or "").strip():
                self.save_status_label.setText("No machine translation preview")
                return
            self._inject_machine_translation_to_target(piece_index, row_index, translated, editor)
        except Exception as exc:
            try:
                self.save_status_label.setText(f"Inject failed: {exc}")
            except Exception:
                pass

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
        tooltip_translation=None,
        tooltip_pending=False,
        tooltip_state=None,
        tooltip_detail=None,
        translate_tooltip_callback=None,
        inject_machine_translation_callback=None,
    ):
        label = QLabel(text if text else ("[missing]" if missing else "[empty]"))
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

    def _add_review_row(self, piece, row_data, idx, max_len, colors, row_model=None, updates_enabled=True):
        bg, source_bar, target_bar, dot_color, border_color = colors.get(row_data["status"], colors["green"])
        row_model = row_model if isinstance(row_model, dict) else {}
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
        two_column_layout = bool(row_model.get("two_column_layout", row_model.get("one_column_layout", row_model.get("one_row_layout", getattr(self, "_two_column_layout_enabled", True)))))
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
        grid.setContentsMargins(10, 5, 10, 5)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(0)
        grid.setColumnMinimumWidth(0, 48)
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

        tag_label = self._tag_label(row_data.get("source_tag"), row_data.get("target_tag"), row_data.get("status"))
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
        self.rows_layout.addWidget(frame)
        return frame

    def _render_piece(self, row, show_loading=True):
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
        self._cancel_active_review_render()

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
            self._finish_rows_rebuild(final=True)
            self._queue_refresh_current_visible_dirty_source_previews()
            self._restore_review_scroll(row)
            QTimer.singleShot(0, self._queue_refresh_current_visible_dirty_source_previews)
            self._queue_review_page_preloads(row)
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
            self._queue_review_page_preloads(row)
            return

        rows = piece.get("rows") or []
        if not rows:
            empty = QLabel("No p/h1-h6 text units found in this sidecar.")
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
            self._queue_review_page_preloads(row)
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
            self._queue_review_page_preloads(row)
            return

        if len(rows) <= self.REVIEW_SYNC_RENDER_ROW_LIMIT:
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
                self._queue_review_page_preloads(row)
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
                self._queue_review_page_preloads(row)
                return

        render_timer = QTimer(self)
        render_timer.setSingleShot(True)
        row_state = {"idx": 0}
        batch_size = 12

        if show_loading:
            self.rows_stack.setCurrentWidget(page)
            self._finish_seamless_review_swap(page)
            self._finish_rows_rebuild(final=False)

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
                visible_stream = self.rows_stack.currentWidget() is page
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
                    end = min(len(rows), start + batch_size)
                    batch_frames = []
                    for idx in range(start, end):
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
                        if frame is not None:
                            batch_frames.append(frame)
                    row_state["idx"] = end
                    if visible_stream:
                        self._refresh_review_stream_geometry(final=False)
                    for frame in batch_frames:
                        try:
                            frame.setUpdatesEnabled(True)
                            frame.update()
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
                    render_timer.start(1)
                    return

                if render_token != self._render_token:
                    _finish_active_render_timer()
                    _discard_active_render_page()
                    return
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
                    self._queue_review_page_preloads(row)
                if self._active_render_page is page:
                    self._active_render_row = None
                    self._active_render_page = None

        render_timer.timeout.connect(_run_render_batch)
        self._active_render_timer = render_timer
        render_timer.start(self._review_loading_minimum_ms)


class RetranslationMixin:
    """Mixin class containing retranslation methods for TranslatorGUI"""

    _RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY = "retranslation_show_model_info"
    _SDLXLIFF_AUTOGEN_STATUSES = {
        "completed",
        "qa_failed",
        "completed_empty",
        "completed_image_only",
    }

    @staticmethod
    def _sdlxliff_autogen_output_path(output_dir, output_file):
        if not output_dir or not output_file:
            return None
        normalized = str(output_file).replace("\\", "/")
        path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
        return os.path.normpath(path)

    @staticmethod
    def _sdlxliff_autogen_source_candidates(entry, output_file=None, progress_key=None):
        entry = entry if isinstance(entry, dict) else {}
        raw_candidates = [
            entry.get("original_basename"),
            entry.get("original_filename"),
            entry.get("chapter_file"),
            entry.get("source_filename"),
            entry.get("filename"),
            progress_key,
        ]
        if output_file:
            output_name = os.path.basename(str(output_file).replace("\\", "/"))
            if output_name.lower().startswith("response_"):
                raw_candidates.append(output_name[len("response_"):])
        candidates = []
        seen = set()
        for candidate in raw_candidates:
            if not candidate:
                continue
            text = str(candidate).replace("\\", "/")
            variants = [text, os.path.basename(text)]
            stem, ext = os.path.splitext(text)
            if not ext and stem:
                variants.extend([f"{text}.xhtml", f"{text}.html", f"{text}.htm"])
                base = os.path.basename(text)
                variants.extend([f"{base}.xhtml", f"{base}.html", f"{base}.htm"])
            elif ext.lower() in (".html", ".htm", ".xhtml"):
                for html_ext in (".xhtml", ".html", ".htm"):
                    variants.append(f"{stem}{html_ext}")
                base = os.path.basename(text)
                base_stem, _base_ext = os.path.splitext(base)
                if base_stem:
                    for html_ext in (".xhtml", ".html", ".htm"):
                        variants.append(f"{base_stem}{html_ext}")
            for variant in variants:
                if variant and variant not in seen:
                    seen.add(variant)
                    candidates.append(variant)
        return candidates

    @staticmethod
    def _sdlxliff_autogen_decode(data):
        for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _sdlxliff_is_extracted_epub_dir(path):
        if not path or not os.path.isdir(path):
            return False
        direct_candidates = [
            os.path.join(path, "content.opf"),
            os.path.join(path, "OEBPS", "content.opf"),
            os.path.join(path, "EPUB", "content.opf"),
            os.path.join(path, "META-INF", "container.xml"),
        ]
        if any(os.path.isfile(candidate) for candidate in direct_candidates):
            return True
        try:
            for _root_dir, _dirs, files in os.walk(path):
                if any(str(fname).lower().endswith(".opf") for fname in files):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _sdlxliff_autogen_read_source_from_directory(root_dir, candidate_names, candidate_basenames):
        if not root_dir or not os.path.isdir(root_dir):
            return None, None
        try:
            root_abs = os.path.abspath(root_dir)
            for dirpath, _dirs, files in os.walk(root_abs):
                for fname in files:
                    rel = os.path.relpath(os.path.join(dirpath, fname), root_abs).replace("\\", "/")
                    normalized = rel.lower().strip("/")
                    basename = os.path.basename(normalized)
                    if normalized not in candidate_names and basename not in candidate_basenames:
                        continue
                    path = os.path.join(dirpath, fname)
                    try:
                        with open(path, "rb") as f:
                            return RetranslationMixin._sdlxliff_autogen_decode(f.read()), path
                    except Exception:
                        continue
        except Exception:
            return None, None
        return None, None

    def _sdlxliff_current_input_file_candidates(self):
        candidates = []

        def _add_many(values):
            if not values:
                return
            if isinstance(values, (str, bytes, os.PathLike)):
                values = [values]
            for value in values:
                if not value:
                    continue
                try:
                    candidates.append(str(value))
                except Exception:
                    continue

        _add_many(getattr(self, "selected_files", None))
        try:
            entry = getattr(self, "entry_epub", None)
            if entry is not None and hasattr(entry, "text"):
                _add_many(entry.text())
        except Exception:
            pass
        try:
            cfg = getattr(self, "config", None)
            if isinstance(cfg, dict):
                _add_many(cfg.get("selected_files"))
                _add_many(cfg.get("last_input_files"))
        except Exception:
            pass

        resolved = []
        seen = set()
        for candidate in candidates:
            path = os.path.normpath(candidate)
            try:
                norm = os.path.normcase(os.path.abspath(path))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.exists(path):
                resolved.append(path)
        return resolved

    def _sdlxliff_exact_input_epub_candidates(self, output_dir):
        output_name = os.path.basename(os.path.normpath(str(output_dir or "")))
        output_key = os.path.normcase(output_name)
        if not output_key:
            return []
        matches = []
        for path in self._sdlxliff_current_input_file_candidates():
            if os.path.isfile(path) and str(path).lower().endswith(".epub"):
                candidate_key = os.path.splitext(os.path.basename(path))[0]
            elif self._sdlxliff_is_extracted_epub_dir(path):
                candidate_key = os.path.basename(os.path.normpath(path))
            else:
                continue
            if os.path.normcase(candidate_key) == output_key:
                matches.append(path)
        return matches

    def _sdlxliff_valid_current_input_epub_candidates(self):
        candidates = []
        seen = set()
        for path in self._sdlxliff_current_input_file_candidates():
            resolved = self._sdlxliff_valid_epub_path("", path)
            if not resolved:
                continue
            try:
                norm = os.path.normcase(os.path.abspath(resolved))
            except Exception:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append(resolved)
        return candidates

    @staticmethod
    def _sdlxliff_valid_epub_path(output_dir, path):
        if not path:
            return None
        candidate = path if os.path.isabs(str(path)) else os.path.join(output_dir or "", str(path))
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate) and candidate.lower().endswith(".epub"):
            return candidate
        if RetranslationMixin._sdlxliff_is_extracted_epub_dir(candidate):
            return candidate
        return None

    def _sdlxliff_preferred_input_epub(self, output_dir, file_path=None):
        direct = self._sdlxliff_valid_epub_path(output_dir, file_path)
        if direct:
            return direct
        exact = self._sdlxliff_exact_input_epub_candidates(output_dir)
        if exact:
            return exact[0]
        current = self._sdlxliff_valid_current_input_epub_candidates()
        return current[0] if len(current) == 1 else None

    def _sdlxliff_update_source_epub_ref(self, output_dir, epub_path):
        epub_path = self._sdlxliff_valid_epub_path(output_dir, epub_path)
        if not output_dir or not epub_path:
            return False
        source_ref = os.path.join(output_dir, "source_epub.txt")
        try:
            current = ""
            if os.path.isfile(source_ref):
                with open(source_ref, "r", encoding="utf-8", errors="ignore") as f:
                    current = f.read().strip()
            current_path = self._sdlxliff_valid_epub_path(output_dir, current)
            if current_path and os.path.normcase(os.path.abspath(current_path)) == os.path.normcase(os.path.abspath(epub_path)):
                return False
            with open(source_ref, "w", encoding="utf-8") as f:
                f.write(os.path.abspath(epub_path))
            return True
        except Exception:
            return False

    def _sdlxliff_autogen_epub_candidates(self, output_dir, file_path=None):
        candidates = []
        preferred = self._sdlxliff_preferred_input_epub(output_dir, file_path)
        if preferred:
            candidates.append(preferred)
            self._sdlxliff_update_source_epub_ref(output_dir, preferred)
        source_ref = os.path.join(output_dir or "", "source_epub.txt")
        try:
            if os.path.isfile(source_ref):
                with open(source_ref, "r", encoding="utf-8", errors="ignore") as f:
                    ref = f.read().strip()
                if ref:
                    candidates.append(ref)
        except Exception:
            pass
        candidates.extend(self._sdlxliff_exact_input_epub_candidates(output_dir))
        candidates.extend(self._sdlxliff_valid_current_input_epub_candidates())
        try:
            for fname in os.listdir(output_dir or ""):
                if str(fname).lower().endswith(".epub"):
                    candidates.append(os.path.join(output_dir, fname))
                else:
                    path = os.path.join(output_dir, fname)
                    if self._sdlxliff_is_extracted_epub_dir(path):
                        candidates.append(path)
        except Exception:
            pass
        resolved = []
        seen = set()
        for candidate in candidates:
            path = self._sdlxliff_valid_epub_path(output_dir, candidate)
            if not path:
                continue
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(path)
        return resolved

    def _sdlxliff_autogen_read_source_html(self, output_dir, entry, output_file=None, progress_key=None, file_path=None):
        candidates = self._sdlxliff_autogen_source_candidates(entry, output_file, progress_key)
        candidate_names = {str(c).replace("\\", "/").lower().strip("/") for c in candidates if c}
        candidate_basenames = {os.path.basename(str(c).replace("\\", "/")).lower() for c in candidates if c}
        candidate_names.discard("")
        candidate_basenames.discard("")
        if candidate_names or candidate_basenames:
            for epub_path in self._sdlxliff_autogen_epub_candidates(output_dir, file_path):
                if os.path.isdir(epub_path):
                    source_text, source_path = self._sdlxliff_autogen_read_source_from_directory(
                        epub_path,
                        candidate_names,
                        candidate_basenames,
                    )
                    if source_text:
                        return source_text, source_path
                    continue
                try:
                    with zipfile.ZipFile(epub_path, "r") as zf:
                        for name in zf.namelist():
                            normalized = str(name).replace("\\", "/").lower().strip("/")
                            if normalized in candidate_names or os.path.basename(normalized) in candidate_basenames:
                                return self._sdlxliff_autogen_decode(zf.read(name)), f"{epub_path}!{name}"
                except Exception:
                    continue

        output_path = self._sdlxliff_autogen_output_path(output_dir, output_file)
        output_norm = os.path.normcase(os.path.abspath(output_path)) if output_path else ""
        output_base = os.path.basename(str(output_file or "").replace("\\", "/")).lower()
        for candidate in candidates:
            text = str(candidate).replace("\\", "/")
            variants = [text]
            basename = os.path.basename(text)
            if basename and basename != text:
                variants.append(basename)
            for variant in variants:
                path = variant if os.path.isabs(variant) else os.path.join(output_dir or "", variant)
                path = os.path.normpath(path)
                if output_norm and os.path.normcase(os.path.abspath(path)) == output_norm:
                    continue
                if output_base and os.path.basename(path).lower() == output_base:
                    continue
                if os.path.isfile(path):
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            return f.read(), path
                    except Exception:
                        continue
        return None, None

    def _generate_sdlxliff_sidecars_from_completed_entries(
        self,
        output_dir,
        file_path=None,
        progress_data=None,
        output_files=None,
        overwrite=True,
    ):
        stats = {
            "considered": 0,
            "created": 0,
            "skipped": 0,
            "missing_source": 0,
            "missing_output": 0,
            "failed": 0,
            "paths": [],
        }
        if not output_dir or not os.path.isdir(output_dir):
            return stats

        if progress_data is None:
            progress_path = os.path.join(output_dir, "translation_progress.json")
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
            except Exception:
                progress_data = {}

        chapters = progress_data.get("chapters") if isinstance(progress_data, dict) else None
        if not isinstance(chapters, dict):
            chapters = progress_data if isinstance(progress_data, dict) else {}
        if not isinstance(chapters, dict):
            return stats

        requested = None
        if output_files:
            requested = {
                os.path.basename(str(name).replace("\\", "/")).lower()
                for name in output_files
                if name
            }
            requested.discard("")

        seen_outputs = set()
        old_output_sdlxliff = os.environ.get("OUTPUT_SDLXLIFF")
        os.environ["OUTPUT_SDLXLIFF"] = "1"
        try:
            from TransateKRtoEN import _write_html_sdlxliff_sidecar

            for progress_key, entry in chapters.items():
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status", "") or "").lower()
                if status not in self._SDLXLIFF_AUTOGEN_STATUSES:
                    continue
                output_file = entry.get("output_file")
                output_name = os.path.basename(str(output_file or "").replace("\\", "/"))
                if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
                    continue
                if requested is not None and output_name.lower() not in requested:
                    continue
                if output_name.lower() in seen_outputs:
                    stats["skipped"] += 1
                    continue
                seen_outputs.add(output_name.lower())
                stats["considered"] += 1

                sidecar_path = os.path.join(output_dir, "SDLXLIFF", f"{output_name}.sdlxliff")
                if not overwrite and os.path.isfile(sidecar_path):
                    stats["skipped"] += 1
                    stats["paths"].append(sidecar_path)
                    continue

                output_path = self._sdlxliff_autogen_output_path(output_dir, output_file)
                if not output_path or not os.path.isfile(output_path):
                    stats["missing_output"] += 1
                    continue
                try:
                    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                        target_html = f.read()
                except Exception:
                    stats["missing_output"] += 1
                    continue

                source_html, source_path = self._sdlxliff_autogen_read_source_html(
                    output_dir,
                    entry,
                    output_file=output_file,
                    progress_key=progress_key,
                    file_path=file_path,
                )
                if not source_html:
                    stats["missing_source"] += 1
                    continue

                chapter = dict(entry)
                if not chapter.get("original_basename"):
                    chapter["original_basename"] = os.path.basename(str(source_path or output_name).split("!", 1)[-1])
                result_path = _write_html_sdlxliff_sidecar(output_dir, output_name, chapter, source_html, target_html)
                if result_path:
                    stats["created"] += 1
                    stats["paths"].append(result_path)
                else:
                    stats["failed"] += 1
        finally:
            if old_output_sdlxliff is None:
                os.environ.pop("OUTPUT_SDLXLIFF", None)
            else:
                os.environ["OUTPUT_SDLXLIFF"] = old_output_sdlxliff

        return stats

    def _open_or_reuse_sdlxliff_review(self, output_dir, review_path=None, parent=None):
        try:
            key = os.path.normcase(os.path.abspath(output_dir))
        except Exception:
            key = str(output_dir or "")
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
            except Exception:
                pass
            review_dialog.reopen_for_path(output_dir, review_path)

        review_dialog.show()
        review_dialog.raise_()
        review_dialog.activateWindow()
        return review_dialog

    def _get_retranslation_show_model_info_state(self, file_path=None):
        """Return the persisted Show Model Info preference, with live dialog cache first."""
        try:
            if file_path:
                file_key = os.path.abspath(file_path)
                cached = getattr(self, '_retranslation_dialog_cache', {}).get(file_key, {})
                if isinstance(cached, dict) and 'show_model_info_state' in cached:
                    return bool(cached.get('show_model_info_state'))
        except Exception:
            pass
        try:
            return bool(getattr(self, 'config', {}).get(self._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY, True))
        except Exception:
            return True

    def _persist_retranslation_show_model_info_state(self, enabled):
        """Persist the Show Model Info preference across app sessions."""
        try:
            if not hasattr(self, 'config') or not isinstance(self.config, dict):
                self.config = {}
            self.config[self._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY] = bool(enabled)
            if hasattr(self, 'save_config') and callable(self.save_config):
                self.save_config(show_message=False)
        except Exception as exc:
            try:
                print(f"⚠️ Could not persist Show Model Info state: {exc}")
            except Exception:
                pass

    def _progress_file_is_skipped_special(self, filename, fallback_is_special=False):
        """Return True only for special files that translation would skip."""
        translate_special = bool(
            getattr(self, 'translate_special_files_var', False)
            or getattr(self, 'config', {}).get('translate_special_files', False)
        )
        if hasattr(self, '_should_skip_special_file'):
            return self._should_skip_special_file(filename, translate_special)
        if translate_special:
            return False
        is_special = fallback_is_special
        if filename and hasattr(self, '_is_special_file'):
            is_special = self._is_special_file(filename)
        if not is_special:
            return False
        translate_all_numbered = bool(
            getattr(self, 'translate_all_numbered_html_var', True)
            or getattr(self, 'config', {}).get('translate_all_numbered_html', True)
        )
        if translate_all_numbered:
            stem = os.path.splitext(os.path.basename(str(filename or '')))[0]
            if stem.lower().startswith('response_'):
                stem = stem[len('response_'):]
            if re.search(r'\d', stem):
                return False
        return True

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

    def _create_retranslation_shell_dialog(self, title="Progress Manager", width_ratio=0.38, height_ratio=0.4):
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

    def _show_retranslation_shell_then_build(self, file_path, show_special_files_state=False):
        dialog, dialog_layout, loading_widget, loading_label = self._create_retranslation_shell_dialog("Progress Manager")
        file_key = os.path.abspath(file_path)

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
                content = QWidget(dialog)
                content_layout = QVBoxLayout(content)
                content_layout.setContentsMargins(0, 0, 0, 0)

                result = self._force_retranslation_epub_or_text(
                    file_path,
                    parent_dialog=dialog,
                    tab_frame=content,
                    show_special_files_state=show_special_files_state,
                )
                if not result:
                    dialog.hide()
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

        QTimer.singleShot(50, _build_dialog_contents)

    def force_retranslation(self):
        """Force retranslation of specific chapters or images with improved display"""
        
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
        # Default to True for .txt, .pdf, .csv, and .json files, False for .epub
        show_special_extensions = ('.txt', '.pdf', '.csv', '.json')
        show_special = input_path.lower().endswith(show_special_extensions)
        
        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
            cached_data = self._retranslation_dialog_cache[file_key]
            if cached_data:
                show_special = cached_data.get('show_special_files_state', show_special)
        
        self._show_retranslation_shell_then_build(input_path, show_special_files_state=show_special)


    def _force_retranslation_epub_or_text(self, file_path, parent_dialog=None, tab_frame=None, show_special_files_state=False):
        """
        Shared logic for force retranslation of EPUB/text files with OPF support
        Can be used standalone or embedded in a tab
        
        Args:
            file_path: Path to the EPUB/text file
            parent_dialog: If provided, won't create its own dialog
            tab_frame: If provided, will render into this frame instead of creating dialog
            show_special_files_state: Initial state for showing special files toggle
        
        Returns:
            dict: Contains all the UI elements and data for external access
        """
        
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check for output directory override
        override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
        if not override_dir and hasattr(self, 'config'):
            override_dir = self.config.get('output_directory')
            
        if override_dir:
            output_dir = os.path.join(override_dir, epub_base)
        else:
            output_dir = epub_base
        # On macOS .app bundles, cwd can be '/' (read-only root).
        # Resolve relative output paths against the input file's directory.
        # Only on macOS — on Windows this would change the output dir and break progress tracking.
        if _IS_MACOS and not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), output_dir)
        
        if not os.path.exists(output_dir):
            # Output folder doesn't exist - create it with an empty progress file
            try:
                os.makedirs(output_dir, exist_ok=True)
                empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                progress_file_path = os.path.join(output_dir, "translation_progress.json")
                with open(progress_file_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_prog, f, ensure_ascii=False, indent=2)
                print(f"📁 Created output folder: {output_dir}")
                # Flash the PM button green to signal folder creation
                self._flash_pm_button_green(output_dir)
            except Exception as e:
                if not parent_dialog:
                    self._show_message('error', "Error", f"Could not create output folder: {e}")
                return None
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            # No progress file - create empty progress structure
            # This allows fuzzy matching to discover existing files
            print("⚠️ No progress file found - will attempt to discover existing translations")
            prog = {
                "chapters": {},
                "chapter_chunks": {},
                "version": "2.1"
            }
        else:
            with open(progress_file, 'r', encoding='utf-8') as f:
                prog = json.load(f)

        # Helper: auto-discover completed files when no OPF is available
        def _auto_discover_from_output_dir(output_dir, prog):
            updated = False
            try:
                # Only exclude _translated.* combined output files when the source
                # file itself does NOT contain "_translated" in its name
                source_has_translated = "_translated" in os.path.basename(file_path).lower()
                files = [
                    f for f in os.listdir(output_dir)
                    if os.path.isfile(os.path.join(output_dir, f))
                    # accept any extension except known non-chapter files
                    and (source_has_translated or not f.lower().endswith("_translated.txt"))
                    and (source_has_translated or not f.lower().endswith("_translated.pdf"))
                    and (source_has_translated or not f.lower().endswith("_translated.html"))
                    and f != "translation_progress.json"
                    and f.lower() not in ("glossary.csv", "metadata.json", "styles.css", "rolling_summary.txt")
                    and not f.lower().endswith(".epub")
                    and not f.lower().endswith(".cache")
                ]
                for fname in files:
                    base = os.path.basename(fname)
                    # Normalize by stripping response_ and all extensions
                    if base.startswith("response_"):
                        base = base[len("response_"):]
                    while True:
                        new_base, ext = os.path.splitext(base)
                        if not ext:
                            break
                        base = new_base

                    import re
                    m = re.findall(r"(\d+)", base)
                    chapter_num = int(m[-1]) if m else None
                    key = str(chapter_num) if chapter_num is not None else f"special_{base}"
                    actual_num = chapter_num if chapter_num is not None else 0

                    if key in prog.get("chapters", {}):
                        continue
                    
                    # Also check if any existing entry already references this output file
                    already_tracked = any(
                        ch.get("output_file") == fname
                        for ch in prog.get("chapters", {}).values()
                    )
                    if already_tracked:
                        continue

                    prog.setdefault("chapters", {})[key] = {
                        "actual_num": actual_num,
                        "content_hash": "",
                        "output_file": fname,
                        "status": "completed",
                        "last_updated": os.path.getmtime(os.path.join(output_dir, fname)),
                        "auto_discovered": True,
                        "original_basename": fname
                    }
                    updated = True
            except Exception as e:
                print(f"⚠️ Auto-discovery (no OPF) failed: {e}")
            return updated
        
        # Clean up missing files and merged children when opening the GUI
        # This handles the case where parent files were manually deleted
        from TransateKRtoEN import ProgressManager
        temp_progress = ProgressManager(os.path.dirname(progress_file))
        temp_progress.prog = prog
        temp_progress.cleanup_missing_files(output_dir)
        prog = temp_progress.prog
        
        # Save the cleaned progress back to file
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(prog, f, ensure_ascii=False, indent=2)
        
        # =====================================================
        # PARSE CONTENT.OPF FOR CHAPTER MANIFEST
        # =====================================================
        
        # State variables for title-row toggles (lists allow nested handlers to mutate them)
        show_special_files = [show_special_files_state]
        show_model_info_state = self._get_retranslation_show_model_info_state(file_path)
        show_model_info = [show_model_info_state]
        
        spine_chapters = []
        opf_chapter_order = {}
        is_epub = file_path.lower().endswith('.epub')
        opf_parsed = False

        if is_epub and os.path.exists(file_path):
            try:
                import xml.etree.ElementTree as ET
                import zipfile
                
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Find content.opf file
                    opf_path = None
                    opf_content = None
                    
                    # First try to find via container.xml
                    try:
                        container_content = zf.read('META-INF/container.xml')
                        container_root = ET.fromstring(container_content)
                        rootfile = container_root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                        if rootfile is not None:
                            opf_path = rootfile.get('full-path')
                    except:
                        pass
                    
                    # Fallback: search for content.opf
                    if not opf_path:
                        for name in zf.namelist():
                            if name.endswith('content.opf'):
                                opf_path = name
                                break
                    
                    if opf_path:
                        opf_content = zf.read(opf_path)
                        
                        # Parse OPF
                        root = ET.fromstring(opf_content)
                        
                        # Handle namespaces
                        ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        if root.tag.startswith('{'):
                            default_ns = root.tag[1:root.tag.index('}')]
                            ns = {'opf': default_ns}
                        
                        # Get manifest - all chapter files
                        manifest_chapters = {}
                        
                        for item in root.findall('.//opf:manifest/opf:item', ns):
                            item_id = item.get('id')
                            href = item.get('href')
                            media_type = item.get('media-type', '')
                            
                            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                                filename = os.path.basename(href)
                                
                                # Detect special files using configured keyword lists
                                # (mirrors TransateKRtoEN._is_configured_special_file)
                                is_special = self._is_special_file(filename) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename)))
                                
                                # Add all files - UI will handle filtering based on toggle
                                manifest_chapters[item_id] = {
                                    'filename': filename,
                                    'href': href,
                                    'media_type': media_type,
                                    'is_special': is_special
                                }
                        
                        # Get spine order - the reading order
                        spine = root.find('.//opf:spine', ns)
                        
                        if spine is not None:
                            for itemref in spine.findall('opf:itemref', ns):
                                idref = itemref.get('idref')
                                if idref and idref in manifest_chapters:
                                    chapter_info = manifest_chapters[idref]
                                    filename = chapter_info['filename']
                                    is_special = chapter_info.get('is_special', False)
                                    
                                    # Extract chapter number from filename
                                    import re
                                    matches = re.findall(r'(\d+)', filename)
                                    if matches:
                                        file_chapter_num = 0 if is_special else int(matches[-1])
                                    elif is_special:
                                        # Special files without numbers should be chapter 0
                                        file_chapter_num = 0
                                    else:
                                        # Non-numbered OPF files like info.xhtml are
                                        # not real chapter numbers in the progress UI.
                                        file_chapter_num = 0
                                    
                                    # Add all files - UI will handle filtering based on toggle
                                    spine_chapters.append({
                                        'id': idref,
                                        'filename': filename,
                                        'position': len(spine_chapters),
                                        'file_chapter_num': file_chapter_num,
                                        'status': 'unknown',  # Will be updated
                                        'output_file': None,    # Will be updated
                                        'is_special': is_special
                                    })
                                    
                                    # Store the order for later use
                                    opf_chapter_order[filename] = len(spine_chapters) - 1
                                    
                                    # Also store without extension for matching
                                    filename_noext = os.path.splitext(filename)[0]
                                    opf_chapter_order[filename_noext] = len(spine_chapters) - 1
                                    opf_parsed = True
                        
            except Exception as e:
                print(f"Warning: Could not parse OPF: {e}")

        # If no OPF/spine, fall back to auto-discovery from output_dir
        if not opf_parsed or len(spine_chapters) == 0:
            if _auto_discover_from_output_dir(output_dir, prog):
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(prog, f, ensure_ascii=False, indent=2)
                    print("💾 Saved auto-discovered progress (no OPF available)")
                except Exception as e:
                    print(f"⚠️ Failed to save auto-discovered progress: {e}")
        else:
            # OPF-AWARE AUTO-DISCOVERY: Use OPF filenames as original_basename
            # This ensures correct mapping between OPF entries and response files
            progress_updated = False
            for spine_ch in spine_chapters:
                opf_filename = spine_ch['filename']  # e.g., "0009_10_.xhtml"
                base_name = os.path.splitext(opf_filename)[0]  # e.g., "0009_10_"
                
                # Look for corresponding response file on disk
                response_file = f"response_{base_name}.html"
                response_path = os.path.join(output_dir, response_file)
                
                if os.path.exists(response_path):
                    # Check if we already have a progress entry with correct original_basename
                    already_tracked = False
                    for ch_info in prog.get("chapters", {}).values():
                        if ch_info.get("original_basename") == opf_filename:
                            already_tracked = True
                            break
                        # Also check by output_file
                        if ch_info.get("output_file") == response_file:
                            # Update original_basename if missing or wrong
                            if ch_info.get("original_basename") != opf_filename:
                                ch_info["original_basename"] = opf_filename
                                progress_updated = True
                            already_tracked = True
                            break
                    
                    if not already_tracked:
                        # Create new progress entry with correct original_basename
                        chapter_num = spine_ch['file_chapter_num']
                        key = str(chapter_num) if chapter_num else f"special_{base_name}"
                        
                        # Avoid duplicate keys
                        if key not in prog.get("chapters", {}):
                            prog.setdefault("chapters", {})[key] = {
                                "actual_num": chapter_num,
                                "content_hash": "",
                                "output_file": response_file,
                                "status": "completed",
                                "last_updated": os.path.getmtime(response_path),
                                "auto_discovered": True,
                                "original_basename": opf_filename  # CORRECT: OPF filename
                            }
                            progress_updated = True
                            print(f"✅ OPF-aware discovery: {opf_filename} -> {response_file}")
            
            if progress_updated:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(prog, f, ensure_ascii=False, indent=2)
                    #print("💾 Saved OPF-aware auto-discovered progress")
                except Exception as e:
                    print(f"⚠️ Failed to save progress: {e}")
        
        # =====================================================
        # MATCH OPF CHAPTERS WITH TRANSLATION PROGRESS
        # =====================================================
        
        # Helper: normalize filenames for OPF / progress matching
        # We intentionally strip a leading "response_" prefix so that
        # files like "chapter001.xhtml" and "response_chapter001.xhtml"
        # are treated as referring to the same logical entry.
        def _normalize_opf_match_name(name: str) -> str:
            if not name:
                return ""
            base = os.path.basename(name)
            # Remove response_ prefix
            if base.startswith("response_"):
                base = base[len("response_"):]
            # Remove all extensions so that .html, .xhtml, .htm, etc. all match
            # and double extensions like .html.xhtml collapse to the stem.
            while True:
                new_base, ext = os.path.splitext(base)
                if not ext:
                    break
                base = new_base
            return base

        def _opf_names_equal(a: str, b: str) -> bool:
            return _normalize_opf_match_name(a) == _normalize_opf_match_name(b)

        # Build a map of original basenames to progress entries (normalized)
        basename_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            original_basename = chapter_info.get("original_basename", "")
            if original_basename:
                norm_key = _normalize_opf_match_name(original_basename)
                if norm_key not in basename_to_progress:
                    basename_to_progress[norm_key] = []
                basename_to_progress[norm_key].append((chapter_key, chapter_info))
        
        # Also build a map of response files (include both exact and normalized keys)
        response_file_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            if output_file:
                # Exact key
                if output_file not in response_file_to_progress:
                    response_file_to_progress[output_file] = []
                response_file_to_progress[output_file].append((chapter_key, chapter_info))
                # Normalized key (ignoring response_ prefix)
                norm_key = _normalize_opf_match_name(output_file)
                if norm_key != output_file:
                    if norm_key not in response_file_to_progress:
                        response_file_to_progress[norm_key] = []
                    response_file_to_progress[norm_key].append((chapter_key, chapter_info))
        
        # Update spine chapters with translation status
        for idx, spine_ch in enumerate(spine_chapters):
            if idx % 80 == 0:
                self._ui_yield()
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)
            
            # Find the actual response file that exists
            base_name = os.path.splitext(filename)[0]
            expected_response = None
            
            # Special files need to check what actually exists on disk
            if is_special:
                # Check for response_ prefix version
                response_with_prefix = f"response_{base_name}.html"
                retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                
                if retain:
                    expected_response = filename
                elif os.path.exists(os.path.join(output_dir, response_with_prefix)):
                    expected_response = response_with_prefix
                else:
                    # Fallback to original filename
                    expected_response = filename
            else:
                # Use OPF filename directly to avoid mismatching
                retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                if retain:
                    expected_response = filename
                else:
                    # Handle .htm.html -> .html conversion
                    stripped_base_name = base_name
                    if base_name.endswith('.htm'):
                        stripped_base_name = base_name[:-4]  # Remove .htm suffix
                    expected_response = filename  # Use exact OPF filename
                    
                    # Also check for response_ prefix version (used by the translator
                    # when TRANSLATE_ALL_NUMBERED_HTML overrides the special-file skip)
                    response_with_prefix = f"response_{base_name}.html"
                    if not os.path.exists(os.path.join(output_dir, expected_response)) and \
                       os.path.exists(os.path.join(output_dir, response_with_prefix)):
                        expected_response = response_with_prefix
            
            response_path = os.path.join(output_dir, expected_response)
            
            # Check various ways to find the translation progress info
            matched_info = None
            
            # Method 1: Check by original basename (ignoring response_ prefix)
            basename_key = _normalize_opf_match_name(filename)
            if basename_key in basename_to_progress:
                entries = basename_to_progress[basename_key]
                if entries:
                    _, chapter_info = entries[0]
                    # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                    status = chapter_info.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num:
                            matched_info = chapter_info
                    else:
                        matched_info = chapter_info
            
            # Method 2: Check by response file (with corrected extension)
            if not matched_info and expected_response in response_file_to_progress:
                entries = response_file_to_progress[expected_response]
                if entries:
                    _, chapter_info = entries[0]
                    # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                    status = chapter_info.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num:
                            matched_info = chapter_info
                    else:
                        matched_info = chapter_info
            
            # Method 3: Search through all progress entries for matching output file
            if not matched_info:
                for chapter_key, chapter_info in prog.get("chapters", {}).items():
                    out_file = chapter_info.get('output_file')
                    if out_file == expected_response or _opf_names_equal(out_file, expected_response):
                        # For in_progress/failed/qa_failed/pending, also verify actual_num matches
                        status = chapter_info.get('status', '')
                        if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                            if chapter_info.get('actual_num') == chapter_num:
                                matched_info = chapter_info
                                break
                        else:
                            matched_info = chapter_info
                            break
            
            # Method 4: CRUCIAL - Match by chapter number (actual_num vs file_chapter_num)
            # Also check composite keys for special files (e.g., "0_message", "0_TOC")
            if not matched_info:
                # First try simple chapter number key
                simple_key = str(chapter_num)
                if simple_key in prog.get("chapters", {}):
                    chapter_info = prog["chapters"][simple_key]
                    out_file = chapter_info.get('output_file')
                    status = chapter_info.get('status', '')
                    orig_base = chapter_info.get('original_basename', '')
                    if orig_base:
                        orig_base = os.path.basename(orig_base)
                    
                    # Merged chapters: check if parent exists AND original_basename matches
                    if status == 'merged':
                        parent_num = chapter_info.get('merged_parent_chapter')
                        # For merged chapters, match by original_basename (not output_file)
                        # because output_file points to parent's file, not this chapter's source file
                        # Strip extension for comparison since orig_base may not have it
                        filename_noext = os.path.splitext(filename)[0]
                        if parent_num is not None and (
                            _opf_names_equal(orig_base, filename)
                            or _opf_names_equal(orig_base, filename_noext)
                            or not orig_base
                        ):
                            parent_key = str(parent_num)
                            if parent_key in prog.get("chapters", {}):
                                # Just verify parent exists, don't enforce 'completed' status
                                # This ensures we show 'merged' even if parent is completed_empty or other states
                                matched_info = chapter_info
                    # In-progress/failed/pending chapters: require BOTH actual_num AND output_file
                    # to match to avoid cross-matching files.
                    elif status in ['in_progress', 'failed', 'pending']:
                        if chapter_info.get('actual_num') == chapter_num and (
                            out_file == expected_response or _opf_names_equal(out_file, expected_response)
                        ):
                            matched_info = chapter_info
                    # qa_failed chapters: match by chapter number only so they are always visible
                    elif status == 'qa_failed':
                        if chapter_info.get('actual_num') == chapter_num:
                            matched_info = chapter_info
                    # Normal match: output file matches expected (ignoring response_ prefix)
                    elif out_file == expected_response or _opf_names_equal(out_file, expected_response):
                        matched_info = chapter_info
                
                # If not found, check for composite key (chapter_num + filename)
                if not matched_info and is_special:
                    # For special files, try composite key format: "{chapter_num}_{filename_without_extension}"
                    base_name = os.path.splitext(filename)[0]
                    # Remove "response_" prefix if present in the filename
                    if base_name.startswith("response_"):
                        base_name = base_name[9:]
                    composite_key = f"{chapter_num}_{base_name}"
                    
                    if composite_key in prog.get("chapters", {}):
                        matched_info = prog["chapters"][composite_key]
                
                # Fallback: iterate through all entries matching chapter number,
                # but only accept when it clearly refers to the same source file.
                # This prevents files like "000_information.xhtml" and "0153_0.xhtml"
                # (both parsed as chapter 0) from being conflated.
                if not matched_info:
                    for chapter_key, chapter_info in prog.get("chapters", {}).items():
                        actual_num = chapter_info.get('actual_num')
                        # Also check 'chapter_num' as fallback
                        if actual_num is None:
                            actual_num = chapter_info.get('chapter_num')
                        
                        if actual_num is not None and actual_num == chapter_num:
                            orig_base = chapter_info.get('original_basename', '')
                            if orig_base:
                                orig_base = os.path.basename(orig_base)
                            out_file = chapter_info.get('output_file')
                            status = chapter_info.get('status', '')
                            qa_issues = chapter_info.get('qa_issues_found', [])
                            
                            # Merged chapters: match by actual_num AND original_basename
                            # For merged, output_file points to parent so we must match by source filename
                            if status == 'merged':
                                parent_num = chapter_info.get('merged_parent_chapter')
                                # Match by original_basename (the source file), not output_file (parent's file)
                                # Strip extension for comparison since orig_base may not have it
                                filename_noext = os.path.splitext(filename)[0]
                                if parent_num is not None and (
                                    _opf_names_equal(orig_base, filename)
                                    or _opf_names_equal(orig_base, filename_noext)
                                    or not orig_base
                                ):
                                    # Check if parent chapter exists
                                    parent_key = str(parent_num)
                                    if parent_key in prog.get("chapters", {}):
                                        # Just verify parent exists, don't enforce 'completed' status
                                        matched_info = chapter_info
                                        break
                            
                            # In-progress/failed/pending chapters: require BOTH actual_num AND output_file
                            # to match to avoid cross-matching files.
                            if status in ['in_progress', 'failed', 'pending']:
                                if actual_num == chapter_num and (
                                    out_file == expected_response or _opf_names_equal(out_file, expected_response)
                                ):
                                    matched_info = chapter_info
                                    break
                            # qa_failed chapters: match by chapter number only so they are always visible,
                            # even when filenames don't line up perfectly.
                            elif status == 'qa_failed':
                                if actual_num == chapter_num:
                                    matched_info = chapter_info
                                    break
                            
                            # Only treat as a match for other statuses if the original basename matches
                            # this filename, or, when original_basename is missing, the output_file matches
                            # what we expect.
                            if status not in ['in_progress', 'failed', 'qa_failed', 'pending']:
                                if (
                                    orig_base and _opf_names_equal(orig_base, filename)
                                ) or (
                                    not orig_base and out_file and (
                                        out_file == expected_response or _opf_names_equal(out_file, expected_response)
                                    )
                                ):
                                    matched_info = chapter_info
                                    break
            
            # Determine if translation file exists
            file_exists = os.path.exists(response_path)
            
            # Set status and output file based on findings
            if matched_info:
                # We found progress tracking info - use its status
                status = matched_info.get('status', 'unknown')
                spine_ch['progress_key'] = matched_info.get('_key')
                
                # CRITICAL: For failed/in_progress/qa_failed/pending, ALWAYS use progress status
                # Never let file existence override these statuses
                if status in ['failed', 'in_progress', 'qa_failed', 'pending']:
                    spine_ch['status'] = status
                    spine_ch['output_file'] = matched_info.get('output_file') or expected_response
                    spine_ch['progress_entry'] = matched_info
                    # Skip all other logic - don't check file existence
                    continue
                
                # For other statuses (completed, merged, etc.)
                spine_ch['status'] = status
                
                # For special files, always use the original filename (ignore what's in progress JSON)
                if is_special:
                    spine_ch['output_file'] = expected_response
                else:
                    spine_ch['output_file'] = matched_info.get('output_file', expected_response)
                
                spine_ch['progress_entry'] = matched_info
                
                # Handle null output_file
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response
                
                # Verify file actually exists for completed status
                if status == 'completed':
                    output_path = os.path.join(output_dir, spine_ch['output_file'])
                    if not os.path.exists(output_path):
                        # If the expected_response file exists, prefer that and
                        # transparently update the progress entry.
                        if file_exists and expected_response:
                            fixed_output_path = os.path.join(output_dir, expected_response)
                            if os.path.exists(fixed_output_path):
                                spine_ch['output_file'] = expected_response

                                # If this spine chapter is tied to a concrete
                                # progress entry, keep it consistent.
                                if 'progress_entry' in spine_ch and spine_ch['progress_entry'] is not None:
                                    spine_ch['progress_entry']['output_file'] = expected_response

                                    # Also update the master prog dict so the
                                    # corrected value is written back later.
                                    for ch_key, ch_info in prog.get('chapters', {}).items():
                                        if ch_info is spine_ch['progress_entry']:
                                            prog['chapters'][ch_key]['output_file'] = expected_response
                                            break
                            else:
                                # No matching file anywhere – mark as missing.
                                spine_ch['status'] = 'not_translated'
                        else:
                            # Legacy behaviour: nothing on disk for this entry.
                            spine_ch['status'] = 'not_translated'
            
            elif file_exists:
                # File exists but no progress tracking - mark as completed
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response
            
            else:
                # No file and no progress tracking - LAST RESORT: Try exact filename matching
                # This handles the case where progress file was deleted but files exist
                # Match by filename only (ignore response_ prefix and all extensions)
                
                def normalize_filename(fname):
                    """Remove response_ prefix and all extensions for exact comparison"""
                    base = os.path.basename(fname)
                    # Remove response_ prefix
                    if base.startswith('response_'):
                        base = base[9:]
                    # Remove all extensions (including double extensions like .html.xhtml)
                    while True:
                        new_base, ext = os.path.splitext(base)
                        if not ext:
                            break
                        base = new_base
                    return base
                
                # Normalize the OPF filename
                normalized_opf = normalize_filename(filename)
                
                # Search for exact matching file in output directory
                matched_file = None
                if os.path.exists(output_dir):
                    try:
                        for existing_file in os.listdir(output_dir):
                            if os.path.isfile(os.path.join(output_dir, existing_file)):
                                normalized_existing = normalize_filename(existing_file)
                                # Exact match only - no fuzzy logic
                                if normalized_existing == normalized_opf:
                                    matched_file = existing_file
                                    break
                    except Exception as e:
                        print(f"Warning: Error scanning output directory for match: {e}")
                
                if matched_file:
                    # Found an exact matching file by normalized name - mark as completed
                    spine_ch['status'] = 'completed'
                    spine_ch['output_file'] = matched_file
                    print(f"📁 Matched: {filename} -> {matched_file}")
                else:
                    # No file and no progress tracking - not translated
                    spine_ch['status'] = 'not_translated'
                    spine_ch['output_file'] = expected_response
        
        # =====================================================
        # SAVE AUTO-DISCOVERED FILES TO PROGRESS
        # =====================================================
        
        # Check if we discovered any new completed files (exact matched by normalized filename)
        # and add them to the progress file
        progress_updated = False
        for spine_ch in spine_chapters:
            # Only add entries that were marked as completed but have no progress entry
            if spine_ch['status'] == 'completed' and 'progress_entry' not in spine_ch:
                chapter_num = spine_ch['file_chapter_num']
                output_file = spine_ch['output_file']
                filename = spine_ch['filename']
                
                # Create a progress entry for this auto-discovered file
                chapter_key = str(chapter_num)
                
                # Check if key already exists (avoid duplicates)
                # If the key exists but points to a DIFFERENT file, use a composite
                # key to avoid overwriting (e.g. chapter0003 vs chapter_notice0003).
                existing = prog.get("chapters", {}).get(chapter_key)
                if existing:
                    existing_out = existing.get('output_file', '')
                    existing_base = existing.get('original_basename', '')
                    # If same output file, this is already tracked
                    if existing_out == output_file:
                        continue
                    # Different file occupies this key — use composite key
                    base_noext = os.path.splitext(filename)[0]
                    chapter_key = f"{chapter_num}_{base_noext}"
                    # Also skip if composite key already exists
                    if chapter_key in prog.get("chapters", {}):
                        continue
                
                prog.setdefault("chapters", {})[chapter_key] = {
                    "actual_num": chapter_num,
                    "content_hash": "",  # Unknown since we don't have the source
                    "output_file": output_file,
                    "status": "completed",
                    "last_updated": os.path.getmtime(os.path.join(output_dir, output_file)),
                    "auto_discovered": True,
                    "original_basename": filename
                }
                progress_updated = True
                print(f"✅ Auto-discovered and tracked: {filename} -> {output_file} (key: {chapter_key})")
        
        # Save progress file if we added new entries
        if progress_updated:
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved {sum(1 for ch in spine_chapters if ch['status'] == 'completed' and 'progress_entry' not in ch)} auto-discovered files to progress file")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress file: {e}")
        
        # =====================================================
        # BUILD DISPLAY INFO
        # =====================================================
        
        chapter_display_info = []
        
        if spine_chapters:
            # Use OPF order
            for spine_ch in spine_chapters:
                display_info = {
                    'key': spine_ch.get('filename', ''),
                    'num': spine_ch['file_chapter_num'],
                    'info': spine_ch.get('progress_entry', {}),
                    'output_file': spine_ch['output_file'],
                    'status': spine_ch['status'],
                    'duplicate_count': 1,
                    'entries': [],
                    'opf_position': spine_ch['position'],
                    'original_filename': spine_ch['filename'],
                    'is_special': spine_ch.get('is_special', False)
                }
                chapter_display_info.append(display_info)
        else:
            # Fallback to original logic if no OPF
            # Known non-chapter files that should never appear in the progress list
            _non_chapter_files = {"glossary.csv", "metadata.json", "styles.css", "rolling_summary.txt"}
            _source_has_translated = "_translated" in os.path.basename(file_path).lower()
            files_to_entries = {}
            for chapter_key, chapter_info in prog.get("chapters", {}).items():
                output_file = chapter_info.get("output_file", "")
                status = chapter_info.get("status", "")
                
                # Skip known non-chapter files
                if output_file and output_file.lower() in _non_chapter_files:
                    continue
                # Skip combined _translated output files (unless source itself has _translated)
                if output_file and not _source_has_translated and any(
                    output_file.lower().endswith(s) for s in ("_translated.txt", "_translated.pdf", "_translated.html")
                ):
                    continue
                
                # Include chapters with output files OR transient statuses with null output file (legacy)
                # (composite keys like "0_TOC" should still be represented in the UI)
                if output_file or status in ["in_progress", "pending", "failed", "qa_failed"]:
                    # For merged chapters, use a unique key (chapter_key) instead of output_file
                    # This ensures merged chapters appear as separate entries in the list
                    if status == "merged":
                        file_key = f"_merged_{chapter_key}"
                    elif output_file:
                        file_key = output_file
                    elif status == "in_progress":
                        file_key = f"_in_progress_{chapter_key}"
                    elif status == "pending":
                        file_key = f"_pending_{chapter_key}"
                    elif status == "qa_failed":
                        file_key = f"_qa_failed_{chapter_key}"
                    else:  # failed
                        file_key = f"_failed_{chapter_key}"
                    
                    if file_key not in files_to_entries:
                        files_to_entries[file_key] = []
                    files_to_entries[file_key].append((chapter_key, chapter_info))
            
            for output_file, entries in files_to_entries.items():
                chapter_key, chapter_info = entries[0]
                
                # Get the actual output file (strip placeholder prefix if present)
                actual_output_file = output_file
                if (
                    output_file.startswith("_merged_")
                    or output_file.startswith("_in_progress_")
                    or output_file.startswith("_pending_")
                    or output_file.startswith("_failed_")
                    or output_file.startswith("_qa_failed_")
                ):
                    # For merged/in_progress/pending/failed/qa_failed, get the actual output_file from chapter_info
                    actual_output_file = chapter_info.get("output_file", "")
                    if not actual_output_file:
                        # Generate expected filename based on actual_num
                        actual_num = chapter_info.get("actual_num")
                        if actual_num is not None:
                            # Use .txt extension for text files, .html for EPUB
                            ext = ".txt" if file_path.endswith(".txt") else ".html"
                            actual_output_file = f"response_section_{actual_num}{ext}"
                
                # Check if this is a special file (files without numbers)
                original_basename = chapter_info.get("original_basename", "")
                filename_to_check = original_basename if original_basename else actual_output_file
                
                is_special = self._is_special_file(filename_to_check) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename_to_check)))
                
                # Extract chapter number - prioritize stored values
                chapter_num = None
                if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                    chapter_num = chapter_info['actual_num']
                elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                    chapter_num = chapter_info['chapter_num']
                
                # Fallback: extract from filename
                if chapter_num is None:
                    import re
                    matches = re.findall(r'(\d+)', actual_output_file)
                    if matches:
                        chapter_num = int(matches[-1])
                    else:
                        chapter_num = 999999
                
                status = chapter_info.get("status", "unknown")
                if status in ("completed_empty", "completed_image_only"):
                    status = "completed"
                
                # Check file existence
                if status == "completed":
                    output_path = os.path.join(output_dir, actual_output_file)
                    if not os.path.exists(output_path):
                        status = "file_missing"
                
                chapter_display_info.append({
                    'key': chapter_key,
                    'num': chapter_num,
                    'info': chapter_info,
                    'output_file': actual_output_file,  # Use actual output file, not placeholder
                    'status': status,
                    'duplicate_count': len(entries),
                    'entries': entries,
                    'is_special': is_special
                })
            
            # Sort by chapter number
            chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)

        self._append_pdf_ocr_display_info({'prog': prog, 'file_path': file_path, 'output_dir': output_dir}, chapter_display_info)
        self._append_image_gen_display_info({'prog': prog, 'file_path': file_path, 'output_dir': output_dir}, chapter_display_info)
        
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
            # Use 38% width, 40% height for 1920x1080
            width, height = self._get_dialog_size(0.38, 0.4)
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
        
        title_text = "Chapters from content.opf (in reading order):" if spine_chapters else "Select chapters to retranslate:"
        title_label = QLabel(title_text)
        title_font = QFont('Arial', 12 if not tab_frame else 11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Add toggle for showing special files
        from PySide6.QtWidgets import QCheckBox
        show_special_files_cb = QCheckBox("Show skipped files")
        show_special_files_cb.setChecked(show_special_files[0])  # Preserve the current state
        show_special_files_cb.setToolTip("When enabled, shows files that would be skipped during translation\n(matching the special file keywords configured in Other Settings).")
        
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
        def _glossary_progress_search_dirs(base):
            """Return likely glossary progress locations, newest per-book layout first."""
            search_dirs = []
            seen = set()

            def _add(path):
                if not path:
                    return
                path = os.path.abspath(path)
                key = os.path.normcase(path)
                if key not in seen:
                    seen.add(key)
                    search_dirs.append(path)

            def _add_root(root):
                if not root:
                    return
                root = os.path.abspath(root)
                shared = os.path.join(root, 'Glossary')
                try:
                    from glossary_paths import get_book_glossary_dir
                    _add(get_book_glossary_dir(shared, base, create=False))
                except Exception:
                    _add(os.path.join(shared, base))
                _add(shared)
                _add(os.path.join(root, base, 'Glossary'))
                _add(os.path.join(root, base))

            _override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
            if not _override_dir and hasattr(self, 'config'):
                _override_dir = self.config.get('output_directory')
            if _override_dir:
                _add_root(_override_dir)

            try:
                from translator_gui import _get_app_dir
                _app_dir = _get_app_dir()
            except Exception:
                _app_dir = os.getcwd()
            _add_root(_app_dir)

            if hasattr(self, 'base_dir'):
                _add_root(self.base_dir)

            return search_dirs

        def _find_progress_in_dir(directory, progress_name):
            candidate = os.path.join(directory, progress_name)
            if os.path.isfile(candidate):
                return candidate
            generic = os.path.join(directory, 'glossary_progress.json')
            if os.path.isfile(generic):
                return generic
            if os.path.basename(directory).lower() != 'glossary':
                try:
                    matches = [
                        os.path.join(directory, name)
                        for name in os.listdir(directory)
                        if name.lower().endswith('_glossary_progress.json')
                    ]
                    if matches:
                        return max(matches, key=lambda path: os.path.getmtime(path))
                except Exception:
                    pass
            return None

        def _find_glossary_progress_file():
            """Locate the glossary progress file for the current EPUB."""
            try:
                base = os.path.splitext(os.path.basename(file_path))[0]
                progress_name = f"{base}_glossary_progress.json"
                for d in _glossary_progress_search_dirs(base):
                    if not os.path.isdir(d):
                        continue
                    found = _find_progress_in_dir(d, progress_name)
                    if found:
                        return found
            except Exception:
                pass
            return None
        
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
        def _find_gp_for_file(fp):
            """Locate the glossary progress file for a given EPUB path."""
            try:
                base = os.path.splitext(os.path.basename(fp))[0]
                progress_name = f"{base}_glossary_progress.json"
                for d in _glossary_progress_search_dirs(base):
                    if not os.path.isdir(d):
                        continue
                    found = _find_progress_in_dir(d, progress_name)
                    if found:
                        return found
            except Exception:
                pass
            return None

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

        text_analysis_btn = QPushButton("🔍 Review source -> output")
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
            sidecars = _text_analysis_sidecars()
            if not sidecars:
                stats = self._generate_sdlxliff_sidecars_from_completed_entries(
                    output_dir,
                    file_path=file_path,
                    progress_data=prog,
                    overwrite=True,
                )
                sidecars = _text_analysis_sidecars()
                if sidecars:
                    _queue_text_analysis_button_update()
            if not sidecars:
                detail = ""
                try:
                    detail = (
                        f"\n\nAuto-generation considered {stats.get('considered', 0)} completed entries; "
                        f"missing source: {stats.get('missing_source', 0)}, "
                        f"missing output: {stats.get('missing_output', 0)}, "
                        f"failed: {stats.get('failed', 0)}."
                    )
                except Exception:
                    detail = ""
                self._show_message(
                    'info',
                    "Text Analysis Unavailable",
                    "No SDLXLIFF sidecars were found or generated for this output folder." + detail,
                    parent=dialog,
                )
                return
            try:
                self._open_or_reuse_sdlxliff_review(output_dir, None, dialog)
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=dialog)

        text_analysis_btn.clicked.connect(_show_text_analysis)
        _update_text_analysis_button()

        def _queue_text_analysis_button_update():
            for delay in (0, 75, 250, 750):
                QTimer.singleShot(delay, _update_text_analysis_button)
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

        def _bool_setting(value):
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 'yes', 'on')
            return bool(value)

        def _glossary_refinement_settings_enabled():
            try:
                cfg = getattr(self, 'config', {}) or {}
                if _bool_setting(cfg.get('glossary_refinement_enabled', False)):
                    return True
                return os.getenv('GLOSSARY_REFINEMENT_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')
            except Exception:
                return False

        def _glossary_refinement_expected_entries():
            if not _glossary_refinement_settings_enabled():
                return {}
            cfg = getattr(self, 'config', {}) or {}
            custom_types = getattr(self, 'custom_entry_types', None) or cfg.get('custom_entry_types', {}) or {}
            if not isinstance(custom_types, dict) or not custom_types:
                custom_types = {
                    'character': {'enabled': True},
                    'terms': {'enabled': True},
                }

            active_types = []
            for type_name, type_cfg in custom_types.items():
                if isinstance(type_cfg, dict) and not type_cfg.get('enabled', True):
                    continue
                type_name = str(type_name or '').strip()
                if type_name:
                    active_types.append(type_name)

            type_mode = str(cfg.get('glossary_refinement_type_mode', 'all') or 'all').lower()
            if type_mode == 'selected':
                selected = cfg.get('glossary_refinement_selected_types', [])
                if isinstance(selected, str):
                    selected = [t.strip() for t in selected.split(',') if t.strip()]
                selected_lc = {str(t).strip().lower() for t in selected if str(t).strip()}
                active_types = [t for t in active_types if t.lower() in selected_lc]

            if not active_types:
                return {}

            chunking_mode = str(cfg.get('glossary_refinement_chunking_mode', 'separate') or 'separate').lower()
            if chunking_mode in ('all', 'all_types', 'all_in_one'):
                entry_type = 'all selected entry types'
                return {
                    f"all::{','.join(active_types)}": {
                        'entry_type': entry_type,
                        'status': 'not_refined',
                        'chunking_mode': 'all',
                    }
                }

            return {
                f"type::{entry_type}": {
                    'entry_type': entry_type,
                    'status': 'not_refined',
                    'chunking_mode': 'separate',
                }
                for entry_type in active_types
            }

        if _glossary_refinement_settings_enabled():
            glossary_progress_btn.setVisible(True)
        
        def _build_gp_panel(fp, gp_path, parent_widget, pump_loading=None):
            """Build a glossary progress panel for a single EPUB. Returns (panel_widget, refresh_func)."""
            from PySide6.QtWidgets import QStackedWidget, QComboBox

            def _pump_loading_frame():
                if callable(pump_loading):
                    try:
                        pump_loading()
                    except Exception:
                        pass
            
            def _gp_load_progress_dict(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                except Exception as e:
                    print(f"⚠️ Could not load glossary progress file {path}: {e}")
                    return {}
                if isinstance(loaded, dict):
                    return loaded
                print(f"⚠️ Glossary progress file has legacy non-dict shape: {type(loaded).__name__}")
                return {}

            def _gp_int_list(values):
                if values is None:
                    return []
                if isinstance(values, dict):
                    values = values.keys()
                elif isinstance(values, (str, int, float)):
                    values = [values]
                result = []
                seen = set()
                try:
                    iterator = iter(values)
                except TypeError:
                    iterator = iter([values])
                for value in iterator:
                    if isinstance(value, dict):
                        value = value.get('chapter_index', value.get('actual_num', value.get('chapter_num')))
                    try:
                        ivalue = int(value)
                    except (TypeError, ValueError):
                        continue
                    if ivalue not in seen:
                        seen.add(ivalue)
                        result.append(ivalue)
                return result

            def _gp_safe_int(value, default=0):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            gp_data = _gp_load_progress_dict(gp_path)
            _pump_loading_frame()

            completed_indices = _gp_int_list(gp_data.get('completed', []))
            failed_indices = _gp_int_list(gp_data.get('failed', []))
            merged_indices = _gp_int_list(gp_data.get('merged_indices', []))
            book_title = gp_data.get('book_title', '')

            def _gp_qa_issue_map(_d):
                if not isinstance(_d, dict):
                    _d = {}
                issues = {}

                def _add(idx, values):
                    try:
                        key = int(idx)
                    except (TypeError, ValueError):
                        return
                    if isinstance(values, str):
                        values = [values]
                    if not isinstance(values, list):
                        return
                    bucket = issues.setdefault(key, [])
                    for value in values:
                        text = str(value).strip()
                        if text and text not in bucket:
                            bucket.append(text)

                raw_map = _d.get('qa_issues_found', {})
                if isinstance(raw_map, dict):
                    for idx, values in raw_map.items():
                        if isinstance(values, dict):
                            values = values.get('qa_issues_found') or values.get('issues') or []
                        mapped_idx = _gp_index_for_progress_value(idx, _d)
                        _add(mapped_idx if mapped_idx is not None else idx, values)

                chapters = _d.get('chapters', {})
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        idx = _gp_index_for_entry(info, key, _d)
                        _add(idx, info.get('qa_issues_found', []))
                return issues

            def _gp_filename_chapter_num(fname):
                import re as _re_gp_num
                nums = _re_gp_num.findall(r'[0-9]+', os.path.splitext(str(fname or ""))[0])
                if nums:
                    try:
                        return int(nums[-1])
                    except (TypeError, ValueError):
                        return None
                return None

            def _gp_display_chapter_num(ci, fname):
                ch_num = _gp_filename_chapter_num(fname)
                if ch_num is not None:
                    return ch_num
                return 0

            def _gp_filename_keys(name):
                """Normalize a filename into a set of lowercase lookup keys."""
                base = os.path.basename(str(name or ""))
                if not base:
                    return set()
                stem = os.path.splitext(base)[0]
                keys = {base.lower(), stem.lower()}
                if stem.lower().startswith('response_'):
                    keys.add(stem[9:].lower())
                return {k for k in keys if k}

            def _rebuild_reverse_lookups():
                """Rebuild O(1) lookup dicts from chapter_map. Call after chapter_map changes."""
                cmap = panel_state.get('chapter_map') or {}
                # filename key -> chapter index (first wins)
                fk_to_ci = {}
                for lookup_idx, (ci, mapped_name) in enumerate(cmap.items()):
                    for key in _gp_filename_keys(mapped_name):
                        fk_to_ci.setdefault(key, ci)
                    if lookup_idx and lookup_idx % 200 == 0:
                        _pump_loading_frame()
                panel_state['_fk_to_ci'] = fk_to_ci
                # actual_num (from filename) -> list of chapter indices
                anum_to_ci = {}
                for lookup_idx, (ci, fname) in enumerate(cmap.items()):
                    num = _gp_filename_chapter_num(fname)
                    if num is not None:
                        anum_to_ci.setdefault(num, []).append(ci)
                    if lookup_idx and lookup_idx % 200 == 0:
                        _pump_loading_frame()
                panel_state['_anum_to_ci'] = anum_to_ci
                # auto-completed (cover pages) — cached set
                auto_comp = set()
                for lookup_idx, (ci, fname) in enumerate(cmap.items()):
                    stem = os.path.splitext(os.path.basename(str(fname or "")))[0].lower()
                    if stem == 'cover':
                        auto_comp.add(ci)
                    if lookup_idx and lookup_idx % 200 == 0:
                        _pump_loading_frame()
                panel_state['_auto_completed'] = auto_comp

            def _gp_auto_completed_indices():
                return panel_state.get('_auto_completed') or set()

            def _gp_index_for_actual_num(actual_num, _d=None):
                try:
                    actual_num = int(actual_num)
                except (TypeError, ValueError):
                    return None
                # O(1) reverse lookup from cached dict
                matches = list(panel_state.get('_anum_to_ci', {}).get(actual_num, []))
                if not matches:
                    chapter_numbers = (_d or gp_data).get('chapter_numbers', {})
                    if isinstance(chapter_numbers, dict):
                        for ci, num in chapter_numbers.items():
                            try:
                                if int(num) == actual_num:
                                    matches.append(int(ci))
                            except (TypeError, ValueError):
                                pass
                if len(matches) == 1:
                    return matches[0]
                return None

            def _gp_index_for_entry(info, key=None, _d=None):
                if not isinstance(info, dict):
                    info = {}

                had_filename_anchor = False
                fk_to_ci = panel_state.get('_fk_to_ci') or {}
                for fname_key in ('output_file', 'original_basename', 'chapter_file', 'source_filename', 'filename'):
                    fname = os.path.basename(str(info.get(fname_key, "") or ""))
                    if not fname:
                        continue
                    had_filename_anchor = True
                    # O(1) lookup via reverse dict instead of scanning chapter_map
                    for k in _gp_filename_keys(fname):
                        if k in fk_to_ci:
                            return fk_to_ci[k]
                if had_filename_anchor:
                    return None

                for num_key in ('actual_num', 'chapter_num'):
                    ci = _gp_index_for_actual_num(info.get(num_key), _d)
                    if ci is not None:
                        return ci

                if key is not None and 'chapter_index' not in info:
                    ci = _gp_index_for_actual_num(key, _d)
                    if ci is not None:
                        return ci

                try:
                    return int(info.get('chapter_index', key))
                except (TypeError, ValueError):
                    return None

            def _gp_index_for_progress_value(value, _d):
                if not isinstance(_d, dict):
                    _d = {}
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    return None
                if str(_d.get('indexing', '')).lower() == 'chapter_index_zero_based':
                    return ivalue
                positions = _d.get('chapter_positions', {})
                if isinstance(positions, dict) and str(ivalue) in positions:
                    return ivalue
                ci = _gp_index_for_actual_num(ivalue, _d)
                return ci if ci is not None else ivalue

            def _gp_sets(_d):
                if not isinstance(_d, dict):
                    _d = {}

                def _index_set(values):
                    result = set()
                    for value in _gp_int_list(values):
                        ci = _gp_index_for_progress_value(value, _d)
                        if ci is not None:
                            result.add(ci)
                    return result

                comp = set()
                fail = set()
                merg = set()
                chapters = _d.get('chapters', {})
                used_chapter_entries = False
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        ci = _gp_index_for_entry(info, key, _d)
                        if ci is None:
                            continue
                        used_chapter_entries = True
                        status = str(info.get('status', '')).lower()
                        if status in ('failed', 'qa_failed', 'error'):
                            fail.add(ci)
                        elif status == 'merged':
                            merg.add(ci)
                        elif status == 'completed':
                            comp.add(ci)
                if not used_chapter_entries:
                    comp = _index_set(_d.get('completed', []))
                    fail = _index_set(_d.get('failed', []))
                    merg = _index_set(_d.get('merged_indices', []))
                comp |= _gp_auto_completed_indices()
                # Failed should win over completed in the UI.
                comp -= fail
                return comp, fail, merg

            def _gp_in_progress_set(_d, _precomputed_sets=None):
                if not isinstance(_d, dict):
                    _d = {}
                result = set()
                chapters = _d.get('chapters', {})
                used_chapter_entries = False
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        status = str(info.get('status', '')).lower()
                        if status:
                            used_chapter_entries = True
                        if status != 'in_progress':
                            continue
                        ci = _gp_index_for_entry(info, key, _d)
                        if ci is not None:
                            result.add(ci)
                if not used_chapter_entries:
                    for value in _gp_int_list(_d.get('in_progress', [])):
                        ci = _gp_index_for_progress_value(value, _d)
                        if ci is not None:
                            result.add(ci)
                if _precomputed_sets:
                    comp, fail, merg = _precomputed_sets
                else:
                    comp, fail, merg = _gp_sets(_d)
                return result - comp - fail - merg

            def _gp_status_cache(_d):
                comp, fail, merg = _gp_sets(_d)
                in_prog = _gp_in_progress_set(_d, _precomputed_sets=(comp, fail, merg))
                issues = _gp_qa_issue_map(_d)
                qa_failed = set()
                chapters = _d.get('chapters', {}) if isinstance(_d, dict) else {}
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        if str(info.get('status', '')).lower() != 'qa_failed':
                            continue
                        ci = _gp_index_for_entry(info, key, _d)
                        if ci is not None:
                            qa_failed.add(ci)
                return {
                    'completed': comp,
                    'failed': fail,
                    'merged': merg,
                    'in_progress': in_prog,
                    'issues': issues,
                    'qa_failed': qa_failed,
                }

            def _gp_status_for(ci, _d, cache=None):
                if not isinstance(_d, dict):
                    _d = {}
                cache = cache or _gp_status_cache(_d)
                comp = cache['completed']
                fail = cache['failed']
                merg = cache['merged']
                in_prog = cache['in_progress']
                issues = cache['issues']
                if ci in fail:
                    return ('qa_failed' if issues.get(ci) or ci in cache['qa_failed'] else 'failed'), issues.get(ci, [])
                if ci in merg:
                    return 'merged', []
                if ci in comp:
                    return 'completed', []
                if ci in in_prog:
                    return 'in_progress', []
                return 'not_completed', []

            def _gp_model_for(ci, _d):
                if not isinstance(_d, dict):
                    _d = {}
                chapters = _d.get('chapters', {})
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        if _gp_index_for_entry(info, key, _d) != ci:
                            continue
                        model_name = str(info.get('model_name') or info.get('model') or '').strip()
                        if model_name:
                            return model_name
                return '(model unknown)'

            def _gp_entry_for(ci, _d):
                if not isinstance(_d, dict):
                    return {}
                chapters = _d.get('chapters', {})
                if isinstance(chapters, dict):
                    for key, info in chapters.items():
                        if not isinstance(info, dict):
                            continue
                        if _gp_index_for_entry(info, key, _d) == ci:
                            return info
                return {}

            def _gp_display_for(ci, fname, _d, cache=None):
                opf_pos = (panel_state.get('spine_index_map') or {}).get(ci, ci + 1)
                ch_num = _gp_display_chapter_num(ci, fname)
                status, issues = _gp_status_for(ci, _d, cache)
                model_name = _gp_model_for(ci, _d)
                icons = {
                    'completed': '\u2705',
                    'failed': '\u274c',
                    'qa_failed': '\u274c',
                    'merged': '\U0001f517',
                    'in_progress': '\U0001f504',
                    'not_completed': '\u2b1c',
                }
                icon = icons.get(status) or '\u2b1c'
                status_label = status.replace('_', ' ').title()
                entry = _gp_entry_for(ci, _d)
                if status == 'completed' and str(entry.get('refinement_status') or '').lower().strip() in ('refined', 'completed'):
                    status_label = f"{status_label} ⭐"
                display = f"[{opf_pos:03d}] Ch.{ch_num:03d} | {icon} {status_label:14s} | {fname} -> {model_name}"
                if issues:
                    issues_display = ', '.join(issues[:2])
                    if len(issues) > 2:
                        issues_display += f' (+{len(issues)-2} more)'
                    display += f" | {issues_display}"
                return display, status

            def _gp_refinement_rows(_d):
                refinement = _d.get('refinement', {}) if isinstance(_d, dict) else {}
                if not isinstance(refinement, dict):
                    refinement = {}
                refinement = dict(refinement)
                for expected_key, expected_info in _glossary_refinement_expected_entries().items():
                    refinement.setdefault(expected_key, expected_info)
                rows = []
                for key, info in sorted(refinement.items()):
                    if not isinstance(info, dict):
                        continue
                    entry_type = str(info.get('entry_type') or key.replace('type::', '')).strip() or 'entry type'
                    status = str(info.get('status') or 'unknown').lower()
                    before = info.get('entry_count_before')
                    after = info.get('entry_count_after')
                    total_chunks = info.get('total_chunks')
                    completed_chunks = info.get('completed_chunks')
                    model_name = str(info.get('model_name') or info.get('model') or '').strip() or '(model unknown)'
                    detail = ""
                    if before is not None and after is not None:
                        detail = f" | {before} -> {after} entries"
                    elif total_chunks:
                        detail = f" | chunks {completed_chunks or 0}/{total_chunks}"
                    icon_map = {
                        'completed': '\u2705',
                        'failed': '\u274c',
                        'qa_failed': '\u274c',
                        'in_progress': '\U0001f504',
                        'not_refined': '\u2728',
                    }
                    icon = icon_map.get(status, '\u2b1c')
                    display = f"Refinement | {icon} {status.replace('_', ' ').title():14s} | {entry_type} -> {model_name}{detail}"
                    rows.append((key, display, status))
                return rows

            def _gp_refinement_status_counts(_d):
                counts = {}
                for _key, _display, status in _gp_refinement_rows(_d):
                    status = str(status or 'unknown').lower().replace(' ', '_')
                    counts[status] = counts.get(status, 0) + 1
                return counts

            def _gp_color_for(status):
                if status == 'completed':
                    return '#27ae60'
                if status == 'merged':
                    return '#17a2b8'
                if status == 'in_progress':
                    return '#f59e0b'
                if status in ('failed', 'qa_failed'):
                    return '#e74c3c'
                return '#5a9fd4'

            def _gp_restore_in_progress_entry(info):
                if not isinstance(info, dict):
                    return None
                previous_status = str(info.get('previous_status', '') or '').lower()
                previous_entry = info.get('previous_progress_entry')
                if isinstance(previous_entry, dict):
                    restored = dict(previous_entry)
                    restored_status = str(restored.get('status', previous_status) or previous_status).lower()
                    if restored_status and restored_status not in ('in_progress', 'not_completed', 'not translated', 'not_translated'):
                        restored.pop('previous_status', None)
                        restored.pop('previous_progress_entry', None)
                        return restored
                if previous_status in ('qa_failed', 'failed', 'error', 'pending', 'merged', 'completed'):
                    restored = dict(info)
                    restored['status'] = 'failed' if previous_status == 'error' else previous_status
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
                if info.get('previous_status_unknown'):
                    restored = dict(info)
                    restored['status'] = 'failed'
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
                if previous_status in ('not_completed', 'not translated', 'not_translated'):
                    return None
                if info.get('output_file'):
                    restored = dict(info)
                    restored['status'] = 'failed'
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
                return None

            # Lightweight spine reader - returns (chapter_map, total_chapters, spine_index_map)
            def _read_spine_map(epub_path, translate_special):
                """Read OPF spine and return (chapter_map, total_chapters, spine_index_map)."""
                cmap = {}
                spine_index_map = {}
                if not (epub_path.lower().endswith('.epub') and os.path.exists(epub_path)):
                    return cmap, 0, spine_index_map
                try:
                    import zipfile
                    from xml.etree import ElementTree as ET
                    with zipfile.ZipFile(epub_path, 'r') as zf:
                        opf_path = None
                        try:
                            container = ET.fromstring(zf.read('META-INF/container.xml'))
                            ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                            rootfile = container.find('.//c:rootfile', ns)
                            if rootfile is not None:
                                opf_path = rootfile.get('full-path')
                        except Exception:
                            opf_path = next((n for n in zf.namelist() if n.endswith('.opf')), None)
                        
                        if not opf_path:
                            return cmap, 0, spine_index_map
                        
                        opf_xml = ET.fromstring(zf.read(opf_path))
                        opf_ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        
                        id_to_href = {}
                        html_types = {'application/xhtml+xml', 'text/html', 'application/html+xml'}
                        for manifest_idx, item in enumerate(opf_xml.findall('.//opf:manifest/opf:item', opf_ns)):
                            mid = item.get('id', '')
                            mtype = item.get('media-type', '')
                            href = item.get('href', '')
                            if mtype in html_types:
                                id_to_href[mid] = href
                            if manifest_idx and manifest_idx % 200 == 0:
                                _pump_loading_frame()
                        
                        spine_hrefs = []
                        for spine_ref_idx, itemref in enumerate(opf_xml.findall('.//opf:spine/opf:itemref', opf_ns)):
                            idref = itemref.get('idref', '')
                            if idref in id_to_href:
                                spine_hrefs.append(id_to_href[idref])
                            if spine_ref_idx and spine_ref_idx % 200 == 0:
                                _pump_loading_frame()
                        
                        _kw_env = os.environ.get('SPECIAL_FILE_KEYWORDS', '')
                        special_keywords = [k.strip().lower() for k in _kw_env.split(',') if k.strip()] if _kw_env else [
                            'title', 'toc', 'copyright', 'preface', 'nav',
                            'message', 'notice', 'colophon', 'dedication', 'epigraph',
                            'foreword', 'acknowledgment', 'author', 'appendix',
                            'bibliography'
                        ]
                        _exact_env = os.environ.get('SPECIAL_FILE_EXACT', '')
                        special_exact = [k.strip().lower() for k in _exact_env.split(',') if k.strip()] if _exact_env else ['index', 'glossary', 'glossary_extension']
                        import re as _re_spine
                        ci = 0
                        for opf_pos, href in enumerate(spine_hrefs, start=1):
                            basename = os.path.basename(href)
                            if not translate_special:
                                name_noext = os.path.splitext(basename)[0]
                                name_lower = name_noext.lower()
                                name_stripped = _re_spine.sub(r'\d+$', '', name_lower).rstrip('_- ')
                                # Exact match: these are special only when the basename matches exactly
                                if name_lower in special_exact:
                                    continue
                                if any(kw in name_lower for kw in special_keywords):
                                    has_digits = bool(_re_spine.search(r'\d', name_noext))
                                    if not has_digits or any(kw == name_stripped or kw in name_stripped for kw in special_keywords):
                                        continue
                            cmap[ci] = basename
                            spine_index_map[ci] = opf_pos
                            ci += 1
                            if opf_pos % 200 == 0:
                                _pump_loading_frame()
                        return cmap, ci, spine_index_map
                except Exception:
                    return cmap, 0, spine_index_map
            
            # Mutable state so refresh can update chapter_map when toggle changes
            _ts_init = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
            _cmap_init, _total_init, _spine_idx_init = _read_spine_map(fp, _ts_init)
            _pump_loading_frame()
            
            if _total_init == 0:
                _total_init = _gp_safe_int(gp_data.get('chapter_count'), 0)
                if _total_init <= 0:
                    chapter_filenames = gp_data.get('chapter_filenames', {})
                    if isinstance(chapter_filenames, dict) and chapter_filenames:
                        _total_init = max((int(k) for k in chapter_filenames.keys() if str(k).isdigit()), default=-1) + 1
                if _total_init <= 0:
                    _idx_values = []
                    for _values in (completed_indices, failed_indices, merged_indices):
                        _idx_values.extend(_gp_int_list(_values))
                    _total_init = (max(_idx_values) + 1) if _idx_values else 1
            
            # Store in mutable dict so closures can update
            panel_state = {
                'chapter_map': _cmap_init,
                'spine_index_map': _spine_idx_init,
                'total': _total_init,
                'translate_special': _ts_init,
                'populate_generation': 0,
            }
            if not panel_state['chapter_map']:
                chapter_filenames = gp_data.get('chapter_filenames', {})
                if isinstance(chapter_filenames, dict):
                    try:
                        panel_state['chapter_map'] = {
                            int(k): os.path.basename(str(v or ""))
                            for k, v in chapter_filenames.items()
                            if str(k).lstrip('-').isdigit() and v
                        }
                        panel_state['spine_index_map'] = {
                            int(k): int(k) + 1
                            for k in chapter_filenames.keys()
                            if str(k).lstrip('-').isdigit()
                        }
                    except Exception:
                        panel_state['chapter_map'] = {}
                        panel_state['spine_index_map'] = {}
            
            # Build O(1) reverse lookups from chapter_map
            _rebuild_reverse_lookups()
            _pump_loading_frame()
            
            # Track file mtime for dirty-checking on refresh
            try:
                panel_state['_last_mtime'] = os.path.getmtime(gp_path) if os.path.isfile(gp_path) else 0
            except OSError:
                panel_state['_last_mtime'] = 0
            
            panel = QWidget(parent_widget)
            p_layout = QVBoxLayout(panel)
            p_layout.setContentsMargins(4, 4, 4, 4)
            
            if book_title:
                bt_label = QLabel(f"📖 {book_title}")
                bt_label.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 10pt;")
                p_layout.addWidget(bt_label)
            else:
                bt_label = None
            
            # Stats row (clickable)
            _comp_set_init, _fail_set_init, _merg_set_init = _gp_sets(gp_data)
            _in_prog_set_init = _gp_in_progress_set(gp_data, _precomputed_sets=(_comp_set_init, _fail_set_init, _merg_set_init))
            # Completed count excludes chapters that are also merged
            n_completed = len(_comp_set_init - _merg_set_init - _fail_set_init)
            n_failed = len(_fail_set_init)
            n_merged = len(_merg_set_init)
            n_in_progress = len(_in_prog_set_init)
            n_remaining = max(0, panel_state['total'] - len(_comp_set_init | _fail_set_init | _merg_set_init | _in_prog_set_init))
            n_not_refined = _gp_refinement_status_counts(gp_data).get('not_refined', 0)
            
            gp_stats_frame = QWidget()
            gp_stats_layout = QHBoxLayout(gp_stats_frame)
            gp_stats_layout.setContentsMargins(0, 5, 0, 5)
            gp_stats_font = QFont('Arial', 10)
            
            lbl_total = QLabel(f"Total: {panel_state['total']} | ")
            lbl_total.setFont(gp_stats_font)
            gp_stats_layout.addWidget(lbl_total)
            
            lbl_gp_completed = QLabel(f"✅ Completed: {n_completed} | ")
            lbl_gp_completed.setFont(gp_stats_font)
            lbl_gp_completed.setStyleSheet("color: #27ae60;")
            lbl_gp_completed.setCursor(Qt.PointingHandCursor)
            gp_stats_layout.addWidget(lbl_gp_completed)

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
            
            lbl_gp_remaining = QLabel(f"⬜ Not Translated: {n_remaining}{' | ' if n_not_refined else ''}")
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
            
            gp_stats_layout.addStretch()
            p_layout.addWidget(gp_stats_frame)
            _pump_loading_frame()
            
            # Chapter list
            gp_listbox = QListWidget()
            self._apply_compact_inline_list_style(gp_listbox, QFont('Courier', 10))
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

            def _refresh_refinement_rows(_d, keep_updates_disabled=False):
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

                    for ref_key, ref_display, ref_status in _gp_refinement_rows(_d):
                        item = QListWidgetItem(ref_display)
                        item.setForeground(QColor(_gp_color_for(ref_status)))
                        item.setData(Qt.UserRole, ref_status)
                        item.setData(Qt.UserRole + 1, None)
                        item.setData(Qt.UserRole + 3, ref_key)
                        self._add_compact_inline_list_item(gp_listbox, item)
                        if ref_key in selected_ref_keys:
                            item.setSelected(True)
                finally:
                    if not keep_updates_disabled:
                        gp_listbox.setUpdatesEnabled(True)
                        gp_listbox.viewport().update()
            
            def _populate_gp_listbox(_d, chunk_size=150):
                panel_state['populate_generation'] = panel_state.get('populate_generation', 0) + 1
                generation = panel_state['populate_generation']
                cache = _gp_status_cache(_d)
                gp_listbox.clear()
                gp_listbox.setUpdatesEnabled(False)
                total = panel_state['total']
                chapter_map = panel_state['chapter_map']
                state = {'ci': 0}

                def _add_chunk():
                    if generation != panel_state.get('populate_generation'):
                        return
                    start_ci = state['ci']
                    end_ci = min(start_ci + chunk_size, total)
                    for ci in range(start_ci, end_ci):
                        fname = chapter_map.get(ci, f'chapter {ci + 1}')
                        display, status = _gp_display_for(ci, fname, _d, cache)
                        item = QListWidgetItem(display)
                        item.setForeground(QColor(_gp_color_for(status)))
                        item.setData(Qt.UserRole, status)
                        item.setData(Qt.UserRole + 1, ci)
                        self._add_compact_inline_list_item(gp_listbox, item)
                        if panel_state.get('select_all_visible'):
                            item.setSelected(not item.isHidden())
                    state['ci'] = end_ci
                    if end_ci < total:
                        QTimer.singleShot(0, _add_chunk)
                    else:
                        _refresh_refinement_rows(_d, keep_updates_disabled=True)
                        gp_listbox.setUpdatesEnabled(True)
                        gp_listbox.viewport().update()

                QTimer.singleShot(0, _add_chunk)

            _populate_gp_listbox(gp_data)

            # Helper to refresh stats labels from a loaded progress dict
            def _refresh_stats_from_dict(_d):
                _comp2, _fail2, _merg2 = _gp_sets(_d)
                _prog2 = _gp_in_progress_set(_d, _precomputed_sets=(_comp2, _fail2, _merg2))
                _total = panel_state['total']
                _not_refined2 = _gp_refinement_status_counts(_d).get('not_refined', 0)
                lbl_total.setText(f"Total: {_total} | ")
                lbl_gp_completed.setText(f"✅ Completed: {len(_comp2 - _merg2)} | ")
                lbl_gp_in_progress.setText(f"🔄 In Progress: {len(_prog2)} | ")
                lbl_gp_in_progress.setVisible(True)
                lbl_gp_failed.setText(f"❌ Failed: {len(_fail2)} | ")
                lbl_gp_merged.setText(f"🔗 Merged: {len(_merg2)} | ")
                lbl_gp_merged.setVisible(len(_merg2) > 0)
                lbl_gp_remaining.setText(f"⬜ Not Translated: {max(0, _total - len(_comp2 | _fail2 | _merg2 | _prog2))}{' | ' if _not_refined2 else ''}")
                lbl_gp_not_refined.setText(f"✨ Not Refined: {_not_refined2}")
                lbl_gp_not_refined.setVisible(_not_refined2 > 0)
            
            # Right-click context menu to delete entries from progress
            def _gp_context_menu(pos):
                # Gather selected items that are deletable
                clicked_item = gp_listbox.itemAt(pos)
                if clicked_item is not None and not clicked_item.isSelected():
                    gp_listbox.clearSelection()
                    clicked_item.setSelected(True)
                selected = gp_listbox.selectedItems()
                deletable_statuses = ('completed', 'merged', 'in_progress', 'failed', 'qa_failed', 'not_refined')
                targets = []
                for it in selected:
                    if it.data(Qt.UserRole) not in deletable_statuses:
                        continue
                    refinement_key = it.data(Qt.UserRole + 3)
                    chapter_index = it.data(Qt.UserRole + 1)
                    if refinement_key:
                        targets.append((it, ('refinement', refinement_key)))
                    elif chapter_index is not None:
                        targets.append((it, ('chapter', chapter_index)))
                if not targets:
                    return
                
                from PySide6.QtWidgets import QMenu
                menu = QMenu(gp_listbox)
                menu.setStyleSheet(
                    "QMenu { background-color: #2d2d2d; color: white; border: 1px solid #555; }"
                    "QMenu::item:selected { background-color: #c0392b; }"
                )
                
                n = len(targets)
                if n == 1:
                    target_kind, target_value = targets[0][1]
                    status = targets[0][0].data(Qt.UserRole)
                    display_text = targets[0][0].text() or ""
                    if target_kind == 'refinement':
                        chapter_label = display_text.split('|')[-1].strip() or str(target_value)
                    else:
                        label_match = re.search(r'\bCh\.\d+(?:\.\d+)?\b', display_text)
                        chapter_label = label_match.group(0) if label_match else f"Ch.{target_value+1}"
                    action = menu.addAction(f"🗑️ Remove {chapter_label} from progress ({status})")
                else:
                    action = menu.addAction(f"🗑️ Remove {n} chapters from progress")
                
                chosen = menu.exec(gp_listbox.viewport().mapToGlobal(pos))
                if chosen != action:
                    return
                
                # Remove from progress JSON
                try:
                    _rp = _find_gp_for_file(fp)
                    if not _rp or not os.path.isfile(_rp):
                        return
                    _d = _gp_load_progress_dict(_rp)
                    
                    indices_to_remove = set(value for _, (kind, value) in targets if kind == 'chapter')
                    refinement_keys_to_remove = set(value for _, (kind, value) in targets if kind == 'refinement')
                    changed = False
                    if indices_to_remove:
                        removed_indices = _d.get('manual_removed_indices', [])
                        if not isinstance(removed_indices, list):
                            removed_indices = []
                        removed_set = set()
                        for value in removed_indices:
                            mapped_ci = _gp_index_for_progress_value(value, _d)
                            if mapped_ci is not None:
                                removed_set.add(mapped_ci)
                        removed_set.update(indices_to_remove)
                        _d['manual_removed_indices'] = sorted(removed_set)
                        _d['manual_removed_session_id'] = _d.get('progress_session_id')
                        changed = True

                    for key in ('completed', 'failed', 'merged_indices', 'in_progress'):
                        lst = _d.get(key, [])
                        new_lst = []
                        for v in lst:
                            mapped_ci = _gp_index_for_progress_value(v, _d)
                            if not indices_to_remove or mapped_ci not in indices_to_remove:
                                new_lst.append(v)
                        if len(new_lst) != len(lst):
                            _d[key] = new_lst
                            changed = True

                    qa_map = _d.get('qa_issues_found', {})
                    if indices_to_remove and isinstance(qa_map, dict):
                        new_qa_map = {}
                        for k, v in qa_map.items():
                            mapped_ci = _gp_index_for_progress_value(k, _d)
                            keep = mapped_ci not in indices_to_remove if mapped_ci is not None else True
                            if keep:
                                new_qa_map[k] = v
                        if len(new_qa_map) != len(qa_map):
                            _d['qa_issues_found'] = new_qa_map
                            changed = True

                    chapters = _d.get('chapters', {})
                    if indices_to_remove and isinstance(chapters, dict):
                        new_chapters = {}
                        chapters_changed = False
                        for k, v in chapters.items():
                            ci = _gp_index_for_entry(v, k, _d) if isinstance(v, dict) else None
                            keep = ci not in indices_to_remove if ci is not None else True
                            if keep:
                                new_chapters[k] = v
                            else:
                                chapters_changed = True
                                changed = True
                        if chapters_changed or len(new_chapters) != len(chapters):
                            _d['chapters'] = new_chapters
                            changed = True

                    if refinement_keys_to_remove and isinstance(_d.get('refinement'), dict):
                        for ref_key in refinement_keys_to_remove:
                            if ref_key in _d['refinement']:
                                del _d['refinement'][ref_key]
                                changed = True
                    
                    if changed:
                        with open(_rp, 'w', encoding='utf-8') as _f:
                            json.dump(_d, _f, ensure_ascii=False, indent=2)
                        # Update all affected items
                        _cmap = panel_state['chapter_map']
                        for it, (kind, value) in targets:
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
            lbl_gp_in_progress.mousePressEvent = _gp_make_cycle(('in_progress',), gp_listbox)
            lbl_gp_failed.mousePressEvent = _gp_make_cycle(('failed', 'qa_failed'), gp_listbox)
            lbl_gp_merged.mousePressEvent = _gp_make_cycle(('merged',), gp_listbox)
            lbl_gp_remaining.mousePressEvent = _gp_make_cycle(('not_completed', 'not_translated', 'no_tts'), gp_listbox)
            lbl_gp_not_refined.mousePressEvent = _gp_make_cycle(('not_refined',), gp_listbox)
            
            p_layout.addWidget(gp_listbox)
            
            # Progress file path + open folder button + open glossary button
            path_row = QHBoxLayout()
            path_label = QLabel(f"📁 {gp_path}")
            path_label.setStyleSheet("color: #666; font-size: 8pt;")
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
            
            def _find_glossary_file(_gp_dir=_gp_folder, _epub_path=fp):
                """Find the glossary file (csv/json/txt) in the same directory as the progress file."""
                import glob
                base = os.path.splitext(os.path.basename(_epub_path))[0]
                # Search priority: book-specific glossary > generic glossary
                for ext in ['.csv', '.json', '.txt', '.md']:
                    for pattern in [
                        os.path.join(_gp_dir, f"{base}_glossary{ext}"),
                        os.path.join(_gp_dir, f"{base}{ext}"),
                        os.path.join(_gp_dir, f"glossary{ext}"),
                    ]:
                        if os.path.isfile(pattern):
                            return pattern
                # Also check parent dir (if progress is in Glossary/ subfolder)
                parent = os.path.dirname(_gp_dir)
                if os.path.basename(_gp_dir).lower() == 'glossary':
                    for ext in ['.csv', '.json', '.txt', '.md']:
                        for pattern in [
                            os.path.join(parent, f"glossary{ext}"),
                            os.path.join(parent, f"{base}_glossary{ext}"),
                        ]:
                            if os.path.isfile(pattern):
                                return pattern
                return None
            
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
                _cmap = panel_state['chapter_map']
                _total = 0
                _comp, _fail, _merg = _gp_sets(_d)
                
                gp_listbox.clear()
                for ci in range(_total):
                    fname = _cmap.get(ci, f'chapter {ci + 1}')
                    ch_num = _gp_display_chapter_num(ci, fname)
                    
                    if ci in _merg:
                        icon, status, color = '🔗', 'merged', '#17a2b8'
                    elif ci in _comp:
                        icon, status, color = '✅', 'completed', '#27ae60'
                    elif ci in _fail:
                        icon, status, color = '❌', 'failed', '#e74c3c'
                    else:
                        icon, status, color = '⬜', 'not_completed', '#5a9fd4'
                    
                    display, status = _gp_display_for(ci, fname, _d)
                    color = _gp_color_for(status)
                    item = QListWidgetItem(display)
                    item.setForeground(QColor(color))
                    item.setData(Qt.UserRole, status)
                    item.setData(Qt.UserRole + 1, ci)
                    self._add_compact_inline_list_item(gp_listbox, item)
            
                _populate_gp_listbox(_d)

            # Refresh function (called by timer)
            def _refresh():
                try:
                    _rp = _find_gp_for_file(fp)
                    if not _rp or not os.path.isfile(_rp):
                        return
                    
                    # Dirty-check: skip full recomputation if file hasn't changed
                    # and the special-files toggle is the same
                    try:
                        _cur_mtime = os.path.getmtime(_rp)
                    except OSError:
                        _cur_mtime = 0
                    _cur_ts = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
                    if (_cur_mtime == panel_state.get('_last_mtime', -1)
                            and _cur_ts == panel_state.get('translate_special')):
                        return  # Nothing changed — skip
                    panel_state['_last_mtime'] = _cur_mtime
                    
                    _d = _gp_load_progress_dict(_rp)
                    
                    # Check if TRANSLATE_SPECIAL_FILES toggle changed — rebuild chapter map if so
                    if _cur_ts != panel_state['translate_special']:
                        panel_state['translate_special'] = _cur_ts
                        new_cmap, new_total, new_spine_idx = _read_spine_map(fp, _cur_ts)
                        if new_total > 0:
                            panel_state['chapter_map'] = new_cmap
                            panel_state['spine_index_map'] = new_spine_idx
                            panel_state['total'] = new_total
                        elif new_total == 0:
                            _comp0, _fail0, _merg0 = _gp_sets(_d)
                            _all_idx = _comp0 | _fail0 | _merg0
                            panel_state['total'] = (max(_all_idx, default=0) + 1) if _all_idx else 1
                        # Rebuild reverse lookups after chapter_map change
                        _rebuild_reverse_lookups()
                        _rebuild_listbox(_d)
                        _refresh_stats_from_dict(_d)
                        return
                    
                    _comp, _fail, _merg = _gp_sets(_d)
                    _prog = _gp_in_progress_set(_d, _precomputed_sets=(_comp, _fail, _merg))
                    _not_refined = _gp_refinement_status_counts(_d).get('not_refined', 0)
                    _total = panel_state['total']
                    _cmap = panel_state['chapter_map']
                    
                    _nr = max(0, _total - len(_comp | _fail | _merg | _prog))
                    lbl_total.setText(f"Total: {_total} | ")
                    lbl_gp_completed.setText(f"✅ Completed: {len(_comp - _merg)} | ")
                    lbl_gp_in_progress.setText(f"🔄 In Progress: {len(_prog)} | ")
                    lbl_gp_in_progress.setVisible(True)
                    lbl_gp_failed.setText(f"❌ Failed: {len(_fail)} | ")
                    lbl_gp_merged.setText(f"🔗 Merged: {len(_merg)} | ")
                    lbl_gp_merged.setVisible(len(_merg) > 0)
                    lbl_gp_remaining.setText(f"⬜ Not Translated: {_nr}{' | ' if _not_refined else ''}")
                    lbl_gp_not_refined.setText(f"✨ Not Refined: {_not_refined}")
                    lbl_gp_not_refined.setVisible(_not_refined > 0)
                    
                    _bt = _d.get('book_title', '')
                    if _bt and bt_label:
                        bt_label.setText(f"📖 {_bt}")
                    
                    _cache = _gp_status_cache(_d)
                    for ci in range(min(gp_listbox.count(), _total)):
                        item = gp_listbox.item(ci)
                        if not item:
                            continue
                        if item.data(Qt.UserRole + 3):
                            continue
                        old_status = item.data(Qt.UserRole)
                        new_status, _issues = _gp_status_for(ci, _d, _cache)
                        new_color = _gp_color_for(new_status)
                        
                        if new_status != old_status or _issues:
                            fname = _cmap.get(ci, f'chapter {ci + 1}')
                            ch_num2 = _gp_display_chapter_num(ci, fname)
                            _icons = {'completed': '✅', 'failed': '❌', 'merged': '🔗', 'not_completed': '⬜'}
                            display2 = f"Ch.{ch_num2:03d} | {_icons.get(new_status, '⬜')} {new_status.replace('_', ' ').title():14s} | {fname}"
                            display2, _ = _gp_display_for(ci, fname, _d, _cache)
                            item.setText(display2)
                            item.setForeground(QColor(new_color))
                            item.setData(Qt.UserRole, new_status)
                    _refresh_refinement_rows(_d)
                except Exception:
                    pass
            
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

                        def _refresh_cached_gp_panels():
                            for rfn in getattr(dialog, '_gp_refresh_funcs', []):
                                try:
                                    rfn()
                                except Exception:
                                    pass

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
                    width_ratio=0.35,
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

                def _pump_gp_loading():
                    try:
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
                    
                    expected_refinement = _glossary_refinement_expected_entries()
                    empty_label = QLabel(f"No glossary extraction progress found for:\n{epub_base}")
                    empty_label.setAlignment(Qt.AlignCenter)
                    empty_label.setStyleSheet("color: #7a8a9e; font-size: 11pt;")
                    empty_label.setWordWrap(True)
                    p_layout.addWidget(empty_label)
                    
                    hint_text = "Run glossary extraction to see progress here."
                    if expected_refinement:
                        hint_text = "Glossary refinement is enabled; expected refinement entry types are listed below."
                    hint_label = QLabel(hint_text)
                    hint_label.setAlignment(Qt.AlignCenter)
                    hint_label.setStyleSheet("color: #555; font-size: 9pt; font-style: italic;")
                    p_layout.addWidget(hint_label)

                    if expected_refinement:
                        ref_list = QListWidget(panel)
                        ref_list.setSelectionMode(QAbstractItemView.NoSelection)
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
                        for _ref_key, ref_info in sorted(expected_refinement.items()):
                            entry_type = str(ref_info.get('entry_type') or _ref_key.replace('type::', '')).strip() or 'entry type'
                            item = QListWidgetItem(f"Refinement | \u2728 Not Refined    | {entry_type}")
                            item.setForeground(QColor('#5a9fd4'))
                            self._add_compact_inline_list_item(ref_list, item)
                        p_layout.addWidget(ref_list)
                    
                    p_layout.addStretch()
                    
                    # State container for upgrade tracking
                    _state = {'upgraded': False}
                    
                    def _empty_refresh():
                        """Check if a progress file appeared and upgrade the panel in-place."""
                        if _state['upgraded']:
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
                    
                    return panel, _empty_refresh
                
                if n_files == 1:
                    # Single file — no tabs needed
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
                
                else:
                    # Dropdown navigation for >3 files
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
                
                # Close button
                close_btn = QPushButton("Close")
                close_btn.setStyleSheet(
                    "QPushButton { background-color: #555; color: white; padding: 6px 20px; "
                    "border-radius: 4px; font-size: 10pt; } "
                    "QPushButton:hover { background-color: #666; }"
                )
                close_btn.clicked.connect(gp_dialog.hide)
                gp_content_layout.addWidget(close_btn, alignment=Qt.AlignCenter)

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
                
                # Auto-refresh timer (2s) — calls all panel refresh functions
                def _gp_refresh_all():
                    try:
                        if not gp_dialog.isVisible():
                            return
                        for rfn in all_refresh_funcs:
                            try:
                                rfn()
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                _gp_timer = QTimer(gp_dialog)
                _gp_timer.setInterval(2000)
                _gp_timer.timeout.connect(_gp_refresh_all)
                _gp_timer.start()
                
                # Cache on parent dialog so all tabs share the same instance
                dialog._glossary_progress_dialog = gp_dialog
                dialog._gp_refresh_funcs = all_refresh_funcs
                
                gp_dialog.show()
            
            except Exception as e:
                print(f"⚠️ Error showing glossary progress: {e}")
                import traceback
                traceback.print_exc()
        
        glossary_progress_btn.clicked.connect(_show_glossary_progress)
        title_layout.addWidget(glossary_progress_btn)
        
        # Periodic check: show/hide button based on file existence (3s)
        # Uses single-pass caching to avoid redundant filesystem scans
        def _check_glossary_btn_visibility():
            try:
                # Skip if parent dialog is not visible (no point scanning filesystem)
                if hasattr(dialog, 'isVisible') and not dialog.isVisible():
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
                refinement_expected = _glossary_refinement_settings_enabled()
                glossary_progress_btn.setVisible(True)
                if any_exists:
                    count = len(found_paths)
                    if count == 1:
                        gp = next(iter(found_paths.values()))
                        glossary_progress_btn.setToolTip(f"View glossary extraction progress\n{gp}")
                    else:
                        glossary_progress_btn.setToolTip(f"View glossary extraction progress ({count}/{len(all_epubs)} files)")
                elif refinement_expected:
                    glossary_progress_btn.setToolTip("View glossary extraction and refinement progress")
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
                            _info = item_data.get('info') or {}
                            _fname = _info.get('original_filename', '') or _info.get('output_file', '') or _info.get('key', '')
                            is_skipped_special = self._progress_file_is_skipped_special(
                                _fname,
                                item_data.get('is_special', False),
                            )
                            # Show all items if toggle is on, hide only files that translation skips.
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
        
        # Calculate stats from the appropriate source
        _stats_data = {'prog': prog}
        if spine_chapters:
            total_chapters = len(spine_chapters)
            completed = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'completed')
            merged = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'merged')
            in_progress = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'in_progress')
            pending = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) == 'pending')
            missing = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) in ['not_translated', 'not_refined', 'no_tts'])
            failed = sum(1 for ch in spine_chapters if self._progress_display_status(ch, _stats_data) in ['failed', 'qa_failed'])
        else:
            # For non-OPF files, calculate from chapter_display_info
            total_chapters = len(chapter_display_info)
            completed = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'completed')
            merged = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'merged')
            in_progress = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'in_progress')
            pending = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) == 'pending')
            missing = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) in ['not_translated', 'not_refined', 'no_tts'])
            failed = sum(1 for ch in chapter_display_info if self._progress_display_status(ch, _stats_data) in ['failed', 'qa_failed'])
        
        # Create labels (outside the if/else so they always appear)
        stats_font = QFont('Arial', 10)
        
        lbl_total = QLabel(f"Total: {total_chapters} | ")
        lbl_total.setFont(stats_font)
        stats_layout.addWidget(lbl_total)
        
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
        _missing_label_text = "✨ Not Refined" if _current_output_mode == 'refinement' else ("🔊 No TTS" if _current_output_mode == 'audio' else "⬜ Not Translated")
        lbl_missing = QLabel(f"{_missing_label_text}: {missing} | ")
        lbl_missing.setFont(stats_font)
        lbl_missing.setStyleSheet("color: #2b6cb0;")
        lbl_missing.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_missing)
        
        # Match list status: failed/qa_failed use ❌ and red (clickable — jumps to next failure)
        lbl_failed = QLabel(f"❌ Failed: {failed} | ")
        lbl_failed.setFont(stats_font)
        lbl_failed.setStyleSheet("color: red;")
        lbl_failed.setCursor(Qt.PointingHandCursor)
        stats_layout.addWidget(lbl_failed)
        
        
        stats_layout.addStretch()
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
        lbl_failed.mousePressEvent      = _make_cycle_handler(('failed', 'qa_failed'))
        
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
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': chapter_display_info,
            'listbox': listbox,
            'selection_count_label': selection_count_label,
            'dialog': dialog,
            'container': container,
            'show_special_files_state': show_special_files[0],  # Store current toggle state
            'show_special_files_cb': show_special_files_cb,  # Store checkbox reference
            'show_model_info_state': show_model_info[0],
            'show_model_info_cb': show_model_info_cb
        }
        show_model_info_cb._progress_data_ref = result
        
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
            data['listbox'].selectAll()
            data['selection_count_label'].setText(f"Selected: {data['listbox'].count()}")
        
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
                    matched = display_status in ['failed', 'qa_failed']
                elif status_to_select == 'qa_failed':
                    matched = display_status == 'qa_failed'
                else:
                    matched = display_status == status_to_select
                if matched:
                    item.setSelected(True)
            count = len(data['listbox'].selectedItems())
            data['selection_count_label'].setText(f"Selected: {count}")

        def _normalize_filename(name: str) -> str:
            if not name:
                return ""
            base = os.path.basename(name)
            if base.startswith("response_"):
                base = base[len("response_"):]
            while True:
                new_base, ext = os.path.splitext(base)
                if not ext:
                    break
                base = new_base
            return base

        def _find_progress_entry(chapter_info, prog):
            """Strict: match only identical output_file string."""
            target_out = chapter_info.get('output_file')
            if not target_out:
                return None
            for key, ch in prog.get("chapters", {}).items():
                if ch.get('output_file') == target_out:
                    return key, ch
            return None

        def _sdlxliff_sidecar_path_for_output_file(output_file):
            if not output_file:
                return None
            output_name = os.path.basename(str(output_file).replace("\\", "/"))
            if not output_name:
                return None
            return os.path.join(data['output_dir'], "SDLXLIFF", f"{output_name}.sdlxliff")

        def _machine_translation_path_for_output_file(output_file):
            return _sdlxliff_machine_translation_path(data['output_dir'], output_file)

        def _clear_refinement_progress_fields(entry):
            """Remove stale refinement metadata when a chapter is queued again."""
            if not isinstance(entry, dict):
                return 0
            removed = 0
            for field in ("refinement_status", "refined_at", "refinement_error", "unrefined_backup_file"):
                if field in entry:
                    entry.pop(field, None)
                    removed += 1
            previous_entry = entry.get("previous_progress_entry")
            if isinstance(previous_entry, dict):
                for field in ("refinement_status", "refined_at", "refinement_error", "unrefined_backup_file"):
                    previous_entry.pop(field, None)
            return removed

        def _restore_regular_in_progress_entry(info):
            if not isinstance(info, dict):
                return None
            previous_status = str(info.get('previous_status', '') or '').lower()
            previous_entry = info.get('previous_progress_entry')
            transient_statuses = {'in_progress', 'not_translated', 'not translated', 'not_completed'}
            if isinstance(previous_entry, dict):
                restored = dict(previous_entry)
                restored_status = str(restored.get('status', previous_status) or previous_status).lower()
                if restored_status and restored_status not in transient_statuses:
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    return restored
            if previous_status in ('qa_failed', 'failed', 'error', 'pending', 'merged', 'completed'):
                restored = dict(info)
                restored['status'] = 'failed' if previous_status == 'error' else previous_status
                restored.pop('previous_status', None)
                restored.pop('previous_progress_entry', None)
                restored.pop('previous_status_unknown', None)
                return restored
            if info.get('previous_status_unknown'):
                restored = dict(info)
                restored['status'] = 'failed'
                restored.pop('previous_status', None)
                restored.pop('previous_progress_entry', None)
                restored.pop('previous_status_unknown', None)
                return restored
            output_file = info.get('output_file')
            output_exists = bool(output_file and os.path.exists(os.path.join(data['output_dir'], output_file)))
            if previous_status in ('not_translated', 'not translated', 'not_completed', ''):
                if previous_status and not output_exists:
                    return None
                if output_exists:
                    restored = dict(info)
                    restored['status'] = 'failed'
                    restored.pop('previous_status', None)
                    restored.pop('previous_progress_entry', None)
                    restored.pop('previous_status_unknown', None)
                    return restored
            return None

        def restore_in_progress_marks():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]
            in_progress_chapters = [ch for ch in selected_chapters if ch.get('status') == 'in_progress']

            if not in_progress_chapters:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No In Progress Chapters",
                                     "None of the selected chapters have 'in_progress' status.")
                return

            restored_count = 0
            deleted_count = 0
            failed_count = 0
            progress_updated = False

            for info in in_progress_chapters:
                match = None
                progress_key = info.get('progress_key')
                if progress_key and progress_key in data['prog'].get("chapters", {}):
                    match = (progress_key, data['prog']["chapters"][progress_key])
                else:
                    match = _find_progress_entry(info, data['prog'])

                if not match:
                    print(f"WARNING: Could not find in-progress entry for {info.get('num')} ({info.get('output_file')})")
                    continue

                key, entry = match
                restored = _restore_regular_in_progress_entry(entry)
                if restored:
                    data['prog']["chapters"][key] = restored
                    progress_updated = True
                    if restored.get('status') == 'failed':
                        failed_count += 1
                    else:
                        restored_count += 1
                else:
                    del data['prog']["chapters"][key]
                    progress_updated = True
                    deleted_count += 1

            if progress_updated:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
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
        
        def remove_qa_failed_mark():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            # Skip dedup here to avoid merging distinct chapters that share filenames
            

            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]
            failed_chapters = [ch for ch in selected_chapters if ch['status'] in ['qa_failed', 'failed']]
            
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
            cleared_count = 0
            progress_updated = False
            for info in failed_chapters:
                match = None
                progress_key = info.get('progress_key')
                if progress_key and progress_key in data['prog'].get("chapters", {}):
                    match = (progress_key, data['prog']["chapters"][progress_key])
                else:
                    match = _find_progress_entry(info, data['prog'])

                # Normalize target output for multi-entry cleanup
                target_out = info.get('output_file')
                target_norm = _normalize_filename(target_out)
                if match:
                    # Clear failed/qa_failed on ALL entries sharing this output file (normalized)
                    fields_to_remove = ["qa_issues", "qa_timestamp", "qa_issues_found", "duplicate_confidence", "failure_reason", "error_message"]
                    for key, entry in data['prog'].get("chapters", {}).items():
                        entry_out = entry.get('output_file')
                        if not entry_out:
                            continue
                        if _normalize_filename(entry_out) == target_norm:
                            if entry.get('status') in ['qa_failed', 'failed']:
                                entry["status"] = "completed"
                                for field in fields_to_remove:
                                    entry.pop(field, None)
                                cleared_count += 1
                                progress_updated = True
                else:
                    print(f"WARNING: Could not find chapter entry for {info.get('num')} ({info.get('output_file')})")
            
            # Save the updated progress
            if progress_updated:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            
            # Auto-refresh the display
            self._refresh_retranslation_data(data)
            
            self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Success", f"Removed failed mark from {cleared_count} chapters.")
        
        def retranslate_selected():
            selected_items = data['listbox'].selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "No Selection", "Please select at least one chapter.")
                return

            # Do NOT dedup here; it can collapse distinct chapters sharing filenames
            
            selected_indices = [data['listbox'].row(item) for item in selected_items]
            selected_chapters = [data['chapter_display_info'][i] for i in selected_indices]

            if self._current_progress_output_mode(data) == 'audio':
                count = len(selected_chapters)
                reply = self._styled_msgbox(
                    QMessageBox.Question,
                    data.get('dialog', self),
                    "Confirm TTS Reset",
                    f"This will delete only generated TTS audio for {count} selected chapter(s), mark them as No TTS, and leave translated HTML files untouched.\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

                deleted_count = 0
                status_reset_count = 0
                missing_audio_count = 0
                progress_updated = False

                def _audio_candidates(ch_entry, ch_info):
                    candidates = []
                    stored_tts_file = ch_entry.get('tts_file') if isinstance(ch_entry, dict) else None
                    if stored_tts_file:
                        candidates.append(stored_tts_file)

                    output_file = (ch_entry or {}).get('output_file') or ch_info.get('output_file')
                    if output_file:
                        configured_ext = str(os.environ.get('TTS_AUDIO_FORMAT') or 'mp3').lower().strip().lstrip('.')
                        for stem in self._audio_stem_variants(output_file):
                            for ext in [configured_ext, 'mp3', 'wav']:
                                if ext:
                                    candidates.append(os.path.join('text_to_speech', f"{stem}.{ext}"))

                    seen = set()
                    paths = []
                    for candidate in candidates:
                        normalized = str(candidate).replace('\\', '/')
                        if normalized in seen:
                            continue
                        seen.add(normalized)
                        full_path = normalized if os.path.isabs(normalized) else os.path.join(data['output_dir'], normalized)
                        paths.append(full_path)
                    return paths

                for ch_info in selected_chapters:
                    match = None
                    progress_key = ch_info.get('progress_key')
                    if progress_key and progress_key in data['prog'].get("chapters", {}):
                        match = (progress_key, data['prog']["chapters"][progress_key])
                    else:
                        match = _find_progress_entry(ch_info, data['prog'])

                    ch_entry = match[1] if match else {}
                    deleted_for_chapter = False
                    for audio_path in _audio_candidates(ch_entry, ch_info):
                        try:
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                                deleted_count += 1
                                deleted_for_chapter = True
                                print(f"Deleted TTS audio: {audio_path}")
                        except Exception as e:
                            print(f"Failed to delete TTS audio {audio_path}: {e}")

                    if not deleted_for_chapter:
                        missing_audio_count += 1

                    if match:
                        chapter_key, ch_entry = match
                        ch_entry["tts_status"] = "no_tts"
                        ch_entry.pop("tts_file", None)
                        ch_entry.pop("tts_at", None)
                        ch_entry.pop("tts_error", None)
                        ch_entry["last_updated"] = time.time()
                        progress_updated = True
                        status_reset_count += 1
                        print(f"Reset TTS status to no_tts for chapter {ch_info.get('num')} (key: {chapter_key})")
                    else:
                        print(f"WARNING: Could not find exact progress entry for {ch_info.get('output_file')}; skipped TTS status reset")

                if progress_updated:
                    try:
                        with open(data['progress_file'], 'w', encoding='utf-8') as f:
                            json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                        print(f"Updated progress tracking file - reset {status_reset_count} TTS statuses to no_tts")
                    except Exception as e:
                        print(f"Failed to update progress file: {e}")

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
            
            # Count different types
            missing_count = sum(1 for ch in selected_chapters if ch['status'] == 'not_translated')
            existing_count = sum(1 for ch in selected_chapters if ch['status'] != 'not_translated')
            
            count = len(selected_chapters)
            if count > 10:
                if missing_count > 0 and existing_count > 0:
                    confirm_msg = f"This will:\n• Mark {missing_count} missing chapters for translation\n• Delete and retranslate {existing_count} existing chapters and their SDLXLIFF sidecars\n\nTotal: {count} chapters\n\nContinue?"
                elif missing_count > 0:
                    confirm_msg = f"This will mark {missing_count} missing chapters for translation.\n\nContinue?"
                else:
                    confirm_msg = f"This will delete {existing_count} translated chapters and their SDLXLIFF sidecars, then mark them for retranslation.\n\nContinue?"
            else:
                chapters = [f"Ch.{ch['num']}" for ch in selected_chapters]
                confirm_msg = f"This will process:\n\n{', '.join(chapters)}\n\n"
                if missing_count > 0:
                    confirm_msg += f"• {missing_count} missing chapters will be marked for translation\n"
                if existing_count > 0:
                    confirm_msg += f"• {existing_count} existing chapters and SDLXLIFF sidecars will be deleted and retranslated\n"
                confirm_msg += "\nContinue?"
            
            reply = self._styled_msgbox(QMessageBox.Question, data.get('dialog', self), "Confirm Retranslation", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Process chapters - DELETE FILES AND UPDATE PROGRESS
            deleted_count = 0
            marked_count = 0
            status_reset_count = 0
            refinement_cleared_count = 0
            merged_cleared_count = 0
            sidecar_deleted_count = 0
            sidecar_failed_count = 0
            machine_translation_deleted_count = 0
            machine_translation_failed_count = 0
            progress_updated = False

            for ch_info in selected_chapters:
                output_file = ch_info['output_file']
                actual_num = ch_info['num']
                progress_key = ch_info.get('progress_key')
                
                if ch_info['status'] != 'not_translated':
                    # Reset status to pending for ALL non-not_translated chapters, but only if we can match the exact progress entry
                    match = None
                    if progress_key and progress_key in data['prog']["chapters"]:
                        match = (progress_key, data['prog']["chapters"][progress_key])
                    else:
                        match = _find_progress_entry(ch_info, data['prog'])
                    old_status = ch_info['status']
                    
                    if match:
                        chapter_key, ch_entry = match
                        target_output_file = ch_entry.get('output_file') or ch_info['output_file']
                        # Delete existing file only after we know which entry to update
                        if output_file:
                            output_path = os.path.join(data['output_dir'], output_file)
                            try:
                                if os.path.exists(output_path):
                                    os.remove(output_path)
                                    deleted_count += 1
                                    print(f"Deleted: {output_path}")
                            except Exception as e:
                                print(f"Failed to delete {output_path}: {e}")

                        sidecar_paths = []
                        seen_sidecars = set()
                        machine_translation_paths = []
                        seen_machine_translation = set()
                        for candidate_output in (output_file, target_output_file):
                            sidecar_path = _sdlxliff_sidecar_path_for_output_file(candidate_output)
                            if not sidecar_path:
                                continue
                            sidecar_key = os.path.normcase(os.path.abspath(sidecar_path))
                            if sidecar_key in seen_sidecars:
                                continue
                            seen_sidecars.add(sidecar_key)
                            sidecar_paths.append(sidecar_path)
                            machine_translation_path = _machine_translation_path_for_output_file(candidate_output)
                            if machine_translation_path:
                                machine_translation_key = os.path.normcase(os.path.abspath(machine_translation_path))
                                if machine_translation_key not in seen_machine_translation:
                                    seen_machine_translation.add(machine_translation_key)
                                    machine_translation_paths.append(machine_translation_path)
                        for sidecar_path in sidecar_paths:
                            try:
                                if os.path.exists(sidecar_path):
                                    os.remove(sidecar_path)
                                    sidecar_deleted_count += 1
                                    print(f"Deleted SDLXLIFF sidecar: {sidecar_path}")
                            except Exception as e:
                                sidecar_failed_count += 1
                                print(f"Failed to delete SDLXLIFF sidecar {sidecar_path}: {e}")
                        for machine_translation_path in machine_translation_paths:
                            try:
                                if os.path.exists(machine_translation_path):
                                    os.remove(machine_translation_path)
                                    machine_translation_deleted_count += 1
                                    print(f"Deleted Machine Translation preview: {machine_translation_path}")
                            except Exception as e:
                                machine_translation_failed_count += 1
                                print(f"Failed to delete Machine Translation preview {machine_translation_path}: {e}")

                        print(f"Resetting {old_status} status to pending for chapter {actual_num} (key: {chapter_key}, output file: {target_output_file})")
                        ch_entry["status"] = "pending"
                        ch_entry["failure_reason"] = ""
                        ch_entry["error_message"] = ""
                        if _clear_refinement_progress_fields(ch_entry):
                            refinement_cleared_count += 1
                        progress_updated = True
                        status_reset_count += 1
                    else:
                        print(f"WARNING: Could not find exact progress entry for {output_file}; skipped deletion and status reset")
                    
                    # MERGED CHILDREN FIX: Clear any merged children of this chapter
                    # ONLY clear children that still have "merged" status
                    # If split-the-merge succeeded, children will have their own status (completed/qa_failed)
                    # and should NOT be deleted when parent is retranslated
                    for child_key, child_data in list(data['prog']["chapters"].items()):
                        child_status = child_data.get("status")
                        if child_status == "merged" and child_data.get("merged_parent_chapter") == actual_num:
                            child_actual_num = child_data.get("actual_num")
                            print(f"🔓 Clearing merged status for child chapter {child_actual_num} (parent {actual_num} being retranslated)")
                            del data['prog']["chapters"][child_key]
                            merged_cleared_count += 1
                            progress_updated = True
                else:
                    # Just marking for translation (no file to delete)
                    marked_count += 1
            
            # Save the updated progress if we made changes
            if progress_updated:
                try:
                    with open(data['progress_file'], 'w', encoding='utf-8') as f:
                        json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file - reset {status_reset_count} chapter statuses to pending")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            data['skip_cleanup'] = True  # Disable cleanup for this dialog after retranslate to avoid deleting pending/failed
            self._refresh_retranslation_data(data)
            
            # Build success message
            success_parts = []
            if deleted_count > 0:
                success_parts.append(f"Deleted {deleted_count} files")
            if sidecar_deleted_count > 0:
                success_parts.append(f"deleted {sidecar_deleted_count} SDLXLIFF sidecar(s)")
            if machine_translation_deleted_count > 0:
                success_parts.append(f"deleted {machine_translation_deleted_count} Machine Translation preview file(s)")
            if marked_count > 0:
                success_parts.append(f"marked {marked_count} missing chapters for translation")
            if status_reset_count > 0:
                success_parts.append(f"reset {status_reset_count} chapter statuses to pending")
            if refinement_cleared_count > 0:
                success_parts.append(f"cleared refinement state for {refinement_cleared_count} chapter(s)")
            if merged_cleared_count > 0:
                success_parts.append(f"cleared {merged_cleared_count} merged child chapters")
            if sidecar_failed_count > 0:
                success_parts.append(f"failed to delete {sidecar_failed_count} SDLXLIFF sidecar(s)")
            if machine_translation_failed_count > 0:
                success_parts.append(f"failed to delete {machine_translation_failed_count} Machine Translation preview file(s)")
            
            if success_parts:
                success_msg = "Successfully " + ", ".join(success_parts) + "."
                if deleted_count > 0 or marked_count > 0 or merged_cleared_count > 0:
                    total_to_translate = len(selected_indices) + merged_cleared_count
                    success_msg += f"\n\nTotal {total_to_translate} chapters ready for translation."
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Success", success_msg)
            else:
                self._styled_msgbox(QMessageBox.Information, data.get('dialog', self), "Info", "No changes made.")
        
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
        btn_retranslate.clicked.connect(retranslate_selected)
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
                            self._show_retranslation_shell_then_build(file_path, show_special_files_state=show_special)
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

        # Auto-refresh every 3 seconds (silent, no animation)
        def _silent_refresh():
            try:
                # Skip if a manual refresh is already in progress
                if not btn_refresh.isEnabled():
                    return
                dlg = data.get('dialog')
                if dlg and dlg.isVisible():
                    self._refresh_retranslation_data(data)
            except Exception:
                pass

        _auto_refresh_timer = QTimer(data.get('dialog') or self)
        _auto_refresh_timer.setInterval(2000)
        _auto_refresh_timer.timeout.connect(_silent_refresh)
        _auto_refresh_timer.start()
        data['_auto_refresh_timer'] = _auto_refresh_timer

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

        def _open_sdlxliff_review_for_item(display_info):
            review_path = _sdlxliff_review_path_for_item(display_info)
            if not _source_exists_for_item(display_info):
                self._show_message('error', "Source Missing", "The raw source file for this entry was not found.", parent=data.get('dialog', self))
                return
            if not review_path:
                progress_entry = display_info.get('info', {}) or {}
                output_file = display_info.get('output_file') or progress_entry.get('output_file')
                self._generate_sdlxliff_sidecars_from_completed_entries(
                    data['output_dir'],
                    file_path=data.get('file_path'),
                    progress_data=data.get('prog'),
                    output_files=[output_file] if output_file else None,
                    overwrite=True,
                )
                review_path = _sdlxliff_review_path_for_item(display_info)
                if not review_path:
                    self._show_message('error', "SDLXLIFF Missing", "No matching SDLXLIFF review file could be generated for this exact output filename.", parent=data.get('dialog', self))
                    return
            try:
                self._open_or_reuse_sdlxliff_review(data['output_dir'], review_path, data.get('dialog', self))
            except Exception as e:
                self._show_message('error', "Open Failed", str(e), parent=data.get('dialog', self))

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

        def _find_audio_file_for_item(display_info):
            """Return the generated TTS file path associated with an HTML row, if one exists."""
            progress_entry = display_info.get('info', {}) or {}
            output_file = display_info.get('output_file') or progress_entry.get('output_file')
            candidates = []

            stored_tts_file = progress_entry.get('tts_file')
            if stored_tts_file:
                candidates.append(stored_tts_file)

            progress_key = display_info.get('progress_key')
            if progress_key and progress_key in data.get('prog', {}).get('chapters', {}):
                tracked_tts_file = data['prog']['chapters'][progress_key].get('tts_file')
                if tracked_tts_file:
                    candidates.append(tracked_tts_file)

            if output_file:
                for _key, tracked in data.get('prog', {}).get('chapters', {}).items():
                    if isinstance(tracked, dict) and tracked.get('output_file') == output_file and tracked.get('tts_file'):
                        candidates.append(tracked.get('tts_file'))

                for stem in self._audio_stem_variants(output_file):
                    for ext in ("wav", "mp3", "pcm", "m4a", "ogg", "flac"):
                        candidates.append(os.path.join("text_to_speech", f"{stem}.{ext}"))

            seen = set()
            for candidate in candidates:
                if not candidate:
                    continue
                normalized = str(candidate).replace("\\", "/")
                if normalized in seen:
                    continue
                seen.add(normalized)
                path = normalized if os.path.isabs(normalized) else os.path.join(data['output_dir'], normalized)
                if os.path.exists(path):
                    return path
            return None

        def _reset_tts_progress_for_output(output_file):
            if not output_file:
                return 0
            updated = 0
            now = time.time()
            for _key, tracked in data.get('prog', {}).get('chapters', {}).items():
                if not isinstance(tracked, dict):
                    continue
                if tracked.get('output_file') != output_file:
                    continue
                tracked['tts_status'] = 'no_tts'
                tracked.pop('tts_file', None)
                tracked.pop('tts_at', None)
                tracked.pop('tts_error', None)
                tracked['last_updated'] = now
                updated += 1
            if updated:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            return updated

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
                
            has_missing_images = any('missing_images' in str(issue) for issue in qa_issues)
            
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
            act_open = menu.addAction("📂 Open File")
            act_review_sdlxliff = None
            if _source_exists_for_item(display_info) and qa_file_path:
                act_review_sdlxliff = menu.addAction(" 🔍Review source -> output")
            act_open_audio = None
            act_delete_audio = None
            if _find_audio_file_for_item(display_info):
                act_open_audio = menu.addAction("🔊 Open Audio File")
                act_delete_audio = menu.addAction("🗑️ Delete Audio File")
            act_notepad_qa = None
            if qa_file_path:
                _label = "✏️ Edit File (find QA issue)" if qa_issues else "✏️ Edit File"
                act_notepad_qa = menu.addAction(_label)
            act_retranslate = menu.addAction("🔁 Retranslate Selected")
            
            act_insert_img = None
            if has_missing_images:
                act_insert_img = menu.addAction("🖼️ Insert Missing Image")
                
            act_remove_qa = menu.addAction("🧹 Remove QA Failed Mark")
            selected_infos = []
            try:
                for selected_item in listbox.selectedItems():
                    wrapper = selected_item.data(Qt.UserRole) or {}
                    selected_infos.append(wrapper.get('info', {}))
            except RuntimeError:
                selected_infos = [display_info]
            act_restore_in_progress = None
            if any((info or {}).get('status') == 'in_progress' for info in selected_infos):
                act_restore_in_progress = menu.addAction("Restore In Progress Status")
            chosen = menu.exec(listbox.mapToGlobal(pos))
            if chosen == act_open:
                _open_file_for_item(display_info)
            elif act_review_sdlxliff and chosen == act_review_sdlxliff:
                _open_sdlxliff_review_for_item(display_info)
            elif act_open_audio and chosen == act_open_audio:
                _open_audio_file_for_item(display_info)
            elif act_delete_audio and chosen == act_delete_audio:
                _delete_audio_file_for_item(display_info)
            elif chosen == act_retranslate:
                retranslate_selected()
            elif act_insert_img and chosen == act_insert_img:
                # IN-PLACE RESTORATION LOGIC using ContentProcessor
                try:
                    from bs4 import BeautifulSoup
                    import zipfile
                    from TransateKRtoEN import ContentProcessor
                    
                    # Load rename map from output directory
                    rename_map = None
                    rename_map_path = os.path.join(data['output_dir'], 'image_rename_map.json')
                    if os.path.exists(rename_map_path):
                        try:
                            with open(rename_map_path, 'r', encoding='utf-8') as f:
                                rename_map = json.load(f) or {}
                        except Exception:
                            pass

                    # 1. Get Source Content from EPUB
                    epub_path = data['file_path']
                    original_filename = display_info.get('original_filename')
                    source_html = None
                    
                    if original_filename:
                        try:
                            def normalize_name(n):
                                base = os.path.basename(n)
                                if base.startswith('response_'):
                                    base = base[9:]
                                return os.path.splitext(base)[0].lower()
                                
                            target_base = normalize_name(original_filename)
                            
                            with zipfile.ZipFile(epub_path, 'r') as zf:
                                for fname in zf.namelist():
                                    if normalize_name(fname) == target_base:
                                        source_html = zf.read(fname).decode('utf-8', errors='ignore')
                                        break
                        except Exception as ex:
                            print(f"Extraction error: {ex}")
                    
                    if not source_html:
                        self._show_message('error', "Error", "Could not extract source HTML for this chapter.")
                    else:
                        # 2. Get Translated Content
                        output_file = display_info.get('output_file')
                        output_path = os.path.join(data['output_dir'], output_file)
                        
                        if os.path.exists(output_path):
                            with open(output_path, 'r', encoding='utf-8') as f:
                                translated_html = f.read()
                                
                            # 3. Restore using ContentProcessor (supports all image formats + rename map)
                            restored_html = ContentProcessor.emergency_restore_images(
                                translated_html, source_html, verbose=True, rename_map=rename_map
                            )
                            
                            if restored_html != translated_html:
                                # 4. Save
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(restored_html)
                                    
                                # 5. Update Progress (Clear QA flags)
                                found_key = None
                                target_out = display_info.get('output_file')
                                
                                if target_out:
                                    for k, v in data['prog'].get('chapters', {}).items():
                                        if v.get('output_file') == target_out:
                                            found_key = k
                                            break
                                
                                if found_key:
                                    real_entry = data['prog']['chapters'][found_key]
                                    real_entry['status'] = 'completed'
                                    for key in ['qa_issues', 'qa_issues_found', 'qa_timestamp', 'failure_reason', 'error_message']:
                                        real_entry.pop(key, None)
                                else:
                                    progress_entry['status'] = 'completed'
                                    for key in ['qa_issues', 'qa_issues_found', 'qa_timestamp', 'failure_reason', 'error_message']:
                                        progress_entry.pop(key, None)
                                
                                # Save progress
                                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                                    json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                                    
                                # 6. Refresh
                                self._refresh_retranslation_data(data)
                                self._show_message('info', "Success", "Images restored and QA flags cleared.")
                            else:
                                self._show_message('info', "Info", "No missing images could be automatically restored.")
                        else:
                            self._show_message('error', "Error", "Output file not found.")
                            
                except Exception as e:
                    self._show_message('error', "Error", f"Failed to restore images: {e}")
                    import traceback
                    traceback.print_exc()
            elif act_restore_in_progress and chosen == act_restore_in_progress:
                restore_in_progress_marks()
            elif chosen == act_remove_qa:
                remove_qa_failed_mark()
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

        # Automatically refresh once when dialog is opened
        # Skip for multi-tab dialogs — the parent will do a single bulk refresh
        is_multi_tab = data.get('dialog') and hasattr(data['dialog'], '_tab_data')
        if not is_multi_tab:
            animated_refresh()

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
            try:
                file_path = data.get('file_path')
                if file_path:
                    epub_base = os.path.splitext(os.path.basename(file_path))[0]
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
                        expected_output_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), expected_output_dir)
                    expected_progress_file = os.path.join(expected_output_dir, "translation_progress.json")

                    # Update in-place if changed
                    if expected_output_dir and data.get('output_dir') != expected_output_dir:
                        data['output_dir'] = expected_output_dir
                    if expected_progress_file and data.get('progress_file') != expected_progress_file:
                        data['progress_file'] = expected_progress_file

                    # Keep cache consistent too (if present)
                    try:
                        file_key = os.path.abspath(file_path)
                        if hasattr(self, '_retranslation_dialog_cache') and file_key in self._retranslation_dialog_cache:
                            cached = self._retranslation_dialog_cache[file_key]
                            if isinstance(cached, dict):
                                cached['output_dir'] = data.get('output_dir')
                                cached['progress_file'] = data.get('progress_file')
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] Could not re-resolve output override on refresh: {e}")

            def _read_progress_json_safely(path):
                import random
                import time as _time
                last_error = None
                for _attempt in range(20):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            loaded = json.load(f)
                        data['_last_good_prog'] = copy.deepcopy(loaded)
                        return loaded
                    except (PermissionError, FileNotFoundError, json.JSONDecodeError, OSError) as e:
                        last_error = e
                        _time.sleep(min(0.5, 0.03 * (2 ** min(_attempt, 5))) + random.uniform(0, 0.03))
                snapshot = data.get('_last_good_prog') or data.get('prog')
                if isinstance(snapshot, dict):
                    print(f"⚠️ Progress file locked during refresh; using last good snapshot this tick: {last_error}")
                    return copy.deepcopy(snapshot)
                raise last_error

            def _write_progress_json_safely(path, payload):
                import random
                import tempfile
                import time as _time
                progress_dir = os.path.dirname(path) or '.'
                if progress_dir:
                    os.makedirs(progress_dir, exist_ok=True)
                last_error = None
                for _attempt in range(20):
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=progress_dir, delete=False, suffix='.tmp') as tmp:
                            temp_path = tmp.name
                            json.dump(payload, tmp, ensure_ascii=False, indent=2)
                            tmp.flush()
                            try:
                                os.fsync(tmp.fileno())
                            except Exception:
                                pass
                        os.replace(temp_path, path)
                        return True
                    except (PermissionError, OSError) as e:
                        last_error = e
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
                        _time.sleep(min(0.5, 0.03 * (2 ** min(_attempt, 5))) + random.uniform(0, 0.03))
                raise last_error
            
            # Save current scroll position (and first visible row/offset) to restore after refresh
            saved_scroll = None
            updates_were_enabled = True
            signals_were_blocked = False
            self._suspend_yield = True
            first_visible_row = None
            first_visible_offset = 0
            if 'listbox' in data and data['listbox']:
                try:
                    from PySide6.QtCore import QPoint
                    saved_scroll = data['listbox'].verticalScrollBar().value()
                    updates_were_enabled = data['listbox'].updatesEnabled()
                    signals_were_blocked = data['listbox'].signalsBlocked()
                    idx = data['listbox'].indexAt(QPoint(0, 0))
                    if idx and idx.isValid():
                        first_visible_row = idx.row()
                        rect = data['listbox'].visualItemRect(data['listbox'].item(first_visible_row))
                        first_visible_offset = -rect.top()
                    data['listbox'].blockSignals(True)
                    data['listbox'].setUpdatesEnabled(False)
                except Exception:
                    saved_scroll = None
            
            # Save current selections to restore after refresh
            selected_indices = []
            try:
                selected_indices = [data['listbox'].row(item) for item in data['listbox'].selectedItems()]
            except RuntimeError:
                print("⚠️ Could not save selection state - widget was deleted")
                return
            
            # Reload progress file - check if it exists first
            if not os.path.exists(data['progress_file']):
                print(f"⚠️ Progress file not found: {data['progress_file']}")
                # Recreate a minimal progress file and auto-discover completed files from output_dir
                prog = {
                    "chapters": {},
                    "chapter_chunks": {},
                    "version": "2.1"
                }

                def _auto_discover_from_output_dir(output_dir, prog):
                    updated = False
                    try:
                        files = [
                            f for f in os.listdir(output_dir)
                            if os.path.isfile(os.path.join(output_dir, f))
                            # accept any extension except .epub
                            and not f.lower().endswith("_translated.txt")
                            and f != "translation_progress.json"
                            and not f.lower().endswith(".epub")
                            and not f.lower().endswith(".cache")
                        ]
                        for fname in files:
                            base = os.path.basename(fname)
                            if base.startswith("response_"):
                                base = base[len("response_"):]
                            while True:
                                new_base, ext = os.path.splitext(base)
                                if not ext:
                                    break
                                base = new_base
                            import re
                            m = re.findall(r"(\\d+)", base)
                            chapter_num = int(m[-1]) if m else None
                            key = str(chapter_num) if chapter_num is not None else f"special_{base}"
                            actual_num = chapter_num if chapter_num is not None else 0
                            if key in prog.get("chapters", {}):
                                continue
                            prog.setdefault("chapters", {})[key] = {
                                "actual_num": actual_num,
                                "content_hash": "",
                                "output_file": fname,
                                "status": "completed",
                                "last_updated": os.path.getmtime(os.path.join(output_dir, fname)),
                                "auto_discovered": True,
                                "original_basename": fname
                            }
                            updated = True
                    except Exception as e:
                        print(f"⚠️ Auto-discovery (refresh no OPF) failed: {e}")
                    return updated

                if _auto_discover_from_output_dir(data['output_dir'], prog):
                    print("💾 Recreated progress file via auto-discovery (refresh)")
                try:
                    _write_progress_json_safely(data['progress_file'], prog)
                except PermissionError as e:
                    print(f"⚠️ Progress file locked during refresh recreate; will retry on next refresh tick: {e}")
                    return
                except Exception as e:
                    self._styled_msgbox(QMessageBox.Warning, data.get('dialog', self), "Progress File Error",
                                        f"Could not recreate progress file:\n{e}")
                    return
            
            # The translator may briefly lock/replace the JSON; retry and skip this tick if it stays locked.
            data['prog'] = _read_progress_json_safely(data['progress_file'])
            data['_last_good_prog'] = copy.deepcopy(data['prog'])

            def _progress_has_active_entries(prog):
                try:
                    return any(
                        isinstance(info, dict)
                        and str(info.get('status', '')).lower() == 'in_progress'
                        for info in (prog or {}).get('chapters', {}).values()
                    )
                except Exception:
                    return False
            
            # Clean up missing files and merged children before display unless disabled
            if not data.get('skip_cleanup', False) and not _progress_has_active_entries(data['prog']):
                from TransateKRtoEN import ProgressManager
                before_cleanup = copy.deepcopy(data['prog'])
                temp_progress = ProgressManager(os.path.dirname(data['progress_file']))
                temp_progress.prog = data['prog']
                temp_progress.cleanup_missing_files(data['output_dir'])
                data['prog'] = temp_progress.prog
                
                # Save only if cleanup really changed the file. During active translation
                # refresh should be a reader, not another progress writer.
                if data['prog'] != before_cleanup:
                    _write_progress_json_safely(data['progress_file'], data['prog'])

            if self._reconcile_tts_audio_files(data):
                _write_progress_json_safely(data['progress_file'], data['prog'])
            
            # Check if we're using OPF-based display or fallback
            if data.get('spine_chapters'):
                # OPF-based: Re-run full matching logic to update merged status correctly
                # We need to re-match spine chapters against the updated progress JSON
                self._rematch_spine_chapters(data)
            else:
                # Fallback mode: REBUILD chapter_display_info from scratch to pick up new entries
                # This is necessary for text files or EPUBs without OPF
                self._rebuild_chapter_display_info(data)
            
            # Note: chapter_display_info is already rebuilt/updated above
            # For OPF mode: _update_chapter_status_info updated existing entries
            # For fallback mode: _rebuild_chapter_display_info rebuilt from scratch
            
            # Update the listbox display
            self._update_listbox_display(data)
            
            # Update statistics if available
            self._update_statistics_display(data)

            # Ensure the special-files toggle is applied after every refresh.
            try:
                show_special = data.get('show_special_files_state', False)
                cb = data.get('show_special_files_cb')
                if cb:
                    show_special = cb.isChecked()
                listbox = data.get('listbox')
                if listbox:
                    for i in range(listbox.count()):
                        item = listbox.item(i)
                        if not item:
                            continue
                        meta = item.data(Qt.UserRole) or {}
                        _info = meta.get('info') or {}
                        _fname = _info.get('original_filename', '') or _info.get('output_file', '') or _info.get('key', '')
                        is_skipped_special = self._progress_file_is_skipped_special(
                            _fname,
                            meta.get('is_special', False),
                        )
                        item.setHidden(is_skipped_special and not show_special)
                data['show_special_files_state'] = show_special
            except Exception:
                pass
            
            # Restore scroll position and repaint immediately after rebuild
            if 'listbox' in data and data['listbox']:
                try:
                    sb = data['listbox'].verticalScrollBar()
                    if first_visible_row is not None and first_visible_row < data['listbox'].count():
                        item = data['listbox'].item(first_visible_row)
                        data['listbox'].scrollToItem(item, data['listbox'].PositionAtTop)
                        sb.setValue(sb.value() - first_visible_offset)
                    elif saved_scroll is not None:
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
            
            # Restore selections
            try:
                if selected_indices:
                    for idx in selected_indices:
                        if idx < data['listbox'].count():
                            data['listbox'].item(idx).setSelected(True)
                    # Update selection count
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText(f"Selected: {len(selected_indices)}")
                else:
                    # Clear selections if there were none
                    data['listbox'].clearSelection()
                    if 'selection_count_label' in data and data['selection_count_label']:
                        data['selection_count_label'].setText("Selected: 0")

                # Re-apply scroll AFTER selections (since selecting can auto-scroll)
                if saved_scroll is not None and 'listbox' in data and data['listbox']:
                    from PySide6.QtCore import QTimer
                    def _restore_scroll_again():
                        try:
                            sb = data['listbox'].verticalScrollBar()
                            target = min(saved_scroll, sb.maximum())
                            if sb.value() != target:
                                sb.setValue(target)
                        except Exception:
                            pass
                    QTimer.singleShot(0, _restore_scroll_again)
            except RuntimeError:
                print("⚠️ Could not restore selection state - widget was deleted during refresh")
            
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
    
    def _rematch_spine_chapters(self, data):
        """Re-run the full spine chapter matching logic against updated progress JSON"""
        prog = data['prog']
        output_dir = data['output_dir']
        spine_chapters = data['spine_chapters']

        def _normalize_opf_match_name(name: str) -> str:
            if not name:
                return ""
            base = os.path.basename(name)
            if base.startswith("response_"):
                base = base[len("response_"):]
            while True:
                new_base, ext = os.path.splitext(base)
                if not ext:
                    break
                base = new_base
            return base

        def _opf_names_equal(a: str, b: str) -> bool:
            return _normalize_opf_match_name(a) == _normalize_opf_match_name(b)

        # Build indexes once (O(n))
        basename_to_progress = {}
        response_to_progress = {}
        actualnum_to_progress = {}
        composite_to_progress = {}

        chapters_dict = prog.get("chapters", {})
        for ch in chapters_dict.values():
            orig = ch.get("original_basename", "")
            out = ch.get("output_file", "")
            actual_num = ch.get("actual_num")

            if orig:
                basename_to_progress.setdefault(_normalize_opf_match_name(orig), []).append(ch)
            if out:
                response_to_progress.setdefault(out, []).append(ch)
                norm_out = _normalize_opf_match_name(out)
                if norm_out != out:
                    response_to_progress.setdefault(norm_out, []).append(ch)
            if actual_num is not None:
                actualnum_to_progress.setdefault(actual_num, []).append(ch)

            fname_for_comp = orig or out
            if fname_for_comp and actual_num is not None:
                filename_noext = os.path.splitext(_normalize_opf_match_name(fname_for_comp))[0]
                composite_to_progress[f"{actual_num}_{filename_noext}"] = ch

        # Cache directory listing to avoid thousands of exists calls
        try:
            existing_files = {f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))}
        except Exception:
            existing_files = set()

        def file_exists_fast(fname: str) -> bool:
            return fname in existing_files

        for spine_ch in spine_chapters:
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            is_special = spine_ch.get('is_special', False)

            base_name = os.path.splitext(filename)[0]
            retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)

            if is_special:
                response_with_prefix = f"response_{base_name}.html"
                if retain:
                    expected_response = filename
                elif file_exists_fast(response_with_prefix):
                    expected_response = response_with_prefix
                else:
                    expected_response = filename
            else:
                expected_response = filename if retain else filename

            matched_info = None
            basename_key = _normalize_opf_match_name(filename)

            # 1) original basename map
            lst = basename_to_progress.get(basename_key)
            if lst:
                for ch in lst:
                    status = ch.get('status', '')
                    if status in ['in_progress', 'failed', 'qa_failed', 'pending']:
                        if ch.get('actual_num') == chapter_num:
                            matched_info = ch
                            break
                    else:
                        matched_info = ch
                        break

            # 2) response map (choose highest severity, prefer matching chapter_num)
            if not matched_info:
                lookup_keys = [
                    expected_response,
                    _normalize_opf_match_name(expected_response),
                    f"response_{expected_response}" if not expected_response.startswith("response_") else expected_response,
                    basename_key
                ]
                lst = None
                for k in lookup_keys:
                    if k in response_to_progress:
                        lst = response_to_progress[k]
                        break
                if lst:
                    has_qa = any(ch.get('status') == 'qa_failed' for ch in lst)
                    if has_qa:
                        lst = [ch for ch in lst if ch.get('status') != 'pending']
                    severity = {'qa_failed': 4, 'failed': 3, 'pending': 2, 'in_progress': 1, 'completed': 0}
                    best = None
                    best_score = -1
                    for ch in lst:
                        status = ch.get('status', '')
                        score = severity.get(status, -1)
                        matches_num = ch.get('actual_num') == chapter_num
                        if score > best_score or (score == best_score and matches_num):
                            best = ch
                            best_score = score
                            # If exact chapter match and highest severity, keep going in case of even higher severity
                    if best:
                        matched_info = best

            # 3) composite key
            if not matched_info:
                filename_noext = base_name
                if filename_noext.startswith("response_"):
                    filename_noext = filename_noext[len("response_"):]
                comp_key = f"{chapter_num}_{filename_noext}"
                matched_info = composite_to_progress.get(comp_key)

            # 4) actual_num map fallback (avoid mis-matching special files)
            if not matched_info and chapter_num in actualnum_to_progress:
                for ch in actualnum_to_progress[chapter_num]:
                    status = ch.get('status', '')
                    out_file = ch.get('output_file')
                    orig_base = os.path.basename(ch.get('original_basename', '') or '')

                    # If this spine entry is a special file (no digits), require filename match to avoid hijacking by other chapter 0 entries
                    if is_special:
                        fname_matches = (
                            (orig_base and _opf_names_equal(orig_base, filename)) or
                            (out_file and (_opf_names_equal(out_file, expected_response) or out_file == expected_response))
                        )
                        if not fname_matches:
                            continue

                    if status == 'merged':
                        if _opf_names_equal(orig_base, filename) or not orig_base:
                            matched_info = ch
                            break
                    elif status in ['in_progress', 'failed', 'pending', 'qa_failed']:
                        if out_file and (_opf_names_equal(out_file, expected_response) or out_file == expected_response):
                            matched_info = ch
                            break
                    else:
                        if (orig_base and _opf_names_equal(orig_base, filename)) or (out_file and (_opf_names_equal(out_file, expected_response) or out_file == expected_response)):
                            matched_info = ch
                            break

            file_exists = file_exists_fast(expected_response)

            if matched_info:
                status = matched_info.get('status', 'unknown')

                if status in ['failed', 'in_progress', 'qa_failed', 'pending']:
                    spine_ch['status'] = status
                    spine_ch['output_file'] = matched_info.get('output_file') or expected_response
                    spine_ch['progress_entry'] = matched_info
                    continue

                spine_ch['status'] = status
                spine_ch['output_file'] = expected_response if is_special else matched_info.get('output_file', expected_response)
                spine_ch['progress_entry'] = matched_info
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response

                if status == 'completed':
                    output_file = spine_ch['output_file']
                    if not file_exists_fast(output_file):
                        if file_exists and expected_response:
                            spine_ch['output_file'] = expected_response
                            matched_info['output_file'] = expected_response
                        else:
                            spine_ch['status'] = 'not_translated'

            elif file_exists:
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response

            else:
                norm_target = _normalize_opf_match_name(filename)
                matched_file = None
                for f in existing_files:
                    if _normalize_opf_match_name(f) == norm_target:
                        matched_file = f
                        break
                if matched_file:
                    spine_ch['status'] = 'completed'
                    spine_ch['output_file'] = matched_file
                else:
                    spine_ch['status'] = 'not_translated'
                    spine_ch['output_file'] = expected_response
        
        # =====================================================
        # SAVE AUTO-DISCOVERED FILES TO PROGRESS (refresh path)
        # =====================================================
        
        progress_updated = False
        for spine_ch in spine_chapters:
            # Only add entries that were marked as completed but have no progress entry
            if spine_ch['status'] == 'completed' and 'progress_entry' not in spine_ch:
                chapter_num = spine_ch['file_chapter_num']
                output_file = spine_ch['output_file']
                filename = spine_ch['filename']

                # Require normalized filename match between spine file and output file, and the file must exist
                norm_spine = _normalize_opf_match_name(filename)
                norm_out = _normalize_opf_match_name(output_file)
                file_exists = os.path.exists(os.path.join(output_dir, output_file))
                if norm_spine != norm_out or not file_exists:
                    continue

                # Create a progress entry for this auto-discovered file
                chapter_key = str(chapter_num)
                
                # Check if key already exists (avoid duplicates)
                if chapter_key not in prog.get("chapters", {}):
                    prog.setdefault("chapters", {})[chapter_key] = {
                        "actual_num": chapter_num,
                        "content_hash": "",  # Unknown since we don't have the source
                        "output_file": output_file,
                        "status": "completed",
                        "last_updated": os.path.getmtime(os.path.join(output_dir, output_file)),
                        "auto_discovered": True,
                        "original_basename": filename
                    }
                    progress_updated = True
                    print(f"✅ Auto-discovered and tracked (refresh): {filename} -> {output_file}")
        
        # Save progress file if we added new entries
        if progress_updated:
            try:
                with open(data['progress_file'], 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved {sum(1 for ch in spine_chapters if ch['status'] == 'completed' and 'progress_entry' not in ch)} auto-discovered files to progress file (refresh)")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress file during refresh: {e}")
        
        # Rebuild chapter_display_info from updated spine_chapters
        chapter_display_info = []
        for spine_ch in spine_chapters:
            display_info = {
                'key': spine_ch.get('filename', ''),
                'num': spine_ch['file_chapter_num'],
                'info': spine_ch.get('progress_entry', {}),
                'output_file': spine_ch['output_file'],
                'status': spine_ch['status'],
                'duplicate_count': 1,
                'entries': [],
                'opf_position': spine_ch['position'],
                'original_filename': spine_ch['filename'],
                'is_special': spine_ch.get('is_special', False),
                'progress_key': spine_ch.get('progress_key')
            }
            chapter_display_info.append(display_info)
        
        self._append_pdf_ocr_display_info(data, chapter_display_info)
        self._append_image_gen_display_info(data, chapter_display_info)
        data['chapter_display_info'] = chapter_display_info
    
    def _rebuild_chapter_display_info(self, data):
        """Rebuild chapter_display_info from scratch (for fallback mode without OPF)"""
        # This is the same logic as the initial build in _force_retranslation_epub_or_text
        # but extracted here so refresh can use it
        
        prog = data['prog']
        output_dir = data['output_dir']
        file_path = data.get('file_path', '')
        show_special = data.get('show_special_files_state', False)
        
        # Known non-chapter files that should never appear in the progress list
        _non_chapter_files = {"glossary.csv", "metadata.json", "styles.css", "rolling_summary.txt"}
        _source_has_translated = "_translated" in os.path.basename(file_path).lower()
        files_to_entries = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            status = chapter_info.get("status", "")
            
            # Skip known non-chapter files
            if output_file and output_file.lower() in _non_chapter_files:
                continue
            # Skip combined _translated output files (unless source itself has _translated)
            if output_file and not _source_has_translated and any(
                output_file.lower().endswith(s) for s in ("_translated.txt", "_translated.pdf", "_translated.html")
            ):
                continue
            
            # Include chapters with output files OR in_progress/failed/qa_failed with null output file (legacy)
            if output_file or status in ["in_progress", "failed", "qa_failed"]:
                # For merged chapters, use a unique key (chapter_key) instead of output_file
                # This ensures merged chapters appear as separate entries in the list
                if status == "merged":
                    file_key = f"_merged_{chapter_key}"
                elif output_file:
                    file_key = output_file
                elif status == "in_progress":
                    file_key = f"_in_progress_{chapter_key}"
                elif status == "qa_failed":
                    file_key = f"_qa_failed_{chapter_key}"
                else:  # failed
                    file_key = f"_failed_{chapter_key}"
                
                if file_key not in files_to_entries:
                    files_to_entries[file_key] = []
                files_to_entries[file_key].append((chapter_key, chapter_info))
        
        chapter_display_info = []
        
        for output_file, entries in files_to_entries.items():
            chapter_key, chapter_info = entries[0]
            
            # Get the actual output file (strip placeholder prefix if present)
            actual_output_file = output_file
            if output_file.startswith("_merged_") or output_file.startswith("_in_progress_") or output_file.startswith("_failed_") or output_file.startswith("_qa_failed_"):
                # For merged/in_progress/failed/qa_failed, get the actual output_file from chapter_info
                actual_output_file = chapter_info.get("output_file", "")
                if not actual_output_file:
                    # Generate expected filename based on actual_num
                    actual_num = chapter_info.get("actual_num")
                    if actual_num is not None:
                        # Use .txt extension for text files, .html for EPUB
                        ext = ".txt" if file_path.endswith(".txt") else ".html"
                        actual_output_file = f"response_section_{actual_num}{ext}"
            
            # Check if this is a special file using configured keyword lists
            original_basename = chapter_info.get("original_basename", "")
            filename_to_check = original_basename if original_basename else actual_output_file
            
            is_special = self._is_special_file(filename_to_check) if hasattr(self, '_is_special_file') else (not bool(re.search(r'\d', filename_to_check)))
            
            # Don't skip special files here - let the display logic handle hiding them
            # This ensures chapter_display_info contains all items, and the listbox
            # will properly hide/show items based on the toggle state
            
            # Extract chapter number - prioritize stored values
            chapter_num = None
            if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                chapter_num = chapter_info['actual_num']
            elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                chapter_num = chapter_info['chapter_num']
            
            # Fallback: extract from filename
            if chapter_num is None:
                import re
                matches = re.findall(r'(\d+)', actual_output_file)
                if matches:
                    chapter_num = int(matches[-1])
                else:
                    chapter_num = 999999
            
            status = chapter_info.get("status", "unknown")
            if status in ("completed_empty", "completed_image_only"):
                status = "completed"
            
            # Check file existence
            if status == "completed":
                output_path = os.path.join(output_dir, actual_output_file)
                if not os.path.exists(output_path):
                    status = "not_translated"
            
            chapter_display_info.append({
                'key': chapter_key,
                'num': chapter_num,
                'info': chapter_info,
                'output_file': actual_output_file,  # Use actual output file, not placeholder
                'status': status,
                'duplicate_count': len(entries),
                'entries': entries,
                'is_special': is_special,
                'progress_key': chapter_key
            })
        
        # Sort by chapter number
        chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)
        
        self._append_pdf_ocr_display_info(data, chapter_display_info)
        self._append_image_gen_display_info(data, chapter_display_info)

        # Update data with rebuilt list
        data['chapter_display_info'] = chapter_display_info

    def _append_pdf_ocr_display_info(self, data, chapter_display_info):
        """Add a lightweight summary row for PDF Vision OCR progress."""
        try:
            prog = data.get('prog') or {}
            pdf_ocr = prog.get('pdf_ocr')
            progress_output_mode = str(prog.get('output_mode') or data.get('output_mode') or '').lower().strip()
            ui_output_mode = ""
            try:
                if hasattr(self, '_get_output_mode'):
                    ui_output_mode = str(self._get_output_mode() or '').lower().strip()
            except Exception:
                ui_output_mode = ""
            if ui_output_mode and ui_output_mode != 'vision':
                return
            if progress_output_mode and progress_output_mode != 'vision':
                return
            current_file = str(data.get('file_path') or '')
            if current_file and not current_file.lower().endswith('.pdf'):
                return
            if not current_file:
                pdf_source = ""
                if isinstance(pdf_ocr, dict):
                    pdf_source = str(pdf_ocr.get('source_file') or '')
                if not pdf_source or not pdf_source.lower().endswith('.pdf'):
                    return
            if not isinstance(pdf_ocr, dict):
                output_dir = data.get('output_dir') or ''
                image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')
                image_dir = os.path.join(output_dir, 'images')
                single_dir = os.path.join(output_dir, 'OCR', 'single')
                image_count = 0
                cached_count = 0
                try:
                    if os.path.isdir(image_dir):
                        image_count = sum(
                            1 for name in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, name)) and name.lower().endswith(image_exts)
                        )
                except Exception:
                    image_count = 0
                try:
                    if os.path.isdir(single_dir):
                        cached_count = sum(
                            1 for name in os.listdir(single_dir)
                            if os.path.isfile(os.path.join(single_dir, name)) and name.lower().endswith('.txt')
                        )
                except Exception:
                    cached_count = 0
                total_guess = max(image_count, cached_count)
                if total_guess <= 0:
                    return
                pdf_ocr = {
                    'source_file': data.get('file_path'),
                    'ocr_source_file': '',
                    'status': 'completed' if cached_count >= total_guess else 'in_progress',
                    'total': total_guess,
                    'done': cached_count,
                    'cached': cached_count,
                    'no_text': 0,
                    'failed': 0,
                    'cache_inferred': True,
                }
            total = int(pdf_ocr.get('total') or 0)
            pages = pdf_ocr.get('pages') if isinstance(pdf_ocr.get('pages'), dict) else {}
            if total <= 0 and pages:
                total = len(pages)
            if total <= 0:
                return
            done = int(pdf_ocr.get('done') or 0)
            cached = int(pdf_ocr.get('cached') or 0)
            failed = int(pdf_ocr.get('failed') or 0)
            no_text = int(pdf_ocr.get('no_text') or 0)
            status = str(pdf_ocr.get('status') or 'in_progress').lower().strip()
            if status not in ('completed', 'failed', 'cancelled'):
                status = 'in_progress'
            elif status == 'cancelled':
                status = 'failed'
            source_file = os.path.basename(str(pdf_ocr.get('source_file') or data.get('file_path') or 'PDF'))
            ocr_source_file = os.path.basename(str(pdf_ocr.get('ocr_source_file') or ''))
            label_bits = [f"{min(done, total)}/{total} pages"]
            if cached:
                label_bits.append(f"{cached} cached")
            if no_text:
                label_bits.append(f"{no_text} no-text")
            if failed:
                label_bits.append(f"{failed} failed")
            output_label = f"{source_file} -> {ocr_source_file or '_OCR.pdf'} ({', '.join(label_bits)})"
            info = dict(pdf_ocr)
            info['status'] = status
            info['ocr_progress'] = {
                'done': min(done, total),
                'total': total,
                'label': f"{min(done, total)}/{total}",
            }
            chapter_display_info.insert(0, {
                'key': '__pdf_ocr__',
                'num': 0,
                'info': info,
                'output_file': output_label,
                'status': status,
                'duplicate_count': 1,
                'entries': [],
                'is_special': False,
                'progress_key': '__pdf_ocr__',
                'pdf_ocr': True,
            })
        except Exception as e:
            print(f"Warning: could not read PDF OCR progress: {e}")

    def _append_image_gen_display_info(self, data, chapter_display_info):
        """Add a lightweight summary row for image generation progress (image output mode)."""
        try:
            prog = data.get('prog') or {}
            image_gen = prog.get('image_gen')
            if not isinstance(image_gen, dict):
                return
            # Hide the row when the user switches output mode away from 'image'.
            # Check the live GUI combo directly — it's the most reliable source.
            combo = getattr(self, '_output_mode_combo', None)
            if combo is not None:
                try:
                    live_mode = {0: 'text', 1: 'vision', 2: 'image', 3: 'video', 4: 'audio', 5: 'refinement'}.get(combo.currentIndex(), 'text')
                    if live_mode != 'image':
                        return
                except Exception:
                    pass
            # Verify the source is epub/pdf using the stored source_file or the current file.
            source_file = str(image_gen.get('source_file') or data.get('file_path') or '')
            if source_file and not source_file.lower().endswith(('.epub', '.pdf')):
                return

            total = int(image_gen.get('total') or 0)
            if total <= 0:
                return
            done = int(image_gen.get('done') or 0)
            success = int(image_gen.get('success') or 0)
            skipped = int(image_gen.get('skipped') or 0)
            failed = int(image_gen.get('failed') or 0)
            status = str(image_gen.get('status') or 'in_progress').lower().strip()
            if status not in ('completed', 'failed', 'cancelled'):
                status = 'in_progress'
            elif status == 'cancelled':
                status = 'failed'

            source_file = os.path.basename(str(image_gen.get('source_file') or data.get('file_path') or 'EPUB'))
            label_bits = [f"{min(done, total)}/{total} images"]
            if success:
                label_bits.append(f"{success} generated")
            if skipped:
                label_bits.append(f"{skipped} skipped")
            if failed:
                label_bits.append(f"{failed} failed")
            output_label = f"🎨 Image Generation: {source_file} ({', '.join(label_bits)})"

            info = dict(image_gen)
            info['status'] = status
            info['image_gen_progress'] = {
                'done': min(done, total),
                'total': total,
                'label': f"{min(done, total)}/{total}",
            }
            chapter_display_info.insert(0, {
                'key': '__image_gen__',
                'num': 0,
                'info': info,
                'output_file': output_label,
                'status': status,
                'duplicate_count': 1,
                'entries': [],
                'is_special': False,
                'progress_key': '__image_gen__',
                'image_gen': True,
            })
        except Exception as e:
            print(f"Warning: could not read image gen progress: {e}")
    
    def _current_progress_output_mode(self, data=None, entry=None):
        """Prefer the live GUI output mode over stale mode values saved in progress JSON."""
        candidates = []

        combo = getattr(self, '_output_mode_combo', None)
        if combo is not None:
            try:
                idx_mode = {0: 'text', 1: 'vision', 2: 'image', 3: 'video', 4: 'audio', 5: 'refinement'}.get(combo.currentIndex())
                if idx_mode:
                    candidates.append(idx_mode)
            except RuntimeError:
                pass
            except Exception:
                pass
            try:
                candidates.append(combo.currentText())
            except RuntimeError:
                pass
            except Exception:
                pass

        try:
            if hasattr(self, '_get_output_mode'):
                candidates.append(self._get_output_mode())
        except Exception:
            pass

        candidates.append(getattr(self, 'output_mode_var', None))

        config = getattr(self, 'config', None)
        if isinstance(config, dict):
            candidates.append(config.get('output_mode'))
            if config.get('enable_audio_output_mode'):
                candidates.append('audio')
            if config.get('enable_refinement_output_mode'):
                candidates.append('refinement')

        prog = (data or {}).get('prog') or {}
        candidates.append(prog.get('output_mode'))
        if isinstance(entry, dict):
            candidates.append(entry.get('output_mode'))

        for candidate in candidates:
            mode = str(candidate or '').lower().strip()
            if 'audio' in mode:
                return 'audio'
            if 'refine' in mode or 'refinement' in mode:
                return 'refinement'
            if mode in ('text', 'vision', 'image', 'video'):
                return mode
        return 'text'

    def _audio_stem_variants(self, output_file):
        stem = os.path.splitext(os.path.basename(output_file or ""))[0]
        if not stem:
            return []
        variants = [stem]
        if stem.startswith("response_"):
            variants.append(stem[len("response_"):])
        else:
            variants.append(f"response_{stem}")
        return list(dict.fromkeys(variants))

    def _normalize_progress_output_name(self, name: str) -> str:
        """Normalize translated/source output names for response_/extension-tolerant matching."""
        if not name:
            return ""
        base = os.path.basename(str(name).replace("\\", "/"))
        if base.lower().startswith("response_"):
            base = base[len("response_"):]
        while True:
            stem, ext = os.path.splitext(base)
            if not ext:
                break
            base = stem
        return base.lower()

    def _resolve_existing_output_path(self, output_dir, output_file=None, display_info=None, prog=None):
        """Resolve an output file while tolerating stale OCR rows and filename mode changes."""
        display_info = display_info or {}
        progress_entry = display_info.get("info") or display_info.get("progress_entry") or {}
        prog = prog or {}
        chapters = prog.get("chapters", {}) if isinstance(prog, dict) else {}
        candidates = []

        def add_candidate(value):
            if value:
                text = str(value).replace("\\", "/")
                if text not in candidates:
                    candidates.append(text)

        add_candidate(output_file)
        add_candidate(display_info.get("output_file"))
        add_candidate(progress_entry.get("output_file") if isinstance(progress_entry, dict) else None)
        if isinstance(progress_entry, dict):
            previous = progress_entry.get("previous_progress_entry")
            if isinstance(previous, dict):
                add_candidate(previous.get("output_file"))

        progress_key = display_info.get("progress_key")
        if progress_key and isinstance(chapters.get(progress_key), dict):
            tracked = chapters[progress_key]
            add_candidate(tracked.get("output_file"))
            previous = tracked.get("previous_progress_entry")
            if isinstance(previous, dict):
                add_candidate(previous.get("output_file"))

        target_num = display_info.get("num")
        original_names = {
            self._normalize_progress_output_name(display_info.get("original_filename")),
            self._normalize_progress_output_name(display_info.get("original_basename")),
        }
        original_names.discard("")

        for tracked in chapters.values():
            if not isinstance(tracked, dict):
                continue
            tracked_num = tracked.get("actual_num", tracked.get("chapter_num"))
            tracked_names = {
                self._normalize_progress_output_name(tracked.get("output_file")),
                self._normalize_progress_output_name(tracked.get("original_basename")),
                self._normalize_progress_output_name(tracked.get("original_filename")),
            }
            if str(tracked_num) == str(target_num) or (original_names and tracked_names & original_names):
                add_candidate(tracked.get("output_file"))
                previous = tracked.get("previous_progress_entry")
                if isinstance(previous, dict):
                    add_candidate(previous.get("output_file"))

        for candidate in candidates:
            path = candidate if os.path.isabs(candidate) else os.path.join(output_dir, candidate)
            if os.path.isfile(path):
                rel = os.path.relpath(path, output_dir).replace("\\", "/") if os.path.isabs(candidate) else candidate
                return rel, path

        target_norms = {self._normalize_progress_output_name(value) for value in candidates if value}
        target_norms |= original_names
        target_norms.discard("")
        if not target_norms:
            return None, None
        try:
            for fname in os.listdir(output_dir):
                path = os.path.join(output_dir, fname)
                if os.path.isfile(path) and self._normalize_progress_output_name(fname) in target_norms:
                    return fname, path
        except Exception:
            pass
        return None, None

    def _audio_candidates_for_entry(self, output_dir, entry):
        """Return possible audio files for a progress entry as (relative, absolute) pairs."""
        candidates = []
        if not isinstance(entry, dict):
            return candidates

        stored = entry.get('tts_file')
        if stored:
            candidates.append(stored)

        output_file = entry.get('output_file')
        if output_file:
            for stem in self._audio_stem_variants(output_file):
                for ext in ("wav", "mp3", "pcm", "m4a", "ogg", "flac"):
                    candidates.append(os.path.join("text_to_speech", f"{stem}.{ext}"))

        seen = set()
        resolved = []
        for candidate in candidates:
            if not candidate:
                continue
            normalized = str(candidate).replace("\\", "/")
            if normalized in seen:
                continue
            seen.add(normalized)
            abs_path = normalized if os.path.isabs(normalized) else os.path.join(output_dir, normalized)
            rel_path = os.path.relpath(abs_path, output_dir).replace("\\", "/") if os.path.isabs(normalized) else normalized
            resolved.append((rel_path, abs_path))
        return resolved

    def _existing_audio_for_entry(self, output_dir, entry):
        for rel_path, abs_path in self._audio_candidates_for_entry(output_dir, entry):
            if os.path.exists(abs_path):
                return rel_path, abs_path
        return None, None

    def _reconcile_tts_audio_files(self, data):
        """Keep progress TTS status aligned with generated audio files on disk."""
        prog = data.get('prog') or {}
        output_dir = data.get('output_dir')
        if not output_dir:
            return False

        changed = False
        now = time.time()
        for _key, entry in prog.get('chapters', {}).items():
            if not isinstance(entry, dict):
                continue
            output_file = entry.get('output_file')
            if not output_file:
                continue
            rel_audio, _abs_audio = self._existing_audio_for_entry(output_dir, entry)
            tts_status = str(entry.get('tts_status') or 'no_tts').lower().strip()

            if rel_audio:
                if tts_status not in ('tts_completed', 'completed') or entry.get('tts_file') != rel_audio:
                    entry['tts_status'] = 'tts_completed'
                    entry['tts_file'] = rel_audio
                    entry.pop('tts_error', None)
                    entry.setdefault('tts_at', now)
                    entry['last_updated'] = now
                    changed = True
                continue

            had_audio_state = (
                entry.get('tts_file')
                or tts_status in ('tts_completed', 'completed', 'in_progress')
            )
            if had_audio_state:
                entry['tts_status'] = 'no_tts'
                entry.pop('tts_file', None)
                entry.pop('tts_at', None)
                entry.pop('tts_error', None)
                entry['last_updated'] = now
                changed = True
        return changed

    def _progress_display_status(self, info, data=None):
        """Derive the status shown in Progress Manager for post-processing modes."""
        status = info.get('status', 'unknown')
        entry = info.get('progress_entry') or info.get('info') or {}
        mode = self._current_progress_output_mode(data, entry)

        if status in ('completed_empty', 'completed_image_only'):
            status = 'completed'

        ref_status = str(entry.get('refinement_status') or '').lower().strip()
        tts_status = str(entry.get('tts_status') or '').lower().strip()
        if status == 'in_progress' and (ref_status == 'in_progress' or tts_status == 'in_progress'):
            return 'in_progress'

        if status == 'in_progress' and data and data.get('output_dir'):
            previous_status = str(entry.get('previous_status') or '').lower().strip()
            previous_entry = entry.get('previous_progress_entry')
            if previous_status in ('completed', 'completed_empty', 'completed_image_only') or (
                isinstance(previous_entry, dict)
                and str(previous_entry.get('status') or '').lower().strip() in ('completed', 'completed_empty', 'completed_image_only')
            ):
                _resolved_file, resolved_path = self._resolve_existing_output_path(
                    data.get('output_dir'),
                    info.get('output_file') or entry.get('output_file'),
                    info,
                    data.get('prog'),
                )
                if resolved_path and os.path.exists(resolved_path):
                    return 'completed'

        if status in ('failed', 'qa_failed', 'in_progress', 'pending', 'merged', 'not_translated'):
            return status
        if mode == 'refinement':
            ref_status = ref_status or 'not_refined'
            if ref_status in ('failed', 'error'):
                return 'failed'
            if ref_status == 'in_progress':
                return 'in_progress'
            if ref_status not in ('refined', 'completed'):
                return 'not_refined'
        if mode == 'audio':
            tts_status = tts_status or 'no_tts'
            if tts_status in ('failed', 'error'):
                return 'failed'
            if tts_status == 'in_progress':
                return 'in_progress'
            if tts_status not in ('tts_completed', 'completed'):
                return 'no_tts'
        return status

    def _progress_entry_is_refined(self, info):
        """Return True when a completed progress entry also has refined output."""
        try:
            entry = info.get('progress_entry') or info.get('info') or info
            if not isinstance(entry, dict):
                return False
            return str(entry.get('refinement_status') or '').lower().strip() in ('refined', 'completed')
        except Exception:
            return False

    def _update_chapter_status_info(self, data):
        """Update chapter status information after refresh"""
        # Re-check file existence and update status for each chapter
        for info in data['chapter_display_info']:
            output_file = info['output_file']
            resolved_output_file, resolved_output_path = self._resolve_existing_output_path(
                data['output_dir'],
                output_file,
                info,
                data.get('prog'),
            )
            output_path = resolved_output_path or os.path.join(data['output_dir'], output_file)
            
            # Find matching progress entry
            matched_info = None
            
            # PRIORITY 1: Match by BOTH actual_num AND output_file
            # This prevents cross-matching between files with same chapter number but different filenames
            for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                ch_output = chapter_info.get('output_file')
                
                # BOTH must match - no fallback
                if actual_num is not None and actual_num == info['num'] and ch_output == output_file:
                    matched_info = chapter_info
                    break
            
            # PRIORITY 2: Fall back to output_file matching if no actual_num match
            if not matched_info:
                # Prefer completed over failed/pending/in_progress; keep qa_failed highest
                severity = {'qa_failed': 5, 'completed': 4, 'failed': 3, 'pending': 2, 'in_progress': 1}
                best = None
                best_score = -1
                for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                    if chapter_info.get('output_file') == output_file:
                        status = chapter_info.get('status', 'unknown')
                        score = severity.get(status, -1)
                        # Prefer higher severity; tie-breaker: matching actual_num if present
                        matches_num = (chapter_info.get('actual_num') or chapter_info.get('chapter_num')) == info['num']
                        if score > best_score or (score == best_score and matches_num):
                            best_score = score
                            best = chapter_info
                if best:
                    matched_info = best
            
            # Update status based on current state from progress file
            if matched_info:
                new_status = matched_info.get('status', 'unknown')
                # Handle legacy completed variants as completed for display
                if new_status in ('completed_empty', 'completed_image_only'):
                    new_status = 'completed'
                # Verify file actually exists for completed status (but NOT for merged - merged chapters
                # don't have their own output files, they point to parent's file)
                if new_status == 'completed' and not os.path.exists(output_path):
                    new_status = 'not_translated'
                elif new_status == 'completed' and resolved_output_file:
                    info['output_file'] = resolved_output_file
                info['status'] = new_status
                info['info'] = matched_info
            elif os.path.exists(output_path):
                # Before marking as completed based on file existence, check if this chapter
                # is actually marked as merged in the progress file (by actual_num lookup)
                # This handles the case where old output files exist from before merging was enabled
                is_merged_chapter = False
                for chapter_key, chapter_info in data['prog'].get("chapters", {}).items():
                    actual_num = chapter_info.get('actual_num') or chapter_info.get('chapter_num')
                    if actual_num is not None and actual_num == info['num']:
                        if chapter_info.get('status') == 'merged':
                            is_merged_chapter = True
                            info['status'] = 'merged'
                            info['info'] = chapter_info
                            break
                
                if not is_merged_chapter:
                    info['status'] = 'completed'
                    info.pop('info', None)
                    info.pop('progress_entry', None)
            else:
                info['status'] = 'not_translated'
                info.pop('info', None)
                info.pop('progress_entry', None)

    def _progress_entry_model_name(self, info, data=None):
        """Return the model name attached to a progress row, with old-file fallbacks."""
        candidates = []
        if isinstance(info, dict):
            candidates.append(info)
            for key in ('info', 'progress_entry'):
                value = info.get(key)
                if isinstance(value, dict):
                    candidates.append(value)
                    previous = value.get('previous_progress_entry')
                    if isinstance(previous, dict):
                        candidates.append(previous)
        if isinstance(data, dict):
            prog = data.get('prog')
            if isinstance(prog, dict):
                progress_key = info.get('progress_key') if isinstance(info, dict) else None
                chapters = prog.get('chapters', {})
                if progress_key and isinstance(chapters, dict) and isinstance(chapters.get(progress_key), dict):
                    candidates.append(chapters[progress_key])

        for candidate in candidates:
            model_name = str(candidate.get('model_name') or candidate.get('model') or '').strip()
            if model_name:
                return model_name
        return "(model unknown)"

    def _progress_model_column_text(self, info, data, fallback_output):
        if isinstance(data, dict) and data.get('show_model_info_state'):
            return self._progress_entry_model_name(info, data)
        return fallback_output

    def _progress_list_column_widths(self, chapter_display_info, data):
        max_original_len = 0
        max_output_len = 0
        for info in chapter_display_info or []:
            if 'opf_position' not in info:
                continue
            original_file = info.get('original_filename', '')
            output_file = self._progress_model_column_text(info, data, info.get('output_file', ''))
            max_original_len = max(max_original_len, len(original_file))
            max_output_len = max(max_output_len, len(output_file))
        return max(max_original_len, 20), max(max_output_len, 25)

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

    def _progress_list_item_key(self, info):
        if not isinstance(info, dict):
            return None
        progress_key = info.get('progress_key')
        if progress_key:
            return f"progress:{progress_key}"
        output_file = info.get('output_file')
        if output_file:
            return f"output:{output_file}"
        return f"row:{info.get('num')}:{info.get('original_filename', '')}:{info.get('key', '')}"

    def _progress_list_display_text(self, info, data, max_original_len, max_output_len):
        status_icons = {
            'completed': '✅',
            'merged': '🔗',
            'failed': '❌',
            'qa_failed': '❌',
            'in_progress': '🔄',
            'pending': '❓',
            'not_translated': '⬜',
            'not_refined': '✨',
            'no_tts': '🔊',
            'unknown': '❓'
        }
        status_labels = {
            'completed': 'Completed',
            'merged': 'Merged',
            'failed': 'Failed',
            'qa_failed': 'QA Failed',
            'in_progress': 'In Progress',
            'pending': 'Pending',
            'not_translated': 'Not Translated',
            'not_refined': 'Not Refined',
            'no_tts': 'No TTS',
            'unknown': 'Unknown'
        }

        chapter_num = info['num']
        status = self._progress_display_status(info, data)
        output_file = info['output_file']
        output_display = self._progress_model_column_text(info, data, output_file)
        icon = status_icons.get(status, '❓')
        status_label = status_labels.get(status, status)
        if status == 'completed' and self._progress_entry_is_refined(info):
            status_label = f"{status_label} ⭐"
        chapter_info = info.get('info') or info.get('progress_entry') or {}
        ocr_progress = chapter_info.get('ocr_progress') if isinstance(chapter_info, dict) else None
        if status == 'in_progress' and isinstance(ocr_progress, dict):
            try:
                ocr_done = int(ocr_progress.get('done', 0))
                ocr_total = int(ocr_progress.get('total', 0))
            except (TypeError, ValueError):
                ocr_done = 0
                ocr_total = 0
            if ocr_total > 0:
                status_label = f"{status_label} ({min(ocr_done, ocr_total)}/{ocr_total})"

        if info.get('pdf_ocr'):
            display = f"PDF OCR | {icon} {status_label:18s} | {output_display}"
        elif 'opf_position' in info:
            original_file = info.get('original_filename', '')
            opf_pos = info['opf_position'] + 1
            if isinstance(chapter_num, float):
                if chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_display}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:06.1f} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_display}"
            else:
                display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:11s} | {original_file:<{max_original_len}} -> {output_display}"
        else:
            if isinstance(chapter_num, float) and chapter_num.is_integer():
                display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:11s} | {output_display}"
            elif isinstance(chapter_num, float):
                display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:11s} | {output_display}"
            else:
                display = f"Chapter {chapter_num:03d} | {icon} {status_label:11s} | {output_display}"

        if status == 'qa_failed':
            qa_issues = chapter_info.get('qa_issues_found', []) if isinstance(chapter_info, dict) else []
            if qa_issues:
                issues_display = ', '.join(qa_issues[:2])
                if len(qa_issues) > 2:
                    issues_display += f' (+{len(qa_issues)-2} more)'
                display += f" | {issues_display}"

        if status == 'merged':
            parent_chapter = chapter_info.get('merged_parent_chapter') if isinstance(chapter_info, dict) else None
            if parent_chapter:
                display += f" | → Ch.{parent_chapter}"

        if info.get('duplicate_count', 1) > 1:
            display += f" | ({info['duplicate_count']} entries)"

        return display, status

    def _apply_progress_list_item_visuals(self, item, status):
        if status == 'completed':
            item.setForeground(QColor('green'))
        elif status == 'merged':
            item.setForeground(QColor('#17a2b8'))
        elif status in ['failed', 'qa_failed']:
            item.setForeground(QColor('red'))
        elif status == 'not_translated':
            item.setForeground(QColor('#2b6cb0'))
        elif status in ['not_refined', 'no_tts']:
            item.setForeground(QColor('#8a63d2'))
        elif status == 'in_progress':
            item.setForeground(QColor('orange'))
        else:
            item.setForeground(QColor('white'))

    def _set_progress_list_item_metadata(self, item, info, status, show_special_files):
        is_special = info.get('is_special', False)
        _fname = info.get('original_filename', '') or info.get('output_file', '') or info.get('key', '')
        is_skipped_special = self._progress_file_is_skipped_special(_fname, is_special)
        item.setData(Qt.UserRole, {
            'is_special': is_special,
            'info': info,
            'progress_key': info.get('progress_key'),
            'item_key': self._progress_list_item_key(info),
        })
        item.setData(Qt.UserRole + 2, status)
        item.setHidden(is_skipped_special and not show_special_files)

    def _populate_progress_listbox_streamed(self, data, chunk_size=150, preserve_selection=False, preserve_scroll=False):
        """Populate large progress lists over multiple event-loop turns."""
        if not self._is_data_valid(data):
            return

        listbox = data.get('listbox')
        if not listbox:
            return

        self._progress_list_sync_model_toggle(data)
        infos = list(data.get('chapter_display_info') or [])
        max_original_len, max_output_len = self._progress_list_column_widths(infos, data)

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

        try:
            listbox.blockSignals(True)
            listbox.setUpdatesEnabled(False)
            listbox.clear()
        except RuntimeError:
            data['_listbox_populate_active'] = False
            return

        state = {'idx': 0}

        def _finish():
            if generation != data.get('_listbox_populate_generation'):
                return
            data['_listbox_populate_active'] = False
            try:
                if saved_scroll is not None:
                    sb = listbox.verticalScrollBar()
                    sb.setValue(min(saved_scroll, sb.maximum()))
                listbox.blockSignals(False)
                listbox.setUpdatesEnabled(True)
                label = data.get('selection_count_label')
                if label:
                    label.setText(f"Selected: {len(listbox.selectedItems())}")
                listbox.viewport().update()
            except RuntimeError:
                pass

        def _add_chunk():
            if generation != data.get('_listbox_populate_generation'):
                return
            if not self._is_data_valid(data):
                return
            try:
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
                    item = QListWidgetItem(display)
                    self._apply_progress_list_item_visuals(item, status)
                    self._set_progress_list_item_metadata(item, info, status, show_special_files)
                    self._add_compact_inline_list_item(listbox, item)
                    self._set_progress_list_item_metadata(item, info, status, show_special_files)
                    if selected_keys and self._progress_list_item_key(info) in selected_keys:
                        item.setSelected(True)
                state['idx'] = end_idx
                listbox.setUpdatesEnabled(True)
                listbox.viewport().update()
            except RuntimeError:
                return

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

        show_special_files = self._progress_list_show_special(data)
        max_original_len, max_output_len = self._progress_list_column_widths(
            data.get('chapter_display_info') or [],
            data,
        )

        listbox.setUpdatesEnabled(False)
        listbox.blockSignals(True)
        try:
            for idx, info in enumerate(data.get('chapter_display_info') or []):
                if idx % 120 == 0:
                    self._ui_yield()
                item = listbox.item(idx)
                if not item:
                    continue
                display, display_status = self._progress_list_display_text(
                    info,
                    data,
                    max_original_len,
                    max_output_len,
                )
                item.setText(display)
                self._apply_progress_list_item_visuals(item, display_status)
                self._set_compact_inline_item_size(listbox, item)
                self._set_progress_list_item_metadata(item, info, display_status, show_special_files)
        finally:
            listbox.blockSignals(False)
            listbox.setUpdatesEnabled(True)

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
                        elif text.startswith('❌ Failed:'):
                            labels['failed'] = child
                    
                    # Recursively search children
                    labels.update(find_stats_labels(child))
            return labels
        
        stats_labels = find_stats_labels(container)
        
        if stats_labels:
            # Recalculate statistics from chapter_display_info (works for both OPF and non-OPF)
            chapter_display_info = data.get('chapter_display_info', [])
            pdf_rows = [info for info in chapter_display_info if info.get('pdf_ocr')]
            if pdf_rows and len(pdf_rows) == len(chapter_display_info):
                pdf_info = pdf_rows[0].get('info') or {}
                try:
                    total_chapters = int(pdf_info.get('total') or 0)
                    completed = min(int(pdf_info.get('done') or 0), total_chapters)
                    failed = int(pdf_info.get('failed') or 0)
                except (TypeError, ValueError):
                    total_chapters = len(chapter_display_info)
                    completed = 0
                    failed = 0
                merged = 0
                pending = 0
                status = self._progress_display_status(pdf_rows[0], data)
                in_progress = 1 if status == 'in_progress' else 0
                missing = max(0, total_chapters - completed - failed)
            else:
                total_chapters = len(chapter_display_info)
                display_statuses = [self._progress_display_status(info, data) for info in chapter_display_info]
                completed = sum(1 for status in display_statuses if status == 'completed')
                merged = sum(1 for status in display_statuses if status == 'merged')
                in_progress = sum(1 for status in display_statuses if status == 'in_progress')
                pending = sum(1 for status in display_statuses if status == 'pending')
                missing = sum(1 for status in display_statuses if status in ['not_translated', 'not_refined', 'no_tts'])
                failed = sum(1 for status in display_statuses if status in ['failed', 'qa_failed'])
            
            # Update labels
            if 'total' in stats_labels:
                stats_labels['total'].setText(f"Total: {total_chapters} | ")
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
            if 'missing' in stats_labels:
                mode = self._current_progress_output_mode(data)
                missing_label = "✨ Not Refined" if mode == 'refinement' else ("🔊 No TTS" if mode == 'audio' else "⬜ Not Translated")
                stats_labels['missing'].setText(f"{missing_label}: {missing} | ")
            if 'failed' in stats_labels:
                stats_labels['failed'].setText(f"❌ Failed: {failed} | ")

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
            
            def _normalize_output_file(output_file, output_dir):
                if not output_file:
                    return None
                # Normalize separators
                normalized = str(output_file).replace('\\', '/')
                # If absolute, try to store relative to output_dir when possible
                if os.path.isabs(normalized):
                    try:
                        rel = os.path.relpath(normalized, output_dir)
                        if not rel.startswith('..'):
                            return rel.replace('\\', '/')
                    except Exception:
                        pass
                    return normalized
                # If it's a relative path, keep as-is (preserve subfolders)
                return normalized
            
            # ALWAYS reload progress data from file to catch deletions
            progress_data = None
            html_files = []
            has_progress_tracking = os.path.exists(progress_file)
            
            if has_progress_tracking:
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                    print(f"🔄 Reloaded progress file from disk")
                    
                    # Extract files from progress data (primary source)
                    # Check if this is the newer nested structure with 'images' key
                    images_dict = progress_data.get('images', {})
                    if images_dict:
                        # Newer structure: progress_data['images'][hash] = {entry}
                        for key, value in images_dict.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                
                                # Only include if file actually exists on disk
                                if output_file and output_file not in html_files:
                                    full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)
                                    if os.path.exists(full_path):
                                        html_files.append(output_file)
                                    else:
                                        #print(f"⚠️ File in progress but not on disk: {output_file}")
                                        pass
                    else:
                        # Older structure: progress_data[hash] = {entry}
                        for key, value in progress_data.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                
                                # Only include if file actually exists on disk
                                if output_file and output_file not in html_files:
                                    full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)
                                    if os.path.exists(full_path):
                                        html_files.append(output_file)
                                    else:
                                        #print(f"⚠️ File in progress but not on disk: {output_file}")
                                        pass
                except Exception as e:
                    print(f"Failed to load progress file: {e}")
                    has_progress_tracking = False
            
            # Also scan directory for any HTML files not in progress (fallback)
            if os.path.exists(output_dir):
                try:
                    for file in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file)
                        if (os.path.isfile(file_path) and 
                            file.lower().endswith(('.html', '.xhtml', '.htm')) and 
                            file not in html_files):
                            html_files.append(file)
                except Exception as e:
                    print(f"Error scanning directory: {e}")
            
            # Rescan cover images
            image_files = []
            images_dir = os.path.join(output_dir, "images")
            if os.path.exists(images_dir):
                try:
                    for file in os.listdir(images_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            image_files.append(file)
                except Exception as e:
                    print(f"Error scanning images directory: {e}")
            
            # Rebuild file_info list
            file_info = []
            
            # Add translated files (both HTML and generated images)
            for html_file in sorted(set(html_files)):
                # Determine file type and extract info
                file_name = os.path.basename(html_file)
                is_html = file_name.lower().endswith(('.html', '.xhtml', '.htm'))
                is_image = file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))
                
                if is_html:
                    match = re.match(r'response_(\d+)_(.+)\.html', file_name)
                    if match:
                        index = match.group(1)
                        base_name = match.group(2)
                elif is_image:
                    # For generated images, just use the filename
                    base_name = os.path.splitext(file_name)[0]
                
                # Find hash key if progress tracking exists
                hash_key = None
                if progress_data:
                    # Check nested structure first
                    images_dict = progress_data.get('images', {})
                    if images_dict:
                        for key, value in images_dict.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                if output_file and output_file == html_file:
                                    hash_key = key
                                    break
                    else:
                        # Check flat structure
                        for key, value in progress_data.items():
                            if isinstance(value, dict) and 'output_file' in value:
                                output_file = _normalize_output_file(value['output_file'], output_dir)
                                if output_file and output_file == html_file:
                                    hash_key = key
                                    break
                
                file_info.append({
                    'type': 'translated',
                    'file': html_file,
                    'path': html_file if os.path.isabs(html_file) else os.path.join(output_dir, html_file),
                    'hash_key': hash_key,
                    'output_dir': output_dir
                })
            
            # Add cover images
            for img_file in sorted(image_files):
                file_info.append({
                    'type': 'cover',
                    'file': img_file,
                    'path': os.path.join(images_dir, img_file),
                    'hash_key': None,
                    'output_dir': output_dir
                })
            
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
                if info['type'] == 'translated':
                    file_name = os.path.basename(info['file'])
                    # Check if it's an HTML file or a generated image
                    is_html = file_name.lower().endswith(('.html', '.xhtml', '.htm'))
                    is_image = file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))
                    
                    if is_html:
                        match = re.match(r'response_(\d+)_(.+)\.html', file_name)
                        if match:
                            index = match.group(1)
                            base_name = match.group(2)
                            display = f"📄 Image {index} | {base_name} | ✅ Completed"
                        else:
                            display = f"📄 {file_name} | ✅ Completed"
                    elif is_image:
                        # Generated image file (e.g., Test1.png from imagen)
                        base_name = os.path.splitext(file_name)[0]
                        display = f"🖼️ {base_name} | ✅ Completed"
                    else:
                        display = f"📄 {file_name} | ✅ Completed"
                elif info['type'] == 'cover':
                    display = f"🖼️ Cover | {info['file']} | ⏭️ Skipped (cover)"
                else:
                    display = f"📄 {info['file']}"
                
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
                elif file_path.lower().endswith('.txt'):
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

                def _refresh_cached_tabs():
                    if not hasattr(cached_dialog, '_tab_data') or not cached_dialog._tab_data:
                        return
                    print(f"[DEBUG] Auto-clicking refresh on all {len(cached_dialog._tab_data)} tabs in cached dialog...")
                    for _td in cached_dialog._tab_data:
                        _rf = _td.get('refresh_func') if _td else None
                        if callable(_rf):
                            try:
                                _rf()
                            except Exception as _e:
                                print(f"[WARN] Auto-refresh failed for a tab: {_e}")

                QTimer.singleShot(50, _refresh_cached_tabs)
                return
            
            # If there's an existing dialog for a different selection, destroy it first
            if hasattr(self, '_multi_file_retranslation_dialog') and self._multi_file_retranslation_dialog:
                self._multi_file_retranslation_dialog.close()
                self._multi_file_retranslation_dialog.deleteLater()
                self._multi_file_retranslation_dialog = None
            
            # Create main dialog
            dialog = QDialog(self)
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
            
            # Track all tab data
            tab_data = []
            tabs_created = False
            
            # Store tab_data reference on the dialog for cross-tab operations
            dialog._tab_data = tab_data

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
                build_tasks.append(('epub_text', file_path, file_base, tab_frame))

            for folder_path in folders:
                build_tasks.append(('folder', folder_path, os.path.basename(folder_path) or folder_path, None))

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
                if tab_data:
                    print(f"[DEBUG] Auto-clicking refresh on all {len(tab_data)} tabs on dialog open...")
                    QTimer.singleShot(100, lambda: _refresh_tabs_streamed(0))
                else:
                    print(f"[WARN] No tab data to refresh on dialog open")

            def _build_next_tab():
                if build_state['idx'] >= len(build_tasks):
                    _finish_streamed_tabs()
                    return

                kind, path, label, tab_frame = build_tasks[build_state['idx']]
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
                            show_special_files_state=global_show_special
                        )
                        if tab_result:
                            cdi = tab_result.get('chapter_display_info', [])
                            completed = sum(1 for info in cdi if info.get('status') == 'completed')
                            in_progress = sum(1 for info in cdi if info.get('status') == 'in_progress')
                            tab_data.append(tab_result)
                            build_state['tabs_created'] = True
                            QTimer.singleShot(25, lambda td=tab_result: self._populate_progress_listbox_streamed(td))
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
        
        # Check multiple possible output folder patterns IN THE SCRIPT DIRECTORY
        possible_output_dirs = [
            os.path.join(script_dir, folder_name),  # Script dir + folder name (without extension)
            os.path.join(script_dir, f"{folder_name}_translated"),  # Script dir + folder_translated
            folder_name,  # Just the folder name in current directory
            f"{folder_name}_translated",  # folder_translated in current directory
        ]
        
        # Check for output directory override
        override_dir = os.environ.get('OUTPUT_DIRECTORY')
        if not override_dir and hasattr(self, 'config'):
            override_dir = self.config.get('output_directory')
            
        if override_dir:
            # If override is set, check inside it for the folder name
            possible_output_dirs.insert(0, os.path.join(override_dir, folder_name))
            possible_output_dirs.insert(1, os.path.join(override_dir, f"{folder_name}_translated"))
        
        output_dir = None
        for possible_dir in possible_output_dirs:
            print(f"Checking: {possible_dir}")
            if os.path.exists(possible_dir):
                # Check if it has translation_progress.json or HTML files
                if os.path.exists(os.path.join(possible_dir, "translation_progress.json")):
                    output_dir = possible_dir
                    print(f"Found output directory with progress tracker: {output_dir}")
                    break
                # Check if it has any HTML files
                elif os.path.isdir(possible_dir):
                    try:
                        files = os.listdir(possible_dir)
                        if any(f.lower().endswith(('.html', '.xhtml', '.htm')) for f in files):
                            output_dir = possible_dir
                            print(f"Found output directory with HTML files: {output_dir}")
                            break
                    except:
                        pass
        
        if not output_dir:
            self._styled_msgbox(QMessageBox.Information, self, "Info", 
                f"No translation output found for '{folder_name}'.\n\n"
                f"Selected folder: {folder_path}\n"
                f"Script directory: {script_dir}\n\n"
                f"Checked locations:\n" + "\n".join(f"- {d}" for d in possible_output_dirs))
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
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            moved_count = 0
            failed_count = 0
            
            for idx, item in items_to_move:
                try:
                    # Extract the original image name from the HTML filename
                    # Expected format: response_001_imagename.html (also accept compound extensions)
                    html_file = item['file']
                    html_base = os.path.basename(html_file)
                    match = re.match(r'^response_\d+_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', html_base, re.IGNORECASE)
                    
                    if match:
                        base_name = match.group(1)
                        # Try to find the original image with common extensions
                        original_found = False
                        
                        # Look for the source image in multiple locations
                        search_paths = [
                            folder_path,  # Original folder path
                            os.path.dirname(folder_path),  # Parent of folder path
                            os.getcwd(),  # Script directory
                        ]
                        
                        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            for search_path in search_paths:
                                if not search_path or not os.path.exists(search_path):
                                    continue
                                    
                                # Check in the search path
                                possible_source = os.path.join(search_path, base_name + ext)
                                if os.path.exists(possible_source) and os.path.isfile(possible_source):
                                    # Copy to images folder
                                    dest_path = os.path.join(images_dir, base_name + ext)
                                    if not os.path.exists(dest_path):
                                        import shutil
                                        shutil.copy2(possible_source, dest_path)
                                        print(f"Copied {base_name + ext} from {possible_source} to images folder")
                                    original_found = True
                                    break
                            if original_found:
                                break
                        
                        if not original_found:
                            print(f"Warning: Could not find original image for {html_file} in: {search_paths}")
                            # Even if source not found, we can still delete the HTML and mark it
                    
                    # Delete the HTML translation file
                    if os.path.exists(item['path']):
                        os.remove(item['path'])
                        print(f"Deleted translation: {item['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data_current and item.get('hash_key'):
                            hash_key = item['hash_key']
                            # Check nested structure first
                            if 'images' in progress_data_current and hash_key in progress_data_current['images']:
                                del progress_data_current['images'][hash_key]
                            # Check flat structure
                            elif hash_key in progress_data_current:
                                del progress_data_current[hash_key]
                    
                    # Update the listbox display
                    display = f"🖼️ Skipped | {base_name if match else html_base} | ⏭️ Moved to images folder"
                    listbox.item(idx).setText(display)
                    
                    # Update file_info
                    info_list[idx] = {
                        'type': 'cover',  # Treat as cover type since it's in images folder
                        'file': base_name + ext if match and original_found else html_base,
                        'path': os.path.join(images_dir, base_name + ext if match and original_found else html_base),
                        'hash_key': None,
                        'output_dir': output_dir
                    }
                    refresh_data['file_info'] = info_list
                    
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to process {item['file']}: {e}")
                    failed_count += 1
            
            # Save updated progress if modified
            if progress_data_current:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data_current, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            if 'refresh_data' in locals():
                self._refresh_image_folder_data(refresh_data)
            
            # Update selection count
            update_selection_count()
            
            # Show result
            if failed_count > 0:
                self._styled_msgbox(QMessageBox.Warning, dialog, "Partial Success", 
                    f"Moved {moved_count} image(s) to be skipped.\n"
                    f"Failed to process {failed_count} item(s).")
            else:
                self._styled_msgbox(QMessageBox.Information, dialog, "Success", 
                    f"Moved {moved_count} image(s) to the images folder.\n"
                    "They will be skipped in future translations.")
        
        def retranslate_selected():
            selected_items = listbox.selectedItems()
            if not selected_items:
                self._styled_msgbox(QMessageBox.Warning, dialog, "No Selection", "Please select at least one file.")
                return
            
            selected_indices = [listbox.row(item) for item in selected_items]
            info_list = refresh_data.get('file_info', file_info)
            progress_data_current = refresh_data.get('progress_data', progress_data)
            
            # Count types
            translated_count = sum(1 for i in selected_indices if info_list[i]['type'] == 'translated')
            cover_count = sum(1 for i in selected_indices if info_list[i]['type'] == 'cover')
            
            # Build confirmation message
            msg_parts = []
            if translated_count > 0:
                msg_parts.append(f"{translated_count} translated image(s)")
            if cover_count > 0:
                msg_parts.append(f"{cover_count} cover image(s)")
            
            confirm_msg = f"This will delete {' and '.join(msg_parts)}.\n\nContinue?"
            
            reply = self._styled_msgbox(QMessageBox.Question, dialog, "Confirm Deletion", confirm_msg,
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
            # Delete selected files
            deleted_count = 0
            
            for idx in selected_indices:
                info = info_list[idx]
                try:
                    if os.path.exists(info['path']):
                        os.remove(info['path'])
                        deleted_count += 1
                        print(f"Deleted: {info['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data_current and info.get('hash_key'):
                            hash_key = info['hash_key']
                            # Check nested structure first
                            if 'images' in progress_data_current and hash_key in progress_data_current['images']:
                                del progress_data_current['images'][hash_key]
                                print(f"Removed {hash_key} from progress_data['images']")
                            # Check flat structure
                            elif hash_key in progress_data_current:
                                del progress_data_current[hash_key]
                                print(f"Removed {hash_key} from progress_data")
                            
                except Exception as e:
                    print(f"Failed to delete {info['path']}: {e}")
            
            # ALWAYS save progress file after any deletions
            if deleted_count > 0 and progress_data_current:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data_current, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Auto-refresh the display to show updated status
            if 'refresh_data' in locals():
                self._refresh_image_folder_data(refresh_data)
            
            self._styled_msgbox(QMessageBox.Information, dialog, "Success", 
                f"Deleted {deleted_count} file(s).\n\n"
                "They will be retranslated on the next run.")
            
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
        _auto_refresh_timer.setInterval(2000)
        _auto_refresh_timer.timeout.connect(_silent_refresh_images)
        _auto_refresh_timer.start()
        dialog._auto_refresh_timer = _auto_refresh_timer
        
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
