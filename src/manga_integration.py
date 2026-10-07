# manga_integration.py
"""
Enhanced GUI Integration module for Manga Translation with text visibility controls
Now includes full page context mode with customizable prompt
"""
import sys
import os
import json
import threading
import time
import hashlib
import traceback
import concurrent.futures
import re
import shutil
from PySide6.QtWidgets import (QWidget, QLabel, QFrame, QPushButton, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QListWidget, QComboBox, QLineEdit, QCheckBox,
                               QRadioButton, QSlider, QSpinBox, QDoubleSpinBox, QTextEdit,
                               QProgressBar, QFileDialog, QMessageBox, QColorDialog, QScrollArea,
                               QDialog, QButtonGroup, QApplication, QSizePolicy, QToolButton,
                               QAbstractItemView)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QEvent, QPropertyAnimation, QEasingCurve, Property, QThread, QUrl
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QIcon, QKeyEvent, QPixmap, QTransform, QBrush, QDesktopServices
from typing import List, Dict, Optional, Any
from queue import Queue, Empty
import logging
from manga_translator import MangaTranslator, GOOGLE_CLOUD_VISION_AVAILABLE
from manga_settings_dialog import MangaSettingsDialog
from glossary_paths import get_book_glossary_dir, migrate_legacy_named_files
import ImageRenderer  # Import module-level methods
import manga_ocr_io
# U8: the GUI-free halves of this tab moved verbatim into shared modules; the tab inherits them
from manga_files_core import (MangaFilesMixin, MangaHooksMixin, _get_app_dir, _manga_cmd_debug_logging_enabled,
                              _manga_cmd_debug_print, _translation_run_token_matches, _natural_sort_key,
                              _MANGA_SKIP_PREFIX, _manga_filename_without_skip_prefix)
from manga_env import MangaEnvMixin, MangaOcrSessionMixin, apply_manga_startup_thread_limits
from manga_runner import (MangaRunMixin, _IS_WINDOWS, _lower_current_thread_priority_and_affinity,
                          _demote_non_main_threads)

# Optional: psutil/ctypes helpers to reduce GUI lag by lowering background thread priority
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import ctypes
import platform


# Natural/numerical sort helper function
# Try to import UnifiedClient for API initialization
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None


# U8: the per-image editor state (ImageStateManager + its spawn worker) moved to the GUI-free
# manga_editor_core, shared with the mobile editor (which runs it without the worker process).
from manga_editor_core import _state_manager_worker_process, ImageStateManager  # noqa: E402,F401



class _MangaGuiLogHandler(logging.Handler):
    """Forward logging records into MangaTranslationTab._log.
    Matches translator_gui.py's GuiLogHandler implementation.
    """
    def __init__(self, gui_ref, level=logging.INFO):
        super().__init__(level)
        self.gui_ref = gui_ref
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Use the raw message without logger name/level prefixes (matches translator_gui)
            msg = record.getMessage()
            
            # Call _log directly if it exists
            if hasattr(self.gui_ref, '_log'):
                self.gui_ref._log(msg, 'info')
        except Exception:
            # Never raise from logging path
            pass

class _StreamToGuiLog:
    """A minimal file-like stream that forwards lines to _log."""
    def __init__(self, write_cb):
        self._write_cb = write_cb
        self._buf = ''

    def write(self, s: str):
        try:
            self._buf += s
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                if line.strip():
                    self._write_cb(line)
        except Exception:
            pass

    def flush(self):
        try:
            if self._buf.strip():
                self._write_cb(self._buf)
            self._buf = ''
        except Exception:
            pass

class SavePositionWorker(QObject):
    """Worker for heavy save position rendering work"""
    
    finished = Signal(bool, int, str)  # success, region_index, rendered_path
    progress = Signal(str)  # progress message
    
    def __init__(self, manga_integration, region_index, trans_text, render_data):
        super().__init__()
        self.manga_integration = manga_integration
        self.region_index = region_index
        self.trans_text = trans_text
        self.render_data = render_data  # Contains all GUI data needed for rendering
    
    def run(self):
        """Run heavy rendering work in background thread"""
        try:
            # Use sys.__stdout__ to avoid print() redirection loops
            import sys
            sys.__stdout__.write(f"[WORKER] Starting heavy rendering work for region {self.region_index}\n")
            sys.__stdout__.flush()
            self.progress.emit("Rendering overlay...")
            
            # Extract render data
            current_image = self.render_data['current_image']
            regions = self.render_data['regions']
            base_image = self.render_data['base_image']
            output_path = self.render_data['output_path']
            
            # Call the heavy rendering method
            # This is the blocking operation that was freezing the GUI
            import ImageRenderer
            result = ImageRenderer._render_with_manga_translator_thread_safe(
                self.manga_integration, base_image, regions, output_path, current_image
            )
            
            # Find the rendered image path
            rendered_path = ""
            if result:
                current_image_path = current_image
                if current_image_path:
                    source_dir = os.path.dirname(current_image_path)
                    source_filename = os.path.basename(current_image_path)
                    
                    possible_paths = [
                        os.path.join(source_dir, "3_translated", source_filename),
                        os.path.join(source_dir, f"{os.path.splitext(source_filename)[0]}_translated", source_filename),
                        os.path.join(source_dir, f"{os.path.splitext(source_filename)[0]}_translated{os.path.splitext(source_filename)[1]}")
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            rendered_path = path
                            break
            
            self.finished.emit(True, self.region_index, rendered_path)
            
        except Exception as e:
            import sys
            sys.__stdout__.write(f"[WORKER] Rendering failed: {e}\n")
            sys.__stdout__.flush()
            import traceback
            traceback.print_exc()
            self.finished.emit(False, self.region_index, "")
    

class MangaTranslationTab(MangaRunMixin, MangaOcrSessionMixin, MangaEnvMixin, MangaFilesMixin, MangaHooksMixin, QObject):
    """GUI interface for manga translation integrated with TranslatorGUI"""
    
    # Signal for save position completion (emitted from background thread)
    save_position_completed = Signal(bool, int)  # success, region_index
    
    # Signals for OCR completion (emitted from background thread)
    ocr_result_signal = Signal(object, object, int, object, str)  # recognized_texts, rect_item, region_index, bbox, provider
    ocr_error_signal = Signal(object, object, int)  # error, rect_item, region_index
    
    # Class-level log storage to persist across window closures
    _persistent_log = []
    _persistent_log_lock = threading.RLock()
    
    # Class-level preload tracking to prevent duplicate loading
    _preload_in_progress = False
    _preload_lock = threading.RLock()
    _preload_completed_models = set()  # Track which models have been loaded
    
    def __init__(self, parent_widget, main_gui, dialog, scroll_area=None):
        """Initialize manga translation interface
        
        Args:
            parent_widget: The content widget for the interface (PySide6 QWidget)
            main_gui: Reference to TranslatorGUI instance
            dialog: The dialog window (PySide6 QDialog)
            scroll_area: The scroll area widget (PySide6 QScrollArea, optional)
        """
        apply_manga_startup_thread_limits(main_gui)  # moved to manga_env (U8)
        
        super().__init__()
        
        self.parent_widget = parent_widget
        self.main_gui = main_gui
        self.dialog = dialog
        self.scroll_area = scroll_area
        self._translator_gui_stylesheet = self._get_translator_gui_stylesheet()
        self._apply_translator_gui_stylesheet()
        
        # Initialize worker thread variables for save position
        self.worker_thread = None
        self.save_worker = None
        
        # Initialize auto-save progress tracking
        self._auto_save_in_progress = False
        
        # Shutdown flag to prevent spawning new processes during cleanup
        self._shutting_down = False
        
        # Record main GUI thread native id for selective demotion of background threads (Windows)
        try:
            self._main_thread_tid = threading.get_native_id()
        except Exception:
            self._main_thread_tid = None
        
        self._init_manga_run_state()  # moved to manga_env.MangaEnvMixin (U8)
        
        # Initialize image state manager for persistent overlays/rectangles
        state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.glossarion')
        state_file = os.path.join(state_dir, 'image_state.json')
        self.image_state_manager = ImageStateManager(state_file)
        
        # Track current image path for state persistence on navigation
        self._current_image_path = None
        
        # Auto-scroll control: delay forcing scroll on new runs
        self._autoscroll_delay_until = 0.0  # epoch seconds
        self._user_scrolled_up = False  # Track if user manually scrolled up
        self._log_autoscroll_pending = False
        self._log_auto_scroll_disabled = False
        
        # Flags for stdio redirection to avoid duplicate GUI logs
        self._stdout_redirect_on = False
        self._stderr_redirect_on = False
        self._stdio_redirect_active = False
        
        # Flag to prevent saving during initialization
        self._initializing = True
        
        # IMPORTANT: Load settings BEFORE building interface
        # This ensures all variables are initialized before they're used in the GUI
        self._load_rendering_settings()
        
        self._init_manga_prompt_state()  # moved to manga_env.MangaEnvMixin (U8)
        
        # flag to skip status checks during init
        self._initializing_gui = True
        
        # Build interface AFTER loading settings
        self._build_interface()

        # Now allow status checks
        self._initializing_gui = False
        
        # Do one status check after everything is built
        # Use QTimer for PySide6 dialog
        QTimer.singleShot(100, self._check_provider_status)
        
        # Add additional status refresh with longer delay to ensure correct status on startup
        # This fixes cases where bubble detection settings aren't reflected immediately
        QTimer.singleShot(500, self._check_provider_status)
        
        # DISABLED BY DEFAULT: Model preloading causes RAM spikes and 'event' errors
        # Users can enable via Advanced Settings > preload_local_models_on_open
        # QTimer.singleShot(200, self._start_model_preloading)
        
        # Now that everything is initialized, allow saving
        self._initializing = False
        
        # Preload shared models in background to avoid lag on first use
        # This spawns worker processes/loads models ahead of time so GUI stays responsive
        self._preload_complete = False
        self._inpainter_preload_failed = False
        self._inpainter_preload_error = ""
        self._inpainter_preload_failure_notified = False
        try:
            def _bg_preload_models():
                try:
                    # Check shutdown flag before preloading
                    if self._shutting_down:
                        return
                    # Preload bubble detector first (lighter)
                    ImageRenderer._preload_shared_bubble_detector(self)
                    # Check again before second preload
                    if self._shutting_down:
                        return
                    # Then preload inpainter (heavier, C++ worker process)
                    result = ImageRenderer._preload_shared_inpainter(self)
                    self._record_inpainter_preload_result(result, "startup")
                except Exception as e:
                    print(f"[INIT_PRELOAD] Background model preload failed: {e}")
                    self._record_inpainter_preload_result(False, "startup", str(e))
                finally:
                    # Mark preload complete regardless of success/failure
                    self._preload_complete = True
            
            self._preload_thread = threading.Thread(target=_bg_preload_models, daemon=True)
            self._preload_thread.start()
            
            # Start monitoring preload status to update button states
            QTimer.singleShot(100, self._check_preload_status)
        except Exception:
            self._preload_complete = True  # Mark complete on error so buttons aren't stuck
        
        # Circle mode for selections and pipeline masks (rect by default)
        self._use_circle_shapes = False
        
        # Load persisted manga image list from previous session
        QTimer.singleShot(200, self._load_persisted_files)
        
        # Attach logging bridge so library logs appear in our log area
        self._attach_logging_bridge()
        
        # Connect OCR signals to handlers
        self.ocr_result_signal.connect(lambda recognized_texts, rect_item, region_index, bbox, ocr_provider: ImageRenderer._process_ocr_result(self, recognized_texts, rect_item, region_index, bbox, ocr_provider))
        self.ocr_error_signal.connect(lambda error, rect_item, region_index: ImageRenderer._handle_ocr_error(self, error, rect_item, region_index))

        # Start update loop
        self._process_updates()
        
        # Install event filter for F11 fullscreen toggle
        self._install_fullscreen_handler()
        
        # Connect dialog close event to cleanup
        if self.dialog:
            self.dialog.finished.connect(self.cleanup)

    def _get_translator_gui_stylesheet(self) -> str:
        """Return the parent TranslatorGUI stylesheet, with QApplication as fallback."""
        try:
            if self.main_gui is not None and hasattr(self.main_gui, 'styleSheet'):
                style = self.main_gui.styleSheet()
                if style:
                    return style
        except Exception:
            pass
        try:
            app = QApplication.instance()
            if app is not None:
                return app.styleSheet() or ""
        except Exception:
            pass
        return ""

    def _apply_translator_gui_stylesheet(self):
        """Apply parent GUI styling to this top-level manga dialog and containers."""
        style = getattr(self, '_translator_gui_stylesheet', '') or ''
        if not style:
            return
        for widget in (self.dialog, self.scroll_area):
            try:
                if widget is not None and not widget.styleSheet():
                    widget.setStyleSheet(style)
            except Exception:
                pass
    
    def cleanup(self, fast: bool = False):
        """Cleanup method called when dialog closes - prevents RAM leaks"""
        try:
            if getattr(self, '_cleanup_done', False):
                print("[CLEANUP] MangaTranslationTab cleanup already completed")
                return
            self._cleanup_done = True

            mode = "fast" if fast else "full"
            print(f"[CLEANUP] Starting MangaTranslationTab cleanup ({mode})...")
            
            # Set shutdown flag to prevent new processes from spawning
            self._shutting_down = True
            
            # Stop pool update timer
            if hasattr(self, 'pool_update_timer') and self.pool_update_timer:
                print("[CLEANUP] Stopping pool update timer...")
                self.pool_update_timer.stop()
                self.pool_update_timer = None
            
            # Shutdown parallel save system and its ThreadPoolExecutor
            if hasattr(self, '_parallel_save_system') and self._parallel_save_system:
                print("[CLEANUP] Shutting down parallel save system...")
                try:
                    self._parallel_save_system.shutdown(wait=not fast)
                except TypeError:
                    self._parallel_save_system.shutdown()
                self._parallel_save_system = None
            
            # Flush image state manager and stop worker process
            if hasattr(self, 'image_state_manager'):
                print("[CLEANUP] Cancelling any pending save timers...")
                # Cancel any pending save timer first
                with self.image_state_manager._timer_lock:
                    if self.image_state_manager._save_timer is not None:
                        self.image_state_manager._save_timer.cancel()
                        self.image_state_manager._save_timer = None
                print("[CLEANUP] Flushing image state manager...")
                self.image_state_manager.flush()
                print("[CLEANUP] Stopping state manager worker process...")
                self.image_state_manager._stop_worker()
                print("[CLEANUP] State manager worker stopped")
            
            if not fast:
                # Full dialog cleanup can collect cycles; app shutdown exits the
                # process immediately, so this only makes closing slower there.
                import gc
                gc.collect()
            print("[CLEANUP] Cleanup completed")
            
        except Exception as e:
            print(f"[CLEANUP] Error during cleanup: {e}")

    def set_use_circle_shapes(self, enabled: bool):
        """Enable/disable circle mode for ROI shapes across detect/clean/recognize/translate."""
        try:
            self._use_circle_shapes = bool(enabled)
            # Optionally refresh drawing style later if needed
        except Exception:
            self._use_circle_shapes = False
        
        # Reset MangaTranslator class flag
        try:
            from manga_translator import MangaTranslator
            MangaTranslator.set_global_cancellation(False)
        except ImportError:
            pass
            
        # Reset UnifiedClient flag
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient.set_global_cancellation(False)
        except ImportError:
            pass
    
    def _is_local_inpainting_enabled(self) -> bool:
        """Check if local inpainting is enabled (not skipped and method is 'local')."""
        skip_inpainting = self.main_gui.config.get('manga_skip_inpainting', False)
        if skip_inpainting:
            return False
        inpaint_settings = self.main_gui.config.get('manga_settings', {}).get('inpainting', {})
        inpainting_method = inpaint_settings.get('method', 'local')
        return inpainting_method == 'local'

    def _record_inpainter_preload_result(self, result, source: str = "preload", error: str = ""):
        """Record whether a local inpainter preload actually produced a usable model."""
        try:
            if result is False:
                self._inpainter_preload_failed = True
                self._inpainter_preload_error = error or f"Local inpainter preload failed during {source}"
                self._inpainter_preload_failure_notified = False
            elif result is True:
                self._clear_inpainter_preload_failure()
        except Exception:
            pass

    def _clear_inpainter_preload_failure(self):
        """Clear the remembered failed-preload state."""
        self._inpainter_preload_failed = False
        self._inpainter_preload_error = ""
        self._inpainter_preload_failure_notified = False

    def _set_translation_buttons_model_failed(self):
        """Disable inpainting-dependent actions after a local inpainter load failure."""
        try:
            self._waiting_for_model = False
            if not getattr(self, '_inpainter_preload_failure_notified', False):
                msg = getattr(self, '_inpainter_preload_error', '') or "Local inpainter failed to load"
                self._log(f"Local inpainter preload failed: {msg}", "error")
                self._log("Load or download a valid inpainting model, or enable Skip Inpainter to continue without inpainting.", "warning")
                self._inpainter_preload_failure_notified = True

            if hasattr(self, 'local_model_status_label') and self.local_model_status_label:
                self.local_model_status_label.setText("Local inpainter failed to load")
                self.local_model_status_label.setStyleSheet("color: orange;")

            if hasattr(self, 'start_button') and self.start_button:
                self.start_button.setEnabled(False)
                if hasattr(self, 'start_button_text'):
                    self.start_button_text.setText("Inpainter failed")

            if hasattr(self, 'image_preview_widget'):
                ipw = self.image_preview_widget
                for attr, text in (
                    ('translate_btn', 'Inpainter failed'),
                    ('translate_all_btn', 'Inpainter failed'),
                    ('clean_btn', 'Inpainter failed'),
                ):
                    btn = getattr(ipw, attr, None)
                    if btn:
                        btn.setEnabled(False)
                        btn.setText(text)
        except Exception as e:
            print(f"[BUTTON_STATE] Error setting inpainter failure state: {e}")
    
    def _check_preload_status(self):
        """Check preload thread status and update button states accordingly.

        Detects an in-flight preload from ANY source so the
        Translate / Translate All / Clean / Start Translation buttons show
        the waiting state instead of letting the user kick off a translation
        that immediately stalls in `_initialize_local_inpainter` with
        repeated '⏳ Still waiting for inpainter pool...' log lines.
        """
        try:
            # Init-driven preload (started in __init__)
            init_preload = getattr(self, '_preload_thread', None)
            init_running = bool(init_preload and init_preload.is_alive())

            # Toggle-off driven preload (started by _trigger_inpainter_preload_after_toggle)
            toggle_preload = getattr(self, '_toggle_preload_thread', None)
            toggle_running = bool(toggle_preload and toggle_preload.is_alive())

            # Manual local model loads started by the Local Model combo / Load button
            # do not use the startup preload thread, but they still populate the
            # same inpainter pool. Treat them as a preload wait state too.
            manual_loading = bool(getattr(self, '_model_loading_in_progress', False))

            preload_running = init_running or toggle_running or manual_loading
            local_inpainting = self._is_local_inpainting_enabled()
            custom_image_edit_inpainting = (
                local_inpainting
                and str(self.main_gui.config.get('manga_local_inpaint_model', '') or '').lower() == 'custom-image-edit'
            )
            should_wait = preload_running and local_inpainting and not custom_image_edit_inpainting

            if should_wait:
                # Disable buttons and show "Waiting..." state
                self._set_translation_buttons_waiting(True)
                # Check again in 500ms
                QTimer.singleShot(500, self._check_preload_status)
            elif custom_image_edit_inpainting:
                self._clear_inpainter_preload_failure()
                self._set_translation_buttons_waiting(False)
            elif local_inpainting and bool(getattr(self, '_inpainter_preload_failed', False)):
                self._set_translation_buttons_model_failed()
            else:
                # Preload complete or not needed - enable buttons
                if not local_inpainting:
                    self._clear_inpainter_preload_failure()
                self._set_translation_buttons_waiting(False)
        except Exception as e:
            print(f"[PRELOAD_CHECK] Error: {e}")
            # On error, enable buttons to avoid being stuck
            self._set_translation_buttons_waiting(False)
    
    def _set_translation_buttons_waiting(self, waiting: bool):
        """Set translation buttons to waiting or ready state.
        
        Args:
            waiting: If True, disable buttons and show 'Waiting...' text.
                    If False, enable buttons and restore normal text.
        """
        try:
            # Store waiting state for context menu checks
            self._waiting_for_model = waiting
            
            # If a workflow operation (OCR, clean, translate) is actively running,
            # handle model state transitions without touching button text/enabled state.
            if getattr(self, '_workflow_operation_active', False):
                if waiting:
                    # Model still loading — hide stop button (stop kills the loader).
                    if hasattr(self, 'image_preview_widget'):
                        ipw = self.image_preview_widget
                        if hasattr(ipw, 'stop_translation_btn'):
                            ipw.stop_translation_btn.setVisible(False)
                else:
                    # Model just finished loading — show stop button (now safe).
                    if hasattr(self, 'image_preview_widget'):
                        ipw = self.image_preview_widget
                        if hasattr(ipw, 'stop_translation_btn'):
                            ipw.stop_translation_btn.setVisible(True)
                            ipw.stop_translation_btn.setEnabled(True)
                            ipw.stop_translation_btn.setText("\u23f9 Stop")
                # Don't touch button text/enabled — the operation owns them.
                return
            
            # No operation active — hide stop during model loading.
            if waiting and hasattr(self, 'image_preview_widget'):
                ipw = self.image_preview_widget
                if hasattr(ipw, 'stop_translation_btn'):
                    ipw.stop_translation_btn.setVisible(False)
            
            # Start Translation button
            if hasattr(self, 'start_button') and self.start_button:
                if waiting:
                    self.start_button.setEnabled(False)
                    if hasattr(self, 'start_button_text'):
                        self.start_button_text.setText("⏳ Waiting for model...")
                    self.start_button.setStyleSheet(
                        "QPushButton { "
                        "  background-color: #6c757d; "
                        "  color: white; "
                        "  padding: 28px 30px; "
                        "  font-size: 14pt; "
                        "  font-weight: bold; "
                        "  border-radius: 8px; "
                        "} "
                        "QPushButton:disabled { "
                        "  background-color: #6c757d; "
                        "  color: #cccccc; "
                        "}"
                    )
                else:
                    # Only enable if setup is ready (has API key, etc.)
                    is_ready = self._check_translation_ready()
                    self.start_button.setEnabled(is_ready)
                    if hasattr(self, 'start_button_text'):
                        self.start_button_text.setText("▶ Start Translation")
                    self.start_button.setStyleSheet(
                        "QPushButton { "
                        "  background-color: #28a745; "
                        "  color: white; "
                        "  padding: 28px 30px; "
                        "  font-size: 14pt; "
                        "  font-weight: bold; "
                        "  border-radius: 8px; "
                        "} "
                        "QPushButton:hover { background-color: #218838; } "
                        "QPushButton:disabled { "
                        "  background-color: #2d2d2d; "
                        "  color: #666666; "
                        "}"
                    )
            
            # Translate, Translate All, and Clean buttons in image preview
            if hasattr(self, 'image_preview_widget'):
                ipw = self.image_preview_widget
                
                if hasattr(ipw, 'translate_btn') and ipw.translate_btn:
                    if waiting:
                        ipw.translate_btn.setEnabled(False)
                        ipw.translate_btn.setText("⏳ Waiting...")
                    else:
                        ipw.translate_btn.setEnabled(True)
                        ipw.translate_btn.setText("Translate")
                
                if hasattr(ipw, 'translate_all_btn') and ipw.translate_all_btn:
                    if waiting:
                        ipw.translate_all_btn.setEnabled(False)
                        ipw.translate_all_btn.setText("⏳ Waiting...")
                    else:
                        ipw.translate_all_btn.setEnabled(True)
                        ipw.translate_all_btn.setText("Translate All")
                
                # Clean button in image preview
                if hasattr(ipw, 'clean_btn') and ipw.clean_btn:
                    if waiting:
                        ipw.clean_btn.setEnabled(False)
                        ipw.clean_btn.setText("⏳ Waiting...")
                    else:
                        ipw.clean_btn.setEnabled(True)
                        ipw.clean_btn.setText("Clean")
                
        except Exception as e:
            print(f"[BUTTON_STATE] Error setting button state: {e}")
    
    def _check_translation_ready(self) -> bool:
        """Check if translation prerequisites are met (API key, credentials, etc.)."""
        try:
            # Check API key
            if hasattr(self.main_gui, 'api_key_entry'):
                if hasattr(self.main_gui.api_key_entry, 'text'):
                    has_api_key = bool(self.main_gui.api_key_entry.text().strip())
                elif hasattr(self.main_gui.api_key_entry, 'get'):
                    has_api_key = bool(self.main_gui.api_key_entry.get().strip())
                else:
                    has_api_key = False
            else:
                has_api_key = False
            
            # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
            if not has_api_key:
                _model = self.main_gui.config.get('model', '') if hasattr(self, 'main_gui') else ''
                try:
                    from unified_api_client import UnifiedClient
                    if not UnifiedClient._model_needs_api_key(_model):
                        has_api_key = True
                except Exception:
                    pass
            
            if not has_api_key:
                return False
            
            # Check provider-specific credentials
            saved_provider = self.main_gui.config.get('manga_ocr_provider', 'custom-api')
            if saved_provider == 'custom-api':
                has_api_key = True
            if saved_provider == 'google':
                google_path = self.main_gui.config.get('google_vision_credentials', '')
                ok, _ = self._validate_google_credentials(google_path)
                return ok
            elif saved_provider == 'azure':
                return bool(self.main_gui.config.get('azure_vision_key', ''))
            else:
                return True  # Local providers only need API key
        except Exception:
            return True  # Default to ready on error
    
    def _install_fullscreen_handler(self):
        """Install event filter to handle F11 key for fullscreen toggle"""
        if not self.dialog:
            return
        
        # Create event filter for the dialog
        class FullscreenEventFilter(QObject):
            def __init__(self, dialog_ref):
                super().__init__()
                self.dialog = dialog_ref
                self.is_fullscreen = False
                self.normal_geometry = None
            
            def eventFilter(self, obj, event):
                if event.type() == QEvent.KeyPress:
                    key_event = event
                    if key_event.key() == Qt.Key_F11:
                        self.toggle_fullscreen()
                        return True
                return False
            
            def toggle_fullscreen(self):
                if self.is_fullscreen:
                    # Exit fullscreen
                    self.dialog.setWindowState(self.dialog.windowState() & ~Qt.WindowFullScreen)
                    if self.normal_geometry:
                        self.dialog.setGeometry(self.normal_geometry)
                    self.is_fullscreen = False
                else:
                    # Enter fullscreen
                    self.normal_geometry = self.dialog.geometry()
                    self.dialog.setWindowState(self.dialog.windowState() | Qt.WindowFullScreen)
                    self.is_fullscreen = True
        
        # Create and install the event filter
        self._fullscreen_filter = FullscreenEventFilter(self.dialog)
        self.dialog.installEventFilter(self._fullscreen_filter)
    
    def _start_model_preloading(self):
        """Start preloading models in background thread for efficient loading"""
        import queue
        
        # Check if preload is already in progress
        with MangaTranslationTab._preload_lock:
            if MangaTranslationTab._preload_in_progress:
                print("Model preloading already in progress, skipping...")
                return
        
        # Get settings
        manga_settings = self.main_gui.config.get('manga_settings', {})
        ocr_settings = manga_settings.get('ocr', {})
        inpaint_settings = manga_settings.get('inpainting', {})
        
        models_to_load = []
        bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', True)
        skip_inpainting = self.main_gui.config.get('manga_skip_inpainting', False)
        inpainting_method = inpaint_settings.get('method', 'local')
        inpainting_enabled = not skip_inpainting and inpainting_method == 'local'
        
        # Check if models need loading
        try:
            from manga_translator import MangaTranslator
            
            if bubble_detection_enabled:
                detector_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
                model_url = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
                onnx_filename = ocr_settings.get('rtdetr_onnx_variant', 'detector.onnx')
                # Sanitize model_url to avoid unrelated JSON paths
                try:
                    if model_url and model_url.lower().endswith('.json'):
                        model_url = ''
                except Exception:
                    pass
                # If no valid model reference, skip preloading (do not auto-download)
                if detector_type in ('rtdetr', 'rtdetr_onnx') and not model_url:
                    model_url = ''
                # If no valid model reference, skip preloading
                if detector_type in ('rtdetr', 'rtdetr_onnx') and not model_url:
                    pass
                else:
                    key = (detector_type, model_url, onnx_filename) if detector_type == 'rtdetr_onnx' else (detector_type, model_url)
                    model_id = f"detector_{detector_type}_{model_url}_{onnx_filename}" if detector_type == 'rtdetr_onnx' else f"detector_{detector_type}_{model_url}"
                
                # Skip if already loaded in this session
                if detector_type in ('rtdetr', 'rtdetr_onnx') and not model_url:
                    pass
                else:
                    if model_id not in MangaTranslationTab._preload_completed_models:
                        with MangaTranslator._detector_pool_lock:
                            rec = MangaTranslator._detector_pool.get(key)
                            if not rec or not rec.get('spares'):
                                detector_name = 'RT-DETR ONNX' if detector_type == 'rtdetr_onnx' else 'RT-DETR' if detector_type == 'rtdetr' else 'YOLO'
                                models_to_load.append(('detector', detector_type, detector_name, model_url, onnx_filename))
            
            if inpainting_enabled:
                # Check top-level config first (manga_local_inpaint_model), then nested config
                local_method = self.main_gui.config.get('manga_local_inpaint_model', 
                                                        inpaint_settings.get('local_method', 'anime_onnx'))
                model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')
                # Fallback to non-prefixed key if not found
                if not model_path:
                    model_path = self.main_gui.config.get(f'{local_method}_model_path', '')
                key = (local_method, model_path or '')
                model_id = f"inpainter_{local_method}_{model_path}"
                
                # Skip if already loaded in this session
                if model_id not in MangaTranslationTab._preload_completed_models:
                    with MangaTranslator._inpaint_pool_lock:
                        rec = MangaTranslator._inpaint_pool.get(key)
                        if not rec or not rec.get('spares'):
                            models_to_load.append(('inpainter', local_method, local_method.capitalize(), model_path, None))
        except Exception as e:
            print(f"Error checking models: {e}")
            return
        
        if not models_to_load:
            return
        
        # Set preload in progress flag
        with MangaTranslationTab._preload_lock:
            MangaTranslationTab._preload_in_progress = True
        
        # Store models being loaded for tracking
        models_being_loaded = []
        for model_type, model_key, model_name, model_path, onnx_filename in models_to_load:
            if model_type == 'detector':
                suffix = f"_{onnx_filename}" if model_key == 'rtdetr_onnx' and onnx_filename else ""
                models_being_loaded.append(f"detector_{model_key}_{model_path}{suffix}")
            elif model_type == 'inpainter':
                models_being_loaded.append(f"inpainter_{model_key}_{model_path}")
        
        print(f"🔄 Preloading {len(models_to_load)} model(s) in background...")
        
        # Load models directly in background thread (no multiprocessing overhead)
        def load_models_bg():
            try:
                import os
                from manga_translator import MangaTranslator
                from bubble_detector import BubbleDetector
                from local_inpainter import LocalInpainter
                
                loaded_count = 0
                
                for model_type, model_key, model_name, model_path, onnx_filename in models_to_load:
                    try:
                        
                        # Guard: ignore JSON paths (credentials) for inpainter models
                        try:
                            if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                                model_path = ''
                        except Exception:
                            pass
                        
                        if model_type == 'detector':
                            key = (model_key, model_path or '', onnx_filename or 'detector.onnx') if model_key == 'rtdetr_onnx' else (model_key, model_path or '')
                            
                            # Check if already in pool
                            with MangaTranslator._detector_pool_lock:
                                rec = MangaTranslator._detector_pool.get(key)
                                if rec and rec.get('spares'):
                                    print(f"  ⏭️ {model_name} already in pool")
                                    loaded_count += 1
                                    continue
                            
                            # Load into pool
                            print(f"  🔄 Loading {model_name}...")
                            bd = BubbleDetector()
                            if model_key == 'rtdetr_onnx':
                                model_repo = model_path if model_path else 'ogkalu/comic-text-and-bubble-detector'
                                bd.load_rtdetr_onnx_model(model_repo, onnx_filename=onnx_filename or 'detector.onnx')
                            elif model_key == 'rtdetr':
                                bd.load_rtdetr_model()
                            elif model_key == 'yolo':
                                if model_path:
                                    bd.load_model(model_path)
                            
                            # Store in pool
                            with MangaTranslator._detector_pool_lock:
                                rec = MangaTranslator._detector_pool.get(key)
                                if not rec:
                                    rec = {'spares': [], 'checked_out': []}
                                    MangaTranslator._detector_pool[key] = rec
                                if 'checked_out' not in rec:
                                    rec['checked_out'] = []
                                rec['spares'].append(bd)
                            print(f"  ✓ {model_name} loaded")
                            loaded_count += 1
                            
                        elif model_type == 'inpainter':
                            key = (model_key, model_path or '')
                            
                            # Check if already in pool
                            with MangaTranslator._inpaint_pool_lock:
                                rec = MangaTranslator._inpaint_pool.get(key)
                                if rec and rec.get('spares'):
                                    print(f"  ⏭️ {model_name} already in pool")
                                    loaded_count += 1
                                    continue
                            
                            # Load into pool
                            print(f"  🔄 Loading {model_name}...")
                            # Get worker process setting from config
                            # Note: During preload we always disable workers to save RAM
                            # since workers would load duplicate model copies
                            inp = LocalInpainter(enable_worker_process=False)

                            if str(model_key or '').lower() == 'custom-image-edit':
                                success = inp.load_model('custom-image-edit', model_path or '', force_reload=False)
                                if success and getattr(inp, 'model_loaded', False):
                                    pool_key = (model_key, model_path or '__default_image_edit_endpoint__')
                                    with MangaTranslator._inpaint_pool_lock:
                                        rec = MangaTranslator._inpaint_pool.get(pool_key)
                                        if not rec:
                                            rec = {'spares': [], 'checked_out': [], 'loaded': True}
                                            MangaTranslator._inpaint_pool[pool_key] = rec
                                        if 'checked_out' not in rec:
                                            rec['checked_out'] = []
                                        rec['spares'].append(inp)
                                    print(f"  OK {model_name} endpoint ready")
                                    loaded_count += 1
                                else:
                                    print(f"  Failed to configure {model_name}")
                                continue
                            
                            # Ensure model file exists or download it
                            resolved_path = model_path
                            if not resolved_path or not os.path.exists(resolved_path):
                                try:
                                    resolved_path = inp.download_jit_model(model_key)
                                except:
                                    resolved_path = None
                            
                            if resolved_path and os.path.exists(resolved_path):
                                success = inp.load_model_with_retry(model_key, resolved_path)
                                if success:
                                    # Store in pool
                                    with MangaTranslator._inpaint_pool_lock:
                                        rec = MangaTranslator._inpaint_pool.get(key)
                                        if not rec:
                                            rec = {'spares': [], 'checked_out': []}
                                            MangaTranslator._inpaint_pool[key] = rec
                                        if 'checked_out' not in rec:
                                            rec['checked_out'] = []
                                        rec['spares'].append(inp)
                                    print(f"  ✓ {model_name} loaded")
                                    loaded_count += 1
                                else:
                                    print(f"  ✗ Failed to load {model_name}")
                                    
                    except Exception as e:
                        print(f"  ✗ Error loading {model_name}: {e}")
                
                # Mark completion
                if loaded_count > 0:
                    print(f"✅ Preloaded {loaded_count} model(s) into memory")
                else:
                    print("⚠️ No models were successfully preloaded")
                
                # Mark all models as completed
                with MangaTranslationTab._preload_lock:
                    MangaTranslationTab._preload_completed_models.update(models_being_loaded)
                    MangaTranslationTab._preload_in_progress = False
                
            except Exception as e:
                print(f"✗ Error during model preloading: {e}")
                
                # Reset flag on error
                with MangaTranslationTab._preload_lock:
                    MangaTranslationTab._preload_in_progress = False
        
        # Start loading in background thread (non-daemon to allow child processes)
        import threading
        loading_thread = threading.Thread(target=load_models_bg, daemon=False)
        loading_thread.start()
    
    def _disable_spinbox_mousewheel(self, spinbox):
        """Disable mousewheel scrolling on a spinbox (PySide6)"""
        # Override wheelEvent to prevent scrolling
        spinbox.wheelEvent = lambda event: None
    
    def _disable_combobox_mousewheel(self, combobox):
        """Disable mousewheel scrolling on a combobox (PySide6)"""
        # Override wheelEvent to prevent scrolling
        combobox.wheelEvent = lambda event: None
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay"""
        from PySide6.QtWidgets import QCheckBox, QLabel
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QFont
        
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet("""
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
        
        # Create checkmark overlay
        checkmark = QLabel("\u2713", checkbox)
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
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)  # Make checkmark click-through
        
        # Position checkmark properly after widget is shown
        def position_checkmark():
            # Position over the checkbox indicator
            checkmark.setGeometry(2, 1, 14, 14)
        
        # Show/hide checkmark based on checked state
        def update_checkmark():
            if checkbox.isChecked():
                position_checkmark()
                checkmark.show()
                checkmark.raise_()
                checkmark.update()
            else:
                checkmark.hide()
        
        # Expose helpers for programmatic refresh (e.g., when signals are blocked)
        try:
            checkbox._checkmark_label = checkmark
            checkbox._position_checkmark = position_checkmark
            checkbox._update_checkmark = update_checkmark
        except Exception:
            pass
        
        checkbox.stateChanged.connect(update_checkmark)
        # Delay initial positioning to ensure widget is properly rendered
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark()))
        
        return checkbox

    def _download_hf_model(self):
        """Download HuggingFace models with progress tracking - PySide6 version"""
        from PySide6.QtCore import QTimer
        
        def check_download_status():
            # If there's still a download in progress (button disabled), check again in 500ms
            download_btn = getattr(self, 'ocr_download_model_btn', None)
            if download_btn and download_btn.isEnabled() == False:
                QTimer.singleShot(500, check_download_status)
                return
            
            # Re-enable Download Model button once inpainter shows up as loaded
            if download_btn:
                download_btn.setEnabled(True)
            self._check_provider_status()
        
        try:
            # Disable the button during download
            download_btn = getattr(self, 'ocr_download_model_btn', None)
            if download_btn:
                download_btn.setEnabled(False)
            # Schedule status check
            QTimer.singleShot(500, check_download_status)
            
            provider = self.ocr_provider_value
            
            # Initialize OCR manager if needed
            self._ensure_ocr_manager()
            
            # Model sizes (approximate in MB)
            model_sizes = {
                'manga-ocr': 450,
                'Qwen2-VL': {
                    '2B': 4000,
                    '7B': 14000,
                    '72B': 144000,
                    'custom': 10000  # Default estimate for custom models
                }
            }
            
            # Create download dialog using PySide6
            # ... rest of your original download dialog code ...
        except Exception as e:
            self._log(f"Failed to initialize download: {e}", "error")
            download_btn = getattr(self, 'ocr_download_model_btn', None)
            if download_btn:
                download_btn.setEnabled(True)
        """Download HuggingFace models with progress tracking - PySide6 version"""
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                        QRadioButton, QButtonGroup, QLineEdit, QPushButton,
                                        QGroupBox, QTextEdit, QProgressBar, QFrame,
                                        QScrollArea, QWidget, QSizePolicy, QApplication)
        from PySide6.QtCore import Qt, QThread, Signal, QTimer
        from PySide6.QtGui import QFont
        
        provider = self.ocr_provider_value
        
        # Model sizes (approximate in MB)
        model_sizes = {
            'manga-ocr': 450,
            'Qwen2-VL': {
                '2B': 4000,
                '7B': 14000,
                '72B': 144000,
                'custom': 10000  # Default estimate for custom models
            }
        }
        
        # For Qwen2-VL, show model selection dialog first
        if provider == 'Qwen2-VL':
            # Create PySide6 dialog
            selection_dialog = QDialog(self.dialog)
            selection_dialog.setWindowTitle("Select Qwen2-VL Model Size")
            # Use screen ratios for sizing
            screen = QApplication.primaryScreen().geometry()
            width = int(screen.width() * 0.31)  # 31% of screen width
            height = int(screen.height() * 0.46)  # 46% of screen height
            selection_dialog.setMinimumSize(width, height)
            main_layout = QVBoxLayout(selection_dialog)
            
            # Title
            title_label = QLabel("Select Qwen2-VL Model Size")
            title_font = QFont("Arial", 14, QFont.Weight.Bold)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(title_label)
            
            # Model selection frame
            model_frame = QGroupBox("Model Options")
            model_frame_font = QFont("Arial", 11, QFont.Weight.Bold)
            model_frame.setFont(model_frame_font)
            model_frame_layout = QVBoxLayout(model_frame)
            model_frame_layout.setContentsMargins(15, 15, 15, 15)
            model_frame_layout.setSpacing(10)
            
            model_options = {
                "2B": {
                    "title": "2B Model",
                    "desc": "• Smallest model (~4GB download, 4-8GB VRAM)\n• Fast but less accurate\n• Good for quick testing"
                },
                "7B": {
                    "title": "7B Model", 
                    "desc": "• Medium model (~14GB download, 12-16GB VRAM)\n• Best balance of speed and quality\n• Recommended for most users"
                },
                "72B": {
                    "title": "72B Model",
                    "desc": "• Largest model (~144GB download, 80GB+ VRAM)\n• Highest quality but very slow\n• Requires high-end GPU"
                },
                "custom": {
                    "title": "Custom Model",
                    "desc": "• Enter any Hugging Face model ID\n• For advanced users\n• Size varies by model"
                }
            }
            
            # Store selected model
            selected_model_key = {"value": "2B"}
            custom_model_id_text = {"value": ""}
            
            # Radio button group
            button_group = QButtonGroup(selection_dialog)
            
            for idx, (key, info) in enumerate(model_options.items()):
                # Radio button
                rb = QRadioButton(info["title"])
                rb_font = QFont("Arial", 11, QFont.Weight.Bold)
                rb.setFont(rb_font)
                if idx == 0:
                    rb.setChecked(True)
                rb.clicked.connect(lambda checked, k=key: selected_model_key.update({"value": k}))
                button_group.addButton(rb)
                model_frame_layout.addWidget(rb)
                
                # Description
                desc_label = QLabel(info["desc"])
                desc_font = QFont("Arial", 9)
                desc_label.setFont(desc_font)
                desc_label.setStyleSheet("color: #666666; margin-left: 20px;")
                model_frame_layout.addWidget(desc_label)
                
                # Separator
                if key != "custom":
                    separator = QFrame()
                    separator.setFrameShape(QFrame.Shape.HLine)
                    separator.setFrameShadow(QFrame.Shadow.Sunken)
                    model_frame_layout.addWidget(separator)
            
            main_layout.addWidget(model_frame)
            
            # Custom model ID frame (initially hidden)
            custom_frame = QGroupBox("Custom Model ID")
            custom_frame_font = QFont("Arial", 11, QFont.Weight.Bold)
            custom_frame.setFont(custom_frame_font)
            custom_frame_layout = QHBoxLayout(custom_frame)
            custom_frame_layout.setContentsMargins(15, 15, 15, 15)
            
            custom_label = QLabel("Model ID:")
            custom_label_font = QFont("Arial", 10)
            custom_label.setFont(custom_label_font)
            custom_frame_layout.addWidget(custom_label)
            
            custom_entry = QLineEdit()
            custom_entry.setPlaceholderText("e.g., Qwen/Qwen2-VL-2B-Instruct")
            custom_entry.setFont(custom_label_font)
            custom_entry.textChanged.connect(lambda text: custom_model_id_text.update({"value": text}))
            custom_frame_layout.addWidget(custom_entry)
            
            custom_frame.hide()  # Hidden by default
            main_layout.addWidget(custom_frame)
            
            # Toggle custom frame visibility
            def toggle_custom_frame():
                if selected_model_key["value"] == "custom":
                    custom_frame.show()
                else:
                    custom_frame.hide()
            
            for rb in button_group.buttons():
                rb.clicked.connect(toggle_custom_frame)
            
            # GPU status frame
            gpu_frame = QGroupBox("System Status")
            gpu_frame_font = QFont("Arial", 11, QFont.Weight.Bold)
            gpu_frame.setFont(gpu_frame_font)
            gpu_frame_layout = QVBoxLayout(gpu_frame)
            gpu_frame_layout.setContentsMargins(15, 15, 15, 15)
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_text = f"✓ GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)"
                    gpu_color = '#4CAF50'
                else:
                    gpu_text = "✗ No GPU detected - will use CPU (very slow)"
                    gpu_color = '#f44336'
            except:
                gpu_text = "? GPU status unknown - install torch with CUDA"
                gpu_color = '#FF9800'
            
            gpu_label = QLabel(gpu_text)
            gpu_label_font = QFont("Arial", 10)
            gpu_label.setFont(gpu_label_font)
            gpu_label.setStyleSheet(f"color: {gpu_color};")
            gpu_frame_layout.addWidget(gpu_label)
            
            main_layout.addWidget(gpu_frame)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            model_confirmed = {'value': False, 'model_key': None, 'model_id': None}
            
            def confirm_selection():
                selected = selected_model_key["value"]
                if selected == "custom":
                    if not custom_model_id_text["value"].strip():
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.critical(selection_dialog, "Error", "Please enter a model ID")
                        return
                    model_confirmed['model_key'] = selected
                    model_confirmed['model_id'] = custom_model_id_text["value"].strip()
                else:
                    model_confirmed['model_key'] = selected
                    model_confirmed['model_id'] = f"Qwen/Qwen2-VL-{selected}-Instruct"
                model_confirmed['value'] = True
                selection_dialog.accept()
            
            proceed_btn = QPushButton("Continue")
            proceed_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px 20px; font-weight: bold; }")
            proceed_btn.clicked.connect(confirm_selection)
            button_layout.addWidget(proceed_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setMinimumWidth(100)  # Ensure enough width for text
            cancel_btn.setStyleSheet("QPushButton { background-color: #9E9E9E; color: white; padding: 8px 20px; }")
            cancel_btn.clicked.connect(selection_dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            button_layout.addStretch()
            main_layout.addLayout(button_layout)
            
            # Show dialog and wait for result
            result = selection_dialog.exec()
            
            if not model_confirmed['value'] or result == QDialog.DialogCode.Rejected:
                return
            
            selected_model_key = model_confirmed['model_key']
            model_id = model_confirmed['model_id']
            total_size_mb = model_sizes['Qwen2-VL'][selected_model_key]
        elif provider == 'rapidocr':
            total_size_mb = 50  # Approximate size for display
            model_id = None
            selected_model_key = None
        else:
            total_size_mb = model_sizes.get(provider, 500)
            model_id = None
            selected_model_key = None
        
        # Create download dialog using PySide6
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QTextEdit, QGroupBox, QApplication
        from PySide6.QtCore import Qt, QObject, Signal
        from PySide6.QtGui import QFont
            
        # Create status update function for model download
        def update_status(percent, downloaded_mb, total_mb, speed_mb):
            if progress_label and status_label:
                try:
                    progress_label.setText(f"Downloading: {percent}%")
                    status_label.setText(f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB @ {speed_mb:.1f} MB/s")
                    progress_bar.setValue(percent)
                except Exception:
                    pass
        
        # Use the manga integration dialog as parent so it stays on top of it
        parent = self.dialog if hasattr(self, 'dialog') else (self.main_gui if hasattr(self.main_gui, 'centralWidget') else None)
        download_dialog = QDialog(parent)
        download_dialog.setWindowTitle(f"Download {provider} Model")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.31)  # 31% of screen width
        height = int(screen.height() * 0.42)  # 42% of screen height
        download_dialog.setFixedSize(width, height)
        
        # Make it non-modal and stay on top
        download_dialog.setModal(False)
        download_dialog.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint)
        download_dialog.setAttribute(Qt.WA_ShowWithoutActivating, False)  # Allow activation

        class DownloadDialogBridge(QObject):
            update = Signal(object)

        download_dialog_bridge = DownloadDialogBridge(download_dialog)
        download_dialog_bridge.update.connect(lambda func: func())
        
        # Apply dark theme styling
        download_dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #5a9fd4;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #4a7ba7;
                border-radius: 2px;
            }
            QTextEdit {
                background-color: #252525;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 5px;
                font-family: Courier;
            }
        """)
        
        # Main layout
        main_dialog_layout = QVBoxLayout(download_dialog)
        main_dialog_layout.setContentsMargins(20, 20, 20, 20)
        main_dialog_layout.setSpacing(10)
        
        # Info section
        info_frame = QGroupBox("Model Information")
        info_frame_layout = QVBoxLayout(info_frame)
        info_frame_layout.setContentsMargins(15, 15, 15, 10)
        
        if provider == 'Qwen2-VL':
            info_text = f"📚 Qwen2-VL {selected_model_key} Model\n"
            info_text += f"Model ID: {model_id}\n"
            info_text += f"Estimated size: ~{total_size_mb/1000:.1f}GB\n"
            info_text += "Vision-Language model for Korean OCR"
        else:
            info_text = f"📚 {provider} Model\nOptimized for manga/manhwa text detection"
        
        info_label = QLabel(info_text)
        info_label_font = QFont("Arial", 10)
        info_label.setFont(info_label_font)
        info_label.setAlignment(Qt.AlignLeft)
        info_label.setWordWrap(True)
        info_frame_layout.addWidget(info_label)
        
        main_dialog_layout.addWidget(info_frame)
        
        # Progress section
        progress_frame = QGroupBox("Download Progress")
        progress_frame_layout = QVBoxLayout(progress_frame)
        progress_frame_layout.setContentsMargins(15, 15, 15, 10)
        
        progress_label = QLabel("Ready to download")
        progress_label_font = QFont("Arial", 10)
        progress_label.setFont(progress_label_font)
        progress_frame_layout.addWidget(progress_label)
        
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.setMinimumWidth(550)
        progress_frame_layout.addWidget(progress_bar)
        
        size_label = QLabel("")
        size_label_font = QFont("Arial", 9)
        size_label.setFont(size_label_font)
        size_label.setStyleSheet("color: #999999;")
        progress_frame_layout.addWidget(size_label)
        
        speed_label = QLabel("")
        speed_label_font = QFont("Arial", 9)
        speed_label.setFont(speed_label_font)
        speed_label.setStyleSheet("color: #999999;")
        progress_frame_layout.addWidget(speed_label)
        
        status_label = QLabel("Click 'Download' to begin")
        status_label_font = QFont("Arial", 9)
        status_label.setFont(status_label_font)
        status_label.setStyleSheet("color: #999999;")
        progress_frame_layout.addWidget(status_label)
        
        main_dialog_layout.addWidget(progress_frame)
        
        # Log section
        log_frame = QGroupBox("Download Log")
        log_frame_layout = QVBoxLayout(log_frame)
        log_frame_layout.setContentsMargins(15, 15, 15, 10)
        
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details_text.setMinimumHeight(150)
        details_text_font = QFont("Courier", 9)
        details_text.setFont(details_text_font)
        log_frame_layout.addWidget(details_text)
        
        main_dialog_layout.addWidget(log_frame)
            
        def add_log(message):
            """Add message to log"""
            details_text.append(message)
            try:
                if provider == 'manga-ocr':
                    transfer_match = re.search(r"Starting transfer:\s+(\d+)\s+files", str(message))
                    if transfer_match:
                        progress_bar.setValue(max(progress_bar.value(), 20))
                        progress_label.setText("Starting transfer...")
                        status_label.setText(f"0/{transfer_match.group(1)} files")

                    file_match = re.search(r"Downloading \[(\d+)/(\d+)\]:\s+(.+)", str(message))
                    if file_match:
                        idx = int(file_match.group(1))
                        count = max(1, int(file_match.group(2)))
                        file_name = file_match.group(3)
                        file_progress = max(0.0, (idx - 1) / count)
                        progress = min(20 + file_progress * 75, 95)
                        progress_bar.setValue(max(progress_bar.value(), int(progress)))
                        progress_label.setText(f"Downloading: {progress_bar.value()}%")
                        status_label.setText(f"[{idx}/{count}] {file_name} (connecting...)")
                        speed_label.setText("Speed: waiting for first bytes")

                    done_match = re.search(r"Already downloaded:\s+(.+)", str(message))
                    if done_match:
                        speed_label.setText("Speed: cached file")
            except Exception:
                pass
            # Scroll to bottom
            from PySide6.QtGui import QTextCursor
            cursor = details_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            details_text.setTextCursor(cursor)

        def post_dialog_update(func):
            """Run a small UI update on the download dialog's Qt thread."""
            try:
                download_dialog_bridge.update.emit(func)
            except Exception:
                pass
        
        # Buttons frame
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Download tracking variables
        download_active = {'value': False}
        
        def get_dir_size(path):
            """Get total size of directory"""
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        if filename.endswith(".part"):
                            continue
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total += os.path.getsize(filepath)
            except:
                pass
            return total
        
        def download_with_progress():
            """Download model with real progress tracking"""
            import time
            
            download_active['value'] = True
            total_size = total_size_mb * 1024 * 1024
            
            try:
                if provider == 'manga-ocr':
                    progress_label.setText("Downloading manga-ocr model...")
                    add_log("Downloading manga-ocr model from Hugging Face...")
                    add_log("This will download ~450MB of model files")
                    progress_bar.setValue(10)
                    
                    try:
                        import queue
                        from huggingface_hub import snapshot_download
                        from tqdm.auto import tqdm as hf_tqdm
                        
                        # Download the model files directly without importing manga_ocr
                        model_repo = "kha-white/manga-ocr-base"
                        add_log(f"Repository: {model_repo}")
                        
                        if getattr(sys, 'frozen', False):
                            app_data_root = (
                                os.environ.get('LOCALAPPDATA')
                                or os.environ.get('APPDATA')
                                or os.path.expanduser('~')
                            )
                            local_model_dir = os.path.join(app_data_root, 'Glossarion', 'models', 'manga-ocr-base')
                        else:
                            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                            local_model_dir = os.path.join(root_dir, 'models', 'manga-ocr-base')
                        os.makedirs(local_model_dir, exist_ok=True)
                        os.environ['MANGA_OCR_LOCAL_DIR'] = local_model_dir
                        start_time = time.time()
                        initial_local_size = get_dir_size(local_model_dir)
                        
                        add_log("Preparing Hugging Face file list...")
                        progress_bar.setValue(20)
                        
                        # Download with progress tracking
                        import threading
                        download_complete = threading.Event()
                        download_error = [None]
                        progress_lock = threading.Lock()
                        log_queue = queue.Queue()
                        progress_state = {
                            'downloaded': 0,
                            'total': 0,
                            'current_file': '',
                            'file_index': 0,
                            'file_count': 0,
                            'file_downloaded': 0,
                            'file_size': 0,
                            'file_started_at': 0.0,
                            'hf_rate': 0.0,
                            'auto_loaded': False,
                        }

                        class MangaOcrDownloadTqdm(hf_tqdm):
                            """Bridge Hugging Face tqdm progress into the PySide dialog."""
                            def __init__(self, *args, **kwargs):
                                self._glossarion_unit = kwargs.get('unit') or ''
                                super().__init__(*args, **kwargs)
                                self._publish_progress()

                            def update(self, n=1):
                                result = super().update(n)
                                self._publish_progress()
                                return result

                            def close(self):
                                self._publish_progress()
                                return super().close()

                            def _publish_progress(self):
                                try:
                                    desc = (getattr(self, 'desc', '') or '').strip()
                                    total = int(getattr(self, 'total', 0) or 0)
                                    current = int(getattr(self, 'n', 0) or 0)
                                    rate = 0.0
                                    try:
                                        rate = float((getattr(self, 'format_dict', {}) or {}).get('rate') or 0.0)
                                    except Exception:
                                        rate = 0.0

                                    with progress_lock:
                                        progress_state['current_file'] = desc or "Hugging Face transfer"
                                        progress_state['hf_rate'] = rate
                                        progress_state['file_started_at'] = progress_state.get('file_started_at') or time.time()
                                        if self._glossarion_unit == 'B' or total > 1024 * 1024:
                                            progress_state['file_downloaded'] = current
                                            progress_state['file_size'] = total
                                            progress_state['downloaded'] = max(int(progress_state.get('downloaded', 0) or 0), current)
                                            if total:
                                                progress_state['total'] = max(int(progress_state.get('total', 0) or 0), total)
                                        else:
                                            progress_state['file_index'] = current
                                            progress_state['file_count'] = total
                                            progress_state['file_downloaded'] = 0
                                            progress_state['file_size'] = 0
                                            progress_state['total'] = total_size
                                    def apply_hf_progress():
                                        try:
                                            if total:
                                                pct = int(min(95, 20 + (current / max(1, total)) * 75))
                                                progress_bar.setValue(pct)
                                                progress_label.setText(f"Downloading: {pct}%")
                                                if self._glossarion_unit == 'B' or total > 1024 * 1024:
                                                    status_label.setText(
                                                        f"{desc or 'Downloading file'}: "
                                                        f"{current / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB"
                                                    )
                                                    if rate > 0:
                                                        speed_label.setText(f"Speed: {rate / (1024 * 1024):.1f} MB/s")
                                                else:
                                                    status_label.setText(f"{desc or 'Fetching files'}: {current}/{total} files")
                                                    if rate > 0:
                                                        speed_label.setText(f"Speed: {rate:.1f} files/s")
                                        except Exception:
                                            pass
                                    post_dialog_update(apply_hf_progress)
                                except Exception:
                                    pass

                        def _manga_ocr_snapshot_complete_and_stable(path: str, timeout_s: float = 90.0) -> bool:
                            """Wait until the local manga-ocr snapshot has required files and stable weights."""
                            required_files = [
                                'config.json',
                                'preprocessor_config.json',
                                'pytorch_model.bin',
                            ]
                            tokenizer_alternates = ('tokenizer.json', 'tokenizer_config.json')
                            incomplete_suffixes = ('.part', '.incomplete', '.tmp')
                            deadline = time.time() + timeout_s
                            last_sizes = None
                            stable_seen = 0

                            def _snapshot_state():
                                if not os.path.isdir(path):
                                    return None, "model directory does not exist yet"
                                for dirpath, _, filenames in os.walk(path):
                                    for filename in filenames:
                                        if filename.lower().endswith(incomplete_suffixes):
                                            return None, f"still writing {filename}"
                                missing = [name for name in required_files if not os.path.exists(os.path.join(path, name))]
                                if not any(os.path.exists(os.path.join(path, name)) for name in tokenizer_alternates):
                                    missing.append('tokenizer.json/tokenizer_config.json')
                                if missing:
                                    return None, f"missing {', '.join(missing)}"
                                weight_paths = [
                                    os.path.join(path, name)
                                    for name in ('pytorch_model.bin', 'model.safetensors')
                                    if os.path.exists(os.path.join(path, name))
                                ]
                                if not weight_paths:
                                    return None, "missing model weights"
                                sizes = tuple((p, os.path.getsize(p)) for p in weight_paths)
                                if any(size <= 0 for _, size in sizes):
                                    return None, "model weight file is empty"
                                return sizes, "ready"

                            while time.time() < deadline:
                                sizes, reason = _snapshot_state()
                                if sizes is not None and sizes == last_sizes:
                                    stable_seen += 1
                                    if stable_seen >= 3:
                                        return True
                                else:
                                    stable_seen = 0
                                    last_sizes = sizes
                                log_queue.put(f"Waiting for manga-ocr files to finish: {reason}")
                                time.sleep(1.0)
                            return False
                        
                        def download_model():
                            try:
                                if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1":
                                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
                                os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
                                log_queue.put("Using Hugging Face snapshot downloader in the background...")
                                with progress_lock:
                                    progress_state['file_count'] = 0
                                    progress_state['total'] = total_size
                                    progress_state['current_file'] = "Hugging Face snapshot"
                                    progress_state['file_started_at'] = time.time()
                                log_queue.put("Starting transfer with resume support. Large files may pause while Hugging Face connects.")
                                snapshot_download(
                                    repo_id=model_repo,
                                    repo_type="model",
                                    local_dir=local_model_dir,
                                    local_dir_use_symlinks=False,
                                    resume_download=True,
                                    ignore_patterns=[".gitattributes"],
                                    tqdm_class=MangaOcrDownloadTqdm,
                                )
                                os.environ['MANGA_OCR_LOCAL_DIR'] = local_model_dir
                                log_queue.put(f"Saved model to: {local_model_dir}")
                                log_queue.put("Verifying manga-ocr files are complete before loading...")
                                post_dialog_update(lambda: (
                                    add_log("Verifying manga-ocr files are complete before loading..."),
                                    progress_bar.setValue(97),
                                    progress_label.setText("Verifying model files..."),
                                    status_label.setText("Waiting for Hugging Face file writes to settle...")
                                ))
                                if not _manga_ocr_snapshot_complete_and_stable(local_model_dir):
                                    raise RuntimeError("manga-ocr download did not settle into a complete local snapshot; wait for downloads to finish and retry.")
                                log_queue.put("Download complete. Loading manga-ocr model now...")
                                post_dialog_update(lambda: (
                                    add_log(f"Saved model to: {local_model_dir}"),
                                    add_log("Download complete. Loading manga-ocr model now..."),
                                    progress_bar.setValue(99),
                                    progress_label.setText("Loading manga-ocr model..."),
                                    status_label.setText("Initializing model on CPU...")
                                ))
                                with progress_lock:
                                    progress_state['current_file'] = "Loading manga-ocr model"
                                    progress_state['file_index'] = 0
                                    progress_state['file_count'] = 0
                                    progress_state['file_downloaded'] = 0
                                    progress_state['file_size'] = 0
                                    progress_state['downloaded'] = total_size
                                    progress_state['total'] = total_size
                                    progress_state['hf_rate'] = 0.0
                                    progress_state['file_started_at'] = time.time()
                                load_success = False
                                last_load_error = ""
                                ocr_manager = self._ensure_ocr_manager()
                                for load_attempt in range(1, 4):
                                    log_queue.put(f"Loading manga-ocr model (attempt {load_attempt}/3)...")
                                    if load_attempt > 1:
                                        time.sleep(3.0)
                                        try:
                                            from ocr_manager import MangaOCRProvider
                                            ocr_manager.providers['manga-ocr'] = MangaOCRProvider(ocr_manager.log_callback)
                                        except Exception:
                                            pass
                                    load_success = bool(ocr_manager.load_provider('manga-ocr'))
                                    if load_success:
                                        break
                                    try:
                                        provider_obj = ocr_manager.get_provider('manga-ocr')
                                        last_load_error = str(getattr(provider_obj, 'last_error', '') or '')
                                    except Exception:
                                        last_load_error = ""
                                    if last_load_error:
                                        log_queue.put(f"manga-ocr load attempt {load_attempt}/3 failed: {last_load_error}")
                                    else:
                                        log_queue.put(f"manga-ocr load attempt {load_attempt}/3 failed without a provider error")
                                if not load_success:
                                    detail = f" Last provider error: {last_load_error}" if last_load_error else ""
                                    raise RuntimeError(f"Model downloaded, but manga-ocr failed to auto-load after 3 attempts.{detail}")
                                with progress_lock:
                                    progress_state['auto_loaded'] = True
                                log_queue.put("manga-ocr model loaded successfully.")
                                post_dialog_update(lambda: (
                                    add_log("manga-ocr model loaded successfully."),
                                    progress_bar.setValue(100),
                                    progress_label.setText("✅ manga-ocr ready!"),
                                    status_label.setText("Model downloaded and loaded"),
                                    speed_label.setText("Ready"),
                                    cancel_btn.setText("Close"),
                                    self._check_provider_status()
                                ))
                                download_complete.set()
                            except Exception as e:
                                download_error[0] = e
                                err_text = str(e)
                                post_dialog_update(lambda err_text=err_text: (
                                    add_log(f"Download/load error: {err_text}"),
                                    progress_label.setText("Download/load failed"),
                                    status_label.setText("Error occurred"),
                                    speed_label.setText(""),
                                    cancel_btn.setText("Close")
                                ))
                                download_complete.set()
                        
                        def check_progress():
                            """Recursively check progress using QTimer.singleShot"""
                            try:
                                while True:
                                    try:
                                        add_log(log_queue.get_nowait())
                                    except queue.Empty:
                                        break

                                with progress_lock:
                                    downloaded = int(progress_state.get('downloaded', 0) or 0)
                                    total = int(progress_state.get('total', 0) or 0)
                                    current_file = str(progress_state.get('current_file', '') or '')
                                    file_index = int(progress_state.get('file_index', 0) or 0)
                                    file_count = int(progress_state.get('file_count', 0) or 0)
                                    file_downloaded = int(progress_state.get('file_downloaded', 0) or 0)
                                    file_size = int(progress_state.get('file_size', 0) or 0)
                                    file_started_at = float(progress_state.get('file_started_at', 0.0) or 0.0)
                                    hf_rate = float(progress_state.get('hf_rate', 0.0) or 0.0)
                                    auto_loaded = bool(progress_state.get('auto_loaded', False))

                                local_size = get_dir_size(local_model_dir)
                                downloaded = max(downloaded, local_size)
                                total = total or total_size

                                if download_complete.is_set():
                                    # Handle completion
                                    if download_error[0]:
                                        progress_label.setText("❌ Download failed")
                                        status_label.setText("Error occurred")
                                        # Use QTimer.singleShot to defer log updates
                                        QTimer.singleShot(0, lambda: add_log(f"❌ Download error: {str(download_error[0])}"))
                                    else:
                                        progress_bar.setValue(100)
                                        progress_label.setText("✅ manga-ocr ready!")
                                        status_label.setText("Model downloaded and loaded")
                                        # Defer log updates to avoid paint conflicts
                                        QTimer.singleShot(0, lambda: add_log("✅ Model files downloaded successfully"))
                                        QTimer.singleShot(10, lambda: add_log(""))
                                        if auto_loaded:
                                            QTimer.singleShot(20, lambda: add_log("manga-ocr is loaded and ready to use."))
                                        try:
                                            self.update_queue.put(('call_method', self._check_provider_status, ()))
                                        except:
                                            pass
                                    return
                                
                                if not download_active['value']:
                                    return
                                
                                if file_count > 0:
                                    if file_size > 0:
                                        completed_files = max(0, file_index - 1)
                                        current_fraction = min(1.0, file_downloaded / file_size)
                                    else:
                                        completed_files = max(0, file_index)
                                        current_fraction = 0.0
                                    file_progress = (completed_files + current_fraction) / max(1, file_count)
                                    progress = min(20 + file_progress * 75, 95)
                                    progress_bar.setValue(int(progress))
                                    mb_total = (total or total_size) / (1024 * 1024)
                                elif total > 0:
                                    progress = min(20 + (downloaded / total) * 75, 95)
                                    progress_bar.setValue(int(progress))
                                    mb_total = total / (1024 * 1024)
                                else:
                                    progress_bar.setValue(20)
                                    mb_total = total_size / (1024 * 1024)

                                elapsed = time.time() - start_time
                                transferred = max(0, downloaded - initial_local_size)
                                if file_size > 0 and hf_rate > 0:
                                    speed_label.setText(f"Speed: {hf_rate / (1024 * 1024):.1f} MB/s")
                                elif file_count > 0 and hf_rate > 0:
                                    speed_label.setText(f"Speed: {hf_rate:.1f} files/s")
                                elif elapsed > 1 and transferred > 0:
                                    speed = transferred / elapsed
                                    speed_mb = speed / (1024 * 1024)
                                    speed_label.setText(f"Speed: {speed_mb:.1f} MB/s")
                                elif current_file and file_started_at:
                                    connect_elapsed = max(0.0, time.time() - file_started_at)
                                    speed_label.setText(f"Waiting for Hugging Face file writes: {connect_elapsed:.0f}s")

                                mb_downloaded = downloaded / (1024 * 1024)
                                size_label.setText(f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB")
                                if current_file:
                                    if file_size > 0:
                                        status_label.setText(
                                            f"[{file_index}/{file_count}] {current_file} "
                                            f"({file_downloaded / (1024 * 1024):.1f}/{file_size / (1024 * 1024):.1f} MB)"
                                        )
                                    elif file_count > 0:
                                        status_label.setText(f"{current_file}: {file_index}/{file_count} files")
                                    else:
                                        status_label.setText(f"{current_file} ({mb_downloaded:.1f} MB on disk)")
                                progress_label.setText(f"Downloading: {progress_bar.value()}%")
                                
                                # Schedule next check
                                QTimer.singleShot(250, check_progress)
                            except Exception as progress_error:
                                try:
                                    add_log(f"Progress updater error: {progress_error}")
                                except Exception:
                                    pass
                        
                        # Start download thread
                        download_thread = threading.Thread(target=download_model, daemon=True)
                        download_thread.start()
                        
                        # Start progress checking
                        QTimer.singleShot(250, check_progress)
                            
                    except ImportError:
                        progress_label.setText("❌ Missing huggingface_hub")
                        status_label.setText("Install huggingface_hub first")
                        add_log("ERROR: huggingface_hub not installed")
                        add_log("Run: pip install huggingface_hub")
                    except Exception as e:
                        raise  # Re-raise to be caught by outer exception handler
                        
                elif provider == 'Qwen2-VL':
                    try:
                        from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
                        import torch
                    except ImportError as e:
                        progress_label.setText("❌ Missing dependencies")
                        status_label.setText("Install dependencies first")
                        add_log(f"ERROR: {str(e)}")
                        add_log("Please install manually:")
                        add_log("pip install transformers torch torchvision")
                        return
                    
                    progress_label.setText(f"Downloading model...")
                    add_log(f"Starting download of {model_id}")
                    progress_bar.setValue(10)
                    
                    add_log("Downloading processor...")
                    status_label.setText("Downloading processor...")
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    progress_bar.setValue(30)
                    add_log("✓ Processor downloaded")
                    
                    add_log("Downloading tokenizer...")
                    status_label.setText("Downloading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    progress_bar.setValue(50)
                    add_log("✓ Tokenizer downloaded")
                    
                    add_log("Downloading model weights (this may take several minutes)...")
                    status_label.setText("Downloading model weights...")
                    progress_label.setText("Downloading model weights...")
                    
                    if torch.cuda.is_available():
                        add_log(f"Using GPU: {torch.cuda.get_device_name(0)}")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        add_log("No GPU detected, will load on CPU")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                    
                    progress_bar.setValue(90)
                    add_log("✓ Model weights downloaded")
                    
                    add_log("Initializing model...")
                    status_label.setText("Initializing...")
                    
                    qwen_provider = self._ensure_ocr_manager().get_provider('Qwen2-VL')
                    if qwen_provider:
                        qwen_provider.processor = processor
                        qwen_provider.tokenizer = tokenizer  
                        qwen_provider.model = model
                        qwen_provider.model.eval()
                        qwen_provider.is_loaded = True
                        qwen_provider.is_installed = True
                        
                        if selected_model_key:
                            qwen_provider.loaded_model_size = selected_model_key
                    
                    progress_bar.setValue(100)
                    progress_label.setText("✅ Download complete!")
                    status_label.setText("Model ready for Korean OCR!")
                    add_log("✓ Model ready to use!")
                    
                    # Schedule status check on main thread
                    self.update_queue.put(('call_method', self._check_provider_status, ()))
                    
                elif provider == 'rapidocr':
                    progress_label.setText("📦 RapidOCR Installation Instructions")
                    add_log("RapidOCR requires manual pip installation")
                    progress_bar.setValue(20)
                    
                    add_log("Command to run:")
                    add_log("pip install rapidocr-onnxruntime")
                    progress_bar.setValue(50)
                    
                    add_log("")
                    add_log("After installation:")
                    add_log("1. Close this dialog")
                    add_log("2. Click 'Load Model' to initialize RapidOCR")
                    add_log("3. Status should show '✅ Model loaded'")
                    progress_bar.setValue(100)
                    
                    progress_label.setText("📦 Installation instructions shown")
                    status_label.setText("Manual pip install required")
                    
                    download_btn.setEnabled(False)
                    cancel_btn.setText("Close")
                        
            except Exception as e:
                progress_label.setText("❌ Download failed")
                status_label.setText(f"Error: {str(e)[:50]}")
                add_log(f"ERROR: {str(e)}")
                self._log(f"Download error: {str(e)}", "error")
                
            finally:
                download_active['value'] = False
        
        def start_download():
            """Start download - runs on main thread, spawns background thread internally"""
            download_btn.setEnabled(False)
            cancel_btn.setText("Cancel")
            
            # Call download_with_progress directly on main thread
            # It will spawn its own background thread for the actual download
            download_with_progress()
        
        def cancel_download():
            """Cancel or close dialog"""
            if download_active['value']:
                download_active['value'] = False
                status_label.setText("Cancelling...")
            else:
                download_dialog.close()
        
        download_btn = QPushButton("Download")
        download_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4a7ba7; 
                color: white; 
                padding: 8px 20px; 
                font-weight: bold;
                border: 1px solid #5a9fd4;
            }
            QPushButton:hover { 
                background-color: #5a9fd4; 
            }
            QPushButton:pressed { 
                background-color: #3a6a94; 
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666666;
                border-color: #3a3a3a;
            }
        """)
        download_btn.clicked.connect(start_download)
        button_layout.addWidget(download_btn)
        
        cancel_btn = QPushButton("Close")
        cancel_btn.setMinimumWidth(100)  # Ensure enough width for text
        cancel_btn.setStyleSheet("""
            QPushButton { 
                background-color: #666666; 
                color: white; 
                padding: 8px 20px;
                border: 1px solid #777777;
            }
            QPushButton:hover { 
                background-color: #777777; 
            }
            QPushButton:pressed { 
                background-color: #555555; 
            }
        """)
        cancel_btn.clicked.connect(cancel_download)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        main_dialog_layout.addLayout(button_layout)
        
        # Show dialog (non-modal)
        download_dialog.show()
    
    def _validate_google_credentials(self, creds_path: str):
        """Validate Google Cloud Vision credentials JSON file.
        Returns (ok: bool, message: str).
        """
        try:
            if not creds_path:
                return False, "❌ Credentials needed"
            if not os.path.exists(creds_path):
                return False, "❌ Credentials file not found"
            # Basic JSON validation
            try:
                with open(creds_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return False, "❌ Invalid credentials JSON"
            # Required keys for service account
            required = ['type', 'project_id', 'private_key', 'client_email']
            missing = [k for k in required if k not in data]
            if missing:
                return False, "❌ Invalid credentials JSON"
            if str(data.get('type', '')).lower() != 'service_account':
                return False, "❌ Invalid credentials JSON"
            return True, ""
        except Exception:
            return False, "❌ Invalid credentials JSON"
    
    def _check_provider_status(self):
        """Check and display OCR provider status"""
        # Skip during initialization to prevent lag
        if hasattr(self, '_initializing_gui') and self._initializing_gui:
            if hasattr(self, 'provider_status_label'):
                self.provider_status_label.setText("")
                self.provider_status_label.setStyleSheet("color: black;")
            return
        
        # Get provider value
        if not hasattr(self, 'ocr_provider_value'):
            # Not initialized yet, skip
            return
        provider = self.ocr_provider_value
        
        # Hide ALL buttons first
        if hasattr(self, 'provider_setup_btn'):
            self.provider_setup_btn.setVisible(False)
        if hasattr(self, 'ocr_download_model_btn'):
            self.ocr_download_model_btn.setVisible(False)
        
        if provider == 'google':
            # Google - check for credentials file
            google_creds = self.main_gui.config.get('google_vision_credentials', '')
            ok, msg = self._validate_google_credentials(google_creds)
            if ok:
                self.provider_status_label.setText("✅ Ready")
                self.provider_status_label.setStyleSheet("color: green;")
            else:
                self.provider_status_label.setText(msg or "❌ Credentials needed")
                self.provider_status_label.setStyleSheet("color: red;")
            
        elif provider == 'azure':
            # Azure - check for API key
            azure_key = self.main_gui.config.get('azure_vision_key', '')
            if azure_key:
                self.provider_status_label.setText("✅ Ready")
                self.provider_status_label.setStyleSheet("color: green;")
            else:
                self.provider_status_label.setText("❌ Key needed")
                self.provider_status_label.setStyleSheet("color: red;")
        
        elif provider == 'azure-document-intelligence':
            # Azure Document Intelligence - check for API key (uses same config as Azure CV)
            azure_key = self.main_gui.config.get('azure_vision_key', '') or self.main_gui.config.get('azure_document_intelligence_key', '')
            azure_endpoint = self.main_gui.config.get('azure_vision_endpoint', '') or self.main_gui.config.get('azure_document_intelligence_endpoint', '')
            if azure_key and azure_endpoint:
                self.provider_status_label.setText("✅ Ready (successor to Azure AI Vision)")
                self.provider_status_label.setStyleSheet("color: green;")
            elif azure_key:
                self.provider_status_label.setText("⚠️ Endpoint needed")
                self.provider_status_label.setStyleSheet("color: orange;")
            else:
                self.provider_status_label.setText("❌ Key & Endpoint needed")
                self.provider_status_label.setStyleSheet("color: red;")

        elif provider == 'custom-api':
            # Custom API - check for main API key
            api_key = None
            if hasattr(self.main_gui, 'api_key_entry'):
                try:
                    # PySide6 QLineEdit uses .text()
                    api_key = self.main_gui.api_key_entry.text().strip() if hasattr(self.main_gui.api_key_entry, 'text') else self.main_gui.api_key_entry.get().strip()
                except:
                    pass
            if not api_key and hasattr(self.main_gui, 'config') and self.main_gui.config.get('api_key'):
                api_key = self.main_gui.config.get('api_key')
            
            # Check if AI bubble detection is enabled
            manga_settings = self.main_gui.config.get('manga_settings', {})
            ocr_settings = manga_settings.get('ocr', {})
            bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', True)
            
            if api_key:
                if bubble_detection_enabled:
                    self.provider_status_label.setText("✅ Ready")
                    self.provider_status_label.setStyleSheet("color: green;")
                else:
                    self.provider_status_label.setText("⚠️ Enable AI bubble detection for best results")
                    self.provider_status_label.setStyleSheet("color: orange;")
            else:
                # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
                _model = (self.main_gui.config.get('model', '') or '').lower()
                try:
                    from unified_api_client import UnifiedClient as _UC
                    _uses_own_auth = not _UC._model_needs_api_key(_model)
                except Exception:
                    _uses_own_auth = False
                if _uses_own_auth or not api_key:
                    _custom_ep_on = os.environ.get('USE_CUSTOM_OPENAI_ENDPOINT', '0') == '1'
                    _custom_ep_url = (os.environ.get('OPENAI_CUSTOM_BASE_URL', '') or '').lower()
                    _is_local_endpoint = _custom_ep_on and any(h in _custom_ep_url for h in ('localhost', '127.0.0.1', '0.0.0.0', '::1'))
                    _label_suffix = "local endpoint" if _is_local_endpoint else ("own-auth model" if _uses_own_auth else "custom-api")
                    if bubble_detection_enabled:
                        self.provider_status_label.setText(f"✅ Ready ({_label_suffix})")
                        self.provider_status_label.setStyleSheet("color: green;")
                    else:
                        self.provider_status_label.setText("⚠️ Enable AI bubble detection for best results")
                        self.provider_status_label.setStyleSheet("color: orange;")
     
        elif provider == 'Qwen2-VL':
            # Initialize OCR manager if needed
            ocr_manager = self._ensure_ocr_manager()
            
            # Get provider instance to check its loading status
            provider_instance = None
            try:
                provider_instance = ocr_manager.get_provider(provider)
                # Pass model size to provider for proper loading
                if provider == 'Qwen2-VL':
                    saved_model_size = getattr(self, 'qwen2vl_model_size', self.main_gui.config.get('qwen2vl_model_size', '1'))
                    provider_instance.qwen2vl_model_size = saved_model_size
            except Exception:
                pass
                
            # Check status first
            status = ocr_manager.check_provider_status(provider)
            
            # When displaying status for loaded model
            if status['loaded']:
                # Map the saved size to display name
                size_names = {'1': '2B', '2': '7B', '3': '72B', '4': 'custom'}
                display_size = size_names.get(saved_model_size, saved_model_size)
                self.provider_status_label.setText(f"✅ {display_size} model loaded")
                self.provider_status_label.setStyleSheet("color: green;")
                
                # Show reload button
                self.provider_setup_btn.setText("Reload")
                self.provider_setup_btn.setVisible(True)
                
            elif status['installed']:
                # Dependencies installed but model not loaded
                self.provider_status_label.setText("📦 Dependencies ready")
                self.provider_status_label.setStyleSheet("color: orange;")
                
                # Show Load button
                self.provider_setup_btn.setText("Load Model")
                self.provider_setup_btn.setVisible(True)
                
                # Also show Download button
                self.ocr_download_model_btn.setText("Download Model")
                self.ocr_download_model_btn.setVisible(True)
                
            else:
                # Not installed
                self.provider_status_label.setText("❌ Not installed")
                self.provider_status_label.setStyleSheet("color: red;")
                
                # Show BOTH buttons
                self.provider_setup_btn.setText("Load Model")
                self.provider_setup_btn.setVisible(True)
                
                self.ocr_download_model_btn.setText("Download Qwen2-VL")
                self.ocr_download_model_btn.setVisible(True)
            
            # Additional GPU status check for Qwen2-VL
            if not status['loaded']:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self._log("⚠️ No GPU detected - Qwen2-VL will run slowly on CPU", "warning")
                except ImportError:
                    pass
 
        else:
            # Local OCR providers
            ocr_manager = self._ensure_ocr_manager()
                
            # Get provider instance to check its loading status
            provider_instance = None
            try:
                provider_instance = ocr_manager.get_provider(provider)
            except Exception:
                pass
                
            # Check if model is loaded and ready
            status = ocr_manager.check_provider_status(provider)
            
            # Determine loaded status for local providers, including JIT models
            is_loaded = status['loaded']
            try:
                # If translator has a local inpainter loaded, reflect it
                if hasattr(self, 'translator') and self.translator:
                    inp = getattr(self.translator, 'local_inpainter', None)
                    if inp and getattr(inp, 'model_loaded', False):
                        is_loaded = True
                # Also check any thread-local inpainter instances
                if hasattr(self.translator, '_thread_local'):
                    tlocal = self.translator._thread_local
                    if hasattr(tlocal, 'local_inpainters') and isinstance(tlocal.local_inpainters, dict):
                        for _k, _inp in tlocal.local_inpainters.items():
                            if getattr(_inp, 'model_loaded', False):
                                is_loaded = True
                                break
            except Exception:
                pass
            
            if is_loaded:
                self.provider_status_label.setText("✅ Model loaded")
                self.provider_status_label.setStyleSheet("color: green;")
                self.provider_setup_btn.setText("Reload")
                self.provider_setup_btn.setVisible(True)
            elif hasattr(self.translator, '_downloading_model') and self.translator._downloading_model:
                self.provider_status_label.setText("📥 Downloading...")
                self.provider_status_label.setStyleSheet("color: blue;")
                self.provider_setup_btn.setEnabled(False)
                self.ocr_download_model_btn.setEnabled(False)
            elif status['installed']:
                # Dependencies installed but model not loaded
                self.provider_status_label.setText("📦 Dependencies ready")
                self.provider_status_label.setStyleSheet("color: orange;")
                self.provider_setup_btn.setText("Load Model")
                self.provider_setup_btn.setVisible(True)
                if provider in ['Qwen2-VL', 'manga-ocr']:
                    self.ocr_download_model_btn.setText("Download Model")
                    self.ocr_download_model_btn.setVisible(True)
            else:
                # Not installed
                self.provider_status_label.setText("❌ Not installed")
                self.provider_status_label.setStyleSheet("color: red;")
                # Categorize providers
                huggingface_providers = ['manga-ocr', 'Qwen2-VL', 'rapidocr']  # Move rapidocr here
                pip_providers = ['easyocr', 'paddleocr', 'doctr']  # Remove rapidocr from here

                if provider in huggingface_providers:
                    # For HuggingFace models, show BOTH buttons
                    self.provider_setup_btn.setText("Load Model")
                    self.provider_setup_btn.setVisible(True)
                    
                    # Download button
                    if provider == 'rapidocr':
                        self.ocr_download_model_btn.setText("Install RapidOCR")
                    else:
                        self.ocr_download_model_btn.setText(f"Download {provider}")
                    self.ocr_download_model_btn.setVisible(True)

                elif provider in pip_providers:
                    # Check if running as .exe
                    if getattr(sys, 'frozen', False):
                        # Running as .exe - can't pip install
                        self.provider_status_label.setText("❌ Not available in .exe")
                        self.provider_status_label.setStyleSheet("color: red;")
                        self._log(f"⚠️ {provider} cannot be installed in standalone .exe version", "warning")
                    else:
                        # Running from Python - can pip install
                        self.provider_setup_btn.setText("Install")
                        self.provider_setup_btn.setVisible(True)

    def _setup_ocr_provider(self):
        """Setup/install/load OCR provider"""
        provider = self.ocr_provider_value
        
        if provider in ['google', 'azure', 'azure-document-intelligence']:
            return  # Cloud providers don't need setup/model loading

        # your own api key
        if provider == 'custom-api':
            # Open configuration dialog for custom API
            try:
                from custom_api_config_dialog import CustomAPIConfigDialog
                dialog = CustomAPIConfigDialog(
                    self.manga_window,
                    self.main_gui.config,
                    self.main_gui.save_config
                )
                # After dialog closes, refresh status
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, self._check_provider_status)
            except ImportError:
                # If dialog not available, show message
                from PySide6.QtWidgets import QMessageBox
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: QMessageBox.information(
                    self.dialog,
                    "Custom API Configuration",
                    "This mode uses your own API key in the main GUI:\n\n"
                    "- Make sure your API supports vision\n"
                    "- api_key: Your API key\n"
                    "- model: Model name\n"
                    "- custom url: You can override API endpoint under Other settings"
                ))
            return
        
        ocr_manager = self._ensure_ocr_manager()
        status = ocr_manager.check_provider_status(provider)
        
        # For Qwen2-VL, check if we need to select model size first
        model_size = None
        if provider == 'Qwen2-VL' and status['installed'] and not status['loaded']:
            # Create PySide6 dialog for model selection
            from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                            QRadioButton, QButtonGroup, QLineEdit, QPushButton,
                                            QGroupBox, QFrame, QMessageBox)
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont
            
            selection_dialog = QDialog(self.dialog)
            selection_dialog.setWindowTitle("Select Qwen2-VL Model Size")
            # Use screen ratios for sizing
            screen = QApplication.primaryScreen().geometry()
            width = int(screen.width() * 0.31)  # 31% of screen width
            height = int(screen.height() * 0.46)  # 46% of screen height
            selection_dialog.setMinimumSize(width, height)
            main_layout = QVBoxLayout(selection_dialog)
            
            # Title
            title_label = QLabel("Select Model Size to Load")
            title_font = QFont("Arial", 12, QFont.Weight.Bold)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(title_label)
            
            # Model selection frame
            model_frame = QGroupBox("Available Models")
            model_frame_font = QFont("Arial", 11, QFont.Weight.Bold)
            model_frame.setFont(model_frame_font)
            model_frame_layout = QVBoxLayout(model_frame)
            model_frame_layout.setContentsMargins(15, 15, 15, 15)
            model_frame_layout.setSpacing(10)
            
            # Model options
            model_options = {
                "1": {"name": "Qwen2-VL 2B", "desc": "Smallest (4-8GB VRAM)"},
                "2": {"name": "Qwen2-VL 7B", "desc": "Medium (12-16GB VRAM)"},
                "3": {"name": "Qwen2-VL 72B", "desc": "Largest (80GB+ VRAM)"},
                "4": {"name": "Custom Model", "desc": "Enter any HF model ID"},
            }
            
            # Store selected model
            selected_model_key = {"value": "1"}
            custom_model_id_text = {"value": ""}
            
            # Radio button group
            button_group = QButtonGroup(selection_dialog)
            
            for idx, (key, info) in enumerate(model_options.items()):
                # Radio button
                rb = QRadioButton(f"{info['name']} - {info['desc']}")
                rb_font = QFont("Arial", 10)
                rb.setFont(rb_font)
                if idx == 0:
                    rb.setChecked(True)
                rb.clicked.connect(lambda checked, k=key: selected_model_key.update({"value": k}))
                button_group.addButton(rb)
                model_frame_layout.addWidget(rb)
                
                # Separator
                if key != "4":
                    separator = QFrame()
                    separator.setFrameShape(QFrame.Shape.HLine)
                    separator.setFrameShadow(QFrame.Shadow.Sunken)
                    model_frame_layout.addWidget(separator)
            
            main_layout.addWidget(model_frame)
            
            # Custom model ID frame (initially hidden)
            custom_frame = QGroupBox("Custom Model Configuration")
            custom_frame_font = QFont("Arial", 11, QFont.Weight.Bold)
            custom_frame.setFont(custom_frame_font)
            custom_frame_layout = QHBoxLayout(custom_frame)
            custom_frame_layout.setContentsMargins(15, 15, 15, 15)
            
            custom_label = QLabel("Model ID:")
            custom_label_font = QFont("Arial", 10)
            custom_label.setFont(custom_label_font)
            custom_frame_layout.addWidget(custom_label)
            
            custom_entry = QLineEdit()
            custom_entry.setPlaceholderText("e.g., Qwen/Qwen2-VL-2B-Instruct")
            custom_entry.setFont(custom_label_font)
            custom_entry.textChanged.connect(lambda text: custom_model_id_text.update({"value": text}))
            custom_frame_layout.addWidget(custom_entry)
            
            custom_frame.hide()  # Hidden by default
            main_layout.addWidget(custom_frame)
            
            # Toggle custom frame visibility
            def toggle_custom_frame():
                if selected_model_key["value"] == "4":
                    custom_frame.show()
                else:
                    custom_frame.hide()
            
            for rb in button_group.buttons():
                rb.clicked.connect(toggle_custom_frame)
            
            # Buttons with centering
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            model_confirmed = {'value': False, 'size': None}
            
            def confirm_selection():
                selected = selected_model_key["value"]
                self._log(f"DEBUG: Radio button selection = {selected}")
                if selected == "4":
                    if not custom_model_id_text["value"].strip():
                        QMessageBox.critical(selection_dialog, "Error", "Please enter a model ID")
                        return
                    model_confirmed['size'] = f"custom:{custom_model_id_text['value'].strip()}"
                else:
                    model_confirmed['size'] = selected
                model_confirmed['value'] = True
                selection_dialog.accept()
            
            load_btn = QPushButton("Load")
            load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px 20px; font-weight: bold; }")
            load_btn.clicked.connect(confirm_selection)
            button_layout.addWidget(load_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setMinimumWidth(100)  # Ensure enough width for text
            cancel_btn.setStyleSheet("QPushButton { background-color: #9E9E9E; color: white; padding: 8px 20px; }")
            cancel_btn.clicked.connect(selection_dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            button_layout.addStretch()
            main_layout.addLayout(button_layout)
            
            # Show dialog and wait for result (PySide6 modal dialog)
            result = selection_dialog.exec()
            
            if result != QDialog.DialogCode.Accepted or not model_confirmed['value']:
                return
            
            model_size = model_confirmed['size']
            self._log(f"DEBUG: Dialog closed, model_size set to: {model_size}")
        
        # Create PySide6 progress dialog
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QGroupBox
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont
        
        progress_dialog = QDialog(self.dialog)
        progress_dialog.setWindowTitle(f"Setting up {provider}")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.21)  # 21% of screen width
        height = int(screen.height() * 0.19)  # 19% of screen height
        progress_dialog.setMinimumSize(width, height)
        progress_layout = QVBoxLayout(progress_dialog)
        
        # Progress section
        progress_section = QGroupBox("Setup Progress")
        progress_section_font = QFont("Arial", 11, QFont.Weight.Bold)
        progress_section.setFont(progress_section_font)
        progress_section_layout = QVBoxLayout(progress_section)
        progress_section_layout.setContentsMargins(15, 15, 15, 15)
        progress_section_layout.setSpacing(10)
        
        progress_label = QLabel("Initializing...")
        progress_label_font = QFont("Arial", 10)
        progress_label.setFont(progress_label_font)
        progress_section_layout.addWidget(progress_label)
        
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)  # Indeterminate mode
        progress_bar.setMinimumWidth(350)
        progress_section_layout.addWidget(progress_bar)
        
        status_label = QLabel("")
        status_label_font = QFont("Arial", 9)
        status_label.setFont(status_label_font)
        status_label.setStyleSheet("color: #666666;")
        progress_section_layout.addWidget(status_label)
        
        progress_layout.addWidget(progress_section)
        
        def update_progress(message, percent=None):
            """Update progress display (thread-safe)"""
            # Use lambda to ensure we capture the correct widget references
            def update_ui():
                progress_label.setText(message)
                if percent is not None:
                    progress_bar.setMaximum(100)  # Switch to determinate mode
                    progress_bar.setValue(int(percent))
            
            # Schedule on main thread
            self.update_queue.put(('call_method', update_ui, ()))
        
        def setup_thread():
            """Run setup in background thread"""
            nonlocal model_size
            print(f"\n=== SETUP THREAD STARTED for {provider} ===")
            print(f"Status: {status}")
            print(f"Model size: {model_size}")
            
            try:
                # Check if we need to install
                if not status['installed']:
                    # Install provider
                    print(f"Installing {provider}...")
                    update_progress(f"Installing {provider}...")
                    success = ocr_manager.install_provider(provider, update_progress)
                    print(f"Install result: {success}")
                    
                    if not success:
                        print("Installation FAILED")
                        update_progress("❌ Installation failed!", 0)
                        self._log(f"Failed to install {provider}", "error")
                        return
                else:
                    # Already installed, skip installation
                    print(f"{provider} dependencies already installed")
                    self._log(f"DEBUG: {provider} dependencies already installed")
                    success = True  # Mark as success since deps are ready
                
                # Load model
                print(f"About to load {provider} model...")
                update_progress(f"Loading {provider} model...")
                self._log(f"DEBUG: Loading provider {provider}, status['installed']={status.get('installed', False)}")
                
                # Special handling for Qwen2-VL - pass model_size
                if provider == 'Qwen2-VL':
                    if success and model_size:
                        # Save the model size to config
                        self.qwen2vl_model_size = model_size
                        self.main_gui.config['qwen2vl_model_size'] = model_size
                        
                        # Save config immediately
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                    self._log(f"DEBUG: In thread, about to load with model_size={model_size}")
                    if model_size:
                        try:
                            success = ocr_manager.load_provider(provider, model_size=model_size)
                            self._log(f"DEBUG: load_provider call completed, success={success}")
                        except Exception as load_error:
                            self._log(f"Exception during load_provider: {str(load_error)}", "error")
                            import traceback
                            self._log(f"Load error traceback: {traceback.format_exc()}", "debug")
                            success = False
                        
                        if success:
                            provider_obj = ocr_manager.get_provider('Qwen2-VL')
                            if provider_obj:
                                provider_obj.loaded_model_size = {
                                    "1": "2B",
                                    "2": "7B", 
                                    "3": "72B",
                                    "4": "custom"
                                }.get(model_size, model_size)
                                self._log(f"DEBUG: Set loaded_model_size to {provider_obj.loaded_model_size}")
                            else:
                                self._log("Warning: Could not get Qwen2-VL provider object after successful load", "warning")
                        else:
                            self._log(f"Failed to load Qwen2-VL model with size {model_size}", "error")
                    else:
                        self._log("Warning: No model size specified for Qwen2-VL, defaulting to 2B", "warning")
                        try:
                            success = ocr_manager.load_provider(provider, model_size="1")
                            self._log(f"DEBUG: Default load_provider call completed, success={success}")
                        except Exception as load_error:
                            self._log(f"Exception during default load_provider: {str(load_error)}", "error")
                            success = False
                else:
                    print(f"Loading {provider} without model_size parameter")
                    self._log(f"DEBUG: Loading {provider} without model_size parameter")
                    success = ocr_manager.load_provider(provider)
                    print(f"load_provider returned: {success}")
                    self._log(f"DEBUG: load_provider returned success={success}")
                
                print(f"\nFinal success value: {success}")
                if success:
                    print("SUCCESS! Model loaded successfully")
                    update_progress(f"✅ {provider} ready!", 100)
                    self._log(f"✅ {provider} is ready to use", "success")
                    # Schedule status check on main thread
                    self.update_queue.put(('call_method', self._check_provider_status, ()))
                else:
                    print("FAILED! Model did not load")
                    update_progress("❌ Failed to load model!", 0)
                    self._log(f"Failed to load {provider} model", "error")
                
            except Exception as e:
                print(f"\n!!! EXCEPTION CAUGHT !!!")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Traceback:\n{traceback_str}")
                
                error_msg = f"❌ Error: {str(e)}"
                update_progress(error_msg, 0)
                self._log(f"Setup error: {str(e)}", "error")
                self._log(traceback_str, "debug")
                # Don't close dialog on error - let user read the error
                return
            
            # Only close dialog on success
            if success:
                # Schedule dialog close on main thread after 2 seconds
                import time
                time.sleep(2)
                self.update_queue.put(('call_method', progress_dialog.close, ()))
            else:
                # On failure, keep dialog open so user can see the error
                import time
                time.sleep(5)
                self.update_queue.put(('call_method', progress_dialog.close, ()))
        
        # Show progress dialog (non-blocking)
        progress_dialog.show()
        
        # Start setup in background via executor if available
        try:
            if hasattr(self.main_gui, '_ensure_executor'):
                self.main_gui._ensure_executor()
            execu = getattr(self.main_gui, 'executor', None)
            if execu:
                execu.submit(setup_thread)
            else:
                import threading
                threading.Thread(target=setup_thread, daemon=True).start()
        except Exception:
            import threading
            threading.Thread(target=setup_thread, daemon=True).start()

    def _on_ocr_provider_change(self, event=None):
        """Handle OCR provider change"""
        # Get the new provider value from combo box
        if hasattr(self, 'provider_combo'):
            provider = self.provider_combo.currentText()
            self.ocr_provider_value = provider
        else:
            provider = self.ocr_provider_value
        
        # Hide ALL provider-specific frames first (PySide6)
        if hasattr(self, 'google_creds_frame'):
            self.google_creds_frame.setVisible(False)
        if hasattr(self, 'azure_frame'):
            self.azure_frame.setVisible(False)
        if hasattr(self, 'azure_doc_intel_frame'):
            self.azure_doc_intel_frame.setVisible(False)
        if hasattr(self, 'custom_api_ocr_batch_frame'):
            self.custom_api_ocr_batch_frame.setVisible(provider == 'custom-api')
        
        # Show only the relevant settings frame for the selected provider
        if provider == 'google':
            # Show Google credentials frame
            if hasattr(self, 'google_creds_frame'):
                self.google_creds_frame.setVisible(True)
            
        elif provider == 'azure':
            # Show Azure Computer Vision settings frame
            if hasattr(self, 'azure_frame'):
                self.azure_frame.setVisible(True)
        
        elif provider == 'azure-document-intelligence':
            # Show Azure Document Intelligence settings frame (separate)
            if hasattr(self, 'azure_doc_intel_frame'):
                self.azure_doc_intel_frame.setVisible(True)
            
        # For all other providers (manga-ocr, Qwen2-VL, easyocr, paddleocr, doctr)
        # Don't show any cloud credential frames - they use local models
        
        # Check provider status to show appropriate buttons
        self._check_provider_status()
        
        # Update the main status label at the top based on new provider
        self._update_main_status_label()
        
        # Log the change
        provider_descriptions = {
            'custom-api': "Custom API - use your own vision model",
            'google': "Google Cloud Vision (requires credentials)",
            'azure': "Azure Computer Vision (requires API key)",
            'azure-document-intelligence': "Azure Document Intelligence - successor to Azure AI Vision (requires API key)",
            'manga-ocr': "Manga OCR - optimized for Japanese manga",
            'rapidocr': "RapidOCR - fast local OCR with region detection",
            'Qwen2-VL': "Qwen2-VL - a big model", 
            'easyocr': "EasyOCR - multi-language support",
            'paddleocr': "PaddleOCR - CJK language support",
            'doctr': "DocTR - document text recognition"
        }
        
        self._log(f"📋 OCR provider changed to: {provider_descriptions.get(provider, provider)}", "info")
        
        # Save the selection (write BOTH keys to keep UIs in sync)
        self.main_gui.config['manga_ocr_provider'] = provider
        self.main_gui.config['ocr_provider'] = provider
        if hasattr(self, 'manga_ocr_disable_thinking_checkbox'):
            ms = self.main_gui.config.setdefault('manga_settings', {})
            ocr_set = ms.setdefault('ocr', {})
            ocr_set['manga_ocr_disable_thinking'] = bool(self.manga_ocr_disable_thinking_checkbox.isChecked())
            os.environ['MANGA_OCR_DISABLE_THINKING'] = '1' if self.manga_ocr_disable_thinking_checkbox.isChecked() else '0'
        if hasattr(self, 'custom_api_ocr_batch_checkbox'):
            self.main_gui.config['manga_custom_api_ocr_batch_enabled'] = bool(self.custom_api_ocr_batch_checkbox.isChecked())
        if hasattr(self, 'custom_api_ocr_batch_size_spinbox'):
            self.main_gui.config['manga_custom_api_ocr_batch_size'] = int(self.custom_api_ocr_batch_size_spinbox.value())
        if hasattr(self.main_gui, 'save_config'):
            self.main_gui.save_config(show_message=False)
        
        # IMPORTANT: Reset translator to force recreation with new OCR provider
        if hasattr(self, 'translator') and self.translator:
            self._log(f"OCR provider changed to {provider.upper()}. Translator will be recreated on next run.", "info")
            self.translator = None  # Force recreation on next translation

    def _open_multi_api_key_pool_preview(self, pool_name: str):
        """Open the multi-key manager with only one dedicated pool rendered."""
        try:
            from multi_api_key_manager import open_multi_api_key_pool_preview
            parent = getattr(self, 'dialog', None) or getattr(self, 'parent_widget', None)
            open_multi_api_key_pool_preview(parent, self.main_gui, pool_name)
        except Exception as exc:
            QMessageBox.critical(self.dialog if hasattr(self, 'dialog') else None, "Error", f"Failed to open key pool manager: {exc}")

    def _preview_pool_button_stylesheet(self) -> str:
        from multi_api_key_manager import preview_pool_button_stylesheet
        return preview_pool_button_stylesheet()

    def _style_preview_pool_button(self, button):
        from multi_api_key_manager import style_preview_pool_button
        style_preview_pool_button(button)
    
    def _update_main_status_label(self):
        """Update the main status label at the top based on current provider and credentials"""
        if not hasattr(self, 'status_label'):
            return
        
        # Get API key
        try:
            if hasattr(self.main_gui, 'api_key_entry'):
                if hasattr(self.main_gui.api_key_entry, 'text'):  # PySide6
                    has_api_key = bool(self.main_gui.api_key_entry.text().strip())
                elif hasattr(self.main_gui.api_key_entry, 'get'):  # Tkinter
                    has_api_key = bool(self.main_gui.api_key_entry.get().strip())
                else:
                    has_api_key = False
            else:
                has_api_key = False
        except:
            has_api_key = False
        
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        if not has_api_key:
            _model = self.main_gui.config.get('model', '') if hasattr(self, 'main_gui') else ''
            try:
                from unified_api_client import UnifiedClient as _UC
                if not _UC._model_needs_api_key(_model):
                    has_api_key = True
            except Exception:
                pass
        
        # Get current provider
        provider = self.ocr_provider_value if hasattr(self, 'ocr_provider_value') else self.main_gui.config.get('manga_ocr_provider', 'custom-api')
        if provider == 'custom-api':
            has_api_key = True
        
        # Determine readiness based on provider
        if provider == 'google':
            has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
            is_ready = has_api_key and has_vision
        elif provider == 'azure':
            has_azure = bool(self.main_gui.config.get('azure_vision_key', ''))
            is_ready = has_api_key and has_azure
        elif provider == 'azure-document-intelligence':
            # Azure Document Intelligence uses same credentials storage as Azure CV for now
            has_azure = bool(self.main_gui.config.get('azure_document_intelligence_key', '') or self.main_gui.config.get('azure_vision_key', ''))
            is_ready = has_api_key and has_azure
        else:
            # Local providers or custom-api only need API key for translation
            is_ready = has_api_key
        
        # Update label
        status_text = "✅ Ready" if is_ready else "❌ Setup Required"
        status_color = "green" if is_ready else "red"
        
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color};")
    
    def _build_interface(self):
        """Build the enhanced manga translation interface using PySide6"""
        # Create main layout for PySide6 widget
        main_layout = QVBoxLayout(self.parent_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)
        self._build_pyside6_interface(main_layout)

    @staticmethod
    def _pump_build_events():
        """Let the GUI breathe at section boundaries during the heavy
        interface build, so the host dialog's loading placeholder keeps
        animating instead of freezing. User input events are excluded to
        avoid re-entrancy while the UI is only half built."""
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QEventLoop
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        except Exception:
            pass
    
    def _build_pyside6_interface(self, main_layout):
        # Import QSizePolicy for layout management
        from PySide6.QtWidgets import QSizePolicy
        from PySide6.QtCore import QTimer
        
        # Apply global stylesheet for checkboxes and radio buttons
        checkbox_radio_style = """
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
            QRadioButton {
                color: white;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
                border: 2px solid #5a9fd4;
                border-radius: 7px;
                background-color: #2d2d2d;
            }
            QRadioButton::indicator:checked {
                background-color: #5a9fd4;
                border: 2px solid #5a9fd4;
            }
            QRadioButton::indicator:hover {
                border-color: #7bb3e0;
            }
            QRadioButton:disabled {
                color: #666666;
            }
            QRadioButton::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
            /* Disabled fields styling */
            QLineEdit:disabled, QComboBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QLabel:disabled {
                color: #666666;
            }
        """
        parent_style = getattr(self, '_translator_gui_stylesheet', '') or ''
        try:
            icon_base = getattr(self.main_gui, 'base_dir', '') or os.path.dirname(os.path.abspath(__file__))
            combo_arrow_path = os.path.join(icon_base, 'Halgakos.ico').replace('\\', '/')
        except Exception:
            combo_arrow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico').replace('\\', '/')
        combo_arrow_style = f"""
            QComboBox::down-arrow {{
                image: url("{combo_arrow_path}");
                width: 11px;
                height: 11px;
            }}
            QComboBox::down-arrow:on {{
                top: 1px;
            }}
        """
        self.parent_widget.setStyleSheet(
            f"{parent_style}\n{checkbox_radio_style}\n{combo_arrow_style}" if parent_style else f"{checkbox_radio_style}\n{combo_arrow_style}"
        )
        
        # Title (at the very top)
        title_frame = QWidget()
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        title_label = QLabel("🎌 Manga Translation")
        title_font = QFont("Arial", 13)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        # Requirements check - based on selected OCR provider
        try:
            if hasattr(self.main_gui, 'api_key_entry'):
                if hasattr(self.main_gui.api_key_entry, 'text'):  # PySide6
                    has_api_key = bool(self.main_gui.api_key_entry.text().strip())
                elif hasattr(self.main_gui.api_key_entry, 'get'):  # Tkinter
                    has_api_key = bool(self.main_gui.api_key_entry.get().strip())
                else:
                    has_api_key = False
            else:
                has_api_key = False
        except:
            has_api_key = False
        
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        if not has_api_key:
            _model = self.main_gui.config.get('model', '') if hasattr(self, 'main_gui') else ''
            try:
                from unified_api_client import UnifiedClient as _UC
                if not _UC._model_needs_api_key(_model):
                    has_api_key = True
            except Exception:
                pass
        
        # Get the saved OCR provider to check appropriate credentials
        saved_provider = self.main_gui.config.get('manga_ocr_provider', 'custom-api')
        if saved_provider == 'custom-api':
            has_api_key = True
        
        # Determine readiness based on provider
        if saved_provider == 'google':
            has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
            is_ready = has_api_key and has_vision
        elif saved_provider == 'azure':
            has_azure = bool(self.main_gui.config.get('azure_vision_key', ''))
            is_ready = has_api_key and has_azure
        else:
            # Local providers or custom-api only need API key for translation
            is_ready = has_api_key
        
        status_text = "✅ Ready" if is_ready else "❌ Setup Required"
        status_color = "green" if is_ready else "red"
        
        status_label = QLabel(status_text)
        status_font = QFont("Arial", 10)
        status_label.setFont(status_font)
        status_label.setStyleSheet(f"color: {status_color};")
        title_layout.addStretch()
        title_layout.addWidget(status_label)

        main_layout.addWidget(title_frame)
        self._pump_build_events()
        
        # Store reference for updates
        self.status_label = status_label
        
        # Model Preloading Progress Bar (initially hidden)
        self.preload_progress_frame = QWidget()
        self.preload_progress_frame.setStyleSheet(
            "background-color: #2d2d2d; "
            "border: 1px solid #4a5568; "
            "border-radius: 4px; "
            "padding: 6px;"
        )
        preload_layout = QVBoxLayout(self.preload_progress_frame)
        preload_layout.setContentsMargins(8, 6, 8, 6)
        preload_layout.setSpacing(4)
        
        self.preload_status_label = QLabel("Loading models...")
        preload_status_font = QFont("Segoe UI", 9)
        preload_status_font.setBold(True)
        self.preload_status_label.setFont(preload_status_font)
        self.preload_status_label.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        self.preload_status_label.setAlignment(Qt.AlignCenter)
        preload_layout.addWidget(self.preload_status_label)
        
        self.preload_progress_bar = QProgressBar()
        self.preload_progress_bar.setRange(0, 100)
        self.preload_progress_bar.setValue(0)
        self.preload_progress_bar.setTextVisible(True)
        self.preload_progress_bar.setMinimumHeight(22)
        self.preload_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4a5568;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
                font-weight: bold;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2d6a4f, stop:0.5 #1b4332, stop:1 #081c15);
                border-radius: 2px;
                margin: 0px;
            }
        """)
        preload_layout.addWidget(self.preload_progress_bar)
        
        self.preload_progress_frame.setVisible(False)  # Hidden by default
        main_layout.addWidget(self.preload_progress_frame)
        
        # Add instructions based on selected provider
        if not is_ready:
            req_frame = QWidget()
            req_layout = QVBoxLayout(req_frame)
            req_layout.setContentsMargins(0, 5, 0, 5)
            
            req_text = []
            if not has_api_key:
                req_text.append("• API Key not configured")
            
            # Only show provider-specific credential warnings
            if saved_provider == 'google':
                has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
                if not has_vision:
                    req_text.append("• Google Cloud Vision credentials not set")
            elif saved_provider == 'azure':
                has_azure = bool(self.main_gui.config.get('azure_vision_key', ''))
                if not has_azure:
                    req_text.append("• Azure credentials not configured")
            
            if req_text:  # Only show frame if there are actual missing requirements
                req_label = QLabel("\n".join(req_text))
                req_font = QFont("Arial", 10)
                req_label.setFont(req_font)
                req_label.setStyleSheet("color: red;")
                req_label.setAlignment(Qt.AlignLeft)
                req_layout.addWidget(req_label)
                main_layout.addWidget(req_frame)
        else:
            # Create empty frame to maintain layout consistency
            req_frame = QWidget()
            req_frame.setVisible(False)
            main_layout.addWidget(req_frame)
        
        # File selection frame - SPANS BOTH COLUMNS
        file_frame = QGroupBox("Select Manga Images")
        self.manga_file_frame = file_frame
        file_frame.setAcceptDrops(True)
        file_frame.installEventFilter(self)
        file_frame_font = QFont("Arial", 10)
        file_frame_font.setBold(True)
        file_frame.setFont(file_frame_font)
        file_frame_layout = QVBoxLayout(file_frame)
        file_frame_layout.setContentsMargins(10, 10, 10, 8)
        file_frame_layout.setSpacing(6)
        
        # Sorting buttons (first row - above file list)
        sort_btn_frame = QWidget()
        sort_btn_layout = QHBoxLayout(sort_btn_frame)
        sort_btn_layout.setContentsMargins(0, 0, 0, 0)
        sort_btn_layout.setSpacing(4)
        
        sort_label = QLabel("Sort:")
        sort_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        sort_btn_layout.addWidget(sort_label)
        
        # Number starts sorted ascending, so its next click reverses the order.
        self._sort_ascending = {'name': True, 'numeric': False, 'date': True}
        
        # Name sort button
        self.sort_name_btn = QPushButton("↑ Name")
        self.sort_name_btn.setToolTip("Sort by filename (A-Z). Click again to reverse.")
        self.sort_name_btn.clicked.connect(lambda: self._toggle_sort('name'))
        self.sort_name_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 4px 8px; font-size: 9pt; } QPushButton:hover { background-color: #5a6268; }")
        sort_btn_layout.addWidget(self.sort_name_btn)
        
        # Numeric sort button
        self.sort_numeric_btn = QPushButton("↑ Number")
        self.sort_numeric_btn.setToolTip("Sorted numerically (1, 2, 10). Click to reverse.")
        self.sort_numeric_btn.clicked.connect(lambda: self._toggle_sort('numeric'))
        self.sort_numeric_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 4px 8px; font-size: 9pt; } QPushButton:hover { background-color: #5a6268; }")
        sort_btn_layout.addWidget(self.sort_numeric_btn)
        
        # Date sort button
        self.sort_date_btn = QPushButton("↑ Date")
        self.sort_date_btn.setToolTip("Sort by file date (oldest first). Click again to reverse.")
        self.sort_date_btn.clicked.connect(lambda: self._toggle_sort('date'))
        self.sort_date_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 4px 8px; font-size: 9pt; } QPushButton:hover { background-color: #5a6268; }")
        sort_btn_layout.addWidget(self.sort_date_btn)
        
        # Reverse button
        reverse_btn = QPushButton("⇅ Reverse")
        reverse_btn.setToolTip("Reverse current order")
        reverse_btn.clicked.connect(lambda: self._sort_files('reverse'))
        reverse_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 4px 8px; font-size: 9pt; } QPushButton:hover { background-color: #5a6268; }")
        sort_btn_layout.addWidget(reverse_btn)
        
        sort_btn_layout.addStretch()
        file_frame_layout.addWidget(sort_btn_frame)

        range_frame = QWidget()
        range_layout = QHBoxLayout(range_frame)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(6)

        range_label = QLabel("Image Range:")
        range_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        range_layout.addWidget(range_label)

        self.manga_image_range_entry = QLineEdit()
        self.manga_image_range_entry.setPlaceholderText("blank = all, e.g. 3-8")
        self.manga_image_range_entry.setFixedWidth(150)
        self.manga_image_range_entry.setToolTip(
            "Process only these 1-based rows in the currently visible order. "
            "Supports ranges like 3-8 and lists like 1,3-5."
        )
        self.manga_image_range_entry.setStyleSheet(
            "QLineEdit { background-color: #2b2b2b; color: #ffffff; "
            "border: 1px solid #555555; padding: 4px 6px; }"
        )
        self.manga_image_range_entry.textChanged.connect(self._on_manga_image_range_changed)
        range_layout.addWidget(self.manga_image_range_entry, 0)

        self.manga_image_range_status_label = QLabel("All images")
        self.manga_image_range_status_label.setStyleSheet("color: #9fb7d5; font-size: 8pt;")
        range_layout.addWidget(self.manga_image_range_status_label)

        self.manga_process_nav_widget = QWidget()
        process_nav_layout = QHBoxLayout(self.manga_process_nav_widget)
        process_nav_layout.setContentsMargins(8, 0, 0, 0)
        process_nav_layout.setSpacing(6)

        self.manga_process_prev_btn = QPushButton("◀")
        self.manga_process_prev_btn.setFixedWidth(36)
        self.manga_process_prev_btn.setToolTip("Previous manga process group")
        self.manga_process_prev_btn.setStyleSheet(
            "QPushButton { background-color:#3a3a3a; color:white; font-weight:bold; "
            "font-size:13pt; border:1px solid #5a9fd4; border-radius:4px; padding:4px; }"
            "QPushButton:hover { background-color:#4a8fc4; }"
            "QPushButton:disabled { color:#666; background-color:#2a2a2a; }"
        )
        process_nav_layout.addWidget(self.manga_process_prev_btn)

        self.manga_process_combo = QComboBox()
        self.manga_process_combo.setMinimumWidth(240)
        self.manga_process_combo.setToolTip("Separate manga process groups")
        self.manga_process_combo.setStyleSheet(
            "QComboBox { background-color:#3a3a3a; color:white; font-weight:bold; "
            "font-size:10pt; padding:4px 8px; border:1px solid #5a9fd4; border-radius:4px; }"
            "QComboBox::drop-down { border:none; }"
            "QComboBox QAbstractItemView { background-color:#2d2d2d; color:white; "
            "selection-background-color:#5a9fd4; }"
        )
        self.manga_process_combo.currentIndexChanged.connect(self._on_manga_process_group_changed)
        process_nav_layout.addWidget(self.manga_process_combo, stretch=1)

        self.manga_process_counter_label = QLabel("1 / 1")
        self.manga_process_counter_label.setStyleSheet("color:#94a3b8; font-size:9pt; font-weight:bold;")
        self.manga_process_counter_label.setFixedWidth(52)
        self.manga_process_counter_label.setAlignment(Qt.AlignCenter)
        process_nav_layout.addWidget(self.manga_process_counter_label)

        self.manga_process_next_btn = QPushButton("▶")
        self.manga_process_next_btn.setFixedWidth(36)
        self.manga_process_next_btn.setToolTip("Next manga process group")
        self.manga_process_next_btn.setStyleSheet(self.manga_process_prev_btn.styleSheet())
        process_nav_layout.addWidget(self.manga_process_next_btn)

        self.manga_process_prev_btn.clicked.connect(
            lambda: self.manga_process_combo.setCurrentIndex(self.manga_process_combo.currentIndex() - 1)
        )
        self.manga_process_next_btn.clicked.connect(
            lambda: self.manga_process_combo.setCurrentIndex(self.manga_process_combo.currentIndex() + 1)
        )
        self.manga_process_nav_widget.setVisible(False)
        range_layout.addWidget(self.manga_process_nav_widget, stretch=1)
        range_layout.addStretch()
        file_frame_layout.addWidget(range_frame)

        self.manga_loaded_directory_label = QLabel("Loaded directory: none")
        self.manga_loaded_directory_label.setStyleSheet("color: #b8c7d9; font-size: 8pt;")
        self.manga_loaded_directory_label.setWordWrap(True)
        file_frame_layout.addWidget(self.manga_loaded_directory_label)

        grouping_frame = QWidget()
        grouping_layout = QHBoxLayout(grouping_frame)
        grouping_layout.setContentsMargins(0, 0, 0, 0)
        grouping_layout.setSpacing(8)
        grouping_label = QLabel("Process Grouping:")
        grouping_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        grouping_layout.addWidget(grouping_label)

        self.manga_split_first_level_subfolders_checkbox = self._create_styled_checkbox(
            "Process subfolders separately"
        )
        self.manga_split_first_level_subfolders_checkbox.setChecked(
            bool(getattr(
                self,
                'manga_split_first_level_subfolders_value',
                self.main_gui.config.get('manga_split_first_level_subfolders', False)
            ))
        )
        self.manga_split_first_level_subfolders_checkbox.setToolTip(
            "OFF: Folders under the same manga directory are treated as one manga. "
            "One glossary is generated from OCR across all of those subfolders.\n\n"
            "ON: Each immediate volume/subfolder under a selected manga directory is treated separately. "
            "Each gets its own OCR pass and its own glossary.\n\n"
            "Nested folders inside a volume/subfolder still stay together with that volume."
        )
        self.manga_split_first_level_subfolders_checkbox.stateChanged.connect(
            self._on_manga_split_first_level_subfolders_toggle
        )
        grouping_layout.addWidget(self.manga_split_first_level_subfolders_checkbox)
        grouping_layout.addStretch()
        file_frame_layout.addWidget(grouping_frame)

        # File listbox (QListWidget handles scrolling automatically)
        self.file_listbox = QListWidget()
        self.file_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_listbox.setMinimumHeight(250)
        self.file_listbox.setStyleSheet(
            "QListWidget { background-color: #17191c; color: #d8dee7; "
            "border: 1px solid #3a4149; border-radius: 4px; outline: none; }"
            "QListWidget[mangaDropActive=\"true\"] { background-color: #1b2730; "
            "border: 2px dashed #62c8ff; }"
            "QListWidget::item { background-color: #23262a; color: #d8dee7; "
            "padding: 1px 8px; border-left: 4px solid transparent; "
            "border-bottom: 1px solid #17191c; }"
            "QListWidget::item:hover:!selected { background-color: #30363d; color: #ffffff; }"
            "QListWidget::item:selected { background-color: #175f86; color: #ffffff; "
            "border-left: 4px solid #62c8ff; font-weight: bold; }"
            "QListWidget::item:selected:!active { background-color: #175f86; color: #ffffff; "
            "border-left: 4px solid #62c8ff; }"
        )
        # Enable drag and drop reordering
        self.file_listbox.setDragDropMode(QListWidget.InternalMove)
        self.file_listbox.setDefaultDropAction(Qt.MoveAction)
        self.file_listbox.setAcceptDrops(True)
        self.file_listbox.installEventFilter(self)
        self.file_listbox.viewport().setAcceptDrops(True)
        self.file_listbox.viewport().installEventFilter(self)
        self._manga_drop_targets = {
            file_frame,
            self.file_listbox,
            self.file_listbox.viewport(),
        }
        self._file_selection_editing_enabled = True
        # Connect model changed signal to sync selected_files list
        self.file_listbox.model().rowsMoved.connect(self._on_files_reordered)
        file_frame_layout.addWidget(self.file_listbox)
        
        # File management buttons (second row - below file list)
        file_btn_frame = QWidget()
        file_btn_layout = QHBoxLayout(file_btn_frame)
        file_btn_layout.setContentsMargins(0, 4, 0, 0)
        file_btn_layout.setSpacing(4)
        
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self._add_files)
        add_files_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 4px 10px; font-size: 10pt; font-weight: bold; }")
        self.add_files_btn = add_files_btn
        file_btn_layout.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self._add_folder)
        add_folder_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 4px 10px; font-size: 10pt; font-weight: bold; }")
        self.add_folder_btn = add_folder_btn
        file_btn_layout.addWidget(add_folder_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        remove_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 4px 10px; font-size: 10pt; font-weight: bold; }")
        self.remove_files_btn = remove_btn
        file_btn_layout.addWidget(remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        clear_btn.setStyleSheet("QPushButton { background-color: #ffc107; color: black; padding: 4px 10px; font-size: 10pt; font-weight: bold; }")
        self.clear_files_btn = clear_btn
        file_btn_layout.addWidget(clear_btn)
        
        file_btn_layout.addStretch()
        file_frame_layout.addWidget(file_btn_frame)
        
        main_layout.addWidget(file_frame)
        self._pump_build_events()
        
        # Connect file list selection to image preview
        self.file_listbox.itemSelectionChanged.connect(self._on_file_selection_changed)
        self.file_listbox.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_listbox.customContextMenuRequested.connect(self._show_file_list_context_menu)
        
        # Add method to main_gui for parallel processing thread-safe GUI updates
        def _execute_parallel_gui_update(region_index: int, trans_text: str) -> bool:
            """Execute GUI update on main thread for parallel processing (thread-safe)"""
            try:
                # This runs on the main thread, so it's safe to call GUI methods
                return ImageRenderer._update_single_text_overlay(self, region_index, trans_text)
            except Exception as e:
                print(f"[PARALLEL] GUI update failed: {e}")
                return False
        
        # Dynamically add the method to main_gui so it can be invoked by parallel workers
        self.main_gui._execute_parallel_gui_update = _execute_parallel_gui_update
        
        # Create layout for settings (always tabs) + image preview
        columns_container = QWidget()
        columns_layout = QHBoxLayout(columns_container)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(10)
        
        # Create a container for tabs + advanced settings button
        tabs_container = QWidget()
        tabs_container_layout = QVBoxLayout(tabs_container)
        tabs_container_layout.setContentsMargins(0, 0, 0, 0)
        tabs_container_layout.setSpacing(0)
        
        # Create tab widget for settings
        from PySide6.QtWidgets import QTabWidget, QScrollArea
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.settings_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: white;
                padding: 8px 16px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #5a9fd4;
                border-color: #7bb3e0;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3a3a3a;
            }
        """)
        
        # Add Advanced Settings button to the tab bar's corner
        advanced_settings_btn_corner = QPushButton("⚙️ Advanced Settings")
        advanced_settings_btn_corner.clicked.connect(self._open_advanced_settings)
        advanced_settings_btn_corner.setMinimumHeight(34)
        advanced_settings_btn_corner.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                padding: 7px 16px 9px 16px;
                margin-top: 2px;
                margin-bottom: 2px;
                font-weight: bold;
                font-size: 9pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4b92ff;
            }
            QPushButton:pressed {
                background-color: #2b72e6;
            }
        """)
        self.settings_tabs.setCornerWidget(advanced_settings_btn_corner, Qt.TopRightCorner)
        
        # Left column (Column 1 - Translation Settings) - for tab content
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 15, 0, 0)  # Added 15px top padding
        left_column_layout.setSpacing(6)
        
        # Right column (Column 2 - Rendering Settings) - for tab content
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        right_column_layout.setContentsMargins(0, 15, 0, 0)  # Added 15px top padding
        right_column_layout.setSpacing(6)
        
        # Right column (Column 3 - Image Preview & Editing) - Always Visible
        from manga_image_preview import MangaImagePreviewWidget
        
        # Image preview widget (always visible) - pass main_gui for config persistence
        self.image_preview_widget = MangaImagePreviewWidget(main_gui=self.main_gui)
        self.image_preview_widget.setMinimumWidth(600)  # Increased from 350 to 600 for better viewing
        self.image_preview_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Store reference to manga_integration for clearing overlays
        self.image_preview_widget.manga_integration = self
        self.image_preview_widget.viewer.manga_integration = self
        
        # Connect image preview workflow signals to translation methods
        self.image_preview_widget.detect_text_clicked.connect(lambda: ImageRenderer._on_detect_text_clicked(self))
        self.image_preview_widget.clean_image_clicked.connect(lambda: ImageRenderer._on_clean_image_clicked(self))
        self.image_preview_widget.recognize_text_clicked.connect(lambda: ImageRenderer._on_recognize_text_clicked(self))
        self.image_preview_widget.translate_text_clicked.connect(lambda: ImageRenderer._on_translate_text_clicked(self))
        self.image_preview_widget.import_ocr_clicked.connect(self._import_manual_ocr_text)
        self.image_preview_widget.export_ocr_clicked.connect(self._export_manual_ocr_text)
        self.image_preview_widget.translate_all_clicked.connect(lambda: ImageRenderer._on_translate_all_clicked(self))
        
        # Connect viewer click signal to sync file list selection with current image
        self.image_preview_widget.viewer.viewer_clicked.connect(self._sync_file_selection_to_preview)
        
        # Settings frame - GOES TO LEFT COLUMN
        settings_frame = QGroupBox("Translation Settings")
        settings_frame_font = QFont("Arial", 10)
        settings_frame_font.setBold(True)
        settings_frame.setFont(settings_frame_font)
        settings_frame_layout = QVBoxLayout(settings_frame)
        settings_frame_layout.setContentsMargins(10, 15, 10, 15)  # Increased vertical padding from 10/8 to 15/15
        settings_frame_layout.setSpacing(6)
        
        # API Settings - Hybrid approach
        api_frame = QWidget()
        api_layout = QHBoxLayout(api_frame)
        api_layout.setContentsMargins(0, 0, 0, 10)
        api_layout.setSpacing(10)
        
        api_label = QLabel("OCR: Google Cloud Vision | Translation: API Key")
        api_font = QFont("Arial", 10)
        api_font.setItalic(True)
        api_label.setFont(api_font)
        api_label.setStyleSheet("color: gray;")
        api_layout.addWidget(api_label)
        
        # Show current model from main GUI
        current_model = 'Unknown'
        try:
            if hasattr(self.main_gui, 'model_combo'):
                if hasattr(self.main_gui.model_combo, 'currentText'):  # PySide6
                    current_model = self.main_gui.model_combo.currentText()
                elif hasattr(self.main_gui.model_combo, 'get'):  # Tkinter
                    current_model = self.main_gui.model_combo.get()
            elif hasattr(self.main_gui, 'model_var'):
                # Variable attribute
                current_model = self.main_gui.model_var if isinstance(self.main_gui.model_var, str) else str(self.main_gui.model_var)
            elif hasattr(self.main_gui, 'config'):
                # Fallback to config
                current_model = self.main_gui.config.get('model', 'Unknown')
        except Exception as e:
            print(f"Error getting model: {e}")
            current_model = 'Unknown'
        
        api_layout.addStretch()
        
        settings_frame_layout.addWidget(api_frame)
        self._pump_build_events()

        # OCR Provider Selection - ENHANCED VERSION
        self.ocr_provider_frame = QWidget()
        ocr_provider_layout = QHBoxLayout(self.ocr_provider_frame)
        ocr_provider_layout.setContentsMargins(0, 0, 0, 14)
        ocr_provider_layout.setSpacing(10)
        ocr_label_column_width = 105

        provider_label = QLabel("OCR Provider:")
        provider_label.setFixedWidth(ocr_label_column_width)
        provider_label.setAlignment(Qt.AlignLeft)
        ocr_provider_layout.addWidget(provider_label)

        # Expanded provider list with descriptions
        ocr_providers = [
            ('custom-api', 'Your Own key'),
            ('google', 'Google Cloud Vision'),
            ('azure', 'Azure Computer Vision'),
            ('azure-document-intelligence', '📋 Azure Document Intelligence (successor to Azure AI Vision)'),
            ('rapidocr', '⚡ RapidOCR (Fast & Local)'),
            ('manga-ocr', '🇯🇵 Manga OCR (Japanese)'),
            ('Qwen2-VL', '🇰🇷 Qwen2-VL (Korean)'),
            ('easyocr', '🌏 EasyOCR (Multi-lang)'),
            #('paddleocr', '🐼 PaddleOCR'),
            ('doctr', '📄 DocTR'),
        ]

        # Just the values for the combobox
        provider_values = [p[0] for p in ocr_providers]
        provider_display = [f"{p[0]} - {p[1]}" for p in ocr_providers]

        # Resolve initial provider with robust fallback to avoid accidental 'custom-api'
        # Prefer explicit manga_ocr_provider, then generic ocr_provider, then default
        self.ocr_provider_value = (
            self.main_gui.config.get('manga_ocr_provider')
            or self.main_gui.config.get('ocr_provider')
            or 'custom-api'
        )
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(provider_values)
        self.provider_combo.setCurrentText(self.ocr_provider_value)
        self.provider_combo.setMinimumWidth(120)  # Reduced for better fit
        self.provider_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.provider_combo.currentTextChanged.connect(self._on_ocr_provider_change)
        self._disable_combobox_mousewheel(self.provider_combo)  # Disable mousewheel scrolling
        ocr_provider_layout.addWidget(self.provider_combo)

        # Provider status indicator with more detail
        self.provider_status_label = QLabel("")
        status_font = QFont("Arial", 9)
        self.provider_status_label.setFont(status_font)
        self.provider_status_label.setWordWrap(True)  # Allow text wrapping
        self.provider_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ocr_provider_layout.addWidget(self.provider_status_label)

        # Setup/Install button for non-cloud providers
        self.provider_setup_btn = QPushButton("Setup")
        self.provider_setup_btn.clicked.connect(self._setup_ocr_provider)
        self.provider_setup_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        self.provider_setup_btn.setMinimumWidth(100)
        self.provider_setup_btn.setVisible(False)  # Hidden by default, _check_provider_status will show it
        ocr_provider_layout.addWidget(self.provider_setup_btn)

        # Add explicit download button for Hugging Face models
        self.ocr_download_model_btn = QPushButton("Download")
        self.ocr_download_model_btn.clicked.connect(self._download_hf_model)
        self.ocr_download_model_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; }")
        self.ocr_download_model_btn.setMinimumWidth(150)
        self.ocr_download_model_btn.setVisible(False)  # Hidden by default
        ocr_provider_layout.addWidget(self.ocr_download_model_btn)
        
        ocr_provider_layout.addStretch()
        settings_frame_layout.addWidget(self.ocr_provider_frame)

        self.custom_api_ocr_batch_frame = QWidget()
        custom_api_batch_layout = QHBoxLayout(self.custom_api_ocr_batch_frame)
        custom_api_batch_layout.setContentsMargins(0, 0, 0, 14)
        custom_api_batch_layout.setSpacing(10)

        batch_label = QLabel("Custom API OCR:")
        batch_label.setFixedWidth(ocr_label_column_width)
        batch_label.setAlignment(Qt.AlignLeft)
        custom_api_batch_layout.addWidget(batch_label)

        self.manga_ocr_disable_thinking_checkbox = self._create_styled_checkbox("Disable all thinking")
        self.manga_ocr_disable_thinking_checkbox.setChecked(bool(getattr(self, 'manga_ocr_disable_thinking_value', True)))
        self.manga_ocr_disable_thinking_checkbox.setToolTip("Applies only to custom-api manga OCR requests. Removes thinking params for faster OCR.")
        self.manga_ocr_disable_thinking_checkbox.stateChanged.connect(
            lambda: (
                setattr(self, 'manga_ocr_disable_thinking_value', self.manga_ocr_disable_thinking_checkbox.isChecked()),
                self._save_rendering_settings()
            )
        )
        custom_api_batch_layout.addWidget(self.manga_ocr_disable_thinking_checkbox)

        self.custom_api_ocr_batch_checkbox = self._create_styled_checkbox("Batch OCR requests")
        self.custom_api_ocr_batch_checkbox.setChecked(bool(getattr(self, 'custom_api_ocr_batch_enabled_value', True)))
        self.custom_api_ocr_batch_checkbox.setToolTip("Run custom-api OCR region requests in parallel. This is separate from translation batch mode.")
        self.custom_api_ocr_batch_checkbox.stateChanged.connect(
            lambda: (
                setattr(self, 'custom_api_ocr_batch_enabled_value', self.custom_api_ocr_batch_checkbox.isChecked()),
                self._save_rendering_settings()
            )
        )
        custom_api_batch_layout.addWidget(self.custom_api_ocr_batch_checkbox)

        self.custom_api_ocr_batch_size_spinbox = QSpinBox()
        self.custom_api_ocr_batch_size_spinbox.setRange(1, 32)
        self.custom_api_ocr_batch_size_spinbox.setValue(int(getattr(self, 'custom_api_ocr_batch_size_value', 5)))
        self.custom_api_ocr_batch_size_spinbox.setToolTip("Maximum concurrent custom-api OCR requests.")
        self.custom_api_ocr_batch_size_spinbox.valueChanged.connect(
            lambda value: (
                setattr(self, 'custom_api_ocr_batch_size_value', int(value)),
                self._save_rendering_settings()
            )
        )
        self._disable_combobox_mousewheel(self.custom_api_ocr_batch_size_spinbox)
        custom_api_batch_layout.addWidget(self.custom_api_ocr_batch_size_spinbox)

        self.vision_key_pool_btn = QPushButton("Vision Keys")
        self.vision_key_pool_btn.setToolTip("Open the Multi API Key Manager focused on the Vision key pool.")
        self._style_preview_pool_button(self.vision_key_pool_btn)
        self.vision_key_pool_btn.clicked.connect(lambda: self._open_multi_api_key_pool_preview('qa_scan'))
        custom_api_batch_layout.addWidget(self.vision_key_pool_btn)

        custom_api_batch_layout.addStretch()
        settings_frame_layout.addWidget(self.custom_api_ocr_batch_frame)
        self.custom_api_ocr_batch_frame.setVisible(self.ocr_provider_value == 'custom-api')

        # Initialize OCR manager
        from ocr_manager import OCRManager
        self.ocr_manager = OCRManager(log_callback=self._log)

        # Check initial provider status
        self._check_provider_status()

        # Google Cloud Credentials section (now in a frame that can be hidden)
        self.google_creds_frame = QWidget()
        google_creds_layout = QHBoxLayout(self.google_creds_frame)
        google_creds_layout.setContentsMargins(0, 0, 0, 10)
        google_creds_layout.setSpacing(10)

        google_label = QLabel("Google Cloud Credentials:")
        google_label.setMinimumWidth(150)
        google_label.setAlignment(Qt.AlignLeft)
        google_creds_layout.addWidget(google_label)

        # Show current credentials file
        google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        creds_display = os.path.basename(google_creds_path) if google_creds_path else "Not Set"

        self.creds_label = QLabel(creds_display)
        creds_font = QFont("Arial", 9)
        self.creds_label.setFont(creds_font)
        self.creds_label.setStyleSheet(f"color: {'green' if google_creds_path else 'red'};")
        google_creds_layout.addWidget(self.creds_label)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_google_credentials_permanent)
        browse_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 5px 15px; }")
        google_creds_layout.addWidget(browse_btn)
        
        google_creds_layout.addStretch()
        settings_frame_layout.addWidget(self.google_creds_frame)
        self.google_creds_frame.setVisible(False)  # Hidden by default

        # Azure settings frame (hidden by default)
        self.azure_frame = QWidget()
        azure_frame_layout = QVBoxLayout(self.azure_frame)
        azure_frame_layout.setContentsMargins(0, 0, 0, 10)
        azure_frame_layout.setSpacing(5)

        # Azure Key
        azure_key_frame = QWidget()
        azure_key_layout = QHBoxLayout(azure_key_frame)
        azure_key_layout.setContentsMargins(0, 0, 0, 0)
        azure_key_layout.setSpacing(10)

        azure_key_label = QLabel("Azure Key:")
        azure_key_label.setMinimumWidth(150)
        azure_key_label.setAlignment(Qt.AlignLeft)
        azure_key_layout.addWidget(azure_key_label)
        
        self.azure_key_entry = QLineEdit()
        self.azure_key_entry.setEchoMode(QLineEdit.Password)
        self.azure_key_entry.setMinimumWidth(150)  # Reduced for better fit
        self.azure_key_entry.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.azure_key_entry.textChanged.connect(self._on_azure_credentials_change)
        azure_key_layout.addWidget(self.azure_key_entry)

        # Show/Hide button for Azure key
        self.show_azure_key_checkbox = self._create_styled_checkbox("Show")
        self.show_azure_key_checkbox.stateChanged.connect(self._toggle_azure_key_visibility)
        azure_key_layout.addWidget(self.show_azure_key_checkbox)
        azure_key_layout.addStretch()
        azure_frame_layout.addWidget(azure_key_frame)

        # Azure Endpoint
        azure_endpoint_frame = QWidget()
        azure_endpoint_layout = QHBoxLayout(azure_endpoint_frame)
        azure_endpoint_layout.setContentsMargins(0, 0, 0, 0)
        azure_endpoint_layout.setSpacing(10)

        azure_endpoint_label = QLabel("Azure Endpoint:")
        azure_endpoint_label.setMinimumWidth(150)
        azure_endpoint_label.setAlignment(Qt.AlignLeft)
        azure_endpoint_layout.addWidget(azure_endpoint_label)
        
        self.azure_endpoint_entry = QLineEdit()
        self.azure_endpoint_entry.setMinimumWidth(150)  # Reduced for better fit
        self.azure_endpoint_entry.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.azure_endpoint_entry.textChanged.connect(self._on_azure_credentials_change)
        azure_endpoint_layout.addWidget(self.azure_endpoint_entry)
        azure_endpoint_layout.addStretch()
        azure_frame_layout.addWidget(azure_endpoint_frame)

        # Load saved Azure settings
        saved_key = self.main_gui.config.get('azure_vision_key', '')
        saved_endpoint = self.main_gui.config.get('azure_vision_endpoint', 'https://YOUR-RESOURCE.cognitiveservices.azure.com/')
        self.azure_key_entry.setText(saved_key)
        self.azure_endpoint_entry.setText(saved_endpoint)
        
        settings_frame_layout.addWidget(self.azure_frame)
        self.azure_frame.setVisible(False)  # Hidden by default

        # Azure Document Intelligence settings frame (separate from Azure CV)
        self.azure_doc_intel_frame = QWidget()
        azure_doc_intel_layout = QVBoxLayout(self.azure_doc_intel_frame)
        azure_doc_intel_layout.setContentsMargins(0, 0, 0, 10)
        azure_doc_intel_layout.setSpacing(5)

        # Azure Document Intelligence Key
        azure_doc_key_frame = QWidget()
        azure_doc_key_layout = QHBoxLayout(azure_doc_key_frame)
        azure_doc_key_layout.setContentsMargins(0, 0, 0, 0)
        azure_doc_key_layout.setSpacing(10)

        azure_doc_key_label = QLabel("Document Intelligence Key:")
        azure_doc_key_label.setMinimumWidth(150)
        azure_doc_key_label.setAlignment(Qt.AlignLeft)
        azure_doc_key_layout.addWidget(azure_doc_key_label)
        
        self.azure_doc_intel_key_entry = QLineEdit()
        self.azure_doc_intel_key_entry.setEchoMode(QLineEdit.Password)
        self.azure_doc_intel_key_entry.setMinimumWidth(150)
        self.azure_doc_intel_key_entry.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.azure_doc_intel_key_entry.textChanged.connect(self._on_azure_doc_intel_credentials_change)
        azure_doc_key_layout.addWidget(self.azure_doc_intel_key_entry)

        # Show/Hide button for Azure Document Intelligence key
        self.show_azure_doc_key_checkbox = self._create_styled_checkbox("Show")
        self.show_azure_doc_key_checkbox.stateChanged.connect(self._toggle_azure_doc_intel_key_visibility)
        azure_doc_key_layout.addWidget(self.show_azure_doc_key_checkbox)
        azure_doc_key_layout.addStretch()
        azure_doc_intel_layout.addWidget(azure_doc_key_frame)

        # Azure Document Intelligence Endpoint
        azure_doc_endpoint_frame = QWidget()
        azure_doc_endpoint_layout = QHBoxLayout(azure_doc_endpoint_frame)
        azure_doc_endpoint_layout.setContentsMargins(0, 0, 0, 0)
        azure_doc_endpoint_layout.setSpacing(10)

        azure_doc_endpoint_label = QLabel("Document Intelligence Endpoint:")
        azure_doc_endpoint_label.setMinimumWidth(150)
        azure_doc_endpoint_label.setAlignment(Qt.AlignLeft)
        azure_doc_endpoint_layout.addWidget(azure_doc_endpoint_label)
        
        self.azure_doc_intel_endpoint_entry = QLineEdit()
        self.azure_doc_intel_endpoint_entry.setPlaceholderText("https://your-resource.cognitiveservices.azure.com/")
        self.azure_doc_intel_endpoint_entry.setMinimumWidth(150)
        self.azure_doc_intel_endpoint_entry.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.azure_doc_intel_endpoint_entry.textChanged.connect(self._on_azure_doc_intel_credentials_change)
        azure_doc_endpoint_layout.addWidget(self.azure_doc_intel_endpoint_entry)
        azure_doc_endpoint_layout.addStretch()
        azure_doc_intel_layout.addWidget(azure_doc_endpoint_frame)

        # Load saved Azure Document Intelligence settings
        saved_doc_key = self.main_gui.config.get('azure_document_intelligence_key', '')
        saved_doc_endpoint = self.main_gui.config.get('azure_document_intelligence_endpoint', '')
        self.azure_doc_intel_key_entry.setText(saved_doc_key)
        self.azure_doc_intel_endpoint_entry.setText(saved_doc_endpoint)
        
        settings_frame_layout.addWidget(self.azure_doc_intel_frame)
        self.azure_doc_intel_frame.setVisible(False)  # Hidden by default
        self._pump_build_events()

        # Initially show/hide based on saved provider
        self._on_ocr_provider_change()

        # Separator for context settings
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        settings_frame_layout.addWidget(separator1)
        
        # Context and Full Page Mode Settings
        context_frame = QGroupBox("🔄 Context & Translation Mode")
        context_frame_font = QFont("Arial", 11)
        context_frame_font.setBold(True)
        context_frame.setFont(context_frame_font)
        context_frame_layout = QVBoxLayout(context_frame)
        context_frame_layout.setContentsMargins(10, 10, 10, 10)
        context_frame_layout.setSpacing(10)
        
        # Show current contextual settings from main GUI
        context_info = QWidget()
        context_info_layout = QVBoxLayout(context_info)
        context_info_layout.setContentsMargins(0, 0, 0, 10)
        context_info_layout.setSpacing(5)
        
        context_title = QLabel("Current Main GUI Settings:")
        title_font = QFont("Arial", 10)
        title_font.setBold(True)
        context_title.setFont(title_font)
        context_info_layout.addWidget(context_title)
        
        # Display current settings
        settings_frame_display = QWidget()
        settings_display_layout = QVBoxLayout(settings_frame_display)
        settings_display_layout.setContentsMargins(20, 0, 0, 0)
        settings_display_layout.setSpacing(3)
        
        # Contextual enabled status
        contextual_status = "Enabled" if self.main_gui.contextual_var else "Disabled"
        self.contextual_status_label = QLabel(f"• Contextual Translation: {contextual_status}")
        status_font = QFont("Arial", 10)
        self.contextual_status_label.setFont(status_font)
        settings_display_layout.addWidget(self.contextual_status_label)
        
        # History limit - handle QLineEdit widget properly
        history_limit = "3"  # default
        if hasattr(self.main_gui, 'trans_history'):
            try:
                # If it's a QLineEdit widget, get its text content
                if hasattr(self.main_gui.trans_history, 'text'):
                    history_limit = self.main_gui.trans_history.text()
                else:
                    history_limit = str(self.main_gui.trans_history)
            except Exception:
                history_limit = "3"
        self.history_limit_label = QLabel(f"• Translation History Limit: {history_limit} exchanges")
        self.history_limit_label.setFont(status_font)
        settings_display_layout.addWidget(self.history_limit_label)
        
        # Rolling history status
        rolling_status = "Enabled (Rolling Window)"
        self.rolling_status_label = QLabel(f"• Rolling History: {rolling_status}")
        self.rolling_status_label.setFont(status_font)
        settings_display_layout.addWidget(self.rolling_status_label)
        
        context_info_layout.addWidget(settings_frame_display)
        
        # API Settings Status within context section
        api_status_frame = QWidget()
        api_status_layout = QVBoxLayout(api_status_frame)
        api_status_layout.setContentsMargins(20, 0, 0, 0)
        api_status_layout.setSpacing(3)
        
        # Show current model from main GUI
        current_model = 'Unknown'
        try:
            if hasattr(self.main_gui, 'model_combo'):
                if hasattr(self.main_gui.model_combo, 'currentText'):  # PySide6
                    current_model = self.main_gui.model_combo.currentText()
                elif hasattr(self.main_gui.model_combo, 'get'):  # Tkinter
                    current_model = self.main_gui.model_combo.get()
            elif hasattr(self.main_gui, 'model_var'):
                # Variable attribute
                current_model = self.main_gui.model_var if isinstance(self.main_gui.model_var, str) else str(self.main_gui.model_var)
            elif hasattr(self.main_gui, 'config'):
                # Fallback to config
                current_model = self.main_gui.config.get('model', 'Unknown')
        except Exception as e:
            print(f"Error getting model: {e}")
            current_model = 'Unknown'
        
        # Store as instance variable so refresh can update it
        self.model_label = QLabel(f"• Model: {current_model}")
        model_font = QFont("Arial", 10)
        self.model_label.setFont(model_font)
        api_status_layout.addWidget(self.model_label)
        
        # Multi-key status
        multi_key_enabled = self.main_gui.config.get('use_multi_api_keys', False)
        multi_key_text = "• Multi-Key: ON" if multi_key_enabled else "• Multi-Key: OFF"
        multi_key_color = "green" if multi_key_enabled else "gray"
        
        self.multi_key_label = QLabel(multi_key_text)
        self.multi_key_label.setFont(model_font)
        self.multi_key_label.setStyleSheet(f"color: {multi_key_color};")
        api_status_layout.addWidget(self.multi_key_label)
        
        
        context_info_layout.addWidget(api_status_frame)
        context_frame_layout.addWidget(context_info)

        # Refresh button to update from main GUI
        self.refresh_btn = QPushButton("↻ Refresh from Main GUI")
        self.refresh_btn.clicked.connect(self._refresh_context_settings_with_feedback)
        self.refresh_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        context_frame_layout.addWidget(self.refresh_btn)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        context_frame_layout.addWidget(separator2)
        
        # Full Page Context Translation Settings
        full_page_frame = QWidget()
        full_page_layout = QVBoxLayout(full_page_frame)
        full_page_layout.setContentsMargins(0, 0, 0, 0)
        full_page_layout.setSpacing(5)

        full_page_title = QLabel("Full Page Context Mode (Manga-specific):")
        title_font2 = QFont("Arial", 10)
        title_font2.setBold(True)
        full_page_title.setFont(title_font2)
        full_page_layout.addWidget(full_page_title)

        # Enable/disable toggle
        # Use value loaded in _load_rendering_settings during startup
        toggle_frame = QWidget()
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(20, 0, 0, 0)
        toggle_layout.setSpacing(10)

        self.context_checkbox = self._create_styled_checkbox("Enable Full Page Context Translation")
        self.context_checkbox.setChecked(bool(getattr(self, 'full_page_context_value', self.main_gui.config.get('manga_full_page_context', True))))
        self.context_checkbox.stateChanged.connect(self._on_context_toggle)
        toggle_layout.addWidget(self.context_checkbox)

        # Edit prompt button
        edit_prompt_btn = QPushButton("Edit Prompt")
        edit_prompt_btn.clicked.connect(self._edit_context_prompt)
        edit_prompt_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        toggle_layout.addWidget(edit_prompt_btn)

        # Help button for full page context
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(30)
        help_btn.clicked.connect(lambda: self._show_help_dialog(
            "Full Page Context Mode",
            "Full page context sends all text regions from the page together in a single request.\n\n"
            "This allows the AI to see all text at once for more contextually accurate translations, "
            "especially useful for maintaining character name consistency and understanding "
            "conversation flow across multiple speech bubbles.\n\n"
            "✅ Pros:\n"
            "• Better context awareness\n"
            "• Consistent character names\n"
            "• Understanding of conversation flow\n"
            "• Maintains tone across bubbles\n\n"
            "❌ Cons:\n"
            "• Single API call failure affects all text\n"
            "• May use more tokens\n"
            "• Slower for pages with many text regions"
        ))
        help_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px; }")
        toggle_layout.addWidget(help_btn)
        toggle_layout.addStretch()
        
        full_page_layout.addWidget(toggle_frame)
        context_frame_layout.addWidget(full_page_frame)

        # Manga glossary generation workflow
        manga_glossary_frame = QWidget()
        manga_glossary_layout = QVBoxLayout(manga_glossary_frame)
        manga_glossary_layout.setContentsMargins(0, 8, 0, 0)
        manga_glossary_layout.setSpacing(5)

        manga_glossary_title = QLabel("Manga Glossary Workflow:")
        manga_glossary_title.setFont(title_font2)
        manga_glossary_layout.addWidget(manga_glossary_title)

        manga_glossary_toggle_frame = QWidget()
        manga_glossary_toggle_layout = QHBoxLayout(manga_glossary_toggle_frame)
        manga_glossary_toggle_layout.setContentsMargins(20, 0, 0, 0)
        manga_glossary_toggle_layout.setSpacing(10)

        self.manga_glossary_checkbox = self._create_styled_checkbox("Use loaded/generated glossary for translation")
        self.manga_glossary_checkbox.setChecked(bool(getattr(self, 'manga_glossary_enabled_value', self.main_gui.config.get('manga_glossary_enabled', False))))
        self.manga_glossary_checkbox.stateChanged.connect(self._on_manga_glossary_toggle)
        manga_glossary_toggle_layout.addWidget(self.manga_glossary_checkbox)

        edit_manga_glossary_btn = QPushButton("Glossary Prompt")
        edit_manga_glossary_btn.clicked.connect(self._edit_manga_glossary_prompt)
        edit_manga_glossary_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        manga_glossary_toggle_layout.addWidget(edit_manga_glossary_btn)

        generate_manga_glossary_btn = QPushButton("Generate Glossary")
        generate_manga_glossary_btn.clicked.connect(self._generate_manga_glossary_button_clicked)
        generate_manga_glossary_btn.setStyleSheet("QPushButton { background-color: #198754; color: white; padding: 5px 15px; }")
        manga_glossary_toggle_layout.addWidget(generate_manga_glossary_btn)

        manga_glossary_help_btn = QPushButton("?")
        manga_glossary_help_btn.setFixedWidth(30)
        manga_glossary_help_btn.clicked.connect(lambda: self._show_help_dialog(
            "Manga Glossary Workflow",
            "When enabled, the batch first runs OCR on every selected page, sends all OCR text in one glossary generation request, and then translates each page with that glossary appended to the translation prompt.\n\n"
            "The Generate Glossary button runs the OCR and glossary generation step without translating pages."
        ))
        manga_glossary_help_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px; }")
        manga_glossary_toggle_layout.addWidget(manga_glossary_help_btn)
        manga_glossary_toggle_layout.addStretch()

        manga_glossary_layout.addWidget(manga_glossary_toggle_frame)

        manga_glossary_compress_frame = QWidget()
        manga_glossary_compress_layout = QHBoxLayout(manga_glossary_compress_frame)
        manga_glossary_compress_layout.setContentsMargins(20, 0, 0, 0)
        manga_glossary_compress_layout.setSpacing(10)

        self.manga_compress_glossary_checkbox = self._create_styled_checkbox("Compress Glossary Prompt")
        self.manga_compress_glossary_checkbox.setChecked(self._get_compress_glossary_prompt_value())
        self.manga_compress_glossary_checkbox.setToolTip(
            "Only send glossary entries that appear in the current manga OCR text.\n"
            "This mirrors the Glossary Settings compression setting."
        )
        self.manga_compress_glossary_checkbox.stateChanged.connect(self._on_manga_compress_glossary_toggle)
        manga_glossary_compress_layout.addWidget(self.manga_compress_glossary_checkbox)

        manga_compress_hint = QLabel("(same setting as Glossary Settings)")
        manga_compress_hint.setStyleSheet("color: #b8c7d9; font-size: 8pt;")
        manga_glossary_compress_layout.addWidget(manga_compress_hint)
        manga_glossary_compress_layout.addStretch()
        manga_glossary_layout.addWidget(manga_glossary_compress_frame)

        manga_glossary_debug_frame = QWidget()
        manga_glossary_debug_layout = QHBoxLayout(manga_glossary_debug_frame)
        manga_glossary_debug_layout.setContentsMargins(20, 0, 0, 0)
        manga_glossary_debug_layout.setSpacing(10)

        self.manga_glossary_debug_ocr_checkbox = self._create_styled_checkbox("Save OCR/glossary debug subfolder")
        self.manga_glossary_debug_ocr_checkbox.setChecked(bool(getattr(self, 'manga_glossary_debug_ocr_text_value', self.main_gui.config.get('manga_glossary_debug_ocr_text', False))))
        self.manga_glossary_debug_ocr_checkbox.setToolTip(
            "When generating a manga glossary, save a debug subfolder next to the generated glossary.\n"
            "Includes the OCR text sent to glossary generation and the raw undeduped glossary output."
        )
        self.manga_glossary_debug_ocr_checkbox.stateChanged.connect(self._on_manga_glossary_debug_ocr_toggle)
        manga_glossary_debug_layout.addWidget(self.manga_glossary_debug_ocr_checkbox)

        manga_glossary_debug_hint = QLabel("(OCR text + raw undeduped glossary output)")
        manga_glossary_debug_hint.setStyleSheet("color: #b8c7d9; font-size: 8pt;")
        manga_glossary_debug_layout.addWidget(manga_glossary_debug_hint)
        manga_glossary_debug_layout.addStretch()
        manga_glossary_layout.addWidget(manga_glossary_debug_frame)

        manga_glossary_load_frame = QWidget()
        manga_glossary_load_layout = QHBoxLayout(manga_glossary_load_frame)
        manga_glossary_load_layout.setContentsMargins(20, 0, 0, 0)
        manga_glossary_load_layout.setSpacing(10)

        load_manga_glossary_btn = QPushButton("Load Glossary")
        load_manga_glossary_btn.clicked.connect(self._load_manga_custom_glossary)
        load_manga_glossary_btn.setStyleSheet("QPushButton { background-color: #0d6efd; color: white; padding: 5px 15px; }")
        manga_glossary_load_layout.addWidget(load_manga_glossary_btn)

        clear_manga_glossary_btn = QPushButton("Clear")
        clear_manga_glossary_btn.clicked.connect(self._clear_manga_custom_glossary)
        clear_manga_glossary_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        manga_glossary_load_layout.addWidget(clear_manga_glossary_btn)
        manga_glossary_load_layout.addStretch()

        manga_glossary_layout.addWidget(manga_glossary_load_frame)

        self.manga_glossary_status_label = QLabel("")
        self.manga_glossary_status_label.setStyleSheet("color: #b8c7d9; font-size: 8pt; margin-left: 20px;")
        self.manga_glossary_status_label.setWordWrap(True)
        manga_glossary_layout.addWidget(self.manga_glossary_status_label)
        self._update_manga_glossary_status_label()

        context_frame_layout.addWidget(manga_glossary_frame)

        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        context_frame_layout.addWidget(separator3)

        # Visual Context Settings (for non-vision model support)
        visual_frame = QWidget()
        visual_layout = QVBoxLayout(visual_frame)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(5)

        visual_title = QLabel("Visual Context (Image Support):")
        title_font3 = QFont("Arial", 10)
        title_font3.setBold(True)
        visual_title.setFont(title_font3)
        visual_layout.addWidget(visual_title)

        # Visual context toggle
        visual_toggle_frame = QWidget()
        visual_toggle_layout = QHBoxLayout(visual_toggle_frame)
        visual_toggle_layout.setContentsMargins(20, 0, 0, 0)
        visual_toggle_layout.setSpacing(10)

        self.visual_context_checkbox = self._create_styled_checkbox("Include page image in translation requests")
        self.visual_context_checkbox.setChecked(bool(getattr(self, 'visual_context_enabled_value', self.main_gui.config.get('manga_visual_context_enabled', True))))
        self.visual_context_checkbox.stateChanged.connect(self._on_visual_context_toggle)
        visual_toggle_layout.addWidget(self.visual_context_checkbox)

        # Help button for visual context
        visual_help_btn = QPushButton("?")
        visual_help_btn.setFixedWidth(30)
        visual_help_btn.clicked.connect(lambda: self._show_help_dialog(
            "Visual Context Settings",
            "Visual context includes the manga page image with translation requests.\n\n"
            "⚠️ WHEN TO DISABLE:\n"
            "• Using text-only models (Claude, GPT-3.5, standard Gemini)\n"
            "• Model doesn't support images\n"
            "• Want to reduce token usage\n"
            "• Testing text-only translation\n\n"
            "✅ WHEN TO ENABLE:\n"
            "• Using vision models (Gemini Vision, GPT-4V, Claude 3)\n"
            "• Want spatial awareness of text position\n"
            "• Need visual context for better translation\n\n"
            "Impact:\n"
            "• Disabled: Only text is sent (compatible with any model)\n"
            "• Enabled: Text + image sent (requires vision model)\n\n"
            "Note: Disabling may reduce translation quality as the AI won't see\n"
            "the artwork context or spatial layout of the text."
        ))
        visual_help_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px; }")
        visual_toggle_layout.addWidget(visual_help_btn)
        visual_toggle_layout.addStretch()
        
        visual_layout.addWidget(visual_toggle_frame)

        image_quality_frame = QWidget()
        image_quality_layout = QHBoxLayout(image_quality_frame)
        image_quality_layout.setContentsMargins(20, 0, 0, 0)
        image_quality_layout.setSpacing(10)

        self.manga_image_request_quality_checkbox = self._create_styled_checkbox("Image request quality")
        self.manga_image_request_quality_checkbox.setChecked(bool(getattr(self, 'manga_image_request_quality_enabled_value', False)))
        self.manga_image_request_quality_checkbox.setToolTip(
            "When off, manga image requests preserve original images where possible. "
            "When on, the selected format and quality are applied to manga OCR, page-image translation, and image edit requests."
        )
        self.manga_image_request_quality_checkbox.stateChanged.connect(
            lambda: self._on_manga_image_request_quality_toggled(
                self.manga_image_request_quality_checkbox.isChecked()
            )
        )
        image_quality_layout.addWidget(self.manga_image_request_quality_checkbox)

        image_quality_layout.addWidget(QLabel("Format:"))
        self.manga_image_request_format_combo = QComboBox()
        self.manga_image_request_format_combo.addItems(["jpeg", "png", "webp"])
        try:
            comp_cfg = ((self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {}) or {})
            self.manga_image_request_format_combo.setCurrentText(str(comp_cfg.get('format', 'jpeg') or 'jpeg').lower())
        except Exception:
            self.manga_image_request_format_combo.setCurrentText("jpeg")
        self.manga_image_request_format_combo.currentTextChanged.connect(self._on_manga_image_request_format_changed)
        self._disable_combobox_mousewheel(self.manga_image_request_format_combo)
        image_quality_layout.addWidget(self.manga_image_request_format_combo)

        self.manga_image_request_quality_label = QLabel("Quality:")
        image_quality_layout.addWidget(self.manga_image_request_quality_label)

        self.manga_image_request_quality_spinbox = QSpinBox()
        self.manga_image_request_quality_spinbox.valueChanged.connect(self._on_manga_image_request_quality_value_changed)
        self._disable_combobox_mousewheel(self.manga_image_request_quality_spinbox)
        image_quality_layout.addWidget(self.manga_image_request_quality_spinbox)
        image_quality_layout.addStretch()
        visual_layout.addWidget(image_quality_frame)
        self._refresh_manga_image_request_quality_controls()
        
        # Output settings - moved here to be below visual context
        output_settings_frame = QWidget()
        output_settings_layout = QVBoxLayout(output_settings_frame)
        output_settings_layout.setContentsMargins(20, 10, 0, 0)
        output_settings_layout.setSpacing(5)
        
        # Create .cbz checkbox
        cbz_row = QWidget()
        cbz_row_layout = QHBoxLayout(cbz_row)
        cbz_row_layout.setContentsMargins(0, 0, 0, 0)
        cbz_row_layout.setSpacing(10)
        
        self.create_cbz_checkbox = self._create_styled_checkbox("Create .cbz file at translation end")
        self.create_cbz_checkbox.setChecked(bool(getattr(self, 'create_cbz_at_end_value', self.main_gui.config.get('manga_create_cbz_at_end', True))))
        self.create_cbz_checkbox.stateChanged.connect(self._on_create_cbz_toggle)
        cbz_row_layout.addWidget(self.create_cbz_checkbox)
        cbz_row_layout.addStretch()
        output_settings_layout.addWidget(cbz_row)
        
        # Auto consolidate images checkbox
        consolidate_row = QWidget()
        consolidate_row_layout = QHBoxLayout(consolidate_row)
        consolidate_row_layout.setContentsMargins(0, 0, 0, 0)
        consolidate_row_layout.setSpacing(10)
        
        self.auto_consolidate_checkbox = self._create_styled_checkbox("Auto consolidate images at translation end")
        self.auto_consolidate_checkbox.setChecked(bool(getattr(self, 'auto_consolidate_images_value', self.main_gui.config.get('manga_auto_consolidate_images', True))))
        self.auto_consolidate_checkbox.stateChanged.connect(self._on_auto_consolidate_toggle)
        consolidate_row_layout.addWidget(self.auto_consolidate_checkbox)
        consolidate_row_layout.addStretch()
        output_settings_layout.addWidget(consolidate_row)
        
        # Manga Output Token Limit
        token_limit_row = QWidget()
        token_limit_row_layout = QHBoxLayout(token_limit_row)
        token_limit_row_layout.setContentsMargins(0, 5, 0, 0)
        token_limit_row_layout.setSpacing(10)
        
        token_limit_label = QLabel("Output Token Limit:")
        token_limit_label.setToolTip("Max tokens for manga translations. -1 = use main GUI limit.")
        token_limit_row_layout.addWidget(token_limit_label)
        
        self.manga_output_token_limit_spin = QSpinBox()
        self.manga_output_token_limit_spin.setRange(-1, 1000000)
        self.manga_output_token_limit_spin.setSingleStep(1000)
        self.manga_output_token_limit_spin.setSpecialValueText("Use Main GUI Limit")
        self.manga_output_token_limit_spin.setToolTip(f"Max tokens for manga translations. -1 = use main GUI limit ({getattr(self.main_gui, 'max_output_tokens', 65536)}).")
        self.manga_output_token_limit_spin.setMinimumWidth(150)
        self.manga_output_token_limit_spin.wheelEvent = lambda event: event.ignore()
        
        # Load saved value
        manga_settings = self.main_gui.config.get('manga_settings', {}) or {}
        manual_edit = manga_settings.get('manual_edit', {}) or {}
        saved_token_limit = manual_edit.get('manga_output_token_limit', -1)
        self.manga_output_token_limit_spin.setValue(saved_token_limit)
        self.manga_output_token_limit_spin.valueChanged.connect(self._on_manga_output_token_limit_change)
        token_limit_row_layout.addWidget(self.manga_output_token_limit_spin)
        
        token_limit_row_layout.addStretch()
        output_settings_layout.addWidget(token_limit_row)
        
        visual_layout.addWidget(output_settings_frame)
        
        context_frame_layout.addWidget(visual_frame)
        
        # Add the completed context_frame to settings_frame
        settings_frame_layout.addWidget(context_frame)
        
        # Add main settings frame to left column
        left_column_layout.addWidget(settings_frame)
        self._pump_build_events()
        
        # Text Rendering Settings Frame - SPLIT BETWEEN COLUMNS
        render_frame = QGroupBox("Text Visibility Settings")
        render_frame_font = QFont("Arial", 12)
        render_frame_font.setBold(True)
        render_frame.setFont(render_frame_font)
        render_frame_layout = QVBoxLayout(render_frame)
        render_frame_layout.setContentsMargins(15, 15, 15, 10)
        render_frame_layout.setSpacing(10)
        
        # Inpainting section
        inpaint_group = QGroupBox("Inpainting")
        inpaint_group_font = QFont("Arial", 11)
        inpaint_group_font.setBold(True)
        inpaint_group.setFont(inpaint_group_font)
        inpaint_group_layout = QVBoxLayout(inpaint_group)
        inpaint_group_layout.setContentsMargins(15, 15, 15, 10)
        inpaint_group_layout.setSpacing(10)

        # Skip inpainting toggle - use value loaded from config
        self.skip_inpainting_checkbox = self._create_styled_checkbox("Skip Inpainter")
        self.skip_inpainting_checkbox.setToolTip("Skip local inpainting and render translated text over the original image.")
        self.skip_inpainting_checkbox.setChecked(self.skip_inpainting_value)
        self.skip_inpainting_checkbox.stateChanged.connect(self._toggle_inpaint_visibility)
        inpaint_group_layout.addWidget(self.skip_inpainting_checkbox)

        # Inpainting method selection (only visible when inpainting is enabled)
        self.inpaint_method_frame = QWidget(inpaint_group)
        inpaint_method_layout = QHBoxLayout(self.inpaint_method_frame)
        inpaint_method_layout.setContentsMargins(0, 0, 0, 0)
        inpaint_method_layout.setSpacing(10)

        method_label = QLabel("Inpaint Method:")
        method_label_font = QFont('Arial', 9)
        method_label.setFont(method_label_font)
        method_label.setMinimumWidth(95)
        method_label.setAlignment(Qt.AlignLeft)
        inpaint_method_layout.addWidget(method_label)

        # Radio buttons for inpaint method
        method_selection_frame = QWidget()
        method_selection_layout = QHBoxLayout(method_selection_frame)
        method_selection_layout.setContentsMargins(0, 0, 0, 0)
        method_selection_layout.setSpacing(10)

        self.inpaint_method_value = self.main_gui.config.get('manga_inpaint_method', 'local')
        self.inpaint_method_group = QButtonGroup()

        # Set smaller font for radio buttons
        radio_font = QFont('Arial', 9)
        
        cloud_radio = QRadioButton("Replicate API")
        cloud_radio.setFont(radio_font)
        cloud_radio.setToolTip(
            "Use Replicate for cloud inpainting with the configured Replicate API key and cloud inpaint model.\n"
            "Not recommended: this option has performed poorly in tests."
        )
        cloud_radio.setChecked(self.inpaint_method_value == 'cloud')
        cloud_radio.toggled.connect(lambda checked: self._on_inpaint_method_change() if checked else None)
        self.inpaint_method_group.addButton(cloud_radio, 0)
        method_selection_layout.addWidget(cloud_radio)

        local_radio = QRadioButton("Local / API Model")
        local_radio.setFont(radio_font)
        local_radio.setToolTip(
            "For best local performance, select anime_onnx from the model dropdown.\n"
            "For best results, select custom-image-edit and use a modern image edit model.\n"
            "Recommended for image edit models such as Nano Banana 2 and Wan 2.6 Image Edit."
        )
        local_radio.setChecked(self.inpaint_method_value == 'local')
        local_radio.toggled.connect(lambda checked: self._on_inpaint_method_change() if checked else None)
        self.inpaint_method_group.addButton(local_radio, 1)
        method_selection_layout.addWidget(local_radio)

        hybrid_radio = QRadioButton("Hybrid")
        hybrid_radio.setFont(radio_font)
        hybrid_radio.setToolTip(
            "Show and use both cloud and local/API inpainting settings.\n"
            "Experimental and not recommended for normal use."
        )
        hybrid_radio.setChecked(self.inpaint_method_value == 'hybrid')
        hybrid_radio.toggled.connect(lambda checked: self._on_inpaint_method_change() if checked else None)
        self.inpaint_method_group.addButton(hybrid_radio, 2)
        method_selection_layout.addWidget(hybrid_radio)
        
        # Store references to radio buttons
        self.cloud_radio = cloud_radio
        self.local_radio = local_radio
        self.hybrid_radio = hybrid_radio
        
        inpaint_method_layout.addWidget(method_selection_frame)
        inpaint_method_layout.addStretch()
        inpaint_group_layout.addWidget(self.inpaint_method_frame)

        # Cloud settings frame
        self.cloud_inpaint_frame = QWidget(inpaint_group)
        cloud_inpaint_layout = QVBoxLayout(self.cloud_inpaint_frame)
        cloud_inpaint_layout.setContentsMargins(0, 0, 0, 0)
        cloud_inpaint_layout.setSpacing(5)

        # Quality selection for cloud
        quality_frame = QWidget()
        quality_layout = QHBoxLayout(quality_frame)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(10)

        quality_label = QLabel("Cloud Quality:")
        quality_label_font = QFont('Arial', 9)
        quality_label.setFont(quality_label_font)
        quality_label.setMinimumWidth(95)
        quality_label.setAlignment(Qt.AlignLeft)
        quality_layout.addWidget(quality_label)

        # inpaint_quality_value is already loaded from config in _load_rendering_settings
        self.quality_button_group = QButtonGroup()
        
        quality_options = [('high', 'High Quality'), ('fast', 'Fast')]
        for idx, (value, text) in enumerate(quality_options):
            quality_radio = QRadioButton(text)
            quality_radio.setChecked(self.inpaint_quality_value == value)
            quality_radio.toggled.connect(lambda checked, v=value: self._save_rendering_settings() if checked else None)
            self.quality_button_group.addButton(quality_radio, idx)
            quality_layout.addWidget(quality_radio)
        
        quality_layout.addStretch()
        cloud_inpaint_layout.addWidget(quality_frame)

        # Conditional separator
        self.inpaint_separator = QFrame()
        self.inpaint_separator.setFrameShape(QFrame.HLine)
        self.inpaint_separator.setFrameShadow(QFrame.Sunken)
        if not self.skip_inpainting_value:
            cloud_inpaint_layout.addWidget(self.inpaint_separator)

        # Cloud API status
        api_status_frame = QWidget()
        api_status_layout = QHBoxLayout(api_status_frame)
        api_status_layout.setContentsMargins(0, 10, 0, 0)
        api_status_layout.setSpacing(10)

        # Check if API key exists
        saved_api_key = self.main_gui.config.get('replicate_api_key', '')
        if saved_api_key:
            status_text = "✅ Cloud API configured"
            status_color = 'green'
        else:
            status_text = "❌ Cloud API not configured"
            status_color = 'red'

        self.inpaint_api_status_label = QLabel(status_text)
        api_status_font = QFont('Arial', 9)
        self.inpaint_api_status_label.setFont(api_status_font)
        self.inpaint_api_status_label.setStyleSheet(f"color: {status_color};")
        api_status_layout.addWidget(self.inpaint_api_status_label)

        configure_api_btn = QPushButton("Configure API Key")
        configure_api_btn.clicked.connect(self._configure_inpaint_api)
        configure_api_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        api_status_layout.addWidget(configure_api_btn)

        if saved_api_key:
            clear_api_btn = QPushButton("Clear")
            clear_api_btn.clicked.connect(self._clear_inpaint_api)
            clear_api_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
            api_status_layout.addWidget(clear_api_btn)
        
        api_status_layout.addStretch()
        cloud_inpaint_layout.addWidget(api_status_frame)
        inpaint_group_layout.addWidget(self.cloud_inpaint_frame)

        # Local inpainting settings frame
        self.local_inpaint_frame = QWidget(inpaint_group)
        local_inpaint_layout = QVBoxLayout(self.local_inpaint_frame)
        local_inpaint_layout.setContentsMargins(0, 0, 0, 0)
        local_inpaint_layout.setSpacing(5)

        # Local model selection
        local_model_frame = QWidget()
        local_model_layout = QHBoxLayout(local_model_frame)
        local_model_layout.setContentsMargins(0, 0, 0, 0)
        local_model_layout.setSpacing(10)

        local_model_label = QLabel("Local Model:")
        local_model_label_font = QFont('Arial', 9)
        local_model_label.setFont(local_model_label_font)
        local_model_label.setMinimumWidth(95)
        local_model_label.setAlignment(Qt.AlignLeft)
        local_model_layout.addWidget(local_model_label)
        self.local_model_label = local_model_label

        self.local_model_type_value = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        if self.local_model_type_value == 'qwen_image_edit':
            self.local_model_type_value = 'custom-image-edit'
            self.main_gui.config['manga_local_inpaint_model'] = self.local_model_type_value
            if (
                self.main_gui.config.get('manga_qwen_image_edit_model_path')
                and not self.main_gui.config.get('manga_custom-image-edit_model_path')
            ):
                endpoint = self.main_gui.config.get('custom_image_edit_endpoint', '')
                if endpoint:
                    self.main_gui.config['manga_custom-image-edit_model_path'] = endpoint
        local_model_combo = QComboBox()
        local_model_combo.addItems(['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'custom-image-edit', 'mat', 'ollama', 'sd_local'])
        local_model_combo.setCurrentText(self.local_model_type_value)
        local_model_combo.setMinimumWidth(150)
        local_model_combo.setMaximumWidth(150)
        local_combo_font = QFont('Arial', 9)
        local_model_combo.setFont(local_combo_font)
        local_model_combo.currentTextChanged.connect(self._on_local_model_change)
        self._disable_combobox_mousewheel(local_model_combo)  # Disable mousewheel scrolling
        local_model_layout.addWidget(local_model_combo)
        self.local_model_combo = local_model_combo

        self.use_custom_image_edit_endpoint_value = self.main_gui.config.get(
            'use_custom_image_edit_endpoint',
            False
        )
        self.custom_image_edit_endpoint_value = self.main_gui.config.get('custom_image_edit_endpoint', '')
        self.custom_image_edit_system_prompt_value = self.main_gui.config.get(
            'custom_image_edit_system_prompt',
            self.main_gui.config.get('custom_image_edit_prompt', self._default_custom_image_edit_system_prompt())
        )
        if self._is_old_custom_image_edit_default_prompt(self.custom_image_edit_system_prompt_value):
            self.custom_image_edit_system_prompt_value = self._default_custom_image_edit_system_prompt()
            self.main_gui.config['custom_image_edit_system_prompt'] = self.custom_image_edit_system_prompt_value
        self.custom_image_edit_user_prompt_value = self.main_gui.config.get('custom_image_edit_user_prompt', '')
        _raw_full_page = self.main_gui.config.get('custom_image_edit_full_page_output', 10)
        if isinstance(_raw_full_page, bool):
            _raw_full_page = 100 if _raw_full_page else 10
        self.custom_image_edit_full_page_output_value = max(0, min(100, int(_raw_full_page)))
        # Write the migrated int back so the inpainter sees it (not the old bool)
        self.main_gui.config['custom_image_edit_full_page_output'] = self.custom_image_edit_full_page_output_value
        self.main_gui.custom_image_edit_full_page_output_var = self.custom_image_edit_full_page_output_value
        custom_image_edit_cb = self._create_styled_checkbox("Enable Custom Image Edit Endpoint")
        custom_image_edit_cb.setToolTip(
            "Uses the Custom Image Edit Endpoint only for manga custom-image-edit inpainting. "
            "Text translation keeps using the normal LLM endpoint/provider."
        )
        try:
            custom_image_edit_cb.setChecked(bool(self.use_custom_image_edit_endpoint_value))
        except Exception:
            pass
        custom_image_edit_cb.toggled.connect(self._on_custom_image_edit_endpoint_toggle)
        self.custom_image_edit_endpoint_checkbox = custom_image_edit_cb

        custom_image_edit_prompt_btn = QPushButton("Edit Prompt")
        custom_image_edit_prompt_btn.clicked.connect(self._edit_custom_image_edit_prompt)
        custom_image_edit_prompt_btn.setToolTip(
            "Edit the prompt sent only to the custom-image-edit endpoint. "
            "It does not change the Translator GUI image prompt."
        )
        custom_image_edit_prompt_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        self.custom_image_edit_prompt_btn = custom_image_edit_prompt_btn

        custom_image_edit_area_label = QLabel("Mask Expansion:")
        custom_image_edit_area_label.setToolTip(
            "Expands the inpainting mask beyond detected text.\n"
            "0% = exact text mask only.\n"
            "100% = no masking; uses the generated image as-is.\n"
            "Higher values give the model more room around text."
        )
        area_label_font = QFont('Arial', 9)
        custom_image_edit_area_label.setFont(area_label_font)
        self.custom_image_edit_area_label = custom_image_edit_area_label

        custom_image_edit_area_spin = QSpinBox()
        custom_image_edit_area_spin.setRange(0, 100)
        custom_image_edit_area_spin.setSuffix("%")
        custom_image_edit_area_spin.setSingleStep(5)
        custom_image_edit_area_spin.setValue(self.custom_image_edit_full_page_output_value)
        custom_image_edit_area_spin.setToolTip(
            "0% = mask-only blend (conservative)\n"
            "1–99% = gradually expands the mask around text\n"
            "100% = no masking, uses the generated image exactly as returned"
        )
        custom_image_edit_area_spin.setMinimumWidth(80)
        custom_image_edit_area_spin.setMaximumWidth(100)
        custom_image_edit_area_spin.valueChanged.connect(self._on_custom_image_edit_full_page_output_changed)
        self._disable_spinbox_mousewheel(custom_image_edit_area_spin)
        self.custom_image_edit_area_spin = custom_image_edit_area_spin

        # Model descriptions
        model_desc = {
            'lama': 'LaMa (Best quality)',
            'aot': 'AOT GAN (Fast)',
            'aot_onnx': 'AOT ONNX (Optimized)',
            'mat': 'MAT (High-res)',
            'sd_local': 'Stable Diffusion (Anime)',
            'anime': 'Anime/Manga Inpainting',
            'anime_onnx': 'Anime ONNX (Fast/Optimized)',
            'lama_onnx': 'LaMa ONNX (Optimized)',
            'custom-image-edit': 'Custom OpenAI-compatible image edit endpoint (.gguf)',
        }
        self.model_desc_label = QLabel(model_desc.get(self.local_model_type_value, ''))
        desc_font = QFont('Arial', 8)
        self.model_desc_label.setFont(desc_font)
        self.model_desc_label.setStyleSheet("color: gray;")
        self.model_desc_label.setMaximumWidth(420)
        local_model_layout.addWidget(self.model_desc_label)
        local_model_layout.addStretch()
        
        local_inpaint_layout.addWidget(local_model_frame)

        custom_image_edit_controls_frame = QWidget()
        custom_image_edit_controls_layout = QHBoxLayout(custom_image_edit_controls_frame)
        custom_image_edit_controls_layout.setContentsMargins(0, 0, 0, 0)
        custom_image_edit_controls_layout.setSpacing(10)
        custom_controls_spacer = QLabel("")
        custom_controls_spacer.setMinimumWidth(95)
        custom_image_edit_controls_layout.addWidget(custom_controls_spacer)
        custom_image_edit_controls_layout.addWidget(custom_image_edit_cb)
        custom_image_edit_controls_layout.addWidget(custom_image_edit_prompt_btn)
        custom_image_edit_keys_btn = QPushButton("Image Keys")
        custom_image_edit_keys_btn.setToolTip("Open the Multi API Key Manager focused on the Image Gen/Edit key pool.")
        self._style_preview_pool_button(custom_image_edit_keys_btn)
        custom_image_edit_keys_btn.clicked.connect(lambda: self._open_multi_api_key_pool_preview('inpainter'))
        custom_image_edit_controls_layout.addWidget(custom_image_edit_keys_btn)
        self.custom_image_edit_keys_btn = custom_image_edit_keys_btn
        custom_image_edit_controls_layout.addStretch()
        local_inpaint_layout.addWidget(custom_image_edit_controls_frame)
        self.custom_image_edit_controls_frame = custom_image_edit_controls_frame

        custom_image_edit_output_frame = QWidget()
        custom_image_edit_output_layout = QHBoxLayout(custom_image_edit_output_frame)
        custom_image_edit_output_layout.setContentsMargins(0, 0, 0, 0)
        custom_image_edit_output_layout.setSpacing(10)
        custom_output_spacer = QLabel("")
        custom_output_spacer.setMinimumWidth(95)
        custom_image_edit_output_layout.addWidget(custom_output_spacer)
        custom_image_edit_output_layout.addWidget(custom_image_edit_area_label)
        custom_image_edit_output_layout.addWidget(custom_image_edit_area_spin)
        custom_image_edit_output_layout.addStretch()
        local_inpaint_layout.addWidget(custom_image_edit_output_frame)
        self.custom_image_edit_output_frame = custom_image_edit_output_frame

        self.batch_image_requests_frame = QWidget()
        batch_image_layout = QHBoxLayout(self.batch_image_requests_frame)
        batch_image_layout.setContentsMargins(0, 0, 0, 0)
        batch_image_layout.setSpacing(10)
        batch_image_spacer = QLabel("")
        batch_image_spacer.setMinimumWidth(95)
        batch_image_layout.addWidget(batch_image_spacer)
        self.batch_image_requests_checkbox = self._create_styled_checkbox("Batch Image Requests")
        self.batch_image_requests_checkbox.setChecked(bool(getattr(self, 'batch_image_requests_enabled_value', True)))
        self.batch_image_requests_checkbox.setToolTip(
            "Send image edit crops in parallel. When off, use the Batch Translation size instead."
        )
        batch_image_layout.addWidget(self.batch_image_requests_checkbox)
        self.batch_image_requests_spinbox = QSpinBox()
        self.batch_image_requests_spinbox.setRange(1, 32)
        self.batch_image_requests_spinbox.setValue(int(getattr(self, 'batch_image_requests_size_value', 5)))
        self.batch_image_requests_spinbox.setToolTip("Maximum simultaneous image edit crop requests.")
        self.batch_image_requests_spinbox.setEnabled(self.batch_image_requests_checkbox.isChecked())
        self._disable_combobox_mousewheel(self.batch_image_requests_spinbox)
        self.batch_image_requests_checkbox.toggled.connect(
            lambda checked: (
                setattr(self, 'batch_image_requests_enabled_value', bool(checked)),
                self.batch_image_requests_spinbox.setEnabled(bool(checked)),
                self._save_rendering_settings(),
            )
        )
        self.batch_image_requests_spinbox.valueChanged.connect(
            lambda value: (
                setattr(self, 'batch_image_requests_size_value', int(value)),
                self._save_rendering_settings(),
            )
        )
        batch_image_layout.addWidget(self.batch_image_requests_spinbox)
        batch_image_layout.addStretch()
        local_inpaint_layout.addWidget(self.batch_image_requests_frame)

        # Model file selection
        model_path_frame = QWidget()
        model_path_layout = QHBoxLayout(model_path_frame)
        model_path_layout.setContentsMargins(0, 5, 0, 0)
        model_path_layout.setSpacing(5)

        model_file_label = QLabel("Model File:")
        model_file_label_font = QFont('Arial', 9)
        model_file_label.setFont(model_file_label_font)
        model_file_label.setMinimumWidth(95)
        model_file_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        model_path_layout.addWidget(model_file_label)
        self.model_file_label = model_file_label

        self.local_model_path_value = self.main_gui.config.get(f'manga_{self.local_model_type_value}_model_path', '')
        self.local_model_entry = QLineEdit(self.local_model_path_value)
        self.local_model_entry.setReadOnly(True)
        self.local_model_entry.textChanged.connect(self._on_local_model_entry_text_changed)
        self.local_model_entry.setMinimumWidth(250)  # Increased width for better path visibility
        self.local_model_entry.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.local_model_entry.setStyleSheet(
            "QLineEdit { background-color: #2b2b2b; color: #ffffff; }"
        )
        model_path_layout.addWidget(self.local_model_entry)

        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self._browse_local_model)
        browse_model_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 5px 15px; }")
        model_path_layout.addWidget(browse_model_btn)
        self.browse_model_btn = browse_model_btn
        
        # Manual load button to avoid auto-loading on dialog open
        load_model_btn = QPushButton("Load")
        load_model_btn.clicked.connect(self._click_load_local_model)
        load_model_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; }")
        model_path_layout.addWidget(load_model_btn)
        self.load_model_btn = load_model_btn
        self.load_local_model_button = load_model_btn
        model_path_layout.addStretch()
        
        local_inpaint_layout.addWidget(model_path_frame)

        # Model status
        self.local_model_status_label = QLabel("")
        status_font = QFont('Arial', 9)
        self.local_model_status_label.setFont(status_font)
        local_inpaint_layout.addWidget(self.local_model_status_label)

        disable_performance_frame = QWidget()
        disable_performance_layout = QHBoxLayout(disable_performance_frame)
        disable_performance_layout.setContentsMargins(0, 0, 0, 0)
        disable_performance_layout.setSpacing(10)
        disable_performance_spacer = QLabel("")
        disable_performance_spacer.setMinimumWidth(95)
        disable_performance_layout.addWidget(disable_performance_spacer)
        disable_performance_cb = self._create_styled_checkbox("Disable Performance Mode")
        disable_performance_cb.setToolTip(
            "Off: use faster resize, crop, and tiling optimizations\n"
            "for local LaMa/ONNX inpainters.\n\n"
            "On: process the full image locally when possible.\n"
            "This can be slower and use more memory.\n\n"
            "This setting does not apply to custom image edit."
        )
        try:
            disable_performance_cb.setChecked(bool(self.disable_inpaint_performance_mode_value))
        except Exception:
            pass
        disable_performance_cb.toggled.connect(self._on_disable_inpaint_performance_mode_toggle)
        disable_performance_layout.addWidget(disable_performance_cb)
        disable_performance_layout.addStretch()
        local_inpaint_layout.addWidget(disable_performance_frame)
        self.disable_inpaint_performance_mode_checkbox = disable_performance_cb
        self.disable_inpaint_performance_mode_frame = disable_performance_frame

        # Download model button
        download_model_btn = QPushButton("Download Model")
        download_model_btn.clicked.connect(self._download_model)
        download_model_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        local_inpaint_layout.addWidget(download_model_btn)
        self.local_download_model_btn = download_model_btn
        self.download_model_btn = download_model_btn

        # Model info button
        model_info_btn = QPushButton("Model Info")
        model_info_btn.clicked.connect(self._show_model_info)
        model_info_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        local_inpaint_layout.addWidget(model_info_btn)
        
        # Add local_inpaint_frame to inpaint_group
        inpaint_group_layout.addWidget(self.local_inpaint_frame)
        
        # Both frames start visible but will be managed by _on_inpaint_method_change
        # Don't hide them here - let the method visibility logic handle it

        # Try to load saved model for current type on dialog open
        initial_model_type = self.local_model_type_value
        initial_model_path = self.main_gui.config.get(f'manga_{initial_model_type}_model_path', '')

        if initial_model_path and os.path.exists(initial_model_path):
            self.local_model_entry.setText(initial_model_path)
            if getattr(self, 'preload_local_models_on_open', False):
                self.local_model_status_label.setText("⏳ Loading saved model...")
                self.local_model_status_label.setStyleSheet("color: orange;")
                # Auto-load after dialog is ready
                QTimer.singleShot(500, lambda: self._try_load_model(initial_model_type, initial_model_path))
            else:
                # Do not auto-load large models at startup to avoid crashes on some systems
                self.local_model_status_label.setText("💤 Saved model detected (not loaded). Click 'Load' to initialize.")
                self.local_model_status_label.setStyleSheet("color: #5dade2;")  # Light cyan for better contrast
        else:
            self.local_model_status_label.setText("No model loaded")
            self.local_model_status_label.setStyleSheet("color: gray;")

        self._apply_custom_image_edit_ui_state()

        # Initialize visibility based on current settings
        self._toggle_inpaint_visibility()
        
        # Add decorative icon at the bottom to balance the layout
        icon_frame = QWidget()
        icon_layout = QVBoxLayout(icon_frame)
        icon_layout.setContentsMargins(0, 20, 0, 10)
        icon_layout.setAlignment(Qt.AlignCenter)
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        if os.path.exists(icon_path):
            icon_label = QLabel()
            icon_pixmap = QPixmap(icon_path)
            try:
                dpr = self.dialog.devicePixelRatio() if hasattr(self, 'dialog') and self.dialog else 1.0
            except Exception:
                dpr = 1.0
            target_logical = 125
            fitted = icon_pixmap.scaled(int(target_logical * dpr), int(target_logical * dpr),
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            try:
                fitted.setDevicePixelRatio(dpr)
            except Exception:
                pass
            icon_label.setPixmap(fitted)
            icon_label.setFixedSize(target_logical, target_logical)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_layout.addWidget(icon_label)
            
            # Optional: Add a subtle label
            icon_text = QLabel("Glossarion")
            icon_text.setStyleSheet("color: #5a9fd4; font-size: 10pt; font-weight: bold;")
            icon_text.setAlignment(Qt.AlignCenter)
            icon_layout.addWidget(icon_text)
        
        inpaint_group_layout.addWidget(icon_frame)
        
        # Add inpaint_group to render_frame
        render_frame_layout.addWidget(inpaint_group)
        
        # Add render_frame (inpainting only) to LEFT COLUMN
        left_column_layout.addWidget(render_frame)
        self._pump_build_events()
        
        # Background Settings - MOVED TO RIGHT COLUMN
        self.bg_settings_frame = QGroupBox("Background Settings")
        bg_settings_font = QFont("Arial", 10)
        bg_settings_font.setBold(True)
        self.bg_settings_frame.setFont(bg_settings_font)
        bg_settings_layout = QVBoxLayout(self.bg_settings_frame)
        bg_settings_layout.setContentsMargins(10, 15, 10, 15)  # Increased vertical padding from 10 to 15
        bg_settings_layout.setSpacing(8)
        
        # Free text preservation toggle (renders free text with BG opacity and skips inpainting)
        self.ft_only_checkbox = self._create_styled_checkbox("Preserve free text (skip inpaint, use background opacity)")
        self.ft_only_checkbox.setChecked(self.free_text_only_bg_opacity_value)
        # Connect directly to save+apply (working pattern)
        self.ft_only_checkbox.stateChanged.connect(lambda: (self._on_ft_only_bg_opacity_changed(), self._save_rendering_settings(), self._apply_rendering_settings()))
        bg_settings_layout.addWidget(self.ft_only_checkbox)

        # Background opacity slider
        opacity_frame = QWidget()
        opacity_layout = QHBoxLayout(opacity_frame)
        opacity_layout.setContentsMargins(0, 5, 0, 5)
        opacity_layout.setSpacing(10)
        
        opacity_label_text = QLabel("Background Opacity:")
        opacity_label_text.setMinimumWidth(150)
        opacity_layout.addWidget(opacity_label_text)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(255)
        self.opacity_slider.setValue(self.bg_opacity_value)
        self.opacity_slider.setMinimumWidth(200)
        self.opacity_slider.valueChanged.connect(lambda value: (self._update_opacity_label(value), self._save_rendering_settings(), self._apply_rendering_settings()))
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(50)
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addStretch()
        
        bg_settings_layout.addWidget(opacity_frame)
        
        # Initialize the label with the loaded value
        self._update_opacity_label(self.bg_opacity_value)

        # Background size reduction
        reduction_frame = QWidget()
        reduction_layout = QHBoxLayout(reduction_frame)
        reduction_layout.setContentsMargins(0, 5, 0, 5)
        reduction_layout.setSpacing(10)
        
        reduction_label_text = QLabel("Background Size:")
        reduction_label_text.setMinimumWidth(150)
        reduction_layout.addWidget(reduction_label_text)
        
        self.reduction_slider = QDoubleSpinBox()
        self.reduction_slider.setMinimum(0.5)
        self.reduction_slider.setMaximum(2.0)
        self.reduction_slider.setSingleStep(0.05)
        self.reduction_slider.setValue(self.bg_reduction_value)
        self.reduction_slider.setMinimumWidth(100)
        self.reduction_slider.valueChanged.connect(lambda value: (self._update_reduction_label(value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.reduction_slider)
        reduction_layout.addWidget(self.reduction_slider)
        
        self.reduction_label = QLabel("100%")
        self.reduction_label.setMinimumWidth(50)
        reduction_layout.addWidget(self.reduction_label)
        reduction_layout.addStretch()
        
        bg_settings_layout.addWidget(reduction_frame)
        
        # Initialize the label with the loaded value
        self._update_reduction_label(self.bg_reduction_value)

        # Background style selection
        style_frame = QWidget()
        style_layout = QHBoxLayout(style_frame)
        style_layout.setContentsMargins(0, 5, 0, 5)
        style_layout.setSpacing(10)

        style_label = QLabel("Background Style:")
        style_label.setMinimumWidth(150)
        style_layout.addWidget(style_label)

        # Radio buttons for background style
        self.bg_style_group = QButtonGroup()
        
        box_radio = QRadioButton("Box")
        box_radio.setChecked(self.bg_style_value == "box")
        box_radio.toggled.connect(lambda checked: (setattr(self, 'bg_style_value', 'box'), self._save_rendering_settings(), self._apply_rendering_settings()) if checked else None)
        self.bg_style_group.addButton(box_radio, 0)
        style_layout.addWidget(box_radio)

        circle_radio = QRadioButton("Circle")
        circle_radio.setChecked(self.bg_style_value == "circle")
        circle_radio.toggled.connect(lambda checked: (setattr(self, 'bg_style_value', 'circle'), self._save_rendering_settings(), self._apply_rendering_settings()) if checked else None)
        self.bg_style_group.addButton(circle_radio, 1)
        style_layout.addWidget(circle_radio)

        wrap_radio = QRadioButton("Wrap")
        wrap_radio.setChecked(self.bg_style_value == "wrap")
        wrap_radio.toggled.connect(lambda checked: (setattr(self, 'bg_style_value', 'wrap'), self._save_rendering_settings(), self._apply_rendering_settings()) if checked else None)
        self.bg_style_group.addButton(wrap_radio, 2)
        style_layout.addWidget(wrap_radio)
        
        # Store references
        self.box_radio = box_radio
        self.circle_radio = circle_radio
        self.wrap_radio = wrap_radio

        # Add tooltips or descriptions
        style_help = QLabel("(Box: rounded rectangle, Circle: ellipse, Wrap: per-line)")
        style_help_font = QFont('Arial', 9)
        style_help.setFont(style_help_font)
        style_help.setStyleSheet("color: gray;")
        style_layout.addWidget(style_help)
        style_layout.addStretch()
        
        bg_settings_layout.addWidget(style_frame)
        
        # Add Background Settings to RIGHT COLUMN
        right_column_layout.addWidget(self.bg_settings_frame)
        
        # Font Settings group (consolidated) - GOES TO RIGHT COLUMN (after background settings)
        font_render_frame = QGroupBox("Font & Text Settings")
        font_render_frame_font = QFont("Arial", 10)
        font_render_frame_font.setBold(True)
        font_render_frame.setFont(font_render_frame_font)
        font_render_frame_layout = QVBoxLayout(font_render_frame)
        font_render_frame_layout.setContentsMargins(15, 15, 15, 10)
        font_render_frame_layout.setSpacing(10)
        self.sizing_group = QGroupBox("Font Settings")
        sizing_group_font = QFont("Arial", 9)
        sizing_group_font.setBold(True)
        self.sizing_group.setFont(sizing_group_font)
        sizing_group_layout = QVBoxLayout(self.sizing_group)
        sizing_group_layout.setContentsMargins(10, 10, 10, 10)
        sizing_group_layout.setSpacing(8)
 
        # Font sizing algorithm selection
        algo_frame = QWidget()
        algo_layout = QHBoxLayout(algo_frame)
        algo_layout.setContentsMargins(0, 6, 0, 0)
        algo_layout.setSpacing(10)
        
        algo_label = QLabel("Font Size Algorithm:")
        algo_label.setMinimumWidth(150)
        algo_layout.addWidget(algo_label)
        
        # Radio buttons for algorithm selection
        self.font_algorithm_group = QButtonGroup()
        
        for idx, (value, text) in enumerate([
            ('conservative', 'Conservative'),
            ('smart', 'Smart'),
            ('aggressive', 'Aggressive')
        ]):
            rb = QRadioButton(text)
            rb.setChecked(self.font_algorithm_value == value)
            rb.toggled.connect(lambda checked, v=value: (setattr(self, 'font_algorithm_value', v), self._save_rendering_settings(), self._apply_rendering_settings()) if checked else None)
            self.font_algorithm_group.addButton(rb, idx)
            algo_layout.addWidget(rb)
        
        algo_layout.addStretch()
        sizing_group_layout.addWidget(algo_frame)

        # Font size selection with mode toggle
        font_frame_container = QWidget()
        font_frame_layout = QVBoxLayout(font_frame_container)
        font_frame_layout.setContentsMargins(0, 5, 0, 5)
        font_frame_layout.setSpacing(10)
        
        # Mode selection frame
        mode_frame = QWidget()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(10)

        mode_label = QLabel("Font Size Mode:")
        mode_label.setMinimumWidth(150)
        mode_layout.addWidget(mode_label)

        # Radio buttons for mode selection
        self.font_size_mode_group = QButtonGroup()
        
        auto_radio = QRadioButton("Auto")
        auto_radio.setChecked(self.font_size_mode_value == "auto")
        auto_radio.toggled.connect(lambda checked: (setattr(self, 'font_size_mode_value', 'auto'), self._toggle_font_size_mode()) if checked else None)
        self.font_size_mode_group.addButton(auto_radio, 0)
        mode_layout.addWidget(auto_radio)
        
        fixed_radio = QRadioButton("Fixed Size")
        fixed_radio.setChecked(self.font_size_mode_value == "fixed")
        fixed_radio.toggled.connect(lambda checked: (setattr(self, 'font_size_mode_value', 'fixed'), self._toggle_font_size_mode()) if checked else None)
        self.font_size_mode_group.addButton(fixed_radio, 1)
        mode_layout.addWidget(fixed_radio)
        
        multiplier_radio = QRadioButton("Dynamic Multiplier")
        multiplier_radio.setChecked(self.font_size_mode_value == "multiplier")
        multiplier_radio.toggled.connect(lambda checked: (setattr(self, 'font_size_mode_value', 'multiplier'), self._toggle_font_size_mode()) if checked else None)
        self.font_size_mode_group.addButton(multiplier_radio, 2)
        mode_layout.addWidget(multiplier_radio)
        
        # Store references
        self.auto_mode_radio = auto_radio
        self.fixed_mode_radio = fixed_radio
        self.multiplier_mode_radio = multiplier_radio
        
        mode_layout.addStretch()
        font_frame_layout.addWidget(mode_frame)

        # Fixed font size frame
        self.fixed_size_frame = QWidget()
        fixed_size_layout = QHBoxLayout(self.fixed_size_frame)
        fixed_size_layout.setContentsMargins(0, 8, 0, 0)
        fixed_size_layout.setSpacing(10)

        fixed_size_label = QLabel("Font Size:")
        fixed_size_label.setMinimumWidth(150)
        fixed_size_layout.addWidget(fixed_size_label)

        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setMinimum(0)
        self.font_size_spinbox.setMaximum(72)
        self.font_size_spinbox.setValue(self.font_size_value)
        self.font_size_spinbox.setMinimumWidth(100)
        self.font_size_spinbox.valueChanged.connect(lambda value: (setattr(self, 'font_size_value', value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.font_size_spinbox)
        fixed_size_layout.addWidget(self.font_size_spinbox)

        fixed_help_label = QLabel("(0 = Auto)")
        fixed_help_font = QFont('Arial', 9)
        fixed_help_label.setFont(fixed_help_font)
        fixed_help_label.setStyleSheet("color: gray;")
        fixed_size_layout.addWidget(fixed_help_label)
        fixed_size_layout.addStretch()
        
        font_frame_layout.addWidget(self.fixed_size_frame)

        # Dynamic multiplier frame
        self.multiplier_frame = QWidget()
        multiplier_layout = QHBoxLayout(self.multiplier_frame)
        multiplier_layout.setContentsMargins(0, 8, 0, 0)
        multiplier_layout.setSpacing(10)

        multiplier_label_text = QLabel("Size Multiplier:")
        multiplier_label_text.setMinimumWidth(150)
        multiplier_layout.addWidget(multiplier_label_text)

        self.multiplier_slider = QDoubleSpinBox()
        self.multiplier_slider.setMinimum(0.5)
        self.multiplier_slider.setMaximum(2.0)
        self.multiplier_slider.setSingleStep(0.1)
        self.multiplier_slider.setValue(self.font_size_multiplier_value)
        self.multiplier_slider.setMinimumWidth(100)
        self.multiplier_slider.valueChanged.connect(lambda value: (self._update_multiplier_label(value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.multiplier_slider)
        multiplier_layout.addWidget(self.multiplier_slider)

        self.multiplier_label = QLabel("1.0x")
        self.multiplier_label.setMinimumWidth(50)
        multiplier_layout.addWidget(self.multiplier_label)

        multiplier_help_label = QLabel("(Scales with panel size)")
        multiplier_help_font = QFont('Arial', 9)
        multiplier_help_label.setFont(multiplier_help_font)
        multiplier_help_label.setStyleSheet("color: gray;")
        multiplier_layout.addWidget(multiplier_help_label)
        multiplier_layout.addStretch()
        
        font_frame_layout.addWidget(self.multiplier_frame)

        # Constraint checkbox frame (only visible in multiplier mode)
        self.constraint_frame = QWidget()
        constraint_layout = QHBoxLayout(self.constraint_frame)
        constraint_layout.setContentsMargins(20, 0, 0, 0)
        constraint_layout.setSpacing(10)
        
        self.constrain_checkbox = self._create_styled_checkbox("Constrain text to bubble boundaries")
        self.constrain_checkbox.setChecked(self.constrain_to_bubble_value)
        self.constrain_checkbox.stateChanged.connect(lambda: (setattr(self, 'constrain_to_bubble_value', self.constrain_checkbox.isChecked()), self._save_rendering_settings(), self._apply_rendering_settings()))
        constraint_layout.addWidget(self.constrain_checkbox)

        constraint_help_label = QLabel("(Unchecked allows text to exceed bubbles)")
        constraint_help_font = QFont('Arial', 9)
        constraint_help_label.setFont(constraint_help_font)
        constraint_help_label.setStyleSheet("color: gray;")
        constraint_layout.addWidget(constraint_help_label)
        constraint_layout.addStretch()
        
        font_frame_layout.addWidget(self.constraint_frame)
        
        # Add font_frame_container to sizing_group_layout
        sizing_group_layout.addWidget(font_frame_container)

        # Minimum Font Size (Auto mode lower bound)
        self.min_size_frame = QWidget()
        min_size_layout = QHBoxLayout(self.min_size_frame)
        min_size_layout.setContentsMargins(0, 5, 0, 5)
        min_size_layout.setSpacing(10)

        min_size_label = QLabel("Minimum Font Size:")
        min_size_label.setMinimumWidth(150)
        min_size_layout.addWidget(min_size_label)

        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setMinimum(1)
        self.min_size_spinbox.setMaximum(999)
        self.min_size_spinbox.setValue(self.auto_min_size_value)
        self.min_size_spinbox.setMinimumWidth(100)
        self.min_size_spinbox.valueChanged.connect(lambda value: setattr(self, 'auto_min_size_value', value))
        self.min_size_spinbox.editingFinished.connect(lambda: self._validate_font_size_range_manga('min'))
        self._disable_spinbox_mousewheel(self.min_size_spinbox)
        min_size_layout.addWidget(self.min_size_spinbox)

        min_help_label = QLabel("(Auto mode won't go below this)")
        min_help_font = QFont('Arial', 9)
        min_help_label.setFont(min_help_font)
        min_help_label.setStyleSheet("color: gray;")
        min_size_layout.addWidget(min_help_label)
        min_size_layout.addStretch()
        
        sizing_group_layout.addWidget(self.min_size_frame)
    
        # Maximum Font Size (Auto mode upper bound)
        self.max_size_frame = QWidget()
        max_size_layout = QHBoxLayout(self.max_size_frame)
        max_size_layout.setContentsMargins(0, 5, 0, 5)
        max_size_layout.setSpacing(10)

        max_size_label = QLabel("Maximum Font Size:")
        max_size_label.setMinimumWidth(150)
        max_size_layout.addWidget(max_size_label)

        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setMinimum(1)
        self.max_size_spinbox.setMaximum(999)
        self.max_size_spinbox.setValue(self.max_font_size_value)
        self.max_size_spinbox.setMinimumWidth(100)
        self.max_size_spinbox.valueChanged.connect(lambda value: setattr(self, 'max_font_size_value', value))
        self.max_size_spinbox.editingFinished.connect(lambda: self._validate_font_size_range_manga('max'))
        self._disable_spinbox_mousewheel(self.max_size_spinbox)
        max_size_layout.addWidget(self.max_size_spinbox)

        max_help_label = QLabel("(Limits maximum text size)")
        max_help_font = QFont('Arial', 9)
        max_help_label.setFont(max_help_font)
        max_help_label.setStyleSheet("color: gray;")
        max_size_layout.addWidget(max_help_label)
        max_size_layout.addStretch()
        
        sizing_group_layout.addWidget(self.max_size_frame)

        # Initialize visibility AFTER all frames are created
        self._toggle_font_size_mode()

        # Auto Fit Style (applies to Auto mode)
        fit_row = QWidget()
        fit_layout = QHBoxLayout(fit_row)
        fit_layout.setContentsMargins(0, 0, 0, 6)
        fit_layout.setSpacing(10)
        
        fit_label = QLabel("Auto Fit Style:")
        fit_label.setMinimumWidth(110)
        fit_layout.addWidget(fit_label)
        
        # Radio buttons for auto fit style
        self.auto_fit_style_group = QButtonGroup()
        
        for idx, (value, text) in enumerate([('compact','Compact'), ('balanced','Balanced'), ('readable','Readable')]):
            rb = QRadioButton(text)
            rb.setChecked(self.auto_fit_style_value == value)
            rb.toggled.connect(lambda checked, v=value: (setattr(self, 'auto_fit_style_value', v), self._save_rendering_settings(), self._apply_rendering_settings()) if checked else None)
            self.auto_fit_style_group.addButton(rb, idx)
            fit_layout.addWidget(rb)
        
        fit_layout.addStretch()
        sizing_group_layout.addWidget(fit_row)

        # Behavior toggles
        self.prefer_larger_checkbox = self._create_styled_checkbox("Prefer larger text")
        self.prefer_larger_checkbox.setChecked(self.prefer_larger_value)
        self.prefer_larger_checkbox.stateChanged.connect(lambda: (setattr(self, 'prefer_larger_value', self.prefer_larger_checkbox.isChecked()), self._save_rendering_settings(), self._apply_rendering_settings()))
        sizing_group_layout.addWidget(self.prefer_larger_checkbox)
        
        self.bubble_size_factor_checkbox = self._create_styled_checkbox("Scale with bubble size")
        self.bubble_size_factor_checkbox.setChecked(self.bubble_size_factor_value)
        self.bubble_size_factor_checkbox.stateChanged.connect(lambda: (setattr(self, 'bubble_size_factor_value', self.bubble_size_factor_checkbox.isChecked()), self._save_rendering_settings(), self._apply_rendering_settings()))
        sizing_group_layout.addWidget(self.bubble_size_factor_checkbox)

        # Safe area controls (checkbox + inline scale)
        safe_area_row = QWidget()
        sa_row_layout = QHBoxLayout(safe_area_row)
        sa_row_layout.setContentsMargins(0, 2, 0, 4)
        sa_row_layout.setSpacing(10)
        
        self.safe_area_enabled_checkbox = self._create_styled_checkbox("Use safe area (mask/polygon)")
        self.safe_area_enabled_checkbox.setChecked(getattr(self, 'safe_area_enabled_value', False))
        self.safe_area_enabled_checkbox.stateChanged.connect(lambda: (setattr(self, 'safe_area_enabled_value', self.safe_area_enabled_checkbox.isChecked()), self._save_rendering_settings(), self._apply_rendering_settings()))
        sa_row_layout.addWidget(self.safe_area_enabled_checkbox)
        
        # Inline scale (no label) with x-suffix
        self.safe_area_scale_spinbox = QDoubleSpinBox()
        self.safe_area_scale_spinbox.setMinimum(0.70)
        self.safe_area_scale_spinbox.setMaximum(1.10)
        self.safe_area_scale_spinbox.setSingleStep(0.01)
        self.safe_area_scale_spinbox.setValue(getattr(self, 'safe_area_scale_value', 1.0))
        self.safe_area_scale_spinbox.setMinimumWidth(80)
        try:
            self.safe_area_scale_spinbox.setSuffix("x")
        except Exception:
            pass
        self.safe_area_scale_spinbox.setToolTip("Safe area scale factor")
        self.safe_area_scale_spinbox.valueChanged.connect(lambda value: (self._on_safe_area_scale_changed(value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.safe_area_scale_spinbox)
        sa_row_layout.addWidget(self.safe_area_scale_spinbox)
        sa_row_layout.addStretch()
        sizing_group_layout.addWidget(safe_area_row)

        # Line Spacing row with live value label
        row_ls = QWidget()
        ls_layout = QHBoxLayout(row_ls)
        ls_layout.setContentsMargins(0, 6, 0, 2)
        ls_layout.setSpacing(10)
        
        ls_label = QLabel("Line Spacing:")
        ls_label.setMinimumWidth(110)
        ls_layout.addWidget(ls_label)
        
        self.line_spacing_spinbox = QDoubleSpinBox()
        self.line_spacing_spinbox.setMinimum(1.0)
        self.line_spacing_spinbox.setMaximum(2.0)
        self.line_spacing_spinbox.setSingleStep(0.01)
        self.line_spacing_spinbox.setValue(self.line_spacing_value)
        self.line_spacing_spinbox.setMinimumWidth(100)
        self.line_spacing_spinbox.valueChanged.connect(lambda value: (self._on_line_spacing_changed(value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.line_spacing_spinbox)
        ls_layout.addWidget(self.line_spacing_spinbox)
        
        self.line_spacing_value_label = QLabel(f"{self.line_spacing_value:.2f}")
        self.line_spacing_value_label.setMinimumWidth(50)
        ls_layout.addWidget(self.line_spacing_value_label)
        ls_layout.addStretch()
        
        sizing_group_layout.addWidget(row_ls)

        # Quick Presets (horizontal) merged into sizing group
        row_presets = QWidget()
        presets_layout = QHBoxLayout(row_presets)
        presets_layout.setContentsMargins(0, 6, 0, 2)
        presets_layout.setSpacing(10)
        
        presets_label = QLabel("Quick Presets:")
        presets_label.setMinimumWidth(110)
        presets_layout.addWidget(presets_label)
        
        small_preset_btn = QPushButton("Manga")
        small_preset_btn.setMinimumWidth(120)
        small_preset_btn.clicked.connect(lambda: self._set_font_preset('small'))
        presets_layout.addWidget(small_preset_btn)
        
        balanced_preset_btn = QPushButton("Manhwa")
        balanced_preset_btn.setMinimumWidth(120)
        balanced_preset_btn.clicked.connect(lambda: self._set_font_preset('balanced'))
        presets_layout.addWidget(balanced_preset_btn)
        
        large_preset_btn = QPushButton("Large Text")
        large_preset_btn.setMinimumWidth(120)
        large_preset_btn.clicked.connect(lambda: self._set_font_preset('large'))
        presets_layout.addWidget(large_preset_btn)
        
        presets_layout.addStretch()
        sizing_group_layout.addWidget(row_presets)

        # Text wrapping mode (moved into Font Settings)
        wrap_frame = QWidget()
        wrap_layout = QVBoxLayout(wrap_frame)
        wrap_layout.setContentsMargins(0, 12, 0, 4)
        wrap_layout.setSpacing(5)

        self.strict_wrap_checkbox = self._create_styled_checkbox("Strict text wrapping (force text to fit within bubbles)")
        self.strict_wrap_checkbox.setChecked(self.strict_text_wrapping_value)
        self.strict_wrap_checkbox.stateChanged.connect(lambda: (setattr(self, 'strict_text_wrapping_value', self.strict_wrap_checkbox.isChecked()), self._save_rendering_settings(), self._apply_rendering_settings()))
        wrap_layout.addWidget(self.strict_wrap_checkbox)

        wrap_help_label = QLabel("(Break words with hyphens if needed)")
        wrap_help_font = QFont('Arial', 9)
        wrap_help_label.setFont(wrap_help_font)
        wrap_help_label.setStyleSheet("color: gray; margin-left: 20px;")
        wrap_layout.addWidget(wrap_help_label)

        # Force CAPS LOCK directly below strict wrapping
        self.force_caps_checkbox = self._create_styled_checkbox("Force CAPS LOCK")
        self.force_caps_checkbox.setChecked(self.force_caps_lock_value)
        self.force_caps_checkbox.stateChanged.connect(lambda: (setattr(self, 'force_caps_lock_value', self.force_caps_checkbox.isChecked()), self._save_rendering_settings(), self._apply_rendering_settings()))
        wrap_layout.addWidget(self.force_caps_checkbox)
        
        sizing_group_layout.addWidget(wrap_frame)
        self._pump_build_events()
    
        # Update multiplier label with loaded value
        self._update_multiplier_label(self.font_size_multiplier_value)
        
        # Add sizing_group to font_render_frame (right column)
        font_render_frame_layout.addWidget(self.sizing_group)
        
        # Font style selection (moved into Font Settings)
        font_style_frame = QWidget()
        font_style_layout = QHBoxLayout(font_style_frame)
        font_style_layout.setContentsMargins(0, 6, 0, 4)
        font_style_layout.setSpacing(10)
        
        font_style_label = QLabel("Font Style:")
        font_style_label.setMinimumWidth(110)
        font_style_layout.addWidget(font_style_label)
        
        # Font style will be set from loaded config in _load_rendering_settings
        self.font_combo = QComboBox()
        self.font_combo.addItems(self._get_available_fonts())
        self.font_combo.setCurrentText(self.font_style_value)
        self.font_combo.setMinimumWidth(120)  # Reduced for better fit
        self.font_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.font_combo.currentTextChanged.connect(lambda: (self._on_font_selected(), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_combobox_mousewheel(self.font_combo)  # Disable mousewheel scrolling
        font_style_layout.addWidget(self.font_combo)
        font_style_layout.addStretch()
        
        font_render_frame_layout.addWidget(font_style_frame)
        
        # Font color selection (moved into Font Settings)
        color_frame = QWidget()
        color_layout = QHBoxLayout(color_frame)
        color_layout.setContentsMargins(0, 6, 0, 12)
        color_layout.setSpacing(10)
        
        color_label = QLabel("Font Color:")
        color_label.setMinimumWidth(110)
        color_layout.addWidget(color_label)
        
        # Color preview frame
        self.color_preview_frame = QFrame()
        self.color_preview_frame.setFixedSize(40, 30)
        self.color_preview_frame.setFrameShape(QFrame.Box)
        self.color_preview_frame.setLineWidth(1)
        # Initialize with current color
        r, g, b = self.text_color_r_value, self.text_color_g_value, self.text_color_b_value
        self.color_preview_frame.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #5a9fd4;")
        color_layout.addWidget(self.color_preview_frame)
        
        # RGB display label
        r, g, b = self.text_color_r_value, self.text_color_g_value, self.text_color_b_value
        self.rgb_label = QLabel(f"RGB({r},{g},{b})")
        self.rgb_label.setMinimumWidth(100)
        color_layout.addWidget(self.rgb_label)
        
        # Color picker button
        def pick_font_color():
            # Get current color
            current_color = QColor(self.text_color_r_value, self.text_color_g_value, self.text_color_b_value)
            
            # Open color dialog
            color = QColorDialog.getColor(current_color, self.dialog, "Choose Font Color")
            if color.isValid():
                # Update RGB values
                self.text_color_r_value = color.red()
                self.text_color_g_value = color.green()
                self.text_color_b_value = color.blue()
                # Update display
                self.rgb_label.setText(f"RGB({color.red()},{color.green()},{color.blue()})")
                self._update_color_preview(None)
                # Save settings to config
                self._save_rendering_settings()
        
        choose_color_btn = QPushButton("Choose Color")
        choose_color_btn.clicked.connect(pick_font_color)
        choose_color_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        color_layout.addWidget(choose_color_btn)
        color_layout.addStretch()
        
        font_render_frame_layout.addWidget(color_frame)
        
        self._update_color_preview(None)  # Initialize with loaded colors
        
        # Text Shadow settings (moved into Font Settings)
        shadow_header = QWidget()
        shadow_header_layout = QHBoxLayout(shadow_header)
        shadow_header_layout.setContentsMargins(0, 4, 0, 4)
        
        # Shadow enabled checkbox
        self.shadow_enabled_checkbox = self._create_styled_checkbox("Enable Shadow")
        self.shadow_enabled_checkbox.setChecked(self.shadow_enabled_value)
        self.shadow_enabled_checkbox.stateChanged.connect(lambda: (setattr(self, 'shadow_enabled_value', self.shadow_enabled_checkbox.isChecked()), self._toggle_shadow_controls(), self._save_rendering_settings(), self._apply_rendering_settings(), ImageRenderer._relayout_all_overlays_for_current_image(self, )))
        shadow_header_layout.addWidget(self.shadow_enabled_checkbox)
        shadow_header_layout.addStretch()
        
        font_render_frame_layout.addWidget(shadow_header)
        
        # Shadow controls container
        self.shadow_controls = QWidget()
        shadow_controls_layout = QVBoxLayout(self.shadow_controls)
        shadow_controls_layout.setContentsMargins(0, 2, 0, 6)
        shadow_controls_layout.setSpacing(5)
        
        # Shadow color
        shadow_color_frame = QWidget()
        shadow_color_layout = QHBoxLayout(shadow_color_frame)
        shadow_color_layout.setContentsMargins(0, 2, 0, 8)
        shadow_color_layout.setSpacing(10)
        
        shadow_color_label = QLabel("Shadow Color:")
        shadow_color_label.setMinimumWidth(110)
        shadow_color_layout.addWidget(shadow_color_label)
        
        # Shadow color preview
        self.shadow_preview_frame = QFrame()
        self.shadow_preview_frame.setFixedSize(30, 25)
        self.shadow_preview_frame.setFrameShape(QFrame.Box)
        self.shadow_preview_frame.setLineWidth(1)
        # Initialize with current color
        sr, sg, sb = self.shadow_color_r_value, self.shadow_color_g_value, self.shadow_color_b_value
        self.shadow_preview_frame.setStyleSheet(f"background-color: rgb({sr},{sg},{sb}); border: 1px solid #5a9fd4;")
        shadow_color_layout.addWidget(self.shadow_preview_frame)
        
        # Shadow RGB display label
        sr, sg, sb = self.shadow_color_r_value, self.shadow_color_g_value, self.shadow_color_b_value
        self.shadow_rgb_label = QLabel(f"RGB({sr},{sg},{sb})")
        self.shadow_rgb_label.setMinimumWidth(120)
        shadow_color_layout.addWidget(self.shadow_rgb_label)
        
        # Shadow color picker button
        def pick_shadow_color():
            # Get current color
            current_color = QColor(self.shadow_color_r_value, self.shadow_color_g_value, self.shadow_color_b_value)
            
            # Open color dialog
            color = QColorDialog.getColor(current_color, self.dialog, "Choose Shadow Color")
            if color.isValid():
                # Update RGB values
                self.shadow_color_r_value = color.red()
                self.shadow_color_g_value = color.green()
                self.shadow_color_b_value = color.blue()
                # Update display
                self.shadow_rgb_label.setText(f"RGB({color.red()},{color.green()},{color.blue()})")
                self._update_shadow_preview(None)
                # Save + apply immediately, then reflow overlays and (optionally) rerender output
                self._save_rendering_settings()
                self._apply_rendering_settings()
                try:
                    ImageRenderer._relayout_all_overlays_for_current_image(self, )
                except Exception:
                    pass
                try:
                    import ImageRenderer
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: ImageRenderer.save_positions_and_rerender(self))
                except Exception:
                    pass
        
        choose_shadow_btn = QPushButton("Choose Color")
        choose_shadow_btn.setMinimumWidth(120)
        choose_shadow_btn.clicked.connect(pick_shadow_color)
        choose_shadow_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        shadow_color_layout.addWidget(choose_shadow_btn)
        shadow_color_layout.addStretch()
        
        shadow_controls_layout.addWidget(shadow_color_frame)
        
        self._update_shadow_preview(None)  # Initialize with loaded colors
        
        # Shadow offset
        offset_frame = QWidget()
        offset_layout = QHBoxLayout(offset_frame)
        offset_layout.setContentsMargins(0, 2, 0, 0)
        offset_layout.setSpacing(10)
        
        offset_label = QLabel("Shadow Offset:")
        offset_label.setMinimumWidth(110)
        offset_layout.addWidget(offset_label)
        
        # X offset
        x_label = QLabel("X:")
        offset_layout.addWidget(x_label)
        
        self.shadow_offset_x_spinbox = QSpinBox()
        self.shadow_offset_x_spinbox.setMinimum(-10)
        self.shadow_offset_x_spinbox.setMaximum(10)
        self.shadow_offset_x_spinbox.setValue(self.shadow_offset_x_value)
        self.shadow_offset_x_spinbox.setMinimumWidth(60)
        self.shadow_offset_x_spinbox.valueChanged.connect(lambda value: (setattr(self, 'shadow_offset_x_value', value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.shadow_offset_x_spinbox)
        offset_layout.addWidget(self.shadow_offset_x_spinbox)
        
        # Y offset
        y_label = QLabel("Y:")
        offset_layout.addWidget(y_label)
        
        self.shadow_offset_y_spinbox = QSpinBox()
        self.shadow_offset_y_spinbox.setMinimum(-10)
        self.shadow_offset_y_spinbox.setMaximum(10)
        self.shadow_offset_y_spinbox.setValue(self.shadow_offset_y_value)
        self.shadow_offset_y_spinbox.setMinimumWidth(60)
        self.shadow_offset_y_spinbox.valueChanged.connect(lambda value: (setattr(self, 'shadow_offset_y_value', value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.shadow_offset_y_spinbox)
        offset_layout.addWidget(self.shadow_offset_y_spinbox)
        offset_layout.addStretch()
        
        shadow_controls_layout.addWidget(offset_frame)
        
        # Shadow blur
        blur_frame = QWidget()
        blur_layout = QHBoxLayout(blur_frame)
        blur_layout.setContentsMargins(0, 2, 0, 0)
        blur_layout.setSpacing(10)
        
        blur_label = QLabel("Shadow Blur:")
        blur_label.setMinimumWidth(110)
        blur_layout.addWidget(blur_label)
        
        self.shadow_blur_spinbox = QSpinBox()
        self.shadow_blur_spinbox.setMinimum(0)
        self.shadow_blur_spinbox.setMaximum(10)
        self.shadow_blur_spinbox.setValue(self.shadow_blur_value)
        self.shadow_blur_spinbox.setMinimumWidth(100)
        self.shadow_blur_spinbox.valueChanged.connect(lambda value: (self._on_shadow_blur_changed(value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.shadow_blur_spinbox)
        blur_layout.addWidget(self.shadow_blur_spinbox)
        
        # Shadow blur value label
        self.shadow_blur_value_label = QLabel(f"{self.shadow_blur_value}")
        self.shadow_blur_value_label.setMinimumWidth(30)
        blur_layout.addWidget(self.shadow_blur_value_label)
        
        blur_help_label = QLabel("(0=sharp, 10=blurry)")
        blur_help_font = QFont('Arial', 9)
        blur_help_label.setFont(blur_help_font)
        blur_help_label.setStyleSheet("color: gray;")
        blur_layout.addWidget(blur_help_label)
        blur_layout.addStretch()
        
        shadow_controls_layout.addWidget(blur_frame)
        
        # Add shadow_controls to font_render_frame_layout
        font_render_frame_layout.addWidget(self.shadow_controls)
        
        # Initially disable shadow controls
        self._toggle_shadow_controls()
        
        # Reset to Defaults Button
        reset_btn_frame = QWidget()
        reset_btn_layout = QHBoxLayout(reset_btn_frame)
        reset_btn_layout.setContentsMargins(0, 15, 0, 10)
        reset_btn_layout.setSpacing(10)
        
        reset_defaults_btn = QPushButton("🔄 Reset to Defaults")
        reset_defaults_btn.clicked.connect(self._reset_rendering_to_defaults)
        reset_defaults_btn.setStyleSheet(
            "QPushButton { "
            "  background-color: #ff9800; "
            "  color: white; "
            "  padding: 8px 20px; "
            "  font-size: 10pt; "
            "  font-weight: bold; "
            "  border-radius: 4px; "
            "} "
            "QPushButton:hover { background-color: #fb8c00; }"
        )
        reset_btn_layout.addWidget(reset_defaults_btn)
        reset_btn_layout.addStretch()
        
        font_render_frame_layout.addWidget(reset_btn_frame)
        
        # Add decorative icon at the bottom of Font & Text Settings to balance the layout
        font_icon_frame = QWidget()
        font_icon_layout = QVBoxLayout(font_icon_frame)
        font_icon_layout.setContentsMargins(0, 8, 0, 3)
        font_icon_layout.setAlignment(Qt.AlignCenter)
        
        icon_path_font = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        if os.path.exists(icon_path_font):
            font_icon_label = QLabel()
            font_icon_pixmap = QPixmap(icon_path_font)
            try:
                from PySide6.QtGui import QScreen
                dpr = self.dialog.devicePixelRatio() if hasattr(self, 'dialog') and self.dialog else 1.0
            except Exception:
                dpr = 1.0
            target_logical = 180
            fitted = font_icon_pixmap.scaled(int(target_logical * dpr), int(target_logical * dpr),
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
            try:
                fitted.setDevicePixelRatio(dpr)
            except Exception:
                pass
            font_icon_label.setPixmap(fitted)
            font_icon_label.setFixedSize(target_logical, target_logical)
            font_icon_label.setAlignment(Qt.AlignCenter)
            font_icon_layout.addWidget(font_icon_label)
            
            # Optional: Add a subtle label
            font_icon_text = QLabel("Glossarion")
            font_icon_text.setStyleSheet("color: #5a9fd4; font-size: 7pt; font-weight: bold;")
            font_icon_text.setAlignment(Qt.AlignCenter)
            font_icon_layout.addWidget(font_icon_text)
        
        font_render_frame_layout.addWidget(font_icon_frame)
        self._pump_build_events()
        
        # Add font_render_frame to RIGHT COLUMN
        right_column_layout.addWidget(font_render_frame)
        
        # Control buttons - IN LEFT COLUMN
        # Check if ready based on selected provider
        # Get API key from main GUI - handle both Tkinter and PySide6
        try:
            if hasattr(self.main_gui.api_key_entry, 'text'):  # PySide6 QLineEdit
                has_api_key = bool(self.main_gui.api_key_entry.text().strip())
            elif hasattr(self.main_gui.api_key_entry, 'get'):  # Tkinter Entry
                has_api_key = bool(self.main_gui.api_key_entry.get().strip())
            else:
                has_api_key = False
        except:
            has_api_key = False
        
        # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
        if not has_api_key:
            _model = self.main_gui.config.get('model', '') if hasattr(self, 'main_gui') else ''
            try:
                from unified_api_client import UnifiedClient as _UC
                if not _UC._model_needs_api_key(_model):
                    has_api_key = True
            except Exception:
                pass
            
        provider = self.ocr_provider_value

        # Determine readiness based on provider
        if provider == 'google':
            has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
            is_ready = has_api_key and has_vision
        elif provider == 'azure':
            has_azure = bool(self.main_gui.config.get('azure_vision_key', ''))
            is_ready = has_api_key and has_azure
        elif provider == 'custom-api':
            is_ready = has_api_key  # Only needs API key
        else:
            # Local providers (manga-ocr, easyocr, etc.) only need API key for translation
            is_ready = has_api_key
        
        control_frame = QWidget()
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 15, 10, 10)
        control_layout.setSpacing(15)
        
        # Create start button with spinning icon
        self.start_button = QPushButton()
        self.start_button.clicked.connect(self._toggle_translation)
        self.start_button.setEnabled(is_ready)
        self.start_button.setMinimumHeight(120)  # Increased to prevent icon clipping
        # Set size policy to expand vertically to fill available space
        self.start_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create button content with icon and text (horizontal layout)
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)  # Changed to horizontal
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(15)  # Space between icon and text
        button_layout.setAlignment(Qt.AlignCenter)
        
        # Icon path
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        
        # Rotatable label class for animation
        class RotatableLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._rotation = 0
                self._original_pixmap = None
            
            def set_rotation(self, angle):
                self._rotation = angle
                if self._original_pixmap:
                    transform = QTransform()
                    transform.rotate(angle)
                    rotated = self._original_pixmap.transformed(transform, Qt.SmoothTransformation)
                    try:
                        rotated.setDevicePixelRatio(self._original_pixmap.devicePixelRatio())
                    except Exception:
                        pass
                    self.setPixmap(rotated)
            
            def get_rotation(self):
                return self._rotation
            
            rotation = Property(float, get_rotation, set_rotation)
            
            def set_original_pixmap(self, pixmap):
                self._original_pixmap = pixmap
                self.setPixmap(pixmap)
        
        # Icon container (HiDPI-aware 100x100)
        icon_container = QWidget()
        icon_container.setFixedSize(100, 100)
        icon_container.setStyleSheet("background-color: transparent;")
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setAlignment(Qt.AlignCenter)

        self.start_button_icon = RotatableLabel(icon_container)
        self.start_button_icon.setStyleSheet("background-color: transparent;")
        if os.path.exists(icon_path):
            from PySide6.QtCore import QSize
            icon = QIcon(icon_path)
            try:
                dpr = self.devicePixelRatioF()
            except Exception:
                dpr = 1.0
            target_logical = 100
            target_dpr = max(1.0, dpr)
            dev_px = int(target_logical * target_dpr)

            avail = icon.availableSizes()
            if avail:
                best = max(avail, key=lambda s: s.width() * s.height())
                pm = icon.pixmap(best * int(target_dpr))
            else:
                pm = icon.pixmap(QSize(dev_px, dev_px))

            if pm.isNull():
                pm = QPixmap(icon_path)

            if not pm.isNull():
                try:
                    pm.setDevicePixelRatio(target_dpr)
                except Exception:
                    pass
                fitted = pm.scaled(int(target_logical * target_dpr),
                                   int(target_logical * target_dpr),
                                   Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
                try:
                    fitted.setDevicePixelRatio(target_dpr)
                except Exception:
                    pass
                self.start_button_icon.set_original_pixmap(fitted)
                self.start_button_icon.setFixedSize(target_logical, target_logical)
        self.start_button_icon.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(self.start_button_icon)
        
        button_layout.addWidget(icon_container)
        
        # Create animations
        self.start_icon_spin_animation = QPropertyAnimation(self.start_button_icon, b"rotation")
        self.start_icon_spin_animation.setDuration(800)  # Faster spin: 0.8 seconds per rotation
        self.start_icon_spin_animation.setStartValue(0)
        self.start_icon_spin_animation.setEndValue(360)
        self.start_icon_spin_animation.setLoopCount(-1)
        self.start_icon_spin_animation.setEasingCurve(QEasingCurve.Linear)
        
        # Create a dedicated timer to keep animation smooth during heavy logging
        self._animation_refresh_timer = QTimer()
        self._animation_refresh_timer.setInterval(8)  # ~125fps for ultra-smooth animation
        # Process events on each tick to ensure animation frames are rendered
        def _refresh_animation():
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
            except:
                pass
        self._animation_refresh_timer.timeout.connect(_refresh_animation)
        
        self.start_icon_stop_animation = QPropertyAnimation(self.start_button_icon, b"rotation")
        self.start_icon_stop_animation.setDuration(800)
        self.start_icon_stop_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Text label
        self.start_button_text = QLabel("▶ Start Translation")
        self.start_button_text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Left-aligned vertically centered
        self.start_button_text.setStyleSheet("color: white; font-size: 14pt; font-weight: bold; background-color: transparent;")
        button_layout.addWidget(self.start_button_text)
        
        self.start_button.setLayout(button_layout)
        
        self.start_button.setStyleSheet(
            "QPushButton { "
            "  background-color: #28a745; "
            "  color: white; "
            "  padding: 28px 30px; "
            "  font-size: 14pt; "
            "  font-weight: bold; "
            "  border-radius: 8px; "
            "} "
            "QPushButton:hover { background-color: #218838; } "
            "QPushButton:disabled { "
            "  background-color: #2d2d2d; "
            "  color: #666666; "
            "}"
        )
        control_layout.addWidget(self.start_button, stretch=1)  # Give it stretch factor

        ocr_io_row = QWidget()
        ocr_io_layout = QHBoxLayout(ocr_io_row)
        ocr_io_layout.setContentsMargins(0, 0, 0, 0)
        ocr_io_layout.setSpacing(8)
        self.batch_ocr_import_btn = QPushButton("📥 Import OCR")
        self.batch_ocr_import_btn.setToolTip(
            "Click to select, or drag and drop, a saved OCR JSON session. "
            "Matching pages will skip OCR."
        )
        self.batch_ocr_import_btn.setAcceptDrops(True)
        self.batch_ocr_import_btn.setProperty("ocrDropActive", False)
        self.batch_ocr_import_btn.installEventFilter(self)
        self._ocr_import_drop_targets = {self.batch_ocr_import_btn}
        self.batch_ocr_import_btn.clicked.connect(self._import_batch_ocr_text)
        self.batch_ocr_open_btn = QPushButton("📂 Open Auto-Saved OCR Files")
        self.batch_ocr_open_btn.setToolTip(
            "Open the dedicated folder containing timestamped OCR and translation session files."
        )
        self.batch_ocr_open_btn.clicked.connect(self._open_automatic_ocr_folder)
        def style_ocr_io_button(button, background, border, hover, pressed):
            button.setMinimumHeight(36)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {background};
                    color: white;
                    border: 1px solid {border};
                    border-radius: 6px;
                    padding: 7px 16px;
                    font-size: 9pt;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {hover};
                    border-color: white;
                }}
                QPushButton:pressed {{
                    background-color: {pressed};
                }}
                QPushButton[ocrDropActive="true"] {{
                    background-color: #7a5a18;
                    border: 2px dashed #ffd166;
                    color: white;
                }}
                QPushButton:disabled {{
                    background-color: #252525;
                    color: #6f6f6f;
                    border-color: #3b3b3b;
                }}
            """)

        style_ocr_io_button(
            self.batch_ocr_import_btn,
            background="#245f7a",
            border="#3287aa",
            hover="#2d7696",
            pressed="#19475d",
        )
        style_ocr_io_button(
            self.batch_ocr_open_btn,
            background="#28734c",
            border="#3b9b69",
            hover="#338b5d",
            pressed="#1d5739",
        )
        ocr_io_layout.addWidget(self.batch_ocr_import_btn, stretch=1)
        ocr_io_layout.addWidget(self.batch_ocr_open_btn, stretch=1)
        control_layout.addWidget(ocr_io_row)

        # Add tooltip to show why button is disabled
        if not is_ready:
            reasons = []
            if not has_api_key:
                reasons.append("API key not configured")
            if provider == 'google' and not os.path.exists(self.main_gui.config.get('google_vision_credentials', '')):
                reasons.append("Google Vision credentials not set")
            elif provider == 'azure' and not self.main_gui.config.get('azure_vision_key', ''):
                reasons.append("Azure credentials not configured")
            tooltip_text = "Cannot start: " + ", ".join(reasons)
            self.start_button.setToolTip(tooltip_text)
        
        # Don't add control_frame to left_column - will be added to progress section instead
        
        # Add stretch to right column to balance
        right_column_layout.addStretch()
        
        # Add tabs with increased minimum height
        self.settings_tabs.addTab(left_column, "⚙️ Translation Settings")
        self.settings_tabs.addTab(right_column, "🎨 Rendering Settings")
        self.settings_tabs.setMinimumHeight(900)  # Increased height to fit content without scrolling
        
        # Add settings tabs to the tabs container (below the advanced button)
        tabs_container_layout.addWidget(self.settings_tabs)
        
        # Add tabs container (with advanced button on top) to main columns layout
        columns_layout.addWidget(tabs_container, stretch=1)
        
        # Add image preview widget directly (always visible) with higher stretch priority
        columns_layout.addWidget(self.image_preview_widget, stretch=2)
        
        # Make the columns container itself have proper size policy
        columns_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Add columns container to main layout
        main_layout.addWidget(columns_container)
        self._pump_build_events()
        
        # Progress frame
        progress_frame = QGroupBox("Progress")
        progress_frame_font = QFont('Arial', 10)
        progress_frame_font.setBold(True)
        progress_frame.setFont(progress_frame_font)
        progress_frame_layout = QVBoxLayout(progress_frame)
        progress_frame_layout.setContentsMargins(10, 10, 10, 8)
        progress_frame_layout.setSpacing(6)
        
        # Add start button at the top of progress section
        progress_frame_layout.addWidget(control_frame)
        
        # Overall progress
        self.progress_label = QLabel("Ready to start")
        progress_label_font = QFont('Arial', 9)
        self.progress_label.setFont(progress_label_font)
        self.progress_label.setStyleSheet("color: white;")
        progress_frame_layout.addWidget(self.progress_label)
        
        # Create and configure progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(18)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4a5568;
                border-radius: 3px;
                background-color: #2d3748;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: white;
            }
        """)
        progress_frame_layout.addWidget(self.progress_bar)
        
        # Current file status
        self.current_file_label = QLabel("")
        current_file_font = QFont('Arial', 10)
        self.current_file_label.setFont(current_file_font)
        self.current_file_label.setStyleSheet("color: lightgray;")
        progress_frame_layout.addWidget(self.current_file_label)
        
        # Pool tracker (shows loaded model instances)
        self.pool_tracker_label = QLabel()
        pool_tracker_font = QFont('Arial', 9)
        self.pool_tracker_label.setFont(pool_tracker_font)
        self.pool_tracker_label.setStyleSheet("color: #17a2b8;")
        self._update_pool_tracker_label()
        progress_frame_layout.addWidget(self.pool_tracker_label)
        
        # Start auto-update timer for pool tracker (update every 2 seconds)
        self.pool_update_timer = QTimer()
        self.pool_update_timer.timeout.connect(self._update_pool_tracker_label)
        self.pool_update_timer.start(2000)  # Update every 2 seconds
        
        main_layout.addWidget(progress_frame)
        
        # Log frame
        log_frame = QGroupBox("Translation Log")
        log_frame_font = QFont('Arial', 10)
        log_frame_font.setBold(True)
        log_frame.setFont(log_frame_font)
        log_frame_layout = QVBoxLayout(log_frame)
        log_frame_layout.setContentsMargins(10, 10, 10, 8)
        log_frame_layout.setSpacing(6)
        
        # Log text widget (QTextEdit handles scrolling automatically)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(600)  # Increased from 400 to 600 for better visibility
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #4a5568;
            }
        """)
        log_frame_layout.addWidget(self.log_text)
        self.log_text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_text.customContextMenuRequested.connect(self._show_log_context_menu)
        
        # Connect scrollbar to detect manual scrolling
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_log_scroll)

        # Auto-scroll helper button (appears when scrolled up)
        self.log_scroll_btn = QToolButton(self.log_text.viewport())
        self.log_scroll_btn.setText("Scroll to bottom")
        self.log_scroll_btn.setStyleSheet("QToolButton { background: #2d2d2d; color: white; border: 1px solid #4a5568; padding: 4px 8px; border-radius: 3px; }")
        self.log_scroll_btn.hide()
        self.log_scroll_btn.clicked.connect(self._scroll_log_to_bottom)
        # Track resize to keep button anchored
        self.log_text.installEventFilter(self)
        self.log_text.viewport().installEventFilter(self)
        QTimer.singleShot(0, self._update_log_scroll_button)
        
        main_layout.addWidget(log_frame)
        self._pump_build_events()
        
        # Restore persistent log from previous sessions
        self._restore_persistent_log()

    def _restore_persistent_log(self):
        """Restore log messages from persistent storage"""
        try:
            with MangaTranslationTab._persistent_log_lock:
                if MangaTranslationTab._persistent_log:
                    # PySide6 QTextEdit
                    color_map = {
                        'info': 'white',
                        'success': 'green',
                        'warning': 'orange',
                        'error': 'red',
                        'debug': 'lightblue'
                    }
                    for message, level in MangaTranslationTab._persistent_log:
                        if self._should_suppress_debug_log(message, level):
                            continue
                        color = color_map.get(level, 'white')
                        self.log_text.setTextColor(QColor(color))
                        self.log_text.append(message)
        except Exception as e:
            print(f"Failed to restore persistent log: {e}")
    
    def _show_help_dialog(self, title: str, message: str):
        """Show a help dialog with the given title and message"""
        # Create a PySide6 dialog
        help_dialog = QDialog(self.dialog)
        help_dialog.setWindowTitle(title)
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.26)  # 26% of screen width
        height = int(screen.height() * 0.37)  # 37% of screen height
        help_dialog.resize(width, height)
        help_dialog.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(help_dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        
        # Icon and title
        title_frame = QWidget()
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 10)
        
        icon_label = QLabel("ℹ️")
        icon_font = QFont('Arial', 20)
        icon_label.setFont(icon_font)
        title_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_font = QFont('Arial', 12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        main_layout.addWidget(title_frame)
        
        # Help text in a scrollable text widget
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setPlainText(message)
        text_font = QFont('Arial', 10)
        text_widget.setFont(text_font)
        main_layout.addWidget(text_widget)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(help_dialog.accept)
        close_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 20px; }")
        main_layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        # Show the dialog
        help_dialog.exec()

    def _on_visual_context_toggle(self, state=None):
        """Handle visual context toggle"""
        # Determine the new state from the checkbox if available; fall back to signal state
        try:
            enabled = bool(self.visual_context_checkbox.isChecked()) if hasattr(self, 'visual_context_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        # Update backing value and persist
        self.visual_context_enabled_value = enabled
        self.main_gui.config['manga_visual_context_enabled'] = enabled
        
        # Update translator if it exists
        if self.translator:
            self.translator.visual_context_enabled = enabled
        
        # Save config
        if hasattr(self.main_gui, 'save_config'):
            self.main_gui.save_config(show_message=False)
        
        # Log the change
        if enabled:
            self._log("📷 Visual context ENABLED - Images will be sent to API", "info")
            self._log("   Make sure you're using a vision-capable model", "warning")
        else:
            self._log("📝 Visual context DISABLED - Text-only mode", "info")
            self._log("   Compatible with non-vision models (Claude, GPT-3.5, etc.)", "success")
 
    def _open_advanced_settings(self):
        """Open the manga advanced settings dialog"""
        try:
            def on_settings_saved(settings):
                """Callback when settings are saved"""
                # Update config with new settings
                self.main_gui.config['manga_settings'] = settings
                try:
                    self._apply_manga_image_request_quality_from_settings(settings, save=False)
                except Exception:
                    pass

                # Mirror critical font size values into nested settings (avoid legacy top-level min key)
                try:
                    rendering = settings.get('rendering', {}) if isinstance(settings, dict) else {}
                    font_sizing = settings.get('font_sizing', {}) if isinstance(settings, dict) else {}
                    min_from_dialog = rendering.get('auto_min_size', font_sizing.get('min_readable', font_sizing.get('min_size')))
                    max_from_dialog = rendering.get('auto_max_size', font_sizing.get('max_size'))
                    if min_from_dialog is not None:
                        ms = self.main_gui.config.setdefault('manga_settings', {})
                        rend = ms.setdefault('rendering', {})
                        font = ms.setdefault('font_sizing', {})
                        rend['auto_min_size'] = int(min_from_dialog)
                        font['min_size'] = int(min_from_dialog)
                        if hasattr(self, 'auto_min_size_value'):
                            self.auto_min_size_value = int(min_from_dialog)
                    if max_from_dialog is not None:
                        self.main_gui.config['manga_max_font_size'] = int(max_from_dialog)
                        if hasattr(self, 'max_font_size_value'):
                            self.max_font_size_value = int(max_from_dialog)
                except Exception:
                    pass

                # Persist mirrored values
                try:
                    if hasattr(self.main_gui, 'save_config'):
                        self.main_gui.save_config(show_message=False)
                except Exception:
                    pass
                
                # Reload settings in translator if it exists
                if self.translator:
                    self._log("📋 Reloading settings in translator...", "info")
                    # The translator will pick up new settings on next operation
                
                self._log("✅ Advanced settings saved and applied", "success")
            
            # Open the settings dialog
            # MangaSettingsDialog is PySide6-based, so pass the manga integration dialog as parent
            self.main_gui.manga_settings_dialog = MangaSettingsDialog(
                parent=self.dialog,  # Use PySide6 manga integration dialog as parent
                main_gui=self.main_gui,
                config=self.main_gui.config,
                callback=on_settings_saved
            )
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            self._log(f"❌ Error opening settings dialog: {str(e)}", "error")
            QMessageBox.critical(self.dialog, "Error", f"Failed to open settings dialog:\n{str(e)}")
        
    def _toggle_font_size_mode(self):
        """Toggle between auto, fixed size and multiplier modes"""
        mode = self.font_size_mode_value
        
        # Handle main frames (fixed size and multiplier)
        if hasattr(self, 'fixed_size_frame') and hasattr(self, 'multiplier_frame'):
            if mode == "fixed":
                self.fixed_size_frame.show()
                self.multiplier_frame.hide()
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.hide()
            elif mode == "multiplier":
                self.fixed_size_frame.hide()
                self.multiplier_frame.show()
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.show()
            else:  # auto
                self.fixed_size_frame.hide()
                self.multiplier_frame.hide()
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.hide()
        
        # MIN/MAX FIELDS ARE ALWAYS VISIBLE - NEVER HIDE THEM
        # They are packed at creation time and stay visible in all modes
        
        # Only save and apply if we're not initializing
        if not hasattr(self, '_initializing') or not self._initializing:
            self._save_rendering_settings()
            self._apply_rendering_settings()

    def _validate_font_size_range_manga(self, which):
        """Ensure min font size doesn't exceed max font size in manga integration"""
        min_val = self.min_size_spinbox.value()
        max_val = self.max_size_spinbox.value()
        
        if min_val > max_val:
            # Adjust min to match max since that's the safer default
            self.min_size_spinbox.blockSignals(True)
            self.min_size_spinbox.setValue(max_val)
            self.min_size_spinbox.blockSignals(False)
            self.auto_min_size_value = max_val
        
        # Always save and apply after validation
        self._save_rendering_settings()
        self._apply_rendering_settings()
    
    def _update_multiplier_label(self, value):
        """Update multiplier label and value variable"""
        self.font_size_multiplier_value = float(value)  # UPDATE THE VALUE VARIABLE!
        self.multiplier_label.setText(f"{float(value):.1f}x")

    def _on_line_spacing_changed(self, value):
        """Update line spacing value label and value variable"""
        self.line_spacing_value = float(value)  # UPDATE THE VALUE VARIABLE!
        try:
            if hasattr(self, 'line_spacing_value_label'):
                self.line_spacing_value_label.setText(f"{float(value):.2f}")
        except Exception:
            pass
    
    def _on_shadow_blur_changed(self, value):
        """Update shadow blur label, persist, apply, and reflow overlays"""
        self.shadow_blur_value = int(float(value))
        try:
            if hasattr(self, 'shadow_blur_value_label'):
                self.shadow_blur_value_label.setText(f"{int(float(value))}")
        except Exception:
            pass
        # Persist and apply live
        try:
            self._save_rendering_settings()
            self._apply_rendering_settings()
        except Exception:
            pass
        # Reflow overlays to reflect new blur
        try:
            ImageRenderer._relayout_all_overlays_for_current_image(self, )
        except Exception:
            pass
    
    def _on_ft_only_bg_opacity_changed(self):
        """Handle free text only background opacity checkbox change (PySide6)"""
        try:
            self.free_text_only_bg_opacity_value = bool(self.ft_only_checkbox.isChecked())
            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config'):
                self.main_gui.config['manga_free_text_only_bg_opacity'] = self.free_text_only_bg_opacity_value
            if hasattr(self, 'translator') and self.translator:
                self.translator.free_text_only_bg_opacity = self.free_text_only_bg_opacity_value
        except Exception:
            pass

    def _on_safe_area_scale_changed(self, value: float):
        """Handle safe area scale change (PySide6)"""
        try:
            self.safe_area_scale_value = float(value)
            if hasattr(self, 'safe_area_scale_value_label'):
                self.safe_area_scale_value_label.setText(f"{float(value):.2f}")
        except Exception:
            pass
    
    def _update_color_preview(self, event=None):
        """Update the font color preview"""
        r = self.text_color_r_value
        g = self.text_color_g_value
        b = self.text_color_b_value
        if hasattr(self, 'color_preview_frame'):
            self.color_preview_frame.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #5a9fd4;")
        # Auto-save and apply on change
        if event is not None:  # Only save on user interaction, not initial load
            self._save_rendering_settings()
            self._apply_rendering_settings()
    
    def _update_shadow_preview(self, event=None):
        """Update the shadow color preview and optionally apply live."""
        r = self.shadow_color_r_value
        g = self.shadow_color_g_value
        b = self.shadow_color_b_value
        if hasattr(self, 'shadow_preview_frame'):
            self.shadow_preview_frame.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #5a9fd4;")
        # Auto-save and apply on change
        if event is not None:
            self._save_rendering_settings()
            self._apply_rendering_settings()
            try:
                ImageRenderer._relayout_all_overlays_for_current_image(self, )
            except Exception:
                pass
    
    def _toggle_azure_key_visibility(self, state):
        """Toggle visibility of Azure Computer Vision API key"""
        from PySide6.QtWidgets import QLineEdit
        from PySide6.QtCore import Qt
        
        # Check the checkbox state directly to be sure
        is_checked = self.show_azure_key_checkbox.isChecked()
        
        if is_checked:
            # Show the key
            self.azure_key_entry.setEchoMode(QLineEdit.Normal)
        else:
            # Hide the key
            self.azure_key_entry.setEchoMode(QLineEdit.Password)

    def _on_azure_credentials_change(self, text=None):
        """Save Azure Computer Vision credentials when the user edits them."""
        try:
            key = self.azure_key_entry.text() if hasattr(self, 'azure_key_entry') else ''
            endpoint = self.azure_endpoint_entry.text() if hasattr(self, 'azure_endpoint_entry') else ''

            self.main_gui.config['azure_vision_key'] = key
            self.main_gui.config['azure_vision_endpoint'] = endpoint

            # Keep status indicator fresh
            self._check_provider_status()

            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
        except Exception as e:
            self._log(f"Error saving Azure credentials: {e}", "error")
    
    def _toggle_azure_doc_intel_key_visibility(self, state):
        """Toggle visibility of Azure Document Intelligence API key"""
        from PySide6.QtWidgets import QLineEdit
        from PySide6.QtCore import Qt
        
        # Check the checkbox state directly to be sure
        is_checked = self.show_azure_doc_key_checkbox.isChecked()
        
        if is_checked:
            # Show the key
            self.azure_doc_intel_key_entry.setEchoMode(QLineEdit.Normal)
        else:
            # Hide the key
            self.azure_doc_intel_key_entry.setEchoMode(QLineEdit.Password)
    
    def _on_azure_doc_intel_credentials_change(self, text=None):
        """Save Azure Document Intelligence credentials when they change"""
        try:
            # Get current values
            key = self.azure_doc_intel_key_entry.text() if hasattr(self, 'azure_doc_intel_key_entry') else ''
            endpoint = self.azure_doc_intel_endpoint_entry.text() if hasattr(self, 'azure_doc_intel_endpoint_entry') else ''
            
            # Save to config
            self.main_gui.config['azure_document_intelligence_key'] = key
            self.main_gui.config['azure_document_intelligence_endpoint'] = endpoint
            
            # Save config file
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            
            # Update status
            self._check_provider_status()
        except Exception as e:
            self._log(f"Error saving Azure Document Intelligence credentials: {e}", "error")
    
    def _toggle_shadow_controls(self):
        """Enable/disable shadow controls based on checkbox"""
        if self.shadow_enabled_value:
            if hasattr(self, 'shadow_controls'):
                self.shadow_controls.setEnabled(True)
        else:
            if hasattr(self, 'shadow_controls'):
                self.shadow_controls.setEnabled(False)

    def _enable_widget_tree(self, widget):
        """Recursively enable a widget and its children (PySide6 version)"""
        try:
            widget.setEnabled(True)
        except:
            pass
        # PySide6 way to iterate children
        try:
            for child in widget.children():
                if hasattr(child, 'setEnabled'):
                    self._enable_widget_tree(child)
        except:
            pass
    
    def _disable_widget_tree(self, widget):
        """Recursively disable a widget and its children (PySide6 version)"""
        try:
            widget.setEnabled(False)
        except:
            pass
        # PySide6 way to iterate children
        try:
            for child in widget.children():
                if hasattr(child, 'setEnabled'):
                    self._disable_widget_tree(child)
        except:
            pass
        
    def _on_manga_image_request_quality_toggled(self, enabled: bool):
        if getattr(self, '_syncing_manga_image_request_quality_widgets', False):
            self.manga_image_request_quality_enabled_value = bool(enabled)
            return
        try:
            self.manga_image_request_quality_enabled_value = bool(enabled)
            ms = self.main_gui.config.setdefault('manga_settings', {})
            comp = ms.setdefault('compression', {})
            comp['enabled'] = bool(enabled)
            self._refresh_manga_image_request_quality_controls()
            self._sync_manga_image_request_quality_env()
            self._sync_manga_image_request_quality_dialog_widgets()
            self._save_rendering_settings()
        except Exception as e:
            self._log(f"⚠️ Failed to update image request quality setting: {e}", "warning")

    def _apply_manga_image_request_quality_from_settings(self, settings: Dict[str, Any], save: bool = True):
        """Apply dialog-side image request quality changes to the manga panel immediately."""
        try:
            if not isinstance(settings, dict):
                return
            ms = self.main_gui.config.setdefault('manga_settings', {})
            src_comp = (settings.get('compression', {}) or {})
            comp = ms.setdefault('compression', {})
            for key in ('enabled', 'format', 'jpeg_quality', 'png_compress_level', 'webp_quality'):
                if key in src_comp:
                    comp[key] = src_comp[key]

            self.manga_image_request_quality_enabled_value = bool(comp.get('enabled', False))

            self._syncing_manga_image_request_quality_widgets = True
            try:
                if hasattr(self, 'manga_image_request_quality_checkbox'):
                    # Do not block signals here: the custom checkmark overlay relies on stateChanged.
                    self.manga_image_request_quality_checkbox.setChecked(self.manga_image_request_quality_enabled_value)
            finally:
                self._syncing_manga_image_request_quality_widgets = False

            self._refresh_manga_image_request_quality_controls()
            self._sync_manga_image_request_quality_env()
            self._sync_manga_image_request_quality_dialog_widgets()
            if save and hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
        except Exception as e:
            self._log(f"⚠️ Failed to apply live image request quality setting: {e}", "warning")

    def _sync_manga_image_request_quality_dialog_widgets(self):
        """Mirror manga-panel image request quality changes into the open settings dialog."""
        try:
            dialog = getattr(self.main_gui, 'manga_settings_dialog', None)
            if not dialog:
                return
            comp = ((self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {}) or {})
            if not hasattr(dialog, 'compression_enabled'):
                return

            dialog._syncing_image_request_quality_widgets = True
            try:
                if hasattr(dialog, 'compression_enabled'):
                    # Keep signals unblocked so the styled checkbox checkmark updates.
                    dialog.compression_enabled.setChecked(bool(comp.get('enabled', False)))
                if hasattr(dialog, 'compression_format_combo'):
                    dialog.compression_format_combo.blockSignals(True)
                    dialog.compression_format_combo.setCurrentText(str(comp.get('format', 'jpeg') or 'jpeg').lower())
                    dialog.compression_format_combo.blockSignals(False)
                if hasattr(dialog, 'jpeg_quality_spin'):
                    dialog.jpeg_quality_spin.blockSignals(True)
                    dialog.jpeg_quality_spin.setValue(int(comp.get('jpeg_quality', 85)))
                    dialog.jpeg_quality_spin.blockSignals(False)
                if hasattr(dialog, 'png_level_spin'):
                    dialog.png_level_spin.blockSignals(True)
                    dialog.png_level_spin.setValue(int(comp.get('png_compress_level', 6)))
                    dialog.png_level_spin.blockSignals(False)
                if hasattr(dialog, 'webp_quality_spin'):
                    dialog.webp_quality_spin.blockSignals(True)
                    dialog.webp_quality_spin.setValue(int(comp.get('webp_quality', 85)))
                    dialog.webp_quality_spin.blockSignals(False)
                if hasattr(dialog, '_toggle_compression_format'):
                    dialog._toggle_compression_format()
                if hasattr(dialog, '_toggle_compression_enabled'):
                    dialog._toggle_compression_enabled()
            finally:
                dialog._syncing_image_request_quality_widgets = False
        except Exception:
            pass

    def _refresh_manga_image_request_quality_controls(self):
        try:
            comp = ((self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {}) or {})
            fmt = str(comp.get('format', 'jpeg') or 'jpeg').strip().lower()
            if fmt not in ('jpeg', 'png', 'webp'):
                fmt = 'jpeg'
            enabled = bool(comp.get('enabled', False))
            if hasattr(self, 'manga_image_request_format_combo'):
                self.manga_image_request_format_combo.blockSignals(True)
                self.manga_image_request_format_combo.setCurrentText(fmt)
                self.manga_image_request_format_combo.setEnabled(enabled)
                self.manga_image_request_format_combo.blockSignals(False)
            if hasattr(self, 'manga_image_request_quality_spinbox'):
                self.manga_image_request_quality_spinbox.blockSignals(True)
                if fmt == 'png':
                    self.manga_image_request_quality_label.setText("PNG Level:")
                    self.manga_image_request_quality_spinbox.setRange(0, 9)
                    self.manga_image_request_quality_spinbox.setValue(int(comp.get('png_compress_level', 6)))
                elif fmt == 'webp':
                    self.manga_image_request_quality_label.setText("WEBP Quality:")
                    self.manga_image_request_quality_spinbox.setRange(1, 100)
                    self.manga_image_request_quality_spinbox.setValue(int(comp.get('webp_quality', 85)))
                else:
                    self.manga_image_request_quality_label.setText("JPEG Quality:")
                    self.manga_image_request_quality_spinbox.setRange(1, 95)
                    self.manga_image_request_quality_spinbox.setValue(int(comp.get('jpeg_quality', 85)))
                self.manga_image_request_quality_spinbox.setEnabled(enabled)
                from PySide6.QtWidgets import QGraphicsOpacityEffect
                if not enabled:
                    eff = QGraphicsOpacityEffect(self.manga_image_request_quality_spinbox)
                    eff.setOpacity(0.35)
                    self.manga_image_request_quality_spinbox.setGraphicsEffect(eff)
                else:
                    self.manga_image_request_quality_spinbox.setGraphicsEffect(None)
                self.manga_image_request_quality_spinbox.blockSignals(False)
            if hasattr(self, 'manga_image_request_quality_label'):
                self.manga_image_request_quality_label.setEnabled(enabled)
        except Exception:
            pass

    def _on_manga_image_request_format_changed(self, fmt: str):
        try:
            ms = self.main_gui.config.setdefault('manga_settings', {})
            comp = ms.setdefault('compression', {})
            comp['format'] = str(fmt or 'jpeg').strip().lower()
            if hasattr(self, 'manga_image_request_quality_checkbox'):
                comp['enabled'] = bool(self.manga_image_request_quality_checkbox.isChecked())
            self._refresh_manga_image_request_quality_controls()
            self._sync_manga_image_request_quality_env()
            self._sync_manga_image_request_quality_dialog_widgets()
            self._save_rendering_settings()
        except Exception as e:
            self._log(f"⚠️ Failed to update image request format: {e}", "warning")

    def _on_manga_image_request_quality_value_changed(self, value: int):
        try:
            ms = self.main_gui.config.setdefault('manga_settings', {})
            comp = ms.setdefault('compression', {})
            fmt = str(comp.get('format', 'jpeg') or 'jpeg').strip().lower()
            if fmt == 'png':
                comp['png_compress_level'] = int(value)
            elif fmt == 'webp':
                comp['webp_quality'] = int(value)
            else:
                comp['jpeg_quality'] = int(value)
            if hasattr(self, 'manga_image_request_quality_checkbox'):
                comp['enabled'] = bool(self.manga_image_request_quality_checkbox.isChecked())
            self._sync_manga_image_request_quality_env()
            self._sync_manga_image_request_quality_dialog_widgets()
            self._save_rendering_settings()
        except Exception as e:
            self._log(f"⚠️ Failed to update image request quality: {e}", "warning")
    
    def _reset_rendering_to_defaults(self):
        """Reset all rendering settings to their default values"""
        from PySide6.QtWidgets import QMessageBox
        
        # Confirm with user
        reply = QMessageBox.question(
            self.dialog,
            "Reset to Defaults",
            "Are you sure you want to reset all rendering settings to their default values?\n\nThis will reset:\n" +
            "• Background opacity and style\n" +
            "• Font size and style settings\n" +
            "• Text color and shadow settings\n" +
            "• Auto-fit style and text wrapping\n" +
            "• All other rendering options",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Background settings
            self.bg_opacity_value = 0
            self.free_text_only_bg_opacity_value = False
            self.bg_style_value = 'circle'
            self.bg_reduction_value = 1.0
            
            # Font settings
            self.font_size_value = 0
            self.font_size_mode_value = 'auto'
            self.font_size_multiplier_value = 1.0
            self.selected_font_path = None
            self.font_style_value = 'Default'
            
            # Auto fit style
            self.auto_fit_style_value = 'compact'
            self.auto_min_size_value = 8
            self.max_font_size_value = 48
            
            # Text wrapping and constraints
            self.strict_text_wrapping_value = True
            self.constrain_to_bubble_value = True
            self.force_caps_lock_value = True
            
            # Font algorithm settings
            self.font_algorithm_value = 'smart'
            self.prefer_larger_value = True
            self.bubble_size_factor_value = True
            self.line_spacing_value = 1.3
            
            # Text color
            self.text_color_r_value = 102
            self.text_color_g_value = 0
            self.text_color_b_value = 0
            
            # Shadow settings
            self.shadow_enabled_value = True
            self.shadow_color_r_value = 255
            self.shadow_color_g_value = 255
            self.shadow_color_b_value = 255
            self.shadow_offset_x_value = 2
            self.shadow_offset_y_value = 2
            self.shadow_blur_value = 0
            
            # Safe area
            self.safe_area_enabled_value = False
            self.safe_area_scale_value = 1.0
            
            # Update UI widgets
            if hasattr(self, 'opacity_slider'):
                self.opacity_slider.setValue(self.bg_opacity_value)
            if hasattr(self, 'ft_only_checkbox'):
                self.ft_only_checkbox.setChecked(self.free_text_only_bg_opacity_value)
            if hasattr(self, 'reduction_slider'):
                self.reduction_slider.setValue(self.bg_reduction_value)
            if hasattr(self, 'bg_style_combo'):
                self.bg_style_combo.setCurrentText(self.bg_style_value)
            
            # Update font mode
            if hasattr(self, 'font_size_mode_group'):
                for button in self.font_size_mode_group.buttons():
                    btn_text = button.text().strip().lower()
                    if btn_text.startswith('auto'):
                        button.setChecked(True)
                        break
            # Update font algorithm radio
            if hasattr(self, 'font_algorithm_group'):
                for button in self.font_algorithm_group.buttons():
                    if button.text().lower() == 'smart':
                        button.setChecked(True)
                        break
            
            # Update auto fit style
            if hasattr(self, 'auto_fit_style_group'):
                for button in self.auto_fit_style_group.buttons():
                    if button.text().lower() == 'compact':
                        button.setChecked(True)
            
            # Update text wrapping checkbox
            if hasattr(self, 'strict_wrap_checkbox'):
                self.strict_wrap_checkbox.setChecked(True)
            
            # Update constrain checkbox
            if hasattr(self, 'constrain_checkbox'):
                self.constrain_checkbox.setChecked(True)
            
            # Update force caps checkbox
            if hasattr(self, 'force_caps_checkbox'):
                self.force_caps_checkbox.setChecked(True)
            
            # Update shadow checkbox
            if hasattr(self, 'shadow_enabled_checkbox'):
                self.shadow_enabled_checkbox.setChecked(True)
            
            # Update min/max font size spinboxes
            if hasattr(self, 'min_size_spinbox'):
                self.min_size_spinbox.setValue(self.auto_min_size_value)
            if hasattr(self, 'max_size_spinbox'):
                self.max_size_spinbox.setValue(self.max_font_size_value)
            # Update fixed size spinbox
            if hasattr(self, 'font_size_spinbox'):
                self.font_size_spinbox.setValue(self.font_size_value)
            # Update multiplier control and label
            if hasattr(self, 'multiplier_slider'):
                self.multiplier_slider.setValue(self.font_size_multiplier_value)
            if hasattr(self, 'multiplier_label'):
                self._update_multiplier_label(self.font_size_multiplier_value)
            
            # Update line spacing
            if hasattr(self, 'line_spacing_spinbox'):
                self.line_spacing_spinbox.setValue(self.line_spacing_value)
            if hasattr(self, 'line_spacing_value_label'):
                self.line_spacing_value_label.setText(f"{self.line_spacing_value:.2f}")
            
            # Update prefer larger checkbox
            if hasattr(self, 'prefer_larger_checkbox'):
                self.prefer_larger_checkbox.setChecked(self.prefer_larger_value)
            
            # Update bubble size factor checkbox
            if hasattr(self, 'bubble_size_factor_checkbox'):
                self.bubble_size_factor_checkbox.setChecked(self.bubble_size_factor_value)
            
            # Update safe area controls
            if hasattr(self, 'safe_area_enabled_checkbox'):
                self.safe_area_enabled_checkbox.setChecked(self.safe_area_enabled_value)
            if hasattr(self, 'safe_area_scale_spinbox'):
                self.safe_area_scale_spinbox.setValue(self.safe_area_scale_value)
            
            # Update font combo
            if hasattr(self, 'font_combo'):
                self.font_combo.setCurrentText(self.font_style_value)
            
            # Update color displays
            if hasattr(self, 'rgb_label'):
                self.rgb_label.setText(f"RGB({self.text_color_r_value},{self.text_color_g_value},{self.text_color_b_value})")
            self._update_color_preview(None)
            
            if hasattr(self, 'shadow_rgb_label'):
                self.shadow_rgb_label.setText(f"RGB({self.shadow_color_r_value},{self.shadow_color_g_value},{self.shadow_color_b_value})")
            self._update_shadow_preview(None)

            # Update shadow offset/blur widgets
            if hasattr(self, 'shadow_offset_x_spinbox'):
                self.shadow_offset_x_spinbox.setValue(self.shadow_offset_x_value)
            if hasattr(self, 'shadow_offset_y_spinbox'):
                self.shadow_offset_y_spinbox.setValue(self.shadow_offset_y_value)
            if hasattr(self, 'shadow_blur_spinbox'):
                self.shadow_blur_spinbox.setValue(self.shadow_blur_value)
            if hasattr(self, 'shadow_blur_value_label'):
                self.shadow_blur_value_label.setText(f"{self.shadow_blur_value}")
            
            # Trigger font mode change to update visibility
            if hasattr(self, '_toggle_font_size_mode'):
                self._toggle_font_size_mode()
            
            # Toggle shadow controls visibility
            if hasattr(self, '_toggle_shadow_controls'):
                self._toggle_shadow_controls()
            
            # Update multiplier label
            if hasattr(self, '_update_multiplier_label'):
                self._update_multiplier_label(self.font_size_multiplier_value)
            
            # Update opacity label
            if hasattr(self, '_update_opacity_label'):
                self._update_opacity_label(self.bg_opacity_value)
            
            # Update reduction label
            if hasattr(self, '_update_reduction_label'):
                self._update_reduction_label(self.bg_reduction_value)
            
            # Save and apply settings
            self._save_rendering_settings()
            self._apply_rendering_settings()
            
            # Show confirmation
            QMessageBox.information(
                self.dialog,
                "Settings Reset",
                "All rendering settings have been reset to their default values."
            )
            
        except Exception as e:
            QMessageBox.warning(
                self.dialog,
                "Error",
                f"Failed to reset settings: {str(e)}"
            )
    
    def _on_context_toggle(self, state=None):
        """Handle full page context toggle"""
        # Read the checkbox state to update the backing value
        try:
            enabled = bool(self.context_checkbox.isChecked()) if hasattr(self, 'context_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self.full_page_context_value = enabled
        # Persist via unified save path
        self._save_rendering_settings()
    
    def _edit_context_prompt(self):
        """Open dialog to edit full page context prompt and OCR prompt"""
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTextEdit,
                                        QPushButton, QHBoxLayout, QMessageBox)
        from PySide6.QtCore import Qt
        
        # Create PySide6 dialog
        dialog = QDialog(self.dialog)
        dialog.setWindowTitle("Edit Prompts")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.37)  # 37% of screen width
        height = int(screen.height() * 0.58)  # 58% of screen height
        dialog.setMinimumSize(width, height)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel(
            "Edit the prompt used for full page context translation.\n"
            "This will be appended to the main translation system prompt."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Full Page Context label
        context_label = QLabel("Full Page Context Prompt:")
        font = context_label.font()
        font.setBold(True)
        context_label.setFont(font)
        layout.addWidget(context_label)
        
        # Text editor for context
        text_editor = QTextEdit()
        text_editor.setMinimumHeight(200)
        text_editor.setPlainText(self.full_page_context_prompt)
        layout.addWidget(text_editor)
        
        # OCR Prompt label
        ocr_label = QLabel("OCR System Prompt:")
        ocr_label.setFont(font)
        layout.addWidget(ocr_label)
        
        # Text editor for OCR
        ocr_editor = QTextEdit()
        ocr_editor.setMinimumHeight(200)
        
        # Get current OCR prompt
        if hasattr(self, 'ocr_prompt'):
            ocr_editor.setPlainText(self.ocr_prompt)
        else:
            ocr_editor.setPlainText("")
        
        layout.addWidget(ocr_editor)
        
        def save_prompt():
            self.full_page_context_prompt = text_editor.toPlainText().strip()
            self.ocr_prompt = ocr_editor.toPlainText().strip()
            
            # Save to config
            self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
            self.main_gui.config['manga_ocr_prompt'] = self.ocr_prompt
            
            # Debug: Verify the config was updated
            print(f"[OCR_PROMPT_SAVE] Saved OCR prompt to config: {len(self.ocr_prompt)} chars")
            print(f"[OCR_PROMPT_SAVE] Config has key 'manga_ocr_prompt': {'manga_ocr_prompt' in self.main_gui.config}")
            
            self._save_rendering_settings()
            self._log("✅ Updated prompts", "success")
            dialog.accept()
        
        def reset_prompt():
            choice = QMessageBox.warning(
                dialog,
                "Reset Prompts",
                "Reset both prompts to their defaults? This replaces the text currently shown in both editors. "
                "Changes are saved only when you click Save.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

            default_prompt = (
                "You will receive multiple text segments from a manga page, each prefixed with an index like [0], [1], etc. "
                "Translate each segment considering the context of all segments together. "
                "Maintain consistency in character names, tone, and style across all translations.\n\n"
                "CRITICAL: Return your response as a valid JSON object where each key includes BOTH the index prefix "
                "AND the original text EXACTLY as provided (e.g., '[0] こんにちは'), and each value is the translation.\n"
                "This is essential for correct mapping - do not modify or omit the index prefixes!\n\n"
                "Make sure to properly escape any special characters in the JSON:\n"
                "- Use \\n for newlines\n"
                "- Use \\\" for quotes\n"
                "- Use \\\\ for backslashes\n\n"
                "Example:\n"
                '{\n'
                '  "[0] こんにちは": "Hello",\n'
                '  "[1] ありがとう": "Thank you",\n'
                '  "[2] さようなら": "Goodbye"\n'
                '}\n\n'
                'REMEMBER: Keep the [index] prefix in each JSON key exactly as shown in the input!'
            )
            text_editor.setPlainText(default_prompt)
            
            ocr_editor.setPlainText(self._default_manga_ocr_prompt())
        
        # Button layout
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(save_prompt)
        button_layout.addWidget(save_btn)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(reset_prompt)
        button_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(100)  # Ensure enough width for text
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Show dialog
        dialog.exec()

    def _on_manga_glossary_toggle(self, state=None):
        """Persist manga glossary workflow toggle."""
        try:
            enabled = bool(self.manga_glossary_checkbox.isChecked()) if hasattr(self, 'manga_glossary_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self.manga_glossary_enabled_value = enabled
        self._save_rendering_settings()

    def _on_manga_compress_glossary_toggle(self, state=None):
        """Persist the shared glossary compression toggle from the manga panel."""
        try:
            enabled = bool(self.manga_compress_glossary_checkbox.isChecked()) if hasattr(self, 'manga_compress_glossary_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self._sync_compress_glossary_prompt_setting(enabled)
        self._save_rendering_settings()

    def _on_manga_glossary_debug_ocr_toggle(self, state=None):
        """Persist the manga glossary OCR/debug output toggle."""
        try:
            enabled = bool(self.manga_glossary_debug_ocr_checkbox.isChecked()) if hasattr(self, 'manga_glossary_debug_ocr_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self.manga_glossary_debug_ocr_text_value = enabled
        try:
            self.main_gui.config['manga_glossary_debug_ocr_text'] = enabled
        except Exception:
            pass
        self._save_rendering_settings()

    def _on_manga_split_first_level_subfolders_toggle(self, state=None) -> None:
        try:
            enabled = bool(self.manga_split_first_level_subfolders_checkbox.isChecked())
        except Exception:
            enabled = bool(state)
        self.manga_split_first_level_subfolders_value = enabled
        self.main_gui.config['manga_split_first_level_subfolders'] = enabled
        self.manga_process_group_index = 0
        self._refresh_manga_selection_status(allow_autoload=True)
        self._update_manga_preview_image_list_for_range()
        self._save_rendering_settings()

    def _on_manga_process_group_changed(self, index: int) -> None:
        groups = self._manga_current_process_groups()
        if not groups:
            self.manga_process_group_index = 0
            return
        index = max(0, min(int(index), len(groups) - 1))
        self.manga_process_group_index = index
        group = groups[index]
        if hasattr(self, 'manga_process_counter_label'):
            self.manga_process_counter_label.setText(f"{index + 1} / {len(groups)}")
        if hasattr(self, 'manga_process_prev_btn'):
            self.manga_process_prev_btn.setEnabled(index > 0)
        if hasattr(self, 'manga_process_next_btn'):
            self.manga_process_next_btn.setEnabled(index < len(groups) - 1)

        first_path = next((path for path in group.get('files', []) if path in self.selected_files), None)
        if first_path and hasattr(self, 'file_listbox') and self.file_listbox:
            try:
                self.file_listbox.setCurrentRow(self.selected_files.index(first_path))
            except Exception:
                pass
        self._update_manga_preview_image_list_for_range()
        self._update_manga_loaded_directory_label()

    def _warn_manga_source_mismatch(self, message: str) -> None:
        self._log(f"⚠️ {message}", "warning")
        try:
            QMessageBox.warning(self.dialog, "One Manga Folder Only", message)
        except Exception:
            pass

    def _clear_manga_selection_for_source_switch(self) -> None:
        """Clear the current manga selection before loading a different source folder."""
        try:
            self._clear_all()
        except Exception:
            try:
                self.file_listbox.clear()
                self.selected_files.clear()
                if hasattr(self, 'image_preview_widget'):
                    self.image_preview_widget.clear()
                    self.image_preview_widget.set_image_list([])
                self._current_image_path = None
            except Exception:
                pass
        self._reset_manga_glossary_selection_for_source_switch()
        self._log("📂 Cleared previous manga folder; loading new selection", "info")

    def _add_manga_file_item(self, filepath: str):
        """Append a file row and store the full path separately from the visible text."""
        if not hasattr(self, 'file_listbox') or not self.file_listbox:
            return None
        self.file_listbox.addItem(os.path.basename(filepath))
        item = self.file_listbox.item(self.file_listbox.count() - 1)
        if item:
            item.setData(Qt.UserRole, filepath)
            item.setToolTip(filepath)
        return item

    def _rebuild_manga_file_listbox(self, current_path: Optional[str] = None) -> None:
        """Rebuild rows from selected_files while preserving full-path row data."""
        if not hasattr(self, 'file_listbox') or not self.file_listbox:
            return
        if current_path is None:
            row = self.file_listbox.currentRow()
            if 0 <= row < self.file_listbox.count():
                item = self.file_listbox.item(row)
                current_path = item.data(Qt.UserRole) if item else None
            if not current_path and 0 <= row < len(self.selected_files):
                current_path = self.selected_files[row]

        self.file_listbox.blockSignals(True)
        try:
            self.file_listbox.clear()
            for filepath in self.selected_files:
                self._add_manga_file_item(filepath)
        finally:
            self.file_listbox.blockSignals(False)

        if current_path and current_path in self.selected_files:
            self.file_listbox.setCurrentRow(self.selected_files.index(current_path))
        elif self.file_listbox.count() > 0:
            self.file_listbox.setCurrentRow(0)
        self._update_manga_image_range_display()

    def _sync_manga_process_group_for_image(self, image_path: str) -> None:
        """Keep the process-group dropdown aligned with a clicked file-list row."""
        if not image_path:
            return
        groups = self._manga_current_process_groups()
        if len(groups) <= 1:
            return
        current_index = self._manga_selected_process_group_index()
        target_index = current_index
        for index, group in enumerate(groups):
            if image_path in (group.get('files', []) or []):
                target_index = index
                break
        if target_index == current_index:
            return

        self.manga_process_group_index = target_index
        combo = getattr(self, 'manga_process_combo', None)
        if combo:
            combo.blockSignals(True)
            try:
                if 0 <= target_index < combo.count():
                    combo.setCurrentIndex(target_index)
            finally:
                combo.blockSignals(False)

        if hasattr(self, 'manga_process_counter_label'):
            self.manga_process_counter_label.setText(f"{target_index + 1} / {len(groups)}")
        if hasattr(self, 'manga_process_prev_btn'):
            self.manga_process_prev_btn.setEnabled(target_index > 0)
        if hasattr(self, 'manga_process_next_btn'):
            self.manga_process_next_btn.setEnabled(target_index < len(groups) - 1)
        self._update_manga_loaded_directory_label()

    def _on_manga_image_range_changed(self, text: str = "") -> None:
        self.manga_image_range_value = str(text or '').strip()
        self._update_manga_image_range_display()
        self._update_manga_preview_image_list_for_range()

    def _show_file_list_context_menu(self, position) -> None:
        try:
            from PySide6.QtWidgets import QMenu
            item = self.file_listbox.itemAt(position)
            if not item:
                return
            filepath = item.data(Qt.UserRole)
            row = self.file_listbox.row(item)
            if not filepath and 0 <= row < len(self.selected_files):
                filepath = self.selected_files[row]
            if not filepath:
                return

            menu = QMenu(self.dialog if hasattr(self, 'dialog') else None)
            menu.setStyleSheet("QMenu::item { padding: 4px 16px; }")
            is_skipped = self._is_manually_skipped_processing_file(filepath)
            action = menu.addAction("▶️ Process This Image" if is_skipped else "⏭️ Skip Processing")
            action.triggered.connect(lambda checked=False, path=filepath: self._toggle_skip_processing_for_path(path))

            menu.addSeparator()
            can_reorder = bool(getattr(self, '_file_selection_editing_enabled', True))
            last_row = max(0, len(self.selected_files) - 1)
            move_up_action = menu.addAction("↑ Move Up One")
            move_up_action.setEnabled(can_reorder and row > 0)
            move_up_action.triggered.connect(
                lambda checked=False, path=filepath: self._move_manga_file_entry(path, 'up')
            )
            move_down_action = menu.addAction("↓ Move Down One")
            move_down_action.setEnabled(can_reorder and row < last_row)
            move_down_action.triggered.connect(
                lambda checked=False, path=filepath: self._move_manga_file_entry(path, 'down')
            )
            move_top_action = menu.addAction("⇈ Move to Top")
            move_top_action.setEnabled(can_reorder and row > 0)
            move_top_action.triggered.connect(
                lambda checked=False, path=filepath: self._move_manga_file_entry(path, 'top')
            )
            move_bottom_action = menu.addAction("⇊ Move to Bottom")
            move_bottom_action.setEnabled(can_reorder and row < last_row)
            move_bottom_action.triggered.connect(
                lambda checked=False, path=filepath: self._move_manga_file_entry(path, 'bottom')
            )
            menu.exec(self.file_listbox.mapToGlobal(position))
        except Exception as e:
            print(f"[FILE_SKIP] Failed to show file context menu: {e}")

    def _update_manga_image_range_display(self) -> None:
        """Grey out rows skipped by the current visible-order image range."""
        if not hasattr(self, 'file_listbox') or not self.file_listbox:
            return
        total = self.file_listbox.count()
        indices, error = self._parse_manga_image_range(total)
        active_filter = indices is not None and error is None

        active_brush = QBrush(QColor("#ffffff"))
        skipped_brush = QBrush(QColor("#7d8795"))
        active_background = QBrush(QColor("#2b2b2b"))
        skipped_background = QBrush(QColor("#202020"))
        manual_skipped = self._manual_skipped_processing_keys()

        for row in range(total):
            item = self.file_listbox.item(row)
            if not item:
                continue
            filepath = item.data(Qt.UserRole)
            if not filepath and row < len(self.selected_files):
                filepath = self.selected_files[row]
                item.setData(Qt.UserRole, filepath)
            basename = os.path.basename(
                filepath or _manga_filename_without_skip_prefix(item.text())
            )
            key = self._skip_key_for_path(filepath or basename)
            skipped_by_range = active_filter and (row + 1) not in indices
            skipped_manually = key in manual_skipped
            skipped = skipped_by_range or skipped_manually
            item.setText(f"{_MANGA_SKIP_PREFIX}{basename}" if skipped else basename)
            item.setForeground(skipped_brush if skipped else active_brush)
            item.setBackground(skipped_background if skipped else active_background)
            if skipped_manually:
                reason = "Skipped manually"
                if skipped_by_range:
                    reason += " and by image range"
                item.setToolTip(f"{reason} ({row + 1}): {filepath or basename}")
            elif skipped_by_range:
                item.setToolTip(f"Skipped by image range ({row + 1}): {filepath or basename}")
            else:
                item.setToolTip(filepath or basename)

        status = getattr(self, 'manga_image_range_status_label', None)
        if status:
            if error:
                status.setText(error)
                status.setStyleSheet("color: #ff6b6b; font-size: 8pt;")
            elif not active_filter:
                manual_count = sum(
                    1 for path in getattr(self, 'selected_files', []) or []
                    if self._skip_key_for_path(path) in manual_skipped
                )
                active_count = max(0, total - manual_count)
                if manual_count:
                    status.setText(f"Processing {active_count}/{total} ({manual_count} skipped)")
                else:
                    status.setText(f"All {total} images" if total else "All images")
                status.setStyleSheet("color: #9fb7d5; font-size: 8pt;")
            else:
                active_count = sum(
                    1 for idx, path in enumerate(getattr(self, 'selected_files', []) or [], start=1)
                    if idx in indices and self._skip_key_for_path(path) not in manual_skipped
                )
                skipped_count = max(0, total - active_count)
                suffix = f" ({skipped_count} skipped)" if skipped_count else ""
                status.setText(f"Processing {active_count}/{total}{suffix}")
                status.setStyleSheet("color: #9be38f; font-size: 8pt;")

    def _load_manga_custom_glossary(self):
        """Pick and load a custom glossary for manga translation."""
        start_dir = ""
        try:
            current_path = getattr(self, 'manga_custom_glossary_path', '') or self.main_gui.config.get('manga_custom_glossary_path', '')
            if current_path and os.path.exists(current_path):
                start_dir = os.path.dirname(current_path)
            else:
                glossary_dir = self._manga_glossary_output_dir()
                if glossary_dir and os.path.isdir(glossary_dir):
                    start_dir = glossary_dir
                else:
                    backup_dir = self._manga_glossary_backup_dir()
                    if backup_dir:
                        os.makedirs(backup_dir, exist_ok=True)
                        start_dir = backup_dir
        except Exception:
            start_dir = ""

        path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            "Load Manga Glossary",
            start_dir,
            "Glossary files (*.csv *.json *.txt);;All files (*.*)"
        )
        if not path:
            return

        try:
            glossary_text = self._load_manga_glossary_file_as_prompt_text(path)
            if not glossary_text:
                QMessageBox.warning(self.dialog, "Glossary Not Loaded", "No usable glossary entries were found in that file.")
                return

            source_path = path
            copied_path = self._copy_loaded_manga_glossary_to_output(source_path, glossary_text)
            active_path = copied_path or source_path

            self.manga_custom_glossary_path = active_path
            self.manga_loaded_glossary_text = glossary_text
            self.manga_generated_glossary_path = ''
            self.manga_glossary_auto_load_suppressed = False
            self.manga_glossary_auto_load_suppressed_root = ''
            self.manga_glossary_enabled_value = True
            self.main_gui.config['manga_custom_glossary_path'] = active_path
            self.main_gui.config['manga_generated_glossary_path'] = ''
            self.main_gui.config['manga_glossary_auto_load_suppressed'] = False
            self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = ''
            self.main_gui.config['manga_glossary_enabled'] = True
            setattr(self.main_gui, 'manga_custom_glossary_path', active_path)
            setattr(self.main_gui, 'manga_generated_glossary_path', '')
            setattr(self.main_gui, 'manga_generated_glossary_text', glossary_text)
            if hasattr(self, 'translator') and self.translator:
                self.translator.manga_generated_glossary_text = glossary_text
                self.translator.manga_generated_glossary_path = active_path
                self.translator._manga_glossary_prompt_logged = False
            if hasattr(self, 'manga_glossary_checkbox'):
                self.manga_glossary_checkbox.setChecked(True)
            self._update_manga_glossary_status_label()
            self._save_rendering_settings()
            entry_count = sum(1 for line in glossary_text.splitlines() if line.lstrip().startswith("* "))
            if copied_path and os.path.normcase(os.path.abspath(copied_path)) != os.path.normcase(os.path.abspath(source_path)):
                self._log(f"📚 Copied loaded manga glossary to: {copied_path}", "info")
            self._log(f"📚 Loaded custom manga glossary: {os.path.basename(active_path)} ({entry_count} entries)", "success")
        except Exception as e:
            self._log(f"❌ Failed to load manga glossary: {e}", "error")
            self._log(traceback.format_exc(), "debug")
            QMessageBox.critical(self.dialog, "Error", f"Failed to load glossary:\n\n{e}")

    def _confirm_delete_manga_glossary_folder(self, glossary_dir: str) -> bool:
        """Ask before deleting the output-side manga Glossary folder."""
        try:
            entries = []
            for root, _, files in os.walk(glossary_dir):
                for filename in files:
                    entries.append(os.path.join(root, filename))
            entries.sort()
            shown_entries = entries[:25]
            if entries:
                file_lines = "\n".join(f"  {path}" for path in shown_entries)
                if len(entries) > len(shown_entries):
                    file_lines += f"\n  ...and {len(entries) - len(shown_entries)} more file(s)"
            else:
                file_lines = "  (folder is empty)"
            message = (
                "This will delete the manga glossary folder used for auto-loading.\n\n"
                f"Files to delete:\n{file_lines}\n\n"
                "Continue?"
            )
            reply = QMessageBox.question(
                self.dialog,
                "Delete Manga Glossary Folder?",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            return reply == QMessageBox.Yes
        except Exception:
            return False

    def _clear_manga_custom_glossary(self):
        """Clear loaded/generated manga glossary path/text and suppress auto-reload."""
        deleted_glossary_dir = ''
        try:
            glossary_dir = os.path.abspath(self._manga_glossary_output_dir())
            # This button is scoped to manga glossary state. Only remove the
            # configured output-side Glossary subfolder; backups stay intact.
            if glossary_dir and os.path.basename(os.path.normpath(glossary_dir)).lower() == 'glossary' and os.path.isdir(glossary_dir):
                if not self._confirm_delete_manga_glossary_folder(glossary_dir):
                    self._log("📚 Manga glossary clear cancelled", "info")
                    return
                shutil.rmtree(glossary_dir)
                deleted_glossary_dir = glossary_dir
        except Exception as delete_err:
            self._log(f"⚠️ Could not delete manga glossary folder: {delete_err}", "warning")

        self.manga_custom_glossary_path = ''
        self.manga_loaded_glossary_text = ''
        self.manga_generated_glossary_text = ''
        self.manga_generated_glossary_entries = []
        self.manga_generated_glossary_path = ''
        self.manga_glossary_auto_load_suppressed = True
        self.manga_glossary_auto_load_suppressed_root = self._current_manga_source_dir()
        self.main_gui.config['manga_custom_glossary_path'] = ''
        self.main_gui.config['manga_generated_glossary_path'] = ''
        self.main_gui.config['manga_glossary_auto_load_suppressed'] = True
        self.main_gui.config['manga_glossary_auto_load_suppressed_root'] = self.manga_glossary_auto_load_suppressed_root
        setattr(self.main_gui, 'manga_generated_glossary_text', '')
        setattr(self.main_gui, 'manga_generated_glossary_entries', [])
        setattr(self.main_gui, 'manga_generated_glossary_path', '')
        if hasattr(self, 'translator') and self.translator:
            self.translator.manga_generated_glossary_text = ''
            self.translator.manga_generated_glossary_path = ''
            self.translator._manga_glossary_prompt_logged = False
        self._update_manga_glossary_status_label()
        self._save_rendering_settings()
        self._log("📚 Cleared loaded/generated manga glossary", "info")
        if deleted_glossary_dir:
            self._log(f"🗑️ Deleted manga glossary folder: {deleted_glossary_dir}", "info")

    def _edit_manga_glossary_prompt(self):
        """Open dialog to edit the manga glossary generation prompt."""
        dialog = QDialog(self.dialog)
        dialog.setWindowTitle("Manga Glossary Prompt")
        screen = QApplication.primaryScreen().geometry()
        dialog.setMinimumSize(int(screen.width() * 0.42), int(screen.height() * 0.55))

        layout = QVBoxLayout(dialog)
        instructions = QLabel(
            "Edit the prompt used after the OCR pass to generate one glossary from all selected manga pages."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        prompt_editor = QTextEdit()
        prompt_editor.setPlainText(getattr(self, 'manga_glossary_prompt', self._default_manga_glossary_prompt()))
        layout.addWidget(prompt_editor)

        button_layout = QHBoxLayout()

        def save_prompt():
            self.manga_glossary_prompt = prompt_editor.toPlainText().strip() or self._default_manga_glossary_prompt()
            self.main_gui.config['manga_glossary_prompt'] = self.manga_glossary_prompt
            self._save_rendering_settings()
            self._log("✅ Updated manga glossary prompt", "success")
            dialog.accept()

        def reset_prompt():
            prompt_editor.setPlainText(self._default_manga_glossary_prompt())

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(save_prompt)
        button_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(reset_prompt)
        button_layout.addWidget(reset_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        dialog.exec()

    def _generate_manga_glossary_button_clicked(self):
        """Run the OCR + glossary generation pass without translating pages."""
        if not getattr(self, 'selected_files', None):
            QMessageBox.warning(self.dialog, "No Images", "Add manga images before generating a glossary.")
            return
        if getattr(self, 'is_running', False):
            QMessageBox.information(self.dialog, "Already Running", "Wait for the current manga job to finish first.")
            return

        self.manga_glossary_prompt = getattr(self, 'manga_glossary_prompt', self._default_manga_glossary_prompt())
        self.main_gui.config['manga_glossary_prompt'] = self.manga_glossary_prompt
        self._manga_glossary_only_run = True
        self._log("📚 Starting glossary-only manga OCR pass", "info")
        self._start_translation()
    
    def _update_pool_tracker_label(self):
        """Update the pool tracker label with current preload pool status"""
        try:
            from manga_translator import MangaTranslator
            
            # Count bubble detectors in pool (total and available)
            detector_total = 0
            detector_in_use = 0
            try:
                with MangaTranslator._detector_pool_lock:
                    for rec in MangaTranslator._detector_pool.values():
                        spares = rec.get('spares', [])
                        checked_out = rec.get('checked_out', [])
                        detector_total += len(spares)
                        detector_in_use += len(checked_out)
            except Exception:
                pass
            
            # Count inpainters in pool (total and available)
            inpainter_total = 0
            inpainter_in_use = 0
            try:
                with MangaTranslator._inpaint_pool_lock:
                    for rec in MangaTranslator._inpaint_pool.values():
                        spares = rec.get('spares', [])
                        checked_out = rec.get('checked_out', [])
                        inpainter_total += len(spares)
                        inpainter_in_use += len(checked_out)
            except Exception:
                pass
            
            # Build pool text with available/total counts
            detector_avail = detector_total - detector_in_use
            inpainter_avail = inpainter_total - inpainter_in_use
            pool_text = f"• Pool: 🤖 {detector_avail}/{detector_total} | 🎨 {inpainter_avail}/{inpainter_total}"
            
            if hasattr(self, 'pool_tracker_label'):
                self.pool_tracker_label.setText(pool_text)
            self._sync_skip_inpainter_pool_lock(inpainter_in_use)
        except Exception as e:
            print(f"[POOL_TRACKER] Error updating: {e}")

    def _get_inpainter_pool_usage(self):
        """Return (total, checked_out) for the shared local inpainter pool."""
        total = 0
        in_use = 0
        try:
            from manga_translator import MangaTranslator
            with MangaTranslator._inpaint_pool_lock:
                for rec in MangaTranslator._inpaint_pool.values():
                    total += len(rec.get('spares', []) or [])
                    in_use += len(rec.get('checked_out', []) or [])
        except Exception:
            pass
        return total, in_use

    def _sync_skip_inpainter_pool_lock(self, inpainter_in_use: Optional[int] = None):
        """Disable Skip Inpainter while a pooled local inpainter is checked out."""
        checkbox = getattr(self, 'skip_inpainting_checkbox', None)
        if not checkbox:
            return
        try:
            if inpainter_in_use is None:
                _, inpainter_in_use = self._get_inpainter_pool_usage()
            locked = int(inpainter_in_use or 0) > 0
            checkbox.setEnabled(not locked)
            if locked:
                checkbox.setToolTip(
                    f"Locked while {int(inpainter_in_use)} local inpainter instance(s) are in use."
                )
            else:
                checkbox.setToolTip("Skip local inpainting and render translated text over the original image.")
        except Exception as e:
            print(f"[POOL_TRACKER] Failed to sync Skip Inpainter lock: {e}")
    
    def _refresh_context_settings_with_feedback(self):
        """Refresh context settings from main GUI and show feedback"""
        self._refresh_context_settings()
        self._update_pool_tracker_label()  # Also update pool tracker
        
        # Store original button state
        original_text = self.refresh_btn.text()
        original_style = self.refresh_btn.styleSheet()
        
        # Show loading state
        self.refresh_btn.setText("⏳ Refreshing...")
        self.refresh_btn.setStyleSheet("QPushButton { background-color: #ffc107; color: black; padding: 5px 15px; }")
        self.refresh_btn.setEnabled(False)
        
        # Process the refresh after a short delay to show loading state
        def do_refresh():
            try:
                self._refresh_context_settings()
                
                # Show success state briefly with what was refreshed
                success_text = "✅ Settings Refreshed!"
                self.refresh_btn.setText(success_text)
                self.refresh_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; }")
                
                # Reset to original state after 2 seconds
                QTimer.singleShot(2000, lambda: self._reset_refresh_button(original_text, original_style))
                
            except Exception as e:
                # Show error state
                self.refresh_btn.setText("❌ Error")
                self.refresh_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 5px 15px; }")
                
                # Log the error
                self._log(f"Error refreshing context settings: {str(e)}", "error")
                
                # Reset to original state after 3 seconds
                QTimer.singleShot(3000, lambda: self._reset_refresh_button(original_text, original_style))
        
        # Execute the refresh after a small delay to ensure loading state is visible
        QTimer.singleShot(100, do_refresh)
    
    def _reset_refresh_button(self, original_text, original_style):
        """Reset refresh button to original state"""
        if hasattr(self, 'refresh_btn'):
            self.refresh_btn.setText(original_text)
            self.refresh_btn.setStyleSheet(original_style)
            self.refresh_btn.setEnabled(True)
    
    def _refresh_context_settings(self):
        """Refresh context settings from main GUI - reads LIVE GUI state"""
        # CRITICAL FIX: Read from LIVE GUI state, not from disk
        # The main GUI should be the source of truth for current settings
        try:
            self._log("🔄 Refreshing from LIVE main GUI state...", "info")
            
            # Read directly from main GUI variables which are kept in sync with widgets
            # Variables like model_var, contextual_var are automatically updated when widgets change
            self._log("🔍 Reading from live GUI variables (auto-synced with widgets)...", "debug")
            
            # Get current live GUI state using the comprehensive method
            # This reads ALL current GUI widget values, not just cached variables
            live_api_key = ''
            try:
                if hasattr(self.main_gui, 'api_key_entry'):
                    if hasattr(self.main_gui.api_key_entry, 'text'):  # PySide6
                        live_api_key = self.main_gui.api_key_entry.text().strip()
                    elif hasattr(self.main_gui.api_key_entry, 'get'):  # Tkinter
                        live_api_key = self.main_gui.api_key_entry.get().strip()
            except Exception as e:
                self._log(f"⚠️ Could not get live API key: {e}", "debug")
            
            # Get ALL current GUI values using the comprehensive method
            env_vars = self.main_gui._get_environment_variables(
                epub_path='',  # Not needed for manga
                api_key=live_api_key or 'dummy'  # Use live API key or dummy
            )
            
            # Extract values from the environment variables
            live_model = env_vars.get('MODEL', 'Unknown')
            live_contextual = env_vars.get('CONTEXTUAL_MODE', '0') == '1'
            live_batch = env_vars.get('BATCH_TRANSLATION', '0') == '1'
            
            # API key presence already determined above
            live_api_key_present = bool(live_api_key)
            
            # Log what we found
            self._log(f"🔍 Live model: {live_model}", "debug")
            self._log(f"🔍 Live contextual: {live_contextual}", "debug")
            self._log(f"🔍 Live batch: {live_batch}", "debug")
            self._log(f"🔍 Live API key present: {live_api_key_present}", "debug")
            
            # Update manga integration displays with fresh values
            try:
                # Update model display
                if hasattr(self, 'model_label'):
                    self.model_label.setText(f"• Model: {live_model}")
                    self._log(f"✅ Updated model display to: {live_model}", "info")
                    
                # Update multi-key status display (this comes from config, not variables)
                if hasattr(self, 'multi_key_label'):
                    multi_key_enabled = self.main_gui.config.get('use_multi_api_keys', False)
                    multi_key_text = "• Multi-Key: ON" if multi_key_enabled else "• Multi-Key: OFF"
                    multi_key_color = "green" if multi_key_enabled else "gray"
                    self.multi_key_label.setText(multi_key_text)
                    self.multi_key_label.setStyleSheet(f"color: {multi_key_color};")
                    self._log(f"🔑 Multi-key status: {multi_key_text}", "debug")
                
                self._log("✅ Live GUI state refresh completed - no config sync needed!", "info")
                
            except Exception as e:
                self._log(f"⚠️ Error updating displays: {e}", "debug")
                
        except Exception as e:
            self._log(f"❌ Error in live GUI refresh: {e}", "error")
        
        # Keep the existing context settings logic
        if hasattr(self.main_gui, 'contextual_var'):
            contextual_enabled = self.main_gui.contextual_var
            if hasattr(self, 'contextual_status_label'):
                self.contextual_status_label.setText(f"• Contextual Translation: {'Enabled' if contextual_enabled else 'Disabled'}")
        
        if hasattr(self.main_gui, 'trans_history'):
            try:
                # Handle QLineEdit widget properly
                if hasattr(self.main_gui.trans_history, 'text'):
                    history_limit = self.main_gui.trans_history.text()
                else:
                    history_limit = str(self.main_gui.trans_history)
            except Exception:
                history_limit = "3"
            
            if hasattr(self, 'history_limit_label'):
                self.history_limit_label.setText(f"• Translation History Limit: {history_limit} exchanges")
        
        if True:
            rolling_enabled = True
            rolling_status = "Enabled (Rolling Window)"
            if hasattr(self, 'rolling_status_label'):
                self.rolling_status_label.setText(f"• Rolling History: {rolling_status}")
        
        # Get and update model from main GUI
        current_model = None
        model_changed = False
        
        if hasattr(self.main_gui, 'model_combo'):
            if hasattr(self.main_gui.model_combo, 'currentText'):  # PySide6
                current_model = self.main_gui.model_combo.currentText()
            elif hasattr(self.main_gui.model_combo, 'get'):  # Tkinter
                current_model = self.main_gui.model_combo.get()
        elif hasattr(self.main_gui, 'model_var'):
            current_model = self.main_gui.model_var if isinstance(self.main_gui.model_var, str) else str(self.main_gui.model_var)
        elif hasattr(self.main_gui, 'config'):
            current_model = self.main_gui.config.get('model', 'Unknown')
        
        # Update model display in the API Settings frame (skip if parent_frame doesn't exist)
        if hasattr(self, 'parent_frame') and hasattr(self.parent_frame, 'winfo_children'):
            try:
                for widget in self.parent_frame.winfo_children():
                    if isinstance(widget, tk.LabelFrame) and "Translation Settings" in widget.cget("text"):
                        for child in widget.winfo_children():
                            if isinstance(child, tk.Frame):
                                for subchild in child.winfo_children():
                                    if isinstance(subchild, tk.Label) and "Model:" in subchild.cget("text"):
                                        old_model_text = subchild.cget("text")
                                        old_model = old_model_text.split("Model: ")[-1] if "Model: " in old_model_text else None
                                        if old_model != current_model:
                                            model_changed = True
                                        subchild.config(text=f"Model: {current_model}")
                                        break
            except Exception:
                pass  # Silently skip if there's an issue with Tkinter widgets
        
        # If model changed, reset translator and client to force recreation
        if model_changed and current_model:
            if self.translator:
                self._log(f"Model changed to {current_model}. Translator will be recreated on next run.", "info")
                self.translator = None  # Force recreation on next translation
            
            # Also reset the client if it exists to ensure new model is used
            if hasattr(self.main_gui, 'client') and self.main_gui.client:
                if hasattr(self.main_gui.client, 'model') and self.main_gui.client.model != current_model:
                    self.main_gui.client = None  # Force recreation with new model
        
        # If translator exists, update its history manager settings
        if self.translator and hasattr(self.translator, 'history_manager'):
            try:
                # Update the history manager with current main GUI settings
                if hasattr(self.main_gui, 'contextual_var'):
                    self.translator.history_manager.contextual_enabled = self.main_gui.contextual_var
                
                if hasattr(self.main_gui, 'trans_history'):
                    try:
                        # Handle QLineEdit widget properly
                        if hasattr(self.main_gui.trans_history, 'text'):
                            history_value = self.main_gui.trans_history.text()
                        else:
                            history_value = str(self.main_gui.trans_history)
                        self.translator.history_manager.max_history = int(history_value)
                    except Exception:
                        self.translator.history_manager.max_history = 3
                
                self.translator.history_manager.rolling_enabled = True
                
                # Reset the history to apply new settings
                self.translator.history_manager.reset()
                
            except Exception as e:
                # Silently handle any translator update errors - visual feedback will show success
                pass
    
    def _browse_google_credentials_permanent(self):
        """Browse and set Google Cloud Vision credentials from the permanent button"""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            "Select Google Cloud Service Account JSON",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            # Save to config with both keys for compatibility
            self.main_gui.config['google_vision_credentials'] = file_path
            self.main_gui.config['google_cloud_credentials'] = file_path
            
            # Save configuration
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)

            
            from PySide6.QtWidgets import QMessageBox
            
            # Update button state immediately
            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(True)
            
            # Update credentials display
            if hasattr(self, 'creds_label'):
                self.creds_label.setText(os.path.basename(file_path))
                self.creds_label.setStyleSheet("color: green;")
            
            # Update the main status label and provider status
            self._update_main_status_label()
            self._check_provider_status()
            
            QMessageBox.information(self.dialog, "Success", "Google Cloud credentials set successfully!")
    
    def _update_status_display(self):
        """Update the status display after credentials change"""
        # This would update the status label if we had a reference to it
        # For now, we'll just ensure the button is enabled
        google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        has_vision = os.path.exists(google_creds_path) if google_creds_path else False
        
        if has_vision and hasattr(self, 'start_button'):
            self.start_button.setEnabled(True)
    
    def _get_available_fonts(self):
        """Get list of available fonts from system and custom directories"""
        fonts = ["Default"]  # Default option
        
        # Reset font mapping
        self.font_mapping = {}
        
        # Comprehensive map of Windows font filenames to proper display names
        font_name_map = {
            # === BASIC LATIN FONTS ===
            # Arial family
            'arial': 'Arial',
            'ariali': 'Arial Italic',
            'arialbd': 'Arial Bold',
            'arialbi': 'Arial Bold Italic',
            'ariblk': 'Arial Black',
            
            # Times New Roman
            'times': 'Times New Roman',
            'timesbd': 'Times New Roman Bold',
            'timesi': 'Times New Roman Italic',
            'timesbi': 'Times New Roman Bold Italic',
            
            # Calibri family
            'calibri': 'Calibri',
            'calibrib': 'Calibri Bold',
            'calibrii': 'Calibri Italic',
            'calibriz': 'Calibri Bold Italic',
            'calibril': 'Calibri Light',
            'calibrili': 'Calibri Light Italic',
            
            # Comic Sans family
            'comic': 'Comic Sans MS',
            'comici': 'Comic Sans MS Italic',
            'comicbd': 'Comic Sans MS Bold',
            'comicz': 'Comic Sans MS Bold Italic',
            
            # Segoe UI family
            'segoeui': 'Segoe UI',
            'segoeuib': 'Segoe UI Bold',
            'segoeuii': 'Segoe UI Italic',
            'segoeuiz': 'Segoe UI Bold Italic',
            'segoeuil': 'Segoe UI Light',
            'segoeuisl': 'Segoe UI Semilight',
            'seguisb': 'Segoe UI Semibold',
            'seguisbi': 'Segoe UI Semibold Italic',
            'seguisli': 'Segoe UI Semilight Italic',
            'seguili': 'Segoe UI Light Italic',
            'seguibl': 'Segoe UI Black',
            'seguibli': 'Segoe UI Black Italic',
            'seguihis': 'Segoe UI Historic',
            'seguiemj': 'Segoe UI Emoji',
            'seguisym': 'Segoe UI Symbol',
            
            # Courier
            'cour': 'Courier New',
            'courbd': 'Courier New Bold',
            'couri': 'Courier New Italic',
            'courbi': 'Courier New Bold Italic',
            
            # Verdana
            'verdana': 'Verdana',
            'verdanab': 'Verdana Bold',
            'verdanai': 'Verdana Italic',
            'verdanaz': 'Verdana Bold Italic',
            
            # Georgia
            'georgia': 'Georgia',
            'georgiab': 'Georgia Bold',
            'georgiai': 'Georgia Italic',
            'georgiaz': 'Georgia Bold Italic',
            
            # Tahoma
            'tahoma': 'Tahoma',
            'tahomabd': 'Tahoma Bold',
            
            # Trebuchet
            'trebuc': 'Trebuchet MS',
            'trebucbd': 'Trebuchet MS Bold',
            'trebucit': 'Trebuchet MS Italic',
            'trebucbi': 'Trebuchet MS Bold Italic',
            
            # Impact
            'impact': 'Impact',
            
            # Consolas
            'consola': 'Consolas',
            'consolab': 'Consolas Bold',
            'consolai': 'Consolas Italic',
            'consolaz': 'Consolas Bold Italic',
            
            # Sitka family (from your screenshot)
            'sitka': 'Sitka Small',
            'sitkab': 'Sitka Small Bold',
            'sitkai': 'Sitka Small Italic',
            'sitkaz': 'Sitka Small Bold Italic',
            'sitkavf': 'Sitka Text',
            'sitkavfb': 'Sitka Text Bold',
            'sitkavfi': 'Sitka Text Italic',
            'sitkavfz': 'Sitka Text Bold Italic',
            'sitkasubheading': 'Sitka Subheading',
            'sitkasubheadingb': 'Sitka Subheading Bold',
            'sitkasubheadingi': 'Sitka Subheading Italic',
            'sitkasubheadingz': 'Sitka Subheading Bold Italic',
            'sitkaheading': 'Sitka Heading',
            'sitkaheadingb': 'Sitka Heading Bold',
            'sitkaheadingi': 'Sitka Heading Italic',
            'sitkaheadingz': 'Sitka Heading Bold Italic',
            'sitkadisplay': 'Sitka Display',
            'sitkadisplayb': 'Sitka Display Bold',
            'sitkadisplayi': 'Sitka Display Italic',
            'sitkadisplayz': 'Sitka Display Bold Italic',
            'sitkabanner': 'Sitka Banner',
            'sitkabannerb': 'Sitka Banner Bold',
            'sitkabanneri': 'Sitka Banner Italic',
            'sitkabannerz': 'Sitka Banner Bold Italic',
            
            # Ink Free (from your screenshot)
            'inkfree': 'Ink Free',
            
            # Lucida family
            'l_10646': 'Lucida Sans Unicode',
            'lucon': 'Lucida Console',
            'ltype': 'Lucida Sans Typewriter',
            'ltypeb': 'Lucida Sans Typewriter Bold',
            'ltypei': 'Lucida Sans Typewriter Italic',
            'ltypebi': 'Lucida Sans Typewriter Bold Italic',

            # Palatino Linotype
            'pala': 'Palatino Linotype',
            'palab': 'Palatino Linotype Bold',
            'palabi': 'Palatino Linotype Bold Italic',
            'palai': 'Palatino Linotype Italic',

            # Noto fonts
            'notosansjp': 'Noto Sans JP',
            'notoserifjp': 'Noto Serif JP',

            # UD Digi Kyokasho (Japanese educational font)
            'uddigikyokashon-b': 'UD Digi Kyokasho NK-B',
            'uddigikyokashon-r': 'UD Digi Kyokasho NK-R',
            'uddigikyokashonk-b': 'UD Digi Kyokasho NK-B',
            'uddigikyokashonk-r': 'UD Digi Kyokasho NK-R',

            # Urdu Typesetting
            'urdtype': 'Urdu Typesetting',
            'urdtypeb': 'Urdu Typesetting Bold',

            # Segoe variants
            'segmdl2': 'Segoe MDL2 Assets',
            'segoeicons': 'Segoe Fluent Icons',
            'segoepr': 'Segoe Print',
            'segoeprb': 'Segoe Print Bold',
            'segoesc': 'Segoe Script',
            'segoescb': 'Segoe Script Bold',
            'seguivar': 'Segoe UI Variable',

            # Sans Serif Collection
            'sansserifcollection': 'Sans Serif Collection',

            # Additional common Windows 10/11 fonts
            'holomdl2': 'HoloLens MDL2 Assets',
            'gadugi': 'Gadugi',
            'gadugib': 'Gadugi Bold',

            # Cascadia Code (developer font)
            'cascadiacode': 'Cascadia Code',
            'cascadiacodepl': 'Cascadia Code PL',
            'cascadiamono': 'Cascadia Mono',
            'cascadiamonopl': 'Cascadia Mono PL',

            # More Segoe UI variants
            'seguibli': 'Segoe UI Black Italic',
            'segoeuiblack': 'Segoe UI Black',

            # Other fonts
            'aldhabi': 'Aldhabi',
            'andiso': 'Andalus',  # This is likely Andalus font
            'arabtype': 'Arabic Typesetting',
            'mstmc': 'Myanmar Text',  # Alternate file name
            'monbaiti': 'Mongolian Baiti',  # Shorter filename variant
            'leeluisl': 'Leelawadee UI Semilight',  # Missing variant
            'simsunextg': 'SimSun-ExtG',  # Extended SimSun variant
            'ebrima': 'Ebrima',
            'ebrimabd': 'Ebrima Bold',
            'gabriola': 'Gabriola',

            # Bahnschrift variants
            'bahnschrift': 'Bahnschrift',
            'bahnschriftlight': 'Bahnschrift Light',
            'bahnschriftsemibold': 'Bahnschrift SemiBold',
            'bahnschriftbold': 'Bahnschrift Bold',

            # Majalla (African language font)
            'majalla': 'Sakkal Majalla',
            'majallab': 'Sakkal Majalla Bold',

            # Additional fonts that might be missing
            'amiri': 'Amiri',
            'amiri-bold': 'Amiri Bold',
            'amiri-slanted': 'Amiri Slanted',
            'amiri-boldslanted': 'Amiri Bold Slanted',
            'aparaj': 'Aparajita',
            'aparajb': 'Aparajita Bold',
            'aparaji': 'Aparajita Italic',
            'aparajbi': 'Aparajita Bold Italic',
            'kokila': 'Kokila',
            'kokilab': 'Kokila Bold',
            'kokilai': 'Kokila Italic',
            'kokilabi': 'Kokila Bold Italic',
            'utsaah': 'Utsaah',
            'utsaahb': 'Utsaah Bold',
            'utsaahi': 'Utsaah Italic',
            'utsaahbi': 'Utsaah Bold Italic',
            'vani': 'Vani',
            'vanib': 'Vani Bold',
            
            # === JAPANESE FONTS ===
            'msgothic': 'MS Gothic',
            'mspgothic': 'MS PGothic',
            'msmincho': 'MS Mincho',
            'mspmincho': 'MS PMincho',
            'meiryo': 'Meiryo',
            'meiryob': 'Meiryo Bold',
            'yugothic': 'Yu Gothic',
            'yugothb': 'Yu Gothic Bold',
            'yugothl': 'Yu Gothic Light',
            'yugothm': 'Yu Gothic Medium',
            'yugothr': 'Yu Gothic Regular',
            'yumin': 'Yu Mincho',
            'yumindb': 'Yu Mincho Demibold',
            'yuminl': 'Yu Mincho Light',
            
            # === KOREAN FONTS ===
            'malgun': 'Malgun Gothic',
            'malgunbd': 'Malgun Gothic Bold',
            'malgunsl': 'Malgun Gothic Semilight',
            'gulim': 'Gulim',
            'gulimche': 'GulimChe',
            'dotum': 'Dotum',
            'dotumche': 'DotumChe',
            'batang': 'Batang',
            'batangche': 'BatangChe',
            'gungsuh': 'Gungsuh',
            'gungsuhche': 'GungsuhChe',
            
            # === CHINESE FONTS ===
            # Simplified Chinese
            'simsun': 'SimSun',
            'simsunb': 'SimSun Bold',
            'simsunextb': 'SimSun ExtB',
            'nsimsun': 'NSimSun',
            'simhei': 'SimHei',
            'simkai': 'KaiTi',
            'simfang': 'FangSong',
            'simli': 'LiSu',
            'simyou': 'YouYuan',
            'stcaiyun': 'STCaiyun',
            'stfangsong': 'STFangsong',
            'sthupo': 'STHupo',
            'stkaiti': 'STKaiti',
            'stliti': 'STLiti',
            'stsong': 'STSong',
            'stxihei': 'STXihei',
            'stxingkai': 'STXingkai',
            'stxinwei': 'STXinwei',
            'stzhongsong': 'STZhongsong',
            
            # Traditional Chinese  
            'msjh': 'Microsoft JhengHei',
            'msjhbd': 'Microsoft JhengHei Bold',
            'msjhl': 'Microsoft JhengHei Light',
            'mingliu': 'MingLiU',
            'pmingliu': 'PMingLiU',
            'mingliub': 'MingLiU Bold',
            'mingliuhk': 'MingLiU_HKSCS',
            'mingliuextb': 'MingLiU ExtB',
            'pmingliuextb': 'PMingLiU ExtB',
            'mingliuhkextb': 'MingLiU_HKSCS ExtB',
            'kaiu': 'DFKai-SB',
            
            # Microsoft YaHei
            'msyh': 'Microsoft YaHei',
            'msyhbd': 'Microsoft YaHei Bold',
            'msyhl': 'Microsoft YaHei Light',
            
            # === THAI FONTS ===
            'leelawui': 'Leelawadee UI',
            'leelauib': 'Leelawadee UI Bold',
            'leelauisl': 'Leelawadee UI Semilight',
            'leelawad': 'Leelawadee',
            'leelawdb': 'Leelawadee Bold',
            
            # === INDIC FONTS ===
            'mangal': 'Mangal',
            'vrinda': 'Vrinda',
            'raavi': 'Raavi',
            'shruti': 'Shruti',
            'tunga': 'Tunga',
            'gautami': 'Gautami',
            'kartika': 'Kartika',
            'latha': 'Latha',
            'kalinga': 'Kalinga',
            'vijaya': 'Vijaya',
            'nirmala': 'Nirmala UI',
            'nirmalab': 'Nirmala UI Bold',
            'nirmalas': 'Nirmala UI Semilight',
            
            # === ARABIC FONTS ===
            'arial': 'Arial',
            'trado': 'Traditional Arabic',
            'tradbdo': 'Traditional Arabic Bold',
            'simpo': 'Simplified Arabic',
            'simpbdo': 'Simplified Arabic Bold',
            'simpfxo': 'Simplified Arabic Fixed',
            
            # === OTHER ASIAN FONTS ===
            'javatext': 'Javanese Text',
            'himalaya': 'Microsoft Himalaya',
            'mongolianbaiti': 'Mongolian Baiti',
            'msuighur': 'Microsoft Uighur',
            'msuighub': 'Microsoft Uighur Bold',
            'msyi': 'Microsoft Yi Baiti',
            'taileb': 'Microsoft Tai Le Bold',
            'taile': 'Microsoft Tai Le',
            'ntailu': 'Microsoft New Tai Lue',
            'ntailub': 'Microsoft New Tai Lue Bold',
            'phagspa': 'Microsoft PhagsPa',
            'phagspab': 'Microsoft PhagsPa Bold',
            'mmrtext': 'Myanmar Text',
            'mmrtextb': 'Myanmar Text Bold',
            
            # === SYMBOL FONTS ===
            'symbol': 'Symbol',
            'webdings': 'Webdings',
            'wingding': 'Wingdings',
            'wingdng2': 'Wingdings 2',
            'wingdng3': 'Wingdings 3',
            'mtextra': 'MT Extra',
            'marlett': 'Marlett',
            
            # === OTHER FONTS ===
            'mvboli': 'MV Boli',
            'sylfaen': 'Sylfaen',
            'estrangelo': 'Estrangelo Edessa',
            'euphemia': 'Euphemia',
            'plantagenet': 'Plantagenet Cherokee',
            'micross': 'Microsoft Sans Serif',
            
            # Franklin Gothic
            'framd': 'Franklin Gothic Medium',
            'framdit': 'Franklin Gothic Medium Italic',
            'fradm': 'Franklin Gothic Demi',
            'fradmcn': 'Franklin Gothic Demi Cond',
            'fradmit': 'Franklin Gothic Demi Italic',
            'frahv': 'Franklin Gothic Heavy',
            'frahvit': 'Franklin Gothic Heavy Italic',
            'frabook': 'Franklin Gothic Book',
            'frabookit': 'Franklin Gothic Book Italic',
            
            # Cambria
            'cambria': 'Cambria',
            'cambriab': 'Cambria Bold',
            'cambriai': 'Cambria Italic',
            'cambriaz': 'Cambria Bold Italic',
            'cambria&cambria math': 'Cambria Math',
            
            # Candara
            'candara': 'Candara',
            'candarab': 'Candara Bold',
            'candarai': 'Candara Italic',
            'candaraz': 'Candara Bold Italic',
            'candaral': 'Candara Light',
            'candarali': 'Candara Light Italic',
            
            # Constantia
            'constan': 'Constantia',
            'constanb': 'Constantia Bold',
            'constani': 'Constantia Italic',
            'constanz': 'Constantia Bold Italic',
            
            # Corbel
            'corbel': 'Corbel',
            'corbelb': 'Corbel Bold',
            'corbeli': 'Corbel Italic',
            'corbelz': 'Corbel Bold Italic',
            'corbell': 'Corbel Light',
            'corbelli': 'Corbel Light Italic',
            
            # Bahnschrift
            'bahnschrift': 'Bahnschrift',
            
            # Garamond
            'gara': 'Garamond',
            'garabd': 'Garamond Bold',
            'garait': 'Garamond Italic',
            
            # Century Gothic
            'gothic': 'Century Gothic',
            'gothicb': 'Century Gothic Bold',
            'gothici': 'Century Gothic Italic',
            'gothicz': 'Century Gothic Bold Italic',
            
            # Bookman Old Style
            'bookos': 'Bookman Old Style',
            'bookosb': 'Bookman Old Style Bold',
            'bookosi': 'Bookman Old Style Italic',
            'bookosbi': 'Bookman Old Style Bold Italic',
        }
        
        # Dynamically discover all Windows fonts
        windows_fonts = []
        windows_font_dir = "C:/Windows/Fonts"
        
        if os.path.exists(windows_font_dir):
            for font_file in os.listdir(windows_font_dir):
                font_path = os.path.join(windows_font_dir, font_file)
                
                # Check if it's a font file
                if os.path.isfile(font_path) and font_file.lower().endswith(('.ttf', '.ttc', '.otf')):
                    # Get base name without extension
                    base_name = os.path.splitext(font_file)[0]
                    base_name_lower = base_name.lower()
                    
                    # Check if we have a proper name mapping
                    if base_name_lower in font_name_map:
                        display_name = font_name_map[base_name_lower]
                    else:
                        # Generic cleanup for unmapped fonts
                        display_name = base_name.replace('_', ' ').replace('-', ' ')
                        display_name = ' '.join(word.capitalize() for word in display_name.split())
                    
                    windows_fonts.append((display_name, font_path))
        
        # Sort alphabetically
        windows_fonts.sort(key=lambda x: x[0])
        
        # Add all discovered fonts to the list
        for font_name, font_path in windows_fonts:
            fonts.append(font_name)
            self.font_mapping[font_name] = font_path
        
        # Check for custom fonts directory (keep your existing code)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(script_dir, "fonts")
        
        if os.path.exists(fonts_dir):
            for root, dirs, files in os.walk(fonts_dir):
                for font_file in files:
                    if font_file.endswith(('.ttf', '.ttc', '.otf')):
                        font_path = os.path.join(root, font_file)
                        font_name = os.path.splitext(font_file)[0]
                        # Add category from folder
                        category = os.path.basename(root)
                        if category != "fonts":
                            font_name = f"{font_name} ({category})"
                        fonts.append(font_name)
                        self.font_mapping[font_name] = font_path
        
        # Load previously saved custom fonts (keep your existing code)
        if 'custom_fonts' in self.main_gui.config:
            for custom_font in self.main_gui.config['custom_fonts']:
                if os.path.exists(custom_font['path']):
                    # Check if this font is already in the list
                    if custom_font['name'] not in fonts:
                        fonts.append(custom_font['name'])
                        self.font_mapping[custom_font['name']] = custom_font['path']
        
        # Add custom fonts option at the end
        fonts.append("Browse Custom Font...")
        
        return fonts
    
    def _on_font_selected(self):
        """Handle font selection - updates font path AND font_style_value, save+apply called by widget"""
        if not hasattr(self, 'font_combo'):
            return
        selected = self.font_combo.currentText()
        
        # Update font_style_value to persist the selection
        self.font_style_value = selected
        
        if selected == "Default":
            self.selected_font_path = None
        elif selected == "Browse Custom Font...":
            # Open file dialog to select custom font using PySide6
            font_path, _ = QFileDialog.getOpenFileName(
                self.dialog if hasattr(self, 'dialog') else None,
                "Select Font File",
                "",
                "Font files (*.ttf *.ttc *.otf);;TrueType fonts (*.ttf);;TrueType collections (*.ttc);;OpenType fonts (*.otf);;All files (*.*)"
            )
            
            # Check if user selected a file (not cancelled)
            if font_path and font_path.strip():
                # Add to combo box
                font_name = os.path.basename(font_path)
                
                # Insert before "Browse Custom Font..." option
                if font_name not in [n for n in self.font_mapping.keys()]:
                    # Add to combo box (PySide6)
                    self.font_combo.insertItem(self.font_combo.count() - 1, font_name)
                    self.font_combo.setCurrentText(font_name)
                    
                    # Update font mapping
                    self.font_mapping[font_name] = font_path
                    self.selected_font_path = font_path
                    
                    # Save custom font to config
                    if 'custom_fonts' not in self.main_gui.config:
                        self.main_gui.config['custom_fonts'] = []
                    
                    custom_font_entry = {'name': font_name, 'path': font_path}
                    # Check if this exact entry already exists
                    font_exists = False
                    for existing_font in self.main_gui.config['custom_fonts']:
                        if existing_font['path'] == font_path:
                            font_exists = True
                            break
                    
                    if not font_exists:
                        self.main_gui.config['custom_fonts'].append(custom_font_entry)
                        # Save config immediately to persist custom fonts
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                else:
                    # Font already exists, just select it
                    self.font_combo.setCurrentText(font_name)
                    self.selected_font_path = self.font_mapping[font_name]
            else:
                # User cancelled, revert to previous selection
                if hasattr(self, 'previous_font_selection'):
                    self.font_combo.setCurrentText(self.previous_font_selection)
                else:
                    self.font_combo.setCurrentText("Default")
                return
        else:
            # Check if it's in the font mapping
            if selected in self.font_mapping:
                self.selected_font_path = self.font_mapping[selected]
            else:
                # This shouldn't happen, but just in case
                self.selected_font_path = None
        
        # Store current selection for next time
        self.previous_font_selection = selected
    
    def _update_opacity_label(self, value):
        """Update opacity percentage label and value variable"""
        self.bg_opacity_value = int(value)  # UPDATE THE VALUE VARIABLE!
        percentage = int((float(value) / 255) * 100)
        self.opacity_label.setText(f"{percentage}%")
    
    def _update_reduction_label(self, value):
        """Update size reduction percentage label and value variable"""
        self.bg_reduction_value = float(value)  # UPDATE THE VALUE VARIABLE!
        percentage = int(float(value) * 100)
        self.reduction_label.setText(f"{percentage}%")
        
    def _toggle_inpaint_quality_visibility(self):
        """Show/hide inpaint quality options based on skip_inpainting setting"""
        if hasattr(self, 'inpaint_quality_frame'):
            if self.skip_inpainting_value:
                # Hide quality options when inpainting is skipped
                self.inpaint_quality_frame.hide()
            else:
                # Show quality options when inpainting is enabled
                self.inpaint_quality_frame.show()

    def _toggle_inpaint_visibility(self):
        """Enable/disable inpainting options based on skip toggle (no animations to prevent dialog issues)"""
        try:
            if not getattr(self, '_initializing', False):
                _, inpainter_in_use = self._get_inpainter_pool_usage()
                if inpainter_in_use > 0:
                    previous_value = bool(getattr(self, 'skip_inpainting_value', False))
                    current_value = self.skip_inpainting_checkbox.isChecked()
                    if current_value != previous_value:
                        self.skip_inpainting_checkbox.blockSignals(True)
                        self.skip_inpainting_checkbox.setChecked(previous_value)
                        self.skip_inpainting_checkbox.blockSignals(False)
                        self._log(
                            f"⏳ Skip Inpainter is locked while {inpainter_in_use} local inpainter instance(s) are in use",
                            "warning"
                        )
                    self._sync_skip_inpainter_pool_lock(inpainter_in_use)
                    return

            # Update the value from the checkbox
            self.skip_inpainting_value = self.skip_inpainting_checkbox.isChecked()
            
            # Simple enable/disable logic - no animations, no show/hide, no empty dialogs!
            if self.skip_inpainting_value:
                print("🚫 Skip Inpainter: ENABLED - Inpainting will be skipped")
                self._clear_inpainter_preload_failure()
                self._set_translation_buttons_waiting(False)
                # Disable all inpainting options
                try:
                    # Disable parent frames
                    for widget in [self.inpaint_method_frame, self.cloud_inpaint_frame, 
                                   self.local_inpaint_frame, self.inpaint_separator]:
                        widget.setEnabled(False)
                    
                    # Apply disabled styling DIRECTLY to each child widget
                    disabled_button_style = "QPushButton { background-color: #2a2a2a; color: #555555; border: 1px solid #333333; padding: 5px 15px; }"
                    disabled_input_style = "QLineEdit { background-color: #252525; color: #555555; border: 1px solid #333333; }"
                    disabled_combo_style = "QComboBox { background-color: #252525; color: #555555; border: 1px solid #333333; }"
                    disabled_label_style = "QLabel { color: #555555; }"
                    disabled_radio_style = "QRadioButton { color: #555555; }"
                    
                    # Style all buttons in the frames
                    for frame in [self.inpaint_method_frame, self.cloud_inpaint_frame, self.local_inpaint_frame]:
                        for button in frame.findChildren(QPushButton):
                            button.setStyleSheet(disabled_button_style)
                        for lineedit in frame.findChildren(QLineEdit):
                            lineedit.setStyleSheet(disabled_input_style)
                        for combo in frame.findChildren(QComboBox):
                            combo.setStyleSheet(disabled_combo_style)
                        for label in frame.findChildren(QLabel):
                            label.setStyleSheet(disabled_label_style)
                        for radio in frame.findChildren(QRadioButton):
                            radio.setStyleSheet(disabled_radio_style)
                            
                except Exception as ve:
                    print(f"⚠️ Error disabling widgets: {ve}")
            else:
                print("✅ Skip Inpainter: DISABLED - Inpainting will be performed")
                # Enable all inpainting options
                try:
                    widgets_to_enable = [
                        self.inpaint_method_frame,
                        self.cloud_inpaint_frame,
                        self.local_inpaint_frame,
                        self.inpaint_separator
                    ]
                    
                    # Re-enable widgets
                    for widget in widgets_to_enable:
                        widget.setEnabled(True)
                    
                    # Clear disabled styling by setting empty stylesheet on parent frames
                    self.inpaint_method_frame.setStyleSheet("")
                    self.cloud_inpaint_frame.setStyleSheet("")
                    self.local_inpaint_frame.setStyleSheet("")
                    self.inpaint_separator.setStyleSheet("")
                    
                    # Restore styling for ALL child widgets (reverse of disable)
                    for frame in [self.inpaint_method_frame, self.cloud_inpaint_frame, self.local_inpaint_frame]:
                        # Restore labels to white text
                        for label in frame.findChildren(QLabel):
                            label.setStyleSheet("QLabel { color: white; }")
                        
                        # Restore radio buttons to white text  
                        for radio in frame.findChildren(QRadioButton):
                            radio.setStyleSheet("QRadioButton { color: white; }")
                        
                        # Restore combo boxes - ensure they're fully enabled with proper styling
                        for combo in frame.findChildren(QComboBox):
                            combo.setEnabled(True)  # Explicitly re-enable
                            # Apply an enabled style to override the disabled style completely
                            combo.setStyleSheet("QComboBox { background-color: palette(base); color: palette(text); border: 1px solid palette(mid); }")
                            combo.update()  # Force visual refresh
                        
                        # Restore line edits
                        for lineedit in frame.findChildren(QLineEdit):
                            if lineedit == getattr(self, 'local_model_entry', None):
                                lineedit.setStyleSheet("QLineEdit { background-color: #2b2b2b; color: #ffffff; }")
                            else:
                                lineedit.setStyleSheet("")
                        
                        # Restore buttons with their original colors
                        for button in frame.findChildren(QPushButton):
                            btn_text = button.text()
                            if button is getattr(self, 'custom_image_edit_keys_btn', None) or btn_text in ("Image Gen/Edit Keys", "Image Keys"):
                                self._style_preview_pool_button(button)
                            elif "Browse" in btn_text:
                                button.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 5px 15px; }")
                            elif "Load" in btn_text:
                                button.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; }")
                            elif "Download" in btn_text:
                                button.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
                            elif "Configure" in btn_text:
                                button.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
                            elif "Clear" in btn_text:
                                button.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
                            elif "Info" in btn_text:
                                button.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
                            else:
                                button.setStyleSheet("")
                    
                    # Make all frames visible first before calling the method change handler
                    self.cloud_inpaint_frame.show()
                    self.local_inpaint_frame.show()
                    
                    # Update method-specific frames visibility based on selected method
                    self._on_inpaint_method_change()
                except Exception as ve:
                    print(f"⚠️ Error enabling widgets: {ve}")
            
            # Don't save during initialization
            if not (hasattr(self, '_initializing') and self._initializing):
                print(f"💾 Saving config with skip_inpainting={self.skip_inpainting_value}")
                self._save_rendering_settings()
                print("✅ Config saved and environment variables reinitialized")

                # Verify the environment variable was set correctly
                import os
                env_value = os.environ.get('MANGA_SKIP_INPAINTING', 'NOT SET')
                print(f"🔍 Environment variable check: MANGA_SKIP_INPAINTING = {env_value}")
                expected_value = '1' if self.skip_inpainting_value else '0'
                if env_value == expected_value:
                    print(f"✅ Environment variable matches toggle state (expected={expected_value})")
                else:
                    print(f"⚠️ WARNING: Environment variable mismatch! Expected '{expected_value}' but got '{env_value}'")

                # If the user just turned Skip Inpainter OFF, kick off the
                # shared-inpainter preload in the background so the first
                # Translate/Clean action doesn't pay the cold-load cost.
                # We only preload for the 'local' method — cloud/hybrid either
                # don't need a local model or handle it lazily.
                try:
                    if not self.skip_inpainting_value and not getattr(self, '_shutting_down', False):
                        method = str(getattr(self, 'inpaint_method_value', '') or '').lower()
                        if method == 'local':
                            self._trigger_inpainter_preload_after_toggle()
                except Exception as _pre_err:
                    print(f"⚠️ Failed to schedule inpainter preload after toggle: {_pre_err}")
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in toggle function: {e}")
            print(traceback.format_exc())

    def _trigger_inpainter_preload_after_toggle(self):
        """Kick off `ImageRenderer._preload_shared_inpainter` in a daemon thread.

        Called from `_toggle_inpaint_visibility` when the Skip Inpainter
        checkbox is turned OFF so that the inpainter pool is warm by the time
        the user presses Translate / Clean / Translate All. Mirrors the
        background preload started in `MangaTranslationTab.__init__`.
        """
        try:
            # Avoid stacking concurrent preload threads
            existing = getattr(self, '_toggle_preload_thread', None)
            if existing is not None and existing.is_alive():
                print("🔁 Inpainter preload already in progress — skipping duplicate trigger")
                return
            self._log("🔁 Skip Inpainter disabled — preloading inpainter in background", "info")

            self._clear_inpainter_preload_failure()

            def _bg_preload():
                try:
                    if getattr(self, '_shutting_down', False):
                        return
                    # Re-check the flag inside the worker in case the user
                    # toggled Skip back ON while we were scheduling.
                    if bool(getattr(self, 'skip_inpainting_value', False)):
                        return
                    result = ImageRenderer._preload_shared_inpainter(self)
                    self._record_inpainter_preload_result(result, "skip-toggle")
                except Exception as e:
                    print(f"[TOGGLE_PRELOAD] Background inpainter preload failed: {e}")
                    self._record_inpainter_preload_result(False, "skip-toggle", str(e))

            self._toggle_preload_thread = threading.Thread(
                target=_bg_preload, name="InpainterPreload-AfterToggle", daemon=True
            )
            self._toggle_preload_thread.start()

            # Flip the workflow/start buttons to the "⏳ Waiting for model..."
            # state right away so the user can't click Translate before the
            # pool is warm. `_check_preload_status` polls every 500ms and
            # restores the buttons once the thread finishes.
            try:
                QTimer.singleShot(0, self._check_preload_status)
            except Exception:
                pass
        except Exception as e:
            print(f"[TOGGLE_PRELOAD] Failed to start preload thread: {e}")

    def _on_inpaint_method_change(self):
        """Show appropriate inpainting settings based on method"""
        # Don't change visibility AT ALL if skip inpainting is enabled
        # The frames stay visible but disabled
        if getattr(self, 'skip_inpainting_value', False):
            return
        
        # Determine current method from radio buttons
        if self.cloud_radio.isChecked():
            method = 'cloud'
        elif self.local_radio.isChecked():
            method = 'local'
        elif self.hybrid_radio.isChecked():
            method = 'hybrid'
        else:
            method = 'local'  # Default fallback
        
        # Update the stored value
        self.inpaint_method_value = method
        
        # Show/hide frames based on method
        # Cloud frame: visible for cloud and hybrid
        # Local frame: visible for local and hybrid
        if method == 'cloud':
            self.cloud_inpaint_frame.show()
            self.local_inpaint_frame.hide()
        elif method == 'local':
            self.cloud_inpaint_frame.hide()
            self.local_inpaint_frame.show()
        elif method == 'hybrid':
            self.cloud_inpaint_frame.show()
            self.local_inpaint_frame.show()
        
        # Force layout update
        if hasattr(self, 'parent_widget') and self.parent_widget:
            self.parent_widget.updateGeometry()
            self.parent_widget.update()
        
        # Don't save during initialization
        if not (hasattr(self, '_initializing') and self._initializing):
            self._save_rendering_settings()

    def _is_custom_image_edit_selected(self):
        return str(getattr(self, 'local_model_type_value', '') or '').lower() == 'custom-image-edit'

    def _is_old_custom_image_edit_default_prompt(self, prompt):
        text = " ".join(str(prompt or '').split()).lower()
        return (
            "replacing all foreign-language text" in text
            and "{target_lang}" in text
            and "if the image has no translatable text" in text
        )

    def _default_image_edit_endpoint_url(self):
        try:
            if getattr(self.main_gui, 'use_custom_openai_endpoint_var', False):
                url = str(getattr(self.main_gui, 'openai_base_url_var', '') or self.main_gui.config.get('openai_base_url', '') or '').strip()
                if url:
                    return url
            return 'https://api.openai.com/v1'
        except Exception:
            return 'https://api.openai.com/v1'

    def _sync_custom_image_edit_prompt(self, system_prompt=None, user_prompt=None, source='manga'):
        """Persist custom-image-edit prompts without touching Translator GUI prompts."""
        try:
            system_prompt = str(
                system_prompt if system_prompt is not None else getattr(self, 'custom_image_edit_system_prompt_value', '') or ''
            ).strip()
            if not system_prompt:
                system_prompt = self._default_custom_image_edit_system_prompt()
            elif self._is_old_custom_image_edit_default_prompt(system_prompt):
                system_prompt = self._default_custom_image_edit_system_prompt()
            user_prompt = str(
                user_prompt if user_prompt is not None else getattr(self, 'custom_image_edit_user_prompt_value', '') or ''
            ).strip()
            self.custom_image_edit_system_prompt_value = system_prompt
            self.custom_image_edit_user_prompt_value = user_prompt
            self.main_gui.custom_image_edit_system_prompt_var = system_prompt
            self.main_gui.custom_image_edit_user_prompt_var = user_prompt
            self.main_gui.config['custom_image_edit_system_prompt'] = system_prompt
            self.main_gui.config['custom_image_edit_user_prompt'] = user_prompt
            if hasattr(self.main_gui, 'save_config') and source != 'save':
                try:
                    self.main_gui.save_config(show_message=False)
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_custom_image_edit_controls(self, url=None, enabled=None, source='manga'):
        """Keep manga and Other Settings custom image edit controls in lockstep."""
        try:
            if url is None:
                url = getattr(self, 'custom_image_edit_endpoint_value', '')
            if enabled is None:
                enabled = getattr(self, 'use_custom_image_edit_endpoint_value', False)
            url = str(url or '').strip()
            enabled = bool(enabled)

            self.custom_image_edit_endpoint_value = url
            self.use_custom_image_edit_endpoint_value = enabled
            self.main_gui.custom_image_edit_endpoint_var = url
            self.main_gui.use_custom_image_edit_endpoint_var = enabled
            self.main_gui.config['custom_image_edit_endpoint'] = url
            self.main_gui.config['use_custom_image_edit_endpoint'] = enabled
            self.main_gui.config['manga_custom-image-edit_model_path'] = url
            self._set_custom_image_edit_env()

            for checkbox in (
                getattr(self, 'custom_image_edit_endpoint_checkbox', None),
                getattr(self.main_gui, 'use_custom_image_edit_endpoint_checkbox', None),
            ):
                if checkbox is not None:
                    try:
                        checkbox.blockSignals(True)
                        checkbox.setChecked(enabled)
                    finally:
                        try:
                            checkbox.blockSignals(False)
                        except Exception:
                            pass
                    try:
                        updater = getattr(checkbox, '_update_checkmark', None)
                        if callable(updater):
                            updater()
                        else:
                            checkbox.update()
                    except Exception:
                        pass

            for entry in (
                getattr(self.main_gui, 'custom_image_edit_endpoint_entry', None),
                getattr(self, 'local_model_entry', None) if self._is_custom_image_edit_selected() else None,
            ):
                if entry is not None and entry.text() != url:
                    try:
                        entry.blockSignals(True)
                        entry.setText(url)
                    finally:
                        try:
                            entry.blockSignals(False)
                        except Exception:
                            pass

            if self._is_custom_image_edit_selected():
                self.local_model_path_value = url
            if hasattr(self.main_gui, 'save_config') and source != 'save':
                try:
                    self.main_gui.save_config(show_message=False)
                except Exception:
                    pass
        except Exception:
            pass

    def _edit_custom_image_edit_prompt(self):
        """Open dialog to edit the prompt used by manga custom-image-edit inpainting."""
        dialog = QDialog(self.dialog)
        dialog.setWindowTitle("Custom Image Edit Prompt")
        screen = QApplication.primaryScreen().geometry()
        dialog.setMinimumSize(int(screen.width() * 0.42), int(screen.height() * 0.45))

        layout = QVBoxLayout(dialog)
        instructions = QLabel(
            "Edit prompts sent only to the custom-image-edit inpainter request. "
            "Use {target_lang} where the selected target language should be inserted."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        system_label = QLabel("System Prompt:")
        layout.addWidget(system_label)
        system_prompt_editor = QTextEdit()
        system_prompt_editor.setPlainText(
            getattr(self, 'custom_image_edit_system_prompt_value', '')
            or self.main_gui.config.get('custom_image_edit_system_prompt', '')
            or self.main_gui.config.get('custom_image_edit_prompt', '')
            or self._default_custom_image_edit_system_prompt()
        )
        layout.addWidget(system_prompt_editor)

        user_label = QLabel("User Prompt (optional):")
        layout.addWidget(user_label)
        user_prompt_editor = QTextEdit()
        user_prompt_editor.setPlaceholderText("Optional extra instruction for this image edit request")
        user_prompt_editor.setPlainText(
            getattr(self, 'custom_image_edit_user_prompt_value', '')
            or self.main_gui.config.get('custom_image_edit_user_prompt', '')
        )
        layout.addWidget(user_prompt_editor)

        button_layout = QHBoxLayout()

        def save_prompt():
            system_prompt = system_prompt_editor.toPlainText().strip() or self._default_custom_image_edit_system_prompt()
            user_prompt = user_prompt_editor.toPlainText().strip()
            self._sync_custom_image_edit_prompt(system_prompt=system_prompt, user_prompt=user_prompt, source='manga')
            self._save_rendering_settings()
            self._log("Updated custom image edit prompts", "success")
            dialog.accept()

        def reset_prompt():
            reply = QMessageBox.question(
                dialog,
                "Reset Custom Image Edit Prompts",
                "Reset the custom image edit system prompt to the default and clear the optional user prompt?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            system_prompt_editor.setPlainText(self._default_custom_image_edit_system_prompt())
            user_prompt_editor.clear()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(save_prompt)
        button_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(reset_prompt)
        button_layout.addWidget(reset_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        dialog.exec()

    def _on_custom_image_edit_endpoint_toggle(self, checked):
        self._sync_custom_image_edit_controls(
            url=getattr(self, 'custom_image_edit_endpoint_value', ''),
            enabled=bool(checked),
            source='manga'
        )
        self._apply_custom_image_edit_ui_state()

    def _on_custom_image_edit_full_page_output_changed(self, value):
        self.custom_image_edit_full_page_output_value = int(value)
        self.main_gui.custom_image_edit_full_page_output_var = int(value)
        self.main_gui.config['custom_image_edit_full_page_output'] = int(value)
        self._set_custom_image_edit_env()
        if not (hasattr(self, '_initializing') and self._initializing):
            self._save_rendering_settings()

    def _on_disable_inpaint_performance_mode_toggle(self, checked):
        self.disable_inpaint_performance_mode_value = bool(checked)
        self.main_gui.manga_disable_inpaint_performance_mode_var = bool(checked)
        self.main_gui.config['manga_disable_inpaint_performance_mode'] = bool(checked)
        if not (hasattr(self, '_initializing') and self._initializing):
            self._save_rendering_settings()

    def _on_local_model_entry_text_changed(self, text):
        if not self._is_custom_image_edit_selected():
            return
        self._sync_custom_image_edit_controls(
            url=text,
            enabled=getattr(self, 'use_custom_image_edit_endpoint_value', False),
            source='manga'
        )
        self._apply_custom_image_edit_ui_state(update_entry=False)

    def _apply_custom_image_edit_ui_state(self, update_entry=True):
        custom_selected = self._is_custom_image_edit_selected()
        if custom_selected:
            try:
                self._local_model_load_generation = int(getattr(self, '_local_model_load_generation', 0) or 0) + 1
                self._model_loading_in_progress = False
                self._clear_inpainter_preload_failure()
                self._set_translation_buttons_waiting(False)
            except Exception:
                pass
        if hasattr(self, 'custom_image_edit_controls_frame'):
            self.custom_image_edit_controls_frame.setVisible(custom_selected)
        if hasattr(self, 'custom_image_edit_endpoint_checkbox'):
            self.custom_image_edit_endpoint_checkbox.setVisible(custom_selected)
        if hasattr(self, 'custom_image_edit_prompt_btn'):
            self.custom_image_edit_prompt_btn.setVisible(custom_selected)
        if hasattr(self, 'custom_image_edit_keys_btn'):
            self._style_preview_pool_button(self.custom_image_edit_keys_btn)
            self.custom_image_edit_keys_btn.setVisible(custom_selected)
        if hasattr(self, 'custom_image_edit_output_frame'):
            self.custom_image_edit_output_frame.setVisible(custom_selected)
        if hasattr(self, 'batch_image_requests_frame'):
            self.batch_image_requests_frame.setVisible(custom_selected)
        if hasattr(self, 'custom_image_edit_area_spin'):
            self.custom_image_edit_area_spin.setVisible(custom_selected)
        if hasattr(self, 'custom_image_edit_area_label'):
            self.custom_image_edit_area_label.setVisible(custom_selected)
        if hasattr(self, 'disable_inpaint_performance_mode_frame'):
            self.disable_inpaint_performance_mode_frame.setVisible(not custom_selected)
        if hasattr(self, 'model_file_label'):
            self.model_file_label.setText("Image Edit URL:" if custom_selected else "Model File:")
        if hasattr(self, 'local_model_entry'):
            self.local_model_entry.setReadOnly(not custom_selected)
            self.local_model_entry.setPlaceholderText("blank = default image endpoint" if custom_selected else "")
            if custom_selected and update_entry:
                url = str(getattr(self, 'custom_image_edit_endpoint_value', '') or self.main_gui.config.get('custom_image_edit_endpoint', '') or '')
                if self.local_model_entry.text() != url:
                    try:
                        self.local_model_entry.blockSignals(True)
                        self.local_model_entry.setText(url)
                    finally:
                        self.local_model_entry.blockSignals(False)
                self.local_model_path_value = url
        if hasattr(self, 'browse_model_btn'):
            self.browse_model_btn.setText("Clear" if custom_selected else "Browse")
            self.browse_model_btn.setStyleSheet(
                "QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }"
                if custom_selected else
                "QPushButton { background-color: #007bff; color: white; padding: 5px 15px; }"
            )
        if hasattr(self, 'load_model_btn'):
            self.load_model_btn.setVisible(not custom_selected)
        local_download_btn = getattr(self, 'local_download_model_btn', getattr(self, 'download_model_btn', None))
        if local_download_btn:
            local_download_btn.setVisible(True)
            local_download_btn.setText("Test" if custom_selected else "Download Model")
        if custom_selected and hasattr(self, 'local_model_status_label'):
            enabled = bool(getattr(self, 'use_custom_image_edit_endpoint_value', False))
            url = str(getattr(self, 'custom_image_edit_endpoint_value', '') or '').strip()
            if not enabled:
                self.local_model_status_label.setText("Using default image edit endpoint")
                self.local_model_status_label.setStyleSheet("color: gray;")
            elif not url:
                self.local_model_status_label.setText("Using default image edit endpoint")
                self.local_model_status_label.setStyleSheet("color: #5dade2;")
            else:
                self.local_model_status_label.setText("Custom Image Edit Endpoint override configured")
                self.local_model_status_label.setStyleSheet("color: #5dade2;")

    def _on_local_model_change(self, new_model_type=None):
        """Handle model type change and auto-load if model exists"""
        # Get model type from combo box (PySide6)
        if new_model_type is None:
            model_type = self.local_model_combo.currentText()
        else:
            model_type = new_model_type

        # Don't start another real local-model load while one is already running.
        # custom-image-edit is endpoint-backed, so switching to it must still be
        # allowed so it can clear the "Waiting for model" state.
        if getattr(self, '_model_loading_in_progress', False) and model_type != 'custom-image-edit':
            return
        
        # Update stored value
        self.local_model_type_value = model_type
        
        # Update description
        model_desc = {
            'lama': 'LaMa (Best quality)',
            'aot': 'AOT GAN (Fast)',
            'aot_onnx': 'AOT ONNX (Optimized)',
            'mat': 'MAT (High-res)',
            'sd_local': 'Stable Diffusion (Anime)',
            'anime': 'Anime/Manga Inpainting',
            'anime_onnx': 'Anime ONNX (Fast/Optimized)',
            'lama_onnx': 'LaMa ONNX (Optimized)',
            'custom-image-edit': 'Custom OpenAI-compatible image edit endpoint (.gguf)',
        }
        self.model_desc_label.setText(model_desc.get(model_type, ''))

        if model_type == 'custom-image-edit':
            self._local_model_load_generation = int(getattr(self, '_local_model_load_generation', 0) or 0) + 1
            self._model_loading_in_progress = False
            self._clear_inpainter_preload_failure()
            self._set_translation_buttons_waiting(False)
            self.custom_image_edit_endpoint_value = self.main_gui.config.get('custom_image_edit_endpoint', '')
            self.use_custom_image_edit_endpoint_value = self.main_gui.config.get(
                'use_custom_image_edit_endpoint',
                False
            )
            self._sync_custom_image_edit_controls(
                url=self.custom_image_edit_endpoint_value,
                enabled=self.use_custom_image_edit_endpoint_value,
                source='manga'
            )
            self._apply_custom_image_edit_ui_state()
            self._save_rendering_settings()
            return
        
        # Check for saved path for this model type
        saved_path = self.main_gui.config.get(f'manga_{model_type}_model_path', '')
        
        if saved_path and os.path.exists(saved_path):
            # Update the path display
            self.local_model_entry.setText(saved_path)
            self.local_model_path_value = saved_path
            self.local_model_status_label.setText("⏳ Loading saved model...")
            self.local_model_status_label.setStyleSheet("color: orange;")
            
            try:
                if self._is_local_inpainting_enabled():
                    self._set_translation_buttons_waiting(True)
            except Exception:
                pass

            # Auto-load the model after a short delay using QTimer
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._try_load_model(model_type, saved_path))
        else:
            # Clear the path display
            self.local_model_entry.setText("")
            self.local_model_path_value = ""
            self.local_model_status_label.setText("No model loaded")
            self.local_model_status_label.setStyleSheet("color: gray;")

        self._apply_custom_image_edit_ui_state()
        
        self._save_rendering_settings()

    def _browse_local_model(self):
        """Browse for local inpainting model and auto-load"""
        from PySide6.QtWidgets import QFileDialog
        from PySide6.QtCore import QTimer
        
        model_type = self.local_model_type_value

        if model_type == 'custom-image-edit':
            self._sync_custom_image_edit_controls(url="", enabled=getattr(self, 'use_custom_image_edit_endpoint_value', False), source='manga')
            self._apply_custom_image_edit_ui_state()
            return

        if model_type == 'sd_local':
            filter_str = "Model files (*.safetensors *.pt *.pth *.ckpt *.onnx);;SafeTensors (*.safetensors);;Checkpoint files (*.ckpt);;PyTorch models (*.pt *.pth);;ONNX models (*.onnx);;All files (*.*)"
        else:
            filter_str = "Model files (*.safetensors *.pt *.pth *.ckpt *.onnx);;SafeTensors (*.safetensors);;Checkpoint files (*.ckpt);;PyTorch models (*.pt *.pth);;ONNX models (*.onnx);;All files (*.*)"
        
        path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            f"Select {model_type.upper()} Model",
            "",
            filter_str
        )
        
        if path:
            self.local_model_entry.setText(path)
            self.local_model_path_value = path
            # Save to config
            self.main_gui.config[f'manga_{model_type}_model_path'] = path
            self._save_rendering_settings()
            
            # Update status first
            self._update_local_model_status()
            
            # Auto-load the selected model using QTimer
            QTimer.singleShot(100, lambda: self._try_load_model(model_type, path))

    def _click_load_local_model(self):
        """Manually trigger loading of the selected local inpainting model"""
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import QTimer
        
        # Don't allow if already loading
        if getattr(self, '_model_loading_in_progress', False):
            QMessageBox.information(self.dialog, "Loading", "A model is already being loaded. Please wait...")
            return
        
        try:
            model_type = self.local_model_type_value if hasattr(self, 'local_model_type_value') else None
            path = self.local_model_path_value if hasattr(self, 'local_model_path_value') else ''
            if not model_type:
                QMessageBox.information(self.dialog, "Load Model", "Please select a model type first.")
                return
            if model_type == 'custom-image-edit':
                self._test_custom_image_edit_endpoint()
                return
            if not path:
                QMessageBox.information(self.dialog, "Load Model", "Please select a model file first using the Browse button.")
                return
            # Defer to keep UI responsive using QTimer
            QTimer.singleShot(50, lambda: self._try_load_model(model_type, path))
        except Exception:
            pass

    def _try_load_model(self, method: str, model_path: str, show_completion_dialog: bool = False):
        """Try to load a model in background thread with proper GUI updates."""
        import threading
        from PySide6.QtWidgets import QApplication
        if str(method or '').lower() == 'custom-image-edit':
            endpoint = str(model_path or getattr(self, 'custom_image_edit_endpoint_value', '') or self.main_gui.config.get('custom_image_edit_endpoint', '') or '').strip()
            self._sync_custom_image_edit_controls(endpoint, getattr(self, 'use_custom_image_edit_endpoint_value', False), source='manga')
            try:
                from local_inpainter import LocalInpainter
                inp = LocalInpainter(enable_worker_process=False)
                ok = inp.load_model('custom-image-edit', endpoint, force_reload=True)
                self._handle_model_load_complete(method, bool(ok), None if ok else "Custom Image Edit Endpoint is not configured", show_completion_dialog, endpoint)
                return bool(ok)
            except Exception as e:
                self._handle_model_load_complete(method, False, str(e), show_completion_dialog, endpoint)
                return False
        
        # Check if already loading
        if getattr(self, '_model_loading_in_progress', False):
            self.main_gui.append_log("⚠️ Model load already in progress, skipping...")
            return False
        
        # Set loading flag
        self._model_loading_in_progress = True
        self._local_model_load_generation = int(getattr(self, '_local_model_load_generation', 0) or 0) + 1
        load_generation = self._local_model_load_generation
        try:
            self._clear_inpainter_preload_failure()
            if self._is_local_inpainting_enabled():
                self._set_translation_buttons_waiting(True)
        except Exception:
            pass
        
        # Disable load button while loading
        if hasattr(self, 'load_local_model_button'):
            self.load_local_model_button.setEnabled(False)
        
        # Show loading status immediately
        self.local_model_status_label.setText("⏳ Loading model...")
        self.local_model_status_label.setStyleSheet("color: orange;")
        self.main_gui.append_log(f"⏳ Loading {method.upper()} model...")
        
        # Track result
        load_result = {'success': False, 'error_msg': None, 'done': False, 'model_path': None}
        
        def load_in_background():
            """Background thread for model loading - preload into pool"""
            try:
                # Normalize model path
                import os
                normalized_path = model_path
                if model_path:
                    try:
                        normalized_path = os.path.abspath(os.path.normpath(model_path))
                    except Exception:
                        pass
                # Guard: ignore JSON paths (credentials, etc.)
                try:
                    if isinstance(normalized_path, str) and normalized_path.lower().endswith('.json'):
                        normalized_path = ''
                except Exception:
                    normalized_path = ''
                
                # Create a minimal translator to access the preload system
                from manga_translator import MangaTranslator
                
                # IMPORTANT: Do NOT touch OCR/google credentials when loading inpainting models
                ocr_config = {'provider': 'local'}
                
                # Get unified client
                try:
                    from unified_api_client import UnifiedClient
                    api_key = self.main_gui.config.get('api_key', '') or 'dummy'
                    model_name = self.main_gui.config.get('model', 'gpt-4o-mini')
                    uc = UnifiedClient(model=model_name, api_key=api_key)
                except Exception:
                    uc = None
                
                # Logging callback - use sys.stdout.write to avoid recursion
                import sys
                def log_cb(msg, level='info'):
                    sys.stdout.write(f"[INPAINT_LOAD] {level.upper()}: {msg}\n")
                    sys.stdout.flush()
                
                # Create translator instance
                mt = MangaTranslator(
                    ocr_config=ocr_config,
                    unified_client=uc,
                    main_gui=self.main_gui,
                    log_callback=log_cb,
                    skip_inpainter_init=True
                )

                # If path missing, try to download a valid model before preloading
                if not normalized_path or not os.path.exists(normalized_path):
                    try:
                        from local_inpainter import LocalInpainter
                        tmp_inp = LocalInpainter()
                        dl_path = tmp_inp.download_jit_model(method)
                        if dl_path and os.path.exists(dl_path):
                            normalized_path = os.path.abspath(os.path.normpath(dl_path))
                    except Exception:
                        pass

                if normalized_path:
                    load_result['model_path'] = normalized_path
                    try:
                        self.main_gui.config[f'manga_{method}_model_path'] = normalized_path
                    except Exception:
                        pass
                
                # Preload 1 inpainter into the pool (concurrent for faster loading)
                created = mt.preload_local_inpainters_concurrent(method, normalized_path, 1)
                
                if created > 0:
                    print(f"[INPAINT_LOAD] Successfully preloaded {method} inpainter")
                    load_result['success'] = True
                else:
                    print(f"[INPAINT_LOAD] Failed to preload {method} inpainter")
                    load_result['success'] = False
                    load_result['error_msg'] = "Preload returned 0 instances"
                    
            except Exception as e:
                print(f"[INPAINT_LOAD] Error: {e}")
                import traceback
                print(traceback.format_exc())
                load_result['error_msg'] = str(e)
                load_result['success'] = False
            finally:
                load_result['done'] = True
        
        # Start loading in background thread
        thread = threading.Thread(target=load_in_background, daemon=True)
        thread.start()
        
        # Poll for completion in a non-blocking way using QTimer on main thread
        from PySide6.QtCore import QTimer
        check_timer = QTimer()
        
        def check_load_complete():
            if load_result['done']:
                check_timer.stop()
                if load_generation != int(getattr(self, '_local_model_load_generation', 0) or 0):
                    return
                if method != getattr(self, 'local_model_type_value', method):
                    return
                # Call completion handler on main thread
                self._handle_model_load_complete(method, load_result['success'], load_result['error_msg'], show_completion_dialog, load_result.get('model_path'))
            else:
                # Process events while waiting
                QApplication.processEvents()
        
        check_timer.timeout.connect(check_load_complete)
        check_timer.start(100)  # Check every 100ms
        
        # Return True to indicate load was initiated (not necessarily completed)
        return True
    
    def _handle_model_load_complete(self, method: str, success: bool, error_msg: str = None, show_dialog: bool = False, model_path: str = None):
        """Handle model load completion on main thread"""
        from PySide6.QtWidgets import QMessageBox
        
        # Clear loading flag
        self._model_loading_in_progress = False
        try:
            if success:
                self._set_translation_buttons_waiting(False)
        except Exception:
            pass
        
        # Re-enable load button
        if hasattr(self, 'load_local_model_button'):
            self.load_local_model_button.setEnabled(True)
        
        print(f"DEBUG: Updating UI after load, success={success}")
        
        if success:
            if model_path:
                try:
                    import os
                    if str(method or '').lower() == 'custom-image-edit':
                        resolved_path = str(model_path or '').strip()
                    else:
                        resolved_path = os.path.abspath(os.path.normpath(model_path))
                    self.local_model_path_value = resolved_path
                    self.main_gui.config[f'manga_{method}_model_path'] = resolved_path
                    if str(method or '').lower() == 'custom-image-edit':
                        self._sync_custom_image_edit_controls(resolved_path, True, source='manga')
                    if method == getattr(self, 'local_model_type_value', method) and hasattr(self, 'local_model_entry'):
                        self.local_model_entry.setText(resolved_path)
                    self._save_rendering_settings()
                except Exception:
                    pass
            self._clear_inpainter_preload_failure()
            self.local_model_status_label.setText(f"✅ {method.upper()} model ready")
            self.local_model_status_label.setStyleSheet("color: green;")
            self.main_gui.append_log(f"✅ {method.upper()} model loaded successfully!")
            
            # Clear translator cache
            if hasattr(self, 'translator') and self.translator:
                for attr in ('local_inpainter', '_last_local_method', '_last_local_model_path'):
                    if hasattr(self.translator, attr):
                        try:
                            delattr(self.translator, attr)
                        except Exception:
                            pass
            
            # Show success dialog if requested
            if show_dialog:
                QMessageBox.information(self.dialog, "Success", f"{method.upper()} model loaded successfully!")
            try:
                QTimer.singleShot(0, self._check_preload_status)
            except Exception:
                pass
        else:
            self.local_model_status_label.setText("⚠️ Model file found but failed to load")
            self.local_model_status_label.setStyleSheet("color: orange;")
            if self._is_local_inpainting_enabled():
                self._record_inpainter_preload_result(False, "manual-load", error_msg or "Manual model load returned 0 instances")
            if error_msg:
                self.main_gui.append_log(f"❌ Error loading model: {error_msg}")
            else:
                self.main_gui.append_log("⚠️ Model file found but failed to load")
            
            # Show error dialog if requested
            if show_dialog:
                msg = f"Failed to load {method.upper()} model"
                if error_msg:
                    msg += f":\n{error_msg}"
                QMessageBox.warning(self.dialog, "Load Failed", msg)
            try:
                QTimer.singleShot(0, self._check_preload_status)
            except Exception:
                pass
        
        print(f"DEBUG: UI update completed")
        
    def _open_output_folder(self):
        """Open the output folder in file explorer, respecting output_directory override"""
        import subprocess
        import platform
        from PySide6.QtWidgets import QMessageBox
        
        # Get output directory from config or environment (respecting other_settings.py override)
        output_dir = self.main_gui.config.get('output_directory', os.environ.get('OUTPUT_DIRECTORY', ''))
        
        # If override is set, use it directly
        if output_dir:
            # Ensure it exists
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(self.dialog, "Error", f"Failed to create output folder:\n{str(e)}")
                    return
        else:
            # No override - check if we have selected files and use their parent directory
            if hasattr(self, 'selected_files') and self.selected_files:
                # Use parent directory of first selected file
                first_file = self.selected_files[0]
                parent_dir = os.path.dirname(first_file)
                source_name_no_ext = os.path.splitext(os.path.basename(first_file))[0]
                
                # Check for the specific translated folder for the current file
                translated_folder = os.path.join(parent_dir, f"{source_name_no_ext}_translated")
                
                if os.path.exists(translated_folder) and os.path.isdir(translated_folder):
                    # Open the specific translated folder for this file
                    output_dir = translated_folder
                else:
                    # Look for any *_translated folders in parent directory
                    translated_folders = []
                    if os.path.exists(parent_dir):
                        for item in os.listdir(parent_dir):
                            item_path = os.path.join(parent_dir, item)
                            if os.path.isdir(item_path) and item.endswith('_translated'):
                                translated_folders.append(item_path)
                    
                    # If we found translated folders, open the first one
                    if translated_folders:
                        output_dir = translated_folders[0]
                    else:
                        # Fall back to default output folder in current working directory
                        output_dir = os.path.join(_get_app_dir(), 'output')
                        if not os.path.exists(output_dir):
                            try:
                                os.makedirs(output_dir, exist_ok=True)
                            except Exception as e:
                                QMessageBox.warning(self.dialog, "Error", f"Failed to create output folder:\n{str(e)}")
                                return
            else:
                # No selected files - use default output folder
                output_dir = os.path.join(_get_app_dir(), 'output')
                if not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                    except Exception as e:
                        QMessageBox.warning(self.dialog, "Error", f"Failed to create output folder:\n{str(e)}")
                        return
        
        # Open folder in file explorer
        try:
            if platform.system() == 'Windows':
                os.startfile(output_dir)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', output_dir])
            else:  # Linux and others
                subprocess.run(['xdg-open', output_dir])
        except Exception as e:
            QMessageBox.warning(self.dialog, "Error", f"Failed to open folder:\n{str(e)}")
    
    def _update_local_model_status(self):
        """Update local model status display"""
        if self._is_custom_image_edit_selected():
            self._apply_custom_image_edit_ui_state()
            return
        path = self.local_model_path_value if hasattr(self, 'local_model_path_value') else ''
        
        if not path:
            self.local_model_status_label.setText("⚠️ No model selected")
            self.local_model_status_label.setStyleSheet("color: orange;")
            return
        
        if not os.path.exists(path):
            self.local_model_status_label.setText("❌ Model file not found")
            self.local_model_status_label.setStyleSheet("color: red;")
            return
        
        # Check for ONNX cache
        if path.endswith(('.pt', '.pth', '.safetensors')):
            onnx_dir = os.path.join(os.path.dirname(path), 'models')
            if os.path.exists(onnx_dir):
                # Check if ONNX file exists for this model
                model_hash = hashlib.sha256(path.encode()).hexdigest()[:8]
                onnx_files = [f for f in os.listdir(onnx_dir) if model_hash in f]
                if onnx_files:
                    self.local_model_status_label.setText("✅ Model ready (ONNX cached)")
                    self.local_model_status_label.setStyleSheet("color: green;")
                else:
                    self.local_model_status_label.setText("ℹ️ Will convert to ONNX on first use")
                    self.local_model_status_label.setStyleSheet("color: #5dade2;")  # Light cyan for better contrast
            else:
                self.local_model_status_label.setText("ℹ️ Will convert to ONNX on first use")
                self.local_model_status_label.setStyleSheet("color: #5dade2;")  # Light cyan for better contrast
        else:
            self.local_model_status_label.setText("✅ ONNX model ready")
            self.local_model_status_label.setStyleSheet("color: green;")

    def _test_custom_image_edit_endpoint(self):
        """Lightweight connectivity test for the custom image edit endpoint."""
        from PySide6.QtWidgets import QMessageBox
        try:
            import requests
            url = str(getattr(self, 'custom_image_edit_endpoint_value', '') or '').strip()
            enabled = bool(getattr(self, 'use_custom_image_edit_endpoint_value', False))
            if not enabled or not url:
                self.local_model_status_label.setText("Using current image provider/model")
                self.local_model_status_label.setStyleSheet("color: #5dade2;")
                QMessageBox.information(
                    self.dialog,
                    "Custom Image Edit Endpoint",
                    "Blank URL uses the current main image provider/model. There is no separate custom endpoint URL to test."
                )
                return
            if not url.startswith(('http://', 'https://')):
                lower = url.lower()
                url = ('http://' if lower.startswith(('localhost', '127.', '0.0.0.0', '[')) else 'https://') + url
            url = url.rstrip('/')
            if str(getattr(self, 'custom_image_edit_endpoint_value', '') or '').strip():
                self._sync_custom_image_edit_controls(url=url, enabled=True, source='manga')

            self.local_model_status_label.setText("Testing image edit endpoint...")
            self.local_model_status_label.setStyleSheet("color: orange;")
            headers = {
                'Authorization': f"Bearer {os.environ.get('CUSTOM_IMAGE_EDIT_API_KEY') or os.environ.get('OPENAI_API_KEY') or self.main_gui.config.get('api_key', '') or 'sk-local'}"
            }
            resp = requests.get(f"{url}/models", headers=headers, timeout=10)
            if resp.status_code in (200, 201):
                self.local_model_status_label.setText("Image edit endpoint reachable")
                self.local_model_status_label.setStyleSheet("color: green;")
                QMessageBox.information(self.dialog, "Image Edit Endpoint", f"Endpoint is reachable:\n{url}")
            elif resp.status_code in (401, 403):
                self.local_model_status_label.setText("Endpoint reached, but authentication failed")
                self.local_model_status_label.setStyleSheet("color: orange;")
                QMessageBox.warning(self.dialog, "Custom Image Edit Endpoint", f"Endpoint responded with authentication error ({resp.status_code}).")
            else:
                self.local_model_status_label.setText(f"Endpoint responded: HTTP {resp.status_code}")
                self.local_model_status_label.setStyleSheet("color: orange;")
                QMessageBox.information(self.dialog, "Custom Image Edit Endpoint", f"Endpoint responded with HTTP {resp.status_code}.")
        except Exception as e:
            self.local_model_status_label.setText("Custom Image Edit Endpoint test failed")
            self.local_model_status_label.setStyleSheet("color: red;")
            QMessageBox.warning(self.dialog, "Custom Image Edit Endpoint", f"Test failed:\n{e}")

    def _download_model(self):
        """Actually download the model for the selected type.
        Checks cache first and auto-loads if present."""
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import QTimer
        
        model_type = self.local_model_type_value
        if model_type == 'custom-image-edit':
            self._test_custom_image_edit_endpoint()
            return
        
        # Guard: if config has a .json path (e.g., credentials), clear it before downloading
        try:
            bad_path = self.main_gui.config.get(f'manga_{model_type}_model_path', '')
            if isinstance(bad_path, str) and bad_path.lower().endswith('.json'):
                self.main_gui.config[f'manga_{model_type}_model_path'] = ''
                try:
                    # Also clear nested inpainting setting if present
                    ms = self.main_gui.config.get('manga_settings', {}) or {}
                    inpaint = ms.get('inpainting', {}) or {}
                    inpaint[f'{model_type}_model_path'] = ''
                    ms['inpainting'] = inpaint
                    self.main_gui.config['manga_settings'] = ms
                except Exception:
                    pass
                # Clear UI state to avoid loading the wrong path
                self.local_model_entry.setText("")
                self.local_model_path_value = ""
                if hasattr(self, 'local_model_status_label'):
                    self.local_model_status_label.setText("No model loaded")
                    self.local_model_status_label.setStyleSheet("color: gray;")
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
        except Exception:
            pass
        
        try:
            from local_inpainter import LocalInpainter
            # Temporarily create instance just to check cache (doesn't load model)
            temp_inp = LocalInpainter()
            cached_path = temp_inp.get_cached_model_path(model_type)

            # Fast path: already cached — stay synchronous, it's instant.
            if cached_path and os.path.exists(cached_path):
                self.local_model_entry.setText(cached_path)
                self.local_model_path_value = cached_path
                self.main_gui.config[f'manga_{model_type}_model_path'] = cached_path
                self._save_rendering_settings()
                self._try_load_model(model_type, cached_path, show_completion_dialog=True)
                return

            # Slow path: needs a network fetch. Run the actual download on a
            # background thread and poll with a QTimer so the GUI stays responsive.
            model_name = model_type.upper()
            if model_type in ['anime_onnx', 'aot_onnx', 'lama_onnx']:
                model_name = f"Optimized {model_type.split('_')[0].upper()}"
            elif model_type == 'custom-image-edit':
                model_name = "Custom Image Edit"
            elif model_type == 'sd_local':
                model_name = "Stable Diffusion"

            if hasattr(self, 'local_model_status_label'):
                self.local_model_status_label.setText(f"📥 Downloading {model_name} model...")
                self.local_model_status_label.setStyleSheet("color: #4a9eff;")
                # Force an immediate repaint so the user sees the initial status
                # before the worker thread spins up.
                try:
                    self.local_model_status_label.repaint()
                    from PySide6.QtWidgets import QApplication as _QApp
                    _QApp.processEvents()
                except Exception:
                    pass

            # Guard: don't kick off a second concurrent download if one is
            # already running for this dialog instance.
            if getattr(self, '_model_download_in_progress', False):
                QMessageBox.information(
                    self.dialog, "Download in progress",
                    "Another model download is already running. Please wait."
                )
                return
            self._model_download_in_progress = True

            import threading
            import time as _dl_time
            from queue import Queue as _DLQueue, Empty as _DLEmpty
            from PySide6.QtCore import QTimer

            # Wire up a progress queue so LocalInpainter.download_jit_model can
            # push status strings back to us while the download runs on the
            # worker thread. (It uses the ('model_file_status', text) tuple format.)
            temp_inp.progress_queue = _DLQueue()

            # Snapshot the cache dir BEFORE the download starts so our watcher
            # can find the "growing file" without false positives from older
            # cached models.
            try:
                from local_inpainter import CACHE_DIR as _CACHE_DIR
            except Exception:
                _CACHE_DIR = os.path.expanduser('~/.cache/inpainting')

            def _snapshot_sizes(root):
                seen = {}
                try:
                    for dp, _dn, fn in os.walk(root):
                        for f in fn:
                            p = os.path.join(dp, f)
                            try:
                                seen[p] = os.path.getsize(p)
                            except OSError:
                                pass
                except Exception:
                    pass
                return seen

            pre_sizes = _snapshot_sizes(_CACHE_DIR)

            dl_result = {
                'path': None,
                'error': None,
                'done': False,
                'last_status': f"📥 Downloading {model_name} model...",
                'started_at': _dl_time.time(),
                'live_bytes': 0,
                'live_file': None,
            }

            def _dl_worker():
                try:
                    dl_result['path'] = temp_inp.download_jit_model(model_type)
                except Exception as worker_err:
                    dl_result['error'] = str(worker_err)
                finally:
                    dl_result['done'] = True

            thread = threading.Thread(target=_dl_worker, daemon=True)
            thread.start()

            # Lightweight file-size watcher. Finds whichever file under the
            # cache dir is currently growing the fastest and reports its size.
            # (HF's hf_hub_download doesn't expose a native progress callback,
            # so this is how we surface live MB/s to the status label.)
            def _watch_loop():
                while not dl_result['done']:
                    try:
                        cur = _snapshot_sizes(_CACHE_DIR)
                        best_path = None
                        best_size = 0
                        for p, sz in cur.items():
                            delta = sz - pre_sizes.get(p, 0)
                            if delta > best_size and sz > 1024 * 1024:
                                best_size = delta
                                best_path = p
                        if best_path:
                            dl_result['live_bytes'] = best_size
                            dl_result['live_file'] = os.path.basename(best_path)
                    except Exception:
                        pass
                    _dl_time.sleep(0.5)

            threading.Thread(target=_watch_loop, daemon=True).start()

            poll_timer = QTimer(self.dialog)

            def _drain_queue():
                """Pop any fresh status messages from the worker's queue."""
                try:
                    while True:
                        msg = temp_inp.progress_queue.get_nowait()
                        if isinstance(msg, tuple) and len(msg) >= 2:
                            dl_result['last_status'] = str(msg[1])
                except _DLEmpty:
                    pass
                except Exception:
                    pass

            def _tick():
                # Always drain new status lines so the label stays fresh.
                _drain_queue()
                if hasattr(self, 'local_model_status_label'):
                    elapsed = int(_dl_time.time() - dl_result['started_at'])
                    bytes_dl = dl_result['live_bytes']
                    parts = [dl_result['last_status']]
                    extras = []
                    if elapsed > 1:
                        extras.append(f"{elapsed}s")
                    if bytes_dl > 1024 * 1024:
                        mb = bytes_dl / (1024 * 1024)
                        if elapsed > 0:
                            rate = mb / elapsed
                            extras.append(f"{mb:.1f} MB @ {rate:.1f} MB/s")
                        else:
                            extras.append(f"{mb:.1f} MB")
                    if extras:
                        parts.append("  (" + " — ".join(extras) + ")")
                    self.local_model_status_label.setText("".join(parts))
                    self.local_model_status_label.setStyleSheet("color: #4a9eff;")
                if dl_result['done']:
                    _finalize()

            def _finalize():
                poll_timer.stop()
                # Drain any last-mile messages (e.g. the final '✅ Download complete').
                _drain_queue()
                self._model_download_in_progress = False
                err = dl_result['error']
                path = dl_result['path']
                if err:
                    if hasattr(self, 'local_model_status_label'):
                        self.local_model_status_label.setText(f"Download failed: {err}")
                        self.local_model_status_label.setStyleSheet("color: red;")
                    QMessageBox.critical(self.dialog, "Download Error", err)
                    return
                if path and os.path.exists(path):
                    if hasattr(self, 'local_model_status_label'):
                        self.local_model_status_label.setText("✅ Download complete")
                        self.local_model_status_label.setStyleSheet("color: green;")
                    self.local_model_entry.setText(path)
                    self.local_model_path_value = path
                    self.main_gui.config[f'manga_{model_type}_model_path'] = path
                    self._save_rendering_settings()
                    self._try_load_model(model_type, path, show_completion_dialog=True)
                else:
                    if hasattr(self, 'local_model_status_label'):
                        self.local_model_status_label.setText("❌ Failed to download model")
                        self.local_model_status_label.setStyleSheet("color: red;")

            poll_timer.timeout.connect(_tick)
            # 250 ms keeps the elapsed-counter readable without hammering the
            # event loop; the worker-thread I/O is unaffected.
            poll_timer.start(250)
            return
        except Exception as e:
            self._model_download_in_progress = False
            QMessageBox.critical(self.dialog, "Error", str(e))
            return
        
        # Define URLs for each model type
        model_urls = {
            'aot': 'https://huggingface.co/ogkalu/aot-inpainting-jit/resolve/main/aot_traced.pt',
            'aot_onnx': 'https://huggingface.co/ogkalu/aot-inpainting/resolve/main/aot.onnx',
            'lama': 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
            'lama_onnx': 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',  
            'anime': 'https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt',
            'anime_onnx': 'https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx',
            'custom-image-edit': '',  # User selects a local GGUF served by the custom image edit endpoint
            'mat': '',  # User must provide
            'ollama': '',  # Not applicable
            'sd_local': ''  # User must provide
        }
        
        url = model_urls.get(model_type, '')
        
        if not url:
            QMessageBox.information(self.dialog, "Manual Download",
                f"Please manually download and browse for {model_type} model")
            return
        
        # Determine filename
        filename_map = {
            'aot': 'aot_traced.pt',
            'aot_onnx': 'aot.onnx',
            'lama': 'big-lama.pt',
            'anime': 'anime-manga-big-lama.pt',
            'anime_onnx': 'lama-manga-dynamic.onnx',
            'lama_onnx': 'lama_fp32.onnx',
            'custom-image-edit': 'custom-image-edit.gguf',
            'fcf_onnx': 'fcf.onnx',
            'sd_inpaint_onnx': 'sd_inpaint_unet.onnx'
        }
        
        filename = filename_map.get(model_type, f'{model_type}.pt')
        save_path = os.path.join('models', filename)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Check if already exists
        if os.path.exists(save_path):
            self.local_model_entry.setText(save_path)
            self.local_model_path_value = save_path
            self.local_model_status_label.setText("✅ Model already downloaded")
            self.local_model_status_label.setStyleSheet("color: green;")
            QMessageBox.information(self.dialog, "Model Ready", f"Model already exists at:\n{save_path}")
            return
        
        # Download the model
        self._perform_download(url, save_path, model_type)

    def _perform_download(self, url: str, save_path: str, model_name: str):
        """Perform the actual download with status updates"""
        import threading
        import requests
        from PySide6.QtWidgets import QApplication
        
        # Show downloading status
        self.local_model_status_label.setText(f"📥 Downloading {model_name.upper()}...")
        self.local_model_status_label.setStyleSheet("color: #17a2b8;")  # Cyan
        QApplication.processEvents()
        
        def download_thread():
            try:
                # Check if it's a HuggingFace URL and use their API if so
                if 'huggingface.co' in url:
                    try:
                        from huggingface_hub import hf_hub_download
                        import re
                        
                        # Parse HuggingFace URL: https://huggingface.co/USER/REPO/resolve/BRANCH/FILENAME
                        match = re.match(r'https://huggingface\.co/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)', url)
                        if match:
                            user, repo, branch, filename = match.groups()
                            repo_id = f"{user}/{repo}"
                            
                            # Download using huggingface_hub (it has its own progress in console)
                            downloaded_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=filename,
                                revision=branch,
                                local_dir=os.path.dirname(save_path),
                                local_dir_use_symlinks=False
                            )
                            
                            # Copy to expected location if different
                            if downloaded_path != save_path:
                                import shutil
                                shutil.copy2(downloaded_path, save_path)
                            
                            return  # Success
                    except Exception as hf_error:
                        print(f"[ERROR] HuggingFace download failed: {hf_error}")
                        raise
                
                # Fallback: Download with requests for non-HuggingFace URLs
                import time
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
            except Exception as e:
                download_result['error'] = str(e)
        
        # Track download result
        download_result = {'error': None}
        
        # Start download in background thread
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()
        
        # Poll for completion using QTimer
        from PySide6.QtCore import QTimer
        check_timer = QTimer()
        
        def check_download_complete():
            if not thread.is_alive():
                check_timer.stop()
                # Handle result on main thread
                if download_result['error']:
                    self._download_failed(download_result['error'])
                else:
                    self._download_complete(save_path, model_name)
            else:
                QApplication.processEvents()
        
        check_timer.timeout.connect(check_download_complete)
        check_timer.start(100)  # Check every 100ms

    def _download_complete(self, save_path: str, model_name: str):
        """Handle successful download and attempt to load the model"""
        from PySide6.QtCore import QTimer
        
        # Update the model path entry
        self.local_model_entry.setText(save_path)
        self.local_model_path_value = save_path
        
        # Save to config
        self.main_gui.config[f'manga_{model_name}_model_path'] = save_path
        self._save_rendering_settings()
        
        # Log to main GUI
        self.main_gui.append_log(f"✅ Downloaded {model_name} model to: {save_path}")
        
        # Attempt to load the downloaded model after a short delay
        QTimer.singleShot(100, lambda: self._try_load_model(model_name, save_path, show_completion_dialog=True))
        
        # Auto-load the downloaded model in background with completion dialog
        self.local_model_status_label.setText("⏳ Loading downloaded model...")
        self.local_model_status_label.setStyleSheet("color: orange;")
        
        # Load in background - completion dialog will be shown when done
        self._try_load_model(model_name, save_path, show_completion_dialog=True)

    def _download_failed(self, error: str):
        """Handle download failure"""
        from PySide6.QtWidgets import QMessageBox
        
        QMessageBox.critical(self.dialog, "Download Failed", f"Failed to download model:\n{error}")
        self.main_gui.append_log(f"❌ Model download failed: {error}")

    def _show_model_info(self):
        """Show information about models"""
        model_type = self.local_model_type_value
        
        info = {
            'aot': "AOT GAN Model:\n\n"
                   "• Auto-downloads from HuggingFace\n"
                   "• Traced PyTorch JIT model\n"
                   "• Good for general inpainting\n"
                   "• Fast processing speed\n"
                   "• File size: ~100MB",
            
            'aot_onnx': "AOT ONNX Model:\n\n"
                        "• Optimized ONNX version\n"
                        "• Auto-downloads from HuggingFace\n"
                        "• 2-3x faster than PyTorch version\n"
                        "• Great for batch processing\n"
                        "• Lower memory usage\n"
                        "• File size: ~100MB",
            
            'lama': "LaMa Model:\n\n"
                    "• Auto-downloads anime-optimized version\n"
                    "• Best quality for manga/anime\n"
                    "• Large model (~200MB)\n"
                    "• Excellent at removing text from bubbles\n"
                    "• Preserves art style well",
            
            'anime': "Anime-Specific Model:\n\n"
                     "• Same as LaMa anime version\n"
                     "• Optimized for manga/anime art\n"
                     "• Auto-downloads from GitHub\n"
                     "• Recommended for manga translation\n"
                     "• Preserves screen tones and patterns",
            
            'anime_onnx': "Anime ONNX Model:\n\n"
                          "• Optimized ONNX version for speed\n"
                          "• Auto-downloads from HuggingFace\n"
                          "• 2-3x faster than PyTorch version\n"
                          "• Perfect for batch processing\n"
                          "• Same quality as anime model\n"
                          "• File size: ~190MB\n"
                          "• DEFAULT for inpainting",
            
            'custom-image-edit': "Custom Image Edit Endpoint:\n\n"
                                 "- Select a local .gguf model file\n"
                                 "- Sends masked manga cleanup requests to the Custom Image Edit Endpoint\n"
                                 "- Uses the OpenAI-compatible /images/edits API\n"
                                 "- Text requests keep using your normal LLM endpoint\n"
                                 "- Best for running a local image-edit model beside a separate cloud/text LLM",

            'mat': "MAT Model:\n\n"
                   "• Manual download required\n"
                   "• Get from: github.com/fenglinglwb/MAT\n"
                   "• Good for high-resolution images\n"
                   "• Slower but high quality\n"
                   "• File size: ~500MB",
            
            'ollama': "Ollama:\n\n"
                      "• Uses local Ollama server\n"
                      "• No model download needed here\n"
                      "• Run: ollama pull llava\n"
                      "• Context-aware inpainting\n"
                      "• Requires Ollama running locally",
            
            'sd_local': "Stable Diffusion:\n\n"
                        "• Manual download required\n"
                        "• Get from HuggingFace\n"
                        "• Requires significant VRAM (4-8GB)\n"
                        "• Best quality but slowest\n"
                        "• Can use custom prompts"
        }
        
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        from PySide6.QtCore import Qt
        
        # Create info dialog
        info_dialog = QDialog(self.dialog)
        info_dialog.setWindowTitle(f"{model_type.upper()} Model Information")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.23)  # 23% of screen width
        height = int(screen.height() * 0.32)  # 32% of screen height
        info_dialog.setFixedSize(width, height)
        info_dialog.setModal(True)
        
        layout = QVBoxLayout(info_dialog)
        
        # Info text
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setPlainText(info.get(model_type, "Please select a model type first"))
        layout.addWidget(text_widget)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(info_dialog.close)
        close_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 5px 15px; }")
        layout.addWidget(close_btn)
        
        info_dialog.exec()

    def _toggle_inpaint_controls_visibility(self):
            """Toggle visibility of inpaint controls (mask expansion and passes) based on skip inpainting setting"""
            # Just return if the frame doesn't exist - prevents AttributeError
            if not hasattr(self, 'inpaint_controls_frame'):
                return
                
            if self.skip_inpainting_value:
                self.inpaint_controls_frame.hide()
            else:
                # Show it back
                self.inpaint_controls_frame.show()

    def _configure_inpaint_api(self):
        """Configure cloud inpainting API"""
        from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
        from PySide6.QtCore import Qt
        import webbrowser
        
        # Show instructions
        result = QMessageBox.question(
            self.dialog,
            "Configure Cloud Inpainting",
            "Cloud inpainting uses Replicate API for questionable results.\n\n"
            "1. Go to replicate.com and sign up (free tier available?)\n"
            "2. Get your API token from Account Settings\n"
            "3. Enter it here\n\n"
            "Pricing: ~$0.0023 per image?\n"
            "Free tier: ~100 images per month?\n\n"
            "Would you like to proceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result != QMessageBox.Yes:
            return
        
        # Open Replicate page
        webbrowser.open("https://replicate.com/account/api-tokens")
        
        # Create API key input dialog
        api_dialog = QDialog(self.dialog)
        api_dialog.setWindowTitle("Replicate API Key")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.21)  # 21% of screen width
        height = int(screen.height() * 0.14)  # 14% of screen height
        api_dialog.setFixedSize(width, height)
        api_dialog.setModal(True)
        
        layout = QVBoxLayout(api_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Label
        label = QLabel("Enter your Replicate API key:")
        layout.addWidget(label)
        
        # Entry with show/hide
        entry_layout = QHBoxLayout()
        entry = QLineEdit()
        entry.setEchoMode(QLineEdit.Password)
        entry_layout.addWidget(entry)
        
        # Toggle show/hide
        show_btn = QPushButton("Show")
        show_btn.setFixedWidth(60)
        def toggle_show():
            if entry.echoMode() == QLineEdit.Password:
                entry.setEchoMode(QLineEdit.Normal)
                show_btn.setText("Hide")
            else:
                entry.setEchoMode(QLineEdit.Password)
                show_btn.setText("Show")
        show_btn.clicked.connect(toggle_show)
        entry_layout.addWidget(show_btn)
        
        layout.addLayout(entry_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(100)  # Ensure enough width for text
        cancel_btn.clicked.connect(api_dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; }")
        ok_btn.clicked.connect(api_dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
        
        # Focus and key bindings
        entry.setFocus()
        
        # Execute dialog
        if api_dialog.exec() == QDialog.Accepted:
            api_key = entry.text().strip()
            
            if api_key:
                try:
                    # Save the API key
                    self.main_gui.config['replicate_api_key'] = api_key
                    self.main_gui.save_config(show_message=False)
                    
                    # Update UI
                    self.inpaint_api_status_label.setText("✅ Cloud inpainting configured")
                    self.inpaint_api_status_label.setStyleSheet("color: green;")
                    
                    # Set flag on translator
                    if self.translator:
                        self.translator.use_cloud_inpainting = True
                        self.translator.replicate_api_key = api_key
                        
                    self._log("✅ Cloud inpainting API configured", "success")
                    
                except Exception as e:
                    QMessageBox.critical(self.dialog, "Error", f"Failed to save API key:\n{str(e)}")

    def _clear_inpaint_api(self):
        """Clear the inpainting API configuration"""
        self.main_gui.config['replicate_api_key'] = ''
        self.main_gui.save_config(show_message=False)
        
        self.inpaint_api_status_label.setText("❌ Inpainting API not configured")
        self.inpaint_api_status_label.setStyleSheet("color: red;")
        
        if hasattr(self, 'translator') and self.translator:
            self.translator.use_cloud_inpainting = False
            self.translator.replicate_api_key = None
            
        self._log("🗑️ Cleared inpainting API configuration", "info")
        
        # Note: Clear button management would need to be handled differently in PySide6
        # For now, we'll skip automatic button removal

    def _manga_drop_local_paths(self, mime_data) -> List[str]:
        """Extract supported local files/folders from a Qt drop payload."""
        try:
            if not mime_data or not mime_data.hasUrls():
                return []
            supported_extensions = {
                '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.cbz'
            }
            paths = []
            seen = set()
            for url in mime_data.urls():
                if not url.isLocalFile():
                    continue
                path = os.path.abspath(url.toLocalFile())
                if not os.path.isdir(path):
                    if not os.path.isfile(path):
                        continue
                    if os.path.splitext(path)[1].lower() not in supported_extensions:
                        continue
                key = os.path.normcase(path)
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
            return paths
        except Exception:
            return []

    def _set_manga_drop_highlight(self, active: bool) -> None:
        """Show visual feedback while supported paths are dragged over the section."""
        file_list = getattr(self, 'file_listbox', None)
        if not file_list:
            return
        file_list.setProperty("mangaDropActive", bool(active))
        file_list.style().unpolish(file_list)
        file_list.style().polish(file_list)
        file_list.update()

    def _select_manga_folders(self) -> List[str]:
        """Open a folder picker that supports multi-select when Qt allows it."""
        folders: List[str] = []
        try:
            from PySide6.QtWidgets import QFileDialog, QAbstractItemView, QListView, QTreeView
            dialog = QFileDialog(self.dialog, "Select Manga Folder(s)")
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            for view in dialog.findChildren((QListView, QTreeView)):
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            if dialog.exec():
                folders = [path for path in dialog.selectedFiles() if os.path.isdir(path)]
        except Exception:
            folders = []

        if not folders:
            try:
                from PySide6.QtWidgets import QFileDialog
                folder = QFileDialog.getExistingDirectory(
                    self.dialog,
                    "Select Folder with Manga Images or CBZ"
                )
                if folder:
                    folders = [folder]
            except Exception:
                folders = []
        return folders

    def _add_files(self):
        """Add image files (and CBZ archives) to the list"""
        from PySide6.QtWidgets import QFileDialog
        
        files, _ = QFileDialog.getOpenFileNames(
            self.dialog,
            "Select Manga Images or CBZ",
            "",
            "Images / CBZ (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.cbz);;Image files (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;Comic Book Zip (*.cbz);;All files (*.*)"
        )
        
        if not files:
            return

        if not self._can_add_manga_paths_from_single_source(files):
            return
        
        # Ensure temp root for CBZ extraction lives for the session
        cbz_temp_root = getattr(self, 'cbz_temp_root', None)
        if cbz_temp_root is None:
            try:
                import tempfile
                cbz_temp_root = tempfile.mkdtemp(prefix='glossarion_cbz_')
                self.cbz_temp_root = cbz_temp_root
            except Exception:
                cbz_temp_root = None
        
        for path in files:
            lower = path.lower()
            if lower.endswith('.cbz'):
                # Extract images from CBZ and add them in natural sort order
                try:
                    import zipfile, shutil
                    base = os.path.splitext(os.path.basename(path))[0]
                    extract_dir = os.path.join(self.cbz_temp_root or os.path.dirname(path), base)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(path, 'r') as zf:
                        # Extract all to preserve subfolders and avoid name collisions
                        zf.extractall(extract_dir)
                    # Initialize CBZ job tracking
                    if not hasattr(self, 'cbz_jobs'):
                        self.cbz_jobs = {}
                    if not hasattr(self, 'cbz_image_to_job'):
                        self.cbz_image_to_job = {}
                    # Prepare output dir next to source CBZ
                    out_dir = os.path.join(os.path.dirname(path), f"{base}_translated")
                    self.cbz_jobs[path] = {
                        'extract_dir': extract_dir,
                        'out_dir': out_dir,
                    }
                    # Collect all images recursively from extract_dir
                    added = 0
                    for root, _, files_in_dir in os.walk(extract_dir):
                        for fn in sorted(files_in_dir, key=_natural_sort_key):
                            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                                target_path = os.path.join(root, fn)
                                if target_path not in self.selected_files:
                                    self.selected_files.append(target_path)
                                    self._add_manga_file_item(target_path)
                                    added += 1
                                # Map extracted image to its CBZ job
                                self.cbz_image_to_job[target_path] = path
                    self._log(f"📦 Added {added} images from CBZ: {os.path.basename(path)}", "info")
                except Exception as e:
                    self._log(f"❌ Failed to read CBZ {os.path.basename(path)}: {e}", "error")
            else:
                if path not in self.selected_files:
                    self.selected_files.append(path)
                    self._add_manga_file_item(path)
        
        self._apply_manga_file_sort()

        # Auto-select first image to trigger preview
        if len(self.selected_files) > 0 and self.file_listbox.count() > 0:
            self.file_listbox.setCurrentRow(0)
        self._update_manga_image_range_display()
        
        # Update thumbnail preview list
        if hasattr(self, 'image_preview_widget'):
            self._update_manga_preview_image_list_for_range()
        
        # Persist the file list
        self._persist_selected_files()
    
    def _add_folder(self):
        """Add all images (and CBZ archives) from one or more folders."""
        folders = self._select_manga_folders()
        if not folders:
            return

        if not self._can_add_manga_paths_from_single_source(folders):
            return

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        cbz_ext = '.cbz'
        added_images = 0
        if not hasattr(self, 'manga_selected_folder_roots'):
            self.manga_selected_folder_roots = []
        for folder in folders:
            folder_abs = os.path.abspath(folder)
            if folder_abs not in self.manga_selected_folder_roots:
                self.manga_selected_folder_roots.append(folder_abs)

        for folder in folders:
            for root, dirnames, filenames in os.walk(folder):
                dirnames[:] = [
                    dirname for dirname in dirnames
                    if not dirname.lower().endswith('_translated')
                    and dirname.lower() not in {'glossary', 'mangaglossary_backup', '__macosx'}
                ]
                dirnames.sort(key=_natural_sort_key)
                for filename in sorted(filenames, key=_natural_sort_key):
                    filepath = os.path.join(root, filename)
                    if not os.path.isfile(filepath):
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        if filepath not in self.selected_files:
                            self.selected_files.append(filepath)
                            self._add_manga_file_item(filepath)
                            added_images += 1
                            if len(self.selected_files) == 1:
                                self.file_listbox.setCurrentRow(0)
                    elif ext == cbz_ext:
                        added_images += self._add_cbz_archive_images(filepath, image_extensions)

        self._apply_manga_file_sort()

        if added_images:
            if len(folders) > 1:
                self._log(f"📂 Added {added_images} manga images from {len(folders)} folders", "info")
            else:
                self._log(f"📂 Added {added_images} manga images from folder: {os.path.basename(folders[0])}", "info")
        
        # Update thumbnail preview list
        if hasattr(self, 'image_preview_widget'):
            self._update_manga_preview_image_list_for_range()
        self._update_manga_image_range_display()
        
        # Persist the file list
        self._update_manga_image_range_display()
        self._persist_selected_files()
    
    def _set_file_selection_editing_enabled(self, enabled: bool):
        """Enable/disable file-list mutations without disabling list scrolling."""
        try:
            enabled = bool(enabled)
            self._file_selection_editing_enabled = enabled
            self._set_manga_drop_highlight(False)
            for attr in ('add_files_btn', 'add_folder_btn', 'remove_files_btn', 'clear_files_btn'):
                btn = getattr(self, attr, None)
                if btn is not None:
                    btn.setEnabled(enabled)
            if hasattr(self, 'file_listbox') and self.file_listbox:
                self.file_listbox.setEnabled(True)
                self.file_listbox.setDragDropMode(
                    QAbstractItemView.DragDropMode.InternalMove
                    if enabled else QAbstractItemView.DragDropMode.NoDragDrop
                )
        except Exception as e:
            print(f"[FILE_LIST] Failed to update editing controls: {e}")

    def _set_file_list_current_row_without_scroll(self, row: int):
        """Select a file-list row while preserving the user's scrollbar position."""
        if not hasattr(self, 'file_listbox') or not self.file_listbox:
            return
        if row < 0 or row >= self.file_listbox.count():
            return
        scrollbar = self.file_listbox.verticalScrollBar()
        saved_scroll = scrollbar.value() if scrollbar else None
        self.file_listbox.blockSignals(True)
        try:
            self.file_listbox.setCurrentRow(row)
        finally:
            self.file_listbox.blockSignals(False)
        if scrollbar is not None and saved_scroll is not None:
            scrollbar.setValue(saved_scroll)
            QTimer.singleShot(0, lambda sb=scrollbar, value=saved_scroll: sb.setValue(value))

    def _remove_selected(self):
        """Remove selected files from the list and clear preview if the current image was removed.
        Before removing, persist the current image state to avoid losing OCR/rectangles.
        """
        selected_items = self.file_listbox.selectedItems()
        
        if not selected_items:
            return
        
        # Persist current image state before mutating the list
        try:
            if hasattr(self, '_current_image_path') and self._current_image_path:
                ImageRenderer._persist_current_image_state(self)
        except Exception:
            pass
        
        # Track current image and removed paths
        current_path = getattr(self, '_current_image_path', None)
        removed_paths = set()
        
        # Remove in reverse order to maintain indices
        rows = sorted([self.file_listbox.row(item) for item in selected_items], reverse=True)
        for row in rows:
            self.file_listbox.takeItem(row)
            if 0 <= row < len(self.selected_files):
                removed_paths.add(self.selected_files[row])
                del self.selected_files[row]
        if removed_paths:
            removed_keys = {self._skip_key_for_path(path) for path in removed_paths}
            self.skipped_processing_files = {
                key for key in self._manual_skipped_processing_keys()
                if key not in removed_keys
            }
        
        # If the current image was removed or list is now empty, clear the active preview state.
        if self.file_listbox.count() == 0:
            if hasattr(self, 'image_preview_widget'):
                self.image_preview_widget.clear(clear_image_list=True)
                self.image_preview_widget.set_image_list([])
            self._current_image_path = None
        elif current_path and current_path in removed_paths:
            self._current_image_path = None
            if hasattr(self, 'image_preview_widget'):
                self.image_preview_widget.clear()
            self.file_listbox.setCurrentRow(0)
            self._update_manga_preview_image_list_for_range()
        
        # Persist the file list
        self._persist_selected_files()
    
    def _clear_all(self):
        """Clear all files from the list.
        Persist current image state first to preserve OCR/rectangles.
        """
        # Persist current image state before clearing
        try:
            if hasattr(self, '_current_image_path') and self._current_image_path:
                ImageRenderer._persist_current_image_state(self)
        except Exception:
            pass
        
        self.file_listbox.clear()
        self.selected_files.clear()
        self.skipped_processing_files.clear()
        if hasattr(self, 'manga_selected_folder_roots'):
            self.manga_selected_folder_roots.clear()
        self._current_image_path = None
        for attr in (
            '_original_image_path',
            '_cleaned_image_path',
            '_current_regions',
            '_recognized_texts',
            '_translated_texts',
            '_recognition_data',
            '_translation_data',
        ):
            if hasattr(self, attr):
                try:
                    if attr in ('_current_regions', '_recognized_texts', '_translated_texts'):
                        setattr(self, attr, [])
                    elif attr in ('_recognition_data', '_translation_data'):
                        setattr(self, attr, {})
                    else:
                        setattr(self, attr, None)
                except Exception:
                    pass
        try:
            ImageRenderer._remove_processing_overlay(self, clear_all=True)
            ImageRenderer._clear_cross_image_state(self)
        except Exception:
            pass
        self._update_manga_image_range_display()
        # Clear image preview when list is cleared
        if hasattr(self, 'image_preview_widget'):
            self.image_preview_widget.clear(clear_image_list=True)
            self.image_preview_widget.set_image_list([])
        
        # Persist the empty file list
        self._persist_selected_files()
    
    def _toggle_sort(self, sort_type):
        """Toggle between ascending and descending sort for the given type.
        
        Args:
            sort_type: One of 'name', 'numeric', 'date'
        """
        if not self.selected_files:
            return
        
        # Get current state and toggle it
        if not hasattr(self, '_sort_ascending'):
            self._sort_ascending = {'name': True, 'numeric': False, 'date': True}
        
        is_ascending = self._sort_ascending.get(sort_type, True)
        
        # Apply sort with current direction
        self._sort_files(sort_type, reverse=not is_ascending)
        
        # Toggle the state for next click
        self._sort_ascending[sort_type] = not is_ascending
        
        # Update button text to show current direction
        arrow = "↓" if not is_ascending else "↑"  # Down arrow for descending, up for ascending
        
        if sort_type == 'name':
            self.sort_name_btn.setText(f"{arrow} Name")
            self.sort_name_btn.setToolTip(f"Sort by filename ({'Z-A' if not is_ascending else 'A-Z'}). Click to switch.")
        elif sort_type == 'numeric':
            self.sort_numeric_btn.setText(f"{arrow} Number")
            self.sort_numeric_btn.setToolTip(f"Sort numerically ({'10, 2, 1' if not is_ascending else '1, 2, 10'}). Click to switch.")
        elif sort_type == 'date':
            self.sort_date_btn.setText(f"{arrow} Date")
            self.sort_date_btn.setToolTip(f"Sort by date ({'newest first' if not is_ascending else 'oldest first'}). Click to switch.")

    def _on_files_reordered(self, parent, start, end, destination, row):
        """Handle drag-and-drop reordering of files in the listbox.
        
        This is called when the user drags items to reorder them.
        We need to sync the selected_files list with the new listbox order.
        """
        try:
            old_order = list(self.selected_files)
            # Rebuild selected_files list based on current listbox order
            new_order = []
            seen_paths = set()
            for i in range(self.file_listbox.count()):
                item = self.file_listbox.item(i)
                filepath = item.data(Qt.UserRole) if item else None
                if not filepath:
                    filename = _manga_filename_without_skip_prefix(item.text()) if item else ""
                    # Fallback for older rows that do not have full-path data yet.
                    for candidate in old_order:
                        if candidate in seen_paths:
                            continue
                        if os.path.basename(candidate) == filename:
                            filepath = candidate
                            break
                if filepath and filepath in old_order and filepath not in seen_paths:
                    new_order.append(filepath)
                    seen_paths.add(filepath)

            for filepath in old_order:
                if filepath not in seen_paths:
                    new_order.append(filepath)
            
            # Update selected_files with new order
            self.selected_files = new_order
            self._manga_file_sort = None
            
            # Update thumbnail preview list
            if hasattr(self, 'image_preview_widget'):
                self._update_manga_preview_image_list_for_range()
            
            print(f"[FILE_REORDER] Files reordered via drag-and-drop")
            
            # Persist the new order
            self._persist_selected_files()
        except Exception as e:
            print(f"[FILE_REORDER] Error syncing file order: {e}")
    
    def _sync_file_selection_to_preview(self):
        """Sync file list selection to match the currently displayed image in preview.
        
        Called when user clicks on the image viewer to ensure the file list selection
        matches the displayed image, which triggers proper state restoration.
        """
        print = _manga_cmd_debug_print
        try:
            # Get the currently displayed image path from the preview
            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
            if not current_image:
                return
            
            # Normalize for comparison
            current_normalized = os.path.normcase(os.path.normpath(current_image))
            
            # Find this image in selected_files
            target_row = -1
            for idx, filepath in enumerate(self.selected_files):
                if os.path.normcase(os.path.normpath(filepath)) == current_normalized:
                    target_row = idx
                    break
            
            if target_row < 0:
                return
            
            # Check if already selected
            current_row = self.file_listbox.currentRow()
            if current_row == target_row:
                # Already selected - just trigger state restoration manually
                print(f"[SYNC_SELECTION] Already at correct row {target_row}, triggering state restoration")
                ImageRenderer._restore_image_state(self, current_image)
                return
            
            # Select the correct row (will trigger _on_file_selection_changed)
            print(f"[SYNC_SELECTION] Syncing file list selection to row {target_row} for {os.path.basename(current_image)}")
            self.file_listbox.setCurrentRow(target_row)
        except Exception as e:
            print(f"[SYNC_SELECTION] Error syncing selection: {e}")
    
    def _on_file_selection_changed(self):
        """Handle file list selection changes to update image preview"""
        print = _manga_cmd_debug_print
        try:
            print(f"[FILE_SELECTION] _on_file_selection_changed triggered")
            
            # PERSIST CURRENT IMAGE STATE before switching
            if hasattr(self, '_current_image_path') and self._current_image_path:
                ImageRenderer._persist_current_image_state(self)
            
            selected_items = self.file_listbox.selectedItems()
            
            # Prevent empty selection - always keep at least one item selected
            if not selected_items and self.file_listbox.count() > 0:
                # Re-select the previously selected item or the first item
                last_row = getattr(self, '_last_selected_row', 0)
                if last_row >= self.file_listbox.count():
                    last_row = 0
                self.file_listbox.blockSignals(True)  # Prevent recursive signal
                self.file_listbox.setCurrentRow(last_row)
                self.file_listbox.blockSignals(False)
                print(f"[FILE_SELECTION] Prevented empty selection, re-selected row {last_row}")
                return
            
            if not selected_items:
                # Only clear if there are truly no files in the list
                if hasattr(self, 'image_preview_widget'):
                    self.image_preview_widget.clear()
                # CLEAR STATE ISOLATION FIX: Clear recognition and translation data when no image is selected
                ImageRenderer._clear_cross_image_state(self)
                self._current_image_path = None
                return
            
            # Prefer current row (important when ExtendedSelection keeps multiple items selected)
            row = self.file_listbox.currentRow()
            if row < 0 and selected_items:
                # Fallback to any selected item
                first_item = selected_items[0]
                row = self.file_listbox.row(first_item)
            
            # Track the selected row for preventing empty selection
            self._last_selected_row = row
            
            # Get the corresponding file path
            current_item = self.file_listbox.item(row) if 0 <= row < self.file_listbox.count() else None
            image_path = current_item.data(Qt.UserRole) if current_item else None
            if not image_path and 0 <= row < len(self.selected_files):
                image_path = self.selected_files[row]

            if image_path:
                self._sync_manga_process_group_for_image(image_path)
                self._update_manga_preview_image_list_for_range()
                
                # Update current image path for state tracking
                self._current_image_path = image_path
                
                # CLEAR STATE ISOLATION FIX: Clear recognition and translation data for new image to prevent cross-contamination
                ImageRenderer._clear_cross_image_state(self)
                
                # DEBUG: Log the image loading attempt
                self._log(f"🖼️ Loading preview: {os.path.basename(image_path)}", "debug")
                
                # Load the image into the preview (always visible) - ALWAYS USE SOURCE
                if hasattr(self, 'image_preview_widget'):
                    if os.path.exists(image_path):
                        # Clear translated path reference when loading new image
                        try:
                            if hasattr(self.image_preview_widget, 'current_translated_path'):
                                self.image_preview_widget.current_translated_path = None
                        except Exception:
                            pass
                # Just load the source image - tabbed view will handle translated output
                        self.image_preview_widget.load_image(image_path)
                        # After loading, restore any saved rectangles/overlays for this image
                        try:
                            ImageRenderer._restore_image_state(self, image_path)
                        except Exception:
                            pass
                        # Sync thumbnail selection to match file list selection
                        try:
                            if hasattr(self.image_preview_widget, '_update_thumbnail_selection'):
                                self.image_preview_widget._update_thumbnail_selection(image_path)
                        except Exception:
                            pass
                        self._log(f"✅ Preview loaded: {os.path.basename(image_path)}", "debug")
                    else:
                        self._log(f"❌ Image file not found: {image_path}", "error")
                else:
                    self._log("❌ Image preview widget not initialized", "error")
            else:
                self._log(f"❌ Invalid row index: {row} (total files: {len(self.selected_files)})", "error")
        except Exception as e:
            # Log errors for debugging
            self._log(f"❌ Error loading image preview: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
    
    def _attach_logging_bridge(self):
        """Attach logging handlers so library/client logs appear in the GUI log.
        Safe to call multiple times (won't duplicate handlers). Matches translator_gui.py implementation.
        """
        try:
            if getattr(self, '_gui_log_handler', None) is None:
                # Build handler
                handler = _MangaGuiLogHandler(self, level=logging.INFO)
                fmt = logging.Formatter('%(message)s')
                handler.setFormatter(fmt)
                self._gui_log_handler = handler
                
                # Target relevant loggers - includes httpx for HTTP request logs
                target_loggers = [
                    'unified_api_client',
                    'httpx',
                    'requests.packages.urllib3',
                    'openai',
                    'bubble_detector',
                    'local_inpainter',
                    'manga_translator'
                ]
                
                for name in target_loggers:
                    try:
                        lg = logging.getLogger(name)
                        # Avoid duplicate handler attachments
                        if not any(isinstance(h, _MangaGuiLogHandler) for h in lg.handlers):
                            lg.addHandler(handler)
                        # Ensure at least INFO level to see HTTP requests and retry/backoff notices
                        if lg.level > logging.INFO or lg.level == logging.NOTSET:
                            lg.setLevel(logging.INFO)
                    except Exception:
                        pass
        except Exception as e:
            try:
                self._log(f"⚠️ Failed to attach GUI log handlers: {e}")
            except Exception:
                pass

    def _monitor_translation_output(self, image_path: str):
        """Monitor translation progress and auto-update preview with latest results"""
        try:
            import threading
            import time
            import glob
            
            def monitor_files():
                """Monitor for new files and update preview automatically"""
                base_dir = os.path.dirname(image_path)
                image_name = os.path.basename(image_path)
                
                # Define the expected workflow folders in order
                workflow_folders = [
                    "1_detected",
                    "2_ocr", 
                    "3_segmented",
                    "4_cleaned",
                    "5_rendered",
                    "translated"  # Final output
                ]
                
                last_updated = None
                
                # Monitor for 60 seconds max
                start_time = time.time()
                while time.time() - start_time < 60:
                    try:
                        # Check each workflow folder for the latest result
                        latest_file = None
                        latest_step = None
                        
                        for step_name in reversed(workflow_folders):  # Check latest first
                            step_dir = os.path.join(base_dir, step_name)
                            if os.path.exists(step_dir):
                                step_file = os.path.join(step_dir, image_name)
                                if os.path.exists(step_file) and step_file != last_updated:
                                    latest_file = step_file
                                    latest_step = step_name
                                    break
                        
                        # Update preview if we found a new file
                        if latest_file and latest_file != last_updated:
                            def update_preview():
                                try:
                                    self.image_preview_widget.load_image(latest_file)
                                    step_display = latest_step.replace("_", " ").title()
                                    self._log(f"🖼️ Preview updated: {step_display}", "info")
                                except Exception as e:
                                    self._log(f"Failed to update preview: {e}", "error")
                            
                            # Use QTimer to update on main thread
                            from PySide6.QtCore import QTimer
                            QTimer.singleShot(0, update_preview)
                            last_updated = latest_file
                        
                        time.sleep(0.5)  # Check every 500ms
                        
                    except Exception as e:
                        # Don't let monitoring errors crash the thread
                        time.sleep(1)
                        continue
            
            # Start monitoring in background thread
            monitor_thread = threading.Thread(target=monitor_files, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            # Silently ignore monitoring setup errors
            pass
    
    def _redirect_stderr(self, enable: bool):
        """Temporarily redirect stderr to the GUI log (captures tqdm/HF progress)."""
        try:
            if enable:
                if not hasattr(self, '_old_stderr') or self._old_stderr is None:
                    self._old_stderr = sys.stderr
                    sys.stderr = _StreamToGuiLog(lambda s: self._log(s, 'info'))
                self._stderr_redirect_on = True
            else:
                if hasattr(self, '_old_stderr') and self._old_stderr is not None:
                    sys.stderr = self._old_stderr
                    self._old_stderr = None
                self._stderr_redirect_on = False
            # Update combined flag to avoid double-forwarding with logging handler
            self._stdio_redirect_active = bool(self._stdout_redirect_on or self._stderr_redirect_on)
        except Exception:
            pass

    def _redirect_stdout(self, enable: bool):
        """Temporarily redirect stdout to the GUI log."""
        try:
            if enable:
                if not hasattr(self, '_old_stdout') or self._old_stdout is None:
                    self._old_stdout = sys.stdout
                    sys.stdout = _StreamToGuiLog(lambda s: self._log(s, 'info'))
                self._stdout_redirect_on = True
            else:
                if hasattr(self, '_old_stdout') and self._old_stdout is not None:
                    sys.stdout = self._old_stdout
                    self._old_stdout = None
                self._stdout_redirect_on = False
            # Update combined flag to avoid double-forwarding with logging handler
            self._stdio_redirect_active = bool(self._stdout_redirect_on or self._stderr_redirect_on)
        except Exception:
            pass

    def _on_log_scroll(self, value):
        """Detect when user manually scrolls in the log"""
        try:
            scrollbar = self.log_text.verticalScrollBar()
            # Distance from bottom
            distance = max(0, scrollbar.maximum() - int(value))
            # Consider 'near bottom' generously so manual override is disabled when close
            near_threshold = max(20, int(scrollbar.pageStep() * 0.9))  # ~one page or ≥20px
            disable_threshold = max(near_threshold + 40, int(scrollbar.pageStep() * 1.5))  # clearly away from bottom

            if distance <= near_threshold:
                # Close enough to bottom — treat as at bottom and re-enable auto-scroll
                self._user_scrolled_up = False
                self._was_at_bottom = True
            elif distance >= disable_threshold:
                # Only mark as manually scrolled when clearly away from bottom
                self._user_scrolled_up = True
                self._was_at_bottom = False
            # If in between thresholds, keep prior state (prevents flapping)

            # Update helper button
            self._update_log_scroll_button()
        except Exception:
            pass

    def _set_log_auto_scroll_disabled(self, disabled):
        try:
            self._log_auto_scroll_disabled = bool(disabled)
            if self._log_auto_scroll_disabled:
                self._user_scrolled_up = True
            else:
                self._user_scrolled_up = False
                if hasattr(self, 'log_text') and self.log_text:
                    sb = self.log_text.verticalScrollBar()
                    sb.setValue(sb.maximum())
            self._update_log_scroll_button()
        except Exception:
            pass

    def _show_log_context_menu(self, pos):
        try:
            from PySide6.QtWidgets import QMenu
            menu = QMenu(self.log_text)
            menu.setStyleSheet("QMenu::item { padding: 4px 16px; }")
            cursor = self.log_text.textCursor()
            has_selection = bool(cursor and cursor.hasSelection())

            copy_action = menu.addAction("Copy")
            copy_action.setEnabled(has_selection)
            copy_action.triggered.connect(lambda: self.log_text.copy())

            menu.addSeparator()

            select_all_action = menu.addAction("Select All")
            select_all_action.triggered.connect(self.log_text.selectAll)

            menu.addSeparator()

            auto_scroll_disabled = bool(getattr(self, '_log_auto_scroll_disabled', False))
            auto_scroll_action = menu.addAction("Enable Auto Scroll" if auto_scroll_disabled else "Disable Auto Scroll")
            auto_scroll_action.triggered.connect(
                lambda _checked=False, disabled=auto_scroll_disabled: self._set_log_auto_scroll_disabled(not disabled)
            )

            menu.exec(self.log_text.mapToGlobal(pos))
        except Exception:
            pass

    def _position_log_scroll_button(self):
        try:
            if not hasattr(self, 'log_scroll_btn') or not hasattr(self, 'log_text'):
                return
            btn = self.log_scroll_btn
            vp = self.log_text.viewport()
            margin = 8
            x = max(0, vp.width() - btn.sizeHint().width() - margin)
            y = max(0, vp.height() - btn.sizeHint().height() - margin)
            btn.move(x, y)
        except Exception:
            pass

    def _update_log_scroll_button(self):
        try:
            if not hasattr(self, 'log_scroll_btn') or not hasattr(self, 'log_text'):
                return
            visible = bool(getattr(self, '_user_scrolled_up', False) or getattr(self, '_log_auto_scroll_disabled', False))
            self.log_scroll_btn.setVisible(visible)
            if visible:
                self._position_log_scroll_button()
        except Exception:
            pass

    def _scroll_log_to_bottom(self):
        try:
            if hasattr(self, 'log_text'):
                self._set_log_auto_scroll_disabled(False)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        try:
            ocr_drop_targets = getattr(self, '_ocr_import_drop_targets', set())
            if obj in ocr_drop_targets:
                event_type = event.type()
                if event_type in (QEvent.DragEnter, QEvent.DragMove):
                    paths = self._ocr_drop_local_json_paths(event.mimeData())
                    if len(paths) == 1 and obj.isEnabled():
                        event.acceptProposedAction()
                        self._set_ocr_import_drop_highlight(True)
                        return True
                elif event_type == QEvent.DragLeave:
                    self._set_ocr_import_drop_highlight(False)
                elif event_type == QEvent.Drop:
                    paths = self._ocr_drop_local_json_paths(event.mimeData())
                    self._set_ocr_import_drop_highlight(False)
                    if len(paths) == 1 and obj.isEnabled():
                        event.acceptProposedAction()
                        self._import_batch_ocr_path(paths[0])
                        return True

            drop_targets = getattr(self, '_manga_drop_targets', set())
            if obj in drop_targets:
                event_type = event.type()
                if event_type in (QEvent.DragEnter, QEvent.DragMove):
                    paths = self._manga_drop_local_paths(event.mimeData())
                    if paths and getattr(self, '_file_selection_editing_enabled', True):
                        event.acceptProposedAction()
                        self._set_manga_drop_highlight(True)
                        return True
                elif event_type == QEvent.DragLeave:
                    self._set_manga_drop_highlight(False)
                elif event_type == QEvent.Drop:
                    paths = self._manga_drop_local_paths(event.mimeData())
                    self._set_manga_drop_highlight(False)
                    if paths and getattr(self, '_file_selection_editing_enabled', True):
                        event.acceptProposedAction()
                        self._add_dropped_manga_paths(paths)
                        return True
            if hasattr(self, 'log_text') and obj in (self.log_text, self.log_text.viewport()):
                if event.type() in (QEvent.Resize, QEvent.Show):
                    QTimer.singleShot(0, self._position_log_scroll_button)
        except Exception:
            pass
        return False
    
    def _should_autoscroll(self) -> bool:
        """Return True if we should auto-scroll to bottom based on current scrollbar position and flags."""
        try:
            if getattr(self, '_log_auto_scroll_disabled', False):
                return False
            if not hasattr(self, 'log_text') or not self.log_text:
                return True
            sb = self.log_text.verticalScrollBar()
            # If delay not elapsed, don't autoscroll
            import time as _time
            if _time.time() < getattr(self, '_autoscroll_delay_until', 0):
                return False
            # If user is near bottom, allow auto-scroll even if they briefly scrolled
            distance = max(0, sb.maximum() - sb.value())
            near_threshold = max(20, int(sb.pageStep() * 0.9))
            if distance <= near_threshold:
                # Reset manual scroll flag when we detect we're near bottom again
                self._user_scrolled_up = False
                return True
            # Otherwise, respect explicit manual-scroll flag
            return not getattr(self, '_user_scrolled_up', False)
        except Exception:
            return True
    
    def _start_autoscroll_delay(self, ms=500):
        """Delay auto-scroll for the specified milliseconds"""
        try:
            import time as _time
            self._autoscroll_delay_until = _time.time() + (ms / 1000.0)
            # Reset manual scroll flag when starting new operation
            self._user_scrolled_up = False
        except Exception:
            self._autoscroll_delay_until = 0.0

    def _schedule_log_autoscroll(self, scrollbar):
        """Coalesce log autoscroll timers so heavy logging does not flood the UI."""
        try:
            if getattr(self, '_log_autoscroll_pending', False):
                return
            self._log_autoscroll_pending = True

            def scroll_once():
                try:
                    if self._should_autoscroll() and scrollbar:
                        scrollbar.setValue(scrollbar.maximum())
                finally:
                    self._log_autoscroll_pending = False

            QTimer.singleShot(50, scroll_once)
        except Exception:
            self._log_autoscroll_pending = False

    def _debug_logging_enabled(self) -> bool:
        try:
            config = getattr(getattr(self, 'main_gui', None), 'config', {}) or {}
            manga_settings = config.get('manga_settings', {}) if isinstance(config.get('manga_settings', {}), dict) else {}
            manga_advanced = manga_settings.get('advanced', {}) if isinstance(manga_settings.get('advanced', {}), dict) else {}
            return bool(
                config.get('show_debug_buttons', False)
                or manga_advanced.get('debug_mode', False)
                or os.environ.get('DEBUG_MODE') == '1'
                or os.environ.get('SHOW_DEBUG_BUTTONS') == '1'
                or os.environ.get('MANGA_DEBUG_MODE') == '1'
            )
        except Exception:
            return os.environ.get('DEBUG_MODE') == '1'

    def _should_suppress_debug_log(self, message: str, level: str = "info") -> bool:
        if self._debug_logging_enabled():
            return False

        text = str(message or '').strip()
        if not text:
            return False

        if str(level or '').lower() == 'debug':
            return True

        lower = text.lower()
        if lower.startswith('debug:') or lower.startswith('[debug]'):
            return True

        noisy_prefixes = (
            '[STATE DEBUG]',
            '[STATE_ISOLATION]',
            '[FILE_PERSIST]',
            '[FILE_SELECTION]',
            '[SYNC_SELECTION]',
            '[SRC]',
            '[LOADED]',
            '[STATE_CLEAN]',
            '[STATE]',
            '[RECT_0_DEBUG]',
            '[EXCLUDE_RESTORE]',
            '[ITERATIONS_RESTORE]',
            '[PRELOAD_DETECTOR]',
            '[PRELOAD_INPAINTER]',
            '[PRELOAD_CHECK]',
            '[MANGA_CLOSE]',
            '[OCR_PROMPT_LOAD]',
            '[OCR_PROMPT_SAVE]',
            '[BUTTON_STATE]',
            '[POOL_TRACKER]',
            '[PREVIEW_UPDATE]',
            '[LOAD_IMAGE]',
            '[DISPLAY_MODE]',
            '[QUEUE]',
            '[BATCH_SYNC]',
            '[FILE_REORDER]',
            '[FILE_LIST]',
            '[CLEANED]',
            '[TRANSLATE_THIS_TEXT]',
            '[TRANSLATE_RESULT]',
            '[START_TRANSLATION]',
            '[START_HEAVY]',
            '[INIT_PRELOAD]',
            '[TOGGLE_PRELOAD]',
            '[INPAINT_LOAD]',
        )
        if text.startswith(noisy_prefixes):
            important_terms = (
                'error',
                'failed',
                'failure',
                'exception',
                'traceback',
                'critical',
                'warning',
            )
            return not any(term in lower for term in important_terms)

        noisy_exact_prefixes = (
            'Initialized OCR Manager for ',
            'Model preloading already in progress',
        )
        if text.startswith(noisy_exact_prefixes):
            return True

        if text and set(text) == {'='}:
            return True

        return False
    
    def _log(self, message: str, level: str = "info"):
        """Log messages to the manga panel regardless of translation stop state."""
        if self._should_suppress_debug_log(message, level):
            return
        
        # Lightweight deduplication: ignore identical lines within a short interval
        try:
            now = time.time()
            last_msg = getattr(self, '_last_log_msg', None)
            last_ts = getattr(self, '_last_log_time', 0)
            if last_msg == message and (now - last_ts) < 0.7:
                return
        except Exception:
            pass
        
        # Store in persistent log (thread-safe)
        try:
            with MangaTranslationTab._persistent_log_lock:
                # Keep only last 1000 messages to avoid unbounded growth
                if len(MangaTranslationTab._persistent_log) >= 1000:
                    MangaTranslationTab._persistent_log.pop(0)
                MangaTranslationTab._persistent_log.append((message, level))
        except Exception:
            pass
            
        # Check if log_text widget exists yet
        if hasattr(self, 'log_text') and self.log_text:
            # Thread-safe logging to GUI
            if threading.current_thread() == threading.main_thread():
                # We're in the main thread, update directly
                try:
                    # PySide6 QTextEdit - append with color
                    color_map = {
                        'info': 'white',
                        'success': 'green',
                        'warning': 'orange',
                        'error': 'red',
                        'debug': 'lightblue'
                    }
                    color = color_map.get(level, 'white')
                    # Use textCursor for more compact logging (no extra spacing)
                    from PySide6.QtGui import QTextCursor, QTextCharFormat
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    
                    # Set color format BEFORE inserting text
                    format = QTextCharFormat()
                    format.setForeground(QColor(color))
                    
                    # Add newline if not first message
                    if not cursor.atStart():
                        cursor.insertText("\n")
                    
                    cursor.insertText(message, format)
                    
                    # Auto-scroll behavior: favor auto-scroll when near bottom, respect manual scroll otherwise
                    try:
                        from PySide6.QtGui import QTextCursor as _QtTextCursor
                        from PySide6.QtCore import QTimer
                        
                        # Move cursor to end (where we inserted) so ensureCursorVisible works when allowed
                        self.log_text.moveCursor(_QtTextCursor.End)
                        
                        if self._should_autoscroll():
                            self.log_text.ensureCursorVisible()
                            scrollbar = self.log_text.verticalScrollBar()
                            scrollbar.setValue(scrollbar.maximum())
                            self._schedule_log_autoscroll(scrollbar)
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                # We're in a background thread, use queue
                self.update_queue.put(('log', message, level))
        else:
            # Widget doesn't exist yet or we're in initialization, print to console
            print(message)
        
        # Update deduplication state
        try:
            self._last_log_msg = message
            self._last_log_time = time.time()
        except Exception:
            pass
    
    def _start_startup_heartbeat(self):
        """Show a small spinner in the progress label during startup so there is no silence."""
        try:
            self._startup_heartbeat_running = True
            self._heartbeat_idx = 0
            chars = ['|', '/', '-', '\\']
            def tick():
                if not getattr(self, '_startup_heartbeat_running', False):
                    return
                try:
                    c = chars[self._heartbeat_idx % len(chars)]
                    if hasattr(self, 'progress_label'):
                        self.progress_label.setText(f"Starting… {c}")
                        self.progress_label.setStyleSheet("color: white;")
                except Exception:
                    pass
                self._heartbeat_idx += 1
                # Schedule next tick with QTimer - only if still running
                if getattr(self, '_startup_heartbeat_running', False):
                    QTimer.singleShot(250, tick)
            # Kick off
            QTimer.singleShot(0, tick)
        except Exception:
            pass

    def _apply_completed_translation_preview(self, data) -> bool:
        """Show a completed manga output in both preview surfaces."""
        if isinstance(data, dict):
            translated_path = data.get('translated_path')
            source_path = data.get('source_path')
            switch_to_output = bool(data.get('switch_to_output', True))
        else:
            translated_path = data
            source_path = None
            switch_to_output = True

        preview = getattr(self, 'image_preview_widget', None)
        if preview is None or not translated_path or not os.path.isfile(translated_path):
            return False

        current_path = getattr(preview, 'current_image_path', None)

        def _same_path(left, right):
            if not left or not right:
                return False
            try:
                return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
            except Exception:
                return left == right

        # Only the visible source page should take over the preview. Every
        # producer now supplies source_path, but retain compatibility with old
        # string queue entries by treating the current page as their source.
        if source_path and not (
            _same_path(current_path, source_path)
            or _same_path(current_path, translated_path)
        ):
            return False
        source_path = source_path or current_path
        if not source_path:
            return False

        preview.current_translated_path = translated_path
        if switch_to_output:
            preview.source_display_mode = 'translated'
            preview.cleaned_images_enabled = True
            toggle = getattr(preview, 'cleaned_toggle_btn', None)
            if toggle is not None:
                toggle.setText("✒️")
                toggle.setToolTip("Showing translated output (click to cycle)")

        if not hasattr(self, '_rendered_images_map'):
            self._rendered_images_map = {}
        self._rendered_images_map[source_path] = translated_path

        state_manager = getattr(self, 'image_state_manager', None)
        if state_manager is not None:
            try:
                state_manager.update_state(source_path, {
                    'rendered_image_path': translated_path,
                })
            except Exception as state_error:
                self._log(f"Could not persist translated preview path: {state_error}", "warning")

        output_viewer = getattr(preview, 'output_viewer', None)
        if output_viewer is not None:
            output_viewer.load_image(translated_path)
        preview.load_image(
            source_path,
            preserve_rectangles=True,
            preserve_text_overlays=True,
        )
        return True
    
    def _process_updates(self):
        """Process queued GUI updates"""
        processed = 0
        max_updates_per_tick = 80
        try:
            while processed < max_updates_per_tick:
                update = self.update_queue.get_nowait()
                processed += 1
                
                if update[0] == 'log':
                    _, message, level = update
                    try:
                        # PySide6 QTextEdit
                        color_map = {
                            'info': 'white',
                            'success': 'green',
                            'warning': 'orange',
                            'error': 'red',
                            'debug': 'lightblue'
                        }
                        color = color_map.get(level, 'white')
                        # Use textCursor for more compact logging (no extra spacing)
                        from PySide6.QtGui import QTextCursor, QTextCharFormat
                        cursor = self.log_text.textCursor()
                        cursor.movePosition(QTextCursor.End)
                        
                        # Set color format BEFORE inserting text
                        format = QTextCharFormat()
                        format.setForeground(QColor(color))
                        
                        # Add newline if not first message
                        if not cursor.atStart():
                            cursor.insertText("\n")
                        
                        cursor.insertText(message, format)
                        
                        # AGGRESSIVE: Scroll to bottom (favor auto-scroll when near bottom)
                        try:
                            if self._should_autoscroll():
                                self.log_text.ensureCursorVisible()
                                scrollbar = self.log_text.verticalScrollBar()
                                scrollbar.setValue(scrollbar.maximum())
                                self._schedule_log_autoscroll(scrollbar)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    
                elif update[0] == 'progress':
                    _, current, total, status = update
                    if total > 0:
                        percentage = (current / total) * 100
                        self.progress_bar.setValue(int(percentage))
                    
                    # Check if this is a stopped status and style accordingly
                    if "stopped" in status.lower() or "cancelled" in status.lower():
                        # Make the status more prominent for stopped translations
                        self.progress_label.setText(f"⏹️ {status}")
                        self.progress_label.setStyleSheet("color: orange;")
                    elif "complete" in status.lower() or "finished" in status.lower():
                        # Success status
                        self.progress_label.setText(f"✅ {status}")
                        self.progress_label.setStyleSheet("color: green;")
                    elif "error" in status.lower() or "failed" in status.lower():
                        # Error status
                        self.progress_label.setText(f"❌ {status}")
                        self.progress_label.setStyleSheet("color: red;")
                    else:
                        # Normal status - white for dark mode
                        self.progress_label.setText(status)
                        self.progress_label.setStyleSheet("color: white;")
                    
                elif update[0] == 'current_file':
                    _, filename = update
                    # Style the current file display based on the status
                    if "stopped" in filename.lower() or "cancelled" in filename.lower():
                        self.current_file_label.setText(f"⏹️ {filename}")
                        self.current_file_label.setStyleSheet("color: orange;")
                    elif "complete" in filename.lower() or "finished" in filename.lower():
                        self.current_file_label.setText(f"✅ {filename}")
                        self.current_file_label.setStyleSheet("color: green;")
                    elif "error" in filename.lower() or "failed" in filename.lower():
                        self.current_file_label.setText(f"❌ {filename}")
                        self.current_file_label.setStyleSheet("color: red;")
                    else:
                        self.current_file_label.setText(f"Current: {filename}")
                        self.current_file_label.setStyleSheet("color: lightgray;")
                
                elif update[0] == 'ui_state':
                    state = update[1]
                    state_token = update[2] if len(update) > 2 else None
                    if not _translation_run_token_matches(
                        getattr(self, '_translation_start_token', None),
                        state_token,
                    ):
                        # A stopped/completed worker can finish after the user
                        # has already started another run.  Its UI event belongs
                        # to the old run and must not reset the new one.
                        continue
                    if state == 'translation_started':
                        try:
                            # Keep the file list enabled so its scrollbar remains usable during long runs.
                            # Only disable file mutations and drag reordering while translation is active.
                            self._set_file_selection_editing_enabled(False)
                        except Exception:
                            pass
                    elif state == 'translation_complete':
                        try:
                            # Reset UI to ready state when translation completes
                            self._reset_ui_state(state_token)
                            # REMOVED: Don't auto-switch tabs - let user manually switch
                            # try:
                            #     if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'viewer_tabs'):
                            #         self.image_preview_widget.viewer_tabs.setCurrentIndex(1)
                            # except Exception:
                            #     pass
                        except Exception as e:
                            import traceback
                            print(f"Error resetting UI state: {e}")
                            print(traceback.format_exc())
                
                elif update[0] == 'call_method':
                    # Call a method on the main thread
                    _, method, args = update
                    try:
                        method(*args)
                    except Exception as e:
                        import traceback
                        print(f"Error calling method {method}: {e}")
                        print(traceback.format_exc())

                elif update[0] == 'model_file_status':
                    _, status = update
                    if hasattr(self, 'local_model_status_label'):
                        self.local_model_status_label.setText(status)
                        if "Downloading" in status:
                            self.local_model_status_label.setStyleSheet("color: #4a9eff;")  # Blue
                        elif "error" in status.lower() or "failed" in status.lower():
                            self.local_model_status_label.setStyleSheet("color: red;")
                        elif "complete" in status.lower() or "ready" in status.lower():
                            self.local_model_status_label.setStyleSheet("color: green;")
                        else:
                            self.local_model_status_label.setStyleSheet("color: gray;")
                
                elif update[0] == 'clean_preview_update':
                    # Legacy branch not used; we update output via 'preview_update'
                    pass
                
                # No in-memory overlay branch needed now
                
                elif update[0] == 'clean_button_restore':
                    # Restore the clean button to its normal state
                    try:
                        ImageRenderer._restore_clean_button(self, )
                    except Exception as e:
                        self._log(f"❌ Failed to restore clean button: {str(e)}", "error")
                
                elif update[0] == 'single_clean_complete':
                    # Handle single rectangle clean completion
                    _, data = update
                    try:
                        region_index = data['region_index']
                        result_image = data['result_image']
                        original_path = data['original_path']
                        
                        # STATE ISOLATION: Only stop pulse effect if operation is for currently displayed image
                        current_img = getattr(self.image_preview_widget, 'current_image_path', None) if hasattr(self, 'image_preview_widget') else None
                        if current_img and original_path:
                            try:
                                import os
                                if os.path.abspath(original_path) != os.path.abspath(current_img):
                                    print(f"[STATE_ISOLATION] Clean complete for different image - skipping pulse stop")
                                    return
                            except Exception:
                                if original_path != current_img:
                                    return
                        
                        # Stop pulse effect on the rectangle
                        try:
                            if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
                                rectangles = self.image_preview_widget.viewer.rectangles
                                if 0 <= region_index < len(rectangles):
                                    rect_item = rectangles[region_index]
                                    ImageRenderer._remove_rectangle_pulse_effect(self, rect_item, region_index)
                        except Exception as pulse_err:
                            print(f"[CLEAN_COMPLETE] Error stopping pulse effect: {pulse_err}")
                        
                        # Update the image preview with the result
                        ImageRenderer._update_image_preview_with_result(self, result_image, original_path)
                        self._log(f"✅ Successfully cleaned rectangle {region_index}", "success")
                        
                    except Exception as e:
                        self._log(f"❌ Failed to handle single clean result: {str(e)}", "error")
                
                elif update[0] == 'single_clean_error':
                    # Handle single rectangle clean error
                    _, data = update
                    try:
                        region_index = data['region_index']
                        error = data['error']
                        original_path = data.get('original_path')  # May not always be present
                        
                        # STATE ISOLATION: Only stop pulse effect if operation is for currently displayed image
                        if original_path:
                            current_img = getattr(self.image_preview_widget, 'current_image_path', None) if hasattr(self, 'image_preview_widget') else None
                            if current_img:
                                try:
                                    import os
                                    if os.path.abspath(original_path) != os.path.abspath(current_img):
                                        print(f"[STATE_ISOLATION] Clean error for different image - skipping pulse stop")
                                        return
                                except Exception:
                                    if original_path != current_img:
                                        return
                        
                        # Stop pulse effect on the rectangle
                        try:
                            if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
                                rectangles = self.image_preview_widget.viewer.rectangles
                                if 0 <= region_index < len(rectangles):
                                    rect_item = rectangles[region_index]
                                    ImageRenderer._remove_rectangle_pulse_effect(self, rect_item, region_index)
                        except Exception as pulse_err:
                            print(f"[CLEAN_ERROR] Error stopping pulse effect: {pulse_err}")
                        
                        self._log(f"❌ Rectangle {region_index} cleaning failed: {error}", "error")
                        
                    except Exception as e:
                        self._log(f"❌ Failed to handle single clean error: {str(e)}", "error")
                
                elif update[0] == 'detect_results':
                    _, results = update
                    # Process detection results and update preview
                    try:
                        ImageRenderer._process_detect_results(self, results)
                    except Exception as e:
                        self._log(f"❌ Failed to process detection results: {str(e)}", "error")
                
                elif update[0] == 'detect_button_restore':
                    # Restore the detect button to its normal state
                    try:
                        ImageRenderer._restore_detect_button(self)
                    except Exception as e:
                        self._log(f"❌ Failed to restore detect button: {str(e)}", "error")
                
                elif update[0] == 'update_pool_tracker':
                    # Immediately update pool tracker label
                    try:
                        self._update_pool_tracker_label()
                    except Exception as e:
                        print(f"[POOL_TRACKER] Failed to update: {e}")
                
                elif update[0] == 'parallel_button_state':
                    # Update parallel save button on main thread (marshalled from background)
                    _, active_count = update
                    try:
                        ImageRenderer._update_parallel_save_button_state(self, active_count)
                    except Exception as e:
                        print(f"[PARALLEL] Failed to update button state: {e}")
                
                elif update[0] == 'parallel_gui_update':
                    # Execute parallel save GUI update on main thread (marshalled from ThreadPoolExecutor)
                    _, data = update
                    try:
                        region_index = data['region_index']
                        trans_text = data['trans_text']
                        ImageRenderer._update_single_text_overlay(self, region_index, trans_text)
                    except Exception as e:
                        print(f"[PARALLEL] GUI update failed for region {data.get('region_index')}: {e}")
                
                elif update[0] == 'recognize_results':
                    _, results = update
                    # Process recognition results
                    try:
                        ImageRenderer._process_recognize_results(self, results)
                    except Exception as e:
                        self._log(f"❌ Failed to process recognition results: {str(e)}", "error")
                
                elif update[0] == 'recognize_button_restore':
                    # Restore the recognize button to its normal state
                    try:
                        ImageRenderer._restore_recognize_button(self, )
                    except Exception as e:
                        self._log(f"❌ Failed to restore recognize button: {str(e)}", "error")
                
                elif update[0] == 'translate_results':
                    _, results = update
                    # Process translation results
                    try:
                        ImageRenderer._process_translate_results(self, results)
                    except Exception as e:
                        self._log(f"❌ Failed to process translation results: {str(e)}", "error")
                
                elif update[0] == 'translate_button_restore':
                    # Restore the translate button to its normal state
                    try:
                        ImageRenderer._restore_translate_button(self, )
                    except Exception as e:
                        self._log(f"❌ Failed to restore translate button: {str(e)}", "error")
                
                elif update[0] == 'preview_update':
                    # Update preview to show translated/cleaned image (thread-safe)
                    _, data = update
                    try:
                        self._apply_completed_translation_preview(data)
                    except Exception as e:
                        print(f"[PREVIEW_UPDATE] ❌ Error updating preview: {e}")
                        import traceback
                        print(traceback.format_exc())
                
                elif update[0] == 'translate_all_progress':
                    # Update translate all button progress
                    _, progress_data = update
                    try:
                        # Don't update progress if stop was clicked (preserve "Stopping..." text)
                        should_skip = False
                        if hasattr(self, 'stop_flag') and self.stop_flag and self.stop_flag.is_set():
                            print(f"[PROGRESS] Skipping progress update - stop flag is set")
                            should_skip = True
                        if hasattr(self, '_global_cancellation') and self._global_cancellation:
                            print(f"[PROGRESS] Skipping progress update - global cancellation set")
                            should_skip = True
                        
                        if not should_skip:
                            current = progress_data['current']
                            total = progress_data['total']
                            if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'translate_all_btn'):
                                self.image_preview_widget.translate_all_btn.setText(f"Translating... ({current}/{total})")
                    except Exception as e:
                        print(f"Error updating translate all progress: {str(e)}")
                
                elif update[0] == 'translate_this_text_result':
                    # Process translation result on the GUI thread and add/replace a single overlay
                    _, data = update
                    try:
                        region_index = int(data.get('region_index'))
                        original_text = data.get('original_text', '')
                        translation_result = data.get('translation_result', '')
                        bbox = data.get('bbox') or [0, 0, 100, 100]
                        
                        # Stop pulse effect on the rectangle
                        try:
                            if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
                                rectangles = self.image_preview_widget.viewer.rectangles
                                if 0 <= region_index < len(rectangles):
                                    rect_item = rectangles[region_index]
                                    ImageRenderer._remove_rectangle_pulse_effect(self, rect_item, region_index)
                        except Exception as pulse_err:
                            print(f"[TRANSLATE_RESULT] Error stopping pulse effect: {pulse_err}")
                        
                        # Ensure in-memory translation map is updated
                        if not hasattr(self, '_translation_data'):
                            self._translation_data = {}
                        self._translation_data[region_index] = {
                            'original': original_text,
                            'translation': translation_result
                        }
                        # Add or replace overlay for just this region
                        try:
                            ImageRenderer._add_text_overlay_for_region(self, region_index, original_text, translation_result, bbox)
                            #self._log(f"✅ Added text overlay for region {region_index}", "success")
                        except Exception as overlay_err:
                            self._log(f"⚠️ Failed to add overlay: {overlay_err}", "warning")
                        
                        # Persist translated_texts entry for this image so overlays restore next session
                        try:
                            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                            if current_image and hasattr(self, 'image_state_manager') and self.image_state_manager:
                                state = self.image_state_manager.get_state(current_image) or {}
                                tlist = state.get('translated_texts') or []
                                # Ensure list is large enough
                                if len(tlist) <= region_index:
                                    tlist = list(tlist) + [{} for _ in range(region_index + 1 - len(tlist))]
                                tlist[region_index] = {
                                    'original': {'text': original_text, 'region_index': region_index},
                                    'translation': translation_result,
                                    'bbox': bbox or [0, 0, 100, 100]
                                }
                                state['translated_texts'] = tlist
                                self.image_state_manager.set_state(current_image, state, save=True)
                        except Exception as persist_err:
                            self._log(f"⚠️ Failed to persist overlay text: {persist_err}", "warning")
                        
                        # Render to actual output file (same as main Translate button)
                        # This ensures the translated text is saved to the output image file
                        try:
                            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                            if current_image and hasattr(self, '_translation_data') and self._translation_data:
                                print(f"[TRANSLATE_THIS_TEXT] Triggering render to output file for region {region_index}")
                                self._log(f"🎨 Rendering translation to output file...", "info")
                                
                                # Use the same async overlay rendering as Save & Update Overlay
                                # This will render all translations (including this new one) to the output file
                                try:
                                    ImageRenderer._save_overlay_async(self, region_index, translation_result)
                                except Exception as render_err:
                                    print(f"[TRANSLATE_THIS_TEXT] Render error: {render_err}")
                                    self._log(f"⚠️ Failed to render to output file: {render_err}", "warning")
                                
                                # Switch display mode to 'translated' so user sees the result
                                try:
                                    ipw = self.image_preview_widget
                                    ipw.source_display_mode = 'translated'
                                    ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
                                    
                                    # Update the toggle button appearance to match 'translated' state
                                    if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                                        ipw.cleaned_toggle_btn.setText("✒️")  # Pen for translated output
                                        ipw.cleaned_toggle_btn.setToolTip("Showing translated output (click to cycle)")
                                        ipw.cleaned_toggle_btn.setStyleSheet("""
                                            QToolButton {
                                                background-color: #28a745;
                                                border: 2px solid #34ce57;
                                                font-size: 12pt;
                                                min-width: 32px;
                                                min-height: 32px;
                                                max-width: 36px;
                                                max-height: 36px;
                                                padding: 3px;
                                                border-radius: 3px;
                                                color: white;
                                            }
                                            QToolButton:hover {
                                                background-color: #34ce57;
                                            }
                                        """)
                                    print(f"[TRANSLATE_THIS_TEXT] Switched display mode to 'translated'")
                                except Exception as mode_err:
                                    print(f"[TRANSLATE_THIS_TEXT] Failed to switch display mode: {mode_err}")
                        except Exception as file_render_err:
                            print(f"[TRANSLATE_THIS_TEXT] File render setup error: {file_render_err}")
                            self._log(f"⚠️ Failed to setup file rendering: {file_render_err}", "warning")
                    except Exception as e:
                        import traceback
                        self._log(f"❌ Error handling translate result: {e}", "error")
                        print(traceback.format_exc())
                
                elif update[0] == 'load_preview_image':
                    # Load an image in the preview
                    _, data = update
                    try:
                        import os
                        if hasattr(self, 'image_preview_widget'):
                            # Handle both string (image_path) and dict (with options)
                            if isinstance(data, dict):
                                image_path = data.get('path')
                                preserve_rectangles = data.get('preserve_rectangles', False)
                                preserve_overlays = data.get('preserve_overlays', False)
                            else:
                                image_path = data
                                preserve_rectangles = False
                                preserve_overlays = False
                            
                            # Gate: only allow loading into the preview if this path matches the current selection
                            # EXCEPTION: During batch mode, allow any image to be loaded so user can see progress
                            should_skip_load = False
                            is_batch_mode = getattr(self, '_batch_mode_active', False)
                            if not is_batch_mode:
                                try:
                                    current_selected = None
                                    if hasattr(self, 'file_listbox') and self.file_listbox and self.file_listbox.currentRow() >= 0:
                                        idx = self.file_listbox.currentRow()
                                        if 0 <= idx < len(self.selected_files):
                                            current_selected = self.selected_files[idx]
                                    current_loaded = getattr(self.image_preview_widget, 'current_image_path', None)
                                    if image_path and current_selected and os.path.normcase(os.path.normpath(image_path)) != os.path.normcase(os.path.normpath(current_selected)):
                                        # Ignore background updates for non-current images
                                        print(f"[LOAD_IMAGE] Skipping non-current image update: {os.path.basename(image_path)}")
                                        should_skip_load = True
                                    # Also ignore if we're trying to replace a different image than currently loaded
                                    if not should_skip_load and image_path and current_loaded and os.path.normcase(os.path.normpath(image_path)) != os.path.normcase(os.path.normpath(current_loaded)):
                                        print(f"[LOAD_IMAGE] Skipping update that doesn't match current loaded image")
                                        should_skip_load = True
                                except Exception:
                                    should_skip_load = False  # Don't skip on error
                            else:
                                print(f"[LOAD_IMAGE] Batch mode active - bypassing gate for: {os.path.basename(image_path)}")
                            
                            # Only proceed if not skipping
                            if not should_skip_load:
                                # Store original path for state restoration
                                original_image_path = image_path
                                
                                # Determine if a translated image exists for the OUTPUT tab
                                translated_image_path = None
                                filename = os.path.basename(image_path)
                                base_name = os.path.splitext(filename)[0]
                                parent_dir = os.path.dirname(image_path)
                                isolated_folder = os.path.join(parent_dir, f"{base_name}_translated")
                                isolated_image = os.path.join(isolated_folder, filename)
                                
                                if os.path.exists(isolated_image):
                                    translated_image_path = isolated_image
                                    print(f"[LOAD_IMAGE] Found translated image in isolated folder: {os.path.basename(isolated_image)}")
                                else:
                                    # Check state manager for rendered image
                                    if hasattr(self, 'image_state_manager'):
                                        state = self.image_state_manager.get_state(original_image_path)
                                        if state and 'rendered_image_path' in state and os.path.exists(state['rendered_image_path']):
                                            translated_image_path = state['rendered_image_path']
                                            print(f"[LOAD_IMAGE] Found rendered image from state: {os.path.basename(translated_image_path)}")
                                    # Check _rendered_images_map
                                    if translated_image_path is None and hasattr(self, '_rendered_images_map') and original_image_path in self._rendered_images_map:
                                        mapped_path = self._rendered_images_map[original_image_path]
                                        if os.path.exists(mapped_path):
                                            translated_image_path = mapped_path
                                            print(f"[LOAD_IMAGE] Found rendered image from map: {os.path.basename(mapped_path)}")
                                
                                # Load the SOURCE into the source viewer (no batch gating)
                                self.image_preview_widget.load_image(original_image_path, 
                                                                    preserve_rectangles=preserve_rectangles,
                                                                    preserve_text_overlays=preserve_overlays)
                                
                                # Store translated path if available
                                if translated_image_path:
                                    self.image_preview_widget.current_translated_path = translated_image_path
                                
                                # Update current image path for state tracking
                                self._current_image_path = original_image_path
                    except Exception as e:
                        print(f"Error loading preview image: {str(e)}")
                
                elif update[0] == 'translate_all_button_restore':
                    # Restore the translate all button to its normal state
                    print(f"[QUEUE] Received translate_all_button_restore message")
                    try:
                        ImageRenderer._restore_translate_all_button(self, )
                        print(f"[QUEUE] translate_all_button_restore completed successfully")
                    except Exception as e:
                        print(f"[QUEUE] translate_all_button_restore FAILED: {e}")
                        self._log(f"❌ Failed to restore translate all button: {str(e)}", "error")
                
                elif update[0] == 'remove_processing_overlay':
                    # Remove the blue pulse overlay
                    try:
                        ImageRenderer._remove_processing_overlay(self, )
                    except Exception as e:
                        print(f"Error removing processing overlay: {str(e)}")
                
                elif update[0] == 'add_processing_overlay':
                    # Add the blue pulse overlay (used to restore after image switch in batch mode)
                    try:
                        ImageRenderer._add_processing_overlay(self, )
                    except Exception as e:
                        print(f"Error adding processing overlay: {str(e)}")
                
                elif update[0] == 'sync_file_selection':
                    # Sync file list and thumbnail selection to match batch processing
                    _, data = update
                    try:
                        image_path = data.get('image_path') if isinstance(data, dict) else None
                        if image_path and hasattr(self, 'file_listbox') and hasattr(self, 'selected_files'):
                            # Find index of this image in selected_files
                            try:
                                index = self.selected_files.index(image_path)
                                # Keep the user's scroll position stable while still updating selection.
                                self._set_file_list_current_row_without_scroll(index)
                                print(f"[BATCH_SYNC] Synced file list selection to row {index}")
                                
                                # Also sync thumbnail selection
                                if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, '_update_thumbnail_selection'):
                                    self.image_preview_widget._update_thumbnail_selection(image_path)
                            except ValueError:
                                print(f"[BATCH_SYNC] Image not found in selected_files: {os.path.basename(image_path)}")
                    except Exception as e:
                        print(f"Error syncing file selection: {str(e)}")
                
                elif update[0] == 'update_preview_to_rendered':
                    # Update the preview to show all rendered images
                    try:
                        ImageRenderer._update_preview_to_rendered_images(self, )
                    except Exception as e:
                        self._log(f"❌ Failed to update preview to rendered images: {str(e)}", "error")
                
                elif update[0] == 'switch_to_translated_mode':
                    # Switch display mode to 'translated' and refresh preview
                    _, data = update
                    try:
                        image_path = data.get('image_path') if isinstance(data, dict) else None
                        ipw = self.image_preview_widget if hasattr(self, 'image_preview_widget') else None
                        if ipw:
                            ipw.source_display_mode = 'translated'
                            ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
                            
                            # Update the toggle button appearance to match 'translated' state
                            if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                                ipw.cleaned_toggle_btn.setText("✒️")  # Pen for translated output
                                ipw.cleaned_toggle_btn.setToolTip("Showing translated output (click to cycle)")
                                ipw.cleaned_toggle_btn.setStyleSheet("""
                                    QToolButton {
                                        background-color: #28a745;
                                        border: 2px solid #34ce57;
                                        font-size: 12pt;
                                        min-width: 32px;
                                        min-height: 32px;
                                        max-width: 36px;
                                        max-height: 36px;
                                        padding: 3px;
                                        border-radius: 3px;
                                        color: white;
                                    }
                                    QToolButton:hover {
                                        background-color: #34ce57;
                                    }
                                """)
                            
                            # Refresh preview to show translated output
                            if image_path:
                                ipw.load_image(image_path, preserve_rectangles=True, preserve_text_overlays=True)
                            print(f"[DISPLAY_MODE] Switched to 'translated' mode")
                    except Exception as e:
                        print(f"[DISPLAY_MODE] Error switching mode: {e}")
                
                elif update[0] == 'set_translated_folder':
                    # Set translated folder for preview mode and download button
                    _, folder_path = update
                    try:
                        if hasattr(self, 'image_preview_widget'):
                            self.image_preview_widget.set_translated_folder(folder_path)
                            # Only log if manual editing is disabled (preview mode active)
                            self._log(f"✅ Preview mode updated with translated images", "success")
                    except Exception as e:
                        self._log(f"❌ Failed to set translated folder: {str(e)}", "error")
                    
        except Empty:
            # Queue is empty.
            pass
        except Exception:
            pass
        finally:
            # Keep large update bursts from monopolizing the event loop.
            try:
                delay = 10 if not self.update_queue.empty() else 100
            except Exception:
                delay = 100
            QTimer.singleShot(delay, self._process_updates)

    # Periodic demoter to keep UI responsive by lowering new worker thread priorities (Windows-only).
    def _start_periodic_thread_demoter(self):
        try:
            if not _IS_WINDOWS:
                return
            self._demoter_active = True
            def _tick():
                try:
                    if not getattr(self, '_demoter_active', False):
                        return
                    if getattr(self, 'is_running', False) is False:
                        return
                    if getattr(self, '_main_thread_tid', None):
                        _demote_non_main_threads(self._main_thread_tid, 'MANGA_RESERVE_CORES')
                except Exception:
                    pass
                finally:
                    # Schedule next demotion pass
                    QTimer.singleShot(750, _tick)
            QTimer.singleShot(0, _tick)
        except Exception:
            pass

    def _stop_periodic_thread_demoter(self):
        try:
            self._demoter_active = False
        except Exception:
            pass

    def load_local_inpainting_model(self, model_path):
        """Load a local inpainting model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if successful
        """
        try:
            # Store the model path
            self.local_inpaint_model_path = model_path
            
            # If using diffusers/torch models, load them here
            if model_path.endswith('.safetensors') or model_path.endswith('.ckpt'):
                # Initialize your inpainting pipeline
                # This depends on your specific inpainting implementation
                # Example:
                # from diffusers import StableDiffusionInpaintPipeline
                # self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_single_file(model_path)
                pass
                
            return True
        except Exception as e:
            self._log(f"Failed to load inpainting model: {e}", "error")
            return False
            
    def _toggle_translation(self):
        """Toggle between start and stop translation"""
        if (
            self.is_running
            or getattr(self, '_graceful_stop_pending', False)
            or getattr(self, '_translation_startup_pending', False)
        ):
            self._stop_translation()
        else:
            self._start_translation()
    
    def _initialize_translator(self):
        """Initialize the translator if not already initialized"""
        if hasattr(self, 'translator') and self.translator is not None:
            self._log("✅ Translator already initialized", "debug")
            return True
        
        try:
            self._log("🔧 Initializing manga translator...", "info")
            
            # Get manga settings — apply defaults on first launch (no config.json)
            manga_settings = self.main_gui.config.get('manga_settings', {})
            if not manga_settings or 'ocr' not in manga_settings:
                # First launch: ensure critical OCR defaults match MangaSettingsDialog
                default_ocr = {
                    'bubble_detection_enabled': True,
                    'detector_type': 'rtdetr_onnx',
                    'rtdetr_onnx_variant': 'detector.onnx',
                    'rtdetr_confidence': 0.3,
                    'detect_empty_bubbles': True,
                    'detect_text_bubbles': True,
                    'detect_free_text': True,
                    'use_rtdetr_for_ocr_regions': True,
                }
                if 'manga_settings' not in self.main_gui.config:
                    self.main_gui.config['manga_settings'] = {}
                ms = self.main_gui.config['manga_settings']
                if 'ocr' not in ms:
                    ms['ocr'] = {}
                for k, v in default_ocr.items():
                    ms['ocr'].setdefault(k, v)
                manga_settings = ms
                self._log("📋 Applied default manga OCR settings (first launch)", "info")
            ocr_settings = manga_settings.get('ocr', {})
            
            # Build OCR config
            ocr_config = {'provider': self.ocr_provider_value}
            
            if ocr_config['provider'] == 'google':
                google_creds = self.main_gui.config.get('google_vision_credentials', '') or \
                              self.main_gui.config.get('google_cloud_credentials', '')
                if google_creds and os.path.exists(google_creds):
                    ocr_config['google_credentials_path'] = google_creds
                else:
                    self._log("❌ Google Cloud Vision credentials not found", "error")
                    return False
                    
            elif ocr_config['provider'] == 'azure':
                azure_key = self.main_gui.config.get('azure_vision_key', '')
                azure_endpoint = self.main_gui.config.get('azure_vision_endpoint', '')
                if azure_key and azure_endpoint:
                    ocr_config['azure_key'] = azure_key
                    ocr_config['azure_endpoint'] = azure_endpoint
                else:
                    self._log("❌ Azure credentials not configured", "error")
                    return False
            
            # Initialize unified client for translation (if not using custom API)
            if not hasattr(self.main_gui, 'client') or self.main_gui.client is None:
                # Try to get API key
                api_key = None
                if hasattr(self.main_gui, 'api_key_entry'):
                    try:
                        api_key = self.main_gui.api_key_entry.text().strip() if hasattr(self.main_gui.api_key_entry, 'text') else self.main_gui.api_key_entry.get().strip()
                    except:
                        pass
                if not api_key and hasattr(self.main_gui, 'config'):
                    api_key = self.main_gui.config.get('api_key', '')
                
                # Check if model uses own auth (no API key needed) — delegate to UnifiedClient
                if not api_key:
                    _model = self.main_gui.config.get('model', '') if hasattr(self.main_gui, 'config') else ''
                    try:
                        from unified_api_client import UnifiedClient as _UC
                        if not _UC._model_needs_api_key(_model):
                            api_key = 'own-auth'  # placeholder — actual auth handled by provider/local endpoint
                    except Exception:
                        pass
                
                if not api_key:
                    self._log("❌ API key not configured for translation", "error")
                    return False
                
                # Initialize UnifiedClient
                try:
                    if UnifiedClient:
                        self.main_gui.client = UnifiedClient(api_key=api_key)
                        self._log("🔌 Initialized translation client", "debug")
                except Exception as e:
                    self._log(f"❌ Failed to initialize translation client: {e}", "error")
                    return False
            
            # Create MangaTranslator
            from manga_translator import MangaTranslator
            self.translator = MangaTranslator(
                ocr_config=ocr_config,
                client=self.main_gui.client,
                main_gui=self.main_gui,
                log_callback=self._log
            )
            self._configure_manga_ocr_io(self.translator)
            
            # Set stop flag
            if hasattr(self, 'stop_flag'):
                self.translator.set_stop_flag(self.stop_flag)
            
            self._log("✅ Manga translator initialized successfully", "success")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to initialize translator: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
            return False
    
    def _start_translation(self):
        """Start the translation process"""
        # Check files BEFORE redirecting stdout to avoid deadlock
        if not self.selected_files:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self.dialog, "No Files", "Please select manga images to translate.")
            return

        processing_files, range_error = self._manga_range_filtered_files()
        if range_error:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self.dialog, "Invalid Image Range", range_error)
            return
        if not processing_files:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.dialog,
                "No Files In Range",
                "The image range does not include any loaded rows."
            )
            return
        self._manga_processing_files = None

        try:
            self._prepare_automatic_ocr_export(processing_files)
            if getattr(self, '_imported_ocr_document', None):
                matches = self._refresh_imported_ocr_page_map(processing_files)
                self._log(
                    f"Imported OCR will be reused for {len(matches)}/{len(processing_files)} pages",
                    "info",
                )
        except Exception as ocr_export_error:
            self._log(f"Could not initialize automatic OCR export: {ocr_export_error}", "warning")
        
        # Disable ALL workflow buttons to prevent concurrent operations
        # Note: show_stop_button=False because Start Translation has its own stop mechanism
        try:
            import ImageRenderer
            ImageRenderer._disable_workflow_buttons(self, exclude=None, show_stop_button=False)
        except Exception as e:
            print(f"[START_TRANSLATION] Error disabling workflow buttons: {e}")
        
        # Immediately update button to Stop state (red)
        try:
            if hasattr(self, 'start_button') and self.start_button:
                # Update text label instead of button text
                if hasattr(self, 'start_button_text'):
                    self.start_button_text.setText("⏹ Stop Translation")
                self.start_button.setStyleSheet(
                    "QPushButton { "
                    "  background-color: #dc3545; "
                    "  color: white; "
                    "  padding: 22px 30px; "
                    "  font-size: 14pt; "
                    "  font-weight: bold; "
                    "  border-radius: 8px; "
                    "} "
                    "QPushButton:hover { background-color: #c82333; } "
                    "QPushButton:disabled { "
                    "  background-color: #2d2d2d; "
                    "  color: #666666; "
                    "}"
                )
                self.start_button.setEnabled(True)
                # Start spinning animation immediately
                if hasattr(self, 'start_icon_spin_animation') and hasattr(self, 'start_button_icon'):
                    if self.start_icon_spin_animation.state() != QPropertyAnimation.Running:
                        self.start_icon_spin_animation.start()
                        # Start refresh timer to keep animation smooth
                        if hasattr(self, '_animation_refresh_timer'):
                            self._animation_refresh_timer.start()
        except Exception:
            pass
        
        # Delay auto-scroll so first log is readable
        self._start_autoscroll_delay(100)
        
        # Immediate minimal feedback using direct log append
        try:
            if hasattr(self, 'log_text') and self.log_text:
                from PySide6.QtGui import QColor, QTextCursor, QTextCharFormat
                # Use textCursor for more compact logging (no extra spacing)
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                
                # Set color format BEFORE inserting text
                format = QTextCharFormat()
                format.setForeground(QColor('white'))
                
                # Add newline if not first message
                if not cursor.atStart():
                    cursor.insertText("\n")
                
                cursor.insertText("Starting translation...", format)
                # Note: Auto-scroll delay just started above, so this will scroll
        except Exception:
            pass
        
        # Start heartbeat spinner so there's visible activity until logs stream
        self._start_startup_heartbeat()
        
        previous_future = getattr(self, 'translation_future', None)
        previous_thread = getattr(self, 'translation_thread', None)
        self._translation_start_token = int(getattr(self, '_translation_start_token', 0) or 0) + 1
        start_token = self._translation_start_token
        self._translation_startup_pending = True
        self._translation_start_cancel_requested = False

        # Mark the startup as active immediately so the same button can cancel it.
        # Stop/cancellation flags are cleared later in MangaStartHeavy, after any
        # stale worker has actually finished.
        self.is_running = True
        
        # Log start directly to GUI
        try:
            if hasattr(self, 'log_text') and self.log_text:
                from PySide6.QtGui import QColor, QTextCursor, QTextCharFormat
                from PySide6.QtCore import QTimer
                # Use textCursor for more compact logging (no extra spacing)
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                
                # Set color format BEFORE inserting text
                format = QTextCharFormat()
                format.setForeground(QColor('white'))
                
                # Add newline if not first message
                if not cursor.atStart():
                    cursor.insertText("\n")
                
                cursor.insertText("🚀 Starting new manga translation batch", format)
                # Note: Auto-scroll delay just started above, so initial scrolls will work
                
                # Scroll to bottom after a short delay to ensure it happens after button processing
                def scroll_to_bottom():
                    try:
                        if hasattr(self, 'log_text') and self.log_text:
                            # Only auto-scroll LOG if user hasn't manually scrolled up (respects delay)
                            if self._should_autoscroll():
                                self.log_text.moveCursor(QTextCursor.End)
                                self.log_text.ensureCursorVisible()
                            # Always scroll the entire GUI scroll area to bottom (no delay check)
                            if hasattr(self, 'scroll_area') and self.scroll_area:
                                scrollbar = self.scroll_area.verticalScrollBar()
                                if scrollbar:
                                    scrollbar.setValue(scrollbar.maximum())
                    except Exception:
                        pass
                
                # Schedule scroll with a small delay
                QTimer.singleShot(50, scroll_to_bottom)
                QTimer.singleShot(150, scroll_to_bottom)  # Second attempt to be sure
        except Exception:
            pass
        
        # Begin periodic demotion of background threads while translation runs (Windows only)
        try:
            self._start_periodic_thread_demoter()
        except Exception:
            pass
        
        # Run the heavy preparation and kickoff in a background thread to avoid GUI freeze
        threading.Thread(
            target=self._start_translation_heavy,
            args=(previous_future, previous_thread, start_token),
            name="MangaStartHeavy",
            daemon=True
        ).start()
        return
    
    def _on_create_cbz_toggle(self, state=None):
        """Handle create .cbz file at translation end toggle"""
        try:
            enabled = bool(self.create_cbz_checkbox.isChecked()) if hasattr(self, 'create_cbz_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self.create_cbz_at_end_value = enabled
        # Persist with the existing save mechanism
        self._save_rendering_settings()
    
    def _on_auto_consolidate_toggle(self, state=None):
        """Handle auto consolidate images at translation end toggle"""
        try:
            enabled = bool(self.auto_consolidate_checkbox.isChecked()) if hasattr(self, 'auto_consolidate_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self.auto_consolidate_images_value = enabled
        # Persist with the existing save mechanism (consistent with create_cbz_toggle)
        self._save_rendering_settings()
    
    def _on_manga_output_token_limit_change(self, value=None):
        """Handle manga output token limit spinbox change"""
        try:
            if value is None and hasattr(self, 'manga_output_token_limit_spin'):
                value = self.manga_output_token_limit_spin.value()
            if value is None:
                return
            
            # Save to config
            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config'):
                if 'manga_settings' not in self.main_gui.config:
                    self.main_gui.config['manga_settings'] = {}
                if 'manual_edit' not in self.main_gui.config['manga_settings']:
                    self.main_gui.config['manga_settings']['manual_edit'] = {}
                self.main_gui.config['manga_settings']['manual_edit']['manga_output_token_limit'] = value
                
                # Persist with the existing save mechanism (consistent with other handlers)
                self._save_rendering_settings()
        except Exception as e:
            self._log(f"Error saving manga output token limit: {e}", "warning")

    def _choose_ocr_document_path(self, title: str) -> Optional[str]:
        initial_dir = self._manga_ocr_output_dir()
        try:
            os.makedirs(initial_dir, exist_ok=True)
        except OSError:
            # QFileDialog can still fall back to its normal location if the
            # configured output directory is temporarily unavailable.
            initial_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            title,
            initial_dir,
            "Glossarion Manga OCR (*.json);;JSON Files (*.json);;All Files (*)",
        )
        return path or None

    def _load_ocr_document_from_dialog(self, title: str):
        """Synchronous compatibility helper; production imports use a worker."""
        path = self._choose_ocr_document_path(title)
        if not path:
            return None, None
        try:
            return path, manga_ocr_io.load_document(path)
        except manga_ocr_io.MangaOcrFormatError as error:
            QMessageBox.warning(self.dialog, "Invalid OCR File", str(error))
            return None, None

    def _ocr_drop_local_json_paths(self, mime_data) -> List[str]:
        """Return unique local JSON files from an OCR import drop payload."""
        if mime_data is None or not mime_data.hasUrls():
            return []
        paths: List[str] = []
        seen = set()
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = os.path.abspath(url.toLocalFile())
            key = os.path.normcase(os.path.normpath(path))
            if key in seen or not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].casefold() != '.json':
                continue
            seen.add(key)
            paths.append(path)
        return paths

    def _set_ocr_import_drop_highlight(self, active: bool) -> None:
        button = getattr(self, 'batch_ocr_import_btn', None)
        if button is None:
            return
        button.setProperty("ocrDropActive", bool(active))
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _import_batch_ocr_path(self, path: str) -> bool:
        """Start a non-blocking import for a dropped OCR session."""
        files = list(self._current_manga_processing_files() or self.selected_files or [])
        if not files:
            QMessageBox.warning(self.dialog, "No Files", "Load the manga images before importing OCR text.")
            return False
        self._start_ocr_import_worker(path, files)
        return True

    def _import_batch_ocr_text(self) -> None:
        """Load OCR that the next Start Translation run should reuse."""
        files = list(self._current_manga_processing_files() or self.selected_files or [])
        if not files:
            QMessageBox.warning(self.dialog, "No Files", "Load the manga images before importing OCR text.")
            return
        path = self._choose_ocr_document_path("Import Manga OCR Text")
        if not path:
            return
        self._start_ocr_import_worker(path, files)

    def _set_ocr_import_busy(self, busy: bool) -> None:
        batch_button = getattr(self, 'batch_ocr_import_btn', None)
        if batch_button is not None:
            batch_button.setEnabled(not busy)
            if busy:
                batch_button.setText("⏳ Importing OCR...")
            elif not getattr(self, '_imported_ocr_document', None):
                batch_button.setText("📥 Import OCR")
        preview_button = getattr(getattr(self, 'image_preview_widget', None), 'import_ocr_btn', None)
        if preview_button is not None:
            preview_button.setEnabled(not busy)
            preview_button.setText("⏳ Importing...") if busy else preview_button.setText("📥 Import")

    def _start_ocr_import_worker(self, path: str, files: List[str]) -> None:
        """Parse and prepare an OCR session without blocking Qt's event loop."""
        generation = int(getattr(self, '_ocr_import_generation', 0)) + 1
        self._ocr_import_generation = generation
        self._set_ocr_import_busy(True)

        def load_session():
            try:
                document = manga_ocr_io.load_document(path)
                matches = manga_ocr_io.match_document_pages(document, files)
                imported_states = {
                    image_path: manga_ocr_io.editor_state_from_page(page)
                    for image_path, page in matches.items()
                }
                error = None
            except Exception as import_error:
                document = None
                matches = {}
                imported_states = {}
                error = import_error
            self.update_queue.put(('call_method', self._finish_ocr_import_worker, (
                generation,
                path,
                document,
                list(files),
                matches,
                imported_states,
                error,
            )))

        worker = threading.Thread(
            target=load_session,
            name=f"MangaOcrImport-{generation}",
            daemon=True,
        )
        self._ocr_import_thread = worker
        worker.start()

    def _finish_ocr_import_worker(
        self,
        generation: int,
        path: str,
        document: Optional[Dict[str, Any]],
        files: List[str],
        matches: Dict[str, Dict[str, Any]],
        imported_states: Dict[str, Dict[str, Any]],
        error: Optional[Exception],
    ) -> None:
        if generation != getattr(self, '_ocr_import_generation', generation):
            return
        self._set_ocr_import_busy(False)
        if error is not None or document is None:
            QMessageBox.warning(self.dialog, "Invalid OCR File", str(error or "Could not load OCR session"))
            return
        self._apply_imported_batch_ocr_document(
            path,
            document,
            files,
            matches_override=matches,
            imported_states=imported_states,
        )

    def _apply_imported_batch_ocr_document(
        self,
        path: str,
        document: Dict[str, Any],
        files: List[str],
        *,
        matches_override: Optional[Dict[str, Dict[str, Any]]] = None,
        imported_states: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        """Apply and report an OCR document loaded by either click or drop."""
        self._imported_ocr_document = document
        if matches_override is None:
            matches = self._refresh_imported_ocr_page_map(files)
        else:
            matches = dict(matches_override)
            self._imported_ocr_page_map = {
                os.path.normcase(os.path.abspath(image_path)): page
                for image_path, page in matches.items()
            }
        if not matches:
            self._imported_ocr_document = None
            QMessageBox.warning(
                self.dialog,
                "No Matching Pages",
                "The OCR file does not match any currently loaded manga images.",
            )
            return False

        translated_region_count = sum(
            1
            for page in matches.values()
            for region in (page.get('regions') or [])
            if isinstance(region, dict)
            and str(region.get('translated_text') or '').strip()
        )

        # Batch import is also a session restore. Populate the manual editor's
        # state from canonical regions so imported translations appear now,
        # rather than only becoming visible after another translation run.
        current_source = None
        try:
            state_manager = getattr(self, 'image_state_manager', None)
            preview = getattr(self, 'image_preview_widget', None)
            if state_manager is not None:
                for image_path, page in matches.items():
                    imported_state = (
                        imported_states.get(image_path)
                        if imported_states is not None else None
                    ) or manga_ocr_io.editor_state_from_page(page)
                    existing_state = state_manager.get_state(image_path) or {}
                    existing_state.update(imported_state)
                    state_manager.set_state(image_path, existing_state, save=False)
                if hasattr(state_manager, 'flush_async'):
                    state_manager.flush_async()
                elif hasattr(state_manager, 'flush'):
                    threading.Thread(target=state_manager.flush, daemon=True).start()

                current = getattr(preview, 'current_image_path', None) if preview else None
                current_source = None
                if current:
                    current_key = os.path.normcase(os.path.abspath(current))
                    current_source = next(
                        (
                            image_path for image_path in matches
                            if os.path.normcase(os.path.abspath(image_path)) == current_key
                        ),
                        None,
                    )
                if current_source:
                    ImageRenderer._clear_cross_image_state(self)
                    ImageRenderer._rehydrate_text_state_from_persisted(self, current_source)
                    ImageRenderer._restore_image_state_overlays_only(self, current_source)
        except Exception as restore_error:
            self._log(f"Could not refresh imported OCR session preview: {restore_error}", "warning")

        background_refresh = getattr(self, '_start_imported_ocr_preview_refresh', None)
        if callable(background_refresh):
            background_refresh(matches, priority_path=current_source)

        if hasattr(self, 'batch_ocr_import_btn'):
            self.batch_ocr_import_btn.setText(f"📥 Imported OCR ({len(matches)})")
            self.batch_ocr_import_btn.setToolTip(path)
        self._log(
            f"Imported OCR for {len(matches)}/{len(files)} loaded pages "
            f"with {translated_region_count} translated regions",
            "success",
        )
        if translated_region_count:
            QMessageBox.information(
                self.dialog,
                "OCR Session Imported",
                f"Matched {len(matches)} of {len(files)} loaded pages and restored "
                f"{translated_region_count} translated regions.\n\n"
                "Start Translation will reuse both OCR and existing translations.",
            )
        else:
            self._log("The imported session contains OCR only and no translated text", "warning")
            QMessageBox.warning(
                self.dialog,
                "OCR-Only Session Imported",
                f"Matched {len(matches)} of {len(files)} loaded pages, but this JSON "
                "contains no translated text.\n\nOCR will be reused, but translations "
                "cannot be restored from this file.",
            )
        return True

    def _start_imported_ocr_preview_refresh(
        self,
        matches: Dict[str, Dict[str, Any]],
        *,
        exclude_path: Optional[str] = None,
        priority_path: Optional[str] = None,
    ) -> None:
        """Render every non-visible imported page from persisted state."""
        targets = []
        excluded_key = (
            os.path.normcase(os.path.abspath(exclude_path))
            if exclude_path else None
        )
        for image_path, page in matches.items():
            image_key = os.path.normcase(os.path.abspath(image_path))
            if excluded_key and image_key == excluded_key:
                continue
            has_translation = any(
                isinstance(region, dict)
                and str(region.get('translated_text') or '').strip()
                for region in (page.get('regions') or [])
            )
            if has_translation:
                targets.append(image_path)
        if priority_path:
            priority_key = os.path.normcase(os.path.abspath(priority_path))
            targets.sort(
                key=lambda image_path: 0
                if os.path.normcase(os.path.abspath(image_path)) == priority_key
                else 1
            )
        if not targets:
            return

        generation = int(getattr(self, '_imported_ocr_preview_generation', 0)) + 1
        self._imported_ocr_preview_generation = generation
        self._log(
            f"Rendering imported translations for {len(targets)} additional pages...",
            "info",
        )

        def render_pages():
            rendered_count = 0
            for image_path in targets:
                if generation != getattr(self, '_imported_ocr_preview_generation', generation):
                    break
                output_path = ImageRenderer.render_persisted_translation_state(
                    self,
                    image_path,
                    refresh_preview=False,
                )
                if not output_path:
                    continue
                rendered_count += 1
                self.update_queue.put(('preview_update', {
                    'translated_path': output_path,
                    'source_path': image_path,
                    'switch_to_output': True,
                }))
            self._log(
                f"Finished rendering imported translations for {rendered_count}/{len(targets)} additional pages",
                "success" if rendered_count == len(targets) else "warning",
            )

        worker = threading.Thread(
            target=render_pages,
            name=f"ImportedOcrPreview-{generation}",
            daemon=True,
        )
        self._imported_ocr_preview_thread = worker
        worker.start()

    def _open_automatic_ocr_folder(self) -> None:
        """Open the dedicated auto-saved manga OCR folder."""
        output_dir = self._manga_ocr_output_dir()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as error:
            QMessageBox.warning(
                self.dialog,
                "Could Not Open OCR Folder",
                f"Glossarion could not create the OCR folder:\n{output_dir}\n\n{error}",
            )
            return

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir)):
            QMessageBox.warning(
                self.dialog,
                "Could Not Open OCR Folder",
                f"Glossarion could not open:\n{output_dir}",
            )
            return
        self._log(f"Opened auto-saved OCR folder: {output_dir}", "info")

    def _export_automatic_ocr_text(self) -> None:
        """Copy the automatically generated OCR manifest to a user-selected path."""
        source_path = self._latest_automatic_ocr_path()
        if not source_path:
            QMessageBox.warning(
                self.dialog,
                "No OCR Export",
                "Run manga translation first so Glossarion can create the OCR text file.",
            )
            return
        destination, _ = QFileDialog.getSaveFileName(
            self.dialog,
            "Export Manga OCR Text",
            self._manga_ocr_save_dialog_path(os.path.basename(source_path)),
            "Glossarion Manga OCR (*.json);;JSON Files (*.json)",
        )
        if not destination:
            return
        if not os.path.splitext(destination)[1]:
            destination += '.json'
        try:
            if os.path.normcase(os.path.abspath(destination)) != os.path.normcase(os.path.abspath(source_path)):
                shutil.copy2(source_path, destination)
            self._log(f"Exported OCR text to: {destination}", "success")
            QMessageBox.information(self.dialog, "OCR Exported", f"OCR text exported to:\n{destination}")
        except Exception as error:
            QMessageBox.warning(self.dialog, "Export Failed", str(error))

    def _manual_ocr_files(self) -> List[str]:
        preview_files = list(getattr(self.image_preview_widget, 'image_paths', []) or [])
        files = preview_files or list(self._current_manga_processing_files() or self.selected_files or [])
        current = getattr(self.image_preview_widget, 'current_image_path', None)
        if current and current not in files:
            files.append(current)
        return files

    def _manual_editor_state_for_export(self, image_path: str) -> Dict[str, Any]:
        """Return persisted state plus the current editor's unsaved text maps."""
        state = dict(self.image_state_manager.get_state(image_path) or {})
        current = getattr(self.image_preview_widget, 'current_image_path', None)
        try:
            is_current = (
                current
                and os.path.normcase(os.path.abspath(current))
                == os.path.normcase(os.path.abspath(image_path))
            )
        except Exception:
            is_current = current == image_path
        if not is_current:
            return state

        recognized_owner = getattr(self, '_recognized_texts_image_path', None)
        if not recognized_owner or os.path.normcase(os.path.abspath(recognized_owner)) == os.path.normcase(os.path.abspath(image_path)):
            live_recognized = list(getattr(self, '_recognized_texts', []) or [])
            if live_recognized:
                state['recognized_texts'] = live_recognized

        translation_owner = (
            getattr(self, '_translation_data_image_path', None)
            or getattr(self, '_translating_image_path', None)
        )
        if translation_owner:
            try:
                owns_translation = (
                    os.path.normcase(os.path.abspath(translation_owner))
                    == os.path.normcase(os.path.abspath(image_path))
                )
            except Exception:
                owns_translation = translation_owner == image_path
        else:
            owns_translation = True

        if not owns_translation:
            return state

        # The queued state update normally contains this list, but Export can
        # be clicked while the current editor still has the newest copy only
        # in memory.
        live_translated = list(getattr(self, '_translated_texts', []) or [])
        if live_translated:
            state['translated_texts'] = live_translated

        translation_data = getattr(self, '_translation_data', None)
        if not isinstance(translation_data, dict) or not translation_data:
            return state

        translated = list(state.get('translated_texts') or [])
        recognized_by_index = {}
        for fallback_index, record in enumerate(state.get('recognized_texts') or []):
            if not isinstance(record, dict):
                continue
            try:
                record_index = int(record.get('region_index', fallback_index))
            except (TypeError, ValueError):
                continue
            recognized_by_index[record_index] = record

        for raw_index, data in translation_data.items():
            if not isinstance(data, dict):
                continue
            try:
                region_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            while len(translated) <= region_index:
                translated.append({})
            prior = translated[region_index] if isinstance(translated[region_index], dict) else {}
            recognized = recognized_by_index.get(region_index, {})
            translated[region_index] = {
                'original': {
                    'region_index': region_index,
                    'text': data.get('original', recognized.get('text', '')),
                },
                'translation': data.get('translation', prior.get('translation', '')),
                'bbox': prior.get('bbox') or recognized.get('bbox') or [0, 0, 1, 1],
            }
        state['translated_texts'] = translated
        return state

    def _export_manual_ocr_text(self) -> None:
        """Export editor state without serializing it on Qt's GUI thread."""
        current = getattr(self.image_preview_widget, 'current_image_path', None)
        if current:
            ImageRenderer._persist_current_image_state(self)
        files = self._manual_ocr_files()
        source_root = self._current_manga_source_dir()
        destination, _ = QFileDialog.getSaveFileName(
            self.dialog,
            "Export Manual Manga OCR",
            self._manga_ocr_save_dialog_path(
                self._manga_ocr_timestamped_export_filename()
            ),
            "Glossarion Manga OCR (*.json);;JSON Files (*.json)",
        )
        if not destination:
            return
        if not os.path.splitext(destination)[1]:
            destination += '.json'
        generation = int(getattr(self, '_ocr_export_generation', 0)) + 1
        self._ocr_export_generation = generation
        self._set_manual_ocr_export_busy(True)

        def export_session():
            pages = []
            translated_region_count = 0
            error = None
            try:
                for index, image_path in enumerate(files, start=1):
                    state = self._manual_editor_state_for_export(image_path)
                    regions = manga_ocr_io.canonical_regions_from_editor_state(state)
                    if not regions and not state.get('recognized_texts'):
                        continue
                    translated_region_count += sum(
                        1 for region in regions
                        if str(region.get('translated_text') or '').strip()
                    )
                    pages.append(manga_ocr_io.make_page(
                        image_path,
                        regions,
                        index=index,
                        source_root=source_root,
                        editor_state=state,
                    ))
                if pages:
                    document = manga_ocr_io.create_document(
                        pages,
                        workflow="manual-editor",
                        source_root=source_root,
                    )
                    manga_ocr_io.write_document(destination, document)
            except Exception as export_error:
                error = export_error
            self.update_queue.put(('call_method', self._finish_manual_ocr_export, (
                generation,
                destination,
                len(pages),
                translated_region_count,
                error,
            )))

        worker = threading.Thread(
            target=export_session,
            name=f"MangaOcrExport-{generation}",
            daemon=True,
        )
        self._ocr_export_thread = worker
        worker.start()

    def _set_manual_ocr_export_busy(self, busy: bool) -> None:
        button = getattr(getattr(self, 'image_preview_widget', None), 'export_ocr_btn', None)
        if button is None:
            return
        button.setEnabled(not busy)
        button.setText("⏳ Exporting...") if busy else button.setText("📤 Export")

    def _finish_manual_ocr_export(
        self,
        generation: int,
        destination: str,
        page_count: int,
        translated_region_count: int,
        error: Optional[Exception],
    ) -> None:
        if generation != getattr(self, '_ocr_export_generation', generation):
            return
        self._set_manual_ocr_export_busy(False)
        if error is not None:
            QMessageBox.warning(self.dialog, "Export Failed", str(error))
            return
        if not page_count:
            QMessageBox.warning(
                self.dialog,
                "No OCR Text",
                "Recognize text in at least one image before exporting.",
            )
            return
        self._log(f"Exported manual OCR/translation mappings for {page_count} pages", "success")
        QMessageBox.information(
            self.dialog,
            "Session Exported",
            f"Exported OCR text, {translated_region_count} translated regions, "
            f"and mappings for {page_count} pages to:\n{destination}",
        )

    def _import_manual_ocr_text(self) -> None:
        """Restore OCR, translations, and editor mappings into the current manga image set."""
        files = self._manual_ocr_files()
        if not files:
            QMessageBox.warning(self.dialog, "No Files", "Load the manga images before importing OCR text.")
            return
        path = self._choose_ocr_document_path("Import Manual Manga OCR")
        if not path:
            return
        self._start_ocr_import_worker(path, files)

    def _refresh_imported_manual_preview(self, image_path: str) -> bool:
        """Render imported translations and reload both preview surfaces."""
        state = self.image_state_manager.get_state(image_path) or {}
        translated_texts = [
            entry for entry in (state.get('translated_texts') or [])
            if isinstance(entry, dict)
            and not entry.get('deleted')
            and str(entry.get('translation') or '').strip()
        ]
        if not translated_texts:
            return False

        preview = self.image_preview_widget
        preview.source_display_mode = 'translated'
        preview.cleaned_images_enabled = True
        self._log(
            f"Rendering {len(translated_texts)} imported translations for preview...",
            "info",
        )

        # This uses the restored rectangles and imported text map, then writes
        # the normal isolated translated image for the current page.
        ImageRenderer.save_positions_and_rerender(self)

        refreshed_state = self.image_state_manager.get_state(image_path) or {}
        rendered_path = refreshed_state.get('rendered_image_path')
        if not rendered_path or not os.path.isfile(rendered_path):
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_root = (
                getattr(self.main_gui, 'config', {}).get('output_directory')
                or os.environ.get('OUTPUT_DIRECTORY')
                or os.path.dirname(image_path)
            )
            candidate = os.path.join(output_root, f"{base_name}_translated", filename)
            rendered_path = candidate if os.path.isfile(candidate) else None

        if rendered_path:
            preview.current_translated_path = rendered_path
            if not hasattr(self, '_rendered_images_map'):
                self._rendered_images_map = {}
            self._rendered_images_map[image_path] = rendered_path
            if getattr(preview, 'output_viewer', None):
                preview.output_viewer.load_image(rendered_path)

        # load_image resolves translated mode to the newly rendered output and
        # preserve flags keep imported rectangles/context menus attached.
        preview.load_image(
            image_path,
            preserve_rectangles=True,
            preserve_text_overlays=True,
        )
        if rendered_path:
            self._log("Imported translation preview refreshed", "success")
            return True

        self._log(
            "Imported text was restored, but the translated preview could not be rendered",
            "warning",
        )
        return False

    def _reset_ui_state(self, expected_start_token=None):
        """Reset UI to ready state - with widget existence checks (PySide6)"""
        if not _translation_run_token_matches(
            getattr(self, '_translation_start_token', None),
            expected_start_token,
        ):
            # Delayed stop timers and old worker completion events are allowed
            # to arrive, but they cannot reset a newer translation run.
            return False

        # Check if the dialog still exists first (PySide6)
        if not hasattr(self, 'dialog') or not self.dialog:
            return False
        
        # Restore stdio redirection if active
        self._redirect_stderr(False)
        self._redirect_stdout(False)
        # Stop any startup heartbeat if still running
        try:
            self._stop_startup_heartbeat()
        except Exception:
            pass
        try:
            # Reset running flag and graceful stop state
            self.is_running = False
            self._graceful_stop_pending = False
            self._translation_startup_pending = False
            self._translation_start_cancel_requested = False
            self._stop_click_times = []
            
            # Reset start button to original Start state (green)
            if hasattr(self, 'start_button') and self.start_button:
                # Update text label instead of button text
                if hasattr(self, 'start_button_text'):
                    self.start_button_text.setText("▶ Start Translation")
                self.start_button.setStyleSheet(
                    "QPushButton { "
                    "  background-color: #28a745; "
                    "  color: white; "
                    "  padding: 22px 30px; "
                    "  font-size: 14pt; "
                    "  font-weight: bold; "
                    "  border-radius: 8px; "
                    "} "
                    "QPushButton:hover { background-color: #218838; } "
                    "QPushButton:disabled { "
                    "  background-color: #2d2d2d; "
                    "  color: #666666; "
                    "}"
                )
                self.start_button.setEnabled(True)
                # Stop spinning animation gracefully immediately (no delay)
                if hasattr(self, 'start_icon_spin_animation') and hasattr(self, 'start_button_icon') and hasattr(self, 'start_icon_stop_animation'):
                    def stop_spinning():
                        if not hasattr(self, 'start_icon_spin_animation'):
                            return
                        if self.start_icon_spin_animation.state() == QPropertyAnimation.Running:
                            self.start_icon_spin_animation.stop()
                            # Stop refresh timer
                            if hasattr(self, '_animation_refresh_timer'):
                                self._animation_refresh_timer.stop()
                            current_rotation = self.start_button_icon.get_rotation()
                            current_rotation = current_rotation % 360
                            if current_rotation > 180:
                                target_rotation = 360
                            else:
                                target_rotation = 0
                            self.start_icon_stop_animation.setStartValue(current_rotation)
                            self.start_icon_stop_animation.setEndValue(target_rotation)
                            self.start_icon_stop_animation.start()
                        elif self.start_icon_stop_animation.state() != QPropertyAnimation.Running:
                            self.start_button_icon.set_rotation(0)
                            # Ensure refresh timer is stopped
                            if hasattr(self, '_animation_refresh_timer'):
                                self._animation_refresh_timer.stop()
                    # Call immediately when button turns green
                    stop_spinning()
            
            # Re-enable file modification - check if listbox exists (PySide6)
            try:
                self._set_file_selection_editing_enabled(True)
            except Exception:
                if hasattr(self, 'file_listbox') and self.file_listbox:
                    self.file_listbox.setEnabled(True)
            
            # Re-enable ALL workflow buttons after translation stops/completes
            try:
                import ImageRenderer
                ImageRenderer._enable_workflow_buttons(self)
            except Exception as e:
                print(f"[RESET_UI] Error enabling workflow buttons: {e}")
                
        except Exception as e:
            # Log the error but don't crash
            if hasattr(self, '_log'):
                self._log(f"Error resetting UI state: {str(e)}", "warning")
            return False

        return True
