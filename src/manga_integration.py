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
from PySide6.QtWidgets import (QWidget, QLabel, QFrame, QPushButton, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QListWidget, QComboBox, QLineEdit, QCheckBox,
                               QRadioButton, QSlider, QSpinBox, QDoubleSpinBox, QTextEdit,
                               QProgressBar, QFileDialog, QMessageBox, QColorDialog, QScrollArea,
                               QDialog, QButtonGroup, QApplication, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QEvent, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QIcon, QKeyEvent, QPixmap, QTransform
from typing import List, Dict, Optional, Any
from queue import Queue
import logging
from manga_translator import MangaTranslator, GOOGLE_CLOUD_VISION_AVAILABLE
from manga_settings_dialog import MangaSettingsDialog

# Optional: psutil/ctypes helpers to reduce GUI lag by lowering background thread priority
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import ctypes
import platform

_IS_WINDOWS = platform.system().lower().startswith('win')

# Windows thread priority constants
if _IS_WINDOWS:
    _THREAD_PRIORITY_IDLE = -15
    _THREAD_PRIORITY_LOWEST = -2
    _THREAD_PRIORITY_BELOW_NORMAL = -1
    _THREAD_PRIORITY_NORMAL = 0
    _THREAD_SET_INFORMATION = 0x0020
    _THREAD_QUERY_INFORMATION = 0x0040

    _kernel32 = ctypes.windll.kernel32

    def _win_set_current_thread_priority(level=_THREAD_PRIORITY_BELOW_NORMAL):
        try:
            _kernel32.SetThreadPriority(_kernel32.GetCurrentThread(), ctypes.c_int(level))
        except Exception:
            pass

    def _win_set_current_thread_affinity(reserve_cores: int = 1):
        try:
            # Build an affinity mask that leaves some low-index cores free for the GUI
            cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count() or 1
            allow = max(1, cpu_count - max(0, reserve_cores))
            mask = 0
            for i in range(allow):
                mask |= (1 << i)
            _kernel32.SetThreadAffinityMask(_kernel32.GetCurrentThread(), ctypes.c_size_t(mask))
        except Exception:
            pass

    def _win_set_thread_priority_by_tid(tid: int, level=_THREAD_PRIORITY_BELOW_NORMAL):
        try:
            handle = _kernel32.OpenThread(_THREAD_SET_INFORMATION | _THREAD_QUERY_INFORMATION, False, ctypes.c_uint32(tid))
            if handle:
                _kernel32.SetThreadPriority(handle, ctypes.c_int(level))
                _kernel32.CloseHandle(handle)
        except Exception:
            pass

    def _win_set_thread_affinity_by_tid(tid: int, reserve_cores: int = 1):
        try:
            handle = _kernel32.OpenThread(_THREAD_SET_INFORMATION | _THREAD_QUERY_INFORMATION, False, ctypes.c_uint32(tid))
            if handle:
                cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count() or 1
                allow = max(1, cpu_count - max(0, reserve_cores))
                mask = 0
                for i in range(allow):
                    mask |= (1 << i)
                _kernel32.SetThreadAffinityMask(handle, ctypes.c_size_t(mask))
                _kernel32.CloseHandle(handle)
        except Exception:
            pass

    def _lower_current_thread_priority_and_affinity(reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        try:
            reserve = 1
            try:
                reserve = int(os.environ.get(reserve_env_key, '1'))
            except Exception:
                reserve = 1
            _win_set_current_thread_priority(_THREAD_PRIORITY_BELOW_NORMAL)
            _win_set_current_thread_affinity(reserve)
        except Exception:
            pass

    def _demote_non_main_threads(main_tid: int, reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        try:
            reserve = 1
            try:
                reserve = int(os.environ.get(reserve_env_key, '1'))
            except Exception:
                reserve = 1
            if psutil is None:
                return
            p = psutil.Process()
            for th in p.threads():
                tid = int(th.id)
                if tid != main_tid:
                    _win_set_thread_priority_by_tid(tid, _THREAD_PRIORITY_BELOW_NORMAL)
                    _win_set_thread_affinity_by_tid(tid, reserve)
        except Exception:
            pass
else:
    def _lower_current_thread_priority_and_affinity(reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        # Non-Windows: no-op (per-thread priority not easily portable without extra deps)
        return

    def _demote_non_main_threads(main_tid: int, reserve_env_key: str = 'MANGA_RESERVE_CORES'):
        return

# Try to import UnifiedClient for API initialization
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

# Module-level function for multiprocessing (must be picklable)
def _preload_models_worker(models_list, progress_queue):
    """Worker function to preload models in separate process (module-level for pickling)"""
    try:
        total_steps = len(models_list)
        
        for idx, (model_type, model_key, model_name, model_path) in enumerate(models_list):
            try:
                # Send start progress
                base_progress = int((idx / total_steps) * 100)
                progress_queue.put(('progress', base_progress, model_name))
                
                if model_type == 'detector':
                    from bubble_detector import BubbleDetector
                    from manga_translator import MangaTranslator
                    
                    # Progress: 0-25% of this model's portion
                    progress_queue.put(('progress', base_progress + int(25 / total_steps), f"{model_name} - Initializing"))
                    bd = BubbleDetector()
                    
                    # Progress: 25-75% - loading model
                    progress_queue.put(('progress', base_progress + int(50 / total_steps), f"{model_name} - Downloading/Loading"))
                    if model_key == 'rtdetr_onnx':
                        model_repo = model_path if model_path else 'ogkalu/comic-text-and-bubble-detector'
                        bd.load_rtdetr_onnx_model(model_repo)
                    elif model_key == 'rtdetr':
                        bd.load_rtdetr_model()
                    elif model_key == 'yolo':
                        if model_path:
                            bd.load_model(model_path)
                    
                    # Progress: 75-100% - finalizing
                    progress_queue.put(('progress', base_progress + int(75 / total_steps), f"{model_name} - Finalizing"))
                    progress_queue.put(('loaded', model_type, model_name))
                    
                elif model_type == 'inpainter':
                    from local_inpainter import LocalInpainter
                    
                    # Progress: 0-25%
                    progress_queue.put(('progress', base_progress + int(25 / total_steps), f"{model_name} - Initializing"))
                    inp = LocalInpainter()
                    resolved_path = model_path
                    
                    if not resolved_path or not os.path.exists(resolved_path):
                        # Progress: 25-50% - downloading
                        progress_queue.put(('progress', base_progress + int(40 / total_steps), f"{model_name} - Downloading"))
                        try:
                            resolved_path = inp.download_jit_model(model_key)
                        except:
                            resolved_path = None
                    
                    if resolved_path and os.path.exists(resolved_path):
                        # Progress: 50-90% - loading
                        progress_queue.put(('progress', base_progress + int(60 / total_steps), f"{model_name} - Loading model"))
                        success = inp.load_model_with_retry(model_key, resolved_path)
                        
                        # Progress: 90-100% - finalizing
                        progress_queue.put(('progress', base_progress + int(85 / total_steps), f"{model_name} - Finalizing"))
                        if success:
                            progress_queue.put(('loaded', model_type, model_name))
            
            except Exception as e:
                progress_queue.put(('error', model_name, str(e)))
        
        # Send completion signal
        progress_queue.put(('complete', None, None))
        
    except Exception as e:
        progress_queue.put(('error', 'Process', str(e)))

class _MangaGuiLogHandler(logging.Handler):
    """Forward logging records into MangaTranslationTab._log."""
    def __init__(self, gui_ref, level=logging.INFO):
        super().__init__(level)
        self.gui_ref = gui_ref
        self._last_msg = None
        self.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        # Avoid looping/duplicates from this module's own messages or when stdio is redirected
        try:
            if getattr(self.gui_ref, '_stdio_redirect_active', False):
                return
            # Filter out manga_translator, bubble_detector, local_inpainter logs as they're already shown
            if record and isinstance(record.name, str):
                if record.name.startswith(('manga_integration', 'manga_translator', 'bubble_detector', 'local_inpainter', 'unified_api_client', 'google_genai', 'httpx')):
                    return
        except Exception:
            pass
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # Deduplicate identical consecutive messages
        if msg == self._last_msg:
            return
        self._last_msg = msg
        
        # Map logging levels to our tag levels
        lvl = record.levelname.lower()
        tag = 'info'
        if lvl.startswith('warn'):
            tag = 'warning'
        elif lvl.startswith('err') or lvl.startswith('crit'):
            tag = 'error'
        elif lvl.startswith('debug'):
            tag = 'debug'
        elif lvl.startswith('info'):
            tag = 'info'
        
        # Always store to persistent log (even if GUI is closed)
        try:
            with MangaTranslationTab._persistent_log_lock:
                if len(MangaTranslationTab._persistent_log) >= 1000:
                    MangaTranslationTab._persistent_log.pop(0)
                MangaTranslationTab._persistent_log.append((msg, tag))
        except Exception:
            pass
        
        # Also try to display in GUI if it exists
        try:
            if hasattr(self.gui_ref, '_log'):
                self.gui_ref._log(msg, tag)
        except Exception:
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

class MangaTranslationTab:
    """GUI interface for manga translation integrated with TranslatorGUI"""
    
    # Class-level cancellation flag for all instances
    _global_cancelled = False
    _global_cancel_lock = threading.RLock()
    
    # Class-level log storage to persist across window closures
    _persistent_log = []
    _persistent_log_lock = threading.RLock()
    
    # Class-level preload tracking to prevent duplicate loading
    _preload_in_progress = False
    _preload_lock = threading.RLock()
    _preload_completed_models = set()  # Track which models have been loaded
    
    @classmethod
    def set_global_cancellation(cls, cancelled: bool):
        """Set global cancellation flag for all translation instances"""
        with cls._global_cancel_lock:
            cls._global_cancelled = cancelled
    
    @classmethod
    def is_globally_cancelled(cls) -> bool:
        """Check if globally cancelled"""
        with cls._global_cancel_lock:
            return cls._global_cancelled
    
    def __init__(self, parent_widget, main_gui, dialog, scroll_area=None):
        """Initialize manga translation interface
        
        Args:
            parent_widget: The content widget for the interface (PySide6 QWidget)
            main_gui: Reference to TranslatorGUI instance
            dialog: The dialog window (PySide6 QDialog)
            scroll_area: The scroll area widget (PySide6 QScrollArea, optional)
        """
        # CRITICAL: Set thread limits FIRST before any imports or processing
        import os
        parallel_enabled = main_gui.config.get('manga_settings', {}).get('advanced', {}).get('parallel_processing', False)
        if not parallel_enabled:
            # Force single-threaded mode for all libraries
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
            os.environ['ONNXRUNTIME_NUM_THREADS'] = '1'
            # Also set torch and cv2 thread limits if already imported
            try:
                import torch
                torch.set_num_threads(1)
            except (ImportError, RuntimeError):
                pass
            try:
                import cv2
                cv2.setNumThreads(1)
            except (ImportError, AttributeError):
                pass
        
        self.parent_widget = parent_widget
        self.main_gui = main_gui
        self.dialog = dialog
        self.scroll_area = scroll_area
        
        # Record main GUI thread native id for selective demotion of background threads (Windows)
        try:
            self._main_thread_tid = threading.get_native_id()
        except Exception:
            self._main_thread_tid = None
        
        # Translation state
        self.translator = None
        self.is_running = False
        self.stop_flag = threading.Event()
        self.translation_thread = None
        self.translation_future = None
        # Shared executor from main GUI if available
        try:
            if hasattr(self.main_gui, 'executor') and self.main_gui.executor:
                self.executor = self.main_gui.executor
            else:
                self.executor = None
        except Exception:
            self.executor = None
        self.selected_files = []
        self.current_file_index = 0
        self.font_mapping = {}  # Initialize font mapping dictionary

        
        # Progress tracking
        self.total_files = 0
        self.completed_files = 0
        self.failed_files = 0
        self.qwen2vl_model_size = self.main_gui.config.get('qwen2vl_model_size', '1')
        
        # Advanced performance toggles
        try:
            adv_cfg = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
        except Exception:
            adv_cfg = {}
        # In singleton mode, reduce OpenCV thread usage to avoid CPU spikes
        try:
            if bool(adv_cfg.get('use_singleton_models', False)):
                import cv2 as _cv2
                try:
                    _cv2.setNumThreads(1)
                except Exception:
                    pass
        except Exception:
            pass
        # Do NOT preload big local models by default to avoid startup crashes
        self.preload_local_models_on_open = bool(adv_cfg.get('preload_local_models_on_open', False))
        
        # Queue for thread-safe GUI updates
        self.update_queue = Queue()
        
        # Auto-scroll control: delay forcing scroll on new runs
        self._autoscroll_delay_until = 0.0  # epoch seconds
        self._user_scrolled_up = False  # Track if user manually scrolled up
        
        # Flags for stdio redirection to avoid duplicate GUI logs
        self._stdout_redirect_on = False
        self._stderr_redirect_on = False
        self._stdio_redirect_active = False
        
        # Flag to prevent saving during initialization
        self._initializing = True
        
        # IMPORTANT: Load settings BEFORE building interface
        # This ensures all variables are initialized before they're used in the GUI
        self._load_rendering_settings()
        
        # Initialize the full page context prompt
        self.full_page_context_prompt = (
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

        # Initialize the OCR system prompt
        self.ocr_prompt = self.main_gui.config.get('manga_ocr_prompt', 
            "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT TRANSLATE ANYTHING\n"
            "2. DO NOT MODIFY THE TEXT\n"
            "3. DO NOT EXPLAIN OR COMMENT\n"
            "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
            "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
            "If you see Korean text, output it in Korean.\n"
            "If you see Japanese text, output it in Japanese.\n"
            "If you see Chinese text, output it in Chinese.\n"
            "If you see English text, output it in English.\n\n"
            "IMPORTANT: Only use line breaks where they naturally occur in the original text "
            "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
            "between every word/character.\n\n"
            "For vertical text common in manga/comics, transcribe it as a continuous line unless "
            "there are clear visual breaks.\n\n"
            "NEVER translate. ONLY extract exactly what is written.\n"
            "Output ONLY the raw text, nothing else."
        )
        
        # flag to skip status checks during init
        self._initializing_gui = True
        
        # Build interface AFTER loading settings
        self._build_interface()

        # Now allow status checks
        self._initializing_gui = False
        
        # Do one status check after everything is built
        # Use QTimer for PySide6 dialog
        QTimer.singleShot(100, self._check_provider_status)
        
        # Start model preloading in background
        QTimer.singleShot(200, self._start_model_preloading)
        
        # Now that everything is initialized, allow saving
        self._initializing = False
        
        # Attach logging bridge so library logs appear in our log area
        self._attach_logging_bridge()

        # Start update loop
        self._process_updates()
        
        # Install event filter for F11 fullscreen toggle
        self._install_fullscreen_handler()
    
    def _is_stop_requested(self) -> bool:
        """Check if stop has been requested using multiple sources"""
        # Check global cancellation first
        if self.is_globally_cancelled():
            return True
            
        # Check local stop flag
        if hasattr(self, 'stop_flag') and self.stop_flag.is_set():
            return True
            
        # Check running state
        if hasattr(self, 'is_running') and not self.is_running:
            return True
            
        return False
    
    def _reset_global_cancellation(self):
        """Reset all global cancellation flags for new translation"""
        # Reset local class flag
        self.set_global_cancellation(False)
        
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
    
    def reset_stop_flags(self):
        """Reset all stop flags when starting new translation"""
        self.is_running = False
        if hasattr(self, 'stop_flag'):
            self.stop_flag.clear()
        self._reset_global_cancellation()
        self._log("🔄 Stop flags reset for new translation", "debug")
    
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
    
    def _distribute_stop_flags(self):
        """Distribute stop flags to all manga translation components"""
        if not hasattr(self, 'translator') or not self.translator:
            return
        
        # Set stop flag on translator
        if hasattr(self.translator, 'set_stop_flag'):
            self.translator.set_stop_flag(self.stop_flag)
        
        # Set stop flag on OCR manager and all providers
        if hasattr(self.translator, 'ocr_manager') and self.translator.ocr_manager:
            if hasattr(self.translator.ocr_manager, 'set_stop_flag'):
                self.translator.ocr_manager.set_stop_flag(self.stop_flag)
        
        # Set stop flag on bubble detector if available
        if hasattr(self.translator, 'bubble_detector') and self.translator.bubble_detector:
            if hasattr(self.translator.bubble_detector, 'set_stop_flag'):
                self.translator.bubble_detector.set_stop_flag(self.stop_flag)
                
        # Set stop flag on local inpainter if available
        if hasattr(self.translator, 'local_inpainter') and self.translator.local_inpainter:
            if hasattr(self.translator.local_inpainter, 'set_stop_flag'):
                self.translator.local_inpainter.set_stop_flag(self.stop_flag)
                
        # Also try to set on thread-local components if accessible
        if hasattr(self.translator, '_thread_local'):
            thread_local = self.translator._thread_local
            # Set on thread-local bubble detector
            if hasattr(thread_local, 'bubble_detector') and thread_local.bubble_detector:
                if hasattr(thread_local.bubble_detector, 'set_stop_flag'):
                    thread_local.bubble_detector.set_stop_flag(self.stop_flag)
            
            # Set on thread-local inpainters
            if hasattr(thread_local, 'local_inpainters') and isinstance(thread_local.local_inpainters, dict):
                for inpainter in thread_local.local_inpainters.values():
                    if hasattr(inpainter, 'set_stop_flag'):
                        inpainter.set_stop_flag(self.stop_flag)
        
        self._log("🔄 Stop flags distributed to all components", "debug")
    
    def _preflight_bubble_detector(self, ocr_settings: dict) -> bool:
        """Check if bubble detector is preloaded in the pool or already loaded.
        Returns True if a ready instance or preloaded spare is available; no heavy loads are performed here.
        """
        try:
            import time as _time
            start = _time.time()
            if not ocr_settings.get('bubble_detection_enabled', False):
                return False
            det_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
            model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''

            # 1) If translator already has a ready detector, report success
            try:
                bd = getattr(self, 'translator', None) and getattr(self.translator, 'bubble_detector', None)
                if bd and (getattr(bd, 'rtdetr_loaded', False) or getattr(bd, 'rtdetr_onnx_loaded', False) or getattr(bd, 'model_loaded', False)):
                    self._log("🤖 Bubble detector already loaded", "debug")
                    return True
            except Exception:
                pass

            # 2) Check shared preload pool for spares
            try:
                from manga_translator import MangaTranslator
                key = (det_type, model_id)
                with MangaTranslator._detector_pool_lock:
                    rec = MangaTranslator._detector_pool.get(key)
                    spares = (rec or {}).get('spares') or []
                    if len(spares) > 0:
                        self._log(f"🤖 Preflight: found {len(spares)} preloaded bubble detector spare(s) for key={key}", "info")
                        return True
            except Exception:
                pass

            # 3) No spares/ready detector yet; do not load here. Just report timing and return False.
            elapsed = _time.time() - start
            self._log(f"⏱️ Preflight checked bubble detector pool in {elapsed:.2f}s — no ready instance", "debug")
            return False
        except Exception:
            return False
    
    def _start_model_preloading(self):
        """Start preloading models in separate process for true background loading"""
        from multiprocessing import Process, Queue as MPQueue
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
        bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', False)
        skip_inpainting = self.main_gui.config.get('manga_skip_inpainting', False)
        inpainting_method = inpaint_settings.get('method', 'local')
        inpainting_enabled = not skip_inpainting and inpainting_method == 'local'
        
        # Check if models need loading
        try:
            from manga_translator import MangaTranslator
            
            if bubble_detection_enabled:
                detector_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
                model_url = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
                key = (detector_type, model_url)
                model_id = f"detector_{detector_type}_{model_url}"
                
                # Skip if already loaded in this session
                if model_id not in MangaTranslationTab._preload_completed_models:
                    with MangaTranslator._detector_pool_lock:
                        rec = MangaTranslator._detector_pool.get(key)
                        if not rec or (not rec.get('spares') and not rec.get('loaded')):
                            detector_name = 'RT-DETR ONNX' if detector_type == 'rtdetr_onnx' else 'RT-DETR' if detector_type == 'rtdetr' else 'YOLO'
                            models_to_load.append(('detector', detector_type, detector_name, model_url))
            
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
                        if not rec or (not rec.get('loaded') and not rec.get('spares')):
                            models_to_load.append(('inpainter', local_method, local_method.capitalize(), model_path))
        except Exception as e:
            print(f"Error checking models: {e}")
            return
        
        if not models_to_load:
            return
        
        # Set preload in progress flag
        with MangaTranslationTab._preload_lock:
            MangaTranslationTab._preload_in_progress = True
        
        # Show progress bar
        self.preload_progress_frame.setVisible(True)
        
        # Create queue for IPC
        progress_queue = MPQueue()
        
        # Start loading in separate process using module-level function
        load_process = Process(target=_preload_models_worker, args=(models_to_load, progress_queue), daemon=True)
        load_process.start()
        
        # Store models being loaded for tracking
        models_being_loaded = []
        for model_type, model_key, model_name, model_path in models_to_load:
            if model_type == 'detector':
                models_being_loaded.append(f"detector_{model_key}_{model_path}")
            elif model_type == 'inpainter':
                models_being_loaded.append(f"inpainter_{model_key}_{model_path}")
        
        # Monitor progress with QTimer
        def check_progress():
            try:
                while True:
                    try:
                        msg = progress_queue.get_nowait()
                        msg_type = msg[0]
                        
                        if msg_type == 'progress':
                            _, progress, model_name = msg
                            self.preload_progress_bar.setValue(progress)
                            self.preload_status_label.setText(f"Loading {model_name}...")
                        
                        elif msg_type == 'loaded':
                            _, model_type, model_name = msg
                            print(f"✓ Loaded {model_name}")
                        
                        elif msg_type == 'error':
                            _, model_name, error = msg
                            print(f"✗ Failed to load {model_name}: {error}")
                        
                        elif msg_type == 'complete':
                            # Child process cached models
                            self.preload_progress_bar.setValue(100)
                            self.preload_status_label.setText("✓ Models ready")
                            
                            # Mark all models as completed
                            with MangaTranslationTab._preload_lock:
                                MangaTranslationTab._preload_completed_models.update(models_being_loaded)
                                MangaTranslationTab._preload_in_progress = False
                            
                            # Load RT-DETR into pool in background (doesn't block GUI)
                            def load_rtdetr_bg():
                                try:
                                    from manga_translator import MangaTranslator
                                    from bubble_detector import BubbleDetector
                                    
                                    for model_type, model_key, model_name, model_path in models_to_load:
                                        if model_type == 'detector' and model_key == 'rtdetr_onnx':
                                            key = (model_key, model_path)
                                            
                                            # Check if already loaded
                                            with MangaTranslator._detector_pool_lock:
                                                rec = MangaTranslator._detector_pool.get(key)
                                                if rec and rec.get('spares'):
                                                    print(f"⏭️  {model_name} already in pool")
                                                    continue
                                            
                                            # Load into pool
                                            bd = BubbleDetector()
                                            model_repo = model_path if model_path else 'ogkalu/comic-text-and-bubble-detector'
                                            bd.load_rtdetr_onnx_model(model_repo)
                                            
                                            with MangaTranslator._detector_pool_lock:
                                                rec = MangaTranslator._detector_pool.get(key)
                                                if not rec:
                                                    rec = {'spares': []}
                                                    MangaTranslator._detector_pool[key] = rec
                                                rec['spares'].append(bd)
                                                print(f"✓ Loaded {model_name} into pool (background)")
                                except Exception as e:
                                    print(f"✗ Background RT-DETR loading error: {e}")
                            
                            # Start background loading
                            threading.Thread(target=load_rtdetr_bg, daemon=True).start()
                            
                            QTimer.singleShot(2000, lambda: self.preload_progress_frame.setVisible(False))
                            return
                    
                    except queue.Empty:
                        break
                
                QTimer.singleShot(100, check_progress)
            
            except Exception as e:
                print(f"Progress check error: {e}")
                self.preload_progress_frame.setVisible(False)
                # Reset flag on error
                with MangaTranslationTab._preload_lock:
                    MangaTranslationTab._preload_in_progress = False
        
        QTimer.singleShot(100, check_progress)
    
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
        checkmark = QLabel("✓", checkbox)
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
            else:
                checkmark.hide()
        
        checkbox.stateChanged.connect(update_checkmark)
        # Delay initial positioning to ensure widget is properly rendered
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark()))
        
        return checkbox

    def _download_hf_model(self):
        """Download HuggingFace models with progress tracking - PySide6 version"""
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                        QRadioButton, QButtonGroup, QLineEdit, QPushButton,
                                        QGroupBox, QTextEdit, QProgressBar, QFrame,
                                        QScrollArea, QWidget, QSizePolicy)
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
        
        # Create download dialog using Tkinter directly (transitional code)
        download_dialog = tk.Toplevel(self.main_gui.master)
        download_dialog.title(f"Download {provider} Model")
        download_dialog.geometry("600x450")
        scrollable_frame = tk.Frame(download_dialog)
        scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info section
        info_frame = tk.LabelFrame(
            scrollable_frame,
            text="Model Information",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if provider == 'Qwen2-VL':
            info_text = f"📚 Qwen2-VL {selected_model_key} Model\n"
            info_text += f"Model ID: {model_id}\n"
            info_text += f"Estimated size: ~{total_size_mb/1000:.1f}GB\n"
            info_text += "Vision-Language model for Korean OCR"
        else:
            info_text = f"📚 {provider} Model\nOptimized for manga/manhwa text detection"
        
        tk.Label(info_frame, text=info_text, font=('Arial', 10), justify=tk.LEFT).pack(anchor='w')
        
        # Progress section
        progress_frame = tk.LabelFrame(
            scrollable_frame,
            text="Download Progress",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        progress_label = tk.Label(progress_frame, text="Ready to download", font=('Arial', 10))
        progress_label.pack(pady=(5, 10))
        
        progress_var = tk.DoubleVar()
        try:
            # Try to use our custom progress bar style
            progress_bar = ttk.Progressbar(progress_frame, length=550, mode='determinate', 
                                          variable=progress_var,
                                          style="MangaProgress.Horizontal.TProgressbar")
        except Exception:
            # Fallback to default if style not available yet
            progress_bar = ttk.Progressbar(progress_frame, length=550, mode='determinate', 
                                          variable=progress_var)
        progress_bar.pack(pady=(0, 5))
        
        size_label = tk.Label(progress_frame, text="", font=('Arial', 9), fg='#666666')
        size_label.pack()
        
        speed_label = tk.Label(progress_frame, text="", font=('Arial', 9), fg='#666666')
        speed_label.pack()
        
        status_label = tk.Label(progress_frame, text="Click 'Download' to begin", 
                              font=('Arial', 9), fg='#666666')
        status_label.pack(pady=(5, 0))
        
        # Log section
        log_frame = tk.LabelFrame(
            scrollable_frame,
            text="Download Log",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create a frame to hold the text widget and scrollbar
        text_frame = tk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        details_text = tk.Text(
            text_frame, 
            height=12, 
            width=70, 
            font=('Courier', 9), 
            bg='#f5f5f5'
        )
        details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Attach scrollbar to the frame, not the text widget
        scrollbar = ttk.Scrollbar(text_frame, command=details_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        details_text.config(yscrollcommand=scrollbar.set)
            
        def add_log(message):
            """Add message to log"""
            details_text.insert(tk.END, f"{message}\n")
            details_text.see(tk.END)
            details_text.update()
        
        # Buttons frame
        button_frame = tk.Frame(download_dialog)
        button_frame.pack(pady=15)
        
        # Download tracking variables
        download_active = {'value': False}
        
        def get_dir_size(path):
            """Get total size of directory"""
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
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
                    progress_label.config(text="Downloading manga-ocr model...")
                    add_log("Downloading manga-ocr model from Hugging Face...")
                    add_log("This will download ~450MB of model files")
                    progress_var.set(10)
                    
                    try:
                        from huggingface_hub import snapshot_download
                        
                        # Download the model files directly without importing manga_ocr
                        model_repo = "kha-white/manga-ocr-base"
                        add_log(f"Repository: {model_repo}")
                        
                        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                        initial_size = get_dir_size(cache_dir) if os.path.exists(cache_dir) else 0
                        start_time = time.time()
                        
                        add_log("Starting download...")
                        progress_var.set(20)
                        
                        # Download with progress tracking
                        import threading
                        download_complete = threading.Event()
                        download_error = [None]
                        
                        def download_model():
                            try:
                                snapshot_download(
                                    repo_id=model_repo,
                                    repo_type="model",
                                    resume_download=True,
                                    local_files_only=False
                                )
                                download_complete.set()
                            except Exception as e:
                                download_error[0] = e
                                download_complete.set()
                        
                        download_thread = threading.Thread(target=download_model, daemon=True)
                        download_thread.start()
                        
                        # Show progress while downloading
                        while not download_complete.is_set() and download_active['value']:
                            current_size = get_dir_size(cache_dir) if os.path.exists(cache_dir) else 0
                            downloaded = current_size - initial_size
                            
                            if downloaded > 0:
                                progress = min(20 + (downloaded / total_size) * 70, 95)
                                progress_var.set(progress)
                                
                                elapsed = time.time() - start_time
                                if elapsed > 1:
                                    speed = downloaded / elapsed
                                    speed_mb = speed / (1024 * 1024)
                                    speed_label.config(text=f"Speed: {speed_mb:.1f} MB/s")
                                
                                mb_downloaded = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                size_label.config(text=f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB")
                                progress_label.config(text=f"Downloading: {progress:.1f}%")
                            
                            time.sleep(0.5)
                        
                        download_thread.join(timeout=5)
                        
                        if download_error[0]:
                            raise download_error[0]
                        
                        if download_complete.is_set() and not download_error[0]:
                            progress_var.set(100)
                            progress_label.config(text="✅ Download complete!")
                            status_label.config(text="Model files downloaded")
                            add_log("✅ Model files downloaded successfully")
                            add_log("")
                            add_log("Next step: Click 'Load Model' to initialize manga-ocr")
                            # Schedule status check on main thread
                            self.update_queue.put(('call_method', self._check_provider_status, ()))
                        else:
                            raise Exception("Download was cancelled")
                            
                    except ImportError:
                        progress_label.config(text="❌ Missing huggingface_hub")
                        status_label.config(text="Install huggingface_hub first")
                        add_log("ERROR: huggingface_hub not installed")
                        add_log("Run: pip install huggingface_hub")
                    except Exception as e:
                        raise  # Re-raise to be caught by outer exception handler
                        
                elif provider == 'Qwen2-VL':
                    try:
                        from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
                        import torch
                    except ImportError as e:
                        progress_label.config(text="❌ Missing dependencies")
                        status_label.config(text="Install dependencies first")
                        add_log(f"ERROR: {str(e)}")
                        add_log("Please install manually:")
                        add_log("pip install transformers torch torchvision")
                        return
                    
                    progress_label.config(text=f"Downloading model...")
                    add_log(f"Starting download of {model_id}")
                    progress_var.set(10)
                    
                    add_log("Downloading processor...")
                    status_label.config(text="Downloading processor...")
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    progress_var.set(30)
                    add_log("✓ Processor downloaded")
                    
                    add_log("Downloading tokenizer...")
                    status_label.config(text="Downloading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    progress_var.set(50)
                    add_log("✓ Tokenizer downloaded")
                    
                    add_log("Downloading model weights (this may take several minutes)...")
                    status_label.config(text="Downloading model weights...")
                    progress_label.config(text="Downloading model weights...")
                    
                    if torch.cuda.is_available():
                        add_log(f"Using GPU: {torch.cuda.get_device_name(0)}")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        add_log("No GPU detected, will load on CPU")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            dtype=torch.float32,
                            trust_remote_code=True
                        )
                    
                    progress_var.set(90)
                    add_log("✓ Model weights downloaded")
                    
                    add_log("Initializing model...")
                    status_label.config(text="Initializing...")
                    
                    qwen_provider = self.ocr_manager.get_provider('Qwen2-VL')
                    if qwen_provider:
                        qwen_provider.processor = processor
                        qwen_provider.tokenizer = tokenizer  
                        qwen_provider.model = model
                        qwen_provider.model.eval()
                        qwen_provider.is_loaded = True
                        qwen_provider.is_installed = True
                        
                        if selected_model_key:
                            qwen_provider.loaded_model_size = selected_model_key
                    
                    progress_var.set(100)
                    progress_label.config(text="✅ Download complete!")
                    status_label.config(text="Model ready for Korean OCR!")
                    add_log("✓ Model ready to use!")
                    
                    # Schedule status check on main thread
                    self.update_queue.put(('call_method', self._check_provider_status, ()))
                    
                elif provider == 'rapidocr':
                    progress_label.config(text="📦 RapidOCR Installation Instructions")
                    add_log("RapidOCR requires manual pip installation")
                    progress_var.set(20)
                    
                    add_log("Command to run:")
                    add_log("pip install rapidocr-onnxruntime")
                    progress_var.set(50)
                    
                    add_log("")
                    add_log("After installation:")
                    add_log("1. Close this dialog")
                    add_log("2. Click 'Load Model' to initialize RapidOCR")
                    add_log("3. Status should show '✅ Model loaded'")
                    progress_var.set(100)
                    
                    progress_label.config(text="📦 Installation instructions shown")
                    status_label.config(text="Manual pip install required")
                    
                    download_btn.config(state=tk.DISABLED)
                    cancel_btn.config(text="Close")                        
                        
            except Exception as e:
                progress_label.config(text="❌ Download failed")
                status_label.config(text=f"Error: {str(e)[:50]}")
                add_log(f"ERROR: {str(e)}")
                self._log(f"Download error: {str(e)}", "error")
                
            finally:
                download_active['value'] = False
        
        def start_download():
            """Start download in background thread or executor"""
            download_btn.config(state=tk.DISABLED)
            cancel_btn.config(text="Cancel")
            
            try:
                if hasattr(self.main_gui, '_ensure_executor'):
                    self.main_gui._ensure_executor()
                execu = getattr(self.main_gui, 'executor', None)
                if execu:
                    execu.submit(download_with_progress)
                else:
                    import threading
                    download_thread = threading.Thread(target=download_with_progress, daemon=True)
                    download_thread.start()
            except Exception:
                import threading
                download_thread = threading.Thread(target=download_with_progress, daemon=True)
                download_thread.start()
        
        def cancel_download():
            """Cancel or close dialog"""
            if download_active['value']:
                download_active['value'] = False
                status_label.config(text="Cancelling...")
            else:
                download_dialog.destroy()
        
        download_btn = tb.Button(button_frame, text="Download", command=start_download, bootstyle="primary")
        download_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tb.Button(button_frame, text="Close", command=cancel_download, bootstyle="secondary")
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
        # Window sizing handled by Tkinter geometry
    
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
        if hasattr(self, 'download_model_btn'):
            self.download_model_btn.setVisible(False)
        
        if provider == 'google':
            # Google - check for credentials file
            google_creds = self.main_gui.config.get('google_vision_credentials', '')
            if google_creds and os.path.exists(google_creds):
                self.provider_status_label.setText("✅ Ready")
                self.provider_status_label.setStyleSheet("color: green;")
            else:
                self.provider_status_label.setText("❌ Credentials needed")
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
            bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', False)
            
            if api_key:
                if bubble_detection_enabled:
                    self.provider_status_label.setText("✅ Ready")
                    self.provider_status_label.setStyleSheet("color: green;")
                else:
                    self.provider_status_label.setText("⚠️ Enable AI bubble detection for best results")
                    self.provider_status_label.setStyleSheet("color: orange;")
            else:
                self.provider_status_label.setText("❌ API key needed")
                self.provider_status_label.setStyleSheet("color: red;")
     
        elif provider == 'Qwen2-VL':
            # Initialize OCR manager if needed
            if not hasattr(self, 'ocr_manager'):
                from ocr_manager import OCRManager
                self.ocr_manager = OCRManager(log_callback=self._log)
            
            # Check status first
            status = self.ocr_manager.check_provider_status(provider)
            
            # Load saved model size if available
            if hasattr(self, 'qwen2vl_model_size'):
                saved_model_size = self.qwen2vl_model_size
            else:
                saved_model_size = self.main_gui.config.get('qwen2vl_model_size', '1')
            
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
                self.download_model_btn.setText("📥 Download Model")
                self.download_model_btn.setVisible(True)
                
            else:
                # Not installed
                self.provider_status_label.setText("❌ Not installed")
                self.provider_status_label.setStyleSheet("color: red;")
                
                # Show BOTH buttons
                self.provider_setup_btn.setText("Load Model")
                self.provider_setup_btn.setVisible(True)
                
                self.download_model_btn.setText("📥 Download Qwen2-VL")
                self.download_model_btn.setVisible(True)
            
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
            if not hasattr(self, 'ocr_manager'):
                from ocr_manager import OCRManager
                self.ocr_manager = OCRManager(log_callback=self._log)
                
            status = self.ocr_manager.check_provider_status(provider)
            
            if status['loaded']:
                # Model is loaded and ready
                if provider == 'Qwen2-VL':
                    # Check which model size is loaded
                    qwen_provider = self.ocr_manager.get_provider('Qwen2-VL')
                    if qwen_provider and hasattr(qwen_provider, 'loaded_model_size'):
                        model_size = qwen_provider.loaded_model_size
                        status_text = f"✅ {model_size} model loaded"
                    else:
                        status_text = "✅ Model loaded"
                    self.provider_status_label.setText(status_text)
                    self.provider_status_label.setStyleSheet("color: green;")
                else:
                    self.provider_status_label.setText("✅ Model loaded")
                    self.provider_status_label.setStyleSheet("color: green;")
                
                # Show reload button for all local providers
                self.provider_setup_btn.setText("Reload")
                self.provider_setup_btn.setVisible(True)
                
            elif status['installed']:
                # Dependencies installed but model not loaded
                self.provider_status_label.setText("📦 Dependencies ready")
                self.provider_status_label.setStyleSheet("color: orange;")
                
                # Show Load button for all providers
                self.provider_setup_btn.setText("Load Model")
                self.provider_setup_btn.setVisible(True)
                
                # Also show Download button for models that need downloading
                if provider in ['Qwen2-VL', 'manga-ocr']:
                    self.download_model_btn.setText("📥 Download Model")
                    self.download_model_btn.setVisible(True)
                
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
                        self.download_model_btn.setText("📥 Install RapidOCR")
                    else:
                        self.download_model_btn.setText(f"📥 Download {provider}")
                    self.download_model_btn.setVisible(True)

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
        
        status = self.ocr_manager.check_provider_status(provider)
        
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
                    success = self.ocr_manager.install_provider(provider, update_progress)
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
                        success = self.ocr_manager.load_provider(provider, model_size=model_size)
                        
                        if success:
                            provider_obj = self.ocr_manager.get_provider('Qwen2-VL')
                            if provider_obj:
                                provider_obj.loaded_model_size = {
                                    "1": "2B",
                                    "2": "7B", 
                                    "3": "72B",
                                    "4": "custom"
                                }.get(model_size, model_size)
                    else:
                        self._log("Warning: No model size specified for Qwen2-VL, defaulting to 2B", "warning")
                        success = self.ocr_manager.load_provider(provider, model_size="1")
                else:
                    print(f"Loading {provider} without model_size parameter")
                    self._log(f"DEBUG: Loading {provider} without model_size parameter")
                    success = self.ocr_manager.load_provider(provider)
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
        
        # Save the selection
        self.main_gui.config['manga_ocr_provider'] = provider
        if hasattr(self.main_gui, 'save_config'):
            self.main_gui.save_config(show_message=False)
        
        # IMPORTANT: Reset translator to force recreation with new OCR provider
        if hasattr(self, 'translator') and self.translator:
            self._log(f"OCR provider changed to {provider.upper()}. Translator will be recreated on next run.", "info")
            self.translator = None  # Force recreation on next translation
    
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
        
        # Get current provider
        provider = self.ocr_provider_value if hasattr(self, 'ocr_provider_value') else self.main_gui.config.get('manga_ocr_provider', 'custom-api')
        
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
    
    def _build_pyside6_interface(self, main_layout):
        # Import QSizePolicy for layout management
        from PySide6.QtWidgets import QSizePolicy
        
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
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QLabel:disabled {
                color: #666666;
            }
        """
        self.parent_widget.setStyleSheet(checkbox_radio_style)
        
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
        
        # Get the saved OCR provider to check appropriate credentials
        saved_provider = self.main_gui.config.get('manga_ocr_provider', 'custom-api')
        
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
        
        # Store reference for updates
        self.status_label = status_label
        
        # Model Preloading Progress Bar (right after title, initially hidden)
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
        file_frame_font = QFont("Arial", 10)
        file_frame_font.setBold(True)
        file_frame.setFont(file_frame_font)
        file_frame_layout = QVBoxLayout(file_frame)
        file_frame_layout.setContentsMargins(10, 10, 10, 8)
        file_frame_layout.setSpacing(6)
        
        # File listbox (QListWidget handles scrolling automatically)
        self.file_listbox = QListWidget()
        self.file_listbox.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_listbox.setMinimumHeight(200)
        file_frame_layout.addWidget(self.file_listbox)
        
        # File buttons
        file_btn_frame = QWidget()
        file_btn_layout = QHBoxLayout(file_btn_frame)
        file_btn_layout.setContentsMargins(0, 6, 0, 0)
        file_btn_layout.setSpacing(4)
        
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self._add_files)
        add_files_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 3px 10px; font-size: 9pt; }")
        file_btn_layout.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self._add_folder)
        add_folder_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; padding: 3px 10px; font-size: 9pt; }")
        file_btn_layout.addWidget(add_folder_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        remove_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; padding: 3px 10px; font-size: 9pt; }")
        file_btn_layout.addWidget(remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        clear_btn.setStyleSheet("QPushButton { background-color: #ffc107; color: black; padding: 3px 10px; font-size: 9pt; }")
        file_btn_layout.addWidget(clear_btn)
        
        file_btn_layout.addStretch()
        file_frame_layout.addWidget(file_btn_frame)
        
        main_layout.addWidget(file_frame)
        
        # Create 2-column layout for settings
        columns_container = QWidget()
        columns_layout = QHBoxLayout(columns_container)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(10)
        
        # Left column (Column 1)
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(6)
        
        # Right column (Column 2)
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        right_column_layout.setContentsMargins(0, 0, 0, 0)
        right_column_layout.setSpacing(6)
        
        # Settings frame - GOES TO LEFT COLUMN
        settings_frame = QGroupBox("Translation Settings")
        settings_frame_font = QFont("Arial", 10)
        settings_frame_font.setBold(True)
        settings_frame.setFont(settings_frame_font)
        settings_frame_layout = QVBoxLayout(settings_frame)
        settings_frame_layout.setContentsMargins(10, 10, 10, 8)
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
        
        model_label = QLabel(f"Model: {current_model}")
        model_font = QFont("Arial", 10)
        model_font.setItalic(True)
        model_label.setFont(model_font)
        model_label.setStyleSheet("color: gray;")
        api_layout.addStretch()
        api_layout.addWidget(model_label)
        
        settings_frame_layout.addWidget(api_frame)

        # OCR Provider Selection - ENHANCED VERSION
        self.ocr_provider_frame = QWidget()
        ocr_provider_layout = QHBoxLayout(self.ocr_provider_frame)
        ocr_provider_layout.setContentsMargins(0, 0, 0, 10)
        ocr_provider_layout.setSpacing(10)

        provider_label = QLabel("OCR Provider:")
        provider_label.setMinimumWidth(150)
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

        self.ocr_provider_value = self.main_gui.config.get('manga_ocr_provider', 'custom-api')
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
        self.download_model_btn = QPushButton("📥 Download")
        self.download_model_btn.clicked.connect(self._download_hf_model)
        self.download_model_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 5px 15px; }")
        self.download_model_btn.setMinimumWidth(150)
        self.download_model_btn.setVisible(False)  # Hidden by default
        ocr_provider_layout.addWidget(self.download_model_btn)
        
        ocr_provider_layout.addStretch()
        settings_frame_layout.addWidget(self.ocr_provider_frame)

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
        
        context_title = QLabel("Main GUI Context Settings:")
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
        rolling_status = "Enabled (Rolling Window)" if self.main_gui.translation_history_rolling_var else "Disabled (Reset on Limit)"
        self.rolling_status_label = QLabel(f"• Rolling History: {rolling_status}")
        self.rolling_status_label.setFont(status_font)
        settings_display_layout.addWidget(self.rolling_status_label)
        
        context_info_layout.addWidget(settings_frame_display)
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
        self.context_checkbox.setChecked(bool(getattr(self, 'full_page_context_value', self.main_gui.config.get('manga_full_page_context', False))))
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
        
        # Output settings - moved here to be below visual context
        output_settings_frame = QWidget()
        output_settings_layout = QHBoxLayout(output_settings_frame)
        output_settings_layout.setContentsMargins(20, 10, 0, 0)
        output_settings_layout.setSpacing(10)
        
        self.create_subfolder_checkbox = self._create_styled_checkbox("Create 'translated' subfolder for output")
        self.create_subfolder_checkbox.setChecked(bool(getattr(self, 'create_subfolder_value', self.main_gui.config.get('manga_create_subfolder', True))))
        # Persist when toggled
        self.create_subfolder_checkbox.stateChanged.connect(self._on_create_subfolder_toggle)
        output_settings_layout.addWidget(self.create_subfolder_checkbox)
        output_settings_layout.addStretch()
        
        visual_layout.addWidget(output_settings_frame)
        
        context_frame_layout.addWidget(visual_frame)
        
        # Add the completed context_frame to settings_frame
        settings_frame_layout.addWidget(context_frame)
        
        # Add main settings frame to left column
        left_column_layout.addWidget(settings_frame)
        
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
        
        cloud_radio = QRadioButton("Cloud API")
        cloud_radio.setFont(radio_font)
        cloud_radio.setChecked(self.inpaint_method_value == 'cloud')
        cloud_radio.toggled.connect(lambda checked: self._on_inpaint_method_change() if checked else None)
        self.inpaint_method_group.addButton(cloud_radio, 0)
        method_selection_layout.addWidget(cloud_radio)

        local_radio = QRadioButton("Local Model")
        local_radio.setFont(radio_font)
        local_radio.setChecked(self.inpaint_method_value == 'local')
        local_radio.toggled.connect(lambda checked: self._on_inpaint_method_change() if checked else None)
        self.inpaint_method_group.addButton(local_radio, 1)
        method_selection_layout.addWidget(local_radio)

        hybrid_radio = QRadioButton("Hybrid")
        hybrid_radio.setFont(radio_font)
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
        local_model_combo = QComboBox()
        local_model_combo.addItems(['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'mat', 'ollama', 'sd_local'])
        local_model_combo.setCurrentText(self.local_model_type_value)
        local_model_combo.setMinimumWidth(120)
        local_model_combo.setMaximumWidth(120)
        local_combo_font = QFont('Arial', 9)
        local_model_combo.setFont(local_combo_font)
        local_model_combo.currentTextChanged.connect(self._on_local_model_change)
        self._disable_combobox_mousewheel(local_model_combo)  # Disable mousewheel scrolling
        local_model_layout.addWidget(local_model_combo)
        self.local_model_combo = local_model_combo

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
        }
        self.model_desc_label = QLabel(model_desc.get(self.local_model_type_value, ''))
        desc_font = QFont('Arial', 8)
        self.model_desc_label.setFont(desc_font)
        self.model_desc_label.setStyleSheet("color: gray;")
        self.model_desc_label.setMaximumWidth(200)
        local_model_layout.addWidget(self.model_desc_label)
        local_model_layout.addStretch()
        
        local_inpaint_layout.addWidget(local_model_frame)

        # Model file selection
        model_path_frame = QWidget()
        model_path_layout = QHBoxLayout(model_path_frame)
        model_path_layout.setContentsMargins(0, 5, 0, 0)
        model_path_layout.setSpacing(10)

        model_file_label = QLabel("Model File:")
        model_file_label_font = QFont('Arial', 9)
        model_file_label.setFont(model_file_label_font)
        model_file_label.setMinimumWidth(95)
        model_file_label.setAlignment(Qt.AlignLeft)
        model_path_layout.addWidget(model_file_label)
        self.model_file_label = model_file_label

        self.local_model_path_value = self.main_gui.config.get(f'manga_{self.local_model_type_value}_model_path', '')
        self.local_model_entry = QLineEdit(self.local_model_path_value)
        self.local_model_entry.setReadOnly(True)
        self.local_model_entry.setMinimumWidth(100)  # Reduced for better fit
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
        model_path_layout.addStretch()
        
        local_inpaint_layout.addWidget(model_path_frame)

        # Model status
        self.local_model_status_label = QLabel("")
        status_font = QFont('Arial', 9)
        self.local_model_status_label.setFont(status_font)
        local_inpaint_layout.addWidget(self.local_model_status_label)

        # Download model button
        download_model_btn = QPushButton("📥 Download Model")
        download_model_btn.clicked.connect(self._download_model)
        download_model_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        local_inpaint_layout.addWidget(download_model_btn)

        # Model info button
        model_info_btn = QPushButton("ℹ️ Model Info")
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

        # Initialize visibility based on current settings
        self._toggle_inpaint_visibility()
        
        # Add inpaint_group to render_frame
        render_frame_layout.addWidget(inpaint_group)
        
        # Add render_frame (inpainting only) to LEFT COLUMN
        left_column_layout.addWidget(render_frame)
        
        # Advanced Settings button at the TOP OF RIGHT COLUMN
        advanced_button_frame = QWidget()
        advanced_button_layout = QHBoxLayout(advanced_button_frame)
        advanced_button_layout.setContentsMargins(0, 0, 0, 10)
        advanced_button_layout.setSpacing(10)

        advanced_settings_desc = QLabel("Configure OCR, preprocessing, and performance options")
        desc_font = QFont("Arial", 9)
        advanced_settings_desc.setFont(desc_font)
        advanced_settings_desc.setStyleSheet("color: gray;")
        advanced_button_layout.addWidget(advanced_settings_desc)
        
        advanced_button_layout.addStretch()
        
        advanced_settings_btn = QPushButton("⚙️ Advanced Settings")
        advanced_settings_btn.clicked.connect(self._open_advanced_settings)
        advanced_settings_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; padding: 5px 15px; }")
        advanced_button_layout.addWidget(advanced_settings_btn)
        
        right_column_layout.addWidget(advanced_button_frame)
        
        # Background Settings - MOVED TO RIGHT COLUMN
        self.bg_settings_frame = QGroupBox("Background Settings")
        bg_settings_font = QFont("Arial", 10)
        bg_settings_font.setBold(True)
        self.bg_settings_frame.setFont(bg_settings_font)
        bg_settings_layout = QVBoxLayout(self.bg_settings_frame)
        bg_settings_layout.setContentsMargins(10, 10, 10, 10)
        bg_settings_layout.setSpacing(8)
        
        # Free text only background opacity toggle (applies BG opacity only to free-text regions)
        self.ft_only_checkbox = self._create_styled_checkbox("Free text only background opacity")
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
        self.min_size_spinbox.setMinimum(8)
        self.min_size_spinbox.setMaximum(20)
        self.min_size_spinbox.setValue(self.auto_min_size_value)
        self.min_size_spinbox.setMinimumWidth(100)
        self.min_size_spinbox.valueChanged.connect(lambda value: (setattr(self, 'auto_min_size_value', value), self._save_rendering_settings(), self._apply_rendering_settings()))
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
        self.max_size_spinbox.setMinimum(20)
        self.max_size_spinbox.setMaximum(100)
        self.max_size_spinbox.setValue(self.max_font_size_value)
        self.max_size_spinbox.setMinimumWidth(100)
        self.max_size_spinbox.valueChanged.connect(lambda value: (setattr(self, 'max_font_size_value', value), self._save_rendering_settings(), self._apply_rendering_settings()))
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

        # Max Lines
        row_ml = QWidget()
        ml_layout = QHBoxLayout(row_ml)
        ml_layout.setContentsMargins(0, 2, 0, 4)
        ml_layout.setSpacing(10)
        
        ml_label = QLabel("Max Lines:")
        ml_label.setMinimumWidth(110)
        ml_layout.addWidget(ml_label)
        
        self.max_lines_spinbox = QSpinBox()
        self.max_lines_spinbox.setMinimum(5)
        self.max_lines_spinbox.setMaximum(20)
        self.max_lines_spinbox.setValue(self.max_lines_value)
        self.max_lines_spinbox.setMinimumWidth(100)
        self.max_lines_spinbox.valueChanged.connect(lambda value: (setattr(self, 'max_lines_value', value), self._save_rendering_settings(), self._apply_rendering_settings()))
        self._disable_spinbox_mousewheel(self.max_lines_spinbox)
        ml_layout.addWidget(self.max_lines_spinbox)
        ml_layout.addStretch()
        
        sizing_group_layout.addWidget(row_ml)

        # Quick Presets (horizontal) merged into sizing group
        row_presets = QWidget()
        presets_layout = QHBoxLayout(row_presets)
        presets_layout.setContentsMargins(0, 6, 0, 2)
        presets_layout.setSpacing(10)
        
        presets_label = QLabel("Quick Presets:")
        presets_label.setMinimumWidth(110)
        presets_layout.addWidget(presets_label)
        
        small_preset_btn = QPushButton("Small Bubbles")
        small_preset_btn.setMinimumWidth(120)
        small_preset_btn.clicked.connect(lambda: self._set_font_preset('small'))
        presets_layout.addWidget(small_preset_btn)
        
        balanced_preset_btn = QPushButton("Balanced")
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
        self.shadow_enabled_checkbox.stateChanged.connect(lambda: (setattr(self, 'shadow_enabled_value', self.shadow_enabled_checkbox.isChecked()), self._toggle_shadow_controls(), self._save_rendering_settings(), self._apply_rendering_settings()))
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
                # Save settings to config
                self._save_rendering_settings()
        
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
        self.start_button.setMinimumHeight(90)  # Minimum height when space is constrained
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
                    self.setPixmap(rotated)
            
            def get_rotation(self):
                return self._rotation
            
            rotation = Property(float, get_rotation, set_rotation)
            
            def set_original_pixmap(self, pixmap):
                self._original_pixmap = pixmap
                self.setPixmap(pixmap)
        
        # Icon container
        icon_container = QWidget()
        icon_container.setFixedSize(100, 100)  # Increased size
        icon_container.setStyleSheet("background-color: transparent;")
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setAlignment(Qt.AlignCenter)
        
        self.start_button_icon = RotatableLabel(icon_container)
        self.start_button_icon.setStyleSheet("background-color: transparent;")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            available_sizes = icon.availableSizes()
            if available_sizes:
                largest_size = max(available_sizes, key=lambda s: s.width() * s.height())
                pixmap = icon.pixmap(largest_size)
            else:
                pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Increased size
            self.start_button_icon.set_original_pixmap(scaled_pixmap)
        self.start_button_icon.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(self.start_button_icon)
        
        button_layout.addWidget(icon_container)
        
        # Create animations
        self.start_icon_spin_animation = QPropertyAnimation(self.start_button_icon, b"rotation")
        self.start_icon_spin_animation.setDuration(2400)  # Even slower for smoother feel (2.4 seconds per rotation)
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
        control_layout.addWidget(self.start_button, stretch=1)  # Give it stretch factor

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
        
        # Add control buttons to LEFT COLUMN - with stretch so it expands to fill space
        left_column_layout.addWidget(control_frame, stretch=1)
        
        # Add stretch to right column to balance
        right_column_layout.addStretch()
        
        # Set size policies to make columns expand and shrink properly
        left_column.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        right_column.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Set minimum widths for columns to allow shrinking
        left_column.setMinimumWidth(300)
        right_column.setMinimumWidth(300)
        
        # Add columns to container with equal stretch factors
        columns_layout.addWidget(left_column, stretch=1)
        columns_layout.addWidget(right_column, stretch=1)
        
        # Make the columns container itself have proper size policy
        columns_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Add columns container to main layout
        main_layout.addWidget(columns_container)
        
        # Progress frame
        progress_frame = QGroupBox("Progress")
        progress_frame_font = QFont('Arial', 10)
        progress_frame_font.setBold(True)
        progress_frame.setFont(progress_frame_font)
        progress_frame_layout = QVBoxLayout(progress_frame)
        progress_frame_layout.setContentsMargins(10, 10, 10, 8)
        progress_frame_layout.setSpacing(6)
        
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
        
        # Connect scrollbar to detect manual scrolling
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_log_scroll)
        
        main_layout.addWidget(log_frame)
        
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
            MangaSettingsDialog(
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
        """Update shadow blur value label and value variable"""
        self.shadow_blur_value = int(float(value))  # UPDATE THE VALUE VARIABLE!
        try:
            if hasattr(self, 'shadow_blur_value_label'):
                self.shadow_blur_value_label.setText(f"{int(float(value))}")
        except Exception:
            pass
    
    def _on_ft_only_bg_opacity_changed(self):
        """Handle free text only background opacity checkbox change (PySide6)"""
        # Update the value from checkbox state
        self.free_text_only_bg_opacity_value = self.ft_only_checkbox.isChecked()
    
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
        """Update the shadow color preview"""
        r = self.shadow_color_r_value
        g = self.shadow_color_g_value
        b = self.shadow_color_b_value
        if hasattr(self, 'shadow_preview_frame'):
            self.shadow_preview_frame.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #5a9fd4;")
        # Auto-save and apply on change
        if event is not None:  # Only save on user interaction, not initial load
            self._save_rendering_settings()
            self._apply_rendering_settings()
    
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

    def _set_font_preset(self, preset: str):
        """Apply font sizing preset (moved from dialog)"""
        try:
            if preset == 'small':
                self.font_algorithm_value = 'conservative'
                self.auto_min_size_value = 10
                self.max_font_size_value = 32
                self.prefer_larger_value = False
                self.bubble_size_factor_value = True
                self.line_spacing_value = 1.2
                self.max_lines_value = 8
            elif preset == 'balanced':
                self.font_algorithm_value = 'smart'
                self.auto_min_size_value = 12
                self.max_font_size_value = 48
                self.prefer_larger_value = True
                self.bubble_size_factor_value = True
                self.line_spacing_value = 1.3
                self.max_lines_value = 10
            elif preset == 'large':
                self.font_algorithm_value = 'aggressive'
                self.auto_min_size_value = 14
                self.max_font_size_value = 64
                self.prefer_larger_value = True
                self.bubble_size_factor_value = False
                self.line_spacing_value = 1.4
                self.max_lines_value = 12
            
            # Update all spinboxes with new values
            if hasattr(self, 'min_size_spinbox'):
                self.min_size_spinbox.setValue(self.auto_min_size_value)
            if hasattr(self, 'max_size_spinbox'):
                self.max_size_spinbox.setValue(self.max_font_size_value)
            if hasattr(self, 'line_spacing_spinbox'):
                self.line_spacing_spinbox.setValue(self.line_spacing_value)
            if hasattr(self, 'max_lines_spinbox'):
                self.max_lines_spinbox.setValue(self.max_lines_value)
            
            # Update checkboxes
            if hasattr(self, 'prefer_larger_checkbox'):
                self.prefer_larger_checkbox.setChecked(self.prefer_larger_value)
            if hasattr(self, 'bubble_size_factor_checkbox'):
                self.bubble_size_factor_checkbox.setChecked(self.bubble_size_factor_value)
            
            # Update the line spacing label
            if hasattr(self, 'line_spacing_value_label'):
                self.line_spacing_value_label.setText(f"{float(self.line_spacing_value):.2f}")
            
            self._save_rendering_settings()
        except Exception as e:
            self._log(f"Error setting preset: {e}", "debug")
    
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
        
    def _load_rendering_settings(self):
        """Load text rendering settings from config"""
        config = self.main_gui.config

        # One-time migration for legacy min font size key
        try:
            legacy_min = config.get('manga_min_readable_size', None)
            if legacy_min is not None:
                ms = config.setdefault('manga_settings', {})
                rend = ms.setdefault('rendering', {})
                font = ms.setdefault('font_sizing', {})
                current_min = rend.get('auto_min_size', font.get('min_size'))
                if current_min is None or int(current_min) < int(legacy_min):
                    rend['auto_min_size'] = int(legacy_min)
                    font['min_size'] = int(legacy_min)
                # Remove legacy key
                try:
                    del config['manga_min_readable_size']
                except Exception:
                    pass
                # Persist migration silently
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
        except Exception:
            pass
        
        # Get inpainting settings from the nested location
        manga_settings = config.get('manga_settings', {})
        inpaint_settings = manga_settings.get('inpainting', {})
        
        # Load inpaint method from the correct location (no Tkinter variables in PySide6)
        self.inpaint_method_value = inpaint_settings.get('method', 'local')
        self.local_model_type_value = inpaint_settings.get('local_method', 'anime_onnx')
        
        # Load model paths
        self.local_model_path_value = ''
        for model_type in  ['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'mat', 'ollama', 'sd_local']:
            path = inpaint_settings.get(f'{model_type}_model_path', '')
            if model_type == self.local_model_type_value:
                self.local_model_path_value = path
        
        # Initialize with defaults (plain Python values, no Tkinter variables)
        self.bg_opacity_value = config.get('manga_bg_opacity', 130)
        self.free_text_only_bg_opacity_value = config.get('manga_free_text_only_bg_opacity', True)
        self.bg_style_value = config.get('manga_bg_style', 'circle')
        self.bg_reduction_value = config.get('manga_bg_reduction', 1.0)
        self.font_size_value = config.get('manga_font_size', 0)
        
        self.selected_font_path = config.get('manga_font_path', None)
        self.skip_inpainting_value = config.get('manga_skip_inpainting', False)
        self.inpaint_quality_value = config.get('manga_inpaint_quality', 'high')
        self.inpaint_dilation_value = config.get('manga_inpaint_dilation', 15)
        self.inpaint_passes_value = config.get('manga_inpaint_passes', 2)
        
        self.font_size_mode_value = config.get('manga_font_size_mode', 'fixed')
        self.font_size_multiplier_value = config.get('manga_font_size_multiplier', 1.0)
        
        # Auto fit style for auto mode
        try:
            rend_cfg = (config.get('manga_settings', {}) or {}).get('rendering', {})
        except Exception:
            rend_cfg = {}
        self.auto_fit_style_value = rend_cfg.get('auto_fit_style', 'balanced')
        
        # Auto minimum font size (from rendering or font_sizing)
        try:
            font_cfg = (config.get('manga_settings', {}) or {}).get('font_sizing', {})
        except Exception:
            font_cfg = {}
        auto_min_default = rend_cfg.get('auto_min_size', font_cfg.get('min_size', 10))
        self.auto_min_size_value = int(auto_min_default)
        
        self.force_caps_lock_value = config.get('manga_force_caps_lock', True)
        self.constrain_to_bubble_value = config.get('manga_constrain_to_bubble', True)
        
        # Advanced font sizing (from manga_settings.font_sizing)
        font_settings = (config.get('manga_settings', {}) or {}).get('font_sizing', {})
        self.font_algorithm_value = str(font_settings.get('algorithm', 'smart'))
        self.prefer_larger_value = bool(font_settings.get('prefer_larger', True))
        self.bubble_size_factor_value = bool(font_settings.get('bubble_size_factor', True))
        self.line_spacing_value = float(font_settings.get('line_spacing', 1.3))
        self.max_lines_value = int(font_settings.get('max_lines', 10))
        
        # Determine effective max font size with fallback
        font_max_top = config.get('manga_max_font_size', None)
        nested_ms = config.get('manga_settings', {}) if isinstance(config.get('manga_settings', {}), dict) else {}
        nested_render = nested_ms.get('rendering', {}) if isinstance(nested_ms.get('rendering', {}), dict) else {}
        nested_font = nested_ms.get('font_sizing', {}) if isinstance(nested_ms.get('font_sizing', {}), dict) else {}
        effective_max = font_max_top if font_max_top is not None else (
            nested_render.get('auto_max_size', nested_font.get('max_size', 48))
        )
        self.max_font_size_value = int(effective_max)
        
        # If top-level keys were missing, mirror max now (won't save during initialization)
        if font_max_top is None:
            self.main_gui.config['manga_max_font_size'] = int(effective_max)
        
        self.strict_text_wrapping_value = config.get('manga_strict_text_wrapping', True)
        
        # Font color settings
        manga_text_color = config.get('manga_text_color', [102, 0, 0])
        self.text_color_r_value = manga_text_color[0]
        self.text_color_g_value = manga_text_color[1]
        self.text_color_b_value = manga_text_color[2]
        
        # Shadow settings
        self.shadow_enabled_value = config.get('manga_shadow_enabled', True)
        
        manga_shadow_color = config.get('manga_shadow_color', [204, 128, 128])
        self.shadow_color_r_value = manga_shadow_color[0]
        self.shadow_color_g_value = manga_shadow_color[1]
        self.shadow_color_b_value = manga_shadow_color[2]
        
        self.shadow_offset_x_value = config.get('manga_shadow_offset_x', 2)
        self.shadow_offset_y_value = config.get('manga_shadow_offset_y', 2)
        self.shadow_blur_value = config.get('manga_shadow_blur', 0)
        
        # Initialize font_style with saved value or default
        self.font_style_value = config.get('manga_font_style', 'Default')
        
        # Full page context settings
        self.full_page_context_value = config.get('manga_full_page_context', False)
        
        self.full_page_context_prompt = config.get('manga_full_page_context_prompt', 
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
 
        # Load OCR prompt (UPDATED: Improved default)
        self.ocr_prompt = config.get('manga_ocr_prompt', 
            "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\n\n"
            "ABSOLUTE RULES:\n"
            "1. OUTPUT ONLY THE VISIBLE TEXT/SYMBOLS - NOTHING ELSE\n"
            "2. NEVER TRANSLATE OR MODIFY\n"
            "3. NEVER EXPLAIN, DESCRIBE, OR COMMENT\n"
            "4. NEVER SAY \"I can't\" or \"I cannot\" or \"no text\" or \"blank image\"\n"
            "5. IF YOU SEE DOTS, OUTPUT THE DOTS: .\n"
            "6. IF YOU SEE PUNCTUATION, OUTPUT THE PUNCTUATION\n"
            "7. IF YOU SEE A SINGLE CHARACTER, OUTPUT THAT CHARACTER\n"
            "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)\n\n"
            "LANGUAGE PRESERVATION:\n"
            "- Korean text → Output in Korean\n"
            "- Japanese text → Output in Japanese\n"
            "- Chinese text → Output in Chinese\n"
            "- English text → Output in English\n"
            "- CJK quotation marks (「」『』【】《》〈〉) → Preserve exactly as shown\n\n"
            "FORMATTING:\n"
            "- OUTPUT ALL TEXT ON A SINGLE LINE WITH NO LINE BREAKS\n"
            "- NEVER use \\n or line breaks in your output\n\n"
            "FORBIDDEN RESPONSES:\n"
            "- \"I can see this appears to be...\"\n"
            "- \"I cannot make out any clear text...\"\n"
            "- \"This appears to be blank...\"\n"
            "- \"If there is text present...\"\n"
            "- ANY explanatory text\n\n"
            "YOUR ONLY OUTPUT: The exact visible text. Nothing more. Nothing less.\n"
            "If image has a dot → Output: .\n"
            "If image has two dots → Output: . .\n"
            "If image has text → Output: [that text]\n"
            "If image is truly blank → Output: [empty/no response]"
        )
        # Visual context setting
        self.visual_context_enabled_value = self.main_gui.config.get('manga_visual_context_enabled', True)
        self.qwen2vl_model_size = config.get('qwen2vl_model_size', '1')  # Default to '1' (2B)
        
        # Initialize RapidOCR settings
        self.rapidocr_use_recognition_value = self.main_gui.config.get('rapidocr_use_recognition', True)
        self.rapidocr_language_value = self.main_gui.config.get('rapidocr_language', 'auto')
        self.rapidocr_detection_mode_value = self.main_gui.config.get('rapidocr_detection_mode', 'document')

        # Output settings
        self.create_subfolder_value = config.get('manga_create_subfolder', True)
    
    def _save_rendering_settings(self):
        """Save rendering settings with validation"""
        # Don't save during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
        
        # Before saving, refresh key toggle values from widgets if present
        try:
            if hasattr(self, 'context_checkbox'):
                self.full_page_context_value = bool(self.context_checkbox.isChecked())
            if hasattr(self, 'visual_context_checkbox'):
                self.visual_context_enabled_value = bool(self.visual_context_checkbox.isChecked())
            if hasattr(self, 'create_subfolder_checkbox'):
                self.create_subfolder_value = bool(self.create_subfolder_checkbox.isChecked())
        except Exception:
            pass
        
        # Validate that variables exist and have valid values before saving
        try:
            # Ensure manga_settings structure exists
            if 'manga_settings' not in self.main_gui.config:
                self.main_gui.config['manga_settings'] = {}
            if 'inpainting' not in self.main_gui.config['manga_settings']:
                self.main_gui.config['manga_settings']['inpainting'] = {}
            
            # Save to nested location
            inpaint = self.main_gui.config['manga_settings']['inpainting']
            if hasattr(self, 'inpaint_method_value'):
                inpaint['method'] = self.inpaint_method_value
            if hasattr(self, 'local_model_type_value'):
                inpaint['local_method'] = self.local_model_type_value
                model_type = self.local_model_type_value
                if hasattr(self, 'local_model_path_value'):
                    inpaint[f'{model_type}_model_path'] = self.local_model_path_value
            
            # Add new inpainting settings
            if hasattr(self, 'inpaint_method_value'):
                self.main_gui.config['manga_inpaint_method'] = self.inpaint_method_value
            if hasattr(self, 'local_model_type_value'):
                self.main_gui.config['manga_local_inpaint_model'] = self.local_model_type_value
            
            # Save model paths for each type
            for model_type in  ['aot', 'lama', 'lama_onnx', 'anime', 'mat', 'ollama', 'sd_local']:
                if hasattr(self, 'local_model_type_value'):
                    if model_type == self.local_model_type_value:
                        if hasattr(self, 'local_model_path_value'):
                            path = self.local_model_path_value
                            if path:
                                self.main_gui.config[f'manga_{model_type}_model_path'] = path
            
            # Save all other settings with validation
            if hasattr(self, 'bg_opacity_value'):
                self.main_gui.config['manga_bg_opacity'] = self.bg_opacity_value
            if hasattr(self, 'bg_style_value'):
                self.main_gui.config['manga_bg_style'] = self.bg_style_value
            if hasattr(self, 'bg_reduction_value'):
                self.main_gui.config['manga_bg_reduction'] = self.bg_reduction_value
            
            # Save free-text-only background opacity toggle
            if hasattr(self, 'free_text_only_bg_opacity_value'):
                self.main_gui.config['manga_free_text_only_bg_opacity'] = bool(self.free_text_only_bg_opacity_value)
            
            # CRITICAL: Font size settings - validate before saving
            if hasattr(self, 'font_size_value'):
                value = self.font_size_value
                self.main_gui.config['manga_font_size'] = value
            
            if hasattr(self, 'max_font_size_value'):
                value = self.max_font_size_value
                # Validate the value is reasonable
                if 0 <= value <= 200:
                    self.main_gui.config['manga_max_font_size'] = value
            
            # Mirror these into nested manga_settings so the dialog and integration stay in sync
            try:
                ms = self.main_gui.config.setdefault('manga_settings', {})
                rend = ms.setdefault('rendering', {})
                font = ms.setdefault('font_sizing', {})
                # Mirror bounds
                if hasattr(self, 'auto_min_size_value'):
                    rend['auto_min_size'] = int(self.auto_min_size_value)
                    font['min_size'] = int(self.auto_min_size_value)
                if hasattr(self, 'max_font_size_value'):
                    rend['auto_max_size'] = int(self.max_font_size_value)
                    font['max_size'] = int(self.max_font_size_value)
                # Persist advanced font sizing controls
                if hasattr(self, 'font_algorithm_value'):
                    font['algorithm'] = str(self.font_algorithm_value)
                if hasattr(self, 'prefer_larger_value'):
                    font['prefer_larger'] = bool(self.prefer_larger_value)
                if hasattr(self, 'bubble_size_factor_value'):
                    font['bubble_size_factor'] = bool(self.bubble_size_factor_value)
                if hasattr(self, 'line_spacing_value'):
                    font['line_spacing'] = float(self.line_spacing_value)
                if hasattr(self, 'max_lines_value'):
                    font['max_lines'] = int(self.max_lines_value)
                if hasattr(self, 'auto_fit_style_value'):
                    rend['auto_fit_style'] = str(self.auto_fit_style_value)
            except Exception:
                pass
            
            # Continue with other settings
            self.main_gui.config['manga_font_path'] = self.selected_font_path
            
            if hasattr(self, 'skip_inpainting_value'):
                self.main_gui.config['manga_skip_inpainting'] = self.skip_inpainting_value
            if hasattr(self, 'inpaint_quality_value'):
                self.main_gui.config['manga_inpaint_quality'] = self.inpaint_quality_value
            if hasattr(self, 'inpaint_dilation_value'):
                self.main_gui.config['manga_inpaint_dilation'] = self.inpaint_dilation_value
            if hasattr(self, 'inpaint_passes_value'):
                self.main_gui.config['manga_inpaint_passes'] = self.inpaint_passes_value
            if hasattr(self, 'font_size_mode_value'):
                self.main_gui.config['manga_font_size_mode'] = self.font_size_mode_value
            if hasattr(self, 'font_size_multiplier_value'):
                self.main_gui.config['manga_font_size_multiplier'] = self.font_size_multiplier_value
            if hasattr(self, 'font_style_value'):
                self.main_gui.config['manga_font_style'] = self.font_style_value
            if hasattr(self, 'constrain_to_bubble_value'):
                self.main_gui.config['manga_constrain_to_bubble'] = self.constrain_to_bubble_value
            if hasattr(self, 'strict_text_wrapping_value'):
                self.main_gui.config['manga_strict_text_wrapping'] = self.strict_text_wrapping_value
            if hasattr(self, 'force_caps_lock_value'):
                self.main_gui.config['manga_force_caps_lock'] = self.force_caps_lock_value
            
            # Save font color as list
            if hasattr(self, 'text_color_r_value') and hasattr(self, 'text_color_g_value') and hasattr(self, 'text_color_b_value'):
                self.main_gui.config['manga_text_color'] = [
                    self.text_color_r_value,
                    self.text_color_g_value,
                    self.text_color_b_value
                ]
            
            # Save shadow settings
            if hasattr(self, 'shadow_enabled_value'):
                self.main_gui.config['manga_shadow_enabled'] = self.shadow_enabled_value
            if hasattr(self, 'shadow_color_r_value') and hasattr(self, 'shadow_color_g_value') and hasattr(self, 'shadow_color_b_value'):
                self.main_gui.config['manga_shadow_color'] = [
                    self.shadow_color_r_value,
                    self.shadow_color_g_value,
                    self.shadow_color_b_value
                ]
            if hasattr(self, 'shadow_offset_x_value'):
                self.main_gui.config['manga_shadow_offset_x'] = self.shadow_offset_x_value
            if hasattr(self, 'shadow_offset_y_value'):
                self.main_gui.config['manga_shadow_offset_y'] = self.shadow_offset_y_value
            if hasattr(self, 'shadow_blur_value'):
                self.main_gui.config['manga_shadow_blur'] = self.shadow_blur_value
            
            # Save output settings
            if hasattr(self, 'create_subfolder_value'):
                self.main_gui.config['manga_create_subfolder'] = self.create_subfolder_value
            
            # Save full page context settings
            if hasattr(self, 'full_page_context_value'):
                self.main_gui.config['manga_full_page_context'] = self.full_page_context_value
            if hasattr(self, 'full_page_context_prompt'):
                self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
            
            # Persist visual context setting alongside other toggles
            if hasattr(self, 'visual_context_enabled_value'):
                self.main_gui.config['manga_visual_context_enabled'] = self.visual_context_enabled_value
            
            # OCR prompt
            if hasattr(self, 'ocr_prompt'):
                self.main_gui.config['manga_ocr_prompt'] = self.ocr_prompt
             
            # Qwen and custom models             
            if hasattr(self, 'qwen2vl_model_size'):
                self.main_gui.config['qwen2vl_model_size'] = self.qwen2vl_model_size

            # RapidOCR specific settings
            if hasattr(self, 'rapidocr_use_recognition_value'):
                self.main_gui.config['rapidocr_use_recognition'] = self.rapidocr_use_recognition_value
            if hasattr(self, 'rapidocr_detection_mode_value'):
                self.main_gui.config['rapidocr_detection_mode'] = self.rapidocr_detection_mode_value
            if hasattr(self, 'rapidocr_language_value'):
                self.main_gui.config['rapidocr_language'] = self.rapidocr_language_value

            # Auto-save to disk (PySide6 version - no Tkinter black window issue)
            # Settings are stored in self.main_gui.config and persisted immediately
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            
            # Reinitialize environment variables to reflect the new settings
            if hasattr(self.main_gui, 'initialize_environment_variables'):
                self.main_gui.initialize_environment_variables()
                
        except Exception as e:
            # Log error but don't crash
            print(f"Error saving manga settings: {e}")
    
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
                                        QPushButton, QHBoxLayout)
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
            
            self._save_rendering_settings()
            self._log("✅ Updated prompts", "success")
            dialog.accept()
        
        def reset_prompt():
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
            
            # UPDATED: Improved OCR prompt (matches ocr_manager.py)
            default_ocr = (
                "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\n\n"
                "ABSOLUTE RULES:\n"
                "1. OUTPUT ONLY THE VISIBLE TEXT/SYMBOLS - NOTHING ELSE\n"
                "2. NEVER TRANSLATE OR MODIFY\n"
                "3. NEVER EXPLAIN, DESCRIBE, OR COMMENT\n"
                "4. NEVER SAY \"I can't\" or \"I cannot\" or \"no text\" or \"blank image\"\n"
                "5. IF YOU SEE DOTS, OUTPUT THE DOTS: .\n"
                "6. IF YOU SEE PUNCTUATION, OUTPUT THE PUNCTUATION\n"
                "7. IF YOU SEE A SINGLE CHARACTER, OUTPUT THAT CHARACTER\n"
                "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)\n\n"
                "LANGUAGE PRESERVATION:\n"
                "- Korean text → Output in Korean\n"
                "- Japanese text → Output in Japanese\n"
                "- Chinese text → Output in Chinese\n"
                "- English text → Output in English\n"
                "- CJK quotation marks (「」『』【】《》〈〉) → Preserve exactly as shown\n\n"
                "FORMATTING:\n"
                "- OUTPUT ALL TEXT ON A SINGLE LINE WITH NO LINE BREAKS\n"
                "- NEVER use \\n or line breaks in your output\n\n"
                "FORBIDDEN RESPONSES:\n"
                "- \"I can see this appears to be...\"\n"
                "- \"I cannot make out any clear text...\"\n"
                "- \"This appears to be blank...\"\n"
                "- \"If there is text present...\"\n"
                "- ANY explanatory text\n\n"
                "YOUR ONLY OUTPUT: The exact visible text. Nothing more. Nothing less.\n"
                "If image has a dot → Output: .\n"
                "If image has two dots → Output: . .\n"
                "If image has text → Output: [that text]\n"
                "If image is truly blank → Output: [empty/no response]"
            )
            ocr_editor.setPlainText(default_ocr)
        
        # Button layout
        button_layout = QHBoxLayout()
        
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
        
        # Show dialog
        dialog.exec()
    
    def _refresh_context_settings_with_feedback(self):
        """Refresh context settings from main GUI with visual feedback"""
        from PySide6.QtCore import QTimer
        
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
        """Refresh context settings from main GUI"""
        # Actually fetch the current values from main GUI
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
        
        if hasattr(self.main_gui, 'translation_history_rolling_var'):
            rolling_enabled = self.main_gui.translation_history_rolling_var
            rolling_status = "Enabled (Rolling Window)" if rolling_enabled else "Disabled (Reset on Limit)"
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
                
                if hasattr(self.main_gui, 'translation_history_rolling_var'):
                    self.translator.history_manager.rolling_enabled = self.main_gui.translation_history_rolling_var
                
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
        print("\n" + "="*80)
        print("TOGGLE FUNCTION CALLED!")
        print(f"Checkbox state: {self.skip_inpainting_checkbox.isChecked() if hasattr(self, 'skip_inpainting_checkbox') else 'NO CHECKBOX'}")
        print(f"Initializing: {getattr(self, '_initializing', 'NO FLAG')}")
        print("="*80 + "\n")
        try:
            # Update the value from the checkbox
            self.skip_inpainting_value = self.skip_inpainting_checkbox.isChecked()
            
            # Simple enable/disable logic - no animations, no show/hide, no empty dialogs!
            if self.skip_inpainting_value:
                print("🚫 Skip Inpainter: ENABLED - Inpainting will be skipped")
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
                            if "Browse" in btn_text:
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
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR in toggle function: {e}")
            print(traceback.format_exc())

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

    def _on_local_model_change(self, new_model_type=None):
        """Handle model type change and auto-load if model exists"""
        # Get model type from combo box (PySide6)
        if new_model_type is None:
            model_type = self.local_model_combo.currentText()
        else:
            model_type = new_model_type
        
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
        }
        self.model_desc_label.setText(model_desc.get(model_type, ''))
        
        # Check for saved path for this model type
        saved_path = self.main_gui.config.get(f'manga_{model_type}_model_path', '')
        
        if saved_path and os.path.exists(saved_path):
            # Update the path display
            self.local_model_entry.setText(saved_path)
            self.local_model_path_value = saved_path
            self.local_model_status_label.setText("⏳ Loading saved model...")
            self.local_model_status_label.setStyleSheet("color: orange;")
            
            # Auto-load the model after a short delay using QTimer
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._try_load_model(model_type, saved_path))
        else:
            # Clear the path display
            self.local_model_entry.setText("")
            self.local_model_path_value = ""
            self.local_model_status_label.setText("No model loaded")
            self.local_model_status_label.setStyleSheet("color: gray;")
        
        self._save_rendering_settings()

    def _browse_local_model(self):
        """Browse for local inpainting model and auto-load"""
        from PySide6.QtWidgets import QFileDialog
        from PySide6.QtCore import QTimer
        
        model_type = self.local_model_type_value
        
        if model_type == 'sd_local':
            filter_str = "Model files (*.safetensors *.pt *.pth *.ckpt *.onnx);;SafeTensors (*.safetensors);;Checkpoint files (*.ckpt);;PyTorch models (*.pt *.pth);;ONNX models (*.onnx);;All files (*.*)"
        else:
            filter_str = "Model files (*.pt *.pth *.ckpt *.onnx);;Checkpoint files (*.ckpt);;PyTorch models (*.pt *.pth);;ONNX models (*.onnx);;All files (*.*)"
        
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
        
        try:
            model_type = self.local_model_type_value if hasattr(self, 'local_model_type_value') else None
            path = self.local_model_path_value if hasattr(self, 'local_model_path_value') else ''
            if not model_type or not path:
                QMessageBox.information(self.dialog, "Load Model", "Please select a model file first using the Browse button.")
                return
            # Defer to keep UI responsive using QTimer
            QTimer.singleShot(50, lambda: self._try_load_model(model_type, path))
        except Exception:
            pass

    def _try_load_model(self, method: str, model_path: str):
        """Try to load a model and update status without threading for now."""
        from PySide6.QtWidgets import QApplication
        
        try:
            # Show loading status immediately
            self.local_model_status_label.setText("⏳ Loading model...")
            self.local_model_status_label.setStyleSheet("color: orange;")
            QApplication.processEvents()  # Process pending events to update UI
            self.main_gui.append_log(f"⏳ Loading {method.upper()} model...")
            
            from local_inpainter import LocalInpainter
            success = False
            try:
                test_inpainter = LocalInpainter()
                success = test_inpainter.load_model_with_retry(method, model_path, force_reload=True)
                print(f"DEBUG: Model loading completed, success={success}")
            except Exception as e:
                print(f"DEBUG: Model loading exception: {e}")
                self.main_gui.append_log(f"❌ Error loading model: {e}")
                success = False
            
            # Update UI directly on main thread
            print(f"DEBUG: Updating UI, success={success}")
            if success:
                self.local_model_status_label.setText(f"✅ {method.upper()} model ready")
                self.local_model_status_label.setStyleSheet("color: green;")
                self.main_gui.append_log(f"✅ {method.upper()} model loaded successfully!")
                if hasattr(self, 'translator') and self.translator:
                    for attr in ('local_inpainter', '_last_local_method', '_last_local_model_path'):
                        if hasattr(self.translator, attr):
                            try:
                                delattr(self.translator, attr)
                            except Exception:
                                pass
            else:
                self.local_model_status_label.setText("⚠️ Model file found but failed to load")
                self.local_model_status_label.setStyleSheet("color: orange;")
                self.main_gui.append_log("⚠️ Model file found but failed to load")
            print(f"DEBUG: UI update completed")
            return success
        except Exception as e:
            try:
                self.local_model_status_label.setText(f"❌ Error: {str(e)[:50]}")
                self.local_model_status_label.setStyleSheet("color: red;")
            except Exception:
                pass
            self.main_gui.append_log(f"❌ Error loading model: {e}")
            return False
        
    def _update_local_model_status(self):
        """Update local model status display"""
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
                model_hash = hashlib.md5(path.encode()).hexdigest()[:8]
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

    def _download_model(self):
        """Actually download the model for the selected type"""
        from PySide6.QtWidgets import QMessageBox
        
        model_type = self.local_model_type_value
        
        # Define URLs for each model type
        model_urls = {
            'aot': 'https://huggingface.co/ogkalu/aot-inpainting-jit/resolve/main/aot_traced.pt',
            'aot_onnx': 'https://huggingface.co/ogkalu/aot-inpainting/resolve/main/aot.onnx',
            'lama': 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
            'lama_onnx': 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',  
            'anime': 'https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt',
            'anime_onnx': 'https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx',
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
        """Perform the actual download with progress indication"""
        import threading
        import requests
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QIcon
        
        # Create a progress dialog
        progress_dialog = QDialog(self.dialog)
        progress_dialog.setWindowTitle(f"Downloading {model_name.upper()} Model")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.21)  # 21% of screen width
        height = int(screen.height() * 0.14)  # 14% of screen height
        progress_dialog.setFixedSize(width, height)
        progress_dialog.setModal(True)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'halgakos.ico')
        if os.path.exists(icon_path):
            progress_dialog.setWindowIcon(QIcon(icon_path))
        
        layout = QVBoxLayout(progress_dialog)
        
        # Progress label
        progress_label = QLabel("⏳ Downloading...")
        progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(progress_label)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        layout.addWidget(progress_bar)
        
        # Status label
        status_label = QLabel("0%")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        # Cancel flag
        cancel_download = {'value': False}
        
        def on_cancel():
            cancel_download['value'] = True
            progress_dialog.close()
        
        progress_dialog.closeEvent = lambda event: on_cancel()
        
        def download_thread():
            import time
            try:
                # Download with progress and speed tracking
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                start_time = time.time()
                last_update = start_time
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if cancel_download['value']:
                            # Clean up partial file
                            f.close()
                            if os.path.exists(save_path):
                                os.remove(save_path)
                            return
                        
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress (throttle updates to every 0.1 seconds)
                            current_time = time.time()
                            if total_size > 0 and (current_time - last_update > 0.1):
                                last_update = current_time
                                elapsed = current_time - start_time
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                speed_mb = speed / (1024 * 1024)
                                progress = (downloaded / total_size) * 100
                                
                                # Direct widget updates
                                try:
                                    progress_bar.setValue(int(progress))
                                    status_label.setText(f"{progress:.1f}% - {speed_mb:.2f} MB/s")
                                    progress_label.setText(f"⏳ Downloading... {downloaded//1024//1024}MB / {total_size//1024//1024}MB")
                                except RuntimeError:
                                    # Widget was destroyed, exit
                                    cancel_download['value'] = True
                                    return
                
                # Success - direct call
                try:
                    progress_dialog.close()
                    self._download_complete(save_path, model_name)
                except Exception as e:
                    print(f"Error in download completion: {e}")
                
            except requests.exceptions.RequestException as e:
                # Error - direct call
                if not cancel_download['value']:
                    try:
                        progress_dialog.close()
                        self._download_failed(str(e))
                    except Exception as ex:
                        print(f"Error handling download failure: {ex}")
            except Exception as e:
                if not cancel_download['value']:
                    try:
                        progress_dialog.close()
                        self._download_failed(str(e))
                    except Exception as ex:
                        print(f"Error handling download failure: {ex}")
        
        # Start download in background thread
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()
        
        # Show dialog
        progress_dialog.exec()

    def _download_complete(self, save_path: str, model_name: str):
        """Handle successful download"""
        from PySide6.QtWidgets import QMessageBox
        
        # Update the model path entry
        self.local_model_entry.setText(save_path)
        self.local_model_path_value = save_path
        
        # Save to config
        self.main_gui.config[f'manga_{model_name}_model_path'] = save_path
        self._save_rendering_settings()
        
        # Log to main GUI
        self.main_gui.append_log(f"✅ Downloaded {model_name} model to: {save_path}")
        
        # Auto-load the downloaded model (direct call)
        self.local_model_status_label.setText("⏳ Loading downloaded model...")
        self.local_model_status_label.setStyleSheet("color: orange;")
        
        # Try to load immediately
        if self._try_load_model(model_name, save_path):
            QMessageBox.information(self.dialog, "Success", f"{model_name.upper()} model downloaded and loaded!")
        else:
            QMessageBox.information(self.dialog, "Download Complete", f"{model_name.upper()} model downloaded but needs manual loading")

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
                        for fn in sorted(files_in_dir):
                            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                                target_path = os.path.join(root, fn)
                                if target_path not in self.selected_files:
                                    self.selected_files.append(target_path)
                                    self.file_listbox.addItem(os.path.basename(target_path))
                                    added += 1
                                # Map extracted image to its CBZ job
                                self.cbz_image_to_job[target_path] = path
                    self._log(f"📦 Added {added} images from CBZ: {os.path.basename(path)}", "info")
                except Exception as e:
                    self._log(f"❌ Failed to read CBZ {os.path.basename(path)}: {e}", "error")
            else:
                if path not in self.selected_files:
                    self.selected_files.append(path)
                    self.file_listbox.addItem(os.path.basename(path))
    
    def _add_folder(self):
        """Add all images (and CBZ archives) from a folder"""
        from PySide6.QtWidgets import QFileDialog
        
        folder = QFileDialog.getExistingDirectory(
            self.dialog,
            "Select Folder with Manga Images or CBZ"
        )
        if not folder:
            return
        
        # Extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        cbz_ext = '.cbz'
        
        # Ensure temp root for CBZ extraction lives for the session
        cbz_temp_root = getattr(self, 'cbz_temp_root', None)
        if cbz_temp_root is None:
            try:
                import tempfile
                cbz_temp_root = tempfile.mkdtemp(prefix='glossarion_cbz_')
                self.cbz_temp_root = cbz_temp_root
            except Exception:
                cbz_temp_root = None
        
        for filename in sorted(os.listdir(folder)):
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                continue
            lower = filename.lower()
            if any(lower.endswith(ext) for ext in image_extensions):
                if filepath not in self.selected_files:
                    self.selected_files.append(filepath)
                    self.file_listbox.addItem(filename)
            elif lower.endswith(cbz_ext):
                # Extract images from CBZ archive
                try:
                    import zipfile, shutil
                    base = os.path.splitext(os.path.basename(filepath))[0]
                    extract_dir = os.path.join(self.cbz_temp_root or folder, base)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(filepath, 'r') as zf:
                        zf.extractall(extract_dir)
                    # Initialize CBZ job tracking
                    if not hasattr(self, 'cbz_jobs'):
                        self.cbz_jobs = {}
                    if not hasattr(self, 'cbz_image_to_job'):
                        self.cbz_image_to_job = {}
                    # Prepare output dir next to source CBZ
                    out_dir = os.path.join(os.path.dirname(filepath), f"{base}_translated")
                    self.cbz_jobs[filepath] = {
                        'extract_dir': extract_dir,
                        'out_dir': out_dir,
                    }
                    # Collect all images recursively
                    added = 0
                    for root, _, files_in_dir in os.walk(extract_dir):
                        for fn in sorted(files_in_dir):
                            if fn.lower().endswith(tuple(image_extensions)):
                                target_path = os.path.join(root, fn)
                                if target_path not in self.selected_files:
                                    self.selected_files.append(target_path)
                                    self.file_listbox.addItem(os.path.basename(target_path))
                                    added += 1
                                # Map extracted image to its CBZ job
                                self.cbz_image_to_job[target_path] = filepath
                    self._log(f"📦 Added {added} images from CBZ: {filename}", "info")
                except Exception as e:
                    self._log(f"❌ Failed to read CBZ {filename}: {e}", "error")
    
    def _remove_selected(self):
        """Remove selected files from the list"""
        selected_items = self.file_listbox.selectedItems()
        
        if not selected_items:
            return
        
        # Remove in reverse order to maintain indices
        for item in selected_items:
            row = self.file_listbox.row(item)
            self.file_listbox.takeItem(row)
            if 0 <= row < len(self.selected_files):
                del self.selected_files[row]
    
    def _clear_all(self):
        """Clear all files from the list"""
        self.file_listbox.clear()
        self.selected_files.clear()
    
    def _finalize_cbz_jobs(self):
        """Package translated outputs back into .cbz for each imported CBZ.
        - Always creates a CLEAN archive with only final translated pages.
        - If save_intermediate is enabled in settings, also creates a DEBUG archive that
          contains the same final pages at root plus debug/raw artifacts under subfolders.
        """
        try:
            if not hasattr(self, 'cbz_jobs') or not self.cbz_jobs:
                return
            import zipfile
            # Read debug flag from settings
            save_debug = False
            try:
                save_debug = bool(self.main_gui.config.get('manga_settings', {}).get('advanced', {}).get('save_intermediate', False))
            except Exception:
                save_debug = False
            image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            text_exts = ('.txt', '.json', '.csv', '.log')
            excluded_patterns = ('_mask', '_overlay', '_debug', '_raw', '_ocr', '_regions', '_chunk', '_clean', '_cleaned', '_inpaint', '_inpainted')

            for cbz_path, job in self.cbz_jobs.items():
                out_dir = job.get('out_dir')
                if not out_dir or not os.path.isdir(out_dir):
                    continue
                parent = os.path.dirname(cbz_path)
                base = os.path.splitext(os.path.basename(cbz_path))[0]

                # Compute original basenames from extracted images mapping
                original_basenames = set()
                try:
                    if hasattr(self, 'cbz_image_to_job'):
                        for img_path, job_path in self.cbz_image_to_job.items():
                            if job_path == cbz_path:
                                original_basenames.add(os.path.basename(img_path))
                except Exception:
                    pass

                # Helper to iterate files in out_dir
                all_files = []
                for root, _, files in os.walk(out_dir):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        rel = os.path.relpath(fp, out_dir)
                        all_files.append((fp, rel, fn))

                # 1) CLEAN ARCHIVE: only final images matching original basenames
                clean_zip = os.path.join(parent, f"{base}_translated.cbz")
                clean_count = 0
                with zipfile.ZipFile(clean_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fp, rel, fn in all_files:
                        fn_lower = fn.lower()
                        if not fn_lower.endswith(image_exts):
                            continue
                        if original_basenames and fn not in original_basenames:
                            # Only include pages corresponding to original entries
                            continue
                        # Also skip obvious debug artifacts by pattern (extra safeguard)
                        if any(p in fn_lower for p in excluded_patterns):
                            continue
                        zf.write(fp, fn)  # place at root with page filename
                        clean_count += 1
                self._log(f"📦 Compiled CLEAN {clean_count} pages into {os.path.basename(clean_zip)}", "success")

                # 2) DEBUG ARCHIVE: include final pages + extras under subfolders
                if save_debug:
                    debug_zip = os.path.join(parent, f"{base}_translated_debug.cbz")
                    dbg_count = 0
                    raw_count = 0
                    page_count = 0
                    with zipfile.ZipFile(debug_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fp, rel, fn in all_files:
                            fn_lower = fn.lower()
                            # Final pages at root
                            if fn_lower.endswith(image_exts) and (not original_basenames or fn in original_basenames) and not any(p in fn_lower for p in excluded_patterns):
                                zf.write(fp, fn)
                                page_count += 1
                                continue
                            # Raw text/logs
                            if fn_lower.endswith(text_exts):
                                zf.write(fp, os.path.join('raw', rel))
                                raw_count += 1
                                continue
                            # Other images or artifacts -> debug/
                            zf.write(fp, os.path.join('debug', rel))
                            dbg_count += 1
                    self._log(f"📦 Compiled DEBUG archive: pages={page_count}, debug_files={dbg_count}, raw={raw_count} -> {os.path.basename(debug_zip)}", "info")
        except Exception as e:
            self._log(f"⚠️ Failed to compile CBZ packages: {e}", "warning")

    def _attach_logging_bridge(self):
        """Attach a root logging handler that forwards records into the GUI log."""
        try:
            if getattr(self, '_gui_log_handler', None) is None:
                handler = _MangaGuiLogHandler(self, level=logging.INFO)
                root_logger = logging.getLogger()
                # Avoid duplicates
                if all(not isinstance(h, _MangaGuiLogHandler) for h in root_logger.handlers):
                    root_logger.addHandler(handler)
                self._gui_log_handler = handler
                # Ensure common module loggers propagate
                for name in ['bubble_detector', 'local_inpainter', 'manga_translator']:
                    try:
                        lg = logging.getLogger(name)
                        lg.setLevel(logging.INFO)
                        lg.propagate = True
                    except Exception:
                        pass
        except Exception:
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
        """Detect when user manually scrolls up in the log"""
        try:
            scrollbar = self.log_text.verticalScrollBar()
            # If user scrolled up (not at bottom), mark it
            at_bottom = value >= scrollbar.maximum() - 10
            if not at_bottom:
                self._user_scrolled_up = True
            else:
                # User scrolled back to bottom, resume auto-scroll
                self._user_scrolled_up = False
        except Exception:
            pass
    
    def _start_autoscroll_delay(self, ms=500):
        """Delay auto-scroll for the specified milliseconds"""
        try:
            import time as _time
            self._autoscroll_delay_until = _time.time() + (ms / 1000.0)
            # Reset manual scroll flag when starting new operation
            self._user_scrolled_up = False
        except Exception:
            self._autoscroll_delay_until = 0.0
    
    def _log(self, message: str, level: str = "info"):
        """Log message to GUI text widget or console with enhanced stop suppression"""
        # Enhanced stop suppression - allow only essential stop confirmation messages
        if self._is_stop_requested() or self.is_globally_cancelled():
            # Only allow very specific stop confirmation messages - nothing else
            essential_stop_keywords = [
                "⏹️ Translation stopped by user",
                "🧹 Cleaning up models to free RAM",
                "✅ Model cleanup complete - RAM should be freed",
                "✅ All models cleaned up - RAM freed!"
            ]
            # Suppress ALL other messages when stopped - be very restrictive
            if not any(keyword in message for keyword in essential_stop_keywords):
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
                    
                    # Scroll to bottom (respect delay and manual scrolling)
                    try:
                        import time as _time
                        # Only auto-scroll if delay passed AND user hasn't scrolled up
                        if (_time.time() >= getattr(self, '_autoscroll_delay_until', 0) and 
                            not getattr(self, '_user_scrolled_up', False)):
                            self.log_text.ensureCursorVisible()
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
    
    def _update_progress(self, current: int, total: int, status: str):
        """Thread-safe progress update"""
        self.update_queue.put(('progress', current, total, status))
    
    def _update_current_file(self, filename: str):
        """Thread-safe current file update"""
        self.update_queue.put(('current_file', filename))
    
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
                        # Force update to ensure it's visible
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents()
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

    def _stop_startup_heartbeat(self):
        """Stop the startup heartbeat spinner"""
        try:
            self._startup_heartbeat_running = False
            # Clear the spinner text immediately
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.setText("Initializing...")
                self.progress_label.setStyleSheet("color: white;")
        except Exception:
            pass
    
    def _process_updates(self):
        """Process queued GUI updates"""
        try:
            while True:
                update = self.update_queue.get_nowait()
                
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
                        
                        # Scroll to bottom (respect delay and manual scrolling)
                        try:
                            import time as _time
                            # Only auto-scroll if delay passed AND user hasn't scrolled up
                            if (_time.time() >= getattr(self, '_autoscroll_delay_until', 0) and 
                                not getattr(self, '_user_scrolled_up', False)):
                                self.log_text.ensureCursorVisible()
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
                    _, state = update
                    if state == 'translation_started':
                        try:
                            # REMOVED: Don't disable start button - it's now a toggle button that should stay red/enabled
                            # The button is already updated to Stop state in _start_translation()
                            # Just disable file list to prevent modification during translation
                            if hasattr(self, 'file_listbox') and self.file_listbox:
                                self.file_listbox.setEnabled(False)
                        except Exception:
                            pass
                    elif state == 'translation_complete':
                        try:
                            # Reset UI to ready state when translation completes
                            self._reset_ui_state()
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
                    
        except Exception:
            # Queue is empty or some other exception
            pass
        
        # Schedule next update with QTimer
        QTimer.singleShot(100, self._process_updates)

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
        if self.is_running:
            self._stop_translation()
        else:
            self._start_translation()
    
    def _start_translation(self):
        """Start the translation process"""
        # Check files BEFORE redirecting stdout to avoid deadlock
        if not self.selected_files:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self.dialog, "No Files", "Please select manga images to translate.")
            return
        
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
                # Force immediate GUI update
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
                # Start spinning animation after 6 second delay
                if hasattr(self, 'start_icon_spin_animation') and hasattr(self, 'start_button_icon'):
                    if self.start_icon_spin_animation.state() != QPropertyAnimation.Running:
                        def start_spinning():
                            if hasattr(self, 'start_icon_spin_animation') and self.is_running:
                                self.start_icon_spin_animation.start()
                                # Start refresh timer to keep animation smooth
                                if hasattr(self, '_animation_refresh_timer'):
                                    self._animation_refresh_timer.start()
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(6000, start_spinning)  # 6 second delay
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
        
        # CRITICAL: Set is_running=True IMMEDIATELY so toggle button works
        self.is_running = True
        if hasattr(self, 'stop_flag'):
            self.stop_flag.clear()
        self._reset_global_cancellation()
        
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
                            # Only auto-scroll if user hasn't manually scrolled up
                            import time as _time
                            if (_time.time() >= getattr(self, '_autoscroll_delay_until', 0) and 
                                not getattr(self, '_user_scrolled_up', False)):
                                self.log_text.moveCursor(QTextCursor.End)
                                self.log_text.ensureCursorVisible()
                                # Also scroll the parent scroll area if it exists
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
        
        # Force GUI update
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass
        
        # Begin periodic demotion of background threads while translation runs (Windows only)
        try:
            self._start_periodic_thread_demoter()
        except Exception:
            pass
        
        # Run the heavy preparation and kickoff in a background thread to avoid GUI freeze
        threading.Thread(target=self._start_translation_heavy, name="MangaStartHeavy", daemon=True).start()
        return
    
    def _start_translation_heavy(self):
        """Heavy part of start: build configs, init client/translator, and launch worker (runs off-main-thread)."""
        try:
            # Lower priority & restrict affinity for this launcher thread (Windows)
            try:
                _lower_current_thread_priority_and_affinity('MANGA_RESERVE_CORES')
            except Exception:
                pass
            # Set thread limits based on parallel processing settings
            try:
                advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                parallel_enabled = advanced.get('parallel_processing', False)
                
                if parallel_enabled:
                    # Allow multiple threads for parallel processing
                    num_threads = advanced.get('max_workers', 4)
                    import os
                    os.environ['OMP_NUM_THREADS'] = str(num_threads)
                    os.environ['MKL_NUM_THREADS'] = str(num_threads)
                    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
                    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
                    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
                    os.environ['ONNXRUNTIME_NUM_THREADS'] = str(num_threads)
                    try:
                        import torch
                        torch.set_num_threads(num_threads)
                    except ImportError:
                        pass
                    try:
                        import cv2
                        cv2.setNumThreads(num_threads)
                    except (ImportError, AttributeError):
                        pass
                    self._log(f"⚡ Thread limit: {num_threads} threads (parallel processing enabled)", "debug")
                else:
                    # HARDCODED: Limit to exactly 1 thread for sequential processing
                    import os
                    os.environ['OMP_NUM_THREADS'] = '1'
                    os.environ['MKL_NUM_THREADS'] = '1'
                    os.environ['OPENBLAS_NUM_THREADS'] = '1'
                    os.environ['NUMEXPR_NUM_THREADS'] = '1'
                    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
                    os.environ['ONNXRUNTIME_NUM_THREADS'] = '1'
                    try:
                        import torch
                        torch.set_num_threads(1)  # Hardcoded to 1
                    except ImportError:
                        pass
                    try:
                        import cv2
                        cv2.setNumThreads(1)  # Limit OpenCV to 1 thread
                    except (ImportError, AttributeError):
                        pass
                    self._log("⚡ Thread limit: 1 thread (sequential processing)", "debug")
            except Exception as e:
                self._log(f"⚠️ Warning: Could not set thread limits: {e}", "warning")
            
            # Early feedback
            self._log("⏳ Preparing configuration...", "info")
            
            # Reload OCR prompt from config (in case it was edited in the dialog)
            if 'manga_ocr_prompt' in self.main_gui.config:
                self.ocr_prompt = self.main_gui.config['manga_ocr_prompt']
                self._log(f"✅ Loaded OCR prompt from config ({len(self.ocr_prompt)} chars)", "info")
                self._log(f"OCR Prompt preview: {self.ocr_prompt[:100]}...", "debug")
            else:
                self._log("⚠️ manga_ocr_prompt not found in config, using default", "warning")
            
            # Build OCR configuration
            ocr_config = {'provider': self.ocr_provider_value}

            if ocr_config['provider'] == 'Qwen2-VL':
                qwen_provider = self.ocr_manager.get_provider('Qwen2-VL')
                if qwen_provider:
                    # Set model size configuration
                    if hasattr(qwen_provider, 'loaded_model_size'):
                        if qwen_provider.loaded_model_size == "Custom":
                            ocr_config['model_size'] = f"custom:{qwen_provider.model_id}"
                        else:
                            size_map = {'2B': '1', '7B': '2', '72B': '3'}
                            ocr_config['model_size'] = size_map.get(qwen_provider.loaded_model_size, '2')
                        self._log(f"Setting ocr_config['model_size'] = {ocr_config['model_size']}", "info")
                    
                    # Set OCR prompt if available
                    if hasattr(self, 'ocr_prompt'):
                        # Set it via environment variable (Qwen2VL will read this)
                        os.environ['OCR_SYSTEM_PROMPT'] = self.ocr_prompt
                        
                        # Also set it directly on the provider if it has the method
                        if hasattr(qwen_provider, 'set_ocr_prompt'):
                            qwen_provider.set_ocr_prompt(self.ocr_prompt)
                        else:
                            # If no setter method, set it directly
                            qwen_provider.ocr_prompt = self.ocr_prompt
                        
                        self._log("✅ Set custom OCR prompt for Qwen2-VL", "info")
           
            elif ocr_config['provider'] == 'google':
                import os
                google_creds = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
                if not google_creds or not os.path.exists(google_creds):
                    self._log("❌ Google Cloud Vision credentials not found. Please set up credentials in the main settings.", "error")
                    self._stop_startup_heartbeat()
                    self._reset_ui_state()
                    return
                ocr_config['google_credentials_path'] = google_creds
                
            elif ocr_config['provider'] == 'azure':
                # Support both PySide6 QLineEdit (.text()) and Tkinter Entry (.get())
                if hasattr(self.azure_key_entry, 'text'):
                    azure_key = self.azure_key_entry.text().strip()
                elif hasattr(self.azure_key_entry, 'get'):
                    azure_key = self.azure_key_entry.get().strip()
                else:
                    azure_key = ''
                if hasattr(self.azure_endpoint_entry, 'text'):
                    azure_endpoint = self.azure_endpoint_entry.text().strip()
                elif hasattr(self.azure_endpoint_entry, 'get'):
                    azure_endpoint = self.azure_endpoint_entry.get().strip()
                else:
                    azure_endpoint = ''
                
                if not azure_key or not azure_endpoint:
                    self._log("❌ Azure credentials not configured.", "error")
                    self._stop_startup_heartbeat()
                    self._reset_ui_state()
                    return
                
                # Save Azure settings
                self.main_gui.config['azure_vision_key'] = azure_key
                self.main_gui.config['azure_vision_endpoint'] = azure_endpoint
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
                
                ocr_config['azure_key'] = azure_key
                ocr_config['azure_endpoint'] = azure_endpoint
            
            # Get current API key and model for translation
            api_key = None
            model = 'gemini-2.5-flash'  # default
            
            # Try to get API key from various sources (support PySide6 and Tkinter widgets)
            if hasattr(self.main_gui, 'api_key_entry'):
                try:
                    if hasattr(self.main_gui.api_key_entry, 'text'):
                        api_key_candidate = self.main_gui.api_key_entry.text()
                    elif hasattr(self.main_gui.api_key_entry, 'get'):
                        api_key_candidate = self.main_gui.api_key_entry.get()
                    else:
                        api_key_candidate = ''
                    if api_key_candidate and api_key_candidate.strip():
                        api_key = api_key_candidate.strip()
                except Exception:
                    pass
            if not api_key and hasattr(self.main_gui, 'config') and self.main_gui.config.get('api_key'):
                api_key = self.main_gui.config.get('api_key')
            
            # Try to get model - ALWAYS get the current selection from GUI
            # Support both PySide6 (plain string) and Tkinter (StringVar with .get())
            if hasattr(self.main_gui, 'model_var'):
                try:
                    # Check if it's a tkinter StringVar (has .get() method)
                    if hasattr(self.main_gui.model_var, 'get'):
                        model = self.main_gui.model_var.get()
                    else:
                        # PySide6 - model_var is just a string
                        model = self.main_gui.model_var
                except Exception as e:
                    print(f"[DEBUG] Error getting model from model_var: {e}")
                    model = 'gemini-2.5-flash'  # fallback
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
                model = self.main_gui.config.get('model')
            
            if not api_key:
                self._log("❌ API key not found. Please configure your API key in the main settings.", "error")
                self._stop_startup_heartbeat()
                self._reset_ui_state()
                return
            
            # Check if we need to create or update the client
            needs_new_client = False
            self._log("🔎 Checking API client...", "debug")
            
            if not hasattr(self.main_gui, 'client') or not self.main_gui.client:
                needs_new_client = True
                self._log(f"🛠 Creating new API client with model: {model}", "info")
            elif hasattr(self.main_gui.client, 'model') and self.main_gui.client.model != model:
                needs_new_client = True
                self._log(f"🛠 Model changed from {self.main_gui.client.model} to {model}, creating new client", "info")
            else:
                self._log("♻️ Reusing existing API client", "debug")
            
            if needs_new_client:
                # Apply multi-key settings from config so UnifiedClient picks them up
                try:
                    import os  # Import os here
                    use_mk = bool(self.main_gui.config.get('use_multi_api_keys', False))
                    mk_list = self.main_gui.config.get('multi_api_keys', [])
                    if use_mk and mk_list:
                        os.environ['USE_MULTI_API_KEYS'] = '1'
                        os.environ['USE_MULTI_KEYS'] = '1'  # backward-compat for retry paths
                        os.environ['MULTI_API_KEYS'] = json.dumps(mk_list)
                        os.environ['FORCE_KEY_ROTATION'] = '1' if self.main_gui.config.get('force_key_rotation', True) else '0'
                        os.environ['ROTATION_FREQUENCY'] = str(self.main_gui.config.get('rotation_frequency', 1))
                        self._log("🔑 Multi-key mode ENABLED for manga translator", "info")
                    else:
                        # Explicitly disable if not configured
                        os.environ['USE_MULTI_API_KEYS'] = '0'
                        os.environ['USE_MULTI_KEYS'] = '0'
                    # Fallback keys (optional)
                    if self.main_gui.config.get('use_fallback_keys', False):
                        os.environ['USE_FALLBACK_KEYS'] = '1'
                        os.environ['FALLBACK_KEYS'] = json.dumps(self.main_gui.config.get('fallback_keys', []))
                    else:
                        os.environ['USE_FALLBACK_KEYS'] = '0'
                        os.environ['FALLBACK_KEYS'] = '[]'
                except Exception as env_err:
                    self._log(f"⚠️ Failed to apply multi-key settings: {env_err}", "warning")
                
                # Create the unified client with the current model
                try:
                    from unified_api_client import UnifiedClient
                    self._log("⏳ Creating API client (network/model handshake)...", "debug")
                    self.main_gui.client = UnifiedClient(model=model, api_key=api_key)
                    self._log(f"✅ API client ready (model: {model})", "info")
                    try:
                        time.sleep(0.05)
                    except Exception:
                        pass
                except Exception as e:
                    self._log(f"❌ Failed to create API client: {str(e)}", "error")
                    import traceback
                    self._log(traceback.format_exc(), "debug")
                    self._stop_startup_heartbeat()
                    self._reset_ui_state()
                    return
            
            # Reset the translator's history manager for new batch
            if hasattr(self, 'translator') and self.translator and hasattr(self.translator, 'reset_history_manager'):
                self.translator.reset_history_manager()

            # Set environment variables for custom-api provider
            if ocr_config['provider'] == 'custom-api':
                import os  # Import os for environment variables
                env_vars = self.main_gui._get_environment_variables(
                    epub_path='',  # Not needed for manga
                    api_key=api_key
                )
                
                # Apply all environment variables EXCEPT SYSTEM_PROMPT
                for key, value in env_vars.items():
                    if key == 'SYSTEM_PROMPT':
                        # DON'T SET THE TRANSLATION SYSTEM PROMPT FOR OCR
                        continue
                    os.environ[key] = str(value)
                
                # Use custom OCR prompt from GUI if available, otherwise use default
                if hasattr(self, 'ocr_prompt') and self.ocr_prompt:
                    os.environ['OCR_SYSTEM_PROMPT'] = self.ocr_prompt
                    self._log(f"✅ Using custom OCR prompt from GUI ({len(self.ocr_prompt)} chars)", "info")
                    self._log(f"OCR Prompt being set: {self.ocr_prompt[:150]}...", "debug")
                else:
                    # Fallback to default OCR prompt
                    os.environ['OCR_SYSTEM_PROMPT'] = (
                    "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\n\n"
                    "ABSOLUTE RULES:\n"
                    "1. OUTPUT ONLY THE VISIBLE TEXT/SYMBOLS - NOTHING ELSE\n"
                    "2. NEVER TRANSLATE OR MODIFY\n"
                    "3. NEVER EXPLAIN, DESCRIBE, OR COMMENT\n"
                    "4. NEVER SAY \"I can't\" or \"I cannot\" or \"no text\" or \"blank image\"\n"
                    "5. IF YOU SEE DOTS, OUTPUT THE DOTS: .\n"
                    "6. IF YOU SEE PUNCTUATION, OUTPUT THE PUNCTUATION\n"
                    "7. IF YOU SEE A SINGLE CHARACTER, OUTPUT THAT CHARACTER\n"
                    "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)\n\n"
                    "LANGUAGE PRESERVATION:\n"
                    "- Korean text → Output in Korean\n"
                    "- Japanese text → Output in Japanese\n"
                    "- Chinese text → Output in Chinese\n"
                    "- English text → Output in English\n"
                    "- CJK quotation marks (「」『』【】《》〈〉) → Preserve exactly as shown\n\n"
                    "FORMATTING:\n"
                    "- OUTPUT ALL TEXT ON A SINGLE LINE WITH NO LINE BREAKS\n"
                    "- NEVER use \\n or line breaks in your output\n\n"
                    "FORBIDDEN RESPONSES:\n"
                    "- \"I can see this appears to be...\"\n"
                    "- \"I cannot make out any clear text...\"\n"
                    "- \"This appears to be blank...\"\n"
                    "- \"If there is text present...\"\n"
                    "- ANY explanatory text\n\n"
                    "YOUR ONLY OUTPUT: The exact visible text. Nothing more. Nothing less.\n"
                    "If image has a dot → Output: .\n"
                    "If image has two dots → Output: . .\n"
                    "If image has text → Output: [that text]\n"
                    "If image is truly blank → Output: [empty/no response]"
                    )
                    self._log("✅ Using default OCR prompt", "info")
                
                self._log("✅ Set environment variables for custom-api OCR (excluded SYSTEM_PROMPT)")
                
                # Respect user settings: only set default detector values when bubble detection is OFF.
                try:
                    ms = self.main_gui.config.setdefault('manga_settings', {})
                    ocr_set = ms.setdefault('ocr', {})
                    changed = False
                    bubble_enabled = bool(ocr_set.get('bubble_detection_enabled', False))
                    
                    if not bubble_enabled:
                        # User has bubble detection OFF -> set non-intrusive defaults only
                        if 'detector_type' not in ocr_set:
                            ocr_set['detector_type'] = 'rtdetr_onnx'
                            changed = True
                        if not ocr_set.get('rtdetr_model_url') and not ocr_set.get('bubble_model_path'):
                            # Default HF repo (detector.onnx lives here)
                            ocr_set['rtdetr_model_url'] = 'ogkalu/comic-text-and-bubble-detector'
                            changed = True
                        if changed and hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                    # Do not preload bubble detector for custom-api here; it will load on use or via panel preloading
                    self._preloaded_bd = None
                except Exception:
                    self._preloaded_bd = None
        except Exception as e:
            # Surface any startup error and reset UI so the app doesn't look stuck
            try:
                import traceback
                self._log(f"❌ Startup error: {e}", "error")
                self._log(traceback.format_exc(), "debug")
            except Exception:
                pass
            self._stop_startup_heartbeat()
            self._reset_ui_state()
            return
        
        # Initialize translator if needed (or if it was reset or client was cleared during shutdown)
        needs_new_translator = (not hasattr(self, 'translator')) or (self.translator is None)
        if not needs_new_translator:
            try:
                needs_new_translator = getattr(self.translator, 'client', None) is None
                if needs_new_translator:
                    self._log("♻️ Translator exists but client was cleared — reinitializing translator", "debug")
            except Exception:
                needs_new_translator = True
        if needs_new_translator:
            self._log("⚙️ Initializing translator...", "info")
            
            # CRITICAL: Set batch environment variables BEFORE creating translator
            # This ensures MangaTranslator picks up the batch settings on initialization
            try:
                # Get batch translation setting from main GUI
                batch_translation_enabled = False
                batch_size_value = 1
                
                if hasattr(self.main_gui, 'batch_translation_var'):
                    # Check if batch translation is enabled in GUI
                    try:
                        if hasattr(self.main_gui.batch_translation_var, 'get'):
                            batch_translation_enabled = bool(self.main_gui.batch_translation_var.get())
                        else:
                            batch_translation_enabled = bool(self.main_gui.batch_translation_var)
                    except Exception:
                        pass
                
                if hasattr(self.main_gui, 'batch_size_var'):
                    # Get batch size from GUI
                    try:
                        if hasattr(self.main_gui.batch_size_var, 'get'):
                            batch_size_value = int(self.main_gui.batch_size_var.get())
                        else:
                            batch_size_value = int(self.main_gui.batch_size_var)
                    except Exception:
                        batch_size_value = 1
                
                # Set environment variables for the translator to pick up
                if batch_translation_enabled:
                    os.environ['BATCH_TRANSLATION'] = '1'
                    os.environ['BATCH_SIZE'] = str(max(1, batch_size_value))
                    self._log(f"📦 Batch Translation ENABLED: {batch_size_value} concurrent API calls", "info")
                else:
                    os.environ['BATCH_TRANSLATION'] = '0'
                    os.environ['BATCH_SIZE'] = '1'
                    self._log("📦 Batch Translation DISABLED: Sequential API calls", "info")
            except Exception as e:
                self._log(f"⚠️ Warning: Could not set batch settings: {e}", "warning")
                os.environ['BATCH_TRANSLATION'] = '0'
                os.environ['BATCH_SIZE'] = '1'
            
            try:
                self.translator = MangaTranslator(
                    ocr_config,
                    self.main_gui.client,
                    self.main_gui,
                    log_callback=self._log
                )
                
                # Fix 4: Safely set OCR manager
                if hasattr(self, 'ocr_manager'):
                    self.translator.ocr_manager = self.ocr_manager
                else:
                    from ocr_manager import OCRManager
                    self.ocr_manager = OCRManager(log_callback=self._log)
                    self.translator.ocr_manager = self.ocr_manager
                    
                    # Attach preloaded RT-DETR if available
                    try:
                        if hasattr(self, '_preloaded_bd') and self._preloaded_bd:
                            self.translator.bubble_detector = self._preloaded_bd
                            self._log("🤖 RT-DETR preloaded and attached to translator", "debug")
                    except Exception:
                        pass
                    
                    # Distribute stop flags to all components
                    self._distribute_stop_flags()
                    
                # Provide Replicate API key to translator if present, but DO NOT force-enable cloud mode here.
                # Actual inpainting mode is chosen by the UI and applied in _apply_rendering_settings.
                saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                if saved_api_key:
                    self.translator.replicate_api_key = saved_api_key
                
                # Apply text rendering settings (this sets skip/cloud/local based on UI)
                self._apply_rendering_settings()
                
                try:
                    time.sleep(0.05)
                except Exception:
                    pass
                self._log("✅ Translator ready", "info")
                
            except Exception as e:
                self._log(f"❌ Failed to initialize translator: {str(e)}", "error")
                import traceback
                self._log(traceback.format_exc(), "error")
                self._stop_startup_heartbeat()
                self._reset_ui_state()
                return
        else:
            # Update batch settings for existing translator
            try:
                batch_translation_enabled = False
                batch_size_value = 1
                
                if hasattr(self.main_gui, 'batch_translation_var'):
                    try:
                        if hasattr(self.main_gui.batch_translation_var, 'get'):
                            batch_translation_enabled = bool(self.main_gui.batch_translation_var.get())
                        else:
                            batch_translation_enabled = bool(self.main_gui.batch_translation_var)
                    except Exception:
                        pass
                
                if hasattr(self.main_gui, 'batch_size_var'):
                    try:
                        if hasattr(self.main_gui.batch_size_var, 'get'):
                            batch_size_value = int(self.main_gui.batch_size_var.get())
                        else:
                            batch_size_value = int(self.main_gui.batch_size_var)
                    except Exception:
                        batch_size_value = 1
                
                # Update environment variables and translator attributes
                if batch_translation_enabled:
                    os.environ['BATCH_TRANSLATION'] = '1'
                    os.environ['BATCH_SIZE'] = str(max(1, batch_size_value))
                    self.translator.batch_mode = True
                    self.translator.batch_size = max(1, batch_size_value)
                    self._log(f"📦 Batch Translation UPDATED: {batch_size_value} concurrent API calls", "info")
                else:
                    os.environ['BATCH_TRANSLATION'] = '0'
                    os.environ['BATCH_SIZE'] = '1'
                    self.translator.batch_mode = False
                    self.translator.batch_size = 1
                    self._log("📦 Batch Translation UPDATED: Sequential API calls", "info")
            except Exception as e:
                self._log(f"⚠️ Warning: Could not update batch settings: {e}", "warning")
            
            # Update the translator with the new client if model changed
            if needs_new_client and hasattr(self.translator, 'client'):
                self.translator.client = self.main_gui.client
                self._log(f"Updated translator with new API client", "info")
            
            # Distribute stop flags to all components
            self._distribute_stop_flags()
            
            # Update rendering settings
            self._apply_rendering_settings()
            
            # Ensure inpainting settings are properly synchronized
            if hasattr(self, 'inpainting_mode_var'):
                inpainting_mode = self.inpainting_mode_var.get()
                
                if inpainting_mode == 'skip':
                    self.translator.skip_inpainting = True
                    self.translator.use_cloud_inpainting = False
                    self._log("Inpainting: SKIP", "debug")
                    
                elif inpainting_mode == 'local':
                    self.translator.skip_inpainting = False
                    self.translator.use_cloud_inpainting = False
                    
                    # IMPORTANT: Load the local inpainting model if not already loaded
                    if hasattr(self, 'local_model_var'):
                        selected_model = self.local_model_var.get()
                        if selected_model and selected_model != "None":
                            # Get model path from available models
                            model_info = self.available_local_models.get(selected_model)
                            if model_info:
                                model_path = model_info['path']
                                # Load the model into translator
                                if hasattr(self.translator, 'load_local_inpainting_model'):
                                    success = self.translator.load_local_inpainting_model(model_path)
                                    if success:
                                        self._log(f"Inpainting: LOCAL - Loaded {selected_model}", "info")
                                    else:
                                        self._log(f"Inpainting: Failed to load local model {selected_model}", "error")
                                else:
                                    # Set the model path directly if no load method
                                    self.translator.local_inpaint_model_path = model_path
                                    self._log(f"Inpainting: LOCAL - Set model path for {selected_model}", "info")
                            else:
                                self._log("Inpainting: LOCAL - No model selected", "warning")
                        else:
                            self._log("Inpainting: LOCAL - No model configured", "warning")
                    else:
                        self._log("Inpainting: LOCAL (default)", "debug")
                    
                elif inpainting_mode == 'cloud':
                    self.translator.skip_inpainting = False
                    saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                    if saved_api_key:
                        self.translator.use_cloud_inpainting = True
                        self.translator.replicate_api_key = saved_api_key
                        self._log("Inpainting: CLOUD (Replicate)", "debug")
                    else:
                        # Fallback to local if no API key
                        self.translator.use_cloud_inpainting = False
                        self._log("Inpainting: LOCAL (no Replicate key, fallback)", "warning")
            else:
                # Default to local inpainting if variable doesn't exist
                self.translator.skip_inpainting = False
                self.translator.use_cloud_inpainting = False
                self._log("Inpainting: LOCAL (default)", "debug")

            # Double-check the settings are applied correctly
            self._log(f"Inpainting final status:", "debug")
            self._log(f"  - Skip: {self.translator.skip_inpainting}", "debug")
            self._log(f"  - Cloud: {self.translator.use_cloud_inpainting}", "debug")
            self._log(f"  - Mode: {'SKIP' if self.translator.skip_inpainting else 'CLOUD' if self.translator.use_cloud_inpainting else 'LOCAL'}", "debug")
        
        # Preflight RT-DETR to avoid first-page fallback after aggressive cleanup
        try:
            ocr_set = self.main_gui.config.get('manga_settings', {}).get('ocr', {}) or {}
            if ocr_set.get('bubble_detection_enabled', False):
                # Ensure a default RT-DETR model id exists when required
                if ocr_set.get('detector_type', 'rtdetr') in ('rtdetr', 'auto'):
                    if not ocr_set.get('rtdetr_model_url') and not ocr_set.get('bubble_model_path'):
                        ocr_set['rtdetr_model_url'] = 'ogkalu/comic-text-and-bubble-detector'
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                self._preflight_bubble_detector(ocr_set)
        except Exception:
            pass
        
        # Reset progress
        self.total_files = len(self.selected_files)
        self.completed_files = 0
        self.failed_files = 0
        self.current_file_index = 0
        
        # Reset all global cancellation flags for new translation
        self._reset_global_cancellation()
        
        # Note: is_running is already True from _start_translation()
        # Just ensure stop_flag is clear
        self.stop_flag.clear()
        # Queue UI updates to be processed by main thread (just for file list disable)
        self.update_queue.put(('ui_state', 'translation_started'))
        
        # Log start message
        self._log(f"Starting translation of {self.total_files} files...", "info")
        self._log(f"Using OCR provider: {ocr_config['provider'].upper()}", "info")
        if ocr_config['provider'] == 'google':
            self._log(f"Using Google Vision credentials: {os.path.basename(ocr_config['google_credentials_path'])}", "info")
        elif ocr_config['provider'] == 'azure':
            self._log(f"Using Azure endpoint: {ocr_config['azure_endpoint']}", "info")
        else:
            self._log(f"Using local OCR provider: {ocr_config['provider'].upper()}", "info")
            # Report effective API routing/model with multi-key awareness
            try:
                c = getattr(self.main_gui, 'client', None)
                if c is not None:
                    if getattr(c, 'use_multi_keys', False):
                        total_keys = 0
                        try:
                            stats = c.get_stats()
                            total_keys = stats.get('total_keys', 0)
                        except Exception:
                            pass
                        self._log(
                            f"API routing: Multi-key pool enabled — starting model '{getattr(c, 'model', 'unknown')}', keys={total_keys}, rotation={getattr(c, '_rotation_frequency', 1)}",
                            "info"
                        )
                    else:
                        self._log(f"API model: {getattr(c, 'model', 'unknown')}", "info")
            except Exception:
                pass
            # Support both Tkinter (with .get()) and PySide6 (plain values)
            contextual_enabled = self.main_gui.contextual_var.get() if hasattr(self.main_gui.contextual_var, 'get') else self.main_gui.contextual_var
            trans_history = self.main_gui.trans_history.get() if hasattr(self.main_gui.trans_history, 'get') else self.main_gui.trans_history
            rolling_enabled = self.main_gui.translation_history_rolling_var.get() if hasattr(self.main_gui.translation_history_rolling_var, 'get') else self.main_gui.translation_history_rolling_var
            
            self._log(f"Contextual: {'Enabled' if contextual_enabled else 'Disabled'}", "info")
            self._log(f"History limit: {trans_history} exchanges", "info")
            self._log(f"Rolling history: {'Enabled' if rolling_enabled else 'Disabled'}", "info")
            self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_value else 'Disabled'}", "info")
        
        # Stop heartbeat before launching worker; now regular progress takes over
        self._stop_startup_heartbeat()
        
        # Update progress to show we're starting the translation worker
        self._log("🚀 Launching translation worker...", "info")
        self._update_progress(0, self.total_files, "Starting translation...")
        
        # Start translation via executor
        try:
            # Sync with main GUI executor if possible and update EXTRACTION_WORKERS
            if hasattr(self.main_gui, '_ensure_executor'):
                self.main_gui._ensure_executor()
                self.executor = self.main_gui.executor
            # Ensure env var reflects current worker setting from main GUI
            try:
                # Support both Tkinter (with .get()) and PySide6 (plain value)
                if hasattr(self.main_gui.extraction_workers_var, 'get'):
                    workers = self.main_gui.extraction_workers_var.get()
                else:
                    workers = self.main_gui.extraction_workers_var
                os.environ["EXTRACTION_WORKERS"] = str(workers)
            except Exception:
                pass
            
            if self.executor:
                self.translation_future = self.executor.submit(self._translation_worker)
            else:
                # Fallback to dedicated thread
                self.translation_thread = threading.Thread(
                    target=self._translation_worker,
                    daemon=True
                )
                self.translation_thread.start()
        except Exception:
            # Last resort fallback to thread
            self.translation_thread = threading.Thread(
                target=self._translation_worker,
                daemon=True
            )
            self.translation_thread.start()
    
    def _apply_rendering_settings(self):
        """Apply current rendering settings to translator (PySide6 version)"""
        if not self.translator:
            return
        
        # Read all values from PySide6 widgets to ensure they're current
        # Background opacity slider
        if hasattr(self, 'opacity_slider'):
            self.bg_opacity_value = self.opacity_slider.value()
        
        # Background reduction slider
        if hasattr(self, 'reduction_slider'):
            self.bg_reduction_value = self.reduction_slider.value()
        
        # Background style (radio buttons)
        if hasattr(self, 'bg_style_group'):
            checked_id = self.bg_style_group.checkedId()
            if checked_id == 0:
                self.bg_style_value = "box"
            elif checked_id == 1:
                self.bg_style_value = "circle"
            elif checked_id == 2:
                self.bg_style_value = "wrap"
        
        # Font selection
        if hasattr(self, 'font_combo'):
            selected = self.font_combo.currentText()
            if selected == "Default":
                self.selected_font_path = None
            elif selected in self.font_mapping:
                self.selected_font_path = self.font_mapping[selected]
        
        # Text color (stored in value variables updated by color picker)
        text_color = (
            self.text_color_r_value,
            self.text_color_g_value,
            self.text_color_b_value
        )
        
        # Shadow enabled checkbox
        if hasattr(self, 'shadow_enabled_checkbox'):
            self.shadow_enabled_value = self.shadow_enabled_checkbox.isChecked()
        
        # Shadow color (stored in value variables updated by color picker)
        shadow_color = (
            self.shadow_color_r_value,
            self.shadow_color_g_value,
            self.shadow_color_b_value
        )
        
        # Shadow offset spinboxes
        if hasattr(self, 'shadow_offset_x_spinbox'):
            self.shadow_offset_x_value = self.shadow_offset_x_spinbox.value()
        if hasattr(self, 'shadow_offset_y_spinbox'):
            self.shadow_offset_y_value = self.shadow_offset_y_spinbox.value()
        
        # Shadow blur spinbox
        if hasattr(self, 'shadow_blur_spinbox'):
            self.shadow_blur_value = self.shadow_blur_spinbox.value()
        
        # Force caps lock checkbox
        if hasattr(self, 'force_caps_checkbox'):
            self.force_caps_lock_value = self.force_caps_checkbox.isChecked()
        
        # Strict text wrapping checkbox
        if hasattr(self, 'strict_wrap_checkbox'):
            self.strict_text_wrapping_value = self.strict_wrap_checkbox.isChecked()
        
        # Font sizing controls
        if hasattr(self, 'min_size_spinbox'):
            self.auto_min_size_value = self.min_size_spinbox.value()
        if hasattr(self, 'max_size_spinbox'):
            self.max_font_size_value = self.max_size_spinbox.value()
        if hasattr(self, 'multiplier_slider'):
            self.font_size_multiplier_value = self.multiplier_slider.value()
        
        # Determine font size value based on mode
        if self.font_size_mode_value == 'multiplier':
            # Pass negative value to indicate multiplier mode
            font_size = -self.font_size_multiplier_value
        else:
            # Fixed mode - use the font size value directly
            font_size = self.font_size_value if self.font_size_value > 0 else None
        
        # Apply concise logging toggle from Advanced settings
        try:
            adv_cfg = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
            self.translator.concise_logs = bool(adv_cfg.get('concise_logs', False))
        except Exception:
            pass
        
        # Push rendering settings to translator
        self.translator.update_text_rendering_settings(
            bg_opacity=self.bg_opacity_value,
            bg_style=self.bg_style_value,
            bg_reduction=self.bg_reduction_value,
            font_style=self.selected_font_path,
            font_size=font_size,
            text_color=text_color,
            shadow_enabled=self.shadow_enabled_value,
            shadow_color=shadow_color,
            shadow_offset_x=self.shadow_offset_x_value,
            shadow_offset_y=self.shadow_offset_y_value,
            shadow_blur=self.shadow_blur_value,
            force_caps_lock=self.force_caps_lock_value
        )
        
        # Free-text-only background opacity toggle -> read from checkbox (PySide6)
        try:
            if hasattr(self, 'ft_only_checkbox'):
                ft_only_enabled = self.ft_only_checkbox.isChecked()
                self.translator.free_text_only_bg_opacity = bool(ft_only_enabled)
                # Also update the value variable
                self.free_text_only_bg_opacity_value = ft_only_enabled
        except Exception:
            pass
        
        # Update font mode and multiplier explicitly
        self.translator.font_size_mode = self.font_size_mode_value
        self.translator.font_size_multiplier = self.font_size_multiplier_value
        self.translator.min_readable_size = self.auto_min_size_value
        self.translator.max_font_size_limit = self.max_font_size_value
        self.translator.strict_text_wrapping = self.strict_text_wrapping_value
        self.translator.force_caps_lock = self.force_caps_lock_value
        
        # Update constrain to bubble setting
        if hasattr(self, 'constrain_to_bubble_value'):
            self.translator.constrain_to_bubble = self.constrain_to_bubble_value
        
        # Handle inpainting mode (radio: skip/local/cloud/hybrid)
        mode = None
        if hasattr(self, 'inpainting_mode_var'):
            mode = self.inpainting_mode_var.get()
        else:
            mode = 'local'
        
        # Persist selected mode on translator
        self.translator.inpaint_mode = mode
        
        if mode == 'skip':
            self.translator.skip_inpainting = True
            self.translator.use_cloud_inpainting = False
            self._log("  Inpainting: Skipped", "info")
        elif mode == 'cloud':
            self.translator.skip_inpainting = False
            saved_api_key = self.main_gui.config.get('replicate_api_key', '')
            if saved_api_key:
                self.translator.use_cloud_inpainting = True
                self.translator.replicate_api_key = saved_api_key
                self._log("  Inpainting: Cloud (Replicate)", "info")
            else:
                self.translator.use_cloud_inpainting = False
                self._log("  Inpainting: Local (no Replicate key, fallback)", "warning")
        elif mode == 'hybrid':
            self.translator.skip_inpainting = False
            self.translator.use_cloud_inpainting = False
            self._log("  Inpainting: Hybrid", "info")
        else:
            # Local (default)
            self.translator.skip_inpainting = False
            self.translator.use_cloud_inpainting = False
            self._log("  Inpainting: Local", "info")
        
        # Persist free-text-only BG opacity setting to config (handled in _save_rendering_settings)
        # Value is now read directly from checkbox in PySide6
        
        # Log the applied rendering and inpainting settings
        self._log(f"Applied rendering settings:", "info")
        self._log(f"  Background: {self.bg_style_value} @ {int(self.bg_opacity_value/255*100)}% opacity", "info")
        import os
        self._log(f"  Font: {os.path.basename(self.selected_font_path) if self.selected_font_path else 'Default'}", "info")
        self._log(f"  Minimum Font Size: {self.auto_min_size_value}pt", "info")
        self._log(f"  Maximum Font Size: {self.max_font_size_value}pt", "info")
        self._log(f"  Strict Text Wrapping: {'Enabled (force fit)' if self.strict_text_wrapping_value else 'Disabled (allow overflow)'}", "info")
        if self.font_size_mode_value == 'multiplier':
            self._log(f"  Font Size: Dynamic multiplier ({self.font_size_multiplier_value:.1f}x)", "info")
            if hasattr(self, 'constrain_to_bubble_value'):
                constraint_status = "constrained" if self.constrain_to_bubble_value else "unconstrained"
                self._log(f"  Text Constraint: {constraint_status}", "info")
        else:
            size_text = f"{self.font_size_value}pt" if self.font_size_value > 0 else "Auto"
            self._log(f"  Font Size: Fixed ({size_text})", "info")
        self._log(f"  Text Color: RGB({text_color[0]}, {text_color[1]}, {text_color[2]})", "info")
        self._log(f"  Shadow: {'Enabled' if self.shadow_enabled_value else 'Disabled'}", "info")
        try:
            self._log(f"  Free-text-only BG opacity: {'Enabled' if getattr(self, 'free_text_only_bg_opacity_value', False) else 'Disabled'}", "info")
        except Exception:
            pass
        self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_value else 'Disabled'}", "info")
    
    def _on_create_subfolder_toggle(self, state=None):
        """Handle create 'translated' subfolder toggle"""
        try:
            enabled = bool(self.create_subfolder_checkbox.isChecked()) if hasattr(self, 'create_subfolder_checkbox') else bool(state)
        except Exception:
            enabled = bool(state)
        self.create_subfolder_value = enabled
        # Persist with the existing save mechanism
        self._save_rendering_settings()
    
    def _translation_worker(self):
        """Worker thread for translation"""
        try:
            # Defensive: ensure translator exists before using it (legacy callers may start this worker early)
            if not hasattr(self, 'translator') or self.translator is None:
                self._log("⚠️ Translator not initialized yet; skipping worker start", "warning")
                return
            if hasattr(self.translator, 'set_stop_flag'):
                self.translator.set_stop_flag(self.stop_flag)
            
            # Ensure API parallelism (batch API calls) is controlled independently of local parallel processing.
            # Propagate the GUI "Batch Translation" toggle into environment so Unified API Client applies it globally
            # for all providers (including custom endpoints).
            try:
                import os as _os
                # Support both Tkinter (with .get()) and PySide6 (plain value)
                batch_enabled = False
                if hasattr(self.main_gui, 'batch_translation_var'):
                    if hasattr(self.main_gui.batch_translation_var, 'get'):
                        batch_enabled = bool(self.main_gui.batch_translation_var.get())
                    else:
                        batch_enabled = bool(self.main_gui.batch_translation_var)
                _os.environ['BATCH_TRANSLATION'] = '1' if batch_enabled else '0'
                
                # Use GUI batch size if available; default to 3 to match existing default
                bs_val = None
                try:
                    if hasattr(self.main_gui, 'batch_size_var'):
                        if hasattr(self.main_gui.batch_size_var, 'get'):
                            bs_val = str(int(self.main_gui.batch_size_var.get()))
                        else:
                            bs_val = str(int(self.main_gui.batch_size_var))
                except Exception:
                    bs_val = None
                _os.environ['BATCH_SIZE'] = bs_val or _os.environ.get('BATCH_SIZE', '3')
            except Exception:
                # Non-fatal if env cannot be set
                pass
            
            # Panel-level parallelization setting (LOCAL threading for panels)
            advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
            panel_parallel = bool(advanced.get('parallel_panel_translation', False))
            requested_panel_workers = int(advanced.get('panel_max_workers', 2))

            # Decouple from global parallel processing: panel concurrency is governed ONLY by panel settings
            effective_workers = requested_panel_workers if (panel_parallel and len(self.selected_files) > 1) else 1

            # Hint translator about preferred BD ownership: use singleton only when not using panel parallelism
            try:
                if hasattr(self, 'translator') and self.translator:
                    self.translator.use_singleton_bubble_detector = not (panel_parallel and effective_workers > 1)
            except Exception:
                pass

            # Model preloading phase
            self._log("🔧 Model preloading phase", "info")
            # Log current counters (diagnostic)
            try:
                st = self.translator.get_preload_counters() if hasattr(self.translator, 'get_preload_counters') else None
                if st:
                    self._log(f"   Preload counters before: inpaint_spares={st.get('inpaint_spares',0)}, detector_spares={st.get('detector_spares',0)}", "debug")
            except Exception:
                pass
            # 1) Warm up bubble detector instances first (so detection can start immediately)
            try:
                ocr_set = self.main_gui.config.get('manga_settings', {}).get('ocr', {}) or {}
                if (
                    effective_workers > 1
                    and ocr_set.get('bubble_detection_enabled', True)
                    and hasattr(self, 'translator')
                    and self.translator
                ):
                    # For parallel panel translation, prefer thread-local detectors (avoid singleton for concurrency)
                    try:
                        self.translator.use_singleton_bubble_detector = False
                    except Exception:
                        pass
                    desired_bd = min(int(effective_workers), max(1, int(len(self.selected_files) or 1)))
                    self._log(f"🧰 Preloading bubble detector instances for {desired_bd} panel worker(s)...", "info")
                    try:
                        import time as _time
                        t0 = _time.time()
                        self.translator.preload_bubble_detectors(ocr_set, desired_bd)
                        dt = _time.time() - t0
                        self._log(f"⏱️ Bubble detector preload finished in {dt:.2f}s", "info")
                    except Exception as _e:
                        self._log(f"⚠️ Bubble detector preload skipped: {_e}", "warning")
            except Exception:
                pass
            # 2) Preload LOCAL inpainting instances for panel parallelism
            inpaint_preload_event = None
            try:
                inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'cloud')
                if (
                    effective_workers > 1
                    and inpaint_method == 'local'
                    and hasattr(self, 'translator')
                    and self.translator
                ):
                    local_method = self.main_gui.config.get('manga_local_inpaint_model', 'anime')
                    model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')
                    if not model_path:
                        model_path = self.main_gui.config.get(f'{local_method}_model_path', '')
                    
                    # Preload one shared instance plus spares for parallel panel processing
                    # Constrain to actual number of files (no need for more workers than files)
                    desired_inp = min(int(effective_workers), max(1, int(len(self.selected_files) or 1)))
                    self._log(f"🧰 Preloading {desired_inp} local inpainting instance(s) for panel workers...", "info")
                    try:
                        import time as _time
                        t0 = _time.time()
                        # Use synchronous preload to ensure instances are ready before panel processing starts
                        self.translator.preload_local_inpainters(local_method, model_path, desired_inp)
                        dt = _time.time() - t0
                        self._log(f"⏱️ Local inpainting preload finished in {dt:.2f}s", "info")
                    except Exception as _e:
                        self._log(f"⚠️ Local inpainting preload failed: {_e}", "warning")
                        import traceback
                        self._log(traceback.format_exc(), "debug")
            except Exception as preload_err:
                self._log(f"⚠️ Inpainting preload setup failed: {preload_err}", "warning")
            
            # Log updated counters (diagnostic)
            try:
                st2 = self.translator.get_preload_counters() if hasattr(self.translator, 'get_preload_counters') else None
                if st2:
                    self._log(f"   Preload counters after: inpaint_spares={st2.get('inpaint_spares',0)}, detector_spares={st2.get('detector_spares',0)}", "debug")
            except Exception:
                pass

            if panel_parallel and len(self.selected_files) > 1 and effective_workers > 1:
                self._log(f"🚀 Parallel PANEL translation ENABLED ({effective_workers} workers)", "info")
                
                import concurrent.futures
                import threading as _threading
                progress_lock = _threading.Lock()
                # Memory barrier: ensures resources are fully released before next panel starts
                completion_barrier = _threading.Semaphore(1)  # Only one panel can complete at a time
                counters = {
                    'started': 0,
                    'done': 0,
                    'failed': 0
                }
                total = self.total_files
                
                def process_single(idx, filepath):
                    # Check stop flag at the very beginning
                    if self.stop_flag.is_set():
                        return False
                    
                    # Create an isolated translator instance per panel
                    translator = None  # Initialize outside try block for cleanup
                    try:
                        # Check again before starting expensive work
                        if self.stop_flag.is_set():
                            return False
                        from manga_translator import MangaTranslator
                        import os
                        # Build full OCR config for this thread (mirror _start_translation)
                        ocr_config = {'provider': self.ocr_provider_value}
                        if ocr_config['provider'] == 'google':
                            google_creds = self.main_gui.config.get('google_vision_credentials', '') or \
                                           self.main_gui.config.get('google_cloud_credentials', '')
                            if google_creds and os.path.exists(google_creds):
                                ocr_config['google_credentials_path'] = google_creds
                            else:
                                self._log("⚠️ Google Cloud Vision credentials not found for parallel task", "warning")
                        elif ocr_config['provider'] == 'azure':
                            azure_key = self.main_gui.config.get('azure_vision_key', '')
                            azure_endpoint = self.main_gui.config.get('azure_vision_endpoint', '')
                            if azure_key and azure_endpoint:
                                ocr_config['azure_key'] = azure_key
                                ocr_config['azure_endpoint'] = azure_endpoint
                            else:
                                self._log("⚠️ Azure credentials not found for parallel task", "warning")

                        translator = MangaTranslator(ocr_config, self.main_gui.client, self.main_gui, log_callback=self._log)
                        translator.set_stop_flag(self.stop_flag)
                        
                        # CRITICAL: Disable singleton bubble detector for parallel panel processing
                        # Each panel should use pool-based detectors for true parallelism
                        try:
                            translator.use_singleton_bubble_detector = False
                            self._log(f"   🤖 Panel translator: bubble detector pool mode enabled", "debug")
                        except Exception:
                            pass
                        
                        # Ensure parallel processing settings are properly applied to each panel translator
                        # The web UI maps parallel_panel_translation to parallel_processing for MangaTranslator compatibility
                        try:
                            advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                            if advanced.get('parallel_panel_translation', False):
                                # Override the manga_settings in this translator instance to enable parallel processing
                                # for bubble regions within each panel
                                translator.manga_settings.setdefault('advanced', {})['parallel_processing'] = True
                                panel_workers = int(advanced.get('panel_max_workers', 2))
                                translator.manga_settings.setdefault('advanced', {})['max_workers'] = panel_workers
                                # Also set the instance attributes directly
                                translator.parallel_processing = True
                                translator.max_workers = panel_workers
                                self._log(f"   📋 Panel translator configured: parallel_processing={translator.parallel_processing}, max_workers={translator.max_workers}", "debug")
                            else:
                                self._log(f"   📋 Panel translator: parallel_panel_translation=False, using sequential bubble processing", "debug")
                        except Exception as e:
                            self._log(f"   ⚠️ Warning: Failed to configure parallel processing for panel translator: {e}", "warning")
                        
                        # Also propagate global cancellation to isolated translator
                        from manga_translator import MangaTranslator as MTClass
                        if MTClass.is_globally_cancelled():
                            return False
                        
                        # Check stop flag before configuration
                        if self.stop_flag.is_set():
                            return False
                            
                        # Apply inpainting and rendering options roughly matching current translator
                        try:
                            translator.constrain_to_bubble = getattr(self, 'constrain_to_bubble_var').get() if hasattr(self, 'constrain_to_bubble_var') else True
                        except Exception:
                            pass
                        
                        # Set full page context based on UI
                        try:
                            translator.set_full_page_context(
                                enabled=self.full_page_context_var.get(),
                                custom_prompt=self.full_page_context_prompt
                            )
                        except Exception:
                            pass
                            
                        # Another check before path setup
                        if self.stop_flag.is_set():
                            return False
                        
                        # Determine output path (route CBZ images to job out_dir)
                        filename = os.path.basename(filepath)
                        output_path = None
                        try:
                            if hasattr(self, 'cbz_image_to_job') and filepath in self.cbz_image_to_job:
                                cbz_file = self.cbz_image_to_job[filepath]
                                job = getattr(self, 'cbz_jobs', {}).get(cbz_file)
                                if job:
                                    output_dir = job.get('out_dir')
                                    os.makedirs(output_dir, exist_ok=True)
                                    output_path = os.path.join(output_dir, filename)
                        except Exception:
                            output_path = None
                        if not output_path:
                            if self.create_subfolder_value:
                                output_dir = os.path.join(os.path.dirname(filepath), 'translated')
                                os.makedirs(output_dir, exist_ok=True)
                                output_path = os.path.join(output_dir, filename)
                            else:
                                base, ext = os.path.splitext(filepath)
                                output_path = f"{base}_translated{ext}"
                        
                        # Announce start
                        self._update_current_file(filename)
                        with progress_lock:
                            counters['started'] += 1
                            self._update_progress(counters['done'], total, f"Processing {counters['started']}/{total}: {filename}")
                        
                        # Final check before expensive processing
                        if self.stop_flag.is_set():
                            return False
                            
                        # Process image
                        result = translator.process_image(filepath, output_path, batch_index=idx+1, batch_total=total)
                        
                        # CRITICAL: Explicitly cleanup this panel's translator resources
                        # This prevents resource leaks and partial translation issues
                        try:
                            if translator:
                                # Return checked-out inpainter to pool for reuse
                                if hasattr(translator, '_return_inpainter_to_pool'):
                                    translator._return_inpainter_to_pool()
                                # Return bubble detector to pool for reuse
                                if hasattr(translator, '_return_bubble_detector_to_pool'):
                                    translator._return_bubble_detector_to_pool()
                                # Clear all caches and state
                                if hasattr(translator, 'reset_for_new_image'):
                                    translator.reset_for_new_image()
                                # Clear internal state
                                if hasattr(translator, 'clear_internal_state'):
                                    translator.clear_internal_state()
                        except Exception as cleanup_err:
                            self._log(f"⚠️ Panel translator cleanup warning: {cleanup_err}", "debug")
                        
                        # CRITICAL: Use completion barrier to prevent resource conflicts
                        # This ensures only one panel completes/cleans up at a time
                        with completion_barrier:
                            # Update counters only if not stopped
                            with progress_lock:
                                if self.stop_flag.is_set():
                                    # Don't update counters if translation was stopped
                                    return False
                            
                            # Check if translation actually produced valid output
                            translation_successful = False
                            if result.get('success', False) and not result.get('interrupted', False):
                                # Verify there's an actual output file and translated regions
                                output_exists = result.get('output_path') and os.path.exists(result.get('output_path', ''))
                                regions = result.get('regions', [])
                                has_translations = any(r.get('translated_text', '') for r in regions)
                                
                                # CRITICAL: Verify all detected regions got translated
                                # Partial failures indicate inpainting or rendering issues
                                if has_translations and regions:
                                    translated_count = sum(1 for r in regions if r.get('translated_text', '').strip())
                                    detected_count = len(regions)
                                    completion_rate = translated_count / detected_count if detected_count > 0 else 0
                                    
                                    # Log warning if completion rate is less than 100%
                                    if completion_rate < 1.0:
                                        self._log(f"⚠️ Partial translation: {translated_count}/{detected_count} regions translated ({completion_rate*100:.1f}%)", "warning")
                                        self._log(f"   API may have skipped some regions (sound effects, symbols, or cleaning removed content)", "warning")
                                    
                                    # Only consider successful if at least 50% of regions translated
                                    # This prevents marking completely failed images as successful
                                    translation_successful = output_exists and completion_rate >= 0.5
                                else:
                                    translation_successful = output_exists and has_translations
                            
                                if translation_successful:
                                    self.completed_files += 1
                                    self._log(f"✅ Translation completed: {filename}", "success")
                                    # Memory barrier: ensure resources are released before next completion
                                    time.sleep(0.15)  # Slightly longer pause for stability
                                    self._log("💤 Panel completion pausing for resource cleanup", "debug")
                                else:
                                    self.failed_files += 1
                                    # Log the specific reason for failure
                                    if result.get('interrupted', False):
                                        self._log(f"⚠️ Translation interrupted: {filename}", "warning")
                                    elif not result.get('success', False):
                                        self._log(f"❌ Translation failed: {filename}", "error")
                                    elif not result.get('output_path') or not os.path.exists(result.get('output_path', '')):
                                        self._log(f"❌ Output file not created: {filename}", "error")
                                    else:
                                        self._log(f"❌ No text was translated: {filename}", "error")
                                    counters['failed'] += 1
                                counters['done'] += 1
                                self._update_progress(counters['done'], total, f"Completed {counters['done']}/{total}")
                            # End of completion_barrier block - resources now released for next panel
                        
                        return result.get('success', False)
                    except Exception as e:
                        with progress_lock:
                            # Don't update error counters if stopped
                            if not self.stop_flag.is_set():
                                self.failed_files += 1
                                counters['failed'] += 1
                                counters['done'] += 1
                        if not self.stop_flag.is_set():
                            self._log(f"❌ Error in panel task: {str(e)}", "error")
                            self._log(traceback.format_exc(), "error")
                        return False
                    finally:
                        # CRITICAL: Always cleanup translator resources, even on error
                        # This prevents resource leaks and ensures proper cleanup in parallel mode
                        try:
                            if translator:
                                # Return checked-out inpainter to pool for reuse
                                if hasattr(translator, '_return_inpainter_to_pool'):
                                    translator._return_inpainter_to_pool()
                                # Return bubble detector to pool for reuse
                                if hasattr(translator, '_return_bubble_detector_to_pool'):
                                    translator._return_bubble_detector_to_pool()
                                # Force cleanup of all models and caches
                                if hasattr(translator, 'clear_internal_state'):
                                    translator.clear_internal_state()
                                # Clear any remaining references
                                translator = None
                        except Exception:
                            pass  # Never let cleanup fail the finally block
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, effective_workers)) as executor:
                    futures = []
                    stagger_ms = int(advanced.get('panel_start_stagger_ms', 30))
                    for idx, filepath in enumerate(self.selected_files):
                        if self.stop_flag.is_set():
                            break
                        futures.append(executor.submit(process_single, idx, filepath))
                        if stagger_ms > 0:
                            time.sleep(stagger_ms / 1000.0)
                            time.sleep(0.1)  # Brief pause for stability
                            self._log("💤 Staggered submission pausing briefly for stability", "debug")
                    
                    # Handle completion and stop behavior
                    try:
                        for f in concurrent.futures.as_completed(futures):
                            if self.stop_flag.is_set():
                                # More aggressive cancellation
                                for rem in futures:
                                    rem.cancel()
                                # Try to shutdown executor immediately
                                try:
                                    executor.shutdown(wait=False)
                                except Exception:
                                    pass
                                break
                            try:
                                # Consume future result to let it raise exceptions or return
                                f.result(timeout=0.1)  # Very short timeout
                            except Exception:
                                # Ignore; counters are updated inside process_single
                                pass
                    except Exception:
                        # If as_completed fails due to shutdown, that's ok
                        pass
                    
                    # If stopped during parallel processing, do not log panel completion
                    if self.stop_flag.is_set():
                        pass
                    else:
                        # After parallel processing, skip sequential loop
                        pass
                
                # After parallel processing, skip sequential loop
                
                # Finalize CBZ packaging after parallel mode finishes
                try:
                    self._finalize_cbz_jobs()
                except Exception:
                    pass
                
            else:
                # Sequential processing (or panel parallel requested but capped to 1 by global setting)
                for index, filepath in enumerate(self.selected_files):
                    if self.stop_flag.is_set():
                        self._log("\n⏹️ Translation stopped by user", "warning")
                        break
                    
                    # IMPORTANT: Reset translator state for each new image
                    if hasattr(self.translator, 'reset_for_new_image'):
                        self.translator.reset_for_new_image()
                    
                    self.current_file_index = index
                    filename = os.path.basename(filepath)
                    
                    self._update_current_file(filename)
                    self._update_progress(
                        index,
                        self.total_files,
                        f"Processing {index + 1}/{self.total_files}: {filename}"
                    )
                    
                    try:
                        # Determine output path (route CBZ images to job out_dir)
                        job_output_path = None
                        try:
                            if hasattr(self, 'cbz_image_to_job') and filepath in self.cbz_image_to_job:
                                cbz_file = self.cbz_image_to_job[filepath]
                                job = getattr(self, 'cbz_jobs', {}).get(cbz_file)
                                if job:
                                    output_dir = job.get('out_dir')
                                    os.makedirs(output_dir, exist_ok=True)
                                    job_output_path = os.path.join(output_dir, filename)
                        except Exception:
                            job_output_path = None
                        if job_output_path:
                            output_path = job_output_path
                        else:
                            if self.create_subfolder_value:
                                output_dir = os.path.join(os.path.dirname(filepath), 'translated')
                                os.makedirs(output_dir, exist_ok=True)
                                output_path = os.path.join(output_dir, filename)
                            else:
                                base, ext = os.path.splitext(filepath)
                                output_path = f"{base}_translated{ext}"
                        
                        # Process the image
                        result = self.translator.process_image(filepath, output_path)
                        
                        # Check if translation was interrupted
                        if result.get('interrupted', False):
                            self._log(f"⏸️ Translation of {filename} was interrupted", "warning")
                            self.failed_files += 1
                            if self.stop_flag.is_set():
                                break
                        elif result.get('success', False):
                            # Verify translation actually produced valid output
                            output_exists = result.get('output_path') and os.path.exists(result.get('output_path', ''))
                            has_translations = any(r.get('translated_text', '') for r in result.get('regions', []))
                            
                            if output_exists and has_translations:
                                self.completed_files += 1
                                self._log(f"✅ Translation completed: {filename}", "success")
                                time.sleep(0.1)  # Brief pause for stability
                                self._log("💤 Sequential completion pausing briefly for stability", "debug")
                            else:
                                self.failed_files += 1
                                if not output_exists:
                                    self._log(f"❌ Output file not created: {filename}", "error")
                                else:
                                    self._log(f"❌ No text was translated: {filename}", "error")
                        else:
                            self.failed_files += 1
                            errors = '\n'.join(result.get('errors', ['Unknown error']))
                            self._log(f"❌ Translation failed: {filename}\n{errors}", "error")
                            
                            # Check for specific error types in the error messages
                            errors_lower = errors.lower()
                            if '429' in errors or 'rate limit' in errors_lower:
                                self._log(f"⚠️ RATE LIMIT DETECTED - Please wait before continuing", "error")
                                self._log(f"   The API provider is limiting your requests", "error")
                                self._log(f"   Consider increasing delay between requests in settings", "error")
                                
                                # Optionally pause for a bit
                                self._log(f"   Pausing for 60 seconds...", "warning")
                                for sec in range(60):
                                    if self.stop_flag.is_set():
                                        break
                                    time.sleep(1)
                                    if sec % 10 == 0:
                                        self._log(f"   Waiting... {60-sec} seconds remaining", "warning")
                        
                    except Exception as e:
                        self.failed_files += 1
                        error_str = str(e)
                        error_type = type(e).__name__
                        
                        self._log(f"❌ Error processing {filename}:", "error")
                        self._log(f"   Error type: {error_type}", "error")
                        self._log(f"   Details: {error_str}", "error")
                        
                        # Check for specific API errors
                        if "429" in error_str or "rate limit" in error_str.lower():
                            self._log(f"⚠️ RATE LIMIT ERROR (429) - API is throttling requests", "error")
                            self._log(f"   Please wait before continuing or reduce request frequency", "error")
                            self._log(f"   Consider increasing the API delay in settings", "error")
                            
                            # Pause for rate limit
                            self._log(f"   Pausing for 60 seconds...", "warning")
                            for sec in range(60):
                                if self.stop_flag.is_set():
                                    break
                                time.sleep(1)
                                if sec % 10 == 0:
                                    self._log(f"   Waiting... {60-sec} seconds remaining", "warning")
                            
                        elif "401" in error_str or "unauthorized" in error_str.lower():
                            self._log(f"❌ AUTHENTICATION ERROR (401) - Check your API key", "error")
                            self._log(f"   The API key appears to be invalid or expired", "error")
                            
                        elif "403" in error_str or "forbidden" in error_str.lower():
                            self._log(f"❌ FORBIDDEN ERROR (403) - Access denied", "error")
                            self._log(f"   Check your API subscription and permissions", "error")
                            
                        elif "timeout" in error_str.lower():
                            self._log(f"⏱️ TIMEOUT ERROR - Request took too long", "error")
                            self._log(f"   Consider increasing timeout settings", "error")
                            
                        else:
                            # Generic error with full traceback
                            self._log(f"   Full traceback:", "error")
                            self._log(traceback.format_exc(), "error")
                        
            
            # Finalize CBZ packaging (both modes)
            try:
                self._finalize_cbz_jobs()
            except Exception:
                pass
            
            # Final summary - only if not stopped
            if not self.stop_flag.is_set():
                self._log(f"\n{'='*60}", "info")
                self._log(f"📊 Translation Summary:", "info")
                self._log(f"   Total files: {self.total_files}", "info")
                self._log(f"   ✅ Successful: {self.completed_files}", "success")
                self._log(f"   ❌ Failed: {self.failed_files}", "error" if self.failed_files > 0 else "info")
                self._log(f"{'='*60}\n", "info")
                
                self._update_progress(
                    self.total_files,
                    self.total_files,
                    f"Complete! {self.completed_files} successful, {self.failed_files} failed"
                )
            
        except Exception as e:
            self._log(f"\n❌ Translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
        
        finally:
            # Check if auto cleanup is enabled in settings
            auto_cleanup_enabled = False  # Default disabled by default
            try:
                advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                auto_cleanup_enabled = advanced_settings.get('auto_cleanup_models', False)
            except Exception:
                pass
            
            if auto_cleanup_enabled:
                # Clean up all models to free RAM
                try:
                    # For parallel panel translation, cleanup happens here after ALL panels complete
                    is_parallel_panel = False
                    try:
                        advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                        is_parallel_panel = advanced_settings.get('parallel_panel_translation', True)
                    except Exception:
                        pass
                    
                    # Skip the "all parallel panels complete" message if stopped
                    if is_parallel_panel and not self.stop_flag.is_set():
                        self._log("\n🧹 All parallel panels complete - cleaning up models to free RAM...", "info")
                    elif not is_parallel_panel:
                        self._log("\n🧹 Cleaning up models to free RAM...", "info")
                    
                    # Clean up the shared translator if parallel processing was used
                    if 'translator' in locals():
                        translator.cleanup_all_models()
                        self._log("✅ Shared translator models cleaned up!", "info")
                    
                    # Also clean up the instance translator if it exists
                    if hasattr(self, 'translator') and self.translator:
                        self.translator.cleanup_all_models()
                        # Set to None to ensure it's released
                        self.translator = None
                        self._log("✅ Instance translator models cleaned up!", "info")
                    
                    self._log("✅ All models cleaned up - RAM freed!", "info")
                    
                except Exception as e:
                    self._log(f"⚠️ Warning: Model cleanup failed: {e}", "warning")
                
                # Force garbage collection to ensure memory is freed
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
            else:
                # Only log if not stopped
                if not self.stop_flag.is_set():
                    self._log("🔑 Auto cleanup disabled - models will remain in RAM for faster subsequent translations", "info")
            
            # IMPORTANT: Reset the entire translator instance to free ALL memory
            # Controlled by a separate "Unload models after translation" toggle
            try:
                # Check if we should reset the translator instance
                reset_translator = False  # default disabled
                try:
                    advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                    reset_translator = bool(advanced_settings.get('unload_models_after_translation', False))
                except Exception:
                    reset_translator = False
                
                if reset_translator:
                    self._log("\n🗑️ Resetting translator instance to free all memory...", "info")
                    
                    # Clear the instance translator completely
                    if hasattr(self, 'translator'):
                        # First ensure models are cleaned if not already done
                        try:
                            if self.translator and hasattr(self.translator, 'cleanup_all_models'):
                                self.translator.cleanup_all_models()
                        except Exception:
                            pass
                        
                        # Clear all internal state using the dedicated method
                        try:
                            if self.translator and hasattr(self.translator, 'clear_internal_state'):
                                self.translator.clear_internal_state()
                        except Exception:
                            pass
                        
                        # Clear remaining references with proper cleanup
                        try:
                            if self.translator:
                                # Properly unload OCR manager and all its providers
                                if hasattr(self.translator, 'ocr_manager') and self.translator.ocr_manager:
                                    try:
                                        ocr_manager = self.translator.ocr_manager
                                        # Clear all loaded OCR providers
                                        if hasattr(ocr_manager, 'providers'):
                                            for provider_name, provider in ocr_manager.providers.items():
                                                # Unload each provider's models
                                                if hasattr(provider, 'model'):
                                                    provider.model = None
                                                if hasattr(provider, 'processor'):
                                                    provider.processor = None
                                                if hasattr(provider, 'tokenizer'):
                                                    provider.tokenizer = None
                                                if hasattr(provider, 'reader'):
                                                    provider.reader = None
                                                if hasattr(provider, 'is_loaded'):
                                                    provider.is_loaded = False
                                                self._log(f"      ✓ Unloaded OCR provider: {provider_name}", "debug")
                                            ocr_manager.providers.clear()
                                        self._log("   ✓ OCR manager fully unloaded", "debug")
                                    except Exception as e:
                                        self._log(f"   Warning: OCR manager cleanup failed: {e}", "debug")
                                    finally:
                                        self.translator.ocr_manager = None
                                
                                # Properly unload local inpainter
                                if hasattr(self.translator, 'local_inpainter') and self.translator.local_inpainter:
                                    try:
                                        if hasattr(self.translator.local_inpainter, 'unload'):
                                            self.translator.local_inpainter.unload()
                                            self._log("   ✓ Local inpainter unloaded", "debug")
                                    except Exception as e:
                                        self._log(f"   Warning: Local inpainter cleanup failed: {e}", "debug")
                                    finally:
                                        self.translator.local_inpainter = None
                                
                                # Properly unload bubble detector
                                if hasattr(self.translator, 'bubble_detector') and self.translator.bubble_detector:
                                    try:
                                        if hasattr(self.translator.bubble_detector, 'unload'):
                                            self.translator.bubble_detector.unload(release_shared=True)
                                            self._log("   ✓ Bubble detector unloaded", "debug")
                                    except Exception as e:
                                        self._log(f"   Warning: Bubble detector cleanup failed: {e}", "debug")
                                    finally:
                                        self.translator.bubble_detector = None
                                
                                # Clear API clients
                                if hasattr(self.translator, 'client'):
                                    self.translator.client = None
                                if hasattr(self.translator, 'vision_client'):
                                    self.translator.vision_client = None
                        except Exception:
                            pass
                        
                        # Call translator shutdown to free all resources
                        try:
                            if translator and hasattr(translator, 'shutdown'):
                                translator.shutdown()
                        except Exception:
                            pass
                        # Finally, delete the translator instance entirely
                        self.translator = None
                        self._log("✅ Translator instance reset - all memory freed!", "info")
                    
                    # Also clear the shared translator from parallel processing if it exists
                    if 'translator' in locals():
                        try:
                            # Clear internal references
                            if hasattr(translator, 'cache'):
                                translator.cache = None
                            if hasattr(translator, 'text_regions'):
                                translator.text_regions = None
                            if hasattr(translator, 'translated_regions'):
                                translator.translated_regions = None
                            # Delete the local reference
                            del translator
                        except Exception:
                            pass
                    
                    # Clear standalone OCR manager if it exists in manga_integration
                    if hasattr(self, 'ocr_manager') and self.ocr_manager:
                        try:
                            ocr_manager = self.ocr_manager
                            # Clear all loaded OCR providers
                            if hasattr(ocr_manager, 'providers'):
                                for provider_name, provider in ocr_manager.providers.items():
                                    # Unload each provider's models
                                    if hasattr(provider, 'model'):
                                        provider.model = None
                                    if hasattr(provider, 'processor'):
                                        provider.processor = None
                                    if hasattr(provider, 'tokenizer'):
                                        provider.tokenizer = None
                                    if hasattr(provider, 'reader'):
                                        provider.reader = None
                                    if hasattr(provider, 'is_loaded'):
                                        provider.is_loaded = False
                                ocr_manager.providers.clear()
                            self.ocr_manager = None
                            self._log("   ✓ Standalone OCR manager cleared", "debug")
                        except Exception as e:
                            self._log(f"   Warning: Standalone OCR manager cleanup failed: {e}", "debug")
                    
                    # Force multiple garbage collection passes to ensure everything is freed
                    try:
                        import gc
                        gc.collect()
                        gc.collect()  # Multiple passes for stubborn references
                        gc.collect()
                        self._log("✅ Memory fully reclaimed", "debug")
                    except Exception:
                        pass
                else:
                    # Only log if not stopped
                    if not self.stop_flag.is_set():
                        self._log("🔑 Translator instance preserved for faster subsequent translations", "debug")
                    
            except Exception as e:
                self._log(f"⚠️ Warning: Failed to reset translator instance: {e}", "warning")
            
            # Restore print hijack to original
            try:
                if hasattr(self, 'translator') and self.translator:
                    if hasattr(self.translator, 'restore_print'):
                        self.translator.restore_print()
                        self._log("✅ Print function restored to original", "debug")
            except Exception as e:
                self._log(f"⚠️ Warning: Failed to restore print: {e}", "debug")
            
            # Reset UI state (PySide6 - must call on main thread)
            try:
                # Use the existing update_queue to schedule UI reset on main thread
                # This queue is processed by the main thread's timer
                self.update_queue.put(('ui_state', 'translation_complete'))
            except Exception as e:
                self._log(f"Error resetting UI: {e}", "warning")
    
    def _stop_translation(self):
        """Stop the translation process"""
        if self.is_running:
            # Set local stop flag first for immediate response
            self.stop_flag.set()
            
            # Save current scroll position before updating button
            saved_scroll_pos = None
            try:
                if hasattr(self, 'scroll_area') and self.scroll_area:
                    scrollbar = self.scroll_area.verticalScrollBar()
                    if scrollbar:
                        saved_scroll_pos = scrollbar.value()
            except Exception:
                pass
            
            # Update button to show "Stopping..." state (gray, disabled) - do this quickly
            try:
                if hasattr(self, 'start_button') and self.start_button:
                    # Clear focus from button to prevent scroll
                    self.start_button.clearFocus()
                    
                    self.start_button.setEnabled(False)
                    # Update text label instead of button text
                    if hasattr(self, 'start_button_text'):
                        self.start_button_text.setText("Stopping...")
                    self.start_button.setStyleSheet(
                        "QPushButton { "
                        "  background-color: #6c757d; "
                        "  color: white; "
                        "  padding: 22px 30px; "
                        "  font-size: 14pt; "
                        "  font-weight: bold; "
                        "  border-radius: 8px; "
                        "} "
                        "QPushButton:disabled { "
                        "  background-color: #6c757d; "
                        "  color: white; "
                        "}"
                    )
                    # Force immediate GUI update
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()
            except Exception:
                pass
            
            # Restore scroll position after button update
            try:
                if saved_scroll_pos is not None and hasattr(self, 'scroll_area') and self.scroll_area:
                    scrollbar = self.scroll_area.verticalScrollBar()
                    if scrollbar:
                        scrollbar.setValue(saved_scroll_pos)
            except Exception:
                pass
            
            # Set global cancellation flags for coordinated stopping
            self.set_global_cancellation(True)
            
            # Also propagate to MangaTranslator class
            try:
                from manga_translator import MangaTranslator
                MangaTranslator.set_global_cancellation(True)
            except ImportError:
                pass
            
            # Also propagate to UnifiedClient if available
            try:
                from unified_api_client import UnifiedClient
                UnifiedClient.set_global_cancellation(True)
            except ImportError:
                pass
            
            # Update progress to show stopped status
            self._update_progress(
                self.completed_files,
                self.total_files, 
                f"Stopped - {self.completed_files}/{self.total_files} completed"
            )
            
            # Try to style the progress bar to indicate stopped status
            try:
                # Set progress bar to a distinctive value and try to change appearance
                if hasattr(self, 'progress_bar'):
                    # You could also set a custom style here if needed
                    # For now, we'll rely on the text indicators
                    pass
            except Exception:
                pass
            
            # Update current file display to show stopped
            self._update_current_file("Translation stopped")
            
            # Restore print hijack when translation is stopped
            try:
                if hasattr(self, 'translator') and self.translator:
                    if hasattr(self.translator, 'restore_print'):
                        self.translator.restore_print()
            except Exception:
                pass
            
            # Check if cleanup is enabled before shutting down translator on stop
            try:
                # Check user's cleanup preference
                auto_cleanup_enabled = False
                try:
                    advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                    auto_cleanup_enabled = advanced_settings.get('auto_cleanup_models', False)
                except Exception:
                    pass
                
                if auto_cleanup_enabled:
                    # User wants cleanup - shutdown translator to free RAM
                    tr = getattr(self, 'translator', None)
                    if tr and hasattr(tr, 'shutdown'):
                        import threading
                        threading.Thread(target=tr.shutdown, name="MangaTranslatorShutdown", daemon=True).start()
                        self._log("🧹 Initiated translator resource shutdown", "info")
                        # Important: clear the stale translator reference so the next Start creates a fresh instance
                        self.translator = None
                else:
                    # User wants to keep models loaded - just log that we're preserving them
                    self._log("🔑 Models preserved in RAM (cleanup disabled)", "info")
            except Exception as e:
                self._log(f"⚠️ Failed to check cleanup settings: {e}", "warning")
            
            self._log("\n⏹️ Translation stopped by user", "warning")
            
            # Schedule UI reset after a longer delay to keep "Stopping..." visible
            # The finally block in _run_translation_worker will also call _reset_ui_state,
            # but this fallback ensures UI doesn't get stuck in stopping state
            try:
                from PySide6.QtCore import QTimer
                # Wait 2 seconds for cleanup to allow "Stopping..." to be visible longer
                QTimer.singleShot(2000, self._reset_ui_state)
            except Exception:
                pass
    
    def _reset_ui_state(self):
        """Reset UI to ready state - with widget existence checks (PySide6)"""
        # Check if the dialog still exists first (PySide6)
        if not hasattr(self, 'dialog') or not self.dialog:
            return
        
        # Restore stdio redirection if active
        self._redirect_stderr(False)
        self._redirect_stdout(False)
        # Stop any startup heartbeat if still running
        try:
            self._stop_startup_heartbeat()
        except Exception:
            pass
        try:
            # Reset running flag
            self.is_running = False
            
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
            if hasattr(self, 'file_listbox') and self.file_listbox:
                if not self.file_listbox.isEnabled():
                    self.file_listbox.setEnabled(True)
                
        except Exception as e:
            # Log the error but don't crash
            if hasattr(self, '_log'):
                self._log(f"Error resetting UI state: {str(e)}", "warning")
