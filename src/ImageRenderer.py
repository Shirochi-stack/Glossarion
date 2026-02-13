import sys
import os
import json
import threading
import time
import hashlib
import traceback
import concurrent.futures
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (QWidget, QLabel, QFrame, QPushButton, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QListWidget, QComboBox, QLineEdit, QCheckBox,
                               QRadioButton, QSlider, QSpinBox, QDoubleSpinBox, QTextEdit,
                               QProgressBar, QFileDialog, QMessageBox, QColorDialog, QScrollArea,
                               QDialog, QButtonGroup, QApplication, QSizePolicy, QToolButton)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QEvent, QPropertyAnimation, QEasingCurve, Property, QThread
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QIcon, QKeyEvent, QPixmap, QTransform
from typing import List, Dict, Optional, Any
from queue import Queue, Empty
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

# MODULE-LEVEL HELPER: Reset all cancellation flags before starting an operation
def _reset_cancellation_flags(self):
    """Reset all cancellation flags before starting a new operation.
    
    MUST be called at the very start of each background operation to clear
    any stale flags from previous operations.
    """
    print("[CANCEL_RESET] Starting flag reset...")
    try:
        # Reset environment-level stop flags first — these survive across operations
        was_graceful = os.environ.get('GRACEFUL_STOP', '0')
        was_cancelled = os.environ.get('TRANSLATION_CANCELLED', '0')
        os.environ['GRACEFUL_STOP'] = '0'
        os.environ['TRANSLATION_CANCELLED'] = '0'
        os.environ['WAIT_FOR_CHUNKS'] = '0'
        print(f"[CANCEL_RESET] GRACEFUL_STOP was {was_graceful}, TRANSLATION_CANCELLED was {was_cancelled}, now '0'")
        
        # Reset stop_flag (threading.Event)
        if hasattr(self, 'stop_flag') and self.stop_flag:
            was_set = self.stop_flag.is_set()
            self.stop_flag.clear()
            print(f"[CANCEL_RESET] stop_flag was {was_set}, now cleared")
        
        # Reset global cancellation on self
        was_global = getattr(self, '_global_cancellation', None)
        if hasattr(self, '_global_cancellation'):
            self._global_cancellation = False
        print(f"[CANCEL_RESET] _global_cancellation was {was_global}, now False")
        
        # Reset MangaTranslator global cancellation AND its internal flags
        try:
            from manga_translator import MangaTranslator
            was_mt_cancelled = MangaTranslator.is_globally_cancelled()
            MangaTranslator.set_global_cancellation(False)
            MangaTranslator.reset_global_flags()  # Also call the class reset method
            print(f"[CANCEL_RESET] MangaTranslator was {was_mt_cancelled}, now reset")
        except Exception as e:
            print(f"[CANCEL_RESET] MangaTranslator reset failed: {e}")
        
        # CRITICAL: Reset instance-level cancel_requested on MangaTranslator
        # _check_stop() latches this to True and keeps returning True!
        try:
            if hasattr(self, '_manga_translator') and self._manga_translator:
                self._manga_translator.cancel_requested = False
                if hasattr(self._manga_translator, 'reset_stop_flags'):
                    self._manga_translator.reset_stop_flags()
                print(f"[CANCEL_RESET] Reset _manga_translator.cancel_requested and stop flags")
        except Exception as e:
            print(f"[CANCEL_RESET] _manga_translator.cancel_requested reset failed: {e}")
        
        # Reset UnifiedClient global cancellation
        try:
            from unified_api_client import UnifiedClient
            was_uc_cancelled = UnifiedClient.is_globally_cancelled()
            UnifiedClient.set_global_cancellation(False)
            print(f"[CANCEL_RESET] UnifiedClient class-level was {was_uc_cancelled}, now reset")
        except Exception as e:
            print(f"[CANCEL_RESET] UnifiedClient reset failed: {e}")
        
        # Reset module-level global_stop_flag in unified_api_client
        try:
            from unified_api_client import set_stop_flag
            set_stop_flag(False)
            print(f"[CANCEL_RESET] unified_api_client global_stop_flag reset")
        except Exception as e:
            print(f"[CANCEL_RESET] set_stop_flag failed: {e}")
        
        # Reset module-level _stop_requested in TransateKRtoEN
        # This flag is checked by UnifiedClient._is_stop_requested()
        try:
            from TransateKRtoEN import set_stop_flag as translate_set_stop_flag
            translate_set_stop_flag(False)
            print(f"[CANCEL_RESET] TransateKRtoEN _stop_requested reset")
        except Exception as e:
            print(f"[CANCEL_RESET] TransateKRtoEN set_stop_flag failed: {e}")
        
        # CRITICAL: Reset instance-level _cancelled flag on any existing UnifiedClient instances
        # This flag gets latched to True and prevents API calls until explicitly reset
        try:
            # Reset on manga_translator's unified_client if it exists
            if hasattr(self, '_manga_translator') and self._manga_translator:
                if hasattr(self._manga_translator, 'unified_client') and self._manga_translator.unified_client:
                    self._manga_translator.unified_client._cancelled = False
                    print(f"[CANCEL_RESET] Reset _manga_translator.unified_client._cancelled")
            # Reset on translator's unified_client if it exists
            if hasattr(self, 'translator') and self.translator:
                if hasattr(self.translator, 'unified_client') and self.translator.unified_client:
                    self.translator.unified_client._cancelled = False
                    print(f"[CANCEL_RESET] Reset translator.unified_client._cancelled")
        except Exception as e:
            print(f"[CANCEL_RESET] UnifiedClient instance reset failed: {e}")
        
        # CRITICAL: Reset OCR manager's _stopped flag
        # This flag gets latched to True and must be explicitly reset
        if hasattr(self, 'ocr_manager') and self.ocr_manager:
            if hasattr(self.ocr_manager, 'reset_stop_flags'):
                self.ocr_manager.reset_stop_flags()
                print(f"[CANCEL_RESET] Called ocr_manager.reset_stop_flags()")
            # Also reset individual providers (attribute is 'providers' not '_providers')
            if hasattr(self.ocr_manager, 'providers'):
                for name, provider in self.ocr_manager.providers.items():
                    if hasattr(provider, 'reset_stop_flags'):
                        provider.reset_stop_flags()
                    if hasattr(provider, '_stopped'):
                        provider._stopped = False
                print(f"[CANCEL_RESET] Reset {len(self.ocr_manager.providers)} OCR provider flags")
        
        # Reset translator's OCR manager if available
        if hasattr(self, 'translator') and self.translator:
            if hasattr(self.translator, 'ocr_manager') and self.translator.ocr_manager:
                if hasattr(self.translator.ocr_manager, 'reset_stop_flags'):
                    self.translator.ocr_manager.reset_stop_flags()
                if hasattr(self.translator.ocr_manager, 'providers'):
                    for name, provider in self.translator.ocr_manager.providers.items():
                        if hasattr(provider, 'reset_stop_flags'):
                            provider.reset_stop_flags()
                        if hasattr(provider, '_stopped'):
                            provider._stopped = False
        
        # CRITICAL: Reset local inpainter _stopped flag
        # The inpainter latches _stopped = True and keeps skipping inpainting!
        inpainter_reset_count = 0
        try:
            # Reset inpainter on manga_translator if it exists
            if hasattr(self, '_manga_translator') and self._manga_translator:
                if hasattr(self._manga_translator, 'local_inpainter') and self._manga_translator.local_inpainter:
                    self._manga_translator.local_inpainter._stopped = False
                    if hasattr(self._manga_translator.local_inpainter, 'reset_stop_flags'):
                        self._manga_translator.local_inpainter.reset_stop_flags()
                    inpainter_reset_count += 1
            
            # CRITICAL: Reset ALL inpainters in the MangaTranslator pool
            # Pool stores inpainters in _inpaint_pool[key]['spares']
            try:
                from manga_translator import MangaTranslator
                if hasattr(MangaTranslator, '_inpaint_pool') and MangaTranslator._inpaint_pool:
                    for key, rec in MangaTranslator._inpaint_pool.items():
                        if rec and 'spares' in rec:
                            for inpainter in rec['spares']:
                                if inpainter is not None:
                                    if hasattr(inpainter, '_stopped'):
                                        inpainter._stopped = False
                                    if hasattr(inpainter, 'reset_stop_flags'):
                                        inpainter.reset_stop_flags()
                                    inpainter_reset_count += 1
            except Exception as pool_err:
                print(f"[CANCEL_RESET] Inpainter pool reset error: {pool_err}")
            
            if inpainter_reset_count > 0:
                print(f"[CANCEL_RESET] Reset {inpainter_reset_count} inpainter _stopped flag(s)")
            
            # Restart dead inpainter workers if any (workers are now protected from
            # psutil kill during stop, so this is only a safety net for unexpected crashes).
            try:
                from manga_translator import MangaTranslator
                if hasattr(MangaTranslator, '_inpaint_pool') and MangaTranslator._inpaint_pool:
                    for key, rec in MangaTranslator._inpaint_pool.items():
                        if rec and 'spares' in rec:
                            for inpainter in rec['spares']:
                                if inpainter is not None and getattr(inpainter, '_mp_enabled', False):
                                    if not inpainter._check_worker_health():
                                        print(f"[CANCEL_RESET] Inpainter worker dead for '{key}' — restarting silently")
                                        try:
                                            inpainter._stop_worker()
                                            inpainter._start_worker()
                                            # Reload model so it's ready to go
                                            method = getattr(inpainter, 'current_method', None)
                                            model_path = getattr(inpainter, '_last_model_path', None)
                                            if method and model_path:
                                                inpainter._mp_load_model(method, model_path, force_reload=True)
                                                print(f"[CANCEL_RESET] Worker restarted and model reloaded for '{key}'")
                                            else:
                                                print(f"[CANCEL_RESET] Worker restarted (no model to reload)")
                                        except Exception as restart_err:
                                            print(f"[CANCEL_RESET] Worker restart failed for '{key}': {restart_err}")
            except Exception as pool_restart_err:
                print(f"[CANCEL_RESET] Inpainter pool worker restart error: {pool_restart_err}")
        except Exception as e:
            print(f"[CANCEL_RESET] Local inpainter reset failed: {e}")
        
        print("[CANCEL_RESET] All cancellation flags reset")
    except Exception as e:
        print(f"[CANCEL_RESET] Error resetting flags: {e}")

# MODULE-LEVEL HELPER: Check if translation is cancelled
def _is_translation_cancelled(self) -> bool:
    """Check all stop flags to determine if translation should be cancelled.
    
    Returns True if any cancellation flag is explicitly set by stop button.
    Only checks flags that are SET when stop is clicked - not default states.
    """
    try:
        # Check stop_flag (threading.Event) - explicitly set by stop button
        if hasattr(self, 'stop_flag') and self.stop_flag and self.stop_flag.is_set():
            print("[CANCEL_CHECK] stop_flag is set")
            return True
        
        # NOTE: Do NOT check is_running here - it's False by default and would
        # cause false positives. Only flags explicitly SET by stop button.
        
        # Check global cancellation on self - explicitly set by stop button
        if getattr(self, '_global_cancellation', False):
            print("[CANCEL_CHECK] _global_cancellation is True")
            return True
        
        # Check MangaTranslator global cancellation - explicitly set by stop button
        try:
            from manga_translator import MangaTranslator
            if MangaTranslator.is_globally_cancelled():
                print("[CANCEL_CHECK] MangaTranslator global cancellation is True")
                return True
        except Exception:
            pass
        
        # Check UnifiedClient global cancellation - explicitly set by stop button
        try:
            from unified_api_client import UnifiedClient
            if UnifiedClient.is_globally_cancelled():
                print("[CANCEL_CHECK] UnifiedClient global cancellation is True")
                return True
        except Exception:
            pass
        
        return False
    except Exception as e:
        print(f"[CANCEL_CHECK] Error checking cancellation: {e}")
        return False

# MODULE-LEVEL WORKER FUNCTION for parallel save processing (pickleable)
def _process_save_task_worker(region_index: int, current_image: str, trans_text: str, task: dict) -> dict:
    """Worker function for ProcessPoolExecutor - processes a single save task.
    
    This function is pickleable and runs in a separate process.
    It cannot access GUI state or non-pickleable objects.
    
    Args:
        region_index: Index of the region to process
        current_image: Path to the current image
        trans_text: Translation text for the region
        task: Task metadata dict
        
    Returns:
        dict with 'success' bool and optional 'error' string
    """
    import time
    start_time = time.time_ns()
    
    try:
        print(f"[PARALLEL_WORKER] Processing region {region_index} in process")
        
        # Since we can't access GUI state from a worker process,
        # we just return success to indicate the task was received
        # The actual GUI update will happen in the completion callback
        
        end_time = time.time_ns()
        duration_ms = (end_time - start_time) / 1_000_000
        
        print(f"[PARALLEL_WORKER] Region {region_index} processed in {duration_ms:.2f}ms")
        
        return {
            'success': True,
            'region_index': region_index,
            'duration_ms': duration_ms
        }
        
    except Exception as e:
        print(f"[PARALLEL_WORKER] Error processing region {region_index}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'region_index': region_index,
            'error': str(e)
        }

# MODULE-LEVEL RENDER FUNCTION (pickle-able for ProcessPoolExecutor)
def _render_single_region_overlay(region_data: dict, image_size: tuple, render_settings: dict) -> Image.Image:
    """
    Render a single region overlay as RGBA PIL Image (pickle-able for multiprocessing)
    
    Args:
        region_data: dict with 'text', 'bbox' (x,y,w,h), 'vertices'
        image_size: (width, height)
        render_settings: dict with font/color/outline settings
    
    Returns:
        PIL RGBA Image of full size with transparent overlay
    """
    try:
        # Create transparent overlay
        overlay = Image.new('RGBA', image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Extract data
        text = region_data.get('translated_text', '')
        if not text:
            return overlay
        
        x, y, w, h = region_data.get('bbox', (0, 0, 100, 100))
        
        # Get settings
        font_size = render_settings.get('font_size', 24)
        font_path = render_settings.get('font_path')
        text_color = tuple(render_settings.get('text_color', (102, 0, 0))) + (255,)
        outline_color = tuple(render_settings.get('outline_color', (255, 255, 255))) + (255,)
        outline_width = render_settings.get('outline_width', 2)
        force_caps = render_settings.get('force_caps_lock', False)
        
        if force_caps:
            text = text.upper()
        
        # Load font
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Simple text wrapping
        lines = text.split('\n')
        line_height = int(font_size * 1.2)
        total_height = len(lines) * line_height
        start_y = y + (h - total_height) // 2
        
        # Render each line
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Get text width
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except Exception:
                text_width = len(line) * font_size * 0.6
            
            tx = x + (w - text_width) // 2
            ty = start_y + i * line_height
            
            # Clamp to bounds
            tx = max(0, min(tx, image_size[0] - 10))
            ty = max(0, min(ty, image_size[1] - 10))
            
            # Render with outline using PIL stroke parameter
            try:
                draw.text(
                    (tx, ty), line, font=font,
                    fill=text_color,
                    stroke_width=outline_width,
                    stroke_fill=outline_color
                )
            except TypeError:
                # Fallback for older PIL
                if outline_width > 0:
                    for dx in range(-outline_width, outline_width + 1):
                        for dy in range(-outline_width, outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((tx + dx, ty + dy), line, font=font, fill=outline_color)
                draw.text((tx, ty), line, font=font, fill=text_color)
        
        return overlay
    except Exception as e:
        print(f"[RENDER] Error rendering region: {e}")
        return Image.new('RGBA', image_size, (0, 0, 0, 0))
    
def _on_detect_text_clicked(self):
    """Detect text button - run detection in background thread"""
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    # This MUST happen on the main thread BEFORE any cancellation checks
    _reset_cancellation_flags(self)
    
    try:
        # GUARD: Prevent processing during rendering
        if hasattr(self, '_rendering_in_progress') and self._rendering_in_progress:
            print("[DEBUG] Rendering in progress, ignoring detect click")
            return
        
        # Get current image path
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            self._log("⚠️ No image loaded for detection", "warning")
            return
        
        # Disable the detect button to prevent multiple clicks
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'detect_btn'):
            self.image_preview_widget.detect_btn.setEnabled(False)
            self.image_preview_widget.detect_btn.setText("Detecting...")
        
        # Clear cleaned image path when starting new detection (new workflow)
        if hasattr(self, '_cleaned_image_path'):
            self._cleaned_image_path = None
            print(f"[DETECT] Cleared cleaned image path for new workflow")
        
        image_path = self.image_preview_widget.current_image_path
        
        # STATE ISOLATION: Track which image detection was started for
        # This prevents results from appearing on the wrong image if user switches images
        self._detection_started_for_image = os.path.abspath(image_path)
        print(f"[STATE_ISOLATION] Detection started for: {os.path.basename(self._detection_started_for_image)}")
        
        # Add processing overlay effect (after tracking image)
        _add_processing_overlay(self, )
        
        # Get detection settings for the background thread
        detection_config = _get_detection_config(self, )
        
        # Manual detection should exclude EMPTY BUBBLES to avoid duplicate container boxes
        if detection_config.get('detect_empty_bubbles', True):
            detection_config['detect_empty_bubbles'] = False
            self._log("🚫 Manual detection: Excluding empty bubble regions (container boxes)", "info")
        
        self._log(f"🔍 Starting background detection: {os.path.basename(image_path)}", "info")
        
        # Run detection in background thread
        import threading
        thread = threading.Thread(target=_run_detect_background, args=(self, image_path, detection_config),
                                daemon=True)
        thread.start()
        
    except Exception as e:
        import traceback
        self._log(f"❌ Detect setup failed: {str(e)}", "error")
        print(f"Detect setup error traceback: {traceback.format_exc()}")
        _restore_detect_button(self, )

def _run_detect_background(self, image_path: str, detection_config: dict):
    """Run the actual detection process in background thread"""
    detector = None  # Initialize for cleanup
    temp_translator = None  # Track temporary translator for pool cleanup
    
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    _reset_cancellation_flags(self)
    
    try:
        # ===== CANCELLATION CHECK: At start of detection =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Detection cancelled before starting", "warning")
            print(f"[DETECT] Cancelled at start")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        import cv2
        from bubble_detector import BubbleDetector
        from manga_translator import MangaTranslator
        from unified_api_client import UnifiedClient
        
        # Use pool-aware detector checkout like we do for inpainter
        try:
            ocr_config = _get_ocr_config(self, ) if hasattr(self, 'main_gui') else {}
            api_key = self.main_gui.config.get('api_key', '') or 'dummy'
            model = self.main_gui.config.get('model', 'gpt-4o-mini')
            uc = UnifiedClient(model=model, api_key=api_key)
            temp_translator = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=lambda m, l: None, skip_inpainter_init=True)
            # Use the translator's pool-aware method to get detector
            detector = temp_translator._get_thread_bubble_detector()
            # Immediately update GUI pool tracker after checkout
            self.update_queue.put(('update_pool_tracker', None))
        except Exception as e:
            print(f"[DETECT] Failed to get detector from pool, creating standalone: {e}")
            detector = BubbleDetector()
            temp_translator = None
        
        # Extract settings from config
        detector_type = detection_config['detector_type']
        model_path = detection_config['model_path']
        model_url = detection_config['model_url']
        confidence = detection_config['confidence']
        detect_free_text = detection_config.get('detect_free_text', True)
        detect_empty_bubbles = detection_config.get('detect_empty_bubbles', True)
        detect_text_bubbles = detection_config.get('detect_text_bubbles', True)
        
        # Log detection settings
        self._log(f"📋 Detection settings: Empty bubbles={'✓' if detect_empty_bubbles else '✗'}, Text bubbles={'✓' if detect_text_bubbles else '✗'}, Free text={'✓' if detect_free_text else '✗'}", "info")
        
        # Load the appropriate model based on user settings
        success = False
        if detector_type == 'rtdetr_onnx':
            # Use model_path if available, otherwise use model_url
            model_source = model_path if (model_path and os.path.exists(model_path)) else model_url
            success = detector.load_rtdetr_onnx_model(model_source)
            self._log(f"📥 Loading RT-DETR ONNX model: {os.path.basename(model_source) if model_path else model_source}", "info")
        elif detector_type == 'rtdetr':
            success = detector.load_rtdetr_model(model_id=model_url or 'ogkalu/comic-text-and-bubble-detector')
            self._log(f"📥 Loading RT-DETR model: {model_url}", "info")
        elif detector_type == 'yolo' and model_path:
            success = detector.load_model(model_path)
            self._log(f"📥 Loading YOLO model: {os.path.basename(model_path)}", "info")
        elif detector_type == 'custom' and model_path:
            success = detector.load_model(model_path)
            self._log(f"📥 Loading custom model: {os.path.basename(model_path)}", "info")
        else:
            # Default fallback
            success = detector.load_rtdetr_onnx_model('ogkalu/comic-text-and-bubble-detector')
            self._log(f"📥 Loading default RT-DETR ONNX model", "info")
        
        if not success:
            self._log("❌ Failed to load bubble detection model", "error")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            self._log(f"❌ Failed to load image: {os.path.basename(image_path)}", "error")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        # ===== CANCELLATION CHECK: After model loading, before detection =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Detection cancelled after model loading", "warning")
            print(f"[DETECT] Cancelled after model loading")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        # Run bubble detection
        self._log(f"🤖 Running bubble detection (confidence: {confidence:.2f})", "info")
        
        # Use appropriate detection method based on detector type
        if detector_type in ['rtdetr_onnx', 'rtdetr']:
            # For RT-DETR, get detailed detection results to avoid double boxes
            if detector_type == 'rtdetr_onnx' and hasattr(detector, 'detect_with_rtdetr_onnx'):
                detection_results = detector.detect_with_rtdetr_onnx(image_path, confidence=confidence, return_all_bubbles=False)
                # Combine enabled bubble types based on settings
                empty_bubbles = detection_results.get('bubbles', [])
                text_bubbles = detection_results.get('text_bubbles', [])
                text_free = detection_results.get('text_free', [])
                
                boxes = []
                if detect_empty_bubbles:
                    boxes.extend(empty_bubbles)
                if detect_text_bubbles:
                    boxes.extend(text_bubbles)
                if detect_free_text:
                    boxes.extend(text_free)
                
                self._log(f"📋 RT-DETR ONNX: {len(empty_bubbles)} empty + {len(text_bubbles)} text bubbles + {len(text_free)} free text", "info")
                self._log(f"📊 After filtering: {len(boxes)} regions included", "info")
            elif detector_type == 'rtdetr' and hasattr(detector, 'detect_with_rtdetr'):
                detection_results = detector.detect_with_rtdetr(image_path, confidence=confidence, return_all_bubbles=False)
                # Combine enabled bubble types based on settings
                empty_bubbles = detection_results.get('bubbles', [])
                text_bubbles = detection_results.get('text_bubbles', [])
                text_free = detection_results.get('text_free', [])
                
                boxes = []
                if detect_empty_bubbles:
                    boxes.extend(empty_bubbles)
                if detect_text_bubbles:
                    boxes.extend(text_bubbles)
                if detect_free_text:
                    boxes.extend(text_free)
                
                self._log(f"📋 RT-DETR: {len(empty_bubbles)} empty + {len(text_bubbles)} text bubbles + {len(text_free)} free text", "info")
                self._log(f"📊 After filtering: {len(boxes)} regions included", "info")
            else:
                # Fallback to old method
                boxes = detector.detect_bubbles(image_path, confidence=confidence, use_rtdetr=True)
        else:
            boxes = detector.detect_bubbles(image_path, confidence=confidence)
        
        if not boxes:
            self._log("⚠️ No text regions detected", "warning")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        # ===== CANCELLATION CHECK: After detection, before processing results =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Detection cancelled after detection", "warning")
            print(f"[DETECT] Cancelled after detection")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        self._log(f"✅ Found {len(boxes)} text regions", "success")
        
        # Merge overlapping/nested boxes to avoid duplicates (match regular pipeline)
        try:
            from manga_translator import merge_overlapping_boxes
            # Normalize to int (x,y,w,h)
            norm_boxes = []
            for b in boxes:
                try:
                    x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    norm_boxes.append([x, y, w, h])
                except Exception:
                    continue
            original_count = len(norm_boxes)
            merged_boxes = merge_overlapping_boxes(norm_boxes, containment_threshold=0.3, overlap_threshold=0.5)
            if merged_boxes and len(merged_boxes) < original_count:
                self._log(f"✅ Merged {original_count} boxes → {len(merged_boxes)} unique regions", "debug")
            boxes = merged_boxes or norm_boxes
        except Exception as me:
            print(f"[DETECT] Merge step failed or unavailable: {me}")
        
        # Debug: Print first few boxes to inspect
        print(f"[DETECT] First 3 boxes from detector/merge:")
        for i, box in enumerate(boxes[:3]):
            print(f"[DETECT]   Box {i}: {box}")
        
        # Build RT-DETR class membership sets (if available) for bubble-aware metadata
        def _norm_box_local(b):
            try:
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            except Exception:
                return tuple(b)
        text_bubble_set, free_text_set, empty_bubble_set = set(), set(), set()
        try:
            if isinstance(detection_results, dict):
                text_bubble_set = set(_norm_box_local(b) for b in (detection_results.get('text_bubbles') or []))
                free_text_set = set(_norm_box_local(b) for b in (detection_results.get('text_free') or []))
                empty_bubble_set = set(_norm_box_local(b) for b in (detection_results.get('bubbles') or []))
        except Exception:
            # No RT-DETR class info available
            pass
        
        # ===== CANCELLATION CHECK: Before processing boxes =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Detection cancelled before processing boxes", "warning")
            print(f"[DETECT] Cancelled before processing boxes")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        # Process detection boxes and store regions
        regions = []
        seen_boxes = set()  # Track boxes to detect duplicates
        
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                # Validate and convert coordinates
                try:
                    x, y, width, height = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # Calculate x2, y2 from width and height, then clamp to image bounds
                    x1 = max(0, min(x, image.shape[1] - 1))
                    y1 = max(0, min(y, image.shape[0] - 1))
                    x2 = max(x1 + 1, min(x + width, image.shape[1]))
                    y2 = max(y1 + 1, min(y + height, image.shape[0]))
                    
                    # Create box signature for duplicate detection
                    box_sig = (x1, y1, x2, y2)
                    
                    # Skip if we've already seen this exact box
                    if box_sig in seen_boxes:
                        print(f"[DETECT] Skipping duplicate box {i}: {box_sig}")
                        continue
                    seen_boxes.add(box_sig)
                    
                    # Expand ellipse by 10% if circle mode is active (only for Detect)
                    if getattr(self, '_use_circle_shapes', False):
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        w = (x2 - x1)
                        h = (y2 - y1)
                        scale = 1.20
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        nx1 = int(round(cx - new_w / 2))
                        ny1 = int(round(cy - new_h / 2))
                        nx2 = nx1 + new_w
                        ny2 = ny1 + new_h
                        # Clamp to image bounds
                        nx1 = max(0, min(nx1, image.shape[1] - 1))
                        ny1 = max(0, min(ny1, image.shape[0] - 1))
                        nx2 = max(nx1 + 1, min(nx2, image.shape[1]))
                        ny2 = max(ny1 + 1, min(ny2, image.shape[0]))
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                    
                    # Classify bubble type using RT-DETR sets if available
                    norm_box = (x1, y1, x2 - x1, y2 - y1)
                    if norm_box in free_text_set:
                        bubble_type = 'free_text'
                    elif norm_box in text_bubble_set:
                        bubble_type = 'text_bubble'
                    elif norm_box in empty_bubble_set:
                        bubble_type = 'empty_bubble'
                    else:
                        # Heuristic fallback
                        bubble_type = 'text_bubble'
                    region_type = 'free_text' if bubble_type == 'free_text' else 'text_bubble'
                    
                    # Store region for workflow continuity
                    region_dict = {
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # (x, y, width, height)
                        'coords': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  # Corner coordinates (use clamped values)
                        'confidence': getattr(box, 'confidence', confidence) if hasattr(box, 'confidence') else confidence,
                        'shape': 'ellipse' if getattr(self, '_use_circle_shapes', False) else 'rect',
                        'bubble_type': bubble_type,
                        'region_type': region_type,
                        'bubble_bounds': [x1, y1, x2 - x1, y2 - y1]
                    }
                    regions.append(region_dict)
                    print(f"[DETECT] Added region {len(regions)-1}: bbox={region_dict['bbox']}, type={bubble_type}")
                    
                except (ValueError, IndexError) as e:
                    self._log(f"⚠️ Skipping invalid box {i}: {e}", "warning")
                    continue
        
        print(f"[DETECT] Total regions after deduplication: {len(regions)}")
        
        # ===== CANCELLATION CHECK: Final check before sending results =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Detection cancelled - NOT sending results", "warning")
            print(f"[DETECT] Cancelled at final check - NOT sending detect_results")
            self.update_queue.put(('detect_button_restore', None))
            return
        
        # Send detection results to main thread using update queue
        self.update_queue.put(('detect_results', {
            'image_path': image_path,
            'regions': regions
        }))
        
        self._log(f"🎯 Detection complete! Found {len(regions)} valid regions", "success")
        
    except Exception as e:
        import traceback
        self._log(f"❌ Background detection failed: {str(e)}", "error")
        print(f"Background detect error traceback: {traceback.format_exc()}")
    finally:
        # Return detector to pool if checked out via temporary translator
        try:
            if temp_translator is not None:
                temp_translator._return_bubble_detector_to_pool()
                # Immediately update GUI pool tracker after return
                self.update_queue.put(('update_pool_tracker', None))
        except Exception as e:
            print(f"[DETECT] Failed to return detector to pool: {e}")
        # Always restore the button using thread-safe update queue
        self.update_queue.put(('detect_button_restore', None))

def _restore_detect_button(self):
    """Restore the detect button to its original state"""
    try:
        # Remove processing overlay effect for the image that was being processed
        image_path = getattr(self, '_detection_started_for_image', None)
        _remove_processing_overlay(self, image_path)
        
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'detect_btn'):
            self.image_preview_widget.detect_btn.setEnabled(True)
            self.image_preview_widget.detect_btn.setText("Detect Text")
    except Exception:
        pass

def _clear_detection_state_for_image(self, image_path: str):
    """Completely clear detection/recognition rectangles for a specific image.
    - Clears in-memory _current_regions if this is the active image
    - Clears persisted detection_regions, viewer_rectangles and recognized_texts
    """
    try:
        if not image_path:
            return
        # Clear in-memory regions if we're on this image
        try:
            if getattr(self, '_current_image_path', None) == image_path:
                self._current_regions = []
        except Exception:
            pass
        # Clear persisted state
        if hasattr(self, 'image_state_manager') and self.image_state_manager:
            try:
                self.image_state_manager.update_state(image_path, {
                    'detection_regions': [],
                    'viewer_rectangles': [],
                    'recognized_texts': []
                }, save=True)
            except Exception:
                pass
    except Exception as e:
        print(f"[STATE] Failed to clear detection state: {e}")

def _clear_cross_image_state(self):
    """Clear recognition and translation data to prevent state leaking between images.
    This is the main fix for cross-contamination of OCR/translation states.
    """
    try:
        # Clear recognition data to prevent "Edit OCR" tooltips from previous images
        if hasattr(self, '_recognition_data'):
            old_count = len(self._recognition_data) if self._recognition_data else 0
            self._recognition_data = {}
            if old_count > 0:
                print(f"[STATE_ISOLATION] Cleared {old_count} recognition data entries")
        
        # Clear translation data to prevent "Edit Translation" tooltips from previous images  
        if hasattr(self, '_translation_data'):
            old_count = len(self._translation_data) if self._translation_data else 0
            self._translation_data = {}
            if old_count > 0:
                print(f"[STATE_ISOLATION] Cleared {old_count} translation data entries")
        
        # Clear recognized texts list
        if hasattr(self, '_recognized_texts'):
            old_count = len(self._recognized_texts) if self._recognized_texts else 0
            self._recognized_texts = []
            if old_count > 0:
                print(f"[STATE_ISOLATION] Cleared {old_count} recognized texts")
        
        # Clear translated texts list
        if hasattr(self, '_translated_texts'):
            old_count = len(self._translated_texts) if self._translated_texts else 0
            self._translated_texts = []
            if old_count > 0:
                print(f"[STATE_ISOLATION] Cleared {old_count} translated texts")
        
        # Clear image-specific tracking variables
        if hasattr(self, '_recognized_texts_image_path'):
            self._recognized_texts_image_path = None
            print(f"[STATE_ISOLATION] Cleared recognized texts image path tracking")
        
        # Clear current state image path to prevent stale references
        if hasattr(self, '_current_state_image_path'):
            old_path = getattr(self, '_current_state_image_path', None)
            self._current_state_image_path = None
            if old_path:
                print(f"[STATE_ISOLATION] Cleared state image path tracking (was: {os.path.basename(old_path)})")
        
        # Clear cleaned image path to prevent using previous image's cleaned base for rendering
        if hasattr(self, '_cleaned_image_path'):
            old_cleaned = getattr(self, '_cleaned_image_path', None)
            self._cleaned_image_path = None
            if old_cleaned:
                print(f"[STATE_ISOLATION] Cleared cleaned image path (was: {os.path.basename(old_cleaned)})")
        
        # Clear detection tracking to prevent results from appearing on wrong image
        if hasattr(self, '_detection_started_for_image'):
            old_detection = getattr(self, '_detection_started_for_image', None)
            self._detection_started_for_image = None
            if old_detection:
                print(f"[STATE_ISOLATION] Cleared detection tracking (was: {os.path.basename(old_detection)})")
        
        print(f"[STATE_ISOLATION] Cross-image state isolation completed")
        
    except Exception as e:
        print(f"[STATE_ISOLATION] Failed to clear cross-image state: {e}")

def _persist_current_image_state(self):
    """Persist the current image's state (rectangles, overlays, paths) before switching.
    IMPORTANT: Merge into existing state to avoid wiping recognized/translated texts.
    Also cleans up old overlays to prevent RAM accumulation.
    """
    try:
        if not hasattr(self, '_current_image_path') or not self._current_image_path:
            return
        
        if not hasattr(self, 'image_state_manager'):
            return
        
        image_path = self._current_image_path
        
        # MEMORY OPTIMIZATION: Clean up overlays for the image we're leaving
        # This prevents overlay accumulation in memory when switching between many images
        try:
            if hasattr(self, '_text_overlays_by_image') and image_path in self._text_overlays_by_image:
                # Keep track of overlay count for logging
                overlay_count = len(self._text_overlays_by_image[image_path])
                if overlay_count > 0:
                    # Clean up overlays (they'll be recreated from state when we return)
                    self.clear_text_overlays_for_image(image_path)
                    print(f"[MEMORY] Cleaned up {overlay_count} overlays for {os.path.basename(image_path)} to free RAM")
        except Exception as cleanup_err:
            print(f"[MEMORY] Warning: Could not clean up overlays: {cleanup_err}")
        
        # Collect current state (partial)
        partial = {}
        
        # Store detection rectangles
        if hasattr(self, '_current_regions') and self._current_regions:
            partial['detection_regions'] = self._current_regions
            #print(f"[STATE] Persisting {len(self._current_regions)} detection regions for {os.path.basename(image_path)}")
        
        # Store recognition overlays (if any)
        if hasattr(self.image_preview_widget.viewer, 'overlay_rects'):
            partial['overlay_rects'] = self.image_preview_widget.viewer.overlay_rects.copy()
            if partial['overlay_rects']:
                print(f"[STATE] Persisting {len(partial['overlay_rects'])} overlay rects for {os.path.basename(image_path)}")
        
        # Store cleaned image path (only if it belongs to this image)
        if hasattr(self, '_cleaned_image_path') and self._cleaned_image_path:
            try:
                cleaned_base = os.path.splitext(os.path.basename(self._cleaned_image_path))[0].replace('_cleaned', '')
                current_base = os.path.splitext(os.path.basename(image_path))[0].replace('_cleaned', '')
                if cleaned_base == current_base:
                    partial['cleaned_image_path'] = self._cleaned_image_path
                    print(f"[STATE] Persisting cleaned image path: {os.path.basename(self._cleaned_image_path)}")
                else:
                    print(f"[STATE] Skipping stale cleaned_image_path: {os.path.basename(self._cleaned_image_path)} (current: {os.path.basename(image_path)})")
            except Exception:
                pass
        
        # Store rendered image path (if exists)
        if hasattr(self, '_rendered_images_map') and image_path in self._rendered_images_map:
            partial['rendered_image_path'] = self._rendered_images_map[image_path]
            print(f"[STATE] Persisting rendered image path: {os.path.basename(self._rendered_images_map[image_path])}")
        
        # Store viewer rectangles for visual state
        if hasattr(self.image_preview_widget, 'viewer') and self.image_preview_widget.viewer.rectangles:
            # Store geometry + shape metadata (rect/ellipse/polygon) in SCENE coords
            rect_data = []
            for rect_item in self.image_preview_widget.viewer.rectangles:
                try:
                    br = rect_item.sceneBoundingRect()
                except Exception:
                    br = rect_item.rect()
                entry = {
                    'x': br.x(),
                    'y': br.y(),
                    'width': br.width(),
                    'height': br.height(),
                    'shape': getattr(rect_item, 'shape_type', 'rect')
                }
                # Persist polygon points if lasso/path
                try:
                    if entry['shape'] == 'polygon' and hasattr(rect_item, 'path'):
                        poly = rect_item.mapToScene(rect_item.path().toFillPolygon())
                        pts = []
                        for p in poly:
                            pts.append([float(p.x()), float(p.y())])
                        if len(pts) >= 3:
                            entry['polygon'] = pts
                except Exception:
                    pass
                rect_data.append(entry)
            partial['viewer_rectangles'] = rect_data
            print(f"[STATE] Persisting {len(rect_data)} viewer shapes for {os.path.basename(image_path)}")
        
        # Merge with existing state to preserve recognized/translated texts and other keys
        prev = self.image_state_manager.get_state(image_path) or {}
        merged = {**prev, **partial}
        self.image_state_manager.set_state(image_path, merged, save=True)
        print(f"[STATE] Saved merged state for {os.path.basename(image_path)} (preserved OCR/translation)")
        
        # Force immediate flush to disk to ensure state persists across sessions
        try:
            self.image_state_manager.flush()
        except Exception as flush_err:
            print(f"[STATE] Warning: Failed to flush state: {flush_err}")
        
    except Exception as e:
        print(f"[STATE] Failed to persist state: {e}")
        import traceback
        traceback.print_exc()

def _rehydrate_text_state_from_persisted(self, image_path: str):
    """Rebuild in-memory recognition/translation data from persisted state without redrawing.
    Returns (ocr_count, trans_count).
    
    STATE ISOLATION: This function properly scopes data to image_path and tags the restored
    state with _current_state_image_path to prevent cross-contamination.
    """
    try:
        if not hasattr(self, 'image_state_manager'):
            return (0, 0)
        state = self.image_state_manager.get_state(image_path) or {}
        
        # STATE ISOLATION: Tag the current image path so we can validate later
        self._current_state_image_path = image_path
        
        # Recognized texts
        rec = state.get('recognized_texts') or []
        active_rec = []
        recognition_data = {}
        for i, r in enumerate(rec):
            if isinstance(r, dict) and r.get('deleted'):
                continue
            if isinstance(r, str):
                active_rec.append({'text': r, 'bbox': [0, 0, 100, 100], 'region_index': i})
                recognition_data[int(i)] = {'text': r, 'bbox': [0, 0, 100, 100]}
            elif isinstance(r, dict) and 'text' in r:
                idx = r.get('region_index', i)
                active_rec.append({'text': r.get('text', ''), 'bbox': r.get('bbox', [0, 0, 100, 100]), 'region_index': idx})
                recognition_data[int(idx)] = {'text': r.get('text', ''), 'bbox': r.get('bbox', [0, 0, 100, 100])}
        self._recognized_texts = active_rec
        try:
            self._recognized_texts_image_path = image_path
        except Exception:
            pass
        self._recognition_data = recognition_data
        
        # Translated texts
        trans = state.get('translated_texts') or []
        active_trans = []
        translation_data = {}
        for i, t in enumerate(trans):
            if isinstance(t, dict) and t.get('deleted'):
                continue
            idx = t.get('original', {}).get('region_index', i) if isinstance(t, dict) else i
            if isinstance(t, dict):
                translation_data[int(idx)] = {
                    'original': t.get('original', {}).get('text', ''),
                    'translation': t.get('translation', '')
                }
            active_trans.append(t)
        self._translated_texts = active_trans
        self._translation_data = translation_data
        
        # CRITICAL: Update translation data image path to match the current image
        # This prevents "Cannot render: Translation data is for X but you're viewing Y" errors
        self._translation_data_image_path = image_path
        self._translating_image_path = image_path
        
        print(f"[STATE_ISOLATION] Rehydrated state for {os.path.basename(image_path)}: {len(active_rec)} OCR, {len(active_trans)} translations")
        return (len(active_rec), len(active_trans))
    except Exception as e:
        print(f"[STATE_ISOLATION] Failed to rehydrate state: {e}")
        return (0, 0)

def _validate_and_clean_stale_state(self, image_path: str):
    """Validate state and clear references to non-existent output files.
    
    If cleaned_image_path or rendered_image_path don't exist anymore,
    clear them from state along with ALL dependent data:
    - translated_texts
    - recognized_texts (OCR data)
    - detection_regions (detection boxes)
    - viewer_rectangles (manually adjusted boxes)
    - overlay_rects
    """
    try:
        if not hasattr(self, 'image_state_manager') or not self.image_state_manager:
            return
        
        state = self.image_state_manager.get_state(image_path)
        if not state:
            return
        
        state_changed = False
        cleaned_exists = False
        rendered_exists = False
        
        # Check cleaned_image_path
        cleaned_path = state.get('cleaned_image_path')
        if cleaned_path:
            if os.path.exists(cleaned_path):
                print(f"[STATE_CLEAN] Cleaned image exists: {os.path.basename(cleaned_path)}")
                cleaned_exists = True
            else:
                print(f"[STATE_CLEAN] Cleaned image no longer exists: {os.path.basename(cleaned_path)}")
                state.pop('cleaned_image_path', None)
                state_changed = True
        
        # Check rendered_image_path (translated output)
        rendered_path = state.get('rendered_image_path')
        if rendered_path:
            if os.path.exists(rendered_path):
                print(f"[STATE_CLEAN] Rendered/translated image exists: {os.path.basename(rendered_path)}")
                rendered_exists = True
            else:
                print(f"[STATE_CLEAN] Rendered/translated image no longer exists: {os.path.basename(rendered_path)}")
                state.pop('rendered_image_path', None)
                state_changed = True
        
        # If NEITHER cleaned nor rendered output exists, clear only TRANSLATION output state.
        # Detection regions, viewer rectangles, and recognized texts are NOT orphaned —
        # they are created by Detect/Recognize steps BEFORE any output files exist.
        if not cleaned_exists and not rendered_exists:
            orphaned_keys = []
            
            # Only clear translated_texts — these genuinely require a rendered output
            if state.get('translated_texts'):
                orphaned_keys.append('translated_texts')
                state.pop('translated_texts', None)
            
            # Keep recognized_texts, detection_regions, viewer_rectangles, overlay_rects
            # — they are valid intermediate state from Detect/Recognize steps
            
            if orphaned_keys:
                print(f"[STATE_CLEAN] No output files exist - clearing orphaned translation state: {', '.join(orphaned_keys)}")
                state_changed = True
        
        # If only translated output is missing but cleaned exists, just clear translation data
        elif cleaned_exists and not rendered_exists:
            if state.get('translated_texts'):
                print(f"[STATE_CLEAN] Translated output missing but cleaned exists - clearing translated_texts only")
                state.pop('translated_texts', None)
                state_changed = True
        
        # Save cleaned state if changed
        if state_changed:
            self.image_state_manager.set_state(image_path, state, save=True)
            print(f"[STATE_CLEAN] Cleaned stale state for {os.path.basename(image_path)}")
        else:
            print(f"[STATE_CLEAN] No stale state found for {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"[STATE_CLEAN] Error validating state: {e}")
        import traceback
        traceback.print_exc()

def _restore_image_state(self, image_path: str):
    """Restore persisted state for an image (rectangles, overlays, paths)"""
    try:
        if not hasattr(self, 'image_state_manager'):
            return
        
        # CRITICAL: Always reset path tracking variables to current image to prevent
        # "Cannot render: Translation data is for X but you're viewing Y" errors
        # This must happen BEFORE we check for saved state
        self._translation_data_image_path = image_path
        self._translating_image_path = image_path
        self._current_state_image_path = image_path
        print(f"[STATE_ISOLATION] Reset path tracking to: {os.path.basename(image_path)}")
        
        # CRITICAL: Validate and clean stale state BEFORE restoration
        _validate_and_clean_stale_state(self, image_path)
        
        # Get saved state (after cleaning)
        state = self.image_state_manager.get_state(image_path)
        if not state:
            print(f"[STATE] No saved state for {os.path.basename(image_path)}")
            return
        
        print(f"[STATE] Restoring state for {os.path.basename(image_path)}")
        
        # Prefer viewer_rectangles (latest manual adjustments) to restore boxes; fallback to detection_regions
        used_boxes = False
        if 'viewer_rectangles' in state and state['viewer_rectangles']:
            viewer = self.image_preview_widget.viewer
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem

            # Determine coloring based on recognized/translated texts
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            has_any_text = bool([t for t in (recognized_texts or []) if not (isinstance(t, dict) and t.get('deleted'))]) or \
                           bool([t for t in (translated_texts or []) if not (isinstance(t, dict) and t.get('deleted'))])

            for idx, rect_data in enumerate(state['viewer_rectangles']):
                shape = rect_data.get('shape', 'rect')
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])

                # Blue if this rectangle has recognized/translated text, else green
                has_text_for_this_rect = False
                if has_any_text:
                    has_recognized = (idx < len(recognized_texts) and not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')))
                    has_translation = (idx < len(translated_texts) and not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')))
                    has_text_for_this_rect = has_recognized or has_translation

                if has_text_for_this_rect:
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                pen.setCosmetic(True)

                if shape == 'ellipse':
                    item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    item = MoveableRectItem(rect, pen=pen, brush=brush)

                # Attach viewer and metadata
                try:
                    item._viewer = viewer
                    item.region_index = idx
                    # CRITICAL: Always set is_recognized based on text state, not just when true
                    item.is_recognized = has_text_for_this_rect
                    # Attach move sync and context menu
                    _attach_move_sync_to_rectangle(self, item, idx)
                    _add_context_menu_to_rectangle(self, item, idx)
                except Exception:
                    pass

                viewer._scene.addItem(item)
                viewer.rectangles.append(item)
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes (preferred)")
            
            # CRITICAL: Rehydrate text state so OCR/translation data is available in memory
            ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
            if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                try:
                    _update_rectangles_with_recognition(self, self._recognized_texts)
                    print(f"[STATE] Attached OCR tooltips to {ocr_count} rectangles (viewer_rectangles branch)")
                except Exception:
                    pass
            used_boxes = True
        
        if not used_boxes and 'detection_regions' in state:
            self._current_regions = state['detection_regions']
            print(f"[STATE] Restored {len(self._current_regions)} detection regions")
            # Redraw detection boxes on preview
            if hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                self.image_preview_widget.viewer.clear_rectangles()
            _draw_detection_boxes_on_preview(self, )

            # If recognized/translated texts exist, promote matching rectangles to BLUE and attach metadata
            try:
                viewer = self.image_preview_widget.viewer
                rects = getattr(viewer, 'rectangles', []) or []
                recognized_texts = state.get('recognized_texts') or []
                translated_texts = state.get('translated_texts') or []
                max_len = max(len(recognized_texts), len(translated_texts))
                if max_len and rects:
                    from PySide6.QtGui import QPen, QBrush, QColor
                    for idx, rect_item in enumerate(rects):
                        has_recognized = (idx < len(recognized_texts) and not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')))
                        has_translation = (idx < len(translated_texts) and not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')))
                        has_text = has_recognized or has_translation
                        if has_text:
                            # Make blue and mark recognized
                            try:
                                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                            except Exception:
                                pass
                        # CRITICAL: Always set is_recognized based on text state, not just when true
                        try:
                            rect_item.is_recognized = has_text
                        except Exception:
                            pass
                        # Always ensure region_index and context menu are attached
                        try:
                            rect_item.region_index = idx
                            _attach_move_sync_to_rectangle(self, rect_item, idx)
                            _add_context_menu_to_rectangle(self, rect_item, idx)
                        except Exception:
                            pass
                    print(f"[STATE] Promoted {sum(1 for i in range(min(len(rects), max_len)) if (i < len(recognized_texts) and not (isinstance(recognized_texts[i], dict) and recognized_texts[i].get('deleted'))) or (i < len(translated_texts) and not (isinstance(translated_texts[i], dict) and translated_texts[i].get('deleted'))))} rectangles to BLUE from recognized/translated texts")
                # Populate in-memory text state and add OCR tooltips
                ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
                if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                    try:
                        _update_rectangles_with_recognition(self, self._recognized_texts)
                        print(f"[STATE] Attached OCR tooltips to {ocr_count} rectangles (detection branch)")
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Restore exclusion status after rectangles are drawn
        try:
            print(f"[RECT_0_DEBUG] About to call _restore_exclusion_status_from_state for: {image_path}")
            _restore_exclusion_status_from_state(self, image_path)
        except Exception as e:
            print(f"[STATE] Failed to restore exclusion status: {e}")
        
        # Restore custom inpainting iterations after rectangles are drawn
        try:
            _restore_inpainting_iterations_from_state(self, image_path)
        except Exception as e:
            print(f"[STATE] Failed to restore inpainting iterations: {e}")
        
        # Restore overlay rectangles
        if 'overlay_rects' in state and state['overlay_rects']:
            if hasattr(self.image_preview_widget.viewer, 'overlay_rects'):
                self.image_preview_widget.viewer.overlay_rects = state['overlay_rects'].copy()
                print(f"[STATE] Restored {len(state['overlay_rects'])} overlay rects")
        
        # Restore cleaned image path (with validation + filesystem discovery)
        resolved_cleaned = _resolve_cleaned_image_for_render(self, image_path)
        if resolved_cleaned:
            self._cleaned_image_path = resolved_cleaned
            print(f"[STATE] Restored cleaned image path: {os.path.basename(self._cleaned_image_path)}")
        elif 'cleaned_image_path' in state:
            print(f"[STATE] Skipped corrupted cleaned_image_path: {os.path.basename(state['cleaned_image_path'])}")
        
        # Restore rendered image path mapping if available
        if 'rendered_image_path' in state:
            rendered_path = state['rendered_image_path']
            if os.path.exists(rendered_path):
                # Store the translated path for reference
                if hasattr(self.image_preview_widget, 'current_translated_path'):
                    self.image_preview_widget.current_translated_path = rendered_path
                print(f"[STATE] Restored rendered image path reference: {os.path.basename(rendered_path)}")
                
                # Store mapping
                if not hasattr(self, '_rendered_images_map'):
                    self._rendered_images_map = {}
                self._rendered_images_map[image_path] = rendered_path
        
        # Restore viewer rectangles (if no detection regions were restored)
        if 'viewer_rectangles' in state and not ('detection_regions' in state):
            viewer = self.image_preview_widget.viewer
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem
            
            # Check for recognized/translated texts to determine rectangle colors
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                shape = rect_data.get('shape', 'rect')
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])
                
                # Check if this rectangle has OCR/translation data
                has_recognized = (idx < len(recognized_texts) and 
                                 not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')) and
                                 (isinstance(recognized_texts[idx], str) and recognized_texts[idx].strip() or
                                  isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('text', '').strip()))
                has_translation = (idx < len(translated_texts) and 
                                  not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')) and
                                  isinstance(translated_texts[idx], dict) and translated_texts[idx].get('translation', '').strip())
                has_text = has_recognized or has_translation
                
                # Use blue for recognized/translated, green for detection-only
                if has_text:
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                pen.setCosmetic(True)
                
                if shape == 'ellipse':
                    item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    item = MoveableRectItem(rect, pen=pen, brush=brush)
                
                # Attach viewer reference so moved emits
                try:
                    item._viewer = viewer
                except Exception:
                    pass
                viewer._scene.addItem(item)
                viewer.rectangles.append(item)
                
                # Attach region index, is_recognized flag, and handlers
                try:
                    item.region_index = idx
                    item.is_recognized = has_text  # CRITICAL: Set is_recognized flag
                    _attach_move_sync_to_rectangle(self, item, idx)
                    # Add context menu to restored rectangles
                    _add_context_menu_to_rectangle(self, item, idx)
                except Exception:
                    pass
            
            # Rehydrate in-memory text state for this path as well
            try:
                ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
                print(f"[STATE] Rehydrated {ocr_count} OCR and {trans_count} translation entries (viewer_rectangles branch)")
            except Exception:
                pass
            
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes")
        
        print(f"[STATE] State restoration complete for {os.path.basename(image_path)}")
        
        # CRITICAL: Comprehensive graphics scene and overlay synchronization
        # This fixes the cosmetic issue where overlays appear disconnected until user interaction
        try:
            from PySide6.QtCore import QTimer
            viewer = self.image_preview_widget.viewer
            
            def comprehensive_refresh():
                try:
                    # 1. Force complete scene update
                    viewer._scene.update()
                    viewer.update()
                    viewer.repaint()
                    
                    # 2. Ensure text overlays are visible and properly positioned
                    if hasattr(self, 'show_text_overlays_for_image'):
                        self.show_text_overlays_for_image(image_path)
                    
                    # 3. Force overlay position synchronization with rectangles
                    try:
                        _synchronize_overlay_positions_with_rectangles(self, image_path)
                    except Exception:
                        pass
                    
                    # 4. Final scene update to reflect changes
                    viewer._scene.update()
                    viewer.viewport().update()
                    
                    print(f"[STATE] Comprehensive refresh completed for {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"[STATE] Comprehensive refresh failed: {e}")
            
            # Schedule multiple refreshes with increasing delays for maximum reliability
            QTimer.singleShot(10, comprehensive_refresh)
            QTimer.singleShot(100, comprehensive_refresh)
            QTimer.singleShot(250, comprehensive_refresh)
        except Exception:
            pass
        
    except Exception as e:
        print(f"[STATE] Failed to restore state: {e}")
        import traceback
        traceback.print_exc()

def _restore_image_state_overlays_only(self, image_path: str):
    """Restore ONLY rectangles/overlays for an image, without loading images
    
    This is used after manually loading the correct image to avoid double-loading.
    
    STATE ISOLATION: Validates that we're restoring the correct image's state.
    """
    try:
        if not hasattr(self, 'image_state_manager'):
            return
        
        # STATE ISOLATION: Verify we don't have stale state from another image
        if hasattr(self, '_current_state_image_path') and self._current_state_image_path:
            if self._current_state_image_path != image_path:
                print(f"[STATE_ISOLATION] WARNING: Stale state detected! Current={os.path.basename(self._current_state_image_path)}, Requested={os.path.basename(image_path)}")
                print(f"[STATE_ISOLATION] Clearing stale state before restoration")
                _clear_cross_image_state(self)
        
        # CRITICAL: Validate and clean stale state BEFORE restoration
        _validate_and_clean_stale_state(self, image_path)
        
        # Get saved state (after cleaning)
        state = self.image_state_manager.get_state(image_path)
        if not state:
            print(f"[STATE] No saved state for {os.path.basename(image_path)}")
            return
        
        print(f"[STATE] Restoring overlays for {os.path.basename(image_path)}")
        
        # Prefer viewer_rectangles (latest manual adjustments) to restore boxes; fallback to detection_regions
        used_boxes = False
        if 'viewer_rectangles' in state and state['viewer_rectangles']:
            viewer = self.image_preview_widget.viewer
            
            # STATE ISOLATION: Track rectangle count before and after clearing
            rect_count_before_clear = len(viewer.rectangles) if hasattr(viewer, 'rectangles') else 0
            print(f"[STATE_ISOLATION] Rectangle count BEFORE clear: {rect_count_before_clear}")
            
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            
            rect_count_after_clear = len(viewer.rectangles) if hasattr(viewer, 'rectangles') else 0
            print(f"[STATE_ISOLATION] Rectangle count AFTER clear: {rect_count_after_clear}")
            print(f"[STATE_ISOLATION] About to restore {len(state['viewer_rectangles'])} rectangles for {os.path.basename(image_path)}")
            
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem
            # Check if there are OCR or translated texts to determine rectangle colors
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            has_ocr_text = bool([t for t in recognized_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_translated_text = bool([t for t in translated_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_any_text = has_ocr_text or has_translated_text
            
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])
                
                # Use blue color if this rectangle has any recognized text (OCR or translated), green otherwise
                has_text_for_this_rect = False
                if has_any_text and idx < max(len(recognized_texts), len(translated_texts)):
                    # Check if this specific rectangle has text (either OCR or translated)
                    has_recognized = False
                    if idx < len(recognized_texts):
                        text_entry = recognized_texts[idx]
                        # Check if it's a valid text entry (not deleted, not empty)
                        if isinstance(text_entry, dict):
                            if not text_entry.get('deleted') and text_entry.get('text', '').strip():
                                has_recognized = True
                        elif isinstance(text_entry, str) and text_entry.strip():
                            has_recognized = True
                    
                    has_translation = False
                    if idx < len(translated_texts):
                        trans_entry = translated_texts[idx]
                        # Check if it's a valid translation entry (not deleted, not empty)
                        if isinstance(trans_entry, dict):
                            if not trans_entry.get('deleted') and trans_entry.get('translation', '').strip():
                                has_translation = True
                    
                    has_text_for_this_rect = has_recognized or has_translation
                    
                    if has_text_for_this_rect:
                        print(f"[STATE] Rectangle {idx} has text - will be BLUE (OCR={has_recognized}, Trans={has_translation})")
                    else:
                        print(f"[STATE] Rectangle {idx} has no valid text - will be GREEN")
                
                if has_text_for_this_rect:
                    # Blue for rectangles with recognized/translated text
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    # Green for detection-only rectangles
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                
                pen.setCosmetic(True)
                shape = rect_data.get('shape', 'rect')
                if shape == 'ellipse':
                    rect_item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    rect_item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    rect_item = MoveableRectItem(rect, pen=pen, brush=brush)
                # Attach viewer for move signal
                try:
                    rect_item._viewer = viewer
                except Exception:
                    pass
                viewer._scene.addItem(rect_item)
                viewer.rectangles.append(rect_item)
                
                # Add context menu to restored rectangles
                try:
                    rect_item.region_index = idx
                    # CRITICAL: Always set is_recognized based on text state, not just when true
                    rect_item.is_recognized = has_text_for_this_rect
                    _attach_move_sync_to_rectangle(self, rect_item, idx)
                    _add_context_menu_to_rectangle(self, rect_item, idx)
                except Exception:
                    pass
                    
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes (preferred)")
            # Populate in-memory text state and add OCR tooltips
            try:
                ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, image_path)
                if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                    _update_rectangles_with_recognition(self, self._recognized_texts)
                    print(f"[STATE] Attached OCR tooltips to {ocr_count} rectangles (viewer_rectangles branch)")
            except Exception:
                pass
            used_boxes = True
        
        if not used_boxes and 'detection_regions' in state:
            self._current_regions = state['detection_regions']
            print(f"[STATE] Restored {len(self._current_regions)} detection regions")
            # Redraw detection boxes on preview
            if hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                self.image_preview_widget.viewer.clear_rectangles()
            _draw_detection_boxes_on_preview(self, )
        
        # Restore overlay rectangles
        if 'overlay_rects' in state and state['overlay_rects']:
            if hasattr(self.image_preview_widget.viewer, 'overlay_rects'):
                self.image_preview_widget.viewer.overlay_rects = state['overlay_rects'].copy()
                print(f"[STATE] Restored {len(state['overlay_rects'])} overlay rects")
        
        # Restore cleaned image path reference (with validation + filesystem discovery)
        resolved_cleaned = _resolve_cleaned_image_for_render(self, image_path)
        if resolved_cleaned:
            self._cleaned_image_path = resolved_cleaned
            print(f"[STATE] Restored cleaned image path: {os.path.basename(self._cleaned_image_path)}")
        elif 'cleaned_image_path' in state:
            print(f"[STATE] Skipped corrupted cleaned_image_path: {os.path.basename(state['cleaned_image_path'])}")
        
        # Restore recognition_data from persisted recognized_texts so Edit OCR menu works after reload
        try:
            recognized_texts = state.get('recognized_texts') or []
            if recognized_texts:
                self._recognition_data = {}
                for i, result in enumerate(recognized_texts):
                    # Skip deleted entries
                    if isinstance(result, dict) and result.get('deleted'):
                        continue
                    # Handle both simple string format and complex dict format
                    # Match the format expected by context menu (dict with 'text' and 'bbox' keys)
                    if isinstance(result, str):
                        self._recognition_data[int(i)] = {'text': result, 'bbox': [0, 0, 100, 100]}
                    elif isinstance(result, dict) and 'text' in result:
                        idx = result.get('region_index', i)
                        self._recognition_data[int(idx)] = {
                            'text': result.get('text', ''),
                            'bbox': result.get('bbox', [0, 0, 100, 100])
                        }
                print(f"[STATE] Restored recognition_data for {len(self._recognition_data)} regions")
        except Exception as re:
            print(f"[STATE] Failed to restore recognition_data: {re}")
        
        # Restore translation_data from persisted translated_texts so Edit Translation menu works after reload
        try:
            translated_texts = state.get('translated_texts') or []
            if translated_texts:
                self._translation_data = {}
                for i, result in enumerate(translated_texts):
                    # Skip deleted entries
                    if isinstance(result, dict) and result.get('deleted'):
                        continue
                    idx = result.get('original', {}).get('region_index', i)
                    self._translation_data[int(idx)] = {
                        'original': result.get('original', {}).get('text', ''),
                        'translation': result.get('translation', '')
                    }
                print(f"[STATE] Restored translation_data for {len(self._translation_data)} regions")
        except Exception as te:
            print(f"[STATE] Failed to restore translation_data: {te}")
        
        # Reattach context menus for rectangles (after both recognition and translation data are restored)
        try:
            rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
            print(f"[STATE] Debug - Available recognition_data keys: {list(getattr(self, '_recognition_data', {}).keys())}")
            print(f"[STATE] Debug - Available translation_data keys: {list(getattr(self, '_translation_data', {}).keys())}")
            for idx, rect_item in enumerate(rects):
                try:
                    # Make sure region_index is set correctly on the rectangle
                    if not hasattr(rect_item, 'region_index'):
                        rect_item.region_index = idx
                    print(f"[STATE] Debug - Rectangle {idx} has region_index: {getattr(rect_item, 'region_index', 'None')}")
                    _add_context_menu_to_rectangle(self, rect_item, rect_item.region_index)
                except Exception as e:
                    print(f"[STATE] Error attaching context menu to rect {idx}: {e}")
            print(f"[STATE] Reattached context menus to {len(rects)} rectangles")
        except Exception as cm:
            print(f"[STATE] Failed to reattach context menus: {cm}")
        
        # Restore viewer rectangles (if no detection regions were restored)
        if 'viewer_rectangles' in state and not ('detection_regions' in state):
            viewer = self.image_preview_widget.viewer
            if hasattr(viewer, 'clear_rectangles'):
                viewer.clear_rectangles()
            
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath
            from manga_image_preview import MoveableRectItem, MoveableEllipseItem, MoveablePathItem
            
            # Check if there are OCR or translated texts to determine rectangle colors
            recognized_texts = state.get('recognized_texts', [])
            translated_texts = state.get('translated_texts', [])
            has_ocr_text = bool([t for t in recognized_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_translated_text = bool([t for t in translated_texts if not (isinstance(t, dict) and t.get('deleted'))])
            has_any_text = has_ocr_text or has_translated_text
            
            for idx, rect_data in enumerate(state['viewer_rectangles']):
                rect = QRectF(rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height'])
                
                # Use blue color if this rectangle has any recognized text (OCR or translated), green otherwise
                has_text_for_this_rect = False
                if has_any_text and idx < max(len(recognized_texts), len(translated_texts)):
                    # Check if this specific rectangle has text (either OCR or translated)
                    has_recognized = (idx < len(recognized_texts) and 
                                    not (isinstance(recognized_texts[idx], dict) and recognized_texts[idx].get('deleted')))
                    has_translation = (idx < len(translated_texts) and 
                                     not (isinstance(translated_texts[idx], dict) and translated_texts[idx].get('deleted')))
                    has_text_for_this_rect = has_recognized or has_translation
                
                if has_text_for_this_rect:
                    # Blue for rectangles with recognized/translated text
                    pen = QPen(QColor(0, 150, 255), 2)
                    brush = QBrush(QColor(0, 150, 255, 50))
                else:
                    # Green for detection-only rectangles
                    pen = QPen(QColor(0, 255, 0), 1)
                    brush = QBrush(QColor(0, 255, 0, 50))
                
                pen.setCosmetic(True)
                shape = rect_data.get('shape', 'rect')
                if shape == 'ellipse':
                    rect_item = MoveableEllipseItem(rect, pen=pen, brush=brush)
                elif shape == 'polygon' and rect_data.get('polygon'):
                    path = QPainterPath()
                    pts = rect_data.get('polygon') or []
                    if pts:
                        path.moveTo(pts[0][0], pts[0][1])
                        for px, py in pts[1:]:
                            path.lineTo(px, py)
                        path.closeSubpath()
                    rect_item = MoveablePathItem(path, pen=pen, brush=brush)
                else:
                    rect_item = MoveableRectItem(rect, pen=pen, brush=brush)
                # Attach viewer reference so moved emits
                try:
                    rect_item._viewer = viewer
                except Exception:
                    pass
                viewer._scene.addItem(rect_item)
                viewer.rectangles.append(rect_item)
                # Attach region index and move-sync handler for ALL rectangles
                try:
                    rect_item.region_index = idx
                    # CRITICAL: Always set is_recognized based on text state, not just when true
                    rect_item.is_recognized = has_text_for_this_rect
                    _attach_move_sync_to_rectangle(self, rect_item, idx)
                    # CRITICAL: Add context menu to ALL rectangles (both blue and green)
                    _add_context_menu_to_rectangle(self, rect_item, idx)
                except Exception:
                    pass
            
            print(f"[STATE] Restored {len(state['viewer_rectangles'])} viewer shapes")
        
        # If translated_texts exist and rectangles are present, restore text overlays on source viewer
        try:
            translated_texts = state.get('translated_texts') or []
            rects_exist = bool(getattr(self.image_preview_widget.viewer, 'rectangles', []))
            if translated_texts and rects_exist and hasattr(self, '_add_text_overlay_to_viewer'):
                # Filter out deleted text overlays
                active_translated_texts = []
                for i, result in enumerate(translated_texts):
                    if not (isinstance(result, dict) and result.get('deleted')):
                        active_translated_texts.append(result)
                
                if active_translated_texts:
                    _add_text_overlay_to_viewer(self, active_translated_texts)
                    print(f"[STATE] Restored {len(active_translated_texts)} text overlays from persisted state (skipped {len(translated_texts) - len(active_translated_texts)} deleted)")
        except Exception as e2:
            print(f"[STATE] Failed to restore text overlays: {e2}")
        
        # If translated_texts exist and rectangles are present, restore text overlays on source viewer
        try:
            translated_texts = state.get('translated_texts') or []
            rects_exist = bool(getattr(self.image_preview_widget.viewer, 'rectangles', []))
            if translated_texts and rects_exist and hasattr(self, '_add_text_overlay_to_viewer'):
                # Filter out deleted text overlays
                active_translated_texts = []
                for i, result in enumerate(translated_texts):
                    if not (isinstance(result, dict) and result.get('deleted')):
                        active_translated_texts.append(result)
                
                if active_translated_texts:
                    # Ensure _translation_data is populated for context menu (only for active overlays)
                    if not hasattr(self, '_translation_data') or not self._translation_data:
                        self._translation_data = {}
                        for i, result in enumerate(active_translated_texts):
                            idx = result.get('original', {}).get('region_index', i)
                            self._translation_data[int(idx)] = {
                                'original': result.get('original', {}).get('text', ''),
                                'translation': result.get('translation', '')
                            }
                    _add_text_overlay_to_viewer(self, active_translated_texts)
                    # Reattach context menus
                    rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                    for idx, rect_item in enumerate(rects):
                        try:
                            _add_context_menu_to_rectangle(self, rect_item, idx)
                        except Exception:
                            pass
                    print(f"[STATE] Restored {len(active_translated_texts)} text overlays from persisted state (skipped {len(translated_texts) - len(active_translated_texts)} deleted)")
        except Exception as e2:
            print(f"[STATE] Failed to restore text overlays: {e2}")
        
        print(f"[STATE] Overlay restoration complete for {os.path.basename(image_path)}")
        
        # CRITICAL: Comprehensive graphics scene and overlay synchronization for overlays-only mode
        # This fixes the cosmetic issue where overlays appear disconnected until user interaction
        try:
            from PySide6.QtCore import QTimer
            viewer = self.image_preview_widget.viewer
            
            def comprehensive_overlay_refresh():
                try:
                    # 1. Force complete scene update
                    viewer._scene.update()
                    viewer.update()
                    viewer.repaint()
                    
                    # 2. Ensure text overlays are visible and properly positioned
                    if hasattr(self, 'show_text_overlays_for_image'):
                        self.show_text_overlays_for_image(image_path)
                    
                    # 3. Force overlay position synchronization with rectangles
                    try:
                        _synchronize_overlay_positions_with_rectangles(self, image_path)
                    except Exception:
                        pass
                    
                    # 4. Re-attach move sync handlers to ensure interactivity
                    try:
                        rectangles = getattr(viewer, 'rectangles', []) or []
                        for idx, rect_item in enumerate(rectangles):
                            if hasattr(rect_item, 'region_index'):
                                _attach_move_sync_to_rectangle(self, rect_item, rect_item.region_index)
                            else:
                                _attach_move_sync_to_rectangle(self, rect_item, idx)
                    except Exception:
                        pass
                    
                    # 5. Final comprehensive scene update
                    viewer._scene.update()
                    viewer.viewport().update()
                    viewer.repaint()
                    
                    print(f"[STATE] Comprehensive overlay refresh completed for {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"[STATE] Comprehensive overlay refresh failed: {e}")
            
            # Schedule multiple refreshes with increasing delays for maximum reliability
            QTimer.singleShot(15, comprehensive_overlay_refresh)
            QTimer.singleShot(100, comprehensive_overlay_refresh)
            QTimer.singleShot(300, comprehensive_overlay_refresh)
        except Exception:
            pass
        
    except Exception as e:
        print(f"[STATE] Failed to restore overlays: {e}")
        import traceback
        traceback.print_exc()

def _process_detect_results(self, results: dict):
    """Process detection results on main thread and update preview (image-aware).
    
    STATE ISOLATION: Only draws rectangles if the detection results are for the currently displayed image.
    """
    # ===== CANCELLATION CHECK: Discard results if stop was clicked =====
    if _is_translation_cancelled(self):
        print(f"[DETECT_RESULTS] Discarding results - stop was clicked")
        return
    
    try:
        image_path = results['image_path']
        regions = results['regions']
        preserve_rectangles = results.get('preserve_rectangles', False)
        
        # Persist regions for this image ONLY if no OCR data exists yet
        if hasattr(self, 'image_state_manager'):
            try:
                current_state = self.image_state_manager.get_state(image_path)
                # Don't overwrite OCR data with just detection regions
                if not current_state.get('recognized_texts') and not current_state.get('translated_texts'):
                    self.image_state_manager.update_state(image_path, {'detection_regions': regions})
                else:
                    print(f"[STATE] Skipping detection_regions save - OCR/translation data exists")
            except Exception:
                pass
        
        # STATE ISOLATION: Only draw if this image is currently displayed in the source viewer
        # NOTE: We no longer suppress drawing during batch mode - users want to see rectangles
        # being drawn during translation for visual feedback
        
        # Check if detection is for current image (normalize paths for comparison)
        current_img = getattr(self.image_preview_widget, 'current_image_path', None) if hasattr(self, 'image_preview_widget') else None
        if not current_img:
            print(f"[DETECT_RESULTS] No current image in preview - skipping draw")
            return
        
        # Normalize paths for comparison (resolve to absolute paths)
        import os
        try:
            image_path_abs = os.path.abspath(image_path)
            current_img_abs = os.path.abspath(current_img)
            
            # Additional check: Verify results match the image detection was started for
            # This catches the race condition where user switches images during detection
            if hasattr(self, '_detection_started_for_image') and self._detection_started_for_image:
                detection_started_abs = self._detection_started_for_image
                if image_path_abs != detection_started_abs:
                    print(f"[STATE_ISOLATION] Results don't match detection start image")
                    print(f"[STATE_ISOLATION] Started for: {os.path.basename(detection_started_abs)}")
                    print(f"[STATE_ISOLATION] Results for: {os.path.basename(image_path)}")
                    print(f"[STATE_ISOLATION] Skipping rectangle draw")
                    return
            
            # Check if results are for currently displayed image
            if image_path_abs != current_img_abs:
                print(f"[STATE_ISOLATION] Detection for different image - Detected: {os.path.basename(image_path)}, Current: {os.path.basename(current_img)}")
                print(f"[STATE_ISOLATION] Skipping rectangle draw to prevent cross-contamination")
                return
        except Exception as e:
            print(f"[STATE_ISOLATION] Path validation error: {e}")
            # Fallback to simple comparison
            if image_path != current_img:
                print(f"[DETECT_RESULTS] Skipping draw; not current image: {os.path.basename(image_path)}")
                return
        
        # Update working state and draw
        self._current_regions = regions
        self._original_image_path = image_path
        
        # STATE ISOLATION: Clear detection tracking since results were successfully applied
        if hasattr(self, '_detection_started_for_image'):
            self._detection_started_for_image = None
            print(f"[STATE_ISOLATION] Cleared detection tracking after successful draw")
        
        # Only clear rectangles if not preserving them (e.g., during clean operations)
        if not preserve_rectangles and hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
            self.image_preview_widget.viewer.clear_rectangles()
            print(f"[DETECT_RESULTS] Cleared existing rectangles before drawing detection results")
        elif preserve_rectangles:
            rectangle_count = len(getattr(self.image_preview_widget.viewer, 'rectangles', []))
            print(f"[DETECT_RESULTS] Preserving {rectangle_count} existing rectangles during detection update")
        
        _draw_detection_boxes_on_preview(self, )
        
        # PERSIST: Save viewer_rectangles to state so they survive panel/session switches
        try:
            _persist_current_image_state(self)
        except Exception:
            pass
        
    except Exception as e:
        self._log(f"❌ Failed to process detection results: {str(e)}", "error")

def _draw_detection_boxes_on_preview(self):
    """Draw detection boxes on the preview widget using region data"""
    try:
        if not hasattr(self, '_current_regions') or not self._current_regions:
            return
        
        viewer = self.image_preview_widget.viewer
        
        # If we already have rectangles, don't redraw (preserve existing rectangles during clean operations)
        if hasattr(viewer, 'rectangles') and viewer.rectangles and len(viewer.rectangles) > 0:
            print(f"[DRAW_BOXES] Skipping rectangle drawing - {len(viewer.rectangles)} rectangles already exist")
            return
        
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QPen, QBrush, QColor
        from manga_image_preview import MoveableRectItem, MoveableEllipseItem
        
        print(f"[DRAW_BOXES] Drawing {len(self._current_regions)} regions")
        print(f"[DRAW_BOXES] Current rectangles count before drawing: {len(viewer.rectangles)}")
        
        # Draw boxes for each region
        for i, region in enumerate(self._current_regions):
            bbox = region.get('bbox', [])
            if len(bbox) >= 4:
                x, y, width, height = bbox
                rect = QRectF(x, y, width, height)
                print(f"[DRAW_BOXES] Drawing shape {i}: x={x}, y={y}, w={width}, h={height}")
                
                # Create pen and brush with detection colors
                pen = QPen(QColor(0, 255, 0), 1)  # Green border (width 1 to avoid double-line artifact)
                pen.setCosmetic(True)  # Pen width stays constant regardless of zoom
                pen.setCapStyle(Qt.PenCapStyle.SquareCap)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                brush = QBrush(QColor(0, 255, 0, 50))  # Semi-transparent green fill
                
                # Create shape item (ellipse or rectangle)
                item = MoveableEllipseItem(rect, pen=pen, brush=brush) if getattr(self, '_use_circle_shapes', False) else MoveableRectItem(rect, pen=pen, brush=brush)
                
                # Explicitly disable antialiasing on this item to prevent blur artifacts
                from PySide6.QtWidgets import QGraphicsItem
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, False)
                
                # Attach viewer reference so item can emit moved signal
                try:
                    item._viewer = viewer
                except Exception:
                    pass
                
                viewer._scene.addItem(item)
                viewer.rectangles.append(item)
                
                # Track region index on the item and attach move-sync handler
                try:
                    item.region_index = i
                    _attach_move_sync_to_rectangle(self, item, i)
                    # Add context menu to green detection rectangles
                    _add_context_menu_to_rectangle(self, item, i)
                except Exception:
                    pass
        
        print(f"[DRAW_BOXES] Final rectangles count after drawing: {len(viewer.rectangles)}")
    
    except Exception as e:
        self._log(f"⚠️ Error drawing detection boxes: {str(e)}", "warning")

def _on_clean_image_clicked(self):
    """Clean button: ensure regions exist (auto-detect if needed) then run inpainting in background."""
    try:
        # Get current image path
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            self._log("⚠️ No image loaded for cleaning", "warning")
            return

        # Disable the clean button to prevent multiple clicks
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'clean_btn'):
            self.image_preview_widget.clean_btn.setEnabled(False)
            self.image_preview_widget.clean_btn.setText("Cleaning...")

        # Determine base image path
        image_path = self._original_image_path if hasattr(self, '_original_image_path') and self._original_image_path else self.image_preview_widget.current_image_path
        
        # Track which image we're cleaning for overlay removal
        self._original_image_path = image_path
        
        # Add processing overlay effect (after tracking image)
        _add_processing_overlay(self, )

        # Prepare regions: use existing rectangles if any; otherwise run detection synchronously
        has_rectangles = (hasattr(self.image_preview_widget, 'viewer') and
                          self.image_preview_widget.viewer.rectangles and
                          len(self.image_preview_widget.viewer.rectangles) > 0)

        regions = None
        if has_rectangles:
            regions = _extract_regions_from_preview(self, )
        else:
            # Auto-run detection (equivalent to clicking Detect Text first)
            self._log("🔍 No regions found — running automatic detection before cleaning...", "info")
            detection_config = _get_detection_config(self, ) or {}
            # Exclude empty container bubbles to avoid cleaning non-text areas
            if detection_config.get('detect_empty_bubbles', True):
                detection_config['detect_empty_bubbles'] = False
            regions = _run_detection_sync(self, image_path, detection_config)
            if not regions or len(regions) == 0:
                self._log("⚠️ No text regions detected to clean", "warning")
                _restore_clean_button(self, )
                return
            # Draw detected boxes on preview for user feedback (preserve any existing rectangles during clean operation)
            try:
                self.update_queue.put(('detect_results', {
                    'image_path': image_path,
                    'regions': regions,
                    'preserve_rectangles': True  # Don't clear existing rectangles during clean operation
                }))
                # Persist detection state ONLY if no OCR data exists yet
                if hasattr(self, 'image_state_manager'):
                    current_state = self.image_state_manager.get_state(image_path)
                    if not current_state.get('recognized_texts') and not current_state.get('translated_texts'):
                        self.image_state_manager.update_state(image_path, {'detection_regions': regions})
                    else:
                        print(f"[STATE] Skipping detection_regions save - OCR/translation data exists")
            except Exception:
                pass

        self._log(f"🧽 Starting background cleaning: {os.path.basename(image_path)}", "info")

        # Run inpainting in background thread
        import threading
        thread = threading.Thread(target=_run_clean_background, args=(self, image_path, regions),
                                  daemon=True)
        thread.start()

    except Exception as e:
        import traceback
        self._log(f"❌ Clean setup failed: {str(e)}", "error")
        print(f"Clean setup error traceback: {traceback.format_exc()}")
        _restore_clean_button(self, )

def _extract_regions_from_preview(self) -> list:
    """Extract regions from currently displayed rectangles in the preview widget,
    preserving rectangle indices to allow exclusion filtering.
    """
    regions = []
    try:
        if hasattr(self.image_preview_widget, 'viewer') and self.image_preview_widget.viewer.rectangles:
            # Build region dicts directly from shapes without merging to preserve indices
            for i, rect_item in enumerate(self.image_preview_widget.viewer.rectangles):
                br = rect_item.sceneBoundingRect()
                x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
                shape = getattr(rect_item, 'shape_type', 'rect')
                region_dict = {
                    'bbox': [x, y, w, h],
                    'coords': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    'confidence': 1.0,
                    'rect_index': i,
                    'shape': shape
                }
                # If polygon, capture points in scene coordinates
                try:
                    if shape == 'polygon' and hasattr(rect_item, 'path'):
                        poly = rect_item.mapToScene(rect_item.path().toFillPolygon())
                        pts = []
                        for p in poly:
                            pts.append([int(p.x()), int(p.y())])
                        if len(pts) >= 3:
                            region_dict['polygon'] = pts
                except Exception:
                    pass
                regions.append(region_dict)
            
            self._log(f"🎯 Extracted {len(regions)} regions from preview shapes (preserving indices)", "info")
        else:
            self._log("⚠️ No rectangles found in preview widget", "warning")
    except Exception as e:
        self._log(f"❌ Error extracting regions from preview: {str(e)}", "error")
        import traceback
        print(f"Extract regions error: {traceback.format_exc()}")
    
    return regions
def _run_clean_background(self, image_path: str, regions: list):
    """Run the actual cleaning process in background thread with explicit memory cleanup"""
    image = None  # Initialize for cleanup in finally block
    mask = None
    inpainter = None  # Track for pool return
    temp_translator = None  # Track temporary translator for pool cleanup
    
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    _reset_cancellation_flags(self)
    
    try:
        # ===== CANCELLATION CHECK: At start of cleaning =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Cleaning cancelled before starting", "warning")
            print(f"[CLEAN] Cancelled at start")
            self.update_queue.put(('clean_button_restore', None))
            return
        
        import cv2
        import numpy as np
        from local_inpainter import LocalInpainter
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self._log(f"❌ Failed to load image: {os.path.basename(image_path)}", "error")
            self.update_queue.put(('clean_button_restore', None))
            return
        
        # Get exclusion list directly from rectangle objects (session-only)
        excluded_regions = []
        try:
            if hasattr(self.image_preview_widget, 'viewer') and self.image_preview_widget.viewer.rectangles:
                rectangles = self.image_preview_widget.viewer.rectangles
                for i, rect_item in enumerate(rectangles):
                    if getattr(rect_item, 'exclude_from_clean', False):
                        excluded_regions.append(i)
                
                print(f"[CLEAN_DEBUG] Found exclusions from rectangles: {excluded_regions}")
                
                if excluded_regions:
                    self._log(f"🚫 Found {len(excluded_regions)} excluded regions: {excluded_regions}", "info")
                else:
                    self._log(f"✅ No regions excluded from cleaning", "info")
            else:
                print(f"[CLEAN_DEBUG] No rectangles available to check exclusions")
        except Exception as e:
            print(f"[CLEAN_DEBUG] Error getting exclusion list from rectangles: {e}")
            import traceback
            print(f"[CLEAN_DEBUG] Traceback: {traceback.format_exc()}")
            self._log(f"⚠️ Error getting exclusion list: {e}", "warning")
        
        # Filter regions based on exclusion status
        filtered_regions = []
        excluded_count = 0
        
        print(f"[CLEAN_DEBUG] Processing {len(regions)} regions for exclusion filtering")
        for i, region in enumerate(regions):
            # Check if this region should be excluded (using rect_index if available)
            rect_index = region.get('rect_index', None)
            print(f"[CLEAN_DEBUG] Region {i}: rect_index={rect_index}, excluded_regions={excluded_regions}")
            
            if rect_index is not None and rect_index in excluded_regions:
                excluded_count += 1
                print(f"[CLEAN_DEBUG] EXCLUDING region {i} (rect_index={rect_index})")
                self._log(f"🚫 Skipping region {rect_index} (excluded from clean)", "info")
                continue
            else:
                print(f"[CLEAN_DEBUG] INCLUDING region {i} (rect_index={rect_index})")
            
            filtered_regions.append(region)
        
        self._log(f"🎨 Creating mask from {len(filtered_regions)} regions ({excluded_count} excluded)", "info")
        
        # ===== CANCELLATION CHECK: Before creating mask =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Cleaning cancelled before mask creation", "warning")
            print(f"[CLEAN] Cancelled before mask creation")
            self.update_queue.put(('clean_button_restore', None))
            return
        
        # Create mask from filtered regions
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for region in filtered_regions:
            # Handle both dictionary format (from detect) and object format (from translator)
            if isinstance(region, dict):
                # Dictionary format from detect button
                bbox = region.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, width, height = bbox
                    x1, y1, x2, y2 = x, y, x + width, y + height
                else:
                    continue
            else:
                # Object format from translator
                x1, y1, x2, y2 = int(region.x1), int(region.y1), int(region.x2), int(region.y2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, image.shape[1]))
            y2 = max(y1 + 1, min(y2, image.shape[0]))
            
            # Determine shape for mask
            shape = None
            try:
                if isinstance(region, dict):
                    shape = region.get('shape')
            except Exception:
                shape = None
            use_ellipse = bool(shape == 'ellipse' or getattr(self, '_use_circle_shapes', False))
            
            if shape == 'polygon' and isinstance(region.get('polygon'), list) and len(region.get('polygon')) >= 3:
                import numpy as _np
                pts = _np.array(region['polygon'], dtype=_np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            elif use_ellipse:
                # Draw filled ellipse that fits the bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                rx = max(1, int((x2 - x1) / 2))
                ry = max(1, int((y2 - y1) / 2))
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            else:
                # Draw filled rectangle on mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Get inpainting settings from manga integration config
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        
        if inpaint_method == 'local':
            # Use local inpainter with the same method as manga_translator
            self._log(f"🖼️ Using local inpainter: {local_model}", "info")
            
            # Get model path from config (same way as manga_translator)
            model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
            try:
                if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                    model_path = ''
            except Exception:
                pass
            
            # Ensure we have a model path (download if needed)
            resolved_model_path = model_path
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                try:
                    from local_inpainter import LocalInpainter
                    self._log(f"📥 Downloading {local_model} model...", "info")
                    temp_inp = LocalInpainter()
                    resolved_model_path = temp_inp.download_jit_model(local_model)
                except Exception as e:
                    self._log(f"⚠️ Model download failed: {e}", "warning")
                    resolved_model_path = None
            
            # Use shared inpainter from pool - track the temporary translator for cleanup
            if resolved_model_path and os.path.exists(resolved_model_path):
                self._log(f"🎨 Using shared inpainter from pool: {os.path.basename(resolved_model_path)}", "info")
                # Create translator for pool access and track it for cleanup
                try:
                    from manga_translator import MangaTranslator
                    from unified_api_client import UnifiedClient
                    import time
                    
                    ocr_config = _get_ocr_config(self, ) if hasattr(self, 'main_gui') else {}
                    api_key = self.main_gui.config.get('api_key', '') or 'dummy'
                    model = self.main_gui.config.get('model', 'gpt-4o-mini')
                    uc = UnifiedClient(model=model, api_key=api_key)
                    temp_translator = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=lambda m, l: None, skip_inpainter_init=True)
                    
                    # POLL for inpainter with timeout (same as translator initialization)
                    inpainter = None
                    poll_timeout = 30  # 30 seconds
                    poll_interval = 0.5  # Check every 500ms
                    start_time = time.time()
                    
                    while time.time() - start_time < poll_timeout:
                        inpainter = temp_translator._get_or_init_shared_local_inpainter(local_model, resolved_model_path, force_reload=False)
                        if inpainter:
                            break
                        
                        # No inpainter yet - wait and retry
                        elapsed = time.time() - start_time
                        if elapsed >= 2 and int(elapsed) % 5 == 0:  # Log every 5 seconds after first 2s
                            self._log(f"⏳ Waiting for inpainter pool... ({int(elapsed)}s)", "info")
                        time.sleep(poll_interval)
                    
                    if inpainter:
                        # Immediately update GUI pool tracker after checkout
                        self.update_queue.put(('update_pool_tracker', None))
                        
                        # ===== CANCELLATION CHECK: After getting inpainter =====
                        if _is_translation_cancelled(self):
                            self._log(f"⏹ Cleaning cancelled after getting inpainter", "warning")
                            print(f"[CLEAN] Cancelled after getting inpainter")
                            self.update_queue.put(('clean_button_restore', None))
                            return
                    else:
                        self._log(f"⚠️ No inpainter available after {poll_timeout}s timeout", "warning")
                        
                except Exception as e:
                    print(f"[CLEAN] Failed to create translator/inpainter: {e}")
                    inpainter = None
                    temp_translator = None
                
                if not inpainter:
                    self._log(f"❌ Failed to get shared inpainter", "error")
                    self.update_queue.put(('clean_button_restore', None))
                    return
            else:
                self._log(f"❌ No valid model path for {local_model}", "error")
                self.update_queue.put(('clean_button_restore', None))
                return
            # Get custom iteration values from rectangles
            custom_iterations = _get_custom_iterations_for_regions(self, filtered_regions)
            
            # ===== CANCELLATION CHECK: Before running inpainting =====
            if _is_translation_cancelled(self):
                self._log(f"⏹ Cleaning cancelled before inpainting", "warning")
                print(f"[CLEAN] Cancelled before inpainting")
                self.update_queue.put(('clean_button_restore', None))
                return
            
            if custom_iterations:
                iterations_str = ', '.join([f"{region}:{iters}" for region, iters in custom_iterations.items()])
                self._log(f"🧽 Running local inpainting with custom iterations: {iterations_str}", "info")
                # For now, use the first custom iteration value found
                # TODO: Implement per-region inpainting with different iterations
                first_iteration_value = next(iter(custom_iterations.values()))
                cleaned_image = inpainter.inpaint(image, mask, iterations=first_iteration_value)
            else:
                self._log("🧽 Running local inpainting with auto iterations", "info")
                cleaned_image = inpainter.inpaint(image, mask)
            
        else:
            # For cloud/hybrid methods, would need more complex setup
            # For now, fallback to basic OpenCV inpainting
            self._log("🧽 Using OpenCV inpainting (fallback)", "info")
            cleaned_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        if cleaned_image is not None:
            # Save cleaned image into per-image isolated folder and show it on Output tab
            parent_dir = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            
            # Check for OUTPUT_DIRECTORY override (prefer config over env var)
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            if override_dir:
                output_dir = os.path.join(override_dir, f"{base}_translated")
            else:
                output_dir = os.path.join(parent_dir, f"{base}_translated")
            
            os.makedirs(output_dir, exist_ok=True)
            cleaned_path = os.path.join(output_dir, f"{base}_cleaned{ext}")

            cv2.imwrite(cleaned_path, cleaned_image)
            self._log(f"💾 Saved cleaned image: {os.path.relpath(cleaned_path, parent_dir)}", "info")

            # Persist cleaned path to state
            try:
                if hasattr(self, 'image_state_manager'):
                    self.image_state_manager.update_state(image_path, {'cleaned_image_path': cleaned_path})
            except Exception:
                pass

            # Update Output viewer with cleaned image (no tab switch)
            # Update Output viewer with cleaned image (include source for image-aware gating)
            self.update_queue.put(('preview_update', {
                'translated_path': cleaned_path,
                'source_path': image_path,
                'switch_to_output': False  # Don't auto-switch tabs, let user manually switch
            }))
            self._log(f"✅ Cleaning complete!", "success")
        else:
            self._log("❌ Inpainting failed", "error")
            
    except Exception as e:
        import traceback
        self._log(f"❌ Background cleaning failed: {str(e)}", "error")
        print(f"Background clean error traceback: {traceback.format_exc()}")
    finally:
        # Return inpainter to pool if checked out via temporary translator
        try:
            if temp_translator is not None:
                temp_translator._return_inpainter_to_pool()
                # Immediately update GUI pool tracker after return
                self.update_queue.put(('update_pool_tracker', None))
        except Exception as e:
            print(f"[CLEAN] Failed to return inpainter to pool: {e}")
        
        # MEMORY CLEANUP: Explicitly delete numpy arrays to free RAM
        try:
            if 'image' in locals() and image is not None:
                del image
            if 'mask' in locals() and mask is not None:
                del mask
            if 'cleaned_image' in locals() and 'cleaned_image' in dir():
                try:
                    del cleaned_image
                except:
                    pass
            # Force garbage collection to release memory immediately
            import gc
            gc.collect()
        except Exception:
            pass
        
        # Always restore the button using thread-safe update queue
        self.update_queue.put(('clean_button_restore', None))

def _update_preview_after_clean(self, output_path: str):
    """Update preview on main thread after cleaning is complete"""
    try:
        # Store cleaned image path so Translate button can use it
        self._cleaned_image_path = output_path
        print(f"[CLEAN] Stored cleaned image path: {output_path}")
        
        # Before switching image, alias overlays from original path to cleaned path
        if hasattr(self, '_original_image_path') and self._original_image_path:
            _alias_text_overlays_for_image(self, self._original_image_path, output_path)
        
        # Load cleaned image while preserving rectangles and text overlays for workflow continuity
        self.image_preview_widget.load_image(output_path, preserve_rectangles=True, preserve_text_overlays=True)
    except Exception as e:
        self._log(f"❌ Failed to update preview: {str(e)}", "error")

def _restore_clean_button(self):
    """Restore the clean button to its original state and switch display mode to cleaned"""
    try:
        # Remove processing overlay effect for the image that was being processed
        # For clean button, use _original_image_path if available, otherwise current image
        image_path = getattr(self, '_original_image_path', None)
        if not image_path and hasattr(self, 'image_preview_widget'):
            image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        _remove_processing_overlay(self, image_path)
        
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'clean_btn'):
            self.image_preview_widget.clean_btn.setEnabled(True)
            self.image_preview_widget.clean_btn.setText("Clean")
            
            # Switch display mode to 'cleaned' so user sees the result
            try:
                ipw = self.image_preview_widget
                ipw.source_display_mode = 'cleaned'
                ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
                
                # Update the toggle button appearance to match 'cleaned' state
                if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                    ipw.cleaned_toggle_btn.setText("🧽")  # Sponge for cleaned
                    ipw.cleaned_toggle_btn.setToolTip("Showing cleaned images (click to cycle)")
                    ipw.cleaned_toggle_btn.setStyleSheet("""
                        QToolButton {
                            background-color: #4a7ba7;
                            border: 2px solid #5a9fd4;
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
                            background-color: #5a9fd4;
                        }
                    """)
                
                # Reload the image to show the cleaned version
                if image_path:
                    ipw.load_image(image_path, preserve_rectangles=True, preserve_text_overlays=True)
                print(f"[CLEAN_RESTORE] Switched display mode to 'cleaned'")
            except Exception as mode_err:
                print(f"[CLEAN_RESTORE] Failed to switch display mode: {mode_err}")
    except Exception:
        pass

def _get_detection_config(self) -> dict:
    """Get detection configuration from settings"""
    manga_settings = self.main_gui.config.get('manga_settings', {})
    ocr_settings = manga_settings.get('ocr', {})
    model_path = ocr_settings.get('bubble_model_path', '')
    model_url = ocr_settings.get('rtdetr_model_url', 'ogkalu/comic-text-and-bubble-detector')
    # Sanitize JSON paths
    try:
        if isinstance(model_path, str) and model_path.lower().endswith('.json'):
            model_path = ''
        if isinstance(model_url, str) and model_url.lower().endswith('.json'):
            model_url = 'ogkalu/comic-text-and-bubble-detector'
    except Exception:
        pass
    detection_config = {
        'detector_type': ocr_settings.get('detector_type', 'rtdetr_onnx'),
        'model_path': model_path,
        'model_url': model_url,
        'confidence': ocr_settings.get('bubble_confidence', 0.3),
        'detect_free_text': ocr_settings.get('detect_free_text', True),  # Free text checkbox setting
        'detect_empty_bubbles': ocr_settings.get('detect_empty_bubbles', True),
        'detect_text_bubbles': ocr_settings.get('detect_text_bubbles', True)
    }
    return detection_config

def _get_inpaint_config(self) -> dict:
    """Get inpainting configuration from settings"""
    inpaint_config = {
        'method': self.inpaint_method_value if hasattr(self, 'inpaint_method_value') else 'none',
        'model_type': self.local_model_type_value if hasattr(self, 'local_model_type_value') else 'lama',
        'model_path': self.local_model_path_value if hasattr(self, 'local_model_path_value') else '',
        'quality': self.inpaint_quality_value if hasattr(self, 'inpaint_quality_value') else 'high',
        'dilation': self.inpaint_dilation_value if hasattr(self, 'inpaint_dilation_value') else 0,
        'passes': self.inpaint_passes_value if hasattr(self, 'inpaint_passes_value') else 2
    }
    return inpaint_config

def _extract_regions_for_background(self):
    """Helper to extract regions from preview and store for background thread"""
    try:
        regions = _extract_regions_from_preview(self, )
        self._temp_regions_extracted = regions
        print(f"[EXTRACT] Extracted {len(regions)} regions for background thread")
    except Exception as e:
        print(f"[EXTRACT] Error extracting regions: {e}")
        self._temp_regions_extracted = []

def _run_detection_sync(self, image_path: str, detection_config: dict) -> list:
    """Run detection synchronously (for Recognize button) and return regions
    
    Args:
        image_path: Path to the image
        detection_config: Detection configuration dict
        
    Returns:
        list: List of region dictionaries, or empty list if detection failed
    """
    detector = None  # Initialize for cleanup
    temp_translator = None  # Track temporary translator for pool cleanup
    try:
        # ===== CANCELLATION CHECK: At start of sync detection =====
        if _is_translation_cancelled(self):
            print(f"[DETECT_SYNC] Cancelled at start")
            return []
        
        import cv2
        from bubble_detector import BubbleDetector
        from manga_translator import MangaTranslator
        from unified_api_client import UnifiedClient
        
        # Use pool-aware detector checkout
        try:
            ocr_config = _get_ocr_config(self, ) if hasattr(self, 'main_gui') else {}
            api_key = self.main_gui.config.get('api_key', '') or 'dummy'
            model = self.main_gui.config.get('model', 'gpt-4o-mini')
            uc = UnifiedClient(model=model, api_key=api_key)
            temp_translator = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=lambda m, l: None, skip_inpainter_init=True)
            detector = temp_translator._get_thread_bubble_detector()
            # Check if detector is None (pool checkout can return None)
            if detector is None:
                print(f"[DETECT_SYNC] Pool returned None, creating standalone detector")
                detector = BubbleDetector()
                temp_translator = None
            # Immediately update GUI pool tracker after checkout
            elif hasattr(self, 'update_queue'):
                self.update_queue.put(('update_pool_tracker', None))
        except Exception as e:
            print(f"[DETECT_SYNC] Failed to get detector from pool, creating standalone: {e}")
            detector = BubbleDetector()
            temp_translator = None
        
        # Extract settings from config
        detector_type = detection_config['detector_type']
        model_path = detection_config['model_path']
        model_url = detection_config['model_url']
        confidence = detection_config['confidence']
        detect_free_text = detection_config.get('detect_free_text', True)
        detect_empty_bubbles = detection_config.get('detect_empty_bubbles', True)
        detect_text_bubbles = detection_config.get('detect_text_bubbles', True)
        
        # Load the appropriate model based on user settings
        success = False
        if detector_type == 'rtdetr_onnx':
            # Use model_path if available, otherwise use model_url
            model_source = model_path if (model_path and os.path.exists(model_path)) else model_url
            success = detector.load_rtdetr_onnx_model(model_source)
            print(f"[DETECT_SYNC] Loading RT-DETR ONNX model: {os.path.basename(model_source) if model_path else model_source}")
        elif detector_type == 'rtdetr':
            success = detector.load_rtdetr_model(model_id=model_url or 'ogkalu/comic-text-and-bubble-detector')
            print(f"[DETECT_SYNC] Loading RT-DETR model: {model_url}")
        elif detector_type == 'yolo' and model_path:
            success = detector.load_model(model_path)
            print(f"[DETECT_SYNC] Loading YOLO model: {os.path.basename(model_path)}")
        elif detector_type == 'custom' and model_path:
            success = detector.load_model(model_path)
            print(f"[DETECT_SYNC] Loading custom model: {os.path.basename(model_path)}")
        else:
            # Default fallback
            success = detector.load_rtdetr_onnx_model('ogkalu/comic-text-and-bubble-detector')
            print(f"[DETECT_SYNC] Loading default RT-DETR ONNX model")
        
        if not success:
            print(f"[DETECT_SYNC] Failed to load bubble detection model")
            return []
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[DETECT_SYNC] Failed to load image: {os.path.basename(image_path)}")
            return []
        
        # ===== CANCELLATION CHECK: Before running detection =====
        if _is_translation_cancelled(self):
            print(f"[DETECT_SYNC] Cancelled before detection")
            return []
        
        # Run bubble detection
        print(f"[DETECT_SYNC] Running bubble detection (confidence: {confidence:.2f})")
        
        # Use appropriate detection method based on detector type
        if detector_type in ['rtdetr_onnx', 'rtdetr']:
            # For RT-DETR, get detailed detection results to avoid double boxes
            if detector_type == 'rtdetr_onnx' and hasattr(detector, 'detect_with_rtdetr_onnx'):
                detection_results = detector.detect_with_rtdetr_onnx(image_path, confidence=confidence, return_all_bubbles=False)
                # Combine enabled bubble types based on settings
                empty_bubbles = detection_results.get('bubbles', [])
                text_bubbles = detection_results.get('text_bubbles', [])
                text_free = detection_results.get('text_free', [])
                
                boxes = []
                if detect_empty_bubbles:
                    boxes.extend(empty_bubbles)
                if detect_text_bubbles:
                    boxes.extend(text_bubbles)
                if detect_free_text:
                    boxes.extend(text_free)
                
                print(f"[DETECT_SYNC] RT-DETR ONNX: {len(empty_bubbles)} empty + {len(text_bubbles)} text bubbles + {len(text_free)} free text")
                print(f"[DETECT_SYNC] Filters: empty={detect_empty_bubbles}, text_bubbles={detect_text_bubbles}, free_text={detect_free_text}")
                print(f"[DETECT_SYNC] Result: {len(boxes)} regions included after filtering")
            elif detector_type == 'rtdetr' and hasattr(detector, 'detect_with_rtdetr'):
                detection_results = detector.detect_with_rtdetr(image_path, confidence=confidence, return_all_bubbles=False)
                # Combine enabled bubble types based on settings
                empty_bubbles = detection_results.get('bubbles', [])
                text_bubbles = detection_results.get('text_bubbles', [])
                text_free = detection_results.get('text_free', [])
                
                boxes = []
                if detect_empty_bubbles:
                    boxes.extend(empty_bubbles)
                if detect_text_bubbles:
                    boxes.extend(text_bubbles)
                if detect_free_text:
                    boxes.extend(text_free)
                
                print(f"[DETECT_SYNC] RT-DETR: {len(empty_bubbles)} empty + {len(text_bubbles)} text bubbles + {len(text_free)} free text")
                print(f"[DETECT_SYNC] Filters: empty={detect_empty_bubbles}, text_bubbles={detect_text_bubbles}, free_text={detect_free_text}")
                print(f"[DETECT_SYNC] Result: {len(boxes)} regions included after filtering")
            else:
                # Fallback to old method
                boxes = detector.detect_bubbles(image_path, confidence=confidence, use_rtdetr=True)
        else:
            boxes = detector.detect_bubbles(image_path, confidence=confidence)
        
        if not boxes:
            print(f"[DETECT_SYNC] No text regions detected")
            return []
        
        print(f"[DETECT_SYNC] Found {len(boxes)} text regions")
        
        # Merge overlapping/nested boxes to avoid duplicates (align with regular pipeline)
        try:
            from manga_translator import merge_overlapping_boxes
            norm_boxes = []
            for b in boxes:
                try:
                    x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    norm_boxes.append([x, y, w, h])
                except Exception:
                    continue
            original_count = len(norm_boxes)
            merged_boxes = merge_overlapping_boxes(norm_boxes, containment_threshold=0.3, overlap_threshold=0.5)
            if merged_boxes and len(merged_boxes) < original_count:
                print(f"[DETECT_SYNC] Merged {original_count} boxes → {len(merged_boxes)} unique regions")
            boxes = merged_boxes or norm_boxes
        except Exception as me:
            print(f"[DETECT_SYNC] Merge step failed or unavailable: {me}")
        
        # Build RT-DETR class membership sets (if available) for bubble-aware metadata
        def _norm_box_local(b):
            try:
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            except Exception:
                return tuple(b)
        text_bubble_set, free_text_set, empty_bubble_set = set(), set(), set()
        try:
            if isinstance(detection_results, dict):
                text_bubble_set = set(_norm_box_local(b) for b in (detection_results.get('text_bubbles') or []))
                free_text_set = set(_norm_box_local(b) for b in (detection_results.get('text_free') or []))
                empty_bubble_set = set(_norm_box_local(b) for b in (detection_results.get('bubbles') or []))
        except Exception:
            pass
        
        # Process detection boxes and store regions
        regions = []
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                # Validate and convert coordinates
                try:
                    # Extract box coordinates (x, y, width, height format)
                    x, y, width, height = [int(v) for v in box[:4]]
                    
                    # Calculate bottom-right coordinates from dimensions
                    x2 = x + width
                    y2 = y + height
                    
                    # Clamp coordinates to image bounds
                    x = max(0, min(x, image.shape[1] - 1))
                    y = max(0, min(y, image.shape[0] - 1))
                    x2 = max(x + 1, min(x2, image.shape[1]))
                    y2 = max(y + 1, min(y2, image.shape[0]))
                    
                    # Recalculate width and height after clamping
                    width = x2 - x
                    height = y2 - y
                    
                    # Expand ellipse by 10% if circle mode is active (only for Detect Sync)
                    if getattr(self, '_use_circle_shapes', False):
                        cx = x + width / 2.0
                        cy = y + height / 2.0
                        scale = 1.20
                        new_w = max(1, int(round(width * scale)))
                        new_h = max(1, int(round(height * scale)))
                        nx = int(round(cx - new_w / 2))
                        ny = int(round(cy - new_h / 2))
                        nx2 = nx + new_w
                        ny2 = ny + new_h
                        nx = max(0, min(nx, image.shape[1] - 1))
                        ny = max(0, min(ny, image.shape[0] - 1))
                        nx2 = max(nx + 1, min(nx2, image.shape[1]))
                        ny2 = max(ny + 1, min(ny2, image.shape[0]))
                        x, y, width, height = nx, ny, (nx2 - nx), (ny2 - ny)
                    
                    # Classify bubble type using RT-DETR sets if available
                    norm_box = (x, y, width, height)
                    if norm_box in free_text_set:
                        bubble_type = 'free_text'
                    elif norm_box in text_bubble_set:
                        bubble_type = 'text_bubble'
                    elif norm_box in empty_bubble_set:
                        bubble_type = 'empty_bubble'
                    else:
                        bubble_type = 'text_bubble'
                    region_type = 'free_text' if bubble_type == 'free_text' else 'text_bubble'
                    
                    region_dict = {
                        'bbox': [x, y, width, height],  # (x, y, width, height)
                        'coords': [[x, y], [x2, y], [x2, y2], [x, y2]],  # Corner coordinates
                        'confidence': getattr(box, 'confidence', confidence) if hasattr(box, 'confidence') else confidence,
                        'shape': 'ellipse' if getattr(self, '_use_circle_shapes', False) else 'rect',
                        'bubble_type': bubble_type,
                        'region_type': region_type,
                        'bubble_bounds': [x, y, width, height]
                    }
                    regions.append(region_dict)
                    
                except (ValueError, IndexError) as e:
                    print(f"[DETECT_SYNC] Skipping invalid box {i}: {e}")
                    continue
        
        print(f"[DETECT_SYNC] Detection complete! Found {len(regions)} valid regions")
        return regions
        
    except Exception as e:
        import traceback
        print(f"[DETECT_SYNC] Synchronous detection failed: {str(e)}")
        print(f"[DETECT_SYNC] Traceback: {traceback.format_exc()}")
        return []
    finally:
        # Return detector to pool if checked out via temporary translator
        try:
            if temp_translator is not None:
                temp_translator._return_bubble_detector_to_pool()
                # Immediately update GUI pool tracker after return
                if hasattr(self, 'update_queue'):
                    self.update_queue.put(('update_pool_tracker', None))
        except Exception as e:
            print(f"[DETECT_SYNC] Failed to return detector to pool: {e}")

def _run_inpainting_sync(self, image_path: str, regions: list) -> str:
    """Run inpainting synchronously (for Translate button) and return cleaned image path
    
    Args:
        image_path: Path to the original image
        regions: List of region dictionaries with 'bbox' keys
        
    Returns:
        str: Path to cleaned image, or None if inpainting failed
    """
    temp_translator = None  # Track temporary translator for pool cleanup
    try:
        # ===== CANCELLATION CHECK: At start of sync inpainting =====
        if _is_translation_cancelled(self):
            print(f"[INPAINT_SYNC] Cancelled at start")
            return None
        
        import cv2
        import numpy as np
        from local_inpainter import LocalInpainter
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[INPAINT_SYNC] Failed to load image: {os.path.basename(image_path)}")
            return None
        
        # Get exclusion list from state management
        excluded_regions = []
        try:
            if hasattr(self, 'image_state_manager'):
                state = self.image_state_manager.get_state(image_path)
                excluded_regions = state.get('excluded_from_clean', [])
                if excluded_regions:
                    print(f"[INPAINT_SYNC] Found {len(excluded_regions)} excluded regions: {excluded_regions}")
        except Exception as e:
            print(f"[INPAINT_SYNC] Error getting exclusion list: {e}")
        
        # Create mask from detected regions (excluding marked ones)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        regions_to_inpaint = []
        excluded_count = 0
        
        print(f"[INPAINT_SYNC] Processing {len(regions)} regions for inpainting")
        for i, region in enumerate(regions):
            # Check if this region should be excluded
            if i in excluded_regions:
                excluded_count += 1
                print(f"[INPAINT_SYNC] Skipping region {i} (excluded from clean)")
                continue
            
            regions_to_inpaint.append((i, region))
        
        print(f"[INPAINT_SYNC] Creating mask from {len(regions_to_inpaint)} regions ({excluded_count} excluded)")
        for region_index, region in regions_to_inpaint:
            # Handle both dictionary format (from detect) and object format (from translator)
            if isinstance(region, dict):
                # Dictionary format from detect button
                bbox = region.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, width, height = bbox
                    x1, y1, x2, y2 = x, y, x + width, y + height
                else:
                    continue
            else:
                # Object format from translator
                x1, y1, x2, y2 = int(region.x1), int(region.y1), int(region.x2), int(region.y2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, image.shape[1]))
            y2 = max(y1 + 1, min(y2, image.shape[0]))
            
            # Determine shape for mask
            shape = None
            try:
                if isinstance(region, dict):
                    shape = region.get('shape')
            except Exception:
                shape = None
            use_ellipse = bool(shape == 'ellipse' or getattr(self, '_use_circle_shapes', False))
            
            if shape == 'polygon' and isinstance(region.get('polygon'), list) and len(region.get('polygon')) >= 3:
                import numpy as _np
                pts = _np.array(region['polygon'], dtype=_np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            elif use_ellipse:
                # Draw filled ellipse that fits the bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                rx = max(1, int((x2 - x1) / 2))
                ry = max(1, int((y2 - y1) / 2))
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            else:
                # Draw filled rectangle on mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Get inpainting settings from manga integration config
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        
        if inpaint_method == 'local':
            # Use local inpainter with the same method as manga_translator
            print(f"[INPAINT_SYNC] Using local inpainter: {local_model}")
            
            # Get model path from config (same way as manga_translator)
            model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
            
            # Ensure we have a model path (download if needed)
            resolved_model_path = model_path
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                try:
                    from local_inpainter import LocalInpainter
                    print(f"[INPAINT_SYNC] Downloading {local_model} model...")
                    temp_inp = LocalInpainter()
                    resolved_model_path = temp_inp.download_jit_model(local_model)
                except Exception as e:
                    print(f"[INPAINT_SYNC] Model download failed: {e}")
                    resolved_model_path = None
            
            # Use shared inpainter via pool - check out from class-level pool
            # IMPORTANT: Wait/poll for preloaded inpainter instead of creating new instance
            if resolved_model_path and os.path.exists(resolved_model_path):
                try:
                    from manga_translator import MangaTranslator
                    from local_inpainter import LocalInpainter
                    import time
                    
                    # Normalize model path to match pool key
                    resolved_model_path = os.path.abspath(os.path.normpath(resolved_model_path))
                    key = (local_model, resolved_model_path)
                    
                    # Poll for inpainter from pool with timeout
                    # This waits for preloading to complete instead of creating a new instance
                    inpainter = None
                    poll_timeout = 60  # Wait up to 60 seconds for preloading
                    poll_interval = 0.5  # Check every 500ms
                    start_time = time.time()
                    attempt = 0
                    
                    while time.time() - start_time < poll_timeout:
                        attempt += 1
                        
                        # Check for cancellation while polling
                        if _is_translation_cancelled(self):
                            print(f"[INPAINT_SYNC] Cancelled while waiting for inpainter")
                            return None
                        
                        with MangaTranslator._inpaint_pool_lock:
                            rec = MangaTranslator._inpaint_pool.get(key)
                            
                            if rec and rec.get('spares'):
                                spares = rec.get('spares', [])
                                checked_out = rec.setdefault('checked_out', [])
                                
                                # Find an available spare that's fully loaded
                                for spare in spares:
                                    if spare not in checked_out and spare and getattr(spare, 'model_loaded', False):
                                        checked_out.append(spare)
                                        inpainter = spare
                                        print(f"[INPAINT_SYNC] Checked out inpainter from pool ({len(checked_out)}/{len(spares)} in use)")
                                        break
                        
                        if inpainter:
                            break
                        
                        # Log waiting status periodically
                        elapsed = time.time() - start_time
                        if attempt == 1 or (elapsed >= 2 and int(elapsed) % 5 == 0):
                            print(f"[INPAINT_SYNC] Waiting for preloaded inpainter... ({int(elapsed)}s)")
                            self._log(f"⏳ Waiting for inpainter to load... ({int(elapsed)}s)", "info")
                        
                        time.sleep(poll_interval)
                    
                    # If still no inpainter after timeout, create one as last resort
                    if not inpainter:
                        print(f"[INPAINT_SYNC] Timeout waiting for pool, creating new inpainter...")
                        self._log(f"⚠️ Inpainter pool not ready, loading new instance...", "warning")
                        new_inpainter = LocalInpainter()
                        if new_inpainter.load_model(local_model, resolved_model_path):
                            with MangaTranslator._inpaint_pool_lock:
                                rec = MangaTranslator._inpaint_pool.get(key)
                                if not rec:
                                    MangaTranslator._inpaint_pool[key] = {
                                        'spares': [],
                                        'checked_out': [],
                                        'model_type': local_model,
                                        'model_path': resolved_model_path
                                    }
                                    rec = MangaTranslator._inpaint_pool[key]
                                rec['spares'].append(new_inpainter)
                                rec['checked_out'].append(new_inpainter)
                                inpainter = new_inpainter
                            print(f"[INPAINT_SYNC] Created and checked out new inpainter")
                        else:
                            print(f"[INPAINT_SYNC] Failed to load new inpainter model")
                            return None
                    
                    if inpainter:
                        # Store key for return
                        inpainter._pool_key = key
                        # Update GUI pool tracker after checkout
                        if hasattr(self, 'update_queue'):
                            self.update_queue.put(('update_pool_tracker', None))
                    else:
                        print(f"[INPAINT_SYNC] Failed to get inpainter from pool")
                        return None
                        
                except Exception as e:
                    print(f"[INPAINT_SYNC] Failed to checkout inpainter: {e}")
                    return None
            else:
                print(f"[INPAINT_SYNC] No valid model path for {local_model}")
                return None
            
            # ===== CANCELLATION CHECK: Before running inpainting =====
            if _is_translation_cancelled(self):
                print(f"[INPAINT_SYNC] Cancelled before inpainting")
                return None
            
            # Run inpainting
            print(f"[INPAINT_SYNC] Running local inpainting...")
            cleaned_image = inpainter.inpaint(image, mask)
            
            # Return inpainter to pool AFTER inpainting completes
            try:
                if inpainter and hasattr(inpainter, '_pool_key'):
                    from manga_translator import MangaTranslator
                    key = inpainter._pool_key
                    
                    # Log the return operation
                    try:
                        method, path = key
                        path_basename = os.path.basename(path) if path else 'None'
                        logging.info(f"🔑 Return inpainter model: {method}/{path_basename}")
                    except Exception:
                        pass
                    
                    with MangaTranslator._inpaint_pool_lock:
                        rec = MangaTranslator._inpaint_pool.get(key)
                        if rec and 'checked_out' in rec:
                            checked_out = rec['checked_out']
                            if inpainter in checked_out:
                                checked_out.remove(inpainter)
                                
                                # Log pool status after return
                                spares_list = rec.get('spares', [])
                                total_spares = len(spares_list)
                                checked_out_count = len(checked_out)
                                available_count = total_spares - checked_out_count
                                valid_spares = sum(1 for s in spares_list if s and getattr(s, 'model_loaded', False))
                                
                                try:
                                    method, path = key
                                    path_basename = os.path.basename(path) if path else 'None'
                                    logging.info(f"🔄 Returned inpainter to pool [key: {method}/{path_basename}] ({checked_out_count}/{total_spares} in use, {available_count} available, {valid_spares} valid)")
                                except Exception:
                                    logging.info(f"🔄 Returned inpainter to pool ({checked_out_count}/{total_spares} in use, {available_count} available, {valid_spares} valid)")
                    
                    # Update GUI pool tracker after return
                    if hasattr(self, 'update_queue'):
                        self.update_queue.put(('update_pool_tracker', None))
            except Exception as e:
                logging.warning(f"⚠️ Failed to return inpainter to pool: {e}")
            
        else:
            # For cloud/hybrid methods, would need more complex setup
            # For now, fallback to basic OpenCV inpainting
            print(f"[INPAINT_SYNC] Using OpenCV inpainting (fallback)")
            cleaned_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        if cleaned_image is not None:
            # Save cleaned image into per-image isolated folder and return path
            # Check for OUTPUT_DIRECTORY override (prefer config over env var)
            parent_dir = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            base, ext = os.path.splitext(filename)
            
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            if override_dir:
                output_dir = os.path.join(override_dir, f"{base}_translated")
                print(f"[INPAINT_SYNC] Using output directory override: {override_dir}")
            else:
                output_dir = os.path.join(parent_dir, f"{base}_translated")
            
            os.makedirs(output_dir, exist_ok=True)
            cleaned_path = os.path.join(output_dir, f"{base}_cleaned{ext}")

            cv2.imwrite(cleaned_path, cleaned_image)
            print(f"[INPAINT_SYNC] Saved cleaned image to: {cleaned_path}")
            return cleaned_path
        else:
            print(f"[INPAINT_SYNC] Inpainting returned None")
            return None
            
    except Exception as e:
        import traceback
        print(f"[INPAINT_SYNC] Synchronous inpainting failed: {str(e)}")
        print(f"[INPAINT_SYNC] Traceback: {traceback.format_exc()}")
        # Return inpainter to pool on error
        try:
            if temp_translator is not None:
                temp_translator._return_inpainter_to_pool()
                if hasattr(self, 'update_queue'):
                    self.update_queue.put(('update_pool_tracker', None))
                print(f"[INPAINT_SYNC] Returned inpainter to pool after error")
        except Exception:
            pass
        return None

def _run_ocr_on_regions(self, image_path: str, regions: list, ocr_config: dict) -> list:
    """Run OCR on regions and return recognized texts
    
    This is the core OCR logic extracted for reuse by both recognize and translate.
    Runs OCR on full image and matches results to detected regions.
    
    Args:
        image_path: Path to image
        regions: List of region dicts to recognize text in
        ocr_config: OCR configuration dict
        
    Returns:
        list: List of recognized text dicts with 'region_index', 'bbox', 'text', 'confidence'
    """
    try:
        # ===== CANCELLATION CHECK: At start of OCR =====
        if _is_translation_cancelled(self):
            print(f"[OCR_REGIONS] Cancelled at start")
            return []
        
        import cv2
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[OCR_REGIONS] Failed to load image: {os.path.basename(image_path)}")
            return []
        
        # Initialize OCR manager if not already done
        if not hasattr(self, 'ocr_manager') or not self.ocr_manager:
            from ocr_manager import OCRManager
            self.ocr_manager = OCRManager(log_callback=self._log)
        
        recognized_texts = []
        
        print(f"[OCR_REGIONS] Running OCR on full image, then matching to {len(regions)} regions")
        
        # STEP 1: Run OCR on regions
        provider = ocr_config['provider']
        full_image_ocr_results = []
        
        # SPECIAL HANDLING: custom-api and Qwen2-VL process CROPPED regions due to API call indexing
        if provider in ['custom-api', 'Qwen2-VL']:
            print(f"[OCR_REGIONS] Running custom-api OCR on cropped regions (required for API indexing)")
            try:
                # Set environment variables exactly like Start Translation (excluding SYSTEM_PROMPT)
                # 1) Fetch API key and model from GUI
                api_key = None
                if hasattr(self, 'main_gui'):
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

                # 2) Apply all environment variables from GUI except SYSTEM_PROMPT
                try:
                    if hasattr(self, 'main_gui') and hasattr(self.main_gui, '_get_environment_variables'):
                        env_vars = self.main_gui._get_environment_variables(
                            epub_path='',  # Not needed for manga
                            api_key=api_key or ''
                        )
                        for key, value in env_vars.items():
                            if key == 'SYSTEM_PROMPT':
                                # DON'T SET THE TRANSLATION SYSTEM PROMPT FOR OCR
                                continue
                            os.environ[key] = str(value)
                        self._log("✅ Set environment variables for custom-api OCR (excluded SYSTEM_PROMPT)", "info")
                    else:
                        print("[OCR_REGIONS] _get_environment_variables not available on main_gui")
                except Exception as env_err:
                    print(f"[OCR_REGIONS] Failed to apply GUI environment variables: {env_err}")

                # 3) Set OCR prompt from GUI or fallback to default strict OCR prompt
                try:
                    if hasattr(self, 'ocr_prompt') and self.ocr_prompt:
                        os.environ['OCR_SYSTEM_PROMPT'] = self.ocr_prompt
                        self._log(f"✅ Using custom OCR prompt from GUI ({len(self.ocr_prompt)} chars)", "info")
                        self._log(f"OCR Prompt being set: {self.ocr_prompt[:150]}...", "debug")
                    else:
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
                except Exception:
                    pass

                # 4) Respect user settings: set non-intrusive defaults only when bubble detection is OFF
                try:
                    ms = self.main_gui.config.setdefault('manga_settings', {})
                    ocr_set = ms.setdefault('ocr', {})
                    changed = False
                    bubble_enabled = bool(ocr_set.get('bubble_detection_enabled', False))
                    if not bubble_enabled:
                        if 'detector_type' not in ocr_set:
                            ocr_set['detector_type'] = 'rtdetr_onnx'
                            changed = True
                        if not ocr_set.get('rtdetr_model_url') and not ocr_set.get('bubble_model_path'):
                            ocr_set['rtdetr_model_url'] = 'ogkalu/comic-text-and-bubble-detector'
                            changed = True
                        if changed and hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                    # Do not preload bubble detector here for custom-api
                    self._preloaded_bd = None
                except Exception:
                    self._preloaded_bd = None

                # 5) Load custom-api provider if needed - MUST PASS API KEY/MODEL
                if not self.ocr_manager.get_provider(provider).is_loaded:
                    print(f"[OCR_REGIONS] Loading OCR provider: {provider}")
                    load_kwargs = {}
                    if api_key:
                        load_kwargs['api_key'] = api_key
                        print(f"[OCR_REGIONS] Got API key from GUI")
                    # Model from GUI
                    model = 'gpt-4o-mini'
                    if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'model_var'):
                        try:
                            if hasattr(self.main_gui.model_var, 'get'):
                                model = self.main_gui.model_var.get()
                            else:
                                model = self.main_gui.model_var
                        except Exception as e:
                            print(f"[OCR_REGIONS] Error getting model from model_var: {e}")
                            model = 'gpt-4o-mini'
                    elif hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
                        model = self.main_gui.config.get('model')
                    if model:
                        load_kwargs['model'] = model
                        print(f"[OCR_REGIONS] Using model: {model}")
                    load_success = self.ocr_manager.load_provider(provider, **load_kwargs)
                    print(f"[OCR_REGIONS] Provider load result: {load_success}")
                    if not load_success:
                        self._log(f"❌ Failed to load {provider}", "error")
                        return []
                
                # Process each region individually (cropped)
                for i, region in enumerate(regions):
                    # ===== CANCELLATION CHECK: In OCR loop =====
                    if _is_translation_cancelled(self):
                        print(f"[OCR_REGIONS] Cancelled at region {i+1}/{len(regions)}")
                        return []
                    
                    bbox = region.get('bbox', [])
                    if len(bbox) >= 4:
                        region_x, region_y, region_w, region_h = bbox
                        
                        # Crop the region from the full image
                        cropped_region = image[region_y:region_y+region_h, region_x:region_x+region_w]
                        
                        # Run OCR on cropped region
                        ocr_results = self.ocr_manager.detect_text(
                            cropped_region,
                            provider,
                            confidence=0.5
                        )
                        
                        # Combine all text from this region
                        if ocr_results:
                            region_text = " ".join([ocr.text.strip() for ocr in ocr_results if ocr.text.strip()])
                            if region_text:
                                recognized_texts.append({
                                    'region_index': i,
                                    'bbox': bbox,
                                    'text': region_text.strip(),
                                    'confidence': region.get('confidence', 1.0),
                                    'bubble_type': region.get('bubble_type'),
                                    'region_type': region.get('region_type'),
                                    'bubble_bounds': region.get('bubble_bounds', bbox)
                                })
                                print(f"[OCR_REGIONS] Region {i+1}: '{region_text.strip()}'")
                
                print(f"[OCR_REGIONS] custom-api recognized text in {len(recognized_texts)}/{len(regions)} regions")
                return recognized_texts
                
            except Exception as e:
                print(f"[OCR_REGIONS] custom-api OCR error: {str(e)}")
                import traceback
                print(f"[OCR_REGIONS] custom-api traceback: {traceback.format_exc()}")
                self._log(f"❌ custom-api OCR failed: {str(e)}", "error")
                return []
        
        elif provider == 'google':
            # ===== CANCELLATION CHECK: Before Google OCR =====
            if _is_translation_cancelled(self):
                print(f"[OCR_REGIONS] Cancelled before Google OCR")
                return []
            
            # Use Google Cloud Vision OCR on full image
            print(f"[OCR_REGIONS] Running Google Cloud Vision OCR on full image")
            try:
                from google.cloud import vision
                import io
                
                # Get credentials path
                google_creds = ocr_config.get('google_credentials_path', '')
                if not google_creds or not os.path.exists(google_creds):
                    self._log("❌ Google Cloud Vision credentials not found", "error")
                    print(f"[OCR_REGIONS] Google credentials not found: {google_creds}")
                    return []
                
                # Set credentials environment variable
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                
                # Create client
                client = vision.ImageAnnotatorClient()
                
                # Convert full image to bytes
                _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                image_bytes = encoded.tobytes()
                
                # Call Google Cloud Vision API
                vision_image = vision.Image(content=image_bytes)
                response = client.text_detection(image=vision_image)
                
                if response.error.message:
                    raise Exception(f"Google API error: {response.error.message}")
                
                # Extract text annotations
                texts = response.text_annotations
                
                if texts:
                    # Skip first annotation (full text) and process individual words
                    for text in texts[1:]:
                        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                        xs = [v[0] for v in vertices]
                        ys = [v[1] for v in vertices]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        from ocr_manager import OCRResult
                        ocr_line = OCRResult(
                            text=text.description,
                            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=0.9,
                            vertices=vertices
                        )
                        full_image_ocr_results.append(ocr_line)
                
                print(f"[OCR_REGIONS] Google OCR found {len(full_image_ocr_results)} text regions")
                
                # ===== CANCELLATION CHECK: After Google OCR =====
                if _is_translation_cancelled(self):
                    print(f"[OCR_REGIONS] Cancelled after Google OCR - discarding results")
                    return []
                
            except Exception as e:
                print(f"[OCR_REGIONS] Google OCR error: {str(e)}")
                import traceback
                print(f"[OCR_REGIONS] Google OCR traceback: {traceback.format_exc()}")
                self._log(f"❌ Google Cloud Vision failed: {str(e)}", "error")
                return []
        
        elif provider in ['azure', 'azure-document-intelligence']:
            # ===== CANCELLATION CHECK: Before Azure OCR =====
            if _is_translation_cancelled(self):
                print(f"[OCR_REGIONS] Cancelled before Azure OCR")
                return []
            
            # Use correct Azure API per provider name
            print(f"[OCR_REGIONS] Running {provider} OCR on full image")
            try:
                if provider == 'azure':
                    # Azure Computer Vision (Image Analysis) path
                    from azure.ai.vision.imageanalysis import ImageAnalysisClient
                    from azure.core.credentials import AzureKeyCredential
                    from azure.ai.vision.imageanalysis.models import VisualFeatures
                    import time
                    
                    azure_endpoint = ocr_config.get('azure_endpoint') or ocr_config.get('endpoint', '')
                    azure_key = ocr_config.get('azure_key') or ocr_config.get('key', '')
                    if not azure_endpoint or not azure_key:
                        print(f"[OCR_REGIONS] Missing Azure credentials: endpoint={bool(azure_endpoint)}, key={bool(azure_key)}")
                        self._log(f"❌ Azure credentials not configured", "error")
                        return []
                    vision_client = ImageAnalysisClient(
                        endpoint=azure_endpoint,
                        credential=AzureKeyCredential(azure_key)
                    )
                    # Convert full image to bytes
                    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    image_bytes = encoded.tobytes()
                    # Call Azure OCR on full image with retry logic
                    import concurrent.futures
                    max_retries = 3
                    result = None
                    
                    for attempt in range(max_retries):
                        try:
                            start_time = time.time()
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(
                                    vision_client.analyze,
                                    image_data=image_bytes,
                                    visual_features=[VisualFeatures.READ]
                                )
                                result = future.result(timeout=30.0)
                                elapsed = time.time() - start_time
                                
                                if attempt > 0:
                                    print(f"[OCR_REGIONS] ✅ Azure OCR succeeded on retry {attempt} ({elapsed:.2f}s)")
                                    self._log(f"✅ Azure OCR succeeded on retry {attempt}", "info")
                                else:
                                    print(f"[OCR_REGIONS] Azure OCR completed in {elapsed:.2f}s")
                                break
                                
                        except (concurrent.futures.TimeoutError, Exception) as e:
                            elapsed = time.time() - start_time
                            error_msg = str(e)
                            
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                                print(f"[OCR_REGIONS] Azure OCR attempt {attempt + 1} failed after {elapsed:.1f}s: {error_msg}")
                                print(f"[OCR_REGIONS] Retrying in {wait_time}s...")
                                self._log(f"⚠️ Azure OCR timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})", "warning")
                                time.sleep(wait_time)
                            else:
                                print(f"[OCR_REGIONS] Azure OCR failed after {max_retries} attempts")
                                self._log(f"❌ Azure OCR failed after {max_retries} attempts", "error")
                                return []
                    
                    if not result:
                        return []
                    # Extract all text lines from full image OCR
                    if result.read and result.read.blocks:
                        for line in result.read.blocks[0].lines:
                            if hasattr(line, 'bounding_polygon') and line.bounding_polygon:
                                points = line.bounding_polygon
                                xs = [p.x for p in points]
                                ys = [p.y for p in points]
                                x_min, x_max = int(min(xs)), int(max(xs))
                                y_min, y_max = int(min(ys)), int(max(ys))
                                from ocr_manager import OCRResult
                                ocr_line = OCRResult(
                                    text=line.text,
                                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                                    confidence=0.9,
                                    vertices=[(int(p.x), int(p.y)) for p in points]
                                )
                                full_image_ocr_results.append(ocr_line)
                    print(f"[OCR_REGIONS] Azure OCR found {len(full_image_ocr_results)} text lines")
                else:
                    # Azure Document Intelligence path via OCRManager provider (Form Recognizer)
                    provider_obj = self.ocr_manager.get_provider('azure-document-intelligence')
                    if provider_obj is None:
                        self._log(f"❌ OCR provider 'azure-document-intelligence' not available in OCRManager", "error")
                        return []
                    # Load provider with endpoint/key if needed
                    if not provider_obj.is_loaded:
                        print(f"[OCR_REGIONS] Loading OCR provider: azure-document-intelligence")
                        load_success = self.ocr_manager.load_provider('azure-document-intelligence', **ocr_config)
                        print(f"[OCR_REGIONS] Provider load result: {load_success}")
                        if not load_success:
                            self._log(f"❌ Failed to load azure-document-intelligence", "error")
                            return []
                    # Run full-image OCR using Document Intelligence
                    full_image_ocr_results = self.ocr_manager.detect_text(
                        image,
                        'azure-document-intelligence',
                        **ocr_config
                    )
                    print(f"[OCR_REGIONS] azure-document-intelligence found {len(full_image_ocr_results)} text regions")
                
                # ===== CANCELLATION CHECK: After Azure OCR =====
                if _is_translation_cancelled(self):
                    print(f"[OCR_REGIONS] Cancelled after Azure OCR - discarding results")
                    return []
            except Exception as e:
                print(f"[OCR_REGIONS] Azure OCR error: {str(e)}")
                import traceback
                print(f"[OCR_REGIONS] Azure OCR traceback: {traceback.format_exc()}")
                self._log(f"❌ Azure OCR failed: {str(e)}", "error")
                return []
                
        elif provider == 'manga-ocr':
            # For manga-ocr, process each region individually (cropped) for better accuracy
            print(f"[OCR_REGIONS] Running manga-ocr OCR on cropped regions")
            try:
                # Check if provider exists in OCRManager
                provider_obj = self.ocr_manager.get_provider(provider)
                if provider_obj is None:
                    self._log(f"❌ OCR provider 'manga-ocr' not available in OCRManager", "error")
                    print(f"[OCR_REGIONS] Provider 'manga-ocr' not found in OCRManager")
                    return []
                
                # Load provider if not already loaded
                if not provider_obj.is_loaded:
                    print(f"[OCR_REGIONS] Loading OCR provider: manga-ocr")
                    load_success = self.ocr_manager.load_provider(provider, **ocr_config)
                    print(f"[OCR_REGIONS] Provider load result: {load_success}")
                    if not load_success:
                        self._log(f"❌ Failed to load manga-ocr", "error")
                        return []
                
                # Process each region individually (cropped) - same approach as custom-api
                for i, region in enumerate(regions):
                    # ===== CANCELLATION CHECK: In manga-ocr loop =====
                    if _is_translation_cancelled(self):
                        print(f"[OCR_REGIONS] Cancelled at manga-ocr region {i+1}/{len(regions)}")
                        return []
                    
                    bbox = region.get('bbox', [])
                    if len(bbox) >= 4:
                        region_x, region_y, region_w, region_h = bbox
                        
                        # Crop the region from the full image
                        cropped_region = image[region_y:region_y+region_h, region_x:region_x+region_w]
                        
                        # Run OCR on cropped region
                        ocr_results = self.ocr_manager.detect_text(
                            cropped_region,
                            provider,
                            confidence=0.5
                        )
                        
                        # Combine all text from this region
                        if ocr_results:
                            region_text = " ".join([ocr.text.strip() for ocr in ocr_results if ocr.text.strip()])
                            if region_text:
                                recognized_texts.append({
                                    'region_index': i,
                                    'bbox': bbox,
                                    'text': region_text.strip(),
                                    'confidence': region.get('confidence', 1.0),
                                    'bubble_type': region.get('bubble_type'),
                                    'region_type': region.get('region_type'),
                                    'bubble_bounds': region.get('bubble_bounds', bbox)
                                })
                                print(f"[OCR_REGIONS] Region {i+1}: '{region_text.strip()}'")
                
                print(f"[OCR_REGIONS] manga-ocr recognized text in {len(recognized_texts)}/{len(regions)} regions")
                return recognized_texts
                
            except Exception as e:
                print(f"[OCR_REGIONS] manga-ocr OCR error: {str(e)}")
                import traceback
                print(f"[OCR_REGIONS] manga-ocr OCR traceback: {traceback.format_exc()}")
                self._log(f"❌ manga-ocr OCR failed: {str(e)}", "error")
                return []
                
        else:
            # ===== CANCELLATION CHECK: Before other OCR providers =====
            if _is_translation_cancelled(self):
                print(f"[OCR_REGIONS] Cancelled before {provider} OCR")
                return []
            
            # For non-Azure/custom-api providers, use OCRManager on full image
            print(f"[OCR_REGIONS] Running {provider} OCR on full image")
            try:
                # Check if provider exists in OCRManager
                provider_obj = self.ocr_manager.get_provider(provider)
                if provider_obj is None:
                    self._log(f"❌ OCR provider '{provider}' not available in OCRManager", "error")
                    print(f"[OCR_REGIONS] Provider '{provider}' not found in OCRManager")
                    return []
                
                # Load provider if not already loaded
                if not provider_obj.is_loaded:
                    print(f"[OCR_REGIONS] Loading OCR provider: {provider}")
                    load_success = self.ocr_manager.load_provider(provider, **ocr_config)
                    print(f"[OCR_REGIONS] Provider load result: {load_success}")
                    if not load_success:
                        self._log(f"❌ Failed to load {provider}", "error")
                        return []
                
                full_image_ocr_results = self.ocr_manager.detect_text(
                    image, 
                    provider,
                    confidence=0.5
                )
                print(f"[OCR_REGIONS] {provider} OCR found {len(full_image_ocr_results)} text regions")
                
                # ===== CANCELLATION CHECK: After other OCR =====
                if _is_translation_cancelled(self):
                    print(f"[OCR_REGIONS] Cancelled after {provider} OCR - discarding results")
                    return []
                
            except Exception as e:
                print(f"[OCR_REGIONS] {provider} OCR error: {str(e)}")
                import traceback
                print(f"[OCR_REGIONS] {provider} OCR traceback: {traceback.format_exc()}")
                self._log(f"❌ {provider} OCR failed: {str(e)}", "error")
                return []
        
        # ===== CANCELLATION CHECK: Before region matching =====
        if _is_translation_cancelled(self):
            print(f"[OCR_REGIONS] Cancelled before region matching")
            return []
        
        # STEP 2: Match OCR results to detected regions
        print(f"[OCR_REGIONS] Matching {len(full_image_ocr_results)} OCR results to {len(regions)} regions")
        
        for i, region in enumerate(regions):
            # Check cancellation periodically during region matching
            if i > 0 and i % 5 == 0 and _is_translation_cancelled(self):
                print(f"[OCR_REGIONS] Cancelled during region matching at region {i}")
                return []
            bbox = region.get('bbox', [])
            if len(bbox) >= 4:
                region_x, region_y, region_w, region_h = bbox
                region_center_x = region_x + region_w / 2
                region_center_y = region_y + region_h / 2
                
                # Find OCR results that overlap with this region
                matching_ocr = []
                for ocr_result in full_image_ocr_results:
                    ocr_x, ocr_y, ocr_w, ocr_h = ocr_result.bbox
                    ocr_center_x = ocr_x + ocr_w / 2
                    ocr_center_y = ocr_y + ocr_h / 2
                    
                    # Check if OCR result center is within region bounds
                    if (region_x <= ocr_center_x <= region_x + region_w and
                        region_y <= ocr_center_y <= region_y + region_h):
                        matching_ocr.append(ocr_result)
                
                # Combine matching OCR texts
                region_text = " ".join([ocr.text.strip() for ocr in matching_ocr if ocr.text.strip()])
                
                print(f"[OCR_REGIONS] Region {i+1}: Found {len(matching_ocr)} matching OCR results")
                
                if region_text:
                    recognized_texts.append({
                        'region_index': i,
                        'bbox': bbox,
                        'text': region_text.strip(),
                        'confidence': region.get('confidence', 1.0),
                        'bubble_type': region.get('bubble_type'),
                        'region_type': region.get('region_type'),
                        'bubble_bounds': region.get('bubble_bounds', bbox)
                    })
                    print(f"[OCR_REGIONS] Region {i+1}: '{region_text.strip()}'")
        
        print(f"[OCR_REGIONS] Recognized text in {len(recognized_texts)}/{len(regions)} regions")
        return recognized_texts
        
    except Exception as e:
        import traceback
        print(f"[OCR_REGIONS] Error: {str(e)}")
        print(f"[OCR_REGIONS] Traceback: {traceback.format_exc()}")
        return []

def _on_recognize_text_clicked(self):
    """Recognize text in current preview rectangles using selected OCR provider"""
    print("[DEBUG] _on_recognize_text_clicked called!")
    self._log("🐛 DEBUG: Recognize text button clicked", "debug")
    
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    # This MUST happen on the main thread BEFORE any cancellation checks
    _reset_cancellation_flags(self)
    
    try:
        # Debug: Check widget existence
        print(f"[DEBUG] Has image_preview_widget: {hasattr(self, 'image_preview_widget')}")
        if hasattr(self, 'image_preview_widget'):
            print(f"[DEBUG] Current image path: {self.image_preview_widget.current_image_path}")
            print(f"[DEBUG] Has viewer: {hasattr(self.image_preview_widget, 'viewer')}")
            if hasattr(self.image_preview_widget, 'viewer'):
                print(f"[DEBUG] Number of rectangles: {len(self.image_preview_widget.viewer.rectangles)}")
        
        # Check if we have an image
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            self._log("⚠️ No image loaded for text recognition", "warning")
            print("[DEBUG] No image loaded - returning early")
            return
        
        # Disable the recognize button to prevent multiple clicks
        print(f"[DEBUG] Has recognize_btn: {hasattr(self.image_preview_widget, 'recognize_btn')}")
        if hasattr(self.image_preview_widget, 'recognize_btn'):
            print(f"[DEBUG] Disabling recognize button")
            self.image_preview_widget.recognize_btn.setEnabled(False)
            self.image_preview_widget.recognize_btn.setText("Recognizing...")
        else:
            print("[DEBUG] No recognize_btn found!")
        
        image_path = self.image_preview_widget.current_image_path
        
        # Track which image we're recognizing for overlay removal
        self._recognized_texts_image_path = image_path
        
        # Add processing overlay effect (after tracking image)
        _add_processing_overlay(self, )
        
        # Get OCR settings
        ocr_config = _get_ocr_config(self, )
        print(f"[DEBUG] OCR config: {ocr_config}")
        self._log(f"🤖 Using OCR provider: {ocr_config['provider']}", "info")
        
        # Check if we have rectangles - if yes, extract them now; if no, will detect in background
        has_rectangles = (hasattr(self.image_preview_widget, 'viewer') and 
                        self.image_preview_widget.viewer.rectangles and 
                        len(self.image_preview_widget.viewer.rectangles) > 0)
        
        # Extract regions NOW if rectangles exist (don't pass to background thread)
        regions = None
        if has_rectangles:
            print("[DEBUG] Rectangles exist - extracting regions now")
            regions = _extract_regions_from_preview(self, )
            print(f"[DEBUG] Extracted {len(regions)} regions from preview")
            if not regions or len(regions) == 0:
                self._log("⚠️ No valid regions found in preview", "warning")
                _restore_recognize_button(self, )
                return
            # Replace viewer rectangles with merged regions so indices align with recognition
            try:
                self._current_regions = regions
                if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                    self.image_preview_widget.viewer.clear_rectangles()
                _draw_detection_boxes_on_preview(self, )
            except Exception:
                pass
            # Persist merged regions to state ONLY if no OCR data exists yet
            try:
                if hasattr(self, 'image_state_manager'):
                    current_state = self.image_state_manager.get_state(image_path)
                    if not current_state.get('recognized_texts') and not current_state.get('translated_texts'):
                        self.image_state_manager.update_state(image_path, {'detection_regions': regions})
                    else:
                        print(f"[STATE] Skipping detection_regions save - OCR/translation data exists")
            except Exception:
                pass
            self._log(f"📝 Starting text recognition on {len(regions)} existing regions using {ocr_config['provider']}", "info")
        else:
            print("[DEBUG] No rectangles found - will run detection in background")
            self._log("🔍 No text regions found - running automatic detection first...", "info")
        
        # Run OCR in background thread
        # Pass regions (if extracted) or None (will trigger detection in background)
        import threading
        thread = threading.Thread(target=_run_recognize_background, args=(self, image_path, regions, ocr_config),
                                daemon=True)
        thread.start()
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Recognize setup failed: {str(e)}"
        traceback_msg = traceback.format_exc()
        self._log(error_msg, "error")
        print(f"[DEBUG] {error_msg}")
        print(f"[DEBUG] Recognize setup error traceback: {traceback_msg}")
        _restore_recognize_button(self, )

def _run_recognize_background(self, image_path: str, regions: list, ocr_config: dict):
    """Run text recognition in background thread
    
    Args:
        image_path: Path to image
        regions: List of region dicts (if provided), or None to trigger detection
        ocr_config: OCR configuration dict
    """
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    _reset_cancellation_flags(self)
    
    try:
        # ===== CANCELLATION CHECK: At start of recognition =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Recognition cancelled before starting", "warning")
            print(f"[RECOGNIZE] Cancelled at start")
            self.update_queue.put(('recognize_button_restore', None))
            return
        
        import cv2
        
        # STEP 0: Check if we need to run detection first
        if regions is None:
            print("[RECOGNIZE] No regions provided - running detection first...")
            self._log("🔍 Running automatic text detection...", "info")
            
            # Run detection synchronously
            detection_config = _get_detection_config(self, )
            # Exclude empty bubble container rectangles in recognize pipeline to avoid doubles
            if detection_config.get('detect_empty_bubbles', True):
                detection_config['detect_empty_bubbles'] = False
                self._log("🚫 Recognize pipeline: Excluding empty bubble regions (container boxes)", "info")
            regions = _run_detection_sync(self, image_path, detection_config)
            
            if not regions or len(regions) == 0:
                self._log("⚠️ No text regions detected in image", "warning")
                self.update_queue.put(('recognize_button_restore', None))
                return
            
            print(f"[RECOGNIZE] Detection found {len(regions)} regions")
            self._log(f"✅ Detected {len(regions)} text regions", "success")
            
            # ===== CANCELLATION CHECK: After detection =====
            if _is_translation_cancelled(self):
                self._log(f"⏹ Recognition cancelled after detection", "warning")
                print(f"[RECOGNIZE] Cancelled after detection")
                self.update_queue.put(('recognize_button_restore', None))
                return
            
            # Send detection results to main thread to draw boxes
            self.update_queue.put(('detect_results', {
                'image_path': image_path,
                'regions': regions
            }))
        else:
            # Using provided regions from existing rectangles
            print(f"[RECOGNIZE] Using {len(regions)} provided regions from existing rectangles")
        
        # ===== CANCELLATION CHECK: Before OCR =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Recognition cancelled before OCR", "warning")
            print(f"[RECOGNIZE] Cancelled before OCR")
            self.update_queue.put(('recognize_button_restore', None))
            return
        
        # Run OCR on regions using the reusable helper method
        self._log(f"🔍 Running OCR on full image and matching to {len(regions)} regions...", "info")
        recognized_texts = _run_ocr_on_regions(self, image_path, regions, ocr_config)
        
        if not recognized_texts or len(recognized_texts) == 0:
            self._log("⚠️ No text recognized in any regions", "warning")
            self.update_queue.put(('recognize_button_restore', None))
            return
        
        # ===== CANCELLATION CHECK: After OCR, before sending results =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Recognition cancelled - discarding results", "warning")
            print(f"[RECOGNIZE] Cancelled after OCR - NOT sending results")
            self.update_queue.put(('recognize_button_restore', None))
            return
        
        # Send results to main thread
        print(f"[DEBUG] Sending {len(recognized_texts)} recognition results to main thread")
        self.update_queue.put(('recognize_results', {
            'image_path': image_path,
            'recognized_texts': recognized_texts
        }))
        
        self._log(f"✅ Text recognition complete! Found text in {len(recognized_texts)}/{len(regions)} regions", "success")
        print(f"[DEBUG] Recognition background thread completed successfully")
        
    except Exception as e:
        import traceback
        self._log(f"❌ Background recognition failed: {str(e)}", "error")
        print(f"Background recognize error traceback: {traceback.format_exc()}")
    finally:
        # Always restore the button
        self.update_queue.put(('recognize_button_restore', None))

def _restore_recognize_button(self):
    """Restore the recognize button to its original state"""
    try:
        # Remove processing overlay effect for the image that was being processed
        # For recognize, track the image path from the operation
        image_path = None
        if hasattr(self, '_recognized_texts_image_path'):
            image_path = self._recognized_texts_image_path
        elif hasattr(self, 'image_preview_widget'):
            image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        _remove_processing_overlay(self, image_path)
        
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'recognize_btn'):
            self.image_preview_widget.recognize_btn.setEnabled(True)
            self.image_preview_widget.recognize_btn.setText("Recognize Text")
    except Exception:
        pass

def _on_translate_text_clicked(self):
    """Translate recognized text using the selected API - runs full pipeline if needed"""
    self._log("🐛 Translate button clicked - starting translation", "info")
    
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    # This MUST happen on the main thread BEFORE any cancellation checks
    _reset_cancellation_flags(self)
    
    try:
        # GUARD: Prevent processing during rendering
        if hasattr(self, '_rendering_in_progress') and self._rendering_in_progress:
            return
        
        # Check if we have an image loaded
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            self._log("⚠️ No image loaded for translation", "warning")
            return
        
        # STEP 1: Check if we have rectangles (detection done)
        has_rectangles = (hasattr(self.image_preview_widget, 'viewer') and 
                        self.image_preview_widget.viewer.rectangles and 
                        len(self.image_preview_widget.viewer.rectangles) > 0)
        
        if not has_rectangles:
            # No rectangles - need to run detection first
            self._log("🔍 No text regions found - running automatic detection and recognition...", "info")
            # Clear any stale recognized texts
            if hasattr(self, '_recognized_texts'):
                del self._recognized_texts
        
        # STEP 2: Check if we have recognized text
        has_recognized_text = (hasattr(self, '_recognized_texts') and 
                              self._recognized_texts and 
                              len(self._recognized_texts) > 0)
        
        if not has_recognized_text:
            if has_rectangles:
                # Have rectangles but no recognized text - need to run recognition only
                self._log("📝 Text regions found but not recognized - running OCR...", "info")
            # If no rectangles, we already logged the message above
            # In both cases, we need to run recognition (which will detect if needed)
        
        # Get current image path first
        image_path = self.image_preview_widget.current_image_path
        
        # Track which image we're translating for overlay removal
        self._translating_image_path = image_path
        
        # Disable ALL workflow buttons to prevent concurrent operations
        _disable_workflow_buttons(self, exclude=None)
        
        # Update translate button text to show progress
        if hasattr(self.image_preview_widget, 'translate_btn'):
            self.image_preview_widget.translate_btn.setText("Translating...")
        
        # Disable thumbnail list to prevent user from switching images during translation
        if hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(False)
            print(f"[TRANSLATE] Disabled thumbnail list during translation")
        
        # Add processing overlay effect (after tracking image)
        _add_processing_overlay(self, )

        # Invalidate stale recognized text from another image
        try:
            if has_recognized_text and getattr(self, '_recognized_texts_image_path', None) != image_path:
                has_recognized_text = False
        except Exception:
            pass
        
        # STEP 3: Prepare regions for recognition if needed
        regions_for_recognition = None
        if has_rectangles and not has_recognized_text:
            # Extract existing rectangles for recognition
            regions_for_recognition = _extract_regions_from_preview(self, )
            # Replace viewer rectangles with merged regions so indices align with recognition
            try:
                self._current_regions = regions_for_recognition
                if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                    self.image_preview_widget.viewer.clear_rectangles()
                _draw_detection_boxes_on_preview(self, )
            except Exception:
                pass
            # Persist merged regions to state ONLY if no OCR data exists yet
            try:
                if hasattr(self, 'image_state_manager'):
                    current_state = self.image_state_manager.get_state(image_path)
                    if not current_state.get('recognized_texts') and not current_state.get('translated_texts'):
                        self.image_state_manager.update_state(image_path, {'detection_regions': regions_for_recognition})
                    else:
                        print(f"[STATE] Skipping detection_regions save - OCR/translation data exists")
            except Exception:
                pass
        elif not has_rectangles:
            # No rectangles - will trigger detection in background
            regions_for_recognition = None
        
        # STEP 4: Start translation workflow
        if has_recognized_text:
            # Already have recognized text - proceed directly to translation
            self._log(f"🌍 Starting translation of {len(self._recognized_texts)} text regions", "info")
            
            import threading
            thread = threading.Thread(target=_run_translate_background, args=(self, self._recognized_texts.copy(), image_path),
                                    daemon=True)
            thread.start()
        else:
            # Need to run detection/recognition first, then translate
            self._log("🚀 Running full translation pipeline...", "info")
            
            import threading
            thread = threading.Thread(target=_run_full_translate_pipeline, args=(self, image_path, regions_for_recognition),
                                    daemon=True)
            thread.start()
        
    except Exception as e:
        import traceback
        self._log(f"❌ Translate setup failed: {str(e)}", "error")
        print(f"Translate setup error traceback: {traceback.format_exc()}")
        _restore_translate_button(self, )

def _run_full_translate_pipeline(self, image_path: str, regions: list):
    """Run full translation pipeline: detect (if needed) -> recognize -> translate
    
    Args:
        image_path: Path to the image
        regions: List of region dicts (if provided), or None to trigger detection
    """
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    _reset_cancellation_flags(self)
    
    print(f"[FULL_PIPELINE] Starting full translation pipeline")
    try:
        import cv2
        
        # STEP 1: Detection (if needed)
        if regions is None:
            print("[FULL_PIPELINE] Running detection...")
            self._log("🔍 Step 1/3: Detecting text regions...", "info")
            
            detection_config = _get_detection_config(self, )
            # Exclude empty bubble container rectangles in translate pipeline to avoid doubles
            if detection_config.get('detect_empty_bubbles', True):
                detection_config['detect_empty_bubbles'] = False
                self._log("🚫 Translate pipeline: Excluding empty bubble regions (container boxes)", "info")
            
            regions = _run_detection_sync(self, image_path, detection_config)
            
            if not regions or len(regions) == 0:
                self._log("⚠️ No text regions detected - cannot translate", "warning")
                self.update_queue.put(('translate_button_restore', None))
                return
            
            print(f"[FULL_PIPELINE] Detection found {len(regions)} regions")
            self._log(f"✅ Detected {len(regions)} text regions", "success")
            
            # Send detection results to main thread to draw boxes
            self.update_queue.put(('detect_results', {
                'image_path': image_path,
                'regions': regions
            }))
        else:
            print(f"[FULL_PIPELINE] Using {len(regions)} provided regions (skipping detection)")
        
        # STEP 2: Recognition
        print("[FULL_PIPELINE] Running recognition...")
        self._log("📝 Step 2/3: Recognizing text in regions...", "info")
        
        # Get OCR config
        ocr_config = _get_ocr_config(self, )
        
        # Run OCR using the same robust method as _run_recognize_background
        # This will run OCR on full image and match to regions
        recognized_texts = _run_ocr_on_regions(self, image_path, regions, ocr_config)
        
        if not recognized_texts or len(recognized_texts) == 0:
            self._log("⚠️ No text recognized - cannot translate", "warning")
            self.update_queue.put(('translate_button_restore', None))
            return
        
        print(f"[FULL_PIPELINE] Recognition found text in {len(recognized_texts)}/{len(regions)} regions")
        self._log(f"✅ Recognized text in {len(recognized_texts)} regions", "success")
        
        # Store recognized texts for potential manual edits
        self.update_queue.put(('recognize_results', {
            'image_path': image_path,
            'recognized_texts': recognized_texts
        }))
        
        # STEP 3: Translation (reuse existing translation logic)
        print("[FULL_PIPELINE] Running translation...")
        self._log("🌍 Step 3/3: Translating text...", "info")
        
        # Call the existing translation logic
        _run_translate_background(self, recognized_texts, image_path)
        
    except Exception as e:
        import traceback
        self._log(f"❌ Full pipeline failed: {str(e)}", "error")
        print(f"[FULL_PIPELINE] Error traceback: {traceback.format_exc()}")
        self.update_queue.put(('translate_button_restore', None))

def _run_translate_background(self, recognized_texts: list, image_path: str):
    """Run translation in background thread with concurrent inpainting and translation"""
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    _reset_cancellation_flags(self)
    
    # Track inpaint thread at function scope so finally block can wait for it
    inpaint_thread = None
    
    try:
        import threading
        
        # CONCURRENT EXECUTION: Start both inpainting and translation in parallel
        # Translation will start immediately while inpainting runs in background
        
        # Shared state for inpainting result
        inpaint_result = {'cleaned_path': None, 'completed': False}
        inpaint_lock = threading.Lock()
        
        def run_inpainting_concurrent():
            """Run inpainting in parallel with translation"""
            try:
                self._log(f"🧽 Running automatic inpainting (concurrent)...", "info")
                regions = []
                for text_data in recognized_texts:
                    bbox = text_data['bbox']
                    region_dict = {
                        'bbox': bbox,
                        'coords': [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]],
                        'confidence': text_data.get('confidence', 1.0),
                        'shape': 'ellipse' if getattr(self, '_use_circle_shapes', False) else 'rect'
                    }
                    regions.append(region_dict)

                cleaned_image_path = _run_inpainting_sync(self, image_path, regions)

                with inpaint_lock:
                    if cleaned_image_path and os.path.exists(cleaned_image_path):
                        print(f"[TRANSLATE_CONCURRENT] Inpainting successful: {os.path.basename(cleaned_image_path)}")
                        self._log(f"✅ Inpainting complete!", "success")
                        self._cleaned_image_path = cleaned_image_path
                        inpaint_result['cleaned_path'] = cleaned_image_path
                        # Show cleaned image in Output tab (no auto switch)
                        self.update_queue.put(('preview_update', {
                            'translated_path': cleaned_image_path,
                            'source_path': image_path
                        }))
                    else:
                        print(f"[TRANSLATE_CONCURRENT] Inpainting failed or returned no path")
                        self._log(f"⚠️ Inpainting failed, using original image", "warning")
                    inpaint_result['completed'] = True
            except Exception as e:
                print(f"[TRANSLATE_CONCURRENT] Inpainting error: {e}")
                import traceback
                traceback.print_exc()
                with inpaint_lock:
                    inpaint_result['completed'] = True
        
        # Start inpainting in background thread (non-blocking)
        inpaint_thread = threading.Thread(target=run_inpainting_concurrent, daemon=True)
        inpaint_thread.start()
        print(f"[TRANSLATE_CONCURRENT] Started inpainting in parallel thread")
        
        # ===== CANCELLATION CHECK: Before starting translation =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Translation cancelled before starting", "warning")
            print(f"[TRANSLATE_CONCURRENT] Cancelled before starting translation")
            return
        
        # STEP 1: Translation (runs immediately without waiting for inpainting)
        full_page_context_enabled = False
        if hasattr(self, '_batch_full_page_context_enabled'):
            full_page_context_enabled = bool(self._batch_full_page_context_enabled)
        else:
            try:
                full_page_context_enabled = bool(self.main_gui.config.get('manga_full_page_context', False))
            except Exception:
                full_page_context_enabled = False


        if full_page_context_enabled:
            self._log(f"📄 Using full page context translation for {len(recognized_texts)} regions", "info")
            translated_texts = _translate_with_full_page_context(self, recognized_texts, image_path)
        else:
            self._log(f"📝 Using individual translation for {len(recognized_texts)} regions", "info")
            translated_texts = _translate_individually(self, recognized_texts, image_path)
        
        print(f"[TRANSLATE_CONCURRENT] Translation completed, checking inpainting status...")
        
        # ===== CANCELLATION CHECK: After translation, before rendering =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Translation cancelled - discarding results", "warning")
            print(f"[TRANSLATE_CONCURRENT] Cancelled after translation - NOT sending results")
            return
        
        # Check if translation returned empty (cancelled inside translate function)
        if not translated_texts:
            self._log(f"⏹ Translation returned no results (likely cancelled)", "warning")
            print(f"[TRANSLATE_CONCURRENT] No translation results - skipping render")
            return
        
        # Wait for inpainting to complete (if not already done) to determine render path
        # This doesn't block the translation API calls - those already happened above
        inpaint_thread.join(timeout=30)  # Wait up to 30 seconds for inpainting
        
        with inpaint_lock:
            cleaned_image_path = inpaint_result.get('cleaned_path')
            if not inpaint_result['completed']:
                print(f"[TRANSLATE_CONCURRENT] Inpainting still running after translation, will use original image")
                self._log(f"⏱️ Inpainting still running, rendering on original image", "info")
        
        # ===== CANCELLATION CHECK: Final gate before sending results =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Translation cancelled - NOT rendering results", "warning")
            print(f"[TRANSLATE_CONCURRENT] Cancelled at final gate - NOT sending translate_results")
            return
        
        # Send results to main thread with render image path
        render_image_path = cleaned_image_path if cleaned_image_path else image_path
        self.update_queue.put(('translate_results', {
            'translated_texts': translated_texts,
            'image_path': render_image_path,  # Render on cleaned if available
            'original_image_path': image_path  # For state mapping
        }))
        
        self._log(f"✅ Translation complete! Translated {len(translated_texts)} text regions", "success")
        
    except Exception as e:
        import traceback
        self._log(f"❌ Background translation failed: {str(e)}", "error")
        print(f"Background translate error traceback: {traceback.format_exc()}")
    finally:
        # Wait for inpainting thread to complete before restoring buttons
        # This prevents "Stopping..." from clearing while inpainting is still running
        if inpaint_thread is not None and inpaint_thread.is_alive():
            print(f"[TRANSLATE_CONCURRENT] Waiting for inpainting thread to complete before restoring buttons...")
            inpaint_thread.join(timeout=60)  # Wait up to 60 seconds
            print(f"[TRANSLATE_CONCURRENT] Inpainting thread completed (or timed out)")
        
        # Always restore the button
        self.update_queue.put(('translate_button_restore', None))

def _translate_with_full_page_context(self, recognized_texts: list, image_path: str) -> list:
    """Translate all texts using full page context like the regular pipeline"""
    try:
        from manga_translator import TextRegion
        
        # Convert recognized texts to TextRegion objects
        regions = []
        for i, text_data in enumerate(recognized_texts):
            bbox = text_data['bbox']
            # Convert bbox from (x, y, w, h) to vertices for TextRegion
            x, y, w, h = bbox
            vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            
            region = TextRegion(
                text=text_data['text'],
                vertices=vertices,
                bounding_box=(x, y, w, h),
                confidence=text_data.get('confidence', 1.0),
                region_type='text_block'
            )
            regions.append(region)
            print(f"[DEBUG] Created TextRegion {i+1}: '{text_data['text'][:30]}...' at {bbox}")
        
        # Get or create MangaTranslator instance
        if not hasattr(self, '_manga_translator') or self._manga_translator is None:
            from manga_translator import MangaTranslator
            from unified_api_client import UnifiedClient
            import os, json, hashlib
            
            # Get OCR config (required by MangaTranslator)
            ocr_config = _get_ocr_config(self, )
            
            # Create UnifiedClient (required by MangaTranslator) - same method as regular translation
            # Get API key - support both PySide6 and Tkinter
            api_key = None
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
            
            # Get model - support both PySide6 and Tkinter (same as regular translation)
            model = 'gpt-4o-mini'  # default
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
                    model = 'gpt-4o-mini'  # fallback
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
                model = self.main_gui.config.get('model')
            
            if not api_key:
                raise ValueError("No API key found in main GUI - cannot create MangaTranslator")
            
            print(f"[DEBUG] Full page context using API key: {'*' * min(8, len(api_key))}... (model: {model})")
            
            # CRITICAL: Apply ALL environment variables exactly like Start Translation
            try:
                env_vars = self.main_gui._get_environment_variables(
                    epub_path='',  # Not needed for manga
                    api_key=api_key
                )
                # Apply ALL environment variables (excluding SYSTEM_PROMPT for OCR)
                for key, value in env_vars.items():
                    if key == 'SYSTEM_PROMPT':
                        continue  # Don't set translation prompt for OCR
                    os.environ[key] = str(value)
                print(f"[DEBUG] Applied {len(env_vars)} environment variables from main GUI exactly like Start Translation")
                # Clear any cached manga translator instance since environment changed
                if hasattr(self, '_manga_translator'):
                    self._manga_translator = None
                    print(f"[DEBUG] Cleared MangaTranslator cache due to environment changes")
            except Exception as env_err:
                print(f"[DEBUG] Failed to apply GUI environment variables: {env_err}")
            
            # Apply multi-key env from GUI settings so UnifiedClient picks it up
            try:
                use_mk = bool(self.main_gui.config.get('use_multi_api_keys', False))
                mk_list = self.main_gui.config.get('multi_api_keys', []) or []
                force_rotation = bool(self.main_gui.config.get('force_key_rotation', True))
                rotation_frequency = int(self.main_gui.config.get('rotation_frequency', 1))
                if use_mk and mk_list:
                    os.environ['USE_MULTI_API_KEYS'] = '1'
                    os.environ['USE_MULTI_KEYS'] = '1'
                    os.environ['FORCE_KEY_ROTATION'] = '1' if force_rotation else '0'
                    os.environ['ROTATION_FREQUENCY'] = str(rotation_frequency)

                    # Avoid Windows env var length limit by keeping keys in memory
                    try:
                        UnifiedClient.set_in_memory_multi_keys(
                            mk_list,
                            force_rotation=force_rotation,
                            rotation_frequency=rotation_frequency,
                        )
                    except Exception:
                        pass
                else:
                    os.environ['USE_MULTI_API_KEYS'] = '0'
                    os.environ['USE_MULTI_KEYS'] = '0'
                    try:
                        UnifiedClient.clear_in_memory_multi_keys()
                    except Exception:
                        pass
            except Exception as _mk_err:
                print(f"[DEBUG] Failed to apply multi-key env: {_mk_err}")
            
            unified_client = UnifiedClient(model=model, api_key=api_key)
            # If multi-key desired, (re)setup pool to ensure keys are loaded for this session
            try:
                if os.getenv('USE_MULTI_API_KEYS', '0') == '1' and mk_list:
                    UnifiedClient.setup_multi_key_pool(mk_list, force_rotation=force_rotation, rotation_frequency=rotation_frequency)
            except Exception as _pool_err:
                print(f"[DEBUG] setup_multi_key_pool failed: {_pool_err}")
            
            # Create MangaTranslator with all required parameters
            self._manga_translator = MangaTranslator(
                ocr_config=ocr_config,
                unified_client=unified_client,
                main_gui=self.main_gui,
                log_callback=self._log,
                skip_inpainter_init=True  # Full page context - translation only, no inpainting
            )
            print(f"[DEBUG] Created MangaTranslator instance for full page context")
        
        # ===== CANCELLATION CHECK: Before calling full page context translation =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Translation cancelled before full page context", "warning")
            print(f"[TRANSLATE_FULL_PAGE] Cancelled before translate_full_page_context call")
            return []
        
        # Use the MangaTranslator's full page context method
        print(f"[DEBUG] Calling translate_full_page_context for {len(regions)} regions")
        self._log(f"🌍 Starting full page context translation...", "info")
        
        translations_dict = self._manga_translator.translate_full_page_context(regions, image_path)
        print(f"[DEBUG] Got translations dict: {list(translations_dict.keys()) if translations_dict else 'None'}")
        
        # ===== CANCELLATION CHECK: After getting translation results =====
        if _is_translation_cancelled(self):
            self._log(f"⏹ Translation cancelled - discarding full page context results", "warning")
            print(f"[TRANSLATE_FULL_PAGE] Cancelled after translate_full_page_context - returning empty")
            return []
        
        # Convert the results back to the expected format
        translated_texts = []
        for i, (region, text_data) in enumerate(zip(regions, recognized_texts)):
            if hasattr(region, 'translated_text') and region.translated_text:
                translation = region.translated_text
                print(f"[DEBUG] Region {i+1} translated: '{region.text[:20]}...' -> '{translation[:20]}...'")
            else:
                translation = text_data['text']  # Fallback to original
                print(f"[DEBUG] Region {i+1} no translation, using original: '{translation[:20]}...'")
            
            translated_texts.append({
                'original': text_data,
                'translation': translation,
                'bbox': text_data['bbox']
            })
        
        self._log(f"✅ Full page context translation complete: {len(translated_texts)} regions", "success")
        return translated_texts
        
    except Exception as e:
        import traceback
        self._log(f"❌ Full page context translation failed: {str(e)}", "error")
        print(f"[DEBUG] Full page context error traceback: {traceback.format_exc()}")
        # Fallback to original texts
        return [{
            'original': text_data,
            'translation': f"[Full Page Context Error: {str(e)}]",
            'bbox': text_data['bbox']
        } for text_data in recognized_texts]

def _translate_individually(self, recognized_texts: list, image_path: str) -> list:
    """Translate each text individually (original behavior)"""
    # CRITICAL: Import these at function start to avoid UnboundLocalError in except blocks
    import os
    import json
    import hashlib
    import traceback
    from unified_api_client import UnifiedClient
    
    try:
        # Check if visual context is enabled (SAFE for background thread)
        # Prefer batch snapshot captured on UI thread, else fall back to config
        include_page_image = False
        if hasattr(self, '_batch_visual_context_enabled'):
            include_page_image = bool(self._batch_visual_context_enabled)
            print(f"[DEBUG] Visual context (batch snapshot): {include_page_image}")
        else:
            try:
                include_page_image = bool(self.main_gui.config.get('manga_visual_context_enabled', False))
            except Exception:
                include_page_image = False
            print(f"[DEBUG] Visual context (from config): {include_page_image}")
        
        print(f"[DEBUG] Visual context enabled: {include_page_image}")
        print(f"[DEBUG] Image path: {image_path}")
        
        # Get API key and model from main GUI once (same method as regular translation)
        # (imports already done at top of function)
        
        # Get API key - support both PySide6 and Tkinter
        api_key = None
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
        
        # Get model - support both PySide6 and Tkinter (same as regular translation)
        model = 'gpt-4o-mini'  # default
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
                model = 'gpt-4o-mini'  # fallback
        elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
            model = self.main_gui.config.get('model')
        
        if not api_key:
            raise ValueError("No API key found in main GUI")
        
        print(f"[DEBUG] Using API key: {'*' * min(8, len(api_key))}... (model: {model})")
        
        # STEP 1: Manually ensure critical GUI variables are set FIRST
        # This ensures the most important values are definitely applied
        try:
            # Send interval / delay
            if hasattr(self.main_gui, 'delay_entry'):
                if hasattr(self.main_gui.delay_entry, 'text'):
                    delay = self.main_gui.delay_entry.text()
                elif hasattr(self.main_gui.delay_entry, 'get'):
                    delay = self.main_gui.delay_entry.get()
                else:
                    delay = '1.0'
                os.environ['SEND_INTERVAL_SECONDS'] = str(delay)
                print(f"[DEBUG] Set SEND_INTERVAL_SECONDS: {delay}")
            
            # Max output tokens
            if hasattr(self.main_gui, 'max_output_tokens'):
                os.environ['MAX_OUTPUT_TOKENS'] = str(self.main_gui.max_output_tokens)
                print(f"[DEBUG] Set MAX_OUTPUT_TOKENS: {self.main_gui.max_output_tokens}")
            
            # Batch translation settings
            if hasattr(self.main_gui, 'batch_translation_var'):
                batch_enabled = '1' if self.main_gui.batch_translation_var else '0'
                os.environ['BATCH_TRANSLATION'] = batch_enabled
                print(f"[DEBUG] Set BATCH_TRANSLATION: {batch_enabled}")
            
            if hasattr(self.main_gui, 'batch_size_var'):
                os.environ['BATCH_SIZE'] = str(self.main_gui.batch_size_var)
                print(f"[DEBUG] Set BATCH_SIZE: {self.main_gui.batch_size_var}")
            
            # Batching mode + group size (with tkinter / PySide detection)
            if hasattr(self.main_gui, 'batch_mode_var'):
                try:
                    val = self.main_gui.batch_mode_var.get() if hasattr(self.main_gui.batch_mode_var, 'get') else self.main_gui.batch_mode_var
                    val_str = str(val).strip().lower() if val else 'aggressive'
                except Exception:
                    val_str = 'aggressive'
                os.environ['BATCHING_MODE'] = val_str
                # Backward compatibility
                os.environ['CONSERVATIVE_BATCHING'] = '1' if val_str == 'conservative' else '0'
                print(f"[DEBUG] Set BATCHING_MODE: {val_str}")
            if hasattr(self.main_gui, 'batch_group_size_var'):
                try:
                    gval = self.main_gui.batch_group_size_var.get() if hasattr(self.main_gui.batch_group_size_var, 'get') else self.main_gui.batch_group_size_var
                    gval_int = int(gval) if str(gval).strip() else 3
                except Exception:
                    gval_int = 3
                os.environ['BATCH_GROUP_SIZE'] = str(max(1, gval_int))
                print(f"[DEBUG] Set BATCH_GROUP_SIZE: {gval_int}")
            
            # Temperature
            if hasattr(self.main_gui, 'trans_temp'):
                if hasattr(self.main_gui.trans_temp, 'text'):
                    temp = self.main_gui.trans_temp.text()
                elif hasattr(self.main_gui.trans_temp, 'get'):
                    temp = self.main_gui.trans_temp.get()
                else:
                    temp = '0.3'
                os.environ['TRANSLATION_TEMPERATURE'] = str(temp)
                print(f"[DEBUG] Set TRANSLATION_TEMPERATURE: {temp}")
            
            # Translation history limit
            if hasattr(self.main_gui, 'trans_history'):
                if hasattr(self.main_gui.trans_history, 'text'):
                    hist = self.main_gui.trans_history.text()
                elif hasattr(self.main_gui.trans_history, 'get'):
                    hist = self.main_gui.trans_history.get()
                else:
                    hist = '3'
                os.environ['TRANSLATION_HISTORY_LIMIT'] = str(hist)
                print(f"[DEBUG] Set TRANSLATION_HISTORY_LIMIT: {hist}")
            
            print(f"[DEBUG] Manually set critical GUI variables for individual translate")
            
        except Exception as manual_env_err:
            print(f"[DEBUG] Failed to set manual GUI variables: {manual_env_err}")
        
        # STEP 2: Apply all other environment variables from main GUI
        # This ensures all GUI settings are respected
        try:
            if hasattr(self.main_gui, '_get_environment_variables'):
                env_vars = self.main_gui._get_environment_variables(
                    epub_path='',  # Not needed for manga
                    api_key=api_key
                )
                # Apply all environment variables
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
                print(f"[DEBUG] Applied {len(env_vars)} environment variables from main GUI")
        except Exception as env_err:
            print(f"[DEBUG] Failed to apply GUI environment variables: {env_err}")
        
        # Apply multi-key env from GUI settings so UnifiedClient picks it up
        use_mk = False
        mk_list = []
        force_rotation = True
        rotation_frequency = 1
        try:
            use_mk = bool(self.main_gui.config.get('use_multi_api_keys', False))
            mk_list = self.main_gui.config.get('multi_api_keys', []) or []
            force_rotation = bool(self.main_gui.config.get('force_key_rotation', True))
            rotation_frequency = int(self.main_gui.config.get('rotation_frequency', 1))
            if use_mk and mk_list:
                os.environ['USE_MULTI_API_KEYS'] = '1'
                os.environ['USE_MULTI_KEYS'] = '1'
                os.environ['FORCE_KEY_ROTATION'] = '1' if force_rotation else '0'
                os.environ['ROTATION_FREQUENCY'] = str(rotation_frequency)

                # Avoid Windows env var length limit by keeping keys in memory
                try:
                    UnifiedClient.set_in_memory_multi_keys(
                        mk_list,
                        force_rotation=force_rotation,
                        rotation_frequency=rotation_frequency,
                    )
                except Exception:
                    pass
            else:
                os.environ['USE_MULTI_API_KEYS'] = '0'
                os.environ['USE_MULTI_KEYS'] = '0'
                try:
                    UnifiedClient.clear_in_memory_multi_keys()
                except Exception:
                    pass
        except Exception as _mk_err:
            print(f"[DEBUG] Failed to apply multi-key env: {_mk_err}")
        
        # Create fresh UnifiedClient (no caching to avoid environment variable conflicts)
        print(f"[DEBUG] Creating fresh UnifiedClient with model: {model} (multi_key={use_mk})")
        client = UnifiedClient(model=model, api_key=api_key)
        # If multi-key desired, ensure pool is initialized/refreshed
        try:
            if use_mk and mk_list:
                UnifiedClient.setup_multi_key_pool(mk_list, force_rotation=force_rotation, rotation_frequency=rotation_frequency)
        except Exception as _pool_err:
            print(f"[DEBUG] setup_multi_key_pool failed: {_pool_err}")
        
        # Get system prompt from GUI profile (same as regular pipeline)
        system_prompt = _get_system_prompt_from_gui(self, )
        if not system_prompt:
            raise ValueError("No system prompt configured in GUI profile - translation cannot proceed")
        
        # Preload image data once if visual context is enabled
        image_base64 = None
        if include_page_image and image_path and os.path.exists(image_path):
            print(f"[DEBUG] Preloading image data for visual context...")
            try:
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                import base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                print(f"[DEBUG] Image data preloaded: {len(image_base64)} bytes (base64)")
            except Exception as img_error:
                print(f"[DEBUG] Failed to preload image: {str(img_error)}")
                image_base64 = None
        
        translated_texts = []
        
        # CRITICAL: Apply ALL environment variables exactly like Start Translation
        try:
            env_vars = self.main_gui._get_environment_variables(
                epub_path='',  # Not needed for manga
                api_key=api_key
            )
            # Apply ALL environment variables
            for key, value in env_vars.items():
                os.environ[key] = str(value)
            print(f"[DEBUG] Applied {len(env_vars)} environment variables from main GUI exactly like Start Translation")
        except Exception as env_err:
            print(f"[DEBUG] Failed to apply GUI environment variables: {env_err}")
        
        # Read parameters from environment variables (now set from GUI)
        temperature = float(os.environ.get('TRANSLATION_TEMPERATURE', '0.3'))
        
        # Check for manga-specific output token limit first, fallback to environment/GUI limit
        default_max_tokens = int(os.environ.get('MAX_OUTPUT_TOKENS', '4000'))
        manga_token_limit = -1
        try:
            manga_settings = self.main_gui.config.get('manga_settings', {}) or {}
            manual_edit = manga_settings.get('manual_edit', {}) or {}
            manga_token_limit = int(manual_edit.get('manga_output_token_limit', -1))
        except Exception:
            manga_token_limit = -1
        
        # If manga token limit is > 0, use it; otherwise use default from environment
        if manga_token_limit > 0:
            max_tokens = manga_token_limit
            print(f"[DEBUG] Using manga-specific output token limit: {max_tokens}")
        else:
            max_tokens = default_max_tokens
            print(f"[DEBUG] Using main GUI output token limit: {max_tokens}")
        
        print(f"[DEBUG] Using parameters from environment (set from GUI): temperature={temperature}, max_tokens={max_tokens}")
        print(f"[DEBUG] Processing {len(recognized_texts)} recognized texts for individual translation")
        for i, text_data in enumerate(recognized_texts):
            # ===== CANCELLATION CHECK: At start of each text =====
            if _is_translation_cancelled(self):
                self._log(f"⏹ Translation cancelled at text {i+1}/{len(recognized_texts)}", "warning")
                print(f"[TRANSLATE_INDIVIDUAL] Cancelled at start of text {i+1}")
                # Return empty list to signal cancellation - no partial results
                return []
            
            text = text_data['text']
            print(f"[DEBUG] Translating text {i+1}/{len(recognized_texts)}: '{text[:30]}...'")
            
            try:
                # Prepare translation request
                prompt = text
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                if image_base64:
                    # Visual context translation
                    print(f"[DEBUG] Using visual context for translation")
                    self._log(f"🖼️ Translating with visual context: '{text[:50]}...'", "info")
                    print(f"[DEBUG] Calling client.send_image() with GUI settings (temp={temperature}, max_tokens={max_tokens})...")
                    response = client.send_image(messages, image_base64, temperature=temperature, max_tokens=max_tokens)
                    print(f"[DEBUG] Got response: {response[:100] if response else 'None'}...")
                else:
                    # Text-only translation
                    print(f"[DEBUG] Using text-only translation")
                    self._log(f"📝 Translating text: '{text[:50]}...'", "info")
                    print(f"[DEBUG] Calling client.send() with GUI settings (temp={temperature}, max_tokens={max_tokens})...")
                    response = client.send(messages, temperature=temperature, max_tokens=max_tokens)
                    print(f"[DEBUG] Got response: {response[:100] if response else 'None'}...")
                
                # Extract translated text from response (UnifiedClient returns tuple or response object)
                if hasattr(response, 'content'):
                    translated_text = response.content
                elif isinstance(response, tuple) and len(response) >= 1:
                    translated_text = response[0]  # (content, finish_reason)
                else:
                    translated_text = str(response)
                
                print(f"[DEBUG] Processed response: '{translated_text[:50]}...'")
                
                # ===== CANCELLATION CHECK: After getting response (prevent raw text return) =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled after response - discarding all results", "warning")
                    print(f"[TRANSLATE_INDIVIDUAL] Cancelled after getting response for text {i+1} - returning empty")
                    return []
                
                translated_texts.append({
                    'original': text_data,
                    'translation': translated_text.strip(),
                    'bbox': text_data['bbox']
                })
                
                self._log(f"✅ Translated: '{text}' → '{translated_text.strip()}'", "success")
                print(f"[DEBUG] Successfully translated text {i+1}")
                
            except Exception as e:
                # traceback already imported at top of function
                error_msg = f"Translation failed for '{text}': {str(e)}"
                self._log(f"❌ {error_msg}", "error")
                print(f"[DEBUG] {error_msg}")
                print(f"[DEBUG] Translation error traceback: {traceback.format_exc()}")
                translated_texts.append({
                    'original': text_data,
                    'translation': f"[Translation Error: {str(e)}]",
                    'bbox': text_data['bbox']
                })
        
        return translated_texts
        
    except Exception as e:
        # traceback already imported at top of function
        self._log(f"❌ Individual translation failed: {str(e)}", "error")
        print(f"[DEBUG] Individual translation error traceback: {traceback.format_exc()}")
        # Fallback to original texts
        return [{
            'original': text_data,
            'translation': f"[Individual Translation Error: {str(e)}]",
            'bbox': text_data['bbox']
        } for text_data in recognized_texts]

def _get_system_prompt_from_gui(self) -> str:
    """Get system prompt from GUI profile (same as regular pipeline) - fails if no prompt found"""
    try:
        # Get profile name from GUI (support both Tkinter and PySide6)
        profile_name = 'Default'
        try:
            if hasattr(self.main_gui, 'profile_var'):
                if hasattr(self.main_gui.profile_var, 'get'):
                    profile_name = self.main_gui.profile_var.get()
                else:
                    profile_name = self.main_gui.profile_var
        except Exception:
            profile_name = 'Default'
        
        # Get the prompt from prompt_profiles dictionary - NO FALLBACKS
        system_prompt = ''
        if hasattr(self.main_gui, 'prompt_profiles') and profile_name in self.main_gui.prompt_profiles:
            system_prompt = self.main_gui.prompt_profiles[profile_name]
            if system_prompt.strip():  # Only accept non-empty prompts
                print(f"[DEBUG] Using system prompt from profile: {profile_name}")
                return system_prompt.strip()
        
        # NO FALLBACKS - fail if no proper prompt found
        print(f"[DEBUG] No valid system prompt found for profile: {profile_name}")
        return ''
        
    except Exception as e:
        print(f"[DEBUG] Error getting system prompt: {str(e)}")
        return ''

def _update_rectangles_with_recognition(self, recognized_texts: list):
    """Update rectangles with proper context menu tooltips for OCR'd text"""
    try:
        if not hasattr(self, 'image_preview_widget') or not hasattr(self.image_preview_widget, 'viewer'):
            return
        
        rectangles = self.image_preview_widget.viewer.rectangles
        print(f"[DEBUG] Adding OCR tooltips to {len(rectangles)} rectangles with {len(recognized_texts)} recognition results")
        
        # Store recognition data for context menu access
        self._recognition_data = {}

        # Helper: compute IoU between two (x,y,w,h) boxes
        def _iou_xywh(a, b):
            try:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ax2, ay2 = ax + aw, ay + ah
                bx2, by2 = bx + bw, by + bh
                x1 = max(ax, bx)
                y1 = max(ay, by)
                x2 = min(ax2, bx2)
                y2 = min(ay2, by2)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = max(0, aw) * max(0, ah)
                area_b = max(0, bw) * max(0, bh)
                denom = area_a + area_b - inter
                return (inter / denom) if denom > 0 else 0.0
            except Exception:
                return 0.0
        
        for i, text_data in enumerate(recognized_texts):
            region_index = text_data.get('region_index', i)
            rect_item = None

            # Primary mapping: by index
            if 0 <= region_index < len(rectangles):
                rect_item = rectangles[region_index]
            else:
                # Fallback: spatially match by highest IoU
                bbox = text_data.get('bbox')
                if bbox and len(bbox) >= 4 and rectangles:
                    best_idx = -1
                    best_iou = 0.0
                    for idx, r in enumerate(rectangles):
                        rr = r.sceneBoundingRect()
                        cand = [rr.x(), rr.y(), rr.width(), rr.height()]
                        iou = _iou_xywh(bbox, cand)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_idx != -1 and best_iou > 0.05:
                        rect_item = rectangles[best_idx]
                        region_index = best_idx  # remap to matched rectangle
            
            if rect_item is not None:
                recognized_text = text_data['text']
                
                # Store recognition data for context menu
                self._recognition_data[region_index] = {
                    'text': recognized_text,
                    'bbox': text_data['bbox'],
                    'bubble_type': text_data.get('bubble_type'),
                    'bubble_bounds': text_data.get('bubble_bounds', text_data['bbox'])
                }
                
                # Change rectangle color to BLUE when text is recognized
                from PySide6.QtGui import QPen, QBrush, QColor
                rect_item.setPen(QPen(QColor(0, 150, 255), 2))  # Blue border
                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))  # Semi-transparent blue fill
                # Mark as recognized so selection restore keeps it blue
                try:
                    rect_item.is_recognized = True
                except Exception:
                    pass
                
                # Add context menu support to rectangle
                _add_context_menu_to_rectangle(self, rect_item, region_index)
                # Attach move-sync so moving the rectangle moves the overlay
                try:
                    _attach_move_sync_to_rectangle(self, rect_item, region_index)
                except Exception:
                    pass
                
                print(f"[DEBUG] Added OCR tooltip to rectangle {region_index}: '{recognized_text[:30]}...'")
            else:
                print(f"[DEBUG] Warning: No rectangle match for recognition item {i} (region_index={region_index})")
        
        # Force scene update to show color changes immediately
        try:
            viewer = self.image_preview_widget.viewer
            viewer._scene.update()
            viewer.viewport().update()
            print(f"[DEBUG] Forced scene refresh to show blue rectangles")
        except Exception as e:
            print(f"[DEBUG] Failed to refresh scene: {e}")
        
        print(f"[DEBUG] OCR tooltip setup complete")
        
    except Exception as e:
        print(f"[DEBUG] Error adding OCR tooltips: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")

def _update_rectangles_with_translations(self, translated_texts: list):
    """Add translated text as overlay layer without modifying original image"""
    try:
        if not hasattr(self, 'image_preview_widget') or not hasattr(self.image_preview_widget, 'viewer'):
            return
        
        print(f"[DEBUG] Adding translated text overlay for {len(translated_texts)} regions")
        
        # Store translation data for context menu access
        self._translation_data = {}
        for i, result in enumerate(translated_texts):
            region_index = result['original'].get('region_index', i)
            # Capture bbox for stable remapping via IoU during re-render
            bbox_val = result.get('bbox')
            if not bbox_val and 0 <= region_index < len(self.image_preview_widget.viewer.rectangles):
                rr = self.image_preview_widget.viewer.rectangles[region_index].sceneBoundingRect()
                bbox_val = [int(rr.x()), int(rr.y()), int(rr.width()), int(rr.height())]
            self._translation_data[region_index] = {
                'original': result['original']['text'],
                'translation': result['translation'],
                'bbox': bbox_val
            }
        
        # Add text overlay to the viewer
        _add_text_overlay_to_viewer(self, translated_texts)
        
        self._log(f"✅ Added {len(translated_texts)} translation overlays to image preview", "success")
        
    except Exception as e:
        print(f"[DEBUG] Error adding translation overlays: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Error adding translation overlays: {str(e)}", "error")

def _add_processing_overlay(self):
    """Add a pulsing overlay effect to indicate processing on the active viewer (source or output)."""
    try:
        # ===== CANCELLATION CHECK: Don't add overlay if stop was clicked =====
        if _is_translation_cancelled(self):
            print(f"[OVERLAY] Not adding overlay - stop was clicked")
            return
        
        if not hasattr(self, 'image_preview_widget'):
            return
        
        # Get current image path for per-image overlay tracking
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtCore import QRectF, QPropertyAnimation, QEasingCurve, Qt, QObject, Property
        from PySide6.QtGui import QBrush, QColor
        
        # Initialize per-image overlay storage
        if not hasattr(self, '_processing_overlays_by_image'):
            self._processing_overlays_by_image = {}
        
        # Don't add overlay if one already exists for this image
        if current_image in self._processing_overlays_by_image:
            print(f"[OVERLAY] Processing overlay already exists for {os.path.basename(current_image)}")
            return
        
        # Use source viewer (no more separate output viewer)
        viewer = getattr(self.image_preview_widget, 'viewer', None)
        if viewer is None:
            return
        
        # Create overlay rectangle covering entire scene
        scene_rect = viewer._scene.sceneRect()
        overlay = QGraphicsRectItem(scene_rect)
        overlay.setBrush(QBrush(QColor(0, 150, 255, 30)))  # Blue semi-transparent
        overlay.setPen(Qt.NoPen)
        overlay.setZValue(1000)  # On top of everything
        
        # Add to scene
        viewer._scene.addItem(overlay)
        
        # Create pulsing animation using QObject wrapper
        class OpacityItem(QObject):
            def __init__(self, item, parent=None):
                super().__init__(parent)
                self._item = item
                self._opacity = 30
            
            def get_opacity(self):
                return self._opacity
            
            def set_opacity(self, value):
                self._opacity = value
                self._item.setBrush(QBrush(QColor(0, 150, 255, int(value))))
            
            opacity = Property(int, get_opacity, set_opacity)
        
        opacity_wrapper = OpacityItem(overlay)
        
        pulse_animation = QPropertyAnimation(opacity_wrapper, b"opacity")
        pulse_animation.setDuration(1500)  # 1.5 seconds
        pulse_animation.setStartValue(15)
        pulse_animation.setEndValue(50)
        pulse_animation.setEasingCurve(QEasingCurve.InOutQuad)
        pulse_animation.setLoopCount(-1)  # Infinite loop
        pulse_animation.start()
        
        # Store per-image
        self._processing_overlays_by_image[current_image] = {
            'overlay': overlay,
            'animation': pulse_animation,
            'wrapper': opacity_wrapper,
            'viewer': viewer
        }
        
        print(f"[OVERLAY] Added processing overlay for {os.path.basename(current_image)}")
        
    except Exception as e:
        print(f"[OVERLAY] Error adding processing overlay: {str(e)}")

def _remove_processing_overlay(self, image_path=None, clear_all=False):
    """Remove the processing overlay effect for a specific image, current image, or all images"""
    try:
        if not hasattr(self, '_processing_overlays_by_image'):
            return
        
        # If clear_all or no image_path specified, remove ALL overlays (for batch end)
        if clear_all or image_path is None:
            paths_to_remove = list(self._processing_overlays_by_image.keys())
            for path in paths_to_remove:
                try:
                    overlay_data = self._processing_overlays_by_image[path]
                    
                    # Stop animation
                    try:
                        overlay_data['animation'].stop()
                    except Exception:
                        pass
                    
                    # Remove overlay from scene
                    try:
                        overlay_data['viewer']._scene.removeItem(overlay_data['overlay'])
                    except Exception:
                        pass
                    
                    del self._processing_overlays_by_image[path]
                    print(f"[OVERLAY] Removed processing overlay for {os.path.basename(path)}")
                except Exception:
                    pass
            
            print(f"[OVERLAY] Cleared all processing overlays ({len(paths_to_remove)} total)")
            return
        
        # Remove overlay for specific image path
        if image_path in self._processing_overlays_by_image:
            overlay_data = self._processing_overlays_by_image[image_path]
            
            # Stop animation
            try:
                overlay_data['animation'].stop()
            except Exception:
                pass
            
            # Remove overlay from scene
            try:
                overlay_data['viewer']._scene.removeItem(overlay_data['overlay'])
            except Exception:
                pass
            
            # Clean up references
            del self._processing_overlays_by_image[image_path]
            
            print(f"[OVERLAY] Removed processing overlay for {os.path.basename(image_path)}")
        
    except Exception as e:
        print(f"[OVERLAY] Error removing processing overlay: {str(e)}")


def _handle_ocr_this_text(self, region_index: int, rect_item=None):
    """Handle OCR this text context menu action for any rectangle type"""
    try:
        print(f"[OCR_CONTEXT] Starting OCR for region {region_index}")
        
        # Get current image path
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            self._log("⚠️ No image loaded for OCR", "warning")
            return
        
        image_path = self.image_preview_widget.current_image_path
        
        # Get the rectangle from viewer
        if rect_item is None:
            rectangles = getattr(self.image_preview_widget.viewer, 'rectangles', [])
            if region_index >= len(rectangles):
                self._log(f"⚠️ Rectangle index {region_index} out of range", "warning")
                return
            rect_item = rectangles[region_index]
        
        # Get rectangle bounds
        rect = rect_item.sceneBoundingRect()
        bbox = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
        
        # Create a region dict for OCR processing (reusing existing format)
        region = {
            'bbox': bbox,
            'confidence': 1.0
        }
        
        # Get OCR configuration
        ocr_config = _get_ocr_config(self, )
        print(f"[OCR_CONTEXT] Using OCR provider: {ocr_config['provider']}")
        
        # Add comprehensive logging for OCR operation
        self._log(f"🔍 Starting OCR on region using {ocr_config['provider']}", "info")
        
        # Start pulse effect on the rectangle
        _add_rectangle_pulse_effect(self, rect_item, region_index)
        
        # Run OCR in background thread to avoid GUI lag
        import threading
        
        def ocr_background():
            try:
                # Reuse the existing _run_ocr_on_regions method with a single region
                recognized_texts = _run_ocr_on_regions(self, image_path, [region], ocr_config)
                
                # Emit signal to process results on main thread
                self.ocr_result_signal.emit(recognized_texts, rect_item, region_index, bbox, ocr_config['provider'])
                
            except Exception as e:
                # Emit signal to handle error on main thread
                self.ocr_error_signal.emit(e, rect_item, region_index)
        
        # Start background thread
        threading.Thread(target=ocr_background, daemon=True).start()
            
    except Exception as e:
        # Stop pulse effect on error
        if 'rect_item' in locals() and 'region_index' in locals():
            _remove_rectangle_pulse_effect(self, rect_item, region_index)
        print(f"[OCR_CONTEXT] Error in _handle_ocr_this_text: {e}")
        import traceback
        print(f"[OCR_CONTEXT] Traceback: {traceback.format_exc()}")
        self._log(f"❌ OCR failed: {str(e)}", "error")

def _process_ocr_result(self, recognized_texts, rect_item, region_index, bbox, ocr_provider):
    """Process OCR result on main thread (called via signal)"""
    try:
        # Stop pulse effect regardless of outcome
        _remove_rectangle_pulse_effect(self, rect_item, region_index)
        
        # Log the results
        if recognized_texts and len(recognized_texts) > 0:
            self._log(f"✅ OCR completed successfully using {ocr_provider}", "success")
        else:
            self._log(f"⚠️ OCR found no text using {ocr_provider}", "warning")
        
        if recognized_texts and len(recognized_texts) > 0:
            recognized_text = recognized_texts[0]['text']
            self._log(f"✅ OCR result: {recognized_text}", "success")
            
            # Store recognition data for future use
            if not hasattr(self, '_recognition_data'):
                self._recognition_data = {}
            
            self._recognition_data[region_index] = {
                'text': recognized_text,
                'confidence': recognized_texts[0].get('confidence', 1.0),
                'bbox': bbox
            }
            
            # Also persist OCR result into image_state_manager so it survives image switches
            try:
                image_path = getattr(self.image_preview_widget, 'current_image_path', None)
                if image_path and hasattr(self, 'image_state_manager') and self.image_state_manager:
                    state = self.image_state_manager.get_state(image_path) or {}
                    rec_list = state.get('recognized_texts', [])
                    # Extend list if needed to accommodate this region_index
                    while len(rec_list) <= region_index:
                        rec_list.append({'deleted': True})
                    rec_list[region_index] = {
                        'text': recognized_text,
                        'bbox': bbox,
                        'region_index': region_index
                    }
                    state['recognized_texts'] = rec_list
                    self.image_state_manager.set_state(image_path, state, save=True)
                    print(f"[OCR_CONTEXT] Persisted OCR result to state for region {region_index}")
            except Exception as persist_err:
                print(f"[OCR_CONTEXT] Failed to persist OCR to state: {persist_err}")
            
            # Change rectangle color to blue to indicate it now has recognized text
            from PySide6.QtGui import QPen, QBrush, QColor
            rect_item.setPen(QPen(QColor(0, 150, 255), 2))  # Blue border
            rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))  # Semi-transparent blue fill
            rect_item.is_recognized = True
            rect_item.region_index = region_index
            
            # Add/update context menu for the now-blue rectangle
            _add_context_menu_to_rectangle(self, rect_item, region_index)
            
            print(f"[OCR_CONTEXT] Successfully recognized text in region {region_index}")
            
            # PERSIST: Save updated state so rectangles/OCR survive panel switches and sessions
            try:
                _persist_current_image_state(self)
            except Exception:
                pass
            
        else:
            self._log("⚠️ No text found in selected region", "warning")
            
    except Exception as e:
        print(f"[OCR_CONTEXT] Error processing OCR result: {e}")
        import traceback
        print(f"[OCR_CONTEXT] Traceback: {traceback.format_exc()}")
        self._log(f"❌ OCR result processing failed: {str(e)}", "error")

def _handle_ocr_error(self, error, rect_item, region_index):
    """Handle OCR error on main thread (called via signal)"""
    try:
        # Stop pulse effect on error
        _remove_rectangle_pulse_effect(self, rect_item, region_index)
        print(f"[OCR_CONTEXT] Error in background OCR: {error}")
        import traceback
        print(f"[OCR_CONTEXT] Background OCR error traceback: {traceback.format_exc()}")
        self._log(f"❌ OCR failed: {str(error)}", "error")
    except Exception as e:
        print(f"[OCR_CONTEXT] Error handling OCR error: {e}")

def _add_context_menu_to_rectangle(self, rect_item, region_index: int):
    """Add context menu to rectangle for OCR and translation options on right-click"""
    try:
        from PySide6.QtWidgets import QMenu, QMessageBox
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QAction
        
        # Store the region index on the rectangle for later retrieval
        rect_item.region_index = region_index
        
        # Override the mouse press event to handle right-clicks
        original_mouse_press = rect_item.mousePressEvent
        
        def handle_mouse_press(event):
            try:
                from PySide6.QtCore import Qt
                from PySide6.QtGui import QCursor
                
                if event.button() == Qt.MouseButton.RightButton:
                    # Handle right-click for context menu
                    menu = QMenu()
                    
                    # Get recognition data (fresh lookup for up-to-date text)
                    # Use rect_item.region_index to get the actual stored index for this specific rectangle
                    actual_index = rect_item.region_index
                    
                    # ALWAYS add "OCR this text" option first - available for all rectangles
                    ocr_this_action = QAction("🔍 OCR This Text", menu)
                    def make_ocr_this_handler(idx, rect):
                        return lambda: _handle_ocr_this_text(self, idx, rect)
                    ocr_this_action.triggered.connect(make_ocr_this_handler(actual_index, rect_item))
                    menu.addAction(ocr_this_action)
                    
                    # Add "Edit OCR" option if text already exists
                    if hasattr(self, '_recognition_data') and actual_index in self._recognition_data:
                        recognition_text = self._recognition_data[actual_index]['text']
                        # Add action to show OCR text (with better preview formatting)
                        preview_text = (recognition_text[:22] + "...") if len(recognition_text) > 25 else recognition_text
                        ocr_action = QAction(f"📝 Edit OCR: \"{preview_text}\"", menu)
                        # Create a proper closure by defining a function that captures the current index
                        def make_ocr_handler(idx):
                            return lambda: _show_ocr_popup(self, self._recognition_data[idx]['text'], idx)
                        ocr_action.triggered.connect(make_ocr_handler(actual_index))
                        menu.addAction(ocr_action)
                    
                    # Get translation data if available (fresh lookup)
                    if hasattr(self, '_translation_data') and actual_index in self._translation_data:
                        current_trans = self._translation_data[actual_index]['translation']
                        preview_trans = (current_trans[:22] + "...") if len(current_trans) > 25 else current_trans
                        trans_action = QAction(f"🌍 Edit Translation: \"{preview_trans}\"", menu)
                        # Create a proper closure by defining a function that captures the current index
                        def make_trans_handler(idx):
                            return lambda: _show_translation_popup(self, self._translation_data[idx], idx)
                        trans_action.triggered.connect(make_trans_handler(actual_index))
                        menu.addAction(trans_action)
                    
                    # Add "Translate This Text" option for manual editing (only if text already recognized)
                    if hasattr(self, '_recognition_data') and actual_index in self._recognition_data:
                        # Get manual edit settings from config
                        translate_prompt = 'output only the {language} translation of this text:'  # default
                        target_language = 'English'  # default
                        
                        try:
                            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'config'):
                                manga_settings = self.main_gui.config.get('manga_settings', {})
                                manual_edit = manga_settings.get('manual_edit', {})
                                translate_prompt = manual_edit.get('translate_prompt', translate_prompt)
                                target_language = manual_edit.get('translate_target_language', target_language)
                        except Exception:
                            pass
                        
                        # Create the actual prompt by replacing {language} placeholder
                        actual_prompt = translate_prompt.replace('{language}', target_language)
                        
                        # Add the translate action
                        translate_action = QAction(f"📞 Translate This Text ({target_language})", menu)
                        def make_translate_handler(idx, prompt_text):
                            return lambda: _handle_translate_this_text(self, idx, prompt_text)
                        translate_action.triggered.connect(make_translate_handler(actual_index, actual_prompt))
                        menu.addAction(translate_action)
                    
                    # Add separator before utility options
                    if not menu.isEmpty():
                        menu.addSeparator()
                    
                    # Add "Exclude from Clean" toggle option (always available)
                    is_excluded = getattr(rect_item, 'exclude_from_clean', False)
                    if is_excluded:
                        exclude_action = QAction("❌ Exclude from Clean (ON)", menu)
                    else:
                        exclude_action = QAction("✅ Exclude from Clean (OFF)", menu)
                    def make_exclude_handler(idx, rect):
                        return lambda: _handle_toggle_exclude_clean(self, idx, rect)
                    exclude_action.triggered.connect(make_exclude_handler(actual_index, rect_item))
                    menu.addAction(exclude_action)
                    
                    # Add "Set Inpainting Iterations" option
                    current_iterations = getattr(rect_item, 'inpaint_iterations', None)
                    if current_iterations is not None:
                        iterations_text = f"🔧 Set Inpainting Iterations (Current: {current_iterations})"
                    else:
                        iterations_text = "🔧 Set Inpainting Iterations (Auto)"
                    iterations_action = QAction(iterations_text, menu)
                    def make_iterations_handler(idx, rect):
                        return lambda: _handle_set_inpainting_iterations(self, idx, rect)
                    iterations_action.triggered.connect(make_iterations_handler(actual_index, rect_item))
                    menu.addAction(iterations_action)
                    
                    # Add "Clean This Rectangle" option (disabled when waiting for model)
                    is_waiting = getattr(self, '_waiting_for_model', False)
                    if is_waiting:
                        clean_action = QAction("⏳ Clean This Rectangle (Waiting...)", menu)
                        clean_action.setEnabled(False)
                    else:
                        clean_action = QAction("🧽 Clean This Rectangle", menu)
                        def make_clean_handler(idx, rect):
                            return lambda: _handle_clean_this_rectangle(self, idx, rect)
                        clean_action.triggered.connect(make_clean_handler(actual_index, rect_item))
                    menu.addAction(clean_action)
                    
                    # Add "Delete Selected" option
                    delete_action = QAction("🗑️ Delete Selected", menu)
                    def make_delete_handler(idx, rect):
                        return lambda: _handle_delete_rectangle(self, idx, rect)
                    delete_action.triggered.connect(make_delete_handler(actual_index, rect_item))
                    menu.addAction(delete_action)
                    
                    if not menu.isEmpty():
                        # Set menu properties for better display
                        menu.setMinimumWidth(250)  # Increase minimum width for better readability
                        menu.setMaximumWidth(500)  # Allow wider menus for longer text
                        
                        # Apply better styling to the menu
                        menu.setStyleSheet("""
                            QMenu {
                                background-color: #2d2d2d;
                                border: 1px solid #5a9fd4;
                                color: white;
                                padding: 4px;
                                border-radius: 4px;
                            }
                            QMenu::item {
                                background-color: transparent;
                                padding: 8px 12px;
                                margin: 2px;
                                border-radius: 3px;
                                font-size: 11pt;
                                white-space: nowrap;
                            }
                            QMenu::item:selected {
                                background-color: #5a9fd4;
                                color: white;
                            }
                            QMenu::separator {
                                height: 1px;
                                background-color: #5a9fd4;
                                margin: 4px 8px;
                            }
                        """)
                        
                        # Show the context menu at the actual cursor position
                        menu.exec(QCursor.pos())
                    
                    # Don't propagate the right-click event
                    event.accept()
                    return
                
                # For other mouse buttons, use original behavior
                original_mouse_press(event)
                
            except Exception as e:
                print(f"[DEBUG] Mouse press error: {str(e)}")
                # Fallback to original behavior
                original_mouse_press(event)
        
        # Replace the rectangle's mouse press event
        rect_item.mousePressEvent = handle_mouse_press
        
    except Exception as e:
        print(f"[DEBUG] Error adding context menu: {str(e)}")

def _handle_delete_rectangle(self, region_index: int, rect_item):
    """Handle deleting a rectangle from the preview"""
    try:
        print(f"[DELETE_RECT] Deleting rectangle at index {region_index}")
        
        # Get the viewer and rectangles list
        if not hasattr(self, 'image_preview_widget') or not hasattr(self.image_preview_widget, 'viewer'):
            self._log("⚠️ No image preview available for delete", "warning")
            return
        
        viewer = self.image_preview_widget.viewer
        if not hasattr(viewer, 'rectangles') or not viewer.rectangles:
            self._log("⚠️ No rectangles to delete", "warning")
            return
        
        # Remove the rectangle from the scene
        if rect_item and hasattr(viewer, '_scene'):
            try:
                viewer._scene.removeItem(rect_item)
                print(f"[DELETE_RECT] Removed rectangle from scene")
            except Exception as e:
                print(f"[DELETE_RECT] Error removing from scene: {e}")
        
        # Remove from rectangles list
        if rect_item in viewer.rectangles:
            viewer.rectangles.remove(rect_item)
            print(f"[DELETE_RECT] Removed rectangle from list")
        
        # Clean up any associated data
        if hasattr(self, '_recognition_data') and region_index in self._recognition_data:
            del self._recognition_data[region_index]
            print(f"[DELETE_RECT] Cleaned up recognition data for region {region_index}")
        
        if hasattr(self, '_translation_data') and region_index in self._translation_data:
            del self._translation_data[region_index]
            print(f"[DELETE_RECT] Cleaned up translation data for region {region_index}")
        
        # Remove any text overlays for this region
        try:
            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
            if current_image and hasattr(self, '_text_overlays_by_image'):
                overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
                groups = overlays_map.get(current_image, [])
                overlays_to_remove = []
                for group in groups:
                    if hasattr(group, '_overlay_region_index') and group._overlay_region_index == region_index:
                        overlays_to_remove.append(group)
                
                for group in overlays_to_remove:
                    try:
                        if hasattr(viewer, '_scene'):
                            viewer._scene.removeItem(group)
                        groups.remove(group)
                        print(f"[DELETE_RECT] Removed text overlay for region {region_index}")
                    except Exception as e:
                        print(f"[DELETE_RECT] Error removing overlay: {e}")
        except Exception as e:
            print(f"[DELETE_RECT] Error cleaning up overlays: {e}")
        
        # Update state management if available
        try:
            if hasattr(self, 'image_state_manager') and hasattr(self.image_preview_widget, 'current_image_path'):
                current_image = self.image_preview_widget.current_image_path
                if current_image:
                    # Get current state
                    state = self.image_state_manager.get_state(current_image)
                    
                    # Remove from detection regions if present
                    if 'detection_regions' in state:
                        regions = state['detection_regions']
                        if isinstance(regions, list) and 0 <= region_index < len(regions):
                            regions.pop(region_index)
                            state['detection_regions'] = regions
                    
                    # Update state
                    self.image_state_manager.set_state(current_image, state)
                    print(f"[DELETE_RECT] Updated state management")
        except Exception as e:
            print(f"[DELETE_RECT] Error updating state: {e}")
        
        # Force scene update
        try:
            if hasattr(viewer, '_scene'):
                viewer._scene.update()
        except Exception:
            pass
        
        self._log(f"🗑️ Deleted rectangle {region_index}", "info")
        print(f"[DELETE_RECT] Successfully deleted rectangle at index {region_index}")
        
    except Exception as e:
        print(f"[DELETE_RECT] Error deleting rectangle: {e}")
        import traceback
        print(f"[DELETE_RECT] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Failed to delete rectangle: {str(e)}", "error")

def _handle_toggle_exclude_clean(self, region_index: int, rect_item):
    """Toggle exclude from clean status for a rectangle"""
    try:
        print(f"[TOGGLE_DEBUG] === TOGGLE CALLED FOR RECTANGLE {region_index} ===")
        
        # Special verbose debug for rectangle 0
        if region_index == 0:
            print(f"[RECT_0_DEBUG] *** Special debugging for rectangle 0 ***")
            print(f"[RECT_0_DEBUG] This is the problematic rectangle!")
        
        # Get current exclude status
        current_status = getattr(rect_item, 'exclude_from_clean', False)
        new_status = not current_status
        
        print(f"[TOGGLE_DEBUG] Rectangle {region_index}: current_status={current_status}, new_status={new_status}")
        print(f"[TOGGLE_DEBUG] Rectangle object: {rect_item}")
        print(f"[TOGGLE_DEBUG] Rectangle has exclude_from_clean attr: {hasattr(rect_item, 'exclude_from_clean')}")
        
        # Set the new status on the rectangle
        rect_item.exclude_from_clean = new_status
        
        # Visual feedback - change rectangle appearance to indicate excluded status
        from PySide6.QtGui import QPen, QBrush, QColor
        if new_status:
            # Excluded - use red/orange styling
            rect_item.setPen(QPen(QColor(255, 140, 0), 3))  # Orange border, thicker
            rect_item.setBrush(QBrush(QColor(255, 140, 0, 30)))  # Semi-transparent orange fill
            self._log(f"❌ Rectangle {region_index} excluded from inpainting", "info")
        else:
            # Not excluded - restore normal styling based on rectangle type
            if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                # Blue for recognized text
                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
            else:
                # Green for detection boxes
                rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
            self._log(f"✅ Rectangle {region_index} included in inpainting", "info")
        
        # EXCLUSION PERSISTENCE REMOVAL: Don't save exclusion state to persist across sessions
        # Exclusions are now session-only and reset when the app is restarted
        print(f"[EXCLUDE_CLEAN] Rectangle {region_index} exclude status: {new_status} (session-only, not persisted)")
        
        print(f"[EXCLUDE_CLEAN] Rectangle {region_index} exclude status: {new_status}")
        
    except Exception as e:
        print(f"[EXCLUDE_CLEAN] Error toggling exclude status: {e}")
        import traceback
        print(f"[EXCLUDE_CLEAN] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Failed to toggle exclude status: {str(e)}", "error")

def _handle_set_inpainting_iterations(self, region_index: int, rect_item):
    """Handle setting custom inpainting iterations for a rectangle"""
    try:
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from PySide6.QtCore import Qt
        
        print(f"[INPAINT_ITERATIONS] Setting iterations for rectangle {region_index}")
        
        # Get current value
        current_iterations = getattr(rect_item, 'inpaint_iterations', None)
        current_display = current_iterations if current_iterations is not None else "Auto"
        
        # Show input dialog
        dialog_text = (
            f"Set inpainting iterations for rectangle {region_index}\n\n"
            f"Current: {current_display}\n\n"
            f"Enter number of iterations (-1 to 50):\n-1 = Auto, 0-50 = Custom iterations"
        )
        
        value, ok = QInputDialog.getInt(
            self.dialog,
            "Set Inpainting Iterations",
            dialog_text,
            current_iterations if current_iterations is not None else -1
        )
        
        if ok:
            # Validate input range (-1 to 50, where -1 means auto)
            if value < -1 or value > 50:
                QMessageBox.warning(
                    self.dialog,
                    "Invalid Input", 
                    f"Please enter a value between -1 and 50.\n-1 = Auto, 0-50 = Custom iterations"
                )
                return
            
            if value == -1:
                # Reset to auto
                rect_item.inpaint_iterations = None
                self._log(f"🔧 Rectangle {region_index}: inpainting set to AUTO", "info")
                print(f"[INPAINT_ITERATIONS] Rectangle {region_index} set to auto iterations")
            else:
                # Set custom value
                rect_item.inpaint_iterations = value
                self._log(f"🔧 Rectangle {region_index}: inpainting set to {value} iterations", "info")
                print(f"[INPAINT_ITERATIONS] Rectangle {region_index} set to {value} iterations")
            
            # Store in state management for persistence
            try:
                if hasattr(self, 'image_state_manager') and hasattr(self.image_preview_widget, 'current_image_path'):
                    current_image = self.image_preview_widget.current_image_path
                    if current_image:
                        # Get current state
                        state = self.image_state_manager.get_state(current_image)
                        
                        # Initialize inpainting iterations dict if not present
                        if 'inpaint_iterations' not in state:
                            state['inpaint_iterations'] = {}
                        
                        # Store the iteration value for this region
                        if value == -1:
                            # Remove from dict when set to auto
                            if str(region_index) in state['inpaint_iterations']:
                                del state['inpaint_iterations'][str(region_index)]
                                print(f"[INPAINT_ITERATIONS] Removed custom iterations for region {region_index} from state")
                        else:
                            state['inpaint_iterations'][str(region_index)] = value
                            print(f"[INPAINT_ITERATIONS] Saved {value} iterations for region {region_index} to state")
                        
                        # Update state
                        self.image_state_manager.set_state(current_image, state)
                        print(f"[INPAINT_ITERATIONS] Updated state management")
            except Exception as e:
                print(f"[INPAINT_ITERATIONS] Error updating state: {e}")
            
            # Visual feedback - update rectangle appearance slightly
            from PySide6.QtGui import QPen, QBrush, QColor
            if value != -1:
                # Custom iterations - add slight blue tint to border
                if getattr(rect_item, 'exclude_from_clean', False):
                    # Keep orange if excluded
                    pass
                else:
                    # Add blue tint to show custom iterations
                    if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                        # Slightly brighter blue for recognized + custom iterations
                        rect_item.setPen(QPen(QColor(50, 170, 255), 2))
                    else:
                        # Slightly blue-green for detection + custom iterations
                        rect_item.setPen(QPen(QColor(0, 200, 150), 2))
        
    except Exception as e:
        print(f"[INPAINT_ITERATIONS] Error setting iterations: {e}")
        import traceback
        print(f"[INPAINT_ITERATIONS] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Failed to set inpainting iterations: {str(e)}", "error")

def _handle_clean_this_rectangle(self, region_index: int, rect_item):
    """Handle cleaning/inpainting a specific rectangle region in background thread"""
    try:
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import QThread
        import numpy as np
        import cv2
        import threading
        
        print(f"[CLEAN_RECT] Starting single rectangle clean for region {region_index}")
        self._log(f"🎯 Starting clean for rectangle {region_index}...", "info")
        
        # Validate current state
        if not hasattr(self.image_preview_widget, 'current_image_path') or not self.image_preview_widget.current_image_path:
            self._log(f"❌ No image loaded", "error")
            return
        
        current_image_path = self.image_preview_widget.current_image_path
        
        # Get the rectangle bounds
        if not hasattr(self.image_preview_widget, 'viewer') or not hasattr(self.image_preview_widget.viewer, 'rectangles'):
            self._log(f"❌ No rectangles available", "error")
            return
        
        rectangles = self.image_preview_widget.viewer.rectangles
        if region_index < 0 or region_index >= len(rectangles):
            self._log(f"❌ Invalid rectangle index: {region_index}", "error")
            return
        
        target_rect = rectangles[region_index]
        shape_type = getattr(target_rect, 'shape_type', 'rect')
        
        # Prefer sceneBoundingRect for robust coordinates across item types
        rect_bounds = target_rect.sceneBoundingRect()
        print(f"[CLEAN_RECT] Target shape {region_index} type={shape_type} bounds: {rect_bounds.x()}, {rect_bounds.y()}, {rect_bounds.width()}, {rect_bounds.height()}")
        
        # Check if rectangle is excluded from cleaning
        is_excluded = getattr(target_rect, 'exclude_from_clean', False)
        if is_excluded:
            reply = QMessageBox.question(
                self.dialog,
                "Rectangle Excluded",
                f"Rectangle {region_index} is currently excluded from inpainting.\n\nDo you want to clean it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                print(f"[CLEAN_RECT] User cancelled - rectangle {region_index} is excluded")
                return
        
        # Get viewer dimensions for scaling (lightweight operation)
        viewer_rect = self.image_preview_widget.viewer.sceneRect()
        
        # Get custom iterations for this rectangle if set
        custom_iterations = getattr(target_rect, 'inpaint_iterations', None)
        print(f"[CLEAN_RECT] Custom iterations for rectangle {region_index}: {custom_iterations}")
        
        # Start pulse effect on the rectangle
        _add_rectangle_pulse_effect(self, target_rect, region_index)
        
        # Lazy preloading: if this is the first time using clean rectangle, preload the model
        # This happens in the background thread so it doesn't block the UI
        if (self._shared_inpainter is None or not getattr(self._shared_inpainter, 'model_loaded', False)):
            print(f"[CLEAN_RECT] First use detected - will preload inpainter in background")
        
        # Run everything in background thread to avoid GUI lag
        def run_single_rect_clean():
            try:
                print(f"[CLEAN_RECT_THREAD] Starting background processing for region {region_index}")
                
                # Lazy preload inpainter if not already loaded (first use optimization)
                if (self._shared_inpainter is None or not getattr(self._shared_inpainter, 'model_loaded', False)):
                    print(f"[CLEAN_RECT_THREAD] Preloading inpainter for first use...")
                    _preload_shared_inpainter(self, )
                
                # Determine which image to use as base - prefer output image if available
                base_image_path = current_image_path
                if (hasattr(self.image_preview_widget, 'current_translated_path') and 
                    self.image_preview_widget.current_translated_path and 
                    os.path.exists(self.image_preview_widget.current_translated_path)):
                    base_image_path = self.image_preview_widget.current_translated_path
                    print(f"[CLEAN_RECT_THREAD] Using output image as base: {os.path.basename(base_image_path)}")
                else:
                    print(f"[CLEAN_RECT_THREAD] Using source image as base: {os.path.basename(base_image_path)}")
                
                # Load the base image in background thread
                original_image = cv2.imread(base_image_path)
                if original_image is None:
                    self.update_queue.put(('single_clean_error', {
                        'region_index': region_index,
                        'error': f'Failed to load base image: {base_image_path}'
                    }))
                    return
                
                print(f"[CLEAN_RECT_THREAD] Loaded image: {original_image.shape}")
                
                # Convert rectangle bounds to image coordinates
                img_height, img_width = original_image.shape[:2]
                
                scale_x = img_width / viewer_rect.width()
                scale_y = img_height / viewer_rect.height()
                
                # Convert to image coordinates
                x = int(rect_bounds.x() * scale_x)
                y = int(rect_bounds.y() * scale_y)
                w = int(rect_bounds.width() * scale_x)
                h = int(rect_bounds.height() * scale_y)
                
                # Ensure bounds are within image
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(1, min(w, img_width - x))
                h = max(1, min(h, img_height - y))
                
                print(f"[CLEAN_RECT_THREAD] Image coordinates: x={x}, y={y}, w={w}, h={h} (image: {img_width}x{img_height})")
                
                # Create a mask for this specific region
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                try:
                    if shape_type == 'polygon' and hasattr(target_rect, 'path'):
                        # Convert polygon points to image coordinates
                        poly_scene = target_rect.mapToScene(target_rect.path().toFillPolygon())
                        pts_img = []
                        for p in poly_scene:
                            px = int(round(p.x() * scale_x))
                            py = int(round(p.y() * scale_y))
                            # Clamp
                            px = max(0, min(px, img_width - 1))
                            py = max(0, min(py, img_height - 1))
                            pts_img.append([px, py])
                        if len(pts_img) >= 3:
                            import numpy as _np
                            arr = _np.array(pts_img, dtype=_np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [arr], 255)
                        else:
                            # Fallback to bounding rect
                            mask[y:y+h, x:x+w] = 255
                    elif shape_type == 'ellipse':
                        # Draw ellipse mask from bounding rect
                        cx = int(round((x + x + w) / 2))
                        cy = int(round((y + y + h) / 2))
                        rx = max(1, int(round(w / 2)))
                        ry = max(1, int(round(h / 2)))
                        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                    else:
                        # Rectangle
                        mask[y:y+h, x:x+w] = 255
                except Exception as me:
                    print(f"[CLEAN_RECT_THREAD] Mask build error, using rectangle fallback: {me}")
                    mask[y:y+h, x:x+w] = 255
                
                print(f"[CLEAN_RECT_THREAD] Created mask with {np.sum(mask > 0)} white pixels")
                
                # Run inpainting
                result = _run_inpainting_on_region(self, 
                    original_image, 
                    mask, 
                    region_index, 
                    custom_iterations
                )
                
                # Return inpainter to pool after use
                try:
                    from manga_translator import MangaTranslator
                    released_inp, released_det = MangaTranslator.force_release_all_pool_checkouts()
                    if released_inp > 0:
                        print(f"[CLEAN_RECT_THREAD] Returned {released_inp} inpainter(s) to pool")
                        self.update_queue.put(('update_pool_tracker', None))
                except Exception as e:
                    print(f"[CLEAN_RECT_THREAD] Error returning inpainter to pool: {e}")
                    import traceback
                    print(f"[CLEAN_RECT_THREAD] Pool return traceback: {traceback.format_exc()}")
                
                if result is None:
                    self.update_queue.put(('single_clean_error', {
                        'region_index': region_index,
                        'error': 'Inpainting failed'
                    }))
                    return
                
                print(f"[CLEAN_RECT_THREAD] Inpainting completed for rectangle {region_index}")
                
                # Send result back to main thread
                self.update_queue.put(('single_clean_complete', {
                    'region_index': region_index,
                    'result_image': result,
                    'original_path': current_image_path
                }))
                
            except Exception as e:
                print(f"[CLEAN_RECT_THREAD] Error during inpainting: {e}")
                import traceback
                print(f"[CLEAN_RECT_THREAD] Traceback: {traceback.format_exc()}")
                self.update_queue.put(('single_clean_error', {
                    'region_index': region_index,
                    'error': str(e)
                }))
        
        # Start background thread
        clean_thread = threading.Thread(target=run_single_rect_clean, daemon=True)
        clean_thread.start()
        print(f"[CLEAN_RECT] Started background thread for region {region_index} cleaning")
        
    except Exception as e:
        print(f"[CLEAN_RECT] Error setting up rectangle cleaning: {e}")
        import traceback
        print(f"[CLEAN_RECT] Traceback: {traceback.format_exc()}")
        self._log(f"❌ Failed to start rectangle cleaning: {str(e)}", "error")

def _get_or_create_shared_inpainter(self, method: str, model_path: str):
    """Get or create a LocalInpainter via MangaTranslator's preload pool.
    Checks out a spare instance from the pool, or creates a new one if none available.
    """
    try:
        import os
        # Normalize model_path to match pool keys used by MangaTranslator
        if model_path:
            try:
                model_path = os.path.abspath(os.path.normpath(model_path))
            except Exception:
                pass
        
        from manga_translator import MangaTranslator
        # If we already have a translator, delegate to its pool-aware method
        if hasattr(self, 'translator') and self.translator:
            return self.translator._get_or_init_shared_local_inpainter(method, model_path, force_reload=False)
        
        # Otherwise, create a lightweight translator to initialize/access the pool
        try:
            ocr_config = _get_ocr_config(self, )
        except Exception:
            ocr_config = {}
        try:
            from unified_api_client import UnifiedClient
            api_key = self.main_gui.config.get('api_key', '') or 'dummy'
            model = self.main_gui.config.get('model', 'gpt-4o-mini')
            uc = UnifiedClient(model=model, api_key=api_key)
        except Exception:
            uc = None
        
        # Fallback logging callback that prints to stdout if _log is unavailable
        def _cb(msg, level='info'):
            try:
                if hasattr(self, '_log'):
                    self._log(msg, level)
                else:
                    print(f"[GUI] {level.upper()}: {msg}")
            except Exception:
                pass
        
        mt = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=_cb, skip_inpainter_init=True)
        return mt._get_or_init_shared_local_inpainter(method, model_path, force_reload=False)
    except Exception as e:
        print(f"[SHARED_INPAINTER] Pool access error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def _preload_shared_bubble_detector(self):
    """Preload the shared bubble detector with current settings if not already loaded"""
    try:
        # Get bubble detection settings
        ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {})
        bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', True)
        
        if not bubble_detection_enabled:
            print(f"[PRELOAD_DETECTOR] Skipping preload - bubble detection is disabled")
            return
        
        # Create a temporary translator to access the detector pool
        try:
            ocr_config = _get_ocr_config(self, )
        except Exception:
            ocr_config = {}
        
        try:
            from unified_api_client import UnifiedClient
            api_key = self.main_gui.config.get('api_key', '') or 'dummy'
            model = self.main_gui.config.get('model', 'gpt-4o-mini')
            uc = UnifiedClient(model=model, api_key=api_key)
        except Exception:
            uc = None
        
        def _cb(msg, level='info'):
            try:
                if hasattr(self, '_log'):
                    self._log(msg, level)
                else:
                    print(f"[GUI] {level.upper()}: {msg}")
            except Exception:
                pass
        
        from manga_translator import MangaTranslator
        mt = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=_cb, skip_inpainter_init=True)
        
        # Preload 1 detector instance
        print(f"[PRELOAD_DETECTOR] Preloading bubble detector...")
        created = mt.preload_bubble_detectors(ocr_settings, 1)
        if created > 0:
            print(f"[PRELOAD_DETECTOR] Successfully preloaded {created} bubble detector(s)")
            self._log(f"🎯 Preloaded bubble detector", "info")
        else:
            print(f"[PRELOAD_DETECTOR] Bubble detector already loaded or preload skipped")
        
    except Exception as e:
        print(f"[PRELOAD_DETECTOR] Error during preload: {e}")
        import traceback
        print(f"[PRELOAD_DETECTOR] Traceback: {traceback.format_exc()}")

def _preload_shared_inpainter(self):
    """Preload inpainter into the pool for fast access on first use"""
    try:
        import os
        from manga_translator import MangaTranslator
        
        # Get current inpainting settings
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        
        if inpaint_method != 'local':
            print(f"[PRELOAD_INPAINTER] Skipping preload - method is {inpaint_method}, not local")
            return
        
        # Get model path
        model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
        try:
            if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                model_path = ''
        except Exception:
            pass
        
        # Normalize path
        if model_path:
            try:
                model_path = os.path.abspath(os.path.normpath(model_path))
            except Exception:
                pass
        
        key = (local_model, model_path or '')
        
        # Check if already in pool
        with MangaTranslator._inpaint_pool_lock:
            rec = MangaTranslator._inpaint_pool.get(key)
            if rec and rec.get('spares'):
                print(f"[PRELOAD_INPAINTER] {local_model} already in pool ({len(rec['spares'])} instance(s))")
                return
        
        # Try to download if path not found
        if not model_path or not os.path.exists(model_path):
            try:
                print(f"[PRELOAD_INPAINTER] Downloading {local_model} model for preloading...")
                from local_inpainter import LocalInpainter
                temp_inpainter = LocalInpainter()
                model_path = temp_inpainter.download_jit_model(local_model)
                if model_path:
                    # Update config with downloaded path
                    self.main_gui.config[f'manga_{local_model}_model_path'] = model_path
                    print(f"[PRELOAD_INPAINTER] Downloaded and cached model path: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"[PRELOAD_INPAINTER] Failed to download {local_model}: {e}")
                return
        
        if model_path and os.path.exists(model_path):
            print(f"[PRELOAD_INPAINTER] Preloading {local_model} inpainter into pool...")
            
            # Create a temporary translator to use preload_local_inpainters
            try:
                ocr_config = _get_ocr_config(self, )
            except Exception:
                ocr_config = {}
            
            try:
                from unified_api_client import UnifiedClient
                api_key = self.main_gui.config.get('api_key', '') or 'dummy'
                model = self.main_gui.config.get('model', 'gpt-4o-mini')
                uc = UnifiedClient(model=model, api_key=api_key)
            except Exception:
                uc = None
            
            def _cb(msg, level='info'):
                try:
                    if hasattr(self, '_log'):
                        self._log(msg, level)
                    else:
                        print(f"[GUI] {level.upper()}: {msg}")
                except Exception:
                    pass
            
            mt = MangaTranslator(ocr_config=ocr_config, unified_client=uc, main_gui=self.main_gui, log_callback=_cb, skip_inpainter_init=True)
            
            # Preload 1 inpainter instance into the pool (use concurrent for faster loading)
            print(f"[PRELOAD_INPAINTER] Calling preload_local_inpainters_concurrent for {local_model}...")
            created = mt.preload_local_inpainters_concurrent(local_model, model_path, 1)
            if created > 0:
                print(f"[PRELOAD_INPAINTER] Successfully preloaded {created} inpainter instance(s)")
                self._log(f"🎯 Preloaded {local_model.upper()} inpainting model", "info")
            else:
                print(f"[PRELOAD_INPAINTER] No instances preloaded (may already exist in pool)")
        
    except Exception as e:
        print(f"[PRELOAD_INPAINTER] Error during preload: {e}")
        import traceback
        print(f"[PRELOAD_INPAINTER] Traceback: {traceback.format_exc()}")

def _run_inpainting_on_region(self, image, mask, region_index, custom_iterations=None):
    """Run inpainting on a specific region with the given mask"""
    try:
        import cv2
        import numpy as np
        import os
        from local_inpainter import LocalInpainter
        print(f"[INPAINT_REGION] Running local inpainting on region {region_index}")
        
        # Get inpainting settings from manga integration config (same as _run_clean_background)
        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')
        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')
        
        print(f"[INPAINT_REGION] Using method: {inpaint_method}, model: {local_model}")
        
        if inpaint_method == 'local':
            # Get model path from config (same way as _run_clean_background)
            model_path = self.main_gui.config.get(f'manga_{local_model}_model_path', '')
            try:
                if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                    model_path = ''
            except Exception:
                pass
            
            # Ensure we have a model path (download if needed)
            resolved_model_path = model_path
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                try:
                    print(f"[INPAINT_REGION] Model path not found, downloading {local_model} model...")
                    from local_inpainter import LocalInpainter
                    temp_inpainter = LocalInpainter()
                    resolved_model_path = temp_inpainter.download_jit_model(local_model)
                except Exception as e:
                    print(f"[INPAINT_REGION] Model download failed: {e}")
                    resolved_model_path = None
            
            if not resolved_model_path or not os.path.exists(resolved_model_path):
                print(f"[INPAINT_REGION] No valid model path for {local_model}")
                return None
            
            # Use shared inpainter instance instead of creating new one
            print(f"[INPAINT_REGION] Getting shared inpainter for {local_model}")
            inpainter = _get_or_create_shared_inpainter(self, local_model, resolved_model_path)
            if inpainter is None:
                print(f"[INPAINT_REGION] Failed to get shared inpainter")
                return None
            
            # Run inpainting with custom iterations if available
            if custom_iterations is not None:
                print(f"[INPAINT_REGION] Using custom iterations: {custom_iterations}")
                cleaned_image = inpainter.inpaint(image, mask, iterations=custom_iterations)
            else:
                print(f"[INPAINT_REGION] Using auto iterations")
                cleaned_image = inpainter.inpaint(image, mask)
            
            print(f"[INPAINT_REGION] Local inpainting completed for region {region_index}")
            return cleaned_image
            
        else:
            # For cloud/hybrid methods, fallback to OpenCV inpainting
            print(f"[INPAINT_REGION] Using OpenCV fallback for method: {inpaint_method}")
            cleaned_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            return cleaned_image
        
    except Exception as e:
        print(f"[INPAINT_REGION] Error during inpainting: {e}")
        import traceback
        print(f"[INPAINT_REGION] Traceback: {traceback.format_exc()}")
        return None

def _update_image_preview_with_result(self, result_image, original_path):
    """Update the image preview with the inpainting result while preserving rectangles and save to disk"""
    try:
        import cv2
        import tempfile
        import os
        
        # Save cleaned image to permanent location (same as Clean button)
        # Check for OUTPUT_DIRECTORY override (prefer config over env var)
        parent_dir = os.path.dirname(original_path)
        filename = os.path.basename(original_path)
        base, ext = os.path.splitext(filename)
        
        override_dir = None
        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
            override_dir = self.main_gui.config.get('output_directory', '')
        if not override_dir:
            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
        
        if override_dir:
            output_dir = os.path.join(override_dir, f"{base}_translated")
        else:
            output_dir = os.path.join(parent_dir, f"{base}_translated")
        
        os.makedirs(output_dir, exist_ok=True)
        cleaned_path = os.path.join(output_dir, f"{base}_cleaned{ext}")
        
        # Save to both permanent location and temporary file
        cv2.imwrite(cleaned_path, result_image)
        print(f"[UPDATE_PREVIEW] Saved cleaned image to: {os.path.relpath(cleaned_path, parent_dir)}")
        
        # Also save temporary file for immediate preview update
        temp_dir = tempfile.gettempdir()
        temp_filename = f"manga_clean_result_{os.path.basename(original_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        cv2.imwrite(temp_path, result_image)
        
        # Persist cleaned path to state (same as Clean button)
        try:
            if hasattr(self, 'image_state_manager'):
                self.image_state_manager.update_state(original_path, {'cleaned_image_path': cleaned_path})
                print(f"[UPDATE_PREVIEW] Persisted cleaned image path to state")
        except Exception as e:
            print(f"[UPDATE_PREVIEW] Failed to persist state: {e}")
        
        # Check current display mode and handle accordingly
        try:
            ipw = self.image_preview_widget
            current_mode = getattr(ipw, 'source_display_mode', 'original')
            
            if current_mode == 'translated':
                # User is in 'translated' mode - stay there and trigger save & update overlay
                # This re-renders the translated output with the newly cleaned area
                print(f"[UPDATE_PREVIEW] In 'translated' mode - staying and triggering save & update overlay")
                
                # Store the cleaned path
                ipw.current_translated_path = cleaned_path
                
                # Reload the image first to pick up the cleaned version as base
                ipw.load_image(original_path, preserve_rectangles=True, preserve_text_overlays=True)
                
                # Trigger save & update overlay to re-render translated output on top of cleaned image
                try:
                    save_positions_and_rerender(self)
                    print(f"[UPDATE_PREVIEW] Triggered save & update overlay for re-render")
                except Exception as render_err:
                    print(f"[UPDATE_PREVIEW] Failed to trigger save & update overlay: {render_err}")
            else:
                # Switch display mode to 'cleaned' and update the preview
                ipw.source_display_mode = 'cleaned'
                ipw.cleaned_images_enabled = True  # Deprecated flag for compatibility
                
                # Update the cleaned toggle button appearance to match 'cleaned' state
                if hasattr(ipw, 'cleaned_toggle_btn') and ipw.cleaned_toggle_btn:
                    ipw.cleaned_toggle_btn.setText("🧽")  # Sponge for cleaned
                    ipw.cleaned_toggle_btn.setToolTip("Showing cleaned images (click to cycle)")
                    ipw.cleaned_toggle_btn.setStyleSheet("""
                        QToolButton {
                            background-color: #4a7ba7;
                            border: 2px solid #5a9fd4;
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
                            background-color: #5a9fd4;
                        }
                    """)
                
                # Store the cleaned path
                ipw.current_translated_path = cleaned_path
                print(f"[UPDATE_PREVIEW] Switched display mode to 'cleaned'")
                
                # Reload the image to show the cleaned version while preserving rectangles
                ipw.load_image(original_path, preserve_rectangles=True, preserve_text_overlays=True)
                print(f"[UPDATE_PREVIEW] Refreshed preview with cleaned image")
                    
        except Exception as e:
            print(f"[UPDATE_PREVIEW] Failed to update preview: {e}")
        
    except Exception as e:
        print(f"[UPDATE_PREVIEW] Error updating preview: {e}")

def _add_rectangle_pulse_effect(self, rect_item, region_index, auto_remove=False):
    """Add a purple pulse effect to a specific rectangle during operations
    
    Args:
        rect_item: The rectangle item to pulse
        region_index: Index of the region
        auto_remove: If True, auto-remove after 0.75s (for save position). 
                    If False, loop indefinitely until manually removed (for OCR/clean/translate)
    """
    try:
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtCore import QRectF, QPropertyAnimation, QEasingCurve, Qt, QObject, Property
        from PySide6.QtGui import QPen, QBrush, QColor
        
        mode = "auto-remove" if auto_remove else "loop until complete"
        print(f"[RECT_PULSE] Adding pulse effect to rectangle {region_index} (mode: {mode})")
        
        # Store original pen for restoration
        original_pen = rect_item.pen()
        setattr(rect_item, '_original_pen', original_pen)
        
        # Create pulsing animation using QObject wrapper
        class PulsingRectangle(QObject):
            def __init__(self, rect_item, parent=None):
                super().__init__(parent)
                self._rect_item = rect_item
                self._intensity = 100
            
            def get_intensity(self):
                return self._intensity
            
            def set_intensity(self, value):
                self._intensity = value
                # Create purple pen with varying intensity
                purple_color = QColor(147, 112, 219, int(value))  # Medium slate blue/purple
                pen = QPen(purple_color, 3)  # Thicker pen for visibility
                self._rect_item.setPen(pen)
            
            intensity = Property(int, get_intensity, set_intensity)
        
        # Store pulse wrapper on the rectangle item
        pulse_wrapper = PulsingRectangle(rect_item)
        setattr(rect_item, '_pulse_wrapper', pulse_wrapper)
        
        # Create the animation
        pulse_animation = QPropertyAnimation(pulse_wrapper, b"intensity")
        pulse_animation.setDuration(750)  # 0.75 seconds (750ms)
        pulse_animation.setStartValue(80)
        pulse_animation.setEndValue(255)
        pulse_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        if auto_remove:
            # For save position: run once and auto-remove
            pulse_animation.setLoopCount(1)
            # Auto-remove pulse effect when animation finishes
            def on_animation_finished():
                try:
                    _remove_rectangle_pulse_effect(self, rect_item, region_index)
                except Exception as e:
                    print(f"[RECT_PULSE] Error removing pulse on finish: {e}")
            pulse_animation.finished.connect(on_animation_finished)
        else:
            # For other operations: loop indefinitely until manually removed
            pulse_animation.setLoopCount(-1)  # Loop forever
        
        pulse_animation.start()
        
        # Store animation on the rectangle item
        setattr(rect_item, '_pulse_animation', pulse_animation)
        
        print(f"[RECT_PULSE] Started pulse animation for rectangle {region_index}")
        
    except Exception as e:
        print(f"[RECT_PULSE] Error adding pulse effect: {e}")
        import traceback
        print(f"[RECT_PULSE] Traceback: {traceback.format_exc()}")

def _remove_rectangle_pulse_effect(self, rect_item, region_index):
    """Remove the pulse effect from a specific rectangle"""
    try:
        print(f"[RECT_PULSE] Removing pulse effect from rectangle {region_index}")
        
        # Stop animation
        if hasattr(rect_item, '_pulse_animation'):
            rect_item._pulse_animation.stop()
            delattr(rect_item, '_pulse_animation')
        
        # Remove pulse wrapper
        if hasattr(rect_item, '_pulse_wrapper'):
            delattr(rect_item, '_pulse_wrapper')
        
        # Restore original pen
        if hasattr(rect_item, '_original_pen'):
            rect_item.setPen(rect_item._original_pen)
            delattr(rect_item, '_original_pen')
        else:
            # Fallback: restore normal styling based on rectangle type
            from PySide6.QtGui import QPen, QBrush, QColor
            if getattr(rect_item, 'exclude_from_clean', False):
                # Orange for excluded
                rect_item.setPen(QPen(QColor(255, 165, 0), 2))
            elif hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                # Blue for recognized text
                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
            else:
                # Green for detection boxes
                rect_item.setPen(QPen(QColor(0, 255, 0), 2))
        
        print(f"[RECT_PULSE] Removed pulse effect from rectangle {region_index}")
        
    except Exception as e:
        print(f"[RECT_PULSE] Error removing pulse effect: {e}")

def _restore_exclusion_status_from_state(self, image_path: str):
    """Restore exclusion status for rectangles from saved state - DISABLED"""
    try:
        print(f"[EXCLUDE_RESTORE] === EXCLUSION RESTORATION DISABLED - ALWAYS START WITH NO EXCLUSIONS ===")
        print(f"[EXCLUDE_RESTORE] Exclusion toggle state no longer persists across sessions")
        
        # Ensure all rectangles start with no exclusion styling
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[EXCLUDE_RESTORE] Ensuring {len(rectangles)} rectangles start with normal styling")
            
            from PySide6.QtGui import QPen, QBrush, QColor
            for i, rect_item in enumerate(rectangles):
                # Ensure exclude flag is False
                rect_item.exclude_from_clean = False
                
                # Apply normal styling based on rectangle type
                if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                    # Blue for recognized text
                    rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                    rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                else:
                    # Green for detection boxes
                    rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                    rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
        
        print(f"[EXCLUDE_RESTORE] All rectangles initialized with no exclusions")
        
    except Exception as e:
        print(f"[EXCLUDE_RESTORE] Error initializing exclusion status: {e}")

def _get_custom_iterations_for_regions(self, regions: list) -> dict:
    """Get custom inpainting iterations for regions from rectangles.
    
    Args:
        regions: List of region dictionaries with rect_index
        
    Returns:
        dict: Mapping of region_index -> custom_iterations for regions with custom values
    """
    custom_iterations = {}
    try:
        if not hasattr(self.image_preview_widget, 'viewer') or not self.image_preview_widget.viewer.rectangles:
            return custom_iterations
        
        rectangles = self.image_preview_widget.viewer.rectangles
        print(f"[CUSTOM_ITERATIONS] Checking {len(rectangles)} rectangles for custom iterations")
        
        for region in regions:
            rect_index = region.get('rect_index', None)
            if rect_index is not None and 0 <= rect_index < len(rectangles):
                rect_item = rectangles[rect_index]
                custom_iters = getattr(rect_item, 'inpaint_iterations', None)
                if custom_iters is not None:
                    custom_iterations[rect_index] = custom_iters
                    print(f"[CUSTOM_ITERATIONS] Found custom iterations for region {rect_index}: {custom_iters}")
        
        print(f"[CUSTOM_ITERATIONS] Total custom iterations found: {len(custom_iterations)}")
        return custom_iterations
        
    except Exception as e:
        print(f"[CUSTOM_ITERATIONS] Error getting custom iterations: {e}")
        return custom_iterations

def _restore_inpainting_iterations_from_state(self, image_path: str):
    """Restore custom inpainting iterations for rectangles from saved state"""
    try:
        print(f"[ITERATIONS_RESTORE] === ITERATION RESTORATION CALLED FOR IMAGE: {image_path} ===")
        
        if not hasattr(self, 'image_state_manager'):
            print(f"[ITERATIONS_RESTORE] No image_state_manager available")
            return
        
        # Get iterations dict from state
        state = self.image_state_manager.get_state(image_path)
        print(f"[ITERATIONS_RESTORE] State for {image_path}: {state}")
        
        custom_iterations = state.get('inpaint_iterations', {})
        print(f"[ITERATIONS_RESTORE] Custom iterations from state: {custom_iterations}")
        
        if not custom_iterations:
            print(f"[ITERATIONS_RESTORE] No custom iterations to restore")
            return
        
        # Apply custom iterations to rectangles
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[ITERATIONS_RESTORE] Found {len(rectangles)} rectangles in viewer")
            
            for region_str, iterations in custom_iterations.items():
                try:
                    region_index = int(region_str)
                    print(f"[ITERATIONS_RESTORE] Processing custom iterations for region {region_index}: {iterations}")
                    
                    if 0 <= region_index < len(rectangles):
                        rect_item = rectangles[region_index]
                        
                        # Set custom iterations
                        rect_item.inpaint_iterations = iterations
                        print(f"[ITERATIONS_RESTORE] Set inpaint_iterations={iterations} for region {region_index}")
                        
                        # Apply visual styling to indicate custom iterations
                        from PySide6.QtGui import QPen, QBrush, QColor
                        if not getattr(rect_item, 'exclude_from_clean', False):
                            # Only apply styling if not excluded (excluded styling takes priority)
                            if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                                # Slightly brighter blue for recognized + custom iterations
                                rect_item.setPen(QPen(QColor(50, 170, 255), 2))
                            else:
                                # Slightly blue-green for detection + custom iterations
                                rect_item.setPen(QPen(QColor(0, 200, 150), 2))
                        
                        print(f"[ITERATIONS_RESTORE] Applied styling for custom iterations to region {region_index}")
                    else:
                        print(f"[ITERATIONS_RESTORE] Region index {region_index} out of bounds (rectangles: {len(rectangles)})")
                except (ValueError, TypeError) as e:
                    print(f"[ITERATIONS_RESTORE] Error processing region {region_str}: {e}")
        else:
            print(f"[ITERATIONS_RESTORE] No rectangles available in viewer")
        
        print(f"[ITERATIONS_RESTORE] Completed restoration for {len(custom_iterations)} custom iteration settings")
        
    except Exception as e:
        print(f"[ITERATIONS_RESTORE] Error restoring inpainting iterations: {e}")

def _debug_exclusion_status(self):
    """Debug function to show current exclusion status"""
    try:
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            print(f"[EXCLUSION_DEBUG] No current image")
            return
        
        current_image = self.image_preview_widget.current_image_path
        print(f"[EXCLUSION_DEBUG] Current image: {current_image}")
        
        # Check state
        if hasattr(self, 'image_state_manager'):
            state = self.image_state_manager.get_state(current_image)
            excluded_regions = state.get('excluded_from_clean', []) if state else []
            print(f"[EXCLUSION_DEBUG] Excluded regions in state: {excluded_regions}")
        
        # Check rectangles
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[EXCLUSION_DEBUG] Total rectangles: {len(rectangles)}")
            
            for i, rect_item in enumerate(rectangles):
                is_excluded = getattr(rect_item, 'exclude_from_clean', False)
                print(f"[EXCLUSION_DEBUG] Rectangle {i}: excluded={is_excluded}")
        
    except Exception as e:
        print(f"[EXCLUSION_DEBUG] Error in debug: {e}")

def _clear_all_exclusion_states(self):
    """Clear exclusion states for all images - useful for debugging"""
    try:
        if not hasattr(self, 'image_state_manager'):
            print(f"[CLEAR_EXCLUSION] No image_state_manager available")
            return
        
        # Get current image if available
        current_image = None
        if hasattr(self, 'image_preview_widget') and self.image_preview_widget.current_image_path:
            current_image = self.image_preview_widget.current_image_path
        
        # Clear current image's exclusion state
        if current_image:
            state = self.image_state_manager.get_state(current_image)
            if 'excluded_from_clean' in state:
                old_exclusions = state['excluded_from_clean']
                state['excluded_from_clean'] = []
                self.image_state_manager.set_state(current_image, state)
                print(f"[CLEAR_EXCLUSION] Cleared exclusions for current image: {old_exclusions} -> []")
                
                # Also clear visual state from rectangles
                if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
                    rectangles = self.image_preview_widget.viewer.rectangles
                    for i, rect_item in enumerate(rectangles):
                        if getattr(rect_item, 'exclude_from_clean', False):
                            rect_item.exclude_from_clean = False
                            # Restore normal styling
                            from PySide6.QtGui import QPen, QBrush, QColor
                            if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                                rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                                rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                            else:
                                rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                                rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                            print(f"[CLEAR_EXCLUSION] Cleared visual exclusion for rectangle {i}")
                
                self._log("✅ Cleared all exclusion states for current image", "info")
            else:
                print(f"[CLEAR_EXCLUSION] No exclusions found for current image")
        
    except Exception as e:
        print(f"[CLEAR_EXCLUSION] Error clearing exclusions: {e}")
        import traceback
        print(f"[CLEAR_EXCLUSION] Traceback: {traceback.format_exc()}")

def _clear_all_exclusions(self):
    """Clear all exclusion flags and reset rectangles to normal appearance"""
    try:
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget.current_image_path:
            print(f"[CLEAR_EXCLUSIONS] No current image")
            return
        
        current_image = self.image_preview_widget.current_image_path
        print(f"[CLEAR_EXCLUSIONS] Clearing exclusions for: {current_image}")
        
        # Clear state
        if hasattr(self, 'image_state_manager'):
            state = self.image_state_manager.get_state(current_image)
            if state is None:
                state = {}
            state['excluded_from_clean'] = []
            self.image_state_manager.set_state(current_image, state)
            print(f"[CLEAR_EXCLUSIONS] Cleared exclusion list in state")
        
        # Reset all rectangles
        if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
            rectangles = self.image_preview_widget.viewer.rectangles
            from PySide6.QtGui import QPen, QBrush, QColor
            
            for i, rect_item in enumerate(rectangles):
                # Clear exclude flag
                rect_item.exclude_from_clean = False
                
                # Reset to normal appearance based on rectangle type
                if hasattr(rect_item, 'is_recognized') and rect_item.is_recognized:
                    # Blue for recognized text
                    rect_item.setPen(QPen(QColor(0, 150, 255), 2))
                    rect_item.setBrush(QBrush(QColor(0, 150, 255, 50)))
                else:
                    # Green for detection boxes
                    rect_item.setPen(QPen(QColor(0, 255, 0), 2))
                    rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                
                print(f"[CLEAR_EXCLUSIONS] Reset rectangle {i} to normal appearance")
        
        self._log("✅ Cleared all exclusions - all rectangles will be included in cleaning", "info")
        print(f"[CLEAR_EXCLUSIONS] All exclusions cleared successfully")
        
    except Exception as e:
        print(f"[CLEAR_EXCLUSIONS] Error: {e}")
        import traceback
        print(f"[CLEAR_EXCLUSIONS] Traceback: {traceback.format_exc()}")

def _relayout_overlay_for_region(self, region_index: int):
    """Re-layout the overlay text to fit the current blue rectangle (auto-resize like pipeline)."""
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(current_image, [])
        target = None
        for g in groups:
            if getattr(g, '_overlay_region_index', None) == region_index:
                target = g
                break
        if target is None:
            return
        # Get rectangle bounds
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        if not (0 <= int(region_index) < len(rects)):
            return
        br = rects[int(region_index)].sceneBoundingRect()
        x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
        if w <= 0 or h <= 0:
            return
        # Get text
        text = None
        try:
            if hasattr(self, '_translation_data') and isinstance(self._translation_data, dict):
                td = self._translation_data.get(int(region_index))
                if td:
                    text = td.get('translation')
        except Exception:
            pass
        if not text:
            text = getattr(target, '_overlay_original_text', '')
        if text is None:
            text = ''
        # Settings with background forced off for source tab
        settings = _get_manga_rendering_settings(self, )
        try:
            settings['show_background'] = False
            settings['bg_opacity'] = 0
        except Exception:
            pass
        # Create new text item sized to current rectangle
        new_text_item, _ = _create_manga_text_item(self, text, x, y, w, h, settings)
        if new_text_item is None:
            return
        viewer = self.image_preview_widget.viewer
        # Replace text item in group with proper cleanup
        try:
            old_text = getattr(target, '_overlay_text_item', None)
            if old_text is not None:
                try:
                    target.removeFromGroup(old_text)
                except Exception:
                    pass
                try:
                    viewer._scene.removeItem(old_text)
                    # Explicitly delete old text item to free memory
                    old_text.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass
        # Add new text to scene and group
        viewer._scene.addItem(new_text_item)
        try:
            target.addToGroup(new_text_item)
        except Exception:
            pass
        try:
            target._overlay_text_item = new_text_item
            target._overlay_bbox_size = (w, h)
        except Exception:
            pass
        # Update transparent overlay rect to new size if present
        try:
            for child in target.childItems():
                if hasattr(child, 'region_index') and getattr(child, 'region_index') == region_index:
                    # This is the transparent overlay rect
                    child.setRect(x, y, w, h)
                    break
        except Exception:
            pass
        # Force scene update
        try:
            viewer._scene.update()
            viewer.viewport().update()
        except Exception:
            pass
    except Exception:
        pass

def _relayout_all_overlays_for_current_image(self):
    """Re-layout all overlays for the current image using current settings."""
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(current_image, [])
        for g in groups:
            idx = getattr(g, '_overlay_region_index', None)
            if idx is not None:
                try:
                    _relayout_overlay_for_region(self, int(idx))
                except Exception:
                    continue
    except Exception:
        pass

def _attach_move_sync_to_rectangle(self, rect_item, region_index: int):
    """Attach a move handler so that when the blue rectangle is moved,
    the corresponding text overlay group moves with it and state is persisted.
    Safe to call multiple times; attaches once per item.
    """
    try:
        # Avoid duplicate attachment
        if getattr(rect_item, '_move_sync_attached', False):
            return
        rect_item._move_sync_attached = True
        # Ensure region_index is present on the item
        rect_item.region_index = region_index
        
        original_release = rect_item.mouseReleaseEvent
        
        def _on_rect_release(ev, r=rect_item, idx=region_index):
            try:
                # Call original release
                try:
                    original_release(ev)
                except Exception:
                    pass
                
                # Desired new top-left in SCENE coordinates (use sceneBoundingRect, not local rect)
                rr = r.sceneBoundingRect()
                new_x, new_y = int(rr.x()), int(rr.y())
                
                # Find the overlay group to move — prefer exact index, else best IoU match
                current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
                groups = overlays_map.get(current_image, [])
                target = None
                for g in groups:
                    if getattr(g, '_overlay_region_index', None) == idx:
                        target = g
                        break
                if target is None:
                    # Fallback to IoU-based match
                    def _iou(a, b):
                        try:
                            ax, ay, aw, ah = a
                            bx, by, bw, bh = b
                            ax2, ay2 = ax + aw, ay + ah
                            bx2, by2 = bx + bw, by + bh
                            x1 = max(ax, bx); y1 = max(ay, by)
                            x2 = min(ax2, bx2); y2 = min(ay2, by2)
                            inter = max(0, x2 - x1) * max(0, y2 - y1)
                            area_a = max(0, aw) * max(0, ah)
                            area_b = max(0, bw) * max(0, bh)
                            den = area_a + area_b - inter
                            return (inter / den) if den > 0 else 0.0
                        except Exception:
                            return 0.0
                    best_iou = 0.0
                    for g in groups:
                        brg = g.sceneBoundingRect()
                        iou = _iou([new_x, new_y, int(rr.width()), int(rr.height())], [int(brg.x()), int(brg.y()), int(brg.width()), int(brg.height())])
                        if iou > best_iou:
                            best_iou = iou
                            target = g
                
                if target is not None:
                    # Compute desired target position preserving any saved offset for this rectangle index
                    try:
                        saved_offsets = {}
                        if hasattr(self, 'image_state_manager') and current_image:
                            st = self.image_state_manager.get_state(current_image) or {}
                            saved_offsets = st.get('overlay_offsets') or {}
                        off = saved_offsets.get(str(int(idx))) or saved_offsets.get(int(idx)) or [0, 0]
                        off_x, off_y = int(off[0]) if isinstance(off, (list, tuple)) and len(off) > 0 else 0, int(off[1]) if isinstance(off, (list, tuple)) and len(off) > 1 else 0
                    except Exception:
                        off_x, off_y = 0, 0
                    br = target.sceneBoundingRect()
                    dx = (new_x + off_x) - int(br.x())
                    dy = (new_y + off_y) - int(br.y())
                    if dx != 0 or dy != 0:
                        try:
                            target.moveBy(dx, dy)
                        except Exception:
                            try:
                                from PySide6.QtCore import QPointF
                                target.setPos(target.pos() + QPointF(dx, dy))
                            except Exception:
                                pass
                        
                        # Toggle overlay visibility based on overlap IoU with original bbox
                        try:
                            def _iou_xywh(a, b):
                                try:
                                    ax, ay, aw, ah = int(a[0]), int(a[1]), int(a[2]), int(a[3])
                                    bx, by, bw, bh = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                                    ax2, ay2 = ax + aw, ay + ah
                                    bx2, by2 = bx + bw, by + bh
                                    x1 = max(ax, bx); y1 = max(ay, by)
                                    x2 = min(ax2, bx2); y2 = min(ay2, by2)
                                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                                    area_a = max(0, aw) * max(0, ah)
                                    area_b = max(0, bw) * max(0, bh)
                                    den = area_a + area_b - inter
                                    return (inter / den) if den > 0 else 0.0
                                except Exception:
                                    return 0.0
                            brg = target.sceneBoundingRect()
                            cur = [int(brg.x()), int(brg.y()), int(brg.width()), int(brg.height())]
                            ob = getattr(target, '_overlay_original_bbox', None)
                            if ob and len(ob) >= 4:
                                overlap = _iou_xywh(cur, ob)
                                target.setVisible(overlap < 0.5)
                            else:
                                target.setVisible(True)
                        except Exception as vis_err:
                            print(f"[DEBUG] Error setting overlay visibility on move: {vis_err}")
                        
                        # Force scene update
                        try:
                            self.image_preview_widget.viewer._scene.update()
                        except Exception:
                            pass
                    
                    # Persist only this region's overlay offset to avoid touching others
                    try:
                        _persist_single_overlay_offset(self, current_image, idx, target)
                    except Exception:
                        pass

                    # Only re-layout if rectangle SIZE changed (width/height), not just position
                    # This avoids expensive text rendering on simple moves
                    try:
                        if hasattr(target, '_overlay_bbox_size'):
                            old_w, old_h = target._overlay_bbox_size
                            new_w, new_h = int(rr.width()), int(rr.height())
                            # Only re-layout if size changed by more than 2 pixels (avoid rounding noise)
                            if abs(new_w - old_w) > 2 or abs(new_h - old_h) > 2:
                                print(f"[PERF] Rectangle {idx} resized ({old_w}x{old_h} -> {new_w}x{new_h}), re-layouting text")
                                _relayout_overlay_for_region(self, idx)
                            # else: just moved, text overlay already moved with it via moveBy()
                    except Exception:
                        pass
                
                # Persist rectangles state
                try:
                    if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                        self.image_preview_widget._persist_rectangles_state()
                except Exception:
                    pass
            except Exception as e:
                print(f"[DEBUG] Rectangle move sync failed: {e}")
        
        rect_item.mouseReleaseEvent = _on_rect_release
    except Exception as e:
        print(f"[DEBUG] Failed to attach move sync: {e}")

def _persist_overlay_offsets_for_current_image(self):
    """Persist overlay offsets relative to the best-matching rectangle for the current image.
    Uses IoU to robustly tie each overlay group to a rectangle, then stores dx,dy per matched index.
    NOTE: Prefer _persist_single_overlay_offset during interactive edits to avoid global remap.
    """
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(current_image, [])
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        offsets = {}
        
        def _iou(a, b):
            try:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ax2, ay2 = ax + aw, ay + ah
                bx2, by2 = bx + bw, by + bh
                x1 = max(ax, bx); y1 = max(ay, by)
                x2 = min(ax2, bx2); y2 = min(ay2, by2)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = max(0, aw) * max(0, ah)
                area_b = max(0, bw) * max(0, bh)
                den = area_a + area_b - inter
                return (inter / den) if den > 0 else 0.0
            except Exception:
                return 0.0
        
        for g in groups:
            try:
                br_g = g.sceneBoundingRect()
                gx, gy, gw, gh = int(br_g.x()), int(br_g.y()), int(br_g.width()), int(br_g.height())
                # Find best rectangle by IoU
                best_idx, best_iou = -1, 0.0
                for i, r in enumerate(rects):
                    br_r = r.sceneBoundingRect()
                    rx, ry, rw, rh = int(br_r.x()), int(br_r.y()), int(br_r.width()), int(br_r.height())
                    iou = _iou([gx, gy, gw, gh], [rx, ry, rw, rh])
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_idx != -1:
                    br_r = rects[best_idx].sceneBoundingRect()
                    dx = int(br_g.x() - br_r.x())
                    dy = int(br_g.y() - br_r.y())
                    offsets[str(best_idx)] = [dx, dy]
            except Exception:
                continue
        if hasattr(self, 'image_state_manager'):
            self.image_state_manager.update_state(current_image, {'overlay_offsets': offsets}, save=True)
            print(f"[STATE] Persisted overlay offsets for {os.path.basename(current_image)}: {len(offsets)} entries")
    except Exception as e:
        print(f"[STATE] Failed to persist overlay offsets: {e}")

def _persist_single_overlay_offset(self, image_path: str, rect_index: int, group):
    """Persist only one overlay offset (dx,dy) for the given rectangle index.
    Avoids recomputing offsets for other overlays to prevent global shifts.
    """
    try:
        if not image_path or rect_index is None:
            return
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        if not (0 <= int(rect_index) < len(rects)):
            return
        br_g = group.sceneBoundingRect()
        br_r = rects[int(rect_index)].sceneBoundingRect()
        dx = int(br_g.x() - br_r.x())
        dy = int(br_g.y() - br_r.y())
        
        # Always keep overlay hidden
        try:
            group.setVisible(False)
        except Exception as vis_err:
            print(f"[STATE] Error updating overlay visibility: {vis_err}")
        
        if hasattr(self, 'image_state_manager'):
            state = self.image_state_manager.get_state(image_path) or {}
            off = state.get('overlay_offsets') or {}
            off[str(int(rect_index))] = [dx, dy]
            self.image_state_manager.update_state(image_path, {'overlay_offsets': off}, save=True)
            print(f"[STATE] Persisted single overlay offset for idx={rect_index}: ({dx},{dy})")
    except Exception as e:
        print(f"[STATE] Failed to persist single overlay offset: {e}")

def _synchronize_overlay_positions_with_rectangles(self, image_path: str):
    """Force synchronization of text overlay positions with their corresponding rectangles.
    This ensures overlays appear properly positioned after state restoration.
    """
    try:
        if not image_path:
            return
        
        viewer = self.image_preview_widget.viewer
        rectangles = getattr(viewer, 'rectangles', []) or []
        
        # Get overlay groups for this image
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        groups = overlays_map.get(image_path, [])
        
        if not groups or not rectangles:
            return
        
        # Load saved overlay offsets
        saved_offsets = {}
        try:
            if hasattr(self, 'image_state_manager'):
                state = self.image_state_manager.get_state(image_path) or {}
                saved_offsets = state.get('overlay_offsets', {})
        except Exception:
            saved_offsets = {}
        
        print(f"[SYNC] Synchronizing {len(groups)} overlay groups with {len(rectangles)} rectangles")
        
        # For each overlay group, find its matching rectangle and sync position
        for group in groups:
            try:
                # Try to get the region index stored on the overlay group
                region_index = getattr(group, '_overlay_region_index', None)
                target_rect = None
                
                # First try direct index match
                if region_index is not None and 0 <= region_index < len(rectangles):
                    target_rect = rectangles[region_index]
                else:
                    # Fall back to IoU-based matching
                    best_iou = 0.0
                    group_rect = group.sceneBoundingRect()
                    gx, gy, gw, gh = int(group_rect.x()), int(group_rect.y()), int(group_rect.width()), int(group_rect.height())
                    
                    for idx, rect in enumerate(rectangles):
                        rect_bounds = rect.sceneBoundingRect()
                        rx, ry, rw, rh = int(rect_bounds.x()), int(rect_bounds.y()), int(rect_bounds.width()), int(rect_bounds.height())
                        
                        # Calculate IoU
                        iou = _calculate_iou(self, [gx, gy, gw, gh], [rx, ry, rw, rh])
                        if iou > best_iou:
                            best_iou = iou
                            target_rect = rect
                            region_index = idx
                
                if target_rect is not None and region_index is not None:
                    # Get saved offset for this region
                    offset_key = str(region_index)
                    saved_offset = saved_offsets.get(offset_key, [0, 0])
                    if isinstance(saved_offset, (list, tuple)) and len(saved_offset) >= 2:
                        off_x, off_y = int(saved_offset[0]), int(saved_offset[1])
                    else:
                        off_x, off_y = 0, 0
                    
                    # Calculate desired position
                    rect_bounds = target_rect.sceneBoundingRect()
                    desired_x = int(rect_bounds.x()) + off_x
                    desired_y = int(rect_bounds.y()) + off_y
                    
                    # Get current position
                    group_bounds = group.sceneBoundingRect()
                    current_x = int(group_bounds.x())
                    current_y = int(group_bounds.y())
                    
                    # Calculate movement needed
                    dx = desired_x - current_x
                    dy = desired_y - current_y
                    
                    # Apply movement if needed
                    if abs(dx) > 1 or abs(dy) > 1:  # Only move if significant difference
                        try:
                            group.moveBy(dx, dy)
                            print(f"[SYNC] Moved overlay for region {region_index} by ({dx}, {dy})")
                        except Exception:
                            try:
                                from PySide6.QtCore import QPointF
                                group.setPos(group.pos() + QPointF(dx, dy))
                                print(f"[SYNC] Set overlay position for region {region_index} with offset ({dx}, {dy})")
                            except Exception as e:
                                print(f"[SYNC] Failed to move overlay for region {region_index}: {e}")
                    
                    # Always keep overlay hidden
                    if hasattr(group, 'setVisible'):
                        group.setVisible(False)
                    
                    # Update the region index on the group if not set
                    if not hasattr(group, '_overlay_region_index'):
                        group._overlay_region_index = region_index
                
            except Exception as e:
                print(f"[SYNC] Failed to sync overlay group: {e}")
        
        # Force scene update after all synchronization
        try:
            viewer._scene.update()
            print(f"[SYNC] Overlay synchronization completed for {os.path.basename(image_path)}")
        except Exception:
            pass
            
    except Exception as e:
        print(f"[SYNC] Overlay synchronization failed: {e}")

def _calculate_iou(self, box1, box2):
    """Calculate Intersection over Union for two bounding boxes [x, y, w, h]"""
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0

def _show_ocr_popup(self, ocr_text: str, region_index: int = None):
    """Show OCR text in a popup dialog with edit capability"""
    try:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        
        dialog = QDialog(self.image_preview_widget)
        dialog.setWindowTitle("📝 OCR Recognition Result")
        dialog.resize(400, 250)
        dialog.setModal(True)
        
        # Apply dark theme styling
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #5a9fd4;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                selection-background-color: #5a9fd4;
            }
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton#save_btn {
                background-color: #28a745;
            }
            QPushButton#save_btn:hover {
                background-color: #34ce57;
            }
            QLabel {
                color: white;
                font-weight: bold;
                margin-bottom: 8px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title label
        title_label = QLabel("Recognized Text (editable):")
        layout.addWidget(title_label)
        
        # Text display - EDITABLE
        text_edit = QTextEdit()
        text_edit.setPlainText(ocr_text)
        text_edit.setReadOnly(False)  # Make it editable
        
        # Apply user's selected font from manga settings
        font_name = getattr(self, 'font_style_value', 'Comic Sans MS Bold')
        if font_name == 'Default':
            font_name = 'Comic Sans MS Bold'  # Use Comic Sans Bold as default
        edit_font = QFont(font_name, 11)  # 11pt size for readability in dialog
        text_edit.setFont(edit_font)
        
        layout.addWidget(text_edit)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Save button
        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("save_btn")
        def save_changes():
            new_text = text_edit.toPlainText()
            if new_text != ocr_text and region_index is not None:
                # Update the stored recognition data (for context menu)
                if hasattr(self, '_recognition_data') and region_index in self._recognition_data:
                    self._recognition_data[region_index]['text'] = new_text
                
                # Update _recognized_texts (for translation button)
                if hasattr(self, '_recognized_texts'):
                    for text_data in self._recognized_texts:
                        if text_data.get('region_index') == region_index:
                            text_data['text'] = new_text
                            print(f"[DEBUG] Updated _recognized_texts for region {region_index}")
                            break
                
                # Update translation data if it exists (or create it)
                if not hasattr(self, '_translation_data'):
                    self._translation_data = {}
                if region_index not in self._translation_data:
                    # Initialize entry if it doesn't exist yet
                    self._translation_data[region_index] = {
                        'original': new_text,
                        'translation': ''  # Will be filled when translation happens
                    }
                else:
                    self._translation_data[region_index]['original'] = new_text
                
                print(f"[DEBUG] Updated OCR text for region {region_index}: '{new_text[:50]}...'")
            dialog.accept()
        save_btn.clicked.connect(save_changes)
        button_layout.addWidget(save_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
        
    except Exception as e:
        print(f"[DEBUG] Error showing OCR popup: {str(e)}")

def _show_translation_popup(self, translation_data: dict, region_index: int = None):
    """Show translation in a popup dialog with edit capability"""
    try:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QFrame
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        
        original = translation_data['original']
        translation = translation_data['translation']
        
        dialog = QDialog(self.image_preview_widget)
        dialog.setWindowTitle("🌍 Translation Result")
        dialog.resize(500, 380)
        dialog.setModal(True)
        
        # Apply dark theme styling
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #5a9fd4;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                selection-background-color: #5a9fd4;
            }
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton#save_btn {
                background-color: #28a745;
            }
            QPushButton#save_btn:hover {
                background-color: #34ce57;
            }
            QLabel {
                color: white;
                font-weight: bold;
                margin-bottom: 4px;
            }
            QFrame {
                border: none;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Original text section
        orig_label = QLabel("Original Text (editable):")
        layout.addWidget(orig_label)
        
        orig_text = QTextEdit()
        orig_text.setPlainText(original)
        orig_text.setReadOnly(False)  # Make it editable
        orig_text.setMaximumHeight(100)
        
        # Apply user's selected font from manga settings
        font_name = getattr(self, 'font_style_value', 'Comic Sans MS Bold')
        if font_name == 'Default':
            font_name = 'Comic Sans MS Bold'  # Use Comic Sans Bold as default
        edit_font = QFont(font_name, 11)  # 11pt size for readability in dialog
        orig_text.setFont(edit_font)
        
        layout.addWidget(orig_text)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #5a9fd4;")
        layout.addWidget(separator)
        
        # Translation section
        trans_label = QLabel("Translation (editable):")
        layout.addWidget(trans_label)
        
        trans_text = QTextEdit()
        trans_text.setPlainText(translation)
        trans_text.setReadOnly(False)  # Make it editable
        trans_text.setMaximumHeight(100)
        
        # Apply user's selected font from manga settings
        font_name = getattr(self, 'font_style_value', 'Comic Sans MS Bold')
        if font_name == 'Default':
            font_name = 'Comic Sans MS Bold'  # Use Comic Sans Bold as default
        edit_font = QFont(font_name, 11)  # 11pt size for readability in dialog
        trans_text.setFont(edit_font)
        
        layout.addWidget(trans_text)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Save button
        save_btn = QPushButton("💾 Save & Update Overlay")
        save_btn.setObjectName("save_btn")
        def save_changes():
            new_original = orig_text.toPlainText()
            new_translation = trans_text.toPlainText()
            changed = False
            
            if region_index is not None:
                # Update the stored data
                if new_original != original:
                    if hasattr(self, '_recognition_data') and region_index in self._recognition_data:
                        self._recognition_data[region_index]['text'] = new_original
                    if hasattr(self, '_translation_data') and region_index in self._translation_data:
                        self._translation_data[region_index]['original'] = new_original
                    changed = True
                    print(f"[DEBUG] Updated original text for region {region_index}")
                
                if new_translation != translation:
                    if hasattr(self, '_translation_data') and region_index in self._translation_data:
                        self._translation_data[region_index]['translation'] = new_translation
                    changed = True
                    print(f"[DEBUG] Updated translation for region {region_index}")
                
                # Refresh the text overlay for this region
                if changed:
                    try:
        # Persist updated translated_texts to state so overlays restore across sessions
                        try:
                            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                            
                            # CRITICAL: Validate that we're editing the correct image's translation
                            # Check if the translation data belongs to the currently displayed image
                            translation_image_path = None
                            if hasattr(self, '_translating_image_path'):
                                translation_image_path = self._translating_image_path
                            elif hasattr(self, '_translation_data_image_path'):
                                translation_image_path = self._translation_data_image_path
                            
                            # If translation belongs to a different image, abort with clear error
                            if translation_image_path and current_image and os.path.abspath(translation_image_path) != os.path.abspath(current_image):
                                error_msg = f"⚠️ Cannot update: Translation is for {os.path.basename(translation_image_path)} but you're viewing {os.path.basename(current_image)}"
                                self._log(error_msg, "error")
                                print(f"[CRITICAL] {error_msg}")
                                dialog.accept()
                                return
                            
                            if current_image and hasattr(self, 'image_state_manager') and self.image_state_manager:
                                state = self.image_state_manager.get_state(current_image) or {}
                                tlist = state.get('translated_texts') or []
                                # Ensure list size
                                if len(tlist) <= int(region_index):
                                    tlist = list(tlist) + [{} for _ in range(int(region_index) + 1 - len(tlist))]
                                # Determine bbox for this region
                                bbox = None
                                try:
                                    if hasattr(self, '_recognition_data') and int(region_index) in self._recognition_data:
                                        bbox = self._recognition_data[int(region_index)].get('bbox')
                                except Exception:
                                    bbox = None
                                if not bbox:
                                    try:
                                        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                                        if 0 <= int(region_index) < len(rects):
                                            br = rects[int(region_index)].sceneBoundingRect()
                                            bbox = [int(br.x()), int(br.y()), int(br.width()), int(br.height())]
                                    except Exception:
                                        bbox = [0, 0, 100, 100]
                                tlist[int(region_index)] = {
                                    'original': {'text': new_original, 'region_index': int(region_index)},
                                    'translation': new_translation,
                                    'bbox': bbox or [0, 0, 100, 100]
                                }
                                state['translated_texts'] = tlist
                                self.image_state_manager.set_state(current_image, state, save=True)
                        except Exception as persist_err:
                            self._log(f"⚠️ Failed to persist translation change: {persist_err}", "warning")
                        
                        # Animate button during async operation
                        old_text = save_btn.text()
                        save_btn.setEnabled(False)
                        save_btn.setText("Saving…")
                        
                        # Use the async method that utilizes ThreadPoolExecutor
                        # This handles processing overlay internally
                        _save_overlay_async(self, region_index, new_translation)
                    finally:
                        # Restore button state
                        try:
                            save_btn.setText(old_text)
                            save_btn.setEnabled(True)
                        except Exception:
                            pass
            
            dialog.accept()
        save_btn.clicked.connect(save_changes)
        button_layout.addWidget(save_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
        
    except Exception as e:
        print(f"[DEBUG] Error showing translation popup: {str(e)}")

def _handle_translate_this_text(self, region_index: int, prompt: str):
    """Handle the 'Translate This Text' menu action
    
    Gets the recognized text and sends it for translation using the unified API client.
    """
    try:
        # Get the recognized text for this region
        if not hasattr(self, '_recognition_data') or region_index not in self._recognition_data:
            self._log(f"❌ No recognition data for region {region_index}", "error")
            return
        
        recognized_text = self._recognition_data[region_index]['text']
        if not recognized_text.strip():
            self._log(f"❌ Recognition text is empty for region {region_index}", "error")
            return
        
        # Build the full prompt combining template + recognized text
        full_message = f"{prompt}\n\n{recognized_text}"
        
        self._log(f"🌍 Translating text from region {region_index}...", "info")
        
        # Start pulse effect on the rectangle
        try:
            if hasattr(self.image_preview_widget, 'viewer') and hasattr(self.image_preview_widget.viewer, 'rectangles'):
                rectangles = self.image_preview_widget.viewer.rectangles
                if 0 <= region_index < len(rectangles):
                    rect_item = rectangles[region_index]
                    _add_rectangle_pulse_effect(self, rect_item, region_index)
        except Exception as pulse_err:
            print(f"[TRANSLATE] Error starting pulse effect: {pulse_err}")
        
        # Run translation in background thread
        import threading
        thread = threading.Thread(
            target=_translate_this_text_background,
            args=(self, full_message, region_index),
            daemon=True
        )
        thread.start()
    
    except Exception as e:
        # Stop pulse effect on error
        try:
            if 'region_index' in locals() and hasattr(self.image_preview_widget, 'viewer'):
                rectangles = self.image_preview_widget.viewer.rectangles
                if 0 <= region_index < len(rectangles):
                    rect_item = rectangles[region_index]
                    _remove_rectangle_pulse_effect(self, rect_item, region_index)
        except Exception as pulse_err:
            print(f"[TRANSLATE_HANDLER_ERROR] Error stopping pulse effect: {pulse_err}")
        
        self._log(f"❌ Error in translate this text: {e}", "error")
        print(f"[TRANSLATE] Error traceback: {e}")
        import traceback
        traceback.print_exc()

def _translate_this_text_background(self, message: str, region_index: int):
    """Send text for translation using MangaTranslator in background"""
    try:
        # Get API configuration from main GUI
        api_key = None
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
        
        if not api_key:
            self._log("❌ No API key configured", "error")
            return
        
        # Get model
        model = 'gpt-4o-mini'  # default
        if hasattr(self.main_gui, 'model_var'):
            try:
                if hasattr(self.main_gui.model_var, 'get'):
                    model = self.main_gui.model_var.get()
                else:
                    model = self.main_gui.model_var
            except Exception:
                pass
        elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
            model = self.main_gui.config.get('model')
        
        # Apply GUI environment variables like Start Translation does
        from unified_api_client import UnifiedClient
        import os, json
        
        # Apply temperature
        if hasattr(self.main_gui, 'trans_temp'):
            try:
                if hasattr(self.main_gui.trans_temp, 'text'):
                    temp = self.main_gui.trans_temp.text()
                elif hasattr(self.main_gui.trans_temp, 'get'):
                    temp = self.main_gui.trans_temp.get()
                else:
                    temp = '0.3'
                os.environ['TRANSLATION_TEMPERATURE'] = str(temp)
            except Exception:
                os.environ['TRANSLATION_TEMPERATURE'] = '0.3'
        
        # Apply delay
        if hasattr(self.main_gui, 'delay_entry'):
            try:
                if hasattr(self.main_gui.delay_entry, 'text'):
                    delay = self.main_gui.delay_entry.text()
                elif hasattr(self.main_gui.delay_entry, 'get'):
                    delay = self.main_gui.delay_entry.get()
                else:
                    delay = '1.0'
                os.environ['SEND_INTERVAL_SECONDS'] = str(delay)
            except Exception:
                os.environ['SEND_INTERVAL_SECONDS'] = '1.0'
        
        # Create unified client with main GUI config
        unified_client = UnifiedClient(model=model, api_key=api_key)
        
        # Get token limit from manga settings
        max_tokens = 2048  # default
        try:
            manga_settings = self.main_gui.config.get('manga_settings', {}) or {}
            manual_edit = manga_settings.get('manual_edit', {}) or {}
            ttt_tokens = int(manual_edit.get('translate_this_text_tokens', 2048))
            
            if ttt_tokens <= 0:
                # Use manga output token limit, or fall back to main GUI limit
                manga_limit = int(manual_edit.get('manga_output_token_limit', -1))
                if manga_limit > 0:
                    max_tokens = manga_limit
                else:
                    max_tokens = int(getattr(self.main_gui, 'max_output_tokens', 4000))
            else:
                max_tokens = ttt_tokens
        except Exception:
            max_tokens = 2048
        
        temperature = float(os.environ.get('TRANSLATION_TEMPERATURE', 0.3))
        self._log(f"📤 Sending to API ({model})...", "info")
        print(f"[TRANSLATE_THIS] Temperature: {temperature}, Max tokens: {max_tokens}")
        
        # Use unified_client.send() method which returns (content, finish_reason)
        translation_result, finish_reason = unified_client.send(
            messages=[{"role": "user", "content": message}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Check if we got a valid translation
        if not translation_result or not translation_result.strip():
            self._log(f"❌ Empty response from API for region {region_index}", "error")
            return
        
        # Store the translation result
        if not hasattr(self, '_translation_data'):
            self._translation_data = {}
        
        # Get original text and bbox from recognition data
        original_text = self._recognition_data[region_index]['text']
        bbox = self._recognition_data[region_index].get('bbox', [0, 0, 100, 100])
        
        self._translation_data[region_index] = {
            'original': original_text,
            'translation': translation_result
        }
        
        self._log(f"✅ Translation complete for region {region_index}", "success")
        
        # Defer ALL GUI updates to the main thread to avoid Qt timer/thread warnings
        # Post translation result to main thread via queue where overlay will be added
        self.update_queue.put(('translate_this_text_result', {
            'original_text': original_text,
            'translation_result': translation_result,
            'region_index': region_index,
            'bbox': bbox
        }))
    
    except Exception as e:
        # Stop pulse effect on error
        try:
            if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'viewer'):
                rectangles = self.image_preview_widget.viewer.rectangles
                if 0 <= region_index < len(rectangles):
                    rect_item = rectangles[region_index]
                    _remove_rectangle_pulse_effect(self, rect_item, region_index)
        except Exception as pulse_err:
            print(f"[TRANSLATE_ERROR] Error stopping pulse effect: {pulse_err}")
        
        self._log(f"❌ API translation failed: {e}", "error")
        import traceback
        traceback.print_exc()

def _clean_up_deleted_rectangle_overlays(self, region_index: int):
    """Clean up text overlays and data associated with a deleted rectangle"""
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            print(f"[CLEANUP] No current image, skipping cleanup for region {region_index}")
            return
        
        print(f"[CLEANUP] Starting cleanup for region {region_index} on image {os.path.basename(current_image)}")
        
        # Remove from recognition data
        if hasattr(self, '_recognition_data') and region_index in self._recognition_data:
            del self._recognition_data[region_index]
            print(f"[DELETE] Removed recognition data for region {region_index}")
        
        # Remove from translation data
        if hasattr(self, '_translation_data') and region_index in self._translation_data:
            del self._translation_data[region_index]
            print(f"[DELETE] Removed translation data for region {region_index}")
        
        # Remove associated text overlays from the scene
        overlays_map = getattr(self, '_text_overlays_by_image', {}) or {}
        current_overlays = overlays_map.get(current_image, [])
        
        # Find and remove overlay groups that match this region index
        overlays_to_remove = []
        for overlay_group in current_overlays:
            if getattr(overlay_group, '_overlay_region_index', None) == region_index:
                overlays_to_remove.append(overlay_group)
        
        # Remove overlays from scene and tracking
        for overlay_group in overlays_to_remove:
            try:
                self.image_preview_widget.viewer._scene.removeItem(overlay_group)
                current_overlays.remove(overlay_group)
                print(f"[DELETE] Removed text overlay for region {region_index}")
            except Exception as e:
                print(f"[DELETE] Error removing overlay: {e}")
        
        # Update the overlay map
        overlays_map[current_image] = current_overlays
        
        # Clear state data for this region from persistence
        if hasattr(self, 'image_state_manager') and self.image_state_manager:
            try:
                state = self.image_state_manager.get_state(current_image) or {}
                
                # Remove overlay offsets for this region
                overlay_offsets = state.get('overlay_offsets', {})
                overlay_offsets.pop(str(region_index), None)
                overlay_offsets.pop(region_index, None)
                
                # Remove from recognized texts if present
                recognized_texts = state.get('recognized_texts', [])
                if recognized_texts and region_index < len(recognized_texts):
                    # Mark as deleted rather than removing to preserve indices
                    if region_index < len(recognized_texts):
                        recognized_texts[region_index] = {'deleted': True}
                
                # CRITICAL: Also remove from translated_texts which are restored on image load
                translated_texts = state.get('translated_texts', [])
                if translated_texts and region_index < len(translated_texts):
                    # Mark as deleted rather than removing to preserve indices
                    if region_index < len(translated_texts):
                        translated_texts[region_index] = {'deleted': True}
                
                # Update state
                state['overlay_offsets'] = overlay_offsets
                state['recognized_texts'] = recognized_texts
                state['translated_texts'] = translated_texts
                
                self.image_state_manager.set_state(current_image, state, save=True)
                print(f"[DELETE] Cleaned up persisted state for region {region_index}")
            except Exception as e:
                print(f"[DELETE] Error cleaning persisted state: {e}")
        
    except Exception as e:
        print(f"[DELETE] Error cleaning up deleted rectangle overlays: {e}")

def clear_text_overlays_for_image(self, image_path: str = None):
    """Clear text overlays for a specific image (or all if no path given) with proper Qt cleanup"""
    try:
        viewer = self.image_preview_widget.viewer
        
        # Initialize overlay dictionary if not exists
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        
        if image_path is None:
            # Clear all overlays from all images with proper cleanup
            for overlays in self._text_overlays_by_image.values():
                for overlay in overlays:
                    try:
                        # Remove from scene first
                        viewer._scene.removeItem(overlay)
                        # Destroy all child items explicitly
                        for child in overlay.childItems():
                            try:
                                child.setParentItem(None)
                                child.deleteLater()
                            except Exception:
                                pass
                        # Destroy the group itself
                        overlay.deleteLater()
                    except Exception:
                        pass
            self._text_overlays_by_image = {}
            print("[DEBUG] Cleared all text overlays for all images with Qt cleanup")
        else:
            # Clear overlays for specific image with proper cleanup
            if image_path in self._text_overlays_by_image:
                for overlay in self._text_overlays_by_image[image_path]:
                    try:
                        # Remove from scene first
                        viewer._scene.removeItem(overlay)
                        # Destroy all child items explicitly
                        for child in overlay.childItems():
                            try:
                                child.setParentItem(None)
                                child.deleteLater()
                            except Exception:
                                pass
                        # Destroy the group itself
                        overlay.deleteLater()
                    except Exception:
                        pass
                del self._text_overlays_by_image[image_path]
                print(f"[DEBUG] Cleared text overlays for image with Qt cleanup: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"[DEBUG] Error clearing text overlays: {e}")

def show_text_overlays_for_image(self, image_path: str):
    """Keep text overlays for a specific image (but always hidden)"""
    try:
        viewer = self.image_preview_widget.viewer
        
        # Initialize overlay dictionary if not exists
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        
        # Always hide all overlays (overlays exist but are invisible)
        for overlays in self._text_overlays_by_image.values():
            for overlay in overlays:
                overlay.setVisible(False)
        
        # Keep overlays for the requested image (but hidden)
        if image_path in self._text_overlays_by_image:
            # Don't show them - keep them hidden
            for overlay in self._text_overlays_by_image[image_path]:
                overlay.setVisible(False)
            # Force scene update
            viewer._scene.update()
            viewer.update()
            print(f"[DEBUG] Kept {len(self._text_overlays_by_image[image_path])} overlays hidden for image: {os.path.basename(image_path)}")
        else:
            #print(f"[DEBUG] No overlays found for image: {os.path.basename(image_path)}")
            pass
    except Exception as e:
        print(f"[DEBUG] Error hiding text overlays: {str(e)}")

def _alias_text_overlays_for_image(self, from_path: str, to_path: str):
    """Alias the overlays list from one image path to another (e.g., original -> cleaned)"""
    try:
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        if from_path in self._text_overlays_by_image:
            self._text_overlays_by_image[to_path] = self._text_overlays_by_image[from_path]
            print(f"[DEBUG] Aliased overlays from {os.path.basename(from_path)} to {os.path.basename(to_path)}")
    except Exception as e:
        print(f"[DEBUG] Error aliasing overlays: {str(e)}")

def _save_position_async(self, region_index: int):
    """Save position and update overlay for a specific region using thread pool executor with microsecond locks.
    
    Enhanced version that supports parallel processing with race condition protection.
    """
    try:
        # Initialize parallel processing system if not exists
        if not hasattr(self, '_parallel_save_system'):
            _init_parallel_save_system(self, )
        
        # Submit to parallel processing queue
        self._parallel_save_system.queue_save_task(region_index)
        
    except Exception as e:
        print(f"[PARALLEL] Error in _save_position_async: {e}")
        # Fallback to single region processing
        _fallback_single_save(self, region_index)

def _schedule_source_refresh(self):
    """Instant refresh of source preview - called from completion callback."""
    try:
        print(f"[REFRESH] Starting instant preview refresh...")
        _do_source_refresh(self)
        print(f"[REFRESH] Instant refresh completed")
    except Exception as e:
        print(f"[REFRESH] Refresh failed: {e}")
        import traceback
        traceback.print_exc()

def _do_source_refresh(self):
    """Actually perform the source refresh (must be called from main thread)
    
    This reloads the rendered/translated image to show updates from auto-save position.
    """
    try:
        ipw = getattr(self, 'image_preview_widget', None)
        if not ipw:
            print(f"[REFRESH] No image_preview_widget available")
            return
            
        current_image = getattr(ipw, 'current_image_path', None)
        if not current_image:
            print(f"[REFRESH] No current_image_path available")
            return
        
        print(f"[REFRESH] Executing preview refresh for: {os.path.basename(current_image)}")
        
        # Get the rendered image path from state
        rendered_path = None
        try:
            if hasattr(self, 'image_state_manager') and self.image_state_manager:
                state = self.image_state_manager.get_state(current_image) or {}
                rendered_path = state.get('rendered_image_path')
                if rendered_path and os.path.exists(rendered_path):
                    print(f"[REFRESH] Found rendered image: {os.path.basename(rendered_path)}")
                else:
                    print(f"[REFRESH] No valid rendered image in state")
                    rendered_path = None
        except Exception as e:
            print(f"[REFRESH] Error getting rendered path from state: {e}")
        
        # If no rendered path, check the rendered images map
        if not rendered_path and hasattr(self, '_rendered_images_map'):
            rendered_path = self._rendered_images_map.get(current_image)
            if rendered_path and os.path.exists(rendered_path):
                print(f"[REFRESH] Found rendered image in map: {os.path.basename(rendered_path)}")
            else:
                rendered_path = None
        
        # If still no rendered path, search in override directory and source directory
        if not rendered_path:
            source_dir = os.path.dirname(current_image)
            source_filename = os.path.basename(current_image)
            base_name = os.path.splitext(source_filename)[0]
            
            # Check for OUTPUT_DIRECTORY override
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            # Build search paths - check override first if set
            search_paths = []
            if override_dir:
                search_paths.append(os.path.join(override_dir, f"{base_name}_translated", source_filename))
            search_paths.append(os.path.join(source_dir, f"{base_name}_translated", source_filename))
            
            for path in search_paths:
                if os.path.exists(path):
                    rendered_path = path
                    print(f"[REFRESH] Found rendered image via directory search: {os.path.basename(rendered_path)}")
                    break
        
        # Reload the appropriate image
        if rendered_path:
            print(f"[REFRESH] Loading rendered image: {rendered_path}")
            ipw.load_image(rendered_path, preserve_rectangles=True, preserve_text_overlays=False)
        else:
            print(f"[REFRESH] No rendered image found, reloading source: {current_image}")
            ipw.load_image(current_image, preserve_rectangles=True, preserve_text_overlays=True)
        
        # Rehydrate text state
        try:
            ocr_count, trans_count = _rehydrate_text_state_from_persisted(self, current_image)
            if ocr_count and hasattr(self, '_update_rectangles_with_recognition'):
                _update_rectangles_with_recognition(self, self._recognized_texts)
        except Exception as e:
            print(f"[REFRESH] Error rehydrating state: {e}")
        
        print(f"[REFRESH] Preview refresh completed")
        
    except Exception as _e:
        print(f"[REFRESH] Preview refresh failed: {_e}")
        import traceback
        traceback.print_exc()

def _init_parallel_save_system(self):
    """Initialize the parallel save processing system with ProcessPoolExecutor."""
    try:
        import threading
        import time
        from queue import Queue
        from concurrent.futures import ThreadPoolExecutor
        
        class ParallelSaveSystem:
            def __init__(self, parent):
                self.parent = parent
                self.pending_tasks = Queue()
                self.active_tasks = set()  # Track active region indices
                self.microsecond_lock = threading.Lock()  # Microsecond precision lock
                
                # Get max_workers from manga settings
                max_workers = 3  # Default - lower for threads since they're lighter
                try:
                    if hasattr(parent, 'main_gui') and hasattr(parent.main_gui, 'config'):
                        manga_settings = parent.main_gui.config.get('manga_settings', {})
                        advanced_settings = manga_settings.get('advanced', {})
                        max_workers = advanced_settings.get('max_workers', 3)
                        print(f"[PARALLEL] Using max_workers from settings: {max_workers}")
                    else:
                        print(f"[PARALLEL] Using default max_workers: {max_workers}")
                except Exception as e:
                    print(f"[PARALLEL] Error reading max_workers from settings, using default: {e}")
                
                # Use ThreadPoolExecutor for instant response (no process startup overhead)
                self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AutoSave")
                self.processing_count = 0  # Track number of active threads
                self.count_lock = threading.Lock()  # Lock for processing count
                
                # Start the coordinator thread
                self.coordinator_thread = threading.Thread(
                    target=self._coordinate_tasks, 
                    daemon=True, 
                    name="AutoSaveCoordinator"
                )
                self.coordinator_thread.start()
                print(f"[PARALLEL] Initialized parallel save system with {max_workers} worker threads (ThreadPoolExecutor - instant response)")
            
            def queue_save_task(self, region_index: int):
                """Queue a save task for the given region index."""
                try:
                    timestamp = time.time_ns()  # Microsecond precision timestamp
                    task = {
                        'region_index': region_index,
                        'timestamp': timestamp,
                        'retry_count': 0
                    }
                    
                    with self.microsecond_lock:
                        # Check if this region is already being processed
                        if region_index in self.active_tasks:
                            print(f"[PARALLEL] Region {region_index} already being processed, skipping")
                            return False
                        
                        # Add to pending tasks
                        self.pending_tasks.put(task)
                        print(f"[PARALLEL] Queued save task for region {region_index} at {timestamp}")
                        return True
                        
                except Exception as e:
                    print(f"[PARALLEL] Error queuing task for region {region_index}: {e}")
                    return False
            
            def queue_batch_save_tasks(self, region_indices: list) -> int:
                """Queue multiple save tasks for batch processing optimization.
                
                Args:
                    region_indices: List of region indices to queue for processing
                    
                Returns:
                    Number of tasks successfully queued
                """
                if not region_indices:
                    return 0
                    
                queued_count = 0
                batch_timestamp = time.time_ns()
                
                with self.microsecond_lock:
                    for region_index in region_indices:
                        # Check if this region is already being processed
                        if region_index in self.active_tasks:
                            print(f"[PARALLEL] Region {region_index} already being processed, skipping from batch")
                            continue
                        
                        # Create task with batch timestamp
                        task = {
                            'region_index': region_index,
                            'timestamp': batch_timestamp,
                            'retry_count': 0,
                            'batch_id': batch_timestamp  # Add batch identifier
                        }
                        
                        # Add to pending tasks
                        self.pending_tasks.put(task)
                        queued_count += 1
                
                print(f"[PARALLEL] Batch queued {queued_count}/{len(region_indices)} save tasks")
                return queued_count
            
            def _coordinate_tasks(self):
                """Coordinate parallel task execution with microsecond precision."""
                print(f"[PARALLEL] Coordinator thread started")
                
                while True:
                    try:
                        # Wait for a task (blocking)
                        task = self.pending_tasks.get(timeout=1.0)
                        
                        region_index = task['region_index']
                        
                        # Acquire microsecond lock
                        with self.microsecond_lock:
                            # Double-check the region isn't already active
                            if region_index in self.active_tasks:
                                print(f"[PARALLEL] Region {region_index} became active while waiting, skipping")
                                continue
                            
                            # Mark region as active
                            self.active_tasks.add(region_index)
                            
                            # Increment processing count
                            with self.count_lock:
                                self.processing_count += 1
                                print(f"[PARALLEL] Started processing region {region_index} (active: {self.processing_count})")
                                
                                # Update button state for parallel processing
                                _update_parallel_save_button_state(self.parent, self.processing_count)
                        
                        # Submit the work to thread pool (threads can access parent directly)
                        region_idx = task['region_index']
                        
                        # Get translation text
                        trans_text = _get_translation_text_for_region(self.parent, region_idx)
                        if not trans_text:
                            print(f"[PARALLEL] No translation text for region {region_idx}, skipping")
                            # Clean up
                            with self.microsecond_lock:
                                self.active_tasks.discard(region_idx)
                                with self.count_lock:
                                    self.processing_count = max(0, self.processing_count - 1)
                            continue
                        
                        # Submit to ThreadPoolExecutor (instant - no process startup)
                        future = self.executor.submit(self._execute_save_task, region_idx, trans_text)
                        
                        # Add callback to handle completion
                        future.add_done_callback(lambda f, idx=region_idx: self._handle_task_completion(f, idx))
                        
                        # Don't wait for completion - let it run in parallel
                        
                    except Empty:
                        continue
                    except Exception as e:
                        print(f"[PARALLEL] Coordinator error: {e}")
            
            def _execute_save_task(self, region_index, trans_text):
                """Execute save task in worker thread (instant - no process startup)."""
                try:
                    print(f"[PARALLEL] Thread worker processing region {region_index}")
                    # Do the actual GUI update (thread-safe via _update_single_text_overlay_parallel)
                    success = _update_single_text_overlay_parallel(self.parent, region_index, trans_text)
                    if success:
                        print(f"[PARALLEL] Successfully updated region {region_index}")
                    return {'success': success, 'region_index': region_index}
                except Exception as e:
                    print(f"[PARALLEL] Thread worker error for region {region_index}: {e}")
                    return {'success': False, 'region_index': region_index}
            
            def _handle_task_completion(self, future, region_index):
                """Handle completion of a save task from ThreadPoolExecutor."""
                try:
                    result = future.result(timeout=5.0)
                    success = result.get('success', False) if isinstance(result, dict) else False
                    
                    if not success:
                        print(f"[PARALLEL] Task failed for region {region_index}")
                        
                except Exception as e:
                    print(f"[PARALLEL] Error handling completion for region {region_index}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always clean up
                    with self.microsecond_lock:
                        self.active_tasks.discard(region_index)
                        
                        with self.count_lock:
                            self.processing_count = max(0, self.processing_count - 1)
                            print(f"[PARALLEL] Finished processing region {region_index} (active: {self.processing_count})")
                            
                            # Update button state
                            _update_parallel_save_button_state(self.parent, self.processing_count)
            
            def get_active_count(self):
                """Get the number of currently active processing tasks."""
                with self.count_lock:
                    return self.processing_count
            
            def queue_batch_save_tasks(self, region_indices: list) -> int:
                """Queue multiple save tasks for batch processing optimization.
                
                Args:
                    region_indices: List of region indices to queue for saving
                    
                Returns:
                    Number of tasks successfully queued
                """
                if not region_indices:
                    return 0
                    
                queued_count = 0
                batch_timestamp = time.time_ns()
                
                with self.microsecond_lock:
                    for region_index in region_indices:
                        # Check if this region is already being processed
                        if region_index in self.active_tasks:
                            print(f"[PARALLEL] Region {region_index} already being processed, skipping from batch")
                            continue
                        
                        # Create task with batch timestamp
                        task = {
                            'region_index': region_index,
                            'timestamp': batch_timestamp,
                            'retry_count': 0,
                            'batch_id': batch_timestamp  # Add batch identifier
                        }
                        
                        # Add to pending tasks
                        self.pending_tasks.put(task)
                        queued_count += 1
                
                print(f"[PARALLEL] Batch queued {queued_count}/{len(region_indices)} save tasks")
                return queued_count
            
            def shutdown(self):
                """Shutdown the parallel processing system."""
                try:
                    print(f"[PARALLEL] Shutting down executor (waiting for tasks to complete)...")
                    # Wait for tasks to complete with short timeout
                    self.executor.shutdown(wait=True, cancel_futures=True)
                    print(f"[PARALLEL] Parallel save system shutdown completed")
                except Exception as e:
                    print(f"[PARALLEL] Error during shutdown: {e}")
                    pass
        
        self._parallel_save_system = ParallelSaveSystem(self)
        
    except Exception as e:
        print(f"[PARALLEL] Failed to initialize parallel save system: {e}")
        self._parallel_save_system = None

def _save_positions_batch(self, region_indices: list) -> bool:
    """Save positions for multiple regions using batch processing.
    
    Args:
        region_indices: List of region indices to save
        
    Returns:
        True if batch queuing succeeded, False if fallback needed
    """
    if not region_indices:
        print(f"[PARALLEL] No region indices provided for batch save")
        return False
        
    try:
        # Initialize parallel save system if not already done
        if not hasattr(self, '_parallel_save_system') or not self._parallel_save_system:
            _init_parallel_save_system(self, )
        
        # Check if parallel system is available
        if not self._parallel_save_system:
            print(f"[PARALLEL] Parallel system unavailable, falling back to sequential saves")
            # Fall back to sequential single saves
            for region_index in region_indices:
                _fallback_single_save(self, region_index)
            return False
        
        # Queue batch save tasks
        queued_count = self._parallel_save_system.queue_batch_save_tasks(region_indices)
        
        if queued_count > 0:
            print(f"[PARALLEL] Successfully queued {queued_count} tasks for batch processing")
            return True
        else:
            print(f"[PARALLEL] No tasks queued, using fallback")
            # Fall back to sequential saves
            for region_index in region_indices:
                _fallback_single_save(self, region_index)
            return False
            
    except Exception as e:
        print(f"[PARALLEL] Batch save failed: {e}, using fallback")
        # Fall back to sequential saves
        for region_index in region_indices:
            _fallback_single_save(self, region_index)
        return False

def _fallback_single_save(self, region_index: int):
    """Fallback to single save processing when parallel system fails."""
    try:
        print(f"[PARALLEL] Using fallback single save for region {region_index}")
        
        # Mark auto-save as in progress
        self._auto_save_in_progress = True
        _update_save_overlay_button_state(self, )
        
        # Get translation text first
        trans_text = _get_translation_text_for_region(self, region_index)
        if not trans_text:
            self._auto_save_in_progress = False
            _update_save_overlay_button_state(self, )
            return
        
        # Use thread pool executor for background processing
        def render_task():
            try:
                # Update the single text overlay
                _update_single_text_overlay(self, region_index, trans_text)
                print(f"[FALLBACK] Region {region_index} saved successfully")
                # Signal-based refresh happens automatically after file save
                return True
            except Exception as e:
                print(f"[FALLBACK] Render task failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                self._auto_save_in_progress = False
                _update_save_overlay_button_state(self, )
                # Note: Pulse effect auto-removes after 0.1s via animation finished callback
        
        # Submit to executor if available
        if hasattr(self.main_gui, 'executor') and self.main_gui.executor:
            future = self.main_gui.executor.submit(render_task)
        else:
            render_task()
        
    except Exception as e:
        print(f"[PARALLEL] Fallback single save failed: {e}")
        self._auto_save_in_progress = False
        _update_save_overlay_button_state(self, )

def _update_save_overlay_button_state(self):
    """Update the save overlay button enabled/disabled state based on auto-save progress"""
    try:
        # Check if we have access to the button
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'save_overlay_btn'):
            in_progress = getattr(self, '_auto_save_in_progress', False)
            button = self.image_preview_widget.save_overlay_btn
            button.setEnabled(not in_progress)
            
            if in_progress:
                # Store original text if not already stored
                if not hasattr(button, '_original_text'):
                    button._original_text = button.text()
                button.setText("⏳")
            else:
                # Restore original text
                if hasattr(button, '_original_text'):
                    button.setText(button._original_text)
                else:
                    button.setText("💾")  # Fallback text
    except Exception:
        pass

def _update_parallel_save_button_state(self, active_count: int):
    """Update the save overlay button state for parallel processing.
    Shows different ⏳ emojis based on the number of parallel tasks.
    """
    try:
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'save_overlay_btn'):
            button = self.image_preview_widget.save_overlay_btn
            
            if active_count > 0:
                # Store original text if not already stored
                if not hasattr(button, '_original_text'):
                    button._original_text = button.text()
                
                # Show different indicators based on parallel count
                if active_count == 1:
                    button.setText("⏳1")
                    button.setToolTip("Auto-saving 1 rectangle position...")
                else:
                    button.setText(f"⏳{active_count}")
                    button.setToolTip(f"Auto-saving {active_count} rectangle positions in parallel...")
                
                button.setEnabled(False)
            else:
                # Restore original text and state
                if hasattr(button, '_original_text'):
                    button.setText(button._original_text)
                else:
                    button.setText("💾")
                
                button.setToolTip("Save & Update Overlay")
                button.setEnabled(True)
    except Exception as e:
        print(f"[PARALLEL] Error updating button state: {e}")


def _update_single_text_overlay_parallel(self, region_index: int, trans_text: str) -> bool:
    """Thread-safe version of overlay update for parallel processing.
    
    This method can be called from worker threads and handles thread safety.
    """
    try:
        # Import thread-safe utilities
        import time
        from PySide6.QtCore import QMetaObject, Qt
        
        # For thread safety, we'll queue the GUI update to the main thread
        def gui_update():
            try:
                # Persist rectangles state on main thread
                if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                    self.image_preview_widget._persist_rectangles_state()
                
                # Update the overlay using the existing method (main thread only)
                return _update_single_text_overlay(self, region_index, trans_text)
            except Exception as e:
                print(f"[PARALLEL] GUI update error for region {region_index}: {e}")
                return False
        
        # Use QMetaObject to invoke on main thread
        result = [False]  # Use list to allow modification in nested function
        
        def set_result(success):
            result[0] = success
        
        # Execute on main thread and wait for completion
        try:
        # Queue the GUI update to main thread
            if hasattr(self.main_gui, '_execute_parallel_gui_update'):
                success = self.main_gui._execute_parallel_gui_update(region_index, trans_text)
            else:
                # Fallback - call directly on main thread
                success = gui_update()
            
            if success is None:
                # Fallback: use the direct method (may cause thread issues but better than failing)
                return gui_update()
            
            return bool(success)
            
        except Exception:
            # Final fallback: direct call (not thread-safe but functional)
            print(f"[PARALLEL] Using direct GUI update fallback for region {region_index}")
            return gui_update()
            
    except Exception as e:
        print(f"[PARALLEL] Error in parallel overlay update for region {region_index}: {e}")
        return False

# NOTE: Auto-save position is now simplified to only update overlays without attempting
# to update the translated output preview. Users can manually click "Save & Update Overlay"
# if they want to see the updated preview in the output tab.
# The Save & Update Overlay button is disabled during auto-save to prevent conflicts.

def _get_translation_text_for_region(self, region_index: int) -> str:
    """Get translation text for a region (main thread safe)"""
    try:
        td = getattr(self, '_translation_data', {}) or {}
        if region_index in td:
            return td[region_index].get('translation', '')
        elif hasattr(self, '_translated_texts') and self._translated_texts:
            for t in self._translated_texts:
                if t.get('original', {}).get('region_index') == region_index:
                    return t.get('translation', '')
    except Exception as e:
        print(f"[DEBUG] Failed to get translation text: {e}")
    return ""

def _resolve_cleaned_image_for_render(self, current_image: str):
    """Resolve the correct cleaned image path for rendering.
    
    Checks state, in-memory cache, and falls back to filesystem discovery.
    Also repairs corrupted state if a correct cleaned image is found by discovery.
    Returns the cleaned image path or None.
    """
    try:
        current_base = os.path.splitext(os.path.basename(current_image))[0].replace('_cleaned', '')
        
        # 1. Check state manager
        try:
            if hasattr(self, 'image_state_manager') and self.image_state_manager:
                st = self.image_state_manager.get_state(current_image) or {}
                cand = st.get('cleaned_image_path')
                if cand and os.path.exists(cand):
                    cand_base = os.path.splitext(os.path.basename(cand))[0].replace('_cleaned', '')
                    if cand_base == current_base:
                        return cand
                    else:
                        print(f"[CLEAN_RESOLVE] Rejecting corrupted cleaned_image_path from state: {os.path.basename(cand)} (expected base: {current_base})")
        except Exception:
            pass
        
        # 2. Check in-memory _cleaned_image_path
        try:
            cand = getattr(self, '_cleaned_image_path', None)
            if cand and os.path.exists(cand):
                cand_base = os.path.splitext(os.path.basename(cand))[0].replace('_cleaned', '')
                if cand_base == current_base:
                    return cand
                else:
                    print(f"[CLEAN_RESOLVE] Rejecting stale _cleaned_image_path: {os.path.basename(cand)} (expected base: {current_base})")
        except Exception:
            pass
        
        # 3. Filesystem discovery fallback: look in expected folder
        try:
            ext = os.path.splitext(current_image)[1]
            parent_dir = os.path.dirname(current_image)
            
            # Check for OUTPUT_DIRECTORY override
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            search_dirs = []
            if override_dir:
                search_dirs.append(os.path.join(override_dir, f"{current_base}_translated"))
            search_dirs.append(os.path.join(parent_dir, f"{current_base}_translated"))
            
            for search_dir in search_dirs:
                expected_cleaned = os.path.join(search_dir, f"{current_base}_cleaned{ext}")
                if os.path.exists(expected_cleaned):
                    print(f"[CLEAN_RESOLVE] Discovered cleaned image via filesystem: {os.path.basename(expected_cleaned)}")
                    # Repair corrupted state and cache
                    self._cleaned_image_path = expected_cleaned
                    try:
                        if hasattr(self, 'image_state_manager') and self.image_state_manager:
                            self.image_state_manager.update_state(current_image, {'cleaned_image_path': expected_cleaned})
                            print(f"[CLEAN_RESOLVE] Repaired state with correct cleaned_image_path")
                    except Exception:
                        pass
                    return expected_cleaned
        except Exception:
            pass
        
        return None
    except Exception as e:
        print(f"[CLEAN_RESOLVE] Error: {e}")
        return None

def _extract_render_data_for_region(self, region_index: int) -> dict:
    """Extract all GUI data needed for rendering (main thread only)"""
    try:
        current_image = self.image_preview_widget.current_image_path
        if not current_image:
            print(f"[DEBUG] No current image path")
            return None
        
        # Extract the same data that _update_single_text_overlay uses
        if not (hasattr(self, '_translation_data') and self._translation_data):
            print(f"[DEBUG] No translation data available")
            return None
        
        from manga_translator import TextRegion
        rectangles = self.image_preview_widget.viewer.rectangles
        print(f"[DEBUG] Found {len(rectangles)} rectangles and {len(self._translation_data)} translations")
        
        # Prepare dimensions and positions (same as _update_single_text_overlay)
        from PIL import Image as _PILImage
        try:
            src_w, src_h = _PILImage.open(current_image).size
        except Exception:
            src_w, src_h = (1, 1)
        
        saved_offsets = {}
        last_pos = {}
        try:
            current_state = self.image_state_manager.get_state(current_image) if hasattr(self, 'image_state_manager') else None
            if current_state:
                saved_offsets = current_state.get('overlay_offsets') or {}
                last_pos = current_state.get('last_render_positions') or {}
        except Exception:
            saved_offsets, last_pos = {}, {}
        
        # Build TextRegion objects for ALL regions (same logic as _update_single_text_overlay)
        regions = []
        for idx_key in sorted(self._translation_data.keys()):
            trans_data = self._translation_data[idx_key]
            if region_index is not None and int(idx_key) == int(region_index):
                # Edited region — use current rectangle
                if int(idx_key) < len(rectangles):
                    rect = rectangles[int(idx_key)].sceneBoundingRect()
                    sx, sy, sw, sh = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                else:
                    lp = last_pos.get(str(int(idx_key)))
                    if not lp:
                        continue
                    sx, sy, sw, sh = map(int, lp)
            else:
                # Unedited region — lock to last render position if available
                lp = last_pos.get(str(int(idx_key)))
                if not lp:
                    if int(idx_key) < len(rectangles):
                        rect = rectangles[int(idx_key)].sceneBoundingRect()
                        lp = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
                    else:
                        continue
                sx, sy, sw, sh = map(int, lp)
            
            region = TextRegion(
                text=trans_data['original'],
                vertices=[(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh)],
                bounding_box=(sx, sy, sw, sh),
                confidence=1.0,
                region_type='text_block'
            )
            region.translated_text = trans_data['translation']
            regions.append(region)
        
        if not regions:
            print(f"[DEBUG] No regions to render")
            return None
        
        # Choose base image (state -> memory -> filesystem discovery)
        base_image = _resolve_cleaned_image_for_render(self, current_image)
        if base_image is None:
            base_image = current_image
            print(f"[DEBUG] Using original image as base (no cleaned image available)")
        
        # Scale regions if needed (same logic as _update_single_text_overlay)
        try:
            base_w, base_h = _PILImage.open(base_image).size
            if (src_w, src_h) != (base_w, base_h):
                sx = base_w / max(1, float(src_w))
                sy = base_h / max(1, float(src_h))
                print(f"[DEBUG] Scaling regions from src ({src_w}x{src_h}) -> base ({base_w}x{base_h}) with factors (sx={sx:.4f}, sy={sy:.4f})")
                from manga_translator import TextRegion as _TR
                scaled = []
                for r in regions:
                    x, y, w, h = r.bounding_box
                    nx = int(round(x * sx)); ny = int(round(y * sy)); nw = int(round(w * sx)); nh = int(round(h * sy))
                    v = [(nx, ny), (nx + nw, ny), (nx + nw, ny + nh), (nx, ny + nh)]
                    nr = _TR(text=r.text, vertices=v, bounding_box=(nx, ny, nw, nh), confidence=r.confidence, region_type=r.region_type)
                    nr.translated_text = r.translated_text
                    scaled.append(nr)
                regions = scaled
        except Exception as scale_err:
            print(f"[DEBUG] Region scaling skipped due to error: {scale_err}")
        
        # Determine output path (same logic as _update_single_text_overlay)
        try:
            rendered_path = getattr(self.image_preview_widget, 'current_translated_path', None)
            if not rendered_path and hasattr(self, 'image_state_manager') and self.image_state_manager:
                st = self.image_state_manager.get_state(current_image) or {}
                rendered_path = st.get('rendered_image_path')
            if not rendered_path and hasattr(self, '_rendered_images_map'):
                rendered_path = self._rendered_images_map.get(current_image)
        except Exception:
            rendered_path = None
        output_path = rendered_path if (rendered_path and os.path.exists(os.path.dirname(rendered_path))) else None
        
        return {
            'current_image': current_image,
            'regions': regions,
            'base_image': base_image,
            'output_path': output_path
        }
        
    except Exception as e:
        print(f"[DEBUG] Failed to extract render data: {e}")
        import traceback
        traceback.print_exc()
        return None

@Slot(str)
def _on_save_progress(self, message: str):
    """Handle progress updates from worker"""
    print(f"[PROGRESS] {message}")

@Slot(bool, int, str)
def _on_save_position_finished(self, success: bool, region_index: int, rendered_path: str):
    """Handle completion from background worker (main thread)"""
    try:
        print(f"[DEBUG] Save position finished - success: {success}, region: {region_index}")
        
        if success and rendered_path:
            # Store the rendered image path for reference and refresh the source preview
            try:
                print(f"[DEBUG] Rendered image saved: {os.path.basename(rendered_path)}")
                if hasattr(self.image_preview_widget, 'current_translated_path'):
                    self.image_preview_widget.current_translated_path = rendered_path
                # Refresh source preview to show the newly rendered image
                if hasattr(self.image_preview_widget, 'current_image_path'):
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(500, lambda: self.image_preview_widget.load_image(
                        self.image_preview_widget.current_image_path, 
                        preserve_rectangles=True, 
                        preserve_text_overlays=True
                    ))
                print(f"[DEBUG] Source preview refresh scheduled")
            except Exception as e:
                print(f"[DEBUG] Error handling rendered output: {e}")
        
        print(f"[DEBUG] Save Position completed for region {region_index}")
        
    except Exception as err:
        print(f"[DEBUG] Save Position completion failed: {err}")
        import traceback
        traceback.print_exc()
    finally:
        # Always remove processing overlay
        _remove_processing_overlay(self, )

def _save_overlay_async(self, region_index: int = 0, new_translation: str = "", update_all_regions: bool = False):
    """Save & Update Overlay functionality using ThreadPoolExecutor on main thread.
    
    Args:
        region_index: Region index to update (default 0 for full re-render)
        new_translation: Specific translation text to use (empty string for original behavior)
        update_all_regions: If True, update all regions with current settings (not just one)
    """
    print(f"\n{'='*80}")
    print(f"[DEBUG] _save_overlay_async: METHOD ENTRY")
    print(f"[DEBUG] Args: region_index={region_index}, new_translation='{new_translation}', update_all_regions={update_all_regions}")
    print(f"{'='*80}\n")
    
    try:
        print(f"[DEBUG] Save & Update Overlay triggered for region {region_index}, translation='{new_translation}', update_all={update_all_regions}")
        
        # Show processing overlay immediately on main thread
        _add_processing_overlay(self, )
        
        def _save_overlay_task():
            """The actual save overlay task - runs via executor but stays on the main thread"""
            print(f"[DEBUG] _save_overlay_task: TASK FUNCTION ENTRY")
            try:
                print(f"[DEBUG] Save & Update Overlay task executing for region {region_index}")
                
                # Persist rectangles state
                try:
                    if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                        print(f"[DEBUG] Persisting rectangles state...")
                        self.image_preview_widget._persist_rectangles_state()
                        print(f"[DEBUG] Rectangles state persisted successfully")
                    else:
                        print(f"[DEBUG] No _persist_rectangles_state method available")
                except Exception as e:
                    print(f"[DEBUG] Failed to persist rectangles state: {e}")
                
                # Call _update_single_text_overlay directly with the provided parameters
                # This matches the original behavior exactly
                print(f"[DEBUG] Calling _update_single_text_overlay({region_index}, '{new_translation}', update_all={update_all_regions})")
                _update_single_text_overlay(self, region_index, new_translation, update_all_regions=update_all_regions)
                print(f"[DEBUG] _update_single_text_overlay call completed successfully")
                print(f"[DEBUG] Save & Update Overlay task completed for region {region_index}")
                return True
                
            except Exception as e:
                print(f"[DEBUG] Save & Update Overlay task failed for region {region_index}: {e}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                return False
        
        # Check executor availability with detailed logging
        has_main_gui = hasattr(self, 'main_gui')
        has_executor_attr = has_main_gui and hasattr(self.main_gui, 'executor')
        executor_exists = has_executor_attr and self.main_gui.executor is not None
        
        print(f"[DEBUG] Executor check: has_main_gui={has_main_gui}, has_executor_attr={has_executor_attr}, executor_exists={executor_exists}")
        
        if executor_exists:
            print(f"[DEBUG] Using ThreadPoolExecutor for overlay task")
            future = self.main_gui.executor.submit(_save_overlay_task)
            print(f"[DEBUG] Task submitted to executor (fire-and-forget for responsiveness)")
            # Don't wait for completion - fire and forget for responsiveness
            from PySide6.QtCore import QTimer
            
            # Refresh source preview to show updated translated/cleaned image
            def _refresh_source_preview():
                try:
                    ipw = getattr(self, 'image_preview_widget', None)
                    if ipw and getattr(ipw, 'current_image_path', None):
                        ipw.load_image(ipw.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                except Exception as _e:
                    print(f"[DEBUG] Source preview refresh failed: {_e}")
            QTimer.singleShot(1200, _refresh_source_preview)
            QTimer.singleShot(3000, _refresh_source_preview)
        else:
            print(f"[DEBUG] No executor available, running save overlay synchronously")
            _save_overlay_task()
            # Immediately refresh view since it was synchronous
            from PySide6.QtCore import QTimer
            def _refresh_source_preview_sync():
                try:
                    ipw = getattr(self, 'image_preview_widget', None)
                    if ipw and getattr(ipw, 'current_image_path', None):
                        ipw.load_image(ipw.current_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                except Exception as _e:
                    print(f"[DEBUG] Source preview refresh (sync) failed: {_e}")
            QTimer.singleShot(600, _refresh_source_preview_sync)
        
        print(f"[DEBUG] _save_overlay_async: METHOD COMPLETION")
        
    except Exception as err:
        print(f"[DEBUG] Save & Update Overlay failed to start for region {region_index}: {err}")
        import traceback
        print(f"[DEBUG] Method error traceback: {traceback.format_exc()}")
    finally:
        # Always remove the processing overlay
        try:
            print(f"[DEBUG] Removing processing overlay...")
            _remove_processing_overlay(self, )
            print(f"[DEBUG] Processing overlay removed successfully")
        except Exception as e:
            print(f"[DEBUG] Failed to remove processing overlay: {e}")

def _start_output_refresh_check(self):
    """Start periodic checking for output image updates"""
    try:
        from PySide6.QtCore import QTimer
        # Check every 1 second for up to 10 seconds
        if not hasattr(self, '_output_refresh_timer'):
            self._output_refresh_timer = QTimer()
            self._output_refresh_timer.setSingleShot(False)
            self._output_refresh_timer.timeout.connect(self._check_and_refresh_output)
        
        # Store the check start time and count
        import time
        self._output_refresh_start_time = time.time()
        self._output_refresh_count = 0
        
        self._output_refresh_timer.start(2000)  # Check every 2 seconds (reduced frequency)
        print(f"[DEBUG] Started output refresh checking timer")
    except Exception as e:
        print(f"[DEBUG] Error starting output refresh check: {e}")

def _check_and_refresh_output(self):
    """Periodic check for output image updates"""
    try:
        import time
        current_time = time.time()
        elapsed = current_time - getattr(self, '_output_refresh_start_time', current_time)
        self._output_refresh_count = getattr(self, '_output_refresh_count', 0) + 1
        
        # Stop checking after 10 seconds or 10 attempts
        if elapsed > 10.0 or self._output_refresh_count > 10:
            if hasattr(self, '_output_refresh_timer'):
                self._output_refresh_timer.stop()
            print(f"[DEBUG] Stopped output refresh checking after {elapsed:.1f}s and {self._output_refresh_count} attempts")
            return
        
        # Try to refresh the output
        if _refresh_output_tab(self, ):
            # Success - stop checking
            if hasattr(self, '_output_refresh_timer'):
                self._output_refresh_timer.stop()
            print(f"[DEBUG] Output refreshed successfully, stopped checking")
            
    except Exception as e:
        print(f"[DEBUG] Error in output refresh check: {e}")

def _refresh_output_tab(self) -> bool:
    """Refresh the output tab with the latest rendered image"""
    try:
        current_image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image_path:
            return False
        
        # Look for rendered image in the expected location
        source_dir = os.path.dirname(current_image_path)
        source_filename = os.path.basename(current_image_path)
        
        # Check various possible locations for translated images
        possible_paths = [
            # 3_translated folder
            os.path.join(source_dir, "3_translated", source_filename),
            # isolated folder
            os.path.join(source_dir, f"{os.path.splitext(source_filename)[0]}_translated", source_filename),
            # same directory with _translated suffix
            os.path.join(source_dir, f"{os.path.splitext(source_filename)[0]}_translated{os.path.splitext(source_filename)[1]}")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if this file is newer than what we currently have loaded
                current_translated = getattr(self.image_preview_widget, 'current_translated_path', None)
                if current_translated != path or _is_file_newer(self, path, current_translated):
                    print(f"[DEBUG] Refreshing output tab with: {os.path.basename(path)}")
                    self.image_preview_widget.output_viewer.load_image(path)
                    self.image_preview_widget.current_translated_path = path
                    return True
        
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error refreshing output tab: {e}")
        return False

def _is_file_newer(self, file_path: str, reference_path: str) -> bool:
    """Check if file_path is newer than reference_path"""
    try:
        if not reference_path or not os.path.exists(reference_path):
            return True  # New file is always "newer" than non-existent reference
        
        import os
        file_mtime = os.path.getmtime(file_path)
        ref_mtime = os.path.getmtime(reference_path)
        return file_mtime > ref_mtime
    except Exception:
        return True  # Assume newer on error

def _update_single_text_overlay(self, region_index: int, new_translation: str, update_all_regions: bool = False):
    """Update overlay after editing by rendering with MangaTranslator (same as regular pipeline)
    
    Args:
        region_index: Region index to update (ignored if update_all_regions=True)
        new_translation: Specific translation text to use (empty string for original behavior)
        update_all_regions: If True, update all regions with current rectangle positions and rendering settings
    """
    print(f"\n{'='*60}")
    print(f"[DEBUG] _update_single_text_overlay called for region {region_index}, update_all={update_all_regions}")
    print(f"[DEBUG] new_translation: '{new_translation[:50] if new_translation else ''}...'")
    print(f"{'='*60}\n")
    
    # ⚡ FAST PATH DISABLED - needs more work to cache all regions first
    # The current implementation only renders the moved region, causing others to disappear
    # TODO: Pre-cache all regions on initial render, then update incrementally
    
    try:
        current_image = self.image_preview_widget.current_image_path
        
        if not current_image:
            print(f"[DEBUG] ERROR: No current image path, cannot update overlay")
            self._log("❌ No image loaded", "error")
            return False
        
        # CRITICAL: Validate that translation data belongs to the current image
        translation_image_path = None
        if hasattr(self, '_translating_image_path'):
            translation_image_path = self._translating_image_path
        elif hasattr(self, '_translation_data_image_path'):
            translation_image_path = self._translation_data_image_path
        
        # If translation belongs to a different image, abort with clear error
        if translation_image_path and os.path.abspath(translation_image_path) != os.path.abspath(current_image):
            error_msg = f"Cannot render: Translation data is for {os.path.basename(translation_image_path)} but you're viewing {os.path.basename(current_image)}"
            print(f"[CRITICAL] {error_msg}")
            self._log(f"❌ {error_msg}", "error")
            return False
        
        print(f"[DEBUG] Current image: {current_image}")
        print(f"[DEBUG] Manual edit complete for region {region_index}. Rendering with MangaTranslator...")
        self._log(f"🔄 Rendering edited translation...", "info")
        
        # Prepare data for rendering
        if hasattr(self, '_translation_data') and self._translation_data:
            from manga_translator import TextRegion
            
            rectangles = self.image_preview_widget.viewer.rectangles
            print(f"[DEBUG] Found {len(rectangles)} rectangles and {len(self._translation_data)} translations")
            
            regions = []
            
            # Prepare dimensions and last positions
            from PIL import Image as _PILImage
            try:
                src_w, src_h = _PILImage.open(current_image).size
            except Exception:
                src_w, src_h = (1, 1)
            saved_offsets = {}
            last_pos = {}
            try:
                current_state = self.image_state_manager.get_state(current_image) if hasattr(self, 'image_state_manager') else None
                if current_state:
                    saved_offsets = current_state.get('overlay_offsets') or {}
                    last_pos = current_state.get('last_render_positions') or {}
            except Exception:
                saved_offsets, last_pos = {}, {}
            
            # Build TextRegion objects for ALL regions
            for idx_key in sorted(self._translation_data.keys()):
                trans_data = self._translation_data[idx_key]
                
                # Determine which position to use based on update_all_regions flag
                if update_all_regions:
                    # Update all regions - use current rectangle positions for ALL
                    if int(idx_key) < len(rectangles):
                        rect = rectangles[int(idx_key)].sceneBoundingRect()
                        sx, sy, sw, sh = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                    else:
                        # Fall back to last position if rectangle doesn't exist
                        lp = last_pos.get(str(int(idx_key)))
                        if not lp:
                            continue
                        sx, sy, sw, sh = map(int, lp)
                elif region_index is not None and int(idx_key) == int(region_index):
                    # Edited region — use current rectangle
                    if int(idx_key) < len(rectangles):
                        rect = rectangles[int(idx_key)].sceneBoundingRect()
                        sx, sy, sw, sh = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                    else:
                        lp = last_pos.get(str(int(idx_key)))
                        if not lp:
                            continue
                        sx, sy, sw, sh = map(int, lp)
                else:
                    # Unedited region — lock to last render position if available
                    lp = last_pos.get(str(int(idx_key)))
                    if not lp:
                        if int(idx_key) < len(rectangles):
                            rect = rectangles[int(idx_key)].sceneBoundingRect()
                            lp = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
                        else:
                            continue
                    sx, sy, sw, sh = map(int, lp)
                
                region = TextRegion(
                    text=trans_data['original'],
                    vertices=[(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh)],
                    bounding_box=(sx, sy, sw, sh),
                    confidence=1.0,
                    region_type='text_block'
                )
                region.translated_text = trans_data['translation']
                regions.append(region)
            
            print(f"[DEBUG] Prepared {len(regions)} regions (edited idx={region_index}) using last_render_positions for stability")
            
            if regions:
                print(f"[DEBUG] ✅ Built {len(regions)} regions, selecting base image for renderer...")
                # Choose base image (state -> memory -> filesystem discovery)
                base_image = _resolve_cleaned_image_for_render(self, current_image)
                if base_image is None:
                    base_image = current_image
                    print(f"[DEBUG] Using original image as base (no cleaned image available)")
                else:
                    print(f"[DEBUG] Using cleaned image as base for incremental preview")
                
                # Scale regions (from source coords) to base image dimensions if needed
                # CRITICAL FIX: Only scale regions that are in source coordinates.
                # Regions from last_render_positions are already in base image coordinates.
                try:
                    from PIL import Image as _PILImage
                    base_w, base_h = _PILImage.open(base_image).size
                    if (src_w, src_h) != (base_w, base_h):
                        sx = base_w / max(1, float(src_w))
                        sy = base_h / max(1, float(src_h))
                        print(f"[DEBUG] Scaling regions from src ({src_w}x{src_h}) -> base ({base_w}x{base_h}) with factors (sx={sx:.4f}, sy={sy:.4f})")
                        from manga_translator import TextRegion as _TR
                        scaled = []
                        for idx, r in enumerate(regions):
                            # Determine if this region came from last_pos (already in base coords)
                            # or from current rectangle position (in source coords, needs scaling)
                            idx_key = sorted(self._translation_data.keys())[idx]
                            is_moved_region = (region_index is not None and int(idx_key) == int(region_index))
                            has_last_pos = str(int(idx_key)) in last_pos
                            
                            # Only scale if: it's the moved region OR there's no last_pos (first render)
                            # Don't scale regions using locked last_pos - they're already in base coords!
                            if is_moved_region or not has_last_pos:
                                x, y, w, h = r.bounding_box
                                nx = int(round(x * sx)); ny = int(round(y * sy)); nw = int(round(w * sx)); nh = int(round(h * sy))
                            else:
                                # Already in base coords from last_pos - don't scale again!
                                x, y, w, h = r.bounding_box
                                nx, ny, nw, nh = x, y, w, h
                            
                            v = [(nx, ny), (nx + nw, ny), (nx + nw, ny + nh), (nx, ny + nh)]
                            nr = _TR(text=r.text, vertices=v, bounding_box=(nx, ny, nw, nh), confidence=r.confidence, region_type=r.region_type)
                            nr.translated_text = r.translated_text
                            scaled.append(nr)
                        regions = scaled
                except Exception as scale_err:
                    print(f"[DEBUG] Region scaling skipped due to error: {scale_err}")
                
                print(f"[DEBUG] Rendering base image: {os.path.basename(base_image)} (original: {os.path.basename(current_image)})")
                # Generate proper isolated output path for this specific image
                output_path = None
                try:
                    # Create isolated folder path based on current image (not reuse from other images)
                    filename = os.path.basename(current_image)
                    base_name = os.path.splitext(filename)[0]
                    parent_dir = os.path.dirname(current_image)
                    
                    # Check for OUTPUT_DIRECTORY override (prefer config over env var)
                    override_dir = None
                    if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                        override_dir = self.main_gui.config.get('output_directory', '')
                    if not override_dir:
                        override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
                    
                    if override_dir:
                        output_dir = os.path.join(override_dir, f"{base_name}_translated")
                    else:
                        output_dir = os.path.join(parent_dir, f"{base_name}_translated")
                    
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, filename)
                    print(f"[DEBUG] Generated isolated output path: {output_path}")
                except Exception as e:
                    print(f"[DEBUG] Failed to generate output path: {e}")
                    output_path = None
                _render_with_manga_translator(self, base_image, regions, output_path=output_path, original_image_path=current_image, switch_tab=False)
                return True  # Success!
            else:
                print(f"[DEBUG] ❌ No regions to render")
                self._log("⚠️ No regions to render", "warning")
                return False
        else:
            print(f"[DEBUG] ❌ No translation data available")
            self._log("⚠️ No translation data", "warning")
            return False
        
    except Exception as e:
        print(f"[DEBUG] ❌ ERROR in _update_single_text_overlay: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[DEBUG] Traceback:\n{traceback_str}")
        self._log(f"❌ Rendering failed: {str(e)}", "error")
        return False

def save_positions_and_rerender(self):
    """Persist current positions and re-render entire output using locked positions for stability.
    - Uses translated_texts from state if available; falls back to in-memory _translated_texts/_translation_data
    - Prefers cleaned base for quality; overwrites existing translated image if present
    """
    try:
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            print("[SAVE_POS] No current image path")
            return
        # Build text regions list
        # Load translated_texts
        translated_texts = []
        try:
            if hasattr(self, 'image_state_manager') and self.image_state_manager:
                st = self.image_state_manager.get_state(current_image) or {}
                translated_texts = st.get('translated_texts') or []
        except Exception as e:
            print(f"[SAVE_POS] Error loading translated_texts from state: {e}")
            import traceback
            print(f"[SAVE_POS] Traceback:\n{traceback.format_exc()}")
            translated_texts = []
        if not translated_texts and hasattr(self, '_translated_texts'):
            translated_texts = self._translated_texts or []
        if not translated_texts and hasattr(self, '_translation_data') and isinstance(self._translation_data, dict):
            # Fallback: synthesize from rectangles and _translation_data
            for idx, td in self._translation_data.items():
                try:
                    rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                    if 0 <= int(idx) < len(rects):
                        br = rects[int(idx)].sceneBoundingRect()
                        bbox = [int(br.x()), int(br.y()), int(br.width()), int(br.height())]
                        translated_texts.append({'original': {'text': td.get('original',''), 'region_index': int(idx)}, 'translation': td.get('translation',''), 'bbox': bbox})
                except Exception as e:
                    print(f"[SAVE_POS] Error synthesizing text for region {idx}: {e}")
                    import traceback
                    print(f"[SAVE_POS] Traceback:\n{traceback.format_exc()}")
                    continue
        if not translated_texts:
            print("[SAVE_POS] No translated_texts available to render; aborting")
            return
        
        # Load last positions
        last_pos = {}
        try:
            st = self.image_state_manager.get_state(current_image) if hasattr(self, 'image_state_manager') else {}
            last_pos = (st or {}).get('last_render_positions', {}) or {}
        except Exception as e:
            print(f"[SAVE_POS] Error loading last_render_positions: {e}")
            import traceback
            print(f"[SAVE_POS] Traceback:\n{traceback.format_exc()}")
            last_pos = {}
        
        # Build regions from last_pos or rectangles
        from manga_translator import TextRegion
        regions = []
        rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
        for i, result in enumerate(translated_texts):
            try:
                region_index = result.get('original', {}).get('region_index', i)
                lp = last_pos.get(str(int(region_index)))
                if lp and len(lp) >= 4:
                    x, y, w, h = map(int, lp)
                elif 0 <= int(region_index) < len(rects):
                    br = rects[int(region_index)].sceneBoundingRect()
                    x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
                else:
                    bbox = result.get('bbox') or []
                    if len(bbox) >= 4:
                        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    else:
                        continue
                vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                tr = TextRegion(text=result['original']['text'], vertices=vertices, bounding_box=(x, y, w, h), confidence=1.0, region_type='text_block')
                tr.translated_text = result['translation']
                regions.append(tr)
            except Exception as e:
                print(f"[SAVE_POS] Error building region {i}: {e}")
                import traceback
                print(f"[SAVE_POS] Traceback:\n{traceback.format_exc()}")
                continue
        if not regions:
            print("[SAVE_POS] No regions built; aborting")
            return
        
        # Choose base image (state -> memory -> filesystem discovery)
        base_image = _resolve_cleaned_image_for_render(self, current_image)
        if base_image is None:
            base_image = current_image
            print(f"[SAVE_POS] Using original image as base (no cleaned image available)")
        
        # Scale regions if base dims differ
        try:
            from PIL import Image as _PIL
            src_w, src_h = _PIL.open(current_image).size
            base_w, base_h = _PIL.open(base_image).size
            if (src_w, src_h) != (base_w, base_h):
                sx = base_w / max(1, float(src_w)); sy = base_h / max(1, float(src_h))
                from manga_translator import TextRegion as _TR
                scaled = []
                for r in regions:
                    x, y, w, h = r.bounding_box
                    nx, ny, nw, nh = int(round(x*sx)), int(round(y*sy)), int(round(w*sx)), int(round(h*sy))
                    v = [(nx, ny), (nx+nw, ny), (nx+nw, ny+nh), (nx, ny+nh)]
                    nr = _TR(text=r.text, vertices=v, bounding_box=(nx, ny, nw, nh), confidence=r.confidence, region_type=r.region_type)
                    nr.translated_text = r.translated_text
                    scaled.append(nr)
                regions = scaled
        except Exception as e:
            print(f"[SAVE_POS] Error scaling regions: {e}")
            import traceback
            print(f"[SAVE_POS] Traceback:\n{traceback.format_exc()}")
        
        # Generate proper isolated output path for this specific image
        output_path = None
        try:
            # Create isolated folder path based on current image (not reuse from other images)
            filename = os.path.basename(current_image)
            base_name = os.path.splitext(filename)[0]
            parent_dir = os.path.dirname(current_image)
            
            # Check for OUTPUT_DIRECTORY override (prefer config over env var)
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            if override_dir:
                output_dir = os.path.join(override_dir, f"{base_name}_translated")
            else:
                output_dir = os.path.join(parent_dir, f"{base_name}_translated")
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            print(f"[SAVE_POS] Generated isolated output path: {output_path}")
        except Exception as e:
            print(f"[SAVE_POS] Failed to generate output path: {e}")
            output_path = None
        
        # Render
        _render_with_manga_translator(self, base_image, regions, output_path=output_path, original_image_path=current_image, switch_tab=False)
    except Exception as e:
        print(f"[SAVE_POS] Error: {e}")
        import traceback
        print(f"[SAVE_POS] Traceback:\n{traceback.format_exc()}")

def _render_with_manga_translator(self, image_path: str, regions, output_path: str = None, image_bgr=None, original_image_path: str = None, switch_tab: bool = True):
    """Render translated text using MangaTranslator's PIL pipeline.
    - image_bgr: optional OpenCV BGR image to render on (in-memory, preferred if provided)
    - output_path: where to save the rendered image (isolated per-image folder)
    - original_image_path: the original source image path for mapping/state
    """
    print(f"{'='*80}\n")
    
    try:
        from manga_translator import MangaTranslator
        from unified_api_client import UnifiedClient
        import tempfile
        import shutil
        from PIL import Image
        import cv2
        import numpy as np
        
        print(f"[RENDER] Imports successful")
        self._log(f"🎨 Rendering with PIL pipeline...", "info")
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"[RENDER] Image exists: {os.path.exists(image_path)}")
        
        # Decide which translator to use: prefer existing main translator for consistent settings
        translator_inst = None
        try:
            if hasattr(self, 'translator') and self.translator:
                translator_inst = self.translator
                # Ensure latest GUI settings are applied to the main translator
                try:
                    if hasattr(self, '_apply_rendering_settings'):
                        self._apply_rendering_settings()
                except Exception:
                    pass
        except Exception:
            translator_inst = None
        
        if translator_inst is None:
            # Fallback: create or reuse a lightweight translator dedicated to rendering
            if not hasattr(self, '_manga_translator') or self._manga_translator is None:
                print(f"[RENDER] Creating new MangaTranslator instance (render-only)...")
                ocr_config = _get_ocr_config(self, )
                api_key = self.main_gui.config.get('api_key', '') if hasattr(self, 'main_gui') else ''
                model = self.main_gui.config.get('model', 'gpt-4o-mini') if hasattr(self, 'main_gui') else 'gpt-4o-mini'
                if not api_key:
                    print(f"[RENDER] ERROR: No API key found!")
                    raise ValueError("No API key found")
                unified_client = UnifiedClient(model=model, api_key=api_key)
                self._manga_translator = MangaTranslator(
                    ocr_config=ocr_config,
                    unified_client=unified_client,
                    main_gui=self.main_gui,
                    log_callback=self._log,
                    skip_inpainter_init=True
                )
                print(f"[RENDER] MangaTranslator instance created")
            else:
                print(f"[RENDER] Using existing MangaTranslator instance (render-only)")
            translator_inst = self._manga_translator
            
            # Apply current GUI rendering settings to the render-only translator
            try:
                # Safe area controls
                try:
                    translator_inst.safe_area_enabled = bool(getattr(self, 'safe_area_enabled_value', True))
                    translator_inst.safe_area_scale = float(getattr(self, 'safe_area_scale_value', 1.0))
                except Exception:
                    pass
                # Text color & shadow
                text_color = (
                    getattr(self, 'text_color_r_value', 102),
                    getattr(self, 'text_color_g_value', 0),
                    getattr(self, 'text_color_b_value', 0),
                )
                shadow_color = (
                    getattr(self, 'shadow_color_r_value', 255),
                    getattr(self, 'shadow_color_g_value', 255),
                    getattr(self, 'shadow_color_b_value', 255),
                )
                translator_inst.update_text_rendering_settings(
                    bg_opacity=getattr(self, 'bg_opacity_value', 0),
                    bg_style=getattr(self, 'bg_style_value', 'circle'),
                    bg_reduction=getattr(self, 'bg_reduction_value', 1.0),
                    font_style=getattr(self, 'selected_font_path', None),
                    font_size=(-getattr(self, 'font_size_multiplier_value', 1.0)) if getattr(self, 'font_size_mode_value', 'fixed') == 'multiplier' else getattr(self, 'font_size_value', 0),
                    text_color=text_color,
                    shadow_enabled=getattr(self, 'shadow_enabled_value', True),
                    shadow_color=shadow_color,
                    shadow_offset_x=getattr(self, 'shadow_offset_x_value', 2),
                    shadow_offset_y=getattr(self, 'shadow_offset_y_value', 2),
                    shadow_blur=getattr(self, 'shadow_blur_value', 0),
                    force_caps_lock=getattr(self, 'force_caps_lock_value', True)
                )
                # Mode and bounds
                translator_inst.font_size_mode = getattr(self, 'font_size_mode_value', 'fixed')
                translator_inst.font_size_multiplier = getattr(self, 'font_size_multiplier_value', 1.0)
                translator_inst.min_readable_size = int(getattr(self, 'auto_min_size_value', 10))
                translator_inst.max_font_size_limit = int(getattr(self, 'max_font_size_value', 48))
                translator_inst.strict_text_wrapping = getattr(self, 'strict_text_wrapping_value', True)
                translator_inst.force_caps_lock = getattr(self, 'force_caps_lock_value', True)
                translator_inst.constrain_to_bubble = getattr(self, 'constrain_to_bubble_value', True)
                # Free-text-only BG opacity toggle
                try:
                    translator_inst.free_text_only_bg_opacity = bool(getattr(self, 'free_text_only_bg_opacity_value', False))
                except Exception:
                    pass
            except Exception as _rs:
                print(f"[RENDER] Failed to apply rendering settings to render-only translator: {_rs}")
        
        
        # Prepare image as numpy BGR array
        if image_bgr is None:
            print(f"[RENDER] Loading image from path...")
            pil_image = Image.open(image_path)
            print(f"[RENDER] Image size: {pil_image.size}")
            image_rgb = np.array(pil_image.convert('RGB'))
            
            # Convert RGB to BGR
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        print(f"[RENDER] Using BGR image, shape: {image_bgr.shape}")
        
        # Pre-clear old region rectangles from translated output using cleaned image if available
        try:
            clear_rects = getattr(self, '_pending_clear_rects', []) if hasattr(self, '_pending_clear_rects') else []
            if clear_rects:
                # Find cleaned image for original_image_path
                cleaned_bgr = None
                try:
                    cleaned_path = None
                    if original_image_path and hasattr(self, 'image_state_manager') and self.image_state_manager:
                        st = self.image_state_manager.get_state(original_image_path) or {}
                        cleaned_path = st.get('cleaned_image_path')
                    if not cleaned_path:
                        cand = getattr(self, '_cleaned_image_path', None)
                        cleaned_path = cand if cand and os.path.exists(cand) else None
                    if cleaned_path and os.path.exists(cleaned_path):
                        pil_clean = Image.open(cleaned_path).convert('RGB')
                        clean_rgb = np.array(pil_clean)
                        # Convert RGB to BGR
                        cleaned_bgr_full = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
                        # Scale cleaned to match current base dims if needed
                        if (cleaned_bgr_full.shape[1], cleaned_bgr_full.shape[0]) != (image_bgr.shape[1], image_bgr.shape[0]):
                            cleaned_bgr = cv2.resize(
                                cleaned_bgr_full,
                                (image_bgr.shape[1], image_bgr.shape[0]),
                                interpolation=cv2.INTER_CUBIC
                            )
                        else:
                            cleaned_bgr = cleaned_bgr_full
                except Exception as _ce:
                    print(f"[RENDER] Cleaned preload failed: {_ce}")
                    cleaned_bgr = None
                
                for (cx, cy, cw, ch) in clear_rects:
                    x1 = max(0, int(cx)); y1 = max(0, int(cy))
                    x2 = min(image_bgr.shape[1], int(cx + cw)); y2 = min(image_bgr.shape[0], int(cy + ch))
                    if x2 > x1 and y2 > y1:
                        if cleaned_bgr is not None:
                            image_bgr[y1:y2, x1:x2] = cleaned_bgr[y1:y2, x1:x2]
                        else:
                            # Fallback: fill with background color (white)
                            image_bgr[y1:y2, x1:x2] = (255, 255, 255)
                print(f"[RENDER] Cleared {len(clear_rects)} old region(s) prior to re-render")
        except Exception as _clr:
            print(f"[RENDER] Pre-clear failed: {_clr}")
        
        # Filter out excluded regions before rendering (get from rectangle objects)
        excluded_regions = []
        try:
            if hasattr(self.image_preview_widget, 'viewer') and self.image_preview_widget.viewer.rectangles:
                rectangles = self.image_preview_widget.viewer.rectangles
                for i, rect_item in enumerate(rectangles):
                    if getattr(rect_item, 'exclude_from_clean', False):
                        excluded_regions.append(i)
                
                if excluded_regions:
                    print(f"[RENDER] Found {len(excluded_regions)} excluded regions: {excluded_regions}")
                else:
                    print(f"[RENDER] No regions excluded from rendering")
            else:
                print(f"[RENDER] No rectangles available to check exclusions")
        except Exception as e:
            print(f"[RENDER] Failed to get excluded regions from rectangles: {e}")
        
        # Filter regions based on exclusion status
        filtered_regions = []
        for i, region in enumerate(regions):
            if i in excluded_regions:
                print(f"[RENDER] EXCLUDING Region {i}: text='{region.text[:30] if region.text else 'None'}...', translated='{region.translated_text[:30] if region.translated_text else 'None'}...' (marked as excluded)")
            else:
                print(f"[RENDER] INCLUDING Region {i}: text='{region.text[:30] if region.text else 'None'}...', translated='{region.translated_text[:30] if region.translated_text else 'None'}...'")
                filtered_regions.append(region)
        
        print(f"[RENDER] Filtered regions: {len(regions)} -> {len(filtered_regions)} (excluded {len(regions) - len(filtered_regions)} regions)")
        
        # Call MangaTranslator's render_translated_text method with filtered regions
        print(f"[RENDER] Calling render_translated_text with {len(filtered_regions)} regions...")
        rendered_bgr = translator_inst.render_translated_text(image_bgr, filtered_regions)
        print(f"[RENDER] Rendering complete, output shape: {rendered_bgr.shape}")
        
        # Convert back to PIL and save
        rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        rendered_pil = Image.fromarray(rendered_rgb)
        
        # Determine output path
        if output_path is None:
            input_dir = os.path.dirname(image_path)
            output_dir = os.path.join(input_dir, "3_translated")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, output_filename)
        else:
            # Ensure directory exists for provided output path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_filename = os.path.basename(output_path)
        
        print(f"[RENDER] Saving to: {output_path}\n")
        rendered_pil.save(output_path)
        print(f"[RENDER] Saved successfully, file exists: {os.path.exists(output_path)}")
        
        # Trigger instant preview refresh now that file is saved
        try:
            if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'refresh_preview_signal'):
                self.main_gui.refresh_preview_signal.emit()
                print(f"[RENDER] ✓ Triggered preview refresh signal after file save")
        except Exception as e:
            print(f"[RENDER] Failed to emit refresh signal: {e}")
        
        # Store rendered image path mapped to ORIGINAL source image (not cleaned)
        # This allows navigation to work properly
        if not hasattr(self, '_rendered_images_map'):
            self._rendered_images_map = {}
        
        # Determine the original image path for mapping
        # If original_image_path was explicitly provided, use it directly (important for output directory override)
        # Only try to derive from output path when original_image_path was not provided
        if original_image_path:
            original_path = original_image_path
            print(f"[RENDER] Using explicitly provided original_image_path: {os.path.basename(original_path)}")
        else:
            original_path = image_path
            try:
                # Only derive from output path when no original was explicitly provided
                if output_path and os.path.basename(os.path.dirname(output_path)).endswith('_translated'):
                    original_path = os.path.join(os.path.dirname(os.path.dirname(output_path)), os.path.basename(output_path))
                    print(f"[RENDER] Mapped output back to original: {os.path.basename(original_path)} -> {os.path.basename(output_path)}")
            except Exception:
                pass
        
        # Store mapping
        self._rendered_images_map[original_path] = output_path
        
        # SAVE RENDERED IMAGE PATH TO STATE MANAGER for persistence
        if hasattr(self, 'image_state_manager'):
            self.image_state_manager.update_state(original_path, {
                'rendered_image_path': output_path
            }, save=True)
            print(f"[RENDER] Saved rendered image path to state for {os.path.basename(original_path)}")
        
        # Update last_render_positions for robust future single-region updates
        try:
            if original_image_path and hasattr(self, 'image_state_manager') and self.image_state_manager:
                state = self.image_state_manager.get_state(original_image_path) or {}
                last_pos = state.get('last_render_positions') or {}
                # Map each rendered region back to a rectangle index via IoU
                try:
                    rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                    def _iou(a, b):
                        ax, ay, aw, ah = a; bx, by, bw, bh = b
                        ax2, ay2 = ax + aw, ay + ah; bx2, by2 = bx + bw, by + bh
                        x1 = max(ax, bx); y1 = max(ay, by); x2 = min(ax2, bx2); y2 = min(ay2, by2)
                        inter = max(0, x2 - x1) * max(0, y2 - y1)
                        area_a = max(0, aw) * max(0, ah); area_b = max(0, bw) * max(0, bh)
                        den = area_a + area_b - inter
                        return (inter / den) if den > 0 else 0.0
                    for r in regions:
                        rx, ry, rw, rh = int(r.bounding_box[0]), int(r.bounding_box[1]), int(r.bounding_box[2]), int(r.bounding_box[3])
                        best_idx, best_iou = None, 0.0
                        for i, rect_item in enumerate(rects):
                            br = rect_item.sceneBoundingRect()
                            cand = [int(br.x()), int(br.y()), int(br.width()), int(br.height())]
                            iou = _iou([rx, ry, rw, rh], cand)
                            if iou > best_iou:
                                best_iou, best_idx = iou, i
                        if best_idx is not None:
                            last_pos[str(int(best_idx))] = [rx, ry, rw, rh]
                except Exception:
                    pass
                state['last_render_positions'] = last_pos
                self.image_state_manager.set_state(original_image_path, state, save=True)
                print(f"[RENDER] Updated last_render_positions for {len(last_pos)} region(s)")
        except Exception as _lp:
            print(f"[RENDER] Failed to update last_render_positions: {_lp}")
        
        # Show the rendered image in the OUTPUT tab (keep source image intact)
        print(f"[RENDER] About to call GUI method to load rendered image...")
        print(f"[RENDER] output_path exists: {os.path.exists(output_path)}")
        print(f"[RENDER] switch_tab: {switch_tab}")
        print(f"[RENDER] Calling _load_rendered_image_to_output_tab now...")
        
        # For GUI operations, we need to be on the main thread
        # The renderer itself completed, now handle the GUI update
        # Use QTimer to ensure this runs on the main thread
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: _load_rendered_image_to_output_tab(self, rendered_pil, output_path, switch_tab))
        
        print(f"[RENDER] GUI method call completed")
        
        self._log(f"✅ Rendered to: {output_filename}", "success")
        
        print(f"[RENDER] _render_with_manga_translator COMPLETED SUCCESSFULLY\n{'='*80}\n")
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"[RENDER] ERROR in _render_with_manga_translator")
        print(f"[RENDER] Error: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[RENDER] Traceback:\n{traceback_str}")
        print(f"{'='*80}\n")
        self._log(f"❌ Rendering error: {str(e)}", "error")
        print(f"[RENDER] Traceback:\n{traceback_str}")
        print(f"{'='*80}\n")
        self._log(f"❌ Rendering error: {str(e)}", "error")

def _load_rendered_image_to_output_tab(self, rendered_pil, output_path, switch_tab=True):
    """Load rendered image into the output tab - must be called on main thread"""
    print(f"[GUI] === _load_rendered_image_to_output_tab CALLED ===")
    print(f"[GUI] output_path: {output_path}")
    print(f"[GUI] switch_tab: {switch_tab}")
    try:
        print(f"[GUI] Loading rendered image into output tab: {os.path.basename(output_path)}")
        
        # Check current thread for debugging
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        
        current_thread = QThread.currentThread()
        main_thread = QApplication.instance().thread() if QApplication.instance() else None
        print(f"[GUI] Thread check: current={current_thread}, main={main_thread}, same={current_thread == main_thread}")
        
        # Display in output viewer (using correct load_image method)
        try:
            if hasattr(self.image_preview_widget, 'output_viewer') and self.image_preview_widget.output_viewer:
                # Use the same method as _check_and_load_translated_output
                self.image_preview_widget.output_viewer.load_image(output_path)
                # Store the translated image path
                self.image_preview_widget.current_translated_path = output_path
                print(f"[GUI] Successfully loaded image into output viewer using load_image")
            else:
                print(f"[GUI] No output_viewer available")
        except Exception as output_err:
            print(f"[GUI] Error loading image to output viewer: {output_err}")
            import traceback
            traceback.print_exc()
        
        
    except Exception as e:
        print(f"[GUI] Error in _load_rendered_image_to_output_tab: {e}")
        import traceback
        traceback.print_exc()

def _load_save_position_output(self):
    """Helper method to load rendered output after save position completes - runs on main thread"""
    try:
        current_image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image_path:
            print(f"[DEBUG] Save Position Output: No current image path")
            return
        
        # Look for rendered image in the expected location
        source_dir = os.path.dirname(current_image_path)
        source_filename = os.path.basename(current_image_path)
        base_name = os.path.splitext(source_filename)[0]
        
        # Check for OUTPUT_DIRECTORY override (prefer config over env var)
        override_dir = None
        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
            override_dir = self.main_gui.config.get('output_directory', '')
        if not override_dir:
            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
        
        # Build list of possible paths
        # STRICT: If OUTPUT_DIRECTORY is set, ONLY check there (no source fallback)
        possible_paths = []
        if override_dir:
            # Check override directory ONLY
            possible_paths.extend([
                os.path.join(override_dir, f"{base_name}_translated", source_filename),
                os.path.join(override_dir, "3_translated", source_filename),
            ])
            print(f"[DEBUG] Save Position Output: Using OUTPUT_DIRECTORY ONLY: {override_dir}")
        else:
            # No override, check source directory
            possible_paths.extend([
                # 3_translated folder
                os.path.join(source_dir, "3_translated", source_filename),
                # isolated folder
                os.path.join(source_dir, f"{base_name}_translated", source_filename),
                # same directory with _translated suffix
                os.path.join(source_dir, f"{base_name}_translated{os.path.splitext(source_filename)[1]}")
            ])
        
        print(f"[DEBUG] Save Position Output: Looking for rendered images at:")
        for path in possible_paths:
            print(f"[DEBUG] Save Position Output:   {path} - exists: {os.path.exists(path)}")
            if os.path.exists(path):
                print(f"[DEBUG] Save Position Output: Found rendered image, loading into output viewer...")
                self.image_preview_widget.output_viewer.load_image(path)
                self.image_preview_widget.current_translated_path = path
                # REMOVED: Don't auto-switch tabs - let user manually switch
                # if hasattr(self.image_preview_widget, 'viewer_tabs'):
                #     self.image_preview_widget.viewer_tabs.setCurrentIndex(1)  # Switch to output tab
                print(f"[DEBUG] Save Position Output: Successfully loaded {os.path.basename(path)} into output viewer")
                return
        
        print(f"[DEBUG] Save Position Output: No rendered image found in expected locations")
        
    except Exception as e:
        print(f"[DEBUG] Save Position Output: Error loading rendered output: {e}")
        import traceback
        traceback.print_exc()

@Slot()
def _load_rendered_output_direct(self):
    """Direct method to find and load rendered image into output tab - same as working button approach"""
    try:
        current_image_path = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image_path:
            print(f"[DIRECT] No current image path")
            return
        
        # Look for rendered image in the expected location
        source_dir = os.path.dirname(current_image_path)
        source_filename = os.path.basename(current_image_path)
        base_name = os.path.splitext(source_filename)[0]
        
        # Check for OUTPUT_DIRECTORY override (prefer config over env var)
        override_dir = None
        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
            override_dir = self.main_gui.config.get('output_directory', '')
        if not override_dir:
            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
        
        # Build list of possible paths
        # STRICT: If OUTPUT_DIRECTORY is set, ONLY check there (no source fallback)
        possible_paths = []
        if override_dir:
            # Check override directory ONLY
            possible_paths.extend([
                os.path.join(override_dir, f"{base_name}_translated", source_filename),
                os.path.join(override_dir, "3_translated", source_filename),
            ])
            print(f"[DIRECT] Using OUTPUT_DIRECTORY ONLY: {override_dir}")
        else:
            # No override, check source directory
            possible_paths.extend([
                # 3_translated folder
                os.path.join(source_dir, "3_translated", source_filename),
                # isolated folder
                os.path.join(source_dir, f"{base_name}_translated", source_filename),
                # same directory with _translated suffix
                os.path.join(source_dir, f"{base_name}_translated{os.path.splitext(source_filename)[1]}")
            ])
        
        print(f"[DIRECT] Looking for rendered images at:")
        for path in possible_paths:
            print(f"[DIRECT]   {path} - exists: {os.path.exists(path)}")
            if os.path.exists(path):
                print(f"[DIRECT] Found rendered image, loading into output viewer...")
                self.image_preview_widget.output_viewer.load_image(path)
                self.image_preview_widget.current_translated_path = path
                # REMOVED: Don't auto-switch tabs - let user manually switch
                # if hasattr(self.image_preview_widget, 'viewer_tabs'):
                #     self.image_preview_widget.viewer_tabs.setCurrentIndex(1)  # Switch to output tab
                print(f"[DIRECT] Successfully loaded {os.path.basename(path)} into output viewer")
                return
        
        print(f"[DIRECT] No rendered image found in expected locations")
        
    except Exception as e:
        print(f"[DIRECT] Error in _load_rendered_output_direct: {e}")
        import traceback
        traceback.print_exc()

def _add_text_overlay_to_viewer(self, translated_texts: list):
    """Add translated text as graphics items overlay on the viewer
    
    Overlays are hidden by default if at original position (overlaps with rendered output).
    When user moves blue rectangles (auto-save position), overlays move with them and become visible.
    """
    try:
        from PySide6.QtWidgets import QGraphicsTextItem, QGraphicsRectItem
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QBrush, QPen, QFont
        
        viewer = self.image_preview_widget.viewer
        
        # Get current image path
        current_image = self.image_preview_widget.current_image_path
        if not current_image:
            print("[DEBUG] No current image path, cannot add overlays")
            return
        
        # Initialize overlay dictionary if not exists
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        
        # Clear any existing overlays for this specific image with proper Qt cleanup
        if current_image in self._text_overlays_by_image:
            for overlay in self._text_overlays_by_image[current_image]:
                try:
                    # Remove from scene
                    viewer._scene.removeItem(overlay)
                    # Destroy all child items to free memory
                    for child in overlay.childItems():
                        try:
                            child.setParentItem(None)
                            child.deleteLater()
                        except Exception:
                            pass
                    # Destroy the group itself
                    overlay.deleteLater()
                except Exception:
                    pass
        
        # Create new list for this image's overlays
        self._text_overlays_by_image[current_image] = []
        
        # Load any saved overlay offsets for this image
        saved_offsets = {}
        try:
            if hasattr(self, 'image_state_manager') and self.image_state_manager:
                st = self.image_state_manager.get_state(current_image) or {}
                saved_offsets = st.get('overlay_offsets') or {}
        except Exception:
            saved_offsets = {}
        
        # Get manga rendering settings
        manga_settings = _get_manga_rendering_settings(self, )
        
        # Source tab overlays should not force any background opacity
        try:
            manga_settings['show_background'] = False
            manga_settings['bg_opacity'] = 0
        except Exception:
            pass
        
        
        for i, result in enumerate(translated_texts):
            try:
                bbox = result.get('bbox')
                translation = result.get('translation', '')
                region_index = (result.get('original', {}) or {}).get('region_index', i)
                
                # Prefer current BLUE rectangle geometry if available; fallback to bbox
                x = y = w = h = None
                original_bbox = bbox  # Store original bbox for overlap detection
                try:
                    rects = getattr(self.image_preview_widget.viewer, 'rectangles', []) or []
                    if 0 <= int(region_index) < len(rects):
                        br = rects[int(region_index)].sceneBoundingRect()
                        x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
                except Exception:
                    pass
                if (x is None or y is None or w is None or h is None) and bbox and len(bbox) >= 4:
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                if x is not None and w is not None and h is not None and w > 0 and h > 0:
                    
                    # Create background rectangle if enabled in settings
                    bg_rect = None
                    if manga_settings.get('show_background', True):
                        bg_rect = _create_background_shape(self, x, y, w, h, manga_settings)
                        if bg_rect:
                            bg_rect.setZValue(10)  # Above image, below text
                            viewer._scene.addItem(bg_rect)
                    
                    # Apply text formatting
                    text = translation.upper() if manga_settings.get('force_caps', False) else translation
                    
                    # Create text item with proper manga text rendering
                    text_item, final_font_size = _create_manga_text_item(self, text, x, y, w, h, manga_settings)
                    if text_item is None:
                        # Clean up orphan bg if created
                        try:
                            if bg_rect:
                                viewer._scene.removeItem(bg_rect)
                        except Exception:
                            pass
                        continue
                    
                    # Add text to scene (needed before grouping)
                    viewer._scene.addItem(text_item)
                    
                    # Make text item completely non-interactive
                    try:
                        from PySide6.QtCore import Qt
                        from PySide6.QtWidgets import QGraphicsItem
                        text_item.setAcceptedMouseButtons(Qt.NoButton)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                        text_item.setAcceptHoverEvents(False)
                    except Exception:
                        pass
                    
                    # Create a transparent rectangle overlay that covers the text and forwards clicks
                    overlay_rect = QGraphicsRectItem(x, y, w, h)
                    # Keep overlay rect in sync with rectangle moves by storing size
                    try:
                        overlay_rect._overlay_bbox_size = (w, h)
                    except Exception:
                        pass
                    overlay_rect.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Completely transparent
                    overlay_rect.setPen(QPen(QColor(0, 0, 0, 0)))  # No border
                    overlay_rect.setZValue(20)  # Above everything else to capture clicks
                    
                    # Store the region index on the overlay for identification
                    overlay_rect.region_index = region_index
                    
                    # Make text item completely non-interactive to avoid conflicts
                    try:
                        text_item.setAcceptedMouseButtons(Qt.NoButton)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                        text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                        text_item.setAcceptHoverEvents(False)
                    except Exception:
                        pass
                    
                    viewer._scene.addItem(overlay_rect)
                    
                    # Group background + text + overlay so they stay together
                    group_items = [text_item, overlay_rect]
                    if bg_rect:
                        group_items.insert(0, bg_rect)  # bg_rect first (lowest z-order)
                        # Make bg_rect completely non-interactive
                        try:
                            from PySide6.QtCore import Qt
                            from PySide6.QtWidgets import QGraphicsItem
                            bg_rect.setAcceptedMouseButtons(Qt.NoButton)
                            bg_rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                            bg_rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                            bg_rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                            bg_rect.setAcceptHoverEvents(False)
                        except Exception:
                            pass
                    
                    group = viewer._scene.createItemGroup(group_items)
                    group.setZValue(12)
                    # Store references for later reflowing/resizing
                    try:
                        group._overlay_text_item = text_item
                        group._overlay_bg_item = bg_rect
                        group._overlay_original_text = translation
                    except Exception:
                        pass
                    from PySide6.QtWidgets import QGraphicsItem
                    # Disable user interaction on overlays; movement controlled by rectangles only
                    # But allow child items (like text) to handle mouse events
                    try:
                        from PySide6.QtCore import Qt
                        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                        # Don't block mouse buttons on group level - let child items handle them
                        # group.setAcceptedMouseButtons(Qt.NoButton)
                    except Exception:
                        pass
                    
                    # Attach metadata for sync from rectangle moves
                    group._overlay_region_index = region_index
                    group._overlay_bbox_size = (w, h)
                    group._overlay_image_path = current_image
                    
                    # Store original bbox for overlap detection
                    group._overlay_original_bbox = original_bbox
                    
                    # Apply saved offset for this region if present
                    has_offset = False
                    try:
                        off = None
                        # support str and int keys
                        if str(int(region_index)) in saved_offsets:
                            off = saved_offsets[str(int(region_index))]
                        elif int(region_index) in saved_offsets:
                            off = saved_offsets[int(region_index)]
                        if off and len(off) >= 2:
                            dx_off, dy_off = int(off[0]), int(off[1])
                            if dx_off != 0 or dy_off != 0:
                                group.moveBy(dx_off, dy_off)
                                has_offset = True
                    except Exception:
                        pass
                    
                    # Always hide text overlays (keep them hidden)
                    try:
                        group.setVisible(False)
                        #print(f"[DEBUG] Hiding overlay for region {region_index} - text overlays kept hidden")
                    except Exception as hide_err:
                        print(f"[DEBUG] Error setting overlay visibility: {hide_err}")
                    
                    # Monkey-patch mouse release remains (harmless since overlays are not movable)
                    original_release = group.mouseReleaseEvent
                    def _on_release(ev, grp=group):
                        try:
                            original_release(ev)
                        except Exception:
                            pass
                        try:
                            # Compute new top-left from scene bounding rect
                            br = grp.sceneBoundingRect()
                            new_x, new_y = int(br.x()), int(br.y())
                            w_, h_ = grp._overlay_bbox_size
                            # Update viewer rectangle for this region if available
                            rects = getattr(self.image_preview_widget.viewer, 'rectangles', [])
                            idx = int(grp._overlay_region_index) if grp._overlay_region_index is not None else -1
                            if 0 <= idx < len(rects):
                                from PySide6.QtCore import QRectF as _QRectF
                                rects[idx].setRect(_QRectF(new_x, new_y, w_, h_))
                                # Also trigger a scene update so handles refresh
                                try:
                                    self.image_preview_widget.viewer._scene.update()
                                except Exception:
                                    pass
                            # Persist new rectangles state
                            try:
                                if hasattr(self.image_preview_widget, '_persist_rectangles_state'):
                                    self.image_preview_widget._persist_rectangles_state()
                            except Exception:
                                pass
                        except Exception as move_err:
                            print(f"[DEBUG] Overlay move update failed: {move_err}")
                    group.mouseReleaseEvent = _on_release
                    
                    # Track overlay group for cleanup
                    self._text_overlays_by_image[current_image].append(group)
                    
                    print(f"[DEBUG] Added text overlay at ({x},{y}) with font size {final_font_size}: region={region_index}, text='{translation[:30]}...'")
            
            except Exception as text_error:
                print(f"[DEBUG] Error adding text overlay: {str(text_error)}")
                import traceback
                print(f"[DEBUG] Text overlay traceback: {traceback.format_exc()}")
                continue
        
        overlay_count = len(self._text_overlays_by_image.get(current_image, []))
        print(f"[DEBUG] Added {overlay_count} text overlay items for image: {os.path.basename(current_image)}")
        
        # Force scene update to ensure overlays are visible
        viewer._scene.update()
        print(f"[DEBUG] Forced scene update")
        
    except Exception as e:
        print(f"[DEBUG] Error adding text overlays: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")

def _add_text_overlay_for_region(self, region_index: int, original_text: str, translation: str, bbox: list = None):
    """Add or replace a single text overlay for the given region on the main thread.
    Does NOT clear overlays for other regions.
    """
    try:
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtGui import QColor, QBrush, QPen
        from PySide6.QtCore import QRectF
        
        viewer = self.image_preview_widget.viewer
        current_image = getattr(self.image_preview_widget, 'current_image_path', None)
        if not current_image:
            return
        
        # Init tracking map
        if not hasattr(self, '_text_overlays_by_image'):
            self._text_overlays_by_image = {}
        if current_image not in self._text_overlays_by_image:
            self._text_overlays_by_image[current_image] = []
        
        # Remove existing overlay for this region with proper Qt cleanup
        to_remove = []
        for grp in list(self._text_overlays_by_image[current_image]):
            if getattr(grp, '_overlay_region_index', None) == int(region_index):
                to_remove.append(grp)
        for grp in to_remove:
            try:
                # Remove from scene
                viewer._scene.removeItem(grp)
                # Destroy all child items
                for child in grp.childItems():
                    try:
                        child.setParentItem(None)
                        child.deleteLater()
                    except Exception:
                        pass
                # Destroy the group
                grp.deleteLater()
                # Remove from tracking list
                self._text_overlays_by_image[current_image].remove(grp)
            except Exception:
                pass
        
        # Determine geometry from current rectangle if present
        x = y = w = h = None
        try:
            rects = getattr(viewer, 'rectangles', []) or []
            if 0 <= int(region_index) < len(rects):
                br = rects[int(region_index)].sceneBoundingRect()
                x, y, w, h = int(br.x()), int(br.y()), int(br.width()), int(br.height())
        except Exception:
            pass
        if (x is None or y is None or w is None or h is None) and bbox and len(bbox) >= 4:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x is None or w is None or h is None or w <= 0 or h <= 0:
            return
        
        # Render text item using existing helper (uses current GUI settings)
        settings = _get_manga_rendering_settings(self, ) or {}
        try:
            settings['show_background'] = False
            settings['bg_opacity'] = 0
        except Exception:
            pass
        text_item, _ = _create_manga_text_item(self, translation, x, y, w, h, settings)
        if text_item is None:
            return
        viewer._scene.addItem(text_item)
        
        # Transparent overlay rect for click passthrough/context
        overlay_rect = QGraphicsRectItem(x, y, w, h)
        overlay_rect.setBrush(QBrush(QColor(0, 0, 0, 0)))
        overlay_rect.setPen(QPen(QColor(0, 0, 0, 0)))
        overlay_rect.setZValue(20)
        overlay_rect.region_index = int(region_index)
        viewer._scene.addItem(overlay_rect)
        
        # Optional background (disabled for source view)
        bg_rect = None
        
        # Group items and tag metadata
        group_items = [text_item, overlay_rect]
        group = viewer._scene.createItemGroup(group_items)
        group.setZValue(12)
        try:
            group._overlay_text_item = text_item
            group._overlay_bg_item = bg_rect
            group._overlay_original_text = translation
            group._overlay_region_index = int(region_index)
            group._overlay_bbox_size = (w, h)
            group._overlay_image_path = current_image
        except Exception:
            pass
        
        self._text_overlays_by_image[current_image].append(group)
        viewer._scene.update()
    except Exception:
        pass

def _get_manga_rendering_settings(self) -> dict:
    """Stub - overlays are hidden, rendering happens in manga_translator.py"""
    return {}

def show_recognized_overlays_for_image(self, image_path: str):
    """Restore recognized (OCR) overlays for the given image and attach tooltips.
    - Draw detection rectangles if not present
    - Apply recognition tooltips/updates if recognized_texts exist in state
    """
    try:
        if not hasattr(self, 'image_state_manager') or not image_path:
            return
        state = self.image_state_manager.get_state(image_path)
        if not state:
            return
        # Ensure rectangles exist; if not, draw detection regions first
        need_draw = False
        try:
            rects = getattr(self.image_preview_widget.viewer, 'rectangles', [])
            need_draw = (not rects)
        except Exception:
            need_draw = True
        if need_draw:
            regions = state.get('detection_regions') or []
            if regions:
                self._current_regions = regions
                try:
                    if hasattr(self.image_preview_widget.viewer, 'clear_rectangles'):
                        self.image_preview_widget.viewer.clear_rectangles()
                except Exception:
                    pass
                _draw_detection_boxes_on_preview(self, )
        # Apply recognition data if available
        recognized_texts = state.get('recognized_texts') or []
        if recognized_texts:
            _update_rectangles_with_recognition(self, recognized_texts)
    except Exception as e:
        print(f"[DEBUG] Error restoring recognized overlays: {e}")

def _create_manga_text_item(self, text: str, x: int, y: int, w: int, h: int, settings: dict):
    """Stub - overlays are hidden, actual rendering happens in manga_translator.py"""
    return None, 0


def _calculate_auto_font_size(self, text: str, width: int, height: int, settings: dict) -> int:
    """Stub - redundant, manga_translator.py has the real algorithm"""
    return 12


def _text_fits_in_bounds(self, text: str, width: int, height: int, font, settings: dict) -> bool:
    """Stub - redundant"""
    return True

def _wrap_text_for_bubble(self, text: str, max_width: int, font, settings: dict) -> str:
    """Stub - redundant"""
    return text


def _create_background_shape(self, x: int, y: int, w: int, h: int, settings: dict):
    """Stub - overlays are hidden"""
    return None

def _wrap_text_to_width(self, text: str, max_width: int, font) -> str:
    """Stub - redundant"""
    return text

def _disable_workflow_buttons(self, exclude=None, show_stop_button=True):
    """Disable all workflow buttons to prevent concurrent operations.
    
    Args:
        exclude: Button name to exclude from disabling (e.g., 'translate' keeps translate enabled)
        show_stop_button: Whether to show the workflow stop button (False for Start Translation which has its own stop)
    """
    try:
        ipw = self.image_preview_widget if hasattr(self, 'image_preview_widget') else None
        if not ipw:
            return
        
        buttons = [
            ('detect_btn', 'Detect Text'),
            ('clean_btn', 'Clean'),
            ('recognize_btn', 'Recognize Text'),
            ('translate_btn', 'Translate'),
            ('translate_all_btn', 'Translate All'),
        ]
        
        for btn_name, _ in buttons:
            if exclude and btn_name == exclude:
                continue
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(False)
        
        # Disable editing tools during processing
        editing_buttons = ['save_overlay_btn', 'delete_btn', 'clear_boxes_btn', 'box_draw_btn', 'circle_draw_btn', 'lasso_btn']
        for btn_name in editing_buttons:
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(False)
        
        # Also disable start_button in manga_integration if it exists
        if exclude != 'start_button' and hasattr(self, 'start_button') and self.start_button:
            self.start_button.setEnabled(False)
        
        # Show the stop button when workflow operations start (not for Start Translation)
        if show_stop_button and hasattr(ipw, 'stop_translation_btn'):
            ipw.stop_translation_btn.setVisible(True)
            ipw.stop_translation_btn.setEnabled(True)
            ipw.stop_translation_btn.setText("⏹ Stop")
        
        print(f"[WORKFLOW] Disabled workflow buttons (exclude={exclude}, show_stop={show_stop_button})")
    except Exception as e:
        print(f"[WORKFLOW] Error disabling buttons: {e}")

def _enable_workflow_buttons(self):
    """Re-enable all workflow buttons after operation completes."""
    try:
        ipw = self.image_preview_widget if hasattr(self, 'image_preview_widget') else None
        if not ipw:
            return
        
        buttons = [
            ('detect_btn', 'Detect Text'),
            ('clean_btn', 'Clean'),
            ('recognize_btn', 'Recognize Text'),
            ('translate_btn', 'Translate'),
            ('translate_all_btn', 'Translate All'),
        ]
        
        for btn_name, default_text in buttons:
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(True)
                btn.setText(default_text)
        
        # Also re-enable start_button in manga_integration if it exists
        if hasattr(self, 'start_button') and self.start_button:
            self.start_button.setEnabled(True)
        
        # Re-enable editing tools after processing
        editing_buttons = ['save_overlay_btn', 'delete_btn', 'clear_boxes_btn', 'box_draw_btn', 'circle_draw_btn', 'lasso_btn']
        for btn_name in editing_buttons:
            if hasattr(ipw, btn_name):
                btn = getattr(ipw, btn_name)
                btn.setEnabled(True)
        
        # Hide the stop button when operations complete
        if hasattr(ipw, 'stop_translation_btn'):
            ipw.stop_translation_btn.setVisible(False)
            ipw.stop_translation_btn.setEnabled(True)
            ipw.stop_translation_btn.setText("⏹ Stop")
        
        print(f"[WORKFLOW] Re-enabled all workflow buttons")
    except Exception as e:
        print(f"[WORKFLOW] Error enabling buttons: {e}")

def _restore_translate_button(self):
    """Restore the translate button to its original state"""
    try:
        # Remove processing overlay effect for the image that was being processed
        image_path = getattr(self, '_translating_image_path', None)
        _remove_processing_overlay(self, image_path)
        
        # CRITICAL: Restore print hijacking if MangaTranslator exists
        if hasattr(self, '_manga_translator') and self._manga_translator:
            try:
                self._manga_translator.restore_print()
            except Exception:
                pass
        
        # Re-enable ALL workflow buttons (not just translate)
        _enable_workflow_buttons(self)
        
        # Re-enable thumbnail list
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(True)
            print(f"[TRANSLATE] Re-enabled thumbnail list after translation")
        
        # Switch display mode to 'translated' so user sees the result
        try:
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
                current_image = getattr(ipw, 'current_image_path', None)
                if current_image:
                    ipw.load_image(current_image, preserve_rectangles=True, preserve_text_overlays=True)
                
                print(f"[TRANSLATE] Switched display mode to 'translated'")
        except Exception as mode_err:
            print(f"[TRANSLATE] Failed to switch display mode: {mode_err}")
    except Exception:
        pass

def _on_translate_all_clicked(self):
    """Translate all images in the preview list"""
    self._log("🚀 Starting batch translation of all images", "info")
    
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    # This MUST happen on the main thread BEFORE any cancellation checks
    _reset_cancellation_flags(self)
    
    try:
        # Mark batch mode active (used to suppress preview shuffles)
        self._batch_mode_active = True
        # Snapshot toggles on UI thread so background can read safely
        try:
            fp = False
            vc = False
            if hasattr(self, 'context_checkbox'):
                fp = bool(self.context_checkbox.isChecked())
            else:
                fp = bool(self.main_gui.config.get('manga_full_page_context', False))
            if hasattr(self, 'visual_context_checkbox'):
                vc = bool(self.visual_context_checkbox.isChecked())
            else:
                vc = bool(self.main_gui.config.get('manga_visual_context_enabled', False))
            # Store snapshots for background thread
            self._batch_full_page_context_enabled = fp
            self._batch_visual_context_enabled = vc
            print(f"[DEBUG] Snapshot toggles for batch: full_page_context={fp}, visual_context={vc}")
        except Exception as snap_err:
            print(f"[DEBUG] Toggle snapshot failed: {snap_err}")
        
        # Check if we have images in the preview
        if not hasattr(self.image_preview_widget, 'image_paths') or not self.image_preview_widget.image_paths:
            self._log("⚠️ No images loaded in preview", "warning")
            return
        
        image_paths = self.image_preview_widget.image_paths
        total_images = len(image_paths)
        
        self._log(f"📋 Found {total_images} images to translate", "info")
        
        # Disable ALL workflow buttons to prevent concurrent operations
        _disable_workflow_buttons(self, exclude=None)
        
        # Update translate all button text to show progress
        if hasattr(self.image_preview_widget, 'translate_all_btn'):
            self.image_preview_widget.translate_all_btn.setText(f"Translating... (0/{total_images})")
        
        # Disable thumbnail list to prevent user from switching images during translation
        if hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(False)
            print(f"[TRANSLATE_ALL] Disabled thumbnail list during batch translation")
        
        # Add blue pulse processing overlay
        _add_processing_overlay(self, )
        
        # Run in background thread
        import threading
        thread = threading.Thread(target=_run_translate_all_background, args=(self, image_paths),
                                daemon=True)
        thread.start()
        
    except Exception as e:
        import traceback
        self._log(f"❌ Translate all setup failed: {str(e)}", "error")
        print(f"Translate all error traceback: {traceback.format_exc()}")
        _restore_translate_all_button(self, )

def _run_translate_all_background(self, image_paths: list):
    """Run translation for all images in background"""
    # ===== RESET FLAGS: Clear any stale cancellation from previous ops =====
    _reset_cancellation_flags(self)
    
    try:
        total = len(image_paths)
        translated_count = 0
        failed_count = 0
        
        self._log(f"🌍 Starting batch translation: {total} images", "info")
        
        for idx, image_path in enumerate(image_paths, 1):
            # ===== CANCELLATION CHECK: Start of each image =====
            if _is_translation_cancelled(self):
                self._log(f"⏹ Translation cancelled at image {idx}/{total}", "warning")
                print(f"[TRANSLATE_ALL] Cancelled at start of image {idx}")
                break
            
            try:
                import time
                
                # Load this image into the preview FIRST so user can see what's being processed
                self.update_queue.put(('load_preview_image', {
                    'path': image_path,
                    'preserve_rectangles': False,
                    'preserve_overlays': False
                }))
                
                # Sync file list and thumbnail selection to match current image
                self.update_queue.put(('sync_file_selection', {
                    'image_path': image_path
                }))
                print(f"[TRANSLATE_ALL] Switched preview to: {os.path.basename(image_path)}")
                
                # Wait for UI to process the image load before starting detection
                # This ensures the user sees the image switch before detection boxes appear
                time.sleep(0.8)
                
                # Re-add processing overlay after image switch (overlay gets cleared when scene changes)
                self.update_queue.put(('add_processing_overlay', None))
                
                # ===== CANCELLATION CHECK: After image load =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled after loading image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled after loading image {idx}")
                    break
                
                # Update progress
                self._log(f"📄 [{idx}/{total}] Processing: {os.path.basename(image_path)}", "info")
                
                # Update button text on main thread
                self.update_queue.put(('translate_all_progress', {
                    'current': idx,
                    'total': total
                }))
                
                # Run the full translate pipeline for this image
                # This includes detect -> recognize -> translate
                regions_for_recognition = None  # Will trigger auto-detection
                
                # Get OCR config
                ocr_config = _get_ocr_config(self, )
                
                # Step 1: Run detection
                detection_config = _get_detection_config(self, )
                if detection_config.get('detect_empty_bubbles', True):
                    detection_config['detect_empty_bubbles'] = False
                
                regions = _run_detection_sync(self, image_path, detection_config)
                if not regions:
                    self._log(f"⚠️ [{idx}/{total}] No text regions detected", "warning")
                    failed_count += 1
                    continue
                
                # ===== CANCELLATION CHECK: After detection =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled after detection on image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled after detection on image {idx}")
                    break
                
                self._log(f"✅ [{idx}/{total}] Detected {len(regions)} regions", "success")
                
                # Send detection results to main thread to draw GREEN boxes
                self.update_queue.put(('detect_results', {
                    'image_path': image_path,
                    'regions': regions
                }))
                
                # Save state after detection
                self.image_state_manager.update_state(image_path, {
                    'detection_regions': regions,
                    'step': 'detected'
                })
                
                # Brief pause so user can see green detection boxes
                time.sleep(0.3)
                
                # Step 2: Run OCR
                # ===== CANCELLATION CHECK: Before OCR =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled before OCR on image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled before OCR on image {idx}")
                    break
                
                recognized_texts = _run_ocr_on_regions(self, image_path, regions, ocr_config)
                if not recognized_texts:
                    self._log(f"⚠️ [{idx}/{total}] No text recognized", "warning")
                    failed_count += 1
                    continue
                
                # ===== CANCELLATION CHECK: After OCR =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled after OCR on image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled after OCR on image {idx}")
                    break
                
                self._log(f"✅ [{idx}/{total}] Recognized {len(recognized_texts)} text regions", "success")
                
                # Send recognition results to main thread to draw BLUE boxes
                self.update_queue.put(('recognize_results', {
                    'image_path': image_path,
                    'recognized_texts': recognized_texts
                }))
                
                # Save state after recognition
                self.image_state_manager.update_state(image_path, {
                    'recognized_texts': recognized_texts,
                    'step': 'recognized'
                })
                
                # Brief pause so user can see blue recognition boxes
                time.sleep(0.3)
                
                # Step 2.5: Run inpainting/cleaning if enabled (optional visual step)
                cleaned_path = None
                try:
                    inpaint_config = _get_inpaint_config(self, )
                    inpaint_method = inpaint_config.get('method', 'none')
                    
                    # Only run inpainting if method is not 'none' and is 'local' or 'hybrid'
                    if inpaint_method in ['local', 'hybrid']:
                        self._log(f"🧹 [{idx}/{total}] Cleaning image...", "info")
                        cleaned_path = _run_inpainting_sync(self, image_path, regions)
                        
                        if cleaned_path and os.path.exists(cleaned_path):
                            # Store cleaned image path for rendering
                            self._cleaned_image_path = cleaned_path
                            # Load cleaned image in preview, preserving rectangles
                            self.update_queue.put(('load_preview_image', {
                                'path': cleaned_path,
                                'preserve_rectangles': True,
                                'preserve_overlays': True
                            }))
                            self._log(f"✅ [{idx}/{total}] Image cleaned", "success")
                            
                            # Save state after cleaning
                            self.image_state_manager.update_state(image_path, {
                                'cleaned_image_path': cleaned_path,
                                'step': 'cleaned'
                            })
                            
                            # Brief pause to show cleaned image
                            time.sleep(0.5)
                            
                            # Re-add processing overlay after cleaned image load
                            self.update_queue.put(('add_processing_overlay', None))
                            # Use cleaned image for translation rendering
                            image_path_for_rendering = cleaned_path
                        else:
                            self._log(f"⚠️ [{idx}/{total}] Cleaning failed, using original", "warning")
                            image_path_for_rendering = image_path
                    else:
                        # No cleaning, use original image
                        image_path_for_rendering = image_path
                        self._cleaned_image_path = None
                except Exception as e:
                    self._log(f"⚠️ [{idx}/{total}] Cleaning error: {str(e)}", "warning")
                    import traceback
                    print(f"[TRANSLATE_ALL] Cleaning error: {traceback.format_exc()}")
                    image_path_for_rendering = image_path
                    self._cleaned_image_path = None
                
                # Step 3: Run translation (mirror regular translate behavior)
                # Decide full-page vs individual using batch snapshot (set on UI thread)
                full_page_context_enabled = False
                if hasattr(self, '_batch_full_page_context_enabled'):
                    full_page_context_enabled = bool(self._batch_full_page_context_enabled)
                    print(f"[DEBUG] (Batch) Full page context: {full_page_context_enabled}")
                else:
                    try:
                        full_page_context_enabled = bool(self.main_gui.config.get('manga_full_page_context', False))
                    except Exception:
                        full_page_context_enabled = False
                    print(f"[DEBUG] (Batch) Full page context from config: {full_page_context_enabled}")
                
                # ===== CANCELLATION CHECK: Before translation =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled before translating image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled before translation on image {idx}")
                    break
                
                if full_page_context_enabled:
                    print(f"[DEBUG] Using FULL PAGE CONTEXT translation mode (batch)")
                    self._log(f"📄 [{idx}/{total}] Using full page context translation for {len(recognized_texts)} regions", "info")
                    translated_texts = _translate_with_full_page_context(self, recognized_texts, image_path)
                else:
                    print(f"[DEBUG] Using INDIVIDUAL translation mode (batch)")
                    self._log(f"📝 [{idx}/{total}] Using individual translation for {len(recognized_texts)} regions", "info")
                    translated_texts = _translate_individually(self, recognized_texts, image_path)
                
                # ===== CANCELLATION CHECK: After translation (critical - prevents raw text from being shown) =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled - discarding results for image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled after translation - NOT sending results for image {idx}")
                    break
                
                if not translated_texts:
                    self._log(f"⚠️ [{idx}/{total}] Translation failed", "warning")
                    failed_count += 1
                    continue
                
                self._log(f"✅ [{idx}/{total}] Translated {len(translated_texts)} regions", "success")
                
                # ===== CANCELLATION CHECK: Before sending results (final gate) =====
                if _is_translation_cancelled(self):
                    self._log(f"⏹ Translation cancelled - NOT rendering results for image {idx}/{total}", "warning")
                    print(f"[TRANSLATE_ALL] Cancelled - NOT sending translate_results for image {idx}")
                    break
                
                # Send results to main thread for rendering
                # Use cleaned image if available, otherwise original
                render_image_path = image_path_for_rendering if 'image_path_for_rendering' in locals() else image_path
                self.update_queue.put(('translate_results', {
                    'image_path': render_image_path,
                    'translated_texts': translated_texts,
                    'original_image_path': image_path  # Keep track of original for mapping
                }))
                
                # Save state after translation
                self.image_state_manager.update_state(image_path, {
                    'translated_texts': translated_texts,
                    'step': 'translated'
                })
                
                # Wait for rendering to complete
                time.sleep(1.0)
                
                # Switch to translated display mode and refresh preview to show the result
                self.update_queue.put(('switch_to_translated_mode', {
                    'image_path': image_path
                }))
                print(f"[TRANSLATE_ALL] Switched to translated mode for: {os.path.basename(image_path)}")
                
                # Give user time to see the final result before moving to next image
                time.sleep(1.0)
                
                translated_count += 1
                
            except Exception as e:
                import traceback
                self._log(f"❌ [{idx}/{total}] Error: {str(e)}", "error")
                print(f"[TRANSLATE_ALL] Error on image {idx}: {traceback.format_exc()}")
                failed_count += 1
                continue
        
        # Final summary
        self._log(f"\n🎉 Batch translation complete!", "success")
        self._log(f"   ✅ Successful: {translated_count}/{total}", "success")
        if failed_count > 0:
            self._log(f"   ❌ Failed: {failed_count}/{total}", "error")
        
        # After all processing, update the thumbnail list to show rendered images
        self.update_queue.put(('update_preview_to_rendered', None))
        
    except Exception as e:
        import traceback
        self._log(f"❌ Batch translation failed: {str(e)}", "error")
        print(f"[TRANSLATE_ALL] Fatal error: {traceback.format_exc()}")
    finally:
        print(f"[TRANSLATE_ALL] Finally block executing - sending restore messages")
        # Remove blue pulse overlay
        self.update_queue.put(('remove_processing_overlay', None))
        # Restore button
        self.update_queue.put(('translate_all_button_restore', None))
        print(f"[TRANSLATE_ALL] Sent translate_all_button_restore to queue")
        # Clear batch mode flag
        try:
            self._batch_mode_active = False
        except Exception:
            pass

def _restore_translate_all_button(self):
    """Restore the translate all button to its original state"""
    print(f"[TRANSLATE_ALL] _restore_translate_all_button called")
    try:
        # Re-enable ALL workflow buttons (not just translate all)
        _enable_workflow_buttons(self)
        
        # Re-enable thumbnail list
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'thumbnail_list'):
            self.image_preview_widget.thumbnail_list.setEnabled(True)
            print(f"[TRANSLATE_ALL] Re-enabled thumbnail list after batch translation")
        
        # Explicitly reset translate_all_btn text (in case _enable_workflow_buttons missed it)
        if hasattr(self, 'image_preview_widget') and hasattr(self.image_preview_widget, 'translate_all_btn'):
            self.image_preview_widget.translate_all_btn.setText("Translate All")
            self.image_preview_widget.translate_all_btn.setEnabled(True)
            print(f"[TRANSLATE_ALL] Explicitly reset translate_all_btn text")
        
        # Switch display mode to 'translated' so user sees the result
        try:
            ipw = self.image_preview_widget
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
                
                # Go back to the first image in the preview list
                if hasattr(ipw, 'image_paths') and ipw.image_paths:
                    first_image = ipw.image_paths[0]
                    ipw.load_image(first_image, preserve_rectangles=True, preserve_text_overlays=True)
                    # Also update thumbnail selection to first item
                    if hasattr(ipw, 'thumbnail_list') and ipw.thumbnail_list.count() > 0:
                        ipw.thumbnail_list.setCurrentRow(0)
                    # Sync file_listbox selection to first item
                    if hasattr(self, 'file_listbox') and self.file_listbox and self.file_listbox.count() > 0:
                        self.file_listbox.setCurrentRow(0)
                        print(f"[TRANSLATE_ALL] Synced file_listbox to first item")
                    print(f"[TRANSLATE_ALL] Returned to first image: {os.path.basename(first_image)}")
                else:
                    # Fallback: refresh current image
                    current_image = getattr(ipw, 'current_image_path', None)
                    if current_image:
                        ipw.load_image(current_image, preserve_rectangles=True, preserve_text_overlays=True)
                
                print(f"[TRANSLATE_ALL] Switched display mode to 'translated'")
        except Exception as mode_err:
            print(f"[TRANSLATE_ALL] Failed to switch display mode: {mode_err}")
    except Exception as e:
        print(f"[TRANSLATE_ALL] Error in _restore_translate_all_button: {e}")
        import traceback
        traceback.print_exc()

def _update_preview_to_rendered_images(self):
    """On batch end, show translated result for the CURRENT selection only; do not change lists or selection."""
    try:
        if not hasattr(self, '_rendered_images_map') or not self._rendered_images_map:
            print("[UPDATE_PREVIEW] No rendered images to show")
            return
        
        # Determine current selection
        current_row = self.file_listbox.currentRow() if hasattr(self, 'file_listbox') else -1
        if current_row is None or current_row < 0 or current_row >= len(self.selected_files):
            print("[UPDATE_PREVIEW] No current selection; not changing preview")
            return
        current_source = self.selected_files[current_row]
        
        # If a rendered image exists for the currently selected source, store the path and refresh preview
        rendered_path = self._rendered_images_map.get(current_source)
        if rendered_path and os.path.exists(rendered_path):
            if hasattr(self, 'image_preview_widget'):
                # Store the translated path
                self.image_preview_widget.current_translated_path = rendered_path
                # Refresh the preview to show the rendered result
                self.image_preview_widget.load_image(current_source, preserve_rectangles=True, preserve_text_overlays=True)
                print(f"[UPDATE_PREVIEW] Updated preview for current image: {os.path.basename(rendered_path)}")
        else:
            print("[UPDATE_PREVIEW] No rendered image for current selection")
    
    except Exception as e:
        print(f"[UPDATE_PREVIEW] Error: {str(e)}")
        import traceback
        print(f"[UPDATE_PREVIEW] Traceback: {traceback.format_exc()}")

def _get_ocr_config(self) -> dict:
    """Get OCR configuration for the selected provider (same as regular pipeline)"""
    # Resolve provider robustly: prefer current value, then config fallback
    provider = None
    try:
        provider = getattr(self, 'ocr_provider_value', None)
        if not provider:
            provider = self.main_gui.config.get('manga_ocr_provider') or self.main_gui.config.get('ocr_provider')
    except Exception:
        provider = None
    if not provider:
        provider = 'custom-api'
    # Normalize aliases
    if provider in ['azure_document_intelligence', 'azure-document-intel', 'azure_doc_intel']:
        provider = 'azure-document-intelligence'
    print(f"[DEBUG] Building OCR config for provider: {provider}")
    config = {'provider': provider}
    
    if config['provider'] == 'google':
        google_creds = self.main_gui.config.get('google_vision_credentials', '') or \
                      self.main_gui.config.get('google_cloud_credentials', '')
        print(f"[DEBUG] Google credentials path: {google_creds}")
        if google_creds and os.path.exists(google_creds):
            config['google_credentials_path'] = google_creds
            print(f"[DEBUG] Google credentials found and added to config")
        else:
            print(f"[DEBUG] Google credentials not found or missing")
    elif config['provider'] == 'azure':
        # Pull from both Vision and Doc Intelligence keys as fallback
        azure_key = (
            self.main_gui.config.get('azure_vision_key', '')
            or self.main_gui.config.get('azure_document_intelligence_key', '')
        )
        azure_endpoint = (
            self.main_gui.config.get('azure_vision_endpoint', '')
            or self.main_gui.config.get('azure_document_intelligence_endpoint', '')
        )
        print(f"[DEBUG] Azure key exists: {bool(azure_key)}")
        print(f"[DEBUG] Azure endpoint: {azure_endpoint}")
        if azure_key and azure_endpoint:
            config['azure_key'] = azure_key
            config['azure_endpoint'] = azure_endpoint
            print(f"[DEBUG] Azure credentials added to config")
        else:
            print(f"[DEBUG] Azure credentials not complete")
    elif config['provider'] == 'custom-api':
        print(f"[DEBUG] Using custom-api provider (requires API key)")
        # For custom-api provider, we need to ensure the API key is available
        api_key = os.environ.get('API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
        if api_key:
            print(f"[DEBUG] API key available for custom-api OCR")
        else:
            print(f"[DEBUG] WARNING: No API key available for custom-api OCR")
        # Also ensure OCR prompt is set in environment
        if hasattr(self, 'ocr_prompt') and self.ocr_prompt:
            os.environ['OCR_SYSTEM_PROMPT'] = self.ocr_prompt
            print(f"[DEBUG] Set OCR_SYSTEM_PROMPT for custom-api")
    elif config['provider'] == 'azure-document-intelligence':
        # Azure Document Intelligence uses same credentials schema; pull from either bucket
        azure_key = (
            self.main_gui.config.get('azure_document_intelligence_key', '')
            or self.main_gui.config.get('azure_vision_key', '')
        )
        azure_endpoint = (
            self.main_gui.config.get('azure_document_intelligence_endpoint', '')
            or self.main_gui.config.get('azure_vision_endpoint', '')
        )
        print(f"[DEBUG] Azure Document Intelligence key exists: {bool(azure_key)}")
        print(f"[DEBUG] Azure Document Intelligence endpoint: {azure_endpoint}")
        if azure_key and azure_endpoint:
            config['azure_key'] = azure_key
            config['azure_endpoint'] = azure_endpoint
            print(f"[DEBUG] Azure Document Intelligence credentials added to config")
        else:
            print(f"[DEBUG] Azure Document Intelligence credentials not complete")
    else:
        print(f"[DEBUG] Using local OCR provider: {config['provider']}")
    
    print(f"[DEBUG] Final OCR config: {config}")
    return config

def _process_recognize_results(self, results: dict):
    """Process recognition results on main thread (image-aware)."""
    # ===== CANCELLATION CHECK: Discard results if stop was clicked =====
    if _is_translation_cancelled(self):
        print(f"[RECOG_RESULTS] Discarding results - stop was clicked")
        return
    
    try:
        recognized_texts = results['recognized_texts']
        image_path = results.get('image_path') or getattr(self, '_current_image_path', None)
        
        # Persist recognized texts to state for this image
        if hasattr(self, 'image_state_manager') and image_path:
            try:
                print(f"[STATE DEBUG] Saving recognized_texts for {os.path.basename(image_path)}: count={len(recognized_texts)}")
                if recognized_texts:
                    print(f"[STATE DEBUG] First OCR result: {recognized_texts[0]}")
                self.image_state_manager.update_state(image_path, {'recognized_texts': recognized_texts})
                # Immediately flush to disk to ensure OCR state persists across sessions
                self.image_state_manager.flush()
            except Exception as e:
                print(f"[STATE DEBUG] Failed to save recognized_texts: {e}")
        
        # NOTE: We no longer suppress UI updates during batch mode - users want to see
        # rectangles turn blue during recognition for visual feedback
        
        # Only update UI and working memory if this is the current image
        if not hasattr(self, 'image_preview_widget') or image_path != getattr(self.image_preview_widget, 'current_image_path', None):
            print(f"[RECOG_RESULTS] Skipping UI update; not current image: {os.path.basename(image_path) if image_path else 'unknown'}")
            return
        
        # Store recognized texts for translation on the active image
        self._recognized_texts = recognized_texts
        # Track which image these recognitions belong to to avoid cross-image reuse
        try:
            self._recognized_texts_image_path = image_path
        except Exception:
            self._recognized_texts_image_path = None
        
        if recognized_texts:
            self._log(f"🎉 Recognition Results ({len(recognized_texts)} regions with text):", "success")
            for i, text_data in enumerate(recognized_texts):
                bbox = text_data['bbox']
                text = text_data['text']
                self._log(f"  Region {i+1} at ({bbox[0]},{bbox[1]}) [{bbox[2]}x{bbox[3]}]: '{text}'", "info")
            
            # Update UI with recognition tooltips
            _update_rectangles_with_recognition(self, recognized_texts)
            self._log(f"📋 Ready for translation! Click 'Translate' to proceed.", "info")
            
            # PERSIST: Save viewer_rectangles (now blue) to state so they survive panel/session switches
            try:
                _persist_current_image_state(self)
            except Exception:
                pass
        else:
            self._log("⚠️ No text was recognized in any regions", "warning")
    
    except Exception as e:
        self._log(f"❌ Failed to process recognition results: {str(e)}", "error")

def _process_translate_results(self, results: dict):       
    """Process translation results on main thread - USE PIL RENDERING!"""
    # ===== CANCELLATION CHECK: Discard results if stop was clicked =====
    if _is_translation_cancelled(self):
        print(f"[TRANSLATE_RESULTS] Discarding results - stop was clicked")
        return
    
    try:
        translated_texts = results['translated_texts']
        image_path = results.get('image_path')  # This might be cleaned image
        original_image_path = results.get('original_image_path', image_path)  # Original for mapping
        
        # Store translated texts
        self._translated_texts = translated_texts
        
        # CRITICAL: Track which image these translations belong to
        self._translation_data_image_path = original_image_path
        print(f"[TRANSLATE_RESULTS] Translation data now belongs to: {os.path.basename(original_image_path)}")
        
        # Persist translated_texts to state for overlay restoration across sessions
        try:
            if hasattr(self, 'image_state_manager') and original_image_path:
                self.image_state_manager.update_state(original_image_path, {'translated_texts': translated_texts})
                # Immediately flush to disk to ensure translation state persists across sessions
                self.image_state_manager.flush()
        except Exception:
            pass
        
        # Log summary of translations
        if translated_texts:
            self._log(f"🎉 Translation Results ({len(translated_texts)} regions translated):", "success")
            for i, result in enumerate(translated_texts):
                original_text = result['original']['text']
                translation = result['translation']
                bbox = result['bbox']
                region_index = result['original'].get('region_index', i)
                self._log(f"  Region {region_index+1} at ({bbox[0]},{bbox[1]}): '{original_text}' → '{translation}'", "info")
            
            # Store translation data keyed by region_index for accurate mapping to rectangles
            self._translation_data = {}
            rectangles = self.image_preview_widget.viewer.rectangles
            for i, result in enumerate(translated_texts):
                region_index = result['original'].get('region_index', i)
                if region_index < len(rectangles):
                    self._translation_data[region_index] = {
                        'original': result['original']['text'],
                        'translation': result['translation']
                    }
            
            # USE PIL RENDERING (same as manual edit) instead of Qt overlays!
            print(f"\n[TRANSLATE] Using PIL rendering for {len(translated_texts)} translations")
            print(f"[TRANSLATE] Available viewer rectangles: {len(rectangles)}")
            self._log(f"🎨 Rendering translations with PIL pipeline...", "info")
            
            # Build TextRegion objects using bbox from results (image-aware, no dependency on viewer)
            from manga_translator import TextRegion
            regions = []
            
            for i, result in enumerate(translated_texts):
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                
                region = TextRegion(
                    text=result['original']['text'],
                    vertices=vertices,
                    bounding_box=(x, y, w, h),
                    confidence=1.0,
                    region_type=result.get('original', {}).get('region_type', 'text_block')
                )
                try:
                    ob = result.get('original', {})
                    bb = ob.get('bubble_bounds') or [x, y, w, h]
                    region.bubble_bounds = tuple(bb) if isinstance(bb, (list, tuple)) else None
                    if ob.get('bubble_type'):
                        region.bubble_type = ob.get('bubble_type')
                except Exception:
                    pass
                region.translated_text = result['translation']
                regions.append(region)
            
            print(f"\n[TRANSLATE] Final region count: {len(regions)}")
            
            if regions:
                # Use the image_path passed in results (already handles cleaned vs original)
                render_image = image_path
                print(f"[TRANSLATE] render_image from results: {render_image}")
                print(f"[TRANSLATE] Image exists: {os.path.exists(render_image) if render_image else False}")
                
                if render_image and os.path.exists(render_image):
                    # Check if we're using a cleaned image (saved as *_cleaned in isolated folder)
                    is_cleaned = (os.path.basename(render_image).lower().endswith('_cleaned' + os.path.splitext(render_image)[1].lower()) or
                                  (hasattr(self, '_cleaned_image_path') and 
                                   self._cleaned_image_path and 
                                   os.path.normpath(render_image) == os.path.normpath(self._cleaned_image_path)))
                    
                    if is_cleaned:
                        print(f"[TRANSLATE] Using cleaned image: {os.path.basename(render_image)}")
                        self._log(f"🧹 Rendering on cleaned image", "info")
                    else:
                        print(f"[TRANSLATE] No cleaned image available, rendering on current image: {os.path.basename(render_image)}")
                        self._log(f"📝 Rendering on original image (click Clean first to remove original text)", "info")
                    
                    print(f"[TRANSLATE] ✅ About to call _render_with_manga_translator with {len(regions)} regions")
                    print(f"[TRANSLATE] Regions summary:")
                    for i, r in enumerate(regions):
                        print(f"[TRANSLATE]   Region {i}: bbox={r.bounding_box}, text='{r.text[:20]}...', trans='{r.translated_text[:20]}...'")
                    
                    # Compute per-image isolated output path (match single-translate isolation)
                    original_path_for_output = original_image_path
                    try:
                        filename = os.path.basename(original_path_for_output)
                        base_name = os.path.splitext(filename)[0]
                        parent_dir = os.path.dirname(original_path_for_output)
                        
                        # Check for OUTPUT_DIRECTORY override (prefer config over env var)
                        override_dir = None
                        if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                            override_dir = self.main_gui.config.get('output_directory', '')
                        if not override_dir:
                            override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
                        
                        if override_dir:
                            output_dir = os.path.join(override_dir, f"{base_name}_translated")
                        else:
                            output_dir = os.path.join(parent_dir, f"{base_name}_translated")
                        
                        os.makedirs(output_dir, exist_ok=True)
                        target_output_path = os.path.join(output_dir, filename)
                    except Exception:
                        target_output_path = None
                    
                    # Prefer in-memory cleaned image if provided in results
                    image_bgr = results.get('image_bgr') if isinstance(results, dict) else None
                    
                    _render_with_manga_translator(self, 
                        render_image,
                        regions,
                        output_path=target_output_path,
                        image_bgr=image_bgr,
                        original_image_path=original_image_path,
                        switch_tab=False  # Don't auto-switch tabs - let user manually switch
                    )
                    print(f"[TRANSLATE] Returned from _render_with_manga_translator")
                    
                    # CRITICAL: Always remove processing overlay for the translated image
                    # This must happen regardless of which image is currently displayed
                    try:
                        _remove_processing_overlay(self, original_image_path)
                        print(f"[TRANSLATE] Removed processing overlay for {os.path.basename(original_image_path)}")
                    except Exception as e:
                        print(f"[TRANSLATE] Failed to remove processing overlay: {e}")
                    
                    # Refresh image preview to show translated output (only if currently viewing this image)
                    try:
                        if not getattr(self, '_batch_mode_active', False):
                            current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                            if current_image and original_image_path and os.path.normpath(current_image) == os.path.normpath(original_image_path):
                                print(f"[TRANSLATE] Refreshing preview to show translated output")
                                self.image_preview_widget.load_image(original_image_path, preserve_rectangles=True, preserve_text_overlays=True)
                    except Exception as e:
                        print(f"[TRANSLATE] Preview refresh failed: {e}")
                else:
                    print(f"[TRANSLATE] ERROR: No image path available for rendering or image doesn't exist")
                    print(f"[TRANSLATE]   render_image={render_image}")
                    self._log("⚠️ Could not render: no image loaded", "warning")
            else:
                print(f"[TRANSLATE] ❌ No regions to render (regions list is empty after building)")
                self._log("⚠️ No regions to render", "warning")
            
            # If allowed, update on-canvas text overlays for the CURRENT image only
            try:
                current_image = getattr(self.image_preview_widget, 'current_image_path', None)
                # Skip overlay updates during batch or when results are for a different image
                if getattr(self, '_batch_mode_active', False):
                    print(f"[TRANSLATE] Batch active — skipping overlay update for {os.path.basename(original_image_path) if original_image_path else 'unknown'}")
                elif current_image and original_image_path and os.path.normpath(current_image) == os.path.normpath(original_image_path):
                    _add_text_overlay_to_viewer(self, translated_texts)
                else:
                    print(f"[TRANSLATE] Skipping overlay update; not current image: {os.path.basename(original_image_path) if original_image_path else 'unknown'}")
            except Exception as _ov_err:
                print(f"[TRANSLATE] Overlay update skipped/failed: {_ov_err}")
            
            self._log(f"✅ Translation workflow complete!", "success")
        else:
            self._log("⚠️ No translations were generated", "warning")
    
    except Exception as e:
        print(f"[TRANSLATE] ERROR in _process_translate_results: {str(e)}")
        import traceback
        print(traceback.format_exc())
        self._log(f"❌ Failed to process translation results: {str(e)}", "error")


def _render_with_manga_translator_thread_safe(self, base_image_path, regions, output_path, original_image_path):
    """Thread-safe rendering using existing translator instance (no heavy initialization)"""
    try:
        import sys
        import cv2
        import os
        
        sys.__stdout__.write(f"[WORKER] Starting render: base={os.path.basename(base_image_path)}\n")
        sys.__stdout__.write(f"[WORKER] Regions to render: {len(regions)}\n")
        sys.__stdout__.write(f"[WORKER] Output path: {output_path}\n")
        sys.__stdout__.flush()
        
        # Use existing translator instance from manga_integration (already has loaded models)
        translator = self.manga_integration.translator
        if not translator:
            sys.__stdout__.write("[WORKER] ERROR: No translator instance available\n")
            sys.__stdout__.flush()
            return False
        
        # Load the base image for rendering
        base_image_array = cv2.imread(base_image_path)
        if base_image_array is None:
            raise ValueError(f"Failed to load base image: {base_image_path}")
        
        # Ensure all regions have translated_text set
        filtered_regions = []
        for region in regions:
            if not hasattr(region, 'translated_text') or not region.translated_text:
                # Fallback to original text if translation missing
                region.translated_text = region.text
            filtered_regions.append(region)
        
        sys.__stdout__.write(f"[WORKER] Rendering {len(filtered_regions)} regions with translations\n")
        sys.__stdout__.flush()
        
        # Render using the existing translator's render method (thread-safe for rendering)
        # The translator already has loaded models in the shared pool
        rendered_image_array = translator.render_translated_text(base_image_array, filtered_regions)
        
        # Save the rendered image
        if output_path:
            result_success = cv2.imwrite(output_path, rendered_image_array)
            result_path = output_path if result_success else None
        else:
            # Generate output path if not provided
            base_dir = os.path.dirname(base_image_path)
            base_name = os.path.splitext(os.path.basename(base_image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_translated.png")
            result_success = cv2.imwrite(output_path, rendered_image_array)
            result_path = output_path if result_success else None
        
        # Clean up only local resources (images)
        # DO NOT clean up translator or models - they're shared and reused!
        del base_image_array
        del rendered_image_array
        
        sys.__stdout__.write(f"[WORKER] Render completed: {result_path}\n")
        sys.__stdout__.flush()
        return result_path is not None
        
    except Exception as e:
        sys.__stdout__.write(f"[WORKER] Render failed: {e}\n")
        sys.__stdout__.flush()
        import traceback
        traceback.print_exc()
        return False
